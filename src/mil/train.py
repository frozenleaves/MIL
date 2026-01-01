import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .config import Config
from .dataset import MultimodalDataset, collate_fn
from .model import UnifiedMultimodalModel

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    多标签分类的交叉熵 (ZLPR Loss)
    Reference: https://spaces.ac.cn/archives/7359
    y_true: multi-hot vector (0 or 1)
    y_pred: logits (before sigmoid/softmax)
    """
    # 调整 y_pred，使得正例 > 0，负例 < 0
    y_pred = (1 - 2 * y_true) * y_pred
    
    # 构造 log(1 + sum(e^neg)) + log(1 + sum(e^pos))
    # 使用 1e12 这种大数来 mask 掉不需要的部分
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    
    zeros = torch.zeros_like(y_pred[..., :1])
    
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    
    return neg_loss + pos_loss

def train():
    # 1. 准备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 确保保存目录存在
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # 2. 数据
    print("Initializing Dataset...")
    
    # [修改] 读取全量数据并划分
    full_df = pd.read_csv(Config.DATA_INDEX_PATH)
    # 打乱数据
    full_df = full_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    
    split_idx = int(len(full_df) * Config.TRAIN_VAL_SPLIT)
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    val_df = full_df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Data Split: Train {len(train_df)} | Val {len(val_df)}")
    
    # 训练集：开启 expand_factor
    train_dataset = MultimodalDataset(
        train_df, 
        mode='train',
        expand_factor=Config.DATA_EXPAND_FACTOR
    )
    # 验证集：不扩充
    val_dataset = MultimodalDataset(
        val_df, 
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 3. 模型
    print("Initializing Model (This may take time loading Qwen)...")
    model = UnifiedMultimodalModel().to(device)

    # 4. 优化器 & Loss
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    # [修改] 使用自定义 Multilabel Loss，不需要实例化标准 Loss
    # criterion = nn.CrossEntropyLoss()
    
    # 混合精度 Scaler
    scaler = torch.amp.GradScaler(device=device.type, enabled=Config.USE_AMP)

    best_val_acc = 0.0 # [修改] 关注验证集准确率
    required_acc_least = 0.6 # 稍微降低门槛，因为验证集更难
    
    
    total_steps_per_epoch = len(train_loader)

    # 5. 循环
    for epoch in range(Config.EPOCHS):
        # ================= Training =================
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        global_step = 0

        optimizer.zero_grad() # 梯度清零
        
        # 移除 tqdm，使用普通循环并打印日志
        print(f"\n***** Epoch {epoch+1}/{Config.EPOCHS} Start *****")
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, batch in pbar:
            # 搬运数据
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            normal_imgs = batch['normal_imgs'].to(device)
            wsi_feat = batch['wsi_feat'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)
            labels = batch['labels'].to(device)

            # AMP Forward
            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                # [修改] 使用 Multi-label Loss
                loss = torch.mean(multilabel_categorical_crossentropy(logits, labels))
                loss = loss / Config.GRAD_ACCUM_STEPS

            # AMP Backward
            scaler.scale(loss).backward()
            
            grad_norm = 0.0
            
            # 梯度累积更新
            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                # 计算 Grad Norm (在 unscale 之前或之后都可以，通常在 unscale 后 clip_grad_norm 时计算)
                # 为了打印真实的 grad norm，我们先 unscale
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 也可以设为很大来只观测
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            # 统计
            current_loss = loss.item() * Config.GRAD_ACCUM_STEPS
            train_loss += current_loss
            
            # [修改] 多标签预测: logits > 0 为正类
            preds = (logits > 0).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # 打印日志 (Step, Loss, Grad Norm)
            # {'loss': 0.1234, 'grad_norm': 0.5678, 'learning_rate': 1e-5, 'epoch': 1.01, 'step': 100}
            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                 log_msg = f"{{'loss': {current_loss:.4f}, 'grad_norm': {grad_norm:.4f}, 'epoch': {epoch + (i + 1) / total_steps_per_epoch:.2f}, 'step': {global_step}}}"
                 tqdm.write(log_msg)
                 pbar.set_postfix({'loss': f"{current_loss:.4f}"})

        # [新增] Epoch 结束，清理显存和内存
        try:
            del logits, loss, preds, input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask, labels
        except NameError:
            pass # 可能 loop 一次都没进
        torch.cuda.empty_cache()
        gc.collect()

        # Train Metrics
        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        # [新增] 释放列表以节省内存
        del train_preds, train_labels
        gc.collect()
        
        # ================= Validation =================
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        print(f"\nRunning Validation...")
        
        # 验证阶段不需要梯度
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                normal_imgs = batch['normal_imgs'].to(device)
                wsi_feat = batch['wsi_feat'].to(device)
                wsi_mask = batch['wsi_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                    logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                    # [修改] Val Loss
                    loss = torch.mean(multilabel_categorical_crossentropy(logits, labels))
                
                val_loss += loss.item()
                # [修改] Val Preds
                preds = (logits > 0).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Val Metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        avg_val_loss = val_loss / len(val_loader)
        
        # [新增] 打印详细的 Classification Report
        print("\n" + "="*30 + " Validation Report " + "="*30)
        try:
            report = classification_report(
                val_labels, 
                val_preds, 
                target_names=Config.TARGET_CLASS_NAMES,
                zero_division=0,
                digits=4
            )
            print(report)
        except Exception as e:
            print(f"Error generating classification report: {e}")
        print("="*80)
        
        # [新增] 清理 Val 临时变量
        del val_preds, val_labels
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Results Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save checkpoint
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1
        }
    
        # 保存 Best (根据 Val Acc)
        if val_acc > max(best_val_acc, required_acc_least):
            best_val_acc = val_acc
            best_path = os.path.join(Config.CHECKPOINT_DIR, "best_val.pth")
            torch.save(state, best_path)
            print(f"🌟 New Best Model (Val Acc: {best_val_acc:.4f}) Saved to {best_path}")
    
        # 保存 Last
        last_path = os.path.join(Config.CHECKPOINT_DIR, "last.pth")
        torch.save(state, last_path)
        print(f"💾 Last Model Saved")
        print("-" * 50)

if __name__ == "__main__":
    train()
