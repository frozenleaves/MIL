import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from config import Config
from dataset import MultimodalDataset, collate_fn
from model import UnifiedMultimodalModel

def train():
    # 1. 准备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 确保保存目录存在
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # 2. 数据
    print("Initializing Dataset...")
    train_dataset = MultimodalDataset(Config.DATA_INDEX_PATH, mode='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
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
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度 Scaler
    scaler = torch.amp.GradScaler(device_type=device.type, enabled=Config.USE_AMP)

    best_acc = 0.0
    required_acc_least = 0.8

    # 5. 循环
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for batch in loop:
            # 搬运数据
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            normal_imgs = batch['normal_imgs'].to(device)
            wsi_feat = batch['wsi_feat'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # AMP Forward
            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                loss = criterion(logits, labels)

            # AMP Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

        # Epoch 结束统计
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1} Result - Loss: {train_loss/len(train_loader):.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
        
        # Save checkpoint (Best & Last)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'f1': f1
        }
    
        
        # 保存 Best
        if acc > max(best_acc, required_acc_least):
            best_acc = acc
            best_path = os.path.join(Config.CHECKPOINT_DIR, "best.pth")
            torch.save(state, best_path)
            print(f"🌟 New Best Model (Acc: {best_acc:.4f}) Saved to {best_path}")
    
    # 保存 Last
    last_path = os.path.join(Config.CHECKPOINT_DIR, "last.pth")
    torch.save(state, last_path)
    print(f"💾 Last Model Saved to {last_path}")

if __name__ == "__main__":
    train()