import os
import glob
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, roc_curve, auc

from mil.config import Config
from mil.inference import InferencePipeline

plt.rcParams['font.family'] = 'AR PL UKai CN'

def evaluate_test_set(test_root, ckpt_path=None, use_txt=True, use_img=True, use_svs=True,  prob_threshold=0.5, enable_fallback=True, roc_save_path=None):
    if not os.path.exists(test_root):
        print(f"Error: Test dataset not found at {test_root}")
        return

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR , "best_val.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Best checkpoint not found at {checkpoint_path}, trying last.pth")
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "last.pth")
    if ckpt_path:
        checkpoint_path = ckpt_path
    if not os.path.exists(checkpoint_path):
        print("Error: No checkpoint found.")
        return

    # 2. 初始化模型
    print("Initializing Inference Pipeline...")
    pipeline = InferencePipeline(checkpoint_path)
    
    # 3. 收集测试样本
    samples = []
    
    # [修改] 定义旧 ID 到新 Multi-hot 索引的映射
    # 这与 Dataset 中的逻辑保持一致
    label_mapping_rules = {
        0: [0],    # OLK
        1: [1],    # OLP
        2: [2],    # OSCC
        3: [3],    # OSF
        4: [0, 3], # OSF+OLK -> OLK + OSF
        5: [4],    # 乳头状瘤
        6: [5],    # 粘液囊肿
        7: [6]     # 纤维增生
    }
    
    # 原始物理文件夹映射 (用于读取)
    raw_class_map = Config.RAW_CLASS_MAP
    
    print(f"Scanning test dataset at {test_root}...")
    
    for class_name in os.listdir(test_root):
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        if class_name not in raw_class_map:
            print(f"Skipping unknown class folder: {class_name}")
            continue
            
        raw_label_id = raw_class_map[class_name]
        
        # [修改] 构建 Multi-hot Label
        true_label_indices = label_mapping_rules.get(raw_label_id, [])
        # 转化为 0/1 向量
        current_label_vec = np.zeros(Config.NUM_CLASSES, dtype=int)
        current_label_vec[true_label_indices] = 1
        
        for sample_name in os.listdir(class_dir):
            sample_dir = os.path.join(class_dir, sample_name)
            if not os.path.isdir(sample_dir):
                continue
                
            # 查找输入文件
            txt_files = glob.glob(os.path.join(sample_dir, "*.txt"))
            text_content = ""
            if use_txt and txt_files:
                with open(txt_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
            
            img_paths = []
            if use_img:
                for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg']:
                    img_paths.extend(glob.glob(os.path.join(sample_dir, ext)))
                    
            svs_paths = []
            if use_svs and os.path.exists(os.path.join(sample_dir, "processed")):
                svs_paths = glob.glob(os.path.join(sample_dir, "processed", "*.pt"))
                
            samples.append({
                "path": sample_dir,
                "class_name": class_name,
                "label_vec": current_label_vec, 
                "text": text_content,
                "images": img_paths,
                "svs": svs_paths
            })
            
    print(f"Found {len(samples)} test samples.")
    if len(samples) == 0: return

    # 4. 执行推理
    y_true = []
    y_pred = [] # 0/1 向量
    y_probs = [] # 概率向量
    
    print("\nStarting Inference Loop...")
    
    results_log = []
    
    for i, sample in enumerate(tqdm(samples)):
        try:
            probs = pipeline.predict_proba(
                sample['text'], 
                sample['images'], 
                sample['svs']
            )
            
            # 阈值判断 (默认0.5)
            # 也可以改为 0.0 (如果训练时用的是 logits > 0)，但这里 pipeline 输出已经是 sigmoid 后的 probs
            # 如果设置了阈值，存在任何标签都没有被选中的情况，此时可以添加一个保底机制，至少选择一个概率值最高的标签
            pred_vec = (probs > prob_threshold).astype(int)
            
            # [新增] 保底机制 (Fallback): 如果什么都没选，就选概率最大的
            is_fallback = False
            if enable_fallback and pred_vec.sum() == 0:
                max_idx = np.argmax(probs)
                pred_vec[max_idx] = 1
                is_fallback = True
            
            y_true.append(sample['label_vec'])
            y_pred.append(pred_vec)
            y_probs.append(probs)
            
            # 记录结果
            pred_names = [Config.TARGET_CLASS_NAMES[idx] for idx, val in enumerate(pred_vec) if val == 1]
            true_names = [Config.TARGET_CLASS_NAMES[idx] for idx, val in enumerate(sample['label_vec']) if val == 1]
            
            is_correct = np.array_equal(sample['label_vec'], pred_vec)
            
            # 实时打印
            status_mark = "✅" if is_correct else "❌"
            if is_fallback:
                status_mark += " (Fallback)"
                
            tqdm.write(f"{status_mark} {os.path.basename(sample['path'])}")
            tqdm.write(f"   True: {true_names}")
            tqdm.write(f"   Pred: {pred_names}")
            tqdm.write("-" * 30)
            
            results_log.append({
                "sample": os.path.basename(sample['path']),
                "true": true_names,
                "pred": pred_names,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"Error inference on {sample['path']}: {e}")
            # import traceback
            # traceback.print_exc()
            
    # 5. 计算指标
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Exact Match Accuracy
    acc = accuracy_score(y_true, y_pred)
    correct_count = np.sum(np.all(y_true == y_pred, axis=1))
    total_count = len(y_true)
    
    print("\n" + "="*50)
    print(f"Test Set Exact Match Accuracy: {acc:.4f} ({correct_count}/{total_count})")
    print("="*50)
    
    print("\nClassification Report:")
    try:
        print(classification_report(
            y_true, 
            y_pred, 
            target_names=Config.TARGET_CLASS_NAMES, 
            digits=4,
            zero_division=0
        ))
    except Exception as e:
        print(f"Error generating report: {e}")
    
    # [新增] 混淆矩阵计算与绘制
    print("\nMultilabel Confusion Matrix:")
    try:
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        for i, class_name in enumerate(Config.TARGET_CLASS_NAMES):
            print(f"\nClass: {class_name}")
            print(mcm[i])
        
        # 绘制混淆矩阵热力图 (7个类别的 2x2 矩阵)
        # 布局：2行4列 (最后一个位置留空)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
        axes = axes.flatten()
        
        for i, class_name in enumerate(Config.TARGET_CLASS_NAMES):
            ax = axes[i]
            cm = mcm[i]
            
            # 绘制热力图 (使用 imshow)
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            
            # 添加数值标签
            thresh = cm.max() / 2.
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, format(cm[r, c], 'd'),
                            ha="center", va="center",
                            color="white" if cm[r, c] > thresh else "black",
                            fontsize=12)
            
            # 设置坐标轴标签
            ax.set_title(f'Confusion Matrix - {class_name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['False', 'True'])
            ax.set_yticklabels(['False', 'True'])
            
        # 隐藏多余的子图
        for j in range(len(Config.TARGET_CLASS_NAMES), len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        
        # 构建保存路径
        ft = 'txt_' if use_txt else ''
        fi = 'img_' if use_img else ''
        fs = 'svs_' if use_svs else ''
        cm_save_path = f"results/confusion_matrix/txt_only/{ft}{fi}{fs}{time.strftime('%Y%m%d_%H%M%S')}_multilabel_cm.png"
        
        os.makedirs(os.path.dirname(cm_save_path), exist_ok=True)
        # plt.savefig(cm_save_path)
        print(f"\nConfusion Matrix Heatmap saved to {cm_save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

    # 6. 绘制 ROC 曲线
    n_classes = y_true.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        label_name = Config.TARGET_CLASS_NAMES[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.4f})'.format(label_name, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-label ROC Curve')
    plt.legend(loc="lower right")
    ft = 'txt_' if use_txt else ''
    fi = 'img_' if use_img else ''
    fs = 'svs_' if use_svs else ''
    
    save_path = f"results/roc_plot/txt_only/{ft}{fi}{fs}{time.strftime('%Y%m%d_%H%M%S')}_multilabel_roc.png"
    if roc_save_path: save_path = roc_save_path
    
    # Ensure results dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # plt.savefig(save_path)
    print(f"\nROC curve saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    test_root = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets/test"
    ckpt_path = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints-txt_only/best_val.pth"
    ckpt_path = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_multi_label/best_val.pth"
    evaluate_test_set(test_root, ckpt_path=ckpt_path, use_txt=True, use_img=False , use_svs=True, enable_fallback=True)
