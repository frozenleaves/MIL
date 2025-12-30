import os
import glob
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from mil.config import Config
from mil.inference import InferencePipeline

plt.rcParams['font.family'] = 'AR PL UKai CN'

def evaluate_test_set(test_root, use_txt=True, use_img=True, use_svs=True, roc_save_path=None):
    # 1. 确定路径
    # test_root = Config.RAW_DATA_ROOT + "-test-dataset-0.1"
    if not os.path.exists(test_root):
        print(f"Error: Test dataset not found at {test_root}")
        return

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR , "best_val.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Warning: Best checkpoint not found at {checkpoint_path}, trying last.pth")
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "last.pth")
    
    if not os.path.exists(checkpoint_path):
        print("Error: No checkpoint found.")
        return

    # 2. 初始化模型
    print("Initializing Inference Pipeline...")
    pipeline = InferencePipeline(checkpoint_path)
    
    # 3. 收集测试样本
    samples = []
    
    # 获取类别映射 (Name -> ID)
    class_map = Config.CLASS_MAP
    # 反向映射 (ID -> Name) 用于打印
    idx_to_class = {v: k for k, v in class_map.items()}
    
    print(f"Scanning test dataset at {test_root}...")
    
    for class_name in os.listdir(test_root):
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        if class_name not in class_map:
            print(f"Skipping unknown class folder: {class_name}")
            continue
            
        label = class_map[class_name]
        
        for sample_name in os.listdir(class_dir):
            sample_dir = os.path.join(class_dir, sample_name)
            if not os.path.isdir(sample_dir):
                continue
                
            # 查找输入文件
            # 1. 文本 (txt)
            txt_files = glob.glob(os.path.join(sample_dir, "*.txt"))
            text_content = ""
            if use_txt:
                if txt_files:
                    with open(txt_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
            
            # 2. 图片 (jpg, png, etc.)
            img_extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg']
            img_paths = []
            if use_img:
                for ext in img_extensions:
                    img_paths.extend(glob.glob(os.path.join(sample_dir, ext)))
                    
            # 3. SVS (processed/*.svs)
            svs_dir = os.path.join(sample_dir, "processed")
            svs_paths = []
            if use_svs:
                if os.path.exists(svs_dir):
                    svs_paths = glob.glob(os.path.join(svs_dir, "*.pt"))
                
            samples.append({
                "path": sample_dir,
                "class_name": class_name,
                "label": label,
                "text": text_content,
                "images": img_paths,
                "svs": svs_paths
            })
            
    print(f"Found {len(samples)} test samples.")
    
    if len(samples) == 0:
        return

    # 4. 执行推理
    y_true = []
    y_pred = []
    y_probs = []
    
    print("\nStarting Inference Loop...")
    
    # 记录每个样本的结果
    results_log = []
    
    for i, sample in enumerate(tqdm(samples)):
        try:
            pred_idx, probs = pipeline.predict(
                sample['text'], 
                sample['images'], 
                sample['svs']
            )
            
            y_true.append(sample['label'])
            y_pred.append(pred_idx)
            y_probs.append(probs)
            
            is_correct = (pred_idx == sample['label'])
            result_str = "Correct" if is_correct else "Wrong"
            pred_name = idx_to_class.get(pred_idx, "Unknown")
            
            # 简单Log
            # print(f"[{i+1}/{len(samples)}] {sample['class_name']}/{os.path.basename(sample['path'])} -> Pred: {pred_name} ({result_str})")
            
            results_log.append({
                "sample": os.path.basename(sample['path']),
                "true": sample['class_name'],
                "pred": pred_name,
                "correct": is_correct,
                "probs": probs
            })
            
        except Exception as e:
            print(f"Error inference on {sample['path']}: {e}")
            
    # 5. 计算指标
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print(f"Test Set Accuracy: {acc:.4f} ({sum(y_true[i] == y_pred[i] for i in range(len(y_true)))}/{len(y_true)})")
    print("="*50)
    
    print("\nClassification Report:")
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    # 确保 target_names 顺序对应 0, 1, 2...
    sorted_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    print(classification_report(y_true, y_pred, target_names=sorted_names, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 6. 绘制 ROC 曲线
    # 需要将标签二值化
    y_true_bin = label_binarize(y_true, classes=range(len(class_map)))
    n_classes = y_true_bin.shape[1]
    y_probs = np.array(y_probs)

    # 计算每一类的 ROC 曲线和 AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均 (Micro-average) ROC 曲线和 AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 绘制所有 ROC 曲线
    plt.figure(figsize=(10, 8))
    
    # 绘制微平均 ROC 曲线
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle='-', linewidth=4)

    # 绘制每一类的 ROC 曲线
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])
    for i, color in zip(range(n_classes), colors):
        class_name = idx_to_class.get(i, f"Class {i}")
        # 如果类别名称包含中文，Matplotlib 可能需要字体设置，这里暂时保持原样，或可尝试使用英文
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.4f})'
                       ''.format(class_name, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for Multi-class')
    plt.legend(loc="lower right")
    
    # 保存图片
    save_path = f"{time.strftime('%Y%m%d_%H%M%S')}_roc_curve_test_txt_{use_txt}_img_{use_img}_svs_{use_svs}.png"
    if roc_save_path is not None:
        save_path = roc_save_path
    plt.savefig(save_path)
    print(f"\nROC curve saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    test_root = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets/test"
    evaluate_test_set(test_root, use_txt=True, use_img=True, use_svs=True)

