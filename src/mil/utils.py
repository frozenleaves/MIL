import os
import glob
import shutil
import random
import math
from tqdm import tqdm
try:
    from .config import Config
except ImportError:
    from .config import Config

def split_dataset(source_root, target_root, ratio=0.1, seed=42):
    """
    从源数据集中抽取一定比例的数据作为测试集，移动到目标目录。
    保持原有目录结构 (Class/Sample/...)。
    
    Args:
        source_root (str): 原始数据集根目录
        target_root (str): 测试集目标根目录
        ratio (float): 测试集比例 (默认 0.1)
        seed (int): 随机种子
    """
    if not os.path.exists(source_root):
        print(f"Error: Source root {source_root} does not exist.")
        return

    if seed is not None:
        random.seed(seed)

    print(f"Starting dataset split from {source_root} to {target_root} with ratio {ratio}")

    # 获取所有类别文件夹
    class_dirs = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    
    # 过滤掉非类别文件夹（比如已整理-xxx 之外的）
    # 这里假设所有目录都是类别，或者参考 config.CLASS_MAP
    # 简单起见，处理所有子文件夹
    
    for class_name in class_dirs:
        class_src_path = os.path.join(source_root, class_name)
        class_dst_path = os.path.join(target_root, class_name)
        
        if not os.path.exists(class_dst_path):
            os.makedirs(class_dst_path)

        # 获取该类别下的样本文件夹
        samples = [d for d in os.listdir(class_src_path) if os.path.isdir(os.path.join(class_src_path, d))]
        
        total_samples = len(samples)
        if total_samples == 0:
            print(f"Warning: No samples found in {class_name}")
            continue
            
        # 计算抽取数量
        num_test = int(total_samples * ratio)
        
        # 保证每个类别至少有一个数据（如果总数允许）
        if num_test == 0 and total_samples > 0:
            num_test = 1
            
        # 随机打乱
        random.shuffle(samples)
        
        # 选取测试集样本
        test_samples = samples[:num_test]
        
        print(f"Processing Class: {class_name}")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Moving {len(test_samples)} samples to test set...")

        for sample_name in test_samples:
            src_sample_path = os.path.join(class_src_path, sample_name)
            dst_sample_path = os.path.join(class_dst_path, sample_name)

            print(f"Moving {src_sample_path} to {dst_sample_path}")
            
            # 移动文件夹
            shutil.move(src_sample_path, dst_sample_path)
            
    print("Dataset split completed.")

def check_processed_files():
    """
    检查已处理的 .pt 文件大小 (原有功能)
    """
    if not os.path.exists(Config.RAW_DATA_ROOT):
        print(f"Warning: Config.RAW_DATA_ROOT {Config.RAW_DATA_ROOT} does not exist.")
        return

    class_dirs = [d for d in os.listdir(Config.RAW_DATA_ROOT) if os.path.isdir(os.path.join(Config.RAW_DATA_ROOT, d))]
    
    for class_name in class_dirs:
        class_path = os.path.join(Config.RAW_DATA_ROOT, class_name)

        # 2. 遍历该类别下的 N 个样本文件夹
        sample_names = os.listdir(class_path)

        for sample_name in tqdm(sample_names, desc=f"Checking {class_name}"):
            sample_dir = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_dir): continue

            # --- C. 处理 SVS 文件 (processed 目录下) ---
            svs_dir = os.path.join(sample_dir, "processed")
            
            if os.path.exists(svs_dir):
                svs_files = glob.glob(os.path.join(svs_dir, "*.svs"))

                for svs_file in svs_files:
                    svs_filename = os.path.basename(svs_file)
                    svs_dir_path = os.path.dirname(svs_file)
                    save_name = os.path.splitext(svs_filename)[0] + ".pt"
                    save_path = os.path.join(svs_dir_path, save_name)

                    # 检查是否已存在
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path)
                        if file_size < 1024:
                            size_str = f"{file_size} B"
                        elif file_size < 1024**2:
                            size_str = f"{file_size / 1024:.2f} KB"
                        else:
                            size_str = f"{file_size / (1024**2):.2f} MB"
                        # print(f" existing: {save_path} (size: {size_str})")

if __name__ == "__main__":
    # 1. 检查文件 (原有逻辑)
    # check_processed_files()
    
    # 2. 数据集划分示例 (根据需要取消注释运行)
    # source_dir = Config.RAW_DATA_ROOT
    # target_dir = source_dir + "-test-dataset-0.1" # 例如: /media/codingma/LLM/data-1005_test_set
    # split_dataset(source_dir, target_dir, ratio=0.1)

    pass
    
