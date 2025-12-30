import os
import glob
import shutil
import random
import math
from tqdm import tqdm

import os
import random
import shutil

from .config import Config
from .doc2txt import doc2txt 


def extract_txt_from_doc(data_dir):
    doc2txt(data_dir)
    print(f"提取txt文件完成，文件位于doc源文件同位置下，同名txt文件")




def create_symlink_split(source_root, target_root, train_ratio=0.5, seed=42):
    """
    Creates symlinks for a train/test split from source_root to target_root.
    target_root should contain 'train' and 'test' directories.
    """
    random.seed(seed)
    
    # 确保目标目录存在 train 和 test
    train_root = os.path.join(target_root, "train")
    test_root = os.path.join(target_root, "test")
    
    if not os.path.exists(train_root):
        print(f"创建目录: {train_root}")
        os.makedirs(train_root)
    if not os.path.exists(test_root):
        print(f"创建目录: {test_root}")
        os.makedirs(test_root)

    # 获取源目录下的所有主类别文件夹
    categories = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    
    print(f"找到以下类别: {categories}")
    
    total_train = 0
    total_test = 0

    for category in categories:
        src_category_path = os.path.join(source_root, category)
        
        # 获取该类别下的样本文件夹
        samples = [s for s in os.listdir(src_category_path) if os.path.isdir(os.path.join(src_category_path, s))]
        
        # 随机打乱
        random.shuffle(samples)
        
        # 计算分割点
        current_train_ratio = train_ratio
        if len(samples) < 40:
            print(f"类别 {category} 样本数 ({len(samples)}) 少于 40，强制设置 train_ratio 为 0.8")
            current_train_ratio = 0.8

        split_idx = int(len(samples) * current_train_ratio)
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        print(f"\n处理类别: {category}")
        print(f"  总样本数: {len(samples)}")
        print(f"  训练集数量: {len(train_samples)}")
        print(f"  测试集数量: {len(test_samples)}")
        
        # 创建软链接的辅助函数
        def make_links(sample_list, split_root):
            count = 0
            dst_category_path = os.path.join(split_root, category)
            os.makedirs(dst_category_path, exist_ok=True)
            
            for sample in sample_list:
                src_path = os.path.join(src_category_path, sample)
                dst_path = os.path.join(dst_category_path, sample)
                
                # 如果目标已存在（可能是之前的链接），先删除
                if os.path.exists(dst_path) or os.path.islink(dst_path):
                    try:
                        os.unlink(dst_path)
                    except IsADirectoryError:
                        # 如果确实是一个目录而不是软链接（不应该发生，但为了安全）
                        print(f"⚠️ 警告: 目标路径是一个实际目录，跳过: {dst_path}")
                        continue
                
                try:
                    os.symlink(src_path, dst_path)
                    count += 1
                except Exception as e:
                    print(f"❌ 创建软链接失败 {sample}: {e}")
            return count

        # 创建训练集链接
        t_count = make_links(train_samples, train_root)
        total_train += t_count
        
        # 创建测试集链接
        v_count = make_links(test_samples, test_root)
        total_test += v_count

    print(f"\n✅ 数据集划分完成！")
    print(f"总训练集样本链接数: {total_train}")
    print(f"总测试集样本链接数: {total_test}")




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
    source_dir = "/media/codingma/LLM/data-1005"
    target_dir = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets"
    
    create_symlink_split(source_dir, target_dir)

