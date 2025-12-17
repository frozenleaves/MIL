import os
import glob
import pandas as pd
import torch
import openslide
import numpy as np
import timm
from tqdm import tqdm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from .config import Config
# 导入修改后的函数和模型加载器
from .wsi_processor import extract_wsi_features, get_virchow2_backbone
from .wsi_processor_fast import extract_wsi_features as extract_wsi_features_fast


def main():
    data_records = []

    # 0. 全局加载模型 (只加载一次)
    print("正在初始化 Virchow2 模型...")
    try:
        model, transform = get_virchow2_backbone(Config.VIRCHOW2_MODEL_ID, Config.DEVICE)
        print("模型加载完成。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 1. 遍历 6 个类别文件夹
    if not os.path.exists(Config.RAW_DATA_ROOT):
        print(f"数据根目录不存在: {Config.RAW_DATA_ROOT}")
        return

    class_dirs = [d for d in os.listdir(Config.RAW_DATA_ROOT) if os.path.isdir(os.path.join(Config.RAW_DATA_ROOT, d))]

    print(f"Found classes: {class_dirs}")

    for class_name in class_dirs:
        if class_name not in Config.CLASS_MAP:
            print(f"Skipping unknown folder: {class_name}")
            continue

        label = Config.CLASS_MAP[class_name]
        class_path = os.path.join(Config.RAW_DATA_ROOT, class_name)

        # 2. 遍历该类别下的 N 个样本文件夹
        sample_names = os.listdir(class_path)

        for sample_name in tqdm(sample_names, desc=f"Processing {class_name}"):
            sample_dir = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_dir): continue

            # --- A. 查找 TXT 文件 ---
            txt_files = glob.glob(os.path.join(sample_dir, "*.txt"))
            txt_path = txt_files[0] if txt_files else ""

            # --- B. 查找 JPG 图片 ---
            jpg_files = glob.glob(os.path.join(sample_dir, "*.JPG"))
            jpg_paths_str = ";".join(jpg_files)

            # --- C. 处理 SVS 文件 (processed 目录下) ---
            svs_dir = os.path.join(sample_dir, "processed")
            wsi_feat_paths = []

            if os.path.exists(svs_dir):
                svs_files = glob.glob(os.path.join(svs_dir, "*.svs"))

                for svs_file in svs_files:
                    svs_filename = os.path.basename(svs_file)
                    svs_dir_path = os.path.dirname(svs_file)
                    save_name = os.path.splitext(svs_filename)[0] + ".pt"
                    save_path = os.path.join(svs_dir_path, save_name)

                    # 检查是否已存在，避免重复跑 (可选)
                    if os.path.exists(save_path):
                        print(f" existing: {save_path}")
                    if not Config.OVERWRITE_SWI_FEATURES and os.path.exists(save_path):
                        wsi_feat_paths.append(save_path)
                        continue

                    # 传入预加载的模型和transform
                    if Config.USE_FAST_VERSION:
                        success = extract_wsi_features_fast(svs_file, save_path, model, transform)
                    else:
                        success = extract_wsi_features(svs_file, save_path, model, transform)
                    if success:
                        wsi_feat_paths.append(save_path)

            wsi_paths_str = ";".join(wsi_feat_paths)

            # --- D. 记录到列表 ---
            # 只有当至少有 txt 或者 jpg 或者 wsi 时才记录，防止空目录
            if txt_path or jpg_paths_str or wsi_paths_str:
                data_records.append({
                    "sample_id": sample_name,
                    "txt_path": txt_path,
                    "img_paths": jpg_paths_str,
                    "wsi_paths": wsi_paths_str,
                    "label": label
                })

    # 3. 生成 CSV
    if data_records:
        df = pd.DataFrame(data_records)
        df.to_csv(Config.DATA_INDEX_PATH, index=False)
        print(f"Done! Index saved to {Config.DATA_INDEX_PATH}. Total samples: {len(df)}")
    else:
        print("No records found.")


if __name__ == "__main__":
    main()
