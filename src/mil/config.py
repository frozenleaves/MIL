import os
import torch

class Config:
    # ================= 路径配置 =================
    # 数据集根目录 (你的目录结构)
    RAW_DATA_ROOT = "/media/codingma/LLM/lcx/data-1005"
    DATA_INDEX_PATH = "/media/codingma/LLM/lcx/data-1005/index.csv"
    OVERWRITE_SWI_FEATURES = True

    # 模型路径
    VIRCHOW2_MODEL_ID = "local-dir:/media/codingma/LLM/lcx/Virchow2" # HuggingFace ID
    QWEN_MODEL_PATH = "/media/codingma/LLM/lcx/Qwen3-Embedding-0.6B" 
    
    # ================= 数据参数 =================
    # 类别映射 (根据你的实际文件夹名修改)
    # 假设你的6个文件夹名如下，这里定义 Class Name -> Label ID 的映射
    CLASS_MAP = {
        "已整理-OLK": 0,
        "已整理-OLP": 1,
        "已整理-OSCC": 2,
        "已整理-乳头状瘤": 3,
        "已整理-粘液囊肿": 4,
        "已整理-纤维增生": 5
    }
    
    # WSI 处理参数
    PATCH_SIZE = 224       # Virchow2 固定尺寸
    TILE_LEVEL = 0         # 40x / 20x 最高倍率
    BATCH_SIZE_WSI = 256    # 特征提取时的 Batch Size
    NUM_WORKERS = 8        # DataLoader workers
    BG_THRESHOLD = 220     # 去除背景的阈值
    SAVE_COORDS = False

    # ================= 训练参数 =================
    NUM_CLASSES = 6

    # 权重保存目录
    CHECKPOINT_DIR = "/media/codingma/LLM/lcx/checkpoints"
    
    # [模型容量配置]
    FUSION_DIM = 768       # 512 -> 768 (增大维度)
    FUSION_LAYERS = 6      # 2 -> 6 (增加深度)
    FUSION_HEADS = 12      # 768 / 12 = 64 (必须能整除 FUSION_DIM)
    FUSION_DROPOUT = 0.1

    WSI_INPUT_DIM = 2560
    MAX_TEXT_LEN = 512
    FREEZE_TEXT_MODEL = True
    
    BATCH_SIZE = 4
    EPOCHS = 20
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.0001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True
