import os
import torch

class Config:
    # ================= 路径配置 =================
    # 数据集根目录 
    RAW_DATA_ROOT = "/mnt/g/myq/data-1005"
    DATA_INDEX_PATH = "/mnt/g/myq/data-1005/index.csv"
    OVERWRITE_SWI_FEATURES = False

    # 模型路径
    VIRCHOW2_MODEL_ID = "local-dir:/mnt/g/myq/Virchow2" 
    QWEN_MODEL_PATH = "/mnt/g/myq/Qwen3-Embedding-0.6B" 
    
    # ================= 数据参数 =================
    # 类别映射 (根据实际文件夹名修改)
    # 定义 Class Name -> Label ID 的映射
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
    CHECKPOINT_DIR = "/mnt/g/myq/checkpoints"
    
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
    
    # [新增策略]
    DATA_EXPAND_FACTOR = 5  # 虚拟扩充数据集大小 
    GRAD_ACCUM_STEPS = 4    # 梯度累积步数 
    
    # [验证集划分]
    TRAIN_VAL_SPLIT = 0.8   # 训练集比例
    SEED = 42               # 随机种子

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True
