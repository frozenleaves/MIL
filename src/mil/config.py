

class Config:
    # ================= 路径配置 =================
    # 数据集根目录 
    RAW_DATA_ROOT = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets/train"
    #RAW_DATA_ROOT = "/media/codingma/LLM/data-1005"

    DATA_INDEX_PATH = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets/train/index_20260207.csv"
    OVERWRITE_SWI_FEATURES = False

    # 模型路径
    VIRCHOW2_MODEL_ID = "local-dir:/media/codingma/LLM/lcx/Virchow2" # HuggingFace ID
    QWEN_MODEL_PATH = "/media/codingma/LLM/lcx/Qwen3-Embedding-0.6B" 

    # ================= 数据参数 =================
    # 类别映射 (根据实际文件夹名修改)
    # [修改] 
    # RAW_CLASS_MAP: 对应 CSV 中已有的 0-7 标签 (物理存储)
    # NUM_CLASSES: 模型实际输出的维度 (7)
    
    NUM_CLASSES = 7 
    
    # 原始数据集的类别映射
    RAW_CLASS_MAP = {
        "已整理-OLK": 0,
        "已整理-OLP": 1,
        "已整理-OSCC": 2,
        "已整理-OSF": 3,
        "已整理-OSF+OLK": 4,
        "已整理-乳头状瘤": 5,
        "已整理-粘液囊肿": 6,
        "已整理-纤维增生": 7,
    }
    
    # 模型输出label-id映射
    TARGET_CLASS_NAMES = [
        "OLK",      # 0
        "OLP",      # 1
        "OSCC",     # 2
        "OSF",      # 3
        "乳头状瘤",  # 4
        "粘液囊肿",  # 5
        "纤维增生"   # 6
    ]

    # 兼容旧代码，保留 CLASS_MAP，但指向 RAW (如果其他地方用到的话)
    CLASS_MAP = RAW_CLASS_MAP
    
    # WSI 处理参数
    PATCH_SIZE = 224       # Virchow2 固定尺寸
    TILE_LEVEL = 0         # 40x / 20x 最高倍率
    BATCH_SIZE_WSI = 256    # 特征提取时的 Batch Size
    NUM_WORKERS = 8        # DataLoader workers
    BG_THRESHOLD = 220     # 去除背景的阈值
    SAVE_COORDS = False
    USE_FAST_VERSION = True

    # ================= 训练参数 =================
    # 权重保存目录
    # CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints-txt_only"
    CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_multi_label_20260207"
    #CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_multi_label"

    
    # [模型容量配置]
    FUSION_DIM = 768       # 512 -> 768 (增大维度)
    FUSION_LAYERS = 6      # 2 -> 6 (增加深度)
    FUSION_HEADS = 12      # 768 / 12 = 64 (必须能整除 FUSION_DIM)
    FUSION_DROPOUT = 0.1

    WSI_INPUT_DIM = 2560
    MAX_TEXT_LEN = 512
    FREEZE_TEXT_MODEL = True
    
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    
    # [新增策略]
    DATA_EXPAND_FACTOR = 10  # 虚拟扩充数据集大小 
    GRAD_ACCUM_STEPS = 4    # 梯度累积步数 
    
    # [验证集划分]
    TRAIN_VAL_SPLIT = 0.8   # 训练集比例
    SEED = 42               # 随机种子

    DEVICE = "cuda"
    USE_AMP = True

    # [模态开关]
    USE_TXT = True
    USE_IMG = True
    USE_SVS = True

    # ================= 学习曲线参数 =================
    # 训练集规模可用“比例(<=1)”或“绝对数量(>1)”
    LEARNING_CURVE_ENABLE = False
    LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    LEARNING_CURVE_REPEATS = 1
    LEARNING_CURVE_BASE_SEED = 42
