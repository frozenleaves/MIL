## 简介
项目以Qwen3-0.6B-Embedding模型编码文本特征，以 ResNet-101 编码常规图像特征，以 Virchow2 编码病理切片特征







## 参数

### 配置文件参数说明

`RAW_DATA_ROOT：` 数据集的根目录，例如 `/media/codingma/LLM/lcx/Medical_Info_Classification/datasets/train`

`DATA_INDEX_PATH`: 数据集索引文件所在目录，该文件可由Medical_Info_Classification/MIL/prepare_dataset.py 这一脚本自动生成，用于获取每个样本中有多少数据文件，只需要运行 ` python prepare_dataset.py` 即可在DATA_INDEX_PATH指定的路径生成索引文件。

`OVERWRITE_SWI_FEATURES：`在执行prepare_dataset脚本时，是否重新生成切片特征，如果为True则会重新由Virchow2来为每个切片文件提取特征，如果为False，当特征文件已经提取过后，则不会重新提取。

`VIRCHOW2_MODEL_ID`：Virchow2 模型权重路径或者hugguing face id，该模型需要获取访问权限，传入的时候应该是以下两种形式：1. "hf-hub:paige-ai/Virchow2"；2. "local-dir:/media/codingma/LLM/lcx/Virchow2"，1是在线路径，2是本地路径

`QWEN_MODEL_PATH:` Qwen3-Embedding模型权重路径，或者 hf url

`CLASS_MAP:` 类别id——类别名的映射

`CHECKPOINT_DIR:` 训练权重保存路径

`NUM_CLASSES：` 模型分类头输出类别数量


以下为 WSI 在提取切片特征时的处理参数
```yaml

PATCH_SIZE = 224            # Virchow2 固定尺寸
TILE_LEVEL = 0              # 40x / 20x 最高倍率
BATCH_SIZE_WSI = 256        # 特征提取时的 Batch Size
NUM_WORKERS = 8             # DataLoader workers
BG_THRESHOLD = 220          # 去除背景的阈值
SAVE_COORDS = False         # 是否保存每个patch的坐标
```

以下为模型容量参数配置
```yaml
FUSION_DIM = 768            # GatedAttentionMIL模型的隐藏层维度，512 -> 768 (增大维度)
FUSION_LAYERS = 6           # Fusion module的隐藏层数量，2 -> 6 (增加深度)
FUSION_HEADS = 12           # Fusion module的attention head数量，768 / 12 = 64 (必须能整除 FUSION_DIM)
FUSION_DROPOUT = 0.1        # dropout参数

WSI_INPUT_DIM = 2560        # 切片预处理特征向量的输出维度，也即切片数据送入模型的输入维度
MAX_TEXT_LEN = 512          # Embedding 模型最大文本长度
FREEZE_TEXT_MODEL = True    # 是否冻结文本模型训练

```

以下为训练过程的超参数
```yaml
BATCH_SIZE = 4              # 训练的 batch size
EPOCHS = 10                 # 训练 epoch
LEARNING_RATE = 2e-5        # 学习率
WEIGHT_DECAY = 0.0001       # 学习率衰减指数

# [新增策略]
DATA_EXPAND_FACTOR = 10     # 虚拟扩充数据集大小，训练过程中会对参与训练的数据集进行数据增强，增强后的训练数据总量为原始数据量乘以 DATA_EXPAND_FACTOR
GRAD_ACCUM_STEPS = 4        # 梯度累积步数 

# [验证集划分]
TRAIN_VAL_SPLIT = 0.8       # 训练集与验证集的比例，测试集不在此划分序列内
SEED = 42                   # 随机种子

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True              # 是否启用混合精度

```

## 模型


## 数据集
当前数据集存放在/media/codingma/LLM/lcx/Medical_Info_Classification/datasets，该路径下的train和test目录下的样本文件均为软连接，删除不影响原文件，如有重新划分的需求，可以随时清空。

在启动训练之前，需要按照如下步骤准备数据集：

0. 将doc文件提取内容生成txt文本，这一步可

1. 为切片数据离线提取特征
   通过 `Medical_Info_Classification/MIL/prepare_dataset.py`提供的`generate_wsi_features`函数，可以为RAW_DATA_ROOT路径下的所有切片数据，离线提取特征，该函数接受一个overwrite的bool参数，为True时表示每次运行都重新生成特征文件，为False时则跳过已生成特征的切片数据，只为没有生成过的切片数据提取特征。

2. 划分训练数据集合测试数据集
    
    通过`Medical_Info_Classification/MIL/prepare_dataset.py`提供的`create_symlink_split`,可以将数据集通过软连接的方式，按照传入的train_ratio按比例随机切分为训练集和测试集，当前代码中数据集的默认划分比例为0.5.

2. 为训练数据集生成索引文件
    索引文件是为了获取每个数据集中，各个文件的所在路径，是一个CSV表格，该文件可以通过`Medical_Info_Classification/MIL/prepare_dataset.py`提供的`generate_index_file`函数生成, 索引文件保存路径为config.py中DATA_INDEX_PATH指定的路径。

在准备完成上述数据后，即可开始训练模型


## 训练

训练开始前需要准备好数据索引文件，如果需要重新划分测试集，可以在生成索引文件之前，通过Medical_Info_Classification/MIL/src/mil/utils.py中提供的脚本来重新划分，配置完相关切分参数，将原有的软连接删除或者重新指定保存路径，重新运行即可。
```shell
# 项目根目录
cd /media/codingma/LLM/lcx/Medical_Info_Classification/MIL

# 安装
pip install -e .


启动训练方式： 

# 以默认配置启动训练
python train.py

```

权重文件会保存两份：一份是在每个epoch训练完成时，验证准确率最好的模型权重best_val.pth，另一份是训练结束时的权重last.pth,使用时以best_val.pth为准。


## 推理

inference.py 提供了推理脚本，入口为 `evaluate_test_set`，该函数接受一个test_root参数，指定测试数据集的路径，可选use_txt、use_img、use_svs三个bool参数，默认均为True，设为False表示相应地禁用文本、照片、切片数据。输出结果包括如下部分：

1. 推理使用样本数量和样本分布情况

2. 测试精确度Accuracy

3. Classification Report，包括precision、recall、f1-score、 macro avg、weighted avg等指标

4. Confusion Matrix和ROC曲线图

```shell
# 以默认配置启动推理
cd /media/codingma/LLM/lcx/Medical_Info_Classification/MIL

python inference.py

```



## 实验结果