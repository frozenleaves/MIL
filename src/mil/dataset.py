import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from config import Config

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, mode='train'):
        self.data = pd.read_csv(csv_path)
        
        # 初始化 Qwen Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.QWEN_MODEL_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 普通图片预处理
        self.normal_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # ================= 1. 读取文本文件 =================
        txt_path = row['txt_path']
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
        except Exception:
            text_content = "" # 容错
            
        text_enc = self.tokenizer(
            text_content,
            max_length=Config.MAX_TEXT_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_enc['input_ids'].squeeze(0)
        attention_mask = text_enc['attention_mask'].squeeze(0)

        # ================= 2. 读取普通图片 (多个) =================
        img_paths_str = str(row['img_paths'])
        img_tensors = []
        if img_paths_str and img_paths_str != 'nan':
            paths = img_paths_str.split(';')
            for path in paths:
                if os.path.exists(path):
                    try:
                        img = Image.open(path).convert('RGB')
                        img_tensors.append(self.normal_transform(img))
                    except:
                        pass
        
        if len(img_tensors) > 0:
            normal_imgs = torch.stack(img_tensors)
        else:
            normal_imgs = torch.zeros(1, 3, 224, 224) # 占位

        # ================= 3. 读取 WSI 特征 (可能多个) =================
        wsi_paths_str = str(row['wsi_paths'])
        wsi_feat_list = []
        
        if wsi_paths_str and wsi_paths_str != 'nan':
            paths = wsi_paths_str.split(';')
            for path in paths:
                if os.path.exists(path):
                    # Load: [M, 261, 1280] or [M, 2560]
                    # print(f"Loading WSI feat from {path}")
                    try:
                        feat = torch.load(path, map_location='cpu')
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        continue
                        
                    if feat.ndim == 3:
                        M, T, D = feat.shape
                        # Flatten: [M*261, 1280]
                        feat = feat.view(M * T, D)
                    # 如果已经是 2D [M, D]，则不需要处理
                    
                    wsi_feat_list.append(feat)
        
        if len(wsi_feat_list) > 0:
            # 如果有多个 svs，直接在序列维度拼接 (Concatenate along sequence dim)
            wsi_feat = torch.cat(wsi_feat_list, dim=0) # [Total_Tokens, 1280]
        else:
            wsi_feat = torch.zeros(1, Config.WSI_INPUT_DIM)

        label = torch.tensor(int(row['label']), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'normal_imgs': normal_imgs,
            'wsi_feat': wsi_feat,
            'label': label
        }

def collate_fn(batch):
    """(保持不变)"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Normal Imgs padding
    normal_imgs = [item['normal_imgs'] for item in batch]
    max_n = max([img.shape[0] for img in normal_imgs])
    padded_imgs = []
    for img in normal_imgs:
        pad_size = max_n - img.shape[0]
        if pad_size > 0:
            pad = torch.zeros(pad_size, 3, 224, 224)
            img = torch.cat([img, pad], dim=0)
        padded_imgs.append(img)
    normal_imgs_batch = torch.stack(padded_imgs)
    
    # WSI padding
    wsi_feats = [item['wsi_feat'] for item in batch]
    wsi_batch = torch.nn.utils.rnn.pad_sequence(wsi_feats, batch_first=True, padding_value=0)
    
    wsi_mask = torch.zeros(wsi_batch.shape[0], wsi_batch.shape[1])
    for i, feat in enumerate(wsi_feats):
        wsi_mask[i, :feat.shape[0]] = 1
        
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'normal_imgs': normal_imgs_batch,
        'wsi_feat': wsi_batch,
        'wsi_mask': wsi_mask,
        'labels': labels
    }