#!/usr/bin/env python3
"""
对 /media/codingma/code/verify-data/zju 数据集：
1) 提取 SVS 特征（生成 .pt 到 processed）
2) 全模态推理
3) 导出 calibration_long-2.csv 格式（每样本每标签概率 + 是否预测准确）
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from tqdm import tqdm

from mil.config import Config
from mil.inference import InferencePipeline
from mil.wsi_processor_fast import get_virchow2_backbone, extract_wsi_features as extract_wsi_features_fast
from mil.utils import extract_txt_from_doc

# 中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Serif CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

try:
    from sklearn.metrics import accuracy_score, classification_report
except Exception:
    accuracy_score = None
    classification_report = None

TEST_ROOT = Path("/media/codingma/code/verify-data/zju")
# CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints-70_30_multi_label_txt_img_svs"
CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_multi_label_20260207"
OUT_CSV = Path("/media/codingma/LLM/lcx/Medical_Info_Classification/MIL/script/zju/calibration_long-zju-txt-img-svs-mask.csv")
THRESHOLD = 0.5

# Modality switches
USE_TXT = True
USE_IMG = True
USE_SVS = True
ENABLE_FALLBACK = True

# Grad-CAM 可视化开关
SAVE_HEATMAP = False
HEATMAP_DIR = Path("/media/codingma/LLM/lcx/Medical_Info_Classification/MIL/script/zju/heatmaps")

LABEL_MAPPING_RULES = {
    0: [0],  # OLK
    1: [1],  # OLP
    2: [2],  # OSCC
    3: [3],  # OSF
    4: [0, 3],  # OSF+OLK
    5: [4],  # 乳头状瘤
    6: [5],  # 粘液囊肿
    7: [6],  # 纤维增生
}


def _pick_checkpoint() -> str:
    best_path = os.path.join(CHECKPOINT_DIR, "best_val.pth")
    last_path = os.path.join(CHECKPOINT_DIR, "last.pth")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path
    raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")


def _ensure_svs_features(test_root: Path, overwrite: bool = False) -> None:
    model, transform = get_virchow2_backbone(Config.VIRCHOW2_MODEL_ID, Config.DEVICE)
    class_dirs = [d for d in test_root.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        for sample_dir in [p for p in class_dir.iterdir() if p.is_dir()]:
            processed = sample_dir / "processed"
            if not processed.exists():
                continue
            svs_files = glob.glob(str(processed / "*.svs"))
            for svs_path in svs_files:
                save_path = os.path.splitext(svs_path)[0] + ".pt"
                if (not overwrite) and os.path.exists(save_path):
                    continue
                extract_wsi_features_fast(
                    svs_path=svs_path,
                    save_path=save_path,
                    model=model,
                    transform=transform,
                    config=Config(),
                )


def _scan_samples(test_root: Path, use_txt: bool, use_img: bool, use_svs: bool) -> List[Dict]:
    samples: List[Dict] = []
    # ZJU dataset specific mapping
    ZJU_CLASS_MAP = {
        "OLK": 0,
        "OLP": 1,
        "OSCC": 2,
        "OSF": 3,
        "OSF+OLK": 4,
        "乳头状瘤": 5,
        "粘液囊肿": 6,
        "纤维增生": 7,
    }

    for class_name in os.listdir(test_root):
        class_dir = test_root / class_name
        if not class_dir.is_dir():
            continue
        if class_name not in ZJU_CLASS_MAP:
            print(f"Skipping unknown folder: {class_name}")
            continue

        raw_label_id = ZJU_CLASS_MAP[class_name]
        idxs = LABEL_MAPPING_RULES.get(raw_label_id, [])
        label_vec = np.zeros(Config.NUM_CLASSES, dtype=int)
        label_vec[idxs] = 1

        for sample_name in os.listdir(class_dir):
            sample_dir = class_dir / sample_name
            if not sample_dir.is_dir():
                continue

            # text
            text_content = ""
            if use_txt:
                txt_files = glob.glob(str(sample_dir / "*.txt"))
                if txt_files:
                    try:
                        text_content = Path(txt_files[0]).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        text_content = ""

            # images
            img_paths: List[str] = []
            if use_img:
                for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]:
                    img_paths.extend(glob.glob(str(sample_dir / ext)))

            # svs features
            svs_paths: List[str] = []
            if use_svs:
                processed = sample_dir / "processed"
                if processed.exists():
                    svs_paths = glob.glob(str(processed / "*.pt"))

            samples.append(
                {
                    "sample_id": sample_name,
                    "class_name": class_name,
                    "text": text_content,
                    "images": img_paths,
                    "svs": svs_paths,
                    "y_true": label_vec,
                }
            )

    return samples


class ResNetGradCAM:
    """
    Grad-CAM visualizer for the ResNet101 image branch.
    Hooks into pipeline.model.img_backbone to capture activations and gradients,
    then generates class-specific heatmaps overlaid on original images.
    """

    def __init__(self, pipeline: InferencePipeline):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = pipeline.device
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    # ---- hooks ----
    def _fwd_hook(self, module, inp, out):
        self._activations = out  # (N, 2048, h, w) — 保留计算图以便 backward

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]  # (N, 2048, h, w)

    # ---- public API ----
    def __call__(
        self,
        text_input: str,
        image_paths: list,
        svs_paths: list,
        save_dir: Path,
        sample_id: str,
        target_class: Optional[int] = None,
    ) -> None:
        """
        对单个样本生成 Grad-CAM 热力图并保存。

        Args:
            target_class: 可视化的目标类别索引。为 None 时自动使用预测概率最高的类别。
        """
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            return

        # 注册 forward / backward hook
        fwd_h = self.model.img_backbone.register_forward_hook(self._fwd_hook)
        bwd_h = self.model.img_backbone.register_full_backward_hook(self._bwd_hook)

        try:
            # ---- 1. 准备输入（与 InferencePipeline.predict_proba 保持一致）----
            text_enc = self.pipeline.tokenizer(
                text_input if text_input else "",
                max_length=Config.MAX_TEXT_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            input_ids = text_enc['input_ids'].to(self.device)
            attention_mask = text_enc['attention_mask'].to(self.device)

            img_tensors: list = []
            loaded_paths: list = []
            for p in valid_paths:
                try:
                    img = PILImage.open(p).convert('RGB')
                    img_tensors.append(self.pipeline.normal_transform(img))
                    loaded_paths.append(p)
                except Exception:
                    continue
            if not img_tensors:
                return

            normal_imgs = torch.stack(img_tensors).unsqueeze(0).to(self.device)  # (1, N, 3, 224, 224)

            # WSI features
            wsi_feat_list: list = []
            for sp in svs_paths:
                if not os.path.exists(sp) or not sp.endswith('.pt'):
                    continue
                try:
                    feat = torch.load(sp, map_location='cpu')
                    if feat.ndim == 3:
                        feat = feat.view(-1, feat.size(-1))
                    wsi_feat_list.append(feat)
                except Exception:
                    continue

            if wsi_feat_list:
                wsi_feat = torch.cat(wsi_feat_list, dim=0).unsqueeze(0).to(self.device)
                wsi_mask = torch.ones(1, wsi_feat.size(1)).to(self.device)
            else:
                wsi_feat = torch.zeros(1, 1, Config.WSI_INPUT_DIM).to(self.device)
                wsi_mask = torch.zeros(1, 1).to(self.device)

            # ---- 2. 带梯度的前向 + 反向 ----
            self.model.eval()
            with torch.enable_grad():
                with torch.amp.autocast(self.device.type, enabled=Config.USE_AMP, dtype=torch.float16):
                    logits = self.model(input_ids, attention_mask, normal_imgs, wsi_feat, wsi_mask)
                    probs = torch.sigmoid(logits)

                tc = target_class if target_class is not None else int(probs[0].argmax().item())
                self.model.zero_grad()
                logits[0, tc].backward()

            if self._activations is None or self._gradients is None:
                print(f"[GradCAM] 未捕获到特征图/梯度 (sample={sample_id})，跳过。")
                return

            # ---- 3. 计算 Grad-CAM ----
            acts = self._activations.detach().float()   # (N, 2048, h, w)
            grads = self._gradients.detach().float()    # (N, 2048, h, w)

            weights = grads.mean(dim=[2, 3], keepdim=True)            # (N, 2048, 1, 1)
            cam = torch.relu((weights * acts).sum(dim=1)).cpu().numpy()  # (N, h, w)

            # ---- 4. 生成并保存热力图 ----
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            pred_name = Config.TARGET_CLASS_NAMES[tc]

            for i, img_path in enumerate(loaded_paths):
                orig = cv2.imread(img_path)
                if orig is None:
                    continue
                orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                h, w = orig_rgb.shape[:2]

                # 归一化 CAM 到 [0, 1]
                c = cam[i]
                c = c - c.min()
                if c.max() > 0:
                    c = c / c.max()
                cam_resized = cv2.resize(c, (w, h))

                # 伪彩色 + 叠加
                heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                overlay = np.uint8(0.6 * orig_rgb + 0.4 * heatmap_rgb)

                # 绘制三联图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(orig_rgb)
                axes[0].set_title("Original", fontsize=14)
                axes[0].axis('off')

                axes[1].imshow(cam_resized, cmap='jet')
                axes[1].set_title(f"Grad-CAM (class: {pred_name})", fontsize=14)
                axes[1].axis('off')

                axes[2].imshow(overlay)
                axes[2].set_title("Overlay", fontsize=14)
                axes[2].axis('off')

                img_stem = Path(img_path).stem
                plt.suptitle(f"Sample: {sample_id}  |  Image: {img_stem}", fontsize=16)
                plt.tight_layout()
                fname = f"{img_stem}_gradcam.png"
                plt.savefig(str(save_dir / fname), dpi=150, bbox_inches='tight')
                plt.close(fig)

        finally:
            fwd_h.remove()
            bwd_h.remove()
            self._activations = None
            self._gradients = None
            self.model.zero_grad()


def main() -> None:
    if not TEST_ROOT.exists():
        raise SystemExit(f"Test root not found: {TEST_ROOT}")

    # if USE_TXT:
    #     print("Step 0/2: Extracting txt from doc (if needed)...")
    #     extract_txt_from_doc(str(TEST_ROOT))

    if USE_SVS:
        print("Step 1/2: Extracting SVS features (if needed)...")
        _ensure_svs_features(TEST_ROOT, overwrite=False)

    print("Step 2/2: Inference and export...")
    samples = _scan_samples(TEST_ROOT, use_txt=USE_TXT, use_img=USE_IMG, use_svs=USE_SVS)
    if not samples:
        raise SystemExit("No samples found under test root.")

    checkpoint_path = _pick_checkpoint()
    pipeline = InferencePipeline(checkpoint_path)

    # 初始化 Grad-CAM（仅在开启热力图 + 开启图片模态时有效）
    gradcam = None
    if SAVE_HEATMAP and USE_IMG:
        gradcam = ResNetGradCAM(pipeline)
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[GradCAM] 热力图将保存到: {HEATMAP_DIR}")
    elif SAVE_HEATMAP and not USE_IMG:
        print("[GradCAM] 警告: SAVE_HEATMAP=True 但 USE_IMG=False，跳过热力图生成。")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    y_true_mat: List[np.ndarray] = []
    y_prob_mat: List[np.ndarray] = []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        f.write("sample_id,label,y_true,y_prob,y_pred,y_correct\n")
        for s in tqdm(samples, desc="Processing samples"):
            probs = pipeline.predict_proba(s["text"], s["images"], s["svs"])

            # 生成 Grad-CAM 热力图（按 类别/样本名/ 组织目录）
            if gradcam is not None and s["images"]:
                sample_heatmap_dir = HEATMAP_DIR / s["class_name"] / s["sample_id"]
                try:
                    gradcam(
                        text_input=s["text"],
                        image_paths=s["images"],
                        svs_paths=s["svs"],
                        save_dir=sample_heatmap_dir,
                        sample_id=s["sample_id"],
                    )
                except Exception as e:
                    print(f"[GradCAM] {s['sample_id']} 生成失败: {e}")
            y_true = s["y_true"]
            y_true_mat.append(np.asarray(y_true, dtype=int))
            y_prob_mat.append(np.asarray(probs, dtype=float))
            y_pred = (np.asarray(probs) >= THRESHOLD).astype(int)
            if ENABLE_FALLBACK and y_pred.sum() == 0:
                print(">>>>>>>>>>>>>>>>>>>>>>>Fallback<<<<<<<<<<<<<<<<<<<<<<<<")
                y_pred[np.argmax(probs)] = 1

            for i, label in enumerate(Config.TARGET_CLASS_NAMES):
                correct = int(y_pred[i] == y_true[i])
                f.write(
                    f"{s['sample_id']},{label},{int(y_true[i])},{float(probs[i]):.8f},{int(y_pred[i])},{correct}\n"
                )

    print(f"Wrote: {OUT_CSV} (rows={len(samples)*Config.NUM_CLASSES}, samples={len(samples)})")

    if accuracy_score is None or classification_report is None:
        print("sklearn 未安装，已跳过 accuracy_score / classification_report 打印。")
        return

    y_true_arr = np.stack(y_true_mat, axis=0)
    y_prob_arr = np.stack(y_prob_mat, axis=0)
    y_pred_arr = (y_prob_arr >= float(THRESHOLD)).astype(int)

    if ENABLE_FALLBACK and np.any(y_pred_arr.sum(axis=1) == 0):
        empty_mask = (y_pred_arr.sum(axis=1) == 0)
        max_idxs = np.argmax(y_prob_arr[empty_mask], axis=1)
        y_pred_arr[empty_mask] = 0
        y_pred_arr[np.where(empty_mask)[0], max_idxs] = 1

    acc = accuracy_score(y_true_arr, y_pred_arr)
    print(f"Accuracy (subset, threshold={THRESHOLD:.3f}): {acc:.6f}")
    print(
        classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=list(Config.TARGET_CLASS_NAMES),
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
