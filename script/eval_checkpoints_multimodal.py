import os
import re
import csv
import gc
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm
from mil.config import Config
from mil.dataset import MultimodalDataset, collate_fn
from mil.model import UnifiedMultimodalModel
from mil.train import multilabel_categorical_crossentropy


CHECKPOINT_ROOT = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_multi_label_20260121"
TEST_DATA_ROOT = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets/test"
OUT_CSV = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_multi_label_20260121/inference_metrics.csv"


def _parse_train_size(path: str):
    m = re.search(r"size_(\d+)_seed", path)
    if m:
        return int(m.group(1))
    return None


def _find_checkpoint_dirs(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        if "best_val.pth" in filenames or "last.pth" in filenames:
            yield dirpath


def _pick_checkpoint(ckpt_dir: str):
    best_path = os.path.join(ckpt_dir, "best_val.pth")
    last_path = os.path.join(ckpt_dir, "last.pth")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path
    return None


def _load_state_dict(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _build_test_df_from_dir(test_root: str):
    data_records = []
    if not os.path.exists(test_root):
        raise FileNotFoundError(f"Test root not found: {test_root}")

    class_dirs = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
    for class_name in class_dirs:
        if class_name not in Config.CLASS_MAP:
            continue
        label = Config.CLASS_MAP[class_name]
        class_path = os.path.join(test_root, class_name)
        sample_names = os.listdir(class_path)

        for sample_name in sample_names:
            sample_dir = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_dir):
                continue

            txt_files = glob.glob(os.path.join(sample_dir, "*.txt"))
            txt_path = txt_files[0] if txt_files else ""

            jpg_files = []
            jpg_files.extend(glob.glob(os.path.join(sample_dir, "*.JPG")))
            jpg_files.extend(glob.glob(os.path.join(sample_dir, "*.jpg")))
            jpg_files.extend(glob.glob(os.path.join(sample_dir, "*.jpeg")))
            jpg_paths_str = ";".join(jpg_files)

            svs_dir = os.path.join(sample_dir, "processed")
            wsi_feat_paths = []
            if os.path.exists(svs_dir):
                pt_files = glob.glob(os.path.join(svs_dir, "*.pt"))
                wsi_feat_paths.extend(pt_files)
            wsi_paths_str = ";".join(wsi_feat_paths)

            if txt_path or jpg_paths_str or wsi_paths_str:
                data_records.append(
                    {
                        "sample_id": sample_name,
                        "txt_path": txt_path,
                        "img_paths": jpg_paths_str,
                        "wsi_paths": wsi_paths_str,
                        "label": label,
                    }
                )

    return pd.DataFrame(data_records)


def _build_test_loader(test_root: str):
    test_df = _build_test_df_from_dir(test_root).reset_index(drop=True)

    dataset = MultimodalDataset(
        test_df,
        mode="test",
        use_txt=True,
        use_img=True,
        use_svs=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return loader, len(test_df)


def _eval_one_checkpoint(checkpoint_path: str, test_loader, device):
    model = UnifiedMultimodalModel().to(device)
    state_dict = _load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    total_loss = 0.0
    preds_all = []
    labels_all = []
    probs_all = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            normal_imgs = batch["normal_imgs"].to(device)
            wsi_feat = batch["wsi_feat"].to(device)
            wsi_mask = batch["wsi_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=Config.USE_AMP):
                logits = model(input_ids, attn_mask, normal_imgs, wsi_feat, wsi_mask)
                loss = torch.mean(multilabel_categorical_crossentropy(logits.float(), labels.float()))

            total_loss += float(loss.item())
            preds = (logits > 0).float()
            probs = torch.sigmoid(logits.float())
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(1, len(test_loader))
    y_true = np.asarray(labels_all)
    y_pred = np.asarray(preds_all)
    y_prob = np.asarray(probs_all)

    acc = accuracy_score(y_true, y_pred)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    try:
        auc_macro = roc_auc_score(y_true, y_prob, average="macro")
    except Exception:
        auc_macro = float("nan")
    try:
        auc_micro = roc_auc_score(y_true, y_prob, average="micro")
    except Exception:
        auc_micro = float("nan")
    try:
        auc_weighted = roc_auc_score(y_true, y_prob, average="weighted")
    except Exception:
        auc_weighted = float("nan")

    per_precision, per_recall, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_acc = (y_true == y_pred).mean(axis=0)

    del model, preds_all, labels_all, y_true, y_pred
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "loss": avg_loss,
        "acc": acc,
        "recall_macro": recall_macro,
        "precision_macro": precision_macro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        "auc_weighted": auc_weighted,
        "per_precision": per_precision,
        "per_recall": per_recall,
        "per_f1": per_f1,
        "per_acc": per_acc,
    }


def _append_row(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(row)


def main():
    device = torch.device(Config.DEVICE)
    test_loader, test_size = _build_test_loader(TEST_DATA_ROOT)

    for ckpt_dir in sorted(_find_checkpoint_dirs(CHECKPOINT_ROOT)):
        ckpt_path = _pick_checkpoint(ckpt_dir)
        if not ckpt_path:
            continue

        metrics = _eval_one_checkpoint(ckpt_path, test_loader, device)
        row = {
            "checkpoint_dir": ckpt_dir,
            "checkpoint_file": os.path.basename(ckpt_path),
            "train_size": _parse_train_size(ckpt_dir),
            "test_samples": test_size,
            "loss": f"{metrics['loss']:.8f}",
            "acc": f"{metrics['acc']:.8f}",
            "recall_macro": f"{metrics['recall_macro']:.8f}",
            "precision_macro": f"{metrics['precision_macro']:.8f}",
            "f1_macro": f"{metrics['f1_macro']:.8f}",
            "f1_micro": f"{metrics['f1_micro']:.8f}",
            "auc_macro": f"{metrics['auc_macro']:.8f}",
            "auc_micro": f"{metrics['auc_micro']:.8f}",
            "auc_weighted": f"{metrics['auc_weighted']:.8f}",
        }
        for i in range(min(Config.NUM_CLASSES, len(metrics["per_acc"]))):
            if i < len(Config.TARGET_CLASS_NAMES):
                class_name = Config.TARGET_CLASS_NAMES[i]
            else:
                class_name = f"class_{i}"
            row[f"{class_name}_acc"] = f"{metrics['per_acc'][i]:.8f}"
            row[f"{class_name}_precision"] = f"{metrics['per_precision'][i]:.8f}"
            row[f"{class_name}_recall"] = f"{metrics['per_recall'][i]:.8f}"
            row[f"{class_name}_f1"] = f"{metrics['per_f1'][i]:.8f}"
        _append_row(OUT_CSV, row)
        print(f"✅ Evaluated: {ckpt_dir}")


if __name__ == "__main__":
    main()
