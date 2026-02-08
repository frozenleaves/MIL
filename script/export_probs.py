#!/usr/bin/env python3
"""
导出“校准曲线所需”的逐样本预测概率数据（多标签）。

输出为长表 CSV（推荐用于 micro-average 校准）：
  sample_id,label,y_true,y_prob

解释：
- 多标签校准常用做法之一：把每个 (样本, 类别) 当成一个二分类样本（y_true∈{0,1}，y_prob 为该类 sigmoid 概率）。
  这样就能画 micro-average calibration curve（概率->实际发生率）。

注意：
- 需要可用的 checkpoint + 数据集路径 +（若启用 WSI 且是 .svs）Virchow2 资源。
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from mil.config import Config
from mil.inference import InferencePipeline

try:
    from sklearn.metrics import accuracy_score, classification_report
except Exception:  # sklearn 可能未安装；不影响导出功能
    accuracy_score = None
    classification_report = None


LABEL_MAPPING_RULES = {
    0: [0],  # OLK
    1: [1],  # OLP
    2: [2],  # OSCC
    3: [3],  # OSF
    4: [0, 3],  # OSF+OLK -> OLK + OSF
    5: [4],  # 乳头状瘤
    6: [5],  # 粘液囊肿
    7: [6],  # 纤维增生
}


def scan_test_root(test_root: Path, use_txt: bool, use_img: bool, use_svs: bool) -> List[Dict]:
    samples: List[Dict] = []
    raw_class_map = Config.RAW_CLASS_MAP

    for class_name in os.listdir(test_root):
        class_dir = test_root / class_name
        if not class_dir.is_dir():
            continue
        if class_name not in raw_class_map:
            continue

        raw_label_id = raw_class_map[class_name]
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

            # svs/pt
            svs_paths: List[str] = []
            if use_svs:
                processed = sample_dir / "processed"
                if processed.exists():
                    svs_paths = glob.glob(str(processed / "*.pt"))

            samples.append(
                {
                    "sample_id": sample_name,
                    "text": text_content,
                    "images": img_paths,
                    "svs": svs_paths,
                    "y_true": label_vec,
                }
            )

    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", type=Path, required=True, help="测试集根目录（含各类别子文件夹）")
    ap.add_argument("--out", type=Path, default=Path("calibration_long.csv"))
    ap.add_argument("--threshold", type=float, default=0.5, help="多标签判定阈值（用于 accuracy / report）")
    ap.add_argument(
        "--enable-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="当阈值后预测全 0 时，回退为 argmax(probs)=1（与 inference.py 行为一致）",
    )


    args = ap.parse_args()


    if not args.test_root.exists():
        raise SystemExit(f"test-root 不存在：{args.test_root}")


    # 默认全开（与你的最佳模型 test1 一致）
    use_txt = False
    use_img = True
    use_svs = True

    samples = scan_test_root(args.test_root, use_txt=use_txt, use_img=use_img, use_svs=use_svs)
    if not samples:
        raise SystemExit("未扫描到样本，请检查 test-root 目录结构")
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR , "best_val.pth")
    pipeline = InferencePipeline(checkpoint_path)

    # 写出长表
    y_true_mat: List[np.ndarray] = []
    y_prob_mat: List[np.ndarray] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        f.write("sample_id,label,y_true,y_prob\n")
        for s in tqdm(samples, desc="Processing samples"):
            probs = pipeline.predict_proba(s["text"], s["images"], s["svs"])  # shape (7,)
            y_true = s["y_true"]
            y_true_mat.append(np.asarray(y_true, dtype=int))
            y_prob_mat.append(np.asarray(probs, dtype=float))
            for i, label in enumerate(Config.TARGET_CLASS_NAMES):
                f.write(f"{s['sample_id']},{label},{int(y_true[i])},{float(probs[i]):.8f}\n")

    print(f"Wrote: {args.out}  (rows={len(samples)*Config.NUM_CLASSES}, samples={len(samples)})")

    # 打印 accuracy 与 classification report（多标签：按阈值二值化）
    if accuracy_score is None or classification_report is None:
        print("sklearn 未安装，已跳过 accuracy / classification_report 打印。可安装：pip install scikit-learn")
        return

    y_true_arr = np.stack(y_true_mat, axis=0)  # (N, C)
    y_prob_arr = np.stack(y_prob_mat, axis=0)  # (N, C)
    y_pred_arr = (y_prob_arr >= float(args.threshold)).astype(int)

    # 与 inference.py 保持一致：若预测全 0，则回退到概率最大的一类
    if bool(args.enable_fallback):
        empty_mask = (y_pred_arr.sum(axis=1) == 0)
        if np.any(empty_mask):
            max_idxs = np.argmax(y_prob_arr[empty_mask], axis=1)
            y_pred_arr[empty_mask] = 0
            y_pred_arr[np.where(empty_mask)[0], max_idxs] = 1

    acc = accuracy_score(y_true_arr, y_pred_arr)  # subset accuracy（样本级完全匹配）
    print(f"Accuracy (subset, threshold={args.threshold:.3f}): {acc:.6f}")
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

