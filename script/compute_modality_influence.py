#!/usr/bin/env python3
"""
基于 3 模态模型在不同“模态子集”推理得到的 calibration_long-*.csv，
计算类似论文中“模态影响力热力图”的数据（行=模态，列=类别）。

注意：
- 这不是严格 Shapley（缺少空集 baseline 的 v(∅)）；这里使用“移除某模态的敏感度”近似：
    influence_m(c) = mean( | logit(p_full(c)) - logit(p_without_m(c)) | )
- full 指三模态都输入（txt+img+svs）
- without_m 指从 full 中移除某一模态后的组合（例如 without_txt = img+svs）

输入：
  同目录下 7 个 CSV（排除 txt-only），文件名包含 txt/img/svs 表示该组合，例如：
    calibration_long-txt.csv
    calibration_long-img.csv
    calibration_long-svs.csv
    calibration_long-txt-img.csv
    calibration_long-txt-svs.csv
    calibration_long-img-svs.csv
    calibration_long-txt-img-svs.csv

输出：
  modality_influence_heatmap.csv（长表）
    label,modality,mean_abs_all,mean_abs_pos,n_all,n_pos,norm_abs_all,norm_abs_pos
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Tuple

import numpy as np
import pandas as pd


MODALITIES = ("txt", "img", "svs")


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _infer_modalities_from_name(path: Path) -> FrozenSet[str]:
    name = path.name.lower()
    present = set()
    for m in MODALITIES:
        if m in name:
            present.add(m)
    return frozenset(present)


def _load_long_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"sample_id", "label", "y_true", "y_prob"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列：{sorted(missing)}")
    df = df[["sample_id", "label", "y_true", "y_prob"]].copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)
    df = df.set_index(["sample_id", "label"]).sort_index()
    return df


def _discover_files(script_dir: Path) -> Dict[FrozenSet[str], Path]:
    files = {}
    for p in script_dir.glob("calibration_long-*.csv"):
        # 排除另一个模型
        if "txt-only" in p.name.lower():
            continue
        mods = _infer_modalities_from_name(p)
        if not mods:
            # 如 calibration_long-2.csv 之类不带模态标记的文件，跳过
            continue
        files[mods] = p
    return files


def _calc_influence(
    full_df: pd.DataFrame,
    minus_df: pd.DataFrame,
    eps: float,
) -> pd.DataFrame:
    """
    返回按 (sample_id,label) 对齐后的 influence_abs 列（全样本），以及 y_true（来自 full）。
    """
    j = full_df[["y_true", "y_prob"]].join(
        minus_df[["y_prob"]].rename(columns={"y_prob": "y_prob_minus"}),
        how="inner",
    )
    if j.empty:
        raise ValueError("full 与 minus 没有可对齐的 (sample_id,label) 记录")
    lf = _logit(j["y_prob"].to_numpy(), eps)
    lm = _logit(j["y_prob_minus"].to_numpy(), eps)
    j["influence_abs"] = np.abs(lf - lm)
    return j[["y_true", "influence_abs"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--script-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="包含 calibration_long-*.csv 的目录（默认当前脚本目录）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 CSV 路径（默认写到 script-dir/modality_influence_heatmap.csv）",
    )
    ap.add_argument("--eps", type=float, default=1e-6, help="logit 裁剪用 eps，避免 0/1 概率")
    args = ap.parse_args()

    script_dir: Path = args.script_dir
    out_path: Path = args.out or (script_dir / "modality_influence_heatmap.csv")

    files = _discover_files(script_dir)
    required = {
        frozenset({"txt"}),
        frozenset({"img"}),
        frozenset({"svs"}),
        frozenset({"txt", "img"}),
        frozenset({"txt", "svs"}),
        frozenset({"img", "svs"}),
        frozenset({"txt", "img", "svs"}),
    }
    missing = required - set(files.keys())
    if missing:
        msg = ", ".join(["+".join(sorted(s)) for s in sorted(missing, key=lambda x: (len(x), sorted(x)))])
        raise SystemExit(f"缺少这些模态组合对应的 CSV：{msg}")

    full_path = files[frozenset({"txt", "img", "svs"})]
    print(f"Using full (txt+img+svs): {full_path.name}")

    full_df = _load_long_csv(full_path)

    # 计算三种“移除模态”影响：T vs IS, I vs TS, S vs TI
    mapping: Dict[str, Tuple[FrozenSet[str], str]] = {
        "txt": (frozenset({"img", "svs"}), "img+svs"),
        "img": (frozenset({"txt", "svs"}), "txt+svs"),
        "svs": (frozenset({"txt", "img"}), "txt+img"),
    }

    rows = []
    labels = sorted({idx[1] for idx in full_df.index})

    for m in MODALITIES:
        minus_key, minus_desc = mapping[m]
        minus_path = files[minus_key]
        minus_df = _load_long_csv(minus_path)

        infl = _calc_influence(full_df, minus_df, eps=float(args.eps))
        # 统计：全样本、正样本（y_true==1）
        for lab in labels:
            lab_df = infl.xs(lab, level="label", drop_level=False)
            n_all = int(len(lab_df))
            mean_all = float(lab_df["influence_abs"].mean()) if n_all else 0.0

            pos_df = lab_df[lab_df["y_true"] == 1]
            n_pos = int(len(pos_df))
            mean_pos = float(pos_df["influence_abs"].mean()) if n_pos else 0.0

            rows.append(
                {
                    "label": lab,
                    "modality": m,
                    "minus_combo": minus_desc,
                    "mean_abs_all": mean_all,
                    "mean_abs_pos": mean_pos,
                    "n_all": n_all,
                    "n_pos": n_pos,
                }
            )

    out_df = pd.DataFrame(rows)

    # 归一化：每个 label 下，让 3 个模态的影响之和为 1（便于画“红=最重要”）
    def _norm_group(col: str) -> pd.Series:
        s = out_df.groupby("label")[col].transform("sum")
        return np.where(s.to_numpy() > 0, out_df[col].to_numpy() / s.to_numpy(), 0.0)

    out_df["norm_abs_all"] = _norm_group("mean_abs_all")
    out_df["norm_abs_pos"] = _norm_group("mean_abs_pos")

    out_df = out_df.sort_values(["label", "modality"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote: {out_path}  (rows={len(out_df)})")


if __name__ == "__main__":
    main()

