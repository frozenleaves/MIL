import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _try_get_default_root() -> str:
    try:
        from mil.config import Config

        return getattr(Config, "CHECKPOINT_DIR", ".")
    except Exception:
        return "."


def _find_auc_logs(root: str) -> List[str]:
    auc_logs = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == "auc_log.csv":
                auc_logs.append(os.path.join(dirpath, name))
    return auc_logs


def _pick_run_row(df: pd.DataFrame, metric: str) -> pd.Series:
    if df.empty:
        return df
    if metric == "last":
        return df.sort_values("epoch").iloc[-1]
    # best by val_auc
    df_sorted = df.sort_values("val_auc", ascending=False)
    if df_sorted["val_auc"].notna().any():
        return df_sorted.iloc[0]
    return df.sort_values("epoch").iloc[-1]


def _collect_runs(root: str, metric: str) -> pd.DataFrame:
    rows = []
    for path in _find_auc_logs(root):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        required = {"run_tag", "seed", "train_size", "epoch", "train_auc", "val_auc"}
        if not required.issubset(set(df.columns)):
            continue
        row = _pick_run_row(df, metric)
        if isinstance(row, pd.Series):
            rows.append(
                {
                    "run_tag": row["run_tag"],
                    "seed": int(row["seed"]),
                    "train_size": int(row["train_size"]),
                    "train_auc": float(row["train_auc"]),
                    "val_auc": float(row["val_auc"]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["run_tag", "seed", "train_size", "train_auc", "val_auc"])
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby("train_size", as_index=False)
    summary = grouped.agg(
        train_auc_mean=("train_auc", "mean"),
        train_auc_std=("train_auc", "std"),
        val_auc_mean=("val_auc", "mean"),
        val_auc_std=("val_auc", "std"),
        n_runs=("run_tag", "count"),
    )
    return summary.sort_values("train_size")


def _plot(summary: pd.DataFrame, out_png: str, title: str) -> None:
    if summary.empty:
        raise ValueError("no data to plot")
    x = summary["train_size"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        summary["train_auc_mean"].values,
        yerr=summary["train_auc_std"].values,
        fmt="-o",
        capsize=3,
        label="Train AUC",
    )
    ax.errorbar(
        x,
        summary["val_auc_mean"].values,
        yerr=summary["val_auc_std"].values,
        fmt="-o",
        capsize=3,
        label="Val AUC",
    )
    ax.set_xlabel("Train size")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot learning curve (AUC vs train size).")
    parser.add_argument("--root", type=str, default=_try_get_default_root(), help="Checkpoint root dir")
    parser.add_argument("--metric", type=str, choices=["best", "last"], default="best", help="Select epoch")
    parser.add_argument("--out_csv", type=str, default="learning_curve_auc_summary.csv", help="Summary CSV")
    parser.add_argument("--out_png", type=str, default="learning_curve_auc.png", help="Output plot PNG")
    args = parser.parse_args()

    runs_df = _collect_runs(args.root, args.metric)
    if runs_df.empty:
        raise RuntimeError(f"No auc_log.csv found under: {args.root}")

    summary = _aggregate(runs_df)
    summary.to_csv(args.out_csv, index=False)

    title = f"Learning Curve (metric={args.metric})"
    _plot(summary, args.out_png, title)
    print(f"Saved summary: {args.out_csv}")
    print(f"Saved plot: {args.out_png}")


if __name__ == "__main__":
    main()
