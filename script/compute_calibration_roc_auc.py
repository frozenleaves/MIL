import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def _compute_one_file(path: str, fpr_grid: np.ndarray):
    df = pd.read_csv(path)
    required_cols = {"sample_id", "label", "y_true", "y_prob"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {required_cols - set(df.columns)}")

    file_id = os.path.splitext(os.path.basename(path))[0]

    auc_rows = []
    
    tpr_list = []

    for label, g in df.groupby("label"):
        y_true = g["y_true"].astype(int).values
        y_prob = g["y_prob"].astype(float).values

        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)

        if np.unique(y_true).size < 2:
            auc_val = np.nan
        else:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = float(auc(fpr, tpr))
            tpr_interp = np.interp(fpr_grid, fpr, tpr, left=0.0, right=1.0)
            tpr_list.append(tpr_interp)

        auc_rows.append(
            {
                "file": file_id,
                "label": label,
                "auc": auc_val,
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
        )

    if tpr_list:
        mean_tpr = np.mean(np.stack(tpr_list, axis=0), axis=0)
        roc_rows = [
            {"file": file_id, "fpr": float(fpr_grid[i]), "tpr_mean": float(mean_tpr[i])}
            for i in range(len(fpr_grid))
        ]
    else:
        roc_rows = []

    return auc_rows, roc_rows


def main():
    parser = argparse.ArgumentParser(description="Compute macro-averaged ROC and AUC per label.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Directory containing calibration*.csv",
    )
    parser.add_argument("--pattern", type=str, default="calibration*.csv", help="File pattern")
    parser.add_argument("--out_auc_csv", type=str, default="auc_per_label.csv", help="AUC output CSV")
    parser.add_argument("--out_roc_csv", type=str, default="roc_mean_curve.csv", help="ROC output CSV")
    parser.add_argument("--grid_points", type=int, default=101, help="FPR grid points")
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise RuntimeError(f"No files found under {args.input_dir} with pattern {args.pattern}")

    fpr_grid = np.linspace(0.0, 1.0, num=args.grid_points)

    all_auc = []
    all_roc = []
    for path in files:
        auc_rows, roc_rows = _compute_one_file(path, fpr_grid)
        all_auc.extend(auc_rows)
        all_roc.extend(roc_rows)

    auc_df = pd.DataFrame(all_auc)
    roc_df = pd.DataFrame(all_roc)

    auc_df.to_csv(args.out_auc_csv, index=False)
    roc_df.to_csv(args.out_roc_csv, index=False)

    print(f"Saved AUC: {args.out_auc_csv}")
    print(f"Saved ROC: {args.out_roc_csv}")


if __name__ == "__main__":
    main()
