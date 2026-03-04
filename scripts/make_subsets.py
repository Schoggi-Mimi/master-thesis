# scripts/make_subsets.py

# Run example:
# python -m scripts.make_subsets \
#   --gt_csv data/isic2018/val_gt.csv \
#   --pred_csv outputs/preds_isic7_val.csv \
#   --out_dir data/isic2018/subsets \
#   --tau 0.10 \
#   --delta 0.30 \
#   --split_name val

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


GT_CLASS_COLS_DEFAULT = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
PRED_PROB_COLS_DEFAULT = ["p_AK", "p_BCC", "p_BKL", "p_DF", "p_MEL", "p_NV", "p_VASC"]
PRED_LABELS_DEFAULT = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create ISIC2018 subsets for MEL vs NV analysis.")
    p.add_argument("--gt_csv", type=str, required=True, help="Path to GT CSV (one-hot columns + 'image').")
    p.add_argument("--pred_csv", type=str, required=True, help="Path to prediction CSV (p_* columns, optional pred_label).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for subset CSVs.")
    p.add_argument("--tau", type=float, default=0.10, help="Threshold for min(p_MEL, p_NV) in subset2a/2b.")
    p.add_argument("--delta", type=float, default=0.30, help="Threshold for |p_MEL - p_NV| in subset2b.")
    p.add_argument("--split_name", type=str, default="val", help="Prefix used in output filenames (e.g., val/train/test).")
    p.add_argument("--require_row_sum_one", action="store_true",
                   help="If set, assert GT one-hot rows sum to 1 (useful sanity check).")
    return p.parse_args()


def ensure_columns_exist(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def compute_gt_label(df_gt: pd.DataFrame, class_cols: List[str]) -> pd.DataFrame:
    """
    Adds gt_label by taking the argmax over one-hot GT columns.
    """
    ensure_columns_exist(df_gt, ["image"], "GT CSV (needs 'image')")
    ensure_columns_exist(df_gt, class_cols, "GT CSV one-hot class columns")
    df_gt = df_gt.copy()
    df_gt["gt_label"] = df_gt[class_cols].idxmax(axis=1)
    return df_gt


def ensure_pred_label(df_pred: pd.DataFrame, prob_cols: List[str], labels: List[str]) -> pd.DataFrame:
    """
    Ensures pred_label exists, computed from max probability if missing.
    """
    ensure_columns_exist(df_pred, ["image"], "Pred CSV (needs 'image')")
    ensure_columns_exist(df_pred, prob_cols, "Pred CSV probability columns")
    df_pred = df_pred.copy()

    if "pred_label" not in df_pred.columns:
        pred_idx = df_pred[prob_cols].values.argmax(axis=1)
        df_pred["pred_label"] = pd.Series(pred_idx).map({i: labels[i] for i in range(len(labels))})
    return df_pred


def add_top2_cols(df: pd.DataFrame, prob_cols: List[str], labels: List[str]) -> pd.DataFrame:
    """
    Adds top1 and top2 label columns based on probability columns.
    Vectorized (fast), avoids per-row Python sorting.
    """
    df = df.copy()
    probs = df[prob_cols].values  # shape (N,7)

    # argsort ascending; take last two columns for top2; reverse for top1/top2 order
    top2_idx = np.argsort(probs, axis=1)[:, -2:][:, ::-1]  # shape (N,2), [top1, top2]
    df["top1"] = [labels[i] for i in top2_idx[:, 0]]
    df["top2"] = [labels[i] for i in top2_idx[:, 1]]
    return df


def save_subset(df: pd.DataFrame, out_path: Path, key_cols: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path}  rows={len(df)}")
    # quick peek at key distribution if present
    for c in key_cols:
        if c in df.columns:
            vc = df[c].value_counts()
            print(f"  {c} counts:\n{vc.to_string()}")


def main() -> None:
    args = parse_args()

    gt_csv = Path(args.gt_csv)
    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir)

    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Pred CSV not found: {pred_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df_gt = pd.read_csv(gt_csv)
    df_pred = pd.read_csv(pred_csv)

    # Build labels
    gt_class_cols = GT_CLASS_COLS_DEFAULT
    prob_cols = PRED_PROB_COLS_DEFAULT
    pred_labels = PRED_LABELS_DEFAULT

    # Compute labels
    df_gt = compute_gt_label(df_gt, gt_class_cols)

    if args.require_row_sum_one:
        row_sums = df_gt[gt_class_cols].sum(axis=1)
        bad = (row_sums != 1).sum()
        if bad:
            raise ValueError(f"GT one-hot rows not summing to 1: bad_rows={bad}")
        print("[ok] GT one-hot row sums all equal 1")

    df_pred = ensure_pred_label(df_pred, prob_cols, pred_labels)

    # Merge
    merged = df_gt.merge(df_pred, on="image", how="inner")
    print(f"[info] gt_rows={len(df_gt)}  pred_rows={len(df_pred)}  merged_rows={len(merged)}")

    # Subset1: GT in MEL/NV
    subset1 = merged[merged["gt_label"].isin(["MEL", "NV"])].copy()
    path1 = out_dir / f"{args.split_name}_subset1_gt_MEL_or_NV.csv"
    save_subset(subset1, path1, key_cols=["gt_label", "pred_label"])

    # Subset2a: both probs above tau (min prob threshold)
    minprob = subset1[["p_MEL", "p_NV"]].min(axis=1)
    subset2a = subset1[minprob > args.tau].copy()
    path2a = out_dir / f"{args.split_name}_subset2a_gt_MEL_NV_minprob_gt_{args.tau:.2f}.csv"
    save_subset(subset2a, path2a, key_cols=["gt_label", "pred_label"])

    # Subset2b: close probabilities AND both above tau
    subset2b = subset1[
        (subset1[["p_MEL", "p_NV"]].min(axis=1) > args.tau) &
        ((subset1["p_MEL"] - subset1["p_NV"]).abs() < args.delta)
    ].copy()
    path2b = out_dir / f"{args.split_name}_subset2b_gt_MEL_NV_minprob_gt_{args.tau:.2f}_absdiff_lt_{args.delta:.2f}.csv"
    save_subset(subset2b, path2b, key_cols=["gt_label", "pred_label"])

    # Subset2c: top2 predictions are MEL/NV (any order)
    subset1_top2 = add_top2_cols(subset1, prob_cols, pred_labels)
    subset2c = subset1_top2[
        ((subset1_top2["top1"] == "MEL") & (subset1_top2["top2"] == "NV")) |
        ((subset1_top2["top1"] == "NV") & (subset1_top2["top2"] == "MEL"))
    ].copy()
    path2c = out_dir / f"{args.split_name}_subset2c_gt_MEL_NV_top2_is_MEL_NV.csv"
    save_subset(subset2c, path2c, key_cols=["gt_label", "pred_label", "top1", "top2"])

    # Subset3: flip mistakes MEL <-> NV
    subset3 = subset1[
        ((subset1["gt_label"] == "MEL") & (subset1["pred_label"] == "NV")) |
        ((subset1["gt_label"] == "NV") & (subset1["pred_label"] == "MEL"))
    ].copy()
    path3 = out_dir / f"{args.split_name}_subset3_flip_MEL_NV_wrong_to_other.csv"
    save_subset(subset3, path3, key_cols=["gt_label", "pred_label"])

    print(f"[done] subsets written to: {out_dir}")


if __name__ == "__main__":
    main()