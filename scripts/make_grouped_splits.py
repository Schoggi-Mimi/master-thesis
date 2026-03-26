#!/usr/bin/env python3

# python -m scripts.make_grouped_splits \
#   --metadata-csv data/processed/primary6_keep_only_metadata.csv \
#   --out-dir data/processed/splits \
#   --seed 42

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PRIMARY_CLASSES = ["NV", "MEL", "BCC", "BKL", "AKIEC", "DF"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Create lesion-grouped splits for HAM base training and BCN fine-tuning."
    )
    p.add_argument(
        "--metadata-csv",
        type=str,
        default=str(repo_root / "data" / "processed" / "primary6_keep_only_metadata.csv"),
        help="Cleaned metadata CSV produced by clean_ham_bcn.py",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(repo_root / "data" / "processed" / "splits"),
        help="Output directory for split CSV files",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--group-col",
        type=str,
        default="lesion_id",
        help="Grouping column. Defaults to lesion_id and falls back to group_id then isic_id.",
    )
    p.add_argument("--label-col", type=str, default="harmonized_label")
    p.add_argument("--source-col", type=str, default="source_dataset")
    p.add_argument("--ham-train-frac", type=float, default=0.80)
    p.add_argument("--bcn-train-frac", type=float, default=0.60)
    p.add_argument("--bcn-val-frac", type=float, default=0.20)
    p.add_argument("--bcn-test-frac", type=float, default=0.20)
    return p.parse_args()


def resolve_group_column(df: pd.DataFrame, preferred_group_col: str) -> Tuple[pd.DataFrame, str]:
    df = df.copy()

    candidate_cols: List[str] = []
    if preferred_group_col:
        candidate_cols.append(preferred_group_col)
    for fallback in ["lesion_id", "group_id", "isic_id"]:
        if fallback not in candidate_cols:
            candidate_cols.append(fallback)

    used_group_col = None
    group_values = None
    for col in candidate_cols:
        if col in df.columns and df[col].notna().any():
            used_group_col = col
            group_values = df[col]
            break

    if used_group_col is None or group_values is None:
        raise ValueError("Could not resolve a valid grouping column.")

    df["split_group_id"] = group_values.fillna(df["isic_id"]).astype(str)
    return df, used_group_col


def normalize_keep_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "keep_for_primary_experiment" in df.columns:
        col = df["keep_for_primary_experiment"]
        if col.dtype != bool:
            df["keep_for_primary_experiment"] = (
                col.astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
            )
    return df


def normalize_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "image_path" not in df.columns:
        return df

    def _fix_path(x: object) -> object:
        if pd.isna(x):
            return x
        s = str(x).strip()
        if s.startswith("../data/"):
            return s
        if s.startswith("data/"):
            return "../" + s
        return s

    df["image_path"] = df["image_path"].map(_fix_path)
    return df


def build_group_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for gid, g in df.groupby("split_group_id", sort=False):
        labels = g[label_col].dropna().unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"Group {gid} has multiple labels: {labels}")

        rows.append(
            {
                "split_group_id": gid,
                "label": labels[0],
                "n_images": int(len(g)),
            }
        )

    group_df = pd.DataFrame(rows)
    if group_df.empty:
        raise ValueError("No groups were created. Check the input metadata.")
    return group_df


def compute_split_group_counts(n_groups: int, target_fracs: Sequence[float]) -> List[int]:
    if n_groups <= 0:
        return [0 for _ in target_fracs]

    raw = np.array(target_fracs, dtype=float) * float(n_groups)
    counts = np.floor(raw).astype(int)
    remainder = int(n_groups - counts.sum())

    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1

    return counts.tolist()


def stratified_group_split_assignments(
    group_df: pd.DataFrame,
    target_fracs: Sequence[float],
    seed: int,
) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    assignments: Dict[str, int] = {}
    n_splits = len(target_fracs)

    for label, label_df in group_df.groupby("label", sort=True):
        label_df = label_df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000_000))).reset_index(drop=True)
        counts = compute_split_group_counts(len(label_df), target_fracs)

        start = 0
        for split_idx in range(n_splits):
            end = start + counts[split_idx]
            part = label_df.iloc[start:end]
            for gid in part["split_group_id"].astype(str).tolist():
                assignments[gid] = split_idx
            start = end

    missing = set(group_df["split_group_id"].astype(str)) - set(assignments.keys())
    if missing:
        raise ValueError(f"Missing assignments for {len(missing)} groups. First few: {sorted(list(missing))[:10]}")

    return assignments


def apply_assignments(
    df: pd.DataFrame,
    assignments: Dict[str, int],
    split_names: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()
    missing = sorted(set(df["split_group_id"].astype(str)) - set(assignments.keys()))
    if missing:
        raise ValueError(f"Missing assignments for {len(missing)} groups. First few: {missing[:10]}")
    df["split"] = df["split_group_id"].astype(str).map(lambda x: split_names[assignments[x]])
    return df


def print_split_report(name: str, df: pd.DataFrame, label_col: str) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    for split_name, g in df.groupby("split", sort=False):
        print(f"{split_name} {g.shape}")
        print(g[label_col].value_counts().reindex(PRIMARY_CLASSES, fill_value=0))
        print("-" * 80)


def check_no_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, col: str, name_a: str, name_b: str) -> int:
    if col not in df_a.columns or col not in df_b.columns:
        return 0
    a = set(df_a[col].dropna().astype(str).unique())
    b = set(df_b[col].dropna().astype(str).unique())
    overlap = a & b
    print(f"Overlap check for '{col}' between {name_a} and {name_b}: {len(overlap)}")
    return len(overlap)


def summarize_split(df: pd.DataFrame, split_name: str, label_col: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "split": split_name,
        "n_rows": int(len(df)),
        "n_images": int(df["isic_id"].nunique(dropna=True)),
        "n_groups": int(df["split_group_id"].nunique(dropna=True)),
    }
    for cls in PRIMARY_CLASSES:
        row[f"class_{cls}"] = int((df[label_col] == cls).sum())
    return row


def save_split(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  shape={df.shape}")


def main() -> None:
    args = parse_args()

    if not np.isclose(args.bcn_train_frac + args.bcn_val_frac + args.bcn_test_frac, 1.0):
        raise ValueError("BCN fractions must sum to 1.0")

    metadata_csv = Path(args.metadata_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv, low_memory=False)
    df = normalize_keep_flag(df)
    df = normalize_image_paths(df)
    df, used_group_col = resolve_group_column(df, args.group_col)

    if "keep_for_primary_experiment" in df.columns:
        df = df[df["keep_for_primary_experiment"]].copy()

    df = df[df[args.label_col].isin(PRIMARY_CLASSES)].copy()

    ham = df[df[args.source_col] == "HAM10000"].copy()
    bcn = df[df[args.source_col] == "BCN20000"].copy()

    if len(ham) == 0 or len(bcn) == 0:
        raise ValueError("Expected both HAM10000 and BCN20000 rows.")

    ham_groups = build_group_table(ham, args.label_col)
    ham_assign = stratified_group_split_assignments(
        ham_groups,
        target_fracs=[args.ham_train_frac, 1.0 - args.ham_train_frac],
        seed=args.seed,
    )
    ham = apply_assignments(ham, ham_assign, ["train_base", "val_base"])

    bcn_groups = build_group_table(bcn, args.label_col)
    bcn_assign = stratified_group_split_assignments(
        bcn_groups,
        target_fracs=[args.bcn_train_frac, args.bcn_val_frac, args.bcn_test_frac],
        seed=args.seed + 1,
    )
    bcn = apply_assignments(bcn, bcn_assign, ["train_bcn_ft", "val_bcn_ft", "test_bcn"])

    print_split_report("HAM lesion-grouped split report", ham, args.label_col)
    print_split_report("BCN lesion-grouped split report", bcn, args.label_col)

    train_base = ham[ham["split"] == "train_base"].copy()
    val_base = ham[ham["split"] == "val_base"].copy()
    train_bcn_ft = bcn[bcn["split"] == "train_bcn_ft"].copy()
    val_bcn_ft = bcn[bcn["split"] == "val_bcn_ft"].copy()
    test_bcn = bcn[bcn["split"] == "test_bcn"].copy()

    print("\n" + "=" * 80)
    print("LEAKAGE CHECKS")
    print("=" * 80)
    print(f"Requested group column: {args.group_col} | used group column: {used_group_col}")
    check_no_overlap(train_base, val_base, "split_group_id", "train_base", "val_base")
    check_no_overlap(train_base, val_base, "isic_id", "train_base", "val_base")
    check_no_overlap(train_bcn_ft, val_bcn_ft, "split_group_id", "train_bcn_ft", "val_bcn_ft")
    check_no_overlap(train_bcn_ft, test_bcn, "split_group_id", "train_bcn_ft", "test_bcn")
    check_no_overlap(val_bcn_ft, test_bcn, "split_group_id", "val_bcn_ft", "test_bcn")
    check_no_overlap(train_bcn_ft, val_bcn_ft, "isic_id", "train_bcn_ft", "val_bcn_ft")
    check_no_overlap(train_bcn_ft, test_bcn, "isic_id", "train_bcn_ft", "test_bcn")
    check_no_overlap(val_bcn_ft, test_bcn, "isic_id", "val_bcn_ft", "test_bcn")

    save_split(train_base, out_dir / "train_base.csv")
    save_split(val_base, out_dir / "val_base.csv")
    save_split(train_bcn_ft, out_dir / "train_bcn_ft.csv")
    save_split(val_bcn_ft, out_dir / "val_bcn_ft.csv")
    save_split(test_bcn, out_dir / "test_bcn.csv")

    overview_rows = [
        summarize_split(train_base, "train_base", args.label_col),
        summarize_split(val_base, "val_base", args.label_col),
        summarize_split(train_bcn_ft, "train_bcn_ft", args.label_col),
        summarize_split(val_bcn_ft, "val_bcn_ft", args.label_col),
        summarize_split(test_bcn, "test_bcn", args.label_col),
    ]
    overview_df = pd.DataFrame(overview_rows)
    overview_csv = out_dir / "split_overview.csv"
    overview_df.to_csv(overview_csv, index=False)
    print("Saved:", overview_csv)

    summary = {
        "metadata_csv": str(metadata_csv),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "ham_train_frac": args.ham_train_frac,
        "ham_val_frac": 1.0 - args.ham_train_frac,
        "bcn_train_frac": args.bcn_train_frac,
        "bcn_val_frac": args.bcn_val_frac,
        "bcn_test_frac": args.bcn_test_frac,
        "group_col_requested": args.group_col,
        "group_col_used": used_group_col,
        "counts": {
            "train_base": int(len(train_base)),
            "val_base": int(len(val_base)),
            "train_bcn_ft": int(len(train_bcn_ft)),
            "val_bcn_ft": int(len(val_bcn_ft)),
            "test_bcn": int(len(test_bcn)),
        },
        "n_groups": {
            "train_base": int(train_base["split_group_id"].nunique()),
            "val_base": int(val_base["split_group_id"].nunique()),
            "train_bcn_ft": int(train_bcn_ft["split_group_id"].nunique()),
            "val_bcn_ft": int(val_bcn_ft["split_group_id"].nunique()),
            "test_bcn": int(test_bcn["split_group_id"].nunique()),
        },
    }
    summary_json = out_dir / "split_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", summary_json)


if __name__ == "__main__":
    main()
