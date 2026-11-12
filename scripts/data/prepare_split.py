"""Converted from notebooks/mel_nv/00_prepare_mel_nv_clean_csv.ipynb by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python scripts/data/prepare_split.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parents[2]
_CWD = _REPO / "notebooks" / "mel_nv"
_CWD.mkdir(parents=True, exist_ok=True)
os.chdir(_CWD)
sys.path.insert(0, str(_REPO))
try:
    from IPython.display import display
except ImportError:
    display = print


# ---- cell 1 ------------------------------------------------------------
from pathlib import Path
REPO_ROOT = Path("../../").resolve()
REPO_ROOT


# ---- cell 2 ------------------------------------------------------------
from pathlib import Path
import json
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

SEED = 42
USE_BALANCED_DEFAULT = False

# If this notebook is stored in data/HAM10000/mel_nv, this resolves to the repo root.
REPO_ROOT = Path("../..").resolve() if Path.cwd().name == "mel_nv" else Path("..").resolve()
REPO_ROOT = REPO_ROOT / "master-thesis"
HAM_ROOT = REPO_ROOT / "data" / "HAM10000"
OUT_DIR = HAM_ROOT / "mel_nv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAM_CSV = HAM_ROOT / "HAM10000.csv"
SEG_OVERLAP_CSV = HAM_ROOT / "ham_segmentation_overlap.csv"

print("REPO_ROOT:", REPO_ROOT)
print("HAM_CSV exists:", HAM_CSV.exists(), HAM_CSV)
print("SEG_OVERLAP_CSV exists:", SEG_OVERLAP_CSV.exists(), SEG_OVERLAP_CSV)
print("OUT_DIR:", OUT_DIR)


# ---- cell 4 ------------------------------------------------------------
ham_df = pd.read_csv(HAM_CSV, low_memory=False)
seg_df = pd.read_csv(SEG_OVERLAP_CSV, low_memory=False)

ham_df["image_id"] = ham_df["image_id"].astype(str)
seg_df["image_id"] = seg_df["image_id"].astype(str)

seg_keep = [c for c in ["image_id", "image_rel_path", "mask_rel_path", "mask_found"] if c in seg_df.columns]
merged = ham_df.merge(seg_df[seg_keep], on="image_id", how="left", suffixes=("", "_seg"))

if "image_rel_path_seg" in merged.columns:
    merged["image_rel_path"] = merged["image_rel_path_seg"].combine_first(merged.get("image_rel_path"))
if "mask_rel_path_seg" in merged.columns:
    merged["mask_rel_path"] = merged["mask_rel_path_seg"].combine_first(merged.get("mask_rel_path"))

if "image_rel_path" not in merged.columns or merged["image_rel_path"].isna().all():
    merged["image_rel_path"] = "images/" + merged["image"].astype(str)

if "mask_found" not in merged.columns:
    raise ValueError("mask_found column missing after merge. Check ham_segmentation_overlap.csv.")

merged["gt_label"] = merged["dx"].astype(str).str.upper()
merged["dx_norm"] = merged["dx"].astype(str).str.lower()

print("Merged:", merged.shape)
display(merged.head())


# ---- cell 6 ------------------------------------------------------------
CLASS2_TO_IDX = {"MEL": 0, "NV": 1}
IDX_TO_CLASS2 = {v: k for k, v in CLASS2_TO_IDX.items()}

mel_nv_imbalanced = merged[
    merged["gt_label"].isin(["MEL", "NV"])
    & merged["mask_found"].fillna(0).astype(int).eq(1)
].copy()

mel_nv_imbalanced["label_2class"] = mel_nv_imbalanced["gt_label"].map(CLASS2_TO_IDX).astype(int)
mel_nv_imbalanced["binary_label"] = mel_nv_imbalanced["label_2class"].astype(int)
mel_nv_imbalanced["cue_applied"] = False
mel_nv_imbalanced["cue_mask_rel_path"] = ""
mel_nv_imbalanced["cue_mode"] = "clean"

front_cols = [
    "lesion_id", "image_id", "image", "dx", "gt_label", "label", "label_2class", "binary_label",
    "split", "image_rel_path", "mask_rel_path", "mask_found",
    "cue_applied", "cue_mask_rel_path", "cue_mode",
]
front_cols = [c for c in front_cols if c in mel_nv_imbalanced.columns]
remaining_cols = [c for c in mel_nv_imbalanced.columns if c not in front_cols]
mel_nv_imbalanced = mel_nv_imbalanced[front_cols + remaining_cols].sort_values(["split", "gt_label", "image_id"]).reset_index(drop=True)

print("MEL/NV imbalanced:", mel_nv_imbalanced.shape)
display(mel_nv_imbalanced.groupby(["split", "gt_label"]).size().unstack(fill_value=0))
display(mel_nv_imbalanced.head())


# ---- cell 8 ------------------------------------------------------------
def make_balanced_by_split(df, class_col="gt_label", split_col="split", classes=("MEL", "NV"), seed=42):
    parts = []
    for split_name, split_df in df.groupby(split_col):
        counts = split_df[class_col].value_counts()
        if not all(cls in counts for cls in classes):
            print(f"[WARN] split={split_name}: missing one class, skipping")
            continue
        n_min = min(counts[cls] for cls in classes)
        for cls in classes:
            parts.append(split_df[split_df[class_col] == cls].sample(n=n_min, random_state=seed))
        print(f"split={split_name}: selected {n_min} {classes[0]} and {n_min} {classes[1]}")
    return pd.concat(parts, axis=0).sort_values([split_col, class_col, "image_id"]).reset_index(drop=True)

mel_nv_balanced = make_balanced_by_split(mel_nv_imbalanced, seed=SEED)

print("MEL/NV balanced:", mel_nv_balanced.shape)
display(mel_nv_balanced.groupby(["split", "gt_label"]).size().unstack(fill_value=0))


# ---- cell 10 ------------------------------------------------------------
IMBALANCED_CSV = OUT_DIR / "ham_mel_nv_clean_imbalanced.csv"
BALANCED_CSV = OUT_DIR / "ham_mel_nv_clean_balanced_by_split.csv"
DEFAULT_CSV = OUT_DIR / "ham_mel_nv_clean.csv"

mel_nv_imbalanced.to_csv(IMBALANCED_CSV, index=False)
mel_nv_balanced.to_csv(BALANCED_CSV, index=False)

selected_default = mel_nv_balanced if USE_BALANCED_DEFAULT else mel_nv_imbalanced
if DEFAULT_CSV.exists():
    prev = pd.read_csv(DEFAULT_CSV)
    old_ids = set(prev["image_id"].astype(str))
    new_ids = set(selected_default["image_id"].astype(str))
    assert old_ids == new_ids, (
        f"image_id set changed. added {len(new_ids - old_ids)}, removed {len(old_ids - new_ids)}"
    )
    old_split = prev.set_index("image_id")["split"].astype(str).sort_index()
    new_split = selected_default.set_index("image_id")["split"].astype(str).sort_index()
    assert (old_split == new_split).all(), "split assignment changed"
    print("guard passed: same images, same splits")
    print("new columns added:", sorted(set(selected_default.columns) - set(prev.columns)))
selected_default.to_csv(DEFAULT_CSV, index=False)

print("Saved imbalanced:", IMBALANCED_CSV)
print("Saved balanced:", BALANCED_CSV)
print("Saved default:", DEFAULT_CSV)
print("Default version:", "balanced_by_split" if USE_BALANCED_DEFAULT else "imbalanced_natural")
display(selected_default.groupby(["split", "gt_label"]).size().unstack(fill_value=0))


# ---- cell 12 ------------------------------------------------------------
def make_qualitative_csv(df, out_path, n_per_class=10, split="test", seed=42):
    test_df = df[df["split"].astype(str).str.lower().eq(split)].copy()
    parts = []
    for cls in ["MEL", "NV"]:
        cls_df = test_df[test_df["gt_label"].eq(cls)].copy()
        n = min(n_per_class, len(cls_df))
        parts.append(cls_df.sample(n=n, random_state=seed))
    out = pd.concat(parts, axis=0)
    out["order"] = out["gt_label"].map({"MEL": 0, "NV": 1})
    out = out.sort_values(["order", "image_id"]).drop(columns=["order"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    display(out.groupby(["split", "gt_label"]).size().unstack(fill_value=0))
    return out

N_PER_CLASS = 10
QUAL_CSV = OUT_DIR / f"ham_mel_nv_clean_qualitative_{N_PER_CLASS}_per_class_seed{SEED}.csv"
qual_df = make_qualitative_csv(mel_nv_imbalanced, QUAL_CSV, n_per_class=N_PER_CLASS, seed=SEED)

display(qual_df[["image_id", "gt_label", "split", "image_rel_path", "mask_rel_path", "binary_label"]].head(20))


# ---- cell 14 ------------------------------------------------------------
def resolve_from_ham_root(rel_or_abs):
    p = Path(str(rel_or_abs))
    if p.is_absolute():
        return p
    return (HAM_ROOT / p).resolve()

check_df = selected_default.copy()
check_df["image_exists"] = check_df["image_rel_path"].apply(lambda x: resolve_from_ham_root(x).exists())
check_df["mask_exists"] = check_df["mask_rel_path"].apply(lambda x: resolve_from_ham_root(x).exists())

display(check_df["image_exists"].value_counts(dropna=False))
display(check_df["mask_exists"].value_counts(dropna=False))

if not check_df["image_exists"].all():
    display(check_df.loc[~check_df["image_exists"], ["image_id", "image_rel_path"]].head())
    raise FileNotFoundError("Some image paths do not exist.")

if not check_df["mask_exists"].all():
    display(check_df.loc[~check_df["mask_exists"], ["image_id", "mask_rel_path"]].head())
    raise FileNotFoundError("Some mask paths do not exist.")

print("All selected image and mask paths exist.")


# ---- cell 16 ------------------------------------------------------------
metadata = {
    "seed": SEED,
    "task": "binary MEL vs NV clean fine-tuning",
    "class_mapping": CLASS2_TO_IDX,
    "use_balanced_default": USE_BALANCED_DEFAULT,
    "default_csv": str(DEFAULT_CSV.relative_to(REPO_ROOT)),
    "imbalanced_csv": str(IMBALANCED_CSV.relative_to(REPO_ROOT)),
    "balanced_csv": str(BALANCED_CSV.relative_to(REPO_ROOT)),
    "qualitative_csv": str(QUAL_CSV.relative_to(REPO_ROOT)),
    "recommendation": [
        "Start with imbalanced CSV and WeightedRandomSampler.",
        "Use NB_CLASSES=2.",
        "Training script automatically uses binary_label when nb_classes=2.",
        "Monitor recall_macro and balanced_accuracy.",
        "Train both POOLING=mean and POOLING=cls first.",
    ],
}
metadata_path = OUT_DIR / "metadata_mel_nv_clean.json"
metadata_path.write_text(json.dumps(metadata, indent=2))
print("Saved metadata:", metadata_path)
print(json.dumps(metadata, indent=2))
