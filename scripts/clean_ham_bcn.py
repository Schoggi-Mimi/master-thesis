#!/usr/bin/env python3

# python -m scripts.clean_ham_bcn \
#   --ham-dir data/HAM10k \
#   --bcn-dir data/BCN20k \
#   --out-dir data/processed \
#   --compute-hashes

# python -m scripts.clean_ham_bcn \
#   --ham-dir data/HAM10k \
#   --bcn-dir data/BCN20k \
#   --msk-dir data/MSKCC \
#   --out-dir data/processed \
#   --include-msk \
#   --compute-hashes

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PRIMARY_CLASSES = ["NV", "MEL", "BCC", "BKL", "AKIEC", "DF"]

PRIMARY_7_MAP = {
    "melanoma nos": "MEL",
    "nevus": "NV",
    "basal cell carcinoma": "BCC",
    "solar or actinic keratosis": "AKIEC",
    "pigmented benign keratosis": "BKL",
    "seborrheic keratosis": "BKL",
    "solar lentigo": "BKL",
    "dermatofibroma": "DF",
}

PRIMARY_EXCLUDE = {
    "squamous cell carcinoma": "exclude_scc",
    "squamous cell carcinoma nos": "exclude_scc",
    "scc": "exclude_scc",
    "melanoma metastasis": "exclude_melanoma_metastasis",
    "scar": "exclude_scar",
    "unknown": "exclude_unknown",
    "other": "exclude_other",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    out_dir = data_dir / "processed"

    p = argparse.ArgumentParser(description="Clean HAM/BCN(/MSK) metadata for the thesis setup.")
    p.add_argument("--ham-dir", type=str, default=str(data_dir / "HAM10k"))
    p.add_argument("--bcn-dir", type=str, default=str(data_dir / "BCN20k"))
    p.add_argument("--msk-dir", type=str, default=str(data_dir / "MSKCC"))
    p.add_argument("--out-dir", type=str, default=str(out_dir))
    p.add_argument("--include-msk", action="store_true", help="Include MSKCC if metadata.csv exists.")
    p.add_argument("--compute-hashes", action="store_true", help="Compute SHA256 hashes for exact duplicate audit.")
    p.add_argument("--keep-non-dermoscopic", action="store_true", help="Do not exclude non-dermoscopic images.")
    return p.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_lower(value):
    if pd.isna(value):
        return np.nan
    return normalize_text(value).lower()


def normalize_label(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def harmonize_primary_label(raw_label):
    norm = normalize_label(raw_label)
    if norm is None:
        return np.nan, "exclude_missing_diagnosis"
    if norm in PRIMARY_7_MAP:
        return PRIMARY_7_MAP[norm], "keep"
    if norm in PRIMARY_EXCLUDE:
        return np.nan, PRIMARY_EXCLUDE[norm]
    return np.nan, "exclude_unmapped_label"


def find_image_path(dataset_dir: Path, isic_id: str) -> Optional[Path]:
    jpg = dataset_dir / f"{isic_id}.jpg"
    png = dataset_dir / f"{isic_id}.png"
    if jpg.exists():
        return jpg
    if png.exists():
        return png
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_unique_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c in df.columns:
            rows.append(
                {
                    "column": c,
                    "n_unique": int(df[c].nunique(dropna=True)),
                    "n_missing": int(df[c].isna().sum()),
                }
            )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def print_basic_overview(df: pd.DataFrame, name: str) -> None:
    print("\n" + "=" * 80)
    print(f"{name} overview")
    print("=" * 80)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("\nnull counts:")
    print(df.isna().sum().sort_values(ascending=False).head(20).to_string())


def load_and_standardize_metadata(
    meta_path: Path,
    dataset_dir: Path,
    source_name: str,
    keep_non_dermoscopic: bool = False,
) -> pd.DataFrame:
    require_file(meta_path)
    df = pd.read_csv(meta_path)

    print(f"Loaded {source_name}: {df.shape}")
    print(f"{source_name} columns:")
    print(list(df.columns))

    df.columns = [normalize_text(c) for c in df.columns]

    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].map(normalize_text)

    if "isic_id" not in df.columns:
        raise KeyError(f"{source_name}: expected column 'isic_id' not found")
    if "lesion_id" not in df.columns:
        df["lesion_id"] = np.nan
    if "diagnosis_3" not in df.columns:
        raise KeyError(f"{source_name}: expected column 'diagnosis_3' not found")

    df["source_dataset"] = source_name
    df["image_path"] = df["isic_id"].map(lambda x: find_image_path(dataset_dir, x))
    df["image_exists"] = df["image_path"].notna()

    df["raw_label"] = df["diagnosis_3"]
    df["raw_label_norm"] = df["raw_label"].apply(normalize_label)

    mapped = df["raw_label"].apply(harmonize_primary_label)
    df["harmonized_label"] = mapped.apply(lambda x: x[0])
    df["primary_status"] = mapped.apply(lambda x: x[1])

    for c in ["concomitant_biopsy", "melanocytic"]:
        if c in df.columns:
            df[c] = df[c].replace({True: True, False: False, "True": True, "False": False})

    if "sex" in df.columns:
        sex_map = {"male": "male", "female": "female", "unknown": np.nan, "": np.nan}
        df["sex_clean"] = df["sex"].map(
            lambda x: sex_map.get(normalize_lower(x), normalize_lower(x) if pd.notna(x) else np.nan)
        )
    else:
        df["sex_clean"] = np.nan

    if "age_approx" in df.columns:
        df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
    else:
        df["age_approx"] = np.nan

    if "anatom_site_general" not in df.columns:
        df["anatom_site_general"] = np.nan

    if "image_type" in df.columns and not keep_non_dermoscopic:
        dermo_ok = df["image_type"].eq("dermoscopic")
    else:
        dermo_ok = True

    df["keep_for_primary_experiment"] = (
        df["primary_status"].eq("keep") & df["image_exists"] & dermo_ok
    )

    df["group_id"] = df["lesion_id"].fillna(df["isic_id"])

    cols_front = [
        "isic_id",
        "source_dataset",
        "image_path",
        "image_exists",
        "lesion_id",
        "group_id",
        "raw_label",
        "harmonized_label",
        "primary_status",
        "keep_for_primary_experiment",
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_confirm_type",
        "image_type",
        "melanocytic",
        "sex",
        "sex_clean",
        "age_approx",
        "anatom_site_general",
    ]
    cols_front = [c for c in cols_front if c in df.columns]
    cols_rest = [c for c in df.columns if c not in cols_front]
    return df[cols_front + cols_rest]


def lesion_summary(df: pd.DataFrame, name: str) -> None:
    lesion_counts = df.groupby("lesion_id", dropna=False)["isic_id"].nunique().rename("n_images")
    print(f"\n{name}: lesion-level summary")
    print("Rows:", len(df))
    print("Unique lesions:", int(df["lesion_id"].nunique(dropna=True)))
    print("Rows with missing lesion_id:", int(df["lesion_id"].isna().sum()))
    print("\nImages per lesion distribution:")
    print(lesion_counts.value_counts().sort_index().to_string())


def compute_hashes(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    hash_csv = out_dir / "image_hashes_sha256.csv"
    rows_to_hash = master.loc[
        master["image_exists"],
        ["isic_id", "source_dataset", "image_path", "lesion_id", "harmonized_label", "keep_for_primary_experiment"],
    ].copy()

    print("Number of images to hash:", len(rows_to_hash))
    records = []
    for i, row in enumerate(rows_to_hash.itertuples(index=False), start=1):
        img_path = Path(row.image_path)
        try:
            sha = sha256_file(img_path)
        except Exception as e:
            sha = np.nan
            print(f"[WARN] Could not hash {img_path}: {e}")

        records.append(
            {
                "isic_id": row.isic_id,
                "source_dataset": row.source_dataset,
                "image_path": str(img_path),
                "lesion_id": row.lesion_id,
                "harmonized_label": row.harmonized_label,
                "keep_for_primary_experiment": row.keep_for_primary_experiment,
                "sha256": sha,
            }
        )
        if i % 500 == 0 or i == len(rows_to_hash):
            print(f"Hashed {i}/{len(rows_to_hash)} images")

    hashes = pd.DataFrame(records)
    hashes.to_csv(hash_csv, index=False)
    print("Saved hashes to:", hash_csv)
    return hashes


def summarize_hash_duplicates(df: pd.DataFrame, hash_col: str = "sha256", name: str = "dataset") -> None:
    print("\n" + "=" * 80)
    print(f"{name}: SHA256 duplicate summary")
    print("=" * 80)

    dup_mask = df[hash_col].duplicated(keep=False) & df[hash_col].notna()
    dup_df = df.loc[dup_mask].copy()

    print("rows:", len(df))
    print("missing hashes:", int(df[hash_col].isna().sum()))
    print("unique hashes:", int(df[hash_col].nunique(dropna=True)))
    print("rows involved in duplicate hashes:", int(dup_mask.sum()))
    print("number of duplicate hash groups:", int(dup_df[hash_col].nunique()))

    if len(dup_df) == 0:
        print("No exact duplicate hashes found.")


def lesion_label_conflict_report(df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    kept = df[df["keep_for_primary_experiment"]].copy()
    lesion_summary_df = (
        kept.groupby("lesion_id")
        .agg(
            n_images=("isic_id", "size"),
            n_labels=("harmonized_label", lambda x: x.nunique(dropna=True)),
            labels=("harmonized_label", lambda x: sorted(pd.Series(x).dropna().unique())),
        )
        .reset_index()
    )
    conflicts = lesion_summary_df[lesion_summary_df["n_labels"] > 1].copy()

    print("\n" + "=" * 80)
    print(f"{dataset_name}: lesion label conflict report")
    print("=" * 80)
    print("kept rows:", len(kept))
    print("unique lesions:", int(kept["lesion_id"].nunique()))
    print("lesions with >1 kept label:", len(conflicts))
    if len(conflicts) == 0:
        print("No lesion-level label conflicts found.")
    return conflicts


def main() -> None:
    args = parse_args()

    ham_dir = Path(args.ham_dir)
    bcn_dir = Path(args.bcn_dir)
    msk_dir = Path(args.msk_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("HAM10000", ham_dir, ham_dir / "metadata.csv"),
        ("BCN20000", bcn_dir, bcn_dir / "metadata.csv"),
    ]

    if args.include_msk:
        msk_meta = msk_dir / "metadata.csv"
        if msk_meta.exists():
            datasets.append(("MSKCC", msk_dir, msk_meta))
        else:
            print(f"[WARN] --include-msk was set but no metadata found at {msk_meta}. Skipping MSKCC.")

    cleaned = []
    audit_cols = [
        "isic_id",
        "lesion_id",
        "raw_label",
        "harmonized_label",
        "primary_status",
        "diagnosis_confirm_type",
        "image_type",
        "image_exists",
        "sex",
        "sex_clean",
        "age_approx",
        "anatom_site_general",
    ]

    for source_name, dataset_dir, meta_path in datasets:
        df = load_and_standardize_metadata(
            meta_path=meta_path,
            dataset_dir=dataset_dir,
            source_name=source_name,
            keep_non_dermoscopic=args.keep_non_dermoscopic,
        )
        print_basic_overview(df, source_name)

        print(f"\n{source_name} unique/missing summary:")
        print(count_unique_summary(df, audit_cols).to_string(index=False))

        print(f"\n{source_name} image_type distribution:")
        if "image_type" in df.columns:
            print(df["image_type"].value_counts(dropna=False).to_string())
        else:
            print("No image_type column")

        print(f"\n{source_name} diagnosis_3 distribution:")
        print(df["diagnosis_3"].value_counts(dropna=False).head(30).to_string())

        print(f"\n{'='*80}\n{source_name} label cleaning report\n{'='*80}")
        print("\nRaw diagnosis_3 counts:")
        print(df["raw_label"].value_counts(dropna=False).to_string())
        print("\nPrimary status counts:")
        print(df["primary_status"].value_counts(dropna=False).to_string())
        print("\nHarmonized class counts (kept only):")
        print(df.loc[df["keep_for_primary_experiment"], "harmonized_label"].value_counts(dropna=False).to_string())

        lesion_summary(df, source_name)
        cleaned.append(df)

    master = pd.concat(cleaned, ignore_index=True, sort=False)
    print("\nMaster shape:", master.shape)
    print("\nMaster source distribution:")
    print(master["source_dataset"].value_counts(dropna=False).to_string())
    print("\nMaster keep_for_primary_experiment:")
    print(master["keep_for_primary_experiment"].value_counts(dropna=False).to_string())
    print("\nMaster harmonized label counts (kept only):")
    print(master.loc[master["keep_for_primary_experiment"], "harmonized_label"].value_counts(dropna=False).to_string())
    print("\nCross-tab: source x harmonized label (kept only)")
    print(
        pd.crosstab(
            master.loc[master["keep_for_primary_experiment"], "source_dataset"],
            master.loc[master["keep_for_primary_experiment"], "harmonized_label"],
            margins=True,
        ).to_string()
    )

    if args.compute_hashes:
        hashes = compute_hashes(master, out_dir)
        master = master.merge(hashes[["isic_id", "sha256"]], on="isic_id", how="left")
        sha_counts = master["sha256"].value_counts(dropna=True)
        master["exact_duplicate_group_size"] = master["sha256"].map(sha_counts).fillna(1).astype(int)
        master["has_exact_duplicate"] = master["exact_duplicate_group_size"] > 1

        summarize_hash_duplicates(master[master["source_dataset"] == "HAM10000"], name="HAM10000")
        summarize_hash_duplicates(master[master["source_dataset"] == "BCN20000"], name="BCN20000")
        if "MSKCC" in set(master["source_dataset"]):
            summarize_hash_duplicates(master[master["source_dataset"] == "MSKCC"], name="MSKCC")
        summarize_hash_duplicates(master, name="MASTER")

    for ds_name in master["source_dataset"].dropna().unique():
        lesion_label_conflict_report(master[master["source_dataset"] == ds_name], ds_name)
    lesion_label_conflict_report(master, "MASTER")

    per_dataset = {}
    for ds_name in master["source_dataset"].dropna().unique():
        out_path = out_dir / f"{ds_name.lower()}_clean_metadata.csv"
        subset = master.query("source_dataset == @ds_name").copy()
        subset.to_csv(out_path, index=False)
        per_dataset[ds_name] = str(out_path)

    primary_clean = master.loc[master["keep_for_primary_experiment"]].copy()
    master_out = out_dir / "master_clean_metadata.csv"
    primary_out = out_dir / f"primary{len(PRIMARY_CLASSES)}_keep_only_metadata.csv"

    master.to_csv(master_out, index=False)
    primary_clean.to_csv(primary_out, index=False)

    summary = {
        "datasets": sorted(master["source_dataset"].dropna().unique().tolist()),
        "master_shape": list(master.shape),
        "primary_shape": list(primary_clean.shape),
        "primary_label_counts": {
            k: int(v) for k, v in primary_clean["harmonized_label"].value_counts().to_dict().items()
        },
        "per_dataset_outputs": per_dataset,
        "master_output": str(master_out),
        "primary_output": str(primary_out),
        "compute_hashes": bool(args.compute_hashes),
    }

    summary_path = out_dir / "clean_metadata_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    for name, path in per_dataset.items():
        print(f" - {name}: {path}")
    print(" -", master_out)
    print(" -", primary_out)
    print(" -", summary_path)


if __name__ == "__main__":
    main()
