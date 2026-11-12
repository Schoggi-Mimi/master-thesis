"""Converted from notebooks/mel_nv/12a_annotation_qc.ipynb by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python scripts/data/annotation_qc.py
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


# ---- cell 0 ------------------------------------------------------------
# =============================================================================
# 12a — Brush annotation QC
#
# Validates clinician brush masks before any overlap analysis.
# Nothing downstream is trustworthy until this notebook is clean.
#
# GEOMETRY. Set once here. Everything follows.
#   squash224  Resize((224,224)). Matches SimplePairedDermTransform used by
#              HA fine tuning. Whole frame kept, aspect distorted.
#   crop224    Resize(256) + CenterCrop(224). PanDerm eval default. Discards
#              roughly 42 percent of a 600x450 frame.
#
# Writes
#   results/annotation_qc/<geometry>/annotation_qc_manifest.csv
#   results/annotation_qc/<geometry>/curated_case_lookup.csv
#   results/annotation_qc/<geometry>/label_pair_overlap.csv
#   results/annotation_qc/<geometry>/union_masks_summary.csv
# =============================================================================
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def find_repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "scripts" / "generate_finer_cam_panderm.py").exists():
            return p.resolve()
    raise FileNotFoundError("Could not find repo root.")


REPO = find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.cam_eval_utils import (
    transform_mask, mask_to_cam_grid_geom, transform_rgb, patch_footprint,
    canonical_label, label_group,
    RESIZE_SHORT, CROP_SIZE, CAM_GRID,
)

GEOMETRY = "squash224"          # must match training and every later notebook

ANN_DIR   = REPO / "data" / "brush_annotation_masks"
ANN_CSV   = ANN_DIR / "brush_annotation_summary.csv"
HAM_ROOT  = REPO / "data" / "HAM10000"
CLEAN_CSV = HAM_ROOT / "mel_nv" / "ham_mel_nv_clean.csv"

OUT_DIR = REPO / "results" / "annotation_qc" / GEOMETRY
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PATCH_50  = 1       # at least one patch at least half covered at 14x14
MIN_CROP_KEEP = 0.80    # fraction surviving the transform. always 1.0 under squash

print("REPO:", REPO)
for p in [ANN_DIR, ANN_CSV, CLEAN_CSV]:
    print(f"{'OK      ' if p.exists() else 'MISSING '} {p}")

fy, fx = patch_footprint(GEOMETRY)
print(f"\ngeometry {GEOMETRY}   grid {CAM_GRID}x{CAM_GRID}")
print(f"one patch covers {fy:.1f} x {fx:.1f} original pixels, height x width")
print(f"out dir {OUT_DIR.relative_to(REPO)}")


# ---- cell 1 ------------------------------------------------------------
# =============================================================================
# Show both geometries side by side so the choice is documented, not assumed.
#
# EXPECT squash224 to retain 1.000 of the frame and crop224 about 0.577.
# =============================================================================
sys.path.insert(0, str(REPO / "external" / "PanDerm" / "classification"))
from models.builder import get_eval_transforms

print("crop224, PanDerm eval default:")
print(get_eval_transforms(which_img_norm="imagenet",
                          img_resize=256, center_crop=True))

print("\nsquash224, SimplePairedDermTransform used in HA training:")
print("  TF.resize(img, (224, 224), bilinear)  then normalise")

H, W = 450, 600
s = RESIZE_SHORT / min(H, W)
nh, nw = round(H * s), round(W * s)
top, left = (nh - CROP_SIZE) // 2, (nw - CROP_SIZE) // 2
area = (1 - 2 * left / nw) * (1 - 2 * top / nh)

print(f"\nfor a {H}x{W} image")
print(f"  crop224    resize to {nh}x{nw}, discard {top/s:.0f} px top and bottom, "
      f"{left/s:.0f} px each side. area retained {area:.3f}")
print(f"  squash224  resize to 224x224 directly. area retained 1.000")
print(f"\nACTIVE: {GEOMETRY}")


# ---- cell 2 ------------------------------------------------------------
ann = pd.read_csv(ANN_CSV)
ann["Image_ID"]   = ann["Image_ID"].astype(str).str.strip()
ann["Label_ID"]   = ann["Label_ID"].astype(str).str.strip()
ann["Label_Name"] = ann["Label_Name"].astype(str).str.strip()
ann["label"] = [canonical_label(a, b)
                for a, b in zip(ann["Label_ID"], ann["Label_Name"])]

print(f"rows {len(ann)}   images {ann['Image_ID'].nunique()}   "
      f"labels {ann['label'].nunique()}")
print()
print(ann[["Image_ID", "label", "Stroke_Count"]].head(6).to_string(index=False))
print()
print("annotations per label:")
print(ann["label"].value_counts().to_string())
print()
print("annotations per image:")
print(ann.groupby("Image_ID").size().describe().round(2).to_string())
print()
print("stroke count:")
print(ann["Stroke_Count"].describe().round(2).to_string())

zero_strokes = ann[ann["Stroke_Count"] == 0]
print(f"\nzero stroke rows: {len(zero_strokes)}")
if len(zero_strokes):
    print(zero_strokes[["Image_ID", "label"]].to_string(index=False))


# ---- cell 3 ------------------------------------------------------------
def resolve_mask(fname):
    direct = ANN_DIR / str(fname)
    if direct.exists():
        return direct
    hits = list(ANN_DIR.rglob(str(fname)))
    return hits[0] if hits else None


ann["mask_path"] = ann["Mask_Filename"].map(resolve_mask)
ann["mask_found"] = ann["mask_path"].notna()

print(f"resolved {int(ann['mask_found'].sum())} / {len(ann)}")
missing = ann[~ann["mask_found"]]
if len(missing):
    print("\nUNRESOLVED, stop and fix before continuing")
    print(missing[["Image_ID", "label", "Mask_Filename"]].to_string(index=False))
else:
    print("all mask files resolved")

# also check for orphan PNGs not listed in the CSV
listed = set(ann["Mask_Filename"].astype(str))
on_disk = {p.name for p in ANN_DIR.rglob("*.png")}
orphans = sorted(on_disk - listed)
print(f"\nPNGs on disk not in CSV: {len(orphans)}")
if orphans:
    print(orphans[:10])


# ---- cell 4 ------------------------------------------------------------
# =============================================================================
# The 08b export manifest was never written, since those cells were
# commented out. Recover case types by searching for the subset CSVs.
# =============================================================================
SUBSET_PATTERNS = {
    "TP_MEL":                "*typical_mel*.csv",
    "TN_NV":                 "*typical_nv*.csv",
    "FN_MEL":                "*fn_mel*.csv",
    "FP_MEL":                "*fp_mel*.csv",
    "DIFFICULT_CORRECT_MEL": "*difficult_correct_mel*.csv",
}

search_root = REPO / "outputs"
frames = []
for case_label, pattern in SUBSET_PATTERNS.items():
    hits = [p for p in search_root.rglob(pattern) if p.is_file()]
    if not hits:
        print(f"  not found: {case_label}")
        continue
    p = sorted(hits, key=lambda x: len(str(x)))[0]
    d = pd.read_csv(p)
    if "image_id" not in d.columns:
        print(f"  no image_id column in {p.name}")
        continue
    d["image_id"] = d["image_id"].astype(str).str.strip()
    d["case_type"] = case_label
    keep = [c for c in ["image_id", "gt_label", "selection_reason", "case_type"]
            if c in d.columns]
    frames.append(d[keep])
    print(f"  {case_label:<24} {len(d):>3} rows   {p.relative_to(REPO)}")

if frames:
    curated = pd.concat(frames, ignore_index=True)
    dup_across = curated[curated.duplicated("image_id", keep=False)]
    if len(dup_across):
        print(f"\nimages in more than one subset: "
              f"{dup_across['image_id'].nunique()}")
        print(dup_across.sort_values("image_id")
              [["image_id", "case_type"]].to_string(index=False))
    curated = curated.drop_duplicates("image_id", keep="first")
    curated.to_csv(OUT_DIR / "curated_case_lookup.csv", index=False)
    print(f"\ncurated lookup rows: {len(curated)}")
else:
    curated = None
    print("\nNo subset CSVs found. Case types will be missing.")


# ---- cell 5 ------------------------------------------------------------
# =============================================================================
# One loop, every measurement. Avoids merge key bugs entirely.
# All mask geometry goes through transform_mask and mask_to_cam_grid_geom.
# =============================================================================
clean = pd.read_csv(CLEAN_CSV).drop_duplicates("image_id")
clean["image_id"] = clean["image_id"].astype(str).str.strip()
img_lookup    = clean.set_index("image_id")["image_rel_path"].to_dict()
lesion_lookup = clean.set_index("image_id")["mask_rel_path"].to_dict()

_lesion_cache = {}


def load_lesion(image_id, shape):
    """Lesion mask at ORIGINAL resolution. Geometry applied later."""
    if image_id in _lesion_cache:
        return _lesion_cache[image_id]
    rel = lesion_lookup.get(image_id)
    out = None
    if rel is not None:
        p = (HAM_ROOT / str(rel)).resolve()
        if p.exists():
            m = Image.open(p).convert("L").resize((shape[1], shape[0]),
                                                  Image.NEAREST)
            out = np.array(m) > 127
    _lesion_cache[image_id] = out
    return out


records = []
for _, r in ann.iterrows():
    rec = {
        "Image_ID":      r["Image_ID"],
        "Annotation_ID": r["Annotation_ID"],
        "Stroke_Count":  r["Stroke_Count"],
        "mask_found":    bool(r["mask_found"]),
        "mask_path":     str(r["mask_path"]) if r["mask_found"] else "",
        "Label_ID":      r["Label_ID"],
        "Label_Name":    r["Label_Name"],
        "label":         r["label"],
        "geometry":      GEOMETRY,
    }

    if not r["mask_found"]:
        records.append(rec)
        continue

    raw = np.array(Image.open(r["mask_path"]).convert("L"))
    m = raw > 127
    rec.update({
        "mask_h": int(raw.shape[0]),
        "mask_w": int(raw.shape[1]),
        "n_grey_levels": int(len(np.unique(raw))),
        "px_positive": int(m.sum()),
        "area_frac": float(m.mean()),
    })

    if m.sum() == 0:
        records.append(rec)
        continue

    crop, kept = transform_mask(m, GEOMETRY)
    grid = mask_to_cam_grid_geom(m, GEOMETRY)

    lesion = load_lesion(r["Image_ID"], m.shape)
    inside = float((m & lesion).sum() / m.sum()) if lesion is not None else np.nan

    rec.update({
        "frac_kept_after_crop": kept,
        "px_after_crop":        int(crop.sum()),
        "area_frac_cropped":    float(crop.mean()),
        "grid_sum":             float(grid.sum()),
        "grid_max":             float(grid.max()),
        "n_patch_50":           int((grid >= 0.5).sum()),
        "n_patch_any":          int((grid > 0.01).sum()),
        "grid_area_frac":       float((grid >= 0.5).mean()),
        "frac_inside_lesion":   inside,
    })
    records.append(rec)

qc = pd.DataFrame(records)
qc["label_group"] = qc["label"].map(label_group)

print(f"measured {len(qc)} annotations under {GEOMETRY}")
print()
print("mask resolutions:")
print(qc.groupby(["mask_h", "mask_w"]).size().to_string())
print()
print("grey levels per mask:")
print(qc["n_grey_levels"].value_counts().to_string())
print()
print("label groups:")
print(qc["label_group"].value_counts().to_string())

print("\ncanonical label mapping:")
print(qc[["Label_Name", "label", "label_group"]]
      .drop_duplicates()
      .sort_values(["label_group", "label"])
      .to_string(index=False))

n_unknown = int((qc.label_group == "UNKNOWN").sum())
print(f"\nUNKNOWN: {n_unknown}")
assert n_unknown == 0, "unmapped labels remain"


# ---- cell 6 ------------------------------------------------------------
# =============================================================================
# Fraction of the annotation surviving the geometric transform.
#
# Under squash224 nothing is discarded, so this is 1.000 everywhere and the
# check is vacuous. It is kept so the manifest schema matches crop224 runs.
# Under crop224 anything below 0.80 was drawn near the horizontal edge.
# =============================================================================
if GEOMETRY == "squash224":
    print("squash224 keeps the whole frame. Expect 1.000 everywhere.")
print("fraction surviving the transform, by label:")
print(qc.groupby("label")["frac_kept_after_crop"]
      .agg(["count", "median", "min"]).round(3).to_string())

lost = qc[qc["frac_kept_after_crop"] < MIN_CROP_KEEP]
print(f"\nannotations below {MIN_CROP_KEEP:.2f}: {len(lost)}")
if len(lost):
    print(lost[["Image_ID", "label", "frac_kept_after_crop", "px_positive"]]
          .sort_values("frac_kept_after_crop").round(3).to_string(index=False))


# ---- cell 7 ------------------------------------------------------------
# =============================================================================
# Sensitivity of the pass count to the transform threshold.
# Under squash224 all three rows are identical by construction.
# =============================================================================
for thr in [0.80, 0.75, 0.70]:
    sub = qc[
        qc.mask_found
        & (qc.px_positive.fillna(0) > 0)
        & (qc.n_patch_50.fillna(0) >= MIN_PATCH_50)
        & (qc.frac_kept_after_crop.fillna(0) >= thr)
    ]
    print(f"threshold {thr:.2f}  annotations {len(sub):3d}  "
          f"MEL {int((sub.label_group == 'MEL').sum()):3d}  "
          f"images {sub.Image_ID.nunique()}")


# ---- cell 8 ------------------------------------------------------------
# =============================================================================
# Survival at CAM resolution. This is the binding constraint under squash224.
#
# One patch covers more original area under squash than under crop, roughly
# 43 px horizontally versus 28. Small structures therefore span fewer patches
# and some may fall below the one half covered patch threshold.
#
# WATCH any label whose median n_patch_50 reaches 0.
# =============================================================================
fy, fx = patch_footprint(GEOMETRY)
print(f"{GEOMETRY}: one patch covers {fy:.1f} x {fx:.1f} original pixels\n")

print("effective coverage at CAM resolution, by label:")
print(qc.groupby("label")[["grid_sum", "n_patch_50", "n_patch_any"]]
      .agg(["median", "min", "max"]).round(2).to_string())

dead = qc[(qc["px_positive"] > 0) & (qc["n_patch_50"] == 0)]
print(f"\nannotations with zero half covered patches: {len(dead)}")
if len(dead):
    print(dead[["Image_ID", "label", "px_positive", "grid_sum", "grid_max"]]
          .round(3).to_string(index=False))

print("\narea fraction at grid resolution, by group. this is the chance baseline:")
print(qc.groupby("label_group")["grid_area_frac"]
      .agg(["count", "median", "max"]).round(3).to_string())


# ---- cell 9 ------------------------------------------------------------
# =============================================================================
# EXPECT: clinical feature labels near 1.00, they sit inside the lesion.
#         artifact_ignore LOW, it marks hair, rulers, bubbles outside.
# WATCH:  a clinical label well below 0.9 suggests a registration problem.
# =============================================================================
print("fraction of annotation inside the Tschandl lesion mask:")
print(qc.groupby("label")["frac_inside_lesion"]
      .agg(["count", "median", "min"]).round(3).to_string())
print()
print("by group:")
print(qc.groupby("label_group")["frac_inside_lesion"]
      .agg(["count", "median"]).round(3).to_string())

odd = qc[(qc["label_group"].isin(["MEL", "NV", "COARSE"])) &
         (qc["frac_inside_lesion"] < 0.7)]
print(f"\nclinical labels less than 70 percent inside the lesion: {len(odd)}")
if len(odd):
    print(odd[["Image_ID", "label", "frac_inside_lesion"]]
          .round(3).to_string(index=False))


# ---- cell 10 ------------------------------------------------------------
dup = (qc.groupby(["Image_ID", "label"]).size()
         .reset_index(name="n").query("n > 1"))
print(f"image-label pairs with multiple annotations: {len(dup)}")
if len(dup):
    print(dup.to_string(index=False))
    print("These get merged by union in 12c.")

rows = []
for img, grp in qc[qc.mask_found & (qc.px_positive > 0)].groupby("Image_ID"):
    recs = grp.to_dict("records")
    masks = {i: np.array(Image.open(r["mask_path"]).convert("L")) > 127
             for i, r in enumerate(recs)}
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            u = (masks[i] | masks[j]).sum()
            if u == 0:
                continue
            rows.append({
                "Image_ID": img,
                "label_a": recs[i]["label"],
                "label_b": recs[j]["label"],
                "iou": float((masks[i] & masks[j]).sum() / u),
            })

ov = pd.DataFrame(rows)
if len(ov):
    print(f"\nlabel pairs compared: {len(ov)}")
    print(f"median IoU: {ov['iou'].median():.3f}")
    print("\nhighest overlaps:")
    print(ov.nlargest(10, "iou").round(3).to_string(index=False))
    ov.to_csv(OUT_DIR / "label_pair_overlap.csv", index=False)


# ---- cell 11 ------------------------------------------------------------
if curated is not None:
    sent = set(curated["image_id"])
    got  = set(qc["Image_ID"])

    print(f"curated and sent  : {len(sent)}")
    print(f"returned annotated: {len(got)}")
    print(f"not returned      : {len(sent - got)}")
    if sent - got:
        print("  ", sorted(sent - got))
    if got - sent:
        print(f"annotated but not in any subset: {sorted(got - sent)}")

    cov = curated[curated["image_id"].isin(got)]
    print("\nannotated by case type:")
    print(cov["case_type"].value_counts().to_string())
    if "gt_label" in cov.columns:
        print("\nannotated by ground truth:")
        print(cov["gt_label"].value_counts().to_string())

    qc["case_type"] = qc["Image_ID"].map(
        curated.set_index("image_id")["case_type"])
    if "gt_label" in curated.columns:
        qc["gt_label"] = qc["Image_ID"].map(
            curated.set_index("image_id")["gt_label"])
else:
    qc["case_type"] = np.nan
    qc["gt_label"] = qc["Image_ID"].map(
        clean.set_index("image_id")["gt_label"]
        if "gt_label" in clean.columns else {})
    print("using gt_label from the clean CSV, case_type unavailable")

print("\nMEL images annotated:",
      qc.loc[qc.gt_label == "MEL", "Image_ID"].nunique()
      if "gt_label" in qc.columns else "unknown")


# ---- cell 12 ------------------------------------------------------------
# =============================================================================
# FIGURE. One row per image, rendered in the SAME geometry the model sees.
#
# Column 0 is the transformed image with the lesion outline in white.
# Remaining columns add one annotation each, red fill, green outline.
#
# EXPECT red regions on the named feature, inside the white lesion outline.
# WATCH  a mask that looks shifted means image and mask geometry disagree.
# =============================================================================
def show_annotations(image_id, max_cols=6):
    rel = img_lookup.get(image_id)
    img = np.array(Image.open((HAM_ROOT / str(rel)).resolve()).convert("RGB"))
    rgb = transform_rgb(img, GEOMETRY)

    les = load_lesion(image_id, img.shape[:2])
    les_t = transform_mask(les, GEOMETRY)[0] if les is not None else None

    sub = qc[(qc.Image_ID == image_id) & qc.mask_found & (qc.px_positive > 0)]
    n = len(sub) + 1
    ncol = min(max_cols, n)
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(rgb)
    if les_t is not None:
        axes[0].contour(les_t.astype(float), [0.5],
                        colors="white", linewidths=1.5)
    axes[0].set_title(f"{image_id}\n{GEOMETRY} view", fontsize=8)

    for ax, (_, r) in zip(axes[1:], sub.iterrows()):
        m = np.array(Image.open(r["mask_path"]).convert("L")) > 127
        mt, _ = transform_mask(m, GEOMETRY)
        ax.imshow(rgb)
        if les_t is not None:
            ax.contour(les_t.astype(float), [0.5],
                       colors="white", linewidths=1.0)
        ax.imshow(np.ma.masked_where(~mt, mt.astype(float)),
                  cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        ax.contour(mt.astype(float), [0.5], colors="lime", linewidths=1.2)
        ax.set_title(f"{r['label']}\n{r['n_patch_50']} patches", fontsize=7)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


for iid in qc["Image_ID"].drop_duplicates().head(3):
    show_annotations(iid)


# ---- cell 13 ------------------------------------------------------------
qc["qc_pass"] = (
    qc["mask_found"]
    & (qc["px_positive"].fillna(0) > 0)
    & (qc["n_patch_50"].fillna(0) >= MIN_PATCH_50)
    & (qc["frac_kept_after_crop"].fillna(0) >= MIN_CROP_KEEP)
)

out = OUT_DIR / "annotation_qc_manifest.csv"
qc.to_csv(out, index=False)
print(f"written: {out}")

print(f"\ngeometry {GEOMETRY}")
print(f"total {len(qc)}   passing {int(qc['qc_pass'].sum())}")
print()
print(qc.groupby(["label_group", "label"])["qc_pass"]
      .agg(n="size", passing="sum").to_string())
print()
print("failure reasons:")
print(f"  mask not found     : {int((~qc.mask_found).sum())}")
print(f"  empty mask         : {int((qc.px_positive.fillna(0) == 0).sum())}")
print(f"  lost at {CAM_GRID}x{CAM_GRID}      : "
      f"{int((qc.n_patch_50.fillna(0) < MIN_PATCH_50).sum())}")
print(f"  removed by transform: "
      f"{int((qc.frac_kept_after_crop.fillna(0) < MIN_CROP_KEEP).sum())}")


# ---- cell 14 ------------------------------------------------------------
print("=" * 64)
print(f"QC SUMMARY, geometry {GEOMETRY}")
print("=" * 64)
p = qc[qc.qc_pass]
print(f"images annotated        : {qc['Image_ID'].nunique()}")
print(f"images with usable data : {p['Image_ID'].nunique()}")
print(f"annotations total       : {len(qc)}")
print(f"annotations passing     : {len(p)}")
print()
print(f"mask resolution         : "
      f"{set(zip(qc.mask_h.dropna().astype(int), qc.mask_w.dropna().astype(int)))}")
print(f"usable by group         : {p['label_group'].value_counts().to_dict()}")
if "gt_label" in qc.columns:
    print(f"usable images by class  : "
          f"{p.drop_duplicates('Image_ID')['gt_label'].value_counts().to_dict()}")
print()
print("median grid area fraction by group, this is the chance baseline:")
print(p.groupby("label_group")["grid_area_frac"].median().round(3).to_string())
print()
n_mel_img = (p.loc[p.get("gt_label") == "MEL", "Image_ID"].nunique()
             if "gt_label" in p else 0)
print(f"E1 sample size, all images : {p['Image_ID'].nunique()}")
print(f"E2 sample size, MEL only   : {n_mel_img}")


# ---- cell 15 ------------------------------------------------------------
# =============================================================================
# Union masks per image and per group. This is the real E1 and E2 target.
#
# EXPECT MEL union larger than any single label, since features stack.
# WATCH  if median union area fraction approaches 0.5 the chance baseline is
#        high and overlap metrics lose discriminative power.
# =============================================================================
rows = []
for (img, grp), sub in p.groupby(["Image_ID", "label_group"]):
    acc = None
    for _, r in sub.iterrows():
        m = np.array(Image.open(r["mask_path"]).convert("L")) > 127
        acc = m if acc is None else (acc | m)
    grid = mask_to_cam_grid_geom(acc, GEOMETRY)
    rows.append({
        "Image_ID": img,
        "label_group": grp,
        "geometry": GEOMETRY,
        "n_labels": len(sub),
        "union_patches_50": int((grid >= 0.5).sum()),
        "union_area_frac": float((grid >= 0.5).mean()),
    })

uni = pd.DataFrame(rows)
uni.to_csv(OUT_DIR / "union_masks_summary.csv", index=False)

print("union mask size by group:")
print(uni.groupby("label_group")[["n_labels", "union_patches_50",
                                  "union_area_frac"]]
      .agg(["count", "median", "min", "max"]).round(3).to_string())

mel = uni[uni.label_group == "MEL"]
print(f"\nMEL union, {len(mel)} images")
print(f"  median area fraction : {mel.union_area_frac.median():.3f}")
print(f"  range                : {mel.union_area_frac.min():.3f} to "
      f"{mel.union_area_frac.max():.3f}")
print(f"  images above 0.40    : {int((mel.union_area_frac > 0.40).sum())}")


# ---- cell 16 ------------------------------------------------------------
# =============================================================================
# Is the MEL union driven by one dominant label.
# Reads the manifest just written, so it reflects the active geometry.
# =============================================================================
q = pd.read_csv(OUT_DIR / "annotation_qc_manifest.csv").query("qc_pass")
mel_q = q[q.label_group == "MEL"]

print(f"geometry {GEOMETRY}")
print(f"MEL annotations : {len(mel_q)}   images {mel_q.Image_ID.nunique()}")
print()
print(mel_q.label.value_counts().to_string())

for drop in ["structureless_area", "atypical_network"]:
    sub = mel_q[mel_q.label != drop]
    print(f"\nwithout {drop}")
    print(f"  annotations {len(sub)}   images {sub.Image_ID.nunique()}")
    lost = set(mel_q.Image_ID) - set(sub.Image_ID)
    if lost:
        print(f"  images lost entirely: {len(lost)}  {sorted(lost)}")


# ---- cell 17 ------------------------------------------------------------
# =============================================================================
# Which label groups appear on which class of image.
#
# EXPECT MEL features on many NV images. The curated set deliberately
#        included nevi the model confidently called melanoma.
# READ   "MEL union" is a FEATURE label, not a CLASS label. 31 images carry
#        melanoma features but only 16 are melanoma.
# =============================================================================
p = qc[qc.qc_pass].copy()

img = (p.groupby(["Image_ID", "label_group"]).size()
         .unstack(fill_value=0)
         .join(p.drop_duplicates("Image_ID").set_index("Image_ID")["gt_label"]))

grp_cols = [c for c in ["MEL", "NV", "COARSE", "OTHER"] if c in img.columns]

print("images carrying at least one annotation of each group, by true class")
print("=" * 64)
print(pd.DataFrame({g: img.groupby("gt_label")[g].apply(lambda s: (s > 0).sum())
                    for g in grp_cols}).to_string())
print(f"\ntotal images per class: "
      f"{img.gt_label.value_counts().to_dict()}")

print("\n\nannotation counts by group and class")
print("=" * 64)
print(pd.crosstab(p.label_group, p.gt_label, margins=True).to_string())

print("\n\nMEL feature annotations on NV images")
print("=" * 64)
nv_mel = p[(p.gt_label == "NV") & (p.label_group == "MEL")]
print(f"{len(nv_mel)} annotations across {nv_mel.Image_ID.nunique()} nevi")
print(nv_mel.label.value_counts().to_string())
