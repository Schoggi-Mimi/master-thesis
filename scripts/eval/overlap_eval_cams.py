"""Converted from notebooks/mel_nv/12c_overlap_eval_cams.ipynb by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python scripts/eval/overlap_eval_cams.py
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
# 12c — Overlap metrics
#
# Runs once per pooling. GEOMETRY must match 12a and 12b.
#
# Reads   outputs/mel_nv/eval_cams/<pool>/<geometry>/
#         results/annotation_qc/<geometry>/
# Writes  results/overlap/<pool>/<geometry>/
# =============================================================================
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score


def find_repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "scripts" / "generate_finer_cam_panderm.py").exists():
            return p.resolve()
    raise FileNotFoundError


REPO = find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.cam_eval_utils import (
    transform_mask, mask_to_cam_grid_geom, cam_to_grid, norm01,
    CAM_GRID, patch_footprint)

# ---- switches, must match 12a and 12b -------------------------------------
GEOMETRY = "squash224"
POOLING  = "mean"                 # "mean" for GAP, "cls" for CLS
POOL_TAG = "gap" if POOLING == "mean" else "cls"

HAM_ROOT  = REPO / "data" / "HAM10000"
EVAL_ROOT = REPO / "outputs" / "mel_nv" / "eval_cams" / POOL_TAG / GEOMETRY
OUT_DIR   = REPO / "results" / "overlap" / POOL_TAG / GEOMETRY
QC_CSV    = REPO / "results" / "annotation_qc" / GEOMETRY / "annotation_qc_manifest.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K      = 20      # fixed patch count, ~10 percent of 196
ANN_THRESH = 0.5     # patch counts as annotated if at least half covered
N_PERM     = 2000

for p in [EVAL_ROOT / "predictions.csv", EVAL_ROOT / "cam_manifest.csv",
          EVAL_ROOT / "eval_images.csv", QC_CSV]:
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}. Run 12a and 12b with GEOMETRY={GEOMETRY} "
            f"and POOLING={POOLING} first.")

qc      = pd.read_csv(QC_CSV)
qcp     = qc[qc.qc_pass].copy()
pred    = pd.read_csv(EVAL_ROOT / "predictions.csv")
man     = pd.read_csv(EVAL_ROOT / "cam_manifest.csv")
eval_df = pd.read_csv(EVAL_ROOT / "eval_images.csv")

gt   = pred.set_index("image_id")["gt_label"].to_dict()
case = pred.set_index("image_id")["case_type"].to_dict()
conf = pred.set_index("image_id")["confidence"].to_dict()
corr = pred.set_index("image_id")["correct"].to_dict()

fy, fx = patch_footprint(GEOMETRY)
print(f"geometry               : {GEOMETRY}")
print(f"pooling                : {POOLING}  tag {POOL_TAG}")
print(f"eval root              : {EVAL_ROOT.relative_to(REPO)}")
print(f"out dir                : {OUT_DIR.relative_to(REPO)}")
print(f"annotations passing QC : {len(qcp)}")
print(f"images                 : {qcp.Image_ID.nunique()}")
print(f"CAM arrays             : {len(man)}")
print(f"top-k                  : {TOP_K} of {CAM_GRID**2} "
      f"({TOP_K/CAM_GRID**2:.1%})")
print(f"one patch covers       : {fy:.1f} x {fx:.1f} original px")
print(f"\naccuracy on annotated subset: {pred.correct.mean():.3f}")
print(pred.case_type.value_counts().to_string())


# ---- cell 1 ------------------------------------------------------------
# =============================================================================
# Targets, all at 14x14 binary, under the active geometry.
#   group unions  MEL, NV, COARSE   -> E1 and E2
#   single labels                   -> E3, descriptive
#
# Dice ceiling note. At fixed top-k the maximum achievable Dice is
# 2*min(k,T)/(k+T). Smaller targets under squash224 raise that ceiling,
# so raw Dice will look better than under crop224 without localisation
# changing. Always read dice_norm alongside.
# =============================================================================
def union_grid(sub):
    """Union of annotation masks at ORIGINAL resolution, then to grid."""
    acc = None
    for _, r in sub.iterrows():
        m = np.array(Image.open(r.mask_path).convert("L")) > 127
        acc = m if acc is None else (acc | m)
    return mask_to_cam_grid_geom(acc, GEOMETRY) >= ANN_THRESH


targets = {}   # (image_id, target_name) -> bool 14x14

for (img, grp), sub in qcp.groupby(["Image_ID", "label_group"]):
    targets[(img, f"union_{grp}")] = union_grid(sub)

for (img, lab), sub in qcp.groupby(["Image_ID", "label"]):
    targets[(img, f"label_{lab}")] = union_grid(sub)

tinfo = pd.DataFrame([
    {"image_id": k[0], "target": k[1],
     "n_patches": int(v.sum()),
     "area_frac": float(v.mean()),
     "dice_ceiling": float(2 * min(TOP_K, v.sum()) / (TOP_K + v.sum()))
                     if v.sum() else np.nan}
    for k, v in targets.items()
])
tinfo["geometry"] = GEOMETRY
tinfo.to_csv(OUT_DIR / "target_summary.csv", index=False)

print(f"targets built: {len(targets)}")
print()
print(tinfo[tinfo.target.str.startswith("union_")]
      .groupby("target")[["n_patches", "area_frac", "dice_ceiling"]]
      .agg(["count", "median", "min", "max"]).round(3).to_string())

degenerate = tinfo[(tinfo.n_patches == 0) | (tinfo.area_frac > 0.9)]
print(f"\ndegenerate targets, empty or near full: {len(degenerate)}")
if len(degenerate):
    print(degenerate.to_string(index=False))


# ---- cell 2 ------------------------------------------------------------
# =============================================================================
# hit_rate   share of top-k CAM MASS inside the target
# lift       hit_rate / area_frac. 1.0 is chance
# dice       symmetric overlap of the top-k selection with the target
# dice_max   achievable ceiling at this k and target size
# dice_norm  dice / dice_max. comparable across target sizes
# iou        same information as dice, monotonically related
# pointing   hottest patch inside the target, 0 or 1
# cam_auc    threshold free ranking quality. 0.5 is chance. PRIMARY
# recall_k   share of target covered by top-k. diagnostic only
# =============================================================================
def topk_fixed(cam, k=TOP_K):
    flat = np.asarray(cam, np.float32).ravel()
    out = np.zeros(flat.size, bool)
    out[np.argpartition(-flat, k - 1)[:k]] = True
    return out.reshape(cam.shape)


_METRIC_KEYS = ["hit_rate", "lift", "dice", "dice_max", "dice_norm",
                "iou", "pointing", "cam_auc", "recall_k"]


def overlap_metrics(cam_grid, target, k=TOP_K):
    cam = norm01(cam_grid)
    t = np.asarray(target, bool)

    out = {"area_frac": float(t.mean()), "n_target": int(t.sum())}
    if t.sum() == 0 or t.all():
        out.update({m: np.nan for m in _METRIC_KEYS})
        return out

    sel = topk_fixed(cam, k)
    inter = int((sel & t).sum())
    n_sel, n_tgt = int(sel.sum()), int(t.sum())

    dice = 2 * inter / (n_sel + n_tgt + 1e-8)
    dice_max = 2 * min(n_sel, n_tgt) / (n_sel + n_tgt + 1e-8)

    out["hit_rate"]  = float(cam[sel & t].sum() / (cam[sel].sum() + 1e-8))
    out["lift"]      = out["hit_rate"] / (out["area_frac"] + 1e-8)
    out["dice"]      = float(dice)
    out["dice_max"]  = float(dice_max)
    out["dice_norm"] = float(dice / (dice_max + 1e-8))
    out["iou"]       = float(inter / ((sel | t).sum() + 1e-8))
    out["pointing"]  = float(t[np.unravel_index(np.argmax(cam), cam.shape)])
    out["cam_auc"]   = float(roc_auc_score(t.ravel(), cam.ravel()))
    out["recall_k"]  = float(inter / n_tgt)
    return out


# ---- cell 3 ------------------------------------------------------------
# =============================================================================
# gradcam_a, gradcam_b, map_diff are identical across gamma for a given
# checkpoint, since gamma only affects FinerCAM. Keep gamma=0.6 for those
# and both gammas for finercam.
# =============================================================================
cams = {}      # (image_id, source) -> 14x14 float
src_meta = {}  # source -> dict of fields

for _, r in man.iterrows():
    if r.family == "activation":
        if r.cam_type != "finercam" and float(r.gamma) != 0.6:
            continue
        gtag = str(r.gamma).replace(".", "p")
        src = (f"act_{r.ckpt}_{r.cam_type}_gam{gtag}"
               if r.cam_type == "finercam"
               else f"act_{r.ckpt}_{r.cam_type}")
        gamma = float(r.gamma) if r.cam_type == "finercam" else np.nan
    else:
        src = f"attn_{r.ckpt}_{r.cam_type}"
        gamma = np.nan

    arr = np.load(REPO / r.path)
    cams[(r.image_id, src)] = (
        cam_to_grid(arr) if arr.shape != (CAM_GRID, CAM_GRID) else arr)

    src_meta[src] = {
        "family":   "attn" if r.family == "attention" else "act",
        "ckpt":     r.ckpt,
        "cam_type": r.cam_type,
        "gamma":    gamma,
    }

sources = sorted(src_meta)
print(f"loaded {len(cams)} arrays across {len(sources)} sources\n")
print(pd.DataFrame(src_meta).T.to_string())

expect = {(i, s) for i in eval_df.image_id for s in sources}
missing_pairs = expect - set(cams)
if missing_pairs:
    print(f"\nMISSING {len(missing_pairs)} image-source pairs")
    print(sorted(missing_pairs)[:5])


# ---- cell 4 ------------------------------------------------------------
rows = []
for (img, tname), tmask in targets.items():
    for src in sources:
        cam = cams.get((img, src))
        if cam is None:
            continue
        rows.append({
            "image_id": img,
            "target": tname,
            "target_kind": "union" if tname.startswith("union_") else "label",
            "source": src,
            **src_meta[src],
            "gt_label": gt.get(img),
            "case_type": case.get(img),
            "confidence": conf.get(img),
            "correct": corr.get(img),
            **overlap_metrics(cam, tmask),
        })

long = pd.DataFrame(rows)
long["pooling"] = POOL_TAG
long["geometry"] = GEOMETRY
long.to_csv(OUT_DIR / "overlap_metrics_long.csv", index=False)

print(f"rows: {len(long)}")
print(f"nan cam_auc: {int(long.cam_auc.isna().sum())}")
print()
print("MEL union, by source:")
print(long[long.target == "union_MEL"]
      .groupby(["family", "ckpt", "cam_type", "gamma"], dropna=False)
      [["cam_auc", "dice", "dice_norm", "pointing"]]
      .median().round(3).to_string())


# ---- cell 5 ------------------------------------------------------------
# =============================================================================
# Score each CAM against OTHER images' targets. Gives an empirical null
# that respects real CAM shape statistics, unlike the area baseline.
# =============================================================================
rng = np.random.default_rng(0)
null_rows = []

for tname in ["union_MEL", "union_NV", "union_COARSE"]:
    ids = [i for (i, t) in targets if t == tname]
    if len(ids) < 4:
        continue
    for src in sources:
        vals_auc, vals_lift = [], []
        for _ in range(N_PERM // 4):
            i, j = rng.choice(len(ids), 2, replace=False)
            cam = cams.get((ids[i], src))
            if cam is None:
                continue
            m = overlap_metrics(cam, targets[(ids[j], tname)])
            if not np.isnan(m["cam_auc"]):
                vals_auc.append(m["cam_auc"])
                vals_lift.append(m["lift"])
        if vals_auc:
            null_rows.append({
                "target": tname, "source": src,
                "null_auc_mean": float(np.mean(vals_auc)),
                "null_auc_p95": float(np.percentile(vals_auc, 95)),
                "null_lift_mean": float(np.mean(vals_lift)),
                "null_lift_p95": float(np.percentile(vals_lift, 95)),
            })

null = pd.DataFrame(null_rows)
null["pooling"] = POOL_TAG
null["geometry"] = GEOMETRY
null.to_csv(OUT_DIR / "permutation_null.csv", index=False)

obs = (long[long.target_kind == "union"]
       .groupby(["target", "source"])[["cam_auc", "lift"]]
       .median().reset_index()
       .rename(columns={"cam_auc": "obs_auc", "lift": "obs_lift"}))
cmp_ = obs.merge(null, on=["target", "source"], how="left")
cmp_["auc_above_null"]  = cmp_.obs_auc - cmp_.null_auc_mean
cmp_["beats_null_p95"]  = cmp_.obs_auc > cmp_.null_auc_p95
cmp_["pooling"] = POOL_TAG
cmp_["geometry"] = GEOMETRY
cmp_.to_csv(OUT_DIR / "observed_vs_null.csv", index=False)

print("MEL union, observed versus null:")
print(cmp_[cmp_.target == "union_MEL"]
      .sort_values("obs_auc", ascending=False)
      [["source", "obs_auc", "null_auc_mean", "auc_above_null", "beats_null_p95"]]
      .round(3).to_string(index=False))


# ---- cell 6 ------------------------------------------------------------
# =============================================================================
# Lesion conditioned metrics.
#
# Restrict the CAM and the target to lesion patches only. Removes the
# trivial advantage of firing inside a large lesion where all annotations
# happen to live.
#
# READ AUC near 0.5 means HA learned lesion localisation, not feature
#      localisation. Well above 0.5 means genuine within lesion signal.
# =============================================================================
lesion_grid = {}
for _, r in eval_df.iterrows():
    m = np.array(Image.open((HAM_ROOT / str(r.mask_rel_path)).resolve())
                 .convert("L")) > 127
    lesion_grid[r.image_id] = mask_to_cam_grid_geom(m, GEOMETRY) >= 0.5

print(f"lesion coverage at {CAM_GRID}x{CAM_GRID}, geometry {GEOMETRY}:")
la = pd.Series({k: v.mean() for k, v in lesion_grid.items()})
print(la.describe().round(3).to_string())


def conditioned_metrics(cam, target, lesion, k_frac=0.10):
    """Score only inside the lesion. k scales with lesion size."""
    inside = np.asarray(lesion, bool)
    t = np.asarray(target, bool) & inside
    n_in = int(inside.sum())

    if n_in < 10 or t.sum() == 0 or t.sum() == n_in:
        return {"c_auc": np.nan, "c_lift": np.nan, "c_area": np.nan}

    c = norm01(cam)[inside]
    tt = t[inside]
    k = max(1, int(round(k_frac * n_in)))
    sel = np.zeros(n_in, bool)
    sel[np.argpartition(-c, k - 1)[:k]] = True

    area = float(tt.mean())
    hit = float(c[sel & tt].sum() / (c[sel].sum() + 1e-8))
    return {"c_auc": float(roc_auc_score(tt, c)),
            "c_lift": hit / (area + 1e-8),
            "c_area": area}


rows = []
for (img, tname), tmask in targets.items():
    if not tname.startswith("union_"):
        continue
    les = lesion_grid.get(img)
    if les is None:
        continue
    for src in sources:
        cam = cams.get((img, src))
        if cam is None:
            continue
        rows.append({"image_id": img, "target": tname, "source": src,
                     **src_meta[src], "gt_label": gt.get(img),
                     "case_type": case.get(img),
                     **conditioned_metrics(cam, tmask, les)})

cond = pd.DataFrame(rows)
cond["pooling"] = POOL_TAG
cond["geometry"] = GEOMETRY
cond.to_csv(OUT_DIR / "lesion_conditioned_metrics.csv", index=False)

print("\n\nMEL union, lesion conditioned AUC:")
print(cond[cond.target == "union_MEL"]
      .groupby(["family", "ckpt", "cam_type", "gamma"], dropna=False)["c_auc"]
      .agg(["count", "median"]).round(3).to_string())

print("\n\nside by side, target class CAM only:")
raw = (long[(long.target == "union_MEL") &
            (long.cam_type.isin(["gradcam_a", "attncam_a"]))]
       .groupby(["family", "ckpt"])["cam_auc"].median())
con = (cond[(cond.target == "union_MEL") &
            (cond.cam_type.isin(["gradcam_a", "attncam_a"]))]
       .groupby(["family", "ckpt"])["c_auc"].median())
print(pd.DataFrame({"raw_auc": raw, "lesion_conditioned_auc": con})
      .round(3).to_string())


# ---- cell 7 ------------------------------------------------------------
ALPHAS = sorted(man.ckpt.unique(),
                key=lambda c: float(str(c).replace("ha", "").replace("p", ".")))

rows = []
for fam, a, b in [("act", "gradcam_a", "gradcam_b"),
                  ("attn", "attncam_a", "attncam_b")]:
    for ck in ALPHAS:
        sa, sb = f"{fam}_{ck}_{a}", f"{fam}_{ck}_{b}"
        vals = []
        for img in eval_df.image_id:
            ca, cb = cams.get((img, sa)), cams.get((img, sb))
            if ca is None or cb is None:
                continue
            na, nb = norm01(ca).ravel(), norm01(cb).ravel()
            ma, mb = topk_fixed(norm01(ca)), topk_fixed(norm01(cb))
            vals.append({
                "pearson": float(np.corrcoef(na, nb)[0, 1]),
                "topk_iou": float((ma & mb).sum() / ((ma | mb).sum() + 1e-8)),
                "mad": float(np.abs(na - nb).mean()),
            })
        if vals:
            rows.append({"family": fam, "ckpt": ck,
                         **pd.DataFrame(vals).median().to_dict()})

disc = pd.DataFrame(rows)
disc["pooling"] = POOL_TAG
disc["geometry"] = GEOMETRY
disc.to_csv(OUT_DIR / "class_discriminativeness.csv", index=False)

print(f"target versus reference map, {POOL_TAG} {GEOMETRY}, median over images:")
print(disc.round(3).to_string(index=False))


# ---- cell 8 ------------------------------------------------------------
# =============================================================================
# Not the analysis. Just enough to see whether anything is there.
# EXPECT E1: cam_auc rising ha0 -> ha3 -> ha5, at least for attention-CAM.
# EXPECT E2: finercam close to gradcam_a. Any large gap is worth a look.
# =============================================================================
mel = long[(long.target == "union_MEL") & long.cam_auc.notna()]

print("E1 preview, target class CAM on MEL union, all 34 images")
for fam, ct in [("attn", "attncam_a"), ("act", "gradcam_a")]:
    s = (mel[(mel.family == fam) & (mel.cam_type == ct)]
         .groupby("ckpt")["cam_auc"].agg(["count", "median"]).round(3))
    print(f"\n  {fam} / {ct}")
    print(s.to_string())

print("\n\nE2 preview, activation ha5, MEL images only")
e2 = mel[(mel.family == "act") & (mel.ckpt == "ha5") & (mel.gt_label == "MEL")]
print(e2.groupby(["cam_type", "gamma"], dropna=False)["cam_auc"]
      .agg(["count", "median"]).round(3).to_string())

print("\n\nby case type, attention ha5:")
print(mel[(mel.family == "attn") & (mel.ckpt == "ha5") &
          (mel.cam_type == "attncam_a")]
      .groupby("case_type")["cam_auc"].agg(["count", "median"]).round(3).to_string())


# ---- cell 9 ------------------------------------------------------------
ROOT = REPO / "results" / "overlap"

for tag in ["gap", "cls"]:
    sub = ROOT / tag / GEOMETRY
    if not sub.exists():
        continue
    for f in sub.glob("*.csv"):
        d = pd.read_csv(f)
        changed = False
        if "pooling" not in d.columns:
            d["pooling"] = tag
            changed = True
        if "geometry" not in d.columns:
            d["geometry"] = GEOMETRY
            changed = True
        if changed:
            d.to_csv(f, index=False)
            print(f"  backfilled {f.name}")

have = [t for t in ["gap", "cls"]
        if (ROOT / t / GEOMETRY / "class_discriminativeness.csv").exists()]
print(f"geometry {GEOMETRY}, available: {have}")

if len(have) == 2:
    rd = lambda name: pd.concat([pd.read_csv(ROOT / t / GEOMETRY / name)
                                 for t in have])
    TGT = ["attncam_a", "gradcam_a"]

    d = rd("class_discriminativeness.csv")
    print("\nclass discriminativeness, both poolings:")
    print(d.pivot_table(index=["family", "ckpt"], columns="pooling",
                        values=["pearson", "topk_iou"]).round(3).to_string())

    l = rd("overlap_metrics_long.csv")
    m = l[(l.target == "union_MEL") & l.cam_type.isin(TGT)]
    for metric in ["cam_auc", "dice", "dice_norm", "pointing"]:
        print(f"\n\nMEL union {metric}, both poolings:")
        print(m.pivot_table(index=["family", "ckpt"], columns="pooling",
                            values=metric, aggfunc="median")
              .round(3).to_string())

    c = rd("lesion_conditioned_metrics.csv")
    cm = c[(c.target == "union_MEL") & c.cam_type.isin(TGT)]
    print("\n\nlesion conditioned AUC, both poolings:")
    print(cm.pivot_table(index=["family", "ckpt"], columns="pooling",
                         values="c_auc", aggfunc="median").round(3).to_string())

    n = rd("observed_vs_null.csv")
    nm = n[(n.target == "union_MEL") & n.source.str.contains("cam_a$")]
    print("\n\nmargin above permutation null:")
    print(nm.pivot_table(index="source", columns="pooling",
                         values="auc_above_null").round(3).to_string())
else:
    print("run 12c for both POOLING values before comparing")


# ---- cell 10 ------------------------------------------------------------
# =============================================================================
# Sensitivity: does E1 survive without structureless_area, 29 of 86 MEL.
# EXPECT the same direction and a similar magnitude on 27 images.
# =============================================================================
targets_nosa = {}
for img, sub in qcp[(qcp.label_group == "MEL") &
                    (qcp.label != "structureless_area")].groupby("Image_ID"):
    targets_nosa[img] = union_grid(sub)

rows = []
for img, tmask in targets_nosa.items():
    for src in sources:
        cam = cams.get((img, src))
        if cam is None:
            continue
        rows.append({"image_id": img, "source": src, **src_meta[src],
                     **overlap_metrics(cam, tmask)})

sa = pd.DataFrame(rows)
sa["pooling"] = POOL_TAG
sa.to_csv(OUT_DIR / "sensitivity_no_structureless.csv", index=False)

print(f"images: {sa.image_id.nunique()}\n")
print("MEL union without structureless_area, target class CAM:")
print(sa[sa.cam_type.isin(["attncam_a", "gradcam_a"])]
      .groupby(["family", "ckpt"]).cam_auc
      .agg(["count", "median"]).round(3).to_string())
print("\nfull MEL union for comparison:")
print(long[(long.target == "union_MEL") &
           long.cam_type.isin(["attncam_a", "gradcam_a"])]
      .groupby(["family", "ckpt"]).cam_auc
      .agg(["count", "median"]).round(3).to_string())


# ---- cell 11 ------------------------------------------------------------
# =============================================================================
# Main results table. Both poolings, both CAM formulations.
# =============================================================================
ROOT = REPO / "results" / "overlap"
rd = lambda name: pd.concat([pd.read_csv(ROOT / t / GEOMETRY / name)
                             for t in ["gap", "cls"]])

L, C, D, N = (rd("overlap_metrics_long.csv"),
              rd("lesion_conditioned_metrics.csv"),
              rd("class_discriminativeness.csv"),
              rd("observed_vs_null.csv"))

TGT = ["attncam_a", "gradcam_a"]
key = ["pooling", "family", "ckpt"]

sub = L[(L.target == "union_MEL") & L.cam_type.isin(TGT)]
agg = sub.groupby(key)[["cam_auc", "dice", "dice_norm", "pointing"]].median()

con = (C[(C.target == "union_MEL") & C.cam_type.isin(TGT)]
       .groupby(key).c_auc.median())
dis = D.set_index(key)[["pearson", "topk_iou"]]

N["fam"] = N.source.str.split("_").str[0]
N["ck"]  = N.source.str.split("_").str[1]
nul = (N[(N.target == "union_MEL") & N.source.str.contains("cam_a$")]
       .set_index(["pooling", "fam", "ck"]).auc_above_null)
nul.index.names = key

tab = pd.DataFrame({
    "auc_raw":        agg.cam_auc,
    "auc_in_lesion":  con,
    "auc_above_null": nul,
    "dice":           agg.dice,
    "dice_norm":      agg.dice_norm,
    "pointing":       agg.pointing,
    "class_pearson":  dis.pearson,
    "class_iou":      dis.topk_iou,
}).round(3).sort_index()

out = ROOT / f"main_results_table_{GEOMETRY}.csv"
tab.to_csv(out)
print(f"geometry {GEOMETRY}")
print(tab.to_string())
print(f"\nwritten {out.relative_to(REPO)}")


# ---- cell 12 ------------------------------------------------------------
n = pd.concat([pd.read_csv(REPO / "results" / "overlap" / t / GEOMETRY /
                           "observed_vs_null.csv") for t in ["gap", "cls"]])
print(n[(n.target == "union_MEL") & n.source.str.contains("ha0")]
      [["pooling", "source", "obs_auc", "null_auc_mean",
        "null_auc_p95", "auc_above_null", "beats_null_p95"]]
      .round(3).to_string(index=False))
