"""Converted from notebooks/mel_nv/16g_verify.ipynb by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python scripts/figures/final_figures_statistics.py
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
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import sys
import numpy as np
import pandas as pd

def find_repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "scripts" / "generate_finer_cam_panderm.py").exists():
            return p.resolve()
    raise FileNotFoundError


REPO = find_repo_root()
for extra in [REPO, REPO / "external" / "PanDerm" / "classification"]:
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from src.thesis.config import *
from src.thesis import data, models, cams, metrics, plotting, audit
import importlib; importlib.reload(plotting); from src.thesis.plotting import *

EV = data.load_eval()
LESION = data.load_lesions(EV)
QC = data.load_qc()
HUM = data.load_hum(QC, LESION)
X_TEST, Y_MEL, ID_TEST, TEST_DF = data.load_test()
M0, M5 = models.ck5("ha0"), models.ck5("ha5")

AUD = audit.run_audit(EV, LESION, HUM, X_TEST, Y_MEL, ID_TEST, QC, M0, M5)


# ---- cell 1 ------------------------------------------------------------
F0 = pd.read_csv(MELNV / "f2_fold0.csv"); F0["image_id"] = F0.image_id.astype(str).str.strip()
ANN = sorted(set(QC.Image_ID))

# 03:19 site MEL share
meta_p = next(iter(HAM.rglob("*metadata*.csv")), None)
if meta_p is not None:
    md = pd.read_csv(meta_p); md = md[md.dx.isin(["mel", "nv"])]
    sh = md.groupby("dataset").dx.apply(lambda s: (s == "mel").mean()).round(3).to_dict()
    print(f"03:19 site MEL share, n={len(md)} images ({md.lesion_id.nunique()} lesions): {sh}")
else:
    print("03:19 metadata csv not found under HAM, give me its path")

# 03:47 split composition
g = F0.groupby(["split", "gt_label"]).agg(img=("image_id", "size"), les=("lesion_id", "nunique"))
for (s, c), r in g.iterrows():
    print(f"03:47 {s:5} {c}: images {r.img}, lesions {r.les}, images/lesion {r.img / r.les:.2f}")

# 03:73 straddling lesions, per fold
STRAD, TRAIN_SIDE = {}, {}
for k in range(5):
    d = pd.read_csv(MELNV / f"f2_fold{k}.csv"); d["image_id"] = d.image_id.astype(str).str.strip()
    tr, te = d[d.split == "train"], d[d.split == "test"]
    STRAD[k] = []
    for l in sorted(set(tr.lesion_id) & set(te.lesion_id)):
        tri, tei = tr[tr.lesion_id == l].image_id.tolist(), te[te.lesion_id == l].image_id.tolist()
        STRAD[k] += tei; TRAIN_SIDE[l] = tri
        print(f"03:73 fold{k} lesion {l}: train {tri} annotated {[i in ANN for i in tri]} | "
              f"test {tei} gt {te[te.lesion_id == l].gt_label.tolist()}")
STRAD_ALL = sorted({i for v in STRAD.values() for i in v})
print(f"03:73 straddling test images, union over folds: {STRAD_ALL}")

# 03:103 QC and the "two excluded" images
raw = pd.read_csv(QC_CSV)
FM = data.load_feat_manifest(); FM["image_id"] = FM.image_id.astype(str).str.strip()
sup = FM[FM.rule != "nevus_anchor"]
print(f"03:103 annotated images {raw.Image_ID.nunique()}, masks {len(raw)}, masks passing {int(raw.qc_pass.sum())}, "
      f"images passing {QC.Image_ID.nunique()}")
print(f"03:103 images in concept manifest {sup.image_id.nunique()}, images with usable MEL union (HUM) {len(HUM)}")
print(f"03:103 not in concept manifest: {sorted(set(ANN) - set(sup.image_id))}")
for iid in sorted(set(ANN) - set(HUM)):
    u, L = data.mel_union(QC, iid), LESION.get(iid)
    why = ("no MEL structures" if u is None else
           f"lesion cells {int(L.sum())}, MEL cells in lesion {int(u[L].sum())}" if L is not None else "no lesion mask")
    print(f"03:103 not in HUM: {iid} ({EV.loc[iid].gt_label}) -> {why}")

# 04:356, 05:83 controls, 04:358, 05:161 deletion
for f in ["app_robustness.csv", "ch5_deletion.csv", "ch5_deletion_v2.csv"]:
    p = TAB / f
    if not p.exists():
        print(f"04:35x {f}: not present"); continue
    d = pd.read_csv(p); keys = [c for c in ("cond", "model") if c in d]
    n_img = d.image_id.nunique() if "image_id" in d else "no image_id column"
    print(f"04:35x {f}: images {n_img}, rows per {keys}: {d.groupby(keys).size().to_dict() if keys else len(d)}")

# 07:179 the 28
F7 = pd.read_csv(TAB / "ch7_grounding.csv")
s = F7[F7.setting == "HA2 b5"]
nan = s[s.mel_auc_in.isna()]
print(f"07:179 HA2 b5 images {s.image_id.nunique()}, NaN mel_auc_in: {nan.image_id.tolist()} "
      f"(mel_in there: {nan.mel_in.round(4).tolist()})")


# ---- cell 2 ------------------------------------------------------------
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ann_ids = [i for i in ANN if i in EV.index]
A_ = EV.loc[ann_ids]
X_ann = torch.cat([data.to_tensor(data.native(r)) for r in A_.image_rel_path])
y_ann = (A_.gt_label == "MEL").astype(int).values
rng = np.random.default_rng(SEED)
for nm, m in [("ha0 unaligned", M0), ("ha5 aligned", M5)]:
    p987, pa = models.predict_mel(m, X_TEST), models.predict_mel(m, X_ann)
    y, p = np.r_[Y_MEL, y_ann], np.r_[p987, pa]
    a987, a1021, aann = roc_auc_score(Y_MEL, p987), roc_auc_score(y, p), roc_auc_score(y_ann, pa)
    obs = a987 - a1021
    mel, nv = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    null = []
    for _ in range(2000):
        drop = np.r_[rng.choice(mel, y_ann.sum(), replace=False), rng.choice(nv, (1 - y_ann).sum(), replace=False)]
        keep = np.setdiff1d(np.arange(len(y)), drop)
        null.append(roc_auc_score(y[keep], p[keep]) - a1021)
    null = np.array(null)
    print(f"03:71 {nm}: AUC 1021={a1021:.4f} 987={a987:.4f} annotated-only={aann:.4f} "
          f"(n={len(y_ann)}, MEL={y_ann.sum()}) | rise {obs:+.4f}, random removal: mean {null.mean():+.4f}, "
          f"95th pct {np.quantile(null, .95):+.4f}, p={np.mean(null >= obs):.4f}")

rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L = LESION[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for nm, m in [("unaligned", M0), ("aligned", M5)]:
        mel, nv = cams.class_cams(m, x); t = mel if r.gt_label == "MEL" else nv
        rows.append({"model": nm, "loc": metrics.lesion_auc(t, L),
                     "pearson_in": metrics.corr_in(mel, nv, L),
                     "spearman_in": spearmanr(mel[L], nv[L]).statistic,
                     "pearson_all": float(np.corrcoef(mel.ravel(), nv.ravel())[0, 1])})
R1 = pd.DataFrame(rows); R1.to_csv(TAB / "verify_ch5_annotated.csv", index=False)
for nm, g in R1.groupby("model"):
    print(f"05:74 {nm}: localisation median {g['loc'].median():.3f} (n={len(g)})")
    print(f"05:100 {nm} GAP: pearson_in {g.pearson_in.median():.3f}, spearman_in {g.spearman_in.median():.3f}, "
          f"pearson_all {g.pearson_all.median():.3f}")


# ---- cell 3 ------------------------------------------------------------
from src.thesis.config import BLOCK, MEL_IDX, NV_IDX
MC0 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha0.pth", cache=False)
MC5 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha5.pth", cache=False)
print("05:108 pooling of CLS checkpoints:", models.ckpt_args(CK5 / "checkpoint-best-cls-ha0.pth").get("pooling"),
      models.ckpt_args(CK5 / "checkpoint-best-cls-ha5.pth").get("pooling"))

def cos(a, b):
    a, b = a.ravel(), b.ravel(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def steps(model, x):
    with torch.enable_grad():
        base, logits = cams._forward(model, x)
        A = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A, retain_graph=True)[0]
        Gn = torch.autograd.grad(logits[0, NV_IDX], A)[0]
    gap = bool(getattr(base, "use_mean_pooling", False)); base.clear_xai_state()
    f = (lambda T: T[0, :, 1:, 1:]) if gap else (lambda T: T[0, :, 0, 1:])
    A, Gm, Gn = [f(T).detach().float().cpu().numpy() for T in (A, Gm, Gn)]
    Pm, Pn = A * Gm, A * Gn
    Hm, Hn = (Pm.sum(1), Pn.sum(1)) if gap else (Pm, Pn)
    fin = lambda H: cams.norm01(np.maximum(H.mean(0), 0))
    return {"s1_grad": cos(Gm, Gn), "s2_attnxgrad": cos(Pm, Pn),
            "s3_head_median": float(np.median([cos(Hm[h], Hn[h]) for h in range(Hm.shape[0])])),
            "s4_head_mean": cos(Hm.mean(0), Hn.mean(0)), "s5_final": cos(fin(Hm), fin(Hn))}

rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L = LESION[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for pool, nm, m in [("GAP", "unaligned", M0), ("GAP", "aligned", M5), ("CLS", "unaligned", MC0), ("CLS", "aligned", MC5)]:
        mel, nv = cams.class_cams(m, x)
        rows.append({"pool": pool, "model": nm, **steps(m, x),
                     "pearson_in": metrics.corr_in(mel, nv, L), "spearman_in": spearmanr(mel[L], nv[L]).statistic,
                     "pearson_all": float(np.corrcoef(mel.ravel(), nv.ravel())[0, 1])})
R3 = pd.DataFrame(rows); R3.to_csv(TAB / "verify_ch5_heads.csv", index=False)
for (pool, nm), g in R3.groupby(["pool", "model"]):
    md = g.median(numeric_only=True).round(3)
    print(f"05:108 {pool} {nm}: pearson_in {md.pearson_in}, spearman_in {md.spearman_in}, pearson_all {md.pearson_all}")
    print(f"05:135 {pool} {nm}: grad {md.s1_grad}, attn x grad {md.s2_attnxgrad}, typical head {md.s3_head_median}, "
          f"head mean {md.s4_head_mean}, final {md.s5_final}  (n={len(g)})")


# ---- cell 4 ------------------------------------------------------------
from PIL import Image
import cv2
def mask224(rel):
    m = np.array(Image.open((HAM / str(rel)).resolve()).convert("L")) > 127
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
grid = lambda m: m.reshape(14, 16, 14, 16).mean((1, 3)) >= 0.5
def inpaint(img, m):
    return cv2.inpaint(img, cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) * 255, 7, cv2.INPAINT_TELEA)
def share(v, G):
    v = cams.norm01(v); return float(v[G].sum() / (v.sum() + 1e-8))
def place_patch(m, rng):
    ys, xs = np.where(m); h, w = ys.ptp() + 1, xs.ptp() + 1
    if h > 200 or w > 200: return None
    ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    for _ in range(500):
        dy, dx = rng.integers(0, 224 - h + 1) - ys.min(), rng.integers(0, 224 - w + 1) - xs.min()
        p = cv2.warpAffine(m.astype(np.uint8), np.float32([[1, 0, dx], [0, 1, dy]]), (224, 224),
                           flags=cv2.INTER_NEAREST).astype(bool)
        if p.sum() == m.sum() and not (p & ring).any(): return p
    return None

rng = np.random.default_rng(SEED); rows = []
for _, r in TEST_DF.sample(100, random_state=SEED).iterrows():
    img, m = data.rgb224(r.image_rel_path), mask224(r.mask_rel_path); Lg = grid(m)
    if not (0 < Lg.sum() < Lg.size): continue
    p = place_patch(m, rng)
    for nm, mdl in [("unaligned", M0), ("aligned", M5)]:
        row = {"image_id": r.image_id, "model": nm, "lesion_area": Lg.mean()}
        mel0, nv0 = cams.class_cams(mdl, data.to_tensor(img))
        melr, nvr = cams.class_cams(mdl, data.to_tensor(inpaint(img, m)))
        row.update(nv_orig=share(nv0, Lg), nv_removed=share(nvr, Lg), mel_orig=share(mel0, Lg), mel_removed=share(melr, Lg))
        if p is not None and grid(p).any():
            Pg = grid(p); t = cams.target_ref(mdl, data.to_tensor(inpaint(img, p)), r.gt_label)[0]
            row.update(bp_lesion=share(t, Lg), bp_patch=share(t, Pg), patch_area=Pg.mean())
        rows.append(row)
RC = pd.DataFrame(rows); RC.to_csv(TAB / "verify_controls.csv", index=False)
for nm, g in RC.groupby("model"):
    b = g.dropna(subset=["bp_patch"])
    print(f"05:83 background patch {nm}: n={len(b)}, lesion share {b.bp_lesion.median():.3f} (area {b.lesion_area.median():.3f}), "
          f"patch share {b.bp_patch.median():.3f} (area {b.patch_area.median():.3f}), "
          f"patch vs area {metrics.paired(b.bp_patch, b.patch_area)}")
    print(f"05:83 lesion removed {nm}: n={len(g)}, NV share {g.nv_orig.median():.3f} -> {g.nv_removed.median():.3f} "
          f"{metrics.paired(g.nv_removed, g.nv_orig)} | MEL share {g.mel_orig.median():.3f} -> {g.mel_removed.median():.3f}")


# ---- cell 5 ------------------------------------------------------------
import torch.nn.functional as tF
d0 = pd.concat([pd.read_csv(MELNV / "f2_fold0.csv"), pd.read_csv(MELNV / "f2_fold0_heldout.csv")])
d0["image_id"] = d0.image_id.astype(str).str.strip(); d0 = d0.drop_duplicates("image_id")
area, dis_grid, dis_les = [], [], []
for _, r in d0.iterrows():
    m = np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127
    area.append(m.mean())
d0["area"] = area
print(f"03:60 mean lesion area, native resolution, n={len(d0)}: "
      f"MEL {d0[d0.gt_label == 'MEL'].area.mean():.3f}, NV {d0[d0.gt_label == 'NV'].area.mean():.3f}")

for _, r in TEST_DF.sample(300, random_state=SEED).iterrows():
    m = np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127
    m224 = cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    nn = tF.interpolate(torch.from_numpy(m224)[None, None], size=(14, 14), mode="nearest")[0, 0].numpy() > 0.5
    cov = data.to_grid(m)
    dis_grid.append((nn != cov).mean()); dis_les.append((nn != cov).sum() / max(cov.sum(), 1))
print(f"04:171 nearest vs half-coverage, 300 test masks: differing cells median {np.median(dis_grid):.3f} of grid "
      f"(mean {np.mean(dis_grid):.3f}), {np.median(dis_les):.3f} of lesion cells")


# ---- cell 6 ------------------------------------------------------------
SWEEP = {0.0: CK5 / "checkpoint-best-gap-ha0.pth"}
for lam in [0.1, 0.25, 0.5, 1, 3, 5, 7, 11, 17]:
    SWEEP[float(lam)] = CK_ALPHA / f"checkpoint-best-gap-ha{str(lam).replace('.', 'p')}.pth"
ID = np.asarray(ID_TEST).astype(str)

# Part A. Sweep models, excluding the straddling test images
keep = ~np.isin(ID, STRAD_ALL); P = {}
for lam, path in SWEEP.items():
    m = models.load_ckpt(path, cache=False); P[lam] = models.predict_mel(m, X_TEST); del m
    full, ex = roc_auc_score(Y_MEL, P[lam]), roc_auc_score(Y_MEL[keep], P[lam][keep])
    print(f"03:73 sweep lambda {lam:5}: AUC {full:.4f} -> {ex:.4f} excluding {len(STRAD_ALL)}, delta {ex - full:+.4f}")
pd.DataFrame(P, index=ID).to_csv(TAB / "pred_sweep.csv")

# Part B. F2 models, per fold, only if cell B shows annotated train-side images (about 40 min)
RUN_F2 = False
if RUN_F2:
    IDX = {(float(a["ha_lambda"]), float(a["feat_lambda"]), int(a["fold"])): p
           for p in CK_F2.glob("*.pth") for a in [models.ckpt_args(p)]}
    out = []
    for (al, be, k), path in sorted(IDX.items()):
        if not STRAD[k]: continue
        kk = ~np.isin(ID, STRAD[k])
        m = models.load_ckpt(path, cache=False); p = models.predict_mel(m, X_TEST); del m
        out.append({"alpha": al, "beta": be, "fold": k,
                    "delta": roc_auc_score(Y_MEL[kk], p[kk]) - roc_auc_score(Y_MEL, p)})
    OB = pd.DataFrame(out); OB.to_csv(TAB / "verify_f2_leakage.csv", index=False)
    for (al, be), g in OB.groupby(["alpha", "beta"]):
        print(f"03:73 F2 alpha {al} beta {be}: delta per fold {g.delta.round(4).tolist()}, mean {g.delta.mean():+.4f}")


# ---- cell 7 ------------------------------------------------------------
# 03:19 site shares: find the metadata file
print([str(p) for p in REPO.rglob("*.csv") if "metadata" in p.name.lower()][:10])

# 03:103 why ISIC_0032143 and ISIC_0032462 are not in the concept manifest
raw = pd.read_csv(QC_CSV)
print(raw[raw.Image_ID.isin(["ISIC_0032143", "ISIC_0032462"])][["Image_ID", "label", "label_group", "qc_pass", "n_patch_50"]].to_string())

# 5.7 deletion v2 numbers
DL = pd.read_csv(TAB / "ch5_deletion_v2.csv")
REG = ["top 10%", "random 10%", "clinician (n)", "unannotated (n)"]
print(DL.groupby("model")[REG].median().round(3).to_string())
for mo in DL.model.unique():
    s = DL[DL.model == mo]
    print(mo, "top vs random", metrics.paired(s["top 10%"], s["random 10%"]),
          "| clinician vs unannotated", metrics.paired(s["clinician (n)"], s["unannotated (n)"]))


# ---- cell 8 ------------------------------------------------------------
F = pd.concat([pd.read_csv(MELNV / "f2_fold0.csv"), pd.read_csv(MELNV / "f2_fold0_heldout.csv")]).drop_duplicates("image_id")
print(list(F.columns))
sc = next((c for c in F.columns if c.lower() in ("dataset", "site", "source", "dataset_name")), None)
if sc:
    print(f"03:19 n={len(F)}", F.groupby(sc).gt_label.apply(lambda s: round((s == "MEL").mean(), 3)).to_dict())


# ---- cell 9 ------------------------------------------------------------
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
rows, picks, zero = [], [], []
for iid in sorted(HUM):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    Hm, Hn = cams.head_maps(M5, x, "MEL"), cams.head_maps(M5, x, "NV")
    order = np.argsort([cos(Hm[h], Hn[h]) for h in range(Hm.shape[0])])
    for k in (12, 3, 1):
        idx = order[:k]
        mel = cams.norm01(np.maximum(Hm[idx].mean(0), 0)).reshape(14, 14)
        nv = cams.norm01(np.maximum(Hn[idx].mean(0), 0)).reshape(14, 14)
        t = mel if r.gt_label == "MEL" else nv
        rows.append({"k": k, "image_id": iid, "corr_in": metrics.corr_in(mel, nv, L),
                     "loc": metrics.lesion_auc(t, L), "auc_in": metrics.auc_in(t, L, U)})
    picks += list(order[:3])
    m0, n0 = cams.class_cams(M0, x)
    zero.append(n0.ravel()[np.argmax(m0)] < 1e-6)
HS = pd.DataFrame(rows); HS.to_csv(TAB / "ch5_head_selection.csv", index=False)
print(HS.groupby("k")[["corr_in", "loc", "auc_in"]].median().round(3).to_string())
q = HS.pivot_table(index="image_id", columns="k", values="auc_in")
print("3 heads vs all, auc_in:", metrics.paired(q[3], q[12]))
cnt = pd.Series(picks).value_counts()
print(f"heads selected at least once: {len(cnt)} of 12, most frequent on {cnt.iloc[0] / len(HUM):.0%} of images")
print(f"5.4 unaligned NV map zero at MEL peak: {np.mean(zero):.0%} of {len(zero)}")


# ---- cell 10 ------------------------------------------------------------
RB = pd.read_csv(TAB / "app_robustness.csv")
for mo in ["aligned", "unaligned"]:
    s = RB[(RB.cond == "shifted") & (RB.model == mo)].dropna(subset=["share", "area"])
    r = metrics.paired(s.share, s.area)
    tag = "5.3 shift" if mo == "aligned" else "5.3 shift (unaligned, context)"
    print(f"{tag}: heat={s.share.median():.3f} area={s.area.median():.3f} "
          f"p={r['p']:.2g} wins={r['wins']}/{len(s)}")


# ---- cell 11 ------------------------------------------------------------
ids = ["ISIC_0032143", "ISIC_0032462"]
for p in sorted(MELNV.rglob("*.csv")):
    try:
        d = pd.read_csv(p, usecols=lambda c: c in ("image_id", "Image_ID", "split", "rule", "has_feat", "mask_found"))
    except Exception:
        continue
    col = "image_id" if "image_id" in d else ("Image_ID" if "Image_ID" in d else None)
    if col is None:
        continue
    d[col] = d[col].astype(str).str.strip()
    hit = d[d[col].isin(ids)]
    print(f"{str(p.relative_to(MELNV)):55} n={len(d):5}  "
          f"{hit.drop(columns=col).to_dict('records') if len(hit) else 'absent'}")
for i in ids:
    print(i, "files:", [str(x.relative_to(HAM)) for x in HAM.rglob(f"{i}*")][:6])


# ---- cell 12 ------------------------------------------------------------
FOLD = data.fold_of_heldout()
held = sorted(i for i in FOLD if i in HUM)

def gap(m, img, gt):
    with torch.no_grad():
        o = m(data.to_tensor(img))
        l = o["logits"] if isinstance(o, dict) else (o[0] if isinstance(o, (tuple, list)) else o)
    a, b = (MEL_IDX, NV_IDX) if gt == "MEL" else (NV_IDX, MEL_IDX)
    return float(l[0, a] - l[0, b])

def remove(img, cells):
    mk = np.kron(cells.astype(np.uint8), np.ones((16, 16), np.uint8)) * 255
    return cv2.inpaint(img, mk, 7, cv2.INPAINT_TELEA)

def cellmask(idx):
    c = np.zeros(196, bool); c[idx] = True; return c.reshape(14, 14)

def regions(model, iid, rng, frac=0.10):
    r = EV.loc[iid]; img = data.rgb224(r.image_rel_path)
    L, U = LESION[iid], HUM[iid]
    lidx = np.flatnonzero(L); aidx = np.flatnonzero(U & L); uidx = np.flatnonzero(L & ~U)
    k = max(3, int(round(frac * L.sum())))
    t = cams.target_ref(model, data.to_tensor(img), r.gt_label)[0]
    ku = min(len(aidx), len(uidx))
    return img, r.gt_label, {
        "top 10%": cellmask(lidx[np.argsort(-t.ravel()[lidx])[:k]]),
        "random 10%": cellmask(rng.choice(lidx, k, replace=False)),
        "clinician (n)": cellmask(rng.choice(aidx, ku, replace=False)),
        "unannotated (n)": cellmask(rng.choice(uidx, ku, replace=False))}, t

rng = np.random.default_rng(SEED); rows = []
for k in range(5):
    m = models.load_ckpt(REPO / "external/checkpoints_f2/ha2b1_ep9" / f"fold{k}.pth", cache=False)
    for iid in [i for i in held if FOLD[i] == k]:
        img, gt, R, _ = regions(m, iid, rng); base = gap(m, img, gt)
        rows.append({"image_id": iid, **{nm: (base - gap(m, remove(img, c), gt)) / max(abs(base), 0.1)
                                         for nm, c in R.items()}})
    del m
E9 = pd.DataFrame(rows); E9.to_csv(TAB / "ch5_deletion_v2_expert_ep9.csv", index=False)
print(E9[["top 10%", "random 10%", "clinician (n)", "unannotated (n)"]].median().round(3).to_string())
print("ep9 top vs random:", metrics.paired(E9["top 10%"], E9["random 10%"]))
print("ep9 clinician vs unannotated:", metrics.paired(E9["clinician (n)"], E9["unannotated (n)"]))


# ---- cell 13 ------------------------------------------------------------
from sklearn.metrics import roc_auc_score, average_precision_score
PS = pd.read_csv(TAB / "pred_sweep.csv", index_col=0)          # written by cell C2, columns = lambda_global
PS.columns = [float(c) for c in PS.columns]
PS.index = PS.index.astype(str)
assert list(PS.index) == list(np.asarray(ID_TEST).astype(str)), "row order differs from X_TEST"
P = {f"sweep {c:g}": PS[c].values for c in PS.columns}
P["main unaligned (ha0)"] = models.predict_mel(M0, X_TEST)
P["main aligned (ha5)"] = models.predict_mel(M5, X_TEST)

rng = np.random.default_rng(SEED)
IDX = [i for i in (rng.integers(0, len(Y_MEL), len(Y_MEL)) for _ in range(2000)) if 0 < Y_MEL[i].sum() < len(i)]
rows = []
for nm, p in P.items():
    auc = np.array([roc_auc_score(Y_MEL[i], p[i]) for i in IDX])
    ap = np.array([average_precision_score(Y_MEL[i], p[i]) for i in IDX])
    rows.append({"model": nm, "AUC": roc_auc_score(Y_MEL, p), "AUC_sd": auc.std(ddof=1),
                 "AP": average_precision_score(Y_MEL, p), "AP_sd": ap.std(ddof=1)})
BS = pd.DataFrame(rows); BS.to_csv(TAB / "ch5_bootstrap_sd.csv", index=False)
print(f"1. bootstrap resamples used: {len(IDX)}")
for _, r in BS.iterrows():
    print(f"1. {r.model:22} AUC {r.AUC:.4f} ± {r.AUC_sd:.4f}   AP {r.AP:.4f} ± {r.AP_sd:.4f}")


# ---- cell 14 ------------------------------------------------------------
def miqr(s):
    s = pd.Series(s).dropna()
    return f"{s.median():.3f} [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}] n={len(s)}"

# localisation and heat share, 986 test images
LOC = pd.read_csv(TAB / "ch5_localisation_test.csv")
for mo, g in LOC.groupby("model"):
    print(f"2. test {mo:9} localisation   {miqr(g.loc_auc)}")
    print(f"2. test {mo:9} heat GT map    {miqr(g.heat_in_lesion)}")
    print(f"2. test {mo:9} heat MEL map   {miqr(g.mel_in_lesion)}")
    print(f"2. test {mo:9} heat NV map    {miqr(g.nv_in_lesion)}")

# explanation methods, 31 annotated images
M61 = pd.read_csv(TAB / "ch6_metrics.csv")
for meth in ["Grad-CAM", "Diff-CAM", "FinerCAM 0.6", "random"]:
    g = M61[M61.method == meth]
    print(f"2. ch6 {meth:13} AUC_in {miqr(g.auc_in)}   E_in {miqr(g.energy_in)}")

# class map correlation, 31 annotated images
VA = pd.read_csv(TAB / "verify_ch5_annotated.csv")
for mo, g in VA.groupby("model"):
    print(f"2. r_in {mo:9} pearson {miqr(g.pearson_in)}   spearman {miqr(g.spearman_in)}")

# expert grid, 29 held-out images, MEL map; selected checkpoints, then epoch 9
F7 = pd.concat([pd.read_csv(TAB / "ch7_grounding.csv"), pd.read_csv(TAB / "ch7_ep9.csv")])
ORDER = ["control", "control ep9", "HA1 only", "joint b1", "joint b5", "joint b20",
         "HA2 b1", "HA2 b1 ep9", "HA2 b5", "HA2 b20"]
for st in ORDER:
    g = F7[F7.setting == st]
    if g.empty: continue
    print(f"2. grid {st:12} AUC_in {miqr(g.mel_auc_in)}  E_in {miqr(g.mel_in)}  "
          f"E_grid {miqr(g.mel_grid)}  r_in {miqr(g.corr_in)}")


# ---- cell 15 ------------------------------------------------------------
FM = data.load_feat_manifest(); FM["image_id"] = FM.image_id.astype(str).str.strip()
ANN = set(QC.Image_ID)
for k in range(5):
    d = pd.read_csv(MELNV / f"f2_fold{k}.csv"); d["image_id"] = d.image_id.astype(str).str.strip()
    h = pd.read_csv(MELNV / f"f2_fold{k}_heldout.csv"); h["image_id"] = h.image_id.astype(str).str.strip()
    sup = d[(d.split == "train") & (d.has_feat == 1) & d.image_id.isin(ANN)]
    hh = h[h.image_id.isin(HUM)]
    gt = EV.loc[hh.image_id, "gt_label"]
    print(f"3. fold {k}: supervised annotated {len(sup)} (MEL {(sup.gt_label == 'MEL').sum()}, "
          f"NV {(sup.gt_label == 'NV').sum()}) | held-out {len(h)}, with MEL target {len(hh)}, "
          f"of which melanoma {(gt == 'MEL').sum()}")


# ---- cell 16 ------------------------------------------------------------
cells = LOC[LOC.model == LOC.model.iloc[0]].lesion_share * 196
print(f"4. lesion size on {len(cells)} test images, grid cells: median {cells.median():.1f} "
      f"[{cells.quantile(.25):.1f}, {cells.quantile(.75):.1f}]")


# ---- cell 17 ------------------------------------------------------------
def px(rel):
    if not isinstance(rel, str) or not rel: return None
    p = (HAM / rel).resolve()
    return int((np.array(Image.open(p).convert("L")) > 127).sum()) if p.exists() else -1

mel = FM[(FM.dx.str.lower().str.startswith("mel")) & (FM.rule != "nevus_anchor")]
print("5A. melanomas in the concept manifest by rule:", mel.rule.value_counts().to_dict())
for rule, g in mel.groupby("rule"):
    nv = [px(r) for r in g.feat_nv_rel_path]; me = [px(r) for r in g.feat_mel_rel_path]
    print(f"5A. rule {rule:14} n={len(g)}  T_NV foreground px: {nv}  T_MEL foreground px: {me}")


# ---- cell 18 ------------------------------------------------------------
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from src.thesis.plotting import overlay, outline, save, TEXTWIDTH, C_LESION
import matplotlib.patches as mpatches
import cv2

NEW = {"lesion aligned": "global aligned", "concept only": "expert aligned", "aligned": "global aligned",
       "control": "control", "control ep9": "control, epoch 9", "HA1 only": "global only",
       "joint b1": "joint, λ_expert 1", "joint b5": "joint, λ_expert 5", "joint b20": "joint, λ_expert 20",
       "HA2 b1": "expert only, λ_expert 1", "HA2 b5": "expert only, λ_expert 5",
       "HA2 b20": "expert only, λ_expert 20", "HA2 b1 ep9": "expert only, λ_expert 1, epoch 9"}
NEW["global only ep9"] = "global only, epoch 9"
rn = lambda s: NEW.get(s, s)

def miqr(s, nd=3):
    s = pd.Series(s).dropna()
    return f"{s.median():.{nd}f} [{s.quantile(.25):.{nd}f}, {s.quantile(.75):.{nd}f}]" + ("" if len(s) in (29, 31) else f" n={len(s)}")

FOLD = data.fold_of_heldout()
held = [i for i in sorted(HUM) if i in FOLD]
def egrid(v, T):
    v = cams.norm01(v); return float(v[T].sum() / (v.sum() + 1e-8) - T.mean())
print("held-out", len(held), " annotated with MEL target", len(HUM))


# ---- cell 19 ------------------------------------------------------------
DL = pd.read_csv(TAB / "ch5_deletion_v2.csv")
REGS = ["top 10%", "random 10%", "clinician (n)", "unannotated (n)"]
rows = []
for mo in ["unaligned", "lesion aligned", "concept only"]:
    g = DL[DL.model == mo]
    rows.append({"model": rn(mo), **{r: miqr(g[r]) for r in REGS}, "n": len(g)})
A1 = pd.DataFrame(rows); A1.to_csv(TAB / "table_deletion_iqr.csv", index=False)
print("A1 source ch5_deletion_v2.csv. Expert aligned = lambda_global 0, lambda_expert 1, recall-selected\n", A1.to_string(index=False))


# ---- cell 20 ------------------------------------------------------------
G9 = REPO / "external/checkpoints_f2/ha1only_ep9"
rows = []
for k in range(5):
    p = G9 / f"fold{k}.pth"
    ck = torch.load(p, map_location="cpu", weights_only=False); a = models.ckpt_args(p)
    assert int(ck.get("epoch")) == 9 and int(a["fold"]) == k and float(a["ha_lambda"]) == 5.0 \
        and float(a.get("feat_lambda") or 0) == 0.0, (k, ck.get("epoch"), a.get("ha_lambda"), a.get("feat_lambda"))
    m = models.load_ckpt(p, cache=False)
    for iid in [i for i in held if FOLD[i] == k]:
        r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]
        mel, nv = cams.class_cams(m, data.to_tensor(data.native(r.image_rel_path)))
        t = mel if r.gt_label == "MEL" else nv
        rows.append({"setting": "global only ep9", "image_id": iid, "gt": r.gt_label,
                     "mel_in": metrics.energy_excess(mel, L, U), "mel_grid": egrid(mel, U & L),
                     "mel_auc_in": metrics.auc_in(mel, L, U), "gt_loc": metrics.lesion_auc(t, L),
                     "corr_in": metrics.corr_in(mel, nv, L)})
    del m
G9D = pd.DataFrame(rows); G9D.to_csv(TAB / "ch7_global_ep9.csv", index=False)
E9 = pd.read_csv(TAB / "ch7_ep9.csv"); E9 = E9[E9.setting == "HA2 b1 ep9"]
print("A2 global only, epoch 9:", {c: miqr(G9D[c]) for c in ["mel_auc_in", "mel_in", "mel_grid", "gt_loc", "corr_in"]})
for c in ["mel_auc_in", "mel_in", "mel_grid", "gt_loc"]:
    q = pd.concat([E9.set_index("image_id")[c].rename("e"), G9D.set_index("image_id")[c].rename("g")], axis=1).dropna()
    print(f"A2 expert ep9 vs global ep9, {c}: {metrics.paired(q.e, q.g)}")


# ---- cell 21 ------------------------------------------------------------
GE = pd.read_csv(TAB / "ch6_group_energy.csv")
print("A3 source ch6_group_energy.csv")
print("A3 images:", GE.image_id.nunique(), " all held-out:", set(GE.image_id) <= set(held),
      " models:", GE.model.unique().tolist())


# ---- cell 22 ------------------------------------------------------------
MC0 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha0.pth", cache=False)
MC5 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha5.pth", cache=False)
rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for nm, m in [("unaligned", MC0), ("global aligned", MC5)]:
        t = cams.target_ref(m, x, r.gt_label)[0]
        rows.append({"model": nm, "image_id": iid, "loc": metrics.lesion_auc(t, L),
                     "auc_in": metrics.auc_in(t, L, U), "energy_in": metrics.energy_excess(t, L, U)})
CL = pd.DataFrame(rows); CL.to_csv(TAB / "appC_cls.csv", index=False)
H3 = pd.read_csv(TAB / "verify_ch5_heads.csv"); H3 = H3[H3.pool == "CLS"].replace({"model": {"aligned": "global aligned"}})
print("A4 CLS pooling, 31 annotated images, ground-truth class map")
for nm in ["unaligned", "global aligned"]:
    g, h = CL[CL.model == nm], H3[H3.model == nm]
    print(f"A4 {nm:15} loc {miqr(g['loc'])}  AUC_in {miqr(g.auc_in)}  E_in {miqr(g.energy_in)}  "
          f"r_in Pearson {miqr(h.pearson_in)}  Spearman {miqr(h.spearman_in)}")


# ---- cell 23 ------------------------------------------------------------
M61 = pd.read_csv(TAB / "ch6_metrics.csv")
COLS = {"auc_in": "in-lesion AUC", "energy_in": "energy excess, lesion", "energy_grid": "energy excess, whole grid",
        "topk_exc": "top-k excess", "struct_hit": "structures hit by a top-k cell",
        "peak_dist": "distance hottest cell to target", "in_lesion": "heat inside lesion"}
ORDM = ["Grad-CAM", "Diff-CAM", "FinerCAM 0.6", "random"]
T5 = pd.DataFrame({m: {COLS[c]: miqr(M61[M61.method == m][c]) for c in COLS} for m in ORDM})
T5.to_csv(TAB / "table_metric_validity.csv"); T5.to_latex(TAB / "table_metric_validity.tex", escape=True)
print("A5 source ch6_metrics.csv, global aligned model, 31 annotated images, median [IQR]\n", T5.to_string())


# ---- cell 24 ------------------------------------------------------------
H = pd.read_csv(MELNV / "ham_mel_nv_clean.csv"); H["image_id"] = H.image_id.astype(str).str.strip()
SF = pd.concat([pd.read_csv(MELNV / "f2_fold0.csv"), pd.read_csv(MELNV / "f2_fold0_heldout.csv")])
SF["image_id"] = SF.image_id.astype(str).str.strip(); SF = SF.drop_duplicates("image_id")

site = SF.groupby("dataset").agg(images=("image_id", "size"), MEL=("gt_label", lambda s: (s == "MEL").sum()),
                                 NV=("gt_label", lambda s: (s == "NV").sum()))
site["MEL_share"] = (site.MEL / site.images).round(3)
site.to_csv(TAB / "appA_sites.csv"); site.to_latex(TAB / "appA_sites.tex")
print(f"B6 source split files, n={len(SF)}\n", site.to_string())

def split_table(d, tag):
    g = d.groupby(["split", "gt_label"]).agg(images=("image_id", "size"), lesions=("lesion_id", "nunique"))
    g["img_per_lesion"] = (g.images / g.lesions).round(2)
    g.to_csv(TAB / f"appA_splits_{tag}.csv"); g.to_latex(TAB / f"appA_splits_{tag}.tex")
    print(f"B7 {tag}, n={len(d)}\n", g.to_string())
split_table(H, "original")
split_table(SF.assign(split=np.where(SF.image_id.isin(set(QC.Image_ID)) & (SF.split != "train"), "held-out", SF.split)), "final_fold0")


# ---- cell 25 ------------------------------------------------------------
MORPH = {"structureless_area": "STRUCTURELESS", "homogeneous": "STRUCTURELESS",
         "atypical_network": "NETWORK", "negative_network": "NETWORK", "regular_network": "NETWORK",
         "peppering": "REGRESSION", "regression": "REGRESSION",
         "regression_scar_like_depigmentation_and_peppering": "REGRESSION",
         "blue_white_structureless_area": "BLUE_WHITE", "atypical_dots": "DOTS",
         "atypical_streaks": "STREAKS", "atypical_vascular_pattern": "VASCULAR"}
CHAN = {"MEL": "T_MEL, melanoma indicative", "NV": "T_NV, benign", "OTHER": "not used", "COARSE": "not used"}
q = pd.read_csv(QC_CSV); q = q[q.qc_pass.astype(bool)]
T8 = (q.groupby(["label", "label_group"]).apply(lambda g: pd.Series({
        "on MEL": int((g.gt_label == "MEL").sum()), "on NV": int((g.gt_label == "NV").sum()),
        "images": g.Image_ID.nunique()}))
      .reset_index())
T8["target channel"] = T8.label_group.map(CHAN)
T8["morphology group"] = T8.label.map(MORPH).fillna("none")
T8 = T8.sort_values(["label_group", "images"], ascending=[True, False])
T8.to_csv(TAB / "appA_labels.csv", index=False); T8.to_latex(TAB / "appA_labels.tex", index=False, escape=True)
print(f"B8 passed masks {len(q)}, labels {T8.label.nunique()}\n", T8.to_string(index=False))


# ---- cell 26 ------------------------------------------------------------
import matplotlib.pyplot as plt
FM = data.load_feat_manifest(); FM["image_id"] = FM.image_id.astype(str).str.strip()
ann = sorted(set(QC.Image_ID)); fold_set = sorted(set(FM[FM.rule != "nevus_anchor"].image_id) & set(ann))
mel = lambda ids: int((EV.loc[[i for i in ids if i in EV.index]].gt_label == "MEL").sum())
o = H.groupby("split").gt_label.agg(n="size", m=lambda s: (s == "MEL").sum())
N = {"ham": (10015, None), "sub": (len(H), int((H.gt_label == "MEL").sum())),
     "sf": (len(SF), int((SF.gt_label == "MEL").sum())),
     "tr": tuple(o.loc["train"]), "va": tuple(o.loc["val"]), "te": tuple(o.loc["test"]),
     "t987": (len(TEST_DF), int((TEST_DF.gt_label == "MEL").sum())), "ann": (len(ann), mel(ann)),
     "f32": (len(fold_set), mel(fold_set)), "h29": (len(held), mel(held))}
print("C9 counts:", N)

# N_SF = (len(SF), int((SF.gt_label == "MEL").sum())); print("data flow, split files box:", N_SF)

def box(ax, xy, title, nm, w=0.21, h=0.085, fc="#F2F2F2"):
    n, m = N[nm]; txt = f"{title}\n{n:,}" + (f" ({m})" if m is not None else "")
    ax.add_patch(mpatches.FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle="round,pad=0.008",
                                         fc=fc, ec="0.3", lw=0.8))
    ax.text(*xy, txt, ha="center", va="center", fontsize=7)
def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3"))

P = {"ham": (.5, .93), "sub": (.5, .78), "tr": (.18, .60), "va": (.42, .60), "te": (.70, .60),
     "t987": (.58, .42), "ann": (.84, .42), "f32": (.84, .25), "h29": (.84, .08), "sf": (.26, .25)}
fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.72)); ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1))
box(ax, P["ham"], "HAM10000", "ham"); box(ax, P["sub"], "melanoma and nevus subset", "sub")
box(ax, P["tr"], "train, original", "tr"); box(ax, P["va"], "validation", "va"); box(ax, P["te"], "test, original", "te")
box(ax, P["t987"], "test without annotated images", "t987", fc="#DCEBF7")
box(ax, P["ann"], "annotated images", "ann", fc="#FCE8D5"); box(ax, P["f32"], "annotated set for folds", "f32", fc="#FCE8D5")
box(ax, P["h29"], "held-out with melanoma target", "h29", fc="#FCE8D5")
box(ax, P["sf"], "split files: train, validation,\ntest and annotated set", "sf", w=0.30, h=0.10)
for a, b in [("ham", "sub"), ("sub", "tr"), ("sub", "va"), ("sub", "te"), ("te", "t987"), ("te", "ann"),
             ("ann", "f32"), ("f32", "h29")]:
    arrow(ax, (P[a][0], P[a][1] - .045), (P[b][0], P[b][1] + .045))
for a in ["tr", "va", "t987", "f32"]:
    arrow(ax, (P[a][0], P[a][1] - .045), (P["sf"][0] + (P[a][0] - .5) * .25, P["sf"][1] + .05))
ax.text(0.5, 0.0, "Counts give images, melanomas in brackets. Two annotated images are not part of the split files.",
        ha="center", fontsize=6.5, color="0.35")
fig.tight_layout(); save(fig, "fig_ch3_data_flow"); plt.show()


# ---- cell 27 ------------------------------------------------------------
import matplotlib as mpl
def full224(m):
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
def label_union(iid, lab):
    acc = None
    for p in QC[(QC.Image_ID == iid) & (QC.label == lab)].mask_path:
        m = data._full(p); acc = m if acc is None else (acc | m)
    return full224(acc)

cand = [i for i in sorted(HUM) if EV.loc[i].gt_label == "MEL" and 40 <= LESION[i].sum() <= 80
        and 2 <= QC[(QC.Image_ID == i) & (QC.label_group == "MEL")].label.nunique() <= 3]
iid = cand[0]; print("C10 image (first by ID meeting the rule):", iid, " candidates:", len(cand))
r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); L, U = LESION[iid], HUM[iid]
Lf = full224(np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127)
labs = QC[(QC.Image_ID == iid) & (QC.label_group == "MEL")].label.unique()

fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH / 3 + 0.35))
ax[0].imshow(rgb); ax[0].contour(Lf.astype(float), [.5], colors=[C_LESION], linewidths=1.1)
for lab in labs:
    ax[0].contour(label_union(iid, lab).astype(float), [.5], colors=["white"], linewidths=0.9)
for j, (G, c, txt) in enumerate([(L, C_LESION, f"lesion cells: {int(L.sum())}"),
                                 (U & L, "white", f"melanoma structure cells: {int((U & L).sum())}")], start=1):
    ax[j].imshow(rgb, alpha=0.45)
    ov = np.zeros((224, 224, 4)); ov[np.kron(G, np.ones((16, 16), bool))] = [*mpl.colors.to_rgb(c), 0.55]
    ax[j].imshow(ov)
    for g in range(0, 225, 16):
        ax[j].axhline(g - .5, color="0.5", lw=0.3); ax[j].axvline(g - .5, color="0.5", lw=0.3)
    ax[j].set_xlabel(txt, fontsize=7)
ax[0].set_xlabel(", ".join(l.replace("_", " ") for l in labs), fontsize=6)
for a in ax: a.set_xticks([]); a.set_yticks([])
# fig.tight_layout(); save(fig, "fig_ch3_grid_coverage"); plt.show()


# ---- cell 28 ------------------------------------------------------------
# C11
fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.55, 2.2))
# ax.bar(site.index, site.MEL_share, color=plotting.C_FOCUS)
ax.bar(site.index, site.MEL_share, color=plotting.C_MEL)
for i, (s, r) in enumerate(site.iterrows()):
    ax.text(i, r.MEL_share + .005, f"{r.MEL_share:.3f}\nn={r.images}", ha="center", fontsize=6)
ax.set(ylabel="melanoma share", ylim=(0, site.MEL_share.max() * 1.3)); ax.tick_params(axis="x", labelsize=7)
fig.tight_layout(); save(fig, "fig_appA_sites"); plt.show()

# C12
LOC = pd.read_csv(TAB / "ch5_localisation_test.csv"); L1 = LOC[LOC.model == LOC.model.iloc[0]]
fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.6, 2.2)); bins = np.arange(0, 197, 8)
# for g, c in [("NV", plotting.C_CONTEXT), ("MEL", plotting.C_FOCUS)]:
for g, c in [("NV", plotting.C_NV), ("MEL", plotting.C_MEL)]:
    v = L1[L1["gt"] == g].lesion_share * 196
    ax.hist(v, bins=bins, density=True, alpha=0.55, color=c, label=f"{g} (n={len(v)}, median {v.median():.0f})")
ax.set(xlabel="lesion size, grid cells of 196", ylabel="density"); ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); save(fig, "fig_appA_lesion_size"); plt.show()

# C13
D = pd.DataFrame([{"image_id": i, "gt": EV.loc[i].gt_label, "dens": HUM[i][LESION[i]].sum() / LESION[i].sum()}
                  for i in sorted(HUM)]).sort_values("dens").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.7, 2.2))
# ax.bar(D.index, D.dens, color=[plotting.C_FOCUS if g == "MEL" else plotting.C_CONTEXT for g in D["gt"]])
ax.bar(D.index, D.dens, color=[plotting.C_MEL if g == "MEL" else plotting.C_NV for g in D["gt"]])
ax.axhline(D.dens.median(), color="k", ls=":", lw=0.8)
ax.text(len(D) - .5, D.dens.median() + .02, f"median {D.dens.median():.2f}", ha="right", fontsize=6)
ax.set(xlabel="annotated image, sorted", ylabel="share of lesion cells in target", xticks=[], ylim=(0, 1))
# ax.legend([mpatches.Patch(color=plotting.C_FOCUS), mpatches.Patch(color=plotting.C_CONTEXT)], ["MEL", "NV"], frameon=False, fontsize=6, loc="upper left")
ax.legend([mpatches.Patch(color=plotting.C_MEL), mpatches.Patch(color=plotting.C_NV)], ["MEL", "NV"],
          frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(); save(fig, "fig_appA_target_density"); plt.show()
print(f"C13 target density median {D.dens.median():.3f}, n={len(D)}")


# ---- cell 29 ------------------------------------------------------------
try:
    import fitz
    read = lambda p: "".join(pg.get_text() for pg in fitz.open(p))
except ImportError:
    from pypdf import PdfReader
    read = lambda p: "".join(pg.extract_text() or "" for pg in PdfReader(p).pages)
OLD = ["lesion align", "concept align", "lesion only", "concept only", "les}", "con}", "λles", "λcon",
       "lambda_les", "lambda_con", "HA1", "HA2", "centre", "center baseline"]
for p in sorted(FIG.glob("fig_*.pdf")):
    t = read(p).replace("\n", " ")
    hits = [o for o in OLD if o.lower() in t.lower()]
    if hits: print(f"C14 REGENERATE {p.name}: {hits}")
print("C14 done. Files not listed are clean, or their text is not extractable (check those visually).")


# ---- cell 30 ------------------------------------------------------------
FOLD = data.fold_of_heldout()
held = [i for i in sorted(HUM) if i in FOLD]
pick = [max([i for i in held if EV.loc[i].gt_label == g],
            key=lambda i: len(data.group_masks(QC, i, LESION[i]))) for g in ("MEL", "NV")]
fig, ax = plt.subplots(2, 4, figsize=(TEXTWIDTH, 3.4))
for i, iid in enumerate(pick):
    r = EV.loc[iid]; L = LESION[iid]
    x = data.to_tensor(data.native(r.image_rel_path)); rgb = data.rgb224(r.image_rel_path)
    G = data.group_masks(QC, iid, L)
    mexp = models.f2(0.0, 1.0, FOLD[iid])
    ax[i, 0].imshow(rgb)
    for g, mm in G.items():
        outline(ax[i, 0], mm, GROUP_COLOR[g], lw=1.3)
    ax[i, 0].set_ylabel(r.gt_label, fontsize=7)
    for j, (nm, mdl) in enumerate([("unaligned", M0), ("global aligned", M5), ("expert aligned", mexp)]):
        mel = cams.class_cams(mdl, x)[0]
        overlay(ax[i, j + 1], rgb, mel)
        for g, mm in G.items():
            outline(ax[i, j + 1], mm, GROUP_COLOR[g], lw=0.8, ls="dashed")
        top = max(G, key=lambda g: metrics.energy_excess(mel, L, G[g]))
        ax[i, j + 1].set_title(f"{nm}\nMEL map, top: {top}", fontsize=5.5)
    del mexp
    for a in ax[i]: a.set_xticks([]); a.set_yticks([])
fig.legend(handles=[mpatches.Patch(color=GROUP_COLOR[g], label=g)
                    for g in sorted({g for i in pick for g in data.group_masks(QC, i, LESION[i])})],
           loc="lower center", ncol=4, fontsize=5.5, frameon=False)
fig.tight_layout(rect=[0, 0.06, 1, 1]); save(fig, "fig_ch6_group_examples"); plt.show()


# ---- cell 31 ------------------------------------------------------------
REG = []
def add(ch, comp, metric, a, b=None, fam=1, src="", test="Wilcoxon"):
    if b is None:
        d = pd.Series(a).dropna(); n = len(d)
        p = wilcoxon(d).pvalue if (d != 0).any() and n >= 5 else np.nan; diff = d.median()
    else:
        q = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1).dropna()
        r = metrics.paired(q.a, q.b); diff, p, n = r["diff"], r["p"], r["n"]
    REG.append(dict(chapter=ch, comparison=comp, metric=metric, n=n, median_diff=diff, p=p,
                    family=fam, test=test, source=src))
piv = lambda d, col, val, idx="image_id": d.pivot_table(index=idx, columns=col, values=val)

# ---- Chapter 5 ---------------------------------------------------------------
q = piv(LOC, "model", "loc_auc"); add(5, "global aligned vs unaligned", "localisation", q.aligned, q.unaligned, 1, "ch5_localisation_test.csv")
SW = pd.read_csv(TAB / "ch5_lambda_sweep.csv"); b0 = SW[SW["lambda"] == 0].iloc[0]
for _, r in SW[SW["lambda"] > 0].iterrows():
    for met in ["auc", "ap"]:
        REG.append(dict(chapter=5, comparison=f"λ_global {r['lambda']:g} vs 0", metric=f"test {met.upper()}", n=987,
                        median_diff=r[met] - b0[met], p=r[f"p_{met}"], family=9, test="bootstrap, one-sided",
                        source="ch5_lambda_sweep.csv"))
P01 = pd.read_csv(TAB / "ch5_lam01.csv")
for met in ["energy_in", "auc_in"]:
    q = piv(P01, "model", met); add(5, "λ_global 0.1 vs 5", met, q["lam0.1"], q["lam5"], 10, "ch5_lam01.csv")
HS = pd.read_csv(TAB / "ch5_head_selection.csv"); q = piv(HS, "k", "auc_in")
add(5, "3 selected heads vs all 12", "AUC_in", q[3], q[12], 1, "ch5_head_selection.csv")
OR = pd.read_csv(TAB / "ch5_head_oracle.csv").set_index("image_id")
add(5, "head oracle, in-sample vs uniform", "AUC_in", OR.oracle, OR.uniform, 2, "ch5_head_oracle.csv")
add(5, "head weights, held-out vs uniform", "AUC_in", OR.held_out, OR.uniform, 2, "ch5_head_oracle.csv")
RB = pd.read_csv(TAB / "app_robustness.csv")
for cond in ["shifted", "uniform"]:
    for mo in ["aligned", "unaligned"]:
        s = RB[(RB.cond == cond) & (RB.model == mo)].reset_index(drop=True)
        add(5, f"{cond}: heat vs area, {rn(mo)}", "heat share", s.share, s.area, 6, "app_robustness.csv")
ua = [RB[(RB.cond == "uniform") & (RB.model == m)].share.reset_index(drop=True) for m in ["aligned", "unaligned"]]
add(5, "uniform: global aligned vs unaligned", "heat in centre", ua[0], ua[1], 6, "app_robustness.csv (paired by row order)")
VC = pd.read_csv(TAB / "verify_controls.csv")
for mo in ["aligned", "unaligned"]:
    g = VC[VC.model == mo].set_index("image_id")
    add(5, f"background patch: heat vs area, {rn(mo)}", "heat on patch", g.bp_patch, g.patch_area, 6, "verify_controls.csv")
g = VC[VC.model == "unaligned"].set_index("image_id")
add(5, "lesion removed: NV map before vs after, unaligned", "heat on region", g.nv_removed, g.nv_orig, 6, "verify_controls.csv")
for mo in ["unaligned", "lesion aligned", "concept only"]:
    g = DL[DL.model == mo].set_index("image_id")
    add(5, f"deletion top vs random, {rn(mo)}", "logit gap drop", g["top 10%"], g["random 10%"], 3, "ch5_deletion_v2.csv")
    add(5, f"deletion clinician vs unannotated, {rn(mo)}", "logit gap drop", g["clinician (n)"], g["unannotated (n)"], 3, "ch5_deletion_v2.csv")
E9D = pd.read_csv(TAB / "ch5_deletion_v2_expert_ep9.csv").set_index("image_id")
add(5, "deletion top vs random, expert aligned epoch 9", "logit gap drop", E9D["top 10%"], E9D["random 10%"], 1, "ch5_deletion_v2_expert_ep9.csv")

# ---- Chapter 6 ---------------------------------------------------------------
q = piv(M61, "method", "auc_in")
for m in ["Diff-CAM", "FinerCAM 0.6", "random"]:
    add(6, f"{m} vs Grad-CAM", "AUC_in", q[m], q["Grad-CAM"], 3, "ch6_metrics.csv")
FG = pd.read_csv(TAB / "ch6_finercam_gamma_auc.csv")
qa = piv(FG[FG.model == "aligned"], "gamma", "auc")
for gm in [0.2, 0.4, 0.6, 0.8, 1.0]:
    add(6, f"FinerCAM γ {gm} vs 0, global aligned", "AUC_in", qa[gm], qa[0.0], 5, "ch6_finercam_gamma_auc.csv")
for gm in sorted(FG.gamma.unique()):
    q = piv(FG[FG.gamma == gm], "model", "auc")
    add(6, f"global aligned vs unaligned, γ {gm}", "AUC_in", q.aligned, q.unaligned, 6, "ch6_finercam_gamma_auc.csv")
rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for nm, m in [("unaligned", M0), ("aligned", M5)]:
        mel, nv = cams.class_cams(m, x)
        rows.append({"model": nm, "image_id": iid, "mel_in": metrics.energy_excess(mel, L, U),
                     "mel_grid": egrid(mel, U & L), "mel_auc": metrics.auc_in(mel, L, U),
                     "nv_in": metrics.energy_excess(nv, L, L & ~U) if r.gt_label == "NV" else np.nan,
                     "nv_grid": egrid(nv, L & ~U) if r.gt_label == "NV" else np.nan})
UA = pd.DataFrame(rows); UA.to_csv(TAB / "ch6_unaligned_vs_aligned.csv", index=False)
for met in ["mel_in", "mel_grid", "mel_auc", "nv_in", "nv_grid"]:
    q = piv(UA, "model", met); add(6, "global aligned vs unaligned", met, q.aligned, q.unaligned, 5, "ch6_unaligned_vs_aligned.csv")
for mo in GE.model.unique():
    grp = GE[(GE.model == mo) & (GE.group != "UNASSIGNED")].group.unique()
    for gname in list(grp) + ["UNASSIGNED"]:
        s = GE[(GE.model == mo) & (GE.group == gname)].energy_exc
        if len(s.dropna()) >= 5:
            add(6, f"group {gname} vs 0, {rn(mo)}", "energy excess", s.values, None, len(grp) + 1, "ch6_group_energy.csv", "Wilcoxon, one-sample")
LD = pd.read_csv(TAB / "ch5_reference_ladder.csv")
for met in ["auc_in", "energy_in"]:
    q = piv(LD, "ref", met)
    for mo in ["lesion aligned", "concept only"]:
        add(6, f"{rn(mo)} vs unaligned", met, q[mo], q.unaligned, 2, "ch5_reference_ladder.csv")
        add(6, f"{rn(mo)} vs darkness", met, q[mo], q.darkness, 2, "ch5_reference_ladder.csv")

# ---- Chapter 7 ---------------------------------------------------------------
F7 = pd.read_csv(TAB / "ch7_grounding.csv"); F9 = pd.read_csv(TAB / "ch7_ep9.csv"); G9D = pd.read_csv(TAB / "ch7_global_ep9.csv")
ALL7 = pd.concat([F7, F9, G9D]); MET7 = ["mel_in", "mel_grid", "mel_auc_in", "nv_rest_in", "gt_loc"]
SETS = ["HA1 only", "joint b1", "joint b5", "joint b20", "HA2 b1", "HA2 b5", "HA2 b20"]
for ref in ["control", "HA1 only"]:
    for st in [s for s in SETS if s != ref]:
    # for st in [s for s in SETS + ["control"] if s != ref]:
        for met in MET7:
            q = piv(ALL7[ALL7.setting.isin([st, ref])], "setting", met)
            if st in q and ref in q:
                add(7, f"{rn(st)} vs {rn(ref)}", met, q[st], q[ref], 7, "ch7_grounding.csv")
for a, b in [("HA2 b1 ep9", "control ep9"), ("HA2 b1 ep9", "global only ep9"), ("HA2 b1 ep9", "HA2 b1"),
             ("global only ep9", "control ep9")]:
    for met in ["mel_in", "mel_grid", "mel_auc_in", "gt_loc"]:
        q = piv(ALL7[ALL7.setting.isin([a, b])], "setting", met)
        if a in q and b in q:
            add(7, f"{rn(a)} vs {rn(b)}", met, q[a], q[b], 4, "ch7_ep9.csv, ch7_global_ep9.csv")
FX = pd.read_csv(TAB / "ch7_contrastive_expert.csv")
for meth in FX.method.unique():
    q = piv(FX[FX.method == meth], "model", "auc_in"); add(7, f"{meth}: expert vs global", "AUC_in", q.expert, q["global"], 3, "ch7_contrastive_expert.csv")
for mo in ["global", "expert"]:
    q = piv(FX[FX.model == mo], "method", "auc_in"); add(7, f"{mo}: FinerCAM vs Grad-CAM", "AUC_in", q["FinerCAM 0.6"], q["Grad-CAM"], 2, "ch7_contrastive_expert.csv")
TT = pd.read_csv(TAB / "ch17_test_expert_vs_global.csv"); q = piv(TT, "model", "loc")
add(7, "test images: expert vs global, epoch 9, fold 0", "localisation", q.expert, q["global"], 1, "ch17_test_expert_vs_global.csv")

ST = pd.DataFrame(REG)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
out = ST.assign(median_diff=ST.median_diff.round(3), p=ST.p.map(lambda v: f"{v:.2g}"), threshold=ST.threshold.map(lambda v: f"{v:.3g}"))
out.drop(columns="source").to_latex(TAB / "appD_statistics.tex", index=False, longtable=True, escape=True)
print(f"D15 {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())}")
print(out.groupby("chapter").size().to_string())
print(out[out.survives == "no"][["chapter", "comparison", "metric", "p", "threshold"]].to_string(index=False))


# ---- cell 32 ------------------------------------------------------------
F7a = pd.concat([pd.read_csv(TAB / "ch7_grounding.csv"), pd.read_csv(TAB / "ch7_ep9.csv"),
                 pd.read_csv(TAB / "ch7_global_ep9.csv")])
S7 = ["control", "control ep9", "HA1 only", "global only ep9", "joint b1", "joint b5", "joint b20",
      "HA2 b1", "HA2 b1 ep9", "HA2 b5", "HA2 b20"]
for st in S7:
    s = F7a[(F7a.setting == st)].nv_rest_in.dropna()
    if len(s) < 5: continue
    p = wilcoxon(s).pvalue
    print(f"NV {rn(st):35} n={len(s)} median {s.median():+.3f}  below 0 on {(s < 0).sum()}  p={p:.3g}")
q = F7a[F7a.setting.isin(["HA2 b1 ep9", "control ep9"])].pivot_table(index="image_id", columns="setting", values="nv_rest_in")
print("NV expert ep9 vs control ep9:", metrics.paired(q["HA2 b1 ep9"], q["control ep9"]))


# ---- cell 33 ------------------------------------------------------------
for st in S7:
    s = F7a[F7a.setting == st].nv_rest_in.dropna()
    if len(s) >= 5:
        add(7, f"NV map vs 0, {rn(st)}", "nv_rest_in", s.values, None, 10, "ch7_grounding.csv, ch7_ep9.csv", "Wilcoxon, one-sample")
q = F7a[F7a.setting.isin(["HA2 b1 ep9", "control ep9"])].pivot_table(index="image_id", columns="setting", values="nv_rest_in")
add(7, "NV map, expert only epoch 9 vs control epoch 9", "nv_rest_in", q["HA2 b1 ep9"], q["control ep9"], 1, "ch7_ep9.csv")


# ---- cell 34 ------------------------------------------------------------
# from matplotlib.collections import LineCollection
# from matplotlib.patches import Circle, Ellipse
# rng = np.random.default_rng(3)
# BG, BR, DK = "#E9CBB5", "#8B5A3C", "#4A2C1D"

# def lesion(ax, irregular):
#     t = np.linspace(0, 2 * np.pi, 200)
#     rad = 1 + (0.12 * np.sin(3 * t) + 0.08 * np.sin(7 * t + 1) if irregular else 0)
#     ax.fill(rad * np.cos(t), rad * np.sin(t), color="#C79A7A", zorder=0)
#     return t, rad
# def inside(x, y, t, rad):
#     a = np.arctan2(y, x) % (2 * np.pi); return np.hypot(x, y) < np.interp(a, t, rad) * 0.95

# def network(ax, irr):
#     t, rad = lesion(ax, irr); segs, lw = [], []
#     g = np.arange(-1.2, 1.25, 0.22)
#     for x0 in g:
#         for y0 in g:
#             for dx, dy in [(0.22, 0), (0, 0.22)]:
#                 p = np.array([[x0, y0], [x0 + dx, y0 + dy]])
#                 if irr: p += rng.normal(0, 0.05, p.shape)
#                 if inside(*p.mean(0), t, rad) and not (irr and rng.random() < 0.18):
#                     segs.append(p); lw.append(rng.uniform(0.6, 2.6) if irr else 1.1)
#     ax.add_collection(LineCollection(segs, colors=DK, linewidths=lw))

# def dots(ax, irr):
#     t, rad = lesion(ax, irr)
#     if irr:
#         pts = np.r_[rng.normal([0.45, 0.3], 0.18, (25, 2)), rng.uniform(-0.9, 0.9, (8, 2))]; sz = rng.uniform(4, 40, len(pts))
#     else:
#         a = np.linspace(0, 2 * np.pi, 16, endpoint=False); pts = np.c_[0.75 * np.cos(a), 0.75 * np.sin(a)]; sz = np.full(16, 22)
#     m = inside(pts[:, 0], pts[:, 1], t, rad); ax.scatter(*pts[m].T, s=sz[m], color=DK, zorder=2)

# def streaks(ax, irr):
#     t, rad = lesion(ax, irr)
#     a = np.sort(rng.uniform(0.2, 1.6, 7)) if irr else np.linspace(0, 2 * np.pi, 24, endpoint=False)
#     r0 = np.interp(a % (2 * np.pi), t, rad)
#     ax.add_collection(LineCollection([[[0.85 * r * np.cos(x), 0.85 * r * np.sin(x)], [1.18 * r * np.cos(x), 1.18 * r * np.sin(x)]]
#                                       for x, r in zip(a, r0)], colors=DK, linewidths=1.6))

# def blotch(ax, irr):
#     t, rad = lesion(ax, irr)
#     ax.add_patch(Ellipse((0.35, -0.25) if irr else (0, 0), 0.9 if irr else 0.7, 0.55 if irr else 0.7,
#                          angle=30 if irr else 0, color=DK, zorder=2))

# ROWS = [("pigment network", network), ("dots and globules", dots), ("streaks", streaks), ("blotch", blotch)]
# fig, ax = plt.subplots(len(ROWS), 2, figsize=(TEXTWIDTH * 0.55, TEXTWIDTH * 1.05))
# for i, (name, f) in enumerate(ROWS):
#     for j, irr in enumerate([False, True]):
#         a = ax[i, j]; a.set_facecolor(BG); a.set(xlim=(-1.45, 1.45), ylim=(-1.45, 1.45), aspect="equal", xticks=[], yticks=[])
#         f(a, irr)
#         if i == 0: a.set_title("regular" if j == 0 else "atypical", fontsize=8)
#     ax[i, 0].set_ylabel(name, fontsize=8)
# fig.tight_layout(); save(fig, "fig_ch2_structures"); plt.show()


# ---- cell 35 ------------------------------------------------------------
H = pd.read_csv(MELNV / "ham_mel_nv_clean.csv"); H["image_id"] = H.image_id.astype(str).str.strip()
ann = sorted(set(QC.Image_ID))
print("4. original split of the 34:", H[H.image_id.isin(ann)].split.value_counts().to_dict())
sheet = next(iter(REPO.rglob("08b_annotation_sheet_blank.csv")), None)
if sheet:
    S8 = pd.read_csv(sheet); print("1-3. sheet columns:", list(S8.columns)); print(S8.head(3).to_string())


# ---- cell 36 ------------------------------------------------------------
C1 = pd.read_csv(REPO / "results/final/clinician_study_cases.csv")
print("8.", C1.groupby("case").agg(n=("image_id", "size"), conf_min=("conf", "min"), conf_max=("conf", "max")).round(3).to_string())
print("8. any case below 0.9:", bool((C1.conf < 0.9).any()))


# ---- cell 37 ------------------------------------------------------------
F7a = pd.concat([pd.read_csv(TAB / f) for f in ["ch7_grounding.csv", "ch7_ep9.csv", "ch7_global_ep9.csv"]])
S7 = ["control", "control ep9", "HA1 only", "global only ep9", "joint b1", "joint b5", "joint b20",
      "HA2 b1", "HA2 b1 ep9", "HA2 b5", "HA2 b20"]
rows = []
for st in S7:
    s = F7a[F7a.setting == st].nv_rest_in.dropna()
    if len(s) == 0: continue
    rows.append({"setting": rn(st), "n": len(s), "median [IQR]": miqr(s), "below 0": int((s < 0).sum()),
                 "p vs 0": f"{wilcoxon(s).pvalue:.2g}" if len(s) >= 5 else "n/a"})
NVT = pd.DataFrame(rows); NVT.to_csv(TAB / "table_nv_map.csv", index=False)
NVT.to_latex(TAB / "table_nv_map.tex", index=False, escape=True)
print("11. NV map energy excess on lesion minus melanoma structures, held-out nevi\n", NVT.to_string(index=False))


# ---- cell 38 ------------------------------------------------------------
# item 8: confirm there were no false negatives
pa = models.predict_mel(M5, X_TEST)
print("8. aligned model, test FN at 0.5:", int(((Y_MEL == 1) & (pa < 0.5)).sum()),
      " FP at 0.5:", int(((Y_MEL == 0) & (pa >= 0.5)).sum()), " FP with p >= 0.9:", int(((Y_MEL == 0) & (pa >= 0.9)).sum()))

# items 1-3: selection code in 08b
import json
nb = json.load(open(REPO / "notebooks/mel_nv/08b_clinician_model_choice.ipynb"))
for c in nb["cells"]:
    s = "".join(c["source"])
    if any(k in s for k in ["FN MEL (missed", "FP MEL (NV", "near decision boundary", "TP_MEL", "checkpoint", "ckpt"]):
        print("-" * 60); print(s[:1800])

# example pages of both study PDFs, for the appendix
# import fitz
import pymupdf as fitz
for f, out in [("clinician_study.pdf", "fig_appK_round1_part1"), ("clinician_study.pdf", "fig_appK_round1_part2"),
               ("clinician_study_round2.pdf", "fig_appK_round2")]:
    doc = fitz.open(OUT / f)
    page = {"fig_appK_round1_part1": 3, "fig_appK_round1_part2": 15, "fig_appK_round2": 2}[out]
    doc.select([page]); doc.save(FIG / f"{out}.pdf"); print("saved", out, "from page", page + 1, "of", f)


# ---- cell 39 ------------------------------------------------------------
s = "".join(next(c for c in nb["cells"] if "CELL 5" in "".join(c["source"]))["source"])
print(s[1800:5000])


# ---- cell 40 ------------------------------------------------------------
import json
nb5 = json.load(open(REPO / "notebooks/mel_nv/05b_difficult_cases_plotly.ipynb"))
for c in nb5["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "code" and any(k in s for k in ["curation_group", "fp_mel_confident", "difficult_correct_mel", "nearest_mel_centroid"]):
        print("-" * 60); print(s[:3000])


# ---- cell 41 ------------------------------------------------------------
s9b = next("".join(c["source"]) for c in nb5["cells"] if "9b. Export clinician" in "".join(c["source"]))
print(s9b[s9b.find("# 4. All FN MEL"):s9b.find("# 4. All FN MEL") + 2500])
for c in nb5["cells"]:
    s = "".join(c["source"])
    if '"correctness"' in s or "'correctness'" in s or "FP MEL" in s and "np.where" in s:
        print("-" * 60); print(s[:1500]); break


# ---- cell 42 ------------------------------------------------------------
for c in nb5["cells"]:
    s = "".join(c["source"])
    if "FEATURE_MODE =" in s or "low_confidence" in s and "=" in s.split("low_confidence")[1][:5]:
        print("-" * 60); print(s[:1200])


# ---- cell 43 ------------------------------------------------------------
import re
for c in nb5["cells"]:
    for line in "".join(c["source"]).splitlines():
        if re.search(r'^\s*FEATURE_MODE\s*=|\["low_confidence"\]\s*=|\["confidence_margin_to_0_5"\]\s*=|LOW_CONF\w*\s*=|CONF\w*THRESH\w*\s*=', line):
            print(line.strip())


# ---- cell 44 ------------------------------------------------------------
import re
from scipy.stats import wilcoxon

# ---------------------------------------------------------------- formatting
TOK = [("λ_global", r"$\lambda_{\mathrm{global}}$"), ("λ_expert", r"$\lambda_{\mathrm{expert}}$"),
       ("λ", r"$\lambda$"), ("γ", r"$\gamma$"), ("±", r"$\pm$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
       ("×", r"$\times$"), ("–", "-"), ("—", "-"), ("’", "'")]
ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
       "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
TOKRE = re.compile("(" + "|".join(re.escape(t) for t, _ in TOK) + ")")
TOKD = dict(TOK)

def tx(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return "--"
    return "".join(TOKD[p] if p in TOKD else "".join(ESC.get(ch, ch) for ch in p) for p in TOKRE.split(str(s)))
fi = lambda v: "--" if pd.isna(v) else str(int(round(float(v))))
f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
def fp(p):
    if pd.isna(p): return "n/a"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{p:.3f}"
def mi(s):
    s = pd.Series(s).dropna()
    return "--" if s.empty else f"{s.median():.3f} [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]"
def pm(s):
    s = pd.Series(s).dropna()
    return "--" if s.empty else f"{s.mean():.3f} $\\pm$ {s.std(ddof=1):.3f}"
lab = lambda s: str(s).replace("preminent", "prominent").replace("_", " ")

# ---------------------------------------------------------------- writer
CHANGED = []
def write(name, header, cells, colfmt, longtable=False, caption=None, label=None, small=False):
    h = " & ".join(header) + r" \\"
    body = [r if isinstance(r, str) else " & ".join(r) + r" \\" for r in cells]
    if longtable:
        n = len(header)
        L = [r"\begingroup\small" if small else None, rf"\begin{{longtable}}{{{colfmt}}}",
             rf"\caption{{{tx(caption)}}}\label{{{label}}}\\", r"\toprule", h, r"\midrule", r"\endfirsthead",
             rf"\multicolumn{{{n}}}{{l}}{{\small\itshape continued from previous page}}\\",
             r"\toprule", h, r"\midrule", r"\endhead",
             r"\midrule", rf"\multicolumn{{{n}}}{{r}}{{\small\itshape continued on next page}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot", *body, r"\end{longtable}", r"\endgroup" if small else None]
    else:
        L = [rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule", h, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    txt = "\n".join(l for l in L if l) + "\n"
    bad = sorted({c for c in txt if ord(c) > 127})
    assert not bad, f"{name}: non-ASCII characters {bad}"
    (TAB / name).write_text(txt, encoding="ascii")
    CHANGED.append(name); print(f"written {name}: {sum(not isinstance(c, str) for c in cells)} rows")

# ---------------------------------------------------------------- 1. f2_full
F7 = pd.read_csv(TAB / "ch7_grounding.csv"); F7 = F7[F7.alpha.notna()]
C7 = pd.read_csv(TAB / "ch7_classification.csv")
GRID = [(1e-8, 0, "control"), (5, 0, "global only"), (5, 1, "joint"), (5, 5, "joint"), (5, 20, "joint"),
        (0, 1, "expert only"), (0, 5, "expert only"), (0, 20, "expert only")]
lam = lambda v: r"$10^{-8}$" if np.isclose(v, 1e-8, rtol=0, atol=1e-12) else f"{v:g}"
sel = lambda d, a, b: d[np.isclose(d.alpha, a, rtol=0, atol=1e-12) & np.isclose(d.beta, b, rtol=0, atol=1e-12)]
MET = [("mel_auc_in", r"$\mathrm{AUC}_{\mathrm{in}}$"), ("mel_in", r"$E_{\mathrm{in}}$"),
       ("mel_grid", r"$E_{\mathrm{grid}}$"), ("nv_rest_in", r"$E_{\mathrm{in}}^{\mathrm{NV}}$"),
       ("gt_loc", "localisation"), ("corr_in", r"$r_{\mathrm{in}}$")]
cells = []
for a, b, nm in GRID:
    g, c = sel(F7, a, b), sel(C7, a, b)
    assert len(c) == 5, (a, b, len(c))
    cells.append([nm, lam(a), lam(b)] + [pm(g[k]) for k, _ in MET] + [pm(c.auc), pm(c.ap)])
write("f2_full.tex", ["setting", r"$\lambda_{\mathrm{global}}$", r"$\lambda_{\mathrm{expert}}$"]
      + [h for _, h in MET] + ["test AUC", "test AP"], cells, "lrr" + "c" * 8)

# ---------------------------------------------------------------- 2. appA_labels
LB = pd.read_csv(TAB / "appA_labels.csv")
CH = {"MEL": r"melanoma-indicative ($T_{\mathrm{MEL}}$)", "NV": r"benign ($T_{\mathrm{NV}}$)",
      "OTHER": "not used", "COARSE": "not used"}
cells = [[tx(lab(r.label)), CH[r.label_group], fi(r["on MEL"]), fi(r["on NV"]), fi(r.images),
          "--" if r["morphology group"] == "none" else tx(r["morphology group"])] for _, r in LB.iterrows()]
write("appA_labels.tex", ["label", "target channel", "on MEL", "on NV", "images", "morphology group"],
      cells, "p{5cm}lrrrl")

# ---------------------------------------------------------------- 3. app_case_manifest
GNAME = {"TP_MEL": "typical melanoma", "TN_NV": "typical nevus", "FN_MEL": "missed melanoma",
         "FP_MEL": "nevus classified as melanoma", "DIFFICULT_CORRECT_MEL": "melanoma near boundary"}
GRP = {}
d8 = REPO / "outputs/mel_nv/clinician_model_choice_08b"
for f, nm in [("08b_typical_mel.csv", "typical melanoma"), ("08b_typical_nv.csv", "typical nevus"),
              ("08b_fn_mel.csv", "missed melanoma"), ("08b_fp_mel.csv", "nevus classified as melanoma"),
              ("08b_difficult_correct_mel.csv", "melanoma near boundary")]:
    p = d8 / "csv" / f
    if p.exists():
        for i in pd.read_csv(p).image_id.astype(str).str.strip(): GRP.setdefault(i, nm)
raw = pd.read_csv(QC_CSV); raw["Image_ID"] = raw.Image_ID.astype(str).str.strip()
if "case_type" in raw.columns:
    for i, c in raw.dropna(subset=["case_type"]).drop_duplicates("Image_ID")[["Image_ID", "case_type"]].values:
        GRP.setdefault(i, GNAME.get(str(c), str(c)))
if d8.exists():
    for p in d8.rglob("*.csv"):
        try: d = pd.read_csv(p)
        except Exception: continue
        idc = next((c for c in d.columns if c.lower() in ("image_id", "image")), None)
        for col in d.columns:
            if idc and d[col].astype(str).isin(GNAME).any():
                for i, c in d[[idc, col]].astype(str).values:
                    if c in GNAME: GRP.setdefault(i.strip().replace(".jpg", ""), GNAME[c])
qp = raw[raw.qc_pass.astype(bool)]
ids = sorted(raw.Image_ID.unique())
cells = []
for i in ids:
    labs = sorted({lab(l) for l in qp[qp.Image_ID == i].label})
    cells.append([tx(i), EV.loc[i].gt_label if i in EV.index else "--", tx(GRP.get(i, "not recorded")),
                  str(int(FOLD[i])) if i in FOLD else "--", fi(len(qp[qp.Image_ID == i])),
                  tx(", ".join(labs)), "yes" if i in HUM else "no"])
write("app_case_manifest.tex", ["image", "diagnosis", "selection group", "fold", "masks", "labels", "MEL target"],
      cells, "llp{2.8cm}rrp{6cm}l", longtable=True, small=True, label="tab:app-case-manifest",
      caption="The 34 annotated images: diagnosis, selection group, fold, labels, and whether a melanoma structure target exists.")
print("   selection group not recorded for:", [i for i in ids if i not in GRP])

# ---------------------------------------------------------------- 4. appD_statistics
ST = pd.read_csv(TAB / "appD_statistics.csv")
EIN, EGR, AUC = r"$E_{\mathrm{in}}$", r"$E_{\mathrm{grid}}$", r"$\mathrm{AUC}_{\mathrm{in}}$"
METN = {"energy_in": EIN, "mel_in": EIN + " MEL", "mel_grid": EGR + " MEL", "auc_in": AUC, "AUC_in": AUC,
        "mel_auc_in": AUC + " MEL", "mel_auc": AUC + " MEL", "nv_in": r"$E_{\mathrm{in}}^{\mathrm{NV}}$",
        "nv_rest_in": r"$E_{\mathrm{in}}^{\mathrm{NV}}$", "nv_grid": r"$E_{\mathrm{grid}}^{\mathrm{NV}}$",
        "gt_loc": "localisation"}
cells = [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
          fp(r.p), fi(r.family), fp(r.threshold), tx(r.survives)] for _, r in ST.iterrows()]
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      cells, "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")

# ---------------------------------------------------------------- 5. bare tabulars
S = pd.read_csv(TAB / "appA_sites.csv")
cells = [[tx(r.dataset), fi(r.images), fi(r.MEL), fi(r.NV), f3(r.MEL_share)] for _, r in S.iterrows()]
cells += [r"\midrule", ["total", fi(S.images.sum()), fi(S.MEL.sum()), fi(S.NV.sum()), f3(S.MEL.sum() / S.images.sum())]]
write("appA_sites.tex", ["source", "images", "MEL", "NV", "MEL share"], cells, "lrrrr")

ORD = {"train": 0, "val": 1, "held-out": 2, "test": 3}
for tag in ["original", "final_fold0"]:
    d = pd.read_csv(TAB / f"appA_splits_{tag}.csv")
    d = d.assign(o=d.split.map(ORD)).sort_values(["o", "gt_label"])
    cells = [[tx({"val": "validation"}.get(r.split, r.split)), r.gt_label, fi(r.images), fi(r.lesions),
              f"{r.img_per_lesion:.2f}"] for _, r in d.iterrows()]
    write(f"appA_splits_{tag}.tex", ["split", "class", "images", "lesions", "images per lesion"], cells, "llrrr")

F7a = pd.concat([pd.read_csv(TAB / f) for f in ["ch7_grounding.csv", "ch7_ep9.csv", "ch7_global_ep9.csv"]])
S7 = [("control", "control"), ("control ep9", "control, epoch 9"), ("HA1 only", "global only"),
      ("global only ep9", "global only, epoch 9"), ("joint b1", "joint, λ_expert 1"), ("joint b5", "joint, λ_expert 5"),
      ("joint b20", "joint, λ_expert 20"), ("HA2 b1", "expert only, λ_expert 1"),
      ("HA2 b1 ep9", "expert only, λ_expert 1, epoch 9"), ("HA2 b5", "expert only, λ_expert 5"),
      ("HA2 b20", "expert only, λ_expert 20")]
cells = []
for key, nm in S7:
    s = F7a[F7a.setting == key].nv_rest_in.dropna()
    if len(s): cells.append([tx(nm), fi(len(s)), mi(s), fi((s < 0).sum()), fp(wilcoxon(s).pvalue)])
write("table_nv_map.tex", ["setting", "n", "median [IQR]", "below 0", "p"], cells, "lrcrr")

M61 = pd.read_csv(TAB / "ch6_metrics.csv")
ROWS = [("auc_in", r"in-lesion AUC"), ("energy_in", r"energy excess, lesion"),
        ("energy_grid", "energy excess, whole grid"), ("topk_exc", "top-k excess"),
        ("struct_hit", "structures hit by a top-k cell"), ("peak_dist", "distance hottest cell to target"),
        ("in_lesion", "heat inside lesion")]
METH = [("Grad-CAM", "Grad-CAM"), ("Diff-CAM", "Diff-CAM"), ("FinerCAM 0.6", r"FinerCAM ($\gamma = 0.6$)"), ("random", "random")]
cells = [[tx(nm)] + [mi(M61[M61.method == m][k]) for m, _ in METH] for k, nm in ROWS]
write("table_metric_validity.tex", ["metric"] + [h for _, h in METH], cells, "lcccc")

print("\nchanged:", CHANGED)
print("unchanged: app_hyperparameters.tex, data_*.tex (not used in the thesis)")


# ---- cell 45 ------------------------------------------------------------
G9 = REPO / "external/checkpoints_f2/ha1only_ep9"
G9D = pd.read_csv(TAB / "ch7_global_ep9.csv").set_index("image_id")
for k in range(5):
    m = models.load_ckpt(G9 / f"fold{k}.pth", cache=False)
    for iid in [i for i in held if FOLD[i] == k and EV.loc[i].gt_label == "NV"]:
        r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]
        nv = cams.class_cams(m, data.to_tensor(data.native(r.image_rel_path)))[1]
        G9D.loc[iid, "nv_rest_in"] = metrics.energy_excess(nv, L, L & ~U)
    del m
G9D.reset_index().to_csv(TAB / "ch7_global_ep9.csv", index=False)

s = G9D.nv_rest_in.dropna()
p = wilcoxon(s).pvalue
print(f"global only, epoch 9: n={len(s)}  median {s.median():+.3f}  IQR [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]  "
      f"below 0 on {(s < 0).sum()}  p={p:.2g}  survives at 0.0045: {p < 0.05 / 11}")


# ---- cell 46 ------------------------------------------------------------
import re
from scipy.stats import wilcoxon

# ---------------------------------------------------------------- names and formatting
NEW = {"control": "control", "control ep9": "control, epoch 9", "HA1 only": "global only",
       "global only ep9": "global only, epoch 9", "joint b1": "joint, λ_expert 1",
       "joint b5": "joint, λ_expert 5", "joint b20": "joint, λ_expert 20",
       "HA2 b1": "expert only, λ_expert 1", "HA2 b1 ep9": "expert only, λ_expert 1, epoch 9",
       "HA2 b5": "expert only, λ_expert 5", "HA2 b20": "expert only, λ_expert 20"}
rn = lambda s: NEW.get(s, s)
TOK = [("λ_global", r"$\lambda_{\mathrm{global}}$"), ("λ_expert", r"$\lambda_{\mathrm{expert}}$"),
       ("λ", r"$\lambda$"), ("γ", r"$\gamma$"), ("±", r"$\pm$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
       ("×", r"$\times$"), ("–", "-"), ("—", "-"), ("’", "'")]
ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
       "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
TOKRE = re.compile("(" + "|".join(re.escape(t) for t, _ in TOK) + ")"); TOKD = dict(TOK)
def tx(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): return "--"
    return "".join(TOKD[p] if p in TOKD else "".join(ESC.get(c, c) for c in p) for p in TOKRE.split(str(s)))
fi = lambda v: "--" if pd.isna(v) else str(int(round(float(v))))
f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
def fp(p):
    if pd.isna(p): return "n/a"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{p:.3f}"
def mi(s):
    s = pd.Series(s).dropna()
    return "--" if s.empty else f"{s.median():.3f} [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]"

def write(name, header, cells, colfmt, longtable=False, caption=None, label=None, small=False):
    h = " & ".join(header) + r" \\"
    body = [r if isinstance(r, str) else " & ".join(r) + r" \\" for r in cells]
    if longtable:
        n = len(header)
        L = [r"\begingroup\small" if small else None, rf"\begin{{longtable}}{{{colfmt}}}",
             rf"\caption{{{tx(caption)}}}\label{{{label}}}\\", r"\toprule", h, r"\midrule", r"\endfirsthead",
             rf"\multicolumn{{{n}}}{{l}}{{\small\itshape continued from previous page}}\\",
             r"\toprule", h, r"\midrule", r"\endhead",
             r"\midrule", rf"\multicolumn{{{n}}}{{r}}{{\small\itshape continued on next page}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot", *body, r"\end{longtable}", r"\endgroup" if small else None]
    else:
        L = [rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule", h, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    txt = "\n".join(l for l in L if l) + "\n"
    bad = sorted({c for c in txt if ord(c) > 127})
    assert not bad, f"{name}: non-ASCII {bad}"
    (TAB / name).write_text(txt, encoding="ascii"); print(f"written {name}")

# ---------------------------------------------------------------- NV rows, family 11
F7a = pd.concat([pd.read_csv(TAB / f) for f in ["ch7_grounding.csv", "ch7_ep9.csv", "ch7_global_ep9.csv"]])
S7 = ["control", "control ep9", "HA1 only", "global only ep9", "joint b1", "joint b5", "joint b20",
      "HA2 b1", "HA2 b1 ep9", "HA2 b5", "HA2 b20"]
NVS = {st: F7a[F7a.setting == st].nv_rest_in.dropna() for st in S7}
NVS = {k: v for k, v in NVS.items() if len(v)}
FAM = len(NVS)
print(f"NV settings with data: {FAM}, threshold {0.05 / FAM:.4f}")

ST = pd.read_csv(TAB / "appD_statistics.csv")
ST = ST[~ST.comparison.astype(str).str.startswith("NV map")]
rows = [dict(chapter=7, comparison=f"NV map vs 0, {rn(st)}", metric="nv_rest_in", n=len(v),
             median_diff=v.median(), p=wilcoxon(v).pvalue, family=FAM, test="Wilcoxon, one-sample",
             source="ch7_grounding.csv, ch7_ep9.csv, ch7_global_ep9.csv") for st, v in NVS.items()]
q = F7a[F7a.setting.isin(["HA2 b1 ep9", "control ep9"])].pivot_table(index="image_id", columns="setting", values="nv_rest_in").dropna()
r = metrics.paired(q["HA2 b1 ep9"], q["control ep9"])
rows.append(dict(chapter=7, comparison="NV map, expert only epoch 9 vs control epoch 9", metric="nv_rest_in",
                 n=r["n"], median_diff=r["diff"], p=r["p"], family=1, test="Wilcoxon",
                 source="ch7_ep9.csv, ch7_grounding.csv"))
ST = pd.concat([ST, pd.DataFrame(rows)], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
print(f"appD: {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())}")

# ---------------------------------------------------------------- re-export both tables
EIN, EGR, AUC = r"$E_{\mathrm{in}}$", r"$E_{\mathrm{grid}}$", r"$\mathrm{AUC}_{\mathrm{in}}$"
METN = {"energy_in": EIN, "mel_in": EIN + " MEL", "mel_grid": EGR + " MEL", "auc_in": AUC, "AUC_in": AUC,
        "mel_auc_in": AUC + " MEL", "mel_auc": AUC + " MEL", "nv_in": r"$E_{\mathrm{in}}^{\mathrm{NV}}$",
        "nv_rest_in": r"$E_{\mathrm{in}}^{\mathrm{NV}}$", "nv_grid": r"$E_{\mathrm{grid}}^{\mathrm{NV}}$",
        "gt_loc": "localisation"}
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fp(r.p), fi(r.family), fp(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")

write("table_nv_map.tex", ["setting", "n", "median [IQR]", "below 0", "p"],
      [[tx(rn(st)), fi(len(v)), mi(v), fi((v < 0).sum()), fp(wilcoxon(v).pvalue)] for st, v in NVS.items()],
      "lrcrr")
print("\nNV table rows:")
for st, v in NVS.items():
    print(f"  {rn(st):36} n={len(v)} {mi(v)}  below 0 {int((v < 0).sum())}  p={wilcoxon(v).pvalue:.2g}")


# ---- cell 47 ------------------------------------------------------------
# """Hybrid Dermoscopic Structure Schematics (Best of Claude + GPT + Ref Fixes).

# Outputs:
#   fig_ch2_structures_hybrid.pdf / .png
#   fig_appA_labels_visual_hybrid.pdf / .png
# """
# MODEL_TAG = "hybrid"

# import copy
# import zlib
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection, PatchCollection
# from matplotlib.colors import to_rgb, to_rgba
# from matplotlib.patches import Circle, Polygon
# from scipy.ndimage import gaussian_filter
# from scipy.spatial import Voronoi
# from scipy.special import expit

# OUTPUT_DIR = Path(".")

# # Palette
# BROWN, MID_BROWN, DARK, BLACK = "#7A4A24", "#925B2B", "#3E2410", "#170D06"
# REDS = ("#B5302C", "#C8403A", "#9E2A2B")
# GRAINS = ("#56657E", "#3E4553", "#77849A")
# TONES = {  # (lesion centre, feathered rim)
#     "tan": ("#D9B07F", "#EED6BC"),
#     "pink": ("#EFC3BA", "#F7DFDA"),
#     "warm": ("#D6A26C", "#EED2B0"),
# }

# # Raster grid in data coordinates
# EXT = (-1.35, 1.35, -1.25, 1.25)
# NX, NY = 540, 500
# X, Y = np.meshgrid(np.linspace(*EXT[:2], NX), np.linspace(*EXT[2:], NY))
# rng = np.random.default_rng


# # Generic helpers
# def save(fig, base_name):
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     for ext in ("pdf", "png"):
#         fig.savefig(
#             OUTPUT_DIR / f"{base_name}_{MODEL_TAG}.{ext}",
#             dpi=300,
#             bbox_inches="tight",
#         )


# def mix(c1, c2, t):
#     a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
#     return a + (b - a) * t


# def noise(rs, sigma):
#     n = gaussian_filter(rs.normal(size=(NY, NX)), sigma)
#     return n / (n.std() + 1e-12)


# def poisson(rs, dmin, n_try=6000, bx=1.25, by=1.15):
#     """Dart throwing with a spatially varying minimum distance dmin(p)."""
#     pts = np.empty((0, 2))
#     for _ in range(n_try):
#         p = rs.uniform((-bx, -by), (bx, by))
#         if len(pts) == 0 or np.min(np.sum((pts - p) ** 2, 1)) > dmin(p) ** 2:
#             pts = np.vstack([pts, p])
#     return pts


# def ray(c, ang, f, thr, above=True):
#     """Distance along a ray from c where field f first crosses thr."""
#     s = np.linspace(0, 1.9, 380)
#     v = f(c[0] + s * np.cos(ang), c[1] + s * np.sin(ang))
#     hit = v > thr if above else v < thr
#     return s[np.argmax(hit)] if hit.any() else s[-1]


# def union(*fs):
#     return lambda x, y: np.maximum.reduce([f(x, y) for f in fs])


# def ribbon(pts, half_w):
#     d = np.gradient(pts, axis=0)
#     d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
#     n = np.c_[-d[:, 1], d[:, 0]]
#     w = np.broadcast_to(half_w, (len(pts),))[:, None]
#     return np.vstack([pts + n * w, (pts - n * w)[::-1]])


# def var_line(ax, pts, lw, colour, clip=None, z=4, alpha=1.0):
#     """Polyline with per-vertex width (points). Round caps give smooth tapering."""
#     segs = np.stack([pts[:-1], pts[1:]], 1)
#     lw = np.broadcast_to(np.asarray(lw, float), (len(pts),))
#     lc = LineCollection(
#         segs,
#         linewidths=0.5 * (lw[:-1] + lw[1:]),
#         colors=colour,
#         alpha=alpha,
#         capstyle="round",
#         joinstyle="round",
#         zorder=z,
#     )
#     if clip is not None:
#         lc.set_clip_path(clip)
#     ax.add_collection(lc)


# def add_polys(ax, L, polys, fcs, z=4, ec="none", lw=0.0):
#     pc = PatchCollection(
#         [Polygon(p, closed=True) for p in polys],
#         facecolors=fcs,
#         edgecolors=ec,
#         linewidths=lw,
#         zorder=z,
#     )
#     pc.set_clip_path(L.clip)
#     ax.add_collection(pc)


# def glob_shape(c, r, rs, irr, k=(2, 3, 4, 5)):
#     th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
#     rr = np.ones_like(th)
#     for kk in k:
#         rr += irr * rs.uniform(0.3, 1) / kk**0.5 * np.sin(kk * th + rs.uniform(0, 2 * np.pi))
#     rr = r * np.clip(rr, 0.4, None)
#     return np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)]


# class Blob:
#     """Organic soft-edged shape. Callable as a field in [0, 1]. Exposes its outline."""

#     def __init__(self, rs, c, rx, ry, irr=0.25, soft=0.05, angle=None, k=(2, 3, 5)):
#         self.c, self.rx, self.ry, self.soft, self.k = np.asarray(c, float), rx, ry, soft, k
#         self.ang = rs.uniform(0, np.pi) if angle is None else angle
#         self.a = irr * rs.uniform(0.5, 1, len(k)) / np.sqrt(np.arange(1, len(k) + 1))
#         self.ph = rs.uniform(0, 2 * np.pi, len(k))

#     def rr(self, th):
#         r = np.ones_like(th, dtype=float)
#         for a, k, p in zip(self.a, self.k, self.ph):
#             r = r + a * np.sin(k * th + p)
#         return np.clip(r, 0.25, None)

#     def __call__(self, x, y):
#         dx, dy = np.asarray(x) - self.c[0], np.asarray(y) - self.c[1]
#         ca, sa = np.cos(self.ang), np.sin(self.ang)
#         u, v = (ca * dx + sa * dy) / self.rx, (-sa * dx + ca * dy) / self.ry
#         return expit((self.rr(np.arctan2(v, u)) - np.hypot(u, v)) / self.soft)

#     def scaled(self, s, soft=None):
#         b = copy.copy(self)
#         b.rx, b.ry = self.rx * s, self.ry * s
#         b.soft = self.soft if soft is None else soft
#         return b

#     def outline(self, s=1.0, n=240):
#         th = np.linspace(0, 2 * np.pi, n)
#         r = self.rr(th) * s
#         u, v = r * np.cos(th) * self.rx, r * np.sin(th) * self.ry
#         ca, sa = np.cos(self.ang), np.sin(self.ang)
#         return np.c_[self.c[0] + ca * u - sa * v, self.c[1] + sa * u + ca * v]


# class Lesion:
#     def __init__(self, rs, ax):
#         self.ax, self.ph = ax, rs.uniform(0, 2 * np.pi, 3)

#     def r(self, th):
#         p = self.ph
#         return (
#             1
#             + 0.07 * np.sin(3 * th + p[0])
#             + 0.035 * np.sin(5 * th + p[1])
#             + 0.015 * np.sin(8 * th + p[2])
#         )

#     def rho(self, x, y):
#         """Normalised radius. Below 1 means inside the lesion."""
#         yy = np.asarray(y) / 0.88
#         return np.hypot(x, yy) / self.r(np.arctan2(yy, x))

#     def fade(self, x, y, inner=0.55, outer=1.0):
#         return np.clip((outer - self.rho(x, y)) / (outer - inner), 0, 1)


# def paint(L, colour, field, z=1, tex=None, amp=0.0):
#     """Raster layer. Alpha = field x feathered lesion core."""
#     rgb = (
#         np.broadcast_to(np.array(to_rgb(colour)), (NY, NX, 3))
#         if isinstance(colour, str)
#         else colour
#     )
#     if tex is not None:
#         rgb = rgb + amp * tex[..., None]
#     a = np.clip(np.broadcast_to(field, (NY, NX)) * L.core, 0, 1)
#     L.ax.imshow(
#         np.dstack([np.clip(rgb, 0, 1), a]),
#         extent=EXT,
#         origin="lower",
#         interpolation="bilinear",
#         zorder=z,
#     )


# def panel(ax, seed=0, tone="tan"):
#     """Feathered lesion base. Returns a Lesion with .clip (Polygon) and .core (soft mask)."""
#     rs = rng(seed)
#     L = Lesion(rs, ax)
#     ax.set_xlim(*EXT[:2])
#     ax.set_ylim(*EXT[2:])
#     ax.set_aspect("equal")
#     ax.axis("off")
#     R = L.rho(X, Y)
#     L.core = gaussian_filter((R < 0.96).astype(float), 4)
#     body = gaussian_filter((R < 0.99).astype(float), 5)
#     halo = 0.6 * gaussian_filter((R < 1.06).astype(float), 14)
#     centre, rim = TONES[tone]
#     rgb = (
#         mix(rim, centre, np.clip(1.3 * (1 - R), 0, 1)[..., None])
#         + 0.03 * noise(rs, 10)[..., None]
#     )
#     ax.imshow(
#         np.dstack([np.clip(rgb, 0, 1), np.clip(np.maximum(body, halo), 0, 1)]),
#         extent=EXT,
#         origin="lower",
#         interpolation="bilinear",
#         zorder=0,
#     )
#     th = np.linspace(0, 2 * np.pi, 360)
#     r = 0.97 * L.r(th)
#     L.clip = Polygon(
#         np.c_[r * np.cos(th), 0.88 * r * np.sin(th)],
#         closed=True,
#         fc="none",
#         ec="none",
#     )
#     ax.add_patch(L.clip)
#     return L


# # 1. Pigment networks (Claude Regular + GPT/Ref Balanced Atypical)
# def vor_segments(pts, rs, wobble):
#     vor = Voronoi(pts)
#     segs = []
#     for a, b in vor.ridge_vertices:
#         if a < 0 or b < 0:
#             continue
#         p, q = vor.vertices[a], vor.vertices[b]
#         if max(np.abs(p).max(), np.abs(q).max()) > 1.6:
#             continue
#         line = p + (q - p) * np.linspace(0, 1, 7)[:, None]
#         n = np.array([p[1] - q[1], q[0] - p[0]])
#         n /= np.linalg.norm(n) + 1e-12
#         off = rs.normal(0, wobble, 7)
#         off[[0, -1]] = 0
#         segs.append(line + off[:, None] * n)
#     return segs


# def network(ax, L, atypical, seed=1, keep=None, strength=1.0):
#     rs = rng(seed)
#     if not atypical:
#         dmin = lambda p: 0.108
#         if keep is None:
#             paint(L, "#C3925F", 0.45 * np.clip(1 - L.rho(X, Y), 0, 1) ** 0.8, z=1)
#         pts = poisson(rs, dmin, 5000)
#         out, cols, lws = [], [], []
#         for s in vor_segments(pts, rs, 0.009):
#             m = s.mean(0)
#             rho = L.rho(*m)
#             if rho > 1.0:
#                 continue
#             w = 1.0 if keep is None else keep(*m)
#             a = (np.clip((0.98 - rho) / 0.45, 0, 1) ** 0.8) * w * strength
#             if a < 0.04:
#                 continue
#             lw = max(0.4, rs.normal(1.05, 0.16))
#             c = mix(BROWN, "#9B6B42", rho**2)
#             out.append(s)
#             lws.append(lw)
#             cols.append((*c, a))
#         lc = LineCollection(
#             out, linewidths=lws, colors=cols, capstyle="round", joinstyle="round", zorder=3
#         )
#         lc.set_clip_path(L.clip)
#         ax.add_collection(lc)
#         return

#     # Hybrid Atypical Network: multi-focal fine + coarse cells, thickened dark lines & gaps
#     if keep is None:
#         b_patch = Blob(rs, (-0.15, 0.1), 0.55, 0.45, irr=0.4, soft=0.22)
#         paint(L, "#B37E4C", 0.45 * b_patch(X, Y), z=1)

#     # Fine honeycomb islands (like the red-brown small cells in Dermoscopedia Card 3)
#     fine_centers = [np.array([-0.28, 0.22]), np.array([-0.15, -0.32]), np.array([0.32, 0.08])]
#     def dmin_atyp(p):
#         d_fine = min(np.linalg.norm(p - fc) for fc in fine_centers)
#         return 0.068 if d_fine < 0.26 else 0.145

#     pts = poisson(rs, dmin_atyp, 7000)
#     holes = [
#         Blob(rs, (0.35, -0.28), 0.24, 0.18, irr=0.35, soft=0.06),
#         Blob(rs, (-0.42, -0.10), 0.20, 0.16, irr=0.35, soft=0.06),
#     ]

#     out, cols, lws = [], [], []
#     for s in vor_segments(pts, rs, 0.020):
#         m = s.mean(0)
#         rho = L.rho(*m)
#         if rho > 0.96:
#             continue
#         if max(h(*m) for h in holes) > 0.55 or rs.random() < 0.14:
#             continue  # patchy / broken segments

#         d_fine = min(np.linalg.norm(m - fc) for fc in fine_centers)
#         is_fine = d_fine < 0.27
#         a = np.clip((0.96 - rho) / 0.14, 0, 1) * strength

#         if is_fine:
#             lw = rs.uniform(0.7, 1.35)
#             c = mix("#8B3A22", BROWN, rs.uniform(0.2, 0.8))
#         else:
#             # Thickened irregular dark brown / black trabeculae
#             thick_zone = 0.5 + 0.5 * np.sin(2.5 * m[0] - 1.8 * m[1])
#             lw = rs.uniform(1.4, 3.3) if thick_zone > 0.45 else rs.uniform(0.8, 1.8)
#             c = mix(BROWN, BLACK, rs.uniform(0.25, 0.85) if thick_zone > 0.45 else rs.uniform(0.0, 0.35))

#         out.append(s)
#         lws.append(lw)
#         cols.append((*c, a))

#         # Branched spurs on thick segments
#         if lw > 2.0 and rs.random() < 0.25:
#             ang = rs.uniform(0, 2 * np.pi)
#             spur = np.array([m, m + rs.uniform(0.04, 0.09) * np.array([np.cos(ang), np.sin(ang)])])
#             out.append(spur)
#             lws.append(0.65 * lw)
#             cols.append((*c, a))

#     lc = LineCollection(
#         out, linewidths=lws, colors=cols, capstyle="round", joinstyle="round", zorder=3
#     )
#     lc.set_clip_path(L.clip)
#     ax.add_collection(lc)


# # 2. Prominent Skin Markings (Multi-angle intersecting pale furrows over warm tan)
# def skin_markings(ax, L, seed=21):
#     rs = rng(seed)
#     paint(L, "#D19D68", 0.65, z=1, tex=noise(rs, 14), amp=0.03)
#     lines = []
#     # 3 intersecting orientation families like natural skin cleavage lines (Card 3)
#     base_angles = [0.18, 1.15, 2.15]
#     for base_ang in base_angles:
#         for pos in np.linspace(-0.78, 0.78, 4):
#             ang = base_ang + rs.normal(0, 0.10)
#             d = np.array([np.cos(ang), np.sin(ang)])
#             n = np.array([-d[1], d[0]])
#             t = np.linspace(-1.35, 1.35, 90)
#             off = (
#                 pos
#                 + rs.normal(0, 0.06)
#                 + 0.045 * np.sin(t * rs.uniform(1.6, 2.8) + rs.uniform(0, 6))
#             )
#             lines.append(t[:, None] * d + off[:, None] * n)

#     fills = rs.uniform(2.5, 4.0, len(lines))
#     for lws, col, z in ((fills + 1.8, "#96653B", 3), (fills, "#F5E5CD", 4)):
#         lc = LineCollection(lines, linewidths=lws, colors=col, capstyle="round", zorder=z)
#         lc.set_clip_path(L.clip)
#         ax.add_collection(lc)
# skin_markings.tone = "warm"


# # 3. Negative Network (Packed elongated sausage-like curved brown tubules + pale septa)
# def negative_network(ax, L, seed=7):
#     rs = rng(seed)
#     paint(L, "#F0DEC2", 0.96, z=1, tex=noise(rs, 8), amp=0.02)
#     samps, polys, fcs = [], [], []

#     # Two-pass packing: first place long curved sausage globules, then fill medium ones
#     passes = [
#         (1800, (0.26, 0.44), (0.060, 0.076), 0.024),
#         (2200, (0.16, 0.26), (0.048, 0.062), 0.020),
#     ]
#     for n_try, (l_lo, l_hi), (w_lo, w_hi), gap in passes:
#         for _ in range(n_try):
#             c = rs.uniform((-0.95, -0.85), (0.95, 0.85))
#             if L.rho(*c) > 0.86:
#                 continue
#             ln = rs.uniform(l_lo, l_hi)
#             w = rs.uniform(w_lo, w_hi)
#             t = np.linspace(-0.5, 0.5, 22)
#             # Smooth C-curve or S-curve
#             ang0 = rs.uniform(0, np.pi)
#             bend = rs.choice([-1, 1]) * rs.uniform(1.8, 3.8)
#             th = ang0 + bend * t + 0.6 * np.sin(2 * np.pi * t)
#             pts = np.cumsum(np.c_[np.cos(th), np.sin(th)] * (ln / 21), 0)
#             pts += c - pts.mean(0)

#             if any(L.rho(pts[:, 0], pts[:, 1]) > 0.91):
#                 continue
#             if any(
#                 np.min(np.linalg.norm(sp[:, None] - pts[None], axis=2)) < sw + w + gap
#                 for sp, sw in samps
#             ):
#                 continue

#             samps.append((pts, w))
#             # Rounded capsule profile
#             profile = w * np.clip(1 - (1.92 * t) ** 4, 0.12, 1) ** 0.45
#             polys.append(ribbon(pts, profile))
#             col = mix("#6B3E1E", "#8A5630", rs.uniform(0.0, 0.7))
#             fcs.append((*col, 0.94))

#     add_polys(ax, L, polys, fcs, z=3, ec=(0.30, 0.17, 0.08, 0.65), lw=0.45)


# # 4. Dots, globules, hyperpigmented areas
# def dots(ax, L, atypical, seed=2):
#     rs = rng(seed)
#     if not atypical:
#         paint(L, "#CFA474", 0.45 * np.clip(1 - L.rho(X, Y) / 0.8, 0, 1), z=1)
#         pts = poisson(rs, lambda p: 0.135, 3000, 0.9, 0.8)
#         pts = pts[L.rho(pts[:, 0], pts[:, 1]) < 0.74]
#         polys = [glob_shape(p, 0.047 * rs.uniform(0.88, 1.12), rs, 0.07) for p in pts]
#         fcs = [(*mix(BROWN, "#5C3519", rs.uniform(0, 0.5)), 1.0) for _ in pts]
#         add_polys(ax, L, polys, fcs, z=4, ec=(0.25, 0.14, 0.06, 0.5), lw=0.3)
#         return
#     paint(L, "#B07D4E", 0.5 * Blob(rs, (0.3, -0.15), 0.55, 0.45, irr=0.35, soft=0.2)(X, Y), z=1)
#     big = rs.normal((0.3, -0.15), (0.35, 0.3), (8, 2))
#     big = big[L.rho(big[:, 0], big[:, 1]) < 0.85]
#     med = rs.normal((0.2, -0.1), (0.45, 0.4), (18, 2))
#     med = med[L.rho(med[:, 0], med[:, 1]) < 0.9]
#     add_polys(
#         ax,
#         L,
#         [glob_shape(p, rs.uniform(0.028, 0.05), rs, 0.18) for p in med],
#         [(*mix(BROWN, BLACK, rs.uniform(0.2, 0.8)), 1.0) for _ in med],
#         z=4,
#     )
#     add_polys(
#         ax,
#         L,
#         [glob_shape(p, rs.uniform(0.06, 0.11), rs, 0.28) for p in big],
#         [to_rgba(rs.choice([DARK, BLACK, "#6A3D1D"])) for _ in big],
#         z=5,
#     )
#     a0 = rs.uniform(0, 2 * np.pi)
#     ang, rad = a0 + rs.uniform(-0.6, 0.6, 30), rs.uniform(0.55, 0.85, 30)
#     tiny = np.vstack(
#         [
#             rs.normal((0.4, -0.25), 0.28, (80, 2)),
#             np.c_[rad * np.cos(ang), 0.88 * rad * np.sin(ang)],
#         ]
#     )
#     tiny = tiny[L.rho(tiny[:, 0], tiny[:, 1]) < 0.93]
#     pc = PatchCollection(
#         [Circle(p, rs.uniform(0.006, 0.015)) for p in tiny],
#         facecolors=[rs.choice(["#141414", "#2B211B", "#4A4A55"]) for _ in tiny],
#         edgecolors="none",
#         zorder=6,
#     )
#     pc.set_clip_path(L.clip)
#     ax.add_collection(pc)


# def hyper_areas(ax, L, seed=22):
#     """Small irregular dark brown smudges with soft darker cores."""
#     rs = rng(seed)
#     paint(L, "#CFA474", 0.3, z=1, tex=noise(rs, 12), amp=0.04)
#     pts = poisson(rs, lambda p: 0.38, 600, 0.78, 0.68)
#     pts = pts[L.rho(pts[:, 0], pts[:, 1]) < 0.78][:8]
#     shell, core = np.zeros_like(X), np.zeros_like(X)
#     for p in pts:
#         b = Blob(rs, p, rs.uniform(0.09, 0.16), rs.uniform(0.07, 0.12), irr=0.45, soft=0.12, k=(2, 3, 4, 6))
#         cb = Blob(rs, p + rs.normal(0, 0.02, 2), 0.48 * b.rx, 0.48 * b.ry, irr=0.4, soft=0.14)
#         shell, core = np.maximum(shell, b(X, Y)), np.maximum(core, cb(X, Y))
#     paint(L, "#7A4A26", 0.85 * shell, z=2, tex=noise(rs, 4), amp=0.04)
#     paint(L, "#2A170B", 0.75 * core, z=3)


# # 5. Streaks (Claude's tapered streaks + pseudopods)
# def radial(ax, c, ang, s0, s1, wb, wt, rs, colour, tip=0.0, wob=0.012, z=5):
#     s = np.linspace(s0, s1, 24)
#     d = np.array([np.cos(ang), np.sin(ang)])
#     n = np.array([-d[1], d[0]])
#     frac = (s - s0) / max(s1 - s0, 1e-6)
#     off = wob * np.sin(np.pi * rs.uniform(1, 2) * frac + rs.uniform(0, 2 * np.pi)) * frac
#     pts = np.asarray(c) + s[:, None] * d + off[:, None] * n
#     var_line(ax, pts, np.linspace(wb, wt, s.size), colour, None, z)
#     if tip:
#         ax.add_patch(Circle(pts[-1], tip, fc=colour, ec="none", zorder=z))
#     return pts


# def streaks(ax, L, atypical, seed=3):
#     rs = rng(seed)
#     if not atypical:
#         c = np.zeros(2)
#         core = Blob(rs, c, 0.46, 0.40, irr=0.06, soft=0.05, angle=0.0)
#         paint(L, "#7E512D", 0.8 * core.scaled(1.45, 0.18)(X, Y), z=2)
#         paint(L, DARK, 0.97 * core(X, Y), z=3, tex=noise(rs, 6), amp=0.04)
#         paint(L, BLACK, 0.7 * core.scaled(0.6, 0.15)(X, Y), z=3)
#         angs = np.linspace(0, 2 * np.pi, 46, endpoint=False) + rs.normal(0, 0.025, 46)
#         for i, a in enumerate(angs):
#             tip = 0.034 if i % 2 == 0 else 0.0
#             s0 = 0.8 * ray(c, a, core, 0.5, above=False)
#             s1 = ray(c, a, L.rho, 0.97 + rs.normal(0, 0.015)) - 0.6 * tip
#             radial(ax, c, a, s0, s1, 2.6, 1.1 if tip else 0.7, rs, DARK, tip)
#         return
#     c = np.array([0.16, -0.1])
#     b1 = Blob(rs, c, 0.40, 0.33, irr=0.3, soft=0.06)
#     b2 = Blob(rs, c + (-0.18, 0.12), 0.22, 0.2, irr=0.35, soft=0.07)
#     core = union(b1, b2)
#     paint(L, "#B58658", 0.45 * Blob(rs, (-0.3, 0.25), 0.5, 0.4, irr=0.4, soft=0.2)(X, Y), z=1)
#     paint(L, "#8A5C36", 0.65 * union(b1.scaled(1.4, 0.2), b2.scaled(1.5, 0.2))(X, Y), z=2)
#     paint(L, DARK, 0.96 * core(X, Y), z=3, tex=noise(rs, 6), amp=0.04)
#     paint(L, BLACK, 0.7 * b1.scaled(0.55, 0.15)(X, Y), z=3)
#     a0 = rs.uniform(0, 2 * np.pi)
#     for ca, hw, n_p, n_f in ((a0, 0.65, 13, 6), (a0 + rs.uniform(2.0, 2.8), 0.32, 6, 3)):
#         for a in ca + rs.uniform(-hw, hw, n_p):
#             tip = rs.uniform(0.02, 0.05) if rs.random() < 0.6 else 0.0
#             s0 = 0.8 * ray(c, a, core, 0.5, above=False)
#             s1 = ray(c, a, L.rho, rs.uniform(0.9, 1.06)) - 0.5 * tip
#             pts = radial(
#                 ax, c, a, s0, s1, rs.uniform(1.6, 3.8), rs.uniform(0.8, 1.6),
#                 rs, rs.choice([DARK, BLACK]), tip, wob=0.03
#             )
#             if rs.random() < 0.3:
#                 k = int(0.6 * len(pts))
#                 a2 = a + rs.choice([-1, 1]) * rs.uniform(0.25, 0.5)
#                 radial(
#                     ax, pts[k], a2, 0, 0.35 * (s1 - s0), 1.4, 0.6, rs, DARK,
#                     rs.uniform(0.015, 0.03) if rs.random() < 0.5 else 0.0
#                 )
#         for a in ca + rs.uniform(-1.1 * hw, 1.1 * hw, n_f):
#             s0 = 0.9 * ray(c, a, core, 0.5, above=False)
#             radial(ax, c, a, s0, ray(c, a, L.rho, rs.uniform(0.95, 1.08)), 0.9, 0.4, rs, "#2A190C", wob=0.02)


# # 6. Blotches & Blue-White Veil
# def blotch(ax, L, atypical, seed=4):
#     rs = rng(seed)
#     if not atypical:
#         network(ax, L, False, seed=seed + 11)
#         b = Blob(rs, (0, 0), 0.5, 0.43, irr=0.12, soft=0.09)
#         paint(L, "#6E4424", 0.75 * b.scaled(1.25, 0.22)(X, Y), z=4)
#         paint(L, DARK, 0.96 * b(X, Y), z=5, tex=noise(rs, 6), amp=0.03)
#         return
#     tan = Blob(rs, (0.15, -0.1), 0.62, 0.5, irr=0.3, soft=0.12, k=(2, 3, 4))
#     d1 = Blob(rs, (0.4, -0.22), 0.34, 0.24, irr=0.42, soft=0.06, k=(2, 3, 4, 5))
#     d2 = Blob(rs, (0.02, -0.52), 0.16, 0.11, irr=0.4, soft=0.08)
#     keep = lambda x, y: np.clip((-0.9 * x + 0.45 * y + 0.15) / 0.45, 0, 1) * (1 - tan(x, y))
#     network(ax, L, False, seed=seed + 5, keep=keep)
#     paint(L, "#C59A6A", 0.85 * tan(X, Y), z=4, tex=noise(rs, 10), amp=0.03)
#     paint(L, "#4A2A14", 0.95 * d1(X, Y), z=5, tex=noise(rs, 5), amp=0.04)
#     paint(L, "#6B3F1F", 0.85 * d2(X, Y), z=5)


# def homogeneous(ax, L, seed=9):
#     rs = rng(seed)
#     rgb = mix("#C08D58", "#A5743F", np.clip(0.5 + 0.3 * noise(rs, 22), 0, 1)[..., None])
#     paint(L, rgb, 0.92 * gaussian_filter((L.rho(X, Y) < 0.93).astype(float), 8), z=1)


# def blue_white(ax, L, seed=5):
#     rs = rng(seed)
#     paint(L, "#946440", 0.75, z=1, tex=noise(rs, 12), amp=0.04)
#     blobs = [
#         Blob(rs, rs.normal(0, 0.2, 2), rs.uniform(0.25, 0.42), rs.uniform(0.2, 0.35), irr=0.3, soft=0.25)
#         for _ in range(5)
#     ]
#     f = gaussian_filter(np.maximum.reduce([b(X, Y) for b in blobs]), 4)
#     rgb = mix("#4A69BB", "#6F7FA3", np.clip(0.5 + 0.35 * noise(rs, 18), 0, 1)[..., None])
#     paint(L, rgb, 0.92 * f, z=4)
#     paint(L, "#3C4F86", 0.38 * union(*[b.scaled(0.5, 0.3) for b in blobs[:3]])(X, Y), z=4.5)
#     paint(L, "#F2F4F9", 0.62 * f * np.clip(0.5 + 0.35 * noise(rs, 10), 0, 1), z=5)


# # 7. Regression family (Claude scar/granules + GPT clean structureless tan background)
# def pepper(ax, L, rs, centres, sd=0.2, n_sub=10, per=34, z=6):
#     pts = []
#     for c in centres:
#         for sc in rs.normal(c, sd, (n_sub, 2)):
#             pts.append(rs.normal(sc, 0.06 * rs.uniform(0.6, 1.3), (rs.integers(per // 2, per), 2)))
#     pts = np.vstack(pts)
#     pts = pts[L.rho(pts[:, 0], pts[:, 1]) < 0.94]
#     haze = np.maximum.reduce(
#         [np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (2 * (1.2 * sd) ** 2)) for c in centres]
#     )
#     paint(L, "#8E98A8", 0.32 * haze, z=2)
#     cols = np.array([to_rgba(g) for g in GRAINS])[rs.integers(0, 3, len(pts))]
#     sc = ax.scatter(
#         pts[:, 0], pts[:, 1], s=rs.uniform(1.2, 4.8, len(pts)), c=cols, linewidths=0, alpha=0.9, zorder=z
#     )
#     sc.set_clip_path(L.clip)


# def scar(ax, L, rs, c=(-0.18, 0.08)):
#     b1 = Blob(rs, c, 0.44, 0.32, irr=0.32, soft=0.05, k=(2, 3, 5, 7))
#     b2 = Blob(rs, np.add(c, (0.32, -0.22)), 0.22, 0.16, irr=0.35, soft=0.06)
#     white, rim = union(b1, b2), union(b1.scaled(1.22, 0.12), b2.scaled(1.35, 0.12))
#     # Smooth tan-pinkish background (NO peripheral honeycomb network, matching Card 3)
#     paint(L, "#E9B9AE", 0.75 * rim(X, Y), z=4)
#     paint(L, "#FDFBF8", 0.97 * white(X, Y), z=5, tex=noise(rs, 5), amp=0.012)
#     return b1


# def regression(ax, L, seed=6):
#     scar(ax, L, rng(seed))


# def regression_peppering(ax, L, seed=16):
#     rs = rng(seed)
#     c = scar(ax, L, rs).c
#     pepper(ax, L, rs, [c + (0.45, -0.15), c + (0.12, 0.32), c + (-0.02, -0.08)], sd=0.16)


# def peppering(ax, L, seed=26):
#     rs = rng(seed)
#     centres = [p for p in rs.uniform(-0.55, 0.55, (8, 2)) if L.rho(*p) < 0.6][:4]
#     pepper(ax, L, rs, centres, sd=0.2)


# # 8. Vessels & Diagnostic Region
# def serpentine(ax, L, rs, p, h, n, step, w0, depth):
#     hs = h + rs.uniform(0.5, 0.9) * np.sin(rs.uniform(0.35, 0.6) * np.arange(n) + rs.uniform(0, 2 * np.pi))
#     hs = hs + np.cumsum(rs.normal(0, 0.05, n))
#     pts = p + np.vstack([[0, 0], np.cumsum(step * np.c_[np.cos(hs), np.sin(hs)], 0)])
#     lw = np.linspace(w0, 0.45 * w0, len(pts))
#     var_line(ax, pts, lw, rs.choice(REDS), L.clip, z=4)
#     if depth:
#         for _ in range(rs.integers(1, 3)):
#             k = rs.integers(n // 5, 3 * n // 5)
#             serpentine(
#                 ax, L, rs, pts[k], hs[k] + rs.choice([-1, 1]) * rs.uniform(0.6, 1.2),
#                 n // 2, step, 0.75 * lw[k], depth - 1
#             )


# def vessels(ax, L, seed=8):
#     """Polymorphous: serpentine branched, corkscrew, dotted, irregular linear."""
#     rs = rng(seed)
#     paint(L, "#F3C6C0", 0.95, z=1, tex=noise(rs, 10), amp=0.025)
#     for a in rs.uniform(0, 2 * np.pi, 3):
#         p = 0.7 * L.r(a) * np.array([np.cos(a), 0.88 * np.sin(a)])
#         serpentine(ax, L, rs, p, a + np.pi + rs.normal(0, 0.4), 45, 0.022, 2.4, 1)
#     for _ in range(4):
#         c, ang, ln = rs.uniform(-0.55, 0.55, 2), rs.uniform(0, np.pi), rs.uniform(0.25, 0.4)
#         t, k, A = np.linspace(0, 1, 220), rs.uniform(5, 8) * 2 * np.pi, rs.uniform(0.025, 0.04)
#         d, n = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
#         pts = c + ((t - 0.5) * ln + 0.7 * A * np.cos(k * t))[:, None] * d + (A * np.sin(k * t))[:, None] * n
#         var_line(ax, pts, 1.3, rs.choice(REDS), L.clip, z=5)
#     for _ in range(4):
#         h = rs.uniform(0, 2 * np.pi) + np.cumsum(rs.normal(0, 0.5, 12))
#         pts = rs.uniform(-0.6, 0.6, 2) + np.vstack([[0, 0], np.cumsum(0.03 * np.c_[np.cos(h), np.sin(h)], 0)])
#         var_line(ax, pts, rs.uniform(0.8, 3.0, len(pts)), rs.choice(REDS), L.clip, z=4)
#     cl = rs.uniform(-0.5, 0.5, (2, 2))
#     pts = np.vstack(
#         [rs.normal(cl[0], 0.14, (20, 2)), rs.normal(cl[1], 0.14, (15, 2)), rs.uniform(-0.8, 0.8, (12, 2))]
#     )
#     pts = pts[L.rho(pts[:, 0], pts[:, 1]) < 0.9]
#     pc = PatchCollection(
#         [Circle(p, rs.uniform(0.012, 0.026)) for p in pts],
#         facecolors=[rs.choice(REDS) for _ in pts],
#         edgecolors="none",
#         zorder=5,
#     )
#     pc.set_clip_path(L.clip)
#     ax.add_collection(pc)
# vessels.tone = "pink"


# def diagnostic_region(ax, L, seed=23):
#     rs = rng(seed)
#     network(ax, L, False, seed=seed, strength=0.55)
#     roi = Blob(rs, (0.2, -0.05), 0.48, 0.38, irr=0.25, soft=0.1)
#     paint(L, "#9C6A3E", 0.45 * roi(X, Y), z=2)
#     xy = roi.outline(1.15)
#     ax.add_patch(
#         Polygon(
#             xy, closed=True, fc=(0.12, 0.31, 0.47, 0.06),
#             ec="#1F4E79", lw=1.2, ls=(0, (4, 2.5)), zorder=8
#         )
#     )
#     (x0, y0), (x1, y1), k = xy.min(0) - 0.08, xy.max(0) + 0.08, 0.16
#     for x, y, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
#         ax.plot([x + sx * k, x, x], [y, y, y + sy * k], color="#1F4E79", lw=1.4, solid_capstyle="butt", zorder=9)
#     ax.text(x0 + 0.02, y1 + 0.04, "ROI", fontsize=6, color="#1F4E79", va="bottom", zorder=9)


# # Rendering
# def render(ax, draw, seed, title, colour="black", fs=8.5):
#     L = panel(ax, seed, getattr(draw, "tone", "tan"))
#     draw(ax, L)
#     ax.set_xlim(*EXT[:2])
#     ax.set_ylim(*EXT[2:])
#     ax.set_title(title, fontsize=fs, pad=5, color=colour)


# ROWS = [
#     ("Benign form", [
#         ("Regular network", lambda a, c: network(a, c, False)),
#         ("Regular dots and globules", lambda a, c: dots(a, c, False)),
#         ("Symmetric streaks", lambda a, c: streaks(a, c, False)),
#         ("Central blotch", lambda a, c: blotch(a, c, False)),
#     ]),
#     ("Melanoma form", [
#         ("Atypical network", lambda a, c: network(a, c, True)),
#         ("Atypical dots and globules", lambda a, c: dots(a, c, True)),
#         ("Irregular streaks", lambda a, c: streaks(a, c, True)),
#         ("Eccentric structureless area", lambda a, c: blotch(a, c, True)),
#     ]),
#     ("Melanoma associated", [
#         ("Blue-white structureless area", blue_white),
#         ("Regression with peppering", regression_peppering),
#         ("Negative network", negative_network),
#         ("Atypical vascular pattern", vessels),
#     ]),
# ]

# fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.8))
# for i, (rowname, items) in enumerate(ROWS):
#     for j, (title, draw) in enumerate(items):
#         render(axes[i, j], draw, 10 * i + j, title)
#     axes[i, 0].text(-1.75, 0, rowname, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
# fig.tight_layout(pad=0.6)
# save(fig, "fig_ch2_structures")
# plt.show()


# LABELS = [
#     ("Atypical network", "melanoma", lambda a, c: network(a, c, True)),
#     ("Regular network", "benign", lambda a, c: network(a, c, False)),
#     ("Negative network", "melanoma", negative_network),
#     ("Atypical dots", "melanoma", lambda a, c: dots(a, c, True)),
#     ("Atypical streaks", "melanoma", lambda a, c: streaks(a, c, True)),
#     ("Structureless area", "melanoma", lambda a, c: blotch(a, c, True)),
#     ("Blue-white structureless area", "melanoma", blue_white),
#     ("Regression", "melanoma", regression),
#     ("Regression with scar-like\ndepigmentation and peppering", "melanoma", regression_peppering),
#     ("Peppering", "melanoma", peppering),
#     ("Atypical vascular pattern", "melanoma", vessels),
#     ("Homogeneous", "benign", homogeneous),
#     ("Prominent skin markings", "other", skin_markings),
#     ("Small hyperpigmented area", "other", hyper_areas),
#     ("Diagnostic region", "other", diagnostic_region),
# ]
# COL = {"melanoma": "#B23A48", "benign": "#2E6E4E", "other": "#555555"}

# fig, axes = plt.subplots(3, 5, figsize=(12, 8))
# for ax, (name, grp, draw) in zip(axes.ravel(), LABELS):
#     render(ax, draw, zlib.crc32(name.encode()) % 97, name, COL[grp], fs=8)
# fig.tight_layout(pad=0.6)
# save(fig, "fig_appA_labels_visual")
# plt.show()


# ---- cell 48 ------------------------------------------------------------
import json, re
pat = re.compile(r"\b(dice\w*|iou\w*|pointing\w*|peak_dist|struct_hit|topk\w*|top10\w*|auc_raw|auc_above_null|"
                 r"inside_minus_area\w*|energy\w*|hit_rate\w*|jaccard\w*)\b", re.I)
found = {}
for p in list((REPO / "notebooks/mel_nv").glob("*.ipynb")) + list((REPO / "src/thesis").glob("*.py")):
    try:
        src = p.read_text() if p.suffix == ".py" else "".join("".join(c["source"]) for c in json.load(open(p))["cells"])
    except Exception:
        continue
    for m in set(x.lower() for x in pat.findall(src)):
        found.setdefault(m, set()).add(p.name)
for k in sorted(found):
    print(f"{k:28} {sorted(found[k])[:6]}")


# ---- cell 49 ------------------------------------------------------------
LD = pd.read_csv(TAB / "ch5_reference_ladder.csv")
print("refs:", LD.ref.unique().tolist(), " images:", LD.image_id.nunique())
for met in ["auc_in", "energy_in"]:
    print(met, {rn(r): miqr(LD[LD.ref == r][met]) for r in LD.ref.unique()})
    q = LD.pivot_table(index="image_id", columns="ref", values=met)
    for mo in ["lesion aligned", "concept only", "unaligned"]:
        if mo in q:
            print(f"  {rn(mo)} vs darkness:", metrics.paired(q[mo], q.darkness))


# ---- cell 50 ------------------------------------------------------------
H = pd.read_csv(MELNV / "ham_mel_nv_clean.csv"); H["image_id"] = H.image_id.astype(str).str.strip()
META = H.drop_duplicates("image_id").set_index("image_id")[["gt_label", "dataset", "lesion_id"]]
d = pd.read_csv(MELNV / "f2_fold0.csv"); h = pd.read_csv(MELNV / "f2_fold0_heldout.csv")
for x in (d, h): x["image_id"] = x.image_id.astype(str).str.strip()
SF = (pd.concat([d[["image_id", "split"]], h[["image_id"]].assign(split="held-out")])
        .drop_duplicates("image_id").join(META, on="image_id"))
assert SF.gt_label.notna().all() and SF.dataset.notna().all()
miss = sorted(set(H.image_id) - set(SF.image_id))
print(f"1. subset {len(H)} ({(H.gt_label == 'MEL').sum()} MEL) | split files {len(SF)} ({(SF.gt_label == 'MEL').sum()} MEL)")
print("1. missing from split files:", {i: META.loc[i, "gt_label"] for i in miss})

def prev(df, tag):
    g = df.groupby("split").agg(images=("image_id", "size"), MEL=("gt_label", lambda s: (s == "MEL").sum()))
    g["share"] = (g.MEL / g.images).round(3); print(f"2. {tag}\n{g.to_string()}"); return g
prev(H, "original"); prev(SF, "split files, fold 0")

site = SF.groupby("dataset").agg(images=("image_id", "size"), MEL=("gt_label", lambda s: (s == "MEL").sum()),
                                 NV=("gt_label", lambda s: (s == "NV").sum()))
site["MEL_share"] = site.MEL / site.images
site.reset_index().to_csv(TAB / "appA_sites.csv", index=False)
cells = [[tx(i), fi(r.images), fi(r.MEL), fi(r.NV), f3(r.MEL_share)] for i, r in site.iterrows()]
cells += [r"\midrule", ["total", fi(site.images.sum()), fi(site.MEL.sum()), fi(site.NV.sum()),
                        f3(site.MEL.sum() / site.images.sum())]]
write("appA_sites.tex", ["source", "images", "MEL", "NV", "MEL share"], cells, "lrrrr")
print(site.round(3).to_string())

g = SF.groupby(["split", "gt_label"]).agg(images=("image_id", "size"), lesions=("lesion_id", "nunique")).reset_index()
g["img_per_lesion"] = g.images / g.lesions
g.to_csv(TAB / "appA_splits_final_fold0.csv", index=False)
ORD = {"train": 0, "val": 1, "held-out": 2, "test": 3}
g = g.assign(o=g.split.map(ORD)).sort_values(["o", "gt_label"])
write("appA_splits_final_fold0.tex", ["split", "class", "images", "lesions", "images per lesion"],
      [[tx({"val": "validation"}.get(r.split, r.split)), r.gt_label, fi(r.images), fi(r.lesions), f"{r.img_per_lesion:.2f}"]
       for _, r in g.iterrows()], "llrrr")
N_SF = (len(SF), int((SF.gt_label == "MEL").sum())); print("data flow, split files box:", N_SF)


# ---- cell 51 ------------------------------------------------------------
import re
IDX = {(float(a["ha_lambda"]), float(a.get("feat_lambda") or 0), int(a["fold"])): p
       for p in CK_F2.glob("*.pth") for a in [models.ckpt_args(p)]}
CKS = {"unaligned (4.1)": CK5 / "checkpoint-best-gap-ha0.pth", "global aligned (4.1)": CK5 / "checkpoint-best-gap-ha5.pth",
       "grid control f0": IDX[min(k for k in IDX if k[1] == 0 and k[2] == 0)],
       "grid global only f0": IDX[(5.0, 0.0, 0)], "grid expert only f0": IDX[(0.0, 1.0, 0)]}
KEY = re.compile(r"batch|epoch|^lr$|min_lr|warmup|weight_decay|layer_decay|drop_path|smoothing|mixup|cutmix|"
                 r"^aa$|reprob|color|seed|sampler|balance|^ha_|^feat_|start|opt|sched|input_size|pool|clip|ema|eval_metric|save", re.I)
A = pd.DataFrame({nm: {k: v for k, v in models.ckpt_args(p).items() if KEY.search(k)} for nm, p in CKS.items()})
for nm, p in CKS.items():
    A.loc["saved_epoch", nm] = torch.load(p, map_location="cpu", weights_only=False).get("epoch")
pd.set_option("display.max_rows", 200)
print(A.astype(str).to_string())
print("\ndiffering rows:\n", A[A.astype(str).nunique(axis=1) > 1].astype(str).to_string())


# ---- cell 52 ------------------------------------------------------------
from sklearn.metrics import roc_auc_score, average_precision_score
SIB = {k: [] for k in range(5)}
for k in range(5):
    dd = pd.read_csv(MELNV / f"f2_fold{k}.csv"); dd["image_id"] = dd.image_id.astype(str).str.strip()
    tr, te = dd[dd.split == "train"], dd[dd.split == "test"]
    SIB[k] = te[te.lesion_id.isin(set(tr.lesion_id))].image_id.tolist()
ID = np.asarray(ID_TEST).astype(str); rows = []
for (al, be, nm) in [(5.0, 0.0, "global only"), (0.0, 1.0, "expert only")]:
    for k in range(5):
        m = models.load_ckpt(IDX[(al, be, k)], cache=False); p = models.predict_mel(m, X_TEST); del m
        kk = ~np.isin(ID, SIB[k])
        rows.append({"setting": nm, "fold": k, "n_excl": int((~kk).sum()),
                     "auc": roc_auc_score(Y_MEL, p), "auc_ex": roc_auc_score(Y_MEL[kk], p[kk]),
                     "ap": average_precision_score(Y_MEL, p), "ap_ex": average_precision_score(Y_MEL[kk], p[kk])})
LK = pd.DataFrame(rows); LK.to_csv(TAB / "ch7_sibling_leakage.csv", index=False)
print("5. siblings per fold:", {k: len(v) for k, v in SIB.items()})
print(LK.groupby("setting")[["auc", "auc_ex", "ap", "ap_ex"]].mean().round(4).to_string())
print((LK.assign(d_auc=LK.auc_ex - LK.auc, d_ap=LK.ap_ex - LK.ap).groupby("setting")[["d_auc", "d_ap"]]
       .agg(["mean", "min", "max"]).round(4)).to_string())


# ---- cell 53 ------------------------------------------------------------
import matplotlib.patches as mpatches
import cv2
def full224(m): return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
def label_union(iid, lab):
    acc = None
    for p in QC[(QC.Image_ID == iid) & (QC.label == lab)].mask_path:
        m = data._full(p); acc = m if acc is None else (acc | m)
    return full224(acc)
def cells(ax, G, face, edge):
    for (r, c) in zip(*np.where(G)):
        ax.add_patch(mpatches.Rectangle((c * 16 - .5, r * 16 - .5), 16, 16, fc=face, ec=edge, lw=0.6, alpha=0.6))
    for g in range(0, 225, 16):
        ax.axhline(g - .5, color="0.35", lw=0.35); ax.axvline(g - .5, color="0.35", lw=0.35)

cand = [i for i in sorted(HUM) if EV.loc[i].gt_label == "MEL" and 40 <= LESION[i].sum() <= 80
        and 2 <= QC[(QC.Image_ID == i) & (QC.label_group == "MEL")].label.nunique() <= 3]
iid = cand[0]; r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path)
L = np.asarray(LESION[iid], bool); U = np.asarray(HUM[iid], bool); S = U & L
Lf = full224(np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127)
labs = QC[(QC.Image_ID == iid) & (QC.label_group == "MEL")].label.unique()
print("6. image", iid, " lesion cells", int(L.sum()), " structure cells in lesion", int(S.sum()))

FS = 9
fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH / 3 + 0.55))
# ax[0].imshow(rgb); ax[0].contour(Lf.astype(float), [.5], colors=["#2CA02C"], linewidths=1.4)
ax[0].imshow(rgb); ax[0].contour(Lf.astype(float), [.5], colors=[C_LESION], linewidths=1.4)
for lab in labs: ax[0].contour(label_union(iid, lab).astype(float), [.5], colors=["white"], linewidths=1.1)
# ax[1].imshow(rgb, alpha=0.6); cells(ax[1], L, "#2CA02C", "#1B5E20")
ax[1].imshow(rgb, alpha=0.6); cells(ax[1], L, C_LESION, "#1B5E20")
# ax[2].imshow(rgb, alpha=0.6); cells(ax[2], L & ~S, "none", "#2CA02C"); cells(ax[2], S, "#FF00FF", "#7B007B")
ax[2].imshow(rgb, alpha=0.6); cells(ax[2], L & ~S, "none", C_LESION); cells(ax[2], S, "#FF00FF", "#7B007B")
ax[0].set_xlabel("(a) lesion and structures", fontsize=FS)
ax[1].set_xlabel(f"(b) lesion cells: {int(L.sum())}", fontsize=FS)
ax[2].set_xlabel(f"(c) structure cells: {int(S.sum())}", fontsize=FS)
for a in ax: a.set_xticks([]); a.set_yticks([]); a.set(xlim=(-.5, 223.5), ylim=(223.5, -.5))
fig.tight_layout(); save(fig, "fig_ch3_grid_coverage"); plt.show()


# ---- cell 54 ------------------------------------------------------------
for nm, p in CKS.items():
    a = models.ckpt_args(p)
    print(f"{nm:22} weights={a.get('weights')}  label_col binary={a.get('binary', a.get('nb_classes'))}")


# ---- cell 55 ------------------------------------------------------------
from src.thesis.config import MEL_IDX, NV_IDX
from src.thesis.cams import _unwrap_model, unpack_model_outputs   # remove this line if the import fails, the names are used by cams._forward

def block_maps(model, x, b):
    base = _unwrap_model(model); base.clear_xai_state()
    with torch.enable_grad():
        logits, _ = unpack_model_outputs(model(x, return_patch_tokens=True, store_attn=True, attn_layer=b))
        A = base.blocks[b].attn.last_attn
        assert A is not None, f"block {b}: no attention stored"
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A, retain_graph=True)[0]
        Gn = torch.autograd.grad(logits[0, NV_IDX], A)[0]
    gap = bool(getattr(base, "use_mean_pooling", False)); base.clear_xai_state()
    f = (lambda T: T[0, :, 1:, 1:]) if gap else (lambda T: T[0, :, 0, 1:])
    A, Gm, Gn = [f(T).detach().float().cpu().numpy() for T in (A, Gm, Gn)]
    red = (lambda P: P.sum(1)) if gap else (lambda P: P)
    fin = lambda P: cams.norm01(np.maximum(red(P).mean(0), 0)).reshape(14, 14)
    return fin(A * Gm), fin(A * Gn)

# sanity check: block 11 must equal the standard construction
x0 = data.to_tensor(data.native(EV.loc[sorted(HUM)[0]].image_rel_path))
m_ref = cams.class_cams(M5, x0)[0]; m_new = block_maps(M5, x0, 11)[0]
print("7. block 11 matches class_cams:", np.allclose(m_ref.reshape(14, 14), m_new, atol=1e-4))

rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L = LESION[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for nm, m in [("unaligned", M0), ("global aligned", M5)]:
        for b in range(12):
            mel, nv = block_maps(m, x, b)
            rows.append({"model": nm, "block": b + 1, "image_id": iid, "r_in": metrics.corr_in(mel, nv, L)})
LY = pd.DataFrame(rows); LY.to_csv(TAB / "appC_layers.csv", index=False)
S = LY.groupby(["model", "block"]).r_in.agg(med="median", q1=lambda s: s.quantile(.25),
                                           q3=lambda s: s.quantile(.75), n="count")
print("7.\n", S.round(3).unstack(0).to_string())

fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.62, 2.3))
for nm, c in [("unaligned", plotting.C_CONTEXT), ("global aligned", plotting.C_FOCUS)]:
    s = S.loc[nm]; ax.plot(s.index, s.med, "-o", ms=3, color=c, label=nm)
    ax.fill_between(s.index, s.q1, s.q3, color=c, alpha=0.2)
ax.axhline(0, color="0.5", lw=0.6, ls=":")
ax.set(xlabel="transformer block", ylabel=r"$r_{\mathrm{in}}$, MEL vs NV map", xticks=range(1, 13), ylim=(-1, 1))
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); save(fig, "fig_appC_layers"); plt.show()


# ---- cell 56 ------------------------------------------------------------
rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for nm, m in [("unaligned", M0), ("global aligned", M5)]:
        for b in range(12):
            mel, nv = block_maps(m, x, b)
            rows.append({"model": nm, "block": b + 1, "image_id": iid, "gt": r.gt_label,
                         "mel_loc": metrics.lesion_auc(mel, L), "nv_loc": metrics.lesion_auc(nv, L),
                         "mel_auc_in": metrics.auc_in(mel, L, U), "r_in": metrics.corr_in(mel, nv, L)})
LY = pd.DataFrame(rows); LY.to_csv(TAB / "appC_layers.csv", index=False)
MET = ["mel_loc", "nv_loc", "mel_auc_in", "r_in"]
S = LY.groupby(["model", "block"])[MET].median()
print("7. medians per block\n", S.round(3).unstack(0).to_string())
for nm in ["unaligned", "global aligned"]:
    print(f"7. {nm:15} peak block: " + "  ".join(f"{c} {int(S.loc[nm][c].idxmax())} ({S.loc[nm][c].max():.3f})"
                                              for c in ["mel_loc", "nv_loc", "mel_auc_in"]))

Q = LY.groupby(["model", "block"])[MET].quantile([.25, .75]).unstack()
PAN = [("mel_loc", "MEL map, lesion localisation", (0, 1), 0.5),
       ("nv_loc", "NV map, lesion localisation", (0, 1), 0.5),
       ("r_in", r"$r_{\mathrm{in}}$, MEL vs NV map", (-1, 1), 0.0)]
fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.3), sharex=True)
for j, (c, lab_, yl, ch) in enumerate(PAN):
    for nm, col in [("unaligned", plotting.C_CONTEXT), ("global aligned", plotting.C_FOCUS)]:
        s = S.loc[nm][c]; q1, q3 = Q.loc[nm][(c, .25)], Q.loc[nm][(c, .75)]
        ax[j].plot(s.index, s.values, "-o", ms=2.5, lw=1, color=col, label=nm)
        ax[j].fill_between(s.index, q1, q3, color=col, alpha=0.18)
        if c != "r_in":
            b = int(s.idxmax()); ax[j].plot(b, s.max(), "*", ms=8, color=col, mec="k", mew=0.4)
    ax[j].axhline(ch, color="0.5", lw=0.6, ls=":")
    ax[j].set(ylim=yl, xticks=[1, 4, 8, 12], xlabel="transformer block")
    ax[j].set_ylabel(lab_, fontsize=7); ax[j].tick_params(labelsize=7)
ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(); save(fig, "fig_appC_layers"); plt.show()


# ---- cell 57 ------------------------------------------------------------
rows = []
for k in range(5):
    m = models.f2(0.0, 1.0, k)
    for iid in [i for i in held if FOLD[i] == k]:
        r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]; x = data.to_tensor(data.native(r.image_rel_path))
        for b in range(12):
            mel, nv = block_maps(m, x, b)
            rows.append({"model": "expert aligned", "block": b + 1, "image_id": iid, "gt": r.gt_label,
                         "mel_loc": metrics.lesion_auc(mel, L), "nv_loc": metrics.lesion_auc(nv, L),
                         "mel_auc_in": metrics.auc_in(mel, L, U), "r_in": metrics.corr_in(mel, nv, L)})
    del m
LY3 = pd.concat([LY[LY.model != "expert aligned"], pd.DataFrame(rows)], ignore_index=True)
LY3.to_csv(TAB / "appC_layers.csv", index=False)
MET = ["mel_loc", "nv_loc", "mel_auc_in", "r_in"]
S = LY3.groupby(["model", "block"])[MET].median()
Q = LY3.groupby(["model", "block"])[MET].quantile([.25, .75]).unstack()
print("7. expert aligned, medians per block\n", S.loc["expert aligned"].round(3).to_string())
print("7. expert block 12 check, expect r_in about 0.07:", S.loc[("expert aligned", 12)].round(3).to_dict())

# C_MEL, C_NV = "#C0392B", "#2471A3"
MODELS = [("unaligned", "(a) unaligned, 31 images"), ("global aligned", "(b) global aligned, 31 images"),
          ("expert aligned", "(c) expert aligned, 29 held-out images")]
fig, ax = plt.subplots(2, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.62), sharex=True)
ax = ax.ravel()
def band(a, nm, c, col, lab_):
    s = S.loc[nm][c]
    a.plot(s.index, s.values, "-o", ms=2.5, lw=1, color=col, label=lab_)
    a.fill_between(s.index, Q.loc[nm][(c, .25)], Q.loc[nm][(c, .75)], color=col, alpha=0.18)
for j, (nm, title) in enumerate(MODELS):
    # band(ax[j], nm, "mel_loc", C_FOCUS, "MEL map"); band(ax[j], nm, "nv_loc", C_NV, "NV map")
    band(ax[j], nm, "mel_loc", plotting.C_MEL, "MEL map"); band(ax[j], nm, "nv_loc", plotting.C_NV, "NV map")
    ax[j].axhline(0.5, color="0.5", lw=0.6, ls=":")
    ax[j].set(ylim=(0, 1), ylabel="lesion localisation"); ax[j].set_title(title, fontsize=8)
# for nm, col in [("unaligned", plotting.C_CONTEXT), ("global aligned", plotting.C_FOCUS), ("expert aligned", "#7D3C98")]:
for nm, col in [(m, plotting.MODEL_COLOR[m]) for m in ["unaligned", "global aligned", "expert aligned"]]:
    band(ax[3], nm, "r_in", col, nm)
ax[3].axhline(0, color="0.5", lw=0.6, ls=":")
ax[3].set(ylim=(-1, 1), ylabel=r"$r_{\mathrm{in}}$, MEL vs NV map"); ax[3].set_title("(d) class map correlation", fontsize=8)
ax[0].legend(frameon=False, fontsize=7, loc="lower left"); ax[3].legend(frameon=False, fontsize=6.5, loc="upper left")
for a in ax:
    a.set_xticks(range(1, 13)); a.tick_params(labelsize=7)
for a in ax[2:]: a.set_xlabel("transformer block", fontsize=8)
fig.tight_layout(); save(fig, "fig_appC_layers"); plt.show()


# ---- cell 58 ------------------------------------------------------------
import json, re
hits = {}
for p in sorted((REPO / "notebooks/mel_nv").glob("16*.ipynb")):
    for c in json.load(open(p))["cells"]:
        s = "".join(c["source"])
        used = sorted(set(re.findall(r"C_(?:FOCUS|CONTEXT|REF|CONTROL)|#C0392B|#2471A3|#7D3C98", s)))
        figs = re.findall(r'save\(fig,\s*"([^"]+)"', s)
        for f in figs:
            if used: hits[f] = (p.name, used)
for f, (nb, u) in sorted(hits.items()):
    print(f"{f:40} {nb:28} {u}")


# ---- cell 59 ------------------------------------------------------------
import json
RB = pd.read_csv(TAB / "app_robustness.csv")
print("1. columns:", list(RB.columns))
for c in [c for c in RB.columns if any(k in c.lower() for k in ("dx", "dy", "shift", "offset"))]:
    print("1.", c, RB[RB.cond == "shifted"][c].describe().round(1).to_dict())
for p in sorted((REPO / "notebooks/mel_nv").glob("16*.ipynb")):
    for c in json.load(open(p))["cells"]:
        s = "".join(c["source"])
        if "app_robustness" in s:
            keep = [l.strip() for l in s.splitlines()
                    if any(k in l.lower() for k in ("shift", "roll", "offset", "translate", "warpaffine", "dx", "dy"))]
            if keep: print("-" * 50, p.name); print("\n".join(keep))


# ---- cell 60 ------------------------------------------------------------
# from matplotlib.colors import to_rgba
# M61 = pd.read_csv(TAB / "ch6_metrics.csv")
# ORD = ["Grad-CAM", "FinerCAM 0.6", "Diff-CAM", "random"]
# LAB = {"Grad-CAM": "Grad-CAM", "FinerCAM 0.6": r"FinerCAM ($\gamma=0.6$)", "Diff-CAM": "Diff-CAM", "random": "random"}
# FOC = {"Grad-CAM", "FinerCAM 0.6"}
# rng = np.random.default_rng(0)
# fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.6, 2.5))
# for i, m in enumerate(ORD):
#     v = M61[M61.method == m].auc_in.dropna().values; f = m in FOC; col = METHOD_COLOR[m]
#     ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
#                boxprops=dict(facecolor=to_rgba(col, 0.25 if f else 0.10), edgecolor=col if f else "0.6", lw=1),
#                medianprops=dict(color=col if f else "0.5", lw=2), whiskerprops=dict(color=col if f else "0.6"),
#                capprops=dict(color=col if f else "0.6"))
#     ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=col, alpha=0.8 if f else 0.35,
#                linewidths=0, zorder=3)
#     ax.hlines(np.median(v), i - 0.28, i + 0.28, color=METHOD_COLOR[m] if f else "0.5", lw=2.2, zorder=4)
# ax.axhline(0.5, color=C_CHANCE, ls=":", lw=1)
# ax.set_xticks(range(len(ORD))); ax.set_xticklabels([LAB[m] for m in ORD], fontsize=7)
# for t, m in zip(ax.get_xticklabels(), ORD): t.set_color("black" if m in FOC else "0.5")
# ax.set_ylabel(r"$\mathrm{AUC}_{\mathrm{in}}$", fontsize=8); ax.set_ylim(0, 1); ax.tick_params(labelsize=7)
# fig.tight_layout(); save(fig, "fig_ch6_methods_auc"); plt.show()


# ---- cell 61 ------------------------------------------------------------
# from scipy.stats import wilcoxon
# UP, DN = r" $\uparrow$", r" $\downarrow$"
# def pm(s):
#     s = pd.Series(s).dropna()
#     return "--" if s.empty else f"{s.mean():.3f} $\\pm$ {s.std(ddof=1):.3f}"
# lam = lambda v: r"$10^{-8}$" if np.isclose(v, 1e-8, rtol=0, atol=1e-12) else f"{v:g}"
# sel = lambda d, a, b: d[np.isclose(d.alpha, a, rtol=0, atol=1e-12) & np.isclose(d.beta, b, rtol=0, atol=1e-12)]

# F7 = pd.read_csv(TAB / "ch7_grounding.csv"); F7 = F7[F7.alpha.notna()]
# C7 = pd.read_csv(TAB / "ch7_classification.csv")
# GRID = [(1e-8, 0, "control"), (5, 0, "global only"), (5, 1, "joint"), (5, 5, "joint"), (5, 20, "joint"),
#         (0, 1, "expert only"), (0, 5, "expert only"), (0, 20, "expert only")]
# MET = [("mel_auc_in", r"$\mathrm{AUC}_{\mathrm{in}}$" + UP), ("mel_in", r"$E_{\mathrm{in}}$" + UP),
#        ("mel_grid", r"$E_{\mathrm{grid}}$" + UP), ("nv_rest_in", r"$E_{\mathrm{in}}^{\mathrm{NV}}$" + UP),
#        ("gt_loc", "localisation" + UP), ("corr_in", r"$r_{\mathrm{in}}$")]
# write("f2_full.tex",
#       ["setting", r"$\lambda_{\mathrm{global}}$", r"$\lambda_{\mathrm{expert}}$"] + [h for _, h in MET] + ["test AUC" + UP, "test AP" + UP],
#       [[nm, lam(a), lam(b)] + [pm(sel(F7, a, b)[k]) for k, _ in MET] + [pm(sel(C7, a, b).auc), pm(sel(C7, a, b).ap)]
#        for a, b, nm in GRID], "lrr" + "c" * 8)

# write("table_nv_map.tex", ["setting", "n", "median [IQR]" + UP, "below 0" + DN, "p"],
#       [[tx(rn(st)), fi(len(v)), mi(v), fi((v < 0).sum()), fp(wilcoxon(v).pvalue)] for st, v in NVS.items()], "lrcrr")

# M61 = pd.read_csv(TAB / "ch6_metrics.csv")
# ROWS = [("auc_in", "in-lesion AUC", UP), ("energy_in", "energy excess, lesion", UP),
#         ("energy_grid", "energy excess, whole grid", UP), ("topk_exc", "top-k excess", UP),
#         ("struct_hit", "structures hit by a top-k cell", UP), ("peak_dist", "distance hottest cell to target", DN),
#         ("in_lesion", "heat inside lesion", UP)]
# METH = [("Grad-CAM", "Grad-CAM"), ("Diff-CAM", "Diff-CAM"), ("FinerCAM 0.6", r"FinerCAM ($\gamma = 0.6$)"), ("random", "random")]
# write("table_metric_validity.tex", ["metric"] + [h for _, h in METH],
#       [[tx(nm) + arr] + [mi(M61[M61.method == m][k]) for m, _ in METH] for k, nm, arr in ROWS], "lcccc")


# ---- cell 62 ------------------------------------------------------------
rows = []
for _, r in TEST_DF.sample(100, random_state=SEED).iterrows():
    m = cv2.resize((np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127).astype(np.uint8),
                   (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not m.any(): continue
    ys, xs = np.where(m)
    dx = min(223 - xs.max(), 60) if 223 - xs.max() > xs.min() else -min(xs.min(), 60)
    dy = min(223 - ys.max(), 60) if 223 - ys.max() > ys.min() else -min(ys.min(), 60)
    mir = (dx > 0 and xs.min() < dx) or (dx < 0 and 223 - xs.max() < -dx) or \
          (dy > 0 and ys.min() < dy) or (dy < 0 and 223 - ys.max() < -dy)
    rows.append({"dx": dx, "dy": dy, "dist": np.hypot(dx, dy), "mirror": mir})
S1 = pd.DataFrame(rows)
print(f"1. n={len(S1)}  |dx| median {S1.dx.abs().median():.0f} px [{S1.dx.abs().quantile(.25):.0f}, {S1.dx.abs().quantile(.75):.0f}]  "
      f"|dy| median {S1.dy.abs().median():.0f} px  distance median {S1.dist.median():.0f} px "
      f"[{S1.dist.quantile(.25):.0f}, {S1.dist.quantile(.75):.0f}]  max {S1.dist.max():.0f}  "
      f"at cap 60 on both axes: {int(((S1.dx.abs() == 60) & (S1.dy.abs() == 60)).sum())}  "
      f"mirrored lesion part possible: {int(S1.mirror.sum())}")


# ---- cell 63 ------------------------------------------------------------
import cv2, matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
def mask224(rel):
    m = np.array(Image.open((HAM / str(rel)).resolve()).convert("L")) > 127
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)

LOC = pd.read_csv(TAB / "ch5_localisation_test.csv")
q = LOC.pivot_table(index="image_id", columns="model", values="loc_auc")
q["gain"] = q["aligned"] - q["unaligned"]
q = q.join(LOC.drop_duplicates("image_id").set_index("image_id")["gt"])
IDS = None                                   # or ["ISIC_...", "ISIC_..."]
if IDS is None:
    IDS = [(g.gain - g.gain.median()).abs().idxmin() for _, g in q.groupby("gt")]
    # IDS = [g.index[np.argsort((g.gain - g.gain.quantile(qq)).abs().values)[0]]
    #    for _, g in q.groupby("gt") for qq in (0.25, 0.5, 0.75)]
print("2. images:", q.loc[IDS, ["gt", "unaligned", "aligned"]].round(3).to_dict("index"))

T = TEST_DF.assign(image_id=TEST_DF.image_id.astype(str)).set_index("image_id")
fig, ax = plt.subplots(len(IDS), 5, figsize=(TEXTWIDTH, TEXTWIDTH / 5 * len(IDS) + 0.6), squeeze=False)
for i, iid in enumerate(IDS):
    r = T.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    M = mask224(r.mask_rel_path).astype(float)
    ax[i, 0].imshow(rgb); ax[i, 0].contour(M, [.5], colors=[C_LESION], linewidths=1.0)
    ax[i, 0].set_ylabel(r.gt_label, fontsize=9)
    for j, m in enumerate([M0, M5]):
        for k, v in enumerate(cams.class_cams(m, x)):
            a = ax[i, 1 + 2 * j + k]; overlay(a, rgb, v); a.contour(M, [.5], colors=[C_LESION], linewidths=0.6)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
for j, t in enumerate(["image", "MEL map", "NV map", "MEL map", "NV map"]): ax[0, j].set_title(t, fontsize=8)
fig.subplots_adjust(left=0.04, right=0.995, bottom=0.01, top=0.84, wspace=0.04, hspace=0.04)
for (c0, c1), lab, col in [((1, 2), "Unaligned", C_UNAL), ((3, 4), "Globally aligned", C_GLOB)]:
    p0, p1 = ax[0, c0].get_position(), ax[-1, c1].get_position()
    x0, x1, y0 = p0.x0 - 0.006, p1.x1 + 0.006, p1.y0 - 0.008
    fig.add_artist(mpatches.Rectangle((x0, y0), x1 - x0, 0.995 - y0, transform=fig.transFigure,
                                      facecolor=to_rgba(col, 0.15), edgecolor="none", zorder=-1))
    fig.text((x0 + x1) / 2, 0.955, lab, ha="center", va="center", fontsize=9, fontweight="bold", color=col)
save(fig, "fig_ch5_localisation_examples"); plt.show()


# ---- cell 64 ------------------------------------------------------------
for p in sorted((REPO / "notebooks/mel_nv").glob("16*.ipynb")):
    for c in json.load(open(p))["cells"]:
        s = "".join(c["source"])
        if "fig_ch5_lambda_sweep" in s:
            print("-" * 50, p.name); print(s[:2500])


# ---- cell 65 ------------------------------------------------------------
grid = lambda m: m.reshape(14, 16, 14, 16).mean((1, 3)) >= 0.5
def share(v, G): v = cams.norm01(v); return float(v[G].sum() / (v.sum() + 1e-8))
rows = []
for _, r in TEST_DF.sample(100, random_state=SEED).iterrows():
    img = data.rgb224(r.image_rel_path)
    m = cv2.resize((np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127).astype(np.uint8),
                   (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not m.any(): continue
    ys, xs = np.where(m)
    dx = min(223 - xs.max(), 60) if 223 - xs.max() > xs.min() else -min(xs.min(), 60)
    dy = min(223 - ys.max(), 60) if 223 - ys.max() > ys.min() else -min(ys.min(), 60)
    W = np.float32([[1, 0, dx], [0, 1, dy]])
    s_img = cv2.warpAffine(img, W, (224, 224), borderMode=cv2.BORDER_REFLECT)
    s_m = cv2.warpAffine(m.astype(np.uint8), W, (224, 224), flags=cv2.INTER_NEAREST).astype(bool)
    s_ref = cv2.warpAffine(m.astype(np.uint8), W, (224, 224), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REFLECT).astype(bool)
    L, Mr = grid(s_m), grid(s_ref & ~s_m)
    for nm, mdl in [("unaligned", M0), ("aligned", M5)]:
        t = cams.target_ref(mdl, data.to_tensor(s_img), r.gt_label)[0]
        rows.append({"model": nm, "lesion": share(t, L), "mirror": share(t, Mr) if Mr.any() else np.nan,
                     "area": L.mean(), "mirror_area": Mr.mean(), "has_mirror": bool(Mr.any())})
SM = pd.DataFrame(rows); SM.to_csv(TAB / "app_shift_mirror.csv", index=False)
for nm, g in SM.groupby("model"):
    h = g[g.has_mirror]
    print(f"1. {nm:9} lesion {g.lesion.median():.3f} (area {g.area.median():.3f}) | mirror copy on grid in {len(h)} images: "
          f"heat {h.mirror.median():.3f} (area {h.mirror_area.median():.3f}) | lesion + mirror {(h.lesion + h.mirror).median():.3f}")


# ---- cell 66 ------------------------------------------------------------
sw = pd.read_csv(TAB / "ch5_lambda_sweep.csv")
lc = pd.read_csv(TAB / "ch5_lambda_localisation.csv")
co = pd.read_csv(TAB / "ch5_lambda_collapse.csv").groupby("lambda")[["corr_in", "sat_09"]].median().reset_index()
D = sw.merge(lc[["lambda", "loc_auc_median"]], on="lambda").merge(co, on="lambda").sort_values("lambda")
lam_ = D["lambda"].astype(float).values; X = np.where(lam_ == 0, 0.03, lam_); MAIN = 5.0
print("3.", D[["lambda", "loc_auc_median", "ap_drop_pct", "auc_drop_pct", "corr_in", "sat_09"]].round(3).to_string(index=False))
print("3. first lambda_global with AP drop above 10 %:", D.loc[D.ap_drop_pct > 10, "lambda"].min())

def xaxis(a):
    a.set_xscale("log"); a.minorticks_off(); a.set_xticks(X)
    a.set_xticklabels(["0" if v == 0 else f"{v:g}" for v in lam_], fontsize=7)
    a.axvline(MAIN, color="0.5", lw=0.8, ls=":", zorder=0)
    a.set_xlabel(r"$\lambda_{\mathrm{global}}$", fontsize=8)
def ring(a, y, col):
    k = np.flatnonzero(lam_ == MAIN)[0]; a.plot(X[k], y[k], "o", ms=8, mfc="none", mec=col, mew=1.2, zorder=5)

fig, a1 = plt.subplots(figsize=(TEXTWIDTH * 0.75, 2.6))
a1.plot(X, D.loc_auc_median, "-o", ms=3.5, color=C_GLOB, label="lesion localisation")
ring(a1, D.loc_auc_median.values, C_GLOB)
a1.set_ylim(0, 1.02); a1.set_ylabel("lesion localisation", color=C_GLOB, fontsize=8)
a2 = a1.twinx()
a2.plot(X, D.ap_drop_pct, "-s", ms=3.5, color=C_REF, label=r"relative AP drop vs $\lambda_{\mathrm{global}}=0$")
ring(a2, D.ap_drop_pct.values, C_REF)
a2.axhline(10, color=C_CHANCE, ls=":", lw=1)
a2.text(X[0], 11, "10 %", fontsize=7, color="0.45")
a2.set_ylim(0, D.ap_drop_pct.max() * 1.12); a2.set_ylabel("relative AP drop (%)", color=C_REF, fontsize=8)
xaxis(a1)
for a in (a1, a2): a.tick_params(labelsize=7)
h = [a1.get_lines()[0], a2.get_lines()[0]]
a1.legend(h, [l.get_label() for l in h], frameon=False, fontsize=7, loc="center left")
fig.tight_layout(); save(fig, "fig_ch4_sweep_main"); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.3))
for a, c, lab, yl, ch in [(ax[0], "corr_in", r"$r_{\mathrm{in}}$, MEL vs NV map", (-1, 1), 0.0),
                          (ax[1], "sat_09", r"$s_{0.9}$", (0, 1), None)]:
    a.plot(X, D[c], "-o", ms=3.5, color=C_GLOB); ring(a, D[c].values, C_GLOB); xaxis(a)
    a.set_ylabel(lab, fontsize=8); a.set_ylim(*yl); a.tick_params(labelsize=7)
    if ch is not None: a.axhline(ch, color=C_CHANCE, ls=":", lw=1)
fig.tight_layout(); save(fig, "fig_appC_sweep_details"); plt.show()


# ---- cell 67 ------------------------------------------------------------
DAL = sorted((REPO / "external").rglob("*dal*.pth"))
print("2. DAL checkpoints:", [str(p.relative_to(REPO)) for p in DAL])
from sklearn.metrics import roc_auc_score, average_precision_score
rows, M_ = [], {"global aligned": CK5 / "checkpoint-best-gap-ha5.pth", **{p.stem: p for p in DAL}}
for nm, p in M_.items():
    a = models.ckpt_args(p); m = models.load_ckpt(p, cache=False); pr = models.predict_mel(m, X_TEST)
    per = []
    for iid in sorted(HUM):
        r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]
        mel, nv = cams.class_cams(m, data.to_tensor(data.native(r.image_rel_path)))
        t = mel if r.gt_label == "MEL" else nv
        per.append((metrics.auc_in(t, L, U), metrics.energy_excess(t, L, U), metrics.corr_in(mel, nv, L)))
    per = np.array(per, float); del m
    rows.append({"model": nm, "ha_lambda": a.get("ha_lambda"), "dal_lambda": a.get("dal_lambda"),
                 "dal_mode": a.get("dal_mode", a.get("dal_target")), "epochs": a.get("epochs"), "lr": a.get("lr"),
                 "test_auc": roc_auc_score(Y_MEL, pr), "test_ap": average_precision_score(Y_MEL, pr),
                 "auc_in": np.nanmedian(per[:, 0]), "e_in": np.nanmedian(per[:, 1]), "r_in": np.nanmedian(per[:, 2])})
DT = pd.DataFrame(rows); DT.to_csv(TAB / "appC_dal.csv", index=False)
print(DT.round(3).to_string(index=False))


# ---- cell 68 ------------------------------------------------------------
P5 = TEST_DF.assign(image_id=TEST_DF.image_id.astype(str))[["image_id", "gt_label", "dataset", "dx_type"]].copy()
P5["p"] = models.predict_mel(M5, X_TEST); P5["pred"] = np.where(P5.p >= 0.5, "MEL", "NV")
P5["err"] = P5.pred != P5.gt_label
def tab(by):
    g = P5.groupby(by).agg(images=("err", "size"), MEL=("gt_label", lambda s: (s == "MEL").sum()),
                           missed_MEL=("err", lambda s: int((s & (P5.loc[s.index, "gt_label"] == "MEL")).sum())),
                           false_alarms=("err", lambda s: int((s & (P5.loc[s.index, "gt_label"] == "NV")).sum())))
    g["NV"] = g.images - g.MEL
    g["NV_error_rate"] = (g.false_alarms / g.NV.replace(0, np.nan)).round(3)
    g["pred_MEL_rate"] = P5.groupby(by).pred.apply(lambda s: (s == "MEL").mean()).round(3)
    return g[["images", "MEL", "NV", "missed_MEL", "false_alarms", "NV_error_rate", "pred_MEL_rate"]]
for by, nm in [("dataset", "site"), ("dx_type", "confirmation")]:
    t = tab(by); t.to_csv(TAB / f"appA_errors_{nm}.csv")
    write(f"appA_errors_{nm}.tex", [nm, "images", "MEL", "NV", "missed MEL", "false alarms", "NV error rate", "predicted MEL rate"],
          [[tx(i), fi(r.images), fi(r.MEL), fi(r.NV), fi(r.missed_MEL), fi(r.false_alarms), f3(r.NV_error_rate),
            f3(r.pred_MEL_rate)] for i, r in t.iterrows()], "lrrrrrrr")
    print(f"4. by {nm}\n", t.to_string())


# ---- cell 69 ------------------------------------------------------------
A3 = pd.DataFrame({k: models.ckpt_args(p) for k, p in
                   {"main": CK5 / "checkpoint-best-gap-ha5.pth",
                    "dal0.001": DAL[0], "ha5+dal0.5": DAL[3]}.items()})
d = A3[A3.astype(str).nunique(axis=1) > 1]
print(d.astype(str).to_string())


# ---- cell 70 ------------------------------------------------------------
import datetime as dt
from pypdf import PdfReader
D8 = REPO / "outputs/mel_nv/clinician_model_choice_08b"
rows = []
for p in sorted((D8 / "pdf").glob("08b_model_choice_*.pdf")):
    st = p.stat()
    rows.append({"file": p.name, "pages": len(PdfReader(p).pages),
                 "modified": dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")})
PD = pd.DataFrame(rows)
print(PD.to_string(index=False) if len(PD) else "1. no 08b PDFs found under " + str(D8 / "pdf"))
if len(PD): print(f"1. total images judged: {PD.pages.sum()}  (PDF dates {PD.modified.min()} to {PD.modified.max()})")
for f in sorted((D8 / "csv").glob("08b_*.csv")):
    print(f"1. {f.name}: {len(pd.read_csv(f))} rows")


# ---- cell 71 ------------------------------------------------------------
import json
keys = sorted(REPO.rglob("*clinician_study*KEY*.csv")) + sorted(REPO.rglob("*clinician_study_key*.csv"))
print("2. key files:", [str(k.relative_to(REPO)) for k in keys])
K = pd.read_csv(next(k for k in keys if "round2" not in k.name))
print("2. columns:", list(K.columns), " parts:", K["part"].unique().tolist() if "part" in K else "no part column")
k2 = K[K["part"].astype(str).isin(["2", "B", "part 2", "Part 2"])] if "part" in K else K[K.case_no > 10]
print(f"2. Part 2 cases: {k2.case_no.nunique()}")
print(pd.crosstab(k2.method, k2.letter, margins=True).to_string())

# seed: printed config value, and the seed line in the notebook that wrote the key
from src.thesis.config import SEED as S0
print("2. config SEED:", S0)
for p in sorted((REPO / "notebooks/mel_nv").glob("16e*.ipynb")):
    for c in json.load(open(p))["cells"]:
        s = "".join(c["source"])
        if "KEY" in s or "orders(" in s:
            print("-" * 40, p.name)
            print("\n".join(l.strip() for l in s.splitlines() if any(t in l for t in ("SEED", "seed", "permutation", "orders("))))


# ---- cell 72 ------------------------------------------------------------
MC0 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha0.pth", cache=False)
MC5 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha5.pth", cache=False)
assert models.ckpt_args(CK5 / "checkpoint-best-cls-ha0.pth").get("pooling") == "cls"
x0 = data.to_tensor(data.native(EV.loc[sorted(HUM)[0]].image_rel_path))
print("check block 12 = class_cams:", np.allclose(cams.class_cams(MC5, x0)[0].reshape(14, 14),
                                                   block_maps(MC5, x0, 11)[0], atol=1e-4))
rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L = LESION[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for nm, m in [("unaligned CLS", MC0), ("global aligned CLS", MC5)]:
        for b in range(12):
            mel, nv = block_maps(m, x, b)
            rows.append({"model": nm, "block": b + 1, "image_id": iid,
                         "mel_loc": metrics.lesion_auc(mel, L), "nv_loc": metrics.lesion_auc(nv, L),
                         "r_in": metrics.corr_in(mel, nv, L)})
LC2 = pd.DataFrame(rows); LC2.to_csv(TAB / "appC_layers_cls.csv", index=False)
S = LC2.groupby(["model", "block"])[["mel_loc", "nv_loc", "r_in"]].median()
Q = LC2.groupby(["model", "block"])[["mel_loc", "nv_loc"]].quantile([.25, .75]).unstack()
print(S.round(3).unstack(0).to_string())
for nm in S.index.levels[0]:
    s = S.loc[nm]
    print(f"{nm:20} block 1: MEL {s.mel_loc[1]:.3f} NV {s.nv_loc[1]:.3f} | block 12: MEL {s.mel_loc[12]:.3f} "
          f"NV {s.nv_loc[12]:.3f} | peak MEL block {int(s.mel_loc.idxmax())} ({s.mel_loc.max():.3f}), "
          f"peak NV block {int(s.nv_loc.idxmax())} ({s.nv_loc.max():.3f})")

fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4), sharey=True)
for a, (nm, title) in zip(ax, [("unaligned CLS", "(a) unaligned, CLS pooling"),
                              ("global aligned CLS", "(b) global aligned, CLS pooling")]):
    for c, col, lab in [("mel_loc", C_MEL, "MEL map"), ("nv_loc", C_NV, "NV map")]:
        s = S.loc[nm][c]
        a.plot(s.index, s.values, "-o", ms=2.5, lw=1, color=col, label=lab)
        a.fill_between(s.index, Q.loc[nm][(c, .25)], Q.loc[nm][(c, .75)], color=col, alpha=0.18)
    a.axhline(0.5, color=C_CHANCE, ls=":", lw=0.8)
    a.set(ylim=(0, 1), xticks=range(1, 13), xlabel="transformer block"); a.set_title(title, fontsize=8)
    a.tick_params(labelsize=7)
ax[0].set_ylabel("lesion localisation", fontsize=8); ax[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); save(fig, "fig_appC_layers_cls"); plt.show()


# ---- cell 73 ------------------------------------------------------------
import importlib, json, re; importlib.reload(plotting)
P = {k: getattr(plotting, k) for k in ["C_MEL", "C_NV", "C_UNAL", "C_GLOB", "C_EXP", "C_JOINT", "C_REF", "C_CHANCE", "C_LESION"]}
P.update({f"method {k}": v for k, v in plotting.METHOD_COLOR.items() if k != "random"})
print(pd.Series(P).to_string())
EXPECT = {"C_MEL": "#d62728", "C_NV": "#1f77b4", "C_GLOB": "#ff7f0e", "C_UNAL": "#7f7f7f", "C_EXP": "#9467bd"}
bad = {k: (P[k], v) for k, v in EXPECT.items() if P[k].lower() != v}
print("palette mismatches:", bad or "none")
dup = pd.Series(P).str.lower().value_counts(); print("colours used twice:", dup[dup > 1].index.tolist() or "none")
for p in sorted((REPO / "notebooks/mel_nv").glob("16*.ipynb")):
    for c in json.load(open(p))["cells"]:
        s = "".join(c["source"]); figs = re.findall(r'save\(fig,\s*"([^"]+)"', s)
        if figs and re.search(r"C_FOCUS|C_CONTROL", s): print("legacy colour in", p.name, figs)


# ---- cell 74 ------------------------------------------------------------
import json, re
USE = {"fig_appA_sites", "fig_appA_lesion_size", "fig_appA_target_density", "fig_appC_layers",
       "fig_appC_layers_cls", "fig_ch5_deletion", "fig_ch6_group_ranking", "fig_ch6_group_examples", "fig_ch7_tradeoff"}
for p in sorted((REPO / "notebooks/mel_nv").glob("16*.ipynb")):
    for ci, c in enumerate(json.load(open(p))["cells"]):
        s = "".join(c["source"]); figs = set(re.findall(r'save\(fig,\s*"([^"]+)"', s)) & USE
        if not figs: continue
        hits = [(i + 1, l.strip()) for i, l in enumerate(s.splitlines()) if re.search(r"C_FOCUS|C_CONTROL", l)]
        if hits:
            print(f"--- {p.name}, cell {ci}, {sorted(figs)}")
            for n, l in hits: print(f"  line {n}: {l}")


# ---- cell 75 ------------------------------------------------------------
b = LC2[LC2.block == 12].merge(EV[["gt_label"]], left_on="image_id", right_index=True)
b["gt_loc"] = np.where(b.gt_label == "MEL", b.mel_loc, b.nv_loc)
print(b.groupby("model").gt_loc.median().round(3).to_dict())
print(b.groupby(["model", "gt_label"])[["mel_loc", "nv_loc"]].median().round(3).to_string())


# ---- cell 76 ------------------------------------------------------------
import json, re
# 2. what ha_fp_weight does in the loss
for line in open(REPO / "external/PanDerm/classification/run_class_finetuning_ha.py"):
    if re.search(r"fp_weight|def .*dice|N_FP|false_pos", line): print("2.", line.rstrip())

# 3. weak global setting on the grid
WK = {k: p for (a, b, k), p in IDX.items() if np.isclose(a, 0.1) and np.isclose(b, 1.0)}
print("3. weak global folds found:", sorted(WK))
if WK:
    rows, cls = [], []
    for k, p in sorted(WK.items()):
        m = models.load_ckpt(p, cache=False); pr = models.predict_mel(m, X_TEST)
        cls.append({"fold": k, "auc": roc_auc_score(Y_MEL, pr), "ap": average_precision_score(Y_MEL, pr)})
        for iid in [i for i in held if FOLD[i] == k]:
            r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]
            mel, nv = cams.class_cams(m, data.to_tensor(data.native(r.image_rel_path)))
            t = mel if r.gt_label == "MEL" else nv
            rows.append({"setting": "weak global", "image_id": iid, "gt": r.gt_label,
                         "mel_auc_in": metrics.auc_in(mel, L, U), "mel_in": metrics.energy_excess(mel, L, U),
                         "mel_grid": egrid(mel, U & L), "corr_in": metrics.corr_in(mel, nv, L),
                         "gt_loc": metrics.lesion_auc(t, L),
                         "nv_rest_in": metrics.energy_excess(nv, L, L & ~U) if r.gt_label == "NV" else np.nan})
        del m
    WG = pd.DataFrame(rows); WG.to_csv(TAB / "ch7_weak_global.csv", index=False)
    C7 = pd.read_csv(TAB / "ch7_classification.csv")
    ref = F7[F7.setting == "HA2 b1"].merge(EV[["gt_label"]], left_on="image_id", right_index=True)
    for nm, d in [("weak global", WG.rename(columns={"gt": "gt_label"})), ("expert only b1", ref)]:
        print(f"3. {nm:15} AUC_in {miqr(d.mel_auc_in)}  E_in {miqr(d.mel_in)}  E_grid {miqr(d.mel_grid)}  r_in {miqr(d.corr_in)}")
        print(f"3. {'':15} loc MEL {miqr(d[d.gt_label == 'MEL'].gt_loc)}  loc NV {miqr(d[d.gt_label == 'NV'].gt_loc)}  NV E_in {miqr(d.nv_rest_in)}")
    c = pd.DataFrame(cls); e = sel(C7, 0, 1)
    print(f"3. test AUC weak {c.auc.mean():.3f} ± {c.auc.std():.3f}  AP {c.ap.mean():.3f} ± {c.ap.std():.3f} | "
          f"expert only AUC {e.auc.mean():.3f} ± {e.auc.std():.3f}  AP {e.ap.mean():.3f} ± {e.ap.std():.3f}")

# 4. nevus anchor rule
FM = data.load_feat_manifest()
print("4. rules in manifest:", FM.rule.value_counts().to_dict())
print("4. runs mentioning anchor:", [p.name for p in REPO.rglob("*anchor*") if p.suffix in (".pth", ".csv", ".sh")][:10])

# 5. melanoma structures on the 18 nevi
raw = pd.read_csv(QC_CSV); q = raw[raw.qc_pass.astype(bool)]
nv = q[q.gt_label == "NV"]; GRP_ = {i: GRP.get(i, "not recorded") for i in nv.Image_ID.unique()}
H0 = pd.read_csv(MELNV / "ham_mel_nv_clean.csv").set_index("image_id")
img = pd.DataFrame({"image_id": sorted(raw[raw.gt_label == "NV"].Image_ID.unique())})
img["group"] = img.image_id.map(GRP); img["dx_type"] = img.image_id.map(H0.dx_type)
img["mel_labels"] = img.image_id.map(lambda i: sorted(set(nv[(nv.Image_ID == i) & (nv.label_group == "MEL")].label)))
img["has_mel"] = img.mel_labels.map(len) > 0
print(f"5. nevi {len(img)}, with melanoma structure {int(img.has_mel.sum())}")
print(pd.crosstab([img.group, img.dx_type], img.has_mel, margins=True).to_string())
print("5. labels:", nv[nv.label_group == "MEL"].label.value_counts().to_dict())

# 6 and 7. depth and activation Grad-CAM in the code
for p in list((REPO / "notebooks/mel_nv").glob("1[3-6]*.ipynb")) + list((REPO / "src/thesis").glob("*.py")):
    src = p.read_text() if p.suffix == ".py" else "".join("".join(c["source"]) for c in json.load(open(p))["cells"])
    for kw in ["depth", "distance_transform", "edt", "pytorch_grad_cam", "GradCAM(", "activation"]:
        if kw in src: print(f"6/7. {p.name}: '{kw}'")
for f in TAB.glob("*.csv"):
    if re.search(r"depth|activation|act_gradcam", f.name): print("6/7. csv:", f.name)


# ---- cell 77 ------------------------------------------------------------
import json
FM = data.load_feat_manifest(); FM["image_id"] = FM.image_id.astype(str).str.strip()
RULE = FM.drop_duplicates("image_id").set_index("image_id").rule
ANN = set(QC.Image_ID)
for k in range(5):
    d = pd.read_csv(MELNV / f"f2_fold{k}.csv"); d["image_id"] = d.image_id.astype(str).str.strip()
    t = d[(d.split == "train") & (d.has_feat == 1)]
    print(f"4. fold {k}: has_feat images {len(t)}, annotated {int(t.image_id.isin(ANN).sum())}, "
          f"by rule {t.image_id.map(RULE).fillna('no rule').value_counts().to_dict()}")
print("4. anchor images that are annotated:", int(FM[FM.rule == "nevus_anchor"].image_id.isin(ANN).sum()), "of 40")

for nb, kws in [("16c_explanations.ipynb", ["depth", "distance_transform"]),
                ("14b_follow_up.ipynb", ["depth", "distance_transform", "activation"]),
                ("14d.ipynb", ["depth", "activation"]), ("13_structure_analysis.ipynb", ["activation"])]:
    for c in json.load(open(REPO / "notebooks/mel_nv" / nb))["cells"]:
        hits = [l.strip() for l in "".join(c["source"]).splitlines() if any(k in l for k in kws)]
        if hits: print(f"--- {nb}"); print("\n".join(hits[:12]))
for f in TAB.glob("*.csv"):
    if any(k in f.name for k in ("depth", "act")): print("csv:", f.name, list(pd.read_csv(f, nrows=1).columns))


# ---- cell 78 ------------------------------------------------------------
import json
nb = json.load(open(REPO / "notebooks/mel_nv/16c_explanations.ipynb"))
s = next("".join(c["source"]) for c in nb["cells"] if "rows_depth" in "".join(c["source"]))
i = s.find("d = distance_transform_edt(L)"); print("6. 16c depth block:\n" + s[i - 200:i + 500])
nb14 = json.load(open(REPO / "notebooks/mel_nv/14d.ipynb"))
s = next("".join(c["source"]) for c in nb14["cells"] if "def depth_map" in "".join(c["source"]))
i = s.find("def depth_map"); print("6. 14d depth_map:\n" + "\n".join(s[i:].splitlines()[:8]))
MR = pd.read_csv(REPO / "results/overlap/main_results_table_squash224.csv")
print("7. columns:", list(MR.columns))
print(MR.head(12).to_string())


# ---- cell 79 ------------------------------------------------------------
FM = data.load_feat_manifest(); FM["image_id"] = FM.image_id.astype(str).str.strip()
RULE = FM.drop_duplicates("image_id").set_index("image_id").rule
rows = []
for k in range(5):
    d = pd.read_csv(MELNV / f"f2_fold{k}.csv"); d["image_id"] = d.image_id.astype(str).str.strip()
    t = d[(d.split == "train") & (d.has_feat == 1)].copy()
    t["rule"] = t.image_id.map(RULE)
    for (ru, gt), n in t.groupby(["rule", "gt_label"]).size().items():
        rows.append({"fold": k, "rule": ru, "gt": gt, "n": n})
CB = pd.DataFrame(rows).pivot_table(index="fold", columns=["rule", "gt"], values="n", fill_value=0).astype(int)
CB["total"] = CB.sum(axis=1)
print(CB.to_string())
print("mel_only nevi:", sorted(set(FM[(FM.rule == "mel_only")].image_id) & set(EV[EV.gt_label == "NV"].index)))


# ---- cell 80 ------------------------------------------------------------
ST = pd.read_csv(TAB / "appD_statistics.csv")
ST = ST[ST.comparison != "depth, annotated vs unannotated lesion cells"]
ST = pd.concat([ST, pd.DataFrame([dict(chapter=5, comparison="depth, annotated vs unannotated lesion cells",
                metric="normalised depth", n=31, median_diff=0.222, p=9.3e-10, family=1,
                test="Wilcoxon", source="notebook 16c")])], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fp(r.p), fi(r.family), fp(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")
print(f"appD: {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())}")


# ---- cell 81 ------------------------------------------------------------
def act_cams(model, x):
    base = cams._unwrap_model(model); acts = {}
    h = base.blocks[11].norm1.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    try:
        with torch.enable_grad():
            out = model(x); logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, (tuple, list)) else out)
            A = acts["a"]
            res = []
            for c in (MEL_IDX, NV_IDX):
                g = torch.autograd.grad(logits[0, c], A, retain_graph=True)[0][0, 1:]   # patch tokens
                w = g.mean(0)
                cam = torch.relu((A[0, 1:] * w).sum(-1)).detach().cpu().numpy().reshape(14, 14)
                res.append(cams.norm01(cam))
    finally:
        h.remove(); base.clear_xai_state()
    return res

rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for nm, m in [("unaligned", M0), ("global aligned", M5)]:
        mel, nv = act_cams(m, x); t = mel if r.gt_label == "MEL" else nv
        rows.append({"model": nm, "image_id": iid, "auc_in": metrics.auc_in(t, L, U),
                     "energy_in": metrics.energy_excess(t, L, U), "r_in": metrics.corr_in(mel, nv, L),
                     "r_all": float(np.corrcoef(mel.ravel(), nv.ravel())[0, 1])})
AC = pd.DataFrame(rows); AC.to_csv(TAB / "appC_activation_cams.csv", index=False)
for nm, g in AC.groupby("model"):
    print(f"3. {nm:15} AUC_in {miqr(g.auc_in)}  E_in {miqr(g.energy_in)}  r_in {miqr(g.r_in)}  r_all {miqr(g.r_all)}")


# ---- cell 82 ------------------------------------------------------------
for iid in ["ISIC_0030225", "ISIC_0033556"]:
    r = FM[FM.image_id == iid].iloc[0]; L, U = LESION[iid], HUM[iid]
    def px(rel):
        if not isinstance(rel, str) or not rel: return None
        p = (HAM / rel).resolve(); return int((np.array(Image.open(p).convert("L")) > 127).sum()) if p.exists() else -1
    labs = sorted(QC[QC.Image_ID == iid].label.unique())
    print(f"{iid} rule {r.rule} | labels {labs}")
    print(f"  grid: lesion cells {int(L.sum())}, MEL structure cells in lesion {int((U & L).sum())}, "
          f"lesion minus structures {int((L & ~U).sum())}")
    print(f"  pixels: T_MEL {px(r.feat_mel_rel_path)}, T_NV {px(r.feat_nv_rel_path)}, "
          f"lesion mask {int((np.array(Image.open((HAM / str(EV.loc[iid].mask_rel_path)).resolve()).convert('L')) > 127).sum())}")


# ---- cell 83 ------------------------------------------------------------
import json, re
for iid in ["ISIC_0030225", "ISIC_0033556"]:
    L, U = LESION[iid], HUM[iid]
    rest = L & ~U
    r = FM[FM.image_id == iid].iloc[0]
    print(f"{iid}: lesion cells {int(L.sum())}, MEL structure cells in lesion {int((U & L).sum())}, "
          f"remainder {int(rest.sum())}  | rule {r.rule}  nv_path {'set' if isinstance(r.feat_nv_rel_path, str) and r.feat_nv_rel_path else 'empty'}")
    px = QC[QC.Image_ID == iid].groupby("label_group").px_positive.sum().to_dict()
    print(f"   pixels by group {px}, labels {sorted(set(QC[QC.Image_ID == iid].label))}")

# the rule that assigns mel_only vs mel_on_nevus
for p in sorted((REPO / "notebooks/mel_nv").glob("*.ipynb")) + sorted((REPO / "src/thesis").glob("*.py")):
    src = p.read_text() if p.suffix == ".py" else "".join("".join(c["source"]) for c in json.load(open(p))["cells"])
    if "mel_on_nevus" in src and "rule" in src:
        i = src.find("mel_on_nevus")
        print("-" * 50, p.name); print(src[max(0, i - 1200):i + 600])
        break


# ---- cell 84 ------------------------------------------------------------
import cv2
from matplotlib.lines import Line2D
FOLD = data.fold_of_heldout(); held = [i for i in sorted(HUM) if i in FOLD]
FR = np.round(np.arange(0, 1.0001, 0.05), 2); SEEDS = [SEED + s for s in range(5)]

def remove(img, cells):
    mk = np.kron(cells.astype(np.uint8), np.ones((16, 16), np.uint8)) * 255
    return cv2.inpaint(img, mk, 7, cv2.INPAINT_TELEA) if mk.any() else img

def gaps(m, imgs, gt):
    with torch.no_grad():
        o = m(torch.cat([data.to_tensor(i) for i in imgs]))
        l = o["logits"] if isinstance(o, dict) else (o[0] if isinstance(o, (tuple, list)) else o)
    a, b = (MEL_IDX, NV_IDX) if gt == "MEL" else (NV_IDX, MEL_IDX)
    return (l[:, a] - l[:, b]).float().cpu().numpy()

def curve(m, img, gt, order, base):
    imgs = []
    for f in FR:
        c = np.zeros(196, bool); c[order[:int(round(f * len(order)))]] = True
        imgs.append(remove(img, c.reshape(14, 14)))
    return (base - gaps(m, imgs, gt)) / max(abs(base), 0.1)

rows = []
for nm in ["unaligned", "global aligned", "expert aligned"]:
    for k in range(5):
        mk = {"unaligned": M0, "global aligned": M5}.get(nm) or models.f2(0.0, 1.0, k)
        for iid in [i for i in held if FOLD[i] == k]:
            r = EV.loc[iid]; img = data.rgb224(r.image_rel_path); L = LESION[iid]
            lidx = np.flatnonzero(L); gt = r.gt_label
            base = float(gaps(mk, [img], gt)[0])
            t = cams.target_ref(mk, data.to_tensor(img), gt)[0].ravel()
            dmap = curve(mk, img, gt, lidx[np.argsort(-t[lidx], kind="stable")], base)
            drnd = np.mean([curve(mk, img, gt, np.random.default_rng(s).permutation(lidx), base) for s in SEEDS], axis=0)
            for f, a, b in zip(FR, dmap, drnd):
                rows.append({"model": nm, "image_id": iid, "frac": f, "map": a, "random": b})
        if nm == "expert aligned": del mk
DC = pd.DataFrame(rows); DC.to_csv(TAB / "ch5_deletion_curves.csv", index=False)

# auc = (DC.groupby(["model", "image_id"])
#          .apply(lambda g: pd.Series({"auc_map": np.trapezoid(g.sort_values("frac")["map"], g.sort_values("frac").frac),
#                                      "auc_random": np.trapezoid(g.sort_values("frac")["random"], g.sort_values("frac").frac)}))
#          .reset_index())
auc = (DC.groupby(["model", "image_id"])
         .apply(lambda g: pd.Series({
             "auc_map": np.trapz(g.sort_values("frac")["map"], g.sort_values("frac").frac),
             "auc_random": np.trapz(g.sort_values("frac")["random"], g.sort_values("frac").frac)
         }))
         .reset_index())
auc.to_csv(TAB / "ch5_deletion_curves_auc.csv", index=False)
for nm in ["unaligned", "global aligned", "expert aligned"]:
    g = auc[auc.model == nm]
    print(f"deletion AUC {nm:15} map {miqr(g.auc_map)}  random {miqr(g.auc_random)}  "
          f"map vs random {metrics.paired(g.auc_map, g.auc_random)}")
q10 = DC[DC.frac == 0.10].groupby("model")[["map", "random"]].median().round(3)
print("check at 10 %:\n", q10.to_string())

fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharey=True)
for a, nm in zip(ax, ["unaligned", "global aligned", "expert aligned"]):
    g = DC[DC.model == nm].groupby("frac"); col = plotting.MODEL_COLOR[nm]
    a.plot(FR * 100, g["map"].median(), "-", color=col, lw=1.4)
    a.fill_between(FR * 100, g["map"].quantile(.25), g["map"].quantile(.75), color=col, alpha=0.15)
    a.plot(FR * 100, g["random"].median(), "--", color=col, lw=1.2)
    a.axhline(0, color=plotting.C_CHANCE, ls=":", lw=0.8)
    a.set_title(nm, fontsize=8); a.set_xlabel("lesion cells removed (%)", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("normalised logit gap drop", fontsize=8)
ax[0].legend([Line2D([], [], color="0.3", ls="-"), Line2D([], [], color="0.3", ls="--")],
             ["map order", "random order (5 seeds)"], frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); save(fig, "fig_appC_deletion_curves"); plt.show()


# ---- cell 85 ------------------------------------------------------------
rng0 = np.random.default_rng(SEED)
def order_of(v, lidx):
    return lidx[np.lexsort((rng0.random(len(lidx)), -v[lidx]))]

rows, empty = [], []
for nm in ["unaligned", "global aligned", "expert aligned"]:
    for k in range(5):
        mk = {"unaligned": M0, "global aligned": M5}.get(nm) or models.f2(0.0, 1.0, k)
        for iid in [i for i in held if FOLD[i] == k]:
            r = EV.loc[iid]; img = data.rgb224(r.image_rel_path); L = LESION[iid]; gt = r.gt_label
            lidx = np.flatnonzero(L); x = data.to_tensor(img); base = float(gaps(mk, [img], gt)[0])
            gc = cams.target_ref(mk, x, gt)[0].ravel()
            fc = np.asarray(cams.finer_cam(mk, x, gt, 0.6)).ravel()
            for meth, v in [("Grad-CAM", gc), ("FinerCAM 0.6", fc)]:
                if v[lidx].max() < 1e-6:
                    empty.append((nm, meth, iid)); continue
                for f, d in zip(FR, curve(mk, img, gt, order_of(v, lidx), base)):
                    rows.append({"model": nm, "method": meth, "image_id": iid, "frac": f, "drop": d})
        if nm == "expert aligned": del mk
DM = pd.DataFrame(rows)
RND = DC[["model", "image_id", "frac", "random"]]
DM = DM.merge(RND, on=["model", "image_id", "frac"]); DM.to_csv(TAB / "ch5_deletion_curves_methods.csv", index=False)
print("empty maps skipped:", pd.Series([f"{a} | {b}" for a, b, _ in empty]).value_counts().to_dict() or "none")

def area(g, c): g = g.sort_values("frac"); return np.trapezoid(g[c], g.frac)
# AM = (DM.groupby(["model", "method", "image_id"])
#         .apply(lambda g: pd.Series({"auc": area(g, "drop"), "auc_random": area(g, "random"),
#                                     "at05": g.loc[g.frac == 0.05, "drop"].iloc[0], "at10": g.loc[g.frac == 0.10, "drop"].iloc[0]}))
#         .reset_index())
AM = (DM.groupby(["model", "method", "image_id"])
        .apply(lambda g: pd.Series({
            "auc": np.trapz(g.sort_values("frac")["drop"], g.sort_values("frac").frac),
            "auc_random": np.trapz(g.sort_values("frac")["random"], g.sort_values("frac").frac),
            "at05": g.loc[g.frac == 0.05, "drop"].iloc[0], 
            "at10": g.loc[g.frac == 0.10, "drop"].iloc[0]
        }))
        .reset_index())

AM.to_csv(TAB / "ch5_deletion_curves_methods_auc.csv", index=False)
for nm in ["unaligned", "global aligned", "expert aligned"]:
    for meth in ["Grad-CAM", "FinerCAM 0.6"]:
        g = AM[(AM.model == nm) & (AM.method == meth)]
        print(f"{nm:15} {meth:13} AUC {miqr(g.auc)}  at 5 % {g.at05.median():.3f}  at 10 % {g.at10.median():.3f}  "
              f"vs random {metrics.paired(g.auc, g.auc_random)}")
    q = AM[AM.model == nm].pivot_table(index="image_id", columns="method", values="auc").dropna()
    print(f"{nm:15} FinerCAM vs Grad-CAM {metrics.paired(q['FinerCAM 0.6'], q['Grad-CAM'])}\n")


# ---- cell 86 ------------------------------------------------------------
from matplotlib.lines import Line2D
MODELS = ["unaligned", "global aligned", "expert aligned"]
fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharey=True)
for a, nm in zip(ax, MODELS):
    for meth in ["Grad-CAM", "FinerCAM 0.6"]:
        g = DM[(DM.model == nm) & (DM.method == meth)].groupby("frac")["drop"]; col = plotting.METHOD_COLOR[meth]
        a.plot(FR * 100, g.median().values, "-", color=col, lw=1.4)
        a.fill_between(FR * 100, g.quantile(.25).values, g.quantile(.75).values, color=col, alpha=0.12)
    a.plot(FR * 100, DC[DC.model == nm].groupby("frac")["random"].median().values, "--", color="0.55", lw=1.3)
    a.axhline(0, color=plotting.C_CHANCE, ls=":", lw=0.8)
    a.set_title(nm, fontsize=8); a.set_xlabel("lesion cells removed (%)", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("normalised logit gap drop", fontsize=8)
ax[0].legend([Line2D([], [], color=plotting.METHOD_COLOR["Grad-CAM"]), Line2D([], [], color=plotting.METHOD_COLOR["FinerCAM 0.6"]),
              Line2D([], [], color="0.55", ls="--")], ["Grad-CAM order", r"FinerCAM order ($\gamma=0.6$)", "random order"],
             frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); save(fig, "fig_appC_deletion_curves"); plt.show()

ST = pd.read_csv(TAB / "appD_statistics.csv")
ST = ST[~ST.comparison.str.startswith("deletion curve")]
new = []
for nm in MODELS:
    for meth in ["Grad-CAM", "FinerCAM 0.6"]:
        g = AM[(AM.model == nm) & (AM.method == meth)]; r = metrics.paired(g.auc, g.auc_random)
        new.append(dict(chapter=4, comparison=f"deletion curve, {meth} order vs random, {nm}", metric="area under drop curve",
                        n=r["n"], median_diff=r["diff"], p=r["p"], family=6, test="Wilcoxon", source="ch5_deletion_curves_methods_auc.csv"))
    q = AM[AM.model == nm].pivot_table(index="image_id", columns="method", values="auc").dropna()
    r = metrics.paired(q["FinerCAM 0.6"], q["Grad-CAM"])
    new.append(dict(chapter=4, comparison=f"deletion curve, FinerCAM vs Grad-CAM order, {nm}", metric="area under drop curve",
                    n=r["n"], median_diff=r["diff"], p=r["p"], family=3, test="Wilcoxon", source="ch5_deletion_curves_methods_auc.csv"))
ST = pd.concat([ST, pd.DataFrame(new)], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fp(r.p), fi(r.family), fp(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")
print(f"appD: {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())}")


# ---- cell 87 ------------------------------------------------------------
print(AM.groupby(["model", "method"]).auc_random.median().round(3).to_string())


# ---- cell 88 ------------------------------------------------------------
both = DM.pivot_table(index=["model", "image_id", "frac"], columns="method", values="drop").dropna().reset_index()
end = both[both.frac == 1.0].merge(DC[DC.frac == 1.0][["model", "image_id", "random"]], on=["model", "image_id"])
end["max_gap"] = end[["Grad-CAM", "FinerCAM 0.6", "random"]].max(axis=1) - end[["Grad-CAM", "FinerCAM 0.6", "random"]].min(axis=1)
print("endpoint mismatch per image, max:", end.max_gap.max().round(4), "| images with gap > 0.01:", int((end.max_gap > 0.01).sum()))
print(end.groupby("model")[["Grad-CAM", "FinerCAM 0.6", "random"]].median().round(3).to_string())
print("excluded expert images, endpoint drop:",
      DC[(DC.model == "expert aligned") & (DC.frac == 1.0) &
         ~DC.image_id.isin(both[both.model == "expert aligned"].image_id)][["image_id", "random"]].round(3).to_dict("records"))


# ---- cell 89 ------------------------------------------------------------
from matplotlib.lines import Line2D
MODELS = ["unaligned", "global aligned", "expert aligned"]
keep = (AM.pivot_table(index=["model", "image_id"], columns="method", values="auc").dropna()
          .reset_index()[["model", "image_id"]])
AMs = AM.merge(keep, on=["model", "image_id"]); DMs = DM.merge(keep, on=["model", "image_id"])
DCs = DC.merge(keep, on=["model", "image_id"])

res = []
for nm in MODELS:
    for meth in ["Grad-CAM", "FinerCAM 0.6"]:
        g = AMs[(AMs.model == nm) & (AMs.method == meth)]; r = metrics.paired(g.auc, g.auc_random)
        res.append(dict(model=nm, method=meth, n=r["n"], auc=g.auc.median(), auc_random=g.auc_random.median(),
                        diff=r["diff"], p=r["p"], family=6))
    q = AMs[AMs.model == nm].pivot_table(index="image_id", columns="method", values="auc")
    r = metrics.paired(q["FinerCAM 0.6"], q["Grad-CAM"])
    res.append(dict(model=nm, method="FinerCAM vs Grad-CAM", n=r["n"], diff=r["diff"], p=r["p"], family=3))
RS = pd.DataFrame(res); RS["survives"] = RS.p < 0.05 / RS.family
RS.to_csv(TAB / "ch5_deletion_curves_shared.csv", index=False)
print(RS.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharey=True)
for a, nm in zip(ax, MODELS):
    n = keep[keep.model == nm].shape[0]
    for meth in ["Grad-CAM", "FinerCAM 0.6"]:
        g = DMs[(DMs.model == nm) & (DMs.method == meth)].groupby("frac")["drop"]; col = plotting.METHOD_COLOR[meth]
        a.plot(FR * 100, g.median().values, "-", color=col, lw=1.4)
        a.fill_between(FR * 100, g.quantile(.25).values, g.quantile(.75).values, color=col, alpha=0.12)
    a.plot(FR * 100, DCs[DCs.model == nm].groupby("frac")["random"].median().values, "--", color="0.55", lw=1.3)
    a.axhline(0, color=plotting.C_CHANCE, ls=":", lw=0.8)
    a.set_title(f"{nm} (n = {n})", fontsize=8); a.set_xlabel("lesion cells removed (%)", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("normalised logit gap drop", fontsize=8)
ax[0].legend([Line2D([], [], color=plotting.METHOD_COLOR["Grad-CAM"]), Line2D([], [], color=plotting.METHOD_COLOR["FinerCAM 0.6"]),
              Line2D([], [], color="0.55", ls="--")], ["Grad-CAM order", r"FinerCAM order ($\gamma=0.6$)", "random order"],
             frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); save(fig, "fig_appC_deletion_curves"); plt.show()

ST = pd.read_csv(TAB / "appD_statistics.csv"); ST = ST[~ST.comparison.str.startswith("deletion curve")]
ST = pd.concat([ST, pd.DataFrame([dict(chapter=4, comparison=f"deletion curve, {r.method}{'' if 'vs' in r.method else ' order vs random'}, {r.model}",
                metric="area under drop curve", n=r.n, median_diff=r["diff"], p=r.p, family=r.family, test="Wilcoxon",
                source="ch5_deletion_curves_shared.csv") for _, r in RS.iterrows()])], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fp(r.p), fi(r.family), fp(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")
print(f"appD: {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())}")


# ---- cell 90 ------------------------------------------------------------
ST = pd.read_csv(TAB / "appD_statistics.csv")
print(f"appD: {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())}")
print("deletion curve rows:", int(ST.comparison.str.startswith("deletion curve").sum()),
      "| surviving:", int(ST[ST.comparison.str.startswith("deletion curve")].survives.eq("yes").sum()))
print("tex rows:", sum(1 for l in open(TAB / "appD_statistics.tex") if l.rstrip().endswith(r"\\") and "&" in l) - 2)


# ---- cell 91 ------------------------------------------------------------
ST = pd.read_csv(TAB / "appD_statistics.csv")
DEPTH = "depth, annotated vs unannotated lesion cells"
print("depth row present before:", (ST.comparison == DEPTH).any())
ST = ST[ST.comparison != DEPTH]
ST = pd.concat([ST, pd.DataFrame([dict(chapter=5, comparison=DEPTH, metric="normalised depth", n=31,
                median_diff=0.222, p=9.3e-10, family=1, test="Wilcoxon", source="notebook 16c")])], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fp(r.p), fi(r.family), fp(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")
ST2 = pd.read_csv(TAB / "appD_statistics.csv")
n_tex = sum(1 for l in open(TAB / "appD_statistics.tex") if l.rstrip().endswith(r"\\") and "&" in l) - 2
print(f"appD: {len(ST2)} tests, surviving {int((ST2.survives == 'yes').sum())} | tex rows {n_tex} | "
      f"depth present {int((ST2.comparison == DEPTH).sum())} | deletion curve rows {int(ST2.comparison.str.startswith('deletion curve').sum())}")


# ---- cell 92 ------------------------------------------------------------
from src.thesis.config import BLOCK, MEL_IDX, NV_IDX
from scipy.stats import spearmanr
MC0 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha0.pth", cache=False)
MC5 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha5.pth", cache=False)
def cos(a, b): a, b = a.ravel(), b.ravel(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def stages(model, x, L):
    with torch.enable_grad():
        base, logits = cams._forward(model, x)
        A = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A, retain_graph=True)[0]
        Gn = torch.autograd.grad(logits[0, NV_IDX], A)[0]
    gap = bool(getattr(base, "use_mean_pooling", False)); base.clear_xai_state()
    f = (lambda T: T[0, :, 1:, 1:]) if gap else (lambda T: T[0, :, 0, 1:])
    A, Gm, Gn = [f(T).detach().float().cpu().numpy() for T in (A, Gm, Gn)]
    Pm, Pn = A * Gm, A * Gn
    Hm, Hn = (Pm.sum(1), Pn.sum(1)) if gap else (Pm, Pn)
    fin = lambda H: cams.norm01(np.maximum(H.mean(0), 0))
    out = {}
    for tag, k in [("all", np.arange(196)), ("lesion", np.flatnonzero(L.ravel()))]:
        sl = (lambda T: T[..., k])
        out.update({f"{tag}_grad": cos(sl(Gm), sl(Gn)), f"{tag}_attnxgrad": cos(sl(Pm), sl(Pn)),
                    f"{tag}_head_median": float(np.median([cos(Hm[h][k], Hn[h][k]) for h in range(Hm.shape[0])])),
                    f"{tag}_head_mean": cos(Hm.mean(0)[k], Hn.mean(0)[k]), f"{tag}_final": cos(fin(Hm)[k], fin(Hn)[k])})
    return out

rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L = LESION[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for pool, nm, m in [("GAP", "unaligned", M0), ("GAP", "global aligned", M5), ("CLS", "unaligned", MC0), ("CLS", "global aligned", MC5)]:
        rows.append({"pool": pool, "model": nm, "image_id": iid, **stages(m, x, L)})
STG = pd.DataFrame(rows); STG.to_csv(TAB / "ch4_head_stages.csv", index=False)
S = ["grad", "attnxgrad", "head_median", "head_mean", "final"]
for tag in ["all", "lesion"]:
    print(f"1. {tag} cells, median over 31 images\n", STG.groupby(["pool", "model"])[[f"{tag}_{s}" for s in S]].median().round(3).to_string())


# ---- cell 93 ------------------------------------------------------------
import json
for p in sorted((REPO / "notebooks/mel_nv").glob("1[4-6]*.ipynb")) + sorted((REPO / "src/thesis").glob("*.py")):
    src = p.read_text() if p.suffix == ".py" else "\n".join("".join(c["source"]) for c in json.load(open(p))["cells"])
    for kw in ["darkness", "def dark", "luminance", "cvtColor", "reference_ladder"]:
        for m in re.finditer(re.escape(kw), src):
            print(f"--- {p.name} [{kw}]"); print(src[max(0, m.start() - 300): m.start() + 500]); break


# ---- cell 94 ------------------------------------------------------------
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv")
SWEEP = {0.0: CK5 / "checkpoint-best-gap-ha0.pth",
         **{float(l): CK_ALPHA / f"checkpoint-best-gap-ha{str(l).replace('.', 'p')}.pth" for l in [0.1, 1, 5, 17]}}
mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
g = LAMC[LAMC.image_id.isin(mel_ids)].pivot_table(index="image_id", columns="lambda", values="auc_in")
gain = (g[5.0] - g[0.0]).dropna(); iid = (gain - gain.median()).abs().idxmin()
print("4. image", iid, "gain", round(gain[iid], 3), "median gain", round(gain.median(), 3))

r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
L, U = LESION[iid], HUM[iid]
LAMS = [0.0, 0.1, 1.0, 5.0, 17.0]
fig, ax = plt.subplots(2, 5, figsize=(TEXTWIDTH, TEXTWIDTH * 0.48))
for j, lam in enumerate(LAMS):
    m = models.load_ckpt(SWEEP[lam], cache=False); mel, nv = cams.class_cams(m, x); del m
    for i, v in enumerate([mel, nv]):
        a = ax[i, j]; overlay(a, rgb, v); outline(a, L, C_LESION, lw=0.9); outline(a, U & L, "#FFFFFF", lw=0.8)
        a.set_xticks([]); a.set_yticks([])
    s09 = float((cams.norm01(mel).reshape(14, 14)[L] > 0.9).mean())
    ax[0, j].set_title(r"$\lambda_{\mathrm{global}}$ = " + f"{lam:g}", fontsize=8)
    ax[1, j].set_xlabel(f"$r_{{\\mathrm{{in}}}}$ {metrics.corr_in(mel, nv, L):.2f}\n$s_{{0.9}}$ {s09:.2f}", fontsize=7)
ax[0, 0].set_ylabel("MEL map", fontsize=8); ax[1, 0].set_ylabel("NV map", fontsize=8)
fig.tight_layout(); save(fig, "fig_ch4_sweep_examples"); plt.show()


# ---- cell 95 ------------------------------------------------------------
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
def nv_union(iid):
    acc = None
    for p in QC[(QC.Image_ID == iid) & (QC.label_group == "NV")].mask_path:
        m = data._full(p); acc = m if acc is None else (acc | m)
    return None if acc is None else data.to_grid(acc) & LESION[iid]
NVS_IDS = [i for i in sorted(HUM) if EV.loc[i].gt_label == "NV" and nv_union(i) is not None and nv_union(i).any()]
print("5. nevi with benign structures and a melanoma target:", NVS_IDS)
M61 = pd.read_csv(TAB / "ch6_metrics.csv"); gm = M61[M61.method == "Grad-CAM"].set_index("image_id").auc_in
mels = gm[[i for i in gm.index if EV.loc[i].gt_label == "MEL"]]
mels = mels.iloc[np.argsort((mels - mels.median()).abs().values)[:2]].index.tolist()
PICK = mels + NVS_IDS[:2]

fig, ax = plt.subplots(len(PICK), 4, figsize=(TEXTWIDTH, TEXTWIDTH / 4 * len(PICK) + 0.5), squeeze=False)
for i, iid in enumerate(PICK):
    r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    L, U, N = LESION[iid], HUM[iid], nv_union(iid)
    mel, nv = cams.class_cams(M5, x); t, o = (mel, nv) if r.gt_label == "MEL" else (nv, mel)
    maps = [t, cams.norm01(np.maximum(t - o, 0)), cams.finer_cam(M5, x, r.gt_label, 0.6)]
    ax[i, 0].imshow(rgb)
    for j, v in enumerate(maps): overlay(ax[i, j + 1], rgb, v)
    for a in ax[i]:
        outline(a, L, C_LESION, lw=0.8); outline(a, U & L, "#FFFFFF", lw=0.8)
        if N is not None and N.any(): outline(a, N, "#FFFFFF", lw=0.8, ls="dashed")
        a.set_xticks([]); a.set_yticks([])
    ax[i, 0].set_ylabel(r.gt_label, fontsize=8)
for j, t_ in enumerate(["image", "Grad-CAM", "Diff-CAM", r"FinerCAM ($\gamma=0.6$)"]): ax[0, j].set_title(t_, fontsize=8)
fig.subplots_adjust(left=0.04, right=0.995, bottom=0.01, top=0.86, wspace=0.04, hspace=0.04)
p0, p1 = ax[0, 1].get_position(), ax[-1, 3].get_position(); x0, x1 = p0.x0 - 0.006, p1.x1 + 0.006
fig.add_artist(mpatches.Rectangle((x0, p1.y0 - 0.008), x1 - x0, 0.995 - p1.y0 + 0.008, transform=fig.transFigure,
                                  facecolor=to_rgba(C_GLOB, 0.15), edgecolor="none", zorder=-1))
fig.text((x0 + x1) / 2, 0.965, "Globally aligned model", ha="center", fontsize=9, fontweight="bold", color=C_GLOB)
save(fig, "fig_ch6_methods_examples"); plt.show()


# ---- cell 96 ------------------------------------------------------------
LANDSCAPE = True
F7 = pd.read_csv(TAB / "ch7_grounding.csv"); e = F7[F7.setting == "HA2 b1"].set_index("image_id").mel_in
PICK7 = []
for cls in ["MEL", "NV"]:
    s = e[[i for i in e.index if EV.loc[i].gt_label == cls]]; PICK7 += s.iloc[np.argsort((s - s.median()).abs().values)[:2]].index.tolist()
GR = [("global only", (5.0, 0.0), C_GLOB), (r"expert only, $\lambda_{\mathrm{expert}}$ 1", (0.0, 1.0), C_EXP),
      (r"joint, $\lambda_{\mathrm{expert}}$ 5", (5.0, 5.0), C_JOINT)]
W = 9.0 if LANDSCAPE else TEXTWIDTH
fig, ax = plt.subplots(len(PICK7), 6, figsize=(W, W / 6 * len(PICK7) + 0.6), squeeze=False)
for i, iid in enumerate(PICK7):
    r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    L, U, N = LESION[iid], HUM[iid], nv_union(iid)
    for g_, (_, (a_, b_), _) in enumerate(GR):
        m = models.f2(a_, b_, FOLD[iid]); mel, nv = cams.class_cams(m, x); del m
        for k, v in enumerate([mel, nv]):
            a = ax[i, 2 * g_ + k]; overlay(a, rgb, v); outline(a, L, C_LESION, lw=0.8); outline(a, U & L, "#FFFFFF", lw=0.8)
            if N is not None and N.any(): outline(a, N, "#FFFFFF", lw=0.8, ls="dashed")
            a.set_xticks([]); a.set_yticks([])
    ax[i, 0].set_ylabel(r.gt_label, fontsize=9)
for j in range(6): ax[0, j].set_title("MEL map" if j % 2 == 0 else "NV map", fontsize=8)
fig.subplots_adjust(left=0.03, right=0.995, bottom=0.01, top=0.86, wspace=0.03, hspace=0.03)
for g_, (lab, _, col) in enumerate(GR):
    p0, p1 = ax[0, 2 * g_].get_position(), ax[-1, 2 * g_ + 1].get_position(); x0, x1 = p0.x0 - 0.004, p1.x1 + 0.004
    fig.add_artist(mpatches.Rectangle((x0, p1.y0 - 0.006), x1 - x0, 0.995 - p1.y0 + 0.006, transform=fig.transFigure,
                                      facecolor=to_rgba(col, 0.15), edgecolor="none", zorder=-1))
    fig.text((x0 + x1) / 2, 0.955, lab, ha="center", fontsize=10, fontweight="bold", color=col)
save(fig, "fig_ch7_channels"); plt.show()
print("7. images:", {i: EV.loc[i].gt_label for i in PICK7})


# ---- cell 97 ------------------------------------------------------------
import cv2
def mask224(rel):
    m = np.array(Image.open((HAM / str(rel)).resolve()).convert("L")) > 127
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
grid = lambda m: m.reshape(14, 16, 14, 16).mean((1, 3)) >= 0.5
rows = []
for _, r in TEST_DF.sample(100, random_state=SEED).iterrows():
    img, m = data.rgb224(r.image_rel_path), mask224(r.mask_rel_path); Lg = grid(m)
    if not (0 < Lg.sum() < Lg.size): continue
    ip = cv2.inpaint(img, cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) * 255, 7, cv2.INPAINT_TELEA)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((49, 49), np.uint8)).astype(bool) & ~cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    Y = cv2.cvtColor(ip, cv2.COLOR_RGB2GRAY).astype(float); Y0 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    v = cams.class_cams(M5, data.to_tensor(ip))[0 if r.gt_label == "MEL" else 1]; v = cams.norm01(v)
    rows.append({"image_id": r.image_id, "share": float(v[Lg].sum() / (v.sum() + 1e-8)), "area": Lg.mean(),
                 "lum_region": Y[m].mean(), "lum_ring": Y[ring].mean() if ring.any() else np.nan, "lum_lesion_orig": Y0[m].mean(),
                 "sd_region": Y[m].std(), "sd_ring": Y[ring].std() if ring.any() else np.nan})
IP = pd.DataFrame(rows).dropna(); IP["contrast"] = IP.lum_ring - IP.lum_region
IP.to_csv(TAB / "app_inpaint_contrast.csv", index=False)
print(f"8. n={len(IP)}  heat share {IP.share.median():.3f} (area {IP.area.median():.3f})")
print(f"8. luminance: region {IP.lum_region.median():.1f}, ring {IP.lum_ring.median():.1f}, original lesion {IP.lum_lesion_orig.median():.1f}")
print(f"8. region darker than ring on {int((IP.contrast > 0).sum())} of {len(IP)}, {metrics.paired(IP.lum_ring, IP.lum_region)}")
print(f"8. texture sd: region {IP.sd_region.median():.1f}, ring {IP.sd_ring.median():.1f}")
print(f"8. heat share vs contrast: Spearman {spearmanr(IP.share, IP.contrast).statistic:.3f} (p {spearmanr(IP.share, IP.contrast).pvalue:.2g})")


# ---- cell 98 ------------------------------------------------------------
LM = LAMC.merge(EV[["gt_label"]], left_on="image_id", right_index=True)
print("4. AUC_in by class:\n", LM[LM["lambda"].isin([0.0, 5.0])].groupby(["gt_label", "lambda"]).auc_in.median().unstack().round(3).to_string())
g = LAMC[LAMC.image_id.isin(mel_ids)].pivot_table(index="image_id", columns="lambda", values="corr_in")
gain = (g[5.0] - g[0.0]).dropna(); iid = (gain - gain.median()).abs().idxmin()
print("4. image", iid, "r_in gain", round(gain[iid], 3), "median", round(gain.median(), 3))


# ---- cell 99 ------------------------------------------------------------
r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
L, U = LESION[iid], HUM[iid]
LAMS = [0.0, 0.1, 1.0, 5.0, 17.0]
fig, ax = plt.subplots(2, 5, figsize=(TEXTWIDTH, TEXTWIDTH * 0.48))
for j, lam in enumerate(LAMS):
    m = models.load_ckpt(SWEEP[lam], cache=False); mel, nv = cams.class_cams(m, x); del m
    for i, v in enumerate([mel, nv]):
        a = ax[i, j]; overlay(a, rgb, v); outline(a, L, C_LESION, lw=0.9); outline(a, U & L, "#FFFFFF", lw=0.8)
        a.set_xticks([]); a.set_yticks([])
    s09 = float((cams.norm01(mel).reshape(14, 14)[L] > 0.9).mean())
    ax[0, j].set_title(r"$\lambda_{\mathrm{global}}$ = " + f"{lam:g}", fontsize=8)
    ax[1, j].set_xlabel(f"$r_{{\\mathrm{{in}}}}$ {metrics.corr_in(mel, nv, L):.2f}\n$s_{{0.9}}$ {s09:.2f}", fontsize=7)
ax[0, 0].set_ylabel("MEL map", fontsize=8); ax[1, 0].set_ylabel("NV map", fontsize=8)
fig.tight_layout(); save(fig, "fig_ch4_sweep_examples"); plt.show()


# ---- cell 100 ------------------------------------------------------------
q = LM.pivot_table(index=["image_id", "gt_label"], columns="lambda", values="auc_in").reset_index()
for c in ["MEL", "NV"]:
    s = q[q.gt_label == c]
    print(f"4. {c}: n {len(s)}  median 0 {s[0.0].median():.3f}  median 5 {s[5.0].median():.3f}  paired {metrics.paired(s[5.0], s[0.0])}")
print(LM.groupby(["gt_label", "lambda"]).auc_in.median().unstack(0).round(3).to_string())


# ---- cell 101 ------------------------------------------------------------
ST = pd.read_csv(TAB / "appD_statistics.csv")
TAG = "AUC_in, global aligned vs unaligned, by class"
ST = ST[~ST.comparison.str.startswith(TAG)]
new = []
for c in ["MEL", "NV"]:
    s = q[q.gt_label == c]; r = metrics.paired(s[5.0], s[0.0])
    new.append(dict(chapter=4, comparison=f"{TAG}, {c}", metric="auc_in", n=r["n"], median_diff=r["diff"],
                    p=r["p"], family=2, test="Wilcoxon", source="ch5_lambda_collapse.csv"))
print("4. unpaired nevus:", sorted(set(q[(q.gt_label == "NV") & q[[0.0, 5.0]].isna().any(axis=1)].image_id)))
ST = pd.concat([ST, pd.DataFrame(new)], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fp(r.p), fi(r.family), fp(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")
print(f"appD: {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())} | depth present {int((ST.comparison.str.startswith('depth')).sum())}")


# ---- cell 102 ------------------------------------------------------------
import json
# 1. which map each reference row scores, and the class split
nb = json.load(open(REPO / "notebooks/mel_nv/16f_figures_tables.ipynb"))
s = next("".join(c["source"]) for c in nb["cells"] if "def lad" in "".join(c["source"]))
print("1. lad calls:\n  " + "\n  ".join(l.strip() for l in s.splitlines() if "lad(" in l and "def lad" not in l))
LD = pd.read_csv(TAB / "ch5_reference_ladder.csv").merge(EV[["gt_label"]], left_on="image_id", right_index=True)
REFS = ["darkness", "unaligned", "lesion aligned", "concept only"]
for met in ["auc_in", "energy_in"]:
    print(f"\n1. {met}, median by class\n", LD[LD.ref.isin(REFS)].groupby(["ref", "gt_label"])[met].median().unstack().reindex(REFS).round(3).to_string())
    q = LD.pivot_table(index=["image_id", "gt_label"], columns="ref", values=met).reset_index()
    for c in ["MEL", "NV"]:
        for mo in REFS[1:]:
            print(f"   {met} {c} {mo:15} vs darkness {metrics.paired(q[q.gt_label == c][mo], q[q.gt_label == c].darkness)}")

# 2. methods by class, FinerCAM vs Grad-CAM per class, gamma sweep
M6 = pd.read_csv(TAB / "ch6_metrics.csv").merge(EV[["gt_label"]], left_on="image_id", right_index=True)
E = [c for c in ["auc_in", "energy_in", "energy_excess"] if c in M6.columns]
FN = next(m for m in M6.method.unique() if m.startswith("FinerCAM") and "0.6" in m)
print("\n2. methods:", M6.method.unique().tolist(), "| metrics:", E)
print(M6.groupby(["gt_label", "method"])[E].median().round(3).to_string())
print("2. n per class:", M6[M6.method == "Grad-CAM"].gt_label.value_counts().to_dict())
for c in ["MEL", "NV"]:
    for met in E:
        q = M6[M6.gt_label == c].pivot_table(index="image_id", columns="method", values=met)
        print(f"   {c} {met:12} {FN} vs Grad-CAM {metrics.paired(q[FN], q['Grad-CAM'])}")
for f in sorted(TAB.glob("*gamma*.csv")):
    G = pd.read_csv(f); print(f"\n2. {f.name}: {list(G.columns)}")
    gc = next((c for c in G.columns if "gamma" in c.lower()), None)
    if gc and "image_id" in G.columns and "auc_in" in G.columns:
        G = G.merge(EV[["gt_label"]], left_on="image_id", right_index=True)
        print(G.groupby([gc, "gt_label"]).auc_in.median().unstack().round(3).to_string())

# 3. values behind the sweep example
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv")
print("\n3. ISIC_0028897\n", LAMC[LAMC.image_id == "ISIC_0028897"].set_index("lambda")[["corr_in", "sat_09", "auc_in"]].round(3).to_string())


# ---- cell 103 ------------------------------------------------------------
G = pd.read_csv(TAB / "ch6_finercam_gamma_auc.csv")
GAM = sorted(G.gamma.unique())
print("2. gamma file:", G.model.unique().tolist(), "| gammas", GAM, "| images", G.image_id.nunique())
rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]; x = data.to_tensor(data.native(r.image_rel_path))
    for g in GAM:
        v = np.asarray(cams.finer_cam(M5, x, r.gt_label, float(g)), float).reshape(14, 14)
        rows.append({"gamma": g, "image_id": iid, "gt": r.gt_label, "auc_in": metrics.auc_in(v, L, U),
                     "energy_in": metrics.energy_excess(v, L, U), "dead": int(v[L].max() < 1e-6)})
GS = pd.DataFrame(rows); GS.to_csv(TAB / "ch4_finercam_gamma_class.csv", index=False)
print(GS.groupby(["gamma", "gt"]).agg(auc_in=("auc_in", "median"), energy_in=("energy_in", "median"),
                                      dead=("dead", "sum")).unstack("gt").round(3).to_string())

ST = pd.read_csv(TAB / "appD_statistics.csv")
T1, T2 = "reference by class", "FinerCAM vs Grad-CAM by class"
ST = ST[~ST.comparison.str.match(f"{T1}|{T2}")]
RN = {"unaligned": "unaligned", "lesion aligned": "globally aligned", "concept only": "expert aligned"}
new = []
for met in ["auc_in", "energy_in"]:
    q = LD.pivot_table(index=["image_id", "gt_label"], columns="ref", values=met).reset_index()
    for c in ["MEL", "NV"]:
        s = q[q.gt_label == c]
        for mo, lab in RN.items():
            r = metrics.paired(s[mo], s.darkness)
            new.append(dict(chapter=4, comparison=f"{T1}, MEL map, {lab} vs darkness, {c}", metric=met, n=r["n"],
                            median_diff=r["diff"], p=r["p"], family=6, test="Wilcoxon", source="ch5_reference_ladder.csv"))
    q = M6.pivot_table(index=["image_id", "gt_label"], columns="method", values=met).reset_index()
    for c in ["MEL", "NV"]:
        s = q[q.gt_label == c]; r = metrics.paired(s[FN], s["Grad-CAM"])
        new.append(dict(chapter=4, comparison=f"{T2}, globally aligned, {c}", metric=met, n=r["n"],
                        median_diff=r["diff"], p=r["p"], family=2, test="Wilcoxon", source="ch6_metrics.csv"))
ST = pd.concat([ST, pd.DataFrame(new)], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fp(r.p), fi(r.family), fp(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")
print(f"appD: {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())} | depth present {int(ST.comparison.str.startswith('depth').sum())}")


# ---- cell 104 ------------------------------------------------------------
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv")
LAMS = [0.0, 0.1, 1.0, 5.0, 17.0]
SWEEP = {0.0: CK5 / "checkpoint-best-gap-ha0.pth",
         **{l: CK_ALPHA / f"checkpoint-best-gap-ha{str(l if l % 1 else int(l)).replace('.', 'p')}.pth" for l in LAMS[1:]}}
mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
R = (LAMC[LAMC.image_id.isin(mel_ids) & LAMC["lambda"].isin(LAMS)]
     .pivot_table(index="image_id", columns="lambda", values="corr_in").dropna())
med = R.median(); iid = (R - med).abs().sum(axis=1).idxmin()

r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
L, U = LESION[iid], HUM[iid]; VAL = {}
fig, ax = plt.subplots(2, 5, figsize=(TEXTWIDTH, TEXTWIDTH * 0.48))
for j, lam in enumerate(LAMS):
    m = models.load_ckpt(SWEEP[lam], cache=False); mel, nv = cams.class_cams(m, x); del m
    for i, v in enumerate([mel, nv]):
        a = ax[i, j]; overlay(a, rgb, v); outline(a, L, C_LESION, lw=0.9); outline(a, U & L, "#FFFFFF", lw=0.8)
        a.set_xticks([]); a.set_yticks([])
    VAL[lam] = (metrics.corr_in(mel, nv, L), float((cams.norm01(mel).reshape(14, 14)[L] > 0.9).mean()))
    ax[0, j].set_title(r"$\lambda_{\mathrm{global}}$ = " + f"{lam:g}", fontsize=8)
    ax[1, j].set_xlabel(f"$r_{{\\mathrm{{in}}}}$ {VAL[lam][0]:.2f}\n$s_{{0.9}}$ {VAL[lam][1]:.2f}", fontsize=7)
ax[0, 0].set_ylabel("MEL map", fontsize=8); ax[1, 0].set_ylabel("NV map", fontsize=8)
fig.tight_layout(); save(fig, "fig_ch4_sweep_examples"); plt.show()

f = lambda v: f"{v:.2f}"
print(f"""=== PASTE INTO ITEM 5 OF THE PROMPT ===
Caption fig:sweep-examples: "Melanoma map (top) and nevus map (bottom) of one annotated melanoma, {iid}, across the sweep (lambda_global = 0, 0.1, 1, 5, 17). The image is the annotated melanoma whose r_in over the five weights lies closest to the median of the {len(R)} annotated melanomas. Below each column: r_in, and s_0.9 of the melanoma map. Green: lesion outline. White: melanoma structures."
Text after the figure: "Figure~\\ref{{fig:sweep-examples}} shows the collapse on one melanoma. The correlation of its two maps inside the lesion is {f(VAL[0.0][0])} without alignment and {f(VAL[0.1][0])}, {f(VAL[1.0][0])}, {f(VAL[5.0][0])} and {f(VAL[17.0][0])} at lambda_global = 0.1, 1, 5 and 17. The median over the {len(R)} annotated melanomas is {f(med[0.0])}, {f(med[0.1])}, {f(med[1.0])}, {f(med[5.0])} and {f(med[17.0])} at the same weights. In the melanoma map, the share of lesion cells above 0.9 rises from {f(VAL[0.0][1])} without alignment to {f(VAL[5.0][1])} at lambda_global = 5 and {f(VAL[17.0][1])} at lambda_global = 17." [ch5_lambda_collapse.csv]
=== END ===""")


# ---- cell 105 ------------------------------------------------------------
import re
COLOUR_FIGS = {"fig_ch5_localisation_examples", "fig_ch6_methods_examples", "fig_ch7_channels"}
REN = [(r"^MEL\b", "Melanoma"), (r"^NV\b", "Nevus"), (r"^melanoma\b", "Melanoma"), (r"^nevus\b", "Nevus")]
def class_colours(fig):
    def fix(t):
        s = t.get_text()
        for pat, rep in REN: s = re.sub(pat, rep, s)
        if s.startswith("Melanoma"): t.set_text(s); t.set_color(plotting.C_MEL)
        elif s.startswith("Nevus"): t.set_text(s); t.set_color(plotting.C_NV)
    for a in fig.axes:
        for t in [a.title, a.xaxis.label, a.yaxis.label, *a.texts]: fix(t)
    for t in fig.texts: fix(t)
_save0 = globals().get("_save0", save)
def save(fig, name, *a, **k):
    if name in COLOUR_FIGS: class_colours(fig)
    return _save0(fig, name, *a, **k)
plotting.save = save

# app_case_manifest.tex: labels column p{8cm}, every other column l (as narrow as its content)
p = TAB / "app_case_manifest.tex"; s = p.read_text()
m = re.search(r"\\begin\{(tabular|longtable)\}(\[[^\]]*\])?\{([^}]*)\}", s)
hdr = next(l for l in s.splitlines() if "&" in l and l.rstrip().endswith("\\\\"))
cols = [c.strip().lower() for c in hdr.rstrip("\\ ").split("&")]
spec = "".join("p{8cm}" if "label" in c else "l" for c in cols)
s = s[:m.start(3)] + spec + s[m.end(3):]
p.write_text(s)
print("case manifest columns:", cols, "| new spec:", spec)


# ---- cell 106 ------------------------------------------------------------
import re, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from PIL import Image
from matplotlib.colors import to_rgba
from src.thesis import plotting
from src.thesis.plotting import overlay, outline, C_LESION, TEXTWIDTH
save = globals().get("save", plotting.save)
need = ["EV", "HUM", "LESION", "TEST_DF", "HAM", "M0", "M5", "TAB", "SEED", "models", "cams", "data", "metrics"]
missing = [n for n in need if n not in globals()]
assert not missing, f"Run the 16g header and cell 0 first. Missing: {missing}"
FOLD = data.fold_of_heldout(); held = [i for i in sorted(HUM) if i in FOLD]
print("setup ok |", len(held), "held-out images")


# ---- cell 107: skipped (raised an error in the notebook)


# ---- cell 108 ------------------------------------------------------------
METH = {"Grad-CAM": lambda m, x: cams.target_ref(m, x, "MEL")[0],
        "Diff-CAM": lambda m, x: cams.diff_cam(m, x, "MEL"),
        "FinerCAM 0.6": lambda m, x: cams.finer_cam(m, x, "MEL", 0.6)}
TT = ["Grad-CAM", "Diff-CAM", r"FinerCAM ($\gamma=0.6$)"]
GRP = [("Globally aligned", (5.0, 0.0), plotting.C_GLOB), ("Expert aligned", (0.0, 1.0), plotting.C_EXP)]
held_mel = [i for i in held if EV.loc[i].gt_label == "MEL"]; SHOW = held_mel[:3]

fig, ax = plt.subplots(len(SHOW), 7, figsize=(TEXTWIDTH, TEXTWIDTH / 7 * len(SHOW) + 0.6), squeeze=False)
for i, iid in enumerate(SHOW):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]
    x = data.to_tensor(data.native(r.image_rel_path)); rgb = data.rgb224(r.image_rel_path)
    ax[i, 0].imshow(rgb); outline(ax[i, 0], L, C_LESION, lw=0.8); outline(ax[i, 0], U & L, "#FFFFFF", lw=0.8)
    for g, (_, (a_, b_), _) in enumerate(GRP):
        m = models.f2(a_, b_, FOLD[iid])
        for j, mk in enumerate(METH):
            a = ax[i, 1 + 3 * g + j]; overlay(a, rgb, METH[mk](m, x))
            outline(a, L, C_LESION, lw=0.6); outline(a, U & L, "#FFFFFF", lw=0.6)
        del m
    ax[i, 0].set_ylabel(iid, fontsize=7)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_title("Melanoma", fontsize=8, color=plotting.C_MEL)
for g in range(2):
    for j, t in enumerate(TT): ax[0, 1 + 3 * g + j].set_title(t, fontsize=8)
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.01, top=0.80, wspace=0.04, hspace=0.04)
for g, (lab, _, col) in enumerate(GRP):
    p0, p1 = ax[0, 1 + 3 * g].get_position(), ax[-1, 3 + 3 * g].get_position(); x0, x1 = p0.x0 - 0.004, p1.x1 + 0.004
    fig.add_artist(mpatches.Rectangle((x0, p1.y0 - 0.006), x1 - x0, 0.995 - p1.y0 + 0.006, transform=fig.transFigure,
                                      facecolor=to_rgba(col, 0.15), edgecolor="none", zorder=-1))
    fig.text((x0 + x1) / 2, 0.93, lab, ha="center", fontsize=9, fontweight="bold", color=col)
save(fig, "fig_ch7_contrastive_expert"); plt.show()
print("contrastive figure saved | images:", SHOW)


# ---- cell 109 ------------------------------------------------------------
DL = pd.read_csv(TAB / "ch5_deletion_v2.csv")
NAME = {"unaligned": "unaligned", "lesion aligned": "global aligned", "global aligned": "global aligned", "aligned": "global aligned",
        "concept only": "expert aligned", "expert only": "expert aligned", "expert aligned": "expert aligned"}
raw = DL.model.unique().tolist(); DL["model"] = DL.model.map(NAME)
print("models in file:", raw, "| unmapped:", [m for m in raw if m not in NAME] or "none")
REG = ["top 10%", "random 10%", "clinician (n)", "unannotated (n)"]
CC = ["#252525", plotting.C_CHANCE, "#737373", "#d9d9d9"]; HH = ["", "", "///", "///"]
ORDM = ["unaligned", "global aligned", "expert aligned"]
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.4)); pos, ticks = 0, []
for mo in ORDM:
    for reg, c, h in zip(REG, CC, HH):
        b = ax.boxplot(DL[DL.model == mo][reg].dropna(), positions=[pos], widths=0.7, patch_artist=True, showfliers=False)
        b["boxes"][0].set(facecolor=c, hatch=h, edgecolor="0.3"); pos += 1
    ticks.append(pos - 2.5); pos += 1
ax.axhline(0, color="k", lw=0.6)
ax.set(xticks=ticks, xticklabels=ORDM, ylabel="normalised drop")
ax.legend([mpatches.Patch(facecolor=c, hatch=h, edgecolor="0.3") for c, h in zip(CC, HH)], REG, frameon=False, fontsize=6, ncol=4)
fig.tight_layout(); save(fig, "fig_ch5_deletion"); plt.show()


# ---- cell 110 ------------------------------------------------------------
ex = sorted(set(RB[RB["cond"] == "shifted"].image_id) - set(RB[RB["cond"] == "original"].image_id))
IPC = pd.read_csv(TAB / "app_inpaint_contrast.csv")
for iid in ex:
    r = TEST_DF[TEST_DF.image_id == iid].iloc[0]; G = grid(mask224(r.mask_rel_path))
    print(iid, r.gt_label, "| lesion cells on the grid:", int(G.sum()), "of 196",
          "| also missing from app_inpaint_contrast:", iid not in set(IPC.image_id.astype(str)))


# ---- cell 111 ------------------------------------------------------------
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from PIL import Image
from matplotlib.colors import to_rgba
from src.thesis import plotting
from src.thesis.plotting import overlay, outline, C_LESION, TEXTWIDTH
if "_save0" in globals(): plotting.save = _save0          # switch off the old label wrapper
save = plotting.save
need = ["EV", "HUM", "LESION", "QC", "TEST_DF", "HAM", "M0", "M5", "TAB", "SEED", "CK5", "CK_ALPHA", "models", "cams", "data", "metrics"]
missing = [n for n in need if n not in globals()]
assert not missing, f"Run the 16g header and cell 0 first. Missing: {missing}"
C_MEL, C_NV, C_UNAL, C_GLOB, C_EXP, C_JOINT = (plotting.C_MEL, plotting.C_NV, plotting.C_UNAL,
                                               plotting.C_GLOB, plotting.C_EXP, plotting.C_JOINT)
FOLD = data.fold_of_heldout(); held = [i for i in sorted(HUM) if i in FOLD]
GT = TEST_DF.assign(image_id=TEST_DF.image_id.astype(str)).set_index("image_id").gt_label.to_dict()
GT.update(EV.gt_label.to_dict())
IDS = []                                                   # (figure, panel, image id) for the appendix table

def numbered(ids):
    n = {"MEL": 0, "NV": 0}; out = []
    for i in ids:
        g = GT[i]; n[g] += 1; out.append((i, f"{'Melanoma' if g == 'MEL' else 'Nevus'} {n[g]}"))
    return out
def row_label(a, text, fs=8): a.set_ylabel(text, fontsize=fs, color="black")
def map_head(a, which, fs=8, title=True):
    t, c = ("Melanoma map", C_MEL) if which == "MEL" else ("Nevus map", C_NV)
    (a.set_title if title else a.set_ylabel)(t, fontsize=fs, color=c)
def band(fig, ax, c0, c1, label, col, text_y):
    p0, p1 = ax[0, c0].get_position(), ax[-1, c1].get_position(); x0, x1 = p0.x0 - 0.004, p1.x1 + 0.004
    fig.add_artist(mpatches.Rectangle((x0, p1.y0 - 0.005), x1 - x0, 0.997 - p1.y0 + 0.005, transform=fig.transFigure,
                                      facecolor=to_rgba(col, 0.15), edgecolor="none", zorder=-1))
    fig.text((x0 + x1) / 2, text_y, label, ha="center", fontsize=9, fontweight="bold", color=col)
def nv_union(iid):
    acc = None
    for p in QC[(QC.Image_ID == iid) & (QC.label_group == "NV")].mask_path:
        m = data._full(p); acc = m if acc is None else (acc | m)
    return None if acc is None else data.to_grid(acc) & LESION[iid]
def clean(ax):
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
print("setup ok |", len(held), "held-out images")


# ---- cell 112 ------------------------------------------------------------
pick = (TEST_DF[TEST_DF.gt_label == "MEL"].sort_values("image_id").head(3).image_id.astype(str).tolist()
        + TEST_DF[TEST_DF.gt_label == "NV"].sort_values("image_id").head(3).image_id.astype(str).tolist())
T = TEST_DF.assign(image_id=TEST_DF.image_id.astype(str)).set_index("image_id")
fig, ax = plt.subplots(6, 5, figsize=(TEXTWIDTH, 7.9))
for i, (iid, lab) in enumerate(numbered(pick)):
    r = T.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    L = data.to_grid(np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127)
    a0, b0 = cams.class_cams(M0, x); a5, b5 = cams.class_cams(M5, x)
    ax[i, 0].imshow(rgb); outline(ax[i, 0], L, C_LESION)
    for j, v in enumerate([a0, b0, a5, b5]):
        overlay(ax[i, j + 1], rgb, v); outline(ax[i, j + 1], L, C_LESION, lw=0.6)
    row_label(ax[i, 0], lab); IDS.append(("localisation examples", lab, iid))
clean(ax); ax[0, 0].set_title("Image", fontsize=8)
for j, w in enumerate(["MEL", "NV", "MEL", "NV"]): map_head(ax[0, j + 1], w)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.005, top=0.935, wspace=0.03, hspace=0.03)
band(fig, ax, 1, 2, "Unaligned", C_UNAL, 0.972); band(fig, ax, 3, 4, "Globally aligned", C_GLOB, 0.972)
save(fig, "fig_ch5_localisation_examples"); plt.show()


# ---- cell 113 ------------------------------------------------------------
def mask224(rel):
    m = np.array(Image.open((HAM / str(rel)).resolve()).convert("L")) > 127
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
def grid(m): return m.reshape(14, 16, 14, 16).mean((1, 3)) >= 0.5
def shift(img, m):
    ys, xs = np.where(m)
    dx = min(223 - xs.max(), 60) if 223 - xs.max() > xs.min() else -min(xs.min(), 60)
    dy = min(223 - ys.max(), 60) if 223 - ys.max() > ys.min() else -min(ys.min(), 60)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return (cv2.warpAffine(img, M, (224, 224), borderMode=cv2.BORDER_REFLECT),
            cv2.warpAffine(m.astype(np.uint8), M, (224, 224), flags=cv2.INTER_NEAREST).astype(bool))
def inpaint(img, m): return cv2.inpaint(img, cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) * 255, 7, cv2.INPAINT_TELEA)
def uniform(img, m): return np.broadcast_to(np.median(img[~m], axis=0).astype(np.uint8), img.shape).copy()
CENTRE = np.zeros((14, 14), bool); CENTRE[4:11, 4:11] = True
def heat_share(v, G): v = cams.norm01(v); return float(v[G].sum() / (v.sum() + 1e-8))
def conditions(r):
    img, m = data.rgb224(r.image_rel_path), mask224(r.mask_rel_path); s_img, s_m = shift(img, m)
    return {"original": (img, grid(m)), "shifted": (s_img, grid(s_m)),
            "lesion inpainted": (inpaint(img, m), grid(m)), "uniform": (uniform(img, m), CENTRE)}
def gtmap(m, x, gt): mel, nv = cams.class_cams(m, x); return mel if gt == "MEL" else nv

CONDS = ["original", "shifted", "lesion inpainted", "uniform"]
RB = pd.read_csv(TAB / "app_robustness.csv"); RB["image_id"] = RB.image_id.astype(str)
def pick_rb(gt):
    P = (RB[(RB["gt"] == gt) & (RB["model"] == "globally aligned")]
         .pivot_table(index="image_id", columns="cond", values="share").reindex(columns=CONDS).dropna())
    return (P - P.median()).abs().sum(axis=1).idxmin()

for gt in ["MEL", "NV"]:
    iid = pick_rb(gt); r = T.loc[iid]; C = conditions(r)
    fig, ax = plt.subplots(3, 4, figsize=(TEXTWIDTH, TEXTWIDTH * 0.80))
    for j, cond in enumerate(CONDS):
        img, G = C[cond]; x = data.to_tensor(img)
        ax[0, j].imshow(img); outline(ax[0, j], G, C_LESION, lw=1.0)
        ax[0, j].set_title(cond.capitalize(), fontsize=9); ax[0, j].set_xlabel(f"region area {G.mean():.2f}", fontsize=8)
        for i, m in enumerate([M0, M5], start=1):
            v = gtmap(m, x, gt); overlay(ax[i, j], img, v); outline(ax[i, j], G, C_LESION, lw=0.8)
            ax[i, j].set_xlabel(f"heat share {heat_share(v, G):.2f}", fontsize=8)
    clean(ax)
    for i, t in enumerate(["Image", "Unaligned", "Globally aligned"]): row_label(ax[i, 0], t, fs=9)
    fig.tight_layout(); save(fig, f"fig_ch4_robustness_{gt.lower()}"); plt.show()
    IDS.append((f"lesion tracking, {'melanoma' if gt == 'MEL' else 'nevus'}", "all columns", iid))


# ---- cell 114 ------------------------------------------------------------
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv")
LAMS = [0.0, 0.1, 1.0, 5.0, 17.0]
ck = lambda l: CK5 / "checkpoint-best-gap-ha0.pth" if l == 0 else \
     CK_ALPHA / f"checkpoint-best-gap-ha{str(l if l % 1 else int(l)).replace('.', 'p')}.pth"
mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
R = (LAMC[LAMC.image_id.isin(mel_ids) & LAMC["lambda"].isin(LAMS)]
     .pivot_table(index="image_id", columns="lambda", values="corr_in").dropna())
iid = (R - R.median()).abs().sum(axis=1).idxmin()
r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
L, U = LESION[iid], HUM[iid]
fig, ax = plt.subplots(2, 5, figsize=(TEXTWIDTH, TEXTWIDTH * 0.48))
for j, lam in enumerate(LAMS):
    m = models.load_ckpt(ck(lam), cache=False); mel, nv = cams.class_cams(m, x); del m
    for i, v in enumerate([mel, nv]):
        overlay(ax[i, j], rgb, v); outline(ax[i, j], L, C_LESION, lw=0.9); outline(ax[i, j], U & L, "#FFFFFF", lw=0.8)
    s09 = float((cams.norm01(mel).reshape(14, 14)[L] > 0.9).mean())
    ax[0, j].set_title(r"$\lambda_{\mathrm{global}}$ = " + f"{lam:g}", fontsize=8)
    ax[1, j].set_xlabel(f"$r_{{\\mathrm{{in}}}}$ {metrics.corr_in(mel, nv, L):.2f}\n$s_{{0.9}}$ {s09:.2f}", fontsize=7)
clean(ax); map_head(ax[0, 0], "MEL", title=False); map_head(ax[1, 0], "NV", title=False)
fig.tight_layout(); save(fig, "fig_ch4_sweep_examples"); plt.show()
IDS.append(("sweep example", "all columns", iid))


# ---- cell 115 ------------------------------------------------------------
NVS_IDS = [i for i in sorted(HUM) if EV.loc[i].gt_label == "NV" and nv_union(i) is not None and nv_union(i).any()]
M61 = pd.read_csv(TAB / "ch6_metrics.csv"); gm = M61[M61.method == "Grad-CAM"].set_index("image_id").auc_in
mels = gm[[i for i in gm.index if EV.loc[i].gt_label == "MEL"]]
PICK = mels.iloc[np.argsort((mels - mels.median()).abs().values)[:2]].index.tolist() + NVS_IDS[:2]
fig, ax = plt.subplots(4, 4, figsize=(TEXTWIDTH, TEXTWIDTH + 0.5))
for i, (iid, lab) in enumerate(numbered(PICK)):
    r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    L, U, N = LESION[iid], HUM[iid], nv_union(iid)
    mel, nv = cams.class_cams(M5, x); t, o = (mel, nv) if r.gt_label == "MEL" else (nv, mel)
    maps = [t, cams.norm01(np.maximum(t - o, 0)), cams.finer_cam(M5, x, r.gt_label, 0.6)]
    ax[i, 0].imshow(rgb)
    for j, v in enumerate(maps): overlay(ax[i, j + 1], rgb, v)
    for a in ax[i]:
        outline(a, L, C_LESION, lw=0.8); outline(a, U & L, "#FFFFFF", lw=0.8)
        if N is not None and N.any(): outline(a, N, "#FFFFFF", lw=0.8, ls="dashed")
    row_label(ax[i, 0], lab); IDS.append(("methods examples", lab, iid))
clean(ax)
for j, t_ in enumerate(["Image", "Grad-CAM", "Diff-CAM", r"FinerCAM ($\gamma=0.6$)"]): ax[0, j].set_title(t_, fontsize=8)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.01, top=0.88, wspace=0.04, hspace=0.04)
band(fig, ax, 1, 3, "Globally aligned model", C_GLOB, 0.96)
save(fig, "fig_ch6_methods_examples"); plt.show()


# ---- cell 116 ------------------------------------------------------------
F7 = pd.read_csv(TAB / "ch7_grounding.csv"); e = F7[F7.setting == "HA2 b1"].set_index("image_id").mel_in
PICK7 = []
for c in ["MEL", "NV"]:
    s = e[[i for i in e.index if EV.loc[i].gt_label == c]]; PICK7 += s.iloc[np.argsort((s - s.median()).abs().values)[:2]].index.tolist()
GR = [("Global only", (5.0, 0.0), C_GLOB), (r"Expert only, $\lambda_{\mathrm{expert}}$ 1", (0.0, 1.0), C_EXP),
      (r"Joint, $\lambda_{\mathrm{expert}}$ 5", (5.0, 5.0), C_JOINT)]
W = 9.0
fig, ax = plt.subplots(4, 6, figsize=(W, W / 6 * 4 + 0.6))
for i, (iid, lab) in enumerate(numbered(PICK7)):
    r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    L, U, N = LESION[iid], HUM[iid], nv_union(iid)
    for g, (_, (a_, b_), _) in enumerate(GR):
        m = models.f2(a_, b_, FOLD[iid]); mel, nv = cams.class_cams(m, x); del m
        for k, v in enumerate([mel, nv]):
            a = ax[i, 2 * g + k]; overlay(a, rgb, v); outline(a, L, C_LESION, lw=0.8); outline(a, U & L, "#FFFFFF", lw=0.8)
            if N is not None and N.any(): outline(a, N, "#FFFFFF", lw=0.8, ls="dashed")
    row_label(ax[i, 0], lab, fs=9); IDS.append(("channels", lab, iid))
clean(ax)
for j in range(6): map_head(ax[0, j], "MEL" if j % 2 == 0 else "NV")
fig.subplots_adjust(left=0.04, right=0.995, bottom=0.01, top=0.86, wspace=0.03, hspace=0.03)
for g, (lab, _, col) in enumerate(GR): band(fig, ax, 2 * g, 2 * g + 1, lab, col, 0.955)
save(fig, "fig_ch7_channels"); plt.show()


# ---- cell 117 ------------------------------------------------------------
METH = {"Grad-CAM": lambda m, x: cams.target_ref(m, x, "MEL")[0],
        "Diff-CAM": lambda m, x: cams.diff_cam(m, x, "MEL"),
        "FinerCAM 0.6": lambda m, x: cams.finer_cam(m, x, "MEL", 0.6)}
TT = ["Grad-CAM", "Diff-CAM", r"FinerCAM ($\gamma=0.6$)"]
GRP = [("Globally aligned", (5.0, 0.0), C_GLOB), ("Expert aligned", (0.0, 1.0), C_EXP)]
SHOW = [i for i in held if EV.loc[i].gt_label == "MEL"][:3]
fig, ax = plt.subplots(3, 7, figsize=(TEXTWIDTH, TEXTWIDTH / 7 * 3 + 0.6))
for i, (iid, lab) in enumerate(numbered(SHOW)):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]
    x = data.to_tensor(data.native(r.image_rel_path)); rgb = data.rgb224(r.image_rel_path)
    ax[i, 0].imshow(rgb); outline(ax[i, 0], L, C_LESION, lw=0.8); outline(ax[i, 0], U & L, "#FFFFFF", lw=0.8)
    for g, (_, (a_, b_), _) in enumerate(GRP):
        m = models.f2(a_, b_, FOLD[iid])
        for j, mk in enumerate(METH):
            a = ax[i, 1 + 3 * g + j]; overlay(a, rgb, METH[mk](m, x))
            outline(a, L, C_LESION, lw=0.6); outline(a, U & L, "#FFFFFF", lw=0.6)
        del m
    row_label(ax[i, 0], lab, fs=7); IDS.append(("contrastive expert", lab, iid))
clean(ax); ax[0, 0].set_title("Image", fontsize=8)
for g in range(2):
    for j, t in enumerate(TT): ax[0, 1 + 3 * g + j].set_title(t, fontsize=8)
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.01, top=0.80, wspace=0.04, hspace=0.04)
for g, (lab, _, col) in enumerate(GRP): band(fig, ax, 1 + 3 * g, 3 + 3 * g, lab, col, 0.93)
save(fig, "fig_ch7_contrastive_expert"); plt.show()


# ---- cell 118 ------------------------------------------------------------
ID_T = pd.DataFrame(IDS, columns=["figure", "panel", "image_id"]).drop_duplicates()
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
print(ID_T.to_string(index=False))
esc = lambda s: str(s).replace("_", r"\_")
lines = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
lines += [f"{esc(r.figure)} & {esc(r.panel)} & {esc(r.image_id)} \\\\" for r in ID_T.itertuples()]
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "app_figure_images.tex").write_text("\n".join(lines) + "\n")
print("written app_figure_images.tex |", len(ID_T), "rows")


# ---- cell 119 ------------------------------------------------------------
GX = pd.read_csv(TAB / "ch6_group_excess.csv"); GX["image_id"] = GX.image_id.astype(str)
net = GX[(GX.group == "NETWORK") & (GX.image_id != "ISIC_0029698")].nlargest(2, "excess").image_id.tolist()
ATTR = ["ISIC_0029698"] + net
print("structure analysis rows (figure's own rule):", ATTR)

ID_T = pd.read_csv(TAB / "app_figure_images.csv")
ID_T = ID_T[ID_T.figure != "structure analysis"]
ID_T = pd.concat([ID_T, pd.DataFrame({"figure": "structure analysis", "panel": [f"row {k}" for k in (1, 2, 3)],
                                      "image_id": ATTR})], ignore_index=True)
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
esc = lambda s: str(s).replace("_", r"\_")
lines = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
lines += [f"{esc(r.figure)} & {esc(r.panel)} & {esc(r.image_id)} \\\\" for r in ID_T.itertuples()]
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "app_figure_images.tex").write_text("\n".join(lines) + "\n")

EXPECT = {"localisation examples": 6, "lesion tracking, melanoma": 1, "lesion tracking, nevus": 1, "sweep example": 1,
          "methods examples": 4, "channels": 4, "contrastive expert": 3, "structure analysis": 3}
got = ID_T.figure.value_counts().to_dict()
print("rows per figure:", got)
print("missing or short:", {k: f"{got.get(k, 0)} of {v}" for k, v in EXPECT.items() if got.get(k, 0) != v} or "none")
print("written app_figure_images.tex |", len(ID_T), "rows")


# ---- cell 120 ------------------------------------------------------------
def logit_gaps(m, X, bs=32):
    p = next(m.parameters()); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs]).to(device=p.device, dtype=p.dtype)
            o = m(xb); l = o["logits"] if isinstance(o, dict) else (o[0] if isinstance(o, (tuple, list)) else o)
            out.append((l[:, MEL_IDX] - l[:, NV_IDX]).float().cpu().numpy())
    return np.concatenate(out)

print("X_TEST:", type(X_TEST).__name__, getattr(X_TEST, "dtype", None), "| images", len(X_TEST))
for nm, m in [("unaligned", M0), ("global aligned", M5)]:
    g = np.abs(logit_gaps(m, X_TEST))
    print(f"{nm:15} |logit gap| median {np.median(g):.2f}  99th pct {np.percentile(g, 99):.2f}  "
          f"max {g.max():.2f}  above 16.6: {int((g > 16.6).sum())} of {len(g)}  (fp32 cliff 16.6)")


# ---- cell 121 ------------------------------------------------------------
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from PIL import Image
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from src.thesis import plotting
from src.thesis.plotting import overlay, outline, C_LESION, TEXTWIDTH
from src.thesis.config import BLOCK, MEL_IDX, NV_IDX
if "_save0" in globals(): plotting.save = _save0
save = plotting.save
need = ["EV", "HUM", "LESION", "TEST_DF", "HAM", "M0", "M5", "TAB", "SEED", "models", "cams", "data", "metrics"]
missing = [n for n in need if n not in globals()]
assert not missing, f"Run the 16g header and cell 0 first. Missing: {missing}"
C_MEL, C_NV, C_UNAL, C_GLOB, C_REF, C_CHANCE = (plotting.C_MEL, plotting.C_NV, plotting.C_UNAL,
                                                plotting.C_GLOB, plotting.C_REF, plotting.C_CHANCE)
OUT = {}
def lesion_grid(rel): return data.to_grid(np.array(Image.open((HAM / str(rel)).resolve()).convert("L")) > 127)
def cos(a, b): a, b = np.ravel(a), np.ravel(b); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def mm(v): v = np.asarray(v, float); return (v - v.min()) / (v.max() - v.min() + 1e-12)
def fp(p): return "p < 10^-10" if p < 1e-10 else (f"p = {p:.2g}" if p < 0.001 else f"p = {p:.3f}")
def map_head(a, which, fs=8, title=True):
    t, c = ("Melanoma map", C_MEL) if which == "MEL" else ("Nevus map", C_NV)
    (a.set_title if title else a.set_ylabel)(t, fontsize=fs, color=c)
def clean(ax):
    for a in np.ravel(ax): a.set_xticks([]); a.set_yticks([])
print("setup ok")


# ---- cell 122 ------------------------------------------------------------
T = TEST_DF.assign(image_id=TEST_DF.image_id.astype(str)).reset_index(drop=True)
rows = []
for _, r in T.iterrows():
    L = lesion_grid(r.mask_rel_path); rec = {"image_id": r.image_id, "gt": r.gt_label, "cells": int(L.sum())}
    if 0 < L.sum() < L.size:
        x = data.to_tensor(data.native(r.image_rel_path)); rec["area"] = float(L.mean())
        for nm, m in [("unaligned", M0), ("globally aligned", M5)]:
            mel, nv = cams.class_cams(m, x); t = np.asarray(mel if r.gt_label == "MEL" else nv, float).reshape(14, 14)
            tn = mm(t) if t.max() > t.min() else t
            rec[f"loc_{nm}"] = metrics.lesion_auc(t, L)
            rec[f"share_{nm}"] = float(tn[L].sum() / (tn.sum() + 1e-8))
    rows.append(rec)
LT = pd.DataFrame(rows); LT.to_csv(TAB / "ch4_localisation_test.csv", index=False)
bad = LT[(LT["cells"] == 0) | (LT["cells"] == 196)]
D = LT.dropna(subset=["loc_unaligned", "loc_globally aligned"])
OUT.update(bad_ids=bad["image_id"].tolist(), bad_gt=bad["gt"].tolist(), bad_cells=bad["cells"].tolist(),
           n_loc=len(D), area=float(D["area"].median()))
for key in ["loc", "share"]:
    r_ = metrics.paired(D[f"{key}_globally aligned"], D[f"{key}_unaligned"])
    OUT[key] = dict(u=float(D[f"{key}_unaligned"].median()), g=float(D[f"{key}_globally aligned"].median()),
                    p=r_["p"], wins=r_["wins"], n=r_["n"])
print("1. excluded:", list(zip(OUT["bad_ids"], OUT["bad_gt"], OUT["bad_cells"])),
      "| same as lesion tracking exclusion:", OUT["bad_ids"] == ["ISIC_0027211"], "| images used:", OUT["n_loc"])
print("2.", {k: OUT[k] for k in ["loc", "share"]}, "| median lesion area", round(OUT["area"], 3))

fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7))
for a, key, chance, ylab, tag in [(ax[0], "loc", 0.5, "lesion localisation", "(a)"),
                                  (ax[1], "share", OUT["area"], "heat share in lesion", "(b)")]:
    for j, (nm, col) in enumerate([("unaligned", C_UNAL), ("globally aligned", C_GLOB)]):
        v = D[f"{key}_{nm}"].values; jit = np.random.default_rng(SEED + j).uniform(-0.18, 0.18, len(v))
        a.scatter(j + jit, v, s=2, color=col, alpha=0.15, lw=0, zorder=1)
        b = a.boxplot(v, positions=[j], widths=0.5, patch_artist=True, showfliers=False, zorder=2)
        b["boxes"][0].set(facecolor=to_rgba(col, 0.35), edgecolor=col)
        for k_ in ["whiskers", "caps"]:
            for ln in b[k_]: ln.set_color(col)
        b["medians"][0].set(color="black", lw=1.4)
    a.axhline(chance, color="0.2", ls=":", lw=1.0, zorder=0)
    a.set_xticks([0, 1]); a.set_xticklabels(["Unaligned", "Globally aligned"], fontsize=8)
    a.set_ylim(0, 1); a.set_ylabel(ylab, fontsize=8); a.tick_params(labelsize=7)
    a.text(0.02, 0.98, tag, transform=a.transAxes, va="top", fontsize=8)
fig.tight_layout(); save(fig, "fig_ch4_localisation_box"); plt.show()


# ---- cell 123 ------------------------------------------------------------
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv")
mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
R = (LAMC[LAMC.image_id.isin(mel_ids) & LAMC["lambda"].isin([0.0, 0.1, 1.0, 5.0, 17.0])]
     .pivot_table(index="image_id", columns="lambda", values="corr_in").dropna())
IID = (R - R.median()).abs().sum(axis=1).idxmin()
r = EV.loc[IID]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
L, U = LESION[IID], HUM[IID]; k = np.flatnonzero(L.ravel())

def head_maps(model):
    with torch.enable_grad():
        base, logits = cams._forward(model, x)
        A = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A, retain_graph=True)[0]
        Gn = torch.autograd.grad(logits[0, NV_IDX], A)[0]
    gap = bool(getattr(base, "use_mean_pooling", False)); base.clear_xai_state()
    f = (lambda T_: T_[0, :, 1:, 1:]) if gap else (lambda T_: T_[0, :, 0, 1:])
    A, Gm, Gn = [f(T_).detach().float().cpu().numpy() for T_ in (A, Gm, Gn)]
    Hm, Hn = A * Gm, A * Gn
    if gap: Hm, Hn = Hm.sum(1), Hn.sum(1)
    return Hm.reshape(-1, 14, 14), Hn.reshape(-1, 14, 14)

def heads_fig(model, name, tag):
    Hm, Hn = head_maps(model); nh = Hm.shape[0]
    cv = [cos(Hm[h].ravel()[k], Hn[h].ravel()[k]) for h in range(nh)]
    cmean = cos(Hm.mean(0).ravel()[k], Hn.mean(0).ravel()[k]); low = sorted(np.argsort(cv)[:3].tolist())
    fig, ax = plt.subplots(2, nh + 1, figsize=(TEXTWIDTH, 1.75))
    for j in range(nh + 1):
        vm, vn = (Hm[j], Hn[j]) if j < nh else (Hm.mean(0), Hn.mean(0))
        for i, v in enumerate([vm, vn]):
            overlay(ax[i, j], rgb, mm(v)); outline(ax[i, j], L, C_LESION, lw=0.4)
        ax[0, j].set_title(f"Head {j + 1}" if j < nh else "Mean", fontsize=6)
        ax[1, j].set_xlabel(f"{(cv[j] if j < nh else cmean):.2f}", fontsize=6)
    clean(ax); map_head(ax[0, 0], "MEL", fs=6.5, title=False); map_head(ax[1, 0], "NV", fs=6.5, title=False)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.86, bottom=0.15, wspace=0.05, hspace=0.05)
    for j in low:
        p0, p1 = ax[0, j].get_position(), ax[1, j].get_position()
        fig.add_artist(mpatches.Rectangle((p1.x0 - 0.003, p1.y0 - 0.006), p0.x1 - p0.x0 + 0.006, p0.y1 - p1.y0 + 0.012,
                                          transform=fig.transFigure, fill=False, edgecolor="black", lw=1.2))
    save(fig, name); plt.show()
    OUT[tag] = dict(cos=cv, mean=cmean, low=[h + 1 for h in low], med=float(np.median(cv)))
    print(f"3. {tag}: head cosines {np.round(cv, 3).tolist()} | mean over heads {cmean:.3f} | framed heads {OUT[tag]['low']}")

heads_fig(M5, "fig_ch4_heads", "heads_aligned")
heads_fig(M0, "fig_appC_heads_unaligned", "heads_unaligned")

ID_T = pd.read_csv(TAB / "app_figure_images.csv"); ID_T = ID_T[~ID_T.figure.str.startswith("head maps")]
ID_T = pd.concat([ID_T, pd.DataFrame({"figure": ["head maps", "head maps, unaligned"], "panel": "all columns",
                                      "image_id": IID})], ignore_index=True)
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
esc = lambda s: str(s).replace("_", r"\_")
lines = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
lines += [f"{esc(q.figure)} & {esc(q.panel)} & {esc(q.image_id)} \\\\" for q in ID_T.itertuples()]
(TAB / "app_figure_images.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
print("app_figure_images.tex |", len(ID_T), "rows")


# ---- cell 124 ------------------------------------------------------------
SW = pd.read_csv(TAB / "ch5_lambda_sweep.csv"); LC = pd.read_csv(TAB / "ch5_lambda_localisation.csv")
S = SW.merge(LC, on="lambda").sort_values("lambda")
fig, a1 = plt.subplots(figsize=(TEXTWIDTH * 0.8, 2.7)); a2 = a1.twinx()
a1.plot(S["lambda"], S.loc_auc_median, "-o", color=C_GLOB, ms=3, lw=1.4)
a2.plot(S["lambda"], S.ap_drop_pct, "-s", color=C_REF, ms=3, lw=1.2)
a2.plot(S["lambda"], S.auc_drop_pct, "--^", color=C_REF, ms=3, lw=1.2)
a2.axhline(10, color=C_CHANCE, ls=":", lw=1.0)
a1.set_xscale("symlog", linthresh=0.1); XT = [0, 0.1, 0.5, 1, 3, 5, 11, 17]
a1.set_xticks(XT); a1.set_xticklabels([f"{t:g}" for t in XT], fontsize=7)
a1.set_xlabel(r"$\lambda_{\mathrm{global}}$", fontsize=8)
a1.set_ylim(0, 1); a1.set_ylabel("lesion localisation (median)", fontsize=8, color=C_GLOB)
a2.set_ylim(0, max(45, S.ap_drop_pct.max() * 1.1)); a2.set_ylabel("drop against $\\lambda_{\\mathrm{global}}$ = 0 (%)", fontsize=8, color=C_REF)
a1.tick_params(labelsize=7); a2.tick_params(labelsize=7)
a1.legend([Line2D([], [], color=C_GLOB, marker="o", ms=3), Line2D([], [], color=C_REF, marker="s", ms=3),
           Line2D([], [], color=C_REF, ls="--", marker="^", ms=3)],
          ["lesion localisation", "AP drop", "AUC drop"], frameon=False, fontsize=7, loc="center right")
fig.tight_layout(); save(fig, "fig_ch4_sweep_main"); plt.show()
OUT["sweep"] = dict(auc_max=float(S.auc_drop_pct.max()), lam_auc_max=float(S.loc[S.auc_drop_pct.idxmax(), "lambda"]),
                    ap17=float(S.loc[S["lambda"] == 17, "ap_drop_pct"].iloc[0]), auc5=float(S.loc[S["lambda"] == 5, "auc_drop_pct"].iloc[0]))
print("4.", OUT["sweep"])


# ---- cell 125 ------------------------------------------------------------
from matplotlib.collections import LineCollection
_plain_save = plotting.save
def _chance_hook(fig, name, *a, **kw):
    if name == "fig_ch6_methods_auc":
        n = 0
        for ax_ in fig.axes:
            hit = False
            for ln in list(ax_.lines):
                y = np.asarray(ln.get_ydata(), float)
                if y.size and np.allclose(y, 0.5) and ln.get_linestyle() in (":", "--", "-."): ln.remove(); hit = True
            for c in list(ax_.collections):
                seg = c.get_segments() if isinstance(c, LineCollection) else []
                if len(seg) and all(np.allclose(s[:, 1], 0.5) for s in seg): c.remove(); hit = True
            if hit: ax_.axhline(0.5, color="0.15", ls=":", lw=1.0, zorder=0.5); n += 1
        print(f"chance line restyled on {n} panel(s)" if n else "no dotted 0.5 line found")
        plotting.save = _plain_save; globals()["save"] = _plain_save
    return _plain_save(fig, name, *a, **kw)
plotting.save = _chance_hook; save = _chance_hook
print('hook on. Now rerun the 16g cell that ends with save(fig, "fig_ch6_methods_auc").')


# ---- cell 126 ------------------------------------------------------------
DM = pd.read_csv(TAB / "ch5_deletion_curves_methods.csv"); DM["image_id"] = DM.image_id.astype(str)
g = DM[(DM["model"] == "global aligned") & (DM["method"] == "Grad-CAM")].copy()
g["f"] = g["frac"].round(2); g["cls"] = g["image_id"].map(lambda i: EV.loc[i].gt_label)
g = g[g["f"].isin([0.6, 0.8, 1.0])]
OUT["del_all"] = g.groupby("f")["drop"].median().round(3).to_dict()
OUT["del_cls"] = g.groupby(["f", "cls"])["drop"].median().round(3).unstack().to_dict()
e = g[g["f"] == 1.0]
OUT["del_neg"] = {c: (int((e[e["cls"] == c]["drop"] < 0).sum()), int((e["cls"] == c).sum())) for c in ["MEL", "NV"]}
print("6. all:", OUT["del_all"], "| by class:", OUT["del_cls"], "| negative at 100 %:", OUT["del_neg"])


# ---- cell 127 ------------------------------------------------------------
import re, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from matplotlib.figure import Figure
from src.thesis import plotting
from src.thesis.plotting import TEXTWIDTH
need = ["EV", "HUM", "LESION", "QC", "TEST_DF", "M0", "M5", "TAB", "REPO", "MELNV", "models", "cams", "data", "metrics"]
missing = [n for n in need if n not in globals()]
assert not missing, f"Run the 16g header and cell 0 first. Missing: {missing}"
assert "block_maps" in globals(), "Run the 16g cell that saves fig_appC_layers once, it defines block_maps. Then rerun this cell."
FOLD = data.fold_of_heldout(); held = [i for i in sorted(HUM) if i in FOLD]
OUT = {}
def numbered(ids):
    n = {"MEL": 0, "NV": 0}; out = []
    for i in ids:
        g = EV.loc[i].gt_label; n[g] += 1; out.append(f"{'Melanoma' if g == 'MEL' else 'Nevus'} {n[g]}")
    return out
def cells_of(line): return [c.strip() for c in re.split(r"(?<!\\)&", line.rstrip().removesuffix("\\\\").rstrip())]
def is_row(line): return "&" in line and line.rstrip().endswith("\\\\")
print("setup ok |", len(held), "held-out images")


# ---- cell 128 ------------------------------------------------------------
# item 2: median r_in per block, 29 held-out images, expert aligned scored by its own fold
rows = []
for k in range(5):
    mods = {"unaligned": M0, "global aligned": M5, "expert aligned": models.f2(0.0, 1.0, k)}
    for iid in [i for i in held if FOLD[i] == k]:
        x = data.to_tensor(data.native(EV.loc[iid].image_rel_path)); L = LESION[iid]
        for nm, m in mods.items():
            for b in range(12):
                mel, nv = block_maps(m, x, b)
                rows.append({"model": nm, "block": b + 1, "image_id": iid, "r_in": metrics.corr_in(mel, nv, L)})
    del mods["expert aligned"]
RIN = pd.DataFrame(rows); RIN.to_csv(TAB / "appC_layers_rin.csv", index=False)
RM = RIN.groupby(["model", "block"]).r_in.median().unstack(0)
OUT["rin12"] = RM.loc[12].round(3).to_dict(); OUT["rin_n"] = RIN.image_id.nunique()
print("2. median r_in per block:\n", RM.round(3).to_string())

# item 7: lesion size figure, how its y axis is built, and median size in cells
src = ""
for c in json.load(open(REPO / "notebooks/mel_nv/16g_verify.ipynb"))["cells"]:
    s = "".join(c["source"])
    if '"fig_appA_lesion_size"' in s: src = s
seg = src[max(0, src.find("lesion")): ] if src else ""
OUT["density"] = ("density=True" in src) or ("kde" in src.lower())
OUT["size_uses_test"] = "TEST_DF" in src
print("7. lesion size source lines:\n  " + "\n  ".join(l.strip() for l in src.splitlines()
      if re.search(r"hist|kde|density|ylabel|TEST_DF|lesion_size", l)))
LT = pd.read_csv(TAB / "ch4_localisation_test.csv"); LT = LT[LT["cells"] > 0]
OUT["size"] = LT.groupby("gt")["cells"].median().to_dict(); OUT["size_n"] = LT.groupby("gt").size().to_dict()
print("7. density per class:", OUT["density"], "| median lesion cells (test):", OUT["size"], OUT["size_n"])

# item 8: images per lesion by split and class
H0 = pd.read_csv(MELNV / "ham_mel_nv_clean.csv")
sc = next((c for c in ["split", "Split", "partition"] if c in H0.columns), None)
assert sc and "lesion_id" in H0.columns, f"columns in clean csv: {list(H0.columns)}"
H0["cls"] = H0["gt_label"] if "gt_label" in H0.columns else H0["dx"].map({"mel": "MEL", "nv": "NV"})
def per_lesion(df, split_name):
    out = []
    for c in ["MEL", "NV", "all"]:
        d = df if c == "all" else df[df["cls"] == c]; n = d.groupby("lesion_id").size()
        out.append({"split": split_name, "class": c, "images": len(d), "lesions": len(n),
                    "images per lesion": round(len(d) / max(len(n), 1), 2), "lesions with >1 image": int((n > 1).sum())})
    return out
rows = []
for sp in H0[sc].dropna().unique(): rows += per_lesion(H0[H0[sc] == sp], str(sp))
rows += per_lesion(H0[H0.image_id.astype(str).isin(set(TEST_DF.image_id.astype(str)))], "test, final 987")
IPL = pd.DataFrame(rows); IPL.to_csv(TAB / "appA_images_per_lesion.csv", index=False); OUT["ipl"] = IPL
print("8. images per lesion:\n", IPL.to_string(index=False))


# ---- cell 129 ------------------------------------------------------------
# item 4: case manifest
p = TAB / "app_case_manifest.tex"; lines = p.read_text().splitlines()
hdr = next(l for l in lines if is_row(l)); cols = [c.lower() for c in cells_of(hdr)]
gi = cols.index("selection group") if "selection group" in cols else None
ti = next(i for i, c in enumerate(cols) if "target" in c)
YES, NO = {"yes", "true", "1", "x", "y"}, {"no", "false", "0", "", "-", "--", "n"}
new, marks = [], {"cmark": 0, "xmark": 0}
for l in lines:
    if is_row(l):
        c = cells_of(l)
        v = c[ti].lower()
        if v in YES: c[ti] = r"\cmark"; marks["cmark"] += 1
        elif v in NO: c[ti] = r"\xmark"; marks["xmark"] += 1
        elif v in cols[ti]: c[ti] = "melanoma target"
        if gi is not None: del c[gi]
        l = " & ".join(c) + r" \\"
    new.append(re.sub("preeminent", "prominent", l, flags=re.I))
s = "\n".join(new)
keep = [c for i, c in enumerate(cols) if i != gi]
spec = "".join("p{8cm}" if "label" in c else ("c" if "target" in c else "l") for c in keep)
m = re.search(r"\\begin\{(tabular|longtable)\}(\[[^\]]*\])?\{([^}]*)\}", s); s = s[:m.start(3)] + spec + s[m.end(3):]
p.write_text(s + "\n")
print("4. case manifest columns:", keep, "| spec", spec, "| marks", marks,
      "| check" if marks["cmark"] + marks["xmark"] == 0 else "| ok")

# item 6: metric validity table, medians only, Grad-CAM and random map, metrics as rows
p = TAB / "table_metric_validity.tex"; t = p.read_text(); R = [cells_of(l) for l in t.splitlines() if is_row(l)]
strip = lambda c: re.sub(r"\s*[\[\(][^\]\)]*[\]\)]", "", re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)).strip()
head = [strip(c) for c in R[0]]
def find(names, pool): return next((i for i, c in enumerate(pool) if c.lower().replace(" ", "") in names), None)
GC, RN = {"grad-cam", "gradcam"}, {"random", "randommap"}
ci_g, ci_r = find(GC, head), find(RN, head)
if ci_g is not None and ci_r is not None:          # methods are columns, metrics are rows
    MV = [(strip(r[0]), strip(r[ci_g]), strip(r[ci_r])) for r in R[1:]]
else:                                              # methods are rows, metrics are columns
    rows_ = {strip(r[0]).lower().replace(" ", ""): r for r in R[1:]}
    rg = next(v for k, v in rows_.items() if k in GC); rr = next(v for k, v in rows_.items() if k in RN)
    MV = [(head[j], strip(rg[j]), strip(rr[j])) for j in range(1, len(head))]
def differ(a, b):
    try: return abs(float(a) - float(b)) > 1e-9
    except ValueError: return a != b
out = [r"\begin{tabular}{lccc}", r"\toprule", r"metric & Grad-CAM & random map & separates from random \\", r"\midrule"]
# out += [f"{mn} & {a} & {b} & {r'\cmark' if differ(a, b) else r'\xmark'} \\\\" for mn, a, b in MV]
# p.write_text("\n".join(out + [r"\bottomrule", r"\end{tabular}"]) + "\n")
# OUT["mv"] = [(mn, a, b, differ(a, b)) for mn, a, b in MV]
# print("6. metric validity:\n  " + "\n  ".join(f"{mn}: {a} | {b} | {'yes' if d else 'no'}" for mn, a, b, d in OUT["mv"]),
#       "\n  check" if not MV else "")
CM, XM = r"\cmark", r"\xmark"
out = [r"\begin{tabular}{lccc}", r"\toprule",
       r"metric & Grad-CAM & random map & separates from random \\", r"\midrule"]
out += [f"{mn} & {a} & {b} & {CM if differ(a, b) else XM} \\\\" for mn, a, b in MV]
p.write_text("\n".join(out + [r"\bottomrule", r"\end{tabular}"]) + "\n")
OUT["mv"] = [(mn, a, b, differ(a, b)) for mn, a, b in MV]
print("6. metric validity:\n  " + "\n  ".join(
    f"{mn}: {a} | {b} | {'yes' if d else 'no'}" for mn, a, b, d in OUT["mv"]))

# item 3: structure analysis rows, labelled by diagnosis
GX = pd.read_csv(TAB / "ch6_group_excess.csv"); GX["image_id"] = GX.image_id.astype(str)
ATTR = ["ISIC_0029698"] + GX[(GX.group == "NETWORK") & (GX.image_id != "ISIC_0029698")].nlargest(2, "excess").image_id.tolist()
ATTR_LAB = numbered(ATTR); OUT["attr"] = list(zip(ATTR_LAB, ATTR))
ID_T = pd.read_csv(TAB / "app_figure_images.csv"); ID_T = ID_T[ID_T.figure != "structure analysis"]
ID_T = pd.concat([ID_T, pd.DataFrame({"figure": "structure analysis", "panel": ATTR_LAB, "image_id": ATTR})], ignore_index=True)
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
esc = lambda v: str(v).replace("_", r"\_")
L_ = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
L_ += [f"{esc(q.figure)} & {esc(q.panel)} & {esc(q.image_id)} \\\\" for q in ID_T.itertuples()]
(TAB / "app_figure_images.tex").write_text("\n".join(L_ + [r"\bottomrule", r"\end{tabular}"]) + "\n")
print("3. structure analysis rows:", OUT["attr"], "| app_figure_images:", len(ID_T), "rows")


# ---- cell 130 ------------------------------------------------------------
import matplotlib.patches as mpatches
def h_sites(fig):
    w, h = fig.get_size_inches(); fig.set_size_inches(TEXTWIDTH, max(h, 2.6))
    for a in fig.axes: plt.setp(a.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
def h_layers(fig):
    w, h = fig.get_size_inches(); extra = 1.9; H = h + extra
    for a in fig.axes:
        q = a.get_position(); a.set_position([q.x0, (q.y0 * h + extra) / H, q.width, q.height * h / H])
    fig.set_size_inches(w, H); f = extra / H
    ax = fig.add_axes([0.10, 0.30 * f, 0.86, 0.60 * f])
    for nm in ["unaligned", "global aligned", "expert aligned"]:
        ax.plot(RM.index, RM[nm], "-o", ms=2.5, lw=1.2, color=plotting.MODEL_COLOR[nm], label=nm)
    ax.axhline(0, color=plotting.C_CHANCE, lw=0.8, ls=":"); ax.axvline(12, color="0.3", lw=0.9, ls="--")
    ax.text(11.85, 0.92, "final block", ha="right", va="top", fontsize=7, color="0.3")
    ax.set(xlim=(0.5, 12.5), ylim=(-1, 1), xticks=range(1, 13)); ax.tick_params(labelsize=7)
    ax.set_xlabel("transformer block", fontsize=8); ax.set_ylabel(r"$r_{\mathrm{in}}$ (median)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=3)
    ax.text(-0.07, 1.02, "(e)", transform=ax.transAxes, fontsize=8)
def h_attr(fig):
    for a in fig.axes:
        for t in [a.title, a.xaxis.label, a.yaxis.label, *a.texts]: t.set_text(re.sub(r"ISIC_\d+\s*", "", t.get_text()).strip())
    for t in fig.texts: t.set_text(re.sub(r"ISIC_\d+\s*", "", t.get_text()).strip())
    for i, lab in enumerate(ATTR_LAB): fig.axes[3 * i].set_ylabel(lab, fontsize=7, color="black")
HANDLERS = {"fig_appA_sites": h_sites, "fig_appC_layers": h_layers, "fig_ch6_attribution_examples": h_attr}
if not hasattr(Figure, "_orig_savefig"): Figure._orig_savefig = Figure.savefig
def _savefig(self, fname, *a, **k):
    stem = Path(str(fname)).stem; done = getattr(self, "_hooked", set())
    if stem in HANDLERS and stem not in done: HANDLERS[stem](self); done.add(stem); self._hooked = done
    return Figure._orig_savefig(self, fname, *a, **k)
Figure.savefig = _savefig
print("hooks on. Rerun: the 16g cell with fig_appA_sites, the 16g fig_appC_layers cell, and 16f cell 7 copied into 16g.")


# ---- cell 131 ------------------------------------------------------------
import json, re
for c in json.load(open(REPO / "notebooks/mel_nv/16g_verify.ipynb"))["cells"]:
    s = "".join(c["source"])
    if re.search(r'save\(fig,\s*"fig_appA_lesion_size"', s):
        print("\n".join(l.strip() for l in s.splitlines() if re.search(r"hist|kde|density|ylabel", l)))


# ---- cell 132 ------------------------------------------------------------
import re, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.figure import Figure
GX = pd.read_csv(TAB / "ch6_group_excess.csv"); GX["image_id"] = GX.image_id.astype(str)
ATTR = ["ISIC_0029698"] + GX[(GX.group == "NETWORK") & (GX.image_id != "ISIC_0029698")].nlargest(2, "excess").image_id.tolist()
cnt = {"MEL": 0, "NV": 0}; ATTR_LAB = []
for i in ATTR:
    g = EV.loc[i].gt_label; cnt[g] += 1; ATTR_LAB.append(f"{'Melanoma' if g == 'MEL' else 'Nevus'} {cnt[g]}")
def h_attr(fig):
    for a in fig.axes:
        for t in [a.title, a.xaxis.label, a.yaxis.label, *a.texts]: t.set_text(re.sub(r"ISIC_\d+\s*", "", t.get_text()).strip())
    for t in fig.texts: t.set_text(re.sub(r"ISIC_\d+\s*", "", t.get_text()).strip())
    for i, lab in enumerate(ATTR_LAB): fig.axes[3 * i].set_ylabel(lab, fontsize=7, color="black")
if not hasattr(Figure, "_orig_savefig"): Figure._orig_savefig = Figure.savefig
def _savefig(self, fname, *a, **k):
    if Path(str(fname)).stem == "fig_ch6_attribution_examples" and not getattr(self, "_attr_done", False):
        h_attr(self); self._attr_done = True
    return Figure._orig_savefig(self, fname, *a, **k)
Figure.savefig = _savefig
print("ready |", list(zip(ATTR_LAB, ATTR)), "| now paste and run 16f cell 7 below")


# ---- cell 133 ------------------------------------------------------------
import json, re, matplotlib.text as mtext
from matplotlib.figure import Figure
if hasattr(Figure, "_orig_savefig"): Figure.savefig = Figure._orig_savefig     # switch off the earlier save hooks

GX = pd.read_csv(TAB / "ch6_group_excess.csv"); GX["image_id"] = GX.image_id.astype(str)
ATTR = ["ISIC_0029698"] + GX[(GX.group == "NETWORK") & (GX.image_id != "ISIC_0029698")].nlargest(2, "excess").image_id.tolist()
cnt = {"MEL": 0, "NV": 0}; ATTR_LAB = []
for i in ATTR:
    g = EV.loc[i].gt_label; cnt[g] += 1; ATTR_LAB.append(f"{'Melanoma' if g == 'MEL' else 'Nevus'} {cnt[g]}")

def _fix_attr(fig):
    for t in fig.findobj(mtext.Text):                                      # every text in the figure
        s = t.get_text()
        if "ISIC" in s: t.set_text(re.sub(r"ISIC_\d+[\s,:|()-]*", "", s).strip())
    rows = {}
    for a in fig.axes:
        rows.setdefault(round(a.get_position().y0, 2), []).append(a)
    left = [min(v, key=lambda a: a.get_position().x0) for k, v in sorted(rows.items(), reverse=True) if len(v) >= 2]
    for a, lab in zip(left, ATTR_LAB): a.set_ylabel(lab, fontsize=7, color="black")
    left_isic = sum("ISIC" in t.get_text() for t in fig.findobj(mtext.Text))
    print(f"rows labelled: {len(left)} {ATTR_LAB[:len(left)]} | ISIC texts left: {left_isic}")

nb = json.load(open(REPO / "notebooks/mel_nv/16f_figures_tables.ipynb"))
src = next("".join(c["source"]) for c in nb["cells"]
           if '"fig_ch6_attribution_examples"' in "".join(c["source"]) and "ch6_group_excess" in "".join(c["source"]))
key = 'save(fig, "fig_ch6_attribution_examples")'
assert src.count(key) == 1, "save line not found exactly once in 16f cell 7"
exec(src.replace(key, "_fix_attr(fig); " + key), globals())


# ---- cell 134 ------------------------------------------------------------
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from src.thesis import plotting
from src.thesis.plotting import overlay, outline, C_LESION
from src.thesis.config import BLOCK, MEL_IDX, NV_IDX
if hasattr(Figure, "_orig_savefig"): Figure.savefig = Figure._orig_savefig
save = plotting.save; C_MEL, C_NV = plotting.C_MEL, plotting.C_NV
def cos(a, b): a, b = np.ravel(a), np.ravel(b); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def mm(v): v = np.asarray(v, float); return (v - v.min()) / (v.max() - v.min() + 1e-12)
def head_maps(model, x):
    with torch.enable_grad():
        base, logits = cams._forward(model, x)
        A = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A, retain_graph=True)[0]
        Gn = torch.autograd.grad(logits[0, NV_IDX], A)[0]
    base.clear_xai_state()
    A, Gm, Gn = [T_[0, :, 1:, 1:].detach().float().cpu().numpy() for T_ in (A, Gm, Gn)]
    return (A * Gm).sum(1).reshape(-1, 14, 14), (A * Gn).sum(1).reshape(-1, 14, 14)

LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv")
mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
R = (LAMC[LAMC.image_id.isin(mel_ids) & LAMC["lambda"].isin([0.0, 0.1, 1.0, 5.0, 17.0])]
     .pivot_table(index="image_id", columns="lambda", values="corr_in").dropna())
IID = (R - R.median()).abs().sum(axis=1).idxmin()

def heads_fig(model, name):
    r = EV.loc[IID]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    L = LESION[IID]; k = np.flatnonzero(L.ravel()); Hm, Hn = head_maps(model, x); nh = Hm.shape[0]
    cv = [cos(Hm[h].ravel()[k], Hn[h].ravel()[k]) for h in range(nh)]; low = sorted(np.argsort(cv)[:3].tolist())
    pairs = [("Mean of all 12 heads (thesis map)", list(range(nh))), ("Mean of the 3 framed heads", low)]
    pc = [cos(Hm[s].mean(0).ravel()[k], Hn[s].mean(0).ravel()[k]) for _, s in pairs]
    fig = plt.figure(figsize=(9.0, 3.9))
    gs = GridSpec(4, nh, figure=fig, height_ratios=[1, 1, 0.45, 1], left=0.055, right=0.995, top=0.93, bottom=0.07, wspace=0.05, hspace=0.1)
    top = [[fig.add_subplot(gs[i, j]) for j in range(nh)] for i in range(2)]
    for j in range(nh):
        for i, v in enumerate([Hm[j], Hn[j]]): overlay(top[i][j], rgb, mm(v)); outline(top[i][j], L, C_LESION, lw=0.4)
        top[0][j].set_title(f"Head {j + 1}", fontsize=7); top[1][j].set_xlabel(f"{cv[j]:.2f}", fontsize=7)
    top[0][0].set_ylabel("Melanoma map", fontsize=7, color=C_MEL); top[1][0].set_ylabel("Nevus map", fontsize=7, color=C_NV)
    for p, (lab, s) in enumerate(pairs):
        c0 = 3 + 4 * p; ax2 = []
        for q, (v, w) in enumerate([(Hm[s].mean(0), "MEL"), (Hn[s].mean(0), "NV")]):
            a = fig.add_subplot(gs[3, c0 + q]); overlay(a, rgb, cams.norm01(np.maximum(v, 0))); outline(a, L, C_LESION, lw=0.4)
            a.set_title("Melanoma map" if w == "MEL" else "Nevus map", fontsize=7, color=C_MEL if w == "MEL" else C_NV); ax2.append(a)
        p0, p1 = ax2[0].get_position(), ax2[1].get_position()
        fig.text((p0.x0 + p1.x1) / 2, p0.y1 + 0.075, lab, ha="center", fontsize=8, fontweight="bold")
        fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.05, f"{pc[p]:.2f}", ha="center", fontsize=7)
    for a in fig.axes: a.set_xticks([]); a.set_yticks([])
    for j in low:
        p0, p1 = top[0][j].get_position(), top[1][j].get_position()
        fig.add_artist(mpatches.Rectangle((p1.x0 - 0.003, p1.y0 - 0.045), p0.x1 - p0.x0 + 0.006, p0.y1 - p1.y0 + 0.05,
                                          transform=fig.transFigure, fill=False, edgecolor="black", lw=1.2))
    save(fig, name); plt.show()
    return low, pc

def stats31(model, fixed):
    rows = []
    for iid in sorted(HUM):
        x = data.to_tensor(data.native(EV.loc[iid].image_rel_path)); k = np.flatnonzero(LESION[iid].ravel())
        Hm, Hn = head_maps(model, x); cv = [cos(Hm[h].ravel()[k], Hn[h].ravel()[k]) for h in range(Hm.shape[0])]
        own = np.argsort(cv)[:3]; f = lambda s: cos(Hm[s].mean(0).ravel()[k], Hn[s].mean(0).ravel()[k])
        rows.append({"image_id": iid, "all12": f(list(range(Hm.shape[0]))), "low3_own": f(own), "low3_fixed": f(fixed)})
    return pd.DataFrame(rows)

ALLS = []
for tag, model, name in [("aligned", M5, "fig_ch4_heads"), ("unaligned", M0, "fig_appC_heads_unaligned")]:
    low, pc = heads_fig(model, name); S = stats31(model, low).assign(model=tag); ALLS.append(S)
    o = S[S.image_id != IID]
    print(f"RESULT heads {tag}: example all12 {pc[0]:.3f}, framed heads {[h + 1 for h in low]} {pc[1]:.3f} | "
          f"31 images median all12 {S.all12.median():.3f}, own low3 {S.low3_own.median():.3f} | "
          f"fixed heads on other 30: median {o.low3_fixed.median():.3f}, below all12 on {int((o.low3_fixed < o.all12).sum())} of {len(o)}")
pd.concat(ALLS).to_csv(TAB / "ch4_head_subsets.csv", index=False)


# ---- cell 135 ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from PIL import Image
H0 = pd.read_csv(MELNV / "ham_mel_nv_clean.csv"); H0["image_id"] = H0.image_id.astype(str)
sc = next(c for c in ["split", "Split", "partition"] if c in H0.columns)
assert {"image_rel_path", "mask_rel_path"} <= set(H0.columns), f"columns: {list(H0.columns)}"
H0["y"] = (H0["gt_label"] if "gt_label" in H0.columns else H0["dx"].map({"mel": "MEL", "nv": "NV"})).eq("MEL").astype(int)
TR = H0[H0[sc] == "train"]
TRS = pd.concat([TR[TR["y"] == 1], TR[TR["y"] == 0].sample(int((TR["y"] == 1).sum()), random_state=SEED)])
TE = TEST_DF.assign(image_id=TEST_DF.image_id.astype(str), y=TEST_DF.gt_label.eq("MEL").astype(int))

def batches(df, bs=32):
    for i in range(0, len(df), bs): yield df.iloc[i:i + bs]
def load(d, dev, dt):
    x = torch.cat([data.to_tensor(data.native(r)) for r in d.image_rel_path]).to(dev, dt)
    Ls = [data.to_grid(np.array(Image.open((HAM / str(m)).resolve()).convert("L")) > 127) for m in d.mask_rel_path]
    return x, Ls
def block_tokens(model, x):
    base = cams._unwrap_model(model); acts = {}
    def hook(b):
        def f(m, i, o): o = o[0] if isinstance(o, (tuple, list)) else o; acts[b] = o[:, 1:].detach().float().cpu()
        return f
    hs = [base.blocks[b].register_forward_hook(hook(b)) for b in range(12)]
    try:
        with torch.no_grad(): model(x)
    finally:
        for h in hs: h.remove()
        base.clear_xai_state()
    return acts

def run_probe(model, tag):
    p = next(model.parameters()); rng = np.random.default_rng(SEED)
    Xl = {b: [] for b in range(12)}; Xb = {b: [] for b in range(12)}; yl, yb = [], []
    for d in batches(TRS):
        x, Ls = load(d, p.device, p.dtype); A = block_tokens(model, x)
        for n, (L, y) in enumerate(zip(Ls, d["y"].values)):
            li, bi = np.flatnonzero(L.ravel()), np.flatnonzero(~L.ravel())
            sl = rng.choice(li, min(8, len(li)), replace=False) if len(li) else np.array([], int)
            sb = rng.choice(bi, min(8, len(bi)), replace=False) if len(bi) else np.array([], int)
            for b in range(12):
                if len(sl): Xl[b].append(A[b][n, sl].numpy().astype(np.float16))
                if len(sb): Xb[b].append(A[b][n, sb].numpy().astype(np.float16))
            yl += [y] * len(sl); yb += [y] * len(sb)
    PR = {}
    for b in range(12):
        for reg, X, Y in [("lesion", Xl[b], yl), ("background", Xb[b], yb)]:
            PR[(b, reg)] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")) \
                .fit(np.concatenate(X).astype(np.float32), np.array(Y))
    del Xl, Xb; rows = []
    for d in batches(TE):
        x, Ls = load(d, p.device, p.dtype); A = block_tokens(model, x)
        for n, (L, y, iid) in enumerate(zip(Ls, d["y"].values, d.image_id.values)):
            li, bi = np.flatnonzero(L.ravel()), np.flatnonzero(~L.ravel())
            for b in range(12):
                rec = {"model": tag, "image_id": iid, "y": y, "block": b + 1}
                for reg, idx in [("lesion", li), ("background", bi)]:
                    rec[reg] = float(PR[(b, reg)].decision_function(A[b][n, idx].numpy()).mean()) if len(idx) else np.nan
                rows.append(rec)
    out = pd.DataFrame(rows); out.to_csv(TAB / f"appC_token_probe_{tag.replace(' ', '_')}.csv", index=False)
    print(f"{tag}: done, {out.image_id.nunique()} test images")
    return out

M_EXP = models.f2(0.0, 1.0, 0)
PB = pd.concat([run_probe(m, t) for t, m in [("unaligned", M0), ("global aligned", M5), ("expert aligned", M_EXP)]])


# ---- cell 136 ------------------------------------------------------------
from sklearn.metrics import roc_auc_score
from matplotlib.lines import Line2D
def auc(s, y): m = s.notna(); return roc_auc_score(y[m], s[m])
S = (PB.groupby(["model", "block"])
       .apply(lambda g: pd.Series({"lesion": auc(g["lesion"], g["y"]), "background": auc(g["background"], g["y"])})).reset_index())
S.to_csv(TAB / "appC_token_probe_auc.csv", index=False)
rng = np.random.default_rng(SEED)
MODELS = [("unaligned", M0), ("global aligned", M5), ("expert aligned", M_EXP)]
fig, ax = plt.subplots(figsize=(plotting.TEXTWIDTH * 0.8, 2.8))
for nm, m in MODELS:
    s = S[S.model == nm]; col = plotting.MODEL_COLOR[nm]
    ax.plot(s.block, s.lesion, "-o", ms=2.5, lw=1.3, color=col); ax.plot(s.block, s.background, "--", lw=1.2, color=col)
    g = PB[(PB.model == nm) & (PB.block == 12)].dropna(subset=["lesion", "background"]); d = []
    for _ in range(1000):
        i = rng.integers(0, len(g), len(g)); gi = g.iloc[i]
        if 0 < gi["y"].sum() < len(gi): d.append(roc_auc_score(gi["y"], gi["lesion"]) - roc_auc_score(gi["y"], gi["background"]))
    full = roc_auc_score(TE["y"], models.predict_mel(m, X_TEST))
    v = lambda r, b: s.loc[s.block == b, r].iloc[0]
    print(f"RESULT probe {nm}: model test AUC {full:.3f} | lesion b1 {v('lesion',1):.3f} b4 {v('lesion',4):.3f} b8 {v('lesion',8):.3f} b12 {v('lesion',12):.3f} | "
          f"background b1 {v('background',1):.3f} b4 {v('background',4):.3f} b8 {v('background',8):.3f} b12 {v('background',12):.3f} | "
          f"b12 lesion minus background {np.median(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}], n {len(g)}")
ax.axhline(0.5, color="0.3", ls=":", lw=0.9)
ax.set(xlim=(0.5, 12.5), ylim=(0.4, 1.0), xticks=range(1, 13)); ax.tick_params(labelsize=7)
ax.set_xlabel("transformer block", fontsize=8); ax.set_ylabel("probe AUC (987 test images)", fontsize=8)
ax.legend([Line2D([], [], color=plotting.MODEL_COLOR[n]) for n, _ in MODELS] + [Line2D([], [], color="0.3", marker="o", ms=2.5), Line2D([], [], color="0.3", ls="--")],
          [n for n, _ in MODELS] + ["lesion tokens", "background tokens"], frameon=False, fontsize=6.5, ncol=2, loc="lower right")
fig.tight_layout(); save(fig, "fig_appC_token_probe"); plt.show()


# ---- cell 137 ------------------------------------------------------------
def flag_stats(p, y):
    pr = np.asarray(p) >= 0.5; y = np.asarray(y).astype(bool)
    tp, fp, fn = int((pr & y).sum()), int((pr & ~y).sum()), int((~pr & y).sum())
    return dict(tp=tp, fn=fn, fp=fp, flagged=tp + fp, per_mel=(tp + fp) / max(tp, 1), benign_per_mel=fp / max(tp, 1))

R = {"unaligned": flag_stats(models.predict_mel(M0, X_TEST), Y_MEL),
     "globally aligned": flag_stats(models.predict_mel(M5, X_TEST), Y_MEL)}
EX = []
for k in range(5):
    m = models.f2(0.0, 1.0, k); EX.append(flag_stats(models.predict_mel(m, X_TEST), Y_MEL)); del m
E = pd.DataFrame(EX)
R["expert aligned (mean of 5 folds)"] = {c: E[c].mean() for c in E.columns}
NN = pd.DataFrame(R).T; NN.to_csv(TAB / "ch5_nne_equivalent.csv")
print(NN.round(2).to_string()); print("expert per fold:\n", E.round(2).to_string())
g, u, x = R["globally aligned"], R["unaligned"], R["expert aligned (mean of 5 folds)"]
print(f"""=== PROMPT PART 1 ===
Section 5.1 or 5.2, clinical context of false alarms. Plain sentences without antithesis constructions.
"In clinical practice, dermatologists excise about 7.5 to 9.6 lesions per melanoma found and primary care about 22.6 [Nelson et al. 2019, Petty et al. 2020]. On the 987 test images at threshold 0.5, the globally aligned model flags {g['flagged']} lesions to find {g['tp']} of 54 melanomas, {g['per_mel']:.2f} per melanoma. The unaligned model flags {u['flagged']} for {u['tp']} melanomas ({u['per_mel']:.2f}), and the expert aligned model flags {x['flagged']:.1f} for {x['tp']:.1f} on average over its five fold models ({x['per_mel']:.2f}). The figures are not directly comparable. The clinical figures count excisions chosen in practice in populations with a different melanoma prevalence, and the model figure counts curated test images above an arbitrary threshold, so both depend on prevalence and threshold in ways that are not matched." [ch5_nne_equivalent.csv]
References, both [VERIFY: numbers and DOI against the abstracts]:
Nelson, C. A., et al. (2019). Evaluation of the number-needed-to-biopsy metric for the diagnosis of cutaneous melanoma: a systematic review and meta-analysis. JAMA Dermatology, 155(10), 1167-1174. https://doi.org/10.1001/jamadermatol.2019.1514
Petty, A. J., et al. (2020). Meta-analysis of number needed to treat for diagnosis of melanoma by clinical setting. Journal of the American Academy of Dermatology, 82(5), 1158-1165. https://doi.org/10.1016/j.jaad.2019.12.063
=== END ===""")


# ---- cell 138 ------------------------------------------------------------
import matplotlib.patches as mpatches
from src.thesis import plotting
M6 = pd.read_csv(TAB / "ch6_metrics.csv").merge(EV[["gt_label"]], left_on="image_id", right_index=True)
METH = ["Grad-CAM", "FinerCAM 0.6", "Diff-CAM", "random"]
LAB = ["Grad-CAM", r"FinerCAM" + "\n" + r"($\gamma=0.6$)", "Diff-CAM", "random"]
CC = {"MEL": plotting.C_MEL, "NV": plotting.C_NV}
fig, ax = plt.subplots(1, 2, figsize=(plotting.TEXTWIDTH, 2.9))
for a, met, chance, yl, tag in [(ax[0], "auc_in", 0.5, r"$\mathrm{AUC}_{\mathrm{in}}$", "(a)"),
                                (ax[1], "energy_in", 0.0, r"$E_{\mathrm{in}}$", "(b)")]:
    for i, mk in enumerate(METH):
        for j, c in enumerate(["MEL", "NV"]):
            v = M6[(M6.method == mk) & (M6.gt_label == c)][met].dropna().values; pos = i + (j - 0.5) * 0.36
            a.scatter(pos + np.random.default_rng(SEED + i * 2 + j).uniform(-0.07, 0.07, len(v)), v,
                      s=5, color=CC[c], alpha=0.3, lw=0, zorder=1)
            b = a.boxplot(v, positions=[pos], widths=0.3, patch_artist=True, showfliers=False, zorder=2)
            b["boxes"][0].set(facecolor="none", edgecolor=CC[c], lw=1.3)
            for k in ("whiskers", "caps"):
                for ln in b[k]: ln.set(color=CC[c], lw=1.1)
            b["medians"][0].set(color="black", lw=1.5)
    a.axhline(chance, color="0.15", ls=":", lw=1.0, zorder=0)
    a.set_xlim(-0.6, len(METH) - 0.4); a.set_xticks(range(len(METH))); a.set_xticklabels(LAB, fontsize=7)
    a.set_ylabel(yl, fontsize=8); a.tick_params(axis="y", labelsize=7)
    a.text(0.01, 0.99, tag, transform=a.transAxes, va="top", fontsize=8)
ax[0].legend([mpatches.Patch(facecolor="none", edgecolor=CC["MEL"]), mpatches.Patch(facecolor="none", edgecolor=CC["NV"])],
             ["Melanoma", "Nevus"], frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); plotting.save(fig, "fig_ch4_methods_class"); plt.show()

q = lambda s: f"{s.median():.3f} [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]"
n = M6[M6.method == "Grad-CAM"].gt_label.value_counts().to_dict()
rows = []
for mk in METH:
    r = [mk]
    for met in ["auc_in", "energy_in"]:
        for c in ["MEL", "NV"]: r.append(q(M6[(M6.method == mk) & (M6.gt_label == c)][met].dropna()))
    rows.append(" | ".join(r))
P = {}
for met in ["auc_in", "energy_in"]:
    for c in ["MEL", "NV"]:
        t = M6[M6.gt_label == c].pivot_table(index="image_id", columns="method", values=met)
        P[(met, c)] = metrics.paired(t["FinerCAM 0.6"], t["Grad-CAM"])
fp = lambda r: f"{r['diff']:+.3f}, p = {r['p']:.2g}, {r['wins']} of {r['n']} higher"
print(f"""=== PROMPT PART 2 ===
Section 4.4, methods by class. Insert fig_ch4_methods_class.pdf, label fig:methods-class. It replaces the pooled methods table, the by-class methods table (tab:methods-class) and fig_ch6_methods_auc (fig:methods-auc). Remove those three and redirect all references to fig:methods-class.
Caption: "In-lesion AUC (a) and energy excess (b) of four explanation methods for the ground-truth class map of the globally aligned model, 31 annotated images. Red outline: melanomas (n = {n.get('MEL')}). Blue outline: nevi (n = {n.get('NV')}). Boxes show median and interquartile range, whiskers 1.5 times the interquartile range, points individual images. Dotted line: chance, 0.5 in (a) and 0 in (b). On nevi the nevus map is scored against melanoma-indicative structures, so higher values there do not indicate better agreement (Section 5.1)."
Keep the existing text of 4.4 on methods by class and point it to Figure~\\ref{{fig:methods-class}}.
FinerCAM against Grad-CAM, paired Wilcoxon (family 2 per measure, threshold 0.025):
in-lesion AUC, melanomas {fp(P[('auc_in','MEL')])}. Nevi {fp(P[('auc_in','NV')])}.
energy excess, melanomas {fp(P[('energy_in','MEL')])}. Nevi {fp(P[('energy_in','NV')])}.
Appendix C table, label tab:app-methods-class, median [IQR]. Columns: method | AUC_in MEL | AUC_in NV | E_in MEL | E_in NV
""" + "\n".join(rows) + """
Caption: "Values behind Figure~\\ref{fig:methods-class}, median and interquartile range." [ch6_metrics.csv]
=== END ===""")


# ---- cell 139 ------------------------------------------------------------
from scipy.stats import binomtest
for f in ["write", "tx", "fi", "f3", "fp", "METN"]:
    assert f in globals(), f"run the statistics export cell first ({f} missing)"
K1P = REPO / "results/final/clinician_study_KEY_do_not_send.csv"
K2P = REPO / "results/final/clinician_study_round2_KEY_do_not_send.csv"
for p in (K1P, K2P): assert p.exists(), f"STOP: key file missing: {p}"
K1, K2 = pd.read_csv(K1P), pd.read_csv(K2P)
for nm, K in [("round 1", K1), ("round 2", K2)]:
    miss = {"case_no", "letter", "method"} - set(K.columns)
    assert not miss, f"STOP: {nm} key lacks {miss}. Columns: {list(K.columns)}"
    K["letter"] = K.letter.astype(str).str.strip().str.upper(); K["case_no"] = K.case_no.astype(int)
if "block" in K1.columns:
    p2 = K1.block.astype(str).isin(["2", "B"])
    if K1.loc[p2, "case_no"].max() <= 10: K1.loc[p2, "case_no"] += 10

ANS1 = dict(zip(range(1, 21), "BAAABBAABB" + "ACABBBCAAC"))
RANK2 = dict(zip(range(1, 21), ["ACDB", "BCAD", "DBCA", "ACDB", "CDAB", "BACD", "ABCD", "ABCD", "ADBC", "ADBC",
                                "ACBD", "CDAB", "DCAB", "ADBC", "DBAC", "DBCA", "CDBA", "BCDA", "DABC", "CDAB"]))
def dec(K, case, letter):
    r = K[(K.case_no == case) & (K.letter == letter)]
    assert len(r) == 1, f"STOP: key has {len(r)} rows for case {case}, letter {letter}"
    return r.iloc[0]

H0 = pd.read_csv(MELNV / "ham_mel_nv_clean.csv"); H0["image_id"] = H0.image_id.astype(str)
H0 = H0.drop_duplicates("image_id").set_index("image_id")
M_EXP0 = models.f2(0.0, 1.0, 0)
def role(method):
    s = method.lower()
    return ("expert" if ("expert" in s or "concept" in s) else ("unaligned" if "unaligned" in s else "global"),
            "FinerCAM" if "finer" in s else ("Diff-CAM" if "diff" in s else "Grad-CAM"))
def get_map(method, iid, gt):
    mdl, cam = role(method); m = {"expert": M_EXP0, "unaligned": M0, "global": M5}[mdl]
    x = data.to_tensor(data.native(H0.loc[iid, "image_rel_path"]))
    if cam == "FinerCAM": return np.asarray(cams.finer_cam(m, x, gt, 0.6), float).reshape(14, 14)
    mel, nv = cams.class_cams(m, x); t, o = (mel, nv) if gt == "MEL" else (nv, mel)
    t, o = np.asarray(t, float).reshape(14, 14), np.asarray(o, float).reshape(14, 14)
    return cams.norm01(np.maximum(t - o, 0)) if cam == "Diff-CAM" else t
def scores(v, L):
    if v[L].std() < 1e-9 or not (0 < L.sum() < L.size): return np.nan, np.nan
    vn = cams.norm01(v); return metrics.lesion_auc(v, L), float(vn[L].sum() / (vn.sum() + 1e-8))
def lesion(iid): return data.to_grid(np.array(Image.open((HAM / str(H0.loc[iid, "mask_rel_path"])).resolve()).convert("L")) > 127)

out = []
# ---------- round 1 ----------
rows1 = []
for c in range(1, 21):
    sub = K1[K1.case_no == c]; ch = dec(K1, c, ANS1[c]); iid = str(ch.image_id)
    gt = str(ch.get("gt", H0.loc[iid, "dx"])).upper().replace("NEVUS", "NV")
    pred = str(ch["pred"]) if "pred" in ch else "n/a"; L = lesion(iid)
    opt = {r.letter: (r.method, *scores(get_map(r.method, iid, gt), L)) for r in sub.itertuples()}
    rows1.append(dict(part=1 if c <= 10 else 2, case=c, image_id=iid, gt=gt, pred=pred, correct=(gt == pred),
                      letter=ANS1[c], chosen=ch.method, k=len(opt),
                      top_loc=ch.method == max(opt.values(), key=lambda t: -np.inf if np.isnan(t[1]) else t[1])[0],
                      top_share=ch.method == max(opt.values(), key=lambda t: -np.inf if np.isnan(t[2]) else t[2])[0],
                      **{f"loc|{m}": a for m, a, b in opt.values()}, **{f"share|{m}": b for m, a, b in opt.values()}))
R1 = pd.DataFrame(rows1); R1.to_csv(TAB / "ch4_clinician_r1_decoded.csv", index=False)
p1, p2_ = R1[R1.part == 1], R1[R1.part == 2]
out.append("COMPOSITION round 1:\n" + R1[["part", "case", "image_id", "gt", "pred", "correct"]].to_string(index=False))
meth1 = sorted(K1[K1.case_no <= 10].method.unique()); glob = [m for m in meth1 if role(m) == ("global", "Grad-CAM")]
assert len(glob) == 1, f"STOP: cannot identify the globally aligned Grad-CAM in part 1 methods {meth1}"
kg = int((p1.chosen == glob[0]).sum()); bt = binomtest(kg, 10, 0.5).pvalue
out.append(f"R1P1 methods {meth1} | global Grad-CAM chosen {kg}/10 (correct pred {int(((p1.chosen == glob[0]) & p1.correct).sum())}/{int(p1.correct.sum())}, "
           f"incorrect pred {int(((p1.chosen == glob[0]) & ~p1.correct).sum())}/{int((~p1.correct).sum())}) | sign test p = {bt:.4f}")
out.append("R1P2 best counts (all | correct | incorrect):\n" + pd.DataFrame({
    "all": p2_.chosen.value_counts(), "correct": p2_[p2_.correct].chosen.value_counts(),
    "incorrect": p2_[~p2_.correct].chosen.value_counts()}).fillna(0).astype(int).to_string())
for nm, d in [("R1P1", p1), ("R1P2", p2_)]:
    out.append(f"POSITION {nm}: {d.letter.value_counts().sort_index().to_dict()} | METRICS {nm}: chosen map has highest lesion localisation "
               f"{int(d.top_loc.sum())}/{len(d)}, highest heat share {int(d.top_share.sum())}/{len(d)} (chance {len(d) / d.k.iloc[0]:.1f})")
    for met in ["loc", "share"]:
        cols = [c for c in d.columns if c.startswith(met + "|")]
        out.append(f"  {nm} median {met}: " + ", ".join(f"{c.split('|')[1]} {d[c].median():.3f}" for c in cols))

# ---------- round 2 ----------
rows2 = []
for c in range(1, 21):
    sub = K2[K2.case_no == c]; rk = RANK2[c]
    roles = {r.letter: role(r.method) for r in sub.itertuples()}
    assert len(set(roles.values())) == 4 and set(rk) == set(roles), f"STOP: round 2 case {c} roles {roles} vs ranking {rk}"
    r0 = sub.iloc[0]; iid = str(r0.image_id); gt = str(r0.get("gt", H0.loc[iid, "dx"])).upper().replace("NEVUS", "NV"); L = lesion(iid)
    rec = dict(case=c, image_id=iid, gt=gt, pred=str(r0["pred"]) if "pred" in sub.columns else "n/a",
               best=" ".join(roles[rk[0]]), letter=rk[0])
    sc = {}
    for r in sub.itertuples():
        nm = " ".join(roles[r.letter]); rec[f"rank|{nm}"] = rk.index(r.letter) + 1
        sc[nm] = scores(get_map(r.method, iid, gt), L); rec[f"loc|{nm}"], rec[f"share|{nm}"] = sc[nm]
    rec["top_loc"] = rec["best"] == max(sc, key=lambda k: -np.inf if np.isnan(sc[k][0]) else sc[k][0])
    rec["top_share"] = rec["best"] == max(sc, key=lambda k: -np.inf if np.isnan(sc[k][1]) else sc[k][1])
    rows2.append(rec)
R2 = pd.DataFrame(rows2); R2.to_csv(TAB / "ch4_clinician_r2_decoded.csv", index=False)
if "pred" in K2.columns:
    out.append(f"COMPOSITION round 2: gt {R2['gt'].value_counts().to_dict()} | correct pred {int((R2['gt'] == R2['pred']).sum())}/20")
else:
    out.append(f"COMPOSITION round 2 (no pred column, global model p_mel>=0.5): gt {R2['gt'].value_counts().to_dict()} | "
               f"predicted MEL {sum(v >= 0.5 for v in pr.values())}/20")
RC = [c for c in R2.columns if c.startswith("rank|")]
out.append("R2 best counts: " + repr(R2["best"].value_counts().to_dict()))
out.append("R2 mean rank: " + ", ".join(f"{c.split('|')[1]} {R2[c].mean():.2f}" for c in RC))
PAIRS = [("expert Grad-CAM", "global Grad-CAM"), ("expert FinerCAM", "global FinerCAM"),
         ("global FinerCAM", "global Grad-CAM"), ("expert FinerCAM", "expert Grad-CAM")]
pw = []
for a, b in PAIRS:
    k = int((R2[f"rank|{a}"] < R2[f"rank|{b}"]).sum()); p = binomtest(k, 20, 0.5).pvalue; pw.append((a, b, k, p))
    out.append(f"R2 PAIR {a} over {b}: {k}/20, sign test p = {p:.4f} (Bonferroni threshold 0.0125)")
out.append(f"POSITION R2 best: {R2['letter'].value_counts().sort_index().to_dict()} | METRICS R2: best map has highest lesion localisation "
           f"{int(R2['top_loc'].sum())}/20, highest heat share {int(R2['top_share'].sum())}/20 (chance 5.0)")
for met in ["loc", "share"]:
    out.append(f"  R2 median {met}: " + ", ".join(f"{c.split('|')[1]} {R2[c].median():.3f}" for c in R2.columns if c.startswith(met + "|")))

ST = pd.read_csv(TAB / "appD_statistics.csv"); ST = ST[~ST.comparison.str.startswith("clinician preference")]
new = [dict(chapter=4, comparison="clinician preference, round 1, globally aligned vs unaligned Grad-CAM", metric="cases chosen",
            n=10, median_diff=kg / 10, p=bt, family=1, test="sign test", source="ch4_clinician_r1_decoded.csv")]
new += [dict(chapter=4, comparison=f"clinician preference, round 2, {a} ranked above {b}", metric="cases ranked higher",
             n=20, median_diff=k / 20, p=p, family=4, test="sign test", source="ch4_clinician_r2_decoded.csv") for a, b, k, p in pw]
ST = pd.concat([ST, pd.DataFrame(new)], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
def fmt_p(v):
    if pd.isna(v): return "--"
    if v < 1e-3:
        m, e = f"{v:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
fp = fmt_p
ST = pd.read_csv(TAB / "appD_statistics.csv")
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fmt_p(r.p), fi(r.family), fmt_p(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. The Bonferroni threshold is 0.05 divided by the family size.")
new5 = ST[ST.comparison.str.startswith("clinician preference")]
out.append(f"APPD: {len(ST)} tests, surviving {int((ST.survives == 'yes').sum())}, depth present {int(ST.comparison.str.startswith('depth').sum())}, "
           f"clinician rows {len(new5)}, surviving {int((new5.survives == 'yes').sum())}")
print("=== RESULTS ===\n" + "\n".join(out) + "\n=== END ===")


# ---- cell 140 ------------------------------------------------------------
for f in ["K2", "R2", "H0", "role", "scores", "lesion"]:
    assert f in globals(), f"STOP: {f} missing. Rerun the clinician decoding cell first."
EP9 = {"global": (REPO / "external/checkpoints_f2/ha1only_ep9/fold0.pth", 5.0, 0.0),
       "expert": (REPO / "external/checkpoints_f2/ha2b1_ep9/fold0.pth", 0.0, 1.0)}
MF = {}
for k, (p, hl, fl) in EP9.items():
    assert p.exists(), f"STOP: missing {p}"
    ck = torch.load(p, map_location="cpu", weights_only=False); a = models.ckpt_args(p)
    ok = (int(ck.get("epoch")) == 9 and int(a["fold"]) == 0 and float(a["ha_lambda"]) == hl
          and float(a.get("feat_lambda") or 0) == fl)
    assert ok, f"STOP: {k} args epoch {ck.get('epoch')}, fold {a.get('fold')}, ha {a.get('ha_lambda')}, feat {a.get('feat_lambda')}"
    MF[k] = models.load_ckpt(p, cache=False)
print("final epoch checkpoints verified")

def get_map_fe(method, iid):
    mdl, cam = role(method); m = MF["expert" if mdl == "expert" else "global"]
    x = data.to_tensor(data.native(H0.loc[iid, "image_rel_path"]))
    if cam == "FinerCAM": return np.asarray(cams.finer_cam(m, x, "MEL", 0.6), float).reshape(14, 14)
    return np.asarray(cams.class_cams(m, x)[0], float).reshape(14, 14)

rows = []
for c in range(1, 21):
    sub = K2[K2.case_no == c]; iid = str(sub.iloc[0].image_id); L = lesion(iid)
    best = R2.loc[R2.case == c, "best"].iloc[0]
    sc = {" ".join(role(r.method)): scores(get_map_fe(r.method, iid), L) for r in sub.itertuples()}
    rows.append({"case": c, "image_id": iid, "best": best,
                 **{f"loc|{k}": v[0] for k, v in sc.items()}, **{f"share|{k}": v[1] for k, v in sc.items()},
                 "top_loc": best == max(sc, key=lambda k: sc[k][0]), "top_share": best == max(sc, key=lambda k: sc[k][1])})
F2 = pd.DataFrame(rows); F2.to_csv(TAB / "ch4_clinician_r2_metrics_final_epoch.csv", index=False)
md = {k: (F2[f"loc|{k}"].median(), F2[f"share|{k}"].median())
      for k in ["expert Grad-CAM", "expert FinerCAM", "global Grad-CAM", "global FinerCAM"]}
eL, eS = md["expert Grad-CAM"]; gL, gS = md["global Grad-CAM"]; tl, ts = int(F2.top_loc.sum()), int(F2.top_share.sum())
if eL < gL and eS > gS:
    s51 = (f"In round 2 she preferred the expert aligned maps, which localise the lesion less well than the globally aligned maps "
           f"(median {eL:.3f} against {gL:.3f} for Grad-CAM) but place more of their heat inside it ({eS:.3f} against {gS:.3f}). "
           f"The map she chose best had the highest heat share in {ts} of 20 cases and the highest lesion localisation in {tl}. "
           "This fits the plateau of the globally aligned map (Section 4.3): a map that covers the whole lesion evenly localises it well but marks no region inside it.")
else:
    s51 = (f"In round 2 she preferred the expert aligned maps. Their median lesion localisation is {eL:.3f} against {gL:.3f} for the globally "
           f"aligned Grad-CAM, and their heat share inside the lesion {eS:.3f} against {gS:.3f}. The map she chose best had the highest heat "
           f"share in {ts} of 20 cases and the highest lesion localisation in {tl}, so neither measure accounts for her ranking.")
r1 = ("In round 1 she preferred the map that localised the lesion better in all 10 Part 1 cases and in 8 of 10 Part 2 cases. ")
print(f"""=== PROMPT ===
Correction to the round 2 preference results. This replaces the round 2 table values, the caption sentence on lesion metrics, and the Section 5.1 paragraph from the previous messages. Plain sentences without antithesis constructions. Source: ch4_clinician_r2_metrics_final_epoch.csv.

1. tab:r-clinician-r2. Keep the chosen best and mean rank columns. Replace the two lesion metric columns:
expert aligned | Grad-CAM | 11 | 1.50 | {md['expert Grad-CAM'][0]:.3f} | {md['expert Grad-CAM'][1]:.3f}
expert aligned | FinerCAM | 6 | 2.05 | {md['expert FinerCAM'][0]:.3f} | {md['expert FinerCAM'][1]:.3f}
globally aligned | Grad-CAM | 3 | 2.55 | {md['global Grad-CAM'][0]:.3f} | {md['global Grad-CAM'][1]:.3f}
globally aligned | FinerCAM | 0 | 3.90 | {md['global FinerCAM'][0]:.3f} | {md['global FinerCAM'][1]:.3f}
Caption sentence on lesion metrics, replacing any earlier version and any [VERIFY]: "Lesion metrics are computed on the full maps of the fold 0 models at the final epoch, the checkpoints shown to the dermatologist (global only, lambda_global 5, and expert only, lambda_expert 1). They are descriptive and not tested."

2. Section 5.1, preference study paragraph. Replace with:
"The dermatologist's choices follow lesion localisation in round 1. {r1}{s51} Her ranking of FinerCAM below Grad-CAM matches its lower agreement with the annotated structures on melanomas (Section 4.4). It differs from the deletion result, where only FinerCAM on the expert aligned model separated from random lesion cells (Section 4.6). A preference shows which explanation this expert found convincing. It does not show which one reflects the model's evidence."

3. Remove the earlier instruction that added "(fold 0 model, Section 4.5)" to Section 5.1.
=== END ===""")


# ---- cell 141 ------------------------------------------------------------
import cv2, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import distance_transform_edt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import ConnectionPatch
from matplotlib.figure import Figure
from src.thesis import plotting
from src.thesis.config import BLOCK, MEL_IDX, NV_IDX
from src.thesis.plotting import overlay, C_LESION, TEXTWIDTH
if hasattr(Figure, "_orig_savefig"): Figure.savefig = Figure._orig_savefig
if "_save0" in globals(): plotting.save = _save0
save = plotting.save
need = ["EV", "HUM", "LESION", "TEST_DF", "HAM", "M0", "M5", "TAB", "SEED", "CK5", "CK_ALPHA", "models", "cams", "data", "metrics"]
assert not [n for n in need if n not in globals()], f"run the 16g header and cell 0 first"
FOLD = data.fold_of_heldout(); held = [i for i in sorted(HUM) if i in FOLD]
C_MEL, C_NV, C_GLOB, C_REF, C_SMEL = plotting.C_MEL, plotting.C_NV, plotting.C_GLOB, plotting.C_REF, "#FF00FF"

_outline0 = getattr(plotting, "_outline0", plotting.outline); plotting._outline0 = _outline0
def outline(ax, m, color=None, *a, **kw):
    ls = kw.get("ls", kw.get("linestyle", "solid"))
    if color is not None and str(color).lower() in ("#ffffff", "white", "w") and ls in ("solid", "-"):
        color = C_SMEL                     # melanoma structures: magenta solid
    return _outline0(ax, m, color, *a, **kw)
plotting.outline = outline
print("setup ok | outlines: melanoma structures magenta solid, benign structures white dashed | C_REF =", C_REF)


# ---- cell 142 ------------------------------------------------------------
r = EV.loc[IID]; orig = np.array(Image.open((HAM / str(r.image_rel_path)).resolve()).convert("RGB"))
H, W = orig.shape[:2]; rgb = data.rgb224(r.image_rel_path); L = LESION[IID]
mel, nv = cams.class_cams(M5, data.to_tensor(data.native(r.image_rel_path)))
cam = cams.norm01(np.asarray(mel, float).reshape(14, 14))
fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH * 0.85, TEXTWIDTH * 0.31))
ax[0].imshow(rgb)
for i in range(1, 14):
    ax[0].axhline(i * 16 - 0.5, color="white", lw=0.4, alpha=0.7); ax[0].axvline(i * 16 - 0.5, color="white", lw=0.4, alpha=0.7)
ax[0].set_title("Input, 14 x 14 patch grid", fontsize=8)
ax[1].imshow(cam, cmap="jet", vmin=0, vmax=1, interpolation="nearest"); ax[1].set_title("Map at native resolution", fontsize=8)
overlay(ax[2], rgb, cam); outline(ax[2], L, C_LESION, lw=0.9); ax[2].set_title("Overlay", fontsize=8)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); save(fig, "fig_ch3_pipeline"); plt.show()
print(f"original {W} x {H}, one patch covers {W / 14:.1f} x {H / 14:.1f} original pixels")


# ---- cell 143 ------------------------------------------------------------
FR = np.round(np.arange(0, 1.0001, 0.05), 2); SEEDS = [SEED + s for s in range(5)]
FILLS = ["mean skin", "grey", "black"]
def mask224(rel):
    m = np.array(Image.open((HAM / str(rel)).resolve()).convert("L")) > 127
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
def fill_cells(img, cells, fill, m224):
    mk = np.kron(cells.astype(np.uint8), np.ones((16, 16), np.uint8)).astype(bool)
    if not mk.any(): return img
    if fill == "inpaint": return cv2.inpaint(img, mk.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    if fill == "mean skin": col = img[~m224].mean(0) if (~m224).any() else img.reshape(-1, 3).mean(0)
    else: col = np.full(3, 128.0 if fill == "grey" else 0.0)
    out = img.copy(); out[mk] = np.clip(np.round(col), 0, 255).astype(img.dtype); return out
def gaps(m, imgs, gt, bs=42):
    a, b = (MEL_IDX, NV_IDX) if gt == "MEL" else (NV_IDX, MEL_IDX); out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            o = m(torch.cat([data.to_tensor(x) for x in imgs[i:i + bs]]))
            l = o["logits"] if isinstance(o, dict) else (o[0] if isinstance(o, (tuple, list)) else o)
            out.append((l[:, a] - l[:, b]).float().cpu().numpy())
    return np.concatenate(out)
def frames(img, order, fill, m224):
    res = []
    for f in FR:
        c = np.zeros(196, bool); c[order[:int(round(f * len(order)))]] = True
        res.append(fill_cells(img, c.reshape(14, 14), fill, m224))
    return res
def drops(m, imgs, gt, base): return (base - gaps(m, imgs, gt)) / max(abs(base), 0.1)

rows, skipped = [], []
for nm in ["unaligned", "global aligned", "expert aligned"]:
    for k in range(5):
        mk = M0 if nm == "unaligned" else (M5 if nm == "global aligned" else models.f2(0.0, 1.0, k))
        for iid in [i for i in held if FOLD[i] == k]:
            r = EV.loc[iid]; img = data.rgb224(r.image_rel_path); m224 = mask224(r.mask_rel_path)
            L = LESION[iid]; gt = r.gt_label; lidx = np.flatnonzero(L)
            t = np.asarray(cams.target_ref(mk, data.to_tensor(img), gt)[0], float).ravel()
            if t[lidx].max() < 1e-6: skipped.append((nm, iid)); continue
            base = float(gaps(mk, [img], gt)[0]); rng = np.random.default_rng(SEED)
            o_map = lidx[np.lexsort((rng.random(len(lidx)), -t[lidx]))]
            D = distance_transform_edt(L); dn = D.ravel() / D[L].max()
            o_dep = lidx[np.lexsort((rng.random(len(lidx)), dn[lidx]))]          # rim first
            perms = [np.random.default_rng(s).permutation(lidx) for s in SEEDS]
            for fill in FILLS:
                dm = drops(mk, frames(img, o_map, fill, m224), gt, base)
                rnd = drops(mk, sum([frames(img, p, fill, m224) for p in perms], []), gt, base).reshape(len(SEEDS), len(FR)).mean(0)
                rows += [dict(model=nm, image_id=iid, fill=fill, order=o, frac=f, drop=v)
                         for o, vals in [("Grad-CAM", dm), ("random", rnd)] for f, v in zip(FR, vals)]
            dd = drops(mk, frames(img, o_dep, "inpaint", m224), gt, base)
            rows += [dict(model=nm, image_id=iid, fill="inpaint", order="depth", frac=f, drop=v) for f, v in zip(FR, dd)]
        if nm == "expert aligned": del mk
    pd.DataFrame(rows).to_csv(TAB / "ch4_deletion_fills.csv", index=False); print(nm, "done |", len(rows), "rows")
print("skipped, Grad-CAM empty in lesion:", skipped)


# ---- cell 144 ------------------------------------------------------------
for f in ["write", "tx", "fi", "f3", "METN"]: assert f in globals(), f"run the statistics export cell first ({f})"
def fmt_p(v):
    if pd.isna(v): return "--"
    if v < 1e-3: m, e = f"{v:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
def area(g): g = g.sort_values("frac"); return np.trapz(g["drop"], g["frac"])
MOD = ["unaligned", "global aligned", "expert aligned"]; FILLS = ["mean skin", "grey", "black"]
DF = pd.read_csv(TAB / "ch4_deletion_fills.csv")
DM0 = pd.read_csv(TAB / "ch5_deletion_curves_methods.csv"); DC0 = pd.read_csv(TAB / "ch5_deletion_curves.csv")
cols = ["model", "image_id", "fill", "order", "frac", "drop"]
inp = pd.concat([DM0[DM0.method == "Grad-CAM"].assign(fill="inpaint", order="Grad-CAM")[cols],
                 DC0.rename(columns={"random": "drop"}).assign(fill="inpaint", order="random")[cols]])
A = pd.concat([DF, inp], ignore_index=True); A["frac"] = A.frac.round(2); A["image_id"] = A.image_id.astype(str)
AR = A.groupby(["model", "fill", "order", "image_id"]).apply(area).rename("area").reset_index()
S10 = A[A.frac == 0.10]
def test(df, val, fill, model, a, b):
    q = df[(df.fill == fill) & (df.model == model)].pivot_table(index="image_id", columns="order", values=val).dropna(subset=[a, b])
    return metrics.paired(q[a], q[b])
res = []
for fill in ["inpaint"] + FILLS:
    for nm in MOD:
        s, c = test(S10, "drop", fill, nm, "Grad-CAM", "random"), test(AR, "area", fill, nm, "Grad-CAM", "random")
        res.append(dict(fill=fill, model=nm, step_diff=s["diff"], step_p=s["p"], n=s["n"], area_diff=c["diff"], area_p=c["p"]))
RES = pd.DataFrame(res); TH = 0.05 / 3
RES["step_sig"], RES["area_sig"] = RES.step_p < TH, RES.area_p < TH
RES.to_csv(TAB / "ch4_deletion_fills_tests.csv", index=False)
dep = []
for nm in MOD:
    for b in ["random", "Grad-CAM"]:
        c = test(AR, "area", "inpaint", nm, "depth", b); dep.append(dict(model=nm, vs=b, n=c["n"], diff=c["diff"], p=c["p"]))
DEP = pd.DataFrame(dep); DEP["sig"] = DEP.p < 0.05 / 6

pat = lambda f: [tuple(r) for r in RES[RES.fill == f].set_index("model").loc[MOD, ["step_sig", "area_sig"]].values]
holds = {f: pat(f) == pat("inpaint") for f in FILLS}
line = lambda r: (f"{r.model}: top 10 percent {r.step_diff:+.3f} (p = {r.step_p:.2g}), curve area {r.area_diff:+.3f} "
                  f"(p = {r.area_p:.2g}), n {r.n}")
txt = []
for f in FILLS:
    sub = RES[RES.fill == f].set_index("model").loc[MOD]
    ch = [m for m, a, b in zip(MOD, pat(f), pat("inpaint")) if a != b]
    txt.append(f"With {f} filling: " + "; ".join(line(r) for r in sub.reset_index().itertuples()) + ". " +
               ("The conclusions of the inpainting result hold for all three models." if holds[f] else
                f"The conclusion changes for the {', '.join(ch)} model, where significance after correction differs from inpainting."))
where = "Appendix F" if all(holds.values()) else "Section 4.6"

MK = {"inpaint": None, "mean skin": "o", "grey": "s", "black": "^"}
fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5), sharey=True)
for a, nm in zip(ax, MOD):
    col = plotting.MODEL_COLOR[nm]; d = A[A.model == nm]
    keep = set.intersection(*[set(d[(d.fill == f) & (d.order == o)].image_id) for f in MK for o in ["Grad-CAM", "random"]])
    d = d[d.image_id.isin(keep)]
    for f, mkr in MK.items():
        for o, ls, al in [("Grad-CAM", "-", 1.0), ("random", "--", 0.55)]:
            g = d[(d.fill == f) & (d.order == o)].groupby("frac")["drop"].median()
            a.plot(g.index * 100, g.values, ls=ls, color=col, alpha=al, lw=1.1, marker=mkr, ms=2.8, markevery=4)
    a.axhline(0, color=plotting.C_CHANCE, ls=":", lw=0.8)
    a.set_title(f"{nm} (n = {len(keep)})", fontsize=8); a.set_xlabel("lesion cells removed (%)", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("normalised logit gap drop", fontsize=8)
ax[0].legend([Line2D([], [], color="0.3", marker=m, ms=3) for m in MK.values()] + [Line2D([], [], color="0.3"), Line2D([], [], color="0.3", ls="--")],
             ["inpainting", "mean skin colour", "grey (128)", "black (0)", "Grad-CAM order", "random order"],
             frameon=False, fontsize=6, ncol=2, loc="upper left")
fig.tight_layout(); save(fig, "fig_ch4_deletion_fills"); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharey=True)
for a, nm in zip(ax, MOD):
    piv = DM0[DM0.model == nm].pivot_table(index="image_id", columns="method", values="drop", aggfunc="size").dropna()
    keep = set(piv.index.astype(str)) & set(A[(A.model == nm) & (A.order == "depth")].image_id)
    for meth in ["Grad-CAM", "FinerCAM 0.6"]:
        g = DM0[(DM0.model == nm) & (DM0.method == meth) & DM0.image_id.astype(str).isin(keep)].groupby("frac")["drop"]
        a.plot(g.median().index * 100, g.median().values, "-", color=plotting.METHOD_COLOR[meth], lw=1.4)
        a.fill_between(g.median().index * 100, g.quantile(.25).values, g.quantile(.75).values, color=plotting.METHOD_COLOR[meth], alpha=0.12)
    g = DC0[(DC0.model == nm) & DC0.image_id.astype(str).isin(keep)].groupby("frac")["random"].median()
    a.plot(g.index * 100, g.values, "--", color="0.55", lw=1.3)
    g = A[(A.model == nm) & (A.order == "depth") & A.image_id.isin(keep)].groupby("frac")["drop"].median()
    a.plot(g.index * 100, g.values, "-.", color=C_REF, lw=1.3)
    a.axhline(0, color=plotting.C_CHANCE, ls=":", lw=0.8)
    a.set_title(f"{nm} (n = {len(keep)})", fontsize=8); a.set_xlabel("lesion cells removed (%)", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("normalised logit gap drop", fontsize=8)
ax[0].legend([Line2D([], [], color=plotting.METHOD_COLOR["Grad-CAM"]), Line2D([], [], color=plotting.METHOD_COLOR["FinerCAM 0.6"]),
              Line2D([], [], color="0.55", ls="--"), Line2D([], [], color=C_REF, ls="-.")],
             ["Grad-CAM order", r"FinerCAM order ($\gamma=0.6$)", "random order", "rim inward"], frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(); save(fig, "fig_appC_deletion_curves"); plt.show()

ST = pd.read_csv(TAB / "appD_statistics.csv")
ST = ST[~ST.comparison.str.startswith(("deletion fill", "deletion depth order"))]
new = []
for r in RES[RES.fill != "inpaint"].itertuples():
    new += [dict(chapter=4, comparison=f"deletion fill {r.fill}, top 10% vs random, {r.model}", metric="logit gap drop", n=r.n,
                 median_diff=r.step_diff, p=r.step_p, family=3, test="Wilcoxon", source="ch4_deletion_fills_tests.csv"),
            dict(chapter=4, comparison=f"deletion fill {r.fill}, curve Grad-CAM vs random, {r.model}", metric="area under drop curve", n=r.n,
                 median_diff=r.area_diff, p=r.area_p, family=3, test="Wilcoxon", source="ch4_deletion_fills_tests.csv")]
new += [dict(chapter=4, comparison=f"deletion depth order, rim inward vs {r.vs}, {r.model}", metric="area under drop curve", n=r.n,
             median_diff=r.diff, p=r.p, family=6, test="Wilcoxon", source="ch4_deletion_fills.csv") for r in DEP.itertuples()]
ST = pd.concat([ST, pd.DataFrame(new)], ignore_index=True)
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fmt_p(r.p), fi(r.family), fmt_p(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. Preference tests are two-sided sign tests. The Bonferroni threshold is 0.05 divided by the family size.")

inr = RES[RES.fill == "inpaint"].set_index("model").loc[MOD].reset_index()
dtx = "; ".join(f"{r.model}: rim inward vs {r.vs} {r.diff:+.3f} (p = {r.p:.2g})" for r in DEP.itertuples())
print(f"""=== PROMPT DELETION ===
Deletion with other fills and the rim inward order. Plain sentences without antithesis constructions. Sources: ch4_deletion_fills.csv, ch4_deletion_fills_tests.csv.

1. Section 3.8, after the deletion curve description: "As a robustness check, removed cells were also filled with the mean colour of the image outside the lesion, with constant grey (128) or with black (0). For comparability across fills, the single step test is read from the curves at 10 percent of lesion cells against the mean of five random orders. A further order removes lesion cells from the rim inward, by normalised depth, with ties broken at random."

2. Section 4.6 (or {where}), fills. Grad-CAM order against random order, paired Wilcoxon, family 3 per fill and measure (threshold 0.0167).
Inpainting under the same rule: {'; '.join(line(r) for r in inr.itertuples())}.
{chr(10).join(txt)}
Figure fig_ch4_deletion_fills.pdf goes to {where}, label fig:deletion-fills. Caption: "Deletion curves of the Grad-CAM order (solid) and random order (dashed, mean of five seeds) under four fills, median over the held-out images, one panel per model. No marker: inpainting. Circle: mean skin colour. Square: grey (128). Triangle: black (0). Grey: unaligned. Orange: globally aligned. Purple: expert aligned."
Appendix C table, label tab:app-deletion-fills, columns: fill | model | n | top 10 percent minus random | p | curve area difference | p, with the rows above.

3. Section 4.6, rim inward order (inpainting, family 6, threshold 0.0083): {dtx}. Add one sentence stating, for each model, whether removing cells from the rim inward lowers the prediction more than random order or more than Grad-CAM order, using these values only.
Update the caption of fig:app-deletion-curves: "Brown dash-dot: cells removed from the lesion rim inward." Keep the rest.

4. Appendix D: appD_statistics.tex was re-exported. It now lists {len(ST)} tests, {int((ST.survives == 'yes').sum())} of which survive correction (18 fill tests, family 3, and 6 rim inward tests, family 6, added).
=== END ===""")


# ---- cell 145 ------------------------------------------------------------
def head_maps(model, x):
    with torch.enable_grad():
        base, logits = cams._forward(model, x); A_ = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A_, retain_graph=True)[0]; Gn = torch.autograd.grad(logits[0, NV_IDX], A_)[0]
    base.clear_xai_state()
    A_, Gm, Gn = [T_[0, :, 1:, 1:].detach().float().cpu().numpy() for T_ in (A_, Gm, Gn)]
    return (A_ * Gm).sum(1).reshape(-1, 14, 14), (A_ * Gn).sum(1).reshape(-1, 14, 14)
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv"); mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
R = (LAMC[LAMC.image_id.isin(mel_ids) & LAMC["lambda"].isin([0.0, 0.1, 1.0, 5.0, 17.0])]
     .pivot_table(index="image_id", columns="lambda", values="corr_in").dropna())
IID = (R - R.median()).abs().sum(axis=1).idxmin()
x = data.to_tensor(data.native(EV.loc[IID].image_rel_path)); L = LESION[IID]
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.85")
fig, ax = plt.subplots(2, 2, figsize=(TEXTWIDTH * 0.6, TEXTWIDTH * 0.68)); VAL = {}
for i, (lab, m) in enumerate([("Unaligned", M0), ("Globally aligned", M5)]):
    Hm, Hn = head_maps(m, x); pre = [Hm.mean(0), Hn.mean(0)]
    mel, nv = cams.class_cams(m, x)
    VAL[lab] = (metrics.corr_in(mel, nv, L), float(np.corrcoef(pre[0][L], pre[1][L])[0, 1]))
    for j, v in enumerate(pre):
        s = np.abs(v[L]).max()
        ax[i, j].imshow(np.where(L, v, np.nan), cmap=cmap, norm=TwoSlopeNorm(0, -s, s), interpolation="nearest")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[i, 0].set_ylabel(lab, fontsize=8, color="black")
ax[0, 0].set_title("Melanoma map", fontsize=8, color=C_MEL); ax[0, 1].set_title("Nevus map", fontsize=8, color=C_NV)
fig.subplots_adjust(left=0.1, right=0.8, top=0.93, bottom=0.07, wspace=0.05, hspace=0.3)
for i, lab in enumerate(VAL):
    p0, p1 = ax[i, 0].get_position(), ax[i, 1].get_position()
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.035, f"$r_{{\\mathrm{{in}}}}$ = {VAL[lab][0]:.2f}", ha="center", fontsize=8)
cb = fig.colorbar(ScalarMappable(norm=TwoSlopeNorm(0, -1, 1), cmap=cmap), cax=fig.add_axes([0.84, 0.25, 0.03, 0.5]), ticks=[-1, 0, 1])
cb.ax.set_yticklabels(["negative", "0", "positive"], fontsize=7)
save(fig, "fig_ch4_correlation_example"); plt.show()
u, g = VAL["Unaligned"], VAL["Globally aligned"]
print(f"""=== PROMPT CORRELATION ===
Section 4.2, add fig_ch4_correlation_example.pdf, label fig:correlation-example, next to the stage table.
Caption: "Melanoma map (left) and nevus map (right) of one annotated melanoma, the sweep example of Figure~\\ref{{fig:sweep-examples}}, before the final rectification, on the 14 x 14 grid. Only lesion cells are coloured, other cells are grey. Cell colour shows the sign of the value: red positive, blue negative, each map scaled symmetrically around zero. Top: unaligned. Bottom: globally aligned. Below each pair: r_in of the final maps."
Text: "Figure~\\ref{{fig:correlation-example}} shows the collapse at the cell level. Without alignment the two maps carry opposite signs over much of the lesion (r_in {u[0]:.2f}, correlation before rectification {u[1]:.2f}). With global alignment both maps are positive over nearly the same cells (r_in {g[0]:.2f}, before rectification {g[1]:.2f})." [VERIFY: check that the figure shows this before keeping the second and third sentence]
=== END ===""")


# ---- cell 146 ------------------------------------------------------------
SW = pd.read_csv(TAB / "ch5_lambda_sweep.csv"); LC = pd.read_csv(TAB / "ch5_lambda_localisation.csv")
S = SW.merge(LC, on="lambda").sort_values("lambda"); z = S[S["lambda"] == 0].iloc[0]; P = S[S["lambda"] > 0]
fig = plt.figure(figsize=(TEXTWIDTH * 0.85, 2.8)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 9], wspace=0.05)
a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharey=a0); b0, b1 = a0.twinx(), a1.twinx()
ymax = max(45, S.ap_drop_pct.max() * 1.1)
SER = [(a0, a1, "loc_auc_median", C_GLOB, "-", "o"), (b0, b1, "ap_drop_pct", C_REF, "-", "s"), (b0, b1, "auc_drop_pct", C_REF, "--", "^")]
for l, r, c, col, ls, mk in SER:
    l.plot([0], [z[c]], marker=mk, color=col, ms=3)
    r.plot(P["lambda"], P[c], ls=ls, marker=mk, color=col, ms=3, lw=1.3)
    fig.add_artist(ConnectionPatch(xyA=(0, z[c]), xyB=(0.1, P[c].iloc[0]), coordsA="data", coordsB="data",
                                   axesA=l, axesB=r, color=col, ls=ls, lw=1.3))
for b in (b0, b1): b.set_ylim(0, ymax); b.axhline(10, color=plotting.C_CHANCE, ls=":", lw=1.0)
a0.set_ylim(0, 1); a0.set_xlim(-0.5, 0.5); a0.set_xticks([0]); a0.set_xticklabels(["0"], fontsize=7)
a1.set_xscale("log"); a1.set_xlim(0.08, 22); XT = [0.1, 0.25, 0.5, 1, 3, 5, 11, 17]
a1.set_xticks(XT); a1.set_xticklabels([f"{t:g}" for t in XT], fontsize=7); a1.minorticks_off()
a0.spines["right"].set_visible(False); b0.spines["right"].set_visible(False); b0.set_yticks([])
a1.spines["left"].set_visible(False); b1.spines["left"].set_visible(False); a1.tick_params(left=False, labelleft=False)
for a in (a0, a1, b1): a.tick_params(labelsize=7)
d = 0.015
for a, xs in [(a0, (1 - d * 9, 1 + d * 9)), (a1, (-d, d))]:
    for y in (0, 1): a.plot(xs, (y - d, y + d), transform=a.transAxes, color="k", lw=0.8, clip_on=False)
a0.set_ylabel("lesion localisation (median)", fontsize=8, color=C_GLOB)
b1.set_ylabel(r"drop against $\lambda_{\mathrm{global}}$ = 0 (%)", fontsize=8, color=C_REF)
fig.text(0.5, 0.01, r"$\lambda_{\mathrm{global}}$", ha="center", fontsize=8)
a1.legend([Line2D([], [], color=C_GLOB, marker="o", ms=3), Line2D([], [], color=C_REF, marker="s", ms=3),
           Line2D([], [], color=C_REF, ls="--", marker="^", ms=3)], ["lesion localisation", "AP drop", "AUC drop"],
          frameon=False, fontsize=7, loc="center right")
fig.subplots_adjust(bottom=0.17); save(fig, "fig_ch4_sweep_main"); plt.show()
print("drop line colour:", C_REF)


# ---- cell 147 ------------------------------------------------------------
LAMS = [0.0, 0.1, 1.0, 5.0, 17.0]
ck = lambda l: CK5 / "checkpoint-best-gap-ha0.pth" if l == 0 else \
     CK_ALPHA / f"checkpoint-best-gap-ha{str(l if l % 1 else int(l)).replace('.', 'p')}.pth"
r = EV.loc[IID]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path)); L, U = LESION[IID], HUM[IID]
fig, ax = plt.subplots(2, 5, figsize=(TEXTWIDTH, TEXTWIDTH * 0.44))
for j, lam in enumerate(LAMS):
    m = models.load_ckpt(ck(lam), cache=False); mel, nv = cams.class_cams(m, x); del m
    for i, v in enumerate([mel, nv]):
        overlay(ax[i, j], rgb, v); outline(ax[i, j], L, C_LESION, lw=0.9); outline(ax[i, j], U & L, "#FFFFFF", lw=0.9)
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[0, j].set_title(r"$\lambda_{\mathrm{global}}$ = " + f"{lam:g}", fontsize=8)
ax[0, 0].set_ylabel("Melanoma map", fontsize=8, color=C_MEL); ax[1, 0].set_ylabel("Nevus map", fontsize=8, color=C_NV)
fig.tight_layout(); save(fig, "fig_ch4_sweep_examples"); plt.show()


# ---- cell 148 ------------------------------------------------------------
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
assert "QC" in globals() and plotting.outline is outline, "run the 16g header, cell 0 and Cell A first"
C_EXP, C_JOINT = plotting.C_EXP, plotting.C_JOINT
def numbered(ids):
    n = {"MEL": 0, "NV": 0}; out = []
    for i in ids:
        g = EV.loc[i].gt_label; n[g] += 1; out.append((i, f"{'Melanoma' if g == 'MEL' else 'Nevus'} {n[g]}"))
    return out
def map_head(a, which, fs=8):
    a.set_title("Melanoma map" if which == "MEL" else "Nevus map", fontsize=fs, color=C_MEL if which == "MEL" else C_NV)
def band(fig, ax, c0, c1, label, col, text_y):
    p0, p1 = ax[0, c0].get_position(), ax[-1, c1].get_position(); x0, x1 = p0.x0 - 0.004, p1.x1 + 0.004
    fig.add_artist(mpatches.Rectangle((x0, p1.y0 - 0.005), x1 - x0, 0.997 - p1.y0 + 0.005, transform=fig.transFigure,
                                      facecolor=to_rgba(col, 0.15), edgecolor="none", zorder=-1))
    fig.text((x0 + x1) / 2, text_y, label, ha="center", fontsize=9, fontweight="bold", color=col)
def nv_union(iid):
    acc = None
    for p in QC[(QC.Image_ID == iid) & (QC.label_group == "NV")].mask_path:
        m = data._full(p); acc = m if acc is None else (acc | m)
    return None if acc is None else data.to_grid(acc) & LESION[iid]
def draw_outlines(a, L, U, N, lw=0.8):
    outline(a, L, C_LESION, lw=lw); outline(a, U & L, "#FFFFFF", lw=lw)
    if N is not None and N.any(): outline(a, N, "#FFFFFF", lw=lw, ls="dashed")
def clean(ax):
    for a in np.ravel(ax): a.set_xticks([]); a.set_yticks([])

# methods examples
NVS = [i for i in sorted(HUM) if EV.loc[i].gt_label == "NV" and nv_union(i) is not None and nv_union(i).any()]
M61 = pd.read_csv(TAB / "ch6_metrics.csv"); gm = M61[M61.method == "Grad-CAM"].set_index("image_id").auc_in
mels = gm[[i for i in gm.index if EV.loc[i].gt_label == "MEL"]]
PICK = mels.iloc[np.argsort((mels - mels.median()).abs().values)[:2]].index.tolist() + NVS[:2]
fig, ax = plt.subplots(4, 4, figsize=(TEXTWIDTH, TEXTWIDTH + 0.5))
for i, (iid, lab) in enumerate(numbered(PICK)):
    r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    L, U, N = LESION[iid], HUM[iid], nv_union(iid)
    mel, nv = cams.class_cams(M5, x); t, o = (mel, nv) if r.gt_label == "MEL" else (nv, mel)
    ax[i, 0].imshow(rgb)
    for j, v in enumerate([t, cams.norm01(np.maximum(t - o, 0)), cams.finer_cam(M5, x, r.gt_label, 0.6)]): overlay(ax[i, j + 1], rgb, v)
    for a in ax[i]: draw_outlines(a, L, U, N)
    ax[i, 0].set_ylabel(lab, fontsize=8, color="black")
clean(ax)
for j, t_ in enumerate(["Image", "Grad-CAM", "Diff-CAM", r"FinerCAM ($\gamma=0.6$)"]): ax[0, j].set_title(t_, fontsize=8)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.01, top=0.88, wspace=0.04, hspace=0.04)
band(fig, ax, 1, 3, "Globally aligned model", C_GLOB, 0.96); save(fig, "fig_ch6_methods_examples"); plt.show()

# channels, landscape
F7 = pd.read_csv(TAB / "ch7_grounding.csv"); e = F7[F7.setting == "HA2 b1"].set_index("image_id").mel_in
PICK7 = []
for c in ["MEL", "NV"]:
    s = e[[i for i in e.index if EV.loc[i].gt_label == c]]; PICK7 += s.iloc[np.argsort((s - s.median()).abs().values)[:2]].index.tolist()
GR = [("Global only", (5.0, 0.0), C_GLOB), (r"Expert only, $\lambda_{\mathrm{expert}}$ 1", (0.0, 1.0), C_EXP),
      (r"Joint, $\lambda_{\mathrm{expert}}$ 5", (5.0, 5.0), C_JOINT)]
fig, ax = plt.subplots(4, 6, figsize=(9.0, 9.0 / 6 * 4 + 0.6))
for i, (iid, lab) in enumerate(numbered(PICK7)):
    r = EV.loc[iid]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
    L, U, N = LESION[iid], HUM[iid], nv_union(iid)
    for g, (_, (a_, b_), _) in enumerate(GR):
        m = models.f2(a_, b_, FOLD[iid]); mel, nv = cams.class_cams(m, x); del m
        for k, v in enumerate([mel, nv]): overlay(ax[i, 2 * g + k], rgb, v); draw_outlines(ax[i, 2 * g + k], L, U, N)
    ax[i, 0].set_ylabel(lab, fontsize=9, color="black")
clean(ax)
for j in range(6): map_head(ax[0, j], "MEL" if j % 2 == 0 else "NV")
fig.subplots_adjust(left=0.04, right=0.995, bottom=0.01, top=0.86, wspace=0.03, hspace=0.03)
for g, (lab, _, col) in enumerate(GR): band(fig, ax, 2 * g, 2 * g + 1, lab, col, 0.955)
save(fig, "fig_ch7_channels"); plt.show()

# contrastive expert, Appendix C
METH = {"Grad-CAM": lambda m, x: cams.target_ref(m, x, "MEL")[0], "Diff-CAM": lambda m, x: cams.diff_cam(m, x, "MEL"),
        "FinerCAM 0.6": lambda m, x: cams.finer_cam(m, x, "MEL", 0.6)}
GRP = [("Globally aligned", (5.0, 0.0), C_GLOB), ("Expert aligned", (0.0, 1.0), C_EXP)]
SHOW = [i for i in held if EV.loc[i].gt_label == "MEL"][:3]
fig, ax = plt.subplots(3, 7, figsize=(TEXTWIDTH, TEXTWIDTH / 7 * 3 + 0.6))
for i, (iid, lab) in enumerate(numbered(SHOW)):
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]; x = data.to_tensor(data.native(r.image_rel_path)); rgb = data.rgb224(r.image_rel_path)
    ax[i, 0].imshow(rgb); draw_outlines(ax[i, 0], L, U, None)
    for g, (_, (a_, b_), _) in enumerate(GRP):
        m = models.f2(a_, b_, FOLD[iid])
        for j, mk in enumerate(METH): overlay(ax[i, 1 + 3 * g + j], rgb, METH[mk](m, x)); draw_outlines(ax[i, 1 + 3 * g + j], L, U, None, lw=0.6)
        del m
    ax[i, 0].set_ylabel(lab, fontsize=7, color="black")
clean(ax); ax[0, 0].set_title("Image", fontsize=8)
for g in range(2):
    for j, t in enumerate(["Grad-CAM", "Diff-CAM", r"FinerCAM ($\gamma=0.6$)"]): ax[0, 1 + 3 * g + j].set_title(t, fontsize=8)
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.01, top=0.80, wspace=0.04, hspace=0.04)
for g, (lab, _, col) in enumerate(GRP): band(fig, ax, 1 + 3 * g, 3 + 3 * g, lab, col, 0.93)
save(fig, "fig_ch7_contrastive_expert"); plt.show()


# ---- cell 149 ------------------------------------------------------------
from scipy.stats import pearsonr
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv"); mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
R = (LAMC[LAMC.image_id.isin(mel_ids) & LAMC["lambda"].isin([0.0, 0.1, 1.0, 5.0, 17.0])]
     .pivot_table(index="image_id", columns="lambda", values="corr_in").dropna())
IID = (R - R.median()).abs().sum(axis=1).idxmin(); r = EV.loc[IID]; L = LESION[IID]

# ---------- 1. resolution figure ----------
orig = Image.open((HAM / str(r.image_rel_path)).resolve()); W, H = orig.size
assert (W, H) == (600, 450), f"STOP: image is {W} x {H}, caption needs {W * 16 / 224:.1f} x {H * 16 / 224:.1f}"
rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
cam = cams.norm01(np.asarray(cams.class_cams(M5, x)[0], float).reshape(14, 14))
fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH * 0.85, TEXTWIDTH * 0.31))
ax[0].imshow(rgb)
for i in range(1, 14):
    ax[0].axhline(i * 16 - 0.5, color="white", lw=0.4, alpha=0.7); ax[0].axvline(i * 16 - 0.5, color="white", lw=0.4, alpha=0.7)
ax[1].imshow(cam, cmap="jet", vmin=0, vmax=1, interpolation="nearest")
overlay(ax[2], rgb, cam); outline(ax[2], L, C_LESION, lw=0.9)
for a, t in zip(ax, ["(a)", "(b)", "(c)"]):
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel(t, fontsize=8)
fig.tight_layout(); save(fig, "fig_ch3_resolution"); plt.show()

ID_T = pd.read_csv(TAB / "app_figure_images.csv"); ID_T = ID_T[ID_T.figure != "resolution"]
ID_T = pd.concat([ID_T, pd.DataFrame([{"figure": "resolution", "panel": "all panels", "image_id": IID}])], ignore_index=True)
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
esc = lambda s: str(s).replace("_", r"\_")
Lt = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
Lt += [f"{esc(q.figure)} & {esc(q.panel)} & {esc(q.image_id)} \\\\" for q in ID_T.itertuples()]
(TAB / "app_figure_images.tex").write_text("\n".join(Lt + [r"\bottomrule", r"\end{tabular}"]) + "\n")

# ---------- 2. correlation check ----------
def head_maps(model, x):
    with torch.enable_grad():
        base, logits = cams._forward(model, x); A_ = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A_, retain_graph=True)[0]; Gn = torch.autograd.grad(logits[0, NV_IDX], A_)[0]
    base.clear_xai_state()
    A_, Gm, Gn = [T_[0, :, 1:, 1:].detach().float().cpu().numpy() for T_ in (A_, Gm, Gn)]
    return (A_ * Gm).sum(1).reshape(-1, 14, 14).mean(0), (A_ * Gn).sum(1).reshape(-1, 14, 14).mean(0)
V = {}
for lab, m in [("unaligned", M0), ("globally aligned", M5)]:
    pm, pn = head_maps(m, x); mel, nv = cams.class_cams(m, x); a, b = pm[L], pn[L]
    V[lab] = dict(r_in=metrics.corr_in(mel, nv, L), r_pre=float(pearsonr(a, b)[0]),
                  opposite=float(((a > 0) != (b > 0)).mean()), both_pos=float(((a > 0) & (b > 0)).mean()), n=int(L.sum()))
CV = pd.DataFrame(V).T; CV.to_csv(TAB / "ch4_correlation_example.csv"); print(CV.round(3).to_string())
u, g = V["unaligned"], V["globally aligned"]
ok_u, ok_g = u["opposite"] >= 0.5, g["both_pos"] >= 0.8
s_u = (f"Without alignment, the two maps carry opposite signs on {u['opposite']:.0%} of lesion cells, with an r_in of {u['r_in']:.2f} "
       f"and a correlation of {u['r_pre']:.2f} before rectification." if ok_u else
       f"Without alignment, the two maps are strongly anticorrelated before rectification ({u['r_pre']:.2f}, r_in {u['r_in']:.2f}), "
       f"although their signs differ on only {u['opposite']:.0%} of lesion cells.")
s_g = (f"With global alignment, both maps are positive on {g['both_pos']:.0%} of lesion cells, with {g['r_in']:.2f} and {g['r_pre']:.2f}." if ok_g else
       f"With global alignment, the two maps are strongly correlated ({g['r_in']:.2f} and {g['r_pre']:.2f}), and both are positive on {g['both_pos']:.0%} of lesion cells.")
print(f"""=== PROMPT ===
Two updates. Plain sentences without antithesis constructions. No image IDs in captions.

1. Section 3.3. Remove fig_ch3_pipeline and the label fig:pipeline if present, and redirect references to fig:resolution. Add fig_ch3_resolution.pdf, label fig:resolution.
Caption: "Resolution of the explanation map for one annotated melanoma, the sweep example of Figure~\\ref{{fig:sweep-examples}}, globally aligned model (lambda_global 5), melanoma map. (a) Input resized to 224 x 224 pixels with the 14 x 14 patch grid. (b) Map at its native resolution, one value per cell. (c) Map upsampled onto the image, lesion outline in green. One cell covers about 43 x 32 pixels of the 600 x 450 original image, so the map cannot resolve structures smaller than a cell. The image ID is listed in Table~\\ref{{tab:app-figure-images}}."
Patch size confirmed: 600 x 16 / 224 = 42.9 and 450 x 16 / 224 = 32.1.
Appendix A: tab:app-figure-images was re-exported with a row for fig:resolution. Replace "resolution" in the figure column with the reference to fig:resolution.

2. Section 4.2, fig:correlation-example. Source: ch4_correlation_example.csv ({u['n']} lesion cells). Replace the two sentences on the figure with:
"{s_u} {s_g}"
Remove the [VERIFY] marker.
=== END ===""")


# ---- cell 150 ------------------------------------------------------------
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
for f in ["write", "tx", "fi", "f3", "METN"]: assert f in globals(), f"run the statistics export cell first ({f})"
assert plotting.outline is outline, "run Cell A of the fills message first"
def fmt_p(v):
    if pd.isna(v): return "--"
    if v < 1e-3: m, e = f"{v:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
def band(fig, ax, c0, c1, label, col, text_y):
    p0, p1 = ax[c0].get_position(), ax[c1].get_position(); x0, x1 = p0.x0 - 0.006, p1.x1 + 0.006
    fig.add_artist(mpatches.Rectangle((x0, p0.y0 - 0.02), x1 - x0, 0.995 - p0.y0 + 0.02, transform=fig.transFigure,
                                      facecolor=to_rgba(col, 0.15), edgecolor="none", zorder=-1))
    fig.text((x0 + x1) / 2, text_y, label, ha="center", fontsize=9, fontweight="bold", color=col)

# ---------- 1. correlation example ----------
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv"); mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
R = (LAMC[LAMC.image_id.isin(mel_ids) & LAMC["lambda"].isin([0.0, 0.1, 1.0, 5.0, 17.0])]
     .pivot_table(index="image_id", columns="lambda", values="corr_in").dropna())
IID = (R - R.median()).abs().sum(axis=1).idxmin(); x = data.to_tensor(data.native(EV.loc[IID].image_rel_path)); L = LESION[IID]
def pre_maps(model):
    with torch.enable_grad():
        base, logits = cams._forward(model, x); A_ = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A_, retain_graph=True)[0]; Gn = torch.autograd.grad(logits[0, NV_IDX], A_)[0]
    base.clear_xai_state()
    A_, Gm, Gn = [T_[0, :, 1:, 1:].detach().float().cpu().numpy() for T_ in (A_, Gm, Gn)]
    return (A_ * Gm).sum(1).reshape(-1, 14, 14).mean(0), (A_ * Gn).sum(1).reshape(-1, 14, 14).mean(0)
COLS6 = ["#08519c", "#6baed6", "#c6dbef", "#fcbba1", "#fb6a4a", "#a50f15"]   # strong, medium, light blue | light, medium, strong red
cmap6 = ListedColormap(COLS6); cmap6.set_bad("#e6e6e6")
norm6 = BoundaryNorm([-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1], cmap6.N)
fig, ax = plt.subplots(1, 4, figsize=(TEXTWIDTH, TEXTWIDTH * 0.36)); RIN = {}
for k, (lab, m) in enumerate([("Unaligned", M0), ("Globally aligned", M5)]):
    pm, pn = pre_maps(m); mel, nv = cams.class_cams(m, x); RIN[lab] = metrics.corr_in(mel, nv, L)
    for j, v in enumerate([pm, pn]):
        s = np.abs(v[L]).max(); a = ax[2 * k + j]
        a.imshow(np.where(L, v / s, np.nan), cmap=cmap6, norm=norm6, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        a.set_title("Melanoma map" if j == 0 else "Nevus map", fontsize=8, color=C_MEL if j == 0 else C_NV)
fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.27, wspace=0.06)
band(fig, ax, 0, 1, "Unaligned", plotting.C_UNAL, 0.93); band(fig, ax, 2, 3, "Globally aligned", C_GLOB, 0.93)
for k, lab in enumerate(RIN):
    p0, p1 = ax[2 * k].get_position(), ax[2 * k + 1].get_position()
    fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.06, f"$r_{{\\mathrm{{in}}}}$ = {RIN[lab]:.2f}", ha="center", fontsize=8)
LEG = ["strong negative", "medium negative", "light negative", "light positive", "medium positive", "strong positive"]
fig.legend([mpatches.Patch(facecolor=c, edgecolor="0.5", lw=0.3) for c in COLS6] + [mpatches.Patch(facecolor="#e6e6e6", edgecolor="0.5", lw=0.3)],
           LEG + ["outside lesion"], loc="lower center", ncol=7, frameon=False, fontsize=6, handlelength=1.2, columnspacing=0.8)
save(fig, "fig_ch4_correlation_example"); plt.show()

# ---------- 2. sweep main ----------
SW = pd.read_csv(TAB / "ch5_lambda_sweep.csv"); LC = pd.read_csv(TAB / "ch5_lambda_localisation.csv")
S = SW.merge(LC, on="lambda").sort_values("lambda"); z = S[S["lambda"] == 0].iloc[0]; P = S[S["lambda"] > 0]
fig = plt.figure(figsize=(TEXTWIDTH * 0.85, 2.8)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 9], wspace=0.05)
a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharey=a0); b0, b1 = a0.twinx(), a1.twinx()
ymax = max(45, S.ap_drop_pct.max() * 1.1)
for l, r_, c, col, ls, mk in [(a0, a1, "loc_auc_median", C_GLOB, "-", "o"), (b0, b1, "ap_drop_pct", C_REF, "-", "s"),
                              (b0, b1, "auc_drop_pct", C_REF, "--", "^")]:
    l.plot([0], [z[c]], marker=mk, color=col, ms=3)
    r_.plot(P["lambda"], P[c], ls=ls, marker=mk, color=col, ms=3, lw=1.3)
    fig.add_artist(ConnectionPatch(xyA=(0, z[c]), xyB=(0.1, P[c].iloc[0]), coordsA="data", coordsB="data",
                                   axesA=l, axesB=r_, color=col, ls=ls, lw=1.3))
a1.axvline(5, color="0.35", ls=":", lw=1.0, zorder=0)
a1.plot([5], [S.loc[S["lambda"] == 5, "loc_auc_median"].iloc[0]], marker="o", ms=8, mfc="none", mec="black", mew=1.1, zorder=5)
for b in (b0, b1): b.set_ylim(0, ymax); b.axhline(10, color=plotting.C_CHANCE, ls=":", lw=1.0)
a0.set_ylim(0, 1); a0.set_xlim(-0.5, 0.5); a0.set_xticks([0]); a0.set_xticklabels(["0"], fontsize=7)
a1.set_xscale("log"); a1.set_xlim(0.08, 22); XT = [0.1, 0.25, 0.5, 1, 3, 5, 11, 17]
a1.set_xticks(XT); a1.set_xticklabels([f"{t:g}" for t in XT], fontsize=7); a1.minorticks_off()
a0.spines["right"].set_visible(False); b0.spines["right"].set_visible(False); b0.set_yticks([])
a1.spines["left"].set_visible(False); b1.spines["left"].set_visible(False); a1.tick_params(left=False, labelleft=False)
for a in (a0, a1, b1): a.tick_params(labelsize=7)
d = 0.012                                            # break marks on the x axis only
a0.plot((1 - d * 9, 1 + d * 9), (-d, d), transform=a0.transAxes, color="k", lw=0.7, clip_on=False)
a1.plot((-d, d), (-d, d), transform=a1.transAxes, color="k", lw=0.7, clip_on=False)
a0.set_ylabel("lesion localisation (median)", fontsize=8, color=C_GLOB)
b1.set_ylabel(r"drop against $\lambda_{\mathrm{global}}$ = 0 (%)", fontsize=8, color=C_REF)
fig.text(0.5, 0.01, r"$\lambda_{\mathrm{global}}$", ha="center", fontsize=8)
b1.legend([Line2D([], [], color=C_GLOB, marker="o", ms=3), Line2D([], [], color=C_REF, marker="s", ms=3),
           Line2D([], [], color=C_REF, ls="--", marker="^", ms=3),
           Line2D([], [], ls="none", marker="o", ms=7, mfc="none", mec="black")],
          ["lesion localisation", "AP drop", "AUC drop", "main model"],
          frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.02, 0.5))
fig.subplots_adjust(bottom=0.17); save(fig, "fig_ch4_sweep_main"); plt.show()
print("drop lines:", C_REF, "(tab:brown), AP solid, AUC dashed")

# ---------- 3. deletion curves without rim inward ----------
DM0 = pd.read_csv(TAB / "ch5_deletion_curves_methods.csv"); DC0 = pd.read_csv(TAB / "ch5_deletion_curves.csv")
DM0["image_id"] = DM0.image_id.astype(str); DC0["image_id"] = DC0.image_id.astype(str)
FRs = np.round(np.arange(0, 1.0001, 0.05), 2)
fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharey=True)
for a, nm in zip(ax, ["unaligned", "global aligned", "expert aligned"]):
    d = DM0[DM0.model == nm]; keep = set(d.pivot_table(index="image_id", columns="method", values="drop", aggfunc="size").dropna().index)
    for meth in ["Grad-CAM", "FinerCAM 0.6"]:
        g = d[(d.method == meth) & d.image_id.isin(keep)].groupby("frac")["drop"]; col = plotting.METHOD_COLOR[meth]
        a.plot(g.median().index * 100, g.median().values, "-", color=col, lw=1.4)
        a.fill_between(g.median().index * 100, g.quantile(.25).values, g.quantile(.75).values, color=col, alpha=0.12)
    g = DC0[(DC0.model == nm) & DC0.image_id.isin(keep)].groupby("frac")["random"].median()
    a.plot(g.index * 100, g.values, "--", color="0.55", lw=1.3)
    a.axhline(0, color=plotting.C_CHANCE, ls=":", lw=0.8)
    a.set_title(f"{nm} (n = {len(keep)})", fontsize=8); a.set_xlabel("lesion cells removed (%)", fontsize=8); a.tick_params(labelsize=7)
ax[0].set_ylabel("normalised logit gap drop", fontsize=8)
ax[0].legend([Line2D([], [], color=plotting.METHOD_COLOR["Grad-CAM"]), Line2D([], [], color=plotting.METHOD_COLOR["FinerCAM 0.6"]),
              Line2D([], [], color="0.55", ls="--")], ["Grad-CAM order", r"FinerCAM order ($\gamma=0.6$)", "random order"],
             frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); save(fig, "fig_appC_deletion_curves"); plt.show()

# ---------- 4. statistics appendix without rim inward ----------
ST = pd.read_csv(TAB / "appD_statistics.csv"); n0, s0 = len(ST), int((ST.survives == "yes").sum())
ST = ST[~ST.comparison.str.startswith("deletion depth order")]
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST.to_csv(TAB / "appD_statistics.csv", index=False)
write("appD_statistics.tex", ["ch.", "comparison", "metric", "n", "median diff.", "p", "family", "threshold", "survives"],
      [[fi(r.chapter), tx(r.comparison), METN.get(r.metric, tx(r.metric)), fi(r.n), f3(r.median_diff),
        fmt_p(r.p), fi(r.family), fmt_p(r.threshold), tx(r.survives)] for _, r in ST.iterrows()],
      "rp{5cm}lrrrrrl", longtable=True, small=True, label="tab:app-stats",
      caption="Every test reported in the results. Paired Wilcoxon signed-rank tests unless stated. Sweep comparisons use one-sided paired bootstrap p values. Preference tests are two-sided sign tests. The Bonferroni threshold is 0.05 divided by the family size.")
NT, NS = len(ST), int((ST.survives == "yes").sum())
print(f"appD: {n0} -> {NT} tests, surviving {s0} -> {NS}")

u, g = RIN["Unaligned"], RIN["Globally aligned"]
print(f"""=== PROMPT ===
Four updates for Chapter 4. Plain sentences without antithesis constructions. No image IDs in captions.

1. fig_ch4_correlation_example.pdf (label fig:correlation-example) was redone as one row of four panels. Replace the caption with:
"Melanoma and nevus maps of one annotated melanoma, the sweep example of Figure~\\ref{{fig:sweep-examples}}, before the final rectification, on the 14 x 14 grid. Grey band: unaligned model. Orange band: globally aligned model. Only lesion cells are coloured, other cells are light grey. Each map is scaled symmetrically around zero and shown in three steps of sign and strength: blue negative, red positive, darker for larger magnitude. Cell colour shows sign, column header colour shows the class. Below each pair: r_in of the final maps."
Keep the Section 4.2 text on this figure (95 percent opposite signs, r_in {u:.2f}; 98 percent both positive, r_in {g:.2f}).

2. fig_ch4_sweep_main.pdf (label fig:sweep-main): the legend moved to the left, the dotted vertical line at lambda_global = 5 and the open circle marking the main model are restored, and the axis break marks sit on the x axis only. Replace the caption with:
"Lesion localisation (orange, left axis) and relative drop in test AP (brown solid) and test AUC (brown dashed, right axis) against lambda_global = 0. The x axis is broken between 0 and 0.1. Dotted vertical line and open circle: the main model at lambda_global = 5. Dotted horizontal line: 10 percent drop."

3. fig_appC_deletion_curves.pdf was regenerated without the rim inward curve. Remove "Brown dash-dot: cells removed from the lesion rim inward." from its caption.
Remove every mention of the rim inward order: the sentence in Section 3.8 ("A further order removes lesion cells from the rim inward ...") and the rim inward paragraph in Section 4.6. Keep the fill results.

4. Appendix D: the six rim inward tests were removed and appD_statistics.tex was re-exported. It now lists {NT} tests, {NS} of which survive correction. Update the test count in the appendix text.
=== END ===""")


# ---- cell 151 ------------------------------------------------------------
assert plotting.outline is outline, "run Cell A of the fills message first"
LD = pd.read_csv(TAB / "ch5_reference_ladder.csv"); LD["image_id"] = LD.image_id.astype(str)
P = LD.pivot_table(index="image_id", columns="ref", values="auc_in")
hm = [i for i in held if EV.loc[i].gt_label == "MEL"]; P = P.loc[[i for i in hm if i in P.index]]
assert len(P) == 14, f"STOP: expected 14 held-out melanomas, found {len(P)}"
med = P[["random", "darkness", "concept only"]].median()
C = P[(P.random >= 0.40) & (P.random <= 0.60)].copy()
assert len(C), "STOP: no held-out melanoma with random AUC_in between 0.40 and 0.60"
C["score"] = (C.darkness - 0.854).abs() + (C["concept only"] - 0.793).abs()
IID = C.score.idxmin(); v = P.loc[IID]

rng = np.random.default_rng(SEED); rand = None
for iid in held:
    d = rng.random(LESION[iid].shape)
    if iid == IID: rand = d
L, U = LESION[IID], HUM[IID]
assert abs(metrics.auc_in(rand, L, U) - v.random) < 1e-6, "STOP: random map does not reproduce the evaluation value"
r = EV.loc[IID]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
dark = 1 - rgb.astype(float).mean(axis=2).reshape(14, 16, 14, 16).mean((1, 3)) / 255
m = models.f2(0.0, 1.0, FOLD[IID]); expert = np.asarray(cams.class_cams(m, x)[0], float).reshape(14, 14); del m
for nm, mp, ref in [("darkness", dark, v.darkness), ("expert", expert, v["concept only"])]:
    assert abs(metrics.auc_in(mp, L, U) - ref) < 1e-3, f"STOP: {nm} map does not reproduce the evaluation value"

MAPS = [("(b)", rand, v.random), ("(c)", dark, v.darkness), ("(d)", expert, v["concept only"])]
fig, ax = plt.subplots(1, 4, figsize=(TEXTWIDTH, TEXTWIDTH / 4 + 0.35))
ax[0].imshow(rgb); outline(ax[0], L, C_LESION, lw=0.9); outline(ax[0], U & L, C_SMEL, lw=0.9); ax[0].set_xlabel("(a)", fontsize=8)
for a, (t, mp, val) in zip(ax[1:], MAPS):
    overlay(a, rgb, cams.norm01(mp)); outline(a, L, C_LESION, lw=0.8)
    a.set_xlabel(f"{t}   " + r"$\mathrm{AUC}_{\mathrm{in}}$" + f" = {val:.2f}", fontsize=8)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); save(fig, "fig_ch3_reference_maps"); plt.show()

ID_T = pd.read_csv(TAB / "app_figure_images.csv"); ID_T = ID_T[ID_T.figure != "reference maps"]
ID_T = pd.concat([ID_T, pd.DataFrame([{"figure": "reference maps", "panel": "all panels", "image_id": IID}])], ignore_index=True)
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
esc = lambda s: str(s).replace("_", r"\_")
Lt = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
Lt += [f"{esc(q.figure)} & {esc(q.panel)} & {esc(q.image_id)} \\\\" for q in ID_T.itertuples()]
(TAB / "app_figure_images.tex").write_text("\n".join(Lt + [r"\bottomrule", r"\end{tabular}"]) + "\n")

print(f"candidates with random in [0.40, 0.60]: {len(C)} of 14")
print(f"chosen: random {v.random:.3f}, darkness {v.darkness:.3f}, expert {v['concept only']:.3f} | fold {FOLD[IID]}")
print(f"medians over 14 held-out melanomas: random {med.random:.3f}, darkness {med.darkness:.3f}, expert {med['concept only']:.3f}")
print(f"""=== PROMPT ===
Section 4.5, fig_ch3_reference_maps.pdf (label fig:reference-maps) was redone with a different image. Move the figure and its text reference from Section 3.7 to Section 4.5 if it is still in 3.7. Plain sentences without antithesis constructions. No image IDs in captions.
Caption: "Reference maps for one held-out annotated melanoma. (a) Image with the lesion outline (green) and the melanoma structures (magenta). (b) Map of uniform random values, the chance reference. (c) Darkness rule: one minus the mean of the red, green and blue channels per 16 x 16 cell of the 224 x 224 input. (d) Melanoma map of the expert aligned model of this image's fold (lambda_global 0, lambda_expert 1), which did not see the image in training. Panels (b) to (d) are shown on the 14 x 14 grid over the image, lesion outline in green. Below each map: in-lesion AUC against the melanoma structures. The image was chosen as the held-out melanoma whose darkness and expert values lie closest to their medians over the 14 held-out melanomas (0.854 and 0.793), among images whose random map scores between 0.40 and 0.60. The image ID is listed in Table~\\ref{{tab:app-figure-images}}."
Text in Section 4.5, after tab:reference: "Figure~\\ref{{fig:reference-maps}} shows the three references on one held-out melanoma (in-lesion AUC {v.random:.2f} for the random map, {v.darkness:.2f} for darkness and {v['concept only']:.2f} for the expert aligned model). Over the 14 held-out melanomas the random map has a median of {med.random:.3f}." Remove any earlier sentence on this figure that mentions the sweep example or the values 0.85 and 0.38.
Appendix A: tab:app-figure-images was re-exported with the new image in the row for fig:reference-maps.
=== END ===""")


# ---- cell 152 ------------------------------------------------------------
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
EXP = REPO / "results/exploratory"; EXP.mkdir(parents=True, exist_ok=True)
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv"); mel_ids = [i for i in HUM if EV.loc[i].gt_label == "MEL"]
R = (LAMC[LAMC.image_id.isin(mel_ids) & LAMC["lambda"].isin([0.0, 0.1, 1.0, 5.0, 17.0])]
     .pivot_table(index="image_id", columns="lambda", values="corr_in").dropna())
SWEEP = (R - R.median()).abs().sum(axis=1).idxmin()
MODS = [("unaligned", M0), ("globally aligned", M5)]

def head_vectors(model, iid):
    x = data.to_tensor(data.native(EV.loc[iid].image_rel_path))
    with torch.enable_grad():
        base, logits = cams._forward(model, x); A_ = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A_, retain_graph=True)[0]; Gn = torch.autograd.grad(logits[0, NV_IDX], A_)[0]
    base.clear_xai_state()
    A_, Gm, Gn = [T_[0, :, 1:, 1:].detach().float().cpu().numpy() for T_ in (A_, Gm, Gn)]   # heads x queries x keys
    k = np.flatnonzero(LESION[iid].ravel())
    return {"gradient": ((Gm.sum(1)[:, k]).T, (Gn.sum(1)[:, k]).T),
            "attention": ((A_.sum(1)[:, k]).T, (A_.sum(1)[:, k]).T),
            "attention x gradient": (((A_ * Gm).sum(1)[:, k]).T, ((A_ * Gn).sum(1)[:, k]).T)}      # cells x 12

def orth_share(vm, vn, heads=None):
    d = vm - vn if heads is None else (vm - vn)[:, heads]
    if not np.abs(d).sum(): return np.nan, 0.0
    u = np.ones(d.shape[1]) / np.sqrt(d.shape[1]); par = (d @ u)[:, None] * u
    rel = (d ** 2).sum() / (0.5 * ((vm ** 2).sum() + (vn ** 2).sum()) + 1e-12)
    return float(((d - par) ** 2).sum() / (d ** 2).sum()), float(rel)

def low3(vm, vn):
    c = [vm[:, h] @ vn[:, h] / (np.linalg.norm(vm[:, h]) * np.linalg.norm(vn[:, h]) + 1e-12) for h in range(12)]
    return list(np.argsort(c)[:3])

# ---------- 1. geometry, sweep example ----------
STEPS = ["gradient", "attention", "attention x gradient"]
V = {nm: head_vectors(m, SWEEP) for nm, m in MODS}
fig, ax = plt.subplots(3, 2, figsize=(TEXTWIDTH * 0.8, TEXTWIDTH * 1.05)); inplane = {}
for i, st in enumerate(STEPS):
    for j, (nm, _) in enumerate(MODS):
        vm, vn = V[nm][st]; X = np.vstack([vm, vn]); p = PCA(3).fit(X); Z = p.transform(X); n = len(vm)
        a = ax[i, j]; a.scatter(Z[:n, 0], Z[:n, 1], s=9, color=C_MEL, alpha=0.75, lw=0, label="melanoma")
        a.scatter(Z[n:, 0], Z[n:, 1], s=9, color=C_NV, alpha=0.75, lw=0, label="nevus")
        u = np.ones(12) / np.sqrt(12); pu = p.components_[:2] @ u; inplane[(st, nm)] = float(np.linalg.norm(pu) ** 2)
        s = np.abs(Z[:, :2]).max(); a.plot([-s * pu[0], s * pu[0]], [-s * pu[1], s * pu[1]], color="black", lw=1.1)
        ev = p.explained_variance_ratio_
        a.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=7); a.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=7); a.tick_params(labelsize=6)
        if i == 0: a.set_title(nm, fontsize=8, color=plotting.C_UNAL if j == 0 else C_GLOB)
    ax[i, 0].annotate(st, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90, va="center", fontsize=8)
ax[0, 0].legend(frameon=False, fontsize=6, loc="best")
fig.tight_layout(); fig.savefig(EXP / "head_pca_sweep_example.pdf"); fig.savefig(EXP / "head_pca_sweep_example.png", dpi=200); plt.show()

for nm, _ in MODS:
    f3 = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2], subplot_titles=["gradient", "attention x gradient"])
    for c, st in enumerate(["gradient", "attention x gradient"], start=1):
        vm, vn = V[nm][st]; X = np.vstack([vm, vn]); p = PCA(3).fit(X); Z = p.transform(X); n = len(vm)
        f3.add_trace(go.Scatter3d(x=Z[:n, 0], y=Z[:n, 1], z=Z[:n, 2], mode="markers", marker=dict(size=3, color=C_MEL), name=f"MEL {st}"), 1, c)
        f3.add_trace(go.Scatter3d(x=Z[n:, 0], y=Z[n:, 1], z=Z[n:, 2], mode="markers", marker=dict(size=3, color=C_NV), name=f"NV {st}"), 1, c)
        pu = p.components_ @ (np.ones(12) / np.sqrt(12)); s = np.abs(Z).max()
        f3.add_trace(go.Scatter3d(x=[-s * pu[0], s * pu[0]], y=[-s * pu[1], s * pu[1]], z=[-s * pu[2], s * pu[2]],
                                  mode="lines", line=dict(color="black", width=4), name="diagonal (mean over heads)"), 1, c)
    f3.update_layout(title=f"{nm}, PC1 to PC3", height=600, width=1200); f3.write_html(EXP / f"head_pca_3d_{nm.replace(' ', '_')}.html")

# ---------- 2. energy split over the 31 images ----------
rows = []
for iid in sorted(HUM):
    for nm, m in MODS:
        vv = head_vectors(m, iid)
        for st in STEPS:
            vm, vn = vv[st]; s12, r12 = orth_share(vm, vn)
            s3, r3 = orth_share(vm, vn, low3(vm, vn)) if st != "attention" else (np.nan, 0.0)
            rows.append(dict(image_id=iid, model=nm, step=st, orth12=s12, rel12=r12, orth3=s3, rel3=r3))
ES = pd.DataFrame(rows); ES.to_csv(EXP / "head_energy_split.csv", index=False)
sw = ES[ES.image_id == SWEEP].set_index(["model", "step"])
tab = ES.groupby(["model", "step"])[["orth12", "rel12", "orth3", "rel3"]].median()
print("sweep example and median over 31 images")
print(f"{'model':17} {'step':21} {'orth 12 (sweep)':>15} {'orth 12 (median)':>16} {'rel diff 12':>11} {'orth 3 (median)':>15} | diag in PC1-2 plane")
for nm, _ in MODS:
    for st in STEPS:
        o, t = sw.loc[(nm, st)], tab.loc[(nm, st)]
        f = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"
        print(f"{nm:17} {st:21} {f(o.orth12):>15} {f(t.orth12):>16} {t.rel12:>11.3f} {f(t.orth3):>15} | {inplane[(st, nm)]:.2f}")
print("files:", [p.name for p in EXP.glob("head_*")])


# ---- cell 153 ------------------------------------------------------------
ES["kept"] = (1 - ES.orth12) * ES.rel12
print(ES[ES.step == "attention x gradient"].groupby("model").kept.median().round(3))


# ---- cell 154 ------------------------------------------------------------
u = np.ones(12) / np.sqrt(12)
def hv(model, iid):
    x = data.to_tensor(data.native(EV.loc[iid].image_rel_path))
    with torch.enable_grad():
        base, logits = cams._forward(model, x); A_ = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A_, retain_graph=True)[0]; Gn = torch.autograd.grad(logits[0, NV_IDX], A_)[0]
    base.clear_xai_state()
    A_, Gm, Gn = [T_[0, :, 1:, 1:].detach().float().cpu().numpy() for T_ in (A_, Gm, Gn)]
    k = np.flatnonzero(LESION[iid].ravel())
    return (A_ * Gm).sum(1)[:, k].T, (A_ * Gn).sum(1)[:, k].T          # cells x 12, before ReLU
def split(V): x = V @ u; return x, np.linalg.norm(V - x[:, None] * u, axis=1)

MODS = [("Unaligned", M0, plotting.C_UNAL), ("Globally aligned", M5, C_GLOB)]
PTS = {}
for nm, m, _ in MODS:
    for iid in sorted(HUM):
        vm, vn = hv(m, iid)
        for c, V in [("MEL", vm), ("NV", vn)]:
            x_, y_ = split(V)
            PTS.setdefault(nm, []).append(pd.DataFrame({"image_id": iid, "cls": c, "x": x_, "y": y_}))
PTS = {k: pd.concat(v, ignore_index=True) for k, v in PTS.items()}
pd.concat([d.assign(model=k) for k, d in PTS.items()]).to_csv(EXP / "head_diagonal_points.csv", index=False)

fig, ax = plt.subplots(2, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.78))
for i, (row, sel, s, al) in enumerate([("One melanoma", lambda d: d[d.image_id == SWEEP], 12, 0.85),
                                       ("31 annotated images", lambda d: d, 3, 0.15)]):
    D = {nm: sel(PTS[nm]) for nm, _, _ in MODS}
    X = pd.concat(D.values()); xl = np.abs(X.x).max() * 1.05; yl = X.y.max() * 1.05
    for j, (nm, _, col) in enumerate(MODS):
        a = ax[i, j]; d = D[nm]
        for c, cc in [("NV", C_NV), ("MEL", C_MEL)]:
            g = d[d.cls == c]; a.scatter(g.x, g.y, s=s, color=cc, alpha=al, lw=0)
        a.axvline(0, color="0.3", ls=":", lw=0.9)
        a.set_xlim(-xl, xl); a.set_ylim(0, yl); a.tick_params(labelsize=6)
        if i == 0: a.set_title(nm, fontsize=8, color=col)
        if i == 1: a.set_xlabel("projection on the diagonal", fontsize=7)
    ax[i, 0].set_ylabel(f"{row}\nnorm of orthogonal part", fontsize=7)
ax[0, 0].legend([Line2D([], [], ls="none", marker="o", color=C_MEL, ms=4), Line2D([], [], ls="none", marker="o", color=C_NV, ms=4)],
                ["melanoma map", "nevus map"], frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(); save(fig, "fig_ch4_head_diagonal"); plt.show()

for nm, _, _ in MODS:
    for row, d in [("sweep", PTS[nm][PTS[nm].image_id == SWEEP]), ("pooled", PTS[nm])]:
        s_ = " | ".join(f"{c}: median x {g.x.median():+.4f}, x>0 {(g.x > 0).mean():.0%}, median y {g.y.median():.4f}"
                        for c, g in d.groupby("cls"))
        print(f"{nm:17} {row:7} {s_}")

ES = pd.read_csv(EXP / "head_energy_split.csv"); ES["kept"] = (1 - ES.orth12) * ES.rel12
T3 = ES[ES.step == "attention x gradient"].groupby("model")[["rel12", "orth12", "kept"]].median()
T3.columns = ["rel diff", "orthogonal share", "kept"]
print("\nmedian over 31 images, attention x gradient (random baseline for orthogonal share 11/12 = 0.917)")
print(T3.round(3).to_string())


# ---- cell 155 ------------------------------------------------------------
ID_T = pd.read_csv(TAB / "app_figure_images.csv"); ID_T = ID_T[ID_T.figure != "head diagonal"]
ID_T = pd.concat([ID_T, pd.DataFrame([{"figure": "head diagonal", "panel": "top row", "image_id": SWEEP}])], ignore_index=True)
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
esc = lambda s: str(s).replace("_", r"\_")
Lt = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
Lt += [f"{esc(q.figure)} & {esc(q.panel)} & {esc(q.image_id)} \\\\" for q in ID_T.itertuples()]
(TAB / "app_figure_images.tex").write_text("\n".join(Lt + [r"\bottomrule", r"\end{tabular}"]) + "\n")

ES = pd.read_csv(EXP / "head_energy_split.csv"); ES["kept"] = (1 - ES.orth12) * ES.rel12
ES.to_csv(EXP / "head_energy_split.csv", index=False)
print(f"app_figure_images.tex: {len(ID_T)} rows | head_energy_split.csv columns: {list(ES.columns)}")


# ---- cell 156 ------------------------------------------------------------
RAW = pd.read_csv(QC_CSV); RAW["Image_ID"] = RAW.Image_ID.astype(str)
RAW["survives"] = RAW.qc_pass.astype(bool)
GROUP = lambda g: "melanoma indicative" if str(g).upper() == "MEL" else ("benign" if str(g).upper() == "NV" else "outside the checklist")
LG = RAW.drop_duplicates("label").set_index("label").label_group.map(GROUP)
print("label groups:", LG.value_counts().to_dict(), "| raw group values:", sorted(RAW.label_group.astype(str).unique()))
assert len(LG) == 15, f"STOP: found {len(LG)} labels, expected 15: {sorted(LG.index)}"
nice = lambda s: s.replace("preeminent", "prominent").replace("_", " ")
STYLE = {"melanoma indicative": (C_SMEL, "solid"), "benign": ("#FFFFFF", "dashed"), "outside the checklist": ("#FFFFFF", "dotted")}

rows = []
for lab, grp in LG.items():
    surv = RAW[(RAW.label == lab) & RAW.survives]; pool = surv if len(surv) else RAW[RAW.label == lab]
    med = pool.px_positive.median(); pick = pool.iloc[(pool.px_positive - med).abs().argsort().iloc[0]]
    rows.append(dict(label=lab, group=grp, median_px=med, n=len(surv) if len(surv) else len(pool),
                     removed=len(surv) == 0, image_id=pick.Image_ID, mask_path=pick.mask_path))
PK = pd.DataFrame(rows)
ORDER = ["melanoma indicative", "benign", "outside the checklist"]
PK["g"] = PK.group.map(ORDER.index); PK = PK.sort_values(["g", "median_px"]).reset_index(drop=True)
PK.to_csv(TAB / "app_label_masks.csv", index=False)
print(PK[["group", "label", "n", "median_px", "removed"]].to_string(index=False))

H0 = pd.read_csv(MELNV / "ham_mel_nv_clean.csv"); H0["image_id"] = H0.image_id.astype(str); H0 = H0.drop_duplicates("image_id").set_index("image_id")
def load(iid, mpath):
    img = np.array(Image.open((HAM / str(H0.loc[iid, "image_rel_path"])).resolve()).convert("RGB"))
    les = np.array(Image.open((HAM / str(H0.loc[iid, "mask_rel_path"])).resolve()).convert("L")) > 127
    ann = np.asarray(data._full(mpath)).astype(bool)
    for m in (les, ann):
        pass
    if les.shape != img.shape[:2]: les = cv2.resize(les.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    if ann.shape != img.shape[:2]: ann = cv2.resize(ann.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.where(les | ann); h, w = les.shape; pad = 20
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, h), max(xs.min() - pad, 0), min(xs.max() + pad, w)
    return img[y0:y1, x0:x1], les[y0:y1, x0:x1], ann[y0:y1, x0:x1]

NC = int(PK.groupby("group").size().max())
fig, ax = plt.subplots(3, NC, figsize=(TEXTWIDTH, TEXTWIDTH / NC * 3 * 1.25), squeeze=False)
for a in ax.ravel(): a.axis("off")
for gi, grp in enumerate(ORDER):
    sub = PK[PK.group == grp].reset_index(drop=True)
    for j, r in sub.iterrows():
        a = ax[gi, j]; a.axis("on"); img, les, ann = load(r.image_id, r.mask_path); col, ls = STYLE[grp]
        a.imshow(img); a.contour(les.astype(float), levels=[0.5], colors=[C_LESION], linewidths=0.8)
        a.contour(ann.astype(float), levels=[0.5], colors=[col], linewidths=0.9, linestyles=ls)
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(f"{nice(r.label)}\n" + ("removed from grid" if r.removed else f"n = {r.n}"), fontsize=6)
    ax[gi, 0].set_ylabel(grp.capitalize(), fontsize=7, color="black")
fig.tight_layout(); save(fig, "fig_app_label_masks"); plt.show()

ID_T = pd.read_csv(TAB / "app_figure_images.csv"); ID_T = ID_T[ID_T.figure != "label masks"]
ID_T = pd.concat([ID_T, pd.DataFrame({"figure": "label masks", "panel": PK.label.map(nice), "image_id": PK.image_id})], ignore_index=True)
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
esc = lambda s: str(s).replace("_", r"\_")
Lt = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
Lt += [f"{esc(q.figure)} & {esc(q.panel)} & {esc(q.image_id)} \\\\" for q in ID_T.itertuples()]
(TAB / "app_figure_images.tex").write_text("\n".join(Lt + [r"\bottomrule", r"\end{tabular}"]) + "\n")
nrem = int(PK.removed.sum())
print(f"app_figure_images.tex: {len(ID_T)} rows | removed labels: {nrem}")
print(f"""=== PROMPT ===
Appendix C, add fig_app_label_masks.pdf, label fig:app-label-masks. Plain sentences without antithesis constructions. No image IDs in captions.
Caption: "One example annotation for each of the 15 labels, grouped by target class (rows: melanoma indicative, benign, outside the checklist) and ordered within each row by median mask size. Each example is the mask closest to the median size of its label. The image is cropped around the lesion. Green: lesion outline. Magenta solid: melanoma indicative label. White dashed: benign label. White dotted: label outside the checklist. Below each panel: label name and number of masks that survive on the 14 x 14 grid.{' Panels marked removed from grid show a drawn mask whose label has no annotation covering at least half of a grid cell.' if nrem else ''} Image IDs are listed in Table~\\ref{{tab:app-figure-images}}."
Text: one sentence in Appendix C where the label set is introduced, pointing to Figure~\\ref{{fig:app-label-masks}}.
Appendix A: tab:app-figure-images was re-exported with {len(PK)} rows for this figure, one per label. Replace "label masks" in the figure column with the reference to fig:app-label-masks.
=== END ===""")


# ---- cell 157 ------------------------------------------------------------
import re, textwrap
PK = pd.read_csv(TAB / "app_label_masks.csv")
nice = lambda s: re.sub(r"pre+e?minent", "prominent", str(s)).replace("_", " ")
NC = 5
LAYOUT = []
for grp in ORDER:
    sub = PK[PK.group == grp].reset_index(drop=True)
    for start in range(0, len(sub), NC):
        LAYOUT.append((grp, start == 0, sub.iloc[start:start + NC]))
fig, ax = plt.subplots(len(LAYOUT), NC, figsize=(TEXTWIDTH, TEXTWIDTH / NC * len(LAYOUT) * 1.32), squeeze=False)
for a in ax.ravel(): a.axis("off")
for ri, (grp, first, sub) in enumerate(LAYOUT):
    col, ls = STYLE[grp]
    for j, r in enumerate(sub.itertuples()):
        a = ax[ri, j]; a.axis("on"); img, les, ann = load(r.image_id, r.mask_path)
        a.imshow(img); a.contour(les.astype(float), levels=[0.5], colors=[C_LESION], linewidths=0.8)
        a.contour(ann.astype(float), levels=[0.5], colors=[col], linewidths=0.9, linestyles=ls)
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel("\n".join(textwrap.wrap(nice(r.label), 18)) + f"\nn = {r.n}", fontsize=6, linespacing=1.05)
    if first:
        ax[ri, 0].set_ylabel("\n".join(textwrap.wrap(grp.capitalize(), 12)), fontsize=7, color="black")
fig.subplots_adjust(left=0.07, right=0.995, top=0.995, bottom=0.03, wspace=0.08, hspace=0.62)
save(fig, "fig_app_label_masks"); plt.show()

ID_T = pd.read_csv(TAB / "app_figure_images.csv"); ID_T = ID_T[ID_T.figure != "label masks"]
ID_T = pd.concat([ID_T, pd.DataFrame({"figure": "label masks", "panel": PK.label.map(nice), "image_id": PK.image_id})], ignore_index=True)
ID_T.to_csv(TAB / "app_figure_images.csv", index=False)
esc = lambda s: str(s).replace("_", r"\_")
Lt = [r"\begin{tabular}{lll}", r"\toprule", r"figure & panel & image \\", r"\midrule"]
Lt += [f"{esc(q.figure)} & {esc(q.panel)} & {esc(q.image_id)} \\\\" for q in ID_T.itertuples()]
(TAB / "app_figure_images.tex").write_text("\n".join(Lt + [r"\bottomrule", r"\end{tabular}"]) + "\n")

p = TAB / "app_case_manifest.tex"; s = p.read_text()
n_fix = len(re.findall(r"pre+e?minent", s, flags=re.I))
p.write_text(re.sub(r"pre+e?minent", "prominent", s, flags=re.I))
print(f"app_figure_images.tex: {len(ID_T)} rows | app_case_manifest.tex: fixed {n_fix} occurrence(s)")


# ---- cell 158 ------------------------------------------------------------
def cos(a, b): a, b = np.ravel(a), np.ravel(b); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
MC0 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha0.pth", cache=False)
MC5 = models.load_ckpt(CK5 / "checkpoint-best-cls-ha5.pth", cache=False)

def raw(model, x):
    with torch.enable_grad():
        base, logits = cams._forward(model, x); A_ = base.blocks[BLOCK].attn.last_attn
        Gm = torch.autograd.grad(logits[0, MEL_IDX], A_, retain_graph=True)[0]; Gn = torch.autograd.grad(logits[0, NV_IDX], A_)[0]
    gap = bool(getattr(base, "use_mean_pooling", False)); base.clear_xai_state()
    f = (lambda T: T[0, :, 1:, 1:]) if gap else (lambda T: T[0, :, 0, 1:])
    A_, Gm, Gn = [f(T).detach().float().cpu().numpy() for T in (A_, Gm, Gn)]
    Pm, Pn = A_ * Gm, A_ * Gn
    Hm, Hn = (Pm.sum(1), Pn.sum(1)) if gap else (Pm, Pn)          # heads x keys
    return Gm, Gn, Pm, Pn, Hm, Hn

def stages(R, k, heads):
    Gm, Gn, Pm, Pn, Hm, Hn = [T[heads] for T in R]; sl = lambda T: T[..., k]
    fin = lambda H: cams.norm01(np.maximum(H.mean(0), 0))
    return {"gradient": cos(sl(Gm), sl(Gn)), "attention x gradient": cos(sl(Pm), sl(Pn)),
            "typical head": float(np.median([cos(Hm[h][k], Hn[h][k]) for h in range(len(heads))])),
            "mean over heads": cos(Hm.mean(0)[k], Hn.mean(0)[k]), "final map": cos(fin(Hm)[k], fin(Hn)[k])}

rows = []
for iid in sorted(HUM):
    x = data.to_tensor(data.native(EV.loc[iid].image_rel_path)); k = np.flatnonzero(LESION[iid].ravel())
    for pool, nm, m in [("GAP", "aligned", M5), ("GAP", "unaligned", M0), ("CLS", "aligned", MC5), ("CLS", "unaligned", MC0)]:
        R = raw(m, x); all12 = list(range(R[4].shape[0]))
        rows.append({"pool": pool, "model": nm, "heads": "12", "image_id": iid, **stages(R, k, all12)})
        if pool == "GAP":
            low3 = list(np.argsort([cos(R[4][h][k], R[5][h][k]) for h in all12])[:3])
            rows.append({"pool": pool, "model": nm, "heads": "3", "image_id": iid, **stages(R, k, low3)})
HS = pd.DataFrame(rows); HS.to_csv(TAB / "ch4_head_stages_v2.csv", index=False)
ROWS = ["gradient", "attention x gradient", "typical head", "mean over heads", "final map"]
M = HS.groupby(["pool", "model", "heads"])[ROWS].median()

ref = [0.690, 0.717, 0.898, 0.991, 0.993]
assert np.allclose(M.loc[("GAP", "aligned", "12")].values, ref, atol=0.002), f"STOP: 12-head GAP aligned differs: {M.loc[('GAP','aligned','12')].round(3).tolist()}"
assert abs(M.loc[("GAP", "aligned", "3"), "mean over heads"] - 0.663) < 0.005, f"STOP: 3-head mean {M.loc[('GAP','aligned','3'),'mean over heads']:.3f}, expected 0.663"

def tex(cols, heads, fname, label):
    L_ = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", "stage & " + " & ".join(heads) + r" \\", r"\midrule"]
    L_ += [f"{r_} & " + " & ".join(f"{M.loc[c, r_]:.3f}" for c in cols) + r" \\" for r_ in ROWS]
    (TAB / fname).write_text("\n".join(L_ + [r"\bottomrule", r"\end{tabular}"]) + "\n")
tex([("GAP", "aligned", "12"), ("GAP", "unaligned", "12"), ("GAP", "aligned", "3"), ("GAP", "unaligned", "3")],
    ["12 heads, aligned", "12 heads, unaligned", "3 heads, aligned", "3 heads, unaligned"], "tab_head_stages.tex", "tab:head-stages")
tex([("CLS", "aligned", "12"), ("CLS", "unaligned", "12")], ["12 heads, aligned", "12 heads, unaligned"],
    "tab_app_head_stages_cls.tex", "tab:app-head-stages-cls")
print(M.round(3).to_string())
g = lambda h, r_: M.loc[("GAP", "aligned", h), r_]; gu = lambda h, r_: M.loc[("GAP", "unaligned", h), r_]
print(f"""=== PROMPT ===
Section 4.2 and Appendix C, stage tables. Plain sentences without antithesis constructions. Source: ch4_head_stages_v2.csv.
1. Replace tab:head-stages with \\input{{tables/tab_head_stages.tex}} (GAP only, four columns: 12 heads aligned, 12 heads unaligned, three least similar heads aligned, three least similar heads unaligned). Caption: "Cosine similarity between the melanoma and nevus versions of each stage of the map construction, GAP pooling, over lesion cells, median over the 31 annotated images. The three-head columns use, per image, the three heads whose melanoma and nevus attention x gradient are least similar inside the lesion, and every row is restricted to those heads. These columns are low partly by selection. The typical head is the median over the heads of the column. Cosines of the non-negative final maps include their shared mean and are not comparable with r_in. Class token pooling: Table~\\ref{{tab:app-head-stages-cls}}."
2. Appendix C: add \\input{{tables/tab_app_head_stages_cls.tex}}, label tab:app-head-stages-cls. Caption: "Same as Table~\\ref{{tab:head-stages}} for class token pooling, all 12 heads." Keep tab:app-head-stages-all.
3. Section 4.2 text: remove the class token values from the paragraph on the stage table and keep the GAP values: gradient {g('12','gradient'):.3f}, attention x gradient {g('12','attention x gradient'):.3f}, typical head {g('12','typical head'):.3f}, mean over heads {g('12','mean over heads'):.3f} (unaligned {gu('12','mean over heads'):.3f}). Add: "Restricted to the three least similar heads of each image, the aligned model keeps a cosine of {g('3','mean over heads'):.3f} after averaging and {g('3','final map'):.3f} in the final map. The separation these heads carry is lost once all twelve heads are averaged." Add one sentence pointing to Table~\\ref{{tab:app-head-stages-cls}} for class token pooling, which shows the same pattern.
=== END ===""")


# ---- cell 159 ------------------------------------------------------------
_outline0 = plotting._outline0
def outline(ax, m, color=None, *a, **kw):
    ls = kw.get("ls", kw.get("linestyle", "solid")); c = str(color).lower() if color is not None else ""
    if (c in ("#ffffff", "white", "w") and ls in ("solid", "-")) or c == C_SMEL.lower():
        lw = kw.pop("lw", kw.pop("linewidth", 0.8))
        _outline0(ax, m, "black", *a, lw=lw + 0.9, **kw)          # edge
        return _outline0(ax, m, C_SMEL, *a, lw=lw, **kw)
    return _outline0(ax, m, color, *a, **kw)
plotting.outline = outline
print("magenta outlines now drawn with a black edge")


# ---- cell 160 ------------------------------------------------------------
OUT = REPO / "figures/onepager"; OUT.mkdir(parents=True, exist_ok=True)
LD = pd.read_csv(TAB / "ch5_reference_ladder.csv"); LD["image_id"] = LD.image_id.astype(str)
P = LD.pivot_table(index="image_id", columns="ref", values="auc_in")
hm = [i for i in held if EV.loc[i].gt_label == "MEL" and i in P.index]
cover = {i: (HUM[i] & LESION[i]).sum() / LESION[i].sum() for i in hm}
C = [i for i in hm if 0.15 <= cover[i] <= 0.70]
assert C, "STOP: no held-out melanoma with structures covering 15 to 70 percent of the lesion"
IID = min(C, key=lambda i: abs(P.loc[i, "concept only"] - 0.793))
print(f"candidates {len(C)} of {len(hm)} | chosen {IID} | structure cover {cover[IID]:.0%} | AUC_in unaligned "
      f"{P.loc[IID, 'unaligned']:.3f}, global {P.loc[IID, 'lesion aligned']:.3f}, expert {P.loc[IID, 'concept only']:.3f}")

r = EV.loc[IID]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
L, U = LESION[IID], HUM[IID]; S = U & L
ys, xs = np.where(np.kron(L, np.ones((16, 16), bool))); pad = 12
X0, X1, Y0, Y1 = max(xs.min() - pad, 0), min(xs.max() + pad, 223), max(ys.min() - pad, 0), min(ys.max() + pad, 223)
o0 = plotting._outline0
def edge(ax, m, col, lw=0.7, ls="solid"):
    o0(ax, m, "black", lw=lw + 0.8, ls=ls); o0(ax, m, col, lw=lw, ls=ls)
def base_panel(ax, label):
    o0(ax, L, C_LESION, lw=0.8); edge(ax, S, C_SMEL)
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(label)
def top10(v):
    v = np.asarray(v, float).ravel(); li = np.flatnonzero(L.ravel()); k = int(np.ceil(0.10 * len(li)))
    rng = np.random.default_rng(SEED); o = li[np.lexsort((rng.random(len(li)), -v[li]))][:k]
    m = np.zeros(196, bool); m[o] = True; return m.reshape(14, 14)

m_exp = models.f2(0.0, 1.0, FOLD[IID])
SUP = [("Unaligned", cams.class_cams(M0, x)[0]), ("Global alignment", cams.class_cams(M5, x)[0]),
       ("Expert alignment", cams.class_cams(m_exp, x)[0])]
del m_exp
mel, nv = cams.class_cams(M5, x)
MET = [("Grad-CAM", mel), ("Diff-CAM", cams.norm01(np.maximum(np.asarray(mel) - np.asarray(nv), 0))),
       ("FinerCAM", cams.finer_cam(M5, x, "MEL", 0.6))]

W = 8.5 / 2.54
with plt.rc_context({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "svg.fonttype": "none"}):
    for name, panels, heat in [("fig_onepager_supervision", SUP, True), ("fig_onepager_methods", MET, False)]:
        fig, ax = plt.subplots(1, 3, figsize=(W, W / 3 * (Y1 - Y0) / (X1 - X0) + 0.25))
        for a, (lab, v) in zip(ax, panels):
            if heat: overlay(a, rgb, v)
            else:
                a.imshow(rgb)
                if np.asarray(v, float)[L].max() > 1e-6: edge(a, top10(v), "#FFFFFF", lw=0.8)
                else: print(f"{name}: {lab} is empty inside the lesion, no outline drawn")
            base_panel(a, lab)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.14, wspace=0.04)
        fig.savefig(OUT / f"{name}.png", dpi=600); fig.savefig(OUT / f"{name}.svg"); plt.show()
print("saved to", OUT)


# ---- cell 161 ------------------------------------------------------------
PICK = None                      # set to an image ID from the printed list to override
from collections import defaultdict
EXPM = {}
def exp_model(f):
    if f not in EXPM: EXPM[f] = models.f2(0.0, 1.0, f)
    return EXPM[f]
def is_mel(m, x): return float(models.predict_mel(m, x)[0]) >= 0.5
rows = []
for i in hm:
    if not 0.15 <= cover[i] <= 0.70: continue
    xi = data.to_tensor(data.native(EV.loc[i].image_rel_path)); me = exp_model(FOLD[i])
    if not (is_mel(M0, xi) and is_mel(M5, xi) and is_mel(me, xi)): continue
    Li = LESION[i]; g = np.asarray(cams.class_cams(M5, xi)[0], float).reshape(14, 14)
    f = np.asarray(cams.finer_cam(M5, xi, "MEL", 0.6), float).reshape(14, 14)
    r_gf = float(np.corrcoef(g[Li], f[Li])[0, 1]) if f[Li].std() > 1e-9 else 1.0
    rows.append(dict(image_id=i, unaligned=P.loc[i, "unaligned"], glob=P.loc[i, "lesion aligned"],
                     expert=P.loc[i, "concept only"], r_gradcam_finercam=r_gf))
CAND = pd.DataFrame(rows)
assert len(CAND), "STOP: no candidate with visible structures and correct predictions by all three models"
CAND["score"] = (CAND.expert - CAND.glob) + 0.3 * (1 - CAND.r_gradcam_finercam)
CAND = CAND.sort_values("score", ascending=False)
print(CAND.head(5).round(3).to_string(index=False))


# ---- cell 162 ------------------------------------------------------------
PICK = "ISIC_0030798"
IID = PICK or CAND.image_id.iloc[0]
v = CAND.set_index("image_id").loc[IID]
print(f"chosen {IID} | AUC_in unaligned {v.unaligned:.3f}, global {v.glob:.3f}, expert {v.expert:.3f}")

r = EV.loc[IID]; rgb = data.rgb224(r.image_rel_path); x = data.to_tensor(data.native(r.image_rel_path))
L, U = LESION[IID], HUM[IID]; S = U & L
ys, xs = np.where(np.kron(L, np.ones((16, 16), bool))); pad = 12
X0, X1, Y0, Y1 = max(xs.min() - pad, 0), min(xs.max() + pad, 223), max(ys.min() - pad, 0), min(ys.max() + pad, 223)
me = exp_model(FOLD[IID])
name = lambda m: "melanoma" if is_mel(m, x) else "nevus"
SUP = [("Unaligned", M0), ("Global alignment", M5), ("Expert alignment", me)]
mel, nv = cams.class_cams(M5, x)
MET = [("Grad-CAM", mel), ("Diff-CAM", cams.norm01(np.maximum(np.asarray(mel) - np.asarray(nv), 0))),
       ("FinerCAM", cams.finer_cam(M5, x, "MEL", 0.6))]

with plt.rc_context({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "svg.fonttype": "none"}):
    h = W / 3 * (Y1 - Y0) / (X1 - X0)
    fig, ax = plt.subplots(1, 3, figsize=(W, h + 0.42))
    for a, (lab, m) in zip(ax, SUP):
        overlay(a, rgb, cams.class_cams(m, x)[0]); base_panel(a, f"{lab}\nPrediction: {name(m)}")
    ax[0].set_ylabel("Ground truth: melanoma", color="black")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.24, wspace=0.04)
    fig.savefig(OUT / "fig_onepager_supervision.png", dpi=600); fig.savefig(OUT / "fig_onepager_supervision.svg"); plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(W, h + 0.25))
    for a, (lab, v_) in zip(ax, MET):
        overlay(a, rgb, v_); base_panel(a, lab)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.14, wspace=0.04)
    fig.savefig(OUT / "fig_onepager_methods.png", dpi=600); fig.savefig(OUT / "fig_onepager_methods.svg"); plt.show()
print("saved to", OUT)


# ---- cell 163 ------------------------------------------------------------
import matplotlib.patheffects as pe
EDGE = [pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()]
LEG = ([Line2D([], [], color=C_LESION, lw=1.2), Line2D([], [], color=C_SMEL, lw=1.2, path_effects=EDGE)],
       ["lesion outline", "melanoma structures"])
mel_e, nv_e = cams.class_cams(me, x)
fin_e = cams.finer_cam(me, x, "MEL", 0.6)
if np.asarray(fin_e, float).reshape(14, 14)[L].max() < 1e-6: print("note: expert FinerCAM is empty inside the lesion on this image")
FIGS = [("fig_onepager_supervision", [(lab, cams.class_cams(m, x)[0]) for lab, m in SUP]),
        ("fig_onepager_methods", MET),
        ("fig_onepager_methods_expert", [("Grad-CAM", mel_e),
                                         ("Diff-CAM", cams.norm01(np.maximum(np.asarray(mel_e) - np.asarray(nv_e), 0))),
                                         ("FinerCAM", fin_e)])]
with plt.rc_context({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "svg.fonttype": "none"}):
    h = W / 3 * (Y1 - Y0) / (X1 - X0)
    for nm, panels in FIGS:
        fig, ax = plt.subplots(1, 3, figsize=(W, h + 0.42))
        for a, (lab, v_) in zip(ax, panels):
            overlay(a, rgb, v_); base_panel(a, lab)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.24, wspace=0.04)
        fig.legend(*LEG, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, handlelength=1.4, columnspacing=1.2)
        fig.savefig(OUT / f"{nm}.png", dpi=600); fig.savefig(OUT / f"{nm}.svg"); plt.show()
print("saved to", OUT)


# ---- cell 164 ------------------------------------------------------------
def top10_overlay(ax, v):
    overlay(ax, rgb, v)
    keep = np.kron(top10(v), np.ones((16, 16), bool))          # hottest 10 % of lesion cells, 224 x 224
    cover = np.dstack([rgb / 255.0, (~keep).astype(float)])     # plain image everywhere else
    ax.imshow(cover, interpolation="nearest")

with plt.rc_context({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "svg.fonttype": "none"}):
    h = W / 3 * (Y1 - Y0) / (X1 - X0)
    fig, ax = plt.subplots(1, 3, figsize=(W, h + 0.42))
    for a, (lab, v_) in zip(ax, MET):
        if np.asarray(v_, float).reshape(14, 14)[L].max() < 1e-6:
            a.imshow(rgb); print(f"note: {lab} is empty inside the lesion")
        else:
            top10_overlay(a, v_)
        base_panel(a, lab)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.24, wspace=0.04)
    fig.legend(*LEG, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, handlelength=1.4, columnspacing=1.2)
    fig.savefig(OUT / "fig_onepager_methods_top10.png", dpi=600); fig.savefig(OUT / "fig_onepager_methods_top10.svg"); plt.show()
print("saved to", OUT)


# ---- cell 165 ------------------------------------------------------------
import re
TEXP = TAB / "appD_statistics.tex"
SEC_OVERRIDE = {}          # e.g. {"exact comparison text": "4.2"}

def section_of(r):
    c, ch = str(r.comparison).lower(), int(r.chapter)
    if c in (k.lower() for k in SEC_OVERRIDE): return {k.lower(): v for k, v in SEC_OVERRIDE.items()}[c]
    if c.startswith("clinician preference"): return "4.7"
    if c.startswith("deletion"): return "4.6"
    if c.startswith("depth"): return "4.5"
    if "by class" in c and "global aligned vs unaligned" in c: return "4.1"
    if any(k in c for k in ["darkness", "centre", "reference"]): return "4.5"
    if any(k in c for k in ["finercam", "diff-cam", "grad-cam", "group ", "gamma"]): return "4.4"
    if any(k in c for k in ["sweep", "lambda", "alpha", "weight"]): return "4.3"
    if any(k in c for k in ["r_in", "corr", "collapse", "head", "pearson"]): return "4.2"
    return {4: "4.1", 5: "4.1", 6: "4.4", 7: "4.5"}.get(ch, "?")

def fmt_p(v):
    if pd.isna(v): return "--"
    if v < 1e-3: m, e = f"{v:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
def fmt_d(v):
    if pd.isna(v): return "--"
    r = round(float(v), 3)
    return "0.000" if r == 0 else (f"${r:.3f}$" if r < 0 else f"{r:.3f}")

def export_appD(ST):
    lines = TEXP.read_text().splitlines()
    is_row = lambda l: "&" in l and l.rstrip().endswith("\\\\")
    data = [i for i, l in enumerate(lines) if is_row(l) and "comparison" not in l and not l.lstrip().startswith("\\multicolumn")]
    assert data and data == list(range(data[0], data[-1] + 1)), "STOP: data rows in appD_statistics.tex are not one block"
    head = [l.replace("ch.", "Sec.", 1) if "comparison" in l else l for l in lines[:data[0]]]
    rows = [f"{r.section} & {tx(r.comparison)} & {METN.get(r.metric, tx(r.metric))} & {fi(r.n)} & {fmt_d(r.median_diff)} & "
            f"{fmt_p(r.p)} & {fi(r.family)} & {fmt_p(r.threshold)} & {tx(r.survives)} \\\\" for r in ST.itertuples()]
    TEXP.write_text("\n".join(head + rows + lines[data[-1] + 1:]) + "\n")
    return len(rows)

ST = pd.read_csv(TAB / "appD_statistics.csv")
ST["section"] = ST.apply(section_of, axis=1)
assert not (ST.section == "?").any(), f"STOP: unmapped tests:\n{ST[ST.section == '?'].comparison.tolist()}"
ST["_o"] = range(len(ST)); ST = ST.sort_values(["section", "_o"]).drop(columns="_o").reset_index(drop=True)
ST.to_csv(TAB / "appD_statistics.csv", index=False)
n = export_appD(ST)
for s, g in ST.groupby("section"):
    print(f"\n{s}: {len(g)} tests"); print("  " + "\n  ".join(g.comparison.tolist()))
print(f"\nappD_statistics.tex: {n} rows, {int((ST.survives == 'yes').sum())} surviving | header kept, first column 'Sec.'")


# ---- cell 166 ------------------------------------------------------------
import re
DROP_HEAD, B = True, 2000
SEC_OVERRIDE = {"shifted: heat vs area, global aligned": "4.1",
                "uniform: heat vs area, global aligned": "4.1",
                "background patch: heat vs area, global aligned": "4.1"}
TEXP = TAB / "appD_statistics.tex"
GROUPS = {"UNASSIGNED": "unannotated tissue", "NETWORK": "network", "REGRESSION": "regression",
          "STRUCTURELESS": "structureless area", "STRUCTURELESS_AREA": "structureless area",
          "DOTS": "atypical dots", "ATYPICAL_DOTS": "atypical dots",
          "BLUE_WHITE": "blue-white structureless area", "BWSA": "blue-white structureless area",
          "BLUE_WHITE_STRUCTURELESS_AREA": "blue-white structureless area"}

def relabel(s):
    s = str(s)
    s = re.sub(r"\b(lesion aligned|global aligned|globally-aligned)\b", "globally aligned", s)
    s = re.sub(r"\b(concept only|HA2 b1)\b", "expert aligned", s)
    s = re.sub(r"\b(epoch 9|ep9)\b", "final epoch", s)
    s = re.sub(r"group ([A-Z_]+)", lambda m: "group " + GROUPS.get(m.group(1), m.group(1).lower().replace("_", " ")), s)
    return re.sub(r"auc_in|AUC_in|AUC in", "AUCINPH", s)
def tex_label(s): return tx(relabel(s)).replace("AUCINPH", r"$\mathrm{AUC}_{\mathrm{in}}$")

def section_of(c):
    if c in SEC_OVERRIDE: return SEC_OVERRIDE[c]
    l = c.lower()
    if l.startswith("deletion"): return "4.6"
    if l.startswith("clinician preference"): return "4.7"
    if "group " in l: return "F"
    if "head" in l: return "4.2"
    if any(k in l for k in ["sweep", "lambda", "λ"]): return "4.3"
    if any(k in l for k in ["expert", "control", "darkness", "joint", "global only", "centre", "depth", "reference", "ha2", "concept"]): return "4.5"
    if any(k in l for k in ["finercam", "diff-cam", "random", "gamma", "γ"]): return "4.4"
    if "unaligned" in l: return "4.1"
    return "?"

def fmt_p(v, boot=False):
    if pd.isna(v): return "--"
    if boot and v == 0: return f"$<{1 / B:g}$"
    if v < 1e-3: m, e = f"{v:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
def fmt_d(v):
    if pd.isna(v): return "--"
    r = round(float(v), 3); return "0.000" if r == 0 else (f"${r:.3f}$" if r < 0 else f"{r:.3f}")

ST = pd.read_csv(TAB / "appD_statistics.csv")
heads = ST[ST.comparison.str.contains("head", case=False) & ST.comparison.str.contains("oracle|weight", case=False)]
print("head oracle / weight tests:", heads.comparison.tolist() or "none")
if DROP_HEAD: ST = ST[~ST.index.isin(heads.index)]
ST["section"] = ST.comparison.map(section_of)
assert not (ST.section == "?").any(), f"STOP: unmapped:\n{ST[ST.section == '?'].comparison.tolist()}"
ST["threshold"] = 0.05 / ST.family
ST["survives"] = np.where(ST.p.isna(), "n/a", np.where(ST.p < ST.threshold, "yes", "no"))
ST["_k"] = ST.section.map(lambda s: (1, 0) if s == "F" else (0, float(s))); ST["_o"] = range(len(ST))
ST = ST.sort_values(["_k", "_o"]).drop(columns=["_k", "_o"]).reset_index(drop=True)
ST.to_csv(TAB / "appD_statistics.csv", index=False)

lines = TEXP.read_text().splitlines()
is_row = lambda l: "&" in l and l.rstrip().endswith("\\\\")
data = [i for i, l in enumerate(lines) if is_row(l) and "comparison" not in l.lower() and not l.lstrip().startswith("\\multicolumn")]
assert data and data == list(range(data[0], data[-1] + 1)), "STOP: data rows are not one block, header not touched"
boot = ST.test.astype(str).str.contains("bootstrap", case=False)
rows = [f"{r.section} & {tex_label(r.comparison)} & {tex_label(METN.get(r.metric, r.metric))} & {fi(r.n)} & {fmt_d(r.median_diff)} & "
        f"{fmt_p(r.p, b)} & {fi(r.family)} & {fmt_p(r.threshold)} & {tx(r.survives)} \\\\" for r, b in zip(ST.itertuples(), boot)]
TEXP.write_text("\n".join(lines[:data[0]] + rows + lines[data[-1] + 1:]) + "\n")

for s, g in ST.groupby("section", sort=False):
    print(f"\n{s}: {len(g)}"); print("  " + "\n  ".join(relabel(c) for c in g.comparison))
NT, NS = len(ST), int((ST.survives == "yes").sum())
print(f"\nbootstrap p = 0 rows: {int((boot & (ST.p == 0)).sum())} | total {NT} tests, {NS} surviving")
print(f"""=== PROMPT ===
Appendix D: appD_statistics.tex was re-exported with the header block and caption unchanged. The section column now gives the section where each test appears (4.1 to 4.7, F for the structure group tests). Bootstrap p values of 0 print as $<0.0005$ (B = 2000 resamples), labels use AUC_in in math form, the thesis group names, "globally aligned" and "final epoch". {'The head oracle and head weight tests were removed. ' if DROP_HEAD and len(heads) else ''}It now lists {NT} tests, {NS} of which survive correction. Update the test count in the appendix text, and add to the caption: "Bootstrap p values of 0 are reported as below 1/B with B = 2000."
=== END ===""")


# ---- cell 167 ------------------------------------------------------------
import re
TEXP = TAB / "appD_statistics.tex"
ST = pd.read_csv(TAB / "appD_statistics.csv")
lines = TEXP.read_text().splitlines()
is_row = lambda l: "&" in l and l.rstrip().endswith("\\\\")
data = [i for i, l in enumerate(lines) if is_row(l) and "comparison" not in l.lower() and not l.lstrip().startswith("\\multicolumn")]
assert data == list(range(data[0], data[-1] + 1)) and len(data) == len(ST), f"STOP: {len(data)} tex rows vs {len(ST)} csv rows"

# 1. sections: 4.3 = bootstrap sweep rows only, everything else formerly in 4.3 goes to 4.5
src = ST.source.astype(str).str.lower(); cmp_ = ST.comparison.astype(str).str.lower()
not_expert = ~cmp_.str.contains("expert|joint|control|global only|ha2|concept|final epoch|nv map")
sweep = (ST.test.astype(str).str.contains("bootstrap", case=False)
         | src.str.contains("lambda_sweep|lambda_collapse|lambda_localisation")) & not_expert
extra = ST[sweep & ~ST.test.astype(str).str.contains("bootstrap", case=False)]
print("sweep rows found by source, not labelled bootstrap:"); print(extra[["comparison", "test", "source"]].to_string())
was43 = ST.section.astype(str) == "4.3"
print("4.3 rows that are not sweep rows, moved to 4.5:"); print("  " + "\n  ".join(ST[was43 & ~sweep].comparison))
ST.loc[was43 & ~sweep, "section"] = "4.5"
ST.loc[sweep, "section"] = "4.3"
fg = ST.comparison.str.contains("FinerCAM vs Grad-CAM", case=False) & ST.comparison.str.lower().str.startswith("global") & (ST.n == 14)
assert fg.sum() == 1, f"STOP: found {int(fg.sum())} rows for 'global: FinerCAM vs Grad-CAM' (n = 14)"
ST.loc[fg, "section"] = "4.5"
if sweep.sum() != 20:
    print(ST[sweep][["comparison", "test"]].to_string()); raise AssertionError(f"STOP: {int(sweep.sum())} bootstrap rows, expected 20")

# 2. labels: placeholder and by-class rows
AUC = r"$\mathrm{AUC}_{\mathrm{in}}$"
def fix_text(s):
    s = re.sub(r"by class,\s*MEL\b", "melanomas", s); s = re.sub(r"by class,\s*NV\b", "nevi", s)
    return s.replace("AUCINPH", AUC)
ST["comparison"] = ST.comparison.map(lambda s: re.sub(r"by class,\s*MEL\b", "melanomas", re.sub(r"by class,\s*NV\b", "nevi", s)))
rows = []
for li, r in zip(data, ST.itertuples()):
    cells = lines[li].split("&", 1)
    rows.append((r.section, f"{r.section} &" + fix_text(cells[1])))
left = [l for _, l in rows if re.search(r"[A-Z]{3,}PH\b", l)]
assert not left, f"STOP: placeholders remain:\n" + "\n".join(left)

# reorder by section, keep order within a section, write back
key = lambda s: (1, 0) if s == "F" else (0, float(s))
order = sorted(range(len(rows)), key=lambda i: (key(rows[i][0]), i))
TEXP.write_text("\n".join(lines[:data[0]] + [rows[i][1] for i in order] + lines[data[-1] + 1:]) + "\n")
ST.iloc[order].reset_index(drop=True).to_csv(TAB / "appD_statistics.csv", index=False)

cnt = ST.section.value_counts()
print("\nrows per section:", {s: int(cnt[s]) for s in sorted(cnt.index, key=key)})
print(f"total {len(ST)} tests, {int((ST.survives == 'yes').sum())} surviving (unchanged)")


# ---- cell 168 ------------------------------------------------------------
import re
for f in ["tx", "fi", "METN"]: assert f in globals(), f"run the statistics export cell first ({f})"
TEXP, B = TAB / "appD_statistics.tex", 2000
EXPECT = {"4.1": 18, "4.2": 1, "4.3": 20, "4.4": 18, "4.5": 118, "4.6": 34, "4.7": 5, "F": 18}
GROUPS = {"UNASSIGNED": "unannotated tissue", "NETWORK": "network", "REGRESSION": "regression",
          "STRUCTURELESS": "structureless area", "STRUCTURELESS_AREA": "structureless area",
          "DOTS": "atypical dots", "ATYPICAL_DOTS": "atypical dots", "BLUE_WHITE": "blue-white structureless area",
          "BWSA": "blue-white structureless area", "BLUE_WHITE_STRUCTURELESS_AREA": "blue-white structureless area"}
AUC = r"$\mathrm{AUC}_{\mathrm{in}}$"

def relabel(s):
    s = str(s)
    s = re.sub(r"by class,\s*MEL\b", "melanomas", s); s = re.sub(r"by class,\s*NV\b", "nevi", s)
    s = re.sub(r"\b(lesion aligned|global aligned|globally-aligned)\b", "globally aligned", s)
    s = re.sub(r"\b(concept only|HA2 b1)\b", "expert aligned", s)
    s = re.sub(r"\b(epoch 9|ep9)\b", "final epoch", s)
    s = re.sub(r"group ([A-Z_]+)", lambda m: "group " + GROUPS.get(m.group(1), m.group(1).lower().replace("_", " ")), s)
    return re.sub(r"AUCINPH|auc_in|AUC_in|AUC in", "AUCINPH", s)
def tex_label(s): return tx(relabel(s)).replace("AUCINPH", AUC)
def fmt_p(v, boot=False):
    if pd.isna(v): return "--"
    if boot and v == 0: return f"$<{1 / B:g}$"
    if v < 1e-3: m, e = f"{v:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
def fmt_d(v):
    if pd.isna(v): return "--"
    r = round(float(v), 3); return "0.000" if r == 0 else (f"${r:.3f}$" if r < 0 else f"{r:.3f}")

ST = pd.read_csv(TAB / "appD_statistics.csv"); ST["section"] = ST.section.astype(str)
c = ST.comparison.astype(str)

# 1. by-class rows to 4.1
bc = c.str.contains(r"by class|unaligned, (melanomas|nevi)", case=False) & c.str.contains("unaligned") & \
     c.str.contains(r"auc", case=False) & ~ST.test.astype(str).str.contains("bootstrap", case=False)
assert bc.sum() == 2, f"STOP: found {int(bc.sum())} by-class rows:\n{ST[bc].comparison.tolist()}"
ST.loc[bc, "section"] = "4.1"
# 2. lambda 0.1 vs 5 rows to 4.3
l05 = c.str.contains(r"0\.1 vs 5") & (ST.n == 31)
assert l05.sum() == 2, f"STOP: found {int(l05.sum())} 'lambda 0.1 vs 5' rows:\n{ST[l05].comparison.tolist()}"
ST.loc[l05, "section"] = "4.3"
# 3. rename in the csv so a later export keeps it
ST["comparison"] = ST.comparison.map(lambda s: re.sub(r"\b(epoch 9|ep9)\b", "final epoch",
                                     re.sub(r"\bglobal aligned\b", "globally aligned",
                                     re.sub(r"by class,\s*NV\b", "nevi", re.sub(r"by class,\s*MEL\b", "melanomas", s)))))

key = lambda s: (1, 0) if s == "F" else (0, float(s))
ST["_o"] = range(len(ST)); ST = ST.sort_values(["section", "_o"], key=lambda col: col.map(key) if col.name == "section" else col)
ST = ST.drop(columns="_o").reset_index(drop=True)
cnt = {s: int(n) for s, n in ST.section.value_counts().items()}
print("rows per section:", {s: cnt.get(s, 0) for s in sorted(set(cnt) | set(EXPECT), key=key)}, "| total", len(ST))
if cnt != EXPECT or len(ST) != 232:
    for s in ["4.1", "4.3"]: print(f"\n{s}:\n  " + "\n  ".join(ST[ST.section == s].comparison))
    raise AssertionError("STOP: counts differ from expected, nothing written")

boot = ST.test.astype(str).str.contains("bootstrap", case=False)
rows = [f"{r.section} & {tex_label(r.comparison)} & {tex_label(METN.get(r.metric, r.metric))} & {fi(r.n)} & {fmt_d(r.median_diff)} & "
        f"{fmt_p(r.p, b)} & {fi(r.family)} & {fmt_p(r.threshold)} & {tx(r.survives)} \\\\" for r, b in zip(ST.itertuples(), boot)]
left = [l for l in rows if re.search(r"AUC\\?_in|AUCINPH|epoch 9|\bglobal aligned\b", l)]
assert not left, "STOP: leftovers, nothing written:\n" + "\n".join(left)

lines = TEXP.read_text().splitlines()
is_row = lambda l: "&" in l and l.rstrip().endswith("\\\\")
data = [i for i, l in enumerate(lines) if is_row(l) and "comparison" not in l.lower() and not l.lstrip().startswith("\\multicolumn")]
assert data == list(range(data[0], data[-1] + 1)), "STOP: data rows are not one block, nothing written"
TEXP.write_text("\n".join(lines[:data[0]] + rows + lines[data[-1] + 1:]) + "\n")
ST.to_csv(TAB / "appD_statistics.csv", index=False)
print(f"written: {len(rows)} rows, {int((ST.survives == 'yes').sum())} surviving | no leftovers found")


# ---- cell 169 ------------------------------------------------------------
import re
for f in ["tx", "fi", "METN"]: assert f in globals(), f"run the statistics export cell first ({f})"
TEXP, B = TAB / "appD_statistics.tex", 2000
EXPECT = {"4.1": 16, "4.2": 1, "4.3": 20, "4.4": 18, "4.5": 120, "4.6": 34, "4.7": 5, "F": 18}
AUC = r"$\mathrm{AUC}_{\mathrm{in}}$"
METRIC_TEX = {"auc_in": AUC, "mel_auc_in": AUC, "energy_in": r"$E_{\mathrm{in}}$", "mel_in": r"$E_{\mathrm{in}}$",
              "energy excess": r"$E_{\mathrm{in}}$", "mel_grid": r"$E_{\mathrm{grid}}$", "energy_grid": r"$E_{\mathrm{grid}}$",
              "nv_rest_in": r"$E_{\mathrm{in}}^{\mathrm{NV}}$", "corr_in": r"$r_{\mathrm{in}}$", "r_in": r"$r_{\mathrm{in}}$",
              "gt_loc": "lesion localisation", "localisation": "lesion localisation"}
def metric_tex(m):
    if m in METRIC_TEX: return METRIC_TEX[m]
    v = str(METN.get(m, m))
    return v if "$" in v or "\\" in v else tx(v)

GROUPS = {"UNASSIGNED": "unannotated tissue", "NETWORK": "network", "REGRESSION": "regression",
          "STRUCTURELESS": "structureless area", "STRUCTURELESS_AREA": "structureless area", "DOTS": "atypical dots",
          "ATYPICAL_DOTS": "atypical dots", "BLUE_WHITE": "blue-white structureless area", "BWSA": "blue-white structureless area",
          "BLUE_WHITE_STRUCTURELESS_AREA": "blue-white structureless area"}
def clean(s):
    s = str(s)
    s = re.sub(r"\b(lesion aligned|global aligned|globally-aligned)\b", "globally aligned", s)
    s = re.sub(r"\b(concept only|HA2 b1)\b", "expert aligned", s)
    s = re.sub(r"\b(epoch 9|ep9)\b", "final epoch", s)
    s = s.replace(", fold 0", "")
    s = re.sub(r"group ([A-Z_]+)", lambda m: "group " + GROUPS.get(m.group(1), m.group(1).lower().replace("_", " ")), s)
    s = re.sub(r"^reference (?:by class, MEL map|melanomas map), (.+) vs darkness, (MEL|NV|melanomas|nevi)$",
               lambda m: f"melanoma map, {m.group(1)} vs darkness, " + ("melanomas" if m.group(2) in ("MEL", "melanomas") else "nevi"), s)
    s = re.sub(r"^(?:AUC_in|auc_in|AUCINPH|\$\\mathrm\{AUC\}_\{\\mathrm\{in\}\}\$),\s*globally aligned vs unaligned,\s*(?:by class,\s*)?(MEL|NV|melanomas|nevi)$",
               lambda m: "globally aligned vs unaligned, " + ("melanomas" if m.group(1) in ("MEL", "melanomas") else "nevi"), s)
    return re.sub(r"AUCINPH|\bauc_in\b|\bAUC_in\b|\bAUC in\b", "AUCINPH", s)
def tex_label(s): return tx(s).replace("AUCINPH", AUC)
def fmt_p(v, boot=False):
    if pd.isna(v): return "--"
    if boot and v == 0: return f"$<{1 / B:g}$"
    if v < 1e-3: m, e = f"{v:.1e}".split("e"); return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
def fmt_d(v):
    if pd.isna(v): return "--"
    r = round(float(v), 3); return "0.000" if r == 0 else (f"${r:.3f}$" if r < 0 else f"{r:.3f}")

ST = pd.read_csv(TAB / "appD_statistics.csv"); ST["section"] = ST.section.astype(str)
ST["comparison"] = ST.comparison.map(clean)
mv = ST.comparison.str.contains(r"^globally aligned vs unaligned$|globally aligned vs unaligned$", regex=True) & (ST.n == 29) & (ST.family == 2)
assert mv.sum() == 2, f"STOP: found {int(mv.sum())} rows for 'globally aligned vs unaligned', n 29, family 2:\n{ST[mv].comparison.tolist()}"
ST.loc[mv, "section"] = "4.5"

key = lambda s: (1, 0) if s == "F" else (0, float(s))
ST["_o"] = range(len(ST)); ST["_k"] = ST.section.map(key)
ST = ST.sort_values(["_k", "_o"]).drop(columns=["_k", "_o"]).reset_index(drop=True)
cnt = {s: int(n) for s, n in ST.section.value_counts().items()}
print("rows per section:", {s: cnt.get(s, 0) for s in sorted(set(cnt) | set(EXPECT), key=key)}, "| total", len(ST))
if cnt != EXPECT or len(ST) != 232:
    for s in ["4.1", "4.5"]: print(f"\n{s}:\n  " + "\n  ".join(ST[ST.section == s].comparison))
    raise AssertionError("STOP: counts differ from expected, nothing written")

boot = ST.test.astype(str).str.contains("bootstrap", case=False)
rows = [f"{r.section} & {tex_label(r.comparison)} & {metric_tex(r.metric)} & {fi(r.n)} & {fmt_d(r.median_diff)} & "
        f"{fmt_p(r.p, b)} & {fi(r.family)} & {fmt_p(r.threshold)} & {tx(r.survives)} \\\\" for r, b in zip(ST.itertuples(), boot)]
bad = [l for l in rows if re.search(r"AUC\\?_in|AUCINPH|epoch 9|\bglobal aligned\b|, fold 0|reference melanomas|\\textbackslash|\\\$", l)]
assert not bad, "STOP: leftovers or escaped LaTeX, nothing written:\n" + "\n".join(bad[:10])

lines = TEXP.read_text().splitlines()
is_row = lambda l: "&" in l and l.rstrip().endswith("\\\\")
data = [i for i, l in enumerate(lines) if is_row(l) and "comparison" not in l.lower() and not l.lstrip().startswith("\\multicolumn")]
assert data == list(range(data[0], data[-1] + 1)), "STOP: data rows are not one block, nothing written"
head = [re.sub(r"\b[Cc]h\.", "Sec.", l) if "comparison" in l.lower() else l for l in lines[:data[0]]]
TEXP.write_text("\n".join(head + rows + lines[data[-1] + 1:]) + "\n")
ST.to_csv(TAB / "appD_statistics.csv", index=False)
print("header rows:", [l.strip()[:40] for l in head if "comparison" in l.lower()])
print("metric column:", {m: metric_tex(m) for m in ST.metric.unique()})
print(f"written: {len(rows)} rows, {int((ST.survives == 'yes').sum())} surviving | no leftovers")


# ---- cell 170 ------------------------------------------------------------
from PIL import Image as PILImage
DPI, WIN, HIN = 200, 9.0, 6.0                       # 9 x 6 inches at 200 dpi = 1800 x 1200 px
FS = 15                                             # 15 pt at 200 dpi = about 42 px
gap, side, top, bottom = 0.25, 0.35, 0.30, 0.95
asp = (Y1 - Y0) / (X1 - X0)
w = (WIN - 2 * side - gap) / 2; h = w * asp
if h > HIN - top - bottom: h = HIN - top - bottom; w = h / asp
x0 = (WIN - 2 * w - gap) / 2; y0 = bottom + (HIN - top - bottom - h) / 2
PANELS = [("Global alignment", cams.class_cams(M5, x)[0]), ("Expert alignment", cams.class_cams(me, x)[0])]
with plt.rc_context({"font.family": "Arial"}):
    fig = plt.figure(figsize=(WIN, HIN), dpi=DPI, facecolor="white")
    for k, (lab, v) in enumerate(PANELS):
        a = fig.add_axes([(x0 + k * (w + gap)) / WIN, y0 / HIN, w / WIN, h / HIN])
        overlay(a, rgb, v)
        o0(a, L, C_LESION, lw=2.2)
        o0(a, S, "black", lw=4.0); o0(a, S, C_SMEL, lw=2.4)
        a.set_xlim(X0, X1); a.set_ylim(Y1, Y0); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
        fig.text((x0 + k * (w + gap) + w / 2) / WIN, (y0 - 0.22) / HIN, lab, ha="center", va="top", fontsize=FS)
    fig.savefig(OUT / "fig_web_alignment.png", dpi=DPI, facecolor="white")
    fig.savefig(OUT / "fig_web_alignment.jpg", dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
    plt.show()
for ext in ("png", "jpg"):
    print(ext, PILImage.open(OUT / f"fig_web_alignment.{ext}").size)


# ---- cell 171 ------------------------------------------------------------
with plt.rc_context({"font.family": "Arial"}):
    fig = plt.figure(figsize=(WIN, HIN), dpi=DPI, facecolor="white")
    for k, (lab, v) in enumerate(PANELS):
        a = fig.add_axes([(x0 + k * (w + gap)) / WIN, y0 / HIN, w / WIN, h / HIN])
        overlay(a, rgb, v)
        a.set_xlim(X0, X1); a.set_ylim(Y1, Y0); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
        fig.text((x0 + k * (w + gap) + w / 2) / WIN, (y0 - 0.22) / HIN, lab, ha="center", va="top", fontsize=FS)
    fig.savefig(OUT / "fig_web_alignment.png", dpi=DPI, facecolor="white")
    fig.savefig(OUT / "fig_web_alignment.jpg", dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
    plt.show()
for ext in ("png", "jpg"):
    print(ext, PILImage.open(OUT / f"fig_web_alignment.{ext}").size)


# ---- cell 172 ------------------------------------------------------------
import matplotlib.patheffects as pe
LBLUE = "#4FC3F7"
LEVEL = 0.6                                          # fraction of each map's maximum
def hot_outline(ax, v):
    c = cv2.resize(cams.norm01(np.asarray(v, float).reshape(14, 14)), (224, 224), interpolation=cv2.INTER_CUBIC)
    cs = ax.contour(c, levels=[LEVEL], colors=[LBLUE], linewidths=3.0)
    for coll in getattr(cs, "collections", [cs]):
        coll.set_path_effects([pe.Stroke(linewidth=5.0, foreground="black", alpha=0.6), pe.Normal()])

with plt.rc_context({"font.family": "Arial"}):
    fig = plt.figure(figsize=(WIN, HIN), dpi=DPI, facecolor="white")
    for k, (lab, v) in enumerate(PANELS):
        a = fig.add_axes([(x0 + k * (w + gap)) / WIN, y0 / HIN, w / WIN, h / HIN])
        a.imshow(rgb); hot_outline(a, v)
        a.set_xlim(X0, X1); a.set_ylim(Y1, Y0); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
        fig.text((x0 + k * (w + gap) + w / 2) / WIN, (y0 - 0.22) / HIN, lab, ha="center", va="top", fontsize=FS)
    fig.savefig(OUT / "fig_web_alignment_outline.png", dpi=DPI, facecolor="white")
    fig.savefig(OUT / "fig_web_alignment_outline.jpg", dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
    plt.show()
for ext in ("png", "jpg"):
    print(ext, PILImage.open(OUT / f"fig_web_alignment_outline.{ext}").size)


# ---- cell 173 ------------------------------------------------------------
import json, re
NB = sorted((REPO / "notebooks/mel_nv").glob("1[3-6]*.ipynb"))
def cells(pat):
    for nb in NB:
        for c in json.load(open(nb))["cells"]:
            s = "".join(c["source"])
            if re.search(pat, s): yield nb.name, s
def show(tag, pat_cell, pat_line, n=25):
    for nm, s in cells(pat_cell):
        hits = [l.strip() for l in s.splitlines() if re.search(pat_line, l)]
        if hits: print(f"--- {tag} | {nm}"); print("\n".join(hits[:n]))

# 1. target class of FinerCAM and Diff-CAM on the study pages
show("1", r"clinician_study|KEY_do_not_send", r"finer_cam\(|diff_cam\(|target_ref\(|class_cams\(|gt_label|pred")

# 2. empty target loss, normalised or raw map
src = (REPO / "external/PanDerm/classification/run_class_finetuning_ha.py").read_text()
for kw in ["empty", "feat_loss", "def .*feat", "norm01", "min()", "amin"]:
    for m in re.finditer(kw, src):
        i = src.rfind("\n", 0, m.start()); print(f"--- 2 [{kw}]", src[i:i + 240].strip().splitlines()[0][:200])

# 4. lesion tracking: uniform condition and inpainted background patch
show("4", r"uniform|background patch|bg_patch", r"uniform|patch|fit|sample\(|len\(", n=30)

# 5. deletion v2: size of the structure subset and the unannotated subset
show("5", r"clinician \(n\)|unannotated \(n\)|deletion_v2", r"clinician|unannot|choice\(|sample|min\(|n_", n=30)

# 6. annotated images missing from the fold files
ANN = sorted(set(QC.Image_ID.astype(str)))
F = set(pd.read_csv(MELNV / "f2_fold0.csv").image_id.astype(str))
H0 = pd.read_csv(MELNV / "ham_mel_nv_clean.csv"); H0["image_id"] = H0.image_id.astype(str)
miss = [i for i in ANN if i not in F]
print("--- 6 | annotated:", len(ANN), "| missing from fold files:", miss)
for i in miss:
    h = H0[H0.image_id == i]
    lid = h.lesion_id.iloc[0] if len(h) else "not in clean csv"
    sib = H0[(H0.lesion_id == lid) & (H0.image_id != i)][["image_id", "split"]].values.tolist() if len(h) else []
    print(f"   {i}: gt {h.gt_label.iloc[0] if len(h) else '?'}, split {h.split.iloc[0] if len(h) else '?'}, lesion {lid}, siblings {sib}")
show("6", r"f2_fold|has_feat", r"drop|exclude|lesion_id|sibling|isin|~", n=20)


# ---- cell 174 ------------------------------------------------------------
import re
src = (REPO / "external/PanDerm/classification/run_class_finetuning_ha.py").read_text()
for kw in ["interpolate", "avg_pool", "adaptive_avg", "resize", "14", ">= 0.5", "> 0.5", "nearest"]:
    for m in re.finditer(re.escape(kw), src):
        i = src.rfind("\n", 0, m.start()); print(f"[{kw}]", src[i:i + 200].strip().splitlines()[0][:180]); break


# ---- cell 175 ------------------------------------------------------------
import importlib
from src.thesis import data, models, cams, metrics
for nm in ["data", "models", "cams", "metrics"]:
    assert hasattr(globals()[nm], "__file__"), f"{nm} is still not a module"
print("modules restored:", data.__name__, models.__name__, cams.__name__, metrics.__name__)


# ---- cell 176 ------------------------------------------------------------
import json, re
from sklearn.metrics import average_precision_score
T = TEST_DF.assign(image_id=TEST_DF.image_id.astype(str)).set_index("image_id")

# 1. AP on the 300 random test images, final epoch
CMP = pd.read_csv(TAB / "ch17_test_expert_vs_global.csv"); CMP["image_id"] = CMP.image_id.astype(str)
ids = sorted(CMP.image_id.unique()); y = (T.loc[ids, "gt_label"] == "MEL").astype(int).values
X3 = torch.cat([data.to_tensor(data.native(T.loc[i, "image_rel_path"])) for i in ids])
EP9 = {"global": REPO / "external/checkpoints_f2/ha1only_ep9/fold0.pth", "expert": REPO / "external/checkpoints_f2/ha2b1_ep9/fold0.pth"}
rng = np.random.default_rng(SEED); B = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
for k, p in EP9.items():
    m = models.load_ckpt(p, cache=False); pr = np.asarray(models.predict_mel(m, X3)); del m
    bs = [average_precision_score(y[b], pr[b]) for b in B if 0 < y[b].sum() < len(b)]
    print(f"1. {k}: AP {average_precision_score(y, pr):.3f} (bootstrap SD {np.std(bs):.3f}) | images {len(ids)}, melanomas {int(y.sum())}")

# 2. why random subsets, and why 299 against 300
S300 = set(TEST_DF.sample(300, random_state=SEED).image_id.astype(str))
print(f"2. comparison images {len(ids)} | same draw as the sweep subset: {set(ids) == S300} | ISIC_0027211 in sweep subset: {'ISIC_0027211' in S300}")
for nb in sorted((REPO / "notebooks/mel_nv").glob("1[5-6]*.ipynb")):
    for c in json.load(open(nb))["cells"]:
        s = "".join(c["source"])
        if re.search(r"ch17_test_expert_vs_global|ch5_lambda_localisation", s):
            print(f"--- 2 | {nb.name}"); print("\n".join(l.strip() for l in s.splitlines() if re.search(r"sample\(|continue|L\.sum|to_csv", l))[:800])

# 6. unaligned model, AUC_in on melanomas across gamma
mel31 = [i for i in sorted(HUM) if EV.loc[i].gt_label == "MEL"]; rows = []
for iid in mel31:
    x = data.to_tensor(data.native(EV.loc[iid].image_rel_path)); L, U = LESION[iid], HUM[iid]
    for g in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        v = np.asarray(cams.finer_cam(M0, x, "MEL", g), float).reshape(14, 14)
        rows.append({"gamma": g, "auc_in": metrics.auc_in(v, L, U) if v[L].std() > 1e-9 else np.nan})
G0 = pd.DataFrame(rows).groupby("gamma").auc_in.median()
print("6. unaligned, melanomas, median AUC_in by gamma:", G0.round(3).to_dict(), f"| range {G0.min():.3f} to {G0.max():.3f}")

# 9. round 2 selection rule
for nb in sorted((REPO / "notebooks/mel_nv").glob("1[5-6]*.ipynb")):
    for c in json.load(open(nb))["cells"]:
        s = "".join(c["source"])
        if re.search(r"round2|round 2", s, flags=re.I) and re.search(r"gt_label|pred|sample|select", s):
            print(f"--- 9 | {nb.name}"); print("\n".join(l.strip() for l in s.splitlines() if re.search(r"gt_label|pred|sample|select|held|MEL", l))[:1200])


# ---- cell 177 ------------------------------------------------------------
import re, json
for f in ["K1", "ANS1", "role", "scores", "lesion"]: assert f in globals(), f"rerun the decoding cell first ({f})"
H0 = pd.read_csv(MELNV / "ham_mel_nv_clean.csv"); H0["image_id"] = H0.image_id.astype(str); H0 = H0.drop_duplicates("image_id").set_index("image_id")

# A1. round 1 lesion metrics on the maps of the predicted class, as shown on the pages
def map_pred(method, iid, cls):
    mdl, cam = role(method); m = M0 if mdl == "unaligned" else M5
    x = data.to_tensor(data.native(H0.loc[iid, "image_rel_path"]))
    if cam == "FinerCAM": return np.asarray(cams.finer_cam(m, x, cls, 0.6), float).reshape(14, 14)
    if cam == "Diff-CAM": return np.asarray(cams.diff_cam(m, x, cls), float).reshape(14, 14)
    return np.asarray(cams.target_ref(m, x, cls)[0], float).reshape(14, 14)
rows = []
for c in range(1, 21):
    sub = K1[K1.case_no == c]; ch = sub[sub.letter == ANS1[c]].iloc[0]; iid = str(ch.image_id); pred = str(ch["pred"]); L = lesion(iid)
    sc = {r.method: scores(map_pred(r.method, iid, pred), L) for r in sub.itertuples()}
    best = lambda j: max(sc, key=lambda k: -np.inf if np.isnan(sc[k][j]) else sc[k][j])
    rows.append({"part": 1 if c <= 10 else 2, "case": c, "chosen": ch.method, **{f"loc|{k}": v[0] for k, v in sc.items()},
                 **{f"share|{k}": v[1] for k, v in sc.items()}, "top_loc": ch.method == best(0), "top_share": ch.method == best(1)})
P1 = pd.DataFrame(rows); P1.to_csv(TAB / "ch4_clinician_r1_metrics_pred.csv", index=False)
for p in (1, 2):
    d = P1[P1.part == p]; cols = [c for c in d.columns if c.startswith(("loc|", "share|")) and d[c].notna().any()]
    print(f"A1 part {p}: " + ", ".join(f"{c} {d[c].median():.3f}" for c in cols) +
          f" | chosen highest localisation {int(d.top_loc.sum())}/10, highest heat share {int(d.top_share.sum())}/10")

# A2. empty target loss
src = (REPO / "external/PanDerm/classification/run_class_finetuning_ha.py").read_text()
for fn in ["def feature_channel_loss", "def compute_feature_loss"]:
    i = src.find(fn); print(f"--- A2 {fn}\n" + src[i:i + 900])

# A4. background patch control
VC = pd.read_csv(TAB / "verify_controls.csv"); gc = next((c for c in ["model", "ckpt", "name"] if c in VC.columns), None)
print("A4 images:", VC.image_id.nunique() if "image_id" in VC else len(VC), "| patch placed:",
      VC.groupby(gc).bp_patch.apply(lambda s: int(s.notna().sum())).to_dict() if gc else int(VC.bp_patch.notna().sum()))
for c in json.load(open(REPO / "notebooks/mel_nv/16g_verify.ipynb"))["cells"]:
    s = "".join(c["source"])
    if "def place_patch" in s: i = s.find("def place_patch"); print("--- A4 place_patch\n" + "\n".join(s[i:].splitlines()[:18])); break

# A5. deletion subset size per held-out image
ku = np.array([min(int((HUM[i] & LESION[i]).sum()), int((LESION[i] & ~HUM[i]).sum())) for i in held])
full = np.array([int((HUM[i] & LESION[i]).sum()) for i in held])
print(f"A5 cells per subset: median {np.median(ku):.0f}, range {ku.min()} to {ku.max()} | subset is all structure cells on {int((ku == full).sum())} of {len(held)} images")

# A6. the two excluded annotated images
raw = pd.read_csv(QC_CSV); raw["Image_ID"] = raw.Image_ID.astype(str)
FOLDS = set().union(*[set(pd.read_csv(MELNV / f).image_id.astype(str)) for f in ["f2_fold0.csv", "f2_fold0_heldout.csv"]])
for i in ["ISIC_0032143", "ISIC_0032462"]:
    print(f"A6 {i}: in fold files {i in FOLDS}")
    print(raw[raw.Image_ID == i][["label", "label_group", "qc_pass", "n_patch_50"]].to_string(index=False))


# ---- cell 178 ------------------------------------------------------------
import json, re
from sklearn.metrics import roc_auc_score, average_precision_score
NBS = sorted((REPO / "notebooks/mel_nv").glob("1[2-6]*.ipynb"))
def grep(tag, cell_pat, line_pat, n=14):
    for nb in NBS:
        for c in json.load(open(nb))["cells"]:
            s = "".join(c["source"])
            if re.search(cell_pat, s, flags=re.I):
                h = [l.strip() for l in s.splitlines() if re.search(line_pat, l, flags=re.I)]
                if h: print(f"--- {tag} | {nb.name}"); print("\n".join(h[:n]))

# 1, 2 continuation runs with the class separation term
grep("1-2", r"continu|sep_lambda|class separation|dal", r"sep|dal|share|ratio|auc_in|0\.001|lambda.*2\b|to_csv")
# 3 CLS performance on the 986 test images
for nm, m in [("CLS unaligned", MC0), ("CLS globally aligned", MC5)]:
    p = np.asarray(models.predict_mel(m, X_TEST)); print(f"3. {nm}: AUC {roc_auc_score(Y_MEL, p):.3f}, AP {average_precision_score(Y_MEL, p):.3f}, n {len(p)}")
# 5 empty unaligned map in the single step test
DL = pd.read_csv(TAB / "ch5_deletion_v2.csv"); print("5. rows per model:", DL.groupby("model").image_id.nunique().to_dict())
grep("5", r"deletion_v2", r"max\(\)|1e-6|empty|continue|nan|skip")
# 6 main model vs sweep run at lambda 5
M6 = pd.read_csv(TAB / "ch6_metrics.csv").merge(EV[["gt_label"]], left_on="image_id", right_index=True); g6 = M6[M6.method == "Grad-CAM"]
print(f"6. ch6_metrics Grad-CAM (main model): pooled {g6.auc_in.median():.3f}, nevi {g6[g6.gt_label == 'NV'].auc_in.median():.3f}")
LAMC = pd.read_csv(TAB / "ch5_lambda_collapse.csv"); s5 = LAMC[LAMC["lambda"] == 5.0]
for c in [c for c in s5.columns if "auc" in c.lower()]:
    s = s5.merge(EV[["gt_label"]], left_on="image_id", right_index=True)
    print(f"6. sweep lambda 5, {c}: pooled {s[c].median():.3f}, nevi {s[s.gt_label == 'NV'][c].median():.3f}")
grep("6", r"r-agree-class|agree_class|app-methods-class|methods_class", r"read_csv|\.csv")
# 7 lesion tracking: 76 pixels, 0.062, Spearman -0.21
grep("7", r"shift|spearman", r"76|shift|spearman|median|sample\(|n=")
# 8 structure group ranking
grep("8", r"group.*energy|normalis|normaliz", r"normal|/ |area|excess")
# 9 tab:app-nevus family
ST = pd.read_csv(TAB / "appD_statistics.csv")
nv = ST[ST.comparison.str.contains("NV map", na=False)]; print("9.", nv[["comparison", "family", "threshold"]].to_string(index=False))
# 10 sweep tests
sw = ST[ST.test.astype(str).str.contains("bootstrap", na=False)]; print("10.", sw[["comparison", "metric", "p", "family"]].to_string(index=False))


# ---- cell 179 ------------------------------------------------------------
import json, re, inspect
# 1. loss share of the class separation term, weight 0.001, no lesion term
logs = [p for p in REPO.rglob("log.txt") if re.search(r"dal0?[p.]?0*01\b|dal0\.001|dal0p001", str(p)) and re.search(r"ha0[^.1-9]|ha0\.0|ha0p0", str(p))]
logs += [p for p in (CK5 / "dal_logs").glob("*.log.txt") if re.search(r"ha0[p.]?0?-dal0p001|ha0-dal0p001", p.stem)] if (CK5 / "dal_logs").exists() else []
print("1. logs:", [str(p.relative_to(REPO)) for p in logs])
for p in logs[:2]:
    last = [json.loads(l) for l in open(p) if l.strip().startswith("{")][-1]
    print("   last epoch:", {k: round(v, 5) for k, v in last.items() if "loss" in k and isinstance(v, (int, float))})

# 2. weight 2, no lesion term: AUC_in by class
DAL = sorted((REPO / "external").rglob("*dal*.pth"))
pick = [p for p in DAL if (a := models.ckpt_args(p)) and float(a.get("ha_lambda") or 0) == 0 and float(a.get("dal_lambda") or 0) == 2.0]
print("2. checkpoint:", [p.name for p in pick])
if pick:
    m = models.load_ckpt(pick[0], cache=False); rows = []
    for iid in sorted(HUM):
        r = EV.loc[iid]; t = np.asarray(cams.target_ref(m, data.to_tensor(data.native(r.image_rel_path)), r.gt_label)[0], float).reshape(14, 14)
        rows.append({"gt": r.gt_label, "auc": metrics.auc_in(t, LESION[iid], HUM[iid]) if t[LESION[iid]].std() > 1e-9 else np.nan})
    del m; d = pd.DataFrame(rows); print("   AUC_in median:", d.groupby("gt").auc.median().round(3).to_dict(), "| pooled", round(d.auc.median(), 3))

# 5. single step test without the empty unaligned map
DL = pd.read_csv(TAB / "ch5_deletion_v2.csv"); u = DL[DL.model == "unaligned"].set_index(DL[DL.model == "unaligned"].image_id.astype(str))
print("5. ISIC_0026706 row:", u.loc["ISIC_0026706", ["top 10%", "random 10%"]].round(3).to_dict() if "ISIC_0026706" in u.index else "absent")
for nm, s in [("all 29", u), ("without empty map", u.drop("ISIC_0026706", errors="ignore"))]:
    print(f"   {nm}: {metrics.paired(s['top 10%'], s['random 10%'])}")

# 7. lesion tracking: shift distance, 0.062, Spearman -0.21
for c in json.load(open(REPO / "notebooks/mel_nv/16g_verify.ipynb"))["cells"]:
    s = "".join(c["source"])
    if "def shift(img, m)" in s: i = s.find("def shift(img, m)"); print("--- 7 shift\n" + "\n".join(s[i:].splitlines()[:8])); break
RB = pd.read_csv(TAB / "app_robustness.csv")
print("7. app_robustness medians:\n", RB.groupby(["cond", "model"])[["share", "area"]].median().round(3).to_string())
for f in ["app_inpaint_contrast.csv", "app_shift_mirror.csv"]:
    if (TAB / f).exists():
        d = pd.read_csv(TAB / f); print(f"7. {f}: n {len(d)}, columns {list(d.columns)}")
        print("   medians:", d.median(numeric_only=True).round(3).to_dict())
IP = pd.read_csv(TAB / "app_inpaint_contrast.csv")
from scipy.stats import spearmanr
print(f"7. heat share vs contrast: Spearman {spearmanr(IP.share, IP.contrast).statistic:.3f}, n {len(IP)}")


# ---- cell 180 ------------------------------------------------------------
d = [np.hypot(*[min(223 - (np.where(m)[a]).max(), 60) if 223 - np.where(m)[a].max() > np.where(m)[a].min() else min(np.where(m)[a].min(), 60) for a in (1, 0)])
     for m in (mask224(r.mask_rel_path) for _, r in TEST_DF.sample(100, random_state=SEED).iterrows()) if m.any()]
print(f"shift distance, 224 x 224 input: median {np.median(d):.1f} px, range {min(d):.1f} to {max(d):.1f}")
