"""Converted from notebooks/mel_nv/16c_explanations.ipynb by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python scripts/analysis/explanations.py
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

EV = data.load_eval()
LESION = data.load_lesions(EV)
QC = data.load_qc()
HUM = data.load_hum(QC, LESION)
X_TEST, Y_MEL, ID_TEST, TEST_DF = data.load_test()
M0, M5 = models.ck5("ha0"), models.ck5("ha5")

AUD = audit.run_audit(EV, LESION, HUM, X_TEST, Y_MEL, ID_TEST, QC, M0, M5)


# ---- cell 1 ------------------------------------------------------------
import matplotlib.pyplot as plt

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
rows = []
for mname, m in [("unaligned", M0), ("aligned", M5)]:
    for iid in sorted(HUM):
        r = EV.loc[iid]
        x = data.to_tensor(data.native(r.image_rel_path))
        L, U = LESION[iid], HUM[iid]
        for a in ALPHAS:
            fm = cams.finer_cam(m, x, "MEL", a)
            fn = cams.finer_cam(m, x, "NV", a)
            dead = fm[L].max() < 1e-6
            rows.append({"model": mname, "alpha": a, "image_id": iid, "gt": r.gt_label,
                         "mel_on_mel": np.nan if dead else metrics.energy_excess(fm, L, U),
                         "nv_on_rest": (metrics.energy_excess(fn, L, L & ~U)
                                        if r.gt_label == "NV" and fn[L].max() > 1e-6 else np.nan),
                         "corr": metrics.corr_in(fm, fn, L),
                         "mel_mass_in_lesion": float(fm[L].sum() / (fm.sum() + 1e-8)),
                         "dead": dead})
FA = pd.DataFrame(rows)
FA.to_csv(TAB / "finercam_alpha_sweep.csv", index=False)

S = FA.groupby(["model", "alpha"]).agg(mel_on_mel=("mel_on_mel", "median"),
                                       nv_on_rest=("nv_on_rest", "median"),
                                       corr=("corr", "median"),
                                       mass=("mel_mass_in_lesion", "median"),
                                       dead=("dead", "sum")).round(3)
print(S.to_string())

for mname in ["unaligned", "aligned"]:
    q = FA[FA.model == mname].pivot_table(index="image_id", columns="alpha", values="mel_on_mel")
    for a in ALPHAS[1:]:
        res = metrics.paired(q[a], q[0.0])
        print(f"  {mname:9} alpha {a:.1f} vs 0  MEL map  {res['diff']:+.3f}  p {res['p']:.3f}")

def pareto(df):
    keep = []
    for i, r in df.iterrows():
        dom = ((df["corr"] <= r["corr"]) & (df.mel_on_mel >= r.mel_on_mel) &
               ((df["corr"] < r["corr"]) | (df.mel_on_mel > r.mel_on_mel))).any()
        keep.append(not dom)
    return df[keep]

fig, ax = plt.subplots(1, 2, figsize=(plotting.TEXTWIDTH, 2.8))
for j, (col, ttl) in enumerate([("mel_on_mel", "MEL map, melanoma structures"),
                               ("nv_on_rest", "NV map, lesion minus structures (nevi)")]):
    for mname, c in [("unaligned", plotting.C_CONTEXT), ("aligned", plotting.C_FOCUS)]:
        s = S.loc[mname]
        ax[j].plot(s["corr"], s[col], "o-", color=c, label=mname, lw=1.5, ms=4)
        for a in ALPHAS:
            ax[j].annotate(f"{a:g}", (s.loc[a, "corr"], s.loc[a, col]),
                           xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax[j].axhline(0, color=plotting.C_REF, ls=":", lw=1)
    ax[j].set(xlabel="MEL vs NV map correlation in lesion", title=ttl)
ax[0].set_ylabel("energy excess")
ax[0].legend(frameon=False)
fig.tight_layout()
plotting.save(fig, "finercam_alpha_pareto")
plt.show()
print("\nPareto optimal alphas, aligned:", pareto(S.loc["aligned"].reset_index()).alpha.tolist())


# ---- cell 2 ------------------------------------------------------------
def energy_grid(v, T):
    """Heat on target over ALL heat, minus target share of the full grid.
    Not conditional on the lesion. Reported as a complement, not a new primary."""
    v = metrics.norm01(v) if hasattr(metrics, "norm01") else cams.norm01(v)
    return float(v[T].sum() / (v.sum() + 1e-8) - T.mean())

rows = []
for mname, m in [("unaligned", M0), ("aligned", M5)]:
    for iid in sorted(HUM):
        r = EV.loc[iid]
        x = data.to_tensor(data.native(r.image_rel_path))
        L, U = LESION[iid], HUM[iid]
        mel, nv = cams.class_cams(m, x)
        rows.append({"model": mname, "image_id": iid, "gt": r.gt_label,
                     "in_lesion": metrics.energy_excess(mel, L, U),
                     "whole_grid": energy_grid(mel, U & L),
                     "auc_in": metrics.auc_in(mel, L, U),
                     "nv_in_lesion": (metrics.energy_excess(nv, L, L & ~U)
                                      if r.gt_label == "NV" else np.nan),
                     "nv_whole_grid": (energy_grid(nv, L & ~U)
                                       if r.gt_label == "NV" else np.nan)})
UA = pd.DataFrame(rows)
cols = ["in_lesion", "whole_grid", "auc_in", "nv_in_lesion", "nv_whole_grid"]
print(UA.groupby("model")[cols].median().round(3).to_string())
for c in cols:
    q = UA.pivot_table(index="image_id", columns="model", values=c)
    res = metrics.paired(q["aligned"], q["unaligned"])
    print(f"  aligned minus unaligned  {c:14} {res['diff']:+.3f}  p {res['p']:.3f}  n {res['n']}")


# ---- cell 3 ------------------------------------------------------------
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr, wilcoxon
rng = np.random.default_rng(SEED)

# ---- 6.1 metric table on HUM -------------------------------------------------
def egrid(v, T):
    v = cams.norm01(v)
    return float(v[T].sum() / (v.sum() + 1e-8) - T.mean())

rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    L, U = LESION[iid], HUM[iid]
    S = data.group_masks(QC, iid, L)
    S = [m & U for m in S.values() if (m & U).any()] or [U]
    maps = {"Grad-CAM": cams.target_ref(M5, x, r.gt_label)[0],
            "Diff-CAM": cams.diff_cam(M5, x, r.gt_label),
            "FinerCAM 0.6": cams.finer_cam(M5, x, r.gt_label, 0.6),
            "centre": metrics.centre_map(L),
            "random": rng.random(L.shape)}
    for k, v in maps.items():
        hot = metrics.top_mask(np.where(L, cams.norm01(v), -1), frac=20 / 196)
        pk = np.unravel_index(np.argmax(np.where(L, v, -1)), v.shape)
        rows.append({"method": k, "image_id": iid,
                     "auc_in": metrics.auc_in(v, L, U),
                     "energy_in": metrics.energy_excess(v, L, U),
                     "energy_grid": egrid(v, U & L),
                     "topk_exc": metrics.topk_excess(v, L, U),
                     "struct_hit": float(np.mean([(hot & s).any() for s in S])),
                     "peak_dist": float(distance_transform_edt(~U)[pk]),
                     "in_lesion": float(cams.norm01(v)[L].sum() / (cams.norm01(v).sum() + 1e-8))})
M61 = pd.DataFrame(rows)
M61.to_csv(TAB / "ch6_metrics.csv", index=False)
ORD = ["Grad-CAM", "Diff-CAM", "FinerCAM 0.6", "centre", "random"]
print(M61.groupby("method").median(numeric_only=True).reindex(ORD).round(3).to_string())
for k in ORD[1:]:
    q = M61.pivot_table(index="image_id", columns="method", values="auc_in")
    print(f"  {k:13} minus Grad-CAM, auc_in  {metrics.paired(q[k], q['Grad-CAM'])}")

# ---- 6.3 common subset at gamma 1 --------------------------------------------
GAM = [0.0, 0.4, 0.6, 1.0]
rows = []
for name, m in [("unaligned", M0), ("aligned", M5)]:
    for iid in sorted(HUM):
        r = EV.loc[iid]
        x = data.to_tensor(data.native(r.image_rel_path))
        for g in GAM:
            v = cams.finer_cam(m, x, r.gt_label, g)
            rows.append({"model": name, "gamma": g, "image_id": iid,
                         "auc_in": metrics.auc_in(v, LESION[iid], HUM[iid])})
FG = pd.DataFrame(rows)
alive = FG.groupby("image_id").auc_in.apply(lambda s: s.notna().all())
sub = FG[FG.image_id.isin(alive[alive].index)]
print(f"\ncommon subset n = {sub.image_id.nunique()}")
print(sub.pivot_table(index="model", columns="gamma", values="auc_in", aggfunc="median").round(3).to_string())

# ---- 6.4 excess per morphology group -----------------------------------------
rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]
    L = LESION[iid]
    t = cams.target_ref(M5, data.to_tensor(data.native(r.image_rel_path)), r.gt_label)[0]
    idx = np.flatnonzero(L)
    k = min(20, len(idx))
    hot = idx[np.argsort(-t.ravel()[idx])[:k]]
    G = data.group_masks(QC, iid, L)
    U_all = np.zeros_like(L)
    for m in G.values():
        U_all |= m
    G["UNASSIGNED"] = L & ~U_all
    for g, m in G.items():
        if m.sum() == 0:
            continue
        share = m.ravel()[hot].mean()
        null = np.mean([m.ravel()[rng.choice(idx, k, replace=False)].mean() for _ in range(2000)])
        rows.append({"image_id": iid, "group": g, "share": share, "null": null,
                     "excess": share - null, "patches": int(m.sum())})
GX = pd.DataFrame(rows)
GX.to_csv(TAB / "ch6_group_excess.csv", index=False)
groups = GX.group.unique()
print("\nexcess per group, Wilcoxon vs 0, Bonferroni over", len(groups))
for g in sorted(groups, key=lambda g: -GX[GX.group == g].image_id.nunique()):
    s = GX[GX.group == g].excess
    p = wilcoxon(s).pvalue if len(s) >= 5 and s.abs().sum() > 0 else np.nan
    print(f"  {g:14} n {len(s):2}  median {s.median():+.3f}  p {p:.4f}  "
          f"p_bonf {min(1, p * len(groups)):.4f}")


# ---- cell 4 ------------------------------------------------------------
GAM = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
rows = []
for name, m in [("unaligned", M0), ("aligned", M5)]:
    for iid in sorted(HUM):
        r = EV.loc[iid]
        x = data.to_tensor(data.native(r.image_rel_path))
        L, U = LESION[iid], HUM[iid]
        for g in GAM:
            v = cams.finer_cam(m, x, r.gt_label, g)
            a = metrics.auc_in(v, L, U)
            dead = bool(v[L].max() < 1e-6 or np.isnan(a))
            rows.append({"model": name, "gamma": g, "image_id": iid,
                         "auc_raw": a, "auc": 0.5 if dead else a, "dead": dead})
FG = pd.DataFrame(rows)
FG.to_csv(TAB / "ch6_finercam_gamma_auc.csv", index=False)
print("median in-lesion AUC, empty map = 0.5, n = 31")
print(FG.pivot_table(index="model", columns="gamma", values="auc", aggfunc="median").round(3).to_string())
print("\nempty maps")
print(FG.pivot_table(index="model", columns="gamma", values="dead", aggfunc="sum").to_string())
print("\naligned minus unaligned, per gamma")
for g in GAM:
    q = FG[FG.gamma == g].pivot_table(index="image_id", columns="model", values="auc")
    print(f"  gamma {g:.1f}  {metrics.paired(q['aligned'], q['unaligned'])}")
print("\naligned, gamma vs 0")
q = FG[FG.model == "aligned"].pivot_table(index="image_id", columns="gamma", values="auc")
for g in GAM[1:]:
    print(f"  gamma {g:.1f}  {metrics.paired(q[g], q[0.0])}")


# ---- cell 5 ------------------------------------------------------------
from scipy.ndimage import distance_transform_edt
NET = ["atypical_network", "regular_network", "negative_network"]

def label_mask(iid, lab, L):
    sub = QC[(QC.Image_ID == iid) & (QC.label == lab)]
    if not len(sub):
        return None
    acc = None
    for p in sub.mask_path:
        m = data._full(p)
        acc = m if acc is None else (acc | m)
    return data.to_grid(acc) & L

def top_idx(v, L, k=20):
    idx = np.flatnonzero(L)
    return idx[np.argsort(-cams.norm01(v).ravel()[idx])[:min(k, len(idx))]]

rows_net, rows_win, rows_depth = [], [], []
for iid in sorted(HUM):
    r = EV.loc[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    L, U = LESION[iid], HUM[iid]
    gc = cams.target_ref(M5, x, r.gt_label)[0]
    fc = cams.finer_cam(M5, x, r.gt_label, 1.0)
    hot = top_idx(gc, L)
    for lab in NET:
        m = label_mask(iid, lab, L)
        if m is not None and m.sum():
            rows_net.append({"label": lab, "image_id": iid,
                             "excess": m.ravel()[hot].mean() - m[L].mean()})
    G = data.group_masks(QC, iid, L)
    if len(G) >= 2:
        win = {}
        for nm, v in [("gradcam", gc), ("finer1", fc)]:
            h = top_idx(v, L)
            win[nm] = max(G, key=lambda g: G[g].ravel()[h].mean() - G[g][L].mean())
        rows_win.append({"image_id": iid, **win})
    d = distance_transform_edt(L)
    d = d / d[L].max()
    rows_depth.append({"image_id": iid, "depth_annot": d[U & L].mean(),
                       "depth_unannot": d[L & ~U].mean() if (L & ~U).any() else np.nan})

NX = pd.DataFrame(rows_net)
print("NETWORK split by label, excess of top-20 Grad-CAM cells over area share")
for lab, s in NX.groupby("label").excess:
    p = wilcoxon(s).pvalue if len(s) >= 5 and s.abs().sum() > 0 else np.nan
    print(f"  {lab:18} n {len(s):2}  median {s.median():+.3f}  p {p:.4f}")

W = pd.DataFrame(rows_win)
print(f"\nwinning group unchanged, Grad-CAM vs FinerCAM 1.0: {(W.gradcam == W.finer1).sum()} of {len(W)}")

DP = pd.DataFrame(rows_depth)
print(f"\nnormalised depth, 0 = edge, 1 = deepest lesion cell")
print(f"  annotated   median {DP.depth_annot.median():.3f}  mean {DP.depth_annot.mean():.3f}")
print(f"  unannotated median {DP.depth_unannot.median():.3f}  mean {DP.depth_unannot.mean():.3f}")
print(f"  paired {metrics.paired(DP.depth_annot, DP.depth_unannot)}")

ls = pd.Series({i: LESION[i].mean() for i in HUM})
dc = M61[M61.method == "Diff-CAM"].set_index("image_id").in_lesion
print(f"\nDiff-CAM heat in lesion vs lesion share: median {dc.median():.3f} vs {ls.median():.3f}, "
      f"{metrics.paired(dc, ls.reindex(dc.index))}")


# ---- cell 6 ------------------------------------------------------------
from scipy.stats import spearmanr
real = GX[GX.group != "UNASSIGNED"]
print("Spearman size vs share:", spearmanr(real.patches, real.share).statistic.round(3),
      " n pairs", len(real))
w_share = real.loc[real.groupby("image_id").share.idxmax()].set_index("image_id").group
w_exc = real.loc[real.groupby("image_id").excess.idxmax()].set_index("image_id").group
w_big = real.loc[real.groupby("image_id").patches.idxmax()].set_index("image_id").group
multi = real.groupby("image_id").group.nunique()
ids = multi[multi >= 2].index
print(f"images with >= 2 groups: {len(ids)}")
print(f"winner by share != winner by excess: {(w_share[ids] != w_exc[ids]).sum()}")
print("STRUCTURELESS wins by share:", (w_share[ids] == "STRUCTURELESS").sum(),
      " by excess:", (w_exc[ids] == "STRUCTURELESS").sum())
exp = sum(1 / multi[i] for i in ids)
print(f"excess winner is the largest group: {(w_exc[ids] == w_big[ids]).sum()} "
      f"of {len(ids)}, expected by chance {exp:.1f}")
print(GX[(GX.image_id == "ISIC_0029698")][["group", "share", "null", "excess"]].round(3).to_string())

# ---- control run check -------------------------------------------------------
ctrl = [p for p in CK_F2.glob("*.pth")
        if models.ckpt_args(p).get("feat_lambda", 1) == 0
        and 0 < models.ckpt_args(p).get("ha_lambda", 0) < 1e-4]
print(f"\ncontrol checkpoints (ha ~ 0, beta 0): {len(ctrl)} of 5")
for p in sorted(ctrl):
    a = models.ckpt_args(p)
    print(f"  fold {a.get('fold')}  ha {a.get('ha_lambda')}  batch {a.get('batch_size')}  "
          f"epochs {a.get('epochs')}  {p.name[:60]}")


# ---- cell 7 ------------------------------------------------------------
from sklearn.metrics import precision_recall_curve, f1_score
for name, m in [("unaligned", M0), ("aligned", M5)]:
    p = models.predict_mel(M5 if name == "aligned" else M0, X_TEST)
    pr, rc, th = precision_recall_curve(Y_MEL, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = np.nanargmax(f1[:-1])
    print(f"{name:9}  F1 at 0.5 {f1_score(Y_MEL, p >= 0.5):.3f}   "
          f"best F1 {f1[i]:.3f} at threshold {th[i]:.2f}")


# ---- cell 8 ------------------------------------------------------------
d = distance_transform_edt(L)
d = d / d[L].max()


# ---- cell 9 ------------------------------------------------------------
print(f"6. annotated median {DP.depth_annot.median():.3f} | unannotated median {DP.depth_unannot.median():.3f} "
      f"| n {DP.depth_annot.notna().sum()} | {metrics.paired(DP.depth_annot, DP.depth_unannot)}")
