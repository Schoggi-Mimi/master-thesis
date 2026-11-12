"""Converted from notebooks/mel_nv/16f_figures_tables.ipynb by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python scripts/figures/figures_tables.py
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
from src.thesis.plotting import overlay, outline, save, C_LESION
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pick = (TEST_DF[TEST_DF.gt_label == "MEL"].sort_values("image_id").head(3).image_id.tolist()
        + TEST_DF[TEST_DF.gt_label == "NV"].sort_values("image_id").head(3).image_id.tolist())
T = TEST_DF.set_index("image_id")
fig, ax = plt.subplots(6, 5, figsize=(plotting.TEXTWIDTH, 7.6))
heads = ["image", "unaligned MEL", "unaligned NV", "aligned MEL", "aligned NV"]
for i, iid in enumerate(pick):
    r = T.loc[iid]
    rgb = data.rgb224(r.image_rel_path)
    L = data.to_grid(np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127)
    x = data.to_tensor(data.native(r.image_rel_path))
    a0, b0 = cams.class_cams(M0, x)
    a5, b5 = cams.class_cams(M5, x)
    ax[i, 0].imshow(rgb); outline(ax[i, 0], L, C_LESION)
    for j, v in enumerate([a0, b0, a5, b5]):
        overlay(ax[i, j + 1], rgb, v); outline(ax[i, j + 1], L, C_LESION, lw=0.6)
    ax[i, 0].set_ylabel(r.gt_label, fontsize=7)
    for a in ax[i]: a.set_xticks([]); a.set_yticks([])
for j, h in enumerate(heads): ax[0, j].set_title(h, fontsize=7)
fig.tight_layout(); save(fig, "fig_ch5_localisation_examples"); plt.show()


# ---- cell 2 ------------------------------------------------------------
C6 = pd.read_csv(TAB / "ch5_centre.csv")
q = C6.pivot_table(index="image_id", columns="model", values="auc_in").dropna()
q = q.sort_values("aligned")
fig, ax = plt.subplots(figsize=(plotting.TEXTWIDTH * 0.6, 2.6))
for k, (iid, r) in enumerate(q.iterrows()):
    ax.plot([k, k], [r.aligned, r.centre], color=plotting.C_CONTEXT, lw=0.8)
ax.scatter(range(len(q)), q.aligned, s=12, color=plotting.C_MEL, label="aligned Grad-CAM", zorder=3)
ax.scatter(range(len(q)), q.centre, s=12, color=plotting.C_REF, marker="s", label="centre baseline", zorder=3)
ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax.set(xlabel="annotated image, sorted by aligned AUC", ylabel="in-lesion AUC vs clinician", xticks=[])
ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); save(fig, "fig_ch5_centre_baseline"); plt.show()


# ---- cell 3 ------------------------------------------------------------
from src.thesis.plotting import overlay, outline, save, C_LESION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sw = pd.read_csv(TAB / "ch5_lambda_sweep.csv")
lc = pd.read_csv(TAB / "ch5_lambda_localisation.csv")
co = pd.read_csv(TAB / "ch5_lambda_collapse.csv").groupby("lambda")[["corr_in", "sat_09"]].median().reset_index()
X0, MAIN = 0.03, 5.0
xs = lambda s: s.replace(0.0, X0)

def ring(a, df, col, color):
    r = df[df["lambda"] == MAIN]
    a.plot(xs(r["lambda"]), r[col], "o", ms=7, mfc="none", mec=color, mew=1.1, zorder=4)

fig, ax = plt.subplots(1, 3, figsize=(plotting.TEXTWIDTH, 2.3))
ax[0].plot(xs(lc["lambda"]), lc.loc_auc_median, "o-", color=plotting.C_FOCUS, ms=3)
ring(ax[0], lc, "loc_auc_median", plotting.C_FOCUS)
ax[0].set(ylabel="lesion localisation")

ax[1].plot(xs(co["lambda"]), co.corr_in, "o-", color=plotting.C_FOCUS, ms=3, label="MEL vs NV corr")
ax[1].plot(xs(co["lambda"]), co.sat_09, "s--", color=plotting.C_CONTROL, ms=3, label="cells above 0.9")
ring(ax[1], co, "corr_in", plotting.C_FOCUS); ring(ax[1], co, "sat_09", plotting.C_CONTROL)
ax[1].set(ylabel="median value"); ax[1].legend(frameon=False, fontsize=6)

ax[2].plot(xs(sw["lambda"]), sw.auc_drop_pct, "o-", color=plotting.C_CONTROL, ms=3, label="AUC drop")
ax[2].plot(xs(sw["lambda"]), sw.ap_drop_pct, "o-", color=plotting.C_FOCUS, ms=3, label="AP drop")
ring(ax[2], sw, "auc_drop_pct", plotting.C_CONTROL); ring(ax[2], sw, "ap_drop_pct", plotting.C_FOCUS)
ax[2].set(ylabel="drop vs λ=0 (%)"); ax[2].legend(frameon=False, fontsize=6)

for a in ax:
    a.axvline(MAIN, color="grey", ls=":", lw=0.9, zorder=0)
    a.set_xscale("log"); a.set_xticks([X0, 0.1, 1, 10]); a.set_xticklabels(["0", "0.1", "1", "10"])
    a.set_xlabel(r"$\lambda_{global}$")
ax[0].text(MAIN * 1.12, ax[0].get_ylim()[0] + 0.05 * np.ptp(ax[0].get_ylim()),
           "main model", fontsize=6, color="grey")
fig.tight_layout(); save(fig, "fig_ch5_lambda_sweep"); plt.show()


# ---- cell 4 ------------------------------------------------------------
FM = data.load_feat_manifest()
ann = FM.query("rule != 'nevus_anchor'")[["image_id", "dx", "rule", "fold"]]
lab = QC.groupby("Image_ID").label.apply(lambda s: ", ".join(sorted(set(s))))
ann = ann.merge(lab.rename("labels"), left_on="image_id", right_index=True, how="left")
ann["in HUM"] = ann.image_id.isin(HUM)
ann = ann.sort_values(["fold", "image_id"])
ann.to_csv(TAB / "app_case_manifest.csv", index=False)
ann.to_latex(TAB / "app_case_manifest.tex", index=False, escape=True, longtable=True)
print(len(ann), "annotated images written")


# ---- cell 5 ------------------------------------------------------------
from src.thesis.plotting import overlay, outline, save, box_compare, C_LESION, TEXTWIDTH
gt_of = EV.gt_label.to_dict()
ids = sorted(HUM)
SHOW = [i for i in ids if gt_of[i] == "MEL"][:2] + [i for i in ids if gt_of[i] == "NV"][:2]

heads = ["image", "aligned Grad-CAM", "Diff-CAM", r"FinerCAM $\gamma=0.6$"]
fig, ax = plt.subplots(len(SHOW), 4, figsize=(TEXTWIDTH, 1.7 * len(SHOW)))
for i, iid in enumerate(SHOW):
    r = EV.loc[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    rgb = data.rgb224(r.image_rel_path)
    L, U = LESION[iid], HUM[iid]
    maps = [cams.target_ref(M5, x, r.gt_label)[0], cams.diff_cam(M5, x, r.gt_label),
            cams.finer_cam(M5, x, r.gt_label, 0.6)]
    ax[i, 0].imshow(rgb)
    outline(ax[i, 0], L, C_LESION, lw=0.8)
    outline(ax[i, 0], U, "#FFFFFF", lw=1.2)
    ax[i, 0].set_ylabel(r.gt_label, fontsize=7)
    for j, v in enumerate(maps):
        overlay(ax[i, j + 1], rgb, v)
        outline(ax[i, j + 1], U, "#FFFFFF", lw=0.9)
        ax[i, j + 1].set_title(f"AUC {metrics.auc_in(v, L, U):.2f}", fontsize=6, pad=2)
    for a in ax[i]:
        a.set_xticks([]); a.set_yticks([])
for j, h in enumerate(heads):
    ax[0, j].set_title(h + ("\n" + ax[0, j].get_title() if j else ""), fontsize=7)
fig.tight_layout()
save(fig, "fig_ch6_methods_examples"); plt.show()

M61 = pd.read_csv(TAB / "ch6_metrics.csv")
d = {k: M61[M61.method == k].auc_in.values for k in
     ["Grad-CAM", "Diff-CAM", "FinerCAM 0.6", "random"]}
fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.55, 2.4))
box_compare(ax, d, focus=("Diff-CAM", "FinerCAM 0.6"), control=("Grad-CAM",),
            ref=M61[M61.method == "centre"].auc_in.median(), ref_label="centre baseline",
            ylabel="in-lesion AUC vs clinician")
ax.axhline(0.5, color="k", ls=":", lw=0.7)
fig.tight_layout()
save(fig, "fig_ch6_methods_auc"); plt.show()


# ---- cell 6 ------------------------------------------------------------
FG = pd.read_csv(TAB / "ch6_finercam_gamma_auc.csv")
FA = pd.read_csv(TAB / "finercam_alpha_sweep.csv")
fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))
for mname, c in [("aligned", plotting.C_MEL), ("unaligned", plotting.C_CONTEXT)]:
    s = FG[FG.model == mname].groupby("gamma").auc.median()
    ax[0].plot(s.index, s.values, "o-", color=c, ms=3.5, label=mname)
ax[0].axhline(0.5, color="k", ls=":", lw=0.7)
ax[0].set(xlabel=r"$\gamma$", ylabel="in-lesion AUC vs clinician")
ax[0].legend(frameon=False, fontsize=6)
S = FA[FA.model == "aligned"].groupby("alpha")[["corr", "mel_on_mel"]].median()
ax[1].plot(S["corr"], S.mel_on_mel, "o-", color=plotting.C_MEL, ms=3.5)
for g, r in S.iterrows():
    ax[1].annotate(f"{g:g}", (r["corr"], r.mel_on_mel), xytext=(3, 3),
                   textcoords="offset points", fontsize=6)
ax[1].axhline(0, color="k", ls=":", lw=0.7)
ax[1].set(xlabel="MEL vs NV map correlation in lesion", ylabel="MEL map energy excess")
fig.tight_layout()
save(fig, "fig_ch6_finercam_gamma"); plt.show()


# ---- cell 7 ------------------------------------------------------------
import matplotlib.patches as mpatches
GX = pd.read_csv(TAB / "ch6_group_excess.csv")
net = GX[(GX.group == "NETWORK") & (GX.image_id != "ISIC_0029698")].nlargest(2, "excess").image_id.tolist()
SHOW = ["ISIC_0029698"] + net

fig, ax = plt.subplots(len(SHOW), 3, figsize=(TEXTWIDTH * 0.8, 1.9 * len(SHOW)))
for i, iid in enumerate(SHOW):
    r = EV.loc[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    rgb = data.rgb224(r.image_rel_path)
    L = LESION[iid]
    G = data.group_masks(QC, iid, L)
    t = cams.target_ref(M5, x, r.gt_label)[0]
    idx = np.flatnonzero(L)
    hot = np.zeros(L.size, bool)
    hot[idx[np.argsort(-t.ravel()[idx])[:20]]] = True
    hot = hot.reshape(L.shape)
    ax[i, 0].imshow(rgb)
    for g, m in G.items():
        outline(ax[i, 0], m, GROUP_COLOR[g], lw=1.3)
    ax[i, 0].set_ylabel(f"{iid}\n{r.gt_label}", fontsize=6)
    overlay(ax[i, 1], rgb, t)
    outline(ax[i, 1], L, C_LESION, lw=0.7)
    ax[i, 2].imshow(rgb)
    for g, m in G.items():
        outline(ax[i, 2], m, GROUP_COLOR[g], lw=1.0, ls="dashed")
    outline(ax[i, 2], hot, "#FFFFFF", lw=1.4)
    sub = GX[(GX.image_id == iid) & (GX.group != "UNASSIGNED")].nlargest(2, "excess")
    ax[i, 2].set_title("  ".join(f"{g} {e:+.2f}" for g, e in zip(sub.group, sub.excess)), fontsize=5.5)
    for a in ax[i]:
        a.set_xticks([]); a.set_yticks([])
for j, h in enumerate(["annotation groups", "aligned Grad-CAM", "top 20 lesion cells"]):
    ax[0, j].set_title(h + ("\n" + ax[0, j].get_title() if j == 2 else ""), fontsize=7)
present = sorted({g for iid in SHOW for g in data.group_masks(QC, iid, LESION[iid])})
fig.legend(handles=[mpatches.Patch(color=GROUP_COLOR[g], label=g) for g in present],
           loc="lower center", ncol=4, fontsize=5.5, frameon=False)
fig.tight_layout(rect=[0, 0.05, 1, 1])
save(fig, "fig_ch6_attribution_examples"); plt.show()


# ---- cell 8 ------------------------------------------------------------
FOLD = data.fold_of_heldout()
held = sorted(i for i in FOLD if i in HUM)
SHOW = [i for i in held if gt_of[i] == "MEL"][:2] + [i for i in held if gt_of[i] == "NV"][:2]
SETS = [("lesion only", 5.0, 0.0), ("joint $\\lambda_{con}=5$", 5.0, 5.0),
        ("concept only $\\lambda_{con}=1$", 0.0, 1.0)]

fig, ax = plt.subplots(len(SHOW), 7, figsize=(TEXTWIDTH, 1.25 * len(SHOW) + 0.4))
cache = {}
for i, iid in enumerate(SHOW):
    r = EV.loc[iid]
    k = FOLD[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    rgb = data.rgb224(r.image_rel_path)
    L, U = LESION[iid], HUM[iid]
    ax[i, 0].imshow(rgb)
    outline(ax[i, 0], U, "#FFFFFF", lw=1.0)
    ax[i, 0].set_ylabel(r.gt_label, fontsize=7)
    for s, (name, a, b) in enumerate(SETS):
        if (a, b, k) not in cache:
            cache[(a, b, k)] = models.f2(a, b, k)
        mel, nv = cams.class_cams(cache[(a, b, k)], x)
        for c, v in enumerate([mel, nv]):
            axx = ax[i, 1 + 2 * s + c]
            overlay(axx, rgb, v)
            outline(axx, U, "#FFFFFF", lw=0.8)
    for a_ in ax[i]:
        a_.set_xticks([]); a_.set_yticks([])
cache.clear()
ax[0, 0].set_title("image", fontsize=6)
for s, (name, _, _) in enumerate(SETS):
    ax[0, 1 + 2 * s].set_title(f"{name}\nMEL map", fontsize=5.5)
    ax[0, 2 + 2 * s].set_title("\nNV map", fontsize=5.5)
fig.tight_layout()
save(fig, "fig_ch7_channels"); plt.show()


# ---- cell 9 ------------------------------------------------------------
import cv2
from PIL import Image
from src.thesis.plotting import overlay, outline, save, C_LESION, TEXTWIDTH

def mask224(rel):
    m = np.array(Image.open((HAM / str(rel)).resolve()).convert("L")) > 127
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)

def grid(m):
    return m.reshape(14, 16, 14, 16).mean((1, 3)) >= 0.5

def shift(img, m):
    ys, xs = np.where(m)
    dx = min(223 - xs.max(), 60) if 223 - xs.max() > xs.min() else -min(xs.min(), 60)
    dy = min(223 - ys.max(), 60) if 223 - ys.max() > ys.min() else -min(ys.min(), 60)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    i2 = cv2.warpAffine(img, M, (224, 224), borderMode=cv2.BORDER_REFLECT)
    m2 = cv2.warpAffine(m.astype(np.uint8), M, (224, 224), flags=cv2.INTER_NEAREST).astype(bool)
    return i2, m2

def inpaint(img, m):
    k = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) * 255
    return cv2.inpaint(img, k, 7, cv2.INPAINT_TELEA)

def uniform(img, m):
    return np.broadcast_to(np.median(img[~m], axis=0).astype(np.uint8), img.shape).copy()

CENTRE = np.zeros((14, 14), bool); CENTRE[4:11, 4:11] = True     # central quarter, 49 of 196 cells

def heat_share(v, G):
    v = cams.norm01(v)
    return float(v[G].sum() / (v.sum() + 1e-8))

def conditions(r):
    img, m = data.rgb224(r.image_rel_path), mask224(r.mask_rel_path)
    s_img, s_m = shift(img, m)
    return {"original": (img, grid(m)), "shifted": (s_img, grid(s_m)),
            "lesion inpainted": (inpaint(img, m), grid(m)), "uniform": (uniform(img, m), CENTRE)}

# ---- aggregate over 100 test images ------------------------------------------
rows = []
for _, r in TEST_DF.sample(100, random_state=SEED).iterrows():
    for cond, (img, G) in conditions(r).items():
        if not (0 < G.sum() < G.size):
            continue
        x = data.to_tensor(img)
        for name, m in [("unaligned", M0), ("aligned", M5)]:
            t = cams.target_ref(m, x, r.gt_label)[0]
            rows.append({"cond": cond, "model": name, "share": heat_share(t, G), "area": G.mean()})
RB = pd.DataFrame(rows)
RB.to_csv(TAB / "app_robustness.csv", index=False)
print("median heat share on the region (lesion, or central quarter for uniform), vs its area")
print(RB.groupby(["cond", "model"])[["share", "area"]].median().round(3).to_string())

# ---- figure, 2 images --------------------------------------------------------
pick = [TEST_DF[TEST_DF.gt_label == g].sort_values("image_id").iloc[0] for g in ("MEL", "NV")]
CONDS = ["original", "shifted", "lesion inpainted", "uniform"]
fig, ax = plt.subplots(len(CONDS), 6, figsize=(TEXTWIDTH, 4.4))
for b, r in enumerate(pick):
    C = conditions(r)
    for i, cond in enumerate(CONDS):
        img, G = C[cond]
        x = data.to_tensor(img)
        c0 = 3 * b
        ax[i, c0].imshow(img)
        outline(ax[i, c0], G, C_LESION, lw=0.8)
        for j, m in enumerate([M0, M5]):
            t = cams.target_ref(m, x, r.gt_label)[0]
            overlay(ax[i, c0 + 1 + j], img, t)
            outline(ax[i, c0 + 1 + j], G, C_LESION, lw=0.6)
            ax[i, c0 + 1 + j].set_title(f"{heat_share(t, G):.2f}", fontsize=5.5, pad=1)
        ax[i, 0].set_ylabel(cond, fontsize=6)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    ax[0, 3 * b].set_title(f"{r.gt_label} image", fontsize=6)
    ax[0, 3 * b + 1].set_title("unaligned\n" + ax[0, 3 * b + 1].get_title(), fontsize=6)
    ax[0, 3 * b + 2].set_title("aligned\n" + ax[0, 3 * b + 2].get_title(), fontsize=6)
fig.tight_layout()
save(fig, "fig_appX_robustness"); plt.show()


# ---- cell 10 ------------------------------------------------------------
C6 = pd.read_csv(TAB / "ch5_centre.csv").pivot_table(index="image_id", columns="model", values="auc_in")
C6["d"] = C6.aligned - C6.centre
SHOW = C6.nsmallest(2, "d").index.tolist() + C6.nlargest(2, "d").index.tolist()
fig, ax = plt.subplots(4, 3, figsize=(TEXTWIDTH * 0.75, 6.4))
for i, iid in enumerate(SHOW):
    r = EV.loc[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    rgb = data.rgb224(r.image_rel_path)
    L, U = LESION[iid], HUM[iid]
    c, t = metrics.centre_map(L), cams.target_ref(M5, x, r.gt_label)[0]
    ax[i, 0].imshow(rgb); outline(ax[i, 0], L, C_LESION, lw=0.9); outline(ax[i, 0], U, "#FFFFFF", lw=1.3)
    ax[i, 0].set_ylabel("centre wins" if i < 2 else "model wins", fontsize=6)
    for j, (v, nm) in enumerate([(c, "centre baseline"), (t, "aligned Grad-CAM")]):
        overlay(ax[i, j + 1], rgb, np.where(L, v, 0)); outline(ax[i, j + 1], U, "#FFFFFF", lw=1.0)
        ax[i, j + 1].set_title(f"{nm}\nin-lesion AUC {metrics.auc_in(v, L, U):.2f}", fontsize=6)
    for a in ax[i]: a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_title("green lesion, white annotation", fontsize=6)
fig.tight_layout(); save(fig, "fig_ch5_centre_examples"); plt.show()


# ---- cell 11 ------------------------------------------------------------
from scipy.stats import wilcoxon

rows = []
def groups(iid):
    L = LESION[iid]
    G = data.group_masks(QC, iid, L)
    allg = np.zeros_like(L)
    for m in G.values(): allg |= m
    G["UNASSIGNED"] = L & ~allg
    return {g: m for g, m in G.items() if m.sum()}

def ge_rows(model, name, ids):
    for iid in ids:
        r = EV.loc[iid]
        mel, _ = cams.class_cams(model, data.to_tensor(data.native(r.image_rel_path)))
        L = LESION[iid]
        for g, m in groups(iid).items():
            rows.append({"model": name, "image_id": iid, "group": g, "cells": int(m.sum()),
                         "energy_exc": metrics.energy_excess(mel, L, m)})

held = [i for i in sorted(HUM) if i in FOLD]
ge_rows(M0, "unaligned", held); ge_rows(M5, "lesion aligned", held)
for k in range(5):
    m = models.f2(0.0, 1.0, k); ge_rows(m, "concept only", [i for i in held if FOLD[i] == k]); del m
GE = pd.DataFrame(rows); GE.to_csv(TAB / "ch6_group_energy.csv", index=False)

order = GE.groupby("group").image_id.nunique().sort_values(ascending=False).index
print("median energy excess of the MEL map per group (0 = as much heat as its area predicts)")
print(GE.pivot_table(index="group", columns="model", values="energy_exc", aggfunc="median")
      .reindex(order).round(3).to_string())
ng = len(order)
for mo in ["lesion aligned", "concept only"]:
    print(f"\n{mo}, Wilcoxon vs 0, Bonferroni over {ng}")
    for g in order:
        s = GE[(GE.model == mo) & (GE.group == g)].energy_exc.dropna()
        if len(s) >= 5:
            print(f"  {g:14} n {len(s):2}  {s.median():+.3f}  p_bonf {min(1, wilcoxon(s).pvalue * ng):.4f}")


# ---- cell 12 ------------------------------------------------------------
def darkness_map(rel):
    g = data.rgb224(rel).astype(float).mean(axis=2)
    return 1 - g.reshape(14, 16, 14, 16).mean((1, 3)) / 255

rows = []
def lad(name, v, iid):
    L, U = LESION[iid], HUM[iid]
    rows.append({"ref": name, "image_id": iid,
                 "auc_in": metrics.auc_in(v, L, U) if v[L].std() > 1e-9 else 0.5,
                 "energy_in": metrics.energy_excess(v, L, U) if v[L].std() > 1e-9 else 0.0})
rng = np.random.default_rng(SEED)
for iid in held:
    r = EV.loc[iid]; L = LESION[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    lad("random", rng.random(L.shape), iid)
    lad("flat lesion", L.astype(float), iid)
    lad("darkness", darkness_map(r.image_rel_path), iid)
    lad("unaligned", cams.class_cams(M0, x)[0], iid)
    lad("lesion aligned", cams.class_cams(M5, x)[0], iid)
    lad("centre", metrics.centre_map(L), iid)
for k in range(5):
    m = models.f2(0.0, 1.0, k)
    for iid in [i for i in held if FOLD[i] == k]:
        lad("concept only", cams.class_cams(m, data.to_tensor(data.native(EV.loc[iid].image_rel_path)))[0], iid)
    del m
LD = pd.DataFrame(rows); LD.to_csv(TAB / "ch5_reference_ladder.csv", index=False)
ORD = ["random", "flat lesion", "darkness", "unaligned", "lesion aligned", "centre", "concept only"]
print("MEL map against melanoma structures, inside the lesion, 29 held-out images")
print(LD.groupby("ref")[["auc_in", "energy_in"]].median().reindex(ORD).round(3).to_string())
for ref in ["darkness", "centre"]:
    for mo in ["lesion aligned", "concept only"]:
        q = LD.pivot_table(index="image_id", columns="ref", values="auc_in")
        print(f"  {mo} vs {ref}, AUC: {metrics.paired(q[mo], q[ref])}")


# ---- cell 13 ------------------------------------------------------------
RB = pd.read_csv(TAB / "app_robustness.csv")
from scipy.stats import wilcoxon
u = RB[RB.cond == "uniform"]
for m in ["aligned", "unaligned"]:
    s = u[u.model == m]
    print(m, "vs 0.25:", round(s.share.median(), 3), "p", wilcoxon(s.share - s.area).pvalue)
a, b = u[u.model == "aligned"].share.values, u[u.model == "unaligned"].share.values
print("aligned minus unaligned:", round(np.median(a - b), 3), "p", wilcoxon(a, b).pvalue)


# ---- cell 14 ------------------------------------------------------------
from src.thesis.plotting import overlay, outline, save, TEXTWIDTH, C_LESION
import matplotlib.patches as mpatches

# ---- 1. how prominent is each group ------------------------------------------
rows = []
for iid in sorted(HUM):
    L = LESION[iid]; gt = EV.loc[iid].gt_label
    for g, m in data.group_masks(QC, iid, L).items():
        rows.append({"image_id": iid, "gt": gt, "group": g, "area_share": m.sum() / L.sum()})
GP = pd.DataFrame(rows)
n_ann = QC.assign(group=QC.label.map(LAB2GRP)).groupby("group").size()
prom = (GP.groupby("group").agg(images=("image_id", "nunique"),
                                MEL_images=("gt", lambda s: (s == "MEL").sum()),
                                NV_images=("gt", lambda s: (s == "NV").sum()),
                                median_area=("area_share", "median"))
        .join(n_ann.rename("annotations")).sort_values("images", ascending=False))
print("group prominence on the 31 annotated images\n", prom.round(2).to_string())

# ---- 2. normalised energy excess per group and model -------------------------
GE = pd.read_csv(TAB / "ch6_group_energy.csv")
area = GP.set_index(["image_id", "group"]).area_share
GE["area"] = [area.get((i, g), np.nan) for i, g in zip(GE.image_id, GE.group)]
GE.loc[GE.group == "UNASSIGNED", "area"] = np.nan
GE["norm_exc"] = GE.energy_exc / (1 - GE.area)
rank = GE.pivot_table(index="group", columns="model", values="norm_exc", aggfunc="median")
rank = rank.drop(index="UNASSIGNED", errors="ignore").sort_values("concept only", ascending=False)
print("\nnormalised energy excess, median (0 chance, 1 all heat on group)\n", rank.round(3).to_string())
GE.to_csv(TAB / "ch6_group_energy.csv", index=False)


# ---- cell 15 ------------------------------------------------------------
import cv2
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

def drops(model, iid, rng):
    img, gt, R, _ = regions(model, iid, rng)
    base = gap(model, img, gt)
    return {nm: (base - gap(model, remove(img, c), gt)) / max(abs(base), 0.1) for nm, c in R.items()}

rng = np.random.default_rng(SEED)
FOLD = data.fold_of_heldout(); held = [i for i in sorted(HUM) if i in FOLD]
rows = []
for name, m in [("unaligned", M0), ("lesion aligned", M5)]:
    for iid in held:
        rows.append({"model": name, "image_id": iid, **drops(m, iid, rng)})
for k in range(5):
    m = models.f2(0.0, 1.0, k)
    for iid in [i for i in held if FOLD[i] == k]:
        rows.append({"model": "concept only", "image_id": iid, **drops(m, iid, rng)})
    del m
DL = pd.DataFrame(rows); DL.to_csv(TAB / "ch5_deletion_v2.csv", index=False)
REG = ["top 10%", "random 10%", "clinician (n)", "unannotated (n)"]
ORDM = ["unaligned", "lesion aligned", "concept only"]
print("normalised drop in true-class support, median (higher = mattered more)")
print(DL.groupby("model")[REG].median().reindex(ORDM).round(3).to_string())
for mo in ORDM:
    s = DL[DL.model == mo]
    print(f"  {mo:15} top vs random:           {metrics.paired(s['top 10%'], s['random 10%'])}")
    print(f"  {mo:15} clinician vs unannotated: {metrics.paired(s['clinician (n)'], s['unannotated (n)'])}")


# ---- cell 17 ------------------------------------------------------------
from src.thesis.plotting import overlay, outline, save, TEXTWIDTH, C_LESION
FOLD = data.fold_of_heldout()
held = [i for i in sorted(HUM) if i in FOLD]
CTRL9 = REPO / "external/checkpoints_f2/control_ep9"

def three_models(k):
    return {"unaligned": models.load_ckpt(CTRL9 / f"fold{k}.pth", cache=False),
            "global": models.f2(5.0, 0.0, k), "expert": models.f2(0.0, 1.0, k)}

cand = [i for i in held if EV.loc[i].gt_label == "MEL"]
rows, cache = [], {}
fig, ax = plt.subplots(len(cand), 4, figsize=(6, 1.5 * len(cand)))
for i, iid in enumerate(cand):
    k = FOLD[iid]
    if k not in cache:
        cache.clear(); cache[k] = three_models(k)
    r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]
    x = data.to_tensor(data.native(r.image_rel_path)); rgb = data.rgb224(r.image_rel_path)
    ax[i, 0].imshow(rgb); outline(ax[i, 0], U, "#FFFFFF", lw=1.0)
    ax[i, 0].set_ylabel(iid[-7:], fontsize=6)
    for j, (nm, m) in enumerate(cache[k].items()):
        mel = cams.class_cams(m, x)[0]
        e = metrics.energy_excess(mel, L, U)
        overlay(ax[i, j + 1], rgb, mel); outline(ax[i, j + 1], U, "#FFFFFF", lw=0.7)
        ax[i, j + 1].set_title(f"{nm} {e:+.2f}", fontsize=5.5, pad=1)
        rows.append({"image_id": iid, "model": nm, "energy_in": e})
    for a in ax[i]: a.set_xticks([]); a.set_yticks([])
cache.clear()
fig.tight_layout(); plt.show()
CAND = pd.DataFrame(rows).pivot_table(index="image_id", columns="model", values="energy_in")
print(CAND.round(3).sort_values("expert", ascending=False).to_string())


# ---- cell 18 ------------------------------------------------------------
FIG1_ID = "ISIC_0030798"            # set from cell above (Choose an image where the clinician outline is clearly visible and the three maps differ in an obvious way. Don't just take the one with the largest gain. A typical case makes the figure more honest.)

import cv2
import matplotlib.patches as mpatches
from PIL import Image

def full224(m):
    return cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)

def lesion224(iid):
    m = np.array(Image.open((HAM / str(EV.loc[iid].mask_rel_path)).resolve()).convert("L")) > 127
    return full224(m)

def mel_struct224(iid):
    sub = QC[(QC.Image_ID == iid) & (QC.label_group == "MEL")]
    acc = None
    for p in sub.mask_path:
        m = data._full(p)
        acc = m if acc is None else (acc | m)
    return full224(acc)

def contour(ax, m, color, lw):
    ax.contour(m.astype(float), [0.5], colors=[color], linewidths=lw)

r = EV.loc[FIG1_ID]; k = FOLD[FIG1_ID]
L, U = LESION[FIG1_ID], HUM[FIG1_ID]
Lf, Uf = lesion224(FIG1_ID), mel_struct224(FIG1_ID)
x = data.to_tensor(data.native(r.image_rel_path)); rgb = data.rgb224(r.image_rel_path)
M = three_models(k)
C_EXP = "#FFFFFF"

fig, ax = plt.subplots(2, 3, figsize=(TEXTWIDTH * 0.8, TEXTWIDTH * 0.56))
cols = [("no alignment", "no spatial target", None, None),
        ("global alignment", "target: lesion mask", Lf, C_LESION),
        ("expert alignment", "target: clinician structures", Uf, C_EXP)]
for j, (h, sub, m, c) in enumerate(cols):
    ax[0, j].imshow(rgb)
    if m is not None:
        contour(ax[0, j], m, c, 1.3)
    ax[0, j].set_title(f"{h}\n", fontsize=8, fontweight="bold")
    ax[0, j].text(0.5, 1.02, sub, transform=ax[0, j].transAxes, ha="center", va="bottom", fontsize=6.5)
for j, nm in enumerate(["unaligned", "global", "expert"]):
    mel = cams.class_cams(M[nm], x)[0]
    overlay(ax[1, j], rgb, mel)
    contour(ax[1, j], Lf, C_LESION, 0.7)
    contour(ax[1, j], Uf, C_EXP, 1.0)
    ax[1, j].set_xlabel(f"energy excess {metrics.energy_excess(mel, L, U):+.2f}", fontsize=6.5)
del M
ax[0, 0].set_ylabel("supervision", fontsize=7)
ax[1, 0].set_ylabel("melanoma map", fontsize=7)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.legend(handles=[mpatches.Patch(fc=C_LESION, ec="none", label="lesion"),
                    mpatches.Patch(fc="white", ec="0.3", label="clinician melanoma structures")],
           loc="lower center", ncol=2, fontsize=6, frameon=False)
fig.tight_layout(rect=[0, 0.04, 1, 1])
save(fig, "fig_ch1_overview"); plt.show()
print("image used:", FIG1_ID)


# ---- cell 19 ------------------------------------------------------------
held_mel = [i for i in held if EV.loc[i].gt_label == "MEL"]
METH = {"Grad-CAM": lambda m, x: cams.target_ref(m, x, "MEL")[0],
        "FinerCAM 0.6": lambda m, x: cams.finer_cam(m, x, "MEL", 0.6),
        "Diff-CAM": lambda m, x: cams.diff_cam(m, x, "MEL")}
rows = []
for k in range(5):
    mods = {"global": models.f2(5.0, 0.0, k), "expert": models.f2(0.0, 1.0, k)}
    for iid in [i for i in held_mel if FOLD[i] == k]:
        r = EV.loc[iid]; L, U = LESION[iid], HUM[iid]
        x = data.to_tensor(data.native(r.image_rel_path))
        for mn, m in mods.items():
            for meth, f in METH.items():
                v = f(m, x)
                dead = v[L].max() < 1e-6
                rows.append({"model": mn, "method": meth, "image_id": iid,
                             "auc_in": 0.5 if dead else metrics.auc_in(v, L, U),
                             "energy_in": 0.0 if dead else metrics.energy_excess(v, L, U),
                             "dead": dead})
    del mods
FX = pd.DataFrame(rows); FX.to_csv(TAB / "ch7_contrastive_expert.csv", index=False)
print(f"melanoma held-out images: {FX.image_id.nunique()}")
print(FX.pivot_table(index="method", columns="model", values=["auc_in", "energy_in"],
                     aggfunc="median").round(3).to_string())
print("\nempty maps:\n", FX.pivot_table(index="method", columns="model", values="dead", aggfunc="sum").to_string())
for meth in METH:
    q = FX[FX.method == meth].pivot_table(index="image_id", columns="model", values="auc_in")
    print(f"  {meth:12} expert vs global, AUC: {metrics.paired(q['expert'], q['global'])}")
for mn in ["global", "expert"]:
    q = FX[FX.model == mn].pivot_table(index="image_id", columns="method", values="auc_in")
    print(f"  {mn:6} FinerCAM vs Grad-CAM, AUC: {metrics.paired(q['FinerCAM 0.6'], q['Grad-CAM'])}")

# ---- visual: first 3 melanoma held-out images by ID --------------------------
SHOW = held_mel[:3]
fig, ax = plt.subplots(len(SHOW), 7, figsize=(TEXTWIDTH, 1.35 * len(SHOW) + 0.4))
for i, iid in enumerate(SHOW):
    k = FOLD[iid]; r = EV.loc[iid]; U = HUM[iid]
    x = data.to_tensor(data.native(r.image_rel_path)); rgb = data.rgb224(r.image_rel_path)
    ax[i, 0].imshow(rgb); outline(ax[i, 0], U, "#FFFFFF", lw=0.9)
    mods = {"global": models.f2(5.0, 0.0, k), "expert": models.f2(0.0, 1.0, k)}
    col = 1
    for mn, m in mods.items():
        for meth, f in METH.items():
            overlay(ax[i, col], rgb, f(m, x)); outline(ax[i, col], U, "#FFFFFF", lw=0.6)
            if i == 0:
                ax[i, col].set_title(f"{mn}\n{meth}", fontsize=5.5)
            col += 1
    del mods
    for a in ax[i]: a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_title("image\nwhite: structures", fontsize=5.5)
fig.tight_layout(); save(fig, "fig_ch7_contrastive_expert"); plt.show()


# ---- cell 20 ------------------------------------------------------------
from src.thesis.plotting import overlay, outline, save, TEXTWIDTH
import matplotlib.patches as mpatches

NEW = {"lesion aligned": "global aligned", "lesion only": "global only",
       "concept only": "expert only", "HA1 only": "global only",
       "HA2 b1": "expert only", "joint b5": "joint"}
rn = lambda s: NEW.get(s, s)
FOLD = data.fold_of_heldout()
held = [i for i in sorted(HUM) if i in FOLD]
gt_of = EV.gt_label.to_dict()


# ---- cell 21 ------------------------------------------------------------
M61 = pd.read_csv(TAB / "ch6_metrics.csv")
ORD = ["Grad-CAM", "Diff-CAM", "FinerCAM 0.6", "random"]
COL = [plotting.C_CONTROL, plotting.C_MEL, plotting.C_MEL, plotting.C_CONTEXT]
d = [M61[M61.method == k].auc_in.dropna().values for k in ORD]

fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.55, 2.4))
b = ax.boxplot(d, widths=0.55, patch_artist=True, showfliers=False)
for box, c in zip(b["boxes"], COL):
    box.set_facecolor(c); box.set_alpha(0.6)
for m in b["medians"]:
    m.set_color("black")
rng = np.random.default_rng(SEED)
for i, v in enumerate(d, start=1):
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color="black", alpha=0.5, zorder=3)
ax.axhline(0.5, color="black", ls=":", lw=0.8)
ax.set(xticks=range(1, len(ORD) + 1), xticklabels=ORD, ylabel="in-lesion AUC vs clinician", ylim=(0, 1))
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout(); save(fig, "fig_ch6_methods_auc"); plt.show()


# ---- cell 22 ------------------------------------------------------------
F7 = pd.read_csv(TAB / "ch7_grounding.csv")
C7 = pd.read_csv(TAB / "ch7_classification.csv")
g = (F7[F7.setting != "centre"].groupby(["alpha", "beta"])[["mel_in", "mel_grid", "gt_loc"]].median()
     .join(C7.groupby(["alpha", "beta"]).ap.mean()))
curves = [(r"with global alignment, $\lambda_{global}=5$", [(5.0, b) for b in (0.0, 1.0, 5.0, 20.0)], plotting.C_MEL), ("without global alignment", [(1e-8, 0.0), (0.0, 1.0), (0.0, 5.0), (0.0, 20.0)], plotting.C_EXP)]
OFF = {(5.0, 0.0): (3, -9), (1e-8, 0.0): (3, -9)}
fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.3))
for j, (y, lab) in enumerate([("mel_in", "MEL energy, in lesion"),
                              ("mel_grid", "MEL energy, whole grid"),
                              ("gt_loc", "lesion localisation")]):
    for name, pts, c in curves:
        s = g.loc[pts]
        ax[j].plot(s.ap, s[y], "o-", color=c, lw=1.3, ms=3.5, label=name)
        for key, r in s.iterrows():
            ax[j].annotate("C" if key == (1e-8, 0.0) else f"{key[1]:g}", (r.ap, r[y]),
                           xytext=OFF.get(key, (3, 3)), textcoords="offset points", fontsize=6)
    ax[j].set(xlabel="test AP", title=lab)
ax[0].legend(frameon=False, fontsize=5.5, loc="upper left")
fig.tight_layout(); save(fig, "fig_ch7_tradeoff"); plt.show()


# ---- cell 23 ------------------------------------------------------------
SHOW = [i for i in held if gt_of[i] == "MEL"][:2] + [i for i in held if gt_of[i] == "NV"][:2]
SETS = [("global only", 5.0, 0.0), (r"joint $\lambda_{expert}=5$", 5.0, 5.0),
        (r"expert only $\lambda_{expert}=1$", 0.0, 1.0)]
fig, ax = plt.subplots(len(SHOW), 7, figsize=(TEXTWIDTH, 1.25 * len(SHOW) + 0.4))
for i, iid in enumerate(SHOW):
    r = EV.loc[iid]; k = FOLD[iid]; U = HUM[iid]
    x = data.to_tensor(data.native(r.image_rel_path)); rgb = data.rgb224(r.image_rel_path)
    ax[i, 0].imshow(rgb); outline(ax[i, 0], U, "#FFFFFF", lw=1.0)
    ax[i, 0].set_ylabel(r.gt_label, fontsize=7)
    for s, (name, a, b) in enumerate(SETS):
        m = models.f2(a, b, k)
        mel, nv = cams.class_cams(m, x)
        del m
        for c, v in enumerate([mel, nv]):
            overlay(ax[i, 1 + 2 * s + c], rgb, v); outline(ax[i, 1 + 2 * s + c], U, "#FFFFFF", lw=0.8)
    for a_ in ax[i]: a_.set_xticks([]); a_.set_yticks([])
ax[0, 0].set_title("image", fontsize=6)
for s, (name, _, _) in enumerate(SETS):
    ax[0, 1 + 2 * s].set_title(f"{name}\nMEL map", fontsize=5.5)
    ax[0, 2 + 2 * s].set_title("\nNV map", fontsize=5.5)
fig.tight_layout(); save(fig, "fig_ch7_channels"); plt.show()


# ---- cell 24 ------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
have = lambda n: (FIG / f"{n}.pdf").exists()

if have("fig_ch5_deletion"):
    DL = pd.read_csv(TAB / "ch5_deletion_v2.csv").assign(model=lambda d: d.model.map(rn))
    REG = ["top 10%", "random 10%", "clinician (n)", "unannotated (n)"]
    ORDM = ["unaligned", "global aligned", "expert only"]
    CC = ["#252525", plotting.C_CHANCE, "#737373", "#d9d9d9"]
    HH = ["", "", "///", "///"]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.4)); pos, ticks = 0, []
    for mo in ORDM:
        for reg, c, h in zip(REG, CC, HH):
            b = ax.boxplot(DL[DL.model == mo][reg].dropna(), positions=[pos], widths=0.7, patch_artist=True, showfliers=False)
            b["boxes"][0].set(facecolor=c, hatch=h, edgecolor="0.3"); pos += 1
        ticks.append(pos - 2.5); pos += 1
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xticks=ticks, xticklabels=[{"expert only": "expert aligned"}.get(m, m) for m in ORDM], ylabel="normalised drop")
    ax.legend([mpatches.Patch(facecolor=c, hatch=h, edgecolor="0.3") for c, h in zip(CC, HH)], REG, frameon=False, fontsize=6, ncol=4)
    fig.tight_layout(); save(fig, "fig_ch5_deletion"); plt.show()

if have("fig_ch6_group_ranking"):
    GE = pd.read_csv(TAB / "ch6_group_energy.csv").assign(model=lambda d: d.model.map(rn))
    rank = (GE[GE.group != "UNASSIGNED"].pivot_table(index="group", columns="model",
            values="norm_exc", aggfunc="median").sort_values("expert only", ascending=False))
    order = rank.index.tolist(); yy = np.arange(len(order))
    LBL = {"lesion aligned": "global aligned", "concept only": "expert aligned", "expert only": "expert aligned"}
    rank = rank.rename(columns=LBL)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.6, 2.6))
    for j, (mo, c) in enumerate([(m, plotting.MODEL_COLOR[m]) for m in ["unaligned", "global aligned", "expert aligned"]]):
    # for j, (mo, c) in enumerate([("unaligned", plotting.C_CONTEXT), ("global aligned", plotting.C_CONTROL), ("expert only", plotting.C_MEL)]):
        ax.barh(yy + (j - 1) * 0.27, rank[mo], height=0.27, color=c, label=mo)
    ax.axvline(0, color="k", lw=0.7)
    ax.set(yticks=yy, yticklabels=order, xlabel="normalised energy excess"); ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); save(fig, "fig_ch6_group_ranking"); plt.show()

print("other figures to rerun from their original cell if used:",
      [n for n in ["fig_ch6_group_examples", "fig_ch7_contrastive_expert"] if have(n)])


# ---- cell 25 ------------------------------------------------------------
rank


# ---- cell 26 ------------------------------------------------------------
def darkness_map(rel):
    g = data.rgb224(rel).astype(float).mean(axis=2)
    return 1 - g.reshape(14, 16, 14, 16).mean((1, 3)) / 255
