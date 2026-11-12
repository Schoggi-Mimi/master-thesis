"""Converted from notebooks/mel_nv/16b_lesion_alignment.ipynb by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python scripts/analysis/lesion_alignment.py
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
from sklearn.metrics import roc_auc_score, average_precision_score

from PIL import Image
import numpy as np, pandas as pd

p5 = models.predict_mel(M5, X_TEST)
print(f"ha5 test AUC {roc_auc_score(Y_MEL, p5):.4f}  AP {average_precision_score(Y_MEL, p5):.4f}")

rows = []
for _, r in TEST_DF.iterrows():
    L = data.to_grid(np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127)
    if not (0 < L.sum() < L.size):
        continue
    x = data.to_tensor(data.native(r.image_rel_path))
    for name, m in [("unaligned", M0), ("aligned", M5)]:
        mel, nv = cams.class_cams(m, x)
        t = mel if r.gt_label == "MEL" else nv
        rows.append({"model": name, "image_id": r.image_id, "gt": r.gt_label,
                     "loc_auc": metrics.lesion_auc(t, L),
                     "heat_in_lesion": float(t[L].sum() / (t.sum() + 1e-8)),
                     "mel_in_lesion": float(mel[L].sum() / (mel.sum() + 1e-8)),
                     "nv_in_lesion": float(nv[L].sum() / (nv.sum() + 1e-8)),
                     "lesion_share": float(L.mean())})
LOC = pd.DataFrame(rows)
LOC.to_csv(TAB / "ch5_localisation_test.csv", index=False)

cols = ["loc_auc", "heat_in_lesion", "mel_in_lesion", "nv_in_lesion", "lesion_share"]
print(LOC.groupby("model")[cols].agg(lambda s: metrics.summarise(s)["median_iqr"]).to_string())
print(LOC.groupby(["model", "gt"])[["loc_auc", "heat_in_lesion"]].median().round(3).to_string())
q = LOC.pivot_table(index="image_id", columns="model", values="loc_auc")
print("aligned minus unaligned, loc_auc:", metrics.paired(q["aligned"], q["unaligned"]))


# ---- cell 2 ------------------------------------------------------------
import hashlib
from sklearn.metrics import roc_auc_score, average_precision_score

def sha(p, n=1 << 20):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read(n)).hexdigest()[:12]

a5, m5 = CK_ALPHA / "checkpoint-best-gap-ha5.pth", CK5 / "checkpoint-best-gap-ha5.pth"
print("sweep ha5 vs main ha5 identical file:", sha(a5) == sha(m5))

SWEEP = {0.0: CK5 / "checkpoint-best-gap-ha0.pth"}
for lam in [0.1, 0.25, 0.5, 1, 3, 5, 7, 11, 17]:
    SWEEP[float(lam)] = CK_ALPHA / f"checkpoint-best-gap-ha{str(lam).replace('.', 'p')}.pth"
CE_ONLY = CK_ALPHA / "checkpoint-best-foundation_gap_ce_classes2_ep10_lr1e-5.pth"

P = {}
for lam, p in SWEEP.items():
    m = models.load_ckpt(p, cache=False)
    P[lam] = models.predict_mel(m, X_TEST)
    del m
m = models.load_ckpt(CE_ONLY, cache=False)
P_CE = models.predict_mel(m, X_TEST)
del m

rng = np.random.default_rng(SEED)
B = 2000
idx = [rng.integers(0, len(Y_MEL), len(Y_MEL)) for _ in range(B)]
idx = [i for i in idx if 0 < Y_MEL[i].sum() < len(i)]

rows = []
base = P[0.0]
for lam, p in P.items():
    auc, ap = roc_auc_score(Y_MEL, p), average_precision_score(Y_MEL, p)
    d_auc = np.array([roc_auc_score(Y_MEL[i], p[i]) - roc_auc_score(Y_MEL[i], base[i]) for i in idx])
    d_ap = np.array([average_precision_score(Y_MEL[i], p[i]) - average_precision_score(Y_MEL[i], base[i]) for i in idx])
    rows.append({"lambda": lam, "auc": auc, "ap": ap,
                 "auc_drop_pct": 100 * (roc_auc_score(Y_MEL, base) - auc) / roc_auc_score(Y_MEL, base),
                 "ap_drop_pct": 100 * (average_precision_score(Y_MEL, base) - ap) / average_precision_score(Y_MEL, base),
                 "p_auc": float(np.mean(d_auc >= 0)) if lam else np.nan,
                 "p_ap": float(np.mean(d_ap >= 0)) if lam else np.nan})
SW = pd.DataFrame(rows).round(4)
SW.to_csv(TAB / "ch5_lambda_sweep.csv", index=False)
print(SW.to_string(index=False))
print(f"\nCE-only file, other pipeline, not used:  AUC {roc_auc_score(Y_MEL, P_CE):.4f}  "
      f"AP {average_precision_score(Y_MEL, P_CE):.4f}")
print(f"bootstrap resamples used: {len(idx)}")


# ---- cell 3 ------------------------------------------------------------
rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    L, U = LESION[iid], HUM[iid]
    maps = {"unaligned": cams.target_ref(M0, x, r.gt_label)[0],
            "aligned": cams.target_ref(M5, x, r.gt_label)[0],
            "centre": metrics.centre_map(L)}
    for k, v in maps.items():
        rows.append({"model": k, "image_id": iid,
                     "auc_in": metrics.auc_in(v, L, U),
                     "energy_in": metrics.energy_excess(v, L, U),
                     "energy_grid": float(cams.norm01(v)[U & L].sum() / (cams.norm01(v).sum() + 1e-8)
                                          - (U & L).mean()),
                     "sat_09": float((cams.norm01(v)[L] > 0.9).mean())})
C6 = pd.DataFrame(rows)
C6.to_csv(TAB / "ch5_centre.csv", index=False)
print(C6.groupby("model")[["auc_in", "energy_in", "energy_grid", "sat_09"]].median().round(3).to_string())
for c in ["auc_in", "energy_in", "energy_grid"]:
    q = C6.pivot_table(index="image_id", columns="model", values=c)
    print(f"  aligned minus centre  {c:12} {metrics.paired(q['aligned'], q['centre'])}")

sub = TEST_DF.sample(300, random_state=SEED)
loc = []
for lam, p in SWEEP.items():
    m = models.load_ckpt(p, cache=False)
    vals = []
    for _, r in sub.iterrows():
        L = data.to_grid(np.array(Image.open((HAM / str(r.mask_rel_path)).resolve()).convert("L")) > 127)
        if not (0 < L.sum() < L.size):
            continue
        t = cams.target_ref(m, data.to_tensor(data.native(r.image_rel_path)), r.gt_label)[0]
        vals.append(metrics.lesion_auc(t, L))
    loc.append({"lambda": lam, "loc_auc_median": float(np.median(vals)), "n": len(vals)})
    del m
LC = pd.DataFrame(loc)
LC.to_csv(TAB / "ch5_lambda_localisation.csv", index=False)
print(LC.round(3).to_string(index=False))


# ---- cell 4 ------------------------------------------------------------
rows = []
for lam, p in SWEEP.items():
    m = models.load_ckpt(p, cache=False)
    for iid in sorted(HUM):
        r = EV.loc[iid]
        mel, nv = cams.class_cams(m, data.to_tensor(data.native(r.image_rel_path)))
        L = LESION[iid]
        t = mel if r.gt_label == "MEL" else nv
        rows.append({"lambda": lam, "image_id": iid,
                     "corr_in": metrics.corr_in(mel, nv, L),
                     "auc_in": metrics.auc_in(t, L, HUM[iid]),
                     "energy_in": metrics.energy_excess(t, L, HUM[iid]),
                     "sat_09": float((t[L] > 0.9).mean())})
    del m
LAMC = pd.DataFrame(rows)
LAMC.to_csv(TAB / "ch5_lambda_collapse.csv", index=False)
print(LAMC.groupby("lambda")[["corr_in", "auc_in", "energy_in", "sat_09"]].median().round(3).to_string())


# ---- cell 5 ------------------------------------------------------------
M01 = models.load_ckpt(CK_ALPHA / "checkpoint-best-gap-ha0p1.pth", cache=False)
rows = []
for iid in sorted(HUM):
    r = EV.loc[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    L, U = LESION[iid], HUM[iid]
    for k, v in {"lam0.1": cams.target_ref(M01, x, r.gt_label)[0],
                 "lam5": cams.target_ref(M5, x, r.gt_label)[0],
                 "centre": metrics.centre_map(L)}.items():
        rows.append({"model": k, "image_id": iid,
                     "energy_in": metrics.energy_excess(v, L, U),
                     "auc_in": metrics.auc_in(v, L, U)})
    a5, b5 = cams.class_cams(M5, x)
    rows.append({"model": "lam5_allcells", "image_id": iid,
                 "energy_in": np.nan, "auc_in": np.nan,
                 "corr_all": float(np.corrcoef(a5.ravel(), b5.ravel())[0, 1])})
P01 = pd.DataFrame(rows)
P01.to_csv(TAB / "ch5_lam01.csv", index=False)
print(P01.groupby("model")[["energy_in", "auc_in"]].median().round(3).to_string())
print("lam5 all-cell correlation median:",
      round(P01[P01.model == "lam5_allcells"].corr_all.median(), 3))
for a, b in [("lam0.1", "lam5"), ("lam0.1", "centre")]:
    for c in ["energy_in", "auc_in"]:
        q = P01.pivot_table(index="image_id", columns="model", values=c)
        print(f"  {a} minus {b}  {c:10} {metrics.paired(q[a], q[b])}")


# ---- cell 7 ------------------------------------------------------------
from sklearn.metrics import roc_auc_score, average_precision_score

IDX = {}
for p in CK_F2.glob("*.pth"):
    a = models.ckpt_args(p)
    key = (float(a["ha_lambda"]), float(a["feat_lambda"]), int(a["fold"]))
    assert key not in IDX, f"duplicate checkpoint for {key}"
    IDX[key] = (p, a)

SETTINGS = [(1e-8, 0.0), (5.0, 0.0), (5.0, 1.0), (5.0, 5.0), (5.0, 20.0),
            (0.0, 1.0), (0.0, 5.0), (0.0, 20.0)]
NAME = {(1e-8, 0.0): "control", (5.0, 0.0): "HA1 only",
        (5.0, 1.0): "joint b1", (5.0, 5.0): "joint b5", (5.0, 20.0): "joint b20",
        (0.0, 1.0): "HA2 b1", (0.0, 5.0): "HA2 b5", (0.0, 20.0): "HA2 b20"}

missing = [(s, k) for s in SETTINGS for k in range(5) if (*s, k) not in IDX]
print(f"{len(IDX)} checkpoints indexed, missing: {missing}")
A = pd.DataFrame([{**dict(zip(["alpha", "beta", "fold"], k)),
                   "batch": a.get("batch_size"), "epochs": a.get("epochs"),
                   "warm": bool(a.get("init_checkpoint"))} for k, (_, a) in IDX.items()])
assert not missing and A.batch.nunique() == 1 and A.epochs.eq(10).all() and not A.warm.any()
print("recipe identical across all settings")


# ---- cell 8 ------------------------------------------------------------
FOLD = data.fold_of_heldout()
gt_of = EV.gt_label.to_dict()
held = sorted(i for i in FOLD if i in HUM)

def egrid(v, T):
    v = cams.norm01(v)
    return float(v[T].sum() / (v.sum() + 1e-8) - T.mean())

rows, cls = [], []
for s in SETTINGS:
    for k in range(5):
        m = models.load_ckpt(IDX[(*s, k)][0], cache=False)
        for iid in [i for i in held if FOLD[i] == k]:
            r = EV.loc[iid]
            mel, nv = cams.class_cams(m, data.to_tensor(data.native(r.image_rel_path)))
            L, U, g = LESION[iid], HUM[iid], gt_of[iid]
            t = mel if g == "MEL" else nv
            rows.append({"setting": NAME[s], "alpha": s[0], "beta": s[1], "fold": k,
                         "image_id": iid, "gt": g,
                         "mel_in": metrics.energy_excess(mel, L, U),
                         "mel_grid": egrid(mel, U & L),
                         "mel_auc_in": metrics.auc_in(mel, L, U),
                         "mel_heat_in": float(mel[L].sum() / (mel.sum() + 1e-8)),
                         "nv_rest_in": metrics.energy_excess(nv, L, L & ~U) if g == "NV" else np.nan,
                         "nv_rest_grid": egrid(nv, L & ~U) if g == "NV" else np.nan,
                         "gt_loc": metrics.lesion_auc(t, L),
                         "corr_in": metrics.corr_in(mel, nv, L),
                         "sat_09": float((t[L] > 0.9).mean())})
        p = models.predict_mel(m, X_TEST)
        cls.append({"setting": NAME[s], "alpha": s[0], "beta": s[1], "fold": k,
                    "auc": roc_auc_score(Y_MEL, p), "ap": average_precision_score(Y_MEL, p)})
        del m
    print("done", NAME[s])

for iid in held:
    L, U = LESION[iid], HUM[iid]
    c = metrics.centre_map(L)
    rows.append({"setting": "centre", "image_id": iid, "gt": gt_of[iid],
                 "mel_in": metrics.energy_excess(c, L, U),
                 "mel_auc_in": metrics.auc_in(c, L, U)})

F7 = pd.DataFrame(rows)
C7 = pd.DataFrame(cls)
F7.to_csv(TAB / "ch7_grounding.csv", index=False)
C7.to_csv(TAB / "ch7_classification.csv", index=False)


# ---- cell 9 ------------------------------------------------------------
ORD = [NAME[s] for s in SETTINGS] + ["centre"]
COLS = ["mel_in", "mel_grid", "mel_auc_in", "mel_heat_in", "nv_rest_in",
        "nv_rest_grid", "gt_loc", "corr_in", "sat_09"]
print("medians over held-out images")
print(F7.groupby("setting")[COLS].median().reindex(ORD).round(3).to_string())

print("\nclassification, mean ± SD over 5 folds")
print(C7.groupby("setting")[["auc", "ap"]]
      .agg(lambda s: f"{s.mean():.4f} ± {s.std():.4f}").reindex(ORD[:-1]).to_string())

print("\nlocalisation of the ground-truth class map, by diagnosis")
print(F7.groupby(["setting", "gt"]).gt_loc.median().unstack().reindex(ORD[:-1]).round(3).to_string())

TEST = ["mel_in", "mel_grid", "mel_auc_in", "nv_rest_in", "gt_loc", "corr_in"]
for ref in ["control", "HA1 only"]:
    print(f"\npaired against {ref}, Bonferroni threshold {0.05 / 7:.4f}")
    for st in ORD[:-1]:
        if st == ref:
            continue
        line = f"  {st:10}"
        for c in TEST:
            q = F7[F7.setting.isin([ref, st])].pivot_table(index="image_id", columns="setting", values=c)
            if st not in q or ref not in q:
                continue
            r = metrics.paired(q[st], q[ref])
            line += f"  {c} {r['diff']:+.3f} p{r['p']:.4f}"
        print(line)

print("\nagainst the centre baseline, MEL map, in lesion only")
for st in ORD[:-1]:
    line = f"  {st:10}"
    for c in ["mel_in", "mel_auc_in"]:
        q = F7[F7.setting.isin([st, "centre"])].pivot_table(index="image_id", columns="setting", values=c)
        r = metrics.paired(q[st], q["centre"])
        line += f"  {c} {r['diff']:+.3f} p{r['p']:.4f} wins {r['wins']}/{r['n']}"
    print(line)


# ---- cell 10 ------------------------------------------------------------
import matplotlib.pyplot as plt

g = F7[F7.setting != "centre"].groupby(["alpha", "beta"])[["mel_in", "mel_grid", "gt_loc"]].median()
g = g.join(C7.groupby(["alpha", "beta"]).ap.mean())
fig, ax = plt.subplots(1, 3, figsize=(plotting.TEXTWIDTH, 2.4))
curves = [("with HA1, $\\lambda_{les}=5$", [(5.0, b) for b in (0.0, 1.0, 5.0, 20.0)], plotting.C_FOCUS),
          ("without HA1", [(1e-8, 0.0), (0.0, 1.0), (0.0, 5.0), (0.0, 20.0)], plotting.C_CONTROL)]
for j, (y, lab) in enumerate([("mel_in", "MEL map energy excess, in lesion"),
                              ("mel_grid", "MEL map energy excess, whole grid"),
                              ("gt_loc", "lesion localisation")]):
    for name, pts, c in curves:
        s = g.loc[pts]
        ax[j].plot(s.ap, s[y], "o-", color=c, lw=1.4, ms=3.5, label=name)
        for (a, b), r in s.iterrows():
            ax[j].annotate(f"{b:g}", (r.ap, r[y]), xytext=(2, 2), textcoords="offset points", fontsize=6)
    ax[j].set(xlabel="test AP", title=lab)
ax[0].legend(frameon=False, fontsize=6)
fig.tight_layout()
plotting.save(fig, "ch7_tradeoff")
plt.show()


# ---- cell 11 ------------------------------------------------------------
rows = []
for s in [(1e-8, 0.0), (5.0, 0.0), (0.0, 1.0)]:
    for k in range(5):
        p, a = IDX[(*s, k)]
        ck = torch.load(p, map_location="cpu", weights_only=False)
        r = C7[(C7.alpha == s[0]) & (C7.beta == s[1]) & (C7.fold == k)].iloc[0]
        rows.append({"setting": NAME[s], "fold": k, "best_epoch": ck.get("epoch"),
                     "auc": round(r.auc, 4), "ap": round(r.ap, 4)})
PF = pd.DataFrame(rows)
print(PF.pivot_table(index="fold", columns="setting", values=["ap", "best_epoch"], aggfunc="first").to_string())
print(f"\ncheckpoints5 ha0 for reference: AP 0.8498 (batch 64, single run)")

# paired per fold, same held-out split, AP difference against control
q = PF.pivot_table(index="fold", columns="setting", values="ap")
for st in ["HA1 only", "HA2 b1"]:
    d = q[st] - q["control"]
    print(f"{st} minus control, AP per fold: {d.round(3).tolist()}  mean {d.mean():+.3f}")


# ---- cell 12 ------------------------------------------------------------
EP9 = {"control ep9": REPO / "external/checkpoints_f2/control_ep9",
       "HA2 b1 ep9": REPO / "external/checkpoints_f2/ha2b1_ep9"}
for name, d in EP9.items():
    for k in range(5):
        ck = torch.load(d / f"fold{k}.pth", map_location="cpu", weights_only=False)
        a = ck["args"]; a = vars(a) if not isinstance(a, dict) else a
        assert ck.get("epoch") in (9, "9") and a["fold"] == k, (name, k, ck.get("epoch"), a["fold"])
print("all ep9 files verified")

rows, cls = [], []
for name, d in EP9.items():
    for k in range(5):
        m = models.load_ckpt(d / f"fold{k}.pth", cache=False)
        for iid in [i for i in held if FOLD[i] == k]:
            r = EV.loc[iid]
            mel, nv = cams.class_cams(m, data.to_tensor(data.native(r.image_rel_path)))
            L, U, g = LESION[iid], HUM[iid], gt_of[iid]
            t = mel if g == "MEL" else nv
            rows.append({"setting": name, "image_id": iid, "gt": g,
                         "mel_in": metrics.energy_excess(mel, L, U),
                         "mel_grid": egrid(mel, U & L),
                         "mel_auc_in": metrics.auc_in(mel, L, U),
                         "nv_rest_in": metrics.energy_excess(nv, L, L & ~U) if g == "NV" else np.nan,
                         "gt_loc": metrics.lesion_auc(t, L),
                         "corr_in": metrics.corr_in(mel, nv, L)})
        p = models.predict_mel(m, X_TEST)
        cls.append({"setting": name, "fold": k,
                    "auc": roc_auc_score(Y_MEL, p), "ap": average_precision_score(Y_MEL, p)})
        del m
F9, C9 = pd.DataFrame(rows), pd.DataFrame(cls)
F9.to_csv(TAB / "ch7_ep9.csv", index=False); C9.to_csv(TAB / "ch7_ep9_cls.csv", index=False)

cols = ["mel_in", "mel_grid", "mel_auc_in", "nv_rest_in", "gt_loc", "corr_in"]
ALL = pd.concat([F7, F9])
show = ["control", "control ep9", "HA1 only", "joint b5", "HA2 b1", "HA2 b1 ep9", "centre"]
print(ALL.groupby("setting")[cols].median().reindex(show).round(3).to_string())
print("\nclassification, mean ± SD over 5 folds")
print(pd.concat([C7, C9]).groupby("setting")[["auc", "ap"]]
      .agg(lambda s: f"{s.mean():.4f} ± {s.std():.4f}").reindex(show[:-1]).to_string())

def cmp(a, b, cs):
    line = f"  {a} vs {b}:"
    for c in cs:
        q = ALL[ALL.setting.isin([a, b])].pivot_table(index="image_id", columns="setting", values=c)
        r = metrics.paired(q[a], q[b])
        line += f"  {c} {r['diff']:+.3f} p{r['p']:.4f}"
    print(line)
print()
for a in ["HA1 only", "joint b5", "HA2 b1", "HA2 b1 ep9"]:
    cmp(a, "control ep9", ["mel_in", "mel_grid", "mel_auc_in", "nv_rest_in", "gt_loc"])
cmp("HA2 b1 ep9", "HA2 b1", ["mel_in", "mel_grid", "mel_auc_in", "gt_loc", "corr_in"])
cmp("HA2 b1 ep9", "HA1 only", ["mel_in", "mel_grid", "mel_auc_in", "gt_loc", "corr_in"])
for c in ["mel_in", "mel_auc_in"]:
    q = ALL[ALL.setting.isin(["HA2 b1 ep9", "centre"])].pivot_table(index="image_id", columns="setting", values=c)
    print(f"  HA2 b1 ep9 vs centre {c}: {metrics.paired(q['HA2 b1 ep9'], q['centre'])}")


# ---- cell 13 ------------------------------------------------------------
KEYS = ["batch_size", "feat_batch_size", "epochs", "lr", "min_lr", "warmup_epochs",
        "weight_decay", "layer_decay", "drop_path", "seed", "weights", "monitor",
        "disable_color_jitter", "pooling"]
fam = {"Ch 5 unaligned": CK5 / "checkpoint-best-gap-ha0.pth",
       "Ch 5 aligned": CK5 / "checkpoint-best-gap-ha5.pth",
       "Ch 7 grid (fold 0, joint b5)": IDX[(5.0, 5.0, 0)][0]}
G = pd.DataFrame({k: {kk: models.ckpt_args(p).get(kk) for kk in KEYS} for k, p in fam.items()})
print(G.to_string())
G.to_latex(TAB / "app_hyperparameters.tex", escape=True)


# ---- cell 14 ------------------------------------------------------------
import cv2
def logit_gap(m, img, gt):
    with torch.no_grad():
        o = m(data.to_tensor(img))
        l = o["logits"] if isinstance(o, dict) else (o[0] if isinstance(o, (tuple, list)) else o)
    a, b = (MEL_IDX, NV_IDX) if gt == "MEL" else (NV_IDX, MEL_IDX)
    return float(l[0, a] - l[0, b])

def remove(img, cells):
    m = np.kron(cells.astype(np.uint8), np.ones((16, 16), np.uint8)) * 255
    return cv2.inpaint(img, m, 7, cv2.INPAINT_TELEA)

def deletion(model, iid, rng):
    r = EV.loc[iid]
    img = data.rgb224(r.image_rel_path)
    L, U = LESION[iid], HUM[iid]
    T = U & L
    k = int(T.sum())
    t = cams.target_ref(model, data.to_tensor(img), r.gt_label)[0]
    lidx, bidx = np.flatnonzero(L), np.flatnonzero(~L)
    def cells(idx):
        c = np.zeros(196, bool); c[idx] = True; return c.reshape(14, 14)
    regions = {"clinician": T,
               "top cells": cells(lidx[np.argsort(-t.ravel()[lidx])[:k]]),
               "random lesion": cells(rng.choice(lidx, k, replace=False)),
               "random background": cells(rng.choice(bidx, min(k, len(bidx)), replace=False))}
    base = logit_gap(model, img, r.gt_label)
    return {nm: (base - logit_gap(model, remove(img, c), r.gt_label)) / max(abs(base), 0.1)
            for nm, c in regions.items()}

rng = np.random.default_rng(SEED)
rows = []
for name, m in [("unaligned", M0), ("lesion aligned", M5)]:
    for iid in sorted(HUM):
        rows.append({"model": name, "image_id": iid, **deletion(m, iid, rng)})
FOLD = data.fold_of_heldout()
for k in range(5):
    m = models.f2(0.0, 1.0, k)
    for iid in [i for i in sorted(HUM) if FOLD.get(i) == k]:
        rows.append({"model": "concept only", "image_id": iid, **deletion(m, iid, rng)})
    del m
DL = pd.DataFrame(rows); DL.to_csv(TAB / "ch5_deletion.csv", index=False)
REG = ["clinician", "top cells", "random lesion", "random background"]
print("normalised drop in support for the true class, median (higher = region mattered more)")
print(DL.groupby("model")[REG].median().round(3).to_string())
for mo in DL.model.unique():
    s = DL[DL.model == mo]
    for reg in ["top cells", "clinician"]:
        print(f"  {mo:15} {reg:10} vs random lesion: {metrics.paired(s[reg], s['random lesion'])}")


# ---- cell 16 ------------------------------------------------------------
IDX = {}
for p in CK_F2.glob("*.pth"):
    a = models.ckpt_args(p)
    IDX[(float(a["ha_lambda"]), float(a["feat_lambda"]), int(a["fold"]))] = (p, a)
NAME[(0.1, 1.0)] = "weak HA1 + HA2 b1"
rows = []
for k in range(5):
    m = models.load_ckpt(IDX[(0.1, 1.0, k)][0], cache=False)
    for iid in [i for i in held if FOLD[i] == k]:
        r = EV.loc[iid]
        mel, nv = cams.class_cams(m, data.to_tensor(data.native(r.image_rel_path)))
        L, U, g = LESION[iid], HUM[iid], gt_of[iid]
        rows.append({"setting": NAME[(0.1, 1.0)], "image_id": iid, "gt": g,
                     "mel_in": metrics.energy_excess(mel, L, U), "mel_grid": egrid(mel, U & L),
                     "mel_auc_in": metrics.auc_in(mel, L, U),
                     "gt_loc": metrics.lesion_auc(mel if g == "MEL" else nv, L),
                     "corr_in": metrics.corr_in(mel, nv, L)})
    del m
W = pd.concat([F7[F7.setting.isin(["HA1 only", "HA2 b1"])], pd.DataFrame(rows)])
print(W.groupby(["setting", "gt"])[["mel_in", "mel_grid", "mel_auc_in", "gt_loc", "corr_in"]]
      .median().round(3).to_string())
for c in ["mel_in", "mel_grid", "mel_auc_in", "gt_loc"]:
    q = W.pivot_table(index="image_id", columns="setting", values=c)
    print(c, metrics.paired(q[NAME[(0.1, 1.0)]], q["HA2 b1"]))


# ---- cell 17 ------------------------------------------------------------
import json
for k in range(5):
    p, a = IDX[(0.1, 1.0, k)]
    m = models.load_ckpt(p, cache=False)
    pr = models.predict_mel(m, X_TEST)
    del m
    print(f"fold {k}  {p.name[:60]}  ha {a['ha_lambda']}  beta {a['feat_lambda']}  "
          f"fold {a['fold']}  batch {a['batch_size']}  epochs {a['epochs']}  "
          f"AUC {roc_auc_score(Y_MEL, pr):.4f}  AP {average_precision_score(Y_MEL, pr):.4f}")


# ---- cell 18 ------------------------------------------------------------
print([f for f in dir(cams) if "head" in f.lower()])


# ---- cell 19 ------------------------------------------------------------
iid = sorted(HUM)[0]; r = EV.loc[iid]
x = data.to_tensor(data.native(r.image_rel_path))
H = cams.head_maps(M5, x, r.gt_label)
rebuilt = cams.norm01(np.maximum(H.mean(0), 0)).reshape(14, 14)
ref = cams.target_ref(M5, x, r.gt_label)[0]
print("shape", H.shape, " max abs diff", float(np.abs(rebuilt - ref).max()))


# ---- cell 20 ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

rows, feats = [], {}
for iid in sorted(HUM):
    r = EV.loc[iid]
    x = data.to_tensor(data.native(r.image_rel_path))
    L, U = LESION[iid].ravel(), HUM[iid].ravel()
    H = cams.head_maps(M5, x, r.gt_label)[:, L].T      # lesion cells x 12 heads
    y = U[L].astype(int)
    feats[iid] = (H, y)
    uni = cams.target_ref(M5, x, r.gt_label)[0]
    Xs = StandardScaler().fit_transform(H)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    rows.append({"image_id": iid,
                 "uniform": metrics.auc_in(uni, LESION[iid], HUM[iid]),
                 "oracle": roc_auc_score(y, lr.predict_proba(Xs)[:, 1])})

for row in rows:                                         # leave one image out
    tr = [k for k in feats if k != row["image_id"]]
    Xtr = np.vstack([feats[k][0] for k in tr]); ytr = np.concatenate([feats[k][1] for k in tr])
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    H, y = feats[row["image_id"]]
    row["held_out"] = roc_auc_score(y, lr.predict_proba(sc.transform(H))[:, 1])

OR = pd.DataFrame(rows); OR.to_csv(TAB / "ch5_head_oracle.csv", index=False)
print(OR[["uniform", "oracle", "held_out"]].median().round(3).to_string())
print("oracle   vs uniform:", metrics.paired(OR.oracle, OR.uniform))
print("held-out vs uniform:", metrics.paired(OR.held_out, OR.uniform))


# ---- cell 21 ------------------------------------------------------------
KEYS2 = KEYS + ["ha_lambda", "ha_start_epoch", "ha_fp_weight", "ha_loss_type", "nb_classes", "init_checkpoint", "disable_amp"]
pair = {"main (4.1)": CK5 / "checkpoint-best-gap-ha5.pth",
        "sweep, lambda 5": CK_ALPHA / "checkpoint-best-gap-ha5.pth"}
A2 = pd.DataFrame({k: {kk: models.ckpt_args(p).get(kk) for kk in KEYS2} for k, p in pair.items()})
for k, p in pair.items():
    A2.loc["saved_epoch", k] = torch.load(p, map_location="cpu", weights_only=False).get("epoch")
d = A2[A2.astype(str).nunique(axis=1) > 1]
print(A2.astype(str).to_string())
print("\ndiffering rows:", "none" if d.empty else "\n" + d.astype(str).to_string())
