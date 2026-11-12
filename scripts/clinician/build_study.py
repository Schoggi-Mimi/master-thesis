"""Converted from notebooks/mel_nv/16e_clinician_study.ipynb by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python scripts/clinician/build_study.py
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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.thesis.plotting import overlay, draw_top_smooth

CONF = 0.9
TARGET = {"TP": 6, "TN": 6, "FP": 4, "FN": 4}      # 20 cases, 10 per part

# ---- predictions of the aligned model on the 987 test split -----------------
T = TEST_DF.copy()
T["p_mel"] = models.predict_mel(M5, X_TEST)
T["pred"] = np.where(T.p_mel >= 0.5, "MEL", "NV")
T["case"] = np.select(
    [(T.gt_label == "MEL") & (T.pred == "MEL"), (T.gt_label == "NV") & (T.pred == "NV"),
     (T.gt_label == "MEL") & (T.pred == "NV"), (T.gt_label == "NV") & (T.pred == "MEL")],
    ["TP", "TN", "FN", "FP"])
T["conf"] = np.maximum(T.p_mel, 1 - T.p_mel)
assert not (set(T.image_id) & set(QC.Image_ID)), "annotated image in study pool"
print("all test cases\n", T.case.value_counts().to_string())
print("confident cases\n", T[T.conf >= CONF].case.value_counts().to_string(), "\n")

# ---- select all 20 at once, confident first, random inside the pool ---------
pool, short = [], 0
for c, n in TARGET.items():
    base = T[T.case == c]
    conf = base[base.conf >= CONF]
    if len(conf) >= n:
        pick = conf.sample(n, random_state=SEED)
    else:
        pick = base.nlargest(min(n, len(base)), "conf")
    short += n - len(pick)
    pool.append(pick)
    print(f"  {c}: wanted {n}, got {len(pick)}, confident available {len(conf)}, "
          f"conf {pick.conf.min():.2f} to {pick.conf.max():.2f}")
ALL = pd.concat(pool)
if short:
    extra = T[T.case.isin(["TP", "TN"]) & (T.conf >= CONF) & ~T.image_id.isin(ALL.image_id)]
    ALL = pd.concat([ALL, extra.sample(short, random_state=SEED)])
    print(f"  filled {short} with confident correct cases")

# ---- split each stratum evenly between the two parts ------------------------
ALL = ALL.sample(frac=1, random_state=SEED).reset_index(drop=True)
ALL["block"] = ALL.groupby("case").cumcount() % 2 + 1
S1 = ALL[ALL.block == 1].sample(frac=1, random_state=SEED).reset_index(drop=True)
S2 = ALL[ALL.block == 2].sample(frac=1, random_state=SEED + 1).reset_index(drop=True)
S1["case_no"] = np.arange(1, len(S1) + 1)
S2["case_no"] = np.arange(len(S1) + 1, len(S1) + len(S2) + 1)
assert not (set(S1.image_id) & set(S2.image_id))
print(f"\npart 1 {len(S1)} cases, part 2 {len(S2)} cases")
print(pd.crosstab(ALL.case, ALL.block).to_string(), "\n")

B1 = ["unaligned Grad-CAM", "aligned Grad-CAM"]
B2 = ["aligned Grad-CAM", "FinerCAM 0.6", "aligned Diff-CAM"]
LET = ["A", "B", "C"]
# OUTLINE = dict(frac=0.10, color="#00E5FF", fill_alpha=0.05,
#                lw=0.9, line_alpha=0.85, halo_lw=1.6, halo_alpha=0.5)
OUTLINE = dict(frac=0.10, color="#00E5FF", fill_alpha=0,
               lw=0.7, line_alpha=0.9, halo_lw=0, halo_alpha=0)

def maps(rel, pred, which):
    x = data.to_tensor(data.native(rel))
    f = {"unaligned Grad-CAM": lambda: cams.target_ref(M0, x, pred)[0],
         "aligned Grad-CAM": lambda: cams.target_ref(M5, x, pred)[0],
         "aligned Diff-CAM": lambda: cams.diff_cam(M5, x, pred),
         "FinerCAM 0.6": lambda: cams.finer_cam(M5, x, pred, 0.6)}
    return {m: f[m]() for m in which}

# ---- balanced column order, fixed before any map is computed ----------------
def orders_for(n, methods, seed):
    k, let = len(methods), LET[:len(methods)]
    lo, hi = n // k, math.ceil(n / k)
    for s in range(seed, seed + 20000):
        rng = np.random.default_rng(s)
        o = [list(rng.permutation(methods)) for _ in range(n)]
        pos = pd.DataFrame([(m, let[x.index(m)]) for x in o for m in methods],
                           columns=["method", "letter"])
        ct = pd.crosstab(pos.method, pos.letter)
        if ct.values.min() >= lo and ct.values.max() <= hi:
            print(ct.to_string(), "\n")
            return o
    raise RuntimeError("no balanced order found")

O1 = orders_for(len(S1), B1, SEED)
O2 = orders_for(len(S2), B2, SEED + 7)

# ---- pages -------------------------------------------------------------------
def text_page(pdf, title, body, size=(8.3, 5.8), mono=False):
    f, a = plt.subplots(figsize=size)
    a.axis("off")
    a.text(0, 1, title, fontsize=15, fontweight="bold", va="top")
    a.text(0, .88, body, fontsize=10.5, va="top", family="monospace" if mono else None)
    pdf.savefig(f)
    plt.close(f)

def case_figure(r, order, methods, top):
    let = LET[:len(methods)]
    M = maps(r.image_rel_path, r.pred, methods)
    rgb = data.rgb224(r.image_rel_path)
    n = len(methods) + 1
    fig, ax = plt.subplots(1, n, figsize=(3.3 * n, 3.9))
    ax[0].imshow(rgb)
    ax[0].set_title("image", fontsize=11)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    for j, m in enumerate(order):
        if top:
            draw_top_smooth(ax[j + 1], rgb, M[m], **OUTLINE)
        else:
            overlay(ax[j + 1], rgb, M[m], thresh=0.28)
        ax[j + 1].set_title(let[j], fontsize=15, fontweight="bold")
    lab = "melanoma" if r.pred == "MEL" else "nevus"
    fig.suptitle(f"Case {r.case_no}      Model predicts: {lab}", fontsize=13)
    return fig

def case_pages(pdf, S, O, methods, top):
    for (_, r), order in zip(S.iterrows(), O):
        fig = case_figure(r, order, methods, top)
        pdf.savefig(fig)
        plt.close(fig)

n1, n2 = len(S1), len(S2)
with PdfPages(OUT / "clinician_study.pdf") as pdf:
    text_page(pdf, "Explanation preference study",
              "Two short parts. Every page shows one dermoscopic image and several\n"
              "explanations of the SAME model prediction, stated above the image.\n"
              "The order of the letters changes on every page.\n\n"
              f"Part 1, cases 1 to {n1}. Two full heat maps A, B.\n"
              "  Warmer colour means more important for the prediction.\n"
              "  Which map best highlights the regions you consider clinically\n"
              "  relevant to the predicted diagnosis?\n\n"
              f"Part 2, cases {n1 + 1} to {n1 + n2}. Three maps A, B, C.\n"
              "  Only the 10 percent most important regions are outlined in cyan.\n"
              "  Which outline best matches the structures you would use\n"
              "  to reach the predicted diagnosis? A ranking is welcome but optional.\n\n"
              "Please note your answers on the next page.")
    p1 = [f"Case {i:2d}    best: ____" for i in S1.case_no]
    p2 = [f"Case {i:2d}    best: ____    ranking (optional): ___ ___ ___" for i in S2.case_no]
    text_page(pdf, "Answers",
              "Part 1\n\n" + "\n\n".join(p1) + "\n\n\nPart 2\n\n" + "\n\n".join(p2),
              size=(8.3, 11.7), mono=True)
    text_page(pdf, "Part 1", "Full heat maps. Warmer colour = more important.")
    case_pages(pdf, S1, O1, B1, top=False)
    text_page(pdf, "Part 2", "Cyan outline = the 10 percent most important regions.")
    case_pages(pdf, S2, O2, B2, top=True)

# ---- key, never sent ---------------------------------------------------------
KEY = pd.DataFrame([{"block": b, "case_no": r.case_no, "image_id": r.image_id,
                     "letter": LET[j], "method": m, "gt": r.gt_label, "pred": r.pred,
                     "case": r.case, "p_mel": round(float(r.p_mel), 3)}
                    for b, S, O in [(1, S1, O1), (2, S2, O2)]
                    for (_, r), o in zip(S.iterrows(), O) for j, m in enumerate(o)])
KEY.to_csv(OUT / "clinician_study_KEY_do_not_send.csv", index=False)
ALL.to_csv(OUT / "clinician_study_cases.csv", index=False)

# ---- sanity checks -----------------------------------------------------------
assert KEY.groupby("block").method.unique().apply(len).to_dict() == {1: 2, 2: 3}
assert KEY.groupby(["block", "case_no"]).letter.nunique().isin([2, 3]).all()
print(pd.crosstab([KEY.block, KEY.method], KEY.letter).to_string())
print(f"\nwritten {OUT / 'clinician_study.pdf'}")
print(f"key     {OUT / 'clinician_study_KEY_do_not_send.csv'}")
