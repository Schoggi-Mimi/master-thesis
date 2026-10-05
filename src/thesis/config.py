from pathlib import Path
import sys
import torch


def find_repo():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "scripts" / "generate_finer_cam_panderm.py").exists():
            return p.resolve()
    raise FileNotFoundError("repo root not found")


REPO = find_repo()
for extra in [REPO, REPO / "external" / "PanDerm" / "classification"]:
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

GEOMETRY = "squash224"
BLOCK = -1
MEL_IDX, NV_IDX = 0, 1
TOP_FRAC = 0.10
SEED = 2024

HAM = REPO / "data" / "HAM10000"
MELNV = HAM / "mel_nv"
QC_CSV = REPO / "results" / "annotation_qc" / GEOMETRY / "annotation_qc_manifest.csv"
EVAL_CSV = REPO / "outputs" / "mel_nv" / "eval_cams" / "gap" / GEOMETRY / "eval_images.csv"
FEAT_CSV = REPO / "results" / "alignment2" / "feature_masks_manifest.csv"

CK5 = REPO / "external" / "checkpoints5"
CK_ALPHA = REPO / "external" / "checkpoints_alpha"
CK_F2 = REPO / "external" / "checkpoints_f2"

OUT = REPO / "results" / "final"
TAB = OUT / "tables"
FIG = OUT / "figures"
for d in (OUT, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

MORPH = {
    "STRUCTURELESS": ["homogeneous", "structureless_area"],
    "NETWORK": ["atypical_network", "negative_network", "regular_network"],
    "REGRESSION": ["peppering", "regression",
                   "regression_scar_like_depigmentation_and_peppering"],
    "BLUE_WHITE": ["blue_white_structureless_area"],
    "DOTS": ["atypical_dots"],
    "STREAKS": ["atypical_streaks"],
    "VASCULAR": ["atypical_vascular_pattern"],
}
LAB2GRP = {l: g for g, ls in MORPH.items() for l in ls}
GROUP_COLOR = {
    "STRUCTURELESS": "#1F77B4", "NETWORK": "#FF7F0E", "REGRESSION": "#2CA02C",
    "BLUE_WHITE": "#9467BD", "DOTS": "#D62728", "STREAKS": "#8C564B",
    "VASCULAR": "#17BECF", "UNASSIGNED": "#7F7F7F",
}