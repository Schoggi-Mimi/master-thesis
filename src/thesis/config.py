from pathlib import Path
import sys
import torch
import re

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


_TOP = {"data", "results", "outputs", "external", "notebooks", "scripts"}
_INDEX = None


def rebase(p):
    """Map a path written on any machine (Mac, Windows, UBELIX) onto the current repo.
    Splits on both separators, so it never depends on the OS that wrote the path."""
    if p is None or (isinstance(p, float) and p != p):
        return p
    parts = [x for x in re.split(r"[\\/]+", str(p)) if x and not x.endswith(":")]
    if REPO.name in parts:
        i = len(parts) - 1 - parts[::-1].index(REPO.name)
        out = REPO.joinpath(*parts[i + 1:])
    else:
        i = next((k for k, s in enumerate(parts) if s in _TOP), None)
        out = REPO.joinpath(*(parts[i:] if i is not None else parts))
    if out.exists():
        return out
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for f in (REPO / "data").rglob("*.png"):
            _INDEX.setdefault(f.name, []).append(f)
    hits = _INDEX.get(parts[-1], [])
    if len(hits) == 1:
        print(f"rebase fallback: {parts[-1]} -> {hits[0].relative_to(REPO)}")
        return hits[0]
    raise FileNotFoundError(f"{out} missing, {len(hits)} matches by name")