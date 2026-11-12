#!/usr/bin/env python
"""Convert the thesis notebooks into plain scripts.

Code cells are copied unchanged and in their original order. Three edits only:
IPython magics and shell lines are commented out, the working directory is set
to notebooks/mel_nv (where the notebooks ran), and absolute paths to the
author's repository root are replaced by the repository root.

Usage:
    python tools/nb_to_script.py
    python tools/nb_to_script.py --skip 16g_verify.ipynb=12,40,41
"""
import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "mel_nv"
MAP = {
    "00_prepare_mel_nv_clean_csv.ipynb": "scripts/data/prepare_split.py",
    "12a_annotation_qc.ipynb": "scripts/data/annotation_qc.py",
    "12b_generate_eval_cams.ipynb": "scripts/eval/generate_eval_cams.py",
    "12c_overlap_eval_cams.ipynb": "scripts/eval/overlap_eval_cams.py",
    "16b_lesion_alignment.ipynb": "scripts/analysis/lesion_alignment.py",
    "16c_explanations.ipynb": "scripts/analysis/explanations.py",
    "16d_concept_alignment.ipynb": "scripts/analysis/expert_alignment.py",
    "16e_clinician_study.ipynb": "scripts/clinician/build_study.py",
    "16f_figures_tables.ipynb": "scripts/figures/figures_tables.py",
    "16g_verify.ipynb": "scripts/figures/final_figures_statistics.py",
}
ROOTS = ["/Users/choekyelnyungmartsang/Developer/master-thesis",
         "/storage/homefs/cn21m021/projects/master-thesis"]
PRE = '''"""Converted from notebooks/mel_nv/{nb} by tools/nb_to_script.py.

Code cells are unchanged and in their original order. Run from the repository root:
    python {out}
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parents[{depth}]
_CWD = _REPO / "notebooks" / "mel_nv"
_CWD.mkdir(parents=True, exist_ok=True)
os.chdir(_CWD)
sys.path.insert(0, str(_REPO))
try:
    from IPython.display import display
except ImportError:
    display = print
'''


def convert(nb, out, skip):
    cells = json.loads(nb.read_text())["cells"]
    parts = [PRE.format(nb=nb.name, out=out, depth=len(Path(out).parts) - 1)]
    errors = []
    for i, c in enumerate(cells):
        src = "".join(c["source"]).rstrip()
        if c["cell_type"] != "code" or not src:
            continue
        if any(o.get("output_type") == "error" for o in c.get("outputs", [])):
            errors.append(i)
        if i in skip:
            parts.append(f"\n# ---- cell {i}: skipped (raised an error in the notebook)\n")
            continue
        lines = ["# " + l if re.match(r"\s*[%!]", l) or "get_ipython()" in l else l
                 for l in src.splitlines()]
        body = "\n".join(lines)
        for r in ROOTS:
            body = body.replace(f'"{r}', '"" + str(_REPO) + "').replace(f"'{r}", "'' + str(_REPO) + '")
        parts.append(f"\n# ---- cell {i} {'-' * 60}\n{body}\n")
    dst = REPO / out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(parts))
    left = sorted(set(re.findall(r"/Users/[^\s\"']+|/storage/homefs/[^\s\"']+", dst.read_text())))
    print(f"{nb.name:42} -> {out}")
    if errors:
        print(f"    cells with error output: {errors}")
    if left:
        print(f"    PERSONAL PATHS LEFT: {left[:5]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", action="append", default=[], help="notebook=cell,cell,...")
    args = ap.parse_args()
    SKIP = {}
    for s in args.skip:
        name, ids = s.split("=")
        SKIP[name] = {int(x) for x in ids.split(",") if x}
    for nb, out in MAP.items():
        p = NB / nb
        if not p.exists():
            print(f"MISSING {nb}")
            continue
        convert(p, out, SKIP.get(nb, set()))