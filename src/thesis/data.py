import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from src.eval.cam_eval_utils import (CAM_GRID, CROP_SIZE,
                                     mask_to_cam_grid_geom, transform_rgb)

from .config import (DEVICE, EVAL_CSV, FEAT_CSV, GEOMETRY, HAM, LAB2GRP,
                     MEL_IDX, MELNV, QC_CSV)

PREPROCESS = T.Compose([
    T.Resize((CROP_SIZE, CROP_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def native(rel):
    return np.array(Image.open((HAM / str(rel)).resolve()).convert("RGB"))


def to_tensor(arr):
    return PREPROCESS(Image.fromarray(arr)).unsqueeze(0).to(DEVICE)


def rgb224(rel):
    return transform_rgb(native(rel), GEOMETRY)


def to_grid(mask_bool):
    return mask_to_cam_grid_geom(mask_bool, GEOMETRY) >= 0.5


def _strip(s):
    return s.astype(str).str.strip()


def load_eval():
    ev = pd.read_csv(EVAL_CSV)
    ev["image_id"] = _strip(ev.image_id)
    return ev.set_index("image_id", drop=False)


def load_lesions(ev):
    return {iid: to_grid(np.array(Image.open((HAM / str(r.mask_rel_path)).resolve())
                                  .convert("L")) > 127)
            for iid, r in ev.iterrows()}


def load_qc():
    q = pd.read_csv(QC_CSV).query("qc_pass").copy()
    q["Image_ID"] = _strip(q.Image_ID)
    return q


def _full(path):
    return np.array(Image.open(path).convert("L")) > 127


def mel_union(qc, iid):
    """Union at FULL resolution, then one grid projection at >= 0.5.
    Identical to mel_union_grid in 14d, which produced every Section 1 to 3 number."""
    sub = qc[(qc.Image_ID == iid) & (qc.label_group == "MEL")]
    if not len(sub):
        return None
    acc = None
    for p in sub.mask_path:
        m = _full(p)
        acc = m if acc is None else (acc | m)
    return to_grid(acc)


def load_hum(qc, lesion):
    """Melanoma structure union per image, kept only when the lesion has both
    annotated and unannotated patches. Same definition as Sections 1 to 3."""
    out = {}
    for iid in sorted(set(qc.Image_ID) & set(lesion)):
        u = mel_union(qc, iid)
        L = lesion[iid]
        if u is not None and L.sum() >= 8 and 0 < u[L].sum() < L.sum():
            out[iid] = u
    return out


def group_masks(qc, iid, L):
    """Same rule per morphology group: union at full resolution, then grid."""
    full = {}
    for _, a in qc[qc.Image_ID == iid].iterrows():
        g = LAB2GRP.get(a.label)
        if g is None:
            continue
        m = _full(a.mask_path)
        full[g] = m if g not in full else (full[g] | m)
    out = {}
    for g, m in full.items():
        gm = to_grid(m) & L
        if gm.sum():
            out[g] = gm
    return out


def load_feat_manifest():
    f = pd.read_csv(FEAT_CSV).query("usable").copy()
    f["image_id"] = _strip(f.image_id)
    return f


def fold_of_heldout():
    out = {}
    for k in range(5):
        h = pd.read_csv(MELNV / f"f2_fold{k}_heldout.csv")
        for iid in _strip(h.image_id):
            out[iid] = k
    return out


def load_test():
    """The 987 image test split. 1021 minus the 34 annotated images."""
    d = pd.read_csv(MELNV / "f2_fold0.csv")
    d["image_id"] = _strip(d.image_id)
    te = d.query("split == 'test'").reset_index(drop=True)
    X = torch.stack([PREPROCESS(Image.open((HAM / str(r)).resolve()).convert("RGB"))
                     for r in te.image_rel_path])
    y = (te.binary_label.values == MEL_IDX).astype(int)
    assert len(y) == 987 and y.sum() == 54, f"test split is {len(y)} images, {y.sum()} MEL"
    return X, y, te.image_id.tolist(), te