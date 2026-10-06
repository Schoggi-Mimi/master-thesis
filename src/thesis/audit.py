import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

from .cams import class_cams, finer_cam, target_ref
from .config import MEL_IDX, NV_IDX, OUT
from .data import native, to_tensor
from .metrics import auc_in, centre_map


def run_audit(ev, lesion, hum, X_test, y_test, ids_test, qc, m_ha0, m_ha5):
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    chk("index convention", (MEL_IDX, NV_IDX) == (0, 1))
    chk("test split 987 with 54 MEL", len(y_test) == 987 and y_test.sum() == 54,
        f"{len(y_test)} / {y_test.sum()}")
    chk("no annotated image in test", not (set(ids_test) & set(qc.Image_ID)))
    chk("lesion masks non degenerate",
        all(0 < v.sum() < v.size for v in lesion.values()), f"{len(lesion)}")
    chk("HUM has positives and negatives",
        all(0 < hum[i][lesion[i]].sum() < lesion[i].sum() for i in hum), f"{len(hum)}")

    missing = [p for p in qc.mask_path if not Path(p).exists()]
    chk("all annotation mask paths exist", not missing, f"{len(missing)} missing")

    iid = sorted(hum)[0]
    r = ev.loc[iid]
    x = to_tensor(native(r.image_rel_path))
    a, b = class_cams(m_ha5, x)
    t, _ = target_ref(m_ha5, x, r.gt_label)
    chk("ha0 and ha5 differ", not np.allclose(class_cams(m_ha0, x)[0], a))
    chk("target follows ground truth",
        np.allclose(t, a if r.gt_label == "MEL" else b))
    c0 = np.corrcoef(finer_cam(m_ha5, x, r.gt_label, 0.0).ravel(), t.ravel())[0, 1]
    chk("FinerCAM alpha 0 equals Grad-CAM", c0 > 0.99999, f"{c0:.6f}")

    u = hum[iid][lesion[iid]].astype(int)
    v = t[lesion[iid]]
    chk("AUC rank invariant", abs(roc_auc_score(u, v * 7) - roc_auc_score(u, v)) < 1e-12)
    chk("AUC argument order", roc_auc_score(u, u.astype(float)) == 1.0)

    ref_path = OUT / "HUM_reference.npz"
    if ref_path.exists():
        ref = dict(np.load(ref_path))
        same = all(i in hum and np.array_equal(ref[i], hum[i]) for i in ref)
        chk("HUM identical to 14d reference", same and len(ref) == len(hum),
            f"{len(ref)} ref, {len(hum)} new")

    cb = np.median([auc_in(centre_map(lesion[i]), lesion[i], hum[i]) for i in hum])
    chk("centre baseline reproduces 0.828", abs(cb - 0.828) < 0.01, f"{cb:.3f}")
    hi = np.median([auc_in(target_ref(m_ha5, to_tensor(native(ev.loc[i].image_rel_path)),
                                      ev.loc[i].gt_label)[0], lesion[i], hum[i]) for i in hum])
    chk("aligned target vs clinician reproduces 0.743", abs(hi - 0.743) < 0.01, f"{hi:.3f}")

    A = pd.DataFrame(rows)
    print(A.to_string(index=False))
    fails = A[~A["pass"]]
    if len(fails):
        raise RuntimeError(f"{len(fails)} audit failures: {fails.check.tolist()}")
    print(f"\nall {len(A)} checks pass")
    return A