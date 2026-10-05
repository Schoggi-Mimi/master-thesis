import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from .cams import norm01
from .config import TOP_FRAC

# Fixed metric set. Primary: energy_excess. Decided before F2 results.
# Dropped as uninformative here: pointing game, structure hit, peak distance,
# Dice, IoU. They saturate because annotations cover about two thirds of the lesion.


def centre_map(L):
    cy, cx = center_of_mass(L.astype(float))
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    return norm01(-np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2))


def energy_excess(v, L, T):
    v = norm01(v)
    if T[L].sum() == 0 or T[L].all():
        return np.nan
    return float(v[T & L].sum() / (v[L].sum() + 1e-8) - T[L].mean())


def auc_in(v, L, T):
    v = norm01(v)
    y = T[L].astype(int)
    if y.sum() == 0 or y.all() or v[L].std() < 1e-9:
        return np.nan
    return float(roc_auc_score(y, v[L]))


def topk_excess(v, L, T, frac=TOP_FRAC):
    v = norm01(v)
    idx = np.flatnonzero(L)
    k = min(max(1, int(round(frac * L.size))), len(idx))
    hot = idx[np.argsort(-v.ravel()[idx])[:k]]
    return float(T.ravel()[hot].mean() - T[L].mean())


def corr_in(a, b, L):
    if a[L].std() < 1e-9 or b[L].std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a[L], b[L])[0, 1])


def lesion_auc(v, L):
    v = norm01(v)
    return float(roc_auc_score(L.ravel(), v.ravel())) if v.std() > 1e-9 else np.nan


def paired(x, y):
    """Wilcoxon on aligned series. Returns median difference x minus y."""
    q = pd.concat([pd.Series(x, name="x"), pd.Series(y, name="y")], axis=1).dropna()
    d = q.x - q.y
    p = wilcoxon(q.x, q.y).pvalue if len(q) >= 5 and d.abs().sum() > 0 else np.nan
    return {"diff": float(d.median()), "p": float(p), "wins": int((d > 0).sum()), "n": len(q)}


def summarise(s, digits=3):
    """mean ± SD and median [IQR], the format used in every table."""
    s = pd.Series(s).dropna()
    f = f"{{:.{digits}f}}"
    return {"mean_sd": f"{f.format(s.mean())} ± {f.format(s.std())}",
            "median_iqr": (f"{f.format(s.median())} "
                           f"[{f.format(s.quantile(.25))}, {f.format(s.quantile(.75))}]"),
            "n": len(s)}

def top_mask(v, frac=TOP_FRAC):
    k = max(1, int(round(frac * v.size)))
    m = np.zeros(v.size, bool)
    m[np.argsort(-np.asarray(v).ravel())[:k]] = True
    return m.reshape(v.shape)
    