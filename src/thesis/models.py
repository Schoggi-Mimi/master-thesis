import torch
import numpy as np

from .config import CK5, CK_F2, DEVICE, MEL_IDX
from run_class_finetuning_ha import remap_norm_keys_for_pooling
from scripts.generate_finer_cam_panderm import (
    build_panderm_model, remap_official_finetune_checkpoint_keys,
    extract_checkpoint_state_dict, infer_panderm_variant_from_state_dict)

_CACHE = {}


def ckpt_args(path):
    a = torch.load(path, map_location="cpu", weights_only=False).get("args")
    return vars(a) if a is not None and not isinstance(a, dict) else (a or {})


def load_ckpt(path, pooling="mean", cache=True):
    key = (str(path), pooling)
    if cache and key in _CACHE:
        return _CACHE[key]
    obj = torch.load(path, map_location="cpu", weights_only=False)
    raw, _ = extract_checkpoint_state_dict(obj)
    st = remap_official_finetune_checkpoint_keys(raw)
    m = build_panderm_model(num_classes=2, variant=infer_panderm_variant_from_state_dict(st),
                            use_mean_pooling=(pooling == "mean"))
    miss, unexp = m.load_state_dict(remap_norm_keys_for_pooling(st, m), strict=False)
    bad = [k for k in list(miss) + list(unexp) if k.startswith(("norm.", "fc_norm.", "head."))]
    assert not bad, f"{path.name}: critical keys {bad}"
    m.to(DEVICE).eval()
    for q in m.parameters():
        q.requires_grad_(True)
    if cache:
        _CACHE[key] = m
    return m


def ck5(name, pooling="gap"):
    """name in ha0, ha3, ha5. checkpoints5, batch 64."""
    return load_ckpt(CK5 / f"checkpoint-best-{pooling}-{name}.pth",
                     pooling="mean" if pooling == "gap" else "cls")


def f2_path(alpha, beta, fold):
    bt = str(float(beta)).replace(".", "p")
    hits = [p for p in CK_F2.glob(f"checkpoint-best-f2b{bt}fold{fold}_*.pth")
            if f"_ha{float(alpha)}_" in p.name]
    assert len(hits) == 1, f"alpha {alpha} beta {beta} fold {fold}: {len(hits)} files"
    return hits[0]


def f2(alpha, beta, fold):
    return load_ckpt(f2_path(alpha, beta, fold), cache=False)


def clear_cache():
    _CACHE.clear()


@torch.no_grad()
def predict_mel(model, X, bs=32):
    out = []
    for i in range(0, len(X), bs):
        o = model(X[i:i + bs].to(DEVICE))
        l = o["logits"] if isinstance(o, dict) else (o[0] if isinstance(o, (tuple, list)) else o)
        out.append(torch.softmax(l, 1)[:, MEL_IDX].cpu().numpy())
    return np.concatenate(out)