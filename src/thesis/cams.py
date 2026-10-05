import numpy as np
import torch
import torch.nn.functional as tF

from .config import BLOCK, MEL_IDX, NV_IDX
from run_class_finetuning_ha import (build_attention_gradcam_map, _unwrap_model,
                                     unpack_model_outputs)


def norm01(v):
    v = np.asarray(v, np.float32)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-8)


def _forward(model, x):
    base = _unwrap_model(model)
    base.clear_xai_state()
    logits, _ = unpack_model_outputs(
        model(x, return_patch_tokens=True, store_attn=True, attn_layer=BLOCK))
    return base, logits


def class_cams(model, x):
    """MEL map and NV map from one forward pass. Both min max normalised."""
    with torch.enable_grad():
        base, logits = _forward(model, x)
        out = []
        for i in (MEL_IDX, NV_IDX):
            t = torch.full((1,), i, device=logits.device, dtype=torch.long)
            out.append(build_attention_gradcam_map(model, logits, t, create_graph=False)
                       .detach()[0].float().cpu().numpy())
    base.clear_xai_state()
    return norm01(out[0]), norm01(out[1])


def target_ref(model, x, label):
    """The only place target and reference are chosen. label is 'MEL' or 'NV'."""
    assert label in ("MEL", "NV"), label
    a, b = class_cams(model, x)
    return (a, b) if label == "MEL" else (b, a)


def diff_cam(model, x, label):
    t, r = target_ref(model, x, label)
    return np.maximum(t - r, 0)


def finer_cam(model, x, label, alpha=0.6):
    """Contrast formed in logit space before any ReLU. alpha 0 equals Grad-CAM."""
    ti, ri = (MEL_IDX, NV_IDX) if label == "MEL" else (NV_IDX, MEL_IDX)
    with torch.enable_grad():
        base, logits = _forward(model, x)
        A = base.blocks[BLOCK].attn.last_attn
        G = torch.autograd.grad(logits[0, ti] - alpha * logits[0, ri], A)[0]
    h, w = base.patch_embed.patch_shape
    if bool(getattr(base, "use_mean_pooling", False)):
        v = tF.relu((A[:, :, 1:, 1:] * G[:, :, 1:, 1:]).mean(1).sum(1))
    else:
        v = tF.relu((A[:, :, 0, 1:] * G[:, :, 0, 1:]).mean(1))
    base.clear_xai_state()
    return norm01(v[0].detach().float().cpu().numpy().reshape(h, w))