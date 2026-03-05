# src/cam/diff_cam.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pytorch_grad_cam import FinerCAM, GradCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class LogitDiffTarget:
    """
    Target for CAM: logit[A] - logit[B]
    Works with pytorch-grad-cam which often passes a (C,) tensor per sample.
    """

    def __init__(self, a: int, b: int):
        self.a = int(a)
        self.b = int(b)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        # model_output can be (C,) or (B,C)
        if model_output.dim() == 1:
            return model_output[self.a] - model_output[self.b]
        return model_output[:, self.a] - model_output[:, self.b]


def pick_top2_classes(logits: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
    """
    logits: (1,C)
    Returns (A, B, probs) where A is top-1, B is top-2, probs is (C,)
    """
    probs = torch.softmax(logits, dim=1)[0]  # (C,)
    top2 = torch.topk(probs, k=2).indices.detach().cpu().tolist()
    A, B = int(top2[0]), int(top2[1])
    return A, B, probs

def _run_standard_cam(
    cam_cls,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    target_layer,
    A: int,
    B: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff)
    where cam_diff is computed by targeting logit(A)-logit(B).
    """
    cam = cam_cls(model=model, target_layers=[target_layer])

    cam_A = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(A)])[0]
    vis_A = show_cam_on_image(rgb_float, cam_A, use_rgb=True)

    cam_B = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(B)])[0]
    vis_B = show_cam_on_image(rgb_float, cam_B, use_rgb=True)

    cam_diff = cam(input_tensor=input_tensor, targets=[LogitDiffTarget(A, B)])[0]
    vis_diff = show_cam_on_image(rgb_float, cam_diff, use_rgb=True)

    return cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff

def _run_finercam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    target_layer,
    A: int,
    B: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Official FinerCAM:
    - Left/Middle panels: use GradCAM(A) and GradCAM(B) as stable references
    - Right panel: FinerCAM for A vs B

    Since FinerCAM API can differ slightly across versions, we try a few patterns.
    """
    # reference CAMs with GradCAM
    cam_ref = GradCAM(model=model, target_layers=[target_layer])
    cam_A = cam_ref(input_tensor=input_tensor, targets=[ClassifierOutputTarget(A)])[0]
    vis_A = show_cam_on_image(rgb_float, cam_A, use_rgb=True)

    cam_B = cam_ref(input_tensor=input_tensor, targets=[ClassifierOutputTarget(B)])[0]
    vis_B = show_cam_on_image(rgb_float, cam_B, use_rgb=True)

    finer = FinerCAM(model=model, target_layers=[target_layer])

    # Pattern 1 (most common): treat FinerCAM as a CAM method but keep our diff objective
    try:
        # print("[finercam] using Pattern 1: LogitDiffTarget(A,B)")
        cam_diff = finer(input_tensor=input_tensor, targets=[LogitDiffTarget(A, B)])[0]
        vis_diff = show_cam_on_image(rgb_float, cam_diff, use_rgb=True)
        return cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff
    except Exception as e:
        print("[finercam] Pattern 1 failed:", repr(e))
        pass

    # Pattern 2: FinerCAM expects a normal target (A) and a "comparison" argument.
    # Try introspecting common arg names.
    print("[finercam] using Pattern 2: explicit comparison args")
    import inspect
    sig = inspect.signature(finer.__call__)
    params = sig.parameters

    kwargs = {}
    # common names seen in implementations
    for name in ["comparison_targets", "contrast_targets", "negative_targets", "other_targets"]:
        if name in params:
            kwargs[name] = [ClassifierOutputTarget(B)]
            break
    for name in ["comparison_category", "contrast_category", "negative_category", "other_category"]:
        if name in params:
            kwargs[name] = B
            break

    if kwargs:
        cam_diff = finer(input_tensor=input_tensor, targets=[ClassifierOutputTarget(A)], **kwargs)[0]
        vis_diff = show_cam_on_image(rgb_float, cam_diff, use_rgb=True)
        return cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff

    # If we get here, we couldn't find a supported calling pattern.
    raise RuntimeError(
        "FinerCAM call pattern not recognized in this grad-cam version. "
        "Paste the FinerCAM error traceback and I will adapt the wrapper."
    )

def compute_cam_triplet(
    model: torch.nn.Module,
    input_tensor: torch.Tensor, # (1,3,H,W) already on device
    rgb_float: np.ndarray, # (H,W,3) float in [0,1], same size as input_tensor
    target_layer,
    method: str = "gradcam", # "gradcam" or "layercam" or "finercam"
    A: Optional[int] = None,
    B: Optional[int] = None,
):
    """
    Computes:
      - CAM(A)
      - CAM(B)
      - Comparison CAM (A vs B)
        - gradcam/layercam: uses logit(A)-logit(B) target (Diff-CAM)
        - finercam: official FinerCAM (right panel), GradCAM refs on left/middle
    """
    model.eval()

    # First forward pass to get probabilities and pick A/B if not given.
    with torch.no_grad():
        # (B=1,3,H, W) -> (B=1,C=7)
        logits = model(input_tensor)

    if A is None or B is None:
        A, B, probs_t = pick_top2_classes(logits)
    else:
        probs_t = torch.softmax(logits, dim=1)[0]

    probs = probs_t.detach().cpu().numpy()
    probs_top3 = torch.topk(probs_t, 3).values.detach().cpu().tolist()

    method_l = method.lower()
    if method_l == "gradcam":
        cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_standard_cam(
            GradCAM, model, input_tensor, rgb_float, target_layer, A, B
        )
    elif method_l == "layercam":
        cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_standard_cam(
            LayerCAM, model, input_tensor, rgb_float, target_layer, A, B
        )
    elif method_l == "finercam":
        if A is None or B is None:
            A, B, _ = pick_top2_classes(logits)
            print(f"[info] FinerCAM defaulting to top-2 classes: A={A}, B={B}")
            # raise ValueError("FinerCAM requires explicit A and B (use fixed mode).")
        cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_finercam(
            model, input_tensor, rgb_float, target_layer, A, B
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "A": int(A),
        "B": int(B),
        "probs": probs, # np.ndarray (C,)
        "probs_top3": probs_top3, # list
        "cam_A": cam_A, # np.ndarray (H,W)
        "cam_B": cam_B,
        "cam_diff": cam_diff,
        "overlay_A": vis_A, # np.ndarray (H,W,3) uint8
        "overlay_B": vis_B,
        "overlay_diff": vis_diff,
    }