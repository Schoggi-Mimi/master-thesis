# src/cam/diff_cam.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_grad_cam import FinerCAM, GradCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (ClassifierOutputTarget,
                                                  FinerWeightedTarget)


class LogitDiffTarget:
    def __init__(self, a: int, b: int, gamma: float = 0.6):
        self.a = int(a)
        self.b = int(b)
        self.gamma = float(gamma)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.dim() == 1:
            return model_output[self.a] - self.gamma * model_output[self.b]
        return model_output[:, self.a] - self.gamma * model_output[:, self.b]


def pick_top2_classes(logits: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
    """
    logits: (1,C)
    Returns (A, B, probs) where A is top-1, B is top-2, probs is (C,)
    """
    probs = torch.softmax(logits, dim=1)[0]  # (C,)
    top2 = torch.topk(probs, k=2).indices.detach().cpu().tolist()
    A, B = int(top2[0]), int(top2[1])
    return A, B, probs

# def _run_standard_cam(
#     cam_cls,
#     model: torch.nn.Module,
#     input_tensor: torch.Tensor,
#     rgb_float: np.ndarray,
#     target_layer,
#     A: int,
#     B: int,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Returns (cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff)
#     where cam_diff is computed by targeting logit(A)-logit(B).
#     """
#     cam = cam_cls(model=model, target_layers=[target_layer])
def _run_standard_cam(
    cam_cls,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    target_layer,
    A: int,
    B: int,
    reshape_transform=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam = cam_cls(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    cam_A = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(A)])[0]
    vis_A = show_cam_on_image(rgb_float, cam_A, use_rgb=True)

    cam_B = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(B)])[0]
    vis_B = show_cam_on_image(rgb_float, cam_B, use_rgb=True)

    cam_diff = cam(input_tensor=input_tensor, targets=[LogitDiffTarget(A, B, gamma=0.6)])[0]
    vis_diff = show_cam_on_image(rgb_float, cam_diff, use_rgb=True)

    return cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff

def _run_finercam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    target_layer,
    A: int,
    B: int,
    reshape_transform=None,
    comparison_categories: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
# def _run_finercam(
#     model: torch.nn.Module,
#     input_tensor: torch.Tensor,
#     rgb_float: np.ndarray,
#     target_layer,
#     A: int,
#     B: int,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Official FinerCAM:
    - Left/Middle panels: use GradCAM(A) and GradCAM(B) as stable references
    - Right panel: FinerCAM for A vs B

    Since FinerCAM API can differ slightly across versions, we try a few patterns.
    """
    # reference CAMs with GradCAM
    # cam_ref = GradCAM(model=model, target_layers=[target_layer])
    cam_ref = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    cam_A = cam_ref(input_tensor=input_tensor, targets=[ClassifierOutputTarget(A)])[0]
    vis_A = show_cam_on_image(rgb_float, cam_A, use_rgb=True)

    cam_B = cam_ref(input_tensor=input_tensor, targets=[ClassifierOutputTarget(B)])[0]
    vis_B = show_cam_on_image(rgb_float, cam_B, use_rgb=True)

    # finer = FinerCAM(model=model, target_layers=[target_layer])
    finer = FinerCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    try:
        if comparison_categories is None or len(comparison_categories) == 0:
            comparison_categories = [B]
        finer_target = FinerWeightedTarget(
            main_category=A,
            comparison_categories=comparison_categories,
            alpha=0.6,
        )
        cam_diff = finer(input_tensor=input_tensor, targets=[finer_target])[0]
        vis_diff = show_cam_on_image(rgb_float, cam_diff, use_rgb=True)
        return cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff
    except Exception as e:
        raise RuntimeError(f"FinerCAM failed with FinerWeightedTarget: {e}")

def compute_cam_triplet(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    target_layer,
    reshape_transform: Optional[Callable] = None,
    method: str = "gradcam",
    A: Optional[int] = None,
    B: Optional[int] = None,
    comparison_categories: Optional[List[int]] = None,
):
# def compute_cam_triplet(
#     model: torch.nn.Module,
#     input_tensor: torch.Tensor, # (1,3,H,W) already on device
#     rgb_float: np.ndarray, # (H,W,3) float in [0,1], same size as input_tensor
#     target_layer,
#     method: str = "gradcam", # "gradcam" or "layercam" or "finercam"
#     A: Optional[int] = None,
#     B: Optional[int] = None,
# ):
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
        # cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_standard_cam(
        #     GradCAM, model, input_tensor, rgb_float, target_layer, A, B
        # )
        cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_standard_cam(
            GradCAM, model, input_tensor, rgb_float, target_layer, A, B, reshape_transform=reshape_transform
        )
    elif method_l == "layercam":
        # cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_standard_cam(
        #     LayerCAM, model, input_tensor, rgb_float, target_layer, A, B
        # )
        cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_standard_cam(
            LayerCAM, model, input_tensor, rgb_float, target_layer, A, B, reshape_transform=reshape_transform
        )
    elif method_l == "finercam":
        # if A is None or B is None:
        #     A, B, _ = pick_top2_classes(logits)
        #     print(f"[info] FinerCAM defaulting to top-2 classes: A={A}, B={B}")
            # raise ValueError("FinerCAM requires explicit A and B (use fixed mode).")
        # cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_finercam(
        #     model, input_tensor, rgb_float, target_layer, A, B
        # )
        cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff = _run_finercam(
            model,
            input_tensor,
            rgb_float,
            target_layer,
            A,
            B,
            reshape_transform=reshape_transform,
            comparison_categories=comparison_categories,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "A": int(A),
        "B": int(B),
        "comparison_categories": [int(i) for i in (comparison_categories if comparison_categories is not None and len(comparison_categories) > 0 else [B])],
        "probs": probs, # np.ndarray (C,)
        "probs_top3": probs_top3, # list
        "cam_A": cam_A, # np.ndarray (H,W)
        "cam_B": cam_B,
        "cam_diff": cam_diff,
        "overlay_A": vis_A, # np.ndarray (H,W,3) uint8
        "overlay_B": vis_B,
        "overlay_diff": vis_diff,
    }