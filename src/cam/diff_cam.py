# src/cam/diff_cam.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, LayerCAM
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


@dataclass
class CamResult:
    A: int
    B: int
    probs_top3: list
    cam_A: np.ndarray # (H,W)
    cam_B: np.ndarray # (H,W)
    cam_diff: np.ndarray # (H,W)
    overlay_A: np.ndarray # (H,W,3) uint8
    overlay_B: np.ndarray # (H,W,3) uint8
    overlay_diff: np.ndarray # (H,W,3) uint8
    probs: np.ndarray  # (C,)


def pick_top2_classes(logits: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
    """
    logits: (1,C)
    Returns (A, B, probs) where A is top-1, B is top-2, probs is (C,)
    """
    probs = torch.softmax(logits, dim=1)[0]  # (C,)
    top2 = torch.topk(probs, k=2).indices.detach().cpu().tolist()
    A, B = int(top2[0]), int(top2[1])
    return A, B, probs


def compute_gradcam_triplet(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,     # (1,3,H,W) already on device
    rgb_float: np.ndarray,          # (H,W,3) float in [0,1], same size as input_tensor
    target_layer,
    method: str = "gradcam",        # "gradcam" or "layercam"
    A: Optional[int] = None,
    B: Optional[int] = None,
) -> CamResult:
    """
    Computes:
      - CAM(A)
      - CAM(B)
      - Diff-CAM(A-B)

    If A/B not provided, uses top-2 predicted classes.
    """
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)  # (1,C)

    if A is None or B is None:
        A, B, probs = pick_top2_classes(logits)
    else:
        probs = torch.softmax(logits, dim=1)[0]

    cam_impl = LayerCAM if method.lower() == "layercam" else GradCAM
    cam = cam_impl(model=model, target_layers=[target_layer])

    # CAM(A)
    cam_A = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(A)])[0]
    overlay_A = show_cam_on_image(rgb_float, cam_A, use_rgb=True)

    # CAM(B)
    cam_B = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(B)])[0]
    overlay_B = show_cam_on_image(rgb_float, cam_B, use_rgb=True)

    # Diff CAM(A-B)
    cam_diff = cam(input_tensor=input_tensor, targets=[LogitDiffTarget(A, B)])[0]
    overlay_diff = show_cam_on_image(rgb_float, cam_diff, use_rgb=True)

    probs_top3 = torch.topk(probs, 3).values.detach().cpu().tolist()

    probs_np = probs.detach().cpu().numpy()

    return CamResult(
        A=A,
        B=B,
        probs_top3=probs_top3,
        cam_A=cam_A,
        cam_B=cam_B,
        cam_diff=cam_diff,
        overlay_A=overlay_A,
        overlay_B=overlay_B,
        overlay_diff=overlay_diff,
        probs=probs_np,
    )