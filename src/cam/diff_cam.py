from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
    logits: (1, C)
    Returns (A, B, probs) where A is top-1, B is top-2, probs is (C,)
    """
    probs = torch.softmax(logits, dim=1)[0]
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
    reshape_transform: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff)
    where cam_diff is the plain positive map difference max(0, cam_A - cam_B).
    """
    cam = cam_cls(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )

    cam_A = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(A)])[0]
    vis_A = show_cam_on_image(rgb_float, cam_A, use_rgb=True)

    cam_B = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(B)])[0]
    vis_B = show_cam_on_image(rgb_float, cam_B, use_rgb=True)

    cam_diff = np.maximum(cam_A - cam_B, 0.0)
    cam_diff = cam_diff - cam_diff.min()
    cam_diff = cam_diff / (cam_diff.max() + 1e-8)
    vis_diff = show_cam_on_image(rgb_float, cam_diff, use_rgb=True)

    return cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff


def _run_finercam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    target_layer,
    A: int,
    B: int,
    reshape_transform: Optional[Callable] = None,
    comparison_categories: Optional[List[int]] = None,
    alpha: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Official-style FinerCAM helper:
    - left/middle panels use GradCAM(A) and GradCAM(B) as stable references
    - right panel uses FinerCAM for A against one or more comparison classes
    """
    cam_ref = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )

    cam_A = cam_ref(input_tensor=input_tensor, targets=[ClassifierOutputTarget(A)])[0]
    vis_A = show_cam_on_image(rgb_float, cam_A, use_rgb=True)

    cam_B = cam_ref(input_tensor=input_tensor, targets=[ClassifierOutputTarget(B)])[0]
    vis_B = show_cam_on_image(rgb_float, cam_B, use_rgb=True)

    finer = FinerCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )

    if comparison_categories is None or len(comparison_categories) == 0:
        comparison_categories = [B]

    try:
        finer_target = FinerWeightedTarget(
            main_category=A,
            comparison_categories=comparison_categories,
            alpha=alpha,
        )
        cam_diff = finer(input_tensor=input_tensor, targets=[finer_target])[0]
        vis_diff = show_cam_on_image(rgb_float, cam_diff, use_rgb=True)
        return cam_A, cam_B, cam_diff, vis_A, vis_B, vis_diff
    except Exception as e:
        raise RuntimeError(f"FinerCAM failed with FinerWeightedTarget: {e}") from e

def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model

def _compute_rollout_attention_no_norm(all_layer_matrices: list[torch.Tensor], start_layer: int = 0) -> torch.Tensor:
    """
    Chefer-style rollout over per-layer attribution matrices.
    Adds identity / residual, but does not row-normalize before multiplying.
    """
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens, device=all_layer_matrices[0].device).unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    matrices_aug = [mat + eye for mat in all_layer_matrices]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


def _compute_attention_probs_and_values(attn_module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Manual forward for a timm / BEiT-style attention module.
    Returns attention probabilities of shape (B, H, N, N) and values of shape (B, H, N, D).
    """
    bsz, n_tokens, dim = x.shape
    num_heads = attn_module.num_heads
    head_dim = dim // num_heads

    if hasattr(attn_module, "q_bias") and hasattr(attn_module, "v_bias") and attn_module.q_bias is not None and attn_module.v_bias is not None:
        zero_bias = torch.zeros_like(attn_module.v_bias)
        qkv_bias = torch.cat((attn_module.q_bias, zero_bias, attn_module.v_bias), dim=0)
        qkv = F.linear(x, attn_module.qkv.weight, qkv_bias)
    else:
        qkv = attn_module.qkv(x)

    qkv = qkv.reshape(bsz, n_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    scale = getattr(attn_module, "scale", head_dim ** -0.5)
    attn = (q * scale) @ k.transpose(-2, -1)

    if hasattr(attn_module, "relative_position_bias_table") and hasattr(attn_module, "relative_position_index"):
        table = attn_module.relative_position_bias_table
        index = attn_module.relative_position_index.view(-1)
        rel_pos_bias = table[index].view(n_tokens, n_tokens, -1).permute(2, 0, 1)
        attn = attn + rel_pos_bias.unsqueeze(0)

    attn = attn.softmax(dim=-1)
    if hasattr(attn_module, "attn_drop"):
        attn = attn_module.attn_drop(attn)

    return attn, v


def _forward_block_with_attention(block, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Manual forward through one transformer block while exposing attention probabilities
    that keep gradients. This is an approximate Chefer-style path for PanDerm / timm blocks.
    """
    attn_input = block.norm1(x)
    attn_probs, v = _compute_attention_probs_and_values(block.attn, attn_input)
    attn_probs.retain_grad()

    attn_out = (attn_probs @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2])
    attn_out = block.attn.proj(attn_out)
    if hasattr(block.attn, "proj_drop"):
        attn_out = block.attn.proj_drop(attn_out)

    if hasattr(block, "gamma_1") and block.gamma_1 is not None:
        attn_out = block.gamma_1 * attn_out
    if hasattr(block, "drop_path"):
        x = x + block.drop_path(attn_out)
    else:
        x = x + attn_out

    mlp_out = block.mlp(block.norm2(x))
    if hasattr(block, "gamma_2") and block.gamma_2 is not None:
        mlp_out = block.gamma_2 * mlp_out
    if hasattr(block, "drop_path"):
        x = x + block.drop_path(mlp_out)
    else:
        x = x + mlp_out

    return x, attn_probs


def _compute_chefer_like_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    target_idx: int,
    start_layer: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate Chefer-style transformer attribution for a timm / PanDerm ViT.
    This follows the official high-level recipe:
      1) class-specific backward pass,
      2) per-layer positive grad * attention,
      3) rollout across layers,
    but does not use full relprop because PanDerm is not implemented with custom relprop layers.
    """
    base_model = _unwrap_model(model)
    base_model.zero_grad(set_to_none=True)

    x = base_model.patch_embed(input_tensor)
    batch_size = x.shape[0]

    if hasattr(base_model, "cls_token") and base_model.cls_token is not None:
        cls_tokens = base_model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

    if hasattr(base_model, "pos_embed") and base_model.pos_embed is not None:
        x = x + base_model.pos_embed

    if hasattr(base_model, "pos_drop") and base_model.pos_drop is not None:
        x = base_model.pos_drop(x)

    attn_probs_all = []
    for block in base_model.blocks:
        x, attn_probs = _forward_block_with_attention(block, x)
        attn_probs_all.append(attn_probs)

    if hasattr(base_model, "norm") and base_model.norm is not None:
        x = base_model.norm(x)

    if hasattr(base_model, "fc_norm") and base_model.fc_norm is not None:
        pooled = x[:, 1:, :].mean(dim=1)
        pooled = base_model.fc_norm(pooled)
    else:
        pooled = x[:, 0]

    logits = base_model.head(pooled)
    score = logits[:, int(target_idx)].sum()
    score.backward(retain_graph=False)

    cams = []
    for attn_probs in attn_probs_all:
        grad = attn_probs.grad
        if grad is None:
            raise RuntimeError("Attention gradients were not retained for Chefer-style attribution.")
        cam = (grad * attn_probs).clamp(min=0).mean(dim=1)
        cams.append(cam)

    rollout = _compute_rollout_attention_no_norm(cams, start_layer=start_layer)
    cls_rollout = rollout[:, 0, 1:]

    n_patches = cls_rollout.shape[-1]
    side = int(np.sqrt(n_patches))
    if side * side != n_patches:
        raise ValueError(f"Cannot reshape Chefer-style tokens to square map: {n_patches}")

    heatmap_small = cls_rollout[0].reshape(side, side).detach().cpu().numpy()
    heatmap_small = heatmap_small - heatmap_small.min()
    heatmap_small = heatmap_small / (heatmap_small.max() + 1e-8)

    rgb_h, rgb_w = rgb_float.shape[:2]
    heatmap = cv2.resize(heatmap_small, (rgb_w, rgb_h), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
    return heatmap, overlay


@torch.no_grad()
def _compute_attention_map_from_attn_module(attn_module, x: torch.Tensor) -> torch.Tensor:
    bsz, n_tokens, dim = x.shape
    num_heads = attn_module.num_heads
    head_dim = dim // num_heads

    if hasattr(attn_module, "q_bias") and hasattr(attn_module, "v_bias") and attn_module.q_bias is not None and attn_module.v_bias is not None:
        zero_bias = torch.zeros_like(attn_module.v_bias)
        qkv_bias = torch.cat((attn_module.q_bias, zero_bias, attn_module.v_bias), dim=0)
        qkv = F.linear(x, attn_module.qkv.weight, qkv_bias)
    else:
        qkv = attn_module.qkv(x)

    qkv = qkv.reshape(bsz, n_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv[0], qkv[1], qkv[2]

    scale = getattr(attn_module, "scale", head_dim ** -0.5)
    attn = (q * scale) @ k.transpose(-2, -1)

    if hasattr(attn_module, "relative_position_bias_table") and hasattr(attn_module, "relative_position_index"):
        table = attn_module.relative_position_bias_table
        index = attn_module.relative_position_index.view(-1)
        rel_pos_bias = table[index].view(n_tokens, n_tokens, -1).permute(2, 0, 1)
        attn = attn + rel_pos_bias.unsqueeze(0)

    attn = attn.softmax(dim=-1)
    return attn.mean(dim=1)


@torch.no_grad()
def _compute_rollout_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_model = _unwrap_model(model)
    if not hasattr(base_model, "blocks"):
        raise ValueError("Rollout requires a Transformer model with a .blocks attribute.")

    x = base_model.patch_embed(input_tensor)
    batch_size = x.shape[0]

    if hasattr(base_model, "cls_token") and base_model.cls_token is not None:
        cls_tokens = base_model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

    if hasattr(base_model, "pos_embed") and base_model.pos_embed is not None:
        x = x + base_model.pos_embed

    if hasattr(base_model, "pos_drop") and base_model.pos_drop is not None:
        x = base_model.pos_drop(x)

    rollout = torch.eye(x.shape[1], device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)

    for block in base_model.blocks:
        attn_input = block.norm1(x)
        attn_map = _compute_attention_map_from_attn_module(block.attn, attn_input)
        attn_map = attn_map + torch.eye(attn_map.shape[-1], device=attn_map.device).unsqueeze(0)
        attn_map = attn_map / attn_map.sum(dim=-1, keepdim=True)
        rollout = rollout @ attn_map
        x = block(x)

    cls_rollout = rollout[:, 0, 1:]
    n_patches = cls_rollout.shape[-1]
    side = int(np.sqrt(n_patches))
    if side * side != n_patches:
        raise ValueError(f"Cannot reshape rollout tokens to square map: {n_patches}")

    heatmap_small = cls_rollout[0].reshape(side, side).detach().cpu().numpy()
    heatmap_small = heatmap_small - heatmap_small.min()
    heatmap_small = heatmap_small / (heatmap_small.max() + 1e-8)

    rgb_h, rgb_w = rgb_float.shape[:2]
    heatmap = cv2.resize(heatmap_small, (rgb_w, rgb_h), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
    return heatmap, overlay

def compute_cam_bundle(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    target_layer,
    reshape_transform: Optional[Callable] = None,
    method: str = "finercam",
    A: Optional[int] = None,
    B: Optional[int] = None,
    comparison_categories: Optional[List[int]] = None,
    alpha: float = 0.6,
):
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)

    if A is None or B is None:
        A, B, probs_t = pick_top2_classes(logits)
    else:
        probs_t = torch.softmax(logits, dim=1)[0]

    probs = probs_t.detach().cpu().numpy()
    probs_top3 = torch.topk(probs_t, 3).values.detach().cpu().tolist()

    cam_A, cam_B_grad, cam_diff_grad, vis_A, vis_B_grad, vis_diff_grad = _run_standard_cam(
        GradCAM,
        model,
        input_tensor,
        rgb_float,
        target_layer,
        A,
        B,
        reshape_transform=reshape_transform,
    )

    _, _, cam_finer, _, _, vis_finer = _run_finercam(
        model,
        input_tensor,
        rgb_float,
        target_layer,
        A,
        B,
        reshape_transform=reshape_transform,
        comparison_categories=comparison_categories,
        alpha=alpha,
    )

    cam_rollout, vis_rollout = _compute_rollout_heatmap(
        model=model,
        input_tensor=input_tensor,
        rgb_float=rgb_float,
    )

    cam_chefer, vis_chefer = _compute_chefer_like_heatmap(
        model=model,
        input_tensor=input_tensor,
        rgb_float=rgb_float,
        target_idx=A,
        start_layer=0,
    )

    return {
        "A": int(A),
        "B": int(B),
        "comparison_categories": [
            int(i)
            for i in (
                comparison_categories
                if comparison_categories is not None and len(comparison_categories) > 0
                else [B]
            )
        ],
        "probs": probs,
        "probs_top3": probs_top3,
        "cam_gradcam": cam_A,
        "cam_gradcam_B": cam_B_grad,
        "cam_gradcam_diff": cam_diff_grad,
        "cam_finercam": cam_finer,
        "cam_rollout": cam_rollout,
        "cam_chefer": cam_chefer,
        "overlay_gradcam": vis_A,
        "overlay_gradcam_B": vis_B_grad,
        "overlay_gradcam_diff": vis_diff_grad,
        "overlay_finercam": vis_finer,
        "overlay_rollout": vis_rollout,
        "overlay_chefer": vis_chefer,
    }