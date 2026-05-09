"""
python -m scripts.generate_finer_cam_panderm \
  --csv data/HAM10000/ham_test_cam_qualitative_stratified_10.csv \
  --image_col image_rel_path \
  --img_dir data/HAM10000 \
  --gt_col gt_label \
  --checkpoint external/checkpoints/checkpoint-best-ham.pth \
  --checkpoint_model_type panderm \
  --class_preset ham \
  --out_dir outputs/qual/cam_baseline_gt_topk3 \
  --num_samples 10 \
  --method finercam \
  --compare_mode gt_topk_non_target \
  --topk_compare 3 \
  --alpha 0.8 \
  --panel_items rgb_gt_mask,gradcam_a,gradcam_b,map_diff,finercam \
  --mask_root data/HAM10000 \
  --mask_col mask_rel_path

python -m scripts.generate_finer_cam_panderm \
  --csv data/HAM10000/ham_test_cam_qualitative_stratified_10.csv \
  --image_col image_rel_path \
  --img_dir data/HAM10000 \
  --gt_col gt_label \
  --checkpoint external/checkpoints/checkpoint-best-cropmask-base-weighted-nomix.pth \
  --checkpoint_model_type panderm \
  --class_preset ham \
  --out_dir outputs/qual/cam_cropmask_gt_topk3 \
  --num_samples 10 \
  --method finercam \
  --compare_mode gt_topk_non_target \
  --topk_compare 3 \
  --alpha 0.8 \
  --panel_items rgb_gt_mask,gradcam_a,gradcam_b,map_diff,finercam \
  --crop_with_mask \
  --mask_root data/HAM10000 \
  --mask_col mask_rel_path \
  --crop_margin 0.25 \
  --min_crop_frac 0.30

python -m scripts.generate_finer_cam_panderm \
  --csv data/HAM10000/ham_test_cam_qualitative_stratified_10.csv \
  --image_col image_rel_path \
  --img_dir data/HAM10000 \
  --gt_col gt_label \
  --checkpoint external/checkpoints/checkpoint-best-softmask-base-weighted-nomix.pth \
  --checkpoint_model_type panderm \
  --class_preset ham \
  --out_dir outputs/qual/cam_softmask_gt_topk3 \
  --num_samples 10 \
  --method finercam \
  --compare_mode gt_topk_non_target \
  --topk_compare 3 \
  --alpha 0.8 \
  --panel_items rgb_gt_mask,gradcam_a,gradcam_b,map_diff,finercam \
  --mask_root data/HAM10000 \
  --mask_col mask_rel_path

python -m scripts.generate_finer_cam_panderm \
  --csv data/HAM10000/ham_test_cam_qualitative_stratified_10.csv \
  --image_col image_rel_path \
  --img_dir data/HAM10000 \
  --gt_col gt_label \
  --checkpoint external/checkpoints/checkpoint-best-HA_lam1.pth \
  --checkpoint_model_type panderm \
  --class_preset ham \
  --out_dir outputs/qual/cam_ha_gt_topk3 \
  --num_samples 10 \
  --method finercam \
  --compare_mode gt_topk_non_target \
  --topk_compare 3 \
  --alpha 0.8 \
  --panel_items rgb_gt_mask,gradcam_a,gradcam_b,map_diff,finercam \
  --mask_root data/HAM10000 \
  --mask_col mask_rel_path

python -m scripts.generate_finer_cam_panderm \
  --csv data/HAM10000/ham_test_cam_qualitative_stratified_10.csv \
  --image_col image_rel_path \
  --img_dir data/HAM10000 \
  --gt_col gt_label \
  --checkpoint external/weights/checkpoint-best-seggate.pth \
  --checkpoint_model_type seggate \
  --use_seg_gate \
  --class_preset ham \
  --out_dir outputs/qual/cam_seggate_gt_topk3 \
  --num_samples 10 \
  --method finercam \
  --compare_mode gt_topk_non_target \
  --topk_compare 3 \
  --alpha 0.8 \
  --panel_items rgb_gt_mask,gate_weighted_gradcam_a,gate_weighted_gradcam_b,gate_weighted_map_diff,gate_weighted_finercam \
  --mask_root data/HAM10000 \
  --mask_col mask_rel_path
"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.registry is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release.*",
    category=UserWarning,
)

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from src.cam.diff_cam import compute_cam_bundle
from src.utils.vis_panel import make_panel_with_subtitles

REPO_ROOT = Path(__file__).resolve().parents[1]
PANDERM_CLASSIFICATION_DIR = (REPO_ROOT / "external" / "PanDerm" / "classification").resolve()
if str(PANDERM_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(PANDERM_CLASSIFICATION_DIR))

from external.PanDerm.classification.models.modeling_finetune_relprop import \
    build_panderm_relprop_from_model
from models.builder import get_eval_transforms  # type: ignore
from models.modeling_finetune import (  # type: ignore
    panderm_base_patch16_224_finetune, panderm_large_patch16_224_finetune)

# HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM_CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
# BCN_CLASSES = ["AKIEC", "BCC", "MEL", "NV", "SK", "SL", "SCC", "DF", "VAS"]
BCN_CLASSES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "melanoma",
    "nevus",
    "seborrheic keratosis",
    "solar lentigo",
    "squamous cell carcinoma",
    "dermatofibroma",
    "vascular lesion",
]

CLASS_PRESETS = {
    "ham": HAM_CLASSES,
    "bcn": BCN_CLASSES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RGB / A / B / Finer-CAM panels for a fine-tuned PanDerm checkpoint.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with column 'image' or 'isic_id'.")
    parser.add_argument(
        "--image_col",
        type=str,
        default=None,
        help="Optional image column name. Use this for multitask CSVs, e.g. image_rel_path.",
    )
    parser.add_argument("--img_dir", type=str, required=True, help="Folder containing JPG images.")
    parser.add_argument(
        "--gt_col",
        type=str,
        default=None,
        help="Optional ground-truth label column, e.g. dx or label.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PanDerm fine-tuned .pt checkpoint.")
    parser.add_argument(
        "--checkpoint_model_type",
        type=str,
        default="auto",
        choices=["auto", "panderm", "multitask", "seggate"],
        help=(
            "How to load the checkpoint. "
            "panderm loads a normal PanDerm classifier. "
            "multitask loads the classification + segmentation wrapper without gating. "
            "seggate loads the classification + segmentation wrapper with segmentation gating. "
            "auto detects seg_head keys and uses multitask unless --use_seg_gate is passed."
        ),
    )
    parser.add_argument(
        "--use_seg_gate",
        action="store_true",
        default=False,
        help="Use the segmentation-gated classification path when loading a multitask/seggate checkpoint.",
    )
    parser.add_argument(
        "--seg_gate_bg_keep",
        type=float,
        default=0.15,
        help="Background keep value used by the segmentation gate. Must match training.",
    )
    parser.add_argument(
        "--seg_gate_detach",
        action="store_true",
        default=True,
        help="Detach segmentation probability before gating. This should match the training setting.",
    )
    parser.add_argument(
        "--seg_gate_no_detach",
        action="store_false",
        dest="seg_gate_detach",
        help="Do not detach segmentation probability before gating.",
    )
    parser.add_argument(
        "--class_preset",
        type=str,
        default="ham",
        choices=["ham", "bcn"],
        help="Class-name preset matching the fine-tuned baseline-4 checkpoint.",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="Optional comma-separated class names overriding --class_preset.",
    )
    parser.add_argument("--out_dir", type=str, default="outputs/panderm_cam", help="Output folder.")
    parser.add_argument("--image_size", type=int, default=224, help="PanDerm input size. Default: 224.")
    parser.add_argument("--num_samples", type=int, default=10, help="How many images to process.")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps (default: auto).")
    parser.add_argument("--method", type=str, default="finercam", choices=["gradcam", "layercam", "finercam"], help="CAM backend for the main triplet.")
    parser.add_argument(
        "--target_block_index",
        type=int,
        default=-2,
        help=(
            "Transformer block index used for CAM target layer. "
            "Default -2 = second-to-last block. Try -4, -6, -8 for layer sensitivity."
        ),
    )
    parser.add_argument(
        "--compare_mode",
        type=str,
        default="top2",
        choices=["top2", "fixed", "gt_pair", "pred_topk_non_target", "gt_topk_non_target"],
        help=(
            "How to choose the main target class A and the comparison classes.\n"
            "top2 = predicted top1/top2 (single comparison class).\n"
            "fixed = user-defined --A/--B used as A/B exactly (single comparison class).\n"
            "gt_pair = use --A/--B as the pair; per image set A=gt_label and B=the other one (single comparison class).\n"
            "pred_topk_non_target = A is predicted top1, comparison classes are the top-k non-target predictions.\n"
            "gt_topk_non_target = A is gt_label, comparison classes are the top-k non-target predictions."
        ),
    )
    parser.add_argument("--A", type=str, default=None, help="Fixed target class name, e.g. MEL.")
    parser.add_argument("--B", type=str, default=None, help="Fixed comparison class name, e.g. NV.")
    parser.add_argument(
        "--topk_compare",
        type=int,
        default=1,
        help="Number of non-target predicted classes to use as comparison categories for FinerCAM. Used by pred_topk_non_target / gt_topk_non_target.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Alpha weight for FinerCAM comparison categories. Default: 0.6.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="If set, also save one metadata JSON file per image. Default: off.",
    )
    parser.add_argument(
        "--save_raw_cams",
        action="store_true",
        help="If set, save raw CAM arrays as .npy files for quantitative localization metrics.",
    )
    parser.add_argument(
        "--panel_scale",
        type=float,
        default=1.35,
        help="Scale factor used to enlarge the tiles in the saved panel.",
    )
    parser.add_argument(
        "--show_relprop_row",
        action="store_true",
        help="If set, add a third row with relprop-Chefer maps. Default: off.",
    )
    parser.add_argument(
        "--show_extra_rows",
        action="store_true",
        default=False,
        help="If set, also compute and show Rollout and Chefer-style rows. Default: off, only RGB/GradCAM/FinerCAM row is generated.",
    )
    parser.add_argument(
        "--panel_items",
        type=str,
        default="rgb_gt_mask,gradcam_a,gradcam_b,map_diff,finercam",
        help=(
            "Comma-separated first-row panel items. "
            "Default: rgb_gt_mask,gradcam_a,gradcam_b,map_diff,finercam. "
            "Options: rgb,rgb_gt_mask,gt_mask,seg_gate,gradcam_a,gradcam_b,map_diff,finercam,"
            "gate_weighted_gradcam_a,gate_weighted_gradcam_b,gate_weighted_map_diff,gate_weighted_finercam. "
            "For SegGate visualizations, use: "
            "rgb_gt_mask,gate_weighted_gradcam_a,gate_weighted_gradcam_b,gate_weighted_map_diff,gate_weighted_finercam."
        ),
    )
    parser.add_argument(
        "--crop_with_mask",
        action="store_true",
        help="If set, crop each image around its segmentation mask before CAM generation. Use this for cropmask-trained checkpoints.",
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default=None,
        help="Root folder used to resolve mask paths when --crop_with_mask is set. Usually data/HAM10000.",
    )
    parser.add_argument(
        "--mask_col",
        type=str,
        default="mask_rel_path",
        help="CSV column containing the mask path when --crop_with_mask is set.",
    )
    parser.add_argument(
        "--crop_margin",
        type=float,
        default=0.25,
        help="Relative crop margin around the lesion mask bbox. Must match cropmask training.",
    )
    parser.add_argument(
        "--min_crop_frac",
        type=float,
        default=0.30,
        help="Minimum crop side as fraction of the shorter image side. Must match cropmask training.",
    )
    return parser.parse_args()


def get_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_class_names(args: argparse.Namespace) -> list[str]:
    if args.class_names is not None:
        names = [x.strip() for x in args.class_names.split(",") if x.strip()]
        if len(names) == 0:
            raise ValueError("--class_names was provided but no valid class names were parsed.")
        return names
    return CLASS_PRESETS[args.class_preset]



def build_class_maps(class_names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    return class_to_idx, idx_to_class

def resolve_gt_label_from_row(row, class_names, class_to_idx, gt_col=None):
    if gt_col is None:
        gt_col = "gt_label"

    if gt_col not in row:
        raise ValueError(f"Ground-truth column '{gt_col}' not found.")

    raw_gt = row[gt_col]

    if pd.isna(raw_gt):
        raise ValueError(f"Ground-truth value in column '{gt_col}' is NaN.")

    # Numeric label, e.g. label=2
    if isinstance(raw_gt, (int, np.integer)) or (
        isinstance(raw_gt, float) and float(raw_gt).is_integer()
    ):
        idx = int(raw_gt)
        return class_names[idx]

    gt = str(raw_gt).strip()

    # Direct match, e.g. BKL
    if gt in class_to_idx:
        return gt

    # Case-insensitive match, e.g. bkl -> BKL
    gt_upper = gt.upper()
    upper_to_name = {name.upper(): name for name in class_names}
    if gt_upper in upper_to_name:
        return upper_to_name[gt_upper]

    gt_lower = gt.lower()
    lower_to_name = {name.lower(): name for name in class_names}
    if gt_lower in lower_to_name:
        return lower_to_name[gt_lower]

    raise ValueError(f"Could not map ground-truth value '{gt}' to {class_names}.")


def build_panderm_model(num_classes: int, variant: str = "base") -> torch.nn.Module:
    if variant == "base":
        builder = panderm_base_patch16_224_finetune
    elif variant == "large":
        builder = panderm_large_patch16_224_finetune
    else:
        raise ValueError(f"Unknown PanDerm variant: {variant}")

    model = builder(
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.2,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        # init_scale=0.001,
        init_scale=1.0,
        # use_rel_pos_bias=True,
        use_rel_pos_bias=False,
        init_values=0.1,
        lin_probe=False,
    )
    return model


# ---- Begin: Multitask/SegGate wrappers and helpers ----


class PatchSegHead(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 2, 128)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_tokens: torch.Tensor, patch_hw: tuple[int, int]) -> torch.Tensor:
        b, n, _ = patch_tokens.shape
        h, w = patch_hw
        if n != h * w:
            raise ValueError(f"patch token count {n} does not match patch grid {h}x{w}")
        return self.head(patch_tokens).transpose(1, 2).reshape(b, 1, h, w)


class PanDermMultitaskSegCAMWrapper(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        use_seg_gate: bool = False,
        seg_gate_bg_keep: float = 0.15,
        seg_gate_detach: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.seg_head = PatchSegHead(embed_dim=backbone.embed_dim)
        self.use_seg_gate = use_seg_gate
        self.seg_gate_bg_keep = seg_gate_bg_keep
        self.seg_gate_detach = seg_gate_detach

    def classify_from_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "fc_norm") and self.backbone.fc_norm is not None:
            pooled = patch_tokens.mean(dim=1)
            pooled = self.backbone.fc_norm(pooled)
        elif hasattr(self.backbone, "norm") and self.backbone.norm is not None:
            pooled = patch_tokens.mean(dim=1)
            pooled = self.backbone.norm(pooled)
        else:
            pooled = patch_tokens.mean(dim=1)
        return self.backbone.head(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x, return_patch_tokens=True)
        if isinstance(outputs, dict):
            raw_logits = outputs.get("logits")
            patch_tokens = outputs.get("patch_tokens")
        elif isinstance(outputs, (tuple, list)):
            raw_logits = outputs[0]
            patch_tokens = outputs[1] if len(outputs) > 1 else None
        else:
            raw_logits = outputs
            patch_tokens = None

        if patch_tokens is None:
            raise ValueError(
                "The PanDerm backbone did not return patch tokens. "
                "Cannot generate CAM for multitask/seggate checkpoint."
            )

        patch_hw = self.backbone.patch_embed.patch_shape
        seg_logits = self.seg_head(patch_tokens, patch_hw)

        if not self.use_seg_gate:
            return raw_logits

        seg_prob = torch.sigmoid(seg_logits)
        if self.seg_gate_detach:
            seg_prob = seg_prob.detach()

        b, _, h, w = seg_prob.shape
        gate = seg_prob.reshape(b, 1, h * w).transpose(1, 2)
        gate = self.seg_gate_bg_keep + (1.0 - self.seg_gate_bg_keep) * gate
        gated_patch_tokens = patch_tokens * gate
        return self.classify_from_patch_tokens(gated_patch_tokens)

    @torch.no_grad()
    def predict_seg_gate_map(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x, return_patch_tokens=True)
        if isinstance(outputs, dict):
            patch_tokens = outputs.get("patch_tokens")
        elif isinstance(outputs, (tuple, list)):
            patch_tokens = outputs[1] if len(outputs) > 1 else None
        else:
            patch_tokens = None

        if patch_tokens is None:
            raise ValueError("The PanDerm backbone did not return patch tokens. Cannot predict segmentation gate.")

        patch_hw = self.backbone.patch_embed.patch_shape
        seg_logits = self.seg_head(patch_tokens, patch_hw)
        seg_prob = torch.sigmoid(seg_logits)
        return seg_prob


# ---- End: Multitask/SegGate wrappers and helpers ----



def remap_official_finetune_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = dict(state_dict)

    encoder_keys = [k for k in list(remapped.keys()) if k.startswith("encoder.")]
    for key in encoder_keys:
        new_key = key.replace("encoder.", "", 1)
        remapped[new_key] = remapped[key]
        remapped.pop(key)

    for key in list(remapped.keys()):
        if key.startswith("decoder.") or key.startswith("teacher."):
            remapped.pop(key)

    for key in list(remapped.keys()):
        if key.startswith("norm."):
            new_key = key.replace("norm.", "fc_norm.", 1)
            remapped[new_key] = remapped[key]
            remapped.pop(key)

    # Checkpoints saved from SoftMaskWrapper or other wrappers prefix the
    # actual PanDerm keys with "backbone.". The CAM script builds the bare
    # PanDerm model, so strip this prefix before loading.
    backbone_keys = [key for key in list(remapped.keys()) if key.startswith("backbone.")]
    if len(backbone_keys) > 0:
        for key in backbone_keys:
            new_key = key.replace("backbone.", "", 1)
            remapped[new_key] = remapped[key]
        for key in backbone_keys:
            remapped.pop(key, None)

    # Multitask segmentation checkpoints contain an extra segmentation head.
    # The CAM script only needs the classification backbone/head, so remove it.
    for key in list(remapped.keys()):
        if key.startswith("seg_head."):
            remapped.pop(key, None)

    return remapped


# ---- Begin: multitask/seg_gate checkpoint helpers ----

def extract_checkpoint_state_dict(ckpt: dict) -> tuple[dict[str, torch.Tensor], str]:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], "custom_pt"
    if "model" in ckpt:
        return ckpt["model"], "official_pth"
    raise KeyError("Checkpoint must contain either 'model_state_dict' or 'model'.")


def has_multitask_seg_head(state_dict: dict[str, torch.Tensor]) -> bool:
    for key in state_dict.keys():
        clean_key = key[len("module."):] if key.startswith("module.") else key
        if clean_key.startswith("seg_head.") or clean_key.startswith("multitask_model.seg_head."):
            return True
    return False


def prepare_multitask_state_dict_for_cam(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prepared = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("multitask_model."):
            key = key.replace("multitask_model.", "", 1)

        # Already saved from PanDermMultitaskSegWrapper.
        if key.startswith("backbone.") or key.startswith("seg_head."):
            prepared[key] = value
            continue

        # Foundation / bare PanDerm style keys should belong to the backbone.
        if key.startswith("encoder."):
            key = key.replace("encoder.", "", 1)
        if key.startswith("decoder.") or key.startswith("teacher."):
            continue
        if key.startswith("norm."):
            key = key.replace("norm.", "fc_norm.", 1)

        prepared[f"backbone.{key}"] = value

    return prepared

# ---- End: multitask/seg_gate checkpoint helpers ----


# --- Begin: helper functions for variant inference ---

def infer_panderm_variant_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    head_weight = state_dict.get("head.weight")
    if head_weight is None:
        head_weight = state_dict.get("backbone.head.weight")
    if head_weight is not None:
        in_features = int(head_weight.shape[1])
        if in_features == 768:
            return "base"
        if in_features == 1024:
            return "large"

    patch_weight = state_dict.get("patch_embed.proj.weight")
    if patch_weight is None:
        patch_weight = state_dict.get("backbone.patch_embed.proj.weight")
    if patch_weight is not None:
        embed_dim = int(patch_weight.shape[0])
        if embed_dim == 768:
            return "base"
        if embed_dim == 1024:
            return "large"

    raise ValueError(
        "Could not infer PanDerm variant from checkpoint. Expected embed/head dim 768 (base) or 1024 (large)."
    )


def infer_variant_from_checkpoint_dict(ckpt: dict) -> tuple[dict[str, torch.Tensor], str, str]:
    raw_state_dict, checkpoint_format = extract_checkpoint_state_dict(ckpt)
    state_dict = remap_official_finetune_checkpoint_keys(raw_state_dict)
    variant = infer_panderm_variant_from_state_dict(state_dict)
    return state_dict, checkpoint_format, variant



def load_panderm_finetuned_model(
    checkpoint_path: str | Path,
    num_classes: int,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    device: str | torch.device | None = None,
    checkpoint_model_type: str = "auto",
    use_seg_gate: bool = False,
    seg_gate_bg_keep: float = 0.15,
    seg_gate_detach: bool = True,
) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = get_device(None)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict, checkpoint_format = extract_checkpoint_state_dict(ckpt)
    raw_has_seg_head = has_multitask_seg_head(raw_state_dict)

    if checkpoint_model_type == "auto":
        if use_seg_gate:
            checkpoint_model_type = "seggate"
        elif raw_has_seg_head:
            checkpoint_model_type = "multitask"
        else:
            checkpoint_model_type = "panderm"

    if checkpoint_model_type == "panderm":
        state_dict = remap_official_finetune_checkpoint_keys(raw_state_dict)
        variant = infer_panderm_variant_from_state_dict(state_dict)
        model = build_panderm_model(num_classes=num_classes, variant=variant)

        state_dict_model = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in state_dict and k in state_dict_model and state_dict[k].shape != state_dict_model[k].shape:
                raise ValueError(
                    f"Checkpoint head shape mismatch for {k}: checkpoint={tuple(state_dict[k].shape)} vs model={tuple(state_dict_model[k].shape)}. "
                    f"Check that --class_preset / --class_names matches the trained checkpoint."
                )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    elif checkpoint_model_type in ["multitask", "seggate"]:
        prepared_state_dict = prepare_multitask_state_dict_for_cam(raw_state_dict)
        variant_probe_state_dict = remap_official_finetune_checkpoint_keys(raw_state_dict)
        variant = infer_panderm_variant_from_state_dict(variant_probe_state_dict)
        backbone = build_panderm_model(num_classes=num_classes, variant=variant)
        model = PanDermMultitaskSegCAMWrapper(
            backbone=backbone,
            use_seg_gate=(checkpoint_model_type == "seggate" or use_seg_gate),
            seg_gate_bg_keep=seg_gate_bg_keep,
            seg_gate_detach=seg_gate_detach,
        )

        state_dict_model = model.state_dict()
        for k in ["backbone.head.weight", "backbone.head.bias"]:
            if k in prepared_state_dict and k in state_dict_model and prepared_state_dict[k].shape != state_dict_model[k].shape:
                raise ValueError(
                    f"Checkpoint head shape mismatch for {k}: checkpoint={tuple(prepared_state_dict[k].shape)} vs model={tuple(state_dict_model[k].shape)}. "
                    f"Check that --class_preset / --class_names matches the trained checkpoint."
                )

        missing, unexpected = model.load_state_dict(prepared_state_dict, strict=False)

    else:
        raise ValueError(f"Unsupported checkpoint_model_type: {checkpoint_model_type}")

    if len(missing) or len(unexpected):
        print(f"[warn] load_state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing sample:", missing[:10])
        if unexpected:
            print("  unexpected sample:", unexpected[:10])

    model = model.to(device)
    model.eval()

    info = {
        "arch": f"PanDerm {variant.capitalize()} FT",
        "variant": variant,
        "num_classes": num_classes,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_format": checkpoint_format,
        "checkpoint_model_type": checkpoint_model_type,
        "use_seg_gate": bool(checkpoint_model_type == "seggate" or use_seg_gate),
        "seg_gate_bg_keep": float(seg_gate_bg_keep),
        "seg_gate_detach": bool(seg_gate_detach),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": ckpt.get("image_size", 224),
        "stage_name": ckpt.get("stage_name", None),
        "epoch": ckpt.get("epoch", None),
    }
    return model, info


class PanDermCAMWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            if "logits" in out:
                out = out["logits"]
            else:
                first_key = next(iter(out))
                out = out[first_key]
        return out


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Unexpected ViT activation shape: {tuple(tensor.shape)}")

    batch, n_tokens, channels = tensor.shape
    n_patches = n_tokens - 1
    side = int(np.sqrt(n_patches))
    if side * side != n_patches:
        raise ValueError(f"Cannot infer square token grid from n_tokens={n_tokens}")

    tensor = tensor[:, 1:, :].reshape(batch, side, side, channels)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def get_image_id(row: pd.Series, image_col: str | None = None) -> str:
    if image_col is not None:
        if image_col not in row:
            raise ValueError(f"Requested --image_col '{image_col}' not found in CSV row.")
        return str(row[image_col])

    for col in ["image", "isic_id", "image_rel_path"]:
        if col in row:
            return str(row[col])

    raise ValueError("CSV must contain column 'image', 'isic_id', or 'image_rel_path'. Or pass --image_col.")


# --- Begin: helper for safe output stem ---
def make_safe_output_stem(image_id: str) -> str:
    image_path = Path(str(image_id))
    stem = image_path.stem if image_path.suffix else image_path.name
    stem = stem.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return stem

def save_raw_cam_arrays(
    out_dir: Path,
    output_stem: str,
    res: dict,
    gate_weighted_maps: dict[str, np.ndarray | None] | None = None,
) -> Path:
    """Save raw CAM arrays for quantitative localization metrics."""
    raw_dir = out_dir / "raw_cams" / output_stem
    raw_dir.mkdir(parents=True, exist_ok=True)

    flat_raw_dir = out_dir / "raw_cams"
    flat_raw_dir.mkdir(parents=True, exist_ok=True)

    cam_key_to_name = {
        "cam_gradcam": "gradcam_a",
        "cam_gradcam_B": "gradcam_b",
        "cam_gradcam_diff": "map_diff",
        "cam_finercam": "finercam",
        "cam_rollout": "rollout",
        "cam_chefer": "chefer_a",
        "cam_chefer_B": "chefer_b",
        "cam_chefer_diff": "chefer_diff",
        "cam_relprop_chefer": "relprop_chefer_a",
        "cam_relprop_chefer_B": "relprop_chefer_b",
        "cam_relprop_chefer_diff": "relprop_chefer_diff",
    }

    for cam_key, file_name in cam_key_to_name.items():
        cam_value = res.get(cam_key)
        if cam_value is None:
            continue
        cam_array = np.asarray(cam_value, dtype=np.float32)
        np.save(raw_dir / f"{file_name}.npy", cam_array)
        np.save(flat_raw_dir / f"{output_stem}_{file_name}.npy", cam_array)

    if gate_weighted_maps is not None:
        for file_name, cam_value in gate_weighted_maps.items():
            if cam_value is None:
                continue
            cam_array = np.asarray(cam_value, dtype=np.float32)
            np.save(raw_dir / f"{file_name}.npy", cam_array)
            np.save(flat_raw_dir / f"{output_stem}_{file_name}.npy", cam_array)

    return raw_dir
# --- End: helper for safe output stem ---


# --- Begin: mask cropping helpers ---
def _mask_bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_bin = mask_np > 0

    if not mask_bin.any():
        w, h = mask.size
        return 0, 0, w, h

    ys, xs = np.where(mask_bin)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _expand_and_square_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    margin: float,
    min_crop_frac: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    w_img, h_img = image_size

    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    side = max(box_w, box_h)
    side = side * (1.0 + 2.0 * margin)
    side = max(side, min_crop_frac * min(w_img, h_img))
    side = min(side, max(w_img, h_img))

    new_x0 = int(round(cx - side / 2.0))
    new_y0 = int(round(cy - side / 2.0))
    new_x1 = int(round(cx + side / 2.0))
    new_y1 = int(round(cy + side / 2.0))

    if new_x0 < 0:
        new_x1 -= new_x0
        new_x0 = 0
    if new_y0 < 0:
        new_y1 -= new_y0
        new_y0 = 0
    if new_x1 > w_img:
        shift = new_x1 - w_img
        new_x0 -= shift
        new_x1 = w_img
    if new_y1 > h_img:
        shift = new_y1 - h_img
        new_y0 -= shift
        new_y1 = h_img

    new_x0 = max(0, new_x0)
    new_y0 = max(0, new_y0)
    new_x1 = min(w_img, new_x1)
    new_y1 = min(h_img, new_y1)

    if new_x1 <= new_x0 or new_y1 <= new_y0:
        return 0, 0, w_img, h_img

    return new_x0, new_y0, new_x1, new_y1


# --- End: mask cropping helpers ---


def crop_image_with_mask(
    img: Image.Image,
    mask_path: Path,
    margin: float,
    min_crop_frac: float,
) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    bbox = _mask_bbox(mask)
    crop_box = _expand_and_square_bbox(
        bbox=bbox,
        image_size=img.size,
        margin=margin,
        min_crop_frac=min_crop_frac,
    )
    return img.crop(crop_box)

def load_mask_for_row(
    row: pd.Series,
    mask_root: Path | None,
    mask_col: str,
) -> Image.Image | None:
    if mask_col not in row:
        return None

    mask_value = row[mask_col]
    if pd.isna(mask_value):
        return None

    mask_path = Path(str(mask_value))

    if not mask_path.is_absolute():
        if mask_root is not None:
            mask_path = mask_root / mask_path
        else:
            mask_path = REPO_ROOT / mask_path

    if not mask_path.exists():
        return None

    return Image.open(mask_path).convert("L")


def make_gt_mask_overlay(
    mask: Image.Image,
    rgb_float: np.ndarray,
) -> np.ndarray:
    rgb_h, rgb_w = rgb_float.shape[:2]

    mask_resized = mask.resize((rgb_w, rgb_h), resample=Image.Resampling.NEAREST)
    mask_np = np.array(mask_resized).astype(np.float32)

    if mask_np.max() > 0:
        mask_np = mask_np / mask_np.max()

    mask_bin = (mask_np > 0.5).astype(np.uint8)

    overlay_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay_uint8,
        contours,
        contourIdx=-1,
        color=(255, 0, 0),
        thickness=2,
    )

    return overlay_uint8.astype(np.float32) / 255.0

def make_gt_mask_binary_rgb(
    mask: Image.Image,
    rgb_float: np.ndarray,
) -> np.ndarray:
    rgb_h, rgb_w = rgb_float.shape[:2]
    mask_resized = mask.resize((rgb_w, rgb_h), resample=Image.Resampling.NEAREST)
    mask_np = np.array(mask_resized).astype(np.float32)
    if mask_np.max() > 0:
        mask_np = mask_np / mask_np.max()
    mask_np = np.clip(mask_np, 0.0, 1.0)
    return np.stack([mask_np, mask_np, mask_np], axis=-1)

# ---- Predicted segmentation gate visualization helper ----
def predict_seg_gate_overlay(
    model_raw: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not hasattr(model_raw, "predict_seg_gate_map"):
        return None, None

    seg_prob = model_raw.predict_seg_gate_map(input_tensor)
    seg_map_small = seg_prob[0, 0].detach().cpu().numpy()
    rgb_h, rgb_w = rgb_float.shape[:2]
    seg_map = cv2.resize(seg_map_small, (rgb_w, rgb_h), interpolation=cv2.INTER_CUBIC)
    seg_map = np.clip(seg_map, 0.0, 1.0)

    heat = cv2.applyColorMap((seg_map * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(0.55 * rgb_float + 0.45 * heat, 0.0, 1.0)
    return seg_map, overlay

def make_gate_weighted_cam_overlay(
    cam_map: np.ndarray,
    gate_map: np.ndarray | None,
    rgb_float: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if gate_map is None:
        return None, None

    cam = np.asarray(cam_map, dtype=np.float32)
    gate = np.asarray(gate_map, dtype=np.float32)

    rgb_h, rgb_w = rgb_float.shape[:2]

    if cam.shape != (rgb_h, rgb_w):
        cam = cv2.resize(cam, (rgb_w, rgb_h), interpolation=cv2.INTER_CUBIC)

    if gate.shape != (rgb_h, rgb_w):
        gate = cv2.resize(gate, (rgb_w, rgb_h), interpolation=cv2.INTER_CUBIC)

    cam = np.clip(cam, 0.0, 1.0)
    gate = np.clip(gate, 0.0, 1.0)

    weighted = cam * gate
    weighted = weighted - weighted.min()
    weighted = weighted / (weighted.max() + 1e-8)

    heat = cv2.applyColorMap((weighted * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(0.55 * rgb_float + 0.45 * heat, 0.0, 1.0)

    return weighted, overlay

def parse_panel_items(panel_items_arg: str) -> list[str]:
    allowed = {
        "rgb",
        "rgb_gt_mask",
        "gt_mask",
        "seg_gate",
        "gradcam_a",
        "gradcam_b",
        "map_diff",
        "finercam",
        "gate_weighted_gradcam_a",
        "gate_weighted_gradcam_b",
        "gate_weighted_map_diff",
        "gate_weighted_finercam",
    }

    items = [item.strip() for item in str(panel_items_arg).split(",") if item.strip()]
    if not items:
        raise ValueError("--panel_items produced an empty panel item list.")

    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown panel items: {unknown}. Allowed: {sorted(allowed)}")

    return items
# ---- End: predicted segmentation gate visualization helpers ----


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    mask_root = Path(args.mask_root) if args.mask_root is not None else None
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_items = parse_panel_items(args.panel_items)

    device = get_device(args.device)

    class_names = resolve_class_names(args)
    class_to_idx, idx_to_class = build_class_maps(class_names)

    model_raw, info = load_panderm_finetuned_model(
        ckpt_path,
        num_classes=len(class_names),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        device=device,
        checkpoint_model_type=args.checkpoint_model_type,
        use_seg_gate=args.use_seg_gate,
        seg_gate_bg_keep=args.seg_gate_bg_keep,
        seg_gate_detach=args.seg_gate_detach,
    )
    print(f"[info] Loaded {info['arch']} from {ckpt_path.name}")
    print(
        "[info] checkpoint_model_type=",
        info.get("checkpoint_model_type"),
        "use_seg_gate=",
        info.get("use_seg_gate"),
        "seg_gate_bg_keep=",
        info.get("seg_gate_bg_keep"),
        "seg_gate_detach=",
        info.get("seg_gate_detach"),
    )
    if hasattr(model_raw, "backbone"):
        relprop_source_model = model_raw.backbone
    else:
        relprop_source_model = model_raw
    relprop_model = build_panderm_relprop_from_model(relprop_source_model).to(device)
    relprop_model.eval()
    model = PanDermCAMWrapper(model_raw)

    image_size = int(info.get("image_size", args.image_size) or args.image_size)
    if args.image_size is not None:
        image_size = args.image_size

    preprocess = get_eval_transforms(which_img_norm="imagenet", img_resize=256, center_crop=True)
    if preprocess is None:
        preprocess = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    if hasattr(model_raw, "backbone"):
        blocks = model_raw.backbone.blocks
    else:
        blocks = model_raw.blocks

    target_block_index = int(args.target_block_index)

    if target_block_index < 0:
        target_block_index = len(blocks) + target_block_index

    if target_block_index < 0 or target_block_index >= len(blocks):
        raise ValueError(
            f"--target_block_index={args.target_block_index} is invalid. "
            f"Model has {len(blocks)} blocks, valid resolved indices are 0 to {len(blocks) - 1}."
        )

    target_layer = blocks[target_block_index].norm1
    print(
        f"[info] CAM target layer: blocks[{target_block_index}].norm1 "
        f"(requested {args.target_block_index})"
    )

    df = pd.read_csv(csv_path)
    if args.crop_with_mask:
        if mask_root is None:
            raise ValueError("--crop_with_mask requires --mask_root")
        if args.mask_col not in df.columns:
            raise ValueError(
                f"--crop_with_mask requires mask column '{args.mask_col}' in the CSV. "
                f"Found columns: {df.columns.tolist()}"
            )

    if any(item in panel_items for item in ["gt_mask", "rgb_gt_mask"]) and args.mask_col not in df.columns:
        raise ValueError(
            f"Panel item 'gt_mask' or 'rgb_gt_mask' requires mask column '{args.mask_col}' in the CSV. "
            f"Found columns: {df.columns.tolist()}"
        )

    if any(item in panel_items for item in ["gt_mask", "rgb_gt_mask"]) and mask_root is None:
        print(
            "[warn] gt_mask/rgb_gt_mask requested but --mask_root was not provided. "
            "Mask paths will be resolved relative to --img_dir."
        )
    if args.image_col is not None:
        if args.image_col not in df.columns:
            raise ValueError(f"Requested --image_col '{args.image_col}' not found. Found: {df.columns.tolist()}")
    elif "image" not in df.columns and "isic_id" not in df.columns and "image_rel_path" not in df.columns:
        raise ValueError(
            f"CSV must contain column 'image', 'isic_id', or 'image_rel_path'. "
            f"Found: {df.columns.tolist()}"
        )

    df = df.head(args.num_samples)

    for _, row in df.iterrows():
        image_id = get_image_id(row, image_col=args.image_col)
        image_id_path = Path(image_id)
        output_stem = make_safe_output_stem(image_id)
        if image_id_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img_path = img_dir / image_id_path
        else:
            img_path = img_dir / f"{image_id}.jpg"
        if not img_path.exists():
            print(f"[skip] missing image: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        mask_img = None

        if any(item in panel_items for item in ["gt_mask", "rgb_gt_mask"]):
            mask_img = load_mask_for_row(
                row=row,
                mask_root=mask_root if mask_root is not None else img_dir,
                mask_col=args.mask_col,
            )

            if mask_img is None:
                print(f"[skip] missing ground-truth mask for image: {image_id}")
                continue

        if args.crop_with_mask:
            mask_value = row[args.mask_col]
            if pd.isna(mask_value):
                print(f"[skip] missing mask value for image: {image_id}")
                continue
            mask_path = (mask_root / str(mask_value)).resolve()
            if not mask_path.exists():
                print(f"[skip] missing mask: {mask_path}")
                continue
            img = crop_image_with_mask(
                img=img,
                mask_path=mask_path,
                margin=args.crop_margin,
                min_crop_frac=args.min_crop_frac,
            )
            if mask_img is not None:
                crop_box = _expand_and_square_bbox(
                    bbox=_mask_bbox(mask_img),
                    image_size=mask_img.size,
                    margin=args.crop_margin,
                    min_crop_frac=args.min_crop_frac,
                )
                mask_img = mask_img.crop(crop_box)

        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        sorted_idx = np.argsort(probs)[::-1]

        rgb = np.array(img).astype(np.float32) / 255.0
        rgb_resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        gt_mask_overlay = None
        gt_mask_binary = None

        if any(item in panel_items for item in ["gt_mask", "rgb_gt_mask"]):
            if mask_img is None:
                raise ValueError("Internal error: gt_mask/rgb_gt_mask requested but mask_img is None.")
            gt_mask_overlay = make_gt_mask_overlay(mask_img, rgb_resized)
            gt_mask_binary = make_gt_mask_binary_rgb(mask_img, rgb_resized)

        seg_gate_map = None
        seg_gate_overlay = None

        if any(
            item in panel_items
            for item in [
                "seg_gate",
                "gate_weighted_gradcam_a",
                "gate_weighted_gradcam_b",
                "gate_weighted_map_diff",
                "gate_weighted_finercam",
            ]
        ):
            seg_gate_map, seg_gate_overlay = predict_seg_gate_overlay(
                model_raw=model_raw,
                input_tensor=x,
                rgb_float=rgb_resized,
            )

            if seg_gate_map is None:
                raise ValueError(
                    "Requested seg_gate or gate_weighted_* in --panel_items, "
                    "but the loaded model does not expose predict_seg_gate_map(). "
                    "Use these panel items only with multitask/seggate checkpoints."
                )

        A_idx, B_idx = None, None
        comparison_categories = None

        if args.compare_mode == "fixed":
            if args.A is None or args.B is None:
                raise ValueError("compare_mode=fixed requires --A and --B")
            if args.A not in class_to_idx or args.B not in class_to_idx:
                raise ValueError(f"Unknown class name. Allowed: {class_names}")
            A_idx = class_to_idx[args.A]
            B_idx = class_to_idx[args.B]
            comparison_categories = [B_idx]

        elif args.compare_mode == "gt_pair":
            if args.A is None or args.B is None:
                raise ValueError("compare_mode=gt_pair requires --A and --B")
            if args.A not in class_to_idx or args.B not in class_to_idx:
                raise ValueError(f"Unknown class name. Allowed: {class_names}")
            gt = resolve_gt_label_from_row(
                row=row,
                class_names=class_names,
                class_to_idx=class_to_idx,
                gt_col=args.gt_col,
            )
            if gt not in [args.A, args.B]:
                print(f"[skip] {image_id}: gt_label={gt} not in pair ({args.A},{args.B})")
                continue
            if gt == args.A:
                A_idx = class_to_idx[args.A]
                B_idx = class_to_idx[args.B]
            else:
                A_idx = class_to_idx[args.B]
                B_idx = class_to_idx[args.A]
            comparison_categories = [B_idx]

        elif args.compare_mode == "pred_topk_non_target":
            A_idx = int(sorted_idx[0])
            comparison_categories = [int(i) for i in sorted_idx if int(i) != A_idx][: max(1, args.topk_compare)]
            B_idx = comparison_categories[0]

        elif args.compare_mode == "gt_topk_non_target":
            gt = resolve_gt_label_from_row(
                row=row,
                class_names=class_names,
                class_to_idx=class_to_idx,
                gt_col=args.gt_col,
            )
            if gt not in class_to_idx:
                print(f"[skip] {image_id}: gt_label={gt} not in class list")
                continue
            A_idx = class_to_idx[gt]
            comparison_categories = [int(i) for i in sorted_idx if int(i) != A_idx][: max(1, args.topk_compare)]
            if len(comparison_categories) == 0:
                print(f"[skip] {image_id}: could not find non-target comparison categories")
                continue
            B_idx = comparison_categories[0]

        res = compute_cam_bundle(
            model=model,
            input_tensor=x,
            rgb_float=rgb_resized,
            target_layer=target_layer,
            method=args.method,
            A=A_idx,
            B=B_idx,
            comparison_categories=comparison_categories,
            reshape_transform=vit_reshape_transform,
            alpha=args.alpha,
            relprop_model=relprop_model if args.show_extra_rows else None,
            include_extra_maps=args.show_extra_rows,
        )

        gate_weighted_gradcam_a_map, gate_weighted_gradcam_a_overlay = make_gate_weighted_cam_overlay(
            cam_map=res["cam_gradcam"],
            gate_map=seg_gate_map,
            rgb_float=rgb_resized,
        )
        gate_weighted_gradcam_b_map, gate_weighted_gradcam_b_overlay = make_gate_weighted_cam_overlay(
            cam_map=res["cam_gradcam_B"],
            gate_map=seg_gate_map,
            rgb_float=rgb_resized,
        )
        gate_weighted_map_diff_map, gate_weighted_map_diff_overlay = make_gate_weighted_cam_overlay(
            cam_map=res["cam_gradcam_diff"],
            gate_map=seg_gate_map,
            rgb_float=rgb_resized,
        )
        gate_weighted_finercam_map, gate_weighted_finercam_overlay = make_gate_weighted_cam_overlay(
            cam_map=res["cam_finercam"],
            gate_map=seg_gate_map,
            rgb_float=rgb_resized,
        )

        if args.save_raw_cams:
            raw_cam_dir = save_raw_cam_arrays(
                out_dir=out_dir,
                output_stem=output_stem,
                res=res,
                gate_weighted_maps={
                    "gate_weighted_gradcam_a": gate_weighted_gradcam_a_map,
                    "gate_weighted_gradcam_b": gate_weighted_gradcam_b_map,
                    "gate_weighted_map_diff": gate_weighted_map_diff_map,
                    "gate_weighted_finercam": gate_weighted_finercam_map,
                },
            )
            print(f"[saved raw cams] {raw_cam_dir}")

        topk_for_display = min(3, len(res["probs"]))
        top3_idx = np.argsort(res["probs"])[-topk_for_display:][::-1]
        top3_named = ", ".join([f"{idx_to_class[i]}: {res['probs'][i]:.3f}" for i in top3_idx])

        A_name = idx_to_class.get(int(res["A"]), str(res["A"]))
        B_name = idx_to_class.get(int(res["B"]), str(res["B"]))
        comp_named = ", ".join([idx_to_class[int(i)] for i in res.get("comparison_categories", [res["B"]])])
        print(f"[info] {image_id}: A={res['A']}({A_name})  B={res['B']}({B_name})  comparison=[{comp_named}]  top3=[{top3_named}]")

        if args.gt_col is not None and args.gt_col in df.columns:
            gt_label = resolve_gt_label_from_row(
                row=row,
                class_names=class_names,
                class_to_idx=class_to_idx,
                gt_col=args.gt_col,
            )
        elif "gt_label" in df.columns:
            gt_label = resolve_gt_label_from_row(
                row=row,
                class_names=class_names,
                class_to_idx=class_to_idx,
                gt_col="gt_label",
            )
        else:
            gt_label = None
        gradcam_a_prob = float(res["probs"][int(res["A"])])
        gradcam_b_prob = float(res["probs"][int(res["B"])])
        finercam_prob = float(res["probs"][int(res["A"])])
        rollout_prob = float(res["probs"][int(res["A"])])

        first_tile_line1 = str(image_id)
        first_tile_line2 = f"GT={gt_label}" if gt_label is not None else "RGB"

        panel_img_uint8 = make_panel_with_subtitles(
            first_tile_line1=first_tile_line1,
            first_tile_line2=first_tile_line2,
            rgb_float=rgb_resized,
            gt_mask_overlay=gt_mask_overlay,
            gt_mask_binary=gt_mask_binary,
            seg_gate_overlay=seg_gate_overlay,
            gradcam_overlay_a=res["overlay_gradcam"],
            gradcam_overlay_b=res["overlay_gradcam_B"],
            gradcam_diff_overlay=res["overlay_gradcam_diff"],
            finercam_overlay=res["overlay_finercam"],
            gate_weighted_finercam_overlay=gate_weighted_finercam_overlay,
            gate_weighted_gradcam_a_overlay=gate_weighted_gradcam_a_overlay,
            gate_weighted_gradcam_b_overlay=gate_weighted_gradcam_b_overlay,
            gate_weighted_map_diff_overlay=gate_weighted_map_diff_overlay,
            rollout_overlay=res["overlay_rollout"],
            chefer_overlay_a=res["overlay_chefer"],
            chefer_overlay_b=res["overlay_chefer_B"],
            chefer_diff_overlay=res["overlay_chefer_diff"],
            relprop_chefer_overlay_a=res["overlay_relprop_chefer"],
            relprop_chefer_overlay_b=res["overlay_relprop_chefer_B"],
            relprop_chefer_diff_overlay=res["overlay_relprop_chefer_diff"],
            gradcam_a_line1="GradCAM",
            gradcam_a_line2=f"{A_name} ({gradcam_a_prob:.2f})",
            gradcam_b_line1="GradCAM",
            gradcam_b_line2=f"{B_name} ({gradcam_b_prob:.2f})",
            gradcam_diff_line1="Map Diff",
            gradcam_diff_line2=f"max(0, {A_name} - {B_name})",
            finercam_line1="FinerCAM",
            finercam_line2=f"{A_name} vs {B_name} ({finercam_prob:.2f})",
            gate_weighted_finercam_line1="Gate weighted FinerCAM",
            gate_weighted_finercam_line2=f"{A_name} FinerCAM × gate",
            gate_weighted_gradcam_a_line1="GradCAM × gate",
            gate_weighted_gradcam_a_line2=f"{A_name} ({gradcam_a_prob:.2f})",
            gate_weighted_gradcam_b_line1="GradCAM × gate",
            gate_weighted_gradcam_b_line2=f"{B_name} ({gradcam_b_prob:.2f})",
            gate_weighted_map_diff_line1="Map Diff × gate",
            gate_weighted_map_diff_line2=f"max(0, {A_name} - {B_name}) × gate",
            rollout_line1="Rollout",
            rollout_line2=f"{A_name} ({rollout_prob:.2f})",
            chefer_a_line1="Chefer-style",
            chefer_a_line2=f"{A_name}",
            chefer_b_line1="Chefer-style",
            chefer_b_line2=f"{B_name}",
            chefer_diff_line1="Chefer Map Diff",
            chefer_diff_line2=f"max(0, {A_name} - {B_name})",
            relprop_chefer_a_line1="Chefer relprop",
            relprop_chefer_a_line2=f"{A_name}",
            relprop_chefer_b_line1="Chefer relprop",
            relprop_chefer_b_line2=f"{B_name}",
            relprop_chefer_diff_line1="Relprop Map Diff",
            relprop_chefer_diff_line2=f"max(0, {A_name} - {B_name})",
            seg_gate_line1="Predicted Seg Gate",
            seg_gate_line2="auxiliary head",
            scale=args.panel_scale,
            show_extra_row=args.show_extra_rows,
            show_relprop_row=args.show_relprop_row,
            panel_items=panel_items,
        )

        panel_suffix = "_".join(panel_items)
        panel_path = out_dir / f"{output_stem}_{panel_suffix}.png"
        Image.fromarray(panel_img_uint8).save(panel_path)

        if args.save_json:
            meta = {
                "image_id": str(image_id),
                "img_path": str(img_path),
                "checkpoint": ckpt_path.name,
                "model_type": "panderm_ft",
                "image_size": image_size,
                "device": device,
                "A_idx": int(res["A"]),
                "B_idx": int(res["B"]),
                "comparison_categories": [int(i) for i in res.get("comparison_categories", [res["B"]])],
                "comparison_category_names": [idx_to_class[int(i)] for i in res.get("comparison_categories", [res["B"]])],
                "topk_compare": int(args.topk_compare),
                "alpha": float(args.alpha),
                "probs_topk": res["probs_top3"],
                "num_classes": len(class_names),
                "raw_cam_dir": str((out_dir / "raw_cams" / output_stem).relative_to(out_dir)) if args.save_raw_cams else None,
                "method": args.method,
                "panel_path": str(panel_path),
                "compare_mode": args.compare_mode,
                "A_name": A_name,
                "B_name": B_name,
                "target_layer": f"blocks[{target_block_index}].norm1",
                "target_block_index": int(target_block_index),
                "target_block_index_requested": int(args.target_block_index),
                "class_names": class_names,
                "class_preset": args.class_preset,
                "checkpoint_format": info.get("checkpoint_format"),
                "checkpoint_model_type": info.get("checkpoint_model_type"),
                "use_seg_gate": info.get("use_seg_gate"),
                "seg_gate_bg_keep": info.get("seg_gate_bg_keep"),
                "seg_gate_detach": info.get("seg_gate_detach"),
                "gradcam_a_prob": gradcam_a_prob,
                "gradcam_b_prob": gradcam_b_prob,
                "gradcam_diff_desc": f"max(0, {A_name} - {B_name})",
                "finercam_prob": finercam_prob,
                "rollout_prob": rollout_prob,
                "chefer_desc": "Approximate Chefer-style transformer attribution using positive grad*attention rollout on PanDerm",
                "chefer_b_desc": f"Approximate Chefer-style transformer attribution for {B_name}",
                "chefer_diff_desc": f"max(0, {A_name} - {B_name}) on Chefer-style maps",
                "relprop_chefer_desc": "Chefer-style transformer attribution computed through the relprop-enabled PanDerm wrapper",
                "relprop_chefer_b_desc": f"Chefer-style relprop attribution for {B_name}",
                "relprop_chefer_diff_desc": f"max(0, {A_name} - {B_name}) on relprop-based Chefer maps",
                "show_relprop_row": bool(args.show_relprop_row),
                "crop_with_mask": bool(args.crop_with_mask),
                "mask_root": str(mask_root) if mask_root is not None else None,
                "mask_col": args.mask_col,
                "crop_margin": float(args.crop_margin),
                "min_crop_frac": float(args.min_crop_frac),
                "seg_gate_visualized": seg_gate_overlay is not None,
                "gate_weighted_finercam_visualized": gate_weighted_finercam_overlay is not None,
            }
            (out_dir / f"{output_stem}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()