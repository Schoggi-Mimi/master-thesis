"""
Evaluate CAM metrics for fine-tuned PanDerm models.

Outputs:
- per_sample_metrics.csv
- per_method_summary.csv
- metrics_full.json

Typical usage:
python -m scripts.eval_cam_metrics_panderm \
  --csv data/HAM10000/ham_test_for_cam.csv \
  --image_col image_rel_path \
  --img_dir data/HAM10000 \
  --checkpoint external/weights/checkpoint-best-ham.pth \
  --checkpoint_model_type panderm \
  --class_preset ham \
  --out_dir outputs/metrics_baseline \
  --num_samples 200 \
  --compare_mode gt_topk_non_target \
  --topk_compare 3 \
  --mask_root data/HAM10000 \
  --mask_col mask_rel_path \
  --methods gradcam_target,finercam

python -m scripts.eval_cam_metrics_panderm \
  --csv data/HAM10000/ham_test_for_cam.csv \
  --image_col image_rel_path \
  --img_dir data/HAM10000 \
  --checkpoint external/weights/checkpoint-best-seggate.pth \
  --checkpoint_model_type seggate \
  --use_seg_gate \
  --seg_gate_bg_keep 0.15 \
  --class_preset ham \
  --out_dir outputs/metrics_seggate_bg015 \
  --num_samples 200 \
  --compare_mode gt_topk_non_target \
  --topk_compare 3 \
  --mask_root data/HAM10000 \
  --mask_col mask_rel_path \
  --methods gradcam_target,finercam,pred_seg_gate,gate_weighted_finercam
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

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

from scripts.generate_finer_cam_panderm import (PanDermCAMWrapper,
                                                build_class_maps,
                                                load_panderm_finetuned_model,
                                                make_gate_weighted_cam_overlay,
                                                resolve_class_names,
                                                vit_reshape_transform)
from src.cam.diff_cam import compute_cam_bundle
from src.eval.cam_metrics import (DeletionMetric, InsertionMetric,
                                  RelativeConfidenceDropMetric,
                                  summarize_metric_dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--checkpoint_model_type",
        type=str,
        default="auto",
        choices=["auto", "panderm", "multitask", "seggate"],
        help="Checkpoint loading mode. Use seggate for segmentation-gated multitask checkpoints.",
    )
    p.add_argument(
        "--use_seg_gate",
        action="store_true",
        default=False,
        help="Use the segmentation-gated classification path for multitask/seggate checkpoints.",
    )
    p.add_argument(
        "--seg_gate_bg_keep",
        type=float,
        default=0.15,
        help="Background keep value used by the segmentation gate. Must match training.",
    )
    p.add_argument(
        "--seg_gate_detach",
        action="store_true",
        default=True,
        help="Detach segmentation probability before gating. Should match training.",
    )
    p.add_argument(
        "--seg_gate_no_detach",
        action="store_false",
        dest="seg_gate_detach",
        help="Do not detach segmentation probability before gating.",
    )
    p.add_argument("--class_preset", type=str, required=True, choices=["ham", "bcn"])
    p.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="Optional comma-separated class names overriding --class_preset.",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--device", type=str, default=None)

    p.add_argument(
        "--compare_mode",
        type=str,
        default="pred_topk_non_target",
        choices=["fixed", "top2", "pred_topk_non_target", "gt_topk_non_target"],
    )
    p.add_argument("--topk_compare", type=int, default=2)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Alpha weight for FinerCAM comparison categories. Default: 0.6.",
    )
    p.add_argument("--A", type=str, default=None, help="Fixed target class name for compare_mode=fixed.")
    p.add_argument("--B", type=str, default=None, help="Fixed reference class name for compare_mode=fixed.")
    p.add_argument("--cam_target_layer", type=str, default="last_block")
    p.add_argument(
        "--methods",
        type=str,
        default="gradcam_target,finercam",
        help=(
            "Comma-separated CAM maps to evaluate. "
            "Default: gradcam_target,finercam. "
            "Options: gradcam_target,gradcam_reference,gradcam_diff,finercam,"
            "pred_seg_gate,gate_weighted_finercam,rollout,chefer_style."
        ),
    )
    p.add_argument(
        "--include_extra_maps",
        action="store_true",
        default=False,
        help="Compute rollout/Chefer maps. Off by default because these can fail for wrapped SegGate models.",
    )
    p.add_argument("--rollout_start_layer", type=int, default=0)

    p.add_argument("--deletion_steps", type=int, default=100)
    p.add_argument("--insertion_steps", type=int, default=100)
    p.add_argument(
        "--perturbation_steps",
        type=float,
        nargs="*",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    p.add_argument("--mask_value", type=float, default=0.0)

    p.add_argument("--image_col", type=str, default="image")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--id_col", type=str, default="image_id")
    
    p.add_argument("--mask_root", type=str, default=None)
    p.add_argument("--mask_col", type=str, default="mask_rel_path")
    p.add_argument("--cam_threshold", type=float, default=0.5)

    return p.parse_args()

def load_binary_mask(mask_path: Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    with Image.open(mask_path) as mask_img:
        mask = mask_img.convert("L")
    mask_np = np.array(mask).astype(np.float32)
    if mask_np.max() > 1.0:
        mask_np = mask_np / 255.0
    mask_np = cv2.resize(mask_np, size, interpolation=cv2.INTER_NEAREST)
    return (mask_np > 0.5).astype(np.float32)


def compute_mask_overlap_metrics(cam_np: np.ndarray, mask_np: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    cam = cam_np.astype(np.float32)
    mask = mask_np.astype(np.float32)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_bin = (cam >= threshold).astype(np.float32)

    intersection = float((cam_bin * mask).sum())
    cam_area = float(cam_bin.sum())
    mask_area = float(mask.sum())
    union = float(((cam_bin + mask) > 0).sum())

    dice = (2.0 * intersection + 1e-8) / (cam_area + mask_area + 1e-8)
    iou = (intersection + 1e-8) / (union + 1e-8)
    energy_inside = float((cam * mask).sum() / (cam.sum() + 1e-8))

    max_y, max_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    pointing_game = float(mask[max_y, max_x] > 0.5)

    return {
        "mask_dice": dice,
        "mask_iou": iou,
        "mask_energy_inside": energy_inside,
        "mask_pointing_game": pointing_game,
        "mask_area_fraction": mask_area / float(mask.size),
    }

def get_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_eval_transform(image_size: int = 224) -> T.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.228, 0.224, 0.225)
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def to_cam_tensor(cam_np: np.ndarray, device: str) -> torch.Tensor:
    if cam_np.ndim != 2:
        raise ValueError(f"Expected 2D CAM, got shape {cam_np.shape}")
    return torch.from_numpy(cam_np).float().unsqueeze(0).to(device)


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def parse_class_name(spec: str, class_to_idx: Dict[str, int]) -> int:
    spec_str = str(spec).strip()
    if spec_str.isdigit():
        idx = int(spec_str)
        if idx in class_to_idx.values():
            return idx
        raise ValueError(f"Class index out of range: {idx}")

    lowered = {name.lower(): idx for name, idx in class_to_idx.items()}
    key = spec_str.lower()
    if key in lowered:
        return lowered[key]

    raise ValueError(f"Unknown class name: {spec}. Allowed: {list(class_to_idx.keys())}")

def build_compare_tag(args: argparse.Namespace) -> str:
    if args.compare_mode == "fixed":
        a = (args.A or "A").replace(" ", "_")
        b = (args.B or "B").replace(" ", "_")
        return f"fixed_{a}_vs_{b}"
    if args.compare_mode == "top2":
        return "top2_model_compare"
    if args.compare_mode == "pred_topk_non_target":
        return f"pred_topk_k{int(args.topk_compare)}"
    if args.compare_mode == "gt_topk_non_target":
        return f"gt_topk_k{int(args.topk_compare)}"
    return args.compare_mode.replace(" ", "_")

def parse_methods(methods_arg: str) -> list[str]:
    allowed = {
        "gradcam_target",
        "gradcam_reference",
        "gradcam_diff",
        "finercam",
        "pred_seg_gate",
        "gate_weighted_finercam",
        "rollout",
        "chefer_style",
    }

    methods = [m.strip() for m in str(methods_arg).split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods produced an empty method list.")

    unknown = [m for m in methods if m not in allowed]
    if unknown:
        raise ValueError(f"Unknown methods in --methods: {unknown}. Allowed: {sorted(allowed)}")

    return methods


def get_cam_target_layer(model_raw: torch.nn.Module):
    if hasattr(model_raw, "backbone"):
        return model_raw.backbone.blocks[-1].norm1
    return model_raw.blocks[-1].norm1


@torch.no_grad()
def predict_seg_gate_map_for_metrics(
    model_raw: torch.nn.Module,
    input_tensor: torch.Tensor,
    image_size: tuple[int, int] = (224, 224),
) -> np.ndarray | None:
    if not hasattr(model_raw, "predict_seg_gate_map"):
        return None

    seg_prob = model_raw.predict_seg_gate_map(input_tensor)
    seg_map_small = seg_prob[0, 0].detach().cpu().numpy().astype(np.float32)

    seg_map = cv2.resize(seg_map_small, image_size, interpolation=cv2.INTER_CUBIC)
    seg_map = np.clip(seg_map, 0.0, 1.0)

    return seg_map

def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_tag = build_compare_tag(args)

    requested_methods = parse_methods(args.methods)

    needs_extra_maps = any(m in requested_methods for m in ["rollout", "chefer_style"])
    if needs_extra_maps and not args.include_extra_maps:
        raise ValueError(
            "Requested rollout or chefer_style in --methods, but --include_extra_maps is not set."
        )

    df = pd.read_csv(args.csv, low_memory=False)
    if args.num_samples > 0:
        df = df.head(args.num_samples).copy()

    class_names = resolve_class_names(args)
    class_to_idx, idx_to_class = build_class_maps(class_names)

    model_raw, info = load_panderm_finetuned_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(class_names),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        device=device,
        checkpoint_model_type=args.checkpoint_model_type,
        use_seg_gate=args.use_seg_gate,
        seg_gate_bg_keep=args.seg_gate_bg_keep,
        seg_gate_detach=args.seg_gate_detach,
    )
    model = PanDermCAMWrapper(model_raw)
    model.eval()
    target_layer = get_cam_target_layer(model_raw)

    transform = build_eval_transform()
    rel_conf_metric = RelativeConfidenceDropMetric(
        model=model,
        perturbation_steps=args.perturbation_steps,
    )

    del_metric = DeletionMetric(
        model=model,
        num_steps=args.deletion_steps,
    )
    ins_metric = InsertionMetric(
        model=model,
        num_steps=args.insertion_steps,
    )

    per_sample_rows: List[Dict[str, Any]] = []
    metrics_full: List[Dict[str, Any]] = []

    img_dir = Path(args.img_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="evaluating"):
        image_name = str(row[args.image_col])
        image_id = str(row[args.id_col]) if args.id_col in row else Path(image_name).stem
        gt_idx = int(row[args.label_col]) if args.label_col in row else None
        gt_name = class_names[gt_idx] if gt_idx is not None and 0 <= gt_idx < len(class_names) else None

        if image_name.endswith(".jpg") or image_name.endswith(".jpeg") or image_name.endswith(".png"):
            image_path = img_dir / image_name
        else:
            image_path = img_dir / f"{image_name}.jpg"  # Assuming images are named by ID with .jpg extension
        rgb = load_rgb(image_path)
        image_tensor = transform(rgb).unsqueeze(0).to(device)

        rgb_float = np.array(rgb).astype(np.float32) / 255.0
        rgb_float = cv2.resize(rgb_float, (224, 224), interpolation=cv2.INTER_LINEAR)

        lesion_mask_np = None
        if args.mask_root is not None and args.mask_col in row and not pd.isna(row[args.mask_col]):
            mask_path = Path(args.mask_root) / str(row[args.mask_col])
            if mask_path.exists():
                lesion_mask_np = load_binary_mask(mask_path, size=(224, 224))
            else:
                print(f"[warn] missing mask for {image_id}: {mask_path}")

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        sorted_idx = np.argsort(probs)[::-1]
        
        if args.compare_mode == "fixed":
            if args.A is None or args.B is None:
                raise ValueError("compare_mode=fixed requires --A and --B")
            A_idx = parse_class_name(args.A, class_to_idx)
            B_idx = parse_class_name(args.B, class_to_idx)
            comparison_categories = [B_idx]

        elif args.compare_mode == "pred_topk_non_target":
            A_idx = int(sorted_idx[0])
            comparison_categories = [int(i) for i in sorted_idx if int(i) != A_idx][: max(1, args.topk_compare)]
            if len(comparison_categories) == 0:
                raise ValueError(f"Could not find non-target comparison categories for image {image_id}.")
            B_idx = comparison_categories[0]

        elif args.compare_mode == "gt_topk_non_target":
            if gt_idx is None:
                raise ValueError("gt_topk_non_target requires ground-truth labels in the CSV.")
            A_idx = int(gt_idx)
            comparison_categories = [int(i) for i in sorted_idx if int(i) != A_idx][: max(1, args.topk_compare)]
            if len(comparison_categories) == 0:
                raise ValueError(f"Could not find non-target comparison categories for image {image_id}.")
            B_idx = comparison_categories[0]

        elif args.compare_mode == "top2":
            A_idx = int(sorted_idx[0])
            B_idx = int(sorted_idx[1])
            comparison_categories = [B_idx]

        else:
            raise ValueError(
                f"Unsupported compare_mode for this script: {args.compare_mode}. "
                "Use one of: fixed, top2, pred_topk_non_target, gt_topk_non_target."
            )

        bundle = compute_cam_bundle(
            model=model,
            input_tensor=image_tensor,
            rgb_float=rgb_float,
            target_layer=target_layer,
            reshape_transform=vit_reshape_transform,
            method="finercam",
            A=A_idx,
            B=B_idx,
            comparison_categories=comparison_categories,
            alpha=args.alpha,
            include_extra_maps=args.include_extra_maps,
        )

        pred_seg_gate_cam = None
        gate_weighted_finercam_cam = None

        if "pred_seg_gate" in requested_methods or "gate_weighted_finercam" in requested_methods:
            pred_seg_gate_cam = predict_seg_gate_map_for_metrics(
                model_raw=model_raw,
                input_tensor=image_tensor,
                image_size=(224, 224),
            )

            if pred_seg_gate_cam is None:
                raise ValueError(
                    "Requested pred_seg_gate or gate_weighted_finercam, "
                    "but the loaded model does not expose predict_seg_gate_map(). "
                    "Use these methods only with --checkpoint_model_type multitask or seggate."
                )

        if "gate_weighted_finercam" in requested_methods:
            gate_weighted_finercam_cam, _ = make_gate_weighted_cam_overlay(
                cam_map=bundle["cam_finercam"],
                gate_map=pred_seg_gate_cam,
                rgb_float=rgb_float,
            )

        pred_idx = int(np.argmax(bundle["probs"]))
        pred_name = class_names[pred_idx]
        pred_prob = safe_float(bundle["probs"][pred_idx])

        target_idx = int(A_idx)
        reference_idx = int(B_idx)
        target_name = class_names[target_idx]
        reference_name = class_names[reference_idx]
        target_prob = safe_float(bundle["probs"][target_idx])
        reference_prob = safe_float(bundle["probs"][reference_idx])

        cam_lookup = {
            "gradcam_target": bundle.get("cam_gradcam"),
            "gradcam_reference": bundle.get("cam_gradcam_B"),
            "gradcam_diff": bundle.get("cam_diff"),
            "finercam": bundle.get("cam_finercam"),
            "pred_seg_gate": pred_seg_gate_cam,
            "gate_weighted_finercam": gate_weighted_finercam_cam,
            "rollout": bundle.get("cam_rollout"),
            "chefer_style": bundle.get("cam_chefer"),
        }

        method_specs = []
        for method_name in requested_methods:
            cam_value = cam_lookup.get(method_name)
            if cam_value is None:
                print(f"[warn] skipping method={method_name} for image={image_id} because CAM is None")
                continue

            method_specs.append(
                {
                    "method": method_name,
                    "cam": cam_value,
                }
            )

        for spec in method_specs:
            cam_tensor = to_cam_tensor(spec["cam"], device=device)
            target_tensor = torch.tensor([target_idx], dtype=torch.long, device=device)
            reference_indices = comparison_categories if len(comparison_categories) > 0 else [reference_idx]
            reference_tensor = torch.tensor(reference_indices, dtype=torch.long, device=device)

            rel_conf_res = rel_conf_metric.compute(
                images=image_tensor,
                cams=cam_tensor,
                target_classes=target_tensor,
                reference_classes=reference_tensor,
                mask_value=args.mask_value,
            )
            del_res = del_metric.compute(
                images=image_tensor,
                cams=cam_tensor,
                target_classes=target_tensor,
                mask_value=args.mask_value,
            )
            ins_res = ins_metric.compute(
                images=image_tensor,
                cams=cam_tensor,
                target_classes=target_tensor,
            )

            compact = {
                "rel_conf_original_target_confidence": rel_conf_res["original_target_confidence"],
                "rel_conf_original_reference_confidence": rel_conf_res["original_reference_confidence"],
                "rel_conf_average_target_drop": rel_conf_res["average_target_drop"],
                "rel_conf_average_reference_drop": rel_conf_res["average_reference_drop"],
                "rel_conf_average_relative_confidence_drop": rel_conf_res["average_relative_confidence_drop"],
                "rel_conf_max_relative_confidence_drop": rel_conf_res["max_relative_confidence_drop"],
            }
            compact.update({f"del_{k}": v for k, v in summarize_metric_dict(del_res).items()})
            compact.update({f"ins_{k}": v for k, v in summarize_metric_dict(ins_res).items()})

            if lesion_mask_np is not None:
                compact.update(
                    compute_mask_overlap_metrics(
                        spec["cam"],
                        lesion_mask_np,
                        threshold=args.cam_threshold,
                    )
                )
            else:
                compact.update(
                    {
                        "mask_dice": None,
                        "mask_iou": None,
                        "mask_energy_inside": None,
                        "mask_pointing_game": None,
                        "mask_area_fraction": None,
                    }
                )
            sample_row = {
                "image_id": image_id,
                "image": image_name,
                "gt_idx": gt_idx,
                "gt_name": gt_name,
                "pred_idx": pred_idx,
                "pred_name": pred_name,
                "pred_prob": pred_prob,
                "target_idx": target_idx,
                "target_name": target_name,
                "target_prob": target_prob,
                "reference_idx": reference_idx,
                "reference_name": reference_name,
                "reference_prob": reference_prob,
                "comparison_categories": json.dumps([int(i) for i in comparison_categories]),
                "method": spec["method"],
                "compare_mode": args.compare_mode,
                "compare_tag": compare_tag,
                **compact,
            }
            per_sample_rows.append(sample_row)

            metrics_full.append(
                {
                    "image_id": image_id,
                    "image": image_name,
                    "method": spec["method"],
                    "target_idx": target_idx,
                    "target_name": target_name,
                    "comparison_categories": [int(i) for i in comparison_categories],
                    "reference_idx": reference_idx,
                    "reference_name": reference_name,
                    "compare_mode": args.compare_mode,
                    "compare_tag": compare_tag,
                    "checkpoint_model_type": info.get("checkpoint_model_type"),
                    "use_seg_gate": info.get("use_seg_gate"),
                    "seg_gate_bg_keep": info.get("seg_gate_bg_keep"),
                    "methods_requested": requested_methods,
                    "relative_confidence_drop": rel_conf_res,
                    "deletion": del_res,
                    "insertion": ins_res,
                }
            )

    per_sample_df = pd.DataFrame(per_sample_rows)

    summary_df = (
        per_sample_df.groupby(["method", "compare_mode", "compare_tag"], dropna=False)
        .agg(
            n=("image_id", "count"),

            rel_conf_target_mean=("rel_conf_original_target_confidence", "mean"),
            rel_conf_target_std=("rel_conf_original_target_confidence", "std"),

            rel_conf_reference_mean=("rel_conf_original_reference_confidence", "mean"),
            rel_conf_reference_std=("rel_conf_original_reference_confidence", "std"),

            rel_conf_target_drop_mean=("rel_conf_average_target_drop", "mean"),
            rel_conf_target_drop_std=("rel_conf_average_target_drop", "std"),

            rel_conf_reference_drop_mean=("rel_conf_average_reference_drop", "mean"),
            rel_conf_reference_drop_std=("rel_conf_average_reference_drop", "std"),

            rel_conf_rd_mean=("rel_conf_average_relative_confidence_drop", "mean"),
            rel_conf_rd_std=("rel_conf_average_relative_confidence_drop", "std"),

            del_auc_mean=("del_auc", "mean"),
            del_auc_std=("del_auc", "std"),

            ins_auc_mean=("ins_auc", "mean"),
            ins_auc_std=("ins_auc", "std"),

            mask_dice_mean=("mask_dice", "mean"),
            mask_dice_std=("mask_dice", "std"),

            mask_iou_mean=("mask_iou", "mean"),
            mask_iou_std=("mask_iou", "std"),

            mask_energy_inside_mean=("mask_energy_inside", "mean"),
            mask_energy_inside_std=("mask_energy_inside", "std"),

            mask_pointing_game_mean=("mask_pointing_game", "mean"),
            mask_pointing_game_std=("mask_pointing_game", "std"),

            mask_area_fraction_mean=("mask_area_fraction", "mean"),
            mask_area_fraction_std=("mask_area_fraction", "std"),
        )
        .reset_index()
    )

    per_sample_csv = out_dir / f"per_sample_metrics__{compare_tag}.csv"
    summary_csv = out_dir / f"per_method_summary__{compare_tag}.csv"
    full_json = out_dir / f"metrics_full__{compare_tag}.json"

    per_sample_df.to_csv(per_sample_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(metrics_full, f, indent=2)

    print("\nSaved:")
    print(f"- {per_sample_csv}")
    print(f"- {summary_csv}")
    print(f"- {full_json}")
    print("\nPer-method summary with mean/std:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()