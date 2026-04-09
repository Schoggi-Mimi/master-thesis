"""
Evaluate CAM metrics for fine-tuned PanDerm models.

Outputs:
- per_sample_metrics.csv
- per_method_summary.csv
- metrics_full.json

Typical usage:
python -m scripts.eval_cam_metrics_panderm \
  --csv data/HAM10000/ham_test_for_cam.csv \
  --img_dir data/HAM10000/images \
  --checkpoint external/weights/checkpoint-best-ham.pth \
  --class_preset ham \
  --out_dir outputs/panderm_cam_metrics_ham \
  --num_samples 20 \
  --compare_mode pred_topk_non_target \
  --topk_compare 2

python -m scripts.eval_cam_metrics_panderm \
  --csv data/BCN20000/bcn_test_for_cam.csv \
  --img_dir data/BCN20000/images \
  --checkpoint external/weights/checkpoint-best-bcn.pth \
  --class_preset bcn \
  --out_dir outputs/panderm_cam_metrics_bcn \
  --num_samples 20 \
  --compare_mode pred_topk_non_target \
  --topk_compare 2

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
                                                resolve_class_names,
                                                vit_reshape_transform)
from src.cam.diff_cam import compute_cam_bundle
from src.eval.cam_metrics import (ConfidenceDropMetric, DeletionMetric,
                                  InsertionMetric, summarize_metric_dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
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

    p.add_argument("--compare_mode", type=str, default="pred_topk_non_target")
    p.add_argument("--topk_compare", type=int, default=2)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Alpha weight for FinerCAM comparison categories. Default: 0.6.",
    )
    p.add_argument("--cam_target_layer", type=str, default="last_block")
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

    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    )
    model = PanDermCAMWrapper(model_raw)
    model.eval()
    target_layer = model_raw.blocks[-1].norm1

    transform = build_eval_transform()

    conf_metric = ConfidenceDropMetric(
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

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        sorted_idx = np.argsort(probs)[::-1]

        if args.compare_mode == "pred_topk_non_target":
            A_idx = int(sorted_idx[0])
            comparison_categories = [int(i) for i in sorted_idx if int(i) != A_idx][: max(1, args.topk_compare)]
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
                "Use one of: top2, pred_topk_non_target, gt_topk_non_target."
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
        )

        pred_idx = int(np.argmax(bundle["probs"]))
        pred_name = class_names[pred_idx]
        pred_prob = safe_float(bundle["probs"][pred_idx])

        compare_idx = int(bundle["B"])
        compare_name = class_names[compare_idx]
        compare_prob = safe_float(bundle["probs"][compare_idx])

        method_specs = [
            {
                "method": "gradcam_pred",
                "cam": bundle["cam_gradcam"],
                "target_idx": pred_idx,
                "target_name": pred_name,
                "target_prob": pred_prob,
            },
            {
                "method": "gradcam_compare",
                "cam": bundle["cam_gradcam_B"],
                "target_idx": compare_idx,
                "target_name": compare_name,
                "target_prob": compare_prob,
            },
            {
                "method": "finercam",
                "cam": bundle["cam_finercam"],
                "target_idx": pred_idx,
                "target_name": pred_name,
                "target_prob": pred_prob,
            },
            {
                "method": "rollout",
                "cam": bundle["cam_rollout"],
                "target_idx": pred_idx,
                "target_name": pred_name,
                "target_prob": pred_prob,
            },
        ]

        for spec in method_specs:
            cam_tensor = to_cam_tensor(spec["cam"], device=device)
            target_tensor = torch.tensor([spec["target_idx"]], dtype=torch.long, device=device)

            conf_res = conf_metric.compute(
                images=image_tensor,
                cams=cam_tensor,
                target_classes=target_tensor,
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

            compact = {}
            compact.update({f"conf_{k}": v for k, v in summarize_metric_dict(conf_res).items()})
            compact.update({f"del_{k}": v for k, v in summarize_metric_dict(del_res).items()})
            compact.update({f"ins_{k}": v for k, v in summarize_metric_dict(ins_res).items()})

            sample_row = {
                "image_id": image_id,
                "image": image_name,
                "gt_idx": gt_idx,
                "gt_name": gt_name,
                "pred_idx": pred_idx,
                "pred_name": pred_name,
                "pred_prob": pred_prob,
                "compare_idx": compare_idx,
                "compare_name": compare_name,
                "compare_prob": compare_prob,
                "method": spec["method"],
                "target_idx": spec["target_idx"],
                "target_name": spec["target_name"],
                "target_prob": spec["target_prob"],
                **compact,
            }
            per_sample_rows.append(sample_row)

            metrics_full.append(
                {
                    "image_id": image_id,
                    "image": image_name,
                    "method": spec["method"],
                    "target_idx": spec["target_idx"],
                    "target_name": spec["target_name"],
                    "confidence_drop": conf_res,
                    "deletion": del_res,
                    "insertion": ins_res,
                }
            )

    per_sample_df = pd.DataFrame(per_sample_rows)

    summary_df = (
        per_sample_df.groupby("method", dropna=False)
        .agg(
            n=("image_id", "count"),
            conf_original=("conf_original_confidence", "mean"),
            conf_avg_drop=("conf_average_drop", "mean"),
            conf_rel_drop=("conf_relative_confidence_drop", "mean"),
            del_auc=("del_auc", "mean"),
            ins_auc=("ins_auc", "mean"),
        )
        .reset_index()
    )

    per_sample_csv = out_dir / "per_sample_metrics.csv"
    summary_csv = out_dir / "per_method_summary.csv"
    full_json = out_dir / "metrics_full.json"

    per_sample_df.to_csv(per_sample_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(metrics_full, f, indent=2)

    print("\nSaved:")
    print(f"- {per_sample_csv}")
    print(f"- {summary_csv}")
    print(f"- {full_json}")
    print("\nPer-method summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()