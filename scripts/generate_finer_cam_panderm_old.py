# python -m scripts.generate_finer_cam_panderm \
#   --csv data/isic2018/subsets/val_subset1_gt_MEL_or_NV.csv \
#   --img_dir data/isic2018/images_val \
#   --checkpoint external/weights/panderm_bcn_ft_best.pt \
#   --out_dir outputs/isic7_cam/panderm_pred_topk2 \
#   --num_samples 20 \
#   --method finercam \
#   --compare_mode pred_topk_non_target \
#   --topk_compare 2

from __future__ import annotations

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

from src.cam.diff_cam import compute_cam_triplet
from src.utils.vis_panel import make_panel_with_subtitles

REPO_ROOT = Path(__file__).resolve().parents[1]
PANDERM_CLASSIFICATION_DIR = (REPO_ROOT / "external" / "PanDerm" / "classification").resolve()
if str(PANDERM_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(PANDERM_CLASSIFICATION_DIR))

from models.builder import get_eval_transforms  # type: ignore
from models.modeling_finetune import \
    panderm_base_patch16_224_finetune  # type: ignore

PRIMARY_CLASSES = ["NV", "MEL", "BCC", "BKL", "AKIEC", "DF"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PRIMARY_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RGB / A / B / Finer-CAM panels for a fine-tuned PanDerm checkpoint.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with column 'image' or 'isic_id'.")
    parser.add_argument("--img_dir", type=str, required=True, help="Folder containing JPG images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PanDerm fine-tuned .pt checkpoint.")
    parser.add_argument("--out_dir", type=str, default="outputs/panderm_cam", help="Output folder.")
    parser.add_argument("--image_size", type=int, default=224, help="PanDerm input size. Default: 224.")
    parser.add_argument("--num_samples", type=int, default=10, help="How many images to process.")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps (default: auto).")
    parser.add_argument("--method", type=str, default="finercam", choices=["gradcam", "layercam", "finercam"], help="CAM backend for the main triplet.")
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
    return parser.parse_args()


def get_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_panderm_model(num_classes: int) -> torch.nn.Module:
    model = panderm_base_patch16_224_finetune(
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=False,
        init_scale=0.001,
        use_rel_pos_bias=False,
        init_values=1e-5,
        lin_probe=False,
    )
    return model


def load_panderm_finetuned_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = get_device(None)

    model = build_panderm_model(num_classes=len(PRIMARY_CLASSES))

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError("PanDerm checkpoint must contain 'model_state_dict'.")

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if len(missing) or len(unexpected):
        print(f"[warn] load_state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing sample:", missing[:10])
        if unexpected:
            print("  unexpected sample:", unexpected[:10])

    model = model.to(device)
    model.eval()

    info = {
        "arch": "PanDerm Base",
        "num_classes": len(PRIMARY_CLASSES),
        "checkpoint_name": checkpoint_path.name,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
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


def get_image_id(row: pd.Series) -> str:
    if "image" in row:
        return str(row["image"])
    if "isic_id" in row:
        return str(row["isic_id"])
    raise ValueError("CSV must contain column 'image' or 'isic_id'.")


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    model_raw, info = load_panderm_finetuned_model(ckpt_path, device=device)
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

    target_layer = model_raw.blocks[-1].norm1

    df = pd.read_csv(csv_path)
    if "image" not in df.columns and "isic_id" not in df.columns:
        raise ValueError(f"CSV must contain column 'image' or 'isic_id'. Found: {df.columns.tolist()}")

    df = df.head(args.num_samples)

    for _, row in df.iterrows():
        image_id = get_image_id(row)
        img_path = img_dir / f"{image_id}.jpg"
        if not img_path.exists():
            print(f"[skip] missing image: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        sorted_idx = np.argsort(probs)[::-1]

        rgb = np.array(img).astype(np.float32) / 255.0
        rgb_resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        A_idx, B_idx = None, None
        comparison_categories = None

        if args.compare_mode == "fixed":
            if args.A is None or args.B is None:
                raise ValueError("compare_mode=fixed requires --A and --B")
            if args.A not in CLASS_TO_IDX or args.B not in CLASS_TO_IDX:
                raise ValueError(f"Unknown class name. Allowed: {PRIMARY_CLASSES}")
            A_idx = CLASS_TO_IDX[args.A]
            B_idx = CLASS_TO_IDX[args.B]
            comparison_categories = [B_idx]

        elif args.compare_mode == "gt_pair":
            if args.A is None or args.B is None:
                raise ValueError("compare_mode=gt_pair requires --A and --B")
            if args.A not in CLASS_TO_IDX or args.B not in CLASS_TO_IDX:
                raise ValueError(f"Unknown class name. Allowed: {PRIMARY_CLASSES}")
            if "gt_label" not in df.columns:
                raise ValueError("compare_mode=gt_pair requires the CSV to contain a 'gt_label' column.")
            gt = str(row["gt_label"])
            if gt not in [args.A, args.B]:
                print(f"[skip] {image_id}: gt_label={gt} not in pair ({args.A},{args.B})")
                continue
            if gt == args.A:
                A_idx = CLASS_TO_IDX[args.A]
                B_idx = CLASS_TO_IDX[args.B]
            else:
                A_idx = CLASS_TO_IDX[args.B]
                B_idx = CLASS_TO_IDX[args.A]
            comparison_categories = [B_idx]

        elif args.compare_mode == "pred_topk_non_target":
            A_idx = int(sorted_idx[0])
            comparison_categories = [int(i) for i in sorted_idx if int(i) != A_idx][: max(1, args.topk_compare)]
            B_idx = comparison_categories[0]

        elif args.compare_mode == "gt_topk_non_target":
            if "gt_label" not in df.columns:
                raise ValueError("compare_mode=gt_topk_non_target requires the CSV to contain a 'gt_label' column.")
            gt = str(row["gt_label"])
            if gt not in CLASS_TO_IDX:
                print(f"[skip] {image_id}: gt_label={gt} not in PRIMARY_CLASSES")
                continue
            A_idx = CLASS_TO_IDX[gt]
            comparison_categories = [int(i) for i in sorted_idx if int(i) != A_idx][: max(1, args.topk_compare)]
            if len(comparison_categories) == 0:
                print(f"[skip] {image_id}: could not find non-target comparison categories")
                continue
            B_idx = comparison_categories[0]

        res = compute_cam_triplet(
            model=model,
            input_tensor=x,
            rgb_float=rgb_resized,
            target_layer=target_layer,
            method=args.method,
            A=A_idx,
            B=B_idx,
            comparison_categories=comparison_categories,
            reshape_transform=vit_reshape_transform,
        )

        top3_idx = np.argsort(res["probs"])[-3:][::-1]
        top3_named = ", ".join([f"{IDX_TO_CLASS[i]}: {res['probs'][i]:.3f}" for i in top3_idx])

        A_name = IDX_TO_CLASS.get(int(res["A"]), str(res["A"]))
        B_name = IDX_TO_CLASS.get(int(res["B"]), str(res["B"]))
        comp_named = ", ".join([IDX_TO_CLASS[int(i)] for i in res.get("comparison_categories", [res["B"]])])
        print(f"[info] {image_id}: A={res['A']}({A_name})  B={res['B']}({B_name})  comparison=[{comp_named}]  top3=[{top3_named}]")

        gt_label = row["gt_label"] if "gt_label" in df.columns else None
        A_prob = float(res["probs"][int(res["A"])])
        B_prob = float(res["probs"][int(res["B"])])

        panel_img_uint8 = make_panel_with_subtitles(
            image_id=str(image_id),
            rgb_float=rgb_resized,
            overlay_A=res["overlay_A"],
            overlay_B=res["overlay_B"],
            overlay_diff=res["overlay_diff"],
            method=args.method,
            A_name=A_name,
            B_name=B_name,
            A_prob=A_prob,
            B_prob=B_prob,
            gt_label=gt_label,
            include_rgb=True,
        )

        panel_path = out_dir / f"{image_id}_RGB_A_B_DIFF_{args.method}.png"
        Image.fromarray(panel_img_uint8).save(panel_path)

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
            "comparison_category_names": [IDX_TO_CLASS[int(i)] for i in res.get("comparison_categories", [res["B"]])],
            "topk_compare": int(args.topk_compare),
            "probs_top3": res["probs_top3"],
            "method": args.method,
            "panel_path": str(panel_path),
            "compare_mode": args.compare_mode,
            "A_name": A_name,
            "B_name": B_name,
            "target_layer": "model_raw.blocks[-1].norm1",
        }
        (out_dir / f"{image_id}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()