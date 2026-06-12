# Input:
# - MEL image: ISIC_0031408
# - NV image: ISIC_0026181
# - model: GAP, CLS, or both
# - CAM method: FinerCAM by default
# - fixed class direction: GT vs other class

# Output:
# - one PDF per model
# - row 1: MEL image
# - row 2: NV image
# - columns:
#   1. RGB + lesion outline
#   2. late only [-1]
#   3. middle late [-4, -1]
#   4. early late [-10, -1]
#   5. early middle late [-10, -4, -1]
#   6. SkipPLUS style [-6, -5, -4, -3, -2, -1]
#   7. early only [-10]
#   8. middle only [-4]
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
from PIL import Image, ImageDraw, ImageFont
from pytorch_grad_cam import FinerCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, FinerWeightedTarget

REPO_ROOT = Path(__file__).resolve().parents[1]

PANDERM_CLASSIFICATION_DIR = (REPO_ROOT / "external" / "PanDerm" / "classification").resolve()
if str(PANDERM_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(PANDERM_CLASSIFICATION_DIR))

from models.builder import get_eval_transforms  # type: ignore

from scripts.generate_finer_cam_panderm import (  # type: ignore
    PanDermCAMWrapper,
    build_class_maps,
    load_mask_for_row,
    load_panderm_finetuned_model,
    make_gt_mask_overlay,
    make_safe_output_stem,
    resolve_class_names,
    resolve_gt_label_from_row,
    vit_reshape_transform,
)

CLINICIAN_OVERLAY_IMAGE_WEIGHT = 0.65

DEFAULT_VARIANTS = {
    "early_only": [-10],
    "middle_only": [-4],
    "late_only": [-1],
    "early_late": [-10, -1],
    "middle_late": [-4, -1],
    "early_middle_late": [-10, -4, -1],
    "skipplus_mid_late": [-6, -5, -4, -3, -2, -1],
}

DEFAULT_SCENARIOS = {
    "cls_ha05": {
        "name": "CLS HA 0.5",
        "short_name": "cls_ha05",
        "checkpoint": REPO_ROOT / "external" / "checkpoints4" / "checkpoint-best-cls.pth",
        "pooling": "cls",
    },
    "gap_ha025": {
        "name": "GAP HA 0.25",
        "short_name": "gap_ha025",
        "checkpoint": REPO_ROOT / "external" / "checkpoints4" / "checkpoint-best-gap.pth",
        "pooling": "mean",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SkipPLUS-style multi-block CAM fusion panels for MEL vs NV."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=str(REPO_ROOT / "data" / "HAM10000" / "mel_nv" / "ham_mel_nv_clean.csv"),
    )
    parser.add_argument("--image_col", type=str, default="image_rel_path")
    parser.add_argument("--id_col", type=str, default="image_id")
    parser.add_argument("--gt_col", type=str, default="gt_label")
    parser.add_argument("--mask_col", type=str, default="mask_rel_path")
    parser.add_argument("--img_dir", type=str, default=str(REPO_ROOT / "data" / "HAM10000"))
    parser.add_argument("--mask_root", type=str, default=str(REPO_ROOT / "data" / "HAM10000"))
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "mel_nv" / "skipplus_cam_review"),
    )

    parser.add_argument(
        "--image_ids",
        type=str,
        default="ISIC_0031408,ISIC_0026181",
        help="Default: one MEL and one NV image.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gap_ha025,cls_ha05",
        help="Options: gap_ha025, cls_ha05.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="late_only,middle_late,early_late,early_middle_late,skipplus_mid_late,early_only,middle_only",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="finercam",
        choices=["gradcam", "map_diff", "finercam"],
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Fusion after per-block min-max normalization.",
    )
    parser.add_argument(
        "--target_direction",
        type=str,
        default="gt_pair",
        choices=["gt_pair", "inverse_gt_pair"],
        help="gt_pair = GT class vs other class. inverse_gt_pair = other class vs GT class.",
    )

    parser.add_argument("--A", type=str, default="MEL")
    parser.add_argument("--B", type=str, default="NV")
    parser.add_argument("--class_names", type=str, default="MEL,NV")
    parser.add_argument("--class_preset", type=str, default="ham", choices=["ham", "bcn"])
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--checkpoint_model_type", type=str, default="panderm")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--panel_scale", type=float, default=1.35)
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--save_raw_fused", action="store_true")

    return parser.parse_args()


def get_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def minmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).squeeze().astype(np.float32)
    x = x - np.nanmin(x)
    denom = np.nanmax(x) + 1e-8
    return x / denom


def cam_overlay(rgb_float: np.ndarray, cam_map: np.ndarray) -> np.ndarray:
    return show_cam_on_image(
        rgb_float,
        minmax_np(cam_map),
        use_rgb=True,
        image_weight=CLINICIAN_OVERLAY_IMAGE_WEIGHT,
    )


def get_image_id(row: pd.Series, image_col: str, id_col: str) -> str:
    if id_col in row.index and pd.notna(row[id_col]):
        return str(row[id_col])
    value = str(row[image_col])
    return Path(value).stem if Path(value).suffix else value


def resolve_image_path(row: pd.Series, image_col: str, img_dir: Path) -> Path:
    value = Path(str(row[image_col]))
    if value.is_absolute():
        return value
    if value.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        return img_dir / value
    return img_dir / f"{value}.jpg"


def resolve_block_index(requested_block: int, n_blocks: int) -> int:
    block = int(requested_block)
    if block < 0:
        block = n_blocks + block
    if block < 0 or block >= n_blocks:
        raise ValueError(
            f"Requested block {requested_block} resolves to {block}, "
            f"but model has {n_blocks} blocks."
        )
    return block


def resolve_pair_for_row(
    row: pd.Series,
    args: argparse.Namespace,
    class_names: list[str],
    class_to_idx: dict[str, int],
) -> tuple[int, int, str, str, str]:
    gt_name = resolve_gt_label_from_row(
        row=row,
        class_names=class_names,
        class_to_idx=class_to_idx,
        gt_col=args.gt_col,
    )

    pair = [args.A, args.B]
    pair_lookup = {name.upper(): name for name in pair}
    gt_key = gt_name.upper()

    if gt_key not in pair_lookup:
        raise ValueError(f"GT label {gt_name} is not in pair {pair}.")

    gt_pair_name = pair_lookup[gt_key]
    other_pair_name = pair[1] if gt_pair_name == pair[0] else pair[0]

    if args.target_direction == "gt_pair":
        a_name = gt_pair_name
        b_name = other_pair_name
    else:
        a_name = other_pair_name
        b_name = gt_pair_name

    direction_label = f"{a_name} vs {b_name}"
    return class_to_idx[a_name], class_to_idx[b_name], a_name, b_name, direction_label


def compute_block_cam(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_layer,
    method: str,
    a_idx: int,
    b_idx: int,
    alpha: float,
) -> np.ndarray:
    if method == "gradcam":
        cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=vit_reshape_transform,
        )
        return cam(input_tensor=x, targets=[ClassifierOutputTarget(a_idx)])[0]

    cam_ref = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform,
    )
    cam_a = cam_ref(input_tensor=x, targets=[ClassifierOutputTarget(a_idx)])[0]
    cam_b = cam_ref(input_tensor=x, targets=[ClassifierOutputTarget(b_idx)])[0]

    if method == "map_diff":
        return np.maximum(cam_a - cam_b, 0.0)

    finer = FinerCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform,
    )
    target = FinerWeightedTarget(
        main_category=a_idx,
        comparison_categories=[b_idx],
        alpha=alpha,
    )
    return finer(input_tensor=x, targets=[target])[0]


def fuse_cams(cams: list[np.ndarray], fusion: str) -> np.ndarray:
    normalized = [minmax_np(cam) for cam in cams]
    stack = np.stack(normalized, axis=0)

    if fusion == "mean":
        fused = stack.mean(axis=0)
    elif fusion == "max":
        fused = stack.max(axis=0)
    else:
        raise ValueError(f"Unknown fusion: {fusion}")

    return minmax_np(fused)


def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill=(0, 0, 0)) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) / 2
    y = top + (bottom - top - text_h) / 2
    draw.text((x, y), text, font=font, fill=fill)


def to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def make_fusion_panel(
    rows: list[dict],
    variant_names: list[str],
    scenario_name: str,
    method: str,
    fusion: str,
    scale: float,
) -> Image.Image:
    font_title = load_font(30, bold=True)
    font_header = load_font(21, bold=True)
    font_small = load_font(17, bold=False)
    font_row = load_font(18, bold=True)

    tile_h, tile_w = rows[0]["rgb_overlay"].shape[:2]
    scaled_w = max(1, int(round(tile_w * scale)))
    scaled_h = max(1, int(round(tile_h * scale)))

    row_label_w = 230
    gap_x = 14
    gap_y = 30
    margin = 34
    title_h = 90
    header_h = 56
    bottom_h = 40

    n_cols = 1 + len(variant_names)
    page_w = margin * 2 + row_label_w + n_cols * scaled_w + (n_cols - 1) * gap_x
    row_h = header_h + scaled_h + bottom_h
    page_h = margin * 2 + title_h + len(rows) * row_h + (len(rows) - 1) * gap_y

    canvas = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"SkipPLUS-style multi-block CAM fusion | {scenario_name}"
    subtitle = f"Method: {method} | Fusion: normalized {fusion} | Same variants applied to MEL and NV"

    draw.text((margin, margin - 4), title, font=font_title, fill=(0, 0, 0))
    draw.text((margin, margin + 40), subtitle, font=font_small, fill=(70, 70, 70))

    x0 = margin + row_label_w
    y = margin + title_h

    column_titles = ["Image + outline"] + variant_names

    for row in rows:
        draw.text(
            (margin, y + header_h + scaled_h / 2 - 12),
            row["row_label"],
            font=font_row,
            fill=(0, 0, 0),
        )
        draw.text(
            (margin, y + header_h + scaled_h / 2 + 15),
            row["row_subtitle"],
            font=font_small,
            fill=(80, 80, 80),
        )

        tiles = [row["rgb_overlay"]] + [row["variant_overlays"][name] for name in variant_names]
        line2s = [row["image_line2"]] + [row["variant_line2"][name] for name in variant_names]

        x = x0
        for title_text, tile, line2 in zip(column_titles, tiles, line2s):
            tile_uint8 = to_uint8(tile)
            tile_img = Image.fromarray(tile_uint8).resize(
                (scaled_w, scaled_h),
                Image.Resampling.BICUBIC,
            )

            draw_centered_text(
                draw,
                (x, y, x + scaled_w, y + header_h),
                title_text,
                font_header,
            )
            canvas.paste(tile_img, (x, y + header_h))
            draw_centered_text(
                draw,
                (x, y + header_h + scaled_h + 2, x + scaled_w, y + header_h + scaled_h + bottom_h),
                line2,
                font_small,
                fill=(40, 40, 40),
            )

            x += scaled_w + gap_x

        y += row_h + gap_y

    return canvas


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    mask_root = Path(args.mask_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.class_names = args.class_names if args.class_names else None
    class_names = resolve_class_names(args)
    class_to_idx, idx_to_class = build_class_maps(class_names)

    image_ids = [x.strip() for x in args.image_ids.split(",") if x.strip()]
    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    variant_names = [x.strip() for x in args.variants.split(",") if x.strip()]

    unknown_models = [m for m in model_keys if m not in DEFAULT_SCENARIOS]
    if unknown_models:
        raise ValueError(f"Unknown model keys: {unknown_models}. Allowed: {list(DEFAULT_SCENARIOS.keys())}")

    unknown_variants = [v for v in variant_names if v not in DEFAULT_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}. Allowed: {list(DEFAULT_VARIANTS.keys())}")

    df = pd.read_csv(csv_path)

    if args.id_col not in df.columns:
        raise ValueError(f"CSV must contain id_col={args.id_col}. Found: {df.columns.tolist()}")

    selected_df = df[df[args.id_col].astype(str).isin(image_ids)].copy()

    if len(selected_df) != len(image_ids):
        found = set(selected_df[args.id_col].astype(str))
        missing = [image_id for image_id in image_ids if image_id not in found]
        raise ValueError(f"Could not find all requested image IDs. Missing: {missing}")

    selected_df["_order"] = selected_df[args.id_col].astype(str).map(
        {image_id: i for i, image_id in enumerate(image_ids)}
    )
    selected_df = selected_df.sort_values("_order").reset_index(drop=True)

    device = get_device(args.device)

    preprocess = get_eval_transforms(
        which_img_norm="imagenet",
        img_resize=256,
        center_crop=True,
    )
    if preprocess is None:
        preprocess = T.Compose(
            [
                T.Resize((args.image_size, args.image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    manifest = {
        "csv": str(csv_path),
        "image_ids": image_ids,
        "models": model_keys,
        "variants": {name: DEFAULT_VARIANTS[name] for name in variant_names},
        "method": args.method,
        "fusion": args.fusion,
        "target_direction": args.target_direction,
        "alpha": args.alpha,
        "outputs": [],
    }

    for model_key in model_keys:
        scenario = DEFAULT_SCENARIOS[model_key]
        checkpoint = Path(scenario["checkpoint"])

        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {model_key}: {checkpoint}")

        model_raw, info = load_panderm_finetuned_model(
            checkpoint_path=checkpoint,
            num_classes=len(class_names),
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            device=device,
            checkpoint_model_type=args.checkpoint_model_type,
            pooling=scenario["pooling"],
        )
        print(f"[loaded] {scenario['name']} from {checkpoint}", flush=True)

        model = PanDermCAMWrapper(model_raw)
        model.eval()

        blocks = model_raw.backbone.blocks if hasattr(model_raw, "backbone") else model_raw.blocks

        rows_for_panel = []
        raw_root = out_dir / "raw_fused" / scenario["short_name"]
        raw_root.mkdir(parents=True, exist_ok=True)

        for _, row in selected_df.iterrows():
            print(f"[image] starting {row[args.id_col]}", flush=True)
            image_id = get_image_id(row, image_col=args.image_col, id_col=args.id_col)
            img_path = resolve_image_path(row, image_col=args.image_col, img_dir=img_dir)

            if not img_path.exists():
                raise FileNotFoundError(f"Missing image for {image_id}: {img_path}")

            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_label = idx_to_class[pred_idx]
            pred_conf = float(probs[pred_idx])

            gt_label = resolve_gt_label_from_row(
                row=row,
                class_names=class_names,
                class_to_idx=class_to_idx,
                gt_col=args.gt_col,
            )

            rgb = np.array(img).astype(np.float32) / 255.0
            rgb_resized = cv2.resize(
                rgb,
                (args.image_size, args.image_size),
                interpolation=cv2.INTER_LINEAR,
            )

            mask_img = load_mask_for_row(
                row=row,
                mask_root=mask_root,
                mask_col=args.mask_col,
            )
            if mask_img is None:
                raise FileNotFoundError(f"Missing mask for {image_id} using column {args.mask_col}.")

            rgb_overlay = make_gt_mask_overlay(mask_img, rgb_resized)

            a_idx, b_idx, a_name, b_name, direction_label = resolve_pair_for_row(
                row=row,
                args=args,
                class_names=class_names,
                class_to_idx=class_to_idx,
            )

            cache: dict[int, np.ndarray] = {}
            variant_overlays: dict[str, np.ndarray] = {}
            variant_line2: dict[str, str] = {}

            for variant_name in variant_names:
                print(f"[variant] {image_id} | {scenario['short_name']} | {variant_name}", flush=True)
                requested_blocks = DEFAULT_VARIANTS[variant_name]
                cams = []

                for requested_block in requested_blocks:
                    resolved_idx = resolve_block_index(requested_block, len(blocks))

                    if requested_block not in cache:
                        target_layer = blocks[resolved_idx].norm1

                        print(f"  [block] computing block {requested_block}", flush=True)
                        cache[requested_block] = compute_block_cam(
                            model=model,
                            x=x,
                            target_layer=target_layer,
                            method=args.method,
                            a_idx=a_idx,
                            b_idx=b_idx,
                            alpha=args.alpha,
                        )

                    cams.append(cache[requested_block])

                fused = fuse_cams(cams, fusion=args.fusion)
                variant_overlays[variant_name] = cam_overlay(rgb_resized, fused)
                variant_line2[variant_name] = ",".join(str(b) for b in requested_blocks)

                if args.save_raw_fused:
                    output_stem = make_safe_output_stem(image_id)
                    raw_path = raw_root / f"{output_stem}_{variant_name}_{args.method}_{args.fusion}.npy"
                    np.save(raw_path, fused.astype(np.float32))

            prediction_status = "correct" if pred_label == gt_label else "wrong"

            rows_for_panel.append(
                {
                    "row_label": f"{image_id} | GT: {gt_label}",
                    "row_subtitle": f"Pred: {pred_label} ({pred_conf * 100:.1f}%, {prediction_status}) | {direction_label}",
                    "image_line2": "lesion outline",
                    "rgb_overlay": rgb_overlay,
                    "variant_overlays": variant_overlays,
                    "variant_line2": variant_line2,
                }
            )

        panel = make_fusion_panel(
            rows=rows_for_panel,
            variant_names=variant_names,
            scenario_name=scenario["name"],
            method=args.method,
            fusion=args.fusion,
            scale=args.panel_scale,
        )

        base_name = f"skipplus_cam_fusion_{scenario['short_name']}_{args.method}_{args.fusion}_{args.target_direction}"
        pdf_path = out_dir / f"{base_name}.pdf"
        panel.save(pdf_path, resolution=150.0)
        print("Saved PDF:", pdf_path)
        manifest["outputs"].append(str(pdf_path))

        if args.save_png:
            png_path = out_dir / f"{base_name}.png"
            panel.save(png_path)
            print("Saved PNG:", png_path)
            manifest["outputs"].append(str(png_path))

        del model_raw
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = out_dir / "skipplus_cam_fusion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    main()