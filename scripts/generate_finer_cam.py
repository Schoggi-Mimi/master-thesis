# scripts/generate_finer_cam_panderm.py

# python -m scripts.generate_finer_cam_panderm \
#   --csv data/panderm/val_gt.csv \
#   --img_dir data/panderm/images_val \
#   --checkpoint external/weights/checkpoint-best-bcn.pth \
#   --out_dir outputs/panderm_cam/val_bcn \
#   --num_samples 5 \
#   --method gradcam \
#   --save_layercam_diff \
#   --compare_mode gt_pair \
#   --A mel \
#   --B nv

#   --compare_mode top2

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

from models.modeling_finetune import panderm_base_patch16_224_finetune  # type: ignore


HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
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


def parse_args():
    parser = argparse.ArgumentParser(description="Engineering test: generate (Diff) CAMs for PanDerm images.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with column 'image' (no .jpg).")
    parser.add_argument("--img_dir", type=str, required=True, help="Folder containing JPG images.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PanDerm finetuned checkpoint.",
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
    parser.add_argument("--out_dir", type=str, default="outputs/debug_cam", help="Output folder.")
    parser.add_argument("--image_size", type=int, default=224, help="Resize to (image_size, image_size).")
    parser.add_argument("--num_samples", type=int, default=10, help="How many images to process (from top of CSV).")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda or mps (default: auto).")
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=["gradcam", "layercam", "finercam"],
        help="CAM backend for the main triplet.",
    )
    parser.add_argument(
        "--compare_mode",
        type=str,
        default="top2",
        choices=["top2", "fixed", "gt_pair", "pred_topk_non_target", "gt_topk_non_target"],
        help=(
            "How to choose classes A and B.\n"
            "top2 = predicted top1/top2.\n"
            "fixed = user-defined --A/--B used as A/B exactly.\n"
            "gt_pair = use --A/--B as the pair; per image set A=gt_label and B=the other one.\n"
            "pred_topk_non_target = use predicted topk non-target classes for B.\n"
            "gt_topk_non_target = use gt label and topk non-target classes."
        ),
    )
    parser.add_argument("--topk_compare", type=int, default=3, help="Number of top-k classes to compare for certain modes.")
    parser.add_argument("--A", type=str, default=None, help="Fixed target class name (e.g., mel). Used when compare_mode=fixed or gt_pair.")
    parser.add_argument("--B", type=str, default=None, help="Fixed comparison class name (e.g., nv). Used when compare_mode=fixed or gt_pair.")
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


def build_panderm_model(num_classes: int) -> torch.nn.Module:
    model = panderm_base_patch16_224_finetune(
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.2,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        init_values=0.1,
        lin_probe=False,
    )
    return model


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

    return remapped


def load_panderm_finetuned_model(
    checkpoint_path: str | Path,
    num_classes: int,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = get_device(None)

    model = build_panderm_model(num_classes=num_classes)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        checkpoint_format = "custom_pt"
    elif "model" in ckpt:
        state_dict = remap_official_finetune_checkpoint_keys(ckpt["model"])
        checkpoint_format = "official_pth"
    else:
        raise KeyError("Checkpoint must contain either 'model_state_dict' or 'model'.")

    state_dict_model = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in state_dict and k in state_dict_model and state_dict[k].shape != state_dict_model[k].shape:
            raise ValueError(
                f"Checkpoint head shape mismatch for {k}: checkpoint={tuple(state_dict[k].shape)} vs model={tuple(state_dict_model[k].shape)}. "
                f"Check that --class_preset / --class_names matches the trained checkpoint."
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) or len(unexpected):
        print(f"[warn] load_state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing sample:", missing[:10])
        if unexpected:
            print("  unexpected sample:", unexpected[:10])

    model = model.to(device)
    model.eval()

    info = {
        "arch": "PanDerm Base FT",
        "num_classes": num_classes,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_format": checkpoint_format,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": ckpt.get("image_size", 224),
        "stage_name": ckpt.get("stage_name", None),
        "epoch": ckpt.get("epoch", None),
    }
    return model, info


class PanDermCAMWrapper:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    class_names = resolve_class_names(args)
    class_to_idx, idx_to_class = build_class_maps(class_names)

    model_raw, info = load_panderm_finetuned_model(
        ckpt_path,
        num_classes=len(class_names),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        device=device,
    )

    target_layer = model_raw.blocks[-1].norm1

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    df = pd.read_csv(csv_path)
    if not any(col in df.columns for col in ["image_path", "image", "image_id"]):
        raise ValueError(
            "CSV must contain at least one of: 'image_path', 'image', or 'image_id'. "
            f"Found: {df.columns.tolist()}"
        )

    df = df.head(args.num_samples)

    for idx, row in df.iterrows():
        if "image_path" in df.columns and pd.notna(row["image_path"]):
            img_path = Path(str(row["image_path"]))
            image_id = img_path.stem
        else:
            if "image" in df.columns and pd.notna(row["image"]):
                image_token = str(row["image"])
            elif "image_id" in df.columns and pd.notna(row["image_id"]):
                image_token = str(row["image_id"])
            else:
                print(f"[skip] row {idx}: could not resolve image identifier")
                continue

            image_id = Path(image_token).stem
            img_path = img_dir / f"{image_id}.jpg"

        if not img_path.exists():
            print(f"[skip] missing image: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        x = preprocess(img).unsqueeze(0).to(device)

        rgb = np.array(img).astype(np.float32) / 255.0
        rgb_resized = cv2.resize(rgb, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)

        with torch.no_grad():
            logits = model_raw(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            sorted_idx = np.argsort(probs)[::-1]

        A_idx, B_idx = None, None
        comparison_categories = []

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
            if "gt_label" not in df.columns:
                raise ValueError("compare_mode=gt_pair requires the CSV to contain a 'gt_label' column.")
            gt = str(row["gt_label"])
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
            if "gt_label" not in df.columns:
                raise ValueError("compare_mode=gt_topk_non_target requires the CSV to contain a 'gt_label' column.")
            gt = str(row["gt_label"])
            if gt not in class_to_idx:
                print(f"[skip] {image_id}: gt_label={gt} not in class list")
                continue
            A_idx = class_to_idx[gt]
            comparison_categories = [int(i) for i in sorted_idx if int(i) != A_idx][: max(1, args.topk_compare)]
            if len(comparison_categories) == 0:
                print(f"[skip] {image_id}: could not find non-target comparison categories")
                continue
            B_idx = comparison_categories[0]

        else:
            # Default top2
            A_idx = int(sorted_idx[0])
            B_idx = int(sorted_idx[1])
            comparison_categories = [B_idx]

        res = compute_cam_triplet(
            model=model_raw,
            input_tensor=x,
            rgb_float=rgb_resized,
            target_layer=target_layer,
            method=args.method,
            A=A_idx,
            B=B_idx,
            comparison_categories=comparison_categories,
        )

        top3_idx = np.argsort(res["probs"])[-3:][::-1]
        top3_named = ", ".join([f"{idx_to_class[i]}: {res['probs'][i]:.3f}" for i in top3_idx])

        A_name = idx_to_class.get(int(res["A"]), str(res["A"]))
        B_name = idx_to_class.get(int(res["B"]), str(res["B"]))
        comp_named = ", ".join([idx_to_class[int(i)] for i in res.get("comparison_categories", [res["B"]])])

        print(f"[info] {image_id}: A={res['A']}({A_name})  B={res['B']}({B_name})  top3=[{top3_named}]  compare=[{comp_named}]")

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
        overlay_A_path = out_dir / f"{image_id}_overlay_A_{A_name}_{args.method}.png"
        overlay_B_path = out_dir / f"{image_id}_overlay_B_{B_name}_{args.method}.png"
        overlay_diff_path = out_dir / f"{image_id}_overlay_DIFF_{A_name}_vs_{B_name}_{args.method}.png"

        Image.fromarray(res["overlay_A"]).save(overlay_A_path)
        Image.fromarray(res["overlay_B"]).save(overlay_B_path)
        Image.fromarray(res["overlay_diff"]).save(overlay_diff_path)

        meta = {
            "image_id": str(image_id),
            "img_path": str(img_path),
            "checkpoint": ckpt_path.name,
            "model_type": "panderm_finetuned",
            "image_size": args.image_size,
            "device": device,
            "A_idx": int(res["A"]),
            "B_idx": int(res["B"]),
            "A_name": A_name,
            "B_name": B_name,
            "comparison_categories": [int(x) for x in res.get("comparison_categories", [res["B"]])],
            "probs": res["probs"],
            "probs_top3": res["probs_top3"],
            "method": args.method,
            "panel_path": str(panel_path),
            "overlay_A_path": str(overlay_A_path),
            "overlay_B_path": str(overlay_B_path),
            "overlay_diff_path": str(overlay_diff_path),
            "compare_mode": args.compare_mode,
            "target_layer": "model_raw.blocks[-1].norm1",
            "class_names": class_names,
            "class_preset": args.class_preset,
            "checkpoint_format": info.get("checkpoint_format"),
        }
        (out_dir / f"{image_id}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()