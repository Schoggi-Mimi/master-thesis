# scripts/generate_finer_cam.py

# python -m scripts.generate_finer_cam \
#   --model_type isic7_effb4 \
#   --csv data/isic2018/val_gt.csv \
#   --img_dir data/isic2018/images_val \
#   --checkpoint external/weights/isic7_last_effnetb4.pth \
#   --out_dir outputs/isic7_cam/val_mel_vs_nv \
#   --num_samples 5 \
#   --method gradcam \
#   --save_layercam_diff \
#   --compare_mode gt_pair \
#   --A MEL \
#   --B NV

#   --compare_mode top2

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from src.cam.diff_cam import compute_cam_triplet
from src.models.isic7_loader import load_isic7_effnetb4
# load_siim_efficientnet will be replaced by your own load_isic_model later
from src.models.siim_loader import load_siim_efficientnet
from src.utils.vis_panel import make_panel_with_subtitles


def parse_args():
    p = argparse.ArgumentParser(description="Engineering test: generate (Diff) CAMs for images.")
    p.add_argument("--model_type", type=str, default="siim9", choices=["siim9", "isic7_effb4"], help="Which model loader/checkpoint format to use.")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV with column 'image' (no .jpg).")
    p.add_argument("--img_dir", type=str, required=True, help="Folder containing JPG images.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SIIM .pth checkpoint.")
    p.add_argument("--out_dir", type=str, default="outputs/debug_cam", help="Output folder.")
    p.add_argument("--image_size", type=int, default=None, help="Resize to (image_size, image_size). If omitted, uses model default.")
    p.add_argument("--num_samples", type=int, default=10, help="How many images to process (from top of CSV).")
    p.add_argument("--device", type=str, default=None, help="cpu or mps (default: auto).")
    p.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "layercam", "finercam"],
                   help="CAM backend for the main triplet.")
    p.add_argument("--compare_mode", type=str, default="top2", choices=["top2", "fixed", "gt_pair"],
        help=(
            "How to choose classes A and B.\n"
            "top2 = predicted top1/top2.\n"
            "fixed = user-defined --A/--B used as A/B exactly.\n"
            "gt_pair = use --A/--B as the pair; per image set A=gt_label and B=the other one."
        ),
    )
    p.add_argument("--A", type=str, default=None,
                   help="Fixed target class name (e.g., MEL). Used when compare_mode=fixed.")
    p.add_argument("--B", type=str, default=None,
                   help="Fixed comparison class name (e.g., NV). Used when compare_mode=fixed.")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    img_dir = Path(args.img_dir)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = args.device
    if device is None:
        device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

    # Load model
    # Default resize based on checkpoint training setup
    if args.image_size is None:
        if args.model_type == "isic7_effb4":
            args.image_size = 380
        else:
            args.image_size = 640

    # Load model + pick CAM target layer
    if args.model_type == "siim9":
        model, info = load_siim_efficientnet(
            checkpoint_path=ckpt_path,
            enet_type="tf_efficientnet_b4.ns_jft_in1k", # "tf_efficientnet_b4_ns" is deprecated
            device=device,
        )
        target_layer = model.conv_head

    elif args.model_type == "isic7_effb4":
        model, info = load_isic7_effnetb4(
            checkpoint_path=ckpt_path,
            device=device,
        )
        # EfficientNet from efficientnet_pytorch has _conv_head
        target_layer = model._conv_head

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
    # Class name mapping (used for printing and fixed MEL vs NV)
    # For https://huggingface.co/jamus0702/skin-disease-classification, the checkpoint uses: AK, BCC, BKL, DF, MEL, NV, VASC
    # ISIC2018 uses AKIEC instead of AK; here we treat AKIEC as AK for convenience.
    class_to_idx = { # return mapping form 'info' dict if available
        "AK": 0, "AKIEC": 0,
        "BCC": 1,
        "BKL": 2,
        "DF": 3,
        "MEL": 4,
        "NV": 5,
        "VASC": 6,
    }
    idx_to_class = {v: k for k, v in class_to_idx.items() if k in ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]}
    
    # Preprocess
    preprocess = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    # Load CSV
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError(f"CSV must contain column 'image'. Found: {df.columns.tolist()}")

    df = df.head(args.num_samples)


    for idx, row in df.iterrows():
        image_id = row["image"]
        img_path = img_dir / f"{image_id}.jpg"
        if not img_path.exists():
            print(f"[skip] missing image: {img_path}")
            continue

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Model input tensor
        x = preprocess(img).unsqueeze(0).to(device)

        # RGB float for overlay (must match model input resolution)
        rgb = np.array(img).astype(np.float32) / 255.0
        rgb_resized = cv2.resize(rgb, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)

        # Choose A and B
        A_idx, B_idx = None, None
        if args.compare_mode == "fixed":
            if args.A is None or args.B is None:
                raise ValueError("compare_mode=fixed requires --A and --B, e.g. --A MEL --B NV")
            if args.A not in class_to_idx or args.B not in class_to_idx:
                raise ValueError(f"Unknown class name. Allowed: {sorted(set(class_to_idx.keys()))}")
            A_idx = class_to_idx[args.A]
            B_idx = class_to_idx[args.B]

        elif args.compare_mode == "gt_pair":
            # Use --A/--B as the CLOSE PAIR, but pick direction per-image using gt_label
            if args.A is None or args.B is None:
                raise ValueError("compare_mode=gt_pair requires --A and --B (the close pair), e.g. --A MEL --B NV")
            if args.A not in class_to_idx or args.B not in class_to_idx:
                raise ValueError(f"Unknown class name. Allowed: {sorted(set(class_to_idx.keys()))}")
            if "gt_label" not in df.columns:
                raise ValueError("compare_mode=gt_pair requires the CSV to contain a 'gt_label' column.")
            gt = row["gt_label"]
            if gt not in [args.A, args.B]:
                # Not part of the requested pair → skip (or you can choose to fallback to top2)
                print(f"[skip] {image_id}: gt_label={gt} not in pair ({args.A},{args.B})")
                continue
            # A = GT, B = the other class in the pair
            if gt == args.A:
                A_idx = class_to_idx[args.A]
                B_idx = class_to_idx[args.B]
            else:
                A_idx = class_to_idx[args.B]
                B_idx = class_to_idx[args.A]
        # else: top2 mode keeps A_idx/B_idx = None, and diff_cam picks top2

        # Compute CAM triplet (top2 or fixed)
        res = compute_cam_triplet(
            model=model,
            input_tensor=x,
            rgb_float=rgb_resized,
            target_layer=target_layer,
            method=args.method,
            A=A_idx,
            B=B_idx,
        )

        # Temporary: print CAM stats for debugging (check to if normalization is needed)
        # print("cam_A min/max:", float(np.min(res["cam_A"])), float(np.max(res["cam_A"])))
        # print("cam_B min/max:", float(np.min(res["cam_B"])), float(np.max(res["cam_B"])))
        # print("cam_diff min/max:", float(np.min(res["cam_diff"])), float(np.max(res["cam_diff"])))

        # Named top-3
        top3_idx = np.argsort(res["probs"])[-3:][::-1]
        top3_named = ", ".join([f"{idx_to_class[i]}: {res['probs'][i]:.3f}" for i in top3_idx])

        # Print what was used
        A_name = idx_to_class.get(int(res["A"]), str(res["A"]))
        B_name = idx_to_class.get(int(res["B"]), str(res["B"]))
        print(f"[info] {image_id}: A={res['A']}({A_name})  B={res['B']}({B_name})  top3=[{top3_named}]")

        # Build panel image
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

        # Save metadata
        meta = {
            "image_id": str(image_id),
            "img_path": str(img_path),
            # "checkpoint": getattr(info, "checkpoint_name", ckpt_path.name),
            "checkpoint": ckpt_path.name,
            "model_type": args.model_type,
            "image_size": args.image_size,
            "device": device,
            "A_idx": int(res["A"]),
            "B_idx": int(res["B"]),
            "probs_top3": res["probs_top3"],
            "method": args.method,
            "panel_path": str(panel_path),
            "compare_mode": args.compare_mode,
            "A_name": A_name,
            "B_name": B_name,
        }
        (out_dir / f"{image_id}_meta.json").write_text(json.dumps(meta, indent=2))
        # print(f"[ok] {image_id} -> A={res['A']}, B={res['B']}, saved: {panel_path.name}")

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()