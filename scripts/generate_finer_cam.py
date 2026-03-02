# scripts/generate_finer_cam.py

# python -m scripts.generate_finer_cam \
#   --model_type isic7_effb4 \
#   --csv data/isic2018/val_gt.csv \
#   --img_dir data/isic2018/images_val \
#   --checkpoint external/weights/isic7_last_effnetb4.pth \
#   --out_dir outputs/isic7_cam/val_mel_vs_nv \
#   --image_size 380 \
#   --num_samples 5 \
#   --method gradcam \
#   --save_layercam_diff \
#   --compare_mode fixed \
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

from src.cam.diff_cam import compute_gradcam_triplet
from src.models.isic7_loader import load_isic7_effnetb4
# load_siim_efficientnet will be replaced by your own load_isic_model later
from src.models.siim_loader import load_siim_efficientnet


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
    p.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "layercam"],
                   help="CAM backend for the main triplet.")
    p.add_argument("--save_layercam_diff", action="store_true",
                   help="Additionally save a LayerCAM diff map (often sharper).")
    p.add_argument("--compare_mode", type=str, default="top2", choices=["top2", "fixed"],
                   help="How to choose classes A and B. top2 = predicted top1/top2, fixed = user-defined A/B.")
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
            enet_type="tf_efficientnet_b4.ns_jft_in1k",  # "tf_efficientnet_b4_ns" is deprecated
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
    class_to_idx = {
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

        # Compute CAM triplet (top2 or fixed)
        res = compute_gradcam_triplet(
            model=model,
            input_tensor=x,
            rgb_float=rgb_resized,
            target_layer=target_layer,
            method=args.method,
            A=A_idx,
            B=B_idx,
        )

        # Named top-3
        top3_idx = np.argsort(res.probs)[-3:][::-1]
        top3_named = ", ".join([f"{idx_to_class[i]}: {res.probs[i]:.3f}" for i in top3_idx])

        # Print what was used
        A_name = idx_to_class.get(int(res.A), str(res.A))
        B_name = idx_to_class.get(int(res.B), str(res.B))
        print(f"[info] {image_id}: A={res.A}({A_name})  B={res.B}({B_name})  top3=[{top3_named}]")

        # Build 3-panel image
        panel = np.hstack([res.overlay_A, res.overlay_B, res.overlay_diff])
        panel_img = Image.fromarray(panel)

        panel_path = out_dir / f"{image_id}_A_B_DIFF_{args.method}.png"
        panel_img.save(panel_path)

        # Optional: save LayerCAM diff map
        layercam_path = None
        if args.save_layercam_diff:
            # Use a more fine-grained layer
            from pytorch_grad_cam import LayerCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            from src.cam.diff_cam import LogitDiffTarget

            if args.model_type == "siim9":
                layer_for_layercam = model.blocks[-1][-1].conv_pwl
            else:
                # efficientnet_pytorch
                layer_for_layercam = model._conv_head

            cam_layer = LayerCAM(model=model, target_layers=[layer_for_layercam])
            cam_diff = cam_layer(input_tensor=x, targets=[LogitDiffTarget(res.A, res.B)])[0]
            vis_diff = show_cam_on_image(rgb_resized, cam_diff, use_rgb=True)
            layercam_path = out_dir / f"{image_id}_DIFF_{args.model_type}_layercam.png"
            Image.fromarray(vis_diff).save(layercam_path)

        # Save metadata
        meta = {
            "image_id": str(image_id),
            "img_path": str(img_path),
            "checkpoint": getattr(info, "checkpoint_name", ckpt_path.name),
            "model_type": args.model_type,
            "image_size": args.image_size,
            "device": device,
            "A_idx": int(res.A),
            "B_idx": int(res.B),
            "probs_top3": res.probs_top3,
            "method": args.method,
            "panel_path": str(panel_path),
            "layercam_diff_path": str(layercam_path) if layercam_path else None,
            "compare_mode": args.compare_mode,
            "A_name": A_name,
            "B_name": B_name,
        }
        (out_dir / f"{image_id}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[ok] {image_id} -> A={res.A}, B={res.B}, saved: {panel_path.name}")

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()