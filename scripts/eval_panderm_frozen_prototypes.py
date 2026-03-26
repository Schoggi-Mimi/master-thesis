#!/usr/bin/env python3

"""
Evaluates a frozen PanDerm backbone with cosine similarity to train prototypes.

python -m scripts.eval_panderm_frozen_prototypes \
  --csv-path data/HAM10000/HAM10000.csv \
  --root-path data/HAM10000/images \
  --pretrained-checkpoint external/weights/panderm_bb_data6_checkpoint-499.pth \
  --eval-split test \
  --output-json outputs/panderm_zero_shot/ham_test_metrics.json

python -m scripts.eval_panderm_frozen_prototypes \
  --csv-path data/BCN20000/bcn20000.csv \
  --root-path data/BCN20000/images \
  --pretrained-checkpoint external/weights/panderm_bb_data6_checkpoint-499.pth \
  --eval-split test \
  --output-json outputs/panderm_zero_shot/bcn_test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.registry is deprecated, please import via timm\.models",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\.",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PANDERM_CLASSIFICATION_DIR = (REPO_ROOT / "external" / "PanDerm" / "classification").resolve()
if str(PANDERM_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(PANDERM_CLASSIFICATION_DIR))

from models.builder import get_norm_constants  # type: ignore
from models.modeling_finetune import panderm_base_patch16_224_finetune  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen PanDerm prototype baseline.")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Official processed CSV, e.g. data/HAM10000/HAM_clean.csv",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        required=True,
        help="Image root folder, e.g. data/HAM10000/images",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=str(REPO_ROOT / "external" / "weights" / "panderm_bb_data6_checkpoint-499.pth"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--image-col", type=str, default="image")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate against train prototypes.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save metrics json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_open_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def build_eval_transform(image_size: int):
    mean, std = get_norm_constants("imagenet")
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class CSVImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_path: Path, image_col: str, label_col: str, transform):
        self.df = df.reset_index(drop=True).copy()
        self.root_path = root_path
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.root_path / str(row[self.image_col])
        img = safe_open_image(image_path)
        img = self.transform(img)
        label = int(row[self.label_col])

        meta = {
            "image": str(row[self.image_col]),
            "label": label,
        }
        return img, label, meta


def build_model(num_classes: int = 1) -> torch.nn.Module:
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


def load_pretrained_backbone(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("=" * 80)
    print("PanDerm checkpoint loading report")
    print("=" * 80)
    print("Missing keys:", len(missing))
    if missing:
        print("First missing keys:", missing[:20])
    print("Unexpected keys:", len(unexpected))
    if unexpected:
        print("First unexpected keys:", unexpected[:20])


@torch.no_grad()
def extract_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Tries several common ways to get frozen embeddings from the model.
    """
    if hasattr(model, "forward_features"):
        feats = model.forward_features(images)
    else:
        feats = model(images)

    if isinstance(feats, tuple):
        feats = feats[0]

    if feats.ndim == 3:
        # [B, tokens, C] -> use CLS token if present, otherwise mean pool
        feats = feats[:, 0] if feats.shape[1] > 1 else feats.mean(dim=1)

    if feats.ndim > 2:
        feats = feats.flatten(start_dim=1)

    return F.normalize(feats, dim=1)


@torch.no_grad()
def collect_features(model, loader, device: str) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    model.eval()
    all_feats = []
    all_labels = []
    all_meta = []

    for images, labels, meta in tqdm(loader, desc="extract_features", leave=False):
        images = images.to(device, non_blocking=(device == "cuda"))
        feats = extract_features(model, images)
        all_feats.append(feats.cpu())
        all_labels.append(labels.cpu())
        all_meta.extend([{k: meta[k][i] for k in meta} for i in range(len(labels))])

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return feats, labels, all_meta


def build_class_prototypes(train_feats: torch.Tensor, train_labels: torch.Tensor) -> Dict[int, torch.Tensor]:
    prototypes = {}
    for cls in sorted(train_labels.unique().tolist()):
        cls_mask = train_labels == cls
        proto = train_feats[cls_mask].mean(dim=0, keepdim=True)
        proto = F.normalize(proto, dim=1).squeeze(0)
        prototypes[int(cls)] = proto
    return prototypes


def predict_by_cosine(feats: torch.Tensor, prototypes: Dict[int, torch.Tensor]) -> np.ndarray:
    class_ids = sorted(prototypes.keys())
    proto_mat = torch.stack([prototypes[c] for c in class_ids], dim=0)  # [K, D]
    sims = feats @ proto_mat.T
    pred_idx = sims.argmax(dim=1).cpu().numpy()
    preds = np.array([class_ids[i] for i in pred_idx], dtype=int)
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.cpu)

    effective_num_workers = args.num_workers
    if device == "mps" and effective_num_workers > 0:
        print("[INFO] MPS detected: setting num_workers=0 to reduce repeated worker-side warnings and improve stability.")
        effective_num_workers = 0

    csv_path = Path(args.csv_path)
    root_path = Path(args.root_path)
    checkpoint_path = Path(args.pretrained_checkpoint)

    df = pd.read_csv(csv_path, low_memory=False)
    if args.split_col not in df.columns:
        raise ValueError(f"Missing split column: {args.split_col}")
    if args.image_col not in df.columns:
        raise ValueError(f"Missing image column: {args.image_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    train_df = df[df[args.split_col].astype(str).str.lower() == "train"].copy()
    eval_df = df[df[args.split_col].astype(str).str.lower() == args.eval_split].copy()

    if len(train_df) == 0:
        raise ValueError("No train rows found.")
    if len(eval_df) == 0:
        raise ValueError(f"No rows found for eval split='{args.eval_split}'.")

    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print(f"device={device}")
    print(f"csv_path={csv_path}")
    print(f"root_path={root_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"eval_split={args.eval_split}")
    print(f"num_workers={effective_num_workers}")
    print(f"train_rows={len(train_df)}")
    print(f"eval_rows={len(eval_df)}")
    print("\nTrain label counts:")
    print(train_df[args.label_col].value_counts().sort_index())
    print("\nEval label counts:")
    print(eval_df[args.label_col].value_counts().sort_index())

    transform = build_eval_transform(args.image_size)
    train_ds = CSVImageDataset(train_df, root_path, args.image_col, args.label_col, transform)
    eval_ds = CSVImageDataset(eval_df, root_path, args.image_col, args.label_col, transform)

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=effective_num_workers, pin_memory=pin)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=effective_num_workers, pin_memory=pin)

    model = build_model(num_classes=1).to(device)
    load_pretrained_backbone(model, checkpoint_path)

    train_feats, train_labels, _ = collect_features(model, train_loader, device)
    eval_feats, eval_labels, eval_meta = collect_features(model, eval_loader, device)

    prototypes = build_class_prototypes(train_feats, train_labels)
    preds = predict_by_cosine(eval_feats, prototypes)
    y_true = eval_labels.numpy()

    metrics = compute_metrics(y_true, preds)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(json.dumps(
        {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mcc": metrics["mcc"],
        },
        indent=2,
    ))

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()