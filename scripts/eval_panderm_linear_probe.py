#!/usr/bin/env python3
"""
PanDerm linear-probe baseline on official processed CSVs.

python -m scripts.eval_panderm_linear_probe \
  --csv-path data/HAM10000/HAM10000.csv \
  --root-path data/HAM10000/images \
  --pretrained-checkpoint external/weights/panderm_bb_data6_checkpoint-499.pth \
  --eval-split test \
  --class-names "akiec,bcc,bkl,df,mel,nv,vasc" \
  --output-json outputs/panderm_linear_probe/ham_test_metrics.json \
  --output-pred-csv outputs/panderm_linear_probe/ham_test_predictions.csv

python -m scripts.eval_panderm_linear_probe \
  --csv-path data/BCN20000/bcn20000.csv \
  --root-path data/BCN20000/images \
  --pretrained-checkpoint external/weights/panderm_bb_data6_checkpoint-499.pth \
  --eval-split test \
  --class-names "actinic keratosis,basal cell carcinoma,melanoma,nevus,seborrheic keratosis,solar lentigo,squamous cell carcinoma,dermatofibroma,vascular lesion" \
  --output-json outputs/panderm_linear_probe/bcn_test_metrics.json \
  --output-pred-csv outputs/panderm_linear_probe/bcn_test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import random
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Reduce noisy deprecation / upcoming warnings from upstream deps
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

REPO_ROOT = Path(__file__).resolve().parents[1]
PANDERM_CLASSIFICATION_DIR = (REPO_ROOT / "external" / "PanDerm" / "classification").resolve()
if str(PANDERM_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(PANDERM_CLASSIFICATION_DIR))

from models.builder import get_norm_constants  # type: ignore
from models.modeling_finetune import \
    panderm_base_patch16_224_finetune  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PanDerm linear-probe baseline.")
    parser.add_argument("--csv-path", type=str, required=True, help="Official processed CSV.")
    parser.add_argument("--root-path", type=str, required=True, help="Image root folder.")
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
        help="Which split to evaluate after fitting on train.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Optional comma-separated class names in label-id order, e.g. "
             "'akiec,bcc,bkl,df,mel,nv,vasc'",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Use sklearn class_weight for the linear head.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-pred-csv", type=str, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    if hasattr(model, "forward_features"):
        feats = model.forward_features(images)
    else:
        feats = model(images)

    if isinstance(feats, tuple):
        feats = feats[0]

    if feats.ndim == 3:
        feats = feats[:, 0] if feats.shape[1] > 1 else feats.mean(dim=1)

    if feats.ndim > 2:
        feats = feats.flatten(start_dim=1)

    return F.normalize(feats, dim=1)


@torch.no_grad()
def collect_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    model.eval()
    all_feats: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_meta: List[dict] = []

    for images, labels, meta in tqdm(loader, desc="extract_features", leave=False):
        images = images.to(device, non_blocking=(device == "cuda"))
        feats = extract_features(model, images).cpu().numpy()

        all_feats.append(feats)
        all_labels.append(labels.numpy())
        all_meta.extend([{k: meta[k][i] for k in meta} for i in range(len(labels))])

    feats_np = np.concatenate(all_feats, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return feats_np, labels_np, all_meta


def make_class_name_map(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    class_names_arg: str | None,
) -> Dict[int, str]:
    all_ids = sorted(set(y_train.tolist()) | set(y_eval.tolist()))
    if class_names_arg is None:
        return {i: str(i) for i in all_ids}

    names = [x.strip() for x in class_names_arg.split(",")]
    mapping = {}
    for i in all_ids:
        if i < len(names):
            mapping[i] = names[i]
        else:
            mapping[i] = str(i)
    return mapping


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_name_map: Dict[int, str],
) -> dict:
    label_ids = sorted(class_name_map.keys())
    target_names = [class_name_map[i] for i in label_ids]

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    per_class_recall_values = recall_score(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_recall": {
            class_name_map[label_id]: float(per_class_recall_values[idx])
            for idx, label_id in enumerate(label_ids)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.cpu)

    if device == "mps" and args.num_workers > 0:
        print("[INFO] MPS detected: setting num_workers=0 to reduce worker-side warnings and improve stability.")
        args.num_workers = 0

    csv_path = Path(args.csv_path)
    root_path = Path(args.root_path)
    checkpoint_path = Path(args.pretrained_checkpoint)

    df = pd.read_csv(csv_path, low_memory=False)

    for col in [args.split_col, args.image_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    split_series = df[args.split_col].astype(str).str.lower()
    train_df = df[split_series == "train"].copy()
    eval_df = df[split_series == args.eval_split].copy()

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
    print(f"num_workers={args.num_workers}")
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
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = build_model(num_classes=1).to(device)
    load_pretrained_backbone(model, checkpoint_path)

    train_feats, train_labels, _ = collect_features(model, train_loader, device)
    eval_feats, eval_labels, eval_meta = collect_features(model, eval_loader, device)

    class_name_map = make_class_name_map(train_labels, eval_labels, args.class_names)

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=args.max_iter,
            C=args.C,
            class_weight=None if args.class_weight == "none" else args.class_weight,
            solver="lbfgs",
            random_state=args.seed,
        )),
    ])

    probe.fit(train_feats, train_labels)
    preds = probe.predict(eval_feats)
    probs = probe.predict_proba(eval_feats)

    metrics = compute_metrics(eval_labels, preds, class_name_map)

    metrics["config"] = {
        "device": device,
        "csv_path": str(csv_path),
        "root_path": str(root_path),
        "checkpoint": str(checkpoint_path),
        "eval_split": args.eval_split,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "class_names": [class_name_map[i] for i in sorted(class_name_map.keys())],
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "max_iter": int(args.max_iter),
        "C": float(args.C),
        "seed": int(args.seed),
    }

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(json.dumps(
        {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mcc": metrics["mcc"],
            "per_class_recall": metrics["per_class_recall"],
        },
        indent=2,
    ))

    if args.output_pred_csv is not None:
        out_pred = Path(args.output_pred_csv)
        out_pred.parent.mkdir(parents=True, exist_ok=True)

        pred_df = pd.DataFrame(eval_meta)
        pred_df["y_true"] = eval_labels
        pred_df["y_pred"] = preds
        pred_df["y_true_name"] = [class_name_map[int(x)] for x in eval_labels]
        pred_df["y_pred_name"] = [class_name_map[int(x)] for x in preds]
        pred_df["correct"] = pred_df["y_true"] == pred_df["y_pred"]

        class_ids = list(probe.named_steps["clf"].classes_)
        for i, cls_id in enumerate(class_ids):
            pred_df[f"prob_{class_name_map[int(cls_id)]}"] = probs[:, i]

        pred_df.to_csv(out_pred, index=False)
        print(f"Saved predictions to: {out_pred}")

    if args.output_json is not None:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {out_json}")


if __name__ == "__main__":
    main()