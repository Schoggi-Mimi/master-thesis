#!/usr/bin/env python3
"""
PanDerm frozen-backbone + MLP-head baseline.

This baseline:
- loads official processed CSV
- uses ONLY the PanDerm backbone as a frozen feature extractor
- trains a small MLP classifier on extracted train embeddings
- uses val split for model selection / early stopping
- evaluates the best checkpoint on the requested eval split (val or test)

Example HAM:
python -m scripts.eval_panderm_mlp_probe \
  --csv-path data/HAM10000/HAM10000.csv \
  --root-path data/HAM10000/images \
  --pretrained-checkpoint external/weights/panderm_bb_data6_checkpoint-499.pth \
  --eval-split test \
  --class-names "akiec,bcc,bkl,df,mel,nv,vasc" \
  --use-balanced-sampler \
  --use-class-weights \
  --normalize-features \
  --output-json outputs/panderm_mlp_probe/ham_test_metrics.json \
  --output-pred-csv outputs/panderm_mlp_probe/ham_test_predictions.csv

Example BCN:
python -m scripts.eval_panderm_mlp_probe \
  --csv-path data/BCN20000/bcn20000.csv \
  --root-path data/BCN20000/images \
  --pretrained-checkpoint external/weights/panderm_bb_data6_checkpoint-499.pth \
  --eval-split test \
  --class-names "actinic keratosis,basal cell carcinoma,melanoma,nevus,seborrheic keratosis,solar lentigo,squamous cell carcinoma,dermatofibroma,vascular lesion" \
  --use-balanced-sampler \
  --use-class-weights \
  --normalize-features \
  --output-json outputs/panderm_mlp_probe/bcn_test_metrics.json \
  --output-pred-csv outputs/panderm_mlp_probe/bcn_test_predictions.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, recall_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import (DataLoader, Dataset, TensorDataset,
                              WeightedRandomSampler)
from tqdm.auto import tqdm

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
    parser = argparse.ArgumentParser(description="PanDerm frozen-backbone + MLP-head baseline.")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--root-path", type=str, required=True)
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
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--eval-split", type=str, default="test", choices=["val", "test"])

    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Comma-separated class names in label-id order.",
    )

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument(
        "--use-balanced-sampler",
        action="store_true",
        help="Use weighted sampler on train embeddings.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class-weighted CE loss on train embeddings.",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Apply StandardScaler to train/val/eval features before MLP training.",
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_model(num_classes: int = 1) -> nn.Module:
    return panderm_base_patch16_224_finetune(
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


def load_pretrained_backbone(model: nn.Module, checkpoint_path: Path) -> None:
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
def extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
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
    model: nn.Module,
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
    y_val: np.ndarray,
    y_eval: np.ndarray,
    class_names_arg: str | None,
) -> Dict[int, str]:
    all_ids = sorted(set(y_train.tolist()) | set(y_val.tolist()) | set(y_eval.tolist()))
    if class_names_arg is None:
        return {i: str(i) for i in all_ids}

    names = [x.strip() for x in class_names_arg.split(",")]
    mapping = {}
    for i in all_ids:
        mapping[i] = names[i] if i < len(names) else str(i)
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


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_embedding_loader(
    feats: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: str,
    use_balanced_sampler: bool = False,
) -> DataLoader:
    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(x, y)

    if use_balanced_sampler:
        class_counts = np.bincount(labels)
        class_weights = np.zeros_like(class_counts, dtype=np.float64)
        nonzero = class_counts > 0
        class_weights[nonzero] = 1.0 / class_counts[nonzero]
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            pin_memory=(device == "cuda"),
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device == "cuda"),
    )


def fit_mlp_probe(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    device: str,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    batch_size: int,
    use_balanced_sampler: bool,
    use_class_weights: bool,
    seed: int,
) -> Tuple[nn.Module, dict]:
    torch.manual_seed(seed)

    train_loader = make_embedding_loader(
        train_feats,
        train_labels,
        batch_size=batch_size,
        shuffle=not use_balanced_sampler,
        device=device,
        use_balanced_sampler=use_balanced_sampler,
    )
    val_loader = make_embedding_loader(
        val_feats,
        val_labels,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        use_balanced_sampler=False,
    )

    model = MLPHead(
        in_dim=train_feats.shape[1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    class_weights_tensor = None
    if use_class_weights:
        class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
        weights = np.zeros(num_classes, dtype=np.float32)
        nz = class_counts > 0
        weights[nz] = class_counts.sum() / (num_classes * class_counts[nz])
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_epoch = -1
    best_score = -np.inf
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device == "cuda"))
            yb = yb.to(device, non_blocking=(device == "cuda"))

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_logits_all = []
        val_y_all = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=(device == "cuda"))
                logits = model(xb)
                val_logits_all.append(logits.cpu())
                val_y_all.append(yb)

        val_logits = torch.cat(val_logits_all, dim=0)
        val_y = torch.cat(val_y_all, dim=0).numpy()
        val_pred = val_logits.argmax(dim=1).numpy()

        val_bal_acc = balanced_accuracy_score(val_y, val_pred)
        val_macro_f1 = f1_score(val_y, val_pred, average="macro", zero_division=0)
        val_mcc = matthews_corrcoef(val_y, val_pred)

        score = val_bal_acc

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_balanced_accuracy": float(val_bal_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_mcc": float(val_mcc),
        })

        print(
            f"[mlp_probe] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"val_bal_acc={val_bal_acc:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f} | "
            f"val_mcc={val_mcc:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[mlp_probe] Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)

    training_summary = {
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_score),
        "history": history,
    }
    return model, training_summary


@torch.no_grad()
def predict_mlp(
    model: nn.Module,
    feats: np.ndarray,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.tensor(feats, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    model.eval()
    logits_all = []

    for (xb,) in loader:
        xb = xb.to(device, non_blocking=(device == "cuda"))
        logits = model(xb)
        logits_all.append(logits.cpu())

    logits = torch.cat(logits_all, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    return preds, probs


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.cpu)

    if device == "mps" and args.num_workers > 0:
        print("[INFO] MPS detected: setting num_workers=0 for stability.")
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
    val_df = df[split_series == args.val_split].copy()
    eval_df = df[split_series == args.eval_split].copy()

    if len(train_df) == 0:
        raise ValueError("No train rows found.")
    if len(val_df) == 0:
        raise ValueError(f"No rows found for val split='{args.val_split}'.")
    if len(eval_df) == 0:
        raise ValueError(f"No rows found for eval split='{args.eval_split}'.")

    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print(f"device={device}")
    print(f"csv_path={csv_path}")
    print(f"root_path={root_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"val_split={args.val_split}")
    print(f"eval_split={args.eval_split}")
    print(f"num_workers={args.num_workers}")
    print(f"train_rows={len(train_df)}")
    print(f"val_rows={len(val_df)}")
    print(f"eval_rows={len(eval_df)}")

    print("\nTrain label counts:")
    print(train_df[args.label_col].value_counts().sort_index())
    print("\nVal label counts:")
    print(val_df[args.label_col].value_counts().sort_index())
    print("\nEval label counts:")
    print(eval_df[args.label_col].value_counts().sort_index())

    transform = build_eval_transform(args.image_size)

    pin = device == "cuda"

    train_loader = DataLoader(
        CSVImageDataset(train_df, root_path, args.image_col, args.label_col, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        CSVImageDataset(val_df, root_path, args.image_col, args.label_col, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    eval_loader = DataLoader(
        CSVImageDataset(eval_df, root_path, args.image_col, args.label_col, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    backbone = build_model(num_classes=1).to(device)
    load_pretrained_backbone(backbone, checkpoint_path)
    backbone.eval()

    train_feats, train_labels, _ = collect_features(backbone, train_loader, device)
    val_feats, val_labels, _ = collect_features(backbone, val_loader, device)
    eval_feats, eval_labels, eval_meta = collect_features(backbone, eval_loader, device)

    if args.normalize_features:
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        val_feats = scaler.transform(val_feats)
        eval_feats = scaler.transform(eval_feats)

    num_classes = int(max(train_labels.max(), val_labels.max(), eval_labels.max()) + 1)
    class_name_map = make_class_name_map(train_labels, val_labels, eval_labels, args.class_names)

    mlp_model, training_summary = fit_mlp_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        val_feats=val_feats,
        val_labels=val_labels,
        num_classes=num_classes,
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        use_balanced_sampler=args.use_balanced_sampler,
        use_class_weights=args.use_class_weights,
        seed=args.seed,
    )

    preds, probs = predict_mlp(
        model=mlp_model,
        feats=eval_feats,
        batch_size=args.batch_size,
        device=device,
    )

    metrics = compute_metrics(eval_labels, preds, class_name_map)
    metrics["config"] = {
        "device": device,
        "csv_path": str(csv_path),
        "root_path": str(root_path),
        "checkpoint": str(checkpoint_path),
        "val_split": args.val_split,
        "eval_split": args.eval_split,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "eval_rows": int(len(eval_df)),
        "class_names": [class_name_map[i] for i in sorted(class_name_map.keys())],
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "use_balanced_sampler": bool(args.use_balanced_sampler),
        "use_class_weights": bool(args.use_class_weights),
        "normalize_features": bool(args.normalize_features),
        "seed": int(args.seed),
    }
    metrics["training_summary"] = training_summary

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
            "best_epoch": training_summary["best_epoch"],
            "best_val_balanced_accuracy": training_summary["best_val_balanced_accuracy"],
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

        for class_id in range(probs.shape[1]):
            col_name = class_name_map.get(class_id, str(class_id))
            pred_df[f"prob_{col_name}"] = probs[:, class_id]

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