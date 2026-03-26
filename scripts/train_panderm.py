#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, recall_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PANDERM_CLASSIFICATION_DIR = (REPO_ROOT / "external" / "PanDerm" / "classification").resolve()
if str(PANDERM_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(PANDERM_CLASSIFICATION_DIR))

from models.builder import (get_eval_transforms,  # type: ignore
                            get_norm_constants)
from models.modeling_finetune import \
    panderm_base_patch16_224_finetune  # type: ignore

PRIMARY_CLASSES = ["NV", "MEL", "BCC", "BKL", "AKIEC", "DF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PanDerm Base on HAM -> BCN.")
    parser.add_argument("--split-dir", type=str, default=str(REPO_ROOT / "data" / "processed" / "splits"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs" / "panderm_ft"))
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=str(REPO_ROOT / "external" / "weights" / "panderm_bb_data6_checkpoint-499.pth"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-epochs", type=int, default=10)
    parser.add_argument("--ft-epochs", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=5e-4)
    parser.add_argument("--ft-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="balanced_accuracy",
        help=(
            "Comma-separated validation metrics used to select the best checkpoint in priority order. "
            "Example: 'balanced_accuracy,mcc' or 'balanced_accuracy,macro_f1,mcc'. "
            "Allowed metrics: balanced_accuracy, macro_f1, mcc."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Stop a stage early if the validation selection metric does not improve for this many epochs. Use -1 to disable.",
    )
    parser.add_argument(
        "--minority-aug-labels",
        type=str,
        default="AKIEC,DF",
        help="Comma-separated labels that should receive stronger train-time augmentation.",
    )
    parser.add_argument("--head-lr-mult", type=float, default=10.0,
                        help="Multiply LR for classification head parameters.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2,
                        help="For the first N epochs of each stage, train only the classifier head.")
    parser.add_argument("--disable-class-weights", action="store_true",
                        help="If set, do not use inverse-frequency class weights in CrossEntropyLoss.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


# Helper for parsing selection metrics argument
def parse_selection_metrics(selection_metric_arg: str) -> List[str]:
    allowed = ["balanced_accuracy", "macro_f1", "mcc"]
    metrics = [m.strip() for m in selection_metric_arg.split(",") if m.strip()]
    if not metrics:
        raise ValueError("--selection-metric must contain at least one metric name.")

    invalid = [m for m in metrics if m not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid selection metrics: {invalid}. Allowed metrics are: {allowed}"
        )

    deduped = []
    for m in metrics:
        if m not in deduped:
            deduped.append(m)
    return deduped


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved JSON: {path}")



def safe_open_image(path: str | Path) -> Image.Image:
    path = Path(path)
    with Image.open(path) as img:
        return img.convert("RGB")



class RandomRotate90:
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.choice([0, 1, 2, 3])
        if k == 0:
            return img
        return img.rotate(90 * k)


class SkinCSVClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        transform=None,
        return_metadata: bool = False,
        minority_labels: set[str] | None = None,
        minority_transform=None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_metadata = return_metadata
        self.minority_labels = minority_labels or set()
        self.minority_transform = minority_transform

        bad_labels = set(self.df["harmonized_label"].unique()) - set(self.class_to_idx.keys())
        if bad_labels:
            raise ValueError(f"Unknown labels found: {bad_labels}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = safe_open_image(row["image_path"])

        label_str = row["harmonized_label"]
        if self.minority_transform is not None and label_str in self.minority_labels:
            img = self.minority_transform(img)
        elif self.transform is not None:
            img = self.transform(img)

        y = self.class_to_idx[label_str]

        if self.return_metadata:
            meta = {
                "isic_id": row["isic_id"],
                "lesion_id": row["lesion_id"],
                "image_path": row["image_path"],
                "label_str": label_str,
                "source_dataset": row.get("source_dataset", None),
            }
            return img, y, meta

        return img, y



def compute_class_weights_from_df(df: pd.DataFrame, class_order: List[str]) -> Tuple[pd.Series, pd.Series]:
    counts = df["harmonized_label"].value_counts().reindex(class_order, fill_value=0).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return counts, weights



def make_weighted_sampler(df: pd.DataFrame, class_to_idx: Dict[str, int]) -> WeightedRandomSampler:
    labels = df["harmonized_label"].map(class_to_idx).values
    class_counts = np.bincount(labels, minlength=len(class_to_idx))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)





def build_train_transform(image_size: int):
    mean, std = get_norm_constants("imagenet")
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_minority_train_transform(image_size: int):
    mean, std = get_norm_constants("imagenet")
    return T.Compose([
        T.Resize(256),
        RandomRotate90(),
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])



def build_eval_transform(image_size: int):
    _ = image_size
    return get_eval_transforms(which_img_norm="imagenet", img_resize=256, center_crop=True)



def build_panderm_model(num_classes: int) -> nn.Module:
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

def split_head_and_backbone_params(model: nn.Module):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head") or ".head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("head") or ".head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def load_panderm_pretrained_backbone(model: nn.Module, checkpoint_path: Path) -> None:
    print(f"Loading PanDerm pretrained checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("=" * 80)
    print("PanDerm checkpoint loading report")
    print("=" * 80)
    print("Missing keys:", len(missing))
    if len(missing) > 0:
        print("First missing keys:", missing[:20])
    print("Unexpected keys:", len(unexpected))
    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:20])



def build_optimizer(model: nn.Module, lr: float, weight_decay: float, head_lr_mult: float = 1.0):
    backbone_params, head_params = split_head_and_backbone_params(model)
    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": lr, "weight_decay": weight_decay})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": lr * head_lr_mult, "weight_decay": weight_decay})
    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)



def build_loss(class_weights: pd.Series, device: str, label_smoothing: float = 0.0, use_class_weights: bool = True):
    weight_tensor = None
    if use_class_weights:
        weight_tensor = torch.tensor(class_weights.values, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)



def compute_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "per_class_recall": {class_names[i]: float(per_class_recall[i]) for i in range(len(class_names))},
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


# Helper for selection metric tuple
def get_selection_metric_tuple(metrics: dict, selection_metrics: Sequence[str]) -> Tuple[float, ...]:
    return tuple(float(metrics[m]) for m in selection_metrics)



def train_one_epoch(model, loader, optimizer, criterion, device: str, epoch: int, stage_name: str) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    pbar = tqdm(loader, desc=f"{stage_name} | epoch {epoch}", leave=False)

    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=(device == "cuda"))
        targets = targets.to(device, non_blocking=(device == "cuda"))

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        seen += bs
        running_loss += loss.item() * bs
        pbar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device: str, epoch: int, stage_name: str, class_names: List[str]) -> dict:
    model.eval()
    running_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    seen = 0
    pbar = tqdm(loader, desc=f"{stage_name} | epoch {epoch}", leave=False)

    for batch in pbar:
        if len(batch) == 3:
            imgs, targets, _ = batch
        else:
            imgs, targets = batch

        imgs = imgs.to(device, non_blocking=(device == "cuda"))
        targets = targets.to(device, non_blocking=(device == "cuda"))

        logits = model(imgs)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        bs = imgs.size(0)
        seen += bs
        running_loss += loss.item() * bs
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")

    metrics = compute_metrics(all_targets, all_preds, class_names)
    metrics["loss"] = float(running_loss / len(loader.dataset))
    return metrics



def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    stage_name: str,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    image_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "stage_name": stage_name,
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": image_size,
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")



def load_checkpoint_into_model(path: Path, model: nn.Module, map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt



def fit_stage(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    class_weights: pd.Series,
    lr: float,
    weight_decay: float,
    epochs: int,
    stage_name: str,
    checkpoint_path: Path,
    metrics_json_path: Path,
    device: str,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    image_size: int,
    label_smoothing: float = 0.0,
    save_every_epoch: bool = False,
    head_lr_mult: float = 1.0,
    freeze_backbone_epochs: int = 0,
    use_class_weights: bool = True,
    selection_metrics: Sequence[str] = ("balanced_accuracy",),
    early_stopping_patience: int = 4,
):
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, head_lr_mult=head_lr_mult)
    scheduler = build_scheduler(optimizer, epochs=epochs)
    criterion = build_loss(class_weights, device=device, label_smoothing=label_smoothing, use_class_weights=use_class_weights)

    history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric_value: Tuple[float, ...] | None = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if epoch <= freeze_backbone_epochs:
            set_backbone_trainable(model, trainable=False)
        else:
            set_backbone_trainable(model, trainable=True)
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, f"{stage_name}_train")
        val_metrics = evaluate(model, val_loader, criterion, device, epoch, f"{stage_name}_val", class_names)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_mcc": float(val_metrics["mcc"]),
            "epoch_seconds": float(epoch_time),
            "selection_metric_values": {m: float(val_metrics[m]) for m in selection_metrics},
        }
        history.append(row)

        current_metric_value = get_selection_metric_tuple(val_metrics, selection_metrics)
        current_metric_str = ", ".join([f"{m}={float(val_metrics[m]):.4f}" for m in selection_metrics])

        print(
            f"[{stage_name}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_mcc={val_metrics['mcc']:.4f} | "
            f"sel_metric={current_metric_str} | "
            f"time={epoch_time:.1f}s"
        )

        if save_every_epoch:
            ep_path = checkpoint_path.with_name(checkpoint_path.stem + f"_epoch{epoch:02d}.pt")
            save_checkpoint(ep_path, model, optimizer, scheduler, epoch, val_metrics["balanced_accuracy"], stage_name,
                            class_to_idx, idx_to_class, image_size)

        if best_metric_value is None or current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                float(current_metric_value[0]),
                stage_name,
                class_to_idx,
                idx_to_class,
                image_size,
            )
        else:
            epochs_without_improvement += 1

        if early_stopping_patience >= 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"[{stage_name}] Early stopping triggered after {epoch} epochs "
                f"without improvement in {list(selection_metrics)}."
            )
            break

    model.load_state_dict(best_model_wts)
    final_criterion = build_loss(class_weights, device=device, label_smoothing=label_smoothing, use_class_weights=use_class_weights)
    best_val_metrics = evaluate(model, val_loader, final_criterion, device, best_epoch, f"{stage_name}_best", class_names)

    out = {
        "stage_name": stage_name,
        "best_epoch": int(best_epoch),
        "selection_metrics": list(selection_metrics),
        "best_selection_metric": {
            m: float(best_val_metrics[m]) for m in selection_metrics
        },
        "best_balanced_accuracy": float(best_val_metrics["balanced_accuracy"]),
        "best_macro_f1": float(best_val_metrics["macro_f1"]),
        "best_mcc": float(best_val_metrics["mcc"]),
        "history": history,
        "best_val_metrics": best_val_metrics,
    }
    save_json(out, metrics_json_path)
    return model, out


@torch.no_grad()
def predict_with_metadata(model, loader, device: str, idx_to_class: Dict[int, str]) -> pd.DataFrame:
    model.eval()
    rows = []
    softmax = nn.Softmax(dim=1)

    for imgs, targets, metas in tqdm(loader, desc="test_predictions", leave=False):
        imgs = imgs.to(device, non_blocking=(device == "cuda"))
        targets = targets.to(device, non_blocking=(device == "cuda"))
        logits = model(imgs)
        probs = softmax(logits)
        preds = logits.argmax(dim=1)

        for i in range(imgs.size(0)):
            meta = {k: metas[k][i] for k in metas}
            row = {
                "isic_id": meta["isic_id"],
                "lesion_id": meta["lesion_id"],
                "image_path": meta["image_path"],
                "source_dataset": meta["source_dataset"],
                "y_true_idx": int(targets[i].item()),
                "y_true": idx_to_class[int(targets[i].item())],
                "y_pred_idx": int(preds[i].item()),
                "y_pred": idx_to_class[int(preds[i].item())],
            }
            for c_idx, c_name in idx_to_class.items():
                row[f"prob_{c_name}"] = float(probs[i, c_idx].item())
            rows.append(row)

    return pd.DataFrame(rows)



def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")



def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(force_cpu=args.cpu)
    selection_metrics = parse_selection_metrics(args.selection_metric)

    class_to_idx = {c: i for i, c in enumerate(PRIMARY_CLASSES)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    minority_aug_labels = {
        x.strip() for x in args.minority_aug_labels.split(",") if x.strip() in PRIMARY_CLASSES
    }

    split_dir = Path(args.split_dir)
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    fig_dir = out_dir / "figures"
    pred_dir = out_dir / "predictions"

    for p in [out_dir, ckpt_dir, metrics_dir, fig_dir, pred_dir]:
        p.mkdir(parents=True, exist_ok=True)

    train_base_df = pd.read_csv(split_dir / "train_base.csv")
    val_base_df = pd.read_csv(split_dir / "val_base.csv")
    train_bcn_ft_df = pd.read_csv(split_dir / "train_bcn_ft.csv")
    val_bcn_ft_df = pd.read_csv(split_dir / "val_bcn_ft.csv")
    test_bcn_df = pd.read_csv(split_dir / "test_bcn.csv")

    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print(f"device={device}")
    print(f"image_size={args.image_size}")
    print(f"batch_size={args.batch_size}")
    print(f"num_workers={args.num_workers}")
    print(f"base_epochs={args.base_epochs}")
    print(f"ft_epochs={args.ft_epochs}")
    print(f"use_weighted_sampler={args.use_weighted_sampler}")
    print(f"output_dir={out_dir.resolve()}")
    print(f"split_dir={split_dir.resolve()}")
    print(f"pretrained_checkpoint={Path(args.pretrained_checkpoint).resolve()}")
    print(f"head_lr_mult={args.head_lr_mult}")
    print(f"freeze_backbone_epochs={args.freeze_backbone_epochs}")
    print(f"disable_class_weights={args.disable_class_weights}")
    print(f"selection_metrics={selection_metrics}")
    print(f"early_stopping_patience={args.early_stopping_patience}")
    print(f"minority_aug_labels={sorted(minority_aug_labels)}")

    for name, df in [
        ("train_base", train_base_df),
        ("val_base", val_base_df),
        ("train_bcn_ft", train_bcn_ft_df),
        ("val_bcn_ft", val_bcn_ft_df),
        ("test_bcn", test_bcn_df),
    ]:
        print("-" * 80)
        print(name, df.shape)
        print(df["harmonized_label"].value_counts().reindex(PRIMARY_CLASSES, fill_value=0))

    train_transform = build_train_transform(args.image_size)
    minority_train_transform = build_minority_train_transform(args.image_size)
    eval_transform = build_eval_transform(args.image_size)

    train_base_ds = SkinCSVClassificationDataset(
        train_base_df,
        class_to_idx,
        transform=train_transform,
        minority_labels=minority_aug_labels,
        minority_transform=minority_train_transform,
    )
    val_base_ds = SkinCSVClassificationDataset(val_base_df, class_to_idx, transform=eval_transform)
    train_bcn_ft_ds = SkinCSVClassificationDataset(
        train_bcn_ft_df,
        class_to_idx,
        transform=train_transform,
        minority_labels=minority_aug_labels,
        minority_transform=minority_train_transform,
    )
    val_bcn_ft_ds = SkinCSVClassificationDataset(val_bcn_ft_df, class_to_idx, transform=eval_transform)
    test_bcn_ds = SkinCSVClassificationDataset(test_bcn_df, class_to_idx, transform=eval_transform, return_metadata=True)

    if args.use_weighted_sampler:
        train_base_sampler = make_weighted_sampler(train_base_df, class_to_idx)
        train_bcn_ft_sampler = make_weighted_sampler(train_bcn_ft_df, class_to_idx)
    else:
        train_base_sampler = None
        train_bcn_ft_sampler = None

    pin = device == "cuda"
    train_base_loader = DataLoader(train_base_ds, batch_size=args.batch_size, shuffle=(train_base_sampler is None),
                                   sampler=train_base_sampler, num_workers=args.num_workers, pin_memory=pin)
    val_base_loader = DataLoader(val_base_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=pin)
    train_bcn_ft_loader = DataLoader(train_bcn_ft_ds, batch_size=args.batch_size, shuffle=(train_bcn_ft_sampler is None),
                                     sampler=train_bcn_ft_sampler, num_workers=args.num_workers, pin_memory=pin)
    val_bcn_ft_loader = DataLoader(val_bcn_ft_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=pin)
    test_bcn_loader = DataLoader(test_bcn_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=pin)

    ham_counts, ham_weights = compute_class_weights_from_df(train_base_df, PRIMARY_CLASSES)
    bcn_counts, bcn_weights = compute_class_weights_from_df(train_bcn_ft_df, PRIMARY_CLASSES)
    print(pd.DataFrame({"ham_count": ham_counts, "ham_weight": ham_weights, "bcn_count": bcn_counts, "bcn_weight": bcn_weights}))

    print("=" * 80)
    print("STAGE 1: BASE TRAINING ON HAM")
    print("=" * 80)
    base_model = build_panderm_model(num_classes=len(PRIMARY_CLASSES)).to(device)
    load_panderm_pretrained_backbone(base_model, Path(args.pretrained_checkpoint))
    base_model, base_out = fit_stage(
        model=base_model,
        train_loader=train_base_loader,
        val_loader=val_base_loader,
        class_weights=ham_weights,
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        epochs=args.base_epochs,
        stage_name="base_ham",
        checkpoint_path=ckpt_dir / "panderm_base_best.pt",
        metrics_json_path=metrics_dir / "panderm_base_val_metrics.json",
        device=device,
        class_names=PRIMARY_CLASSES,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        image_size=args.image_size,
        label_smoothing=args.label_smoothing,
        save_every_epoch=args.save_every_epoch,
        head_lr_mult=args.head_lr_mult,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        use_class_weights=not args.disable_class_weights,
        selection_metrics=selection_metrics,
        early_stopping_patience=args.early_stopping_patience,
    )

    print("=" * 80)
    print("STAGE 2: FINE-TUNE ON BCN")
    print("=" * 80)
    ft_model = build_panderm_model(num_classes=len(PRIMARY_CLASSES)).to(device)
    load_checkpoint_into_model(ckpt_dir / "panderm_base_best.pt", ft_model, map_location=device)
    ft_model, ft_out = fit_stage(
        model=ft_model,
        train_loader=train_bcn_ft_loader,
        val_loader=val_bcn_ft_loader,
        class_weights=bcn_weights,
        lr=args.ft_lr,
        weight_decay=args.weight_decay,
        epochs=args.ft_epochs,
        stage_name="finetune_bcn",
        checkpoint_path=ckpt_dir / "panderm_bcn_ft_best.pt",
        metrics_json_path=metrics_dir / "panderm_bcn_val_metrics.json",
        device=device,
        class_names=PRIMARY_CLASSES,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        image_size=args.image_size,
        label_smoothing=args.label_smoothing,
        save_every_epoch=args.save_every_epoch,
        head_lr_mult=args.head_lr_mult,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        use_class_weights=not args.disable_class_weights,
        selection_metrics=selection_metrics,
        early_stopping_patience=args.early_stopping_patience,
    )

    print("=" * 80)
    print("FINAL TEST ON BCN")
    print("=" * 80)
    best_ft_model = build_panderm_model(num_classes=len(PRIMARY_CLASSES)).to(device)
    load_checkpoint_into_model(ckpt_dir / "panderm_bcn_ft_best.pt", best_ft_model, map_location=device)

    test_criterion = build_loss(bcn_weights, device=device, label_smoothing=0.0, use_class_weights=not args.disable_class_weights)
    test_metrics = evaluate(best_ft_model, test_bcn_loader, test_criterion, device, 0, "test_bcn", PRIMARY_CLASSES)
    save_json(test_metrics, metrics_dir / "panderm_test_bcn_metrics.json")

    test_cm = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(test_cm, PRIMARY_CLASSES, "PanDerm BCN Test Confusion Matrix", fig_dir / "panderm_test_bcn_confusion_matrix.png")

    test_pred_df = predict_with_metadata(best_ft_model, test_bcn_loader, device, idx_to_class)
    test_pred_df.to_csv(pred_dir / "panderm_test_bcn_predictions.csv", index=False)
    test_pred_df["correct"] = test_pred_df["y_true"] == test_pred_df["y_pred"]
    confusions = (
        test_pred_df.loc[~test_pred_df["correct"]]
        .groupby(["y_true", "y_pred"])
        .size()
        .sort_values(ascending=False)
        .to_frame("count")
        .reset_index()
    )
    confusions.to_csv(metrics_dir / "panderm_test_bcn_top_confusions.csv", index=False)

    summary = {
        "backbone": "PanDerm Base",
        "image_size": args.image_size,
        "device": device,
        "base_epochs": args.base_epochs,
        "ft_epochs": args.ft_epochs,
        "base_lr": args.base_lr,
        "ft_lr": args.ft_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "classes": PRIMARY_CLASSES,
        "selection_metrics": selection_metrics,
        "base_best_epoch": base_out["best_epoch"],
        "base_best_balanced_accuracy": base_out["best_balanced_accuracy"],
        "base_best_macro_f1": base_out["best_macro_f1"],
        "base_best_mcc": base_out["best_mcc"],
        "ft_best_epoch": ft_out["best_epoch"],
        "ft_best_balanced_accuracy": ft_out["best_balanced_accuracy"],
        "ft_best_macro_f1": ft_out["best_macro_f1"],
        "ft_best_mcc": ft_out["best_mcc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class_recall": test_metrics["per_class_recall"],
    }
    summary["test_mcc"] = test_metrics["mcc"]
    save_json(summary, metrics_dir / "panderm_experiment_summary.json")
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()