# src/models/isic7_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from efficientnet_pytorch import EfficientNet


@dataclass
class Isic7ModelInfo:
    arch: str
    num_classes: int
    checkpoint_name: str
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]


def load_isic7_effnetb4(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> Tuple[torch.nn.Module, Isic7ModelInfo]:
    """
    Load a 7-class EfficientNet-B4 checkpoint from jamus0702/skin-disease-classification.

    Classes (7):
      AK, BCC, BKL, DF, MEL, NV, VASC
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Label mapping from the README
    class_to_idx = {
        "AK": 0,
        "BCC": 1,
        "BKL": 2,
        "DF": 3,
        "MEL": 4,
        "NV": 5,
        "VASC": 6,
    }
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Build model
    model = EfficientNet.from_name("efficientnet-b4")
    model._fc = torch.nn.Linear(model._fc.in_features, 7)

    # Load weights
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Sometimes checkpoints have a "module." prefix; handle it safely
    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v
        state = cleaned

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Typically should be 0/0; if not, print helpful info
    if len(missing) or len(unexpected):
        print(f"[warn] load_state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing sample:", missing[:10])
        if unexpected:
            print("  unexpected sample:", unexpected[:10])

    model.eval()
    model = model.to(device)

    info = Isic7ModelInfo(
        arch="efficientnet-b4",
        num_classes=7,
        checkpoint_name=checkpoint_path.name,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
    )
    return model, info