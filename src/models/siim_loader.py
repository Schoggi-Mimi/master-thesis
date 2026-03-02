# src/models/siim_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import timm
import torch


@dataclass
class SiimModelInfo:
    enet_type: str
    num_classes: int
    checkpoint_name: str


def _find_num_classes_from_ckpt(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    The SIIM checkpoint stores the classifier as `module.myfc.weight` (or `myfc.weight`).
    Shape is (num_classes, in_features).
    """
    for k, v in state_dict.items():
        if "myfc.weight" in k:
            return int(v.shape[0])
    raise ValueError("Could not find 'myfc.weight' in checkpoint state_dict.")


def remap_siim_keys_to_timm(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap SIIM checkpoint keys to match timm EfficientNet naming.

    - Drop DataParallel prefix: "module."
    - Drop wrapper prefix: "enet."
    - Map classifier head: "myfc." -> "classifier."
    - Drop batchnorm tracking buffers: "num_batches_tracked"
    """
    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k2 = k

        if k2.startswith("module."):
            k2 = k2[len("module."):]

        if k2.startswith("enet."):
            k2 = k2[len("enet."):]

        if k2.startswith("myfc."):
            k2 = "classifier." + k2[len("myfc."):]

        if "num_batches_tracked" in k2:
            continue

        new_sd[k2] = v

    return new_sd


def load_siim_efficientnet(
    checkpoint_path: str | Path,
    enet_type: str = "tf_efficientnet_b4_ns",
    device: str | torch.device | None = None,
) -> Tuple[torch.nn.Module, SiimModelInfo]:
    """
    Load SIIM EfficientNet model from checkpoint into a timm model.
    Returns (model, model_info).

    This is an engineering helper. Later, you’ll replace with your ISIC2018 model loader.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint (often a raw state_dict OrderedDict)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dict/state_dict-like object.")

    # Determine num classes from checkpoint
    num_classes = _find_num_classes_from_ckpt(ckpt)

    # Remap keys to timm naming
    state_dict_timm = remap_siim_keys_to_timm(ckpt)

    # Build model + load weights
    model = timm.create_model(enet_type, pretrained=False, num_classes=num_classes)
    model.eval()

    missing, unexpected = model.load_state_dict(state_dict_timm, strict=False)
    if len(missing) or len(unexpected):
        # This should be 0/0 in your case.
        raise RuntimeError(
            f"Checkpoint load mismatch.\nMissing({len(missing)}): {missing[:10]}\n"
            f"Unexpected({len(unexpected)}): {unexpected[:10]}"
        )

    # Move to device
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = model.to(device)

    info = SiimModelInfo(
        enet_type=enet_type,
        num_classes=num_classes,
        checkpoint_name=checkpoint_path.name,
    )
    return model, info