"""
This script tests the Chefer et al. relevance propagation implementation on a PanDerm model.

python -m scripts.test_panderm_chefer_relprop \
  --image-path data/HAM10000/images/ISIC_0026150.jpg \
  --checkpoint external/weights/checkpoint-best-ham.pth \
  --model panderm_base_patch16_224_finetune \
  --nb-classes 7 \
  --output-dir outputs/panderm_relprop_test
  
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

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
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting .* in registry with .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\._factory is deprecated.*",
    category=FutureWarning,
)

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from timm.models import create_model

REPO_ROOT = Path(__file__).resolve().parents[1]
PANDERM_CLASSIFICATION_ROOT = REPO_ROOT / "external" / "PanDerm" / "classification"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PANDERM_CLASSIFICATION_ROOT))

from external.PanDerm.classification.models.modeling_finetune import \
    panderm_base_patch16_224_finetune  # type: ignore
from external.PanDerm.classification.models.modeling_finetune_relprop import \
    build_panderm_relprop_from_model

# `models.*` is imported from `external/PanDerm/classification` via the
# `sys.path` insertion above so PanDerm's internal absolute imports resolve.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="panderm_base_patch16_224_finetune",
        help="Model builder name. Default matches the PanDerm fine-tuning architecture.",
    )
    parser.add_argument("--nb-classes", type=int, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--target-idx", type=int, default=None)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/panderm_relprop_test")
    return parser.parse_args()


def build_panderm_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "panderm_base_patch16_224_finetune":
        return panderm_base_patch16_224_finetune(
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

    return create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
    )


def load_image(path: str, input_size: int):
    pil = Image.open(path).convert("RGB")
    rgb = np.array(pil)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    tensor = transform(pil).unsqueeze(0)
    return rgb, tensor


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = 0.35 * rgb.astype(np.float32) + 0.65 * heatmap_color.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = build_panderm_model(args.model, args.nb_classes)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    print("Base model load_state_dict:")
    print("  missing   =", missing)
    print("  unexpected=", unexpected)

    base_model.eval().to(device)
    rel_model = build_panderm_relprop_from_model(base_model).to(device)
    rel_model.eval()

    rgb, x = load_image(args.image_path, args.input_size)
    x = x.to(device)

    with torch.no_grad():
        logits_base = base_model(x)

    # Run rel_model without torch.no_grad() so attention tensors can keep grad hooks
    rel_model.zero_grad(set_to_none=True)
    logits_rel = rel_model(x)

    max_abs_diff = (logits_base - logits_rel).abs().max().item()
    print(f"Forward equivalence check | max abs diff = {max_abs_diff:.8f}")

    # Clear anything from the equivalence-check forward and rebuild a clean graph
    rel_model.zero_grad(set_to_none=True)
    logits = rel_model(x)
    if args.target_idx is None:
        target_idx = int(logits.argmax(dim=-1).item())
    else:
        target_idx = int(args.target_idx)

    one_hot = np.zeros((1, logits.size(-1)), dtype=np.float32)
    one_hot[0, target_idx] = 1.0
    one_hot_vector = torch.tensor(one_hot, device=device)

    rel_model.zero_grad()
    target_score = torch.sum(one_hot_vector * logits)
    target_score.backward(retain_graph=True)

    attribution = rel_model.relprop(
        one_hot_vector,
        method="transformer_attribution",
        start_layer=args.start_layer,
        alpha=1,
    )

    attr = attribution[0].detach().cpu().numpy()
    side = int(math.sqrt(attr.shape[0]))
    heatmap_small = attr.reshape(side, side)
    heatmap_small = np.maximum(heatmap_small, 0)
    if heatmap_small.max() > heatmap_small.min():
        heatmap_small = (heatmap_small - heatmap_small.min()) / (heatmap_small.max() - heatmap_small.min())
    else:
        heatmap_small = np.zeros_like(heatmap_small)

    heatmap = cv2.resize(
        heatmap_small,
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    overlay = overlay_heatmap(rgb, heatmap)

    stem = Path(args.image_path).stem
    np.save(output_dir / f"{stem}_heatmap.npy", heatmap)
    Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")

    print(f"Saved heatmap to: {output_dir / f'{stem}_heatmap.npy'}")
    print(f"Saved overlay  to: {output_dir / f'{stem}_overlay.png'}")
    print(f"Target class index: {target_idx}")


if __name__ == "__main__":
    main()