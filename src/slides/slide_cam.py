from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import textwrap
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
_PANDERM_DIR = (REPO_ROOT / "external" / "PanDerm" / "classification").resolve()
if str(_PANDERM_DIR) not in sys.path:
    sys.path.insert(0, str(_PANDERM_DIR))

from scripts.generate_finer_cam_panderm import (  # noqa: E402
    PanDermCAMWrapper,
    build_class_maps,
    get_device,
    load_panderm_finetuned_model,
    vit_reshape_transform,
)
from src.cam.diff_cam import compute_cam_bundle  # noqa: E402

MEL_NV_CLASSES = ["MEL", "NV"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------
# Preprocessing that matches the model input exactly
# ----------------------------------------------------------------------

def preprocess_crop(img: Image.Image, size: int = 224,
                    nearest: bool = False) -> Image.Image:
    """Match SimplePairedDermTransform: direct resize, no center crop."""
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return img.resize((size, size), resample=resample)


def to_model_tensor(img_crop: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(img_crop).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return t.to(device)


# ----------------------------------------------------------------------
# Model handle
# ----------------------------------------------------------------------

@dataclass
class CamModel:
    wrapper: torch.nn.Module
    raw: torch.nn.Module
    target_layer: torch.nn.Module
    class_to_idx: dict
    idx_to_class: dict
    device: str
    tag: str


def load_cam_model(checkpoint: str | Path, pooling: str = "mean",
                   block_index: int = -1, class_names: list[str] | None = None,
                   device: str | None = None, tag: str = "") -> CamModel:
    class_names = class_names or MEL_NV_CLASSES
    device = get_device(device)
    class_to_idx, idx_to_class = build_class_maps(class_names)

    raw, info = load_panderm_finetuned_model(
        checkpoint_path=checkpoint,
        num_classes=len(class_names),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        device=device,
        checkpoint_model_type="panderm",
        pooling=pooling,
    )
    blocks = raw.blocks
    idx = block_index if block_index >= 0 else len(blocks) + block_index
    if not 0 <= idx < len(blocks):
        raise ValueError(f"block_index {block_index} invalid for {len(blocks)} blocks")

    print(f"[{tag or Path(checkpoint).stem}] {info['arch']} pooling={pooling} "
          f"target=blocks[{idx}].norm1")

    return CamModel(
        wrapper=PanDermCAMWrapper(raw),
        raw=raw,
        target_layer=blocks[idx].norm1,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        device=device,
        tag=tag or Path(checkpoint).stem,
    )


# ----------------------------------------------------------------------
# CAM computation
# ----------------------------------------------------------------------

def compute_cams(cm: CamModel, img_path: Path, a_class: str = "MEL",
                 b_class: str = "NV", method: str = "finercam",
                 alpha: float = 0.8) -> dict:
    img = Image.open(img_path).convert("RGB")
    crop = preprocess_crop(img)
    rgb = np.asarray(crop).astype(np.float32) / 255.0
    x = to_model_tensor(crop, cm.device)

    a_idx = cm.class_to_idx[a_class]
    b_idx = cm.class_to_idx[b_class]

    with torch.no_grad():
        probs = torch.softmax(cm.wrapper(x), dim=1)[0].cpu().numpy()

    res = compute_cam_bundle(
        model=cm.wrapper, input_tensor=x, rgb_float=rgb,
        target_layer=cm.target_layer, method=method,
        A=a_idx, B=b_idx, comparison_categories=[b_idx],
        reshape_transform=vit_reshape_transform, alpha=alpha,
        relprop_model=None, include_extra_maps=False,
    )
    res["rgb"] = rgb
    res["probs"] = probs
    res["pred"] = cm.idx_to_class[int(np.argmax(probs))]
    res["a_class"] = a_class
    res["b_class"] = b_class
    return res


def load_mask_crop(mask_path: Path) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    m = preprocess_crop(m, nearest=True)
    arr = np.asarray(m).astype(np.float32)
    return (arr > 127).astype(np.float32)

def cam_panel(cm: CamModel, img_path: Path, mask_path: Path | None = None,
              a_class: str = "MEL", b_class: str = "NV", alpha: float = 0.8,
              top_pct: float | None = 10.0, image_id: str = "",
              gt_label: str = "", out_path: Path | None = None):
    """Image + outline, GradCAM target, GradCAM reference, diff, FinerCAM."""
    res = compute_cams(cm, img_path, a_class=a_class, b_class=b_class,
                       method="finercam", alpha=alpha)
    rgb = res["rgb"]

    if mask_path is not None:
        mask = load_mask_crop(mask_path)
        first = draw_mask_contour(rgb, mask)
        cap0 = f"{image_id}   ground truth {gt_label}"
    else:
        first = rgb
        cap0 = image_id

    p_a = res["probs"][cm.class_to_idx[a_class]]
    p_b = res["probs"][cm.class_to_idx[b_class]]
    ok = "correct" if res["pred"] == gt_label else "wrong"

    # band = "full CAM" if top_pct is None else f"top {top_pct:.0f} percent"
    band = "full CAM" if (top_pct is None or top_pct >= 100) else f"top {top_pct:.0f} percent"
    return render_row(
        tiles=[
            first,
            heat_overlay(res["cam_gradcam"],      rgb, top_pct=top_pct),
            heat_overlay(res["cam_gradcam_B"],    rgb, top_pct=top_pct),
            heat_overlay(res["cam_gradcam_diff"], rgb, top_pct=top_pct),
            heat_overlay(res["cam_finercam"],     rgb, top_pct=top_pct),
        ],
        titles=[
            "Image + lesion boundary",
            f"GradCAM  {a_class}",
            f"GradCAM  {b_class}",
            "Naive difference",
            "FinerCAM",
        ],
        captions=[
            cap0,
            f"p({a_class}) = {p_a:.2f}   {ok}",
            f"p({b_class}) = {p_b:.2f}",
            f"max(0, {a_class} minus {b_class})",
            f"{a_class} minus {alpha} x {b_class}",
        ],
        out_path=out_path,
    )

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def cam_mass_inside(cam: np.ndarray, mask: np.ndarray) -> float:
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0, None)
    total = cam.sum()
    if total < 1e-8:
        return float("nan")
    return float((cam * mask).sum() / total)


def topk_bool(cam: np.ndarray, pct: float = 10.0) -> np.ndarray:
    a = np.asarray(cam, dtype=np.float32)
    k = max(1, int(round(pct / 100.0 * a.size)))
    flat = np.zeros(a.size, dtype=bool)
    flat[np.argpartition(a.ravel(), -k)[-k:]] = True
    return flat.reshape(a.shape)


def topk_iou(a: np.ndarray, b: np.ndarray, pct: float = 10.0) -> float:
    ma, mb = topk_bool(a, pct), topk_bool(b, pct)
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(ma, mb).sum() / union)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.corrcoef(av, bv)[0, 1])


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def heat_overlay(cam: np.ndarray, rgb: np.ndarray, alpha: float = 0.45,
                 top_pct: float | None = None) -> np.ndarray:
    """top_pct None or 100 renders the full CAM. Lower values keep only the
    hottest fraction and leave the rest as plain image."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.shape != rgb.shape[:2]:
        cam = cv2.resize(cam, rgb.shape[1::-1], interpolation=cv2.INTER_CUBIC)

    full = top_pct is None or top_pct >= 100.0

    if not full:
        thr = np.percentile(cam, 100.0 - top_pct)
        cam = np.where(cam >= thr, cam, 0.0)

    m = cam.max()
    cam_n = cam / m if m > 1e-8 else cam

    heat = cv2.applyColorMap((cam_n * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)

    if full:
        return blended

    keep = (cam_n > 0).astype(np.float32)[..., None]
    return keep * blended + (1 - keep) * rgb


def draw_mask_contour(rgb: np.ndarray, mask: np.ndarray,
                      color=(255, 255, 255), thickness: int = 2) -> np.ndarray:
    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out.astype(np.float32) / 255.0


def render_row(tiles: list[np.ndarray], titles: list[str],
               captions: list[str] | None = None, out_path: Path | None = None,
               tile_in: float = 4.2, title_size: int = 17,
               caption_size: int = 13, caption_width: int = 26,
               suptitle: str | None = None) -> None:
    n = len(tiles)
    cap_lines = 0
    wrapped = None
    if captions:
        wrapped = [textwrap.fill(c, caption_width) for c in captions]
        cap_lines = max(w.count("\n") + 1 for w in wrapped)

    height = tile_in + 0.42 + 0.30 * cap_lines
    fig, axes = plt.subplots(1, n, figsize=(tile_in * n, height))
    axes = np.atleast_1d(axes)

    for ax, tile, title in zip(axes, tiles, titles):
        ax.imshow(np.clip(tile, 0, 1))
        ax.set_title(title, fontsize=title_size, fontweight="bold", pad=10)
        ax.axis("off")

    if wrapped:
        for ax, cap in zip(axes, wrapped):
            ax.text(0.5, -0.03, cap, transform=ax.transAxes, ha="center",
                    va="top", fontsize=caption_size, color="#333333",
                    linespacing=1.35)

    if suptitle:
        fig.suptitle(suptitle, fontsize=title_size + 3, fontweight="bold", y=1.01)

    fig.subplots_adjust(wspace=0.04, top=0.88,
                        bottom=0.06 + 0.055 * cap_lines)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        print(f"saved {out_path}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Candidate ranking
# ----------------------------------------------------------------------

def rank_gap_candidates(cm: CamModel, df: pd.DataFrame, img_root: Path,
                        n: int = 40, top_pct: float = 10.0) -> pd.DataFrame:
    """High score means MEL query and NV query produce near identical maps."""
    rows = []
    for _, r in df.head(n).iterrows():
        p = img_root / str(r["image_rel_path"])
        if not p.exists():
            continue
        res = compute_cams(cm, p)
        rows.append({
            "image_id": r["image_id"],
            "gt_label": r["gt_label"],
            "pred": res["pred"],
            "correct": res["pred"] == r["gt_label"],
            "topk_iou": topk_iou(res["cam_gradcam"], res["cam_gradcam_B"], top_pct),
            "pearson": pearson(res["cam_gradcam"], res["cam_gradcam_B"]),
        })
    return pd.DataFrame(rows).sort_values("topk_iou", ascending=False).reset_index(drop=True)


def rank_alignment_candidates(cm0: CamModel, cm5: CamModel, df: pd.DataFrame,
                              img_root: Path, mask_root: Path,
                              n: int = 40) -> pd.DataFrame:
    """High delta means HA=0 leaks onto background and HA=5 focuses on lesion."""
    rows = []
    for _, r in df.head(n).iterrows():
        p = img_root / str(r["image_rel_path"])
        mp = mask_root / str(r["mask_rel_path"])
        if not (p.exists() and mp.exists()):
            continue
        mask = load_mask_crop(mp)
        gt = str(r["gt_label"])
        r0 = compute_cams(cm0, p, a_class=gt, b_class="NV" if gt == "MEL" else "MEL")
        r5 = compute_cams(cm5, p, a_class=gt, b_class="NV" if gt == "MEL" else "MEL")
        in0 = cam_mass_inside(r0["cam_gradcam"], mask)
        in5 = cam_mass_inside(r5["cam_gradcam"], mask)
        rows.append({
            "image_id": r["image_id"],
            "gt_label": gt,
            "mask_area_frac": float(mask.mean()),
            "inside_ha0": in0,
            "inside_ha5": in5,
            "delta": in5 - in0,
            "pred_ha0": r0["pred"],
            "pred_ha5": r5["pred"],
        })
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)

from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def stratified_subset(df: pd.DataFrame, n_per_class: int = 70,
                      seed: int = 0) -> pd.DataFrame:
    df = ensure_gt_label(df)
    parts = [g.sample(min(len(g), n_per_class), random_state=seed)
             for _, g in df.groupby("gt_label")]
    return pd.concat(parts).reset_index(drop=True)


def quick_classification(cm: CamModel, df: pd.DataFrame, img_root: Path) -> dict:
    ys, ps = [], []
    for _, r in df.iterrows():
        p = img_root / str(r["image_rel_path"])
        if not p.exists():
            continue
        crop = preprocess_crop(Image.open(p).convert("RGB"))
        with torch.no_grad():
            pr = torch.softmax(cm.wrapper(to_model_tensor(crop, cm.device)), 1)[0].cpu().numpy()
        ys.append(cm.class_to_idx[str(r["gt_label"])])
        ps.append(pr)
    ys, ps = np.asarray(ys), np.asarray(ps)
    return {
        "tag": cm.tag,
        "n": int(len(ys)),
        "auc": float(roc_auc_score(ys, ps[:, 1])),
        "bacc": float(balanced_accuracy_score(ys, ps.argmax(1))),
        # ys uses class_to_idx (MEL=0, NV=1). Column 1 is p(NV).
        # Positive class and score are consistent, and AUC is symmetric,
        # so this equals AUC with MEL as positive.
        "auc": float(roc_auc_score(ys, ps[:, 1])),
    }


def mean_cam_map(cm: CamModel, df: pd.DataFrame, img_root: Path) -> np.ndarray:
    acc, k = None, 0
    for _, r in df.iterrows():
        p = img_root / str(r["image_rel_path"])
        if not p.exists():
            continue
        gt = str(r["gt_label"])
        res = compute_cams(cm, p, a_class=gt, b_class="NV" if gt == "MEL" else "MEL")
        cam = np.asarray(res["cam_gradcam"], dtype=np.float32)
        m = cam.max()
        cam = cam / m if m > 1e-8 else cam
        acc = cam if acc is None else acc + cam
        k += 1
    return acc / max(k, 1)


_LABEL_ALIASES = {
    "MEL": "MEL", "MELANOMA": "MEL",
    "NV": "NV", "NEVUS": "NV", "NEVI": "NV",
}


def ensure_gt_label(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a gt_label column with values MEL or NV."""
    df = df.copy()
    if "gt_label" in df.columns:
        df["gt_label"] = df["gt_label"].astype(str).str.strip().str.upper().map(_LABEL_ALIASES)
    else:
        src = next((c for c in ["dx", "dx_norm", "label_2class"] if c in df.columns), None)
        if src is None:
            raise ValueError(f"No usable label column. Found: {df.columns.tolist()}")
        df["gt_label"] = df[src].astype(str).str.strip().str.upper().map(_LABEL_ALIASES)

    bad = df["gt_label"].isna().sum()
    if bad:
        raise ValueError(f"{bad} rows could not be mapped to MEL or NV.")
    return df

def mean_background_cam(cm: CamModel, df: pd.DataFrame, img_root: Path,
                        mask_root: Path) -> np.ndarray:
    """Mean CAM with lesion pixels removed. Reveals where background heat sits."""
    acc, k = None, 0
    for _, r in df.iterrows():
        p = img_root / str(r["image_rel_path"])
        mp = mask_root / str(r["mask_rel_path"])
        if not (p.exists() and mp.exists()):
            continue
        gt = str(r["gt_label"])
        res = compute_cams(cm, p, a_class=gt, b_class="NV" if gt == "MEL" else "MEL")
        mask = load_mask_crop(mp)
        cam = np.asarray(res["cam_gradcam"], dtype=np.float32) * (1.0 - mask)
        m = cam.max()
        cam = cam / m if m > 1e-8 else cam
        acc = cam if acc is None else acc + cam
        k += 1
    return acc / max(k, 1)

def alignment_metrics(cam: np.ndarray, mask: np.ndarray,
                      top_ratio: float = 0.10) -> dict:
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    mb = mask.astype(bool)
    if mb.sum() == 0 or (~mb).sum() == 0:
        return {}
    flat_cam, flat_mask = cam.ravel(), mb.ravel()
    k = max(1, int(round(top_ratio * flat_cam.size)))
    top_idx = np.argpartition(flat_cam, -k)[-k:]
    t10 = float(flat_mask[top_idx].mean())
    area = float(mb.mean())
    inside, outside = float(cam[mb].mean()), float(cam[~mb].mean())
    return {
        "top10_inside": t10,
        "pointing_game": float(flat_mask[int(flat_cam.argmax())]),
        "inside_mean": inside,
        "outside_mean": outside,
        "inside_outside_gap": inside - outside,
        "mask_area_fraction": area,
        "top10_minus_random": t10 - area,
    }


def sweep_blocks(checkpoint: Path, pooling: str, df: pd.DataFrame,
                 img_root: Path, mask_root: Path,
                 blocks=(-1, -2, -4, -6, -8, -10, -12),
                 tag: str = "", top_ratio: float = 0.10) -> pd.DataFrame:
    """Block sensitivity without any intermediate files."""
    cm = load_cam_model(checkpoint, pooling=pooling, block_index=-1, tag=tag)
    n_blocks = len(cm.raw.blocks)
    rows = []

    for b in blocks:
        idx = b if b >= 0 else n_blocks + b
        cm.target_layer = cm.raw.blocks[idx].norm1
        print(f"[{tag}] block {b} -> blocks[{idx}].norm1")

        for _, r in df.iterrows():
            p = img_root / str(r["image_rel_path"])
            mp = mask_root / str(r["mask_rel_path"])
            if not (p.exists() and mp.exists()):
                continue
            gt = str(r["gt_label"])
            other = "NV" if gt == "MEL" else "MEL"
            res = compute_cams(cm, p, a_class=gt, b_class=other, alpha=0.8)
            mask = load_mask_crop(mp)

            for method, cam in [("gradcam_a", res["cam_gradcam"]),
                                ("finercam", res["cam_finercam"])]:
                m = alignment_metrics(cam, mask, top_ratio)
                if not m:
                    continue
                rows.append({
                    "model": tag, "pooling": pooling, "block": b,
                    "cam_method": method, "image_id": r["image_id"],
                    "gt_label": gt, "pred": res["pred"], **m,
                })
    return pd.DataFrame(rows)

def plot_block_sensitivity(res_all: pd.DataFrame, method: str = "gradcam_a",
                           metric: str = "top10_inside",
                           order: list[str] | None = None,
                           titles: dict | None = None,
                           out_path: Path | None = None):
    """One panel per model. One line per class. Dashed line marks chance."""
    g = res_all[res_all.cam_method == method]
    models = order or sorted(g.model.unique())
    titles = titles or {}

    chance = g.groupby("gt_label")["mask_area_fraction"].mean()
    colors = {"MEL": "#c0392b", "NV": "#2471a3"}

    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, m in zip(axes, models):
        sub = g[g.model == m]
        for cls in ["MEL", "NV"]:
            s = (sub[sub.gt_label == cls]
                 .groupby("block")[metric].mean().sort_index())
            ax.plot(range(len(s)), s.values, marker="o", lw=2.2, ms=7,
                    color=colors[cls], label=cls, zorder=3)
            ax.axhline(chance[cls], ls="--", lw=1.2, color=colors[cls],
                       alpha=0.45, zorder=1)
            ax.set_xticks(range(len(s)))
            ax.set_xticklabels([str(b) for b in s.index])

        ax.set_title(titles.get(m, m), fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("transformer block", fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, zorder=0)

    axes[0].set_ylabel("top 10 percent CAM inside lesion", fontsize=11)
    axes[0].legend(fontsize=11, loc="upper left", framealpha=0.9)
    axes[-1].text(0.98, 0.03, "dashed = chance", transform=axes[-1].transAxes,
                  ha="right", fontsize=9, color="#555555")

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        print(f"saved {out_path}")
    plt.close(fig)


def plot_worst_class(res_all: pd.DataFrame, method: str = "gradcam_a",
                     metric: str = "top10_inside",
                     order: list[str] | None = None,
                     labels: dict | None = None,
                     out_path: Path | None = None):
    """Weaker class per block. The single number that decides block choice."""
    g = res_all[res_all.cam_method == method]
    models = order or sorted(g.model.unique())
    labels = labels or {}
    colors = ["#7f8c8d", "#2471a3", "#27ae60", "#c0392b"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, m in enumerate(models):
        s = (g[g.model == m]
             .groupby(["gt_label", "block"])[metric].mean()
             .groupby("block").min().sort_index())
        ax.plot(range(len(s)), s.values, marker="o", lw=2.4, ms=8,
                color=colors[i % len(colors)], label=labels.get(m, m), zorder=3)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels([str(b) for b in s.index])

    ax.set_xlabel("transformer block", fontsize=11)
    ax.set_ylabel("weaker class score", fontsize=11)
    ax.set_title("Localisation on the weaker of the two classes",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        print(f"saved {out_path}")
    plt.close(fig)

def predict_frame(cm: CamModel, df: pd.DataFrame, img_root: Path,
                  batch: int = 32) -> pd.DataFrame:
    """Predictions plus any metadata columns present in df."""
    keep = [c for c in ["image_id", "gt_label", "dx_type", "split"] if c in df.columns]
    rows, buf, meta = [], [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            pr = torch.softmax(cm.wrapper(torch.cat(buf)), 1).cpu().numpy()
        for m, p in zip(meta, pr):
            rows.append({**m, "p_mel": float(p[cm.class_to_idx["MEL"]])})
        buf.clear()
        meta.clear()

    for _, r in df.iterrows():
        p = img_root / str(r["image_rel_path"])
        if not p.exists():
            continue
        buf.append(to_model_tensor(preprocess_crop(Image.open(p).convert("RGB")),
                                   cm.device))
        meta.append({c: r[c] for c in keep})
        if len(buf) == batch:
            flush()
    flush()

    out = pd.DataFrame(rows)
    out["pred"] = np.where(out.p_mel >= 0.5, "MEL", "NV")
    out["correct"] = out.pred == out.gt_label
    return out


def performance_panel(pred: pd.DataFrame, title: str = "",
                      out_path: Path | None = None):
    """Confusion matrix, ROC curve, metric summary. MEL is the positive class."""
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

    y = (pred.gt_label == "MEL").astype(int).values
    s = pred.p_mel.values
    yh = (pred.pred == "MEL").astype(int).values

    cm_ = confusion_matrix(y, yh, labels=[1, 0])   # rows and cols: MEL then NV
    tp, fn, fp, tn = cm_[0, 0], cm_[0, 1], cm_[1, 0], cm_[1, 1]
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    auc = roc_auc_score(y, s)
    bacc = 0.5 * (sens + spec)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4),
                           gridspec_kw={"width_ratios": [1, 1, 0.85]})

    ax[0].imshow(cm_, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f"{cm_[i, j]}", ha="center", va="center",
                       fontsize=20, fontweight="bold",
                       color="white" if cm_[i, j] > cm_.max() / 2 else "black")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["MEL", "NV"], fontsize=12)
    ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["MEL", "NV"], fontsize=12)
    ax[0].set_xlabel("predicted", fontsize=12)
    ax[0].set_ylabel("ground truth", fontsize=12)
    ax[0].set_title("Confusion matrix", fontsize=14, fontweight="bold")

    fpr, tpr, _ = roc_curve(y, s)
    ax[1].plot(fpr, tpr, lw=2.6, color="#2471a3")
    ax[1].plot([0, 1], [0, 1], ls="--", lw=1.2, color="#999999")
    ax[1].scatter([1 - spec], [sens], s=90, zorder=5, color="#c0392b",
                  label="operating point")
    ax[1].set_xlabel("1 minus specificity", fontsize=12)
    ax[1].set_ylabel("sensitivity", fontsize=12)
    ax[1].set_title(f"ROC   AUC {auc:.3f}", fontsize=14, fontweight="bold")
    ax[1].legend(fontsize=10, loc="lower right")
    ax[1].grid(alpha=0.25)

    txt = (f"Melanoma as positive class\n\n"
           f"Sensitivity   {sens:.3f}\n"
           f"Specificity   {spec:.3f}\n"
           f"Precision     {ppv:.3f}\n"
           f"Balanced acc  {bacc:.3f}\n"
           f"AUC           {auc:.3f}\n\n"
           f"Missed melanoma   {fn} of {tp + fn}\n"
           f"False alarms      {fp} of {tn + fp}")
    ax[2].text(0.02, 0.97, txt, va="top", ha="left", fontsize=13,
               family="monospace", transform=ax[2].transAxes)
    ax[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        print(f"saved {out_path}")
    plt.close(fig)

    return {"sens": sens, "spec": spec, "ppv": ppv, "auc": auc,
            "bacc": bacc, "tp": tp, "fn": fn, "fp": fp, "tn": tn}

def auc_of(d: pd.DataFrame) -> float:
    """AUC with MEL as positive class. NaN if only one class present."""
    y = (d.gt_label == "MEL").astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, d.p_mel))


def subset_metrics(pred: pd.DataFrame, name: str = "") -> dict:
    y = (pred.gt_label == "MEL").astype(int).values
    yh = (pred.pred == "MEL").astype(int).values
    tp = int(((y == 1) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum())
    fp = int(((y == 0) & (yh == 1)).sum())
    tn = int(((y == 0) & (yh == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"subset": name, "n": len(pred), "n_mel": tp + fn, "n_nv": tn + fp,
            "sens": round(sens, 3), "spec": round(spec, 3),
            "bacc": round(0.5 * (sens + spec), 3), "fn": fn, "fp": fp}

def source_panel(pred_m: pd.DataFrame, out_path: Path | None = None):
    """Per-source prevalence, prediction rate, AUC and score distribution."""
    srcs = (pred_m.groupby("dataset")
                  .apply(lambda d: pd.Series({
                      "n": len(d),
                      "n_mel": int((d.gt_label == "MEL").sum()),
                      "prev": (d.gt_label == "MEL").mean(),
                      "rate": (d.pred == "MEL").mean(),
                      "auc": auc_of(d),
                  }), include_groups=False)
                  .sort_values("prev"))

    fig, ax = plt.subplots(1, 2, figsize=(14, 0.95 * len(srcs) + 3.2),
                           gridspec_kw={"width_ratios": [1, 1.3]})
    y = np.arange(len(srcs))[::-1]
    h = 0.34

    ax[0].barh(y + h / 2, srcs.prev, height=h, color="#7f8c8d",
               label="true melanoma prevalence", zorder=3)
    ax[0].barh(y - h / 2, srcs.rate, height=h, color="#c0392b",
               label="predicted melanoma rate", zorder=3)
    for yi, r in zip(y, srcs.itertuples()):
        ax[0].text(r.prev + 0.015, yi + h / 2, f"{r.prev:.1%}", va="center", fontsize=10)
        ax[0].text(r.rate + 0.015, yi - h / 2, f"{r.rate:.1%}", va="center",
                   fontsize=10, color="#c0392b")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{i}\nn={int(r.n)}, MEL={int(r.n_mel)}"
                           for i, r in srcs.iterrows()], fontsize=10)
    ax[0].set_xlim(0, 1.05)
    ax[0].set_xlabel("fraction", fontsize=12)
    ax[0].set_title("Prediction rate tracks the site prior",
                    fontsize=14, fontweight="bold")
    ax[0].legend(fontsize=10, loc="lower right")
    ax[0].grid(axis="x", alpha=0.25, zorder=0)

    rng = np.random.default_rng(0)
    for yi, src in zip(y, srcs.index):
        d = pred_m[pred_m.dataset == src]
        for cls, c, mk, s, a in [("NV", "#2471a3", "o", 26, 0.32),
                                 ("MEL", "#c0392b", "D", 60, 0.95)]:
            g = d[d.gt_label == cls]
            if len(g) == 0:
                continue
            ax[1].scatter(g.p_mel, yi + rng.uniform(-0.22, 0.22, len(g)),
                          s=s, c=c, marker=mk, alpha=a, linewidths=0,
                          zorder=3 if cls == "NV" else 4, label=cls)
        ax[1].text(1.01, yi, f"AUC {srcs.loc[src, 'auc']:.2f}",
                   va="center", fontsize=10, color="#333333")
    ax[1].axvline(0.5, ls="--", lw=1.4, color="#111111", zorder=5)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].set_xlim(-0.02, 1.02)
    ax[1].set_xlabel("predicted melanoma probability", fontsize=12)
    ax[1].set_title("Score distribution by site", fontsize=14, fontweight="bold")
    ax[1].grid(axis="x", alpha=0.25, zorder=0)
    hh, ll = ax[1].get_legend_handles_labels()
    seen = dict(zip(ll, hh))
    ax[1].legend(seen.values(), seen.keys(), fontsize=11,
                 loc="upper center", ncol=2, framealpha=0.9)

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        print(f"saved {out_path}")
    plt.close(fig)
    return srcs


def plot_auc_paradox(pooled: float, per_source: pd.Series,
                     out_path: Path | None = None):
    """Pooled AUC against every within-source AUC."""
    s = per_source.sort_values()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.barh(range(len(s)), s.values, color="#2471a3", alpha=0.85,
            height=0.6, zorder=3)
    ax.axvline(pooled, ls="--", lw=2.2, color="#c0392b", zorder=5,
               label=f"pooled AUC {pooled:.3f}")
    for i, v in enumerate(s.values):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=11)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=11)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC", fontsize=12)
    ax.set_title("Every within-site AUC falls below the pooled AUC",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        print(f"saved {out_path}")
    plt.close(fig)