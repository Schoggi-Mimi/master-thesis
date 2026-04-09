from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3), got {arr.shape}")

    if arr.dtype == np.uint8:
        return arr.copy()

    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def _draw_centered_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill=(0, 0, 0)) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) / 2
    y = top + (bottom - top - text_h) / 2
    draw.text((x, y), text, font=font, fill=fill)


def _load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend([
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ])

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_panel_with_subtitles(
    first_tile_line1: str,
    first_tile_line2: str,
    rgb_float: np.ndarray,
    gradcam_overlay_a: np.ndarray,
    gradcam_overlay_b: np.ndarray,
    finercam_overlay: np.ndarray,
    rollout_overlay: np.ndarray,
    gradcam_a_line1: str,
    gradcam_a_line2: str,
    gradcam_b_line1: str,
    gradcam_b_line2: str,
    finercam_line1: str,
    finercam_line2: str,
    rollout_line1: str,
    rollout_line2: str,
    scale: float = 1.35,
) -> np.ndarray:
    rgb_uint8 = _to_uint8_rgb(rgb_float)
    gradcam_a_uint8 = _to_uint8_rgb(gradcam_overlay_a)
    gradcam_b_uint8 = _to_uint8_rgb(gradcam_overlay_b)
    finercam_uint8 = _to_uint8_rgb(finercam_overlay)
    rollout_uint8 = _to_uint8_rgb(rollout_overlay)

    tiles = [
        rgb_uint8,
        gradcam_a_uint8,
        gradcam_b_uint8,
        finercam_uint8,
        rollout_uint8,
    ]
    subtitle_pairs = [
        (first_tile_line1, first_tile_line2),
        (gradcam_a_line1, gradcam_a_line2),
        (gradcam_b_line1, gradcam_b_line2),
        (finercam_line1, finercam_line2),
        (rollout_line1, rollout_line2),
    ]

    h, w, _ = tiles[0].shape
    for tile in tiles:
        if tile.shape != (h, w, 3):
            raise ValueError("All panel tiles must have the same shape.")

    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))

    gap = 16
    subtitle_h = 72
    panel_w = len(tiles) * scaled_w + (len(tiles) - 1) * gap
    panel_h = scaled_h + subtitle_h

    canvas = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    line1_font = _load_font(20, bold=True)
    line2_font = _load_font(18, bold=True)

    x = 0
    for tile, (line1, line2) in zip(tiles, subtitle_pairs):
        tile_img = Image.fromarray(tile).resize((scaled_w, scaled_h), resample=Image.Resampling.BICUBIC)
        canvas.paste(tile_img, (x, 0))

        line1_box = (x, scaled_h + 4, x + scaled_w, scaled_h + 34)
        line2_box = (x, scaled_h + 34, x + scaled_w, scaled_h + subtitle_h)
        _draw_centered_text(draw, line1_box, line1, font=line1_font)
        _draw_centered_text(draw, line2_box, line2, font=line2_font)
        x += scaled_w + gap

    return np.array(canvas)