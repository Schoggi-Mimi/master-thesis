# src/utils/vis_panel.py
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def _draw_centered_text(img: np.ndarray, text: str, *, font_scale: float = 0.7) -> None:
    """Draw centered black text on a white bar."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = img.shape[:2]
    x = max(10, (w - tw) // 2)
    y = (h + th) // 2
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def make_panel_with_subtitles(
    *,
    image_id: str,
    rgb_float: np.ndarray,            # (H,W,3) float in [0,1]
    overlay_A: np.ndarray,            # (H,W,3) uint8
    overlay_B: np.ndarray,            # (H,W,3) uint8
    overlay_diff: np.ndarray,         # (H,W,3) uint8
    method: str,
    A_name: str,
    B_name: str,
    A_prob: float,
    B_prob: float,
    gt_label: Optional[str] = None,
    include_rgb: bool = True,
    subtitle_h: int = 48,
) -> np.ndarray:
    """
    Returns a panel with subtitles UNDER each column (not drawn on top of the images).
    Layout:
      Row 1: RGB | A | B | DIFF   (or A | B | DIFF if include_rgb=False)
      Row 2: subtitles under each column
    """
    # RGB as uint8
    rgb_uint8 = (rgb_float * 255.0).clip(0, 255).astype(np.uint8)

    cols: List[np.ndarray]
    subtitles: List[str]

    if include_rgb:
        cols = [rgb_uint8, overlay_A, overlay_B, overlay_diff]
        subtitles = [
            f"{image_id} GT={gt_label or '?'}",
            f"{A_name}({A_prob:.3f})",
            f"{B_name}({B_prob:.3f})",
            f"method={method}",
        ]
    else:
        cols = [overlay_A, overlay_B, overlay_diff]
        subtitles = [
            f"{A_name}({A_prob:.3f})",
            f"{B_name}({B_prob:.3f})",
            f"method={method}",
        ]

    panel_top = np.hstack(cols)

    # Build subtitle bars (one per column, same width as each column)
    h, w_total, _ = panel_top.shape
    col_w = cols[0].shape[1]
    bars = []
    for s in subtitles:
        bar = np.full((subtitle_h, col_w, 3), 255, dtype=np.uint8)
        _draw_centered_text(bar, s, font_scale=0.7)
        bars.append(bar)

    panel_bottom = np.hstack(bars)
    out = np.vstack([panel_top, panel_bottom])
    return out