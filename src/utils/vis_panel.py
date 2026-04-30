from __future__ import annotations

from typing import Optional, Tuple

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


# def make_panel_with_subtitles_old(
#     first_tile_line1: str,
#     first_tile_line2: str,
#     rgb_float: np.ndarray,
#     gradcam_overlay_a: np.ndarray,
#     gradcam_overlay_b: np.ndarray,
#     gradcam_diff_overlay: np.ndarray,
#     finercam_overlay: np.ndarray,
#     rollout_overlay: np.ndarray,
#     chefer_overlay: np.ndarray,
#     gradcam_a_line1: str,
#     gradcam_a_line2: str,
#     gradcam_b_line1: str,
#     gradcam_b_line2: str,
#     gradcam_diff_line1: str,
#     gradcam_diff_line2: str,
#     finercam_line1: str,
#     finercam_line2: str,
#     rollout_line1: str,
#     rollout_line2: str,
#     chefer_line1: str,
#     chefer_line2: str,
#     scale: float = 1.35,
# ) -> np.ndarray:
#     rgb_uint8 = _to_uint8_rgb(rgb_float)
#     gradcam_a_uint8 = _to_uint8_rgb(gradcam_overlay_a)
#     gradcam_b_uint8 = _to_uint8_rgb(gradcam_overlay_b)
#     gradcam_diff_uint8 = _to_uint8_rgb(gradcam_diff_overlay)
#     finercam_uint8 = _to_uint8_rgb(finercam_overlay)
#     rollout_uint8 = _to_uint8_rgb(rollout_overlay)
#     chefer_uint8 = _to_uint8_rgb(chefer_overlay)

#     tiles = [
#         rgb_uint8,
#         gradcam_a_uint8,
#         gradcam_b_uint8,
#         gradcam_diff_uint8,
#         finercam_uint8,
#         rollout_uint8,
#         chefer_uint8,
#     ]
#     subtitle_pairs = [
#         (first_tile_line1, first_tile_line2),
#         (gradcam_a_line1, gradcam_a_line2),
#         (gradcam_b_line1, gradcam_b_line2),
#         (gradcam_diff_line1, gradcam_diff_line2),
#         (finercam_line1, finercam_line2),
#         (rollout_line1, rollout_line2),
#         (chefer_line1, chefer_line2),
#     ]

#     h, w, _ = tiles[0].shape
#     for tile in tiles:
#         if tile.shape != (h, w, 3):
#             raise ValueError("All panel tiles must have the same shape.")

#     scaled_w = max(1, int(round(w * scale)))
#     scaled_h = max(1, int(round(h * scale)))

#     gap = 16
#     subtitle_h = 72
#     panel_w = len(tiles) * scaled_w + (len(tiles) - 1) * gap
#     panel_h = scaled_h + subtitle_h

#     canvas = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
#     draw = ImageDraw.Draw(canvas)
#     line1_font = _load_font(20, bold=True)
#     line2_font = _load_font(18, bold=True)

#     x = 0
#     for tile, (line1, line2) in zip(tiles, subtitle_pairs):
#         tile_img = Image.fromarray(tile).resize((scaled_w, scaled_h), resample=Image.Resampling.BICUBIC)
#         canvas.paste(tile_img, (x, 0))

#         line1_box = (x, scaled_h + 4, x + scaled_w, scaled_h + 34)
#         line2_box = (x, scaled_h + 34, x + scaled_w, scaled_h + subtitle_h)
#         _draw_centered_text(draw, line1_box, line1, font=line1_font)
#         _draw_centered_text(draw, line2_box, line2, font=line2_font)
#         x += scaled_w + gap

#     return np.array(canvas)

# def make_panel_with_subtitles(
#     first_tile_line1: str,
#     first_tile_line2: str,
#     rgb_float: np.ndarray,
#     gradcam_overlay_a: np.ndarray,
#     gradcam_overlay_b: np.ndarray,
#     gradcam_diff_overlay: np.ndarray,
#     finercam_overlay: np.ndarray,
#     rollout_overlay: np.ndarray,
#     chefer_overlay_a: np.ndarray,
#     chefer_overlay_b: np.ndarray,
#     chefer_diff_overlay: np.ndarray,
#     gradcam_a_line1: str,
#     gradcam_a_line2: str,
#     gradcam_b_line1: str,
#     gradcam_b_line2: str,
#     gradcam_diff_line1: str,
#     gradcam_diff_line2: str,
#     finercam_line1: str,
#     finercam_line2: str,
#     rollout_line1: str,
#     rollout_line2: str,
#     chefer_a_line1: str,
#     chefer_a_line2: str,
#     chefer_b_line1: str,
#     chefer_b_line2: str,
#     chefer_diff_line1: str,
#     chefer_diff_line2: str,
#     scale: float = 1.35,
# ) -> np.ndarray:
#     rgb_uint8 = _to_uint8_rgb(rgb_float)
#     gradcam_a_uint8 = _to_uint8_rgb(gradcam_overlay_a)
#     gradcam_b_uint8 = _to_uint8_rgb(gradcam_overlay_b)
#     gradcam_diff_uint8 = _to_uint8_rgb(gradcam_diff_overlay)
#     finercam_uint8 = _to_uint8_rgb(finercam_overlay)
#     rollout_uint8 = _to_uint8_rgb(rollout_overlay)
#     chefer_a_uint8 = _to_uint8_rgb(chefer_overlay_a)
#     chefer_b_uint8 = _to_uint8_rgb(chefer_overlay_b)
#     chefer_diff_uint8 = _to_uint8_rgb(chefer_diff_overlay)

#     row1_tiles = [
#         rgb_uint8,
#         gradcam_a_uint8,
#         gradcam_b_uint8,
#         gradcam_diff_uint8,
#         finercam_uint8,
#     ]
#     row1_pairs = [
#         (first_tile_line1, first_tile_line2),
#         (gradcam_a_line1, gradcam_a_line2),
#         (gradcam_b_line1, gradcam_b_line2),
#         (gradcam_diff_line1, gradcam_diff_line2),
#         (finercam_line1, finercam_line2),
#     ]

#     row2_tiles = [
#         rollout_uint8,
#         chefer_a_uint8,
#         chefer_b_uint8,
#         chefer_diff_uint8,
#     ]
#     row2_pairs = [
#         (rollout_line1, rollout_line2),
#         (chefer_a_line1, chefer_a_line2),
#         (chefer_b_line1, chefer_b_line2),
#         (chefer_diff_line1, chefer_diff_line2),
#     ]

#     h, w, _ = row1_tiles[0].shape
#     for tile in row1_tiles + row2_tiles:
#         if tile.shape != (h, w, 3):
#             raise ValueError("All panel tiles must have the same shape.")

#     scaled_w = max(1, int(round(w * scale)))
#     scaled_h = max(1, int(round(h * scale)))

#     gap_x = 16
#     gap_y = 24
#     subtitle_h = 72
#     row1_w = len(row1_tiles) * scaled_w + (len(row1_tiles) - 1) * gap_x
#     row2_w = len(row2_tiles) * scaled_w + (len(row2_tiles) - 1) * gap_x
#     panel_w = max(row1_w, row2_w)
#     row_h = scaled_h + subtitle_h
#     panel_h = row_h * 2 + gap_y

#     canvas = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
#     draw = ImageDraw.Draw(canvas)
#     line1_font = _load_font(20, bold=True)
#     line2_font = _load_font(18, bold=True)

#     def paste_row(tiles, subtitle_pairs, y_offset: int) -> None:
#         row_w = len(tiles) * scaled_w + (len(tiles) - 1) * gap_x
#         x = (panel_w - row_w) // 2
#         for tile, (line1, line2) in zip(tiles, subtitle_pairs):
#             tile_img = Image.fromarray(tile).resize((scaled_w, scaled_h), resample=Image.Resampling.BICUBIC)
#             canvas.paste(tile_img, (x, y_offset))

#             line1_box = (x, y_offset + scaled_h + 4, x + scaled_w, y_offset + scaled_h + 34)
#             line2_box = (x, y_offset + scaled_h + 34, x + scaled_w, y_offset + scaled_h + subtitle_h)
#             _draw_centered_text(draw, line1_box, line1, font=line1_font)
#             _draw_centered_text(draw, line2_box, line2, font=line2_font)
#             x += scaled_w + gap_x

#     paste_row(row1_tiles, row1_pairs, 0)
#     paste_row(row2_tiles, row2_pairs, row_h + gap_y)

#     return np.array(canvas)

def make_panel_with_subtitles(
    first_tile_line1: str,
    first_tile_line2: str,
    rgb_float: np.ndarray,
    gt_mask_overlay: Optional[np.ndarray] = None,
    gt_mask_binary: Optional[np.ndarray] = None,
    seg_gate_overlay: Optional[np.ndarray] = None,
    gradcam_overlay_a: np.ndarray = None,
    gradcam_overlay_b: np.ndarray = None,
    gradcam_diff_overlay: np.ndarray = None,
    finercam_overlay: np.ndarray = None,
    gate_weighted_finercam_overlay: Optional[np.ndarray] = None,
    gate_weighted_gradcam_a_overlay: Optional[np.ndarray] = None,
    gate_weighted_gradcam_b_overlay: Optional[np.ndarray] = None,
    gate_weighted_map_diff_overlay: Optional[np.ndarray] = None,
    rollout_overlay: np.ndarray = None,
    chefer_overlay_a: np.ndarray = None,
    chefer_overlay_b: np.ndarray = None,
    chefer_diff_overlay: np.ndarray = None,
    relprop_chefer_overlay_a: Optional[np.ndarray] = None,
    relprop_chefer_overlay_b: Optional[np.ndarray] = None,
    relprop_chefer_diff_overlay: Optional[np.ndarray] = None,
    gradcam_a_line1: str = "GradCAM",
    gradcam_a_line2: str = "",
    gradcam_b_line1: str = "GradCAM",
    gradcam_b_line2: str = "",
    gradcam_diff_line1: str = "Map Diff",
    gradcam_diff_line2: str = "",
    finercam_line1: str = "FinerCAM",
    finercam_line2: str = "",
    gate_weighted_finercam_line1: str = "Gate weighted FinerCAM",
    gate_weighted_finercam_line2: str = "FinerCAM × gate",
    gate_weighted_gradcam_a_line1: str = "GradCAM × gate",
    gate_weighted_gradcam_a_line2: str = "",
    gate_weighted_gradcam_b_line1: str = "GradCAM × gate",
    gate_weighted_gradcam_b_line2: str = "",
    gate_weighted_map_diff_line1: str = "Map Diff × gate",
    gate_weighted_map_diff_line2: str = "",
    rollout_line1: str = "Rollout",
    rollout_line2: str = "",
    chefer_a_line1: str = "Chefer-style",
    chefer_a_line2: str = "",
    chefer_b_line1: str = "Chefer-style",
    chefer_b_line2: str = "",
    chefer_diff_line1: str = "Chefer Map Diff",
    chefer_diff_line2: str = "",
    relprop_chefer_a_line1: str = "Chefer relprop",
    relprop_chefer_a_line2: str = "",
    relprop_chefer_b_line1: str = "Chefer relprop",
    relprop_chefer_b_line2: str = "",
    relprop_chefer_diff_line1: str = "Relprop Map Diff",
    relprop_chefer_diff_line2: str = "",
    seg_gate_line1: str = "Predicted Seg Gate",
    seg_gate_line2: str = "auxiliary head",
    scale: float = 1.35,
    show_extra_row: bool = False,
    show_relprop_row: bool = False,
    panel_items: Optional[list[str]] = None,
) -> np.ndarray:
    rgb_uint8 = _to_uint8_rgb(rgb_float)

    gt_mask_uint8 = None
    if gt_mask_overlay is not None:
        gt_mask_uint8 = _to_uint8_rgb(gt_mask_overlay)

    gt_mask_binary_uint8 = None
    if gt_mask_binary is not None:
        gt_mask_binary_uint8 = _to_uint8_rgb(gt_mask_binary)

    gradcam_a_uint8 = _to_uint8_rgb(gradcam_overlay_a)
    gradcam_b_uint8 = _to_uint8_rgb(gradcam_overlay_b)
    gradcam_diff_uint8 = _to_uint8_rgb(gradcam_diff_overlay)
    finercam_uint8 = _to_uint8_rgb(finercam_overlay)

    seg_gate_uint8 = None
    if seg_gate_overlay is not None:
        seg_gate_uint8 = _to_uint8_rgb(seg_gate_overlay)

    gate_weighted_finercam_uint8 = None
    if gate_weighted_finercam_overlay is not None:
        gate_weighted_finercam_uint8 = _to_uint8_rgb(gate_weighted_finercam_overlay)

    gate_weighted_gradcam_a_uint8 = None
    if gate_weighted_gradcam_a_overlay is not None:
        gate_weighted_gradcam_a_uint8 = _to_uint8_rgb(gate_weighted_gradcam_a_overlay)

    gate_weighted_gradcam_b_uint8 = None
    if gate_weighted_gradcam_b_overlay is not None:
        gate_weighted_gradcam_b_uint8 = _to_uint8_rgb(gate_weighted_gradcam_b_overlay)

    gate_weighted_map_diff_uint8 = None
    if gate_weighted_map_diff_overlay is not None:
        gate_weighted_map_diff_uint8 = _to_uint8_rgb(gate_weighted_map_diff_overlay)
    if panel_items is None:
        panel_items = ["rgb", "gradcam_a", "gradcam_b", "map_diff", "finercam"]
    rollout_uint8 = None
    chefer_a_uint8 = None
    chefer_b_uint8 = None
    chefer_diff_uint8 = None

    if show_extra_row:
        if (
            rollout_overlay is None
            or chefer_overlay_a is None
            or chefer_overlay_b is None
            or chefer_diff_overlay is None
        ):
            raise ValueError("show_extra_row=True requires rollout and Chefer overlays.")

        rollout_uint8 = _to_uint8_rgb(rollout_overlay)
        chefer_a_uint8 = _to_uint8_rgb(chefer_overlay_a)
        chefer_b_uint8 = _to_uint8_rgb(chefer_overlay_b)
        chefer_diff_uint8 = _to_uint8_rgb(chefer_diff_overlay)

    # row1_tiles = [rgb_uint8]
    # row1_pairs = [(first_tile_line1, first_tile_line2)]

    # if seg_gate_uint8 is not None:
    #     row1_tiles.append(seg_gate_uint8)
    #     row1_pairs.append((seg_gate_line1, seg_gate_line2))

    # row1_tiles.extend([
    #     gradcam_a_uint8,
    #     gradcam_b_uint8,
    #     gradcam_diff_uint8,
    #     finercam_uint8,
    # ])
    # row1_pairs.extend([
    #     (gradcam_a_line1, gradcam_a_line2),
    #     (gradcam_b_line1, gradcam_b_line2),
    #     (gradcam_diff_line1, gradcam_diff_line2),
    #     (finercam_line1, finercam_line2),
    # ])
    # if gate_weighted_finercam_uint8 is not None:
    #     row1_tiles.append(gate_weighted_finercam_uint8)
    #     row1_pairs.append((gate_weighted_finercam_line1, gate_weighted_finercam_line2))

    tile_lookup = {
        "rgb": (rgb_uint8, first_tile_line1, first_tile_line2),
        "rgb_gt_mask": (gt_mask_uint8, first_tile_line1, first_tile_line2),
        "gt_mask": (gt_mask_binary_uint8, "GT mask", "binary mask"),
        "seg_gate": (seg_gate_uint8, seg_gate_line1, seg_gate_line2),
        "gradcam_a": (gradcam_a_uint8, gradcam_a_line1, gradcam_a_line2),
        "gradcam_b": (gradcam_b_uint8, gradcam_b_line1, gradcam_b_line2),
        "map_diff": (gradcam_diff_uint8, gradcam_diff_line1, gradcam_diff_line2),
        "finercam": (finercam_uint8, finercam_line1, finercam_line2),
        "gate_weighted_gradcam_a": (
            gate_weighted_gradcam_a_uint8,
            gate_weighted_gradcam_a_line1,
            gate_weighted_gradcam_a_line2,
        ),
        "gate_weighted_gradcam_b": (
            gate_weighted_gradcam_b_uint8,
            gate_weighted_gradcam_b_line1,
            gate_weighted_gradcam_b_line2,
        ),
        "gate_weighted_map_diff": (
            gate_weighted_map_diff_uint8,
            gate_weighted_map_diff_line1,
            gate_weighted_map_diff_line2,
        ),
        "gate_weighted_finercam": (
            gate_weighted_finercam_uint8,
            gate_weighted_finercam_line1,
            gate_weighted_finercam_line2,
        ),
    }

    row1_tiles = []
    row1_pairs = []

    for item in panel_items:
        tile, line1, line2 = tile_lookup[item]

        if tile is None:
            raise ValueError(
                f"Panel item '{item}' was requested, but its image is None. "
                "Check that you loaded the correct checkpoint type and computed the required map."
            )

        row1_tiles.append(tile)
        row1_pairs.append((line1, line2))

    rows_tiles = [row1_tiles]
    rows_pairs = [row1_pairs]

    if show_extra_row:
        row2_tiles = [
            rollout_uint8,
            chefer_a_uint8,
            chefer_b_uint8,
            chefer_diff_uint8,
        ]
        row2_pairs = [
            (rollout_line1, rollout_line2),
            (chefer_a_line1, chefer_a_line2),
            (chefer_b_line1, chefer_b_line2),
            (chefer_diff_line1, chefer_diff_line2),
        ]

        rows_tiles.append(row2_tiles)
        rows_pairs.append(row2_pairs)

    if show_relprop_row:
        if not show_extra_row:
            raise ValueError("show_relprop_row=True requires show_extra_row=True.")
        if (
            relprop_chefer_overlay_a is None
            or relprop_chefer_overlay_b is None
            or relprop_chefer_diff_overlay is None
        ):
            raise ValueError("show_relprop_row=True requires all relprop Chefer overlays.")

        relprop_chefer_a_uint8 = _to_uint8_rgb(relprop_chefer_overlay_a)
        relprop_chefer_b_uint8 = _to_uint8_rgb(relprop_chefer_overlay_b)
        relprop_chefer_diff_uint8 = _to_uint8_rgb(relprop_chefer_diff_overlay)

        row3_tiles = [
            relprop_chefer_a_uint8,
            relprop_chefer_b_uint8,
            relprop_chefer_diff_uint8,
        ]
        row3_pairs = [
            (relprop_chefer_a_line1, relprop_chefer_a_line2),
            (relprop_chefer_b_line1, relprop_chefer_b_line2),
            (relprop_chefer_diff_line1, relprop_chefer_diff_line2),
        ]
        rows_tiles.append(row3_tiles)
        rows_pairs.append(row3_pairs)

    h, w, _ = row1_tiles[0].shape
    for row in rows_tiles:
        for tile in row:
            if tile.shape != (h, w, 3):
                raise ValueError("All panel tiles must have the same shape.")

    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))

    gap_x = 16
    gap_y = 24
    subtitle_h = 72
    row_widths = [len(row) * scaled_w + (len(row) - 1) * gap_x for row in rows_tiles]
    panel_w = max(row_widths)
    row_h = scaled_h + subtitle_h
    panel_h = row_h * len(rows_tiles) + gap_y * (len(rows_tiles) - 1)

    canvas = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    line1_font = _load_font(20, bold=True)
    line2_font = _load_font(18, bold=True)

    def paste_row(tiles, subtitle_pairs, y_offset: int) -> None:
        row_w = len(tiles) * scaled_w + (len(tiles) - 1) * gap_x
        x = (panel_w - row_w) // 2
        for tile, (line1, line2) in zip(tiles, subtitle_pairs):
            tile_img = Image.fromarray(tile).resize((scaled_w, scaled_h), resample=Image.Resampling.BICUBIC)
            canvas.paste(tile_img, (x, y_offset))

            line1_box = (x, y_offset + scaled_h + 4, x + scaled_w, y_offset + scaled_h + 34)
            line2_box = (x, y_offset + scaled_h + 34, x + scaled_w, y_offset + scaled_h + subtitle_h)
            _draw_centered_text(draw, line1_box, line1, font=line1_font)
            _draw_centered_text(draw, line2_box, line2, font=line2_font)
            x += scaled_w + gap_x

    for row_idx, (tiles, subtitle_pairs) in enumerate(zip(rows_tiles, rows_pairs)):
        y_offset = row_idx * (row_h + gap_y)
        paste_row(tiles, subtitle_pairs, y_offset)

    return np.array(canvas)