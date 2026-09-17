"""
Shared utilities for CAM versus annotation overlap evaluation.

GEOMETRY. Two incompatible preprocessing pipelines exist in this project.

  squash224   Resize((224, 224)). Used by SimplePairedDermTransform during
              HA fine tuning, validation and test. Keeps the whole frame,
              distorts aspect ratio. A 600x450 image is compressed 0.373 in
              x and 0.498 in y.

  crop224     Resize(short side to 256) then CenterCrop(224). The PanDerm
              default eval transform. Preserves aspect, discards roughly
              42 percent of a 600x450 frame, almost all horizontally.

Clinician annotations were drawn on the ORIGINAL image, so the mask, the
image tensor and any display RGB must all pass through the SAME geometry.
Use transform_mask, mask_to_cam_grid_geom and transform_rgb. Do not write
local copies.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

RESIZE_SHORT = 256
CROP_SIZE = 224
CAM_GRID = 14


def resize_centercrop_mask(
    mask_bool: np.ndarray,
    resize_short: int = RESIZE_SHORT,
    crop_size: int = CROP_SIZE,
) -> tuple[np.ndarray, float]:
    """
    Apply Resize(short=resize_short) + CenterCrop(crop_size) to a binary mask.

    Returns
        cropped mask, bool, crop_size x crop_size
        fraction of original positive pixels surviving the crop
    """
    m = np.asarray(mask_bool).astype(bool)
    h, w = m.shape
    total = int(m.sum())

    scale = resize_short / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    r = cv2.resize(m.astype(np.uint8), (nw, nh),
                   interpolation=cv2.INTER_NEAREST).astype(bool)

    top = max(0, (nh - crop_size) // 2)
    left = max(0, (nw - crop_size) // 2)
    crop = r[top:top + crop_size, left:left + crop_size]

    if crop.shape != (crop_size, crop_size):
        pad = np.zeros((crop_size, crop_size), dtype=bool)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad

    scaled_total = int(r.sum())
    kept = float(crop.sum() / scaled_total) if scaled_total else np.nan
    return crop, kept


def mask_to_cam_grid(
    mask_bool: np.ndarray,
    grid: int = CAM_GRID,
    already_cropped: bool = False,
) -> np.ndarray:
    """
    Full annotation to CAM grid pipeline.

    Returns a grid x grid float array of occupancy in [0, 1].
    INTER_AREA gives the true area fraction per patch.
    """
    m = np.asarray(mask_bool).astype(bool)
    if not already_cropped:
        m, _ = resize_centercrop_mask(m)
    return cv2.resize(m.astype(np.float32), (grid, grid),
                      interpolation=cv2.INTER_AREA)

def squash_mask(
    mask_bool: np.ndarray,
    size: int = CROP_SIZE,
) -> tuple[np.ndarray, float]:
    """
    Apply Resize((size, size)) to a binary mask. Training geometry.

    Nothing is discarded, so the kept fraction is always 1.0. It is
    returned only so the signature matches resize_centercrop_mask.
    """
    m = np.asarray(mask_bool).astype(np.uint8)
    out = cv2.resize(m, (size, size),
                     interpolation=cv2.INTER_NEAREST).astype(bool)
    return out, 1.0


def transform_mask(
    mask_bool: np.ndarray,
    geometry: str,
) -> tuple[np.ndarray, float]:
    """
    Geometry aware mask transform. Drop in replacement for
    resize_centercrop_mask. Returns (mask_224, kept_fraction).
    """
    if geometry == "squash224":
        return squash_mask(mask_bool)
    if geometry == "crop224":
        return resize_centercrop_mask(mask_bool)
    raise ValueError(f"unknown geometry {geometry!r}. "
                     f"Use 'squash224' or 'crop224'.")


def mask_to_cam_grid_geom(
    mask_bool: np.ndarray,
    geometry: str,
    grid: int = CAM_GRID,
) -> np.ndarray:
    """
    Original mask straight to CAM grid occupancy in [0, 1].
    INTER_AREA gives the true area fraction per patch.
    """
    m, _ = transform_mask(mask_bool, geometry)
    return cv2.resize(m.astype(np.float32), (grid, grid),
                      interpolation=cv2.INTER_AREA)


def transform_rgb(img_rgb: np.ndarray, geometry: str) -> np.ndarray:
    """
    Display only. Produces the 224x224 view the model actually receives.
    Must match transform_mask or every overlay is spatially shifted.
    """
    if geometry == "squash224":
        return cv2.resize(img_rgb, (CROP_SIZE, CROP_SIZE),
                          interpolation=cv2.INTER_LINEAR)
    if geometry == "crop224":
        h, w = img_rgb.shape[:2]
        s = RESIZE_SHORT / min(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        im = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        t, l = (nh - CROP_SIZE) // 2, (nw - CROP_SIZE) // 2
        return im[t:t + CROP_SIZE, l:l + CROP_SIZE]
    raise ValueError(f"unknown geometry {geometry!r}")


def patch_footprint(geometry: str, orig_h: int = 450, orig_w: int = 600,
                    grid: int = CAM_GRID) -> tuple[float, float]:
    """
    How many ORIGINAL pixels one CAM patch covers. Diagnostic only.
    squash224 patches are larger horizontally, so small structures are
    harder to resolve than under crop224.
    """
    px = CROP_SIZE / grid
    if geometry == "squash224":
        return px / (CROP_SIZE / orig_h), px / (CROP_SIZE / orig_w)
    s = RESIZE_SHORT / min(orig_h, orig_w)
    return px / s, px / s

def cam_to_grid(cam_224: np.ndarray, grid: int = CAM_GRID) -> np.ndarray:
    """
    Downsample a 224x224 CAM back to its native grid.

    The saved CAM was upsampled from grid x grid, so this recovers the
    real resolution instead of pretending the interpolation is signal.
    """
    c = np.asarray(cam_224, dtype=np.float32)
    if c.shape == (grid, grid):
        return c
    return cv2.resize(c, (grid, grid), interpolation=cv2.INTER_AREA)


def norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def slugify(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def canonical_label(label_id, label_name) -> str:
    """
    Predefined labels carry a slug Label_ID.
    Custom labels typed by the clinician carry a random UUID,
    so fall back to slugifying the display name.
    """
    lid = str(label_id).strip()
    if lid and not _UUID_RE.match(lid):
        return slugify(lid)
    return slugify(label_name)


LABEL_GROUPS = {
    # melanoma specific
    "atypical_network":           "MEL",
    "atypical_dots":              "MEL",
    "atypical_streaks":           "MEL",
    "atypical_vascular_pattern":  "MEL",
    "structureless_area":         "MEL",
    "regression":                 "MEL",

    # melanoma specific, custom labels added by the clinician
    "blue_white_structureless_area":                       "MEL",
    "peppering":                                           "MEL",
    "regression_scar_like_depigmentation_and_peppering":   "MEL",
    "negative_network":                                    "MEL",

    # nevus specific
    "regular_network":            "NV",
    "homogeneous":                "NV",
    "globular_network":           "NV",

    # coarse region
    "diagnostic_region":          "COARSE",

    # pending clinical confirmation
    "small_hyperpigmented_area":  "OTHER",
    "preminent_skin_markings":    "OTHER",
    "prominent_skin_markings":    "OTHER",

    # artifact
    "artifact_ignore":            "ARTIFACT",
}


def label_group(canonical: str) -> str:
    return LABEL_GROUPS.get(str(canonical).strip(), "UNKNOWN")