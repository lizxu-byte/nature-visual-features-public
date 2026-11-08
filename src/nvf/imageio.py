"""Image loading and conversion helpers shared by all feature modules."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array of shape (H, W, 3)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image; grayscale input is replicated to three channels.

    Float input is assumed to be in [0, 1].
    """
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(np.round(img.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W) or (H, W, 3) image, got shape {image.shape}")
    return img


def to_gray(image: np.ndarray) -> np.ndarray:
    """Luminance in [0, 1] as float64 (ITU-R BT.601 weights, as in OpenCV).

    2-D input is returned as float in [0, 1] (uint8 is divided by 255).
    """
    img = np.asarray(image)
    if img.ndim == 2:
        gray = img.astype(np.float64)
        return gray / 255.0 if img.dtype == np.uint8 else gray
    rgb = as_rgb_uint8(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0


def resize_max_side(image: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    """Downscale so that the longer side is at most ``max_side`` pixels.

    Images are never upscaled. Returns the (possibly) resized image and the
    scale factor applied (1.0 if unchanged). ``max_side`` of None or 0 disables
    resizing.
    """
    h, w = image.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return image, 1.0
    scale = max_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def list_images(directory: str | Path) -> list[Path]:
    """Image files in ``directory`` (recursive), sorted by path."""
    root = Path(directory)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
