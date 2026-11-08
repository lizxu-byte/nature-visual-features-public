"""Combine all feature families for one image."""

from __future__ import annotations

import numpy as np

from nvf.colour import colour_features
from nvf.fractal import fractal_features
from nvf.imageio import as_rgb_uint8, resize_max_side
from nvf.semantic import Segmenter, semantic_features
from nvf.spectral import spectral_features

# Images are downscaled (never upscaled) to this long side before feature
# extraction so that resolution-dependent features (D, alpha) are comparable.
DEFAULT_MAX_SIDE = 1024


def extract_features(image: np.ndarray, segmenter: Segmenter | None = None,
                     max_side: int | None = DEFAULT_MAX_SIDE) -> dict[str, float]:
    """All features for an RGB image; semantic ones only if a segmenter is given."""
    rgb = as_rgb_uint8(image)
    h0, w0 = rgb.shape[:2]
    rgb, scale = resize_max_side(rgb, max_side)
    out: dict[str, float] = {
        "orig_width": w0, "orig_height": h0,
        "width": rgb.shape[1], "height": rgb.shape[0], "scale": scale,
    }
    out.update(fractal_features(rgb))
    out.update(spectral_features(rgb))
    out.update(colour_features(rgb))
    if segmenter is not None:
        out.update(semantic_features(rgb, segmenter))
    return out
