"""Colour statistics: HSV hue entropy, mean saturation/value, colourfulness.

Hue entropy
    Hue is undefined for grey pixels and unstable for dark or weakly saturated
    ones, so it is computed only over pixels with S >= ``s_min`` and
    V >= ``v_min`` (defaults 0.2 and 0.2 on [0, 1] scales). Hue (0-360 deg) is
    binned into ``bins`` equal bins (default 36, i.e. 10 deg), and

        H_hue = -sum_k p_k log2 p_k        (bits; maximum log2(bins))

    ``hue_entropy_norm`` divides by log2(bins). If no pixel passes the mask
    the entropy is NaN; ``hue_valid_fraction`` reports the share of pixels
    used.

Mean saturation and value
    Means of HSV S and V over all pixels (OpenCV HSV, [0, 1]).

Colourfulness (Hasler & Suesstrunk, 2003)
    On 8-bit sRGB values, with rg = R - G and yb = (R + G)/2 - B:

        M = sqrt(sd_rg^2 + sd_yb^2) + 0.3 sqrt(mean_rg^2 + mean_yb^2)
"""

from __future__ import annotations

import cv2
import numpy as np

from nvf.imageio import as_rgb_uint8

DEFAULT_S_MIN = 0.2
DEFAULT_V_MIN = 0.2
DEFAULT_HUE_BINS = 36


def to_hsv(image: np.ndarray) -> np.ndarray:
    """HSV with H in [0, 360) degrees and S, V in [0, 1]."""
    rgb = as_rgb_uint8(image).astype(np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits of a probability (or count) vector."""
    p = np.asarray(p, dtype=float)
    total = p.sum()
    if total <= 0:
        return float("nan")
    p = p[p > 0] / total
    return float(0.0 - (p * np.log2(p)).sum())  # 0.0 - x avoids returning -0.0


def hue_entropy(image: np.ndarray, s_min: float = DEFAULT_S_MIN, v_min: float = DEFAULT_V_MIN,
                bins: int = DEFAULT_HUE_BINS) -> tuple[float, float]:
    """Return (hue entropy in bits, fraction of pixels used)."""
    hsv = to_hsv(image)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (s >= s_min) & (v >= v_min)
    frac = float(valid.mean())
    if not valid.any():
        return float("nan"), frac
    hist, _ = np.histogram(h[valid], bins=bins, range=(0.0, 360.0))
    return shannon_entropy(hist), frac


def colourfulness(image: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric M on 8-bit sRGB."""
    rgb = as_rgb_uint8(image).astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    sd = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mu = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(sd + 0.3 * mu)


def colour_features(image: np.ndarray) -> dict[str, float]:
    """Flat dict of colour features with default parameters."""
    hsv = to_hsv(image)
    ent, frac = hue_entropy(image)
    return {
        "hue_entropy": ent,
        "hue_entropy_norm": ent / np.log2(DEFAULT_HUE_BINS),
        "hue_valid_fraction": frac,
        "mean_saturation": float(hsv[..., 1].mean()),
        "mean_value": float(hsv[..., 2].mean()),
        "colourfulness": colourfulness(image),
    }
