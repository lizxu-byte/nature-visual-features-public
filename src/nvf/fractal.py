"""Box-counting fractal dimension of binary patterns.

The box-counting dimension of a set is estimated from the number N(s) of
square boxes of side s (pixels) that contain at least one foreground pixel:

    N(s) ~ s^(-D)   =>   log N(s) = -D log s + c

D is the negated slope of an ordinary least-squares fit of log N(s) on log s.

Fit range (defaults): box sizes are powers of two, s = 2, 4, 8, ... up to
``min(H, W) / 8``. Size 1 is excluded because at the pixel scale every
digitised pattern is a set of filled squares; sizes above 1/8 of the image
are excluded because fewer than 8 x 8 boxes tile the image and N(s) saturates.
Images whose sides are not multiples of s are zero-padded, which adds no
occupied boxes.

Three binary maps are derived from a grayscale image:

* ``otsu``: pixels brighter than the Otsu (1979) threshold are foreground.
  This measures the bright phase (use ``invert=True`` for the dark phase).
  Box-counting a filled region tends towards D = 2, so this variant mainly
  separates sparse from space-filling bright structure.
* ``boundary``: the one-pixel outline of the Otsu mask (pixels of the mask
  with a 4-neighbour outside it). This is the outline of the binarised
  pattern, in the spirit of silhouette-outline measures, and has no
  smoothing parameter.
* ``edges``: Canny (1986) edge map (Gaussian sigma 2 px by default), which
  also picks up texture and shading edges that do not cross the threshold.

D depends on the binarisation, the image resolution and the fit range; see
the README for validation on patterns with known D.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.feature import canny
from skimage.filters import threshold_otsu

from nvf.imageio import to_gray

DEFAULT_MIN_BOX = 2
DEFAULT_MAX_BOX_FRACTION = 1 / 8
DEFAULT_CANNY_SIGMA = 2.0


@dataclass(frozen=True)
class FractalResult:
    """Box-counting estimate: dimension D, R^2 of the log-log fit, and the data."""

    D: float
    r2: float
    sizes: np.ndarray = field(repr=False)
    counts: np.ndarray = field(repr=False)

    @property
    def scale_range(self) -> tuple[int, int]:
        if len(self.sizes) == 0:
            return (0, 0)
        return int(self.sizes.min()), int(self.sizes.max())


def box_sizes(shape: tuple[int, int], min_box: int = DEFAULT_MIN_BOX,
              max_box: int | None = None) -> np.ndarray:
    """Powers of two from ``min_box`` to ``max_box`` (default min(H, W) / 8)."""
    if max_box is None:
        max_box = int(min(shape) * DEFAULT_MAX_BOX_FRACTION)
    sizes = []
    s = 1
    while s <= max_box:
        if s >= min_box:
            sizes.append(s)
        s *= 2
    return np.asarray(sizes, dtype=int)


def box_count(mask: np.ndarray, size: int) -> int:
    """Number of size x size boxes (grid anchored at the top-left) containing foreground."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    ph, pw = -h % size, -w % size
    if ph or pw:
        mask = np.pad(mask, ((0, ph), (0, pw)))
    hh, ww = mask.shape
    blocks = mask.reshape(hh // size, size, ww // size, size)
    return int(blocks.any(axis=(1, 3)).sum())


def _fit(sizes: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    x = np.log(sizes.astype(float))
    y = np.log(counts.astype(float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-slope), r2


def box_counting_dimension(mask: np.ndarray, min_box: int = DEFAULT_MIN_BOX,
                           max_box: int | None = None,
                           sizes: np.ndarray | None = None) -> FractalResult:
    """Estimate the box-counting dimension of a binary mask.

    Returns NaN for D and R^2 if the mask is empty or fewer than three box
    sizes are available.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    if sizes is None:
        sizes = box_sizes(mask.shape, min_box, max_box)
    sizes = np.asarray(sizes, dtype=int)
    if not mask.any() or len(sizes) < 3:
        return FractalResult(float("nan"), float("nan"), sizes, np.zeros(len(sizes), int))
    counts = np.array([box_count(mask, s) for s in sizes])
    d, r2 = _fit(sizes, counts)
    return FractalResult(d, r2, sizes, counts)


def otsu_mask(image: np.ndarray, invert: bool = False) -> np.ndarray | None:
    """Binary mask of pixels above the Otsu threshold (None for a constant image)."""
    gray = to_gray(image)
    if np.ptp(gray) == 0:
        return None
    mask = gray > threshold_otsu(gray)
    return ~mask if invert else mask


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Inner one-pixel outline of a binary mask; the image border is not an outline."""
    mask = np.asarray(mask, dtype=bool)
    return mask & ~binary_erosion(mask, border_value=1)


def edge_mask(image: np.ndarray, sigma: float = DEFAULT_CANNY_SIGMA,
              low_threshold: float | None = None,
              high_threshold: float | None = None) -> np.ndarray:
    """Canny edge map of the luminance image (scikit-image implementation).

    With thresholds left as None, scikit-image uses 10% and 20% of the maximum
    of the image dtype (here 0.1 and 0.2 on luminance in [0, 1]), which keeps
    the thresholds absolute and therefore comparable across images.
    """
    return canny(to_gray(image), sigma=sigma, low_threshold=low_threshold,
                 high_threshold=high_threshold)


def fractal_dimension_otsu(image: np.ndarray, invert: bool = False,
                           **kwargs) -> FractalResult:
    """Box-counting D of the Otsu-binarised luminance image."""
    mask = otsu_mask(image, invert=invert)
    if mask is None:
        mask = np.zeros(np.asarray(image).shape[:2], dtype=bool)
    return box_counting_dimension(mask, **kwargs)


def fractal_dimension_boundary(image: np.ndarray, invert: bool = False,
                               **kwargs) -> FractalResult:
    """Box-counting D of the outline of the Otsu-binarised luminance image."""
    mask = otsu_mask(image, invert=invert)
    if mask is None:
        mask = np.zeros(np.asarray(image).shape[:2], dtype=bool)
    return box_counting_dimension(boundary_mask(mask), **kwargs)


def fractal_dimension_edges(image: np.ndarray, sigma: float = DEFAULT_CANNY_SIGMA,
                            **kwargs) -> FractalResult:
    """Box-counting D of the Canny edge map of the luminance image."""
    return box_counting_dimension(edge_mask(image, sigma=sigma), **kwargs)


def fractal_features(image: np.ndarray) -> dict[str, float]:
    """Flat dict of fractal features with default parameters."""
    out: dict[str, float] = {}
    for name, res in (("otsu", fractal_dimension_otsu(image)),
                      ("boundary", fractal_dimension_boundary(image)),
                      ("edges", fractal_dimension_edges(image))):
        lo, hi = res.scale_range
        out[f"fd_{name}"] = res.D
        out[f"fd_{name}_r2"] = res.r2
        out[f"fd_{name}_smin"] = lo
        out[f"fd_{name}_smax"] = hi
    return out
