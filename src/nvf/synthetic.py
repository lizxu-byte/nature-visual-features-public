"""Synthetic images with known fractal dimension or spectral slope.

Used for validation and tests. Deterministic fractals are rasterised by
sampling the continuous set at pixel centres, so any resolution ``n`` can be
generated (the set is resolved down to one pixel).
"""

from __future__ import annotations

import math

import numpy as np
from skimage.draw import line as draw_line

# Theoretical similarity dimensions.
KNOWN_D = {
    "sierpinski_carpet": math.log(8) / math.log(3),
    "sierpinski_triangle": math.log(3) / math.log(2),
    "koch_curve": math.log(4) / math.log(3),
    "line": 1.0,
    "filled_square": 2.0,
}


def _pixel_digits(n: int, base: int) -> tuple[np.ndarray, int]:
    """Integer cell index of each pixel centre at a depth where cells <= 1 pixel."""
    depth = max(1, math.ceil(math.log(n, base)))
    centres = (np.arange(n) + 0.5) / n
    return np.floor(centres * base**depth).astype(np.int64), depth


def sierpinski_carpet(n: int) -> np.ndarray:
    """Sierpinski carpet on an n x n grid (D = log 8 / log 3)."""
    idx, depth = _pixel_digits(n, 3)
    yi, xi = np.meshgrid(idx, idx, indexing="ij")
    keep = np.ones((n, n), dtype=bool)
    for _ in range(depth):
        keep &= ~((yi % 3 == 1) & (xi % 3 == 1))
        yi //= 3
        xi //= 3
    return keep


def sierpinski_triangle(n: int) -> np.ndarray:
    """Right-angled Sierpinski triangle (Pascal's triangle mod 2), D = log 3 / log 2."""
    idx, _ = _pixel_digits(n, 2)
    yi, xi = np.meshgrid(idx, idx, indexing="ij")
    return (yi & xi) == 0


def koch_curve_points(order: int) -> np.ndarray:
    """Vertices of the Koch curve of a given order on the unit interval."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    rot = np.array([[0.5, -math.sqrt(3) / 2], [math.sqrt(3) / 2, 0.5]])
    for _ in range(order):
        a, b = pts[:-1], pts[1:]
        d = (b - a) / 3
        p1 = a + d
        p2 = p1 + d @ rot.T
        p3 = a + 2 * d
        new = np.stack([a, p1, p2, p3], axis=1).reshape(-1, 2)
        pts = np.vstack([new, pts[-1:]])
    return pts


def koch_curve(n: int) -> np.ndarray:
    """Koch curve rasterised with 1-pixel lines; D = log 4 / log 3.

    The recursion order is chosen so that the shortest segment is about one
    pixel long. The curve spans the full width with a small margin.
    """
    margin = 0.02 * n
    span = n - 2 * margin - 1
    order = max(1, math.floor(math.log(span) / math.log(3)))
    pts = koch_curve_points(order) * span
    pts[:, 0] += margin
    pts[:, 1] = n / 2 + span * math.sqrt(3) / 12 - pts[:, 1]  # centre the bump vertically
    img = np.zeros((n, n), dtype=bool)
    p = np.round(pts).astype(int)
    for (x0, y0), (x1, y1) in zip(p[:-1], p[1:], strict=True):
        rr, cc = draw_line(y0, x0, y1, x1)
        img[np.clip(rr, 0, n - 1), np.clip(cc, 0, n - 1)] = True
    return img


def diagonal_line(n: int) -> np.ndarray:
    """One-pixel diagonal line across the image (D = 1)."""
    return np.eye(n, dtype=bool)


def filled_square(n: int) -> np.ndarray:
    """Filled square covering the central half of each axis (D = 2)."""
    img = np.zeros((n, n), dtype=bool)
    img[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = True
    return img


PATTERNS = {
    "sierpinski_carpet": sierpinski_carpet,
    "sierpinski_triangle": sierpinski_triangle,
    "koch_curve": koch_curve,
    "line": diagonal_line,
    "filled_square": filled_square,
}


def radial_frequency(n: int) -> np.ndarray:
    """Radial spatial frequency in cycles per image for an n x n FFT grid."""
    f = np.fft.fftfreq(n) * n
    fy, fx = np.meshgrid(f, f, indexing="ij")
    return np.hypot(fx, fy)


def power_law_noise(n: int, alpha: float, seed: int | None = 0) -> np.ndarray:
    """Gaussian noise image with amplitude spectrum proportional to 1 / f^alpha.

    White Gaussian noise is filtered in the Fourier domain, so phases are
    random and amplitudes follow the power law on average. The result is
    rescaled to [0, 1].
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(rng.standard_normal((n, n)))
    f = radial_frequency(n)
    f[0, 0] = 1.0
    spectrum *= f ** (-alpha)
    spectrum[0, 0] = 0.0
    img = np.real(np.fft.ifft2(spectrum))
    img -= img.min()
    return img / img.max()
