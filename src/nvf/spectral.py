"""Slope of the radially averaged Fourier amplitude spectrum.

For many natural scenes the amplitude spectrum falls off approximately as
A(f) ~ 1 / f^alpha with alpha close to 1 (Field, 1987), i.e. a power
spectrum slope near 2.

Procedure (defaults):

1. Luminance in [0, 1]; the largest central square is cropped so that
   frequencies are in cycles per image along both axes.
2. Mean is subtracted and a 2-D Hann window is applied to suppress the
   spurious cross-shaped energy caused by the image borders.
3. A(f) = |FFT| is averaged in rings of width 1 cycle/image
   (f rounded to the nearest integer).
4. alpha is the negated slope of an OLS fit of log A on log f for
   ``f_min <= f <= f_max``; defaults are f_min = 8 cycles/image and
   f_max = N/4 cycles/image (half the Nyquist frequency). The lower bound
   avoids the frequencies most affected by the window and by image size;
   the upper bound avoids aliasing, sensor noise and compression artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nvf.imageio import to_gray

DEFAULT_F_MIN = 8.0
DEFAULT_F_MAX_FRACTION = 0.25


@dataclass(frozen=True)
class SpectrumResult:
    """Amplitude-spectrum slope alpha, R^2 of the log-log fit, and the radial profile."""

    alpha: float
    r2: float
    f_min: float
    f_max: float
    freqs: np.ndarray = field(repr=False)
    amplitude: np.ndarray = field(repr=False)


def central_square(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    n = min(h, w)
    top, left = (h - n) // 2, (w - n) // 2
    return gray[top : top + n, left : left + n]


def radial_amplitude(gray: np.ndarray, window: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged amplitude spectrum of a square 2-D array.

    Returns (frequencies in cycles/image, mean amplitude), for f = 1 .. N/2.
    """
    img = np.asarray(gray, dtype=np.float64)
    n = img.shape[0]
    if img.shape != (n, n):
        raise ValueError("radial_amplitude expects a square array")
    img = img - img.mean()
    if window:
        w = np.hanning(n)
        img = img * np.outer(w, w)
    amp = np.abs(np.fft.fft2(img))
    f = np.fft.fftfreq(n) * n
    fy, fx = np.meshgrid(f, f, indexing="ij")
    ring = np.rint(np.hypot(fx, fy)).astype(int).ravel()
    sums = np.bincount(ring, weights=amp.ravel())
    counts = np.bincount(ring)
    fmax = n // 2
    freqs = np.arange(1, fmax + 1)
    return freqs.astype(float), sums[1 : fmax + 1] / counts[1 : fmax + 1]


def spectral_slope(image: np.ndarray, f_min: float = DEFAULT_F_MIN,
                   f_max: float | None = None, window: bool = True) -> SpectrumResult:
    """Estimate alpha in A(f) ~ 1/f^alpha for an image (RGB or grayscale)."""
    gray = central_square(to_gray(image))
    n = gray.shape[0]
    if f_max is None:
        f_max = DEFAULT_F_MAX_FRACTION * n
    freqs, amp = radial_amplitude(gray, window=window)
    sel = (freqs >= f_min) & (freqs <= f_max) & (amp > 0)
    if sel.sum() < 3 or np.ptp(gray) < 1e-12:
        return SpectrumResult(float("nan"), float("nan"), f_min, f_max, freqs, amp)
    x, y = np.log(freqs[sel]), np.log(amp[sel])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return SpectrumResult(float(-slope), r2, float(f_min), float(f_max), freqs, amp)


def spectral_features(image: np.ndarray) -> dict[str, float]:
    """Flat dict of spectral features with default parameters."""
    res = spectral_slope(image)
    return {"spec_alpha": res.alpha, "spec_r2": res.r2,
            "spec_fmin": res.f_min, "spec_fmax": res.f_max}
