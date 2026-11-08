"""Validate fractal D and spectral alpha estimators on synthetic images.

Writes to results/:
  validation_fractal.csv            D per pattern and resolution (default fit range)
  validation_fractal_fitrange.csv   D at 1024 px for several upper fit limits
  validation_spectral.csv           alpha recovery for 1/f^alpha noise
  validation_noise_coastline.csv    edge D of thresholded noise vs 3 - alpha
  figures/validation.png

Usage: python scripts/validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _style  # noqa: E402

from nvf import fractal, spectral, synthetic  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESOLUTIONS = [256, 512, 1024, 2048]
ALPHAS = [0.5, 1.0, 1.5, 2.0, 2.5]
SEEDS = range(5)


def fractal_validation() -> pd.DataFrame:
    rows = []
    for name, make in synthetic.PATTERNS.items():
        for n in RESOLUTIONS:
            res = fractal.box_counting_dimension(make(n))
            lo, hi = res.scale_range
            rows.append({"pattern": name, "n": n, "D_true": synthetic.KNOWN_D[name],
                         "D_est": res.D, "error": res.D - synthetic.KNOWN_D[name],
                         "r2": res.r2, "s_min": lo, "s_max": hi})
    return pd.DataFrame(rows)


def fit_range_sensitivity(n: int = 1024) -> pd.DataFrame:
    rows = []
    for name, make in synthetic.PATTERNS.items():
        img = make(n)
        for denom in (4, 8, 16, 32):
            res = fractal.box_counting_dimension(img, max_box=n // denom)
            rows.append({"pattern": name, "n": n, "s_max": n // denom,
                         "D_true": synthetic.KNOWN_D[name], "D_est": res.D,
                         "error": res.D - synthetic.KNOWN_D[name], "r2": res.r2})
    return pd.DataFrame(rows)


def spectral_validation() -> pd.DataFrame:
    rows = []
    for n in (256, 512, 1024):
        for alpha in ALPHAS:
            for crop in (False, True):
                est = []
                for seed in SEEDS:
                    if crop:  # non-periodic: central crop of a 2n periodic image
                        big = synthetic.power_law_noise(2 * n, alpha, seed=seed)
                        img = big[n // 2 : n // 2 + n, n // 2 : n // 2 + n]
                    else:
                        img = synthetic.power_law_noise(n, alpha, seed=seed)
                    est.append(spectral.spectral_slope(img).alpha)
                est = np.array(est)
                rows.append({"n": n, "alpha_true": alpha, "non_periodic_crop": crop,
                             "alpha_mean": est.mean(), "alpha_sd": est.std(ddof=1),
                             "bias": est.mean() - alpha,
                             "max_abs_error": np.abs(est - alpha).max(),
                             "n_seeds": len(est)})
    return pd.DataFrame(rows)


def coastline_validation() -> pd.DataFrame:
    """Outlines of level sets of 1/f^alpha surfaces; D -> 3 - alpha (Voss, 1988)."""
    methods = {
        "boundary": lambda b: fractal.fractal_dimension_boundary(b).D,
        "edges_sigma2": lambda b: fractal.fractal_dimension_edges(b, sigma=2.0).D,
        "edges_sigma1": lambda b: fractal.fractal_dimension_edges(b, sigma=1.0).D,
    }
    rows = []
    for n in (512, 1024):
        for alpha in (1.2, 1.4, 1.6, 1.8):
            est = {m: [] for m in methods}
            for seed in SEEDS:
                noise = synthetic.power_law_noise(n, alpha, seed=seed)
                binary = (noise > np.median(noise)).astype(float)
                for m, fn in methods.items():
                    est[m].append(fn(binary))
            for m, values in est.items():
                v = np.array(values)
                rows.append({"n": n, "alpha": alpha, "D_expected": 3 - alpha, "method": m,
                             "D_mean": v.mean(), "D_sd": v.std(ddof=1),
                             "bias": v.mean() - (3 - alpha)})
    return pd.DataFrame(rows)


def plot(frac: pd.DataFrame, spec: pd.DataFrame, path: Path) -> None:
    plt = _style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for colour, (name, g) in zip(_style.SERIES, frac.groupby("pattern", sort=False),
                                 strict=False):
        ax.plot(g["n"], g["error"], "-o", color=colour, label=name.replace("_", " "))
        if g["error"].abs().max() > 0.01:  # exact patterns overlap at 0; noted below
            ax.annotate(name.replace("_", " "), (g["n"].iloc[-1], g["error"].iloc[-1]),
                        xytext=(6, -8), textcoords="offset points", va="center", fontsize=8,
                        color=_style.TEXT_SECONDARY)
    exact = [n.replace("_", " ") for n, g in frac.groupby("pattern", sort=False)
             if g["error"].abs().max() <= 0.01]
    ax.annotate(f"{', '.join(exact)}: |error| < 0.01", (0.22, 0.6), xycoords="axes fraction",
                fontsize=8, color=_style.TEXT_SECONDARY)
    ax.axhline(0, color=_style.TEXT_SECONDARY, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RESOLUTIONS, [str(n) for n in RESOLUTIONS])
    ax.set_xlim(200, 6000)
    ax.set_xlabel("image size N (pixels)")
    ax.set_ylabel("D estimated - D true")
    ax.set_title("Box-counting D: error by resolution", loc="left")
    ax.legend(loc="lower right")

    ax = axes[1]
    sub = spec[(spec["n"] == 512)]
    for colour, (crop, g) in zip(_style.SERIES, sub.groupby("non_periodic_crop"),
                                 strict=False):
        label = "non-periodic crop" if crop else "periodic"
        ax.errorbar(g["alpha_true"], g["alpha_mean"], yerr=g["alpha_sd"], fmt="o",
                    color=colour, label=label, capsize=3, ms=6)
    lim = [0.3, 2.7]
    ax.plot(lim, lim, color=_style.TEXT_SECONDARY, lw=0.8, ls="--", label="identity")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("alpha true")
    ax.set_ylabel("alpha estimated (mean +/- SD, 5 seeds)")
    ax.set_title("Spectral slope recovery (N = 512)", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    frac = fractal_validation()
    frac.to_csv(RESULTS / "validation_fractal.csv", index=False, float_format="%.4f")
    fit = fit_range_sensitivity()
    fit.to_csv(RESULTS / "validation_fractal_fitrange.csv", index=False, float_format="%.4f")
    spec = spectral_validation()
    spec.to_csv(RESULTS / "validation_spectral.csv", index=False, float_format="%.4f")
    coast = coastline_validation()
    coast.to_csv(RESULTS / "validation_noise_coastline.csv", index=False, float_format="%.4f")
    plot(frac, spec, RESULTS / "figures" / "validation.png")

    pd.set_option("display.width", 140)
    print(frac.pivot(index="pattern", columns="n", values="error").round(3), "\n")
    print(fit.pivot(index="pattern", columns="s_max", values="error").round(3), "\n")
    print(spec.round(3).to_string(index=False), "\n")
    print(coast.pivot_table(index=["n", "alpha"], columns="method", values="bias").round(3))


if __name__ == "__main__":
    main()
