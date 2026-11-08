"""Overlay SegFormer group labels on the example photographs, for visual checking.

For every image listed in examples/ATTRIBUTION.csv and present under
data/examples/, writes a side-by-side photo / overlay PNG to
data/overlays/<category>/ (git-ignored) and prints the group shares and the
largest ADE20K classes. With --figure FILE [FILE ...] it also writes
results/figures/segmentation_overlays.png from the named files (paths
relative to data/examples/), with author and licence under each panel.

Usage: python scripts/segmentation_overlays.py [--figure forest/a.jpg water/b.jpg ...]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _style  # noqa: E402

from nvf.imageio import load_rgb, resize_max_side  # noqa: E402
from nvf.pipeline import DEFAULT_MAX_SIDE  # noqa: E402
from nvf.semantic import GROUPS, SegformerSegmenter, load_mapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION = ROOT / "examples" / "ATTRIBUTION.csv"
EXAMPLES_DIR = ROOT / "data" / "examples"
OVERLAY_DIR = ROOT / "data" / "overlays"
FIGURE = ROOT / "results" / "figures" / "segmentation_overlays.png"

# One colour per group (GROUPS order); "other" is left grey.
GROUP_COLOURS = {
    "vegetation": "#1baf7a", "sky": "#7fc4f5", "water": "#2a78d6", "building": "#eb6834",
    "road": "#8a8a8a", "vehicle": "#e34948", "person": "#e87ba4", "window": "#eda100",
    "wall": "#6250d6", "ground": "#a0703c", "mountain_rock": "#5b4a3a", "other": "#d9d9d9",
}


def colour_lut() -> np.ndarray:
    return np.array([[int(GROUP_COLOURS[g][i:i + 2], 16) for i in (1, 3, 5)] for g in GROUPS],
                    dtype=np.float64)


def segment(rgb: np.ndarray, segmenter: SegformerSegmenter) -> tuple[np.ndarray, np.ndarray]:
    """Working-resolution image (as in the pipeline) and its label map at the same size."""
    import cv2

    rgb, _ = resize_max_side(rgb, DEFAULT_MAX_SIDE)
    labels = segmenter.predict(rgb)
    labels = cv2.resize(labels.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    return rgb, labels


def overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    _, lut = load_mapping()
    colours = colour_lut()[lut[labels]]
    return ((1 - alpha) * rgb + alpha * colours).astype(np.uint8)


def describe(labels: np.ndarray, top: int = 6) -> str:
    names, lut = load_mapping()
    counts = np.bincount(labels.ravel(), minlength=len(names)) / labels.size
    groups = np.bincount(lut, weights=counts, minlength=len(GROUPS))
    g = ", ".join(f"{GROUPS[i]} {groups[i]:.2f}" for i in np.argsort(-groups) if groups[i] >= 0.01)
    c = ", ".join(f"{names[i]} {counts[i]:.2f}" for i in np.argsort(-counts)[:top] if counts[i] > 0)
    return f"  groups: {g}\n  classes: {c}"


def legend_handles():
    from matplotlib.patches import Patch

    return [Patch(color=GROUP_COLOURS[g], label=g.replace("_", " ")) for g in GROUPS]


def save_pair(rgb, labels, path: Path, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 10 * rgb.shape[0] / rgb.shape[1] / 2 + 0.8))
    for ax, img in zip(axes, [rgb, overlay(rgb, labels)], strict=True):
        ax.imshow(img)
        ax.axis("off")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def credit(row: pd.Series) -> str:
    title = row["title"].removeprefix("File:")
    return textwrap.fill(f"{title} — {row['author']}, {row['licence']}", 48)


def save_figure(cases: list[tuple[pd.Series, np.ndarray, np.ndarray]], plt) -> None:
    fig, axes = plt.subplots(len(cases), 2, figsize=(8, 2.9 * len(cases)))
    for (row, rgb, labels), (ax0, ax1) in zip(cases, np.atleast_2d(axes), strict=True):
        ax0.imshow(rgb)
        ax1.imshow(overlay(rgb, labels))
        for ax in (ax0, ax1):
            ax.axis("off")
        ax0.set_title(row["category"].replace("_", " "), loc="left", fontsize=10)
        ax1.set_title(credit(row), loc="left", fontsize=6.5, color=_style.TEXT_SECONDARY)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=6, fontsize=7,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", nargs="+", default=[],
                        help="files (relative to data/examples/) for the committed figure")
    args = parser.parse_args(argv)

    meta = pd.read_csv(ATTRIBUTION)
    meta = meta[[(EXAMPLES_DIR / f).exists() for f in meta["file"]]]
    missing = set(args.figure) - set(meta["file"])
    if missing:
        parser.error(f"not in ATTRIBUTION.csv or not downloaded: {sorted(missing)}")
    plt = _style.apply()
    plt.rcParams["axes.grid"] = False
    segmenter = SegformerSegmenter()
    cases = {}
    for _, row in meta.iterrows():
        rgb, labels = segment(load_rgb(EXAMPLES_DIR / row["file"]), segmenter)
        print(f"{row['file']}\n{describe(labels)}")
        save_pair(rgb, labels, (OVERLAY_DIR / row["file"]).with_suffix(".png"), plt)
        if row["file"] in args.figure:
            cases[row["file"]] = (row, rgb, labels)
    if args.figure:
        save_figure([cases[f] for f in args.figure], plt)
        print(f"wrote {FIGURE.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
