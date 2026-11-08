"""Procedural facade-like patterns with controlled structure, and their features.

Patterns (512 x 512 RGB):
  grid_facade        regular window grid on a plain wall
  voronoi_cells      irregular cells (random seeds) with dark joints
  lsystem_branching  bracketed L-system plant, of the kind described by
                     Prusinkiewicz & Lindenmayer (1990)
  noise_alpha_X      grayscale 1/f^alpha noise, alpha in {1.0, 1.5, 2.0}
  levelset_alpha_X   two-tone level set of 1/f^alpha noise; the outline of a
                     level set of such a surface has D = 3 - alpha in the limit
                     of infinite resolution (Voss, 1988), i.e. 1.3 / 1.5 / 1.8
                     for alpha 1.7 / 1.5 / 1.2
  sierpinski_carpet  deterministic fractal ornament

Writes data/patterns/*.png (git-ignored), results/pattern_features.csv and
results/figures/pattern_gallery.png.

Usage: python scripts/make_patterns.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import _style  # noqa: E402

from nvf import synthetic  # noqa: E402
from nvf.pipeline import extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = 512

WALL = (205, 200, 190)
GLASS = (55, 70, 90)
SKY = (205, 222, 238)
LEAF = (48, 112, 56)


def grid_facade(n: int = N) -> np.ndarray:
    img = np.full((n, n, 3), WALL, np.uint8)
    rows, cols = 8, 6
    cell_h, cell_w = n / rows, n / cols
    for r in range(rows):
        for c in range(cols):
            y0, x0 = int(r * cell_h + 0.25 * cell_h), int(c * cell_w + 0.3 * cell_w)
            y1, x1 = int((r + 1) * cell_h - 0.2 * cell_h), int((c + 1) * cell_w - 0.3 * cell_w)
            img[y0:y1, x0:x1] = GLASS
    return img


def voronoi_cells(n: int = N, n_seeds: int = 70, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, n, size=(n_seeds, 2))
    yy, xx = np.mgrid[0:n, 0:n]
    _, label = cKDTree(points).query(np.column_stack([yy.ravel(), xx.ravel()]))
    label = label.reshape(n, n)
    palette = np.array([(196, 160, 120), (170, 140, 105), (210, 190, 160), (150, 120, 90)])
    img = palette[rng.integers(0, len(palette), n_seeds)][label].astype(np.uint8)
    joint = np.zeros((n, n), bool)
    joint[:-1, :] |= label[:-1, :] != label[1:, :]
    joint[:, :-1] |= label[:, :-1] != label[:, 1:]
    joint = cv2.dilate(joint.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    img[joint] = (70, 60, 50)
    return img


def lsystem(axiom: str, rules: dict[str, str], iterations: int) -> str:
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(ch, ch) for ch in s)
    return s


def lsystem_branching(n: int = N, iterations: int = 5, angle_deg: float = 25.0) -> np.ndarray:
    """Branching plant from the bracketed L-system X -> F+[[X]-X]-F[-X]+X, F -> FF."""
    program = lsystem("X", {"X": "F+[[X]-X]-F[-X]+X", "F": "FF"}, iterations)
    x, y, heading = 0.0, 0.0, math.radians(65)
    stack, segments = [], []
    turn = math.radians(angle_deg)
    for ch in program:
        if ch == "F":
            nx, ny = x + math.cos(heading), y + math.sin(heading)
            segments.append((x, y, nx, ny))
            x, y = nx, ny
        elif ch == "+":
            heading -= turn
        elif ch == "-":
            heading += turn
        elif ch == "[":
            stack.append((x, y, heading))
        elif ch == "]":
            x, y, heading = stack.pop()
    seg = np.array(segments)
    xs, ys = seg[:, [0, 2]], seg[:, [1, 3]]
    margin = 0.04 * n
    scale = (n - 2 * margin) / max(np.ptp(xs), np.ptp(ys))
    px = (xs - xs.min()) * scale + margin + (n - 2 * margin - np.ptp(xs) * scale) / 2
    py = n - margin - (ys - ys.min()) * scale
    img = np.full((n, n, 3), SKY, np.uint8)
    for (x0, x1), (y0, y1) in zip(px, py, strict=True):
        cv2.line(img, (round(x0), round(y0)), (round(x1), round(y1)), LEAF, 1, cv2.LINE_AA)
    return img


def noise_gray(alpha: float, n: int = N, seed: int = 0) -> np.ndarray:
    g = (synthetic.power_law_noise(n, alpha, seed=seed) * 255).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


def two_tone_fractal(alpha: float, n: int = N, seed: int = 1) -> np.ndarray:
    noise = synthetic.power_law_noise(n, alpha, seed=seed)
    mask = noise > np.median(noise)
    img = np.empty((n, n, 3), np.uint8)
    img[mask] = LEAF
    img[~mask] = SKY
    return img


def carpet(n: int = N) -> np.ndarray:
    m = synthetic.sierpinski_carpet(n)
    img = np.empty((n, n, 3), np.uint8)
    img[m] = (190, 95, 60)
    img[~m] = (240, 232, 218)
    return img


def patterns() -> dict[str, np.ndarray]:
    return {
        "grid_facade": grid_facade(),
        "voronoi_cells": voronoi_cells(),
        "lsystem_branching": lsystem_branching(),
        "sierpinski_carpet": carpet(),
        "noise_alpha_1.0": noise_gray(1.0),
        "noise_alpha_1.5": noise_gray(1.5),
        "noise_alpha_2.0": noise_gray(2.0),
        "levelset_alpha_1.7": two_tone_fractal(1.7),
        "levelset_alpha_1.5": two_tone_fractal(1.5),
        "levelset_alpha_1.2": two_tone_fractal(1.2),
    }


def gallery(images: dict[str, np.ndarray], feats: pd.DataFrame, path: Path) -> None:
    plt = _style.apply()
    cols = 5
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.25 * rows))
    for ax, (name, img) in zip(axes.ravel(), images.items(), strict=False):
        f = feats.loc[name]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(name.replace("_", " "), fontsize=9, loc="left")
        ax.set_xlabel(f"D otsu {f.fd_otsu:.2f}  bound. {f.fd_boundary:.2f}  "
                      f"edges {f.fd_edges:.2f}\n"
                      f"alpha {f.spec_alpha:.2f}   M {f.colourfulness:.0f}   "
                      f"Hhue {f.hue_entropy:.2f}", fontsize=8, loc="left",
                      color=_style.TEXT_SECONDARY)
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> pd.DataFrame:
    images = patterns()
    out_dir = ROOT / "data" / "patterns"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, img in images.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append({"pattern": name, **extract_features(img)})
    feats = pd.DataFrame(rows).set_index("pattern")
    cols = ["fd_otsu", "fd_otsu_r2", "fd_boundary", "fd_boundary_r2", "fd_edges", "fd_edges_r2",
            "spec_alpha", "spec_r2",
            "hue_entropy", "mean_saturation", "mean_value", "colourfulness"]
    (ROOT / "results").mkdir(exist_ok=True)
    feats[cols].to_csv(ROOT / "results" / "pattern_features.csv", float_format="%.3f")
    gallery(images, feats, ROOT / "results" / "figures" / "pattern_gallery.png")
    print(feats[cols].round(3).to_string())
    return feats


if __name__ == "__main__":
    main()
