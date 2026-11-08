"""End-to-end demo: pattern gallery, synthetic video, and (if downloaded) Commons examples.

Steps
  1. Procedural patterns -> results/pattern_features.csv, figures/pattern_gallery.png
  2. A synthetic 30 s "walk" video (facade grid -> level-set foliage -> branching)
     written to data/demo_walk.avi and processed with nvf.video (one frame
     every 2 s) -> results/demo_video_frames.csv, demo_video_summary.csv,
     figures/demo_video.png
  3. Example photographs listed in examples/ATTRIBUTION.csv and present under
     data/examples/ (run scripts/fetch_examples.py first) ->
     results/example_features.csv, example_summary_by_category.csv (medians),
     figures/example_features_by_category.png.
     Skipped with a message if the images are not available.

Usage: python scripts/run_demo.py [--semantic]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _style  # noqa: E402
import make_patterns  # noqa: E402

from nvf.imageio import load_rgb  # noqa: E402
from nvf.pipeline import extract_features  # noqa: E402
from nvf.video import summarise, video_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
ATTRIBUTION = ROOT / "examples" / "ATTRIBUTION.csv"
EXAMPLES_DIR = ROOT / "data" / "examples"

FPS = 5
KEYFRAMES = ["grid_facade", "levelset_alpha_1.5", "lsystem_branching"]
# Per-category medians written to results/example_summary_by_category.csv.
SUMMARY_COLS = ["fd_boundary", "fd_edges", "spec_alpha", "green_view_index", "sky_view_share",
                "sem_entropy", "hue_entropy", "colourfulness"]


def make_walk_video(path: Path, seconds: int = 30) -> None:
    """Two 15 s segments between three patterns: hold 5 s, cross-fade 5 s, hold 5 s."""
    images = make_patterns.patterns()
    keys = [images[k].astype(np.float32) for k in KEYFRAMES]
    n = keys[0].shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (n, n))
    segment = seconds / (len(keys) - 1)
    for i in range(seconds * FPS):
        t = i / FPS
        k = min(int(t // segment), len(keys) - 2)
        u = np.clip((t - k * segment) / segment * 3 - 1, 0, 1)  # hold, fade, hold
        frame = (1 - u) * keys[k] + u * keys[k + 1]
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def video_demo() -> pd.DataFrame:
    path = ROOT / "data" / "demo_walk.avi"
    make_walk_video(path)
    frames = video_features(path, every_s=2.0)
    frames.to_csv(RESULTS / "demo_video_frames.csv", index=False, float_format="%.4f")
    summarise(frames).to_csv(RESULTS / "demo_video_summary.csv", index=False,
                             float_format="%.4f")

    plt = _style.apply()
    panels = [("fd_boundary", "D (Otsu outline)"), ("spec_alpha", "spectral alpha"),
              ("colourfulness", "colourfulness M")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True)
    for ax, (col, label) in zip(axes, panels, strict=True):
        ax.plot(frames["time_s"], frames[col], "-o", color=_style.SERIES[0])
        ax.set_title(label, loc="left")
        ax.set_xlabel("time (s)")
    fig.suptitle("Synthetic walk: facade grid -> foliage level set -> branching "
                 "(frame every 2 s)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "demo_video.png")
    plt.close(fig)
    return frames


def example_demo(semantic: bool) -> pd.DataFrame | None:
    if not ATTRIBUTION.exists():
        print(f"[examples] skipped: {ATTRIBUTION.relative_to(ROOT)} not found; "
              "run scripts/fetch_examples.py first", file=sys.stderr)
        return None
    meta = pd.read_csv(ATTRIBUTION)
    present = meta[[(EXAMPLES_DIR / f).exists() for f in meta["file"]]]
    if present.empty:
        print("[examples] skipped: no images under data/examples/; run "
              "scripts/fetch_examples.py --from-attribution examples/ATTRIBUTION.csv",
              file=sys.stderr)
        return None
    segmenter = None
    if semantic:
        from nvf.semantic import SegformerSegmenter

        segmenter = SegformerSegmenter()
    rows = []
    for _, r in present.iterrows():
        feats = extract_features(load_rgb(EXAMPLES_DIR / r["file"]), segmenter=segmenter)
        rows.append({"file": r["file"], "category": r["category"], **feats})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "example_features.csv", index=False, float_format="%.4f")
    summary_cols = [c for c in SUMMARY_COLS if c in df]
    summary = df.groupby("category", sort=False)[summary_cols].median()
    summary.insert(0, "n", df.groupby("category", sort=False).size())
    summary.to_csv(RESULTS / "example_summary_by_category.csv", float_format="%.4f")

    plt = _style.apply()
    cols = [("fd_boundary", "D (Otsu outline)"), ("fd_edges", "D (Canny edges)"),
            ("spec_alpha", "spectral alpha"), ("colourfulness", "colourfulness M"),
            ("hue_entropy", "hue entropy (bits)")]
    if "green_view_index" in df:
        cols += [("green_view_index", "green view index"), ("sky_view_share", "sky share"),
                 ("sem_entropy", "semantic entropy (bits)")]
    cats = list(dict.fromkeys(meta["category"]))
    fig, axes = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 0.45 * len(cats) + 1.2),
                             sharey=True)
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, cols, strict=True):
        for i, cat in enumerate(cats):
            v = df.loc[df["category"] == cat, col].dropna()
            ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=18,
                       color=_style.SERIES[0], alpha=0.75, linewidths=0)
            if len(v):
                ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color=_style.TEXT, lw=1.5)
        ax.set_title(label, loc="left", fontsize=9)
        ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats])
    axes[0].invert_yaxis()  # shared y axis: invert once, first category on top
    fig.tight_layout()
    fig.savefig(FIGURES / "example_features_by_category.png")
    plt.close(fig)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--semantic", action="store_true",
                        help="add SegFormer features to the example photographs")
    args = parser.parse_args(argv)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("[1/3] patterns", file=sys.stderr)
    make_patterns.main()
    print("[2/3] synthetic video", file=sys.stderr)
    frames = video_demo()
    print(frames[["time_s", "fd_boundary", "spec_alpha", "colourfulness"]].round(3)
          .to_string(index=False))
    print("[3/3] example photographs", file=sys.stderr)
    df = example_demo(args.semantic)
    if df is not None:
        print(df.groupby("category").median(numeric_only=True).round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
