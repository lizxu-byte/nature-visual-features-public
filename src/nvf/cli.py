"""Command-line interface: ``nvf images`` and ``nvf video``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from nvf import __version__
from nvf.imageio import list_images, load_rgb
from nvf.pipeline import DEFAULT_MAX_SIDE, extract_features
from nvf.semantic import DEFAULT_MODEL


def _segmenter(args):
    if not args.semantic:
        return None
    from nvf.semantic import SegformerSegmenter

    return SegformerSegmenter(model_name=args.model, device=args.device)


def cmd_images(args) -> int:
    root = Path(args.directory)
    paths = list_images(root)
    if not paths:
        print(f"no images found in {root}", file=sys.stderr)
        return 1
    segmenter = _segmenter(args)
    rows = []
    for i, path in enumerate(paths, 1):
        try:
            rgb = load_rgb(path)
        except ValueError as exc:
            print(f"skipping: {exc}", file=sys.stderr)
            continue
        row = {"image": path.relative_to(root).as_posix()}
        row.update(extract_features(rgb, segmenter=segmenter, max_side=args.max_side))
        rows.append(row)
        print(f"[{i}/{len(paths)}] {row['image']}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}", file=sys.stderr)
    return 0


def cmd_video(args) -> int:
    from nvf.video import summarise, video_features

    frames = video_features(args.file, every_s=args.every, segmenter=_segmenter(args),
                            max_side=args.max_side)
    if frames.empty:
        print(f"no frames read from {args.file}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames.to_csv(out, index=False)
    summary_path = Path(args.summary_out) if args.summary_out else (
        out.with_name(out.stem + "_summary.csv"))
    summarise(frames).to_csv(summary_path, index=False)
    print(f"wrote {len(frames)} frames to {out} and summary to {summary_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nvf", description=__doc__)
    parser.add_argument("--version", action="version", version=f"nvf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="output CSV path")
    common.add_argument("--semantic", action="store_true",
                        help="add SegFormer semantic features (needs the [semantic] extra)")
    common.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    common.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    common.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                        help="downscale so the long side is at most this (0 = no resizing)")

    p_img = sub.add_parser("images", parents=[common], help="features for a directory of images")
    p_img.add_argument("directory")
    p_img.set_defaults(func=cmd_images)

    p_vid = sub.add_parser("video", parents=[common], help="features for sampled video frames")
    p_vid.add_argument("file")
    p_vid.add_argument("--every", type=float, default=5.0, help="sampling interval in seconds")
    p_vid.add_argument("--summary-out", help="video-level summary CSV "
                       "(default: <out>_summary.csv)")
    p_vid.set_defaults(func=cmd_video)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
