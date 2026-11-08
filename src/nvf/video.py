"""Frame sampling and per-frame / video-level features."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nvf.pipeline import DEFAULT_MAX_SIDE, extract_features
from nvf.semantic import Segmenter

# Columns that describe the frame or the fit set-up rather than the content.
NON_FEATURE_COLUMNS = {
    "frame_index", "time_s", "orig_width", "orig_height", "width", "height", "scale",
    "fd_otsu_smin", "fd_otsu_smax", "fd_boundary_smin", "fd_boundary_smax",
    "fd_edges_smin", "fd_edges_smax",
    "spec_fmin", "spec_fmax",
}


def sample_frames(path: str | Path, every_s: float) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, time_s, rgb) for one frame every ``every_s`` seconds.

    Frames are read sequentially (``grab`` for skipped frames), which is
    slower than seeking but exact for codecs with sparse keyframes. Frame k is
    taken at index round(k * every_s * fps), starting at t = 0.
    """
    if every_s <= 0:
        raise ValueError("every_s must be positive")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise ValueError(f"video reports no frame rate: {path}")
        k, index = 0, 0
        target = 0
        while cap.grab():
            if index == target:
                ok, bgr = cap.retrieve()
                if ok:
                    yield index, index / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                k += 1
                target = round(k * every_s * fps)
            index += 1
    finally:
        cap.release()


def video_features(path: str | Path, every_s: float = 5.0, segmenter: Segmenter | None = None,
                   max_side: int | None = DEFAULT_MAX_SIDE) -> pd.DataFrame:
    """Per-frame features for frames sampled every ``every_s`` seconds."""
    rows = []
    for index, t, rgb in sample_frames(path, every_s):
        row = {"frame_index": index, "time_s": t}
        row.update(extract_features(rgb, segmenter=segmenter, max_side=max_side))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(frames: pd.DataFrame) -> pd.DataFrame:
    """Video-level mean and SD (ddof=1) of each feature column, plus frame count."""
    cols = [c for c in frames.columns if c not in NON_FEATURE_COLUMNS]
    summary = pd.DataFrame({
        "feature": cols,
        "mean": [frames[c].mean() for c in cols],
        "sd": [frames[c].std(ddof=1) for c in cols],
        "n_frames": [int(frames[c].notna().sum()) for c in cols],
    })
    return summary
