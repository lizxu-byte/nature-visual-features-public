"""Semantic composition from SegFormer-B0 (ADE20K) segmentation.

The 150 ADE20K classes (Zhou et al., 2017) are mapped to 11 groups plus
``other`` by the table in ``nvf/data/ade20k_groups.csv``. From a label map
the following are computed:

* ``sem_share_<group>``: fraction of pixels in each group (sums to 1);
* ``green_view_index``: vegetation share, the image-level quantity used by
  Yang et al. (2009) (who average it over several views per location);
* ``sky_view_share``: sky share of the image (not the hemispherical sky view
  factor);
* ``sem_entropy``: Shannon entropy (bits) of the group distribution,
  ``-sum_g p_g log2 p_g`` over the 12 groups (maximum log2(12) = 3.58).

The model (``nvidia/segformer-b0-finetuned-ade-512-512``; Xie et al., 2021)
is loaded through ``transformers`` and needs ``pip install .[semantic]``.
The feature functions themselves only need numpy and are tested with
synthetic label maps.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib import resources
from typing import Protocol

import numpy as np

from nvf.colour import shannon_entropy

DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
GROUPS = (
    "vegetation", "sky", "water", "building", "road", "vehicle",
    "person", "window", "wall", "ground", "mountain_rock", "other",
)
N_CLASSES = 150
# Logits are upsampled to at most this many pixels on the long side before argmax.
DEFAULT_LABEL_MAX_SIDE = 512


class Segmenter(Protocol):
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return an integer ADE20K label map (0..149) for an RGB uint8 image."""
        ...


@lru_cache(maxsize=1)
def load_mapping() -> tuple[tuple[str, ...], np.ndarray]:
    """Return (ADE20K class names, array mapping class id -> group index)."""
    text = resources.files("nvf").joinpath("data/ade20k_groups.csv").read_text("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    if [int(r["id"]) for r in rows] != list(range(N_CLASSES)):
        raise ValueError("ade20k_groups.csv must list ids 0..149 in order")
    unknown = {r["group"] for r in rows} - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups in mapping table: {sorted(unknown)}")
    names = tuple(r["ade20k_name"] for r in rows)
    lut = np.array([GROUPS.index(r["group"]) for r in rows], dtype=np.int64)
    return names, lut


def check_label_names(id2label: dict[int, str]) -> None:
    """Raise if a model's id2label does not match the mapping table."""
    names, _ = load_mapping()
    model_names = [str(id2label[i]).strip().lower() for i in range(len(id2label))]
    if len(model_names) != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} classes, model has {len(model_names)}")
    mismatched = [(i, a, b) for i, (a, b) in enumerate(zip(names, model_names, strict=True))
                  if a != b]
    if mismatched:
        raise ValueError(f"model labels differ from mapping table: {mismatched[:5]}")


def group_shares(labels: np.ndarray) -> np.ndarray:
    """Pixel share of each group (in GROUPS order) for an ADE20K label map."""
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("empty label map")
    if labels.min() < 0 or labels.max() >= N_CLASSES:
        raise ValueError("labels must be ADE20K class ids in 0..149")
    _, lut = load_mapping()
    counts = np.bincount(lut[labels.ravel()], minlength=len(GROUPS))
    return counts / counts.sum()


def semantic_features_from_labels(labels: np.ndarray) -> dict[str, float]:
    """Group shares, green view index, sky view share and group entropy."""
    shares = group_shares(labels)
    out = {f"sem_share_{g}": float(s) for g, s in zip(GROUPS, shares, strict=True)}
    out["green_view_index"] = out["sem_share_vegetation"]
    out["sky_view_share"] = out["sem_share_sky"]
    out["sem_entropy"] = shannon_entropy(shares)
    return out


class SegformerSegmenter:
    """CPU-friendly wrapper around a SegFormer ADE20K checkpoint (lazy imports)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu",
                 label_max_side: int = DEFAULT_LABEL_MAX_SIDE):
        try:
            import torch
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "semantic features need the optional extra: pip install '.[semantic]'"
            ) from exc
        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.label_max_side = label_max_side
        check_label_names(self.model.config.id2label)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = rgb.shape[:2]
        scale = min(1.0, self.label_max_side / max(h, w))
        size = (max(1, round(h * scale)), max(1, round(w * scale)))
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=size, mode="bilinear",
                                                 align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy()


def semantic_features(rgb: np.ndarray, segmenter: Segmenter) -> dict[str, float]:
    """Run a segmenter on an RGB image and compute semantic features."""
    return semantic_features_from_labels(segmenter.predict(rgb))
