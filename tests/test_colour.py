import math

import cv2
import numpy as np
import pytest

from nvf import colour


def solid(rgb, size=32):
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def test_grey_image_has_zero_colourfulness_and_no_valid_hue():
    img = solid((128, 128, 128))
    feats = colour.colour_features(img)
    assert feats["colourfulness"] == 0.0
    assert feats["mean_saturation"] == 0.0
    assert feats["mean_value"] == pytest.approx(128 / 255, abs=1e-6)
    assert feats["hue_valid_fraction"] == 0.0
    assert math.isnan(feats["hue_entropy"])


def test_single_hue_has_zero_entropy():
    ent, frac = colour.hue_entropy(solid((0, 200, 0)))
    assert ent == 0.0 and frac == 1.0


def test_equal_split_over_hues():
    # Four pure hues (0, 60, 120, 240 deg) in equal areas -> 2 bits.
    img = np.zeros((40, 40, 3), np.uint8)
    img[:20, :20] = (255, 0, 0)     # 0 deg
    img[:20, 20:] = (255, 255, 0)   # 60 deg
    img[20:, :20] = (0, 255, 0)     # 120 deg
    img[20:, 20:] = (0, 0, 255)     # 240 deg
    ent, _ = colour.hue_entropy(img)
    assert ent == pytest.approx(2.0)


def test_uniform_hue_wheel_reaches_maximum_entropy():
    hues = np.repeat(np.arange(0, 360, 1, dtype=np.float32)[None, :], 10, axis=0)
    hsv = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ent, _ = colour.hue_entropy(rgb)
    assert ent == pytest.approx(math.log2(36), abs=0.02)


def test_saturation_mask_excludes_greyish_pixels():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :5] = (200, 0, 0)        # saturated red
    img[:, 5:] = (120, 115, 110)    # nearly grey, S < 0.2
    ent, frac = colour.hue_entropy(img)
    assert frac == pytest.approx(0.5)
    assert ent == 0.0


def test_colourfulness_matches_hand_computation():
    img = np.zeros((2, 2, 3), np.uint8)
    img[0, 0] = (255, 0, 0)
    img[0, 1] = (0, 255, 0)
    img[1, 0] = (0, 0, 255)
    img[1, 1] = (255, 255, 255)
    rgb = img.reshape(-1, 3).astype(float)
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
    expected = math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    assert colour.colourfulness(img) == pytest.approx(expected)


def test_colourfulness_orders_images():
    grey = solid((100, 100, 100))
    muted = np.concatenate([solid((120, 100, 90)), solid((90, 100, 120))])
    vivid = np.concatenate([solid((255, 0, 0)), solid((0, 0, 255))])
    assert colour.colourfulness(grey) < colour.colourfulness(muted) < colour.colourfulness(vivid)


def test_shannon_entropy_of_counts():
    assert colour.shannon_entropy([1, 1, 1, 1]) == pytest.approx(2.0)
    assert colour.shannon_entropy([5, 0, 0]) == 0.0
    assert math.isnan(colour.shannon_entropy([0, 0]))
