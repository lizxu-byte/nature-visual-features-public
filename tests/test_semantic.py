import math

import numpy as np
import pytest

from nvf import semantic

# ADE20K ids used below (see src/nvf/data/ade20k_groups.csv)
WALL, BUILDING, SKY, FLOOR, TREE, ROAD, WINDOW, GRASS = 0, 1, 2, 3, 4, 6, 8, 9
PERSON, CAR, WATER, SEA = 12, 20, 21, 26


def test_mapping_table_is_complete_and_valid():
    names, lut = semantic.load_mapping()
    assert len(names) == 150 and lut.shape == (150,)
    assert len(set(names)) == 150
    assert names[TREE] == "tree" and names[WINDOW] == "windowpane" and names[149] == "flag"
    assert set(lut.tolist()) == set(range(len(semantic.GROUPS)))


@pytest.mark.parametrize(("label", "group"), [
    (TREE, "vegetation"), (GRASS, "vegetation"), (SKY, "sky"), (SEA, "water"),
    (BUILDING, "building"), (ROAD, "road"), (CAR, "vehicle"), (PERSON, "person"),
    (WINDOW, "window"), (WALL, "wall"), (FLOOR, "other"),
])
def test_class_to_group(label, group):
    feats = semantic.semantic_features_from_labels(np.full((4, 4), label))
    assert feats[f"sem_share_{group}"] == 1.0
    assert feats["sem_entropy"] == 0.0


def test_shares_gvi_and_entropy_on_fake_label_map():
    labels = np.zeros((10, 10), int)
    labels[:5, :] = SKY                # 50 %
    labels[5:, :5] = TREE              # 25 %
    labels[5:, 5:] = GRASS             # 25 % -> vegetation 50 %
    feats = semantic.semantic_features_from_labels(labels)
    assert feats["sky_view_share"] == 0.5
    assert feats["green_view_index"] == 0.5
    assert feats["sem_entropy"] == pytest.approx(1.0)
    assert sum(feats[f"sem_share_{g}"] for g in semantic.GROUPS) == pytest.approx(1.0)


def test_entropy_is_maximal_for_uniform_groups():
    names, lut = semantic.load_mapping()
    one_per_group = [int(np.flatnonzero(lut == g)[0]) for g in range(len(semantic.GROUPS))]
    feats = semantic.semantic_features_from_labels(np.array(one_per_group))
    assert feats["sem_entropy"] == pytest.approx(math.log2(len(semantic.GROUPS)))


def test_invalid_labels_raise():
    with pytest.raises(ValueError):
        semantic.group_shares(np.array([[150]]))
    with pytest.raises(ValueError):
        semantic.group_shares(np.array([[-1]]))


def test_check_label_names():
    names, _ = semantic.load_mapping()
    semantic.check_label_names(dict(enumerate(names)))
    semantic.check_label_names({i: n.upper() + " " for i, n in enumerate(names)})
    wrong = dict(enumerate(names))
    wrong[4] = "palm tree"
    with pytest.raises(ValueError, match="differ"):
        semantic.check_label_names(wrong)


class FakeSegmenter:
    """Stands in for SegFormer: left half water, right half building."""

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        labels = np.full((h, w), BUILDING)
        labels[:, : w // 2] = WATER
        return labels


def test_semantic_features_with_mocked_model():
    rgb = np.zeros((8, 8, 3), np.uint8)
    feats = semantic.semantic_features(rgb, FakeSegmenter())
    assert feats["sem_share_water"] == 0.5 and feats["sem_share_building"] == 0.5
    assert feats["green_view_index"] == 0.0
