import numpy as np
import pytest

from nvf import fractal, synthetic

# Tolerances on |D_est - D_true| with the default fit range (s = 2 .. N/8).
# They reflect the validation run in scripts/validate.py; the Koch curve is
# systematically underestimated (see README, "Validation").
TOLERANCE = {
    "sierpinski_carpet": 0.05,
    "sierpinski_triangle": 0.01,
    "koch_curve": 0.15,
    "line": 0.01,
    "filled_square": 0.01,
}


@pytest.mark.parametrize("name", sorted(synthetic.PATTERNS))
@pytest.mark.parametrize("n", [512, 1024])
def test_known_dimension_recovery(name, n):
    res = fractal.box_counting_dimension(synthetic.PATTERNS[name](n))
    assert abs(res.D - synthetic.KNOWN_D[name]) < TOLERANCE[name]
    assert res.r2 > 0.99


def test_koch_bias_is_an_underestimate():
    res = fractal.box_counting_dimension(synthetic.koch_curve(2048))
    assert synthetic.KNOWN_D["koch_curve"] - 0.10 < res.D < synthetic.KNOWN_D["koch_curve"]


def test_default_box_sizes():
    sizes = fractal.box_sizes((1024, 768))
    assert sizes.tolist() == [2, 4, 8, 16, 32, 64]


def test_box_count_handles_non_multiple_shapes():
    mask = np.ones((10, 7), dtype=bool)
    assert fractal.box_count(mask, 4) == 3 * 2
    assert fractal.box_count(mask, 1) == 70


def test_empty_and_constant_inputs_return_nan():
    assert np.isnan(fractal.box_counting_dimension(np.zeros((256, 256), bool)).D)
    assert np.isnan(fractal.fractal_dimension_otsu(np.full((256, 256), 0.5)).D)


def test_too_few_scales_returns_nan():
    res = fractal.box_counting_dimension(np.ones((16, 16), bool))
    assert np.isnan(res.D)


def test_otsu_recovers_binary_pattern_from_grayscale():
    carpet = synthetic.sierpinski_carpet(729)
    gray = np.where(carpet, 0.8, 0.2)
    direct = fractal.box_counting_dimension(carpet).D
    assert fractal.fractal_dimension_otsu(gray).D == pytest.approx(direct)


def test_otsu_invert_selects_dark_phase():
    gray = np.zeros((256, 256))
    gray[64:192, 64:192] = 1.0
    assert fractal.otsu_mask(gray).sum() == 128 * 128
    assert fractal.otsu_mask(gray, invert=True).sum() == 256 * 256 - 128 * 128


def test_edges_of_filled_square_are_one_dimensional():
    img = synthetic.filled_square(512).astype(float)
    res = fractal.fractal_dimension_edges(img)
    assert abs(res.D - 1.0) < 0.05


def test_rgb_input_and_feature_dict():
    tri = synthetic.sierpinski_triangle(512).astype(np.uint8) * 255
    rgb = np.repeat(tri[:, :, None], 3, axis=2)
    feats = fractal.fractal_features(rgb)
    assert abs(feats["fd_otsu"] - synthetic.KNOWN_D["sierpinski_triangle"]) < 0.01
    assert feats["fd_otsu_smin"] == 2 and feats["fd_otsu_smax"] == 64
    assert set(feats) >= {"fd_edges", "fd_edges_r2"}


def test_boundary_of_filled_square_is_one_dimensional():
    # N(s) = 4 L/s - 4 for a square outline: the corner term bends the log-log
    # line at coarse scales, which gives D ~ 1.06 over the default fit range.
    res = fractal.fractal_dimension_boundary(synthetic.filled_square(1024).astype(float))
    assert abs(res.D - 1.0) < 0.07


def test_boundary_mask_ignores_image_border():
    full = np.ones((64, 64), bool)
    assert not fractal.boundary_mask(full).any()
    sq = synthetic.filled_square(64)
    assert fractal.boundary_mask(sq).sum() == 4 * 32 - 4


def test_boundary_of_carpet_keeps_carpet_dimension():
    # The carpet's holes are spread over all scales, so its outline has the same D.
    carpet = synthetic.sierpinski_carpet(729).astype(float)
    res = fractal.fractal_dimension_boundary(carpet)
    assert abs(res.D - synthetic.KNOWN_D["sierpinski_carpet"]) < 0.05
