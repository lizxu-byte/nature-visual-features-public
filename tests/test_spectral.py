import numpy as np
import pytest

from nvf import spectral, synthetic


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0, 2.5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_known_alpha_recovery(alpha, seed):
    img = synthetic.power_law_noise(512, alpha, seed=seed)
    res = spectral.spectral_slope(img)
    assert abs(res.alpha - alpha) < 0.05
    assert res.r2 > 0.95


def test_recovery_on_non_periodic_crop():
    # A crop of a larger image is not periodic; the Hann window must handle the borders.
    big = synthetic.power_law_noise(1024, 1.2, seed=3)
    res = spectral.spectral_slope(big[100:612, 300:812])
    assert abs(res.alpha - 1.2) < 0.1


def test_non_square_image_uses_central_square():
    img = synthetic.power_law_noise(512, 1.0, seed=4)
    wide = np.hstack([np.zeros((512, 100)), img, np.zeros((512, 100))])
    assert spectral.spectral_slope(wide).alpha == pytest.approx(spectral.spectral_slope(img).alpha)


def test_fit_range_defaults():
    res = spectral.spectral_slope(synthetic.power_law_noise(256, 1.0))
    assert res.f_min == 8 and res.f_max == 64


def test_radial_amplitude_of_single_grating():
    n = 128
    x = np.arange(n)
    img = np.tile(np.sin(2 * np.pi * 16 * x / n), (n, 1))
    freqs, amp = spectral.radial_amplitude(img, window=False)
    assert freqs[np.argmax(amp)] == 16


def test_constant_image_gives_nan():
    assert np.isnan(spectral.spectral_slope(np.full((128, 128), 0.3)).alpha)
