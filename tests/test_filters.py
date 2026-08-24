import numpy as np
import pytest

from cv_examples import filters, utils


def test_convolve_identity():
    img = utils.create_synthetic_scene(32, 32).astype(np.float64)
    identity = np.zeros((3, 3))
    identity[1, 1] = 1.0
    assert np.allclose(filters.convolve2d(img, identity), img)


def test_convolve_box_on_constant_interior():
    constant = np.full((16, 16), 7.0)
    out = filters.convolve2d(constant, filters.box_kernel(3))
    assert np.allclose(out[2:-2, 2:-2], 7.0)


def test_convolve_shifts_signal():
    img = np.zeros((9, 9))
    img[4, 4] = 100.0
    kernel = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    out = filters.convolve2d(img, kernel)
    peak = np.unravel_index(np.argmax(out), out.shape)
    assert peak == (4, 5)


def test_gaussian_kernel_properties():
    k = filters.gaussian_kernel1d(7, 1.5)
    assert np.isclose(k.sum(), 1.0)
    assert np.allclose(k, k[::-1])
    assert np.all(k >= 0)


def test_invalid_kernel_sizes():
    with pytest.raises(ValueError):
        filters.box_kernel(2)
    with pytest.raises(ValueError):
        filters.gaussian_kernel1d(0, 1.0)


def test_gaussian_blur_reduces_noise_energy():
    rng = np.random.default_rng(1)
    noise = rng.normal(128, 30, (64, 64))
    blurred = filters.gaussian_blur(noise, 5, 1.5)
    assert blurred.std() < noise.std()


def test_box_blur_preserves_mean_interior():
    img = utils.create_checkerboard(8, 8, 8).astype(np.float64)
    blurred = filters.box_blur(img, 5)
    interior_img = img[8:-8, 8:-8]
    interior_out = blurred[8:-8, 8:-8]
    assert abs(interior_out.mean() - interior_img.mean()) < 15.0


def test_median_filter_removes_impulse():
    img = np.full((11, 11), 50.0)
    img[5, 5] = 255.0
    filtered = filters.median_filter(img, 3)
    assert filtered[5, 5] == 50.0


def test_sharpen_increases_contrast():
    ramp = np.tile(np.linspace(0, 255, 32), (32, 1))
    sharpened = filters.sharpen(ramp, amount=1.0)
    assert sharpened.max() - sharpened.min() > ramp.max() - ramp.min()
