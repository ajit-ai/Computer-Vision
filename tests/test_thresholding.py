import numpy as np
import pytest

from cv_examples import thresholding


def test_otsu_threshold_within_mode_range():
    img = np.full((32, 32), 50, dtype=np.uint8)
    img[16:, :] = 180
    t = thresholding.otsu_threshold(img)
    assert 50 <= t < 180


def test_otsu_perfectly_segments_two_modes():
    img = np.full((40, 40), 50, dtype=np.uint8)
    img[10:30, 10:30] = 180
    truth = img == 180
    binary = thresholding.otsu(img) > 0
    assert np.array_equal(binary, truth)


def test_otsu_bimodal_segmentation_accuracy():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 60, (48, 48)).astype(np.uint8)
    img[8:24, 8:24] = rng.integers(200, 256, (16, 16)).astype(np.uint8)
    truth = img >= 200
    binary = thresholding.otsu(img) > 0
    accuracy = (binary == truth).mean()
    assert accuracy > 0.99


def test_apply_threshold_strictly_greater():
    img = np.array([[10, 50], [50, 90]], dtype=np.uint8)
    out = thresholding.apply_threshold(img, 50)
    expected = np.array([[0, 0], [0, 255]], dtype=np.uint8)
    assert np.array_equal(out, expected)


def test_constant_image_returns_valid_threshold():
    flat = np.full((16, 16), 77, dtype=np.uint8)
    t = thresholding.otsu_threshold(flat)
    assert 0 <= t <= 255


def test_rejects_non_grayscale():
    with pytest.raises(ValueError):
        thresholding.otsu_threshold(np.zeros((4, 4, 3), dtype=np.uint8))


def test_adaptive_finds_dark_stripe_under_illumination_gradient():
    yy = np.linspace(40, 220, 64)[:, None]
    img = np.tile(yy, (1, 64))
    img[:, 30:33] *= 0.5
    img = img.astype(np.uint8)

    adaptive = thresholding.adaptive_threshold(img, block_size=11, c=8)
    stripe_is_background = (adaptive[:, 30:33] == 0).mean()
    surround_is_foreground = (adaptive[:, 20:25] == 255).mean()
    assert stripe_is_background > 0.9
    assert surround_is_foreground > 0.95


def test_adaptive_validates_block_size():
    with pytest.raises(ValueError):
        thresholding.adaptive_threshold(np.zeros((16, 16)), block_size=4)


def test_adaptive_uniform_region_is_foreground():
    flat = np.full((32, 32), 100, dtype=np.uint8)
    out = thresholding.adaptive_threshold(flat, block_size=7, c=5)
    assert (out == 255).mean() > 0.9
