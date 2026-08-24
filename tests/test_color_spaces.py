import numpy as np
import pytest

from cv_examples import color_spaces, utils


def test_rgb_to_gray_weights():
    rgb = np.zeros((2, 2, 3), dtype=np.float64)
    rgb[..., 1] = 255.0
    gray = color_spaces.rgb_to_gray(rgb)
    assert np.allclose(gray, 0.587 * 255)


def test_rgb_to_gray_rejects_2d():
    with pytest.raises(ValueError):
        color_spaces.rgb_to_gray(np.zeros((4, 4)))


def test_rgb_to_hsv_pure_colors():
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    rgb[0, 1] = [0, 255, 0]
    rgb[0, 2] = [0, 0, 255]
    hsv = color_spaces.rgb_to_hsv(rgb)[0]
    assert np.isclose(hsv[0, 0], 0.0)
    assert np.isclose(hsv[1, 0] % 1.0, 1 / 3, atol=1e-6)
    assert np.isclose(hsv[2, 0] % 1.0, 2 / 3, atol=1e-6)
    for i in range(3):
        assert np.isclose(hsv[i, 1], 1.0)
        assert np.isclose(hsv[i, 2], 1.0)


def test_gray_pixel_has_zero_saturation():
    rgb = np.full((1, 1, 3), 120, dtype=np.uint8)
    hsv = color_spaces.rgb_to_hsv(rgb)
    assert np.isclose(hsv[0, 0, 1], 0.0)


def test_hsv_roundtrip():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (8, 8, 3)).astype(np.uint8)
    hsv = color_spaces.rgb_to_hsv(rgb)
    back = color_spaces.hsv_to_rgb(hsv) * 255.0
    assert np.allclose(back, rgb.astype(np.float64), atol=1)


def test_ycbcr_gray_is_neutral():
    gray_img = np.full((2, 2, 3), 128.0)
    ycbcr = color_spaces.rgb_to_ycbcr(gray_img)
    assert np.allclose(ycbcr[..., 1], 128.0, atol=0.5)
    assert np.allclose(ycbcr[..., 2], 128.0, atol=0.5)


def test_histogram_equalization_expands_range():
    low_contrast = np.full((64, 64), 100, dtype=np.uint8)
    low_contrast[16:48, 16:48] = 110
    equalized = color_spaces.histogram_equalization(low_contrast)
    assert int(equalized.min()) < int(low_contrast.min())
    assert int(equalized.max()) > int(low_contrast.max())


def test_histogram_equalization_constant_image():
    flat = np.full((8, 8), 42, dtype=np.uint8)
    out = color_spaces.histogram_equalization(flat)
    assert np.array_equal(out, flat)


def test_equalization_matches_reference():
    img = utils.create_synthetic_scene(32, 32)
    ours = color_spaces.histogram_equalization(img)
    reference = cv_equalize(img)
    assert np.array_equal(ours, reference)


def cv_equalize(img):
    hist = np.bincount(img.ravel(), minlength=256)
    nonzero = np.nonzero(hist)[0]
    cdf = hist.cumsum()
    cdf_min = cdf[nonzero[0]]
    denom = img.size - cdf_min
    lut = np.round((cdf - cdf_min) / denom * 255.0)
    return lut[img].astype(np.uint8)
