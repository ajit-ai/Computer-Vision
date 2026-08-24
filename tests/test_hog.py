import numpy as np

from cv_examples import hog


def vertical_stripes(width=64, height=64, period=8):
    x = np.arange(width)
    row = np.where((x // (period // 2)) % 2 == 0, 255.0, 0.0)
    return np.tile(row, (height, 1))


def test_output_shape():
    img = vertical_stripes()
    features = hog.hog_features(img, cell_size=16, orientations=9)
    assert features.shape == (4, 4, 9)


def test_vertical_stripes_peak_at_bin_zero():
    img = vertical_stripes()
    features = hog.hog_features(img, cell_size=8, orientations=9)
    dominant_bin, _ = hog.dominant_orientation(features)
    assert dominant_bin == 0


def test_horizontal_gradient_peaks_at_90_degrees():
    yy = np.arange(64)[:, None]
    img = np.tile(yy.astype(np.float64), (1, 64)) * 4.0
    img = np.clip(img, 0, 255)
    features = hog.hog_features(img, cell_size=16, orientations=9)
    dominant_bin, _ = hog.dominant_orientation(features)
    bin_width = 180.0 / 9
    assert abs(dominant_bin * bin_width - 90.0) <= bin_width


def test_features_zero_for_flat_image():
    flat = np.full((32, 32), 77.0)
    features = hog.hog_features(flat, cell_size=8, orientations=9)
    assert np.allclose(features, 0.0)


def test_block_normalization_bounds():
    img = vertical_stripes()
    features = hog.hog_features(img, cell_size=8, orientations=9)
    normalized = hog.normalize_blocks(features, block_size=2)
    max_val = normalized.max()
    assert 0 < max_val <= 1.0 + 1e-6


def test_dominant_orientation_values():
    img = vertical_stripes()
    features = hog.hog_features(img, cell_size=16)
    bin_idx, magnitude = hog.dominant_orientation(features)
    assert 0 <= bin_idx < 9
    assert magnitude > 0
