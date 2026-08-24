import numpy as np

from cv_examples import utils


def test_to_grayscale_rgb():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    gray = utils.to_grayscale(rgb)
    assert np.allclose(gray, 0.299 * 255)


def test_to_grayscale_passthrough():
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert np.array_equal(utils.to_grayscale(img), img)


def test_normalize_range():
    img = np.array([[10, 20], [30, 40]], dtype=np.float64)
    norm = utils.normalize(img)
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)


def test_scale_to_uint8_roundtrip():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (8, 8)).astype(np.uint8)
    out = utils.scale_to_uint8(img)
    assert out.dtype == np.uint8
    assert np.allclose(out.astype(np.float64), img.astype(np.float64), atol=1)


def test_checkerboard_values():
    board = utils.create_checkerboard(rows=2, cols=2, cell=4)
    assert board.shape == (8, 8)
    assert set(np.unique(board)) == {0, 255}
    assert board[0, 0] == 0
    assert board[0, -1] == 255


def test_synthetic_scene_regions():
    scene = utils.create_synthetic_scene(height=64, width=64)
    assert scene.shape == (64, 64)
    assert scene[20, 20] == 200
    assert scene[40, 40] == 120
    assert scene[10, 60] == 60


def test_salt_pepper_noise_extremes():
    flat = np.full((50, 50), 128, dtype=np.uint8)
    noisy = utils.add_salt_pepper_noise(flat, amount=0.05, seed=42)
    changed = noisy != 128
    assert changed.sum() > 0
    assert set(np.unique(noisy[changed])) <= {0, 255}


def test_gaussian_noise_statistics():
    flat = np.full((100, 100), 128, dtype=np.float64)
    noisy = utils.add_gaussian_noise(flat, mean=0.0, std=10.0, seed=7)
    assert abs(noisy.mean() - 128) < 3
    assert abs(noisy.std() - 10) < 3


def test_pad():
    img = np.ones((2, 2))
    padded = utils.pad(img, 1)
    assert padded.shape == (4, 4)
    assert padded[0, 0] == 0
    assert padded[1, 1] == 1
