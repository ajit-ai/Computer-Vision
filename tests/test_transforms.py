import numpy as np
import pytest

from cv_examples import transforms, utils


def test_resize_same_size_is_identity():
    img = utils.create_synthetic_scene(32, 32).astype(np.float64)
    out = transforms.resize_bilinear(img, 32, 32)
    assert np.allclose(out, img, atol=1e-9)


def test_resize_downsample_upsample_approximates():
    img = utils.create_synthetic_scene(64, 64).astype(np.float64)
    small = transforms.resize_bilinear(img, 16, 16)
    back = transforms.resize_bilinear(small, 64, 64)
    mse = np.mean((back - img) ** 2)
    assert small.shape == (16, 16)
    assert mse < 900.0


def test_resize_rejects_invalid_size():
    with pytest.raises(ValueError):
        transforms.resize_bilinear(np.zeros((4, 4)), 0, 8)


def test_rotate_90_matches_rot90():
    img = utils.create_synthetic_scene(24, 24).astype(np.float64)
    rotated = transforms.rotate_image(img, 90.0)
    expected = np.rot90(img, k=-1)
    assert np.allclose(rotated, np.asarray(expected), atol=1)


def test_rotate_360_identity():
    img = utils.create_synthetic_scene(24, 24).astype(np.float64)
    rotated = transforms.rotate_image(img, 360.0)
    assert np.allclose(rotated, img, atol=1)


def test_rotate_keeps_center_fixed():
    img = utils.create_synthetic_scene(25, 25).astype(np.float64)
    rotated = transforms.rotate_image(img, 45.0)
    assert abs(rotated[12, 12] - img[12, 12]) < 5


def test_translate_positive_shift():
    img = np.zeros((10, 10))
    img[2, 2] = 100
    shifted = transforms.translate(img, dx=3, dy=1)
    y, x = np.unravel_index(np.argmax(shifted), shifted.shape)
    assert (y, x) == (3, 5)
    assert shifted.sum() == 100


def test_translate_negative_shift():
    img = np.zeros((10, 10))
    img[7, 7] = 100
    shifted = transforms.translate(img, dx=-2, dy=-3)
    y, x = np.unravel_index(np.argmax(shifted), shifted.shape)
    assert (y, x) == (4, 5)


def test_flips():
    img = np.arange(20, dtype=np.float64).reshape(4, 5)
    assert np.array_equal(transforms.flip_horizontal(img), np.fliplr(img))
    assert np.array_equal(transforms.flip_vertical(img), np.flipud(img))


def test_affine_translation_matrix():
    img = np.zeros((12, 12))
    img[3, 3] = 100
    matrix = np.array([[1.0, 0.0, -2.0], [0.0, 1.0, -4.0]])
    out = transforms.apply_affine(img, matrix)
    y, x = np.unravel_index(np.argmax(out), out.shape)
    assert (y, x) == (7, 5)


def test_apply_affine_validates_shape():
    with pytest.raises(ValueError):
        transforms.apply_affine(np.zeros((4, 4)), np.eye(3))
