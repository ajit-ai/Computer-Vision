import numpy as np

from cv_examples import morphology


def cross_selem():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def test_erosion_shrinks_region():
    mask = np.zeros((11, 11), dtype=np.uint8)
    mask[3:8, 3:8] = 255
    eroded = morphology.erosion(mask)
    assert eroded.sum() < mask.sum()
    assert eroded[4:7, 4:7].all()


def test_dilation_grows_region():
    mask = np.zeros((11, 11), dtype=np.uint8)
    mask[5, 5] = 255
    dilated = morphology.dilation(mask)
    assert dilated.sum() > mask.sum()
    assert dilated[5, 6] == 255 and dilated[4, 5] == 255


def test_opening_removes_speckles():
    mask = np.zeros((21, 21), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask[18, 18] = 255
    opened = morphology.opening(mask)
    assert opened[18, 18] == 0
    assert opened[9, 9] == 255


def test_closing_fills_small_hole():
    mask = np.full((15, 15), 255, dtype=np.uint8)
    mask[7, 7] = 0
    closed = morphology.closing(mask)
    assert closed[7, 7] == 255


def test_opening_is_idempotent():
    rng = np.random.default_rng(5)
    mask = (rng.random((32, 32)) > 0.5).astype(np.uint8) * 255
    once = morphology.opening(mask)
    twice = morphology.opening(once)
    assert np.array_equal(once, twice)


def test_custom_selem_cross():
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[4, 3] = 255
    mask[4, 5] = 255
    eroded = morphology.erosion(mask, cross_selem())
    assert eroded[4, 4] == 0
    solid = np.full((9, 9), 255, dtype=np.uint8)
    eroded_solid = morphology.erosion(solid, cross_selem())
    assert eroded_solid[4, 4] == 255
    assert eroded_solid[0, 0] == 0


def test_morphological_gradient():
    mask = np.zeros((13, 13), dtype=np.uint8)
    mask[4:9, 4:9] = 255
    grad = morphology.morphological_gradient(mask)
    assert grad[6, 6] == 0
    assert grad[4, 6] > 0


def test_empty_mask_stays_empty():
    empty = np.zeros((5, 5), dtype=np.uint8)
    assert morphology.dilation(empty).sum() == 0


def test_grayscale_min_max_filters():
    img = np.arange(81, dtype=np.float64).reshape(9, 9)
    eroded = morphology.grayscale_erode(img, 3)
    dilated = morphology.grayscale_dilate(img, 3)
    assert eroded[4, 4] == img[3, 3]
    assert dilated[4, 4] == img[5, 5]
    assert np.all(eroded <= img) and np.all(dilated >= img)
