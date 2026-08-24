import numpy as np

from cv_examples import connected_components


def two_squares_mask():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[3:10, 3:10] = 255
    mask[20:30, 25:35] = 255
    return mask


def test_two_disjoint_squares():
    labels = connected_components.label_components(two_squares_mask(), connectivity=8)
    stats = connected_components.component_stats(labels)
    assert labels.max() == 2
    areas = sorted(s["area"] for s in stats.values())
    assert areas == [49, 100]


def test_bounding_boxes():
    labels = connected_components.label_components(two_squares_mask())
    stats = connected_components.component_stats(labels)
    bboxes = {s["bbox"] for s in stats.values()}
    assert (3, 3, 9, 9) in bboxes
    assert (20, 25, 29, 34) in bboxes


def test_diagonal_touching_4_vs_8_connectivity():
    mask = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
    )
    labels_8 = connected_components.label_components(mask, connectivity=8)
    labels_4 = connected_components.label_components(mask, connectivity=4)
    assert labels_8.max() == 1
    assert labels_4.max() == 3


def test_empty_and_full_masks():
    empty = np.zeros((8, 8), dtype=np.uint8)
    full = np.ones((8, 8), dtype=np.uint8) * 255
    assert connected_components.label_components(empty).max() == 0
    labels_full = connected_components.label_components(full)
    assert labels_full.max() == 1


def test_centroids():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    labels = connected_components.label_components(mask)
    stats = connected_components.component_stats(labels)
    centroid = list(stats.values())[0]["centroid"]
    assert abs(centroid[0] - 9.5) < 1e-6
    assert abs(centroid[1] - 9.5) < 1e-6
