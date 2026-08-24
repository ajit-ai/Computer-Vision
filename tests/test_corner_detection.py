import numpy as np

from cv_examples import corner_detection, utils


def checkerboard_with_known_corners(rows=6, cols=6, cell=12):
    board = utils.create_checkerboard(rows + 2, cols + 2, cell).astype(np.float64)
    corners = [
        (cell * r, cell * c)
        for r in range(1, rows + 2)
        for c in range(1, cols + 2)
    ]
    return board, corners


def test_harris_finds_checkerboard_corners():
    board, expected = checkerboard_with_known_corners()
    points, scores = corner_detection.harris_corners(
        board, window=5, threshold_ratio=0.05
    )
    assert points.size > 0
    hits = 0
    for ey, ex in expected:
        distances = np.hypot(points[:, 0] - ey, points[:, 1] - ex)
        if distances.min() <= 3:
            hits += 1
    assert hits >= len(expected) * 0.7


def test_scores_sorted_descending():
    board, _ = checkerboard_with_known_corners()
    _, scores = corner_detection.harris_corners(board, threshold_ratio=0.02)
    assert scores.size == 0 or np.all(np.diff(scores) <= 1e-9)


def test_flat_image_no_corners():
    flat = np.full((32, 32), 100.0)
    points, scores = corner_detection.harris_corners(flat, threshold_ratio=0.01)
    assert points.shape[0] == 0


def test_single_square_corner_detected():
    img = np.zeros((40, 40))
    img[10:30, 10:30] = 255.0
    points, _ = corner_detection.harris_corners(img, window=5, threshold_ratio=0.05)
    expected = [(10, 10), (10, 29), (29, 10), (29, 29)]
    hits = sum(
        1
        for ey, ex in expected
        if np.any(np.hypot(points[:, 0] - ey, points[:, 1] - ex) <= 4)
    ) if points.size else 0
    assert hits >= 3
