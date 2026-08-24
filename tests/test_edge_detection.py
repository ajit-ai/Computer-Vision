import numpy as np

from cv_examples import edge_detection, utils


def make_vertical_step(width=32, height=32):
    img = np.zeros((height, width), dtype=np.float64)
    img[:, width // 2:] = 200.0
    return img


def test_sobel_detects_vertical_edge():
    img = make_vertical_step()
    magnitude, _, gx, gy = edge_detection.sobel_edges(img)
    center = magnitude[16]
    edge_col = int(np.argmax(center))
    assert abs(edge_col - 16) <= 1
    assert center[edge_col] > 400
    assert np.abs(gy[1:-1, 1:-1]).max() < 1e-9


def test_sobel_direction_horizontal_step():
    img = make_vertical_step().T
    magnitude, direction, _, _ = edge_detection.sobel_edges(img)
    interior = magnitude[2:-2, 2:-2]
    di, dj = np.unravel_index(np.argmax(interior), interior.shape)
    i, j = di + 2, dj + 2
    assert abs(abs(direction[i, j]) - np.pi / 2) < 0.05


def test_laplacian_zero_crossing_at_edge():
    img = make_vertical_step()
    log_img = edge_detection.laplacian_of_gaussian(img, size=5, sigma=1.0)
    crossings = edge_detection.zero_crossings(log_img, threshold=1.0)
    rows_with_crossing = np.nonzero(crossings.any(axis=0))[0]
    assert rows_with_crossing.size > 0
    assert abs(int(rows_with_crossing.mean()) - 16) <= 3


def test_canny_finds_thin_square_outline():
    scene = utils.create_synthetic_scene(64, 64)
    edges = edge_detection.canny(scene, low=40, high=100, sigma=1.0, size=5)
    edges_bool = edges > 0
    assert edges_bool.any()
    interior = edges[20:30, 20:30] > 0
    assert not interior.any()


def test_canny_nms_thins_edges():
    img = make_vertical_step()
    edges = edge_detection.canny(img, low=40, high=120, sigma=1.0)
    row = edges[16] > 0
    runs = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
    starts = np.nonzero(runs == 1)[0]
    ends = np.nonzero(runs == -1)[0]
    widths = ends - starts
    assert (widths <= 2).all()


def test_hysteresis_links_weak_edges():
    img = make_vertical_step()
    mag, direction, _, _ = edge_detection.sobel_edges(
        edge_detection.gaussian_blur(img, 5, 1.0)
    )
    nms = edge_detection._non_max_suppression(mag, direction)
    strong, weak = edge_detection._double_threshold(nms, 40.0, 500.0)
    linked = edge_detection._hysteresis(strong, weak) > 0
    raw_strong = strong.copy()
    assert linked.sum() >= raw_strong.sum()
