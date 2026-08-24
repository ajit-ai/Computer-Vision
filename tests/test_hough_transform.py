import numpy as np

from cv_examples import hough_transform


def make_line_image(size=64, intercept=20):
    img = np.zeros((size, size), dtype=np.uint8)
    for x in range(size):
        y = intercept - x
        if 0 <= y < size:
            img[y, x] = 255
            if y + 1 < size:
                img[y + 1, x] = 255
    return img


def make_vertical_line_image(size=64, col=30):
    img = np.zeros((size, size), dtype=np.uint8)
    img[:, col] = 255
    return img


def test_detects_diagonal_line():
    img = make_line_image()
    lines = hough_transform.hough_lines(img, num_theta=180, min_distance=10)
    assert lines
    top = lines[0]
    angle = abs(top["theta_deg"])
    expected_angle = 45.0
    deviation = min(angle % 90, 90 - angle % 90)
    assert deviation < 3.0 or abs(angle - expected_angle) < 3.0


def test_vertical_line_theta_near_zero_or_180():
    img = make_vertical_line_image(col=30)
    lines = hough_transform.hough_lines(img, num_theta=180)
    assert lines
    theta = lines[0]["theta_deg"]
    assert theta < 2.0 or theta > 178.0


def test_votes_match_line_length():
    img = make_vertical_line_image()
    lines = hough_transform.hough_lines(img, num_theta=180)
    assert lines[0]["votes"] == 64


def test_rho_matches_column():
    img = make_vertical_line_image(col=40)
    lines = hough_transform.hough_lines(img, num_theta=180)
    assert abs(lines[0]["rho"] - 40) <= 1


def test_empty_image_returns_no_peaks():
    empty = np.zeros((32, 32), dtype=np.uint8)
    lines = hough_transform.hough_lines(empty)
    assert lines == []
