"""Edge detection: Sobel, Laplacian, LoG zero crossings, Canny."""

import numpy as np

from .filters import gaussian_blur, convolve2d

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
LAPLACIAN_8 = np.array(
    [[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64
)


def sobel_edges(image):
    img = np.asarray(image, dtype=np.float64)
    gx = convolve2d(img, SOBEL_X)
    gy = convolve2d(img, SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return magnitude, direction, gx, gy


def laplacian(image, kernel=LAPLACIAN_4):
    return convolve2d(np.asarray(image, dtype=np.float64), kernel)


def laplacian_of_gaussian(image, size=9, sigma=1.5):
    img = np.asarray(image, dtype=np.float64)
    return laplacian(gaussian_blur(img, size, sigma))


def zero_crossings(log_image, threshold=1e-6):
    log_img = np.asarray(log_image, dtype=np.float64)
    h, w = log_img.shape
    out = np.zeros((h, w), dtype=np.uint8)
    pos_right = log_img[:, :-1] * log_img[:, 1:] < 0
    diff_right = np.abs(log_img[:, :-1] - log_img[:, 1:])
    out[:, :-1] |= ((pos_right) & (diff_right > threshold)).astype(np.uint8) * 255
    pos_down = log_img[:-1, :] * log_img[1:, :] < 0
    diff_down = np.abs(log_img[:-1, :] - log_img[1:, :])
    out[:-1, :] |= ((pos_down) & (diff_down > threshold)).astype(np.uint8) * 255
    return out * 255


def _non_max_suppression(magnitude, direction):
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float64)
    angle = (np.rad2deg(direction) % 180.0).round()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            if a == 0:
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif a == 45:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif a == 90:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            m = magnitude[i, j]
            if m >= n1 and m >= n2:
                nms[i, j] = m
    return nms


def _double_threshold(nms, low, high):
    strong = nms >= high
    weak = (nms >= low) & ~strong
    return strong, weak


def _dilate_cross(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _hysteresis(strong, weak):
    result = strong.astype(bool).copy()
    weak_mask = weak.astype(bool)
    while True:
        grown = _dilate_cross(result) & weak_mask
        candidate = result | grown
        if int(candidate.sum()) == int(result.sum()):
            break
        result = candidate
    return (result.astype(np.uint8)) * 255


def canny(image, low=50.0, high=150.0, sigma=1.4, size=5):
    img = np.asarray(image, dtype=np.float64)
    smoothed = gaussian_blur(img, size, sigma)
    magnitude, direction, _, _ = sobel_edges(smoothed)
    nms = _non_max_suppression(magnitude, direction)
    strong, weak = _double_threshold(nms, float(low), float(high))
    return _hysteresis(strong, weak)
