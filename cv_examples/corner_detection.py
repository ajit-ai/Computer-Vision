"""Harris corner detection."""

import numpy as np

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

from .edge_detection import SOBEL_X, SOBEL_Y
from .filters import convolve2d, gaussian_blur


def harris_response(image, k=0.04, size=3, sigma=1.0):
    img = np.asarray(image, dtype=np.float64)
    ix = convolve2d(img, SOBEL_X)
    iy = convolve2d(img, SOBEL_Y)
    ixx = gaussian_blur(ix * ix, size, sigma)
    iyy = gaussian_blur(iy * iy, size, sigma)
    ixy = gaussian_blur(ix * iy, size, sigma)
    det = ixx * iyy - ixy ** 2
    trace = ixx + iyy
    return det - k * trace ** 2


def harris_corners(image, k=0.04, size=3, sigma=1.0, window=5, threshold_ratio=0.05):
    if sliding_window_view is None:
        raise RuntimeError("NumPy >= 1.20 required")
    response = harris_response(image, k=k, size=size, sigma=sigma)
    threshold = threshold_ratio * float(response.max())
    p = window // 2
    padded = np.pad(response, p, mode="constant")
    windows = sliding_window_view(padded, (window, window))
    local_max = response >= windows.max(axis=(-1, -2))
    corners = (response > threshold) & local_max
    border = size + window
    corners[:border, :] = False
    corners[-border:, :] = False
    corners[:, :border] = False
    corners[:, -border:] = False
    ys, xs = np.nonzero(corners)
    scores = response[ys, xs]
    order = np.argsort(scores)[::-1]
    return np.stack([ys[order], xs[order]], axis=1), scores[order]
