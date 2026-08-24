"""Blob detection using the Laplacian-of-Gaussian / Difference-of-Gaussians."""

import numpy as np

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

from .filters import gaussian_blur


def dog(image, sigma=2.0, k=1.6):
    img = np.asarray(image, dtype=np.float64)
    size1 = _odd_size_for_sigma(sigma)
    size2 = _odd_size_for_sigma(sigma * k)
    return gaussian_blur(img, size1, sigma) - gaussian_blur(img, size2, sigma * k)


def _odd_size_for_sigma(sigma):
    return max(3, 2 * int(np.ceil(3.0 * sigma)) + 1)


def detect_blobs(image, sigma=2.0, k=1.6, min_ratio=0.5, neighborhood=5):
    if sliding_window_view is None:
        raise RuntimeError("NumPy >= 1.20 required")
    response = dog(image, sigma=sigma, k=k)
    threshold = min_ratio * float(response.max())
    p = neighborhood // 2
    padded = np.pad(response, p, mode="constant")
    windows = sliding_window_view(padded, (neighborhood, neighborhood))
    local_max = response >= windows.max(axis=(-1, -2))
    blobs_mask = local_max & (response > threshold)
    border = (
        (_odd_size_for_sigma(sigma * k) - 1) // 2
        + neighborhood // 2
        + 1
    )
    blobs_mask[:border, :] = False
    blobs_mask[-border:, :] = False
    blobs_mask[:, :border] = False
    blobs_mask[:, -border:] = False
    ys, xs = np.nonzero(blobs_mask)
    scores = response[ys, xs]
    order = np.argsort(scores)[::-1]
    return [
        {"y": int(ys[o]), "x": int(xs[o]), "score": float(scores[o])}
        for o in order
    ]
