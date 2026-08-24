"""Dense Lucas-Kanade optical flow on a pair of grayscale frames."""

import numpy as np

from .filters import convolve2d, gaussian_blur


def _box_sum(image, size):
    kernel = np.ones((size, size), dtype=np.float64)
    return convolve2d(image, kernel)


def lucas_kanade_dense(frame1, frame2, window=15, smooth_sigma=1.5):
    i1 = gaussian_blur(np.asarray(frame1, dtype=np.float64), 5, smooth_sigma)
    i2 = gaussian_blur(np.asarray(frame2, dtype=np.float64), 5, smooth_sigma)
    it = i2 - i1
    iy, ix = np.gradient(i1)
    sxx = _box_sum(ix * ix, window)
    syy = _box_sum(iy * iy, window)
    sxy = _box_sum(ix * iy, window)
    sxt = _box_sum(ix * it, window)
    syt = _box_sum(iy * it, window)
    det = sxx * syy - sxy ** 2
    reliable = np.abs(det) > 1e-6
    u = np.zeros_like(sxx)
    v = np.zeros_like(sxx)
    u[reliable] = (
        -(syy[reliable] * sxt[reliable] - sxy[reliable] * syt[reliable])
        / det[reliable]
    )
    v[reliable] = (
        -(sxx[reliable] * syt[reliable] - sxy[reliable] * sxt[reliable])
        / det[reliable]
    )
    return u, v
