"""Spatial filtering: convolution, smoothing, sharpening, median."""

import numpy as np

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None


def convolve2d(image, kernel):
    img = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim == 1:
        k = k.reshape(1, -1)
    flipped = np.flip(k)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += flipped[i, j] * padded[i:i + h, j:j + w]
    return out


def box_kernel(size):
    if size < 1 or size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer")
    return np.ones((size, size), dtype=np.float64) / (size * size)


def gaussian_kernel1d(size, sigma):
    if size < 1 or size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    center = size // 2
    xs = np.arange(size) - center
    kernel = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(image, size=5, sigma=1.0):
    k = gaussian_kernel1d(size, sigma)
    out = convolve2d(image, k.reshape(1, -1))
    return convolve2d(out, k.reshape(-1, 1))


def box_blur(image, size=3):
    return convolve2d(image, box_kernel(size))


def median_filter(image, size=3):
    if sliding_window_view is None:
        raise RuntimeError("NumPy >= 1.20 required for median_filter")
    img = np.asarray(image, dtype=np.float64)
    p = size // 2
    padded = np.pad(img, p, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-1, -2))


def sharpen(image, amount=1.0):
    img = np.asarray(image, dtype=np.float64)
    blurred = box_blur(img, 3)
    return img + amount * (img - blurred)


def unsharp_mask(image, size=5, sigma=1.0, amount=1.5):
    img = np.asarray(image, dtype=np.float64)
    blurred = gaussian_blur(img, size, sigma)
    return img + amount * (img - blurred)
