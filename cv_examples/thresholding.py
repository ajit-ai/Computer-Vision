"""Thresholding techniques: Otsu, global, adaptive."""

import numpy as np

from .filters import convolve2d


def otsu_threshold(image):
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("Expected a grayscale image of shape (H, W)")
    flat = np.clip(img, 0, 255).round().astype(np.int64)
    hist = np.bincount(flat.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    levels = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_all = float((hist * levels).sum())
    sum_bg = np.cumsum(hist * levels)
    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.divide(sum_bg, weight_bg, out=np.zeros_like(sum_bg), where=valid)
    mean_fg = np.divide(
        sum_all - sum_bg, weight_fg, out=np.zeros_like(sum_bg), where=valid
    )
    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_var[~valid] = -1.0
    return int(np.argmax(between_var))


def apply_threshold(image, threshold):
    img = np.asarray(image)
    return ((img > threshold).astype(np.uint8)) * 255


def otsu(image):
    return apply_threshold(image, otsu_threshold(image))


def adaptive_threshold(image, block_size=11, c=5.0):
    img = np.asarray(image, dtype=np.float64)
    if block_size < 1 or block_size % 2 == 0:
        raise ValueError("block_size must be a positive odd integer")
    kernel = np.ones((block_size, block_size)) / (block_size * block_size)
    local_mean = convolve2d(img, kernel)
    return ((img > local_mean - c).astype(np.uint8)) * 255
