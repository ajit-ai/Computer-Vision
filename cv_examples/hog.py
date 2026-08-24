"""Histogram of Oriented Gradients (cell-based HOG descriptor)."""

import numpy as np


def hog_features(image, cell_size=8, orientations=9):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Expected a grayscale image of shape (H, W)")
    if cell_size < 1 or orientations < 1:
        raise ValueError("cell_size and orientations must be positive")
    gy, gx = np.gradient(img)
    magnitude = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    h, w = img.shape
    cells_y = h // cell_size
    cells_x = w // cell_size
    bin_width = 180.0 / orientations
    bins = np.clip((angle / bin_width).astype(np.int64), 0, orientations - 1)
    features = np.zeros((cells_y, cells_x, orientations), dtype=np.float64)
    for cy in range(cells_y):
        for cx in range(cells_x):
            ys = slice(cy * cell_size, (cy + 1) * cell_size)
            xs = slice(cx * cell_size, (cx + 1) * cell_size)
            cell_mag = magnitude[ys, xs]
            cell_bins = bins[ys, xs]
            for o in range(orientations):
                features[cy, cx, o] = cell_mag[cell_bins == o].sum()
    return features


def normalize_blocks(features, block_size=2, eps=1e-6):
    cy, cx, o = features.shape
    out = np.zeros_like(features)
    counts = np.zeros((cy, cx), dtype=np.float64)
    for by in range(0, cy - block_size + 1):
        for bx in range(0, cx - block_size + 1):
            block = features[by:by + block_size, bx:bx + block_size].ravel()
            norm = np.sqrt((block ** 2).sum()) + eps
            out[by:by + block_size, bx:bx + block_size] += (
                block / norm
            ).reshape(block_size, block_size, o)
            counts[by:by + block_size, bx:bx + block_size] += 1
    return out / np.maximum(counts, 1)[..., None]


def dominant_orientation(features):
    summed = features.sum(axis=(0, 1))
    return int(np.argmax(summed)), float(summed.max())
