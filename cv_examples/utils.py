"""Image utilities: conversions, normalization, synthetic data, noise."""

import numpy as np


def to_grayscale(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        return img.copy()
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Expected an image with shape (H, W) or (H, W, 3)")
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def normalize(image):
    img = np.asarray(image, dtype=np.float64)
    lo = float(img.min())
    hi = float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def scale_to_uint8(image):
    return (np.clip(normalize(image), 0.0, 1.0) * 255).round().astype(np.uint8)


def create_checkerboard(rows=8, cols=8, cell=16):
    r_idx = np.arange(rows * cell) // cell
    c_idx = np.arange(cols * cell) // cell
    board = ((r_idx[:, None] + c_idx[None, :]) % 2) * 255
    return board.astype(np.uint8)


def create_synthetic_scene(height=128, width=128):
    img = np.zeros((height, width), dtype=np.uint8)
    ch, cw = max(height // 4, 1), max(width // 4, 1)
    img[ch:2 * ch, cw:2 * cw] = 200
    img[height // 2:height // 2 + ch, width // 2:width // 2 + cw] = 120
    img[:ch, width - cw:] = 60
    return img


def add_salt_pepper_noise(image, amount=0.01, seed=None):
    rng = np.random.default_rng(seed)
    out = np.asarray(image, dtype=np.uint8).copy()
    count = int(amount * out.shape[0] * out.shape[1])
    rows = rng.integers(0, out.shape[0], count)
    cols = rng.integers(0, out.shape[1], count)
    values = rng.choice([0, 255], count)
    out[rows, cols] = values
    return out


def add_gaussian_noise(image, mean=0.0, std=10.0, seed=None):
    rng = np.random.default_rng(seed)
    img = np.asarray(image, dtype=np.float64)
    noise = rng.normal(mean, std, img.shape)
    return np.clip(img + noise, 0.0, 255.0)


def pad(image, pad_width, mode="constant"):
    return np.pad(np.asarray(image), pad_width, mode=mode)
