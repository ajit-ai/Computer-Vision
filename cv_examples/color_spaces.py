"""Color space conversions and histogram equalization."""

import numpy as np


def rgb_to_gray(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Expected an RGB image of shape (H, W, 3)")
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def rgb_to_hsv(image):
    img = np.clip(np.asarray(image, dtype=np.float64) / 255.0, 0.0, 1.0)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.max(img, axis=-1)
    minc = np.min(img, axis=-1)
    v = maxc
    delta = maxc - minc
    delta_safe = np.maximum(delta, 1e-12)
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-12), 0.0)
    rc = (maxc - r) / delta_safe
    gc = (maxc - g) / delta_safe
    bc = (maxc - b) / delta_safe
    h = np.select(
        [maxc == r, maxc == g],
        [bc - gc, 2.0 + rc - bc],
        default=4.0 + gc - rc,
    )
    h = (h / 6.0) % 1.0
    h = np.where(delta <= 0, 0.0, h)
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv):
    hsv = np.clip(np.asarray(hsv, dtype=np.float64), 0.0, 1.0)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    idx = i.astype(np.int64) % 6
    conds = [idx == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def rgb_to_ycbcr(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Expected an RGB image of shape (H, W, 3)")
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 16.0 + 65.481 * r / 255.0 + 128.553 * g / 255.0 + 24.966 * b / 255.0
    cb = 128.0 - 37.797 * r / 255.0 - 74.203 * g / 255.0 + 112.0 * b / 255.0
    cr = 128.0 + 112.0 * r / 255.0 - 93.786 * g / 255.0 - 18.214 * b / 255.0
    return np.stack([y, cb, cr], axis=-1)


def histogram_equalization(image):
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("Expected a grayscale image of shape (H, W)")
    lo, hi = int(img.min()), int(img.max())
    if lo == hi:
        return img.copy()
    flat = np.clip(img, 0, 255).round().astype(np.int64)
    hist = np.bincount(flat.ravel(), minlength=256)
    cdf = hist.cumsum().astype(np.float64)
    cdf_min = cdf[np.nonzero(hist)[0][0]]
    total = flat.size
    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[flat].reshape(img.shape)
