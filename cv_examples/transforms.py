"""Geometric transforms: resize, rotate, translate, flips."""

import numpy as np


def _sample_bilinear(img, ys, xs, fill=0.0):
    h, w = img.shape[0], img.shape[1]
    tol = 1e-6
    valid = (
        (ys >= -tol) & (ys <= h - 1 + tol) & (xs >= -tol) & (xs <= w - 1 + tol)
    )
    ys_c = np.clip(ys, 0, h - 1)
    xs_c = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys_c).astype(np.int64)
    x0 = np.floor(xs_c).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys_c - y0)[..., None]
    wx = (xs_c - x0)[..., None]
    top = img[y0, x0] * (1.0 - wx) + img[y0, x1] * wx
    bottom = img[y1, x0] * (1.0 - wx) + img[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.where(valid[..., None], sampled, np.asarray(fill, dtype=img.dtype))


def resize_bilinear(image, new_height, new_width):
    img = np.asarray(image, dtype=np.float64)
    if new_height < 1 or new_width < 1:
        raise ValueError("Target size must be positive")
    h, w = img.shape[:2]
    row_ratio = (h - 1) / max(new_height - 1, 1)
    col_ratio = (w - 1) / max(new_width - 1, 1)
    rr = np.arange(new_height) * row_ratio
    cc = np.arange(new_width) * col_ratio
    if img.ndim == 2:
        img3 = img[..., None]
        return _sample_bilinear(img3, rr[:, None], cc[None, :])[..., 0]
    return _sample_bilinear(img, rr[:, None], cc[None, :])


def rotate_image(image, angle_degrees, fill=0.0):
    img = np.asarray(image, dtype=np.float64)
    single = img.ndim == 2
    img3 = img[..., None] if single else img
    h, w = img3.shape[:2]
    theta = np.deg2rad(angle_degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    xr = xx - cx
    yr = yy - cy
    src_x = cos_t * xr + sin_t * yr + cx
    src_y = -sin_t * xr + cos_t * yr + cy
    out = _sample_bilinear(img3, src_y, src_x, fill=fill)
    return out[..., 0] if single else out


def translate(image, dx, dy, fill=0.0):
    img = np.asarray(image, dtype=np.float64)
    out = np.full_like(img, fill)
    h, w = img.shape[:2]
    ys_src = slice(max(0, -dy), min(h, h - dy))
    xs_src = slice(max(0, -dx), min(w, w - dx))
    ys_dst = slice(max(0, dy), min(h, h + dy))
    xs_dst = slice(max(0, dx), min(w, w + dx))
    out[ys_dst, xs_dst] = img[ys_src, xs_src]
    return out


def flip_horizontal(image):
    return np.fliplr(np.asarray(image)).copy()


def flip_vertical(image):
    return np.flipud(np.asarray(image)).copy()


def apply_affine(image, matrix, fill=0.0):
    img = np.asarray(image, dtype=np.float64)
    single = img.ndim == 2
    img3 = img[..., None] if single else img
    h, w = img3.shape[:2]
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3):
        raise ValueError("Affine matrix must have shape (2, 3)")
    yy, xx = np.mgrid[0:h, 0:w]
    dst = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    src = m @ dst
    src_x = src[0].reshape(h, w)
    src_y = src[1].reshape(h, w)
    out = _sample_bilinear(img3, src_y, src_x, fill=fill)
    return out[..., 0] if single else out
