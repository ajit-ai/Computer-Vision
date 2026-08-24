"""Mathematical morphology for binary and grayscale images."""

import numpy as np

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None


def _default_selem(size):
    return np.ones((size, size), dtype=bool)


def _reduce_shifts(binary, selem, combine):
    ph = selem.shape[0] // 2
    pw = selem.shape[1] // 2
    padded = np.pad(
        binary, ((ph, ph), (pw, pw)), mode="constant", constant_values=False
    )
    h, w = binary.shape
    result = None
    for i in range(selem.shape[0]):
        for j in range(selem.shape[1]):
            if not selem[i, j]:
                continue
            view = padded[i:i + h, j:j + w]
            result = view.copy() if result is None else combine(result, view)
    return result if result is not None else np.zeros_like(binary)


def erosion(mask, selem=None):
    binary = np.asarray(mask) > 0
    selem = _default_selem(3) if selem is None else np.asarray(selem, dtype=bool)
    result = _reduce_shifts(binary, selem, np.logical_and)
    return (result.astype(np.uint8)) * 255


def dilation(mask, selem=None):
    binary = np.asarray(mask) > 0
    selem = _default_selem(3) if selem is None else np.asarray(selem, dtype=bool)
    result = _reduce_shifts(binary, selem, np.logical_or)
    return (result.astype(np.uint8)) * 255


def opening(mask, selem=None):
    return dilation(erosion(mask, selem), selem)


def closing(mask, selem=None):
    return erosion(dilation(mask, selem), selem)


def morphological_gradient(mask, selem=None):
    dil = dilation(mask, selem).astype(np.int16)
    ero = erosion(mask, selem).astype(np.int16)
    return (dil - ero).astype(np.uint8)


def grayscale_erode(image, size=3):
    if sliding_window_view is None:
        raise RuntimeError("NumPy >= 1.20 required")
    img = np.asarray(image, dtype=np.float64)
    p = size // 2
    padded = np.pad(img, p, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    return windows.min(axis=(-1, -2))


def grayscale_dilate(image, size=3):
    if sliding_window_view is None:
        raise RuntimeError("NumPy >= 1.20 required")
    img = np.asarray(image, dtype=np.float64)
    p = size // 2
    padded = np.pad(img, p, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    return windows.max(axis=(-1, -2))
