"""Template matching via normalized cross-correlation."""

import numpy as np


def match_template_ncc(image, template):
    img = np.asarray(image, dtype=np.float64)
    tpl = np.asarray(template, dtype=np.float64)
    if tpl.ndim != 2 or img.ndim != 2:
        raise ValueError("Expected grayscale images")
    th, tw = tpl.shape
    if th > img.shape[0] or tw > img.shape[1]:
        raise ValueError("Template must not exceed image dimensions")
    tpl_zero = tpl - tpl.mean()
    tpl_norm = np.sqrt((tpl_zero ** 2).sum())
    out_h = img.shape[0] - th + 1
    out_w = img.shape[1] - tw + 1
    scores = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            window = img[i:i + th, j:j + tw]
            win_zero = window - window.mean()
            denom = np.sqrt((win_zero ** 2).sum()) * tpl_norm
            if denom > 1e-12:
                scores[i, j] = float((win_zero * tpl_zero).sum() / denom)
    return scores


def find_best_match(scores):
    idx = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(idx[0]), int(idx[1]), float(scores[idx])
