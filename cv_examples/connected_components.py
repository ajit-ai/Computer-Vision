"""Connected-component labeling and region statistics."""

import numpy as np
from collections import deque


def _neighborhood(connectivity):
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
    raise ValueError("connectivity must be 4 or 8")


def label_components(mask, connectivity=8):
    binary = np.asarray(mask) > 0
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    neighbors = _neighborhood(connectivity)
    current = 0
    for start_i in range(h):
        for start_j in range(w):
            if not binary[start_i, start_j] or labels[start_i, start_j] != 0:
                continue
            current += 1
            labels[start_i, start_j] = current
            queue = deque([(start_i, start_j)])
            while queue:
                y, x = queue.popleft()
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and binary[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = current
                        queue.append((ny, nx))
    return labels


def component_stats(labels):
    ids = np.unique(labels)
    ids = ids[ids > 0]
    stats = {}
    for cid in ids:
        ys, xs = np.nonzero(labels == cid)
        stats[int(cid)] = {
            "area": int(ys.size),
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
            "centroid": (float(ys.mean()), float(xs.mean())),
        }
    return stats
