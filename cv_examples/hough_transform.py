"""Hough transform for line detection."""

import numpy as np


def hough_lines(edge_map, num_theta=180, threshold=None, min_distance=10, num_peaks=10):
    edges = np.asarray(edge_map) > 0
    h, w = edges.shape
    diag = int(np.ceil(np.hypot(h, w)))
    rhos = np.arange(-diag, diag + 1)
    thetas = np.linspace(0.0, np.pi, num_theta, endpoint=False)
    accumulator = np.zeros((rhos.size, thetas.size), dtype=np.int64)
    ys, xs = np.nonzero(edges)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    rho_idx = np.round(
        xs[:, None] * cos_t[None, :] + ys[:, None] * sin_t[None, :]
    ).astype(np.int64) + diag
    for t in range(num_theta):
        vals, counts = np.unique(rho_idx[:, t], return_counts=True)
        accumulator[vals, t] += counts

    max_votes = int(accumulator.max())
    if threshold is None:
        threshold = max(1, int(max_votes * 0.5))

    flat_order = np.argsort(accumulator, axis=None)[::-1]
    peaks = []
    for flat_idx in flat_order:
        votes = int(accumulator.flat[flat_idx])
        if votes < threshold or len(peaks) >= num_peaks:
            break
        ri, ti = np.unravel_index(int(flat_idx), accumulator.shape)
        duplicate = any(
            abs(ri - pri) < min_distance and abs(ti - pti) < min_distance
            for _, _, _, pri, pti in peaks
        )
        if not duplicate:
            peaks.append(
                (float(rhos[ri]), float(np.rad2deg(thetas[ti])), votes, ri, ti)
            )
    return [
        {"rho": rho, "theta_deg": theta_deg, "votes": votes}
        for rho, theta_deg, votes, _, _ in peaks
    ]
