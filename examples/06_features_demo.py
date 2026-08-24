"""Feature demos: Harris corners, Hough lines, blobs, template matching."""

import numpy as np

from _common import save
from cv_examples import (
    blob_detection,
    corner_detection,
    hough_transform,
    template_matching,
    transforms,
    utils,
)


def main():
    scene = utils.create_checkerboard(8, 8, 16)

    points, scores = corner_detection.harris_corners(
        scene.astype(float), window=5, threshold_ratio=0.05
    )
    print(f"Harris corners found: {len(scores)}")

    edges = utils.create_synthetic_scene(96, 96)
    diag = np.eye(96, dtype=np.uint8) * 255
    edges = np.maximum(edges, diag)
    lines = hough_transform.hough_lines(edges, num_theta=180)
    for line in lines[:3]:
        print(
            f"line rho={line['rho']:.1f} theta={line['theta_deg']:.1f}deg "
            f"votes={line['votes']}"
        )

    yy, xx = np.mgrid[0:96, 0:96]
    disk = (np.hypot(yy - 48, xx - 48) < 12).astype(np.float64) * 255
    blobs = blob_detection.detect_blobs(disk, sigma=7.0, min_ratio=0.4)
    if blobs:
        print(f"strongest blob at y={blobs[0]['y']} x={blobs[0]['x']}")

    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (96, 96)).astype(np.float64)
    patch = base[30:50, 60:80].copy()
    moved = transforms.translate(base, 6, -4)
    score_map = template_matching.match_template_ncc(moved, patch)
    r, c, s = template_matching.find_best_match(score_map)
    print(f"template found at ({r}, {c}) score={s:.4f} (expected 26, 66)")

    save(disk, "01_disk_for_blobs.png")
    save(edges, "02_scene_with_line.png")


if __name__ == "__main__":
    main()
