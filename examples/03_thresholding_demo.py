"""Thresholding demo: global, Otsu, adaptive under uneven illumination."""

import numpy as np

from _common import save
from cv_examples import thresholding, utils


def main():
    scene = utils.create_synthetic_scene(128, 128).astype(np.float64)

    otsu_t = thresholding.otsu_threshold(scene.astype(np.uint8))
    print(f"Otsu threshold on synthetic scene: {otsu_t}")

    yy = np.linspace(0.4, 1.6, 128)[:, None]
    uneven = np.clip(scene * yy, 0, 255).astype(np.uint8)
    otsu_uneven = thresholding.otsu_threshold(uneven)

    save(scene, "01_original.png")
    save(thresholding.apply_threshold(uneven, 100), "02_global_t100.png")
    save(thresholding.otsu(uneven), "03_otsu.png")
    save(
        thresholding.adaptive_threshold(uneven, block_size=15, c=5),
        "04_adaptive.png",
    )
    print("Global vs Otsu vs Adaptive saved for comparison")


if __name__ == "__main__":
    main()
