"""Filtering demo: box blur, Gaussian blur, median denoising, sharpening."""

import numpy as np

from _common import save
from cv_examples import filters, utils


def main():
    scene = utils.create_synthetic_scene(128, 128)
    noisy = utils.add_gaussian_noise(scene, std=25.0, seed=1)

    save(scene, "01_original.png")
    save(filters.box_blur(scene.astype(float), 7), "02_box_blur.png")
    save(filters.gaussian_blur(noisy, 7, 2.0), "03_gaussian_denoise.png")

    sp_noisy = utils.add_salt_pepper_noise(scene, amount=0.05, seed=2)
    save(sp_noisy, "04_salt_pepper.png")
    save(filters.median_filter(sp_noisy, 3), "05_median_denoise.png")
    save(filters.sharpen(scene.astype(float), 1.0), "06_sharpened.png")


if __name__ == "__main__":
    main()
