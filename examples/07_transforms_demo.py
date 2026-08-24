"""Transforms demo: resize, rotate, translate, affine warp."""

import numpy as np

from _common import save
from cv_examples import transforms, utils


def main():
    scene = utils.create_synthetic_scene(128, 128).astype(float)

    save(transforms.resize_bilinear(scene, 64, 64), "01_resized_down.png")
    save(transforms.resize_bilinear(scene, 256, 256), "02_resized_up.png")
    save(transforms.rotate_image(scene, 30.0), "03_rotated_30.png")
    save(transforms.translate(scene, 20, -10), "04_translated.png")
    save(transforms.flip_horizontal(scene), "05_flipped.png")

    shear = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0]])
    save(transforms.apply_affine(scene, shear), "06_sheared.png")


if __name__ == "__main__":
    main()
