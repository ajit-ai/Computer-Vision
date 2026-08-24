"""Morphology demo: erosion, dilation, opening, closing, gradient."""

import numpy as np

from _common import save
from cv_examples import morphology, thresholding, utils


def main():
    scene = utils.create_synthetic_scene(128, 128)
    binary = thresholding.otsu(scene)

    save(binary, "01_binary.png")
    save(morphology.erosion(binary), "02_eroded.png")
    save(morphology.dilation(binary), "03_dilated.png")
    save(morphology.opening(binary), "04_opened.png")
    save(morphology.closing(binary), "05_closed.png")
    save(morphology.morphological_gradient(binary), "06_gradient.png")


if __name__ == "__main__":
    main()
