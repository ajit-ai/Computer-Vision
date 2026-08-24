"""HOG descriptor demo on synthetic textures."""

import numpy as np

from _common import save
from cv_examples import hog


def main():
    yy, xx = np.mgrid[0:128, 0:128]
    vertical = ((xx // 8) % 2) * 255.0
    diagonal = (((xx + yy) // 8) % 2) * 255.0

    for name, img in [("vertical", vertical), ("diagonal", diagonal)]:
        features = hog.hog_features(img, cell_size=16, orientations=9)
        bin_idx, magnitude = hog.dominant_orientation(features)
        print(
            f"{name}: dominant orientation bin {bin_idx} "
            f"(~{bin_idx * 20:.0f} deg), energy {magnitude:.0f}"
        )

    save(vertical, "01_vertical_stripes.png")
    save(diagonal, "02_diagonal_stripes.png")


if __name__ == "__main__":
    main()
