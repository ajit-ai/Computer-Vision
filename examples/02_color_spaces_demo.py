"""Color space demo: grayscale, HSV round trip, histogram equalization."""

import numpy as np

from _common import save
from cv_examples import color_spaces, utils


def main():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:128, 0:128]
    rgb[..., 0] = xx * 2
    rgb[..., 1] = 255 - np.abs(yy - xx) * 2
    rgb[..., 2] = yy * 2

    save(color_spaces.rgb_to_gray(rgb), "01_gray.png")

    hsv = color_spaces.rgb_to_hsv(rgb)
    back = (color_spaces.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    save(back, "02_hsv_roundtrip.png")
    print("HSV roundtrip max error:",
          int(np.abs(back.astype(int) - rgb.astype(int)).max()))

    low_contrast = utils.create_synthetic_scene(128, 128).copy()
    low_contrast = (low_contrast // 4 + 160).astype(np.uint8)
    equalized = color_spaces.histogram_equalization(low_contrast)
    save(low_contrast, "03_low_contrast.png")
    save(equalized, "04_equalized.png")


if __name__ == "__main__":
    main()
