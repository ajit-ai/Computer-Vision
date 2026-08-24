"""Edge detection demo: Sobel magnitude, LoG zero crossings, Canny."""

from _common import save
from cv_examples import edge_detection, utils


def main():
    scene = utils.create_synthetic_scene(128, 128)

    magnitude, direction, gx, gy = edge_detection.sobel_edges(scene.astype(float))
    log_img = edge_detection.laplacian_of_gaussian(scene.astype(float), 9, 1.5)
    crossings = edge_detection.zero_crossings(log_img, threshold=2.0)
    canny_edges = edge_detection.canny(scene, low=40, high=120, sigma=1.4)

    save(magnitude, "01_sobel_magnitude.png")
    save(crossings, "02_log_zero_crossings.png")
    save(canny_edges, "03_canny.png")


if __name__ == "__main__":
    main()
