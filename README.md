# Computer-Vision

Classical computer vision algorithms implemented **from scratch in NumPy** — no
OpenCV required. Every algorithm ships with runnable demo scripts (`examples/`)
and a pytest suite (`tests/`).

## Algorithms

| Module | Features / Algorithms |
| --- | --- |
| `cv_examples/filters.py` | 2D convolution, box & Gaussian blur, median filter, sharpen, unsharp mask |
| `cv_examples/color_spaces.py` | RGB→Gray, RGB↔HSV, RGB→YCbCr, histogram equalization |
| `cv_examples/thresholding.py` | Otsu's method, global threshold, adaptive (local mean) thresholding |
| `cv_examples/morphology.py` | Erosion, dilation, opening, closing, morphological gradient, grayscale min/max filters |
| `cv_examples/edge_detection.py` | Sobel operator, Laplacian, Laplacian-of-Gaussian zero crossings, full Canny (NMS + hysteresis) |
| `cv_examples/corner_detection.py` | Harris corner detector with non-maximum suppression |
| `cv_examples/blob_detection.py` | Difference-of-Gaussians blob detection with local-max selection |
| `cv_examples/connected_components.py` | BFS component labeling (4/8 connectivity), area/bbox/centroid statistics |
| `cv_examples/transforms.py` | Bilinear resize, rotation (inverse mapping), translation, flips, affine warp |
| `cv_examples/template_matching.py` | Normalized cross-correlation template matching |
| `cv_examples/hough_transform.py` | Hough line transform with peak extraction & duplicate suppression |
| `cv_examples/hog.py` | Histogram of Oriented Gradients (cell histograms + block normalization) |
| `cv_examples/optical_flow.py` | Dense Lucas–Kanade optical flow |

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest            # run all tests
python examples/01_filters_demo.py   # any demo writes PNGs to examples/outputs/
```

Demos need Pillow for saving images; the library itself only needs NumPy.

## Project layout

```
cv_examples/     # algorithm library (importable package)
examples/        # self-contained runnable demos per topic
tests/           # pytest suites, one per module
pyproject.toml   # packaging + pytest config
requirements.txt # dev/demo dependencies
```

## Testing

```bash
python -m pytest              # quick run
python -m pytest --cov=cv_examples   # with coverage (pytest-cov)
```

Each module has a dedicated test file covering correctness on synthetic images
with known ground truth (exact rotations, known line angles, planted corners,
known template positions, etc.).

## Roadmap

- SIFT-like scale-space keypoint descriptors
- RANSAC feature matching / image stitching
- Active contours (snakes), watershed segmentation
- Camera calibration & epipolar geometry utilities

## License

MIT — see [LICENSE](LICENSE).
