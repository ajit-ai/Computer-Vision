"""Lucas-Kanade optical flow demo on a translated checkerboard."""

import numpy as np

from _common import save  # noqa: F401  (bootstraps sys.path)
from cv_examples import optical_flow, utils


def main():
    base = utils.create_checkerboard(8, 8, 16).astype(np.float64)

    shifted = np.zeros_like(base)
    dx = 4
    shifted[:, dx:] = base[:, :base.shape[1] - dx]

    u, v = optical_flow.lucas_kanade_dense(base, shifted, window=15)

    h, w = u.shape
    center = u[h // 3:2 * h // 3, w // 2:]
    reliable = center[np.abs(center) > 0.05]
    print(f"true horizontal shift: {dx}px")
    if reliable.size:
        print(f"estimated median u in right half: {np.median(reliable):.2f}px")
    print(f"median v overall (should be ~0): {np.median(np.abs(v)):.2f}px")


if __name__ == "__main__":
    main()
