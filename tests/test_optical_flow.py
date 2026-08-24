import numpy as np

from cv_examples import optical_flow, utils


def shifted_pair(shift=3.0):
    base = utils.create_checkerboard(rows=6, cols=6, cell=16).astype(np.float64)
    base = utils.add_gaussian_noise(base, std=0.0)
    cols = np.arange(base.shape[1])
    shifted = np.zeros_like(base)
    x0 = int(np.floor(shift))
    frac = shift - x0
    if frac < 1e-9:
        shifted[:, x0:] = base[:, :base.shape[1] - x0]
    else:
        shifted[:, x0:] = (
            base[:, :base.shape[1] - x0] * (1 - frac)
            + base[:, :base.shape[1] - x0 - 1] * frac
        )
    return base, shifted


def test_horizontal_shift_estimated():
    base, moved = shifted_pair(2.0)
    u, v = optical_flow.lucas_kanade_dense(
        base, moved, window=15, smooth_sigma=2.0
    )
    h, w = u.shape
    center = u[h // 3:2 * h // 3, w // 2:]
    reliable = center[np.abs(center) > 0.05]
    assert reliable.size > 50
    median_u = float(np.median(reliable))
    assert abs(median_u - 2.0) < 1.0


def test_static_pair_gives_near_zero_flow():
    img = utils.create_checkerboard(6, 6, 16).astype(np.float64)
    u, v = optical_flow.lucas_kanade_dense(img, img, window=15)
    center_u = np.abs(u[u.shape[0] // 2 - 16:u.shape[0] // 2 + 16,
                        u.shape[1] // 2:u.shape[1] // 2 + 16])
    assert np.median(center_u) < 0.5


def test_output_shapes_and_finiteness():
    base, moved = shifted_pair(2.0)
    u, v = optical_flow.lucas_kanade_dense(base, moved, window=11)
    assert u.shape == base.shape
    assert v.shape == base.shape
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
