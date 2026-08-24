import numpy as np

from cv_examples import template_matching, utils


def make_scene_with_patch():
    scene = utils.create_synthetic_scene(64, 64).astype(np.float64)
    rng = np.random.default_rng(11)
    patch = rng.integers(0, 256, (10, 10)).astype(np.float64)
    location = (30, 40)
    scene[location[0]:location[0] + 10, location[1]:location[1] + 10] = patch
    return scene, patch, location


def test_exact_template_found():
    scene, patch, (r, c) = make_scene_with_patch()
    scores = template_matching.match_template_ncc(scene, patch)
    best_r, best_c, score = template_matching.find_best_match(scores)
    assert (best_r, best_c) == (r, c)
    assert score > 0.999


def test_score_map_shape():
    scene, patch, _ = make_scene_with_patch()
    scores = template_matching.match_template_ncc(scene, patch)
    assert scores.shape == (55, 55)


def test_scores_within_unit_range():
    scene, patch, _ = make_scene_with_patch()
    scores = template_matching.match_template_ncc(scene, patch)
    assert scores.max() <= 1.0 + 1e-9
    assert scores.min() >= -1.0 - 1e-9


def test_constant_template_gives_zero_scores():
    scene, _, _ = make_scene_with_patch()
    flat_patch = np.full((8, 8), 100.0)
    scores = template_matching.match_template_ncc(scene, flat_patch)
    assert np.allclose(scores, 0.0)


def test_template_larger_than_image_raises():
    with np.testing.assert_raises(ValueError):
        template_matching.match_template_ncc(
            np.zeros((8, 8)), np.zeros((9, 9))
        )
