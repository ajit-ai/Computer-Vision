import numpy as np

from cv_examples import blob_detection


def make_disk_image(size=80, radius=4, center=(40, 40)):
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.hypot(yy - center[0], xx - center[1])
    return (dist < radius).astype(np.float64) * 255.0


def test_dog_response_positive_inside_bright_blob():
    disk = make_disk_image(radius=12)
    response = blob_detection.dog(disk, sigma=2.0, k=1.6)
    assert response[48, 48] > 0
    assert response.max() > 0


def test_detect_blobs_locates_disk_center():
    disk = make_disk_image(size=64, radius=4, center=(32, 32))
    blobs = blob_detection.detect_blobs(disk, sigma=2.0, min_ratio=0.3)
    assert blobs
    best = blobs[0]
    assert abs(best["y"] - 32) <= 2
    assert abs(best["x"] - 32) <= 2


def test_multiple_disks_detected():
    img = np.zeros((80, 80))
    for cy, cx in [(25, 25), (55, 55)]:
        disk = make_disk_image(size=80, radius=4, center=(cy, cx))
        img = np.maximum(img, disk)
    blobs = blob_detection.detect_blobs(img, sigma=2.0, min_ratio=0.35)
    centers = [(b["y"], b["x"]) for b in blobs[:6]]
    near_first = any(abs(y - 25) <= 4 and abs(x - 25) <= 4 for y, x in centers)
    near_second = any(abs(y - 55) <= 4 and abs(x - 55) <= 4 for y, x in centers)
    assert near_first and near_second


def test_flat_image_has_no_blobs():
    flat = np.full((48, 48), 100.0)
    blobs = blob_detection.detect_blobs(flat, sigma=2.0)
    assert blobs == []


def test_scores_sorted_descending():
    disk = make_disk_image(radius=5)
    blobs = blob_detection.detect_blobs(disk, sigma=2.0, min_ratio=0.3)
    scores = [b["score"] for b in blobs]
    assert scores == sorted(scores, reverse=True)


def test_large_disk_center_with_matching_sigma():
    disk = make_disk_image(size=96, radius=12, center=(48, 48))
    blobs = blob_detection.detect_blobs(disk, sigma=7.0, min_ratio=0.4)
    assert blobs
    assert abs(blobs[0]["y"] - 48) <= 3
    assert abs(blobs[0]["x"] - 48) <= 3
