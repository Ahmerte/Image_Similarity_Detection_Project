import numpy as np
import pytest
from skimage import data
import cv2

from similarity import compare_images, mse_score, phash_score
from evaluation.make_dataset import rotate, scale, crop, jpeg


@pytest.fixture(scope="module")
def astronaut():
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="module")
def coffee():
    return cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)


def test_identical_scores_high(astronaut):
    res = compare_images(astronaut, astronaut)
    assert res.aligned
    assert res.score > 90


@pytest.mark.parametrize("fn", [
    lambda im: rotate(im, 25),
    lambda im: scale(im, 0.5),
    lambda im: crop(im, 0.7),
    lambda im: jpeg(im, 20),
])
def test_transformed_copy_is_similar(astronaut, fn):
    res = compare_images(astronaut, fn(astronaut))
    assert res.aligned
    assert res.score > 50


def test_unrelated_scores_low(astronaut, coffee):
    res = compare_images(astronaut, coffee)
    assert res.score < 20


def test_fallback_on_blank_image(astronaut):
    blank = np.full_like(astronaut, 128)
    res = compare_images(astronaut, blank)
    assert not res.aligned
    assert res.score <= 20


def test_baselines_range(astronaut, coffee):
    for f in (mse_score, phash_score):
        assert 0 <= f(astronaut, coffee) <= 100
        assert f(astronaut, astronaut) == pytest.approx(100)
