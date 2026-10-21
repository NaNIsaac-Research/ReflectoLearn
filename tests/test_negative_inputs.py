"""Measured curves built the wrong way must be rejected or flagged, not turned into model input
quietly.

``predict.preprocess_measured`` is the step every measured curve takes. The curve here is a
synthetic Parratt reflectivity on 300 points from 0.01 to 0.4 1/A, scaled to counts. Cases the code
does not handle yet are strict xfails tied to ReflectoLearn#16, so a fix turns them into failures
that prompt removing the marker.
"""

import warnings

import numpy as np
import pytest
import torch

from reflectolearn.benchmark import SI, parratt
from reflectolearn.config import Config
from reflectolearn.predict import preprocess_measured

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def cfg():
    return Config(device="cpu")


@pytest.fixture(scope="module")
def curve():
    q = np.linspace(0.01, 0.4, 300)
    layers = [(0j, 0.0, 0.0), (complex(45, 1), 250.0, 5.0), (SI, 0.0, 3.0)]
    return q, parratt(q, layers) * 1e7


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_a_length_mismatch_is_rejected(cfg, curve):
    q, counts = curve
    with pytest.raises(ValueError):
        preprocess_measured(q, counts[:-1], cfg, CPU)


def test_the_correct_curve_is_accepted_without_warnings(cfg, curve):
    q, counts = curve
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, mask = preprocess_measured(q, counts, cfg, CPU)
    assert mask.sum() > 0.3 * mask.numel()


def test_a_nan_point_is_treated_as_unmeasured(cfg, curve):
    q, counts = curve
    _, _, ref = preprocess_measured(q, counts, cfg, CPU)
    _, _, mask = preprocess_measured(
        q, np.where(np.arange(q.size) == 150, np.nan, counts), cfg, CPU
    )
    assert 0 < ref.sum() - mask.sum() < 10  # only the grid points next to the NaN drop out


@pytest.mark.xfail(strict=True, reason="ReflectoLearn#16: descending q drops the whole curve")
def test_a_descending_scan_gives_the_same_input_as_an_ascending_one(cfg, curve):
    q, counts = curve
    assert _same(
        preprocess_measured(q[::-1], counts[::-1], cfg, CPU),
        preprocess_measured(q, counts, cfg, CPU),
    )


@pytest.mark.xfail(strict=True, reason="ReflectoLearn#16: unsorted q scrambles the curve")
def test_an_unsorted_scan_is_sorted_or_rejected(cfg, curve):
    q, counts = curve
    order = np.random.default_rng(0).permutation(q.size)
    try:
        got = preprocess_measured(q[order], counts[order], cfg, CPU)
    except ValueError:
        return
    assert _same(got, preprocess_measured(q, counts, cfg, CPU))


@pytest.mark.xfail(strict=True, reason="ReflectoLearn#16: swapped arguments pass silently")
def test_swapped_arguments_are_rejected(cfg, curve):
    q, counts = curve
    with pytest.raises(ValueError):
        preprocess_measured(counts, q, cfg, CPU)


@pytest.mark.xfail(
    strict=True, reason="ReflectoLearn#16: no overlap with the q grid passes silently"
)
def test_a_scan_outside_the_q_grid_is_rejected(cfg, curve):
    q, counts = curve
    with pytest.raises(ValueError):
        preprocess_measured(q + 5.0, counts, cfg, CPU)


@pytest.mark.xfail(strict=True, reason="ReflectoLearn#16: negative counts are clipped silently")
def test_negative_counts_are_flagged(cfg, curve):
    q, counts = curve
    with pytest.warns(UserWarning, match="negative"):
        preprocess_measured(q, counts - 0.5 * counts.max(), cfg, CPU)
