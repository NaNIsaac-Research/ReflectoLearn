"""Measured curves built the wrong way must be rejected or flagged, not turned into model input
quietly.

``predict.preprocess_measured`` is the step every measured curve takes. The curve here is a
synthetic Parratt reflectivity on 300 points from 0.01 to 0.4 1/A, scaled to counts. The cases
below were found silently wrong in ReflectoLearn#16.
"""

import warnings

import numpy as np
import pytest
import torch

from reflectolearn.benchmark import SI, parratt
from reflectolearn.config import Config
from reflectolearn.data.generator import LOG_FLOOR
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


def test_a_descending_scan_gives_the_same_input_as_an_ascending_one(cfg, curve):
    q, counts = curve
    assert _same(
        preprocess_measured(q[::-1], counts[::-1], cfg, CPU),
        preprocess_measured(q, counts, cfg, CPU),
    )


def test_an_unsorted_scan_is_sorted_or_rejected(cfg, curve):
    q, counts = curve
    order = np.random.default_rng(0).permutation(q.size)
    try:
        got = preprocess_measured(q[order], counts[order], cfg, CPU)
    except ValueError:
        return
    assert _same(got, preprocess_measured(q, counts, cfg, CPU))


def test_swapped_arguments_are_rejected(cfg, curve):
    q, counts = curve
    with pytest.raises(ValueError):
        preprocess_measured(counts, q, cfg, CPU)


def test_a_scan_outside_the_q_grid_is_rejected(cfg, curve):
    q, counts = curve
    with pytest.raises(ValueError):
        preprocess_measured(q + 5.0, counts, cfg, CPU)


def test_negative_counts_are_flagged(cfg, curve):
    q, counts = curve
    with pytest.warns(UserWarning, match="negative"):
        preprocess_measured(q, counts - 0.5 * counts.max(), cfg, CPU)


def _preprocess_before_the_option(q, counts, cfg):
    """preprocess_measured as it was before the negative option, step for step."""
    q_grid = cfg.q_grid.cpu().numpy()
    i_grid = np.interp(q_grid, q, counts, left=np.nan, right=np.nan)
    mask = np.isfinite(i_grid)
    i_grid = np.clip(np.nan_to_num(i_grid, nan=0.0), 0.0, None)
    imax = i_grid[mask].max() if mask.any() else 1.0
    r_norm = np.clip(i_grid / max(imax, 1e-30), 1e-15, None)
    r_log = np.where(mask, np.clip(np.log10(r_norm), None, 0.0), LOG_FLOOR)
    x = np.stack([r_log, mask.astype(np.float32)])[None]

    def to(a):
        return torch.tensor(a, dtype=torch.float32)

    return to(x), to(r_norm).unsqueeze(0), to(mask.astype(np.float32)).unsqueeze(0)


@pytest.fixture(scope="module")
def below_zero(curve):
    """The curve after an over-subtracted background: the high-q tail goes negative."""
    q, counts = curve
    return q, counts - 1e-4 * counts.max()


def test_default_negative_handling_is_unchanged(cfg, below_zero):
    q, counts = below_zero
    assert (counts < 0).sum() > 50
    with pytest.warns(UserWarning, match="clipped to 0"):
        got = preprocess_measured(q, counts, cfg, CPU)
    assert _same(got, _preprocess_before_the_option(q, counts, cfg))


def test_negative_mask_marks_only_the_negative_grid_points_unmeasured(cfg, below_zero):
    q, counts = below_zero
    with pytest.warns(UserWarning, match="left out as unmeasured"):
        x, r_obs, mask = preprocess_measured(q, counts, cfg, CPU, negative="mask")
    with pytest.warns(UserWarning):
        _, r_floor, mask_floor = preprocess_measured(q, counts, cfg, CPU)
    i_grid = np.interp(cfg.q_grid.cpu().numpy(), q, counts, left=np.nan, right=np.nan)
    below = torch.from_numpy(i_grid < 0)
    assert below.any()
    assert torch.equal(mask[0].bool(), mask_floor[0].bool() & ~below)
    assert torch.equal(x[0, 0][below], torch.full((int(below.sum()),), LOG_FLOOR))
    assert torch.equal(r_obs, r_floor)


def test_negative_error_refuses_the_curve(cfg, below_zero):
    with pytest.raises(ValueError, match="negative"):
        preprocess_measured(*below_zero, cfg, CPU, negative="error")


def test_an_unknown_negative_mode_is_refused(cfg, curve):
    with pytest.raises(ValueError, match="negative must be one of"):
        preprocess_measured(*curve, cfg, CPU, negative="clip")


@pytest.mark.parametrize("mode", ["floor", "mask"])
def test_the_negative_step_is_recorded(cfg, below_zero, mode):
    q, counts = below_zero
    record = {}
    with pytest.warns(UserWarning):
        preprocess_measured(q, counts, cfg, CPU, negative=mode, record=record)
    assert record == {
        "negative": mode,
        "negative_points": int((counts < 0).sum()),
        "points": q.size,
        "floor": 1e-15,
    }


def test_the_input_arrays_are_not_modified(cfg, below_zero):
    q, counts = below_zero
    before = q.copy(), counts.copy()
    with pytest.warns(UserWarning):
        preprocess_measured(q, counts, cfg, CPU, negative="mask")
    assert np.array_equal(q, before[0]) and np.array_equal(counts, before[1])


def test_repeated_q_values_are_averaged(cfg, curve):
    """Overlapping segments of a multi-range scan repeat q; the points are averaged."""
    q, counts = curve
    twice = np.r_[q[:200], q[150:]], np.r_[counts[:200], 3 * counts[150:]]
    with pytest.warns(UserWarning, match="50 points share a q value"):
        got = preprocess_measured(*twice, cfg, CPU)
    merged = np.r_[counts[:150], 2 * counts[150:200], 3 * counts[200:]]
    assert _same(got, preprocess_measured(q, merged, cfg, CPU))


def test_swapped_arguments_with_normalised_reflectivity_are_rejected(cfg, curve):
    """R already in 0-1 overlaps the q grid, so the range check alone cannot catch the swap."""
    q, counts = curve
    with pytest.raises(ValueError, match="argument order"):
        preprocess_measured(counts / counts.max(), q, cfg, CPU)
