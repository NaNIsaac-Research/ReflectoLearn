"""Regression tests for the refine stages on the benchmark's independent synthetic curves.

The curves come from ``reflectolearn.benchmark.make_synthetic``: a numpy float64 Parratt recursion,
not the package's torch engine, with footprint, resolution, background and Poisson counts.
Refine runs on CUDA when available, else CPU (about a minute).
"""

import numpy as np
import pytest
import torch

from reflectolearn.benchmark import SI, SIO2, make_synthetic, parratt
from reflectolearn.config import Config
from reflectolearn.physics import AbelesMatrix, tth2q
from reflectolearn.predict import preprocess_measured
from reflectolearn.refine import fft_thickness, refine_multistart

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = Config()
WL = CFG.simulation.wavelength


def observed(case_tth_counts):
    case, tth, counts = case_tth_counts
    q_meas = tth2q(tth, WL)
    _, r_obs, mask = preprocess_measured(q_meas, counts, CFG, DEV)
    return case, q_meas, torch.log10(r_obs.clamp_min(1e-15)), mask


def test_reference_parratt_matches_abeles():
    q = np.linspace(0.005, 0.5, 400)
    film = complex(45.0, 2.0)
    ref = parratt(q, [(0j, 0.0, 0.0), (film, 250.0, 6.0), (SIO2, 15.0, 3.0), (SI, 0.0, 4.0)])
    got = AbelesMatrix()(
        torch.tensor(q)[None],
        torch.tensor([[0.0, 250.0, 15.0]]),
        torch.tensor([[6.0, 3.0, 4.0]]),
        torch.tensor([[0j, film, SIO2, SI]], dtype=torch.complex64),
    )[0].double().numpy()
    np.testing.assert_allclose(np.log10(got), np.log10(ref), atol=2e-3)


def test_fft_thickness_finds_the_fringe_period():
    for c in make_synthetic(6, seed=11, wavelength=WL):
        case, _, log_r, mask = observed(c)
        rough_sld = torch.tensor([0.9 * case.sld], device=DEV)  # the network's SLD is about this good
        d = float(fft_thickness(CFG.q_grid.to(DEV), log_r, mask, CFG.param_ranges.thickness, sld=rough_sld)[0])
        assert abs(d / case.thickness - 1) < 0.06, (case.name, case.thickness, d)


@pytest.fixture(scope="module")
def double_start():
    """case_01 of seed 11 refined from twice its thickness: refine alone stays near 2d there."""
    case, q_meas, log_r, mask = observed(make_synthetic(2, seed=11, wavelength=WL)[1])
    p0 = torch.tensor([[2 * case.thickness, case.roughness, case.sld]], device=DEV)
    sub = torch.tensor([[20.0, 3.5]], device=DEV)
    q = CFG.q_grid.to(DEV)
    with torch.enable_grad():
        p, info = refine_multistart(p0, sub, q, log_r, mask, CFG.param_ranges, wavelength=WL, device=DEV)
    inside = int(((q_meas >= float(q[0])) & (q_meas <= float(q[-1]))).sum())
    return case, p, info, (sub, q, log_r, mask, inside)


def test_alias_check_recovers_from_a_double_start(double_start):
    case, p, info, _ = double_start
    assert abs(float(p[0, 0]) / case.thickness - 1) < 0.05, info["alias"]


def test_mcmc_interval_brackets_the_truth(double_start):
    pytest.importorskip("emcee")
    from reflectolearn.refine.mcmc import sample_posterior

    case, _, info, (sub, q, log_r, mask, inside) = double_start
    post = sample_posterior(info, sub, q, log_r, mask, wavelength=WL, n_independent=inside, steps=1000, device=DEV)
    d = post["params"]["thickness"]
    assert d["lo"] < d["median"] < d["hi"]
    assert abs(d["median"] / case.thickness - 1) < 0.03
    assert d["hi"] - d["lo"] < 0.05 * case.thickness
    assert np.isfinite(post["lnsigma"]["median"])
    assert 0.1 < post["acceptance"] < 0.7
