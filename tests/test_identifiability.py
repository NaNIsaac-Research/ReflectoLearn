"""Checks of experiments/001_identifiability/fim.py against references that do not share its code.

The forward model is compared with benchmark.parratt (numpy Parratt recursion) and the autograd
Jacobian with central differences of a numpy re-implementation of the counts model. The Fisher
analysis is checked on its two exact properties: Poisson information is linear in the counting time,
and a parameter that cannot change the curve leaves the information matrix singular.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

FIM_PATH = Path(__file__).parents[1] / "experiments" / "001_identifiability" / "fim.py"


@pytest.fixture(scope="module")
def fim():
    spec = importlib.util.spec_from_file_location("fim_001", FIM_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fim_001"] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules["fim_001"]


def test_forward_model_matches_numpy_parratt(fim):
    assert fim.check_forward_vs_parratt() < 1e-12


def test_autograd_jacobian_matches_finite_differences(fim):
    errors = fim.check_jacobian_vs_fd(tth_max=3.0)
    assert max(errors.values()) < 1e-6, errors


def test_crlb_scales_as_inverse_sqrt_counts(fim):
    q = fim.tth_to_q(fim.tth_grid(3.0))
    p0, free = fim.truth(fim.FILMS[0], "instr"), fim.SCENARIOS["instr"]
    low = fim.fisher(*fim.jacobian(q, p0, free, 1e5), free)
    high = fim.fisher(*fim.jacobian(q, p0, free, 1e7), free)
    np.testing.assert_allclose(low.crlb / high.crlb, 10.0, rtol=1e-9)
    np.testing.assert_allclose(low.corr, high.corr, atol=1e-9)


def test_a_parameter_without_effect_makes_the_information_singular(fim):
    # With no oxide in the stack the oxide SLD does not enter the curve: its column of J is zero.
    q = fim.tth_to_q(fim.tth_grid(3.0))
    p0 = fim.truth(fim.FILMS[0], "film")
    lam, J = fim.jacobian(q, p0, (*fim.FILM, "oxide_sld"), 1e6)
    assert np.all(J[:, 3] == 0)
    F = J.T @ (lam[:, None] * J)
    assert np.linalg.matrix_rank(F) == 3
