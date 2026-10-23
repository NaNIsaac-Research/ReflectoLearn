"""The fit-policy environment (docs/fit_policy/DESIGN.md): the stack model against independent
engines, the fit-quality numbers, action masks and layer edits, and a reproducible episode."""

import math

import numpy as np
import pytest
import torch

from reflectolearn.benchmark import parratt
from reflectolearn.fit_policy import A1, FitEnv, Stack, run_protocol
from reflectolearn.fit_policy import stack as st
from reflectolearn.refine import RefineConfig, make_forward

TRILAYER = Stack(
    np.array([[80.0, 4.0, 45.0], [150.0, 5.0, 12.0], [40.0, 3.0, 60.0]]),
    np.array([1.1, -7.5, 0.004, 0.0]),
)


def test_stack_matches_parratt():
    q = st.q_grid()
    with torch.no_grad():
        r = st.reflectivity(torch.as_tensor(q), torch.as_tensor(TRILAYER.layers), 0.0).numpy()
    layers = [(0j, 0.0, 0.0)]
    layers += [(complex(rho), d, sigma) for d, sigma, rho in TRILAYER.layers]
    layers.append((complex(st.SUB_SLD), 0.0, st.SUB_ROUGH))
    assert np.max(np.abs(r / parratt(q, layers) - 1)) < 1e-12


def test_one_layer_matches_refine_forward():
    """Same film, scale, background and offset through refine's forward model (complex64, so a
    loose tolerance), with the oxide, Kα2, footprint and resolution off in both."""
    q = st.q_grid()
    one = Stack(np.array([[240.0, 6.0, 45.0]]), np.array([1.2, -7.0, 0.01, 0.0]))
    config = RefineConfig(fit_kalpha2=False, fit_footprint=False, fit_oxide=False, fit_dq=False)
    simulate = make_forward(
        torch.as_tensor(q, dtype=torch.float32),
        torch.tensor([[st.SUB_SLD, st.SUB_ROUGH]]),
        wavelength=st.WAVELENGTH,
        config=config,
        device=torch.device("cpu"),
    )
    vals = {
        "thickness": torch.tensor([240.0]),
        "roughness": torch.tensor([6.0]),
        "sld": torch.tensor([45.0]),
        "i0": torch.tensor([1.2]),
        "bkg": torch.tensor([-7.0]),
        "tth_offset": torch.tensor([0.01]),
    }
    with torch.no_grad():
        ref = simulate(vals, {"tth_offset"})[0].double().numpy()
    ours = st.counts(one, q, 1.0)
    assert np.max(np.abs(ours / ref - 1)) < 1e-4


def test_poisson_residuals_and_bic():
    rng = np.random.default_rng(0)
    lam = rng.uniform(0.01, 50.0, 200)
    y = rng.poisson(lam).astype(float)
    r = st.residuals("poisson", lam, y, 1.0)
    assert np.sum(r**2) == pytest.approx(st.deviance(lam, y), rel=1e-12)
    assert st.bic(100.0, 2, 236) == pytest.approx(100.0 + 10 * math.log(236))
    assert st.aic(100.0, 2) == pytest.approx(120.0)


@pytest.mark.parametrize("loss", ["log", "raw", "poisson"])
def test_residual_jacobian_matches_finite_differences(loss):
    rng = np.random.default_rng(1)
    lam0 = rng.uniform(0.5, 30.0, 50)
    y = rng.poisson(lam0).astype(float)
    jlog = rng.normal(size=(50, 3))  # d ln lambda / d theta for lambda = lam0 exp(jlog @ theta)

    def resid(theta):
        return st.residuals(loss, lam0 * np.exp(jlog @ theta), y, 10.0)

    jac = st.residual_jacobian(loss, lam0, y, 10.0, jlog)
    h = 1e-6
    fd = np.column_stack([(resid(h * e) - resid(-h * e)) / (2 * h) for e in np.eye(3)])
    assert np.max(np.abs(jac - fd)) < 1e-5 * max(1.0, np.max(np.abs(fd)))


def test_registered_data():
    q = st.q_grid()
    assert len(q) == 291 and st.held_out_mask(q).sum() == 55
    for n in (1, 2, 3):
        for j in range(16):
            t = st.sample_truth(n, j)
            media = np.append(t.layers[:, 2], st.SUB_SLD)
            assert np.all(np.abs(np.diff(media)) >= 4.0)
            assert np.all(t.layers[:, 1] <= t.layers[:, 0] / 4 + 1e-12)
        assert len(st.starts(n, 0)) == st.N_STARTS


def _env(start: Stack, budget: int = 3000) -> FitEnv:
    q = st.q_grid()
    held = st.held_out_mask(q)
    y = np.random.default_rng(5).poisson(st.counts(TRILAYER, q, 1e6)).astype(float)
    return FitEnv(q[~held], y[~held], 1e6, start, q_ho=q[held], y_ho=y[held], budget=budget)


def test_action_masks():
    env = _env(st.starts(1, 0)[0])
    valid = set(env.valid_actions())
    assert {("fit", 20), ("open", "d", 0), ("add", 0), ("add", 1), ("stop",)} <= valid
    assert not valid & {("open", "d", 1), ("merge", 0), ("revert",), ("loss", "log")}
    env.step(("open", "rho", 0))
    assert not env.valid(("open", "rho", 0)) and env.valid(("close", "rho", 0))


def test_split_then_merge_restores_the_layer():
    start = Stack(np.array([[120.0, 4.0, 30.0]]), st.INSTR_START.copy())
    env = _env(start)
    env.step(("split", 0))
    assert env.state.stack.n_layers == 2
    assert env.state.stack.layers[:, 0].sum() == pytest.approx(120.0)
    env.step(("merge", 0))
    np.testing.assert_allclose(env.state.stack.layers, start.layers)


def test_add_and_budget():
    env = _env(st.starts(1, 0)[0], budget=30)
    env.step(("add", 1))
    assert env.state.stack.n_layers == 2 and env.state.open_layers[1].all()
    env.step(("fit", 100))
    assert env.episode.used <= 30
    assert env.done


def test_protocol_episode_is_reproducible():
    results = [run_protocol(_env(st.starts(3, 0)[0], budget=120), A1).result() for _ in range(2)]
    assert results[0] == results[1]
    assert results[0]["evaluations"] <= 120
