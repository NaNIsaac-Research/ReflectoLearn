"""The opt-in likelihood preset of refine (ReflectoLearn#25): the Poisson deviance loss, the stages,
and that the preset reaches the maximum-likelihood fit where the default strategy stops short.

The curve is a Poisson draw from refine's own forward model, so the model is exact and only the
search is tested. About half a minute on CPU.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reflectolearn.refine import RefineConfig, likelihood_stages, poisson_deviance_loss, refine
from reflectolearn.refine.pipeline import make_forward

WL = 1.5406
S = 1e6  # counts at R = 1
RANGES = SimpleNamespace(thickness=(10.0, 1200.0), roughness=(0.0, 15.0), sld=(5.0, 100.0))
SUB = torch.tensor([[20.07, 3.0]])
TRUTH = {
    "thickness": 240.0,
    "roughness": 6.0,
    "sld": 45.0,
    "i0": 1.0,
    "bkg": -7.0,
    "tth_offset": 0.0,
    "dq": 0.002,
    "kalpha2_ratio": 0.5,
    "footprint": 500.0,
    "oxide_thickness": 0.0,
    "oxide_sld": 18.9,
    "oxide_roughness": 3.0,
}
EFFECTS = {"tth_offset", "dq", "kalpha2", "footprint"}


def deviance(lam: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / lam), 0.0) - (y - lam)
    return 2 * t


def test_poisson_loss_is_the_mean_deviance():
    r = torch.tensor([[1e-1, 1e-3, 1e-5, 2e-7]])
    y = np.array([[99_000.0, 1_050.0, 7.0, 0.0]])  # last point: zero counts
    log_r_obs = torch.log10(torch.as_tensor(np.maximum(y, S * 1e-15) / S)).float()
    mask = torch.ones_like(r)
    got = float(poisson_deviance_loss(S)(r, log_r_obs, mask))
    y_in = S * 10.0 ** log_r_obs.double().numpy()  # the counts the float32 input carries
    want = deviance(S * r.double().numpy(), y_in).mean()
    assert got == pytest.approx(want, rel=1e-5)


def test_default_stages_unchanged():
    cfg = RefineConfig()
    assert cfg.counts_scale is None
    assert [(s.name, s.loss, s.steps, s.lr) for s in cfg.stages] == [
        ("1.scale", "linear_mse", 120, 0.05),
        ("2.structure", "correlation", 200, 0.03),
        ("3.fine", "log_mse", 300, 0.01),
    ]


def test_likelihood_stages():
    assert [(s.name, s.loss, s.steps, s.lr) for s in likelihood_stages()] == [
        ("1.scale", "linear_mse", 1200, 0.05),
        ("3.fine", "poisson", 3000, 0.01),
        ("4.polish0.001", "poisson", 3000, 1e-3),
        ("4.polish0.0001", "poisson", 3000, 1e-4),
    ]


def test_poisson_stage_needs_counts_scale():
    q = torch.linspace(0.02, 0.2, 50)
    with pytest.raises(ValueError, match="counts_scale"):
        refine(
            torch.tensor([[240.0, 6.0, 45.0]]),
            SUB,
            q,
            torch.full((1, 50), -3.0),
            torch.ones(1, 50),
            RANGES,
            wavelength=WL,
            config=RefineConfig(stages=likelihood_stages(budget=1)),
        )


def test_likelihood_preset_reaches_the_ml_fit():
    """From a start 10 % off in thickness, the preset ends at least as well as the truth itself
    (whose deviance is above the maximum likelihood's by about the number of free parameters)."""
    tth = np.linspace(0.2, 6.0, 291)
    q = torch.as_tensor(4 * math.pi / WL * np.sin(np.radians(tth / 2)), dtype=torch.float32)
    cfg = RefineConfig.likelihood(S, fit_oxide=False)
    fwd = make_forward(q, SUB, wavelength=WL, config=cfg, device=torch.device("cpu"))

    def lam(state):
        with torch.no_grad():
            return S * fwd({k: v.reshape(1) for k, v in state.items()}, EFFECTS).double().numpy()[0]

    truth = {k: torch.tensor(v) for k, v in TRUTH.items()}
    y = np.random.default_rng(0).poisson(lam(truth)).astype(float)
    log_r_obs = torch.as_tensor(np.log10(np.maximum(y, S * 1e-15) / S), dtype=torch.float32)

    def fit(config):
        _, info = refine(
            torch.tensor([[264.0, 4.8, 45.0]]),
            SUB,
            q,
            log_r_obs[None],
            torch.ones(1, len(q)),
            RANGES,
            wavelength=WL,
            config=config,
        )
        return {k: v[0] for k, v in info["state"].items()}

    d_truth = deviance(lam(truth), y).sum()
    end = fit(cfg)
    assert deviance(lam(end), y).sum() <= d_truth
    assert abs(float(end["thickness"]) / TRUTH["thickness"] - 1) < 0.002
