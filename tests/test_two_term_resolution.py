"""Two-term resolution sigma(q) = sqrt(dq² + (dq_rel·q)²) in refine (ReflectoLearn#14).

The generator (`benchmark.smeared`) smears with sigma = dq/q · q; refine's default fits a constant
sigma. `fit_dq_rel` adds the proportional term. Units: q and dq in 1/Å, dq_rel dimensionless.
"""

import numpy as np
import pytest
import torch

from reflectolearn.benchmark import SI, parratt, smeared
from reflectolearn.config import Config
from reflectolearn.physics import tth2q
from reflectolearn.predict import preprocess_measured
from reflectolearn.refine.pipeline import RefineConfig, make_forward, refine

DEV = torch.device("cpu")
CFG = Config()
WL = 1.54
FILM = (250.0, 6.0, 45.0)  # thickness Å, roughness Å, SLD 1e-6/Å²


def state(**overrides):
    one = lambda v: torch.tensor([float(v)])
    vals = {
        "thickness": FILM[0],
        "roughness": FILM[1],
        "sld": FILM[2],
        "i0": 1.0,
        "bkg": -12.0,
        "tth_offset": 0.0,
        "dq": 0.003,
        "dq_rel": 0.0,
        "kalpha2_ratio": 0.0,
        "footprint": 8000.0,
        "oxide_thickness": 0.0,
        "oxide_sld": 18.0,
        "oxide_roughness": 3.0,
    }
    vals.update(overrides)
    return {k: one(v) for k, v in vals.items()}


def q_grid(tth_max=6.0):
    return torch.tensor(tth2q(np.arange(0.2, tth_max + 1e-4, 0.02), WL), dtype=torch.float32)


def forward(q, **config):
    sub = torch.tensor([[SI.real, 3.0]])
    return make_forward(q, sub, wavelength=WL, config=RefineConfig(**config), device=DEV)


def test_zero_relative_term_is_the_constant_model():
    q = q_grid()
    old = forward(q)(state(), {"dq"})
    new = forward(q, fit_dq_rel=True)(state(), {"dq"})
    assert float(((old - new).abs() / old).max()) < 1e-5


@pytest.mark.parametrize("dq_over_q", [0.02, 0.04])
def test_proportional_term_matches_the_generator(dq_over_q):
    """Also checks the kernel is wide enough where dq_rel·q is largest (high q)."""
    q = q_grid(8.0)
    got = forward(q, fit_dq_rel=True)(state(dq=0.0, dq_rel=dq_over_q), {"dq"})[0].numpy()
    layers = [(0j, 0, 0), (FILM[2] + 0j, FILM[0], FILM[1]), (SI.real + 0j, 0, 3.0)]
    want = smeared(q.numpy().astype(float), layers, dq_over_q)
    inner = slice(40, -40)  # the replicate padding at the ends differs from the generator
    assert np.abs(np.log10(got[inner]) - np.log10(want[inner])).max() < 0.01


LAYERS = [(0j, 0, 0), (FILM[2] + 0j, FILM[0], FILM[1]), (SI, 0.0, 3.0)]
Q_SCAN = tth2q(np.arange(0.2, 6.0001, 0.02), WL)


def _fit(r: np.ndarray, config: RefineConfig, seed: int = 1):
    """r on Q_SCAN as 1e7 counts plus a 1e-7 background, refined from the truth."""
    counts = np.random.default_rng(seed).poisson(1e7 * (r + 1e-7))
    _, r_obs, mask = preprocess_measured(Q_SCAN, counts.astype(float), CFG, DEV)
    with torch.enable_grad():
        return refine(
            torch.tensor([list(FILM)]),
            torch.tensor([[20.0, 3.0]]),
            CFG.q_grid,
            torch.log10(r_obs.clamp_min(1e-15)),
            mask,
            CFG.param_ranges,
            wavelength=WL,
            config=config,
            device=DEV,
        )


def _refine(two_term: bool, offset: bool, seed: int = 1) -> tuple[float, float, float]:
    """#14's case: film on Si, dq/q = 2 %, 1e7 counts, bkg 1e-7, refined from the truth."""
    config = RefineConfig(fit_dq_rel=two_term, fit_tth_offset=offset)
    p, info = _fit(smeared(Q_SCAN, LAYERS, 0.02), config, seed)
    return float(p[0, 0]), float(p[0, 2]), float(info.get("dq_rel", [float("nan")])[0])


def test_two_term_model_recovers_the_film_with_the_offset_fixed():
    thickness, sld, dq_rel = _refine(two_term=True, offset=False)
    assert thickness == pytest.approx(FILM[0], rel=0.01)
    assert sld == pytest.approx(FILM[2], rel=0.01)  # the constant-only model gives 44.3 here
    # the data's dq/q; an Occam prior on the resolution held it at 0.009
    assert dq_rel == pytest.approx(0.02, rel=0.1)


@pytest.mark.xfail(strict=True, reason="ReflectoLearn#14: the offset drifts in the refine path")
def test_two_term_model_recovers_the_film_with_the_offset_fitted():
    _, sld, _ = _refine(two_term=True, offset=True)
    assert sld == pytest.approx(FILM[2], rel=0.05)  # measured: 34.4, offset -0.068 deg


@pytest.fixture(scope="module")
def both_terms() -> np.ndarray:
    """R smeared with sigma = sqrt(0.002² + (0.02 q)²): a constant and a proportional term."""
    u = np.linspace(-3, 3, 17)
    w = np.exp(-0.5 * u**2) / np.exp(-0.5 * u**2).sum()
    qq = (Q_SCAN[:, None] + np.hypot(0.002, 0.02 * Q_SCAN)[:, None] * u).clip(1e-6)
    return (parratt(qq.ravel(), LAYERS).reshape(qq.shape) * w).sum(1)


def _resolution(r: np.ndarray, **config) -> tuple[float, float]:
    """Fitted (dq, dq_rel) with the offset and the oxide fixed, so no other key carries a prior."""
    _, info = _fit(r, RefineConfig(fit_tth_offset=False, fit_oxide=False, **config))
    return float(info["dq"][0]), float(info.get("dq_rel", [float("nan")])[0])


def test_two_term_model_recovers_both_terms_with_no_prior_on_either(both_terms):
    """A prior left on dq or dq_rel would pull it toward 0 at a large nuisance_reg."""
    dq, dq_rel = _resolution(both_terms, fit_dq_rel=True)
    assert dq == pytest.approx(0.002, rel=0.25)  # measured 0.00224
    assert dq_rel == pytest.approx(0.02, rel=0.1)  # measured 0.0192
    strong = _resolution(both_terms, fit_dq_rel=True, nuisance_reg=20.0)
    assert strong == pytest.approx((dq, dq_rel), rel=1e-3)


def test_constant_model_keeps_its_prior_on_dq(both_terms):
    """The same weight does act on the default path: dq 0.00256 at 0.2, 0.00055 at 20."""
    dq, _ = _resolution(both_terms, nuisance_reg=20.0)
    assert dq < 0.001


def test_kernel_covers_the_top_of_the_refine_grid():
    """refine's grid runs to 2θ ≈ 15° (q ≈ 1.07); dq_rel·q there is several times dq_bounds[1]."""
    q = CFG.q_grid.float()
    got = forward(q, fit_dq_rel=True)(state(dq=0.0, dq_rel=0.04), {"dq"})[0].numpy()
    layers = [(0j, 0, 0), (FILM[2] + 0j, FILM[0], FILM[1]), (SI.real + 0j, 0, 3.0)]
    want = smeared(q.numpy().astype(float), layers, 0.04)
    top = slice(-700, -350)  # q ≈ 0.7-0.9, clear of the padding at the end
    assert np.abs(np.log10(got[top]) - np.log10(want[top])).max() < 0.01
