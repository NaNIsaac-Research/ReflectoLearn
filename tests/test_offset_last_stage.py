"""tth_offset fitted only in the last log-MSE stage (ReflectoLearn#14).

#14's film (250 Å, 6 Å, SLD 45 on Si), 1e7 counts, background 1e-7, refined from the truth.
Units: angles in degrees, q and resolution sigma in 1/Å, SLD in 1e-6/Å².
"""

import numpy as np
import pytest
import torch

from reflectolearn.benchmark import SI, parratt
from reflectolearn.config import Config
from reflectolearn.physics import tth2q
from reflectolearn.predict import preprocess_measured
from reflectolearn.refine.pipeline import RefineConfig, RefineStage, _default_stages, refine

DEV = torch.device("cpu")
CFG = Config()
WL = 1.54
FILM = (250.0, 6.0, 45.0)
TTH = np.arange(0.2, 6.0001, 0.02)
U = np.linspace(-3, 3, 17)
W = np.exp(-0.5 * U**2) / np.exp(-0.5 * U**2).sum()


def observed(resolution: str, seed: int = 1, true_offset: float = 0.0):
    """Counts smeared with a constant sigma of 0.002 or with sigma = 0.02 q; the sample sees
    2θ + true_offset while the file says 2θ."""
    q = tth2q(TTH + true_offset, WL)
    sigma = {"constant": np.full_like(q, 0.002), "dq/q": 0.02 * q}[resolution]
    layers = [(0j, 0, 0), (FILM[2] + 0j, FILM[0], FILM[1]), (SI, 0.0, 3.0)]
    qq = (q[:, None] + sigma[:, None] * U).clip(1e-6)
    r = (parratt(qq.ravel(), layers).reshape(qq.shape) * W).sum(1)
    counts = np.random.default_rng(seed).poisson(1e7 * (r + 1e-7)).astype(float)
    _, r_obs, mask = preprocess_measured(tth2q(TTH, WL), counts, CFG, DEV)
    return torch.log10(r_obs.clamp_min(1e-15)), mask


def run(config: RefineConfig, resolution: str, **data):
    log_r, mask = observed(resolution, **data)
    with torch.enable_grad():
        return refine(
            torch.tensor([list(FILM)]),
            torch.tensor([[20.0, 3.0]]),
            CFG.q_grid,
            log_r,
            mask,
            CFG.param_ranges,
            wavelength=WL,
            config=config,
            device=DEV,
        )


@pytest.mark.parametrize("last_stage", [False, True])
def test_offset_stays_at_zero_before_the_last_log_mse_stage(last_stage):
    """Stages 1-2 alone: the original path moves the offset, the new one leaves it at 0."""
    config = RefineConfig(stages=_default_stages()[:2], tth_offset_last_stage=last_stage)
    _, info = run(config, "dq/q")
    if last_stage:
        assert float(info["tth_offset"][0]) == 0.0
        assert "offset_off" not in info
    else:
        assert float(info["tth_offset"][0]) < -0.03  # measured -0.05 after stage 2


@pytest.fixture(scope="module")
def constant_model_constant_data():
    return run(RefineConfig(tth_offset_last_stage=True), "constant")


def test_correct_model_recovers_the_film_with_the_offset_fitted(constant_model_constant_data):
    """xrr's bar for the path: constant-sigma data, constant model, offset free."""
    p, info = constant_model_constant_data
    assert float(p[0, 2]) == pytest.approx(FILM[2], rel=0.015)  # measured 44.70; old path 42.49
    assert abs(float(info["tth_offset"][0])) < 0.005  # measured -0.0023; old path -0.0158


@pytest.fixture(scope="module")
def original_path():
    return run(RefineConfig(), "constant")


@pytest.mark.xfail(strict=True, reason="ReflectoLearn#14: stages 1-2 drag the offset and the SLD")
def test_original_path_recovers_the_film_with_the_offset_fitted(original_path):
    p, _ = original_path
    assert float(p[0, 2]) == pytest.approx(FILM[2], rel=0.015)  # measured 42.49


def test_flag_off_keeps_the_original_results(original_path):
    """Values of the parent commit, where this path was the only one (equal to its last digit
    there). The new path lands at SLD 44.70 and offset -0.0023, outside these tolerances."""
    p, info = original_path
    assert float(p[0, 2]) == pytest.approx(42.49, abs=1.0)
    assert float(info["tth_offset"][0]) == pytest.approx(-0.0158, abs=0.005)
    assert "offset_off" not in info


def test_offset_off_refit_is_reported(constant_model_constant_data):
    _, info = constant_model_constant_data
    off = info["offset_off"]
    assert off["p"].shape == (1, 3)
    assert float(off["p"][0, 2]) == pytest.approx(FILM[2], rel=0.01)  # measured 44.98
    # nested fits from the same start: the offset adds nothing on data without one, a tie
    assert float(off["fom"][0]) == pytest.approx(float(off["fom_offset_fitted"][0]), rel=0.02)
    assert not bool(off["better"][0])
    assert not bool(info["tth_offset_at_bound"][0])


def test_two_term_model_recovers_the_film_with_the_offset_fitted():
    """#14's own data (dq/q = 2 %) with the model that can describe it."""
    p, info = run(RefineConfig(fit_dq_rel=True, tth_offset_last_stage=True), "dq/q")
    assert float(p[0, 2]) == pytest.approx(FILM[2], rel=0.03)  # measured 44.33; old path 36.97
    assert abs(float(info["tth_offset"][0])) < 0.005  # measured -0.0015


def test_offset_at_its_bound_is_flagged():
    """Bounds that exclude the true offset (0): the fit rests on the lower one."""
    config = RefineConfig(tth_offset_last_stage=True, tth_offset_bounds=(0.01, 0.05))
    _, info = run(config, "constant")
    assert float(info["tth_offset"][0]) == pytest.approx(0.01, abs=4e-4)
    assert bool(info["tth_offset_at_bound"][0])


def test_offset_off_refit_reruns_the_stages_after_the_offset_stage():
    """A stage after the offset's: the refit must run it too, or the two fits are not nested."""
    stages = [
        RefineStage("offset", optimize=("thickness", "sld", "tth_offset"), steps=5, lr=0.01),
        RefineStage("after", optimize=("roughness",), steps=5, lr=0.01),
    ]
    config = RefineConfig(stages=stages, tth_offset_last_stage=True)
    p, info = run(config, "constant")
    # roughness is fitted only after the offset stage: it starts at the truth in both fits
    assert float(p[0, 1]) != FILM[1]
    assert float(info["offset_off"]["p"][0, 1]) != FILM[1]


def test_no_stage_to_fit_the_offset_in_leaves_it_at_zero():
    stages = [RefineStage("film", optimize=("thickness", "sld"), steps=5, lr=0.01)]
    _, info = run(RefineConfig(stages=stages, tth_offset_last_stage=True), "constant")
    assert float(info["state"]["tth_offset"][0]) == 0.0
    assert "offset_off" not in info
