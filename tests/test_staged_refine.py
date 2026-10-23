"""refine on ReflectoFit's staged fit (X1c 3/3), against refine.pipeline.refine.

`staged_refine.refine` keeps refine's signature, configuration, bounds, Occam prior, effect switching
and offset-last-stage refit, and hands every Adam stage to `reflectofit.staged.StagedFit`. Its
forward model is ReflectoFit's unless `forward=` says otherwise.

Pre-registered (written before staged_refine existed):
  1. Port: given refine's own `make_forward`, the refined parameters, fom_before, fom_after, every
     fitted nuisance and r_fit equal refine's within 1e-6 relative (both float32), for the default
     configuration, tth_offset_last_stage (with its offset-off refit), fit_dq_rel, and the likelihood
     stages at budget 1 with counts.
  2. Defaults: staged_refine uses refine's RefineConfig, so DECISIONS #17 (a) holds by construction;
     the default stages are the three of ReflectoLearn#25 and counts_scale is None.
  3. The ReflectoFit forward model at the same values and effects, without resolution and Kα2:
     within 1e-6 relative (float32 output) of refine's model in complex128 without the 1e-9j guard, and
     within max 0.1 / median 1e-3 of refine's shipped model (complex64, guard).
A failure is reported as a failure; the criteria are not changed after the run.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reflectolearn import staged_refine
from reflectolearn.physics import tth2q
from reflectolearn.refine import RefineConfig, likelihood_stages, refine
from reflectolearn.refine.pipeline import make_forward

WL = 1.5406
S = 1e6
RANGES = SimpleNamespace(thickness=(10.0, 1200.0), roughness=(0.0, 15.0), sld=(5.0, 100.0))
SUB = torch.tensor([[20.07, 3.0], [19.0, 4.0]])
Q = torch.as_tensor(tth2q(np.linspace(0.1, 6.0, 400), WL), dtype=torch.float32)
TRUTH = {
    "thickness": torch.tensor([240.0, 120.0]),
    "roughness": torch.tensor([6.0, 3.0]),
    "sld": torch.tensor([45.0, 30.0]),
    "i0": torch.tensor([1.0, 1.1]),
    "bkg": torch.tensor([-7.0, -6.5]),
    "tth_offset": torch.tensor([0.01, -0.005]),
    "dq": torch.tensor([0.002, 0.001]),
    "dq_rel": torch.tensor([0.0, 0.0]),
    "kalpha2_ratio": torch.tensor([0.5, 0.3]),
    "footprint": torch.tensor([500.0, 1500.0]),
    "oxide_thickness": torch.tensor([10.0, 0.0]),
    "oxide_sld": torch.tensor([18.9, 18.9]),
    "oxide_roughness": torch.tensor([3.0, 3.0]),
}
EFFECTS = {"tth_offset", "dq", "kalpha2", "footprint", "oxide"}
P_INIT = torch.tensor([[250.0, 5.0, 43.0], [125.0, 3.5, 31.0]])


def _observed(counts: bool):
    sim = make_forward(Q, SUB, wavelength=WL, config=RefineConfig(), device="cpu")
    with torch.no_grad():
        r = sim(TRUTH, EFFECTS)
    if counts:
        y = torch.poisson(S * r.double(), generator=torch.Generator().manual_seed(2)) / S
        return torch.log10(y.clamp_min(1e-15)).float()
    return torch.log10(r)


CASES = {
    "default": (RefineConfig(), False),
    "offset last stage": (RefineConfig(tth_offset_last_stage=True), False),
    "two-term resolution": (RefineConfig(fit_dq_rel=True), False),
    "likelihood, budget 1": (
        RefineConfig(counts_scale=S, stages=likelihood_stages(budget=1)),
        True,
    ),
}


def _close(a, b, what):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=0, err_msg=what)


@pytest.mark.parametrize("case", CASES)
def test_port_equals_refine_with_its_own_forward_model(case):
    config, counts = CASES[case]
    log_y = _observed(counts)
    mask = torch.ones_like(log_y)
    args = (P_INIT, SUB, Q, log_y, mask, RANGES)
    kw = {"wavelength": WL, "config": config, "device": "cpu"}
    p_old, info_old = refine(*args, **kw)
    p_new, info_new = staged_refine.refine(*args, **kw, forward=make_forward)
    _close(p_new, p_old, "p_refined")
    for key in ("fom_before", "fom_after", "r_fit"):
        _close(info_new[key], info_old[key], key)
    for key in info_old["fitted"]:
        if key in info_old:
            _close(info_new[key], info_old[key], key)
    # added after the physics review of ReflectoFit#7: the optimiser runs in refine's float32 (it
    # takes its dtype from log_y), and its keys map to refine's effect names
    assert all(v.dtype == torch.float32 for v in info_new["state"].values())
    assert info_new["active"] == info_old["active"]
    assert info_new["fitted"] == info_old["fitted"]
    if "offset_off" in info_old:
        for key in ("p", "fom", "fom_offset_fitted"):
            _close(info_new["offset_off"][key], info_old["offset_off"][key], f"offset_off {key}")
        assert torch.equal(info_new["offset_off"]["better"], info_old["offset_off"]["better"])


def test_defaults_are_refines():
    assert staged_refine.RefineConfig is RefineConfig
    cfg = RefineConfig()
    assert cfg.counts_scale is None
    assert [(s.name, s.loss, s.steps) for s in cfg.stages] == [
        ("1.scale", "linear_mse", 120),
        ("2.structure", "correlation", 200),
        ("3.fine", "log_mse", 300),
    ]


def _legacy_c128(vals, effects):
    """refine's model (film + oxide, offset, footprint, i0, bkg) in complex128 without the guard."""
    from reflectolearn.physics import abeles_core, apply_tth_offset

    v = {k: x.double() for k, x in vals.items()}
    q = apply_tth_offset(Q.double(), v["tth_offset"].unsqueeze(1), WL)
    z = torch.zeros(2, dtype=torch.float64)
    sub = SUB.double()
    sld = torch.stack([z, v["oxide_sld"], v["sld"], sub[:, 0]], 1).to(torch.complex128) * 1e-6
    thk = torch.stack([z, v["oxide_thickness"], v["thickness"]], 1)
    rgh = torch.stack([v["oxide_roughness"], v["roughness"], sub[:, 1]], 1)
    r = abeles_core(
        q.to(torch.complex128).unsqueeze(-1), thk.unsqueeze(1), rgh.unsqueeze(1), sld.unsqueeze(1)
    )
    r = r * torch.erf(v["footprint"].unsqueeze(1) * (q * WL / (4 * np.pi)).clamp_min(0.0))
    return r * v["i0"].unsqueeze(1) + 10.0 ** v["bkg"].unsqueeze(1)


def test_reflectofit_forward_model_without_resolution():
    effects = {"tth_offset", "footprint", "oxide"}
    cfg = RefineConfig()
    new = staged_refine.make_reflectofit_forward(Q, SUB, wavelength=WL, config=cfg, device="cpu")
    old = make_forward(Q, SUB, wavelength=WL, config=cfg, device="cpu")
    with torch.no_grad():
        r_new = new(TRUTH, effects).double()
        r_old = old(TRUTH, effects).double()
    ref = _legacy_c128(TRUTH, effects)
    np.testing.assert_allclose(r_new.numpy(), ref.numpy(), rtol=1e-6)
    rel = (r_new / r_old - 1).abs()
    assert rel.max() <= 0.1
    assert rel.median() <= 1e-3
