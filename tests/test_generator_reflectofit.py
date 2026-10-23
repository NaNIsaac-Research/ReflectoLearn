"""The training generator on ReflectoFit (X1c), against the generator it replaces.

`simulation.engine = "legacy"` (the default) must stay the generator of cc3c5c5, frozen in
tests/reference; `"reflectofit"` computes the same draws with `reflectofit.forward` in float64, with
footprint and resolution inside the forward model and a fixed resolution node count.

Pre-registered (written before the reflectofit engine existed; seeds differ from the pilot's 0 and 1):
  1. legacy engine: x, y of generate_synthetic_batch equal to the frozen copy, bit for bit, with and
     without the augmentor (the augmentor is refactored into draw / apply / noise steps).
  2. reflectofit engine: same seed, same draws: y and the mask bit for bit, and the torch RNG state
     after the batch identical.
  3. reflectofit engine, no augmentor: the float64 curve within 1e-10 relative of the legacy model
     (media layout, 2θ offset, Kα2) evaluated with `abeles_core` in complex128 without the 1e-9j
     guard; the float32 output within 2 float32 ulp (2.4e-7) of that.
  4. reflectofit engine, shipped legacy (complex64, guard): differs at most 0.1 relative, median
     at most 1e-3 (X1a section 5 measured 7.3e-3 on its stacks; the X1c pilot 6.8e-2 on thick
     real-SLD films). This bounds the precision change, it is not an equivalence.
  5. reflectofit engine with the augmentor: the curve before noise equals `reflectofit.forward` with
     the instrument built from the augmentor's draws (footprint σ = FWHM / 2.3548, i0, 10**bkg,
     sigma_q, fixed nodes), within 1e-12; with resolution and offset off the footprint equals the
     legacy erf factor within 1e-10.
  6. Same seed twice gives the same batch, bit for bit, for both engines.
A failure is reported as a failure; the tolerances are not widened after the run.
First run: all held except 2 with the augmentor (see test_with_the_augmentor_only_the_draws_before_the_counts_match).
"""

import math

import numpy as np
import pytest
import torch
from reference import reflectolearn_augmentations as ref_aug
from reference import reflectolearn_generator as ref_gen
from reflectofit import Instrument, forward

from reflectolearn.config import Config
from reflectolearn.data import XRRAugmentor, generate_synthetic_batch, simulate_reflectivity
from reflectolearn.data.generator import draw_media, engine_metadata, reflectofit_curve
from reflectolearn.physics import abeles_core, apply_tth_offset

SEED = 20261024
B = 8


def _config(engine: str, **aug) -> Config:
    cfg = Config(device="cpu")
    cfg.simulation.engine = engine
    for k, v in aug.items():
        setattr(cfg.augmentation, k, v)
    return cfg


def _batch(fn, seed=SEED):
    torch.manual_seed(seed)
    out = fn()
    return out, torch.get_rng_state()


def _film(cfg, n=B):
    pr, sub = cfg.param_ranges, cfg.substrate

    def u(r):
        return torch.rand(n) * (r[1] - r[0]) + r[0]

    return u(pr.thickness), u(pr.roughness), u(pr.sld), u(sub.sld), u(sub.roughness)


@pytest.mark.parametrize("augment", [False, True])
def test_legacy_engine_is_the_frozen_generator(augment):
    cfg = _config("legacy")
    q = cfg.q_grid
    new, state_new = _batch(
        lambda: generate_synthetic_batch(cfg, B, XRRAugmentor(cfg, q) if augment else None)
    )
    old, state_old = _batch(
        lambda: ref_gen.generate_synthetic_batch(
            cfg, B, ref_aug.XRRAugmentor(cfg, q) if augment else None
        )
    )
    for a, b in zip(new, old, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(state_new, state_old)


def test_reflectofit_engine_draws_what_the_legacy_one_draws():
    legacy, fit = _config("legacy"), _config("reflectofit")
    (x_old, y_old), state_old = _batch(lambda: generate_synthetic_batch(legacy, B))
    (x_new, y_new), state_new = _batch(lambda: generate_synthetic_batch(fit, B))
    assert torch.equal(y_new, y_old)
    assert torch.equal(x_new[:, 1], x_old[:, 1])  # mask channel
    assert torch.equal(state_new, state_old)


def test_with_the_augmentor_only_the_draws_before_the_counts_match():
    # Criterion 2 with the augmentor was falsified on the first run: torch.poisson draws a number
    # of uniforms that depends on lambda, so a different curve leaves the RNG elsewhere and the
    # mask, drawn after the counts, differs. The targets are drawn before the counts and match.
    legacy, fit = _config("legacy"), _config("reflectofit")
    q = legacy.q_grid
    (_, y_old), _ = _batch(lambda: generate_synthetic_batch(legacy, B, XRRAugmentor(legacy, q)))
    (_, y_new), _ = _batch(lambda: generate_synthetic_batch(fit, B, XRRAugmentor(fit, q)))
    assert torch.equal(y_new, y_old)


def _legacy_model_c128(cfg, q, media):
    """The legacy generator's model (media layout, offset, Kα2) in complex128, no guard."""
    sld, thickness, roughness, offset, ratio = media
    wl = cfg.simulation.wavelength
    q64 = q.double()
    q_eff = apply_tth_offset(q64, offset.double().unsqueeze(1), wl) if offset is not None else q64
    q_eff = q_eff.expand(sld.shape[0], -1)
    s = (sld.to(torch.complex128) - sld[:, :1].to(torch.complex128)) * 1e-6

    def r(qq):
        return abeles_core(
            qq.to(torch.complex128).unsqueeze(-1),
            thickness[:, :3].double().unsqueeze(1),
            roughness[:, 1:].double().unsqueeze(1),
            s.unsqueeze(1),
        )

    out = r(q_eff)
    if ratio is not None:
        a = ratio.double().unsqueeze(1)
        out = (out + a * r(q_eff * (wl / cfg.augmentation.kalpha2_wavelength))) / (1 + a)
    return out


def test_reflectofit_curve_is_the_legacy_model_in_double_precision():
    cfg = _config("reflectofit")
    q = cfg.q_grid
    torch.manual_seed(SEED)
    media = draw_media(cfg, *_film(cfg), "cpu")
    got = reflectofit_curve(cfg, q, media)
    np.testing.assert_allclose(got.numpy(), _legacy_model_c128(cfg, q, media).numpy(), rtol=1e-10)


def test_reflectofit_output_is_the_double_curve_in_float32():
    cfg = _config("reflectofit")
    q = cfg.q_grid
    torch.manual_seed(SEED)
    film = _film(cfg)
    out = simulate_reflectivity(cfg, q, *film, "cpu")
    torch.manual_seed(SEED)
    film = _film(cfg)
    media = draw_media(cfg, *film, "cpu")
    ref = _legacy_model_c128(cfg, q, media).clamp_min(1e-15)
    assert out.dtype == torch.float32
    np.testing.assert_allclose(out.double().numpy(), ref.numpy(), rtol=2.4e-7)


def test_precision_change_against_the_shipped_legacy_generator():
    fit, legacy = _config("reflectofit"), _config("legacy")
    q = fit.q_grid
    new, _ = _batch(lambda: simulate_reflectivity(fit, q, *_film(fit), "cpu"))
    old, _ = _batch(lambda: ref_gen.simulate_reflectivity(legacy, q, *_film(legacy), "cpu"))
    rel = (new.double() / old.double() - 1).abs()
    assert rel.max() <= 0.1
    assert rel.median() <= 1e-3


def test_augmentor_draws_become_the_reflectofit_instrument():
    cfg = _config("reflectofit", prob=1.0)
    q = cfg.q_grid
    aug = XRRAugmentor(cfg, q)
    torch.manual_seed(SEED)
    film = _film(cfg)
    media = draw_media(cfg, *film, "cpu")
    draw = aug.draw(B, "cpu")
    got = reflectofit_curve(cfg, q, media, draw)

    sld, thickness, roughness, offset, ratio = media
    inst = Instrument(
        wavelength=cfg.simulation.wavelength,
        kalpha2_wavelength=cfg.augmentation.kalpha2_wavelength,
        tth_offset=offset.double().numpy(),
        kalpha2_ratio=ratio.double().numpy(),
        sample_length=draw.sample_len,
        beam_sigma=draw.beam_width / 2.3548,
        sigma_q=draw.sigma,
        i0=draw.i0[:, 0].double().numpy(),
        background=10.0 ** draw.bkg[:, 0].double().numpy(),
        nodes=cfg.simulation.resolution_nodes,
    )
    packed = (
        sld.to(torch.complex128).numpy(),
        thickness.double().numpy(),
        roughness.double().numpy(),
    )
    expected = forward(q.double().numpy(), packed, inst)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12)


def test_footprint_is_the_legacy_erf_factor():
    cfg = _config("reflectofit", prob=1.0, tth_offset=[0.0, 0.0], res_sigma_range=[0.0, 0.0])
    q = cfg.q_grid
    aug = XRRAugmentor(cfg, q)
    torch.manual_seed(SEED)
    film = _film(cfg)
    media = draw_media(cfg, *film, "cpu")
    draw = aug.draw(B, "cpu")
    with_fp = reflectofit_curve(cfg, q, media, draw)
    bare = reflectofit_curve(cfg, q, media)
    sin_t = q.double() * cfg.simulation.wavelength / (4 * math.pi)
    sigma_beam = draw.beam_width / 2.3548
    factor = torch.erf(draw.sample_len * sin_t / (2 * math.sqrt(2) * sigma_beam))
    i0, bkg = draw.i0.double(), 10.0 ** draw.bkg.double()
    np.testing.assert_allclose(with_fp.numpy(), (i0 * bare * factor + bkg).numpy(), rtol=1e-10)


@pytest.mark.parametrize("engine", ["legacy", "reflectofit"])
def test_same_seed_same_batch(engine):
    cfg = _config(engine)
    q = cfg.q_grid
    a, _ = _batch(lambda: generate_synthetic_batch(cfg, B, XRRAugmentor(cfg, q)))
    b, _ = _batch(lambda: generate_synthetic_batch(cfg, B, XRRAugmentor(cfg, q)))
    for u, v in zip(a, b, strict=True):
        assert torch.equal(u, v)


def test_engine_metadata_records_nodes_and_error():
    cfg = _config("reflectofit")
    meta = engine_metadata(cfg, n=2, seed=SEED)
    assert meta["engine"] == "reflectofit"
    assert meta["resolution_nodes"] == cfg.simulation.resolution_nodes
    assert meta["precision"] == "float64"
    assert set(meta["resolution_error"]) == {"sigma_q", "max_rel", "median_rel"}
    assert all(e >= 0 for e in meta["resolution_error"]["max_rel"])
    assert engine_metadata(_config("legacy"))["engine"] == "legacy"
