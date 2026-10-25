"""refine on ReflectoFit: the same strategy, ReflectoFit's staged optimiser and forward model.

`refine` here takes the arguments of `reflectolearn.refine.refine` and keeps everything that makes
it ReflectoLearn's refine: `RefineConfig` and its stages, the trust window around the network's
estimate, the global nuisance bounds, the Occam prior, switching an effect on once a stage fits it,
the i0/background baseline before fom_before, and the offset-last-stage refit. Each Adam stage runs
in `reflectofit.staged.StagedFit`.

The forward model is `make_reflectofit_forward` (float64 Parratt in ReflectoFit, without the 1e-9j
guard of `AbelesMatrix`) unless `forward=` gives another one with the signature of
`refine.pipeline.make_forward`; given that one, the result equals `refine.refine`.

Resolution goes through ReflectoFit's quadrature at a fixed node count. On the critical edges of
non-absorbing films a fixed count is not yet accurate (ReflectoFit#6), so results with the
resolution term fitted can differ from refine's grid convolution there.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from loguru import logger
from reflectofit import Instrument, forward
from reflectofit.staged import Param, Stage, StagedFit
from torch import Tensor

from reflectolearn.refine.pipeline import (
    _NUISANCE_FLAG,
    _PENALIZE_NEUTRAL,
    OXIDE_KEYS,
    PARAM_KEYS,
    RefineConfig,
    _film_bounds,
    _stage_effects,
    per_curve_log_mse,
)

__all__ = ["RefineConfig", "make_reflectofit_forward", "refine"]


def make_reflectofit_forward(
    q_grid: Tensor,
    substrate: Tensor,
    *,
    wavelength: float,
    config: RefineConfig,
    device: torch.device,
    nodes: int = 61,
) -> Callable[[dict[str, Tensor], set[str]], Tensor]:
    """refine's model on reflectofit.forward: film (+ oxide on top) on a fixed substrate and the
    active nuisance effects, computed in float64 and returned in the dtype of the values.

    The footprint factor erf(s · sinθ) is ReflectoFit's erf(L sinθ / (2√2 σ_beam)) with L = s and
    σ_beam = 1 / (2√2). The resolution σ(q) = sqrt(dq² + (dq_rel · q)²) uses `nodes` per point.
    """
    q64 = q_grid.to(device).double()
    substrate = substrate.to(device).double()

    def simulate(vals: dict[str, Tensor], active: set[str]) -> Tensor:
        v = {k: x.double() for k, x in vals.items()}
        zeros = torch.zeros_like(v["thickness"])
        b = zeros.shape[0]
        sub = substrate.expand(b, -1) if substrate.shape[0] == 1 else substrate
        if config.fit_oxide and "oxide" in active:
            top, mid = v["oxide_roughness"], v["roughness"]
            ox_thk, ox_sld = v["oxide_thickness"], v["oxide_sld"]
        else:
            top, mid = v["roughness"], zeros
            ox_thk, ox_sld = zeros, v["sld"]
        packed = (
            torch.stack([zeros, ox_sld, v["sld"], sub[:, 0]], 1).to(torch.complex128),
            torch.stack([zeros, ox_thk, v["thickness"], zeros], 1),
            torch.stack([zeros, top, mid, sub[:, 1]], 1),
        )
        inst = {
            "wavelength": wavelength,
            "kalpha2_wavelength": config.kalpha2_wavelength,
            "i0": v["i0"],
            "background": 10.0 ** v["bkg"],
            "nodes": nodes,
        }
        if config.fit_tth_offset and "tth_offset" in active:
            inst["tth_offset"] = v["tth_offset"]
        if config.fit_kalpha2 and "kalpha2" in active:
            inst["kalpha2_ratio"] = v["kalpha2_ratio"]
        if config.fit_footprint and "footprint" in active:
            inst["sample_length"] = v["footprint"]
            inst["beam_sigma"] = 1.0 / (2.0 * math.sqrt(2.0))
        if config.fit_dq_rel and "dq" in active:
            inst["sigma_q"] = v["dq"] if config.fit_dq else zeros
            inst["dq_over_q"] = v["dq_rel"]
        elif config.fit_dq and "dq" in active:
            inst["sigma_q"] = v["dq"]
        r = forward(q64, packed, Instrument(**inst), backend="torch")
        r = torch.nan_to_num(r, nan=1e-15, posinf=1.0, neginf=1e-15).clamp_min(1e-15)
        return r.to(vals["thickness"].dtype)

    return simulate


def refine(
    p_init: Tensor,  # (B, 3) physical values [thickness, roughness, sld] (network estimate)
    substrate: Tensor,  # (B, 2) [sub_sld, sub_rough] (fixed)
    q_grid: Tensor,  # (Q,) [Å^-1]
    log_r_obs: Tensor,  # (B, Q) measured reflectivity log10
    mask: Tensor,  # (B, Q) valid-q mask
    param_ranges,
    *,
    wavelength: float,
    config: RefineConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
    forward: Callable | None = None,
) -> tuple[Tensor, dict]:
    """`reflectolearn.refine.refine` on ReflectoFit; same arguments, returns and info keys.
    `forward` builds the model (default `make_reflectofit_forward`)."""
    config = config or RefineConfig()
    device = device or p_init.device

    p_init = p_init.detach().to(device).float()
    g_lo = torch.tensor([getattr(param_ranges, k)[0] for k in PARAM_KEYS], device=device)
    g_hi = torch.tensor([getattr(param_ranges, k)[1] for k in PARAM_KEYS], device=device)
    p_init = p_init.clamp(g_lo, g_hi)
    substrate = substrate.to(device).float()
    q_grid = q_grid.to(device).float()
    log_r_obs = log_r_obs.to(device).float()
    mask = mask.to(device).float()

    B = p_init.shape[0]
    zeros = torch.zeros(B, device=device)
    build = forward or make_reflectofit_forward
    simulate = build(q_grid, substrate, wavelength=wavelength, config=config, device=device)

    def _scalar_bounds(b: tuple[float, float]) -> tuple[Tensor, Tensor]:
        return (torch.tensor(b[0], device=device), torch.tensor(b[1], device=device))

    bounds = _film_bounds(param_ranges, p_init, config.trust_margin)
    for key in ("i0", "bkg", "tth_offset", "dq", "dq_rel", "kalpha2_ratio", "footprint"):
        bounds[key] = _scalar_bounds(getattr(config, f"{key}_bounds"))
    for key in OXIDE_KEYS:
        bounds[key] = _scalar_bounds(getattr(config, f"{key}_bounds"))

    def _full(v: float) -> Tensor:
        return torch.full((B,), v, device=device)

    start: dict[str, Tensor] = {
        "thickness": p_init[:, 0].clone(),
        "roughness": p_init[:, 1].clone(),
        "sld": p_init[:, 2].clone(),
        "i0": torch.ones(B, device=device),
        "bkg": _full(config.bkg_init),
        "tth_offset": zeros.clone(),
        "dq": _full(config.dq_init),
        "dq_rel": _full(config.dq_rel_init),
        "kalpha2_ratio": _full(config.kalpha2_ratio_init),
        "footprint": _full(config.footprint_init),
        "oxide_thickness": _full(config.oxide_thickness_init),
        "oxide_sld": _full(config.oxide_sld_init),
        "oxide_roughness": _full(config.oxide_roughness_init),
    }

    def _enabled(key: str) -> bool:
        if key in PARAM_KEYS:
            return True
        flag = _NUISANCE_FLAG.get(key)
        return bool(getattr(config, flag)) if flag else False

    penalized = {
        k: v for k, v in _PENALIZE_NEUTRAL.items() if not (config.fit_dq_rel and k == "dq")
    }

    # the optimiser's active keys -> the model's effects (refine's _stage_effects)
    def model(vals: dict[str, Tensor], active: set[str]) -> Tensor:
        return simulate(vals, _stage_effects(list(active)))

    fit = StagedFit(
        model,
        {k: Param(v, *bounds[k]) for k, v in start.items()},
        log_r_obs,
        mask,
        q=q_grid,
        counts_scale=config.counts_scale,
        penalty=penalized,
        penalty_weight=config.nuisance_reg,
    )

    def fom() -> float:
        return float(fit.loss("log_mse").item())

    offset_stage = None
    if config.tth_offset_last_stage and _enabled("tth_offset"):
        listed = [
            i
            for i, st in enumerate(config.stages)
            if st.loss == "log_mse" and "tth_offset" in st.optimize
        ]
        offset_stage = listed[-1] if listed else -1
        if offset_stage < 0:
            logger.warning(
                "refine: tth_offset_last_stage is on but no log_mse stage lists tth_offset; "
                "the offset is not fitted and stays at 0"
            )

    def _run_stages(stages, offset_at: int | None, tag: str = "") -> None:
        for i_stage, stage in stages:
            keys = [k for k in stage.optimize if _enabled(k)]
            if offset_at is not None and i_stage != offset_at:
                keys = [k for k in keys if k != "tth_offset"]
            if not keys:
                continue
            if stage.loss == "poisson" and config.counts_scale is None:
                raise ValueError(
                    f"stage {stage.name!r} uses loss='poisson' but counts_scale is None"
                )
            fit.run(
                Stage(
                    stage.name,
                    tuple(keys),
                    stage.loss,
                    stage.steps,
                    stage.lr,
                    stage.q_min,
                    stage.q_max,
                )
            )
            if verbose:
                logger.info(f"  [refine{tag}] {stage.name:<12} ({stage.loss}) FOM={fom():.5f}")

    # the baseline aligns only i0 and bkg, with no effect on, before fom_before (as refine)
    base_keys = [k for k in ("i0", "bkg") if _enabled(k)]
    fit.optimize(base_keys, "log_mse", steps=80, lr=0.05)
    fom_before = fom()

    stages = list(enumerate(config.stages))
    offset_off = None
    if offset_stage is None or offset_stage < 0:
        _run_stages(stages, offset_stage)
    else:
        _run_stages(stages[:offset_stage], offset_stage)
        fit.state["tth_offset"] = zeros.clone()
        offset_off = ({k: v.clone() for k, v in fit.state.items()}, set(fit.active))
        _run_stages(stages[offset_stage:], offset_stage)

    cur = fit.state
    effects = _stage_effects(list(fit.active))
    fom_after = fom()
    p_refined = torch.stack([cur["thickness"], cur["roughness"], cur["sld"]], dim=1)
    info: dict = {"fom_before": fom_before, "fom_after": fom_after, "r_fit": fit.curve().detach()}
    for k in ("i0", "bkg", "tth_offset", "dq", "dq_rel", "kalpha2_ratio", "footprint", *OXIDE_KEYS):
        if _enabled(k):
            info[k] = cur[k].detach()
    info["state"] = {k: v.detach() for k, v in cur.items()}
    info["active"] = set(effects)
    info["bounds"] = bounds
    info["fitted"] = [k for k in cur if _enabled(k)]
    info["forward"] = simulate

    if _enabled("tth_offset") and "tth_offset" in effects:
        lo, hi = bounds["tth_offset"]
        tol = config.tth_offset_bound_tol * (hi - lo)
        at_bound = (cur["tth_offset"] - lo < tol) | (hi - cur["tth_offset"] < tol)
        info["tth_offset_at_bound"] = at_bound
        if at_bound.any():
            logger.warning(
                f"refine: tth_offset at its bound ({float(lo):g}, {float(hi):g}) deg for "
                f"{int(at_bound.sum())}/{B} curves; the film parameters are likely biased"
            )

    if offset_off is not None:
        on_state, on_active = dict(fit.state), set(fit.active)
        with torch.no_grad():
            fom_on = per_curve_log_mse(fit.curve(), log_r_obs, mask)
        state_off, active_off = offset_off
        fit.state.update(state_off)
        fit.active = set(active_off)
        _run_stages(stages[offset_stage:], -1, tag=" offset 0")
        with torch.no_grad():
            fom_off = per_curve_log_mse(fit.curve(), log_r_obs, mask)
        p_off = torch.stack([fit.state[k] for k in PARAM_KEYS], dim=1).detach()
        info["offset_off"] = {
            "p": p_off,
            "fom": fom_off,
            "fom_offset_fitted": fom_on,
            "better": fom_off < fom_on * (1.0 - config.offset_off_margin),
        }
        fit.state.update(on_state)
        fit.active = on_active
        if info["offset_off"]["better"].any():
            n = int(info["offset_off"]["better"].sum())
            logger.warning(
                f"refine: holding tth_offset at 0 fits better for {n}/{B} curves; "
                f"compare info['offset_off']"
            )
    return p_refined, info
