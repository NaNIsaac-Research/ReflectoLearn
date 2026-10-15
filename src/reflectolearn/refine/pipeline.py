"""Gradient-based physics refinement of AI parameter predictions.

Refines CNN predictions (initial guess) by directly fitting a physics-simulated
curve to the measured curve. The ``AbelesMatrix`` forward model is fully
differentiable via autograd, so the refinement runs as **Adam gradient descent**.
Core elements:

  * Multi-stage strategy: scale/alignment -> film structure -> fine tuning
  * Per-stage loss choice (linear / correlation / log, see
    :mod:`reflectolearn.refine.losses`)
  * Trust-region bounds that narrow the search around the AI estimate
  * Valid-q masking + FOM (LogMSE) to track improvement

To fit the measured curve accurately, **nuisance parameters** (values that affect
the curve shape but are not of interest) are fitted simultaneously. This prevents
forward-model mismatch from leaking into the parameters of interest (thickness etc.):

  * i0  : intensity scale,  bkg : background (log10)
  * tth_offset : q-axis zero (alignment) offset
  * dq         : resolution (Gaussian q-smearing) — disentangles from roughness
  * footprint  : low-angle beam footprint attenuation (beam > sample) —
                 disentangles from low-angle amplitude
  * oxide      : native oxide / surface layer (nuisance layer between ambient and film)

Bound constraints map a latent variable ``z`` through a sigmoid onto [lo, hi],
keeping parameters within their physical range throughout gradient descent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from reflectolearn.physics import AbelesMatrix, apply_tth_offset
from reflectolearn.refine.losses import LOSS_PRESETS, log_mse_loss

# Predicted/fitted film parameters (same order as config.param_ranges).
PARAM_KEYS = ("thickness", "roughness", "sld")

# Oxide-layer keys (4-medium nuisance layer).
OXIDE_KEYS = ("oxide_thickness", "oxide_sld", "oxide_roughness")

# Occam prior: L2 penalty that pulls degeneracy-prone nuisances toward a neutral
# value (= effect off) so they only turn on "when truly needed". {key: neutral}.
#   oxide_thickness=0 -> no oxide layer, dq=0 -> no smearing.
# tth_offset is excluded: it is well identified by fringe positions, so its
# degeneracy risk is low.
_PENALIZE_NEUTRAL = {"oxide_thickness": 0.0, "dq": 0.0}

# nuisance key -> RefineConfig active-flag name
_NUISANCE_FLAG = {
    "i0": "fit_i0",
    "bkg": "fit_bkg",
    "tth_offset": "fit_tth_offset",
    "dq": "fit_dq",
    "kalpha2_ratio": "fit_kalpha2",
    "footprint": "fit_footprint",
    "oxide_thickness": "fit_oxide",
    "oxide_sld": "fit_oxide",
    "oxide_roughness": "fit_oxide",
}


@dataclass
class RefineStage:
    """One refinement stage: which parameters to optimize, with which loss, for how many steps."""

    name: str
    optimize: tuple[str, ...]          # subset of PARAM_KEYS + nuisance keys
    loss: str = "log_mse"              # key into LOSS_PRESETS
    steps: int = 200
    lr: float = 0.03
    q_min: float | None = None         # [Å^-1] lower q bound seen in this stage
    q_max: float | None = None         # [Å^-1] upper q bound seen in this stage


def _default_stages() -> list[RefineStage]:
    """Three-stage default strategy adapted to the simple film model."""
    return [
        # 1) Intensity/background/alignment: fit i0, bkg, tth_offset + footprint
        #    with a linear loss (low-angle amplitude/critical angle; footprint's
        #    low-angle rolloff is captured here too).
        RefineStage("1.scale", optimize=("i0", "bkg", "tth_offset", "footprint"),
                    loss="linear_mse", steps=120, lr=0.05),
        # 2) Film structure (period): fit thickness/SLD + q-zero with a correlation
        #    loss (invariant to scale and attenuation).
        RefineStage("2.structure", optimize=("thickness", "sld", "tth_offset"),
                    loss="correlation", steps=200, lr=0.03),
        # 3) Fine tuning: refine everything (roughness/resolution/Kα2/footprint/oxide)
        #    with a log loss.
        RefineStage("3.fine",
                    optimize=("thickness", "roughness", "sld", "i0", "bkg",
                              "tth_offset", "dq", "kalpha2_ratio", "footprint", *OXIDE_KEYS),
                    loss="log_mse", steps=300, lr=0.01),
    ]


@dataclass
class RefineConfig:
    trust_margin: float = 0.4                       # trust-region half-width ratio about the AI estimate
    # nuisance active flags
    fit_i0: bool = True                             # intensity scale
    fit_bkg: bool = True                            # background (log10)
    fit_tth_offset: bool = True                     # q-axis zero offset
    fit_dq: bool = True                             # resolution (q-smearing)
    fit_kalpha2: bool = True                        # Cu Kα2 doublet (matches generator)
    fit_footprint: bool = True                      # low-angle footprint attenuation (beam > sample)
    fit_oxide: bool = True                          # native oxide (nuisance layer)
    # Occam prior weight: pulls oxide thickness / dq toward neutral to suppress
    # spurious degeneracy. 0 disables regularization. Applied only to keys in
    # _PENALIZE_NEUTRAL (tuned for the log_mse stage).
    nuisance_reg: float = 0.2
    # Global bounds (nuisances use fixed ranges, no trust-region).
    i0_bounds: tuple[float, float] = (0.5, 2.0)
    bkg_bounds: tuple[float, float] = (-10.0, -4.0)
    tth_offset_bounds: tuple[float, float] = (-0.08, 0.08)     # [deg]
    dq_bounds: tuple[float, float] = (0.0, 0.012)              # [Å^-1] resolution sigma
    kalpha2_ratio_bounds: tuple[float, float] = (0.0, 0.6)     # Kα2/Kα1 intensity ratio
    kalpha2_wavelength: float = 1.54439                        # [A] Cu Kα2 (same as generator)
    # footprint effective scale s: factor = erf(s·sinθ). Large s -> erf≈1 -> attenuation off.
    # Generously covers the augmentation (beam_width, sample_len) range (s≈300~5300).
    footprint_bounds: tuple[float, float] = (150.0, 8000.0)
    oxide_thickness_bounds: tuple[float, float] = (0.0, 60.0)  # [A]
    # Constrain near native-oxide (e.g. SiO2) SLD -> eases degeneracy with film SLD.
    oxide_sld_bounds: tuple[float, float] = (15.0, 22.0)       # [x1e-6 A^-2]
    oxide_roughness_bounds: tuple[float, float] = (0.0, 12.0)  # [A]
    # Initial values
    bkg_init: float = -7.0
    dq_init: float = 0.002
    kalpha2_ratio_init: float = 0.5            # start near the Cu physical value
    footprint_init: float = 2000.0             # start at weak attenuation (data shrinks it if needed)
    oxide_thickness_init: float = 5.0
    oxide_sld_init: float = 18.0
    oxide_roughness_init: float = 3.0
    stages: list[RefineStage] = field(default_factory=_default_stages)


def _film_bounds(param_ranges, p_init: Tensor, margin: float) -> dict[str, tuple[Tensor, Tensor]]:
    """trust-region ∩ global range. Half-width scales with |estimate| but is floored
    at 10% of the range (prevents collapse to 0).

    Assumes p_init is clamped to the global range before the call, so both ends stay
    within [g_lo, g_hi].
    """
    bounds: dict[str, tuple[Tensor, Tensor]] = {}
    for i, key in enumerate(PARAM_KEYS):
        g_lo, g_hi = getattr(param_ranges, key)
        span = g_hi - g_lo
        guess = p_init[:, i]
        half = margin * guess.abs().clamp_min(0.1 * span)
        lo = (guess - half).clamp(g_lo, g_hi)
        hi = (guess + half).clamp(g_lo, g_hi)
        lo = torch.minimum(lo, hi - 1e-3 * span)     # ensure lo < hi (hi stays <= g_hi)
        bounds[key] = (lo, hi)
    return bounds


def _stage_effects(opt_keys: list[str]) -> set[str]:
    """Set of forward-model effects enabled by this stage."""
    eff: set[str] = set()
    if "tth_offset" in opt_keys:
        eff.add("tth_offset")
    if "dq" in opt_keys:
        eff.add("dq")
    if "kalpha2_ratio" in opt_keys:
        eff.add("kalpha2")
    if "footprint" in opt_keys:
        eff.add("footprint")
    if any(k in OXIDE_KEYS for k in opt_keys):
        eff.add("oxide")
    return eff


def refine(
    p_init: Tensor,            # (B, 3) physical values [thickness, roughness, sld] (AI initial guess)
    substrate: Tensor,         # (B, 2) [sub_sld, sub_rough] (fixed)
    q_grid: Tensor,            # (Q,) [Å^-1]
    log_r_obs: Tensor,         # (B, Q) measured reflectivity log10
    mask: Tensor,              # (B, Q) valid-q mask
    param_ranges,              # config.param_ranges (.thickness etc. as [lo, hi])
    *,
    wavelength: float,
    config: RefineConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[Tensor, dict]:
    """Refine AI-estimated film parameters to the measured curve via gradient descent
    (with simultaneous nuisance fitting).

    Returns:
        p_refined: (B, 3) refined [thickness, roughness, sld]
        info: {"fom_before", "fom_after" (scalar LogMSE), fitted nuisances (B,)}
              fom_before is measured with film parameters fixed at AI-init and only
              i0/bkg (global scale/offset) aligned, so the gap to fom_after reflects
              film-parameter improvement rather than scale matching.
    """
    config = config or RefineConfig()
    device = device or p_init.device

    p_init = p_init.detach().to(device).float()
    # AI predictions may fall outside the physical range, so project (clamp) to the global range first.
    g_lo = torch.tensor([getattr(param_ranges, k)[0] for k in PARAM_KEYS], device=device)
    g_hi = torch.tensor([getattr(param_ranges, k)[1] for k in PARAM_KEYS], device=device)
    p_init = p_init.clamp(g_lo, g_hi)
    substrate = substrate.to(device).float()
    q_grid = q_grid.to(device).float()
    log_r_obs = log_r_obs.to(device).float()
    mask = mask.to(device).float()

    B = p_init.shape[0]
    Q = q_grid.shape[0]
    zeros = torch.zeros(B, device=device)
    sub_sld, sub_rough = substrate[:, 0], substrate[:, 1]
    abeles = AbelesMatrix(device=device)

    # Grid spacing / kernel half-width for resolution smearing (fixed to max dq).
    grid_dq = (q_grid[-1] - q_grid[0]).abs() / max(Q - 1, 1)
    max_half = int(min(math.ceil(3.0 * config.dq_bounds[1] / grid_dq.item()), Q // 4))
    max_half = max(max_half, 1)

    def _smear(R: Tensor, dq: Tensor) -> Tensor:
        """Differentiable Gaussian q-smearing. dq(B,) = sigma in q units. Kernel width is fixed."""
        sigma_px = (dq / grid_dq).clamp_min(1e-3).unsqueeze(1)               # (B, 1)
        xs = torch.arange(-max_half, max_half + 1, device=device, dtype=R.dtype)
        kernel = torch.exp(-0.5 * (xs.unsqueeze(0) / sigma_px) ** 2)         # (B, K)
        kernel = kernel / kernel.sum(1, keepdim=True)
        inp = F.pad(R.unsqueeze(0), (max_half, max_half), mode="replicate")  # (1, B, Q+2h)
        out = F.conv1d(inp, kernel.unsqueeze(1), groups=B)                   # (1, B, Q)
        return out.squeeze(0)

    def simulate(vals: dict[str, Tensor], active: set[str]) -> Tensor:
        """Film+substrate(+oxide) parameters -> reflectivity. Applies only nuisance effects in `active`."""
        # q-axis zero offset
        q_use = q_grid
        if config.fit_tth_offset and "tth_offset" in active:
            q_use = apply_tth_offset(q_grid, vals["tth_offset"].unsqueeze(1), wavelength)

        # 4 media [ambient, oxide(top), film, substrate]. With oxide inert, this is
        # numerically identical to a single film (verified against the generator).
        if config.fit_oxide and "oxide" in active:
            top_rough, mid_rough = vals["oxide_roughness"], vals["roughness"]
            ox_thk, ox_sld = vals["oxide_thickness"], vals["oxide_sld"]
        else:
            top_rough, mid_rough = vals["roughness"], zeros
            ox_thk, ox_sld = zeros, vals["sld"]

        sld_t = torch.stack([zeros, ox_sld, vals["sld"], sub_sld], dim=1)        # (B, 4)
        thk_t = torch.stack([zeros, ox_thk, vals["thickness"]], dim=1)           # (B, 3)
        rgh_t = torch.stack([top_rough, mid_rough, sub_rough], dim=1)            # (B, 3)

        def _refl(qq: Tensor) -> Tensor:
            R = abeles.forward(qq, thk_t, rgh_t, sld_t, wavelength=wavelength)
            return torch.nan_to_num(R, nan=1e-15, posinf=1.0, neginf=1e-15).clamp_min(1e-15)

        R = _refl(q_use)

        # Cu Kα2 doublet (same form as generator): at the same angle, a λ2 photon corresponds to q·(λ1/λ2).
        if config.fit_kalpha2 and "kalpha2" in active:
            ratio = vals["kalpha2_ratio"].unsqueeze(-1)
            R2 = _refl(q_use * (wavelength / config.kalpha2_wavelength))
            R = (R + ratio * R2) / (1.0 + ratio)

        # Low-angle footprint attenuation: factor = erf(s·sinθ); larger s -> erf≈1 (no attenuation).
        if config.fit_footprint and "footprint" in active:
            sin_theta = q_use * (wavelength / (4.0 * math.pi))
            if sin_theta.dim() == 1:
                sin_theta = sin_theta.unsqueeze(0)                 # (1, Q) -> broadcast
            arg = vals["footprint"].unsqueeze(-1) * sin_theta.clamp_min(0.0)
            R = R * torch.erf(arg)

        if config.fit_dq and "dq" in active:
            R = _smear(R, vals["dq"])

        return R * vals["i0"].unsqueeze(-1) + torch.pow(10.0, vals["bkg"]).unsqueeze(-1)

    # Bounds: film uses trust-region, the rest (nuisances) use global ranges.
    def _scalar_bounds(b: tuple[float, float]) -> tuple[Tensor, Tensor]:
        return (torch.tensor(b[0], device=device), torch.tensor(b[1], device=device))

    bounds: dict[str, tuple[Tensor, Tensor]] = _film_bounds(param_ranges, p_init, config.trust_margin)
    bounds["i0"] = _scalar_bounds(config.i0_bounds)
    bounds["bkg"] = _scalar_bounds(config.bkg_bounds)
    bounds["tth_offset"] = _scalar_bounds(config.tth_offset_bounds)
    bounds["dq"] = _scalar_bounds(config.dq_bounds)
    bounds["kalpha2_ratio"] = _scalar_bounds(config.kalpha2_ratio_bounds)
    bounds["footprint"] = _scalar_bounds(config.footprint_bounds)
    bounds["oxide_thickness"] = _scalar_bounds(config.oxide_thickness_bounds)
    bounds["oxide_sld"] = _scalar_bounds(config.oxide_sld_bounds)
    bounds["oxide_roughness"] = _scalar_bounds(config.oxide_roughness_bounds)

    # Current parameter state (all detached (B,) tensors).
    def _full(v: float) -> Tensor:
        return torch.full((B,), v, device=device)

    cur: dict[str, Tensor] = {
        "thickness": p_init[:, 0].clone(),
        "roughness": p_init[:, 1].clone(),
        "sld": p_init[:, 2].clone(),
        "i0": torch.ones(B, device=device),
        "bkg": _full(config.bkg_init),
        "tth_offset": zeros.clone(),
        "dq": _full(config.dq_init),
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

    def _reg(vals: dict[str, Tensor], keys: list[str]) -> Tensor:
        """Occam prior: L2 penalty pulling risky nuisances under optimization toward
        their neutral (off) value.

        Normalized by the bound range, so it is scale-free; effects turn on only as
        much as the data demands.
        """
        total = q_grid.new_zeros(())
        if config.nuisance_reg <= 0.0:
            return total
        for k, neutral in _PENALIZE_NEUTRAL.items():
            if k in keys:
                lo, hi = bounds[k]
                total = total + (((vals[k] - neutral) / (hi - lo)) ** 2).mean()
        return config.nuisance_reg * total

    @torch.no_grad()
    def fom(active: set[str]) -> float:
        return float(log_mse_loss(simulate(cur, active), log_r_obs, mask).item())

    def _optimize(opt_keys, loss_fn, steps, lr, active_set, stage_mask):
        """Optimize opt_keys with sigmoid-bound + Adam and write results back to cur.

        Keys not in opt_keys enter simulate as constants (detached) from cur.
        """
        # Initialize latent z by inverting through logit so that value == current value.
        z: dict[str, Tensor] = {}
        for k in opt_keys:
            lo, hi = bounds[k]
            frac = ((cur[k] - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
            z[k] = torch.log(frac / (1 - frac)).detach().clone().requires_grad_(True)

        optimizer = torch.optim.Adam(list(z.values()), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            vals = dict(cur)  # non-optimized keys used as constants (detached)
            for k in opt_keys:
                lo, hi = bounds[k]
                vals[k] = lo + (hi - lo) * torch.sigmoid(z[k])
            loss = loss_fn(simulate(vals, active_set), log_r_obs, stage_mask) + _reg(vals, opt_keys)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for k in opt_keys:
                lo, hi = bounds[k]
                cur[k] = (lo + (hi - lo) * torch.sigmoid(z[k])).detach()

    active: set[str] = set()         # once enabled, an effect stays on through later stages / final FOM

    # Fair baseline: so that both fom_before/after measure "film-parameter fit
    # quality", before measuring we keep film parameters **fixed** at AI-init and
    # only align the trivial global scale/offset (i0, bkg). (Previously i0=1, bkg=-7
    # were fixed, so the measured curve's arbitrary intensity scale dominated the
    # before loss, making the improvement look like it came from scale matching
    # rather than film quality.) Since simulate always applies i0/bkg regardless of
    # active, we leave active=∅ here.
    base_keys = [k for k in ("i0", "bkg") if _enabled(k)]
    if base_keys:
        _optimize(base_keys, log_mse_loss, steps=80, lr=0.05,
                  active_set=active, stage_mask=mask)
    fom_before = fom(active)

    for stage in config.stages:
        opt_keys = [k for k in stage.optimize if _enabled(k)]
        if not opt_keys:
            continue
        active |= _stage_effects(opt_keys)

        loss_fn = LOSS_PRESETS[stage.loss]
        stage_mask = mask
        if stage.q_min is not None:
            stage_mask = stage_mask * (q_grid > stage.q_min)
        if stage.q_max is not None:
            stage_mask = stage_mask * (q_grid < stage.q_max)

        _optimize(opt_keys, loss_fn, stage.steps, stage.lr, active, stage_mask)

        if verbose:
            logger.info(f"  [refine] {stage.name:<12} ({stage.loss}) FOM={fom(active):.5f}")

    fom_after = fom(active)
    p_refined = torch.stack([cur["thickness"], cur["roughness"], cur["sld"]], dim=1)
    info: dict = {"fom_before": fom_before, "fom_after": fom_after}
    # Final fitted curve (all active nuisances included) — returned so overlay plots
    # compare against measured under the same conditions (footprint/i0/bkg etc. applied).
    with torch.no_grad():
        info["r_fit"] = simulate(cur, active).detach()
    # Also return the fitted nuisance values (only the active ones).
    for k in ("i0", "bkg", "tth_offset", "dq", "kalpha2_ratio", "footprint", *OXIDE_KEYS):
        if _enabled(k):
            info[k] = cur[k].detach()
    return p_refined, info


__all__ = ["RefineStage", "RefineConfig", "refine", "PARAM_KEYS", "OXIDE_KEYS"]
