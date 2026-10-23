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
  * dq_rel     : q-proportional resolution term, sigma(q) = sqrt(dq² + (dq_rel·q)²); off by default
  * footprint  : low-angle beam footprint attenuation (beam > sample) —
                 disentangles from low-angle amplitude
  * oxide      : native oxide / surface layer (nuisance layer between ambient and film)

Bound constraints map a latent variable ``z`` through a sigmoid onto [lo, hi],
keeping parameters within their physical range throughout gradient descent.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from reflectolearn.physics import AbelesMatrix, apply_tth_offset
from reflectolearn.refine.losses import LOSS_PRESETS, log_mse_loss, poisson_deviance_loss

# Predicted/fitted film parameters (same order as config.param_ranges).
PARAM_KEYS = ("thickness", "roughness", "sld")

# Oxide-layer keys (4-medium nuisance layer).
OXIDE_KEYS = ("oxide_thickness", "oxide_sld", "oxide_roughness")

# Occam prior: L2 penalty that pulls degeneracy-prone nuisances toward a neutral
# value (= effect off) so they only turn on "when truly needed". {key: neutral}.
#   oxide_thickness=0 -> no oxide layer, dq=0 -> no smearing.
# tth_offset is excluded: it is well identified by fringe positions, so its
# degeneracy risk is low. With the two-term resolution (fit_dq_rel) dq is excluded too, see there.
_PENALIZE_NEUTRAL = {"oxide_thickness": 0.0, "dq": 0.0}

# Curves per checkpointed chunk in the q-dependent smearing (bounds its (chunk, Q, K) weights).
_SMEAR_CHUNK = 4

# nuisance key -> RefineConfig active-flag name
_NUISANCE_FLAG = {
    "i0": "fit_i0",
    "bkg": "fit_bkg",
    "tth_offset": "fit_tth_offset",
    "dq": "fit_dq",
    "dq_rel": "fit_dq_rel",
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
    optimize: tuple[str, ...]  # subset of PARAM_KEYS + nuisance keys
    loss: str = "log_mse"  # key into LOSS_PRESETS, or "poisson" (needs RefineConfig.counts_scale)
    steps: int = 200
    lr: float = 0.03
    q_min: float | None = None  # [Å^-1] lower q bound seen in this stage
    q_max: float | None = None  # [Å^-1] upper q bound seen in this stage


def _default_stages() -> list[RefineStage]:
    """Three-stage default strategy adapted to the simple film model."""
    return [
        # 1) Intensity/background/alignment: fit i0, bkg, tth_offset + footprint
        #    with a linear loss (low-angle amplitude/critical angle; footprint's
        #    low-angle rolloff is captured here too).
        RefineStage(
            "1.scale",
            optimize=("i0", "bkg", "tth_offset", "footprint"),
            loss="linear_mse",
            steps=120,
            lr=0.05,
        ),
        # 2) Film structure (period): fit thickness/SLD + q-zero with a correlation
        #    loss (invariant to scale and attenuation).
        RefineStage(
            "2.structure",
            optimize=("thickness", "sld", "tth_offset"),
            loss="correlation",
            steps=200,
            lr=0.03,
        ),
        # 3) Fine tuning: refine everything (roughness/resolution/Kα2/footprint/oxide)
        #    with a log loss.
        RefineStage(
            "3.fine",
            optimize=(
                "thickness",
                "roughness",
                "sld",
                "i0",
                "bkg",
                "tth_offset",
                "dq",
                "dq_rel",
                "kalpha2_ratio",
                "footprint",
                *OXIDE_KEYS,
            ),
            loss="log_mse",
            steps=300,
            lr=0.01,
        ),
    ]


def likelihood_stages(budget: int = 10) -> list[RefineStage]:
    """Stages that fit counting data to its Poisson maximum likelihood (ReflectoLearn#25).

    Changes to the default strategy, each found necessary in the R2b cause separation (removing any
    one of them stops refine short of the optimum on curves from its own forward model):
      * every stage runs `budget` times as many steps;
      * no correlation stage: its loss ignores the level of the curve and accepts other fringe orders;
      * the fine stage minimizes the Poisson deviance instead of the log10 MSE, whose optimum is not
        the maximum-likelihood point;
      * the fine stage is followed by two stages over the same keys at lr 1e-3 and 1e-4, so that
        Adam settles instead of jittering around the optimum.
    About 30 times the default number of steps.
    """
    scale, _, fine = _default_stages()
    fine = RefineStage(fine.name, fine.optimize, "poisson", fine.steps * budget, fine.lr)
    return [
        RefineStage(scale.name, scale.optimize, scale.loss, scale.steps * budget, scale.lr),
        fine,
        *(
            RefineStage(f"4.polish{lr:g}", fine.optimize, "poisson", 300 * budget, lr)
            for lr in (1e-3, 1e-4)
        ),
    ]


@dataclass
class RefineConfig:
    trust_margin: float = 0.4  # trust-region half-width ratio about the AI estimate
    # nuisance active flags
    fit_i0: bool = True  # intensity scale
    fit_bkg: bool = True  # background (log10)
    fit_tth_offset: bool = True  # q-axis zero offset
    # Fit tth_offset only in the last log_mse stage that lists it, starting from 0, and refit from
    # that stage on with the offset fixed at 0 for comparison (info["offset_off"]). Off keeps the
    # original path, where stages 1-2 move the offset too: their scale-insensitive losses let it
    # slide along the offset-SLD valley of a dense film (ReflectoLearn#14).
    tth_offset_last_stage: bool = False
    fit_dq: bool = True  # resolution (q-smearing)
    # q-proportional resolution, sigma(q) = sqrt(dq² + (dq_rel·q)²): angular divergence gives the
    # constant part, the wavelength spread the proportional one. Off keeps the constant-only model.
    # On, neither term gets the Occam prior: the q dependence of the smearing tells them apart, and
    # at dq/q = 2 % the pull, 0.2·(0.02/0.05)², is three times the log-MSE and held dq_rel near 0.009.
    fit_dq_rel: bool = False
    fit_kalpha2: bool = True  # Cu Kα2 doublet (matches generator)
    fit_footprint: bool = True  # low-angle footprint attenuation (beam > sample)
    fit_oxide: bool = True  # native oxide (nuisance layer)
    # Occam prior weight: pulls oxide thickness / dq toward neutral to suppress
    # spurious degeneracy. 0 disables regularization. Applied only to keys in
    # _PENALIZE_NEUTRAL (tuned for the log_mse stage).
    nuisance_reg: float = 0.2
    # Global bounds (nuisances use fixed ranges, no trust-region).
    i0_bounds: tuple[float, float] = (0.5, 2.0)
    bkg_bounds: tuple[float, float] = (-10.0, -4.0)
    tth_offset_bounds: tuple[float, float] = (-0.08, 0.08)  # [deg]
    # tth_offset within this fraction of the bound range from either end counts as at the bound
    tth_offset_bound_tol: float = 0.01
    # offset-off refit counts as better when its log-MSE is lower by more than this fraction. A tie
    # only says the data do not need an offset; the two fits differed by up to 1.2 % that way.
    offset_off_margin: float = 0.05
    dq_bounds: tuple[float, float] = (0.0, 0.012)  # [Å^-1] resolution sigma
    dq_rel_bounds: tuple[float, float] = (0.0, 0.05)  # relative resolution dq/q
    kalpha2_ratio_bounds: tuple[float, float] = (0.0, 0.6)  # Kα2/Kα1 intensity ratio
    kalpha2_wavelength: float = 1.54439  # [A] Cu Kα2 (same as generator)
    # footprint effective scale s: factor = erf(s·sinθ). Large s -> erf≈1 -> attenuation off.
    # Generously covers the augmentation (beam_width, sample_len) range (s≈300~5300).
    footprint_bounds: tuple[float, float] = (150.0, 8000.0)
    oxide_thickness_bounds: tuple[float, float] = (0.0, 60.0)  # [A]
    # Constrain near native-oxide (e.g. SiO2) SLD -> eases degeneracy with film SLD.
    oxide_sld_bounds: tuple[float, float] = (15.0, 22.0)  # [x1e-6 A^-2]
    oxide_roughness_bounds: tuple[float, float] = (0.0, 12.0)  # [A]
    # Initial values
    bkg_init: float = -7.0
    dq_init: float = 0.002
    dq_rel_init: float = 0.01  # mid lab range (Cu + Göbel mirror ~0.5-2 % dq/q)
    kalpha2_ratio_init: float = 0.5  # start near the Cu physical value
    footprint_init: float = 2000.0  # start at weak attenuation (data shrinks it if needed)
    oxide_thickness_init: float = 5.0
    oxide_sld_init: float = 18.0
    oxide_roughness_init: float = 3.0
    # Counts at R = 1, for stages with loss="poisson" (the likelihood preset). None: not counting data.
    counts_scale: float | None = None
    stages: list[RefineStage] = field(default_factory=_default_stages)

    @classmethod
    def likelihood(cls, counts_scale: float, **overrides) -> RefineConfig:
        """The configuration that reaches the Poisson maximum-likelihood fit (ReflectoLearn#25):
        likelihood_stages() with the count scale of the data. Other fields keep their defaults
        unless overridden.

        Use it on raw counts on the measured q points (refine's q_grid = the measured q,
        log_r_obs = log10(counts / counts_scale)), not on a curve interpolated onto the model grid:
        see `poisson_deviance_loss`."""
        return cls(counts_scale=counts_scale, stages=likelihood_stages(), **overrides)


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
        lo = torch.minimum(lo, hi - 1e-3 * span)  # ensure lo < hi (hi stays <= g_hi)
        bounds[key] = (lo, hi)
    return bounds


def _stage_effects(opt_keys: list[str]) -> set[str]:
    """Set of forward-model effects enabled by this stage."""
    eff: set[str] = set()
    if "tth_offset" in opt_keys:
        eff.add("tth_offset")
    if "dq" in opt_keys or "dq_rel" in opt_keys:
        eff.add("dq")
    if "kalpha2_ratio" in opt_keys:
        eff.add("kalpha2")
    if "footprint" in opt_keys:
        eff.add("footprint")
    if any(k in OXIDE_KEYS for k in opt_keys):
        eff.add("oxide")
    return eff


def make_forward(
    q_grid: Tensor,
    substrate: Tensor,
    *,
    wavelength: float,
    config: RefineConfig,
    device: torch.device,
) -> Callable[[dict[str, Tensor], set[str]], Tensor]:
    """Build the forward model refine fits with: film (+oxide) on a fixed substrate, plus the active
    nuisance effects. The returned function takes a dict of (W,) parameter tensors for any batch size W
    (refine's batch, or every MCMC walker at once) and returns (W, Q) reflectivity.
    """
    q_grid = q_grid.to(device).float()
    substrate = substrate.to(device).float()
    Q = q_grid.shape[0]
    abeles = AbelesMatrix(device=device)
    # Grid spacing / kernel half-width for resolution smearing (fixed to the largest sigma).
    grid_dq = (q_grid[-1] - q_grid[0]).abs() / max(Q - 1, 1)
    sigma_max = config.dq_bounds[1]
    if config.fit_dq_rel:
        sigma_max = math.hypot(sigma_max, config.dq_rel_bounds[1] * q_grid.abs().max().item())
    max_half = max(int(min(math.ceil(3.0 * sigma_max / grid_dq.item()), Q // 4)), 1)

    def _smear(R: Tensor, dq: Tensor) -> Tensor:
        """Differentiable Gaussian q-smearing. dq(B,) = sigma in q units. Kernel width is fixed."""
        B = R.shape[0]
        sigma_px = (dq / grid_dq).clamp_min(1e-3).unsqueeze(1)  # (B, 1)
        xs = torch.arange(-max_half, max_half + 1, device=device, dtype=R.dtype)
        kernel = torch.exp(-0.5 * (xs.unsqueeze(0) / sigma_px) ** 2)  # (B, K)
        kernel = kernel / kernel.sum(1, keepdim=True)
        inp = F.pad(R.unsqueeze(0), (max_half, max_half), mode="replicate")  # (1, B, Q+2h)
        out = F.conv1d(inp, kernel.unsqueeze(1), groups=B)  # (1, B, Q)
        return out.squeeze(0)

    def _smear_varying(R: Tensor, sigma: Tensor) -> Tensor:
        """Gaussian q-smearing with a sigma(B, Q) [Å^-1] that varies along q, evaluated at each
        output q like ``benchmark.smeared``. With a constant sigma it equals ``_smear``.

        The weights are (B, Q, K), so the kernel reaches only 5 sigma of the current largest sigma
        (the tail beyond is below 1e-6) and chunks of curves are checkpointed: their weights are
        recomputed in backward instead of being kept for the whole batch.
        """
        half = max(1, min(max_half, math.ceil(5.0 * float(sigma.detach().max()) / grid_dq.item())))
        xs = torch.arange(-half, half + 1, device=device, dtype=R.dtype)

        def smear_chunk(r: Tensor, s: Tensor) -> Tensor:
            sigma_px = (s / grid_dq).clamp_min(1e-3).unsqueeze(-1)  # (b, Q, 1)
            weights = torch.exp(-0.5 * (xs / sigma_px) ** 2)  # (b, Q, K)
            weights = weights / weights.sum(-1, keepdim=True)
            padded = F.pad(r.unsqueeze(0), (half, half), mode="replicate").squeeze(0)
            return torch.einsum("bqk,bqk->bq", padded.unfold(-1, 2 * half + 1, 1), weights)

        chunks = [
            checkpoint(
                smear_chunk,
                R[i : i + _SMEAR_CHUNK],
                sigma[i : i + _SMEAR_CHUNK],
                use_reentrant=False,
            )
            for i in range(0, R.shape[0], _SMEAR_CHUNK)
        ]
        return torch.cat(chunks)

    def simulate(vals: dict[str, Tensor], active: set[str]) -> Tensor:
        """Film+substrate(+oxide) parameters -> reflectivity. Applies only nuisance effects in `active`."""
        zeros = torch.zeros_like(vals["thickness"])
        sub = substrate.expand(zeros.shape[0], -1) if substrate.shape[0] == 1 else substrate
        sub_sld, sub_rough = sub[:, 0], sub[:, 1]
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

        sld_t = torch.stack([zeros, ox_sld, vals["sld"], sub_sld], dim=1)  # (B, 4)
        thk_t = torch.stack([zeros, ox_thk, vals["thickness"]], dim=1)  # (B, 3)
        rgh_t = torch.stack([top_rough, mid_rough, sub_rough], dim=1)  # (B, 3)

        def _refl(qq: Tensor) -> Tensor:
            if qq.dim() == 1:
                qq = qq.unsqueeze(0).expand(zeros.shape[0], -1)
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
                sin_theta = sin_theta.unsqueeze(0)  # (1, Q) -> broadcast
            arg = vals["footprint"].unsqueeze(-1) * sin_theta.clamp_min(0.0)
            R = R * torch.erf(arg)

        if config.fit_dq_rel and "dq" in active:
            dq = vals["dq"] if config.fit_dq else torch.zeros_like(vals["dq_rel"])
            sigma = torch.sqrt(dq.unsqueeze(1) ** 2 + (vals["dq_rel"].unsqueeze(1) * q_grid) ** 2)
            R = _smear_varying(R, sigma)
        elif config.fit_dq and "dq" in active:
            R = _smear(R, vals["dq"])

        return R * vals["i0"].unsqueeze(-1) + torch.pow(10.0, vals["bkg"]).unsqueeze(-1)

    return simulate


def refine(
    p_init: Tensor,  # (B, 3) physical values [thickness, roughness, sld] (AI initial guess)
    substrate: Tensor,  # (B, 2) [sub_sld, sub_rough] (fixed)
    q_grid: Tensor,  # (Q,) [Å^-1]
    log_r_obs: Tensor,  # (B, Q) measured reflectivity log10
    mask: Tensor,  # (B, Q) valid-q mask
    param_ranges,  # config.param_ranges (.thickness etc. as [lo, hi])
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
    zeros = torch.zeros(B, device=device)
    simulate = make_forward(q_grid, substrate, wavelength=wavelength, config=config, device=device)

    # Bounds: film uses trust-region, the rest (nuisances) use global ranges.
    def _scalar_bounds(b: tuple[float, float]) -> tuple[Tensor, Tensor]:
        return (torch.tensor(b[0], device=device), torch.tensor(b[1], device=device))

    bounds: dict[str, tuple[Tensor, Tensor]] = _film_bounds(
        param_ranges, p_init, config.trust_margin
    )
    bounds["i0"] = _scalar_bounds(config.i0_bounds)
    bounds["bkg"] = _scalar_bounds(config.bkg_bounds)
    bounds["tth_offset"] = _scalar_bounds(config.tth_offset_bounds)
    bounds["dq"] = _scalar_bounds(config.dq_bounds)
    bounds["dq_rel"] = _scalar_bounds(config.dq_rel_bounds)
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

    def _reg(vals: dict[str, Tensor], keys: list[str]) -> Tensor:
        """Occam prior: L2 penalty pulling risky nuisances under optimization toward
        their neutral (off) value.

        Normalized by the bound range, so it is scale-free; effects turn on only as
        much as the data demands.
        """
        total = q_grid.new_zeros(())
        if config.nuisance_reg <= 0.0:
            return total
        for k, neutral in penalized.items():
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

    active: set[str] = set()  # once enabled, an effect stays on through later stages / final FOM

    # With tth_offset_last_stage, the stage that fits the offset (index), else every stage does.
    offset_stage = None
    if config.tth_offset_last_stage and _enabled("tth_offset"):
        listed = [
            i
            for i, st in enumerate(config.stages)
            if st.loss == "log_mse" and "tth_offset" in st.optimize
        ]
        offset_stage = listed[-1] if listed else -1  # -1: no such stage, offset stays at 0
        if offset_stage < 0:
            logger.warning(
                "refine: tth_offset_last_stage is on but no log_mse stage lists tth_offset; "
                "the offset is not fitted and stays at 0"
            )
    offset_off = None  # inputs of the offset-off refit: state and effects before the offset stage

    def _run_stages(stages, active_set: set[str], offset_at: int | None, tag: str = "") -> None:
        """Run (index, stage) pairs in order, adding their effects to active_set. With offset_at
        set, tth_offset is optimized only in the stage with that index (-1: in none)."""
        for i_stage, stage in stages:
            opt_keys = [k for k in stage.optimize if _enabled(k)]
            if offset_at is not None and i_stage != offset_at:
                opt_keys = [k for k in opt_keys if k != "tth_offset"]
            if not opt_keys:
                continue
            active_set |= _stage_effects(opt_keys)

            if stage.loss == "poisson":
                if config.counts_scale is None:
                    raise ValueError(
                        f"stage {stage.name!r} uses loss='poisson' but counts_scale is None"
                    )
                loss_fn = poisson_deviance_loss(config.counts_scale)
            else:
                loss_fn = LOSS_PRESETS[stage.loss]
            stage_mask = mask
            if stage.q_min is not None:
                stage_mask = stage_mask * (q_grid > stage.q_min)
            if stage.q_max is not None:
                stage_mask = stage_mask * (q_grid < stage.q_max)

            _optimize(opt_keys, loss_fn, stage.steps, stage.lr, active_set, stage_mask)

            if verbose:
                logger.info(
                    f"  [refine{tag}] {stage.name:<12} ({stage.loss}) FOM={fom(active_set):.5f}"
                )

    # Fair baseline: so that both fom_before/after measure "film-parameter fit
    # quality", before measuring we keep film parameters **fixed** at AI-init and
    # only align the trivial global scale/offset (i0, bkg). (Previously i0=1, bkg=-7
    # were fixed, so the measured curve's arbitrary intensity scale dominated the
    # before loss, making the improvement look like it came from scale matching
    # rather than film quality.) Since simulate always applies i0/bkg regardless of
    # active, we leave active=∅ here.
    base_keys = [k for k in ("i0", "bkg") if _enabled(k)]
    if base_keys:
        _optimize(base_keys, log_mse_loss, steps=80, lr=0.05, active_set=active, stage_mask=mask)
    fom_before = fom(active)

    stages = list(enumerate(config.stages))
    if offset_stage is None or offset_stage < 0:
        _run_stages(stages, active, offset_stage)
    else:
        _run_stages(stages[:offset_stage], active, offset_stage)
        cur["tth_offset"] = zeros.clone()
        offset_off = ({k: v.clone() for k, v in cur.items()}, set(active))
        _run_stages(stages[offset_stage:], active, offset_stage)

    fom_after = fom(active)
    p_refined = torch.stack([cur["thickness"], cur["roughness"], cur["sld"]], dim=1)
    info: dict = {"fom_before": fom_before, "fom_after": fom_after}
    # Final fitted curve (all active nuisances included) — returned so overlay plots
    # compare against measured under the same conditions (footprint/i0/bkg etc. applied).
    with torch.no_grad():
        info["r_fit"] = simulate(cur, active).detach()
    # Also return the fitted nuisance values (only the active ones).
    for k in ("i0", "bkg", "tth_offset", "dq", "dq_rel", "kalpha2_ratio", "footprint", *OXIDE_KEYS):
        if _enabled(k):
            info[k] = cur[k].detach()
    # What a posterior sampler needs to continue from here: the full state, the effects that are on,
    # the bounds of every key, which keys were fitted, and the forward model itself.
    info["state"] = {k: v.detach() for k, v in cur.items()}
    info["active"] = set(active)
    info["bounds"] = bounds
    info["fitted"] = [k for k in cur if _enabled(k)]
    info["forward"] = simulate

    if _enabled("tth_offset") and "tth_offset" in active:
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
        # The same stages, from the offset's to the last, from the same state with the offset held
        # at 0: if this fits as well or better, the fitted offset did not earn its place
        # (ReflectoLearn#14). Rerunning every later stage keeps the two fits nested for any stages.
        on_state, on_active = dict(cur), set(active)
        with torch.no_grad():
            fom_on = per_curve_log_mse(simulate(on_state, on_active), log_r_obs, mask)
        start, active_off = offset_off
        cur.update(start)
        _run_stages(stages[offset_stage:], active_off, -1, tag=" offset 0")
        with torch.no_grad():
            fom_off = per_curve_log_mse(simulate(cur, active_off), log_r_obs, mask)
        info["offset_off"] = {
            "p": torch.stack([cur["thickness"], cur["roughness"], cur["sld"]], dim=1).detach(),
            "fom": fom_off,
            "fom_offset_fitted": fom_on,
            "better": fom_off < fom_on * (1.0 - config.offset_off_margin),
        }
        cur.update(on_state)
        if info["offset_off"]["better"].any():
            n = int(info["offset_off"]["better"].sum())
            logger.warning(
                f"refine: holding tth_offset at 0 fits better for {n}/{B} curves; "
                f"compare info['offset_off']"
            )
    return p_refined, info


def per_curve_log_mse(r: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """(B,) log10 MSE over each curve's valid points (the refine FOM, per curve)."""
    se = ((torch.log10(r.clamp_min(1e-15)) - log_r_obs) ** 2) * mask
    return se.sum(1) / mask.sum(1).clamp_min(1.0)


def fft_thickness(
    q_grid: Tensor,
    log_r_obs: Tensor,
    mask: Tensor,
    d_range: tuple[float, float],
    sld: Tensor | None = None,
    d_min: float = 60.0,
) -> Tensor:
    """(B,) film thickness from the Kiessig period, independent of the network: the strongest peak of
    the real-space transform of the detrended log R (the Patterson-type peak at z = d).

    Fringes are periodic in the film's own wavevector sqrt(q^2 - q_c^2), not in q; transforming over q
    overestimates d by several percent near the edge. With ``sld`` (B,) [1e-6/Å^2] the transform runs
    over the film wavevector, q_c = 4 sqrt(pi sld 1e-6); a rough SLD is enough.

    Fresnel decay is removed with a quintic fit over the valid region above the critical edge, the
    rest is Hann-windowed and transformed; the estimate is the highest local maximum with z in
    [max(d_min, lo), hi]. d_min keeps out the native oxide and the leakage tail at small z. NaN where
    there are fewer than 40 valid points or no such maximum (films thinner than d_min).
    """
    q = q_grid.detach().double().cpu().numpy()
    y_all = log_r_obs.detach().double().cpu().numpy()
    m_all = mask.detach().cpu().numpy() > 0
    qc2 = (
        16e-6 * np.pi * sld.detach().double().cpu().numpy().clip(0)
        if sld is not None
        else np.zeros(len(y_all))
    )
    z = np.arange(max(d_min, d_range[0]) - 1.0, d_range[1] + 2.0, 1.0)
    out = []
    for y, m, c2 in zip(y_all, m_all, qc2, strict=True):
        ok = np.flatnonzero(m)
        if ok.size < 40:
            out.append(np.nan)
            continue
        start = ok[0] + int(np.argmax(y[ok]))  # skip the plateau and the critical edge
        idx = ok[(ok > start + 5) & (q[ok] ** 2 > 1.2 * c2)]
        if idx.size < 40:
            out.append(np.nan)
            continue
        qq, yy = np.sqrt(q[idx] ** 2 - c2), y[idx]
        yy = yy - np.polyval(np.polyfit(qq, yy, 5), qq)
        yy = yy * np.hanning(yy.size)
        amp = np.abs(np.exp(-1j * np.outer(z, qq)) @ yy)
        peak = (amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:])
        out.append(float(z[1:-1][peak][np.argmax(amp[1:-1][peak])]) if peak.any() else np.nan)
    return torch.tensor(out, dtype=torch.float32, device=log_r_obs.device)


def refine_multistart(
    p_init: Tensor,
    substrate: Tensor,
    q_grid: Tensor,
    log_r_obs: Tensor,
    mask: Tensor,
    param_ranges,
    *,
    wavelength: float,
    config: RefineConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
    thickness_factors: tuple[float, ...] = (0.5, 1.0, 2.0),
    fom_tolerance: float = 0.25,
    fft_weight: float = 1.5,
) -> tuple[Tensor, dict]:
    """refine from several thickness starts and keep the one the fringe period supports.

    A single refine cannot correct a guess off by about a factor of two: the trust region keeps
    thickness within ±trust_margin of its start. So refine also runs from multiples of the network's
    thickness and from the FFT estimate of the Kiessig period (fft_thickness), all in one batch.

    The lowest log-MSE alone is not a safe choice: with the oxide, resolution and Kα2 nuisances free,
    a film two or four times as thick often fits as well or better. The FFT estimate alone is not
    safe either: now and then it locks onto a wrong peak. So each start is scored by
    ln FOM + fft_weight · |ln(d / d_fft)|, a soft prior on the fringe period, and the lowest wins.
    On the benchmark the right start wins for weights between about 0.9 and 2.5. Without an estimate,
    the network's own start wins when its FOM is within ``fom_tolerance`` of the best, else the best
    FOM. info["alias"] lists every start's label, refined thickness and FOM, the FFT estimate and the
    choice.
    """
    device = device or p_init.device
    B = p_init.shape[0]
    g_lo, g_hi = param_ranges.thickness
    d_fft = fft_thickness(
        q_grid.to(device), log_r_obs.to(device), mask.to(device), (g_lo, g_hi), sld=p_init[:, 2]
    )
    labels, starts = [], []
    for f in thickness_factors:
        p = p_init.detach().clone().float().to(device)
        p[:, 0] = (p[:, 0] * f).clamp(g_lo, g_hi)
        starts.append(p)
        labels.append(f"x{f:g}")
    p = p_init.detach().clone().float().to(device)
    p[:, 0] = torch.where(torch.isnan(d_fft), p[:, 0], d_fft).clamp(g_lo, g_hi)
    starts.append(p)
    labels.append("fft")
    S = len(starts)

    def rep(t: Tensor) -> Tensor:
        return t.repeat(S, *([1] * (t.dim() - 1)))

    sub = rep(substrate) if substrate.shape[0] == B else substrate
    obs, msk = rep(log_r_obs.to(device)), rep(mask.to(device))
    p_all, info_all = refine(
        torch.cat(starts),
        sub,
        q_grid,
        obs,
        msk,
        param_ranges,
        wavelength=wavelength,
        config=config,
        device=device,
        verbose=verbose,
    )
    fom = per_curve_log_mse(info_all["r_fit"], obs, msk).view(S, B)
    d_all = p_all[:, 0].view(S, B)
    score = fom.clamp_min(1e-12).log() + fft_weight * (d_all / d_fft.unsqueeze(0)).log().abs()
    by_fft = score.argmin(0)
    allowed = fom <= fom.min(0).values * (1.0 + fom_tolerance)
    ai_row = list(thickness_factors).index(1.0) if 1.0 in thickness_factors else 0
    fallback = torch.where(allowed[ai_row], torch.full_like(by_fft, ai_row), fom.argmin(0))
    best = torch.where(torch.isnan(d_fft), fallback, by_fft)  # (B,) winning start per curve
    pick = best * B + torch.arange(B, device=fom.device)  # rows of the stacked batch

    def select(v):
        if isinstance(v, Tensor) and v.dim() >= 1 and v.shape[0] == S * B:
            return v[pick]
        if isinstance(v, dict):
            return {k: select(x) for k, x in v.items()}
        return v

    # The batch's forward model is built for S·B substrates, so it is not passed on; build a new one
    # with make_forward for the chosen curves.
    info = {k: select(v) for k, v in info_all.items() if k not in ("forward", "bounds")}
    info["bounds"] = {k: (select(lo), select(hi)) for k, (lo, hi) in info_all["bounds"].items()}
    info["fom_after"] = float(fom.gather(0, best.unsqueeze(0)).mean())
    info["alias"] = {
        "starts": labels,
        "thickness": d_all.T.tolist(),
        "fom": fom.T.tolist(),
        "fft_thickness": d_fft.tolist(),
        "chosen": [labels[i] for i in best.tolist()],
    }
    return p_all[pick], info


__all__ = [
    "RefineStage",
    "RefineConfig",
    "likelihood_stages",
    "refine",
    "refine_multistart",
    "make_forward",
    "per_curve_log_mse",
    "fft_thickness",
    "PARAM_KEYS",
    "OXIDE_KEYS",
]
