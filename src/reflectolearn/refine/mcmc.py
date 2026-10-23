"""Posterior sampling after refinement: emcee on the same forward model refine fits with.

refine gives one best fit and no uncertainty. Sampling the posterior around it gives the spread of
each parameter and exposes degeneracies (thickness against SLD, roughness against resolution).

Likelihood: Gaussian in log10 R over the valid q points with an unknown spread exp(lnsigma) that
is fitted with the rest. Without it the curve's own scatter has to be assumed, and when the model
does not describe the data to within that scatter the posterior comes out absurdly narrow. The
measured curve is interpolated onto the fixed q grid, so grid points are not independent; the
log-likelihood is scaled to the number of measured points (``n_independent``), or the intervals
would shrink by about sqrt(grid points / measured points).

Priors are flat within the same bounds refine used (trust region for the film, global ranges for
the nuisances). All walkers are evaluated in one batch on the refine device.

The kept half of the chain is checked against its integrated autocorrelation time tau: estimates are
usable when it spans at least 50 tau. The check is necessary, not sufficient: on a short chain tau
itself is underestimated, so a flag set near the boundary can be optimistic; "tau_margin" (kept steps
/ max tau) shows how far from the boundary the chain is. With refine's twelve parameters free a 2000-step chain is often
several times shorter than that (ReflectoLearn#29); the result reports tau and a warning is logged.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from reflectolearn.refine.pipeline import PARAM_KEYS, RefineConfig, make_forward

LNSIGMA_BOUNDS = (-8.0, 1.0)  # spread of log10 R residuals: e^-8 to e^1
TAU_FACTOR = 50  # kept steps per integrated autocorrelation time for a usable estimate


def sample_posterior(
    info: dict,
    substrate: Tensor,  # (1, 2) the substrate refine used for this curve
    q_grid: Tensor,
    log_r_obs: Tensor,  # (1, Q)
    mask: Tensor,  # (1, Q)
    *,
    wavelength: float,
    config: RefineConfig | None = None,
    n_independent: int | None = None,
    steps: int = 2000,
    walkers_per_dim: int = 4,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Sample the posterior of every parameter refine fitted, for one curve.

    Args:
        info: refine's (or refine_multistart's) info for a single curve (batch of 1).
        n_independent: measured points inside the q grid; defaults to the number of valid grid points.
        steps: emcee steps; the first half is discarded as burn-in.

    Returns:
        {"params": {key: {"median", "lo", "hi"}}, "lnsigma": {...}, "acceptance", "steps", "walkers",
         "samples": (N, K) array of the kept samples, "keys": list, "log_prob_start", "log_prob_max",
         "best": the highest-probability point visited, "tau": integrated autocorrelation time of
         each key and of lnsigma over the kept half, "tau_margin": kept steps / max(tau),
         "converged": tau_margin >= 50}.
        lo/hi are the 16th/84th percentiles (68 % interval).
    """
    import emcee  # optional dependency: pip install reflectolearn[mcmc]

    config = config or RefineConfig()
    device = device or log_r_obs.device
    state = {k: v.reshape(-1)[:1].to(device) for k, v in info["state"].items()}
    active = set(info["active"])
    bounds = {
        k: (
            torch.as_tensor(lo, device=device).reshape(-1)[0].item(),
            torch.as_tensor(hi, device=device).reshape(-1)[0].item(),
        )
        for k, (lo, hi) in info["bounds"].items()
    }
    keys = [
        k for k in info["fitted"] if k in PARAM_KEYS or k in ("i0", "bkg") or _effect_on(k, active)
    ]
    forward = make_forward(
        q_grid, substrate[:1], wavelength=wavelength, config=config, device=device
    )

    obs = log_r_obs[:1].to(device).float()
    m = mask[:1].to(device).float()
    n_valid = float(m.sum())
    weight = (n_independent / n_valid) if n_independent else 1.0

    lo = np.array([bounds[k][0] for k in keys] + [LNSIGMA_BOUNDS[0]])
    hi = np.array([bounds[k][1] for k in keys] + [LNSIGMA_BOUNDS[1]])

    @torch.no_grad()
    def residual_sq(theta: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(theta[:, :-1], dtype=torch.float32, device=device)
        W = t.shape[0]
        vals = {k: v.expand(W).clone() for k, v in state.items()}
        for i, k in enumerate(keys):
            vals[k] = t[:, i]
        r = forward(vals, active)
        se = ((torch.log10(r.clamp_min(1e-15)) - obs) ** 2) * m
        return se.sum(1).double().cpu().numpy()

    def log_prob(theta: np.ndarray) -> np.ndarray:
        out = np.full(theta.shape[0], -np.inf)
        ok = np.all((theta > lo) & (theta < hi), axis=1)
        if ok.any():
            s = np.exp(theta[ok, -1])
            out[ok] = weight * (-0.5 * residual_sq(theta[ok]) / s**2 - n_valid * np.log(s))
        return out

    # Start every walker in a small ball around the refined state, inside the bounds.
    x0 = np.array([state[k].item() for k in keys])
    rms = math.sqrt(residual_sq(np.append(x0, 0.0)[None])[0] / max(n_valid, 1.0))
    x0 = np.append(x0, np.clip(math.log(max(rms, 1e-4)), *LNSIGMA_BOUNDS))
    ndim = len(x0)
    nwalk = max(walkers_per_dim * ndim, 2 * ndim + 2)
    rng = np.random.default_rng(seed)
    width = (hi - lo) * 1e-3
    p0 = np.clip(
        x0 + width * rng.standard_normal((nwalk, ndim)),
        lo + 1e-9 * (hi - lo),
        hi - 1e-9 * (hi - lo),
    )

    sampler = emcee.EnsembleSampler(nwalk, ndim, log_prob, vectorize=True)
    sampler.random_state = np.random.RandomState(seed)
    sampler.run_mcmc(p0, steps, progress=False)
    chain = sampler.get_chain(discard=steps // 2, flat=True)
    tau = emcee.autocorr.integrated_time(sampler.get_chain(discard=steps // 2), quiet=True)
    kept = steps - steps // 2
    tau_margin = float(kept / tau.max())
    converged = tau_margin >= TAU_FACTOR
    if not converged:
        logger.warning(
            f"MCMC: {kept} kept steps < {TAU_FACTOR} x max autocorrelation time {tau.max():.0f};"
            " the intervals are not converged, run a longer chain"
        )
    lp = sampler.get_log_prob(flat=True)
    best = sampler.get_chain(flat=True)[int(np.argmax(lp))]

    def summary(col: np.ndarray) -> dict:
        p16, p50, p84 = np.percentile(col, [16, 50, 84])
        return {"median": float(p50), "lo": float(p16), "hi": float(p84)}

    return {
        "params": {k: summary(chain[:, i]) for i, k in enumerate(keys)},
        "lnsigma": summary(chain[:, -1]),
        "acceptance": float(np.mean(sampler.acceptance_fraction)),
        "steps": steps,
        "walkers": nwalk,
        "keys": keys,
        "samples": chain[:, :-1],
        # The chain's best point vs the refined start: a gain of many units means refine stopped short
        # of a better mode, which the median then reports.
        "log_prob_start": float(log_prob(x0[None])[0]),
        "log_prob_max": float(lp.max()),
        "best": dict(zip(keys, best[:-1].tolist(), strict=True)),
        "tau": dict(zip([*keys, "lnsigma"], tau.tolist(), strict=True)),
        "tau_margin": tau_margin,
        "converged": converged,
    }


def _effect_on(key: str, active: set[str]) -> bool:
    """A nuisance key belongs in the posterior only if its effect was switched on in refine."""
    effect = {
        "tth_offset": "tth_offset",
        "dq": "dq",
        "dq_rel": "dq",
        "kalpha2_ratio": "kalpha2",
        "footprint": "footprint",
        "oxide_thickness": "oxide",
        "oxide_sld": "oxide",
        "oxide_roughness": "oxide",
    }.get(key)
    return effect in active


__all__ = ["sample_posterior"]
