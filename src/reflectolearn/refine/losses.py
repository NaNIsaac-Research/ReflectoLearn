"""Curve-fitting loss functions for physics refinement.

All losses follow the same signature over the (B, Q) tensor convention:

    r_sim:     (B, Q) simulated reflectivity (linear, > 0)
    log_r_obs: (B, Q) log10 of observed reflectivity
    mask:      (B, Q) valid-q mask (1 = valid, 0 = ignore; float or bool accepted)

Returns a scalar loss over the batch (per-sample masked mean -> batch mean).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]

_EPS = 1e-12


def _masked_mean(err: Tensor, mask: Tensor) -> Tensor:
    """Mean over the masked region per sample, then batch mean. (B, Q) -> scalar."""
    m = mask.to(err.dtype)
    per_sample = (err * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    return per_sample.mean()


def log_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """MSE in log10 space. Sensitive to dynamic-range (fringe) shape such as roughness/attenuation."""
    log_r_sim = torch.log10(r_sim.clamp_min(_EPS))
    return _masked_mean((log_r_sim - log_r_obs) ** 2, mask)


def linear_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """MSE in linear space. Sensitive to absolute intensity (scale) near the critical angle."""
    r_obs = 10.0**log_r_obs
    return _masked_mean((r_sim - r_obs) ** 2, mask)


def correlation_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """1 - Pearson correlation. Invariant to scale/attenuation, so strong at matching fringe frequency (period)."""
    log_r_sim = torch.log10(r_sim.clamp_min(_EPS))
    m = mask.to(log_r_sim.dtype)
    n = m.sum(-1).clamp_min(1.0)
    sim_mean = (log_r_sim * m).sum(-1) / n
    obs_mean = (log_r_obs * m).sum(-1) / n
    vx = (log_r_sim - sim_mean.unsqueeze(-1)) * m
    vy = (log_r_obs - obs_mean.unsqueeze(-1)) * m
    num = (vx * vy).sum(-1)
    den = torch.sqrt((vx**2).sum(-1)) * torch.sqrt((vy**2).sum(-1)) + 1e-9
    return (1.0 - num / den).mean()


def log_gradient_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """Value MSE + gradient (Sobolev) MSE. Strongly penalizes solutions that flatten the curve (vanishing slope)."""
    log_r_sim = torch.log10(r_sim.clamp_min(_EPS))
    diff_sim = log_r_sim[:, 1:] - log_r_sim[:, :-1]
    diff_obs = log_r_obs[:, 1:] - log_r_obs[:, :-1]
    mask_diff = mask[:, 1:].to(r_sim.dtype) * mask[:, :-1].to(r_sim.dtype)
    val = _masked_mean((log_r_sim - log_r_obs) ** 2, mask)
    grad = _masked_mean((diff_sim - diff_obs) ** 2, mask_diff)
    return val + 10.0 * grad


def poisson_deviance_loss(counts_scale: float) -> LossFn:
    """Poisson deviance per point for counts = counts_scale · R (counts_scale: counts at R = 1).

    Unlike the log losses this is the likelihood of counting data: each point is weighted by its
    counts, and points with zero counts inside the mask contribute too. Built per curve set because
    it needs the count scale, so it is not in LOSS_PRESETS; a stage selects it with loss="poisson".

    It is the likelihood only if counts_scale · 10**log_r_obs are the raw per-point counts on the
    measured points. Values interpolated onto another grid (as `predict.preprocess_measured` does
    for the network) or averaged over equal q are not Poisson counts, and the deviance of such
    values is only an approximation. Pass refine the measured q grid and the raw counts instead.
    """

    def loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
        lam = counts_scale * r_sim.double().clamp_min(_EPS)
        y = counts_scale * torch.pow(10.0, log_r_obs.double())
        dev = 2.0 * (lam - y + torch.special.xlogy(y, y / lam))
        return _masked_mean(dev, mask).to(r_sim.dtype)

    return loss


def hybrid_linear_log_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """Mix of linear (intensity/critical angle) + log (fringe/dynamic range), linear:log = 20:1."""
    return 20.0 * linear_mse_loss(r_sim, log_r_obs, mask) + log_mse_loss(r_sim, log_r_obs, mask)


# Presets (name -> function). Referenced by these keys in refine.RefineStage.loss.
LOG_MSE = log_mse_loss
LINEAR_MSE = linear_mse_loss
CORRELATION = correlation_loss
LOG_GRADIENT = log_gradient_loss
HYBRID = hybrid_linear_log_loss

LOSS_PRESETS: dict[str, LossFn] = {
    "log_mse": LOG_MSE,
    "linear_mse": LINEAR_MSE,
    "correlation": CORRELATION,
    "log_gradient": LOG_GRADIENT,
    "hybrid": HYBRID,
}

__all__ = [
    "LossFn",
    "LOSS_PRESETS",
    "log_mse_loss",
    "linear_mse_loss",
    "correlation_loss",
    "log_gradient_loss",
    "hybrid_linear_log_loss",
    "poisson_deviance_loss",
    "LOG_MSE",
    "LINEAR_MSE",
    "CORRELATION",
    "LOG_GRADIENT",
    "HYBRID",
]
