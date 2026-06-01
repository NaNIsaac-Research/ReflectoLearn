"""Curve-fitting loss functions for physics refinement.

XRefine refine.losses 의 원리를 우리 (B, Q) 텐서 규약에 맞게 이식한 것.
모든 손실은 동일한 시그니처를 따른다:

    r_sim:     (B, Q) 시뮬레이션 반사도 (linear, > 0)
    log_r_obs: (B, Q) 관측 반사도의 log10 값
    mask:      (B, Q) 유효 q 마스크 (1 = 유효, 0 = 무시; float/bool 모두 허용)

반환값은 배치에 대한 스칼라 손실(샘플별 마스크 평균 -> 배치 평균).
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]

_EPS = 1e-12


def _masked_mean(err: Tensor, mask: Tensor) -> Tensor:
    """샘플별로 마스크 영역 평균을 낸 뒤 배치 평균. (B, Q) -> scalar."""
    m = mask.to(err.dtype)
    per_sample = (err * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    return per_sample.mean()


def log_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """log10 공간 MSE. 거칠기/감쇠 같은 동적 영역(fringe) 형상에 민감."""
    log_r_sim = torch.log10(r_sim.clamp_min(_EPS))
    return _masked_mean((log_r_sim - log_r_obs) ** 2, mask)


def linear_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """선형 공간 MSE. 임계각 부근의 절대 강도(scale)에 민감."""
    r_obs = 10.0 ** log_r_obs
    return _masked_mean((r_sim - r_obs) ** 2, mask)


def correlation_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """1 - Pearson 상관. 스케일/감쇠에 불변이라 fringe 주파수(주기) 정합에 강함."""
    log_r_sim = torch.log10(r_sim.clamp_min(_EPS))
    m = mask.to(log_r_sim.dtype)
    n = m.sum(-1).clamp_min(1.0)
    sim_mean = (log_r_sim * m).sum(-1) / n
    obs_mean = (log_r_obs * m).sum(-1) / n
    vx = (log_r_sim - sim_mean.unsqueeze(-1)) * m
    vy = (log_r_obs - obs_mean.unsqueeze(-1)) * m
    num = (vx * vy).sum(-1)
    den = torch.sqrt((vx ** 2).sum(-1)) * torch.sqrt((vy ** 2).sum(-1)) + 1e-9
    return (1.0 - num / den).mean()


def log_gradient_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """값 MSE + 기울기(Sobolev) MSE. 곡선이 밋밋해지는(기울기 소실) 해를 강하게 벌점."""
    log_r_sim = torch.log10(r_sim.clamp_min(_EPS))
    diff_sim = log_r_sim[:, 1:] - log_r_sim[:, :-1]
    diff_obs = log_r_obs[:, 1:] - log_r_obs[:, :-1]
    mask_diff = mask[:, 1:].to(r_sim.dtype) * mask[:, :-1].to(r_sim.dtype)
    val = _masked_mean((log_r_sim - log_r_obs) ** 2, mask)
    grad = _masked_mean((diff_sim - diff_obs) ** 2, mask_diff)
    return val + 10.0 * grad


def hybrid_linear_log_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """선형(강도/임계각) + log(fringe/동적범위) 혼합 (linear:log = 20:1)."""
    return 20.0 * linear_mse_loss(r_sim, log_r_obs, mask) + log_mse_loss(r_sim, log_r_obs, mask)


# Presets (이름 -> 함수). refine.RefineStage.loss 에서 이 키로 참조한다.
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
    "LOG_MSE",
    "LINEAR_MSE",
    "CORRELATION",
    "LOG_GRADIENT",
    "HYBRID",
]
