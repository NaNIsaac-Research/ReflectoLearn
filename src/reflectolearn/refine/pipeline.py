"""Gradient-based physics refinement of AI parameter predictions.

XRefine 의 refinement 원리를 우리 코드에 맞게 이식한다:

  CNN 예측(초기 추정) -> 물리 시뮬레이션 곡선을 측정 곡선에 직접 피팅하여 보정.

XRefine 은 모델이 미분 불가능하다고 보고 `scipy.differential_evolution`(gradient-free)
을 썼지만, 우리 ``AbelesMatrix`` 는 autograd 로 완전히 미분 가능하다. 따라서 동일한
원리를 **Adam 경사하강**으로 구현한다. 핵심 요소는 그대로 가져온다:

  * 다단계(stage) 전략: scale/footprint -> 막 구조 -> fine tuning
  * 단계별 손실 선택 (linear / correlation / log 등, :mod:`reflectolearn.refine.losses`)
  * AI 추정값 주변으로 탐색 범위를 좁히는 trust-region bound
  * 유효 q 마스킹 + FOM(LogMSE) 로 개선 추적

탐색 범위 제약은 latent 변수 ``z`` 를 sigmoid 로 [lo, hi] 에 매핑해 부드럽게 처리하므로
경사하강 중에도 파라미터가 물리적 범위를 벗어나지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from loguru import logger
from torch import Tensor

from reflectolearn.physics import AbelesMatrix
from reflectolearn.refine.losses import LOSS_PRESETS, log_mse_loss

# 예측/피팅 대상 막(film) 파라미터 (config.param_ranges 와 동일 순서)
PARAM_KEYS = ("thickness", "roughness", "sld")


@dataclass
class RefineStage:
    """한 정밀화 단계: 어떤 파라미터를 어떤 손실로 몇 step 최적화할지."""

    name: str
    optimize: tuple[str, ...]          # PARAM_KEYS + ("i0", "bkg") 의 부분집합
    loss: str = "log_mse"              # LOSS_PRESETS 의 키
    steps: int = 200
    lr: float = 0.03
    q_min: float | None = None         # [Å^-1] 이 단계에서 볼 q 하한
    q_max: float | None = None         # [Å^-1] 이 단계에서 볼 q 상한


def _default_stages() -> list[RefineStage]:
    """XRefine 기본 전략을 단순 막 모델에 맞게 축약한 3단계."""
    return [
        # 1) 절대 강도/배경 정합: i0, bkg 를 선형 손실로 맞춘다.
        RefineStage("1.scale", optimize=("i0", "bkg"), loss="linear_mse", steps=120, lr=0.05),
        # 2) 막 구조(주기): 두께/SLD 를 상관 손실로 맞춘다 (스케일·감쇠 불변).
        RefineStage("2.structure", optimize=("thickness", "sld"), loss="correlation", steps=200, lr=0.03),
        # 3) fine tuning: 거칠기 포함 전체를 log 손실로 정밀화.
        RefineStage("3.fine", optimize=("thickness", "roughness", "sld", "i0", "bkg"),
                    loss="log_mse", steps=250, lr=0.01),
    ]


@dataclass
class RefineConfig:
    trust_margin: float = 0.4                       # AI 추정값 기준 trust-region 반폭 비율
    fit_i0: bool = True                             # 강도 스케일 i0 동시 피팅
    fit_bkg: bool = True                            # 배경(log10) bkg 동시 피팅
    i0_bounds: tuple[float, float] = (0.5, 2.0)
    bkg_bounds: tuple[float, float] = (-10.0, -4.0)
    bkg_init: float = -7.0
    stages: list[RefineStage] = field(default_factory=_default_stages)


def _film_bounds(param_ranges, p_init: Tensor, margin: float) -> dict[str, tuple[Tensor, Tensor]]:
    """trust-region ∩ 전역 범위. 반폭은 |추정값|에 비례하되 범위의 10% 로 하한 (0 붕괴 방지).

    p_init 은 호출 전 전역 범위로 clamp 되어 있다고 가정하므로 양 끝 모두 [g_lo, g_hi] 안에 남는다.
    """
    bounds: dict[str, tuple[Tensor, Tensor]] = {}
    for i, key in enumerate(PARAM_KEYS):
        g_lo, g_hi = getattr(param_ranges, key)
        span = g_hi - g_lo
        guess = p_init[:, i]
        half = margin * guess.abs().clamp_min(0.1 * span)
        lo = (guess - half).clamp(g_lo, g_hi)
        hi = (guess + half).clamp(g_lo, g_hi)
        lo = torch.minimum(lo, hi - 1e-3 * span)     # lo < hi 보장 (hi 는 g_hi 이하 유지)
        bounds[key] = (lo, hi)
    return bounds


def refine(
    p_init: Tensor,            # (B, 3) 물리값 [thickness, roughness, sld]  (AI 초기 추정)
    substrate: Tensor,         # (B, 2) [sub_sld, sub_rough]  (고정)
    q_grid: Tensor,            # (Q,) [Å^-1]
    log_r_obs: Tensor,         # (B, Q) 측정 반사도 log10
    mask: Tensor,              # (B, Q) 유효 q 마스크
    param_ranges,              # config.param_ranges (.thickness 등 [lo, hi])
    *,
    wavelength: float,
    config: RefineConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[Tensor, dict]:
    """AI 추정 막 파라미터를 측정 곡선에 맞춰 경사하강으로 정밀화.

    Returns:
        p_refined: (B, 3) 정밀화된 [thickness, roughness, sld]
        info: {"fom_before", "fom_after"(스칼라 LogMSE), "i0", "bkg"(B,)}
    """
    config = config or RefineConfig()
    device = device or p_init.device

    p_init = p_init.detach().to(device).float()
    # AI 예측이 물리 범위를 벗어날 수 있으므로 전역 범위로 사영(clamp) 후 시작.
    g_lo = torch.tensor([getattr(param_ranges, k)[0] for k in PARAM_KEYS], device=device)
    g_hi = torch.tensor([getattr(param_ranges, k)[1] for k in PARAM_KEYS], device=device)
    p_init = p_init.clamp(g_lo, g_hi)
    substrate = substrate.to(device).float()
    q_grid = q_grid.to(device).float()
    log_r_obs = log_r_obs.to(device).float()
    mask = mask.to(device).float()

    B = p_init.shape[0]
    zeros = torch.zeros(B, device=device)
    sub_sld, sub_rough = substrate[:, 0], substrate[:, 1]
    abeles = AbelesMatrix(device=device)

    def simulate(vals: dict[str, Tensor]) -> Tensor:
        """막+기판 파라미터 -> 반사도. i0/bkg 로 절대 스케일까지 모델링."""
        sld_t = torch.stack([zeros, vals["sld"], sub_sld], dim=1)
        thk_t = torch.stack([zeros, vals["thickness"]], dim=1)
        rgh_t = torch.stack([vals["roughness"], sub_rough], dim=1)
        R = abeles.forward(q_grid, thk_t, rgh_t, sld_t, wavelength=wavelength)
        R = torch.nan_to_num(R, nan=1e-15, posinf=1.0, neginf=1e-15).clamp_min(1e-15)
        return R * vals["i0"].unsqueeze(-1) + torch.pow(10.0, vals["bkg"]).unsqueeze(-1)

    # 경계: 막은 trust-region, i0/bkg 는 전역 범위.
    bounds: dict[str, tuple[Tensor, Tensor]] = _film_bounds(param_ranges, p_init, config.trust_margin)
    bounds["i0"] = (torch.tensor(config.i0_bounds[0], device=device),
                    torch.tensor(config.i0_bounds[1], device=device))
    bounds["bkg"] = (torch.tensor(config.bkg_bounds[0], device=device),
                     torch.tensor(config.bkg_bounds[1], device=device))

    # 현재 물리값 상태 (모두 detach 된 (B,) 텐서)
    cur: dict[str, Tensor] = {
        "thickness": p_init[:, 0].clone(),
        "roughness": p_init[:, 1].clone(),
        "sld": p_init[:, 2].clone(),
        "i0": torch.ones(B, device=device),
        "bkg": torch.full((B,), config.bkg_init, device=device),
    }

    @torch.no_grad()
    def fom(vals: dict[str, Tensor]) -> float:
        return float(log_mse_loss(simulate(vals), log_r_obs, mask).item())

    fom_before = fom(cur)

    for stage in config.stages:
        opt_keys = [
            k for k in stage.optimize
            if k in PARAM_KEYS or (k == "i0" and config.fit_i0) or (k == "bkg" and config.fit_bkg)
        ]
        if not opt_keys:
            continue

        loss_fn = LOSS_PRESETS[stage.loss]
        stage_mask = mask
        if stage.q_min is not None:
            stage_mask = stage_mask * (q_grid > stage.q_min)
        if stage.q_max is not None:
            stage_mask = stage_mask * (q_grid < stage.q_max)

        # latent z 초기화: value == 현재값 이 되도록 logit 으로 역산.
        z: dict[str, Tensor] = {}
        for k in opt_keys:
            lo, hi = bounds[k]
            frac = ((cur[k] - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
            z[k] = torch.log(frac / (1 - frac)).detach().clone().requires_grad_(True)

        optimizer = torch.optim.Adam(list(z.values()), lr=stage.lr)
        for _ in range(stage.steps):
            optimizer.zero_grad(set_to_none=True)
            vals = dict(cur)  # 비최적화 키는 상수(detach)로 사용
            for k in opt_keys:
                lo, hi = bounds[k]
                vals[k] = lo + (hi - lo) * torch.sigmoid(z[k])
            loss = loss_fn(simulate(vals), log_r_obs, stage_mask)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for k in opt_keys:
                lo, hi = bounds[k]
                cur[k] = (lo + (hi - lo) * torch.sigmoid(z[k])).detach()

        if verbose:
            logger.info(f"  [refine] {stage.name:<12} ({stage.loss}) FOM={fom(cur):.5f}")

    fom_after = fom(cur)
    p_refined = torch.stack([cur["thickness"], cur["roughness"], cur["sld"]], dim=1)
    info = {
        "fom_before": fom_before,
        "fom_after": fom_after,
        "i0": cur["i0"].detach(),
        "bkg": cur["bkg"].detach(),
    }
    return p_refined, info


__all__ = ["RefineStage", "RefineConfig", "refine", "PARAM_KEYS"]
