"""
Abeles Matrix Method (Final Optimization)
=========================================
Physics engine for X-ray reflectivity calculation.
Optimized for batch processing and gradient descent.
"""
import math
import torch
import torch.nn as nn
from torch import Tensor


def abeles_core(q: Tensor, thickness: Tensor, roughness: Tensor, sld: Tensor) -> Tensor:
    """
    Core Abeles Matrix Calculation.

    Length convention (중요):
        매질(media)은 N개, 인터페이스(interface)는 N-1개.
        thickness / roughness 는 인터페이스 단위라서 sld 보다 길이가 1 짧다.

    Args:
        q:         (B, Q, 1) complex  - scattering vector (k_z = q/2)
        sld:       (B, 1, N) complex  - 매질별 상대 SLD (ambient=0 기준)
        thickness: (B, 1, N-1) real   - 각 인터페이스/유한층 두께
        roughness: (B, 1, N-1) real   - 각 인터페이스 거칠기 (Nevot-Croce)

    Returns:
        (B, Q) real - reflectivity R(q) = |M_10 / M_00|^2
    """
    # k_z calculation (q, sld 모두 complex)
    k_n = torch.sqrt((q / 2.0)**2 - 4.0 * math.pi * sld)

    # Layer boundaries -> 인터페이스 단위 (길이 N-1)
    k_curr, k_next = k_n[..., :-1], k_n[..., 1:]

    # Phase shift (beta)
    beta = 1j * thickness * k_curr
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    # Fresnel coefficient with roughness (Nevot-Croce)
    sig2 = roughness ** 2
    denom = k_curr + k_next
    r_n = ((k_curr - k_next) / denom) * torch.exp(-2.0 * k_curr * k_next * sig2)

    # Matrix elements
    m_00 = exp_beta
    m_01 = r_n * exp_beta
    m_10 = r_n * exp_m_beta
    m_11 = exp_m_beta

    # Matrix Multiplication (Accumulation)
    # Start with first interface
    m_accum_00, m_accum_01 = m_00[..., 0], m_01[..., 0]
    m_accum_10, m_accum_11 = m_10[..., 0], m_11[..., 0]

    num_layers = thickness.shape[2]
    for i in range(1, num_layers):
        # Current layer matrix elements
        e, f, g, h = m_00[..., i], m_01[..., i], m_10[..., i], m_11[..., i]

        # Previous accumulation state
        a, b, c, d = m_accum_00, m_accum_01, m_accum_10, m_accum_11

        # Matrix Multiply: M_acc = M_acc * M_next
        m_accum_00 = a * e + b * g
        m_accum_01 = a * f + b * h
        m_accum_10 = c * e + d * g
        m_accum_11 = c * f + d * h

    # Reflectivity R = |M_10 / M_00|^2
    quotient = m_accum_10 / m_accum_00
    return quotient.real**2 + quotient.imag**2


class AbelesMatrix(nn.Module):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device

    def forward(self,
                q: Tensor,
                thickness: Tensor,
                roughness: Tensor,
                sld: Tensor,
                sample_length: Tensor | None = None,
                beam_width: Tensor | None = None,
                wavelength: float = 1.5406) -> Tensor:
        """
        물리 커널(abeles_core)을 감싸는 전처리/후처리 레이어.

        역할:
          1) 입력 정제: dtype 캐스팅, 상대 SLD 변환, 단위 스케일, 차원 정렬
          2) abeles_core 호출 (순수 물리 계산)
          3) footprint 보정 (선택)
        nn.Module 이므로 호출 시 연산이 autograd 그래프에 올라가 미분 가능.

        Args:
            q:         (B, Q) real - scattering vector
            thickness: (B, N-1)    - 인터페이스 두께
            roughness: (B, N-1)    - 인터페이스 거칠기
            sld:       (B, N)      - 매질별 SLD (real 또는 complex)
            sample_length, beam_width: footprint 보정용 (선택)
            wavelength: X-ray 파장 [Angstrom], 기본 Cu K-alpha
        Returns:
            (B, Q) real - reflectivity
        """
        if self.device is None: self.device = q.device

        # [Optimization] CPU Sync Blocking
        # wavelength를 Scalar가 아닌 0-d Tensor로 변환하여 연산에 참여시킴
        wl_tensor = torch.tensor(wavelength, device=self.device, dtype=torch.float32)

        # 1. Input Preparation
        q_f = q.float()

        # SLD & Complex handling (Strict Complex Mode)
        if torch.is_complex(sld):
            sld_c = sld.to(dtype=torch.complex64)
        else:
            sld_c = sld.float().to(dtype=torch.complex64)

        # Subtract ambient SLD (Relative SLD)
        # 미세 흡수항(+1j*1e-9): 물리적 흡수이자, 모든 층에서 k != 0 을 보장하여
        # 분모가 정확히 0이 되는 일을 원천 차단한다 (=> epsilon 방어 불필요).
        amb_sld = sld_c[:, 0:1]
        sld_comp = (sld_c - amb_sld) * 1e-6 + 1j * 1e-9  # 1e-6 scale for SLD unit

        # Expand dimensions for broadcasting
        q_in = q_f.unsqueeze(-1).to(dtype=torch.complex64)  # (B, Q, 1)
        sld_in = sld_comp.unsqueeze(1)                       # (B, 1, N)
        t_in = thickness.float().unsqueeze(1)                # (B, 1, N-1)
        r_in = roughness.float().unsqueeze(1)                # (B, 1, N-1)

        # 2. Physics Core
        reflectivity = abeles_core(q_in, t_in, r_in, sld_in)

        # 3. Footprint Correction
        # R_corr = R * erf((L * sin(alpha)) / (2*sqrt(2) * sigma_beam))
        if sample_length is not None and beam_width is not None:
            sl = sample_length.float()
            bw = beam_width.float()

            # sin(theta) approx = lambda * q / (4 * pi), 물리적 정의역으로 clamp
            sin_alpha = torch.clamp(q_f * wl_tensor / (4.0 * math.pi), -1.0, 1.0)

            # Beam sigma (FWHM -> Sigma)
            sigma_b = bw / 2.3548

            # Error function argument (2.8284 ~= 2 * sqrt(2))
            arg = (sl * sin_alpha) / (2.8284 * sigma_b)

            reflectivity = reflectivity * torch.erf(arg)

        return reflectivity


def abeles(*args, **kwargs):
    """Functional interface wrapper"""
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)