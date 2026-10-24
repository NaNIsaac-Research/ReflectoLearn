"""
Abeles matrix method.

Physics engine for X-ray reflectivity calculation, optimized for batch
processing and gradient descent.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


def abeles_core(q: Tensor, thickness: Tensor, roughness: Tensor, sld: Tensor) -> Tensor:
    """
    Core Abeles matrix calculation.

    Length convention (important):
        There are N media and N-1 interfaces.
        thickness / roughness are per-interface, so they are 1 shorter than sld.

    Args:
        q:         (B, Q, 1) complex  - scattering vector (k_z = q/2)
        sld:       (B, 1, N) complex  - relative SLD per medium (ambient=0 reference)
        thickness: (B, 1, N-1) real   - thickness of each finite layer/interface
        roughness: (B, 1, N-1) real   - roughness of each interface (Nevot-Croce)

    Returns:
        (B, Q) real - reflectivity R(q) = |M_10 / M_00|^2
    """
    # k_z (both q and sld are complex)
    k_n = torch.sqrt((q / 2.0) ** 2 - 4.0 * math.pi * sld)

    # Per-interface wavevectors (length N-1)
    k_curr, k_next = k_n[..., :-1], k_n[..., 1:]

    # Phase shift (beta)
    beta = 1j * thickness * k_curr
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    # Fresnel coefficient with roughness (Nevot-Croce)
    sig2 = roughness**2
    denom = k_curr + k_next
    r_n = ((k_curr - k_next) / denom) * torch.exp(-2.0 * k_curr * k_next * sig2)

    # Matrix elements
    m_00 = exp_beta
    m_01 = r_n * exp_beta
    m_10 = r_n * exp_m_beta
    m_11 = exp_m_beta

    # Accumulate the product over interfaces, starting with the first one
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

    def forward(
        self,
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
        sample_length: Tensor | None = None,
        beam_width: Tensor | None = None,
        wavelength: float = 1.5406,
    ) -> Tensor:
        """
        Pre/post-processing layer wrapping the physics kernel (abeles_core).

        Responsibilities:
          1) Input preparation: dtype casting, relative-SLD conversion, unit scaling, dimension alignment
          2) Call abeles_core (pure physics calculation)
          3) Footprint correction (optional)
        As an nn.Module, its operations are recorded on the autograd graph and are differentiable.

        Args:
            q:         (B, Q) real - scattering vector [1/Angstrom]
            thickness: (B, N-1)    - thickness of the upper medium at each interface [Angstrom];
                                     the first entry is the ambient's and has no effect
            roughness: (B, N-1)    - rms roughness of each interface, top first [Angstrom]
            sld:       (B, N)      - SLD per medium, ambient first [1e-6/Angstrom^2],
                                     real or complex; computed in complex64
            sample_length, beam_width: for footprint correction (optional)
            wavelength: X-ray wavelength [Angstrom], defaults to Cu K-alpha
        Returns:
            (B, Q) real - reflectivity
        """
        if self.device is None:
            self.device = q.device

        # [Optimization] Avoid CPU sync stalls by converting wavelength from a
        # Python scalar into a 0-d tensor so it participates on-device.
        wl_tensor = torch.tensor(wavelength, device=self.device, dtype=torch.float32)

        q_f = q.float()

        # SLD & Complex handling (Strict Complex Mode)
        if torch.is_complex(sld):
            sld_c = sld.to(dtype=torch.complex64)
        else:
            sld_c = sld.float().to(dtype=torch.complex64)

        # Subtract ambient SLD (relative SLD).
        # The tiny absorption term (+1j*1e-9) represents physical absorption and
        # also guarantees k != 0 in every layer, preventing the denominator from
        # being exactly zero (so no epsilon guard is needed).
        amb_sld = sld_c[:, 0:1]
        sld_comp = (sld_c - amb_sld) * 1e-6 + 1j * 1e-9  # 1e-6 scale for SLD unit

        # Expand dimensions for broadcasting
        q_in = q_f.unsqueeze(-1).to(dtype=torch.complex64)  # (B, Q, 1)
        sld_in = sld_comp.unsqueeze(1)  # (B, 1, N)
        t_in = thickness.float().unsqueeze(1)  # (B, 1, N-1)
        r_in = roughness.float().unsqueeze(1)  # (B, 1, N-1)

        reflectivity = abeles_core(q_in, t_in, r_in, sld_in)

        # R_corr = R * erf((L * sin(alpha)) / (2*sqrt(2) * sigma_beam))
        if sample_length is not None and beam_width is not None:
            sl = sample_length.float()
            bw = beam_width.float()

            # sin(theta) approx = lambda * q / (4 * pi), clamped to the physical domain
            sin_alpha = torch.clamp(q_f * wl_tensor / (4.0 * math.pi), -1.0, 1.0)

            # Beam sigma (FWHM -> Sigma)
            sigma_b = bw / 2.3548

            # Error function argument (2.8284 ~= 2 * sqrt(2))
            arg = (sl * sin_alpha) / (2.8284 * sigma_b)

            reflectivity = reflectivity * torch.erf(arg)

        return reflectivity


def abeles(*args, **kwargs):
    """Functional interface wrapper."""
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)
