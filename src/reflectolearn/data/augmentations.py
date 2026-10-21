"""Realistic XRR augmentations: instrument effects applied to linear R curves.

Contract: forward(q, R) takes linear-scale reflectivity R of shape (B, Q) and
the q grid (Q,), and returns linear-scale R of shape (B, Q). Randomness is
drawn fresh here, since the streaming generator calls this once per batch.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from reflectolearn.config import Config


class XRRAugmentor:
    """Apply realistic instrument effects to reflectivity curves.

    All ops operate on linear-scale R of shape (B, Q) -> (B, Q). Effects:
      - footprint attenuation at low angles (erf geometric correction)
      - resolution (Gaussian q-smearing)
      - i0  : per-sample intensity scaling   (R * i0)
      - bkg : additive background            (+ 10**bkg)
      - Poisson counting noise
      - low/high angle masking (sets points to 0 -> masked downstream)
    Each effect is gated independently by `config.augmentation.prob`.
    """

    def __init__(self, config: Config, q: torch.Tensor):
        self.config = config
        self.q = q
        aug = config.augmentation
        self.prob = aug.prob
        self.i0_range = aug.i0
        self.bkg_range = aug.bkg
        self.res_sigma_range = aug.res_sigma_range
        self.poisson_log_scale = aug.poisson_log_scale
        self.do_footprint = aug.do_footprint
        self.beam_width_range = aug.beam_width_range
        self.sample_len_range = aug.sample_len_range
        self.wavelength = config.simulation.wavelength

    def _maybe(self) -> bool:
        return torch.rand(1).item() < self.prob

    @staticmethod
    def _u(lo: float, hi: float) -> float:
        return torch.empty(1).uniform_(lo, hi).item()

    def _footprint(
        self, q: torch.Tensor, R: torch.Tensor, beam_width: float, sample_len: float
    ) -> torch.Tensor:
        """Multiply R by an erf footprint factor in [0, 1] (low-angle cutoff)."""
        sin_theta = (q * self.wavelength / (4.0 * math.pi)).clamp(-1.0, 1.0)
        sigma_beam = beam_width / 2.3548  # FWHM -> sigma
        arg = (sample_len * sin_theta) / (2.0 * math.sqrt(2.0) * sigma_beam + 1e-12)
        factor = torch.erf(arg.clamp_min(0.0))  # (Q,)
        return R * factor.view(1, -1)

    @staticmethod
    def _smear(q: torch.Tensor, R: torch.Tensor, sigma_q: float) -> torch.Tensor:
        """Gaussian convolution of R along q with std sigma_q (in q units)."""
        Q = R.shape[1]
        dq = ((q[-1] - q[0]) / (Q - 1)).abs().item()
        if dq <= 0 or sigma_q <= 0:
            return R
        sigma_px = sigma_q / dq
        half = max(int(round(3 * sigma_px)), 1)
        if half < 1:
            return R
        x = torch.arange(-half, half + 1, device=R.device, dtype=R.dtype)
        kernel = torch.exp(-0.5 * (x / sigma_px) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)
        inp = R.unsqueeze(1)  # (B, 1, Q)
        inp = F.pad(inp, (half, half), mode="replicate")
        out = F.conv1d(inp, kernel)  # (B, 1, Q)
        return out.squeeze(1)

    def forward(self, q: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply instrument effects.

        R: (B, Q) linear -> returns (R, mask):
          R    : (B, Q) linear, instrument effects applied (still > 0)
          mask : (B, Q) in {0, 1}; 0 marks angles that were NOT measured
                 (low/high-angle cutoff). The generator uses this to fill the
                 masked region with the log-floor (-15) consistently, exactly
                 like the offline exp07 pipeline.
        """
        R = R.to(torch.float32)
        B, Q = R.shape
        device = R.device
        q = q.to(R.dtype).to(device)
        mask = torch.ones(B, Q, device=device, dtype=R.dtype)

        # 1. Footprint attenuation (low-angle geometric correction)
        if self.do_footprint and self._maybe():
            beam_width = self._u(*self.beam_width_range)
            sample_len = self._u(*self.sample_len_range)
            R = self._footprint(q, R, beam_width, sample_len)

        # 2. Resolution smearing (Gaussian convolution in q)
        if self._maybe():
            sigma = self._u(*self.res_sigma_range)
            R = self._smear(q, R, sigma)

        # 3. Intensity scale i0 (per-sample multiplicative)
        if self._maybe():
            i0 = torch.empty(B, 1, device=device).uniform_(*self.i0_range)
            R = R * i0

        # 4. Additive background (per-sample, 10**bkg with bkg in log10)
        if self._maybe():
            bkg_log = torch.empty(B, 1, device=device).uniform_(*self.bkg_range)
            R = R + torch.pow(torch.tensor(10.0), bkg_log)

        # 5. Poisson counting noise (counts = R*s -> sample -> rescale).
        #    s must be large (1e5~1e8); otherwise R(~1e-6) gives lam~0 and the
        #    high-q signal is destroyed. s is per-sample (signal-dependent SNR).
        if self._maybe():
            s = torch.pow(
                torch.tensor(10.0),
                torch.empty(B, 1, device=device).uniform_(*self.poisson_log_scale),
            )
            lam = (R * s).clamp_min(0.0)
            R = torch.poisson(lam) / s

        # 6. Low/high angle masking -> recorded in `mask`, NOT zeroed in R.
        #    The generator fills mask==0 with the log-floor so the network sees
        #    a *consistent* "no data" signal instead of random -15 cliffs.
        if self._maybe():
            idx = torch.arange(Q, device=device).view(1, Q)
            n_low = torch.randint(0, max(Q // 20, 1), (B, 1), device=device)
            n_high = torch.randint(0, max(Q // 20, 1), (B, 1), device=device)
            keep = (idx >= n_low) & (idx < (Q - n_high))
            mask = keep.to(R.dtype)

        # Guarantee finite, non-negative output
        R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return R, mask
