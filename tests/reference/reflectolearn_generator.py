# Verbatim copy of src/reflectolearn/data/generator.py from ReflectoLearn at cc3c5c5, before the generator
# could run on ReflectoFit; kept as the reference for tests/test_generator_reflectofit.py. Do not edit.
"""On-the-fly streaming XRR data generator using the Abeles physics engine."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from reflectolearn.config import Config
from reflectolearn.data.augmentations import XRRAugmentor
from reflectolearn.physics import AbelesMatrix, apply_tth_offset

LOG_FLOOR = -15.0  # "no data" value for masked (unmeasured) angles (exp07 convention)


def param_stats(config: Config) -> dict[str, torch.Tensor]:
    """Analytic mean/std for the uniform parameter ranges.

    For U(lo, hi): mean=(lo+hi)/2, std=(hi-lo)/sqrt(12).
    Returns tensors of shape (3,) ordered [thickness, roughness, sld].
    """
    ranges = [
        config.param_ranges.thickness,
        config.param_ranges.roughness,
        config.param_ranges.sld,
    ]
    lo = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
    hi = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
    mean = (lo + hi) / 2
    std = ((hi - lo) / (12.0**0.5)).clamp_min(1e-8)
    return {"mean": mean, "std": std}


def _sample_uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


def simulate_reflectivity(
    config: Config,
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
    sub_sld: Tensor,
    sub_rough: Tensor,
    device,
) -> Tensor:
    """Map film parameters to reflectivity R(q) with forward-model domain randomization.

    For robustness on real measurements, the following are applied per-sample
    at simulation time as nuisance effects (not prediction targets):
      * tth_offset : q-axis zero-point (alignment) error
      * oxide      : native oxide / surface layer (nuisance layer between ambient and film)
      * Kalpha2    : Cu K-alpha2 doublet (weighted sum)

    Each effect is controlled by the range/probability in config.augmentation and
    is disabled when set to 0. All input film parameters are (B,) tensors; the
    return is (B, Q) linear R.
    """
    aug = config.augmentation
    B = thickness.shape[0]
    zeros = torch.zeros(B, device=device)

    # Native oxide (nuisance). Always build a 4-medium stack; when no oxide is
    # present, set thickness 0 + SLD=film so the layer is inert (identical to the
    # original single-film case).
    has_ox = (torch.rand(B, device=device) < aug.oxide_prob).float()
    ox_thk = _sample_uniform(*aug.oxide_thickness, B, device) * has_ox
    ox_sld = torch.where(has_ox > 0, _sample_uniform(*aug.oxide_sld, B, device), sld)
    ox_rough = _sample_uniform(*aug.oxide_roughness, B, device)
    # top = ambient/top-layer interface, mid = top-layer/film interface.
    top_rough = torch.where(has_ox > 0, ox_rough, roughness)  # oxide surface or film surface
    mid_rough = torch.where(
        has_ox > 0, roughness, zeros
    )  # oxide/film interface (predicted roughness)

    # Media: [ambient, oxide(top), film, substrate]  (N=4)
    sld_tensor = torch.stack([zeros, ox_sld, sld, sub_sld], dim=1)  # (B, 4)
    thickness_tensor = torch.stack([zeros, ox_thk, thickness], dim=1)  # (B, 3)
    roughness_tensor = torch.stack([top_rough, mid_rough, sub_rough], dim=1)  # (B, 3)

    # q-axis zero-point error
    wl1 = config.simulation.wavelength
    if aug.tth_offset[0] != 0.0 or aug.tth_offset[1] != 0.0:
        offset = _sample_uniform(*aug.tth_offset, B, device).unsqueeze(1)  # (B, 1)
        q_eff = apply_tth_offset(q, offset, wl1)  # (B, Q)
    else:
        q_eff = q

    abeles = AbelesMatrix(device=device)
    R = abeles.forward(q_eff, thickness_tensor, roughness_tensor, sld_tensor, wavelength=wl1)

    # Cu Kalpha2 doublet: at a given angle, lambda2 photons map to q2 = q*(lambda1/lambda2) -> weighted sum.
    if aug.kalpha2_ratio > 0.0:
        ratio = _sample_uniform(0.0, aug.kalpha2_ratio, B, device).unsqueeze(1)  # (B, 1)
        q2 = q_eff * (wl1 / aug.kalpha2_wavelength)
        R2 = abeles.forward(q2, thickness_tensor, roughness_tensor, sld_tensor, wavelength=wl1)
        R = (R + ratio * R2) / (1.0 + ratio)

    return torch.nan_to_num(R.to(torch.float32), nan=1e-15).clamp_min(1e-15)


def generate_synthetic_batch(
    config: Config,
    batch_size: int,
    augmentor: XRRAugmentor | None = None,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one batch of synthetic XRR curves and normalized targets.

    Returns:
        x: (B, 2, Q) model input [log10-normalized R, valid-q mask]
        y: (B, 3) normalized targets [thickness, roughness, sld]
    """
    device = device or config.device
    q = config.q_grid.to(device)

    pr = config.param_ranges
    sub = config.substrate

    # Film parameters (prediction targets)
    thickness = _sample_uniform(pr.thickness[0], pr.thickness[1], batch_size, device)
    roughness = _sample_uniform(pr.roughness[0], pr.roughness[1], batch_size, device)
    sld = _sample_uniform(pr.sld[0], pr.sld[1], batch_size, device)

    # Substrate parameters (simulated, not predicted)
    sub_sld = _sample_uniform(sub.sld[0], sub.sld[1], batch_size, device)
    sub_rough = _sample_uniform(sub.roughness[0], sub.roughness[1], batch_size, device)

    # Forward model (+ oxide/tth-offset/Kalpha2 domain randomization) -> linear R.
    R = simulate_reflectivity(config, q, thickness, roughness, sld, sub_sld, sub_rough, device)

    # Apply realistic instrument augmentations on linear R.
    # The augmentor returns a validity mask (0 = angle not measured).
    if augmentor is not None:
        R, mask = augmentor.forward(q, R)
        R = R.clamp_min(1e-15)
    else:
        mask = torch.ones_like(R)

    # Preprocess: per-sample max-norm -> log10 -> clamp(<=0).
    R_max = R.amax(dim=1, keepdim=True).clamp_min(1e-30)
    R_log = torch.log10((R / R_max).clamp_min(1e-15)).clamp_max(0.0)
    # Masked (unmeasured) angles get a consistent floor + mask=0, so the
    # network sees a stable "no data" signal rather than random -15 cliffs.
    R_log = torch.where(mask > 0, R_log, torch.full_like(R_log, LOG_FLOOR))
    x = torch.stack([R_log, mask], dim=1)  # (B, 2, Q)

    # Normalize targets
    stats = param_stats(config)
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    y_raw = torch.stack([thickness, roughness, sld], dim=1)  # (B, 3)
    y = (y_raw - mean) / std

    return x.float(), y.float()


class XRRStreamingDataset(IterableDataset):
    """Infinite streaming dataset that generates XRR batches on-the-fly."""

    def __init__(self, config: Config, augmentor: XRRAugmentor | None = None, steps: int = 200):
        super().__init__()
        self.config = config
        self.augmentor = augmentor
        self.steps = steps
        self.batch_size = config.training.batch_size
        # Trainer logs this attribute
        self.augment = augmentor is not None

    def __iter__(self):
        for _ in range(self.steps):
            yield generate_synthetic_batch(self.config, self.batch_size, self.augmentor)

    def __len__(self) -> int:
        return self.steps


def get_xrr_dataloader(
    config: Config,
    augmentor: XRRAugmentor | None = None,
    steps: int = 200,
) -> DataLoader:
    """DataLoader over streaming XRR data (batching happens inside the generator)."""
    dataset = XRRStreamingDataset(config, augmentor, steps)
    return DataLoader(dataset, batch_size=None, num_workers=0)
