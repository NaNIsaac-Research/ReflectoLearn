"""On-the-fly streaming XRR data generator.

Two engines compute the same draws (config.simulation.engine): "legacy", the Abeles kernel of
`reflectolearn.physics` in complex64 followed by the augmentor's grid convolution, and
"reflectofit", `reflectofit.forward` in float64 with footprint and resolution inside the forward
model at a fixed node count. A seed gives the same parameters, gates and mask with either engine.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import reflectofit
import torch
from reflectofit import Instrument, forward
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from reflectolearn.config import Config
from reflectolearn.data.augmentations import AugmentDraw, XRRAugmentor
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


class Media(NamedTuple):
    """One batch of stacks in ReflectoFit's layout, ambient first, (B, 4) each:
    [ambient, oxide (top), film, substrate]; thickness and the roughness of the interface above
    belong to each medium. offset [deg] and ratio (Kα2/Kα1) are (B,) or None when switched off."""

    sld: Tensor
    thickness: Tensor
    roughness: Tensor
    offset: Tensor | None
    ratio: Tensor | None


def draw_media(
    config: Config,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
    sub_sld: Tensor,
    sub_rough: Tensor,
    device,
) -> Media:
    """Draw the forward-model nuisances of a batch of films and lay out its media.

    For robustness on real measurements, the following are drawn per sample as nuisance effects
    (not prediction targets), each controlled by config.augmentation and disabled when set to 0:
      * oxide      : native oxide / surface layer (between ambient and film)
      * tth_offset : q-axis zero-point (alignment) error
      * Kalpha2    : Cu K-alpha2 doublet (weighted sum)
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

    offset = None
    if aug.tth_offset[0] != 0.0 or aug.tth_offset[1] != 0.0:
        offset = _sample_uniform(*aug.tth_offset, B, device)
    ratio = _sample_uniform(0.0, aug.kalpha2_ratio, B, device) if aug.kalpha2_ratio > 0.0 else None
    return Media(
        torch.stack([zeros, ox_sld, sld, sub_sld], dim=1),
        torch.stack([zeros, ox_thk, thickness, zeros], dim=1),
        torch.stack([zeros, top_rough, mid_rough, sub_rough], dim=1),
        offset,
        ratio,
    )


def legacy_curve(config: Config, q: Tensor, media: Media) -> Tensor:
    """(B, Q) linear R with AbelesMatrix (complex64), as the generator computed it before X1c."""
    aug = config.augmentation
    wl1 = config.simulation.wavelength
    thickness_tensor = media.thickness[:, :3]  # per interface: the medium above it
    roughness_tensor = media.roughness[:, 1:]
    q_eff = apply_tth_offset(q, media.offset.unsqueeze(1), wl1) if media.offset is not None else q

    abeles = AbelesMatrix(device=q.device)
    R = abeles.forward(q_eff, thickness_tensor, roughness_tensor, media.sld, wavelength=wl1)

    # Cu Kalpha2 doublet: at a given angle, lambda2 photons map to q2 = q*(lambda1/lambda2) -> weighted sum.
    if media.ratio is not None:
        ratio = media.ratio.unsqueeze(1)
        q2 = q_eff * (wl1 / aug.kalpha2_wavelength)
        R2 = abeles.forward(q2, thickness_tensor, roughness_tensor, media.sld, wavelength=wl1)
        R = (R + ratio * R2) / (1.0 + ratio)
    return R


def reflectofit_instrument(
    config: Config, media: Media, draw: AugmentDraw | None = None
) -> Instrument:
    """The ReflectoFit instrument of a batch: offset and Kα2 from the media; footprint, resolution,
    i0 and background from the augmentor's draw (beam FWHM -> σ = FWHM / 2.3548)."""
    fields = {
        "wavelength": config.simulation.wavelength,
        "kalpha2_wavelength": config.augmentation.kalpha2_wavelength,
        "tth_offset": 0.0 if media.offset is None else media.offset.double(),
        "kalpha2_ratio": 0.0 if media.ratio is None else media.ratio.double(),
        "nodes": config.simulation.resolution_nodes,
    }
    if draw is not None:
        if draw.beam_width is not None:
            fields["sample_length"] = draw.sample_len
            fields["beam_sigma"] = draw.beam_width / 2.3548
        fields["sigma_q"] = draw.sigma or 0.0
        if draw.i0 is not None:
            fields["i0"] = draw.i0[:, 0].double()
        if draw.bkg is not None:
            fields["background"] = 10.0 ** draw.bkg[:, 0].double()
    return Instrument(**fields)


def reflectofit_curve(
    config: Config, q: Tensor, media: Media, draw: AugmentDraw | None = None
) -> Tensor:
    """(B, Q) float64 observed R with reflectofit.forward (torch backend, on q's device)."""
    packed = (media.sld.to(torch.complex128), media.thickness.double(), media.roughness.double())
    return forward(q.double(), packed, reflectofit_instrument(config, media, draw), backend="torch")


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
    """Map film parameters to reflectivity R(q) with forward-model domain randomization
    (`draw_media`) on the engine config.simulation.engine selects. All input film parameters are
    (B,) tensors; the return is (B, Q) linear R in float32."""
    media = draw_media(config, thickness, roughness, sld, sub_sld, sub_rough, device)
    if config.simulation.engine == "reflectofit":
        R = reflectofit_curve(config, q, media)
    else:
        R = legacy_curve(config, q, media)
    return torch.nan_to_num(R.to(torch.float32), nan=1e-15).clamp_min(1e-15)


def engine_metadata(config: Config, n: int = 2, seed: int = 0) -> dict:
    """What a training run records about its data engine. For "reflectofit", the error of the fixed
    resolution node count against ReflectoFit's per-point rule (within about 1e-5 of a brute-force
    convolution) on n curves of the config's distribution, at the lowest and highest resolution σ.
    The curves are drawn under their own seed; the global RNG is left as it was. The per-point rule
    is slow on real-SLD edges (ReflectoFit#6), so keep n small."""
    if config.simulation.engine != "reflectofit":
        return {"engine": "legacy", "precision": "complex64"}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        pr, sub = config.param_ranges, config.substrate
        film = [
            _sample_uniform(*r, n, "cpu")
            for r in (pr.thickness, pr.roughness, pr.sld, sub.sld, sub.roughness)
        ]
        media = draw_media(config, *film, "cpu")
    q = config.q_grid.double().cpu().numpy()
    packed = (
        media.sld.to(torch.complex128).numpy(),
        media.thickness.double().numpy(),
        media.roughness.double().numpy(),
    )
    base = reflectofit_instrument(config, media)
    base = base.replace(
        tth_offset=np.asarray(base.tth_offset), kalpha2_ratio=np.asarray(base.kalpha2_ratio)
    )
    sigmas = sorted({s for s in config.augmentation.res_sigma_range if s > 0})
    errors = {"sigma_q": sigmas, "max_rel": [], "median_rel": []}
    for s in sigmas:
        fixed = forward(q, packed, base.replace(sigma_q=s))
        ref = forward(q, packed, base.replace(sigma_q=s, nodes=None))
        rel = np.abs(fixed / ref - 1)
        errors["max_rel"].append(float(rel.max()))
        errors["median_rel"].append(float(np.median(rel)))
    return {
        "engine": "reflectofit",
        "reflectofit": reflectofit.__version__,
        "precision": "float64",
        "resolution_nodes": config.simulation.resolution_nodes,
        "resolution_error": errors,
    }


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

    if config.simulation.engine == "reflectofit" and augmentor is not None:
        # Footprint and resolution belong inside the forward model, so the augmentor's effects are
        # drawn first (in the same order as before) and applied by reflectofit.forward.
        media = draw_media(config, thickness, roughness, sld, sub_sld, sub_rough, device)
        draw = augmentor.draw(batch_size, device)
        R = reflectofit_curve(config, q, media, draw).to(torch.float32).clamp_min(1e-15)
        R, mask = augmentor.noise_and_mask(R, draw)
        R = R.clamp_min(1e-15)
    else:
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
