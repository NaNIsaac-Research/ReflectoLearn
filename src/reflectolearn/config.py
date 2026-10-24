"""
ReflectoLearn Configuration (Pydantic, Streaming + 1-Layer)
===========================================================
Single-film 3-parameter (thickness/roughness/sld) prediction, for streaming
Abeles data generation.
"""

from pathlib import Path
from typing import Annotated, Literal, Self

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, field_validator

from reflectolearn.physics import tth2q

# [min, max] range type
Range = Annotated[list[float], Field(min_length=2, max_length=2)]


class ParamRanges(BaseModel):
    """Range [min, max] of the film parameters to predict."""

    thickness: Range = [10.0, 1200.0]  # [A]
    roughness: Range = [0.0, 15.0]  # [A] film surface roughness
    sld: Range = [5.0, 100.0]  # [x1e-6 A^-2]


class SubstrateConfig(BaseModel):
    """Substrate: not predicted, used only in simulation. Set min==max to fix it."""

    sld: Range = [18.0, 22.0]  # Si ~ 20
    roughness: Range = [1.0, 6.0]  # film/substrate interface


class SimulationConfig(BaseModel):
    wavelength: float = 1.54  # matches the GenX overlay validation value
    q_min_tth: float = 0.05  # [deg] 2theta
    q_max_tth: float = 15.0
    q_points: int = 2000  # [verified] encoder 6x MaxPool -> len 31
    n_samples: int = 50_000  # reference only for streaming (offline compatibility)
    # "legacy": AbelesMatrix (complex64) and the augmentor's grid convolution. "reflectofit": the
    # same draws through reflectofit.forward in float64, footprint and resolution inside the forward
    # model with a fixed node count (its error is in engine_metadata). The default stays legacy until
    # ReflectoFit's resolution rule is fast on real-SLD edges (ReflectoFit#6).
    engine: Literal["legacy", "reflectofit"] = "legacy"
    resolution_nodes: int = 61


class ModelConfig(BaseModel):
    """1:1 correspondence with XRRPhysicsModel arguments."""

    use_fourier: bool = True
    fourier_scale: float = 15.0
    fourier_features: int = 32  # Random Fourier mapping width (channels = 2 * this value)
    n_channels: int = 64
    depth: int = 6
    mlp_hidden: int = 512
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 150
    # Streaming: steps per epoch. Must be large enough for BatchNorm running
    # stats to converge (200 was too small and caused eval blow-ups). exp07 was ~31000.
    steps_per_epoch: int = 3000
    val_steps: int = 300  # number of validation steps
    patience: int = 50


class AugmentationConfig(BaseModel):
    """Batch torch augmentation parameters (read by XRRAugmentor)."""

    prob: float = 0.9  # probability of applying augmentation
    # Instrument-induced variation (not predicted, for robustness)
    i0: Range = [0.8, 1.2]  # intensity scale
    bkg: Range = [-8.0, -5.0]  # background log10
    # Noise / resolution.
    # Poisson counting scale s: counts = R*s, then R = poisson(counts)/s.
    # R is reflectivity (1e-6~1), so a small s makes the high-angle signal die to 0.
    # Sampled over 10^[lo,hi] as in exp07 (s = 1e5~1e8).
    poisson_log_scale: Range = [5.0, 8.0]
    intensity_noise_scale: float = 0.25  # (legacy, unused)
    res_sigma_range: Range = [0.0001, 0.006]  # [A^-1] resolution smearing
    # footprint
    do_footprint: bool = True
    beam_width_range: Range = [0.005, 0.015]  # [mm]
    sample_len_range: Range = [5.0, 30.0]  # [mm]

    # Forward-model randomization, read by the generator for sim-to-real robustness.
    # q-axis zero-point error: alignment offset [deg] added to measured 2theta. [0, 0] disables it.
    tth_offset: Range = [-0.02, 0.02]
    # Native oxide / surface contamination layer (nuisance: used only in simulation, not predicted).
    oxide_prob: float = 0.7  # probability that an oxide layer is present
    oxide_thickness: Range = [0.0, 40.0]  # [A]
    oxide_sld: Range = [5.0, 25.0]  # [x1e-6 A^-2] (SiO2 ~ 18-20)
    oxide_roughness: Range = [0.0, 8.0]  # [A] oxide surface roughness
    # Cu K-alpha2 doublet: per-sample ratio U(0, kalpha2_ratio). 0 disables it.
    kalpha2_ratio: float = 0.5
    kalpha2_wavelength: float = 1.54439  # [A] Cu Kα2 (Kα1 = simulation.wavelength)


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    experiment_name: str = "exp"
    save_root: Path = Path("runs")

    device: torch.device = Field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    param_ranges: ParamRanges = Field(default_factory=ParamRanges)
    substrate: SubstrateConfig = Field(default_factory=SubstrateConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    _q_grid_override: torch.Tensor | None = PrivateAttr(default=None)

    @property
    def save_dir(self) -> Path:
        """Path for training artifacts = save_root / experiment_name."""
        return self.save_root / self.experiment_name

    @field_validator("device", mode="before")
    @classmethod
    def _parse_device(cls, v):
        return torch.device(v) if isinstance(v, str) else v

    @field_serializer("device")
    def _ser_device(self, device: torch.device, _info):
        return str(device)

    @property
    def q_grid(self) -> torch.Tensor:
        if self._q_grid_override is not None:
            return self._q_grid_override
        q0 = tth2q(self.simulation.q_min_tth, self.simulation.wavelength)
        q1 = tth2q(self.simulation.q_max_tth, self.simulation.wavelength)
        return torch.linspace(q0, q1, self.simulation.q_points)

    @q_grid.setter
    def q_grid(self, value: torch.Tensor):
        self._q_grid_override = value

    def save_yaml(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


# Default instance
CONFIG = Config()
