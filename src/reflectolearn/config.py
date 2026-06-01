"""
ReflectoLearn Configuration (Pydantic, Streaming + 1-Layer)
===========================================================
단일 막(film) 3-파라미터(thickness/roughness/sld) 예측, Abeles 스트리밍 생성용.
"""
from pathlib import Path
from typing import Annotated, Self

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, field_validator

from reflectolearn.physics import tth2q

# [min, max] 범위 타입
Range = Annotated[list[float], Field(min_length=2, max_length=2)]


# ==============================================================================
# 서브 설정
# ==============================================================================
class ParamRanges(BaseModel):
    """예측 대상 막(film) 파라미터 범위 [min, max]"""
    thickness: Range = [10.0, 1200.0]   # [A]
    roughness: Range = [0.0, 15.0]      # [A] 막 표면 거칠기
    sld: Range = [5.0, 100.0]           # [x1e-6 A^-2]


class SubstrateConfig(BaseModel):
    """기판: 예측 대상 아님, 시뮬에만 사용. 고정하려면 min==max."""
    sld: Range = [18.0, 22.0]           # Si ~ 20
    roughness: Range = [1.0, 6.0]       # film/substrate 계면


class SimulationConfig(BaseModel):
    wavelength: float = 1.54            # GenX overlay 검증값과 일치
    q_min_tth: float = 0.05             # [deg] 2theta
    q_max_tth: float = 15.0
    q_points: int = 2000                # [검증됨] 인코더 6x MaxPool -> len 31
    n_samples: int = 50_000             # 스트리밍에선 참고용 (offline 호환)


class ModelConfig(BaseModel):
    """XRRPhysicsModel 인자와 1:1 대응"""
    use_fourier: bool = True
    fourier_scale: float = 15.0
    n_channels: int = 64
    depth: int = 6
    mlp_hidden: int = 512
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 150
    # 스트리밍: epoch당 step 수. BatchNorm running stats가 수렴하려면
    # 충분히 커야 한다(기존 200은 너무 작아 eval 폭발 유발). exp07은 ~31000.
    steps_per_epoch: int = 3000
    val_steps: int = 300                # 검증 step 수
    patience: int = 50


class AugmentationConfig(BaseModel):
    """배치 torch augmentation 파라미터 (XRRAugmentor 가 읽음)"""
    prob: float = 0.9                   # augmentation 적용 확률
    # 장비 유래 변동 (예측 대상 아님, 강건성용)
    i0: Range = [0.8, 1.2]              # 강도 스케일
    bkg: Range = [-8.0, -5.0]           # 배경 log10
    # 노이즈/해상도
    # Poisson counting scale s: counts = R*s, 다시 R = poisson(counts)/s.
    # R은 reflectivity(1e-6~1)라 s가 작으면 고각 신호가 0으로 죽는다.
    # exp07과 동일하게 10^[lo,hi] 범위에서 샘플링 (s=1e5~1e8).
    poisson_log_scale: Range = [5.0, 8.0]
    intensity_noise_scale: float = 0.25        # (legacy, 미사용)
    res_sigma_range: Range = [0.0001, 0.006]   # [A^-1] 해상도 smearing
    # footprint
    do_footprint: bool = True
    beam_width_range: Range = [0.005, 0.015]   # [mm]
    sample_len_range: Range = [5.0, 30.0]      # [mm]


# ==============================================================================
# 루트 설정
# ==============================================================================
class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    experiment_name: str = "exp"
    save_root: Path = Path(r"C:\WorkSpace\05_Resources\Data\XRR_AI\reflectolearn")

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
        """학습 산출물 저장 경로 = save_root / experiment_name."""
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
        with open(path, "r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


# 기본 인스턴스
CONFIG = Config()