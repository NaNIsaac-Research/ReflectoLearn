from pathlib import Path

import h5py
import numpy as np
import torch
from augmentations import XRRAugmentations
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


class XRRPreprocessor:
    def __init__(self, qs: np.ndarray, stats_file=None, device='cpu'):
        self.target_q = qs
        self.device = device
        self.param_mean = None
        self.param_std = None

        if stats_file and Path(stats_file).exists():
            self.load_stats(stats_file)

    def load_stats(self, stats_file):
        # [Fix] stats.pt 파일을 로드하여 정규화 파라미터 확보
        stats = torch.load(stats_file, map_location=self.device, weights_only=True)
        self.param_mean = stats["param_mean"]
        self.param_std = stats["param_std"]

    def process_input(self, q_raw: np.ndarray, R_raw: np.ndarray) -> torch.Tensor:
        R_raw = np.nan_to_num(R_raw, nan=1e-15, posinf=1e-15, neginf=1e-15)
        R_max = np.max(R_raw) if np.max(R_raw) > 0 else 1.0
        R_log = np.log10(np.maximum(R_raw / R_max, 1e-15))

        if q_raw[0] > q_raw[-1]:
            q_raw, R_log = q_raw[::-1], R_log[::-1]

        try:
            f_cubic = interp1d(q_raw, R_log, kind='cubic',
                               fill_value="extrapolate", bounds_error=False)
            R_interp = f_cubic(self.target_q)
            R_interp = np.minimum(R_interp, 0.0) # Physical constraint
        except ValueError:
            R_interp = np.interp(self.target_q, q_raw, R_log, left=-15.0, right=-15.0)

        q_valid_mask = (self.target_q >= np.min(q_raw)) & (self.target_q <= np.max(q_raw))
        R_interp[~q_valid_mask] = -15.0

        return torch.stack([
            torch.from_numpy(R_interp.astype(np.float32)),
            torch.from_numpy(q_valid_mask.astype(np.float32))
        ], dim=0)

    def normalize_parameters(self, params_real):
        # [Fix] 텐서 연산을 위해 mean/std 타입 체크
        mean = self.param_mean.numpy() if torch.is_tensor(self.param_mean) else self.param_mean
        std = self.param_std.numpy() if torch.is_tensor(self.param_std) else self.param_std
        params_norm = (params_real - mean) / std
        return torch.from_numpy(params_norm.astype(np.float32))

class XRR1LayerDataset(Dataset):
    def __init__(
        self, qs: np.ndarray, h5_file: str | Path, stats_file: str | Path,
        mode: str = "train", val_ratio: float = 0.2, test_ratio: float = 0.1,
        augment: bool = False, expand_factor: int = 1, aug_prob: float = 0.9,
        intensity_scale: float = 0.1, q_shift_sigma: float = 0.002,
        res_sigma_range: tuple = (0.0001, 0.006)
    ):
        self.hf = None
        self.h5_path = Path(h5_file)
        self.stats_path = Path(stats_file)
        self.mode = mode
        self.target_q = qs
        self.augment = augment and (mode == 'train')
        self.expand_factor = expand_factor if (mode == 'train') else 1
        self.aug_prob = aug_prob

        if self.augment:
            self.physics_augmenter = XRRAugmentations(
                intensity_noise_scale=intensity_scale,
                q_shift_sigma=q_shift_sigma,
                res_sigma_range=res_sigma_range,
                delta_q=(qs[1] - qs[0]),
                prob=aug_prob
            )

        self._load_metadata_only()
        self._setup_split(val_ratio, test_ratio)

        self.processor = XRRPreprocessor(self.target_q)

        # [Fix] 학습 모드일 때 통계량이 없으면 자동 계산
        if mode == 'train' and not self.stats_path.exists():
            self._calculate_and_save_stats()

        if self.stats_path.exists():
            self.processor.load_stats(self.stats_path)
        else:
            raise FileNotFoundError(f"Stats file missing for mode '{mode}': {self.stats_path}")

    def _load_metadata_only(self):
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file missing: {self.h5_path}")
        with h5py.File(self.h5_path, "r") as hf:
            self.source_q = hf["q"][:]
            self.n_total = hf["R"].shape[0]
            self.thickness = hf["thickness"][:].squeeze()
            self.roughness = hf["roughness"][:].squeeze()
            self.sld = hf["sld"][:].squeeze()

    def _calculate_and_save_stats(self):
        """[핵심 추가] 학습 데이터셋으로부터 정규화용 통계량(Mean/Std) 계산 및 저장"""
        print(f"[{self.mode}] Calculating normalization statistics from training subset...")
        idx = range(self.train_end) # 학습용 인덱스만 사용

        params = np.stack([
            self.thickness[idx],
            self.roughness[idx],
            self.sld[idx]
        ], axis=1)

        mean = np.mean(params, axis=0)
        std = np.std(params, axis=0) + 1e-8 # 0으로 나누기 방지

        torch.save({
            "param_mean": torch.from_numpy(mean.astype(np.float32)),
            "param_std": torch.from_numpy(std.astype(np.float32))
        }, self.stats_path)
        print(f"   >> Statistics saved to: {self.stats_path}")

    def _setup_split(self, val_ratio, test_ratio):
        train_ratio = 1.0 - val_ratio - test_ratio
        self.train_end = int(self.n_total * train_ratio)
        self.val_end = int(self.n_total * (train_ratio + val_ratio))

        if self.mode == "train":
            self.indices = range(0, self.train_end)
        elif self.mode == "val":
            self.indices = range(self.train_end, self.val_end)
        else:
            self.indices = range(self.val_end, self.n_total)

    def __len__(self):
        return len(self.indices) * self.expand_factor

    def __getitem__(self, idx):
        original_idx = idx % len(self.indices)
        real_idx = self.indices[original_idx]

        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r', swmr=True)

        R_raw = self.hf["R"][real_idx]
        q_raw = self.source_q[real_idx] if self.source_q.ndim == 2 else self.source_q
        params_raw = np.array([self.thickness[real_idx],
                            self.roughness[real_idx],
                            self.sld[real_idx]], dtype=np.float32)

        if self.augment:
            q_raw, R_raw = self.physics_augmenter(q_raw, R_raw)

        input_tensor = self.processor.process_input(q_raw, R_raw)
        params_tensor = self.processor.normalize_parameters(params_raw)

        return input_tensor, params_tensor

    def __del__(self):
        if hasattr(self, 'hf') and self.hf is not None:
            self.hf.close()
