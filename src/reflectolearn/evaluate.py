"""
ReflectoLearn Evaluator
=======================
합성 테스트셋에서 학습된 XRRPhysicsModel의 성능을 분석하고 리포트를 생성한다.

PINN(XRefine) 평가기에서 아이디어를 가져오되, 본 프로젝트 구조에 맞게 단순화:
  - ParamSet 없음 -> 타깃은 정규화된 (B, 3) 텐서 [thickness, roughness, sld]
  - 모델 출력은 단일 (B, 3) (tuple/fixed_values 없음)
  - 예측 파라미터로 Abeles 재시뮬레이션하여 곡선 품질(best/worst)을 평가

생성물 (모두 checkpoint 부모 디렉터리에 저장):
  - eval_metrics_full.csv       샘플별 true/pred/error
  - eval_summary.csv            파라미터별 MAE/MAPE/RMSE/R2
  - eval_1_best_worst.png       best/worst 곡선 + 파라미터 표
  - eval_2_learning_curves.png  train/val loss, lr
  - eval_3_error_vs_value.png   물리값 대비 오차(편향) 산점도
  - eval_4_parity.png           true vs pred parity (R2)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스/윈도우 무관 저장 전용 백엔드
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from tqdm import tqdm

from reflectolearn.abeles import AbelesMatrix
from reflectolearn.augmentations import XRRAugmentor
from reflectolearn.config import Config
from reflectolearn.generator import param_stats
from reflectolearn.model import XRRPhysicsModel

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass
sns.set_palette("muted")

PARAM_NAMES = ["thickness", "roughness", "sld"]
SHORT = {"thickness": "d", "roughness": "sig", "sld": "Rho"}


def _build_model(cfg: Config) -> XRRPhysicsModel:
    """cli._build_model 과 동일 (순환 import 회피용 복제)."""
    m = cfg.model
    return XRRPhysicsModel(
        q_len=cfg.simulation.q_points,
        use_fourier=m.use_fourier,
        fourier_scale=m.fourier_scale,
        n_channels=m.n_channels,
        depth=m.depth,
        mlp_hidden=m.mlp_hidden,
        dropout=m.dropout,
    )


def _rand(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


class XRREvaluator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Config,
        stats_path: str | Path | None = None,
        n_samples: int = 2000,
        batch_size: int = 200,
        device: torch.device | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.exp_dir = self.checkpoint_path.parent
        self.config = config
        self.stats_path = Path(stats_path) if stats_path else None
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model: XRRPhysicsModel | None = None
        self.simulator: AbelesMatrix | None = None
        self.augmentor: XRRAugmentor | None = None
        self.history: dict = {}
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

        # raw 저장
        self.q_grid: np.ndarray | None = None
        self.p_true: np.ndarray | None = None   # (N, 3) 물리값
        self.p_pred: np.ndarray | None = None   # (N, 3) 물리값
        self.r_meas: np.ndarray | None = None   # (N, Q) 측정 곡선 (augmented, linear)
        self.r_pred: np.ndarray | None = None   # (N, Q) 예측 파라미터 재시뮬 곡선 (linear)
        self.curve_logmse: np.ndarray | None = None

        self.results_df: pd.DataFrame | None = None
        self.summary: dict = {}

    # ------------------------------------------------------------------ setup
    def setup(self):
        logger.info(f"Evaluator init: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self.config.q_grid  # property 캐시 트리거 불필요하지만 검증
        self.q_grid = self.config.q_grid.cpu().numpy()

        self.model = _build_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.simulator = AbelesMatrix(device=self.device)
        self.augmentor = XRRAugmentor(self.config, self.config.q_grid)
        self.history = ckpt.get("history", {})

        # 정규화 통계: stats.pt 우선, 없으면 config에서 해석적으로 재계산
        if self.stats_path and self.stats_path.exists():
            st = torch.load(self.stats_path, map_location=self.device)
        else:
            st = param_stats(self.config)
        self.mean = st["mean"].to(self.device)
        self.std = st["std"].to(self.device)
        logger.info("Setup complete.")

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def _gen_eval_batch(self, bs: int):
        """generator 로직을 미러링하되 평가에 필요한 raw 도 함께 반환."""
        cfg, dev = self.config, self.device
        q = cfg.q_grid.to(dev)
        pr, sub = cfg.param_ranges, cfg.substrate

        thickness = _rand(*pr.thickness, bs, dev)
        roughness = _rand(*pr.roughness, bs, dev)
        sld = _rand(*pr.sld, bs, dev)
        sub_sld = _rand(*sub.sld, bs, dev)
        sub_rough = _rand(*sub.roughness, bs, dev)

        zeros = torch.zeros(bs, device=dev)
        sld_tensor = torch.stack([zeros, sld, sub_sld], dim=1)
        thickness_tensor = torch.stack([zeros, thickness], dim=1)
        roughness_tensor = torch.stack([roughness, sub_rough], dim=1)

        R_clean = self.simulator.forward(
            q, thickness_tensor, roughness_tensor, sld_tensor,
            wavelength=cfg.simulation.wavelength,
        ).to(torch.float32)
        R_clean = torch.nan_to_num(R_clean, nan=1e-15).clamp_min(1e-15)

        # 측정 곡선 = augmentation 적용 (mask = 유효 q 영역)
        R_meas, mask = self.augmentor.forward(q, R_clean)
        R_meas = R_meas.clamp_min(1e-15)

        # 모델 입력 전처리 (학습과 동일)
        LOG_FLOOR = -15.0
        R_max = R_meas.amax(dim=1, keepdim=True).clamp_min(1e-30)
        R_log = torch.log10((R_meas / R_max).clamp_min(1e-15)).clamp_max(0.0)
        R_log = torch.where(mask > 0, R_log, torch.full_like(R_log, LOG_FLOOR))
        x = torch.stack([R_log, mask], dim=1)

        p_phys = torch.stack([thickness, roughness, sld], dim=1)  # (B,3)
        sub_phys = torch.stack([sub_sld, sub_rough], dim=1)        # (B,2)
        return x, p_phys, sub_phys, R_meas

    @torch.no_grad()
    def run_inference(self):
        n_iters = (self.n_samples + self.batch_size - 1) // self.batch_size
        cfg, dev = self.config, self.device
        q = cfg.q_grid.to(dev)

        pt, pp, rm, rp = [], [], [], []
        logger.info(f"Inference on ~{self.n_samples} synthetic samples...")
        for _ in tqdm(range(n_iters), desc="eval"):
            x, p_phys, sub_phys, R_meas = self._gen_eval_batch(self.batch_size)

            pred_norm = self.model(x.to(dev))             # (B,3) normalized
            pred_phys = pred_norm * self.std + self.mean   # denormalize

            # 예측 파라미터 + 실제 substrate 로 재시뮬 (곡선 품질 측정)
            bs = pred_phys.shape[0]
            zeros = torch.zeros(bs, device=dev)
            sub_sld, sub_rough = sub_phys[:, 0], sub_phys[:, 1]
            sld_t = torch.stack([zeros, pred_phys[:, 2], sub_sld], dim=1)
            thk_t = torch.stack([zeros, pred_phys[:, 0]], dim=1)
            rgh_t = torch.stack([pred_phys[:, 1], sub_rough], dim=1)
            R_resim = self.simulator.forward(
                q, thk_t, rgh_t, sld_t, wavelength=cfg.simulation.wavelength
            ).to(torch.float32)
            R_resim = torch.nan_to_num(R_resim, nan=1e-15).clamp_min(1e-15)

            pt.append(p_phys.cpu())
            pp.append(pred_phys.cpu())
            rm.append(R_meas.cpu())
            rp.append(R_resim.cpu())

        self.p_true = torch.cat(pt).numpy()
        self.p_pred = torch.cat(pp).numpy()
        self.r_meas = torch.cat(rm).numpy()
        self.r_pred = torch.cat(rp).numpy()

        lt = np.log10(np.clip(self.r_meas, 1e-15, None))
        lp = np.log10(np.clip(self.r_pred, 1e-15, None))
        self.curve_logmse = np.mean((lt - lp) ** 2, axis=1)

    # --------------------------------------------------------------- metrics
    def calculate_metrics(self):
        logger.info("Computing metrics...")
        rows = []
        for i, name in enumerate(PARAM_NAMES):
            vt, vp = self.p_true[:, i], self.p_pred[:, i]
            err = vp - vt
            self.summary[name] = {
                "MAE": float(np.mean(np.abs(err))),
                "MAPE": float(np.mean(np.abs(err) / (np.abs(vt) + 1e-9)) * 100),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "R2": float(1 - np.sum(err ** 2) / (np.sum((vt - vt.mean()) ** 2) + 1e-9)),
            }
            rows.append(pd.DataFrame({
                "Sample_ID": np.arange(len(vt)),
                "Parameter": name,
                "True": vt,
                "Pred": vp,
                "Error": err,
                "AbsError": np.abs(err),
                "CurveLogMSE": self.curve_logmse,
            }))
        self.results_df = pd.concat(rows, ignore_index=True)
        self.results_df.to_csv(self.exp_dir / "eval_metrics_full.csv", index=False)
        pd.DataFrame(self.summary).T.to_csv(self.exp_dir / "eval_summary.csv")

    # ----------------------------------------------------------------- plots
    def plot_best_worst(self, n: int = 3):
        order = np.argsort(self.curve_logmse)
        idxs = np.concatenate([order[:n], order[-n:][::-1]])
        titles = [f"Best #{i+1}" for i in range(n)] + [f"Worst #{i+1}" for i in range(n)]
        q = self.q_grid

        fig = plt.figure(figsize=(16, 3.2 * len(idxs)))
        gs = gridspec.GridSpec(len(idxs), 2, width_ratios=[2, 1])
        for row, (idx, title) in enumerate(zip(idxs, titles)):
            ax = plt.subplot(gs[row, 0])
            ax.semilogy(q, np.clip(self.r_meas[idx], 1e-15, None), "k-", alpha=0.4, lw=2, label="Measured")
            ax.semilogy(q, np.clip(self.r_pred[idx], 1e-15, None), "r--", alpha=0.9, lw=1.5, label="Predicted (resim)")
            ax.set_title(f"{title}  (LogMSE: {self.curve_logmse[idx]:.4f})", fontweight="bold")
            ax.set_xlabel("q (1/Å)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            ax_t = plt.subplot(gs[row, 1]); ax_t.axis("off")
            cells = [["Param", "True", "Pred", "Err%"]]
            for i, name in enumerate(PARAM_NAMES):
                t, p = self.p_true[idx, i], self.p_pred[idx, i]
                errp = abs(p - t) / (abs(t) + 1e-9) * 100
                cells.append([SHORT[name], f"{t:.2f}", f"{p:.2f}", f"{errp:.1f}%"])
            tbl = ax_t.table(cellText=cells, loc="center", cellLoc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_text_props(fontweight="bold"); cell.set_facecolor("#eeeeee")
                elif c == 3:
                    v = float(cells[r][3].strip("%"))
                    if v > 20: cell.set_facecolor("#ffcccc")
                    elif v < 5: cell.set_facecolor("#ccffcc")
        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_1_best_worst.png", dpi=100)
        plt.close()

    def plot_learning_curves(self):
        if not self.history:
            return
        tr = self.history.get("train", [])
        va = self.history.get("val", [])
        lr = self.history.get("lr", [])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if tr or va:
            if tr: axes[0].plot(tr, label="train")
            if va: axes[0].plot(va, label="val")
            axes[0].set_yscale("log"); axes[0].set_title("Loss"); axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        if lr:
            axes[1].plot(lr, color="g"); axes[1].set_title("Learning Rate")
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_2_learning_curves.png", dpi=100)
        plt.close()

    def plot_error_vs_value(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for i, name in enumerate(PARAM_NAMES):
            sub = self.results_df[self.results_df["Parameter"] == name]
            ax = axes[i]
            ax.scatter(sub["True"], sub["AbsError"], alpha=0.2, s=8,
                       c=sub["CurveLogMSE"], cmap="viridis")
            sns.regplot(x="True", y="AbsError", data=sub, ax=ax,
                        scatter=False, color="red", line_kws={"alpha": 0.6})
            ax.set_title(f"{name}: |Err| vs True")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_3_error_vs_value.png", dpi=100)
        plt.close()

    def plot_parity(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, name in enumerate(PARAM_NAMES):
            vt, vp = self.p_true[:, i], self.p_pred[:, i]
            ax = axes[i]
            ax.scatter(vt, vp, alpha=0.15, s=8, color="#2c3e50")
            lo, hi = min(vt.min(), vp.min()), max(vt.max(), vp.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
            ax.set_title(f"{name}\nR²={self.summary[name]['R2']:.3f}")
            ax.set_xlabel("True"); ax.set_ylabel("Pred")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_4_parity.png", dpi=100)
        plt.close()

    def report(self):
        header = f"{'Parameter':<12} | {'MAE':>10} | {'MAPE(%)':>10} | {'RMSE':>10} | {'R2':>8}"
        logger.info("\n" + header + "\n" + "-" * len(header))
        for name in PARAM_NAMES:
            v = self.summary[name]
            logger.info(f"{name:<12} | {v['MAE']:>10.4f} | {v['MAPE']:>10.2f} | "
                        f"{v['RMSE']:>10.4f} | {v['R2']:>8.4f}")
        logger.info(f"curve LogMSE: mean={self.curve_logmse.mean():.4f} "
                    f"median={np.median(self.curve_logmse):.4f}")

    def evaluate_all(self):
        self.run_inference()
        self.calculate_metrics()
        self.report()
        logger.info("Generating plots...")
        self.plot_best_worst()
        self.plot_learning_curves()
        self.plot_error_vs_value()
        self.plot_parity()
        logger.info(f"Done. Reports in: {self.exp_dir}")
