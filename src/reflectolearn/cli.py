"""ReflectoLearn CLI (typer): train / evaluate / config.

Run via:  uv run -m reflectolearn   (or)  reflectolearn  (installed script)
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch
import typer
from loguru import logger

from reflectolearn.augmentations import XRRAugmentor
from reflectolearn.config import Config
from reflectolearn.generator import get_xrr_dataloader, param_stats
from reflectolearn.model import XRRPhysicsModel
from reflectolearn.train import Trainer

app = typer.Typer(add_completion=False, help="ReflectoLearn: XRR inverse-problem CNN.")


def _parse_value(raw: str):
    """Best-effort literal parse; fall back to the raw string."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply dotted CLI overrides like 'training.lr=1e-3' onto the config."""
    for ov in overrides:
        if "=" not in ov:
            raise typer.BadParameter(f"override must be key=value: {ov!r}")
        key, _, raw = ov.partition("=")
        parts = key.strip().split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        value = _parse_value(raw)
        setattr(obj, parts[-1], value)
        logger.info(f"override {key} = {value!r}")
    return cfg


def _load_config(config_path: Path | None, overrides: list[str]) -> Config:
    cfg = Config.load_yaml(config_path) if config_path else Config()
    return _apply_overrides(cfg, overrides)


def _build_model(cfg: Config) -> XRRPhysicsModel:
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


@app.command()
def config(
    out: Path = typer.Option("config.yaml", "--out", "-o", help="Output YAML path."),
):
    """Write a default configuration YAML."""
    Config().save_yaml(out)
    logger.info(f"Wrote default config to {out}")


@app.command()
def train(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Config YAML."),
    save_dir: Path | None = typer.Option(
        None, "--save-dir", "-s",
        help="Output dir. 미지정 시 config의 save_root/experiment_name 사용.",
    ),
    override: list[str] = typer.Option([], "--set", help="Override e.g. training.lr=1e-3"),
    resume: Path | None = typer.Option(None, "--resume", help="Checkpoint to resume."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
):
    """Train the XRR model on streaming synthetic data."""
    cfg = _load_config(config_path, override)
    torch.manual_seed(seed)

    # CLI에서 --save-dir 주면 그걸 쓰고, 없으면 config에서 읽음.
    save_dir = Path(save_dir) if save_dir is not None else cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"save_dir = {save_dir}")
    cfg.save_yaml(save_dir / "config.yaml")  # back up the config actually used

    model = _build_model(cfg)
    augmentor = XRRAugmentor(cfg, cfg.q_grid)
    train_loader = get_xrr_dataloader(cfg, augmentor, steps=cfg.training.steps_per_epoch)
    val_loader = get_xrr_dataloader(cfg, augmentor, steps=cfg.training.val_steps)

    # Save normalization stats for later denormalization at eval/inference.
    torch.save(param_stats(cfg), save_dir / "stats.pt")

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        save_dir,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        patience=cfg.training.patience,
    )
    trainer.train(cfg.training.epochs, resume_from=resume)
    logger.info(f"Training complete. Artifacts in {save_dir}")


@app.command()
def evaluate(
    checkpoint: Path | None = typer.Argument(
        None, help="Checkpoint path. 미지정 시 config의 save_dir/best.pt 사용."),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Config YAML. 미지정 시 cwd의 config.yaml 또는 checkpoint 옆 config.yaml."),
    stats: Path | None = typer.Option(
        None, "--stats", help="stats.pt. 미지정 시 checkpoint 옆 stats.pt 사용."),
    n_samples: int = typer.Option(2000, "--n-samples", help="평가 합성 샘플 수."),
    batch_size: int = typer.Option(200, "--batch-size", help="추론 배치 크기."),
):
    """체크포인트 종합 평가: 메트릭(MAE/MAPE/RMSE/R2) + CSV + 시각화 리포트."""
    from reflectolearn.evaluate import XRREvaluator

    # checkpoint 미지정 시: config의 save_dir(=save_root/experiment_name)/best.pt 를 사용.
    if checkpoint is None:
        # config.yaml 우선순위: --config > cwd/config.yaml > 내장 기본값
        if config_path is None and Path("config.yaml").exists():
            config_path = Path("config.yaml")
        checkpoint = _load_config(config_path, []).save_dir / "best.pt"
        logger.info(f"checkpoint 미지정 → config 기준 {checkpoint}")
        if not checkpoint.exists():
            raise typer.BadParameter(f"기본 체크포인트가 없습니다: {checkpoint}")

    exp_dir = checkpoint.parent
    # config/stats 기본값을 checkpoint 옆에서 자동 탐색
    if config_path is None and (exp_dir / "config.yaml").exists():
        config_path = exp_dir / "config.yaml"
    if stats is None and (exp_dir / "stats.pt").exists():
        stats = exp_dir / "stats.pt"

    cfg = _load_config(config_path, [])
    evaluator = XRREvaluator(
        checkpoint_path=checkpoint,
        config=cfg,
        stats_path=stats,
        n_samples=n_samples,
        batch_size=batch_size,
        device=cfg.device,
    )
    evaluator.setup()
    evaluator.evaluate_all()


def main():
    app()


if __name__ == "__main__":
    main()
