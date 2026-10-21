"""ReflectoLearn CLI (typer): train / evaluate / config.

Run via:  uv run -m reflectolearn   (or)  reflectolearn  (installed script)
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch
import typer
from loguru import logger

from reflectolearn.config import Config
from reflectolearn.data import XRRAugmentor, get_xrr_dataloader, param_stats
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
        fourier_features=m.fourier_features,
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
        None,
        "--save-dir",
        "-s",
        help="Output dir. Defaults to config's save_root/experiment_name.",
    ),
    override: list[str] = typer.Option([], "--set", help="Override e.g. training.lr=1e-3"),
    resume: Path | None = typer.Option(None, "--resume", help="Checkpoint to resume."),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
):
    """Train the XRR model on streaming synthetic data."""
    # config.yaml precedence: --config > cwd/config.yaml > built-in defaults (matches evaluate).
    if config_path is None and Path("config.yaml").exists():
        config_path = Path("config.yaml")
        logger.info(f"using config: {config_path}")
    cfg = _load_config(config_path, override)
    torch.manual_seed(seed)

    # Use --save-dir if given on the CLI; otherwise read it from the config.
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
def tune(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Base config YAML. Defaults to cwd/config.yaml."
    ),
    n_trials: int = typer.Option(30, "--trials", "-n", help="Number of trials to search."),
    epochs: int = typer.Option(12, "--epochs", help="Proxy training epochs per trial (reduced)."),
    steps: int = typer.Option(400, "--steps", help="Steps per epoch per trial (reduced)."),
    val_steps: int = typer.Option(80, "--val-steps", help="Validation steps per trial (reduced)."),
    seed: int = typer.Option(0, "--seed", help="Sampler/data seed (for fair comparison)."),
    out: Path = typer.Option(
        "config_best.yaml", "--out", "-o", help="Path to save the best config."
    ),
    storage: str | None = typer.Option(
        None,
        "--storage",
        help="Optuna storage URL. Defaults to auto-created <out>_study.db SQLite (resumable). "
        "Use ':memory:' to run in-memory.",
    ),
    study_name: str = typer.Option("reflectolearn", "--study-name", help="Optuna study name."),
    param: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Search only these parameters (repeatable). Defaults to searching all; "
        "the rest are fixed at base-config values. e.g. -p model.fourier_features",
    ),
    override: list[str] = typer.Option(
        [], "--set", help="Override fixed base values e.g. simulation.q_points=2000"
    ),
):
    """Hyperparameter search (Optuna): find the best config via reduced training and save it as YAML."""
    from reflectolearn.tune import run_search

    if config_path is None and Path("config.yaml").exists():
        config_path = Path("config.yaml")
        logger.info(f"using base config: {config_path}")
    base_cfg = _load_config(config_path, override)
    run_search(
        base_cfg,
        n_trials=n_trials,
        epochs=epochs,
        steps=steps,
        val_steps=val_steps,
        seed=seed,
        out=out,
        storage=storage,
        study_name=study_name,
        search_params=param or None,
    )


@app.command()
def evaluate(
    checkpoint: Path | None = typer.Argument(
        None, help="Checkpoint path. Defaults to config's save_dir/best.pt."
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config YAML. Defaults to cwd/config.yaml or config.yaml next to the checkpoint.",
    ),
    stats: Path | None = typer.Option(
        None, "--stats", help="stats.pt. Defaults to stats.pt next to the checkpoint."
    ),
    n_samples: int = typer.Option(
        2000, "--n-samples", help="Number of synthetic samples for evaluation."
    ),
    batch_size: int = typer.Option(200, "--batch-size", help="Inference batch size."),
    refine: bool = typer.Option(
        False,
        "--refine/--no-refine",
        help="Physics-refine AI predictions against the measured curve and compare before/after.",
    ),
    trust_margin: float = typer.Option(
        0.4,
        "--trust-margin",
        help="Trust-region half-width ratio about the AI estimate during refine.",
    ),
):
    """Full checkpoint evaluation: metrics (MAE/MAPE/RMSE/R2) + CSV + visualization report."""
    from reflectolearn.evaluate import XRREvaluator
    from reflectolearn.refine import RefineConfig

    # When checkpoint is unset, use config's save_dir (= save_root/experiment_name)/best.pt.
    if checkpoint is None:
        # config.yaml precedence: --config > cwd/config.yaml > built-in defaults.
        if config_path is None and Path("config.yaml").exists():
            config_path = Path("config.yaml")
        checkpoint = _load_config(config_path, []).save_dir / "best.pt"
        logger.info(f"checkpoint unset -> using config default {checkpoint}")
        if not checkpoint.exists():
            raise typer.BadParameter(f"default checkpoint not found: {checkpoint}")

    exp_dir = checkpoint.parent
    # Auto-discover default config/stats next to the checkpoint.
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
        refine_enabled=refine,
        refine_config=RefineConfig(trust_margin=trust_margin) if refine else None,
    )
    evaluator.setup()
    evaluator.evaluate_all()


@app.command()
def predict(
    data: Path = typer.Argument(
        ..., help="Measured curve file (2 columns: 2theta[deg] or q, intensity)."
    ),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", "-k", help="Checkpoint. Defaults to config's save_dir/best.pt."
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config YAML. Defaults to cwd or config.yaml next to the checkpoint.",
    ),
    stats: Path | None = typer.Option(
        None, "--stats", help="stats.pt. Defaults to stats.pt next to the checkpoint."
    ),
    x_axis: str = typer.Option(
        "tth", "--x-axis", help="Input x-axis: 'tth' (2theta deg) or 'q' (Å⁻¹)."
    ),
    refine: bool = typer.Option(
        True,
        "--refine/--no-refine",
        help="Physics-refine AI predictions against the measured curve.",
    ),
    trust_margin: float = typer.Option(
        0.4, "--trust-margin", help="Refine trust-region half-width ratio."
    ),
    alias_check: bool = typer.Option(
        False,
        "--alias-check/--no-alias-check",
        help="Experimental: also refine from d/2, 2d and the FFT thickness; pick by FOM plus FFT prior.",
    ),
    mcmc_steps: int = typer.Option(
        0,
        "--mcmc-steps",
        help="emcee steps for the posterior after refine (0 = off; needs the 'mcmc' extra).",
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", "-o", help="Output folder. Defaults to next to the input file."
    ),
):
    """Infer one measured XRR curve, optionally refine the physics, and save the prediction + overlay plot."""
    from reflectolearn.predict import predict_file
    from reflectolearn.refine import RefineConfig

    if config_path is None and Path("config.yaml").exists():
        config_path = Path("config.yaml")
    # When checkpoint is unset, use config's save_dir/best.pt.
    if checkpoint is None:
        checkpoint = _load_config(config_path, []).save_dir / "best.pt"
        logger.info(f"checkpoint unset -> {checkpoint}")
    exp_dir = checkpoint.parent
    if config_path is None and (exp_dir / "config.yaml").exists():
        config_path = exp_dir / "config.yaml"
    if stats is None and (exp_dir / "stats.pt").exists():
        stats = exp_dir / "stats.pt"
    if not checkpoint.exists():
        raise typer.BadParameter(f"checkpoint not found: {checkpoint}")
    if x_axis not in ("tth", "q"):
        raise typer.BadParameter("--x-axis must be 'tth' or 'q'.")

    cfg = _load_config(config_path, [])
    predict_file(
        data,
        cfg,
        checkpoint,
        stats_path=stats,
        x_axis=x_axis,
        refine_enabled=refine,
        refine_config=RefineConfig(trust_margin=trust_margin) if refine else None,
        alias_check=alias_check,
        mcmc_steps=mcmc_steps,
        device=cfg.device,
        out_dir=out_dir,
    )


@app.command()
def benchmark(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-k", help="Checkpoint to benchmark."),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Config YAML. Defaults to config.yaml next to the checkpoint."
    ),
    stats: Path | None = typer.Option(
        None, "--stats", help="stats.pt. Defaults to next to the checkpoint."
    ),
    n_synthetic: int = typer.Option(
        12, "--n", help="Synthetic curves (independent numpy Parratt, known truth)."
    ),
    seed: int = typer.Option(2026, "--seed", help="Seed of the synthetic sets."),
    alias_check: bool = typer.Option(
        False,
        "--alias-check/--no-alias-check",
        help="Refine from d/2, d, 2d and the FFT thickness (experimental).",
    ),
    mcmc_steps: int = typer.Option(0, "--mcmc-steps", help="emcee steps per curve (0 = off)."),
    out_dir: Path = typer.Option(
        Path("benchmark_out"), "--out-dir", "-o", help="Where benchmark.json/.csv go."
    ),
):
    """Score predict on synthetic curves with known truth: random films and a few fixed example films."""
    from reflectolearn.benchmark import run_benchmark

    exp_dir = checkpoint.parent
    if config_path is None and (exp_dir / "config.yaml").exists():
        config_path = exp_dir / "config.yaml"
    if stats is None and (exp_dir / "stats.pt").exists():
        stats = exp_dir / "stats.pt"
    cfg = _load_config(config_path, [])
    run_benchmark(
        cfg,
        checkpoint,
        stats_path=stats,
        n_synthetic=n_synthetic,
        seed=seed,
        alias_check=alias_check,
        mcmc_steps=mcmc_steps,
        out_dir=out_dir,
        device=cfg.device,
    )


@app.command()
def stitch(
    scans: list[Path] = typer.Argument(
        ..., help="Scan files (x, intensity, error) in increasing 2theta order."
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output text file."),
):
    """Stitch attenuator-segmented scans into one curve (last scan is the intensity reference)."""
    import numpy as np

    from reflectolearn.stitch import read_scan, stitch_scans

    combined = stitch_scans([read_scan(p) for p in scans])
    np.savetxt(out, combined)
    logger.info(f"Wrote {len(combined)} points to {out}")


def main():
    app()


if __name__ == "__main__":
    main()
