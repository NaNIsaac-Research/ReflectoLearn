"""Hyperparameter search (Optuna) for the XRR inverse-problem model.

Search targets (all independent of `q_len`/physical ranges, so they are safe to
vary freely because the encoder folds the length with ``AdaptiveAvgPool1d(1)``):

  * training : lr, weight_decay, batch_size
  * model    : n_channels, depth, mlp_hidden, dropout, fourier_scale, fourier_features

The objective is the **normalized validation loss** (the MSE used by Trainer).
Target normalization is analytic and not a search target, so trials are directly
comparable.

Each trial runs a reduced "proxy" training over fewer epochs/steps and returns
the best val_loss; ``MedianPruner`` watches the val_loss reported each epoch and
prunes unpromising trials early. Data generation uses the same seed per trial to
reduce noise.

Checkpoints are written to a per-trial temporary directory and deleted on exit,
so they do not clutter the disk. Finally the best hyperparameters are applied to
the base config and saved as YAML, and the full trial history is saved as CSV —
just run ``reflectolearn train`` with the saved YAML.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import optuna
import torch
from loguru import logger

from reflectolearn.config import Config
from reflectolearn.data import XRRAugmentor, get_xrr_dataloader
from reflectolearn.model import XRRPhysicsModel
from reflectolearn.train import Trainer


def _build_model(cfg: Config) -> XRRPhysicsModel:
    """Same as cli._build_model (duplicated to avoid a circular import)."""
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


# Full search space: dotted-key -> function that draws a value from the trial.
# Split per key to support partial search (tune only some keys, fix the rest to base config values).
def _search_space(trial: optuna.Trial) -> dict[str, "callable"]:
    return {
        "training.lr": lambda: trial.suggest_float("training.lr", 1e-5, 3e-3, log=True),
        "training.weight_decay": lambda: trial.suggest_float(
            "training.weight_decay", 1e-7, 1e-2, log=True
        ),
        "training.batch_size": lambda: trial.suggest_categorical(
            "training.batch_size", [32, 64, 128, 256]
        ),
        "model.n_channels": lambda: trial.suggest_categorical(
            "model.n_channels", [32, 48, 64, 96, 128]
        ),
        "model.depth": lambda: trial.suggest_int("model.depth", 4, 7),
        "model.mlp_hidden": lambda: trial.suggest_categorical(
            "model.mlp_hidden", [256, 384, 512, 768]
        ),
        "model.dropout": lambda: trial.suggest_float("model.dropout", 0.0, 0.3),
        "model.fourier_scale": lambda: trial.suggest_float("model.fourier_scale", 5.0, 30.0),
        "model.fourier_features": lambda: trial.suggest_categorical(
            "model.fourier_features", [16, 32, 64, 128]
        ),
    }


def suggest_overrides(trial: optuna.Trial, cfg: Config, only: list[str] | None = None) -> None:
    """Apply the search space directly to cfg (in-place). Key names match dotted paths.

    only: list of dotted-keys to search. None searches everything. If given, only
          those keys are searched and the rest stay fixed at the base config values
          (e.g. ["model.fourier_features"]).
    """
    space = _search_space(trial)
    keys = list(space) if only is None else only
    unknown = [k for k in keys if k not in space]
    if unknown:
        raise ValueError(f"unknown search parameter(s): {unknown}. available: {list(space)}")
    for key in keys:
        _set_dotted(cfg, key, space[key]())


def _set_dotted(cfg: Config, key: str, value) -> None:
    """Set a value at a dotted path such as 'training.lr'."""
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def objective(
    trial: optuna.Trial,
    base_cfg: Config,
    *,
    epochs: int,
    steps: int,
    val_steps: int,
    seed: int,
    search_params: list[str] | None = None,
) -> float:
    """One trial: run reduced training and return the best validation loss; TrialPruned if unpromising."""
    cfg = base_cfg.model_copy(deep=True)
    suggest_overrides(trial, cfg, only=search_params)
    torch.manual_seed(seed)  # fix the data stream across trials -> fair comparison

    model = _build_model(cfg)
    augmentor = XRRAugmentor(cfg, cfg.q_grid)
    train_loader = get_xrr_dataloader(cfg, augmentor, steps=steps)
    val_loader = get_xrr_dataloader(cfg, augmentor, steps=val_steps)

    tmp = Path(tempfile.mkdtemp(prefix="rl_tune_"))
    try:
        # Disable built-in early-stop (patience=epochs) and let the pruner handle early stopping.
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            tmp,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            patience=epochs,
            verbose=False,
        )

        def _report(epoch: int, train_loss: float, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trainer.train(epochs, on_epoch_end=_report)
        return trainer.best_val_loss

    except torch.cuda.OutOfMemoryError:
        # Large batch x channel combinations can OOM -> prune as a failed trial.
        logger.warning(
            f"trial {trial.number}: CUDA OOM (batch={cfg.training.batch_size}, "
            f"ch={cfg.model.n_channels}) -> pruned"
        )
        raise optuna.TrialPruned() from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_search(
    base_cfg: Config,
    *,
    n_trials: int = 30,
    epochs: int = 12,
    steps: int = 400,
    val_steps: int = 80,
    seed: int = 0,
    out: str | Path = "config_best.yaml",
    storage: str | None = None,
    study_name: str = "reflectolearn",
    search_params: list[str] | None = None,
) -> optuna.Study:
    """Run the Optuna search and save the best config (YAML) + trial history (CSV).

    When storage is unset, results are written to ``<out>_study.db`` SQLite, which
    supports interrupt/resume by default (re-running with the same study_name picks
    up existing trials and continues searching). To run in-memory, pass
    ``storage=":memory:"`` explicitly.
    """
    out = Path(out)
    if storage is None:
        storage = f"sqlite:///{out.with_name(out.stem + '_study.db')}"
    elif storage == ":memory:":
        storage = None  # Optuna: None = in-memory
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,  # interrupt/resume supported when set, e.g. sqlite:///tune.db
        load_if_exists=True,
    )

    logger.info(
        f"HPO start: n_trials={n_trials}, epochs={epochs}, steps={steps}/val {val_steps}"
        f"{f', params={search_params}' if search_params else ' (search all)'}"
        f"{f', storage={storage}' if storage else ''}"
    )
    study.optimize(
        lambda t: objective(
            t,
            base_cfg,
            epochs=epochs,
            steps=steps,
            val_steps=val_steps,
            seed=seed,
            search_params=search_params,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )

    best = study.best_trial
    logger.info(f"\n best trial #{best.number} | val_loss={best.value:.6f}")
    for k, v in best.params.items():
        logger.info(f"   {k} = {v!r}")

    # Apply the best parameters to the base config and save as a directly trainable YAML.
    best_cfg = base_cfg.model_copy(deep=True)
    for k, v in best.params.items():
        _set_dotted(best_cfg, k, v)
    best_cfg.save_yaml(out)
    logger.info(f"saved best config: {out}  ->  uv run reflectolearn train --config {out}")

    # Full trial history CSV (for analysis/plotting).
    csv_path = out.with_name(out.stem + "_trials.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)
    logger.info(f"saved trial history: {csv_path}")

    return study


__all__ = ["suggest_overrides", "objective", "run_search"]
