# ReflectoLearn

XRR inverse modeling with a differentiable Abeles physics engine and a 1D CNN.

ReflectoLearn trains a convolutional neural network to solve the X-ray
reflectivity (XRR) inverse problem: given a measured reflectivity curve `R(q)`,
predict the **film parameters** of a single thin layer on a substrate —
thickness, surface roughness, and scattering length density (SLD).

Training data is generated **on the fly** by a fully vectorized, autograd-ready
Abeles matrix simulator, with realistic instrument augmentations (intensity
scale, background, Poisson noise, resolution smearing, beam footprint) applied
to each batch. No dataset files are needed.

## How it works

```
random film params ──► Abeles simulator ──► R(q) ──► augment (noise/footprint/...)
   (thickness,             (physics)                        │
    roughness, sld)                                         ▼
                                              log10 + max-norm + valid-q mask
                                                            │
                                                            ▼
                                              XRRPhysicsModel (1D CNN)
                                                            │
                                                            ▼
                                          predicted (thickness, roughness, sld)
```

- **Physics** — `abeles.py`: Abeles transfer-matrix method with Névot–Croce
  roughness, complex SLD, and optional beam-footprint correction. Batched and
  differentiable.
- **Data** — `generator.py`: infinite streaming `IterableDataset` that samples
  parameters uniformly and simulates curves per batch.
- **Augmentation** — `augmentations.py`: `XRRAugmentor` applies instrument
  effects and returns a validity mask for unmeasured angles.
- **Model** — `model.py`: `XRRPhysicsModel`, a 1D CNN encoder with optional
  random Fourier feature mapping of the `q` grid, followed by an MLP regressor
  that outputs 3 normalized parameters.
- **Training** — `train.py`: `Trainer` (AdamW, MSE, `ReduceLROnPlateau`, early
  stopping, checkpointing with resume).
- **Evaluation** — `evaluate.py`: metrics (MAE/MAPE/RMSE/R²) plus diagnostic
  plots (best/worst curves, learning curves, error-vs-value, parity).
- **Refinement** — `refine.py` + `losses.py`: optional physics refinement that
  takes the CNN prediction as an initial guess and fits it to the measured curve
  by gradient descent (Adam) through the differentiable Abeles engine. Uses a
  multi-stage schedule (scale/footprint → film structure → fine tuning),
  per-stage loss functions (log-MSE, linear-MSE, correlation, Sobolev, hybrid),
  and trust-region bounds anchored to the AI guess.

## Installation

This project uses [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

On Windows the PyTorch CUDA 12.8 build is used automatically (for RTX/Blackwell
support); other platforms use the default PyPI `torch` wheel.

## Usage

The CLI is exposed as `reflectolearn` (or run with `uv run -m reflectolearn`).

Write a default config:

```bash
uv run -m reflectolearn config --out config.yaml
```

Train:

```bash
uv run -m reflectolearn train --config config.yaml
# override any setting:
uv run -m reflectolearn train --config config.yaml --set training.lr=1e-3
# resume from a checkpoint:
uv run -m reflectolearn train --config config.yaml --resume path/to/last.pt
```

Training artifacts are written to `save_root / experiment_name` (from the
config): `best.pt`, `last.pt`, `stats.pt`, and a backup `config.yaml`.

Evaluate a checkpoint (config and stats are auto-discovered next to it):

```bash
uv run -m reflectolearn evaluate path/to/best.pt --n-samples 2000
```

This writes `eval_summary.csv`, `eval_metrics_full.csv`, and `eval_*.png`
reports to the checkpoint's directory.

Add `--refine` to physics-refine each prediction against its measured curve and
report AI-vs-refined metrics plus the fit FOM (LogMSE) before/after:

```bash
uv run -m reflectolearn evaluate path/to/best.pt --refine --trust-margin 0.4
```

## Configuration

All settings live in `config.yaml` (validated by Pydantic, see `config.py`).
Key sections:

| Section        | What it controls                                              |
| -------------- | ------------------------------------------------------------- |
| `param_ranges` | Film parameter sampling ranges (prediction targets).          |
| `substrate`    | Substrate SLD/roughness ranges (simulated, not predicted).    |
| `simulation`   | Wavelength, q-range, number of q-points, sample count.        |
| `model`        | CNN width/depth, MLP size, dropout, Fourier features.         |
| `training`     | Batch size, LR, weight decay, epochs, steps/epoch, patience.  |
| `augmentation` | Instrument augmentation (noise, resolution, footprint, ...).  |

Any config field can also be overridden from the command line with
`--set section.key=value`.

## Project layout

```
src/reflectolearn/
├── cli.py             # Typer CLI: train / evaluate / config
├── config.py          # Pydantic config schema + YAML I/O
├── model.py           # XRRPhysicsModel (1D CNN + Fourier features)
├── train.py           # Trainer (train loop, checkpoints, early stopping)
├── evaluate.py        # Metrics + diagnostic plots (+ optional refinement)
├── physics/           # Forward-simulation physics
│   ├── abeles.py      #   differentiable Abeles matrix engine
│   └── utils.py       #   unit helpers (e.g. 2θ → q conversion)
├── data/              # Data pipeline
│   ├── augmentations.py  # instrument augmentations
│   ├── generator.py      # streaming synthetic-data generator
│   └── dataset.py        # legacy offline HDF5 dataset (unused)
└── refine/            # Post-AI physics refinement
    ├── losses.py      #   curve-fitting losses (log/linear/correlation/...)
    └── pipeline.py    #   gradient-based multi-stage refiner
```

Each subpackage re-exports its public API, so imports stay short — e.g.
`from reflectolearn.physics import AbelesMatrix`,
`from reflectolearn.data import get_xrr_dataloader`,
`from reflectolearn.refine import refine`.

## Requirements

Python ≥ 3.11. Core dependencies: PyTorch, Typer, Pydantic, PyYAML, loguru,
NumPy, pandas, SciPy, matplotlib, seaborn, tqdm, tensorboard.
```
