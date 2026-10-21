# ReflectoLearn

[![Tests](https://github.com/NaNIsaac/ReflectoLearn/actions/workflows/tests.yml/badge.svg)](https://github.com/NaNIsaac/ReflectoLearn/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Thin-film thickness, roughness and SLD from one X-ray reflectivity curve: a CNN guess, then a differentiable Abeles fit.**

![A measured Ga2O3 XRR curve, the CNN prediction and the refined fit](https://raw.githubusercontent.com/NaNIsaac/ReflectoLearn/main/docs/refine.png)

A measured ALD Ga₂O₃ curve (grey), the CNN's starting guess (blue, dashed) and the
gradient-refined fit through the same Abeles engine (red), from `reflectolearn predict`.

For anyone who measures single-layer films by X-ray reflectivity (XRR) and wants a fit without hand-building a starting model. A 1D CNN predicts the film's thickness, roughness and scattering-length density (SLD) from the curve; a gradient fit through a differentiable Abeles simulator (PyTorch) then refines that guess against the data, in stages (scale/footprint, film structure, fine tuning) within trust-region bounds around the CNN guess.

The CNN is trained only on simulated curves, generated on the fly, so no dataset files are needed. Film and substrate parameters are sampled, `R(q)` is computed by a batched Abeles transfer-matrix engine (Névot-Croce roughness, complex SLD), and instrument effects (intensity scale, background, Poisson noise, resolution smearing, footprint, masked angles) are applied per batch. For robustness on measured data the simulation also randomizes nuisances that are not predicted: a q-axis zero offset, a native-oxide top layer and the Cu Kα2 doublet (set under `augmentation`).

## Install

Uses [uv](https://docs.astral.sh/uv/); Python 3.11 or newer.

```bash
git clone https://github.com/NaNIsaac/ReflectoLearn.git
cd ReflectoLearn
uv sync
```

On Windows, `torch` is taken from the PyTorch CUDA 12.8 index; other platforms use the default PyPI wheel.

## Usage

```bash
uv run reflectolearn config --out config.yaml       # write the default config
uv run reflectolearn train --config config.yaml
uv run reflectolearn train --config config.yaml --set training.lr=1e-3
uv run reflectolearn train --config config.yaml --resume path/to/last.pt
uv run reflectolearn evaluate path/to/best.pt --n-samples 2000
uv run reflectolearn evaluate path/to/best.pt --refine --trust-margin 0.4
uv run reflectolearn predict measured.txt -k path/to/best.pt       # real curve, 2 columns (2θ or q, intensity)
uv run reflectolearn tune --config config.yaml --trials 30         # Optuna search -> config_best.yaml
uv run reflectolearn stitch 237.dat 238.dat 239.dat -o combined.dat  # join attenuator segments
uv run reflectolearn benchmark -k path/to/best.pt                # synthetic curves + the two bundled In2O3 curves
```

Training writes `best.pt`, `last.pt`, `stats.pt` and a copy of `config.yaml` to `save_root/experiment_name` (set both in the config). `evaluate` simulates fresh test curves and writes `eval_summary.csv`, `eval_metrics_full.csv` and `eval_*.png` (parity, error-vs-value, best/worst fits, learning curves) next to the checkpoint. With `--refine` it also reports AI-vs-refined metrics and the log-MSE before and after fitting.

`predict` runs one measured file (`--x-axis tth` or `q`) through the same preprocessing as training, refines by default (`--no-refine` to skip; substrate fixed to the middle of the `substrate` range) and writes `<name>_predict.png` and `<name>_predict.txt` next to the input or to `--out-dir`. `tune` runs short proxy trainings per trial, saves the best config to `config_best.yaml`, the trial table to `config_best_trials.csv` and a resumable study to `config_best_study.db`; `-p section.key` restricts the search to given parameters.

`stitch` joins scans taken in consecutive 2θ ranges with different attenuators (3 columns `2θ intensity error`, no header, given in increasing 2θ order). The last scan is the intensity reference; each earlier scan is scaled so its last point matches the first point of the next one, and that shared boundary point is dropped. The same function is `reflectolearn.stitch.stitch_scans`.

`uv run python -m reflectolearn.report.figures` regenerates the figures in [`docs/`](docs) in the current directory; `refine.png` also needs a trained checkpoint and a measured curve (path in `REFLECTOLEARN_REFINE_DATA`).

## GenX fit (optional)

A classical fit with [GenX](https://github.com/aglavic/genx) (bumps differential evolution) can be run from a CNN estimate or any other starting guess. It needs the `genx` extra (`genx3`, `bumps`; installs `wxpython` as a GenX dependency):

```bash
uv sync --extra genx
```

```python
from reflectolearn.genx_fit import genx_fit

res = genx_fit(q, intensity, thickness=310.0, roughness=4.2, sld=41.0, wavelength=1.54)
res["thickness"], res["fom"], res["r_fit"]
```

The model is film + oxide layer on a fixed substrate (`sub_sld`, `sub_rough`), with I0, background and sample length fitted. Film parameters are bounded to ±10% of the guess (`film_margin`). It runs two stages (oxide + I0, then all parameters) with the logR2 figure of merit; `steps_scale` and `pop_size` trade time for quality.

## Configuration

`config.yaml` is validated by Pydantic (`config.py`). Any field can be overridden with `--set section.key=value`.

| Section        | Controls |
| -------------- | -------- |
| `param_ranges` | sampling ranges of the predicted film parameters |
| `substrate`    | substrate SLD / roughness ranges (simulated, not predicted) |
| `simulation`   | wavelength, q-range, number of q points |
| `model`        | CNN width/depth, MLP size, dropout, Fourier features |
| `training`     | batch size, learning rate, epochs, steps per epoch, patience |
| `augmentation` | instrument effects; forward-model randomization (q offset, oxide, Kα2) |

## Structure

```
src/reflectolearn/
├── cli.py          Typer CLI: config / train / tune / evaluate / predict / stitch
├── config.py       config schema and YAML I/O
├── model.py        XRRPhysicsModel (1D CNN + optional Fourier features of q)
├── train.py        Trainer (AdamW, ReduceLROnPlateau, early stopping, checkpoints)
├── evaluate.py     metrics (MAE/MAPE/RMSE/R²) and plots
├── predict.py      inference + refinement on measured curves
├── tune.py         Optuna hyperparameter search
├── stitch.py       joining attenuator-segmented scans
├── genx_fit.py     optional GenX/bumps fit (genx extra)
├── physics/        abeles.py (AbelesMatrix), utils.py (2θ → q)
├── data/           generator.py (streaming dataset), augmentations.py (XRRAugmentor)
├── refine/         pipeline.py (refine), losses.py (log/linear MSE, correlation, Sobolev)
└── report/         figures.py (slide figures)
```

## Tests

```bash
uv sync --extra genx --extra mcmc
uv run pytest
```

The two GenX tests are skipped without the `genx` extra and the emcee test without the `mcmc` extra.

## History

This repository supersedes [Reflecto](https://github.com/NaNIsaac/Reflecto) (GenX fitting, now `genx_fit.py`), [XRefine](https://github.com/NaNIsaac/XRefine) and [xrr_combine](https://github.com/NaNIsaac/xrr_combine) (scan stitching, now `stitch.py`). The earlier experiments are in [ReflectoExp](https://github.com/NaNIsaac/ReflectoExp).

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
