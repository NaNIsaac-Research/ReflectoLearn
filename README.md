# ReflectoLearn

A 1D CNN that predicts single-layer thin-film parameters (thickness, roughness, SLD) from X-ray reflectivity (XRR) curves, trained on synthetic data from a differentiable Abeles simulator in PyTorch.

Training data is generated on the fly: film and substrate parameters are sampled, `R(q)` is computed by a batched Abeles transfer-matrix engine (Névot-Croce roughness, complex SLD, optional beam footprint), and instrument effects are applied per batch (intensity scale, background, Poisson noise, resolution smearing, footprint, masked angles). The forward model also adds domain-randomized nuisances for sim-to-real robustness, which are not predicted: a q-axis zero offset (`tth_offset`), a native-oxide top layer and the Cu Kα2 doublet; each is set under `augmentation` and off at 0. No dataset files are needed.

An optional refinement step takes the CNN prediction as the starting point and fits it to the curve by gradient descent through the same Abeles engine, in stages (scale/footprint, film structure, fine tuning) within trust-region bounds around the CNN guess.

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
```

Set `save_root` and `experiment_name` in the config first; the default `save_root` is a local Windows path. Training writes `best.pt`, `last.pt`, `stats.pt` and a copy of `config.yaml` to `save_root/experiment_name`. `evaluate` simulates fresh test curves and writes `eval_summary.csv`, `eval_metrics_full.csv` and `eval_*.png` (parity, error-vs-value, best/worst fits, learning curves) next to the checkpoint. With `--refine` it also reports AI-vs-refined metrics and the log-MSE before and after fitting.

`predict` runs one measured file (`--x-axis tth` or `q`) through the same preprocessing as training, refines by default (`--no-refine` to skip; substrate fixed to the middle of the `substrate` range) and writes `<name>_predict.png` and `<name>_predict.txt` next to the input or to `--out-dir`. `tune` runs short proxy trainings per trial, saves the best config to `config_best.yaml`, the trial table to `config_best_trials.csv` and a resumable study to `config_best_study.db`; `-p section.key` restricts the search to given parameters.

`uv run python -m reflectolearn.report.figures` regenerates the slide figures (`domain_randomization.png`, `sim_to_real.png`, `model_architecture.png`, `refine.png`, `refine_process.png`) in the current directory; `refine.png` needs a trained checkpoint and reads a measured curve from a hard-coded local path (`DEFAULT_REFINE_DATA`).

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
├── cli.py          Typer CLI: config / train / tune / evaluate / predict
├── config.py       config schema and YAML I/O
├── model.py        XRRPhysicsModel (1D CNN + optional Fourier features of q)
├── train.py        Trainer (AdamW, ReduceLROnPlateau, early stopping, checkpoints)
├── evaluate.py     metrics (MAE/MAPE/RMSE/R²) and plots
├── predict.py      inference + refinement on measured curves
├── tune.py         Optuna hyperparameter search
├── physics/        abeles.py (AbelesMatrix), utils.py (2θ → q)
├── data/           generator.py (streaming dataset), augmentations.py (XRRAugmentor)
├── refine/         pipeline.py (refine), losses.py (log/linear MSE, correlation, Sobolev)
└── report/         figures.py (slide figures)
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
