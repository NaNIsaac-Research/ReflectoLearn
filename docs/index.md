# ReflectoLearn

ReflectoLearn estimates the thickness, roughness and scattering-length density (SLD) of a single film
from one X-ray reflectivity (XRR) curve. A 1D CNN, trained only on simulated curves, makes the first
guess; a gradient fit through a differentiable Abeles simulator (PyTorch) then refines that guess
against the data, fitting instrument effects such as scale, background, resolution and footprint at
the same time.

It is for single-layer films measured with Cu Kα radiation (the simulation assumes 1.54 Å and
includes the Kα2 doublet), when you want a fit without building a starting model by hand. The validation so far is on synthetic curves only; measured curves have not
been validated yet (see [Validation](../README.md#validation) in the README).

## Install

Python 3.11 or later, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/NaNIsaac/ReflectoLearn.git
cd ReflectoLearn
uv sync
uv run reflectolearn --help
```

On Windows, `torch` comes from the PyTorch CUDA 12.8 index; other platforms use the default PyPI
wheel.

## First run

No trained network ships with the repository. The fit can be tried without one: the synthetic
film example simulates counts from a known film with an independent Parratt engine, then runs the
same fit `reflectolearn predict` runs after its network.

```bash
uv run python examples/synthetic_film_fit/run.py
```

```
               truth       fit
thickness     250.00    252.13  Å
roughness       6.00      4.15  Å
sld            45.00     31.75  1e-6 Å⁻²
FFT start 250 Å, log10-MSE after fit 0.0181
```

The thickness is right; the SLD is about 30% low. That is a known limit of the default fit on data
with no angular offset ([#14](https://github.com/NaNIsaac/ReflectoLearn/issues/14)), explained in
[Refinement](refine.md#known-limits) and in the [example's own page](../examples/synthetic_film_fit/README.md).

To fit a measured curve, train a network first and pass it to `predict`:

```bash
uv run reflectolearn config --out config.yaml
uv run reflectolearn train --config config.yaml
uv run reflectolearn predict measured.txt -k path/to/best.pt
```

`predict` reads two columns (2θ in degrees or q in Å⁻¹, then intensity) and writes
`<name>_predict.png` and `<name>_predict.txt`. The README's [Usage](../README.md#usage) section
covers `evaluate`, `tune`, `stitch` and `benchmark`.

## On this site

- [Refinement](refine.md): the stages, the nuisance parameters, the options of `predict` and of the
  Python API, and what is known not to work.
- [Training data and network](training.md): how the simulated training curves are made and what the
  CNN looks like.
- [XRR research index](xrr-research-index.md): the XRR work across repositories, what each branch
  measured and what is still open.
- The Python API page lists every public function and class with its docstring.

A study of which film parameters one curve can determine at all (identifiability) is in progress
in [#24](https://github.com/NaNIsaac/ReflectoLearn/pull/24) and is not on this site yet.
