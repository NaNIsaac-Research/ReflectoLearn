# Synthetic film fit

A known film is turned into simulated counts and fitted by ReflectoLearn's own public path. This shows what the fit recovers on a curve whose answer is known. No trained network or measured data is needed.

```powershell
uv run python examples/synthetic_film_fit/run.py      # prints truth vs fit, writes result.json
uv run --group dev pytest -q tests/test_example_synthetic_film_fit.py
```

## What it does

1. **Sample.** A 250 Å film (SLD 45 + 2i, roughness 6 Å) on 15 Å of SiO₂ on Si. This is the "absorbing film on SiO2 on Si" stack of `tests/golden/xrr.json`. SLD is in 1e-6 Å⁻², with the imaginary part absorbing.
2. **Measurement.** A Cu (λ = 1.54 Å) scan from 2θ = 0.2° to 6° in 0.02° steps. The curve comes from `benchmark.smeared`, an independent float64 Parratt engine with Gaussian resolution dq/q = 2%. Background is 1e-7, the plateau holds 1e7 counts, and Poisson noise is drawn at seed 2026.
3. **Fit.** The steps are:
   - `predict.preprocess_measured` puts the counts on the model grid.
   - `refine.fft_thickness` gives the starting thickness from the fringe period.
   - `refine.refine` fits the film together with scale, background, 2θ offset, resolution, Kα₂, footprint and a native-oxide layer.

   These are the calls `reflectolearn predict` makes after its network.

## Result at seed 2026

| | truth | fit |
|---|---|---|
| thickness (Å) | 250 | 252.1 |
| roughness (Å) | 6 | 4.15 |
| SLD (1e-6 Å⁻²) | 45 | 31.8 |

The fringe-period start is 250 Å. The fit ends with a 2θ offset of −0.078° (the bound is ±0.08°) and log10-MSE 0.018.

**The SLD is about 30% low.** The data have no angular offset, but the fit trades the critical edge between the 2θ offset and the film SLD ([ReflectoLearn#14](https://github.com/NaNIsaac/ReflectoLearn/issues/14)). The thickness, which comes from the fringe period, is right.

## What the test checks

`expected.json` holds two sets of numbers:

- **`truth`**, with a tolerance for each value. Thickness must pass. SLD and roughness are strict xfails tied to #14, so a fix turns them into failures that prompt updating this page.
- **`regression`**: today's fitted values with tolerances, so any change in the fit shows up.

It also checks two invariants: the true curve never exceeds R = 1, and its edge sits within 5% above q_c = 4√(π·SLD_film·1e-6) of the film. The film is the densest layer, on top.

## Limitations

- Synthetic data from the same physics family as the fit model. Real-sample effects such as interdiffusion, density gradients, a non-Gaussian beam or detector nonlinearity are absent.
- The fit model has one film, an optional oxide on top of it, and a fixed substrate at the config midpoint (SLD 20, σ 3.5 Å). The simulated SiO₂ sits under the film and the film absorbs, and the fit can represent neither. The SLD bias above also appears without the oxide or absorption, see #14.
- Resolution: the data use a relative width dq/q, while `refine` fits a constant σ_q.
- One seed, one film. `reflectolearn benchmark` covers random films, but it needs a trained checkpoint.
