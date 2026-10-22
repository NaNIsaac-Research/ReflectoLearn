# Experiment 001: how much of a single film does one XRR curve determine?

The question behind this experiment: when a fit of a measured curve returns the wrong film, is that
because the optimizer stopped in the wrong place, or because the curve does not contain the answer
(several structures fit it within the noise)? The two need different fixes: a better fitting path in
the first case, a posterior that keeps several solutions in the second.

This first step is local. It computes, at the true structure, the Fisher information that Poisson
counts carry about the film and the nuisances refine fits with it, and reports which parameter
combinations are weakly determined. A local analysis can show that a direction is weak; it cannot rule
out a distant second solution. That is the next step (R1c).

## Model

- Stack: ambient / film / Si (SLD 20.07 × 10⁻⁶ Å⁻², roughness 3 Å, known). Scenario `oxide` adds a
  10 Å top layer (SLD 18.9, roughness 3 Å) between ambient and film, as in `data.generator` and
  `refine.make_forward`. In that stack the film's roughness is the buried oxide/film interface; the
  top surface roughness is the oxide's.
- Counts: `s · (i0 · R_res(q_eff) + 10^bkg)`, Poisson, as in `data.augmentations` (i0, background,
  Poisson steps). `q_eff` applies a 2θ zero offset as `physics.apply_tth_offset`. `R_res` is R
  convolved with a Gaussian of constant width `dq` in q, refine's resolution model.
- Truth for the nuisances: i0 = 1, offset = 0, dq = 0.002 Å⁻¹, background 10⁻⁷ unless stated.
- Not included: footprint (taken as corrected), Cu Kα2 (taken as monochromated), masked angles,
  any model error. Every result below assumes the fitted model is the true one.
- Grid: 2θ from 0.2° in 0.01° steps up to 3°, 6° or 12° (q_max 0.21, 0.43, 0.85 Å⁻¹), λ = 1.5406 Å.
- Films: `mid` 240 Å / 6 Å / 45 (the README and benchmark `fixed_mid` film), `thin` 90 Å / 3 Å / 28
  (benchmark `fixed_thin`), `low_contrast` 240 Å / 6 Å / 24 (film SLD close to Si).

Free parameters:

| scenario | free |
|---|---|
| `film` | d, σ, ρ |
| `instr` | d, σ, ρ, i0, bkg, offset, dq |
| `oxide` | `instr` + oxide thickness, SLD, roughness |

## Method

The Jacobian `J = ∂ ln counts / ∂θ` is taken by forward-mode autograd (`torch.func.jacfwd`) through
`physics.abeles_core` in complex128; `AbelesMatrix` itself runs in complex64, which is too coarse for
small singular values. Fisher information of Poisson counts is `F = Jᵀ diag(counts) J`. From it:

- CRLB: `sqrt(diag F⁻¹)`, the smallest standard deviation an unbiased estimator can reach under this
  model. Reported in % of the true value.
- Correlations of `F⁻¹`.
- `s_min`: the smallest singular value of the whitened Jacobian `diag(√counts) J` after its columns are
  scaled to unit norm. This makes it independent of the units of each parameter. Its right singular
  vector is the weakest direction in those normalized coordinates.
- Collinearity index `γ = 1/s_min` (Brun, Reichert and Künsch 2001). γ = 20 means some combination of
  parameters changes the curve 20 times less than the parameters do one at a time.

## Reproduce

```bash
uv run python experiments/001_identifiability/fim.py        # ~10 s on CPU
uv run pytest tests/test_identifiability.py
```

Writes `fim_results.csv` (all 243 cases: 3 films × 3 scenarios × 3 q ranges × 3 backgrounds × 3
count levels), `fim_tables.md` (the tables per film and scenario) and `fim.png` to this directory.

## Checks

- Forward model against `benchmark.parratt`, a float64 numpy Parratt recursion that shares no code
  with the torch engine: max relative difference in R of 5 × 10⁻¹⁵ over 2θ 0.2–12°. The engine's
  fixed `+1e-9j` Å⁻² absorption in every medium (ambient included), which the numpy code cannot
  represent, changes R by up to 1.2 × 10⁻³ near the film's critical edge. It is kept, so the model
  here is the one `AbelesMatrix` computes.
- Autograd Jacobian against central differences of a numpy re-implementation of the whole counts
  model (Parratt, offset, resolution, i0, background), all ten parameters of the `oxide` scenario:
  relative column error ≤ 6 × 10⁻⁸. Ten times larger steps give 100 times larger errors, so what
  remains is the finite-difference truncation error.
- Counts model against `refine.make_forward` (float32, the model refine and `refine.mcmc` fit with)
  with offset 0.01° and dq 0.002: max |Δ log₁₀ R| of 1 × 10⁻⁵. A resolution quadrature cut at ±3σ (as
  `benchmark.smeared` does) differed by up to 2 % at deep fringe minima, so the quadrature here spans
  ±5σ.
- `tests/test_identifiability.py` (4 tests, about 3 s) asserts the first two checks, that the CRLB
  scales exactly as `s^-1/2` with unchanged correlations, and that a parameter with no effect on the
  curve leaves F singular.

## Results

At 10⁶ counts at R = 1 (the middle of `augmentation.poisson_log_scale`), background 10⁻⁷, q_max 0.43
Å⁻¹. CRLB in % of the true value.

| film | free | CRLB d | CRLB σ | CRLB ρ | γ | weakest direction | largest correlation |
|---|---|---|---|---|---|---|---|
| mid | film | 0.009 | 0.28 | 0.008 | 1.4 | ρ + σ + d | σ–ρ 0.34 |
| mid | instr | 0.018 | 0.40 | 0.083 | 18 | offset + ρ | ρ–offset 0.995 |
| mid | oxide | 0.51 | 10 | 0.12 | 84 | d − ρ_ox − d_ox | d–ρ_ox −0.998 |
| thin | film | 0.042 | 0.97 | 0.016 | 1.6 | ρ − d + σ | d–ρ −0.52 |
| thin | instr | 0.043 | 1.2 | 0.028 | 2.9 | ρ + offset | ρ–offset 0.72 |
| thin | oxide | 2.6 | 51 | 0.031 | 103 | d − ρ_ox − d_ox | d–ρ_ox −0.998 |
| low_contrast | film | 0.036 | 0.40 | 0.010 | 1.2 | ρ + σ | σ–ρ 0.30 |
| low_contrast | instr | 0.041 | 0.43 | 0.061 | 9.5 | offset + ρ | ρ–offset 0.98 |
| low_contrast | oxide | 4.0 | 63 | 0.066 | 196 | d_ox − d + ρ_ox | d–d_ox −1.000 |

![CRLB vs measured range, correlations, collinearity index](fim.png)

(a) CRLB of the mid film against the upper end of the measured range, with only the film free (solid)
and with the instrument nuisances free (dashed). (b) Correlations of F⁻¹ for the mid film, `instr`.
(c) γ against the measured range for each film, `instr` (solid) and `oxide` (dotted).

Measured range and noise (full tables in [`fim_tables.md`](fim_tables.md)):

- Range: for the two 240 Å films (σ 6 Å) the `film` and `instr` bounds are settled by q ≈ 0.25 Å⁻¹;
  going from 0.43 to 0.85 Å⁻¹ changes them by less than 1 %, because the counts are under the
  background by then. The thin film (σ 3 Å, slower fall-off) still gains on σ: −3 % (`film`) and
  −18 % (`instr`). With the oxide free, the extra range helps every film: d and σ bounds drop by
  24–38 % from 0.43 to 0.85 Å⁻¹, and for the mid film the d bound goes 5.0 % → 0.51 % → 0.37 % and γ
  783 → 84 → 62 for q_max 0.21 / 0.43 / 0.85.
- Counting statistics: in this model F is proportional to s, so every CRLB scales as `s^-1/2` and
  the correlations, γ and the weakest direction do not change with counting time. Counting longer
  shrinks the uncertainty along the weak direction but does not remove it.
- Background (10⁻⁸ to 10⁻⁶): with only the film or the instrument free it moves the d and ρ bounds
  by ≤ 1.2 % and the σ bound by up to 10 % (thin film). With the oxide free it matters: the d bound
  grows by up to 52 % (mid) and 92 % (thin, low contrast), the σ bound by 42–85 %.

## What this shows

Facts, under the stated model (no model error, local, Poisson):

1. With every nuisance known, d, σ and ρ of a single film are determined far better than the
   errors reported for refine: CRLB 0.01–0.04 % for d and ρ, 0.3–1 % for σ, γ < 2.
2. Freeing the 2θ offset creates one near-degenerate direction, offset together with ρ: correlation
   0.995 for the mid film, 0.98 for the low-contrast film, 0.72 for the thin film; γ 18, 9.5, 2.9.
   This is the offset–SLD valley of ReflectoLearn#14; FORUM T2 already computed it with a numpy
   Parratt copy for a 250 Å / 6 Å / 45 film (corr +0.997 at 10⁷ counts, six parameters), and this
   reproduces it through the torch engine. Even so the bounds stay small (ρ 0.08 % for the mid film
   at 10⁶ counts), and T2 showed that refine's 30 % SLD drift along this valley ends at a worse fit
   than the offset-free one: an optimization failure, not a second solution.
3. Freeing a top oxide layer (refine's oxide nuisance) is what makes the film practically
   non-identifiable: the film thickness trades against the oxide SLD and thickness (correlation
   −0.998 to −1.000), γ 60–1300. At q_max 0.21 Å⁻¹ the CRLB of d is 5 % (mid), 34 % (thin), 26 %
   (low contrast), and σ (here the buried oxide/film roughness) is not determined at all (82–590 %).
   ρ stays well determined (≤ 0.17 %).

Inference (not yet tested):

- The SLD errors reported for refine (a few % up to 30 % in FORUM T2 and DECISIONS #1) are about
  two orders of magnitude larger than the `instr` bounds, so with the offset as the only nuisance they come from the fitting path or from model
  error, not from missing information. This is consistent with T2's finding that the drift arises in
  stage 2.
- A model error does not need to be large to move the fit along a weak direction. In a linearized
  fit, the bias is `F⁻¹ Jᵀ W δ` for a model error δ; in normalized coordinates the component along
  the weakest direction is the projection of the whitened δ divided by s_min, so a model error that
  happens to look like that direction is amplified by about γ (≈ 100 with the oxide free) relative
  to a well-determined direction. Whether the oxide nuisance absorbs model errors this way is the
  most useful next test (below).
- Where the oxide is free and q_max ≤ 0.2 Å⁻¹, a single best fit is the wrong output: a posterior
  (or a set of solutions) is needed for d and σ. Whether that posterior is multimodal or only
  elongated is not answered by F.

## Limits

- Local: F describes the curvature at the truth. Distant solutions with the same curve (for example
  across a fringe order, or a swapped layer order) are invisible to it. R1c tests this with
  multi-start fits and MCMC.
- No model error. Real curves carry footprint, Kα2, a q-dependent resolution, non-Gaussian
  roughness, a substrate that is not quite Si; each moves the estimate along the weak directions.
- CRLB is for an efficient (Poisson maximum-likelihood) estimator. refine minimizes a log-MSE with
  uniform weights over q and `refine.mcmc` uses a Gaussian likelihood in log₁₀ R; both are less
  efficient than this bound, by an amount not computed here.
- Three films only, one point each in (d, σ, ρ). γ depends on where in the parameter space the truth
  is; a scan over the training ranges (`config.yaml`) would show where the problem is hard.

## Next minimal experiments

1. Linearized misspecification bias (minutes of CPU): `δθ = F⁻¹ Jᵀ W δ` for model errors we know
   are present (constant vs q-proportional resolution, as in RL#20; Kα2 ratio; footprint; oxide
   under the film as in `benchmark.make_synthetic` vs on top as in refine), per scenario. Shows
   which nuisance turns which error into a film-parameter bias.
2. R1c: for the `oxide` cases with γ > 100, draw curves at the truth, run multi-start refine and
   `refine.mcmc`, and compare the spread of the fits with the CRLB and the posterior. Agreement means
   the spread is information-limited; a wider spread means optimization failure; separate clusters
   mean non-uniqueness.

## R1c pre-registration: non-uniqueness vs optimization failure (written before running)

Registered 2026-10-22 23:59, before any of the computations below were run. Results go in a separate
section; if a rule turns out unworkable it is changed in a new commit that says why, not edited here.

**Setup.** Lab-like scan: q uniform from q(2θ = 0.2°) to q(2θ = 6°), 291 points, λ 1.5406 Å.
Forward model as above plus the two effects refine also fits: Cu Kα2 (ratio 0.5, λ₂ 1.54439 Å, mixed
as in `make_forward`) and the footprint factor `erf(s_fp · sin θ)` with s_fp = 500. Truth: i0 1,
background 10⁻⁷, offset 0, dq 0.002 Å⁻¹, top oxide 10 Å / 18.9 / 3 Å. Free parameters: the twelve
refine fits (d, σ, ρ, i0, bkg, offset, dq, Kα2 ratio, footprint, oxide thickness, SLD, roughness),
within `RefineConfig`'s bounds for the nuisances. Counting level s = 10⁶ unless stated.

**Profile likelihood in d.** For noise-free data (the expected counts at the truth, "asymptotic data"
as in Raue et al. 2009), PL(d) is the Poisson deviance minimized over the other eleven parameters at
fixed d, by bounded least squares on deviance residuals, continued outward from the truth. Since
PL(d₀) = 0, Δχ²(d) = PL(d). The deviance is proportional to s, so one profile gives every counting
level.

**Labels for a case** (from the profile at s = 10⁶):
- *identifiable*: the Δχ² ≤ 1 interval of d is narrower than ±2 % of d.
- *ridge* (practical non-identifiability): the Δχ² ≤ 1 interval is connected and spans ≥ ±2 % of d.
- *multimodal* (non-uniqueness proper): PL has a second local minimum with Δχ² ≤ 1, separated from
  the truth by a barrier with Δχ² > 1.
- A "pair giving the same curve within the noise" is the truth and the profile point at Δχ² = 1
  farthest from it; reported with both curves and their normalized residual.

**Fits on one noisy curve** (Poisson draw, seed 0, s = 10⁶; zero-count points masked):
- Multi-start: `refine.refine` with default `RefineConfig` from 15 starts, d₀ × {0.8, 0.9, 1.0, 1.1,
  1.2} × ρ₀ × {0.85, 1.0, 1.15}, σ start 0.8 σ₀; plus one fit started at the truth.
- For each endpoint, D(endpoint) is the Poisson deviance against the noisy counts and D* the lowest
  D over all endpoints. An endpoint is *in the valley* if D − D* ≤ 4, an *optimization failure* if
  D − D* > 25, undecided in between.
- Verdict for the case: if the endpoints' spread in d is mostly in-valley endpoints and matches the
  profile's Δχ² ≤ 1 width (within a factor 2), the spread is information-limited (ridge or
  multimodal, as labelled above). If the spread comes from failure endpoints, it is optimization
  failure.
- MCMC: `refine.mcmc.sample_posterior` from the lowest-D endpoint (its Gaussian log₁₀ R likelihood,
  as implemented). Its 68 % interval of d is compared with the profile's Δχ² ≤ 1 interval; within a
  factor 2 counts as agreement. Runtime is measured on a short chain first; anything over 2 minutes
  is queued as [HEAVY].

**Cases.**
- Main: mid film (240 Å / 6 Å / 45), `oxide` scenario (γ = 84 at q_max 0.43 in the table above).
- Counter-example of known type: the same film with the oxide fixed (only the offset as a weak
  nuisance), where T2 already showed refine drifts by optimization failure; expected *identifiable*
  with failure endpoints.
- Open case: the 200 Å / 4 Å / 12 film (SLD below Si) from FORUM T2, which refine fitted 13 % high in
  every strategy ("oxide nuisance or film/substrate contrast ambiguity, unconfirmed"). Judged by the
  same rules; no expected label.
