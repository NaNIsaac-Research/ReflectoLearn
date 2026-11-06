# Pre-registration G1: a measured data set, without an independent truth

Written 2026-11-06, before any fit of these data. Nothing in this file has been run. The data are the
public repeated scans of an Au / Cr / Si sample ([Zenodo 10.5281/zenodo.18548583](https://doi.org/10.5281/zenodo.18548583),
Haberl et al., J. Appl. Cryst. 59, 2026, DOI 10.1107/S1600576726004449; licence CC BY 4.0). They were
used before only for the count-dispersion check F9, with no model.

## What this set can and cannot show

- The sample has a nominal structure (Au 300 Å on Cr 80 Å on Si, from the record's title) and a
  conventional reference scan (scan 888, 1 s per point, same beamline). Neither is an independent
  measurement of the structure: there is no TEM, ellipsometry or growth record in the record.
  **No claim below needs a true structure.** The nominal values are reported next to the fitted ones
  as a description, never as the truth.
- The source is a synchrotron (PETRA III P08, 18 keV) with a galvanometer-scanned fast scan, not a
  laboratory Cu Kα step scan. Any conclusion is for this instrument.
- What the set offers: real counts, 100 repeated scans of the same sample (the repeat spread is
  the real noise), and a structure simple enough to refit from many starts.

## Data and counts model

- File: `9.95 ms_50 Hz triangle.ort` (scan 734, 100 scans of 839 points, 17 396 652 bytes, md5 recorded
  in the F9 report). Downloaded from the record into a local read-only directory; about 17 MB. The
  other fast-scan files and the 677 MB `raw.zip` are not used.
- Pre-check before any fit (as in F9): the stored columns give the raw counts and the factors that
  relate counts to R. The expected-counts model is written from the file's own columns:
  λ_i = (normalisation and footprint factors of point i) × R(q_i) × s + b, with the factors read per
  point from the file. If R times the stored factors does not reproduce the raw counts within the
  stated relative error 1e-6 at every point, the relation is not usable and the set is dropped;
  nothing is then fitted.
- Forward model: ReflectoSim (Parratt, error-function roughness, Gaussian q resolution). Stack: Au
  on Cr on Si, each film with thickness, roughness and SLD free; a thin native-oxide layer on Au is
  added only as a second model (M2), not in the primary one (M1). Nuisance parameters: scale s,
  background b (log₁₀), q offset, resolution σ_q. Bounds are written in the code before the run and
  are wide (thickness 0.5 to 2 times nominal, SLD 0.5 to 1.5 times bulk, roughness 0 to 40 Å).
  Photon energy 18 keV, the incidence is from the file's header.
- Objective: Poisson deviance of the raw counts (primary); the overdispersed objective is not used
  here.

## Forward models (added before any run, after the G4 follow-up)

For measured data the resolution integral is part of the model, so **only converged-forward scores
count for any claim**; a deviance computed with the experiment engine's fixed 15-node rule is model
error here and is never reported as a result. The 64 starts per scan run on the experiment engine as
a search only. Every endpoint (not only the distinct basins) is then polished on the converged
ReflectoSim forward: trust-region least squares from the engine endpoint with the converged forward's
residuals, the engine's Jacobian as the approximate Jacobian, at most 30 forward evaluations. Grouping
into basins, ΔD, S₁\*, the repeat fits of step 4 (polished the same way) and H-G1a to H-G1c use the
converged scores only. The ReflectoSim forward takes about 1.6 s for 839 points.

## Procedure

1. **Repeats.** Five scans are drawn from the 100 with `numpy.random.default_rng(20261107)` without
   replacement. Their indices are fixed by that seed and written in the result file.
2. **Multi-start.** Per drawn scan 64 starts, drawn uniformly in the box by
   `SeedSequence([20261107, 1, scan, start])`; local refinement with the basin run's rules
   (budget N_max = 3000 evaluations, tests C1 to C4). Endpoints are grouped as one basin when their
   cross-deviances are both at most 1 at the scan's own counts, and each basin is refined to its
   minimum.
3. **Endpoints.** For each basin, D = the Poisson deviance of the scan at the basin minimum, ΔD = D
   − D_best (the best basin of that scan), and S₁\* between basins = S / ΔD_cross, where ΔD_cross is
   the noise-free deviance of one basin's curve against the other's at the scan's counts and S the
   scan's mean count level. Neither needs a true structure.
4. **Reproducibility.** The best-basin minimum of the first drawn scan is the start of a fit to each of
   all 100 repeats (one start per scan). The across-scan standard deviation of each parameter is
   compared with the Cramér-Rao bound from the Fisher information at the best basin, scaled by the
   measured dispersion φ.
5. **Second model.** Steps 2 and 3 are repeated for M2 on the first drawn scan only.

## Hypotheses

Rules follow the earlier registrations: holds at or above the upper figure, fails below the lower, between
them unresolved.

| id | statement | holds | fails |
|---|---|---|---|
| H-G1a | The best basin is separated from every other distinct basin of the same scan by ΔD ≥ 25 at the scan's counts | at least 80 % of the other basins over the five scans | below 60 % |
| H-G1b | The best basin is the same structure in all five scans (cross-deviance at most 25 between the scans' best minima, each scored on its own scan) | all five | fewer than four |
| H-G1c | The across-scan standard deviation of the Au thickness is at most 2 times the dispersion-scaled Cramér-Rao bound | yes if for Au and Cr thickness | no if more than 3 times for either |

Descriptive and not decided by a rule: the number of distinct basins per scan; S₁\* between the best
and each other basin; the fitted thicknesses next to the nominal ones; the per-q-region dispersion φ
about the best fit, which feeds the overdispersed draws of G2c (a variant that uses this φ in place of
the 1.34 of F9, in [PREREG-G2.md](PREREG-G2.md)); the same quantities for M2.

## Claims allowed and not allowed

Allowed: how many distinct fit endpoints a measured curve has under this model, how far (in deviance
and in S₁\*) the other basins are from the best fit, whether the best fit reproduces across repeats,
and the real dispersion of the counts. Not allowed: that any endpoint is "wrong" or that the best fit
is "true"; that a basin is rejected in the sense of the structure (only that the data prefer the best
basin under this model); anything about laboratory step scans.

## Cost

An engine evaluation of 839 points costs about 3 times a 291-point one. Using the recorded median of
26.5 s per R2a refinement for 291 points, 80 s per refinement: 5 × 64 starts = 320 refinements,
about 7 CPU hours (a pilot of 3 refinements measures the real time before the run). The converged
polish of all 320 endpoints at 30 evaluations of 1.6 s is about 4.3 CPU hours; 100 repeat fits at 60 s
plus polish 50 s about 3 hours; M2 on one scan about 1.5 hours. In all about 16 CPU hours (inference), about
4 hours of wall time on 4 processes.

Runs: 3 processes at most, one torch or numba thread each, BelowNormal priority, a wrapper that stops at
4 GB per process or 16 h of wall time; one result file per scan and start, written once. The code is
committed before the first run and its hash is recorded in each result file.
