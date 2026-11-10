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

## Counts model and bounds (fixed after the pre-check, before any fit)

The pre-check passed: the stored R equals (raw − bg) · af / (etfp · N) to a relative 4.4e-7 at every point
of all 100 scans (N constant within a scan, af and etfp identical across scans, raw counts integer, minimum 1).
The fitted model is

λ_i = k_i · i₀ · R_model(q_i + δq) + 10^b · bg_i,  k_i = etfp_i · N_scan / af_i,

with R_model on the Gaussian-resolution forward (σ_q free), the wavelength fixed at 0.68939 Å (17 984.55 eV,
file header), and S = k₀ · i₀ the count level at R = 1. Free parameters of M1 (11): Au and Cr thickness,
roughness and real SLD; Si roughness; i₀; b; δq; σ_q. Au and Cr absorption scale with their fitted real SLD
from the bulk ratio; the Si SLD is fixed at 19.838 (real) and 0.0905 (imaginary) × 10⁻⁶ Å⁻². Bounds: Au
thickness 150 to 600 Å, Cr 40 to 160 Å; real SLD 0.5 to 1.5 times bulk (Au 126.44, Cr 57.17 × 10⁻⁶ Å⁻²);
roughness 0 to 40 Å; i₀ 0.2 to 5; b −1 to 1; δq ±2e-3 Å⁻¹; σ_q 1e-5 to 1e-3 Å⁻¹. M2 adds a layer on Au
(thickness 0 to 40 Å, roughness 0 to 15 Å, SLD 20 to 120 × 10⁻⁶ Å⁻², absorption 0.1 of its SLD).
The SLD values are the header's bulk figures, not a fit result. Code: `experiments/005_fisher_global/g1_run.py`.

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
   the noise-free deviance of one basin's curve against the other's at the scan's counts and
   S = k₀ · i₀ (the counts at R = 1 at the first q point, attenuator and footprint included).
   Neither needs a true structure.
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

## Changes after the run (2026-11-09)

- Step 3 defined S as the scan's mean count level, while the counts model section and the code use
  S = k₀ · i₀. The wording of step 3 now follows the code; no number changes. S₁\* from G1 is in counts
  at R = 1 at the first q point with its attenuator, so it is not on the same scale as the simulation S₁\*.
- H-G1b is computed by the code with every scan's best minimum scored on the first drawn scan's counts,
  in one direction, with the per-scan i₀, b, δq and σ_q inside the compared vectors. This differs from
  "each scored on its own scan" above, and the threshold 25 was not calibrated for a same-structure
  comparison. The recorded H-G1b numbers are therefore not a verdict; a profiled and calibrated score
  is a follow-up.

## Declared deviation 1: wider δq and σ_q bounds (2026-11-09, written before any fit under it)

Reason. In the recorded run σ_q ends on its upper bound (1e-3 Å⁻¹) in all 100 repeat fits and δq on its
lower bound (−2e-3 Å⁻¹) in 83. While the nuisances sit on their bounds, the structural parameters absorb what
the nuisances cannot, the fits split into two δq families, and none of H-G1a to H-G1c can be read. The
recorded results stay as they are and are reported next to the ones below; the rows below are labelled as
results under this deviation, not as the registered test.

Changes. Model M1W is M1 with δq ±1e-2 Å⁻¹ and σ_q 1e-5 to 3e-3 Å⁻¹; every other bound, the counts model and
the forward are unchanged. M2W is M2 with the same two bounds and the added layer's SLD from 5 (was 20) to
120 × 10⁻⁶ Å⁻², so that a low-density contamination layer is reachable. Code:
`experiments/005_fisher_global/g1_wide.py`; results in `docs/fisher_global/g1/wide/`, each file with the
commit hash of the code that wrote it.

Procedure, in the order the runs are made:

- W1, cross-seeding. Each of the five drawn scans is refined in the M1W box from the five recorded M1 search
  bests (its own and the other four scans'), with the basin run's rules, and polished as in step 2.
- W2, reproducibility. As step 4, from the lowest W1 minimum of the first drawn scan, over all 100 repeats.
- W3, second model on all repeats. For each repeat, M2W is fitted from two starts: the repeat's W2 fit with
  a contamination layer of 10 Å, roughness 3 Å and SLD 10 added, and the recorded M2 best with its i₀, b, δq and
  σ_q replaced by the repeat's W2 values. The lower minimum counts. Reported: ΔD = D(M1W) − D(M2W) per repeat
  and its distribution, the quasi-F (ΔD / 3) / (D(M2W) / (n − 14)), and the fitted layer. Descriptive.
- W4, multi-start. As step 2 in the M1W box: 64 starts per drawn scan from the same seeds. This is the
  costly part and runs last; if it is stopped, the report gives the number of starts done per scan.

Analysis under the deviation. H-G1a as registered, over the W1 and W4 endpoints. H-G1b with the profiled
score of `g1_hg1b.py` (each source structure against the target scan's best curve on the target's counts,
minimised over i₀, b, δq and σ_q inside the M1W bounds), the threshold being the 95th percentile of the same
score over 200 seeded pairs of W2 fits; the registered rule's verdict is given next to it. H-G1c as
registered on the W2 fits, and again with δq and σ_q fixed and dropped from the Fisher matrix. For every set,
the number of fits with a parameter on a bound. The holds and fails figures are the registered ones.

Cost (inference, from the recorded run: 320 M1 starts took 15 CPU hours, 100 warm repeat fits 0.3): W1 25
fits about 1 CPU hour, W2 under 1, W3 200 fits about 3, W4 320 fits about 15 to 20. Runs through the test
queue as a preemptible job, 3 processes at most, BelowNormal priority, one result file per fit, written once.

## Declared deviation 2: W4 cut to 16 starts per scan (2026-11-11, written after 21 W4 fits, all on scan 15)

Reason. The cost above was an estimate. Measured, W4 finished 21 fits in about 4 hours of run time between
preemptions, about 5 fits an hour, so 320 fits would need some 60 more hours of queue time. Some starts take
the full engine budget (about 50 minutes) before the polish, and a preemption discards a fit in progress.

Change. W4 runs the first 16 of the 64 seeded starts per scan (start indices 0 to 15, the same seeds), in
start-major order so that every scan gets its first starts before any scan gets later ones. H-G1a uses the W1
and these W4 endpoints. The 15 fits of scan 15 with start index 16 or above that finished before the cut are
kept on disk, are not part of H-G1a, and enter only the check below.

Check of the cut. Per scan the report gives the best deviance of the 16 starts next to the best of the other
fits of that scan (the five W1 cross-seeded fits, and on scan 15 the 15 later random starts), and whether the
16 reach it within 1. Where they do not, the minimum found by the other fits is the lower one and the report
says so. Code: `N_SEARCH` and `cut_check` in `experiments/005_fisher_global/g1_wide.py`.
