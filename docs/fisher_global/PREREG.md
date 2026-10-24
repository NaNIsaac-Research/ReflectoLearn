# H-F, H-W2, B-2 and H-cal2: registration

xrr-policy lane, 2026-10-25. Written and committed before any quantity below is computed on the
data it names. Owner approval: `docs/isaac-directive-20261025-xrr-q16.md` in the NaNIsaac workspace
(Q16 a, b, c with five conditions). Design and motivation: `docs/xrr/GEN-PHASE1/RESEARCH-DESIGN.md`
there (sections 1.7, 2.2, 3.2). A copy of this file is `docs/xrr/GEN-PHASE1/PREREG-HF-HW2.md`.

Labels used in the result files: **[registered]** a rule from this file; **[fact]** read or computed
from frozen records; **[X2e post hoc]** a Task A number from RESEARCH-DESIGN section 1, computed after
the X2e outcome was known and used here only as motivation; **[inference]** interpretation. Every
hypothesis below is reported with its verdict, including refuted or untestable ones (condition 5).

## 1. What was seen before this registration

- X2e Task A (post hoc, RESEARCH-DESIGN section 1): near states follow the Fisher quadratic to 1 %
  (median ratio 0.998); the 93 far true-L states exceed it by a median factor of 1.5 × 10³; for the
  996 far states the deviance share at q ≤ 0.235 Å⁻¹ is ≥ 0.9 in 83 % and the largest-share region is
  b0, b1 or b2 in 922 of 996 (93 %). The thresholds of H-F and H-W2 were chosen with these numbers in
  view. The test sets below are different records with a different model (top oxide, Kα2, footprint,
  noise-free data, all points fitted, all parameters free), so they are out of sample for both.
- R2a records (ReflectoLearn `research/identifiability` at 829f330): `docs/identifiability/scan_tables.md`
  (reach, cap counts, competitor ΔD and s* per condition) and the endpoints of condition #25
  (`scan/fits_25.json`; its three best ΔD and parameters, RESEARCH-DESIGN 2.1). No Fisher matrix and
  no q-resolved deviance has been computed on any R2a endpoint.
- B-2: the truth-start fit of #25 has never been run (R2a amendment 7, not run).
- H-cal2: only the Zenodo record's description and the ORSO header of one file (column names and
  their comments, below). No count value has been read.
- Pure arithmetic: for a Poisson variable over n = 100 repeats, the dispersion index φ = s²/x̄
  (ddof 1) is distributed as χ²₉₉/99 to first order: P(0.8 ≤ φ ≤ 1.2) = 0.843, P(φ > 1.2) = 0.085,
  P(φ < 0.8) = 0.071, median 0.993.
- No pilot run.

## 2. B-1 and the quantities for H-F and H-W2 (forward only; condition 2)

Inputs: the 37 files `docs/identifiability/scan/fits_00.json` … `fits_36.json` at 829f330, 64
endpoints each (2368), read only. Truth θ_t and the free list are those of each file. Forward model
and grid are `experiments/001_identifiability/scan.py` at the same commit: `q_grid(tth_max)`,
`expected_counts` at S = 10⁶ (complex128, 15-node Gauss–Hermite resolution, top oxide, Kα1 + Kα2,
erf footprint, 2θ offset, i0, background). The data are the truth's expected counts λ_t. No fit is run
in B-1 and no R2a record or label is changed. Outputs go to `docs/fisher_global/` on
`research/fisher-global` only.

Per endpoint e (parameters θ_e over the condition's free parameters; the others equal the truth's):

- **ΔD** = Σ_k 2 λ_t,k g(r_k), r_k = λ_e,k/λ_t,k − 1, g(r) = r − ln(1 + r), evaluated as
  `r - log1p(r)` for |r| ≥ 10⁻³ and by the series r²/2 − r³/3 + r⁴/4 − r⁵/5 below. This is the
  clamped Poisson deviance of R2a with y = λ_t, written without cancellation.
- **Check 1 (stop rule).** For endpoints with recorded deviance ≥ 1:
  |ΔD − D_recorded| / D_recorded ≤ 10⁻⁶. If any fails, no H-F or H-W2 verdict is given and the
  mismatch is reported.
- **Regions** (2θ_max = 6° conditions, 291 points, the same grid as X2e): `low` = 2θ < 0.5° (16 points,
  q 0.014–0.036 Å⁻¹), then `b0` … `b9` = the remaining 275 points in 10 consecutive blocks
  (`numpy.array_split`), exactly as `task_a.py`. b0 0.037–0.075, b1 0.077–0.115, b2 0.117–0.155,
  b3 0.157–0.195, b4 0.196–0.235, b5–b9 up to 0.427 Å⁻¹. Share s_r = ΔD_r / ΔD.
- **S₁** = S / ΔD: the count level at R = 1 below which the endpoint is within ΔD = 1 of the truth.
- **Fisher at the truth.** J = ∂λ/∂θ at θ_t over the free parameters (12 single, 15 bilayer),
  `torch.func.jacfwd` through `scan.model` (J = λ ∂ln λ/∂θ). F = Jᵀ diag(1/λ_t) J, which is half the
  Hessian of the deviance at the truth for noise-free data. **ΔD_F** = Δθᵀ F Δθ, Δθ = θ_e − θ_t.
  **ρ_e** = ΔD_F / ΔD. S_F = S / ΔD_F is the Fisher count level.
- **Check 2 (numerical floor, truth only, before any endpoint ratio).** Per condition, 8 random
  directions u (numpy `default_rng(20261025)`, standard normal in coordinates scaled by the bound
  widths), each scaled so that ΔD_F = 10^k for k = −14 … −1. D_floor is the smallest 10^k such that
  |ΔD/ΔD_F − 1| ≤ 0.02 for all 8 directions at that level and at every level above it. Endpoints with
  ΔD < D_floor of their condition are counted as "at the floor" and left out of H-F (i). If no level
  passes, H-F (i) is not testable for that condition.
- Secondary, descriptive (no verdict): |cos| between Δθ and the weakest eigenvector of F in
  bound-scaled coordinates; ρ and S₁ by condition, stack, nuisance set, 2θ_max and film ρ; ρ against
  ΔD decade; the subset W′ = endpoints with 4 < ΔD ≤ 10⁴ (S₁ ≥ 100).

## 3. H-F [registered]

Sets, pooled over the 37 conditions (each endpoint counts once):
- N = endpoints with D_floor ≤ ΔD ≤ 4 (the R2a "reached" set minus the floor).
- W = endpoints with ΔD > 4.

Statements:
- (i) ρ_e ∈ [0.5, 2] for at least 90 % of N.
- (ii) ρ_e ≥ 10 (S_F at least ten times below S₁) for at least 80 % of W.

Each statement **holds** at ≥ the stated fraction, **fails** below 60 %, and is **unresolved** in
between. H-F is **supported** if (i) and (ii) hold, **refuted** if either fails, **unresolved**
otherwise. If |N| < 20, (i) is not testable and H-F is at most unresolved. The per-condition fractions
are reported beside the pooled ones but do not enter the verdict.

What each outcome would mean [inference, stated in advance]: supported — at R2a's model and counts,
the local Fisher information describes discrimination inside the truth's basin and overstates it by
at least an order of magnitude for the endpoints a local search actually returns; refuted by (ii) —
Fisher is a usable proxy for distinguishing those endpoints, which would itself be reported as the
finding; refuted by (i) — even inside Δχ² ≤ 4 the quadratic misstates R2a's deviance (curved ridges),
and the X2e near-state result does not transfer.

## 4. H-W2 [registered]

Test set: W restricted to the thirteen conditions with 2θ_max = 6° (#6, 7, 8, 15, 16, 17, 24, 25, 26,
33, 34, 35, 36). R2a fixes the layer count, so every endpoint has the true L and was fitted to all 291
points.

Statements:
- (a) s_low + s_b0 + … + s_b4 ≥ 0.90 (at least 90 % of ΔD at q ≤ 0.235 Å⁻¹) in at least 80 % of
  the test set.
- (b) The region with the largest share is b0, b1 or b2 (q 0.037–0.155 Å⁻¹) in at least 80 % of the
  test set.

Same rule as H-F: each holds at ≥ 80 %, fails below 60 %, unresolved between; H-W2 supported if both
hold, refuted if either fails.

Stated limit: `low` ends at 2θ = 0.5° (q 0.036). The critical edge of the ρ = 45 films (q_c ≈ 0.048)
falls inside b0, so for those conditions (b) is not strictly "above the edge". Results are also
broken down by film ρ (descriptive).

Meaning if supported [inference]: for telling wrong basins of this kind from the truth, counts above
q ≈ 0.24 Å⁻¹ add almost nothing, so measurement time is better spent at q ≈ 0.04–0.2 Å⁻¹. It says
nothing about roughness precision or about the location of the basins.

## 5. B-2: truth-start local profile of #25 (observation; condition 3)

- Condition #25 only (bilayer, refine nuisances, 2θ_max = 6°, film ρ 24 / 30, 15 free parameters).
- `scan.fit` from θ_t with cap 200 on the noise-free data, then `scan.half_width` (Δχ² = 1, three
  restarts per profile point, as R2a amendment 7's code) for the four film parameters d₁, ρ₁, d₂, ρ₂.
  Width label: ridge if the half-width is ≥ 2 % of the value, identifiable otherwise (amendment 7).
- Also reported: at_bound flags, lowest_dchi2 (a value < 0 means a restart found a point below the
  start; it is reported, not acted on), the Fisher prediction √((F⁻¹)_kk) of each half-width from
  B-1's F (unbounded quadratic), the number of forward calls and wall time.
- Not concluded: global uniqueness, multimodality, #25's R2a label (stays "not reached"), or any
  change to the no-go. The profile sees only the truth's basin.
- Resources (boss slot, 2026-10-25 05:5x): one process, torch 1 thread, BelowNormal priority. Stop
  and report if wall time passes 45 min or memory 2 GB. The code is written to
  `experiments/005_fisher_global/` and calls `scan.py` functions; it writes nothing under
  `docs/identifiability/`.

## 6. H-cal2: count dispersion in repeated public scans (measured data; condition 4)

Data: Zenodo 10.5281/zenodo.18548583 (Haberl et al., J. Appl. Cryst. 59, 2026,
doi:10.1107/S1600576726004449), one file only: `9.95 ms_50 Hz triangle.ort` (17 396 652 bytes,
md5 43c35b4c4c0df452647328fe33e60f36; scan 734, Au 300 Å / Cr 80 Å / Si, P08 at 18 keV, Eiger2 X 1M,
100 scans of 9.95 ms). Chosen because it has the longest per-scan exposure of the 100-scan sets (most
points with ≥ 10 counts) and a triangle wave (constant angular speed). Stored under
`$DATA/xrr-public/zenodo-18548583/`, set read-only after the md5 check. Columns from the header: Q,
R (normalised, background subtracted), sR, sQ, `raw` (photons in the ROI), `bg` (average additive
background, measured out of plane), `N` (constant normalisation, photons/s), `af` (attenuation factor
of the reflected beam), `etfp` (effective exposure time per pixel).

Pre-checks, in this order, each reported before any dispersion number (condition 4):
1. Counts: the file holds 100 data sets; `raw` is non-negative and integer-valued (|raw − round(raw)|
   ≤ 10⁻⁶). If not integer, `raw` is not a photon count and H-cal2 is not testable on this file.
2. Grid: per point index, the spread of Q across the 100 scans is ≤ 0.1 × the local Q step for at
   least 80 % of indices. Otherwise H-cal2 is not testable without rebinning, which would need a new
   registration; stop and report.
3. Normalisation and exposure: per point index, whether `N`, `af` and `etfp` are the same across scans
   (relative spread ≤ 10⁻⁶). Points where they differ are left out of the primary statistic and
   counted.
4. Drift: total `raw` per scan against scan index (linear slope with its standard error, and the
   dispersion index of the 100 totals); `N` per scan.

Statistic, per eligible point k (mean raw ≥ 10 counts, pre-checks 2 and 3 passed): φ_k = s²/x̄ of
`raw` over the 100 scans (ddof 1). `raw` includes the background; a sum of Poisson counts is Poisson,
so no background is subtracted.

**H-cal2.** Overdispersion: φ_k > 1.2 at ≥ 50 % of eligible points → **supported**.
φ_k ∈ [0.8, 1.2] at ≥ 75 % of eligible points and median φ ∈ [0.9, 1.1] → **refuted** (Poisson
expectation 84 % and 0.993). Otherwise **unresolved**. This replaces RESEARCH-DESIGN 3.2's draft rule
("within 1 ± 0.2 over ≥ 80 %"), which a Poisson variable with 100 repeats would pass only 84 % of the
time per point and so was too close to its own null.

Secondary, descriptive: fraction with φ < 0.8 (underdispersion, e.g. from detector count-rate
correction); φ against mean count in decades; φ after dividing out a per-scan multiplicative factor
estimated from the scan totals (Pearson form, μ_ik = μ̂_k a_i); whether Haberl et al. report the
dispersion (their text was not reachable; checked only if it becomes reachable).

Physical reading is kept apart from the counts [inference]: overdispersion here could come from source
fluctuations at millisecond scale, galvanometer jitter in angle, or detector corrections; this file
alone cannot separate them.

## 7. Cost and order

B-1: minutes of CPU, one process, no HEAVY slot. B-2: the boss slot above. H-cal2: one 17.4 MB
download, no fit. Order a → b → c; c may run beside a. Results: `RESULTS-B1.md`, `RESULTS-B2.md`,
`RESULTS-HCAL2.md` in `docs/xrr/GEN-PHASE1/`, generated tables and JSON in `docs/fisher_global/`
on this branch. No result file of R2a, X2e or any other experiment is overwritten.

## Amendment 1: H-cal2 pre-check 3 (after the pre-checks, before any per-point dispersion)

Written after `hcal2.py precheck` (`docs/fisher_global/hcal2_precheck.json`) and before
`hcal2.py dispersion` was run. B-1's outcome was known; B-2 was running. Nothing else changes.

- Pre-check outcomes: 100 sets of 839 points; `raw` integer, minimum 1; Q identical across scans at
  every index; `af` and `etfp` the same across scans at every index; `N` constant within each scan
  and different between scans (relative SD 5.8 %), correlated with the scan's first-10-point raw sum
  at r = 0.9998. Drift: slope of the scan totals 7.7 ± 7.2 counts per scan (0.7 % over 100 scans),
  lag-1 autocorrelation 0.05; dispersion index of the 100 totals 37.
- Literal reading of pre-check 3: `N` differs across scans at every point, so every point would be
  left out and the primary statistic would be not testable. This is reported as the literal outcome.
- Change: pre-check 3 excludes points only where `af` or `etfp` differ. Reason: by the file's header
  (R "normalized to square mean of the first 10 data points") and the 0.9998 correlation, `N` is
  computed from each scan's own counts; it is not an exposure or attenuation applied to `raw`, so it
  does not change the expected raw count at a point. `bg` also varies across scans; it is a separate
  measurement and `raw` is used without subtracting it, as registered.
- The total-count dispersion (37) means the incident intensity varies from scan to scan beyond
  Poisson. The registered secondary statistic (Pearson with a per-scan factor from the totals) is the
  one that removes it; it stays secondary and does not enter the verdict.
