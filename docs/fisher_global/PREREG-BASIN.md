# Basin minimum validation: registration

xrr-policy lane, 2026-10-25. Written and committed before any candidate below is refined. Owner
decision: `docs/isaac-directive-20261025-xrr-basin-minimum.md` in the NaNIsaac workspace (human-todo
#148, option a with seven conditions; its last sentence is cut at 800 characters and is read here as
"whether S₁ computed at a returned endpoint represents the structural discrimination of the basin it
actually converges to"). Background: `SYNTHESIS.md` 4.2 item 1 and 5. A copy of this file is
`docs/xrr/GEN-PHASE1/PREREG-BASIN.md` in the workspace.

| record | |
|---|---|
| type | Experiment Protocol (pre-registered) |
| research question | Physical identifiability against computational reach in XRR, measured in counts (S₁) |
| question of this protocol | Does S₁ at a returned endpoint represent S₁ at the local minimum of its basin? |
| hypotheses | H-BM1 (primary), Q-BM2 (registered count), H-BM3 (persistence at minima) |
| earlier records it builds on | `PREREG.md` (H-F, H-W2, B-2, H-cal2), `b1.json`, Task A `task_a.json`, R2a `docs/identifiability/scan/` at 829f330, X2e `docs/fit_policy/x2e_runs/` at a14041f |
| earlier verdicts that stay as they are | H-F unresolved, H-W2 supported, H-cal2 supported (amendment 1), X2e fail, R2a labels and the generative no-go |

Labels used in the result files: **[registered]** a rule from this file; **[fact]** read or computed
from records; **[pilot]** a number from the pilot, not used for any verdict; **[inference]**
interpretation.

## 1. What was seen before this registration

- B-1 (`b1.json`): ΔD, S₁, ρ and q shares at every R2a endpoint, so the endpoint values of every R2a
  candidate are known. Task A (`task_a.json`, post hoc): the same for X2e's states.
- `basin_candidates.json` (written by `basin.py candidates` from those two files only): the candidate
  lists and the pilot selection in section 2.
- Smoke test of the machinery on two non-candidates: a truth displaced by 10⁻³ of each bound width
  (R2a #25; X2e class 3, s = 0, both objectives). All three refinements returned to their noise-free
  or noisy minimum and passed C1–C4 in 12–29 evaluations. A batched forward-over-forward Hessian
  peaked at 1.7 GB; it was replaced by a column-wise one (same Hessian to 10⁻¹¹, 0.74 GB).
- No candidate has been refined; no refinement output exists.

## 2. Candidates (condition 1, 2)

- **R2a**: every endpoint of the 37 files `docs/identifiability/scan/fits_*.json` at 829f330 with
  4 < ΔD ≤ 10⁵ (ΔD from B-1, against the truth's noise-free counts at S = 10⁶): **749** endpoints in
  33 conditions, 403 of them stopped at the 200-evaluation cap. The thirteen 2θ_max = 6° conditions
  hold 240 of them.
- **X2e**: the **93** far states with the true layer count (Task A rows with `far` and `L == n`;
  D2 56, C 34, A1, S1, S2 one each), records at a14041f.
- Total 842. No new start is drawn: every refinement begins at a recorded endpoint.
- **Pilot** (8, chosen by rule from the endpoint ΔD only; "median" = closest to the median of log ΔD
  in the group, ties to the lowest id):

| id | why | endpoint ΔD | capped |
|---|---|---|---|
| r2a:06:44 | #6 single refine 6° ρ 12, capped, median | 8.31 × 10⁴ | yes |
| r2a:06:11 | #6 capped endpoint closest in ΔD to r2a:06:44 (exercises the duplicate rule) | 8.31 × 10⁴ | yes |
| r2a:25:25 | #25 bilayer refine 6° ρ 24, lowest ΔD (the closest known competitor, S₁ ≈ 560) | 1.78 × 10³ | no |
| r2a:33:13 | #33 bilayer wide 6° ρ 12, capped, median | 3.53 × 10⁴ | yes |
| r2a:00:58 | #0 single refine 2.2° ρ 12, capped, median | 8.17 × 10⁴ | yes |
| r2a:21:29 | #21 bilayer refine 3° ρ 12, capped, median | 3.58 × 10⁴ | yes |
| x2e:C:2:12:1:1 | X2e arm C, two layers, s = 1 (10⁶ counts), median | 2.18 × 10⁴ | — |
| x2e:D2:3:10:0:0 | X2e arm D2, three layers, s = 0 (10⁸ counts), median | 5.08 × 10⁶ | — |

## 3. Local refinement (condition 3, 4, 7)

Code: `experiments/005_fisher_global/basin.py`. R2a runs under this repository's interpreter with
`scan.py`'s model; X2e under the X2e worktree's interpreter with its `fit_policy.stack` (the code that
produced the records).

### 3.1 Objective and coordinates

- **Primary objective O_nf** for both sources: Poisson deviance of the model against the truth's
  noise-free expected counts λ_t on the full grid. For R2a this is the objective the endpoints were
  fitted to. For X2e it differs from the fitted one (noisy counts, held-out blocks 2 and 6 removed),
  and it is the objective whose value defines S₁ = S / ΔD_nf; the X2e endpoints were fitted to data
  whose noise is small next to their ΔD_nf (5 × 10¹–4 × 10⁷).
- **Secondary objective O_tr**, X2e only, descriptive: the training deviance X2e fitted (noisy
  realisation 0 of root 20261027, training points only). It answers whether the returned state was
  itself a minimum of the objective it was returned under; ΔD_nf and S₁ are also computed at its
  minimum.
- Coordinates: R2a, the condition's free parameters (12 single, 15 bilayer) inside `scan.setup`'s
  bounds, the others fixed at the endpoint's values (equal to the truth's, as in R2a); X2e, all 3L + 4
  parameters inside `Stack.bounds()`. Bound-scaled coordinates u = (θ − lo) / (hi − lo) are used for
  every distance and tolerance below.

### 3.2 Optimiser, stages and budget

- Optimiser: scipy `least_squares` TRF on the signed square-root Poisson residuals, analytic Jacobian
  from `jacfwd`, `x_scale="jac"`, `xtol = ftol = 10⁻¹⁰`, default `gtol`, the start clipped 10⁻⁹ of the
  width inside the box: the settings of `scan.fit` and `stack.fit`.
- Stages of at most 200 evaluations (`max_nfev`), each restarting TRF from the previous stage's end
  (fresh trust region). An evaluation is one forward model plus Jacobian; it is the unit of R2a's cap
  and of X2e's budget.
- **Budget N_max = 3000 evaluations per candidate and objective** (15 × R2a's cap; the X2e episode
  budget). Evaluations spent on the tests in 3.3 (one Hessian = L forward-mode passes, probe values)
  are counted separately and reported.

### 3.3 Convergence (condition 7)

τ = max(10⁻³, 10⁻⁶ D), D the current objective value. At the end of a stage where TRF stopped by its
own tolerance (status ≠ 0), at the point x reached:

- **C1 (gradient).** Exact Hessian H and gradient g of D in u, restricted to coordinates not on a
  bound (on a bound: within 10⁻⁶ in u). Eigen-directions with eigenvalue > 10⁻¹⁰ × the largest:
  Newton decrement ½ Σ (vᵀg)² / λ ≤ τ. Weaker directions: ‖projection of g‖ ≤ τ (moving across the
  whole box along them gains at most τ to first order).
- **C2 (bounds, KKT).** On a lower bound g ≥ −τ; on an upper bound g ≤ τ.
- **C3 (second order).** Along every weak direction, D at x ± 10⁻³ v (clipped into the box) is not
  below D − τ. A lower value means a descent direction: not a minimum.
- **C4 (step and deviance change).** If C1–C3 pass, one more stage is run from x (the confirmation
  stage, counted in the budget). It must move at most 10⁻⁶ in u (max norm) and lower D by at most τ.
  If it moves more, the refinement continues from where it ended.

Outcomes:
- **confirmed** — C1–C4 pass. Sub-labels: **interior**, or **on bound** (names of the parameters on a
  bound). Only confirmed candidates are basin minima.
- **unconfirmed: budget** — N_max spent first. **unconfirmed: stalled** — two stages in a row with
  TRF's own stop, no movement (≤ 10⁻⁶) and no decrease (≤ τ), and C1–C3 failing. **unconfirmed:
  numerical** — a non-finite value. Unconfirmed candidates are reported with their last values and
  never counted as minima.
- **Start check.** The deviance recomputed at the recorded endpoint against its recorded value (R2a
  `deviance`; X2e O_nf Task A `dnf`, O_tr the record's `deviance`): relative difference ≤ 10⁻⁶. A
  candidate that fails is reported and left out of every statistic.
- **Truth basin.** A confirmed minimum with ΔD_nf ≤ 4 is the truth's basin (R2a's "reached"): its
  endpoint was not a wrong basin.

### 3.4 Records (condition 4)

Per stage: evaluations, Jacobian evaluations, model calls, TRF status, D before and after, max move in
u, wall time, whether it was the confirmation stage, and the C1–C3 values when computed. Per
candidate: outcome, start check, D at endpoint and minimum, ΔD_nf at both, total evaluations, model,
value and Hessian calls, final C1–C3 values (decrement, weak gradient, KKT, probe minimum, smallest
relative eigenvalue), parameters on a bound, max drift in u from the endpoint, the parameters and
counts at the minimum, wall time and peak working set. Output: `docs/fisher_global/basin_<set>.jsonl`
(new files; nothing existing is written).

## 4. Recomputed at the minimum (condition 5) and duplicates (condition 6)

- **S₁\*** = S / ΔD\*_nf and **κ** = ΔD_nf(endpoint) / ΔD\*_nf = S₁\* / S₁(endpoint). TRF only accepts
  decreasing steps, so κ ≥ 1 by construction; a test of "no difference" would be empty, and the
  hypotheses below use magnitudes of κ instead of a p-value.
- **ρ\*** = ΔD_F / ΔD\*_nf, ΔD_F = Δθᵀ F Δθ with F the Fisher information at the truth on the full grid
  (B-1's F for R2a, Task A's for X2e) and Δθ the minimum minus the truth.
- **q shares** on the 291-point grids (R2a's 2θ_max = 6° conditions and X2e): the eleven regions of
  B-1 and Task A (`low`, b0 … b9).
- **Duplicates.** Within one data set (an R2a condition; an X2e truth (n, j, s), across arms), two
  confirmed wrong-basin minima a, b are the same basin if max(ΔD(λ_a | λ_b), ΔD(λ_b | λ_a)) ≤ 1, where
  ΔD(λ | μ) is the Poisson deviance of counts λ against noise-free data μ (each minimum's predicted
  curve as the other's data). Groups are the connected components of that relation; each basin is
  represented by its lowest ΔD\*_nf. Reported beside it: the max distance in u inside each group, and
  pairs that are curve-equivalent but more than 10⁻² apart in u (same curve, different parameters).
  Unconfirmed candidates are not grouped.

## 5. Hypotheses (full run only; the pilot gives no verdict)

Each source (R2a, X2e) is judged separately; they use different models.

### H-BM1 [registered, primary]: endpoint S₁ represents its basin

Unit: candidate endpoints with a confirmed O_nf minimum (set E). A **miss** is a truth-basin minimum
or κ > 10.
- **supported**: κ ≤ 2 and not truth basin for ≥ 80 % of E, and misses ≤ 5 % of E.
- **refuted**: misses ≥ 20 % of E.
- **unresolved**: otherwise.
- **not testable**: |E| < 20, or |E| below half the source's candidates (too many unconfirmed for E to
  stand for the candidates).

Sensitivity, reported beside the verdict and not part of it: unconfirmed candidates whose ΔD_nf fell
by more than 10× before the budget ended (κ at that point is a lower bound on any later κ) added to
both the misses and E. Verdicts are also reported without the eight pilot candidates.

Meaning [inference, stated in advance]: supported — S₁ maps built from returned endpoints (B-1, Task
A) describe the basins they sit in, and the cap did not distort them; refuted — returned endpoints
are not basin minima in the sense S₁ needs, so S₁ must be computed at refined minima, and the
truth-basin share tells how many "wrong structures" were only stalled searches.

### Q-BM2 [registered count]: a wrong basin near realistic counts

The number of distinct confirmed wrong basins (section 4) with S₁\* ≥ 10⁴ counts, per source, listed
with their conditions. Meaning [inference]: one or more — within these recorded classes a wrong
structure comes within ΔD = 1 of the truth at 10⁴ counts or above (SYNTHESIS 4.2 item 2, partly);
none — not within these records, which does not exclude other conditions.

### H-BM3 [registered]: B-1's descriptions persist at the minima

Unit: distinct confirmed wrong basins. New tests on new points; they do not change the H-F or H-W2
verdicts.
- (a) ρ\* ≥ 10 for ≥ 80 % of the basins.
- (b) shares low + b0 … b4 (q ≤ 0.235 Å⁻¹) ≥ 0.9 for ≥ 80 % of the 291-grid basins.
- (c) largest region b0, b1 or b2 for ≥ 80 % of the 291-grid basins.

Each **holds** at ≥ 80 %, **fails** below 60 %, **unresolved** between; H-BM3 **persists** if all three
hold, **does not persist** if any fails, unresolved otherwise. Not testable for a statement with fewer
than 20 basins.

### Descriptive (no verdict)

κ by capped / not capped, condition, stack, nuisance set and 2θ_max; evaluations to convergence;
outcomes and stall points; parameters on bounds; drift in u; endpoints per basin; for X2e, O_tr
outcomes, κ under O_tr and the distance between the O_tr and O_nf minima.

### Not concluded

Anything global: the number of basins in the landscape, multimodality, global uniqueness, where
basins lie that no recorded endpoint reached. The minima are local minima of a bounded box reached by
continuing a local optimiser from recorded endpoints. R2a's labels, X2e's fail and the earlier
verdicts are not revisited.

## 6. Pilot and budget report (owner's order)

- The eight candidates of section 2, O_nf for all, O_tr also for the two X2e candidates: ten
  refinements. One process, torch 1 thread, BelowNormal priority. The wrapper stops the run at 30 min
  wall time or 2 GB working set of the Python process tree (the child interpreter, not the venv
  launcher; B-2's memory number failed on that). Estimate: R2a 6 × 3000 × ~0.06 s ≈ 18 min and X2e
  4 × 3000 × ~0.03 s ≈ 6 min if every refinement spends the whole budget; less otherwise.
  Refinements not finished at a stop are reported as not run.
- Measured and reported: wall time, evaluations, model, value and Hessian calls, peak memory, outcome,
  NaN or non-finite values, bound contacts, κ, ρ\*, shares and the duplicate relation of
  r2a:06:44 / r2a:06:11 — all labelled [pilot], none used for a verdict.
- Output: `RESULTS-BASIN-PILOT.md` (workspace `docs/xrr/GEN-PHASE1/` and `reports/` here) with the
  budget for the full run: candidates × time and memory per candidate (pilot median and worst case at
  N_max), for 1, 2 and 4 processes. **The lane stops there; the full run is the owner's decision.**
- The full run, if approved, re-runs the pilot candidates (the code is deterministic; their rows are
  compared with the pilot's) and the verdict code is committed before it starts.

## 7. Cost and constraints

Pilot ≤ 30 min, one process, no HEAVY slot. No global start, no sampler, no change to any R2a, X2e,
B-1 or Task A record, no PR (branch `research/fisher-global`, pushed). Scripts run as files with
`python -I`.
