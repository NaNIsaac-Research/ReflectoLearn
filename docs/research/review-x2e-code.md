# Code review: X2e changes to `src/reflectolearn/fit_policy` (#41)

Scope: the X2e-only part, `git diff b796ab2 a14041f -- src/reflectolearn/fit_policy tests`
(baselines.py +67, env.py +27, protocol.py +92, stack.py +41 of which +8 are the #44 clamp,
tests/test_fit_policy_x2e.py +83). The X2d part is reviewed in `review-x2d-code.md`, the clamp on
#44. Checked against `docs/fit_policy/X2E.md` section 4 and amendments 1-2. The registration and
the verdict were checked by physics on #41; this note covers the code of the registered run (frozen
tag `frozen/exp004-x2e-e1b15ac`). It is a code review by the xr1 lane, not the physics cross-review.

## What was run

| Check | Result |
|---|---|
| `uv run --python 3.13 pytest -q tests/test_fit_policy.py tests/test_fit_policy_x2d.py tests/test_fit_policy_x2e.py` after merging main b5f554c | 26 passed |
| A1 on four unregistered inputs, SHA-256 of the pre-X2d `result()` fields, as in the X2d review | identical to main (`0738331d...`), with the clamp included |

## Against the registration

| Item (X2E.md) | Code | Agrees |
|---|---|---|
| 4.1 S1: up to 9 starts per `add`, three strongest residual peaks (z 20-400 Å) × ρ factors 1.0, 0.75, 1.25 of A1's neighbour mean, σ 3 Å; screen with `loss(log)`, `fit(20)`; lowest BIC refit as A1 refits; no peak: A1's single 50 Å start; `split` unchanged | `_s1_add`: `fringe_peaks(..., RESIDUAL_Z, k=3)` (RESIDUAL_Z 20-400), `S1_FACTORS`, `NEW_LAYER_SIGMA` 3.0, `_screen`; returned fork goes through `_refit` in `_try`; `[(None, 1.0)]` without peaks; only `add` is routed to S1 | yes |
| 4.1 exposure: chosen start differs from A1's (strongest peak, mean ρ) | `starts[0]` is (strongest peak, 1.0), which is A1's own `add` (both take the strongest local maximum of the same transform); `s1_changed.append(best != 0)` | yes |
| 4.2 transform: training points, log10 counts floored at 0.5, q > 1.1 q_c, quintic detrend, Hann window, z 20-800 Å | `_s2_stacks`: `LOG_FLOOR_COUNTS` 0.5, `1.1 * env._qc()`, `S2_Z`; `fringe_peaks` repeats `fringe_spectrum`'s detrend, window and transform | yes |
| 4.2 peak rule: local maxima ≥ 0.25 of the largest, greedy by amplitude, separation 2π / (q_max − q_min), at most 4 | `fringe_peaks`: `PEAK_REL`, `separation`, `PEAK_MAX`; tested by `test_peak_rule` | yes |
| 4.2 peaks to stacks: m strongest sorted by z, read as interface depths, both orders, thickness < 5 Å dropped, ≤ 7 stacks plus A1's step-3 state | `np.diff([0, *sorted(peaks[:m])])`, reversed for m > 1, `LAYER_BOUNDS["d"][0]` = 5 Å; `candidates = [env.state.stack, *stacks]` | yes |
| 4.2 start values: top ρ and σ from the step-3 state, deeper ρ_i = (ρ_{i−1} + ρ_Si)/2, σ 3 Å below, step-3 instrument | `_s2_stacks` | yes |
| 4.2 screening: d and ρ open, `loss(log)`, `fit(20)`, lowest BIC, continue with step 4 | `_s2_start`; the chosen fork becomes the protocol's env, so it is on the kept path | yes |
| 4.3 S12: S2's start, then S1 | `Protocol("S12", s1=True, s2=True)` | yes |
| 4.4 true-L data start from A1's step-3 state, d split equally | `step3_state` replays A1 steps 1-3 (`open bkg`, `fit(20)` with the staged first loss, the two estimates) without charging an arm; `true_l_data_start` | yes |
| 4.4 D1: last kept-path state at the true L, one free Poisson fit with max_nfev = E, outside A1's budget | `kept_state_at` over `history`; `free_fit_deviance` calls `st.fit` directly, not through the env | yes |
| 4.4 D2: DE, popsize 15, maxiter 50, Latin hypercube, x0 the true-L data start, no polish, section-3 seed, then Poisson `fit(100)` | `de_fit`; x0 clipped into the bounds | yes |
| 4.4 D3: A1 from the true-L data start, steps 2-3 skipped | `Protocol("D3", estimate=False)` | yes |
| Every evaluation charged to E | screening forks share the episode, so their `fit` calls draw on the same budget; `set_stack` evaluates without fitting and charges nothing, as amendment 2's estimates | yes |

## Findings

1. **What is open during S1's screening.** X2E.md 4.1 says each start "gets the new layer's
   parameters open". The code opens the new layer and leaves every other open flag as it is; at
   step 7 of A1 every layer and instrument group is already open, so the screening fit moves all
   parameters, as A1's own candidate refit does. That is the natural reading (the new layer opened
   in addition), but the text can be read the other way. No change proposed; recorded so the reading is explicit.
2. **S2 screening also fits the background.** `bkg` is open from step 1 and stays open during the
   `fit(20)` screen, where 4.2 names d and ρ. All eight candidates are screened the same way, so the
   comparison between them is unaffected.
3. **S1 without a peak still screens.** With no residual peak the single A1 start gets `fit(20)`
   before the refit, 20 evaluations A1 would not spend. 4.1 is silent on it; charged to E.
4. **Duplicated transform.** `fringe_peaks` repeats the body of `fringe_spectrum` (detrend, window,
   transform, local-maximum test) instead of sharing it. Both use the same local-maximum test, so
   the first peak of one is the peak of the other, which S1's exposure count relies on. A shared
   helper would keep the two from drifting; not changed in the frozen code.
5. **Tests.** The peak rule, the S2 stack rules, S12's exposure and budget, the true-L data start
   and D2's bounds are tested. No test checks that S1's first start equals A1's `add` start, which
   the exposure count assumes (finding 4).

## Aborted run directory

`docs/fit_policy/x2e_runs_aborted_1c38e61/` (8 files, 3.7 MB) is the run that stopped at 1c38e61.
Amendment 2 keeps it and reads no outcome metric from it; no result, verdict or report reads it.
It is moved out of the tree into a release asset (see `x2e_runs_aborted_1c38e61.md` next to it),
so the registration's statement that the records are kept still holds, with their checksum.

## Verdict of this review

The code implements section 4 as registered. Findings 1-3 are readings of the registration text
that the code makes explicit, none of them changes which arm is compared with which, and no finding
changes a number in `x2e_results.md` (the run was not repeated). Whether that is enough for the
team's physics rule on `src/` changes is for boss to decide.
