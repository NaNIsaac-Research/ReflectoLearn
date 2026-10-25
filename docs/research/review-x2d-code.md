# Code review: X2d changes to `src/reflectolearn/fit_policy` (#39)

Scope: `git diff origin/main...b796ab2 -- src tests` (baselines.py +68, env.py +100/-15, protocol.py
+53/-6, stack.py +29/-10, tests/test_fit_policy_x2d.py +96). Checked against `docs/fit_policy/X2D.md`
section 4 and amendment 1 (a)-(h). The registration and the verdict were checked by physics on #39;
this note covers only the code that produced the registered run (frozen tag
`frozen/exp003-x2d-690a18b`). It is a code review by the xr1 lane, not the physics cross-review.

## What was run

| Check | Result |
|---|---|
| `uv run --python 3.13 pytest -q tests/test_fit_policy.py tests/test_fit_policy_x2d.py` at b796ab2 | 19 passed |
| A1 on four unregistered inputs (n, j = 1..3, 50..53; budget 800), SHA-256 of the fields `result()` had before X2d, at main 1fd2ace and at b796ab2 | identical (`0738331d...`) |
| `critical_q` on all training points vs training minus inner block, root 20261025, n 1..4, j 0..23, every noise level (192 data sets) | 1 differs, by 8.9e-7 Å⁻¹ |

## Against the registration

| Item (X2D.md) | Code | Agrees |
|---|---|---|
| Inner block: training block 4 of 10 above 2θ 0.5°, q 0.1964-0.2348 Å⁻¹, 28 points, n = 208 fit points | `stack.inner_mask`, same split as `held_out_mask`; `run.data` takes it on the full grid and restricts it to training points; the test checks 28, the q range, no overlap with blocks 2 and 6, and 208 | yes |
| A1m: refit the current state as a candidate is; accept if BIC ≤ refit BIC − 10; otherwise continue from the refit state to step 9 | `_refit_current` forks and calls `_refit`, the candidate path; `_accepts` uses `candidate.bic - base.bic <= p.accept` (−10); on rejection `env = base; break` | yes |
| Step 8 matched: keep the merge if its BIC does not exceed the refit's; on rejection continue from the refit state (amendment 1e) | `_merge_pass`: `merged.bic <= base.bic`, else `env = base` | yes |
| A1v (b): inner deviance strictly lower at step 7; at step 8 neither BIC nor inner deviance rises | `<` in `_accepts`, `<=` in `_merge_pass` | yes |
| Kept path: states created by the final env and the forks it adopted; a revert to an unadopted fork's state stays off the path | `FitState.origin` is set to the env's id after every action except `revert`, before `_evaluate`; `lineage` is the set of ids of the env and the forks it descends from; `revert` keeps the target's origin | yes |
| A1k: revert and return use the kept path's lowest BIC | `_revert_target` uses `path_best`; `returned` uses `path_best_all` (BIC on all training points) | yes |
| A1v return: lowest BIC on all training points among kept-path states; step 9 rejoins the inner block | `bic_all` is computed on all training points in `_evaluate`; `use_all_points` after the step-9 revert | yes |
| Cv (amendment 1b): four L fits of (E − 100) / 4 = 725 evaluations on training minus inner, choose by inner deviance, refit on all training with fit(100), charged to E | `inner_full_fit`: `per_l = (budget - CV_FINAL_FIT) // L_MAX` = 725, `data_start` on the fit points (amendment 1h), final `st.fit(..., 100)` with every parameter free; evaluations summed | yes |
| TF: truth as start, all free, Poisson, one fit with max_nfev = E | `truth_fit` | yes |
| A1o: steps 2-3 skipped | `Protocol.estimate=False` | yes |
| Seed root for every stream, default unchanged | `root` argument on `sample_truth`, `noisy_counts`, `starts`, `extra_layers`, default `SEED`; tested | yes |
| Instrumentation does not change behaviour | `decisions`, `history`, `on_kept_path`, `kept_path_states` are written, never read by a protocol; the A1 digest above is unchanged | yes |

Units: q in Å⁻¹ throughout, SLD in 1e-6 Å⁻², BIC = D + k ln n with n the number of points the fit
used (208 for A1v in steps 1-8, the training count otherwise). No unit or formula changes in the
forward model or the deviance; `stack.py` only gains `inner_mask` and the seed root.

## Findings

1. **Inner block still enters the critical-edge estimate of A1v (amendment 1h).** `FitEnv.__init__`
   computes `episode.qc_data` and `episode.kiessig` from the `q, y` it is given, all training points,
   before the fit mask applies. `kiessig` is read only by the observation vector (learned policies,
   not the scripted arms). `qc_data` sets the lower bound (1.1 qc) of the residual fringe search
   that sizes a new layer at step 7, so it reaches A1v. Measured effect: 1 of 192 registered data
   sets has a different qc, by 8.9e-7 Å⁻¹ (the inner block lies far above the edge). The estimates
   that amendment 1h names explicitly, `estimate_rho` and `estimate_d`, do use the masked points.
   Effect on the verdict: none expected (inference: a shift of 1e-6 in the lower bound of a search
   window). Not fixed here, because the registered run is frozen at 690a18b; worth a one-line fix
   (compute both from `self.q, self.y` after the mask is set) before any reuse of A1v; issue #50.
2. **BICs over different n are compared after step 9 in A1v.** `use_all_points` calls `_evaluate`,
   which compares the all-points BIC with `path_best_bic` and `episode.best_bic` from the masked
   phase. Nothing reads either after step 9 (no revert follows; `returned` uses `path_best_all`,
   which is always on all points), so the result is unaffected. A fit mask without `kept_path` is
   not a registered arm but would return `episode.best` with mixed n; an assert in `FitEnv` would
   rule that combination out.
3. **Cv's `capped` refers to the chosen L's 725-evaluation fit, not to the final refit.** Reporting
   only; the cap rule (DESIGN 6.5) is read on the C-style fits.
4. **Test gaps.** No test drives a rejected step-7 candidate in A1m to check that the run continues
   from the refit state, none covers the matched merge branch, and A1o has no test. The A1-unchanged
   claim in the PR is checked by the X2b tests and by the digest above, not by a stored digest in
   the test suite.

## Verdict of this review

The code implements section 4 and amendment 1 as registered, with one deviation (finding 1) whose
measured size is below anything the verdict could resolve. No finding changes a number in
`x2d_results.md` (inference for finding 1, from the measurement above; the run was not repeated).
Whether that is enough for the team's physics rule on `src/` changes is for boss to decide.
