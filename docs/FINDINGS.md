# Findings

What this repository has shown so far, newest first. Each item links its evidence.

Results use synthetic curves except F9, which uses repeated public scans. No measured film has been
validated. F numbers stay fixed; checked refers to the written verification cited in Limits, not a
new run for this index.

## F10: Returned fit endpoints usually represent the minima of the basins they reach

- Question: Does the count level S₁ at a returned R2a or X2e endpoint represent its basin minimum, and how many wrong basins need S₁* ≥ 10⁴ counts?
- Result: H-BM1 is supported for R2a (98.3 % κ ≤ 2, misses 0.2 %) and X2e (95.7 %, misses 1.1 %). There are 5 R2a and 1 X2e wrong basins with S₁* ≥ 10⁴ counts; H-BM3 persists for R2a and is unresolved for X2e ((b) at 77.5 %).
- Figure: none
- Data and script: [basin_full_summary.jsonl](fisher_global/basin_full_summary.jsonl), [basin_verdict.json](fisher_global/basin_verdict.json), [basin_report.py](../experiments/005_fisher_global/basin_report.py), [basin.py](../experiments/005_fisher_global/basin.py)
- Status: checked
- Limits: Local refinement from recorded endpoints on noise-free data inside bounded boxes; 651 of 656 R2a minima are on a bound. Nothing global; interior-only H-BM1 is not testable, and noisy and measured data are not tested. Checked: the report records convergence tests C1–C4, start checks for all 935 refinements, and [bit-for-bit reproduction of the ten pilot refinements](fisher_global/basin_pilot_repro.md).
- Report: [Basin minimum validation: full run, results and verdicts](fisher_global/reports/RESULTS-BASIN.md) (2026-10-25)

## F9: Counts in repeated public scans vary more than Poisson statistics predict

- Question: Are the raw counts of 100 repeated public synchrotron XRR scans overdispersed relative to Poisson?
- Result: Under amendment 1, H-cal2 is supported: φ > 1.2 at 64.2 % of the points against 8.5 % expected for Poisson, with median φ 1.34. Dividing out scan-to-scan intensity leaves a median of 1.31.
- Figure: none
- Data and script: [hcal2.json](fisher_global/hcal2.json), [hcal2_precheck.json](fisher_global/hcal2_precheck.json), [hcal2.py](../experiments/005_fisher_global/hcal2.py)
- Status: draft
- Limits: One sample, one exposure and one fast-scan synchrotron instrument, not laboratory step scans; the noise source is not identified. Amendment 1 followed the input pre-checks; under literal pre-check 3 there is no verdict. The report records input checks, but no independent verification of the dispersion numbers. Raw public scans are stored outside this repository; the linked JSON contains summary statistics.
- Report: [H-cal2 results: count dispersion in 100 repeated public XRR scans](fisher_global/reports/RESULTS-HCAL2.md) (2026-10-25)

## F8: One bilayer is locally determined except for the first film's thickness

- Question: Inside the truth's basin of R2a bilayer condition #25, how well are the film parameters determined locally?
- Result: The truth-start fit ends at the truth; d₁ is a ridge (2.24 % half-width), while ρ₁, d₂ and ρ₂ are identifiable (0.12–0.27 %). Bounded profile half-widths are 0.43–0.82 of the unbounded Fisher values because Fisher's weakest direction runs through Kα2 ratios outside their bounds.
- Figure: none
- Data and script: [b2.json](fisher_global/b2.json), [b2_extra.json](fisher_global/b2_extra.json), [b2.py](../experiments/005_fisher_global/b2.py), [b2_extra.py](../experiments/005_fisher_global/b2_extra.py)
- Status: draft
- Limits: One truth-start fit on noise-free data in one condition; this observation tests no hypothesis. Profile points are local fits, so half-widths can only be too narrow. Memory was not measured; global uniqueness, multimodality and #25's R2a label are not established. No independent check of these widths is recorded.
- Report: [B-2 results: truth-start local profile of R2a bilayer #25](fisher_global/reports/RESULTS-B2.md) (2026-10-25)

## F7: Wrong-fit discrepancies concentrate at low q, while the Fisher-ratio test stays unresolved

- Question: Does the Fisher quadratic match the deviance near the truth and overstate it at wrong endpoints, and does a wrong endpoint's deviance sit at low q?
- Result: H-F is unresolved: part (i) holds for 0.831 of 177 near endpoints (0.90 needed), and part (ii) holds at 0.981 of 2191 wrong endpoints. H-W2 is supported: ≥ 90 % of ΔD lies at q ≤ 0.235 Å⁻¹ for 0.996 of the test endpoints, and the largest share lies in b0–b2 for 0.953.
- Figure: none
- Data and script: [b1.json](fisher_global/b1.json), [b1_verdict.json](fisher_global/b1_verdict.json), [b1.py](../experiments/005_fisher_global/b1.py)
- Status: checked
- Limits: Noise-free data from the fitted model; endpoints under a 200-evaluation cap are not basin minima (F10 examines minima). Fisher is evaluated only at the truth; one count level, background and truth per condition; the H-F (i) band was chosen with X2e's near states in view. Checked: the report records recomputed deviance matching recorded endpoints to 2.8 × 10⁻¹⁰ relative difference and the Fisher quadratic matching ΔD to 2 % along 8 random directions in all 37 conditions.
- Report: [B-1 results: R2a endpoints against the Fisher quadratic (H-F, H-W2)](fisher_global/reports/RESULTS-B1.md) (2026-10-25)

## F6: Better starting values did not remove the extra-layer problem

- Question: Do better starts for new layers or the whole stack remove the extra layers attributed to a wrong start basin?
- Result: The primary S12 beats C in 0 of 4 cells (0 attributable): verdict fail. Basin reading D1 is supported (0 of 53 within margin); sources B1 and B2 are unresolved, B3 is refuted, and no cause is named.
- Figure: none
- Data and script: [x2e_verdict.json](fit_policy/x2e_verdict.json), [x2e_runs/](fit_policy/x2e_runs/), [report.py](../experiments/004_fit_policy_x2e/report.py), [run.py](../experiments/004_fit_policy_x2e/run.py)
- Status: checked
- Limits: Synthetic curves only; the source of the wrong basin remains unresolved. The aborted first run is kept outside the tree as documented in the research index. Checked: the [X2e code review](research/review-x2e-code.md) records 26 passing tests and an unchanged A1 digest, and states that physics checked the registration and verdict. The review did not repeat the run and records ambiguities in the screening rules.
- Report: [Experiment 004 (X2e) results](fit_policy/x2e_results.md) (2026-10-25)

## F5: Extra layers compensate for a wrong basin, but the protocol change has no attributable pass

- Question: Why do the scripted protocol's extra layers fail on held-out points?
- Result: H2, a wrong basin compensated by layers, is supported; H1, H3 and H4 are refuted, with partial replication. The primary A1mk passes in 2 cells, but unchanged A1, A1m and A1k have the same outcomes there, so the pass is not attributable to the protocol changes.
- Figure: none
- Data and script: [x2d_verdict.json](fit_policy/x2d_verdict.json), [x2d_runs/](fit_policy/x2d_runs/), [report.py](../experiments/003_fit_policy_x2d/report.py), [run.py](../experiments/003_fit_policy_x2d/run.py)
- Status: checked
- Limits: Synthetic curves only; the registered confirmation run was not run, and X2c stays unregistered, as the [research index](research/README.md#003-fit-policy-x2d) explains. Checked: the [X2d code review](research/review-x2d-code.md) records 19 passing tests and an unchanged A1 digest; physics checked the registration and verdict. It records one deviation in the inner-block critical-edge estimate (#50); its verdict impact is an inference, and the run was not repeated.
- Report: [Experiment 003 (X2d) results](fit_policy/x2d_results.md) (2026-10-24)

## F4: The scripted fitting protocol did not beat a full fit in the multilayer test cells

- Question: Does a scripted fitting procedure beat fitting every layer count with every parameter free under the same compute budget?
- Result: A1 beats C in 0 of 4 multilayer cells: verdict no room for RL. It beats the random floor B in all 6 cells; against C, A1 wins on BIC for 14–16 of 16 multilayer truths per cell but loses on held-out points for 2–3.
- Figure: [delta_bic.png](fit_policy/delta_bic.png)
- Data and script: [verdict.json](fit_policy/verdict.json), [runs/](fit_policy/runs/), [report.py](../experiments/002_fit_policy/report.py), [run.py](../experiments/002_fit_policy/run.py)
- Status: draft
- Limits: Synthetic curves only; the verdict applies to the registered cells and rules, not every possible learned policy. The generated report records the verdict and per-cell outcomes but no independent verification of these result numbers.
- Report: [Experiment 002 (X2a/X2b) results](fit_policy/results.md) (2026-10-24)

## F3: The analysed conditions did not establish a need for a generative posterior model

- Question: Under which conditions does an XRR curve admit several statistically indistinguishable structures, justifying a generative posterior model?
- Result: The report's verdict is no-go for now: no analysed condition has a multimodal label (19 single-film and 5 reached bilayer conditions, one mode each). However, 13 of 18 bilayer conditions are undecided because no start reached the truth's basin; nothing was trained.
- Figure: none
- Data and script: [scan_tables.md](identifiability/scan_tables.md), [scan/](identifiability/scan/), [scan.py](../experiments/001_identifiability/scan.py)
- Status: checked
- Limits: Noise-free data at 10⁶ counts and a registered 2 % ridge line; weak negatives where only 1–3 of 64 starts reached the truth basin, and many fits stopped at the 200-evaluation cap. The bilayer half is largely untested; no global uniqueness is established. Earlier MCMC widths in the report were withdrawn (#29, #40). Checked: the [research index](research/README.md#001-identifiability) records physics review of R2a single-film and bilayer results, with objections retained.
- Report: [Generative inverse model for XRR: go / no-go (R1d)](identifiability/generative_go_nogo.md) (2026-10-23)

## F2: Several fitting changes are needed together, and the opt-in preset missed confirmation

- Question: Which settings make refine stop short of the maximum-likelihood fit?
- Result: Baseline reaches 0/48 and no factor alone is sufficient; all on reaches 48/48. The ABCF preset reaches 46/48 in sample but 42/48 in the registered out-of-sample check (43 needed), so it is not confirmed and stays opt-in.
- Figure: none
- Data and script: [refine_causes.json](identifiability/refine_causes.json), [refine_confirm.json](identifiability/refine_confirm.json), [refine_causes.py](../experiments/001_identifiability/refine_causes.py)
- Status: checked
- Limits: Three synthetic cases on refine's own forward model, one noise draw each; the D* reference was amended after the first runs, and the ABCF combination was not registered. Three R2f misses sit 0.1 above the ΔD ≤ 4 threshold. Checked: the report records the ×20 convergence check also reaching 48/48 and the registered out-of-sample R2f check; that check did not confirm the preset.
- Report: [Why refine stops short of the best fit (R2b)](identifiability/refine_causes.md) (2026-10-23)

## F1: Three single films are identifiable within the fitting bounds, despite wrong refine endpoints

- Question: When a single-film fit returns the wrong structure, did the optimizer fail or does the curve lack the information?
- Result: All three cases are identifiable given refine's bounds (Δχ² ≤ 1 widths ±0.04 % to ±1.4 %, along the tracked branch). Of 48 wrong refine endpoints, 43 are optimization failures and 5 are objective or model errors; the split rests on unregistered amendment 3.
- Figure: [nonunique.png](identifiability/nonunique.png)
- Data and script: [nonunique_mid_oxide.json](identifiability/nonunique_mid_oxide.json), [nonunique_mid_no_oxide.json](identifiability/nonunique_mid_no_oxide.json), [nonunique_t2_low.json](identifiability/nonunique_t2_low.json), [nonunique.py](../experiments/001_identifiability/nonunique.py)
- Status: checked
- Limits: Three synthetic films; every label depends on refine's bounds. The profile follows one branch and is an upper bound on the profile likelihood; the deciding amendment was not pre-registered. MCMC chains were unconverged and their widths withdrawn. Checked: the [research index](research/README.md#001-identifiability) records physics review and retains these objections; the report's Checks section documents independent Parratt and finite-difference checks for the forward model and Fisher calculation.
- Report: [Experiment 001: how much of a single film does one XRR curve determine?](identifiability/README.md) (2026-10-23)
