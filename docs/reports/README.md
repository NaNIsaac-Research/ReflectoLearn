# Reports

Every report of the registered experiments, newest first. The experiment-by-experiment view, with
registrations, frozen tags and rebuild commands, is [docs/research](../research/README.md). Status:
**checked** is a finished report; **draft** is not final; **superseded** says what replaced or
withdrew it. Reports marked *generated* are written by the named script and rewritten on every
run, so they carry no summary box; their question, result and limits are given below the table.
All results are on synthetic curves except H-cal2 (repeated public scans).

| Date | Report | Result | Status |
|---|---|---|---|
| 2026-10-25 | [Research note: are returned XRR fit endpoints basin minima?](../fisher_global/reports/BASIN-NOTE.md) | Returned fit endpoints stand for their basins (supported); a few wrong basins need ≥ 10⁴ counts. | checked |
| 2026-10-25 | [Basin minimum validation: full run, results and verdicts](../fisher_global/reports/RESULTS-BASIN.md) | Endpoint S₁ represents its basin (supported); 5 R2a and 1 X2e wrong basins need ≥ 10⁴ counts. | checked |
| 2026-10-25 | [Basin minimum validation: pilot and compute budget](../fisher_global/reports/RESULTS-BASIN-PILOT.md) | Pilot: all ten refinements confirmed; full run estimated at 9.8 CPU h (worst 47); no verdict. | draft |
| 2026-10-25 | [Synthesis of B-1, B-2 and H-cal2: is "Fisher against global discrimination" a paper?](../fisher_global/reports/SYNTHESIS.md) | Fisher-versus-global alone is not a paper; conditional yes inside an identifiability-versus-reach methods paper. | checked |
| 2026-10-25 | [H-cal2 results: count dispersion in 100 repeated public XRR scans](../fisher_global/reports/RESULTS-HCAL2.md) | Repeated public scans are overdispersed against Poisson: median φ 1.34, above 1.2 at 64 % of points. | checked |
| 2026-10-25 | [B-2 results: truth-start local profile of R2a bilayer #25](../fisher_global/reports/RESULTS-B2.md) | Bilayer #25 is locally determined except film 1's thickness (2.24 % ridge); unbounded Fisher widths too wide. | checked |
| 2026-10-25 | [B-1 results: R2a endpoints against the Fisher quadratic (H-F, H-W2)](../fisher_global/reports/RESULTS-B1.md) | Fisher test unresolved (0.831 near endpoints in band, 0.90 needed); wrong-fit misfit sits at low q (supported). | checked |
| 2026-10-25 | [Experiment 004 (X2e) results](../fit_policy/x2e_results.md) | Better starts fail: S12 beats C in 0 of 4 cells; wrong-basin reading supported, its source unresolved. | checked, generated (`experiments/004_fit_policy_x2e/report.py`) |
| 2026-10-24 | [Experiment 003 (X2d) results](../fit_policy/x2d_results.md) | Extra layers come from a wrong basin (H2 supported); primary passes in 2 cells but not attributably. | checked, generated (`experiments/003_fit_policy_x2d/report.py`) |
| 2026-10-24 | [Experiment 002 (X2a/X2b) results](../fit_policy/results.md) | No room for RL: the scripted protocol beats a full fit in 0 of 4 multilayer cells. | checked, generated (`experiments/002_fit_policy/report.py`) |
| 2026-10-23 | [Experiment 001: how much of a single film does one XRR curve determine?](../identifiability/README.md) | Within refine's bounds single films are identifiable; 43 of 48 wrong refine fits are optimizer failures. | checked |
| 2026-10-23 | [Why refine stops short of the best fit (R2b, ReflectoLearn#25)](../identifiability/refine_causes.md) | Budget, no stage 2, Poisson objective and step decay all needed; preset 46/48 in sample, 42/48 out-of-sample, not confirmed. | checked |
| 2026-10-23 | [Why refine.mcmc intervals disagree with the profile likelihood (R2c)](../identifiability/mcmc_width.md) | Shipped MCMC chains unconverged; mid_no_oxide's wide interval traced to the likelihood; widths later withdrawn. | superseded: the MCMC widths were withdrawn (#29, #40), no replacement |
| 2026-10-23 | [Generative inverse model for XRR: go / no-go (R1d)](../identifiability/generative_go_nogo.md) | No-go for a generative model: no multimodal condition found, but 13 of 18 bilayer conditions undecided. | checked |

Details for the generated reports:

- **Experiment 004 (X2e) results.** Question: If A1's extra layers come from a wrong start basin (X2d's H2), do better starts for new layers (S1) or for the whole stack (S2) remove them? Result: Verdict 'fail': the primary S12 beats C in 0 of 4 cells (0 attributable). The basin reading D1 is supported (0 of 53 within margin); sources B1 and B2 unresolved, B3 refuted, no cause named; as a third sample of the X2b question, A1 beats C in 1 of 4 cells. Limits: The generated report states no limits; docs/research/README.md: synthetic curves only, and the aborted first run is kept outside the tree.
- **Experiment 003 (X2d) results.** Question: Why do protocol A1's extra layers fail on held-out points: unmatched refit (H1), a wrong basin compensated by layers (H2), held-out blocks leaving q ranges unconstrained (H3), or the best-BIC return bypassing the margin (H4)? Result: H2 supported and named the cause; H1, H3 and H4 refuted; replication partial (held-out failure in 2 of 4 cells). The primary A1mk passes in 2 cells (verdict line 'primary passes: confirmation run on seed root 20261026; X2c stays unregistered'), but A1, A1m and A1k have the same outcomes there, so docs/research/README.md calls the pass not attributable, and the confirmation run was not run. Limits: The generated report states no limits; docs/research/README.md and X2E.md section 1: synthetic only, the pass cannot be attributed to the protocol changes and was not confirmed.
- **Experiment 002 (X2a/X2b) results.** Question: On synthetic single-, two- and three-layer curves, does a scripted fitting procedure (A1) end at a better model than fitting all parameters at once (C) with the same compute, so that a learned policy has room to improve? Result: Verdict 'no room for RL': A1 beats C in 0 of 4 multilayer cells (room_to_learn False, register_X2c False), while A1 beats the random floor B in all 6 cells. Against C, A1 wins on BIC for 14–16 of 16 multilayer truths per cell but loses on held-out points for 2–3. Limits: The generated report states no limits; docs/research/README.md: synthetic curves only, and 'supported'/'refuted' apply to the registered hypotheses on synthetic data.

Earlier XRR work in other repositories: [XRR research index](../xrr-research-index.md).
