# Research experiments

ReflectoLearn holds a numbered series of experiments on fitting X-ray reflectivity (XRR) curves:
what one curve determines, and whether a fitting procedure can be learned. Each experiment has a
registration written before its run, the run records, a report generated from those records by a
script, and a verdict that applies the registered rules. This page lists them.

Where things are:

- `experiments/NNN_<topic>/`: the scripts that run an experiment and write its report.
- `docs/<topic>/`: its registration, records, reports and verdict. The directory names are the ones
  the scripts and registrations use, so they are not renamed.
- Tags `frozen/expNNN-<part>-<sha>`: the commits the registrations cite. They keep those commits
  reachable now that the research branches are squash-merged and deleted.
- Release [`research-data-1`](https://github.com/NaNIsaac-Research/ReflectoLearn/releases/tag/research-data-1): result files too large for the repository, with their SHA-256.
- Reviews: physics cross-reviews by another lane are in the pull-request comments; the code reviews
  of the fit-policy arms are in this directory.

Status labels are the verdicts as the registered rules produced them. "Supported" and "refuted" are
about the registered hypotheses on synthetic data, not general claims.

| # | Topic | Question | Verdict | Pull request | Frozen tag |
|---|---|---|---|---|---|
| 001 | Identifiability | Are wrong single-film fits optimizer failures or missing information? | optimizer failures within refine's bounds | #24 | `frozen/exp001-r2a-829f330` |
| 002 | Fit policy X2a/X2b | Does a scripted fitting protocol beat a full fit, so that learning one is worth it? | no room for RL | #36 | `frozen/exp002-x2b-887c4da` |
| 003 | Fit policy X2d | Why do the protocol's extra layers fail on held-out points? | H2 supported, H1/H3/H4 refuted; primary passes but not attributable | #39 | `frozen/exp003-x2d-690a18b` |
| 004 | Fit policy X2e | Does a better start basin remove the extra layers? | fail; basin reading supported | #41 | `frozen/exp004-x2e-e1b15ac` |
| 005 | Fisher against global fits | Does local Fisher information describe what a global fit can tell apart? | H-F unresolved, H-W2 and H-cal2 supported, H-BM1 supported | fisher-global | `frozen/exp005-basin-51b048a` |
| 006 | Objective | What do log-type objectives cost against the Poisson likelihood? | registered, not run | this PR | |

All results are on synthetic curves except H-cal2 (repeated public scans). No measured film has been
validated (see the top-level README).

## 001: identifiability

**Question.** When a fit of a single film returns the wrong structure, did the optimizer stop in the
wrong place, or does the curve not contain the answer?

**Registration and method.** [`../identifiability/README.md`](../identifiability/README.md) (model,
method, rules); R2a's rules in [`refine_causes.md`](../identifiability/refine_causes.md).

**Results.** Fisher information and CRLB tables (`fim_tables.md`), the non-uniqueness scan
(`scan_tables.md`, `scan/`), refine's failure causes (`refine_causes.md`), MCMC widths
(`mcmc_width.md`), and the go/no-go for a generative model (`generative_go_nogo.md`). Literature:
`literature.md`.

**Verdict.** The three single-film cases are identifiable within refine's bounds; refine's wrong fits
are optimizer failures (issue #25). The MCMC widths were withdrawn (#29, #40). A generative model is a
no-go for now. Physics reviewed R1b, R2a single film and R2a bilayer on #24 and agreed; its
objections (the profile is an upper bound on the profile likelihood; one deciding amendment was not
pre-registered) are recorded in the README.

**Rebuild the reports.**

```bash
uv run python experiments/001_identifiability/scan.py report
uv run python experiments/001_identifiability/refine_causes.py report
uv run --group figures python experiments/001_identifiability/nonunique.py report
uv run python experiments/001_identifiability/mcmc_width.py report
```

## 002: fit policy X2a/X2b

**Question.** A person fitting a multilayer curve decides how many layers to use, which parameters
to open, and when to stop. Does a scripted protocol of those decisions (A1) beat fitting every layer
count with every parameter free (C)? If not, a learned policy has no room to improve.

**Registration.** [`../fit_policy/DESIGN.md`](../fit_policy/DESIGN.md), frozen at 887c4da.

**Results.** [`results.md`](../fit_policy/results.md), [`verdict.json`](../fit_policy/verdict.json),
records in `runs/`.

**Verdict.** No room for RL: A1 beats C in 0 of 4 multilayer cells. A1 wins on BIC but adds layers
that fail on held-out points, which X2d and X2e then examine.

```bash
uv run python experiments/002_fit_policy/report.py
```

## 003: fit policy X2d

**Question.** Why do A1's extra layers fail on held-out points? Four registered causes: unmatched
comparisons (H1), a wrong basin compensated by layers (H2), held-out blocks that leave q ranges
unconstrained (H3), and the best-BIC return bypassing the acceptance margin (H4).

**Registration.** [`../fit_policy/X2D.md`](../fit_policy/X2D.md) with amendment 1 and the physics
conditions on #39.

**Results.** [`x2d_results.md`](../fit_policy/x2d_results.md),
[`x2d_verdict.json`](../fit_policy/x2d_verdict.json), records in `x2d_runs/`. Code review of the
arms: [`review-x2d-code.md`](review-x2d-code.md) (one deviation, #50).

**Verdict.** H2 supported; H1, H3 and H4 refuted. The primary arm passes in 2 cells, but the
unchanged A1 meets the same criteria there, so the pass is not attributable to the changes. The
registered confirmation run was not run; X2e was registered instead.

```bash
uv run python experiments/003_fit_policy_x2d/report.py
```

## 004: fit policy X2e

**Question.** If the extra layers come from a wrong start basin (X2d's H2), do better starts for new
layers (S1) or for the whole stack (S2) remove them?

**Registration.** [`../fit_policy/X2E.md`](../fit_policy/X2E.md) with amendments 1-2 and the physics
conditions on #41.

**Results.** [`x2e_results.md`](../fit_policy/x2e_results.md),
[`x2e_verdict.json`](../fit_policy/x2e_verdict.json), records in `x2e_runs/`. The aborted first run
is kept outside the tree ([`x2e_runs_aborted_1c38e61.md`](../fit_policy/x2e_runs_aborted_1c38e61.md)).
Code review: [`review-x2e-code.md`](review-x2e-code.md).

**Verdict.** Fail: the primary S12 beats C in 0 of 4 cells. The basin reading (D1) is supported;
the source tests B1 and B2 are unresolved, B3 refuted.

```bash
uv run python experiments/004_fit_policy_x2e/report.py
```

## 005: Fisher information against global fits

**Question.** Local Fisher information says which parameters a curve determines near the truth. Does
it also describe what a global fit can tell apart, and where in q the discrimination comes from?
Registered as H-F (Fisher ratio inside and outside the noise region), H-W2 (low-q share of the
deviance at wrong endpoints), B-2 (a truth-start profile, observation) and H-cal2 (overdispersion of
repeated public scans); then the basin-minimum validation (H-BM1-3).

**Registration.** [`../fisher_global/PREREG.md`](../fisher_global/PREREG.md) and
[`PREREG-BASIN.md`](../fisher_global/PREREG-BASIN.md), registered at 51b048a.

**Results.** Reports in [`../fisher_global/reports/`](../fisher_global/reports/) (RESULTS-B1, -B2,
-HCAL2, -BASIN-PILOT, -BASIN, BASIN-NOTE) and their synthesis
[`SYNTHESIS.md`](../fisher_global/reports/SYNTHESIS.md); verdicts in `b1_verdict.json` and
`basin_verdict.json`. The 935 per-candidate files of the full basin run are the release asset
`basin_full.zip`; the repository keeps `basin_full_summary.jsonl` (every field but the optimizer
traces).

**Verdict.** H-F unresolved (0.831 of near endpoints in range, 0.90 needed); H-W2 supported; H-cal2
supported under amendment 1. Basin minima: H-BM1 (the S₁ of a returned endpoint represents the basin
it converges to) supported for the R2a and X2e candidates; H-BM3 (B-1's descriptions hold at the
refined minima) persists for R2a and is unresolved for X2e.

```bash
uv run python experiments/005_fisher_global/b1.py report
uv run python experiments/005_fisher_global/basin_summary.py unpack
uv run python experiments/005_fisher_global/basin_report.py
```

`basin_report.py` orders tied rows of two tables by string hashing, so set `PYTHONHASHSEED` to get
the same file twice.

## 006: objective

**Question.** How much information, bias and interval calibration do log-type XRR objectives (refine's
log-MSE, refnx's unweighted `logY`) cost against the Poisson likelihood, and which part of the objective
causes it?

**Registration.** [`../objective/PREREG.md`](../objective/PREREG.md): six claims with decision rules,
a one-cell pilot and the run plan. Code: `experiments/006_objective/objective.py`.

**Verdict.** Not run yet.

## Before 001

The first exploration (2025, refnx models, notebooks) is an unrelated history kept at tag
`archive/old-2025`. The work in other repositories is in the
[XRR research index](../xrr-research-index.md).
