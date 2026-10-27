# Research note: are returned XRR fit endpoints basin minima?

[Back to all reports](../../reports/README.md)

> **Question.** Does S₁ at a returned endpoint stand for the basin the endpoint sits in, or would continuing the local search change it or reach the truth?
>
> **Result.** H-BM1 supported for R2a (98.3 % κ ≤ 2, misses 0.2 %) and X2e (95.7 %, misses 1.1 %); Q-BM2 counts 5 (R2a) and 1 (X2e) wrong basins with S₁* ≥ 10⁴; H-BM3 persists for R2a and is unresolved for X2e ((b) at 77.5 %). Continuing the search barely changed the deviance (median κ 1.000 for R2a, 1.06 for X2e).
>
> **Limits.** Local only, noise-free objective, box-dependent minima (651 of 656 R2a minima held by bounds), interior-only analysis not testable.

xrr-policy lane, 2026-10-25. Written to move into the NaNIsaac-Research note system once it exists
(XN1). Full record with every table: `RESULTS-BASIN.md` in this folder (ReflectoLearn
`research/fisher-global`, `docs/fisher_global/reports/RESULTS-BASIN.md`). Labels: [registered] fixed
before the run, [fact] measured, [post hoc] looked at after the verdicts, [inference] interpretation.

## Question

Earlier work (B-1, Task A) measured how far wrong XRR fits are from the truth in counts: S₁ = S / ΔD,
the count level at which a wrong structure's curve differs from the true one by one unit of Poisson
deviance. Those numbers were taken where the optimiser stopped, and R2a's optimiser stopped at a
200-evaluation cap in more than half of the cases. The question: does S₁ at a returned endpoint stand
for the basin the endpoint sits in, or would continuing the local search change it (or reach the
truth)?

## Hypotheses [registered, `PREREG-BASIN.md`, 51b048a]

- **H-BM1 (primary).** For confirmed minima, κ = ΔD(endpoint) / ΔD(minimum) ≤ 2 and not the truth for
  at least 80 %, with at most 5 % misses (truth reached, or κ > 10). Refuted at ≥ 20 % misses.
- **Q-BM2 (count).** How many distinct wrong basins have S₁ ≥ 10⁴ counts at their minimum, that is,
  come within one deviance unit of the truth only at 10⁴ counts or more.
- **H-BM3.** At the minima, B-1's descriptions still hold for ≥ 80 % of distinct basins: (a) the Fisher
  quadratic overstates the deviance tenfold or more, (b) ≥ 90 % of the deviance sits at
  q ≤ 0.235 Å⁻¹, (c) the largest share is in the lowest three q blocks.

## Protocol

- Candidates: 749 wrong R2a endpoints (4 < ΔD ≤ 10⁵, 37 simulated conditions: single film or
  bilayer, 2θ_max 2.2°, 3°, 6°, film SLD 12, 24, 45, two nuisance sets) and 93 X2e states with the
  true layer count; X2e also refined on its own training objective (descriptive). 935 refinements.
- From each recorded endpoint only (no new starts): trust-region least squares on Poisson residuals
  against the truth's noise-free counts, inside the original parameter box, in stages of 200
  evaluations, at most 3000 evaluations.
- Convergence must be shown, not assumed: gradient and Newton decrement below tolerance, bound
  conditions, no descent along flat directions, and one more stage that does not move. Otherwise
  "unconfirmed", never counted as a minimum.
- Minima whose predicted curves are within ΔD = 1 of each other count as one basin.
- Verdict code committed before the run; pilot candidates re-run first and matched bit for bit.

## Results [fact]

- 656 of 749 R2a and 92 of 93 X2e candidates reached a confirmed minimum; 76 R2a stopped at the
  budget, 17 stalled, none failed numerically.
- Continuing the search barely changed the deviance: median κ 1.000 (R2a) and 1.06 (X2e), 90th
  percentile 1.01 and 1.29. The parameters moved more: 10 % of R2a minima moved more than 0.66 of a
  box width.
- Only one R2a endpoint (a capped one) turned out to be a stalled search that ends at the truth; one
  X2e state improved its deviance 12.8-fold but stayed wrong.
- 655 wrong R2a minima are 399 distinct basins; X2e 92 → 89.
- 651 of 656 R2a minima sit on at least one bound of the box, mostly nuisance parameters (Kα2 ratio,
  oxide, background, roughness).
- Wrong basins needing ≥ 10⁴ counts: five in R2a, all the same structure [post hoc] — the 10 Å oxide
  removed (thickness at its 0 bound) and the film 8 Å thicker, Kα2 ratio 0 — needing 2.7 × 10⁴ to
  2.3 × 10⁵ counts; one in X2e, a local minimum within 0.2 Å of the true two-layer structure, needing
  2.1 × 10⁴ counts.
- The Fisher quadratic overstates a wrong basin's deviance by a median 10² (R2a) and 10³ (X2e).
- Cost: 50.9 CPU h, 12.9 h on four processes, 614 000 evaluations, 0.73 GB per process. 82 % of the
  time went into the convergence tests, which the pilot's budget had underestimated (worst case 12 h
  quoted; corrected in `RESULTS-BASIN.md` section 9).

## Verdicts [registered]

| hypothesis | R2a | X2e |
|---|---|---|
| H-BM1 endpoint S₁ represents its basin | **supported** (98.3 % κ ≤ 2, misses 0.2 %) | **supported** (95.7 %, misses 1.1 %) |
| Q-BM2 wrong basins with S₁ ≥ 10⁴ | **5** | **1** |
| H-BM3 descriptions persist | **persists** (a 99 %, b 98 %, c 98 %) | **unresolved** (a 96 %, b 77.5 %, c 81 %) |

What this means [inference]: S₁ maps built from returned endpoints describe the basins those
endpoints sit in, and R2a's evaluation cap did not distort them. The wrong basins are mostly minima
of the prior box rather than of the likelihood, so their S₁ belongs to that box. Within these
records, a wrong description can come within one deviance unit of the truth at counts an experiment
reaches (10⁴–10⁵), in R2a only at the box edge.

## Limitations

- Local only: no statement on how many basins the landscape has, multimodality or global uniqueness;
  the basins counted are those reached from recorded starts.
- Noise-free objective; measured or noisy data not tested (except X2e's own training objective).
- Box dependence: 651 of 656 R2a minima are held by bounds; a different box could move or remove them.
- Interior-only analysis not testable (5 R2a, 16 X2e interior minima).
- Earlier verdicts are untouched: X2e fail, H-F unresolved, H-W2 supported, H-cal2 supported, the
  generative no-go.

[Back to all reports](../../reports/README.md)
