# Basin minimum validation: full run, results and verdicts

xrr-policy lane, 2026-10-25. Protocol: `PREREG-BASIN.md` (ReflectoLearn `research/fisher-global`
51b048a, committed before any candidate was refined). Owner approval of the full run:
`docs/isaac-directive-20261025-xrr-basin-full-run.md` in the NaNIsaac workspace. A copy of this file is
`docs/xrr/GEN-PHASE1/RESULTS-BASIN.md` in the workspace; the research-note version is `BASIN-NOTE.md`
there.

| record | |
|---|---|
| type | Experiment Result (pre-registered verdicts) |
| protocol | `PREREG-BASIN.md` sections 3–5, N_max = 3000, criteria unchanged |
| verdict code | `experiments/005_fisher_global/basin_report.py`, committed before the run (6e42e32) |
| run code | `basin.py --set full` at 6e42e32; pilot rows reproduced bit for bit (ff86ec2) |
| raw output | `docs/fisher_global/basin_full/` (935 files, one per candidate and objective) at db4cbaa |
| generated | `basin_full.md` (tables), `basin_verdict.json` (verdicts, basins), `basin_extra.json` (post hoc) |
| verdicts | H-BM1 **supported** (R2a, X2e); Q-BM2 **5** (R2a), **1** (X2e); H-BM3 **persists** (R2a), **unresolved** (X2e) |
| earlier verdicts | unchanged: X2e fail, H-F unresolved, H-W2 supported, H-cal2 supported, generative no-go |

Labels: **[registered]** a rule or verdict from the registration; **[fact]** read or computed from the
records; **[post hoc]** computed after the verdicts, no verdict depends on it; **[inference]**
interpretation. Numbers in this file come from `basin_full.md`, `basin_verdict.json` and
`basin_extra.json` unless stated.

## 0. What was run

- [fact] 935 refinements: R2a O_nf 749, X2e O_nf 93, X2e O_tr 93, each from its recorded endpoint (no
  new start), under the registered optimiser, stages, budget and convergence tests C1–C4.
- [fact] Before the run: the verdict and duplicate-grouping code was committed (6e42e32) with six
  choices the registration leaves open fixed in `OPEN_CHOICES` (section 8). The eight pilot candidates
  (ten refinements) were re-run with the full-run code under four workers and matched the pilot rows
  exactly: outcomes, evaluations, every stage, the minima and their counts
  (`basin_pilot_repro.md`). Those ten files are part of the 935.
- [fact] Four processes, one torch and one BLAS thread each, BelowNormal, wrapper with a 4 GB / 16 h
  stop. 03:53:58–16:49:33, 46 535 s wall, no stop, no error, no lock left, start check passed for all
  935 (relative difference ≤ 10⁻⁶).
- [fact] No R2a, X2e, B-1 or Task A record was written. No PR.

## 1. Confirmed / unconfirmed / budget-limited [fact]

| set | n | confirmed | interior | on bound | budget-limited | stalled | numerical |
|---|---|---|---|---|---|---|---|
| R2a O_nf | 749 | 656 (87.6 %) | 5 | 651 | 76 (10.1 %) | 17 (2.3 %) | 0 |
| X2e O_nf | 93 | 92 (98.9 %) | 16 | 76 | 1 | 0 | 0 |
| X2e O_tr | 93 | 91 (97.8 %) | 11 | 80 | 1 | 1 | 0 |

- Budget-limited R2a by condition: largest shares in #17 (5 of 9), #10 (10 of 43), #13 (8 of 41),
  #1 (7 of 33); by endpoint, capped 48 of 403, not capped 28 of 346.
- Unconfirmed candidates (96 in all): κ at their last point 0.99998–6.49, median 1.0001; none above
  10, so the registered sensitivity adds nothing. Stalled R2a candidates sit at ΔD_nf 3.5 × 10⁴ –
  9.9 × 10⁴; several of #7 stall at the same value (38 396.44), which suggests one flat region
  [inference].

## 2. Basins before and after removing duplicates [fact]

Rule (registered, section 4): within one data set, two confirmed wrong-basin minima are one basin if
max(ΔD(λ_a | λ_b), ΔD(λ_b | λ_a)) ≤ 1; connected components; each represented by its lowest ΔD\*_nf.

| source | confirmed wrong minima | distinct basins | endpoints per basin, median / p90 / max | max u inside a basin, p90 / max | curve-equivalent pairs > 10⁻² apart in u |
|---|---|---|---|---|---|
| R2a | 655 | 399 | 1 / 3 / 11 | 1.7 × 10⁻⁵ / 1.0 | 1 |
| X2e | 92 | 89 | 1 / 1 / 2 | 1.3 × 10⁻⁶ | 0 |

- The one far pair (r2a:06:19 / r2a:06:50, #6) gives the same curve within ΔD = 1 with the Kα2 ratio at
  opposite bounds (0.6 and 0); the other parameters agree within 0.01 u [post hoc]. A parameter that
  can cross its whole box without changing the curve at that point.
- [fact] H-BM1 restricted to basin representatives (descriptive): κ ≤ 2 for 98.5 % (R2a) and 95.5 %
  (X2e), the same as on endpoints.
- [inference] R2a's 33 conditions hold about 12 distinct wrong basins each among the recorded
  endpoints; up to 11 endpoints of one condition land in one basin. These are counts of basins reached
  from R2a's starts, not of basins in the landscape (section 9).

## 3. Endpoint → minimum: deviance and S₁ [fact]

Confirmed O_nf minima (quantiles min / p10 / median / p90 / max):

| quantity | R2a (656) | X2e (92) |
|---|---|---|
| κ = ΔD_nf(end) / ΔD\*_nf | 1 / 1 / 1.000 / 1.01 / 9 × 10¹⁴ (truth basin; wrong basins max 6.5) | 1 / 1.01 / 1.06 / 1.29 / 12.8 |
| ΔD_nf drop 1 − 1/κ (wrong basins) | 0 / 0 / 6 × 10⁻⁷ / 0.0066 / 0.85 | 0.0004 / 0.006 / 0.058 / 0.23 / 0.92 |
| S₁ at endpoint | 10 / 11.5 / 26.3 / 106 / 2.3 × 10⁵ | 1.0 / 4.7 / 39 / 374 / 1.8 × 10⁴ |
| S₁\* at minimum (wrong basins) | 10 / 11.6 / 26.7 / 106 / 2.3 × 10⁵ | 1.4 / 4.8 / 44 / 392 / 2.1 × 10⁴ |
| drift from endpoint (u, max norm) | 10⁻¹¹ / 4 × 10⁻⁷ / 0.005 / 0.66 / 1 | 0.009 / 0.02 / 0.10 / 0.62 / 0.87 |
| evaluations to confirmation | 4 / 6 / 136 / 1720 / 3000 | 9 / 12 / 30 / 119 / 1050 |

- 39 of the 841 O_nf rows have κ between 1 − 10⁻⁹ and 1: the deviance used for ΔD_nf (series form
  near r = 0) and TRF's objective differ in the last digits. Not a movement uphill [fact: TRF accepts
  only decreasing steps of its own objective].
- [fact] Two misses in the H-BM1 sense:
  - r2a:01:59 (#1, 2.2°, ρ 24, single, refine; endpoint capped): ΔD_nf 5.9 × 10⁴ → 6.6 × 10⁻¹¹ in
    139 evaluations; film 246.6 Å / σ 1.35 / ρ 23.44 → 240 / 6 / 24, the truth. A stalled search, not
    a wrong basin.
  - x2e:D2:2:13:0:0: endpoint with d₁ at its 5 Å bound and σ₂ at 15 Å; ΔD_nf 2.9 × 10⁵ → 2.2 × 10⁴
    (κ 12.8), still wrong (d₁ 38.8 Å vs 34.2, ρ₁ 66 vs 51.6) [post hoc].
- [fact] O_tr (X2e, descriptive): 91 of 93 confirmed, κ under O_tr 0.994–1.03, drift median
  1.6 × 10⁻⁶: the returned X2e states were already minima of the objective they were returned under.
  The O_tr and O_nf minima of the same state are 0.009–0.87 apart in u (median 0.09).

## 4. Parameters on a bound [fact]

- R2a: 651 of 656 minima have at least one parameter on a bound (median 4, max 8). Counts at lower /
  upper bound: Kα2 ratio 387 / 163, film roughness 164 / 243, oxide thickness 31 / 320, oxide
  roughness 168 / 174, oxide SLD 163 / 98, background 188 / 65, second-film roughness 28 / 88, 2θ
  offset 0 / 57, film thickness 32 / 1, second-film thickness 11 / 1, film SLD 10 / 0, second-film
  SLD 10 / 0, dq 6 / 0.
- X2e: 76 of 92 minima on a bound (median 1, max 4): background at its lower bound 44, offset at its
  upper bound 23, σ₁ 2 / 13, σ₂ 10 / 13, σ₃ 5 / 5, ρ₂ 8 / 0, ρ₃ 8 / 0, dq 6 / 0, d₂ 5 / 1, d₁ 1 / 2,
  ρ₁ 1 / 1.
- [registered] Minima on a bound are included in every verdict. Interior-only, descriptive: R2a has 5
  interior minima (not testable; 4 with κ ≤ 2, the fifth is the truth-basin miss), X2e 16 (not
  testable; 14 with κ ≤ 2, one miss).
- [inference] For R2a the wrong basins are minima of R2a's prior box, not of the likelihood: almost
  every one is held by nuisance parameters at their limits (Kα2 ratio, oxide, background, roughness).
  S₁\* of such a basin depends on where the box was drawn.

## 5. Fisher approximation against the actual deviance [fact]

ρ = ΔD_F / ΔD_nf with F the Fisher information at the truth (registered, section 4). Confirmed O_nf
minima:

| quantity | R2a | X2e |
|---|---|---|
| ρ at endpoint, median [p10, p90] | 115 [37, 930] | 1.5 × 10³ [72, 1.0 × 10⁴] |
| ρ\* at minimum, median [p10, p90] | 115 [36, 955] | 1.9 × 10³ [95, 1.0 × 10⁴] |
| ρ\* / ρ(end), median [p10, p90] | 1.00 [0.99, 1.02] | 1.07 [1.00, 1.44] |
| log₁₀ ρ\* at wrong minima, median [min, max] | 2.06 [0.73, 3.56] | 3.28 [0.43, 5.57] |

- The quadratic Fisher form overstates the deviance of a wrong basin by a factor of about 10² (R2a)
  and 10³ (X2e) at the median, at the refined minima as at the endpoints. H-BM3 (a) (section 7).
- [fact] Smallest ρ\*: R2a 0.999 (one minimum where the quadratic is exact to the digit), X2e 2.7.

## 6. By q_max, layer count and contrast [fact]

R2a, confirmed O_nf (full tables by condition in `basin_full.md`):

| group | n | confirmed | budget | κ > 10 wrong | κ p90 | S₁ end median | S₁\* wrong median / max | distinct basins |
|---|---|---|---|---|---|---|---|---|
| 2θ_max 2.2° | 258 | 229 | 25 | 0 | 1.01 | 26.6 | 27.2 / 679 | 140 |
| 2θ_max 3° | 251 | 222 | 23 | 0 | 1.01 | 27.1 | 27.2 / 2.3 × 10⁵ | 118 |
| 2θ_max 6° | 240 | 205 | 28 | 0 | 1.00 | 25.8 | 26.0 / 8.6 × 10⁴ | 141 |
| single film | 505 | 438 | 51 | 0 | 1.01 | 21.2 | 21.2 / 2.3 × 10⁵ | 249 |
| bilayer | 244 | 218 | 25 | 0 | 1.02 | 29.3 | 29.9 / 2.7 × 10⁴ | 150 |
| film ρ 12 | 406 | 368 | 29 | 0 | 1.01 | 13.6 | 13.8 / 795 | 218 |
| film ρ 24 | 286 | 236 | 42 | 0 | 1.00 | 28.1 | 28.1 / 2.3 × 10⁵ | 170 |
| film ρ 45 | 57 | 52 | 5 | 0 | 1.00 | 170 | 170 / 565 | 11 |
| refine nuisances | 364 | 321 | 31 | 0 | 1.01 | 23.3 | 25.0 / 2.3 × 10⁵ | 189 |
| wide nuisances | 385 | 335 | 45 | 0 | 1.01 | 27.4 | 28.0 / 2.3 × 10⁵ | 210 |

X2e (one 291-point grid, so no q_max split; contrast varies inside each truth class):

| group | n | confirmed | κ > 10 | κ median / p90 | S₁ end median | S₁\* wrong median / max | distinct basins |
|---|---|---|---|---|---|---|---|
| two layers | 39 | 39 | 1 | 1.03 / 1.19 | 78.5 | 80 / 2.1 × 10⁴ | 37 |
| three layers | 54 | 53 | 0 | 1.08 / 1.34 | 23.2 | 35 / 632 | 52 |
| s = 0 (10⁸ counts) | 42 | 42 | 1 | 1.06 / 1.29 | 35.4 | 39 / 4.5 × 10³ | 41 |
| s = 1 (10⁶ counts) | 51 | 50 | 0 | 1.05 / 1.28 | 41.7 | 46 / 2.1 × 10⁴ | 48 |

- [inference] κ does not depend on q_max, layer count, contrast or nuisance set in any visible way;
  S₁ does, mainly through contrast: R2a ρ 45 basins are few (11) and need about ten times more counts
  than ρ 12 ones. Every S₁\* ≥ 10⁴ basin in R2a has film ρ 24 (section 7).

## 7. Verdicts by the registered rules

### H-BM1: endpoint S₁ represents its basin — R2a **supported**, X2e **supported** [registered]

| source | E (confirmed) | of registered | κ ≤ 2, not truth basin | misses | verdict |
|---|---|---|---|---|---|
| R2a | 656 | 749 | 645 (98.3 %) | 1 (0.2 %; truth basin 1) | **supported** |
| X2e | 92 | 93 | 88 (95.7 %) | 1 (1.1 %; κ > 10 1) | **supported** |

Rule: supported if κ ≤ 2 and not truth basin for ≥ 80 % of E and misses ≤ 5 % of E. Beside the
verdict (not part of it): sensitivity with unconfirmed κ > 10 added — supported for both (nothing
added); without the eight pilot candidates — supported for both (E 650 and 90, the same fractions).

### Q-BM2: wrong basins with S₁\* ≥ 10⁴ — R2a **5**, X2e **1** [registered count]

| source | basin (representative) | condition | ΔD_nf end → min | S₁\* | on bound |
|---|---|---|---|---|---|
| R2a | r2a:04:36 | #4, 3°, single, ρ 24, refine | 4.28 → 4.28 | 2.3 × 10⁵ | film σ, Kα2, oxide thickness |
| R2a | r2a:13:36 | #13, 3°, single, ρ 24, wide | 4.28 → 4.28 | 2.3 × 10⁵ | film σ, Kα2, oxide thickness |
| R2a | r2a:16:36 | #16, 6°, single, ρ 24, wide | 11.7 → 11.7 | 8.6 × 10⁴ | Kα2, oxide thickness |
| R2a | r2a:07:36 | #7, 6°, single, ρ 24, refine | 16.5 → 16.5 | 6.1 × 10⁴ | film σ, Kα2, oxide thickness, oxide SLD |
| R2a | r2a:34:56 | #34, 6°, bilayer, ρ 24, wide | 36.4 → 36.4 | 2.7 × 10⁴ | Kα2, oxide thickness |
| X2e | x2e:S1:2:11:1:1 | two layers, s = 1 | 54.8 → 48.5 | 2.1 × 10⁴ | none |

[post hoc, `basin_extra.json`] What these basins are:
- The five R2a basins are one structure: the 10 Å oxide is gone (oxide thickness at its 0 Å bound) and
  the film is 8 Å thicker (248.0–248.4 Å for 240 Å), film SLD within 0.15 of the truth, the Kα2
  ratio at 0, film roughness at or near 0 (truth 6 Å), oxide roughness 6.9–7.8 Å (truth 3). In #34 the
  second film is right (120.03 Å, ρ 29.89). Each is its own basin (one endpoint each); #4 and #13
  share truth and grid and differ only in nuisance bounds, and reach the same minimum; #7 and #16
  (6°) differ (ΔD_nf 16.5 and 11.7; film roughness 0 and 1.6 Å).
- The X2e basin sits next to the truth: every layer parameter within 0.2 Å or 0.1 SLD of it, interior,
  ρ\* 13; 81 % of its ΔD_nf comes from q below 2θ = 0.5°.
- [inference] Read by the registered meaning, within these records a wrong basin comes within ΔD = 1 of
  the truth at 10⁴ counts or more. The R2a cases are a physically distinct description (oxide merged
  into the film) that the box allows only at its edge (oxide thickness 0, Kα2 0). The X2e case is a
  nearby local minimum of the same structure rather than a different one. Neither is evidence about
  conditions outside these records.

### H-BM3: B-1's descriptions persist at the minima — R2a **persists**, X2e **unresolved** [registered]

| statement | R2a | X2e |
|---|---|---|
| (a) ρ\* ≥ 10 | holds, 396 / 399 (99.2 %) | holds, 85 / 89 (95.5 %) |
| (b) shares low + b0…b4 ≥ 0.9 (291-point grids) | holds, 138 / 141 (97.9 %) | **unresolved**, 69 / 89 (77.5 %) |
| (c) largest region b0, b1 or b2 (291-point grids) | holds, 138 / 141 (97.9 %) | holds, 72 / 89 (80.9 %) |
| overall | **persists** | **unresolved** |

Rule: each statement holds at ≥ 80 %, fails below 60 %, unresolved between; persists if all hold. The
three R2a basins outside (b) are the three 291-grid Q-BM2 basins (#7, #16, #34; low-q share 0.44–0.55)
[post hoc]. X2e (b) at 77.5 % is inside the unresolved band (60–80 %).

## 8. Choices fixed before the run (OPEN_CHOICES, 6e42e32)

1. H-BM1 "half the source's candidates" is half the registered count (749, 93).
2. The sensitivity adds every unconfirmed O_nf candidate with κ > 10 at its last point, whatever the
   outcome (none qualified).
3. Hypotheses use O_nf only; O_tr is descriptive.
4. H-BM3 with a not-testable statement and no failure is "unresolved" (did not occur).
5. Distance in u between two minima uses the mean of their bound widths.
6. A run that raised an exception counts as unconfirmed (none occurred).

None of these decided a verdict [fact]: choices 1, 2, 4 and 6 had no case to act on (no candidate
failed the start check, none raised, no unconfirmed candidate had κ > 10, no H-BM3 statement was not
testable); choice 5 enters only the reported spreads and far pairs; choice 3 restates registration
section 3.1.

## 9. Compute time, forward evaluations, memory [fact]

| set | n | CPU h | s per refinement, median / p90 / max | evaluations | model calls | value calls | Hessians | peak MB per process |
|---|---|---|---|---|---|---|---|---|
| R2a O_nf | 749 | 50.12 | 26.5 / 399 / 8770 | 596 231 | 636 263 | 5 567 | 52 011 | 732 |
| X2e O_nf | 93 | 0.38 | 4.9 / 13.6 / 733 | 8 243 | 8 730 | 109 | 674 | 707 |
| X2e O_tr | 93 | 0.37 | 1.7 / 15.2 / 810 | 9 456 | 9 994 | 95 | 659 | 707 |
| all | 935 | **50.87** | | **613 930** | 654 987 | 5 771 | 53 344 | 2.87 GB (four-process tree) |

- Wall 46 535 s (12.9 h) at four processes; pilot reproduction 210 s before it.
- Where the time went: TRF stages (evaluations and Jacobians) 9.17 CPU h, convergence tests 41.7 CPU h
  (82 %). The longest refinement, r2a:24:40 (#24, 6°, bilayer), spent 3000 evaluations in 1280
  stages and computed 1281 Hessians, 8770 s. A Hessian of a 6° bilayer costs 6.5–7.2 s; 2.2° single
  about 1.1 s.

**Correction to the pilot's budget (RESULTS-BASIN-PILOT.md section 5).** The pilot gave 9.8 CPU h at
the pilot mean and 47 CPU h as the worst case (2.5 h and 12 h at four processes). The run took
50.9 CPU h and 12.9 h. The evaluation part came out as estimated (9.2 CPU h), but the worst case
counted the convergence tests as at most 28 s per candidate, the largest seen in the pilot. That bound
was wrong: the registration counts N_max in evaluations only and runs C1–C3 at the end of every stage
where TRF stops by its own tolerance; along flat directions TRF stops every two or so evaluations, so
the number of Hessians per candidate is limited only by the number of stages (at most N_max; up to
0.49 N_max seen, r2a:13:32). A worst case of N_max / 2 Hessians per candidate would have been about 3 h for a 6°
bilayer candidate, against the 28 s assumed. The run itself stayed inside the approved four processes and 4 GB; it needed one window,
and no candidate was lost or re-run. The code was not changed: running the tests less often would
change when a candidate counts as confirmed, and so the result.

## 10. What can be claimed and what is not verified

Claimable within the registered scope (R2a's 37 scan conditions and X2e's records, noise-free O_nf,
bounded boxes, local refinement from recorded endpoints):

1. [registered] S₁ computed at the returned endpoints (B-1, Task A) represents S₁ at the local minimum
   of the basin each endpoint sits in: κ ≤ 2 for 98 % (R2a) and 96 % (X2e) of the confirmed candidates,
   misses 0.2 % and 1.1 %. R2a's 200-evaluation cap did not distort the S₁ map: κ p90 is 1.02 for
   capped and 1.00 for uncapped endpoints, and only one capped endpoint (of 339 confirmed) was a
   stalled search that ends in the truth.
2. [registered] At the refined minima the Fisher quadratic overstates a wrong basin's deviance by a
   median 10² (R2a) to 10³ (X2e); ρ\* ≥ 10 for 99 % and 96 % of distinct basins.
3. [registered] For R2a the low-q concentration of a wrong basin's deviance (q ≤ 0.235 Å⁻¹, largest
   share in b0–b2) persists at the minima (H-BM3 persists). For X2e it does not reach the registered
   threshold for (b) (unresolved), so it is not claimed there.
4. [fact] Wrong basins reached from R2a's starts are box-limited: 651 of 656 minima rest on 1–8
   bounds, mostly nuisance parameters. Statements about their S₁ are statements about this box.
5. [registered count + post hoc] In both sources at least one wrong basin needs only 10⁴–2 × 10⁵
   counts to reach ΔD = 1: in R2a an "oxide merged into the film" structure at the box edge, in X2e
   a nearby minimum of the true structure.
6. [fact] The returned X2e states were minima of the training objective they were returned under
   (O_tr: 91 of 93 confirmed, κ ≈ 1).

Not verified:

- Anything global: the number of basins in the landscape, multimodality, global uniqueness, basins no
  recorded endpoint reached (registered "not concluded"). 399 and 89 count distinct basins reached
  from recorded starts.
- Whether the box-edge minima survive a wider or physically motivated box (negative oxide thickness is
  not physical, but the Kα2 ratio and roughness limits are choices).
- The interior-only version of H-BM1 (5 R2a and 16 X2e interior minima; not testable).
- Behaviour under noisy data or measured scans: O_nf is noise-free; the noisy-data objective was
  checked only for X2e's own returned states (O_tr).
- Whether S₁\* ≥ 10⁴ basins exist in conditions outside these records (Q-BM2 says "within these
  records").
- The physical meaning of the X2e Q-BM2 basin (low-q discrepancy below 2θ = 0.5°, where footprint and
  offset act); not examined.
- The 93 unconfirmed candidates of R2a are not minima; their κ is at most 6.5, so they do not change
  H-BM1 under the registered sensitivity, but their basins are not counted.

Earlier verdicts stay as they are: X2e fail, H-F unresolved, H-W2 supported, H-cal2 supported, the
generative no-go. Nothing here is a reason to revisit them; H-BM3 tests B-1's descriptions on new
points and does not change H-F or H-W2.
