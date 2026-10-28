# Experiment 006: what log-type objectives cost an XRR fit (registration)

[Back to all experiments](../research/README.md)

> **Question.** How much information, bias and interval calibration do log-type XRR objectives cost against the Poisson likelihood, across film contrast, q range and counting level, and which part of the objective causes it?
>
> **Result.** Registered, not run. A 15-draw pilot on one cell was run before this registration and is reported below.
>
> **Limits.** Synthetic curves from the model that fits them (no model error) except H6; one film thickness; fits start at the truth, so search failures are excluded by design.

Registered 2026-10-28, before the main run. Changes after this commit are made in new commits that give the reason; this text is not edited. Physics readings to be checked by the owner are marked **[owner]**.

## Background and prior work (step 0, read 2026-10-28)

What the earlier experiments found:
- Experiment 001 found that refine's log-MSE disagrees with the likelihood about which fit is better (5 of 48 endpoints, R1c).
- R2b needed the Poisson objective as one of four changes.
- The candidate question is in [`generative_go_nogo.md`](../identifiability/generative_go_nogo.md) (question 2).

A one-off check on 2026-10-28, outside this repository, gave on the mid film:
- log-MSE efficiency 0.03–0.11 for d and below 0.01 for ρ;
- an SLD bias of 0.4–8 % at 10⁴–10⁷ counts;
- Laplace-interval coverage of 0–0.67.

What others do and have shown ([V] metadata checked, [M] from memory; contents as summarised from the sources):
- refnx recommends, for X-ray data, a `logY` transform with weighting switched off (FAQ, refnx.readthedocs.io): the uniform log objective L0 below. GenX's default figure of merit is also log-based, per papers that use it; its documentation of the options was not found. So L0 is the common practice, not a ReflectoLearn quirk.
- Haberl, Eder, Pfeiler, Schumi-Mareček, Bertram, Mikulík & Kowarik, *Microsecond X-ray reflectometry*, J. Appl. Cryst. 59, 1112–1120 (2026), doi:10.1107/S1600576726004449 [V]. On measured fast scans (0–2 photons per point), it compares untransformed weighted fits, a transformed weighted fit and an Anscombe-transformed fit:
  - bias in counts of about −1 for untransformed data, and +0.5 for transformed data, rising to about +1 when zero counts are excluded;
  - the Anscombe transform with a bias correction gives the closest curve;
  - thickness and SLD are consistent across the three over 100 scans;
  - roughness is shifted under the other transform;
  - uncertainties are from emcee.

  It does not compare against Poisson maximum likelihood or the Cramér–Rao bound, does not map bias over contrast, q range or counts, and does not check interval coverage.
- Lass et al. (2021) is cited there for Poisson log-likelihoods in reflectometry; not found and not read.
- Humphrey, Liu & Buote, ApJ 693, 822 (2009), doi:10.1088/0004-637X/693/1/822 [V]: χ² approximations to Poisson data bias parameters even at high counts; they recommend the Cash statistic (Cash, ApJ 228, 939 (1979) [M]). The general effect is known in X-ray astronomy.
- Durant et al. (2021), arXiv:2103.08973 [V]: Fisher information and Cramér–Rao bounds for neutron reflectometry (local, Poisson); no objective comparison.
- McCluskey et al., J. Appl. Cryst. (2023), doi:10.1107/S1600576722011426 [V metadata]: advice on reporting Bayesian reflectometry analyses; the Gaussian likelihood is the common assumption.

[inference] Consequence for the claims: that χ²- and log-type objectives are biased for counts is known (astronomy; Haberl et al. for XRR curves). Not found for XRR:
- efficiency against the Cramér–Rao bound;
- parameter bias mapped over contrast, q range and counts;
- the split into weighting, log transform and masking;
- coverage of the intervals a log-likelihood gives;
- the link to identifiability (H5).

The claims below are written for those gaps. A fuller search (Lass et al.; the GenX figure-of-merit source) is a task before the report, not a condition of this registration.

## Setup

- Code: [`experiments/006_objective/objective.py`](../../experiments/006_objective/objective.py). Forward model: experiment 001's `fim.expected_counts` (Abelès in complex128, constant-dq Gaussian resolution, i0, background 10⁻⁷, 2θ offset); Si substrate; film d = 240 Å, σ = 6 Å; Kα2, footprint and oxide off (except H5).
- Grid, 39 cells (`objective.py list`):
  - 36 cells: film SLD {12, 24, 45} × 2θ_max {3°, 6°, 12°} × counts at R = 1 {10⁴, 10⁵, 10⁶, 10⁷}, with d, σ and ρ free (`film`);
  - 3 cells: the `instr` scenario (d, σ, ρ, i0, background, offset, dq free) at 2θ ≤ 6°, 10⁶ counts, one per SLD.
- 50 Poisson draws per cell. The generator is seeded with (20261028, cell index), so each cell's draws are fixed and the 15 pilot draws are the first 15 of their cell.
- Every fit starts at the truth with `scipy.optimize.least_squares` (LM, xtol = ftol = 10⁻¹²). Differences between objectives are therefore not search failures.
- Objectives:
  - **P:** Poisson deviance (control);
  - **L0:** Σ(ln y − ln λ)² over y > 0 (refine's log-MSE; refnx `logY` without weighting);
  - **L1:** Σ(ln(y + ½) − ln(λ + ½))² over all points (L0 without dropping zeros);
  - **L2:** Σ y (ln y − ln λ)² over y > 0 (L0 weighted by the counts, the delta-method Gaussian in ln y).
- Measures per cell and film parameter (d, ρ; σ reported, not tested):
  - efficiency E = CRLB² / sample variance (CRLB from the Poisson Fisher information at the truth);
  - bias of the mean in CRLB units and in % of the true value, with its t statistic;
  - for L0, the coverage of the ±1σ Laplace interval of a Gaussian ln R likelihood with a fitted spread, σ̂² (JᵀJ)⁻¹ with σ̂² the mean squared residual: the form `refine.mcmc` samples;
  - endpoints with `least_squares` status ≤ 0 are counted and reported; they stay in every measure.

## Claims and decision rules

Cell counts refer to the 36 `film` cells.

| | Claim | Supported if | Refuted if |
|---|---|---|---|
| H1 | L0 discards most of the information on ρ | E_ρ(L0) < 0.1 in ≥ 30 of 36 cells, and E(P) within [0.6, 1.6] for d and ρ in ≥ 33 cells | E_ρ(L0) ≥ 0.3 in ≥ 18 cells |
| H2 | L0 biases ρ beyond its own scatter | \|bias_ρ(L0)\| > 3 CRLB with \|t\| > 2.7 (p < 0.01) in ≥ 14 of the 27 cells at ≤ 10⁶ counts | \|bias_ρ(L0)\| ≤ 1 CRLB in ≥ 14 of those 27 cells |
| H3a | The efficiency loss comes from the weighting | E_ρ(L2) ≥ 0.5 in ≥ 75 % of the cells with E_ρ(L0) < 0.1 | E_ρ(L2) < 0.2 in ≥ 50 % of those cells |
| H3b | The bias does not come from dropping zero counts alone | \|bias_ρ(L1)\| ≥ \|bias_ρ(L0)\| / 3 in ≥ 50 % of the cells at ≤ 10⁵ counts where L0 is biased (H2 sense) | \|bias_ρ(L1)\| < \|bias_ρ(L0)\| / 10 in ≥ 75 % of those cells |
| H4 | The Gaussian ln R interval is miscalibrated where L0 is biased | ±1σ coverage < 0.5 in ≥ 75 % of the cells with \|bias_ρ(L0)\| > 1 CRLB | coverage in [0.55, 0.81] in ≥ 75 % of those cells |
| H5 | Named test (owner Q145): for the SLD-24 film with a 10 Å top oxide, L0 prefers the oxide-free sharp structure more often than P does | Two starts per draw: the truth, and the alternative recorded in R2a (r2a:04:36: 248 Å, σ 0, no oxide, Kα2 0). Refine's bounds, Kα2 and footprint on; 2θ ≤ 3° and 6°; 10⁴, 10⁵, 10⁶ counts; 50 draws. Supported if, at some (q range, counts), the share of draws whose better endpoint is the alternative is ≥ 0.2 higher under L0 than under P | the two shares differ by < 0.1 at every (q range, counts) |
| H6 | The L0–P difference is larger than the measurement scatter on real counts | On the 100 repeated public scans of H-cal2, the mean L0−P difference in ρ exceeds 3 standard errors of the scan-to-scan spread of P | it is within 1 standard error. Qualitative: no ground truth, one instrument |

**[owner]** Physics readings in these claims:
- H3 assumes the uniform weighting is what throws the information away: points at high q with a few counts get the same weight in ln y as points near the critical edge with 10⁵ counts.
- H3b assumes the bias is the log transform's own small-count bias (E[ln y] ≠ ln λ). The pilot below already speaks to both.
- H5 assumes the oxide-free structure is a physically distinct stack, not a stretched copy of the truth.

H3a and H3b are sub-claims of a mechanism. If both are refuted, the mechanism is reported as open.

## Pilot (run before this registration; seen by the author)

Cell 29 (SLD 45, 2θ ≤ 6°, 10⁵ counts, `film`), 15 draws, `objective.py pilot`, 6 threads. Output: [`pilot.json`](pilot.json).

| objective | s per fit | E_d | E_ρ | bias_d / CRLB | bias_ρ / CRLB |
|---|---|---|---|---|---|
| P | 0.36 | 1.44 | 0.88 | −0.16 | −0.31 |
| L0 | 0.56 | 0.075 | 0.0004 | +12.0 | +336 |
| L1 | 1.11 | 0.055 | 0.0025 | −1.2 | −58 |
| L2 | 0.32 | 1.55 | 0.89 | −0.36 | −0.79 |

The claims and rules above were drafted on 2026-10-28 before this pilot (the plan in the workspace's RL-B-PLAN). After the pilot, H3 was split into H3a and H3b with thresholds, and the H2 count rule was fixed to 14 of 27; no other rule changed. In this one cell, the pilot points to the weighting for both loss and bias (L2 is efficient and unbiased) and away from the masking alone (L1 keeps a large bias).

## Runs, cost and order

| Step | What | CPU | Gate |
|---|---|---|---|
| 1 | H1–H4: 39 cells × 4 objectives × 50 draws (`objective.py run --cells 0 … 38 --threads 6`) | ≈ 1.3–1.5 h from the pilot (2.4 s per draw for the four fits; `instr` cells slower), one process, 6 threads (half of 12 cores), < 1 GB | boss slot after this registration merges |
| 2 | H4 check: emcee on 2 cells (Poisson and Gaussian ln R, ≥ 50 τ); code added in a separate commit before it runs | ≈ 20–40 min | boss slot |
| 3 | H5: 12-parameter model with refine's bounds; code added in a separate commit before it runs | ≈ 1 h | boss slot |
| 4 | H6: 100 scans × 2 objectives on the H-cal2 file | ≈ 10 min | after steps 1–3 |
| 5 | `objective.py report` (added with step 1's results): tables, figures, verdict per claim | seconds | |

## Figures (in the report)

1. E_ρ and E_d of each objective over the grid (contrast × q range, one panel per counting level).
2. Bias of ρ in % over the grid, L0 against P.
3. Mechanism: E and bias for P, L0, L1 and L2 in four representative cells, with draw-to-draw errors.
4. Coverage of the L0 Laplace interval against bias / CRLB, every cell; the two emcee cells marked.
5. H5: share of draws won by the oxide-free structure against counts, L0 against P.

[Back to all experiments](../research/README.md)
