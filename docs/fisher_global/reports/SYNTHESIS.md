# Synthesis of B-1, B-2 and H-cal2: is "Fisher against global discrimination" a paper?

[Back to all reports](../../reports/README.md)

> **Question.** Taken together, do B-1, B-2 and H-cal2 make 'Fisher against global discrimination' a paper contribution?
>
> **Result.** Registered outcomes: H-F unresolved (0.831, needs 0.90), H-W2 supported, H-cal2 supported under amendment 1, B-2 an observation (d₁ ridge 2.24 %). Judgement 'conditional yes, reframed': not sufficient alone, but a section of a methods paper on physical identifiability against computational reach measured in counts (S₁); the generative no-go stands.
>
> **Limits.** No global search, sampler or new run; the general claim that Fisher information is local is not new; H-cal2 covers one instrument; evidence still missing includes basin minima, a regime with S₁ near realistic counts, unreached bilayer basins, and noisy and measured data.

xrr-policy lane, 2026-10-25. Owner approval Q16 (`docs/isaac-directive-20261025-xrr-q16.md`). Inputs:
`PREREG-HF-HW2.md`, `RESULTS-B1.md`, `RESULTS-B2.md`, `RESULTS-HCAL2.md`, and RESEARCH-DESIGN.md
(Task A on X2e, post hoc). Code and outputs: ReflectoLearn `research/fisher-global` d1f164c … 92d1983,
pushed, no PR. Labels as in the result files: [fact], [registered], [X2e post hoc], [inference],
[lit] (with [V] metadata verified / [M] content from memory, as in LIT-SURVEY.md).

## 1. What was run, in total

- B-1: forward evaluations at 2368 recorded R2a endpoints and 37 truths, 31 s CPU. No fit.
- B-2: one truth-start fit and four Δχ² = 1 profiles on R2a #25, 451 s, one process. Memory not
  measured (wrapper error, RESULTS-B2 1).
- H-cal2: one 17.4 MB public file, no model.
- Not run: any global search, any sampler, any new X2e or R2a run. No earlier record was changed.
  X2e stays "fail"; R2a's labels and the generative no-go stand.

## 2. Registered outcomes

| test | outcome | the number that decided it |
|---|---|---|
| H-F (i) Fisher ratio in [0.5, 2] inside Δχ² ≤ 4 | **unresolved** | 0.831 (needs 0.90; fails below 0.60) |
| H-F (ii) Fisher ratio ≥ 10 at wrong endpoints | holds | 0.981 |
| **H-F** | **unresolved** | (i) |
| H-W2 (a) ≥ 90 % of ΔD at q ≤ 0.235 | holds | 0.996 |
| H-W2 (b) largest share in b0–b2 | holds | 0.953 |
| **H-W2** | **supported** | |
| B-2 (observation) | d₁ ridge (2.24 %), ρ₁, d₂, ρ₂ identifiable (0.1–0.3 %) | |
| **H-cal2** | **supported** (under amendment 1; not testable under the literal pre-check 3) | φ > 1.2 at 64 % of points |

Failed or weaker-than-registered parts, stated plainly: H-F did not reach "supported"; X2e's block-2
concentration did not reproduce in R2a (b0 dominates instead); H-cal2 needed an amendment after the
pre-checks; B-2's memory was not measured.

## 3. What the three results say together

### 3.1 Facts across X2e and R2a

1. **No wrong structure is close to the truth at realistic counts.** [fact] R2a's 2191 wrong
   endpoints would be indistinguishable from the truth only at S₁ of 0.25–26 counts (condition
   medians); the closest 87 at S₁ ≥ 100 are still far from 10⁶. [X2e post hoc] X2e's far states: S₁
   cell medians 33–185 counts. [fact, RESEARCH-DESIGN 2.1] #25's best endpoints: S₁ ≈ 560.
2. **The quadratic at the truth is wrong in both directions.**
   - Far side [fact, B-1]: at wrong endpoints Fisher overstates the deviance by a median factor of
     ~10² (10³ in X2e, post hoc); the ratio is ≥ 10 for 98 % of them and Fisher puts none of them
     inside Δχ² ≤ 4.
   - Near side [fact, B-1]: inside Δχ² ≤ 4 it is exact for converged points, but 17 % of near
     endpoints lie on curved nuisance valleys where it overstates the deviance by 3 to 3 × 10⁴
     (decision-level disagreement 3 of 177).
   - Local precision [fact, B-2]: for #25 the unbounded Fisher half-widths are 1.2–2.3 times the
     bounded profile half-widths, because F's weakest direction runs through Kα2 ratios outside
     [0, 0.6] (−1.3 to 3.1), where the real deviance is 356–4 × 10⁴, not 1.
3. **Where the discrimination sits.** [fact, B-1] For R2a's wrong endpoints, ≥ 90 % of the deviance
   lies at q ≤ 0.235 Å⁻¹ in 99.6 %, with the first fringe block (q 0.04–0.075) the largest region in
   95 %. [X2e post hoc] X2e's far states peaked at q 0.12–0.16 instead. Only the closest R2a
   subset (ΔD ≤ 10⁴, n = 23) spreads to higher blocks.
4. **The counts of one real instrument are not Poisson.** [fact, H-cal2] Raw counts of 100 repeated
   9.95 ms synchrotron scans: median φ = s²/x̄ = 1.34, rising from 1.16 (10–100 counts) to 1.62
   (100–1000 counts); dividing out the scan-to-scan intensity leaves 1.31.

### 3.2 Interpretation [inference]

- **Physical identifiability is not what fails in these classes; reach is.** Every wrong state we
  have (X2e far states, R2a endpoints, #25's best) is separated from the truth by 10³–10⁷ deviance
  units at the data's counts. Overdispersion of φ ≈ 1.3–1.6, if lab data had it, would shrink those
  separations by the same factor and leave the conclusion unchanged. What makes XRR fits end in
  wrong structures here is the optimiser's basin of attraction, not the likelihood.
- **Fisher at the truth cannot be used as a stand-in for either side.** It cannot locate or rank
  the basins that trap searches (B-1 far side). It is also not the local error bar once a nuisance
  such as Kα2 or a thin oxide is bounded or curved (B-1 near side, B-2). The general phenomenon is
  known outside reflectometry [lit: Raue et al., Bioinformatics (2009), doi:10.1093/bioinformatics/btp358, profile likelihood against asymptotic intervals; Transtrum,
  Machta, Sethna, PRE 83, 036701 (2011), doi:10.1103/PhysRevE.83.036701, curved valleys and stalled
  least squares in sloppy models; both [V] metadata, content [M]]. Our near-side failures (TRF at its
  cap on a curved nuisance valley) look like the Transtrum case.
- **Measurement design.** For rejecting wrong structures of these kinds, counts above
  q ≈ 0.24 Å⁻¹ add almost nothing; the most discriminating region depends on how wrong the structure
  is (first fringes for gross failures, 0.08–0.2 Å⁻¹ for closer ones). This does not address
  roughness precision or helping a search find the truth.

## 4. Is "Fisher against global discrimination" a paper contribution?

### 4.1 Novelty against prior work

- [lit] In reflectometry, Fisher work is local and NR (Durant et al. 2021, 2022); multimodality
  appears as case studies (Kim & Lee 2021, PANPE 2025, Savchenkov 2024); no XRR profile-likelihood
  study or map of local against global structure was found (LIT-SURVEY 3; absence is not proof).
- [inference] **The claim "Fisher information is local and misleads globally" is not new** as a
  statement; the systems-biology and sloppy-model literature makes it in general. A paper built
  only on that claim would be judged incremental, and H-F (ii) on capped endpoints is close to
  expected in kind (deviance saturates, the quadratic does not).
- [inference] What would be new for XRR, and what our records already half support:
  1. a **count-scaled distance of competing structures**, S₁, mapped over contrast, q range,
     stacking and nuisance freedom with a realistic instrument model (Kα2, footprint, resolution,
     oxide), giving the count level at which XRR stops being globally identifiable;
  2. the **separation of physical identifiability from computational reach**, with the finding
     (so far) that at lab counts the second is what fails;
  3. the **two-sided failure of the Fisher bound** in XRR specifically: bounded or curved nuisances
     make it too pessimistic locally while it is too optimistic about distant basins;
  4. a **measured noise check** (φ) that sets how far the Poisson deviance scale can be trusted.

### 4.2 Evidence still missing

1. **Basin minima instead of returned fits.** 72 % of R2a's wrong endpoints and 28 of the 30
   near-side failures stopped at the 200-evaluation cap; X2e's states are returned fits too. S₁ of a
   basin needs its converged local minimum: a local refit to convergence from each distinct recorded
   endpoint (no new starts, so not the repeated initialisation comparison the owner excluded).
   CPU tens of minutes to hours [estimate]; needs registration and a HEAVY slot.
2. **A condition where S₁ approaches realistic counts.** Without one, the paper's result is "XRR in
   this class is globally identifiable at lab counts; failures are computational". That is a
   publishable negative result but a weaker paper. Candidates: low-contrast bilayers with free oxide
   (#25-like), thin layers, q_max ≤ 0.16 Å⁻¹, lower counts (10⁴).
3. **The bilayer basins that R2a never reached.** 13 bilayer conditions have no reached endpoint, so
   their truth-side widths are unknown; B-2 did one (#25). The rest is R2h's scope (BACKLOG, not
   registered).
4. **Noise beyond the inverse crime.** Fits to Poisson and to overdispersed data (φ from H-cal2),
   to show that S₁ and the reach picture survive model and noise misspecification.
5. **Measured data.** At least one public lab XRR set refitted from many starts, to show the
   wrong-basin phenomenon outside simulation. Our own measured data do not exist (RESEARCH-DESIGN 6).
6. **H-cal2 beyond one instrument.** One synchrotron fast-scan file is not a statement about
   laboratory step scans; it is a caveat, not a result.

### 4.3 Graduation (2027 first semester, directive 2026-10-22)

- [inference] Items 1, 2 and 4 are CPU-only and reuse the R2a and X2e code; with registrations and
  physics review they are weeks, not months, of lane time. Item 3 is R2h (hours of CPU, but blocked on
  DotFiles#58/#87 per BACKLOG). Item 5 needs a suitable public data set and is the main schedule risk.
- [inference] The directive of 2026-10-22 puts TraTT first (P0) and XRR identifiability second
  (P1). This line fits P1 as one methods paper, not as a thesis-wide programme.

### 4.4 Judgement

**Conditional yes, reframed.** "Fisher against global discrimination" on its own is not a sufficient
contribution: the general point is known, and our strongest number (H-F (ii)) is partly expected.
It becomes one as a section of a methods paper whose claim is **physical identifiability against
computational reach in XRR, measured in counts (S₁)**, with the Fisher comparison showing why local
tools cannot answer it, and H-W2 and H-cal2 as design and noise consequences. The go/no-go for that
paper hangs on item 1 (basin minima) and item 2 (a regime with S₁ near realistic counts). If item 1
moves S₁ by less than an order of magnitude and no regime in item 2 reaches ≥ 10⁴ counts, the
result is the negative one ("identifiable at lab counts; it is a search problem"), which is still
reportable but should be weighed against TraTT time.

The generative-posterior no-go stands: nothing here shows a structure with S₁ at or above measured
counts (RESEARCH-DESIGN 3.3, condition 1).

## 5. Proposed next step (not started; needs a decision)

Register and run item 1 on the existing records only: converge every R2a wrong endpoint with
ΔD ≤ 10⁵ (240 in the thirteen 2θ_max = 6° conditions, more over all 37) and X2e's 93 far true-L states by an
uncapped local fit, then recompute S₁, ρ and the q shares at the minima. Forward and local fits only,
no new starts, CPU, one process; time to be measured by a pilot under a HEAVY slot. This decides
whether the S₁ map is worth building.

[Back to all reports](../../reports/README.md)
