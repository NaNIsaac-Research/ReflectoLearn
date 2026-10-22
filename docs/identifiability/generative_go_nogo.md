# Generative inverse model for XRR: go / no-go (R1d)

Written 2026-10-23 after Experiment 001 (R1b, R1c). Nothing was trained; no diffusion or flow model
was started.

## Verdict

**No-go for now.** In every case we checked, the posterior of the film parameters has one mode, and
the wrong fits came from the optimizer, not from the curve. A generative posterior model would be
solving a problem these cases do not have. The question that would justify one is open: under which
conditions does a single-film or bilayer XRR curve admit several statistically indistinguishable
structures? That is tomorrow's pre-registered experiment (below). The decision is revisited on its
result.

## What the four approaches did on the checked cases

Cases: 240 Å / 6 Å / 45 film with and without a 10 Å top oxide, 200 Å / 4 Å / 12 film with the
oxide; 2θ 0.2–6°, 10⁶ counts, Kα2, footprint, offset, resolution, i0, background; all twelve refine
parameters free within refine's bounds (`README.md`, R1c).

| approach | result here | cost per curve (CPU) | status |
|---|---|---|---|
| CNN + refine | CNN not run (no checkpoint on this machine, DotFiles#30). refine alone, from 16 starts within ±20 % of the truth: 48/48 endpoints far from the best fit (ΔD > 4.8 × 10⁴), endpoint d −31 % to +45 % | ~5–10 s for 16 starts | fails as an optimizer: ReflectoLearn#25 |
| multi-start Poisson least squares | 43/45 starts reach the maximum-likelihood fit; the other 2 land on a thickness alias at ΔD ≈ 3.7 × 10⁵, clearly rejected | ~20–30 s for 15 starts | works |
| MCMC (`refine.mcmc`, emcee) | unimodal in all cases; from a failed refine endpoint it moves to the same posterior as from the ML fit. Width vs Poisson profile: agrees for 1 case, 4.7× too wide and 2–3× too narrow for the others | ~12 s, 2000 steps × 52 walkers | works for the mode; its likelihood (Gaussian in log₁₀ R) is not calibrated against counts |
| conditional normalizing flow / flow matching | not trained (out of scope tonight) | training: GPU hours; inference: ms | justified only by amortization or multimodality, neither shown here |

## Already solved by existing methods vs needs several solutions

- Solved: a single film with refine's nuisances on a lab scan to q ≈ 0.43 Å⁻¹. The curve determines
  d to ±0.04–1.4 % (profile Δχ² ≤ 1 at 10⁶ counts). A correct local likelihood fit from a handful of
  starts finds it; MCMC gives the uncertainty. What is missing is a working optimizer in refine
  (#25) and a likelihood that matches the counts, not a new model class.
- Possibly needs several solutions (not shown): a thin top layer with a wide free SLD range and a
  short q range (unbounded Fisher analysis in R1b: d–oxide SLD correlation −0.998, γ up to 1300 at
  q_max 0.21 Å⁻¹); bilayers with similar SLDs (layer-order swaps); thickness aliases (a second
  local minimum near 1.9 d was found, but at ΔD ≈ 3.7 × 10⁵ it is not a competing solution).
- Amortization is a separate argument: if thousands of curves must be inverted (in-situ growth,
  mapping), a trained posterior is fast. PANPE (Starostin et al. 2025) already does this for XRR and
  NR with a normalizing flow, q-offset and normalization nuisances, and importance-sampling/MCMC
  correction. A flow-matching or diffusion version of the same thing is an engineering variant, not
  a research question, unless it addresses something PANPE cannot (see `literature.md`).

## Candidate research questions before graduation

Both are hypotheses; neither has been checked against a full literature search.

1. **Where does XRR become multimodal?** A map, over q range, film/substrate contrast, number of
   layers and how freely the nuisance layer is allowed to vary, of where the Poisson likelihood of a
   curve has several indistinguishable structures (multimodal), one stretched solution (ridge), or a
   well-determined one. Criteria pre-registered, profile likelihood plus global multi-start, checked
   against MCMC. Prior work to separate from: Durant et al. 2021 (neutron, local Fisher only), Gil &
   Windover 2012 (X-ray, one contamination case, MCMC), PANPE (multimodal posteriors shown on
   examples, not mapped). If the map shows a realistic region with multimodality, that region is the
   test bed that justifies a generative posterior, and the comparison with PANPE has a defined target.
2. **How much information do common XRR fitting objectives throw away?** Log-MSE in log₁₀ R (refine,
   GenX's logR-type FOMs) against the Poisson likelihood: efficiency against the Cramér–Rao bound,
   bias at low counts, calibration of the resulting MCMC intervals. Tonight's data already show a
   2–5× miscalibration of `refine.mcmc` widths and endpoints where log-MSE and the likelihood
   disagree on which fit is better. Small, concrete, and directly useful to anyone fitting XRR; the
   risk is that it is known folklore without a paper, which the literature check must settle.

## Pre-registered experiment for 2026-10-23: condition scan for multimodality

Registered before running. Changes go in a new commit with the reason.

**Conditions (36).**
- q_max: 2θ_max ∈ {2.2°, 3°, 6°} (q_max ≈ 0.16, 0.21, 0.43 Å⁻¹), 2θ_min 0.2°, uniform q, 0.02° step
  equivalent.
- Contrast: film SLD ∈ {12, 24, 45} (×10⁻⁶ Å⁻², Si 20.07), d 240 Å, σ 6 Å.
- Nuisance freedom: (i) refine's bounds; (ii) wide: oxide SLD 5–40, oxide thickness 0–60 Å, Kα2
  ratio 0–1.
- Stack: (i) single film with top oxide; (ii) bilayer: the film plus a second film of 120 Å, SLD
  ±25 % of the first, under it, with top oxide.
- Fixed: 10⁶ counts at R = 1, background 10⁻⁷, offset 0, dq 0.002 Å⁻¹, Kα2 0.5, footprint 500,
  oxide 10 Å / 18.9 / 3 Å.

**Per condition, on noise-free data (expected counts):**
1. Global multi-start: 64 starts from a Latin hypercube over the full prior box (film parameters
   over `config.yaml` ranges, nuisances over the condition's bounds); Poisson least squares (as in
   `nonunique.py fit`) from each.
2. Modes: endpoints with Δχ² ≤ 4 (deviance against the noise-free data; the truth has 0) are
   grouped; two endpoints are in different modes if any film parameter differs by more than 3 times
   the larger of their two profile Δχ² ≤ 1 half-widths for that parameter (computed at each endpoint
   by the `nonunique.py` profile on that parameter).
3. Labels as in R1c, applied per film parameter (d of each layer, ρ of each layer): *multimodal* if
   more than one mode with Δχ² ≤ 1 exists; *ridge* if one mode whose Δχ² ≤ 1 profile half-width is
   ≥ 2 % of the value; *identifiable* otherwise.
4. Check: for every condition labelled multimodal, draw one Poisson curve and run emcee with the
   Poisson likelihood from each mode; it counts as confirmed if both modes keep posterior mass
   (each ≥ 5 % of samples after burn-in) in at least one of the runs.

**Decision rule.**
- Go for a generative posterior study (R2) if at least one condition with refine's bounds (nuisance
  freedom (i)) and q_max ≥ 0.21 Å⁻¹ is confirmed multimodal.
- Otherwise no-go stands, and the condition map itself is the result (question 1 above, with
  multimodality only in the wide or short-range corners, if anywhere).

**Cost.** 36 × 64 = 2304 least-squares fits at about 1–5 s each (12–15 parameters), 1–3 hours on one
CPU, under 1 GB. Queued as [HEAVY] in two halves (single film, bilayer). Before that, one condition
is timed and the bilayer forward model is checked against `benchmark.parratt` as in R1b.

**Step 0 (reading, 30 minutes).** PANPE's multimodal examples (which stacks, q ranges, priors), so
that the scan does not re-discover a published case without saying so.

## R2a step 0: PANPE's multimodal cases (read 2026-10-23)

Read: arXiv:2407.18648 v1 HTML (the only arXiv version); the Science Advances version and the
supplementary figures were not read.

- Where PANPE shows several modes: a simulated two-layer XRR curve with 10 free parameters gives 7
  modes under a wide prior and one under a narrow prior (Fig. 2); a simulated four-layer structure is
  "highly ambiguous" (SM Fig. 5, not read); the experimental in situ DIP-on-Si curves (one organic
  layer over a fixed oxide) show two branches in the film SLD, of which the upper one disappears
  later in the growth (Fig. 4); one neutron polymer curve in H₂O has two solutions that a D₂O
  contrast removes (Fig. 6).
- Regime: q_max 0.15 Å⁻¹, 20–64 points, Gaussian noise of 5–30 % relative per point (their
  approximation to counting noise), priors over thickness 0–500 Å, roughness 0–50 Å, SLD 0–60;
  nuisances q offset ±0.002 Å⁻¹ and normalization ±5 %. The numeric prior widths of the
  multimodal figures are not given.
- Stated cause: the phase problem, in general terms. No mechanism (layer swap, thickness exchange,
  zero contrast) is given for any specific case. Zero-contrast bilayers, which make the two
  thicknesses exactly collinear, were removed from their test set (under 5 %), not analysed.
- Not in the paper: any map of where multimodality occurs over q range, contrast or prior width;
  any comparison with a profile likelihood or Fisher information.

Consequence for the scan (no change to the registered rules): the registered scan sits at much
better statistics than PANPE's examples (10⁶ counts at R = 1, 291 points, against 5–30 % per point
over 20–64 points) and up to q_max 0.43 Å⁻¹. A single film being unimodal there does not contradict
PANPE's single-film branches. Because the noise-free deviance scales with the counting level s, the
scan can report for every secondary mode it finds the counting level s* = 10⁶ / ΔD below which that
mode is within Δχ² ≤ 1; this is reported as an addition, not used in the registered decision.

### Amendments to the condition scan (before any condition was run with its 64 starts)

1. *Second film of the bilayer.* The registration gives its thickness (120 Å) and SLD "±25 %" but not
   the sign or its roughness. Fixed here: SLD 1.25 × the first film's (15, 30, 56.25), roughness
   6 Å like the first. One sign keeps the count at 36 conditions; the other sign is not scanned.
2. *Resolution quadrature.* `scan.py` integrates the Gaussian resolution with 15 Gauss–Hermite nodes
   instead of `fim.py`'s 41-point ±5σ rule. Against `fim.expected_counts` the counts differ by at most
   2.2 × 10⁻⁵ relative and by a deviance of 0.0004 on the noise-free curve, far below the Δχ²
   thresholds of 1 and 4. The bilayer stack agrees with `benchmark.parratt` to 3.1 × 10⁻¹⁵ (no
   absorption, 2θ 0.2–6°).
3. *Cost estimate correction (2026-10-23).* The registered estimate (1–5 s per fit) was wrong. Measured with two other runs on the
   machine and 2 threads: about 14 s per fit for a single film at 2θ ≤ 6° and about 60 s for the
   bilayer before the quadrature change, with many fits from Latin-hypercube starts stopping at the
   200-evaluation cap far from any minimum. 2304 fits are therefore several hours of CPU, not 1–3.
   The method and the decision rules are unchanged; the scan runs in the two registered halves
   (single film first), each split over three processes.
4. *Profiles restart the inner fit (2026-10-23, physics review of PR #24).* The per-mode profiles of
   rule 2 compute each fixed-parameter point as the lowest of three local fits (continued from the
   previous point, nuisances at refine's initial values, nuisances at their bound midpoints), so that
   a half-width is not set by one tracked nuisance branch. Modes themselves come from the 64 global
   starts, not from the profiles. A restart that finds a point below the mode's own deviance is
   recorded (`lowest_dchi2` < 0).
5. *Named case added: t2_low Kα2 basin (2026-10-23, before the scan ran; decision rules unchanged).*
   Condition 36 is R1c's t2_low film (200 Å / 4 Å / SLD 12, with the 10 Å top oxide, single film,
   refine's bounds, 2θ 0.2–6°), run like the 36 registered conditions. On its noisy R1c curve two ML
   solutions differ only in the Kα2 ratio (0.51 vs 0.00) within ΔD = 1, and in R2c (amendment 3)
   neither MCMC chain converged, with τ growing with chain length. The registered rules label film
   parameters (d, ρ) only, so a basin that differs only in a nuisance is one mode under them. For this
   condition the report also lists the endpoints within Δχ² ≤ 4 whose nuisances differ (Kα2 ratio
   by more than 0.1, or oxide thickness or SLD by more than 10 %), as an observation. It does not enter
   the go/no-go decision, which keeps its registered wording.

### Decision flow (written 2026-10-23, before any scan condition was analysed)

This restates the registered rules and amendments 1–5 as the order the code applies them in
(`scan.py run`, then `scan.py analyze`). It changes no rule.

1. **Global multi-start.** For each condition, 64 Latin-hypercube starts span the whole prior box,
   and a Poisson least-squares fit runs from each (at most 200 evaluations) on the noise-free
   expected counts. The truth has D = 0, so an endpoint's D is its Δχ².
2. **Candidates.** Endpoints with Δχ² ≤ 4.
3. **Modes.** Candidates are taken in order of D. Each one joins the first existing mode it is not
   distinct from, or starts a new mode. Two endpoints are distinct if some film parameter (d or ρ of
   a layer) differs by more than 3 × the larger of their two profile half-widths for that parameter.
4. **Per-mode profiles.** Each film parameter at each mode's best endpoint gets a Δχ² = 1
   half-width. The parameter is stepped out by doubling until Δχ² rises above 1 or the bound is
   reached, then bisected 6 times. Every point is the lowest of three local fits: continued,
   nuisances at refine's initial values, nuisances at bound midpoints (amendment 4). A restart that
   lands below the mode is recorded as `lowest_dchi2` < 0.
5. **Labels, per film parameter.** *Multimodal* if two or more modes have Δχ² ≤ 1 and differ in
   that parameter by the rule in step 3. *Ridge* if not, and the best mode's half-width is ≥ 2 % of
   the value. *Identifiable* otherwise.
6. **Check of multimodal labels.** Draw one Poisson curve, run emcee with the Poisson likelihood from
   each mode, and confirm if both modes keep ≥ 5 % of the samples after burn-in in at least one run.
   This is not implemented yet, and runs only if step 5 gives a multimodal label.
7. **Decision.** Go if at least one condition with refine's bounds and q_max ≥ 0.21 Å⁻¹ (2θ ≥ 3°)
   is confirmed multimodal in step 6; otherwise no-go stands and the map is the result.

Reported alongside but outside the decision:
- the nearest distinct competitor above Δχ² 4, with s* = 10⁶ / ΔD, the counting level at which it
  would come within Δχ² 1;
- for every condition, endpoints within Δχ² ≤ 4 whose nuisances differ (amendment 5).

**The named case, t2_low Kα2 basin (condition 36).** On one noisy curve, R1c found two ML solutions
that differ only in the Kα2 ratio (0.51 vs 0.00) within ΔD = 1. R2c (amendment 3) did not converge
on that curve, and τ grew with chain length. The scan fits the noise-free curve instead, where the
truth (Kα2 0.5) has D = 0. Expected readings, stated before the result:
- If the second basin shows up as endpoints with Kα2 near 0 inside Δχ² ≤ 4, it lies in the
  likelihood itself.
- If it shows up only above Δχ² 4, it is a feature of that noise draw, within reach at the counts
  given by s*.

Either way the film labels follow steps 3–5, which ignore nuisances.

### Interim (2026-10-23 03:20): fits only, no modes, no labels, no verdict

| # | condition | points | starts within Δχ² ≤ 4 | lowest Δχ² above 4 | fit time (one process, 3 threads) |
|---|---|---|---|---|---|
| 0 | single, refine bounds, 2θ ≤ 2.2°, ρ 12 | 101 | 2/64 | 4.6 × 10⁴ | 558 s |
| 1 | single, refine bounds, 2θ ≤ 2.2°, ρ 24 | 101 | 15/64 | 8.0 × 10³ | 575 s |
| 2 | single, refine bounds, 2θ ≤ 2.2°, ρ 45 | 101 | 13/64 | 1.3 × 10⁴ | 471 s |

These are raw fit counts. Whether the endpoints within Δχ² ≤ 4 form one mode or several is decided
in step 3, after all of a process's conditions are fitted.
