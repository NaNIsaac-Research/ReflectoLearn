# B-1 results: R2a endpoints against the Fisher quadratic (H-F, H-W2)

xrr-policy lane, 2026-10-25. Registration: `PREREG-HF-HW2.md` (ReflectoLearn `research/fisher-global`
d1f164c, committed and pushed before any number below). Code and generated output: same branch,
979ac1c, `experiments/005_fisher_global/b1.py` (`run`, `report`) and `b1_extra.py`;
`docs/fisher_global/b1.json`, `b1_tables.md`, `b1_verdict.json` (registered), `b1_extra.md`
(post hoc). These numbers are not X2e's: X2e Task A numbers quoted here are marked [X2e post hoc].

What was run: forward evaluations at the 2368 recorded R2a endpoints and their 37 truths, Jacobians at
the truths, and 8 × 14 check displacements per truth. 31 s of CPU, one process, one thread. No fit, no
sampler. No R2a record, label or verdict was changed (condition 2); nothing was written under
`docs/identifiability/`.

## 1. Checks (registered stop rules)

- [fact] Check 1: the recomputed deviance matches the recorded one for every endpoint with recorded
  D ≥ 1; largest relative difference 2.8 × 10⁻¹⁰ (limit 10⁻⁶). Pass.
- [fact] Check 2: in all 37 conditions the computed ΔD matches the Fisher quadratic to 2 % along 8
  random directions at every level from 10⁻¹⁴ to 10⁻¹, so D_floor = 10⁻¹⁴ everywhere and no near
  endpoint was left out (the smallest recorded near deviance is ~10⁻¹¹).
- [fact] The 2θ_max = 6° grid splits into the same regions as X2e's (low 0.014–0.036, b0 0.037–0.075,
  …, b9 0.390–0.427 Å⁻¹).

## 2. Registered verdicts

| hypothesis | statement | fraction | rule | outcome |
|---|---|---|---|---|
| H-F (i) | ρ = ΔD_F/ΔD in [0.5, 2] for near endpoints (ΔD ≤ 4), n = 177 | 0.831 | holds ≥ 0.90, fails < 0.60 | unresolved |
| H-F (ii) | ρ ≥ 10 for wrong endpoints (ΔD > 4), n = 2191 | 0.981 | holds ≥ 0.80, fails < 0.60 | holds |
| **H-F** | | | | **unresolved** |
| H-W2 (a) | ≥ 90 % of ΔD at q ≤ 0.235 Å⁻¹, n = 769 (13 conditions, 2θ_max 6°) | 0.996 | holds ≥ 0.80 | holds |
| H-W2 (b) | largest-share region is b0, b1 or b2 | 0.953 | holds ≥ 0.80 | holds |
| **H-W2** | | | | **supported** |

H-F is not supported: part (i) failed to reach 90 %. That is the registered outcome and stands.

## 3. Facts behind the verdicts

### 3.1 Near endpoints (H-F (i))

- [fact] 147 of 177 near endpoints have ρ in [0.5, 2]; their ΔD is at most 3.5 × 10⁻¹⁰ (median
  8 × 10⁻¹¹), i.e. the fits converged onto the truth and the quadratic is exact there (median ρ 1.00).
- [fact] The 30 outside the band sit in seven conditions: #1 (8 of 15), #10 (10 of 10), #11 (5 of 11),
  #28 (4 of 4), #2, #9, #19 (one each). In every case ρ > 2 (3.2 to 3.5 × 10⁴); none is below 0.5.
  28 of 30 stopped at the 200-evaluation cap. Their ΔD is 10⁻⁸ to 1.4.
- [fact, `b1_extra.md`] Along the straight line θ_t + t Δθ, ΔD(t)/(t² ΔD_F) falls from 0.37–0.89 at
  t = 0.25 to 3 × 10⁻⁵–0.31 at t = 1. In all but two (#2, #9) ΔD(t) itself is not monotonic: the
  deviance rises along the straight line and comes back down at the endpoint. The largest displacements are in oxide SLD, log background,
  film roughness and oxide roughness/thickness (wide-nuisance conditions #10, #28: background,
  roughness, oxide).
- [fact] Two of the 30 (#1 start 36, #10 start 36) did not hit the cap, have ΔD = 1.4 and a Kα2 ratio
  displaced by 0.83 and 0.5 of its bound width; ρ = 620 and 4.5 × 10³.
- [fact] Decision-level agreement: of the 177 near endpoints, only 3 have ΔD_F > 4 (Fisher would put
  them outside Δχ² ≤ 4); 25 of the 30 band failures have ΔD_F < 1, so Fisher and the deviance agree
  that they are indistinguishable from the truth and differ only in by how much.

### 3.2 Wrong endpoints (H-F (ii))

- [fact] ρ ≥ 10 in 2149 of 2191; ρ ≥ 10 in at least 92 % of the wrong endpoints of every condition;
  no wrong endpoint has ΔD_F ≤ 4. Median ρ by ΔD decade: 10³–10⁴: 336; 10⁴–10⁵: 108; 10⁵–10⁶: 188;
  10⁶–10⁷: 120; 10⁷–10⁸: 24.
- [fact] W′ (4 < ΔD ≤ 10⁴, S₁ ≥ 100): 87 endpoints, ρ median 336 [IQR 124–591], 99 % ≥ 10.
- [fact] 1579 of the 2191 stopped at the cap (ΔD median 1.4 × 10⁶, S₁ median 0.7 counts); 612 stopped
  before it (ΔD median 8.3 × 10⁴, S₁ median 12). ρ ≥ 10 in 98 % and 99 % of them.
- [fact] |cos| between Δθ and F's weakest bound-scaled direction: median 0.24–0.32 for wrong endpoints
  in each decade from 10³ up (82 to 662 endpoints each), 0.99 for near ones.
- [fact] S₁ of wrong endpoints, condition medians: single films 0.6–26 counts, bilayers 0.25–4.5.
  At the data's 10⁶ none comes near indistinguishability.

### 3.3 q regions (H-W2)

- [fact] Median shares over the 769 test endpoints: low 0.126, **b0 0.765**, b1 0.058, b2 0.011, b3
  0.003, b4 and above ≤ 0.001. Largest-share region: b0 in 729, low in 34, b1 in 4, b5 and b4 one each.
- [fact] By film ρ: (b) 0.88 (ρ 12 single), 0.93–1.00 elsewhere; low is the largest region in 12 % of
  ρ = 12 single-film endpoints, ≤ 5 % in the others.
- [fact, post hoc] Subsets: stopped before the cap (205): (a) 0.99, (b) 0.98, b0 median 0.76;
  ΔD ≤ 10⁵ (240): 0.99, 0.97; **ΔD ≤ 10⁴ (23): (a) 0.87, (b) 0.91**, shares spread out (low 0.08,
  b0 0.40, b1 0.16, b2 0.07, b3 0.08).
- [fact] Against X2e [X2e post hoc]: there the largest-share region was b2 (median share 0.33, largest
  in 520 of 996). In R2a b2 is the largest in none of 769.

## 4. Interpretation [inference]

1. **Fisher at the truth describes the truth's basin, not where local searches end.** For every
   wrong endpoint, the quadratic overstates the deviance, by two orders of magnitude at the median,
   so the count level Fisher would assign to their indistinguishability is about 100 times too low.
   Fisher never misclassified a wrong endpoint as near. This repeats X2e's post-hoc picture (factor
   1.5 × 10³ there) on a different model with oxide, Kα2 and footprint, all points fitted and all
   parameters free.
2. **Why (i) fell short is curvature, and it carries little decision weight.** The band failures lie
   on curved, low-deviance valleys of nuisance parameters (oxide SLD with background and roughness;
   Kα2 ratio), where TRF runs out of evaluations. A straight-line quadratic cannot follow a curved
   valley. Fisher still calls 174 of 177 near endpoints near. The registered test asked for a
   stronger property (ratio within a factor 2) and that property does not hold for 17 % of near
   endpoints. So "Fisher is exact inside the basin" is true only for converged points, not for the
   whole Δχ² ≤ 4 region, when nuisances such as Kα2 or a thin oxide are free.
3. **The Kα2 cases are the closest thing to a second nuisance basin.** Two endpoints with Kα2 ratio far
   from the truth reach ΔD = 1.4 (R2a amendment 5 noted Kα2 basins for t2_low). Fisher misses them by
   a factor of 600–4500. This concerns a nuisance, not a film parameter, and says nothing about
   multimodality of film structure.
4. **H-W2 holds, but not for X2e's reason.** R2a's wrong endpoints put their misfit at the first fringe
   block b0 (q 0.04–0.075), not at b2. Most are gross failures (median S₁ < 1 count) whose misfit is
   largest where the counts are. Only the small, closest subset (ΔD ≤ 10⁴, n = 23) spreads toward
   b1–b3 as X2e's did, and there (a) is at 0.87. The X2e b2 peak is therefore not a general property;
   being held out of the fit (X2e) is one explanation, a different distance from the truth another.
   B-1 cannot tell them apart.
5. **What H-W2 does and does not say.** For these endpoints, counts above q ≈ 0.24 Å⁻¹ add almost
   nothing to rejecting them. It says nothing about which q range helps a search find the truth, or
   about roughness precision.

## 5. Limits

- Inverse crime: noise-free data from the fitted model.
- R2a's endpoints are returned fits under a 200-evaluation cap; 72 % of wrong ones and 28 of the 30
  band failures stopped at it. They are not basin minima, so S₁ is a lower bound for any basin.
- Fisher is evaluated only at the truth.
- One count level (10⁶), one background, one truth per condition.
- The H-F (i) band [0.5, 2] and the 90 % threshold were chosen with X2e's near states in view; the
  outcome is reported as registered. The decision-level count in 3.1 (3 of 177) is post hoc.
