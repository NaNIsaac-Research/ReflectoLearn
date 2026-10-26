# Basin minimum validation: pilot and compute budget

xrr-policy lane, 2026-10-25. Protocol: `PREREG-BASIN.md` (ReflectoLearn `research/fisher-global`
51b048a, committed before any candidate was refined). Code: `experiments/005_fisher_global/basin.py`
(run) and `basin_pilot_report.py` (tables). Outputs: `docs/fisher_global/basin_pilot.jsonl` (raw) and
`basin_pilot.md` (generated tables). A copy of this file is `docs/xrr/GEN-PHASE1/RESULTS-BASIN-PILOT.md`
in the workspace.

| record | |
|---|---|
| type | Experiment Result (pilot; no verdict) |
| protocol | `PREREG-BASIN.md` section 6 |
| hypotheses touched | none decided; H-BM1, Q-BM2, H-BM3 wait for the full run |
| earlier verdicts | unchanged: H-F unresolved, H-W2 supported, H-cal2 supported, X2e fail, R2a labels |

Labels: [fact] measured in this pilot; [pilot] a pilot number that is not used for any verdict;
[inference] interpretation. Eight candidates are not a sample from which the hypotheses can be judged,
and the registered rules are not applied to them.

## 1. What was run

- The eight registered pilot candidates (six R2a, two X2e), O_nf for all and O_tr also for the two
  X2e candidates: ten refinements. One process at a time, torch 1 thread, BelowNormal. Wrapper
  (workspace scratchpad `basin_pilot.ps1`) with a 30 min / 2 GB stop on the Python process tree.
- [fact] Wall time 297 s for all ten (R2a 285 s, X2e 12 s); no stop; exit 0 for both interpreters.
  Peak working set 0.725 GB for the process tree (wrapper), 678–738 MB per run by the process's own
  counter.
- No candidate outside the pilot was refined. No earlier record was written.

## 2. Numerical stability [fact]

- Start check: recomputed endpoint deviance against the record, relative difference
  1.4 × 10⁻¹⁵ – 1.8 × 10⁻¹³ (rule ≤ 10⁻⁶). All ten pass.
- No non-finite value in any stage; no "unconfirmed" outcome. All ten reached **confirmed**
  (C1–C4): 9 with a smallest relative Hessian eigenvalue above 10⁻¹⁰, one (r2a:33:13) with one weak
  direction whose ± probes stayed within 5 × 10⁻⁶ of D (τ = 0.035).
- Evaluations to confirmation: R2a 17, 374, 416, 466, 564, 2225; X2e O_nf 19, 29; O_tr 7, 9.
  The largest (r2a:33:13) used 74 % of the 3000 budget: ten capped stages before TRF stopped by its
  own tolerance.
- Bounds: every R2a minimum is **on bound** for 3–6 parameters: the Kα2 ratio in five of six (0 in
  four, 0.6 in one), oxide parameters in all six (oxide thickness at 60 Å in four), film roughness in
  four (0 Å in three, 15 Å in one), the background at its upper bound in r2a:00:58, and the second
  film's thickness at its lower bound (10 Å) in r2a:21:29, which makes that bilayer minimum nearly a
  single film. X2e: x2e:C:2:12:1:1 has σ₂ at its upper bound (15 Å), x2e:D2:3:10:0:0 is interior.

## 3. Pilot numbers at the minima [pilot]

| id | evals | ΔD_nf endpoint → minimum | κ | S₁ endpoint → minimum | ρ endpoint → minimum | top region | drift (u) |
|---|---|---|---|---|---|---|---|
| r2a:06:44 | 466 | 8.307 × 10⁴ → 8.306 × 10⁴ | 1.00 | 12 → 12 | 91 → 90 | b0 → b0 | 0.16 |
| r2a:06:11 | 374 | 8.307 × 10⁴ → 8.306 × 10⁴ | 1.00 | 12 → 12 | 91 → 90 | b0 → b0 | 0.17 |
| r2a:25:25 | 17 | 1779 → 1779 | 1.00 | 562 → 562 | 47 → 47 | b0 → b0 | 0.0007 |
| r2a:33:13 | 2225 | 3.533 × 10⁴ → 3.529 × 10⁴ | 1.00 | 28 → 28 | 181 → 183 | b0 → b0 | 0.61 |
| r2a:00:58 | 564 | 8.169 × 10⁴ → 8.023 × 10⁴ | 1.02 | 12.2 → 12.5 | 88 → 84 | (2.2° grid) | 1.0 |
| r2a:21:29 | 416 | 3.576 × 10⁴ → 3.574 × 10⁴ | 1.00 | 28 → 28 | 185 → 185 | (3° grid) | 0.20 |
| x2e:C:2:12:1:1 | 19 | 2.179 × 10⁴ → 2.153 × 10⁴ | 1.01 | 46 → 46 | 776 → 780 | b1 → b0 | 0.08 |
| x2e:D2:3:10:0:0 | 29 | 5.08 × 10⁶ → 4.26 × 10⁶ | 1.19 | 20 → 24 | 2890 → 3520 | b2 → b1 | 0.20 |

O_tr (X2e, descriptive): both returned states were already minima of the training objective to the
registered tolerance (7 and 9 evaluations, drift ≤ 10⁻⁵, training deviance unchanged to four digits),
so their ΔD_nf and S₁ are unchanged.

- Duplicate rule: r2a:06:44 and r2a:06:11 start 0.20 Å apart in film roughness and converge to minima
  0.008 Å apart with max(ΔD(a|b), ΔD(b|a)) = 1.0 × 10⁻⁵, so the rule would group them as one basin.
- No truth-basin minimum (all ΔD_nf ≥ 1779). None with S₁\* ≥ 10⁴ (largest 562, #25's known
  competitor).

[inference, from eight points] The capped R2a endpoints were already within 2 % of their basin's
minimum in deviance, though not in parameters: r2a:33:13 moved 61 % of a bound width (background
from −6.3 to −10.0, both film thicknesses by 4 Å) for a 0.1 % drop in ΔD, and r2a:00:58 moved film
roughness from its upper bound (15 Å) to its lower bound (0 Å) and the oxide to 60 Å (its upper bound)
for a 1.8 % drop. If the full run looks like this, H-BM1 would come out supported for S₁ while the
location of these basins is poorly fixed along flat, box-limited directions. Eight points do not
show that; the full set has 403 capped R2a endpoints in 33 conditions.

## 4. Observations about the protocol (before the full run)

1. **C4 repeats.** In all 10 refinements the first confirmation stage failed on the step rule
   (moved more than the 10⁻⁶ limit in u, up to 3.7 × 10⁻³) while lowering D by ≤ 1.5 × 10⁻³
   (τ 0.002–4.3), so the refinement continued and was confirmed later: 1 to 25 failed confirmation
   stages (25 in r2a:00:58). These stages are cheap (2–16 evaluations each) and are counted in the
   budget, but they show that along flat directions the step rule, not the deviance, decides when
   a candidate counts as converged. Under the registered rule a candidate that keeps drifting at
   constant D would end as "unconfirmed: budget", which is the conservative direction (condition 7).
   No change proposed.
2. **Box-limited minima.** Every R2a minimum sits on 3–6 bounds. S₁\* of such a minimum is a minimum
   inside R2a's prior box, not of the unconstrained likelihood; the registration labels them "on
   bound" and they are kept in the statistics. Whether to also report H-BM1 restricted to interior
   minima is an owner choice (a descriptive split is already registered).
3. **Budget margin.** One of six R2a candidates used 74 % of N_max. A tail of the 403 capped
   endpoints will likely exceed 3000 evaluations and end "unconfirmed: budget". H-BM1 becomes "not
   testable" only if fewer than half of a source's candidates are confirmed. Raising N_max (for
   example to 6000) would need an amendment before the full run and roughly doubles the worst case.
4. Memory 0.74 GB per process is dominated by the interpreter and torch, not the problem size.

## 5. Compute budget for the full run

Per evaluation 0.044 s (median, R2a; max 0.061) and 0.015–0.021 s (X2e). Mean per candidate from the
pilot: R2a 47 s (median 30, max 159), X2e O_nf 3 s, O_tr 1 s. Worst case: every candidate spends
N_max = 3000 evaluations plus the largest test overhead seen (28 s).

| set | candidates | CPU h, pilot mean | CPU h, worst case | memory |
|---|---|---|---|---|
| R2a O_nf | 749 | 9.7 | 44 | 0.74 GB per process |
| X2e O_nf | 93 | 0.07 | 1.8 | 0.71 GB |
| X2e O_tr | 93 | 0.04 | 1.3 | 0.71 GB |
| **all** | 935 refinements | **9.8** | **47** | |

Wall time with independent processes (linear scaling assumed; one torch thread each):

| processes | wall h, pilot mean | wall h, worst case | peak memory |
|---|---|---|---|
| 1 | 9.8 | 47 | 0.74 GB |
| 2 | 4.9 | 24 | 1.5 GB |
| 4 | 2.5 | 12 | 3.0 GB |

Smaller options with the same protocol: the thirteen 2θ_max = 6° R2a conditions only (240
candidates, where q shares and H-BM3 (b)–(c) are defined) plus X2e: about 3.2 CPU h at the pilot
mean, 15 h worst case. The pilot mean rests on six R2a candidates, five of them capped; the spread
(10–159 s) is wide, so the mean figure is uncertain by a factor of about two [inference].

Every option is over the 30 min / 2 GB that needs no slot, so the full run needs a HEAVY slot.

## 6. Decision for the owner

1. Full run or not, and on which set: all 842 candidates (≈ 10 CPU h, 2.5 h at 4 processes, ≤ 12 h
   worst case), or the 6° subset plus X2e (≈ 3 CPU h).
2. Whether to amend before the run: N_max (section 4.3) and an interior-only H-BM1 split (section
   4.2). The lane's recommendation is to run as registered and report those two as descriptive
   splits, so the criteria stay the ones fixed before any result.
3. Process count: 4 processes BelowNormal fits the 3 GB memory; 2 if another HEAVY job runs.

Before the run: the verdict code (H-BM1, Q-BM2, H-BM3, duplicate grouping) is committed first; the
pilot candidates are re-run and compared with these rows.
