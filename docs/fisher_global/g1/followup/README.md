# G1 follow-ups on the recorded fits

Computed by `experiments/005_fisher_global/g1_followup.py` from the 100 recorded M1 repeat fits. No
fit was re-run. Facts are the numbers in the JSON files; lines marked "inference" are not measured.

## Residuals at the attenuator changes

[`attenuator_steps.png`](attenuator_steps.png), [`attenuator_steps.json`](attenuator_steps.json).

The stored attenuation factor changes by about 11 times at three places, each spread over two or three
points: 149-150 (q 0.105 1/Å), 289-290 (q 0.166) and 488-490 (q 0.254). The mean Pearson residual over the
100 fits is dominated by an oscillation at the fringe period, amplitude 1 to 3, on a slow trend.

The step at each change is the difference at the change between straight lines fitted to the 8 points
before and the 8 points after. The same statistic at 660 points at least 22 points from any change
gives the control distribution (|step| median 0.13, 95th percentile 0.38).

| change | step (mean ± s.e., 100 fits) | controls with a step at least as large |
|---|---|---|
| 149-150 | 0.05 ± 0.14 | 80 % |
| 289-290 | -0.29 ± 0.11 | 11 % |
| 488-490 | 0.17 ± 0.15 | 35 % |

None of the three stands out from the smooth misfit. Inference: a relative error ε in one segment's
factors shifts the Pearson residual by about ε·√λ, with λ of 148 to 379 counts just after the changes, so an
error above about 2 to 4 % would have shown. The excess deviance is not explained by the attenuator factors;
the fringe-period pattern points to the structure or resolution model.

## H-G1c within the q-offset families

[`hg1c_families.md`](hg1c_families.md), [`hg1c_families.json`](hg1c_families.json).

83 of the 100 repeat fits have the q offset at its lower bound. Within that family the across-scan sd over
the dispersion-scaled CRB (median over the family's fits, all 11 parameters free) is 1.17 for the Au thickness
and 2.00 for Cr; the pooled values on the same definition are 2.94 and 2.46. The 17 other fits are spread
wider (4.64 and 3.20). With the q offset and σ_q fixed instead of free, the CRB of the Au thickness drops
about four times and every ratio rises. Neither CRB is the right one while the two nuisances sit on their bounds,
so these rows describe the spread and do not decide H-G1c. The registered rule was written for the pooled sd.

The pooled ratio in `../g1_report.md` (2.97, 2.19) uses the Fisher matrix of repeat 0 alone; the median over
all repeats gives 2.94 and 2.46.

## H-G1b profiled over the nuisances, with a null

[`../hg1b_profiled.md`](../hg1b_profiled.md), one JSON per score in [`../hg1b_profiled/`](../hg1b_profiled/),
code `experiments/005_fisher_global/g1_hg1b.py`. No structural fit was re-run; each score refits only i0, b,
q offset and σ_q of the source fit to the target scan's best curve, on the target scan's counts, inside the
registered bounds.

- The null, 200 seeded pairs of the 100 repeat fits, has a 95th percentile of 557 (421 for pairs in the same
  q-offset family). The Poisson-only reference 2φχ²₇ is 33.6. Fits of one sample on different scans differ far more
  than counting noise allows, so 25 was never a usable same-structure threshold here.
- With the repeat fits of the five drawn scans (scan 68's search best is a missed basin), every pair inside one
  q-offset family scores well below that family's 95th percentile: 42 to 98 among scans 19, 68 and 76, and 204 to 213
  between scans 15 and 41. 4 of the 20 directed scores lie above the all-pairs 95th percentile, all between scan 15
  and scans 68 or 76, which sit in different q-offset families.
- In most cross-family scores the profiled q offset ends on its bound, so the profile cannot bring the two families
  together. Inference: the remaining cross-family difference is the bound-induced offset/Au SLD split, not
  evidence of different structures; it stays unresolved until the q offset and σ_q are refitted inside wider bounds.
- With the search bests as registered, 8 of 20 are above the 95th percentile; the extra four involve scan 68's
  missed basin.
- The 20 directed scores are compared one by one with a single-pair percentile, with no correction for the number
  of comparisons, and the null mixes both families as the repeats do. The nuisance minimum comes from two local
  starts per score.
