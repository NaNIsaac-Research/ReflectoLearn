# Pre-registration G3: the box and the Kα₂ ratio

Written 2026-11-06, before any run below. Nothing in this file has been run. 651 of 656 R2a minima
lie on a bound of the prior box, and the Kα₂ ratio is on a bound in 550, so the S₁\* of a wrong basin
may describe the box, or an over-parametrised nuisance, rather than search reach
([../paper_draft.md](../paper_draft.md) sections 0 and 9).

Subset: `experiments/005_fisher_global/prereg_subsets.py`, seed 20261107, file
[g3_subset.json](g3_subset.json): the 5 distinct R2a basins with S₁\* ≥ 10³, 95 drawn from the basins
whose recorded minimum has the Kα₂ ratio on a bound, and 50 from the others, 150 in all. Code is
committed before the first run and its hash is recorded in each result file.

## Variants

All use the basin run's refinement (`basin.refine`, same objective, budget N_max = 3000 and
convergence tests C1 to C4), start from the candidate's recorded minimum, and score against the same
noise-free truth curve at S = 10⁶.

| variant | change from the recorded run |
|---|---|
| V0 | none; a replication of the recorded minimum, to confirm the harness reproduces S₁\* |
| V1 | Kα₂ ratio fixed at 0.5 (the truth's value in all 37 conditions) and removed from the free parameters |
| V2 | nuisance box widened to twice its width about its centre, clipped at physical limits: thickness and roughness at least 0, Kα₂ ratio in [0, 1], i₀ and dq above 0, oxide SLD between 0 and the film's SLD limits of the scan, footprint above 0 |
| V3 | V1 and V2 together |

The film parameters (thickness, SLD, roughness of each film) keep their recorded bounds in every
variant.

## Outcomes per candidate (descriptive and registered counts)

- **Vanishes.** The refinement ends within ΔD ≤ 4 of the truth (the candidate's basin was a box
  artefact).
- **Moves.** S₁\* changes by more than a factor of 10, up or down.
- **Crosses.** S₁\* crosses 10⁴ in either direction.
- **Stays.** None of these.

## Hypotheses

| hypothesis | statement | holds | fails |
|---|---|---|---|
| H-K1 | with the Kα₂ ratio fixed (V1), the median S₁\* of the candidates that had it on a bound stays within a factor of 3 of V0's | yes if so for at least 80 % of those candidates | no if for fewer than 60 % |
| H-K2 | the exceptions persist: of the 5 basins with S₁\* ≥ 10³, at least 3 stay at S₁\* ≥ 10³ in V1 | yes if at least 3 | no if 0 or 1 |
| H-B1 | with the widened box (V2), the median S₁\* stays within a factor of 3 of V0's | yes if so for at least 80 % of candidates | no if for fewer than 60 % |

Between the bounds a hypothesis is unresolved. If H-K1 and H-B1 hold, the reading that the wrong
basins are search reach for the film and nuisance model stands. If either fails, the draft's central
claim is restated for the Kα₂-fixed (or widened-box) problem and the changed median is reported.

## Cost

150 candidates × 4 variants × mean 241 s (the recorded mean per R2a refinement) is 40 CPU hours as the
upper bound; refinements from a recorded minimum are expected nearer the recorded median of 26.5 s,
about 4 CPU hours for the four variants, so 4 to 40 CPU hours (inference). With 4 processes: 1 to 10
hours of wall time. The runs start with V1 and V0 on the 5 high-S₁\* basins; the remainder runs only
if the 5 finish in under 2 minutes each.

Runs: 4 processes, one torch thread each, BelowNormal priority, a wrapper that stops at 4 GB per
process or 16 h of wall time; one result file per candidate and variant, written once.
