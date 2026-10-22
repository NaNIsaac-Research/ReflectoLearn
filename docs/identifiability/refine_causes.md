# Why refine stops short of the best fit (R2b, ReflectoLearn#25)

R1c found that `refine.refine` with the default `RefineConfig` ends far from the maximum-likelihood
fit from all 48 starts (three cases × 16), while a Poisson least-squares fit from the same starts
reaches it from 43 of 45. Issue #25 lists three candidate causes without separating them. This
experiment switches them on and off one at a time.

## Pre-registration (written before running)

Registered 2026-10-23, before any of the computations below were run. Changes go in a new commit
that says why; they are not edited in here.

**Data.** The three R1c cases (`mid_oxide`, `mid_no_oxide`, `t2_low`; truth, q grid, 10⁶ counts and
seed 0 as in `nonunique.py`), with one difference: the expected counts are computed with refine's own
forward model (`refine.make_forward`, all effects on, oxide off for `mid_no_oxide`), not with
`fim.py`'s. R1c showed that the two differ by up to 3.6 % at the first grid point (edge padding before
smearing). With refine's model generating the data, the model refine fits is exactly the true one, so
whatever is left is the search.

**Score.** For an endpoint, D is the Poisson deviance of the noisy counts under refine's forward model
(float64 deviance from the float32 model). D* is the lowest D found on that curve by any
configuration below, including a reference run (all factors on, budget ×30, started at the truth).
An endpoint has *reached* the best fit if D − D* ≤ 4. The *reach rate* of a configuration is the
number of reached endpoints out of 48 (16 per case: the 15 R1c starts plus the truth).

**Factors** (each one a change to `RefineConfig` or a stage list; nothing in `src/` changes):

| | factor | off (default) | on |
|---|---|---|---|
| A | step budget | 120 / 200 / 300 Adam steps | ×10 in every stage |
| B | stage path | stage 2 (correlation loss on d, ρ, offset) | stage 2 removed |
| C | objective | log₁₀ MSE over nonzero counts in stage 3 | Poisson deviance per point (counts scale s known, zero counts included) in stage 3 |
| D | Occam prior | `nuisance_reg` 0.2 on oxide thickness and dq | 0 |
| E | trust region | `trust_margin` 0.4 | 1.0 |
| F | step size | stage 3 at lr 0.01 to the end | stage 3 followed by two stages over the same keys at lr 10⁻³ and 10⁻⁴, 300 steps each (×10 with A) |

A, B and C are the three causes listed in #25. D, E and F are added here, before running, because
reading the code shows each can stop refine short of the likelihood optimum on its own: D because
under the log-MSE objective the penalty (about 0.006 at the truth's 10 Å oxide) is of the same order
as the objective at the optimum, so it moves the minimum; E because the roughness trust region of the
`t2_low` start (1.9–4.5 Å) excludes the ML roughness of 0 found in R1c; F because Adam at a fixed
learning rate in the sigmoid latent space jitters by roughly lr × (bound width) / 4 ≈ 0.5 Å in d for
the mid film, larger than its profile half-width (0.1 Å).

**Configurations.** Baseline (all off); each factor alone (6); all on; all on minus one factor (6).
Convergence check: all on with budget ×20 instead of ×10; if its reach rate differs from all on by
more than 2 of 48, the ×10 budget is not converged and that is reported before any verdict.

**Decision rules.**
- A factor is *sufficient* if switching it on alone gives a reach rate ≥ 43/48 (the rate of the
  Poisson least-squares fit in R1c, 43/45 ≈ 96 %, rounded down to allow for the three extra truth
  starts).
- A factor is *necessary* if all-on reaches ≥ 43/48 and all-on minus that factor reaches < 43/48.
- If all-on reaches < 43/48, refine's search is not fixable by these settings alone, and the report
  says so instead of naming a cause.
- The verdict for #25 names the sufficient factors if any, else the necessary ones. A fix is proposed
  only for factors that are sufficient or necessary.

**What would falsify the issue's account.** If the baseline on these refine-model curves already
reaches ≥ 43/48, the R1c failures came from the difference between `fim.py`'s model and refine's,
not from the search, and #25 is reclassified as a model mismatch.

**Cost.** Baseline is about 10 s per case; configurations with A run about 10 times longer. Timed on
one configuration first; the sweep is queued as [HEAVY] if it exceeds 2 minutes.

### Amendments (made after the first runs, with reasons)

1. *Reference for D\*.* After the baseline, the single factors B–F and A on `mid_oxide` had run, the
   lowest deviance any configuration had found (457, 566, 536) was above the deviance of the truth
   itself (300, 288, 323), so "lowest D found by any configuration" was not an optimum and would
   have counted fits worse than the truth as reached. D\* now also includes a reference fit that does
   not use refine's optimizer: bounded least squares on deviance residuals under refine's own forward
   model (central-difference Jacobian, the model is float32), started at the truth, within the
   global bounds. It gives 289.9, 279.0, 313.0, below the truth by 10 in each case, as expected for
   about a dozen fitted parameters. The reference run "ref" (refine, all on, ×30, from the truth) is
   kept as registered. Nothing else changes; no configuration with A had been run on more than one
   case when this was decided.
