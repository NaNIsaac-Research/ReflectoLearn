# X2e draft: does a better start basin remove A1's extra layers?

Draft, 2026-10-24, written after the X2d result (ReflectoLearn#39). Nothing here is registered and
nothing has been run. Before registration, physics sets the conditions against hypothesising after
the results, as for X2d, and those override this text. The registration uses a seed root that no
earlier run used.

## 1. What is known, and from where

All of it is stated in #39. No X2d record is analysed further for this draft.

- [registered, X2d] H2 is graded supported and is the cause under X2d's rules. In 42 of 42 truths
  where A1 overshoots the layer count, its kept-path state at the true layer count has a training
  deviance more than 26.4 above that of A1o, the same protocol started at the truth. A1o never
  overshoots (E_A1o = 0 in every multilayer cell).
- [registered, X2d] H1 (unmatched refit), H3 (the held-out split) and H4 (best-BIC return) are
  refuted. The matched-refit and kept-path changes did not change A1's truth-level outcomes.
- [registered, X2d] X2b's held-out failure replicated in 2 of 4 multilayer cells on new seeds.
- [inference] A1 reaches the true layer count in a wrong basin, and the BIC gain of a further layer
  then comes from compensating that basin. The question moves from the order of decisions to where
  the search starts.

## 2. Where the wrong basin can come from

- **B1, the new layer's initial values.** `add` puts a layer at the residual's dominant period,
  with the mean SLD of its neighbours and σ = 3 Å (DESIGN 4.2). If that start lies in the wrong
  basin of the (L+1)-layer problem, the refit stays there.
- **B2, the one-layer fit.** A one-layer model fitted to a multilayer curve is a compromise
  (thickness near the strongest interface pair, SLD between the layers). Adding layers to it starts
  every later fit from that compromise.
- **B3, both, or neither.** Local fits at any layer count may simply lack the reach to leave the
  basin they start in. Then only a broader search at each layer count helps, and that is a compute
  question, not a procedure question.

## 3. Candidate start improvements (to be compared, not adopted)

Each one replaces a part of A1 and keeps the rest (steps, losses, the −10 margin, the returned-model
rule) as in X2b.

- **S1, several starts for a new layer (tests B1).** For each `add` candidate, start the new layer
  from the three strongest peaks of the residual's real-space transform (instead of one), and each
  peak with ρ at the neighbours' mean and at ±25 % of it. That gives up to 9 starts, each refit as a
  candidate. The best is kept and every evaluation is charged to E.
- **S2, a stack-level start from the fringes (tests B2).** Before any layer is added, read up to
  three interface-pair distances off the data's real-space transform (physics noted on #35 that the
  peaks belong to interface pairs, not only to the total thickness). Try the stacks they imply
  (L = 1, 2, 3 with thicknesses from the peak set) as starting points of A1's step 4. Keep the one
  with the lowest BIC after steps 4–6, then run step 7 as A1.
- **S3, a bounded global search at each layer count (tests B3).** Replace step 7's local refit of a
  candidate by a differential evolution over that candidate's layer parameters within the fit
  bounds, with a fixed share of E, polished by the usual `fit(100)`.

S3 changes the compute as well as the procedure. If it is kept, C needs the same global search at
each L with the same share of E (C_DE), so that a win is not just more search.

## 4. Draft hypotheses and what would refute them

Thresholds are proposals, to be fixed in the registration. Unit and rules as in DESIGN 6.2 and
X2D section 5.

- **B1.** Prediction: S1 lowers E (truths whose median returned L exceeds the true L) against A1 by
  the overshoot sign test of X2d amendment 1d in ≥ 2 of 4 multilayer cells, and its kept-path state
  at the true L falls within 26.4 of A1o's in ≥ 50 % of the truths where A1 overshoots. Refuted if
  neither holds in any cell.
- **B2.** The same predictions for S2. Refuted the same way.
- **B3.** S3 lowers E against A1 as above, while C_DE also loses its gap to A1o (C_DE's returned L
  equals the true L as often as S3's). Refuted if S3 does not lower E.
- **Protocol verdict.** One primary arm, named before the run, against C (and C_DE if S3 is the
  primary) under DESIGN 6.3 unchanged. A pass leads to a confirmation run on a further seed root;
  X2c stays unregistered.

## 5. Arms and controls (draft)

A1 (replication control), S1, S2, S3 if kept, C, C_DE if S3 is kept, A1o (truth start, diagnostic),
B (random floor). Same classes, noise levels, grid, held-out blocks, budget and L_max as X2b.

## 6. Open points for physics

- Whether S2's peak-to-thickness reading is sound enough to be a protocol step, or should remain a
  diagnostic.
- Whether S3 belongs in this study at all, since it mixes compute with procedure.
- The seed root: 20261026 was reserved for X2d's confirmation run, which is not being run
  (boss, 2026-10-24); proposing 20261027 for X2e so that no root is reused.
- HARKing conditions (to be written before registration).
