# X2d draft: why protocol A1 adds layers that do not hold out

Draft, 2026-10-24, written after the X2b result (ReflectoLearn#36). Nothing here is registered and
nothing has been run. Before registration, physics writes the conditions that keep this from being
a hypothesis fitted to X2b's data (HARKing). Those conditions go in section 5 and override anything
below. The registration then uses new seeds; X2b's truths, noise and starts are not reused.

## 1. What X2b showed, and the status of each statement

- [registered] A1 beats C on BIC in every multilayer cell, and fails the held-out rule in every one
  (2–3 of 16 truths lose by more than 2 s_ho). Verdict: no room for RL.
- [post hoc, X2b data] In 8 of the 10 losing (truth, cell) pairs, A1 returned 4 layers, the
  maximum, on every start, for two- and three-layer truths. A1's layer count is right in 50 % of
  two-layer episodes and 11–12 % of three-layer ones.
- [post hoc] C's fits stop far below their allowance, and C_oracle recovers almost nothing, so C
  says little about what a good fit of the right structure would do.

The post hoc statements are the reason for this draft. They are not evidence for any of the
hypotheses below, which have to be tested on new data.

## 2. Candidate mechanisms

- **H1, unmatched refit.** In step 7 a candidate is refit with two `fit(100)` calls and compared with
  the current state, which is not refit. A drop of 10 in BIC can then come from the extra fitting
  rather than from the extra layer. Check: refit the current state with the same calls before the
  comparison.
- **H2, a wrong basin compensated by layers.** If the two-layer fit sits in a wrong basin, extra
  layers can lower the training deviance by reshaping the profile, without describing the sample.
  With no model error and S = 10⁸, noise alone is an unlikely source: adding 3 parameters gains
  more than 26.4 by chance with probability about 1e-5 (χ²₃). Check: run A1 from the truth (an
  oracle start, diagnostic only) and see whether it still adds layers.
- **H3, the held-out split itself.** The two contiguous held-out blocks remove q ranges from the
  training data, and structures that differ only there are not constrained. Check: on the same
  truths, compare the held-out deviance of the true-L fit started at the truth with that of the
  4-layer fit. If the truth-started fit also loses, the loss is a property of the split, not of A1.

These are not exclusive.

## 3. Candidate protocol changes (to be compared, not adopted)

- **A1m, matched refit (H1).** Before step 7 compares a candidate, the current state gets the same
  refit (`loss(log)`, `fit(100)`, open all, `loss(poisson)`, `fit(100)`), and the comparison uses the
  refit state.
- **A1v, inner validation (H2, H3).** Step 7 accepts a candidate only if it also lowers the Poisson
  deviance on an inner contiguous block of the training points that was left out of the
  candidate's fit. The outer held-out blocks stay unused.
- **A1o, oracle start.** A diagnostic only, never a protocol: A1 started at the truth.

## 4. Sketch of the comparison

New truths and noise, with seeds not used in X2b (for example a class offset of 100 in the truth seed
tuple). Same classes, noise levels, grid, held-out blocks and budget as X2b, so that only the protocol
changes. Primary questions, each needing a threshold set before the run:

1. Does A1m or A1v remove A1's held-out losses against C (rule 3 of X2b) without losing rules 1–2?
2. Does the layer-count accuracy of A1m or A1v exceed A1's?
3. Does A1o add layers (H2) or lose on held-out (H3)?

If a changed protocol then passes all X2b rules against C, the learning-room test of X2b section 6.4
applies again before any RL is proposed.

## 5. HARKing conditions (physics, to be written before registration)

(empty)
