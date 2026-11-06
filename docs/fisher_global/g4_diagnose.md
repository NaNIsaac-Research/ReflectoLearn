# G4 follow-up: why the two forwards differ

Generated numbers: `experiments/005_fisher_global/g4_diagnose.py` (forward only, no fit, 13 s) writes
`g4_diagnose.json`. It takes the 10 items of `g4_sim_forward.json` with the largest relative difference in
ΔD (6), then the largest difference in counts (3), and the X2e exception, and evaluates each with
(a) the engine of the recorded results, (b) the ReflectoSim forward, (c) the engine's own physics with the
resolution integral done on a fine uniform grid (401 points over ±8σ, with a check at 801 points), and
(d) the engine with 15, 31, 61 and 101 Gauss-Hermite nodes.

## Result

- The engines share their physics: Parratt recursion, roughness, 2θ offset, Kα₂ at the same angle,
  footprint, absorption. Only the resolution integral differs. The engine of the recorded results
  integrates the Gaussian resolution (constant σ in q) with a fixed 15-node Gauss-Hermite rule; ReflectoSim
  integrates it with per-point nodes chosen from the fringes and the critical edge.
- A fixed Gauss-Hermite rule does not converge for these curves: the deviance of the X2e exception is 48
  (15 nodes), 6.8 × 10⁴ (31), 245 (61), 1.8 × 10⁴ (101), 8.6 × 10³ (uniform grid, 801 points). The
  integrand has fringes and a critical edge that nodes spaced about one σ apart do not sample.
- The uniform-grid integral converges (401 against 801 points: ΔD within 7 % in all ten items) and agrees with ReflectoSim: ΔD of the X2e exception 8648 against 8604,
  of R2a condition 33 start 7 1.394 × 10⁵ against 1.395 × 10⁵, of condition 25 start 55
  1.587 × 10⁷ against 1.587 × 10⁷. The engine of the recorded results gives 48, 1.98 × 10⁴ and
  1.49 × 10⁷. Its largest relative error in counts over a curve (against the 401-point uniform grid) is up to 0.09 at
  the truth and 0.15 to 0.96 at wrong states; the 801-point grid differs from the 401-point one by at most
  0.005, and ReflectoSim by at most 0.008, except at single points of one R2a endpoint (condition 25, start
  55: 0.17, a deep minimum, ΔD agrees).
- So the cause is the 15-node resolution quadrature of the engine (`scan.expected_counts`,
  `fit_policy.stack.expected_counts`), not a difference in physics. Kα₂, footprint, background, absorption and
  the roughness model give the same curves in both to 0.5 % or better once the resolution integral is
  converged; the one model difference (ambient absorption 0 against 1e-9 Å⁻²) is negligible.
- ReflectoSim is the right forward. The recorded fits are exact minima of the engine's own model, whose
  instrument function is 15 weighted lines instead of a Gaussian; truth and fits share it, so the
  results are internally consistent for that model but are not the numbers of the Gaussian-resolution
  model.

## What moves

- The five R2a basins with S₁\* ≥ 10⁴ and all R2a threshold crossings: no change (G4).
- The one X2e basin with S₁\* ≥ 10⁴ (ΔD 48 in the engine): in the converged model ΔD is 8.6 × 10³, S₁\* about
  116. It is not an exception. The count of basins with S₁\* ≥ 10⁴ becomes 5 of 488 (R2a only).
- Medians of S₁\* of wrong minima: 26.7 to 22.4 (R2a), 44.2 to 42.2 (X2e).
- Not recomputed: the Fisher quadratic (its Jacobian is the engine's), the basins that a fit with the
  converged forward would find (a refit may end elsewhere), and noise-free separations at 10⁸ counts for
  X2e beyond the minima evaluated.
- Cost: the ReflectoSim forward took 188 to 565 ms per curve against 2.6 to 7.9 ms for the engine (70
  times), so refits on it are not affordable at the planned scale.
