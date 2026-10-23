# Fit policy: the decisions a person makes when fitting XRR, as an environment (X2a)

Written 2026-10-24, before any code or run. This note defines the environment, three baselines and
the pre-registered test that decides whether a learned policy (reinforcement or imitation learning)
is worth building. Nothing here has been run. "RL is not needed" is an acceptable outcome.

## 1. Question

A person fitting a multilayer XRR curve does not free every parameter at once. They decide how many
layers to use and when to add, merge or split one; which parameters to open and which to hold; whether
to judge the fit on log R, on linear R or by the counting likelihood; and when to stop. The question
for this stage is narrow:

> On synthetic single-, two- and three-layer curves, does a scripted version of that procedure end at a
> better model than fitting all parameters at once with the same compute, and does the best procedure
> differ from curve to curve?

Only if both answers are yes is there something for a learned policy to learn (section 6.4). The
learned policy itself (X2c, imitation or PPO) is not part of this note and needs Isaac's approval.

## 2. What is already known

From Experiment 001 (ReflectoLearn#24, draft, `research/identifiability`), #25, #31 and DECISIONS #17.
[fact] marks measured results, [inference] marks conclusions drawn from them.

- [fact] On curves from its own forward model, `refine` with the default stages reaches the Poisson
  maximum-likelihood fit from 0 of 48 starts; with the four changes of `RefineConfig.likelihood`
  (budget ×10, no correlation stage, Poisson deviance, decaying learning rate) 46/48 in sample and
  42/48 out of sample (R2b, R2f). The misses out of sample all start at d × 1.2.
- [fact] The log10-MSE optimum is not the maximum-likelihood point: with every search factor on and the
  log loss kept, all 16 starts of one film meet at a point ΔD ≈ 3.0×10³ from the ML fit (R2b).
- [fact] With the 2θ offset free in the correlation stage, a dense film without offset comes out with
  SLD up to 30 % low, along an offset–SLD valley with correlation 0.995 (#14, T2).
- [fact] The Kα2 ratio lies in a flat valley (Δχ² < 0.1 over 0–0.5) on noise-free 10⁶-count curves
  (R2a, conditions 1, 6, 36).
- [fact] Global multi-start (64 Latin-hypercube starts, Poisson least squares, 200 evaluations each)
  reached the truth's basin on only 5 of 18 bilayer conditions; 41–54 of 64 fits stopped at the
  evaluation cap (R2a). Where the basin was reached, every condition had one mode.
- [inference] Fitting everything at once is weakest exactly on multilayers, and the failures seen so
  far are failures to reach the optimum rather than several equally good optima. That is the regime
  where the order of decisions can matter, and also where more compute alone might close the gap.
  The comparison therefore has to hold compute equal (section 5.4).

Prior work. No paper found applies reinforcement learning to choosing an XRR layer model (web search
2026-10-24). The closest is Penkov, Peng and Fu, "Teaching an LLM agent to fit XRR curves with X-Ray
Calc 3", arXiv:2609.28926 (2026-09): expert decisions (normalising and trimming the curve, freeing
parameters, accepting a fit) written as a 13-step skill for an LLM agent, scored against expert fits
of W/B4C multilayers. Only the abstract was read; its method and baselines are unchecked. It does not
ask whether the procedure beats a global fit, which is the question here. Bayesian model selection of
the number of layers or free parameters is standard in neutron reflectometry (refnx: Nelson and
Prescott, J. Appl. Cryst. 52, 193 (2019); evidence-based choice of free parameters: arXiv:1910.10581,
authors unchecked). BIC is used here instead of the evidence for cost (section 4.6).

## 3. Forward model and data

The data come from the same model the fit uses (an inverse crime, see section 8). ReflectoLearn's
generator only builds one film with an optional top oxide, so X2b adds an N-layer stack on
`physics.abeles_core` and checks it against `benchmark.parratt`, as `scan.py` did for the bilayer
(3.1×10⁻¹⁵). When the shared XRR package (X1b) has its forward model, generator and metrics, X2b
switches to it.

- Stack: ambient / layers 1..L (top first) / Si substrate. Each layer has thickness d, roughness σ of
  its top interface (Névot–Croce) and real SLD ρ in 10⁻⁶ Å⁻². Substrate fixed and known:
  ρ = 20.07, σ = 3 Å. Absorption 10⁻⁹ as in `scan.py`. complex128.
- Instrument: intensity scale i0, background log10 b, constant Gaussian resolution σ_q (15-node
  Gauss–Hermite quadrature as `scan.py`), 2θ zero offset. Kα2 and footprint are left out of both the
  data and the model in this round (section 8).
- Grid: 2θ 0.2–6° in 0.02° steps, uniform in q, 291 points, Cu Kα1 1.5406 Å.
- Counts: y ~ Poisson(S · (i0 · (R ⊗ resolution)(q) + 10^b)), S = counts at R = 1.

### 3.1 Truths (registered)

For n ∈ {1, 2, 3} layers, M = 16 truths each:

| quantity | distribution |
|---|---|
| d_i | U[20, 250] Å |
| σ_i | U[2, min(10, d_i / 4)] Å |
| ρ_i | U[5, 80] ×10⁻⁶ Å⁻² |
| constraint | adjacent media (layer–layer, bottom layer–substrate) differ by \|Δρ\| ≥ 4; redraw otherwise |
| i0 | U[0.8, 1.2] |
| b | U[−8.5, −7.0] |
| 2θ offset | U[−0.01, 0.01]° |
| σ_q | U[0.0005, 0.003] Å⁻¹ |

The contrast constraint keeps the true layer count well defined: two adjacent layers of equal SLD are
one layer. Noise levels S ∈ {10⁸, 10⁶}. Seeds: truth j of class n uses
`np.random.default_rng((20261024, n, j))`; its fitted noise realisation for level s (0, 1) uses
`(20261024, n, j, s, 0)`, and the 20 extra realisations used for the noise scale (section 6.2) use
`(20261024, n, j, s, r)`, r = 1..20.

### 3.2 Held-out points

Points with 2θ < 0.5° (16 points on the q-uniform grid: the plateau, and the critical edge of films
with ρ ≲ 25) always stay in the training set. The rest are split into 10 contiguous blocks of equal
size (`np.array_split`, which governs); blocks 2 and 6 (0-based, q 0.1167–0.1551 and
0.2746–0.3116 Å⁻¹) are held out: 55 of 291 points, 236 for training. The critical edge of denser
films (up to 2θ 0.885° for ρ = 80) falls in block 0, which is training. Contiguous blocks are used because a single held-out point is almost determined by its
neighbours and would not test anything the training fit does not already see. Held-out points are never
used by the policy: not in the state, not in the loss, not in the choice of the returned model.

### 3.3 Fit bounds

d [5, 400] Å, σ [0, 15] Å, ρ [0, 120], i0 [0.5, 2], b [−10, −4], offset ±0.08°, σ_q [0, 0.012] Å⁻¹
(the instrument bounds are `RefineConfig`'s). At most L_max = 4 layers, at least 1.

## 4. Environment

### 4.1 State

Everything is computed on training points only.

- Stack: L, and for each of up to 4 layers (d, σ, ρ) scaled to [0, 1] by the bounds, plus a
  present/absent flag.
- Instrument values (i0, b, offset, σ_q), scaled.
- Open mask: one bit per (group, layer) for d, σ, ρ, and one per instrument group b, σ_q, offset.
  i0 is always open.
- Current loss (one-hot of log, raw, poisson).
- Residual summary: the training q range above the edge in 8 bands of equal q width; per band the
  mean and RMS of the Poisson deviance residual and the mean log10 residual (24 numbers). Dominant
  period and relative amplitude of the FFT of the log10 residual (a fringe the model misses), and
  of the data (`fft_thickness`, the Kiessig thickness). Critical q of the data and of the model.
- Fit quality: Poisson deviance per point D/n, the excess (D − E[D]) / SD[D] over its expectation at the
  current model (the quantity protocol A tests in step 7), and BIC (section 4.6) of the current state
  and of the best state so far.
- History: actions taken, evaluations used out of the budget, last action, number of layer changes.

### 4.2 Actions

Discrete; invalid ones are masked.

| action | effect |
|---|---|
| `add(p)`, p = 0..L | insert a layer at position p (0 = top). d = dominant period of the residual FFT if it lies in [20, 400] Å, else 50 Å; ρ = mean of the media directly above and below, the ambient excluded (top layer: the layer below); σ = 3 Å. Its d, σ, ρ open |
| `merge(i)` | layers i and i+1 become one: d = d_i + d_{i+1}, ρ = thickness-weighted mean, σ = σ_i |
| `split(i)` | layer i becomes two of thickness d_i / 2, ρ_i × 1.05 above and × 0.95 below, new interface σ = 3 Å |
| `open(g, ℓ)`, `close(g, ℓ)` | g ∈ {d, σ, ρ} for layer ℓ |
| `open(g, all)`, `close(g, all)` | the same group for every layer |
| `open(g)`, `close(g)` | g ∈ {b, σ_q, offset} |
| `loss(l)` | l ∈ {log, raw, poisson}: the objective of the next fits |
| `fit(k)` | k ∈ {20, 100} optimiser evaluations over the open parameters (4.3) |
| `revert` | return to the best-BIC state visited so far |
| `stop` | end the episode |

Initial values of new or changed layers are fixed rules of the environment, not choices of the policy,
so every policy starts a new layer the same way. The person's real freedom (where to place a layer and
what to free) stays in the action space.

### 4.3 Transition and losses

`fit(k)` runs `scipy.optimize.least_squares` (trust-region reflective, `x_scale="jac"`, bounds of
3.3, `max_nfev = k`) over the open parameters with residuals

| loss | residual per point |
|---|---|
| log | log10 λ − log10 max(y, 0.5) |
| raw | (λ − y) / S, the difference in linear R |
| poisson | sign(λ − y) · √(2 [λ − y + y ln(y/λ)]) (the deviance, zero counts included) |

with λ the expected counts and the Jacobian from `torch.func.jacfwd`, as in `scan.py`. The raw loss is
dominated by the plateau and the critical edge; the log loss weighs every decade equally; the Poisson
deviance weighs each point by its counts and is the likelihood.

This replaces the brief's "transition = k steps of `refine`" (section 7.1).

### 4.4 Returned model

Every method returns the state with the lowest BIC it visited. A person keeps the best fit they have
seen, and BIC needs no held-out data or truth, so this is allowed on measured data too.

### 4.5 Termination and budget

An episode ends on `stop`, after 60 actions, or when the evaluation budget E = 3000 is used. E counts
`nfev` of `least_squares`, the residual evaluations; rejected trust-region steps evaluate the residual
without the Jacobian, so `nfev` is not exactly "forward plus Jacobian". Jacobian evaluations (`njev`)
and wall time are reported. A `fit(k)` with fewer than k evaluations left runs with `max_nfev` equal
to what is left. The Jacobian cost grows with the number of open parameters, so at equal `nfev` C
(everything open) spends more compute than A1's staged fits; this is against A1, not for it. FFT,
residual summaries and the BIC of a state are not counted.

### 4.6 Reward

Defined now for X2c; the baselines do not use it.

- BIC = D + k ln n, with D the Poisson deviance of all training points (whatever loss the fit used),
  n the number of training points and k = 3L + 4 (every layer parameter and every instrument
  parameter, whether or not it was opened). AIC = D + 2k is reported. With n = 236, one more layer
  costs 3 ln n = 16.4, so a layer kept at ΔBIC ≤ −10 must lower the deviance by at least 26.4.
- Per step: r_t = γΦ(s_{t+1}) − Φ(s_t) with Φ(s) = −BIC_best(s)/n, the best BIC visited so far
  (amendment 3), so that with γ = 1 the rewards sum to the BIC of the returned model, not of the state
  the episode stops in. A potential-based term does not change which policy is optimal; it only gives
  credit earlier. γ is set when X2c is registered.
- At the end: r_T += −D_ho(ŝ)/n_ho for the returned model ŝ, D_ho the Poisson deviance on held-out
  points. A policy trained with this reward has seen the held-out blocks of its training curves, so
  for X2c, M2 is measured on new truths.
- The distance to the truth never enters the reward, because it cannot be computed on a measurement.
  It is used only in evaluation (6.1, M3).

## 5. Baselines

### 5.1 A: scripted human protocol

A1 is the primary script. Each rule has a stated source.

1. Scale. `loss(log)`, open i0 and b, `fit(20)`.
2. Critical edge → SLD. The critical q of the data q_c is where the counts first fall to half the
   plateau. If q_c exceeds the substrate's by more than 5 %, set the top layer's ρ = q_c² / (16π)
   (×10⁶), from q_c = 4√(πρ) (Als-Nielsen and McMorrow, *Elements of Modern X-ray Physics*, ch. 3);
   otherwise keep the start's ρ. The half-plateau point is not exactly the film's q_c and its error
   depends on d (physics review of #35): about 9 % high for 250 Å films, up to three times low for
   20–40 Å dense films whose edge tunnels. The rule gives ρ to about 10 % only for d ≳ 100 Å; it is a
   start value that step 4 frees.
3. Kiessig fringes → thickness. Set the total thickness to `fft_thickness` with ρ from step 2 and
   `d_min = 20` (its default 60 Å would miss the thinnest truths): the strongest real-space peak of the
   detrended log R transformed over the film wavevector √(q² − q_c²). Kiessig fringe period
   Δq ≈ 2π/d; Fourier analysis of the fringes: Sakurai and Iida, Jpn. J. Appl. Phys. 31, L113 (1992),
   reference unchecked. If it returns NaN, the start's d is kept. This step also removes the d × 1.2
   start failures of R2f by construction. In a stack the strongest real-space peak belongs to the
   interface pair with the largest contrast product, not necessarily to the total thickness, so for
   L > 1 `estimate_d` (amendment 2) can scale the total to a sub-layer thickness. At step 3 L = 1.
4. Structure. Open d and ρ of every layer, `fit(100)` twice.
5. Roughness last. Open σ, `fit(100)`.
6. Instrument last, likelihood finish. Open σ_q and the offset, `loss(poisson)`, `fit(100)` twice.
   The offset is held until here because of the offset–SLD valley (#14, DECISIONS #1), and the finish
   is Poisson because the log optimum is not the ML point (R2b, DECISIONS #17).
7. Layers only when a residual remains. If the deviance exceeds its expectation under the current
   model by three standard deviations, D > E[D] + 3 SD[D], with E and SD from 200 Poisson draws at the
   current λ (no forward evaluations; the χ² rule D/n ≈ 1 fails at the low counts of high q;
   the draws use `np.random.default_rng((20261024, 99, 0))` every time), try from the current best
   state. D at a fitted λ sits below D at the generating λ by about the number of open parameters, so
   the trigger is effectively 3.3–3.7 SD for 6–16 open parameters (physics review). Candidates: `add(0)`, `add(L)`
   (above the substrate), and `split` of the thickest layer. For each: open the new parameters,
   `loss(log)`, `fit(100)`, open all, `loss(poisson)`, `fit(100)`. Keep the best candidate if it
   lowers BIC by at least 10; otherwise go to 9.
8. Merge. For adjacent layers with |Δρ| < 2, try `merge`, refit as in 7, keep it if BIC does not rise.
   Return to 7.
9. Finish. Open all, `loss(poisson)`, `fit(100)`, `stop`.

Three variants, used only for the learning-room test (6.4):

- A2 no staging: steps 1–3 with `loss(poisson)` in step 1; then open everything, `loss(poisson)`,
  `fit(100)` five times (the number of fits in steps 4–6); then steps 7–9, where every refit of a
  candidate is: open everything, `fit(100)`, open everything, `fit(100)`, all with the Poisson loss.
- A3 log only: A1 with every `loss(poisson)` replaced by `loss(log)` (no likelihood finish).
- A4 different layer search: A1, but step 7 tries only `add(L)` and accepts at ΔBIC ≤ −2.

### 5.2 B: random actions

Uniform over valid actions; `stop` is masked for the first 5 actions and otherwise has the same
probability as any other action. Same budget, same returned-model rule. B shows how much of any gain
comes from the budget and the best-BIC return alone.

### 5.3 C: everything at once, with model selection

C first applies A1 steps 2–3 (`estimate_rho`, `estimate_d`) to the episode's one-layer start (5.4),
as a person fitting everything at once would also start from the edge and the fringes (amendment 3).
Then, for L = 1, 2, 3, 4: that layer over the first L − 1 extra layers, every parameter open,
`loss(poisson)`, one fit with `max_nfev` = E/4 = 750. Return the L with the lowest BIC. A win of A1
over C is then about the order of decisions, the question of section 1, not about the start.

Two references that do not enter the decision:

- C_oracle: C's start, the true L, everything open, Poisson, the whole budget E. It uses the truth.
- C_refine, single-layer class only: `refine` with `RefineConfig()` (the current default named in the
  brief) from the episode's start, film + oxide model as it is. It cannot represent two or three layers.

### 5.4 Starts and paired episodes

For each truth there are N = 4 starts, the same for both noise levels: (d, σ, ρ) of a one-layer model
from a Latin hypercube over d [20, 250], σ [2, 10], ρ [5, 80] (`scipy.stats.qmc.LatinHypercube`, seed
`(20261024, n, j, 2)`, amendment 1), instrument at i0 1, b −7, offset 0, σ_q 0.002. C's extra layers for L > 1 are the
layers of a second hypercube over the same box (12 points, 3 per start, seed `(20261024, n, j, 1)`),
in order. Every method sees the same truth, noise realisation and start; the
comparison is paired. 3 classes × 16 truths × 2 noise levels × 4 starts = 384 episodes per method.

## 6. Pre-registration

Registered 2026-10-24 before any implementation or run. If a result crosses a threshold, the threshold
is not moved; the result is recorded as a refutation (rule 20). Amendments, if any, are dated and
appended at the end of this note with the reason and whether results had been seen.

**Freeze (amendment 3).** The registration is frozen at the commit that runs the first registered
episode; every episode record carries that commit, and the results name it. After it, nothing in
sections 3–6 or in the definitions of 5.1–5.4 changes, in the note or in the code. A change there is
a new registration with new seeds.

### 6.1 Metrics

- M1 BIC of the returned model (training points).
- M2 held-out Poisson deviance D_ho of the returned model.
- M3 truth recovery, evaluation only: the layer count is right and every layer has |Δd|/d ≤ 5 % and
  |Δρ|/ρ ≤ 5 %. Also reported: layer-count accuracy and the median relative errors.
- M4 cost: evaluations used, actions taken, wall time.

### 6.2 Noise scale and per-episode outcome

For each truth and noise level, s_ho is the standard deviation of the truth model's held-out deviance
over the 20 extra noise realisations (3.1). For a method X against C on a paired episode:

- BIC: win if ΔBIC = BIC_X − BIC_C ≤ −10, loss if ≥ +10, otherwise tie. A difference of 10 is "very
  strong" on the Kass and Raftery scale (J. Am. Stat. Assoc. 90, 773 (1995)).
- Held-out: win if ΔD_ho ≤ −2 s_ho, loss if ≥ +2 s_ho, otherwise tie.

s_ho is the spread of one model's held-out deviance (≈ √(2 n_ho) ≈ 10.5 at high counts), not the
noise of the paired difference ΔD_ho, which goes to zero as two models coincide. The ±2 s_ho band is
therefore a fixed materiality threshold of about 21 deviance units, set by the truth before any
result, not a "within noise" band (wording corrected in amendment 3). At S = 10⁸ a BIC difference of
10 between fits with the same L can be a convergence difference inside one basin; under equal compute
that counts, and the report says how many wins are of that kind (same L as C).

**Truth-level outcome (amendment 3).** The four starts of a truth share its curve and noise, so the
unit of the tests is the truth. For a truth, the median over its four starts of ΔBIC and of ΔD_ho is
classified by the two rules above; that is the truth's BIC outcome and held-out outcome.

### 6.3 Does the procedure matter

A cell is one (layer class, noise level); there are 6 cells of 16 truths × 4 starts. In a cell,
**A1 beats C** when all three hold at the truth level:

1. A1 wins on BIC on at least 25 % of truths (4 of 16);
2. a one-sided sign test of truth wins against truth losses (ties dropped) gives p < 0.01: at least
   14–0 … 14–2 of 16 (13 of 16 without ties gives 0.011); with M = 8 truths (6.6) only 7–0 or 8–0;
3. A1 loses on held-out on at most 10 % of truths (1 of 16; 0 of 8).

The two noise cells of a class share their truths and starts, so "two of four cells" below can be the
same truths at two noise levels; the result names the cells.

Verdict over the four multilayer cells (2 and 3 layers, both noise levels):

- **No room for RL** if A1 beats C in none of them. X2c is not registered. This is the falsifying
  outcome for "a human-like procedure helps".
- **Procedure matters** if A1 beats C in at least two of them.
- **Inconclusive** if it beats C in exactly one. X2c is not registered; a confirmation with new seeds
  is proposed instead.

The single-layer cells are reported but do not enter the verdict.

### 6.4 Is there something to learn

"Procedure matters" alone does not justify a learned policy: a fixed script might be all that is
needed. In the cells where A1 beats C, let the oracle be the per-episode best of A1–A4 by BIC and the
best single variant the one with the lowest median BIC in that cell. There is **room to learn** when
both hold in at least one such cell:

- the oracle beats the best single variant on BIC (≤ −10) in at least 20 % of episodes;
- no single variant is the per-episode best (ties within ΔBIC 2 count for every tied variant) in more
  than 70 % of episodes.

These are fractions of episodes, not tests; episodes of one truth are not independent, and the result
also gives both fractions over truths (median over starts) for reference.

X2c is proposed to Isaac only for "procedure matters" with room to learn. "Procedure matters" without
room to learn is reported as "use the script" (it then becomes a `refine` option candidate, not RL).

### 6.5 Guards

- **Budget (defined in amendment 3).** If more than half of the 64 returned C fits in a cell (the fit
  of the L that C returns) end with `least_squares` status 0, i.e. `nfev` reached `max_nfev`, the cell
  is budget-limited: C is rerun there with 2E and the verdict for that cell uses C(2E). Otherwise A's
  advantage could just be C running out of evaluations. A1, B and the A1–B comparison stay at E.
- **Reward alignment.** In a cell where A1 beats C, compare truth recovery (M3) of A1 and C by an
  exact one-sided McNemar test on the paired episodes. If A1 recovers the truth significantly less
  often (p < 0.05), the reward rewards the wrong thing: X2c is not registered until the reward is
  redefined and re-registered. Episodes of one truth are not independent, which makes this test flag
  misalignment more easily, the conservative direction for registering X2c.
- **Random floor.** A1 against B uses the same truth-level rules (6.3). If A1 does not beat B in the multilayer
  cells, the report says that an advantage over C comes from the budget and the best-BIC return, not
  from the order of decisions.
- With 6 cells, p < 0.01 per cell keeps the family-wise error near 0.05.

### 6.6 Pilot and cost

No pilot has been run. Before the registered run, X2b times one episode per method on truths outside
the registered set, recording only wall time and peak memory, not the metrics. (The seed first
written here, `(20261024, 9, 0)`, would have been a nine-layer truth; the timing uses truth
`(20261024, n, 1000 + n)` of each class, noise level 1, and the starts of `(20261024, 9, 0, 2)`.
Amendment 3.) [inference, unverified] A `scan.py` bilayer fit took 4–9 s per 200 evaluations at 6° with Kα2
(twice the forward cost of this model), so an episode of 3000 evaluations may take 30–70 s and the
2688 episodes of A1–A4, B, C and C_oracle 20–50 CPU-hours. If the timing projects more than 6 hours
wall at 8 processes, M is reduced from 16 to 8 truths per class before the run, decided from timing
alone, and recorded as an amendment. Thresholds are percentages and do not change.

## 7. Departures from the brief

### 7.1 The transition is not `refine`

The brief says the transition runs `refine` for k steps. It does not, for two reasons. `refine`
parameterises one film plus a top oxide and cannot represent two or three layers. And its Adam stages
stop short of the optimum even on curves from its own model (0/48 by default, 42/48 with the
likelihood preset out of sample), so a comparison built on it would measure the optimiser, not the
procedure. Trust-region least squares on the Poisson deviance reached the ML fit from 43/45 starts on
the same curves (R1c) and is what `scan.py` already uses. `refine` keeps a reference role (C_refine,
single layer).

### 7.2 C chooses its layer count

The brief's C is the current full fit. Given the true layer count it would know something neither a
person nor A1 knows; given a fixed count it would be wrong on two of the three classes. C therefore
fits every L with the same budget and picks by BIC, the strongest simple baseline. Beating a weak
baseline would overstate A. C_oracle shows what the true count is worth.

### 7.3 Data source

ReflectoLearn's generator has no multilayer stacks, so the data come from the new stack model on the
same `abeles_core` (section 3) until X1b.

## 8. Limits

- Inverse crime: data and fit share one forward model, so no model error is tested. Model error is where
  a person's judgement may matter most (an unmodelled oxide, graded interfaces). A run with data from a
  richer model (oxide below the film, DECISIONS #19, Kα2, footprint) is a later step, after X1b.
- Box layers with Névot–Croce roughness; constant resolution; known substrate; no Kα2 or footprint.
- Synthetic only. No measured data are used until their publication is decided (DotFiles#83, #84).
- The protocol is one reading of common practice, written down by the authors of this note, not
  elicited from a person fitting curves. A1–A4 span a few choices, not the space of human practice.
- BIC assumes the model is correct and n large; with 236 training points and k up to 16 it is an
  approximation.
  The evidence would be better and costs far more.

## 9. Plan for X2b

- `src/reflectolearn/fit_policy/`: `stack.py` (N-layer forward, residuals, BIC, block split),
  `env.py` (state, actions, transition, reward), `protocol.py` (A1–A4), `baselines.py` (B, C,
  C_oracle).
- `experiments/002_fit_policy/run.py` (episodes, resumable JSON per cell), `report.py` (tables, figures,
  verdict by section 6 only).
- Tests: stack forward against `benchmark.parratt` (≤ 1e-12), a one-layer stack equal to
  `refine.make_forward` with the oxide, Kα2 and footprint off and the same instrument values, Poisson residual squared summing to the deviance, BIC on a
  hand-computed case, action masks, `merge(split(x))` restoring x's thickness and SLD, a seeded
  episode reproducible.
- No GPU, no training. The full run is [HEAVY]; X2b asks for a slot with the timing of 6.6.

## 10. Review requests

physics (rule 22):

1. The q_c → ρ rule (q_c = 4√(πρ)) and the Kiessig rule as used in steps 2–3, and the trigger
   D > E[D] + 3 SD[D] (Monte Carlo at the current λ) in step 7.
2. Whether |Δρ| ≥ 4 ×10⁻⁶ Å⁻² makes the true layer count well defined at 2θ ≤ 6° and S ≥ 10⁶.
3. The held-out design (contiguous blocks 2 and 6 above 0.5°) and the 2 s_ho noise band.
4. BIC with the Poisson deviance and k = 3L + 4; the ΔBIC 10 threshold.

xrr-kit (what X2b needs from the shared package, so that switching is an import change):

- An N-layer forward model, vectorised over parameter sets, float64, differentiable in torch, with
  i0, background, constant resolution and 2θ offset (Kα2 and footprint optional), checked against
  Parratt.
- `deviance(lam, y)` (Poisson, zero counts), `residuals(kind, lam, y, scale)` for log, raw and
  poisson as in 4.3, `information_criteria(deviance, k, n)` returning AIC and BIC.
- `block_split(q, n_blocks, held, q_min)` for contiguous held-out blocks.
- A seeded stack generator with the constraints of 3.1 (adjacent contrast, σ ≤ d/4).
- Estimators `critical_q(q, y)` and `kiessig_thickness(q, y)` (the latter a port of `fft_thickness`).

## Amendments

1. 2026-10-24, before any code ran or any result existed. The starts' hypercube seed was
   `(20261024, n, j)`, the same seed as the truth's generator, so the starts would have been drawn from
   the same random stream as the truth parameters. It is now `(20261024, n, j, 2)`. Found while writing
   the generator for X2b.
2. 2026-10-24, while writing the environment, before any episode ran. Three gaps between section 4
   and protocol A:
   - Steps 2 and 3 of A1 set values from the data, which no action in 4.2 could do. Two actions are
     added: `estimate_rho` sets the top layer's ρ by the q_c rule of step 2 (no change if q_c is
     within 5 % of the substrate's), and `estimate_d` scales every layer's d so that their sum is the
     Kiessig thickness of step 3 (no change if it is NaN). Both are available to every policy.
   - Step 7 tries candidates and keeps the best, which needs branching. Protocol A runs each candidate
     on a copy of the state. Copies share the episode's evaluation budget and its best-state record,
     so every evaluation counts and the returned model is still the lowest BIC visited anywhere; the
     60-action cap counts the actions on the kept path. A policy limited to the action space can only
     go back through `revert`; this asymmetry favours A and is reported with the results.
   - The Kiessig search range is 20–800 Å, since three-layer truths reach 750 Å in total.
3. 2026-10-24, after the physics review of #35 (review 5631065066, FORUM T11), before any registered
   episode ran; the only runs were the timing pilot of 6.6 (time and memory, no metrics) and debugging
   runs on unregistered truths. Changes, each in the section it concerns:
   - Truth-level tests (6.2, 6.3): the four starts of a truth are not independent; outcomes and the
     sign test are per truth (median over starts). Thresholds for M = 8 stated.
   - Budget guard defined on the 64 returned C fits (6.5).
   - C applies `estimate_rho` and `estimate_d` to its start (5.3, also C_oracle), so that A1 against C
     tests the order of decisions, not the start.
   - Freeze at the first registered episode (6); A2 written out step by step (5.1); the Monte Carlo
     seed of step 7 registered; `nfev` as the budget unit, `njev` and wall time reported, the last fit
     truncated to the remaining budget (4.5).
   - Wording: the ±2 s_ho band is a materiality threshold, not a noise band (6.2); held-out block
     ranges and the 2θ < 0.5° description (3.2); the q_c rule bias, the `estimate_d` caveat for L > 1
     and the trigger conservatism (5.1).
   - Reward for X2c (4.6): Φ on the best BIC so far, γ stated at X2c, M2 on new truths for X2c.
   - Timing truths (6.6): the seed first written would have been a nine-layer truth.
