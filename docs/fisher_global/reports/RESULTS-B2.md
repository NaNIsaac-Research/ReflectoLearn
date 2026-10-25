# B-2 results: truth-start local profile of R2a bilayer #25

xrr-policy lane, 2026-10-25. Registration: `PREREG-HF-HW2.md` section 5 (ReflectoLearn
`research/fisher-global` d1f164c). Code b544baf (`experiments/005_fisher_global/b2.py`, committed before
the run); output 92d1983: `docs/fisher_global/b2.json`, `b2_log.txt`, and the post-hoc
`b2_extra.py` / `b2_extra.json`. This is an observation (R2a amendment 7, never run in R2a); it tests
no hypothesis.

## 1. What was run

- Condition #25: Si / film 2 (120 Å, σ 6, ρ 30) / film 1 (240 Å, σ 6, ρ 24) / oxide (10 Å, ρ 18.9,
  σ 3) / air, refine nuisances, 2θ ≤ 6° (291 points), noise-free counts at 10⁶, 15 free parameters.
- `scan.fit` from the truth (cap 200), then `scan.half_width` (Δχ² = 1, three restarts per profile
  point) for d₁, ρ₁, d₂, ρ₂, as R2a's amendment 7 code.
- [fact] Wall 451 s, one process, one torch thread, BelowNormal, 16 000 forward calls (including
  Jacobian evaluations). Inside the boss slot (20–30 min) and the 45 min cap.
- [fact] Memory was **not measured**: the wrapper read the peak working set of the venv's
  `python.exe` launcher (5 MB), not of the interpreter it starts. The same functions ran in R2a's
  `analyze` under 1.5 GB [estimate in RESEARCH-DESIGN 2.2]; nothing here was checked against 2 GB.
- Nothing was written under `docs/identifiability/`; #25's R2a label stays "not reached".

## 2. Facts

- [fact] The truth-start fit ends at D = 2.7 × 10⁻¹⁰ after 14 evaluations, with every parameter
  within rounding of the truth.

| parameter | value | Δχ² = 1 half-width | relative | label (≥ 2 % = ridge) | Fisher √(F⁻¹)_kk | profile / Fisher | at bound | lowest Δχ² |
|---|---|---|---|---|---|---|---|---|
| d₁ (film 1) | 240 Å | 5.38 Å | 2.24 % | **ridge** | 12.6 Å | 0.43 | no | 0 |
| ρ₁ | 24.0 | 0.064 | 0.27 % | identifiable | 0.078 | 0.82 | no | 0 |
| d₂ (film 2) | 120 Å | 0.149 Å | 0.12 % | identifiable | 0.307 Å | 0.49 | no | 0 |
| ρ₂ | 30.0 | 0.056 | 0.19 % | identifiable | 0.088 | 0.64 | no | 0 |

- [fact] No profile restart found a point below the truth-start endpoint (lowest Δχ² 0 in all four).
- [fact, post hoc, `b2_extra.json`] Under the unbounded quadratic, the profile point at
  Δθ_k = ±√((F⁻¹)_kk) moves the Kα2 ratio by 2.1 to 4.3 bound widths, to between −1.3 and 3.1 (bounds
  0–0.6), in all eight cases. For d₁ it also takes the oxide thickness below 0 or the oxide SLD outside
  15–22. At those points the forward-model deviance is 356 to 4.0 × 10⁴ instead of the quadratic's 1;
  for d₁ − h it is undefined (a negative Kα2 ratio gives negative counts).

## 3. Interpretation [inference]

1. **Inside the truth's basin, #25 is locally well determined except for d₁.** Film 2 and both SLDs
   are pinned to 0.1–0.3 %; film 1's thickness has a 2.2 % half-width, just over R2a's ridge line.
   This is the physical-identifiability side of #25 that R2a could not label because no start
   reached the basin. It does not say whether another basin within Δχ² ≤ 1 exists (condition 3); the
   recorded best endpoints of #25 (oxide–film-1 thickness exchange) remain at ΔD ≈ 1.8 × 10³
   (RESEARCH-DESIGN 2.1), i.e. S₁ ≈ 560 counts.
2. **The unbounded Fisher bound is wider than the real local profile, because its weakest direction
   runs through unphysical Kα2 ratios.** The bounded profile is 0.43–0.82 of √((F⁻¹)_kk). F's
   weakest direction trades the film parameters against the Kα2 ratio, and the bounds stop that trade
   long before Δχ² = 1. This is the converse of B-1's finding at the wrong endpoints (there Fisher
   overstates the deviance). Together they show the quadratic at the truth misstates both sides: it
   is too pessimistic about local precision when a nuisance is bounded, and too optimistic about how
   far away other solutions are. R1b already reported the bound effect for a single film with free
   oxide (CRLB of d 5 % against a bounded profile of ±0.4 %).
3. **The d₁ ridge is in the oxide–film-1 direction.** The d₁ profile is the only one whose Fisher
   point also pushes the oxide thickness and SLD out of bounds, the same trade that the best wrong
   endpoints of #25 make on a large scale (film 1 ≈ 200 Å under a 50–60 Å oxide). This links the local
   ridge and the distant basin by direction. It is a reading of directions, not a demonstration that
   the basin is connected to the truth's.

## 4. Limits

- One truth-start fit on noise-free data; one condition.
- Profile points are local fits with three restarts and at most 100 evaluations each. Such a profile
  bounds the profile likelihood from above, so the half-widths can only be too narrow, not too wide.
- Memory not measured (section 1).
- No statement on global uniqueness, multimodality, #25's R2a label or the generative no-go.
