# Poster draft: Sim-to-real domain randomization for single-layer XRR

Draft for Isaac to review. Source of every number: `ReflectoExp/docs/exp02_vs_exp07_report.md`
(Korean, sections noted in brackets). Nothing was re-run for this draft. Figures to reuse from this
repository: `docs/domain_randomization.png`, `docs/model_architecture.png`, `docs/sim_to_real.png`.

Title note: the experiments are simulation only (the report's own limit). "Sim-to-real" here names
the design (the simulator reproduces the measurement chain and the input accepts any real q-grid),
not a demonstrated transfer to measured data. The poster must say so in the first panel.

## Title

Sim-to-real domain randomization for single-layer X-ray reflectivity: what a neural parameter
regressor needs

## 1. Problem

- Input: an X-ray reflectivity (XRR) curve of a single thin film on a Si substrate.
- Output: film thickness, roughness and scattering length density (SLD).
- Question: which design choices decide whether a network trained on simulated curves works on
  curves that carry measurement effects?

## 2. Two designs compared [section 0, 1, 3]

| | exp02 | exp07 |
|---|---|---|
| q range (A^-1) | 0.0071 to 0.4271 | 0.0036 to 1.0651 |
| q points | 200 | 2000 |
| thickness range resolvable from the grid | 15.0 to 1489 A | 5.9 to 5916 A |
| training thickness labels | 5 to 200 A | 10 to 1200 A |
| parameters | 1.37 M | 5.30 M |
| training data | fixed, noise added once | stochastic physical augmentation on the fly |

Thickness limits come from d_min = 2 pi / (q_max - q_min) (one Kiessig fringe in the window) and
d_max = pi / delta_q (two points per fringe). In exp02 about 5 % of the label range (5 to 15 A)
lies below d_min.

## 3. Domain randomization follows the measurement chain [section 4]

Simulation is noise-free; noise is applied at training time in the order of a real measurement:

| step | effect | range |
|---|---|---|
| 1 | beam footprint | beam width U(0.005, 0.015) mm, sample length U(5, 30) mm |
| 2 | instrument resolution (Gaussian smearing) | sigma_q U(1e-4, 6e-3) A^-1 |
| 3a | incident intensity | x U(0.75, 1.25) |
| 3b | background | 10^U(-8, -5) |
| 3c | counting statistics (Poisson) | s = 10^U(5, 8) |
| 4 | q-axis misalignment | N(0, 0.004) A^-1 |

Hidden nuisance variables (surface SiO2 thickness, density and roughness; substrate roughness) are
randomized but not regressed. Inputs are normalized by the maximum, log10, resampled to a target
q-grid, with a valid-mask channel for q values that were not measured, so a measured `.dat` with any
angular step can be fed in [section 5].

## 4. Model [section 2]

1D convolutional encoder (6 blocks, 64 to 512 channels) plus an MLP head, 5.30 M parameters.
Input: log10 R and the valid-mask, plus Random Fourier Features of q (32 frequencies, scale 15).

## 5. Ablation protocol [section 7.2]

One dataset (50,000 curves), five variants, the same optimizer-step budget (93,750 steps, 2 % of the
full 4,687,500-step budget). Equal steps, not equal epochs, so that augmentation is not credited for
extra compute.

| variant | q points | depth | Fourier | augmentation | parameters |
|---|---|---|---|---|---|
| A_full | 2000 | 6 | yes | yes | 5.30 M |
| B_no_fourier | 2000 | 6 | no | yes | 5.27 M |
| C_no_augment | 2000 | 6 | yes | no | 5.30 M |
| D_lowres_200 | 200 | 4 | yes | yes | 1.63 M |
| E_exp02_like | 200 | 4 | no | no | 1.60 M |

## 6. Results [section 7.4]

Thickness MAE (A) and R^2, evaluated on two simulated test sets.

| variant | clean: MAE | clean: R^2 | with measurement effects: MAE | with measurement effects: R^2 |
|---|---|---|---|---|
| A_full | 11.81 | 0.9976 | 47.83 | 0.9256 |
| B_no_fourier | 10.19 | 0.9980 | 51.75 | 0.9226 |
| C_no_augment | 3.54 | 0.9998 | 279.66 | -0.514 |
| D_lowres_200 | 24.22 | 0.9645 | 85.36 | 0.8271 |
| E_exp02_like | 6.91 | 0.9966 | 392.58 | -3.305 |

Main message: on clean simulation the variant without augmentation looks best (3.54 A against
11.81 A for the full model); on data with measurement effects it is 5.8 times worse and its R^2 is
negative. A clean-simulation metric alone does not predict performance under measurement effects,
because it evaluates a model on the distribution it was trained on.

Error growth from clean to measurement-effect domain, thickness: A 4.1x, B 5.1x, C 78.9x, D 3.5x,
E 56.8x.

Findings, in the order the report ranks them:

1. Physical augmentation dominates (A against C).
2. Higher q resolution helps, about 1.8 to 2.1 times lower thickness MAE in both domains (A against
   D). It is confounded with model depth and capacity (depth 4 and 1.63 M against depth 6 and
   5.30 M), so the resolution effect alone is not separated.
3. Fourier features: no contribution can be shown at this budget (differences within 8 %, sign
   reverses between domains). Do not present it as a result.

## 7. Limits (must be on the poster)

- Simulation only. The "measurement effects" test set is generated with the same augmentation as
  training for A, B and D (in distribution) and out of distribution for C and E. It tests behaviour
  under the designed conditions, not independent measured data. Validation on measured XRR is open.
- The budget is 2 % of the full one; read the variants relatively, not as final accuracy.
- exp02 data and checkpoints were lost; E_exp02_like approximates exp02 on the exp07 code, it is not
  a rerun of exp02.
- Resolution and model capacity are not separated (a depth 6, 200-point variant is missing).

## 8. Take-away (one line)

Train on a simulator that reproduces the measurement chain and report on both clean and
measurement-effect data; clean-simulation error alone can rank the models in the wrong order.

## Open points for Isaac

- Whether to keep "sim-to-real" in the title while no measured data is validated.
- Whether to show the full numbers (section 6 table) or only A, C, E on the poster.
- Authors, affiliations and the venue are not filled in.
