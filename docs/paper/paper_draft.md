# Paper draft: identifiability and search reach in X-ray reflectivity fits

Draft, 2026-11-06. Nothing here is new computation: every number is read from a result file in this
repository, and each claim carries the status that [FINDINGS.md](../FINDINGS.md) gives it. Status
words: **checked** (written verification cited in FINDINGS), **supported** (a registered hypothesis
the registered rule confirmed), **draft** (a result without independent verification), **hypothesis**
(registered, no verdict, or stated but not tested), **prior art** (found in the literature read for
this work). Journal choice and authorship are open. Inference is marked as inference.

## 0. Title options and the claim

1. Search reach, not information, limits X-ray reflectivity fits of thin films: a count-scaled
   distance between wrong and true structures
2. When an X-ray reflectivity fit ends in a wrong structure: identifiability against optimiser
   reach, measured in counts
3. Local Fisher information does not describe global discrimination in X-ray reflectivity fits

**Central claim.** In the tested synthetic classes (one to three layers on silicon, Poisson counts at
10⁶ to 10⁸), a wrong fit endpoint is almost always a different basin of attraction that the data
would reject at the working count level by a wide margin, not a structure the curve cannot tell from
the truth. The wrong structures become statistically indistinguishable from the truth only at a
median of about 27 counts (R2a) and 44 counts (X2e), where the curves carry 10⁶ to 10⁸. Failures of
these fits are therefore failures of search reach. The exceptions are measured, not ignored: 6 of
488 distinct wrong basins need 10⁴ counts or more.

**Secondary claim.** The Fisher information at the true structure does not stand in for this global
picture: at the refined minima it overstates a wrong basin's deviance by a median factor of about
10² (R2a) to 10³ (X2e).

**Why these claims.** They are the ones backed by several pre-registered, checked findings that point
the same way and were each tried against a competing reading: F1 (identifiable within the fitting
bounds, wrong endpoints are optimiser failures), F5 and F6 (extra layers compensate for a wrong
basin; better starts did not remove it), F7 and F10 (the deviance of wrong endpoints and minima, and
that an endpoint's count level represents its basin). The other findings are weaker: F4, F8 and F9 are
draft, and F7's Fisher-ratio hypothesis is unresolved. A paper built only on "Fisher information is
local" would not be new (section 7). The reframed claim, identifiability against reach measured in
counts, is the one the records support.

**What would refute it.**

- A condition in the studied classes where a wrong basin has S₁\* above the working counts (the
  closest so far has S₁\* = 2.3 × 10⁵ against 10⁶, a deviance of 4.3 above the truth; section 5).
- Noisy or measured curves in which the wrong basins fall within the noise of the truth, or in which
  the basins merge. All results here use noise-free curves from the fitted model.
- A wider or physically motivated search box that moves the median S₁\* of the wrong basins by orders
  of magnitude. 651 of 656 R2a minima lie on a bound, so their S₁\* depends on where the box was drawn.
- A measured film for which a structure other than the true one fits within the noise.

## 1. Abstract (about 190 words)

Fits of X-ray reflectivity (XRR) curves often end in a wrong structure, and it is rarely clear
whether the curve lacks the information or the optimiser did not find it. We separate the two on
synthetic curves of one-to-three-layer films by computing, for each wrong fit endpoint, its Poisson
deviance from the true curve and the count level S₁ at which it would become indistinguishable from
it. For three single films the curve determines the film within the fitting bounds, and 43 of 48
wrong endpoints of a gradient fitter are optimisation failures (a split that rests on an amendment
made after the first runs). Refining 749 recorded endpoints of one family and 93 of another to their
local minima, an endpoint's S₁ represents its basin in 98.3 % and 95.7 % of cases. Wrong basins
need a median of about 27 and 44 counts to match the truth, against 10⁶ to 10⁸ counts in the
curves; 6 of 488 distinct basins need 10⁴ counts or more. The Fisher quadratic at the true
structure overstates the deviance of a wrong basin by a median factor of about 10² to 10³, so local
Fisher information cannot stand in for global discrimination. The results are synthetic, noise-free
in the fitted model and bounded by the search boxes; no measured film has been validated.

Only F1, F3, F5, F6, F7 (H-W2 part) and F10 enter the abstract. F4, F8 and F9 are draft and are not
used.

## 2. Outline

1. **Introduction and problem.** The XRR inverse problem: the detector records intensity only, so
   different profiles can give the same curve (Pershan 1994; Majkrzak and Berk 1995), and a fit that
   returns a wrong film leaves two readings: the optimiser stopped early, or the curve does not
   decide. The two need different remedies (a better search path against a posterior that keeps
   several solutions). Prior work on Fisher information in reflectometry is local and for neutrons
   (Durant 2021, 2022); on practical identifiability in X-rays it is MCMC-based (Gil and Windover
   2012). Learning-based XRR analysis exists (Greco 2019, 2022; Kim and Lee 2021; Starostin 2025).
   Contribution: a count-scaled global distance (S₁) between a wrong structure and the truth,
   measured on recorded fit endpoints, plus its comparison with local Fisher information.
2. **Forward model and conventions.** Section 3.
3. **Fitting protocol.** Section 4.
4. **Identifiability method.** Section 5: local Fisher and collinearity (F1), profile likelihood,
   the multi-start scan (F3), the deviance and S₁ of wrong endpoints (F7), the basin study (F10).
5. **Results.** Section 6, ordered F1, F3, F5 and F6, F7, F10, with the draft findings flagged.
6. **Limitations.** Section 9.
7. **Software availability.** Section 10.

## 3. Forward model and conventions

The model is the standard Parratt recursion for specular reflectivity from a stack of layers, with a
roughness factor at each interface, a Gaussian resolution convolution in q, a zero offset in 2θ and a
count model `s · (i0 · R_res(q_eff) + 10^bkg)` with Poisson noise
([Experiment 001 model section](../identifiability/README.md#model)). The R2a scan and the fit
policy runs also free a Cu Kα₂ ratio and a footprint parameter (the parameter lists in the basin
records); Experiment 001's local Fisher cases treat both as corrected. Layers are described by
thickness d, roughness σ and scattering length density ρ (10⁻⁶ Å⁻²). Where a truth stack has a top
oxide (10 Å), it sits above the film and the film's roughness is the buried oxide/film interface
(same README, "Where the truth curves come from"); one of the three single-film cases of F1 has none.

Which engine produced the results. Experiments 001 to 005 used this repository's own torch engine
(`physics.abeles_core` for the Jacobians in complex128, the float32 `refine.make_forward` for the
fits), not ReflectoSim. The engine was checked against an independent float64 numpy Parratt
recursion to a relative difference of 5 × 10⁻¹⁵ in R over 2θ 0.2 to 12°, with the engine's fixed
1 × 10⁻⁹ Å⁻² absorption in every medium (which the numpy code cannot represent) changing R by up to
1.2 × 10⁻³ near the film's critical edge; and against central differences of a numpy re-implementation
of the whole counts model to a relative column error of 6 × 10⁻⁸ in the Jacobian
([Experiment 001, Checks](../identifiability/README.md#checks)).

The physics now lives in ReflectoSim (version 0.1.0 on PyPI): a float64 numba Parratt kernel that
agrees with refnx to 1 × 10⁻¹⁰, a differentiable torch backend, and the resolution integral. This
repository's data generator takes its forward physics from ReflectoSim since the change recorded in
its history. The experiments reported here were **not** re-run on the ReflectoSim engine. That
engine agrees with the one used here only through the tests of each against independent references
(gap G4 in section 8).

Not in the model, by design: any model error between the data and the fit. Every result assumes that
the fitted model is the true one (the "inverse crime"); section 9.

## 4. Fitting protocol

The fitter is the repository's `refine` (gradient-based trust-region least squares over the
parameters above, with staged opening of parameters), with a log-type objective as returned by the
tool and a Poisson deviance used for the analysis. The scripted protocol A1 chooses the layer count,
opens parameters and stops by rule; C fits every layer count with every parameter free under the same
budget; B is a random floor ([Experiment 002 report](../fit_policy/results.md)). Starting values
come from the recorded multi-start runs; the number of starts, the bound boxes and the 200-evaluation
cap are those of the registrations, which are the authoritative description
([registration for 002](../fit_policy/DESIGN.md), [003](../fit_policy/X2D.md),
[004](../fit_policy/X2E.md)). ReflectoFit holds a related fitting engine for ordinary use
(least squares, differential evolution, MCMC, model selection). It was not used for these runs.

Facts needed for the sections below: R2a is the scan of 37 conditions (19 single-film, 18 bilayer)
over q range, film contrast and nuisance set; X2e is a grid of two- and three-layer truths at two
count levels. ΔD is the Poisson deviance of an endpoint's curve from the true curve (noise-free),
and S₁ = S / ΔD is the count level at R = 1 below which the endpoint is within ΔD = 1 of the truth
([registration](../fisher_global/PREREG.md)).

## 5. Identifiability method

1. **Local.** Fisher information of Poisson counts, `F = Jᵀ diag(counts) J` from the autograd
   Jacobian; the Cramér-Rao bound, correlations of F⁻¹, and a collinearity index γ = 1 / s_min
   (Brun et al. 2001) from the whitened, column-normalised Jacobian (F1, 243 cases).
2. **Profile likelihood.** The profile of d for three single films, with every other parameter
   re-fitted, as an upper bound on the profile likelihood along one tracked branch (F1).
3. **Multi-start scan.** The R2a scan: which conditions have a start that reaches the truth basin,
   and how many modes (F3).
4. **Deviance of wrong endpoints.** The recorded endpoints are scored against the truth with ΔD and
   S₁, the quadratic Fisher prediction at the truth is compared with them (ratio ρ), and the deviance
   is split by q region (F7: H-F, H-W2).
5. **Basin minima.** Each recorded R2a (749) and X2e (93) endpoint is refined to convergence from
   where it ended, with no new start; two minima are one basin if their cross-deviances are both ≤ 1
   (F10: H-BM1, Q-BM2, H-BM3).

All five are written down in registrations that precede the runs; deviations and amendments are
listed in each report.

## 6. Results

### 6.1 One film at a time (F1, checked)

With every nuisance known, d and ρ of a single film have Cramér-Rao bounds of 0.01 to 0.04 % and σ
0.3 to 1 %. Freeing the 2θ offset adds an offset-ρ valley (correlation 0.995 for the middle film).
Freeing a top oxide makes the *unbounded* local problem nearly degenerate (d against oxide thickness
and SLD, correlation −0.998, γ 60 to 1300), but inside the fitting bounds (oxide SLD 15 to 22,
Kα₂ ratio ≤ 0.6) the weak direction is cut and the three cases are identifiable, with Δχ² ≤ 1 widths
of ±0.04 % to ±1.4 %. Of 48 wrong endpoints of the gradient fitter, 43 are optimisation failures
(the fit stopped where a better fit exists) and 5 are objective or model errors. The split rests on
amendment 3, made after the first runs. Figures 1 and 2.

### 6.2 Scan of conditions (F3, checked)

No analysed condition has a multimodal label: 19 single-film and 5 reached bilayer conditions have
one mode each. 13 of 18 bilayer conditions are undecided because no start reached the truth's basin.
Nothing was trained. The result is "no-go for a generative posterior model for now", not a proof that
none is needed.

### 6.3 Extra layers and wrong basins (F5, F6, checked; F4, draft)

Against a full fit of every layer count with every parameter free, the scripted protocol A1 wins on
BIC but loses on held-out points, and beats it in 0 of 4 multilayer cells (F4, draft; Figure 3).
The registered reading that a wrong basin compensated by extra layers (H2) is supported, and H1, H3
and H4 are refuted (F5). A better start for new layers or for the whole stack beats the full fit in 0
of 4 cells; the basin reading D1 is supported (0 of 53 within margin), the sources B1 and B2 are
unresolved, B3 is refuted, and no cause is named (F6). The protocol change shows no attributable pass
(F5). This part is a negative result about fitting policy; it is the reason the wrong-basin question
is asked.

### 6.4 Distance of wrong endpoints (F7, checked; F8, draft)

Of 2191 wrong R2a endpoints, the quadratic Fisher prediction overstates the deviance by a median of
about 10² and by at least 10 for 98.1 %. Near the truth (177 endpoints) the ratio lies in the
registered band 0.5 to 2 for 0.831, below the 0.90 the rule needed, so H-F is **unresolved**: 17 %
of near endpoints lie on curved nuisance valleys where the quadratic overstates the deviance.
H-W2 is **supported**: at least 90 % of ΔD lies at q ≤ 0.235 Å⁻¹ for 0.996 of the test endpoints,
with the largest share in the first three blocks for 0.953. One truth-start local profile of a
bilayer (F8, draft) shows the first film's thickness as a ridge (2.24 % half-width) with the other
three parameters identifiable (0.12 to 0.27 %); it tests no hypothesis.

### 6.5 Basin minima (F10, checked)

After refining each endpoint to its minimum, H-BM1 is **supported** for R2a (98.3 % with κ ≤ 2, one
miss) and X2e (95.7 %, one miss). The 655 and 92 wrong minima fall into 399 and 89 distinct basins.
Their S₁\* has medians of 26.7 (R2a) and 44 (X2e). In both sources 6 basins need S₁\* ≥ 10⁴ counts:
five R2a basins that are one structure (the 10 Å oxide merged into the film, with the oxide thickness
and the Kα₂ ratio on their bounds) and one X2e basin that is a nearby minimum of the true structure.
H-BM3 (B-1's descriptions persist at the minima) persists for R2a and is unresolved for X2e
(statement (b) at 77.5 % against the 80 % needed). Figure 4 (TODO) shows the S₁\* distribution.

## 7. Prior work and novelty

See section 11 for the verified list. In short:

- Fisher information and Cramér-Rao bounds for reflectometry: neutrons, local (Durant 2021, 2022).
  Experiment 001's local result for X-rays restates known effects (a contamination layer dominating
  the thickness uncertainty, an angle offset biasing thickness and density) in Fisher terms
  (Gil and Windover 2012; Windover 2014). **Prior art.**
- That Fisher information is local and misleads globally is known outside reflectometry: profile
  likelihood against asymptotic intervals (Raue 2009), sloppy models and curved valleys
  (Gutenkunst 2007; Transtrum 2011). **Prior art.** It is not claimed as new here.
- Multimodal posteriors in XRR appear as case studies (Kim and Lee 2021; Starostin 2025). Amortised
  inference for XRR exists (Starostin 2025). Nothing found measures a count-scaled distance of wrong
  structures from the truth across a scan of conditions, or separates identifiability from search
  reach in XRR (not proof of absence; the search was not exhaustive).

The claim of this paper is therefore the measured S₁ picture and the two-sided failure of the
local bound, not the general statement about Fisher information.

## 8. Figures 1 to 5

Commit shas are the last commit that changed the file, in this repository.

| # | content | data | script | commit | reproduce | status |
|---|---|---|---|---|---|---|
| 1 | Local identifiability: Cramér-Rao bound against measured range, correlation matrix, collinearity index γ | [fim_results.csv](../identifiability/fim_results.csv), [fim_tables.md](../identifiability/fim_tables.md), image [fim.png](../identifiability/fim.png) | `experiments/001_identifiability/fim.py` | 21081f4 | `uv run python experiments/001_identifiability/fim.py` (about 10 s CPU) | exists, F1 checked |
| 2 | Three true films, one Poisson curve with refine's wrong endpoint, profile likelihood of d, endpoints of both fitters | [nonunique_mid_oxide.json](../identifiability/nonunique_mid_oxide.json), [nonunique_mid_no_oxide.json](../identifiability/nonunique_mid_no_oxide.json), [nonunique_t2_low.json](../identifiability/nonunique_t2_low.json), images [nonunique.png](../identifiability/nonunique.png), [nonunique.pdf](../identifiability/nonunique.pdf) | `experiments/001_identifiability/nonunique.py` | 3c28d2e | `uv run --group figures python experiments/001_identifiability/nonunique.py report` (redraws from the saved JSON) | exists, F1 checked |
| 3 | Scripted protocol against a full fit: cumulative BIC(A1) − BIC(C) for one, two and three layers at 10⁶ and 10⁸ counts | [verdict.json](../fit_policy/verdict.json), [runs/](../fit_policy/runs/), image [delta_bic.png](../fit_policy/delta_bic.png) | `experiments/002_fit_policy/report.py` | 692d442 | `uv run python experiments/002_fit_policy/report.py` | exists, F4 **draft** |
| 4 | Distribution of S₁\* of the distinct wrong basins, R2a and X2e, by film contrast, with the 10⁴ line and the working counts | [basin_full_summary.jsonl](../fisher_global/basin_full_summary.jsonl) (fields `S1_min`, `truth_basin`, `outcome`, `kappa`, `id`), [basin_verdict.json](../fisher_global/basin_verdict.json) | none yet | n/a | **TODO:** add a plot-only script `experiments/005_fisher_global/basin_fig.py` that reads the summary file and `basin_verdict.json` (no fit, no new run) and run `uv run --group figures python experiments/005_fisher_global/basin_fig.py`; the table `basin_full.md` already holds the numbers | missing; F10 checked |
| 5 | Deviance share per q region of wrong endpoints (H-W2) with the Fisher ratio ρ against ΔD | [b1.json](../fisher_global/b1.json), [b1_tables.md](../fisher_global/b1_tables.md), [b1_verdict.json](../fisher_global/b1_verdict.json) | `experiments/005_fisher_global/b1.py` (tables only) | eef19d4 | **TODO:** add a plot-only script reading `b1.json` (no run); the tables are produced by `uv run python experiments/005_fisher_global/b1.py report` | missing; F7 checked |

Cheap extras not counted above: the one-condition profile of a bilayer (F8 draft) and the
overdispersion of repeated public scans (F9 draft, `hcal2.json`) have no figure.

## 9. Limitations

Every limit FINDINGS states is kept here.

- **Synthetic only, except F9.** Results use synthetic curves; F9 uses repeated public scans. No
  measured film has been validated.
- **Noise-free parts.** F7 and F10 refine and score noise-free data generated from the fitted model;
  noisy and measured data are not tested for them (the noisy-data objective was checked only for X2e's
  own returned states). The inverse crime applies to every result: the fitted model is the true one.
- **Bounds.** Every identifiability label in F1 depends on the fitting bounds; the profile follows
  one branch and is an upper bound on the profile likelihood; the deciding amendment 3 was not
  pre-registered. 651 of 656 R2a minima lie on a bound, so their S₁\* is a statement about the box.
  F3 holds only for noise-free data at 10⁶ counts and a registered 2 % ridge line.
- **Local, not global.** F10 is local refinement from recorded endpoints inside bounded boxes: no
  global uniqueness, number of basins or multimodality is established; interior-only H-BM1 is not
  testable; basins no recorded endpoint reached are not counted. The 13 bilayer conditions where no
  start reached the truth are undecided, so the bilayer half of F3 is largely untested.
- **Capped fits.** Many R2a endpoints stopped at the 200-evaluation cap and are not basin minima
  (F10 examines minima); F3 has weak negatives where only 1 to 3 of 64 starts reached the truth
  basin.
- **Fisher only at the truth.** F7 evaluates it only at the truth, one count level, background and
  truth per condition; the H-F (i) band was chosen with X2e's near states in view. H-F is unresolved.
- **Fit policy.** F4 to F6 apply to the registered cells and rules, not to every possible learned
  policy. The registered X2d confirmation run was not run. The source of the wrong basin in F6
  remains unresolved; the aborted first X2e run is kept outside the tree. F4 has no independent
  verification of its numbers.
- **Draft findings.** F8 is one truth-start fit in one condition; profile points are local fits, so
  half-widths can only be too narrow; memory was not measured. F9 is one sample, one exposure and one
  fast-scan synchrotron instrument, not laboratory step scans; the noise source is not identified; it
  needed an amendment after the input pre-checks and has no verdict under the literal pre-check 3;
  its dispersion numbers have no independent verification.
- **Withdrawn.** The earlier MCMC widths were withdrawn (unconverged chains).
- **Engine.** The experiments ran on this repository's engine, not on ReflectoSim (section 3).

## 10. Software availability

- ReflectoSim 0.1.0 (forward physics, generation) and ReflectoFit 0.1.0 (the fitting engine, with
  ReflectoSim as a dependency): on PyPI as `reflectosim` and `reflectofit`; source at
  `NaNIsaac/ReflectoSim` and `NaNIsaac/ReflectoFit` on GitHub, MIT licence.
- This repository (`NaNIsaac-Research/ReflectoLearn`, public) holds the experiments 001 to 005, the
  registrations, the run records and the reports ([research index](../research/README.md)). Result
  files too large for the repository are in the release
  [`research-data-1`](https://github.com/NaNIsaac-Research/ReflectoLearn/releases/tag/research-data-1)
  with SHA-256 sums. Frozen tags (`frozen/exp001-r2a-829f330` through `frozen/exp005-basin-51b048a`)
  keep the commits the registrations cite.
- Reproduction commands for each report are in the research index. The full basin run took 50.9 CPU
  hours on four processes (12.9 hours of wall time).

## 11. Prior work (verified)

Each entry's DOI was looked up on 2026-11-06 in the Crossref record; title, first author, year and
venue match the entry. Authors beyond the first are not listed. This does not mean the paper's
content was read in full: the one-line "relevance" comes from the abstract or the literature table
[literature.md](../identifiability/literature.md) (written in Korean, same method, from 2026-10-22).

| first author, year | title | venue | DOI | relevance |
|---|---|---|---|---|
| Pershan, 1994 | X-ray or neutron reflectivity: Limitations in the determination of interfacial profiles | Phys. Rev. E 50, 2369 | 10.1103/PhysRevE.50.2369 | non-uniqueness from lost phase |
| Majkrzak, 1995 | Exact determination of the phase in neutron reflectometry | Phys. Rev. B 52, 10827 | 10.1103/PhysRevB.52.10827 | reference layers fix the phase |
| Majkrzak, 2003 | Phase-Sensitive Neutron Reflectometry | Langmuir 19, 7796 | 10.1021/la0341254 | same, review |
| Brun, 2001 | Practical identifiability analysis of large environmental simulation models | Water Resour. Res. 37, 1015 | 10.1029/2000WR900350 | collinearity index γ |
| Gutenkunst, 2007 | Universally Sloppy Parameter Sensitivities in Systems Biology Models | PLoS Comput. Biol. 3, e189 | 10.1371/journal.pcbi.0030189 | sloppy Fisher spectra |
| Raue, 2009 | Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood | Bioinformatics 25, 1923 | 10.1093/bioinformatics/btp358 | profile likelihood against asymptotic intervals |
| Transtrum, 2011 | Geometry of nonlinear least squares with applications to sloppy models and optimization | Phys. Rev. E 83, 036701 | 10.1103/PhysRevE.83.036701 | curved valleys, stalled least squares |
| Gil, 2012 | Limitations of x-ray reflectometry in the presence of surface contamination | J. Phys. D 45, 235301 | 10.1088/0022-3727/45/23/235301 | contamination layer dominates thickness uncertainty (MCMC) |
| Windover, 2014 | Determining sample alignment in x-ray reflectometry using thickness and density from GaAs/AlAs multilayer certified reference materials | Meas. Sci. Technol. 25, 105007 | 10.1088/0957-0233/25/10/105007 | angle offset biases thickness and density |
| Treece, 2019 | Optimization of reflectometry experiments using information theory | J. Appl. Cryst. 52, 47 | 10.1107/S1600576718017016 | information gain from MCMC (neutrons) |
| Durant, 2021 | Determining the maximum information gain and optimizing experimental design in neutron reflectometry using the Fisher information | J. Appl. Cryst. 54, 1100 | 10.1107/S160057672100563X | closest: Poisson Fisher information, neutrons |
| Durant, 2022 | Optimizing experimental design in neutron reflectometry | J. Appl. Cryst. 55, 769 | 10.1107/S1600576722003831 | minimum-eigenvalue design criterion |
| Nelson, 2019 | refnx: neutron and X-ray reflectometry analysis in Python | J. Appl. Cryst. 52, 193 | 10.1107/S1600576718017296 | reference engine; MCMC |
| Björck, 2007 | GenX: an extensible X-ray reflectivity refinement program utilizing differential evolution | J. Appl. Cryst. 40, 1174 | 10.1107/S0021889807045086 | global fitting program |
| Greco, 2019 | Fast fitting of reflectivity data of growing thin films using neural networks | J. Appl. Cryst. 52, 1342 | 10.1107/S1600576719013311 | neural-network XRR fitting |
| Mironov, 2021 | Towards automated analysis for neutron reflectivity | Mach. Learn.: Sci. Technol. 2, 035006 | 10.1088/2632-2153/abe7b5 | NN start values (neutrons) |
| Doucet, 2021 | Machine learning for neutron reflectometry data analysis of two-layer thin films | Mach. Learn.: Sci. Technol. 2, 035001 | 10.1088/2632-2153/abf257 | NN start values (neutrons) |
| Kim, 2021 | Probabilistic parameter estimation using a Gaussian mixture density network: application to X-ray reflectivity data curve fitting | J. Appl. Cryst. 54, 1572 | 10.1107/S1600576721009043 | multimodal XRR posteriors |
| Greco, 2022 | Neural network analysis of neutron and X-ray reflectivity data: automated analysis using mlreflect, experimental errors and feature engineering | J. Appl. Cryst. 55, 362 | 10.1107/S1600576722002230 | NN analysis, experimental errors |
| McCluskey, 2023 | Advice on describing Bayesian analysis of neutron and X-ray reflectometry | J. Appl. Cryst. 56, 12 | 10.1107/S1600576722011426 | reporting practice |
| Munteanu, 2024 | Neural network analysis of neutron and X-ray reflectivity data incorporating prior knowledge | J. Appl. Cryst. 57, 456 | 10.1107/S1600576724002115 | priors as NN input |
| Starostin, 2025 | Fast and reliable probabilistic reflectometry inversion with prior-amortized neural posterior estimation | Sci. Adv. 11, eadr9668 | 10.1126/sciadv.adr9668 | closest for the generative side |

**To check** (not verified in this draft; do not cite until verified):

- Savchenkov 2024, named in the repository's synthesis report as a multimodality case study:
  reference not looked up.
- Items of the literature table that carry only an arXiv number and were not re-checked here:
  Champneys 2025 (arXiv 2509.06924), Papamakarios and Murray 2016 (arXiv 1605.06376), Cranmer 2020,
  Chung 2023 (arXiv 2209.14687), Dax 2023 (arXiv 2305.17161), Lipman 2023 (arXiv 2210.02747).
- Rustamov 2025, a conference poster abstract on domain transfer for reflectometry: no DOI; cite only
  after the abstract is retrieved.
- Several 1990s papers in the literature table (Sivia and Webster 1998 and others): DOIs were
  verified by the earlier lane, but their content was summarised from the titles.
- The repository's refine protocol and the log-MSE objective have no external reference.

## 12. Claims table

| # | claim | status | evidence |
|---|---|---|---|
| C1 | With every nuisance known, d, σ and ρ of one film are determined to 0.01 to 1 % (Cramér-Rao, γ < 2) | checked | F1; [report](../identifiability/README.md) |
| C2 | A free top oxide makes the unbounded local problem nearly degenerate (γ 60 to 1300); inside refine's bounds it is identifiable (Δχ² ≤ 1 widths ±0.04 to ±1.4 %) | checked | F1; same report, R1c |
| C3 | 43 of 48 wrong refine endpoints of three films are optimisation failures, 5 objective or model errors | checked (split rests on unregistered amendment 3) | F1; [report](../identifiability/README.md) |
| C4 | No analysed condition has a multimodal label (19 single-film, 5 reached bilayer; 13 bilayer undecided) | checked | F3; [go/no-go](../identifiability/generative_go_nogo.md) |
| C5 | Several fitting changes are needed together to reach the maximum-likelihood fit; the opt-in preset (ABCF) missed confirmation (42 of 48, 43 needed) | checked | F2; [report](../identifiability/refine_causes.md) |
| C6 | Extra layers compensate for a wrong basin (H2 supported; H1, H3, H4 refuted); no attributable pass of the protocol change | supported (checked) | F5; [report](../fit_policy/x2d_results.md) |
| C7 | Better start values did not remove the extra layers; basin reading D1 supported, cause unresolved | checked | F6; [report](../fit_policy/x2e_results.md) |
| C8 | The scripted protocol does not beat a full fit in the multilayer cells (0 of 4) | draft | F4; [report](../fit_policy/results.md) |
| C9 | Wrong endpoints' deviance concentrates at low q (H-W2) | supported (checked) | F7; [report](../fisher_global/reports/RESULTS-B1.md) |
| C10 | The Fisher quadratic matches the deviance near the truth and overstates it at wrong endpoints (H-F) | hypothesis (unresolved: 0.831, 0.90 needed) | F7; same report |
| C11 | An endpoint's S₁ represents its basin minimum (H-BM1: R2a 98.3 %, X2e 95.7 %) | supported (checked) | F10; [report](../fisher_global/reports/RESULTS-BASIN.md) |
| C12 | Wrong basins need a median of about 27 (R2a) and 44 (X2e) counts to match the truth; 6 of 488 need at least 10⁴ | checked (box-limited) | F10; same report, Q-BM2 and section 3 |
| C13 | At the minima the Fisher quadratic overstates a wrong basin's deviance by a median of 10² (R2a) to 10³ (X2e) | checked (H-BM3 persists for R2a, unresolved for X2e) | F10; same report, section 5 and 7 |
| C14 | One truth-start bilayer fit: d₁ is a ridge (2.24 %), ρ₁, d₂, ρ₂ identifiable (0.12 to 0.27 %) | draft | F8; [report](../fisher_global/reports/RESULTS-B2.md) |
| C15 | Counts of 100 repeated public scans are overdispersed (φ median 1.34) | draft (supported under amendment 1; no verdict under literal pre-check 3) | F9; [report](../fisher_global/reports/RESULTS-HCAL2.md) |
| C16 | Central claim: wrong fit endpoints in the tested classes are search failures, not missing information | checked in the synthetic noise-free scope; **hypothesis** for noisy or measured data | C2, C3, C6, C7, C9, C11, C12 |
| C17 | Fisher information is local and misleads globally | prior art | Raue 2009; Gutenkunst 2007; Transtrum 2011 |
| C18 | Poisson Fisher information for reflectometry | prior art (neutrons) | Durant 2021, 2022 |
| C19 | The XRR contamination-layer and angle-offset effects restated in Fisher terms | prior art | Gil 2012; Windover 2014 |

## 13. Gap list: what would make it submittable

The smallest set, none run here. CPU costs are scaled from the recorded full basin run: 935
refinements took 50.9 CPU hours (mean 241 s per R2a refinement, median 26.5 s) on four processes;
B-1 took 31 s; one truth-start profile took 451 s. Estimates are inference.

| id | experiment | why | cost |
|---|---|---|---|
| G1 | One measured thin-film XRR set (public, or a lab sample with known preparation) refitted from many starts, then the S₁\* of its distinct endpoints computed with the same code | shows the wrong-basin picture outside simulation; without it the paper is a simulation study | data search and registration, owner time about 2 to 5 days; CPU about 1 to 4 hours for a few hundred starts of a single film (about 25 s each) [estimate] |
| G2 | Noisy curves: refit Poisson draws (and draws with the measured overdispersion φ ≈ 1.3) of the existing conditions and recompute S₁ and basins for a subset of about 100 candidates | tests whether the picture survives noise and model mismatch; all current results are noise-free | about 7 CPU hours at the mean refinement time for 100 candidates, one HEAVY slot [estimate] |
| G3 | Repeat the basin refinement for the five R2a box-edge basins and their neighbours (about 100 to 250 candidates) in a wider, physically motivated box (the oxide thickness and Kα₂ limits) | 651 of 656 minima rest on bounds, so S₁\* is a box property | about 7 to 17 CPU hours [estimate] |
| G4 | Evaluate the recorded endpoints and truths of R2a and X2e with the ReflectoSim forward and compare the deviances with the stored ones (forward only, no fit) | experiments ran on this repository's engine, not the published one | about 1 minute CPU (B-1 took 31 s) [estimate] |
| G5 | Replace Figures 4 and 5 TODO with plot-only scripts | needed for the figures | under 1 hour, no run |

Not in the smallest set but named by the records: the 13 unreached bilayer conditions (R2h, not
registered), a repeat on laboratory step scans for F9, and a registered confirmation run for X2d.

## 14. Venue candidates (owner decides)

1. **Journal of Applied Crystallography.** The Fisher-information, Bayesian and neural XRR work in
   section 11 (Durant, McCluskey, Kim, Greco, Munteanu) appeared there; it is where the readers of
   the closest prior work are. Fits a methods paper with a software availability section.
2. **Machine Learning: Science and Technology.** Published two of the neural reflectometry papers
   above (Mironov, Doucet); a fit if the framing leans toward when a learned or generative model is
   needed (the no-go result and the S₁ picture).
3. **Journal of Synchrotron Radiation.** Scope for a methods paper on reflectometry analysis not
   verified here; considered only if G1 uses synchrotron data and F9 (counts overdispersed in
   repeated public scans) is promoted from draft.
