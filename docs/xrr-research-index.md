# XRR research: master index

One place to follow the X-ray reflectivity (XRR) work across repositories. It records what each branch asked, how it worked, what was measured and what is still open. It does not merge code or rank the branches: they tried different models and fitting methods, a later repository is not assumed to be better, and which fitting strategy works best has not been tested.

Written 2026-10-23 from the code, git history, committed notebooks and run outputs (lineage audit, NaNIsaac BACKLOG A1). Labels used below:

- **recorded**: a number that exists in a file or log, with its source.
- **cross-checked**: a result another agent lane reproduced independently (BACKLOG rule 22). This is review between AI lanes, not peer review.
- **unverified**: stated or planned, with no recorded result. Unverified is not failed.

## The question

From one specular XRR curve, recover the layer structure of a thin film (thickness d, roughness σ, scattering length density SLD; for some branches also the native oxide, the substrate and instrument parameters), quickly and reliably enough to use at the instrument. Every branch splits the problem into an initial guess (a neural network trained on simulated curves) and a refinement against the measured curve through a forward model; they differ in the forward model, what the network predicts, how it is trained and how the refinement searches.

Underneath sits a question none of the branches answered until Experiment 001: when a fit is wrong, is it the optimizer, or does the curve not contain the information (non-uniqueness, parameter correlation, nuisance trade-offs)?

## Repositories

| Repository | Visibility | Active (research commits) | Role |
|---|---|---|---|
| [ReflectoLearn `old` branch](https://github.com/NaNIsaac/ReflectoLearn/tree/old) | public | 2025-04-23 to 2025-10-02 | first exploration, started as `SJB7777/ReflectoLearn`: refnx simulation, CNN/MLP regressors, FFT thickness estimate, layer-count classifier. Unrelated history to `main` (no common ancestor) |
| [ReflectoExp](https://github.com/NaNIsaac/ReflectoExp) | public | 2025-11-06 to 2025-12-30, again 2026-08-13 to 08-19 | experiment series exp01-exp07 (began life as `SJB7777/Reflecto`, package renamed `reflecto_exp` on 2025-12-21) |
| [Reflecto](https://github.com/NaNIsaac/Reflecto) | public | 2025-12-19 to 2025-12-24 (branch `refactor` to 12-30) | packaged single-film analysis: exp07 CNN guess, GenX differential-evolution refinement, electron-density output |
| [XRefine](https://github.com/NaNIsaac/XRefine) | private | 2025-12-30 to 2026-01-23 | physics-informed network: differentiable Abeles inside the model and the loss, multilayer with complex SLD, instrument parameters predicted, staged DE / hybrid refinement |
| ReflectoLearn `main` | public | 2026-06-01 onward | library: exp07-style CNN, XRefine's Abeles engine, gradient refinement, MCMC, GenX and stitching ports; research branch `research/identifiability` (Experiment 001) |

Related, not research branches:

- [xrr_combine](https://github.com/NaNIsaac/xrr_combine) (private, 2025-03): standalone beamline CLI that stitches attenuated scan ranges. ReflectoLearn `stitch.py` is a NumPy port of it (c96bf50).
- [xray_ooguy](https://github.com/NaNIsaac/xray_ooguy) (public): offline beamline calculators; its DATA view has its own XRR stitching (median ratio over an overlap of at least two points, first segment as reference), a different algorithm from xrr_combine's (one shared boundary point, last scan as reference). Scans that share only one point, like xrr_combine's example data, are not stitched there; it shows a gap warning.
- [ReflexAuto](https://github.com/NaNIsaac/ReflexAuto) (private): a Windows XRR fitting product (Parratt, automatic model selection, superlattice detection, MCMC; no neural network mentioned) sold under license; the business has ended. Its engine source was not found in any repository. History: ReflexAuto `docs/history.md`.
- DiffractoLearn (private): named and described as XRR machine learning, but its content is a one-line `main.py` and a notebook reading a powder-XRD database; no XRR code.

## Timeline of method changes

| Date | Where | Commit | Change |
|---|---|---|---|
| 2025-03-18 | xrr_combine | 2618901 | scan stitching with attenuator rescaling |
| 2025-04-23 to 10-02 | ReflectoLearn `old` | fb2e434 … b481550 | refnx simulation, CNN/MLP, FFT thickness, n-layer classifier |
| 2025-11-06 | ReflectoExp | 025da09 | init |
| 2025-11-12 | ReflectoExp | 13168bb | exp03: CNN blended with an FFT thickness estimate |
| 2025-11-14 | ReflectoExp | d8bce43 | exp02: single layer on refnx simulations |
| 2025-11-16 | ReflectoExp | 688f578, 78c087b | GenX simulator, exp04 |
| 2025-11-19 | ReflectoExp | 7a83647 | first GenX refinement from the network guess |
| 2025-11-20 | ReflectoExp | 1aa514c | exp05: valid-q mask channel, instrument augmentation |
| 2025-11-23 | ReflectoExp | 17f8530 | exp06: film + SiO2, 6 outputs |
| 2025-11-24 | ReflectoExp | 2fb9de4 | Optuna tuning |
| 2025-12-19 | Reflecto | 60f2a3b | package with exp05 weights |
| 2025-12-21 | ReflectoExp | aed114e, ad85577 | exp07: Fourier features of q, 2000-point grid, GenX engine |
| 2025-12-22/23 | Reflecto | e4230c4, 32e17a6 | exp07 weights (config `cubic`, ReflectoExp b93c91f) |
| 2025-12-30 | ReflectoExp, XRefine | abd6660, 9e3834e | torch Abeles engine; XRefine starts from it the same day |
| 2025-12-31 to 2026-01-23 | XRefine | 314ea17 … a31d647 | PINN-style model, Abeles rewrite, footprint, HPO, ViT/transformer, imaginary SLD, enhanced variant, `pipeline_next` |
| 2026-06-01 | ReflectoLearn | a1a5f7f, f35ab56, e3b26d6 | new root: exp07 model and training, XRefine Abeles, refinement |
| 2026-08-13 to 08-19 | ReflectoExp | 1c2ae53 … e49a08c | exp07 ablation on two evaluation domains |
| 2026-10-16 | ReflectoLearn | 55ddcc8, c96bf50, 6e160db | domain randomization (offset, oxide, Kα2); stitching and GenX ports |
| 2026-10-18 | XRefine | c1bc382 | one forward model for training data, reconstruction and refinement |
| 2026-10-19 | ReflectoLearn | c66e7bd | MCMC, FFT alias multistart, benchmark |
| 2026-10-22/23 | ReflectoLearn `research/identifiability` | 55e96f8 … | Experiment 001 (PR #24) |
| 2026-10-23 | ReflectoLearn | #31 | Poisson objective preset (opt-in) |

The "supersedes" / "superseded by ReflectoLearn" sentences in the READMEs were all added on 2026-10-16 in one documentation pass (ReflectoLearn 96b29cf, Reflecto ee408d4, XRefine 748de20, xrr_combine 5c3a584, ReflectoExp d964243). They describe where code was copied, not a comparison of results; they have since been rewritten to describe the repositories as competing implementations with different approaches.

## Methods by branch

| | ReflectoExp exp02-exp07 | Reflecto | XRefine | ReflectoLearn `main` |
|---|---|---|---|---|
| Forward model for training data | refnx (exp01-03), GenX (exp04-07) | (uses exp07 weights) | own torch Abeles + footprint, resolution, offset (`instrument_reflectivity`, since c1bc382; before it, bare Abeles curves) | own torch Abeles (`physics/abeles.py`, from XRefine), Névot-Croce, complex SLD |
| Structure | film (+ SiO2 from exp06; SiO2 randomized in exp07) on Si; SiO2 under the film in the training simulator, on top of the film in the GenX fitters (see below) | film + SiO2 on Si; the GenX fitter puts SiO2 on top of the film, the bundled exp07 weights were trained with it under the film | substrate + top layer + target film, complex SLD | film + optional nuisance oxide on top, substrate fixed range |
| Network output | exp06: 6 (film + SiO2); exp07: 3 (d, σ, SLD) | 3 | layer parameters incl. imaginary SLD + 6 instrument parameters | 3 (d, σ, SLD), normalized |
| Architecture | 1D CNN; exp05 mask channel; exp07 Fourier features of q | exp07 CNN | `fourier_conv` (Fourier features of normalized intensity), ViT, Transformer-VAE; enhanced variant | exp07-style CNN, optional Fourier features of q |
| Training loss | MSE on normalized parameters | | curve reconstruction (linear, gradient, log) + supervised parameter loss decaying 10000 → 100 + σ < d/2 penalty | MSE on normalized parameters |
| Augmentation / randomization | exp07: footprint, resolution, I0, background, Poisson, q shift | | resampling, scale and background, Poisson, low/high-q masks | footprint, smearing, I0, background, Poisson, end masks; offset, oxide, Kα2 (55ddcc8) |
| Refinement | GenX DE, two stages (exp04-07) | GenX DE: oxide + I0, then all (logR2) | `pipeline`: I0 pre-fit (Adam) then DE stages with a thickness grid search; `pipeline_next`: DE early, Adam/L-BFGS late | three Adam stages (linear, correlation, log); optional FFT alias multistart; optional Poisson preset; emcee MCMC; GenX fit port |

Known model inconsistencies found in this audit (issues open, nothing changed):
- Oxide position (ReflectoExp#9, Reflecto#3, ReflectoLearn#32; a decision for Isaac). GenX stacks layers from the substrate up, so `Stack(Layers=[A, B])` is Si / A / B / ambient (genx 3.8.0 `models/lib/refl_base.py`, `resolveLayerParameters`).
  - Oxide on top of the film:
    - ReflectoLearn training generator, `data/generator.py:77-80`: media `[ambient, oxide(top), film, substrate]`;
    - ReflectoLearn refine, `refine/pipeline.py:23`;
    - ReflectoLearn `genx_fit.py:59-60`, `Stack(Layers=[Film, Oxide])`;
    - the GenX fitters of ReflectoExp exp05–exp07 and of Reflecto.
  - Oxide under the film:
    - ReflectoExp training simulator, `simulate/simul_genx.py:73-79`: `layers = [sio2, film...]`; its docstring states the opposite;
    - ReflectoLearn `benchmark.make_synthetic`, `benchmark.py:102-107`: layers `[ambient, film, SIO2, SI]`.

    So inside ReflectoLearn the synthetic benchmark's truth and the generator and refine that fit it differ. Reflecto ships exp07 weights trained on data with the oxide under the film, together with a GenX fitter that puts it on top.
  - The two orders are different structures, not a small nuisance: for a 240 Å / σ 6 Å / SLD 45 film with 10 Å SiO₂, R(q) differs by a median of 60 % over 2θ 0.2–6° (physics review on #32).
- The background parameter of the Reflecto and exp07 GenX fit is not applied (Reflecto#2, ReflectoExp#9).
- Si/SiO2 GenX constants differ in sign of contrast from ReflectoLearn's (ReflectoLearn#8).

## Code relations

Four kinds of relation are kept apart: functional overlap (similar job), code reuse (one implementation used in two places), code transfer (an implementation copied or moved), project succession (purpose and direction carried on). Only the first three are established anywhere below.

| From | To | Kind | Evidence |
|---|---|---|---|
| ReflectoExp exp05 | Reflecto (first weights) | transfer | Reflecto 60f2a3b config = exp05 config |
| ReflectoExp exp07 | Reflecto `engines/model.py`, weights, GenX engine | transfer, developed side by side | model identical to exp07 `xrr_model.py` at b93c91f; GenX engine 6 lines apart (2025-12-24) |
| ReflectoExp `torch_engine` | XRefine `simulator/` → `core/` | transfer | same day, files differ by at most 6 lines; `torch_engine` is used by no ReflectoExp run |
| ReflectoExp exp07 | XRefine `fourier_conv` | transfer then divergence | docstring "Ported XRRPhysicsModel Encoder" |
| XRefine | ReflectoLearn `physics/abeles.py`, `physics/utils.py`, `refine/losses.py` | transfer | abeles: epsilon guards removed, otherwise identical; utils identical; losses ported (module docstring) |
| XRefine | ReflectoLearn `refine/pipeline.py` | idea reused, reimplemented | staged refinement with a trusted network window; XRefine used DE, ReflectoLearn uses Adam |
| ReflectoExp exp07 | ReflectoLearn `model.py`, `train.py` | transfer | 8 differing lines of 81 in the model |
| Reflecto | ReflectoLearn `genx_fit.py` | transfer (stated in 6e160db) | same bounds and two-stage DE; ReflectoLearn also applies the background |
| xrr_combine | ReflectoLearn `stitch.py` | algorithm port, test data copied | stated in c96bf50 |

Not carried into ReflectoLearn: XRefine's reconstruction-loss training, ViT/VAE models, multilayer and imaginary-SLD outputs, predicted instrument parameters, DE and grid refinement, `pipeline_next`; Reflecto's electron-density and density-ratio output and its split and multilayer fitting work (`fitting_split.py`, `script_builder_multi.py`, only on the `local-snapshot-20261022` branch).

## Experiments and recorded results

### ReflectoExp

| Exp | Question | Recorded | Status |
|---|---|---|---|
| exp01 | multilayer parameters as quantized classes | none | does not run now (imports removed modules) |
| exp02 | single layer, refnx | none; data and checkpoint lost (d746bc9) | |
| exp03 | CNN blended with FFT thickness | FFT thickness MAE 38.80 Å on 2-layer simulations (`runs/exp03_physics_fused/test/test.ipynb` output) | training script imports an undefined loss |
| exp04 | single layer, GenX | notebook outputs: simulated case true (66.63, 1.82, 1.375) vs network (66.48, 1.70, 1.634) (`notebooks/ml_fitting.ipynb`); a measured curve fitted in `ml_fitting_real.ipynb` | single cases |
| exp05 | mask channel, instrument augmentation, Optuna | measured curves in `notebooks/inference_test.ipynb`, `ml_fitting_real.ipynb`: several fitted parameters at their bounds | no reference values; unverified |
| exp06 | film + SiO2 (6 outputs) | measured `#1.dat` in `notebooks/inference_test.ipynb`: several parameters at bounds | unverified |
| exp07 | Fourier features of q, 2000 points, strong augmentation | ablation, below | |

exp07 ablation (recorded, `$DATA/XRR_AI/exp07/1230/ablation/ablation_summary_{clean,augmented}.csv` and `metrics/METRICS.md`; about 937,500 optimizer steps per variant, 5000 test curves, one seed). Thickness MAE in Å, clean / augmented test domain, and R² on the augmented domain:

| Variant | Clean | Augmented | R² (aug.) |
|---|---|---|---|
| A_full (exp07) | 4.75 | 38.42 | 0.940 |
| B_no_fourier | 4.95 | 38.71 | 0.940 |
| C_no_augment | 3.25 | 277.7 | -0.47 |
| D_lowres_200 | 9.92 | 53.64 | 0.911 |
| E_exp02_like | 6.85 | 396.9 | -3.43 |

A_full on the augmented domain: SLD MAE 0.785 ×10⁻⁶ Å⁻², roughness MAE 0.367 Å. E_exp02_like is a model with exp02's settings trained on the exp07 data, not exp02 itself. The A-B difference (Fourier features) is within what one seed can resolve. An earlier write-up of this comparison, `docs/exp02_vs_exp07_report.md` (ReflectoExp ec2f34a, 2026-08-14), was removed from the repository on 2026-10-15 (01ae27e) because it referred to scripts that no longer exist; it is still in git history. Its numbers (47.8 vs 279.7 Å) come from a smaller training budget and differ from the table above. All exp07 numbers are on simulated curves; none on measured data.

### Reflecto

- Bundled checkpoint history: `best.pt` at epoch 89, validation loss 5.7×10⁻⁴ (normalized MSE).
- Measured In₂O₃ 30 nm films (`test_data/In2O3_Ar_30nm/`) with hand-made GenX fits (`.hgx`, electron-density spreadsheets). No network + GenX result is recorded.

### XRefine

- No training or evaluation metrics were ever committed; the configured output directory does not exist on disk.
- HPO (Optuna databases removed in 8196187, readable from git): study `xrefine_hpo_v1`, 5 complete trials of 52, best objective 1616.07. The objective is the weighted training loss after 15 epochs, not a physical error.
- 28a6f28 (2026-10-19) records one observation: 320 Å In₂O₃ films came out at 1250-1500 Å (fringe alias in the correlation stage) before the thickness trust window was added. The checkpoint used is not recorded.
- No comparison of baseline vs enhanced model, or `pipeline` vs `pipeline_next`.

### ReflectoLearn `main`

- No trained-model metrics in the repository; `tests/test_benchmark.py` checks the benchmark plumbing with a stub network.
- A 40-epoch demonstration network (recorded, single author, explicitly not validated): before refinement, synthetic thickness median error 7.8 %, 5 of 12 cases above 10 % (`$DEV/NaNIsaac/_artifacts/reflectolearn-demo-run-20261022/benchmark/benchmark.json`).
- Measured In₂O₃ benchmark curves with GenX reference values were added in c66e7bd and removed in 242d7ec (#5); no measured example is bundled now (issue #2).

### ReflectoLearn Experiment 001: identifiability (`research/identifiability`, PR #24, draft)

Documents: [README](https://github.com/NaNIsaac/ReflectoLearn/blob/research/identifiability/docs/identifiability/README.md), [literature](https://github.com/NaNIsaac/ReflectoLearn/blob/research/identifiability/docs/identifiability/literature.md), [refine_causes](https://github.com/NaNIsaac/ReflectoLearn/blob/research/identifiability/docs/identifiability/refine_causes.md), [mcmc_width](https://github.com/NaNIsaac/ReflectoLearn/blob/research/identifiability/docs/identifiability/mcmc_width.md), [generative_go_nogo](https://github.com/NaNIsaac/ReflectoLearn/blob/research/identifiability/docs/identifiability/generative_go_nogo.md); scripts in `experiments/001_identifiability/`.

| Result | Status |
|---|---|
| R1b Fisher information / Cramér-Rao: with all nuisances known, d and SLD to 0.01-0.04 %, σ to 0.3-1 %; freeing the 2θ offset correlates SLD and offset (0.995); freeing the oxide correlates d and oxide SLD (-0.998) | cross-checked (FIM reproduced by the physics lane); the oxide conclusion was later called overstated because of the bounds (R1c) |
| R1c: within refine's bounds all three cases are identifiable (Δχ² ≤ 1 widths ±0.04-1.4 %; the lower end, mid_no_oxide, is an extrapolation below the 0.25 % profile grid with the Kα2 ratio hitting its bounds in the first step, so the true width lies between about 0.03 % and the unbounded CRLB of 0.28 %); 48 of 48 refine end points stopped away from the optimum: 43 are optimizer failures (worse than the ML fit on refine's own objective), 5 are preferred over the ML structure by refine's objective (an objective or model error, which a better optimizer alone does not fix) | cross-checked with objections (profile is an upper bound of the profile likelihood; the deciding amendment was not pre-registered) |
| R1c MCMC widths | withdrawn (chains not converged) |
| R2b causes of the refine failure: no single factor suffices; budget, removing stage 2, Poisson objective and step decay together reach 46 of 48 | in-sample, single author plus one new-seed point |
| R2f out-of-sample confirmation of the Poisson preset: 42 of 48 against the pre-registered 43 | not confirmed |
| R2c MCMC width: likelihood-limited for one case, undecided for two | single author |
| R2a conditions under which the curve is multimodal: 19 single-film conditions fitted, no verdict yet | interim, no conclusion |
| Generative inverse model (diffusion, flow) | no-go for now; revisited after R2a |

## Open research problems

Each item says where it is stated. None has a recorded answer.

1. **Sim-to-real.** Does domain randomization (ReflectoLearn) or strong augmentation (exp07) transfer to measured curves? exp07's clean-to-augmented gap (4.75 → 38.4 Å) is on simulated data only; there is no measured benchmark with reference values in any repository (ReflectoLearn issue #2, README).
2. **Which refinement works.** GenX DE (Reflecto, ReflectoExp), DE + grid (XRefine `pipeline`), DE + gradient hybrid (XRefine `pipeline_next`), staged Adam (ReflectoLearn), Poisson-objective preset (#31): never compared on the same cases. Experiment 001 shows the default ReflectoLearn refinement stops short of the optimum (#25) and the Poisson preset is not confirmed out of sample.
3. **Physics-informed vs supervised training.** XRefine's reconstruction loss against ReflectoLearn's and exp07's parameter MSE: not compared; no XRefine metrics exist, and XRefine was never retrained after its forward model was unified (c1bc382).
4. **Model structure.** Oxide above or below the film (ReflectoExp#9, Reflecto#3, ReflectoLearn#32). It is mismatched within ReflectoLearn as well: `benchmark.make_synthetic` buries the oxide, while the generator and refine put it on top. The R(q) difference is not small (median 60 % for one film; see "Known model inconsistencies"); Si/SiO2 constants (ReflectoLearn#8); model-misspecification bias `F⁻¹JᵀWδ` (Experiment 001 README, not run).
5. **Identifiability beyond one film.** Multimodality map (R2a), bilayers (not run), the t2_low Kα2 basin (undecided), offset-SLD valley (#14; fixes #20, #21 opt-in pending a decision).
6. **Uncertainty.** MCMC convergence and calibration (#29); FIM/CRB vs posterior widths, started in R1b/R2c.
7. **Information in the curve.** How much information the log-MSE objective discards against a Poisson likelihood (generative_go_nogo, candidate question 2).
8. **Small open items.** Fourier features' contribution (exp07 A vs B, one seed); cubic vs linear interpolation between exp07 training and Reflecto inference; XRefine's 2θ-offset term `q + 0.01·tth_offset` (smaller than the physical dq/d2θ at low angle by roughly 7, not checked); which demonstration or trained checkpoints exist beyond the 40-epoch demo.

### Links to follow-up methods

Where each method would attach to existing work; the literature table with references is `docs/identifiability/literature.md` on the research branch.

| Method | Existing hook | State |
|---|---|---|
| Fisher information, Cramér-Rao | Experiment 001 R1b (autograd Jacobian of the Abeles engine, Poisson weights) | done for three cases |
| Profile likelihood, identifiability | R1c profiles | done for three cases |
| Bayesian inverse problem, MCMC | `refine/mcmc.py` (emcee), R2c | convergence open (#29) |
| Simulation-based inference (NPE) | on-the-fly simulator and torch engine in ReflectoLearn; closest prior work PANPE (Starostin et al. 2025) | not started |
| Diffusion posterior sampling | differentiable Abeles as forward operator | no-go for now (generative_go_nogo) |
| Flow matching, optimal transport | same; FMPE (Dax et al. 2023) is the direct method | no-go for now |
| Multiple solutions, uncertainty | R2a multimodality map; FFT alias multistart (c66e7bd) | interim |

## Where the artifacts are

| What | Where |
|---|---|
| exp07 ablation (checkpoints per variant, summaries, metrics, figures) | `$DATA/XRR_AI/exp07/1230/ablation/` |
| exp07 run (`config.json`, `stats.pt`, `dataset.h5`, logs) | `$DATA/XRR_AI/exp07/1230/` |
| Measured In₂O₃ 30 nm data with GenX fits | `$DATA/XRR_AI/In2O3_Ar_30nm/`; Reflecto `test_data/In2O3_Ar_30nm/` |
| Reflecto bundled weights | Reflecto `src/reflecto/assets/` (older weights at 60f2a3b, e4230c4; others on branch `refactor`) |
| ReflectoExp notebooks with outputs | `runs/exp03_physics_fused/test/`, `runs/exp04_one_genx/notebooks/`, `runs/exp05_1layer_mask/notebooks/`, `runs/exp06_1lay_6out/notebooks/` |
| ReflectoLearn `old` notebooks | branch `old`, `notebooks/` |
| ReflectoLearn demo network | `$DEV/NaNIsaac/_artifacts/reflectolearn-demo-run-20261022/` |
| Experiment 001 tables, figures, fits | branch `research/identifiability`, `docs/identifiability/` |
| XRefine HPO databases | XRefine history before 8196187 |

Paths under `$DATA` and `$DEV` are on Isaac's machines, not in any repository.
