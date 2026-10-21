"""Inference on real measured XRR curves plus physics refinement.

Training uses synthetic data only, but inference targets real measured files.
The model only works correctly if the measured input goes through exactly the
same preprocessing as the synthetic training input:

  measured I(2theta) -> convert to q -> interpolate onto fixed q_grid
            -> per-curve max normalize -> log10 -> clamp(<=0)
            -> unmeasured region: mask=0 + log-floor

There are no ground-truth parameters, so evaluation centers on the refinement
FOM (LogMSE): how well the measured curve is reproduced when the AI prediction
is used as the initial guess for physics fitting. The substrate is unknown for
real measurements, so it is fixed to the midpoint of the config.substrate range.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from reflectolearn.config import Config
from reflectolearn.data import XRRAugmentor, param_stats
from reflectolearn.data.generator import LOG_FLOOR
from reflectolearn.model import XRRPhysicsModel
from reflectolearn.physics import tth2q
from reflectolearn.refine import RefineConfig, RefineStage, refine, refine_multistart

PARAM_NAMES = ("thickness", "roughness", "sld")
UNITS = {"thickness": "Å", "roughness": "Å", "sld": "×1e-6 Å⁻²"}

# Toggle the low-q linear inset in the overlay plot (temporarily off).
SHOW_LOWQ_INSET = False

# Fixed resolution sigma [Å⁻¹] for the blue "AI predict" curve. If dq were fit by
# refine while the film is frozen, it would non-physically absorb film mismatch, so
# instead of fitting it we smear the fringes with a sensible default. This avoids
# overly sharp minima. Adjust if needed.
AI_RES_SIGMA = 0.002


def _ai_align_config() -> RefineConfig:
    """Light refine config for the blue 'AI predict' curve.

    The film parameters (thickness/roughness/sld) are frozen; only intensity (i0),
    background (bkg), and low-angle footprint are matched to the measurement. Since
    no film keys are in `optimize`, refine leaves the film untouched and fits only
    i0/bkg/footprint, so info['r_fit'] is the 'AI structure + intensity/low-angle
    alignment' curve. Resolution (dq) is not fit here because fitting it with the
    film frozen would absorb film error and distort the curve; instead the curve is
    smeared separately with a fixed sigma (AI_RES_SIGMA).
    """
    return RefineConfig(
        fit_i0=True, fit_bkg=True, fit_footprint=True,
        fit_tth_offset=False, fit_dq=False, fit_kalpha2=False, fit_oxide=False,
        nuisance_reg=0.0,
        stages=[RefineStage("ai.align", optimize=("i0", "bkg", "footprint"),
                            loss="log_mse", steps=200, lr=0.03)],
    )


def _build_model(cfg: Config) -> XRRPhysicsModel:
    """Same as cli._build_model (duplicated to avoid a circular import)."""
    m = cfg.model
    return XRRPhysicsModel(
        q_len=cfg.simulation.q_points,
        use_fourier=m.use_fourier, fourier_scale=m.fourier_scale,
        fourier_features=m.fourier_features,
        n_channels=m.n_channels, depth=m.depth,
        mlp_hidden=m.mlp_hidden, dropout=m.dropout,
    )


def load_measured_curve(
    path: str | Path, *, x_axis: str, wavelength: float
) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column text file (2theta or q, intensity) into (q[Å⁻¹], intensity).

    Comment/header lines (non-numeric) are skipped automatically. Both whitespace-
    and comma-separated formats are supported.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line[0] in "#!;%/":
            continue
        parts = line.replace(",", " ").split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue  # header / noise line
    if not rows:
        raise ValueError(f"no parseable 2-column numeric data found: {path}")

    arr = np.asarray(rows, dtype=np.float64)
    x, intensity = arr[:, 0], arr[:, 1]
    q = tth2q(x, wavelength) if x_axis == "tth" else x
    order = np.argsort(q)                          # ensure ascending q
    return q[order], intensity[order]


def preprocess_measured(
    q_meas: np.ndarray, intensity: np.ndarray, cfg: Config, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a measured curve into model input (1, 2, Q).

    Returns:
        x      : (1, 2, Q)  [log10-normalized R, mask]  — model input
        r_norm : (1, Q)     reflectivity-like normalized linear values (for refine's log_r_obs)
        mask   : (1, Q)     {0,1} valid-q mask
    """
    q_grid = cfg.q_grid.cpu().numpy()
    # Interpolate only within the measured q range; outside is NaN -> unmeasured (mask=0).
    i_grid = np.interp(q_grid, q_meas, intensity, left=np.nan, right=np.nan)
    mask = np.isfinite(i_grid)
    i_grid = np.clip(np.nan_to_num(i_grid, nan=0.0), 0.0, None)

    # Normalize by the max over the measured region to match the synthetic R scale (max~1).
    imax = i_grid[mask].max() if mask.any() else 1.0
    r_norm = np.clip(i_grid / max(imax, 1e-30), 1e-15, None)

    # Same input preprocessing as the training generator.
    r_log = np.clip(np.log10(r_norm), None, 0.0)
    r_log = np.where(mask, r_log, LOG_FLOOR)
    x = np.stack([r_log, mask.astype(np.float32)])[None]   # (1, 2, Q)

    def to(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32, device=device)

    return to(x), to(r_norm).unsqueeze(0), to(mask.astype(np.float32)).unsqueeze(0)


def _log_mse(r_pred: torch.Tensor, r_obs: torch.Tensor, mask: torch.Tensor) -> float:
    """log10 MSE over the valid region (same definition as the refine FOM)."""
    lp = torch.log10(r_pred.clamp_min(1e-15))
    lo = torch.log10(r_obs.clamp_min(1e-15))
    se = ((lp - lo) ** 2) * mask
    return float(se.sum() / mask.sum().clamp_min(1.0))


def load_predictor(cfg: Config, checkpoint: str | Path, stats_path: str | Path | None,
                   device: torch.device) -> tuple[XRRPhysicsModel, torch.Tensor, torch.Tensor]:
    """Model in eval mode plus the target de-normalisation (mean, std)."""
    model = _build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    st = torch.load(stats_path, map_location=device) if stats_path and Path(stats_path).exists() else param_stats(cfg)
    return model, st["mean"].to(device), st["std"].to(device)


def predict_curve(
    q_meas: np.ndarray,
    intensity: np.ndarray,
    cfg: Config,
    predictor: tuple[XRRPhysicsModel, torch.Tensor, torch.Tensor],
    *,
    refine_enabled: bool = True,
    refine_config: RefineConfig | None = None,
    alias_check: bool = False,
    mcmc_steps: int = 0,
    device: torch.device | None = None,
    name: str = "curve",
) -> tuple[dict, dict]:
    """AI prediction, refinement (from the AI thickness and, with alias_check, from half and double
    of it) and, with mcmc_steps > 0, posterior sampling for one measured curve.

    Returns (result, curves): result holds the parameters, FOMs and posterior; curves holds the q grid,
    observed, AI and refined curves and the mask, for plotting.
    """
    device = device or cfg.device
    model, mean, std = predictor
    x, r_obs, mask = preprocess_measured(q_meas, intensity, cfg, device)
    q = cfg.q_grid.to(device)
    coverage = float(mask.mean()) * 100
    logger.info(f"'{name}': {len(q_meas)} pts, q∈[{q_meas.min():.4f}, {q_meas.max():.4f}] "
                f"Å⁻¹, grid coverage {coverage:.0f}%")

    # --- AI prediction ---
    with torch.no_grad():
        pred_ai = model(x) * std + mean            # (1, 3) physical values

    # Substrate: unknown for real measurements -> fix to config range midpoint.
    sub = torch.tensor([[sum(cfg.substrate.sld) / 2, sum(cfg.substrate.roughness) / 2]],
                       device=device, dtype=torch.float32)

    log_r_obs = torch.log10(r_obs.clamp_min(1e-15))

    # --- Blue "AI predict": film structure kept as-is from AI; only intensity (i0),
    #     background (bkg), and low-angle footprint matched to the measurement.
    #     (Light refine that leaves the film parameters untouched.) ---
    with torch.enable_grad():
        _, info_ai = refine(
            pred_ai.detach(), sub, q, log_r_obs, mask, cfg.param_ranges,
            wavelength=cfg.simulation.wavelength,
            config=_ai_align_config(), device=device, verbose=False,
        )
    # dq is not fit; resolution is smeared with a fixed sigma instead (avoids the
    # film-frozen dq-absorption distortion). Reuse the augmentor's resolution smear.
    r_ai = XRRAugmentor._smear(q, info_ai["r_fit"], AI_RES_SIGMA)
    fom_ai = _log_mse(r_ai, r_obs, mask)

    # --- Red "Refined": full refine that also fits the film parameters ---
    p_final, info = pred_ai, None
    rcfg = refine_config or RefineConfig()
    if refine_enabled:
        fit = refine_multistart if alias_check else refine
        with torch.enable_grad():
            p_final, info = fit(
                pred_ai.detach(), sub, q, log_r_obs, mask, cfg.param_ranges,
                wavelength=cfg.simulation.wavelength,
                config=rcfg, device=device, verbose=not alias_check,
            )
        if alias_check:
            a = info["alias"]
            starts = ", ".join(f"{lab}: d={d:.1f} Å FOM={m:.5f}"
                               for lab, d, m in zip(a["starts"], a["thickness"][0], a["fom"][0]))
            logger.info(f"  [alias] FFT d={a['fft_thickness'][0]:.0f} Å; starts {starts} -> {a['chosen'][0]}")

    # --- Posterior: emcee around the refined state ---
    posterior = None
    if refine_enabled and mcmc_steps > 0:
        from reflectolearn.refine.mcmc import sample_posterior
        inside = (q_meas >= float(q[0])) & (q_meas <= float(q[-1]))
        posterior = sample_posterior(
            info, sub, q, log_r_obs, mask, wavelength=cfg.simulation.wavelength, config=rcfg,
            n_independent=int(inside.sum()), steps=mcmc_steps, device=device)

    # The refined curve must be refine's final fitted curve (including nuisance terms
    # like footprint/i0/bkg) so it overlays the measurement under identical conditions.
    # Without refine, fall back to the blue AI-predict curve.
    r_final = info["r_fit"] if info else r_ai
    fom_final = info["fom_after"] if info else fom_ai

    ai = pred_ai.squeeze(0).tolist()
    fin = p_final.squeeze(0).tolist()
    result = {
        "file": name,
        "ai": dict(zip(PARAM_NAMES, ai)),
        "refined": dict(zip(PARAM_NAMES, fin)) if refine_enabled else None,
        "fom_ai": fom_ai,
        "fom_refined": fom_final if refine_enabled else None,
        "substrate": {"sld": sub[0, 0].item(), "roughness": sub[0, 1].item()},
        "alias": info["alias"] if info and "alias" in info else None,
        "posterior": None if posterior is None else {
            **{k: v for k, v in posterior["params"].items() if k in PARAM_NAMES},
            "lnsigma": posterior["lnsigma"], "acceptance": posterior["acceptance"]},
    }
    curves = {"q": q, "r_obs": r_obs, "mask": mask, "r_ai": r_ai, "r_final": r_final}
    return result, curves


def predict_file(
    path: str | Path,
    cfg: Config,
    checkpoint: str | Path,
    *,
    stats_path: str | Path | None = None,
    x_axis: str = "tth",
    refine_enabled: bool = True,
    refine_config: RefineConfig | None = None,
    alias_check: bool = False,
    mcmc_steps: int = 0,
    device: torch.device | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Infer a single measured file, optionally physics-refine, then save results/plots/summary."""
    path = Path(path)
    device = device or cfg.device
    out_dir = Path(out_dir) if out_dir else path.parent
    predictor = load_predictor(cfg, checkpoint, stats_path, device)
    q_meas, intensity = load_measured_curve(path, x_axis=x_axis, wavelength=cfg.simulation.wavelength)
    result, c = predict_curve(
        q_meas, intensity, cfg, predictor, refine_enabled=refine_enabled, refine_config=refine_config,
        alias_check=alias_check, mcmc_steps=mcmc_steps, device=device, name=path.name)
    _report(result)
    _save_outputs(path, out_dir, cfg, c["q"], c["r_obs"], c["mask"], c["r_ai"], c["r_final"],
                  result, refine_enabled)
    return result


def _report(r: dict) -> None:
    logger.info(f"\n=== {r['file']} ===")
    head = f"{'param':<10} | {'AI':>12}" + ("" if r["refined"] is None else f" | {'refined':>12}")
    logger.info(head + "\n" + "-" * len(head))
    for k in PARAM_NAMES:
        line = f"{k:<10} | {r['ai'][k]:>12.3f}"
        if r["refined"] is not None:
            line += f" | {r['refined'][k]:>12.3f}"
        logger.info(f"{line}   [{UNITS[k]}]")
    if r["fom_refined"] is not None:
        b, a = r["fom_ai"], r["fom_refined"]
        imp = (b - a) / b * 100 if b else 0.0
        logger.info(f"fit FOM(LogMSE): {b:.5f} -> {a:.5f}  ({imp:+.1f}%)  (lower = better measurement match)")
    else:
        logger.info(f"fit FOM(LogMSE, AI): {r['fom_ai']:.5f}")
    if r.get("posterior"):
        p = r["posterior"]
        parts = ", ".join(f"{k} {p[k]['median']:.2f} [{p[k]['lo']:.2f}, {p[k]['hi']:.2f}]"
                          for k in PARAM_NAMES if k in p)
        spread = np.exp(p["lnsigma"]["median"])
        logger.info(f"posterior (median, 68 % interval): {parts}; residual spread {spread:.3f} in log10 R")


def _save_outputs(path, out_dir, cfg, q, r_obs, mask, r_ai, r_final,
                  result, refine_enabled) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qn = q.cpu().numpy()
    m = mask.squeeze(0).cpu().numpy() > 0
    obs = r_obs.squeeze(0).cpu().numpy()
    ai = r_ai.squeeze(0).cpu().numpy()
    fin = r_final.squeeze(0).cpu().numpy()

    # Overlay plot (measured vs predicted re-simulation)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(qn[m], np.clip(obs[m], 1e-15, None), "k-", lw=2, alpha=0.5, label="Data")
    ax.semilogy(qn, np.clip(ai, 1e-15, None), "b--", lw=1.2, alpha=0.7, label="AI predict")
    if refine_enabled:
        ax.semilogy(qn, np.clip(fin, 1e-15, None), "r-", lw=1.3, alpha=0.9, label="Refined")
    txt = "\n".join(
        f"{k}: {result['refined'][k]:.2f} {UNITS[k]}" if refine_enabled
        else f"{k}: {result['ai'][k]:.2f} {UNITS[k]}"
        for k in PARAM_NAMES)
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    # Crop the x-axis to the measured q range (hide the empty no-data tail at high q).
    if m.any():
        ax.set_xlim(qn[m].min(), qn[m].max() * 1.02)
    ax.set_xlabel("q (Å⁻¹)")
    ax.set_ylabel("Reflectivity (norm.)")
    ax.set_title(f"{path.name}  |  FOM={result['fom_refined'] or result['fom_ai']:.4f}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Low-q linear inset: the footprint rolloff is invisible on the log axis (~14
    # decades), so zoom in near the critical angle on a linear scale to reveal the
    # low-angle attenuation of measured vs refined. (Toggle off temporarily.)
    if SHOW_LOWQ_INSET and m.any():
        q_lo = qn[m][0]
        q_peak = qn[m][np.argmax(obs[m])]          # plateau max ~ critical-angle position
        q_hi = min(q_peak * 3.0, qn[m][-1])
        win = (qn >= q_lo * 0.9) & (qn <= q_hi)
        win_m = win & m
        if win_m.sum() > 3:
            axin = ax.inset_axes([0.40, 0.62, 0.34, 0.33])
            axin.plot(qn[win_m], obs[win_m], "k-", lw=2, alpha=0.5, label="Data")
            axin.plot(qn[win], ai[win], "b--", lw=1.2, alpha=0.7)
            if refine_enabled:
                axin.plot(qn[win], fin[win], "r-", lw=1.3, alpha=0.9)
            axin.axvline(q_peak, color="gray", ls=":", lw=0.8)
            axin.set_title("low-q (linear) — footprint", fontsize=8)
            axin.set_ylim(0, 1.15)
            axin.tick_params(labelsize=7)
            axin.grid(True, alpha=0.3)

    plt.tight_layout()
    png = out_dir / f"{path.stem}_predict.png"
    plt.savefig(png, dpi=120)
    plt.close()

    # Parameter summary text
    txt_path = out_dir / f"{path.stem}_predict.txt"
    lines = [f"# prediction for {path.name}"]
    for k in PARAM_NAMES:
        v = result["refined"][k] if refine_enabled else result["ai"][k]
        lines.append(f"{k}\t{v:.4f}\t{UNITS[k]}")
    lines.append(f"# FOM(LogMSE)\t{result['fom_refined'] or result['fom_ai']:.6f}")
    if result.get("posterior"):
        p = result["posterior"]
        for k in PARAM_NAMES:
            if k in p:
                lines.append(f"# posterior {k}\t{p[k]['median']:.4f}\t[{p[k]['lo']:.4f}, {p[k]['hi']:.4f}]\t{UNITS[k]}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"saved: {png.name}, {txt_path.name}  (in {out_dir})")


__all__ = ["load_measured_curve", "load_predictor", "predict_curve", "predict_file", "preprocess_measured"]
