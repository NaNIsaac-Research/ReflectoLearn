"""Figures for the data slides.

Slide 2 ("Domain Randomization"): a 3x3 panel grid that takes one clean simulated
XRR curve and overlays each randomized effect (oxide layer, 2theta offset, Cu Kalpha2
doublet, footprint, resolution, intensity scale, background, Poisson noise) plus a
final "all combined" realistic curve.

All effects reuse the production code paths (AbelesMatrix, apply_tth_offset,
XRRAugmentor) so the figure stays faithful to what the generator actually does.
Everything is deterministic (fixed film params + fixed seed) for reproducible slides.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from reflectolearn.config import Config
from reflectolearn.data.augmentations import XRRAugmentor
from reflectolearn.physics import AbelesMatrix, apply_tth_offset


@dataclass
class FilmCase:
    """Representative single film on a substrate (units match config)."""

    thickness: float = 150.0   # [Å] (kept modest so fringes stay visually distinct)
    roughness: float = 5.0     # [Å]
    sld: float = 40.0          # [x1e-6 Å^-2]
    sub_sld: float = 20.0      # [x1e-6 Å^-2]
    sub_rough: float = 3.0     # [Å]
    # Oxide (nuisance) layer used in the oxide / combined panels.
    oxide_thickness: float = 30.0
    oxide_sld: float = 18.0
    oxide_roughness: float = 3.0


def _abeles(q: Tensor, film: FilmCase, wl: float, *, with_oxide: bool = False) -> Tensor:
    """Reflectivity R(q) for the film (optionally with the oxide layer). Returns (Q,)."""
    ab = AbelesMatrix(device=q.device)
    q_b = q.unsqueeze(0)  # (1, Q)
    if with_oxide:
        # media: [ambient, oxide, film, substrate]
        sld_t = torch.tensor([[0.0, film.oxide_sld, film.sld, film.sub_sld]])
        thk_t = torch.tensor([[0.0, film.oxide_thickness, film.thickness]])
        rgh_t = torch.tensor([[film.oxide_roughness, film.roughness, film.sub_rough]])
    else:
        # media: [ambient, film, substrate]
        sld_t = torch.tensor([[0.0, film.sld, film.sub_sld]])
        thk_t = torch.tensor([[0.0, film.thickness]])
        rgh_t = torch.tensor([[film.roughness, film.sub_rough]])
    R = ab.forward(q_b, thk_t, rgh_t, sld_t, wavelength=wl)
    return R.squeeze(0).clamp_min(1e-15)


# --- individual effects: each takes the clean curve / q and returns a modified R (Q,) ---

def _eff_tth_offset(q: Tensor, film: FilmCase, wl: float, offset_deg: float = 0.02) -> Tensor:
    q_eff = apply_tth_offset(q, torch.tensor([[offset_deg]]), wl).squeeze(0)
    return _abeles(q_eff, film, wl)


def _eff_kalpha2(q: Tensor, R: Tensor, film: FilmCase, wl: float,
                 ratio: float = 0.5, wl2: float = 1.54439) -> Tensor:
    R2 = _abeles(q * (wl / wl2), film, wl)
    return (R + ratio * R2) / (1.0 + ratio)


def _eff_footprint(aug: XRRAugmentor, q: Tensor, R: Tensor,
                   beam_width: float = 0.01, sample_len: float = 10.0) -> Tensor:
    return aug._footprint(q, R.unsqueeze(0), beam_width, sample_len).squeeze(0)


def _eff_resolution(q: Tensor, R: Tensor, sigma_q: float = 0.004) -> Tensor:
    return XRRAugmentor._smear(q, R.unsqueeze(0), sigma_q).squeeze(0)


def _eff_poisson(R: Tensor, log_scale: float = 7.0, seed: int = 0) -> Tensor:
    g = torch.Generator(device=R.device).manual_seed(seed)
    s = 10.0 ** log_scale
    lam = (R * s).clamp_min(0.0)
    return torch.poisson(lam, generator=g) / s


def domain_randomization_figure(
    cfg: Config | None = None,
    film: FilmCase | None = None,
    out: str | Path = "domain_randomization.png",
    *,
    seed: int = 0,
) -> Path:
    """Render the slide-2 domain-randomization panel grid and save it as PNG."""
    cfg = cfg or Config()
    film = film or FilmCase()
    torch.manual_seed(seed)

    q = cfg.q_grid.cpu().float()
    wl = cfg.simulation.wavelength
    aug = XRRAugmentor(cfg, q)

    clean = _abeles(q, film, wl)

    # (title, modified R, color) for each panel; clean curve is drawn under each.
    panels: list[tuple[str, Tensor, str]] = [
        ("Oxide layer", _abeles(q, film, wl, with_oxide=True), "tab:blue"),
        ("2theta offset", _eff_tth_offset(q, film, wl), "tab:orange"),
        ("Cu Kalpha2 doublet", _eff_kalpha2(q, clean, film, wl), "tab:green"),
        ("Footprint", _eff_footprint(aug, q, clean), "tab:red"),
        ("Resolution smearing", _eff_resolution(q, clean), "tab:purple"),
        ("Intensity scale (i0)", clean * 0.3, "tab:brown"),
        ("Background (bkg)", clean + 1e-6, "tab:pink"),
        ("Poisson noise", _eff_poisson(clean), "tab:gray"),
    ]

    # Combined: apply every effect in physical order, then mask the q-axis ends.
    q_off = apply_tth_offset(q, torch.tensor([[0.02]]), wl).squeeze(0)
    combined = _abeles(q_off, film, wl, with_oxide=True)
    combined = _eff_kalpha2(q, combined, film, wl)
    combined = _eff_footprint(aug, q, combined)
    combined = _eff_resolution(q, combined)
    combined = combined * 0.3 + 1e-6
    combined = _eff_poisson(combined, seed=seed)
    Q = q.shape[0]
    keep = torch.zeros(Q, dtype=torch.bool)
    keep[int(0.05 * Q):int(0.95 * Q)] = True
    combined_masked = combined.clone()
    combined_masked[~keep] = float("nan")
    panels.append(("All combined + masking", combined_masked, "black"))

    qn = q.numpy()
    cl = clean.clamp_min(1e-15).numpy()

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for ax, (title, R, color) in zip(axes.ravel(), panels):
        ax.semilogy(qn, cl, color="0.7", lw=1.0, ls="--", label="clean")
        ax.semilogy(qn, R.clamp_min(1e-15).numpy() if torch.is_tensor(R) else R,
                    color=color, lw=1.3, label="randomized")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(qn[0], 0.6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("q (Å⁻¹)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reflectivity")
    fig.suptitle("Domain Randomization: clean simulation vs each injected effect",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(out)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _systematic_curve(q: Tensor, film: FilmCase, wl: float, aug: XRRAugmentor) -> Tensor:
    """Clean film curve with all *deterministic* (systematic) effects applied, no noise.

    Order: oxide structure -> 2theta offset -> Kalpha2 -> footprint -> resolution
    -> intensity scale -> background. (Masking/Poisson are added by the caller.)
    """
    q_off = apply_tth_offset(q, torch.tensor([[0.02]]), wl).squeeze(0)
    R = _abeles(q_off, film, wl, with_oxide=True)
    R = _eff_kalpha2(q, R, film, wl)
    R = _eff_footprint(aug, q, R)
    R = _eff_resolution(q, R)
    return R * 0.3 + 1e-6


def sim_to_real_figure(
    cfg: Config | None = None,
    film: FilmCase | None = None,
    out: str | Path = "sim_to_real.png",
    *,
    seed: int = 0,
) -> Path:
    """Single-axes figure: clean simulation -> + systematic distortions -> + stochastic noise.

    Visualizes the sim-to-real gap as two stacked layers: deterministic (systematic)
    distortions reshape the curve, and Poisson (stochastic) noise adds randomness.
    """
    cfg = cfg or Config()
    film = film or FilmCase()
    torch.manual_seed(seed)

    q = cfg.q_grid.cpu().float()
    wl = cfg.simulation.wavelength
    aug = XRRAugmentor(cfg, q)

    clean = _abeles(q, film, wl)
    systematic = _systematic_curve(q, film, wl, aug)
    realistic = _eff_poisson(systematic, seed=seed)

    # Mask the q-axis ends (unmeasured region) on the systematic / realistic curves.
    Q = q.shape[0]
    keep = torch.zeros(Q, dtype=torch.bool)
    keep[int(0.05 * Q):int(0.95 * Q)] = True

    def _mask(R: Tensor):
        out_r = R.clamp_min(1e-15).clone()
        out_r[~keep] = float("nan")
        return out_r.numpy()

    qn = q.numpy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(qn, clean.clamp_min(1e-15).numpy(), color="0.6", lw=1.4, ls="--",
                label="Clean simulation (ideal)")
    ax.semilogy(qn, _mask(systematic), color="tab:blue", lw=1.6,
                label="+ Systematic distortions")
    ax.semilogy(qn, _mask(realistic), color="black", lw=0.9, alpha=0.85,
                label="+ Stochastic noise (realistic)")
    ax.set_xlim(qn[0], 0.6)
    ax.set_xlabel("q (Å⁻¹)")
    ax.set_ylabel("Reflectivity")
    ax.set_title("Sim-to-Real Gap: systematic distortions + stochastic noise")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(out)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def model_architecture_figure(
    cfg: Config | None = None,
    out: str | Path = "model_architecture.png",
) -> Path:
    """Left-to-right block diagram of XRRPhysicsModel (curve -> film parameters).

    Stages and shapes are derived from the config so the diagram stays accurate:
    Input (2,Q) -> Fourier features -> 7x Conv block -> Global Avg Pool -> MLP -> 3 params.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    cfg = cfg or Config()
    m = cfg.model
    Q = cfg.simulation.q_points
    depth = m.depth
    fch = m.fourier_features
    # encoder channel progression: n_channels * 2**min(i, 3)
    chans = [m.n_channels * (2 ** min(i, 3)) for i in range(depth)]
    lengths = [Q // (2 ** (i + 1)) for i in range(depth)]

    stages = [
        ("Input", [f"(2, {Q})", "[log10 R, mask]"], "#bcd9b0"),
        ("Fourier\nfeatures", [f"q -> {fch} freq", f"sin/cos -> {2 * fch} ch",
                               f"concat -> {2 + 2 * fch} ch"], "#aac8e6"),
        (f"Conv block x{depth}", ["Conv1d(k=7) / BN", "LeakyReLU / Dropout", "MaxPool(2)",
                                  f"ch {chans[0]}->{chans[-1]}", f"len {Q}->{lengths[-1]}"],
         "#f3c79a"),
        ("Global\nAvg Pool", [f"-> ({chans[-1]},)"], "#cdb6e0"),
        ("MLP", [f"{chans[-1]} -> {m.mlp_hidden}", f"{m.mlp_hidden} -> {m.mlp_hidden // 2}",
                 f"{m.mlp_hidden // 2} -> 3"], "#e8a9a9"),
        ("Output", ["(3,)", "thickness", "roughness", "SLD"], "#bcd9b0"),
    ]

    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.set_xlim(0, len(stages) * 2.4)
    ax.set_ylim(0, 4)
    ax.axis("off")

    box_w, box_h, gap = 1.8, 2.2, 0.6
    centers = []
    x = 0.3
    for title, lines, color in stages:
        cy = 2.0
        box = FancyBboxPatch((x, cy - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.05,rounding_size=0.12",
                             linewidth=1.4, edgecolor="#444", facecolor=color)
        ax.add_patch(box)
        ax.text(x + box_w / 2, cy + box_h / 2 - 0.32, title, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(x + box_w / 2, cy - 0.25, "\n".join(lines), ha="center", va="center",
                fontsize=9, family="monospace")
        centers.append((x + box_w, cy, x))
        x += box_w + gap

    for i in range(len(stages) - 1):
        x_end = centers[i][0]
        x_start = centers[i + 1][2]
        ax.add_patch(FancyArrowPatch((x_end, 2.0), (x_start, 2.0),
                                     arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.6, color="#333"))

    ax.set_title("XRRPhysicsModel: measured curve -> film parameters",
                 fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()

    out = Path(out)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


DEFAULT_REFINE_DATA = Path(
    r"C:\WorkSpace\05_Resources\Data\XRR_AI\XRR_data_sorted\Ga2O3\unknown_t"
    r"\ALD_GPC_XRR__Ga2O3_merged_0.2_8.txt"
)


def refine_figure(
    cfg: Config | None = None,
    data_path: str | Path = DEFAULT_REFINE_DATA,
    out: str | Path = "refine.png",
    *,
    checkpoint: str | Path | None = None,
    x_axis: str = "tth",
) -> Path:
    """Real-data physics-refinement figure: run the full inference + refine pipeline on a
    measured curve and produce the measured / AI-predict / refined overlay (with FOM).

    Mirrors the CLI 'predict' resolution: checkpoint defaults to config's save_dir/best.pt,
    and config/stats next to that checkpoint are auto-discovered. The plot is rendered by
    predict_file; this wrapper just resolves paths and copies the PNG to `out`.
    """
    import shutil

    from reflectolearn.predict import predict_file

    cfg = cfg or (Config.load_yaml(Path("config.yaml")) if Path("config.yaml").exists()
                  else Config())
    data_path = Path(data_path)
    out = Path(out)

    # Resolve checkpoint, then auto-discover config/stats next to it (as the CLI does).
    ckpt = Path(checkpoint) if checkpoint is not None else cfg.save_dir / "best.pt"
    exp_dir = ckpt.parent
    if (exp_dir / "config.yaml").exists():
        cfg = Config.load_yaml(exp_dir / "config.yaml")
    stats = exp_dir / "stats.pt" if (exp_dir / "stats.pt").exists() else None

    predict_file(
        data_path, cfg, ckpt,
        stats_path=stats, x_axis=x_axis, refine_enabled=True,
        device=cfg.device, out_dir=out.parent,
    )

    produced = out.parent / f"{data_path.stem}_predict.png"
    if produced.resolve() != out.resolve():
        shutil.copyfile(produced, out)
    return out


def refine_process_figure(out: str | Path = "refine_process.png") -> Path:
    """Block diagram of the refinement gradient-descent loop.

    AI init -> [Abeles forward -> LogMSE vs measured -> backprop -> Adam update] -> theta*.
    Schematic only (no computation), for the 'refinement principle' slide.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    bw, bh = 3.0, 1.5
    # (key, x, y, title, lines, color) — 4 nodes of the loop placed at rectangle corners.
    nodes = {
        "theta": (1.2, 5.2, "Parameters  θ",
                  ["film + nuisance", "sigmoid z -> [lo, hi]"], "#aac8e6"),
        "fwd":   (7.8, 5.2, "Abeles forward",
                  ["differentiable", "R_sim(θ)"], "#f3c79a"),
        "loss":  (7.8, 1.3, "Loss  (LogMSE)",
                  ["compare to R_obs", "masked, valid-q"], "#e8a9a9"),
        "upd":   (1.2, 1.3, "Adam update",
                  ["θ ← θ − lr·∇θ", "trust-region bound"], "#cdb6e0"),
    }
    cx = {}
    for key, (x, y, title, lines, color) in nodes.items():
        box = FancyBboxPatch((x, y), bw, bh,
                             boxstyle="round,pad=0.05,rounding_size=0.12",
                             linewidth=1.5, edgecolor="#444", facecolor=color)
        ax.add_patch(box)
        ax.text(x + bw / 2, y + bh - 0.42, title, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(x + bw / 2, y + 0.42, "\n".join(lines), ha="center", va="center",
                fontsize=9, family="monospace")
        cx[key] = (x, y, x + bw / 2, y + bh / 2, x + bw, y + bh)

    def arrow(p0, p1, label="", lx=0.0, ly=0.0, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                                     linewidth=1.8, color="#333",
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = (p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly
            ax.text(mx, my, label, ha="center", va="center", fontsize=9, color="#222")

    # Clockwise loop: theta -> forward -> loss -> update -> theta.
    arrow((cx["theta"][4], 5.95), (cx["fwd"][0], 5.95), "simulate", ly=0.35)
    arrow((9.3, cx["fwd"][1]), (9.3, cx["loss"][1] + bh), "evaluate fit", lx=1.15)
    arrow((cx["loss"][0], 2.05), (cx["upd"][4], 2.05), "∇θ  backprop", ly=0.35)
    arrow((2.7, cx["upd"][1] + bh), (2.7, cx["theta"][1]), "step", lx=-0.95)

    # External: AI init in, measured in, refined out.
    arrow((0.1, 5.95), (cx["theta"][0], 5.95))
    ax.text(0.1, 6.45, "AI init  θ₀", ha="left", va="center", fontsize=10,
            fontweight="bold", color="tab:blue")
    arrow((11.9, 2.05), (cx["loss"][4], 2.05))
    ax.text(11.9, 2.55, "R_obs\n(measured)", ha="right", va="center", fontsize=10,
            fontweight="bold", color="0.3")
    arrow((cx["upd"][2], cx["upd"][1]), (cx["upd"][2], 0.2))
    ax.text(cx["upd"][2] + 0.2, 0.35, "θ*  (refined)  — repeat N steps per stage",
            ha="left", va="center", fontsize=10, fontweight="bold", color="tab:red")

    ax.set_title("Refinement loop: differentiable physics fit (AI init → θ*)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    # Use the project config.yaml (actual trained model dims) when present.
    _cfg_path = Path("config.yaml")
    _cfg = Config.load_yaml(_cfg_path) if _cfg_path.exists() else Config()
    print(f"saved: {domain_randomization_figure(_cfg).resolve()}")
    print(f"saved: {sim_to_real_figure(_cfg).resolve()}")
    print(f"saved: {model_architecture_figure(_cfg).resolve()}")
    print(f"saved: {refine_figure(_cfg).resolve()}")
    print(f"saved: {refine_process_figure().resolve()}")
