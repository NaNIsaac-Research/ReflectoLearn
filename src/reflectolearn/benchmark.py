"""Benchmark: the full predict pipeline on two sets of synthetic curves with known truth.

"synthetic": random films made by an independent float64 Parratt recursion written here in numpy,
not by the package's torch engine, so the benchmark does not grade the pipeline with its own forward
model (the "inverse crime"). Each curve mimics a lab scan: 2θ 0.2–6° in 0.02° steps, a Gaussian beam
footprint, Gaussian resolution with constant dq/q, a background and Poisson counts. The stack is a
film on a thin SiO2 layer on Si, so the native oxide the pipeline treats as a nuisance is present.

"synthetic_fixed": a few films with the arbitrary parameters in FIXED_CASES, simulated by the
training forward model (AbelesMatrix, same stack layout as the generator) with the training
augmentor's footprint, resolution smearing, background and Poisson noise at stated values. These
curves come from the model's own simulator, so they check the pipeline in its training domain, not
its transfer to independent physics.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.special import erf

SIO2 = 18.9 + 0.24j  # 1e-6 / Å^2
SI = 20.07 + 0.46j


def parratt(q: np.ndarray, layers: list[tuple[complex, float, float]]) -> np.ndarray:
    """|r|^2 for [(sld, thickness, roughness of its top interface)], ambient first, substrate last.

    SLD in 1e-6 Å^-2 with positive imaginary part for absorption; Névot–Croce roughness. The SLD is
    conjugated so that the principal square root gives Im k > 0 (decay into absorbers) in every layer.
    """
    sld = np.conj(np.array([lay[0] for lay in layers])) * 1e-6
    kz = np.sqrt((q[:, None] / 2) ** 2 - 4 * np.pi * (sld[None, :] - sld[0]) + 0j)
    x = np.zeros(len(q), complex)
    for j in range(len(layers) - 2, -1, -1):
        k1, k2 = kz[:, j], kz[:, j + 1]
        r = (k1 - k2) / (k1 + k2) * np.exp(-2 * k1 * k2 * layers[j + 1][2] ** 2)
        d = layers[j + 1][1] if j + 1 < len(layers) - 1 else 0.0
        p = np.exp(2j * k2 * d)
        x = (r + x * p) / (1 + r * x * p)
    return np.abs(x) ** 2


def smeared(q: np.ndarray, layers, dq_over_q: float, n: int = 17) -> np.ndarray:
    """Reflectivity convolved with a Gaussian of sigma = dq_over_q · q (Gauss–Hermite-like grid, ±3σ)."""
    if dq_over_q <= 0:
        return parratt(q, layers)
    u = np.linspace(-3, 3, n)
    w = np.exp(-0.5 * u**2)
    w /= w.sum()
    qq = (q[:, None] * (1 + dq_over_q * u[None, :])).clip(1e-6)
    return (parratt(qq.ravel(), layers).reshape(qq.shape) * w).sum(1)


@dataclass
class SyntheticCase:
    name: str
    thickness: float
    roughness: float
    sld: float
    isld: float
    oxide_thickness: float
    oxide_roughness: float
    substrate_roughness: float
    sample_length_mm: float
    dq_over_q: float
    i0: float
    background: float


def make_synthetic(
    n: int, seed: int = 2026, wavelength: float = 1.5406, tth: np.ndarray | None = None
) -> list[tuple[SyntheticCase, np.ndarray, np.ndarray]]:
    """n synthetic curves: list of (case, 2θ [deg], counts)."""
    rng = np.random.default_rng(seed)
    tth = np.arange(0.2, 6.0001, 0.02) if tth is None else tth
    q = 4 * np.pi / wavelength * np.sin(np.radians(tth / 2))
    beam_sigma = 0.1 / 2.3548  # mm, Gaussian beam of FWHM 0.1 mm
    out = []
    for i in range(n):
        d = rng.uniform(100, 450)
        c = SyntheticCase(
            name=f"case_{i:02d}",
            thickness=d,
            roughness=min(rng.uniform(2, 14), 0.3 * d),
            sld=rng.uniform(30, 70),
            isld=rng.uniform(0.5, 5),
            oxide_thickness=rng.uniform(5, 25),
            oxide_roughness=rng.uniform(2, 6),
            substrate_roughness=rng.uniform(2, 6),
            sample_length_mm=rng.uniform(8, 25),
            dq_over_q=rng.uniform(0.01, 0.04),
            i0=10 ** rng.uniform(6, 8),
            background=10 ** rng.uniform(-8, -6),
        )
        layers = [
            (0j, 0.0, 0.0),
            (complex(c.sld, c.isld), c.thickness, c.roughness),
            (SIO2, c.oxide_thickness, c.oxide_roughness),
            (SI, 0.0, c.substrate_roughness),
        ]
        r = smeared(q, layers, c.dq_over_q)
        r = r * erf(
            c.sample_length_mm * (q * wavelength / (4 * np.pi)) / beam_sigma / (2 * np.sqrt(2))
        )
        counts = rng.poisson((r + c.background) * c.i0).astype(float).clip(1.0)
        out.append((c, tth.copy(), counts))
    return out


@dataclass
class FixedCase:
    """A film with stated parameters plus the instrument effects applied to its curve.

    Units as in the config: Å, 1e-6 Å^-2; beam_width and sample_length as in
    augmentation.beam_width_range / sample_len_range; res_sigma in Å^-1; bkg and poisson_scale
    as log10 values (augmentation.bkg / poisson_log_scale). oxide_thickness 0 means no oxide.
    """

    name: str
    thickness: float
    roughness: float
    sld: float
    oxide_thickness: float
    oxide_sld: float
    oxide_roughness: float
    sub_sld: float
    sub_roughness: float
    beam_width: float
    sample_length: float
    res_sigma: float
    bkg: float
    poisson_scale: float


# Arbitrary values inside the default training ranges; not fitted to any sample.
FIXED_CASES = (
    FixedCase(
        "fixed_thin",
        thickness=90.0,
        roughness=3.0,
        sld=28.0,
        oxide_thickness=0.0,
        oxide_sld=0.0,
        oxide_roughness=0.0,
        sub_sld=20.0,
        sub_roughness=3.0,
        beam_width=0.01,
        sample_length=12.0,
        res_sigma=0.001,
        bkg=-7.0,
        poisson_scale=7.5,
    ),
    FixedCase(
        "fixed_mid",
        thickness=240.0,
        roughness=6.0,
        sld=45.0,
        oxide_thickness=12.0,
        oxide_sld=18.0,
        oxide_roughness=3.0,
        sub_sld=20.0,
        sub_roughness=2.5,
        beam_width=0.01,
        sample_length=20.0,
        res_sigma=0.002,
        bkg=-6.5,
        poisson_scale=8.0,
    ),
    FixedCase(
        "fixed_thick",
        thickness=600.0,
        roughness=4.5,
        sld=78.0,
        oxide_thickness=20.0,
        oxide_sld=15.0,
        oxide_roughness=4.0,
        sub_sld=21.0,
        sub_roughness=4.0,
        beam_width=0.012,
        sample_length=25.0,
        res_sigma=0.0015,
        bkg=-7.5,
        poisson_scale=7.5,
    ),
)


def make_fixed(
    cfg, seed: int = 2026, cases=FIXED_CASES, tth_range: tuple[float, float] = (0.2, 8.0)
) -> list[tuple[FixedCase, np.ndarray, np.ndarray]]:
    """Synthetic curves of `cases` from the training forward model: list of (case, q [Å^-1], intensity).

    Same media layout as data.generator.simulate_reflectivity, then the XRRAugmentor footprint and
    smearing, background and Poisson noise at the case's values, cropped to `tth_range` [deg].
    The seed only drives the Poisson draw.
    """
    import torch

    from reflectolearn.data.augmentations import XRRAugmentor
    from reflectolearn.physics import AbelesMatrix, tth2q

    wl = cfg.simulation.wavelength
    q = cfg.q_grid.cpu().float()
    keep = ((q >= tth2q(tth_range[0], wl)) & (q <= tth2q(tth_range[1], wl))).numpy()
    aug = XRRAugmentor(cfg, q)
    gen = torch.Generator().manual_seed(seed)
    out = []
    for c in cases:
        ox = c.oxide_thickness > 0
        sld = torch.tensor([[0.0, c.oxide_sld if ox else c.sld, c.sld, c.sub_sld]])
        thk = torch.tensor([[0.0, c.oxide_thickness, c.thickness]])
        rgh = torch.tensor(
            [
                [
                    c.oxide_roughness if ox else c.roughness,
                    c.roughness if ox else 0.0,
                    c.sub_roughness,
                ]
            ]
        )
        r = AbelesMatrix()(q[None], thk, rgh, sld, wavelength=wl)
        r = aug._footprint(q, r, c.beam_width, c.sample_length)
        r = XRRAugmentor._smear(q, r, c.res_sigma) + 10.0**c.bkg
        s = 10.0**c.poisson_scale
        r = torch.poisson((r * s).clamp_min(0.0), generator=gen) / s
        out.append((c, q.numpy()[keep].astype(np.float64), r[0].numpy()[keep].astype(np.float64)))
    return out


def score(kind: str, name: str, res: dict, truth: dict, seconds: float) -> dict:
    """One benchmark row from a predict_curve result and the known truth."""
    got = res["refined"]
    row = {"set": kind, "case": name, "seconds": round(seconds, 1), "fom": res["fom_refined"]}
    for k in ("thickness", "roughness", "sld"):
        row[f"{k}_true"], row[f"{k}_ai"], row[f"{k}"] = truth[k], res["ai"][k], got[k]
    row["thickness_err_pct"] = 100 * (got["thickness"] - truth["thickness"]) / truth["thickness"]
    if res.get("alias"):
        row["alias_start"] = res["alias"]["chosen"][0]
        row["fft_thickness"] = res["alias"]["fft_thickness"][0]
    if res.get("posterior"):
        p = res["posterior"]["thickness"]
        row["thickness_lo"], row["thickness_hi"] = p["lo"], p["hi"]
        row["thickness_in_68"] = p["lo"] <= truth["thickness"] <= p["hi"]
    return row


def summarize(rows: list[dict]) -> dict:
    """Per-set error statistics of the rows from `score`."""
    summary = {}
    for kind in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == kind]
        err = np.array([abs(r["thickness_err_pct"]) for r in rs])
        summary[kind] = {
            "n": len(rs),
            "thickness_median_abs_err_pct": float(np.median(err)),
            "thickness_fail_gt10pct": int((err > 10).sum()),
            "roughness_mae": float(
                np.mean([abs(r["roughness"] - r["roughness_true"]) for r in rs])
            ),
            "sld_mae": float(np.mean([abs(r["sld"] - r["sld_true"]) for r in rs])),
            "seconds_mean": float(np.mean([r["seconds"] for r in rs])),
        }
        if any("thickness_in_68" in r for r in rs):
            summary[kind]["thickness_in_68pct_interval"] = float(
                np.mean([r.get("thickness_in_68", False) for r in rs])
            )
    return summary


def run_benchmark(
    cfg,
    checkpoint,
    *,
    stats_path=None,
    n_synthetic: int = 12,
    seed: int = 2026,
    alias_check: bool = False,
    mcmc_steps: int = 0,
    out_dir: str | Path = "benchmark_out",
    device=None,
) -> dict:
    """Run predict on both synthetic sets and score them; writes JSON and CSV."""
    from reflectolearn.physics import tth2q
    from reflectolearn.predict import load_predictor, predict_curve

    device = device or cfg.device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wl = cfg.simulation.wavelength
    predictor = load_predictor(cfg, checkpoint, stats_path, device)
    rows = []

    def run(kind, name, q, counts, truth):
        t0 = time.time()
        res, _ = predict_curve(
            q,
            counts,
            cfg,
            predictor,
            alias_check=alias_check,
            mcmc_steps=mcmc_steps,
            device=device,
            name=name,
        )
        rows.append(score(kind, name, res, truth, time.time() - t0))

    for case, tth, counts in make_synthetic(n_synthetic, seed=seed, wavelength=wl):
        run("synthetic", case.name, tth2q(tth, wl), counts, asdict(case))
    for case, q, intensity in make_fixed(cfg, seed=seed):
        run("synthetic_fixed", case.name, q, intensity, asdict(case))

    summary = summarize(rows)
    (out_dir / "benchmark.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("set", "case"), k))
    with open(out_dir / "benchmark.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for kind, s in summary.items():
        logger.info(
            f"[{kind}] n={s['n']}  thickness median |err| {s['thickness_median_abs_err_pct']:.2f} %, "
            f">10 % off: {s['thickness_fail_gt10pct']}, roughness MAE {s['roughness_mae']:.2f} Å, "
            f"SLD MAE {s['sld_mae']:.2f}"
            + (
                f", truth in 68 % interval: {100 * s['thickness_in_68pct_interval']:.0f} %"
                if "thickness_in_68pct_interval" in s
                else ""
            )
        )
    return {"summary": summary, "rows": rows}


__all__ = [
    "FIXED_CASES",
    "FixedCase",
    "SyntheticCase",
    "make_fixed",
    "make_synthetic",
    "parratt",
    "run_benchmark",
    "score",
    "smeared",
    "summarize",
]
