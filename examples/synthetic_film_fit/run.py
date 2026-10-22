"""Synthetic end-to-end fit: a known film -> simulated counts -> ReflectoLearn's fit -> parameters.

Runs the package's public path on a curve whose answer is known, without a trained network:
the film thickness starts from the Kiessig-fringe FFT estimate and refine (the fit predict uses) fits the
film plus the nuisances (scale, background, offset, resolution, footprint, native oxide).

    uv run python examples/synthetic_film_fit/run.py

Prints the fitted parameters next to the truth and writes result.json beside this file.
"""

import json
from pathlib import Path

import numpy as np
import torch

from reflectolearn.benchmark import SI, SIO2, smeared
from reflectolearn.config import Config
from reflectolearn.physics import tth2q
from reflectolearn.predict import preprocess_measured
from reflectolearn.refine import RefineConfig, fft_thickness, refine

HERE = Path(__file__).parent
SEED = 2026

# The "absorbing film on SiO2 on Si" stack of tests/golden/xrr.json.
# SLD in 1e-6 Å^-2 (imaginary part > 0 absorbs), thickness and rms roughness in Å.
TRUTH = {
    "film": {"sld": [45.0, 2.0], "thickness": 250.0, "roughness": 6.0},
    "oxide": {"sld": [SIO2.real, SIO2.imag], "thickness": 15.0, "roughness": 3.0},
    "substrate": {"sld": [SI.real, SI.imag], "roughness": 4.0},
}
# A lab Cu Kα scan: 2θ from 0.2° to 6° in 0.02° steps, 1e7 counts on the plateau,
# Gaussian resolution dq/q = 2 %, flat background 1e-7 (relative to R).
INSTRUMENT = {"tth_deg": [0.2, 6.0, 0.02], "i0": 1e7, "background": 1e-7, "dq_over_q": 0.02}


def layers() -> list[tuple[complex, float, float]]:
    """TRUTH as benchmark.parratt layers: (SLD, thickness, top roughness), ambient first."""
    t = TRUTH
    return [
        (0j, 0.0, 0.0),
        (complex(*t["film"]["sld"]), t["film"]["thickness"], t["film"]["roughness"]),
        (complex(*t["oxide"]["sld"]), t["oxide"]["thickness"], t["oxide"]["roughness"]),
        (complex(*t["substrate"]["sld"]), 0.0, t["substrate"]["roughness"]),
    ]


def scan_q(cfg: Config) -> np.ndarray:
    lo, hi, step = INSTRUMENT["tth_deg"]
    return tth2q(np.arange(lo, hi + step / 2, step), cfg.simulation.wavelength)


def simulate(cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Counts on the scan's q points from an independent float64 Parratt engine plus Poisson noise."""
    q = scan_q(cfg)
    r = smeared(q, layers(), INSTRUMENT["dq_over_q"])
    counts = rng.poisson((r + INSTRUMENT["background"]) * INSTRUMENT["i0"]).astype(float)
    return q, counts.clip(1.0)


def run(seed: int = SEED, config: RefineConfig | None = None) -> dict:
    torch.manual_seed(seed)
    cfg = Config(device="cpu")
    q_meas, counts = simulate(cfg, np.random.default_rng(seed))

    # The public path a measured curve takes: interpolate onto the model grid, normalise, mask.
    _, r_obs, mask = preprocess_measured(q_meas, counts, cfg, cfg.device)
    log_r_obs = torch.log10(r_obs.clamp_min(1e-15))
    q = cfg.q_grid

    # Start without a network: thickness from the fringe period, roughness and SLD mid-range.
    sld0, rough0 = 40.0, 5.0
    d0 = fft_thickness(
        q, log_r_obs, mask, tuple(cfg.param_ranges.thickness), sld=torch.tensor([sld0])
    )
    p_init = torch.tensor([[float(d0[0]), rough0, sld0]])
    substrate = torch.tensor([[sum(cfg.substrate.sld) / 2, sum(cfg.substrate.roughness) / 2]])

    p_fit, info = refine(
        p_init,
        substrate,
        q,
        log_r_obs,
        mask,
        cfg.param_ranges,
        wavelength=cfg.simulation.wavelength,
        config=config or RefineConfig(),
        device=cfg.device,
    )
    d, sigma, sld = (float(v) for v in p_fit[0])
    # refine returns a nuisance only when it is fitted; None marks one that was held off
    nuisance = {
        k: float(info[k][0]) if k in info else None for k in ("tth_offset", "oxide_thickness")
    }
    return {
        "seed": seed,
        "fft_thickness": float(d0[0]),
        "fit": {"thickness": d, "roughness": sigma, "sld": sld},
        "fom_after": float(info["fom_after"]),
        **nuisance,
        "q_measured": [float(q_meas[0]), float(q_meas[-1])],
        "points": int(mask.sum()),
    }


if __name__ == "__main__":
    result = run()
    (HERE / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    film = TRUTH["film"]
    print(f"{'':10s}{'truth':>10s}{'fit':>10s}")
    for key, unit in (("thickness", "Å"), ("roughness", "Å"), ("sld", "1e-6 Å⁻²")):
        true = film[key][0] if key == "sld" else film[key]
        print(f"{key:10s}{true:10.2f}{result['fit'][key]:10.2f}  {unit}")
    print(
        f"FFT start {result['fft_thickness']:.0f} Å, log10-MSE after fit {result['fom_after']:.4f}"
    )
