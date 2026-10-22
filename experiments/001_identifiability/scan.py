"""Experiment 001, R2a: where does a single-film or bilayer XRR curve have several solutions?

Decision rules are pre-registered in docs/identifiability/generative_go_nogo.md ("Pre-registered
experiment for 2026-10-23: condition scan for multimodality"). Per condition, on noise-free data
(the expected counts at the truth):

    starts   64 Latin-hypercube starts over the full prior box; Poisson least squares from each
    modes    endpoints with Δχ² ≤ 4 grouped into modes by their film parameters
    profile  Δχ² = 1 half-widths of every film parameter at each mode's best endpoint

The forward model is fim.py's (abeles_core in complex128, top oxide, Kα2, footprint, 2θ offset,
constant-dq resolution, i0, background), extended to a second film under the first.

Run:  uv run python experiments/001_identifiability/scan.py check
      uv run python experiments/001_identifiability/scan.py run --conditions 0 1 2
      uv run python experiments/001_identifiability/scan.py report
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import fim
import numpy as np
import torch
from fim import SUB_ROUGH, SUB_SLD, WAVELENGTH, WAVELENGTH_KA2
from nonunique import BOUNDS, OUT, deviance_terms
from scipy.optimize import least_squares
from scipy.stats import qmc
from torch import Tensor
from torch.func import jacfwd

from reflectolearn.benchmark import parratt
from reflectolearn.physics import abeles_core, apply_tth_offset

S_COUNTS = 1e6
N_STARTS = 64
TTH_MAX = (2.2, 3.0, 6.0)  # deg -> q_max 0.16, 0.21, 0.43 Å^-1
FILM_SLD = (12.0, 24.0, 45.0)
NUISANCE = ("refine", "wide")
STACK = ("single", "bilayer")
WIDE = {"oxide_sld": (5.0, 40.0), "oxide_thickness": (0.0, 60.0), "kalpha2_ratio": (0.0, 1.0)}
SECOND = {"thickness2": 120.0, "roughness2": 6.0, "sld_factor": 1.25}
FILM1 = ("thickness", "roughness", "sld")
FILM2 = ("thickness2", "roughness2", "sld2")
NUIS = (
    "i0",
    "bkg",
    "tth_offset",
    "dq",
    "kalpha2_ratio",
    "footprint",
    "oxide_thickness",
    "oxide_sld",
    "oxide_roughness",
)
SCAN_DIR = OUT / "scan"
# Resolution quadrature: Gauss-Hermite nodes for the Gaussian weight. 15 nodes differ from fim.py's
# 41-point ±5σ rule by a deviance of < 0.002 at 10^6 counts and cost a third as much.
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_U, GH_W = torch.as_tensor(_GH_X), torch.as_tensor(_GH_W / _GH_W.sum())


# ------------------------------------------------------------------ conditions
def conditions() -> list[dict]:
    return [
        {"tth_max": t, "film_sld": s, "nuisance": n, "stack": k}
        for k, n, t, s in itertools.product(STACK, NUISANCE, TTH_MAX, FILM_SLD)
    ]


def q_grid(tth_max: float) -> np.ndarray:
    """Uniform in q from 2θ 0.2° to tth_max, as many points as a 0.02° step."""
    n = round((tth_max - 0.2) / 0.02) + 1
    return np.linspace(fim.tth_to_q(np.array(0.2)), fim.tth_to_q(np.array(tth_max)), n)


def setup(cond: dict) -> tuple[dict[str, float], tuple[str, ...], dict[str, tuple[float, float]]]:
    """Truth, free parameters and bounds for a condition."""
    p0 = {"thickness": 240.0, "roughness": 6.0, "sld": cond["film_sld"]}
    p0 |= fim.TRUE_INSTR | fim.TRUE_OXIDE | {"kalpha2_ratio": 0.5, "footprint": 500.0}
    free = FILM1
    if cond["stack"] == "bilayer":
        p0 |= {
            "thickness2": SECOND["thickness2"],
            "roughness2": SECOND["roughness2"],
            "sld2": SECOND["sld_factor"] * cond["film_sld"],
        }
        free += FILM2
    free += NUIS
    bounds = dict(BOUNDS)
    bounds |= {k2: BOUNDS[k1] for k1, k2 in zip(FILM1, FILM2, strict=True)}
    if cond["nuisance"] == "wide":
        bounds |= WIDE
    return p0, free, bounds


# ------------------------------------------------------------------ forward model (torch, float64)
def reflectivity(q: Tensor, p: dict[str, Tensor], absorption: float = 1e-9) -> Tensor:
    """R(q) of ambient / top oxide / film / [film 2] / substrate through abeles_core in complex128,
    prepared as fim.reflectivity."""
    zero = torch.zeros_like(p["sld"])
    sld = [zero, p["oxide_sld"], p["sld"]]
    thk = [zero, p["oxide_thickness"], p["thickness"]]
    rgh = [p["oxide_roughness"], p["roughness"]]
    if "thickness2" in p:
        sld.append(p["sld2"])
        thk.append(p["thickness2"])
        rgh.append(p["roughness2"])
    sld.append(zero + SUB_SLD)
    rgh.append(zero + SUB_ROUGH)
    sld_c = torch.stack(sld).to(torch.complex128) * 1e-6 + 1j * absorption
    q_c = q.to(torch.complex128).reshape(1, -1, 1)
    return abeles_core(
        q_c,
        torch.stack(thk).reshape(1, 1, -1),
        torch.stack(rgh).reshape(1, 1, -1),
        sld_c.reshape(1, 1, -1),
    )[0]


def expected_counts(q: Tensor, p: dict[str, Tensor], absorption: float = 1e-9) -> Tensor:
    """fim.expected_counts with every effect on, for this module's stacks, with Gauss-Hermite
    quadrature of the resolution."""
    q_eff = apply_tth_offset(q, p["tth_offset"].reshape(1, 1), WAVELENGTH)[0]
    qq = (q_eff[:, None] + p["dq"] * GH_U[None, :]).reshape(-1)
    r = reflectivity(qq, p, absorption)
    r2 = reflectivity(qq * (WAVELENGTH / WAVELENGTH_KA2), p, absorption)
    r = (r + p["kalpha2_ratio"] * r2) / (1 + p["kalpha2_ratio"])
    r = r * torch.erf(p["footprint"] * (qq * WAVELENGTH / (4 * math.pi)).clamp_min(0))
    r = (r.reshape(-1, len(GH_U)) * GH_W).sum(1)
    return S_COUNTS * (p["i0"] * r + 10.0 ** p["bkg"])


def model(q: np.ndarray, p_base: dict[str, float], free: tuple[str, ...]):
    """(counts, d ln counts / d theta) at theta as a function of theta (numpy in, numpy out)."""
    qt = torch.as_tensor(q)
    fixed = {k: torch.tensor(v) for k, v in p_base.items()}

    def log_counts(theta: Tensor) -> Tensor:
        return torch.log(expected_counts(qt, fixed | dict(zip(free, theta, strict=True))))

    jac = jacfwd(log_counts, has_aux=False)

    def f(x: np.ndarray):
        theta = torch.as_tensor(x)
        with torch.no_grad():
            lam = torch.exp(log_counts(theta)).numpy()
        return lam, jac(theta).numpy()

    return f


def fit(
    q: np.ndarray,
    y: np.ndarray,
    p_start: dict[str, float],
    free: tuple[str, ...],
    bounds: dict[str, tuple[float, float]],
    max_nfev: int = 200,
) -> tuple[dict[str, float], float, int]:
    """Poisson deviance minimized over `free` within `bounds` (as nonunique.fit)."""
    lo = np.array([bounds[k][0] for k in free])
    hi = np.array([bounds[k][1] for k in free])
    x0 = np.clip([p_start[k] for k in free], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    f = model(q, p_start, free)
    cache: dict = {}

    def m(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = f(x)
        return cache[key]

    def resid(x):
        lam, _ = m(x)
        return np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))

    def jac(x):
        lam, jlog = m(x)
        r = np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))
        near = np.abs(lam - y) <= 1e-6 * np.maximum(y, 1e-12)
        g = np.where(near, 1 / np.sqrt(lam), (1 - y / lam) / np.where(near, 1.0, r))
        return (g * lam)[:, None] * jlog

    res = least_squares(
        resid,
        x0,
        jac=jac,
        bounds=(lo, hi),
        x_scale="jac",
        xtol=1e-10,
        ftol=1e-10,
        max_nfev=max_nfev,
    )
    return p_start | dict(zip(free, res.x.tolist(), strict=True)), float(2 * res.cost), res.nfev


# ------------------------------------------------------------------ check
def check_vs_parratt() -> float:
    """Max relative difference of R between this module's bilayer stack (no absorption) and
    benchmark.parratt over 2θ 0.2–6°."""
    q = q_grid(6.0)
    p0, _, _ = setup({"tth_max": 6.0, "film_sld": 24.0, "nuisance": "refine", "stack": "bilayer"})
    p = {k: torch.tensor(v) for k, v in p0.items()}
    r_torch = reflectivity(torch.as_tensor(q), p, absorption=0.0).numpy()
    layers = [
        (0j, 0.0, 0.0),
        (complex(p0["oxide_sld"]), p0["oxide_thickness"], p0["oxide_roughness"]),
        (complex(p0["sld"]), p0["thickness"], p0["roughness"]),
        (complex(p0["sld2"]), p0["thickness2"], p0["roughness2"]),
        (complex(SUB_SLD), 0.0, SUB_ROUGH),
    ]
    return float(np.max(np.abs(r_torch / parratt(q, layers) - 1)))


# ------------------------------------------------------------------ one condition
def lhs_starts(p0, free, bounds, n: int = N_STARTS, seed: int = 0) -> list[dict[str, float]]:
    lo = np.array([bounds[k][0] for k in free])
    hi = np.array([bounds[k][1] for k in free])
    u = qmc.LatinHypercube(d=len(free), seed=seed).random(n)
    return [p0 | dict(zip(free, (lo + x * (hi - lo)).tolist(), strict=True)) for x in u]


def run_condition(index: int, n_starts: int = N_STARTS) -> dict:
    cond = conditions()[index]
    p0, free, bounds = setup(cond)
    q = q_grid(cond["tth_max"])
    y = expected_counts(torch.as_tensor(q), {k: torch.tensor(v) for k, v in p0.items()}).numpy()
    ends = []
    t = time.perf_counter()
    for i, start in enumerate(lhs_starts(p0, free, bounds, n_starts)):
        p, dev, nfev = fit(q, y, start, free, bounds)
        ends.append({"start": i, "deviance": dev, "nfev": int(nfev), "params": p})
    return {
        "index": index,
        "condition": cond,
        "truth": p0,
        "free": list(free),
        "n_q": len(q),
        "seconds_fits": time.perf_counter() - t,
        "endpoints": ends,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["check", "run", "list"])
    ap.add_argument("--conditions", nargs="+", type=int, default=[])
    ap.add_argument("--starts", type=int, default=N_STARTS)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.part == "check":
        print(f"bilayer vs benchmark.parratt, max |ΔR/R|: {check_vs_parratt():.2e}")
        return
    if args.part == "list":
        for i, c in enumerate(conditions()):
            print(i, c)
        return
    SCAN_DIR.mkdir(exist_ok=True)
    for i in args.conditions:
        t = time.perf_counter()
        r = run_condition(i, args.starts)
        (SCAN_DIR / f"fits_{i:02d}.json").write_text(
            json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        best = sorted(e["deviance"] for e in r["endpoints"])[:5]
        print(f"condition {i}: {time.perf_counter() - t:.0f} s, lowest D {best}")


if __name__ == "__main__":
    main()
