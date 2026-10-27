"""Experiment 006: what log-type objectives cost an XRR fit against the Poisson likelihood.

Registration: docs/objective/PREREG.md. For one grid cell (film SLD, 2θ_max, counts at R = 1) this
draws Poisson curves from experiment 001's forward model (`fim.expected_counts`) and fits each one
with four objectives, every fit started at the truth so that only the objective differs:

    P   Poisson deviance (the control; efficient)
    L0  sum (ln y - ln lam)^2 over y > 0          (refine's log-MSE)
    L1  sum (ln(y + 1/2) - ln(lam + 1/2))^2       (L0 without dropping the zero counts)
    L2  sum y (ln y - ln lam)^2 over y > 0        (L0 weighted by the counts)

Each record keeps every endpoint, so efficiency, bias and coverage are computed by `report`, not
here. For L0 it also keeps the Laplace interval of a Gaussian ln R likelihood with a fitted spread,
sigma^2 (J^T J)^-1 with sigma^2 the mean squared residual: the form refine.mcmc samples.

    uv run python experiments/006_objective/objective.py pilot
    uv run python experiments/006_objective/objective.py run --cells 0 1 2 --threads 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "001_identifiability"))
import fim

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "objective"
SEED_ROOT = 20261028
THICKNESS, ROUGHNESS = 240.0, 6.0
FILM_SLD = (12.0, 24.0, 45.0)
TTH_MAX = (3.0, 6.0, 12.0)
COUNTS = (1e4, 1e5, 1e6, 1e7)
OBJECTIVES = ("P", "L0", "L1", "L2")
N_DRAWS = 50


def cells() -> list[dict]:
    grid = [
        {"sld": r, "tth_max": t, "counts": s, "scenario": "film"}
        for r, t, s in itertools.product(FILM_SLD, TTH_MAX, COUNTS)
    ]
    # one instrument column: the nuisances refine fits besides the film, at 10^6 counts
    grid += [{"sld": r, "tth_max": 6.0, "counts": 1e6, "scenario": "instr"} for r in FILM_SLD]
    return grid


def setup(cell: dict):
    film = fim.Film(f"sld{cell['sld']:g}", THICKNESS, ROUGHNESS, cell["sld"])
    p0 = fim.truth(film, cell["scenario"])
    free = fim.SCENARIOS[cell["scenario"]]
    q = fim.tth_to_q(fim.tth_grid(cell["tth_max"]))
    return q, p0, free


def lam_jac(q, p0, free, x, s):
    return fim.jacobian(q, p0 | dict(zip(free, x, strict=True)), free, s)


def fit(objective: str, y: np.ndarray, q, p0, free, s, x0) -> tuple[np.ndarray, dict]:
    pos = y > 0
    if objective == "P":

        def res(x):
            lam = lam_jac(q, p0, free, x, s)[0]
            t = np.where(pos, y * np.log(np.where(pos, y, 1.0) / lam), 0.0) - (y - lam)
            return np.sign(lam - y) * np.sqrt(2 * np.clip(t, 0, None))

        def jac(x):
            lam, J = lam_jac(q, p0, free, x, s)
            r = res(x)
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.where(np.abs(r) > 1e-12, (lam - y) / r, np.sqrt(lam))
            return g[:, None] * J

    elif objective == "L0":
        ly = np.log(y[pos])

        def res(x):
            return np.log(lam_jac(q, p0, free, x, s)[0][pos]) - ly

        def jac(x):
            return lam_jac(q, p0, free, x, s)[1][pos]

    elif objective == "L1":
        ly = np.log(y + 0.5)

        def res(x):
            return np.log(lam_jac(q, p0, free, x, s)[0] + 0.5) - ly

        def jac(x):
            lam, J = lam_jac(q, p0, free, x, s)
            return (lam / (lam + 0.5))[:, None] * J

    elif objective == "L2":
        w, ly = np.sqrt(y[pos]), np.log(y[pos])

        def res(x):
            return w * (np.log(lam_jac(q, p0, free, x, s)[0][pos]) - ly)

        def jac(x):
            return w[:, None] * lam_jac(q, p0, free, x, s)[1][pos]

    else:
        raise ValueError(objective)
    r = least_squares(res, x0, jac=jac, method="lm", xtol=1e-12, ftol=1e-12)
    info = {"status": int(r.status), "nfev": int(r.nfev)}
    if objective == "L0":
        info["laplace_sd"] = np.sqrt(
            np.diag(np.mean(r.fun**2) * np.linalg.inv(r.jac.T @ r.jac))
        ).tolist()
    return r.x, info


def run_cell(index: int, n_draws: int = N_DRAWS, objectives=OBJECTIVES) -> dict:
    cell = cells()[index]
    q, p0, free = setup(cell)
    s = cell["counts"]
    x0 = np.array([p0[k] for k in free])
    lam, J = lam_jac(q, p0, free, x0, s)
    crlb = np.sqrt(np.diag(np.linalg.inv(J.T @ (lam[:, None] * J))))
    rng = np.random.default_rng([SEED_ROOT, index])
    record = {
        "index": index, "cell": cell, "free": list(free), "truth": x0.tolist(), "crlb": crlb.tolist(),
        "n_q": len(q), "zero_share_expected": float(np.mean(np.exp(-lam))),
        "fits": {o: [] for o in objectives}, "seconds": {o: 0.0 for o in objectives},
    }  # fmt: skip
    for _ in range(n_draws):
        y = rng.poisson(lam).astype(float)
        for o in objectives:
            t = time.perf_counter()
            x, info = fit(o, y, q, p0, free, s, x0)
            record["seconds"][o] += time.perf_counter() - t
            record["fits"][o].append({"x": x.tolist(), **info})
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["list", "pilot", "run"])
    ap.add_argument("--cells", nargs="+", type=int, default=[])
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_default_dtype(fim.DEFAULT_DTYPE)
    torch.set_num_threads(args.threads)
    if args.part == "list":
        for i, c in enumerate(cells()):
            print(i, c)
        return
    if args.part == "pilot":
        # SLD 45, 2θ ≤ 6°, 10^5 counts, 15 draws: times each objective before the main run
        index = next(
            i
            for i, c in enumerate(cells())
            if c == {"sld": 45.0, "tth_max": 6.0, "counts": 1e5, "scenario": "film"}
        )
        rec = run_cell(index, n_draws=15)
        (OUT / "pilot.json").write_text(json.dumps(rec))
        for o in OBJECTIVES:
            x = np.array([f["x"] for f in rec["fits"][o]])
            t, c = np.array(rec["truth"]), np.array(rec["crlb"])
            eff = (c / x.std(0, ddof=1)) ** 2
            bias = (x.mean(0) - t) / c
            print(
                f"{o:2s} {rec['seconds'][o] / 15:5.2f} s/fit | efficiency d {eff[0]:.3f} rho {eff[2]:.4f}"
                f" | bias/CRLB d {bias[0]:+.2f} rho {bias[2]:+.2f}"
            )
        return
    runs = OUT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for i in args.cells:
        t = time.perf_counter()
        rec = run_cell(i, n_draws=args.draws)
        (runs / f"cell_{i:02d}.json").write_text(json.dumps(rec))
        print(f"cell {i} {rec['cell']} {time.perf_counter() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
