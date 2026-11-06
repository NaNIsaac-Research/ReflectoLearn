"""Shared pieces of the pre-registered follow-up runs G1 to G3 (docs/paper/prereg/).

The search and the refinement run on the experiment engine (`scan.expected_counts`, a fixed 15-node
resolution rule); every refined minimum is then polished on the converged ReflectoSim forward, and
the converged scores are the ones the registered verdicts use (PREREG-G2.md, "Forward models").
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.special import xlogy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "001_identifiability"))
import basin
import g4_sim_forward as g4
import scan

SEED = 20261107
PREREG = ROOT / "docs" / "paper" / "prereg"
QUEUE = ROOT.parent.parent / "DotFiles" / "herdr" / "tools" / "test-queue.ps1"
POLISH_NFEV = 30


def deviance_terms(lam: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Poisson deviance per point for any counts y (zeros allowed)."""
    return np.maximum(2.0 * (lam - y + xlogy(y, y / lam)), 0.0)


def queue_free() -> bool:
    """True when no test-queue job holds the machine."""
    out = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(QUEUE), "-Status"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return out == "free"


def wait_for_queue(poll: float = 60.0) -> None:
    while not queue_free():
        print("test-queue is held, waiting", flush=True)
        time.sleep(poll)


def condition(cond: int) -> tuple[dict, np.ndarray, dict, tuple[str, ...], dict]:
    """(file data, q grid, truth parameters, free names, bounds) of an R2a condition."""
    data = json.loads((g4.SCAN_DIR / f"fits_{cond:02d}.json").read_text(encoding="utf-8"))
    _, _, bounds = scan.setup(data["condition"])
    q = scan.q_grid(data["condition"]["tth_max"])
    return data, q, data["truth"], tuple(data["free"]), bounds


def basin_minimum(rep: str, summary: dict) -> tuple[int, dict]:
    """(condition, full parameter dict) of a basin's representative minimum."""
    cond = int(rep.split(":")[1])
    data = json.loads((g4.SCAN_DIR / f"fits_{cond:02d}.json").read_text(encoding="utf-8"))
    rec = summary[rep]
    return cond, data["truth"] | dict(zip(rec["names"], rec["x_min"], strict=True))


def load_summary() -> dict:
    out = {}
    for line in (
        basin.OUT.joinpath("basin_full_summary.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        r = json.loads(line)
        if r["objective"] == "nf":
            out[r["id"]] = r
    return out


def make_problem(cid: str, y: np.ndarray, x_start: np.ndarray) -> basin.Problem:
    """basin.Problem of recorded candidate `cid`'s condition, with data y and start x_start. Its
    model, bounds and noise-free truth counts (`lam_t`, the cross-score reference) are the condition's."""
    prob = basin.r2a_problems([cid])[0]
    prob.y = y
    prob.x_end = np.asarray(x_start, float)
    prob.theta_t = None
    prob.fisher = None
    prob.region_idx = None
    lam = prob.full_counts(prob.x_end)
    prob.recorded = float(prob.terms(lam, y).sum())
    return prob


def polish(
    prob: basin.Problem, x0: np.ndarray, q: np.ndarray, p_fixed: dict, y: np.ndarray
) -> dict:
    """Trust-region least squares on the converged ReflectoSim forward from x0: the residuals are the
    converged forward's, the Jacobian is the engine's, scaled to the converged counts."""
    ev = basin.Evaluator(prob)
    names, lo, hi = prob.names, prob.lo, prob.hi
    n_eval = 0

    def sim(x: np.ndarray) -> np.ndarray:
        nonlocal n_eval
        n_eval += 1
        return g4.r2a_sim(q, p_fixed | dict(zip(names, x, strict=True)))

    def resid(x):
        lam = sim(x)
        return np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))

    def jac(x):
        lam = sim(x)
        _, jlog = ev.model(x)
        r = np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))
        near = np.abs(lam - y) <= 1e-6 * np.maximum(y, 1e-12)
        g = np.where(near, 1 / np.sqrt(lam), (1 - y / lam) / np.where(near, 1.0, r))
        return (g * lam)[:, None] * jlog

    start = np.clip(x0, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    d0 = float(deviance_terms(sim(start), y).sum())
    res = least_squares(
        resid,
        start,
        jac=jac,
        bounds=(lo, hi),
        x_scale="jac",
        xtol=1e-10,
        ftol=1e-10,
        max_nfev=POLISH_NFEV,
    )
    return {
        "x": res.x.tolist(),
        "dev_converged": float(2 * res.cost),
        "dev_converged_start": d0,
        "nfev": int(res.nfev),
        "sim_calls": n_eval,
    }


def converged_counts(q: np.ndarray, p_fixed: dict, names, x) -> np.ndarray:
    return g4.r2a_sim(q, p_fixed | dict(zip(names, x, strict=True)))


def init_worker() -> None:
    torch.set_num_threads(1)
