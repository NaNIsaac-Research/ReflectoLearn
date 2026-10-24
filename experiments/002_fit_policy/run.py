"""Experiment 002, X2b: the registered comparison of docs/fit_policy/DESIGN.md section 6.

Episodes are paired units (class n, truth j, noise level s, start k): 3 × 16 × 2 × 4 = 384 per
method. Each finished episode is one JSON line in docs/fit_policy/runs/<method>.jsonl, so a run can
be stopped and resumed. Methods: A1–A4 (protocol), B (random), C (all at once, BIC over L), Co
(C_oracle), Cr (C_refine, one-layer class only), C2E (C with twice the budget, the budget guard).

Run:  uv run python experiments/002_fit_policy/run.py noise
      uv run python experiments/002_fit_policy/run.py episodes --methods A1 C --workers 8
      uv run python experiments/002_fit_policy/run.py timing
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from reflectolearn.fit_policy import (
    VARIANTS,
    FitEnv,
    full_fit,
    oracle_fit,
    random_policy,
    refine_fit,
    run_protocol,
)
from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.env import BUDGET

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "fit_policy" / "runs"
CLASSES = (1, 2, 3)
N_TRUTHS = 16
N_NOISE_REALISATIONS = 20
METHODS = (*VARIANTS, "B", "C", "Co", "Cr", "C2E")
TIMING_CLASS = 9  # timing pilot (6.6): truths (n, 1000 + n), starts and random seeds of class 9


def units(n_truths: int = N_TRUTHS) -> list[tuple[int, int, int, int]]:
    return list(
        itertools.product(CLASSES, range(n_truths), range(len(st.NOISE_SCALES)), range(st.N_STARTS))
    )


def data(n: int, j: int, s: int):
    """Training and held-out (q, y) of truth (n, j) at noise level s."""
    q = st.q_grid()
    held = st.held_out_mask(q)
    y = st.noisy_counts(st.counts(st.sample_truth(n, j), q, st.NOISE_SCALES[s]), n, j, s)
    return q[~held], y[~held], q[held], y[held]


def frozen_commit() -> str:
    """The commit the registered episodes run at (DESIGN.md section 6, freeze). Refuses to run with
    uncommitted changes outside the run records."""
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)docs/fit_policy/runs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted changes, not a frozen commit:\n{dirty}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def episode(method: str, n: int, j: int, s: int, k: int, seed_class: int | None = None) -> dict:
    """One episode. seed_class replaces n in the start and random seeds (timing pilot only)."""
    torch.set_num_threads(1)
    q, y, q_ho, y_ho = data(n, j, s)
    sc = n if seed_class is None else seed_class
    start = st.starts(sc, j)[k]
    extra = st.extra_layers(sc, j)[k]
    scale = st.NOISE_SCALES[s]
    t0 = time.perf_counter()
    if method in VARIANTS:
        env = FitEnv(q, y, scale, start, q_ho=q_ho, y_ho=y_ho)
        res = run_protocol(env, VARIANTS[method]).result()
    elif method == "B":
        env = FitEnv(q, y, scale, start, q_ho=q_ho, y_ho=y_ho)
        res = random_policy(env, np.random.default_rng((st.SEED, sc, j, s, k, 7))).result()
    elif method in ("C", "C2E"):
        budget = BUDGET * (2 if method == "C2E" else 1)
        res = full_fit(start, extra, q, y, scale, q_ho=q_ho, y_ho=y_ho, budget=budget)
    elif method == "Co":
        res = oracle_fit(start, extra, n, q, y, scale, q_ho=q_ho, y_ho=y_ho)
    elif method == "Cr":
        res = refine_fit(start, q, y, scale, q_ho=q_ho, y_ho=y_ho)
    else:
        raise ValueError(method)
    return {
        "method": method,
        "n": n,
        "j": j,
        "s": s,
        "k": k,
        "seconds": time.perf_counter() - t0,
    } | res


def _episode(args) -> dict:
    return episode(*args)


def _done(path: Path) -> set[tuple[int, int, int, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["n"], r["j"], r["s"], r["k"]) for r in map(json.loads, f)}


def run_episodes(methods: list[str], workers: int, n_truths: int, classes: list[int]) -> None:
    commit = frozen_commit()
    print(f"commit {commit}")
    RUNS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for m in methods:
        done = _done(RUNS / f"{m}.jsonl")
        for u in units(n_truths):
            if u[0] not in classes or u in done or (m == "Cr" and u[0] != 1):
                continue
            jobs.append((m, *u))
    print(f"{len(jobs)} episodes to run")
    with Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_episode, jobs), 1):
            with (RUNS / f"{rec['method']}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec | {"commit": commit}) + "\n")
            if i % 20 == 0 or i == len(jobs):
                print(
                    f"{i}/{len(jobs)} {rec['method']} n{rec['n']} j{rec['j']} {rec['seconds']:.0f}s",
                    flush=True,
                )


def noise_scale(n_truths: int = N_TRUTHS) -> None:
    """s_ho per (n, j, s): SD of the truth model's held-out deviance over 20 extra realisations."""
    out = {}
    q = st.q_grid()
    held = st.held_out_mask(q)
    for n, j, s in itertools.product(CLASSES, range(n_truths), range(len(st.NOISE_SCALES))):
        lam = st.counts(st.sample_truth(n, j), q, st.NOISE_SCALES[s])[held]
        devs = [
            st.deviance(lam, st.noisy_counts(lam, n, j, s, r))
            for r in range(1, N_NOISE_REALISATIONS + 1)
        ]
        out[f"{n},{j},{s}"] = {"s_ho": float(np.std(devs, ddof=1)), "mean": float(np.mean(devs))}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "noise.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} noise scales written")


def timing() -> None:
    """Section 6.6: one episode per method on the unregistered seed; time only, no metrics."""
    for n in CLASSES:
        for m in (*VARIANTS, "B", "C", "Co", *(["Cr"] if n == 1 else [])):
            rec = episode(m, n, 1000 + n, 1, 0, seed_class=TIMING_CLASS)
            print(
                f"n{n} {m:3s} {rec['seconds']:6.1f} s  evaluations {rec.get('evaluations', '-')}",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("episodes")
    e.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--truths", type=int, default=N_TRUTHS)
    e.add_argument("--classes", type=int, nargs="+", default=list(CLASSES))
    nz = sub.add_parser("noise")
    nz.add_argument("--truths", type=int, default=N_TRUTHS)
    sub.add_parser("timing")
    args = ap.parse_args()
    match args.cmd:
        case "episodes":
            run_episodes(args.methods, args.workers, args.truths, args.classes)
        case "noise":
            noise_scale(args.truths)
        case "timing":
            timing()


if __name__ == "__main__":
    main()
