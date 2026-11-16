"""G2c of docs/paper/prereg/PREREG-G2.md: parametric bootstrap on the 10 closest basins.

Per basin of `g2c_closest`, three sets of draws in this order: 30 Poisson draws of the truth curve, 15
Poisson draws of the basin's curve, 30 negative-binomial draws of the truth curve (variance phi times the
mean, phi = 1.34, the registered fallback). Each draw is fitted from the truth and from the basin's
recorded minimum on the experiment engine and polished on the converged forward, as in G2.
T = D_basin - D_truth on the same draw (T > 0: the truth is preferred). No verdict rule.

One result file per draw, written once; each fit is checkpointed on its own. Seeds:
SeedSequence([SEED, stage, basin index, draw]), stage 4 for truth Poisson, 5 for basin Poisson, 6 for
truth negative-binomial draws.

    <venv python> -I experiments/005_fisher_global/g2c_run.py run --worker 0 --workers 4
    <venv python> -I experiments/005_fisher_global/g2c_run.py status [--kind tp]
    <venv python> -I experiments/005_fisher_global/g2c_run.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g_common as gc
from g2_run import SUBSET, code_hash, fit_one

OUT = gc.ROOT / "docs" / "fisher_global" / "g2c"
PHI = 1.34
CHI2_12_95 = 21.0
SETS = (("tp", 4, 30), ("bp", 5, 15), ("tn", 6, 30))
NAMES = {
    "tp": "truth curve, Poisson",
    "bp": "basin curve, Poisson",
    "tn": f"truth curve, negative-binomial phi {PHI}",
}
CODE = code_hash()


def draws(kind: str | None = None) -> Iterator[tuple[str, int, str, int]]:
    """(set, basin index, rep, draw) for every registered draw, set by set."""
    reps = json.loads(SUBSET.read_text(encoding="utf-8"))["g2c_closest"]
    for k, _, n in SETS:
        if kind and k != kind:
            continue
        for d in range(n):
            for idx, rep in enumerate(reps):
                yield k, idx, rep, d


def draw_counts(kind: str, idx: int, d: int, lam: np.ndarray) -> np.ndarray:
    stage = {k: s for k, s, _ in SETS}[kind]
    rng = np.random.default_rng(np.random.SeedSequence([gc.SEED, stage, idx, d]))
    if kind == "tn":
        # mean lam, variance PHI * lam: p = 1 / PHI, n = lam / (PHI - 1)
        return rng.negative_binomial(lam / (PHI - 1), 1 / PHI).astype(float)
    return rng.poisson(lam).astype(float)


def one_draw(path: Path, kind: str, idx: int, rep: str, d: int, summary: dict) -> dict:
    t0 = time.time()
    cond, p_basin = gc.basin_minimum(rep, summary)
    _, q, p0, free, _ = gc.condition(cond)
    x_t = np.array([p0[k] for k in free])
    x_b = np.array([p_basin[k] for k in free])
    lam_truth = gc.converged_counts(q, p0, free, x_t)
    lam_basin = gc.converged_counts(q, p0, free, x_b)
    y = draw_counts(kind, idx, d, lam_basin if kind == "bp" else lam_truth)
    fits = {}
    for start, x0 in (("truth", x_t), ("basin", x_b)):
        part = path.with_suffix(f".{start}.json")
        if part.exists():
            fits[start] = json.loads(part.read_text(encoding="utf-8"))
            continue
        fits[start] = {"code": CODE, **fit_one(rep, y, x0, q, p0)}
        tmp = part.with_suffix(".tmp")
        tmp.write_text(json.dumps(fits[start]), encoding="utf-8")
        tmp.replace(part)
    return {
        "rep": rep,
        "condition": cond,
        "index": idx,
        "kind": kind,
        "draw": d,
        "code": CODE,
        "fit_codes": {k: v["code"] for k, v in fits.items()},
        "truth": fits["truth"],
        "basin": fits["basin"],
        "D_truth": fits["truth"]["dev_converged"],
        "D_basin": fits["basin"]["dev_converged"],
        "T": fits["basin"]["dev_converged"] - fits["truth"]["dev_converged"],
        "T_engine": fits["basin"]["dev_engine"] - fits["truth"]["dev_engine"],
        "dD0": float(gc.deviance_terms(lam_basin, lam_truth).sum()),
        "seconds": time.time() - t0,
    }


def result_path(kind: str, idx: int, d: int) -> Path:
    return OUT / f"{kind}_{idx:02d}_{d:02d}.json"


def run(worker: int, workers: int, hours: float) -> None:
    gc.init_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = gc.load_summary()
    t_end = time.time() + hours * 3600
    for n, (kind, idx, rep, d) in enumerate(draws()):
        if n % workers != worker:
            continue
        path = result_path(kind, idx, d)
        if path.exists():
            continue
        if time.time() > t_end:
            print("deadline reached", flush=True)
            return
        gc.wait_for_queue()
        lock = path.with_suffix(".lock")
        if not gc.take_lock(lock):
            continue
        try:
            row = one_draw(path, kind, idx, rep, d, summary)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(row), encoding="utf-8")
            tmp.replace(path)
            for start in ("truth", "basin"):
                path.with_suffix(f".{start}.json").unlink(missing_ok=True)
            print(f"{path.stem}: T {row['T']:.2f} ({row['seconds']:.0f} s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failed draw is recorded and skipped
            path.with_suffix(".error.json").write_text(
                json.dumps({"job": path.stem, "error": repr(exc), "code": CODE}), encoding="utf-8"
            )
            print(f"{path.stem}: error {exc!r}", flush=True)
        finally:
            lock.unlink(missing_ok=True)


def left(kind: str | None = None) -> int:
    return sum(
        1
        for k, idx, _, d in draws(kind)
        if not result_path(k, idx, d).exists()
        and not result_path(k, idx, d).with_suffix(".error.json").exists()
    )


def report() -> None:
    reps = json.loads(SUBSET.read_text(encoding="utf-8"))["g2c_closest"]
    summary = gc.load_summary()
    rows = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(OUT.glob("*_[0-9][0-9]_[0-9][0-9].json"))
    ]
    errors = sorted(p.stem for p in OUT.glob("*.error.json"))
    n_reg = sum(1 for _ in draws())
    codes = sorted({r["code"] for r in rows} | {c for r in rows for c in r["fit_codes"].values()})
    lines = [
        "# G2c: parametric bootstrap on the closest basins",
        "",
        "Generated by `experiments/005_fisher_global/g2c_run.py report`. Registered in "
        + "`docs/paper/prereg/PREREG-G2.md` (G2c, and the choices fixed before the G2c run). Descriptive: "
        + "G2c has no verdict rule. T = D_basin - D_truth on converged-forward deviances (engine in brackets); "
        + f"T > 0 prefers the truth. Reference: chi-square with 12 free parameters, 95 % bound {CHI2_12_95:g}.",
        "",
        f"Draws done: {len(rows)} of {n_reg}; errors: {len(errors)}{' (' + ', '.join(errors) + ')' if errors else ''}. "
        + f"Code hashes: {', '.join(codes) or 'none'}. CPU time {sum(r['seconds'] for r in rows) / 3600:.1f} h.",
        "",
    ]
    for kind, _, n in SETS:
        sel = [r for r in rows if r["kind"] == kind]
        lines += [
            f"## {NAMES[kind]} ({len(sel)} of {n * len(reps)} draws)",
            "",
            "| basin | S1* | dD0 | draws | T > 0 | T < 0 | T > 21 | T 5 % | T 50 % | T 95 % | fits at budget |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for idx, rep in enumerate(reps):
            b = [r for r in sel if r["index"] == idx]
            if not b:
                continue
            t = np.array([r["T"] for r in b])
            te = np.array([r["T_engine"] for r in b])
            s1 = summary[rep]["S1_min"]
            bud = sum(r[f]["nfev_engine"] >= 3000 for r in b for f in ("truth", "basin"))
            lines.append(
                f"| {rep} | {s1:.3g} | {b[0]['dD0']:.3g} | {len(b)} | {np.mean(t > 0):.2f} ({np.mean(te > 0):.2f}) | "
                + f"{np.mean(t < 0):.2f} ({np.mean(te < 0):.2f}) | {np.mean(t > CHI2_12_95):.2f} | "
                + f"{np.percentile(t, 5):.3g} | {np.median(t):.3g} | {np.percentile(t, 95):.3g} | {bud} |"
            )
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g2c_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report", "status"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=16.0)
    ap.add_argument("--kind", choices=[k for k, _, _ in SETS])
    args = ap.parse_args()
    if args.mode == "run":
        run(args.worker, args.workers, args.hours)
    elif args.mode == "status":
        print(left(args.kind))
    else:
        report()


if __name__ == "__main__":
    main()
