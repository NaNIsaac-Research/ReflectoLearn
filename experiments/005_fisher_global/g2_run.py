"""G2 of docs/paper/prereg/PREREG-G2.md: do wrong basins stay rejected under noise?

Per candidate basin, draws of the condition's truth curve (Poisson; for the g2_phi candidates also
negative-binomial with variance phi times the mean). Each draw is fitted from the truth and from the basin's
recorded minimum on the experiment engine, and both minima are polished on the converged forward.
G = D_basin - D_truth on the same data; H-N1 to H-N3 are scored on the converged and on the engine deviances.

One result file per draw, written once; each of its two fits is checkpointed on its own, so a preemption
loses at most the fit in progress. Seeds: SeedSequence([SEED, stage, candidate index, draw]) with stage 1 for
Poisson and stage 3 for negative-binomial draws (stage 2 is G2b).

    <venv python> -I experiments/005_fisher_global/g2_run.py run --worker 0 --workers 4
    <venv python> -I experiments/005_fisher_global/g2_run.py report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import basin
import g_common as gc

OUT = gc.ROOT / "docs" / "fisher_global" / "g2"
SUBSET = gc.PREREG / "g2_subset.json"
PHI = 1.34
SEP = 25.0
S1_NEAR = 1e3
S1_LIMIT = 1e4
N_DRAWS, N_DRAWS_NEAR = 3, 20


def code_hash() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=gc.ROOT,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "experiments"],
        capture_output=True,
        text=True,
        cwd=gc.ROOT,
        check=False,
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


CODE = code_hash()


def draws() -> Iterator[tuple[str, int, str, int]]:
    """(kind, candidate index, rep, draw) for every registered draw."""
    sub = json.loads(SUBSET.read_text(encoding="utf-8"))
    for idx, rep in enumerate(sub["g2"]):
        n = N_DRAWS_NEAR if sub["S1"][rep] >= S1_NEAR else N_DRAWS
        for d in range(n):
            yield "poisson", idx, rep, d
    for rep in sub["g2_phi"]:
        idx = sub["g2"].index(rep)
        for d in range(N_DRAWS):
            yield "negbin", idx, rep, d


def draw_counts(kind: str, idx: int, d: int, lam: np.ndarray) -> np.ndarray:
    if kind == "poisson":
        rng = np.random.default_rng(np.random.SeedSequence([gc.SEED, 1, idx, d]))
        return rng.poisson(lam).astype(float)
    # mean lam, variance PHI * lam: p = 1 / PHI, n = lam / (PHI - 1)
    rng = np.random.default_rng(np.random.SeedSequence([gc.SEED, 3, idx, d]))
    return rng.negative_binomial(lam / (PHI - 1), 1 / PHI).astype(float)


def fit_one(rep: str, y: np.ndarray, x0: np.ndarray, q: np.ndarray, p0: dict) -> dict:
    prob = gc.make_problem(rep, y, x0)
    row = basin.refine(prob)
    pol = gc.polish(prob, np.array(row["x_min"]), q, p0, y)
    return {
        "outcome": row["outcome"],
        "dev_engine": row["d_final"],
        "nfev_engine": row["nfev_total"],
        "seconds_engine": row["seconds"],
        "x_engine": row["x_min"],
        **pol,
        "names": prob.names,
    }


def one_draw(path: Path, kind: str, idx: int, rep: str, d: int, summary: dict) -> dict:
    t0 = time.time()
    cond, p_basin = gc.basin_minimum(rep, summary)
    _, q, p0, free, _ = gc.condition(cond)
    x_t = np.array([p0[k] for k in free])
    x_b = np.array([p_basin[k] for k in free])
    lam_truth = gc.converged_counts(q, p0, free, x_t)
    lam_rec = gc.converged_counts(q, p0, free, x_b)
    y = draw_counts(kind, idx, d, lam_truth)
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
    lam_b = gc.converged_counts(q, p0, free, fits["basin"]["x"])
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
        "G": fits["basin"]["dev_converged"] - fits["truth"]["dev_converged"],
        "G_engine": fits["basin"]["dev_engine"] - fits["truth"]["dev_engine"],
        "cross_to_recorded": float(gc.deviance_terms(lam_b, lam_rec).sum()),
        "dD0": float(gc.deviance_terms(lam_rec, lam_truth).sum()),
        "seconds": time.time() - t0,
    }


def run(worker: int, workers: int, hours: float) -> None:
    gc.init_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = gc.load_summary()
    t_end = time.time() + hours * 3600
    for n, (kind, idx, rep, d) in enumerate(draws()):
        if n % workers != worker:
            continue
        path = OUT / f"{kind}_{idx:03d}_{d:02d}.json"
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
            print(f"{path.stem}: G {row['G']:.1f} ({row['seconds']:.0f} s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failed draw is recorded and skipped
            path.with_suffix(".error.json").write_text(
                json.dumps({"job": path.stem, "error": repr(exc), "code": CODE}), encoding="utf-8"
            )
            print(f"{path.stem}: error {exc!r}", flush=True)
        finally:
            lock.unlink(missing_ok=True)


def verdict(frac: float, hold: float, fail: float) -> str:
    return "holds" if frac >= hold else ("fails" if frac < fail else "unresolved")


def report() -> None:
    sub = json.loads(SUBSET.read_text(encoding="utf-8"))
    s1 = sub["S1"]
    rows = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(OUT.glob("*_[0-9][0-9][0-9]_[0-9][0-9].json"))
    ]
    errors = sorted(p.stem for p in OUT.glob("*.error.json"))
    n_reg = sum(1 for _ in draws())
    pois = [r for r in rows if r["kind"] == "poisson"]
    nb = [r for r in rows if r["kind"] == "negbin"]
    codes = sorted({r["code"] for r in rows} | {c for r in rows for c in r["fit_codes"].values()})
    lines = [
        "# G2: wrong basins under noise",
        "",
        "Generated by `experiments/005_fisher_global/g2_run.py report`. Registered in "
        + "`docs/paper/prereg/PREREG-G2.md`. Converged-forward deviances are the headline; where the engine "
        + "deviances give a different verdict, the hypothesis is unresolved.",
        "",
        f"Draws done: {len(rows)} of {n_reg} ({len(pois)} Poisson, {len(nb)} negative-binomial with phi {PHI}); "
        + f"errors: {len(errors)}{' (' + ', '.join(errors) + ')' if errors else ''}. Code hashes: {', '.join(codes)}. "
        + f"CPU time {sum(r['seconds'] for r in rows) / 3600:.1f} h.",
        "",
        "| hypothesis | draws | converged | engine | verdict |",
        "|---|---|---|---|---|",
    ]

    def frac(sel: list[dict], key: str) -> float:
        return float(np.mean([r[key] >= SEP for r in sel])) if sel else float("nan")

    n1 = [r["cross_to_recorded"] <= SEP for r in pois]
    if pois:
        f = float(np.mean(n1))
        lines.append(
            f"| H-N1: noisy basin minimum within cross-deviance {SEP:g} of the recorded one | {len(pois)} | "
            + f"{f:.3f} | n/a | {verdict(f, 0.8, 0.6)} |"
        )
    for name, sel, testable in (
        ("H-N2: G >= 25, Poisson", pois, True),
        (
            "H-N3: G >= 25, negative-binomial",
            nb,
            len({r["rep"] for r in nb if s1[r["rep"]] <= S1_LIMIT}) >= 20,
        ),
    ):
        sel = [r for r in sel if s1[r["rep"]] <= S1_LIMIT]
        if not sel:
            continue
        fc, fe = frac(sel, "G"), frac(sel, "G_engine")
        vc, ve = verdict(fc, 0.95, 0.8), verdict(fe, 0.95, 0.8)
        v = (
            (vc if vc == ve else "unresolved (converged and engine disagree)")
            if testable
            else "not testable"
        )
        lines.append(f"| {name}, S1* <= {S1_LIMIT:g} | {len(sel)} | {fc:.3f} | {fe:.3f} | {v} |")
    lines += [
        "",
        "Rules: H-N1 holds at 0.80 or more and fails below 0.60; H-N2 and H-N3 hold at 0.95 or more and fail "
        + "below 0.80; H-N3 is not testable with fewer than 20 candidates.",
        "",
    ]
    if pois:
        bins = [(0, 25), (25, 100), (100, 1e3), (1e3, 1e4), (1e4, np.inf)]
        lines += [
            "Descriptive: G against the noise-free deviance of the recorded basin (dD0), Poisson draws.",
            "",
            "| dD0 | draws | median G | G < 25 | G < 4 |",
            "|---|---|---|---|---|",
        ]
        for lo, hi in bins:
            sel = [r for r in pois if lo <= r["dD0"] < hi]
            if sel:
                g = np.array([r["G"] for r in sel])
                lines.append(
                    f"| {lo:g} to {hi:g} | {len(sel)} | {np.median(g):.3g} | {int((g < SEP).sum())} | {int((g < 4).sum())} |"
                )
        near = [r for r in pois if s1[r["rep"]] >= S1_NEAR]
        lines += [
            "",
            "Near basins (S1* >= 1e3), fraction of draws with G < 4 (not rejected):",
            "",
            *(
                f"- {rep} (S1* {s1[rep]:.3g}): "
                + f"{np.mean([r['G'] < 4 for r in near if r['rep'] == rep]):.2f} of "
                + f"{sum(r['rep'] == rep for r in near)} draws"
                for rep in sorted({r["rep"] for r in near}, key=lambda k: -s1[k])
            ),
            "",
        ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report", "status"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=16.0)
    args = ap.parse_args()
    if args.mode == "run":
        run(args.worker, args.workers, args.hours)
    elif args.mode == "status":
        left = [
            1
            for kind, idx, _, d in draws()
            if not (OUT / f"{kind}_{idx:03d}_{d:02d}.json").exists()
            and not (OUT / f"{kind}_{idx:03d}_{d:02d}.error.json").exists()
        ]
        print(len(left))
    else:
        report()


if __name__ == "__main__":
    main()
