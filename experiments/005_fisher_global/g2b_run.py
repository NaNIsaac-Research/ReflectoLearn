"""G2b of docs/paper/prereg/PREREG-G2.md: refits against a curve the fitting model cannot produce.

The data curve is the truth stack with its film / substrate interface replaced by a graded transition
(five equal slabs spanning 10 Å, SLD stepping linearly), computed on the converged ReflectoSim forward;
a noise-free dataset and one Poisson draw per candidate. Per dataset the model (error-function
roughness) is fitted from the truth and from the candidate basin's recorded minimum on the experiment
engine, and both minima are polished on the converged forward. M = D_basin - D_truth (converged).

    <venv python> -I experiments/005_fisher_global/g2b_run.py run --worker 0 --workers 3
    <venv python> -I experiments/005_fisher_global/g2b_run.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import basin
import g4_sim_forward as g4
import g_common as gc
import scan
from reflectosim import Instrument, Layer, Stack, forward

OUT = gc.ROOT / "docs" / "fisher_global" / "g2b"
SUBSET = gc.PREREG / "g2_subset.json"
RESULT_REPORT = OUT / "g2b_report.md"
WIDTH, SLABS = 10.0, 5
MARGIN_MIN = 25.0
S1_LIMIT = 1e4


def graded_counts(q: np.ndarray, p: dict) -> np.ndarray:
    """The truth curve with the last film's lower interface graded over WIDTH Å, converged forward."""
    absorb = 1j * g4.ABSORPTION
    layers = [
        Layer(p["oxide_sld"] + absorb, p["oxide_thickness"], p["oxide_roughness"], "oxide"),
        Layer(p["sld"] + absorb, p["thickness"], p["roughness"], "film"),
    ]
    sld_film, d_film = p["sld"], p["thickness"]
    if "thickness2" in p:
        layers.append(Layer(p["sld2"] + absorb, p["thickness2"], p["roughness2"], "film2"))
        sld_film, d_film = p["sld2"], p["thickness2"]
    import fim

    last = layers[-1]
    layers[-1] = Layer(last.sld, max(d_film - WIDTH / 2, 1.0), last.roughness, last.name)
    for k in range(SLABS):
        frac = (k + 0.5) / SLABS
        sld = sld_film + frac * (fim.SUB_SLD - sld_film)
        layers.append(Layer(sld + absorb, WIDTH / SLABS, 0.0, f"graded{k}"))
    stack = Stack(layers, Layer(fim.SUB_SLD + absorb, 0.0, 0.0, "Si"))
    inst = Instrument(
        i0=p["i0"],
        background=10.0 ** p["bkg"],
        tth_offset=p["tth_offset"],
        sigma_q=p["dq"],
        kalpha2_ratio=p["kalpha2_ratio"],
        sample_length=g4.TWO_SQRT2 * p["footprint"],
        beam_sigma=1.0,
    )
    return scan.S_COUNTS * forward(q, stack, inst)


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


def one_candidate(idx: int, rep: str, summary: dict) -> dict:
    t0 = time.time()
    cond, p_basin = gc.basin_minimum(rep, summary)
    _, q, p0, free, _ = gc.condition(cond)
    x_t = np.array([p0[k] for k in free])
    x_b = np.array([p_basin[k] for k in free])
    y_nf = graded_counts(q, p0)
    rng = np.random.default_rng(np.random.SeedSequence([gc.SEED, 2, idx]))
    datasets = {"noise_free": y_nf, "poisson": rng.poisson(y_nf).astype(float)}
    out = {"rep": rep, "condition": cond, "index": idx, "datasets": {}}
    for name, y in datasets.items():
        fits = {
            "truth": fit_one(rep, y, x_t, q, p0),
            "basin": fit_one(rep, y, x_b, q, p0),
        }
        lam_t = gc.converged_counts(q, p0, free, fits["truth"]["x"])
        lam_b = gc.converged_counts(q, p0, free, fits["basin"]["x"])
        out["datasets"][name] = {
            **fits,
            "D_truth": fits["truth"]["dev_converged"],
            "D_basin": fits["basin"]["dev_converged"],
            "margin": fits["basin"]["dev_converged"] - fits["truth"]["dev_converged"],
            "margin_engine": fits["basin"]["dev_engine"] - fits["truth"]["dev_engine"],
            "cross_dev": float(gc.deviance_terms(lam_b, lam_t).sum()),
        }
    out["seconds"] = time.time() - t0
    return out


def run(worker: int, workers: int, hours: float) -> None:
    gc.init_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    reps = json.loads(SUBSET.read_text(encoding="utf-8"))["g2b"]
    summary = gc.load_summary()
    t_end = time.time() + hours * 3600
    for idx, rep in enumerate(reps):
        if idx % workers != worker:
            continue
        path = OUT / f"{idx:02d}_{rep.replace(':', '_')}.json"
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
            row = one_candidate(idx, rep, summary)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(row), encoding="utf-8")
            tmp.replace(path)
            print(f"{rep}: {row['seconds']:.0f} s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failed candidate is recorded and skipped
            path.with_suffix(".error.json").write_text(
                json.dumps({"rep": rep, "error": repr(exc)}), encoding="utf-8"
            )
            print(f"{rep}: error {exc!r}", flush=True)
        finally:
            lock.unlink(missing_ok=True)


def report() -> None:
    sub = json.loads(SUBSET.read_text(encoding="utf-8"))
    s1 = sub["S1"]
    rows = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(OUT.glob("[0-9]*_r2a_*.json"))
        if not p.name.endswith(".error.json")
    ]
    secs = sum(r["seconds"] for r in rows)
    lines = [
        "# G2b: mismatched truth (graded interface)",
        "",
        "Generated by `experiments/005_fisher_global/g2b_run.py report`. Converged-forward scores "
        + "(ReflectoSim) decide; the engine margin is shown for comparison.",
        "",
        f"- candidates run {len(rows)} of {len(sub['g2b'])}, CPU time {secs / 3600:.2f} h",
        f"- candidates with noise-free S1* <= {S1_LIMIT:g}: {sum(s1[r['rep']] <= S1_LIMIT for r in rows)}",
        "",
        "| dataset | n | H-M1: M >= 25 | H-M2: D_truth < M | median D_truth | median M | median M (engine) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("noise_free", "poisson"):
        sel = [r["datasets"][name] for r in rows if s1[r["rep"]] <= S1_LIMIT]
        if not sel:
            continue
        m1 = np.mean([d["margin"] >= MARGIN_MIN for d in sel])
        m2 = np.mean([d["D_truth"] < d["margin"] for d in sel])
        lines.append(
            f"| {name} | {len(sel)} | {m1:.3f} | {m2:.3f} | {np.median([d['D_truth'] for d in sel]):.3g} | "
            f"{np.median([d['margin'] for d in sel]):.3g} | {np.median([d['margin_engine'] for d in sel]):.3g} |"
        )

    def verdict(frac: float) -> str:
        return "holds" if frac >= 0.9 else "fails" if frac < 0.7 else "unresolved"

    lines += [
        "",
        "Registered rule: holds at 0.90 or more, fails below 0.70, unresolved between.",
        "",
    ]
    for name in ("noise_free", "poisson"):
        sel = [r["datasets"][name] for r in rows if s1[r["rep"]] <= S1_LIMIT]
        if sel:
            m1 = np.mean([d["margin"] >= MARGIN_MIN for d in sel])
            m2 = np.mean([d["D_truth"] < d["margin"] for d in sel])
            lines.append(f"- {name}: H-M1 {verdict(m1)}, H-M2 {verdict(m2)}")
    RESULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--hours", type=float, default=16.0)
    args = ap.parse_args()
    if args.mode == "run":
        run(args.worker, args.workers, args.hours)
    else:
        report()


if __name__ == "__main__":
    main()
