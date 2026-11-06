"""G1 of docs/paper/prereg/PREREG-G1.md: public repeated scans of an Au / Cr / Si sample.

Multi-start refits of five drawn repeats, S1* between distinct endpoints, reproducibility across the 100
repeats, measured dispersion. No independent structure exists for this sample; nothing here uses one.
Search and refinement run on the experiment engine's machinery (a torch forward with the fixed 15-node
resolution rule); every endpoint is polished on the converged ReflectoSim forward and only the converged
scores count (PREREG-G1.md, "Forward models").

Counts model, fixed after the pre-check (the stored R equals (raw - bg) * af / (etfp * N) to 1e-15):
    lambda_i = k_i * i0 * R(q_i + offset) + 10**b * bg_i,   k_i = etfp_i * N / af_i

    <venv python> -I experiments/005_fisher_global/g1_run.py precheck
    <venv python> -I experiments/005_fisher_global/g1_run.py search --worker 0 --workers 3 [--model M2]
    <venv python> -I experiments/005_fisher_global/g1_run.py repeats --worker 0 --workers 3
    <venv python> -I experiments/005_fisher_global/g1_run.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import basin
import g_common as gc
from reflectosim import Instrument, Layer, Stack, forward

from reflectolearn.physics import abeles_core

ORT = (
    Path(os.environ.get("DATA", "."))
    / "xrr-public"
    / "zenodo-18548583"
    / "9.95 ms_50 Hz triangle.ort"
)
OUT = gc.ROOT / "docs" / "fisher_global" / "g1"
WAVELENGTH = 0.68939  # Å, 17984.55 eV (ORT header)
N_SCANS, N_DRAW, N_START = 100, 5, 64
SEED_DRAW = np.random.default_rng(gc.SEED)

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_U, GH_WT = torch.as_tensor(GH_X), torch.as_tensor(GH_W / GH_W.sum())

# bulk SLD (real, imaginary) in 1e-6 Å^-2 from the ORT header's model
AU, CR, SI = (126.44, 13.716), (57.173, 1.3496), (19.838, 0.09052)
NOMINAL = {"Au_d": 300.0, "Cr_d": 80.0}


@dataclass(frozen=True)
class Model:
    name: str
    names: tuple[str, ...]
    lo: tuple[float, ...]
    hi: tuple[float, ...]


M1 = Model(
    "M1",
    ("Au_d", "Au_sig", "Au_rho", "Cr_d", "Cr_sig", "Cr_rho", "Si_sig", "i0", "b", "off", "sigq"),
    (150.0, 0.0, 0.5 * AU[0], 40.0, 0.0, 0.5 * CR[0], 0.0, 0.2, -1.0, -2e-3, 1e-5),
    (600.0, 40.0, 1.5 * AU[0], 160.0, 40.0, 1.5 * CR[0], 40.0, 5.0, 1.0, 2e-3, 1e-3),
)
M2 = Model(
    "M2",
    ("ox_d", "ox_sig", "ox_rho", *M1.names),
    (0.0, 0.0, 20.0, *M1.lo),
    (40.0, 15.0, 120.0, *M1.hi),
)
MODELS = {"M1": M1, "M2": M2}


def load_scans() -> dict[str, np.ndarray]:
    sets, cur = [], None
    for line in ORT.read_text(encoding="utf-8").splitlines():
        if line.startswith("# data_set:"):
            cur = []
            sets.append(cur)
        elif line and not line.startswith("#") and cur is not None:
            cur.append([float(v) for v in line.split()])
    a = np.array(sets)
    q, r, _, _, raw, bg, n, af, etfp = (a[..., i] for i in range(9))
    return {
        "q": q[0],
        "R": r,
        "raw": raw,
        "bg": bg,
        "N": n[:, 0],
        "af": af[0],
        "etfp": etfp[0],
        "af_all": af,
    }


def precheck() -> dict:
    d = load_scans()
    x = (d["raw"] - d["bg"]) * d["af_all"] / (d["etfp"][None] * d["N"][:, None])
    rel = float(np.abs(d["R"] / x - 1).max())
    ok = rel <= 1e-6 and float(np.ptp(d["af_all"], axis=0).max()) <= 1e-6 * float(d["af"].max())
    return {
        "scans": len(d["N"]),
        "points": int(d["q"].size),
        "max_rel_R_vs_counts_relation": rel,
        "counts_integer": bool(np.all(d["raw"] == np.round(d["raw"]))),
        "min_counts": float(d["raw"].min()),
        "usable": bool(ok),
    }


# ------------------------------------------------------------------ forwards
def stack_arrays(model: Model, x):
    """(thickness, roughness, complex SLD) lists in the order air, [oxide], Au, Cr, Si for torch or numpy."""
    v = dict(zip(model.names, x, strict=True))
    layers = []
    if "ox_d" in v:
        layers.append((v["ox_d"], v["ox_sig"], v["ox_rho"], 0.1 * v["ox_rho"]))
    layers.append((v["Au_d"], v["Au_sig"], v["Au_rho"], AU[1] * v["Au_rho"] / AU[0]))
    layers.append((v["Cr_d"], v["Cr_sig"], v["Cr_rho"], CR[1] * v["Cr_rho"] / CR[0]))
    return layers, v


def engine_counts(
    model: Model, q: torch.Tensor, x: torch.Tensor, k: torch.Tensor, bgm: torch.Tensor
) -> torch.Tensor:
    layers, v = stack_arrays(model, list(x))
    thk = torch.stack([torch.zeros_like(x[0]), *[layer[0] for layer in layers]])
    rgh = torch.stack([*[layer[1] for layer in layers], v["Si_sig"]])
    sld = torch.stack(
        [
            torch.zeros_like(x[0]).to(torch.complex128),
            *[torch.complex(layer[2], layer[3]) for layer in layers],
            torch.complex(torch.zeros_like(x[0]) + SI[0], torch.zeros_like(x[0]) + SI[1]),
        ]
    )
    qq = (q[:, None] + v["off"] + v["sigq"] * GH_U[None, :]).reshape(-1)
    r = abeles_core(
        qq.to(torch.complex128).reshape(1, -1, 1),
        thk.reshape(1, 1, -1),
        rgh.reshape(1, 1, -1),
        (sld * 1e-6).reshape(1, 1, -1),
    )[0]
    r = (r.reshape(-1, len(GH_U)) * GH_WT).sum(1)
    return k * v["i0"] * r + 10.0 ** v["b"] * bgm


def sim_counts(
    model: Model, q: np.ndarray, x: np.ndarray, k: np.ndarray, bgm: np.ndarray
) -> np.ndarray:
    layers, v = stack_arrays(model, [float(t) for t in x])
    lay = [Layer(complex(rho, im), d, sig) for d, sig, rho, im in layers]
    stack = Stack(lay, Layer(complex(*SI), 0.0, float(v["Si_sig"]), "Si"))
    inst = Instrument(wavelength=WAVELENGTH, i0=1.0, background=0.0, sigma_q=float(v["sigq"]))
    return k * v["i0"] * forward(q + v["off"], stack, inst) + 10.0 ** v["b"] * bgm


def make_problem(model: Model, data: dict, scan: int, cid: str, x0: np.ndarray) -> basin.Problem:
    q = data["q"]
    y = data["raw"][scan]
    k = data["etfp"] * data["N"][scan] / data["af"]
    bgm = data["bg"][scan]
    qt, kt, bt = torch.as_tensor(q), torch.as_tensor(k), torch.as_tensor(bgm)

    def log_counts(theta: torch.Tensor) -> torch.Tensor:
        return torch.log(engine_counts(model, qt, theta, kt, bt))

    def full_counts(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.exp(log_counts(torch.as_tensor(x))).numpy()

    lam0 = full_counts(x0)
    return basin.Problem(
        cid=cid,
        objective="nf",
        names=list(model.names),
        log_counts=log_counts,
        y=y,
        lo=np.array(model.lo),
        hi=np.array(model.hi),
        x_end=np.asarray(x0, float),
        terms=gc.deviance_terms,
        recorded=float(gc.deviance_terms(lam0, y).sum()),
        lam_t=y,
        full_counts=full_counts,
        theta_t=None,
        fisher=None,
        scale=float(k[0]),
        region_idx=None,
    )


def draw_scans() -> list[int]:
    return sorted(
        int(i) for i in np.random.default_rng(gc.SEED).choice(N_SCANS, N_DRAW, replace=False)
    )


def sim_fn(model: Model, data: dict, scan: int):
    k = data["etfp"] * data["N"][scan] / data["af"]
    return lambda x: sim_counts(model, data["q"], np.asarray(x), k, data["bg"][scan])


# ------------------------------------------------------------------ stages
def search(model: Model, worker: int, workers: int, hours: float) -> None:
    gc.init_worker()
    data = load_scans()
    pre = precheck()
    if not pre["usable"]:
        raise SystemExit(f"pre-check failed, set dropped: {pre}")
    out = OUT / f"search_{model.name}"
    out.mkdir(parents=True, exist_ok=True)
    scans = draw_scans()[:1] if model.name == "M2" else draw_scans()
    t_end = time.time() + hours * 3600
    jobs = [(s, j) for s in scans for j in range(N_START)]
    for n, (s, j) in enumerate(jobs):
        if n % workers != worker:
            continue
        path = out / f"s{s:03d}_{j:02d}.json"
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
            rng = np.random.default_rng(np.random.SeedSequence([gc.SEED, 1, s, j]))
            x0 = rng.uniform(model.lo, model.hi)
            prob = make_problem(model, data, s, f"g1:{s}:{j}", x0)
            row = basin.refine(prob)
            pol = gc.polish(
                prob, np.array(row["x_min"]), None, None, data["raw"][s], sim_fn(model, data, s)
            )
            rec = {
                "scan": s,
                "start": j,
                "x0": x0.tolist(),
                "outcome": row["outcome"],
                "dev_engine": row["d_final"],
                "x_engine": row["x_min"],
                "seconds_engine": row["seconds"],
                "names": list(model.names),
                **pol,
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec), encoding="utf-8")
            tmp.replace(path)
            print(
                f"s{s} start {j}: engine {rec['dev_engine']:.1f} converged {rec['dev_converged']:.1f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - recorded and skipped
            path.with_suffix(".error.json").write_text(
                json.dumps({"scan": s, "start": j, "error": repr(exc)}), encoding="utf-8"
            )
            print(f"s{s} start {j}: error {exc!r}", flush=True)
        finally:
            lock.unlink(missing_ok=True)


def best_start() -> np.ndarray:
    """Converged minimum with the lowest deviance of the first drawn scan (M1)."""
    s = draw_scans()[0]
    recs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (OUT / "search_M1").glob(f"s{s:03d}_*.json")
        if not p.name.endswith(".error.json")
    ]
    return np.array(min(recs, key=lambda r: r["dev_converged"])["x"])


def repeats(worker: int, workers: int, hours: float) -> None:
    gc.init_worker()
    data = load_scans()
    out = OUT / "repeats"
    out.mkdir(parents=True, exist_ok=True)
    x0 = best_start()
    t_end = time.time() + hours * 3600
    for s in range(N_SCANS):
        if s % workers != worker:
            continue
        path = out / f"r{s:03d}.json"
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
            prob = make_problem(M1, data, s, f"g1:{s}:rep", x0)
            row = basin.refine(prob)
            pol = gc.polish(
                prob, np.array(row["x_min"]), None, None, data["raw"][s], sim_fn(M1, data, s)
            )
            lam = sim_counts(
                M1,
                data["q"],
                np.array(pol["x"]),
                data["etfp"] * data["N"][s] / data["af"],
                data["bg"][s],
            )
            ev = basin.Evaluator(prob)
            _, jlog = ev.model(np.array(pol["x"]))
            rec = {
                "scan": s,
                "outcome": row["outcome"],
                "dev_engine": row["d_final"],
                "names": list(M1.names),
                "lam": lam.tolist(),
                "jlog": jlog.tolist(),
                **pol,
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec), encoding="utf-8")
            tmp.replace(path)
            print(f"repeat {s}: converged {rec['dev_converged']:.1f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - recorded and skipped
            path.with_suffix(".error.json").write_text(
                json.dumps({"scan": s, "error": repr(exc)}), encoding="utf-8"
            )
            print(f"repeat {s}: error {exc!r}", flush=True)
        finally:
            lock.unlink(missing_ok=True)


# ------------------------------------------------------------------ analysis
CROSS = 1.0  # same basin when both cross-deviances are at most this
H_A_SEP = 25.0
H_B_CROSS = 25.0


def basins_of_scan(model: Model, data: dict, scan: int, recs: list[dict]) -> list[dict]:
    k = data["etfp"] * data["N"][scan] / data["af"]
    lams = [sim_counts(model, data["q"], np.array(r["x"]), k, data["bg"][scan]) for r in recs]
    n = len(recs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(n):
        for b in range(a + 1, n):
            if (
                gc.deviance_terms(lams[a], lams[b]).sum() <= CROSS
                and gc.deviance_terms(lams[b], lams[a]).sum() <= CROSS
            ):
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        best = min(members, key=lambda i: recs[i]["dev_converged"])
        out.append(
            {
                "members": [recs[i]["start"] for i in members],
                "best": best,
                "dev": recs[best]["dev_converged"],
            }
        )
    out.sort(key=lambda b: b["dev"])
    for b in out:
        b["dD"] = b["dev"] - out[0]["dev"]
        b["cross"] = float(gc.deviance_terms(lams[b["best"]], lams[out[0]["best"]]).sum())
        b["S1_star"] = (
            float(
                k[0] * np.array(recs[out[0]["best"]]["x"])[list(M1.names).index("i0")] / b["cross"]
            )
            if b["cross"] > 0
            else None
        )
    return out


def report() -> None:
    data = load_scans()
    scans = draw_scans()
    lines = [
        "# G1: public Au / Cr / Si repeats, no independent truth",
        "",
        "Generated by `experiments/005_fisher_global/g1_run.py report`. Converged-forward scores only. "
        + "Nothing here is compared with a true structure.",
        "",
    ]
    sep, tot = 0, 0
    bests = {}
    for s in scans:
        recs = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((OUT / "search_M1").glob(f"s{s:03d}_*.json"))
            if not p.name.endswith(".error.json")
        ]
        if not recs:
            continue
        bs = basins_of_scan(M1, data, s, recs)
        bests[s] = np.array(recs[bs[0]["best"]]["x"])
        others = bs[1:]
        sep += sum(b["dD"] >= H_A_SEP for b in others)
        tot += len(others)
        lines += [
            f"## scan {s}: {len(recs)} starts, {len(bs)} distinct basins, best deviance {bs[0]['dev']:.1f}",
            "",
            "| basin | starts | deviance | dD | cross-dev with best | S1* (counts at R = 1) |",
            "|---|---|---|---|---|---|",
        ]
        for i, b in enumerate(bs):
            s1 = f"{b['S1_star']:.3g}" if b["S1_star"] else "n/a"
            lines.append(
                f"| {i} | {len(b['members'])} | {b['dev']:.1f} | {b['dD']:.3g} | {b['cross']:.3g} | {s1} |"
            )
        lines.append("")
    if tot:
        lines.append(
            f"H-G1a: {sep} of {tot} other basins separated from the best by dD >= {H_A_SEP:g} ({sep / tot:.3f}); holds >= 0.80, fails < 0.60."
        )
    names = list(M1.names)
    if len(bests) == len(scans) and len(scans) > 1:
        ref = bests[scans[0]]
        k0 = data["etfp"] * data["N"][scans[0]] / data["af"]
        lam_ref = sim_counts(M1, data["q"], ref, k0, data["bg"][scans[0]])
        cross = []
        for s in scans[1:]:
            lam_s = sim_counts(M1, data["q"], bests[s], k0, data["bg"][scans[0]])
            cross.append(float(gc.deviance_terms(lam_s, lam_ref).sum()))
        lines.append(
            f"H-G1b: cross-deviances of the best minima against scan {scans[0]}'s: {[round(c, 1) for c in cross]}; holds if all <= {H_B_CROSS:g}."
        )
    reps = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((OUT / "repeats").glob("r*.json"))
        if not p.name.endswith(".error.json")
    ]
    if reps:
        xs = np.array([r["x"] for r in reps])
        pearson = np.array(
            [
                (data["raw"][r["scan"]] - np.array(r["lam"])) / np.sqrt(np.array(r["lam"]))
                for r in reps
            ]
        )
        phi = float(np.median(np.var(pearson, axis=0, ddof=1)))
        lines.append(
            f"Dispersion of the Pearson residuals about the 100 best fits: median phi {phi:.3f}."
        )
        j = np.array(reps[0]["jlog"])
        for key in ("Au_d", "Cr_d"):
            i = names.index(key)
            sds = float(np.std(xs[:, i], ddof=1))
            lam = np.array(reps[0]["lam"])
            fisher = j.T @ (lam[:, None] * j)
            crb = float(np.sqrt(phi * np.linalg.inv(fisher)[i, i]))
            lines.append(
                f"H-G1c {key}: across-scan sd {sds:.3g} A, dispersion-scaled CRB {crb:.3g} A, ratio {sds / crb:.2f}; holds <= 2, fails > 3."
            )
        lines.append(
            "Fitted values (median over repeats), for description only, not a truth: "
            + ", ".join(f"{n} {np.median(xs[:, i]):.4g}" for i, n in enumerate(names))
        )
    (OUT / "g1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["precheck", "search", "repeats", "report"])
    ap.add_argument("--model", choices=list(MODELS), default="M1")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--hours", type=float, default=16.0)
    args = ap.parse_args()
    if args.mode == "precheck":
        OUT.mkdir(parents=True, exist_ok=True)
        res = precheck()
        (OUT / "precheck.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(res)
    elif args.mode == "search":
        search(MODELS[args.model], args.worker, args.workers, args.hours)
    elif args.mode == "repeats":
        repeats(args.worker, args.workers, args.hours)
    else:
        report()


if __name__ == "__main__":
    main()
