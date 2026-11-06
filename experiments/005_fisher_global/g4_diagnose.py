"""Which forward is right where the two engines of G4 differ? Forward only, no fit.

For the items of `g4_sim_forward.json` where the engine of the recorded results (15-node
Gauss-Hermite resolution quadrature in q) and the ReflectoSim forward (adaptive resolution
integral) give the most different deviances or counts, a third curve is computed with the engine's own
physics and a dense Gauss-Hermite rule (many nodes), which converges to the resolution convolution.
The two engines are scored against it, and ablations of the 15-node rule show the cause.

    <venv python> -I experiments/005_fisher_global/g4_diagnose.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g4_sim_forward as g4
import scan

from reflectolearn.fit_policy import stack as st

DENSE = 401
N_WORST_DD, N_WORST_COUNTS = 6, 3


def dense_nodes(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform grid over +-8 sigma with Gaussian weights: the trapezoid rule on a smooth weight
    converges exponentially in the step, and unlike a fixed Gauss-Hermite rule it samples the fringes
    of the integrand at a spacing of sigma / 25 or finer."""
    x = np.linspace(-8.0, 8.0, n)
    w = np.exp(-0.5 * x**2)
    return torch.as_tensor(x), torch.as_tensor(w / w.sum())


def with_nodes(module, n: int, fn):
    old = module.GH_U, module.GH_W
    module.GH_U, module.GH_W = dense_nodes(n)
    try:
        return fn()
    finally:
        module.GH_U, module.GH_W = old


def load_item(row: dict, summary: dict):
    """(kind, q, params or vector, truth params or vector, extra) of a G4 row."""
    if row["source"] == "r2a":
        cond = int(row["condition"] if "condition" in row else row["id"].split(":")[1])
        data = json.loads((g4.SCAN_DIR / f"fits_{cond:02d}.json").read_text(encoding="utf-8"))
        p0 = data["truth"]
        q = scan.q_grid(data["condition"]["tth_max"])
        if row["kind"] == "endpoint":
            e = next(e for e in data["endpoints"] if e["start"] == row["start"])
            p = e["params"]
        else:
            rec = summary[row["id"]]
            p = p0 | dict(zip(rec["names"], rec["x_min"], strict=True))
        return "r2a", q, p, p0, None
    _, _, n_s, j_s, s_s, _ = row["id"].split(":")[:6]
    n, j, s = int(n_s), int(j_s), int(s_s)
    x_t = st.sample_truth(n, j, g4.X2E_ROOT).vector()
    x = np.asarray(summary[row["id"]]["x_min"], float)
    return "x2e", st.q_grid(), x, x_t, (n, st.NOISE_SCALES[s])


def curves(kind, q, p, extra) -> dict[str, np.ndarray]:
    if kind == "r2a":
        eng = g4.r2a_engine(q, p)
        ref = with_nodes(scan, DENSE, lambda: g4.r2a_engine(q, p))
        check = with_nodes(scan, 2 * DENSE - 1, lambda: g4.r2a_engine(q, p))
        coarse = {
            n: with_nodes(scan, n, lambda n=n: g4.r2a_engine(q, p)) for n in (15, 31, 61, 101)
        }
        return {
            "engine": eng,
            "sim": g4.r2a_sim(q, p),
            "dense": ref,
            "dense2x": check,
            **{f"gh{n}": v for n, v in coarse.items()},
        }
    n, scale = extra
    eng = g4.x2e_engine(q, p, n, scale)
    ref = with_nodes(st, DENSE, lambda: g4.x2e_engine(q, p, n, scale))
    check = with_nodes(st, 2 * DENSE - 1, lambda: g4.x2e_engine(q, p, n, scale))
    coarse = {
        k: with_nodes(st, k, lambda k=k: g4.x2e_engine(q, p, n, scale)) for k in (15, 31, 61, 101)
    }
    return {
        "engine": eng,
        "sim": g4.x2e_sim(q, p, n, scale),
        "dense": ref,
        "dense2x": check,
        **{f"gh{k}": v for k, v in coarse.items()},
    }


def main() -> None:
    torch.set_num_threads(1)
    rows = json.loads(g4.RESULT.read_text(encoding="utf-8"))["rows"]
    summary = {}
    for line in g4.SUMMARY.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["objective"] == "nf":
            summary[r["id"]] = r
    meas = [r for r in rows if r["kind"] in ("endpoint", "minimum") and r.get("rel_dD") is not None]
    meas = [r for r in meas if r["dD_engine"] > 4]
    by_dd = sorted(meas, key=lambda r: -r["rel_dD"])[:N_WORST_DD]
    by_counts = sorted(meas, key=lambda r: -r["max_rel_counts"])
    picked = list(by_dd)
    for r in by_counts:
        if len(picked) >= N_WORST_DD + N_WORST_COUNTS:
            break
        if r not in picked:
            picked.append(r)
    outlier = next(r for r in rows if r.get("id") == "x2e:S1:2:11:1:1" and r["kind"] == "minimum")
    picked.append(outlier)
    out = []
    for row in picked:
        kind, q, p, p0, extra = load_item(row, summary)
        cur_e = curves(kind, q, p, extra)
        cur_t = curves(kind, q, p0, extra)
        entry = {
            "row": {k: row[k] for k in row if k in ("source", "kind", "id", "condition", "start")}
        }
        entry["rel_dD_recorded"] = row["rel_dD"]
        entry["dD"] = {k: float(g4.deviance_terms(cur_e[k], cur_t[k]).sum()) for k in cur_e}
        entry["max_rel_counts_vs_dense"] = {
            k: float(np.abs(cur_e[k] / cur_e["dense"] - 1).max()) for k in cur_e if k != "dense"
        }
        entry["truth_max_rel_counts_vs_dense"] = {
            k: float(np.abs(cur_t[k] / cur_t["dense"] - 1).max()) for k in cur_t if k != "dense"
        }
        out.append(entry)
        print(json.dumps(entry, default=float), flush=True)
    (g4.OUT / "g4_diagnose.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
