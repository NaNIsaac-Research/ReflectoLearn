"""Forward-only re-evaluation of the recorded R2a and X2e endpoints and truths with the ReflectoSim
forward model (gap G4 of docs/paper/paper_draft.md). No fit and no new start.

The experiments of this series computed their counts with this repository's torch engine
(`scan.expected_counts` for R2a, `fit_policy.stack.expected_counts` for X2e). This script evaluates
the same parameters with `reflectosim.forward` and compares the counts and the deviance of every
recorded endpoint from its truth, and the deviance at the refined basin minima of the basin run.

Mapping to ReflectoSim:
- R2a media: ambient / top oxide / film / [film 2] / Si. The engine adds an absorption of 1e-9 Å⁻² to
  every medium including the ambient; ReflectoSim requires a non-absorbing ambient, so the ambient
  stays at 0 and every other medium gets Im SLD 1e-3 (in 1e-6 Å⁻²). That is the one difference in
  the model, and it is the one the engine's own check (docs/identifiability/README.md) names.
- Kα₂ at the same angle as Kα₁ with intensity ratio `kalpha2_ratio`; footprint factor
  erf(footprint · sin θ) is sample_length = 2√2 · footprint with beam_sigma = 1; the constant
  resolution `dq` is sigma_q; the 2θ offset is tth_offset; background 10^bkg is added to i0 · R and
  the sum is scaled by the count level.
- X2e: real SLDs, no Kα₂, no footprint, the same resolution and offset.

    <venv python> -I experiments/005_fisher_global/g4_sim_forward.py run
    <venv python> -I experiments/005_fisher_global/g4_sim_forward.py report

The interpreter needs reflectolearn and reflectosim (the repository's environment).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "001_identifiability"))
sys.path.insert(0, str(ROOT / "src"))
import fim  # noqa: E402
import scan  # noqa: E402
from reflectosim import Instrument, Layer, Stack, forward  # noqa: E402

from reflectolearn.fit_policy import stack as st  # noqa: E402

OUT = ROOT / "docs" / "fisher_global"
SCAN_DIR = ROOT / "docs" / "identifiability" / "scan"
SUMMARY = OUT / "basin_full_summary.jsonl"
RESULT = OUT / "g4_sim_forward.json"
REPORT = OUT / "g4_sim_forward.md"
X2E_ROOT = 20261027
ABSORPTION = 1e-3  # 1e-9 Å⁻² in units of 1e-6 Å⁻²
TWO_SQRT2 = 2.0 * np.sqrt(2.0)


def deviance_terms(lam: np.ndarray, lam_t: np.ndarray) -> np.ndarray:
    """The Poisson deviance against noise-free data y = λ_t, as b1.py writes it."""
    r = lam / lam_t - 1
    small = np.abs(r) < 1e-3
    rs = np.where(small, r, 0.0)
    series = rs**2 / 2 - rs**3 / 3 + rs**4 / 4 - rs**5 / 5
    direct = r - np.log1p(np.where(small, 0.0, r))
    return 2 * lam_t * np.where(small, series, direct)


# ------------------------------------------------------------------ R2a
def r2a_engine(q: np.ndarray, p: dict[str, float]) -> np.ndarray:
    with torch.no_grad():
        return scan.expected_counts(
            torch.as_tensor(q), {k: torch.tensor(v) for k, v in p.items()}
        ).numpy()


def r2a_sim(q: np.ndarray, p: dict[str, float]) -> np.ndarray:
    absorb = 1j * ABSORPTION
    layers = [
        Layer(p["oxide_sld"] + absorb, p["oxide_thickness"], p["oxide_roughness"], "oxide"),
        Layer(p["sld"] + absorb, p["thickness"], p["roughness"], "film"),
    ]
    if "thickness2" in p:
        layers.append(Layer(p["sld2"] + absorb, p["thickness2"], p["roughness2"], "film2"))
    stack = Stack(layers, Layer(fim.SUB_SLD + absorb, 0.0, fim.SUB_ROUGH, "Si"))
    inst = Instrument(
        i0=p["i0"],
        background=10.0 ** p["bkg"],
        tth_offset=p["tth_offset"],
        sigma_q=p["dq"],
        kalpha2_ratio=p["kalpha2_ratio"],
        sample_length=TWO_SQRT2 * p["footprint"],
        beam_sigma=1.0,
    )
    return scan.S_COUNTS * forward(q, stack, inst)


# ------------------------------------------------------------------ X2e
def x2e_engine(q: np.ndarray, x: np.ndarray, n: int, scale: float) -> np.ndarray:
    with torch.no_grad():
        return st.expected_counts(torch.as_tensor(q), torch.as_tensor(x), n, scale).numpy()


def x2e_sim(q: np.ndarray, x: np.ndarray, n: int, scale: float) -> np.ndarray:
    layers = [
        Layer(float(rho), float(d), float(sigma)) for d, sigma, rho in x[: 3 * n].reshape(n, 3)
    ]
    stack = Stack(layers, Layer(st.SUB_SLD, 0.0, st.SUB_ROUGH, "Si"))
    i0, bkg, offset, dq = x[3 * n :]
    inst = Instrument(i0=i0, background=10.0**bkg, tth_offset=offset, sigma_q=dq)
    return scale * forward(q, stack, inst)


def compare(lam_a: np.ndarray, lam_b: np.ndarray, lam_ta: np.ndarray, lam_tb: np.ndarray) -> dict:
    """Engine a against engine b: the counts, and the deviance of the state from the truth."""
    rel = np.abs(lam_b / lam_a - 1)
    da = float(deviance_terms(lam_a, lam_ta).sum())
    db = float(deviance_terms(lam_b, lam_tb).sum())
    return {
        "max_rel_counts": float(rel.max()),
        "dD_engine": da,
        "dD_sim": db,
        "rel_dD": abs(db - da) / da if da > 0 else None,
    }


def run() -> None:
    torch.set_num_threads(1)
    t0 = time.time()
    rows = []
    truths = {}
    for path in sorted(SCAN_DIR.glob("fits_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        idx, cond, p0 = data["index"], data["condition"], data["truth"]
        q = scan.q_grid(cond["tth_max"])
        lam_te, lam_ts = r2a_engine(q, p0), r2a_sim(q, p0)
        truths[idx] = (q, p0, lam_te, lam_ts)
        rows.append(
            {
                "source": "r2a",
                "kind": "truth",
                "condition": idx,
                "max_rel_counts": float(np.abs(lam_ts / lam_te - 1).max()),
            }
        )
        for e in data["endpoints"]:
            c = compare(r2a_engine(q, e["params"]), r2a_sim(q, e["params"]), lam_te, lam_ts)
            rows.append(
                {
                    "source": "r2a",
                    "kind": "endpoint",
                    "condition": idx,
                    "start": e["start"],
                    "dD_recorded": e["deviance"],
                    **c,
                }
            )
    print(f"R2a endpoints done, {time.time() - t0:.0f} s", flush=True)

    x2e_truth = {}
    for line in SUMMARY.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["objective"] != "nf" or rec["outcome"] != "confirmed":
            continue
        cid = rec["id"]
        if cid.startswith("r2a:"):
            _, cc, _ = cid.split(":")
            q, p0, lam_te, lam_ts = truths[int(cc)]
            p = p0 | dict(zip(rec["names"], rec["x_min"], strict=True))
            c = compare(r2a_engine(q, p), r2a_sim(q, p), lam_te, lam_ts)
            rows.append({"source": "r2a", "kind": "minimum", "id": cid, "dD_min": rec["dnf_min"], **c})
        else:
            _, _, n_s, j_s, s_s, _ = cid.split(":")[:6]
            n, j, s = int(n_s), int(j_s), int(s_s)
            q = st.q_grid()
            scale = st.NOISE_SCALES[s]
            key = (n, j, s)
            if key not in x2e_truth:
                x_t = st.sample_truth(n, j, X2E_ROOT).vector()
                x2e_truth[key] = (x_t, x2e_engine(q, x_t, n, scale), x2e_sim(q, x_t, n, scale))
            x_t, lam_te, lam_ts = x2e_truth[key]
            x = np.asarray(rec["x_min"], float)
            c = compare(x2e_engine(q, x, n, scale), x2e_sim(q, x, n, scale), lam_te, lam_ts)
            rows.append({"source": "x2e", "kind": "minimum", "id": cid, "dD_min": rec["dnf_min"], **c})
            rows.append(
                {
                    "source": "x2e",
                    "kind": "truth",
                    "id": cid,
                    "max_rel_counts": float(np.abs(lam_ts / lam_te - 1).max()),
                }
            )
    RESULT.write_text(
        json.dumps({"seconds": time.time() - t0, "rows": rows}, indent=1) + "\n", encoding="utf-8"
    )
    print(f"done, {time.time() - t0:.0f} s, {len(rows)} rows", flush=True)


def quant(x: list[float]) -> str:
    a = np.asarray(x, float)
    if not len(a):
        return "n/a"
    lo, med, p90, hi = np.quantile(a, [0, 0.5, 0.9, 1])
    return f"{lo:.2g} / {med:.2g} / {p90:.2g} / {hi:.2g}"


def report() -> None:
    rows = json.loads(RESULT.read_text(encoding="utf-8"))["rows"]
    lines = [
        "# G4: ReflectoSim forward against this repository's engine",
        "",
        "Generated by `experiments/005_fisher_global/g4_sim_forward.py report` from `g4_sim_forward.json`.",
        "Forward evaluations only; no fit, no new start. Quantiles are min / median / p90 / max.",
        "",
        "| set | n | max relative difference in counts | relative difference of ΔD |",
        "|---|---|---|---|",
    ]
    for source, kind, label in (
        ("r2a", "truth", "R2a truths"),
        ("r2a", "endpoint", "R2a recorded endpoints"),
        ("r2a", "minimum", "R2a basin minima"),
        ("x2e", "truth", "X2e truths"),
        ("x2e", "minimum", "X2e basin minima"),
    ):
        sel = [r for r in rows if r["source"] == source and r["kind"] == kind]
        rel = [r["rel_dD"] for r in sel if r.get("rel_dD") is not None]
        lines.append(
            f"| {label} | {len(sel)} | {quant([r['max_rel_counts'] for r in sel])} | "
            f"{quant(rel) if rel else 'n/a'} |"
        )
    out = ["", "Verdict-relevant crossings (engine ΔD against ReflectoSim ΔD):", ""]
    for source, kind, thresholds in (("r2a", "endpoint", (4.0, 25.0, 1e2)), ("r2a", "minimum", (1.0, 1e2))):
        sel = [r for r in rows if r["source"] == source and r["kind"] == kind]
        for th in thresholds:
            a = np.array([r["dD_engine"] for r in sel]) <= th
            b = np.array([r["dD_sim"] for r in sel]) <= th
            out.append(
                f"- {source} {kind}, ΔD ≤ {th:g}: engine {int(a.sum())}, ReflectoSim {int(b.sum())}, "
                f"changed {int((a != b).sum())} of {len(sel)}"
            )
    REPORT.write_text("\n".join([*lines, *out, ""]), encoding="utf-8")
    print("\n".join([*lines, *out]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    args = ap.parse_args()
    run() if args.mode == "run" else report()


if __name__ == "__main__":
    main()
