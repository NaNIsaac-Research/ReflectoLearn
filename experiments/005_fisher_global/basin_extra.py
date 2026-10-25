"""Post-hoc look at the basins the verdicts single out (descriptive, after basin_report.py).

For the Q-BM2 basins, the H-BM1 misses and the curve-equivalent pairs far apart in u: parameters at
the endpoint, at the minimum and at the truth, the parameters on a bound and the q-region shares.
Runs under the X2e worktree's interpreter, which needs fit_policy for X2e's truths (R2a's are in the
scan files):

    <x2e>/.venv/Scripts/python.exe -I experiments/005_fisher_global/basin_extra.py --x2e <x2e>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "fisher_global"
FULL_DIR = OUT / "basin_full"
X2E_ROOT = 20261027


def load(cid: str, objective: str = "nf") -> dict:
    return json.loads((FULL_DIR / f"{cid.replace(':', '_')}__{objective}.json").read_text("utf-8"))


def describe(cid: str, truth_vec) -> dict:
    r = load(cid)
    w = np.subtract(r["hi"], r["lo"])
    t = truth_vec(cid, r["names"])
    du = (np.asarray(r["x_min"]) - t) / w
    return {
        "id": cid,
        "outcome": r["outcome"],
        "dnf_end": r["dnf_end"],
        "dnf_min": r["dnf_min"],
        "S1_min": r["S1_min"],
        "rho_min": r.get("rho_min"),
        "on_bound": r["final_checks"]["at_lo"] + r["final_checks"]["at_hi"],
        "params": {
            n: {
                "end": r["x_end"][i],
                "min": r["x_min"][i],
                "truth": float(t[i]),
                "du": float(du[i]),
            }
            for i, n in enumerate(r["names"])
        },
        "share_min": r.get("share_min"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--x2e", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.x2e / "src"))
    from reflectolearn.fit_policy import stack as st

    def truth_vec(cid: str, names: list[str]) -> np.ndarray:
        parts = cid.split(":")
        if parts[0] == "r2a":
            scan = ROOT / "docs" / "identifiability" / "scan" / f"fits_{int(parts[1]):02d}.json"
            truth = json.loads(scan.read_text("utf-8"))["truth"]
            return np.array([truth[n] for n in names])
        return st.sample_truth(int(parts[2]), int(parts[3]), X2E_ROOT).vector()

    verdict = json.loads((OUT / "basin_verdict.json").read_text("utf-8"))
    out: dict = {"note": "post hoc, descriptive; no verdict depends on this file"}
    for src in ("r2a", "x2e"):
        v = verdict["verdicts"][src]
        out[f"{src}_qbm2"] = [describe(b["rep"], truth_vec) for b in v["Q-BM2"]["basins"]]
        far = [p for b in verdict["basins"][src] for p in b["far_pairs"]]
        out[f"{src}_far_pairs"] = [
            {
                "pair": p[:2],
                "u": p[2],
                "a": describe(p[0], truth_vec),
                "b": describe(p[1], truth_vec),
            }
            for p in far
        ]
    misses = []
    for p in sorted(FULL_DIR.glob("*__nf.json")):
        r = json.loads(p.read_text("utf-8"))
        if r["outcome"] == "confirmed" and (r["truth_basin"] or r["kappa"] > 10):
            misses.append(describe(r["id"], truth_vec) | {"kappa": r["kappa"]})
    out["misses"] = misses
    (OUT / "basin_extra.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'basin_extra.json'}")


if __name__ == "__main__":
    main()
