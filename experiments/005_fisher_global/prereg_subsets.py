"""Candidate subsets of the G2, G2b, G2c and G3 pre-registrations (docs/paper/prereg/).

Draws the subsets once, from the recorded basin run only, with fixed seeds, and writes
docs/paper/prereg/g2_subset.json and g3_subset.json. Nothing is fitted. The rules are in
PREREG-G2.md and PREREG-G3.md; this file is committed before any of those runs starts.

    <venv python> -I experiments/005_fisher_global/prereg_subsets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "001_identifiability"))
import scan  # noqa: E402

FG = ROOT / "docs" / "fisher_global"
OUT = ROOT / "docs" / "paper" / "prereg"
SEED = 20261107
S1_ALWAYS = 1e3
G2_SIZE, G2_PHI, G2B_SIZE, G2C_SIZE = 100, 30, 30, 10
G3_BOUND, G3_FREE = 100, 50
BOUND_TOL = 1e-6  # bound-scaled distance that counts as on the bound


def load() -> tuple[list[dict], dict[str, dict]]:
    verdict = json.loads((FG / "basin_verdict.json").read_text(encoding="utf-8"))
    basins = sorted(
        (b for b in verdict["basins"]["r2a"]), key=lambda b: (-b["S1_min"], b["rep"])
    )
    summary = {}
    for line in (FG / "basin_full_summary.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        summary[r["id"]] = r
    return basins, summary


def bilayer(cond: int) -> bool:
    d = json.loads(
        (ROOT / "docs" / "identifiability" / "scan" / f"fits_{cond:02d}.json").read_text(
            encoding="utf-8"
        )
    )
    return "thickness2" in d["truth"]


def kalpha2_on_bound(rep: str, summary: dict) -> bool:
    r = summary[rep]
    cond = int(rep.split(":")[1])
    d = json.loads(
        (ROOT / "docs" / "identifiability" / "scan" / f"fits_{cond:02d}.json").read_text(
            encoding="utf-8"
        )
    )
    _, _, bounds = scan.setup(d["condition"])
    lo, hi = bounds["kalpha2_ratio"]
    x = r["x_min"][r["names"].index("kalpha2_ratio")]
    return min(x - lo, hi - x) / (hi - lo) < BOUND_TOL


def pick(rng: np.random.Generator, pool: list[dict], n: int) -> list[dict]:
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    basins, summary = load()
    rng = np.random.default_rng(SEED)
    always = [b for b in basins if b["S1_min"] >= S1_ALWAYS]
    rest = [b for b in basins if b["S1_min"] < S1_ALWAYS]

    singles = [b for b in rest if not bilayer(int(b["rep"].split(":")[1]))]
    bilayers = [b for b in rest if bilayer(int(b["rep"].split(":")[1]))]
    n_extra = G2_SIZE - len(always)
    n_bi = round(n_extra * len(bilayers) / len(rest))
    g2 = [*always, *pick(rng, bilayers, n_bi), *pick(rng, singles, n_extra - n_bi)]
    g2_ids = [b["rep"] for b in g2]
    g2_phi = [b["rep"] for b in pick(rng, g2, G2_PHI)]
    g2b = [b["rep"] for b in pick(rng, g2, G2B_SIZE)]
    g2c = [b["rep"] for b in basins[:G2C_SIZE]]
    (OUT / "g2_subset.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "rule": "all basins with S1* >= 1e3, the rest drawn without replacement, "
                "single-film and bilayer conditions in proportion",
                "g2": g2_ids,
                "g2_phi": g2_phi,
                "g2b": g2b,
                "g2c_closest": g2c,
                "S1": {b["rep"]: b["S1_min"] for b in g2},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    on = [b for b in rest if kalpha2_on_bound(b["rep"], summary)]
    off = [b for b in rest if not kalpha2_on_bound(b["rep"], summary)]
    g3 = [
        *always,
        *pick(rng, on, G3_BOUND - len(always)),
        *pick(rng, off, G3_FREE),
    ]
    (OUT / "g3_subset.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "rule": "all basins with S1* >= 1e3, 100 minus those drawn from basins with the "
                "Kalpha2 ratio on a bound, 50 drawn from the others",
                "g3": [b["rep"] for b in g3],
                "kalpha2_on_bound": [b["rep"] for b in g3 if kalpha2_on_bound(b["rep"], summary)],
                "S1": {b["rep"]: b["S1_min"] for b in g3},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"basins {len(basins)}, always {len(always)}; G2 {len(g2_ids)} (phi {len(g2_phi)}, "
        f"G2b {len(g2b)}, G2c {len(g2c)}); G3 {len(g3)}"
    )


if __name__ == "__main__":
    main()
