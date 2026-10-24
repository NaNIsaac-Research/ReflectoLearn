"""B-2 of docs/fisher_global/PREREG.md: truth-start local fit and Δχ² = 1 profiles of R2a #25.

The observation R2a amendment 7 registered and did not run: a local fit from the truth on the
noise-free data, then the profile half-widths of d and ρ of both films with scan.half_width. It sees
only the truth's basin, so it says nothing about global uniqueness or multimodality.

    .venv/Scripts/python.exe -I experiments/005_fisher_global/b2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "001_identifiability"))
import scan  # noqa: E402

CONDITION = 25
OUT = ROOT / "docs" / "fisher_global"

calls = 0
_expected_counts = scan.expected_counts


def counted(*args, **kwargs):
    global calls
    calls += 1
    return _expected_counts(*args, **kwargs)


scan.expected_counts = counted


def main() -> None:
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    data = json.loads(
        (ROOT / "docs" / "identifiability" / "scan" / f"fits_{CONDITION:02d}.json").read_text(
            encoding="utf-8"
        )
    )
    cond, p0, free = data["condition"], data["truth"], tuple(data["free"])
    _, _, bounds = scan.setup(cond)
    q = scan.q_grid(cond["tth_max"])
    y = _expected_counts(torch.as_tensor(q), {k: torch.tensor(v) for k, v in p0.items()}).numpy()

    p_s, d_s, nfev = scan.fit(q, y, p0, free, bounds)
    print(
        f"truth-start fit: D {d_s:.3g}, nfev {nfev}, {time.perf_counter() - t0:.0f} s", flush=True
    )

    b1 = json.loads((OUT / "b1.json").read_text(encoding="utf-8"))
    b1_cond = next(c for c in b1["conditions"] if c["index"] == CONDITION)
    crlb = dict(zip(free, b1_cond["crlb"], strict=True))

    profiles = {}
    for k in scan.film_keys(free):
        t = time.perf_counter()
        c0 = calls
        w = scan.half_width(q, y, p_s, d_s, k, free, bounds)
        w |= {
            "value": p_s[k],
            "relative": w["half_width"] / abs(p_s[k]),
            "label": "ridge" if w["half_width"] >= 0.02 * abs(p_s[k]) else "identifiable",
            "fisher_half_width": crlb[k],
            "forward_calls": calls - c0,
            "seconds": time.perf_counter() - t,
        }
        profiles[k] = w
        print(
            f"{k}: half-width {w['half_width']:.4g} ({w['relative']:.2%}), Fisher {crlb[k]:.4g},"
            f" at bound {w['at_bound']}, lowest {w['lowest_dchi2']:.3g}, {w['seconds']:.0f} s",
            flush=True,
        )

    out = {
        "condition": CONDITION,
        "truth_start": {
            "deviance": d_s,
            "nfev": nfev,
            "shift": {k: p_s[k] - p0[k] for k in free},
        },
        "profiles": profiles,
        "forward_calls": calls,
        "seconds": time.perf_counter() - t0,
    }
    (OUT / "b2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"total {out['seconds']:.0f} s, forward calls {calls}")


if __name__ == "__main__":
    main()
