"""Figures 4 and 5 of docs/paper/paper_draft.md, drawn from the recorded results only.

    uv run python experiments/005_fisher_global/paper_figs.py

Figure 4: S₁* of the distinct wrong basins of the basin run (basin_verdict.json), R2a and X2e.
Figure 5: (a) the Fisher ratio ρ = ΔD_F / ΔD of R2a's recorded endpoints against their deviance, and
(b) the share of each endpoint's deviance by q region for the 291-point R2a conditions (b1.json).
Nothing is fitted or recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FG = ROOT / "docs" / "fisher_global"
OUT = ROOT / "docs" / "paper"
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#666666"
WRONG_MIN_DD = 4.0  # b1.py's wrong-endpoint threshold
REGIONS = ["low"] + [f"b{i}" for i in range(10)]


def fig4() -> None:
    basins = json.loads((FG / "basin_verdict.json").read_text(encoding="utf-8"))["basins"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, color, label in (("r2a", BLUE, "R2a"), ("x2e", ORANGE, "X2e")):
        s1 = np.sort([b["S1_min"] for b in basins[name]])
        ax.step(s1, np.arange(1, len(s1) + 1) / len(s1), where="post", color=color,
                label=f"{label}, {len(s1)} basins")  # fmt: skip
        ax.plot(s1[-1], 1.0, "o", color=color, ms=4)
    ax.axvline(1e4, color=GREY, ls="--", lw=1)
    ax.text(1.15e4, 0.06, "S₁* = 10⁴", color=GREY, fontsize=8)
    for x, text in ((1e6, "10⁶ counts"), (1e8, "10⁸ counts")):
        ax.axvline(x, color=GREY, ls=":", lw=1)
        ax.text(x * 1.2, 0.5, text, rotation=90, color=GREY, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3e8)
    ax.set_xlabel("S₁* of the wrong basin (counts at R = 1 at which it is within ΔD = 1 of the truth)")
    ax.set_ylabel("fraction of distinct wrong basins")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_basin_s1.png", dpi=200)
    plt.close(fig)


def fig5() -> None:
    conds = json.loads((FG / "b1.json").read_text(encoding="utf-8"))["conditions"]
    dd, rho, shares = [], [], []
    for c in conds:
        for r in c["rows"]:
            if r["dD"] and r["rho"]:
                dd.append(r["dD"])
                rho.append(r["rho"])
            if r.get("share") and r["dD"] > WRONG_MIN_DD:
                shares.append(r["share"])
    dd, rho, shares = np.array(dd), np.array(rho), np.array(shares, float)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1.1]})
    a.scatter(dd, rho, s=3, alpha=0.25, color=BLUE, linewidths=0)
    a.axhline(1, color=GREY, lw=1)
    a.axhline(10, color=GREY, lw=1, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("ΔD of the recorded endpoint")
    a.set_ylabel("ρ = ΔD_Fisher / ΔD")
    a.set_title(f"(a) R2a endpoints, n = {len(dd)}", fontsize=10, loc="left")
    pos = np.arange(len(REGIONS))
    b.boxplot([shares[:, i] for i in pos], positions=pos, showfliers=False, widths=0.6,
              medianprops={"color": ORANGE}, boxprops={"color": BLUE}, whiskerprops={"color": BLUE},
              capprops={"color": BLUE})  # fmt: skip
    b.set_xticks(pos, REGIONS)
    b.set_xlabel("q region (low: 2θ < 0.5°, then ten blocks to q = 0.427 Å⁻¹)")
    b.set_ylabel("share of the endpoint's ΔD")
    b.set_title(f"(b) wrong endpoints, ΔD > 4, 291-point conditions, n = {len(shares)}",
                fontsize=10, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "fig5_fisher_lowq.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig4()
    fig5()


if __name__ == "__main__":
    main()
