"""Experiment 002, X2b: apply the pre-registered rules of docs/fit_policy/DESIGN.md section 6 to the
episodes in docs/fit_policy/runs/ and write docs/fit_policy/results.md, verdict.json and
delta_bic.png. The rules are applied mechanically; nothing here chooses a threshold.

Run:  uv run python experiments/002_fit_policy/report.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from reflectolearn.fit_policy import stack as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "fit_policy"
RUNS = OUT / "runs"
CLASSES = (1, 2, 3)
MULTI = [(n, s) for n in (2, 3) for s in range(len(st.NOISE_SCALES))]
BIC_BAND = 10.0
HO_BAND = 2.0  # in units of s_ho
WIN_FRAC, SIGN_P, HO_LOSS_FRAC = 0.25, 0.01, 0.10
ORACLE_FRAC, SINGLE_MAX, TIE_BIC = 0.20, 0.70, 2.0
MCNEMAR_P = 0.05
RECOVERY_TOL = 0.05
CAP_FRAC = 0.5
VARIANT_NAMES = ("A1", "A2", "A3", "A4")


def load(method: str) -> dict[tuple, dict]:
    path = RUNS / f"{method}.jsonl"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {(r["n"], r["j"], r["s"], r["k"]): r for r in map(json.loads, f)}


def recovered(rec: dict, truth: st.Stack) -> bool:
    """M3: right layer count and every layer within 5 % in d and rho (layers matched top first)."""
    if rec["n_layers"] != truth.n_layers:
        return False
    fit = np.asarray(rec["layers"])
    rel = np.abs(fit[:, [0, 2]] - truth.layers[:, [0, 2]]) / truth.layers[:, [0, 2]]
    return bool(np.all(rel <= RECOVERY_TOL))


def classify(d: float, band: float) -> int:
    """+1 win (d <= -band), -1 loss (d >= band), 0 tie."""
    return 1 if d <= -band else -1 if d >= band else 0


def truth_outcomes(xs: dict, refs: dict, keys: list, noise: dict) -> list[tuple[int, int]]:
    """Per truth of the cell: (BIC outcome, held-out outcome) of the medians over its starts of
    ΔBIC and ΔD_ho (section 6.2, truth level). Truths with a missing episode are skipped."""
    out = []
    for truth_key in sorted({k[:3] for k in keys}):
        ks = [k for k in keys if k[:3] == truth_key]
        if not all(k in xs and k in refs for k in ks):
            continue
        d_bic = float(np.median([xs[k]["bic"] - refs[k]["bic"] for k in ks]))
        d_ho = float(np.median([xs[k]["deviance_ho"] - refs[k]["deviance_ho"] for k in ks]))
        s_ho = noise[f"{truth_key[0]},{truth_key[1]},{truth_key[2]}"]["s_ho"]
        out.append((classify(d_bic, BIC_BAND), classify(d_ho, HO_BAND * s_ho)))
    return out


def beats(xs: dict, refs: dict, keys: list, noise: dict) -> dict:
    """Section 6.3 for one cell, at the truth level."""
    res = truth_outcomes(xs, refs, keys, noise)
    n = len(res)
    wins = sum(b == 1 for b, _ in res)
    losses = sum(b == -1 for b, _ in res)
    ho_losses = sum(h == -1 for _, h in res)
    p = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0
    ok = n > 0 and wins / n >= WIN_FRAC and p < SIGN_P and ho_losses / n <= HO_LOSS_FRAC
    pairs = [k for k in keys if k in xs and k in refs]
    episode_wins = [k for k in pairs if xs[k]["bic"] - refs[k]["bic"] <= -BIC_BAND]
    return {
        "truths": n,
        "bic_wins": wins,
        "bic_losses": losses,
        "bic_ties": n - wins - losses,
        "sign_p": float(p),
        "ho_wins": sum(h == 1 for _, h in res),
        "ho_losses": ho_losses,
        "beats": bool(ok),
        "episodes": len(pairs),
        "episode_bic_wins": len(episode_wins),
        # wins with the same layer count as the reference: convergence inside one basin (6.2)
        "episode_bic_wins_same_L": sum(
            xs[k]["n_layers"] == refs[k]["n_layers"] for k in episode_wins
        ),
    }


def cell_keys(n: int, s: int, n_truths: int) -> list[tuple]:
    return [(n, j, s, k) for j in range(n_truths) for k in range(st.N_STARTS)]


def learning_room(variants: dict[str, dict], keys: list) -> dict:
    """Section 6.4 for one cell."""
    keys = [k for k in keys if all(k in v for v in variants.values())]
    if not keys:
        return {"episodes": 0, "room": False}
    bics = np.array([[variants[m][k]["bic"] for m in VARIANT_NAMES] for k in keys])
    single = int(np.argmin(np.median(bics, axis=0)))
    oracle = bics.min(1)
    oracle_frac = float(np.mean(oracle - bics[:, single] <= -BIC_BAND))
    best = bics <= oracle[:, None] + TIE_BIC
    best_frac = best.mean(0)
    # the same two fractions over truths (median over starts), for reference only (6.4)
    truth_ids = sorted({k[:3] for k in keys})
    tb = np.array(
        [np.median(bics[[i for i, k in enumerate(keys) if k[:3] == t]], axis=0) for t in truth_ids]
    )
    t_oracle = tb.min(1)
    return {
        "episodes": len(keys),
        "best_single": VARIANT_NAMES[single],
        "oracle_win_frac": oracle_frac,
        "best_frac": dict(zip(VARIANT_NAMES, best_frac.tolist(), strict=True)),
        "room": bool(oracle_frac >= ORACLE_FRAC and best_frac.max() <= SINGLE_MAX),
        "truth_oracle_win_frac": float(np.mean(t_oracle - tb[:, single] <= -BIC_BAND)),
        "truth_best_frac": dict(
            zip(VARIANT_NAMES, (tb <= t_oracle[:, None] + TIE_BIC).mean(0).tolist(), strict=True)
        ),
    }


def mcnemar_less(a: dict, c: dict, keys: list, truths: dict) -> dict:
    """Does A recover the truth less often than C? One-sided exact McNemar on discordant pairs."""
    keys = [k for k in keys if k in a and k in c]
    ra = np.array([recovered(a[k], truths[k[:2]]) for k in keys])
    rc = np.array([recovered(c[k], truths[k[:2]]) for k in keys])
    only_c, only_a = int(np.sum(rc & ~ra)), int(np.sum(ra & ~rc))
    p = (
        binomtest(only_c, only_c + only_a, 0.5, alternative="greater").pvalue
        if only_c + only_a
        else 1.0
    )
    return {"only_c": only_c, "only_a": only_a, "p": p, "misaligned": bool(p < MCNEMAR_P)}


def capped_frac(c: dict, keys: list) -> float:
    """Fraction of the cell's returned C fits that stopped at their evaluation cap (6.5)."""
    flags = [c[k]["capped"] for k in keys if k in c]
    return float(np.mean(flags)) if flags else float("nan")


def describe(recs: dict, keys: list, truths: dict, c: dict | None) -> dict:
    keys = [k for k in keys if k in recs]
    if not keys:
        return {}
    out = {
        "episodes": len(keys),
        "layer_count_right": float(np.mean([recs[k]["n_layers"] == k[0] for k in keys])),
        "recovered": float(np.mean([recovered(recs[k], truths[k[:2]]) for k in keys])),
        "evaluations_median": float(np.median([recs[k].get("evaluations", np.nan) for k in keys])),
        "seconds_median": float(np.median([recs[k]["seconds"] for k in keys])),
    }
    if c is not None and "bic" in recs[keys[0]]:
        both = [k for k in keys if k in c]
        out["d_bic_median_vs_C"] = float(np.median([recs[k]["bic"] - c[k]["bic"] for k in both]))
    return out


def main() -> None:
    noise = json.loads((RUNS / "noise.json").read_text(encoding="utf-8"))
    methods = {m: load(m) for m in (*VARIANT_NAMES, "B", "C", "C2E", "Co", "Cr")}
    n_truths = 1 + max((k[1] for k in methods["C"]), default=-1)
    truths = {(n, j): st.sample_truth(n, j) for n in CLASSES for j in range(n_truths)}

    cells = {}
    for n, s in itertools.product(CLASSES, range(len(st.NOISE_SCALES))):
        keys = cell_keys(n, s, n_truths)
        cap = capped_frac(methods["C"], keys)
        ref_name = "C2E" if cap > CAP_FRAC else "C"
        ref = methods[ref_name]
        cell = {
            "c_capped_frac": cap,
            "reference": ref_name,
            "reference_missing": ref_name == "C2E" and not ref,
            "A1_vs_C": beats(methods["A1"], ref, keys, noise),
            "A1_vs_B": beats(methods["A1"], methods["B"], keys, noise),
            "describe": {m: describe(r, keys, truths, ref) for m, r in methods.items() if r},
        }
        if cell["A1_vs_C"]["beats"]:
            cell["learning_room"] = learning_room({m: methods[m] for m in VARIANT_NAMES}, keys)
            cell["reward_alignment"] = mcnemar_less(methods["A1"], ref, keys, truths)
        cells[f"n{n}_s{s}"] = cell

    multi = [cells[f"n{n}_s{s}"] for n, s in MULTI]
    n_beats = sum(c["A1_vs_C"]["beats"] for c in multi)
    verdict = (
        "no room for RL"
        if n_beats == 0
        else "inconclusive"
        if n_beats == 1
        else "procedure matters"
    )
    room = any(c.get("learning_room", {}).get("room", False) for c in multi)
    misaligned = any(c.get("reward_alignment", {}).get("misaligned", False) for c in multi)
    random_floor = sum(c["A1_vs_B"]["beats"] for c in multi)
    summary = {
        "multilayer_cells_where_A1_beats_C": n_beats,
        "verdict": verdict,
        "room_to_learn": room,
        "reward_misaligned": misaligned,
        "multilayer_cells_where_A1_beats_B": random_floor,
        "register_X2c": verdict == "procedure matters" and room and not misaligned,
        "truths_per_class": n_truths,
    }
    (OUT / "verdict.json").write_text(
        json.dumps({"summary": summary, "cells": cells}, indent=1), encoding="utf-8"
    )
    write_markdown(summary, cells)
    plot(methods, n_truths)
    print(json.dumps(summary, indent=1))


def write_markdown(summary: dict, cells: dict) -> None:
    lines = [
        "# Experiment 002 results (generated by experiments/002_fit_policy/report.py)",
        "",
        "Rules: docs/fit_policy/DESIGN.md section 6. Generated; do not edit by hand.",
        "",
        "## Verdict",
        "",
        *(f"- {k}: {v}" for k, v in summary.items()),
        "",
        "## A1 against C, per cell",
        "",
        "Outcomes are per truth (median over its starts). The last column counts episode-level BIC",
        "wins and how many of them have the same layer count as C.",
        "",
        "| cell | C capped | reference | truth BIC win / tie / loss | sign p | "
        "truth held-out win / loss | beats | episode wins (same L) |",
        "|---|---:|---|---|---:|---|---|---|",
    ]
    for name, c in cells.items():
        b = c["A1_vs_C"]
        lines.append(
            f"| {name} | {c['c_capped_frac']:.2f} | {c['reference']} | "
            f"{b['bic_wins']} / {b['bic_ties']} / {b['bic_losses']} | {b['sign_p']:.2g} | "
            f"{b['ho_wins']} / {b['ho_losses']} | {b['beats']} | "
            f"{b['episode_bic_wins']} ({b['episode_bic_wins_same_L']}) |"
        )
    lines += ["", "## A1 against B (random floor)", ""]
    lines += [
        "| cell | BIC win / tie / loss | sign p | held-out loss | beats |",
        "|---|---|---:|---:|---|",
    ]
    for name, c in cells.items():
        b = c["A1_vs_B"]
        lines.append(
            f"| {name} | {b['bic_wins']} / {b['bic_ties']} / {b['bic_losses']} | "
            f"{b['sign_p']:.2g} | {b['ho_losses']} | {b['beats']} |"
        )
    lines += ["", "## Methods per cell (M3, M4; C_oracle and C_refine are references only)", ""]
    lines += [
        "| cell | method | episodes | layer count right | recovered | median ΔBIC vs C | "
        "median evaluations | median s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, c in cells.items():
        for m, d in c["describe"].items():
            if not d:
                continue
            lines.append(
                f"| {name} | {m} | {d['episodes']} | {d['layer_count_right']:.2f} | "
                f"{d['recovered']:.2f} | {d.get('d_bic_median_vs_C', float('nan')):.1f} | "
                f"{d['evaluations_median']:.0f} | {d['seconds_median']:.1f} |"
            )
    for name, c in cells.items():
        if "learning_room" in c:
            lines += ["", f"Learning room {name}: `{json.dumps(c['learning_room'])}`"]
            lines += [f"Reward alignment {name}: `{json.dumps(c['reward_alignment'])}`"]
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(methods: dict, n_truths: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, n in zip(axes, CLASSES, strict=True):
        for s, colour in zip(range(len(st.NOISE_SCALES)), ("tab:blue", "tab:orange"), strict=True):
            keys = [
                k for k in cell_keys(n, s, n_truths) if k in methods["A1"] and k in methods["C"]
            ]
            d = np.sort([methods["A1"][k]["bic"] - methods["C"][k]["bic"] for k in keys])
            ax.plot(
                d, np.linspace(0, 1, len(d)), color=colour, label=f"S = {st.NOISE_SCALES[s]:.0e}"
            )
        ax.axvspan(-BIC_BAND, BIC_BAND, color="0.9")
        ax.set_xscale("symlog", linthresh=BIC_BAND)
        ax.set_title(f"{n} layer{'s' if n > 1 else ''}")
        ax.set_xlabel("BIC(A1) − BIC(C)")
    axes[0].set_ylabel("fraction of episodes")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "delta_bic.png", dpi=150)


if __name__ == "__main__":
    main()
