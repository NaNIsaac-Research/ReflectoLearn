"""G1 follow-ups on the recorded fits, no new fit.

Run from the repository root:

    <venv python> -I experiments/005_fisher_global/g1_followup.py steps

`steps` plots the Pearson residuals of the 100 repeat fits around the attenuator changes and writes
`docs/fisher_global/g1/followup/attenuator_steps.{png,json}`. `families` recomputes H-G1c within each
q-offset family of the repeats and writes `docs/fisher_global/g1/followup/hg1c_families.{md,json}`.
`all` runs both.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_run as g1

OUT = g1.OUT / "followup"
WINDOW = 20
EDGE = 8
OFF_LO = g1.M1.lo[g1.M1.names.index("off")]
PINNED = ("off", "sigq")


def load_repeats() -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((g1.OUT / "repeats").glob("r[0-9][0-9][0-9].json"))
    ]


def pearson(data: dict, reps: list[dict]) -> np.ndarray:
    return np.array(
        [(data["raw"][r["scan"]] - np.array(r["lam"])) / np.sqrt(np.array(r["lam"])) for r in reps]
    )


def off_family(reps: list[dict]) -> np.ndarray:
    i = g1.M1.names.index("off")
    return np.array([abs(r["x"][i] - OFF_LO) < 1e-9 for r in reps])


def steps() -> None:
    data = g1.load_scans()
    reps = load_repeats()
    res = pearson(data, reps)
    q, af = data["q"], data["af"]
    # af drifts by about 1e-6 per point; an attenuator switch is a jump of several times, and the
    # first point of the new setting is the step index
    idx = [int(i) + 1 for i in np.flatnonzero(np.abs(np.log(af[1:] / af[:-1])) > np.log(1.1))]
    # a switch spans two or three points; each run of consecutive changes is one step
    switches: list[list[int]] = []
    for s in idx:
        if switches and s == switches[-1][-1] + 1:
            switches[-1].append(s)
        else:
            switches.append([s])
    fam = off_family(reps)

    def step(a: int, b: int) -> np.ndarray:
        # straight lines through EDGE points on each side, compared at the switch; a line follows the
        # fringe-period oscillation of the misfit far better than a window mean does
        xa = np.arange(a - EDGE, a) - a
        xb = np.arange(b, b + EDGE) - a
        before = np.polyfit(xa, res[:, a - EDGE : a].T, 1)
        after = np.polyfit(xb, res[:, b : b + EDGE].T, 1)
        return after[1] - before[1]

    # the same statistic at points far from any switch measures what the smooth misfit alone gives
    far = [
        c
        for c in range(WINDOW, len(q) - WINDOW)
        if all(c < g[0] - WINDOW - 2 or c > g[-1] + WINDOW + 2 for g in switches)
    ]
    control = np.array([step(c, c).mean() for c in far])
    rows = []
    for g in switches:
        a, b = g[0], g[-1]
        per_rep = step(a, b)
        m = float(per_rep.mean())
        rows.append(
            {
                "points": g,
                "q": float(q[a]),
                "af_ratio": float(af[a - 1] / af[b]),
                "step_mean": m,
                "step_sem": float(per_rep.std(ddof=1) / np.sqrt(len(per_rep))),
                "step_mean_off_bound": float(per_rep[fam].mean()),
                "step_mean_inside": float(per_rep[~fam].mean()) if (~fam).any() else None,
                "control_fraction_at_least_as_large": float(np.mean(np.abs(control) >= abs(m))),
                "lam_before": float(np.median([r["lam"][a - 1] for r in reps])),
                "lam_after": float(np.median([r["lam"][b] for r in reps])),
            }
        )
    idx = [g[0] for g in switches]
    mean = res.mean(axis=0)
    sem = res.std(axis=0, ddof=1) / np.sqrt(len(res))
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    a.fill_between(q, mean - 2 * sem, mean + 2 * sem, color="0.8", lw=0)
    a.plot(q, mean, lw=0.8, color="k")
    a.set_ylabel("mean Pearson residual\n(100 repeat fits)")
    for label, m, c in (("q offset at bound", fam, "C0"), ("q offset inside", ~fam, "C1")):
        if m.any():
            b.plot(q, res[m].mean(axis=0), lw=0.8, color=c, label=f"{label} ({m.sum()})")
    b.set_ylabel("mean Pearson residual")
    b.set_xlabel("q (1/Å)")
    b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.axhline(0, color="0.5", lw=0.5)
        for s in idx:
            ax.axvline(q[s], color="C3", lw=0.6, ls="--")
    a.set_title(
        "G1 M1 residuals; dashed: attenuator changes at points " + ", ".join(map(str, idx)),
        fontsize=9,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "attenuator_steps.png", dpi=200)
    rec = {
        "edge_points": EDGE,
        "n_repeats": len(reps),
        "n_off_bound": int(fam.sum()),
        "control_points": len(far),
        "control_abs_step_p50_p95_max": [
            float(np.percentile(np.abs(control), 50)),
            float(np.percentile(np.abs(control), 95)),
            float(np.abs(control).max()),
        ],
        "steps": rows,
    }
    (OUT / "attenuator_steps.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))


def crb(rep: dict, phi: float, drop: tuple[str, ...]) -> dict[str, float]:
    names = list(g1.M1.names)
    keep = [i for i, n in enumerate(names) if n not in drop]
    j = np.array(rep["jlog"])[:, keep]
    lam = np.array(rep["lam"])
    cov = np.linalg.inv(j.T @ (lam[:, None] * j))
    return {names[k]: float(np.sqrt(phi * cov[n, n])) for n, k in enumerate(keep)}


def families() -> None:
    data = g1.load_scans()
    reps = load_repeats()
    res = pearson(data, reps)
    names = list(g1.M1.names)
    xs = np.array([r["x"] for r in reps])
    fam = off_family(reps)
    groups = {"pooled": np.ones(len(reps), bool), "off at lower bound": fam, "off inside": ~fam}
    lines = [
        "# G1 H-G1c within the q-offset families",
        "",
        "Generated by `experiments/005_fisher_global/g1_followup.py families` from the recorded repeat fits.",
        f"A repeat is in the bound family when its q offset is at the lower bound {OFF_LO:g} 1/Å.",
        "phi is the median over q points of the variance of the Pearson residuals within the group.",
        "The CRB is the median over the group's repeats of sqrt(phi * inverse Fisher), with all 11",
        "parameters free, and with the q offset and sigma_q fixed (dropped from the Fisher matrix).",
        "The registered rule (holds <= 2, fails > 3) was written for the pooled sd; the family rows",
        "are a follow-up, not a registered test.",
        "",
        "| group | repeats | phi | parameter | sd (Å) | CRB all free (Å) | ratio | CRB off, sigq fixed (Å) | ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rec = {}
    for g, m in groups.items():
        n = int(m.sum())
        if n < 3:
            lines.append(f"| {g} | {n} | | | | | | | |")
            continue
        phi = float(np.median(np.var(res[m], axis=0, ddof=1)))
        sel = [r for r, k in zip(reps, m, strict=True) if k]
        free = [crb(r, phi, ()) for r in sel]
        fixed = [crb(r, phi, PINNED) for r in sel]
        rec[g] = {"n": n, "phi": phi}
        for key in ("Au_d", "Cr_d"):
            i = names.index(key)
            sd = float(np.std(xs[m, i], ddof=1))
            c1 = float(np.median([c[key] for c in free]))
            c2 = float(np.median([c[key] for c in fixed]))
            rec[g][key] = {
                "sd": sd,
                "crb_free": c1,
                "crb_fixed": c2,
                "median": float(np.median(xs[m, i])),
            }
            lines.append(
                f"| {g} | {n} | {phi:.3f} | {key} | {sd:.3g} | {c1:.3g} | {sd / c1:.2f} | {c2:.3g} | {sd / c2:.2f} |"
            )
        rec[g]["Au_rho_median"] = float(np.median(xs[m, names.index("Au_rho")]))
    lines += [
        "",
        "Median Au SLD (1e-6/Å²) per group, bulk 126.44: "
        + ", ".join(f"{g} {v['Au_rho_median']:.1f}" for g, v in rec.items()),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hg1c_families.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "hg1c_families.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["steps", "families", "all"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.mode in ("steps", "all"):
        steps()
    if args.mode in ("families", "all"):
        families()


if __name__ == "__main__":
    main()
