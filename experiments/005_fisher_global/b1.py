"""B-1 of docs/fisher_global/PREREG.md: R2a's recorded endpoints against the Fisher quadratic.

Forward evaluations only, at the 2368 endpoints of Experiment 001 R2a (docs/identifiability/scan/)
and at their truths: the deviance of each endpoint against the truth's noise-free counts, its share
per q region, the count level S1 at which it would come within ΔD = 1, and the Fisher quadratic
Δθᵀ F Δθ at the truth. No fit is run and nothing under docs/identifiability/ is written. Verdicts
for H-F and H-W2 follow the registered rules (PREREG sections 3 and 4) in `report`.

Run from the repository root:
    .venv/Scripts/python.exe -I experiments/005_fisher_global/b1.py run
    .venv/Scripts/python.exe -I experiments/005_fisher_global/b1.py report
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
import fim
import scan

SCAN_DIR = ROOT / "docs" / "identifiability" / "scan"
OUT = ROOT / "docs" / "fisher_global"
SEED = 20261025
FLOOR_LEVELS = np.arange(-14, 0)  # log10 of the Fisher deviance of the check-2 displacements
FLOOR_TOL = 0.02
N_DIRECTIONS = 8
REACH = 4.0
W2_CONDITIONS = (6, 7, 8, 15, 16, 17, 24, 25, 26, 33, 34, 35, 36)
REGION_NAMES = ["low"] + [f"b{i}" for i in range(10)]


def deviance_terms(lam: np.ndarray, lam_t: np.ndarray) -> np.ndarray:
    """2 λ_t (r − ln(1 + r)), r = λ/λ_t − 1: the Poisson deviance against noise-free data y = λ_t,
    with the small-r cancellation replaced by its series."""
    r = lam / lam_t - 1
    small = np.abs(r) < 1e-3
    rs = np.where(small, r, 0.0)
    series = rs**2 / 2 - rs**3 / 3 + rs**4 / 4 - rs**5 / 5
    direct = r - np.log1p(np.where(small, 0.0, r))
    return 2 * lam_t * np.where(small, series, direct)


def counts(q: np.ndarray, p: dict[str, float]) -> np.ndarray:
    with torch.no_grad():
        return scan.expected_counts(
            torch.as_tensor(q), {k: torch.tensor(v) for k, v in p.items()}
        ).numpy()


def regions(q: np.ndarray) -> list[np.ndarray]:
    """X2e's split of the 2θ 0.2–6° grid (Task A): 2θ < 0.5°, then ten consecutive blocks."""
    edge = fim.tth_to_q(np.array(0.5))
    above = np.flatnonzero(q >= edge)
    return [np.flatnonzero(q < edge), *np.array_split(above, 10)]


def floor_check(q, p0, free, bounds, fisher, lam_t, rng) -> dict:
    """Check 2: smallest Fisher-deviance level at which the computed ΔD matches the quadratic to
    FLOOR_TOL along random directions, at that level and every level above it."""
    width = np.array([bounds[k][1] - bounds[k][0] for k in free])
    theta_t = np.array([p0[k] for k in free])
    dirs = rng.standard_normal((N_DIRECTIONS, len(free))) * width
    ratios = np.empty((len(FLOOR_LEVELS), N_DIRECTIONS))
    for j, u in enumerate(dirs):
        quad = u @ fisher @ u
        for i, level in enumerate(FLOOR_LEVELS):
            dtheta = u * np.sqrt(10.0**level / quad)
            lam = counts(q, p0 | dict(zip(free, (theta_t + dtheta).tolist(), strict=True)))
            ratios[i, j] = deviance_terms(lam, lam_t).sum() / 10.0**level
    ok = np.all(np.abs(ratios - 1) <= FLOOR_TOL, axis=1)
    floor = None
    for i in range(len(FLOOR_LEVELS)):
        if ok[i:].all():
            floor = float(10.0 ** FLOOR_LEVELS[i])
            break
    return {"floor": floor, "levels": FLOOR_LEVELS.tolist(), "ratios": ratios.tolist()}


def run_condition(path: Path, rng) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cond, p0, free = data["condition"], data["truth"], tuple(data["free"])
    _, _, bounds = scan.setup(cond)
    q = scan.q_grid(cond["tth_max"])
    theta_t = np.array([p0[k] for k in free])
    lam_t, jlog = scan.model(q, p0, free)(theta_t)
    fisher = jlog.T @ (lam_t[:, None] * jlog)
    width = np.array([bounds[k][1] - bounds[k][0] for k in free])
    evals, evecs = np.linalg.eigh(fisher * np.outer(width, width))
    weak = evecs[:, 0]
    regs = regions(q) if len(q) == 291 else None
    check = floor_check(q, p0, free, bounds, fisher, lam_t, rng)

    rows = []
    for e in data["endpoints"]:
        lam = counts(q, e["params"])
        terms = deviance_terms(lam, lam_t)
        dd = float(terms.sum())
        dtheta = np.array([e["params"][k] for k in free]) - theta_t
        d_f = float(dtheta @ fisher @ dtheta)
        u = dtheta / width
        row = {
            "start": e["start"],
            "nfev": e["nfev"],
            "deviance_recorded": e["deviance"],
            "dD": dd,
            "dF": d_f,
            "rho": d_f / dd if dd > 0 else None,
            "S1": scan.S_COUNTS / dd if dd > 0 else None,
            "cos_weak": float(abs(u @ weak) / np.linalg.norm(u)) if np.linalg.norm(u) > 0 else None,
        }
        if regs is not None:
            row["share"] = [float(terms[idx].sum() / dd) if dd > 0 else None for idx in regs]
        rows.append(row)
    return {
        "index": data["index"],
        "condition": cond,
        "free": list(free),
        "n_q": len(q),
        "region_q": (
            [[float(q[idx[0]]), float(q[idx[-1]]), len(idx)] for idx in regs] if regs else None
        ),
        "fisher": fisher.tolist(),
        "fisher_cond_scaled": float(evals[-1] / evals[0]),
        "crlb": np.sqrt(np.diag(np.linalg.inv(fisher))).tolist(),
        "floor_check": check,
        "rows": rows,
    }


def run() -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    out = []
    for path in sorted(SCAN_DIR.glob("fits_*.json")):
        t = time.perf_counter()
        out.append(run_condition(path, rng))
        print(f"{path.name}: {time.perf_counter() - t:.1f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "b1.json").write_text(
        json.dumps({"seconds": time.perf_counter() - t0, "conditions": out}), encoding="utf-8"
    )
    print(f"total {time.perf_counter() - t0:.0f} s")


def grade(frac: float | None, holds: float) -> str:
    if frac is None:
        return "not testable"
    return "holds" if frac >= holds else "fails" if frac < 0.6 else "unresolved"


def combine(*parts: str) -> str:
    if "fails" in parts:
        return "refuted"
    if all(p == "holds" for p in parts):
        return "supported"
    return "unresolved"


def report() -> None:
    res = json.loads((OUT / "b1.json").read_text(encoding="utf-8"))
    conds = res["conditions"]

    check1 = [
        abs(r["dD"] - r["deviance_recorded"]) / r["deviance_recorded"]
        for c in conds
        for r in c["rows"]
        if r["deviance_recorded"] >= 1
    ]
    check1_max = max(check1)
    floors = {c["index"]: c["floor_check"]["floor"] for c in conds}

    near, at_floor, untestable, wrong = [], 0, 0, []
    for c in conds:
        fl = floors[c["index"]]
        for r in c["rows"]:
            if r["dD"] > REACH:
                wrong.append((c, r))
            elif fl is None:
                untestable += 1
            elif r["dD"] < fl:
                at_floor += 1
            else:
                near.append((c, r))

    def frac(items, pred):
        return sum(pred(r) for _, r in items) / len(items) if items else None

    f_i = frac(near, lambda r: 0.5 <= r["rho"] <= 2) if len(near) >= 20 else None
    f_ii = frac(wrong, lambda r: r["rho"] >= 10)
    hf = {
        "check1_max_rel": check1_max,
        "check1_pass": check1_max <= 1e-6,
        "n_near": len(near),
        "n_at_floor": at_floor,
        "n_near_untestable_floor": untestable,
        "n_wrong": len(wrong),
        "frac_i": f_i,
        "frac_ii": f_ii,
        "i": grade(f_i, 0.9),
        "ii": grade(f_ii, 0.8),
    }
    hf["verdict"] = combine(hf["i"], hf["ii"]) if hf["check1_pass"] else "no verdict (check 1)"
    if hf["i"] == "not testable" and hf["verdict"] == "supported":
        hf["verdict"] = "unresolved"

    w2 = [(c, r) for c, r in wrong if c["index"] in W2_CONDITIONS]
    f_a = frac(w2, lambda r: sum(r["share"][:6]) >= 0.9)
    f_b = frac(w2, lambda r: int(np.argmax(r["share"])) in (1, 2, 3))
    hw = {"n": len(w2), "frac_a": f_a, "frac_b": f_b, "a": grade(f_a, 0.8), "b": grade(f_b, 0.8)}
    hw["verdict"] = combine(hw["a"], hw["b"]) if hf["check1_pass"] else "no verdict (check 1)"

    verdict = {"H-F": hf, "H-W2": hw, "floors": floors, "seconds_run": res["seconds"]}
    (OUT / "b1_verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    (OUT / "b1_tables.md").write_text(tables(conds, near, wrong, w2, verdict), encoding="utf-8")
    print(json.dumps({"H-F": hf, "H-W2": hw}, indent=1))


def q3(x) -> str:
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return "—"
    a, b, c = np.percentile(x, [25, 50, 75])
    return f"{b:.3g} [{a:.3g}–{c:.3g}]"


def tables(conds, near, wrong, w2, verdict) -> str:
    hf, hw = verdict["H-F"], verdict["H-W2"]
    lines = [
        "# B-1 tables (generated by b1.py report; do not edit)",
        "",
        (
            "R2a endpoints against the truth's noise-free counts at S = 10⁶. ΔD: Poisson deviance;"
            " ρ = ΔD_F/ΔD with ΔD_F the Fisher quadratic at the truth; S₁ = S/ΔD. Near: D_floor ≤ ΔD ≤ 4;"
            " wrong: ΔD > 4. Rules in PREREG.md sections 3–4."
        ),
        "",
        "## Registered verdicts",
        "",
        (
            f"- Check 1: max |ΔD − recorded| / recorded over recorded ≥ 1: {hf['check1_max_rel']:.2e}"
            f" ({'pass' if hf['check1_pass'] else 'FAIL'})."
        ),
        (
            f"- Near endpoints {hf['n_near']}, at the floor {hf['n_at_floor']}, in conditions without"
            f" a floor {hf['n_near_untestable_floor']}; wrong endpoints {hf['n_wrong']}."
        ),
        (
            f"- H-F (i): {'—' if hf['frac_i'] is None else format(hf['frac_i'], '.3f')} → {hf['i']};"
            f" (ii): {hf['frac_ii']:.3f} → {hf['ii']}. **H-F {hf['verdict']}.**"
        ),
        (
            f"- H-W2 on {hw['n']} endpoints: (a) {hw['frac_a']:.3f} → {hw['a']}; (b) {hw['frac_b']:.3f}"
            f" → {hw['b']}. **H-W2 {hw['verdict']}.**"
        ),
        "",
        "## Per condition",
        "",
        (
            "| # | stack | nuisance | 2θ max | ρ film | D_floor | near | ρ near, median [IQR]"
            " | (i) in [0.5, 2] | wrong | ρ wrong, median [IQR] | (ii) ≥ 10 | S₁ wrong, median [IQR] |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in conds:
        i, cd = c["index"], c["condition"]
        nr = [r for cc, r in near if cc["index"] == i]
        wr = [r for cc, r in wrong if cc["index"] == i]
        fl = c["floor_check"]["floor"]
        fi = f"{np.mean([0.5 <= r['rho'] <= 2 for r in nr]):.2f}" if nr else "—"
        fii = f"{np.mean([r['rho'] >= 10 for r in wr]):.2f}" if wr else "—"
        lines.append(
            f"| {i} | {cd['stack']} | {cd['nuisance']} | {cd['tth_max']:g}° | {cd['film_sld']:g}"
            f" | {'—' if fl is None else format(fl, '.0e')} | {len(nr)} | {q3(r['rho'] for r in nr)} | {fi}"
            f" | {len(wr)} | {q3(r['rho'] for r in wr)} | {fii} | {q3(r['S1'] for r in wr)} |"
        )

    lines += [
        "",
        "## Wrong endpoints by ΔD decade (descriptive)",
        "",
        "| ΔD | n | ρ, median [IQR] | share ρ ≥ 10 | \\|cos\\| to weakest, median |",
        "|---|---|---|---|---|",
    ]
    for lo in range(9):
        sel = [r for _, r in wrong if 10.0**lo < r["dD"] <= 10.0 ** (lo + 1) and r["dD"] > REACH]
        if sel:
            lines.append(
                f"| 10^{lo}–10^{lo + 1} | {len(sel)} | {q3(r['rho'] for r in sel)}"
                f" | {np.mean([r['rho'] >= 10 for r in sel]):.2f}"
                f" | {np.median([r['cos_weak'] for r in sel]):.2f} |"
            )
    sel = [r for _, r in near]
    lines.append(
        f"| near | {len(sel)} | {q3(r['rho'] for r in sel)} | —"
        f" | {np.median([r['cos_weak'] for r in sel]) if sel else float('nan'):.2f} |"
    )

    sub = [(c, r) for c, r in wrong if r["dD"] <= 1e4]
    lines += [
        "",
        (
            f"W′ (4 < ΔD ≤ 10⁴, S₁ ≥ 100): {len(sub)} endpoints, ρ {q3(r['rho'] for _, r in sub)},"
            f" share ρ ≥ 10 {np.mean([r['rho'] >= 10 for _, r in sub]) if sub else float('nan'):.2f}."
        ),
        "",
        "## H-W2 test set: deviance share per q region (descriptive)",
        "",
    ]
    rq = next(c["region_q"] for c in conds if c["region_q"])
    lines.append("| | " + " | ".join(REGION_NAMES) + " |")
    lines.append("|---|" + "---|" * len(REGION_NAMES))
    lines.append("| q (Å⁻¹) | " + " | ".join(f"{a:.3f}–{b:.3f}" for a, b, _ in rq) + " |")
    shares = np.array([r["share"] for _, r in w2])
    lines.append("| median share | " + " | ".join(f"{v:.3f}" for v in np.median(shares, 0)) + " |")
    top = np.bincount(np.argmax(shares, 1), minlength=len(REGION_NAMES))
    lines.append("| largest in | " + " | ".join(str(v) for v in top) + " |")
    lines += [
        "",
        "| film ρ | stack | n | (a) q ≤ 0.235 ≥ 0.9 | (b) largest in b0–b2 | largest in low |",
        "|---|---|---|---|---|---|",
    ]
    for rho in (12.0, 24.0, 45.0):
        for stack in ("single", "bilayer"):
            sel = [
                r
                for c, r in w2
                if c["condition"]["film_sld"] == rho and c["condition"]["stack"] == stack
            ]
            if not sel:
                continue
            s = np.array([r["share"] for r in sel])
            lines.append(
                f"| {rho:g} | {stack} | {len(sel)} | {np.mean(s[:, :6].sum(1) >= 0.9):.2f}"
                f" | {np.mean(np.isin(np.argmax(s, 1), (1, 2, 3))):.2f}"
                f" | {np.mean(np.argmax(s, 1) == 0):.2f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["run", "report"])
    args = ap.parse_args()
    if args.part == "run":
        run()
    else:
        report()


if __name__ == "__main__":
    main()
