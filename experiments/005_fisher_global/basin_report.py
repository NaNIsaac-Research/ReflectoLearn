"""Verdicts and tables of the full basin run (PREREG-BASIN.md sections 4 and 5).

Reads docs/fisher_global/basin_full/*.json (written by basin.py --set full) and writes
basin_verdict.json and basin_full.md next to them. Committed before the full run started; the rules
below are the registered ones, and the choices the registration leaves open are fixed here, in
advance, and listed in OPEN_CHOICES.

    .venv/Scripts/python.exe -I experiments/005_fisher_global/basin_report.py
    .venv/Scripts/python.exe -I experiments/005_fisher_global/basin_report.py --compare-pilot
    .venv/Scripts/python.exe -I experiments/005_fisher_global/basin_report.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "docs" / "fisher_global"
FULL_DIR = OUT / "basin_full"
REGIONS = ["low"] + [f"b{i}" for i in range(10)]
LOW_Q = slice(0, 6)  # low, b0 ... b4: q ≤ 0.235 Å⁻¹
TOP_LOW = {1, 2, 3}  # b0, b1, b2
REGISTERED = {"r2a": 749, "x2e": 93}

START_TOL = 1e-6
REACH = 4.0
KAPPA_GOOD, KAPPA_MISS = 2.0, 10.0
SUPPORT_GOOD, SUPPORT_MISS, REFUTE_MISS = 0.80, 0.05, 0.20
HOLD, FAIL = 0.80, 0.60
MIN_UNIT = 20
SAME_BASIN = 1.0
FAR_U = 1e-2
S1_REALISTIC = 1e4

OPEN_CHOICES = [
    "H-BM1 'half the source's candidates' is half the registered count (749, 93), whatever the"
    " start check removes.",
    "H-BM1 sensitivity adds every unconfirmed O_nf candidate (budget, stalled or numerical) whose"
    " κ at its last point exceeds 10; the counts by outcome are listed.",
    "Hypotheses use O_nf refinements only; O_tr (X2e) is descriptive, as registered in 3.1.",
    "H-BM3: if no statement fails and at least one is not testable, the overall result is"
    " 'unresolved' with the untestable statements named.",
    "Distance in u between two minima uses the mean of their bound widths (equal within a data set"
    " unless the box depends on the state).",
    "A candidate whose run raised an exception counts as unconfirmed (numerical) for the denominator"
    " and is listed.",
]


def curve_dev(lam: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance of counts lam against noise-free data mu (b1.deviance_terms)."""
    r = lam / mu - 1
    small = np.abs(r) < 1e-3
    rs = np.where(small, r, 0.0)
    series = rs**2 / 2 - rs**3 / 3 + rs**4 / 4 - rs**5 / 5
    direct = r - np.log1p(np.where(small, 0.0, r))
    return float((2 * mu * np.where(small, series, direct)).sum())


def frac(n: int, d: int) -> float | None:
    return n / d if d else None


def load() -> tuple[list[dict], list[dict]]:
    rows = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(FULL_DIR.glob("*__*.json"))
        if not p.name.endswith(".error.json")
    ]
    errors = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(FULL_DIR.glob("*.error.json"))
    ]
    return rows, errors


def annotate(rows: list[dict]) -> None:
    cand = json.loads((OUT / "basin_candidates.json").read_text(encoding="utf-8"))
    b1 = json.loads((OUT / "b1.json").read_text(encoding="utf-8"))
    conds = {c["index"]: c["condition"] for c in b1["conditions"]}
    capped = {c["id"]: c["capped"] for c in cand["r2a"]}
    pilot = set(cand["pilot"])
    for r in rows:
        parts = r["id"].split(":")
        r["source"] = parts[0]
        r["pilot"] = r["id"] in pilot
        r["start_ok"] = r["start_check_rel"] <= START_TOL
        r["on_bound"] = r["final_checks"]["at_lo"] + r["final_checks"]["at_hi"]
        if r["source"] == "r2a":
            c = int(parts[1])
            r["dataset"] = f"r2a:{c:02d}"
            r["meta"] = {"cond": c, **conds[c], "capped": capped[r["id"]]}
        else:
            arm, n, j, s = parts[1], int(parts[2]), int(parts[3]), int(parts[4])
            r["dataset"] = f"x2e:{n}:{j}:{s}"
            r["meta"] = {"arm": arm, "n": n, "j": j, "s": s}


def find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def group_basins(minima: list[dict]) -> list[dict]:
    """Section 4: connected components of max(ΔD(a|b), ΔD(b|a)) ≤ 1 within a data set."""
    by_set = defaultdict(list)
    for r in minima:
        by_set[r["dataset"]].append(r)
    basins = []
    for ds, rs in sorted(by_set.items()):
        lam = [np.asarray(r["lam_min"]) for r in rs]
        parent = list(range(len(rs)))
        far_pairs = []
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                if max(curve_dev(lam[a], lam[b]), curve_dev(lam[b], lam[a])) <= SAME_BASIN:
                    parent[find(parent, a)] = find(parent, b)
                    du = u_dist(rs[a], rs[b])
                    if du > FAR_U:
                        far_pairs.append((rs[a]["id"], rs[b]["id"], du))
        comps = defaultdict(list)
        for i in range(len(rs)):
            comps[find(parent, i)].append(rs[i])
        for members in comps.values():
            rep = min(members, key=lambda r: (r["dnf_min"], r["id"]))
            spread = max(
                (u_dist(a, b) for a in members for b in members if a["id"] < b["id"]), default=0.0
            )
            basins.append(
                {
                    "dataset": ds,
                    "rep": rep,
                    "members": [m["id"] for m in members],
                    "max_u": spread,
                    "far_pairs": [p for p in far_pairs if p[0] in {m["id"] for m in members}],
                }
            )
    return basins


def u_dist(a: dict, b: dict) -> float:
    w = (np.subtract(a["hi"], a["lo"]) + np.subtract(b["hi"], b["lo"])) / 2
    return float(np.max(np.abs(np.subtract(a["x_min"], b["x_min"])) / w))


def h_bm1(confirmed: list[dict], n_registered: int, extra_misses: int = 0) -> dict:
    """extra_misses: the sensitivity's unconfirmed candidates, counted in E and as misses."""
    e = len(confirmed) + extra_misses
    miss = sum(r["truth_basin"] or r["kappa"] > KAPPA_MISS for r in confirmed) + extra_misses
    good = sum((not r["truth_basin"]) and r["kappa"] <= KAPPA_GOOD for r in confirmed)
    out = {
        "E": e,
        "registered": n_registered,
        "good": good,
        "miss": miss,
        "truth_basin": sum(r["truth_basin"] for r in confirmed),
        "kappa_gt_10": sum((not r["truth_basin"]) and r["kappa"] > KAPPA_MISS for r in confirmed),
        "good_frac": frac(good, e),
        "miss_frac": frac(miss, e),
    }
    if e < MIN_UNIT or e < n_registered / 2:
        out["verdict"] = "not testable"
    elif good / e >= SUPPORT_GOOD and miss / e <= SUPPORT_MISS:
        out["verdict"] = "supported"
    elif miss / e >= REFUTE_MISS:
        out["verdict"] = "refuted"
    else:
        out["verdict"] = "unresolved"
    return out


def h_bm1_all(rows: list[dict], source: str, n_registered: int) -> dict:
    nf = [r for r in rows if r["source"] == source and r["objective"] == "nf" and r["start_ok"]]
    conf = [r for r in nf if r["outcome"] == "confirmed"]
    unconf = [r for r in nf if r["outcome"] != "confirmed"]
    main = h_bm1(conf, n_registered)
    added = [r for r in unconf if r["kappa"] is None or r["kappa"] > KAPPA_MISS]
    sens = h_bm1(conf, n_registered, len(added))
    sens["added_by_outcome"] = dict(Counter(r["outcome"] for r in added))
    no_pilot = h_bm1([r for r in conf if not r["pilot"]], n_registered)
    interior = h_bm1([r for r in conf if not r["on_bound"]], n_registered)
    return {
        "registered": main,
        "sensitivity_unconfirmed_drop": sens,
        "without_pilot": no_pilot,
        "descriptive_interior_only": interior | {"note": "descriptive, not a verdict"},
    }


def statement(hits: int, n: int) -> dict:
    if n < MIN_UNIT:
        result = "not testable"
    elif hits / n >= HOLD:
        result = "holds"
    elif hits / n < FAIL:
        result = "fails"
    else:
        result = "unresolved"
    return {"hits": hits, "n": n, "frac": frac(hits, n), "result": result}


def h_bm3(basins: list[dict]) -> dict:
    reps = [b["rep"] for b in basins]
    grid = [r for r in reps if r.get("share_min") is not None]
    a = statement(sum(r["rho_min"] is not None and r["rho_min"] >= 10 for r in reps), len(reps))
    b = statement(sum(sum(r["share_min"][LOW_Q]) >= 0.9 for r in grid), len(grid))
    c = statement(sum(int(np.argmax(r["share_min"])) in TOP_LOW for r in grid), len(grid))
    parts = {"a": a, "b": b, "c": c}
    results = [p["result"] for p in parts.values()]
    if "fails" in results:
        overall = "does not persist"
    elif all(x == "holds" for x in results):
        overall = "persists"
    else:
        untestable = [k for k, p in parts.items() if p["result"] == "not testable"]
        overall = "unresolved" + (f" (not testable: {', '.join(untestable)})" if untestable else "")
    return parts | {"overall": overall}


def verdicts(rows: list[dict]) -> tuple[dict, dict[str, list[dict]]]:
    out, all_basins = {}, {}
    for source, n_reg in REGISTERED.items():
        src_rows = [r for r in rows if r["source"] == source]
        minima = [
            r
            for r in src_rows
            if r["objective"] == "nf"
            and r["start_ok"]
            and r["outcome"] == "confirmed"
            and not r["truth_basin"]
        ]
        basins = group_basins(minima)
        all_basins[source] = basins
        realistic = [b for b in basins if b["rep"]["S1_min"] >= S1_REALISTIC]
        out[source] = {
            "H-BM1": h_bm1_all(rows, source, n_reg),
            "Q-BM2": {
                "count": len(realistic),
                "basins": [
                    {
                        "dataset": b["dataset"],
                        "rep": b["rep"]["id"],
                        "S1_min": b["rep"]["S1_min"],
                        "meta": b["rep"]["meta"],
                        "members": len(b["members"]),
                    }
                    for b in realistic
                ],
            },
            "H-BM3": h_bm3(basins),
            "basins": {
                "wrong_minima": len(minima),
                "distinct": len(basins),
                "far_pairs": sum(len(b["far_pairs"]) for b in basins),
            },
        }
    return out, all_basins


def q(xs, p=(0, 0.1, 0.5, 0.9, 1)) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return " / ".join(format(v, ".3g") for v in np.quantile(xs, p))


def table(head: list[str], body: list[list]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(str(x) for x in r) + " |" for r in body),
    ]


def breakdown(rows: list[dict], basins: list[dict], key) -> list[list]:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    reps_by = defaultdict(int)
    for b in basins:
        reps_by[key(b["rep"])] += 1
    body = []
    for k in sorted(groups, key=str):
        rs = groups[k]
        conf = [r for r in rs if r["outcome"] == "confirmed"]
        body.append(
            [
                k,
                len(rs),
                len(conf),
                sum(r["outcome"] == "unconfirmed: budget" for r in rs),
                sum(r["truth_basin"] for r in conf),
                sum((not r["truth_basin"]) and r["kappa"] > KAPPA_MISS for r in conf),
                q([r["kappa"] for r in conf], (0.5, 0.9, 1)),
                q([r["S1_end"] for r in conf], (0.5, 1)),
                q([r["S1_min"] for r in conf if not r["truth_basin"]], (0.5, 1)),
                reps_by[k],
                sum(bool(r["on_bound"]) for r in conf),
            ]
        )
    return body


BREAK_HEAD = [
    "group",
    "n",
    "confirmed",
    "budget",
    "truth basin",
    "κ > 10 (wrong)",
    "κ median / p90 / max",
    "S₁ end median / max",
    "S₁* wrong median / max",
    "distinct wrong basins",
    "on bound",
]


def report(rows: list[dict], errors: list[dict], v: dict, basins: dict) -> list[str]:
    lines = [
        "# Basin full-run tables (generated by basin_report.py; do not edit)",
        "",
        "PREREG-BASIN.md. ΔD_nf against the truth's noise-free counts; κ = ΔD_nf(end) / ΔD_nf(min);"
        " S₁ = S / ΔD_nf; ρ = ΔD_F / ΔD_nf. Quantiles are min / p10 / median / p90 / max unless the"
        " header says otherwise.",
        "",
        "## Choices fixed before the run where the registration is silent",
        "",
        *(f"- {c}" for c in OPEN_CHOICES),
        "",
        "## Outcomes",
        "",
    ]
    body = []
    for src, obj in sorted({(r["source"], r["objective"]) for r in rows}):
        rs = [r for r in rows if r["source"] == src and r["objective"] == obj]
        oc = Counter(r["outcome"] for r in rs)
        n_err = sum(e["id"].startswith(src) and e["objective"] == obj for e in errors)
        conf = [r for r in rs if r["outcome"] == "confirmed"]
        body.append(
            [
                f"{src} {obj}",
                len(rs) + n_err,
                f"{oc['confirmed']} ({oc['confirmed'] / max(len(rs) + n_err, 1):.1%})",
                sum(not r["on_bound"] for r in conf),
                sum(bool(r["on_bound"]) for r in conf),
                oc["unconfirmed: budget"],
                oc["unconfirmed: stalled"],
                oc["unconfirmed: numerical"] + n_err,
                sum(not r["start_ok"] for r in rs),
            ]
        )
    lines += table(
        [
            "set",
            "n",
            "confirmed",
            "interior",
            "on bound",
            "budget-limited",
            "stalled",
            "numerical / error",
            "start check failed",
        ],
        body,
    )
    if errors:
        lines += ["", "Errors:", *(f"- {e['id']} {e['objective']}: {e['error']}" for e in errors)]

    lines += ["", "## Verdicts (registered rules)", ""]
    for src in REGISTERED:
        s = v[src]
        m = s["H-BM1"]["registered"]
        lines += [
            f"### {src}",
            "",
            f"- H-BM1: **{m['verdict']}** — E = {m['E']} of {m['registered']} registered;"
            f" κ ≤ 2 and not truth basin {m['good']} ({g(m['good_frac'], '.1%')}); misses"
            f" {m['miss']} ({g(m['miss_frac'], '.1%')}: truth basin {m['truth_basin']},"
            f" κ > 10 {m['kappa_gt_10']}).",
        ]
        for tag, label in (
            ("sensitivity_unconfirmed_drop", "sensitivity (unconfirmed with κ > 10 added)"),
            ("without_pilot", "without the eight pilot candidates"),
            ("descriptive_interior_only", "interior minima only (descriptive)"),
        ):
            x = s["H-BM1"][tag]
            extra = f", added {x['added_by_outcome']}" if "added_by_outcome" in x else ""
            lines.append(
                f"  - {label}: {x['verdict']} (E {x['E']}, good {g(x['good_frac'], '.1%')},"
                f" misses {g(x['miss_frac'], '.1%')}{extra})"
            )
        qb = s["Q-BM2"]
        lines.append(f"- Q-BM2: **{qb['count']}** distinct wrong basins with S₁* ≥ 10⁴")
        lines += [
            f"  - {b['dataset']} {b['rep']} S₁* {b['S1_min']:.3g} {b['meta']}" for b in qb["basins"]
        ]
        h = s["H-BM3"]
        lines.append(f"- H-BM3: **{h['overall']}**")
        for k, label in (("a", "ρ* ≥ 10"), ("b", "low-q share ≥ 0.9"), ("c", "top region b0–b2")):
            p = h[k]
            lines.append(
                f"  - ({k}) {label}: {p['result']} ({p['hits']}/{p['n']}, {g(p['frac'], '.1%')})"
            )
        bs = s["basins"]
        lines += [
            f"- Basins: {bs['wrong_minima']} confirmed wrong minima → {bs['distinct']} distinct;"
            f" curve-equivalent pairs > 10⁻² apart in u: {bs['far_pairs']}",
            "",
        ]

    for src in REGISTERED:
        nf = [r for r in rows if r["source"] == src and r["objective"] == "nf" and r["start_ok"]]
        conf = [r for r in nf if r["outcome"] == "confirmed"]
        wrong = [r for r in conf if not r["truth_basin"]]
        lines += [
            f"## {src}: endpoint → minimum (confirmed O_nf)",
            "",
            *table(
                ["quantity", "quantiles"],
                [
                    ["κ", q([r["kappa"] for r in conf])],
                    ["ΔD_nf drop fraction 1 − 1/κ", q([1 - 1 / r["kappa"] for r in wrong])],
                    ["S₁ endpoint", q([r["S1_end"] for r in conf])],
                    ["S₁* minimum (wrong basins)", q([r["S1_min"] for r in wrong])],
                    ["drift (u)", q([r["drift_inf"] for r in conf])],
                    ["evaluations", q([r["nfev_total"] for r in conf])],
                    ["ρ endpoint", q([r.get("rho_end") for r in conf])],
                    ["ρ* minimum", q([r.get("rho_min") for r in conf])],
                    [
                        "ρ* / ρ endpoint",
                        q([r["rho_min"] / r["rho_end"] for r in wrong if r.get("rho_end")]),
                    ],
                    [
                        "log10 ΔD_F − log10 ΔD_nf at minimum",
                        q([np.log10(r["dF_min"] / r["dnf_min"]) for r in wrong if r.get("dF_min")]),
                    ],
                ],
            ),
            "",
        ]
        bound = Counter(n for r in conf for n in r["final_checks"]["at_lo"])
        bound_hi = Counter(n for r in conf for n in r["final_checks"]["at_hi"])
        names = sorted(set(bound) | set(bound_hi), key=lambda n: -(bound[n] + bound_hi[n]))
        lines += [
            f"### {src}: parameters on a bound at confirmed minima ({len(conf)} minima,"
            f" {sum(not r['on_bound'] for r in conf)} interior)",
            "",
            *table(
                ["parameter", "at lower", "at upper"], [[n, bound[n], bound_hi[n]] for n in names]
            ),
            "",
            "Number of parameters on a bound per minimum: "
            + ", ".join(
                f"{k}: {c}" for k, c in sorted(Counter(len(r["on_bound"]) for r in conf).items())
            ),
            "",
        ]
        b = basins[src]
        lines += [
            f"### {src}: duplicates",
            "",
            f"- endpoints per basin: {q([len(x['members']) for x in b])}",
            f"- max u-distance inside a basin: {q([x['max_u'] for x in b])}",
            f"- H-BM1 on basin representatives only (descriptive): {g(h_bm1([x['rep'] for x in b], 0)['good_frac'], '.1%')} κ ≤ 2",
            "",
        ]

    r2a = [r for r in rows if r["source"] == "r2a" and r["objective"] == "nf" and r["start_ok"]]
    x2e = [r for r in rows if r["source"] == "x2e" and r["objective"] == "nf" and r["start_ok"]]
    lines += ["## Breakdown by condition (O_nf)", ""]
    for label, key in (
        ("R2a by 2θ_max (q_max)", lambda r: r["meta"]["tth_max"]),
        ("R2a by stack (layer count)", lambda r: r["meta"]["stack"]),
        ("R2a by film SLD (contrast)", lambda r: r["meta"]["film_sld"]),
        ("R2a by nuisance set", lambda r: r["meta"]["nuisance"]),
        ("R2a by capped endpoint", lambda r: r["meta"]["capped"]),
        ("R2a by condition", lambda r: r["meta"]["cond"]),
    ):
        lines += [f"### {label}", "", *table(BREAK_HEAD, breakdown(r2a, basins["r2a"], key)), ""]
    for label, key in (
        ("X2e by layer count", lambda r: r["meta"]["n"]),
        ("X2e by count scale s", lambda r: r["meta"]["s"]),
        ("X2e by arm", lambda r: r["meta"]["arm"]),
    ):
        lines += [f"### {label}", "", *table(BREAK_HEAD, breakdown(x2e, basins["x2e"], key)), ""]

    tr = {r["id"]: r for r in rows if r["objective"] == "tr"}
    nf = {r["id"]: r for r in x2e}
    pairs = [(nf[i], tr[i]) for i in tr if i in nf]
    lines += [
        "## X2e O_tr (descriptive)",
        "",
        f"- outcomes: {dict(Counter(r['outcome'] for r in tr.values()))}",
        f"- κ under O_tr (ΔD_nf end / ΔD_nf at the O_tr minimum): {q([r['kappa'] for r in tr.values()])}",
        f"- drift (u) under O_tr: {q([r['drift_inf'] for r in tr.values()])}",
        f"- u-distance O_tr minimum ↔ O_nf minimum: {q([u_dist(a, b) for a, b in pairs])}",
        "",
        "## Compute",
        "",
    ]
    body = []
    for src, obj in (("r2a", "nf"), ("x2e", "nf"), ("x2e", "tr")):
        rs = [r for r in rows if r["source"] == src and r["objective"] == obj]
        if not rs:
            continue
        body.append(
            [
                f"{src} {obj}",
                len(rs),
                f"{sum(r['seconds'] for r in rs) / 3600:.2f}",
                q([r["seconds"] for r in rs], (0.5, 0.9, 1)),
                sum(r["nfev_total"] for r in rs),
                q([r["nfev_total"] for r in rs], (0.5, 0.9, 1)),
                sum(r["model_calls"] for r in rs),
                sum(r["value_calls"] for r in rs),
                sum(r["hessian_calls"] for r in rs),
                f"{max(r['peak_mb'] for r in rs):.0f}",
            ]
        )
    lines += table(
        [
            "set",
            "n",
            "CPU h",
            "s per refinement median / p90 / max",
            "evaluations",
            "evaluations median / p90 / max",
            "model calls",
            "value calls",
            "Hessian calls",
            "peak MB (process)",
        ],
        body,
    )
    return lines


def g(x, fmt=".3g") -> str:
    return "—" if x is None else format(x, fmt)


PILOT_KEYS = ("outcome", "nfev_total", "d_final", "dnf_end", "dnf_min", "kappa", "drift_inf")


def compare_pilot() -> None:
    pilot = {
        (r["id"], r["objective"]): r
        for r in map(json.loads, (OUT / "basin_pilot.jsonl").read_text().splitlines())
    }
    print("| id | obj | " + " | ".join(PILOT_KEYS) + " | max |Δx_min| | identical |")
    print("|---|---|" + "---|" * (len(PILOT_KEYS) + 2))
    for (cid, obj), p in pilot.items():
        path = FULL_DIR / f"{cid.replace(':', '_')}__{obj}.json"
        if not path.exists():
            print(f"| {cid} | {obj} | not run |")
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        cells, same = [], True
        for k in PILOT_KEYS:
            eq = r[k] == p[k]
            same &= eq
            cells.append(f"{r[k]}" if eq else f"{p[k]} → {r[k]}")
        dx = float(np.max(np.abs(np.subtract(r["x_min"], p["x_min"]))))
        stages_eq = [(s["nfev"], s["status"], s["d_after"], s["moved"]) for s in r["stages"]] == [
            (s["nfev"], s["status"], s["d_after"], s["moved"]) for s in p["stages"]
        ]
        same &= dx == 0 and stages_eq and r["lam_min"] == p["lam_min"]
        print(f"| {cid} | {obj} | " + " | ".join(cells) + f" | {dx:.3g} | {same} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compare-pilot", action="store_true", help="pilot rows vs full-run rows")
    ap.add_argument("--dry-run", action="store_true", help="print the tables, write nothing")
    args = ap.parse_args()
    if args.compare_pilot:
        compare_pilot()
        return
    rows, errors = load()
    annotate(rows)
    v, basins = verdicts(rows)
    lines = report(rows, errors, v, basins)
    if args.dry_run:
        print("\n".join(lines))
        return
    meta = {
        "rows": len(rows),
        "errors": errors,
        "open_choices": OPEN_CHOICES,
        "verdicts": v,
        "basins": {
            src: [
                {
                    "dataset": b["dataset"],
                    "rep": b["rep"]["id"],
                    "members": b["members"],
                    "S1_min": b["rep"]["S1_min"],
                    "max_u": b["max_u"],
                    "far_pairs": b["far_pairs"],
                }
                for b in bs
            ]
            for src, bs in basins.items()
        },
    }
    (OUT / "basin_verdict.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (OUT / "basin_full.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'basin_verdict.json'} and {OUT / 'basin_full.md'}")


if __name__ == "__main__":
    main()
