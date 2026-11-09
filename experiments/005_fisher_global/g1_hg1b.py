"""H-G1b with the nuisances profiled out and a threshold from a null distribution, no new structural fit.

Run from the repository root:

    <venv python> -I experiments/005_fisher_global/g1_hg1b.py profile [--null 200] [--hours H]
    <venv python> -I experiments/005_fisher_global/g1_hg1b.py report

With --variant wide the same score runs on the fits and bounds of declared deviation 1 (g1_wide.py),
written to docs/fisher_global/g1/wide/.

For a target scan a and a source fit b, the score is the noise-free Poisson deviance of b's structure
against a's best curve, on a's counts model (k and background of scan a), minimised over the four
nuisances i0, b, q offset and sigma_q inside their registered bounds. The seven structural parameters
of b are kept. The five drawn scans are scored in both directions, once with their search bests (as
registered) and once with their repeat fits. The null is the same score between seeded random pairs of
the 100 repeat fits, which are fits of one sample on different scans. One JSON per score, written once.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_run as g1
import g_common as gc

M1 = g1.M1
NAMES = list(M1.names)
NUIS = ("i0", "b", "off", "sigq")
STRUCT = [n for n in NAMES if n not in NUIS]
OUT = g1.OUT / "hg1b_profiled"
LO = dict(zip(NAMES, M1.lo, strict=True))
HI = dict(zip(NAMES, M1.hi, strict=True))
LN10 = np.log(10.0)
VARIANT = "registered"


def use_wide() -> None:
    """Switch the module to declared deviation 1: M1W bounds, wide fits, output under g1/wide."""
    global M1, LO, HI, OUT, VARIANT
    import g1_wide

    M1 = g1_wide.M1W
    LO = dict(zip(NAMES, M1.lo, strict=True))
    HI = dict(zip(NAMES, M1.hi, strict=True))
    OUT = g1_wide.OUT / "hg1b_profiled"
    VARIANT = "wide"


def kbg(data: dict, scan: int) -> tuple[np.ndarray, np.ndarray]:
    return data["etfp"] * data["N"][scan] / data["af"], data["bg"][scan]


def counts(data: dict, scan: int, x) -> np.ndarray:
    k, bg = kbg(data, scan)
    return g1.sim_counts(M1, data["q"], np.asarray(x, float), k, bg)


def fit_scale(a: np.ndarray, bg: np.ndarray, y: np.ndarray, i0: float, b: float):
    """Best i0 and b for lam = a*i0 + 10^b*bg against y; no forward call."""

    def f(p):
        lam = a * p[0] + 10.0 ** p[1] * bg
        g = 2.0 * (1.0 - y / lam)
        return gc.deviance_terms(lam, y).sum(), np.array([g @ a, g @ (LN10 * 10.0 ** p[1] * bg)])

    r = minimize(
        f,
        [i0, b],
        jac=True,
        method="L-BFGS-B",
        bounds=[(LO["i0"], HI["i0"]), (LO["b"], HI["b"])],
    )
    return float(r.fun), r.x


def profile(data: dict, target: int, y: np.ndarray, xb: np.ndarray, x_start: np.ndarray) -> dict:
    """Minimise the deviance of b's structure against y over the nuisances, on the target's counts."""
    k, bg = kbg(data, target)
    x = np.array(xb, float)
    calls = 0

    def inner(off: float, lsig: float):
        nonlocal calls
        x[NAMES.index("off")] = off
        x[NAMES.index("sigq")] = 10.0**lsig
        x[NAMES.index("i0")] = 1.0
        x[NAMES.index("b")] = -30.0
        calls += 1
        a = g1.sim_counts(M1, data["q"], x, k, bg)
        return fit_scale(a, bg, y, x_start[NAMES.index("i0")], x_start[NAMES.index("b")])

    def outer(u):
        return inner(u[0] * 1e-3, u[1])[0]

    starts = [
        (x_start[NAMES.index("off")], x_start[NAMES.index("sigq")]),
        (xb[NAMES.index("off")], xb[NAMES.index("sigq")]),
    ]
    best = None
    for off0, s0 in starts:
        r = minimize(
            outer,
            [off0 * 1e3, np.log10(s0)],
            method="L-BFGS-B",
            bounds=[
                (LO["off"] * 1e3, HI["off"] * 1e3),
                (np.log10(LO["sigq"]), np.log10(HI["sigq"])),
            ],
            options={"maxfun": 40, "eps": 1e-3},
        )
        if best is None or r.fun < best.fun:
            best = r
    dev, (i0, b) = inner(best.x[0] * 1e-3, best.x[1])
    nu = {"i0": float(i0), "b": float(b), "off": best.x[0] * 1e-3, "sigq": 10.0 ** best.x[1]}
    return {"dev_profiled": dev, "nuisances": nu, "forward_calls": calls}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fits() -> dict[str, dict[int, np.ndarray]]:
    scans = g1.draw_scans()
    if VARIANT == "wide":
        import g1_wide

        search = {s: np.array(r["x"]) for s, r in g1_wide.wide_bests().items()}
        rep = {
            r["scan"]: np.array(r["x"]) for r in g1_wide.records("repeats", "r[0-9][0-9][0-9].json")
        }
        return {"search": search, "repeat": {s: rep[s] for s in scans}, "all_repeats": rep}
    search = {}
    for s in scans:
        recs = g1.search_records(M1, s)
        search[s] = np.array(min(recs, key=lambda r: r["dev_converged"])["x"])
    rep = {
        int(p.stem[1:]): np.array(load(p)["x"])
        for p in sorted((g1.OUT / "repeats").glob("r[0-9][0-9][0-9].json"))
    }
    return {"search": search, "repeat": {s: rep[s] for s in scans}, "all_repeats": rep}


def jobs(n_null: int) -> list[tuple[str, str, int, int]]:
    scans = g1.draw_scans()
    out = [
        (f"obs_{v}_{a:03d}_{b:03d}", v, a, b)
        for v in ("search", "repeat")
        for a in scans
        for b in scans
        if a != b
    ]
    rng = np.random.default_rng([gc.SEED, 2])
    for i in range(n_null):
        a, b = rng.choice(100, size=2, replace=False)
        out.append((f"null_{i:03d}_{a:03d}_{b:03d}", "all_repeats", int(a), int(b)))
    return out


def run(n_null: int, hours: float) -> None:
    gc.init_worker()
    data = g1.load_scans()
    f = fits()
    OUT.mkdir(parents=True, exist_ok=True)
    target_cache: dict[tuple[str, int], np.ndarray] = {}
    t_end = time.time() + hours * 3600
    for name, v, a, b in jobs(n_null):
        path = OUT / f"{name}.json"
        if path.exists():
            continue
        if time.time() > t_end:
            print("deadline reached", flush=True)
            return
        gc.wait_for_queue()
        xa, xb = f[v][a], f[v][b]
        if (v, a) not in target_cache:
            target_cache[(v, a)] = counts(data, a, xa)
        y = target_cache[(v, a)]
        t0 = time.time()
        own = float(gc.deviance_terms(counts(data, a, xb), y).sum())
        rec = {
            "set": v,
            "target": a,
            "source": b,
            "dev_unprofiled": own,
            **profile(data, a, y, xb, xa),
        }
        rec["seconds"] = time.time() - t0
        rec["off_bound"] = {
            "target": bool(abs(xa[NAMES.index("off")] - LO["off"]) < 1e-9),
            "source": bool(abs(xb[NAMES.index("off")] - LO["off"]) < 1e-9),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(path)
        print(
            f"{name}: unprofiled {own:.1f} profiled {rec['dev_profiled']:.1f} ({rec['seconds']:.0f} s)",
            flush=True,
        )


def report() -> None:
    recs = [load(p) for p in sorted(OUT.glob("*.json"))]
    null = np.array([r["dev_profiled"] for r in recs if r["set"] == "all_repeats"])
    null_fam = np.array(
        [
            r["dev_profiled"]
            for r in recs
            if r["set"] == "all_repeats" and r["off_bound"]["target"] == r["off_bound"]["source"]
        ]
    )
    phi = float(
        load(OUT.parent / "summary.json")["phi"]
        if VARIANT == "wide"
        else load(g1.OUT / "followup" / "hg1c_families.json")["pooled"]["phi"]
    )
    analytic = 2 * phi * float(chi2.ppf(0.95, len(STRUCT)))
    lines = [
        "# H-G1b profiled over the nuisances, with a null threshold",
        "",
        "Generated by `experiments/005_fisher_global/g1_hg1b.py report`. Follow-up to the registered H-G1b,",
        "not a registered test. Score: noise-free Poisson deviance of the source fit's structure against the",
        "target scan's best curve, on the target scan's counts, minimised over i0, b, q offset and sigma_q",
        f"inside the {'registered' if VARIANT == 'registered' else 'deviation-1 (M1W)'} bounds; "
        + "the seven structural parameters of the source are kept.",
        "",
        f"Null: {len(null)} seeded random pairs of the 100 repeat fits (one sample, different scans). "
        + (
            f"95th percentile {np.percentile(null, 95):.1f}, median {np.median(null):.1f}, max {null.max():.1f}."
            if len(null)
            else "not computed."
        ),
        (
            f"Pairs in the same q-offset family ({len(null_fam)}): 95th percentile {np.percentile(null_fam, 95):.1f}."
            if len(null_fam) > 2
            else ""
        ),
        f"Analytic same-truth reference 2 phi chi2_{len(STRUCT)} at 95 %, phi {phi:.2f}: {analytic:.1f} "
        + "(Poisson Fisher, no scan-to-scan variation). The registered threshold was 25.",
        "",
        (
            f"The empirical 95th percentile is {np.percentile(null, 95) / analytic:.0f} times the analytic one. "
            + "Inference: the gap comes from the fits, not from the counts. Every repeat fit has sigma_q on its "
            + "upper bound and most have the q offset on its lower bound, which splits the fits into two "
            + "q-offset families and lets the structural parameters absorb what the pinned nuisances cannot; "
            + "the slab model's misfit (deviance about 2 to 2.8 times n - p) adds to it. The split alone does "
            + f"not account for it, since the same-family percentile is still {np.percentile(null_fam, 95) / analytic:.0f} "
            + "times the analytic one. This threshold is a calibration for the present bounds only and is "
            + "recomputed after the rerun with wider q offset and sigma_q bounds."
            if VARIANT == "registered" and len(null) and len(null_fam) > 2
            else ""
        ),
        "",
    ]
    thr = float(np.percentile(null, 95)) if len(null) else None
    for v, title in (
        ("search", "search bests (as registered)"),
        ("repeat", "repeat fits of the same scans"),
    ):
        rows = [r for r in recs if r["set"] == v]
        if not rows:
            continue
        lines += [
            f"## Five drawn scans, {title}",
            "",
            "| target | source | unprofiled | profiled | q offset | sigma_q | above null 95th |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda r: (r["target"], r["source"])):
            above = "" if thr is None else ("yes" if r["dev_profiled"] > thr else "no")
            lines.append(
                f"| {r['target']} | {r['source']} | {r['dev_unprofiled']:.1f} | {r['dev_profiled']:.1f} | "
                + f"{r['nuisances']['off']:.2e} | {r['nuisances']['sigq']:.2e} | {above} |"
            )
        if thr is not None:
            n_above = sum(r["dev_profiled"] > thr for r in rows)
            lines += [
                "",
                f"{n_above} of {len(rows)} directed scores above the null 95th percentile.",
                "",
            ]
    (OUT.parent / "hg1b_profiled.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["profile", "report"])
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--variant", choices=["registered", "wide"], default="registered")
    args = ap.parse_args()
    if args.variant == "wide":
        use_wide()
    if args.mode == "profile":
        run(args.null, args.hours)
    else:
        report()


if __name__ == "__main__":
    main()
