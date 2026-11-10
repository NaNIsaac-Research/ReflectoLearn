"""G1 under declared deviation 1 of docs/paper/prereg/PREREG-G1.md: wider q offset and sigma_q bounds.

Same counts model, forwards, refinement and polish as g1_run.py; only the box changes (M1W, M2W).
Results go to docs/fisher_global/g1/wide/, one JSON per fit, written once, with the code's commit hash.
Run from the repository root, stages in this order (each resumes where it stopped):

    <venv python> -I experiments/005_fisher_global/g1_wide.py xseed   --worker 0 --workers 3
    <venv python> -I experiments/005_fisher_global/g1_wide.py repeats --worker 0 --workers 3
    <venv python> -I experiments/005_fisher_global/g1_wide.py m2      --worker 0 --workers 3
    <venv python> -I experiments/005_fisher_global/g1_wide.py search  --worker 0 --workers 3
    <venv python> -I experiments/005_fisher_global/g1_hg1b.py profile --variant wide
    <venv python> -I experiments/005_fisher_global/g1_wide.py report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import basin
import g1_run as g1
import g_common as gc

NUIS_WIDE = {"off": (-1e-2, 1e-2), "sigq": (1e-5, 3e-3)}


def widen(model: g1.Model, name: str, extra: dict[str, tuple[float, float]]) -> g1.Model:
    bounds = NUIS_WIDE | extra
    lo = tuple(
        bounds[n][0] if n in bounds else v for n, v in zip(model.names, model.lo, strict=True)
    )
    hi = tuple(
        bounds[n][1] if n in bounds else v for n, v in zip(model.names, model.hi, strict=True)
    )
    return g1.Model(name, model.names, lo, hi)


M1W = widen(g1.M1, "M1W", {})
M2W = widen(g1.M2, "M2W", {"ox_rho": (5.0, 120.0)})
OUT = g1.OUT / "wide"
# declared deviation 2: W4 cut from 64 to 16 starts per scan
N_SEARCH = 16
CONTAMINATION = {"ox_d": 10.0, "ox_sig": 3.0, "ox_rho": 10.0}
NUIS = ("i0", "b", "off", "sigq")
BOUND_TOL = 1e-6


def code_hash() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=gc.ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + (
        "-dirty" if git("status", "--porcelain", "--", "experiments") else ""
    )


CODE = code_hash()


def clip(model: g1.Model, x) -> np.ndarray:
    return np.clip(np.asarray(x, float), model.lo, model.hi)


def on_bound(model: g1.Model, x) -> list[str]:
    lo, hi = np.array(model.lo), np.array(model.hi)
    rel = (np.asarray(x, float) - lo) / (hi - lo)
    return [
        n for n, r in zip(model.names, rel, strict=True) if r <= BOUND_TOL or r >= 1 - BOUND_TOL
    ]


def fit(
    model: g1.Model, data: dict, scan: int, cid: str, x0: np.ndarray, fisher: bool = False
) -> dict:
    prob = g1.make_problem(model, data, scan, cid, x0)
    row = basin.refine(prob)
    pol = gc.polish(
        prob, np.array(row["x_min"]), None, None, data["raw"][scan], g1.sim_fn(model, data, scan)
    )
    rec = {
        "scan": scan,
        "model": model.name,
        "code": CODE,
        "x0": np.asarray(x0).tolist(),
        "outcome": row["outcome"],
        "dev_engine": row["d_final"],
        "seconds_engine": row["seconds"],
        "names": list(model.names),
        **pol,
        "on_bound": on_bound(model, pol["x"]),
    }
    if fisher:
        k = data["etfp"] * data["N"][scan] / data["af"]
        rec["lam"] = g1.sim_counts(
            model, data["q"], np.array(pol["x"]), k, data["bg"][scan]
        ).tolist()
        _, jlog = basin.Evaluator(prob).model(np.array(pol["x"]))
        rec["jlog"] = jlog.tolist()
    return rec


def run_jobs(
    jobs: Iterator[tuple[Path, Callable[[], dict]]], worker: int, workers: int, hours: float
) -> None:
    """Run the jobs of this worker; a job is (result path, thunk). Existing results are skipped."""
    gc.init_worker()
    t_end = time.time() + hours * 3600
    for n, (path, thunk) in enumerate(jobs):
        if n % workers != worker or path.exists():
            continue
        if time.time() > t_end:
            print("deadline reached", flush=True)
            return
        gc.wait_for_queue()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock")
        if not gc.take_lock(lock):
            continue
        try:
            rec = thunk()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec), encoding="utf-8")
            tmp.replace(path)
            print(
                f"{path.stem}: converged {rec['dev_converged']:.1f} on bound {rec['on_bound']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - recorded and skipped
            path.with_suffix(".error.json").write_text(
                json.dumps({"job": path.stem, "error": repr(exc), "code": CODE}), encoding="utf-8"
            )
            print(f"{path.stem}: error {exc!r}", flush=True)
        finally:
            lock.unlink(missing_ok=True)


def records(sub: str, pattern: str = "*.json") -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((OUT / sub).glob(pattern))
        if not p.name.endswith(".error.json")
    ]


def recorded_m1_bests() -> dict[int, np.ndarray]:
    out = {}
    for s in g1.draw_scans():
        recs = g1.search_records(g1.M1, s)
        out[s] = np.array(min(recs, key=lambda r: r["dev_converged"])["x"])
    return out


# ------------------------------------------------------------------ stages
def xseed_jobs(data: dict) -> Iterator[tuple[Path, Callable[[], dict]]]:
    bests = recorded_m1_bests()
    for s in bests:
        for src, xb in bests.items():

            def job(s=s, src=src, xb=xb):
                return {
                    "start": f"x{src:03d}",
                    **fit(M1W, data, s, f"g1w:{s}:x{src}", clip(M1W, xb)),
                }

            yield OUT / "xseed" / f"s{s:03d}_x{src:03d}.json", job


def search_jobs(data: dict) -> Iterator[tuple[Path, Callable[[], dict]]]:
    # start-major, so that every scan gets its first starts before any scan gets later ones
    for j in range(N_SEARCH):
        for s in g1.draw_scans():

            def job(s=s, j=j):
                rng = np.random.default_rng(np.random.SeedSequence([gc.SEED, 1, s, j]))
                x0 = rng.uniform(M1W.lo, M1W.hi)
                return {"start": j, **fit(M1W, data, s, f"g1w:{s}:{j}", x0)}

            yield OUT / "search_M1W" / f"s{s:03d}_{j:02d}.json", job


def search_records(scan: int, all_starts: bool = False) -> list[dict]:
    """W4 fits of a scan; by default only the declared N_SEARCH starts, not the ones run before the cut."""
    recs = records("search_M1W", f"s{scan:03d}_*.json")
    return recs if all_starts else [r for r in recs if r["start"] < N_SEARCH]


def scan_records(scan: int) -> list[dict]:
    return records("xseed", f"s{scan:03d}_*.json") + search_records(scan)


def repeat_start() -> np.ndarray:
    s = g1.draw_scans()[0]
    recs = records("xseed", f"s{s:03d}_*.json")
    if len(recs) < len(g1.draw_scans()):
        raise SystemExit(f"xseed of scan {s} incomplete ({len(recs)} fits); run xseed first")
    return np.array(min(recs, key=lambda r: r["dev_converged"])["x"])


def repeat_jobs(data: dict) -> Iterator[tuple[Path, Callable[[], dict]]]:
    x0 = repeat_start()
    for s in range(g1.N_SCANS):

        def job(s=s):
            return fit(M1W, data, s, f"g1w:{s}:rep", x0, fisher=True)

        yield OUT / "repeats" / f"r{s:03d}.json", job


def m2_starts(rep: dict, m2_best: np.ndarray) -> dict[str, np.ndarray]:
    v1 = dict(zip(rep["names"], rep["x"], strict=True))
    contam = [CONTAMINATION.get(n, v1.get(n)) for n in M2W.names]
    recorded = dict(zip(M2W.names, m2_best, strict=True)) | {n: v1[n] for n in NUIS}
    return {
        "contam": clip(M2W, contam),
        "m2best": clip(M2W, [recorded[n] for n in M2W.names]),
    }


def m2_jobs(data: dict) -> Iterator[tuple[Path, Callable[[], dict]]]:
    s0 = g1.draw_scans()[0]
    m2 = g1.search_records(g1.M2, s0)
    m2_best = np.array(min(m2, key=lambda r: r["dev_converged"])["x"])
    for s in range(g1.N_SCANS):
        rep_path = OUT / "repeats" / f"r{s:03d}.json"
        for tag in ("contam", "m2best"):

            def job(s=s, tag=tag, rep_path=rep_path):
                rep = json.loads(rep_path.read_text(encoding="utf-8"))
                x0 = m2_starts(rep, m2_best)[tag]
                return {"start": tag, **fit(M2W, data, s, f"g1w:{s}:m2{tag}", x0)}

            yield OUT / "m2" / f"m{s:03d}_{tag}.json", job


STAGES = {"xseed": xseed_jobs, "repeats": repeat_jobs, "m2": m2_jobs, "search": search_jobs}


# ------------------------------------------------------------------ report
def wide_bests() -> dict[int, dict]:
    """Lowest W1 or W4 minimum per drawn scan."""
    out = {}
    for s in g1.draw_scans():
        recs = scan_records(s)
        if recs:
            out[s] = min(recs, key=lambda r: r["dev_converged"])
    return out


def crb_ratios(reps: list[dict], phi: float, drop: tuple[str, ...]) -> dict[str, dict]:
    names = reps[0]["names"]
    keep = [i for i, n in enumerate(names) if n not in drop]
    xs = np.array([r["x"] for r in reps])
    out = {}
    for key in ("Au_d", "Cr_d"):
        i = names.index(key)
        crbs = []
        for r in reps:
            j = np.array(r["jlog"])[:, keep]
            lam = np.array(r["lam"])
            cov = np.linalg.pinv(j.T @ (lam[:, None] * j))
            crbs.append(float(np.sqrt(phi * cov[keep.index(i), keep.index(i)])))
        sd = float(np.std(xs[:, i], ddof=1))
        out[key] = {
            "sd": sd,
            "crb_median": float(np.median(crbs)),
            "ratio_median": sd / float(np.median(crbs)),
            "crb_first": crbs[0],
            "ratio_first": sd / crbs[0],
        }
    return out


def fisher_correlations(
    reps: list[dict], pairs: list[tuple[str, str]]
) -> dict[tuple[str, str], float]:
    """Median over repeats of the correlation between two parameters in the inverse Fisher matrix (phi cancels)."""
    names = reps[0]["names"]
    cors: dict[tuple[str, str], list[float]] = {p: [] for p in pairs}
    for r in reps:
        j = np.array(r["jlog"])
        cov = np.linalg.pinv(j.T @ (np.array(r["lam"])[:, None] * j))
        sd = np.sqrt(np.diag(cov))
        for a, b in pairs:
            ia, ib = names.index(a), names.index(b)
            cors[(a, b)].append(float(cov[ia, ib] / (sd[ia] * sd[ib])))
    return {p: float(np.median(v)) for p, v in cors.items()}


def cut_check(scans: list[int]) -> list[str]:
    """Per scan, whether the best of the N_SEARCH random starts reaches the lowest minimum of the other fits."""
    rows = []
    for s in scans:
        w4 = search_records(s)
        if not w4:
            continue
        w1 = records("xseed", f"s{s:03d}_*.json")
        extra = [r for r in search_records(s, all_starts=True) if r["start"] >= N_SEARCH]
        b4 = min(r["dev_converged"] for r in w4)
        ref = min(r["dev_converged"] for r in w1 + extra)
        same = "yes" if b4 <= ref + 1 else "no"
        rows.append(
            f"| {s} | {len(w4)} | {b4:.1f} | {min(r['dev_converged'] for r in w1):.1f} | "
            + (f"{min(r['dev_converged'] for r in extra):.1f} ({len(extra)})" if extra else "none")
            + f" | {b4 - ref:+.1f} | {same} |"
        )
    if not rows:
        return []
    return [
        f"## W4 cut to {N_SEARCH} starts per scan",
        "",
        f"Best deviance of the first {N_SEARCH} random starts against the lowest of the other fits of the scan "
        + f"(W1 cross-seeded, and any random starts {N_SEARCH} and above run before the cut). "
        + "Reached means within 1 of it.",
        "",
        f"| scan | starts done | best of W4 | best of W1 | best of starts >= {N_SEARCH} (n) | difference | reached |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def verdict(ratio: float) -> str:
    return "holds" if ratio <= 2 else ("fails" if ratio > 3 else "unresolved")


def bound_counts(recs: list[dict]) -> str:
    counts: dict[str, int] = {}
    for r in recs:
        for n in r["on_bound"]:
            counts[n] = counts.get(n, 0) + 1
    none = sum(not r["on_bound"] for r in recs)
    listed = ", ".join(f"{n} {c}" for n, c in sorted(counts.items(), key=lambda t: -t[1]))
    return f"{none} of {len(recs)} fits with no parameter on a bound" + (
        f"; on a bound: {listed}" if listed else ""
    )


def report() -> None:
    data = g1.load_scans()
    scans = g1.draw_scans()
    n_pts = int(data["q"].size)
    lines = [
        "# G1 under declared deviation 1: wider q offset and sigma_q bounds",
        "",
        "Generated by `experiments/005_fisher_global/g1_wide.py report`. Results under the deviation declared in",
        "`docs/paper/prereg/PREREG-G1.md`, not the registered test; the registered results are in `../g1_report.md`.",
        f"Box: q offset {M1W.lo[M1W.names.index('off')]:g} to {M1W.hi[M1W.names.index('off')]:g} 1/A, "
        + f"sigma_q {M1W.lo[M1W.names.index('sigq')]:g} to {M1W.hi[M1W.names.index('sigq')]:g} 1/A; "
        + "M2W layer SLD from 5. Converged-forward scores only.",
        "",
    ]
    codes = sorted(
        {r["code"] for sub in ("xseed", "search_M1W", "repeats", "m2") for r in records(sub)}
    )
    lines += [f"Code hashes in the result files: {', '.join(codes) or 'none'}.", ""]

    sep, tot, summary = 0, 0, {}
    for s in scans:
        recs = scan_records(s)
        if not recs:
            continue
        n_x = sum(str(r["start"]).startswith("x") for r in recs)
        bs = g1.basins_of_scan(M1W, data, s, recs)
        others = bs[1:]
        sep += sum(b["dD"] >= g1.H_A_SEP for b in others)
        tot += len(others)
        best = recs[bs[0]["best"]]
        reach = len(bs[0]["members"])
        lines += g1.basin_table(M1W, s, recs, bs)
        lines += [
            f"Scan {s}: {n_x} cross-seeded and {len(recs) - n_x} random starts; {reach} reach the best basin "
            + f"(start {best['start']}). {bound_counts(recs)}.",
            "",
        ]
        if 2 * reach < len(recs):
            nearest = min((b["dD"] for b in others), default=float("nan"))
            lines += [
                f"The global minimum on scan {s} is not certain: the best basin is reached by {reach} of "
                + f"{len(recs)} starts and the nearest other basin is at dD {nearest:.3g}"
                + ("; random starts (W4) have not run yet." if n_x == len(recs) else "."),
                "",
            ]
        summary[s] = {
            "best_dev": best["dev_converged"],
            "best_start": best["start"],
            "reach": reach,
        }
    if tot:
        lines.append(
            f"H-G1a: {sep} of {tot} other basins at dD >= {g1.H_A_SEP:g} ({sep / tot:.3f}); holds >= 0.80, fails < 0.60."
        )
        lines.append("")
    lines += cut_check(scans)

    bests = wide_bests()
    if len(bests) == len(scans):
        k0 = data["etfp"] * data["N"][scans[0]] / data["af"]
        lam_ref = g1.sim_counts(
            M1W, data["q"], np.array(bests[scans[0]]["x"]), k0, data["bg"][scans[0]]
        )
        cross = [
            float(
                gc.deviance_terms(
                    g1.sim_counts(
                        M1W, data["q"], np.array(bests[s]["x"]), k0, data["bg"][scans[0]]
                    ),
                    lam_ref,
                ).sum()
            )
            for s in scans[1:]
        ]
        lines += [
            f"H-G1b as coded for the registered run (bests scored on scan {scans[0]}'s counts, nuisances inside the "
            + f"compared vectors): {[round(c, 1) for c in cross]}; registered rule all <= {g1.H_B_CROSS:g}. "
            + "The profiled score with the null threshold is in `hg1b_profiled.md`.",
            "",
        ]

    reps = records("repeats", "r[0-9][0-9][0-9].json")
    phi = None
    if reps:
        pearson = np.array(
            [
                (data["raw"][r["scan"]] - np.array(r["lam"])) / np.sqrt(np.array(r["lam"]))
                for r in reps
            ]
        )
        phi = float(np.median(np.var(pearson, axis=0, ddof=1)))
        devs = np.array([r["dev_converged"] for r in reps])
        p1 = len(M1W.names)
        lines += [
            f"## W2: {len(reps)} repeat fits from scan {scans[0]}'s best",
            "",
            f"Median phi {phi:.3f}. Deviance over n - p ({n_pts} - {p1}): median "
            + f"{np.median(devs) / (n_pts - p1):.2f}, range {devs.min() / (n_pts - p1):.2f} to {devs.max() / (n_pts - p1):.2f}.",
            bound_counts(reps) + ".",
            "",
            "| parameter | nuisances | across-scan sd (A) | CRB, median over fits (A) | ratio | verdict | ratio with first fit's CRB |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, drop in (("free", ()), ("q offset and sigma_q fixed", ("off", "sigq"))):
            for key, v in crb_ratios(reps, phi, drop).items():
                lines.append(
                    f"| {key} | {label} | {v['sd']:.3g} | {v['crb_median']:.3g} | {v['ratio_median']:.2f} | "
                    + f"{verdict(v['ratio_median'])} | {v['ratio_first']:.2f} |"
                )
        xs = np.array([r["x"] for r in reps])
        cors = fisher_correlations(
            reps, [(p, n) for p in ("Au_d", "Cr_d") for n in ("off", "sigq")] + [("off", "sigq")]
        )
        lines += [
            "",
            "H-G1c rule: holds if the ratio is at most 2 for both, fails if above 3 for either.",
            "",
            "Correlations in the inverse Fisher matrix with all parameters free (median over repeats): "
            + ", ".join(f"{a} with {b} {c:+.3f}" for (a, b), c in cors.items())
            + ". A thickness strongly correlated with the q offset or sigma_q has a CRB that shrinks when "
            + "those are fixed, which is why the two CRB rows differ.",
            "",
            "Fitted values (median over repeats), for description only: "
            + ", ".join(f"{n} {np.median(xs[:, i]):.4g}" for i, n in enumerate(M1W.names)),
            "",
        ]

    m2 = records("m2")
    if m2 and reps:
        by_scan: dict[int, dict] = {}
        for r in m2:
            if r["scan"] not in by_scan or r["dev_converged"] < by_scan[r["scan"]]["dev_converged"]:
                by_scan[r["scan"]] = r
        d1 = {r["scan"]: r["dev_converged"] for r in reps}
        common = sorted(set(by_scan) & set(d1))
        dd = np.array([d1[s] - by_scan[s]["dev_converged"] for s in common])
        p2 = len(M2W.names)
        qf = np.array(
            [
                (d1[s] - by_scan[s]["dev_converged"])
                / 3
                / (by_scan[s]["dev_converged"] / (n_pts - p2))
                for s in common
            ]
        )
        layer = np.array(
            [
                [by_scan[s]["x"][M2W.names.index(n)] for n in ("ox_d", "ox_sig", "ox_rho")]
                for s in common
            ]
        )
        wins = {t: sum(by_scan[s]["start"] == t for s in common) for t in ("contam", "m2best")}
        lines += [
            f"## W3: M2W on {len(common)} repeats",
            "",
            f"dD = D(M1W) - D(M2W): median {np.median(dd):.1f}, quartiles {np.percentile(dd, 25):.1f} to "
            + f"{np.percentile(dd, 75):.1f}, range {dd.min():.1f} to {dd.max():.1f}; {int((dd > 0).sum())} of {len(dd)} positive.",
            f"Quasi-F (dD / 3) / (D(M2W) / (n - {p2})): median {np.median(qf):.1f}, quartiles "
            + f"{np.percentile(qf, 25):.1f} to {np.percentile(qf, 75):.1f}. Descriptive, no rule.",
            f"Added layer (median, quartiles): thickness {np.median(layer[:, 0]):.3g} A "
            + f"({np.percentile(layer[:, 0], 25):.3g} to {np.percentile(layer[:, 0], 75):.3g}), roughness "
            + f"{np.median(layer[:, 1]):.3g} A, SLD {np.median(layer[:, 2]):.3g} "
            + f"({np.percentile(layer[:, 2], 25):.3g} to {np.percentile(layer[:, 2], 75):.3g}) x 1e-6 1/A^2.",
            f"Lower minimum from the contamination start in {wins['contam']} repeats, from the recorded M2 best in {wins['m2best']}.",
            bound_counts(list(by_scan.values())) + ".",
            "",
        ]
        au_rho = float(np.median([by_scan[s]["x"][M2W.names.index("Au_rho")] for s in common]))
        if abs(np.median(layer[:, 2]) - au_rho) <= 0.2 * au_rho:
            lines += [
                f"The added layer's SLD is close to the Au SLD in the same fits ({au_rho:.3g}). It may be an Au "
                + "sub-layer or a split of the Au interface rather than a contamination layer; the fit cannot "
                + "tell which (inference).",
                "",
            ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps({"phi": phi, "scans": summary, "code": CODE}, indent=1) + "\n", encoding="utf-8"
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=[*STAGES, "report"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--hours", type=float, default=16.0)
    args = ap.parse_args()
    if args.mode == "report":
        report()
        return
    data = g1.load_scans()
    if not g1.precheck()["usable"]:
        raise SystemExit("pre-check failed")
    run_jobs(STAGES[args.mode](data), args.worker, args.workers, args.hours)


if __name__ == "__main__":
    main()
