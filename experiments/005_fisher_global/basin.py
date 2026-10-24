"""Basin minimum validation (docs/fisher_global/PREREG-BASIN.md).

Recorded wrong endpoints of R2a (4 < ΔD ≤ 10⁵) and X2e's far states with the true layer count are
refined locally, from the recorded point only, until a registered convergence test passes or the
evaluation budget runs out. At a confirmed minimum S₁, the Fisher ratio and the q-region shares are
recomputed. No new start is drawn and no earlier record is written.

R2a runs under this repository's interpreter, X2e under the X2e worktree's (its fit_policy package
and frozen records at a14041f). From the repository root:

    .venv/Scripts/python.exe -I experiments/005_fisher_global/basin.py candidates --task-a <task_a.json>
    .venv/Scripts/python.exe -I experiments/005_fisher_global/basin.py run --source r2a --set pilot
    <x2e>/.venv/Scripts/python.exe -I experiments/005_fisher_global/basin.py run --source x2e \
        --set pilot --x2e <x2e>
    .venv/Scripts/python.exe -I experiments/005_fisher_global/basin_pilot_report.py

The full run (one result file per candidate and objective in docs/fisher_global/basin_full/, resumable,
several workers per source) adds `--set full --worker <k> --control <file>`; `basin_report.py` then
applies the registered verdict rules.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import Tensor
from torch.func import jacfwd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "fisher_global"
CANDIDATES = OUT / "basin_candidates.json"
FULL_DIR = OUT / "basin_full"

D_MAX = 1e5  # R2a candidates: 4 < ΔD ≤ D_MAX
REACH = 4.0
X2E_ROOT = 20261027
X2E_COMMIT = "a14041f"

STAGE_NFEV = 200
N_MAX = 3000  # evaluations per candidate and objective
TAU_ABS, TAU_REL = 1e-3, 1e-6
STEP_TOL = 1e-6  # bound-scaled
ACTIVE_TOL = 1e-6  # bound-scaled distance that counts as on the bound
WEAK_REL = 1e-10  # eigenvalues below this fraction of the largest are probed, not inverted
PROBE_H = 1e-3  # bound-scaled probe step along weak directions
S1_REALISTIC = 1e4


def stable_terms(lam: np.ndarray, lam_t: np.ndarray) -> np.ndarray:
    """b1.deviance_terms (deviance against noise-free data with the small-r series), repeated
    because b1 imports the R2a model, which the X2e interpreter does not load."""
    r = lam / lam_t - 1
    small = np.abs(r) < 1e-3
    rs = np.where(small, r, 0.0)
    series = rs**2 / 2 - rs**3 / 3 + rs**4 / 4 - rs**5 / 5
    direct = r - np.log1p(np.where(small, 0.0, r))
    return 2 * lam_t * np.where(small, series, direct)


def regions(q: np.ndarray, edge: float) -> list[np.ndarray]:
    """Task A's and B-1's split of the 291-point grid: q below 2θ = 0.5°, then ten blocks."""
    above = np.flatnonzero(q >= edge)
    return [np.flatnonzero(q < edge), *np.array_split(above, 10)]


def peak_mb() -> float:
    """Peak working set of this process (Windows)."""

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    info(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20


def closest_to_median(rows: list[dict], key: str) -> dict:
    logs = np.log10([r[key] for r in rows])
    med = np.median(logs)
    return min(rows, key=lambda r: (abs(np.log10(r[key]) - med), r["id"]))


def candidates(task_a: Path) -> None:
    b1 = json.loads((OUT / "b1.json").read_text(encoding="utf-8"))
    r2a = [
        {
            "id": f"r2a:{c['index']:02d}:{r['start']:02d}",
            "cond": c["index"],
            "start": r["start"],
            "dD": r["dD"],
            "nfev": r["nfev"],
            "capped": r["nfev"] >= 200,
        }
        for c in b1["conditions"]
        for r in c["rows"]
        if REACH < r["dD"] <= D_MAX
    ]
    ta = json.loads(task_a.read_text(encoding="utf-8"))
    x2e = [
        {
            "id": f"x2e:{r['arm']}:{r['n']}:{r['j']}:{r['s']}:{r['k']}",
            "arm": r["arm"],
            "n": r["n"],
            "j": r["j"],
            "s": r["s"],
            "k": r["k"],
            "dnf": r["dnf"],
        }
        for r in ta["rows"]
        if r["far"] and r["L"] == r["n"]
    ]
    if len({c["id"] for c in x2e}) != len(x2e):
        raise SystemExit("X2e candidate keys are not unique")

    def group(rows, **match):
        return [r for r in rows if all(r[k] == v for k, v in match.items())]

    p1 = closest_to_median(group(r2a, cond=6, capped=True), "dD")
    rest = [r for r in group(r2a, cond=6, capped=True) if r["id"] != p1["id"]]
    p2 = min(rest, key=lambda r: (abs(np.log10(r["dD"] / p1["dD"])), r["id"]))
    p3 = min(group(r2a, cond=25), key=lambda r: r["dD"])
    pilot = [
        p1,
        p2,
        p3,
        closest_to_median(group(r2a, cond=33, capped=True), "dD"),
        closest_to_median(group(r2a, cond=0, capped=True), "dD"),
        closest_to_median(group(r2a, cond=21, capped=True), "dD"),
        closest_to_median(group(x2e, arm="C", n=2, s=1), "dnf"),
        closest_to_median(group(x2e, arm="D2", n=3, s=0), "dnf"),
    ]
    out = {
        "r2a_source": "docs/fisher_global/b1.json (endpoints of docs/identifiability/scan at 829f330)",
        "x2e_source": f"Task A task_a.json rows with far and L == n (X2e records at {X2E_COMMIT})",
        "r2a": r2a,
        "x2e": x2e,
        "pilot": [p["id"] for p in pilot],
    }
    CANDIDATES.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"R2a {len(r2a)} (capped {sum(r['capped'] for r in r2a)}), X2e {len(x2e)}")
    for p in pilot:
        print(p)


@dataclass
class Problem:
    """One refinement: Poisson deviance of exp(log_counts(θ)) against y over the bounded box."""

    cid: str
    objective: str
    names: list[str]
    log_counts: Callable[[Tensor], Tensor]
    y: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    x_end: np.ndarray
    terms: Callable[[np.ndarray, np.ndarray], np.ndarray]
    recorded: float  # the endpoint's recorded deviance for this objective, for the start check
    lam_t: np.ndarray  # truth's noise-free counts on the full grid
    full_counts: Callable[[np.ndarray], np.ndarray]  # θ -> counts on the full grid
    theta_t: np.ndarray | None  # truth in the same coordinates (None if not comparable)
    fisher: np.ndarray | None  # Fisher information at the truth, full grid
    scale: float
    region_idx: list[np.ndarray] | None


def r2a_problems(ids: list[str]) -> list[Problem]:
    sys.path.insert(0, str(ROOT / "experiments" / "001_identifiability"))
    import fim
    import nonunique
    import scan

    out = []
    for cid in ids:
        _, cond_s, start_s = cid.split(":")
        data = json.loads(
            (ROOT / "docs" / "identifiability" / "scan" / f"fits_{int(cond_s):02d}.json").read_text(
                encoding="utf-8"
            )
        )
        e = next(e for e in data["endpoints"] if e["start"] == int(start_s))
        cond, p0, free = data["condition"], data["truth"], tuple(data["free"])
        _, _, bounds = scan.setup(cond)
        q = scan.q_grid(cond["tth_max"])
        qt = torch.as_tensor(q)
        fixed = {k: torch.tensor(v) for k, v in e["params"].items()}

        def log_counts(theta: Tensor, fixed=fixed, free=free, qt=qt) -> Tensor:
            return torch.log(scan.expected_counts(qt, fixed | dict(zip(free, theta, strict=True))))

        def full_counts(x: np.ndarray, log_counts=log_counts) -> np.ndarray:
            with torch.no_grad():
                return torch.exp(log_counts(torch.as_tensor(x))).numpy()

        theta_t = np.array([p0[k] for k in free])
        lam_t, jlog = scan.model(q, p0, free)(theta_t)
        out.append(
            Problem(
                cid=cid,
                objective="nf",
                names=list(free),
                log_counts=log_counts,
                y=lam_t,
                lo=np.array([bounds[k][0] for k in free]),
                hi=np.array([bounds[k][1] for k in free]),
                x_end=np.array([e["params"][k] for k in free]),
                terms=nonunique.deviance_terms,
                recorded=e["deviance"],
                lam_t=lam_t,
                full_counts=full_counts,
                theta_t=theta_t,
                fisher=jlog.T @ (lam_t[:, None] * jlog),
                scale=scan.S_COUNTS,
                region_idx=regions(q, fim.tth_to_q(np.array(0.5))) if len(q) == 291 else None,
            )
        )
    return out


def x2e_problems(ids: list[str], wt: Path, objectives: tuple[str, ...]) -> list[Problem]:
    sys.path.insert(0, str(wt / "src"))
    from reflectolearn.fit_policy import stack as st

    runs = wt / "docs" / "fit_policy" / "x2e_runs"
    q = st.q_grid()
    qt = torch.as_tensor(q)
    held = st.held_out_mask(q)
    regs = regions(q, st.tth_to_q(st.TTH_HELD_MIN))
    task_a_dnf = {
        c["id"]: c["dnf"] for c in json.loads(CANDIDATES.read_text(encoding="utf-8"))["x2e"]
    }
    records: dict[str, list[dict]] = {}
    out = []
    for cid in ids:
        _, arm, n_s, j_s, s_s, k_s = cid.split(":")
        n, j, s, k = int(n_s), int(j_s), int(s_s), int(k_s)
        if arm not in records:
            records[arm] = [json.loads(x) for x in (runs / f"{arm}.jsonl").read_text().splitlines()]
        matches = [r for r in records[arm] if (r["n"], r["j"], r["s"], r["k"]) == (n, j, s, k)]
        if len(matches) != 1:
            raise SystemExit(f"{cid}: {len(matches)} records")
        r = matches[0]
        state = st.Stack(np.asarray(r["layers"], float), np.asarray(r["instr"], float))
        truth = st.sample_truth(n, j, X2E_ROOT)
        scale = st.NOISE_SCALES[s]
        lam_t = st.counts(truth, q, scale)
        x_t = torch.as_tensor(truth.vector(), dtype=torch.float64)
        jac = jacfwd(lambda x, n=n, scale=scale: st.expected_counts(qt, x, n, scale))(x_t).numpy()
        fisher = jac.T @ (jac / lam_t[:, None])
        lo, hi = state.bounds()
        names = [f"{key}{i + 1}" for i in range(n) for key in st.LAYER_KEYS] + list(st.INSTR_KEYS)

        def full_counts(x: np.ndarray, n=n, scale=scale) -> np.ndarray:
            with torch.no_grad():
                return st.expected_counts(qt, torch.as_tensor(x), n, scale).numpy()

        for objective in objectives:
            if objective == "nf":
                mask, y = np.ones(len(q), bool), lam_t
                recorded = task_a_dnf[cid]
            else:
                mask = ~held
                y = st.noisy_counts(lam_t, n, j, s, root=X2E_ROOT)[mask]
                recorded = r["deviance"]
            qm = torch.as_tensor(q[mask])

            def log_counts(theta: Tensor, n=n, scale=scale, qm=qm) -> Tensor:
                return torch.log(st.expected_counts(qm, theta, n, scale))

            out.append(
                Problem(
                    cid=cid,
                    objective=objective,
                    names=names,
                    log_counts=log_counts,
                    y=y,
                    lo=lo,
                    hi=hi,
                    x_end=state.vector(),
                    terms=st.deviance_terms,
                    recorded=recorded,
                    lam_t=lam_t,
                    full_counts=full_counts,
                    theta_t=truth.vector(),
                    fisher=fisher,
                    scale=scale,
                    region_idx=regs,
                )
            )
    return out


class Evaluator:
    """Counts and Jacobian with a one-point cache, as scan.fit and stack.fit, plus counters."""

    def __init__(self, prob: Problem):
        self.prob = prob
        self.jac_fn = jacfwd(prob.log_counts)
        self.cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self.model_calls = 0
        self.value_calls = 0
        self.hessian_calls = 0
        y = torch.as_tensor(prob.y)

        def grad(theta: Tensor) -> Tensor:
            lam = torch.exp(prob.log_counts(theta))
            return 2 * ((lam - y)[:, None] * self.jac_fn(theta)).sum(0)

        self.grad_fn = grad

    def model(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = x.tobytes()
        if key not in self.cache:
            self.cache.clear()
            self.model_calls += 1
            theta = torch.as_tensor(x)
            with torch.no_grad():
                lam = torch.exp(self.prob.log_counts(theta)).numpy()
            self.cache[key] = lam, self.jac_fn(theta).numpy()
        return self.cache[key]

    def deviance(self, x: np.ndarray) -> float:
        self.value_calls += 1
        with torch.no_grad():
            lam = torch.exp(self.prob.log_counts(torch.as_tensor(x))).numpy()
        return float(self.prob.terms(lam, self.prob.y).sum())

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Exact Hessian of the deviance, one column per forward-mode pass over the gradient: a
        batched jacfwd of jacfwd peaked at 1.7 GB for 15 parameters in the smoke test."""
        self.hessian_calls += 1
        theta = torch.as_tensor(x)
        eye = torch.eye(len(x), dtype=theta.dtype)
        cols = [torch.func.jvp(self.grad_fn, (theta,), (e,))[1] for e in eye]
        h = torch.stack(cols, 1).detach().numpy()
        return 0.5 * (h + h.T)


def trf_stage(ev: Evaluator, x: np.ndarray, max_nfev: int):
    """One trust-region run with the settings of scan.fit and stack.fit (Poisson residuals)."""
    p = ev.prob
    lo, hi, y = p.lo, p.hi, p.y
    x0 = np.clip(x, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))

    def resid(x):
        lam, _ = ev.model(x)
        return np.sign(lam - y) * np.sqrt(p.terms(lam, y))

    def jac(x):
        lam, jlog = ev.model(x)
        r = np.sign(lam - y) * np.sqrt(p.terms(lam, y))
        near = np.abs(lam - y) <= 1e-6 * np.maximum(y, 1e-12)
        g = np.where(near, 1 / np.sqrt(lam), (1 - y / lam) / np.where(near, 1.0, r))
        return (g * lam)[:, None] * jlog

    return least_squares(
        resid,
        x0,
        jac=jac,
        bounds=(lo, hi),
        x_scale="jac",
        xtol=1e-10,
        ftol=1e-10,
        max_nfev=max_nfev,
    )


def tau(d: float) -> float:
    return max(TAU_ABS, TAU_REL * d)


def first_order(ev: Evaluator, x: np.ndarray, d: float) -> dict:
    """Registered tests C1-C3 at x (PREREG-BASIN 3.3), in bound-scaled coordinates u = θ / width."""
    p = ev.prob
    w = p.hi - p.lo
    lam, jlog = ev.model(x)
    g = 2 * ((lam - p.y)[:, None] * jlog).sum(0) * w
    h = ev.hessian(x) * np.outer(w, w)
    t = tau(d)
    rel = (x - p.lo) / w
    at_lo, at_hi = rel <= ACTIVE_TOL, rel >= 1 - ACTIVE_TOL
    kkt = bool(np.all(g[at_lo] >= -t) and np.all(g[at_hi] <= t))
    free = ~(at_lo | at_hi)
    out = {
        "tau": t,
        "kkt": kkt,
        "at_lo": [p.names[i] for i in np.flatnonzero(at_lo)],
        "at_hi": [p.names[i] for i in np.flatnonzero(at_hi)],
        "grad_inf": float(np.abs(g[free]).max()) if free.any() else 0.0,
    }
    if not free.any():
        return out | {
            "decrement": 0.0,
            "weak_grad": 0.0,
            "n_weak": 0,
            "descent": False,
            "eig_min_rel": None,
            "ok": kkt,
        }
    hf, gf = h[np.ix_(free, free)], g[free]
    evals, evecs = np.linalg.eigh(hf)
    top = np.abs(evals).max()
    strong = evals > WEAK_REL * top
    proj = evecs.T @ gf
    decrement = float(0.5 * np.sum(proj[strong] ** 2 / evals[strong]))
    weak_grad = float(np.linalg.norm(proj[~strong]))
    descent = False
    probes = []
    u = rel.copy()
    for v in evecs[:, ~strong].T:
        for sign in (1, -1):
            uu = u.copy()
            uu[free] = np.clip(u[free] + sign * PROBE_H * v, 0.0, 1.0)
            dp = ev.deviance(p.lo + uu * w)
            probes.append(dp - d)
            descent |= dp < d - t
    return out | {
        "decrement": decrement,
        "weak_grad": weak_grad,
        "n_weak": int((~strong).sum()),
        "eig_min_rel": float(evals[0] / top),
        "probe_min": float(min(probes)) if probes else None,
        "descent": bool(descent),
        "ok": bool(kkt and decrement <= t and weak_grad <= t and not descent),
    }


def refine(prob: Problem) -> dict:
    t0 = time.perf_counter()
    ev = Evaluator(prob)
    w = prob.hi - prob.lo
    x = prob.x_end.copy()
    d = ev.deviance(x)
    start_check = abs(d - prob.recorded) / max(prob.recorded, 1e-300)
    stages, used, pending, stalls = [], 0, False, 0
    outcome, checks = None, None
    while outcome is None:
        if used >= N_MAX:
            outcome = "unconfirmed: budget"
            break
        ts = time.perf_counter()
        calls0 = ev.model_calls
        res = trf_stage(ev, x, min(STAGE_NFEV, N_MAX - used))
        used += res.nfev
        x_new = prob.lo + np.clip((res.x - prob.lo) / w, 0, 1) * w
        d_new = float(2 * res.cost)
        moved = float(np.abs((x_new - x) / w).max())
        stage = {
            "nfev": int(res.nfev),
            "njev": int(res.njev or 0),
            "model_calls": ev.model_calls - calls0,
            "status": int(res.status),
            "d_before": d,
            "d_after": d_new,
            "moved": moved,
            "seconds": time.perf_counter() - ts,
            "confirmation": pending,
        }
        if not np.isfinite(d_new) or not np.all(np.isfinite(res.x)):
            stages.append(stage)
            outcome = "unconfirmed: numerical"
            break
        if pending:
            pending = False
            if moved <= STEP_TOL and d - d_new <= tau(d):
                stages.append(stage)
                outcome = "confirmed"
                break
        x_prev_d = d
        x, d = x_new, d_new
        if res.status != 0:
            checks = first_order(ev, x, d)
            stage["checks"] = checks
            if checks["ok"]:
                pending = True
            elif moved <= STEP_TOL and x_prev_d - d <= tau(d):
                stalls += 1
                if stalls >= 2:
                    outcome = "unconfirmed: stalled"
            else:
                stalls = 0
        stages.append(stage)
    if checks is None or outcome != "confirmed":
        final_checks = first_order(ev, x, d)
    else:
        final_checks = checks
    lam = prob.full_counts(x)
    terms = stable_terms(lam, prob.lam_t)
    d_nf = float(terms.sum())
    lam_end = prob.full_counts(prob.x_end)
    terms_end = stable_terms(lam_end, prob.lam_t)
    d_nf_end = float(terms_end.sum())
    row = {
        "id": prob.cid,
        "objective": prob.objective,
        "outcome": outcome,
        "start_check_rel": start_check,
        "d_recorded": prob.recorded,
        "d_final": d,
        "dnf_end": d_nf_end,
        "dnf_min": d_nf,
        "S1_end": prob.scale / d_nf_end if d_nf_end > 0 else None,
        "S1_min": prob.scale / d_nf if d_nf > 0 else None,
        "kappa": d_nf_end / d_nf if d_nf > 0 else None,
        "truth_basin": d_nf <= REACH,
        "drift_inf": float(np.abs((x - prob.x_end) / w).max()),
        "x_end": prob.x_end.tolist(),
        "x_min": x.tolist(),
        "names": prob.names,
        "lam_min": lam.tolist(),
        "nfev_total": used,
        "model_calls": ev.model_calls,
        "value_calls": ev.value_calls,
        "hessian_calls": ev.hessian_calls,
        "final_checks": final_checks,
        "stages": stages,
        "seconds": time.perf_counter() - t0,
        "peak_mb": peak_mb(),
    }
    if prob.fisher is not None and prob.theta_t is not None:
        for tag, xx, dd in (("end", prob.x_end, d_nf_end), ("min", x, d_nf)):
            dt = xx - prob.theta_t
            d_f = float(dt @ prob.fisher @ dt)
            row[f"dF_{tag}"] = d_f
            row[f"rho_{tag}"] = d_f / dd if dd > 0 else None
    if prob.region_idx is not None:
        row["share_end"] = [float(terms_end[i].sum() / d_nf_end) for i in prob.region_idx]
        row["share_min"] = [float(terms[i].sum() / d_nf) for i in prob.region_idx]
    return row


def result_path(cid: str, objective: str) -> Path:
    return FULL_DIR / f"{cid.replace(':', '_')}__{objective}.json"


def full_run(
    source: str,
    wt: Path | None,
    objectives: tuple[str, ...],
    only_pilot: bool,
    worker: int,
    control: Path | None,
) -> None:
    """The full run: one file per candidate and objective under basin_full/, written once.

    Several workers share the list; a candidate is claimed by creating its lock file, so a restart
    skips finished candidates and a crashed worker's lock (removed by the wrapper before a restart)
    is the only state left behind. A worker stops before its next claim when its index is not below
    the worker count in the control file, which is how the run is narrowed from 4 to 2 processes.
    """
    torch.set_num_threads(1)
    FULL_DIR.mkdir(exist_ok=True)
    cand = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    ids = cand["pilot"] if only_pilot else [c["id"] for c in cand[source]]
    ids = [i for i in ids if i.startswith(source + ":")]
    for cid in ids:
        for objective in objectives:
            if control is not None and control.exists():
                if worker >= json.loads(control.read_text(encoding="utf-8"))["workers"]:
                    print(f"worker {worker}: stopped by {control}", flush=True)
                    return
            path = result_path(cid, objective)
            err = path.with_suffix(".error.json")
            lock = path.with_suffix(".lock")
            if path.exists() or err.exists():
                continue
            try:
                lock.open("x").close()
            except FileExistsError:
                continue
            try:
                if path.exists() or err.exists():
                    continue
                if source == "r2a":
                    prob = r2a_problems([cid])[0]
                else:
                    prob = x2e_problems([cid], wt, (objective,))[0]
                row = refine(prob) | {"lo": prob.lo.tolist(), "hi": prob.hi.tolist()}
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(row), encoding="utf-8")
                tmp.replace(path)
                print(
                    f"{row['id']} {row['objective']}: {row['outcome']}, nfev {row['nfev_total']},"
                    f" {row['seconds']:.0f} s, peak {row['peak_mb']:.0f} MB",
                    flush=True,
                )
            except Exception as exc:
                err.write_text(
                    json.dumps({"id": cid, "objective": objective, "error": repr(exc)}),
                    encoding="utf-8",
                )
                print(f"{cid} {objective}: error {exc!r}", flush=True)
            finally:
                lock.unlink(missing_ok=True)


def run(source: str, which: str, wt: Path | None, objectives: tuple[str, ...], out: Path) -> None:
    torch.set_num_threads(1)
    cand = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    ids = cand["pilot"] if which == "pilot" else [c["id"] for c in cand[source]]
    ids = [i for i in ids if i.startswith(source + ":")]
    done = set()
    if out.exists():
        done = {(r["id"], r["objective"]) for r in map(json.loads, out.read_text().splitlines())}
    probs = r2a_problems(ids) if source == "r2a" else x2e_problems(ids, wt, objectives)
    for prob in probs:
        if (prob.cid, prob.objective) in done:
            continue
        row = refine(prob)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"{row['id']} {row['objective']}: {row['outcome']}, nfev {row['nfev_total']},"
            f" {row['seconds']:.0f} s, peak {row['peak_mb']:.0f} MB",
            flush=True,
        )


def smoke(source: str, wt: Path | None, objective: str = "nf") -> None:
    """Runs the machinery from a small displacement of a truth (not a candidate): checks that the
    refinement returns to the truth and that C1-C4 pass there. Prints, writes nothing."""
    torch.set_num_threads(1)
    if source == "r2a":
        cand = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        cid = next(c["id"] for c in cand["r2a"] if c["cond"] == 25)
        prob = r2a_problems([cid])[0]
    else:
        cand = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        cid = next(c["id"] for c in cand["x2e"] if c["n"] == 3)
        prob = x2e_problems([cid], wt, (objective,))[0]
    rng = np.random.default_rng(1)
    w = prob.hi - prob.lo
    prob.x_end = np.clip(prob.theta_t + 1e-3 * w * rng.standard_normal(len(w)), prob.lo, prob.hi)
    prob.recorded = float(stable_terms(prob.full_counts(prob.x_end), prob.lam_t).sum())
    prob.cid = "smoke"
    row = refine(prob)
    print(
        json.dumps(
            {
                k: row[k]
                for k in ("outcome", "dnf_end", "dnf_min", "nfev_total", "seconds", "peak_mb")
            }
            | {
                "checks": row["final_checks"],
                "stages": [
                    {k: s[k] for k in ("nfev", "status", "d_after", "moved", "seconds")}
                    for s in row["stages"]
                ],
            },
            indent=1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["candidates", "run", "smoke"])
    ap.add_argument("--task-a", type=Path)
    ap.add_argument("--source", choices=["r2a", "x2e"])
    ap.add_argument("--set", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--x2e", type=Path)
    ap.add_argument("--objectives", nargs="+", default=["nf", "tr"])
    ap.add_argument("--only-pilot", action="store_true", help="full run, pilot candidates only")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--control", type=Path, help="JSON file {'workers': n}, read between claims")
    args = ap.parse_args()
    if args.source == "x2e" and args.x2e is None:
        ap.error("--source x2e needs --x2e <X2e worktree>")
    if args.part == "candidates":
        candidates(args.task_a)
    elif args.part == "smoke":
        smoke(args.source, args.x2e, args.objectives[0])
    elif args.set == "full":
        objectives = tuple(args.objectives) if args.source == "x2e" else ("nf",)
        full_run(args.source, args.x2e, objectives, args.only_pilot, args.worker, args.control)
    else:
        out = OUT / f"basin_{args.set}.jsonl"
        objectives = tuple(args.objectives) if args.source == "x2e" else ("nf",)
        run(args.source, args.set, args.x2e, objectives, out)


if __name__ == "__main__":
    main()
