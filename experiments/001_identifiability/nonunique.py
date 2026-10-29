"""Experiment 001, R1c: is a wrong film from a fit a second solution or a failed fit?

Decision rules are pre-registered in docs/identifiability/README.md ("R1c pre-registration"); this
script computes what those rules need, per case:

    profile  profile likelihood in d on noise-free data: Poisson deviance minimized over the other
             free parameters at fixed d (scipy least_squares on deviance residuals, autograd Jacobian)
    fits     one Poisson draw; refine.refine from 15 starts plus one from the truth; the Poisson
             deviance of every endpoint against the draw
    mcmc     refine.mcmc.sample_posterior from the lowest-deviance endpoint (needs the mcmc extra)

The forward model is fim.py's, with Cu Kα2 and the footprint switched on, on a lab-like q grid.

Run:  uv run python experiments/001_identifiability/nonunique.py profile fits --case mid_oxide
      uv run --group figures python experiments/001_identifiability/nonunique.py report
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import fim
import numpy as np
import torch
from fim import FILM, INSTR, OXIDE, Film
from scipy.optimize import least_squares

from reflectolearn.refine.pipeline import RefineConfig, make_forward, refine

S_COUNTS = 1e6
Q = np.linspace(fim.tth_to_q(np.array(0.2)), fim.tth_to_q(np.array(6.0)), 291)
EXTRA = {"kalpha2_ratio": 0.5, "footprint": 500.0}
REFINE_FREE = FILM + INSTR + ("kalpha2_ratio", "footprint") + OXIDE
PARAM_RANGES = SimpleNamespace(thickness=(10.0, 1200.0), roughness=(0.0, 15.0), sld=(5.0, 100.0))
_rc = RefineConfig()
BOUNDS = {
    "thickness": PARAM_RANGES.thickness,
    "roughness": PARAM_RANGES.roughness,
    "sld": PARAM_RANGES.sld,
    "i0": _rc.i0_bounds,
    "bkg": _rc.bkg_bounds,
    "tth_offset": _rc.tth_offset_bounds,
    "dq": _rc.dq_bounds,
    "kalpha2_ratio": _rc.kalpha2_ratio_bounds,
    "footprint": _rc.footprint_bounds,
    "oxide_thickness": _rc.oxide_thickness_bounds,
    "oxide_sld": _rc.oxide_sld_bounds,
    "oxide_roughness": _rc.oxide_roughness_bounds,
}
NO_OXIDE_FREE = tuple(k for k in REFINE_FREE if k not in OXIDE)
# name: (film, free parameters); the truth has the top oxide exactly when the oxide is free.
CASES = {
    "mid_oxide": (Film("mid", 240.0, 6.0, 45.0), REFINE_FREE),
    "mid_no_oxide": (Film("mid", 240.0, 6.0, 45.0), NO_OXIDE_FREE),
    "t2_low": (Film("t2_low", 200.0, 4.0, 12.0), REFINE_FREE),
}
REFINE_INIT = {
    "i0": 1.0,
    "bkg": _rc.bkg_init,
    "tth_offset": 0.0,
    "dq": _rc.dq_init,
    "kalpha2_ratio": _rc.kalpha2_ratio_init,
    "footprint": _rc.footprint_init,
    "oxide_thickness": _rc.oxide_thickness_init,
    "oxide_sld": _rc.oxide_sld_init,
    "oxide_roughness": _rc.oxide_roughness_init,
}
# Relative steps of d away from the truth: fine near it, coarse out to ±29 %.
DELTAS = np.r_[
    np.arange(0.0025, 0.05, 0.0025), np.arange(0.05, 0.15, 0.005), np.arange(0.15, 0.31, 0.02)
]
OUT = Path("docs/identifiability")


def case_truth(name: str) -> dict[str, float]:
    film, free = CASES[name]
    return fim.truth(film, "oxide" if "oxide_sld" in free else "instr") | EXTRA


def counts(p: dict[str, float], s: float = S_COUNTS) -> np.ndarray:
    pt = {k: torch.tensor(v) for k, v in p.items()}
    with torch.no_grad():
        return fim.expected_counts(torch.as_tensor(Q), pt, s, fim.effects_of(p)).numpy()


def deviance_terms(lam: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-point Poisson deviance 2 (y ln(y/lam) - (y - lam)), y = 0 allowed."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / lam), 0.0) - (y - lam)
    return 2 * np.maximum(t, 0.0)


def deviance(p: dict[str, float], y: np.ndarray) -> float:
    return float(deviance_terms(counts(p), y).sum())


def fit(y: np.ndarray, p_start: dict[str, float], free: tuple[str, ...]) -> tuple[dict, float]:
    """Minimize the Poisson deviance against counts y over `free`, within BOUNDS."""
    lo = np.array([BOUNDS[k][0] for k in free])
    hi = np.array([BOUNDS[k][1] for k in free])
    x0 = np.clip([p_start[k] for k in free], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    cache: dict = {}

    def model(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = fim.jacobian(Q, p_start | dict(zip(free, x, strict=True)), free, S_COUNTS)
        return cache[key]

    def resid(x):
        lam, _ = model(x)
        return np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))

    def jac(x):
        lam, jlog = model(x)
        r = np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))
        near = np.abs(lam - y) <= 1e-6 * np.maximum(y, 1e-12)
        g = np.where(near, 1 / np.sqrt(lam), (1 - y / lam) / np.where(near, 1.0, r))
        return (g * lam)[:, None] * jlog

    res = least_squares(
        resid, x0, jac=jac, bounds=(lo, hi), x_scale="jac", xtol=1e-10, ftol=1e-10, max_nfev=100
    )
    p = p_start | dict(zip(free, res.x.tolist(), strict=True))
    return p, float(2 * res.cost)


# ------------------------------------------------------------------ profile likelihood
def run_profile(name: str) -> dict:
    film, free = CASES[name]
    p0 = case_truth(name)
    y = counts(p0)
    others = tuple(k for k in free if k != "thickness")
    lam, J = fim.jacobian(Q, p0, free, S_COUNTS)
    fi = fim.fisher(lam, J, free)
    points = [{"delta": 0.0, "dchi2": 0.0, "params": p0}]
    for sign in (1, -1):
        p = p0
        for delta in DELTAS:
            p, d = fit(y, p | {"thickness": film.thickness * (1 + sign * delta)}, others)
            points.append({"delta": float(sign * delta), "dchi2": d, "params": p})
    points.sort(key=lambda r: r["delta"])
    return {
        "case": name,
        "film": film.__dict__,
        "free": list(free),
        "s": S_COUNTS,
        "crlb_d_pct": 100 * fi.crlb[0] / film.thickness,
        "gamma": fi.gamma,
        "weakest": fi.weakest_str(),
        "profile": points,
        **label(points),
    }


def label(points: list[dict], threshold: float = 1.0) -> dict:
    """Pre-registered labels from a profile at s = 1e6."""
    deltas = np.array([r["delta"] for r in points])
    chi = np.array([r["dchi2"] for r in points])
    i0 = int(np.argmin(np.abs(deltas)))
    lo = hi = i0
    while lo > 0 and chi[lo - 1] <= threshold:
        lo -= 1
    while hi < len(chi) - 1 and chi[hi + 1] <= threshold:
        hi += 1
    second = [
        float(deltas[i])
        for i in range(1, len(chi) - 1)
        if not lo <= i <= hi
        and chi[i] <= threshold
        and chi[i] <= chi[i - 1]
        and chi[i] <= chi[i + 1]
    ]
    half_width = max(-deltas[lo], deltas[hi])
    if second:
        verdict = "multimodal"
    elif half_width >= 0.02:
        verdict = "ridge"
    else:
        verdict = "identifiable"
    far = lo if -deltas[lo] >= deltas[hi] else hi
    return {
        "interval": [float(deltas[lo]), float(deltas[hi])],
        "second_minima": second,
        "label": verdict,
        "pair_delta": float(deltas[far]),
    }


# ------------------------------------------------------------------ fits on one noisy curve
def run_fits(name: str, seed: int = 0) -> dict:
    film, free = CASES[name]
    p0 = case_truth(name)
    rng = np.random.default_rng(seed)
    y = rng.poisson(counts(p0)).astype(float)
    mask = y > 0
    log_r = np.where(mask, np.log10(np.where(mask, y, 1.0) / S_COUNTS), -15.0)

    starts = [
        (film.thickness * fd, 0.8 * film.roughness, film.sld * fr)
        for fd in (0.8, 0.9, 1.0, 1.1, 1.2)
        for fr in (0.85, 1.0, 1.15)
    ] + [(film.thickness, film.roughness, film.sld)]
    labels = [f"d×{fd:g} ρ×{fr:g}" for fd in (0.8, 0.9, 1.0, 1.1, 1.2) for fr in (0.85, 1.0, 1.15)]
    labels.append("truth")
    B = len(starts)
    cfg = RefineConfig()
    if not any(k in free for k in OXIDE):
        cfg.fit_oxide = False
    t = time.perf_counter()
    _, info = refine(
        torch.tensor(starts, dtype=torch.float32),
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]] * B, dtype=torch.float32),
        torch.as_tensor(Q, dtype=torch.float32),
        torch.as_tensor(log_r, dtype=torch.float32).expand(B, -1),
        torch.as_tensor(mask, dtype=torch.float32).expand(B, -1),
        PARAM_RANGES,
        wavelength=fim.WAVELENGTH,
        config=cfg,
    )
    seconds = time.perf_counter() - t

    endpoints = []
    for b in range(B):
        p = dict(p0)
        for k in free:
            p[k] = float(info["state"][k][b])
        if not cfg.fit_oxide:
            p |= {k: p0[k] for k in OXIDE}
        endpoints.append({"start": labels[b], "params": p, "deviance": deviance(p, y)})

    # The Poisson maximum-likelihood fit from the truth, as a reference for D*.
    p_ml, d_ml = fit(y, p0, free)
    d_star = min([d_ml] + [e["deviance"] for e in endpoints])
    for e in endpoints:
        dd = e["deviance"] - d_star
        e["delta_D"] = dd
        e["class"] = "valley" if dd <= 4 else ("failure" if dd > 25 else "undecided")

    # The same points scored by refine's own forward model (float32, grid smearing) and its own
    # objective (log10 MSE over the nonzero counts), to tell a failed search from a model difference.
    fwd = make_forward(
        torch.as_tensor(Q, dtype=torch.float32),
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]], dtype=torch.float32),
        wavelength=fim.WAVELENGTH,
        config=cfg,
        device=torch.device("cpu"),
    )
    points = [p0, p_ml] + [e["params"] for e in endpoints]
    vals = {k: torch.tensor([p[k] for p in points], dtype=torch.float32) for k in p0}
    with torch.no_grad():
        r = fwd(vals, info["active"]).double().numpy()
    log_mse = (((np.log10(r) - log_r) ** 2) * mask).sum(1) / mask.sum()
    d_refine = deviance_terms(r * S_COUNTS, y).sum(1)
    for e, lm, dr in zip(endpoints, log_mse[2:], d_refine[2:], strict=True):
        e["log_mse_refine_model"] = float(lm)
        e["deviance_refine_model"] = float(dr)
    return {
        "case": name,
        "seed": seed,
        "refine_seconds": seconds,
        "deviance_truth": deviance(p0, y),
        "refine_model": {
            "log_mse_truth": float(log_mse[0]),
            "log_mse_ml": float(log_mse[1]),
            "deviance_truth": float(d_refine[0]),
            "deviance_ml": float(d_refine[1]),
        },
        "ml": {"params": p_ml, "deviance": d_ml},
        "d_star": d_star,
        "endpoints": endpoints,
    }


def noisy_counts(name: str, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(counts(case_truth(name))).astype(float)


def fit_starts(film: Film) -> list[tuple[str, tuple[float, float, float]]]:
    """The registered starts: d0 x {0.8..1.2} x rho0 x {0.85, 1, 1.15}, sigma 0.8 sigma0."""
    return [
        (f"d×{fd:g} ρ×{fr:g}", (film.thickness * fd, 0.8 * film.roughness, film.sld * fr))
        for fd in (0.8, 0.9, 1.0, 1.1, 1.2)
        for fr in (0.85, 1.0, 1.15)
    ]


def run_lsq(name: str, seed: int = 0) -> dict:
    """Not registered: the Poisson least-squares fit from the same starts as refine, nuisances at
    refine's initial values."""
    film, free = CASES[name]
    p0 = case_truth(name)
    y = noisy_counts(name, seed)
    out = []
    for label_, (d, sigma, rho) in fit_starts(film):
        start = p0 | {k: v for k, v in REFINE_INIT.items() if k in free}
        start |= {"thickness": d, "roughness": sigma, "sld": rho}
        p, dev = fit(y, start, free)
        out.append({"start": label_, "params": p, "deviance": dev})
    return {"seed": seed, "endpoints": out}


# ------------------------------------------------------------------ MCMC
def run_mcmc(name: str, steps: int, start: str, seed: int = 0) -> dict:
    from reflectolearn.refine.mcmc import sample_posterior

    saved = json.loads((OUT / f"nonunique_{name}.json").read_text(encoding="utf-8"))
    fits = saved["fits"]
    _film, free = CASES[name]
    p0 = case_truth(name)
    rng = np.random.default_rng(fits["seed"])
    y = rng.poisson(counts(p0)).astype(float)
    mask = y > 0
    log_r = np.where(mask, np.log10(np.where(mask, y, 1.0) / S_COUNTS), -15.0)
    # "refine": the lowest-deviance refine endpoint (as registered); "ml": the Poisson ML fit
    if start == "refine":
        best = min(fits["endpoints"], key=lambda e: e["deviance"])["params"]
    else:
        best = fits["ml"]["params"]

    cfg = RefineConfig()
    if not any(k in free for k in OXIDE):
        cfg.fit_oxide = False
    active = {"tth_offset", "dq", "kalpha2", "footprint"} | ({"oxide"} if cfg.fit_oxide else set())
    fitted = [k for k in REFINE_FREE if cfg.fit_oxide or k not in OXIDE]
    info = {
        "state": {k: torch.tensor([best[k]], dtype=torch.float32) for k in REFINE_FREE},
        "active": active,
        "bounds": {k: BOUNDS[k] for k in fitted},
        "fitted": fitted,
    }
    t = time.perf_counter()
    post = sample_posterior(
        info,
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]]),
        torch.as_tensor(Q, dtype=torch.float32),
        torch.as_tensor(log_r, dtype=torch.float32)[None],
        torch.as_tensor(mask, dtype=torch.float32)[None],
        wavelength=fim.WAVELENGTH,
        config=cfg,
        steps=steps,
        seed=seed,
    )
    seconds = time.perf_counter() - t
    d = post["samples"][:, post["keys"].index("thickness")]
    return {
        "start": start,
        "start_thickness": best["thickness"],
        "steps": steps,
        "seconds": seconds,
        "walkers": post["walkers"],
        "acceptance": post["acceptance"],
        "params": post["params"],
        "d_68": [float(np.percentile(d, 16)), float(np.percentile(d, 84))],
        "log_prob_start": post["log_prob_start"],
        "log_prob_max": post["log_prob_max"],
    }


def half_width(points: list[dict], threshold: float = 1.0) -> tuple[float, float]:
    """Relative d at which the profile crosses `threshold` on each side: linear in Δχ² between grid
    points, quadratic from the truth when the first grid point is already above it."""
    out = []
    for sign in (-1, 1):
        side = sorted((r for r in points if sign * r["delta"] >= 0), key=lambda r: abs(r["delta"]))
        cross = float("nan")
        for a, b in itertools.pairwise(side):
            if a["dchi2"] <= threshold < b["dchi2"]:
                if a["delta"] == 0:
                    cross = abs(b["delta"]) * (threshold / b["dchi2"]) ** 0.5
                else:
                    t = (threshold - a["dchi2"]) / (b["dchi2"] - a["dchi2"])
                    cross = abs(a["delta"]) + t * (abs(b["delta"]) - abs(a["delta"]))
                break
        out.append(cross)
    return out[0], out[1]


def summary(name: str) -> dict:
    data = json.loads((OUT / f"nonunique_{name}.json").read_text(encoding="utf-8"))
    pr, fits, lsq = data["profile"], data["fits"], data["lsq"]
    d0 = pr["film"]["thickness"]
    d_star = min([fits["d_star"]] + [e["deviance"] for e in lsq["endpoints"]])
    lm_ml = fits["refine_model"]["log_mse_ml"]
    lo, hi = half_width(pr["profile"])
    row = {
        "case": name,
        "label": pr["label"],
        "profile_pct": (100 * lo, 100 * hi),
        "crlb_pct": pr["crlb_d_pct"],
        "gamma": pr["gamma"],
        "refine_failure": sum(e["class"] == "failure" for e in fits["endpoints"]),
        "refine_n": len(fits["endpoints"]),
        "refine_min_dD": min(e["delta_D"] for e in fits["endpoints"]),
        "refine_worse_own": sum(e["log_mse_refine_model"] > lm_ml for e in fits["endpoints"]),
        "refine_d_pct": [100 * (e["params"]["thickness"] / d0 - 1) for e in fits["endpoints"]],
        "lsq_at_ml": sum(e["deviance"] - d_star <= 4 for e in lsq["endpoints"]),
        "lsq_n": len(lsq["endpoints"]),
        "ml_d_pct": 100 * (fits["ml"]["params"]["thickness"] / d0 - 1),
    }
    for st in ("refine", "ml"):
        m = data.get(f"mcmc_{st}")
        if m:
            row[f"mcmc_{st}_pct"] = [100 * (x / d0 - 1) for x in m["d_68"]]
    return row


def report() -> None:
    rows = [summary(n) for n in CASES]
    lines = [
        "# R1c tables (generated by nonunique.py report; do not edit)",
        "",
        (
            "d in % of the true value. Profile: where the noise-free profile crosses Δχ² = 1 at"
            " s = 1e6 (interpolated). CRLB: unbounded local bound, same model. refine failure:"
            " endpoints with ΔD > 25 / all. Worse on own objective: endpoints whose log10 MSE under"
            " refine's own model is above that of the Poisson ML parameters. LSQ at ML: Poisson"
            " least-squares fits from the same starts that end within ΔD ≤ 4 of the best. MCMC: 68 %"
            " interval of refine.mcmc, started from the best refine endpoint / from the ML fit."
        ),
        "",
        (
            "| case | label | profile Δχ²≤1 | CRLB | γ | refine failure | worse on own objective"
            " | refine d range | LSQ at ML | ML d | MCMC (refine start) | MCMC (ML start) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["profile_pct"]
        mc = [r.get(f"mcmc_{st}_pct") for st in ("refine", "ml")]
        mc_s = [f"{m[0]:+.2f} to {m[1]:+.2f}" if m else "—" for m in mc]
        lines.append(
            f"| {r['case']} | {r['label']} | −{lo:.2f} / +{hi:.2f} | ±{r['crlb_pct']:.2f} "
            f"| {r['gamma']:.0f} | {r['refine_failure']}/{r['refine_n']} "
            f"(min ΔD {r['refine_min_dD']:.3g}) | {r['refine_worse_own']}/{r['refine_n']} "
            f"| {min(r['refine_d_pct']):+.1f} to {max(r['refine_d_pct']):+.1f} "
            f"| {r['lsq_at_ml']}/{r['lsq_n']} | {r['ml_d_pct']:+.2f} | {mc_s[0]} | {mc_s[1]} |"
        )
    (OUT / "nonunique_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    figure()


def sld_profile(p: dict[str, float], z: np.ndarray) -> np.ndarray:
    """SLD against depth z below the surface (Å) for the stack fim.reflectivity builds, with each
    interface drawn as an error function of its roughness. For the picture only."""
    from scipy.special import erf

    if p["oxide_thickness"]:
        rho = [0.0, p["oxide_sld"], p["sld"], fim.SUB_SLD]
        tops = [0.0, p["oxide_thickness"], p["oxide_thickness"] + p["thickness"]]
        sigma = [p["oxide_roughness"], p["roughness"], fim.SUB_ROUGH]
    else:
        rho = [0.0, p["sld"], fim.SUB_SLD]
        tops = [0.0, p["thickness"]]
        sigma = [p["roughness"], fim.SUB_ROUGH]
    out = np.full_like(z, rho[0])
    for j, (z0, sg) in enumerate(zip(tops, sigma, strict=True)):
        out += (rho[j + 1] - rho[j]) * 0.5 * (1 + erf((z - z0) / (np.sqrt(2) * sg)))
    return out


def figure() -> None:
    """nonunique.png (README) and nonunique.pdf (paper), both 7.2 in wide, from the saved JSON.

    Panel (b) evaluates the forward model at saved parameters and redraws the seeded Poisson curve;
    nothing is fitted again.
    """
    import cvdkit
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import qoraplot

    data = {n: json.loads((OUT / f"nonunique_{n}.json").read_text(encoding="utf-8")) for n in CASES}
    pick = cvdkit.pick(3)
    look = {
        n: {"color": c, "marker": m, "ls": ls}
        for n, c, m, ls in zip(CASES, pick.colors, pick.markers, pick.linestyles, strict=True)
    }
    grey, light = "#52514e", "#b5b4ae"
    with qoraplot.style("house", "paper"):
        fig = plt.figure(figsize=(qoraplot.WIDTHS["double"], 6.2), layout="constrained")
        top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.1])
        ax = top.subplot_mosaic(
            [["a", "b"], ["z", "r"]], width_ratios=[1, 2.1], height_ratios=[2, 1]
        )
        a, zoom, b, r = ax["a"], ax["z"], ax["b"], ax["r"]
        b.sharex(r)
        b.tick_params(labelbottom=False)
        c, d = bottom.subplots(1, 2)

        # (a) the three true films, and below them the top 40 Å, where the 10 Å oxide shows
        z = np.linspace(-30.0, 300.0, 1200)
        for n in CASES:
            rho = sld_profile(case_truth(n), z)
            for axis in (a, zoom):
                axis.plot(z, rho, label=n, color=look[n]["color"], ls=look[n]["ls"], marker="none")
        zoom.set_xlim(-15, 40)
        zoom.set_ylim(0, 48)
        zoom.annotate(
            "10 Å oxide", xy=(5, 18.5), xytext=(18, 4), fontsize=7, arrowprops={"arrowstyle": "-"}
        )
        zoom.set_xlabel("z (Å), top 40 Å")
        zoom.set_ylabel("SLD")
        a.set_xlabel("depth below the surface z (Å)")
        a.set_ylabel(r"SLD ($10^{-6}$ Å$^{-2}$)")
        a.set_title("true films (schematic)")
        a.text(-27, 38, "air", fontsize=7, color=grey)
        a.text(266, 22, "Si", fontsize=7, color=grey)
        a.legend(loc="center", bbox_to_anchor=(0.55, 0.6), handlelength=2.2)
        qoraplot.panel_label(a, "a")

        # (b) one Poisson curve of mid_oxide, the bounded ML fit, the best refine endpoint and the
        # pre-registered pair point, with their residuals
        n = "mid_oxide"
        fits = data[n]["fits"]
        y = noisy_counts(n, fits["seed"])
        ml = fits["ml"]["params"]
        # The draw is regenerated, and numpy may change its Poisson stream between versions.
        if not np.isclose(deviance(ml, y), fits["ml"]["deviance"], rtol=1e-6, atol=0):
            raise RuntimeError("the regenerated Poisson curve is not the one the fits used")
        best_refine = min(fits["endpoints"], key=lambda e: e["deviance"])
        pair = next(
            q
            for q in data[n]["profile"]["profile"]
            if q["delta"] == data[n]["profile"]["pair_delta"]
        )
        zero = y == 0
        b.semilogy(Q[~zero], y[~zero], ".", ms=2.5, color=light, label="Poisson counts (s = 10⁶)")
        b.semilogy(
            Q[zero],
            np.full(zero.sum(), 0.5),
            "v",
            ms=2.5,
            color=light,
            label=f"zero counts ({zero.sum()}), drawn at 0.5",
        )
        curves = (
            (f"bounded Poisson ML, d = {ml['thickness']:.1f} Å", ml, "#3b3b3b", "-", 1.2),
            (
                (
                    f"best refine endpoint (start: {best_refine['start']}),"
                    f" d = {best_refine['params']['thickness']:.1f} Å"
                ),
                best_refine["params"],
                "#cc79a7",
                (0, (5, 2)),
                1.2,
            ),
            (
                (
                    f"pre-registered pair: profile point Δχ² = {pair['dchi2']:.2f},"
                    f" d = {pair['params']['thickness']:.1f} Å"
                ),
                pair["params"],
                grey,
                (0, (6, 2, 1, 2)),
                0.8,
            ),
        )
        for label_, pp, col, ls, lw in curves:
            lam = counts(pp)
            b.semilogy(Q, lam, color=col, ls=ls, lw=lw, marker="none", label=label_)
            r.plot(
                Q,
                (y - lam) / np.sqrt(np.maximum(lam, 1.0)),
                color=col,
                ls=ls,
                lw=0.9 * lw,
                marker="none",
            )
        b.set_ylabel("counts")
        b.set_title("mid_oxide, one Poisson curve (seed 0; models recomputed)")
        b.legend(loc="upper right", fontsize=7)
        r.axhline(0, color=grey, lw=0.6, ls=(0, (1, 1.5)))
        r.set_yscale("symlog", linthresh=5)
        r.set_ylim(-200, 200)
        r.set_yticks([-100, -10, 0, 10, 100], ["−100", "−10", "0", "10", "100"])
        r.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        r.set_xlabel(r"q (Å$^{-1}$)")
        r.set_ylabel("(y − λ)/√max(λ, 1)")
        qoraplot.panel_label(b, "b")

        # (c) the profile of d over its whole range against the local Fisher parabola; the truth
        # point (Δχ² = 0) is left out of the lines, which would otherwise dive to the axis limit
        span = 100 * max(abs(q["delta"]) for q in data["mid_oxide"]["profile"]["profile"])
        x = np.linspace(-span, span, 2001)
        lo_x, hi_x = -40.0, 100.0
        for k, (n, dd) in enumerate(data.items()):
            pr = dd["profile"]
            st = look[n]
            # each side of the truth separately, joined by a gap
            left = [q for q in pr["profile"] if q["delta"] < 0]
            right = [q for q in pr["profile"] if q["delta"] > 0]
            c.plot(
                [100 * q["delta"] for q in left] + [np.nan] + [100 * q["delta"] for q in right],
                [q["dchi2"] for q in left] + [np.nan] + [q["dchi2"] for q in right],
                color=st["color"],
                marker=st["marker"],
                ls="-",
                ms=2.5,
                label=f"{n} profile",
            )
            c.plot(
                x,
                (x / pr["crlb_d_pct"]) ** 2,
                color=st["color"],
                ls=(0, (1, 1.5)),
                lw=1.0,
                marker="none",
            )
            lo, hi = half_width(pr["profile"])
            c.plot(
                [-100 * lo, -100 * lo, np.nan, 100 * hi, 100 * hi],
                [0.4, 2.5, np.nan, 0.4, 2.5],
                color=st["color"],
                lw=1.6,
                ls="-",
                marker="none",
            )
            # endpoint d as rug ticks: refine on the lower row, Poisson LSQ above it
            d0 = pr["film"]["thickness"]
            for row, key, mk, size in ((0.02, "fits", "|", 5), (0.065, "lsq", "v", 2.5)):
                c.plot(
                    [100 * (e["params"]["thickness"] / d0 - 1) for e in dd[key]["endpoints"]],
                    np.full(len(dd[key]["endpoints"]), row + 0.012 * k),
                    mk,
                    ms=size,
                    mew=0.9,
                    color=st["color"],
                    transform=c.get_xaxis_transform(),
                )
        c.plot(
            [],
            [],
            color=grey,
            ls=(0, (1, 1.5)),
            lw=1.0,
            marker="none",
            label="local Fisher (CRLB) parabola",
        )
        c.plot([], [], color=grey, lw=1.6, marker="none", label="Δχ² = 1 crossing (table)")
        c.plot([], [], "|", ms=5, mew=0.9, color=grey, label="refine endpoint d (bottom row)")
        c.plot([], [], "v", ms=2.5, color=grey, label="Poisson LSQ endpoint d")
        c.axvspan(lo_x, -span, color="#efeeea", lw=0)
        c.axvspan(span, hi_x, color="#efeeea", lw=0)
        c.text(span * 1.1, 10, f"no profile\nbeyond\n±{span:.0f} %", fontsize=7, color=grey)
        c.axhline(1, color=grey, lw=0.6, ls=(0, (5, 2)))
        c.axvline(0, color=grey, lw=0.6, ls=(0, (3, 3)))
        c.text(0.06, 4, "truth: Δχ² = 0", fontsize=7, color=grey, rotation=90, va="bottom")
        c.set_xscale("symlog", linthresh=1)
        ticks = [-30, -10, -3, -1, 0, 1, 3, 10, 30, 100]
        c.set_xticks(ticks, [f"{t:+d}" if t else "0" for t in ticks])
        c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        c.set_xlim(lo_x, hi_x)
        c.set_yscale("log")
        c.set_ylim(1e-4, 1e10)
        c.set_xlabel("d − d_true (% of d_true, linear within ±1 %), other parameters re-fitted")
        c.set_ylabel("Δχ² (Poisson deviance, noise-free)")
        c.set_title("profile likelihood of d, noise-free curves")
        c.legend(loc="upper center", fontsize=7, ncol=2)
        qoraplot.panel_label(c, "c")

        # (d) where the fits stop on one Poisson curve per case
        at_best = []
        for n, dd in data.items():
            d0 = dd["profile"]["film"]["thickness"]
            d_star = min([dd["fits"]["d_star"]] + [e["deviance"] for e in dd["lsq"]["endpoints"]])
            at_best.append(sum(e["deviance"] - d_star <= 4 for e in dd["lsq"]["endpoints"]))
            lm_ml = dd["fits"]["refine_model"]["log_mse_ml"]
            st = look[n]
            for key, filled in (("fits", False), ("lsq", True)):
                ends = dd[key]["endpoints"]
                d.scatter(
                    [100 * (e["params"]["thickness"] / d0 - 1) for e in ends],
                    [e["deviance"] - d_star + 1 for e in ends],
                    marker=st["marker"],
                    s=18,
                    facecolors=st["color"] if filled else "none",
                    edgecolors=st["color"],
                    linewidths=0.9,
                    label=f"{n}, {'refine' if key == 'fits' else 'Poisson LSQ'}",
                )
            own = [e for e in dd["fits"]["endpoints"] if e["log_mse_refine_model"] <= lm_ml]
            d.scatter(
                [100 * (e["params"]["thickness"] / d0 - 1) for e in own],
                [e["deviance"] - d_star + 1 for e in own],
                marker="x",
                s=10,
                color=st["color"],
                linewidths=0.8,
            )
        d.scatter(
            [], [], marker="x", s=10, color=grey, label="refine endpoint its own objective prefers"
        )
        d.annotate(
            "LSQ at the best fit: " + " / ".join(f"×{k}" for k in at_best),
            xy=(0, 1.2),
            xytext=(8, 1.6),
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": grey, "lw": 0.6},
        )
        d.axhline(5, color=grey, lw=0.6)
        d.axhline(26, color=grey, lw=0.6, ls=(0, (5, 2)))
        d.text(
            0.99,
            5.6,
            "ΔD = 4",
            fontsize=7,
            color=grey,
            ha="right",
            transform=d.get_yaxis_transform(),
        )
        d.text(
            0.99,
            29,
            "ΔD = 25 (failure)",
            fontsize=7,
            color=grey,
            ha="right",
            transform=d.get_yaxis_transform(),
        )
        d.axvline(0, color=grey, lw=0.6, ls=(0, (1, 1.5)))
        d.set_yscale("log")
        d.set_ylim(0.5, 1e8)
        d.set_xlabel("d of the endpoint − d_true (% of d_true)")
        d.set_ylabel("ΔD + 1 above the best fit")
        d.set_title("fit endpoints, one Poisson curve per case (seed 0)")
        d.legend(loc="center right", fontsize=7, ncol=1)
        qoraplot.panel_label(d, "d")

        if collisions := cvdkit.check(fig):
            raise RuntimeError(f"colours not distinguishable under CVD simulation: {collisions}")
        fig.savefig(OUT / "nonunique.png", dpi=200)
        fig.savefig(OUT / "nonunique.pdf")
    plt.close(fig)


def save(name: str, part: str, result: dict) -> None:
    path = OUT / f"nonunique_{name}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[part] = result
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parts", nargs="+", choices=["profile", "fits", "lsq", "mcmc", "report"])
    ap.add_argument("--case", choices=list(CASES))
    ap.add_argument("--mcmc-steps", type=int, default=2000)
    ap.add_argument("--mcmc-start", choices=["refine", "ml"], default="refine")
    args = ap.parse_args()
    torch.set_default_dtype(fim.DEFAULT_DTYPE)
    for part in args.parts:
        t = time.perf_counter()
        if part == "report":
            report()
            continue
        if part == "profile":
            result = run_profile(args.case)
        elif part == "fits":
            result = run_fits(args.case)
        elif part == "lsq":
            result = run_lsq(args.case)
        else:
            result = run_mcmc(args.case, args.mcmc_steps, args.mcmc_start)
            part = f"mcmc_{args.mcmc_start}"
        save(args.case, part, result)
        print(f"{args.case} {part}: {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
