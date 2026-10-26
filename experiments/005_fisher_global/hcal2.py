"""H-cal2 of docs/fisher_global/PREREG.md: count dispersion in 100 repeated public XRR scans.

Data: Zenodo 10.5281/zenodo.18548583, `9.95 ms_50 Hz triangle.ort`, read from
$DATA/xrr-public/zenodo-18548583/ (read only). `precheck` runs the registered checks on counts, grid,
normalisation, exposure and drift and computes no per-point dispersion; `dispersion` computes the
registered statistic and must be run after `precheck` has been read.

    .venv/Scripts/python.exe -I experiments/005_fisher_global/hcal2.py precheck
    .venv/Scripts/python.exe -I experiments/005_fisher_global/hcal2.py dispersion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "fisher_global"
DATA = Path(os.environ["DATA"]) / "xrr-public" / "zenodo-18548583" / "9.95 ms_50 Hz triangle.ort"
MD5 = "43c35b4c4c0df452647328fe33e60f36"
COLUMNS = ("Q", "R", "sR", "sQ", "raw", "bg", "N", "af", "etfp")
MIN_MEAN = 10.0


def load() -> list[np.ndarray]:
    """The ORSO file's data sets, each an array with COLUMNS."""
    blob = DATA.read_bytes()
    if hashlib.md5(blob).hexdigest() != MD5:
        raise SystemExit("md5 mismatch")
    sets, rows = [], []
    for line in blob.decode("utf-8").splitlines():
        if line.startswith("# data_set:"):
            if rows:
                sets.append(np.array(rows))
            rows = []
        elif line and not line.startswith("#"):
            rows.append([float(x) for x in line.split()])
    sets.append(np.array(rows))
    return sets


def stack(sets: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Columns as (scan, point) arrays; requires equal lengths."""
    a = np.stack(sets)
    return {name: a[:, :, i] for i, name in enumerate(COLUMNS)}


def rel_spread(x: np.ndarray) -> np.ndarray:
    """Per point: (max − min) / |mean| across scans."""
    m = np.abs(x.mean(0))
    return np.where(m > 0, (x.max(0) - x.min(0)) / np.where(m > 0, m, 1), 0.0)


def precheck() -> None:
    sets = load()
    lengths = sorted({len(s) for s in sets})
    out: dict = {"n_sets": len(sets), "lengths": lengths, "n_columns": sets[0].shape[1]}
    if len(lengths) != 1:
        out["stop"] = "data sets differ in length"
        (OUT / "hcal2_precheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
        return
    c = stack(sets)
    raw = c["raw"]
    out["check1_counts"] = {
        "min_raw": float(raw.min()),
        "max_abs_noninteger": float(np.abs(raw - np.round(raw)).max()),
        "integer": bool(np.abs(raw - np.round(raw)).max() <= 1e-6 and raw.min() >= 0),
    }
    q = c["Q"]
    step = np.gradient(np.median(q, 0))
    spread_q = q.max(0) - q.min(0)
    out["check2_grid"] = {
        "q_range": [float(q.min()), float(q.max())],
        "median_step": float(np.median(step)),
        "max_spread_over_step": float((spread_q / np.abs(step)).max()),
        "fraction_within_0.1_step": float(np.mean(spread_q <= 0.1 * np.abs(step))),
    }
    out["check3_normalisation"] = {}
    for name in ("N", "af", "etfp", "bg"):
        rs = rel_spread(c[name])
        per_scan_const = bool(np.all(rel_spread(c[name].T) <= 1e-6))
        out["check3_normalisation"][name] = {
            "fraction_points_same_across_scans": float(np.mean(rs <= 1e-6)),
            "median_rel_spread_across_scans": float(np.median(rs)),
            "constant_within_each_scan": per_scan_const,
        }
    n_per_scan = c["N"][:, 0]
    out["check3_normalisation"]["N_per_scan"] = {
        "mean": float(n_per_scan.mean()),
        "rel_sd": float(n_per_scan.std(ddof=1) / n_per_scan.mean()),
        "min": float(n_per_scan.min()),
        "max": float(n_per_scan.max()),
    }
    totals = raw.sum(1)
    idx = np.arange(len(totals))
    slope, intercept = np.polyfit(idx, totals, 1)
    resid = totals - (slope * idx + intercept)
    se = np.sqrt(resid @ resid / (len(idx) - 2) / ((idx - idx.mean()) @ (idx - idx.mean())))
    first10 = raw[:, :10].sum(1)
    out["check4_drift"] = {
        "total_raw_mean": float(totals.mean()),
        "slope_per_scan": float(slope),
        "slope_se": float(se),
        "slope_over_mean_per_100_scans": float(100 * slope / totals.mean()),
        "dispersion_index_of_totals": float(totals.var(ddof=1) / totals.mean()),
        "corr_N_vs_first10_raw": float(np.corrcoef(n_per_scan, first10)[0, 1]),
        "corr_N_vs_total_raw": float(np.corrcoef(n_per_scan, totals)[0, 1]),
        "lag1_autocorr_totals": float(np.corrcoef(totals[:-1], totals[1:])[0, 1]),
    }
    out["points_mean_raw_ge_10"] = int(np.sum(raw.mean(0) >= MIN_MEAN))
    out["n_points"] = int(raw.shape[1])
    (OUT / "hcal2_precheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def dispersion() -> None:
    c = stack(load())
    raw = c["raw"]
    n = raw.shape[0]
    mean = raw.mean(0)
    same = np.ones(raw.shape[1], bool)
    for name in ("af", "etfp"):
        same &= rel_spread(c[name]) <= 1e-6
    eligible = (mean >= MIN_MEAN) & same
    phi = raw.var(0, ddof=1) / np.where(mean > 0, mean, 1)
    pe = phi[eligible]
    f_above = float(np.mean(pe > 1.2))
    f_band = float(np.mean((pe >= 0.8) & (pe <= 1.2)))
    med = float(np.median(pe))
    if f_above >= 0.5:
        verdict = "supported"
    elif f_band >= 0.75 and 0.9 <= med <= 1.1:
        verdict = "refuted"
    else:
        verdict = "unresolved"

    totals = raw.sum(1)
    scale = totals / totals.mean()
    mu = raw.sum(0) / scale.sum()
    pearson = ((raw - scale[:, None] * mu) ** 2 / (scale[:, None] * mu)).sum(0) / (n - 1)
    pp = pearson[eligible]

    decades = []
    for lo in range(1, 7):
        sel = eligible & (mean >= 10.0**lo) & (mean < 10.0 ** (lo + 1))
        if sel.any():
            decades.append(
                {
                    "mean_counts": [10.0**lo, 10.0 ** (lo + 1)],
                    "n": int(sel.sum()),
                    "phi_median": float(np.median(phi[sel])),
                    "phi_iqr": np.percentile(phi[sel], [25, 75]).tolist(),
                    "pearson_scaled_median": float(np.median(pearson[sel])),
                    "fraction_phi_gt_1.2": float(np.mean(phi[sel] > 1.2)),
                }
            )
    p_upper = chi2.sf(pe * (n - 1), n - 1)
    out = {
        "n_scans": n,
        "n_points": int(raw.shape[1]),
        "n_eligible": int(eligible.sum()),
        "excluded_af_etfp_differ": int(np.sum((mean >= MIN_MEAN) & ~same)),
        "fraction_phi_gt_1.2": f_above,
        "fraction_phi_in_0.8_1.2": f_band,
        "median_phi": med,
        "fraction_phi_lt_0.8": float(np.mean(pe < 0.8)),
        "verdict": verdict,
        "poisson_expectation": {"in_band": 0.843, "above_1.2": 0.085, "below_0.8": 0.071},
        "fraction_p_upper_lt_0.01": float(np.mean(p_upper < 0.01)),
        "scaled_pearson": {
            "median": float(np.median(pp)),
            "fraction_gt_1.2": float(np.mean(pp > 1.2)),
            "fraction_in_0.8_1.2": float(np.mean((pp >= 0.8) & (pp <= 1.2))),
        },
        "by_decade": decades,
        "q_eligible": [float(c["Q"][0][eligible].min()), float(c["Q"][0][eligible].max())],
        "phi_points": [
            {"q": float(qq), "mean": float(m), "phi": float(f), "pearson_scaled": float(p)}
            for qq, m, f, p in zip(c["Q"][0], mean, phi, pearson, strict=True)
        ],
    }
    (OUT / "hcal2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "phi_points"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["precheck", "dispersion"])
    args = ap.parse_args()
    if args.part == "precheck":
        precheck()
    else:
        dispersion()


if __name__ == "__main__":
    main()
