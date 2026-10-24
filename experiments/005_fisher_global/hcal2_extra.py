"""Post-hoc description of H-cal2's overdispersion (outside the registered verdict).

Splits the excess variance s² − x̄ of each point's raw counts into a term proportional to x̄²
(multiplicative intensity noise within a scan) and one proportional to (dx̄/dq)² (jitter of the
angle a frame is assigned to), by non-negative least squares. Writes docs/fisher_global/hcal2_extra.json.

    .venv/Scripts/python.exe -I experiments/005_fisher_global/hcal2_extra.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hcal2 import OUT, load, stack  # noqa: E402

c = stack(load())
raw, q = c["raw"], c["Q"][0]
n = raw.shape[0]
mean = raw.mean(0)
var = raw.var(0, ddof=1)
excess = var - mean
slope = np.gradient(mean, q)
# Var of s² for Poisson-like counts is about 2 var² / (n − 1); weight each point by its inverse.
w = 1 / np.sqrt(2 / (n - 1)) / var
design = np.column_stack([mean**2, slope**2])


def fit(cols: list[int]) -> dict:
    coef, _ = nnls(design[:, cols] * w[:, None], excess * w)
    pred = design[:, cols] @ coef
    r = (excess - pred) * w
    return {
        "columns": [["mean^2", "slope^2"][i] for i in cols],
        "coef": coef.tolist(),
        "weighted_chi2_per_point": float(r @ r / len(r)),
        "weighted_r2": float(1 - (r @ r) / ((excess * w - np.average(excess * w)) ** 2).sum()),
    }


fits = {"mean^2": fit([0]), "slope^2": fit([1]), "both": fit([0, 1])}
both = fits["both"]["coef"]
share_slope = both[1] * slope**2 / np.maximum(both[0] * mean**2 + both[1] * slope**2, 1e-300)
out = {
    "fits": fits,
    "relative_intensity_noise_sd": float(np.sqrt(fits["mean^2"]["coef"][0])),
    "q_jitter_sd_per_A": float(np.sqrt(both[1])) if both[1] > 0 else 0.0,
    "median_share_of_excess_from_slope_term": float(np.median(share_slope)),
    "spearman_excess_over_mean2_vs_dlnmean_dq2": float(
        spearmanr(excess / mean**2, (slope / mean) ** 2).statistic
    ),
    "median_step_q": float(np.median(np.diff(q))),
}
(OUT / "hcal2_extra.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
