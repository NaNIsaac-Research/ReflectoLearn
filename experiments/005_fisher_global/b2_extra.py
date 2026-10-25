"""Post-hoc reading of B-2 (outside the registration): where the Fisher profile of each film
parameter of R2a #25 sends the other parameters, and whether that leaves the bounds.

Under the quadratic D = Δθᵀ F Δθ, the profile at Δθ_k = h_k = √((F⁻¹)_kk) sets the others to
−F_oo⁻¹ F_ok h_k. If that point is outside the box, the bounded profile must be narrower than h_k.
The deviance there is also computed with the forward model, to see the nonlinearity.

    .venv/Scripts/python.exe -I experiments/005_fisher_global/b2_extra.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "001_identifiability"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan
from b1 import OUT, counts, deviance_terms

CONDITION = 25
data = json.loads(
    (ROOT / "docs" / "identifiability" / "scan" / f"fits_{CONDITION:02d}.json").read_text(
        encoding="utf-8"
    )
)
p0, free = data["truth"], tuple(data["free"])
_, _, bounds = scan.setup(data["condition"])
q = scan.q_grid(data["condition"]["tth_max"])
lam_t = counts(q, p0)
b1 = json.loads((OUT / "b1.json").read_text(encoding="utf-8"))
fisher = np.array(next(c for c in b1["conditions"] if c["index"] == CONDITION)["fisher"])
cov = np.linalg.inv(fisher)
theta_t = np.array([p0[k] for k in free])
lo = np.array([bounds[k][0] for k in free])
hi = np.array([bounds[k][1] for k in free])

out = {}
for k in scan.film_keys(free):
    i = free.index(k)
    out[k] = {}
    for sign in (1, -1):
        dtheta = sign * cov[:, i] / np.sqrt(cov[i, i])
        theta = theta_t + dtheta
        outside = {
            free[j]: [float(theta[j]), [float(lo[j]), float(hi[j])]]
            for j in range(len(free))
            if theta[j] < lo[j] or theta[j] > hi[j]
        }
        lam = counts(q, p0 | dict(zip(free, theta.tolist(), strict=True)))
        out[k]["+" if sign > 0 else "-"] = {
            "quadratic_D": float(dtheta @ fisher @ dtheta),
            "forward_D": float(deviance_terms(lam, lam_t).sum()),
            "outside_bounds": outside,
            "largest_moves": {
                free[j]: float(dtheta[j] / (hi[j] - lo[j]))
                for j in np.argsort(np.abs(dtheta / (hi - lo)))[::-1][:4]
            },
        }
(OUT / "b2_extra.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=True))
