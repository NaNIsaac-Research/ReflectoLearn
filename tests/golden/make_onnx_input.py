"""Write tests/golden/onnx_input.json: one measured curve and the model input predict builds from it.

Non-Python callers of the exported model (the browser demo) rebuild x from model.json's contract;
this file lets them check that they get the same x. Run from the repository root:

    uv run python tests/golden/make_onnx_input.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from reflectolearn.benchmark import smeared
from reflectolearn.config import Config
from reflectolearn.export import input_contract
from reflectolearn.physics import tth2q
from reflectolearn.predict import preprocess_measured

OUT = Path(__file__).with_name("onnx_input.json")


def main():
    cfg = Config(device="cpu")
    tth = np.arange(0.2, 6.01, 0.02)
    q = tth2q(tth, cfg.simulation.wavelength)
    layers = [(0j, 0, 0), (45 + 0j, 250.0, 6.0), (20.0 + 0j, 0, 4.0)]
    rng = np.random.default_rng(2026)
    counts = rng.poisson((smeared(q, layers, 0.02) + 1e-7) * 1e7).astype(float)
    x, _, _ = preprocess_measured(q, counts, cfg, torch.device("cpu"))
    data = {
        "about": "A 250 A, SLD 45 film on Si (dq/q 2 %, 1e7 counts, Poisson, seed 2026) and the model "
        "input x that predict.preprocess_measured builds from it, float32 as the model sees it.",
        "contract": input_contract(cfg),
        "q": q.tolist(),
        "intensity": counts.tolist(),
        "x_log": [float(v) for v in x[0, 0]],
        "x_mask": [int(v) for v in x[0, 1]],
    }
    OUT.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {OUT} ({int(x[0, 1].sum())} of {x.shape[-1]} grid points measured)")


if __name__ == "__main__":
    main()
