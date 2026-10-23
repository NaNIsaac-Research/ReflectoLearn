"""tests/golden/onnx_input.json still matches predict.preprocess_measured and the export contract.

The browser demo checks its own input builder against the same file, so a change in preprocessing
must regenerate it (tests/golden/make_onnx_input.py) and be copied there.
"""

import json
from pathlib import Path

import numpy as np
import torch

from reflectolearn.config import Config
from reflectolearn.export import input_contract
from reflectolearn.predict import preprocess_measured

GOLDEN = json.loads((Path(__file__).parent / "golden" / "onnx_input.json").read_text("utf-8"))


def test_preprocessing_matches_the_golden_input():
    cfg = Config(device="cpu")
    q, counts = np.array(GOLDEN["q"]), np.array(GOLDEN["intensity"])
    x, _, _ = preprocess_measured(q, counts, cfg, torch.device("cpu"))
    np.testing.assert_array_equal(x[0, 1].numpy(), GOLDEN["x_mask"])
    # q_grid is torch.linspace in float32, whose last bits depend on the platform. Low-count points
    # sit on a steep part of log10, so one ulp of q_grid moves x_log by up to 6.3e-5 (measured by
    # rebuilding the grid in float64 and by shifting it +-1 ulp); the golden came from Windows and
    # Linux CI differed by 4.0e-5 at 104 of 2000 points. Hence atol 2e-4, not 1e-6.
    np.testing.assert_allclose(x[0, 0].numpy(), GOLDEN["x_log"], rtol=0, atol=2e-4)


def test_contract_is_current():
    assert GOLDEN["contract"] == json.loads(json.dumps(input_contract(Config(device="cpu"))))
