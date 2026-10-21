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
    np.testing.assert_allclose(x[0, 0].numpy(), GOLDEN["x_log"], rtol=0, atol=1e-6)


def test_contract_is_current():
    assert GOLDEN["contract"] == json.loads(json.dumps(input_contract(Config(device="cpu"))))
