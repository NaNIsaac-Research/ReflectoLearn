"""ONNX export: the graph gives the same physical parameters as the torch model on a real input.

The model is untrained (no checkpoint in the repository); the export only has to reproduce it.
Skipped without the ``onnx`` extra.
"""

import json

import numpy as np
import pytest
import torch

from reflectolearn.benchmark import parratt
from reflectolearn.config import Config
from reflectolearn.data import param_stats
from reflectolearn.export import PhysicalOutput, export_onnx
from reflectolearn.model import XRRPhysicsModel
from reflectolearn.physics import tth2q
from reflectolearn.predict import preprocess_measured

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

CFG = Config(device="cpu")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = XRRPhysicsModel(q_len=CFG.simulation.q_points, n_channels=8, mlp_hidden=32).eval()
    stats = param_stats(CFG)
    out = tmp_path_factory.mktemp("onnx") / "model.onnx"
    meta = export_onnx(model, stats["mean"], stats["std"], CFG, out, note="untrained test model")
    return model, stats, out, meta


def measured_input():
    """A film on Si from 2θ = 0.2° to 6°, through the same preprocessing as predict."""
    q = tth2q(np.arange(0.2, 6.01, 0.02), CFG.simulation.wavelength)
    r = parratt(q, [(0j, 0, 0), (45 + 0j, 250.0, 6.0), (20.0 + 0j, 0, 4.0)])
    x, _, _ = preprocess_measured(q, r * 1e7 + 1.0, CFG, torch.device("cpu"))
    return x


def test_onnx_matches_torch(exported):
    model, stats, out, _ = exported
    x = measured_input()
    with torch.no_grad():
        want = PhysicalOutput(model, stats["mean"], stats["std"])(x).numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    (got,) = session.run(["params"], {"x": x.numpy()})
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * np.abs(want).max())


def test_weights_are_inside_the_file(exported):
    *_, out, _ = exported
    assert [p.name for p in out.parent.iterdir() if p.suffix != ".json"] == [out.name]


def test_contract_describes_the_input(exported):
    *_, meta = exported
    contract = json.loads(meta.read_text(encoding="utf-8"))
    grid = contract["input"]["q_grid"]
    q = CFG.q_grid
    assert contract["input"]["shape"] == [1, 2, q.numel()]
    assert grid["start"] == pytest.approx(float(q[0]))
    assert grid["stop"] == pytest.approx(float(q[-1]))
    assert contract["output"]["order"] == ["thickness", "roughness", "sld"]
    assert contract["model"]["note"] == "untrained test model"
    x = measured_input()
    assert float(x[0, 0].min()) == pytest.approx(contract["input"]["log_floor"])
