"""The negative-intensity step chosen for a prediction reaches the saved record and the CLI."""

import torch
from typer.testing import CliRunner

from reflectolearn import predict
from reflectolearn.cli import app
from reflectolearn.config import Config


def test_predict_txt_records_the_negative_step(tmp_path):
    cfg = Config(device="cpu")
    q = cfg.q_grid
    ones = torch.ones(1, q.numel())
    params = dict.fromkeys(predict.PARAM_NAMES, 1.0)
    result = {
        "file": "scan.txt",
        "preprocess": {"negative": "mask", "negative_points": 7, "points": 300, "floor": 1e-15},
        "ai": params,
        "refined": None,
        "fom_ai": 0.1,
        "fom_refined": None,
        "posterior": None,
    }
    predict._save_outputs(
        tmp_path / "scan.txt", tmp_path, cfg, q, ones, ones, ones, ones, result, False
    )
    lines = (tmp_path / "scan_predict.txt").read_text(encoding="utf-8").splitlines()
    assert "# preprocess negative=mask (7 of 300 points)" in lines


def test_cli_passes_negative_to_predict_file(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(predict, "predict_file", lambda *a, **kw: seen.update(kw))
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"")
    data = tmp_path / "scan.txt"
    data.write_text("0.1 1\n", encoding="utf-8")
    args = ["predict", str(data), "-k", str(checkpoint), "--no-refine"]
    assert CliRunner().invoke(app, [*args, "--negative", "mask"]).exit_code == 0
    assert seen["negative"] == "mask"
    assert CliRunner().invoke(app, [*args, "--negative", "clip"]).exit_code != 0
