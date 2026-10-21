"""Benchmark plumbing: seeded synthetic sets, scoring and the JSON/CSV output.

predict itself is replaced by a stub, since a real run needs a trained checkpoint.
"""

import csv
import json

import numpy as np
import pytest

from reflectolearn import predict
from reflectolearn.benchmark import FIXED_CASES, make_fixed, make_synthetic, run_benchmark, score
from reflectolearn.config import Config
from reflectolearn.physics import tth2q

CFG = Config(device="cpu")
WL = CFG.simulation.wavelength


def test_synthetic_sets_are_seeded():
    a, b, c = make_fixed(CFG, seed=3), make_fixed(CFG, seed=3), make_fixed(CFG, seed=4)
    for (ca, qa, ia), (_, qb, ib), (_, _, ic) in zip(a, b, c):
        np.testing.assert_array_equal(qa, qb)
        np.testing.assert_array_equal(ia, ib)
        assert not np.array_equal(ia, ic), ca.name
    s1, s2 = make_synthetic(2, seed=3, wavelength=WL), make_synthetic(2, seed=3, wavelength=WL)
    for (c1, _, n1), (c2, _, n2) in zip(s1, s2):
        assert c1 == c2
        np.testing.assert_array_equal(n1, n2)


def test_fixed_cases_lie_in_the_training_ranges():
    pr = CFG.param_ranges
    for c, q, intensity in make_fixed(CFG):
        assert pr.thickness[0] <= c.thickness <= pr.thickness[1]
        assert pr.roughness[0] <= c.roughness <= pr.roughness[1]
        assert pr.sld[0] <= c.sld <= pr.sld[1]
        assert tth2q(0.2, WL) <= q.min() < q.max() <= tth2q(8.0, WL)
        assert np.all(np.isfinite(intensity)) and intensity.min() >= 0
        # the curve must fall over several decades for the fit to mean anything
        assert intensity.max() / max(np.median(intensity[-50:]), 1e-12) > 1e4


def test_score_reports_errors_and_interval():
    res = {"ai": {"thickness": 90.0, "roughness": 4.0, "sld": 30.0},
           "refined": {"thickness": 110.0, "roughness": 5.0, "sld": 31.0}, "fom_refined": 0.01,
           "posterior": {"thickness": {"lo": 105.0, "hi": 115.0}}}
    row = score("synthetic_fixed", "x", res, {"thickness": 100.0, "roughness": 5.0, "sld": 30.0}, 1.0)
    assert row["thickness_err_pct"] == pytest.approx(10.0)
    assert row["thickness_lo"] == 105.0
    assert row["thickness_in_68"] is False


def test_run_benchmark_end_to_end(tmp_path, monkeypatch):
    guess = {"thickness": 200.0, "roughness": 5.0, "sld": 40.0}

    def fake_predict(q, counts, cfg, predictor, **kw):
        assert np.all(np.isfinite(counts)) and len(q) == len(counts)
        return {"ai": guess, "refined": guess, "fom_refined": 0.0}, {}

    monkeypatch.setattr(predict, "load_predictor", lambda *a: None)
    monkeypatch.setattr(predict, "predict_curve", fake_predict)
    out = run_benchmark(CFG, "unused.pt", n_synthetic=2, seed=5, out_dir=tmp_path, device="cpu")

    assert {k: v["n"] for k, v in out["summary"].items()} == {"synthetic": 2, "synthetic_fixed": len(FIXED_CASES)}
    fixed = {r["case"]: r for r in out["rows"] if r["set"] == "synthetic_fixed"}
    for c in FIXED_CASES:
        assert fixed[c.name]["thickness_true"] == c.thickness
        assert fixed[c.name]["thickness_err_pct"] == pytest.approx(100 * (200.0 / c.thickness - 1))
    saved = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert saved["summary"] == out["summary"]
    with open(tmp_path / "benchmark.csv", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 2 + len(FIXED_CASES)
