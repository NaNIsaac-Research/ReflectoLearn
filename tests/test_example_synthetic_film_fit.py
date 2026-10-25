"""examples/synthetic_film_fit end to end: synthetic film -> counts -> preprocess -> fit.

Checks the curve's invariants, the fit against the known truth (SLD and roughness are known to
come out wrong, ReflectoLearn#14, and are strict xfails so a fix shows up), and today's output as
a regression snapshot. Units: Angstrom, 1/Angstrom, SLD in 1e-6/Angstrom^2.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from reflectolearn.benchmark import parratt
from reflectolearn.config import Config
from reflectolearn.refine import RefineConfig

EXAMPLE = Path(__file__).parents[1] / "examples" / "synthetic_film_fit"
EXPECTED = json.loads((EXAMPLE / "expected.json").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("synthetic_film_fit", EXAMPLE / "run.py")
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


@pytest.fixture(scope="module")
def result():
    return example.run()


def _q_c(sld: float) -> float:
    return 4 * np.sqrt(np.pi * sld * 1e-6)


def test_the_true_curve_never_exceeds_one_and_its_edge_follows_the_film_sld():
    q = np.linspace(0.005, 0.08, 3001)
    r = parratt(q, example.layers())
    assert r.max() <= 1.0
    film_sld = example.TRUTH["film"]["sld"][0]
    edge = q[np.argmax(r < 0.5)]  # the densest layer is the film, on top
    assert _q_c(film_sld) < edge < 1.05 * _q_c(film_sld)


def test_the_scan_covers_the_edge_and_the_fringes(result):
    q_lo, q_hi = result["q_measured"]
    assert q_lo < _q_c(example.TRUTH["film"]["sld"][0]) < q_hi
    assert q_hi > 0.4
    assert result["points"] > 700


def _check(name: str, got: float, spec: dict):
    assert abs(got - spec["value"]) <= spec["tol"], (
        f"{name}: {got:.4g} vs {spec['value']} ± {spec['tol']}"
    )


def test_thickness_matches_the_truth(result):
    truth = EXPECTED["truth"]
    _check("thickness", result["fit"]["thickness"], truth["thickness"])
    _check("fft_thickness", result["fft_thickness"], truth["fft_thickness"])


@pytest.mark.xfail(
    strict=True,
    reason="ReflectoLearn#14: refine drifts along the tth_offset-SLD valley, SLD ~30% low",
)
@pytest.mark.parametrize("key", ["sld", "roughness"])
def test_sld_and_roughness_match_the_truth(result, key):
    _check(key, result["fit"][key], EXPECTED["truth"][key])


@pytest.mark.parametrize("key", ["thickness", "sld", "roughness", "tth_offset", "fom_after"])
def test_output_is_unchanged(result, key):
    got = result["fit"][key] if key in result["fit"] else result[key]
    _check(key, got, EXPECTED["regression"][key])


def test_a_nuisance_held_off_is_reported_as_none():
    """refine leaves the nuisances it did not fit out of its info (#22)."""
    got = example.run(config=RefineConfig(fit_tth_offset=False, fit_oxide=False))
    assert got["tth_offset"] is None
    assert got["oxide_thickness"] is None


def test_the_example_uses_the_default_q_grid_and_wavelength():
    cfg = Config(device="cpu")
    assert cfg.simulation.wavelength == pytest.approx(1.54)
    assert len(example.scan_q(cfg)) == 291
