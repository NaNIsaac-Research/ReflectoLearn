"""The shared golden curves (tests/golden/xrr.json) stay what the reference engines compute.

Other engines (xrr.js on the website) test against copies of this file, so it must not drift:
it is regenerated here and compared, and the Abeles core reproduces it once its absorption
guard is set to ~0.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from reflectolearn.physics import abeles_core

GOLDEN = json.loads((Path(__file__).parent / "golden" / "xrr.json").read_text(encoding="utf-8"))


def test_golden_file_matches_its_generator():
    from golden.make_xrr_golden import build

    fresh = build()
    assert [s["name"] for s in fresh["stacks"]] == [s["name"] for s in GOLDEN["stacks"]]
    np.testing.assert_allclose(fresh["q"], GOLDEN["q"], rtol=1e-15)
    for new, old in zip(fresh["stacks"], GOLDEN["stacks"], strict=True):
        assert new["layers"] == old["layers"]
        np.testing.assert_allclose(new["R"], old["R"], rtol=1e-12)


@pytest.mark.parametrize("stack", GOLDEN["stacks"], ids=[s["name"] for s in GOLDEN["stacks"]])
def test_abeles_core_reproduces_the_golden_curve(stack):
    layers = stack["layers"]
    sld = torch.tensor([[complex(*lay["sld"]) for lay in layers]], dtype=torch.complex128)
    sld = (sld - sld[:, :1]) * 1e-6 + 1e-36j
    # the core takes per-interface thickness (ambient's first, ignored) and roughness
    thickness = torch.tensor(
        [[0.0] + [lay["thickness"] for lay in layers[1:-1]]], dtype=torch.float64
    )
    roughness = torch.tensor([[lay["roughness"] for lay in layers[1:]]], dtype=torch.float64)
    r = abeles_core(
        torch.tensor(GOLDEN["q"], dtype=torch.complex128)[None, :, None],
        thickness[:, None],
        roughness[:, None],
        sld[:, None],
    )[0].numpy()
    assert np.abs(np.log10(r) - np.log10(stack["R"])).max() < 1e-10
