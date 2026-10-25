"""ReflectoLearn's Abeles engine as the reference, against the XRefine and ReflectoExp engines.

The other implementations are frozen copies in ``tests/reference`` taken from XRefine and ReflectoExp
at fixed commits. Ground truth is ``benchmark.parratt``: an independent float64 Parratt recursion.

Layouts differ, so each engine gets the same stack in its own convention:
    ReflectoLearn / XRefine torch: thickness and roughness per interface (N - 1), SLD per medium (N)
    ReflectoExp torch / XRefine numpy: thickness per finite layer (N - 2), roughness per interface,
                                       SLD with (ReflectoExp) or without (XRefine numpy) the ambient

What the comparison shows:
    - ReflectoExp's torch engine equals the Parratt recursion to rounding in double precision.
    - ReflectoLearn and XRefine torch differ only by XRefine's extra 1e-9 in two denominators:
      about 1e-7 relative.
    - That absorption term (``AbelesMatrix`` adds 1e-9j to every SLD in 1/A^2) is the whole
      difference from Parratt and scales linearly with it: about 0.06% on the total-reflection
      plateau, and up to 0.5% (2.3e-3 in log10 R) near a deep minimum in one of the random stacks.
    - XRefine's ``abeles_np`` computes in complex64 even for float64 input (``q.dtype is
      np.float64`` is never true), so it agrees only to single precision.
"""

import numpy as np
import pytest
import torch
from reference.reflectoexp_abeles import abeles as reflectoexp_abeles
from reference.xrefine_abeles import abeles_core as xrefine_core
from reference.xrefine_numpy import abeles_np as xrefine_numpy

from reflectolearn.benchmark import SI, SIO2, parratt
from reflectolearn.physics import AbelesMatrix, abeles_core

Q = np.linspace(0.005, 0.6, 600)


def _stacks():
    """(name, sld per medium ambient first, thickness per interface, roughness per interface)."""
    yield "bare Si", [0j, 20.07 + 0j], [0.0], [0.0]
    yield (
        "absorbing film on SiO2 on Si",
        [0j, 45 + 2j, SIO2, SI],
        [0.0, 250.0, 15.0],
        [6.0, 3.0, 4.0],
    )
    rng = np.random.default_rng(7)
    for k in range(4):
        n = int(rng.integers(1, 5))
        sld = [0j, *(rng.uniform(5, 60, n) + 1j * rng.uniform(0, 2, n)), SI]
        yield f"random {k}", sld, [0.0, *rng.uniform(20, 300, n)], list(rng.uniform(0, 8, n + 1))


STACKS = list(_stacks())
IDS = [s[0] for s in STACKS]


def _truth(sld, thickness, roughness):
    finite = [*thickness[1:], 0.0]
    return parratt(
        Q,
        [(sld[0], 0.0, 0.0)]
        + [(s, d, r) for s, d, r in zip(sld[1:], finite, roughness, strict=True)],
    )


def _core128(core, sld, thickness, roughness, absorption):
    """A (B, Q, 1) / (B, 1, N) core called in complex128, laid out as ``AbelesMatrix`` does."""
    s = torch.tensor([sld], dtype=torch.complex128)
    s = (s - s[:, :1]) * 1e-6 + 1j * absorption
    return core(
        torch.tensor(Q, dtype=torch.complex128)[None, :, None],
        torch.tensor([thickness], dtype=torch.float64)[:, None],
        torch.tensor([roughness], dtype=torch.float64)[:, None],
        s[:, None],
    )[0].numpy()


def _dlog(a, b):
    return np.abs(np.log10(a) - np.log10(b)).max()


@pytest.mark.parametrize(("name", "sld", "thickness", "roughness"), STACKS, ids=IDS)
def test_reflectoexp_torch_engine_equals_parratt_in_double_precision(
    name, sld, thickness, roughness
):
    got = reflectoexp_abeles(
        torch.tensor(Q, dtype=torch.float64),
        torch.tensor([thickness[1:]], dtype=torch.float64),
        torch.tensor([roughness], dtype=torch.float64),
        torch.tensor([sld], dtype=torch.complex128),
    )[0].numpy()
    assert _dlog(got, _truth(sld, thickness, roughness)) < 1e-10


@pytest.mark.parametrize(("name", "sld", "thickness", "roughness"), STACKS, ids=IDS)
def test_reflectolearn_core_equals_xrefine_core(name, sld, thickness, roughness):
    ours = _core128(abeles_core, sld, thickness, roughness, 1e-9)
    theirs = _core128(xrefine_core, sld, thickness, roughness, 1e-9)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6)


@pytest.mark.parametrize(("name", "sld", "thickness", "roughness"), STACKS, ids=IDS)
def test_the_only_difference_from_parratt_is_the_absorption_guard(name, sld, thickness, roughness):
    truth = _truth(sld, thickness, roughness)
    assert _dlog(_core128(abeles_core, sld, thickness, roughness, 1e-36), truth) < 1e-10
    with_guard = _dlog(_core128(abeles_core, sld, thickness, roughness, 1e-9), truth)
    assert with_guard < 5e-3
    # linear in the term: a thousand times smaller guard, a thousand times smaller deviation
    smaller = _dlog(_core128(abeles_core, sld, thickness, roughness, 1e-12), truth)
    assert smaller == pytest.approx(with_guard / 1000, rel=0.05)


@pytest.mark.parametrize(("name", "sld", "thickness", "roughness"), STACKS, ids=IDS)
def test_float32_module_agrees_with_parratt(name, sld, thickness, roughness):
    got = AbelesMatrix()(
        torch.tensor(Q)[None],
        torch.tensor([thickness]),
        torch.tensor([roughness]),
        torch.tensor([sld], dtype=torch.complex64),
    )[0].double()
    assert _dlog(got.numpy(), _truth(sld, thickness, roughness)) < 5e-3


@pytest.mark.parametrize(("name", "sld", "thickness", "roughness"), STACKS, ids=IDS)
def test_xrefine_numpy_engine_agrees_to_single_precision(name, sld, thickness, roughness):
    got = xrefine_numpy(
        Q.astype(np.float64), np.array(thickness[1:]), np.array(roughness), np.array(sld[1:])
    )
    deviation = _dlog(got, _truth(sld, thickness, roughness))
    assert 1e-9 < deviation < 1e-3  # complex64 arithmetic despite float64 input
