"""Analytical checks for the Abeles solver (``abeles_core`` / ``AbelesMatrix``).

Units used throughout, matching ``AbelesMatrix.forward``:
    q          1/Angstrom (momentum transfer, k_z = q / 2 in the ambient)
    thickness  Angstrom, one per interface (N - 1 for N media); the first is the
               ambient's and is ignored by the physics, so tests pass 0
    roughness  Angstrom (rms, Nevot-Croce), one per interface
    sld        1e-6 / Angstrom^2, per medium, ambient first

``AbelesMatrix`` adds a 1e-9 absorption term to every medium; it is far below the
tolerances here, so the references are the textbook formulas without it.
Everything runs on CPU without the optional ``genx`` extra, except the last test.
"""

from itertools import pairwise

import numpy as np
import pytest
import torch
from scipy.signal import argrelmin

from reflectolearn.physics import AbelesMatrix, abeles_core

SI = 20.07  # silicon SLD, 1e-6 / A^2
Q_C_SI = 4 * np.sqrt(np.pi * SI * 1e-6)  # critical edge, ~0.0318 1/A


def reflectivity(q, thickness, roughness, sld):
    """AbelesMatrix on a single stack given as plain lists; returns numpy."""
    q = torch.as_tensor(q, dtype=torch.float64)
    out = AbelesMatrix()(
        q[None], torch.tensor([thickness]), torch.tensor([roughness]), torch.tensor([sld])
    )
    return out[0].double().numpy()


def kz(q, sld):
    """Wavevector normal to the surface inside a medium, no absorption."""
    return np.sqrt((q / 2) ** 2 - 4 * np.pi * sld * 1e-6 + 0j)


def fresnel(q, sld):
    k0, k1 = kz(q, 0.0), kz(q, sld)
    return np.abs((k0 - k1) / (k0 + k1)) ** 2


# ----------------------------------------------------------- single interface
def test_bare_substrate_matches_the_fresnel_formula():
    q = np.linspace(1.3 * Q_C_SI, 0.6, 400)
    np.testing.assert_allclose(reflectivity(q, [0.0], [0.0], [0.0, SI]), fresnel(q, SI), rtol=1e-3)


def test_below_the_critical_edge_reflection_is_total():
    q = np.linspace(0.005, 0.8 * Q_C_SI, 50)
    assert np.all(reflectivity(q, [0.0], [0.0], [0.0, SI]) > 0.99)


def test_high_q_tail_follows_q_to_the_minus_four():
    q = np.linspace(0.4, 1.0, 50)
    r = reflectivity(q, [0.0], [0.0], [0.0, SI])
    np.testing.assert_allclose(r * q**4, Q_C_SI**4 / 16, rtol=2e-2)


def test_roughness_damps_a_single_interface_by_the_nevot_croce_factor():
    q = np.linspace(1.3 * Q_C_SI, 0.5, 200)
    sigma = 5.0
    smooth = reflectivity(q, [0.0], [0.0], [0.0, SI])
    rough = reflectivity(q, [0.0], [sigma], [0.0, SI])
    damping = np.exp(-4 * (kz(q, 0.0) * kz(q, SI)).real * sigma**2)
    keep = damping > 1e-6  # stay above float32 noise
    np.testing.assert_allclose(rough[keep] / smooth[keep], damping[keep], rtol=1e-2)


def test_more_roughness_never_raises_high_q_reflectivity():
    q = np.linspace(0.1, 0.5, 100)
    curves = [reflectivity(q, [0.0, 200.0], [s, s], [0.0, 40.0, SI]) for s in (0.0, 2.0, 4.0, 8.0)]
    for smoother, rougher in pairwise(curves):
        assert np.all(rougher <= smoother * (1 + 1e-4))


# ----------------------------------------------------------------- thin film
def test_film_fringes_are_spaced_by_pi_over_the_thickness_in_film_kz():
    d, film = 250.0, 40.0
    q = np.linspace(0.25, 0.5, 8001)  # ~800 samples per fringe
    r = reflectivity(q, [0.0, d], [0.0, 0.0], [0.0, film, SI])
    # complex64 is good to ~5e-5 relative, enough to put a false dip between two
    # neighbouring samples on this grid; a real minimum is lowest over 100 samples.
    minima = argrelmin(r, order=100)[0]
    assert len(minima) >= 9
    # Minima sit where the film's round-trip phase 2 k_z d steps by 2 pi.
    step = np.diff(kz(q[minima], film).real)
    np.testing.assert_allclose(step, np.pi / d, rtol=1e-2)


# ------------------------------------------------------------ batch handling
def test_a_batch_equals_its_stacks_computed_one_by_one():
    q = torch.linspace(0.01, 0.5, 64).repeat(3, 1)
    thickness = torch.tensor([[0.0, 120.0], [0.0, 300.0], [0.0, 55.0]])
    roughness = torch.tensor([[3.0, 2.0], [5.0, 1.0], [0.0, 4.0]])
    sld = torch.tensor([[0.0, 35.0, SI], [0.0, 10.0, SI], [0.0, 60.0, SI]])
    model = AbelesMatrix()

    batch = model(q, thickness, roughness, sld)
    assert batch.shape == (3, 64)
    assert batch.dtype == torch.float32 and batch.device.type == "cpu"
    for i in range(3):
        single = model(q[i : i + 1], thickness[i : i + 1], roughness[i : i + 1], sld[i : i + 1])
        torch.testing.assert_close(batch[i : i + 1], single)
    torch.testing.assert_close(model(q, thickness, roughness, sld), batch)  # a second call


# ----------------------------------------------------------------- gradients
def _core64(q, thickness, roughness, sld_real):
    """abeles_core in double precision, laid out as AbelesMatrix does."""
    sld = (sld_real.to(torch.complex128) - sld_real[:, :1]) * 1e-6 + 1e-9j
    return abeles_core(
        q.to(torch.complex128).unsqueeze(-1),
        thickness.unsqueeze(1),
        roughness.unsqueeze(1),
        sld.unsqueeze(1),
    )


def test_gradients_match_finite_differences():
    q = torch.linspace(0.05, 0.3, 7, dtype=torch.float64)[None]
    thickness = torch.tensor([[0.0, 150.0, 40.0]], dtype=torch.float64, requires_grad=True)
    roughness = torch.tensor([[3.0, 2.5, 4.0]], dtype=torch.float64, requires_grad=True)
    sld = torch.tensor([[0.0, 35.0, 15.0, SI]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t, r, s: _core64(q, t, r, s), (thickness, roughness, sld)
    )


def test_float32_module_gradients_agree_with_double_precision():
    q = torch.linspace(0.05, 0.3, 7)[None]
    stack = ([[0.0, 150.0, 40.0]], [[3.0, 2.5, 4.0]], [[0.0, 35.0, 15.0, SI]])

    params32 = [torch.tensor(p, requires_grad=True) for p in stack]
    torch.log(AbelesMatrix()(q, *params32)).sum().backward()
    params64 = [torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in stack]
    torch.log(_core64(q.double(), *params64)).sum().backward()

    for p32, p64 in zip(params32, params64):
        assert torch.isfinite(p32.grad).all()
        torch.testing.assert_close(p32.grad.double(), p64.grad, rtol=2e-2, atol=1e-3)


# ---------------------------------------------------------------- vs. GenX
def test_matches_genx_on_random_stacks():
    pytest.importorskip("genx")
    from genx.models.lib.physical_constants import r_e
    from genx.models.spec_nx import (
        Coords,
        FootType,
        Instrument,
        Layer,
        Probe,
        ResType,
        Sample,
        Specular,
        Stack,
    )

    inst = Instrument(
        probe=Probe.xray,
        wavelength=1.54,
        coords=Coords.q,
        I0=1.0,
        Ibkg=0.0,
        res=0.0,
        restype=ResType.none,
        footype=FootType.none,
    )
    rng = np.random.default_rng(0)
    q = np.linspace(0.02, 0.4, 300)
    for _ in range(5):
        n = int(rng.integers(1, 4))
        d = rng.uniform(20, 300, n)
        sigma = rng.uniform(0, 8, n + 1)  # top interface of each layer, then the substrate's
        rho = rng.uniform(5, 60, n)
        layers = [
            Layer(d=d[i], sigma=sigma[i], f=complex(rho[i] * 1e-6 / r_e, 0), dens=1)
            for i in range(n)
        ]
        # GenX lists a stack's layers from the substrate up; ours go from the ambient down.
        sample = Sample(
            Stacks=[Stack(Layers=layers[::-1])],
            Ambient=Layer(d=0, f=0, dens=0),
            Substrate=Layer(d=0, f=complex(SI * 1e-6 / r_e, 0), dens=1, sigma=sigma[n]),
        )
        ours = reflectivity(q, [0.0, *d], list(sigma), [0.0, *rho, SI])
        theirs = Specular(q, sample, inst)
        np.testing.assert_allclose(np.log10(ours), np.log10(theirs), atol=1e-2)
