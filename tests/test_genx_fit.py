import numpy as np
import pytest
import torch

from reflectolearn.physics import AbelesMatrix

pytest.importorskip("genx")
from reflectolearn.genx_fit import genx_fit  # noqa: E402


def test_genx_fit_runs_from_guess():
    q = torch.linspace(0.02, 0.4, 300)
    # ambient / film (300 A, 4 A, SLD 40) / substrate (SLD 20.07, 3 A)
    r = AbelesMatrix()(
        q[None],
        torch.tensor([[0.0, 300.0]]),
        torch.tensor([[4.0, 3.0]]),
        torch.tensor([[0.0, 40.0, 20.07]]),
        wavelength=1.54,
    )[0].numpy()

    res = genx_fit(q.numpy(), r, 310.0, 4.2, 41.0, pop_size=10, steps_scale=0.05)

    assert np.isfinite(res["fom"])
    assert res["r_fit"].shape == r.shape
    assert 310.0 * 0.9 <= res["thickness"] <= 310.0 * 1.1
