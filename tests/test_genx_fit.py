import numpy as np
import pytest
import torch

from reflectolearn.benchmark import SI, SIO2, parratt
from reflectolearn.physics import AbelesMatrix

pytest.importorskip("genx")
from genx.model import Model  # noqa: E402

from reflectolearn.genx_fit import _SCRIPT, genx_fit  # noqa: E402


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


def test_genx_model_buries_the_oxide_under_the_film():
    init = dict.fromkeys(("roughness", "sld", "oxide_roughness", "oxide_sld", "i0", "bkg"), 1.0)
    model = Model()
    model.set_script(
        _SCRIPT.format(
            init={**init, "thickness": 300.0, "oxide_thickness": 15.0, "sample_len": 10.0},
            sub_sld=20.07,
            sub_rough=3.0,
            wavelength=1.54,
            beam_width=0.1,
        )
    )
    model.compile_script()
    # GenX resolves a sample from the substrate up
    sample = model.script_module.sample
    assert list(sample.resolveLayerParameters().d) == [0.0, 15.0, 300.0, 0.0]


@pytest.mark.parametrize("guess", [0.97, 1.03])
def test_genx_fit_recovers_a_film_on_buried_oxide(guess):
    # Independent float64 Parratt curve of air / film / SiO2 / Si, smeared with the fit model's
    # Gaussian resolution (sigma 0.002 1/A)
    q = np.linspace(0.05, 0.35, 300)
    qf = np.linspace(0.04, 0.36, 4000)
    w = np.exp(-0.5 * ((qf[None] - q[:, None]) / 0.002) ** 2)
    layers = [
        (0j, 0.0, 0.0),
        (40 + 0j, 200.0, 4.0),
        (SIO2.real + 0j, 20.0, 3.0),
        (SI.real + 0j, 0.0, 3.0),
    ]
    r = (w @ parratt(qf, layers)) / w.sum(1)

    np.random.seed(0)
    res = genx_fit(
        q,
        r,
        200 * guess,
        4 * guess,
        40 * guess,
        sub_sld=SI.real,
        sample_len=45.0,
        pop_size=20,
        steps_scale=0.3,
    )

    assert res["thickness"] == pytest.approx(200.0, rel=0.01)
    assert res["roughness"] == pytest.approx(4.0, rel=0.1)
    assert res["sld"] == pytest.approx(40.0, rel=0.05)
    assert res["oxide_thickness"] == pytest.approx(20.0, abs=6.0)
    assert res["oxide_sld"] == pytest.approx(SIO2.real, abs=1.0)
