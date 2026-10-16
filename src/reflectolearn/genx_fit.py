"""Classical GenX fit (differential evolution via bumps) seeded with a CNN estimate.

Optional: needs the ``genx`` extra (``uv sync --extra genx``). Model: film + oxide
layer on a fixed substrate, with I0, background and footprint (sample length) fitted
in two stages (oxide + I0, then everything).
"""
from __future__ import annotations

import numpy as np

# Fitted parameter -> (init, lo, hi); film entries are filled from the CNN guess.
_OXIDE_INSTRUMENT = {
    "oxide_thickness": (15.0, 0.01, 55.0),
    "oxide_roughness": (3.0, 0.5, 15.0),
    "oxide_sld": (16.0, 5.0, 20.0),
    "i0": (1.0, 0.1, 20.0),
    "bkg": (1e-7, 1e-10, 1e-4),
}
_STAGES = (
    ("oxide", ("oxide_thickness", "oxide_roughness", "i0"), 400),
    ("final", ("thickness", "roughness", "sld", "oxide_thickness", "oxide_sld",
               "i0", "bkg", "sample_len"), 600),
)

_SCRIPT = """
from genx.models.spec_nx import Sample, Stack, Layer, Instrument, Specular
from genx.models.spec_nx import Probe, Coords, ResType, FootType
from genx.models.lib.physical_constants import r_e

def to_f(sld):
    return complex(sld * 1e-6 / r_e, 0)

class Vars:
    pass

for _n, _v in {init!r}.items():
    setattr(Vars, _n, _v)
    setattr(Vars, "set_" + _n, lambda self, x, _n=_n: setattr(self, _n, float(x)))
    setattr(Vars, "get_" + _n, lambda self, _n=_n: getattr(self, _n))

v = Vars()

Amb = Layer(d=0, f=0, dens=0)
Sub = Layer(d=0, f=to_f({sub_sld}), dens=1, sigma={sub_rough})
Film = Layer(d=v.thickness, sigma=v.roughness, f=to_f(v.sld), dens=1)
Oxide = Layer(d=v.oxide_thickness, sigma=v.oxide_roughness, f=to_f(v.oxide_sld), dens=1)
sample = Sample(Stacks=[Stack(Layers=[Film, Oxide])], Ambient=Amb, Substrate=Sub)

inst = Instrument(
    probe=Probe.xray, wavelength={wavelength}, coords=Coords.q,
    I0=v.i0, Ibkg=v.bkg, res=0.002, restype=ResType.fast_conv,
    footype=FootType.gauss, samplelen=v.sample_len, beamw={beam_width},
)

def Sim(data):
    Film.d, Film.sigma, Film.f = v.thickness, v.roughness, to_f(v.sld)
    Oxide.d, Oxide.sigma, Oxide.f = v.oxide_thickness, v.oxide_roughness, to_f(v.oxide_sld)
    inst.I0, inst.Ibkg, inst.samplelen = v.i0, v.bkg, v.sample_len
    return [Specular(d.x, sample, inst) for d in data]
"""


def genx_fit(
    q: np.ndarray,
    intensity: np.ndarray,
    thickness: float,
    roughness: float,
    sld: float,
    *,
    wavelength: float = 1.54,
    sub_sld: float = 20.07,
    sub_rough: float = 3.0,
    beam_width: float = 0.1,
    sample_len: float = 10.0,
    film_margin: float = 0.1,
    pop_size: int = 40,
    steps_scale: float = 1.0,
) -> dict:
    """Fit a measured curve with GenX, starting from (thickness, roughness, sld).

    Film parameters are bounded to ±``film_margin`` of the guess. ``intensity`` is
    normalized to its max. Returns fitted parameters plus ``r_fit`` and ``fom`` (logR2).
    """
    from genx import fom_funcs
    from genx.data import DataList, DataSet
    from genx.model import Model
    from genx.parameters import Parameters

    r = np.asarray(intensity, dtype=float)
    r = np.nan_to_num(r / max(np.nanmax(r), 1e-30), nan=1e-12, posinf=1e-12)

    m = film_margin
    ranges = {
        "thickness": (thickness, max(10.0, thickness * (1 - m)), thickness * (1 + m)),
        "roughness": (roughness, max(0.0, roughness * (1 - m)), roughness * (1 + m)),
        "sld": (sld, max(5.0, sld * (1 - m)), min(150.0, sld * (1 + m))),
        **_OXIDE_INSTRUMENT,
        "sample_len": (sample_len, 5.0, 45.0),
    }

    ds = DataSet(name="reflectolearn")
    ds.x_raw, ds.y_raw = np.asarray(q, dtype=float), r
    ds.error_raw = np.maximum(r * 0.1, 1e-9)
    ds.run_command()

    model = Model()
    model.data = DataList([ds])
    model.set_script(_SCRIPT.format(
        init={k: float(v[0]) for k, v in ranges.items()},
        sub_sld=sub_sld, sub_rough=sub_rough, wavelength=wavelength, beam_width=beam_width,
    ))
    model.compile_script()

    pars = Parameters()
    by_name = {}
    for name, (init, lo, hi) in ranges.items():
        p = pars.append(f"v.set_{name}", model)
        p.min, p.max, p.value = lo, hi, float(np.clip(init, lo, hi))
        by_name[name] = p
    model.parameters = pars
    model.set_fom_func(fom_funcs.logR2)

    for _, active, steps in _STAGES:
        for name, p in by_name.items():
            p.fit = name in active
        model.bumps_update_parameters(
            model.bumps_fit(method="de", steps=max(1, int(steps * steps_scale)), pop=pop_size))

    model.evaluate_sim_func()
    return {
        **{name: float(p.value) for name, p in by_name.items()},
        "r_fit": np.asarray(model.data[0].y_sim),
        "fom": float(model.fom),
    }
