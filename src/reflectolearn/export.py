"""Export the trained CNN to ONNX, with the input contract a non-Python caller needs.

The ONNX graph maps the model input x (1, 2, Q) straight to physical film parameters: the
target de-normalisation (· std + mean) is part of the graph. The JSON written next to it holds
what the graph cannot: the q grid, the input preprocessing of ``predict.preprocess_measured``
and the parameter order and units.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from reflectolearn.config import Config
from reflectolearn.data.generator import LOG_FLOOR
from reflectolearn.refine import PARAM_KEYS

PARAM_UNITS = {"thickness": "Angstrom", "roughness": "Angstrom", "sld": "1e-6 / Angstrom^2"}


class PhysicalOutput(nn.Module):
    """The CNN followed by its target de-normalisation: x (B, 2, Q) -> (B, 3) physical values."""

    def __init__(self, model: nn.Module, mean: Tensor, std: Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean.detach().reshape(1, -1).float())
        self.register_buffer("std", std.detach().reshape(1, -1).float())

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x) * self.std + self.mean


def input_contract(cfg: Config) -> dict:
    """What a caller must reproduce to build x from a measured curve (see preprocess_measured)."""
    q = cfg.q_grid
    return {
        "input": {
            "name": "x",
            "shape": [1, 2, int(q.numel())],
            "channels": ["log10 R, normalised", "mask (1 measured, 0 not)"],
            "q_grid": {
                "start": float(q[0]),
                "stop": float(q[-1]),
                "points": int(q.numel()),
                "spacing": "linear, float32 as torch.linspace builds it",
                "unit": "1/Angstrom",
            },
            "preprocess": [
                (
                    "interpolate the intensity linearly onto q_grid; points outside the measured q "
                    "range are unmeasured (mask 0)"
                ),
                "clip the intensity at 0 and divide by its maximum over the measured points",
                (
                    "R = max(that, 1e-15); channel 0 = min(log10 R, 0) where measured, log_floor "
                    "elsewhere"
                ),
            ],
            "log_floor": float(LOG_FLOOR),
        },
        "output": {
            "name": "params",
            "order": list(PARAM_KEYS),
            "units": [PARAM_UNITS[k] for k in PARAM_KEYS],
        },
        "assumptions": {
            "wavelength_angstrom": cfg.simulation.wavelength,
            "model": "one film on a substrate, ambient on top",
            "substrate_sld_range": list(cfg.substrate.sld),
            "substrate_roughness_range": list(cfg.substrate.roughness),
            "param_ranges": {k: list(getattr(cfg.param_ranges, k)) for k in PARAM_KEYS},
        },
    }


def export_onnx(
    model: nn.Module, mean: Tensor, std: Tensor, cfg: Config, out: Path, note: str | None = None
) -> Path:
    """Write ``out`` (ONNX) and ``out`` with suffix .json (input contract). Returns the JSON path.

    ``note`` says what the model is (training run, status) for whoever shows its output.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapped = PhysicalOutput(model.cpu().eval(), mean.cpu(), std.cpu()).eval()
    x = torch.zeros(1, 2, cfg.simulation.q_points)
    # weights inside the file: browsers and other runtimes then need one file, not .onnx + .onnx.data
    torch.onnx.export(
        wrapped,
        (x,),
        out,
        input_names=["x"],
        output_names=["params"],
        dynamo=True,
        external_data=False,
    )
    meta = out.with_suffix(".json")
    contract = input_contract(cfg)
    if note:
        contract["model"] = {"note": note}
    meta.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return meta
