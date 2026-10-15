import math
from typing import Final, overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

r_e: Final[float] = 2.8179403227e-5  # AA

@overload
def tth2q(tth: float, wavelen: float = 1.54) -> float: ...

@overload
def tth2q(tth: NDArray[np.float64], wavelen: float = 1.54) -> NDArray[np.float64]: ...

def tth2q(tth, wavelen: float = 1.5406):
    """
    Convert 2θ (in degrees) and wavelength (in Å) to scattering vector q (in 1/Å).
    Formula: q = (4 * pi / lambda) * sin(theta)
    """
    # tth is 2*theta, so divide by 2 to get theta
    th_rad = np.deg2rad(0.5 * tth)
    result = (4 * np.pi / wavelen) * np.sin(th_rad)

    if isinstance(tth, (int, float)):
        return float(result)
    return result


def apply_tth_offset(q: Tensor, offset_deg: Tensor, wavelength: float) -> Tensor:
    """Apply a q-axis zero-point (alignment) offset: add offset[deg] to the
    measured 2θ, then convert back to q.

    Differentiable (torch). q: (Q,) or (B, Q); offset_deg: (B, 1) -> returns (B, Q).
    """
    sin_th = (q * wavelength / (4.0 * math.pi)).clamp(-1.0, 1.0)
    tth = 2.0 * torch.rad2deg(torch.asin(sin_th)) + offset_deg
    return (4.0 * math.pi / wavelength) * torch.sin(torch.deg2rad(0.5 * tth))
