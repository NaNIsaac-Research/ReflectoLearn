from typing import Final, overload

import numpy as np
from numpy.typing import NDArray

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
