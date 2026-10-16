"""Stitch XRR scans measured in consecutive 2θ ranges with different attenuators.

The last scan is the intensity reference. Going backwards, each earlier scan is
scaled so its last point equals the first point of the following scan, and that
duplicate boundary point is dropped. Adjacent scans must share one x value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def stitch_scans(scans: list[np.ndarray]) -> np.ndarray:
    """Combine scans of shape (N, 2+) [x, intensity, (error, ...)] in increasing-x order.

    Every column after x is rescaled together.
    """
    if not scans:
        return np.empty((0, 3))
    out = [np.asarray(scans[-1], dtype=float)]
    head = out[0][0, 1]
    for scan in reversed(scans[:-1]):
        scan = np.asarray(scan, dtype=float).copy()
        scan[:, 1:] *= head / scan[-1, 1]
        scan = scan[:-1]
        head = scan[0, 1]
        out.append(scan)
    return np.concatenate(out[::-1])


def read_scan(path: str | Path) -> np.ndarray:
    """Read a whitespace-separated scan file (x, intensity, error), no header."""
    return np.loadtxt(path, ndmin=2)
