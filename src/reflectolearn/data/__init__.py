"""Data pipeline: instrument augmentations + streaming synthetic generator.

(``dataset.py`` is a legacy offline HDF5 dataset, kept for reference and not
re-exported here — the streaming generator is the active data path.)
"""

from reflectolearn.data.augmentations import XRRAugmentor
from reflectolearn.data.generator import (
    LOG_FLOOR,
    XRRStreamingDataset,
    generate_synthetic_batch,
    get_xrr_dataloader,
    param_stats,
)

__all__ = [
    "XRRAugmentor",
    "XRRStreamingDataset",
    "generate_synthetic_batch",
    "get_xrr_dataloader",
    "param_stats",
    "LOG_FLOOR",
]
