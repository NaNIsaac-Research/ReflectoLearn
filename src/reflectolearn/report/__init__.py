"""Report/figure generation utilities (for slides and documentation).

These helpers reuse the production physics/augmentation code to render
explanatory figures; they are not part of the train/inference path.
"""

from reflectolearn.report.figures import (
    domain_randomization_figure,
    model_architecture_figure,
    refine_figure,
    refine_process_figure,
    sim_to_real_figure,
)

__all__ = [
    "domain_randomization_figure",
    "sim_to_real_figure",
    "model_architecture_figure",
    "refine_figure",
    "refine_process_figure",
]
