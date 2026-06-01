"""Post-AI physics refinement: curve-fitting losses + gradient refiner."""

from reflectolearn.refine.losses import (
    CORRELATION,
    HYBRID,
    LINEAR_MSE,
    LOG_GRADIENT,
    LOG_MSE,
    LOSS_PRESETS,
    LossFn,
    correlation_loss,
    hybrid_linear_log_loss,
    linear_mse_loss,
    log_gradient_loss,
    log_mse_loss,
)
from reflectolearn.refine.pipeline import PARAM_KEYS, RefineConfig, RefineStage, refine

__all__ = [
    # pipeline
    "refine",
    "RefineConfig",
    "RefineStage",
    "PARAM_KEYS",
    # losses
    "LossFn",
    "LOSS_PRESETS",
    "log_mse_loss",
    "linear_mse_loss",
    "correlation_loss",
    "log_gradient_loss",
    "hybrid_linear_log_loss",
    "LOG_MSE",
    "LINEAR_MSE",
    "CORRELATION",
    "LOG_GRADIENT",
    "HYBRID",
]
