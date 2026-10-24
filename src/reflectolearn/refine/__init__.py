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
    poisson_deviance_loss,
)
from reflectolearn.refine.pipeline import (
    PARAM_KEYS,
    RefineConfig,
    RefineStage,
    fft_thickness,
    likelihood_stages,
    make_forward,
    per_curve_log_mse,
    refine,
    refine_multistart,
)

__all__ = [
    # pipeline
    "refine",
    "refine_multistart",
    "fft_thickness",
    "make_forward",
    "per_curve_log_mse",
    "RefineConfig",
    "RefineStage",
    "likelihood_stages",
    "PARAM_KEYS",
    # losses
    "LossFn",
    "LOSS_PRESETS",
    "log_mse_loss",
    "linear_mse_loss",
    "correlation_loss",
    "log_gradient_loss",
    "hybrid_linear_log_loss",
    "poisson_deviance_loss",
    "LOG_MSE",
    "LINEAR_MSE",
    "CORRELATION",
    "LOG_GRADIENT",
    "HYBRID",
]
