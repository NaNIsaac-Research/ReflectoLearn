"""Environment and baselines for fitting XRR the way a person does (docs/fit_policy/DESIGN.md)."""

from reflectolearn.fit_policy.baselines import full_fit, oracle_fit, random_policy, refine_fit
from reflectolearn.fit_policy.env import ACTIONS, FitEnv, FitState
from reflectolearn.fit_policy.protocol import A1, A2, A3, A4, VARIANTS, Protocol
from reflectolearn.fit_policy.protocol import run as run_protocol
from reflectolearn.fit_policy.stack import Stack

__all__ = [
    "A1",
    "A2",
    "A3",
    "A4",
    "ACTIONS",
    "VARIANTS",
    "FitEnv",
    "FitState",
    "Protocol",
    "Stack",
    "full_fit",
    "oracle_fit",
    "random_policy",
    "refine_fit",
    "run_protocol",
]
