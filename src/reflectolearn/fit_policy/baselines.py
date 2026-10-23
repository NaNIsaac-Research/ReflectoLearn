"""Baselines B (random actions) and C (everything at once with BIC model selection), and the two
references C_oracle and C_refine (DESIGN.md sections 5.2–5.3)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.env import BUDGET, FitEnv
from reflectolearn.fit_policy.stack import L_MAX, Stack
from reflectolearn.refine import RefineConfig, make_forward, refine

NO_STOP_FIRST = 5


def random_policy(env: FitEnv, rng: np.random.Generator) -> FitEnv:
    """B: uniform over valid actions; stop is masked for the first NO_STOP_FIRST actions."""
    while not env.done:
        actions = env.valid_actions()
        if len(env.trace) < NO_STOP_FIRST and len(actions) > 1:
            actions = [a for a in actions if a != ("stop",)]
        env.step(actions[rng.integers(len(actions))])
    return env


def _score(stack: Stack, q: np.ndarray, y: np.ndarray, scale: float, q_ho=None, y_ho=None) -> dict:
    dev = st.deviance(st.counts(stack, q, scale), y)
    out = {
        "layers": stack.layers.tolist(),
        "instr": stack.instr.tolist(),
        "n_layers": stack.n_layers,
        "bic": st.bic(dev, stack.n_layers, len(q)),
        "aic": st.aic(dev, stack.n_layers),
        "deviance": dev,
    }
    if q_ho is not None:
        out["deviance_ho"] = st.deviance(st.counts(stack, q_ho, scale), y_ho)
    return out


def full_fit(
    start: Stack,
    extra: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    scale: float,
    *,
    q_ho=None,
    y_ho=None,
    budget: int = BUDGET,
) -> dict:
    """C: for L = 1..L_MAX, the start's layer over extra[:L-1], every parameter free, Poisson
    deviance, budget / L_MAX evaluations each; the lowest BIC is returned."""
    fits = []
    for L in range(1, L_MAX + 1):
        stack = Stack(np.vstack([start.layers, extra[: L - 1]]), start.instr.copy())
        fitted, used, status = st.fit(
            stack, np.ones(stack.n_params, bool), q, y, scale, "poisson", budget // L_MAX
        )
        fits.append((fitted, used, status))
    scores = [_score(f, q, y, scale, q_ho, y_ho) for f, _, _ in fits]
    best = int(np.argmin([s["bic"] for s in scores]))
    return scores[best] | {
        "evaluations": sum(u for _, u, _ in fits),
        "capped": [status == 0 for _, _, status in fits],
        "bic_by_L": [s["bic"] for s in scores],
    }


def oracle_fit(
    start: Stack,
    extra: np.ndarray,
    n_true: int,
    q: np.ndarray,
    y: np.ndarray,
    scale: float,
    *,
    q_ho=None,
    y_ho=None,
    budget: int = BUDGET,
) -> dict:
    """C_oracle: the true layer count, every parameter free, Poisson, the whole budget. Uses the
    truth; reported, never in a verdict."""
    stack = Stack(np.vstack([start.layers, extra[: n_true - 1]]), start.instr.copy())
    fitted, used, status = st.fit(
        stack, np.ones(stack.n_params, bool), q, y, scale, "poisson", budget
    )
    return _score(fitted, q, y, scale, q_ho, y_ho) | {"evaluations": used, "capped": [status == 0]}


REFINE_RANGES = SimpleNamespace(thickness=(10.0, 1200.0), roughness=(0.0, 15.0), sld=(5.0, 100.0))


def refine_fit(start: Stack, q: np.ndarray, y: np.ndarray, scale: float, *, q_ho=None, y_ho=None):
    """C_refine, single-layer class only: `refine` with RefineConfig() from the start, on the
    training points with counts > 0. Its model (film + top oxide, Kα2, footprint) is not the stack
    model, so only deviances and the film parameters are reported."""
    ok = y > 0
    substrate = torch.tensor([[st.SUB_SLD, st.SUB_ROUGH]])
    config = RefineConfig()
    p, info = refine(
        torch.tensor(start.layers[:1], dtype=torch.float32),
        substrate,
        torch.as_tensor(q, dtype=torch.float32),
        torch.as_tensor(np.log10(np.maximum(y, 1) / scale), dtype=torch.float32)[None],
        torch.as_tensor(ok, dtype=torch.float32)[None],
        REFINE_RANGES,
        wavelength=st.WAVELENGTH,
        config=config,
        device=torch.device("cpu"),
    )

    def lam(qq: np.ndarray) -> np.ndarray:
        simulate = make_forward(
            torch.as_tensor(qq, dtype=torch.float32),
            substrate,
            wavelength=st.WAVELENGTH,
            config=config,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            return scale * simulate(info["state"], info["active"])[0].double().numpy()

    out = {
        "layers": p.double().numpy().tolist(),
        "n_layers": 1,
        "deviance": st.deviance(lam(q), y),
    }
    if q_ho is not None:
        out["deviance_ho"] = st.deviance(lam(q_ho), y_ho)
    return out
