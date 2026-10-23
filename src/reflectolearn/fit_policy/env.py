"""The fit-policy environment: state, actions, transition and reward (DESIGN.md section 4 and
amendment 2).

An episode starts from a one-layer stack with only the intensity scale open and the log loss. Every
fit consumes evaluations from the episode's budget; the episode keeps the lowest-BIC state it has
seen, which is the returned model. `FitEnv.fork` makes a copy of the state that shares the episode
(budget and best state), for scripted protocols that try candidates (amendment 2).

Held-out points are stored on the episode for the terminal reward and the final score only; nothing
a policy reads (observation, BIC, residual features) uses them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.stack import L_MAX, LAYER_BOUNDS, LAYER_KEYS, LOSSES, Stack

Action = tuple

INSTR_GROUPS = ("bkg", "offset", "dq")  # i0 is always open
FIT_STEPS = (20, 100)
BUDGET = 3000
MAX_ACTIONS = 60
N_BANDS = 8
NEW_LAYER_SIGMA = 3.0
NEW_LAYER_D = 50.0
SPLIT_CONTRAST = 0.05
QC_MARGIN = 1.05  # estimate_rho acts only if q_c exceeds the substrate's by this factor
KIESSIG_RANGE = (20.0, 800.0)
RESIDUAL_Z = np.arange(20.0, 401.0, 1.0)


def _actions() -> tuple[Action, ...]:
    acts: list[Action] = [("add", p) for p in range(L_MAX)]
    acts += [("merge", i) for i in range(L_MAX - 1)]
    acts += [("split", i) for i in range(L_MAX)]
    acts += [(op, g, ell) for op in ("open", "close") for g in LAYER_KEYS for ell in range(L_MAX)]
    acts += [(op, g) for op in ("open_all", "close_all") for g in LAYER_KEYS]
    acts += [(op, g) for op in ("open", "close") for g in INSTR_GROUPS]
    acts += [("loss", name) for name in LOSSES]
    acts += [("fit", k) for k in FIT_STEPS]
    acts += [("estimate_rho",), ("estimate_d",), ("revert",), ("stop",)]
    return tuple(acts)


ACTIONS = _actions()
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
LAYER_CHANGES = ("add", "merge", "split", "revert")


@dataclass
class FitState:
    stack: Stack
    open_layers: np.ndarray  # (L, 3) bool, LAYER_KEYS order
    open_instr: np.ndarray  # (3,) bool, INSTR_GROUPS order
    loss: str = "log"

    def free_mask(self) -> np.ndarray:
        """Bool mask in Stack.vector() order: layers, then i0 (always), bkg, offset, dq."""
        return np.concatenate([self.open_layers.ravel(), [True], self.open_instr])

    def copy(self) -> FitState:
        return FitState(self.stack, self.open_layers.copy(), self.open_instr.copy(), self.loss)


@dataclass
class Episode:
    """What forks of one episode share."""

    q: np.ndarray
    y: np.ndarray
    scale: float
    q_ho: np.ndarray | None
    y_ho: np.ndarray | None
    budget: int
    used: int = 0
    best: FitState | None = None
    best_bic: float = math.inf
    kiessig: float = field(default=math.nan)
    qc_data: float = field(default=math.nan)


class FitEnv:
    def __init__(
        self,
        q: np.ndarray,
        y: np.ndarray,
        scale: float,
        start: Stack,
        *,
        q_ho: np.ndarray | None = None,
        y_ho: np.ndarray | None = None,
        budget: int = BUDGET,
        max_actions: int = MAX_ACTIONS,
    ):
        self.episode = Episode(q, y, scale, q_ho, y_ho, budget)
        self.episode.qc_data = st.critical_q(q, y)
        self.max_actions = max_actions
        self.state = FitState(
            start.clipped(), np.zeros((start.n_layers, 3), bool), np.zeros(len(INSTR_GROUPS), bool)
        )
        top_rho = start.layers[0, 2]
        self.episode.kiessig = st.kiessig_thickness(q, y, scale, top_rho, KIESSIG_RANGE)
        self.trace: list[Action] = []
        self.layer_changes = 0
        self.done = False
        self._evaluate()

    # ------------------------------------------------------------ shared data
    @property
    def q(self) -> np.ndarray:
        return self.episode.q

    @property
    def y(self) -> np.ndarray:
        return self.episode.y

    @property
    def scale(self) -> float:
        return self.episode.scale

    @property
    def budget_left(self) -> int:
        return self.episode.budget - self.episode.used

    @property
    def best(self) -> FitState:
        """The lowest-BIC state of the episode (set from the start state on)."""
        assert self.episode.best is not None
        return self.episode.best

    def fork(self) -> FitEnv:
        other = object.__new__(FitEnv)
        other.__dict__ |= self.__dict__
        other.state = self.state.copy()
        other.trace = list(self.trace)
        return other

    # ------------------------------------------------------------ quality
    def _evaluate(self) -> None:
        s = self.state.stack
        self.lam = st.counts(s, self.q, self.scale)
        self.dev = st.deviance(self.lam, self.y)
        self.bic = st.bic(self.dev, s.n_layers, len(self.q))
        if self.bic < self.episode.best_bic:
            self.episode.best_bic = self.bic
            self.episode.best = self.state.copy()

    def excess(self) -> float:
        """(D - E[D]) / SD[D] at the current model (protocol A step 7)."""
        return st.deviance_excess(self.lam, self.y)

    def _residual_period(self) -> tuple[float, float]:
        resid = np.log10(np.maximum(self.y, st.LOG_FLOOR_COUNTS)) - np.log10(self.lam)
        return st.fringe_spectrum(self.q, resid, 1.1 * self._qc(), RESIDUAL_Z)

    def _qc(self) -> float:
        qc = self.episode.qc_data
        return qc if math.isfinite(qc) else 4 * math.sqrt(math.pi * st.SUB_SLD * 1e-6)

    # ------------------------------------------------------------ actions
    def valid(self, action: Action) -> bool:
        if self.done:
            return False
        s, L = self.state, self.state.stack.n_layers
        match action:
            case ("add", p):
                return L < L_MAX and p <= L
            case ("merge", i):
                return L >= 2 and i <= L - 2
            case ("split", i):
                return L < L_MAX and i < L and s.stack.layers[i, 0] >= 2 * LAYER_BOUNDS["d"][0]
            case ("open" | "close" as op, g, ell):
                return ell < L and bool(s.open_layers[ell, LAYER_KEYS.index(g)]) != (op == "open")
            case ("open_all" | "close_all" as op, g):
                return bool(np.any(s.open_layers[:, LAYER_KEYS.index(g)] != (op == "open_all")))
            case ("open" | "close" as op, g):
                return bool(s.open_instr[INSTR_GROUPS.index(g)]) != (op == "open")
            case ("loss", name):
                return name != s.loss
            case ("fit", _):
                return self.budget_left > 0
            case ("estimate_rho",) | ("estimate_d",) | ("stop",):
                return True
            case ("revert",):
                best = self.episode.best
                return best is not None and not _same_stack(best.stack, s.stack)
        raise ValueError(action)

    def valid_actions(self) -> list[Action]:
        return [a for a in ACTIONS if self.valid(a)]

    def mask(self) -> np.ndarray:
        return np.array([self.valid(a) for a in ACTIONS])

    def step(self, action: Action) -> tuple[float, bool]:
        """Apply an action. Returns (reward, done)."""
        if not self.valid(action):
            raise ValueError(f"invalid action {action}")
        phi = -self.bic / len(self.q)
        self._apply(action)
        self.trace.append(action)
        if action[0] in LAYER_CHANGES:
            self.layer_changes += 1
        self._evaluate()
        reward = -self.bic / len(self.q) - phi
        if action == ("stop",) or self.budget_left <= 0 or len(self.trace) >= self.max_actions:
            self.done = True
            reward += self.terminal_reward()
        return reward, self.done

    def _apply(self, action: Action) -> None:
        s = self.state
        lay = s.stack.layers.copy()
        match action:
            case ("add", p):
                period, _ = self._residual_period()
                d = period if math.isfinite(period) else NEW_LAYER_D
                below = lay[p, 2] if p < len(lay) else st.SUB_SLD
                rho = lay[0, 2] if p == 0 else 0.5 * (lay[p - 1, 2] + below)
                lay = np.insert(lay, p, [d, NEW_LAYER_SIGMA, rho], axis=0)
                s.open_layers = np.insert(s.open_layers, p, True, axis=0)
            case ("merge", i):
                d1, d2 = lay[i, 0], lay[i + 1, 0]
                merged = [d1 + d2, lay[i, 1], (d1 * lay[i, 2] + d2 * lay[i + 1, 2]) / (d1 + d2)]
                lay = np.vstack([lay[:i], [merged], lay[i + 2 :]])
                opened = s.open_layers[i] | s.open_layers[i + 1]
                s.open_layers = np.vstack([s.open_layers[:i], [opened], s.open_layers[i + 2 :]])
            case ("split", i):
                d, sigma, rho = lay[i]
                top = [d / 2, sigma, rho * (1 + SPLIT_CONTRAST)]
                bottom = [d / 2, NEW_LAYER_SIGMA, rho * (1 - SPLIT_CONTRAST)]
                lay = np.vstack([lay[:i], [top, bottom], lay[i + 1 :]])
                s.open_layers = np.insert(s.open_layers, i, s.open_layers[i], axis=0)
            case ("open" | "close" as op, g, ell):
                s.open_layers[ell, LAYER_KEYS.index(g)] = op == "open"
            case ("open_all" | "close_all" as op, g):
                s.open_layers[:, LAYER_KEYS.index(g)] = op == "open_all"
            case ("open" | "close" as op, g):
                s.open_instr[INSTR_GROUPS.index(g)] = op == "open"
            case ("loss", name):
                s.loss = name
            case ("fit", k):
                stack, used, _ = st.fit(
                    s.stack,
                    s.free_mask(),
                    self.q,
                    self.y,
                    self.scale,
                    s.loss,
                    min(k, self.budget_left),
                )
                self.episode.used += used
                s.stack = stack
                return
            case ("estimate_rho",):
                qc, qc_sub = self.episode.qc_data, 4 * math.sqrt(math.pi * st.SUB_SLD * 1e-6)
                if math.isfinite(qc) and qc > QC_MARGIN * qc_sub:
                    lay[0, 2] = qc**2 / (16 * math.pi) * 1e6
            case ("estimate_d",):
                total = st.kiessig_thickness(self.q, self.y, self.scale, lay[0, 2], KIESSIG_RANGE)
                if math.isfinite(total):
                    lay[:, 0] *= total / lay[:, 0].sum()
            case ("revert",):
                self.state = self.best.copy()
                return
            case ("stop",):
                return
        s.stack = Stack(lay, s.stack.instr.copy()).clipped()

    # ------------------------------------------------------------ reward and result
    def terminal_reward(self) -> float:
        """-D_ho / n_ho of the returned model; 0 without held-out data (section 4.6)."""
        if self.episode.q_ho is None:
            return 0.0
        return -self.held_out_deviance() / len(self.episode.q_ho)

    def held_out_deviance(self) -> float:
        best = self.best.stack
        lam = st.counts(best, self.episode.q_ho, self.scale)
        return st.deviance(lam, self.episode.y_ho)

    def result(self) -> dict:
        """The returned model (lowest BIC visited) and the episode's cost."""
        best = self.best.stack
        n = len(self.q)
        dev = st.deviance(st.counts(best, self.q, self.scale), self.y)
        out = {
            "layers": best.layers.tolist(),
            "instr": best.instr.tolist(),
            "n_layers": best.n_layers,
            "bic": st.bic(dev, best.n_layers, n),
            "aic": st.aic(dev, best.n_layers),
            "deviance": dev,
            "evaluations": self.episode.used,
            "actions": len(self.trace),
            "trace": [list(a) for a in self.trace],
        }
        if self.episode.q_ho is not None:
            out["deviance_ho"] = self.held_out_deviance()
        return out

    # ------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        """Section 4.1 as one float vector; training points only."""
        s = self.state
        lay_lo = np.array([LAYER_BOUNDS[k][0] for k in LAYER_KEYS])
        lay_hi = np.array([LAYER_BOUNDS[k][1] for k in LAYER_KEYS])
        layers = np.zeros((L_MAX, 4))
        layers[: s.stack.n_layers, :3] = (s.stack.layers - lay_lo) / (lay_hi - lay_lo)
        layers[: s.stack.n_layers, 3] = 1.0
        ins_b = np.array([st.INSTR_BOUNDS[k] for k in st.INSTR_KEYS])
        instr = (s.stack.instr - ins_b[:, 0]) / (ins_b[:, 1] - ins_b[:, 0])
        open_layers = np.zeros((L_MAX, 3))
        open_layers[: s.stack.n_layers] = s.open_layers
        loss = np.array([s.loss == name for name in LOSSES], float)

        n = len(self.q)
        qc = self._qc()
        pois = st.residuals("poisson", self.lam, self.y, self.scale)
        logr = st.residuals("log", self.lam, self.y, self.scale)
        edges = np.linspace(1.1 * qc, self.q.max(), N_BANDS + 1)
        bands = np.zeros((N_BANDS, 3))
        for b in range(N_BANDS):
            sel = (self.q >= edges[b]) & (self.q <= edges[b + 1])
            if sel.any():
                bands[b] = pois[sel].mean(), np.sqrt((pois[sel] ** 2).mean()), logr[sel].mean()
        period, amp = self._residual_period()
        best_bic = self.episode.best_bic
        qmax = self.q.max()
        scalars = [
            s.stack.n_layers / L_MAX,
            np.nan_to_num(period / RESIDUAL_Z[-1]),
            amp,
            np.nan_to_num(self.episode.kiessig / KIESSIG_RANGE[1]),
            np.nan_to_num(self.episode.qc_data / qmax),
            np.nan_to_num(st.critical_q(self.q, self.lam) / qmax),
            self.dev / n,
            self.excess(),
            self.bic / n,
            best_bic / n,
            len(self.trace) / self.max_actions,
            self.episode.used / self.episode.budget,
            (ACTION_INDEX[self.trace[-1]] + 1) / len(ACTIONS) if self.trace else 0.0,
            self.layer_changes / 10,
        ]
        return np.concatenate(
            [
                layers.ravel(),
                instr,
                open_layers.ravel(),
                s.open_instr.astype(float),
                loss,
                bands.ravel(),
                np.asarray(scalars, float),
            ]
        )


def _same_stack(a: Stack, b: Stack) -> bool:
    return a.n_layers == b.n_layers and np.array_equal(a.vector(), b.vector())
