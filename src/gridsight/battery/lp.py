"""Battery dispatch as a linear program, solved with HiGHS (via scipy.optimize.linprog).

Decision variables per step t: charge power c_t and discharge power d_t (MW).
State of charge (MWh) is implied:  s_t = s_0 + sum_{k<=t} (eta_c * c_k - d_k / eta_d) * dt

    maximise   sum_t  p_t * (d_t - c_t) * dt  -  kappa * sum_t d_t * dt  +  v * s_T
    subject to 0 <= c_t, d_t <= P_max
               S_min <= s_t <= S_max   for every t

``v`` values energy still stored at the end of the horizon. Without it the optimiser always
empties the battery at the horizon's end (a well-known end effect of rolling optimisation).
By default v = mean forecast price x discharge efficiency.

The design doc planned cvxpy; SciPy's built-in HiGHS solves the same LP with one less dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import hstack, tril, vstack

from gridsight.config import battery_config


@dataclass(frozen=True)
class BatterySpec:
    power_mw: float = 100.0
    soc_min_mwh: float = 20.0
    soc_max_mwh: float = 200.0
    soc_initial_mwh: float = 100.0
    eta_charge: float = 0.92
    eta_discharge: float = 0.92
    degradation_aud_per_mwh: float = 10.0
    step_minutes: int = 30
    horizon_hours: int = 48

    @property
    def dt(self) -> float:
        return self.step_minutes / 60.0

    @property
    def horizon_steps(self) -> int:
        return int(self.horizon_hours * 60 / self.step_minutes)

    @classmethod
    def from_config(cls, overrides: dict[str, Any] | None = None) -> BatterySpec:
        cfg = {**battery_config(), **(overrides or {})}
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in cfg.items() if k in fields})


def optimise(
    prices: np.ndarray,
    spec: BatterySpec,
    soc0: float | None = None,
    terminal_value: float | None = None,
) -> pd.DataFrame:
    """Optimal schedule for a price vector ($/MWh per step). Returns charge_mw, discharge_mw,
    soc_mwh (end of step) and net_mw (positive = export).

    ``terminal_value`` ($/MWh stored at the end): None = mean price x discharge efficiency."""
    p = np.asarray(prices, dtype=float)
    if np.isnan(p).any():
        raise ValueError("price vector contains NaN")
    T, dt = len(p), spec.dt
    s0 = spec.soc_initial_mwh if soc0 is None else float(soc0)
    # Objective (minimise): charge costs p*dt, discharge earns p*dt minus degradation.
    v = float(np.mean(p)) * spec.eta_discharge if terminal_value is None else float(terminal_value)
    v = max(v, 0.0)
    cost = np.concatenate([p * dt, -(p - spec.degradation_aud_per_mwh) * dt])
    # + v * s_T, where s_T = s0 + sum(eta_c * c - d / eta_d) * dt  (minimising, so subtract)
    cost[:T] -= v * spec.eta_charge * dt
    cost[T:] += v * dt / spec.eta_discharge
    L = tril(np.ones((T, T)), format="csr")  # cumulative sum
    soc_coef = hstack([L * (spec.eta_charge * dt), L * (-dt / spec.eta_discharge)], format="csr")
    A = vstack([soc_coef, -soc_coef], format="csr")
    b = np.concatenate([np.full(T, spec.soc_max_mwh - s0), np.full(T, s0 - spec.soc_min_mwh)])
    res = linprog(cost, A_ub=A, b_ub=b, bounds=[(0.0, spec.power_mw)] * (2 * T), method="highs")
    if res.status != 0:
        raise RuntimeError(f"battery LP failed: {res.message}")
    c, d = res.x[:T], res.x[T:]
    # Remove numerically-simultaneous charge and discharge.
    net = d - c
    c, d = np.clip(-net, 0, None), np.clip(net, 0, None)
    soc = s0 + np.cumsum(spec.eta_charge * c * dt - d * dt / spec.eta_discharge)
    return pd.DataFrame({"charge_mw": c, "discharge_mw": d, "soc_mwh": soc, "net_mw": d - c})

