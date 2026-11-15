"""Rolling-horizon battery simulation: turns forecast accuracy into dollars.

Every 30 minutes the battery re-optimises over the next ``horizon_hours`` using a price forecast,
commits only the first step, and is paid at the *actual* 5-minute prices of that half-hour.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from gridsight.battery.lp import BatterySpec, optimise
from gridsight.timeutils import MARKET_TZ

log = logging.getLogger(__name__)

ForecastFn = Callable[[pd.Timestamp, int], np.ndarray]


# ---------------------------------------------------------------- forecast providers


def forecast_curves(pred: pd.DataFrame, value_col: str = "q0.5") -> dict[pd.Timestamp, pd.Series]:
    """Per issue time, the forecast by horizon step (1-based)."""
    return {t: g.set_index("horizon")[value_col].sort_index() for t, g in pred.groupby("issued_utc")}


def curve_to_steps(curve: pd.Series, steps: int) -> np.ndarray:
    """Interpolate a sparse horizon curve (e.g. horizons 1, 2, 4, 8, 48) to every step 1..steps."""
    curve = curve.dropna()
    if curve.empty:
        return np.full(steps, np.nan)
    h = np.arange(1, steps + 1)
    return np.interp(h, curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float))


def from_predictions(pred: pd.DataFrame, value_col: str = "q0.5") -> ForecastFn:
    curves = forecast_curves(pred, value_col)

    def fn(t: pd.Timestamp, steps: int) -> np.ndarray:
        c = curves.get(t)
        return curve_to_steps(c, steps) if c is not None else np.full(steps, np.nan)

    return fn


def perfect_foresight(actual_hh: pd.Series) -> ForecastFn:
    """Uses the actual half-hour prices: the upper bound on what any forecast can earn."""

    def fn(t: pd.Timestamp, steps: int) -> np.ndarray:
        idx = pd.date_range(t, periods=steps, freq="30min")
        return actual_hh.reindex(idx).ffill().bfill().to_numpy(dtype=float)

    return fn


def risk_aware(base: ForecastFn, spike_pred: pd.DataFrame, spike_value: float = 1000.0) -> ForecastFn:
    """Raise the forecast price by p_spike x spike_value so the battery holds charge when a spike
    is likely, even if the p50 price forecast is calm."""
    curves = forecast_curves(spike_pred, "p_spike")

    def fn(t: pd.Timestamp, steps: int) -> np.ndarray:
        p = base(t, steps)
        c = curves.get(t)
        if c is not None and not c.dropna().empty:
            ps = np.zeros(steps)
            known = c.dropna()
            for h, v in known.items():
                if 1 <= int(h) <= steps:
                    ps[int(h) - 1] = v
            p = p + ps * spike_value
        return p

    return fn


# ---------------------------------------------------------------- simulation


def half_hour_prices(price_5min: pd.DataFrame) -> pd.Series:
    s = price_5min.set_index("interval_utc")["rrp"].sort_index()
    return s.resample("30min").mean()


def simulate(
    price_5min: pd.DataFrame,
    forecast: ForecastFn,
    spec: BatterySpec,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run the rolling optimisation. ``price_5min`` holds one region's actual prices."""
    p5 = price_5min.set_index("interval_utc")["rrp"].sort_index()
    hh_index = p5.resample("30min").mean().index
    if start is not None:
        hh_index = hh_index[hh_index >= start]
    if end is not None:
        hh_index = hh_index[hh_index < end]
    soc, rows, steps = spec.soc_initial_mwh, [], spec.horizon_steps
    for t in hh_index:
        f = forecast(t, steps)
        if np.isnan(f).all():
            c = d = 0.0
        else:
            f = pd.Series(f).ffill().bfill().to_numpy()
            plan = optimise(f, spec, soc0=soc)
            c, d = float(plan["charge_mw"].iloc[0]), float(plan["discharge_mw"].iloc[0])
        actual = p5[(p5.index >= t) & (p5.index < t + pd.Timedelta("30min"))]
        energy_net = (d - c) * spec.dt  # MWh exported in this half-hour
        revenue = float(actual.mean() * energy_net) if len(actual) else 0.0
        degradation = spec.degradation_aud_per_mwh * d * spec.dt
        soc = soc + spec.eta_charge * c * spec.dt - d * spec.dt / spec.eta_discharge
        soc = float(np.clip(soc, spec.soc_min_mwh, spec.soc_max_mwh))
        rows.append(
            {"interval_utc": t, "charge_mw": c, "discharge_mw": d, "soc_mwh": soc,
             "price": float(actual.mean()) if len(actual) else np.nan,
             "revenue_aud": revenue - degradation}
        )
    return pd.DataFrame(rows)


def rule_based(price_5min: pd.DataFrame, spec: BatterySpec, start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None, charge_hours=(10, 15), discharge_hours=(17, 21)) -> pd.DataFrame:
    """Naive floor: charge 10:00-15:00, discharge 17:00-21:00 market time, full power."""
    hh = half_hour_prices(price_5min)
    if start is not None:
        hh = hh[hh.index >= start]
    if end is not None:
        hh = hh[hh.index < end]
    soc, rows = spec.soc_initial_mwh, []
    for t, price in hh.items():
        hour = t.tz_convert(MARKET_TZ).hour
        c = d = 0.0
        if charge_hours[0] <= hour < charge_hours[1]:
            c = min(spec.power_mw, (spec.soc_max_mwh - soc) / (spec.eta_charge * spec.dt))
        elif discharge_hours[0] <= hour < discharge_hours[1]:
            d = min(spec.power_mw, (soc - spec.soc_min_mwh) * spec.eta_discharge / spec.dt)
        c, d = max(c, 0.0), max(d, 0.0)
        soc = soc + spec.eta_charge * c * spec.dt - d * spec.dt / spec.eta_discharge
        revenue = price * (d - c) * spec.dt - spec.degradation_aud_per_mwh * d * spec.dt
        rows.append({"interval_utc": t, "charge_mw": c, "discharge_mw": d, "soc_mwh": soc, "price": price, "revenue_aud": revenue})
    return pd.DataFrame(rows)


def summarise(runs: dict[str, pd.DataFrame], spec: BatterySpec, reference: str = "perfect_foresight") -> pd.DataFrame:
    """Revenue, share of the perfect-foresight upper bound, and cycles per day per strategy."""
    rows = []
    ref = runs[reference]["revenue_aud"].sum() if reference in runs else np.nan
    usable = spec.soc_max_mwh - spec.soc_min_mwh
    for name, r in runs.items():
        days = max(len(r) * spec.dt / 24, 1e-9)
        total = r["revenue_aud"].sum()
        rows.append(
            {
                "strategy": name,
                "revenue_aud": total,
                "pct_of_perfect": 100 * total / ref if ref and np.isfinite(ref) and ref > 0 else np.nan,
                "revenue_per_day_aud": total / days,
                "cycles_per_day": (r["discharge_mw"].sum() * spec.dt / usable) / days,
                "worst_day_aud": r.set_index("interval_utc")["revenue_aud"].resample("1D").sum().min(),
            }
        )
    return pd.DataFrame(rows).sort_values("revenue_aud", ascending=False).reset_index(drop=True)


def run_strategies(
    price_5min: pd.DataFrame,
    price_pred: pd.DataFrame,
    spec: BatterySpec,
    spike_pred: pd.DataFrame | None = None,
    model_name: str = "price_lgbm",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """All five strategies over the period covered by ``price_pred`` (one region)."""
    start, end = price_pred["issued_utc"].min(), price_pred["issued_utc"].max() + pd.Timedelta("30min")
    actual_hh = half_hour_prices(price_5min)
    model_pred = price_pred[price_pred["model"] == model_name]
    aemo_pred = price_pred[price_pred["model"] == "aemo_predispatch"]
    runs: dict[str, pd.DataFrame] = {
        "perfect_foresight": simulate(price_5min, perfect_foresight(actual_hh), spec, start, end),
        "gridsight_p50": simulate(price_5min, from_predictions(model_pred), spec, start, end),
        "rule_based": rule_based(price_5min, spec, start, end),
    }
    if not aemo_pred.empty and aemo_pred["q0.5"].notna().any():
        runs["aemo_predispatch"] = simulate(price_5min, from_predictions(aemo_pred), spec, start, end)
    if spike_pred is not None and not spike_pred.empty:
        sp = spike_pred[spike_pred["model"] == "spike_lgbm"] if "model" in spike_pred else spike_pred
        runs["gridsight_risk_aware"] = simulate(price_5min, risk_aware(from_predictions(model_pred), sp), spec, start, end)
    return summarise(runs, spec), runs
