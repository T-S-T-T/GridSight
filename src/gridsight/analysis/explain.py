"""Model explanations from TreeSHAP contributions (computed natively by LightGBM)."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Readable names for the dashboard and the "why" text.
LABELS: dict[str, str] = {
    "pd_price": "AEMO pre-dispatch price",
    "pd_demand": "AEMO pre-dispatch demand",
    "price_last": "latest price",
    "price_max_last": "latest 5-min price peak",
    "price_max_24h": "highest price in last 24h",
    "price_mean_24h": "average price in last 24h",
    "price_std_24h": "price volatility in last 24h",
    "spike_count_7d": "spikes in last 7 days",
    "demand_last": "latest demand",
    "demand_lag_day": "demand same time yesterday",
    "demand_lag_week": "demand same time last week",
    "wx_temp_c": "forecast temperature",
    "wx_cdh": "cooling degree hours",
    "wx_ghi_wm2": "forecast solar irradiance",
    "wx_wind_100m_ms": "forecast wind speed",
    "tod": "time of day",
    "dow": "day of week",
    "horizon": "forecast horizon",
}


def label(feature: str) -> str:
    return LABELS.get(feature, feature.replace("_", " "))


def global_importance(model: Any, X: pd.DataFrame, top: int = 15, **kw: Any) -> pd.DataFrame:
    """Mean absolute SHAP value per feature over ``X``."""
    contrib = model.contributions(X, **kw).drop(columns="bias")
    imp = contrib.abs().mean().sort_values(ascending=False).head(top)
    return pd.DataFrame({"feature": imp.index, "label": [label(f) for f in imp.index], "mean_abs_shap": imp.to_numpy()})


def top_drivers(model: Any, row: pd.DataFrame, n: int = 5, **kw: Any) -> pd.DataFrame:
    """The ``n`` features pushing one prediction up or down the most."""
    contrib = model.contributions(row.iloc[[0]], **kw).drop(columns="bias").iloc[0]
    order = contrib.abs().sort_values(ascending=False).head(n).index
    return pd.DataFrame(
        {
            "feature": order,
            "label": [label(f) for f in order],
            "value": [row.iloc[0][f] for f in order],
            "shap": contrib[order].to_numpy(),
            "direction": ["raises" if v > 0 else "lowers" for v in contrib[order]],
        }
    )


def explain_spike(model: Any, row: pd.DataFrame, n: int = 3) -> str:
    """Plain-English summary of one spike alert."""
    p = float(model.predict_proba(row.iloc[[0]])[0])
    drivers = top_drivers(model, row, n)
    parts = [f"{r.label} ({r.value:,.4g}) {r.direction} the risk" for r in drivers.itertuples()]
    region = row.iloc[0].get("region", "")
    when = row.iloc[0].get("target_utc", "")
    return f"Spike risk {p:.0%} for {region} at {when}: " + "; ".join(parts) + "."
