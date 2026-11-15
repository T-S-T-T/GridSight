"""Exploratory analysis of the NEM: the numbers behind the "market story".

All functions take silver-layer frames and return tidy DataFrames, so the same code feeds the
notebook, the CLI report and the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridsight.timeutils import MARKET_TZ


def _local(df: pd.DataFrame, col: str = "interval_utc") -> pd.DataFrame:
    t = df[col].dt.tz_convert(MARKET_TZ)
    return df.assign(
        hour=t.dt.hour + t.dt.minute / 60.0,
        month=t.dt.month,
        year=t.dt.year,
        season=t.dt.month.map(
            {12: "Summer", 1: "Summer", 2: "Summer", 3: "Autumn", 4: "Autumn", 5: "Autumn",
             6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring", 11: "Spring"}
        ),
    )


def to_half_hour(df: pd.DataFrame, value_cols: list[str], by: str = "region") -> pd.DataFrame:
    """Average 5-minute rows into 30-minute intervals (interval start)."""
    g = df.assign(interval_utc=df["interval_utc"].dt.floor("30min"))
    return g.groupby([by, "interval_utc"], as_index=False)[value_cols].mean()


def duck_curve(demand: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    """Mean operational demand by time of day (half-hour, market time) per region and period."""
    d = _local(demand)
    d["hour"] = (d["hour"] * 2).floordiv(1) / 2
    return d.groupby(["region", by, "hour"], as_index=False)["total_demand_mw"].mean()


def duck_depth(demand: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    """Evening peak minus midday trough of the average daily profile: how deep the duck is."""
    curve = duck_curve(demand, by=by)
    mid = curve[curve["hour"].between(10, 15)].groupby(["region", by])["total_demand_mw"].min()
    eve = curve[curve["hour"].between(17, 21)].groupby(["region", by])["total_demand_mw"].max()
    out = pd.DataFrame({"midday_min_mw": mid, "evening_max_mw": eve}).reset_index()
    out["depth_mw"] = out["evening_max_mw"] - out["midday_min_mw"]
    return out


def negative_prices(price: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    """Share of intervals and equivalent hours with a negative price, per region and period."""
    p = _local(price)
    p["negative"] = p["rrp"] < 0
    g = p.groupby(["region", by])
    out = g["negative"].agg(share="mean", intervals="sum").reset_index()
    out["hours"] = out["intervals"] * 5 / 60
    return out


def spike_stats(price: pd.DataFrame, threshold: float = 300.0) -> pd.DataFrame:
    """Spike frequency and severity per region."""
    p = price.assign(spike=price["rrp"] > threshold)
    g = p.groupby("region")
    out = pd.DataFrame(
        {
            "intervals": g.size(),
            "spikes": g["spike"].sum(),
            "spike_share": g["spike"].mean(),
            "mean_spike_price": g.apply(lambda x: x.loc[x["spike"], "rrp"].mean(), include_groups=False),
            "max_price": g["rrp"].max(),
            "spike_revenue_share": g.apply(
                lambda x: x.loc[x["spike"], "rrp"].clip(lower=0).sum() / max(x["rrp"].clip(lower=0).sum(), 1e-9),
                include_groups=False,
            ),
        }
    )
    return out.reset_index()


def spikes_by_hour(price: pd.DataFrame, threshold: float = 300.0) -> pd.DataFrame:
    p = _local(price)
    p["hour"] = p["hour"].floordiv(1).astype(int)
    p["spike"] = p["rrp"] > threshold
    return p.groupby(["region", "hour"], as_index=False)["spike"].mean().rename(columns={"spike": "spike_share"})


def price_duration_curve(price: pd.DataFrame, points: int = 101) -> pd.DataFrame:
    """Price at each percentile of time, per region (sorted high to low)."""
    q = np.linspace(0, 1, points)
    rows = []
    for region, g in price.groupby("region"):
        vals = np.quantile(g["rrp"].to_numpy(), 1 - q)
        rows.append(pd.DataFrame({"region": region, "share_of_time": q, "rrp": vals}))
    return pd.concat(rows, ignore_index=True)


def predispatch_accuracy(
    predispatch: pd.DataFrame,
    price: pd.DataFrame,
    demand: pd.DataFrame,
    threshold: float = 300.0,
) -> pd.DataFrame:
    """AEMO pre-dispatch error by horizon (30-minute steps between run and target interval).

    Returns MAE for price and demand, and how often a real spike was forecast (spike recall)."""
    actual = to_half_hour(price, ["rrp"]).merge(to_half_hour(demand, ["total_demand_mw"]), on=["region", "interval_utc"])
    df = predispatch.merge(actual, on=["region", "interval_utc"], how="inner")
    df["horizon_steps"] = np.ceil((df["interval_utc"] - df["run_utc"]) / pd.Timedelta("30min")).astype(int).clip(lower=0)
    df["price_abs_err"] = (df["rrp_forecast"] - df["rrp"]).abs()
    df["demand_abs_pct_err"] = (df["demand_forecast"] - df["total_demand_mw"]).abs() / df["total_demand_mw"].abs()
    df["actual_spike"] = df["rrp"] > threshold
    df["forecast_spike"] = df["rrp_forecast"] > threshold
    g = df.groupby(["region", "horizon_steps"])
    out = g.agg(
        n=("rrp", "size"),
        price_mae=("price_abs_err", "mean"),
        price_mae_no_spikes=("price_abs_err", lambda s: s[~df.loc[s.index, "actual_spike"]].mean()),
        demand_mape=("demand_abs_pct_err", "mean"),
    ).reset_index()
    spikes = df[df["actual_spike"]].groupby(["region", "horizon_steps"])["forecast_spike"].mean()
    out = out.merge(spikes.rename("spike_recall").reset_index(), on=["region", "horizon_steps"], how="left")
    return out


def market_summary(price: pd.DataFrame, demand: pd.DataFrame, threshold: float = 300.0) -> pd.DataFrame:
    """One row per region: average price, volatility, demand range, negative and spike shares."""
    p = price.groupby("region")["rrp"]
    d = demand.groupby("region")["total_demand_mw"]
    out = pd.DataFrame(
        {
            "mean_price": p.mean(),
            "median_price": p.median(),
            "price_std": p.std(),
            "negative_share": p.apply(lambda s: (s < 0).mean()),
            "spike_share": p.apply(lambda s: (s > threshold).mean()),
            "mean_demand_mw": d.mean(),
            "min_demand_mw": d.min(),
            "max_demand_mw": d.max(),
        }
    )
    return out.reset_index()
