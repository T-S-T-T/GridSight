"""Feature pipeline: silver tables -> one row per (issued_utc, target_utc, region).

**Leakage rule.** A forecast issued at ``issued_utc`` (always a half-hour boundary) may only use:

* actuals for half-hours that *ended* at or before ``issued_utc``;
* weather forecasts with ``issued_utc`` (of the forecast) at or before the issue time;
* pre-dispatch runs published at or before the issue time;
* calendar facts about the target, which are known in advance.

``tests/unit/test_phase3_models.py::test_features_do_not_use_the_future`` enforces this by
scrambling every value after a cut-off and checking that no earlier feature row changes.

The target interval for horizon ``h`` (in 30-minute steps) starts at
``issued_utc + (h - 1) * 30 min``: horizon 1 is the half-hour that starts at the issue time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gridsight.config import regions_config
from gridsight.ingest.reference import national_holidays
from gridsight.timeutils import MARKET_TZ, is_five_minute_settlement

HH = pd.Timedelta("30min")
CALENDAR_COLS = ["tod", "tod_sin", "tod_cos", "dow", "dow_sin", "dow_cos", "month", "doy_sin", "doy_cos", "is_weekend", "is_holiday"]
TARGETS = ["y_demand", "y_price", "y_spike"]


# ---------------------------------------------------------------- half-hourly base table


def half_hourly(
    price: pd.DataFrame,
    demand: pd.DataFrame,
    spike_threshold: float = 300.0,
) -> pd.DataFrame:
    """Half-hour actuals per region: mean price, max 5-min price, spike flag, mean demand."""
    p = price.assign(interval_utc=price["interval_utc"].dt.floor("30min"))
    p = p.groupby(["region", "interval_utc"]).agg(price=("rrp", "mean"), price_max=("rrp", "max"), n5=("rrp", "size"))
    d = demand.assign(interval_utc=demand["interval_utc"].dt.floor("30min"))
    d = d.groupby(["region", "interval_utc"]).agg(demand=("total_demand_mw", "mean"))
    hh = p.join(d, how="outer").reset_index()
    hh["spike"] = (hh["price_max"] > spike_threshold).astype(float)
    hh.loc[hh["price_max"].isna(), "spike"] = np.nan
    return hh.sort_values(["region", "interval_utc"]).reset_index(drop=True)


def regional_weather(weather: pd.DataFrame, regions: Sequence[str]) -> pd.DataFrame:
    """Weight site forecasts into one series per (region, issued_utc, valid_utc)."""
    cfg = regions_config()
    cols = [c for c in ("temp_c", "ghi_wm2", "wind_100m_ms", "cloud_pct", "humidity_pct") if c in weather.columns]
    out = []
    for region in regions:
        weights = cfg[region]["sites"]
        w = weather[weather["site_id"].isin(weights)].copy()
        if w.empty:
            continue
        w["_w"] = w["site_id"].map(weights)
        for c in cols:
            w[c] = w[c] * w["_w"]
        agg = w.groupby(["issued_utc", "valid_utc"])[[*cols, "_w"]].sum()
        agg[cols] = agg[cols].div(agg["_w"], axis=0)
        out.append(agg.drop(columns="_w").reset_index().assign(region=region))
    if not out:
        return pd.DataFrame(columns=["region", "issued_utc", "valid_utc", *cols])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- calendar


def calendar_features(target_utc: pd.Series) -> pd.DataFrame:
    local = target_utc.dt.tz_convert(MARKET_TZ)
    tod = local.dt.hour + local.dt.minute / 60.0
    dow = local.dt.dayofweek
    doy = local.dt.dayofyear
    years = range(int(local.dt.year.min()), int(local.dt.year.max()) + 1) if len(local) else []
    hol = pd.concat([national_holidays(y) for y in years]) if len(local) else pd.DataFrame(columns=["date"])
    hol_dates = set(pd.to_datetime(hol["date"]).dt.date)
    return pd.DataFrame(
        {
            "tod": tod,
            "tod_sin": np.sin(2 * np.pi * tod / 24),
            "tod_cos": np.cos(2 * np.pi * tod / 24),
            "dow": dow,
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month": local.dt.month,
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_weekend": (dow >= 5).astype(int),
            "is_holiday": local.dt.date.map(lambda d: d in hol_dates).astype(int),
        },
        index=target_utc.index,
    )


# ---------------------------------------------------------------- main builder


def _lookup(series: pd.Series, times: pd.Series, known_before: pd.Series) -> np.ndarray:
    """series[times], set to NaN where the looked-up interval had not ended by ``known_before``."""
    vals = series.reindex(times.to_numpy()).to_numpy(dtype=float, copy=True)
    vals[(times + HH > known_before).to_numpy()] = np.nan
    return vals


def build_features(
    price: pd.DataFrame,
    demand: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    predispatch: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 2, 4, 8, 16, 48),
    spike_threshold: float = 300.0,
    issue_every: str = "30min",
    min_history: str = "7D",
) -> pd.DataFrame:
    """Build the gold feature table. See module docstring for the leakage rule."""
    hh = half_hourly(price, demand, spike_threshold)
    regions = sorted(hh["region"].unique())
    wx = regional_weather(weather, regions) if weather is not None and not weather.empty else None
    if wx is not None:
        wx = wx.sort_values("issued_utc")
    pdx = None
    if predispatch is not None and not predispatch.empty:
        pdx = predispatch.sort_values("run_utc")[["region", "run_utc", "interval_utc", "rrp_forecast", "demand_forecast"]]

    frames = []
    for region in regions:
        g = hh[hh["region"] == region].set_index("interval_utc").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="30min")
        g = g.reindex(full_idx)
        first_issue = g.index.min() + pd.Timedelta(min_history)
        issues = pd.Series(pd.date_range(first_issue.ceil(issue_every), g.index.max() + HH, freq=issue_every))
        if issues.empty:
            continue

        # Rolling statistics known at each issue time (window ends at the last completed half-hour).
        roll = pd.DataFrame(index=g.index)
        roll["demand_mean_24h"] = g["demand"].rolling(48, min_periods=24).mean()
        roll["demand_mean_7d"] = g["demand"].rolling(336, min_periods=96).mean()
        roll["price_mean_24h"] = g["price"].rolling(48, min_periods=24).mean()
        roll["price_std_24h"] = g["price"].rolling(48, min_periods=24).std()
        roll["price_max_24h"] = g["price_max"].rolling(48, min_periods=24).max()
        roll["spike_count_7d"] = g["spike"].rolling(336, min_periods=96).sum()
        last_complete = issues - HH  # interval start of the latest half-hour that has ended
        roll_at_issue = roll.reindex(last_complete.to_numpy()).reset_index(drop=True)

        for h in horizons:
            target = issues + (h - 1) * HH
            f = pd.DataFrame({"issued_utc": issues, "target_utc": target, "region": region, "horizon": h})
            f["demand_last"] = _lookup(g["demand"], last_complete, issues)
            f["price_last"] = _lookup(g["price"], last_complete, issues)
            f["price_max_last"] = _lookup(g["price_max"], last_complete, issues)
            for name, lag in (("day", pd.Timedelta("1D")), ("week", pd.Timedelta("7D"))):
                f[f"demand_lag_{name}"] = _lookup(g["demand"], target - lag, issues)
                f[f"price_lag_{name}"] = _lookup(g["price"], target - lag, issues)
            f = pd.concat([f, roll_at_issue], axis=1)
            f = pd.concat([f, calendar_features(f["target_utc"])], axis=1)
            f["five_min_settlement"] = is_five_minute_settlement(f["target_utc"]).astype(int).to_numpy()

            # Targets (NaN when the target half-hour is not observed yet).
            f["y_demand"] = g["demand"].reindex(target.to_numpy()).to_numpy(copy=True)
            f["y_price"] = g["price"].reindex(target.to_numpy()).to_numpy()
            f["y_spike"] = g["spike"].reindex(target.to_numpy()).to_numpy()
            frames.append(f)

    feats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if feats.empty:
        return feats
    if wx is not None:
        feats = _join_weather(feats, wx)
    if pdx is not None:
        feats = _join_predispatch(feats, pdx)
    feats["region_code"] = feats["region"].map({r: i for i, r in enumerate(sorted(regions_config()))})
    return feats.sort_values(["issued_utc", "region", "horizon"]).reset_index(drop=True)


def _join_weather(feats: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """Latest weather forecast issued at or before the issue time, for the target's hour."""
    cols = [c for c in ("temp_c", "ghi_wm2", "wind_100m_ms", "cloud_pct", "humidity_pct") if c in wx.columns]
    left = feats.assign(valid_utc=feats["target_utc"].dt.floor("h")).sort_values("issued_utc")
    right = wx.rename(columns={"issued_utc": "wx_issued_utc"}).sort_values("wx_issued_utc")
    merged = pd.merge_asof(
        left,
        right,
        left_on="issued_utc",
        right_on="wx_issued_utc",
        by=["region", "valid_utc"],
        direction="backward",
    )
    merged = merged.rename(columns={c: f"wx_{c}" for c in cols})
    if "wx_temp_c" in merged:
        merged["wx_cdh"] = (merged["wx_temp_c"] - 22).clip(lower=0)
        merged["wx_hdh"] = (15 - merged["wx_temp_c"]).clip(lower=0)
    return merged.drop(columns=["valid_utc", "wx_issued_utc"])


def _join_predispatch(feats: pd.DataFrame, pdx: pd.DataFrame) -> pd.DataFrame:
    """Latest AEMO pre-dispatch run published at or before the issue time for the target."""
    left = feats.sort_values("issued_utc")
    right = pdx.rename(columns={"interval_utc": "target_utc", "rrp_forecast": "pd_price", "demand_forecast": "pd_demand"})
    merged = pd.merge_asof(
        left,
        right.sort_values("run_utc"),
        left_on="issued_utc",
        right_on="run_utc",
        by=["region", "target_utc"],
        direction="backward",
        tolerance=pd.Timedelta("2D"),
    )
    merged["pd_age_h"] = (merged["issued_utc"] - merged["run_utc"]) / pd.Timedelta("1h")
    return merged.drop(columns=["run_utc"])


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except keys and targets."""
    exclude = {"issued_utc", "target_utc", "region", *TARGETS}
    return [c for c in df.columns if c not in exclude]
