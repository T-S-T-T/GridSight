"""Time handling for NEM data.

AEMO publishes all timestamps in *market time*: AEST, a fixed UTC+10 offset with no daylight
saving. Using ``Australia/Sydney`` instead silently shifts half of every year by an hour.

AEMO interval timestamps (``SETTLEMENTDATE``, ``DATETIME``, ``INTERVAL_DATETIME``) mark the
**end** of the interval. GridSight stores the interval **start** in UTC everywhere.
"""

from __future__ import annotations

from datetime import timedelta, timezone

import pandas as pd

MARKET_TZ = timezone(timedelta(hours=10), name="AEST")
FIVE_MIN_SETTLEMENT_START_UTC = pd.Timestamp("2021-10-01 00:00", tz=MARKET_TZ).tz_convert("UTC")


def market_to_utc(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Parse naive market-time timestamps (strings or datetimes) and return UTC-aware values."""
    ts = pd.to_datetime(pd.Series(values), format="mixed")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(MARKET_TZ)
    return ts.dt.tz_convert("UTC")


def interval_end_to_start(ts: pd.Series, minutes: int) -> pd.Series:
    """Convert AEMO interval-ending timestamps to interval-starting timestamps."""
    return ts - pd.Timedelta(minutes=minutes)


def utc_to_market(ts: pd.Series) -> pd.Series:
    return pd.Series(ts).dt.tz_convert(MARKET_TZ)


def is_five_minute_settlement(interval_utc: pd.Series) -> pd.Series:
    """True for intervals settled under five-minute settlement (from 1 Oct 2021, market time)."""
    return pd.Series(interval_utc) >= FIVE_MIN_SETTLEMENT_START_UTC


def ensure_utc(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(pd.Series(ts))
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")
