"""Data-quality checks: gaps, freshness and summary reports."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def find_gaps(
    df: pd.DataFrame,
    time_col: str = "interval_utc",
    freq: str = "5min",
    by: str | None = "region",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Runs of missing intervals. Columns: [by], gap_start, gap_end, missing_intervals.

    ``start`` / ``end`` (exclusive) extend the expected range beyond the observed data."""
    cols = ([by] if by else []) + ["gap_start", "gap_end", "missing_intervals"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    groups = df.groupby(by) if by else [(None, df)]
    step = pd.Timedelta(freq)
    rows = []
    for key, g in groups:
        observed = pd.DatetimeIndex(g[time_col].drop_duplicates().sort_values())
        lo = start if start is not None else observed.min()
        hi = (end - step) if end is not None else observed.max()
        expected = pd.date_range(lo, hi, freq=freq)
        missing = expected.difference(observed)
        if missing.empty:
            continue
        run_id = (missing.to_series().diff() != step).cumsum()
        for _, run in missing.to_series().groupby(run_id.to_numpy()):
            row = {"gap_start": run.iloc[0], "gap_end": run.iloc[-1] + step, "missing_intervals": len(run)}
            if by:
                row[by] = key
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def max_consecutive_missing(gaps: pd.DataFrame) -> int:
    return int(gaps["missing_intervals"].max()) if not gaps.empty else 0


@dataclass(frozen=True)
class Freshness:
    latest: pd.Timestamp | None
    lag: pd.Timedelta | None
    stale: bool


def freshness(latest: pd.Timestamp | None, now: pd.Timestamp | None = None, max_lag: str = "15min") -> Freshness:
    now = now or pd.Timestamp.now(tz="UTC")
    if latest is None:
        return Freshness(None, None, True)
    lag = now - latest
    return Freshness(latest, lag, lag > pd.Timedelta(max_lag))


def summary(df: pd.DataFrame, time_col: str, by: str = "region") -> pd.DataFrame:
    """Rows, first and last timestamp and null share per group."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(by)
    out = g[time_col].agg(rows="count", first="min", last="max")
    out["null_share"] = g.apply(lambda x: float(x.isna().to_numpy().mean()), include_groups=False)
    return out.reset_index()
