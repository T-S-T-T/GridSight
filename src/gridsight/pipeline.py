"""High-level jobs. The CLI and the Prefect flows both call these functions, so every job can run
with or without an orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from gridsight import storage
from gridsight.ingest import nemweb, openmeteo
from gridsight.synthetic import SyntheticConfig, generate
from gridsight.transform import quality
from gridsight.transform.schemas import validate
from gridsight.transform.silver import bronze_files_to_silver

log = logging.getLogger(__name__)

# MMSDM tables to backfill, in priority order.
BACKFILL_TABLES = ("DISPATCHPRICE", "DISPATCHREGIONSUM", "DISPATCHINTERCONNECTORRES", "ROOFTOP_PV_ACTUAL")
LIVE_REPORTS = ("DispatchIS_Reports", "ROOFTOP_PV/ACTUAL", "PredispatchIS_Reports")


@dataclass
class JobResult:
    name: str
    rows: dict[str, int] = field(default_factory=dict)
    files: int = 0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        rows = ", ".join(f"{k}={v}" for k, v in self.rows.items()) or "no rows"
        return f"[{self.name}] files={self.files} {rows}" + ("" if not self.notes else " | " + "; ".join(self.notes))


def load_synthetic(cfg: SyntheticConfig | None = None) -> JobResult:
    """Write synthetic silver tables (offline demo and tests)."""
    res = JobResult("synthetic")
    for table, df in generate(cfg).items():
        res.rows[table] = storage.write(validate(df, table), table)
    return res


def backfill_mmsdm(start: str, end: str, tables: tuple[str, ...] = BACKFILL_TABLES) -> JobResult:
    """Download MMSDM monthly tables for [start, end] (YYYY-MM) and load them into silver."""
    res = JobResult(f"backfill {start}..{end}")
    for year, month in nemweb.months(start, end):
        for table in tables:
            paths = nemweb.download_mmsdm_table(year, month, table)
            res.files += len(paths)
            for k, v in bronze_files_to_silver(paths).items():
                res.rows[k] = res.rows.get(k, 0) + v
    return res


def ingest_live(reports: tuple[str, ...] = LIVE_REPORTS, limit: int | None = 36) -> JobResult:
    """Pull new files from the NEMweb CURRENT folders and merge them into silver.

    Only files newer than the latest interval already stored are fetched. After loading, gaps
    in the last 24 h of dispatch prices are reported so a backfill can repair them."""
    res = JobResult("ingest_live")
    since = storage.latest_time("dispatch_price")
    paths = []
    for report in reports:
        paths += nemweb.download_report(report, since_utc=since if report == "DispatchIS_Reports" else None, limit=limit)
    res.files = len(paths)
    res.rows = bronze_files_to_silver(paths)
    latest = storage.latest_time("dispatch_price")
    if latest is not None:
        recent = storage.read("dispatch_price", start=latest - pd.Timedelta(hours=24))
        gaps = quality.find_gaps(recent)
        if not gaps.empty:
            res.notes.append(f"{int(gaps['missing_intervals'].sum())} missing intervals in last 24h")
    return res


def ingest_weather(start: str | None, end: str | None, kind: str = "historical_forecast") -> JobResult:
    res = JobResult(f"weather {kind}")
    df = openmeteo.fetch_all_sites(start, end, kind=kind)
    if not df.empty:
        res.rows["weather_forecast"] = storage.write(validate(df, "weather_forecast"), "weather_forecast")
    return res


def quality_report() -> pd.DataFrame:
    """Coverage, gaps and freshness for the main silver tables."""
    rows = []
    for table, freq in (("dispatch_price", "5min"), ("region_demand", "5min"), ("rooftop_pv", "30min")):
        df = storage.read(table)
        if df.empty:
            rows.append({"table": table, "rows": 0})
            continue
        gaps = quality.find_gaps(df, freq=freq)
        fr = quality.freshness(df["interval_utc"].max())
        rows.append(
            {
                "table": table,
                "rows": len(df),
                "first": df["interval_utc"].min(),
                "last": df["interval_utc"].max(),
                "gap_runs": len(gaps),
                "missing_intervals": int(gaps["missing_intervals"].sum()) if not gaps.empty else 0,
                "max_consecutive_missing": quality.max_consecutive_missing(gaps),
                "lag": fr.lag,
            }
        )
    return pd.DataFrame(rows)
