"""Partitioned Parquet storage for the bronze/silver/gold layers.

Layout: ``<data_dir>/<layer>/<table>/<partition>/<YYYY-MM>.parquet`` where ``<partition>`` is the
value of the table's partition column (region, site, ...) or ``all``.

Writes are **idempotent**: a write merges new rows into the affected monthly files, drops
duplicate keys (the newest row wins), and replaces each file atomically. Re-running any job
therefore rewrites the same files and never adds duplicates.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gridsight.config import get_settings


@dataclass(frozen=True)
class TableSpec:
    keys: tuple[str, ...]
    time_col: str
    part_col: str | None = None


TABLES: dict[str, TableSpec] = {
    # silver
    "dispatch_price": TableSpec(("interval_utc", "region"), "interval_utc", "region"),
    "region_demand": TableSpec(("interval_utc", "region"), "interval_utc", "region"),
    "rooftop_pv": TableSpec(("interval_utc", "region"), "interval_utc", "region"),
    "predispatch": TableSpec(("run_utc", "interval_utc", "region"), "interval_utc", "region"),
    "weather_forecast": TableSpec(("issued_utc", "valid_utc", "site_id"), "valid_utc", "site_id"),
    "unit_dispatch": TableSpec(("interval_utc", "duid"), "interval_utc", None),
    "interconnector": TableSpec(("interval_utc", "interconnector_id"), "interval_utc", None),
    "rooftop_capacity": TableSpec(("month_start", "region"), "month_start", "region"),
    # gold
    "features_30m": TableSpec(("issued_utc", "target_utc", "region"), "target_utc", "region"),
    # serving
    "predictions": TableSpec(
        ("model", "issued_utc", "target_utc", "region"), "target_utc", "region"
    ),
}


def table_dir(table: str, layer: str = "silver") -> Path:
    return get_settings().data_dir / layer / table


def _partition_frames(df: pd.DataFrame, spec: TableSpec) -> Iterable[tuple[str, str, pd.DataFrame]]:
    month = pd.to_datetime(df[spec.time_col]).dt.strftime("%Y-%m")
    part = df[spec.part_col].astype(str) if spec.part_col else pd.Series("all", index=df.index)
    for (p, m), g in df.groupby([part, month], sort=True):
        yield str(p), str(m), g


def _atomic_write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def write(df: pd.DataFrame, table: str, layer: str = "silver") -> int:
    """Merge ``df`` into ``layer/table``. Returns the number of rows written to disk."""
    if df.empty:
        return 0
    spec = TABLES[table]
    missing = set(spec.keys) - set(df.columns)
    if missing:
        raise ValueError(f"{table}: missing key columns {sorted(missing)}")
    base = table_dir(table, layer)
    total = 0
    for part, month, g in _partition_frames(df, spec):
        path = base / part / f"{month}.parquet"
        if path.exists():
            old = pq.read_table(path).to_pandas()
            g = pd.concat([old, g], ignore_index=True)
        g = (
            g.drop_duplicates(subset=list(spec.keys), keep="last")
            .sort_values(list(spec.keys))
            .reset_index(drop=True)
        )
        _atomic_write(pa.Table.from_pandas(g, preserve_index=False), path)
        total += len(g)
    return total


def _files(
    table: str,
    layer: str,
    parts: Sequence[str] | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> list[Path]:
    base = table_dir(table, layer)
    if not base.exists():
        return []
    lo = start.strftime("%Y-%m") if start is not None else None
    hi = end.strftime("%Y-%m") if end is not None else None
    out = []
    for part_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if parts is not None and part_dir.name not in parts:
            continue
        for f in sorted(part_dir.glob("*.parquet")):
            m = f.stem
            if (lo and m < lo) or (hi and m > hi):
                continue
            out.append(f)
    return out


def read(
    table: str,
    layer: str = "silver",
    parts: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read a table, optionally filtered by partition values and a [start, end) time range."""
    spec = TABLES[table]
    s = pd.Timestamp(start, tz="UTC") if isinstance(start, str) else start
    e = pd.Timestamp(end, tz="UTC") if isinstance(end, str) else end
    files = _files(table, layer, parts, s, e)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    if s is not None:
        df = df[df[spec.time_col] >= s]
    if e is not None:
        df = df[df[spec.time_col] < e]
    return df.sort_values(list(spec.keys)).reset_index(drop=True)


def query(sql: str, **tables: str) -> pd.DataFrame:
    """Run DuckDB SQL over silver/gold tables. Name each table as a keyword argument:

    ``query("select region, avg(rrp) from p group by 1", p="dispatch_price")``
    """
    con = duckdb.connect()
    try:
        for alias, table in tables.items():
            layer, _, name = table.rpartition(".")
            files = _files(name, layer or "silver", None, None, None)
            if not files:
                raise FileNotFoundError(f"no data for {table}")
            file_list = ", ".join(f"'{f.as_posix()}'" for f in files)
            con.execute(f"create view {alias} as select * from read_parquet([{file_list}])")
        return con.execute(sql).df()
    finally:
        con.close()


def latest_time(table: str, layer: str = "silver", parts: Sequence[str] | None = None) -> pd.Timestamp | None:
    """Most recent value of the table's time column, reading only the newest monthly files."""
    spec = TABLES[table]
    base = table_dir(table, layer)
    if not base.exists():
        return None
    newest: list[Path] = []
    for part_dir in base.iterdir():
        if part_dir.is_dir() and (parts is None or part_dir.name in parts):
            files = sorted(part_dir.glob("*.parquet"))
            if files:
                newest.append(files[-1])
    if not newest:
        return None
    vals = [pq.read_table(f, columns=[spec.time_col]).column(0).to_pandas().max() for f in newest]
    return max(vals)
