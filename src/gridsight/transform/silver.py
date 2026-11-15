"""Bronze-to-silver transforms for AEMO tables.

Each transform takes a parsed AEMO table (see :mod:`gridsight.ingest.aemo_csv`) and returns a
validated silver frame: UTC interval-start timestamps, snake_case columns, one row per key.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from gridsight import REGIONS, storage
from gridsight.ingest.aemo_csv import canonical_key, parse_file
from gridsight.timeutils import interval_end_to_start, market_to_utc
from gridsight.transform.schemas import validate

log = logging.getLogger(__name__)


def _need(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"AEMO table is missing columns {missing}; found {list(df.columns)}")


def _start(df: pd.DataFrame, col: str, minutes: int) -> pd.Series:
    return interval_end_to_start(market_to_utc(df[col]), minutes)


def dispatch_price(raw: pd.DataFrame) -> pd.DataFrame:
    """DISPATCHPRICE. When AEMO intervenes there are two rows per interval; the pricing run
    (INTERVENTION = 0) sets the price, and the interval is flagged."""
    _need(raw, "SETTLEMENTDATE", "REGIONID", "RRP")
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "interval_utc": _start(raw, "SETTLEMENTDATE", 5),
            "region": raw["REGIONID"].astype(str),
            "rrp": pd.to_numeric(raw["RRP"], errors="coerce"),
            "intervention": pd.to_numeric(raw.get("INTERVENTION", 0), errors="coerce").fillna(0).astype(int),
        }
    )
    df = df[df["region"].isin(REGIONS)]
    flag = df.groupby(["interval_utc", "region"])["intervention"].transform("max")
    df = df.assign(_pricing=df["intervention"] == 0, intervention=flag)
    df = (
        df.sort_values(["interval_utc", "region", "_pricing"])
        .drop_duplicates(["interval_utc", "region"], keep="last")
        .drop(columns="_pricing")
    )
    return validate(df.reset_index(drop=True), "dispatch_price")


def region_demand(raw: pd.DataFrame) -> pd.DataFrame:
    _need(raw, "SETTLEMENTDATE", "REGIONID", "TOTALDEMAND")
    if "INTERVENTION" in raw.columns:
        raw = raw[pd.to_numeric(raw["INTERVENTION"], errors="coerce").fillna(0) == 0]
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "interval_utc": _start(raw, "SETTLEMENTDATE", 5),
            "region": raw["REGIONID"].astype(str).to_numpy(),
            "total_demand_mw": pd.to_numeric(raw["TOTALDEMAND"], errors="coerce").to_numpy(),
            "available_gen_mw": pd.to_numeric(raw.get("AVAILABLEGENERATION", np.nan), errors="coerce"),
            "net_interchange_mw": pd.to_numeric(raw.get("NETINTERCHANGE", np.nan), errors="coerce"),
        }
    )
    df = df[df["region"].isin(REGIONS)].drop_duplicates(["interval_utc", "region"], keep="last")
    return validate(df.reset_index(drop=True), "region_demand")


def interconnector(raw: pd.DataFrame) -> pd.DataFrame:
    _need(raw, "SETTLEMENTDATE", "INTERCONNECTORID", "MWFLOW")
    if "INTERVENTION" in raw.columns:
        raw = raw[pd.to_numeric(raw["INTERVENTION"], errors="coerce").fillna(0) == 0]
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "interval_utc": _start(raw, "SETTLEMENTDATE", 5),
            "interconnector_id": raw["INTERCONNECTORID"].astype(str).to_numpy(),
            "mw_flow": pd.to_numeric(raw["MWFLOW"], errors="coerce").to_numpy(),
            "export_limit": pd.to_numeric(raw.get("EXPORTLIMIT", np.nan), errors="coerce"),
            "import_limit": pd.to_numeric(raw.get("IMPORTLIMIT", np.nan), errors="coerce"),
        }
    )
    df = df.drop_duplicates(["interval_utc", "interconnector_id"], keep="last")
    return validate(df.reset_index(drop=True), "interconnector")


def unit_scada(raw: pd.DataFrame) -> pd.DataFrame:
    _need(raw, "SETTLEMENTDATE", "DUID", "SCADAVALUE")
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "interval_utc": _start(raw, "SETTLEMENTDATE", 5),
            "duid": raw["DUID"].astype(str).to_numpy(),
            "scada_mw": pd.to_numeric(raw["SCADAVALUE"], errors="coerce").to_numpy(),
        }
    ).drop_duplicates(["interval_utc", "duid"], keep="last")
    return validate(df.reset_index(drop=True), "unit_dispatch")


def rooftop_pv(raw: pd.DataFrame) -> pd.DataFrame:
    """ROOFTOP_PV_ACTUAL is published per region and per sub-region (e.g. QLDC, QLDN); only the
    five NEM regions are kept. Where several estimate TYPEs exist, MEASUREMENT beats SATELLITE."""
    _need(raw, "INTERVAL_DATETIME", "REGIONID", "POWER")
    raw = raw.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "interval_utc": _start(raw, "INTERVAL_DATETIME", 30),
            "region": raw["REGIONID"].astype(str).to_numpy(),
            "power_mw": pd.to_numeric(raw["POWER"], errors="coerce").clip(lower=0).to_numpy(),
            "_rank": raw.get("TYPE", pd.Series("MEASUREMENT", index=raw.index))
            .astype(str)
            .map({"MEASUREMENT": 2, "SATELLITE": 1})
            .fillna(0)
            .to_numpy(),
        }
    )
    df = df[df["region"].isin(REGIONS)].sort_values("_rank")
    df = df.drop_duplicates(["interval_utc", "region"], keep="last").drop(columns="_rank")
    return validate(df.reset_index(drop=True), "rooftop_pv")


def predispatch(prices: pd.DataFrame | None, regionsum: pd.DataFrame | None) -> pd.DataFrame:
    """Join PREDISPATCHPRICE and PREDISPATCHREGIONSUM into one forecast table.

    The run time is taken from LASTCHANGED (when the run was published). That is the moment the
    forecast became public, so using it is conservative against look-ahead leakage.
    """

    def part(raw: pd.DataFrame | None, src: str, dst: str) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["run_utc", "interval_utc", "region", dst])
        _need(raw, "DATETIME", "REGIONID", src)
        if "INTERVENTION" in raw.columns:
            raw = raw[pd.to_numeric(raw["INTERVENTION"], errors="coerce").fillna(0) == 0]
        raw = raw.reset_index(drop=True)
        run_col = "LASTCHANGED" if "LASTCHANGED" in raw.columns else "DATETIME"
        out = pd.DataFrame(
            {
                "run_utc": market_to_utc(raw[run_col]).dt.floor("5min").to_numpy(),
                "interval_utc": _start(raw, "DATETIME", 30),
                "region": raw["REGIONID"].astype(str).to_numpy(),
                dst: pd.to_numeric(raw[src], errors="coerce").to_numpy(),
            }
        )
        return out.drop_duplicates(["run_utc", "interval_utc", "region"], keep="last")

    p = part(prices, "RRP", "rrp_forecast")
    d = part(regionsum, "TOTALDEMAND", "demand_forecast")
    df = p.merge(d, on=["run_utc", "interval_utc", "region"], how="outer")
    for c in ("rrp_forecast", "demand_forecast"):
        if c not in df.columns:
            df[c] = np.nan
    df = df[df["region"].isin(REGIONS)]
    return validate(df.reset_index(drop=True), "predispatch")


# Parsed-table key -> (silver table, transform)
TRANSFORMS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {
    "DISPATCH_PRICE": ("dispatch_price", dispatch_price),
    "DISPATCH_REGIONSUM": ("region_demand", region_demand),
    "DISPATCH_INTERCONNECTORRES": ("interconnector", interconnector),
    "DISPATCH_UNIT_SCADA": ("unit_dispatch", unit_scada),
    "ROOFTOP_ACTUAL": ("rooftop_pv", rooftop_pv),
}


def bronze_files_to_silver(paths: list[Path]) -> dict[str, int]:
    """Parse bronze files, transform every recognised table and merge it into silver."""
    collected: dict[str, list[pd.DataFrame]] = {}
    for path in paths:
        try:
            parsed = parse_file(path)
        except Exception as exc:  # corrupt or truncated download
            log.error("could not parse %s: %s", path, exc)
            continue
        for key, raw in parsed.items():
            collected.setdefault(canonical_key(key), []).append(raw)

    written: dict[str, int] = {}
    for key, (table, fn) in TRANSFORMS.items():
        if key in collected:
            df = fn(pd.concat(collected[key], ignore_index=True))
            written[table] = storage.write(df, table)
    pp = collected.get("PREDISPATCH_REGION_PRICES")
    ps = collected.get("PREDISPATCH_REGION_SOLUTION")
    if pp or ps:
        df = predispatch(
            pd.concat(pp, ignore_index=True) if pp else None,
            pd.concat(ps, ignore_index=True) if ps else None,
        )
        written["predispatch"] = storage.write(df, "predispatch")
    return written
