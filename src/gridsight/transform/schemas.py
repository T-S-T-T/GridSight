"""pandera schemas for the silver layer. Every silver write is validated against these."""

from __future__ import annotations

import pandas as pd

try:  # pandera >= 0.24
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

from gridsight import REGIONS

# Generous bounds: the market price cap is reset every financial year (above $20,000/MWh in
# 2025-26), so the upper bound only catches parsing errors, not real prices.
PRICE_MIN, PRICE_MAX = -1000.0, 30000.0

def _utc_col() -> pa.Column:
    return pa.Column(pd.DatetimeTZDtype(tz="UTC"), nullable=False)


def _region_col() -> pa.Column:
    return pa.Column(str, pa.Check.isin(REGIONS), nullable=False)


SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "dispatch_price": pa.DataFrameSchema(
        {
            "interval_utc": _utc_col(),
            "region": _region_col(),
            "rrp": pa.Column(float, pa.Check.in_range(PRICE_MIN, PRICE_MAX)),
            "intervention": pa.Column(int, pa.Check.isin([0, 1])),
        },
        unique=["interval_utc", "region"],
        coerce=True,
        strict=False,
    ),
    "region_demand": pa.DataFrameSchema(
        {
            "interval_utc": _utc_col(),
            "region": _region_col(),
            "total_demand_mw": pa.Column(float, pa.Check.in_range(-2000, 20000)),
            "available_gen_mw": pa.Column(float, nullable=True),
            "net_interchange_mw": pa.Column(float, nullable=True),
        },
        unique=["interval_utc", "region"],
        coerce=True,
    ),
    "rooftop_pv": pa.DataFrameSchema(
        {
            "interval_utc": _utc_col(),
            "region": _region_col(),
            "power_mw": pa.Column(float, pa.Check.ge(0)),
        },
        unique=["interval_utc", "region"],
        coerce=True,
    ),
    "predispatch": pa.DataFrameSchema(
        {
            "run_utc": _utc_col(),
            "interval_utc": _utc_col(),
            "region": _region_col(),
            "rrp_forecast": pa.Column(float, pa.Check.in_range(PRICE_MIN, PRICE_MAX), nullable=True),
            "demand_forecast": pa.Column(float, nullable=True),
        },
        checks=[pa.Check(lambda d: d["run_utc"] < d["interval_utc"] + pd.Timedelta(minutes=30), error="run after target")],
        unique=["run_utc", "interval_utc", "region"],
        coerce=True,
    ),
    "weather_forecast": pa.DataFrameSchema(
        {
            "issued_utc": _utc_col(),
            "valid_utc": _utc_col(),
            "site_id": pa.Column(str),
            "temp_c": pa.Column(float, pa.Check.in_range(-30, 60), nullable=True),
            "ghi_wm2": pa.Column(float, pa.Check.in_range(0, 1500), nullable=True),
            "wind_100m_ms": pa.Column(float, pa.Check.in_range(0, 80), nullable=True),
            "cloud_pct": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
        },
        unique=["issued_utc", "valid_utc", "site_id"],
        coerce=True,
    ),
    "unit_dispatch": pa.DataFrameSchema(
        {
            "interval_utc": _utc_col(),
            "duid": pa.Column(str),
            "scada_mw": pa.Column(float, nullable=True),
        },
        unique=["interval_utc", "duid"],
        coerce=True,
    ),
    "interconnector": pa.DataFrameSchema(
        {
            "interval_utc": _utc_col(),
            "interconnector_id": pa.Column(str),
            "mw_flow": pa.Column(float, nullable=True),
            "export_limit": pa.Column(float, nullable=True),
            "import_limit": pa.Column(float, nullable=True),
        },
        unique=["interval_utc", "interconnector_id"],
        coerce=True,
    ),
    "rooftop_capacity": pa.DataFrameSchema(
        {
            "month_start": _utc_col(),
            "region": _region_col(),
            "capacity_mw": pa.Column(float, pa.Check.ge(0)),
        },
        unique=["month_start", "region"],
        coerce=True,
    ),
}


def validate(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Validate (and coerce dtypes) against the table's schema. Raises on failure."""
    schema = SCHEMAS.get(table)
    return schema.validate(df, lazy=True) if schema is not None else df
