"""Weather from Open-Meteo (CC BY 4.0, free for non-commercial use, no API key).

Three endpoints:

* ``archive``: ERA5 reanalysis (observed-like weather). Use for EDA only, never as a model
  feature, because it was not known at forecast time.
* ``historical_forecast``: archived model forecasts. GridSight treats these as issued
  ``assumed_lead_hours`` before the valid time (default 24 h, i.e. day-ahead). This is the
  training-time stand-in for the live forecast.
* ``forecast``: live forecasts, issued now.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

from gridsight.config import get_settings, sites_config
from gridsight.ingest import http

log = logging.getLogger(__name__)

RENAME = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "shortwave_radiation": "ghi_wm2",
    "cloud_cover": "cloud_pct",
    "wind_speed_100m": "wind_100m_ms",
}


def _url(kind: str) -> str:
    cfg = get_settings().openmeteo
    return {"archive": cfg["archive_url"], "historical_forecast": cfg["historical_forecast_url"], "forecast": cfg["forecast_url"]}[kind]


def parse_response(payload: dict, site_id: str) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        raise ValueError(f"Open-Meteo response has no hourly data: {list(payload)}")
    df = pd.DataFrame(hourly)
    df["valid_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df = df.rename(columns=RENAME)
    if "wind_100m_ms" in df:
        df["wind_100m_ms"] = df["wind_100m_ms"] / 3.6  # API default unit is km/h
    df["site_id"] = site_id
    return df


def fetch_site(
    site_id: str,
    start: str | None = None,
    end: str | None = None,
    kind: str = "historical_forecast",
    assumed_lead_hours: int = 24,
    sess: requests.Session | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    site = sites_config()[site_id]
    params: dict[str, object] = {
        "latitude": site["lat"],
        "longitude": site["lon"],
        "hourly": ",".join(get_settings().openmeteo["hourly_variables"]),
        "timezone": "GMT",
    }
    if kind == "forecast":
        params["forecast_days"] = 7
    else:
        if not (start and end):
            raise ValueError("start and end are required for historical data")
        params["start_date"], params["end_date"] = start, end
    df = parse_response(http.get(_url(kind), sess, params=params).json(), site_id)
    if kind == "forecast":
        issued = (now or pd.Timestamp.now(tz="UTC")).floor("h")
        df["issued_utc"] = issued
        df = df[df["valid_utc"] >= issued]
    else:
        df["issued_utc"] = df["valid_utc"] - pd.Timedelta(hours=assumed_lead_hours)
    return df.dropna(subset=["temp_c"]).reset_index(drop=True)


def fetch_all_sites(start: str | None, end: str | None, kind: str = "historical_forecast") -> pd.DataFrame:
    sess = http.session()
    frames = []
    for site_id in sites_config():
        try:
            frames.append(fetch_site(site_id, start, end, kind=kind, sess=sess))
        except Exception as exc:
            log.error("weather for %s failed: %s", site_id, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
