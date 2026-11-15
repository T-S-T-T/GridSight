"""Download raw files from AEMO NEMweb into the bronze layer.

Three sources:

* ``CURRENT`` report folders: recent 5-minute / 30-minute files (rolling, a few days).
* ``ARCHIVE`` report folders: daily zips of zips, kept for about 13 months.
* The MMSDM monthly archive: one folder per month from July 2009, with a zip per table.

Paths on NEMweb are case-sensitive over HTTPS. Folder names are read from the directory
listing and matched by pattern, never assumed. Downloads are idempotent: a file already in
bronze with the same name is skipped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from gridsight.config import get_settings
from gridsight.ingest import http
from gridsight.timeutils import MARKET_TZ

log = logging.getLogger(__name__)

# Report folders used by GridSight (CURRENT and ARCHIVE use the same names).
REPORTS: dict[str, str] = {
    "DispatchIS_Reports": "price, regional demand, interconnectors, constraints",
    "Dispatch_SCADA": "unit SCADA output",
    "PredispatchIS_Reports": "30-minute pre-dispatch forecasts",
    "P5_Reports": "5-minute pre-dispatch forecasts",
    "ROOFTOP_PV/ACTUAL": "rooftop PV estimates",
    "Next_Day_Dispatch": "unit dispatch targets",
}

# Timestamps embedded in NEMweb file names, e.g. PUBLIC_DISPATCHIS_202401010005_0000000403.zip
_FILE_TS = re.compile(r"_(\d{12})(?:\d{2})?(?:_|\.)")
# MMSDM table zips in both the old and new (Aug 2024+) naming schemes:
#   PUBLIC_DVD_DISPATCHPRICE_202401010000.zip
#   PUBLIC_ARCHIVE#DISPATCHPRICE#FILE01#202408010000.zip
_MMSDM_FILE = re.compile(
    r"PUBLIC_(?:DVD_|ARCHIVE#)(?P<table>[A-Z0-9_]+?)(?:#FILE(?P<part>\d+))?[_#](?P<ts>\d{12})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RemoteFile:
    url: str
    name: str
    file_time: pd.Timestamp | None  # market-time timestamp parsed from the name, as UTC


def _base() -> str:
    return get_settings().nemweb["base_url"].rstrip("/")


def bronze_dir(*parts: str) -> Path:
    return get_settings().data_dir.joinpath("bronze", "nemweb", *parts)


_DAILY_TS = re.compile(r"_(\d{8})\.zip$", re.IGNORECASE)


def parse_file_time(name: str) -> pd.Timestamp | None:
    """Market timestamp embedded in a NEMweb file name, returned in UTC (daily archives: 00:00)."""
    m = _FILE_TS.search(name)
    if m:
        ts = pd.to_datetime(m.group(1), format="%Y%m%d%H%M")
    else:
        d = _DAILY_TS.search(name)
        if not d:
            return None
        ts = pd.to_datetime(d.group(1), format="%Y%m%d")
    return pd.Timestamp(ts).tz_localize(MARKET_TZ).tz_convert("UTC")


def parse_mmsdm_name(name: str) -> dict[str, str] | None:
    m = _MMSDM_FILE.search(name)
    if not m:
        return None
    return {"table": m.group("table").upper(), "part": m.group("part") or "01", "ts": m.group("ts")}


def list_report(report: str, archive: bool = False, sess: requests.Session | None = None) -> list[RemoteFile]:
    cfg = get_settings().nemweb
    folder = cfg["archive_path"] if archive else cfg["current_path"]
    url = f"{_base()}{folder}/{report}/"
    files = []
    for link in http.list_links(url, sess):
        name = http.filename(link)
        if name.lower().endswith(".zip"):
            files.append(RemoteFile(link, name, parse_file_time(name)))
    return sorted(files, key=lambda f: (f.file_time is None, f.file_time, f.name))


def download(remote: RemoteFile, dest_dir: Path, sess: requests.Session | None = None, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / remote.name.replace("#", "_")
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    data = http.get(remote.url, sess).content
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def download_report(
    report: str,
    since_utc: pd.Timestamp | None = None,
    archive: bool = False,
    limit: int | None = None,
    sess: requests.Session | None = None,
) -> list[Path]:
    """Download report files newer than ``since_utc`` into bronze. Returns local paths."""
    sess = sess or http.session()
    files = list_report(report, archive=archive, sess=sess)
    if since_utc is not None:
        files = [f for f in files if f.file_time is None or f.file_time > since_utc]
    if limit is not None:
        files = files[-limit:]
    folder = "archive" if archive else "current"
    out = []
    for f in files:
        sub = f.file_time.strftime("%Y-%m") if f.file_time is not None else "unknown"
        out.append(download(f, bronze_dir(folder, report.replace("/", "_"), sub), sess))
    log.info("downloaded %d files from %s/%s", len(out), folder, report)
    return out


def mmsdm_month_url(year: int, month: int) -> str:
    cfg = get_settings().nemweb
    return f"{_base()}{cfg['mmsdm_path']}/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"


def list_mmsdm(year: int, month: int, sess: requests.Session | None = None) -> pd.DataFrame:
    """Every table zip available for a month: columns table, part, url, name."""
    rows = []
    for link in http.list_links(mmsdm_month_url(year, month), sess):
        name = http.filename(link)
        info = parse_mmsdm_name(name)
        if info and name.lower().endswith(".zip"):
            rows.append({**info, "url": link, "name": name})
    return pd.DataFrame(rows, columns=["table", "part", "ts", "url", "name"])


def download_mmsdm_table(
    year: int, month: int, table: str, sess: requests.Session | None = None, force: bool = False
) -> list[Path]:
    sess = sess or http.session()
    listing = list_mmsdm(year, month, sess)
    hits = listing[listing["table"] == table.upper()]
    if hits.empty:
        log.warning("MMSDM %d-%02d has no table %s", year, month, table)
        return []
    dest = bronze_dir("mmsdm", table.upper(), f"{year}-{month:02d}")
    return [
        download(RemoteFile(r.url, r.name, None), dest, sess, force=force)
        for r in hits.sort_values("part").itertuples()
    ]


def months(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) between 'YYYY-MM' strings."""
    rng = pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")
    return [(p.year, p.month) for p in rng]
