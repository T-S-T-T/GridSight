"""Reference data: Clean Energy Regulator rooftop PV installs, postcodes and holidays."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from gridsight.timeutils import MARKET_TZ

# ---------------------------------------------------------------- postcodes -> NEM region


def postcode_region(postcode: int | str) -> str | None:
    """NEM region for an Australian postcode (WA and NT are outside the NEM -> None)."""
    try:
        pc = int(str(postcode).strip())
    except ValueError:
        return None
    if 2600 <= pc <= 2618 or 2900 <= pc <= 2920:  # ACT is part of NSW1
        return "NSW1"
    if 1000 <= pc <= 2999:
        return "NSW1"
    if 3000 <= pc <= 3999 or 8000 <= pc <= 8999:
        return "VIC1"
    if 4000 <= pc <= 4999 or 9000 <= pc <= 9999:
        return "QLD1"
    if 5000 <= pc <= 5999:
        return "SA1"
    if 7000 <= pc <= 7999:
        return "TAS1"
    return None


# ---------------------------------------------------------------- CER small-scale installs

_MONTH_COL = re.compile(r"^(?P<mon>[A-Za-z]{3})\s+(?P<year>\d{4})\s*-\s*SGU Rated Output In kW", re.IGNORECASE)
_PREV_COL = re.compile(r"Previous Years.*SGU Rated Output In kW", re.IGNORECASE)


def parse_cer_postcode(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Parse a CER 'SGU - Solar' postcode file into monthly installed capacity per region.

    Returns columns month_start (UTC), region, capacity_mw where capacity is *cumulative*
    installed rooftop PV at the end of that month. Capacity installed before the first monthly
    column ("Previous Years") is used as the starting stock.
    """
    raw = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df, thousands=",")
    pc_col = next(c for c in raw.columns if "postcode" in c.lower())
    region = raw[pc_col].map(postcode_region)
    df = raw.assign(region=region).dropna(subset=["region"])

    month_cols = {c: _MONTH_COL.match(c.strip()) for c in df.columns}
    month_cols = {c: m for c, m in month_cols.items() if m}
    prev_cols = [c for c in df.columns if _PREV_COL.search(c)]
    if not month_cols:
        raise ValueError("no monthly 'SGU Rated Output In kW' columns found")

    num = df[list(month_cols) + prev_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    by_region = num.groupby(df["region"]).sum()
    base = by_region[prev_cols].sum(axis=1) if prev_cols else 0.0

    records = []
    for col, m in month_cols.items():
        month = pd.Timestamp(f"{m.group('mon')} {m.group('year')}")
        for reg, kw in by_region[col].items():
            records.append({"month_start": month, "region": reg, "added_kw": kw})
    out = pd.DataFrame(records).sort_values(["region", "month_start"])
    out["capacity_mw"] = (
        out.groupby("region")["added_kw"].cumsum() + out["region"].map(base if prev_cols else {}).fillna(0)
    ) / 1000.0
    out["month_start"] = out["month_start"].dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
    return out[["month_start", "region", "capacity_mw"]].reset_index(drop=True)


# ---------------------------------------------------------------- public holidays


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """Move a Saturday/Sunday holiday to the following Monday."""
    if d.weekday() == 5:
        return d + timedelta(days=2)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def national_holidays(year: int) -> pd.DataFrame:
    """National public holidays observed in every NEM state. State-specific holidays
    (e.g. Labour Day, King's Birthday) come from :func:`load_holidays_csv`."""
    easter = easter_sunday(year)
    xmas, boxing = date(year, 12, 25), date(year, 12, 26)
    wd = xmas.weekday()
    if wd == 5:  # Saturday Christmas: Mon 27 and Tue 28
        xmas_obs, boxing_obs = date(year, 12, 27), date(year, 12, 28)
    elif wd == 6:  # Sunday Christmas: Boxing Day Mon 26, Christmas observed Tue 27
        xmas_obs, boxing_obs = date(year, 12, 27), boxing
    elif wd == 4:  # Friday Christmas: Boxing Day observed Mon 28
        xmas_obs, boxing_obs = xmas, date(year, 12, 28)
    else:
        xmas_obs, boxing_obs = xmas, boxing
    days = {
        "New Year's Day": _observed(date(year, 1, 1)),
        "Australia Day": _observed(date(year, 1, 26)),
        "Good Friday": easter - timedelta(days=2),
        "Easter Saturday": easter - timedelta(days=1),
        "Easter Monday": easter + timedelta(days=1),
        "Anzac Day": date(year, 4, 25),
        "Christmas Day": xmas_obs,
        "Boxing Day": boxing_obs,
    }
    return pd.DataFrame({"date": pd.to_datetime(list(days.values())), "name": list(days.keys())})


def load_holidays_csv(path: str | Path) -> pd.DataFrame:
    """data.gov.au 'Australian Public Holidays' file -> date, name, state."""
    raw = pd.read_csv(path)
    cols = {c.lower().replace(" ", "_"): c for c in raw.columns}
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[cols["date"]].astype(str), format="%Y%m%d", errors="coerce"),
            "name": raw[cols.get("holiday_name", cols.get("name", list(raw.columns)[1]))],
            "state": raw[cols["jurisdiction"]].str.upper(),
        }
    )
    return out.dropna(subset=["date"])
