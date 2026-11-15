"""Phase 1: AEMO parsing, silver transforms, storage, ingest (HTTP mocked), reference data."""

from __future__ import annotations

import io
import json
import re
import zipfile

import numpy as np
import pandas as pd
import pytest
import responses
from pandera.errors import SchemaError, SchemaErrors
from requests import HTTPError

from gridsight import storage
from gridsight.cli import main
from gridsight.config import get_settings
from gridsight.ingest import aemo_csv, nemweb, openmeteo
from gridsight.ingest.reference import (
    easter_sunday,
    national_holidays,
    parse_cer_postcode,
    postcode_region,
)
from gridsight.pipeline import backfill_mmsdm, ingest_live, load_synthetic, quality_report
from gridsight.synthetic import SyntheticConfig
from gridsight.transform import quality, silver
from gridsight.transform.schemas import validate
from tests.conftest import FIXTURES


def _zip(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, data)
    return buf.getvalue()


# ------------------------------------------------------------------ AEMO CSV parser


def test_parse_routes_rows_by_header():
    tables = aemo_csv.parse_file(FIXTURES / "dispatchis_sample.csv")
    assert {"DISPATCH_PRICE", "DISPATCH_REGIONSUM", "DISPATCH_INTERCONNECTORRES", "DISPATCH_CASESOLUTION"} <= set(tables)
    price = tables["DISPATCH_PRICE"]
    assert len(price) == 6
    assert price["RRP"].dtype.kind == "f"
    assert price["REGIONID"].tolist()[0] == "NSW1"
    assert len(tables["DISPATCH_REGIONSUM"]) == 5


def test_parse_nested_zip_matches_plain_csv():
    raw = (FIXTURES / "dispatchis_sample.csv").read_bytes()
    nested = _zip("PUBLIC_DISPATCHIS_20240115.zip", _zip("PUBLIC_DISPATCHIS_202401151000_0001.CSV", raw))
    a = aemo_csv.parse_bytes(nested, "outer.zip")["DISPATCH_PRICE"]
    b = aemo_csv.parse_bytes(raw, "x.csv")["DISPATCH_PRICE"]
    pd.testing.assert_frame_equal(a, b)


def test_parse_pads_short_rows_and_keeps_text_columns():
    text = "I,X,Y,1,A,B,C\nD,X,Y,1,1,abc\nD,X,Y,1,2,def,3\n"
    df = aemo_csv.parse_text(text)["X_Y"]
    assert df["A"].tolist() == [1, 2]
    assert df["B"].tolist() == ["abc", "def"]
    assert df["C"].isna().iloc[0]


def test_canonical_key_aliases():
    assert aemo_csv.canonical_key("DISPATCHPRICE") == "DISPATCH_PRICE"
    assert aemo_csv.canonical_key("rooftop_pv_actual") == "ROOFTOP_ACTUAL"
    assert aemo_csv.canonical_key("DISPATCH_PRICE") == "DISPATCH_PRICE"


# ------------------------------------------------------------------ silver transforms


def test_dispatch_price_transform_handles_intervention_and_time():
    raw = aemo_csv.parse_file(FIXTURES / "dispatchis_sample.csv")["DISPATCH_PRICE"]
    df = silver.dispatch_price(raw)
    assert len(df) == 5  # one row per region
    # 10:00 market time interval-ending -> 09:55 interval start -> 23:55 UTC the day before
    assert (df["interval_utc"] == pd.Timestamp("2024-01-14 23:55", tz="UTC")).all()
    sa = df.set_index("region").loc["SA1"]
    assert sa["rrp"] == 15500.0  # pricing run (INTERVENTION=0) wins
    assert sa["intervention"] == 1  # but the interval is flagged
    assert df.set_index("region").loc["QLD1", "rrp"] == -12.5


def test_region_demand_and_interconnector_transforms():
    tables = aemo_csv.parse_file(FIXTURES / "dispatchis_sample.csv")
    d = silver.region_demand(tables["DISPATCH_REGIONSUM"])
    assert d.set_index("region").loc["NSW1", "total_demand_mw"] == pytest.approx(7650.3)
    ic = silver.interconnector(tables["DISPATCH_INTERCONNECTORRES"])
    assert set(ic["interconnector_id"]) == {"V-SA", "NSW1-QLD1"}


def test_rooftop_transform_prefers_measurement_and_drops_subregions():
    raw = aemo_csv.parse_file(FIXTURES / "rooftop_pv_sample.csv")["ROOFTOP_ACTUAL"]
    df = silver.rooftop_pv(raw)
    assert set(df["region"]) == {"NSW1", "SA1"}
    assert df.set_index("region").loc["NSW1", "power_mw"] == pytest.approx(2850.5)
    assert (df["interval_utc"] == pd.Timestamp("2024-01-14 23:30", tz="UTC")).all()


def test_predispatch_transform_joins_price_and_demand():
    t = aemo_csv.parse_file(FIXTURES / "predispatch_sample.csv")
    df = silver.predispatch(t["PREDISPATCH_REGION_PRICES"], t["PREDISPATCH_REGION_SOLUTION"])
    assert len(df) == 4
    assert df[["rrp_forecast", "demand_forecast"]].notna().all().all()
    # Run published 09:31 market -> floored to 09:30 -> 23:30 UTC
    assert (df["run_utc"] == pd.Timestamp("2024-01-14 23:30", tz="UTC")).all()
    assert (df["run_utc"] <= df["interval_utc"]).all()


def test_transform_rejects_missing_columns():
    with pytest.raises(KeyError):
        silver.dispatch_price(pd.DataFrame({"SETTLEMENTDATE": ["2024/01/01 00:05:00"]}))


def test_schema_rejects_bad_price():
    bad = pd.DataFrame(
        {"interval_utc": [pd.Timestamp("2024-01-01", tz="UTC")], "region": ["SA1"], "rrp": [99999.0], "intervention": [0]}
    )
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(bad, "dispatch_price")


def test_schema_rejects_duplicate_keys_and_unknown_region():
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    dup = pd.DataFrame({"interval_utc": [ts, ts], "region": ["SA1", "SA1"], "rrp": [1.0, 2.0], "intervention": [0, 0]})
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(dup, "dispatch_price")
    wa = pd.DataFrame({"interval_utc": [ts], "region": ["WEM"], "rrp": [1.0], "intervention": [0]})
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(wa, "dispatch_price")


def test_bronze_files_to_silver_end_to_end(tmp_path):
    paths = []
    for name in ("dispatchis_sample.csv", "rooftop_pv_sample.csv", "predispatch_sample.csv"):
        p = tmp_path / (name + ".zip")
        p.write_bytes(_zip(name, (FIXTURES / name).read_bytes()))
        paths.append(p)
    bad = tmp_path / "corrupt.zip"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    written = silver.bronze_files_to_silver([*paths, bad])
    assert written["dispatch_price"] == 5
    assert written["predispatch"] == 4
    assert len(storage.read("rooftop_pv")) == 2


# ------------------------------------------------------------------ storage


def _price_rows(n: int, rrp: float = 50.0, start: str = "2024-01-31 23:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"interval_utc": idx, "region": "SA1", "rrp": rrp, "intervention": 0})


def test_storage_write_is_idempotent_and_newest_wins():
    df = _price_rows(24)  # spans two months
    storage.write(df, "dispatch_price")
    storage.write(df, "dispatch_price")
    out = storage.read("dispatch_price")
    assert len(out) == 24
    storage.write(_price_rows(3, rrp=99.0), "dispatch_price")
    out = storage.read("dispatch_price")
    assert len(out) == 24
    assert out["rrp"].head(3).tolist() == [99.0] * 3
    files = sorted((get_settings().data_dir / "silver" / "dispatch_price" / "SA1").glob("*.parquet"))
    assert [f.stem for f in files] == ["2024-01", "2024-02"]


def test_storage_read_filters_and_preserves_utc():
    storage.write(_price_rows(48), "dispatch_price")
    out = storage.read("dispatch_price", parts=["SA1"], start="2024-02-01", end="2024-02-01 01:00")
    assert len(out) == 12
    assert str(out["interval_utc"].dt.tz) == "UTC"
    assert storage.read("dispatch_price", parts=["NSW1"]).empty
    assert storage.read("region_demand").empty


def test_storage_query_with_duckdb():
    storage.write(_price_rows(12), "dispatch_price")
    res = storage.query("select region, count(*) n, avg(rrp) p from t group by 1", t="dispatch_price")
    assert res.loc[0, "n"] == 12 and res.loc[0, "p"] == 50.0


def test_storage_latest_time_and_missing_keys():
    assert storage.latest_time("dispatch_price") is None
    storage.write(_price_rows(10), "dispatch_price")
    assert storage.latest_time("dispatch_price") == pd.Timestamp("2024-01-31 23:45", tz="UTC")
    with pytest.raises(ValueError):
        storage.write(pd.DataFrame({"region": ["SA1"]}), "dispatch_price")


# ------------------------------------------------------------------ quality


def test_find_gaps_reports_runs():
    df = _price_rows(20)
    df = df.drop(index=[5, 6, 7, 12])
    gaps = quality.find_gaps(df)
    assert gaps["missing_intervals"].tolist() == [3, 1]
    assert quality.max_consecutive_missing(gaps) == 3
    assert gaps.iloc[0]["gap_start"] == df["interval_utc"].iloc[4] + pd.Timedelta("5min")


def test_find_gaps_with_expected_range_and_freshness():
    df = _price_rows(4)
    end = df["interval_utc"].max() + pd.Timedelta("30min")
    gaps = quality.find_gaps(df, end=end)
    assert gaps["missing_intervals"].sum() == 5
    f = quality.freshness(df["interval_utc"].max(), now=end)
    assert f.stale and f.lag == pd.Timedelta("30min")
    assert quality.freshness(None).stale
    assert quality.find_gaps(_price_rows(4)).empty


# ------------------------------------------------------------------ NEMweb (HTTP mocked)


def test_file_name_parsing():
    t = nemweb.parse_file_time("PUBLIC_DISPATCHIS_202401151000_0000000412345678.zip")
    assert t == pd.Timestamp("2024-01-15 00:00", tz="UTC")
    assert nemweb.parse_file_time("PUBLIC_DISPATCHIS_20240115.zip") == pd.Timestamp("2024-01-14 14:00", tz="UTC")
    assert nemweb.parse_file_time("readme.txt") is None
    old = nemweb.parse_mmsdm_name("PUBLIC_DVD_DISPATCH_UNIT_SCADA_202401010000.zip")
    new = nemweb.parse_mmsdm_name("PUBLIC_ARCHIVE#DISPATCHPRICE#FILE02#202408010000.zip")
    assert old == {"table": "DISPATCH_UNIT_SCADA", "part": "01", "ts": "202401010000"}
    assert new == {"table": "DISPATCHPRICE", "part": "02", "ts": "202408010000"}
    assert nemweb.months("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def _listing(*names: str) -> str:
    links = "".join(f'<a href="{n}">{n}</a><br>' for n in names)
    return f'<html><body><a href="../">[To Parent Directory]</a><br>{links}</body></html>'


@responses.activate
def test_ingest_live_downloads_only_new_files_and_loads_silver():
    base = get_settings().nemweb["base_url"] + get_settings().nemweb["current_path"]
    raw = (FIXTURES / "dispatchis_sample.csv").read_bytes()
    name = "PUBLIC_DISPATCHIS_202401151000_0000000412345678.zip"
    responses.get(f"{base}/DispatchIS_Reports/", body=_listing(name))
    responses.get(f"{base}/DispatchIS_Reports/{name}", body=_zip("a.CSV", raw))
    roof = "PUBLIC_ROOFTOP_PV_ACTUAL_MEASUREMENT_20240115100000_0000000412345679.zip"
    responses.get(f"{base}/ROOFTOP_PV/ACTUAL/", body=_listing(roof))
    responses.get(f"{base}/ROOFTOP_PV/ACTUAL/{roof}", body=_zip("r.CSV", (FIXTURES / "rooftop_pv_sample.csv").read_bytes()))
    pre = "PUBLIC_PREDISPATCHIS_202401150930_20240115093110.zip"
    responses.get(f"{base}/PredispatchIS_Reports/", body=_listing(pre))
    responses.get(f"{base}/PredispatchIS_Reports/{pre}", body=_zip("p.CSV", (FIXTURES / "predispatch_sample.csv").read_bytes()))

    res = ingest_live()
    assert res.files == 3
    assert res.rows["dispatch_price"] == 5
    assert len(storage.read("predispatch")) == 4

    # Second run: the dispatch file is not newer than stored data -> not downloaded again.
    n_calls = len(responses.calls)
    ingest_live()
    downloaded = [c.request.url for c in responses.calls[n_calls:] if c.request.url.endswith(".zip")]
    assert not any("DISPATCHIS" in u for u in downloaded)


@responses.activate
def test_backfill_mmsdm_handles_both_naming_schemes():
    s = get_settings().nemweb
    raw = (FIXTURES / "dispatchis_sample.csv").read_bytes()
    url_old = nemweb.mmsdm_month_url(2024, 1)
    url_new = nemweb.mmsdm_month_url(2024, 9)
    assert url_old.startswith("https://") and "MMSDM_2024_01" in url_old
    responses.get(url_old, body=_listing("PUBLIC_DVD_DISPATCHPRICE_202401010000.zip", "PUBLIC_DVD_DISPATCHLOAD_202401010000.zip"))
    responses.get(url_old + "PUBLIC_DVD_DISPATCHPRICE_202401010000.zip", body=_zip("x.CSV", raw))
    responses.get(url_new, body=_listing("PUBLIC_ARCHIVE%23DISPATCHPRICE%23FILE01%23202409010000.zip"))
    responses.get(re.compile(re.escape(url_new) + r"PUBLIC_ARCHIVE.*"), body=_zip("y.CSV", raw))
    for y, m in [(2024, 2), (2024, 3), (2024, 4), (2024, 5), (2024, 6), (2024, 7), (2024, 8)]:
        responses.get(nemweb.mmsdm_month_url(y, m), body=_listing())
    res = backfill_mmsdm("2024-01", "2024-09", tables=("DISPATCHPRICE",))
    assert res.files == 2
    assert res.rows["dispatch_price"] == 10  # 5 regions written twice (idempotent merge)
    assert len(storage.read("dispatch_price")) == 5
    assert s["base_url"].startswith("https://")


@responses.activate
def test_http_retries_then_fails_on_404():
    from gridsight.ingest import http

    url = "https://nemweb.com.au/Reports/CURRENT/Nope/"
    responses.get(url, status=404)
    with pytest.raises(HTTPError):
        http.get(url)
    assert len(responses.calls) == 1  # 4xx is not retried


# ------------------------------------------------------------------ Open-Meteo, CER, holidays


@responses.activate
def test_openmeteo_historical_forecast_is_day_ahead():
    payload = json.loads((FIXTURES / "openmeteo_sample.json").read_text(encoding="utf-8"))
    responses.get(re.compile(r"https://historical-forecast-api\.open-meteo\.com/.*"), json=payload)
    df = openmeteo.fetch_site("adelaide", "2024-01-15", "2024-01-15")
    assert len(df) == 3
    assert df["wind_100m_ms"].iloc[0] == pytest.approx(10.0)  # 36 km/h
    assert ((df["valid_utc"] - df["issued_utc"]) == pd.Timedelta(hours=24)).all()
    validate(df, "weather_forecast")


@responses.activate
def test_openmeteo_live_forecast_is_issued_now():
    payload = json.loads((FIXTURES / "openmeteo_sample.json").read_text(encoding="utf-8"))
    responses.get(re.compile(r"https://api\.open-meteo\.com/.*"), json=payload)
    now = pd.Timestamp("2024-01-15 00:30", tz="UTC")
    df = openmeteo.fetch_site("adelaide", kind="forecast", now=now)
    assert (df["issued_utc"] == pd.Timestamp("2024-01-15 00:00", tz="UTC")).all()
    assert (df["valid_utc"] >= df["issued_utc"]).all()


def test_openmeteo_rejects_bad_payload():
    with pytest.raises(ValueError):
        openmeteo.parse_response({"error": True}, "adelaide")


def test_postcode_regions():
    assert postcode_region(2000) == "NSW1"
    assert postcode_region("2600") == "NSW1"  # ACT
    assert postcode_region(3000) == "VIC1"
    assert postcode_region(4000) == "QLD1"
    assert postcode_region(5000) == "SA1"
    assert postcode_region(7000) == "TAS1"
    assert postcode_region(6000) is None  # WA, outside NEM
    assert postcode_region("0800") is None  # NT
    assert postcode_region("abc") is None


def test_parse_cer_cumulative_capacity():
    df = parse_cer_postcode(FIXTURES / "cer_postcode_sample.csv")
    nsw = df[df["region"] == "NSW1"].set_index("month_start")["capacity_mw"]
    # NSW1 = 2000 + 2600 postcodes: base 7500 kW, +90 kW Jan, +102 kW Feb
    assert nsw.iloc[0] == pytest.approx(7.59)
    assert nsw.iloc[1] == pytest.approx(7.692)
    assert "WA" not in set(df["region"]) and len(set(df["region"])) == 3
    validate(df, "rooftop_capacity")


def test_easter_and_national_holidays():
    assert str(easter_sunday(2024)) == "2024-03-31"
    assert str(easter_sunday(2025)) == "2025-04-20"
    h = national_holidays(2022).set_index("name")["date"]
    # Christmas 2022 was a Sunday: Boxing Day Mon 26, Christmas observed Tue 27
    assert str(h["Christmas Day"].date()) == "2022-12-27"
    assert str(h["Boxing Day"].date()) == "2022-12-26"
    assert str(national_holidays(2021).set_index("name").loc["Boxing Day", "date"].date()) == "2021-12-28"


# ------------------------------------------------------------------ pipeline + CLI


def test_load_synthetic_and_quality_report():
    res = load_synthetic(SyntheticConfig(days=3, regions=("SA1", "VIC1")))
    assert res.rows["dispatch_price"] == 3 * 288 * 2
    rep = quality_report().set_index("table")
    assert rep.loc["dispatch_price", "missing_intervals"] == 0
    assert np.isfinite(rep.loc["dispatch_price", "rows"])


def test_cli_synth_and_quality(capsys):
    assert main(["synth", "--days", "2", "--regions", "SA1"]) == 0
    assert main(["quality"]) == 0
    out = capsys.readouterr().out
    assert "dispatch_price" in out


@pytest.mark.network
def test_live_nemweb_listing_is_reachable():
    files = nemweb.list_report("DispatchIS_Reports")
    assert files and files[-1].file_time is not None
