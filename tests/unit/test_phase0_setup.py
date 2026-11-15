"""Phase 0: configuration, time handling, synthetic data and CLI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridsight import REGIONS
from gridsight.cli import main
from gridsight.config import (
    battery_config,
    get_settings,
    models_config,
    regions_config,
    sites_config,
)
from gridsight.timeutils import (
    MARKET_TZ,
    interval_end_to_start,
    is_five_minute_settlement,
    market_to_utc,
)


def test_settings_use_temp_data_dir(tmp_data_dir):
    assert get_settings().data_dir == tmp_data_dir


def test_nemweb_is_https():
    assert get_settings().nemweb["base_url"].startswith("https://")


def test_every_region_has_weighted_sites_with_coordinates():
    regions = regions_config()
    sites = sites_config()
    assert set(regions) == set(REGIONS)
    for region, cfg in regions.items():
        weights = cfg["sites"]
        assert abs(sum(weights.values()) - 1.0) < 1e-9, region
        for site in weights:
            assert site in sites, f"{site} missing from sites.yaml"
            assert -45 < sites[site]["lat"] < -10
            assert 110 < sites[site]["lon"] < 155


def test_battery_and_model_configs_are_consistent():
    b = battery_config()
    assert b["soc_min_mwh"] <= b["soc_initial_mwh"] <= b["soc_max_mwh"] <= b["energy_mwh"]
    assert 0 < b["eta_charge"] <= 1 and 0 < b["eta_discharge"] <= 1
    m = models_config()
    assert sorted(m["quantiles"]) == m["quantiles"]
    assert 0.5 in m["quantiles"]


def test_market_time_is_fixed_utc_plus_10_without_dst():
    # 1 January (Sydney is on daylight saving, UTC+11) must still convert with +10.
    jan = market_to_utc(pd.Series(["2024/01/01 12:00:00"]))
    jul = market_to_utc(pd.Series(["2024/07/01 12:00:00"]))
    assert jan.iloc[0] == pd.Timestamp("2024-01-01 02:00", tz="UTC")
    assert jul.iloc[0] == pd.Timestamp("2024-07-01 02:00", tz="UTC")


def test_interval_end_to_start():
    end = market_to_utc(pd.Series(["2024/01/01 00:05:00"]))
    start = interval_end_to_start(end, 5)
    assert start.iloc[0] == pd.Timestamp("2024-01-01 00:00", tz=MARKET_TZ)


def test_five_minute_settlement_boundary():
    ts = market_to_utc(pd.Series(["2021/09/30 23:55:00", "2021/10/01 00:00:00"]))
    assert is_five_minute_settlement(ts).tolist() == [False, True]


def test_synthetic_data_shapes_and_realism(synthetic):
    price = synthetic["dispatch_price"]
    demand = synthetic["region_demand"]
    assert set(price["region"]) == {"NSW1", "SA1"}
    assert len(price) == 40 * 288 * 2
    assert price["interval_utc"].dt.tz is not None
    assert price["rrp"].between(-1000, 17500).all()
    assert (price["rrp"] > 300).any(), "spikes should occur"
    assert (demand["total_demand_mw"] > 0).all()
    # Evening peak is higher than the early-morning trough.
    local = demand.assign(h=demand["interval_utc"].dt.tz_convert(MARKET_TZ).dt.hour)
    nsw = local[local["region"] == "NSW1"].groupby("h")["total_demand_mw"].mean()
    assert nsw.loc[18] > nsw.loc[3]
    pd_ = synthetic["predispatch"]
    assert (pd_["run_utc"] < pd_["interval_utc"]).all()
    w = synthetic["weather_forecast"]
    assert (w["issued_utc"] < w["valid_utc"]).all()


def test_synthetic_is_deterministic():
    from gridsight.synthetic import SyntheticConfig, generate

    a = generate(SyntheticConfig(days=2, regions=("SA1",), seed=3))
    b = generate(SyntheticConfig(days=2, regions=("SA1",), seed=3))
    np.testing.assert_allclose(a["dispatch_price"]["rrp"], b["dispatch_price"]["rrp"])


def test_cli_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "GridSight" in out and "NSW1" in out


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])
