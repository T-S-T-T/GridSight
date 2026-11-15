"""Battery backtest job: simulate every strategy on the locked test period."""

from __future__ import annotations

import logging

import pandas as pd

from gridsight import storage
from gridsight.battery.lp import BatterySpec
from gridsight.battery.simulate import run_strategies
from gridsight.config import battery_config, get_settings

log = logging.getLogger(__name__)


def run_battery_backtest(region: str | None = None, days: int | None = None, overrides: dict | None = None) -> pd.DataFrame:
    """Uses the test-period predictions written by ``gridsight evaluate``."""
    s = get_settings()
    region = region or battery_config()["region"]
    res_dir = s.reports_dir / "results"
    price_path = res_dir / "price_test_predictions.parquet"
    if not price_path.exists():
        raise RuntimeError("no price predictions: run `gridsight evaluate` first")
    price_pred = pd.read_parquet(price_path)
    price_pred = price_pred[price_pred["region"] == region]
    if price_pred.empty:
        raise RuntimeError(f"no price predictions for {region}")
    spike_path = res_dir / "spike_test_predictions.parquet"
    spike_pred = pd.read_parquet(spike_path) if spike_path.exists() else None
    if spike_pred is not None:
        spike_pred = spike_pred[spike_pred["region"] == region]
    if days is not None:
        start = price_pred["issued_utc"].max() - pd.Timedelta(days=days)
        price_pred = price_pred[price_pred["issued_utc"] >= start]
        if spike_pred is not None:
            spike_pred = spike_pred[spike_pred["issued_utc"] >= start]

    price = storage.read("dispatch_price", parts=[region])
    spec = BatterySpec.from_config(overrides)
    summary, runs = run_strategies(price, price_pred, spec, spike_pred)
    out = s.reports_dir / "battery"
    out.mkdir(parents=True, exist_ok=True)
    summary.insert(0, "region", region)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat([r.assign(strategy=k) for k, r in runs.items()]).to_parquet(out / "runs.parquet", index=False)
    log.info("battery backtest written to %s", out)
    return summary
