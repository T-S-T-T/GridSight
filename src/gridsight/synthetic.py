"""Synthetic NEM-like data in the silver-layer schema.

Used by the test suite and by ``gridsight demo`` so the whole pipeline can run offline. The
generator is deterministic for a given seed and reproduces the main features of real NEM data:
two daily demand peaks, temperature-driven demand, rooftop solar eating midday demand, negative
midday prices, rare price spikes, and pre-dispatch forecasts whose error grows with horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridsight import REGIONS
from gridsight.config import regions_config
from gridsight.timeutils import MARKET_TZ

# Rough scale of each region: (base demand MW, generation capacity MW, mean temp C, rooftop PV MW)
_REGION_SCALE: dict[str, tuple[float, float, float, float]] = {
    "NSW1": (7500, 13000, 18.0, 5500),
    "QLD1": (6200, 11000, 22.0, 5000),
    "VIC1": (5200, 9500, 15.5, 4200),
    "SA1": (1400, 3200, 17.0, 1900),
    "TAS1": (1100, 2600, 12.5, 250),
}


@dataclass(frozen=True)
class SyntheticConfig:
    start: str = "2024-01-01"
    days: int = 60
    regions: tuple[str, ...] = REGIONS
    predispatch_horizon_steps: int = 16  # 30-minute steps (8 h)
    seed: int = 42


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def generate(cfg: SyntheticConfig | None = None) -> dict[str, pd.DataFrame]:
    """Return silver-format tables: dispatch_price, region_demand, rooftop_pv, predispatch,
    weather_forecast."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    start_utc = pd.Timestamp(cfg.start, tz=MARKET_TZ).tz_convert("UTC")
    idx = pd.date_range(start_utc, periods=cfg.days * 288, freq="5min")
    local = idx.tz_convert(MARKET_TZ)
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    doy = local.dayofyear.to_numpy()
    weekend = (local.dayofweek.to_numpy() >= 5).astype(float)

    sites_by_region = {r: list(v["sites"].keys()) for r, v in regions_config().items()}

    prices, demands, pvs, weathers = [], [], [], []
    for r_i, region in enumerate(cfg.regions):
        base, cap, tmean, pv_cap = _REGION_SCALE[region]
        n = len(idx)
        seasonal = np.cos(2 * np.pi * (doy - 15) / 365.25)  # +1 mid-January (summer)
        temp = (
            tmean
            + 6.0 * seasonal
            + 5.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
            + _ar1(rng, n, 0.999, 0.08)
        )
        daylight = np.clip(np.sin(np.pi * (hour - 6.0) / 13.0), 0.0, None)
        cloud = np.clip(0.35 + _ar1(rng, n, 0.998, 0.03), 0.0, 1.0)
        ghi = 1000.0 * daylight * (1.0 - 0.75 * cloud) * (0.85 + 0.15 * seasonal)
        wind = np.clip(8.0 + _ar1(rng, n, 0.998, 0.25), 0.0, 30.0)

        shape = (
            0.10 * np.exp(-((hour - 8.0) ** 2) / 4.0)
            + 0.22 * np.exp(-((hour - 18.5) ** 2) / 6.0)
            - 0.12 * np.exp(-((hour - 3.5) ** 2) / 8.0)
        )
        cooling = np.clip(temp - 22.0, 0.0, None)
        heating = np.clip(15.0 - temp, 0.0, None)
        rooftop = pv_cap * ghi / 1000.0 * 0.75
        demand = (
            base * (1.0 + shape - 0.07 * weekend)
            + base * (0.020 * cooling**1.3 + 0.015 * heating)
            - rooftop
            + _ar1(rng, n, 0.95, base * 0.006)
        )
        demand = np.maximum(demand, 0.05 * base)
        available = cap * (0.92 + 0.04 * np.sin(2 * np.pi * doy / 30.0 + r_i))
        utilisation = demand / available

        wind_share = np.clip(wind / 25.0, 0.0, 1.0)
        rrp = (
            35.0
            + 140.0 * np.exp(3.2 * (utilisation - 0.75))
            - 90.0 * (ghi / 1000.0) * (pv_cap / base)
            - 25.0 * wind_share
            + _ar1(rng, n, 0.9, 8.0)
        )
        spike_p = np.where(utilisation > 0.85, 0.02, 0.0005)
        spikes = rng.random(n) < spike_p
        rrp = np.where(spikes, rng.uniform(300.0, 15000.0, n), rrp)
        rrp = np.clip(rrp, -1000.0, 17500.0)

        prices.append(
            pd.DataFrame(
                {"interval_utc": idx, "region": region, "rrp": rrp, "intervention": 0}
            )
        )
        demands.append(
            pd.DataFrame(
                {
                    "interval_utc": idx,
                    "region": region,
                    "total_demand_mw": demand,
                    "available_gen_mw": available,
                    "net_interchange_mw": _ar1(rng, n, 0.99, 15.0),
                }
            )
        )
        pv30 = pd.Series(rooftop, index=idx).resample("30min").mean()
        pvs.append(
            pd.DataFrame(
                {"interval_utc": pv30.index, "region": region, "power_mw": pv30.to_numpy()}
            )
        )

        hourly = pd.DataFrame(
            {"temp_c": temp, "ghi_wm2": ghi, "wind_100m_ms": wind, "cloud_pct": cloud * 100},
            index=idx,
        ).resample("1h").mean()
        for site in sites_by_region.get(region, [region.lower()]):
            noise = rng.normal(0.0, 1.0, (len(hourly), 4))
            w = pd.DataFrame(
                {
                    "valid_utc": hourly.index,
                    "issued_utc": hourly.index - pd.Timedelta(hours=24),
                    "site_id": site,
                    "temp_c": hourly["temp_c"].to_numpy() + 0.8 * noise[:, 0],
                    "humidity_pct": np.clip(60 - 1.5 * (hourly["temp_c"] - tmean) + 5 * noise[:, 1], 5, 100),
                    "ghi_wm2": np.clip(hourly["ghi_wm2"].to_numpy() + 40 * noise[:, 2], 0, None),
                    "cloud_pct": np.clip(hourly["cloud_pct"].to_numpy() + 5 * noise[:, 3], 0, 100),
                    "wind_100m_ms": np.clip(hourly["wind_100m_ms"].to_numpy() + noise[:, 0], 0, None),
                }
            )
            weathers.append(w)

    price = pd.concat(prices, ignore_index=True)
    demand = pd.concat(demands, ignore_index=True)
    rooftop_pv = pd.concat(pvs, ignore_index=True)
    weather = pd.concat(weathers, ignore_index=True)
    predispatch = _predispatch(price, demand, cfg, rng)
    return {
        "dispatch_price": price,
        "region_demand": demand,
        "rooftop_pv": rooftop_pv,
        "predispatch": predispatch,
        "weather_forecast": weather,
    }


def _predispatch(
    price: pd.DataFrame, demand: pd.DataFrame, cfg: SyntheticConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """AEMO-like pre-dispatch: a run every 30 min forecasting the next H half-hours."""
    merged = price.merge(demand, on=["interval_utc", "region"])
    merged["half_hour"] = merged["interval_utc"].dt.floor("30min")
    hh = (
        merged.groupby(["region", "half_hour"], as_index=False)
        .agg(rrp=("rrp", "mean"), demand=("total_demand_mw", "mean"))
        .rename(columns={"half_hour": "interval_utc"})
    )
    out = []
    for region, g in hh.groupby("region"):
        g = g.sort_values("interval_utc").reset_index(drop=True)
        n = len(g)
        for h in range(1, cfg.predispatch_horizon_steps + 1):
            if h >= n:
                break
            target = g.iloc[h:].reset_index(drop=True)
            run = g["interval_utc"].iloc[: n - h].reset_index(drop=True)
            scale = np.sqrt(h)
            actual = target["rrp"].to_numpy()
            is_spike = actual > 300
            # Pre-dispatch sees a real spike less often the further ahead it looks, and
            # occasionally forecasts a spike that never happens.
            seen = rng.random(len(target)) < 0.75 / scale
            false_alarm = rng.random(len(target)) < 0.003
            normal = 0.9 * np.clip(actual, None, 250) + 10 + rng.normal(0.0, 12.0 * scale, len(target))
            rrp_f = np.where(is_spike & seen, actual * rng.uniform(0.5, 1.2, len(target)), normal)
            rrp_f = np.where(false_alarm, rng.uniform(300, 2000, len(target)), rrp_f)
            d_err = rng.normal(0.0, target["demand"].mean() * 0.02 * scale, len(target))
            out.append(
                pd.DataFrame(
                    {
                        "run_utc": run,
                        "interval_utc": target["interval_utc"],
                        "region": region,
                        "rrp_forecast": rrp_f,
                        "demand_forecast": target["demand"] + d_err,
                    }
                )
            )
    return pd.concat(out, ignore_index=True)
