"""Phase 4: battery LP, rolling simulation, strategies, explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridsight.analysis.explain import explain_spike, global_importance, label, top_drivers
from gridsight.battery.lp import BatterySpec, optimise
from gridsight.battery.simulate import (
    curve_to_steps,
    from_predictions,
    perfect_foresight,
    risk_aware,
    rule_based,
    run_strategies,
    simulate,
    summarise,
)
from gridsight.cli import main
from gridsight.features.build import build_features, feature_columns
from gridsight.models.gbm import QuantileGBM, SpikeClassifier

SPEC = BatterySpec(horizon_hours=8)


# ------------------------------------------------------------------ LP


def test_lp_buys_low_sells_high_and_respects_limits():
    prices = np.r_[np.full(8, 20.0), np.full(8, 300.0)]
    plan = optimise(prices, SPEC)
    assert plan["charge_mw"].iloc[:8].sum() > 0 and plan["discharge_mw"].iloc[:8].sum() == pytest.approx(0)
    assert plan["discharge_mw"].iloc[8:].sum() > 0 and plan["charge_mw"].iloc[8:].sum() == pytest.approx(0)
    assert (plan["soc_mwh"] >= SPEC.soc_min_mwh - 1e-6).all() and (plan["soc_mwh"] <= SPEC.soc_max_mwh + 1e-6).all()
    assert (plan[["charge_mw", "discharge_mw"]] <= SPEC.power_mw + 1e-6).all().all()
    assert ((plan["charge_mw"] > 1e-6) & (plan["discharge_mw"] > 1e-6)).sum() == 0
    assert plan["soc_mwh"].iloc[-1] == pytest.approx(SPEC.soc_min_mwh)  # ends empty when the end price is high


def test_lp_idles_when_spread_does_not_cover_losses():
    # 85% round trip + $10/MWh degradation: a $5 spread is not worth cycling.
    plan = optimise(np.r_[np.full(4, 100.0), np.full(4, 105.0)], SPEC)
    assert plan["discharge_mw"].sum() == pytest.approx(0, abs=1e-6)


def test_terminal_value_controls_end_of_horizon_dumping():
    prices = np.r_[np.full(4, 100.0), np.full(4, 105.0)]
    dumped = optimise(prices, SPEC, terminal_value=0.0)
    assert dumped["soc_mwh"].iloc[-1] == pytest.approx(SPEC.soc_min_mwh)  # stored energy worth nothing
    kept = optimise(prices, SPEC)
    assert kept["soc_mwh"].iloc[-1] == pytest.approx(SPEC.soc_initial_mwh)


def test_lp_charges_on_negative_prices():
    plan = optimise(np.r_[np.full(4, -200.0), np.full(4, 50.0)], BatterySpec(soc_initial_mwh=20))
    assert plan["charge_mw"].iloc[:4].sum() > 0


def test_lp_rejects_nan():
    with pytest.raises(ValueError):
        optimise(np.array([1.0, np.nan]), SPEC)


def test_spec_from_config():
    s = BatterySpec.from_config({"power_mw": 50})
    assert s.power_mw == 50 and s.horizon_steps == 96 and s.dt == 0.5


# ------------------------------------------------------------------ simulation


def _toy_prices(days: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-03-01", periods=days * 288, freq="5min", tz="UTC")
    hour = idx.tz_convert("Australia/Brisbane").hour
    rrp = np.where((hour >= 17) & (hour < 21), 250.0, np.where((hour >= 10) & (hour < 15), 10.0, 70.0))
    return pd.DataFrame({"interval_utc": idx, "region": "SA1", "rrp": rrp})


def test_curve_interpolation():
    c = pd.Series([10.0, 20.0, 60.0], index=[1, 2, 6])
    np.testing.assert_allclose(curve_to_steps(c, 8), [10, 20, 30, 40, 50, 60, 60, 60])
    assert np.isnan(curve_to_steps(pd.Series(dtype=float), 3)).all()


def test_perfect_foresight_beats_naive_rules_and_soc_stays_in_bounds():
    p5 = _toy_prices()
    hh = p5.set_index("interval_utc")["rrp"].resample("30min").mean()
    pf = simulate(p5, perfect_foresight(hh), SPEC)
    rb = rule_based(p5, SPEC)
    assert pf["revenue_aud"].sum() >= rb["revenue_aud"].sum() - 1e-6
    assert pf["revenue_aud"].sum() > 0
    for r in (pf, rb):
        assert r["soc_mwh"].between(SPEC.soc_min_mwh - 1e-6, SPEC.soc_max_mwh + 1e-6).all()
    s = summarise({"perfect_foresight": pf, "rule_based": rb}, SPEC)
    assert s.set_index("strategy").loc["perfect_foresight", "pct_of_perfect"] == pytest.approx(100)
    assert (s["cycles_per_day"] > 0).all()


def test_missing_forecast_means_idle():
    p5 = _toy_prices(1)
    run = simulate(p5, lambda t, n: np.full(n, np.nan), SPEC)
    assert run["revenue_aud"].sum() == 0 and run["discharge_mw"].sum() == 0


def test_risk_aware_raises_prices_with_spike_probability():
    t = pd.Timestamp("2024-03-01", tz="UTC")
    pred = pd.DataFrame({"issued_utc": t, "horizon": [1, 2], "q0.5": [50.0, 50.0]})
    spikes = pd.DataFrame({"issued_utc": t, "horizon": [1, 2], "p_spike": [0.0, 0.5]})
    f = risk_aware(from_predictions(pred), spikes, spike_value=1000)(t, 3)
    np.testing.assert_allclose(f, [50.0, 550.0, 50.0])


def test_run_strategies_on_model_predictions(synthetic_long):
    d = synthetic_long
    feats = build_features(d["dispatch_price"], d["region_demand"], d["weather_forecast"], d["predispatch"], horizons=(1, 2, 4, 8))
    cut = feats["issued_utc"].max() - pd.Timedelta("3D")
    train, test = feats[feats["target_utc"] < cut - pd.Timedelta("1D")], feats[feats["issued_utc"] >= cut]
    cols = feature_columns(feats)
    m = QuantileGBM("price_lgbm", cols, (0.5,), "signed_log", {"n_estimators": 40}).fit(train, train["y_price"])
    pred = pd.concat(
        [
            test[["issued_utc", "region", "horizon"]].assign(model="price_lgbm", **{"q0.5": m.predict(test)["q0.5"].to_numpy()}),
            test[["issued_utc", "region", "horizon"]].assign(model="aemo_predispatch", **{"q0.5": test["pd_price"].to_numpy()}),
        ]
    )
    spike = test[["issued_utc", "region", "horizon"]].assign(model="spike_lgbm", p_spike=0.01)
    summary, runs = run_strategies(d["dispatch_price"], pred, SPEC, spike)
    s = summary.set_index("strategy")
    assert set(s.index) == {"perfect_foresight", "gridsight_p50", "rule_based", "aemo_predispatch", "gridsight_risk_aware"}
    assert s["revenue_aud"].idxmax() == "perfect_foresight"
    assert s.loc["gridsight_p50", "pct_of_perfect"] < 100


# ------------------------------------------------------------------ explainability


@pytest.fixture(scope="module")
def fitted(synthetic_long):
    d = synthetic_long
    feats = build_features(d["dispatch_price"], d["region_demand"], d["weather_forecast"], d["predispatch"], horizons=(1, 2))
    f = feats.dropna(subset=["y_spike"])
    cols = feature_columns(f)
    spike = SpikeClassifier(cols, {"n_estimators": 40}).fit(f, f["y_spike"], order=f["issued_utc"])
    price = QuantileGBM("price_lgbm", cols, (0.5,), "signed_log", {"n_estimators": 40}).fit(f, f["y_price"])
    return f, spike, price


def test_global_importance_ranks_predispatch_high(fitted):
    f, spike, price = fitted
    imp = global_importance(price, f.iloc[:2000], top=10)
    assert len(imp) == 10 and imp["mean_abs_shap"].is_monotonic_decreasing
    assert "pd_price" in set(imp["feature"].head(5)) or "price_last" in set(imp["feature"].head(5))


def test_top_drivers_and_plain_english(fitted):
    f, spike, _ = fitted
    row = f.sort_values("pd_price").tail(1)
    drivers = top_drivers(spike, row, n=3)
    assert len(drivers) == 3 and set(drivers["direction"]) <= {"raises", "lowers"}
    text = explain_spike(spike, row)
    assert text.startswith("Spike risk") and "SA1" in text
    assert label("pd_price") == "AEMO pre-dispatch price" and label("foo_bar") == "foo bar"


# ------------------------------------------------------------------ CLI


def test_cli_battery_after_evaluate(capsys):
    main(["synth", "--days", "45", "--regions", "SA1"])
    main(["features", "--issue-every", "1h"])
    main(["evaluate", "--tasks", "price,spike", "--n-estimators", "30"])
    assert main(["battery", "--region", "SA1", "--days", "2", "--horizon-hours", "6"]) == 0
    out = capsys.readouterr().out
    assert "perfect_foresight" in out and "rule_based" in out


def test_battery_requires_predictions():
    from gridsight.battery.job import run_battery_backtest

    with pytest.raises(RuntimeError):
        run_battery_backtest("SA1")
