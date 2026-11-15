"""Phase 3: features (with leakage tests), models, conformal intervals, metrics, backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridsight.cli import main
from gridsight.evaluate import metrics as M
from gridsight.evaluate.backtest import (
    compare_with_aemo,
    run_backtest,
    score,
    split_test_period,
    walk_forward_folds,
)
from gridsight.evaluate.run import (
    build_and_store_features,
    evaluate_task,
    load_features,
    train_production,
    write_results,
)
from gridsight.features.build import (
    CALENDAR_COLS,
    build_features,
    calendar_features,
    feature_columns,
    half_hourly,
    regional_weather,
)
from gridsight.models.base import (
    ModelCard,
    inv_signed_log,
    list_versions,
    load_model,
    save_model,
    set_stage,
    signed_log,
    sort_quantiles,
)
from gridsight.models.baselines import Climatology, PredispatchBaseline, SeasonalNaive
from gridsight.models.conformal import CQR
from gridsight.models.gbm import QuantileGBM, SpikeClassifier, choose_threshold
from gridsight.models.tasks import TASKS
from gridsight.pipeline import load_synthetic
from gridsight.synthetic import SyntheticConfig

FAST = {"n_estimators": 40, "learning_rate": 0.1}
SMALL_BT = {"n_folds": 2, "fold_days": 14, "gap_days": 1, "min_train_days": 60}


@pytest.fixture(scope="module")
def feats_long(synthetic_long):
    d = synthetic_long
    return build_features(d["dispatch_price"], d["region_demand"], d["weather_forecast"], d["predispatch"], horizons=(1, 2, 4, 8))


# ------------------------------------------------------------------ features


def test_half_hourly_aggregation(synthetic):
    hh = half_hourly(synthetic["dispatch_price"], synthetic["region_demand"])
    p = synthetic["dispatch_price"]
    first = p[(p["region"] == "SA1")].head(6)
    row = hh[(hh["region"] == "SA1")].iloc[0]
    assert row["price"] == pytest.approx(first["rrp"].mean())
    assert row["price_max"] == pytest.approx(first["rrp"].max())
    assert set(hh["spike"].dropna().unique()) <= {0.0, 1.0}


def test_feature_table_shape_and_targets(feats_long, synthetic_long):
    f = feats_long
    assert set(f["horizon"]) == {1, 2, 4, 8}
    assert (f["target_utc"] == f["issued_utc"] + (f["horizon"] - 1) * pd.Timedelta("30min")).all()
    assert f[CALENDAR_COLS].notna().all().all()
    # y_demand equals the half-hour mean of the target interval
    hh = half_hourly(synthetic_long["dispatch_price"], synthetic_long["region_demand"]).set_index("interval_utc")
    sample = f.dropna(subset=["y_demand"]).sample(50, random_state=0)
    np.testing.assert_allclose(sample["y_demand"], hh.loc[sample["target_utc"], "demand"].to_numpy())
    assert {"pd_price", "pd_demand", "wx_temp_c", "wx_cdh"} <= set(f.columns)
    cols = feature_columns(f)
    assert not set(cols) & {"y_demand", "y_price", "y_spike", "issued_utc", "target_utc", "region"}


def test_lags_are_horizon_aware(feats_long):
    f = feats_long
    # The day lag is only known when target - 1 day is a completed half-hour at issue time.
    assert f.loc[f["horizon"] == 1, "demand_lag_day"].notna().all()
    assert f["demand_last"].notna().mean() > 0.99


def test_features_do_not_use_the_future(synthetic):
    """Scramble every value after a cut-off: no feature of a row issued at or before the cut-off
    may change. This guards the whole feature pipeline against look-ahead leakage."""
    d = {k: v.copy() for k, v in synthetic.items()}
    cut = pd.Timestamp("2024-01-25 06:00", tz="UTC")
    kw = {"horizons": (1, 4, 48)}
    base = build_features(d["dispatch_price"], d["region_demand"], d["weather_forecast"], d["predispatch"], **kw)

    rng = np.random.default_rng(0)
    p, dem, w, pre = (d["dispatch_price"].copy(), d["region_demand"].copy(), d["weather_forecast"].copy(), d["predispatch"].copy())
    fut = p["interval_utc"] >= cut
    p.loc[fut, "rrp"] = rng.uniform(-1000, 15000, fut.sum())
    fut = dem["interval_utc"] >= cut
    dem.loc[fut, "total_demand_mw"] = rng.uniform(100, 20000, fut.sum())
    fut = w["issued_utc"] > cut
    w.loc[fut, "temp_c"] = rng.uniform(-20, 50, fut.sum())
    fut = pre["run_utc"] > cut
    pre.loc[fut, "rrp_forecast"] = rng.uniform(-1000, 15000, fut.sum())
    scrambled = build_features(p, dem, w, pre, **kw)

    key = ["issued_utc", "target_utc", "region", "horizon"]
    cols = [c for c in feature_columns(base) if c not in key]
    a = base[base["issued_utc"] <= cut].set_index(key)[cols].sort_index()
    b = scrambled[scrambled["issued_utc"] <= cut].set_index(key)[cols].sort_index()
    assert len(a) > 1000
    pd.testing.assert_frame_equal(a, b)
    # Sanity check: rows issued after the cut-off DO change, so the test can fail.
    later = base[base["issued_utc"] > cut].set_index(key)["price_last"]
    later_s = scrambled[scrambled["issued_utc"] > cut].set_index(key)["price_last"]
    assert not np.allclose(later.to_numpy(), later_s.reindex(later.index).to_numpy())


def test_weather_join_uses_latest_forecast_issued_before_issue_time():
    t = pd.Timestamp("2024-01-10 06:00", tz="UTC")
    idx = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), t + pd.Timedelta("1D"), freq="5min")
    price = pd.DataFrame({"interval_utc": idx, "region": "SA1", "rrp": 50.0})
    demand = pd.DataFrame({"interval_utc": idx, "region": "SA1", "total_demand_mw": 1500.0})
    valid = t + pd.Timedelta("2h")
    weather = pd.DataFrame(
        {
            "site_id": ["adelaide"] * 3,
            "issued_utc": [t - pd.Timedelta("12h"), t - pd.Timedelta("1h"), t + pd.Timedelta("1h")],
            "valid_utc": [valid] * 3,
            "temp_c": [20.0, 30.0, 99.0],
        }
    )
    f = build_features(price, demand, weather=weather, horizons=(5,), min_history="1D")
    row = f[(f["issued_utc"] == t)].iloc[0]
    assert row["target_utc"] == valid
    assert row["wx_temp_c"] == 30.0  # not the one issued after t
    assert row["wx_cdh"] == 8.0


def test_regional_weather_weights_sites():
    w = pd.DataFrame(
        {
            "site_id": ["adelaide", "mount_gambier"],
            "issued_utc": [pd.Timestamp("2024-01-01", tz="UTC")] * 2,
            "valid_utc": [pd.Timestamp("2024-01-02", tz="UTC")] * 2,
            "temp_c": [30.0, 20.0],
        }
    )
    r = regional_weather(w, ["SA1"])
    assert r["temp_c"].iloc[0] == pytest.approx(0.9 * 30 + 0.1 * 20)


def test_calendar_flags_holidays_and_weekends():
    ts = pd.Series(pd.to_datetime(["2024-12-25 02:00", "2024-12-28 02:00", "2024-12-30 02:00"], utc=True))
    cal = calendar_features(ts)
    assert cal["is_holiday"].tolist() == [1, 0, 0]
    assert cal["is_weekend"].tolist() == [0, 1, 0]
    assert cal["tod"].iloc[0] == 12.0  # 02:00 UTC = 12:00 market time


def test_issue_every_subsamples(synthetic):
    d = synthetic
    a = build_features(d["dispatch_price"], d["region_demand"], horizons=(1,))
    b = build_features(d["dispatch_price"], d["region_demand"], horizons=(1,), issue_every="2h")
    assert len(b) == pytest.approx(len(a) / 4, rel=0.05)


# ------------------------------------------------------------------ models


def test_signed_log_roundtrip():
    x = np.array([-1000.0, -1.0, 0.0, 55.5, 17500.0])
    np.testing.assert_allclose(inv_signed_log(signed_log(x)), x)
    assert signed_log(np.array([15000.0]))[0] < 10


def test_sort_quantiles_removes_crossing():
    p = pd.DataFrame({"q0.1": [5.0], "q0.5": [3.0], "q0.9": [4.0]})
    out = sort_quantiles(p)
    assert out.iloc[0].tolist() == [3.0, 4.0, 5.0]


def test_quantile_gbm_and_shap_contributions(feats_long):
    f = feats_long.dropna(subset=["y_price"]).iloc[:4000]
    cols = feature_columns(f)
    m = QuantileGBM("price_lgbm", cols, (0.1, 0.5, 0.9), "signed_log", FAST).fit(f, f["y_price"])
    pred = m.predict(f.iloc[:200])
    assert (pred["q0.1"] <= pred["q0.5"]).all() and (pred["q0.5"] <= pred["q0.9"]).all()
    contrib = m.contributions(f.iloc[:50])
    # TreeSHAP values sum to the raw (signed-log) prediction.
    raw = m.models[0.5].predict(f.iloc[:50][cols])
    np.testing.assert_allclose(contrib.sum(axis=1), raw, rtol=1e-5, atol=1e-5)


def test_cqr_reaches_target_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 20000)
    narrow = pd.DataFrame({"q0.1": np.full_like(y, -0.5), "q0.5": np.zeros_like(y), "q0.9": np.full_like(y, 0.5)})
    cqr = CQR(alpha=0.2).fit(narrow.iloc[:10000], y[:10000])
    adj = cqr.transform(narrow.iloc[10000:])
    cov = M.coverage(y[10000:], adj["q0.1"], adj["q0.9"])
    assert 0.78 < cov < 0.82
    assert cqr.q_hat > 0  # intervals were widened
    with pytest.raises(ValueError):
        CQR().fit(narrow.iloc[:5], np.full(5, np.nan))


def test_spike_classifier_and_threshold():
    rng = np.random.default_rng(2)
    n = 6000
    x = rng.normal(0, 1, n)
    y = pd.Series((x + rng.normal(0, 0.3, n) > 2.3).astype(float))
    X = pd.DataFrame({"x": x, "noise": rng.normal(size=n)})
    m = SpikeClassifier(["x", "noise"], {"n_estimators": 50}).fit(X, y, order=pd.Series(np.arange(n)))
    p = m.predict_proba(X)
    assert ((p >= 0) & (p <= 1)).all()
    assert M.pr_auc(y, p) > 0.6
    t = choose_threshold(p, y.to_numpy(), value_tp=10, cost_fp=1)
    assert 0 <= t <= 1
    assert set(m.predict(X)["alert"].unique()) <= {0, 1}
    assert "bias" in m.contributions(X.iloc[:5]).columns


def test_baselines(feats_long):
    f = feats_long.dropna(subset=["y_demand"])
    sn = SeasonalNaive("demand", (0.1, 0.5, 0.9)).fit(f, f["y_demand"])
    p = sn.predict(f)
    assert p["q0.1"].mean() < p["q0.5"].mean() < p["q0.9"].mean()
    cl = Climatology((0.5,)).fit(f, f["y_demand"])
    assert cl.predict(f)["q0.5"].notna().all()
    pdb = PredispatchBaseline("demand").fit(f, f["y_demand"])
    assert pdb.predict(f)["q0.5"].notna().mean() > 0.9
    assert PredispatchBaseline("price").predict(f.drop(columns=["pd_price"]))["q0.5"].isna().all()


def test_registry_versions_and_promotion():
    card = lambda v: ModelCard("m", v, "demand", "now", ["a"])  # noqa: E731
    save_model({"w": 1}, card("v1"))
    save_model({"w": 2}, card("v2"))
    with pytest.raises(FileNotFoundError):
        load_model("m")  # nothing in Production yet
    set_stage("m", "v1", "Production")
    model, c = load_model("m")
    assert model == {"w": 1} and c.version == "v1"
    set_stage("m", "v2", "Production")
    stages = {c.version: c.stage for c in list_versions("m")}
    assert stages == {"v1": "Archived", "v2": "Production"}
    assert load_model("m", stage=None)[1].version == "v2"
    assert load_model("m", version="v1")[0] == {"w": 1}


# ------------------------------------------------------------------ metrics


def test_metrics_known_values():
    y, yhat = np.array([1.0, 2.0, 3.0, np.nan]), np.array([1.0, 3.0, 1.0, 5.0])
    assert M.mae(y, yhat) == pytest.approx(1.0)
    assert M.rmse(y, yhat) == pytest.approx(np.sqrt(5 / 3))
    assert M.pinball([10.0], [8.0], 0.9) == pytest.approx(1.8)
    assert M.pinball([10.0], [12.0], 0.9) == pytest.approx(0.2)
    assert M.coverage([1, 5, 10], [0, 0, 0], [6, 6, 6]) == pytest.approx(2 / 3)
    assert M.mape([100.0], [90.0]) == pytest.approx(0.1)
    assert np.isnan(M.pr_auc([0, 0], [0.1, 0.2]))
    rel = M.reliability(np.array([0, 1, 1, 0]), np.array([0.05, 0.95, 0.85, 0.15]), bins=2)
    assert rel["observed"].tolist() == [0.0, 1.0]


def test_diebold_mariano():
    rng = np.random.default_rng(3)
    e_good = rng.normal(0, 1, 2000)
    e_bad = rng.normal(0, 2, 2000)
    stat, p = M.diebold_mariano(e_good, e_bad, h=1)
    assert stat < 0 and p < 0.001
    # Two independent forecasts of equal quality: no significant difference.
    stat2, p2 = M.diebold_mariano(e_good, rng.normal(0, 1, 2000), h=4)
    assert p2 > 0.01
    # A forecast that is always slightly worse IS significantly worse.
    stat3, p3 = M.diebold_mariano(e_good, e_good * 1.01, h=1)
    assert stat3 < 0 and p3 < 0.05
    assert np.isnan(M.diebold_mariano([1.0], [2.0])[0])


# ------------------------------------------------------------------ backtesting


def test_walk_forward_folds_are_expanding_and_gapped():
    issued = pd.Series(pd.date_range("2023-01-01", "2024-01-01", freq="30min", tz="UTC"))
    folds = walk_forward_folds(issued, n_folds=4, fold_days=30, gap_days=7, min_train_days=180)
    assert len(folds) == 4
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end == b.test_start
        assert b.train_end > a.train_end
    for f in folds:
        assert f.test_start - f.train_end == pd.Timedelta(days=7)
    assert len(walk_forward_folds(issued, n_folds=20, fold_days=30, min_train_days=180)) < 20


def test_split_test_period():
    issued = pd.Series(pd.date_range("2024-01-01", "2024-12-31", freq="1D", tz="UTC"))
    assert split_test_period(issued, "2024-10-01", 90) == pd.Timestamp("2024-10-01", tz="UTC")
    fallback = split_test_period(issued, "2030-01-01", 90)
    assert pd.Timestamp("2024-10-01", tz="UTC") < fallback < pd.Timestamp("2024-11-01", tz="UTC")


def test_backtest_trains_only_on_the_past(feats_long):
    f = feats_long
    folds = walk_forward_folds(f["issued_utc"], **SMALL_BT)
    pred = run_backtest(f, TASKS["demand"], feature_columns(f), folds, gbm_params=FAST)
    assert set(pred["model"]) == {"seasonal_naive", "climatology", "aemo_predispatch", "demand_lgbm"}
    for fold in folds:
        rows = pred[pred["fold"] == fold.id]
        assert rows["issued_utc"].min() >= fold.test_start
        assert rows["issued_utc"].max() < fold.test_end
    s = score(pred, TASKS["demand"])
    lgbm = s.set_index("model").loc["demand_lgbm"]
    naive = s.set_index("model").loc["seasonal_naive"]
    assert lgbm["mae"] < naive["mae"]  # the model must beat the naive baseline
    assert 0.6 < lgbm["coverage_80"] < 0.95


def test_evaluate_task_price_beats_naive_and_compares_with_aemo(feats_long):
    r = evaluate_task(feats_long, "price", gbm_params=FAST, backtest_cfg=SMALL_BT)
    t = r.test.set_index("model")
    assert t.loc["price_lgbm", "mae"] < t.loc["seasonal_naive", "mae"]
    assert {"model_mae", "aemo_mae", "dm_p", "improvement_pct"} <= set(r.vs_aemo.columns)
    assert set(r.vs_aemo_by_horizon["horizon"]) <= {1, 2, 4, 8}
    path = write_results({"price": r})
    assert path.exists() and "Price" in path.read_text(encoding="utf-8")


def test_evaluate_spike_task(feats_long):
    r = evaluate_task(feats_long, "spike", gbm_params=FAST, backtest_cfg=SMALL_BT)
    t = r.test.set_index("model")
    assert t.loc["spike_lgbm", "pr_auc"] > t.loc["spike_lgbm", "spike_rate"] * 3  # far better than chance
    assert compare_with_aemo(r.test_predictions, TASKS["spike"]).empty


# ------------------------------------------------------------------ jobs + CLI


def test_build_store_train_and_load():
    load_synthetic(SyntheticConfig(days=40, regions=("SA1",), predispatch_horizon_steps=4))
    feats = build_and_store_features(horizons=(1, 2))
    assert len(load_features()) == len(feats)
    cards = train_production(feats, tasks=("demand", "spike"), gbm_params=FAST)
    assert cards["demand"].stage == "Production"
    model, card = load_model("demand_lgbm")
    assert card.version == cards["demand"].version
    assert model.predict(feats.head(10))["q0.5"].notna().all()
    # Second training run is registered as Staging, Production is unchanged.
    again = train_production(feats, tasks=("demand",), gbm_params=FAST)
    assert again["demand"].stage == "Staging"
    assert load_model("demand_lgbm")[1].version == cards["demand"].version


def test_cli_features_evaluate_train(capsys):
    main(["synth", "--days", "45", "--regions", "SA1"])
    assert main(["features", "--issue-every", "1h"]) == 0
    assert main(["evaluate", "--tasks", "demand", "--n-estimators", "30"]) == 0
    assert main(["train", "--tasks", "price", "--n-estimators", "30"]) == 0
    out = capsys.readouterr().out
    assert "demand_lgbm" in out and "price_lgbm" in out
