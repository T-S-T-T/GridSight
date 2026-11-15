"""Phase 5: live scoring, promotion/rollback, monitoring, API, dashboard, CLI and the end-to-end demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import responses
from fastapi.testclient import TestClient

from gridsight import storage
from gridsight.api import app as api_module
from gridsight.cli import main
from gridsight.config import get_settings, project_root
from gridsight.evaluate.run import build_and_store_features, evaluate_all, train_production
from gridsight.models.base import list_versions
from gridsight.monitoring import checks
from gridsight.monitoring.job import retrain, run_monitoring
from gridsight.pipeline import load_synthetic
from gridsight.serving import (
    SERVING_LAYER,
    predictions_with_actuals,
    promote_if_better,
    rollback,
    score_latest,
)
from gridsight.synthetic import SyntheticConfig

FAST = {"n_estimators": 30, "learning_rate": 0.1}


def _prepare(evaluate: bool = True) -> pd.DataFrame:
    load_synthetic(SyntheticConfig(days=45, regions=("SA1",), predispatch_horizon_steps=8))
    feats = build_and_store_features(horizons=(1, 2, 4, 8, 48), issue_every="1h")
    if evaluate:
        evaluate_all(feats, gbm_params=FAST, backtest_cfg={"n_folds": 1, "fold_days": 7, "gap_days": 1, "min_train_days": 20})
    train_production(feats, gbm_params=FAST)
    return feats


@pytest.fixture
def client():
    api_module.clear_cache()
    return TestClient(api_module.app)


# ------------------------------------------------------------------ scoring, promotion, rollback


def test_score_latest_stores_all_tasks_and_aemo():
    _prepare(evaluate=False)
    pred = score_latest()
    assert set(pred["task"]) == {"demand", "price", "spike"}
    assert pred["issued_utc"].nunique() == 1
    assert {"price:aemo_predispatch", "demand:aemo_predispatch"} <= set(pred["model"])
    assert pred.loc[pred["task"] == "spike", "horizon"].max() <= 2
    assert pred.loc[pred["task"] == "spike", "p_spike"].between(0, 1).all()
    stored = storage.read("predictions", layer=SERVING_LAYER)
    assert len(stored) == len(pred)
    score_latest()  # idempotent
    assert len(storage.read("predictions", layer=SERVING_LAYER)) == len(pred)


def test_score_without_models_fails_clearly():
    load_synthetic(SyntheticConfig(days=12, regions=("SA1",)))
    with pytest.raises(RuntimeError, match="no Production models"):
        score_latest()
    storage_dir = get_settings().data_dir / "silver"
    assert storage_dir.exists()


def test_predictions_with_actuals_joins_settled_targets():
    _prepare(evaluate=False)
    pred = score_latest()
    # Pretend the forecast was issued two days earlier so its targets have settled.
    shifted = pred.assign(issued_utc=pred["issued_utc"] - pd.Timedelta("2D"), target_utc=pred["target_utc"] - pd.Timedelta("2D"))
    storage.write(shifted, "predictions", layer=SERVING_LAYER)
    joined = predictions_with_actuals()
    latest_actual = storage.latest_time("dispatch_price")
    settled = joined[(joined["issued_utc"] == shifted["issued_utc"].iloc[0]) & (joined["target_utc"] + pd.Timedelta("30min") <= latest_actual)]
    assert len(settled) > 10 and settled["y"].notna().all()
    future = joined[joined["target_utc"] > latest_actual]
    assert future["y"].isna().all()


def test_promotion_rejects_or_promotes_and_rollback_restores():
    feats = _prepare(evaluate=False)
    first = [c for c in list_versions("price_lgbm") if c.stage == "Production"][0].version
    # A much weaker challenger must be rejected.
    train_production(feats, tasks=("price",), gbm_params={"n_estimators": 2, "learning_rate": 0.01})
    d = promote_if_better("price", feats.tail(2000))
    assert d["promoted"] is False and d["challenger"] > d["champion"]
    # A stronger challenger is promoted; the old version is archived.
    train_production(feats, tasks=("price",), gbm_params={"n_estimators": 150, "learning_rate": 0.1})
    d = promote_if_better("price", feats.tail(2000))
    if d["promoted"]:
        stages = {c.version: c.stage for c in list_versions("price_lgbm")}
        assert stages[first] == "Archived"
        assert rollback("price") == first
        assert {c.version: c.stage for c in list_versions("price_lgbm")}[first] == "Production"
    assert promote_if_better("price", feats.tail(100))["reason"] == "no Staging version"


def test_rollback_without_history_fails():
    _prepare(evaluate=False)
    with pytest.raises(RuntimeError):
        rollback("demand")


def test_retrain_job_returns_decisions():
    _prepare(evaluate=False)
    decisions = retrain(weeks=2, tasks=("demand",), gbm_params=FAST)
    assert decisions[0]["task"] == "demand" and "promoted" in decisions[0]


# ------------------------------------------------------------------ monitoring


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert checks.psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert checks.psi(a, rng.normal(2, 1, 5000)) > 0.5
    assert np.isnan(checks.psi(a[:3], a[:3]))
    assert checks.psi(np.ones(100), np.ones(100)) == 0.0


def test_accuracy_and_coverage_checks():
    t = pd.date_range("2024-01-01", periods=100, freq="30min", tz="UTC")
    pred = pd.DataFrame({"task": "price", "model": "price:price_lgbm", "target_utc": t, "y": 100.0,
                         "q0.5": 130.0, "q0.1": 120.0, "q0.9": 140.0})
    bad = checks.check_live_accuracy(pred, "price", reference_error=10.0)
    assert not bad.ok and bad.value == pytest.approx(30.0)
    assert checks.check_live_accuracy(pred, "price", reference_error=30.0).ok
    cov = checks.check_coverage(pred, "price")
    assert not cov.ok and cov.value == 0.0
    assert checks.check_live_accuracy(pred.iloc[:0], "demand", 1.0).ok


@responses.activate
def test_alerts_post_failed_checks_to_webhook():
    url = "https://hooks.example.com/gridsight"
    responses.post(url, json={})
    results = [checks.CheckResult("ok", True, 1, "", ""), checks.CheckResult("data_freshness", False, 99.0, "lag <= 15min", "alert")]
    failed = checks.alert(results, webhook=url)
    assert [r.name for r in failed] == ["data_freshness"]
    assert len(responses.calls) == 1 and b"data_freshness" in responses.calls[0].request.body


def test_run_monitoring_writes_report():
    _prepare()
    score_latest()
    report = run_monitoring(max_lag="3650D")
    assert {"data_freshness", "feature_drift"} <= set(report["name"])
    assert report.set_index("name").loc["data_freshness", "ok"]
    assert (get_settings().reports_dir / "monitoring" / "latest.json").exists()
    stale = run_monitoring(now=pd.Timestamp("2030-01-01", tz="UTC"))
    assert not stale.set_index("name").loc["data_freshness", "ok"]


# ------------------------------------------------------------------ API


def test_api_health_without_data(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "stale"
    assert client.get("/v1/forecast/price", params={"region": "SA1"}).status_code == 404
    assert client.get("/v1/battery/pnl").status_code == 404


def test_api_serves_forecasts(client):
    _prepare()
    score_latest()
    h = client.get("/health").json()
    assert h["models"]["price"] is not None and h["latest_interval_utc"]
    r = client.get("/v1/forecast/price", params={"region": "sa1", "horizon_hours": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["unit"] == "AUD/MWh"
    models = {f["model"] for f in body["forecasts"]}
    assert models == {"price:price_lgbm", "price:aemo_predispatch"}
    assert max(f["horizon"] for f in body["forecasts"]) <= 8
    d = client.get("/v1/forecast/demand", params={"region": "SA1"}).json()
    assert all(f["q0.5"] is not None for f in d["forecasts"] if f["model"] == "demand:demand_lgbm")
    s = client.get("/v1/spike", params={"region": "SA1"}).json()
    assert s["threshold_aud_mwh"] == 300 and 0 <= s["forecasts"][0]["p_spike"] <= 1
    assert client.get("/v1/forecast/price", params={"region": "WA1"}).status_code == 422
    assert client.get("/v1/forecast/price", params={"region": "NSW1"}).status_code == 404
    assert client.get("/v1/metrics/live").status_code == 200


def test_api_battery_endpoint(client):
    _prepare()
    from gridsight.battery.job import run_battery_backtest

    run_battery_backtest("SA1", days=1, overrides={"horizon_hours": 4})
    r = client.get("/v1/battery/pnl")
    assert r.status_code == 200
    assert {s["strategy"] for s in r.json()["strategies"]} >= {"perfect_foresight", "gridsight_p50", "rule_based"}


# ------------------------------------------------------------------ dashboard


def test_dashboard_pages_render_without_errors():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    _prepare()
    score_latest()
    from gridsight.battery.job import run_battery_backtest

    run_battery_backtest("SA1", days=1, overrides={"horizon_hours": 4})
    path = str(project_root() / "dashboard" / "app.py")
    for page in ["Now", "Accuracy", "Battery", "Explain", "Market story"]:
        at = AppTest.from_file(path, default_timeout=60)
        at.run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"


def test_dashboard_without_data_shows_warning():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(project_root() / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No data yet" in w.value for w in at.warning)


# ------------------------------------------------------------------ CLI + end to end


def test_cli_score_monitor_rollback(capsys):
    _prepare()
    assert main(["score"]) == 0
    assert main(["monitor", "--max-lag", "3650D"]) in (0, 1)
    with pytest.raises(RuntimeError):
        main(["rollback", "spike"])
    assert "stored" in capsys.readouterr().out


def test_cli_loop_runs_one_iteration(monkeypatch):
    _prepare(evaluate=False)
    import gridsight.pipeline as pipeline

    monkeypatch.setattr(pipeline, "ingest_live", lambda: "skipped (offline test)")
    assert main(["loop", "--iterations", "1", "--every", "1"]) == 0
    assert not storage.read("predictions", layer=SERVING_LAYER).empty


@pytest.mark.slow
def test_demo_runs_the_whole_pipeline_end_to_end(capsys):
    assert main(["demo", "--days", "40", "--regions", "SA1", "--n-estimators", "30"]) == 0
    out = capsys.readouterr().out
    for stage in ["synthetic data", "EDA report", "features", "evaluate", "train", "battery", "score", "monitor", "done"]:
        assert f"[demo] {stage}" in out
    s = get_settings()
    assert (s.reports_dir / "results" / "results.md").exists()
    assert (s.reports_dir / "battery" / "summary.csv").exists()
    assert (s.reports_dir / "eda" / "README.md").exists()


def test_prefect_flows_import_when_installed():
    pytest.importorskip("prefect")
    from gridsight.flows import prefect_flows

    assert callable(prefect_flows.ingest_and_score)
