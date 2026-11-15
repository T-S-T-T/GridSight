"""Daily monitoring job and weekly retraining job."""

from __future__ import annotations

import logging

import pandas as pd

from gridsight.config import get_settings
from gridsight.evaluate.run import load_features, train_production
from gridsight.features.build import feature_columns
from gridsight.monitoring.checks import (
    CheckResult,
    alert,
    check_coverage,
    check_drift,
    check_freshness,
    check_live_accuracy,
    write_report,
)
from gridsight.serving import predictions_with_actuals, promote_if_better

log = logging.getLogger(__name__)


def _reference_error(task: str) -> float | None:
    path = get_settings().reports_dir / "results" / f"{task}_test.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path).set_index("model")
    name = f"{task}_lgbm"
    col = "mape" if task == "demand" else "mae"
    return float(t.loc[name, col]) if name in t.index and col in t else None


def run_monitoring(now: pd.Timestamp | None = None, max_lag: str = "15min") -> pd.DataFrame:
    results: list[CheckResult] = [check_freshness(max_lag=max_lag, now=now)]
    feats = load_features()
    if not feats.empty:
        cut = feats["issued_utc"].max() - pd.Timedelta("7D")
        ref, recent = feats[feats["issued_utc"] < cut], feats[feats["issued_utc"] >= cut]
        cols = [c for c in feature_columns(feats) if feats[c].dtype.kind in "fi" and c not in ("horizon", "region_code")]
        if len(ref) and len(recent):
            results.append(check_drift(ref, recent, cols))
    pred = predictions_with_actuals()
    if not pred.empty:
        for task in ("demand", "price"):
            ref_err = _reference_error(task)
            if ref_err is not None:
                results.append(check_live_accuracy(pred, task, ref_err))
            results.append(check_coverage(pred, task))
    alert(results)
    return write_report(results)


def retrain(weeks: int = 8, tasks: tuple[str, ...] = ("demand", "price", "spike"), gbm_params: dict | None = None) -> list[dict]:
    """Train challengers on all data, then promote each only if it beats Production on the most
    recent ``weeks`` of labelled data."""
    feats = load_features()
    if feats.empty:
        raise RuntimeError("no features: run `gridsight features` first")
    train_production(feats, tasks=tasks, gbm_params=gbm_params, promote_if_first=True)
    recent = feats[feats["issued_utc"] >= feats["issued_utc"].max() - pd.Timedelta(weeks=weeks)]
    decisions = [promote_if_better(t, recent if t != "spike" else recent[recent["horizon"] <= 2]) for t in tasks]
    for d in decisions:
        log.info("promotion: %s", d)
    return decisions
