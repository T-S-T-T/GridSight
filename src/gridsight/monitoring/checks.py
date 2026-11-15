"""Monitoring: data freshness, feature drift (PSI), live accuracy and interval coverage.

Each check returns a :class:`CheckResult`; failed checks become alerts that can be posted to a
Slack or Discord webhook set in ``GRIDSIGHT_ALERT_WEBHOOK``. The design doc planned Evidently;
these checks implement the same thresholds with no extra dependency.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import requests

from gridsight import storage
from gridsight.config import get_settings
from gridsight.evaluate import metrics as M
from gridsight.transform.quality import freshness

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    value: float | None
    threshold: str
    action: str
    detail: str = ""


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index between a reference and a recent sample."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < bins or len(a) < bins:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(reference: pd.DataFrame, recent: pd.DataFrame, features: list[str], threshold: float = 0.2) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(reference[f].to_numpy(), recent[f].to_numpy())} for f in features if f in reference and f in recent]
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    df["drifted"] = df["psi"] > threshold
    return df


def check_freshness(max_lag: str = "15min", now: pd.Timestamp | None = None) -> CheckResult:
    f = freshness(storage.latest_time("dispatch_price"), now=now, max_lag=max_lag)
    lag_min = None if f.lag is None else f.lag.total_seconds() / 60
    return CheckResult("data_freshness", not f.stale, lag_min, f"lag <= {max_lag}", "alert; dashboard shows a stale banner",
                       f"latest interval {f.latest}")


def check_drift(reference: pd.DataFrame, recent: pd.DataFrame, features: list[str], threshold: float = 0.2) -> CheckResult:
    d = feature_drift(reference, recent, features, threshold)
    worst = d.iloc[0] if not d.empty else None
    n = int(d["drifted"].sum()) if not d.empty else 0
    return CheckResult("feature_drift", n == 0, float(worst["psi"]) if worst is not None else None, f"PSI <= {threshold}",
                       "alert and trigger an early retrain", f"{n} drifted; worst {worst['feature'] if worst is not None else '-'}")


def check_live_accuracy(pred: pd.DataFrame, task: str, reference_error: float, tolerance: float = 0.25, days: int = 7) -> CheckResult:
    """Rolling error of the Production model vs the error it had on the test period."""
    p = pred[(pred["task"] == task) & (~pred["model"].str.endswith("aemo_predispatch"))].dropna(subset=["y", "q0.5"])
    if p.empty:
        return CheckResult(f"live_accuracy_{task}", True, None, "n/a", "none", "no settled predictions yet")
    p = p[p["target_utc"] >= p["target_utc"].max() - pd.Timedelta(days=days)]
    err = M.mape(p["y"], p["q0.5"]) if task == "demand" else M.mae(p["y"], p["q0.5"])
    limit = reference_error * (1 + tolerance)
    return CheckResult(f"live_accuracy_{task}", err <= limit, err, f"<= {limit:.4g}", "alert and roll back the model",
                       f"{len(p)} settled forecasts")


def check_coverage(pred: pd.DataFrame, task: str, lo: float = 0.7, hi: float = 0.9) -> CheckResult:
    p = pred[(pred["task"] == task)].dropna(subset=["y", "q0.1", "q0.9"]) if {"q0.1", "q0.9"} <= set(pred) else pd.DataFrame()
    if p.empty:
        return CheckResult(f"coverage_{task}", True, None, "n/a", "none", "no settled intervals yet")
    cov = M.coverage(p["y"], p["q0.1"], p["q0.9"])
    return CheckResult(f"coverage_{task}", lo <= cov <= hi, cov, f"{lo:.0%}-{hi:.0%}", "recalibrate the conformal intervals")


def alert(results: list[CheckResult], webhook: str | None = None) -> list[CheckResult]:
    """Post failed checks to a webhook (Slack- and Discord-compatible payload)."""
    failed = [r for r in results if not r.ok]
    webhook = webhook or os.environ.get("GRIDSIGHT_ALERT_WEBHOOK")
    if failed and webhook:
        text = "\n".join(f":warning: {r.name}: {r.value} (limit {r.threshold}) -> {r.action}" for r in failed)
        try:
            requests.post(webhook, json={"text": text, "content": text}, timeout=10)
        except requests.RequestException as exc:
            log.error("alert webhook failed: %s", exc)
    return failed


def write_report(results: list[CheckResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])
    out = get_settings().reports_dir / "monitoring"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "latest.csv", index=False)
    (out / "latest.json").write_text(json.dumps([asdict(r) for r in results], default=str, indent=2), encoding="utf-8")
    return df
