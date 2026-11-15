"""GridSight REST API (read-only). Run with ``uvicorn gridsight.api.app:app``; docs at ``/docs``."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from gridsight import REGIONS, __version__, storage
from gridsight.config import get_settings
from gridsight.models.base import list_versions
from gridsight.serving import PRODUCTION_MODELS, SERVING_LAYER, predictions_with_actuals
from gridsight.transform.quality import freshness

app = FastAPI(
    title="GridSight API",
    version=__version__,
    description="Forecasts of NEM demand, prices and price spikes, compared with AEMO pre-dispatch.",
)

_CACHE: dict[str, tuple[float, Any]] = {}
CACHE_SECONDS = 60


def cached(key: str, fn: Callable[[], Any]) -> Any:
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < CACHE_SECONDS:
        return hit[1]
    value = fn()
    _CACHE[key] = (now, value)
    return value


def clear_cache() -> None:
    _CACHE.clear()


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.replace({np.nan: None}).to_dict(orient="records")


def _region(region: str) -> str:
    region = region.upper()
    if region not in REGIONS:
        raise HTTPException(status_code=422, detail=f"region must be one of {', '.join(REGIONS)}")
    return region


def _latest_predictions(task: str, region: str, horizon_hours: float) -> pd.DataFrame:
    pred = cached("predictions", lambda: storage.read("predictions", layer=SERVING_LAYER))
    if pred.empty:
        raise HTTPException(status_code=404, detail="no predictions yet: run `gridsight score`")
    p = pred[(pred["task"] == task) & (pred["region"] == region)]
    if p.empty:
        raise HTTPException(status_code=404, detail=f"no {task} predictions for {region}")
    p = p[p["issued_utc"] == p["issued_utc"].max()]
    return p[p["horizon"] <= horizon_hours * 2].sort_values(["model", "target_utc"])


@app.get("/health")
def health() -> dict[str, Any]:
    latest = storage.latest_time("dispatch_price")
    f = freshness(latest)
    models = {}
    for task, name in PRODUCTION_MODELS.items():
        prod = [c.version for c in list_versions(name) if c.stage == "Production"]
        models[task] = prod[-1] if prod else None
    return {
        "status": "ok" if not f.stale else "stale",
        "latest_interval_utc": None if latest is None else latest.isoformat(),
        "lag_minutes": None if f.lag is None else round(f.lag.total_seconds() / 60, 1),
        "models": models,
        "version": __version__,
    }


@app.get("/v1/forecast/demand")
def forecast_demand(region: str = Query("NSW1"), horizon_hours: float = Query(48, gt=0, le=168)) -> dict[str, Any]:
    r = _region(region)
    p = _latest_predictions("demand", r, horizon_hours)
    cols = [c for c in ("model", "model_version", "issued_utc", "target_utc", "horizon", "q0.1", "q0.5", "q0.9") if c in p]
    return {"region": r, "unit": "MW", "forecasts": _records(p[cols])}


@app.get("/v1/forecast/price")
def forecast_price(region: str = Query("SA1"), horizon_hours: float = Query(24, gt=0, le=168)) -> dict[str, Any]:
    r = _region(region)
    p = _latest_predictions("price", r, horizon_hours)
    cols = [c for c in ("model", "model_version", "issued_utc", "target_utc", "horizon", "q0.1", "q0.5", "q0.9") if c in p]
    return {"region": r, "unit": "AUD/MWh", "forecasts": _records(p[cols])}


@app.get("/v1/spike")
def spike(region: str = Query("SA1")) -> dict[str, Any]:
    r = _region(region)
    p = _latest_predictions("spike", r, 168)
    threshold = get_settings().spike_threshold
    cols = [c for c in ("model_version", "issued_utc", "target_utc", "horizon", "p_spike", "alert") if c in p]
    return {"region": r, "threshold_aud_mwh": threshold, "forecasts": _records(p[cols])}


@app.get("/v1/battery/pnl")
def battery_pnl() -> dict[str, Any]:
    path = get_settings().reports_dir / "battery" / "summary.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no battery backtest yet: run `gridsight battery`")
    return {"strategies": _records(pd.read_csv(path))}


@app.get("/v1/metrics/live")
def metrics_live(days: int = Query(30, gt=0, le=365)) -> dict[str, Any]:
    from gridsight.evaluate import metrics as M

    df = cached(f"live:{days}", lambda: predictions_with_actuals(days))
    if df.empty:
        raise HTTPException(status_code=404, detail="no stored predictions yet")
    rows = []
    for (task, model, region), g in df.dropna(subset=["y"]).groupby(["task", "model", "region"]):
        if task == "spike":
            rows.append({"task": task, "model": model, "region": region, "n": len(g), "pr_auc": M.pr_auc(g["y"], g["p_spike"])})
        else:
            rows.append({"task": task, "model": model, "region": region, "n": len(g),
                         "mae": M.mae(g["y"], g["q0.5"]), "mape": M.mape(g["y"], g["q0.5"])})
    return {"days": days, "metrics": _records(pd.DataFrame(rows))}
