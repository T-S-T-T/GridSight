"""Live scoring, model promotion and rollback.

Scoring runs every 5 minutes: build features for the latest issue time from recent silver data,
score every Production model, and store the predictions (plus AEMO's pre-dispatch for the same
targets) in the ``predictions`` table. The API reads only that table, so a model failure can
never take the dashboard down.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from gridsight import storage
from gridsight.config import get_settings, models_config
from gridsight.evaluate import metrics as M
from gridsight.features.build import build_features
from gridsight.models.base import list_versions, load_model, set_stage

log = logging.getLogger(__name__)

PRODUCTION_MODELS = {"demand": "demand_lgbm", "price": "price_lgbm", "spike": "spike_lgbm"}
SERVING_LAYER = "serving"


def recent_features(lookback_days: int = 9, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Features for the most recent issue time only."""
    latest = storage.latest_time("dispatch_price")
    if latest is None:
        raise RuntimeError("no silver data to score")
    now = now or latest + pd.Timedelta("5min")
    start = now - pd.Timedelta(days=lookback_days)
    cfg = models_config()
    horizons = tuple(sorted(set(cfg["horizons"]["demand"]) | set(cfg["horizons"]["price"])))
    feats = build_features(
        storage.read("dispatch_price", start=start),
        storage.read("region_demand", start=start),
        weather=storage.read("weather_forecast", start=start - pd.Timedelta("2D")),
        predispatch=storage.read("predispatch", start=start),
        horizons=horizons,
        spike_threshold=get_settings().spike_threshold,
    )
    if feats.empty:
        raise RuntimeError("not enough recent data to build features (need about 8 days)")
    return feats[feats["issued_utc"] == feats["issued_utc"].max()].reset_index(drop=True)


def score_latest(now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Score every Production model on the latest features and store the predictions."""
    feats = recent_features(now=now)
    keys = feats[["issued_utc", "target_utc", "region", "horizon"]]
    frames: list[pd.DataFrame] = []
    for task, name in PRODUCTION_MODELS.items():
        try:
            model, card = load_model(name)
        except FileNotFoundError:
            log.warning("no Production model for %s; skipping", task)
            continue
        X = feats if task != "spike" else feats[feats["horizon"] <= 2]
        pred = model.predict(X)
        pred = pd.concat([keys.loc[X.index].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        frames.append(pred.assign(model=name, model_version=card.version, task=task))
    if not frames:
        raise RuntimeError("no Production models registered: run `gridsight train` first")
    for task, col in (("demand", "pd_demand"), ("price", "pd_price")):
        if col in feats:
            frames.append(keys.assign(**{"q0.5": feats[col].to_numpy()}, model="aemo_predispatch", model_version="aemo", task=task))
    out = pd.concat(frames, ignore_index=True)
    out["model"] = out["task"] + ":" + out["model"]
    storage.write(out, "predictions", layer=SERVING_LAYER)
    log.info("stored %d predictions issued %s", len(out), out["issued_utc"].max())
    return out


def predictions_with_actuals(days: int = 30) -> pd.DataFrame:
    """Stored predictions joined with the half-hour actuals that have settled since."""
    from gridsight.features.build import half_hourly

    pred = storage.read("predictions", layer=SERVING_LAYER)
    if pred.empty:
        return pred
    start = pred["target_utc"].max() - pd.Timedelta(days=days)
    pred = pred[pred["target_utc"] >= start]
    hh = half_hourly(storage.read("dispatch_price", start=start), storage.read("region_demand", start=start), get_settings().spike_threshold)
    hh = hh.rename(columns={"interval_utc": "target_utc"})
    out = pred.merge(hh[["region", "target_utc", "demand", "price", "spike"]], on=["region", "target_utc"], how="left")
    out["y"] = np.select([out["task"] == "demand", out["task"] == "price"], [out["demand"], out["price"]], out["spike"])
    return out.drop(columns=["demand", "price", "spike"])


# ---------------------------------------------------------------- promotion and rollback


def _metric(model: Any, task: str, f: pd.DataFrame) -> float:
    target = {"demand": "y_demand", "price": "y_price", "spike": "y_spike"}[task]
    f = f.dropna(subset=[target])
    if task == "spike":
        return -M.pr_auc(f[target], model.predict_proba(f))  # lower is better
    pred = model.predict(f)["q0.5"]
    return M.mape(f[target], pred) if task == "demand" else M.mae(f[target], pred)


def promote_if_better(task: str, recent: pd.DataFrame, min_improvement: float = 0.0) -> dict[str, Any]:
    """Compare the newest Staging version with Production on recent labelled features and promote
    the challenger only if it is better by more than ``min_improvement`` (relative)."""
    name = PRODUCTION_MODELS[task]
    cards = list_versions(name)
    staging = [c for c in cards if c.stage == "Staging"]
    if not staging:
        return {"task": task, "promoted": False, "reason": "no Staging version"}
    challenger_card = staging[-1]
    challenger, _ = load_model(name, version=challenger_card.version)
    try:
        champion, champion_card = load_model(name)
    except FileNotFoundError:
        set_stage(name, challenger_card.version, "Production")
        return {"task": task, "promoted": True, "reason": "no Production version"}
    m_new, m_old = _metric(challenger, task, recent), _metric(champion, task, recent)
    better = m_new < m_old - abs(m_old) * min_improvement
    if better:
        set_stage(name, challenger_card.version, "Production")
    else:
        set_stage(name, challenger_card.version, "Rejected")
    return {"task": task, "promoted": bool(better), "challenger": m_new, "champion": m_old,
            "version": challenger_card.version, "previous": champion_card.version}


def rollback(task: str) -> str:
    """Put the most recent Archived version back into Production."""
    name = PRODUCTION_MODELS[task]
    archived = [c for c in list_versions(name) if c.stage == "Archived"]
    if not archived:
        raise RuntimeError(f"no archived version of {name} to roll back to")
    current = [c for c in list_versions(name) if c.stage == "Production"]
    set_stage(name, archived[-1].version, "Production")
    for c in current:
        set_stage(name, c.version, "RolledBack")
    return archived[-1].version
