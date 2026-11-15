"""Walk-forward backtesting, scoring and the comparison with AEMO pre-dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gridsight.evaluate import metrics as M
from gridsight.models.tasks import Task, fit_model, model_factories

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    id: int
    train_end: pd.Timestamp  # training targets must be strictly before this
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(
    issued: pd.Series,
    n_folds: int = 6,
    fold_days: int = 60,
    gap_days: int = 7,
    min_train_days: int = 180,
    end: pd.Timestamp | None = None,
) -> list[Fold]:
    """Expanding-window folds ending at ``end`` (default: last issue time).

    A gap of ``gap_days`` separates each fold's training targets from its test issues, so lag
    features computed for the test window never overlap training labels."""
    start, last = issued.min(), (end if end is not None else issued.max() + pd.Timedelta("30min"))
    folds = []
    for k in range(n_folds):
        test_start = last - pd.Timedelta(days=fold_days * (n_folds - k))
        test_end = test_start + pd.Timedelta(days=fold_days)
        train_end = test_start - pd.Timedelta(days=gap_days)
        if train_end - start < pd.Timedelta(days=min_train_days):
            log.debug("skipping fold %d: not enough training data before %s", k, train_end)
            continue
        folds.append(Fold(len(folds), train_end, test_start, test_end))
    return folds


def split_test_period(issued: pd.Series, test_start: str | None, min_train_days: int, fallback_frac: float = 0.2) -> pd.Timestamp:
    """The locked test period starts at ``test_start`` if the data allows it, otherwise the
    last ``fallback_frac`` of issue times is used."""
    lo, hi = issued.min(), issued.max()
    if test_start is not None:
        ts = pd.Timestamp(test_start, tz="UTC")
        if lo + pd.Timedelta(days=min_train_days) < ts < hi:
            return ts
    return (lo + (hi - lo) * (1 - fallback_frac)).floor("30min")


def _predict_frame(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    return model.predict(X)


def run_backtest(
    feats: pd.DataFrame,
    task: Task,
    features: list[str],
    folds: list[Fold],
    **factory_kwargs: Any,
) -> pd.DataFrame:
    """Fit every model in the ladder on each fold and return long-format predictions."""
    factories = model_factories(task, features, **factory_kwargs)
    y_all = feats[task.target]
    out = []
    for fold in folds:
        train = feats[(feats["target_utc"] < fold.train_end) & y_all.notna()]
        test = feats[(feats["issued_utc"] >= fold.test_start) & (feats["issued_utc"] < fold.test_end) & y_all.notna()]
        if train.empty or test.empty:
            continue
        for name, make in factories.items():
            model = fit_model(make(), train, train[task.target], order=train["issued_utc"])
            pred = _predict_frame(model, test)
            frame = test[["issued_utc", "target_utc", "region", "horizon"]].copy()
            frame["y"] = test[task.target].to_numpy()
            frame = pd.concat([frame, pred], axis=1)
            frame["model"], frame["fold"] = name, fold.id
            out.append(frame)
        log.info("fold %d done (%d train, %d test rows)", fold.id, len(train), len(test))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score(pred: pd.DataFrame, task: Task, by: list[str] | None = None) -> pd.DataFrame:
    """Metrics per model (and optional extra grouping such as horizon or region)."""
    keys = ["model", *(by or [])]
    rows = []
    for key, g in pred.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key, strict=True))
        row["n"] = len(g)
        if task.kind == "quantile":
            row["mae"] = M.mae(g["y"], g["q0.5"])
            row["rmse"] = M.rmse(g["y"], g["q0.5"])
            row["mape"] = M.mape(g["y"], g["q0.5"])
            qcols = [c for c in g.columns if c.startswith("q") and g[c].notna().any()]
            row["pinball"] = M.mean_pinball(g["y"], g[qcols]) if qcols else np.nan
            if {"q0.1", "q0.9"} <= set(qcols) and g["q0.1"].ne(g["q0.9"]).any():
                row["coverage_80"] = M.coverage(g["y"], g["q0.1"], g["q0.9"])
                row["width_80"] = M.interval_width(g["q0.1"], g["q0.9"])
        else:
            row["pr_auc"] = M.pr_auc(g["y"], g["p_spike"])
            row["brier"] = M.brier(g["y"], g["p_spike"])
            row["spike_rate"] = float(g["y"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_with_aemo(pred: pd.DataFrame, task: Task, by: list[str] | None = None) -> pd.DataFrame:
    """Each model vs AEMO pre-dispatch on exactly the rows where both forecasts exist.

    Returns the error of each, the improvement in % and a Diebold-Mariano test."""
    if task.kind != "quantile" or "aemo_predispatch" not in set(pred["model"]):
        return pd.DataFrame()
    key = ["issued_utc", "target_utc", "region", "horizon"]
    aemo = pred[pred["model"] == "aemo_predispatch"][[*key, "y", "q0.5"]].rename(columns={"q0.5": "aemo"})
    rows = []
    for model, g in pred[pred["model"] != "aemo_predispatch"].groupby("model"):
        m = g[[*key, "q0.5"]].merge(aemo, on=key).dropna(subset=["q0.5", "aemo", "y"])
        groups = m.groupby(by) if by else [((), m)]
        for gk, mm in groups:
            if len(mm) < 10:
                continue
            e_model, e_aemo = mm["y"] - mm["q0.5"], mm["y"] - mm["aemo"]
            h = int(mm["horizon"].max()) if "horizon" in mm else 1
            stat, p = M.diebold_mariano(e_model, e_aemo, h=max(h, 1))
            metric = M.mape if task.primary_metric == "mape" else M.mae
            em, ea = metric(mm["y"], mm["q0.5"]), metric(mm["y"], mm["aemo"])
            row = {"model": model, "n": len(mm), f"model_{task.primary_metric}": em, f"aemo_{task.primary_metric}": ea,
                   "improvement_pct": 100 * (ea - em) / ea if ea else np.nan, "dm_stat": stat, "dm_p": p,
                   "significant": bool(p < 0.05 and stat < 0) if np.isfinite(p) else False}
            if by:
                row.update(dict(zip(by, gk if isinstance(gk, tuple) else (gk,), strict=True)))
            rows.append(row)
    return pd.DataFrame(rows)
