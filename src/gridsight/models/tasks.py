"""The three forecasting tasks and the model ladder for each."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gridsight.models.baselines import Climatology, PredispatchBaseline, SeasonalNaive
from gridsight.models.conformal import CQR
from gridsight.models.gbm import QuantileGBM, SpikeClassifier


@dataclass(frozen=True)
class Task:
    name: str
    target: str
    kind: str  # "quantile" or "probability"
    primary_metric: str
    transform: str | None = None
    baselines: tuple[str, ...] = field(default_factory=tuple)


TASKS: dict[str, Task] = {
    "demand": Task("demand", "y_demand", "quantile", "mape", None, ("seasonal_naive", "climatology", "aemo_predispatch")),
    "price": Task("price", "y_price", "quantile", "mae", "signed_log", ("seasonal_naive", "aemo_predispatch")),
    "spike": Task("spike", "y_spike", "probability", "pr_auc", None, ("aemo_predispatch",)),
}


class ConformalGBM:
    """QuantileGBM whose outer quantiles are conformalized on the most recent slice of training."""

    def __init__(self, gbm: QuantileGBM, alpha: float = 0.2, calib_frac: float = 0.2):
        self.gbm = gbm
        self.name = gbm.name
        self.quantiles = gbm.quantiles
        self.features = gbm.features
        self.calib_frac = calib_frac
        lo, hi = min(gbm.quantiles), max(gbm.quantiles)
        self.cqr = CQR(alpha=alpha, lower=f"q{lo:g}", upper=f"q{hi:g}")

    def fit(self, X: pd.DataFrame, y: pd.Series, order: pd.Series | None = None) -> ConformalGBM:
        idx = np.argsort(order.to_numpy()) if order is not None else np.arange(len(X))
        n_cal = max(int(len(idx) * self.calib_frac), 1)
        tr, cal = idx[:-n_cal], idx[-n_cal:]
        self.gbm.fit(X.iloc[tr], y.iloc[tr])
        self.cqr.fit(self.gbm.predict(X.iloc[cal]), y.iloc[cal])
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.cqr.transform(self.gbm.predict(X))

    def contributions(self, X: pd.DataFrame, q: float = 0.5) -> pd.DataFrame:
        return self.gbm.contributions(X, q)


class PredispatchSpike:
    """Baseline spike 'probability': 1 if AEMO pre-dispatch price is above the threshold."""

    name = "aemo_predispatch"

    def __init__(self, threshold: float = 300.0):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series, order: pd.Series | None = None) -> PredispatchSpike:
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = X["pd_price"] if "pd_price" in X else pd.Series(np.nan, index=X.index)
        prob = (p > self.threshold).astype(float).where(p.notna())
        return pd.DataFrame({"p_spike": prob}, index=X.index)


def model_factories(
    task: Task,
    features: list[str],
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    gbm_params: dict[str, Any] | None = None,
    alpha: float = 0.2,
    spike_threshold: float = 300.0,
    include_baselines: bool = True,
) -> dict[str, Callable[[], Any]]:
    """Name -> zero-argument constructor for every model in the task's ladder."""
    f: dict[str, Callable[[], Any]] = {}
    short = task.name
    if task.kind == "quantile":
        if include_baselines:
            if "seasonal_naive" in task.baselines:
                f["seasonal_naive"] = lambda: SeasonalNaive(short, quantiles)
            if "climatology" in task.baselines:
                f["climatology"] = lambda: Climatology(quantiles)
            if "aemo_predispatch" in task.baselines:
                f["aemo_predispatch"] = lambda: PredispatchBaseline(short, quantiles)
        f[f"{short}_lgbm"] = lambda: ConformalGBM(
            QuantileGBM(f"{short}_lgbm", features, quantiles, task.transform, gbm_params), alpha=alpha
        )
    else:
        if include_baselines:
            f["aemo_predispatch"] = lambda: PredispatchSpike(spike_threshold)
        f["spike_lgbm"] = lambda: SpikeClassifier(features, gbm_params)
    return f


def fit_model(model: Any, X: pd.DataFrame, y: pd.Series, order: pd.Series) -> Any:
    """Fit any ladder model; models that use time order for calibration receive it."""
    if isinstance(model, ConformalGBM | SpikeClassifier | PredispatchSpike):
        return model.fit(X, y, order=order)
    return model.fit(X, y)
