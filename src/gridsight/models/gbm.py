"""LightGBM models: quantile regression for demand and price, a calibrated spike classifier."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from gridsight.models.base import inv_signed_log, qcol, signed_log, sort_quantiles


class QuantileGBM:
    """One LightGBM model per quantile. ``transform='signed_log'`` is used for prices."""

    def __init__(
        self,
        name: str,
        features: list[str],
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        transform: str | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.name = name
        self.features = features
        self.quantiles = tuple(quantiles)
        self.transform = transform
        self.params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1, **(params or {})}
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def _fwd(self, y: np.ndarray) -> np.ndarray:
        return signed_log(y) if self.transform == "signed_log" else y

    def _inv(self, z: np.ndarray) -> np.ndarray:
        return inv_signed_log(z) if self.transform == "signed_log" else z

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileGBM:
        mask = y.notna().to_numpy()
        Xf, yf = X.loc[mask, self.features], self._fwd(y.to_numpy(dtype=float)[mask])
        for q in self.quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
            m.fit(Xf, yf)
            self.models[q] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for q, m in self.models.items():
            out[qcol(q)] = self._inv(m.predict(X[self.features]))
        return sort_quantiles(out)

    def contributions(self, X: pd.DataFrame, q: float = 0.5) -> pd.DataFrame:
        """Per-row SHAP values (TreeSHAP, built into LightGBM) for one quantile model.
        Values are in model units (signed-log for prices). The last column is the bias."""
        contrib = self.models[q].predict(X[self.features], pred_contrib=True)
        return pd.DataFrame(contrib, columns=[*self.features, "bias"], index=X.index)


# Spikes are rare (well under 1% of half-hours). Row bagging and column sampling can drop the few
# positive examples or the key pre-dispatch column from most trees, so the spike model turns them off.
SPIKE_OVERRIDES: dict[str, Any] = {"subsample": 1.0, "subsample_freq": 0, "colsample_bytree": 1.0, "min_child_samples": 5}


class _Platt:
    """Logistic calibration on the logit of the raw score: monotonic, so ranking is preserved."""

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e4)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> _Platt:
        self.lr.fit(self._logit(raw), y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._logit(raw))[:, 1]


class SpikeClassifier:
    """LightGBM classifier with class weighting and probability calibration on the most recent
    slice of training data: Platt scaling by default, isotonic when there are at least
    ``isotonic_min_pos`` positive examples to calibrate on."""

    def __init__(
        self,
        features: list[str],
        params: dict[str, Any] | None = None,
        calib_frac: float = 0.2,
        isotonic_min_pos: int = 50,
    ):
        self.name = "spike_lgbm"
        self.features = features
        self.calib_frac = calib_frac
        self.isotonic_min_pos = isotonic_min_pos
        self.params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1, **(params or {}), **SPIKE_OVERRIDES}
        self.model: lgb.LGBMClassifier | None = None
        self.calibrator: IsotonicRegression | _Platt | None = None
        self.threshold: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series, order: pd.Series | None = None) -> SpikeClassifier:
        mask = y.notna().to_numpy()
        X, y = X.loc[mask], y[mask].astype(int)
        # Calibrate on the most recent slice (time order), train on the rest.
        idx = np.argsort(order[mask].to_numpy()) if order is not None else np.arange(len(X))
        n_cal = max(int(len(idx) * self.calib_frac), 1)
        tr, cal = idx[:-n_cal], idx[-n_cal:]
        pos = max(int(y.iloc[tr].sum()), 1)
        weight = (len(tr) - pos) / pos
        self.model = lgb.LGBMClassifier(scale_pos_weight=weight, **self.params)
        self.model.fit(X.iloc[tr][self.features], y.iloc[tr])
        raw = self.model.predict_proba(X.iloc[cal][self.features])[:, 1]
        y_cal = y.iloc[cal].to_numpy()
        n_pos = int(y_cal.sum())
        if n_pos >= self.isotonic_min_pos:
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y_cal)
        elif 0 < n_pos < len(y_cal):
            self.calibrator = _Platt().fit(raw, y_cal)
        else:
            self.calibrator = None  # nothing to calibrate against: use raw (weighted) scores
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit first"
        raw = self.model.predict_proba(X[self.features])[:, 1]
        return raw if self.calibrator is None else self.calibrator.predict(raw)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_proba(X)
        return pd.DataFrame({"p_spike": p, "alert": (p >= self.threshold).astype(int)}, index=X.index)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        contrib = self.model.predict_proba(X[self.features], pred_contrib=True)
        return pd.DataFrame(contrib, columns=[*self.features, "bias"], index=X.index)


def choose_threshold(p: np.ndarray, y: np.ndarray, value_tp: float = 1.0, cost_fp: float = 0.1) -> float:
    """Alert threshold that maximises value: each caught spike earns ``value_tp``, each false
    alarm costs ``cost_fp``. Set these from the battery's economics, not from F1."""
    best_t, best_v = 0.5, -np.inf
    for t in np.unique(np.round(p, 3)):
        alert = p >= t
        v = value_tp * np.sum(alert & (y == 1)) - cost_fp * np.sum(alert & (y == 0))
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t
