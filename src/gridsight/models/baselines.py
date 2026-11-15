"""Baselines every model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridsight.models.base import qcol


class SeasonalNaive:
    """Same half-hour yesterday (or last week, when yesterday is not yet known)."""

    def __init__(self, target: str = "demand", quantiles: tuple[float, ...] = (0.5,)):
        self.target = target
        self.name = f"seasonal_naive_{target}"
        self.quantiles = quantiles
        self._spread: dict[float, float] = {}

    def _point(self, X: pd.DataFrame) -> np.ndarray:
        day, week = X[f"{self.target}_lag_day"], X[f"{self.target}_lag_week"]
        return day.fillna(week).fillna(X[f"{self.target}_last"]).to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        resid = y.to_numpy(dtype=float) - self._point(X)
        resid = resid[np.isfinite(resid)]
        self._spread = {q: float(np.quantile(resid, q)) if len(resid) else 0.0 for q in self.quantiles}
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self._point(X)
        return pd.DataFrame({qcol(q): p + self._spread.get(q, 0.0) for q in self.quantiles}, index=X.index)


class Climatology:
    """Training-set mean for each (region, half-hour of week)."""

    def __init__(self, quantiles: tuple[float, ...] = (0.5,)):
        self.name = "climatology"
        self.quantiles = quantiles
        self._table: pd.DataFrame | None = None

    @staticmethod
    def _key(X: pd.DataFrame) -> pd.Series:
        return X["region_code"].astype(int) * 1000 + X["dow"].astype(int) * 48 + (X["tod"] * 2).round().astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Climatology:
        df = pd.DataFrame({"k": self._key(X).to_numpy(), "y": y.to_numpy(dtype=float)}).dropna()
        self._table = df.groupby("k")["y"].quantile(list(self.quantiles)).unstack()
        self._fallback = df["y"].quantile(list(self.quantiles))
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._table is not None, "fit first"
        t = self._table.reindex(self._key(X).to_numpy())
        out = pd.DataFrame(index=X.index)
        for q in self.quantiles:
            out[qcol(q)] = t[q].fillna(self._fallback[q]).to_numpy()
        return out


class PredispatchBaseline:
    """AEMO's own pre-dispatch forecast (the benchmark). No fitting needed."""

    def __init__(self, target: str = "price", quantiles: tuple[float, ...] = (0.5,)):
        self.target = target
        self.name = f"aemo_predispatch_{target}"
        self.quantiles = quantiles

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PredispatchBaseline:
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        col = "pd_price" if self.target == "price" else "pd_demand"
        p = X[col].to_numpy(dtype=float) if col in X else np.full(len(X), np.nan)
        return pd.DataFrame({qcol(q): p for q in self.quantiles}, index=X.index)
