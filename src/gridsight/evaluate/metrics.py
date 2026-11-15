"""Forecast metrics and the Diebold-Mariano test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss


def _clean(*arrays: np.ndarray | pd.Series) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    ok = np.logical_and.reduce([np.isfinite(a) for a in arrs])
    return [a[ok] for a in arrs]


def mae(y, yhat) -> float:
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat))) if len(y) else np.nan


def rmse(y, yhat) -> float:
    y, yhat = _clean(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else np.nan


def mape(y, yhat, eps: float = 1.0) -> float:
    """Mean absolute percentage error (demand). ``eps`` guards against near-zero actuals."""
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat) / np.maximum(np.abs(y), eps))) if len(y) else np.nan


def pinball(y, yhat, q: float) -> float:
    y, yhat = _clean(y, yhat)
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1) * d))) if len(y) else np.nan


def mean_pinball(y, pred: pd.DataFrame) -> float:
    """Average pinball loss over all quantile columns; x2 approximates the CRPS."""
    return float(np.mean([pinball(y, pred[c], float(c[1:])) for c in pred.columns if c.startswith("q")]))


def coverage(y, lo, hi) -> float:
    y, lo, hi = _clean(y, lo, hi)
    return float(np.mean((y >= lo) & (y <= hi))) if len(y) else np.nan


def interval_width(lo, hi) -> float:
    lo, hi = _clean(lo, hi)
    return float(np.mean(hi - lo)) if len(lo) else np.nan


def pr_auc(y, p) -> float:
    y, p = _clean(y, p)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y.astype(int), p))


def brier(y, p) -> float:
    y, p = _clean(y, p)
    return float(brier_score_loss(y.astype(int), p)) if len(y) else np.nan


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed frequency per probability bin."""
    y, p = _clean(y, p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    return df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def diebold_mariano(e1, e2, h: int = 1, power: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    ``e1``, ``e2`` are forecast errors of models 1 and 2 on the same targets. A negative statistic
    means model 1 has lower loss. Returns (statistic, two-sided p-value)."""
    e1, e2 = _clean(e1, e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    d_mean = d.mean()
    gamma = [np.mean((d[k:] - d_mean) * (d[: n - k] - d_mean)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d_mean / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return stat, p
