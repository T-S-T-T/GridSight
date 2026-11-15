"""Conformalized quantile regression (CQR, Romano et al. 2019).

Takes a model's lower and upper quantile predictions and widens (or narrows) them by a constant
learned on a calibration set, so the interval covers ``1 - alpha`` of outcomes on new data from the
same distribution, whatever the underlying model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CQR:
    def __init__(self, alpha: float = 0.2, lower: str = "q0.1", upper: str = "q0.9"):
        self.alpha = alpha
        self.lower, self.upper = lower, upper
        self.q_hat: float | None = None

    def fit(self, pred: pd.DataFrame, y: pd.Series | np.ndarray) -> CQR:
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        lo, hi = pred[self.lower].to_numpy()[ok], pred[self.upper].to_numpy()[ok]
        scores = np.maximum(lo - y[ok], y[ok] - hi)
        n = len(scores)
        if n == 0:
            raise ValueError("no calibration data")
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self

    def transform(self, pred: pd.DataFrame) -> pd.DataFrame:
        assert self.q_hat is not None, "fit first"
        out = pred.copy()
        out[self.lower] = pred[self.lower] - self.q_hat
        out[self.upper] = pred[self.upper] + self.q_hat
        mid = [c for c in out.columns if c not in (self.lower, self.upper)]
        for c in mid:  # keep the median inside the interval
            out[c] = out[c].clip(out[self.lower], out[self.upper])
        return out
