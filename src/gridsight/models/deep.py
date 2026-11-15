"""Optional deep-learning challenger (N-HiTS via neuralforecast).

Install with ``pip install -e ".[deep]"``. Kept out of the core pipeline and the test suite
because it needs PyTorch; GridSight's production models are the LightGBM ones.
"""

from __future__ import annotations

import pandas as pd


def available() -> bool:
    try:
        import neuralforecast  # noqa: F401
    except ImportError:
        return False
    return True


def fit_predict_nhits(hh: pd.DataFrame, target: str = "demand", horizon: int = 48, input_size: int = 336) -> pd.DataFrame:
    """Fit N-HiTS on a half-hourly frame (region, interval_utc, <target>) and forecast ``horizon``
    steps past the end of each region's history. Returns region, interval_utc, y_hat."""
    if not available():
        raise ImportError('neuralforecast is not installed: pip install -e ".[deep]"')
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    df = hh.rename(columns={"region": "unique_id", "interval_utc": "ds", target: "y"})[["unique_id", "ds", "y"]]
    df["ds"] = df["ds"].dt.tz_convert(None)
    nf = NeuralForecast(models=[NHITS(h=horizon, input_size=input_size, max_steps=500)], freq="30min")
    nf.fit(df.dropna())
    out = nf.predict().reset_index()
    return out.rename(columns={"unique_id": "region", "ds": "interval_utc", "NHITS": "y_hat"})
