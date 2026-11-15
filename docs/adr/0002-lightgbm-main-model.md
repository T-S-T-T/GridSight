# 0002 LightGBM as the production model; deep models as challengers

- Status: accepted

## Context
Tabular features (lags, weather, calendar, AEMO pre-dispatch) and a need for quantiles, fast
retraining and explanations.

## Decision
One LightGBM model per quantile (demand, price) and a LightGBM classifier for spikes. N-HiTS
(`models/deep.py`) is an optional challenger and must beat LightGBM on the same folds to be used.

## Consequences
Training takes minutes on a laptop. TreeSHAP explanations come built in (`pred_contrib=True`), so no
`shap` dependency. Prices use a signed-log target so $15,000 spikes do not dominate the loss.
