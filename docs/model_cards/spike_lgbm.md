# Model card: spike_lgbm

- Task: Probability that any 5-minute price in the half-hour exceeds $300/MWh, 30 to 60 minutes ahead (Platt or isotonic calibration)
- Version / promoted / git SHA: _from `data/models/spike_lgbm/<version>.json`_
- Training data: _period, regions, rows_ (excludes intervention and market-suspension intervals)
- Features: calendar, horizon-aware lags, rolling statistics, regional weather forecasts, AEMO pre-dispatch
- Validation: expanding-window walk-forward folds with a 7-day gap; locked test period
- Results: _copy the table from `reports/results/results.md`_
- Top drivers (mean |SHAP|): _from the dashboard Explain page_
- Known weaknesses: _fill in after evaluation on real data_
- Intended use: portfolio demonstration and research. Not for real trading.
- Monitoring: PSI drift, 7-day live error vs test error, interval coverage; weekly retrain with champion/challenger promotion
