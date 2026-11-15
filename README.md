# GridSight

**Forecasting demand, prices and price spikes in Australia's National Electricity Market, benchmarked against AEMO's own pre-dispatch forecasts, and valued in dollars with a simulated grid battery.**

Live dashboard: _add link_ · Technical report: _add link_ · Demo video: _add link_

GridSight ingests public AEMO NEMweb data every 5 minutes, builds leakage-safe features, forecasts
demand and prices with calibrated prediction intervals, predicts price spikes, and tests a battery
trading strategy against those forecasts. Everything runs from one CLI, with 110+ automated tests.

## Results (locked test period)

Fill these in from `reports/results/results.md` after running on real data (`gridsight evaluate`).

| Task | Metric | GridSight | AEMO pre-dispatch | Improvement |
| --- | --- | --- | --- | --- |
| Demand, 24h ahead | MAPE | _x_% | _y_% | _z_% |
| Price, 1h ahead | MAE ($/MWh) | _x_ | _y_ | _z_% |
| Spike, 1h ahead | PR-AUC | _x_ | _y_ | |
| Battery, SA1 | % of perfect foresight | _x_% | _y_% | |

## Architecture

```
Sources            Pipeline (5-min schedule)           Models                         Serving
AEMO NEMweb  ─┐    bronze: raw zips/JSON               LightGBM quantile (demand,     Postgres/Parquet
Open-Meteo   ─┼──► silver: typed Parquet, UTC,   ──►   price) + conformal intervals ─► predictions table
CER, ABS     ─┘    pandera-validated                   LightGBM spike classifier      ├─ FastAPI (read-only)
                   gold: features by issue time        Battery LP (HiGHS)             ├─ Streamlit dashboard
                                                       Registry: Staging/Production   └─ Monitoring + alerts
```

Design notes: [`docs/adr/`](docs/adr). Model cards: [`docs/model_cards/`](docs/model_cards).

## Quickstart (Windows / Anaconda)

```powershell
cd F:\Proj\Python\GridSight
conda create -n gridsight python=3.12 -y
conda activate gridsight
pip install -e ".[dev,dashboard,eda]"

pytest                       # 110+ tests, about 1 minute, fully offline
gridsight demo               # whole pipeline on synthetic data: EDA -> features -> evaluate -> train -> battery -> score -> monitor
streamlit run dashboard/app.py
gridsight serve              # API at http://127.0.0.1:8000/docs
```

macOS/Linux: the same commands work; `tooling/Makefile.txt` has shortcuts (see `tooling/README.md`).

## Running on real AEMO data

```powershell
gridsight backfill --start 2021-10 --end 2026-10      # MMSDM monthly archives -> silver (large: start with a few months)
gridsight weather --start 2021-10-01 --end 2026-10-31 # Open-Meteo archived forecasts for every site
gridsight quality                                     # coverage, gaps, freshness
gridsight eda                                         # reports/eda/
gridsight features                                    # gold feature table
gridsight evaluate                                    # walk-forward + locked test vs AEMO -> reports/results/
gridsight train                                       # register Production models
gridsight battery                                     # 5 strategies -> reports/battery/
gridsight ingest-live                                 # latest 5-min files
gridsight score                                       # live predictions
gridsight monitor                                     # freshness, drift, accuracy, coverage
```

Pre-dispatch history for the AEMO benchmark: install the `aemo` extra and load it with NEMSEER, or
let `ingest-live` collect it going forward.

Keep it running: `gridsight loop` (ingest + score every 5 minutes, no extra software), or Prefect:
`pip install -e ".[orchestration]"` then `python -m gridsight.flows.prefect_flows`. Docker:
`docker compose up` runs the API, dashboard and worker.

## Commands

| Command | What it does |
| --- | --- |
| `gridsight info` | Resolved paths and settings |
| `gridsight synth` | Write synthetic silver data (offline) |
| `gridsight backfill` | MMSDM monthly archive -> silver |
| `gridsight ingest-live` | NEMweb CURRENT reports -> silver |
| `gridsight weather` | Open-Meteo forecasts for every site |
| `gridsight quality` | Coverage, gaps and freshness |
| `gridsight eda` | Market-story tables in `reports/eda/` |
| `gridsight features` | Build the gold feature table |
| `gridsight evaluate` | Walk-forward validation, locked test, AEMO comparison |
| `gridsight train` | Train and register Production models |
| `gridsight battery` | Battery strategy backtest |
| `gridsight score` | Score the latest data |
| `gridsight monitor` | Monitoring checks and alerts |
| `gridsight retrain` | Train challengers, promote only if better |
| `gridsight rollback` | Restore the previous Production model |
| `gridsight loop` | Ingest + score on a timer |
| `gridsight serve` | REST API |
| `gridsight demo` | Everything, offline, end to end |

## Testing

`pytest` runs 110+ tests offline in about a minute (network tests are skipped unless
`GRIDSIGHT_NETWORK_TESTS=1`). Highlights:

- **Leakage test**: scrambles every value after a cut-off and asserts that no feature issued
  before it changes (`tests/unit/test_phase3_models.py::test_features_do_not_use_the_future`).
- **AEMO format**: parser and transforms tested on fixture files in AEMO's multi-table CSV format,
  including nested ZIP archives, intervention pricing and both MMSDM file-naming schemes.
- **Idempotent storage**: re-running any job never duplicates rows.
- **Models must beat baselines**: tests fail if LightGBM does not beat seasonal-naive.
- **Battery LP**: buys low, sells high, respects power and state-of-charge limits, values stored energy.
- **Production**: API, dashboard pages (Streamlit AppTest), promotion/rollback and the full `gridsight demo`.

Coverage is about 95%. CI config for GitHub Actions (Ubuntu + Windows) is in `tooling/ci.yml.txt`.

## Project layout

```
configs/            settings, regions + weather sites, battery, model settings
src/gridsight/
  ingest/           NEMweb, MMSDM, Open-Meteo, CER, holidays, AEMO CSV parser
  transform/        silver transforms, pandera schemas, data-quality checks
  features/         leakage-safe feature builder
  models/           baselines, LightGBM quantile + spike, conformal (CQR), registry
  evaluate/         metrics, Diebold-Mariano, walk-forward backtest, jobs
  battery/          LP (HiGHS), rolling simulation, strategies
  analysis/         market story (EDA), SHAP explanations
  monitoring/       freshness, PSI drift, live accuracy, coverage, alerts
  api/              FastAPI app
  flows/            Prefect deployments
dashboard/          Streamlit app
notebooks/          01_market_story.ipynb
tests/              unit + integration tests, AEMO-format fixtures
docs/               ADRs, model cards
tooling/            CI, pre-commit and Makefile (copy into place)
```

## Data and licences

AEMO data under the [AEMO copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
Open Electricity data under CC BY-NC 4.0. Weather data by [Open-Meteo.com](https://open-meteo.com/) under
CC BY 4.0. Clean Energy Regulator and ABS data under CC BY. GridSight is a portfolio research project;
it is not for real trading.
