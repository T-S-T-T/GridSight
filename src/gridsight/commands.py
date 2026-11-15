"""CLI commands. Each phase adds its commands here; heavy imports stay inside the handlers."""

from __future__ import annotations

import argparse

from gridsight.cli import command

# ------------------------------------------------------------------ Phase 1: data platform


def _synth_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--regions", default="NSW1,QLD1,VIC1,SA1,TAS1")
    p.add_argument("--seed", type=int, default=42)


@command("synth", "Write synthetic NEM data to silver (offline)", _synth_args)
def _synth(args: argparse.Namespace) -> int:
    from gridsight.pipeline import load_synthetic
    from gridsight.synthetic import SyntheticConfig

    cfg = SyntheticConfig(start=args.start, days=args.days, regions=tuple(args.regions.split(",")), seed=args.seed)
    print(load_synthetic(cfg))
    return 0


def _backfill_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", required=True, help="first month, YYYY-MM")
    p.add_argument("--end", required=True, help="last month, YYYY-MM")
    p.add_argument("--tables", default=None, help="comma-separated MMS tables (default: core set)")


@command("backfill", "Download MMSDM monthly archives from NEMweb into silver", _backfill_args)
def _backfill(args: argparse.Namespace) -> int:
    from gridsight.pipeline import BACKFILL_TABLES, backfill_mmsdm

    tables = tuple(args.tables.split(",")) if args.tables else BACKFILL_TABLES
    print(backfill_mmsdm(args.start, args.end, tables))
    return 0


@command("ingest-live", "Fetch the latest NEMweb CURRENT reports into silver")
def _ingest_live(args: argparse.Namespace) -> int:
    from gridsight.pipeline import ingest_live

    print(ingest_live())
    return 0


def _weather_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", help="YYYY-MM-DD (historical kinds)")
    p.add_argument("--end", help="YYYY-MM-DD (historical kinds)")
    p.add_argument("--kind", default="historical_forecast", choices=["historical_forecast", "archive", "forecast"])


@command("weather", "Fetch Open-Meteo weather for every site into silver", _weather_args)
def _weather(args: argparse.Namespace) -> int:
    from gridsight.pipeline import ingest_weather

    print(ingest_weather(args.start, args.end, kind=args.kind))
    return 0


@command("quality", "Print coverage, gaps and freshness of silver tables")
def _quality(args: argparse.Namespace) -> int:
    import pandas as pd

    from gridsight.pipeline import quality_report

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(quality_report().to_string(index=False))
    return 0


# ------------------------------------------------------------------ Phase 2: market story


@command("eda", "Write the EDA tables and summary to reports/eda/")
def _eda(args: argparse.Namespace) -> int:
    from gridsight.analysis.report import build_eda_report
    from gridsight.config import get_settings

    tables = build_eda_report()
    print(f"wrote {len(tables)} tables to {get_settings().reports_dir / 'eda'}")
    return 0


# ------------------------------------------------------------------ Phase 3: features, models, evaluation


def _features_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--issue-every", default="30min", help="issue-time spacing, e.g. 30min or 2h (smaller data)")


@command("features", "Build the gold feature table from silver", _features_args)
def _features(args: argparse.Namespace) -> int:
    from gridsight.evaluate.run import build_and_store_features

    feats = build_and_store_features(issue_every=args.issue_every)
    print(f"built {len(feats):,} feature rows x {feats.shape[1]} columns")
    return 0


def _tasks_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tasks", default="demand,price,spike")
    p.add_argument("--n-estimators", type=int, default=None, help="override LightGBM trees (faster runs)")


@command("evaluate", "Walk-forward validation + locked test vs AEMO; writes reports/results/", _tasks_args)
def _evaluate(args: argparse.Namespace) -> int:
    from gridsight.evaluate.run import evaluate_all

    params = {"n_estimators": args.n_estimators} if args.n_estimators else None
    results = evaluate_all(tasks=tuple(args.tasks.split(",")), gbm_params=params)
    for name, r in results.items():
        print(f"\n== {name} (locked test) ==")
        print(r.test.to_string(index=False))
        if not r.vs_aemo.empty:
            print(r.vs_aemo.to_string(index=False))
    return 0


@command("train", "Train production models on all labelled data and register them", _tasks_args)
def _train(args: argparse.Namespace) -> int:
    from gridsight.evaluate.run import train_production

    params = {"n_estimators": args.n_estimators} if args.n_estimators else None
    for task, card in train_production(tasks=tuple(args.tasks.split(",")), gbm_params=params).items():
        print(f"{task}: {card.name} {card.version} [{card.stage}]")
    return 0


# ------------------------------------------------------------------ Phase 4: battery + explainability


def _battery_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--region", default=None, help="default: configs/battery.yaml")
    p.add_argument("--days", type=int, default=None, help="only simulate the last N days of the test period")
    p.add_argument("--horizon-hours", type=int, default=None)


@command("battery", "Backtest battery strategies on the test-period forecasts", _battery_args)
def _battery(args: argparse.Namespace) -> int:
    from gridsight.battery.job import run_battery_backtest

    overrides = {"horizon_hours": args.horizon_hours} if args.horizon_hours else None
    summary = run_battery_backtest(args.region, args.days, overrides)
    print(summary.round(2).to_string(index=False))
    return 0


# ------------------------------------------------------------------ Phase 5: production


@command("score", "Score the Production models on the latest data and store predictions")
def _score(args: argparse.Namespace) -> int:
    from gridsight.serving import score_latest

    pred = score_latest()
    print(f"stored {len(pred)} predictions issued {pred['issued_utc'].max()}")
    return 0


def _monitor_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-lag", default="15min", help="data older than this counts as stale")


@command("monitor", "Run freshness, drift, live-accuracy and coverage checks", _monitor_args)
def _monitor(args: argparse.Namespace) -> int:
    import pandas as pd

    from gridsight.monitoring.job import run_monitoring

    report = run_monitoring(max_lag=args.max_lag)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(report[["name", "ok", "value", "threshold", "detail"]].to_string(index=False))
    return 0 if report["ok"].all() else 1


def _retrain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--weeks", type=int, default=8, help="recent window used to compare challenger vs champion")
    p.add_argument("--n-estimators", type=int, default=None)


@command("retrain", "Train challengers and promote them only if they beat Production", _retrain_args)
def _retrain(args: argparse.Namespace) -> int:
    from gridsight.monitoring.job import retrain

    params = {"n_estimators": args.n_estimators} if args.n_estimators else None
    for d in retrain(weeks=args.weeks, gbm_params=params):
        print(d)
    return 0


def _rollback_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("task", choices=["demand", "price", "spike"])


@command("rollback", "Restore the previous Production model for a task", _rollback_args)
def _rollback(args: argparse.Namespace) -> int:
    from gridsight.serving import rollback

    print(f"{args.task}: Production is now {rollback(args.task)}")
    return 0


def _loop_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--every", type=int, default=300, help="seconds between runs")
    p.add_argument("--iterations", type=int, default=0, help="stop after N runs (0 = forever)")


@command("loop", "Ingest + score on a timer (simple scheduler, no Prefect needed)", _loop_args)
def _loop(args: argparse.Namespace) -> int:
    import logging
    import time

    from gridsight.pipeline import ingest_live
    from gridsight.serving import score_latest

    log = logging.getLogger("gridsight.loop")
    n = 0
    while True:
        started = time.monotonic()
        try:
            log.info("%s", ingest_live())
            score_latest()
        except Exception:  # keep the loop alive; the error is logged and monitoring will alert
            log.exception("loop iteration failed")
        n += 1
        if args.iterations and n >= args.iterations:
            return 0
        time.sleep(max(args.every - (time.monotonic() - started), 1))


def _serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)


@command("serve", "Run the REST API (docs at /docs)", _serve_args)
def _serve(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    import uvicorn

    uvicorn.run("gridsight.api.app:app", host=args.host, port=args.port)
    return 0


def _demo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--days", type=int, default=90, help="days of synthetic data")
    p.add_argument("--regions", default="NSW1,SA1")
    p.add_argument("--n-estimators", type=int, default=80)


@command("demo", "Run the whole pipeline offline on synthetic data (end-to-end smoke test)", _demo_args)
def _demo(args: argparse.Namespace) -> int:
    from gridsight.analysis.report import build_eda_report
    from gridsight.battery.job import run_battery_backtest
    from gridsight.evaluate.run import build_and_store_features, evaluate_all, train_production
    from gridsight.monitoring.job import run_monitoring
    from gridsight.pipeline import load_synthetic
    from gridsight.serving import score_latest
    from gridsight.synthetic import SyntheticConfig

    params = {"n_estimators": args.n_estimators}
    regions = tuple(args.regions.split(","))
    steps = [
        ("synthetic data", lambda: load_synthetic(SyntheticConfig(days=args.days, regions=regions, predispatch_horizon_steps=16))),
        ("EDA report", build_eda_report),
        ("features", lambda: f"{len(build_and_store_features(issue_every='1h')):,} rows"),
        ("evaluate", lambda: ", ".join(evaluate_all(gbm_params=params))),
        ("train", lambda: ", ".join(f"{c.name}[{c.stage}]" for c in train_production(gbm_params=params).values())),
        ("battery", lambda: run_battery_backtest(regions[-1], days=7, overrides={"horizon_hours": 12})[["strategy", "revenue_aud"]].round(0).to_dict("records")),
        ("score", lambda: f"{len(score_latest())} predictions"),
        ("monitor", lambda: run_monitoring(max_lag="3650D")[["name", "ok"]].to_dict("records")),
    ]
    for name, fn in steps:
        result = fn()
        print(f"[demo] {name}: {result if not hasattr(result, 'shape') else 'ok'}")
    print("[demo] done: every stage ran end to end")
    return 0
