"""Prefect deployments for every scheduled job (install with ``pip install -e ".[orchestration]"``).

Start them all with ``python -m gridsight.flows.prefect_flows`` (keeps running and schedules:
ingest + score every 5 min, pre-dispatch every 30 min, weather hourly, monitoring daily,
retraining weekly). Without Prefect, ``gridsight loop`` runs the 5-minute jobs in a simple loop.
"""

from __future__ import annotations

from prefect import flow, serve, task

from gridsight import pipeline, serving
from gridsight.monitoring.job import retrain, run_monitoring


@task(retries=2, retry_delay_seconds=30)
def _ingest() -> str:
    return str(pipeline.ingest_live())


@task(retries=1, retry_delay_seconds=30)
def _score() -> int:
    return len(serving.score_latest())


@flow(name="ingest-and-score")
def ingest_and_score() -> None:
    _ingest()
    _score()


@flow(name="weather-forecast")
def weather_forecast() -> None:
    pipeline.ingest_weather(None, None, kind="forecast")


@flow(name="monitoring")
def monitoring() -> None:
    run_monitoring()


@flow(name="retrain")
def weekly_retrain() -> None:
    from gridsight.evaluate.run import build_and_store_features

    build_and_store_features()
    retrain()


if __name__ == "__main__":  # pragma: no cover
    serve(
        ingest_and_score.to_deployment("every-5-min", cron="*/5 * * * *"),
        weather_forecast.to_deployment("hourly", cron="7 * * * *"),
        monitoring.to_deployment("daily", cron="13 21 * * *"),
        weekly_retrain.to_deployment("weekly", cron="41 16 * * 6"),
    )
