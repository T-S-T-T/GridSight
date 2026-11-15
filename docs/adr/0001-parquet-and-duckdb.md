# 0001 Use partitioned Parquet + DuckDB instead of a database or Spark

- Status: accepted

## Context
About 50 GB of NEM time series (more with unit-level data), one developer, a laptop and a small VM.

## Decision
Bronze/silver/gold layers stored as Parquet, partitioned by region (or site) and month, written
idempotently by `gridsight.storage`. DuckDB for ad-hoc SQL.

## Alternatives considered
Postgres/TimescaleDB for everything; Spark on Databricks; BigQuery.

## Consequences
No server to run; files are portable and cheap to back up. Monthly partitions make re-running a job
rewrite only the affected files. Beyond about 1 TB, or with many concurrent writers, revisit.
