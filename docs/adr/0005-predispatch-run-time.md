# 0005 Pre-dispatch run time = LASTCHANGED

- Status: accepted

## Context
The AEMO benchmark must use only pre-dispatch runs that were public at GridSight's issue time.

## Decision
A pre-dispatch run's time is its `LASTCHANGED` timestamp (floored to 5 minutes), i.e. when it was
published. Features join the latest run with `run_utc <= issued_utc`.

## Consequences
Conservative: no look-ahead into a run before it was published. If NEMSEER's run times are used
instead, check they also mean "published at".
