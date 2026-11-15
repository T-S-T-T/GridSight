# 0004 Lightweight monitoring checks (instead of Evidently)

- Status: accepted (changed from the design doc)

## Decision
`gridsight.monitoring.checks` implements the planned thresholds directly: data freshness, PSI drift
on numeric features (alert above 0.2), 7-day live error against the test-period error (+25%), and
80% interval coverage (70% to 90%). Failed checks post to a Slack/Discord webhook.

## Consequences
No heavy dependency and fully unit-tested. Evidently can be added later for richer HTML reports.
