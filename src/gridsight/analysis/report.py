"""Build the EDA report: CSV tables plus a markdown summary in ``reports/eda/``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridsight import storage
from gridsight.analysis import market
from gridsight.config import get_settings


def _md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_eda_report(out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    s = get_settings()
    out_dir = out_dir or s.reports_dir / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    price = storage.read("dispatch_price")
    demand = storage.read("region_demand")
    pre = storage.read("predispatch")
    if price.empty or demand.empty:
        raise RuntimeError("no silver data: run `gridsight synth` or `gridsight backfill` first")

    t = s.spike_threshold
    tables = {
        "market_summary": market.market_summary(price, demand, t),
        "spike_stats": market.spike_stats(price, t),
        "negative_prices_by_year": market.negative_prices(price, "year"),
        "duck_depth_by_year": market.duck_depth(demand, "year"),
        "spikes_by_hour": market.spikes_by_hour(price, t),
    }
    if not pre.empty:
        tables["predispatch_accuracy"] = market.predispatch_accuracy(pre, price, demand, t)

    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    md = [
        "# GridSight market story (generated)",
        "",
        f"Data: {price['interval_utc'].min():%Y-%m-%d} to {price['interval_utc'].max():%Y-%m-%d}, "
        f"{price['region'].nunique()} regions, {len(price):,} dispatch intervals.",
        "",
        "## Market summary",
        _md_table(tables["market_summary"]),
        "",
        f"## Price spikes (above ${t:,.0f}/MWh)",
        _md_table(tables["spike_stats"]),
    ]
    if "predispatch_accuracy" in tables:
        acc = tables["predispatch_accuracy"]
        md += ["", "## AEMO pre-dispatch accuracy by horizon", _md_table(acc.head(40))]
    (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return tables
