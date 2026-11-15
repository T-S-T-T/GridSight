"""Phase 2: market analysis functions, the EDA report and the notebook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridsight import storage
from gridsight.analysis import market
from gridsight.analysis.report import build_eda_report
from gridsight.cli import main
from gridsight.config import get_settings, project_root
from gridsight.pipeline import load_synthetic
from gridsight.synthetic import SyntheticConfig


def test_duck_curve_has_midday_dip(synthetic):
    depth = market.duck_depth(synthetic["region_demand"])
    assert (depth["depth_mw"] > 0).all()
    curve = market.duck_curve(synthetic["region_demand"], by="season")
    assert curve["hour"].between(0, 23.5).all()
    assert curve["hour"].nunique() == 48


def test_negative_prices_and_spikes(synthetic):
    neg = market.negative_prices(synthetic["dispatch_price"])
    assert neg["share"].between(0, 1).all()
    assert np.isclose(neg["hours"], neg["intervals"] * 5 / 60).all()
    spikes = market.spike_stats(synthetic["dispatch_price"], 300)
    assert (spikes["spikes"] > 0).all()
    assert (spikes["mean_spike_price"] > 300).all()
    assert spikes["spike_revenue_share"].between(0, 1).all()
    by_hour = market.spikes_by_hour(synthetic["dispatch_price"])
    assert set(by_hour["hour"]) <= set(range(24))


def test_known_values_on_tiny_frame():
    idx = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
    price = pd.DataFrame({"interval_utc": idx, "region": "SA1", "rrp": [-10.0, 50.0, 400.0, 1000.0]})
    s = market.spike_stats(price, 300).iloc[0]
    assert s["spikes"] == 2 and s["mean_spike_price"] == 700.0
    assert s["spike_revenue_share"] == pytest.approx(1400 / 1450)
    n = market.negative_prices(price).iloc[0]
    assert n["share"] == 0.25
    pdc = market.price_duration_curve(price, points=3)
    assert pdc["rrp"].iloc[0] == 1000.0 and pdc["rrp"].iloc[-1] == -10.0


def test_to_half_hour_averages_six_intervals():
    idx = pd.date_range("2024-01-01", periods=12, freq="5min", tz="UTC")
    df = pd.DataFrame({"interval_utc": idx, "region": "SA1", "rrp": np.arange(12.0)})
    hh = market.to_half_hour(df, ["rrp"])
    assert hh["rrp"].tolist() == [2.5, 8.5]


def test_predispatch_accuracy_grows_with_horizon(synthetic):
    acc = market.predispatch_accuracy(synthetic["predispatch"], synthetic["dispatch_price"], synthetic["region_demand"])
    assert acc["horizon_steps"].min() >= 1
    nsw = acc[acc["region"] == "NSW1"].set_index("horizon_steps")
    assert nsw["demand_mape"].iloc[-1] > nsw["demand_mape"].iloc[0]
    assert acc["spike_recall"].dropna().between(0, 1).all()


def test_eda_report_writes_files():
    load_synthetic(SyntheticConfig(days=10, regions=("SA1", "VIC1")))
    tables = build_eda_report()
    out = get_settings().reports_dir / "eda"
    assert "reports" in str(out) and str(get_settings().data_dir.parent) in str(out)
    assert (out / "README.md").exists()
    assert "predispatch_accuracy" in tables
    for name in tables:
        assert (out / f"{name}.csv").exists()


def test_eda_report_requires_data():
    with pytest.raises(RuntimeError):
        build_eda_report(out_dir=Path(get_settings().data_dir) / "eda")


def test_cli_eda(capsys):
    main(["synth", "--days", "3", "--regions", "SA1"])
    assert main(["eda"]) == 0
    assert "wrote" in capsys.readouterr().out


NOTEBOOK = project_root() / "notebooks" / "01_market_story.ipynb"


def test_notebook_is_valid_and_has_no_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(not c["outputs"] for c in code)
    for c in code:
        compile("".join(c["source"]), NOTEBOOK.name, "exec")


def test_notebook_runs_end_to_end():
    """Execute every code cell on synthetic data with a non-interactive backend."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    load_synthetic(SyntheticConfig(days=8, regions=("SA1", "NSW1")))
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: dict = {"display": lambda *a, **k: None}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), NOTEBOOK.name, "exec"), ns)
    plt.close("all")
    assert not ns["price"].empty
    assert storage.read("dispatch_price")["region"].nunique() == 2
