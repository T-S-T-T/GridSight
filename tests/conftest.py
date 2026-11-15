"""Shared test fixtures.

Every test runs against a temporary data directory, so the suite never touches real data.
Tests that need the internet are marked ``network`` and skipped unless
``GRIDSIGHT_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from gridsight.config import reset_settings_cache
from gridsight.synthetic import SyntheticConfig, generate

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("GRIDSIGHT_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test: set GRIDSIGHT_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("GRIDSIGHT_DATA_DIR", str(data))
    monkeypatch.setenv("GRIDSIGHT_REPORTS_DIR", str(tmp_path / "reports"))
    reset_settings_cache()
    yield data
    reset_settings_cache()


@pytest.fixture(scope="session")
def synthetic() -> dict[str, pd.DataFrame]:
    """Small synthetic dataset: 2 regions x 40 days."""
    return generate(SyntheticConfig(start="2024-01-01", days=40, regions=("NSW1", "SA1"), seed=7))


@pytest.fixture(scope="session")
def synthetic_long() -> dict[str, pd.DataFrame]:
    """Longer synthetic dataset for model and backtest tests: 1 region x 150 days."""
    return generate(
        SyntheticConfig(start="2023-09-01", days=150, regions=("SA1",), predispatch_horizon_steps=8, seed=11)
    )
