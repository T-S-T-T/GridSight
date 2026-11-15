"""Configuration loading.

All YAML files live in ``configs/``. The project root is found by walking up from this file
until a directory containing ``configs/settings.yaml`` is found, or taken from the
``GRIDSIGHT_ROOT`` environment variable. ``GRIDSIGHT_DATA_DIR`` and ``GRIDSIGHT_REPORTS_DIR``
override the data and reports folders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    env = os.environ.get("GRIDSIGHT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "settings.yaml").exists():
            return parent
    return Path.cwd()


def load_yaml(name: str) -> dict[str, Any]:
    path = project_root() / "configs" / name
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any] = field(repr=False)
    data_dir: Path
    reports_dir: Path

    @property
    def nemweb(self) -> dict[str, Any]:
        return self.raw["nemweb"]

    @property
    def openmeteo(self) -> dict[str, Any]:
        return self.raw["openmeteo"]

    @property
    def openelectricity(self) -> dict[str, Any]:
        return self.raw["openelectricity"]

    @property
    def spike_threshold(self) -> float:
        return float(self.raw.get("spike_threshold_aud_mwh", 300))

    @property
    def market_floor(self) -> float:
        return float(self.raw.get("market_floor_aud_mwh", -1000))


def _resolve(p: str | os.PathLike[str]) -> Path:
    path = Path(p)
    return path if path.is_absolute() else project_root() / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw = load_yaml("settings.yaml")
    data_dir = _resolve(os.environ.get("GRIDSIGHT_DATA_DIR", raw.get("data_dir", "data")))
    reports_dir = _resolve(os.environ.get("GRIDSIGHT_REPORTS_DIR", raw.get("reports_dir", "reports")))
    return Settings(raw=raw, data_dir=data_dir, reports_dir=reports_dir)


def reset_settings_cache() -> None:
    """Clear the cached settings (used by tests that change environment variables)."""
    get_settings.cache_clear()


def regions_config() -> dict[str, Any]:
    return load_yaml("regions.yaml")["regions"]


def sites_config() -> dict[str, dict[str, float]]:
    return load_yaml("sites.yaml")["sites"]


def battery_config() -> dict[str, Any]:
    return load_yaml("battery.yaml")


def models_config() -> dict[str, Any]:
    return load_yaml("models.yaml")
