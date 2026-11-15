"""Common model interface, target transforms and a file-based model registry."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from gridsight.config import get_settings


class Forecaster(Protocol):
    """Every forecaster returns a frame of quantile columns named like ``q0.5``."""

    name: str
    quantiles: tuple[float, ...]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster: ...

    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...


def qcol(q: float) -> str:
    return f"q{q:g}"


def signed_log(x: np.ndarray | pd.Series) -> np.ndarray:
    """sign(x) * log(1 + |x|): compresses $15,000 spikes without losing negative prices."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def inv_signed_log(z: np.ndarray | pd.Series) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.expm1(np.abs(z))


def sort_quantiles(pred: pd.DataFrame) -> pd.DataFrame:
    """Remove quantile crossing by sorting each row."""
    cols = sorted(pred.columns, key=lambda c: float(c[1:]))
    pred[cols] = np.sort(pred[cols].to_numpy(), axis=1)
    return pred


# ---------------------------------------------------------------- registry


@dataclass
class ModelCard:
    name: str
    version: str
    task: str
    created_utc: str
    features: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    stage: str = "Staging"


def registry_dir() -> Path:
    return get_settings().data_dir / "models"


def save_model(model: Any, card: ModelCard) -> Path:
    d = registry_dir() / card.name
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{card.version}.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    (d / f"{card.version}.json").write_text(json.dumps(card.__dict__, indent=2, default=str), encoding="utf-8")
    return d / f"{card.version}.pkl"


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S%f")


def list_versions(name: str) -> list[ModelCard]:
    d = registry_dir() / name
    if not d.exists():
        return []
    cards = [ModelCard(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(d.glob("*.json"))]
    return sorted(cards, key=lambda c: c.version)


def set_stage(name: str, version: str, stage: str) -> None:
    """Promote a version. Only one version per model can be in Production."""
    d = registry_dir() / name
    for card in list_versions(name):
        if card.version == version:
            card.stage = stage
        elif stage == "Production" and card.stage == "Production":
            card.stage = "Archived"
        (d / f"{card.version}.json").write_text(json.dumps(card.__dict__, indent=2, default=str), encoding="utf-8")


def load_model(name: str, stage: str | None = "Production", version: str | None = None) -> tuple[Any, ModelCard]:
    """Load a version by id, the latest version in ``stage``, or (stage=None) the latest of any."""
    cards = list_versions(name)
    if version is not None:
        cards = [c for c in cards if c.version == version]
    elif stage is not None:
        cards = [c for c in cards if c.stage == stage]
    if not cards:
        raise FileNotFoundError(f"no {stage} version of model {name!r}")
    card = cards[-1]
    with (registry_dir() / name / f"{card.version}.pkl").open("rb") as fh:
        return pickle.load(fh), card  # noqa: S301 - files written by this project only
