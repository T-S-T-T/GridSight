"""Phase 6: documentation stays consistent with the code."""

from __future__ import annotations

import re

from gridsight.cli import _COMMANDS, build_parser
from gridsight.config import project_root

ROOT = project_root()


def test_every_cli_command_is_documented_and_every_documented_command_exists():
    build_parser()  # registers all commands
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`gridsight ([a-z-]+)", readme)) | set(re.findall(r"^gridsight ([a-z-]+)", readme, re.M))
    assert documented <= set(_COMMANDS), f"README mentions unknown commands: {documented - set(_COMMANDS)}"
    assert set(_COMMANDS) <= documented, f"commands missing from README: {set(_COMMANDS) - documented}"


def test_docs_exist():
    for path in [
        "docs/adr/0001-parquet-and-duckdb.md",
        "docs/model_cards/demand_lgbm.md",
        "docs/model_cards/price_lgbm.md",
        "docs/model_cards/spike_lgbm.md",
        "tooling/README.md",
        "tooling/ci.yml.txt",
        "notebooks/01_market_story.ipynb",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
    ]:
        assert (ROOT / path).exists(), path


def test_ci_runs_tests_and_demo():
    ci = (ROOT / "tooling" / "ci.yml.txt").read_text(encoding="utf-8")
    assert "pytest" in ci and "gridsight demo" in ci and "windows-latest" in ci
