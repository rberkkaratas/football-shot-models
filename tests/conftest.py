"""Shared fixtures for the M1.0 / M1.1 tests: one synthetic snapshot and one ``run`` per session.

Builders live in ``xg_cases``. Imports of the package under test are inside the fixtures, so
a missing module fails the tests that need it, not the whole collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
import xg_cases as xc


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fixture")
    xc.write_fixture_snapshot(root)
    return root


@pytest.fixture(scope="session")
def manifest(fixture_root: Path) -> Path:
    return fixture_root / "manifests" / f"{xc.SNAPSHOT}.json"


@pytest.fixture(scope="session")
def xg_data(manifest: Path) -> Any:
    from football_shot_models import data

    return data.load_xg_data(manifest)


@pytest.fixture(scope="session")
def dev_held(xg_data: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    from football_shot_models import data

    dev, held = data.split_holdout(xg_data.shots)
    return dev, held


@pytest.fixture(scope="session")
def run_result(manifest: Path, fixture_root: Path) -> tuple[tuple[Any, ...], Path]:
    """The 4 evaluations of ``run`` on the fixture and the docs dir holding its 3 cards."""
    from football_shot_models import baselines

    docs = fixture_root / "docs" / "models"
    evaluations = baselines.run(manifest, root=fixture_root, docs_dir=docs)
    return evaluations, docs
