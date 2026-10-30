"""The repository metadata rules of football-core's ``football_docs.meta``, applied to this repo.

The author spelling in LICENSE, mkdocs.yml and pyproject.toml, the CI workflow (check.yml) with
football-core checked out beside it while core is a path source, and the Pages wiring: this repo
publishes a site, so docs.yml must build and deploy it."""

from __future__ import annotations

from pathlib import Path

from football_docs import meta

REPO = Path(__file__).resolve().parents[1]


def test_repo_meta() -> None:
    assert meta.problems(REPO, pages=True) == []
