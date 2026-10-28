"""Shared fixtures: one synthetic snapshot and one ``baselines.run`` per session (M1.0 / M1.1),
one ``models.run`` on the ablation fixture plus penalties (MS3.models), and one ``xgot.fit`` per
provider on the ``xgot_cases`` snapshots (MS6 xgot-ready §4).

Builders live in ``xg_cases``, ``ablation_cases`` and ``models_cases``. Imports of the package
under test are inside the fixtures, so a missing module fails the tests that need it, not the
whole collection.
"""

from __future__ import annotations

import contextlib
import inspect
import io
from pathlib import Path
from typing import Any

import models_cases as mc
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


@pytest.fixture(scope="session")
def models_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """``models.run`` once on the models fixture with a fixture-sized GBM grid (``run`` reads the
    module global ``GBM`` at call time), with its stdout and the loaded dev / hold-out."""
    from football_shot_models import data, models

    root = tmp_path_factory.mktemp("models")
    manifest = mc.write_models_snapshot(root)
    docs = root / "docs" / "models"
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
        mp.setattr(models, "GBM", mc.tiny_gbm(models))
        result = models.run(manifest, docs_dir=docs)
    xg = data.load_xg_data(manifest)
    dev, held = data.split_holdout(xg.shots)
    return {
        "manifest": manifest,
        "docs": docs,
        "run": result,
        "stdout": out.getvalue(),
        "xg": xg,
        "dev": dev,
        "held": held,
    }


def tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` (relative posix path -> bytes)."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="session")
def xgot_fits(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """``xgot.fit(p, cache=None)`` once per provider (MS6 §4) on the ``xgot_cases`` snapshots,
    with ``xgot.PROVIDERS`` and the candidate globals patched (``xgot_cases.patched_providers``).

    Run from an empty working directory with stdout and stderr captured and every file under the
    tmp root recorded before and after (``test_fit_is_pure``), and with a pass-through spy on
    ``models.evaluate_candidates`` that records each call's bound arguments, its result and the
    module globals the provider's specs come from (``test_candidates_shared``).
    """
    import xgot_cases as xg

    from football_shot_models import ablation, models, v1, xgot

    root = tmp_path_factory.mktemp("xgot-fit")
    manifests = {}
    for p in ("statsbomb", "whoscored"):
        (root / p).mkdir()
        manifests[p] = xg.write_xgot_snapshot(root / p, p)
    cwd = root / "cwd"
    cwd.mkdir()
    before = tree(root)
    out, err = io.StringIO(), io.StringIO()
    calls: dict[str, list[tuple[dict[str, Any], Any]]] = {}
    fits: dict[str, Any] = {}
    specs: dict[str, tuple[Any, Any]] = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(xgot, "PROVIDERS", xg.patched_providers(xgot, manifests, mp))
        specs = {"statsbomb": (ablation.STEPS, models.GBM), "whoscored": (v1.STEPS, v1.GBM)}
        real = models.evaluate_candidates
        current: list[str] = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            bound = inspect.signature(real).bind(*args, **kwargs)
            result = real(*args, **kwargs)
            calls.setdefault(current[-1], []).append((dict(bound.arguments), result))
            return result

        mp.setattr(models, "evaluate_candidates", spy)
        mp.chdir(cwd)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            for p in ("statsbomb", "whoscored"):
                current.append(p)
                fits[p] = xgot.fit(p, cache=None)
    return {
        "root": root,
        "manifests": manifests,
        "fits": fits,
        "calls": calls,
        "specs": specs,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "before": before,
        "after": tree(root),
    }


GSAX_SPLITS = {"statsbomb": 2, "whoscored": 3}
GSAX_SPIED = ("rate", "ci_counts", "fit_keeper_model", "next_season", "reliability")


@pytest.fixture(scope="session")
def gsax_fits(
    xgot_fits: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """``gsax.fit`` once per provider on the ``xgot_fits`` fits (MS6 gsax §8), with
    ``evaluation.N_SPLITS`` patched per provider (``GSAX_SPLITS``; read when called by
    ``season_folds`` and ``cv_folds``): 2 for StatsBomb (2 dev competition-seasons), 3 for
    WhoScored, whose swapped fit trains on the 6-match 2026/27 season (with 2 match folds the
    3-match training folds fail: the xGOT LR does not converge, or the penalty fold sees one
    parity of ``xgot_cases.CELLS`` and is rank-deficient; probed in QA's tests.md).

    Run from an empty working directory with stdout and stderr captured and every file under
    the tmp roots recorded before and after (``test_fit_is_pure``), and with a pass-through spy
    on the ``evaluate`` name ``xgot`` reads (``xgot.refit``), recording each call's bound
    arguments and its result (key ``"_result"``) per provider (``test_gsax_season_out``), and
    pass-through spies on the ``gsax`` functions ``fit`` calls (``GSAX_SPIED``), recording each
    outermost call's name and bound arguments per provider in ``"gsax_calls"``
    (``test_gsax_fit``).
    """
    from football_shot_models import evaluation, gsax, xgot

    fits = xgot_fits["fits"]
    for p, f in fits.items():  # spec §8 preconditions: kicks and shots in >= 2 matches
        on = f.predictions.filter(pl.col("on_target") & (pl.col("kind") != "shootout"))
        if p == "statsbomb":
            seasons = on.filter(pl.col("split") == "dev").select("competition_id", "season_id")
            groups = seasons.unique().rows()
            assert len(groups) == 2, groups
        else:
            groups = [("ws:710", "ws:72627")]
        for comp, season in groups:
            rows = on.filter((pl.col("competition_id") == comp) & (pl.col("season_id") == season))
            for kind in ("open_play", "penalty"):
                k = rows.filter(pl.col("kind") == kind)
                assert k.height and k["match_id"].n_unique() >= 2, (p, comp, season, kind)
    root = xgot_fits["root"]
    cwd = tmp_path_factory.mktemp("gsax-cwd")
    before = (tree(root), tree(cwd))
    out, err = io.StringIO(), io.StringIO()
    calls: dict[str, list[dict[str, Any]]] = {}
    gsax_calls: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    result: dict[str, Any] = {}
    current: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        real = xgot.evaluate

        def spy(*args: Any, **kwargs: Any) -> Any:
            bound = inspect.signature(real).bind(*args, **kwargs)
            out = real(*args, **kwargs)
            calls.setdefault(current[-1], []).append({**bound.arguments, "_result": out})
            return out

        mp.setattr(xgot, "evaluate", spy)

        depth: list[int] = []  # only the calls ``fit`` makes, not those made inside them

        def gsax_spy(name: str) -> Any:
            fn = getattr(gsax, name)

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                if not depth:
                    bound = inspect.signature(fn).bind(*args, **kwargs)
                    gsax_calls.setdefault(current[-1], []).append((name, dict(bound.arguments)))
                depth.append(1)
                try:
                    return fn(*args, **kwargs)
                finally:
                    depth.pop()

            return wrapped

        for name in GSAX_SPIED:
            mp.setattr(gsax, name, gsax_spy(name))
        mp.chdir(cwd)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            for p in ("statsbomb", "whoscored"):
                current.append(p)
                mp.setattr(evaluation, "N_SPLITS", GSAX_SPLITS[p])
                result[p] = gsax.fit(fits[p])
    return {
        "fits": result,
        "xgot": fits,
        "calls": calls,
        "gsax_calls": gsax_calls,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "before": before,
        "after": (tree(root), tree(cwd)),
    }


# ---- real snapshots (make reproduce; MS6 gsax §5: shared by test_xgot_run and test_gsax_run) ---

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def real(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The real-data ``xgot_cards.run`` per provider, computed once on first use
    (:func:`real_xgot`) and kept for the session."""
    return {"tmp": tmp_path_factory.mktemp("real-xgot")}


def real_xgot(real: dict[str, Any], p: str) -> dict[str, Any]:
    """``{"run": XgotRun, "docs": cards dir}`` for provider ``p`` on the real snapshot (skips
    without it), reading the prediction cache when ``FSM_CACHE`` allows."""
    if p not in real:
        from football_shot_models import cache, xgot, xgot_cards

        manifest = REPO / xgot.PROVIDERS[p].manifest
        snap = "statsbomb-open-v1" if p == "statsbomb" else "whoscored-v1"
        if not ((REPO / "data" / "snapshots" / snap).is_dir() and manifest.is_file()):
            pytest.skip(f"needs the {p} snapshot")
        docs = real["tmp"] / p / "models"
        root = cache.cache_root(REPO / "data" / "cache")
        real[p] = {"run": xgot_cards.run(p, docs_dir=docs, cache=root), "docs": docs}
    return dict(real[p])
