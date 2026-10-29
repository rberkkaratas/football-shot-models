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
    module global ``GBM`` at call time), with its stdout and the loaded dev / hold-out.

    PR1.xg-context §3.3: ``models_cases.patch_level`` also makes POOLED and the level fixes
    fixture-sized and sets ``ADOPTED = None`` with every fix in ``DOWNSTREAM_MISSED``, so the
    run's ADOPTED check agrees with the rule on any fixture data (the limitation branch)."""
    from football_shot_models import data, models

    root = tmp_path_factory.mktemp("models")
    manifest = mc.write_models_snapshot(root)
    docs = root / "docs" / "models"
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
        mc.patch_level(models, mp)
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


GSAX_SPLITS = {"statsbomb": 2, "whoscored": 2}
GSAX_SPIED = ("rate", "ci_counts", "fit_keeper_model", "next_season", "reliability")
# the sensitivity and E4 steps ``assemble`` feeds (integration census, PR1 keeper-fixes): spied
# like ``GSAX_SPIED``, so the ``rate`` calls inside them are not recorded as ``fit``'s own
FIT_SPIED = ("line_rated", "noz_rated", "saw_test")
# synthetic stops for the StatsBomb fixture, whose xGOT rows have a null ``end_x``: goals on the
# line, saves near it (inside ``LINE_YD``) except every ``FAR_EVERY``-th one in ``shot_id`` order
GOAL_X, NEAR_X, FAR_X, FAR_EVERY = 105.0, 104.5, 100.0, 3


def with_stops(f: Any) -> Any:
    """``f`` with ``end_x`` set on ``inp.on_dev`` / ``on_held`` (``GOAL_X``, ``NEAR_X``,
    ``FAR_X``), so ``gsax.fit`` reaches its goal_z line branch (``stops`` -> ``line_rated``).
    ``end_x`` is post-shot and no xGOT feature reads it."""
    import dataclasses

    def stop(frame: pl.DataFrame) -> pl.DataFrame:
        rank = pl.col("shot_id").rank("ordinal").over(pl.col("is_goal")) - 1
        x = (
            pl.when(pl.col("is_goal")).then(pl.lit(GOAL_X))
            .when(rank % FAR_EVERY == 0).then(pl.lit(FAR_X))
            .otherwise(pl.lit(NEAR_X))
        )  # fmt: skip
        return frame.with_columns(x.cast(frame.schema["end_x"]).alias("end_x"))

    inp = dataclasses.replace(f.inp, on_dev=stop(f.inp.on_dev), on_held=stop(f.inp.on_held))
    return dataclasses.replace(f, inp=inp)


# the WhoScored dev (train) competition-seasons its season folds split; StatsBomb's 2 are read
WS_TRAIN = [("ws:710", "ws:72526"), ("ws:730", "ws:72026")]


@pytest.fixture(scope="session")
def gsax_fits(
    xgot_fits: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """``gsax.fit`` once per provider on the ``xgot_fits`` fits (MS6 gsax §8; PR1 keeper-fixes
    §9), with ``evaluation.N_SPLITS`` patched per provider (``GSAX_SPLITS``; read when called by
    ``season_folds`` and ``cv_folds``): 2 for both providers, whose season folds each split 2
    dev competition-seasons (``SEASON_OUT`` is ``season_folds`` for both since PR1; each of the
    2 seasons has kicks and open play in >= 2 matches, checked first). The StatsBomb fit gets
    synthetic stops (:func:`with_stops`; its fixture ``end_x`` is null), so ``fit`` reaches the
    goal_z line branch and, with ``NOZ_PROVIDERS``, the z-free one; WhoScored reaches neither.
    ``"xgot"`` holds the fits ``fit`` received.

    Run from an empty working directory with stdout and stderr captured and every file under
    the tmp roots recorded before and after (``test_fit_is_pure``), and with a pass-through spy
    on the ``evaluate`` name ``season_out`` reads (``season_out.refit``), recording each call's
    bound arguments and its result (key ``"_result"``) per provider (``test_gsax_season_out``),
    and pass-through spies on the ``gsax`` functions ``fit`` calls (``GSAX_SPIED`` and
    ``FIT_SPIED``), recording each outermost call's name and bound arguments per provider in
    ``"gsax_calls"`` (``test_gsax_fit``).
    """
    from football_shot_models import evaluation, gsax, season_out

    fits = {**xgot_fits["fits"], "statsbomb": with_stops(xgot_fits["fits"]["statsbomb"])}
    for p, f in fits.items():  # spec §8 preconditions: kicks and shots in >= 2 matches
        on = f.predictions.filter(pl.col("on_target") & (pl.col("kind") != "shootout"))
        if p == "statsbomb":
            seasons = on.filter(pl.col("split") == "dev").select("competition_id", "season_id")
            groups = seasons.unique().rows()
            assert len(groups) == 2, groups
        else:
            groups = WS_TRAIN
            dev = on.filter(pl.col("split") == "dev").select("competition_id", "season_id")
            assert sorted(dev.unique().rows()) == sorted(groups), dev.unique().rows()
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
        real = season_out.evaluate

        def spy(*args: Any, **kwargs: Any) -> Any:
            bound = inspect.signature(real).bind(*args, **kwargs)
            out = real(*args, **kwargs)
            calls.setdefault(current[-1], []).append({**bound.arguments, "_result": out})
            return out

        mp.setattr(season_out, "evaluate", spy)

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

        for name in (*GSAX_SPIED, *FIT_SPIED):
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


@pytest.fixture(scope="session")
def gsax_census(xgot_fits: dict[str, Any]) -> dict[str, Any]:
    """``gsax.assemble`` on the hand-built census (``gsax_cases.census_frame``; PR1 keeper-fixes
    §9): every branch the wiring feeds (kept and left-out pairs, team-selected, small and
    team-selected-and-small seasons, a fitted and an E2-small penalty pool, one test season per
    leave-out branch, stops on both sides of ``LINE_X``, a seen and an unseen pair). No fit runs.

    ``f`` is the StatsBomb fixture fit with ``predictions`` (the census rows, xGOT scaled by 0.95
    so the match out-of-fold xGOT differs from the rated one) and ``inp.on_dev`` / ``on_held``
    (the census geometry) replaced, so check row 1, ``placement_bands`` and the cards can read
    the census ids (QA's choice, tests.md: the spec's ``assemble(xgot_fits[...], ...)`` would
    leave those joins without a row).
    """
    import dataclasses

    import gsax_cases as gc

    from football_shot_models import gsax

    base = xgot_fits["fits"]["statsbomb"]
    pred = gc.census_frame()
    missing = [c for c in base.predictions.columns if c not in pred.columns]
    assert missing == [], missing
    geo = gc.census_geometry(pred)
    oof = pred.select(base.predictions.columns).with_columns(xgot=pl.col("xgot") * 0.95)
    f = dataclasses.replace(
        base, predictions=oof,
        inp=dataclasses.replace(base.inp, on_dev=geo, on_held=geo.clear()),
    )  # fmt: skip
    stops = gc.census_stops(pred, gsax.LINE_X)
    trained = gc.census_trained()
    g = gsax.assemble(f, pred, stops=stops, trained=trained)
    return {"g": g, "f": f, "pred": pred, "stops": stops, "trained": trained, "geometry": geo}


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


V2_MIN_SEGMENT = 100  # fixture test competitions: 149 and 120 shots (slope), 90 (none)


@pytest.fixture(scope="session")
def v2_fit(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """``v2.fit(manifest, cache=None)`` once on the ``v2_cases`` snapshot (PR1.xg-context §3.4)
    with fixture-sized GBMs for ``v1.GBM`` and ``v2.GBM``, from an empty working directory with
    stdout and stderr captured (``fit`` neither writes nor prints). ``v1.TRAIN_SEASONS`` /
    ``TEST_SEASONS`` are replaced by equal tuples that are new objects, and pass-through spies on
    ``v1.population`` and ``data.season_holdout`` record their calls, so the spy test can tell v1's
    population and split from a copy of them.

    ``models.MIN_SEGMENT_SHOTS`` (read by ``fit`` when called) is ``V2_MIN_SEGMENT``, so the
    per-competition table has a competition with a calibration slope and one without
    (integration census; at 1,000 every fixture competition would be the small kind)."""
    import v2_cases

    from football_shot_models import data, models, v1, v2

    root = tmp_path_factory.mktemp("v2")
    manifest = v2_cases.write_v2_snapshot(root / "snap")
    cwd = root / "cwd"
    cwd.mkdir()
    calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any], Any]]] = {}
    out, err = io.StringIO(), io.StringIO()
    seasons = (tuple(list(v1.TRAIN_SEASONS)), tuple(list(v1.TEST_SEASONS)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(v1, "GBM", mc.tiny_gbm(v1, dict(v1.GBM_SIGNS)))
        mp.setattr(v2, "GBM", mc.tiny_gbm(v2, dict(v1.GBM_SIGNS)))
        mp.setattr(v1, "TRAIN_SEASONS", seasons[0])
        mp.setattr(v1, "TEST_SEASONS", seasons[1])
        mp.setattr(models, "MIN_SEGMENT_SHOTS", V2_MIN_SEGMENT)
        for owner, name in ((v1, "population"), (data, "season_holdout")):
            real = getattr(owner, name)

            def spy(*args: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
                result = _real(*args, **kwargs)
                calls.setdefault(_name, []).append((args, kwargs, result))
                return result

            mp.setattr(owner, name, spy)
        mp.chdir(cwd)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fit = v2.fit(manifest, cache=None)
    return {
        "fit": fit,
        "manifest": manifest,
        "calls": calls,
        "seasons": seasons,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "cwd": cwd,
    }
