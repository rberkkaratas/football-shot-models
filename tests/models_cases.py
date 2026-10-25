"""Hand-built fixtures for the MS3.models tests (spec ``runs/MS3/models``): GBM arrays, a penalty
table, evaluation stand-ins and the ``models.run`` snapshot.

Every id and number is invented. Nothing here imports ``football_shot_models.models`` at module
level, so a missing module fails the tests that need it, not the collection.

- ``signal_xy``: 600 shots in 60 matches, columns ``d`` (distance-like), ``a`` (angle-like) and
  ``z`` (noise, unconstrained); goals depend on ``d`` and ``a``.
- ``no_signal_xy`` / ``constant_xy``: the spec's guard fixtures (``spec_guards.py`` 1 and 6).
- ``penalties_frame``: 10 in-game kicks and 5 shoot-out kicks shaped like ``XgData.penalties``,
  with near misses on the hold-out key ((``sb:11``, ``sb:27``) and (``sb:2``, ``sb:44``) are
  dev) and a taker whose shoot-out kicks must not count.
- ``write_models_snapshot``: the ablation fixture plus in-game penalties (two in the hold-out,
  one women's) and shoot-out kicks (men and women), so every penalty card row exists.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ablation_cases as ac
import numpy as np
import numpy.typing as npt
import polars as pl
import xg_cases as xc
from football_core import schema
from football_core.io import snapshot
from scipy.special import expit

from football_shot_models import data, evaluation, metrics

FloatArray = npt.NDArray[np.float64]
TINY: dict[str, Any] = {
    "leaf_grid": (4,),
    "max_iter": 20,
    "step": 10,
    "inner_splits": 3,
    "min_samples_leaf": 20,
}
SIGNAL: dict[str, Any] = {
    "leaf_grid": (4, 8),
    "max_iter": 60,
    "step": 20,
    "inner_splits": 3,
    "min_samples_leaf": 20,
}
SIGNAL_NAMES = ("d", "a", "z")
SIGNAL_MAP = {"d": -1, "a": 1}


# ---- GBM arrays --------------------------------------------------------------------------------


def signal_xy(seed: int = 7) -> tuple[FloatArray, npt.NDArray[np.bool_], npt.NDArray[np.str_]]:
    """600 rows ``(d, a, z)``, goals from ``expit(1 - 0.15 d + 1.5 a)``, 60 matches of 10."""
    rng = np.random.default_rng(seed)
    n = 600
    d = rng.uniform(5.0, 30.0, n)
    a = rng.uniform(0.1, 1.2, n)
    z = rng.normal(size=n)
    y = rng.uniform(size=n) < expit(1.0 - 0.15 * d + 1.5 * a)
    groups = np.array([f"sb:m{i // 10:02d}" for i in range(n)])
    return np.column_stack([d, a, z]), y, groups


def no_signal_xy() -> tuple[FloatArray, npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """``spec_guards.py`` 1: 300 rows of noise in 30 matches (inner scores rank goals backwards)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    y = rng.uniform(size=300) < 0.3
    return x, y, np.repeat(np.arange(30), 10)


def constant_xy() -> tuple[FloatArray, npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """``spec_guards.py`` 6: constant x and 3 goals in every match of 10 (constant inner scores)."""
    y = np.tile(np.array([True] * 3 + [False] * 7), 40)
    return np.zeros((400, 2)), y, np.repeat(np.arange(40), 10)


# ---- evaluation stand-ins ----------------------------------------------------------------------


def scores(
    *,
    n: int = 1000,
    goals: int = 100,
    xg: float = 100.0,
    log_loss: float = 0.27,
    auc: float = 0.80,
    slope: float | None = 1.0,
) -> metrics.Scores:
    return metrics.Scores(n, goals, xg, log_loss, 0.077, auc, 0.0, slope)


def fake_eval(
    spec: evaluation.ModelSpec | evaluation.Benchmark,
    *,
    cv: metrics.Scores | None = None,
    holdout: metrics.Scores | None = None,
    folds: tuple[metrics.Scores, ...] | None = None,
    oof: FloatArray | None = None,
    holdout_p: FloatArray | None = None,
    coefficients: evaluation.Coefficients = (),
) -> evaluation.Evaluation:
    """An Evaluation with chosen scores; arrays default to empty (unused by the code under test)."""
    folds = (
        folds if folds is not None else tuple(scores(n=200, goals=20, xg=20.0) for _ in range(5))
    )
    cv = cv if cv is not None else metrics.mean_scores(folds)
    return evaluation.Evaluation(
        spec=spec,
        folds=folds,
        cv=cv,
        oof=np.empty(0) if oof is None else oof,
        train=cv,
        holdout=holdout if holdout is not None else scores(),
        coefficients=coefficients,
        holdout_p=np.empty(0) if holdout_p is None else holdout_p,
        fold_test=tuple(np.empty(0, dtype=np.intp) for _ in folds),
    )


def gain(gain: float, low: float, high: float) -> metrics.GainCI:
    return metrics.GainCI(gain, low, high, metrics.N_BOOT, 100)


def statsbomb_sentence(
    models: Any, ev: evaluation.Evaluation, sb_gain: Any, benchmark: evaluation.Evaluation
) -> str:
    """Call ``models.benchmark_sentence`` (``statsbomb_sentence`` until the run3-ms3 fix pass).
    Spec §3.2 gives ``(ev, sb_gain)``, but the sentence prints the benchmark's own hold-out Σ xG,
    which only the benchmark evaluation holds; when the Engineer adds a ``benchmark`` parameter it
    is passed by name (choice for the Architect)."""
    fn: Callable[..., str] = models.benchmark_sentence
    if "benchmark" in inspect.signature(fn).parameters:
        return fn(ev, sb_gain, benchmark=benchmark)
    return fn(ev, sb_gain)


def card_context() -> Any:
    from football_shot_models import cards

    return cards.CardContext(
        manifest_name="statsbomb-open-v1",
        data_digest="0" * 16,
        source_revision="0" * 40,
        sklearn_version="1.9.1",
        holdout_label="Premier League 2015/2016",
        dev_shots=700,
        dev_matches=20,
        holdout_shots=100,
        holdout_matches=3,
        competition_seasons=5,
        male_shots=600,
        female_shots=200,
        holdout_female_shots=0,
        exclusions=tuple((r, 1, 0) for r in data.EXCLUSION_REASONS),
        team_selected_seasons=1,
        team_selected_short=0,
    )


# ---- penalties ---------------------------------------------------------------------------------

# match id: (competition, season, gender, family)
PEN_MATCHES: dict[str, tuple[str, str, str, str]] = {
    "sb:m01": ("sb:2", "sb:27", "male", "league or tournament"),  # hold-out
    "sb:m02": ("sb:2", "sb:27", "male", "league or tournament"),  # hold-out
    "sb:m03": ("sb:11", "sb:27", "male", "team-selected"),  # same season, other competition
    "sb:m04": ("sb:2", "sb:44", "male", "league or tournament"),  # same competition, other season
    "sb:m05": ("sb:37", "sb:90", "female", "women"),
    "sb:m06": ("sb:11", "sb:4", "male", "league or tournament"),
}
# shot id, match, period, goal, taker, StatsBomb xG
PEN_KICKS: tuple[tuple[str, str, int, bool, str, float | None], ...] = (
    ("sb:k01", "sb:m01", 1, True, "sb:p-b", 0.7835),
    ("sb:k02", "sb:m01", 2, True, "sb:p-c", 0.7835),
    ("sb:k03", "sb:m02", 1, False, "sb:p-d", 0.7835),
    ("sb:k04", "sb:m03", 1, True, "sb:p-a", 0.7835),
    ("sb:k05", "sb:m03", 2, False, "sb:p-a", 0.7835),
    ("sb:k06", "sb:m04", 2, True, "sb:p-a", 0.7835),
    ("sb:k07", "sb:m05", 1, True, "sb:p-w", 0.7835),
    ("sb:k08", "sb:m05", 2, True, "sb:p-w", 0.7835),
    ("sb:k09", "sb:m05", 2, False, "sb:p-v", 0.7835),
    ("sb:k10", "sb:m06", 1, True, "sb:p-e", 0.7835),
    ("sb:s01", "sb:m03", 5, True, "sb:p-a", 0.75),  # shoot-outs: sb:p-a would lead with 6
    ("sb:s02", "sb:m03", 5, True, "sb:p-a", 0.80),
    ("sb:s03", "sb:m03", 5, False, "sb:p-a", 0.85),
    ("sb:s04", "sb:m05", 5, True, "sb:p-w", 0.70),
    ("sb:s05", "sb:m05", 5, False, "sb:p-v", None),
)
# (kicks, goals, matches) per row, in the spec §3.3 order
PEN_ROWS: tuple[tuple[int, int, int], ...] = (
    (10, 7, 6),  # in-game all (the model)
    (7, 5, 4),  # dev: m03-m06 (the near misses on the hold-out key included)
    (3, 2, 2),  # hold-out: m01, m02
    (7, 5, 5),  # men
    (3, 2, 1),  # women
    (5, 4, 4),  # family league or tournament: m01, m02, m04, m06
    (2, 1, 1),  # family team-selected: m03 (no family `women` row)
    (5, 3, 2),  # shoot-out all
    (3, 2, 1),  # shoot-out men
    (2, 1, 1),  # shoot-out women
)


def penalties_frame(
    kicks: tuple[tuple[str, str, int, bool, str, float | None], ...] = PEN_KICKS,
) -> pl.DataFrame:
    """``XgData.penalties`` columns: C3 shots, then ``reason``, ``gender``, ``family``."""
    shots = [
        xc.Shot(
            shot_id, match, PEN_MATCHES[match][0], PEN_MATCHES[match][1], x=94.0, y=34.0,
            goal=goal, period=period, minute=120 if period == 5 else 30 + 45 * (period - 1),
            penalty=True, provider_xg=sb,
        )
        for shot_id, match, period, goal, _, sb in kicks
    ]  # fmt: skip
    frame = xc.shots_frame(shots)
    meta = [PEN_MATCHES[k[1]] for k in kicks]
    return frame.with_columns(
        pl.Series("player_id", [k[4] for k in kicks], dtype=pl.String()),
        pl.Series(
            "reason",
            ["shootout" if k[2] == 5 else "penalty" for k in kicks],
            dtype=data.EXCLUSION_REASON,
        ),
        pl.Series("gender", [m[2] for m in meta], dtype=schema.GENDER),
        pl.Series("family", [m[3] for m in meta], dtype=data.FAMILY),
    )


def with_null(frame: pl.DataFrame, column: str, row: int) -> pl.DataFrame:
    """``frame`` with ``column`` null on one row, dtype kept."""
    expr = pl.when(pl.int_range(pl.len()) == row).then(None).otherwise(pl.col(column))
    return frame.with_columns(expr.cast(frame.schema[column]).alias(column))


def set_value(frame: pl.DataFrame, column: str, row: int, value: object) -> pl.DataFrame:
    expr = pl.when(pl.int_range(pl.len()) == row).then(pl.lit(value)).otherwise(pl.col(column))
    return frame.with_columns(expr.cast(frame.schema[column]).alias(column))


# ---- models.run snapshot -----------------------------------------------------------------------

# shot suffix, match, period, goal: in-game kicks in two hold-out matches and a women's match,
# shoot-out kicks in a men's and a women's match (the ablation fixture already has one in-game
# penalty in sb:m06)
_RUN_KICKS = (
    ("pen", "sb:m01", 1, True),
    ("pen", "sb:m02", 2, False),
    ("pen", "sb:m19", 1, True),
    ("so1", "sb:m07", 5, True),
    ("so2", "sb:m07", 5, False),
    ("so1", "sb:m20", 5, True),
)


def models_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    shots, frames, matches = ac.ablation_tables()
    by_id = {r["match_id"]: r for r in matches.iter_rows(named=True)}
    extra = []
    for suffix, match_id, period, goal in _RUN_KICKS:
        m = by_id[match_id]
        extra.append(
            xc.Shot(
                f"sb:s-{match_id[3:]}-{suffix}", match_id, m["competition_id"], m["season_id"],
                m["home_team_id"], m["away_team_id"], True, 94.0, 34.0, goal=goal, period=period,
                minute=120 if period == 5 else 60, penalty=True, provider_xg=0.7835,
                match_date=m["match_date"],
            )
        )  # fmt: skip
    all_shots = pl.concat([shots, xc.shots_frame(extra)]).sort("shot_id")
    return all_shots, frames, matches


def write_models_snapshot(root: Path) -> Path:
    """The ``models.run`` fixture as snapshot ``statsbomb-open-v1`` under ``root``; the manifest."""
    shots, frames, matches = models_tables()
    result = snapshot.write_snapshot(
        xc.SNAPSHOT,
        {"shots": shots, "freeze_frames": frames, "matches": matches},
        xc.SOURCE,
        root=root,
        sort_by={"shots": ("shot_id",), "freeze_frames": ("shot_id",), "matches": ("match_id",)},
        builder="statsbomb",
    )
    return result.manifest_path


def tiny_gbm(models: Any, monotone: dict[str, int] | None = None) -> evaluation.ModelSpec:
    """``models.GBM`` with a fixture-sized grid (and optionally another sign map)."""
    import dataclasses
    import functools

    args = () if monotone is None else (monotone,)
    make = functools.partial(models.MonotoneGBM, *args, **TINY)
    spec: evaluation.ModelSpec = dataclasses.replace(models.GBM, make=make)
    return spec
