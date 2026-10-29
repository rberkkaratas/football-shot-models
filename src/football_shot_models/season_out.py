"""Season-out xGOT for the goalkeeper models: every rated row from fits that never saw its season.

GSAx charges a keeper for the gap between goals and xGOT, so the xGOT of a rated shot must not
come from a fit that trained on that shot's season (MS6). :func:`season_out` refits the
:func:`~football_shot_models.xgot.fit` chain of one provider (xG input, headline xGOT, penalty
xGOT) on folds grouped by competition-season: dev rows read the fold that left their season
out, hold-out (test-season) rows the refit's dev fit. Both providers use the same path (PM
2026-10-28: WhoScored's year swap is gone, ``SEASON_OUT`` is ``season_folds`` for both).

:func:`fold_seasons` and :func:`trained_seasons` say which seasons trained the fit that rated
each season, so a later test can count the pairs whose rated season's xGOT saw the test season
(PM E4, a disclosure). ``season_out(f, without=…)`` refits the open-play headline without the
named placement features (PM E2 (c), PR1: the z-free sensitivity of GSAx).

Hashed by the prediction cache (it fits through ``evaluate``); ``gsax`` calls it for every fit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import polars as pl

from football_shot_models import data, evaluation, models, xgot
from football_shot_models.evaluation import ModelSpec, evaluate

Folds = evaluation.Folds

SEASON_KEY: Final = ("competition_id", "season_id")
SOURCE: Final = pl.Enum(("season_folds", "holdout_fit"))
# how each provider's rated rows get xGOT from fits that never saw their season (PM 2026-10-28)
SEASON_OUT: Final[Mapping[str, str]] = MappingProxyType(
    {"statsbomb": "season_folds", "whoscored": "season_folds"}
)
SUFFIX: Final[Mapping[str, str]] = MappingProxyType({"season_folds": "-season-out"})
_TRAINED: Final = ("competition_id", "season_id", "trained_competition_id", "trained_season_id")


def _renamed(
    spec: object, suffix: str, xg_name: str | None = None, without: tuple[str, ...] = ()
) -> ModelSpec:
    if not isinstance(spec, ModelSpec):
        raise ValueError(f"refit: {spec!r} is not a fitted model spec")
    xg = spec.xg_input if xg_name is None else xg_name
    feats = tuple(c for c in spec.features if c not in without)
    return dataclasses.replace(spec, name=f"{spec.name}{suffix}", features=feats, xg_input=xg)


def refit(
    f: xgot.XgotFit,
    *,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    folds: Folds,
    kicks_dev: pl.DataFrame,
    kicks_held: pl.DataFrame,
    kicks_folds: Folds,
    drop: pl.Series,
    suffix: str,
    without: tuple[str, ...] = (),
) -> pl.DataFrame:
    """The :func:`~football_shot_models.xgot.fit` chain of ``f`` on other frames and folds.

    Every spec is ``f``'s, renamed ``name + suffix`` so its cache entry never evicts the match
    out-of-fold one. As in ``xgot.fit``, the xG input is refitted on ``dev`` with ``folds`` and
    cross-fitted into the frames (``xgot.xg_input``), and only then are the matches ``drop``
    left out (``xgot.drop_matches``: the coarse-grid ones still feed the xG input, R1); the
    headline xGOT and the penalty xGOT are refitted on what is left; shoot-out kicks keep
    ``f``'s scores. With ``without``, the headline is refitted without those features (its name
    gains ``-no-<names>``) and the penalty xGOT is not refitted: ``f``'s kicks are kept, so
    only the open-play rows of the result are season-out.

    Returns
    -------
    polars.DataFrame
        ``xgot.predictions`` of the refitted models (``split`` is the refit's own).

    Raises
    ------
    ValueError
        From the fits and the self-checks of ``xgot.xg_input`` and ``xgot.predictions``.

    Examples
    --------
    >>> pred = refit(f, dev=f.frames[0], held=f.frames[1], folds=folds, kicks_dev=f.kicks.dev,
    ...              kicks_held=f.kicks.held, kicks_folds=k_folds, drop=f.coarse, suffix="-x")
    ... # doctest: +SKIP
    """
    ff = models.freeze_frames_or_none(f.xg.freeze_frames)
    xg_ev = evaluate(
        _renamed(f.xg_input.spec, suffix), dev, held, folds, freeze_frames=ff, cache=f.cache
    )
    inp = xgot.drop_matches(xgot.xg_input(xg_ev, dev, held, folds), drop)
    head_suffix = f"{suffix}-no-{'-'.join(without)}" if without else suffix
    head = evaluate(
        _renamed(f.head.spec, head_suffix, xg_ev.spec.name, without),
        inp.on_dev, inp.on_held, inp.folds, cache=f.cache,
    )  # fmt: skip
    kicks = f.kicks
    if not without:
        pen = evaluate(
            _renamed(f.penalty.spec, suffix), kicks_dev, kicks_held, kicks_folds, cache=f.cache
        )
        kicks = dataclasses.replace(f.kicks, dev=kicks_dev, held=kicks_held, ev=pen)
    return xgot.predictions(inp, head, kicks, f.xg.matches)


def _seasons(frame: pl.DataFrame) -> set[tuple[str, str]]:
    return set(frame.select(SEASON_KEY).unique().iter_rows())


def _check_season_folds(frame: pl.DataFrame, folds: Folds, what: str) -> None:
    keys = frame.select(SEASON_KEY)
    for i, (train, test) in enumerate(folds):
        if shared := _seasons(keys[test]) & _seasons(keys[train]):
            raise ValueError(f"season_out: {what} fold {i} trains on {len(shared)} test season(s)")


def _disjoint(rated: pl.DataFrame, trained: pl.DataFrame, what: str) -> None:
    if shared := _seasons(rated) & _seasons(trained):
        raise ValueError(
            f"season_out: {len(shared)} {what} season(s) are in the fit that rates them"
        )


def _folds(dev: pl.DataFrame) -> Folds:
    """The competition-season folds of the chain (xG input and open-play xGOT)."""
    return evaluation.season_folds(dev, by=SEASON_KEY)


def season_out(f: xgot.XgotFit, *, without: Sequence[str] = ()) -> pl.DataFrame:
    """Every on-target open-play and in-game penalty row of ``f``, its xGOT season-out.

    Each row's ``xgot`` comes from fits (xG input and xGOT both) that never saw its season,
    built by :func:`refit` on ``f``'s own frames (``f.frames``, with ``f.coarse`` dropped after
    the xG input as in ``xgot.fit``), so a population ruling made there (coarse-grid matches
    out of the xGOT, still in the xG input) carries over. Dev rows come from folds grouped by
    competition-season (``season_folds``), hold-out rows from that refit's dev fit
    (``holdout_fit``); both providers (``SEASON_OUT``, read when called). Shoot-out kicks are
    left out. With ``without`` (PM E2 (c)), only the open-play rows, from a headline refitted
    without those features; the xG input is the same refit, read from the cache when ``f`` has
    one. Pure: the cache of ``f`` is the only place it writes.

    Returns
    -------
    polars.DataFrame
        The ``xgot.predictions`` columns (``split`` from ``f.predictions``), plus
        ``competition_name``, ``season_name`` (String), ``family`` (``data.FAMILY``) and
        ``source`` (``SOURCE``); none null.

    Raises
    ------
    TypeError
        ``without`` is a ``str``.
    ValueError
        An unknown ``f.provider`` or a ``SEASON_OUT`` value other than ``season_folds``; a
        ``without`` name the headline does not read; ids that differ from ``f``'s on-target
        non-shoot-out rows; a fold whose train rows hold one of its test seasons; a rated
        season among the training seasons of the fit that scored it; a null added column; a
        family that is not constant within a competition-season; from the fits.

    Examples
    --------
    >>> rows = season_out(xgot.fit("statsbomb", cache="data/cache"))  # doctest: +SKIP
    >>> rows["source"].unique().sort().to_list()  # doctest: +SKIP
    ['season_folds', 'holdout_fit']
    """
    if isinstance(without, str):
        raise TypeError("without must be a sequence of feature names, not a single str")
    drop, spec = tuple(without), f.head.spec
    read = spec.features if isinstance(spec, ModelSpec) else ()
    if bad := [c for c in drop if c not in read]:
        raise ValueError(f"season_out: {bad} not among the headline's features {read}")
    if f.provider not in SEASON_OUT:
        raise ValueError(f"season_out: unknown provider {f.provider!r}; {sorted(SEASON_OUT)}")
    if (source := SEASON_OUT[f.provider]) != "season_folds":
        raise ValueError(f"season_out: SEASON_OUT[{f.provider!r}] is {source!r}; 'season_folds'")
    kinds = ("open_play",) if drop else ("open_play", "penalty")
    rated = pl.col("on_target") & pl.col("kind").cast(pl.String).is_in(kinds)
    dev, held = f.frames
    folds, k_folds = _folds(dev), _folds(f.kicks.dev)
    _check_season_folds(dev, folds, "open-play")
    _check_season_folds(f.kicks.dev, k_folds, "kick")
    _disjoint(pl.concat([held.select(SEASON_KEY), f.kicks.held.select(SEASON_KEY)]),
              pl.concat([dev.select(SEASON_KEY), f.kicks.dev.select(SEASON_KEY)]),
              "hold-out")  # fmt: skip
    new = refit(f, dev=dev, held=held, folds=folds, kicks_dev=f.kicks.dev,
                kicks_held=f.kicks.held, kicks_folds=k_folds, drop=f.coarse,
                suffix=SUFFIX[source], without=drop)  # fmt: skip
    out = new.filter(rated).with_columns(
        source=pl.when(pl.col("split") == "dev").then(pl.lit(source)).otherwise(
            pl.lit("holdout_fit")).cast(SOURCE)
    )  # fmt: skip
    want = f.predictions.filter(rated).select("shot_id", "split")
    ids = out["shot_id"]
    if ids.n_unique() != out.height or set(ids) != set(want["shot_id"]):
        raise ValueError(
            f"season_out: {out.height} row(s) for the {want.height} rated rows of the fit"
        )
    meta = f.xg.matches.select("match_id", "competition_name", "season_name").join(
        data.match_family(f.xg.matches), on="match_id", how="left"
    )
    out = (
        out.drop("split")
        .join(want, on="shot_id", how="left", maintain_order="left")
        .join(meta, on="match_id", how="left", validate="m:1", maintain_order="left")
        .select(*f.predictions.columns, "competition_name", "season_name", "family", "source")
        .sort("shot_id")
    )
    added = ("split", "competition_name", "season_name", "family", "source")
    if nulls := out.select(pl.any_horizontal(pl.col(added).is_null())).to_series().sum():
        raise ValueError(f"season_out: {nulls} row(s) have a null {added} value")
    if (
        out.group_by(SEASON_KEY)
        .agg(pl.col("family").n_unique())
        .filter(pl.col("family") > 1)
        .height
    ):
        raise ValueError("season_out: a competition-season has two families")
    return out


def fold_seasons(dev: pl.DataFrame, held: pl.DataFrame, folds: Folds) -> pl.DataFrame:
    """Which seasons trained the fit that rated each season: one row per (rated, training) pair.

    Fold ``i``'s test seasons are rated by a fit on its train seasons; every ``held`` season by
    the fit on all of ``dev``. Frame-level, for any frames carrying ``SEASON_KEY``.

    Returns
    -------
    polars.DataFrame
        ``competition_id``, ``season_id``, ``trained_competition_id``, ``trained_season_id``
        (String, none null), unique and sorted.

    Raises
    ------
    ValueError
        A missing or null key column (count named); ``folds`` not a partition of ``dev``
        (``evaluation.check_partition``); a fold with no train or no test row.

    Examples
    --------
    >>> import numpy as np
    >>> dev = pl.DataFrame({"competition_id": ["c", "c", "d"], "season_id": ["1", "2", "1"]})
    >>> held = pl.DataFrame({"competition_id": ["c"], "season_id": ["3"]})
    >>> folds = ((np.array([1, 2]), np.array([0])), (np.array([0]), np.array([1, 2])))
    >>> fold_seasons(dev, held, folds).rows()[:3]
    [('c', '1', 'c', '2'), ('c', '1', 'd', '1'), ('c', '2', 'c', '1')]
    >>> fold_seasons(dev, held, folds).height
    7
    """
    for name, frame in (("dev", dev), ("held", held)):
        if missing := [c for c in SEASON_KEY if c not in frame.columns]:
            raise ValueError(f"fold_seasons: {name} is missing column(s) {missing}")
        if nulls := sum(frame[c].null_count() for c in SEASON_KEY):
            raise ValueError(f"fold_seasons: {name} has {nulls} null key value(s)")
    evaluation.check_partition(folds, dev.height)
    keys = dev.select(pl.col(SEASON_KEY).cast(pl.String))
    trained = keys.select(trained_competition_id="competition_id", trained_season_id="season_id")
    parts = []
    for i, (train, test) in enumerate(folds):
        if train.size == 0 or test.size == 0:
            raise ValueError(f"fold_seasons: fold {i} has no train or no test row")
        parts.append(keys[test].unique().join(trained[train].unique(), how="cross"))
    parts.append(
        held.select(pl.col(SEASON_KEY).cast(pl.String)).unique().join(trained.unique(), how="cross")
    )
    schema = dict.fromkeys(_TRAINED, pl.String)
    return pl.concat([pl.DataFrame(schema=schema), *parts]).unique().sort(_TRAINED)


def trained_seasons(f: xgot.XgotFit) -> pl.DataFrame:
    """:func:`fold_seasons` of the chain :func:`season_out` refits: ``f.frames`` on its
    competition-season folds (the xG input's, coarse-grid matches included), never
    ``f.inp.folds``. Fits nothing.

    Examples
    --------
    >>> trained_seasons(xgot.fit("statsbomb", cache="data/cache")).columns  # doctest: +SKIP
    ['competition_id', 'season_id', 'trained_competition_id', 'trained_season_id']
    """
    dev, held = f.frames
    return fold_seasons(dev, held, _folds(dev))
