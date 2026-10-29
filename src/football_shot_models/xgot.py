"""MS5 / MS6 xGOT / PSxG: the post-shot goal probability of on-target shots, for both providers.

**Population (P2.2).** xGOT answers "how likely was this shot to go in, given where it crossed
the goal line", so its rows are the shots that reached the net or the keeper: C3 ``on_target``
(a goal or a keeper save; never an own goal, a block, a clearance off the line or the woodwork).
Two populations are never mixed: **open play** is the provider's xG population restricted to
on-target shots, and **kicks** are the on-target penalties, in-game and shoot-out (labelled
apart). :func:`reconcile` accounts for every C3 on-target shot by its first matching reason, so
the scored rows reconcile with C3.

**The xG input.** Every open-play model reads ``xg_logit``, the logit of the provider's headline
xG (``models.select_headline``) refitted without the keeper-position features (PM X1; a
WhoScored headline reads none, so it is used as it is). It is cross-fitted (:func:`xg_input`):
a dev shot reads its out-of-fold xG, a hold-out shot the xG of the fit on dev (the rows its
spec's ``train_exclude`` keeps: StatsBomb's headline leaves team-selected seasons out, PR1), and
the xGOT folds are the xG folds restricted to on-target shots, so no xGOT row sees an xG fitted
on itself.

**Models (P2.3, P2.4).** Every model goes through ``evaluation.evaluate`` with the xGOT guard
(``features.check_xgot_features``) and registry (``features.XGOT_DERIVED``). The LR reads the
xG input, placement splines, the width past the inner post edge and angle × placement terms
(near / far post included); the GBM is monotone in the xG input only, its placement free (MS6:
chosen by the band rule of :func:`band_table` over a monotone width cut), and is the headline
only if its CV gain over the LR has a match-bootstrap interval above 0. The penalty xGOT reads
placement only, with its own folds; shoot-out kicks are scored by its dev fit.

**Shooting goals added (P2.5)** is xGOT − xG per shot (:func:`sga`), with blocked and
off-target shots at 0 xGOT. It is reported as split-half correlations next to goals − xG
(``metrics.split_half`` by alternating matches): a descriptive measure, not a claim about
finishing skill.

**Fit, then render (MS6).** :func:`fit` computes everything of one provider and returns it as
an :class:`XgotFit`; it writes nothing and prints nothing (a cache root, when given, is the
only place it writes), so the goalkeeper models and charts read predictions and tables from
it without a card. :mod:`football_shot_models.xgot_cards` renders and writes the cards: run
``python -m football_shot_models.xgot_cards --provider both`` (``make xgot``) after ``make
xg-models xg-v1``, so the xG candidates come from the prediction cache. Nothing here writes a
row: :func:`predictions` stays in memory, and the WhoScored cards, README block and output hold
counts and metrics only.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import goalmouth
from football_core.io import snapshot

from football_shot_models import (
    ablation,
    baselines,
    cards,
    data,
    evaluation,
    features,
    gbm,
    metrics,
    models,
    release,
    v1,
)
from football_shot_models.cache import PredictionCache
from football_shot_models.cards import ProviderText, fmt, gain_cell, signed
from football_shot_models.evaluation import Comparison, Evaluation, ModelSpec, evaluate, labels
from football_shot_models.metrics import Check, Scores, SplitHalf

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
StrPath = str | os.PathLike[str]
Folds = evaluation.Folds
CandidateSpecs = Callable[[], tuple[Sequence[ModelSpec], ModelSpec]]

# MS6 (PM E1 (a)): a StatsBomb match on the coarse (1 yd) recording grid leaves the xGOT fit
# and GSAx whole, so "coarse_grid" is the first reason of its on-target rows
RECONCILE_REASONS: Final = ("coarse_grid", *data.EXCLUSION_REASONS, "unresolved_assist", "modelled")
POPULATION_REASONS: Final = ("shootout", "penalty", "modelled")  # the rows xGOT scores
RULED_OUT: Final = ("coarse_grid",)  # left out by a PM ruling: off the reconciliation's base
RECONCILE_TOL: Final[float] = 0.01
AUC_MIN: Final[float] = 0.80
ECE_MAX: Final[float] = 0.02
MIN_SHOTS: Final[tuple[int, ...]] = (20, 40, 80)
FLOOR_REPS: Final[int] = 200
KINDS: Final = ("open_play", "penalty", "shootout")
KIND: Final = pl.Enum(KINDS)
SPLIT: Final = pl.Enum(("dev", "holdout"))
_IDS: Final = ("shot_id", "match_id", "player_id", "competition_id", "season_id")
# lower edges of the cards' |y| calibration bands, metres; the last band starts at the post's
# inner edge, and points recorded outside the frame get a row of their own
Y_BANDS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.0, 3.3, features.INNER_Y)
# the band rule and its minimum are metrics.BAND_SDS / metrics.MIN_JUDGED_SHOTS (PM MS6), read
# when called, so the xGOT, GSAx and xG hold-out level rules stay one rule
CHECKS_MILESTONES: Final[str] = "MS5, MS6"
CHECKS_TITLE: Final[str] = f"Milestone checks ({CHECKS_MILESTONES})"


# ---- P2.2 population and reconciliation ----------------------------------------------------


@dataclass(frozen=True)
class Reconciliation:
    """C3 on-target shots against the rows xGOT scored.

    Attributes
    ----------
    c3_shots, c3_goals : int
        C3 on-target shots and their goals.
    shots, goals : int
        On-target rows of the predictions (the rows actually scored) and their goals.
    unscored : int
        Rows of a ``POPULATION_REASONS`` reason with no prediction (expected 0).
    rows : tuple of (str, int, int)
        ``(reason, shots, goals)`` per ``RECONCILE_REASONS`` reason, every reason, summing to
        ``c3_shots``.
    """

    c3_shots: int
    c3_goals: int
    shots: int
    goals: int
    unscored: int
    rows: tuple[tuple[str, int, int], ...]

    @property
    def ruled(self) -> tuple[int, int]:
        """``(shots, goals)`` of the ``RULED_OUT`` reasons: left out by a PM ruling, so the
        reconciliation compares the scored rows with C3 less them."""
        return (sum(n for r, n, _ in self.rows if r in RULED_OUT),
                sum(g for r, _, g in self.rows if r in RULED_OUT))  # fmt: skip

    @property
    def base_shots(self) -> int:
        """C3 on-target shots less the ruled-out ones."""
        return self.c3_shots - self.ruled[0]

    @property
    def base_goals(self) -> int:
        """C3 on-target goals less the ruled-out ones."""
        return self.c3_goals - self.ruled[1]

    @property
    def shots_gap(self) -> float:
        """``(shots − base_shots) / base_shots``."""
        return (self.shots - self.base_shots) / self.base_shots

    @property
    def goals_gap(self) -> float:
        """``(goals − base_goals) / base_goals``; 0 with no goal on either side."""
        if self.base_goals == 0:
            return 0.0 if self.goals == 0 else math.inf
        return (self.goals - self.base_goals) / self.base_goals


def _keyed(frame: pl.DataFrame, what: str, columns: Sequence[str]) -> None:
    if missing := [c for c in columns if c not in frame.columns]:
        raise ValueError(f"reconcile: {what} lacks column(s) {missing}")
    ids = frame["shot_id"]
    if ids.null_count() or ids.n_unique() != frame.height:
        raise ValueError(f"reconcile: {what} has a null or duplicate shot_id")


def reconcile(
    c3: pl.DataFrame,
    exclusions: pl.DataFrame,
    unresolved: pl.Series,
    pred: pl.DataFrame,
    *,
    coarse: pl.Series | None = None,
) -> Reconciliation:
    """Partition C3's on-target shots by first matching reason and count the scored rows.

    Parameters
    ----------
    c3 : polars.DataFrame
        The snapshot's shots table (``shot_id``, ``on_target``, ``is_goal``).
    exclusions : polars.DataFrame
        ``XgData.exclusions``: one ``reason`` per excluded shot.
    unresolved : polars.Series
        The ``shot_id`` values the provider's population dropped (WhoScored: an unresolved
        assisting event); empty for StatsBomb.
    pred : polars.DataFrame
        :func:`predictions` (``shot_id``, ``on_target``, ``is_goal``, ``kind``).
    coarse : polars.Series, optional
        The ``match_id`` values of coarse-grid matches (:func:`coarse_matches`): their
        on-target shots are ``coarse_grid``, ahead of every other reason, and never scored.
        ``c3`` then needs ``match_id``. None or empty: no such match.

    Returns
    -------
    Reconciliation

    Raises
    ------
    ValueError
        A missing column; a null or duplicate ``shot_id``; no C3 on-target shot; an exclusion
        reason outside ``RECONCILE_REASONS``; an on-target prediction row that is not a C3
        on-target shot of a ``POPULATION_REASONS`` reason (counted: an excluded shot scored is a
        population bug, never reconciled).

    Examples
    --------
    >>> c3 = pl.DataFrame({"shot_id": ["sb:1", "sb:2", "sb:3", "sb:4"],
    ...                    "on_target": [True, True, True, False],
    ...                    "is_goal": [True, False, True, False]})
    >>> log = pl.DataFrame({"shot_id": ["sb:3"], "reason": ["direct_corner"]})
    >>> pred = pl.DataFrame({"shot_id": ["sb:1", "sb:2", "sb:4"], "on_target": [True, True, False],
    ...                      "is_goal": [True, False, False], "kind": ["open_play"] * 3})
    >>> r = reconcile(c3, log, pl.Series([], dtype=pl.String), pred)
    >>> (r.c3_shots, r.shots, r.unscored), [row for row in r.rows if row[1]]
    ((3, 2, 0), [('direct_corner', 1, 1), ('modelled', 2, 1)])
    """
    ids = pl.Series([], dtype=pl.String) if coarse is None else coarse.cast(pl.String)
    grid = ids.len() > 0
    _keyed(c3, "c3", ("shot_id", "on_target", "is_goal", *(("match_id",) if grid else ())))
    _keyed(exclusions, "exclusions", ("shot_id", "reason"))
    _keyed(pred, "pred", ("shot_id", "on_target", "is_goal", "kind"))
    in_grid = pl.col("match_id").is_in(ids.implode()) if grid else pl.lit(False)
    on = c3.filter(pl.col("on_target")).select("shot_id", "is_goal", coarse_grid=in_grid)
    if on.height == 0:
        raise ValueError("reconcile: C3 has no on-target shot")
    log = exclusions.select("shot_id", reason=pl.col("reason").cast(pl.String))
    tagged = on.join(log, on="shot_id", how="left", maintain_order="left").with_columns(
        reason=pl.when(pl.col("coarse_grid"))
        .then(pl.lit("coarse_grid"))
        .when(pl.col("reason").is_not_null())
        .then(pl.col("reason"))
        .when(pl.col("shot_id").is_in(unresolved.cast(pl.String).implode()))
        .then(pl.lit("unresolved_assist"))
        .otherwise(pl.lit("modelled"))
    )
    rows = tuple(
        (r, (g := tagged.filter(pl.col("reason") == r)).height, int(g["is_goal"].sum()))
        for r in RECONCILE_REASONS
    )
    if sum(n for _, n, _ in rows) != on.height:
        other = sorted(set(tagged["reason"]) - set(RECONCILE_REASONS))
        raise ValueError(f"reconcile: exclusion reason(s) {other} are not in RECONCILE_REASONS")
    scored = pred.filter(pl.col("on_target")).select("shot_id", "is_goal")
    wanted = tagged.filter(pl.col("reason").is_in(POPULATION_REASONS))
    if stray := scored.join(wanted, on="shot_id", how="anti").height:
        raise ValueError(
            f"reconcile: {stray} on-target prediction row(s) are not C3 on-target shots of the "
            f"xGOT population {POPULATION_REASONS}"
        )
    return Reconciliation(
        c3_shots=on.height,
        c3_goals=int(on["is_goal"].sum()),
        shots=scored.height,
        goals=int(scored["is_goal"].sum()),
        unscored=wanted.join(scored, on="shot_id", how="anti").height,
        rows=rows,
    )


# ---- the xG input --------------------------------------------------------------------------


def keeper_neutral(spec: ModelSpec) -> ModelSpec:
    """``spec`` without its ``KEEPER_POSITION`` features (PM X1); ``spec`` itself without any.

    A GBM's sign map is restricted to the kept features. Returning the same object for a spec
    that reads no keeper feature keeps its cache entry shared with the xG run.

    Examples
    --------
    >>> ks = keeper_neutral(models.GBM)
    >>> ks.name, [f for f in models.GBM.features if f not in ks.features]
    ('xg-gbm-keeper-neutral', ['open_goal', 'gk_in_cone', 'gk_absent', 'gk_to_goal_m', \
'gk_to_shot_m', 'gk_off_line_m'])
    >>> keeper_neutral(v1.GBM) is v1.GBM
    True
    """
    kept = tuple(f for f in spec.features if f not in features.KEEPER_POSITION)
    if kept == spec.features:
        return spec
    make = spec.make
    if isinstance(model := make(), gbm.MonotoneGBM):
        params = dict(model.get_params())
        params["monotone"] = {k: s for k, s in params["monotone"].items() if k in kept}
        make = functools.partial(gbm.MonotoneGBM, **params)
    return dataclasses.replace(
        spec,
        name=f"{spec.name}-keeper-neutral",
        title=f"{spec.title} without keeper position",
        features=kept,
        make=make,
    )


@dataclass(frozen=True, eq=False)
class XgInput:
    """The xG population with the cross-fitted xG, and its on-target part with its folds.

    Attributes
    ----------
    dev, held : polars.DataFrame
        The xG population's dev and hold-out shots with ``xg`` and ``xg_logit`` (Float64).
    on_dev, on_held : polars.DataFrame
        Their on-target rows, in the same order.
    folds : tuple
        The xG folds restricted to ``on_dev`` (``evaluation.restrict_folds``).
    spec_name : str
        The xG spec the input comes from.
    """

    dev: pl.DataFrame
    held: pl.DataFrame
    on_dev: pl.DataFrame
    on_held: pl.DataFrame
    folds: Folds
    spec_name: str


def xg_input(ev: Evaluation, dev: pl.DataFrame, held: pl.DataFrame, folds: Folds) -> XgInput:
    """Add the cross-fitted xG of ``ev`` to the frames and restrict them to on-target shots.

    Dev rows read ``ev.oof`` (out of fold), hold-out rows ``ev.holdout_p`` (the fit on all of
    dev). An xGOT test row in fold k so reads the xG of the fit that never saw fold k.

    Raises
    ------
    ValueError
        ``folds`` not a partition of ``dev``; ``ev`` scored on other folds (the fold index
        named); prediction counts that differ from ``dev`` / ``held``; a NaN prediction;
        ``on_target`` missing or null.

    Examples
    --------
    >>> inp = xg_input(xg_ev, dev, held, folds)  # doctest: +SKIP
    >>> bool(inp.on_dev["on_target"].all()), len(inp.folds) == len(folds)  # doctest: +SKIP
    (True, True)
    """
    evaluation.check_partition(folds, dev.height)
    if len(ev.fold_test) != len(folds):
        raise ValueError(f"{ev.spec.name}: {len(ev.fold_test)} fold(s), the xGOT run {len(folds)}")
    for i, (test, (_, ours)) in enumerate(zip(ev.fold_test, folds, strict=True)):
        if not np.array_equal(test, ours):
            raise ValueError(f"{ev.spec.name} was scored on other folds (fold {i} differs)")
    out = []
    for name, frame, p in (("dev", dev, ev.oof), ("held", held, ev.holdout_p)):
        if p.shape != (frame.height,):
            raise ValueError(
                f"{ev.spec.name}: {p.size} prediction(s) for the {frame.height} {name}"
            )
        if bad := int((~np.isfinite(p)).sum()):
            raise ValueError(f"{ev.spec.name}: {bad} NaN or infinite prediction(s) in {name}")
        if "on_target" not in frame.columns or frame["on_target"].null_count():
            raise ValueError(f"xg_input: {name} needs a non-null on_target column")
        out.append(frame.with_columns(xg=pl.Series(p), xg_logit=pl.Series(metrics.logit(p))))
    on = out[0]["on_target"].to_numpy()
    return XgInput(
        dev=out[0],
        held=out[1],
        on_dev=out[0].filter(pl.col("on_target")),
        on_held=out[1].filter(pl.col("on_target")),
        folds=evaluation.restrict_folds(folds, on),
        spec_name=ev.spec.name,
    )


def coarse_matches(c3: pl.DataFrame) -> pl.Series:
    """The ``match_id`` values of the StatsBomb matches on the coarse (1 yd) recording grid.

    Core's per-match rule (``goalmouth.match_grids``). PM E1 (a), MS6: their end locations sit
    on whole yards, so the pooled xGOT fit over-predicted them by 16.5 % and netted that against
    the fine grid at the near post; they leave the xGOT fit and GSAx (counted, reason
    ``coarse_grid``). Empty for WhoScored and for a match whose grid is unknown.

    Examples
    --------
    >>> c3 = pl.DataFrame({"provider": ["statsbomb"] * 2, "match_id": ["sb:1", "sb:2"],
    ...                    "is_own_goal": [False] * 2,
    ...                    "raw_qualifiers": ['{"location":[100.0,40.0]}',
    ...                                       '{"location":[100.4,40.0]}']})
    >>> coarse_matches(c3).to_list()
    ['sb:1']
    """
    grids = goalmouth.match_grids(c3)
    return grids.filter(pl.col("fidelity") == "coarse")["match_id"]


def drop_matches(inp: XgInput, match_ids: pl.Series) -> XgInput:
    """``inp`` without the rows of ``match_ids``; every fold keeps its rows' assignment.

    Raises
    ------
    ValueError
        From ``evaluation.restrict_folds``: a fold left with no test row.

    Examples
    --------
    >>> drop_matches(inp, coarse_matches(c3)).on_dev.height <= inp.on_dev.height
    ... # doctest: +SKIP
    True
    """
    if match_ids.len() == 0:
        return inp
    out = pl.col("match_id").is_in(match_ids.cast(pl.String).implode())
    keep = ~inp.on_dev.select(out).to_series().to_numpy()
    return dataclasses.replace(
        inp,
        dev=inp.dev.filter(~out),
        held=inp.held.filter(~out),
        on_dev=inp.on_dev.filter(pl.Series(keep)),
        on_held=inp.on_held.filter(~out),
        folds=evaluation.restrict_folds(inp.folds, keep),
    )


def grid_counts(c3: pl.DataFrame, coarse: pl.Series, matches: pl.DataFrame) -> pl.DataFrame:
    """The on-target shots left out as ``coarse_grid``, per competition-season.

    Returns
    -------
    polars.DataFrame
        ``competition_name``, ``season_name`` (String), ``matches``, ``shots``, ``goals``
        (Int64), sorted by shots descending, then names; empty without a coarse match.

    Examples
    --------
    >>> c3 = pl.DataFrame({"match_id": ["sb:1", "sb:1", "sb:2"], "on_target": [True] * 3,
    ...                    "is_goal": [True, False, False]})
    >>> m = pl.DataFrame({"match_id": ["sb:1", "sb:2"], "competition_name": ["A", "B"],
    ...                   "season_name": ["2018", "2019"]})
    >>> grid_counts(c3, pl.Series(["sb:1"]), m).rows()
    [('A', '2018', 1, 2, 1)]
    """
    rows = c3.filter(pl.col("on_target") & pl.col("match_id").is_in(coarse.implode()))
    return (
        rows.join(matches.select("match_id", "competition_name", "season_name"), on="match_id")
        .group_by("competition_name", "season_name")
        .agg(
            matches=pl.col("match_id").n_unique().cast(pl.Int64),
            shots=pl.len().cast(pl.Int64),
            goals=pl.col("is_goal").sum().cast(pl.Int64),
        )
        .sort(pl.col("shots"), "competition_name", "season_name", descending=[True, False, False])
    )


# ---- providers -----------------------------------------------------------------------------


def _ws_population(xg: data.XgData) -> tuple[pl.DataFrame, pl.Series]:
    shots, _ = v1.population(xg)
    return shots, xg.shots.join(shots, on="shot_id", how="anti")["shot_id"]


def _ws_holdout(matches: pl.DataFrame) -> data.Holdout:
    return data.season_holdout(matches, v1.TRAIN_SEASONS, v1.TEST_SEASONS)


@dataclass(frozen=True, eq=False)
class Provider:
    """What differs between providers in an xGOT run; everything else is shared.

    Attributes
    ----------
    key, builder, manifest : str
        ``--provider`` value, snapshot builder and default manifest.
    prefix : str
        Spec and card name prefix (``"xgot-"``, ``"ws-xgot-"``).
    text : ProviderText
        The cards' provider wording, ``filter_labels`` keyed by ``RECONCILE_REASONS``.
    validation : str
        The Validation design paragraph (the split, from constants, and ``XGOT_CROSS_FIT``).
    split : str
        What the cards call the scored-once split (``"hold-out"`` / ``"test"``).
    holdout : callable
        ``holdout(matches) -> data.Holdout``.
    population : callable
        ``population(xg) -> (shots, unresolved shot ids)``: the xG population.
    candidate_specs : callable
        ``candidate_specs() -> (LR steps, GBM spec)`` of the provider's xG run, read when
        called (module globals: tests patch them); ``models.evaluate_candidates`` evaluates
        them with the baselines.
    mouth_note : str or None
        A provider fact about the goal-mouth point the cards state, ``{deflected}`` filled
        with the count of deflected saves; None prints nothing.
    """

    key: str
    builder: str
    manifest: str
    prefix: str
    text: ProviderText
    validation: str
    split: str
    holdout: Callable[[pl.DataFrame], data.Holdout]
    population: Callable[[data.XgData], tuple[pl.DataFrame, pl.Series]]
    candidate_specs: CandidateSpecs
    mouth_note: str | None = None


# the xG input's dev fit as XGOT_CROSS_FIT names it; xgot_cards names a train_exclude there
XG_INPUT_FIT: Final[str] = "the fit on all of dev"
XGOT_CROSS_FIT: Final[str] = (
    "The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in "
    "test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the "
    f"xG of {XG_INPUT_FIT} (train), so no xGOT row sees an xG fitted on itself. The xGOT "
    "training rows of fold k read xG from fits that did see fold k: the standard cross-fitting "
    "dependence, accepted."
)
_FOLDS: Final[str] = (
    f"Grouped {evaluation.N_SPLITS}-fold cross-validation by `match_id` (`GroupKFold`, no "
    "shuffle), the "
    "folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match "
    "has shots in two folds and every model uses the same folds. **Validation** is the mean "
    "over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored "
    "in-sample."
)
_XGOT_LOG: Final[str] = (
    "C3 on-target shots by first matching reason; penalty and shoot-out rows are scored by the "
    "penalty xGOT, modelled rows by the LR and GBM."
)


def _xgot_text(base: ProviderText) -> ProviderText:
    own = {
        "shootout": "shoot-out kick (period 5; penalty xGOT, reported apart)",
        "penalty": "penalty (in-game; penalty xGOT)",
        "unresolved_assist": "unresolved assisting event (left out of the xG population)",
        "coarse_grid": "coarse recording grid (whole-yard locations; the match is left out of "
        "the xGOT fit, PM MS6 E1)",
        "modelled": "modelled (open-play xGOT)",
    }
    labels_ = MappingProxyType({r: own.get(r) or base.label(r) for r in RECONCILE_REASONS})
    return dataclasses.replace(
        base, filter_labels=labels_, benchmark_name=None, benchmark_caveat=None,
        exclusions_log=_XGOT_LOG,
    )  # fmt: skip


PROVIDERS: Final[Mapping[str, Provider]] = MappingProxyType({
    "statsbomb": Provider(
        "statsbomb", "statsbomb", data.DEFAULT_MANIFEST, "xgot-",
        _xgot_text(cards.STATSBOMB_TEXT),
        f"{_FOLDS} **Test** is the hold-out competition-season named under Data, scored once by "
        f"that fit. {XGOT_CROSS_FIT}",
        "hold-out", data.pair_holdout, lambda xg: (xg.shots, pl.Series("shot_id", [], pl.String)),
        lambda: (ablation.STEPS, models.GBM),
        "StatsBomb records a save where the keeper stopped the ball: its `goal_z` is that height, "
        "not the height at the goal line, and a save stopped short of the line has its `goal_y` "
        "projected along the line from the shot location and snapped to the 0.1 yd goal-mouth "
        "lattice of recorded points ({deflected:,} of the on-target saves were deflected, so "
        "their projection starts at the shot, not at the deflection).",
    ),
    "whoscored": Provider(
        "whoscored", v1.BUILDER, v1.DEFAULT_MANIFEST, "ws-xgot-",
        _xgot_text(release.WHOSCORED_TEXT),
        f"{_FOLDS} Train is the {', '.join(v1.TRAIN_SEASONS)} seasons; **Test** is the "
        f"{', '.join(v1.TEST_SEASONS)} season, scored once by that fit and never used to choose. "
        f"{XGOT_CROSS_FIT}",
        "test", _ws_holdout, _ws_population, lambda: (v1.STEPS, v1.GBM),
    ),
})  # fmt: skip


def xg_input_eval(
    provider: Provider,
    xg: data.XgData,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    folds: Folds,
    cache: PredictionCache | None,
) -> tuple[Evaluation, Evaluation]:
    """The xG input of the xGOT and the provider's headline xG, both on ``folds``.

    The candidates are ``models.evaluate_candidates`` of ``provider.candidate_specs()``, the
    set the xG run evaluates, so the headline (``models.select_headline``) is the xG card's
    headline; the input is that spec made keeper-neutral (:func:`keeper_neutral`, one extra fit
    for StatsBomb, through the cache).

    Raises
    ------
    ValueError
        The input reads a keeper-position or an ablation-only feature.

    Examples
    --------
    >>> inp, head = xg_input_eval(PROVIDERS["statsbomb"], xg, dev, held, folds, pc)
    ... # doctest: +SKIP
    >>> inp.spec.name, head.spec.name  # doctest: +SKIP
    ('xg-gbm-keeper-neutral', 'xg-gbm')
    """
    steps, gbm_spec = provider.candidate_specs()
    c = models.evaluate_candidates(
        steps, gbm_spec, dev, held, folds, freeze_frames=xg.freeze_frames, cache=cache
    )
    head, _ = models.select_headline(c.lrs, c.gbm, dev, held)
    spec = keeper_neutral(head.spec) if isinstance(head.spec, ModelSpec) else None
    if spec is None or spec is head.spec:
        inp = head
    else:
        inp = evaluate(spec, dev, held, folds, freeze_frames=xg.freeze_frames, cache=cache)
    feats = inp.spec.features if isinstance(inp.spec, ModelSpec) else ()
    features.check_keeper_neutral(feats)
    if features.uses_ablation_only(feats):
        raise ValueError(f"{inp.spec.name}: an ablation cannot be the xGOT's xG input")
    return inp, head


# ---- P2.3 / P2.4 models --------------------------------------------------------------------

LR_FEATURES: Final[tuple[str, ...]] = (
    features.XG_INPUT,
    *("mouth_y_abs", *(f"mouth_y_abs_rcs{j}" for j in (1, 2, 3))),
    "mouth_y_past",  # not mouth_out: constant on WhoScored, so an unpenalised LR cannot fit it
    *("mouth_z", *(f"mouth_z_rcs{j}" for j in (1, 2, 3))),
    "angle_x_mouth_y",
    "angle_x_mouth_z",
    "mouth_far",
)
# MS6: placement carries no monotone constraint; distance to the post (3.66 − |y|) and to the
# bar (2.44 − z) are order reversals of mouth_y_abs / mouth_z, which a tree splits identically
GBM_FEATURES: Final[tuple[str, ...]] = (
    features.XG_INPUT, "mouth_y_abs", "mouth_out", "mouth_z", "far_side", "angle_rad",
)  # fmt: skip
GBM_SIGNS: Final[Mapping[str, int]] = MappingProxyType({features.XG_INPUT: 1})
PENALTY_FEATURES: Final[tuple[str, ...]] = ("mouth_y_abs", "mouth_z", "mouth_yz")


def xgot_specs(prefix: str, xg_name: str) -> tuple[ModelSpec, ...]:
    """The xGOT specs: base rate, xG only, LR without near / far, LR, GBM and penalty.

    Every spec has the xGOT guard and registry; a spec reading ``xg_logit`` names ``xg_name``
    as its input.

    Examples
    --------
    >>> [s.name for s in xgot_specs("xgot-", "xg-gbm-keeper-neutral")][3:]
    ['xgot-lr', 'xgot-gbm', 'xgot-penalty']
    """
    lr, coef, base = ablation.make_lr, baselines.lr_coefficients, evaluation.BaseRate
    rows = (
        ("base-rate", "Base rate (on target)", (), base, None),
        ("xg-only", "LR xG only", (features.XG_INPUT,), lr, coef),
        ("lr-no-far", "LR xG + placement, no near / far", LR_FEATURES[:-1], lr, coef),
        ("lr", "LR xG + placement", LR_FEATURES, lr, coef),
        ("gbm", "Monotone GBM + Platt", GBM_FEATURES,
         functools.partial(gbm.MonotoneGBM, GBM_SIGNS), gbm.gbm_settings),
        ("penalty", "Penalty xGOT (placement only)", PENALTY_FEATURES, lr, coef),
    )  # fmt: skip
    guard, derived = features.check_xgot_features, features.XGOT_DERIVED
    return tuple(
        ModelSpec(
            f"{prefix}{name}",
            title,
            feats,
            make,
            coefficients,
            guard,
            derived,
            xg_name if features.XG_INPUT in feats else None,
        )  # fmt: skip
        for name, title, feats, make, coefficients in rows
    )


@dataclass(frozen=True, eq=False)
class Kicks:
    """The on-target penalty kicks and the penalty xGOT.

    Attributes
    ----------
    dev, held, shootouts : polars.DataFrame
        In-game dev and hold-out kicks, and shoot-out kicks of every season.
    ev : Evaluation
        The penalty xGOT on the in-game kicks, with its own match folds.
    shootout_p : numpy.ndarray
        The shoot-out kicks scored by a fit on all in-game dev kicks.
    shootout_held : polars.Series
        Boolean, per shoot-out kick: the hold-out's mask (``data.holdout_mask``), the one rule
        that splits every frame.
    """

    dev: pl.DataFrame
    held: pl.DataFrame
    shootouts: pl.DataFrame
    ev: Evaluation
    shootout_p: FloatArray
    shootout_held: pl.Series


def penalty_kicks(
    spec: ModelSpec, penalties: pl.DataFrame, holdout: data.Holdout, cache: PredictionCache | None
) -> Kicks:
    """Fit the penalty xGOT on the in-game dev kicks and score the hold-out and shoot-outs.

    A shoot-out kick's split is the hold-out's mask on the kick itself, never inferred from
    the other rows of its competition-season.

    Raises
    ------
    ValueError
        From ``data.holdout_mask`` and ``evaluate`` (e.g. fewer dev matches than folds).

    Examples
    --------
    >>> kicks = penalty_kicks(specs[5], xg.penalties, ho, pc)  # doctest: +SKIP
    """
    on = penalties.filter(pl.col("on_target"))
    game, shoot = (on.filter(pl.col("reason") == r) for r in ("penalty", "shootout"))
    held = data.holdout_mask(game, holdout, "kicks")
    dev_k, held_k = game.filter(~held), game.filter(held)
    ev = evaluate(spec, dev_k, held_k, evaluation.cv_folds(dev_k), cache=cache)
    p = evaluation.fit_predict(spec, dev_k, shoot) if shoot.height else np.empty(0)
    return Kicks(dev_k, held_k, shoot, ev, p, data.holdout_mask(shoot, holdout, "shoot-out kicks"))


# ---- metrics and checks --------------------------------------------------------------------


@dataclass(frozen=True)
class EceStats:
    """ECE of one evaluation: CV fold mean, max and pooled, the hold-out and its noise floor."""

    mean: float
    max: float
    pooled: float
    held: float
    floor: float


def ece_stats(ev: Evaluation, y_dev: BoolArray, y_held: BoolArray) -> EceStats:
    """The ECE numbers of the cards and the checks (``metrics.ECE_BINS`` equal-count bins).

    Examples
    --------
    >>> s = ece_stats(head, labels(inp.on_dev), labels(inp.on_held))  # doctest: +SKIP
    >>> s.mean <= s.max  # doctest: +SKIP
    True
    """
    per_fold = [metrics.ece(y_dev[t], ev.oof[t]) for t in ev.fold_test]
    return EceStats(
        float(np.mean(per_fold)),
        max(per_fold),
        metrics.ece(y_dev, ev.oof),
        metrics.ece(y_held, ev.holdout_p),
        metrics.ece_floor(ev.holdout_p, reps=FLOOR_REPS),
    )


def _check(target: str, measured: str, ok: bool) -> Check:
    return Check(target, measured, "met" if ok else "missed")


def band_table(p: FloatArray, dev: pl.DataFrame) -> pl.DataFrame:
    """Calibration by placement band: Σ xGOT against goals per ``|y|`` band of ``dev``.

    The reliability table bins by prediction, so it cannot show a placement the model
    misprices; this table bins by ``mouth_y_abs`` (after rounding, ``Y_BANDS``) with the points
    recorded outside the frame (``mouth_out``) in a row of their own. A band is ``within`` when
    ``|Σ p − goals| <= metrics.BAND_SDS · sd``, ``sd = √Σ p(1 − p)`` (the Poisson-binomial SD
    of the goal count if ``p`` is calibrated): the PM's screening rule, not a proof of
    calibration.

    Parameters
    ----------
    p : numpy.ndarray
        Any model's probabilities on ``dev``, in ``dev`` order (the cards pass ``ev.oof``).
    dev : polars.DataFrame
        On-target shots: ``goal_y``, ``goal_z``, ``on_target`` and ``is_goal``.

    Returns
    -------
    polars.DataFrame
        One row per band, in band order: ``band`` (String), ``shots`` / ``goals`` (Int64),
        ``goal_rate`` / ``mean_xgot`` (Float64, null for an empty band), ``gap`` = Σ p − goals
        and ``sd`` (Float64, 0.0 for an empty band) and ``within`` (Boolean; an empty band is
        within, since 0 <= 0).

    Raises
    ------
    ValueError
        ``p`` not of shape ``(dev.height,)``, not finite or outside [0, 1]; from
        ``features.placement`` (a shot not on target, a null ``goal_y``).

    Examples
    --------
    >>> dev = pl.DataFrame({"goal_y": [34.0, 34.0, 30.8], "goal_z": [1.0, 1.0, 1.0],
    ...                     "on_target": [True] * 3, "is_goal": [True, False, True]})
    >>> t = band_table(np.array([0.5, 0.5, 0.9]), dev)
    >>> t.select("band", "shots", pl.col("gap").round(2), "within").rows()[:4]
    [('[0, 1)', 2, 0.0, True), ('[1, 2)', 0, 0.0, True), ('[2, 3)', 0, 0.0, True), \
('[3, 3.3)', 1, -0.1, True)]
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (dev.height,):
        raise ValueError(f"band_table: {p.size} prediction(s) for the {dev.height} dev row(s)")
    if bad := int((~np.isfinite(p) | (p < 0) | (p > 1)).sum()):
        raise ValueError(f"band_table: {bad} prediction(s) are not finite or outside [0, 1]")
    place = features.placement(dev.select("goal_y", "goal_z", "on_target"))
    ay, out = place["mouth_y_abs"].to_numpy(), place["mouth_out"].to_numpy() > 0
    y = labels(dev)
    masks = [
        (f"[{lo:g}, {hi:g})" if math.isfinite(hi) else f">= {lo:g}", (ay >= lo) & (ay < hi) & ~out)
        for lo, hi in pairwise((*Y_BANDS, math.inf))
    ]
    masks.append((f"outside the frame + {goalmouth.GOAL_TOLERANCE:g} m", out))
    rows = []
    for band, m in masks:
        n, goals = int(m.sum()), int(y[m].sum())
        gap, sd = metrics.poisson_binomial(y[m], p[m])
        rate, mean = (goals / n, float(p[m].mean())) if n else (None, None)
        rows.append((band, n, goals, rate, mean, gap, sd, within(gap, sd)))
    schema = {
        "band": pl.String,
        "shots": pl.Int64,
        "goals": pl.Int64,
        "goal_rate": pl.Float64,
        "mean_xgot": pl.Float64,
        "gap": pl.Float64,
        "sd": pl.Float64,
        "within": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")  # fmt: skip


def within(gap: float, sd: float) -> bool:
    """The band rule (PM, MS6): :func:`metrics.within` (``|gap| <= metrics.BAND_SDS · sd``).

    A thin reader of the one expression behind :func:`band_table`, ``gsax.band_table`` and the
    xG hold-out level rule; a band is judged by it only when :func:`judged`.

    Examples
    --------
    >>> within(-2.0, 1.0), within(2.0001, 1.0), within(0.0, 0.0)
    (True, False, True)
    """
    return metrics.within(gap, sd)


def judged(table: pl.DataFrame) -> pl.Series:
    """Per band of :func:`band_table`: judged by the band rule, i.e. ``shots >=
    metrics.MIN_JUDGED_SHOTS`` (read when called).

    PM (MS6 gsax): the 2-SD rule is a normal approximation, so a band under
    ``metrics.MIN_JUDGED_SHOTS`` shots is printed with its counts and marked not judged.

    Examples
    --------
    >>> judged(pl.DataFrame({"shots": [99, 100]})).to_list()
    [False, True]
    """
    return (table["shots"] >= metrics.MIN_JUDGED_SHOTS).alias("judged")


def _band_check(table: pl.DataFrame) -> Check:
    """Check row 12: the worst judged band by |gap| / sd (inf for sd 0 and gap != 0)."""
    ok = judged(table).to_list()
    ratio = [
        -1.0 if not j else 0.0 if gap == 0 and sd == 0 else math.inf if sd == 0 else abs(gap) / sd
        for gap, sd, j in zip(table["gap"], table["sd"], ok, strict=True)
    ]
    i = max(range(len(ratio)), key=ratio.__getitem__)  # the first band on a tie
    band, gap, sd = table["band"][i], table["gap"][i], table["sd"][i]
    worst = f"{band}: {signed(gap)} (SD {sd:.1f})" if ok[i] else "no band judged"
    return _check(
        f"Σ xGOT − goals within {metrics.BAND_SDS:g} SD (√Σp(1−p)) in every placement band of "
        f"at least {metrics.MIN_JUDGED_SHOTS} shots, dev out of fold",
        f"{worst}; {sum(ok)} of {table.height} bands judged",
        bool((table["within"] | ~judged(table)).all()),
    )


def checks(
    head: Evaluation,
    gbm_gain: Comparison,
    recon: Reconciliation,
    xg_ev: Evaluation,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    split: str,
) -> tuple[Check, ...]:
    """The MS5 and MS6 checks of the headline xGOT model, in the card's order; bounds from
    constants.

    Rows 1-9 can be missed: AUC (CV fold mean, ``split``), ECE (CV fold mean, ``split``),
    calibration slope (CV fold mean, ``split``), the GBM headline rule, reconciliation of
    shots and goals. Rows 10-11 record guards that raise rather than miss: the xG input is
    keeper-neutral and no ablation, and the headline reads no keeper-position feature. Row 12
    (MS6) can be missed: every band of :func:`band_table` of the headline's out-of-fold
    predictions on ``dev`` with at least ``metrics.MIN_JUDGED_SHOTS`` shots is within
    ``metrics.BAND_SDS`` SD; it names the worst judged band.

    Raises
    ------
    ValueError
        The xG input or the headline breaks rows 10-11.

    Examples
    --------
    >>> [c.result for c in checks(head, gg, recon, xg_ev, on_dev, on_held, "hold-out")]
    ... # doctest: +SKIP
    ['met', 'met', 'met', 'met', 'met', 'met', 'met', 'met', 'met', 'met', 'met', 'met']
    """
    xg_feats = xg_ev.spec.features if isinstance(xg_ev.spec, ModelSpec) else ()
    features.check_keeper_neutral(xg_feats)
    if features.uses_ablation_only(xg_feats):
        raise ValueError(f"{xg_ev.spec.name}: an ablation cannot be the xGOT's xG input")
    head_feats = head.spec.features if isinstance(head.spec, ModelSpec) else ()
    features.check_xgot_features(head_feats)
    s = ece_stats(head, labels(dev), labels(held))
    lo, hi = models.SLOPE_RANGE
    band = f"Calibration slope in [{lo:g}, {hi:g}]"
    tol = f"within {RECONCILE_TOL:.0%} (two-sided)"
    less = (
        f" less {recon.ruled[0]:,} coarse-grid (PM E1)",
        f" less {recon.ruled[1]:,} coarse-grid",
    )
    less = less if recon.ruled[0] else ("", "")
    return (
        _check(f"ROC AUC >= {AUC_MIN:.2f}, CV fold mean", fmt(head.cv.auc), head.cv.auc >= AUC_MIN),
        _check(f"ROC AUC >= {AUC_MIN:.2f}, {split}", fmt(head.holdout.auc),
               head.holdout.auc >= AUC_MIN),
        _check(f"ECE < {ECE_MAX:g}, CV fold mean",
               f"{fmt(s.mean)} (max {fmt(s.max)}, pooled {fmt(s.pooled)})", s.mean < ECE_MAX),
        _check(f"ECE < {ECE_MAX:g}, {split}", f"{fmt(s.held)} (noise floor {fmt(s.floor)})",
               s.held < ECE_MAX),
        models.slope_check(f"{band}, CV fold mean", head.cv.cal_slope),
        models.slope_check(f"{band}, {split}", head.holdout.cal_slope),
        _check(f"GBM headline iff its CV gain {gbm_gain.label} has a 95% interval above 0",
               f"CV {gain_cell(gbm_gain.cv)}; headline {head.spec.title}",
               models.is_gbm(head) == (gbm_gain.cv.low > 0)),
        _check(f"On-target shots scored reconcile with C3 (less PM-ruled exclusions) {tol}",
               f"{recon.shots:,} of {recon.c3_shots:,}{less[0]} ({recon.shots_gap:+.3%}); unscored "
               f"{recon.unscored:,}", abs(recon.shots_gap) <= RECONCILE_TOL),
        _check(f"On-target goals scored reconcile with C3 (less PM-ruled exclusions) {tol}",
               f"{recon.goals:,} of {recon.c3_goals:,}{less[1]} ({recon.goals_gap:+.3%})",
               abs(recon.goals_gap) <= RECONCILE_TOL),
        Check(f"xG input {xg_ev.spec.title}: cross-fitted on the same folds, keeper-neutral, not "
              "an ablation (the run raises otherwise)", "guards ran", "met"),
        Check("No keeper-position feature in the headline xGOT (the xGOT guard raises "
              "otherwise)", ", ".join(head_feats) or "none", "met"),
        _band_check(band_table(head.oof, dev)),
    )  # fmt: skip


# ---- P2.5 shooting goals added and the MS6 seam ----------------------------------------------


def _with_xgot(frame: pl.DataFrame, p: FloatArray, what: str) -> pl.Series:
    """``p`` on the on-target rows of ``frame`` (in order), null elsewhere."""
    mask = frame["on_target"].to_numpy()
    if p.shape != (int(mask.sum()),):
        raise ValueError(f"predictions: {p.size} xGOT value(s) for {mask.sum()} on-target {what}")
    out = np.full(frame.height, np.nan)
    out[mask] = p
    return pl.Series("xgot", out).fill_nan(None)


def predictions(
    inp: XgInput, head: Evaluation, kicks: Kicks, matches: pl.DataFrame
) -> pl.DataFrame:
    """One row per shot of the xG population and per on-target kick, with xG and xGOT.

    Built in memory and never written (Rule 1). Every probability is out of sample for its
    row: out of fold on dev, the dev fit on the hold-out, the in-game dev fit on shoot-outs.
    Shoot-out kicks take the split of the hold-out's mask on them (``Kicks.shootout_held``).

    Returns
    -------
    polars.DataFrame
        ``shot_id``, ``match_id``, ``player_id``, ``competition_id``, ``season_id``,
        ``match_date`` (from ``matches``), ``period``, ``seconds_in_period`` (the shot order
        within a match, for ``metrics.split_half(unit="row")``), ``gk_id``, ``kind``
        (``KIND``), ``split`` (``dev`` / ``holdout``), ``is_goal``, ``on_target``, ``outcome``,
        ``open_goal``, ``xg`` (null for kicks) and ``xgot`` (null off target).

    Raises
    ------
    ValueError
        A prediction count that differs from the on-target rows; a match missing from
        ``matches``; a duplicate or null ``shot_id``, a null breaking the rules above or an
        on-target row without ``gk_id`` (self-check).

    Examples
    --------
    >>> pred = predictions(inp, head, kicks, xg.matches)  # doctest: +SKIP
    >>> pred["kind"].dtype.categories.to_list()  # doctest: +SKIP
    ['open_play', 'penalty', 'shootout']
    """
    tail = ("is_goal", "on_target", "outcome", "open_goal")
    cols = (*_IDS, "period", "seconds_in_period", "gk_id", *tail)
    parts: list[tuple[pl.DataFrame, str, bool | pl.Series, pl.Series | None, pl.Series]] = [
        (inp.dev, "open_play", False, inp.dev["xg"], _with_xgot(inp.dev, head.oof, "dev shots")),
        (inp.held, "open_play", True, inp.held["xg"],
         _with_xgot(inp.held, head.holdout_p, "hold-out shots")),
        (kicks.dev, "penalty", False, None, _with_xgot(kicks.dev, kicks.ev.oof, "dev kicks")),
        (kicks.held, "penalty", True, None,
         _with_xgot(kicks.held, kicks.ev.holdout_p, "hold-out kicks")),
        (kicks.shootouts, "shootout", kicks.shootout_held, None,
         _with_xgot(kicks.shootouts, kicks.shootout_p, "shoot-out kicks")),
    ]  # fmt: skip
    frames = [
        frame.select(cols).with_columns(
            kind=pl.lit(kind, dtype=KIND),
            split=pl.when(pl.lit(held) if isinstance(held, bool) else held)
            .then(pl.lit("holdout"))
            .otherwise(pl.lit("dev"))
            .cast(SPLIT),
            xg=pl.lit(None, pl.Float64) if xg is None else xg.cast(pl.Float64),
            xgot=xgot,
        )
        for frame, kind, held, xg, xgot in parts
    ]
    pred = (
        pl.concat(frames)
        .join(matches.select("match_id", "match_date"), on="match_id", how="left",
              validate="m:1", maintain_order="left")
        .select(*_IDS, "match_date", "period", "seconds_in_period", "gk_id", "kind", "split",
                *tail, "xg", "xgot")
    )  # fmt: skip
    if missing := pred["match_date"].null_count():
        raise ValueError(f"predictions: {missing} row(s) have no match in the matches table")
    ids, play = pred["shot_id"], pl.col("kind") == "open_play"
    if ids.null_count() or ids.n_unique() != pred.height:
        raise ValueError("predictions: a null or duplicate shot_id")
    bad = pred.filter(
        (pl.col("xg").is_null() == play)
        | (pl.col("xgot").is_null() != (play & ~pl.col("on_target")))
        | (pl.col("on_target") & pl.col("gk_id").is_null())
    ).height
    if bad:
        raise ValueError(f"predictions: {bad} row(s) break the null rules (xg, xgot, gk_id)")
    return pred


def sga(pred: pl.DataFrame) -> pl.DataFrame:
    """The open-play rows with ``sga = xgot − xg`` (0 xGOT off target) and ``gax = goal − xg``.

    Both subtract the same ``xg`` column, the xGOT's input, so SGA never mixes two xG models.

    Examples
    --------
    >>> pred = pl.DataFrame({"kind": ["open_play", "open_play", "penalty"],
    ...                      "is_goal": [True, False, True], "xg": [0.3, 0.2, None],
    ...                      "xgot": [0.6, None, 0.8]})
    >>> [round(v, 2) for v in sga(pred)["sga"]], [round(v, 2) for v in sga(pred)["gax"]]
    ([0.3, -0.2], [0.7, -0.2])
    """
    return pred.filter(pl.col("kind") == "open_play").with_columns(
        sga=pl.col("xgot").fill_null(0.0) - pl.col("xg"),
        gax=pl.col("is_goal").cast(pl.Float64) - pl.col("xg"),
    )


# ---- fit -------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class XgotFit:
    """Everything :func:`fit` computed for one provider; the cards render from it alone.

    Attributes
    ----------
    provider : str
        The key of :data:`PROVIDERS`.
    setup : Provider
        ``PROVIDERS[provider]`` as read when :func:`fit` ran.
    xg, holdout : data.XgData, data.Holdout
        The loaded snapshot and its scored-once split.
    inp : XgInput
        The xG population with the cross-fitted xG and its on-target part.
    xg_input, xg_headline : Evaluation
        The xG input (keeper-neutral) and the provider's headline xG.
    evaluations : tuple of Evaluation
        Base rate, xG only, LR without near / far, LR, GBM.
    head : Evaluation
        The headline xGOT.
    gbm_gain, far_gain : Comparison
        The GBM over the LR, and the LR over the LR without near / far.
    kicks : Kicks
        The on-target kicks and the penalty xGOT.
    penalty_base : Evaluation
        The base rate on the in-game kicks.
    shootouts : Scores or None
        The shoot-out kicks' scores (None without both outcomes).
    recon : Reconciliation
    grid : polars.DataFrame
        :func:`grid_counts`: the on-target shots of coarse-grid matches left out (PM E1 (a)),
        per competition-season; empty for WhoScored.
    coarse : polars.Series
        :func:`coarse_matches`: the ``match_id`` values left out of ``inp`` after the xG input
        fit (empty for WhoScored).
    frames : tuple of polars.DataFrame
        The xG population's dev and hold-out shots before :func:`drop_matches`: the frames the
        xG input is fitted on, coarse-grid matches included (R1); ``season_out.season_out``
        refits on them.
    checks : tuple of Check
    split_half : tuple
        (min shots, blocked shots left in, SGA, goals − xG) rows.
    predictions : polars.DataFrame
        :func:`predictions`, in memory only.
    cache : PredictionCache or None
        The cache read and written (its ``hits`` / ``written``), None without one.
    """

    provider: str
    setup: Provider
    xg: data.XgData
    holdout: data.Holdout
    inp: XgInput
    xg_input: Evaluation
    xg_headline: Evaluation
    evaluations: tuple[Evaluation, ...]
    head: Evaluation
    gbm_gain: Comparison
    far_gain: Comparison
    kicks: Kicks
    penalty_base: Evaluation
    shootouts: Scores | None
    recon: Reconciliation
    grid: pl.DataFrame
    coarse: pl.Series
    frames: tuple[pl.DataFrame, pl.DataFrame]
    checks: tuple[Check, ...]
    split_half: tuple[tuple[int, bool, SplitHalf, SplitHalf], ...]
    predictions: pl.DataFrame
    cache: PredictionCache | None

    @property
    def penalty(self) -> Evaluation:
        """The penalty xGOT (``kicks.ev``)."""
        return self.kicks.ev


def fit(provider: str, *, cache: StrPath | None) -> XgotFit:
    """Evaluate the xGOT models of one provider: no write, no print.

    Parameters
    ----------
    provider : str
        A key of :data:`PROVIDERS` (read when called).
    cache : str or path or None
        Prediction cache root, the only place this writes; ``None`` fits everything and
        writes nothing.

    Returns
    -------
    XgotFit

    Raises
    ------
    ValueError
        An unknown provider; a StatsBomb snapshot whose ``goal_y`` is off the goal-mouth
        lattice (built before the MS6 snap: rebuild it); from the loader, the fits, the guards
        and the self-checks.

    Examples
    --------
    >>> fit("statsbomb", cache="data/cache").head.spec.name  # doctest: +SKIP
    'xgot-gbm'
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
    p = PROVIDERS[provider]
    xg = data.load_xg_data(p.manifest, builder=p.builder)
    if problems := goalmouth.lattice_problems(xg.shots):
        raise ValueError(
            f"{p.manifest}: {problems[0]}; the snapshot predates the goal-mouth snap, rebuild it "
            "with make snapshot OVERWRITE=--overwrite"
        )
    shots, unresolved = p.population(xg)
    ho = p.holdout(xg.matches)
    dev, held = data.split_holdout(shots, ho)
    folds = evaluation.cv_folds(dev)
    pc = None if cache is None else PredictionCache(cache, xg.manifest.data_digest)
    xg_ev, xg_head = xg_input_eval(p, xg, dev, held, folds, pc)
    c3 = snapshot.load_snapshot(p.manifest, tables=("shots",))["shots"]
    coarse = coarse_matches(c3)
    inp = drop_matches(xg_input(xg_ev, dev, held, folds), coarse)
    on_dev, on_held = inp.on_dev, inp.on_held
    by = {s.name.removeprefix(p.prefix): s for s in xgot_specs(p.prefix, inp.spec_name)}
    names = ("base-rate", "xg-only", "lr-no-far", "lr", "gbm")
    evs = tuple(evaluate(by[n], on_dev, on_held, inp.folds, cache=pc) for n in names)
    head, gbm_gain = models.headline(evs[3], evs[4], on_dev, on_held)
    far_gain = evaluation.compare(
        "`mouth_far` (near / far post): dropped", evs[2], evs[3], on_dev, on_held
    )
    on_grid = pl.col("match_id").is_in(coarse.implode())
    kicks = penalty_kicks(by["penalty"], xg.penalties.filter(~on_grid), ho, pc)
    pen_base = evaluate(
        dataclasses.replace(by["base-rate"], name=f"{p.prefix}penalty-base-rate",
                            title="Penalty base rate (on target)"),
        kicks.dev, kicks.held, evaluation.cv_folds(kicks.dev), cache=pc,
    )  # fmt: skip
    k_shoot = labels(kicks.shootouts)
    shootouts = (
        metrics.score(k_shoot, kicks.shootout_p) if 0 < k_shoot.sum() < k_shoot.size else None
    )
    pred = predictions(inp, head, kicks, xg.matches)
    recon = reconcile(c3, xg.exclusions, unresolved, pred, coarse=coarse)
    run_checks = checks(head, gbm_gain, recon, xg_ev, on_dev, on_held, p.split)
    per_shot = sga(pred)
    halves = tuple(
        (m, blocked,
         metrics.split_half(f, by="player_id", value="sga", min_rows=m, unit="match"),
         metrics.split_half(f, by="player_id", value="gax", min_rows=m, unit="match"))
        for m in MIN_SHOTS
        for blocked in (True, False)
        for f in (per_shot if blocked else per_shot.filter(pl.col("outcome") != "blocked"),)
    )  # fmt: skip
    return XgotFit(
        provider, p, xg, ho, inp, xg_ev, xg_head, evs, head, gbm_gain, far_gain, kicks,
        pen_base, shootouts, recon, grid_counts(c3, coarse, xg.matches), coarse, (dev, held),
        run_checks, halves, pred, pc,
    )  # fmt: skip


if __name__ == "__main__":
    print("fsm: error: run python -m football_shot_models.xgot_cards", file=sys.stderr)
    raise SystemExit(2)
