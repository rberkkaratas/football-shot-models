"""MS5 xGOT / PSxG: the post-shot goal probability of on-target shots, for both providers.

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
a dev shot reads its out-of-fold xG, a hold-out shot the xG of the fit on all of dev, and the
xGOT folds are the xG folds restricted to on-target shots, so no xGOT row sees an xG fitted on
itself.

**Models (P2.3, P2.4).** Every model goes through ``evaluation.evaluate`` with the xGOT guard
(``features.check_xgot_features``) and registry (``features.XGOT_DERIVED``). The LR reads the
xG input, placement splines, the width past the inner post edge and angle × placement terms
(near / far post included); the GBM is monotone in the xG input and in the distance from the
goal centre up to ``features.INNER_Y`` at a fixed side (beyond it a ball hits the post, so the
width past it and the outside-the-frame flag are free), and is the headline only if its CV gain
over the LR has a match-bootstrap interval above 0. The penalty
xGOT reads placement only, with its own folds; shoot-out kicks are scored by its dev fit.

**Shooting goals added (P2.5)** is xGOT − xG per shot (:func:`sga`), with blocked and
off-target shots at 0 xGOT. It is reported as split-half correlations next to goals − xG
(``metrics.split_half`` by alternating matches): a descriptive measure, not a claim about
finishing skill.

Nothing here writes a row: :func:`predictions` stays in memory (for MS6), and the WhoScored
cards, README block and output hold counts and metrics only. Run ``python -m
football_shot_models.xgot --provider both`` (``make xgot``) after ``make xg-models xg-v1``, so
the xG candidates come from the prediction cache.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import goalmouth, pitch
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
from football_shot_models.cards import CardContext, CardText, ProviderText, fmt, gain_cell
from football_shot_models.evaluation import Comparison, Evaluation, ModelSpec, evaluate, labels
from football_shot_models.metrics import Check, Scores, SplitHalf

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
StrPath = str | os.PathLike[str]
Folds = evaluation.Folds
Candidates = Callable[
    [data.XgData, pl.DataFrame, pl.DataFrame, Folds, PredictionCache | None],
    tuple[tuple[Evaluation, ...], Evaluation],
]

RECONCILE_REASONS: Final = (*data.EXCLUSION_REASONS, "unresolved_assist", "modelled")
POPULATION_REASONS: Final = ("shootout", "penalty", "modelled")  # the rows xGOT scores
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
    def shots_gap(self) -> float:
        """``(shots − c3_shots) / c3_shots``."""
        return (self.shots - self.c3_shots) / self.c3_shots

    @property
    def goals_gap(self) -> float:
        """``(goals − c3_goals) / c3_goals``; 0 with no goal on either side."""
        if self.c3_goals == 0:
            return 0.0 if self.goals == 0 else math.inf
        return (self.goals - self.c3_goals) / self.c3_goals


def _keyed(frame: pl.DataFrame, what: str, columns: Sequence[str]) -> None:
    if missing := [c for c in columns if c not in frame.columns]:
        raise ValueError(f"reconcile: {what} lacks column(s) {missing}")
    ids = frame["shot_id"]
    if ids.null_count() or ids.n_unique() != frame.height:
        raise ValueError(f"reconcile: {what} has a null or duplicate shot_id")


def reconcile(
    c3: pl.DataFrame, exclusions: pl.DataFrame, unresolved: pl.Series, pred: pl.DataFrame
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
    _keyed(c3, "c3", ("shot_id", "on_target", "is_goal"))
    _keyed(exclusions, "exclusions", ("shot_id", "reason"))
    _keyed(pred, "pred", ("shot_id", "on_target", "is_goal", "kind"))
    on = c3.filter(pl.col("on_target")).select("shot_id", "is_goal")
    if on.height == 0:
        raise ValueError("reconcile: C3 has no on-target shot")
    log = exclusions.select("shot_id", reason=pl.col("reason").cast(pl.String))
    tagged = on.join(log, on="shot_id", how="left", maintain_order="left").with_columns(
        reason=pl.when(pl.col("reason").is_not_null())
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


# ---- providers -----------------------------------------------------------------------------


def _candidates(
    steps: Sequence[ModelSpec], gbm_spec: ModelSpec, xg: data.XgData, dev: pl.DataFrame,
    held: pl.DataFrame, folds: Folds, pc: PredictionCache | None,
) -> tuple[tuple[Evaluation, ...], Evaluation]:  # fmt: skip
    """The xG run's LR candidates and GBM, as ``models.run`` / ``v1.run`` evaluate them."""
    ff = xg.freeze_frames  # empty for WhoScored, whose specs read no frame feature
    base = tuple(evaluate(b, dev, held, folds, cache=pc) for b in baselines.BASELINES)
    steps_ = tuple(evaluate(s, dev, held, folds, freeze_frames=ff, cache=pc) for s in steps)
    return (*base, *steps_), evaluate(gbm_spec, dev, held, folds, freeze_frames=ff, cache=pc)


def _sb_candidates(
    xg: data.XgData, dev: pl.DataFrame, held: pl.DataFrame, folds: Folds,
    pc: PredictionCache | None,
) -> tuple[tuple[Evaluation, ...], Evaluation]:  # fmt: skip
    # module globals read when called, so a patch of models.GBM is seen
    return _candidates(ablation.STEPS, models.GBM, xg, dev, held, folds, pc)


def _ws_candidates(
    xg: data.XgData, dev: pl.DataFrame, held: pl.DataFrame, folds: Folds,
    pc: PredictionCache | None,
) -> tuple[tuple[Evaluation, ...], Evaluation]:  # fmt: skip
    return _candidates(v1.STEPS, v1.GBM, xg, dev, held, folds, pc)


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
    candidates : callable
        ``candidates(xg, dev, held, folds, cache) -> (LR candidates, GBM)`` of the xG run.
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
    candidates: Candidates
    mouth_note: str | None = None


XGOT_CROSS_FIT: Final[str] = (
    "The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in "
    "test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the "
    "xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT "
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
PENALTY_FOLDS: Final[str] = (
    f"Grouped {evaluation.N_SPLITS}-fold cross-validation by `match_id` on the in-game dev "
    "kicks (their own "
    "folds); the {split} kicks are scored once by the fit on all of them, and the shoot-out "
    "kicks of every season by that fit too. The penalty xGOT reads no xG input."
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
        _sb_candidates,
        "StatsBomb records a save where the keeper stopped the ball: its `goal_z` is that height, "
        "not the height at the goal line, and a save stopped short of the line has its `goal_y` "
        "projected along the line from the shot location ({deflected:,} of the on-target saves "
        "were deflected, so their projection starts at the shot, not at the deflection).",
    ),
    "whoscored": Provider(
        "whoscored", v1.BUILDER, v1.DEFAULT_MANIFEST, "ws-xgot-",
        _xgot_text(release.WHOSCORED_TEXT),
        f"{_FOLDS} Train is the {', '.join(v1.TRAIN_SEASONS)} seasons; **Test** is the "
        f"{', '.join(v1.TEST_SEASONS)} season, scored once by that fit and never used to choose. "
        f"{XGOT_CROSS_FIT}",
        "test", _ws_holdout, _ws_population, _ws_candidates,
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

    The headline is chosen by ``models.select_headline`` over the provider's candidates; the
    input is that spec made keeper-neutral (:func:`keeper_neutral`, one extra fit for
    StatsBomb, through the cache).

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
    candidates, gbm_ev = provider.candidates(xg, dev, held, folds, cache)
    head, _ = models.select_headline(candidates, gbm_ev, dev, held)
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
GBM_FEATURES: Final[tuple[str, ...]] = (
    features.XG_INPUT, "mouth_y_in", "mouth_y_past", "mouth_out", "mouth_z", "far_side",
    "angle_rad",
)  # fmt: skip
GBM_SIGNS: Final[Mapping[str, int]] = MappingProxyType({features.XG_INPUT: 1, "mouth_y_in": 1})
PENALTY_FEATURES: Final[tuple[str, ...]] = ("mouth_y_abs", "mouth_z", "mouth_yz")
XGOT_MONOTONE_REASONS: Final[Mapping[str, str]] = MappingProxyType({
    features.XG_INPUT: "a higher xG at the same placement never lowers xGOT",
    "mouth_y_in": f"further from the centre, up to {features.INNER_Y:g} m, at the same height "
    "and on the same side (`far_side` fixed) never helps the keeper",
})  # fmt: skip
XGOT_MONOTONE_NOTE: Final[str] = (
    "Each constraint holds along one column with every other column held fixed: `mouth_y_in` "
    "moves the ball towards a post at the same height, on the same side (near or far). Beyond "
    f"{features.INNER_Y:g} m from the centre a ball's centre is within its radius of the post, "
    "so a shot there hits the post or is saved: `mouth_y_past` and `mouth_out` are free."
)


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


def ece_lines(
    ev: Evaluation, y_dev: BoolArray, y_held: BoolArray, *, split: str = "hold-out"
) -> list[str]:
    """The ECE section of every xGOT card: a one-row table and the binning sentence.

    Examples
    --------
    >>> ece_lines(head, y_dev, y_held)[0]  # doctest: +SKIP
    '### Expected calibration error'
    """
    s = ece_stats(ev, y_dev, y_held)
    return [
        "### Expected calibration error",
        "",
        f"ECE ({metrics.ECE_BINS} equal-count bins): the shot-weighted mean of |mean prediction "
        "− goal rate| over the bins of the reliability table. CV is "
        "judged on the fold mean, with the worst fold and the pooled out-of-fold value beside "
        f"it; the {split} value has beside it its noise floor, the mean ECE of outcomes drawn "
        f"from its own predictions ({FLOOR_REPS} draws, seed {metrics.SEED}), which a perfectly "
        "calibrated model reaches by chance.",
        "",
        f"| CV fold mean | CV max | CV pooled | {(cap := split.capitalize())} | {cap} floor |",
        "|---|---|---|---|---|",
        f"| {fmt(s.mean)} | {fmt(s.max)} | {fmt(s.pooled)} | {fmt(s.held)} | {fmt(s.floor)} |",
        "",
    ]


def _check(target: str, measured: str, ok: bool) -> Check:
    return Check(target, measured, "met" if ok else "missed")


def checks(
    head: Evaluation,
    gbm_gain: Comparison,
    recon: Reconciliation,
    xg_ev: Evaluation,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    split: str,
) -> tuple[Check, ...]:
    """The MS5 checks of the headline xGOT model, in the card's order; bounds from constants.

    Rows 1-9 can be missed: AUC (CV fold mean, ``split``), ECE (CV fold mean, ``split``),
    calibration slope (CV fold mean, ``split``), the GBM headline rule, reconciliation of
    shots and goals. Rows 10-11 record guards that raise rather than miss: the xG input is
    keeper-neutral and no ablation, and the headline reads no keeper-position feature.

    Raises
    ------
    ValueError
        The xG input or the headline breaks rows 10-11.

    Examples
    --------
    >>> [c.result for c in checks(head, gg, recon, xg_ev, on_dev, on_held, "hold-out")]
    ... # doctest: +SKIP
    ['met', 'met', 'met', 'met', 'met', 'met', 'met', 'met', 'met', 'met', 'met']
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
        _check(f"On-target shots scored reconcile with C3 {tol}",
               f"{recon.shots:,} of {recon.c3_shots:,} ({recon.shots_gap:+.3%}); unscored "
               f"{recon.unscored:,}", abs(recon.shots_gap) <= RECONCILE_TOL),
        _check(f"On-target goals scored reconcile with C3 {tol}",
               f"{recon.goals:,} of {recon.c3_goals:,} ({recon.goals_gap:+.3%})",
               abs(recon.goals_gap) <= RECONCILE_TOL),
        Check(f"xG input {xg_ev.spec.title}: cross-fitted on the same folds, keeper-neutral, not "
              "an ablation (the run raises otherwise)", "guards ran", "met"),
        Check("No keeper-position feature in the headline xGOT (the xGOT guard raises "
              "otherwise)", ", ".join(head_feats) or "none", "met"),
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


# ---- cards ---------------------------------------------------------------------------------

_PLACED: Final[str] = (
    f"clipped to the frame + {goalmouth.GOAL_TOLERANCE:g} m and rounded to "
    f"{features.PLACEMENT_STEP:g} m"
)
_SRC: Final[str] = "`goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only"
FEATURE_ROWS: Final[Mapping[str, str]] = MappingProxyType({
    features.XG_INPUT: "| `xg_logit` | Logit of the shot's cross-fitted xG (the xG input, out "
    "of sample for the row) | xG model |",
    "mouth_y_abs": f"| `mouth_y_abs` | Distance from the goal centre across the goal, m, "
    f"{_PLACED} | {_SRC} |",
    "mouth_z": f"| `mouth_z` | Height above the ground, m, {_PLACED} | {_SRC} |",
    **{
        f"{col}_rcs{j + 1}": f"| `{col}_rcs{j + 1}` | Restricted cubic spline term {j + 1} of "
        f"`{col}`, knots {', '.join(f'{k:g}' for k in knots)} m | `{col}` |"
        for col, knots in (("mouth_y_abs", features.MOUTH_Y_KNOTS),
                           ("mouth_z", features.MOUTH_Z_KNOTS))
        for j in range(len(knots) - 2)
    },
    "far_side": "| `far_side` | +1 far post, −1 near post (the ball's side against the shot's), "
    f"0 for a shot within {features.CENTRE_DEAD_ZONE:g} m of the centre line or a ball at the "
    f"centre | `y`, {_SRC} |",
    "mouth_far": "| `mouth_far` | `far_side` × `mouth_y_abs` | `y`, " + _SRC + " |",
    "angle_x_mouth_y": "| `angle_x_mouth_y` | `angle_rad` × `mouth_y_abs` | `angle_rad`, "
    + _SRC + " |",
    "angle_x_mouth_z": "| `angle_x_mouth_z` | `angle_rad` × `mouth_z` | `angle_rad`, "
    + _SRC + " |",
    "mouth_yz": "| `mouth_yz` | `mouth_y_abs` × `mouth_z` | " + _SRC + " |",
    "mouth_y_in": f"| `mouth_y_in` | `min(mouth_y_abs, {features.INNER_Y:g})`: the distance from "
    "the goal centre up to where a ball starts to touch the post, m | `mouth_y_abs` |",
    "mouth_y_past": "| `mouth_y_past` | `mouth_y_abs − mouth_y_in`: how far past that point "
    "the ball crossed, m | `mouth_y_abs` |",
    "mouth_out": f"| `mouth_out` | 1 when the point was recorded beyond the frame + "
    f"{goalmouth.GOAL_TOLERANCE:g} m (before clipping), else 0 | {_SRC} |",
    "angle_rad": ablation.FEATURE_ROWS["angle_rad"],
})  # fmt: skip
INTENDED_USE: Final[str] = (
    "xGOT (post-shot xG) for on-target shots: P(goal) of a shot that reached the goal frame, "
    "from where it crossed the line and its pre-shot xG. Keeper-neutral (no keeper position), "
    "so goals saved against it credit the keeper's positioning too. For research and as the "
    "input of the MS6 goalkeeper models; not a rating of finishers, and not for other "
    "providers' data without re-fitting."
)
TASK: Final[str] = (
    "binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty "
    "xG population (a goal or a keeper save; own goals are never on target)"
)
PENALTY_TASK: Final[str] = (
    "P(goal | on-target penalty) from placement only; fitted on in-game kicks, shoot-out kicks "
    "scored apart"
)
PENALTY_USE: Final[str] = (
    "Penalty xGOT: P(goal) of an on-target penalty from where it crossed the line, kept apart "
    "from the open-play xGOT. For research and as the penalty input of the MS6 goalkeeper "
    "models; not a rating of takers or keepers on its own."
)
TARGET: Final[str] = (
    "`is_goal` of an on-target shot: the ball went in rather than being saved. Blocked shots "
    "(clearances off the line included), woodwork and misses are not on target (C3 "
    "`on_target`), so they are not in the population."
)
CHANGELOG: Final[tuple[str, ...]] = (
    "v1 (MS5, P2.2–P2.5): first version.",
    "v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and "
    "`mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the "
    "scored-once split's level gap; penalty calibration judged bound by bound.",
)


def _lr_model(ev: Evaluation, c: CardContext) -> list[str]:
    lr = ablation.make_lr()[-1]
    return [
        f"`make_pipeline(StandardScaler(), LogisticRegression(C={lr.C}, solver={lr.solver!r}, "
        f"tol={lr.tol:g}, max_iter={lr.max_iter}))`: unpenalised logistic regression, no class "
        "weights, no calibration step; the scaler is fitted on the training rows of each fit. "
        f"scikit-learn {c.sklearn_version}.",
        "",
        "Coefficients of the fit on all of dev (natural units: log-odds per unit of the feature).",
        "",
        "| Term | Standardised | Natural units |",
        "|---|---|---|",
        *(f"| {term} | {fmt(std)} | {fmt(nat)} |" for term, std, nat in ev.coefficients),
    ]


def _feature_table(names: Sequence[str]) -> tuple[str, ...]:
    return (
        "| Feature | Definition | Source field(s) |",
        "|---|---|---|",
        *(FEATURE_ROWS[f] for f in names),
    )


def _limits(p: Provider, on: pl.DataFrame, *, flagged: bool = True) -> tuple[str, ...]:
    """Limitations computed from the on-target rows ``on`` (dev and the scored-once split);
    ``flagged`` when the card's models read ``mouth_out``."""
    gy, gz = pitch.to_goal_frame(on["goal_y"].to_numpy(), on["goal_z"].to_numpy())
    clipped = int(features.clipped(gy, gz).sum())
    dead = int(features.dead_zone(on["y"].to_numpy()).sum())
    deflected = on.filter(pl.col("deflected").fill_null(False) & ~pl.col("is_goal")).height
    psxg = on["provider_psxg"].null_count()
    lines = [
        f"{clipped:,} of {on.height:,} on-target shots ({p.split} included) are recorded "
        f"beyond the frame + {goalmouth.GOAL_TOLERANCE:g} m and are clipped to its edge"
        + (
            " and flagged by `mouth_out`, since where they would have crossed the line is not "
            "known; "
            if flagged
            else "; "
        )
        + f"{dead:,} are struck within {features.CENTRE_DEAD_ZONE:g} m of the centre line, so they "
        "have no near or far post.",
        f"`provider_psxg` is null on {psxg:,} of {on.height:,} on-target shots, so there is no "
        "provider xGOT benchmark.",
    ]
    if p.mouth_note is not None:
        lines.append(p.mouth_note.format(deflected=deflected))
    return tuple(lines)


def band_lines(ev: Evaluation, dev: pl.DataFrame) -> list[str]:
    """Calibration by placement band: dev out-of-fold xGOT against goals per |y| band.

    The reliability table bins by prediction, so it cannot show a placement the model
    misprices; this table bins by ``mouth_y_abs`` (rounded, ``Y_BANDS``) with the points
    recorded outside the frame (``mouth_out``) in a row of their own.

    Examples
    --------
    >>> band_lines(head, inp.on_dev)[0]  # doctest: +SKIP
    '### Calibration by placement band'
    """
    place = features.placement(dev.select("goal_y", "goal_z", "on_target"))
    ay, out = place["mouth_y_abs"].to_numpy(), place["mouth_out"].to_numpy() > 0
    y, p = labels(dev), ev.oof
    masks = [
        (f"[{lo:g}, {hi:g})" if math.isfinite(hi) else f">= {lo:g}", (ay >= lo) & (ay < hi) & ~out)
        for lo, hi in pairwise((*Y_BANDS, math.inf))
    ]
    masks.append((f"outside the frame + {goalmouth.GOAL_TOLERANCE:g} m", out))
    rows = []
    for label, m in masks:
        n, goals = int(m.sum()), int(y[m].sum())
        rate, mean = (goals / n, float(p[m].mean())) if n else (None, None)
        gap = f"{float(p[m].sum()) - goals:+.1f}" if n else "n/a"
        rows.append(f"| {label} | {n:,} | {goals:,} | {fmt(rate)} | {fmt(mean)} | {gap} |")
    return [
        "### Calibration by placement band",
        "",
        "Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, "
        f"after rounding); a ball beyond {features.INNER_Y:g} m touches the post. The "
        "reliability table bins by prediction, so it cannot show a placement the model "
        "misprices.",
        "",
        "| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def _level(ev: Evaluation, split: str) -> str:
    """The scored-once split's level gap, a limitation the ECE does not show."""
    h = ev.holdout
    gap = f"{h.xg_signed_error:+.2%}" if h.goals else "n/a"
    return (
        f"On the {split}, Σ xGOT is {h.xg:,.1f} against {h.goals:,} goals "
        f"({gap}; calibration intercept {fmt(h.cal_intercept)}). A level gap "
        "moves every keeper's goals saved against this xGOT the same way, so a season's "
        "keepers read better or worse as a group."
    )


def _text(
    role: str, feats: tuple[str, ...], model: Callable[[Evaluation, CardContext], Sequence[str]],
    limits: tuple[str, ...], *, task: str = TASK,
) -> CardText:  # fmt: skip
    return CardText(
        role=role, intended_use=INTENDED_USE, features=feats, model=model, limitations=limits,
        changelog=CHANGELOG, task=task, target=TARGET, best_label="xGOT model",
        checks_title="Milestone checks (MS5)",
        checks_note="Bounds from the PM brief (MS5); AUC >= 0.80 reads \"about 0.80 or better\" "
        "strictly. ECE is judged on the CV fold mean and the point value of the scored-once "
        "split (PM X3).",
        quantity="xGOT",
    )  # fmt: skip


def xgot_context(
    xg: data.XgData,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    recon: Reconciliation,
    *,
    holdout: data.Holdout,
) -> CardContext:
    """``cards.card_context`` with every count taken from the on-target frames passed.

    Examples
    --------
    >>> c = xgot_context(xg, inp.on_dev, inp.on_held, recon, holdout=ho)  # doctest: +SKIP
    >>> c.dev_shots == inp.on_dev.height  # doctest: +SKIP
    True
    """
    both = pl.concat([dev.select("competition_id", "season_id", "gender"),
                      held.select("competition_id", "season_id", "gender")])  # fmt: skip
    return dataclasses.replace(
        cards.card_context(xg, dev, held, holdout),
        competition_seasons=both.select("competition_id", "season_id").n_unique(),
        male_shots=int((both["gender"] == "male").sum()),
        female_shots=int((both["gender"] == "female").sum()),
        exclusions=recon.rows,
    )


def _r(s: SplitHalf) -> str:
    if s.r is None or s.low is None or s.high is None:
        return "n/a"
    return f"{s.r:+.3f} [{s.low:+.3f}, {s.high:+.3f}]"


def _halves(rows: Sequence[tuple[int, bool, SplitHalf, SplitHalf]]) -> list[str]:
    return [
        "### Shooting goals added (descriptive)",
        "",
        "Shooting goals added (SGA) is xGOT − xG per shot, next to goals − xG; both subtract the "
        "same xG, the xG input above. Blocked and off-target shots count as 0 xGOT. Each "
        "player's matches are ordered by date (then match id) and alternated into two halves, "
        "so a match never spans halves; r is the Pearson correlation across players of the "
        "per-shot value in the two halves, with a 95% interval by Fisher z (n/a under 4 "
        "players). A descriptive measure of shots beating their xG on target, not a "
        "finishing-skill claim.",
        "",
        "| Min shots | Blocked shots | Players | SGA per shot: r [95% CI] | Goals − xG per "
        "shot: r [95% CI] |",
        "|---|---|---|---|---|",
        *(
            f"| {m} | {'in' if blocked else 'left out'} | {a.keys:,} | {_r(a)} | {_r(b)} |"
            for m, blocked, a, b in rows
        ),
        "",
    ]


def _recon_lines(recon: Reconciliation) -> list[str]:
    return [
        "### Reconciliation with C3",
        "",
        "Every C3 on-target shot is in one row of the Filters table; the scored rows are the "
        "open-play and penalty xGOT predictions.",
        "",
        "| | Shots | Goals |",
        "|---|---|---|",
        f"| C3 on target | {recon.c3_shots:,} | {recon.c3_goals:,} |",
        f"| Scored by xGOT | {recon.shots:,} | {recon.goals:,} |",
        f"| Difference | {recon.shots_gap:+.3%} | {recon.goals_gap:+.3%} |",
        "",
        f"Population rows without a prediction: {recon.unscored:,}.",
        "",
    ]


# ---- run -----------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class XgotRun:
    """Everything :func:`run` evaluated for one provider.

    ``xg_input`` is the xG input's evaluation (``xg_headline`` the provider's headline xG);
    ``evaluations`` is (base rate, xG only, LR without near / far, LR, GBM); ``split_half``
    holds (min shots, blocked shots left in, SGA, goals − xG) rows; ``cards`` maps card name to
    the Markdown written.
    """

    provider: str
    xg_input: Evaluation
    xg_headline: Evaluation
    evaluations: tuple[Evaluation, ...]
    head: Evaluation
    gbm_gain: Comparison
    far_gain: Comparison
    penalty: Evaluation
    shootouts: Scores | None
    recon: Reconciliation
    checks: tuple[Check, ...]
    split_half: tuple[tuple[int, bool, SplitHalf, SplitHalf], ...]
    predictions: pl.DataFrame
    cards: Mapping[str, str]


def _y(pred: pl.DataFrame, kind: str, split: str) -> BoolArray:
    rows = pred.filter((pl.col("kind") == kind) & (pl.col("split") == split) & pl.col("on_target"))
    y: BoolArray = rows["is_goal"].to_numpy()
    return y


def summary_block(runs: Sequence[XgotRun]) -> str:
    """The README block: every xGOT model per provider, then the headline, penalty and SGA lines.

    Examples
    --------
    >>> print(summary_block([sb_run]))  # doctest: +SKIP
    | Provider | Model | CV AUC | CV ECE | CV cal. slope | ...
    """
    lines = [
        "| Provider | Model | CV AUC | CV ECE | CV cal. slope | Hold-out / test AUC | "
        "Hold-out / test ECE | Hold-out / test cal. slope |",
        "|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for r in runs:
        source = PROVIDERS[r.provider].text.source
        for ev, kind in (*((e, "open_play") for e in r.evaluations), (r.penalty, "penalty")):
            s = ece_stats(ev, _y(r.predictions, kind, "dev"), _y(r.predictions, kind, "holdout"))
            title = f"**{ev.spec.title}**" if ev is r.head else ev.spec.title
            lines.append(
                f"| {source} | {title} | {fmt(ev.cv.auc)} | {fmt(s.mean)} | "
                f"{fmt(ev.cv.cal_slope)} | {fmt(ev.holdout.auc)} | {fmt(s.held)} | "
                f"{fmt(ev.holdout.cal_slope)} |"
            )
        met = sum(c.result == "met" for c in r.checks)
        so = r.shootouts
        sga40 = next(row for row in r.split_half if row[0] == MIN_SHOTS[1] and row[1])
        notes += [
            f"{source}: headline {r.head.spec.title}; xG input {r.xg_input.spec.title} (provider "
            f"headline xG {r.xg_headline.spec.title}); {met} of {len(r.checks)} MS5 checks met; "
            f"{r.recon.shots:,} of {r.recon.c3_shots:,} C3 on-target shots scored "
            f"({r.recon.shots_gap:+.3%}), goals {r.recon.goals_gap:+.3%}.",
            "",
            f"{source}, penalty xGOT (placement only, in-game kicks): CV AUC "
            f"{fmt(r.penalty.cv.auc)}, CV cal. slope {fmt(r.penalty.cv.cal_slope)}; shoot-out "
            + ("kicks: none with both outcomes." if so is None else
               f"kicks scored apart: {so.n:,} kicks, {so.goals:,} goals, Σ xGOT {so.xg:.1f}, log "
               f"loss {fmt(so.log_loss)}."),
            "",
            f"{source}, shooting goals added (descriptive), split-half r at >= {sga40[0]} shots, "
            f"{sga40[2].keys:,} players: SGA per shot {_r(sga40[2])}, goals − xG per shot "
            f"{_r(sga40[3])}.",
            "",
        ]  # fmt: skip
    return "\n".join([*lines, "", *notes[:-1]]) + "\n"


def _penalty_calibration(pen: Evaluation, s: EceStats, split: str) -> str:
    """The penalty card's calibration line: each open-play bound judged on its own."""
    lo, hi = models.SLOPE_RANGE

    def within(v: float | None) -> str:
        return "not estimable" if v is None else "inside" if lo <= v <= hi else "outside"

    slopes = (pen.cv.cal_slope, pen.holdout.cal_slope)
    inside = all(v is not None and lo <= v <= hi for v in slopes) and s.mean < ECE_MAX
    line = (
        f"Against the open-play bounds, each on its own: CV calibration slope "
        f"{fmt(slopes[0])} ({within(slopes[0])} [{lo:g}, {hi:g}]), {split} slope "
        f"{fmt(slopes[1])} ({within(slopes[1])}), CV ECE {fmt(s.mean)} (fold mean; "
        f"{'below' if s.mean < ECE_MAX else 'not below'} {ECE_MAX:g})."
    )
    if inside:
        return line
    return (
        f"{line} Its probabilities are not calibrated to the open-play standard; as a ranking "
        f"of placements its AUC is {fmt(pen.cv.auc)} on CV and {fmt(pen.holdout.auc)} on the "
        f"{split} (0.5 is no ranking)."
    )


def run(provider: str, *, docs_dir: StrPath, cache: StrPath | None) -> XgotRun:
    """Evaluate the xGOT models of one provider, write its three cards and print the results.

    Parameters
    ----------
    provider : str
        A key of :data:`PROVIDERS`.
    docs_dir : str or path
        Where ``<name>.md`` is written.
    cache : str or path, optional
        Prediction cache root; ``None`` fits everything and writes nothing.

    Returns
    -------
    XgotRun

    Raises
    ------
    ValueError
        An unknown provider; from the loader, the fits, the guards and the self-checks.

    Examples
    --------
    >>> run("statsbomb", docs_dir="docs/models", cache="data/cache").head.spec.name
    ... # doctest: +SKIP
    'xgot-gbm'
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
    p = PROVIDERS[provider]
    xg = data.load_xg_data(p.manifest, builder=p.builder)
    shots, unresolved = p.population(xg)
    ho = p.holdout(xg.matches)
    dev, held = data.split_holdout(shots, ho)
    folds = evaluation.cv_folds(dev)
    pc = None if cache is None else PredictionCache(cache, xg.manifest.data_digest)
    xg_ev, xg_head = xg_input_eval(p, xg, dev, held, folds, pc)
    inp = xg_input(xg_ev, dev, held, folds)
    specs = xgot_specs(p.prefix, inp.spec_name)
    on_dev, on_held, y_dev, y_held = (
        inp.on_dev,
        inp.on_held,
        labels(inp.on_dev),
        labels(inp.on_held),
    )
    by = {s.name.removeprefix(p.prefix): s for s in specs}
    names = ("base-rate", "xg-only", "lr-no-far", "lr", "gbm")
    evs = tuple(evaluate(by[n], on_dev, on_held, inp.folds, cache=pc) for n in names)
    head, gbm_gain = models.headline(evs[3], evs[4], on_dev, on_held)
    far_gain = evaluation.compare(
        "`mouth_far` (near / far post): dropped", evs[2], evs[3], on_dev, on_held
    )
    kicks = penalty_kicks(by["penalty"], xg.penalties, ho, pc)
    pen, k_dev, k_held = kicks.ev, labels(kicks.dev), labels(kicks.held)
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
    c3 = snapshot.load_snapshot(p.manifest, tables=("shots",))["shots"]
    recon = reconcile(c3, xg.exclusions, unresolved, pred)
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

    # cards: LR and GBM on the open-play on-target shots, the penalty xGOT on the kicks
    limits = _limits(p, pl.concat([on_dev, on_held]))
    lead = (f"xG input: {xg_ev.spec.title}, cross-fitted; provider headline xG: "
            f"{xg_head.spec.title}.", "")  # fmt: skip
    texts = (
        _text("P2.3: logistic regression of the goal on the logit of the cross-fitted xG, "
              "placement splines, the width past the post's inner edge, angle × placement and "
              "the near / far post.",
              (*lead, *_feature_table(LR_FEATURES)), _lr_model,
              (*limits, _level(evs[3], p.split))),
        _text("P2.4: monotone gradient boosting on the xG input, placement, side and angle, "
              "tuned and Platt-calibrated inside each fit by grouped CV.",
              (*lead, *models.gbm_features(evs[4], FEATURE_ROWS, XGOT_MONOTONE_REASONS,
                                           XGOT_MONOTONE_NOTE)),
              models.gbm_model, (*limits, _level(evs[4], p.split))),
    )  # fmt: skip
    ctx = xgot_context(xg, on_dev, on_held, recon, holdout=ho)
    written = {
        ev.spec.name: cards.render_card(
            ev, text, evaluations=evs, benchmark=None, dev=on_dev, context=ctx, provider=p.text,
            validation=p.validation, thresholds=(), comparisons=(gain,),
            verdict=models.verdict(ev, head, gbm_gain, split=p.split),
            checks=run_checks if ev is head else (),
            extra=[*ece_lines(ev, y_dev, y_held, split=p.split), *band_lines(ev, on_dev),
                   *((*_recon_lines(recon), *_halves(halves)) if ev is head else ())],
        )
        for ev, text, gain in ((evs[3], texts[0], far_gain), (evs[4], texts[1], gbm_gain))
    }  # fmt: skip
    calib = _penalty_calibration(pen, ece_stats(pen, k_dev, k_held), p.split)
    shoot = [
        "### Shoot-out kicks (reported apart)", "",
        f"{k_shoot.size:,} on-target shoot-out kicks ({int(k_shoot.sum()):,} goals), scored by "
        "the fit on all in-game dev kicks and never pooled with them: Σ xGOT "
        f"{float(kicks.shootout_p.sum()):.1f}, log loss "
        f"{fmt(None if shootouts is None else shootouts.log_loss)}.", "",
    ]  # fmt: skip
    pen_text = dataclasses.replace(_text(
        "P2.4: penalty xGOT, a logistic regression on placement only; it reads no xG, since "
        "every in-game penalty has the same pre-shot xG.",
        _feature_table(PENALTY_FEATURES), _lr_model,
        (calib, *_limits(p, pl.concat([kicks.dev, kicks.held]), flagged=False)),
        task=PENALTY_TASK,
    ), intended_use=PENALTY_USE)  # fmt: skip
    written[pen.spec.name] = cards.render_card(
        pen, pen_text, evaluations=(pen_base, pen), benchmark=None, dev=kicks.dev,
        context=xgot_context(xg, kicks.dev, kicks.held, recon, holdout=ho), provider=p.text,
        validation=PENALTY_FOLDS.format(split=p.split), thresholds=(),
        verdict="Placement-only, reported apart from the open-play xGOT.",
        extra=[*ece_lines(pen, k_dev, k_held, split=p.split), *shoot],
    )  # fmt: skip
    out = Path(docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, card in written.items():
        (out / f"{name}.md").write_text(card, encoding="utf-8", newline="\n")
    for ev in dict.fromkeys((xg_head, xg_ev, *evs, pen)):
        print(baselines.summary_line(ev))
    for g in (gbm_gain, far_gain):
        print(f"gain | {g.label}: cv {gain_cell(g.cv)} | {p.split} {gain_cell(g.holdout)}")
    print(f"headline: {head.spec.name}")
    print(*(f"check | {c.target} | {c.measured} | {c.result}" for c in run_checks), sep="\n")
    print(f"cards: {out}")
    if pc is not None:
        print(f"cache: {pc.dir} ({len(pc.hits)} read, {len(pc.written)} written)")
    return XgotRun(
        provider, xg_ev, xg_head, evs, head, gbm_gain, far_gain, pen, shootouts, recon,
        run_checks, halves, pred, MappingProxyType(written),
    )  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; exit 0 even when a check is missed (it reports).

    Examples
    --------
    >>> main(["--provider", "statsbomb"])  # doctest: +SKIP
    0
    """
    parser = argparse.ArgumentParser(
        prog="python -m football_shot_models.xgot",
        description="Evaluate the xGOT models (MS5) and write their cards.",
    )
    parser.add_argument("--provider", choices=(*PROVIDERS, "both"), default="both")
    parser.add_argument("--docs-dir", default="docs/models", help="where the cards go")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--cache", default="data/cache", help="prediction cache root (git-ignored)")
    where.add_argument("--no-cache", action="store_true", help="fit everything, write no cache")
    args = parser.parse_args(argv)
    keys = tuple(PROVIDERS) if args.provider == "both" else (args.provider,)
    cache = None if args.no_cache else args.cache
    try:
        runs = [run(k, docs_dir=args.docs_dir, cache=cache) for k in keys]
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"fsm: error: {e}", file=sys.stderr)
        return 1
    print(summary_block(runs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
