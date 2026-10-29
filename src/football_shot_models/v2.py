"""PR1: WhoScored xG v2, v1 plus the pre-shot context of the previous play event.

The PM's learning-curve probe (2026-10-28) showed that WhoScored xG is limited by
information, not volume: more 2025-26 shots barely move v1's test log loss, while Opta's big
chance (q214, an annotator's grade of the chance) closes most of the gap. v2 adds what the
archive records about the situation *before* the shot and v1 does not read: the previous play
event (``prev_action``: pass, take-on, regain, an earlier shot, the opponent's clearance, block
or keeper action) and whether the shot is a rebound within 3 s of the team's previous shot
(``rebound``). Both come from football-core's loader (schema 1.3,
``io.whoscored.shot_context``). Big chance stays out (PM): only the labelled v1 ablation reads it.

**Same population and split as v1** (:func:`fit` calls ``v1.population`` and
``data.season_holdout`` with ``v1.TRAIN_SEASONS`` / ``v1.TEST_SEASONS``): train 2025-26, test
2026-27. The test season is not clean for v2: the design probe that chose ``prev_action`` and
``rebound`` scored candidates on it (PR1 spec §1), so the evidence it did not choose is the CV
gain over v1 on the training season. Candidates are the MS2 baselines, v1's LR steps plus one LR
on the v2 context, and a monotone GBM on the v2 inputs; ``models.select_headline`` picks the
headline.

**Keeper-derived context stays out of xGOT** (PM MS6 E5): ``prev_action`` and ``rebound`` read
the keeper's previous event, so ``features.KEEPER_DERIVED`` refuses them in the xGOT input
chain, and v1's keeper-neutral headline stays the WhoScored xGOT input (PM PR1 E4 (a)).

**Census.** :func:`census` counts every Opta qualifier on the population's shots by
:data:`QUALIFIER_CLASS` (does it describe the situation, grade the chance, or come after the
shot?) and the shots of each new context value; counts only, never a row or an id.

:func:`fit` is pure (no write, no print); ``v2_cards`` renders the card ``xg-v2``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import polars as pl
from football_core import schema
from football_core.io import whoscored

from football_shot_models import (
    ablation,
    baselines,
    cards,
    data,
    features,
    gbm,
    metrics,
    models,
    v1,
    xgot,
)
from football_shot_models.cache import PredictionCache
from football_shot_models.cards import gain_cell, side
from football_shot_models.evaluation import (
    Comparison,
    Evaluation,
    ModelSpec,
    best,
    compare,
    cv_folds,
    evaluate,
    labels,
)
from football_shot_models.metrics import Check, GroupRow

StrPath = str | os.PathLike[str]

NAME: Final = "xg-v2"
CONTEXT_V2: Final[tuple[str, ...]] = (*v1.CONTEXT, "prev_action", "rebound")
STEPS: Final[tuple[ModelSpec, ...]] = (
    *v1.STEPS,
    ModelSpec(
        "ws-lr-context-v2",
        "LR splines + context v2",
        ablation.SPLINES + CONTEXT_V2,
        ablation.make_lr,
        baselines.lr_coefficients,
    ),
)
GBM: Final = ModelSpec(
    "ws-gbm-v2",
    "Monotone GBM + Platt, context v2",
    ("distance_m", "angle_rad", "body_part", *CONTEXT_V2),
    functools.partial(gbm.MonotoneGBM, v1.GBM_SIGNS),
    gbm.gbm_settings,
)
QUALIFIER_CLASSES: Final = (
    "body part",
    "pattern",
    "location zone",
    "assist link",
    "situation",
    "grading",
    "post-shot",
    "outcome-derived",
)
REFUSED_CLASSES: Final = frozenset({"grading", "post-shot", "outcome-derived"})
# fmt: off
# Opta qualifier id -> what it says. Situation: known before the strike and not a judgement of
# the chance's quality. Grading: an annotator's grade of the chance (q214, PM: ablation only).
# Post-shot: the ball's flight or the stop. Outcome-derived: WhoScored labels set from what the
# shot led to (q11112 on the shot: the attempt created a teammate's chance, spec E1).
QUALIFIER_CLASS: Final[Mapping[int, str]] = MappingProxyType({
    **dict.fromkeys((15, 20, 21, 72), "body part"),
    **dict.fromkeys((9, 22, 23, 24, 25, 26, 160, 263), "pattern"),
    **dict.fromkeys((16, 17, 18, 19, 56, *range(60, 72)), "location zone"),
    **dict.fromkeys((2, 4, 29, 55), "assist link"),
    **dict.fromkeys((89, 94, 108, 154, 215, 328), "situation"),
    214: "grading",
    **dict.fromkeys((*range(73, 88), 100, 101, 102, 103, 146, 147, 178, 233), "post-shot"),
    **dict.fromkeys((11111, 11112, 11113), "outcome-derived"),
})
# WhoScored feature -> the Opta qualifier ids it reads (spline terms read x / y only; q233 is
# prev_action's guard, not a value)
FEATURE_QUALIFIERS: Final[Mapping[str, tuple[int, ...]]] = MappingProxyType({
    "distance_m": (), "angle_rad": (), "body_part": tuple(whoscored.BODY_PARTS),
    **{f: () for f in ablation.SPLINES if f.startswith(("distance_m_", "angle_rad_"))},
    "phase": (*whoscored.PATTERNS, whoscored.DIRECT_CORNER),
    "assist": (whoscored.RELATED, whoscored.CROSS, whoscored.THROUGH_BALL),
    "one_on_one": (whoscored.ONE_ON_ONE,), "first_time": (whoscored.FIRST_TOUCH,),
    "big_chance": (whoscored.BIG_CHANCE,), "prev_action": (whoscored.OUTFIELD_BLOCK,),
    "rebound": (),
})
# fmt: on
CONTEXT_COLUMNS: Final = ("prev_action", "rebound")
CARRY_LINE: Final = "carry: not recorded by Opta; dribble = TakeOn"


@dataclass(frozen=True)
class Census:
    """Counts of the pre-shot information on the population's shots (no row, no id).

    Attributes
    ----------
    qualifiers : polars.DataFrame
        One row per Opta qualifier id on a shot: ``id``, ``name`` (Opta's display name),
        ``class`` (null when unclassified), ``train_shots``, ``train_goals``, ``test_shots``.
    context : polars.DataFrame
        One row per value of ``prev_action`` and ``rebound``: ``column``, ``value``, ``class``,
        ``keeper_derived``, ``train_shots``, ``train_goals``, ``test_shots``.
    """

    qualifiers: pl.DataFrame
    context: pl.DataFrame


@dataclass(frozen=True, eq=False)
class V2Fit:
    """Everything :func:`fit` evaluated, chart-ready: frames and evaluations, no refit needed.

    ``dev`` / ``held`` are the train / test shots every evaluation was scored on (``oof`` and
    ``holdout_p`` follow their ``shot_id`` order). ``groups`` is the headline's test Σ xG error
    per competition; ``checks`` are :func:`v2_checks`.
    """

    context: cards.CardContext
    dev: pl.DataFrame
    held: pl.DataFrame
    candidates: models.Candidates
    head: Evaluation
    gbm_gain: Comparison
    v1_head: Evaluation
    v1_ablation: Evaluation
    gain_over_v1: Comparison
    groups: tuple[GroupRow, ...]
    census: Census
    checks: tuple[Check, ...]


def _counts(ids: list[set[int]], goal: list[bool], train: list[bool]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for found, g, t in zip(ids, goal, train, strict=True):
        for q in found:
            row = out.setdefault(q, [0, 0, 0])
            row[0 if t else 2] += 1
            row[1] += int(g and t)
    return out


def census(shots: pl.DataFrame, train: pl.Series) -> Census:
    """The qualifier and context census of ``shots``, split by the Boolean mask ``train``.

    Qualifiers are read from ``raw_qualifiers`` (lineage, read for the census only), each id
    once per shot; the name is Opta's ``displayName`` (the most common, ties alphabetical).

    Raises
    ------
    ValueError
        ``train`` not a null-free Boolean Series of ``shots.height``; a ``raw_qualifiers``
        value that is not a JSON list of qualifier objects (with the count).

    Examples
    --------
    >>> shots = pl.DataFrame({
    ...     "raw_qualifiers": ['[{"type":{"value":214,"displayName":"BigChance"}}]', "[]"],
    ...     "is_goal": [True, False], "prev_action": ["pass", "opp_block"],
    ...     "rebound": [False, True]}, schema_overrides={"prev_action": schema.PREV_ACTION})
    >>> c = census(shots, pl.Series([True, False]))
    >>> c.qualifiers.row(0)
    (214, 'BigChance', 'grading', 1, 1, 0)
    >>> c.context.filter(pl.col("value") == "opp_block").row(0)
    ('prev_action', 'opp_block', 'situation', 'yes', 0, 0, 1)
    """
    if train.dtype != pl.Boolean or train.len() != shots.height or train.null_count():
        raise ValueError(
            f"census: train must be a null-free Boolean mask of the {shots.height} shot(s), got "
            f"{train.dtype} of length {train.len()} with {train.null_count()} null(s)"
        )
    ids: list[set[int]] = []
    names: dict[int, Counter[str]] = {}
    bad = 0
    for text in shots["raw_qualifiers"].to_list():
        try:
            items = json.loads(text)
            found = {(q["type"]["value"], q["type"]["displayName"]) for q in items}
        except (TypeError, KeyError, ValueError):
            bad += 1
            continue
        ids.append({int(i) for i, _ in found})
        for i, name in found:
            names.setdefault(int(i), Counter())[str(name)] += 1
    if bad:
        raise ValueError(f"census: {bad} raw_qualifiers value(s) are not a JSON qualifier list")
    goal, mask = shots["is_goal"].to_list(), train.to_list()
    rows = [
        (q, min(n.items(), key=lambda kv: (-kv[1], kv[0]))[0], QUALIFIER_CLASS.get(q), *c)
        for q, c in sorted(_counts(ids, goal, mask).items())
        for n in (names[q],)
    ]
    qualifiers = pl.DataFrame(
        rows,
        schema={"id": pl.Int64, "name": pl.String, "class": pl.String, "train_shots": pl.Int64,
                "train_goals": pl.Int64, "test_shots": pl.Int64},
        orient="row",
    )  # fmt: skip
    context = []
    for column in CONTEXT_COLUMNS:
        values = shots[column].cast(pl.String).fill_null("null").to_list()
        levels = schema.PREV_ACTIONS if column == "prev_action" else ("true", "false")
        found_ = _counts([{levels.index(v)} if v in levels else set() for v in values], goal, mask)
        context += [
            (column, level, "situation", "yes", *found_.get(i, [0, 0, 0]))
            for i, level in enumerate(levels)
        ]
    ctx = pl.DataFrame(
        context,
        schema={"column": pl.String, "value": pl.String, "class": pl.String,
                "keeper_derived": pl.String, "train_shots": pl.Int64, "train_goals": pl.Int64,
                "test_shots": pl.Int64},
        orient="row",
    )  # fmt: skip
    return Census(qualifiers, ctx)


def gap_share(
    v1_head: Evaluation, v2_head: Evaluation, ablation_ev: Evaluation, split: str
) -> float:
    """``(L_v1 − L_v2) / (L_v1 − L_ablation)``: the share of the q214 ablation gap v2 closes.

    ``split`` is ``"cv"`` (grouped-CV log loss) or ``"test"`` (the test season).

    Raises
    ------
    ValueError
        Another ``split``; a denominator <= 0 (the ablation is not better than v1).

    Examples
    --------
    >>> from types import SimpleNamespace as N
    >>> ev = lambda x: N(cv=N(log_loss=x), holdout=N(log_loss=x))
    >>> round(gap_share(ev(0.2771), ev(0.2751), ev(0.2611), "test"), 3)
    0.125
    """
    if split not in ("cv", "test"):
        raise ValueError(f"gap_share: split {split!r}; allowed: 'cv', 'test'")
    a, b, c = (
        (e.cv if split == "cv" else e.holdout).log_loss for e in (v1_head, v2_head, ablation_ev)
    )
    if a - c <= 0:
        raise ValueError(
            f"gap_share: v1's {split} log loss {a:.4f} is not above the ablation's {c:.4f}"
        )
    return (a - b) / (a - c)


def _feature_classes(feats: tuple[str, ...]) -> list[str]:
    """The headline features that break check row 4, with the reason."""
    bad = [f"{f} (ablation only)" for f in feats if f in features.ABLATION_ONLY]
    for f in feats:
        if f not in FEATURE_QUALIFIERS:
            bad.append(f"{f} (no qualifier map)")
            continue
        for q in FEATURE_QUALIFIERS[f]:
            if (cls := QUALIFIER_CLASS.get(q)) is None or cls in REFUSED_CLASSES:
                bad.append(f"{f} (q{q} {cls or 'unclassified'})")
    return bad


def xgot_input_features() -> tuple[str, ...]:
    """The features of the WhoScored xGOT input chain's xG candidates, read when called.

    Examples
    --------
    >>> "rebound" in xgot_input_features()
    False
    """
    steps, ws_gbm = xgot.PROVIDERS["whoscored"].candidate_specs()
    return tuple(dict.fromkeys(f for s in (*steps, ws_gbm) for f in s.features))


def gap_text(v1_head: Evaluation, head: Evaluation, ablation_ev: Evaluation, split: str) -> str:
    """:func:`gap_share` as a percentage, or ``n/a`` with the reason when it is undefined.

    Examples
    --------
    >>> from types import SimpleNamespace as N
    >>> ev = lambda x: N(cv=N(log_loss=x), holdout=N(log_loss=x))
    >>> gap_text(ev(0.27), ev(0.27), ev(0.28), "cv")
    'n/a (the ablation is not better than v1)'
    """
    try:
        return f"{gap_share(v1_head, head, ablation_ev, split):.1%}"
    except ValueError:
        return "n/a (the ablation is not better than v1)"


def v2_checks(
    *,
    head: Evaluation,
    candidates: models.Candidates,
    gbm_gain: Comparison,
    gain_over_v1: Comparison,
    census: Census,
    v1_head: Evaluation,
    v1_ablation: Evaluation,
    xgot_features: tuple[str, ...] | None = None,
) -> tuple[Check, ...]:
    """The six PR1 checks of the headline, then the labelled reporting rows (result ``n/a``).

    Parameters are named after the :class:`V2Fit` fields they come from. Rows: (1) test gain
    over v1's headline with a 95 % match-bootstrap interval above 0; (2) test calibration slope
    in ``models.SLOPE_RANGE``; (3) the GBM is headline iff its CV gain over the best LR lies
    above 0; (4) no headline feature is ablation-only, unmapped in :data:`FEATURE_QUALIFIERS`
    or reads a qualifier of a refused or no class; (5) no keeper-derived feature in the
    WhoScored xGOT input chain (``xgot_features``, default :func:`xgot_input_features`; PM MS6
    E5); (6) every qualifier in the census has a class. Reporting: the gap share (test, CV)
    and the test Σ xG error.

    Examples
    --------
    >>> [c.result for c in v2_checks(head=h, candidates=c, gbm_gain=gg, gain_over_v1=g1,
    ...  census=cen, v1_head=v, v1_ablation=a)][:6]  # doctest: +SKIP
    ['met', 'met', 'met', 'met', 'met', 'met']
    """
    lo, hi = models.SLOPE_RANGE
    feats = head.spec.features if isinstance(head.spec, ModelSpec) else ()
    bad = _feature_classes(feats)
    xf = xgot_input_features() if xgot_features is None else xgot_features
    leaked = sorted(set(xf) & features.KEEPER_DERIVED)
    unclassified = census.qualifiers.filter(pl.col("class").is_null())["id"].to_list()
    shares = [gap_text(v1_head, head, v1_ablation, split) for split in ("test", "cv")]
    g = gain_over_v1.holdout
    return (
        Check(
            f"Test log-loss gain {gain_over_v1.label}, 95% match-bootstrap interval above 0",
            f"{head.spec.title}: {gain_cell(g)}, an interval that {side(g)}",
            "met" if g.low > 0 else "missed",
        ),
        models.slope_check(
            f"Calibration slope in [{lo:g}, {hi:g}], test season", head.holdout.cal_slope
        ),  # fmt: skip
        Check(
            f"GBM headline iff its CV gain {gbm_gain.label} has a 95% interval above 0",
            f"CV {gain_cell(gbm_gain.cv)}; headline {head.spec.title}",
            "met" if (head is candidates.gbm) == (gbm_gain.cv.low > 0) else "missed",
        ),
        Check(
            "No headline feature grades the chance or comes after the shot (big chance only in "
            "the v1 ablation; every feature's Opta qualifiers classed as allowed)",
            f"headline features: {', '.join(feats) or 'none'}"
            + (f"; refused: {', '.join(bad)}" if bad else ""),
            "missed" if bad else "met",
        ),
        Check(
            "No keeper-derived context in the WhoScored xGOT input chain (PM MS6 E5)",
            f"keeper-derived features there: {', '.join(leaked) or 'none'}",
            "missed" if leaked else "met",
        ),
        Check(
            "Every Opta qualifier on a population shot has a class",
            f"{census.qualifiers.height} qualifier(s), unclassified: "
            + (", ".join(f"q{q}" for q in unclassified) or "none"),
            "missed" if unclassified else "met",
        ),
        Check(
            "Reporting row (not a check): share of the q214 ablation gap closed, test / CV",
            f"{shares[0]} / {shares[1]}",
            "n/a",
        ),
        Check(
            "Reporting row (not a check): test Σ xG error, signed",
            f"{head.holdout.xg_signed_error:+.2%}",
            "n/a",
        ),
    )


def fit(manifest: StrPath = v1.DEFAULT_MANIFEST, *, cache: StrPath | None) -> V2Fit:
    """Evaluate every v2 candidate on v1's population and split; no write, no print.

    ``v1_head`` is ``v1.GBM`` (xG v1's headline) and ``v1_ablation`` the big-chance ablation of
    v1's best LR, both cache hits after ``make xg-v1``.

    Raises
    ------
    ValueError
        From the loader, the split, the fits or :func:`census`.

    Examples
    --------
    >>> fit(cache="data/cache").head.spec.name  # doctest: +SKIP
    'ws-gbm-v2'
    """
    xg = data.load_xg_data(manifest, builder=v1.BUILDER)
    shots, _ = v1.population(xg)
    ho = data.season_holdout(xg.matches, v1.TRAIN_SEASONS, v1.TEST_SEASONS)
    dev, held = data.split_holdout(shots, ho)
    folds = cv_folds(dev)
    pc = None if cache is None else PredictionCache(cache, xg.manifest.data_digest)
    c = models.evaluate_candidates(
        STEPS, GBM, dev, held, folds, freeze_frames=xg.freeze_frames, cache=pc
    )
    head, gbm_gain = models.select_headline(c.lrs, c.gbm, dev, held)
    v1_head = evaluate(v1.GBM, dev, held, folds, cache=pc)
    # v1's candidates by name, not by position in STEPS: the ablation is v1's labelled one
    v1_names = {s.name for s in (*baselines.BASELINES, *v1.STEPS)}
    v1_best = best(tuple(e for e in c.lrs if e.spec.name in v1_names)).spec
    if not isinstance(v1_best, ModelSpec):
        raise TypeError(f"best returned a benchmark: {v1_best!r}")
    v1_abl = evaluate(v1.ablation_spec(v1_best), dev, held, folds, cache=pc)
    gain_over_v1 = compare("over xG v1", v1_head, head, dev, held)
    comp = held.join(
        xg.matches.select("match_id", "competition_name"),
        on="match_id", how="left", validate="m:1", maintain_order="left",
    )["competition_name"].cast(pl.String)  # fmt: skip
    groups = metrics.by_group(
        labels(held), head.holdout_p, comp, held["match_id"].to_numpy(), models.MIN_SEGMENT_SHOTS
    )
    both = pl.concat([dev, held], how="vertical")
    cen = census(both, pl.Series([True] * dev.height + [False] * held.height))
    checks = v2_checks(
        head=head, candidates=c, gbm_gain=gbm_gain, gain_over_v1=gain_over_v1, census=cen,
        v1_head=v1_head, v1_ablation=v1_abl,
    )  # fmt: skip
    context = cards.card_context(dataclasses.replace(xg, shots=shots), dev, held, ho)
    return V2Fit(
        context=context, dev=dev, held=held, candidates=c, head=head, gbm_gain=gbm_gain,
        v1_head=v1_head, v1_ablation=v1_abl, gain_over_v1=gain_over_v1, groups=groups,
        census=cen, checks=checks,
    )  # fmt: skip
