"""M1.4 and M1.5 run: the GBM spec, the headline rule, the MS3 checks and the cards.

**The GBM** (``xg-gbm``) is :class:`~football_shot_models.gbm.MonotoneGBM` on S3's inputs
without the spline terms (trees bend by themselves). Four columns carry a monotone constraint
for football reasons only (:data:`MONOTONE_REASONS`): further from goal at the same opening
angle never helps, a wider opening never hurts, another outfield body in the cone never helps,
and a closer marker never helps. A constraint holds along one column with every other column
held fixed. The estimator, its tuning and its calibration are in :mod:`football_shot_models.gbm`.

**Headline rule.** The GBM is the headline model only if its grouped-CV log-loss gain over the
best logistic regression has a 95 % match-bootstrap interval that lies above 0
(:func:`headline`); otherwise the LR stays headline and every card says so (:func:`verdict`).
The hold-out gain is reported next to it, and so is the comparison with the benchmark
(StatsBomb's own xG here; :func:`benchmark_sentence`), without the words "beats" or "matches".

**Penalties** (``xg-penalty``) are modelled in :mod:`football_shot_models.penalties`; this run
calls it and writes its card.

Run ``python -m football_shot_models.models`` (``make xg-models``) to evaluate every model
once and write the step cards, the GBM card and the penalty card into ``docs/models/``. The
fits' predictions are cached under ``data/cache`` (:mod:`football_shot_models.cache`); a rerun
on the same data and code reads them instead of fitting.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import polars as pl
from football_core.io import snapshot

from football_shot_models import ablation, baselines, cards, data, features, gbm, penalties
from football_shot_models.cache import PredictionCache
from football_shot_models.cards import (
    STATSBOMB_TEXT,
    CardContext,
    CardText,
    card_context,
    fmt,
    gain_cell,
    pct,
    render_card,
    side,
)
from football_shot_models.evaluation import (
    Comparison,
    Evaluation,
    Folds,
    ModelSpec,
    best,
    compare,
    cv_folds,
    evaluate,
    evaluate_benchmark,
    labels,
)
from football_shot_models.metrics import Check, calibration, fold_xg_error

if TYPE_CHECKING:  # the field named `penalties` hides the module inside ModelsRun
    from football_shot_models.penalties import PenaltyRow, Spread

StrPath = str | os.PathLike[str]

GBM_FEATURES: Final[tuple[str, ...]] = (
    "distance_m",
    "angle_rad",
    "body_part",
    *ablation.CONTEXT,
    *ablation.FRAMES,
)
GBM: Final = ModelSpec(
    "xg-gbm", "Monotone GBM + Platt", GBM_FEATURES, gbm.MonotoneGBM, gbm.gbm_settings
)


def is_gbm(ev: Evaluation) -> bool:
    """Whether ``ev`` is a GBM: its spec's card rows come from ``gbm.gbm_settings``.

    Any sign map and any grid count; the one rule behind the GBM card branches (v1, xGOT) and
    the GBM headline check rows.

    Examples
    --------
    >>> is_gbm(head)  # doctest: +SKIP
    True
    """
    return isinstance(ev.spec, ModelSpec) and ev.spec.coefficients is gbm.gbm_settings


# ---- headline rule and sentences ---------------------------------------------------------


def headline(
    best_lr: Evaluation, gbm: Evaluation, dev: pl.DataFrame, held: pl.DataFrame
) -> tuple[Evaluation, Comparison]:
    """The headline model and the GBM's gain over the best LR.

    The GBM is headline iff its CV gain has ``low > 0``; ``low == 0`` keeps the LR. The
    hold-out gain is reported, never used to decide.

    Examples
    --------
    >>> head, gain = headline(s3, gbm, dev, held)  # doctest: +SKIP
    >>> head.spec.name, gain.label  # doctest: +SKIP
    ('xg-gbm', 'over LR splines + context + freeze frames')
    """
    gain = compare(f"over {best_lr.spec.title}", best_lr, gbm, dev, held)
    return (gbm if gain.cv.low > 0 else best_lr), gain


def select_headline(
    candidates: Sequence[Evaluation], gbm: Evaluation, dev: pl.DataFrame, held: pl.DataFrame
) -> tuple[Evaluation, Comparison]:
    """:func:`headline` of the best of ``candidates`` (``evaluation.best``) and the GBM.

    The one headline rule of every run (MS3, MS4 v1, the MS5 xGOT input).

    Raises
    ------
    TypeError
        ``best`` returns a benchmark evaluation.
    ValueError
        From ``best`` (no candidate) or :func:`headline`.

    Examples
    --------
    >>> head, gain = select_headline((*base, *steps), gbm, dev, held)  # doctest: +SKIP
    >>> head.spec.name  # doctest: +SKIP
    'xg-gbm'
    """
    best_lr = best(candidates)
    if not isinstance(best_lr.spec, ModelSpec):
        raise TypeError(f"best returned a benchmark: {best_lr.spec!r}")
    return headline(best_lr, gbm, dev, held)


@dataclass(frozen=True, eq=False)
class Candidates:
    """The xG candidates of one run, evaluated on one set of folds.

    Attributes
    ----------
    baselines : tuple of Evaluation
        ``baselines.BASELINES``, in that order.
    steps : tuple of Evaluation
        The provider's LR steps, in the order given.
    gbm : Evaluation
        The provider's GBM.
    """

    baselines: tuple[Evaluation, ...]
    steps: tuple[Evaluation, ...]
    gbm: Evaluation

    @property
    def lrs(self) -> tuple[Evaluation, ...]:
        """The LR candidates of the headline rule: the baselines, then the steps."""
        return (*self.baselines, *self.steps)


def freeze_frames_or_none(freeze_frames: pl.DataFrame | None) -> pl.DataFrame | None:
    """``None`` for no freeze frames or an empty table: a frame feature then raises instead of
    being derived from an empty table (WhoScored has no frames).

    Examples
    --------
    >>> freeze_frames_or_none(pl.DataFrame({"shot_id": []})) is None
    True
    """
    return None if freeze_frames is None or freeze_frames.height == 0 else freeze_frames


def evaluate_candidates(
    steps: Sequence[ModelSpec],
    gbm_spec: ModelSpec,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    folds: Folds,
    *,
    freeze_frames: pl.DataFrame | None,
    cache: PredictionCache | None,
) -> Candidates:
    """Evaluate the baselines, the LR steps and the GBM of a run: the one candidate set.

    ``models.run`` (StatsBomb xG), ``v1.run`` (WhoScored xG) and the xGOT input
    (``xgot.xg_input_eval``) all call this, so the xGOT input is chosen from the very
    evaluations the xG card prints. ``baselines.BASELINES`` is read when called. Every spec is
    evaluated with the same ``freeze_frames`` and ``cache``; an empty ``freeze_frames`` is the
    same as ``None``, so a frame feature on a provider without frames raises instead of being
    derived from an empty table.

    Raises
    ------
    ValueError
        Two specs share a name (a duplicate would overwrite a cache entry); from ``evaluate``.

    Examples
    --------
    >>> c = evaluate_candidates(ablation.STEPS, GBM, dev, held, folds,
    ...                         freeze_frames=xg.freeze_frames, cache=pc)  # doctest: +SKIP
    >>> [ev.spec.name for ev in c.steps], c.gbm.spec.name  # doctest: +SKIP
    (['xg-lr-splines', 'xg-lr-context', 'xg-lr-frames'], 'xg-gbm')
    """
    base_specs = tuple(baselines.BASELINES)
    names = [s.name for s in (*base_specs, *steps, gbm_spec)]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"evaluate_candidates: spec name(s) {dupes} used twice")
    ff = freeze_frames_or_none(freeze_frames)

    def run(spec: ModelSpec) -> Evaluation:
        return evaluate(spec, dev, held, folds, freeze_frames=ff, cache=cache)

    return Candidates(
        tuple(run(b) for b in base_specs), tuple(run(s) for s in steps), run(gbm_spec)
    )


def verdict(ev: Evaluation, head: Evaluation, gain: Comparison, *, split: str = "hold-out") -> str:
    """The card overview sentence of ``ev``: is it the headline model, and why.

    ``gain`` is :func:`headline`'s comparison (the GBM over the best LR); ``split`` names the
    scored-once split as the card calls it (the MS3 cards "hold-out", the v1 card "test").

    Examples
    --------
    >>> verdict(gbm, gbm, gain)  # doctest: +SKIP
    'This is the headline model: its CV log-loss gain over LR splines + ... lies above 0; ...'
    """
    lr = gain.label.removeprefix("over ")
    cv = gain_cell(gain.cv)
    if ev is head and is_gbm(head):
        return (
            f"This is the headline model: its CV log-loss gain over {lr} is {cv}, a 95% "
            f"interval that lies above 0; on the {split} it is {gain_cell(gain.holdout)}, an "
            f"interval that {side(gain.holdout)}."
        )
    if ev is head:
        return (
            f"This is the headline model: the GBM's CV gain over it is {cv}, an interval that "
            "does not lie above 0, so the GBM does not ship as headline."
        )
    if is_gbm(ev) and not is_gbm(head):
        return (
            f"Not the headline model: its CV gain over {lr} is {cv}, an interval that does not "
            f"lie above 0, so {lr} stays headline."
        )
    return f"Not the headline model; that is {head.spec.title}."


def benchmark_sentence(ev: Evaluation, sb_gain: Comparison, benchmark: Evaluation) -> str:
    """The comparison with a benchmark, named by its title, on CV and hold-out, with Σ xG.

    ``sb_gain`` is ``compare(..., benchmark, ev, ...)``; any benchmark works (StatsBomb xG
    here, a legacy model in a later milestone).

    Examples
    --------
    >>> benchmark_sentence(gbm, sb_gain, sb)  # doctest: +SKIP
    'Against StatsBomb xG: CV gain +0.0021 [+0.0012, +0.0031], an interval that ...'
    """
    h, b, name = ev.holdout, benchmark.holdout, benchmark.spec.title
    return (
        f"Against {name}: CV gain {gain_cell(sb_gain.cv)}, an interval that "
        f"{side(sb_gain.cv)}; hold-out gain {gain_cell(sb_gain.holdout)}, an interval that "
        f"{side(sb_gain.holdout)}. On the hold-out it predicts {h.xg:.0f} goals for "
        f"{h.goals:,} scored ({h.xg_signed_error:+.2%}); {name} predicts {b.xg:.0f} "
        f"({b.xg_signed_error:+.2%})."
    )


# ---- MS3 milestone checks ----------------------------------------------------------------

SLOPE_RANGE: Final[tuple[float, float]] = (0.9, 1.1)
SB_GAP: Final[float] = 0.006
SB_MS2_CV: Final[float] = 0.2674  # MS2 population (direct corners in), quoted only
FOLD_XG_MEAN: Final[float] = 0.03
FOLD_XG_MAX: Final[float] = 0.05
MS2_BEST_CV: Final[float] = 0.2880  # MS2 population, quoted only
MIN_SEGMENT_SHOTS: Final[int] = 1000
_GENDERS: Final = (("male", "men"), ("female", "women"))


def _check(target: str, measured: str, ok: bool) -> Check:
    return Check(target, measured, "met" if ok else "missed")


def slope_check(target: str, slope: float | None) -> Check:
    """The check row "calibration slope in ``SLOPE_RANGE``" (both ends in); ``n/a`` without one.

    Examples
    --------
    >>> slope_check("slope", 0.9).result, slope_check("slope", None).result
    ('met', 'n/a')
    """
    if slope is None:
        return Check(target, "not estimable", "n/a")
    return _check(target, fmt(slope), SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1])


def milestone_checks(
    head: Evaluation,
    benchmark: Evaluation,
    *,
    s0_gain: Comparison,
    sb_gain: Comparison,
    gbm_gain: Comparison,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    s0: Evaluation | None = None,
) -> tuple[Check, ...]:
    """The MS3 acceptance checks of the headline model, in the card's order.

    Every bound is formatted from its constant.

    Parameters
    ----------
    head, benchmark : Evaluation
        The headline model and StatsBomb xG.
    s0 : Evaluation, optional
        The MS2 best baseline (S0); when given, the first target quotes its CV log loss on
        these folds.
    s0_gain, sb_gain : Comparison
        The headline model over S0 and over StatsBomb (``sb_gain`` is printed by
        :func:`benchmark_sentence`; the checks use the log-loss gap).
    gbm_gain : Comparison
        The GBM over the best LR (:func:`headline`).
    dev, held : polars.DataFrame
        The frames everything was evaluated on (``gender`` for the per-gender slopes).

    Returns
    -------
    tuple of Check
        Gain over S0; gap to StatsBomb; calibration slopes (CV fold mean, hold-out, then men
        and women pooled out of fold, then men and women on the hold-out; a gender group under
        ``MIN_SEGMENT_SHOTS`` shots is ``n/a``); fold Σ xG error mean and max; the GBM over the
        best LR.

    Examples
    --------
    >>> checks = milestone_checks(gbm, sb, s0_gain=g0, sb_gain=gs, gbm_gain=gg, dev=dev,
    ...                           held=held, s0=s0)  # doctest: +SKIP
    >>> [c.result for c in checks]  # doctest: +SKIP
    ['met', 'met', 'met', 'met', 'met', 'met', 'met', 'n/a', 'met', 'met', 'met']
    """
    title, gap = head.spec.title, head.cv.log_loss - benchmark.cv.log_loss
    band = f"calibration slope in [{SLOPE_RANGE[0]:g}, {SLOPE_RANGE[1]:g}]"
    here = "" if s0 is None else f"{s0.spec.title} {s0.cv.log_loss:.4f} on these folds; "
    checks = [
        _check(
            f"CV log-loss gain over S0 with a 95% interval above 0 ({here}the MS2 best baseline "
            f"{MS2_BEST_CV:.4f} is the MS2 population)",
            f"{title}: {gain_cell(s0_gain.cv)}",
            s0_gain.cv.low > 0,
        ),
        _check(
            f"CV log loss not more than {SB_GAP:g} worse than StatsBomb xG (StatsBomb CV "
            f"{benchmark.cv.log_loss:.4f} on these shots; the PM brief's {SB_MS2_CV:.4f} is the "
            "MS2 population, with direct corners)",
            f"{title} {head.cv.log_loss:.4f}, StatsBomb {benchmark.cv.log_loss:.4f}: gap "
            f"{gap:+.4f}",
            gap <= SB_GAP,
        ),
        slope_check(f"{band}, CV fold mean", head.cv.cal_slope),
        slope_check(f"{band}, hold-out", head.holdout.cal_slope),
    ]
    for frame, p, where, name in (
        (dev, head.oof, "pooled out of fold", "dev"),
        (held, head.holdout_p, "hold-out", "the hold-out"),
    ):
        for gender, people in _GENDERS:
            mask, target = (frame["gender"] == gender).to_numpy(), f"{band}, {people}, {where}"
            if (n := int(mask.sum())) < MIN_SEGMENT_SHOTS:
                checks.append(Check(target, f"n/a ({n:,} {people}'s shots in {name})", "n/a"))
            else:
                y = labels(frame)[mask].astype(np.float64)
                checks.append(slope_check(target, calibration(y, p[mask])[1]))
    pooled = f"(pooled {pct(head.cv.xg_error)})"
    mean, worst = fold_xg_error(head.folds)
    for stat, value, bound in (("mean", mean, FOLD_XG_MEAN), ("max", worst, FOLD_XG_MAX)):
        target = f"Σ xG error, fold {stat} <= {bound:.0%}"
        checks.append(_check(target, f"{pct(value)} {pooled}", value <= bound))
    checks.append(
        _check(
            f"GBM CV gain {gbm_gain.label}, 95% interval above 0 (the GBM is headline iff met)",
            f"CV {gain_cell(gbm_gain.cv)}; hold-out {gain_cell(gbm_gain.holdout)}, an interval "
            f"that {side(gbm_gain.holdout)}",
            gbm_gain.cv.low > 0,
        )
    )
    return tuple(checks)


# ---- GBM card text -----------------------------------------------------------------------

NEAR_IDENTITY: Final[float] = 0.05
MONOTONE_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "distance_m": "further from goal at the same opening angle never helps",
        "angle_rad": "a wider opening between the posts never hurts",
        "opponents_in_cone": "another outfield body between ball and goal never helps",
        "nearest_opponent_m": "a closer outfield marker never helps (outfield opponents only, "
        f"capped at {features.NEAREST_CAP_M:g} m, also the value with no marker in the frame)",
    }
)
MONOTONE_NOTE: Final[str] = (
    "Each constraint holds along one column with every other column held fixed: at a fixed "
    "opening angle, more distance is a move along an arc through both posts."
)


def _settings(ev: Evaluation) -> dict[str, float]:
    return {term: value for term, value, _ in ev.coefficients}


def gbm_grid(ev: Evaluation) -> Mapping[str, Any]:
    """The GBM fit's own settings (``get_params``: ``leaf_grid``, ``max_iter``, ...).

    Read from the spec's ``make`` (bound when MonotoneGBM was made), never from gbm's module
    constants, so a patched grid prints as fitted.

    Raises
    ------
    ValueError
        A benchmark evaluation.

    Examples
    --------
    >>> gbm_grid(gbm)["leaf_grid"]  # doctest: +SKIP
    (8, 16, 32)
    """
    if not isinstance(ev.spec, ModelSpec):
        raise ValueError(f"{ev.spec.name}: a GBM card needs a ModelSpec, not a benchmark")
    return ev.spec.make().get_params()


def gbm_features(
    ev: Evaluation,
    rows: Mapping[str, str] = ablation.FEATURE_ROWS,
    reasons: Mapping[str, str] = MONOTONE_REASONS,
    note: str = MONOTONE_NOTE,
) -> tuple[str, ...]:
    """The GBM card's Features section: the feature table, then the fitted sign map.

    The sign lines come from the ``monotone:`` rows of ``ev.coefficients`` (the dev fit), so
    a non-default map prints as fitted. ``rows`` (feature -> table row), ``reasons`` (feature
    -> why its sign) and ``note`` (the sentence after the signs) default to the xG GBM's;
    another GBM passes its own.

    Examples
    --------
    >>> gbm_features(gbm)[-3]  # doctest: +SKIP
    '- `nearest_opponent_m`: non-decreasing — a closer outfield marker never helps (...)'
    """
    signs = [
        (t.removeprefix("monotone:"), v) for t, v, _ in ev.coefficients if t[:9] == "monotone:"
    ]
    feats = ev.spec.features if isinstance(ev.spec, ModelSpec) else ()
    return (
        "| Feature | Definition | Source field(s) |",
        "|---|---|---|",
        *(rows[f] for f in feats),
        "",
        "Monotone constraints of the fitted model (every other column is free):",
        "",
        *(
            f"- {col}: {'non-increasing' if s < 0 else 'non-decreasing'}"
            + (f" — {reasons[col]}" if col in reasons else "")
            for col, s in signs
        ),
        "",
        note,
    )


def gbm_model(ev: Evaluation, c: CardContext) -> list[str]:
    """The Model section of a GBM card: settings, tuned grid values and the Platt step.

    Examples
    --------
    >>> gbm_model(gbm_ev, ctx)[0].startswith("`HistGradientBoostingClassifier(")
    ... # doctest: +SKIP
    True
    """
    g, s = gbm_grid(ev), _settings(ev)
    h = gbm.make_hgb(g["leaf_grid"][0], g["step"], [], g["min_samples_leaf"])  # fixed settings
    a, b = s["platt_intercept"], s["platt_slope"]
    near = abs(b - 1.0) <= NEAR_IDENTITY and abs(a) <= NEAR_IDENTITY
    return [
        f"`HistGradientBoostingClassifier(learning_rate={h.learning_rate:g}, l2_regularization="
        f"{h.l2_regularization:g}, min_samples_leaf={h.min_samples_leaf}, early_stopping="
        f"{h.early_stopping}, random_state={h.random_state})`, `monotonic_cst` placed by column "
        f"name, no class weights. scikit-learn {c.sklearn_version}.",
        "",
        f"Tuning inside each fit: {g['inner_splits']}-fold `GroupKFold` by match on its training "
        f"rows, `max_leaf_nodes` in {g['leaf_grid']} and {g['step']}, {2 * g['step']}, ..., "
        f"{g['max_iter']} iterations "
        "(staged scores of one fit per leaf count); the lowest inner out-of-fold log loss wins, "
        "ties to fewer leaves, then fewer iterations. Platt scaling: an unpenalised logistic "
        "regression of the goal on the chosen configuration's inner out-of-fold log-odds, "
        "applied to the refit on all training rows; a slope <= 0 is refused.",
        "",
        "Tuning and calibration see only the rows of the fit they belong to; each outer test "
        "fold is scored by a model that never saw it.",
        "",
        "Settings of the fit on all of dev:",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| max_leaf_nodes | {s['max_leaf_nodes']:.0f} |",
        f"| iterations | {s['iterations']:.0f} |",
        f"| Platt intercept | {a:+.4f} |",
        f"| Platt slope | {b:.4f} |",
        "",
        f"Calibration is close to identity (Platt slope {b:.3f}, intercept {a:+.3f}): the boosted"
        " model is already calibrated and Platt is a safeguard."
        if near
        else f"Platt moves the scores: slope {b:.3f}, intercept {a:+.3f}.",
    ]


CARD: Final[CardText] = CardText(
    role="M1.4: monotone gradient boosting on S3's inputs without spline terms, tuned and "
    "Platt-calibrated inside each fit by grouped CV.",
    intended_use=ablation.INTENDED_USE,
    features=(),  # from the fitted sign map: gbm_features
    model=gbm_model,
    limitations=(*ablation.FRAME_LIMITS, *ablation.CONTEXT_LIMITS, *ablation.POOLED_LIMITS),
    # gbm_text appends ablation.holdout_women (computed); the GBM has no drop-one rows
    changelog=(
        "v1 (MS3, M1.4): first version.",
        "v2 (MS3 assessment): the benchmark sentence names the benchmark; the training-mix line "
        "makes no causal claim; the hold-out women's line is computed.",
    ),
    task=cards.TASK,
    target=cards.TARGET,
    best_label="model",
)


def _goal_rate(frame: pl.DataFrame, family: str | None = None) -> float | None:
    y = (frame if family is None else frame.filter(pl.col("family") == family))["is_goal"]
    return float(y.sum()) / y.len() if y.len() else None


def gbm_text(
    ev: Evaluation,
    benchmark: Evaluation,
    *,
    gain: Comparison,
    sb_gain: Comparison,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    context: CardContext,
) -> CardText:
    """``CARD`` with the fitted features and the computed limitation lines.

    ``gain`` is :func:`headline`'s comparison and ``sb_gain`` the GBM over StatsBomb.

    Examples
    --------
    >>> gbm_text(gbm, sb, gain=gg, sb_gain=gs, dev=dev, held=held,
    ...          context=c).limitations[0]  # doctest: +SKIP
    'The hold-out gain over LR splines + context + freeze frames is +0.0014 [-0.0005, ...'
    """
    lines = [
        f"The hold-out gain {gain.label} is {gain_cell(gain.holdout)}, an interval that "
        f"{side(gain.holdout)}; the headline rule is decided on CV, and the hold-out "
        f"({context.holdout_label}, {context.holdout_matches:,} matches) is a smaller test."
    ]
    league, team, women = (_goal_rate(dev, f) for f in data.FAMILIES)
    more = league is not None and min(team or 0.0, women or 0.0) > league
    carry = side(sb_gain.cv) == "lies above 0" and side(sb_gain.holdout) != "lies above 0"
    signed = ev.holdout.xg_signed_error
    # no causal claim: a refit on men's league shots only still over-predicts the hold-out
    # (run3-ms3 football probes), so the goal rates are reported, not offered as the cause
    lines.append(
        "The CV population mixes women's and team-selected seasons that "
        f"{'convert more than' if more else 'differ in goal rate from'} a men's league (dev goal "
        f"rate {pct(league)} league, {pct(team)} team-selected, {pct(women)} women; hold-out "
        f"{pct(_goal_rate(held))}). The hold-out {'over' if signed > 0 else 'under'}-prediction "
        f"({signed:+.2%}){' and the CV gain over StatsBomb not carrying over' * carry} "
        f"{'are' if carry else 'is'} reported, not explained: this card does not test whether "
        f"the mix causes {'them' if carry else 'it'}."
    )
    if _settings(ev)["max_leaf_nodes"] == max(gbm_grid(ev)["leaf_grid"]):
        lines.append("The grid's largest leaf count was chosen; a wider grid is a follow-up.")
    if isinstance(ev.spec, ModelSpec) and "phase" in ev.spec.features:
        lines.append(ablation.wide_dfk(ev, benchmark, dev))
    limits = (*lines, *CARD.limitations, ablation.holdout_women(context))
    return dataclasses.replace(CARD, features=gbm_features(ev), limitations=limits)


def summary_table(
    evaluations: Sequence[Evaluation],
    benchmarks: Sequence[Evaluation],
    head: Evaluation,
    penalty: penalties.PenaltyRow,
    sb_gain: Comparison,
) -> str:
    """The README summary block: every model and benchmark, the headline in bold, then lines.

    ``sb_gain`` is the headline model over ``benchmarks[0]``, which the sentence after the
    table names (:func:`benchmark_sentence`); ``penalty`` is the in-game all row of
    :func:`~football_shot_models.penalties.penalty_rows`.

    Raises
    ------
    ValueError
        No benchmark.

    Examples
    --------
    >>> print(summary_table(evs, (sb,), gbm, rows[0], gs))  # doctest: +SKIP
    | Model | CV log loss | CV AUC | CV cal. slope | Hold-out log loss | ...
    """
    if not benchmarks:
        raise ValueError("summary_table needs at least one benchmark, got none")
    r = penalty.rate
    lines = [
        "| Model | CV log loss | CV AUC | CV cal. slope | Hold-out log loss | Hold-out cal. slope "
        "| Hold-out Σ xG error (signed) |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {f'**{e.spec.title}**' if e is head else e.spec.title} | {fmt(e.cv.log_loss)} | "
            f"{fmt(e.cv.auc)} | {fmt(e.cv.cal_slope)} | {fmt(e.holdout.log_loss)} | "
            f"{fmt(e.holdout.cal_slope)} | {e.holdout.xg_signed_error:+.2%} |"
            for e in (*evaluations, *benchmarks)
        ),
        "",
        f"Headline model: {head.spec.title}. {benchmark_sentence(head, sb_gain, benchmarks[0])}",
        "",
        f"Penalties (in-game): P(goal) point estimate {r.mean:.4f} [95% interval {r.low:.4f}, "
        f"{r.high:.4f}], {r.n:,} kicks; shoot-outs reported apart.",
        "",
        penalties.MATCH_XG,
    ]
    return "\n".join(lines) + "\n"


# ---- command line ------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelsRun:
    """Everything :func:`run` evaluated.

    Attributes
    ----------
    evaluations : tuple of Evaluation
        The baselines, the steps and the GBM.
    benchmark, head : Evaluation
        StatsBomb xG and the headline model.
    rows : mapping of str to tuple of Comparison
        The gain rows of each step and of the GBM, by spec name.
    checks : tuple of Check
        :func:`milestone_checks` of the headline model.
    penalties : tuple of PenaltyRow
        :func:`~football_shot_models.penalties.penalty_rows`.
    spread : Spread
        :func:`~football_shot_models.penalties.exchangeability`.
    """

    evaluations: tuple[Evaluation, ...]
    benchmark: Evaluation
    head: Evaluation
    rows: Mapping[str, tuple[Comparison, ...]]
    checks: tuple[Check, ...]
    penalties: tuple[PenaltyRow, ...]
    spread: Spread


def run(
    manifest: StrPath = data.DEFAULT_MANIFEST,
    *,
    docs_dir: StrPath = "docs/models",
    cache: StrPath | None = None,
) -> ModelsRun:
    """Evaluate every model once, pick the headline, write the step, GBM and penalty cards.

    It does not write the exclusion log or the baseline cards (``make xg-baselines`` does).

    Parameters
    ----------
    manifest : str or path, default data.DEFAULT_MANIFEST
        The StatsBomb snapshot manifest.
    docs_dir : str or path, default "docs/models"
        Where ``<name>.md`` is written.
    cache : str or path, optional
        Root of the prediction cache (:class:`~football_shot_models.cache.PredictionCache`):
        every fitted spec reads its predictions there when data, code and spec match, and
        writes them otherwise. ``None`` (the default) fits everything and writes nothing.

    Returns
    -------
    ModelsRun

    Examples
    --------
    >>> run().head.spec.name  # doctest: +SKIP
    'xg-gbm'
    """
    xg = data.load_xg_data(manifest)
    ho = data.pair_holdout(xg.matches)
    dev, held = data.split_holdout(xg.shots, ho)
    folds = cv_folds(dev)
    pc = None if cache is None else PredictionCache(cache, xg.manifest.data_digest)
    ff = xg.freeze_frames
    c = evaluate_candidates(ablation.STEPS, GBM, dev, held, folds, freeze_frames=ff, cache=pc)
    base, steps, gbm = c.baselines, c.steps, c.gbm
    sb = evaluate_benchmark(dev, held, folds)
    variants = ablation.evaluate_variants(dev, held, folds, ff, cache=pc)
    s0, evaluations = base[2], (*base, *steps, gbm)
    by_name = {ev.spec.name: ev for ev in (*base, *steps, *variants)}
    chain = (s0.spec.name, *(ev.spec.name for ev in steps))
    rows = ablation.comparisons(by_name, sb, dev, held, chain=chain)
    head, gbm_gain = select_headline(c.lrs, gbm, dev, held)
    gbm_rows = (gbm_gain, *(compare(f"over {r.spec.title}", r, gbm, dev, held) for r in (s0, sb)))
    s0_gain, sb_gain = (
        gbm_rows[1:] if head is gbm
        else (compare(f"over {r.spec.title}", r, head, dev, held) for r in (s0, sb))
    )  # fmt: skip
    checks = milestone_checks(
        head, sb, s0_gain=s0_gain, sb_gain=sb_gain, gbm_gain=gbm_gain, dev=dev, held=held, s0=s0
    )
    kicks = penalties.penalty_rows(xg.penalties, holdout=ho)
    spread = penalties.exchangeability(xg.penalties)
    context = card_context(xg, dev, held, ho)
    verdicts = {ev.spec.name: verdict(ev, head, gbm_gain) for ev in (*steps, gbm)}
    cards = ablation.step_cards(
        steps, rows, sb, evaluations=evaluations, dev=dev, context=context, verdicts=verdicts,
        checks={head.spec.name: checks},
    )  # fmt: skip
    text = gbm_text(
        gbm, sb, gain=gbm_gain, sb_gain=gbm_rows[2], dev=dev, held=held, context=context
    )
    cards[gbm.spec.name] = render_card(
        gbm, text, evaluations=evaluations, benchmark=sb, dev=dev, context=context,
        provider=STATSBOMB_TEXT, thresholds=(),
        comparisons=gbm_rows, checks=checks if head is gbm else (),
        verdict=f"{verdicts[gbm.spec.name]} {benchmark_sentence(gbm, gbm_rows[2], sb)}",
    )  # fmt: skip
    cards[penalties.PENALTY_NAME] = penalties.penalty_card(kicks, spread, context)
    out = Path(docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, card in cards.items():
        (out / f"{name}.md").write_text(card, encoding="utf-8", newline="\n")
    all_rows = {**rows, gbm.spec.name: gbm_rows}
    for ev in (*evaluations, *variants, sb):
        print(baselines.summary_line(ev))
    for name, gains in all_rows.items():
        for g in gains:
            print(f"{name} | {g.label}: cv {gain_cell(g.cv)} | hold-out {gain_cell(g.holdout)}")
    print(f"headline: {head.spec.name}")
    print(*(f"check | {c.target} | {c.measured} | {c.result}" for c in checks), sep="\n")
    print(summary_table(evaluations, (sb,), head, kicks[0], sb_gain), end="")
    print(f"cards: {out}")
    if pc is not None:
        print(f"cache: {pc.dir} ({len(pc.hits)} read, {len(pc.written)} written)")
    return ModelsRun(evaluations, sb, head, all_rows, checks, kicks, spread)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; exit 0 even when a check is missed (it reports).

    Examples
    --------
    >>> main(["--manifest", "manifests/statsbomb-open-v1.json"])  # doctest: +SKIP
    0
    """
    parser = argparse.ArgumentParser(
        prog="python -m football_shot_models.models",
        description="Evaluate every xG model and the penalty model and write their cards.",
    )
    parser.add_argument("--manifest", default=data.DEFAULT_MANIFEST, help="snapshot manifest")
    parser.add_argument("--docs-dir", default="docs/models", help="where the cards go")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--cache", default="data/cache", help="prediction cache root (git-ignored)")
    where.add_argument("--no-cache", action="store_true", help="fit everything, write no cache")
    args = parser.parse_args(argv)
    try:
        run(args.manifest, docs_dir=args.docs_dir, cache=None if args.no_cache else args.cache)
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"fsm: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
