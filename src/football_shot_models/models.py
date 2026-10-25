"""M1.4 and M1.5: a monotone gradient-boosted xG model, the headline rule and the penalty model.

**The GBM** (``xg-gbm``) is scikit-learn's ``HistGradientBoostingClassifier`` on S3's inputs
without the spline terms (trees bend by themselves). Four columns carry a monotone constraint
for football reasons only (:data:`MONOTONE_REASONS`): further from goal at the same opening
angle never helps, a wider opening never hurts, another outfield body in the cone never helps,
and a closer marker never helps. A constraint holds along one column with every other column
held fixed.

**Tuning and calibration stay inside the fit.** :class:`MonotoneGBM` is a
:class:`~football_shot_models.evaluation.DesignFit`: the harness hands it the training rows
of one fit with their design names and matches. Inside, a grouped inner CV by match picks the
leaf count and the number of iterations, and Platt scaling is fitted on the chosen model's
inner out-of-fold scores (cross-fitted by match), then applied to a refit on all the training
rows. The outer test fold never reaches it, so the CV score is honest.

**Headline rule.** The GBM is the headline model only if its grouped-CV log-loss gain over the
best logistic regression has a 95 % match-bootstrap interval that lies above 0
(:func:`headline`); otherwise the LR stays headline and every card says so (:func:`verdict`).
The hold-out gain is reported next to it, and so is the comparison with StatsBomb's own xG
(:func:`statsbomb_sentence`), without the words "beats" or "matches".

**Penalties** (``xg-penalty``) get one probability: the posterior mean of a Beta-binomial
with the Jeffreys prior over every in-game penalty (:func:`penalty_rows`); shoot-out kicks are
reported in their own rows and never pooled. Match xG = Σ non-penalty xG + P(goal | in-game
penalty) × in-game penalties.

Run ``python -m football_shot_models.models`` (``make xg-models``) to evaluate every model
once and write the step cards, the GBM card and the penalty card into ``docs/models/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core.io import snapshot
from scipy import stats
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from football_shot_models import ablation, baselines, data, features
from football_shot_models.cards import (
    LICENSING,
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
    Coefficients,
    Comparison,
    Estimator,
    Evaluation,
    ModelSpec,
    best,
    compare,
    cv_folds,
    evaluate,
    evaluate_benchmark,
    labels,
)
from football_shot_models.metrics import (
    JEFFREYS,
    LEVEL,
    Check,
    Rate,
    beta_interval,
    calibration,
    fold_xg_error,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]
StrPath = str | os.PathLike[str]

# ---- the GBM (spec §3.1, normative) ------------------------------------------------------

LEARNING_RATE: Final[float] = 0.05
L2_REGULARIZATION: Final[float] = 1.0
MIN_SAMPLES_LEAF: Final[int] = 100
LEAVES: Final[tuple[int, ...]] = (8, 16, 32)  # max_leaf_nodes candidates
MAX_ITER: Final[int] = 600
STEP: Final[int] = 50  # iteration counts tried: STEP, 2 * STEP, ..., MAX_ITER
INNER_SPLITS: Final[int] = 4
MONOTONE: Final[Mapping[str, int]] = MappingProxyType(
    {"distance_m": -1, "angle_rad": 1, "opponents_in_cone": -1, "nearest_opponent_m": 1}
)


def make_hgb(
    leaves: int, iters: int, cst: list[int], min_leaf: int = MIN_SAMPLES_LEAF
) -> HistGradientBoostingClassifier:
    """One histogram GBM with the fixed settings, deterministic and without early stopping.

    Parameters
    ----------
    leaves, iters : int
        ``max_leaf_nodes`` and ``max_iter``.
    cst : list of int
        ``monotonic_cst``, one sign per design column.
    min_leaf : int, default MIN_SAMPLES_LEAF
        ``min_samples_leaf``.

    Examples
    --------
    >>> h = make_hgb(8, 100, [-1, 1])
    >>> h.early_stopping, h.class_weight, h.random_state
    (False, None, 0)
    """
    return HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=iters,
        max_leaf_nodes=leaves,
        min_samples_leaf=min_leaf,
        l2_regularization=L2_REGULARIZATION,
        early_stopping=False,  # 'auto' would draw a random, ungrouped validation split
        monotonic_cst=cst,
        random_state=0,
    )


class MonotoneGBM:
    """Monotone HGB, tuned by inner grouped CV, with Platt scaling cross-fitted by match.

    Fitted through the harness only (:meth:`fit_design`), which passes the design names (to
    place the monotone signs) and the match of every training row (to group the inner folds).

    Parameters
    ----------
    monotone : mapping of str to int, default MONOTONE
        Design column -> -1 (non-increasing) or +1 (non-decreasing); other columns are free.
    leaf_grid : tuple of int, default LEAVES
        ``max_leaf_nodes`` candidates.
    max_iter, step : int, default MAX_ITER, STEP
        Iteration counts tried: ``step, 2 * step, ..., max_iter``.
    inner_splits : int, default INNER_SPLITS
        Inner ``GroupKFold`` splits.
    min_samples_leaf : int, default MIN_SAMPLES_LEAF
        Lowered only by unit fixtures; the shipped ``GBM`` uses the constant.

    Attributes
    ----------
    model_ : HistGradientBoostingClassifier or None
        The refit on all training rows with the chosen configuration.
    leaves_, iters_ : int
        The chosen configuration.
    platt_ : tuple of float
        Platt intercept and slope ``(a, b)``: ``p = expit(a + b * raw score)``.
    inner_test_ : tuple of numpy.ndarray
        Inner test row indices, one array per inner split.
    inner_raw_ : numpy.ndarray
        Cross-fitted raw scores (log-odds) of the chosen configuration, one per training row.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(5.0, 30.0, (400, 1))
    >>> y = rng.uniform(size=400) < expit(1.0 - 0.15 * x[:, 0])
    >>> gbm = MonotoneGBM({"d": -1}, leaf_grid=(4,), max_iter=20, step=10, min_samples_leaf=20)
    >>> gbm = gbm.fit_design(x, y, names=("d",), groups=np.repeat(np.arange(40), 10))
    >>> p = gbm.predict_proba(np.linspace(5.0, 30.0, 50)[:, None])[:, 1]
    >>> bool(np.all(np.diff(p) <= 0)), gbm.platt_[1] > 0
    (True, True)
    """

    def __init__(
        self,
        monotone: Mapping[str, int] = MONOTONE,
        *,
        leaf_grid: tuple[int, ...] = LEAVES,
        max_iter: int = MAX_ITER,
        step: int = STEP,
        inner_splits: int = INNER_SPLITS,
        min_samples_leaf: int = MIN_SAMPLES_LEAF,
    ) -> None:
        self.monotone, self.leaf_grid, self.max_iter = monotone, leaf_grid, max_iter
        self.step, self.inner_splits, self.min_samples_leaf = step, inner_splits, min_samples_leaf
        self.model_: HistGradientBoostingClassifier | None = None
        self.leaves_ = self.iters_ = 0
        self.platt_ = (0.0, 1.0)
        self.inner_test_: tuple[IntArray, ...] = ()
        self.inner_raw_: FloatArray = np.empty(0)

    def fit(self, x: FloatArray, y: BoolArray) -> MonotoneGBM:
        """Refused: the model needs design names and match groups.

        Raises
        ------
        TypeError
            Always; the harness calls :meth:`fit_design`.
        """
        raise TypeError("MonotoneGBM needs design names and match groups: use fit_design")

    def fit_design(
        self,
        x: FloatArray,
        y: BoolArray,
        *,
        names: tuple[str, ...],
        groups: npt.NDArray[np.generic],
    ) -> MonotoneGBM:
        """Tune by inner grouped CV, fit Platt on the inner out-of-fold scores, then refit.

        Parameters
        ----------
        x : numpy.ndarray
            The design matrix of the training rows.
        y : numpy.ndarray of bool
            Goal or not.
        names : tuple of str
            The design column names (``design_matrix``).
        groups : numpy.ndarray
            ``match_id`` of every row; inner folds never split a match.

        Returns
        -------
        MonotoneGBM
            ``self``, fitted.

        Raises
        ------
        ValueError
            A constrained column absent from ``names``, or a sign that is a bool, not an
            ``int``, or not -1 / +1; fewer matches than ``inner_splits`` (from scikit-learn);
            constant inner out-of-fold scores (Platt not estimable); a Platt slope <= 0.
        """
        missing = [n for n in self.monotone if n not in names]
        bad = {
            n: s
            for n, s in self.monotone.items()
            if isinstance(s, bool) or not isinstance(s, int) or s not in (-1, 1)
        }
        if missing or bad:
            raise ValueError(
                f"monotone constraints on design columns that are absent {missing} or with a "
                f"sign other than -1 / +1 {bad}"
            )
        cst = [self.monotone.get(n, 0) for n in names]
        stops = self.max_iter // self.step
        raw = {leaves: np.full((stops, y.size), np.nan) for leaves in self.leaf_grid}
        splits = tuple(GroupKFold(n_splits=self.inner_splits).split(x, y, groups=groups))
        for train, test in splits:
            for leaves in self.leaf_grid:
                fitted = make_hgb(leaves, self.max_iter, cst, self.min_samples_leaf)
                fitted.fit(x[train], y[train])
                for i, s in enumerate(fitted.staged_decision_function(x[test])):
                    if (i + 1) % self.step == 0:
                        raw[leaves][(i + 1) // self.step - 1, test] = s
        loss = {
            (leaves, (k + 1) * self.step): float(
                log_loss(y, expit(raw[leaves][k]), labels=[False, True])
            )
            for leaves in self.leaf_grid
            for k in range(stops)
        }
        # ties go to the smaller model: fewer leaves, then fewer iterations
        self.leaves_, self.iters_ = min(loss, key=lambda key: (loss[key], key))
        self.inner_test_ = tuple(np.asarray(test, dtype=np.intp) for _, test in splits)
        self.inner_raw_ = raw[self.leaves_][self.iters_ // self.step - 1]
        if np.ptp(self.inner_raw_) == 0.0:
            raise ValueError(
                "the inner out-of-fold scores are constant (no inner tree could split with "
                f"min_samples_leaf={self.min_samples_leaf}): Platt scaling is not estimable"
            )
        platt = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=1000)
        platt.fit(self.inner_raw_[:, None], y)
        a, b = float(platt.intercept_[0]), float(platt.coef_[0, 0])
        if not b > 0:
            raise ValueError(
                f"Platt slope {b:.4g} <= 0: the inner out-of-fold scores do not rank goals, "
                "and a slope <= 0 would reverse or erase the monotone constraints"
            )
        self.platt_ = (a, b)
        final = make_hgb(self.leaves_, self.iters_, cst, self.min_samples_leaf)
        self.model_ = final.fit(x, y)
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        """``(n, 2)`` probabilities: Platt applied to the refit's raw scores.

        Raises
        ------
        ValueError
            Not fitted.
        """
        if self.model_ is None:
            raise ValueError("MonotoneGBM is not fitted")
        a, b = self.platt_
        p = expit(a + b * self.model_.decision_function(x))
        out: FloatArray = np.column_stack([1.0 - p, p])
        return out

    def get_params(self, deep: bool = True) -> Mapping[str, Any]:
        """The six settings (no class weight: probabilities are never rebalanced)."""
        return {
            "monotone": dict(self.monotone),
            "leaf_grid": self.leaf_grid,
            "max_iter": self.max_iter,
            "step": self.step,
            "inner_splits": self.inner_splits,
            "min_samples_leaf": self.min_samples_leaf,
        }


def gbm_settings(model: Estimator, names: tuple[str, ...]) -> Coefficients:
    """The fitted configuration as card rows: leaves, iterations, Platt, then the sign map.

    Uses the ``ModelSpec.coefficients`` hook; each value is repeated in both columns.

    Raises
    ------
    TypeError
        ``model`` is not a ``MonotoneGBM``.

    Examples
    --------
    >>> gbm = MonotoneGBM({"d": -1})
    >>> gbm.leaves_, gbm.iters_, gbm.platt_ = 16, 200, (0.01, 0.99)
    >>> [term for term, _, _ in gbm_settings(gbm, ("d",))]
    ['max_leaf_nodes', 'iterations', 'platt_intercept', 'platt_slope', 'monotone:d']
    """
    if not isinstance(model, MonotoneGBM):
        raise TypeError(f"expected a MonotoneGBM, not {type(model).__name__}")
    a, b = model.platt_
    rows: list[tuple[str, float, float]] = [
        ("max_leaf_nodes", model.leaves_, model.leaves_),
        ("iterations", model.iters_, model.iters_),
        ("platt_intercept", a, a),
        ("platt_slope", b, b),
    ]
    rows += [(f"monotone:{col}", s, s) for col, s in model.monotone.items()]
    return tuple((term, float(v), float(w)) for term, v, w in rows)


GBM_FEATURES: Final[tuple[str, ...]] = (
    "distance_m",
    "angle_rad",
    "body_part",
    *ablation.CONTEXT,
    *ablation.FRAMES,
)
GBM: Final = ModelSpec("xg-gbm", "Monotone GBM + Platt", GBM_FEATURES, MonotoneGBM, gbm_settings)


def _is_gbm(ev: Evaluation) -> bool:
    # a GBM spec is one whose card rows come from gbm_settings (any sign map, any grid)
    return isinstance(ev.spec, ModelSpec) and ev.spec.coefficients is gbm_settings


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


def verdict(ev: Evaluation, head: Evaluation, gain: Comparison) -> str:
    """The card overview sentence of ``ev``: is it the headline model, and why.

    ``gain`` is :func:`headline`'s comparison (the GBM over the best LR).

    Examples
    --------
    >>> verdict(gbm, gbm, gain)  # doctest: +SKIP
    'This is the headline model: its CV log-loss gain over LR splines + ... lies above 0; ...'
    """
    lr = gain.label.removeprefix("over ")
    cv = gain_cell(gain.cv)
    if ev is head and _is_gbm(head):
        return (
            f"This is the headline model: its CV log-loss gain over {lr} is {cv}, a 95% "
            f"interval that lies above 0; on the hold-out it is {gain_cell(gain.holdout)}, an "
            f"interval that {side(gain.holdout)}."
        )
    if ev is head:
        return (
            f"This is the headline model: the GBM's CV gain over it is {cv}, an interval that "
            "does not lie above 0, so the GBM does not ship as headline."
        )
    if _is_gbm(ev) and not _is_gbm(head):
        return (
            f"Not the headline model: its CV gain over {lr} is {cv}, an interval that does not "
            f"lie above 0, so {lr} stays headline."
        )
    return f"Not the headline model; that is {head.spec.title}."


def statsbomb_sentence(ev: Evaluation, sb_gain: Comparison, benchmark: Evaluation) -> str:
    """The comparison with StatsBomb's own xG, on CV and on the hold-out, with Σ xG.

    ``sb_gain`` is ``compare(..., benchmark, ev, ...)``.

    Examples
    --------
    >>> statsbomb_sentence(gbm, sb_gain, sb)  # doctest: +SKIP
    "Against StatsBomb's own xG: CV gain +0.0021 [+0.0012, +0.0031], an interval that ..."
    """
    h, b = ev.holdout, benchmark.holdout
    return (
        f"Against StatsBomb's own xG: CV gain {gain_cell(sb_gain.cv)}, an interval that "
        f"{side(sb_gain.cv)}; hold-out gain {gain_cell(sb_gain.holdout)}, an interval that "
        f"{side(sb_gain.holdout)}. On the hold-out it predicts {h.xg:.0f} goals for "
        f"{h.goals:,} scored ({h.xg_signed_error:+.2%}); StatsBomb predicts {b.xg:.0f} "
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


def _slope(target: str, slope: float | None) -> Check:
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
        :func:`statsbomb_sentence`; the checks use the log-loss gap).
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
            f"CV log loss within {SB_GAP:g} of StatsBomb xG (StatsBomb CV "
            f"{benchmark.cv.log_loss:.4f} on these shots; the PM brief's {SB_MS2_CV:.4f} is the "
            "MS2 population, with direct corners)",
            f"{title} {head.cv.log_loss:.4f}, StatsBomb {benchmark.cv.log_loss:.4f}: gap "
            f"{gap:+.4f}",
            gap <= SB_GAP,
        ),
        _slope(f"{band}, CV fold mean", head.cv.cal_slope),
        _slope(f"{band}, hold-out", head.holdout.cal_slope),
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
                checks.append(_slope(target, calibration(y, p[mask])[1]))
    pooled = f"(pooled {pct(head.cv.xg_error)})"
    mean, worst = fold_xg_error(head.folds)
    for stat, value, bound in (("mean", mean, FOLD_XG_MEAN), ("max", worst, FOLD_XG_MAX)):
        target = f"Σ xG error, fold {stat} <= {bound:.0%}"
        checks.append(_check(target, f"{pct(value)} {pooled}", value <= bound))
    checks.append(
        _check(
            f"the GBM is headline only if its CV gain {gbm_gain.label} has a 95% interval above 0",
            f"CV {gain_cell(gbm_gain.cv)}; hold-out {gain_cell(gbm_gain.holdout)}, an interval "
            f"that {side(gbm_gain.holdout)}",
            gbm_gain.cv.low > 0,
        )
    )
    return tuple(checks)


# ---- penalty model -----------------------------------------------------------------------

PENALTY_NAME: Final[str] = "xg-penalty"
PENALTY_TITLE: Final[str] = "Penalty conversion (Beta-binomial)"
PENALTY_RANGE: Final[tuple[float, float]] = (0.76, 0.80)
PENALTY_CHANGELOG: Final[tuple[str, ...]] = ("v1 (MS3, M1.5): first version.",)
IN_GAME, SHOOTOUT = "In-game", "Shoot-out"
MATCH_XG: Final[str] = (
    "Match xG = Σ non-penalty xG (headline model) + P(goal | in-game penalty) × in-game "
    "penalties; shoot-out kicks are not part of match xG, and own goals and direct corners "
    "carry no xG."
)


@dataclass(frozen=True)
class PenaltyRow:
    """One group of penalty kicks: its Beta posterior, matches and StatsBomb's mean xG.

    ``matches`` counts distinct ``match_id``; ``provider_xg`` is ``None`` when any kick in
    the group has no provider xG.
    """

    label: str
    rate: Rate
    matches: int
    provider_xg: float | None


def penalty_rows(
    penalties: pl.DataFrame, *, holdout: pl.Expr | None = None
) -> tuple[PenaltyRow, ...]:
    """The penalty model (the first row) and its breakdown; shoot-outs in their own rows.

    Rows, in order, each skipped when it has no kick: in-game all (the model), dev, hold-out,
    men, women, then each family except ``women`` (that family is the women's row); then
    shoot-out all, men, women.

    Parameters
    ----------
    penalties : polars.DataFrame
        ``XgData.penalties``: ``reason``, ``is_goal``, ``gender``, ``family``, ``match_id``,
        ``provider_xg`` and the columns ``holdout`` reads.
    holdout : polars.Expr, optional
        True for hold-out kicks; default ``data.is_holdout()`` (the full competition-season
        pair, so shots and penalties share one key).

    Returns
    -------
    tuple of PenaltyRow

    Raises
    ------
    ValueError
        A ``reason`` other than ``penalty`` / ``shootout``; a null in ``reason``,
        ``is_goal``, ``gender``, ``family``, ``match_id`` or the hold-out mask; no in-game
        penalty.

    Examples
    --------
    >>> kicks = pl.DataFrame({
    ...     "reason": ["penalty", "penalty", "penalty", "shootout"],
    ...     "is_goal": [True, True, False, True],
    ...     "gender": ["male"] * 4,
    ...     "family": ["league or tournament"] * 4,
    ...     "match_id": ["sb:1", "sb:2", "sb:2", "sb:3"],
    ...     "competition_id": ["sb:2", "sb:11", "sb:11", "sb:11"],
    ...     "season_id": ["sb:27"] * 4,
    ...     "provider_xg": [0.78, 0.78, 0.78, None]})
    >>> [(r.label, r.rate.n, r.matches) for r in penalty_rows(kicks)][:3]
    [('In-game: all', 3, 2), ('In-game: dev', 2, 1), ('In-game: hold-out', 1, 1)]
    >>> penalty_rows(kicks)[-1].label, penalty_rows(kicks)[-1].provider_xg
    ('Shoot-out: men', None)
    """
    for col in ("reason", "is_goal", "gender", "family", "match_id"):
        if nulls := penalties[col].null_count():
            raise ValueError(f"penalties: {col} has {nulls} null value(s)")
    if other := sorted(set(penalties["reason"].cast(pl.String)) - {"penalty", "shootout"}):
        raise ValueError(f"penalties: reason {other} is not 'penalty' or 'shootout'")
    held = penalties.select(data.is_holdout() if holdout is None else holdout).to_series()
    if nulls := held.null_count():
        raise ValueError(f"penalties: the hold-out mask has {nulls} null value(s)")
    frame = penalties.with_columns(_held=held)
    game, shoot = (frame.filter(pl.col("reason") == r) for r in ("penalty", "shootout"))
    if game.height == 0:
        raise ValueError("penalties: no in-game penalty to pool")
    men, women, is_held = pl.col("gender") == "male", pl.col("gender") == "female", pl.col("_held")
    groups = [
        *((f"{IN_GAME}: {k}", game.filter(e)) for k, e in (("all", True), ("dev", ~is_held))),
        *((f"{IN_GAME}: {k}", game.filter(e)) for k, e in (("hold-out", is_held), ("men", men))),
        (f"{IN_GAME}: women", game.filter(women)),
        *((f"{IN_GAME}: {f}", game.filter(pl.col("family") == f)) for f in data.FAMILIES[:-1]),
        *((f"{SHOOTOUT}: {k}", shoot.filter(e)) for k, e in (("all", True), ("men", men))),
        (f"{SHOOTOUT}: women", shoot.filter(women)),
    ]
    rows = []
    for label, g in (pair for pair in groups if pair[1].height):
        xg = g["provider_xg"].cast(pl.Float64)
        mean = None if xg.null_count() else float(xg.sum()) / g.height
        rate = beta_interval(int(g["is_goal"].sum()), g.height)
        rows.append(PenaltyRow(label, rate, g["match_id"].n_unique(), mean))
    return tuple(rows)


@dataclass(frozen=True)
class Spread:
    """How exchangeable the in-game kicks are: dispersion across competition-seasons, takers.

    ``dispersion`` (Pearson chi2 / df) and ``p_value`` are ``None`` with one group or a
    pooled rate of 0 or 1.
    """

    groups: int
    dispersion: float | None
    df: int
    p_value: float | None
    top_taker_kicks: int
    kicks: int


def exchangeability(penalties: pl.DataFrame) -> Spread:
    """Pearson dispersion of in-game conversion across competition-seasons, and the top taker.

    chi2 = Σ over competition-seasons of (goals - n p)² / (n p (1 - p)), with p the pooled
    rate, on ``groups - 1`` df: a dispersion near 1 is what binomial noise alone gives.

    Raises
    ------
    ValueError
        No in-game penalty; a null ``player_id`` or ``is_goal`` among them.

    Examples
    --------
    >>> kicks = pl.DataFrame({"reason": ["penalty"] * 4, "is_goal": [True, False, True, True],
    ...                       "competition_id": ["sb:1", "sb:1", "sb:2", "sb:2"],
    ...                       "season_id": ["sb:9"] * 4,
    ...                       "player_id": ["sb:p1", "sb:p1", "sb:p1", "sb:p2"]})
    >>> s = exchangeability(kicks)
    >>> s.groups, s.df, round(s.dispersion, 4), s.top_taker_kicks, s.kicks
    (2, 1, 1.3333, 3, 4)
    """
    game = penalties.filter(pl.col("reason") == "penalty")
    if game.height == 0:
        raise ValueError("penalties: no in-game penalty")
    for col in ("player_id", "is_goal"):
        if nulls := game[col].null_count():
            raise ValueError(f"penalties: {col} has {nulls} null value(s) among in-game kicks")
    by = game.group_by("competition_id", "season_id").agg(n=pl.len(), goals=pl.col("is_goal").sum())
    n, k = by["n"].to_numpy().astype(np.float64), by["goals"].to_numpy().astype(np.float64)
    p, df = float(k.sum() / n.sum()), by.height - 1
    dispersion = p_value = None
    if df > 0 and 0.0 < p < 1.0:
        chi2 = float(np.sum((k - n * p) ** 2 / (n * p * (1.0 - p))))
        dispersion, p_value = chi2 / df, float(stats.chi2.sf(chi2, df))
    top = int(game.group_by("player_id").len()["len"].to_numpy().max())
    return Spread(by.height, dispersion, df, p_value, top, game.height)


def _interval(r: Rate) -> str:
    return f"{r.mean:.4f} [{r.low:.4f}, {r.high:.4f}]"


def _constant_loss(r: Rate, q: float | None) -> str:
    if q is None or not 0.0 < q < 1.0:
        return "n/a"
    return f"{-(r.goals * math.log(q) + (r.n - r.goals) * math.log1p(-q)) / r.n:.4f}"


def _penalty_check(r: Rate) -> list[str]:
    lo, hi = PENALTY_RANGE
    bound, name = (lo, "lower") if r.mean - lo <= hi - r.mean else (hi, "upper")
    # a mean on a bound is inside the window: "above the lower", "below the upper"
    where = "above" if r.mean > bound or (name == "lower" and r.mean == bound) else "below"
    result = "met" if lo <= r.mean <= hi else "missed"
    lines = [
        "| Target | Measured | Result |",
        "|---|---|---|",
        f"| posterior mean in [{lo:.2f}, {hi:.2f}] (milestone wording) | {_interval(r)}; "
        f"{abs(r.mean - bound):.4f} {where} the {name} bound | {result} |",
    ]
    if outside := [w for w, out in (("below", r.low < lo), ("above", r.high > hi)) if out]:
        width, window = r.high - r.low, hi - lo
        lines += [
            "",
            f"The 95% interval is {width:.3f} wide, {'wider than' if width > window else 'against'}"
            f" the {window:.2f} window, and extends {' and '.join(outside)} it: the check is on "
            "the point estimate.",
        ]
    return lines


def _spread_line(s: Spread) -> str:
    disp = f"the Pearson dispersion is not estimable ({s.df} df)"
    if s.dispersion is not None and s.p_value is not None:
        disp = f"the Pearson dispersion is {s.dispersion:.2f} on {s.df} df (p {s.p_value:.2f})"
    return (
        f"Kicks are pooled as exchangeable: across {s.groups:,} competition-seasons {disp}, and "
        f"the most frequent taker has {s.top_taker_kicks:,} of {s.kicks:,} kicks."
    )


def penalty_card(rows: Sequence[PenaltyRow], spread: Spread, context: CardContext) -> str:
    """The ``xg-penalty`` model card, from :func:`penalty_rows` and :func:`exchangeability`.

    Raises
    ------
    ValueError
        ``rows`` does not start with the in-game all row.

    Examples
    --------
    >>> card = penalty_card(penalty_rows(xg.penalties), exchangeability(xg.penalties),
    ...                     card_context(xg, dev, held))  # doctest: +SKIP
    >>> card.splitlines()[0]  # doctest: +SKIP
    '# Model card: Penalty conversion (Beta-binomial)'
    """
    if not rows or rows[0].label != f"{IN_GAME}: all":
        raise ValueError(f"penalty_card: the first row must be '{IN_GAME}: all'")
    by, c, r, (a0, b0) = {row.label: row for row in rows}, context, rows[0].rate, JEFFREYS
    dev, held = by.get(f"{IN_GAME}: dev"), by.get(f"{IN_GAME}: hold-out")
    shoot = by.get(f"{SHOOTOUT}: all")
    validation = benchmarks = [
        f"The {'hold-out' if held is None else 'dev'} has no in-game penalties."
    ]
    if dev is not None and held is not None:
        h, inside = held.rate, held.rate.low <= dev.rate.mean <= held.rate.high
        validation = [
            f"The hold-out ({c.holdout_label}) has {h.n:,} in-game kicks, so its 95% interval "
            f"[{h.low:.4f}, {h.high:.4f}] is {h.high - h.low:.3f} wide and this is a weak test: "
            f"the dev estimate {dev.rate.mean:.4f} lies {'inside' if inside else 'outside'} it."
        ]
        benchmarks = [
            f"Each model as a constant on the {h.n:,} hold-out kicks:",
            "",
            "| Model | Kicks | Log loss |",
            "|---|---|---|",
            f"| Dev posterior mean ({dev.rate.mean:.4f}) | {h.n:,} | "
            f"{_constant_loss(h, dev.rate.mean)} |",
            f"| StatsBomb mean xG ({fmt(held.provider_xg)}) | {h.n:,} | "
            f"{_constant_loss(h, held.provider_xg)} |",
        ]
    families = (*data.FAMILIES[:-1], "women")
    spread_by_family = "; ".join(
        f"{f} {by[k].rate.rate:.3f} ({by[k].rate.n:,} kicks)"
        for f in families
        if (k := f"{IN_GAME}: {f}") in by
    )
    shootouts = (
        f"{shoot.rate.n:,} shoot-out kicks (period 5) cluster in {shoot.matches:,} matches; they "
        "are reported apart, never pooled, and their interval treats kicks of one shoot-out as "
        "independent."
        if shoot
        else "There are no shoot-out kicks."
    )
    metrics = [
        f"| {row.label} | {row.rate.n:,} | {row.matches:,} | {row.rate.goals:,} | "
        f"{row.rate.rate:.4f} | {_interval(row.rate)} | {fmt(row.provider_xg)} |"
        for row in rows
    ]
    sections: list[tuple[str, list[str]]] = [
        ("Overview", [
            f"- **Name / version:** `{PENALTY_NAME}` v{len(PENALTY_CHANGELOG)}",
            "- **Task:** P(goal | in-game penalty), periods 1–4; shoot-out kicks reported apart",
            "- **Status:** experimental",
            f"- M1.5: pooled conversion of in-game penalties. P(goal) = {_interval(r)} (95% "
            f"interval) from {r.n:,} kicks in {rows[0].matches:,} matches.",
        ]),
        ("Intended use", [
            f"The penalty term of match xG. {MATCH_XG} Not for rating takers or keepers: every "
            "in-game penalty gets the same probability."
        ]),
        ("Data", [
            f"- **Source(s):** StatsBomb open data, revision `{c.source_revision}`; snapshot "
            f"`{c.manifest_name}`, data digest `{c.data_digest}`",
            "- **Rows:** the shots the xG population excludes as in-game penalties, dev and "
            f"hold-out ({c.holdout_label}) together, from {spread.groups:,} competition-seasons.",
        ]),
        ("Target", ["`is_goal`: the penalty kick went in."]),
        ("Features", ["None: one probability for every in-game penalty."]),
        ("Model", [
            f"Beta-binomial with the Jeffreys prior Beta({a0:g}, {b0:g}): the posterior is "
            f"Beta({a0:g} + goals, {b0:g} + misses); P(goal) is its mean and the interval its "
            f"{LEVEL:.0%} equal-tailed interval. Nothing is fitted per taker, keeper or "
            "competition."
        ]),
        ("Validation design", ["Nothing is tuned; the model pools all kicks.", *validation]),
        ("Metrics", [
            "| Group | Kicks | Matches | Goals | Conversion | P(goal) [95% interval] "
            "| StatsBomb mean xG |",
            "|---|---|---|---|---|---|---|",
            *metrics, "", "### Milestone check", "", *_penalty_check(r),
        ]),
        ("Benchmarks", benchmarks),
        ("Limitations & known biases", [
            "- No taker, keeper or foot: the model cannot tell a specialist from anyone else.",
            f"- {_spread_line(spread)}",
            f"- In-game conversion by family: {spread_by_family}.",
            f"- {shootouts}",
        ]),
        ("Data licensing", [LICENSING]),
        ("Changelog", [f"- {item}" for item in PENALTY_CHANGELOG]),
    ]  # fmt: skip
    body = "\n\n".join(f"## {title}\n" + "\n".join(text) for title, text in sections)
    return f"# Model card: {PENALTY_TITLE}\n\n{body}\n"


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


def gbm_features(ev: Evaluation) -> tuple[str, ...]:
    """The GBM card's Features section: the feature table, then the fitted sign map.

    The sign lines come from the ``monotone:`` rows of ``ev.coefficients`` (the dev fit), so
    a non-default map prints as fitted.

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
        *(ablation.FEATURE_ROWS[f] for f in feats),
        "",
        "Monotone constraints of the fitted model (every other column is free):",
        "",
        *(
            f"- {col}: {'non-increasing' if s < 0 else 'non-decreasing'}"
            + (f" — {MONOTONE_REASONS[col]}" if col in MONOTONE_REASONS else "")
            for col, s in signs
        ),
        "",
        MONOTONE_NOTE,
    )


def _gbm_model(ev: Evaluation, c: CardContext) -> list[str]:
    h, s = make_hgb(LEAVES[0], STEP, []), _settings(ev)  # h: the fixed settings of every fit
    a, b = s["platt_intercept"], s["platt_slope"]
    near = abs(b - 1.0) <= NEAR_IDENTITY and abs(a) <= NEAR_IDENTITY
    return [
        f"`HistGradientBoostingClassifier(learning_rate={h.learning_rate:g}, l2_regularization="
        f"{h.l2_regularization:g}, min_samples_leaf={h.min_samples_leaf}, early_stopping="
        f"{h.early_stopping}, random_state={h.random_state})`, `monotonic_cst` placed by column "
        f"name, no class weights. scikit-learn {c.sklearn_version}.",
        "",
        f"Tuning inside each fit: {INNER_SPLITS}-fold `GroupKFold` by match on its training rows,"
        f" `max_leaf_nodes` in {LEAVES} and {STEP}, {2 * STEP}, ..., {MAX_ITER} iterations "
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
    model=_gbm_model,
    limitations=(*ablation.FRAME_LIMITS, *ablation.CONTEXT_LIMITS, *ablation.POOLED_LIMITS),
    changelog=("v1 (MS3, M1.4): first version.",),
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
    lines.append(
        "The CV population mixes women's and team-selected seasons that "
        f"{'convert more than' if more else 'differ in goal rate from'} a men's league (dev goal "
        f"rate {pct(league)} league, {pct(team)} team-selected, {pct(women)} women; hold-out "
        f"{pct(_goal_rate(held))}); a likely cause, not a measured one, of the hold-out "
        f"{'over' if signed > 0 else 'under'}-prediction ({signed:+.2%})"
        f"{' and of the CV gain over StatsBomb not carrying over' * carry}."
    )
    if _settings(ev)["max_leaf_nodes"] == max(LEAVES):
        lines.append("The grid's largest leaf count was chosen; a wider grid is a follow-up.")
    if isinstance(ev.spec, ModelSpec) and "phase" in ev.spec.features:
        lines.append(ablation.wide_dfk(ev, benchmark, dev))
    limits = (*lines, *CARD.limitations)
    return dataclasses.replace(CARD, features=gbm_features(ev), limitations=limits)


def summary_table(
    evaluations: Sequence[Evaluation],
    benchmarks: Sequence[Evaluation],
    head: Evaluation,
    penalty: PenaltyRow,
    sb_gain: Comparison,
) -> str:
    """The README summary block: every model and benchmark, the headline in bold, then lines.

    ``sb_gain`` is the headline model over ``benchmarks[0]`` (StatsBomb xG); ``penalty`` is
    the in-game all row of :func:`penalty_rows`.

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
        raise ValueError("summary_table needs a benchmark (StatsBomb xG)")
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
        f"Headline model: {head.spec.title}. {statsbomb_sentence(head, sb_gain, benchmarks[0])}",
        "",
        f"Penalties (in-game): P(goal) point estimate {r.mean:.4f} [95% interval {r.low:.4f}, "
        f"{r.high:.4f}], {r.n:,} kicks; shoot-outs reported apart.",
        "",
        MATCH_XG,
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
        :func:`penalty_rows`.
    spread : Spread
        :func:`exchangeability`.
    """

    evaluations: tuple[Evaluation, ...]
    benchmark: Evaluation
    head: Evaluation
    rows: Mapping[str, tuple[Comparison, ...]]
    checks: tuple[Check, ...]
    penalties: tuple[PenaltyRow, ...]
    spread: Spread


def run(
    manifest: StrPath = data.DEFAULT_MANIFEST, *, docs_dir: StrPath = "docs/models"
) -> ModelsRun:
    """Evaluate every model once, pick the headline, write the step, GBM and penalty cards.

    It does not write the exclusion log or the baseline cards (``make xg-baselines`` does).

    Parameters
    ----------
    manifest : str or path, default data.DEFAULT_MANIFEST
        The StatsBomb snapshot manifest.
    docs_dir : str or path, default "docs/models"
        Where ``<name>.md`` is written.

    Returns
    -------
    ModelsRun

    Examples
    --------
    >>> run().head.spec.name  # doctest: +SKIP
    'xg-gbm'
    """
    xg = data.load_xg_data(manifest)
    dev, held = data.split_holdout(xg.shots)
    folds = cv_folds(dev)
    base = tuple(evaluate(b, dev, held, folds) for b in baselines.BASELINES)
    sb = evaluate_benchmark(dev, held, folds)
    steps, variants = ablation.evaluate_steps(dev, held, folds, xg.freeze_frames)
    gbm = evaluate(GBM, dev, held, folds, freeze_frames=xg.freeze_frames)
    s0, evaluations = base[2], (*base, *steps, gbm)
    by_name = {ev.spec.name: ev for ev in (*base, *steps, *variants)}
    chain = (s0.spec.name, *(ev.spec.name for ev in steps))
    rows = ablation.comparisons(by_name, sb, dev, held, chain=chain)
    head, gbm_gain = headline(best((*base, *steps)), gbm, dev, held)
    gbm_rows = (gbm_gain, *(compare(f"over {r.spec.title}", r, gbm, dev, held) for r in (s0, sb)))
    s0_gain, sb_gain = (
        gbm_rows[1:] if head is gbm
        else (compare(f"over {r.spec.title}", r, head, dev, held) for r in (s0, sb))
    )  # fmt: skip
    checks = milestone_checks(
        head, sb, s0_gain=s0_gain, sb_gain=sb_gain, gbm_gain=gbm_gain, dev=dev, held=held, s0=s0
    )
    penalties, spread = penalty_rows(xg.penalties), exchangeability(xg.penalties)
    context = card_context(xg, dev, held)
    verdicts = {ev.spec.name: verdict(ev, head, gbm_gain) for ev in (*steps, gbm)}
    cards = ablation.step_cards(
        steps, rows, sb, evaluations=evaluations, dev=dev, context=context, verdicts=verdicts,
        checks={head.spec.name: checks},
    )  # fmt: skip
    text = gbm_text(
        gbm, sb, gain=gbm_gain, sb_gain=gbm_rows[2], dev=dev, held=held, context=context
    )
    cards[gbm.spec.name] = render_card(
        gbm, text, evaluations=evaluations, benchmark=sb, dev=dev, context=context, thresholds=(),
        comparisons=gbm_rows, checks=checks if head is gbm else (),
        verdict=f"{verdicts[gbm.spec.name]} {statsbomb_sentence(gbm, gbm_rows[2], sb)}",
    )  # fmt: skip
    cards[PENALTY_NAME] = penalty_card(penalties, spread, context)
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
    print(summary_table(evaluations, (sb,), head, penalties[0], sb_gain), end="")
    print(f"cards: {out}")
    return ModelsRun(evaluations, sb, head, all_rows, checks, penalties, spread)


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
    args = parser.parse_args(argv)
    try:
        run(args.manifest, docs_dir=args.docs_dir)
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"fsm: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
