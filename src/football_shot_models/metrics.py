"""Probability scores for xG: log loss, Brier, AUC, calibration and Σ xG against goals.

An xG model is a probability forecast, so it is judged by proper scoring rules (log loss and
Brier), by ranking (ROC AUC) and by calibration: does a shot given 0.2 go in one time in
five? Calibration is summarised by logistic recalibration of ``y`` on ``logit(p)``:

- the **intercept** (calibration-in-the-large) fits ``y ~ a + offset(logit p)``; 0 means
  the average prediction matches the goal rate;
- the **slope** fits ``y ~ a + b logit(p)``; 1 means the predictions are neither too
  extreme (b < 1) nor too timid (b > 1).

Both are maximum-likelihood fits by Newton steps, and ``None`` when not estimable (a
constant ``p`` has no slope; perfect separation has no finite one).

Grouped cross-validation reports the **mean over folds** for log loss, Brier and AUC: an AUC
pooled over folds would compare probabilities from different models. Σ xG and goals are
summed, so the Σ xG error is pooled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Op = Literal["<=", ">="]

EPS: Final[float] = 1e-15


@dataclass(frozen=True)
class Scores:
    """Scores of probabilities ``p`` against outcomes ``y``.

    Attributes
    ----------
    n, goals : int
        Shots and goals.
    xg : float
        Σ p.
    log_loss, brier, auc : float
        Log loss, Brier score and ROC AUC.
    cal_intercept, cal_slope : float or None
        Calibration intercept and slope; ``None`` when not estimable.
    """

    n: int
    goals: int
    xg: float
    log_loss: float
    brier: float
    auc: float
    cal_intercept: float | None
    cal_slope: float | None

    @property
    def xg_error(self) -> float:
        """``|Σ xG - goals| / goals``, a fraction."""
        return abs(self.xg - self.goals) / self.goals


@dataclass(frozen=True)
class Threshold:
    """A milestone target: ``metric op value`` (``Scores`` attribute, ``<=`` or ``>=``)."""

    metric: str
    op: Op
    value: float


@dataclass(frozen=True)
class ThresholdCheck:
    """A threshold, the measured value and whether it is met (equality passes)."""

    threshold: Threshold
    measured: float
    passed: bool


THRESHOLDS: Final[tuple[Threshold, ...]] = (
    Threshold("log_loss", "<=", 0.275),
    Threshold("brier", "<=", 0.077),
    Threshold("auc", ">=", 0.75),
    Threshold("xg_error", "<=", 0.03),
)


def _check_pair(y: npt.NDArray[np.generic], p: FloatArray) -> None:
    if y.ndim != 1 or p.ndim != 1:
        raise ValueError(f"y and p must be 1-D, got shapes {y.shape} and {p.shape}")
    if y.size != p.size:
        raise ValueError(f"y and p differ in length: {y.size} vs {p.size}")


def _logit(p: FloatArray) -> FloatArray:
    q = np.clip(p, EPS, 1.0 - EPS)
    out: FloatArray = np.log(q) - np.log1p(-q)
    return out


def _irls(z: FloatArray, y: FloatArray, offset: FloatArray) -> FloatArray | None:
    """Logistic maximum-likelihood coefficients by Newton steps; None if they do not converge."""
    beta: FloatArray = np.zeros(z.shape[1])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(100):
            mu = 1.0 / (1.0 + np.exp(-(offset + z @ beta)))
            w = mu * (1.0 - mu)
            try:
                step = np.linalg.solve(z.T @ (w[:, None] * z), z.T @ (y - mu))
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(step)):
                return None
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10:
                return beta
    return None


def calibration(y: FloatArray, p: FloatArray) -> tuple[float | None, float | None]:
    """(intercept with logit(p) as offset, slope of y ~ a + b logit(p)); None = not estimable.

    Parameters
    ----------
    y : numpy.ndarray
        Outcomes as 0.0 / 1.0.
    p : numpy.ndarray
        Probabilities.

    Returns
    -------
    tuple of (float or None, float or None)
        Calibration intercept and slope.

    Examples
    --------
    >>> y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    >>> p = np.array([0.2, 0.7, 0.4, 0.6, 0.3, 0.5])
    >>> [round(v, 3) for v in calibration(y, p)]
    [0.228, 1.557]
    >>> calibration(y, np.full(6, 0.5))
    (0.0, None)
    """
    lp = _logit(p)
    one = np.ones((lp.size, 1))
    intercept = _irls(one, y, lp)
    slope = _irls(np.column_stack([one, lp]), y, np.zeros(lp.size)) if np.ptp(lp) > 0 else None
    return (
        None if intercept is None else float(intercept[0]),
        None if slope is None else float(slope[1]),
    )


def score(y: BoolArray, p: FloatArray) -> Scores:
    """Score probabilities against outcomes.

    Parameters
    ----------
    y : numpy.ndarray of bool
        Goal or not, at least one of each.
    p : numpy.ndarray of float
        Finite probabilities in [0, 1], same length.

    Returns
    -------
    Scores

    Raises
    ------
    ValueError
        Not 1-D, different lengths, empty, one class only, or ``p`` not finite in [0, 1].

    Examples
    --------
    >>> s = score(np.array([True, False, True, False]), np.array([0.9, 0.1, 0.4, 0.6]))
    >>> round(s.log_loss, 4), round(s.brier, 4), s.auc, s.goals, round(s.xg, 4)
    (0.5108, 0.185, 0.75, 2, 2.0)
    """
    _check_pair(y, p)
    if y.size == 0:
        raise ValueError("cannot score an empty set of shots")
    goals = int(np.count_nonzero(y))
    if goals in (0, y.size):
        raise ValueError(f"y needs both classes; got {goals} goal(s) in {y.size} shot(s)")
    if not (np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0):
        raise ValueError("p must be finite and in [0, 1]")
    intercept, slope = calibration(y.astype(np.float64), p)
    return Scores(
        n=int(y.size),
        goals=goals,
        xg=float(np.sum(p)),
        log_loss=float(log_loss(y, p)),
        brier=float(brier_score_loss(y, p)),
        auc=float(roc_auc_score(y, p)),
        cal_intercept=intercept,
        cal_slope=slope,
    )


def mean_scores(folds: Sequence[Scores]) -> Scores:
    """The grouped-CV score: fold means of the metrics, sums of ``n``, ``goals`` and ``xg``.

    Parameters
    ----------
    folds : sequence of Scores
        One per test fold.

    Returns
    -------
    Scores
        A calibration value is ``None`` when it is ``None`` in any fold.

    Raises
    ------
    ValueError
        If ``folds`` is empty.

    Examples
    --------
    >>> a = Scores(10, 2, 2.5, 0.30, 0.08, 0.70, 0.1, 1.0)
    >>> b = Scores(20, 3, 2.5, 0.40, 0.10, 0.80, 0.3, None)
    >>> m = mean_scores([a, b])
    >>> m.n, m.goals, m.xg, round(m.log_loss, 2), round(m.cal_intercept, 2), m.cal_slope
    (30, 5, 5.0, 0.35, 0.2, None)
    """
    if not folds:
        raise ValueError("mean_scores needs at least one fold")

    def mean(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return float(np.mean(present)) if len(present) == len(values) else None

    return Scores(
        n=sum(f.n for f in folds),
        goals=sum(f.goals for f in folds),
        xg=float(sum(f.xg for f in folds)),
        log_loss=float(np.mean([f.log_loss for f in folds])),
        brier=float(np.mean([f.brier for f in folds])),
        auc=float(np.mean([f.auc for f in folds])),
        cal_intercept=mean([f.cal_intercept for f in folds]),
        cal_slope=mean([f.cal_slope for f in folds]),
    )


def fold_sd(folds: Sequence[Scores], metric: str) -> float:
    """Standard deviation (ddof=1) of one metric over the folds.

    Examples
    --------
    >>> a = Scores(10, 2, 2.5, 0.30, 0.08, 0.70, 0.1, 1.0)
    >>> b = Scores(20, 3, 2.5, 0.40, 0.10, 0.80, 0.3, None)
    >>> round(fold_sd([a, b], "log_loss"), 4)
    0.0707
    """
    return float(np.std([getattr(f, metric) for f in folds], ddof=1))


def reliability(y: BoolArray, p: FloatArray, bins: int = 10) -> pl.DataFrame:
    """Equal-count bins of ``p`` with their mean prediction and observed goal rate.

    A shot's bin is ``(rank - 1) * bins // n``, with ranks by ``p`` and ties in input order, so
    every bin holds ``n // bins`` or one more shots.

    Parameters
    ----------
    y : numpy.ndarray of bool
        Outcomes.
    p : numpy.ndarray of float
        Probabilities.
    bins : int, default 10
        Number of bins, at most ``len(p)``.

    Returns
    -------
    polars.DataFrame
        ``bin`` (Int64), ``shots`` (Int64), ``mean_xg``, ``goal_rate`` (Float64), by bin.

    Raises
    ------
    ValueError
        ``bins < 1``, fewer shots than bins, or ``y`` and ``p`` not matching 1-D arrays.

    Examples
    --------
    >>> y = np.array([False, False, True, True])
    >>> reliability(y, np.array([0.1, 0.3, 0.8, 0.6]), bins=2).rows()
    [(0, 2, 0.2, 0.0), (1, 2, 0.7, 1.0)]
    """
    _check_pair(y, p)
    if bins < 1 or y.size < bins:
        raise ValueError(f"bins must be in 1..{y.size} (the number of shots), got {bins}")
    n = y.size
    frame = pl.DataFrame({"p": p, "y": y.astype(np.float64)})
    return (
        frame.with_columns(bin=(pl.col("p").rank("ordinal").cast(pl.Int64) - 1) * bins // n)
        .group_by("bin")
        .agg(
            shots=pl.len().cast(pl.Int64),
            mean_xg=pl.col("p").mean(),
            goal_rate=pl.col("y").mean(),
        )
        .sort("bin")
    )


def check_thresholds(s: Scores) -> tuple[ThresholdCheck, ...]:
    """Check scores against the milestone ``THRESHOLDS``.

    Examples
    --------
    >>> s = Scores(100, 10, 10.2, 0.2880, 0.0821, 0.7628, 0.0, 1.0)
    >>> [(c.threshold.metric, c.passed) for c in check_thresholds(s)]
    [('log_loss', False), ('brier', False), ('auc', True), ('xg_error', True)]
    """
    checks = []
    for t in THRESHOLDS:
        measured = float(getattr(s, t.metric))
        passed = measured <= t.value if t.op == "<=" else measured >= t.value
        checks.append(ThresholdCheck(t, measured, passed))
    return tuple(checks)
