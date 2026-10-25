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
summed, so the Σ xG error is pooled. For any model fitted with an intercept the pooled error
is close to 0 by construction (fold errors of opposite sign cancel), so
:func:`fold_xg_error` also gives the fold mean and maximum of ``|Σ xG - goals| / goals``.

**Comparing two models.** :func:`bootstrap_gain` gives the log-loss gain of one model over
another on the same shots, with a 95 % percentile interval from resampling **matches**: shots
in one match share the keepers, the game state and the annotators, so resampling shots would
treat them as independent and give too narrow an interval.

**A rate with an interval.** :func:`beta_interval` treats ``goals`` of ``n`` kicks as
binomial draws of one conversion probability with a Beta prior (Jeffreys, Beta(1/2, 1/2), by
default). The posterior is Beta(prior + goals, prior + misses); its mean is the model's
probability and its 2.5 % and 97.5 % quantiles give the interval. Unlike the normal
approximation it stays inside [0, 1] and is defined for 0 or ``n`` goals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Outcomes = npt.NDArray[np.bool_ | np.int64 | np.float64]  # goal or not: bool, or 0 / 1
Op = Literal["<=", ">="]
Metric = Literal["log_loss", "brier", "auc", "xg_error"]  # Scores values that are never None

METRICS: Final[tuple[Metric, ...]] = get_args(Metric)

EPS: Final[float] = 1e-15
N_BOOT: Final[int] = 2000
SEED: Final[int] = 0


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

    @property
    def xg_signed_error(self) -> float:
        """``(Σ xG - goals) / goals``: positive when the model predicts too many goals."""
        return (self.xg - self.goals) / self.goals


@dataclass(frozen=True)
class Threshold:
    """A milestone target: ``metric op value``.

    ``metric`` is one of ``METRICS`` and ``op`` is ``<=`` or ``>=``; anything else is a
    ``ValueError`` when the threshold is made, not when it is checked.
    """

    metric: Metric
    op: Op
    value: float

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"unknown metric {self.metric!r}; expected one of {METRICS}")
        if self.op not in ("<=", ">="):
            raise ValueError(f"unknown op {self.op!r}; expected '<=' or '>='")


@dataclass(frozen=True)
class ThresholdCheck:
    """A threshold, the measured value and whether it is met (equality passes)."""

    threshold: Threshold
    measured: float
    passed: bool


@dataclass(frozen=True)
class Check:
    """One MS3 milestone check, as the card prints it: free text plus a result.

    Attributes
    ----------
    target, measured : str
        The target with its bound, and what was measured.
    result : {"met", "missed", "n/a"}
        ``n/a`` when the check cannot be judged (a group too small, a slope not estimable).
    """

    target: str
    measured: str
    result: Literal["met", "missed", "n/a"]


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


def metric_value(s: Scores, metric: Metric) -> float:
    """One metric of ``s`` by name, typed (no ``getattr``).

    Examples
    --------
    >>> round(metric_value(Scores(100, 10, 10.2, 0.288, 0.082, 0.763, 0.0, 1.0), "xg_error"), 4)
    0.02
    """
    values: dict[Metric, float] = {
        "log_loss": s.log_loss,
        "brier": s.brier,
        "auc": s.auc,
        "xg_error": s.xg_error,
    }
    return values[metric]


def score(y: Outcomes, p: FloatArray) -> Scores:
    """Score probabilities against outcomes.

    Parameters
    ----------
    y : numpy.ndarray of bool, or of 0 / 1 (int64 or float64)
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
    >>> a = Scores(10, 2, 2.5, 0.10, 0.08, 0.70, 0.1, 1.0)
    >>> b = Scores(20, 3, 2.5, 0.20, 0.10, 0.80, 0.3, None)
    >>> c = Scores(30, 5, 6.0, 0.90, 0.12, 0.90, 0.5, 1.0)
    >>> m = mean_scores([a, b, c])
    >>> m.n, m.goals, m.xg, round(m.log_loss, 2), round(m.cal_intercept, 2), m.cal_slope
    (60, 10, 11.0, 0.4, 0.3, None)
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


def fold_sd(folds: Sequence[Scores], metric: Metric) -> float:
    """Standard deviation (ddof=1) of one metric over the folds.

    Examples
    --------
    >>> a = Scores(10, 2, 2.5, 0.30, 0.08, 0.70, 0.1, 1.0)
    >>> b = Scores(20, 3, 2.5, 0.40, 0.10, 0.80, 0.3, None)
    >>> round(fold_sd([a, b], "log_loss"), 4)
    0.0707
    """
    return float(np.std([metric_value(f, metric) for f in folds], ddof=1))


def fold_xg_error(folds: Sequence[Scores]) -> tuple[float, float]:
    """Fold mean and maximum of ``|Σ xG - goals| / goals``: errors that pooling cancels.

    Raises
    ------
    ValueError
        If ``folds`` is empty.

    Examples
    --------
    >>> a = Scores(10, 2, 2.2, 0.3, 0.08, 0.7, 0.0, 1.0)  # +10 %
    >>> b = Scores(10, 2, 1.8, 0.3, 0.08, 0.7, 0.0, 1.0)  # -10 %
    >>> mean_scores([a, b]).xg_error, [round(v, 4) for v in fold_xg_error([a, b])]
    (0.0, [0.1, 0.1])
    """
    if not folds:
        raise ValueError("fold_xg_error needs at least one fold")
    errors = [f.xg_error for f in folds]
    return float(np.mean(errors)), float(max(errors))


def reliability(y: Outcomes, p: FloatArray, bins: int = 10) -> pl.DataFrame:
    """Equal-count bins of ``p`` with their mean prediction and observed goal rate.

    A shot's bin is ``(rank - 1) * bins // n``, with ranks by ``p`` and ties in input order, so
    every bin holds ``n // bins`` or one more shots.

    Parameters
    ----------
    y : numpy.ndarray of bool, or of 0 / 1 (int64 or float64)
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


def check_thresholds(
    s: Scores, thresholds: Sequence[Threshold] = THRESHOLDS
) -> tuple[ThresholdCheck, ...]:
    """Check scores against milestone targets (default: the MS2 ``THRESHOLDS``).

    Examples
    --------
    >>> s = Scores(100, 10, 10.2, 0.2880, 0.0821, 0.7628, 0.0, 1.0)
    >>> [(c.threshold.metric, c.passed) for c in check_thresholds(s)]
    [('log_loss', False), ('brier', False), ('auc', True), ('xg_error', True)]
    >>> [c.passed for c in check_thresholds(s, [Threshold("log_loss", "<=", 0.3)])]
    [True]
    """
    checks = []
    for t in thresholds:
        measured = metric_value(s, t.metric)
        passed = measured <= t.value if t.op == "<=" else measured >= t.value
        checks.append(ThresholdCheck(t, measured, passed))
    return tuple(checks)


def group_weights(
    groups: npt.NDArray[np.generic], n_boot: int, seed: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """The match bootstrap's draws: each shot's group index and the resampling weights.

    A resample draws the ``m`` groups with replacement, so its weights are one multinomial
    draw of ``m`` over ``m`` equal cells. :func:`bootstrap_gain` and :func:`by_group` both
    use these weights, so a seed gives the same resamples in both.

    Parameters
    ----------
    groups : numpy.ndarray
        The resampling unit per shot (``match_id``); no null or NaN, one key type.
    n_boot : int
        Resamples, at least 1.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    inverse : numpy.ndarray
        The index of each shot's group among the sorted unique groups.
    weights : numpy.ndarray
        Shape ``(n_boot, m)``: how often each group is drawn in each resample.

    Raises
    ------
    ValueError
        A null or NaN group, groups of mixed types, ``n_boot`` a bool or ``< 1``, fewer than
        two groups.

    Examples
    --------
    >>> inv, w = group_weights(np.array(["ws:2", "ws:1", "ws:2"]), 4, 0)
    >>> inv.tolist(), w.shape, w.sum(axis=1).tolist()
    ([1, 0, 1], (4, 2), [2, 2, 2, 2])
    """
    if bad := sum(k is None or k != k for k in groups.tolist()):  # k != k: NaN
        raise ValueError(f"groups has {bad} null or NaN key(s)")
    if isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"n_boot must be an int >= 1, got {n_boot!r}")
    try:
        _, inv = np.unique(groups, return_inverse=True)
    except TypeError as e:  # e.g. int and str keys together: not comparable
        raise ValueError(f"groups must be keys of one type (match_id), got: {e}") from e
    m = int(inv.max()) + 1 if inv.size else 0
    if m < 2:
        raise ValueError(f"the bootstrap needs at least 2 groups (matches), got {m}")
    w = np.random.default_rng(seed).multinomial(m, np.full(m, 1.0 / m), size=n_boot)
    return inv, w


@dataclass(frozen=True)
class GainCI:
    """A log-loss gain and its bootstrap interval.

    Attributes
    ----------
    gain : float
        Mean over shots of the reference's log loss minus the new model's; positive means the
        new model is better.
    low, high : float
        The 2.5 % and 97.5 % percentiles of the resampled gains.
    n_boot, groups : int
        Resamples drawn and groups (matches) resampled.
    """

    gain: float
    low: float
    high: float
    n_boot: int
    groups: int


def bootstrap_gain(
    y: Outcomes,
    p_ref: FloatArray,
    p_new: FloatArray,
    groups: npt.NDArray[np.generic],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> GainCI:
    """Log-loss gain of ``p_new`` over ``p_ref``, with a match-bootstrap 95 % interval.

    Each shot's gain is its log loss (probabilities clipped at ``EPS``) under ``p_ref`` minus
    under ``p_new``. A resample draws the groups with replacement (multinomial weights), and
    its gain is the weighted sum of the shot gains over the weighted shot count, so a match
    with many shots counts for its shots. The interval is the percentile interval of the
    ``n_boot`` resamples; it says which matches happened to be sampled, not how a refit would
    vary.

    Parameters
    ----------
    y : numpy.ndarray of bool, or of 0 / 1
        Outcomes, 1-D.
    p_ref, p_new : numpy.ndarray of float
        Probabilities in [0, 1] of the reference and the new model, the shape of ``y``.
    groups : numpy.ndarray
        The resampling unit per shot (``match_id``), the shape of ``y``; no null or NaN.
    n_boot : int, default N_BOOT
        Resamples, at least 1.
    seed : int, default SEED
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    GainCI

    Raises
    ------
    ValueError
        ``y`` not 1-D or a value other than 0 / 1 (NaN included); a shape that differs from
        ``y``'s; a probability NaN or outside [0, 1]; a null or NaN group; groups of mixed
        types; ``n_boot`` a bool or ``< 1``; fewer than two groups (an empty input included).

    Examples
    --------
    >>> y = np.array([1, 0, 0, 1, 0, 0])
    >>> p_ref = np.full(6, 1 / 3)
    >>> p_new = np.array([0.6, 0.2, 0.2, 0.5, 0.3, 0.3])
    >>> ci = bootstrap_gain(y, p_ref, p_new, np.array(["sb:1", "sb:1", "sb:2", "sb:2", "sb:3",
    ...                                                  "sb:3"]))
    >>> round(ci.gain, 4), ci.groups, ci.low <= ci.gain <= ci.high
    (0.2426, 3, True)
    >>> bootstrap_gain(y, p_ref, p_ref, np.arange(6)).gain
    0.0
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError(f"y must be 1-D with values 0 / 1 only (no NaN), got shape {y.shape}")
    for p in (p_ref, p_new):
        if p.shape != y.shape:
            raise ValueError(f"p_ref and p_new must have y's shape {y.shape}, got {p.shape}")
        if not np.all((p >= 0.0) & (p <= 1.0)):
            raise ValueError("probabilities must be in [0, 1] (no NaN)")
    if groups.shape != y.shape:
        raise ValueError(f"groups has shape {groups.shape}, expected {y.shape}")

    def loss(p: FloatArray) -> FloatArray:
        q = np.clip(p, EPS, 1.0 - EPS)
        out: FloatArray = -(y * np.log(q) + (1.0 - y) * np.log1p(-q))
        return out

    diff = loss(p_ref) - loss(p_new)
    inv, w = group_weights(groups, n_boot, seed)
    m = w.shape[1]
    d = np.bincount(inv, weights=diff, minlength=m)
    n = np.bincount(inv, minlength=m).astype(np.float64)
    gains = (w @ d) / (w @ n)
    low, high = np.quantile(gains, [0.025, 0.975])
    return GainCI(float(diff.mean()), float(low), float(high), n_boot, m)


@dataclass(frozen=True)
class GroupRow:
    """Σ xG error and calibration of one group (a competition), with match-bootstrap intervals.

    Attributes
    ----------
    key : str
        The group.
    shots, goals : int
        Shots and goals.
    xg : float
        Σ p.
    signed_error : float or None
        ``(Σ xG - goals) / goals``; ``None`` with 0 goals.
    error_ci : tuple of float or None
        Its 95 % interval over match resamples within the group.
    cal_slope : float or None
        Calibration slope; ``None`` under ``min_shots`` shots or when not estimable.
    slope_ci : tuple of float or None
        Its 95 % interval.
    """

    key: str
    shots: int
    goals: int
    xg: float
    signed_error: float | None
    error_ci: tuple[float, float] | None
    cal_slope: float | None
    slope_ci: tuple[float, float] | None


def _interval(values: FloatArray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def by_group(
    y: Outcomes,
    p: FloatArray,
    keys: pl.Series,
    groups: npt.NDArray[np.generic],
    min_shots: int,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[GroupRow, ...]:
    """Σ xG error and calibration slope per key, each with a match-bootstrap 95 % interval.

    Within each key, a resample draws that key's matches with replacement
    (:func:`group_weights`); the interval is the 2.5-97.5 percentile of the resampled values.
    An interval is ``None`` when its point value is, when the key has fewer than 2 matches, or
    when any resample has no goal (error) or no estimable slope (slope).

    Parameters
    ----------
    y : numpy.ndarray of bool, or of 0 / 1
        Outcomes, 1-D.
    p : numpy.ndarray of float
        Probabilities in [0, 1], the shape of ``y``.
    keys : polars.Series
        The group of each shot, ``String``, no null.
    groups : numpy.ndarray
        The resampling unit of each shot (``match_id``).
    min_shots : int
        A key with fewer shots gets no slope.
    n_boot, seed : int, default N_BOOT, SEED
        Resamples and seed.

    Returns
    -------
    tuple of GroupRow
        One per key, sorted by key.

    Raises
    ------
    ValueError
        ``keys`` not a String Series (a list or a bare ``str`` included) or holding a null;
        ``y`` not 0 / 1; lengths that differ; a probability outside [0, 1] or NaN; no shot.

    Examples
    --------
    >>> y = np.array([1, 0, 0, 1, 0, 0])
    >>> p = np.array([0.5, 0.2, 0.3, 0.4, 0.1, 0.2])
    >>> keys = pl.Series(["A", "A", "A", "A", "B", "B"])
    >>> rows = by_group(y, p, keys, np.array(["ws:1", "ws:1", "ws:2", "ws:2", "ws:3", "ws:4"]),
    ...                 min_shots=10, n_boot=50)
    >>> [(r.key, r.shots, r.goals, round(r.signed_error, 2)) for r in rows if r.goals]
    [('A', 4, 2, -0.3)]
    >>> rows[1].signed_error, rows[1].error_ci, rows[0].cal_slope
    (None, None, None)
    """
    if not isinstance(keys, pl.Series) or keys.dtype != pl.String:
        got = keys.dtype if isinstance(keys, pl.Series) else type(keys).__name__
        raise ValueError(f"keys must be a polars String Series, got {got}")
    yf = np.asarray(y, dtype=np.float64)
    if yf.ndim != 1 or not np.all((yf == 0.0) | (yf == 1.0)):
        raise ValueError(f"y must be 1-D with values 0 / 1 only (no NaN), got shape {yf.shape}")
    if not yf.size:
        raise ValueError("by_group needs at least one shot")
    if p.shape != yf.shape or keys.len() != yf.size or groups.shape != yf.shape:
        raise ValueError(
            f"p, keys and groups must have y's length {yf.size}, got {p.shape}, "
            f"{keys.len()}, {groups.shape}"
        )
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("probabilities must be in [0, 1] (no NaN)")
    if nulls := keys.null_count():
        raise ValueError(f"keys has {nulls} null value(s)")
    rows = []
    for key in sorted(set(keys.to_list())):
        m = (keys == key).to_numpy()
        yk, pk, gk = yf[m], p[m], groups[m]
        goals = int(yk.sum())
        err = None if goals == 0 else float((pk.sum() - goals) / goals)
        slope = calibration(yk, pk)[1] if yk.size >= min_shots else None
        err_ci = slope_ci = None
        if (err is not None or slope is not None) and np.unique(gk).size >= 2:
            inv, w = group_weights(gk, n_boot, seed)
            g_goals = w @ np.bincount(inv, weights=yk)
            if err is not None and np.all(g_goals > 0):
                err_ci = _interval((w @ np.bincount(inv, weights=pk) - g_goals) / g_goals)
            if slope is not None:
                idx = np.arange(yk.size)
                draws = [calibration(yk[i], pk[i])[1] for i in (np.repeat(idx, b[inv]) for b in w)]
                if all(s is not None for s in draws):
                    slope_ci = _interval(np.array(draws, dtype=np.float64))
        rows.append(
            GroupRow(key, int(m.sum()), goals, float(pk.sum()), err, err_ci, slope, slope_ci)
        )
    return tuple(rows)


JEFFREYS: Final[tuple[float, float]] = (0.5, 0.5)
LEVEL: Final[float] = 0.95


@dataclass(frozen=True)
class Rate:
    """A conversion rate with its Beta posterior.

    Attributes
    ----------
    n, goals : int
        Kicks and goals.
    rate : float
        ``goals / n``, the observed rate.
    mean : float
        The posterior mean: the model's P(goal).
    low, high : float
        The equal-tailed ``LEVEL`` posterior interval.
    """

    n: int
    goals: int
    rate: float  # goals / n
    mean: float  # posterior mean: the model's P(goal)
    low: float
    high: float


def beta_interval(goals: int, n: int, *, prior: tuple[float, float] = JEFFREYS) -> Rate:
    """Posterior mean and equal-tailed 95 % interval of a binomial rate under a Beta prior.

    Parameters
    ----------
    goals, n : int
        Successes and trials (numpy integers accepted, bools refused).
    prior : tuple of float, default JEFFREYS
        The Beta prior's two shape parameters, finite and positive.

    Returns
    -------
    Rate

    Raises
    ------
    ValueError
        A bool or non-int ``goals`` / ``n``; ``n < 1``; ``goals`` outside ``0..n``; a prior
        that is not two finite positive numbers.

    Examples
    --------
    >>> r = beta_interval(853, 1118)
    >>> round(r.mean, 4), round(r.low, 4), round(r.high, 4)
    (0.7627, 0.7374, 0.7872)
    >>> round(beta_interval(0, 10).mean, 4)
    0.0455
    """
    for name, v in (("goals", goals), ("n", n)):
        if isinstance(v, bool) or not isinstance(v, int | np.integer):
            raise ValueError(f"{name} must be an int, got {v!r}")
    if n < 1 or not 0 <= goals <= n:
        raise ValueError(f"need n >= 1 and 0 <= goals <= n, got goals={goals}, n={n}")
    a0, b0 = prior
    if not (math.isfinite(a0) and math.isfinite(b0) and a0 > 0 and b0 > 0):
        raise ValueError(f"prior must be two finite positive numbers, got {prior!r}")
    a, b = a0 + goals, b0 + n - goals
    low, high = stats.beta.ppf([(1 - LEVEL) / 2, (1 + LEVEL) / 2], a, b)
    return Rate(int(n), int(goals), goals / n, a / (a + b), float(low), float(high))
