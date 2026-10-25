"""One evaluation harness for every xG model: grouped CV, the hold-out and the benchmark.

A model is a :class:`ModelSpec`: its features and a ``make`` that returns a fresh,
scikit-learn style estimator. Every model, from the base rate to a GBM, goes through the same
:func:`evaluate` on the same :func:`cv_folds`, so "beats the baseline on the same folds" is a
property of the harness, not of whoever copies the CV loop.

**Validation.** Shots from one match share the game state, the keepers and the annotators,
so folds are grouped by ``match_id`` (5-fold ``GroupKFold``, no shuffle). The hold-out
competition-season is scored by the fit on all of dev. Each fit sees its training rows only:
``make`` builds a new estimator per fit, so a scaler inside it is fitted on training rows.

**Benchmark.** A provider's own xG is a :class:`Benchmark`: a column that is scored on the
same folds and never fitted. :func:`fit_predict` and :func:`evaluate` refuse it, so it cannot
be scored as a model by mistake.

**Comparison.** :func:`compare` gives the log-loss gain of one evaluation over another with a
match-bootstrap interval (:func:`~football_shot_models.metrics.bootstrap_gain`), and refuses
two evaluations that were not scored on the same folds and rows: a gain between misaligned
predictions would be wrong without any error.

**Fits fail closed.** A ``ConvergenceWarning`` and, for the ``newton-cholesky`` solver, a
singular Hessian (``scipy.linalg.LinAlgWarning``, after which scikit-learn would quietly switch
to lbfgs) are errors, so every fit is the one the model card describes.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from football_shot_models.features import design_matrix
from football_shot_models.metrics import GainCI, Scores, bootstrap_gain, mean_scores, score

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]
Folds = tuple[tuple[IntArray, IntArray], ...]
Coefficients = tuple[tuple[str, float, float], ...]

N_SPLITS = 5


class Estimator(Protocol):
    """The scikit-learn estimator API the harness uses."""

    def fit(self, x: FloatArray, y: BoolArray) -> object: ...

    def predict_proba(self, x: FloatArray) -> FloatArray: ...

    def get_params(self, deep: bool = True) -> Mapping[str, Any]: ...


@runtime_checkable
class DesignFit(Protocol):
    """An estimator fitted with the design column names and the match of each training row.

    The harness calls ``fit_design`` instead of ``fit`` for such a model, with the training
    rows of the fit only, so tuning or calibration inside it never sees a test fold.
    """

    def fit_design(
        self,
        x: FloatArray,
        y: BoolArray,
        *,
        names: tuple[str, ...],
        groups: npt.NDArray[np.generic],
    ) -> object: ...


class BaseRate:
    """Predicts the training goal rate for every shot; it reads no feature.

    Examples
    --------
    >>> BaseRate().fit(np.empty((4, 0)), np.array([True, False, False, False])).predict_proba(
    ...     np.empty((2, 0)))[:, 1].tolist()
    [0.25, 0.25]
    """

    def __init__(self) -> None:
        self.rate: float | None = None

    def fit(self, x: FloatArray, y: BoolArray) -> BaseRate:
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        if self.rate is None:
            raise ValueError("BaseRate is not fitted")
        p = np.full(x.shape[0], self.rate)
        out: FloatArray = np.column_stack([1.0 - p, p])
        return out

    def get_params(self, deep: bool = True) -> Mapping[str, Any]:
        return {}


@dataclass(frozen=True)
class ModelSpec:
    """A model to evaluate.

    Attributes
    ----------
    name, title : str
        File name (``docs/models/<name>.md``) and display title.
    features : tuple of str
        xG features, checked by the leakage guard on every fit.
    make : callable
        ``make()`` returns a new unfitted estimator, called once per fit.
    coefficients : callable, optional
        ``coefficients(model, design_names)`` gives ``(term, standardised, natural units)``
        rows of the fit on all of dev for the card; ``None`` for a model without any.
    """

    name: str
    title: str
    features: tuple[str, ...]
    make: Callable[[], Estimator]
    coefficients: Callable[[Estimator, tuple[str, ...]], Coefficients] | None = None


@dataclass(frozen=True)
class Benchmark:
    """Probabilities read from a column (a provider's xG): scored, never fitted."""

    name: str
    title: str
    column: str


BENCHMARK: Final[Benchmark] = Benchmark("statsbomb-xg", "StatsBomb xG", "provider_xg")


@dataclass(frozen=True, eq=False)
class Evaluation:
    """One model scored by grouped CV, in-sample on dev and on the hold-out.

    Attributes
    ----------
    spec : ModelSpec or Benchmark
        What was scored.
    folds : tuple of Scores
        Per test fold.
    cv : Scores
        ``mean_scores(folds)``, the grouped-CV score.
    oof : numpy.ndarray
        Out-of-fold probabilities, in ``dev`` order.
    train, holdout : Scores
        The fit on all of dev, scored on dev (in-sample) and on the hold-out.
    coefficients : tuple of (str, float, float)
        From ``spec.coefficients`` for the fit on all of dev; ``()`` without it.
    holdout_p : numpy.ndarray
        Hold-out probabilities of that fit, in ``held`` order.
    fold_test : tuple of numpy.ndarray
        Each fold's test row indices into ``dev``.
    """

    spec: ModelSpec | Benchmark
    folds: tuple[Scores, ...]
    cv: Scores
    oof: FloatArray
    train: Scores
    holdout: Scores
    coefficients: Coefficients
    holdout_p: FloatArray
    fold_test: tuple[IntArray, ...]


def labels(shots: pl.DataFrame) -> BoolArray:
    """``is_goal`` as a numpy bool array.

    Examples
    --------
    >>> labels(pl.DataFrame({"is_goal": [True, False]})).tolist()
    [True, False]
    """
    y: BoolArray = shots["is_goal"].to_numpy()
    return y


def _require_model(spec: object) -> ModelSpec:
    if not isinstance(spec, ModelSpec):
        raise TypeError(
            f"expected a ModelSpec, not {spec!r}; score a Benchmark with evaluate_benchmark"
        )
    return spec


def _fit(spec: ModelSpec, train: pl.DataFrame, freeze_frames: pl.DataFrame | None) -> Estimator:
    if train.height == 0:
        raise ValueError(f"{spec.name}: empty training set")
    x, names = design_matrix(train, spec.features, freeze_frames=freeze_frames)
    model = spec.make()
    if isinstance(model, DesignFit) and (nulls := train["match_id"].null_count()):
        raise ValueError(f"{spec.name}: match_id has {nulls} null value(s) in the training rows")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", LinAlgWarning)
        try:
            if isinstance(model, DesignFit):
                groups = train["match_id"].to_numpy()
                model.fit_design(x, labels(train), names=names, groups=groups)
            else:
                model.fit(x, labels(train))
        except ConvergenceWarning as w:
            final = model[-1] if isinstance(model, Pipeline) else model
            limit = final.get_params().get("max_iter")
            raise ValueError(
                f"{spec.name}: {type(final).__name__} did not converge in {limit} iterations"
            ) from w
        except LinAlgWarning as w:
            constant = np.ptp(x, axis=0) == 0
            cols = ", ".join(n for n, c in zip(names, constant, strict=True) if c)
            raise ValueError(
                f"{spec.name}: rank-deficient design (singular Hessian); constant on the "
                f"training rows: {cols or 'none (collinear columns)'}"
            ) from w
    return model


def _predict(
    model: Estimator, spec: ModelSpec, shots: pl.DataFrame, freeze_frames: pl.DataFrame | None
) -> FloatArray:
    x, _ = design_matrix(shots, spec.features, freeze_frames=freeze_frames)
    p: FloatArray = np.ascontiguousarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    return p


def fit_predict(
    spec: ModelSpec,
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    freeze_frames: pl.DataFrame | None = None,
) -> FloatArray:
    """Fit a new ``spec.make()`` on ``train`` and return P(goal) for each ``test`` shot.

    Parameters
    ----------
    spec : ModelSpec
        The model; a :class:`Benchmark` is refused.
    train, test : polars.DataFrame
        Shots with ``is_goal`` and the model's features.
    freeze_frames : polars.DataFrame, optional
        C3 freeze frames, for derived features that read them (``design_matrix``).

    Returns
    -------
    numpy.ndarray
        One probability per test row.

    Raises
    ------
    TypeError
        ``spec`` is not a ``ModelSpec``.
    ValueError
        An empty training set, a feature problem (``design_matrix``), a
        ``ConvergenceWarning`` from the fit, or a singular Hessian under ``newton-cholesky``
        (the message names the design columns that are constant on the training rows); for a
        :class:`DesignFit`, a null ``match_id`` in the training rows.

    Examples
    --------
    >>> base = ModelSpec("base", "Base rate", (), BaseRate)
    >>> train = pl.DataFrame({"is_goal": [True, False, False, False]})
    >>> fit_predict(base, train, train.head(2)).tolist()
    [0.25, 0.25]
    """
    spec = _require_model(spec)
    return _predict(_fit(spec, train, freeze_frames), spec, test, freeze_frames)


def cv_folds(dev: pl.DataFrame) -> Folds:
    """``N_SPLITS`` grouped folds by ``match_id``, as (train, test) row-index pairs.

    Raises
    ------
    ValueError
        From scikit-learn, with fewer matches than folds.

    Examples
    --------
    >>> dev = pl.DataFrame({"match_id": [f"sb:{i // 2}" for i in range(10)]})
    >>> sorted(test.tolist() for _, test in cv_folds(dev))
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """
    splitter = GroupKFold(n_splits=N_SPLITS)
    groups = dev["match_id"].to_numpy()
    return tuple(
        (np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
        for train, test in splitter.split(np.zeros((dev.height, 1)), groups=groups)
    )


def _evaluation(
    spec: ModelSpec | Benchmark,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    fold_p: list[FloatArray],
    folds: Folds,
    p_train: FloatArray,
    p_held: FloatArray,
    coefficients: Coefficients,
) -> Evaluation:
    y = labels(dev)
    oof = np.full(dev.height, np.nan)
    per_fold = []
    for (_, test), p in zip(folds, fold_p, strict=True):
        oof[test] = p
        per_fold.append(score(y[test], p))
    return Evaluation(
        spec=spec,
        folds=tuple(per_fold),
        cv=mean_scores(per_fold),
        oof=oof,
        train=score(y, p_train),
        holdout=score(labels(held), p_held),
        coefficients=coefficients,
        holdout_p=p_held,
        fold_test=tuple(test for _, test in folds),
    )


def evaluate(
    spec: ModelSpec,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    folds: Folds,
    *,
    freeze_frames: pl.DataFrame | None = None,
) -> Evaluation:
    """Grouped-CV, in-sample and hold-out scores of one model.

    ``freeze_frames`` is passed to every ``design_matrix`` call, as in :func:`fit_predict`.

    Raises
    ------
    TypeError, ValueError
        As :func:`fit_predict`.

    Examples
    --------
    >>> ev = evaluate(BASELINES[1], dev, held, cv_folds(dev))  # doctest: +SKIP
    >>> round(ev.cv.log_loss, 4)  # doctest: +SKIP
    0.2968
    """
    spec, ff = _require_model(spec), freeze_frames
    fold_p = [fit_predict(spec, dev[train], dev[test], freeze_frames=ff) for train, test in folds]
    model = _fit(spec, dev, ff)
    coefficients: Coefficients = ()
    if spec.coefficients is not None:
        _, names = design_matrix(dev.head(0), spec.features, freeze_frames=ff)
        coefficients = spec.coefficients(model, names)
    return _evaluation(
        spec, dev, held, fold_p, folds,
        _predict(model, spec, dev, ff), _predict(model, spec, held, ff), coefficients,
    )  # fmt: skip


def _column(shots: pl.DataFrame, column: str) -> FloatArray:
    nulls = shots[column].null_count()
    if nulls:
        raise ValueError(f"{column} has {nulls} null value(s)")
    p: FloatArray = shots[column].to_numpy()
    return p


def evaluate_benchmark(
    dev: pl.DataFrame, held: pl.DataFrame, folds: Folds, benchmark: Benchmark = BENCHMARK
) -> Evaluation:
    """A benchmark column scored on the same folds and hold-out (no fitting).

    Raises
    ------
    ValueError
        A null in the column, on dev or the hold-out.

    Examples
    --------
    >>> round(evaluate_benchmark(dev, held, folds).cv.log_loss, 4)  # doctest: +SKIP
    0.2674
    """
    p = _column(dev, benchmark.column)
    fold_p = [p[test] for _, test in folds]
    return _evaluation(benchmark, dev, held, fold_p, folds, p, _column(held, benchmark.column), ())


def best(evaluations: Sequence[Evaluation]) -> Evaluation:
    """The model with the lowest grouped-CV log loss; a benchmark never qualifies.

    Raises
    ------
    ValueError
        If no evaluation is of a ``ModelSpec``.

    Examples
    --------
    >>> best(evaluations).spec.name  # doctest: +SKIP
    'xg-lr-distance-angle-body-part'
    """
    candidates = [ev for ev in evaluations if isinstance(ev.spec, ModelSpec)]
    if not candidates:
        raise ValueError("no model evaluation to choose from")
    return min(candidates, key=lambda ev: ev.cv.log_loss)


@dataclass(frozen=True)
class Comparison:
    """The log-loss gain of one evaluation over a reference, on CV and on the hold-out.

    Attributes
    ----------
    label : str
        What is compared, for the card's gain table.
    cv, holdout : GainCI
        Pooled out of fold over all dev shots, and over the hold-out; positive means the new
        model is better.
    """

    label: str
    cv: GainCI
    holdout: GainCI


def _check_rows(ev: Evaluation, dev: pl.DataFrame, held: pl.DataFrame) -> None:
    """``dev`` and ``held`` are the frames ``ev`` was scored on, in the same row order."""
    name = ev.spec.name
    for frame, p, rows in (("dev", ev.oof, dev.height), ("held", ev.holdout_p, held.height)):
        if p.shape != (rows,):
            raise ValueError(f"{name}: {p.size} prediction(s) for the {rows} row(s) of {frame}")
    y, yh = labels(dev), labels(held)
    parts = [
        (f"dev (fold {i})", y[test], ev.oof[test], s)
        for i, (test, s) in enumerate(zip(ev.fold_test, ev.folds, strict=True))
    ]
    # a goal count is the same in any row order; the log loss of the stored predictions is not
    for frame, y_part, p, s in (*parts, ("held", yh, ev.holdout_p, ev.holdout)):
        loss = float(log_loss(y_part, p, labels=[False, True]))
        if int(y_part.sum()) != s.goals or abs(loss - s.log_loss) > 1e-12:
            raise ValueError(
                f"{name}: {frame} is not the frame it was evaluated on (other rows or row order)"
            )


def compare(
    label: str, ref: Evaluation, new: Evaluation, dev: pl.DataFrame, held: pl.DataFrame
) -> Comparison:
    """The log-loss gain of ``new`` over ``ref``, with match-bootstrap 95 % intervals.

    Either side may be a benchmark. The CV gain pairs the two out-of-fold predictions of every
    dev shot, grouped by ``match_id``; the hold-out gain pairs their hold-out predictions.

    Parameters
    ----------
    label : str
        The row label.
    ref, new : Evaluation
        Scored on the same folds of ``dev`` and on ``held``.
    dev, held : polars.DataFrame
        The frames both were evaluated on, in the same row order.

    Returns
    -------
    Comparison

    Raises
    ------
    ValueError
        Different fold counts or fold assignments (equal sizes included); predictions whose
        length differs from ``dev`` or ``held``; a frame that is not the one evaluated, or is
        in another row order (per fold of ``dev``, and for ``held``); a null ``match_id``; or
        from :func:`~football_shot_models.metrics.bootstrap_gain`.

    Examples
    --------
    >>> row = compare("over S0", s0, s1, dev, held)  # doctest: +SKIP
    >>> f"{row.cv.gain:+.4f} [{row.cv.low:+.4f}, {row.cv.high:+.4f}]"  # doctest: +SKIP
    '+0.0012 [+0.0008, +0.0016]'
    """
    if len(ref.fold_test) != len(new.fold_test):
        raise ValueError(
            f"{label}: {ref.spec.name} has {len(ref.fold_test)} fold(s), {new.spec.name} "
            f"{len(new.fold_test)}"
        )
    for i, (a, b) in enumerate(zip(ref.fold_test, new.fold_test, strict=True)):
        if not np.array_equal(a, b):
            raise ValueError(
                f"{label}: {ref.spec.name} and {new.spec.name} were not scored on the same "
                f"folds (fold {i} differs)"
            )
    for ev in (ref, new):
        _check_rows(ev, dev, held)
    for frame_name, frame in (("dev", dev), ("held", held)):
        if nulls := frame["match_id"].null_count():
            raise ValueError(f"{label}: match_id has {nulls} null value(s) in {frame_name}")
    return Comparison(
        label,
        bootstrap_gain(labels(dev), ref.oof, new.oof, dev["match_id"].to_numpy()),
        bootstrap_gain(labels(held), ref.holdout_p, new.holdout_p, held["match_id"].to_numpy()),
    )
