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
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from football_shot_models.features import design_matrix
from football_shot_models.metrics import Scores, mean_scores, score

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
    """

    spec: ModelSpec | Benchmark
    folds: tuple[Scores, ...]
    cv: Scores
    oof: FloatArray
    train: Scores
    holdout: Scores
    coefficients: Coefficients


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


def _fit(spec: ModelSpec, train: pl.DataFrame) -> Estimator:
    if train.height == 0:
        raise ValueError(f"{spec.name}: empty training set")
    x, _ = design_matrix(train, spec.features)
    model = spec.make()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x, labels(train))
        except ConvergenceWarning as w:
            final = model[-1] if isinstance(model, Pipeline) else model
            limit = final.get_params().get("max_iter")
            raise ValueError(
                f"{spec.name}: {type(final).__name__} did not converge in {limit} iterations"
            ) from w
    return model


def _predict(model: Estimator, spec: ModelSpec, shots: pl.DataFrame) -> FloatArray:
    x, _ = design_matrix(shots, spec.features)
    p: FloatArray = np.ascontiguousarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    return p


def fit_predict(spec: ModelSpec, train: pl.DataFrame, test: pl.DataFrame) -> FloatArray:
    """Fit a new ``spec.make()`` on ``train`` and return P(goal) for each ``test`` shot.

    Parameters
    ----------
    spec : ModelSpec
        The model; a :class:`Benchmark` is refused.
    train, test : polars.DataFrame
        Shots with ``is_goal`` and the model's features.

    Returns
    -------
    numpy.ndarray
        One probability per test row.

    Raises
    ------
    TypeError
        ``spec`` is not a ``ModelSpec``.
    ValueError
        An empty training set, a feature problem (``design_matrix``), or a
        ``ConvergenceWarning`` from the fit.

    Examples
    --------
    >>> base = ModelSpec("base", "Base rate", (), BaseRate)
    >>> train = pl.DataFrame({"is_goal": [True, False, False, False]})
    >>> fit_predict(base, train, train.head(2)).tolist()
    [0.25, 0.25]
    """
    spec = _require_model(spec)
    return _predict(_fit(spec, train), spec, test)


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
    )


def evaluate(spec: ModelSpec, dev: pl.DataFrame, held: pl.DataFrame, folds: Folds) -> Evaluation:
    """Grouped-CV, in-sample and hold-out scores of one model.

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
    spec = _require_model(spec)
    fold_p = [fit_predict(spec, dev[train], dev[test]) for train, test in folds]
    model = _fit(spec, dev)
    coefficients: Coefficients = ()
    if spec.coefficients is not None:
        _, names = design_matrix(dev.head(0), spec.features)
        coefficients = spec.coefficients(model, names)
    return _evaluation(
        spec, dev, held, fold_p, folds,
        _predict(model, spec, dev), _predict(model, spec, held), coefficients,
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
