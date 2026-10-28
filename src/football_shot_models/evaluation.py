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

**Cached predictions.** :func:`evaluate` can read a spec's predictions from a
:class:`~football_shot_models.cache.PredictionCache` instead of fitting, when the data, the
code, the spec and the rows all match; everything after the fits (scores, comparisons, cards)
is computed the same way either way. The leakage guard runs on a hit too.

**Other targets and folds.** A spec names its own registry (``derived``) and guard, so an xGOT
model goes through the same harness with the xGOT guard
(:func:`~football_shot_models.features.check_xgot_features`); a spec that reads the cross-fitted
xG (``xg_logit``) must name its source (``xg_input``), which the cache identity hashes with the
column's values. Any fold assignment that partitions dev works
(:func:`check_partition`); :func:`restrict_folds` keeps it on a subset of the rows (the on-target
shots of an xGOT model), so the subset's folds are the parent's.

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

from football_shot_models import features
from football_shot_models.cache import PredictionCache, Predictions, column_digest
from football_shot_models.cache import identity as cache_identity
from football_shot_models.features import DerivedFeature, Guard, design_matrix
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
    guard : callable, optional
        The leakage guard of ``features`` (:data:`~football_shot_models.features.Guard`), run
        on every fit, prediction and cache hit; ``None`` is
        :func:`~football_shot_models.features.check_xg_features`, looked up when called. A
        guard decides only whether a fit may run, never what it computes.
    derived : mapping of str to DerivedFeature, optional
        The derived-feature registry of ``features``, passed to the guard and to
        ``design_matrix``; ``None`` is :data:`~football_shot_models.features.DERIVED`, looked
        up when called (an xGOT spec passes ``features.XGOT_DERIVED``).
    xg_input : str, optional
        The name of the xG spec whose cross-fitted predictions fill ``xg_logit``; required
        when ``xg_logit`` is a feature, and part of the cache identity.
    """

    name: str
    title: str
    features: tuple[str, ...]
    make: Callable[[], Estimator]
    coefficients: Callable[[Estimator, tuple[str, ...]], Coefficients] | None = None
    guard: Guard | None = None
    derived: Mapping[str, DerivedFeature] | None = None
    xg_input: str | None = None


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
    if features.XG_INPUT in spec.features and spec.xg_input is None:
        raise ValueError(
            f"{spec.name}: reads {features.XG_INPUT!r} but names no xg_input (the xG spec whose "
            "cross-fitted predictions fill it)"
        )
    return spec


def _registry(spec: ModelSpec) -> Mapping[str, DerivedFeature]:
    return features.DERIVED if spec.derived is None else spec.derived


def _design(
    spec: ModelSpec, shots: pl.DataFrame, freeze_frames: pl.DataFrame | None
) -> tuple[FloatArray, tuple[str, ...]]:
    return design_matrix(
        shots, spec.features, derived=_registry(spec), freeze_frames=freeze_frames,
        guard=spec.guard,
    )  # fmt: skip


def _fit(spec: ModelSpec, train: pl.DataFrame, freeze_frames: pl.DataFrame | None) -> Estimator:
    if train.height == 0:
        raise ValueError(f"{spec.name}: empty training set")
    x, names = _design(spec, train, freeze_frames)
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
    x, _ = _design(spec, shots, freeze_frames)
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
        ``xg_logit`` among the features without ``spec.xg_input``; an empty training set, a
        feature problem (``design_matrix``), a
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


def check_partition(folds: Folds, n: int) -> None:
    """Refuse folds whose test sets are not a partition of the ``n`` dev rows.

    Forward-chaining or overlapping folds would leave NaN out-of-fold predictions, or score a
    row twice, without any other error; a fold that trains on its own test rows would make
    its "out-of-fold" predictions in-sample.

    Raises
    ------
    ValueError
        An index outside ``0..n-1`` (train or test); a fold whose train and test share a row;
        a row in no test fold or in more than one (counts in the message).

    Examples
    --------
    >>> folds = ((np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3])))
    >>> check_partition(folds, 4)
    >>> check_partition(folds[:1], 4)
    Traceback (most recent call last):
    ...
    ValueError: folds are not a partition of the 4 dev row(s): 2 in no test fold, 0 in more than one
    """
    counts = np.zeros(n, dtype=np.int64)
    for i, (train, test) in enumerate(folds):
        for idx in (train, test):
            if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n):
                raise ValueError(f"fold {i}: an index is outside 0..{n - 1}")
        if shared := np.intersect1d(train, test).size:
            raise ValueError(f"fold {i}: {shared} row(s) are in both its train and test sets")
        np.add.at(counts, test, 1)
    none, many = int((counts == 0).sum()), int((counts > 1).sum())
    if none or many:
        raise ValueError(
            f"folds are not a partition of the {n} dev row(s): {none} in no test fold, {many} in "
            "more than one"
        )


def restrict_folds(folds: Folds, mask: BoolArray) -> Folds:
    """The same fold assignment on the rows where ``mask`` is True, re-indexed in row order.

    A row keeps its fold, so an xGOT row in test fold k reads the xG of the fit that never saw
    fold k. Any partition passes (season-grouped folds included).

    Raises
    ------
    ValueError
        ``mask`` not 1-D bool, or its length not the folds' row count; folds that are not a
        partition (:func:`check_partition`); a fold with no test row left.

    Examples
    --------
    >>> folds = ((np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3])))
    >>> [(tr.tolist(), te.tolist()) for tr, te in restrict_folds(folds, np.array(
    ...     [True, False, True, True]))]
    [([1, 2], [0]), ([0], [1, 2])]
    """
    m = np.asarray(mask)
    rows = sum(test.size for _, test in folds)
    if m.dtype != np.bool_ or m.ndim != 1 or m.size != rows:
        raise ValueError(
            f"mask must be a 1-D bool array of the folds' {rows} row(s), got {m.dtype} {m.shape}"
        )
    check_partition(folds, rows)
    new = np.cumsum(m) - 1
    out = []
    for i, (train, test) in enumerate(folds):
        kept = np.sort(new[test[m[test]]]).astype(np.intp)
        if kept.size == 0:
            raise ValueError(f"fold {i} has no test row left after the mask")
        out.append((np.sort(new[train[m[train]]]).astype(np.intp), kept))
    return tuple(out)


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


def _cached(
    spec: ModelSpec,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    folds: Folds,
    freeze_frames: pl.DataFrame | None,
    cache: PredictionCache,
) -> tuple[str, Predictions | None]:
    """The spec's cache identity and its entry, after the checks a fit would make first."""
    for side, frame in (("dev", dev), ("held", held)):
        if "shot_id" not in frame.columns:
            raise ValueError(f"{spec.name}: the prediction cache needs shot_id in {side}")
    reads_xg = features.XG_INPUT in spec.features
    inputs = f"{spec.xg_input} {column_digest(dev, held, features.XG_INPUT)}" if reads_xg else ""
    key = cache_identity(
        spec.name, spec.features, spec.make().get_params(), dev["shot_id"], held["shot_id"],
        inputs=inputs,
    )  # fmt: skip
    hit = cache.load(spec.name, key, folds)
    if hit is None:
        return key, None
    # a warm run refuses what a cold one refuses: the leakage guard, a frame feature without frames
    registry = _registry(spec)
    (spec.guard or features.check_xg_features)(spec.features, registry)
    for name in spec.features:
        sources = registry[name].sources if name in registry else ()
        if freeze_frames is None and any(table == "freeze_frames" for table, _ in sources):
            raise ValueError(f"derived feature {name!r} reads freeze_frames; pass freeze_frames=")
    sizes = (hit.oof.size, hit.train.size, hit.holdout.size)
    if sizes != (dev.height, dev.height, held.height):
        raise ValueError(f"cache: the {spec.name} entry holds {sizes} predictions; delete it")
    return key, hit


def evaluate(
    spec: ModelSpec,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    folds: Folds,
    *,
    freeze_frames: pl.DataFrame | None = None,
    cache: PredictionCache | None = None,
) -> Evaluation:
    """Grouped-CV, in-sample and hold-out scores of one model.

    ``freeze_frames`` is passed to every ``design_matrix`` call, as in :func:`fit_predict`.
    With a ``cache``, a matching entry replaces the fits (after the leakage guard), and a miss
    fits as usual, then writes the entry.

    Raises
    ------
    TypeError, ValueError
        As :func:`fit_predict`; with a cache, also a frame without ``shot_id``, a bad
        ``shot_id`` (:func:`~football_shot_models.cache.identity`), a missing or null
        ``xg_logit`` for a spec that reads it, or an unreadable entry.

    Examples
    --------
    >>> ev = evaluate(BASELINES[1], dev, held, cv_folds(dev))  # doctest: +SKIP
    >>> round(ev.cv.log_loss, 4)  # doctest: +SKIP
    0.2968
    """
    spec, ff = _require_model(spec), freeze_frames
    key, hit = (None, None) if cache is None else _cached(spec, dev, held, folds, ff, cache)
    if hit is not None:
        fold_p = [hit.oof[test] for _, test in folds]
        return _evaluation(spec, dev, held, fold_p, folds, hit.train, hit.holdout, hit.coefficients)
    fold_p = [fit_predict(spec, dev[train], dev[test], freeze_frames=ff) for train, test in folds]
    model = _fit(spec, dev, ff)
    coefficients: Coefficients = ()
    if spec.coefficients is not None:
        _, names = _design(spec, dev.head(0), ff)
        coefficients = spec.coefficients(model, names)
    p_train, p_held = _predict(model, spec, dev, ff), _predict(model, spec, held, ff)
    ev = _evaluation(spec, dev, held, fold_p, folds, p_train, p_held, coefficients)
    if cache is not None and key is not None:
        cache.save(spec.name, key, folds, Predictions(ev.oof, p_train, p_held, coefficients))
    return ev


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
    """The model with the lowest grouped-CV log loss; a benchmark or an ablation never qualifies.

    An ablation is a spec with an ``ABLATION_ONLY`` feature
    (:func:`~football_shot_models.features.uses_ablation_only`), whatever its guard.

    Raises
    ------
    ValueError
        If no evaluation is of a ``ModelSpec`` without an ablation feature.

    Examples
    --------
    >>> best(evaluations).spec.name  # doctest: +SKIP
    'xg-lr-distance-angle-body-part'
    """
    candidates = [
        ev
        for ev in evaluations
        if isinstance(ev.spec, ModelSpec) and not features.uses_ablation_only(ev.spec.features)
    ]
    if not candidates:
        raise ValueError(
            "no model evaluation to choose from (benchmarks and ablations never qualify)"
        )
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
