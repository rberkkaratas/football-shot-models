"""M1.4: a monotone gradient-boosted xG model, tuned and Platt-calibrated inside each fit.

**The GBM** is scikit-learn's ``HistGradientBoostingClassifier``. Columns named in a sign map
(:data:`MONOTONE`) carry a monotone constraint; a constraint holds along one column with every
other column held fixed. The model spec, its features and the football reasons for each sign
live with the run in :mod:`football_shot_models.models`.

**Tuning and calibration stay inside the fit.** :class:`MonotoneGBM` is a
:class:`~football_shot_models.evaluation.DesignFit`: the harness hands it the training rows
of one fit with their design names and matches. Inside, a grouped inner CV by match picks the
leaf count and the number of iterations, and Platt scaling is fitted on the chosen model's
inner out-of-fold scores (cross-fitted by match), then applied to a refit on all the training
rows. The outer test fold never reaches it, so the CV score is honest.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from football_shot_models.evaluation import Coefficients, Estimator

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]

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
            An empty ``leaf_grid``; ``step`` not in [1, ``max_iter``]; a constrained column
            absent from ``names``, or a sign that is a bool, not an ``int``, or not -1 / +1;
            fewer matches than ``inner_splits`` (from scikit-learn);
            constant inner out-of-fold scores (Platt not estimable); a Platt slope <= 0.
        """
        if not self.leaf_grid or not 1 <= self.step <= self.max_iter:
            raise ValueError(
                f"MonotoneGBM needs a non-empty leaf_grid (got {self.leaf_grid}) and 1 <= step "
                f"<= max_iter (got step={self.step}, max_iter={self.max_iter})"
            )
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
