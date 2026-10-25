"""M1.4 monotone GBM (spec MS3.models §1.2-§1.5, §3.1, §9): constants, the sign guard, nested
tuning by match, cross-fitted Platt scaling, its refusals and the monotone property.

Fixtures are numpy arrays from ``models_cases`` (``signal_xy`` and the spec's guard fixtures).
"""

from __future__ import annotations

import math
from typing import Any

import models_cases as mc
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from football_shot_models import ablation, evaluation, features

DEFAULT_MONOTONE = {
    "distance_m": -1,
    "angle_rad": 1,
    "opponents_in_cone": -1,
    "nearest_opponent_m": 1,
}
REASONS = {
    "distance_m": "further from goal at the same opening angle never helps",
    "angle_rad": "a wider opening between the posts never hurts",
    "opponents_in_cone": "another outfield body between ball and goal never helps",
    "nearest_opponent_m": "a closer outfield marker never helps (outfield opponents only, capped "
    "at 10 m, also the value with no marker in the frame)",
}


@pytest.fixture(scope="module")
def signal_fit() -> tuple[Any, Any, Any, Any]:
    """One MonotoneGBM fitted on the 600-row signal fixture with ``min_samples_leaf=20``."""
    from football_shot_models import models

    x, y, groups = mc.signal_xy()
    m = models.MonotoneGBM(mc.SIGNAL_MAP, **mc.SIGNAL)
    m.fit_design(x, y, names=mc.SIGNAL_NAMES, groups=groups)
    return m, x, y, groups


def test_gbm_constants_and_spec() -> None:
    """§1.3-§1.4, §3.1: the fixed settings, the sign map, its reasons, the spec, ``make_hgb`` and
    no class rebalancing anywhere (§9)."""
    from football_shot_models import models

    assert (models.LEARNING_RATE, models.L2_REGULARIZATION) == (0.05, 1.0)
    assert (models.MIN_SAMPLES_LEAF, models.LEAVES) == (100, (8, 16, 32))
    assert (models.MAX_ITER, models.STEP, models.INNER_SPLITS) == (600, 50, 4)
    assert dict(models.MONOTONE) == DEFAULT_MONOTONE
    with pytest.raises(TypeError):
        models.MONOTONE["gk_to_goal_m"] = 1  # type: ignore[index]  # read-only mapping
    assert dict(models.MONOTONE_REASONS) == REASONS
    spec = models.GBM
    assert (spec.name, spec.title) == ("xg-gbm", "Monotone GBM + Platt")
    assert spec.coefficients is models.gbm_settings
    fresh = spec.make()
    assert isinstance(fresh, models.MonotoneGBM)
    assert isinstance(fresh, evaluation.DesignFit)
    assert dict(fresh.get_params()) == {
        "monotone": DEFAULT_MONOTONE,
        "leaf_grid": (8, 16, 32),
        "max_iter": 600,
        "step": 50,
        "inner_splits": 4,
        "min_samples_leaf": 100,
    }
    assert not any("weight" in k for k in fresh.get_params())
    hgb = models.make_hgb(16, 200, [-1, 1, 0])
    p = hgb.get_params()
    assert (p["learning_rate"], p["max_iter"], p["max_leaf_nodes"]) == (0.05, 200, 16)
    assert (p["min_samples_leaf"], p["l2_regularization"]) == (100, 1.0)
    assert (p["early_stopping"], p["random_state"], p["class_weight"]) == (False, 0, None)
    assert list(p["monotonic_cst"]) == [-1, 1, 0]
    assert models.make_hgb(8, 50, [0], min_leaf=20).get_params()["min_samples_leaf"] == 20
    # the LR steps are not DesignFit: the harness keeps calling their fit(x, y)
    assert not isinstance(ablation.make_lr(), evaluation.DesignFit)


def test_gbm_features_pass_guard(xg_data: Any) -> None:
    """§1.2: S3's inputs without the spline terms, through the leakage guard; 29 design columns,
    every constrained name among them (so the default map never fails closed on S3 data)."""
    from football_shot_models import models

    expected = ("distance_m", "angle_rad", "body_part", *ablation.CONTEXT, *ablation.FRAMES)
    assert expected == models.GBM_FEATURES
    assert models.GBM.features == expected
    assert features.check_xg_features(models.GBM_FEATURES) == expected
    assert not any("_rcs" in f for f in models.GBM_FEATURES)
    _, names = features.design_matrix(
        xg_data.shots.head(0), models.GBM_FEATURES, freeze_frames=xg_data.freeze_frames
    )
    assert len(names) == 29
    assert set(DEFAULT_MONOTONE) <= set(names)


@settings(deadline=None, max_examples=40)
@given(
    base=st.tuples(st.floats(0.0, 40.0), st.floats(0.0, 1.6), st.floats(-4.0, 4.0)),
    column=st.sampled_from([0, 1]),
    lo=st.floats(0.0, 60.0),
    step=st.floats(0.0, 30.0),
)
def test_gbm_monotone_property(
    signal_fit: tuple[Any, Any, Any, Any],
    base: tuple[float, float, float],
    column: int,
    lo: float,
    step: float,
) -> None:
    """§1.3, §9: on ``predict_proba`` (after Platt), with every other column held fixed, P(goal)
    never rises with ``d`` (sign -1) and never falls with ``a`` (+1), also outside the data."""
    m, _, _, _ = signal_fit
    rows = np.array([base, base], dtype=np.float64)
    rows[0, column], rows[1, column] = lo, lo + step
    p = m.predict_proba(rows)[:, 1]
    if column == 0:
        assert p[1] <= p[0]
    else:
        assert p[1] >= p[0]


@pytest.mark.parametrize(
    ("monotone", "named"),
    [
        ({"d": -1, "dist_absent": -1}, "dist_absent"),
        ({"d": 0}, "'d': 0"),
        ({"d": 2}, "'d': 2"),
        ({"d": True}, "'d': True"),
        ({"d": 1.0}, "'d': 1.0"),
        ({"d": -1, "a": -1.0}, "'a': -1.0"),
    ],
    ids=["absent", "zero", "two", "bool", "float", "negative float"],
)
def test_gbm_constraint_names(monotone: dict[str, Any], named: str) -> None:
    """§3.1 step 1: an absent constrained column and every sign that is a bool, not an int, or
    not -1 / +1 is a ValueError naming it, before any fit (two matches only: the inner
    ``GroupKFold`` would raise a different error if the grid ran first)."""
    from football_shot_models import models

    x, y, _ = mc.signal_xy()
    groups = np.array(["sb:m01", "sb:m02"] * 300)
    with pytest.raises(ValueError) as info:
        models.MonotoneGBM(monotone, **mc.SIGNAL).fit_design(
            x, y, names=mc.SIGNAL_NAMES, groups=groups
        )
    assert named in str(info.value)
    assert "-1 / +1" in str(info.value) or "absent" in str(info.value)


def test_gbm_platt_slope_positive() -> None:
    """§1.5, §3.1 step 4: inner out-of-fold scores that rank goals backwards give a Platt slope
    <= 0, which would reverse the constraints: refused (spec_guards.out: b = -4.019)."""
    from football_shot_models import models

    x, y, groups = mc.no_signal_xy()
    m = models.MonotoneGBM({"a": -1}, leaf_grid=(4,), max_iter=20, step=10)
    slope = r"^Platt slope -[0-9.]+ <= 0: the inner out-of-fold scores do not rank goals"
    with pytest.raises(ValueError, match=slope):
        m.fit_design(x, y, names=("a", "b"), groups=groups)
    assert m.model_ is None
    with pytest.raises(ValueError, match="not fitted"):
        m.predict_proba(x)


def test_gbm_constant_inner_scores() -> None:
    """§3.1 step 3: constant inner scores (no tree could split) are refused before Platt, with
    the setting in the message, so the harness never reports a rank-deficient design."""
    from football_shot_models import models

    x, y, groups = mc.constant_xy()
    m = models.MonotoneGBM({"a": -1}, leaf_grid=(4,), max_iter=20, step=10)
    message = (
        "the inner out-of-fold scores are constant (no inner tree could split with "
        "min_samples_leaf=100): Platt scaling is not estimable"
    )
    with pytest.raises(ValueError) as info:
        m.fit_design(x, y, names=("a", "b"), groups=groups)
    assert str(info.value) == message


def test_gbm_fit_requires_design() -> None:
    """§3.1: plain ``fit(x, y)`` is a TypeError (names and groups come from the harness);
    ``predict_proba`` before a fit is a ValueError."""
    from football_shot_models import models

    x, y, _ = mc.signal_xy()
    m = models.MonotoneGBM(mc.SIGNAL_MAP, **mc.SIGNAL)
    with pytest.raises(TypeError, match="fit_design"):
        m.fit(x, y)
    with pytest.raises(ValueError, match="not fitted"):
        m.predict_proba(x)


def test_gbm_deterministic(signal_fit: tuple[Any, Any, Any, Any]) -> None:
    """§1.4: no early stopping and ``random_state=0``, so a refit gives the same bits, with one
    OpenMP thread as with the default."""
    from threadpoolctl import threadpool_limits

    from football_shot_models import models

    first, x, y, groups = signal_fit
    with threadpool_limits(limits=1, user_api="openmp"):
        again = models.MonotoneGBM(mc.SIGNAL_MAP, **mc.SIGNAL).fit_design(
            x, y, names=mc.SIGNAL_NAMES, groups=groups
        )
    assert (again.leaves_, again.iters_, again.platt_) == (
        first.leaves_,
        first.iters_,
        first.platt_,
    )
    assert np.array_equal(again.inner_raw_, first.inner_raw_)
    assert np.array_equal(again.predict_proba(x), first.predict_proba(x))


def test_gbm_inner_folds_grouped() -> None:
    """§1.4, §9: the inner test folds partition the rows and no match is in two of them (near
    miss: two matches whose rows are spread over the whole array, which a positional split
    would cut)."""
    from football_shot_models import models

    x, y, _ = mc.signal_xy()
    # every 15th row belongs to sb:mA or sb:mB in turn: two matches spread over the whole array
    groups = np.array(
        [
            ("sb:mA" if (i // 15) % 2 else "sb:mB") if i % 15 == 0 else f"sb:m{i // 10:02d}"
            for i in range(600)
        ]
    )
    m = models.MonotoneGBM(mc.SIGNAL_MAP, **mc.SIGNAL).fit_design(
        x, y, names=mc.SIGNAL_NAMES, groups=groups
    )
    assert len(m.inner_test_) == mc.SIGNAL["inner_splits"]
    assert np.array_equal(np.sort(np.concatenate(m.inner_test_)), np.arange(600))
    fold_of = {}
    for k, test in enumerate(m.inner_test_):
        for g in set(groups[test].tolist()):
            assert fold_of.setdefault(g, k) == k, g
    assert len(fold_of) == 62


def test_gbm_platt_cross_fitted(signal_fit: tuple[Any, Any, Any, Any]) -> None:
    """§1.5, §9: ``inner_raw_`` on each inner test fold is the score of a ``make_hgb`` fit on the
    other inner folds; ``platt_`` is the unpenalised LR of y on ``inner_raw_`` (never in-sample
    scores); the final model is refitted on all rows with the chosen configuration and
    ``predict_proba = expit(a + b * raw)``."""
    from football_shot_models import models

    m, x, y, _ = signal_fit
    # the monotone property is not vacuous: the fitted model moves with d
    near, far = x[:1].copy(), x[:1].copy()
    near[0, 0], far[0, 0] = 5.0, 30.0
    assert m.predict_proba(near)[0, 1] > m.predict_proba(far)[0, 1]
    assert np.ptp(m.predict_proba(x)[:, 1]) > 0.1
    cst = [mc.SIGNAL_MAP.get(n, 0) for n in mc.SIGNAL_NAMES]
    assert m.leaves_ in mc.SIGNAL["leaf_grid"]
    assert m.iters_ in range(mc.SIGNAL["step"], mc.SIGNAL["max_iter"] + 1, mc.SIGNAL["step"])
    assert m.inner_raw_.shape == (600,)
    for test in m.inner_test_:
        train = np.setdiff1d(np.arange(600), test)
        fold = models.make_hgb(m.leaves_, m.iters_, cst, mc.SIGNAL["min_samples_leaf"])
        raw = fold.fit(x[train], y[train]).decision_function(x[test])
        np.testing.assert_allclose(m.inner_raw_[test], raw, rtol=0, atol=1e-12)
    lr = LogisticRegression(C=math.inf, solver="newton-cholesky", tol=1e-10, max_iter=1000)
    lr.fit(m.inner_raw_[:, None], y)
    a, b = m.platt_
    assert b > 0
    assert a == pytest.approx(float(lr.intercept_[0]), abs=1e-6)
    assert b == pytest.approx(float(lr.coef_[0, 0]), abs=1e-6)
    # an in-sample Platt fit would differ: the cross-fitted scores are not the final model's
    assert not np.allclose(m.inner_raw_, m.model_.decision_function(x))
    p = m.model_.get_params()
    assert (p["max_leaf_nodes"], p["max_iter"]) == (m.leaves_, m.iters_)
    assert list(p["monotonic_cst"]) == cst
    want = expit(a + b * m.model_.decision_function(x))
    got = m.predict_proba(x)
    assert got.shape == (600, 2)
    np.testing.assert_allclose(got[:, 1], want, rtol=0, atol=1e-15)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=0, atol=1e-15)


def test_gbm_tie_break(monkeypatch: pytest.MonkeyPatch) -> None:
    """§1.4: equal inner losses go to the smaller model: fewer leaves (whatever the grid order),
    then fewer iterations."""
    from football_shot_models import models

    monkeypatch.setattr(models, "log_loss", lambda *a, **k: 0.5)
    x, y, groups = mc.signal_xy()
    m = models.MonotoneGBM(mc.SIGNAL_MAP, **{**mc.SIGNAL, "leaf_grid": (8, 4)})
    m.fit_design(x, y, names=mc.SIGNAL_NAMES, groups=groups)
    assert (m.leaves_, m.iters_) == (4, mc.SIGNAL["step"])


def test_gbm_settings(signal_fit: tuple[Any, Any, Any, Any]) -> None:
    """§3.1: the card's settings rows from the fitted instance (its own sign map, in order);
    a model that is not a MonotoneGBM is a TypeError."""
    from football_shot_models import models

    m, _, _, _ = signal_fit
    a, b = m.platt_
    assert models.gbm_settings(m, mc.SIGNAL_NAMES) == (
        ("max_leaf_nodes", m.leaves_, m.leaves_),
        ("iterations", m.iters_, m.iters_),
        ("platt_intercept", a, a),
        ("platt_slope", b, b),
        ("monotone:d", -1, -1),
        ("monotone:a", 1, 1),
    )
    with pytest.raises(TypeError):
        models.gbm_settings(evaluation.BaseRate(), ())


def test_gbm_refit_and_platt_use_chosen_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    """run3-ms3 (QA 12), §1.4-§1.5: with a chosen iteration count below ``max_iter`` (a flat inner
    loss picks the first stop), the final refit uses that count, ``inner_raw_`` holds the staged
    scores at that count (not the last stage), and Platt is fitted on them."""
    from football_shot_models import models

    monkeypatch.setattr(models, "log_loss", lambda *a, **k: 0.5)
    x, y, groups = mc.signal_xy()
    m = models.MonotoneGBM(mc.SIGNAL_MAP, **mc.SIGNAL)
    m.fit_design(x, y, names=mc.SIGNAL_NAMES, groups=groups)
    assert m.iters_ == mc.SIGNAL["step"] < mc.SIGNAL["max_iter"]  # the near miss is live
    assert m.model_ is not None
    assert (m.model_.max_iter, m.model_.max_leaf_nodes) == (m.iters_, m.leaves_)
    cst = [mc.SIGNAL_MAP.get(n, 0) for n in mc.SIGNAL_NAMES]
    for test in m.inner_test_:
        train = np.setdiff1d(np.arange(y.size), test)
        fold = models.make_hgb(m.leaves_, mc.SIGNAL["max_iter"], cst, mc.SIGNAL["min_samples_leaf"])
        staged = list(fold.fit(x[train], y[train]).staged_decision_function(x[test]))
        np.testing.assert_allclose(m.inner_raw_[test], staged[m.iters_ - 1], rtol=0, atol=1e-12)
        assert not np.allclose(m.inner_raw_[test], staged[-1])
    lr = LogisticRegression(C=math.inf, solver="newton-cholesky", tol=1e-10, max_iter=1000)
    lr.fit(m.inner_raw_[:, None], y)
    assert m.platt_ == pytest.approx((float(lr.intercept_[0]), float(lr.coef_[0, 0])), abs=1e-6)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"leaf_grid": ()}, r"non-empty leaf_grid \(got \(\)\)"),
        ({"step": 80}, r"step=80, max_iter=60"),
        ({"step": 0}, r"step=0, max_iter=60"),
    ],
)
def test_gbm_bad_grid(settings: dict[str, Any], match: str) -> None:
    """run3-ms3 (QA 17): an empty leaf grid or a step outside [1, max_iter] is a named
    ValueError, not a bare ``min()`` error."""
    from football_shot_models import models

    x, y, groups = mc.signal_xy()
    m = models.MonotoneGBM(mc.SIGNAL_MAP, **{**mc.SIGNAL, **settings})
    with pytest.raises(ValueError, match=match):
        m.fit_design(x, y, names=mc.SIGNAL_NAMES, groups=groups)
