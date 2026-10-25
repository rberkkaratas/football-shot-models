"""The harness additions of MS3.features (spec §4.2): hold-out probabilities and fold indices on
every Evaluation, ``compare`` on the same folds and rows only, freeze frames threaded to every
design matrix, and a rank-deficient newton-cholesky fit that fails closed."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
import pytest

from football_shot_models import baselines, evaluation, features, metrics

Split = tuple[pl.DataFrame, pl.DataFrame]


@pytest.fixture(scope="module")
def evs(dev_held: Split) -> dict[str, Any]:
    dev, held = dev_held
    folds = evaluation.cv_folds(dev)
    return {
        "folds": folds,
        "da": evaluation.evaluate(baselines.BASELINES[1], dev, held, folds),
        "bp": evaluation.evaluate(baselines.BASELINES[2], dev, held, folds),
        "bench": evaluation.evaluate_benchmark(dev, held, folds),
    }


def test_evaluation_keeps_holdout_p_and_fold_test(dev_held: Split, evs: dict[str, Any]) -> None:
    dev, held = dev_held
    folds = evs["folds"]
    for key in ("bp", "bench"):
        ev = evs[key]
        assert len(ev.fold_test) == len(folds) == 5
        for got, (_, test) in zip(ev.fold_test, folds, strict=True):
            np.testing.assert_array_equal(got, test)
        assert ev.holdout_p.shape == (held.height,)
        assert ev.holdout == metrics.score(evaluation.labels(held), ev.holdout_p)
    spec = baselines.BASELINES[2]
    np.testing.assert_array_equal(evs["bp"].holdout_p, evaluation.fit_predict(spec, dev, held))
    np.testing.assert_array_equal(evs["bench"].holdout_p, held["provider_xg"].to_numpy())


def test_compare_is_bootstrap_gain_on_both_splits(dev_held: Split, evs: dict[str, Any]) -> None:
    """CV: the out-of-fold probabilities of all dev shots, grouped by match; hold-out: the
    hold-out probabilities. A benchmark may be either side."""
    dev, held = dev_held
    c = evaluation.compare("body part added", evs["da"], evs["bp"], dev, held)
    assert c.label == "body part added"
    cv = metrics.bootstrap_gain(
        evaluation.labels(dev), evs["da"].oof, evs["bp"].oof, dev["match_id"].to_numpy()
    )
    ho = metrics.bootstrap_gain(
        evaluation.labels(held), evs["da"].holdout_p, evs["bp"].holdout_p,
        held["match_id"].to_numpy(),
    )  # fmt: skip
    for got, want in ((c.cv, cv), (c.holdout, ho)):
        assert (got.n_boot, got.groups) == (want.n_boot, want.groups)
        assert (got.gain, got.low, got.high) == pytest.approx(
            (want.gain, want.low, want.high), rel=1e-12, abs=1e-15
        )
    assert (c.cv.groups, c.holdout.groups) == (12, 2)  # matches, not shots
    vs = evaluation.compare("vs StatsBomb", evs["bench"], evs["bp"], dev, held)
    back = evaluation.compare("StatsBomb vs", evs["bp"], evs["bench"], dev, held)
    assert vs.cv.gain == pytest.approx(-back.cv.gain, abs=1e-15)
    assert vs.holdout.gain == pytest.approx(-back.holdout.gain, abs=1e-15)


def _other_assignment(folds: evaluation.Folds, n: int) -> evaluation.Folds:
    """Same fold sizes, one test row swapped between folds 0 and 1."""
    tests = [t.copy() for _, t in folds]
    tests[0][0], tests[1][0] = tests[1][0], tests[0][0]
    every = np.arange(n, dtype=np.intp)
    return tuple((np.setdiff1d(every, t).astype(np.intp), np.sort(t)) for t in tests)


@pytest.mark.parametrize(
    "case",
    [
        "other assignment", "fewer folds", "re-sorted dev", "re-sorted held", "short dev",
        "short held", "null dev match", "null held match",
    ],
)  # fmt: skip
def test_compare_refuses_other_folds_and_rows(
    case: str, dev_held: Split, evs: dict[str, Any]
) -> None:
    dev, held = dev_held
    ref, new = evs["da"], evs["bp"]
    needle = "fold"
    if case == "other assignment":
        folds = _other_assignment(evs["folds"], dev.height)
        assert [t.size for _, t in folds] == [t.size for _, t in evs["folds"]]
        new = evaluation.evaluate(baselines.BASELINES[2], dev, held, folds)
    elif case == "fewer folds":
        new = evaluation.evaluate(baselines.BASELINES[2], dev, held, evs["folds"][:4])
    elif case == "re-sorted dev":  # the same rows: goal counts per fold may still agree
        dev, needle = dev.sort("shot_id", descending=True), "dev"
    elif case == "re-sorted held":  # a permutation keeps the goal count; the log loss moves
        held, needle = held.sort("shot_id", descending=True), "held|hold-out"
    elif case == "short dev":
        dev, needle = dev.head(dev.height - 1), "dev"
    elif case == "short held":
        held, needle = held.head(held.height - 1), "held|hold-out"
    else:
        side = dev if case == "null dev match" else held
        side = side.with_columns(
            match_id=pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise("match_id")
        )
        dev, held = (side, held) if case == "null dev match" else (dev, side)
        needle = "match_id"
    with pytest.raises(ValueError, match=needle):
        evaluation.compare("x", ref, new, dev, held)


def test_evaluate_threads_freeze_frames(xg_data: Any, dev_held: Split) -> None:
    """A frame feature reaches every design matrix of a fit: the fold fits, the fit on all of
    dev, its predictions on dev and the hold-out, and the coefficient names."""
    dev, held = dev_held
    ff = xg_data.freeze_frames
    names = ("distance_m", "gk_absent", "gk_to_shot_m")
    spec = evaluation.ModelSpec(
        "lr-frames", "LR frames", names, baselines.make_model, baselines.lr_coefficients
    )
    folds = evaluation.cv_folds(dev)
    with pytest.raises(ValueError, match="reads freeze_frames; pass freeze_frames="):
        evaluation.evaluate(spec, dev, held, folds)
    with pytest.raises(ValueError, match="reads freeze_frames; pass freeze_frames="):
        evaluation.fit_predict(spec, dev, held)
    ev = evaluation.evaluate(spec, dev, held, folds, freeze_frames=ff)
    assert [t for t, _, _ in ev.coefficients] == ["intercept", *names]
    x, _ = features.design_matrix(dev, names, freeze_frames=ff)
    xh, _ = features.design_matrix(held, names, freeze_frames=ff)
    y = evaluation.labels(dev)
    fitted = baselines.make_model().fit(x, y)
    np.testing.assert_allclose(ev.holdout_p, fitted.predict_proba(xh)[:, 1], rtol=0, atol=1e-12)
    train, test = folds[0]
    fold = baselines.make_model().fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    np.testing.assert_allclose(ev.oof[test], fold, rtol=0, atol=1e-12)
    p = evaluation.fit_predict(spec, dev[train], dev[test], freeze_frames=ff)
    np.testing.assert_allclose(p, fold, rtol=0, atol=1e-12)


def test_fit_rank_deficient_raises(dev_held: Split) -> None:
    """newton-cholesky on a singular Hessian fails closed, naming the columns that are constant
    on the training rows, instead of switching to lbfgs while the card says newton-cholesky.
    The fixture has no ``body_part=other`` shot, so that indicator is all zero."""
    from football_shot_models import ablation

    dev, held = dev_held
    spec = evaluation.ModelSpec("rd", "RD", ("distance_m", "body_part"), ablation.make_lr)
    with pytest.raises(ValueError) as err:
        evaluation.fit_predict(spec, dev, held)
    message = str(err.value)
    prefix = "rd: rank-deficient design (singular Hessian); constant on the training rows: "
    assert message.startswith(prefix)
    assert "body_part=other" in message
    assert "body_part=head" not in message and "distance_m" not in message[len(prefix) :]
    # every shot is in period 1 at second 30: seconds_in_period = 60 * minute + 30, collinear
    collinear = replace(spec, name="col", features=("minute", "seconds_in_period"))
    with pytest.raises(ValueError, match=r"^col: rank-deficient .*: none \(collinear columns\)$"):
        evaluation.fit_predict(collinear, dev, held)
    # the MS2 lbfgs pipeline never emits the warning: the same design still fits
    lbfgs = replace(spec, make=baselines.make_model)
    assert evaluation.fit_predict(lbfgs, dev, held).shape == (held.height,)
