"""Baselines: grouped CV, the hold-out, the model settings and the leakage invariance (spec §5)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from football_shot_models import baselines, features, metrics

Split = tuple[pl.DataFrame, pl.DataFrame]


def _y(frame: pl.DataFrame) -> np.ndarray:
    return frame["is_goal"].to_numpy().astype(bool)


def test_folds_group_by_match(dev_held: Split) -> None:
    dev, _ = dev_held
    folds = baselines.cv_folds(dev)
    assert baselines.N_SPLITS == 5
    assert len(folds) == 5
    match = dev["match_id"].to_numpy()
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested.tolist()) == list(range(dev.height))  # every row tested exactly once
    for train, test in folds:
        assert train.dtype == np.intp and test.dtype == np.intp
        assert sorted([*train.tolist(), *test.tolist()]) == list(range(dev.height))
        assert set(match[train]).isdisjoint(match[test])
        assert 0 < _y(dev)[test].sum() < test.size  # the fixture puts goals in every fold
    again = baselines.cv_folds(dev)  # deterministic: no shuffle
    for (a, b), (c, d) in zip(folds, again, strict=True):
        assert np.array_equal(a, c) and np.array_equal(b, d)
    four = dev.filter(pl.col("match_id").is_in(sorted(set(match))[:4]))
    with pytest.raises(ValueError):
        baselines.cv_folds(four)


def test_holdout_not_in_dev(dev_held: Split, run_result: tuple[Any, Any]) -> None:
    dev, held = dev_held
    pair = list(zip(held["competition_id"], held["season_id"], strict=True))
    assert set(pair) == {("sb:2", "sb:27")}
    assert ("sb:2", "sb:27") not in set(zip(dev["competition_id"], dev["season_id"], strict=True))
    assert set(dev["match_id"]).isdisjoint(held["match_id"])
    evaluations, _ = run_result
    for ev in evaluations:
        assert (ev.cv.n, ev.train.n, ev.holdout.n) == (dev.height, dev.height, held.height)
        assert ev.oof.shape == (dev.height,)


def test_base_rate_is_train_mean(dev_held: Split) -> None:
    dev, held = dev_held
    base = baselines.BASELINES[0]
    assert base.features == ()
    folds = baselines.cv_folds(dev)
    train, test = dev[folds[0][0]], dev[folds[0][1]]
    p = baselines.fit_predict(base, train, test)
    assert np.array_equal(p, np.full(test.height, train["is_goal"].mean()))
    with pytest.raises(ValueError, match="xg-base-rate: empty training set"):
        baselines.fit_predict(base, train.clear(), test)
    ev = baselines.evaluate(base, dev, held, folds)
    y = _y(dev)
    for (tr, te), fold in zip(folds, ev.folds, strict=True):
        assert np.all(ev.oof[te] == ev.oof[te][0])  # one constant per fold, in dev order
        assert ev.oof[te][0] == pytest.approx(y[tr].mean(), abs=1e-15)
        assert fold == metrics.score(y[te], ev.oof[te])
    assert ev.cv == metrics.mean_scores(ev.folds)
    assert ev.holdout == metrics.score(_y(held), baselines.fit_predict(base, dev, held))
    assert ev.train == metrics.score(y, baselines.fit_predict(base, dev, dev))
    assert ev.coefficients == ()


def test_lr_settings(dev_held: Split, monkeypatch: pytest.MonkeyPatch) -> None:
    assert (baselines.TOL, baselines.MAX_ITER) == (1e-10, 1000)
    model = baselines.make_model()
    (_, scaler), (_, lr) = model.steps
    assert isinstance(scaler, StandardScaler)
    assert isinstance(lr, LogisticRegression)
    assert math.inf == lr.C
    assert lr.class_weight is None  # no rebalancing
    assert lr.solver == "lbfgs"
    assert (lr.tol, lr.max_iter) == (1e-10, 1000)
    monkeypatch.setattr(baselines, "MAX_ITER", 7)  # read at call time
    assert baselines.make_model().steps[-1][1].max_iter == 7
    monkeypatch.undo()

    dev, held = dev_held
    bp = baselines.BASELINES[2]
    ev = baselines.evaluate(bp, dev, held, baselines.cv_folds(dev))
    x, names = features.design_matrix(dev, bp.features)
    terms = [t for t, _, _ in ev.coefficients]
    assert terms[1:] == list(names)  # the intercept first, then the design columns
    standardised = np.array([s for _, s, _ in ev.coefficients])
    natural = np.array([n for _, _, n in ev.coefficients])
    fitted = baselines.make_model().fit(x, _y(dev))
    np.testing.assert_allclose(standardised[0], fitted.steps[-1][1].intercept_[0], atol=1e-9)
    np.testing.assert_allclose(standardised[1:], fitted.steps[-1][1].coef_[0], atol=1e-9)
    # natural units apply to metres, radians and 0 / 1 indicators directly
    p = 1.0 / (1.0 + np.exp(-(natural[0] + x @ natural[1:])))
    np.testing.assert_allclose(p, fitted.predict_proba(x)[:, 1], atol=1e-9, rtol=0)
    assert p.sum() == pytest.approx(ev.train.xg, abs=1e-9 * dev.height)
    xh, _ = features.design_matrix(held, bp.features)
    ph = 1.0 / (1.0 + np.exp(-(natural[0] + xh @ natural[1:])))
    assert ph.sum() == pytest.approx(ev.holdout.xg, abs=1e-9 * held.height)


@pytest.mark.parametrize("baseline", baselines.BASELINES, ids=lambda b: b.name)
def test_fit_predict_reads_only_its_features(baseline: baselines.Baseline, dev_held: Split) -> None:
    """Reversing every other column (provider_xg, is_own_goal, end_*, goal_*, ...) changes
    nothing: no column outside the baseline's features reaches the model."""
    dev, _ = dev_held
    train_idx, test_idx = baselines.cv_folds(dev)[0]
    train, test = dev[train_idx], dev[test_idx]
    keep = {"match_id", "shot_id", "is_goal", *baseline.features}

    def scramble(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col(c).reverse() for c in frame.columns if c not in keep)

    p = baselines.fit_predict(baseline, train, test)
    assert p.shape == (test.height,) and p.dtype == np.float64
    assert np.all((p > 0) & (p < 1))
    assert np.array_equal(p, baselines.fit_predict(baseline, scramble(train), scramble(test)))


def test_benchmark_reads_provider_xg_only_for_scoring(
    dev_held: Split, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object) -> None:
        raise AssertionError("the benchmark must not go through design_matrix")

    monkeypatch.setattr(features, "design_matrix", boom)
    monkeypatch.setattr(baselines, "design_matrix", boom, raising=False)
    dev, held = dev_held
    folds = baselines.cv_folds(dev)
    ev = baselines.evaluate_benchmark(dev, held, folds)
    assert (ev.baseline.name, ev.baseline.title, ev.baseline.features) == (
        "statsbomb-xg",
        "StatsBomb xG",
        (),
    )
    pxg, y = dev["provider_xg"].to_numpy(), _y(dev)
    assert np.array_equal(ev.oof, pxg)
    assert ev.folds == tuple(metrics.score(y[te], pxg[te]) for _, te in folds)
    assert ev.cv == metrics.mean_scores(ev.folds)
    assert ev.holdout == metrics.score(_y(held), held["provider_xg"].to_numpy())


def test_best_ignores_benchmark(run_result: tuple[Any, Any]) -> None:
    evaluations, _ = run_result
    assert [ev.baseline.name for ev in evaluations] == [
        *(b.name for b in baselines.BASELINES),
        "statsbomb-xg",
    ]
    *three, bench = evaluations
    perfect_bench = replace(bench, cv=replace(bench.cv, log_loss=0.0))
    lowest = min(three, key=lambda ev: ev.cv.log_loss)
    assert baselines.best([*three, perfect_bench]) is lowest
    assert baselines.best([perfect_bench, *reversed(three)]) is lowest
    low_base = replace(three[0], cv=replace(three[0].cv, log_loss=1e-6))
    assert baselines.best([low_base, *three[1:], perfect_bench]) is low_base
