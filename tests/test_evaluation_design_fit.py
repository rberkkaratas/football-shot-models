"""The harness's ``DesignFit`` branch (spec MS3.models §4, §9): an estimator that needs the design
names and match groups gets them, from the training rows only.

Uses the MS2 fixture snapshot (``xg_cases``) through the session fixtures.
"""

from __future__ import annotations

import functools
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest

from football_shot_models import evaluation


class _Spy:
    """A DesignFit that records what the harness hands it and predicts the training rate."""

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls, self.rate = calls, 0.0

    def fit(self, x: Any, y: Any) -> _Spy:
        raise AssertionError("the harness called fit(x, y) on a DesignFit estimator")

    def fit_design(self, x: Any, y: Any, *, names: tuple[str, ...], groups: Any) -> _Spy:
        self.calls.append({"rows": x.shape[0], "y": y.copy(), "names": names, "groups": groups})
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, x: Any) -> Any:
        p = np.full(x.shape[0], self.rate)
        return np.column_stack([1.0 - p, p])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


def _spy_spec(calls: list[dict[str, Any]]) -> evaluation.ModelSpec:
    return evaluation.ModelSpec(
        "spy", "Spy", ("distance_m", "angle_rad"), functools.partial(_Spy, calls)
    )


def test_design_fit_sees_train_rows_only(dev_held: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """§4, §9 (outer fold -> tuning and Platt): each fold fit gets the design names, the labels
    and the ``match_id`` of its training rows, none of its test matches; the dev fit gets all of
    dev."""
    dev, held = dev_held
    folds = evaluation.cv_folds(dev)
    calls: list[dict[str, Any]] = []
    spec = _spy_spec(calls)
    assert isinstance(spec.make(), evaluation.DesignFit)
    evaluation.evaluate(spec, dev, held, folds)
    assert len(calls) == len(folds) + 1
    y = evaluation.labels(dev)
    for call, (train, test) in zip(calls, folds, strict=False):
        assert call["names"] == ("distance_m", "angle_rad")
        assert np.array_equal(call["groups"], dev[train]["match_id"].to_numpy())
        assert np.array_equal(call["y"], y[train])
        assert not set(call["groups"].tolist()) & set(dev[test]["match_id"].to_list())
    assert np.array_equal(calls[-1]["groups"], dev["match_id"].to_numpy())
    assert calls[-1]["rows"] == dev.height


def test_design_fit_null_match_id(dev_held: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """§4: a null ``match_id`` in the training rows of a DesignFit is a ValueError naming the
    spec and the count. Near misses: the same nulls in the test rows only, or a model that is
    not a DesignFit, raise nothing."""
    dev, _ = dev_held
    train = mc.with_null(mc.with_null(dev, "match_id", 3), "match_id", 17)
    calls: list[dict[str, Any]] = []
    with pytest.raises(ValueError) as info:
        evaluation.fit_predict(_spy_spec(calls), train, dev.head(4))
    assert str(info.value) == "spy: match_id has 2 null value(s) in the training rows"
    assert calls == []
    assert evaluation.fit_predict(_spy_spec(calls), dev, train.head(4)).shape == (4,)
    base = evaluation.ModelSpec("base", "Base", (), evaluation.BaseRate)
    assert evaluation.fit_predict(base, train, dev.head(4)).shape == (4,)


def test_design_fit_gbm_refusal_passes_through() -> None:
    """§3.1, §4: the GBM's own refusal (constant inner scores) reaches the caller unchanged,
    not as the harness's rank-deficient-design message."""
    from football_shot_models import gbm

    _, y, groups = mc.constant_xy()
    frame = pl.DataFrame(
        {
            "match_id": pl.Series([f"sb:m{g:02d}" for g in groups], dtype=pl.String()),
            "is_goal": pl.Series(y, dtype=pl.Boolean()),
            "distance_m": pl.Series([12.0] * y.size, dtype=pl.Float64()),
            "angle_rad": pl.Series([0.5] * y.size, dtype=pl.Float64()),
        }
    )
    make = functools.partial(
        gbm.MonotoneGBM, {"distance_m": -1}, leaf_grid=(4,), max_iter=20, step=10
    )
    spec = evaluation.ModelSpec("gbm", "GBM", ("distance_m", "angle_rad"), make)
    with pytest.raises(ValueError) as info:
        evaluation.fit_predict(spec, frame, frame.head(2))
    assert "Platt scaling is not estimable" in str(info.value)
    assert "rank-deficient" not in str(info.value)
