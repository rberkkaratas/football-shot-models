"""Cross-fitting the xG input into xGOT and the folds it runs on (spec ``runs/MS5/xgot`` §5;
test plan 3).

The out-of-fold xG of the provider's xG input (the headline refitted keeper-neutral, PM X1 (b))
fills ``xg_logit``: a dev row reads the fit that never saw its fold, a hold-out row the fit on all
of dev, and the xGOT folds are the same fold assignment restricted to on-target rows. The fold
fixtures have unequal folds and rows that are not in fold order; the fold marker in ``oof``
shows which fit each row read.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
import xgot_cases as xg

from football_shot_models import ablation, cache, data, evaluation, features, gbm, models

DIGEST = "0123456789abcdef" + "e" * 48
SOURCE = "fedcba9876543210" + "f" * 48
# 12 dev rows in 3 unequal folds, not in row order; every fold keeps an on-target row (spec 5.1:
# restrict_folds refuses a fold with no test row left)
TESTS = ([5, 0, 9, 2], [7, 1], [3, 4, 6, 8, 10, 11])
ON_TARGET = [True, True, True, True, False, True, True, False, True, True, False, True]


def _dev(n: int = 12, on_target: list[bool] | None = None) -> pl.DataFrame:
    return pl.DataFrame({
        "shot_id": [f"sb:s{i:02d}" for i in range(n)],
        "match_id": [f"sb:m{i:02d}" for i in range(n)],
        "is_goal": [i % 3 == 0 for i in range(n)],
        "on_target": ON_TARGET if on_target is None else on_target,
    })  # fmt: skip


def _held() -> pl.DataFrame:
    return pl.DataFrame({
        "shot_id": ["sb:h0", "sb:h1", "sb:h2", "sb:h3"],
        "match_id": ["sb:n0", "sb:n0", "sb:n1", "sb:n1"],
        "is_goal": [True, False, False, True],
        "on_target": [True, False, True, True],
    })  # fmt: skip


def _marker_eval(folds: Any, n: int = 12) -> evaluation.Evaluation:
    """An Evaluation whose out-of-fold value is the fold's marker 0.1 * (k + 1)."""
    oof = np.full(n, 0.5)  # a row no fold tests keeps 0.5
    for k, (_, test) in enumerate(folds):
        oof[test] = 0.1 * (k + 1)
    spec = evaluation.ModelSpec("xg-input-t", "xG input", ("distance_m",), ablation.make_lr)
    ev = mc.fake_eval(spec, oof=oof, holdout_p=np.array([0.71, 0.72, 0.73, 0.74]))
    return dataclasses.replace(ev, fold_test=tuple(test for _, test in folds))


def test_restrict_folds() -> None:
    """The same fold assignment on the masked rows, re-indexed in row order (train and test
    sorted); a bad mask length, a non-bool mask and a fold left without test rows raise."""
    folds = xg.folds_from_tests(10, [[3, 0, 7], [1, 8], [2, 4, 5, 6, 9]])
    mask = np.array([True, False, True, True, False, True, False, True, True, False])
    out = evaluation.restrict_folds(folds, mask)
    # kept rows 0, 2, 3, 5, 7, 8 -> 0..5
    expected = ([0, 2, 4], [5], [1, 3])
    assert len(out) == 3
    for (train, test), want in zip(out, expected, strict=True):
        assert test.tolist() == want
        assert train.tolist() == [i for i in range(6) if i not in want]
        assert test.dtype == np.intp and train.dtype == np.intp
    with pytest.raises(ValueError, match="mask"):
        evaluation.restrict_folds(folds, mask[:-1])
    with pytest.raises(ValueError, match="bool"):
        evaluation.restrict_folds(folds, mask.astype(np.int64))
    emptied = mask.copy()
    emptied[[1, 8]] = False  # fold 1 loses its only kept row
    with pytest.raises(ValueError, match="fold"):
        evaluation.restrict_folds(folds, emptied)
    season = xg.folds_from_tests(6, [[0, 1, 2], [3, 4, 5]])  # any partition passes
    assert [t.tolist() for _, t in evaluation.restrict_folds(season, np.ones(6, bool))] == [
        [0, 1, 2], [3, 4, 5],
    ]  # fmt: skip


def test_folds_must_partition() -> None:
    """``check_partition`` refuses rows tested by no fold (forward chaining) or by two, and an
    index out of range, naming the counts; ``restrict_folds`` and ``xg_input`` call it."""
    from football_shot_models import xgot

    evaluation.check_partition(xg.folds_from_tests(12, TESTS), 12)  # near miss: a partition
    forward = (
        (np.arange(0, 4, dtype=np.intp), np.arange(4, 8, dtype=np.intp)),
        (np.arange(0, 8, dtype=np.intp), np.arange(8, 12, dtype=np.intp)),
    )  # rows 0-3 are never tested
    overlap = xg.folds_from_tests(12, ([5, 0, 9, 2, 7], [7, 1], [3, 4, 6, 8, 10, 11]))
    outside = xg.folds_from_tests(12, ([5, 0, 9, 2], [7, 1, 12], [3, 4, 6, 8, 10, 11]))
    for folds, count in ((forward, "4"), (overlap, "1"), (outside, "1|12")):
        with pytest.raises(ValueError, match=rf"\b({count})\b"):
            evaluation.check_partition(folds, 12)
        with pytest.raises(ValueError):
            evaluation.restrict_folds(folds, np.ones(12, dtype=bool))
    ev = _marker_eval(forward)
    with pytest.raises(ValueError, match=r"\b4\b"):
        xgot.xg_input(ev, _dev(), _held(), forward)


def test_folds_refuse_training_on_the_test_rows() -> None:
    """A fold whose train set holds its own test rows is refused (run5-ms5 Q6): its
    out-of-fold predictions would be in-sample. Test sets alone still partition dev."""
    leaky = tuple((np.arange(6, dtype=np.intp), t) for t in (np.arange(3), np.arange(3, 6)))
    clean = xg.folds_from_tests(6, [[0, 1, 2], [3, 4, 5]])
    evaluation.check_partition(clean, 6)  # near miss: the same test sets, disjoint train
    with pytest.raises(ValueError, match=r"fold 0: 3 row\(s\) are in both"):
        evaluation.check_partition(leaky, 6)
    with pytest.raises(ValueError, match="both its train and test"):
        evaluation.restrict_folds(leaky, np.ones(6, dtype=bool))


def test_xg_input_cross_fitted() -> None:
    """Dev rows read their own fold's out-of-fold xG and keep their fold; hold-out rows read the
    dev fit; ``xg_logit`` is the clipped logit; only on-target rows reach the xGOT frames."""
    from football_shot_models import metrics, xgot

    folds = xg.folds_from_tests(12, TESTS)
    ev = _marker_eval(folds)
    dev, held = _dev(), _held()
    inp = xgot.xg_input(ev, dev, held, folds)
    assert inp.spec_name == "xg-input-t"
    assert inp.dev["xg"].to_list() == pytest.approx(ev.oof.tolist())
    assert inp.dev["xg_logit"].to_list() == pytest.approx(metrics.logit(ev.oof).tolist())
    assert inp.dev.schema["xg"] == pl.Float64() and inp.dev.schema["xg_logit"] == pl.Float64()
    assert inp.held["xg"].to_list() == pytest.approx([0.71, 0.72, 0.73, 0.74])
    assert inp.on_dev["shot_id"].to_list() == dev.filter("on_target")["shot_id"].to_list()
    assert inp.on_held["shot_id"].to_list() == ["sb:h0", "sb:h2", "sb:h3"]
    assert inp.on_held["xg"].to_list() == pytest.approx([0.71, 0.73, 0.74])
    assert len(inp.folds) == 3
    for k, ((_, test), (_, orig)) in enumerate(zip(inp.folds, folds, strict=True)):
        rows = inp.on_dev[test]
        assert rows["xg"].to_list() == pytest.approx([0.1 * (k + 1)] * rows.height)
        kept = [f"sb:s{i:02d}" for i in sorted(orig.tolist()) if ON_TARGET[i]]
        assert rows["shot_id"].to_list() == kept
    evaluation.check_partition(inp.folds, inp.on_dev.height)
    assert metrics.logit(np.array([0.0, 1.0, 0.5])).tolist() == pytest.approx(
        [np.log(metrics.EPS), -np.log(metrics.EPS), 0.0], rel=1e-6
    )


def test_xg_input_refuses_a_fold_with_no_on_target_row() -> None:
    """Engineer decision D1 (spec 5.1): when the on-target mask empties an xG fold (fold 1 is
    rows 7 and 1), ``xg_input`` refuses through ``restrict_folds``, naming the fold, rather than
    returning a fold with no test row."""
    from football_shot_models import xgot

    folds = xg.folds_from_tests(12, TESTS)
    on = list(ON_TARGET)
    assert on[7] is False and on[1] is True  # precondition: fold 1 keeps only row 1
    xgot.xg_input(_marker_eval(folds), _dev(on_target=on), _held(), folds)  # near miss passes
    on[1] = False  # the pre-D1 fixture
    with pytest.raises(ValueError, match=r"fold 1\b"):
        xgot.xg_input(_marker_eval(folds), _dev(on_target=on), _held(), folds)


def test_xg_input_refuses_other_folds() -> None:
    """An xG evaluation scored on other folds (same sizes, one row moved), a prediction array of
    the wrong size, or a NaN prediction is refused."""
    from football_shot_models import xgot

    folds = xg.folds_from_tests(12, TESTS)
    moved = xg.folds_from_tests(12, ([5, 0, 9, 1], [7, 2], [3, 4, 6, 8, 10, 11]))
    with pytest.raises(ValueError, match=r"fold [01]\b"):
        xgot.xg_input(_marker_eval(moved), _dev(), _held(), folds)
    ev = _marker_eval(folds)
    with pytest.raises(ValueError):
        xgot.xg_input(dataclasses.replace(ev, oof=ev.oof[:-1]), _dev(), _held(), folds)
    with pytest.raises(ValueError):
        xgot.xg_input(dataclasses.replace(ev, holdout_p=ev.holdout_p[:-1]), _dev(), _held(), folds)
    nan = ev.oof.copy()
    nan[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        xgot.xg_input(dataclasses.replace(ev, oof=nan), _dev(), _held(), folds)


# ---- keeper-neutral input and the headline rule --------------------------------------------


def _scored(n_matches: int = 12, per: int = 5) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = n_matches * per
    dev = pl.DataFrame({
        "shot_id": [f"sb:s{i:03d}" for i in range(n)],
        "match_id": [f"sb:m{i // per:02d}" for i in range(n)],
        "is_goal": [i % 3 == 0 for i in range(n)],
        "distance_m": [5.0 + (i % 7) for i in range(n)],
        "angle_rad": [0.3 + 0.05 * (i % 5) for i in range(n)],
        "open_goal": [i % 11 == 0 for i in range(n)],
        "gk_x": [100.0 + 0.1 * (i % 9) for i in range(n)],
        "big_chance": [i % 4 == 0 for i in range(n)],
    })  # fmt: skip
    held = dev.head(15).with_columns(
        shot_id=pl.Series([f"sb:h{i:03d}" for i in range(15)]),
        match_id=pl.Series([f"sb:n{i // 5}" for i in range(15)]),
    )
    return dev, held


def _sharp(dev: pl.DataFrame, held: pl.DataFrame, spec: Any, folds: Any, q: float) -> Any:
    """An evaluation that gives goals ``q`` and misses ``1 - q``."""
    y, yh = evaluation.labels(dev), evaluation.labels(held)
    return xg.make_eval(spec, dev, held, folds, np.where(y, q, 1 - q), np.where(yh, q, 1 - q))


def test_keeper_neutral_spec() -> None:
    """``keeper_neutral`` drops exactly the ``KEEPER_POSITION`` features (a GBM's sign map with
    them), renames and retitles; a spec without one is returned itself. ``xg_input_eval`` refits
    the keeper-neutral headline and refuses an ablation-only input."""
    from football_shot_models import xgot

    lr = evaluation.ModelSpec(
        "xg-k", "Keeper LR",
        ("distance_m", "gk_in_cone", "angle_rad", "open_goal", "gk_absent", "gk_to_goal_m"),
        ablation.make_lr,
    )  # fmt: skip
    kn = xgot.keeper_neutral(lr)
    assert kn.features == ("distance_m", "angle_rad")
    assert (kn.name, kn.title) == ("xg-k-keeper-neutral", "Keeper LR without keeper position")
    assert (kn.make, kn.coefficients, kn.guard) == (lr.make, lr.coefficients, lr.guard)
    signs = {"distance_m": -1, "gk_to_shot_m": 1, "angle_rad": 1}
    g = evaluation.ModelSpec(
        "xg-g", "G", ("distance_m", "gk_to_shot_m", "gk_absent", "angle_rad"),
        functools.partial(gbm.MonotoneGBM, signs, **mc.TINY), gbm.gbm_settings,
    )  # fmt: skip
    params = xgot.keeper_neutral(g).make().get_params()
    assert params["monotone"] == {"distance_m": -1, "angle_rad": 1}
    assert params["leaf_grid"] == mc.TINY["leaf_grid"]
    ws = evaluation.ModelSpec("ws-xg", "WS", ("distance_m", "one_on_one"), ablation.make_lr)
    assert xgot.keeper_neutral(ws) is ws

    dev, held = _scored()
    folds = evaluation.cv_folds(dev)
    base = evaluation.ModelSpec("xg-lr-t", "LR", ("distance_m",), evaluation.BaseRate)
    lr_ev = _sharp(dev, held, base, folds, 0.5)
    keeper = evaluation.ModelSpec("xg-gbm-t", "GBM", ("distance_m", "open_goal", "gk_x"),
                                  evaluation.BaseRate)  # fmt: skip
    abl = evaluation.ModelSpec(
        "xg-abl-t", "Ablation", ("distance_m", "big_chance"), evaluation.BaseRate,
        guard=features.check_ablation_features,
    )  # fmt: skip
    xg_data = data.XgData(dev, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame(),
                          None)  # type: ignore[arg-type]  # fmt: skip

    def provider(head_spec: Any) -> Any:
        head_ev = _sharp(dev, held, head_spec, folds, 0.9)  # beats the LR: the headline
        return dataclasses.replace(xgot.PROVIDERS["statsbomb"],
                                   candidates=lambda *a: ((lr_ev,), head_ev))  # fmt: skip

    inp, head = xgot.xg_input_eval(provider(keeper), xg_data, dev, held, folds, None)
    assert head.spec is keeper
    assert inp.spec.name == "xg-gbm-t-keeper-neutral" and inp.spec.features == ("distance_m",)
    assert [t.tolist() for t in inp.fold_test] == [t.tolist() for _, t in folds]
    with pytest.raises(ValueError, match="big_chance|ablation"):
        xgot.xg_input_eval(provider(abl), xg_data, dev, held, folds, None)


def test_xgot_cache_identity_reads_input(tmp_path: Path) -> None:
    """The cache identity of a spec reading ``xg_logit`` holds the input's name and a digest of
    its values: another name or one changed value is a miss; the same is a hit. Without
    ``xg_input`` such a spec fails closed."""
    ids = pl.Series(["sb:1"]), pl.Series(["sb:2"])
    base = cache.identity("xgot-x", ("xg_logit",), {}, *ids)
    assert cache.identity("xgot-x", ("xg_logit",), {}, *ids, inputs="") == base
    assert cache.identity("xgot-x", ("xg_logit",), {}, *ids, inputs="xg-a abc") == (
        base + "|inputs xg-a abc"
    )
    frame = xg.mouth_frame([33.0] * 20, [1.0] * 20, [20.0] * 20).with_columns(
        xg_logit=pl.Series([-2.0 + 0.1 * i for i in range(20)])
    )
    dev, held = frame.head(15), frame.tail(5)
    changed = dev.with_columns(xg_logit=pl.Series([9.0] + dev["xg_logit"].to_list()[1:]))
    d1 = cache.column_digest(dev, held, "xg_logit")
    assert d1 == cache.column_digest(dev, held, "xg_logit") and len(d1) == 64
    assert cache.column_digest(changed, held, "xg_logit") != d1
    folds = xg.folds_from_tests(15, [[0, 3, 6, 9, 12], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14]])
    spec = evaluation.ModelSpec("xgot-in-t", "T", ("xg_logit",), evaluation.BaseRate,
                                guard=features.check_xgot_features,
                                derived=features.XGOT_DERIVED, xg_input="xg-a")  # fmt: skip
    c = cache.PredictionCache(tmp_path, DIGEST, SOURCE)
    evaluation.evaluate(spec, dev, held, folds, cache=c)
    evaluation.evaluate(spec, dev, held, folds, cache=c)
    assert (c.written, c.hits) == ([spec.name], [spec.name])
    renamed = dataclasses.replace(spec, xg_input="xg-b")
    evaluation.evaluate(renamed, dev, held, folds, cache=c)  # another input name: a miss
    assert (len(c.written), len(c.hits)) == (2, 1)
    evaluation.evaluate(renamed, dev, held, folds, cache=c)
    assert (len(c.written), len(c.hits)) == (2, 2)
    evaluation.evaluate(renamed, changed, held, folds, cache=c)  # one value changed: a miss
    assert (len(c.written), len(c.hits)) == (3, 2)
    with pytest.raises(ValueError, match="xg_input"):
        evaluation.evaluate(dataclasses.replace(spec, xg_input=None), dev, held, folds, cache=c)


def test_select_headline() -> None:
    """``select_headline`` is ``headline(best(candidates), gbm)``: the best LR by CV log loss
    (an ablation never qualifies, however good), then the GBM iff its CV gain CI lies above 0."""
    dev, held = _scored()
    folds = evaluation.cv_folds(dev)

    def spec(name: str, feats: tuple[str, ...] = ("distance_m",), **kw: Any) -> Any:
        return evaluation.ModelSpec(name, name.upper(), feats, evaluation.BaseRate, **kw)

    worse = _sharp(dev, held, spec("xg-a"), folds, 0.55)
    better = _sharp(dev, held, spec("xg-b"), folds, 0.6)
    abl = _sharp(dev, held, spec("xg-c", ("distance_m", "big_chance"),
                                 guard=features.check_ablation_features), folds, 0.95)  # fmt: skip
    candidates = (worse, better, abl)
    for q, want in ((0.9, "xg-gbm"), (0.52, "xg-b")):
        g = _sharp(dev, held, spec("xg-gbm"), folds, q)
        head, gain = models.select_headline(candidates, g, dev, held)
        ref_head, ref_gain = models.headline(evaluation.best(candidates), g, dev, held)
        assert head is ref_head and head.spec.name == want
        assert gain == ref_gain and gain.label == "over XG-B"
