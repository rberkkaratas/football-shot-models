"""The target-specific leakage guard (spec ``runs/MS5/xgot`` §4, §4.1; test plan 1).

xG keeps :func:`features.check_xg_features`; xGOT runs :func:`features.check_xgot_features`,
which accepts the post-shot goal mouth only through the registered placement features, refuses
every keeper-position feature (PM E3, X1) and leaves every other rule of the xG guard in place.
Both directions are tested: a placement feature is refused by the xG guard, and a raw post-shot
or keeper column is refused by the xGOT guard. The cache tests show a warm hit refuses what a
cold fit refuses, with the spec's own registry.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import xgot_cases as xg

from football_shot_models import evaluation, features
from football_shot_models.cache import PredictionCache

DIGEST = "0123456789abcdef" + "c" * 48
SOURCE = "fedcba9876543210" + "d" * 48


def _frame(n: int = 20) -> pl.DataFrame:
    """On-target shots with placement columns, ``xg_logit`` and ``x`` (for a frame feature)."""
    gy = [(-3.5 + 7.0 * i / (n - 1)) for i in range(n)]
    frame = xg.mouth_frame([34.0 - v for v in gy], [0.1 + 2.2 * i / n for i in range(n)],
                           [20.0 + i for i in range(n)])  # fmt: skip
    return frame.with_columns(
        xg_logit=pl.Series([-2.0 + 0.1 * i for i in range(n)], dtype=pl.Float64()),
        x=pl.Series([90.0 + 0.5 * i for i in range(n)], dtype=pl.Float64()),
    )


def test_xg_guard_refuses_placement() -> None:
    """Every placement feature and ``xg_logit`` is refused by the xG guard, even with the xGOT
    registry passed; a placement feature names its post-shot reads."""
    names = xg.placement_names()
    assert set(names) == set(xg.PLACEMENT_NAMES)  # precondition: the spec's 16 names
    assert features.check_xg_features(["distance_m"], features.XGOT_DERIVED) == ("distance_m",)
    for name in names:
        with pytest.raises(ValueError, match="post-shot") as info:
            features.check_xg_features([name], features.XGOT_DERIVED)
        assert name in str(info.value)
        assert "goal_y" in str(info.value) or "goal_z" in str(info.value)
    with pytest.raises(ValueError, match="'xg_logit' is not a column"):
        features.check_xg_features(["xg_logit"], features.XGOT_DERIVED)
    with pytest.raises(ValueError, match="post-shot"):  # design_matrix's default guard
        features.design_matrix(_frame(), ["mouth_y_abs"], derived=features.XGOT_DERIVED)


def test_xgot_guard_accepts_specs() -> None:
    """The spec table's feature lists pass the xGOT guard; ``xgot_specs`` builds them with the
    xGOT guard, the xGOT registry and the xG input named exactly where ``xg_logit`` is read."""
    from football_shot_models import xgot

    for feats in (xg.LR_FEATURES, xg.GBM_FEATURES, xg.PENALTY_FEATURES, ("xg_logit",), ()):
        assert features.check_xgot_features(feats) == feats
    specs = {s.name: s for s in xgot.xgot_specs("xgot-", "xg-input-name")}
    lr = [f for f in xg.LR_FEATURES]
    expected = {
        "xgot-base-rate": (),
        "xgot-xg-only": ("xg_logit",),
        "xgot-lr-no-far": tuple(f for f in lr if f != "mouth_far"),
        "xgot-lr": xg.LR_FEATURES,
        "xgot-gbm": xg.GBM_FEATURES,
        "xgot-penalty": xg.PENALTY_FEATURES,
    }
    assert {n: set(s.features) for n, s in specs.items()} == {
        n: set(f) for n, f in expected.items()
    }
    for spec in specs.values():
        assert spec.guard is features.check_xgot_features, spec.name
        assert spec.derived is features.XGOT_DERIVED, spec.name
        reads = features.XG_INPUT in spec.features
        assert spec.xg_input == ("xg-input-name" if reads else None), spec.name
        features.check_xgot_features(spec.features, spec.derived)
    ws = {s.name for s in xgot.xgot_specs("ws-xgot-", "ws-x")}
    assert ws == {n.replace("xgot-", "ws-xgot-", 1) for n in expected}


@pytest.mark.parametrize("member", xg.KEEPER)
def test_xgot_guard_refuses_keeper(member: str) -> None:
    """PM E3 / X1: each of the 8 keeper-position features is refused by the xGOT guard and by
    ``check_keeper_neutral``, with its name; the xG guard still allows it (near miss)."""
    assert frozenset(xg.KEEPER) == features.KEEPER_POSITION
    needs = ("gk_absent",) if member.startswith("gk_to") or member == "gk_off_line_m" else ()
    feats = ("xg_logit", "mouth_y_abs", member, *(n for n in needs if n != member))
    with pytest.raises(ValueError, match=member):
        features.check_xgot_features(feats)
    with pytest.raises(ValueError, match=member):
        features.check_keeper_neutral(("distance_m", member))
    assert member in features.check_xg_features(("distance_m", member, *needs))
    kept = ("distance_m", "angle_rad", "one_on_one")  # O2: one_on_one is not a keeper location
    assert features.check_keeper_neutral(kept) == kept


@pytest.mark.parametrize(
    ("column", "pattern"),
    [
        ("goal_y", "placement"),
        ("goal_z", "placement"),
        ("on_target", "placement"),
        ("deflected", "post_shot"),
        ("gk_id", "identity"),
        ("outcome", "outcome"),
        ("end_x", "outcome"),
        ("provider_psxg", "lineage"),
    ],
)
def test_xgot_guard_refuses_raw_goal_mouth(column: str, pattern: str) -> None:
    """A post-shot, outcome, identity or lineage column named directly is refused by the xGOT
    guard: the declared goal mouth and ``on_target`` only through the placement features, the
    rest with their role (``deflected`` is not declared, so it stays refused)."""
    assert features.check_xgot_features(("xg_logit", "mouth_y_abs")) == ("xg_logit", "mouth_y_abs")
    with pytest.raises(ValueError, match=pattern) as info:
        features.check_xgot_features(("xg_logit", column))
    assert column in str(info.value)
    assert set(features.XGOT_DECLARED) == {"goal_y", "goal_z", "on_target"}
    with pytest.raises(TypeError):
        features.check_xgot_features("mouth_y_abs")
    with pytest.raises(ValueError, match="duplicate"):
        features.check_xgot_features(("mouth_y_abs", "mouth_y_abs"))


def test_custom_post_shot_reader_refused() -> None:
    """A derived feature that declares post-shot reads is accepted only as the registered
    ``XGOT_DERIVED`` object of its name: a custom object under a placement name or a new name is
    refused by the xGOT guard, and any reader of post-shot columns by the xG guard."""
    reads = frozenset({"goal_z", "on_target"})
    sources = (("shots", "goal_z"), ("shots", "on_target"))
    fake = features.DerivedFeature("mouth_z", sources, lambda t: t["shots"]["goal_z"], reads=reads)
    other = features.DerivedFeature("height", sources, lambda t: t["shots"]["goal_z"], reads=reads)
    registry = {**features.XGOT_DERIVED, "mouth_z": fake, "height": other}
    assert features.check_xgot_features(("mouth_z",)) == ("mouth_z",)  # the registered one
    for name in ("mouth_z", "height"):
        with pytest.raises(ValueError, match=name):
            features.check_xgot_features((name,), registry)
        with pytest.raises(ValueError, match="post-shot"):
            features.check_xg_features((name,), registry)


def test_reads_outside_declared_raises() -> None:
    """``reads`` must be a subset of ``XGOT_DECLARED`` (the message names the extra columns); a
    source outside ``reads`` keeps the plain role check; reading a declared column works."""
    with pytest.raises(ValueError, match="end_x"):
        features.DerivedFeature("end_dist", (("shots", "end_x"),), lambda t: t["shots"]["end_x"],
                                reads=frozenset({"end_x"}))  # fmt: skip
    with pytest.raises(ValueError, match="deflected"):
        features.DerivedFeature(
            "gy_defl", (("shots", "goal_y"), ("shots", "deflected")),
            lambda t: t["shots"]["goal_y"], reads=frozenset({"goal_y"}),
        )  # fmt: skip
    with pytest.raises(ValueError, match="post_shot"):  # declared column, but no reads
        features.DerivedFeature("gy", (("shots", "goal_y"),), lambda t: t["shots"]["goal_y"])
    ok = features.DerivedFeature("gy", (("shots", "goal_y"),), lambda t: t["shots"]["goal_y"],
                                 reads=frozenset({"goal_y"}))  # fmt: skip
    assert ok.reads == frozenset({"goal_y"})
    assert features.DerivedFeature("d", (("shots", "x"),), lambda t: t["shots"]["x"]).reads == (
        frozenset()
    )


def test_default_guard_unchanged() -> None:
    """A ``ModelSpec`` without a guard keeps ``check_xg_features``: placement features fail in
    ``fit_predict`` even with the xGOT registry; with the xGOT guard the same spec fits. ``DERIVED``
    stays the 17-feature registry and holds no placement feature."""
    frame = _frame()
    spec = evaluation.ModelSpec("t", "T", ("mouth_y_abs",), evaluation.BaseRate,
                                derived=features.XGOT_DERIVED)  # fmt: skip
    assert spec.guard is None and spec.xg_input is None
    with pytest.raises(ValueError, match="post-shot"):
        evaluation.fit_predict(spec, frame, frame)
    ok = dataclasses.replace(spec, guard=features.check_xgot_features)
    assert evaluation.fit_predict(ok, frame, frame).shape == (frame.height,)
    assert len(features.DERIVED) == 17
    assert not set(features.DERIVED) & set(xg.PLACEMENT_NAMES)
    assert dict(features.XGOT_DERIVED) == {
        **features.DERIVED,
        **{n: features.XGOT_DERIVED[n] for n in xg.PLACEMENT_NAMES},
    }
    with pytest.raises(ValueError, match="xg_input"):  # xg_logit without its source: fail closed
        evaluation.fit_predict(
            evaluation.ModelSpec("t", "T", ("xg_logit",), evaluation.BaseRate,
                                 guard=features.check_xgot_features,
                                 derived=features.XGOT_DERIVED),
            frame, frame,
        )  # fmt: skip


def _split(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, Any]:
    dev, held = frame.head(15), frame.tail(5)
    return dev, held, xg.folds_from_tests(15, [[0, 3, 6, 9, 12], [1, 4, 7, 10, 13],
                                               [2, 5, 8, 11, 14]])  # fmt: skip


def test_cache_hit_runs_xgot_guard(tmp_path: Path) -> None:
    """A warm hit runs the spec's guard with the spec's registry: the guard sees
    ``XGOT_DERIVED``, and the same entry read by a spec whose guard is the xG guard raises."""
    dev, held, folds = _split(_frame())
    spec = evaluation.ModelSpec("xgot-guard-t", "T", ("mouth_y_abs", "mouth_z"),
                                evaluation.BaseRate, guard=features.check_xgot_features,
                                derived=features.XGOT_DERIVED)  # fmt: skip
    c = PredictionCache(tmp_path, DIGEST, SOURCE)
    first = evaluation.evaluate(spec, dev, held, folds, cache=c)
    assert c.written == [spec.name]
    seen: list[object] = []

    def spy(names: Any, derived: Any = features.DERIVED) -> tuple[str, ...]:
        seen.append(derived)
        return features.check_xgot_features(names, derived)

    again = evaluation.evaluate(dataclasses.replace(spec, guard=spy), dev, held, folds, cache=c)
    assert c.hits == [spec.name]
    assert np.array_equal(again.oof, first.oof)
    assert seen and all(d is features.XGOT_DERIVED for d in seen)
    refused = dataclasses.replace(spec, guard=None)
    with pytest.raises(ValueError, match="post-shot"):
        evaluation.evaluate(refused, dev, held, folds, cache=c)


def test_cache_hit_checks_spec_derived_frames(tmp_path: Path) -> None:
    """A frame feature registered only in ``spec.derived`` (not in ``features.DERIVED``), read
    warm without freeze frames, raises as the cold fit does."""
    frame = _frame()
    dev, held, folds = _split(frame)
    custom = features.DerivedFeature("frame_x", (("shots", "x"), ("freeze_frames", "x")),
                                     lambda t: t["shots"]["x"])  # fmt: skip
    assert "frame_x" not in features.DERIVED
    spec = evaluation.ModelSpec("xg-frame-t", "T", ("frame_x",), evaluation.BaseRate,
                                derived={**features.DERIVED, "frame_x": custom})  # fmt: skip
    ff = pl.DataFrame({"shot_id": frame["shot_id"], "x": frame["x"]})
    message = "reads freeze_frames"
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate(spec, dev, held, folds)  # cold, no frames
    c = PredictionCache(tmp_path, DIGEST, SOURCE)
    evaluation.evaluate(spec, dev, held, folds, freeze_frames=ff, cache=c)
    assert c.written == [spec.name]
    evaluation.evaluate(spec, dev, held, folds, freeze_frames=ff, cache=c)
    assert c.hits == [spec.name]  # precondition: the entry is read
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate(spec, dev, held, folds, cache=c)
