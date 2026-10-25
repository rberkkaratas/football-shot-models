"""The target-specific guard seam and the labelled big-chance ablation (spec MS4.v1 §2.2, §1.6).

``big_chance`` (Opta q214, role ``restricted``) is refused by the default guard everywhere;
``features.check_ablation_features`` admits it on top of the default rule, and
``evaluation.best`` never picks a spec that reads an ``ABLATION_ONLY`` column, whatever its
guard. The frames are the synthetic WhoScored fixture (``v1_cases``), whose ``big_chance`` is
never null on the kept shots.
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
import v1_cases as vc

from football_shot_models import ablation, evaluation, features

DIGEST, SOURCE = "d" * 64, "s" * 64


def _frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(dev, held) of the fixture's kept shots, split by season (no data.py code involved)."""
    shots, _, matches = vc.v1_tables()
    kept = vc.kept_open_play(shots)
    test = matches.filter(pl.col("season_name").is_in(vc.TEST_SEASONS))["match_id"]
    held = pl.col("match_id").is_in(test.implode())
    return kept.filter(~held), kept.filter(held)


def test_default_guard_refuses_big_chance() -> None:
    """§2.2: ``design_matrix`` without a guard refuses ``big_chance`` (role ``restricted``), and
    with ``guard=check_ablation_features`` builds its 0 / 1 column; ``ModelSpec.guard`` defaults
    to ``None``; ``ABLATION_ONLY`` is a read-only mapping holding ``big_chance`` alone."""
    dev, _ = _frames()
    with pytest.raises(ValueError, match="'big_chance' has role 'restricted'"):
        features.design_matrix(dev, ["distance_m", "big_chance"])
    with pytest.raises(ValueError, match="'big_chance' has role 'restricted'"):
        features.design_matrix(dev, ["distance_m", "big_chance"], guard=None)
    x, names = features.design_matrix(
        dev, ["distance_m", "big_chance"], guard=features.check_ablation_features
    )
    assert names == ("distance_m", "big_chance")
    assert x[:, 1].tolist() == dev["big_chance"].cast(pl.Float64).to_list()
    assert 0 < x[:, 1].sum() < dev.height  # both values occur
    spec = evaluation.ModelSpec("x", "X", ("distance_m",), ablation.make_lr)
    assert spec.guard is None
    assert set(features.ABLATION_ONLY) == {"big_chance"}
    with pytest.raises(TypeError):
        features.ABLATION_ONLY["first_time"] = "no"  # type: ignore[index]


@pytest.mark.parametrize(
    ("names", "error", "message"),
    [
        (["distance_m", "goal_y", "big_chance"], ValueError, "'goal_y' has role 'post_shot'"),
        (["end_x", "big_chance"], ValueError, "'end_x' has role 'outcome'"),
        (["big_chance", "distance_m", "big_chance"], ValueError, "duplicate"),
        (["distance_m", "body_part"], ValueError, "big_chance"),
        ([], ValueError, "big_chance"),
        ("big_chance", TypeError, "single str"),
    ],
)
def test_check_ablation_features(names: Any, error: type[Exception], message: str) -> None:
    """§2.2: the ablation guard admits ``big_chance`` plus allowed columns (in the given order)
    and refuses every other non-``pre_shot`` column, duplicates, a list without an
    ``ABLATION_ONLY`` column and a bare ``str``; ``uses_ablation_only`` is the one rule."""
    assert features.check_ablation_features(["distance_m", "big_chance", "phase"]) == (
        "distance_m", "big_chance", "phase",
    )  # fmt: skip
    assert features.uses_ablation_only(("distance_m", "big_chance")) is True
    assert features.uses_ablation_only(("distance_m", "one_on_one")) is False
    assert features.uses_ablation_only(()) is False
    with pytest.raises(TypeError):
        features.uses_ablation_only("big_chance")
    with pytest.raises(error, match=message):
        features.check_ablation_features(names)


def _spec(name: str, feats: tuple[str, ...], guard: Any = None) -> evaluation.ModelSpec:
    return evaluation.ModelSpec(name, name, feats, ablation.make_lr, guard=guard)


GUARDS: dict[str, Any] = {
    "ablation guard": features.check_ablation_features,
    "partial-wrapped": functools.partial(features.check_ablation_features),
    "default guard": None,
    "permissive lambda": lambda names, derived: tuple(names),
}


@pytest.mark.parametrize("guard", list(GUARDS))
def test_best_skips_ablation(guard: str) -> None:
    """§2.2: ``best`` skips a spec with an ``ABLATION_ONLY`` feature even when it has the lowest
    CV loss, whatever its guard; a spec only *named* like an ablation competes (the rule reads
    the features); with no candidate left the message names both exclusions."""
    lr = mc.fake_eval(_spec("ws-lr", ("distance_m",)), cv=mc.scores(log_loss=0.30))
    named = mc.fake_eval(
        _spec("ws-lr-big-chance", ("distance_m", "angle_rad"), features.check_ablation_features),
        cv=mc.scores(log_loss=0.29),
    )
    abl = mc.fake_eval(_spec("ws-abl", ("distance_m", "big_chance"), GUARDS[guard]),
                       cv=mc.scores(log_loss=0.20))  # fmt: skip
    bench = mc.fake_eval(evaluation.BENCHMARK, cv=mc.scores(log_loss=0.10))
    assert abl.cv.log_loss < named.cv.log_loss < lr.cv.log_loss  # the ablation would win
    assert evaluation.best([abl, lr, bench]) is lr
    assert evaluation.best([lr, named, abl, bench]) is named
    with pytest.raises(ValueError, match=r"(?is)(?=.*ablation)(?=.*benchmark)"):
        evaluation.best([abl, bench])


def test_cached_ablation_runs_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§2.2: the spec's own guard runs on a fit and again on a cache hit (no fit then); the
    default guard is not what a hit checks for an ablation spec (it would refuse
    ``big_chance``); a guard that refuses blocks the hit; the guard is not part of the cache
    identity."""
    from football_shot_models import cache

    dev, held = _frames()
    folds = evaluation.cv_folds(dev)
    seen: list[tuple[str, ...]] = []

    def spy(names: Any, derived: Any = features.DERIVED) -> tuple[str, ...]:
        seen.append(tuple(names))
        return features.check_ablation_features(names, derived)

    spec = _spec("ws-test-ablation", ("distance_m", "big_chance"), spy)
    c = cache.PredictionCache(tmp_path, DIGEST, SOURCE)
    first = evaluation.evaluate(spec, dev, held, folds, cache=c)
    assert c.written == [spec.name] and seen  # the fits ran the spec's guard
    seen.clear()

    def refuse_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fitted on a cache hit")

    monkeypatch.setattr(evaluation, "_fit", refuse_fit)
    again = evaluation.evaluate(spec, dev, held, folds, cache=c)
    assert c.hits == [spec.name]
    assert ("distance_m", "big_chance") in seen
    assert np.array_equal(again.holdout_p, first.holdout_p)

    def no(names: Any, derived: Any = features.DERIVED) -> tuple[str, ...]:
        raise ValueError("guard says no")

    with pytest.raises(ValueError, match="guard says no"):
        evaluation.evaluate(dataclasses.replace(spec, guard=no), dev, held, folds, cache=c)
