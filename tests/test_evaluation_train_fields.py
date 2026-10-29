"""``ModelSpec.train_exclude`` / ``train_indicator``: training-only family fields (spec
``runs/PR1/xg-context`` §3.2, §4).

Frames are hand-written: eight dev shots in four matches over the three families, out of key
order, and three held shots. A recording estimator (a ``DesignFit``, so it receives the design
column names) logs every fit and prediction matrix; its probabilities read every design column,
so a family-derived column at prediction would move them.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from scipy.special import expit

from football_shot_models import data, evaluation
from football_shot_models.cache import PredictionCache
from football_shot_models.evaluation import ModelSpec

LEAGUE, TEAM, WOMEN = "league or tournament", "team-selected", "women"
DEV_FAMILY = [TEAM, LEAGUE, WOMEN, TEAM, LEAGUE, WOMEN, LEAGUE, TEAM]
DEV_DIST = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
FOLDS = (
    (np.array([0, 1, 2, 3], dtype=np.intp), np.array([4, 5, 6, 7], dtype=np.intp)),
    (np.array([4, 5, 6, 7], dtype=np.intp), np.array([0, 1, 2, 3], dtype=np.intp)),
)


def shots(dist: list[float], family: list[str | None], prefix: str) -> pl.DataFrame:
    n = len(dist)
    return pl.DataFrame(
        {
            "shot_id": [f"sb:{prefix}{i}" for i in range(n)],
            "match_id": [f"sb:{prefix}m{i // 2}" for i in range(n)],
            "is_goal": [i % 2 == 0 for i in range(n)],
            "distance_m": dist,
            "family": family,
        },
        schema={"shot_id": pl.String(), "match_id": pl.String(), "is_goal": pl.Boolean(),
                "distance_m": pl.Float64(), "family": data.FAMILY},
    )  # fmt: skip


def dev() -> pl.DataFrame:
    return shots(DEV_DIST, list(DEV_FAMILY), "d")


def held(family: list[str | None] | None = None) -> pl.DataFrame:
    return shots([20.0, 21.0, 22.0], family or [TEAM, LEAGUE, WOMEN], "h")


class Recorder:
    """A ``DesignFit`` that logs ``("fit", x, names)`` and ``("predict", x)``; P(goal) is
    ``expit(Σ x / 100)``, so it depends on every design column."""

    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        self.log = log

    def fit_design(
        self, x: npt.NDArray[np.float64], y: Any, *, names: tuple[str, ...], groups: Any
    ) -> Recorder:
        self.log.append(("fit", x.copy(), tuple(names)))
        return self

    def fit(self, x: npt.NDArray[np.float64], y: Any) -> Recorder:  # pragma: no cover
        raise AssertionError("a DesignFit is fitted through fit_design")

    def predict_proba(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.log.append(("predict", x.copy()))
        p = expit(x.sum(axis=1) / 100.0)
        return np.column_stack([1.0 - p, p])

    def get_params(self, deep: bool = True) -> Mapping[str, Any]:
        return {}


def spec(log: list[tuple[Any, ...]], **kw: Any) -> ModelSpec:
    return ModelSpec("t-fields", "Recorder", ("distance_m",), lambda: Recorder(log), **kw)


def fits(log: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [e for e in log if e[0] == "fit"]


def test_exclude_drops_training_rows_only() -> None:
    """§3.2: ``train_exclude`` rows leave every training frame (each fold fit and the fit on all
    of dev) and nothing else: every dev row gets an out-of-fold prediction, every held row a
    prediction. An exclusion that empties a training set is the existing "empty training set"
    error."""
    log: list[tuple[Any, ...]] = []
    s = spec(log, train_exclude=(TEAM,))
    kept = {d for d, f in zip(DEV_DIST, DEV_FAMILY, strict=True) if f != TEAM}
    ev = evaluation.evaluate(s, dev(), held(), FOLDS)
    trained = [set(e[1][:, 0].tolist()) for e in fits(log)]
    assert len(trained) == 3  # two fold fits and the fit on all of dev
    for (train, _), got in zip(FOLDS, trained, strict=False):
        assert got == {DEV_DIST[i] for i in train} & kept  # unequal: 3 and 2 rows
    assert trained[-1] == kept
    assert not np.isnan(ev.oof).any() and ev.oof.size == 8
    assert ev.holdout_p.size == 3 and ev.holdout.n == 3
    team_only = dev().filter(pl.col("family") == TEAM)
    with pytest.raises(ValueError, match="empty training set"):
        evaluation.fit_predict(s, team_only, held())


@pytest.mark.parametrize(
    "fields",
    [{"train_indicator": (TEAM,)}, {"train_exclude": (WOMEN,), "train_indicator": (TEAM,)}],
)
def test_held_predictions_ignore_family(fields: dict[str, tuple[str, ...]]) -> None:
    """§3.2, §4: the held set's ``family`` is never read: relabelling every held row's family
    (and nulling one) leaves the held predictions unchanged, with the indicator set."""
    log: list[tuple[Any, ...]] = []
    s = spec(log, **fields)
    a = evaluation.evaluate(s, dev(), held(), FOLDS)
    b = evaluation.evaluate(s, dev(), held([WOMEN, TEAM, None]), FOLDS)
    np.testing.assert_array_equal(a.holdout_p, b.holdout_p)
    p1 = evaluation.fit_predict(s, dev(), held())
    p2 = evaluation.fit_predict(s, dev(), held([LEAGUE, LEAGUE, LEAGUE]))
    np.testing.assert_array_equal(p1, p2)


def test_indicator_is_one_when_fitting_and_zero_at_prediction() -> None:
    """§3.2: per ``train_indicator`` level a column ``train_<level, '-' and ' ' → '_'>``, in the
    field's order after the design columns: ``family == level`` as 0/1 when fitting, 0 at every
    prediction (team-selected rows predicted as if not team-selected)."""
    log: list[tuple[Any, ...]] = []
    s = spec(log, train_indicator=(TEAM, LEAGUE))
    d = dev()
    p = evaluation.fit_predict(s, d, d)
    (_, x_fit, names), (_, x_pred) = log
    assert names == ("distance_m", "train_team_selected", "train_league_or_tournament")
    fam = d["family"].cast(pl.String).to_list()
    assert x_fit[:, 1].tolist() == [1.0 if f == TEAM else 0.0 for f in fam]
    assert x_fit[:, 2].tolist() == [1.0 if f == LEAGUE else 0.0 for f in fam]
    assert x_pred.shape == x_fit.shape
    assert (x_pred[:, 1:] == 0.0).all() and x_pred[:, 0].tolist() == DEV_DIST
    np.testing.assert_allclose(p, expit(np.array(DEV_DIST) / 100.0))  # independent route


def test_cache_identity_parts(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """§3.2: with both fields empty the identity ``inputs`` text is today's (``""`` for a spec
    that reads no xG), byte for byte; otherwise ``"|"``-joined parts in order: train_exclude,
    train_indicator, then ``family=<digest>``, and the digest moves with a family value."""
    seen: list[str] = []
    real = evaluation.cache_identity

    def spy(*args: Any, **kwargs: Any) -> str:
        seen.append(kwargs.get("inputs", args[5] if len(args) > 5 else ""))
        return str(real(*args, **kwargs))

    monkeypatch.setattr(evaluation, "cache_identity", spy)
    pc = PredictionCache(tmp_path, "d" * 64, "s" * 64)
    log: list[tuple[Any, ...]] = []

    def run(frame: pl.DataFrame, **kw: Any) -> str:
        evaluation.evaluate(spec(log, **kw), frame, held(), FOLDS, cache=pc)
        return seen[-1]

    assert run(dev()) == ""
    one = run(dev(), train_exclude=(WOMEN,))
    parts = one.split("|")
    assert parts[0] == "train_exclude=women" and len(parts) == 2
    assert re.fullmatch(r"family=[0-9a-f]{64}", parts[1]), parts
    both = run(dev(), train_exclude=(WOMEN, LEAGUE), train_indicator=(TEAM,))
    assert both.split("|")[:2] == ["train_exclude=women,league or tournament",
                                   "train_indicator=team-selected"]  # fmt: skip
    assert both.split("|")[2] == parts[1]
    moved = dev().with_columns(
        pl.when(pl.int_range(pl.len()) == 1).then(pl.lit(WOMEN, dtype=data.FAMILY))
        .otherwise(pl.col("family"))
        .cast(data.FAMILY).alias("family")
    )  # fmt: skip
    assert run(moved, train_exclude=(WOMEN,)).split("|")[1] != parts[1]


@pytest.mark.parametrize(
    ("kw", "error", "needle"),
    [
        ({"train_exclude": TEAM}, TypeError, None),
        ({"train_indicator": WOMEN}, TypeError, None),
        ({"train_exclude": ("team selected",)}, ValueError, "team selected"),
        ({"train_indicator": ("Women",)}, ValueError, "Women"),
        ({"train_exclude": (WOMEN, WOMEN)}, ValueError, WOMEN),
        ({"train_exclude": (WOMEN,), "train_indicator": (WOMEN,)}, ValueError, WOMEN),
    ],
    ids=["exclude-str", "indicator-str", "unknown", "case", "duplicate", "both"],
)
def test_post_init_errors(kw: dict[str, Any], error: type[Exception], needle: str | None) -> None:
    """§3.2: ``__post_init__`` refuses a bare str (TypeError), a level outside
    ``data.FAMILIES``, a duplicate, or a level in both fields (ValueError naming it; the
    unknown-level message lists the allowed)."""
    with pytest.raises(error) as err:
        ModelSpec("t", "T", ("distance_m",), evaluation.BaseRate, **kw)
    if needle is not None:
        assert needle in str(err.value)
    if kw.get("train_exclude") == ("team selected",):
        assert all(f in str(err.value) for f in data.FAMILIES)


def test_family_required_only_when_a_field_is_set() -> None:
    """§3.2: without either field ``family`` is not read (absent is fine); with a field, a missing
    ``family`` or a null one on a training row → ValueError with the count; a null family on a
    predicted row only is fine."""
    log: list[tuple[Any, ...]] = []
    bare = dev().drop("family")
    assert evaluation.fit_predict(spec(log), bare, bare).size == 8
    s = spec(log, train_exclude=(WOMEN,))
    with pytest.raises(ValueError, match="family"):
        evaluation.fit_predict(s, bare, bare)
    nulls = dev().with_columns(
        pl.when(pl.int_range(pl.len()).is_in([1, 6])).then(None).otherwise(pl.col("family"))
        .alias("family")
    )  # fmt: skip
    with pytest.raises(ValueError, match="family") as err:
        evaluation.fit_predict(s, nulls, dev())
    assert re.search(r"\b2\b", str(err.value)), str(err.value)
    assert evaluation.fit_predict(s, dev(), held([None, None, None])).size == 3


def test_fields_default_empty_and_carried_by_replace() -> None:
    """§3.2: both fields default to ``()``; ``dataclasses.replace`` (as in ``xgot.keeper_neutral``
    and ``season_out``) carries them."""
    log: list[tuple[Any, ...]] = []
    base = spec(log)
    assert (base.train_exclude, base.train_indicator) == ((), ())
    s = spec(log, train_exclude=(WOMEN,), train_indicator=(TEAM,))
    r = dataclasses.replace(s, name="t-renamed", features=("distance_m", "angle_rad"))
    assert (r.train_exclude, r.train_indicator) == ((WOMEN,), (TEAM,))


# --- build ratify (runs/PR1/xg-context/build.md §3.2 items 4-5) --------------------------------


def test_family_identity_reads_dev_codes_only() -> None:
    """Ratify 4: the ``family=`` identity part hashes the *dev* rows' family as codes in
    ``data.FAMILIES`` order: relabelling (or nulling) every held row's family leaves it unchanged,
    a String dev column hashes like the Enum one, and an unknown dev family raises (it would be
    null)."""
    s = spec([], train_exclude=(WOMEN,))
    base = evaluation._inputs(s, dev(), held())
    assert base.startswith("train_exclude=women|family=")  # precondition: the part is there
    assert evaluation._inputs(s, dev(), held([WOMEN, None, TEAM])) == base
    as_text = dev().with_columns(pl.col("family").cast(pl.String))
    assert evaluation._inputs(s, as_text, held()) == base
    swapped = dev().with_columns(  # same rows, two families swapped: the codes move
        pl.col("family").cast(pl.String).replace({TEAM: LEAGUE, LEAGUE: TEAM})
    )
    assert evaluation._inputs(s, swapped, held()) != base
    unknown = as_text.with_columns(pl.lit("mixed").alias("family"))
    with pytest.raises(ValueError):
        evaluation._inputs(s, unknown, held())


@pytest.mark.parametrize(
    ("fields", "expected"),
    [({}, False), ({"train_exclude": (TEAM,)}, True), ({"train_indicator": (TEAM,)}, True),
     ({"train_exclude": (WOMEN,), "train_indicator": (TEAM,)}, True)],
)  # fmt: skip
def test_trains_by_family(fields: dict[str, tuple[str, ...]], expected: bool) -> None:
    """Ratify 5: ``ModelSpec.trains_by_family`` is True iff either family field is set; an
    indicator-only spec reads ``family`` (a dev frame without it is refused) and its identity
    gains the ``family=`` part."""
    s = spec([], **fields)
    assert s.trains_by_family is expected
    assert ("family=" in evaluation._inputs(s, dev(), held())) is expected
    if expected:
        with pytest.raises(ValueError, match="family column"):
            evaluation.fit_predict(s, dev().drop("family"), held())
