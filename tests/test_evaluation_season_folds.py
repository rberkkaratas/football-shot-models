"""``evaluation.season_folds``: grouped folds by competition-season (spec ``runs/MS6/gsax`` §3.1).

The StatsBomb season-out xGOT rates a row from fits that never saw its season, so the folds
group by ``(competition_id, season_id)``: a season is in exactly one test set and never in its
own fold's training rows. The fixture holds the composite-key near misses (two seasons of one
competition, one ``season_id`` in two competitions), unequal season sizes and rows out of key
order.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from football_shot_models import evaluation

BY = ("competition_id", "season_id")


def _dev() -> pl.DataFrame:
    # 5 seasons of 7, 2, 4, 3, 5 rows; the rows interleave the seasons
    keys = [("sb:9", "sb:42")] * 7 + [("sb:9", "sb:90")] * 2 + [("sb:72", "sb:30")] * 4
    keys += [("sb:16", "sb:30")] * 3 + [("sb:2", "sb:27")] * 5
    order = [(i * 8) % len(keys) for i in range(len(keys))]  # 8 is coprime with 21
    assert sorted(order) == list(range(len(keys)))
    rows = [keys[i] for i in order]
    return pl.DataFrame(
        {
            "competition_id": [c for c, _ in rows],
            "season_id": [s for _, s in rows],
            "match_id": [f"sb:m{i % 6}" for i in range(len(rows))],  # matches span seasons
        }
    )


def _season(dev: pl.DataFrame, idx: np.ndarray) -> set[tuple[str, str]]:
    return set(dev[idx].select(BY).rows())


def test_season_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    """``N_SPLITS`` (read when called) folds; the test sets partition the rows; every season is
    in exactly one test set, whole, and never among its fold's training rows; the two seasons
    sharing ``season_id`` and the two sharing ``competition_id`` are separate groups."""
    dev = _dev()
    monkeypatch.setattr(evaluation, "N_SPLITS", 3)
    folds = evaluation.season_folds(dev, by=BY)
    assert len(folds) == 3
    evaluation.check_partition(folds, dev.height)
    tests = [np.sort(test) for _, test in folds]
    assert np.array_equal(np.sort(np.concatenate(tests)), np.arange(dev.height))
    seen: list[tuple[str, str]] = []
    for train, test in folds:
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(dev.height))
        assert not _season(dev, test) & _season(dev, train)
        seen.extend(_season(dev, test))
    assert sorted(seen) == sorted(set(dev.select(BY).rows()))  # each season once
    assert len(seen) == 5  # B18 / C18 (season_id sb:30) and A18 / A19 (sb:9) are 4 groups
    for train, test in folds:  # a season is whole in its test set
        for key in _season(dev, test):
            rows = dev.with_row_index("i").filter(
                (pl.col("competition_id") == key[0]) & (pl.col("season_id") == key[1])
            )["i"]
            assert set(rows.to_list()) <= set(test.tolist())
        assert train.dtype == np.intp and test.dtype == np.intp
    monkeypatch.setattr(evaluation, "N_SPLITS", 5)
    assert len(evaluation.season_folds(dev, by=BY)) == 5
    # match folds would mix seasons: the match_id column is not the grouping
    assert any(_season(dev, test) & _season(dev, train) for train, test in evaluation.cv_folds(dev))


def test_season_folds_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``str``; a missing column (named); a null key (counted); fewer seasons than
    ``N_SPLITS``; ``by`` positional or omitted (required keyword)."""
    dev = _dev()
    monkeypatch.setattr(evaluation, "N_SPLITS", 3)
    with pytest.raises((TypeError, ValueError)):
        evaluation.season_folds(dev, by="competition_id")
    with pytest.raises(ValueError, match="season_name"):
        evaluation.season_folds(dev, by=("competition_id", "season_name"))
    nulls = dev.with_columns(
        season_id=pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise(pl.col("season_id"))
    )
    with pytest.raises(ValueError, match=r"\b2\b"):
        evaluation.season_folds(nulls, by=BY)
    monkeypatch.setattr(evaluation, "N_SPLITS", 6)  # 5 seasons
    with pytest.raises(ValueError):
        evaluation.season_folds(dev, by=BY)
    with pytest.raises(TypeError):
        evaluation.season_folds(dev, BY)  # type: ignore[misc]
    with pytest.raises(TypeError):
        evaluation.season_folds(dev)  # type: ignore[call-arg]
