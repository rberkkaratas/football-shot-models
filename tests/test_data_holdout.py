"""The hold-out seam (spec MS4.prep §3, finding A2): one ``Holdout`` value (expression plus label)
read by ``split_holdout``, ``penalties.penalty_rows`` and ``cards.card_context``, and
``holdout_mask``, which turns every polars failure of the expression into a ``ValueError``.

Fixtures are the hand-built ``xg_cases`` tables and ``models_cases.penalties_frame``. Near misses
sit on the key: a ``match_id`` hold-out whose matches share their competition-season with dev
matches, and matches of the hold-out competition (or season) listed before the hold-out pair.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

import models_cases as mc
import polars as pl
import pytest
import xg_cases as xc

from football_shot_models import data

PL_LABEL = "Premier League 2015/2016"


def _matches_with_near_misses() -> pl.DataFrame:
    """The fixture matches, preceded by a match of the hold-out competition in another season and
    one of another competition in the hold-out season: a rule on either id alone, or a label
    taken from the first match whose competition matches, prints the wrong name."""
    _, _, matches = xc.fixture_tables()
    extra = xc.matches_frame(
        [
            xc.Match("sb:m00a", "sb:2", "sb:44", "sb:t90", "sb:t91", competition_name="Premier "
                     "League", season_name="2016/2017", match_date=dt.date(2016, 9, 1)),
            xc.Match("sb:m00b", "sb:11", "sb:27", "sb:t92", "sb:t93", competition_name="La Liga",
                     season_name="2015/2016", match_date=dt.date(2015, 9, 1)),
        ]
    )  # fmt: skip
    return pl.concat([extra, matches])


def test_pair_holdout_label() -> None:
    """§3: the label is "<competition_name> <season_name>" of the first match of the pair (both
    ids), "n/a" for a pair with no match; the expression is ``is_holdout(pair)``; ``Holdout`` is
    frozen and compares by identity (``eq=False``: ``==`` on a ``pl.Expr`` would raise)."""
    matches = _matches_with_near_misses()
    assert matches["match_id"][0] == "sb:m00a"  # the near misses come first
    ho = data.pair_holdout(matches)
    assert isinstance(ho, data.Holdout)
    assert ho.label == PL_LABEL
    mask = data.holdout_mask(matches, ho, "matches")
    assert mask.dtype == pl.Boolean() and mask.len() == matches.height
    assert mask.to_list() == matches.select(data.is_holdout()).to_series().to_list()
    assert int(mask.sum()) == 2  # sb:m01, sb:m02; neither near miss
    assert data.pair_holdout(matches, ("sb:11", "sb:27")).label == "La Liga 2015/2016"
    assert data.pair_holdout(matches, ("sb:99", "sb:5")).label == "n/a"
    assert data.pair_holdout(matches, ("sb:2", "sb:90")).label == "n/a"  # each id exists alone
    again = data.pair_holdout(matches)
    assert ho == ho and ho != again and isinstance(hash(ho), int)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ho.label = "other"  # type: ignore[misc]


def _null_key(frame: pl.DataFrame) -> pl.DataFrame:
    """``competition_id`` null on the first row of the hold-out season: there the mask is
    ``null & True``, i.e. null (on another season ``null & False`` is False, no null)."""
    row = frame.with_row_index().filter(pl.col("season_id") == "sb:27")["index"][0]
    return mc.with_null(frame, "competition_id", int(row))


# case: (frame edit, expression or None for the PL pair, message part)
REFUSALS: dict[str, tuple[Any, Any, str]] = {
    "missing column": (lambda f: f.drop("competition_id"), None, "a column the frame lacks"),
    "non-Boolean": (lambda f: f, pl.col("season_id"), "not Boolean"),
    "null mask": (_null_key, None, "the hold-out mask has 1 null value(s)"),
    "length-changing": (
        lambda f: f,
        pl.col("match_id").head(1).append(pl.col("match_id")),
        "the hold-out expression fails",
    ),
    "dtype mismatch": (lambda f: f, pl.col("match_id") > 3, "the hold-out expression fails"),
}


@pytest.mark.parametrize("what", ["shots", "penalties"])
@pytest.mark.parametrize("case", list(REFUSALS))
def test_holdout_mask_refuses(case: str, what: str) -> None:
    """§3, §6: a column the frame lacks, a non-Boolean mask, a null in the mask, a
    length-changing expression (``ShapeError``) and a dtype mismatch (``ComputeError``) are each
    a plain ``ValueError`` starting ``"{what}: "``, never a polars exception; the same
    expression on the unedited frame is a valid mask (so the edit is what fails)."""
    edit, expr, part = REFUSALS[case]
    frame = xc.fixture_tables()[0] if what == "shots" else mc.penalties_frame()
    holdout = data.pair_holdout(xc.fixture_tables()[2]) if expr is None else expr
    if case in ("missing column", "null mask"):
        ok = data.holdout_mask(frame, holdout, what)
        assert ok.dtype == pl.Boolean() and ok.len() == frame.height and ok.null_count() == 0
    with pytest.raises(ValueError) as info:
        data.holdout_mask(edit(frame), holdout, what)
    assert type(info.value) is ValueError
    assert not isinstance(info.value, pl.exceptions.PolarsError)
    assert str(info.value).startswith(f"{what}: "), str(info.value)
    assert part in str(info.value)


def test_split_holdout_with_holdout() -> None:
    """§3.1: a ``Holdout`` on ``match_id`` (the M1.6 route; shots have no ``season_name``) splits
    by match, so two matches of a competition-season are held out and its other two stay in dev,
    each side in input order; a tuple still means the pair; a literal broadcasts; a null key now
    raises instead of dropping the row; an empty side names the label; a match-level column is
    "a column the frame lacks"."""
    shots = xc.fixture_tables()[0].sort("shot_id", descending=True)
    assert "season_name" not in shots.columns
    test_ids = pl.Series(["sb:m03", "sb:m04"])
    ho = data.Holdout(pl.col("match_id").is_in(test_ids.implode()), "Fixture League test")
    dev, held = data.split_holdout(shots, ho)
    in_test = pl.col("match_id").is_in(["sb:m03", "sb:m04"])
    assert held.equals(shots.filter(in_test)) and dev.equals(shots.filter(~in_test))
    assert set(held["match_id"]) == {"sb:m03", "sb:m04"}
    same_pair = dev.filter((pl.col("competition_id") == "sb:11") & (pl.col("season_id") == "sb:4"))
    assert set(same_pair["match_id"]) == {"sb:m05", "sb:m06"}  # the near miss stays in dev
    pair_dev, pair_held = data.split_holdout(shots, ("sb:2", "sb:27"))
    default_dev, default_held = data.split_holdout(shots)
    assert pair_held.equals(default_held) and pair_dev.equals(default_dev)
    assert pair_held.height == 20
    with pytest.raises(ValueError, match="the hold-out mask has 1 null value"):
        data.split_holdout(_null_key(shots))
    with pytest.raises(ValueError, match=r"^hold-out Nowhere 2099: the hold-out set is empty$"):
        data.split_holdout(shots, data.Holdout(pl.lit(False), "Nowhere 2099"))
    with pytest.raises(ValueError, match=r"^hold-out Everything: the development set is empty$"):
        data.split_holdout(shots, data.Holdout(pl.lit(True), "Everything"))
    by_season = data.Holdout(pl.col("season_name") == "2015/2016", PL_LABEL)
    with pytest.raises(ValueError, match="^shots: .*a column the frame lacks") as info:
        data.split_holdout(shots, by_season)
    assert type(info.value) is ValueError
    with pytest.raises(ValueError, match="hold-out set is empty"):
        data.split_holdout(shots.head(0), ho)  # an empty frame is an empty mask, then "empty"


def test_penalty_rows_holdout_object() -> None:
    """§3.2: ``penalty_rows`` reads a ``Holdout`` exactly as its expression (the custom split's
    own rows, not the default pair), and a ``season_name`` expression on penalties (no such
    column) is a ``ValueError``, not ``ColumnNotFoundError``."""
    from football_shot_models import penalties

    frame = mc.penalties_frame()
    ho = data.Holdout(pl.col("season_id") == "sb:44", "Fixture season 44")
    by_object = penalties.penalty_rows(frame, holdout=ho)
    assert by_object == penalties.penalty_rows(frame, holdout=ho.expr)
    keys = [(r.rate.n, r.rate.goals, r.matches) for r in by_object]
    assert keys[1:3] == [(9, 6, 5), (1, 1, 1)]  # not the default pair's (7, 5, 4), (3, 2, 2)
    pair = data.pair_holdout(xc.fixture_tables()[2])
    assert penalties.penalty_rows(frame, holdout=pair) == penalties.penalty_rows(frame)
    by_season = data.Holdout(pl.col("season_name") == "2015/2016", PL_LABEL)
    with pytest.raises(ValueError, match="^penalties: .*a column the frame lacks") as info:
        penalties.penalty_rows(frame, holdout=by_season)
    assert type(info.value) is ValueError


def test_card_context_follows_holdout(xg_data: Any) -> None:
    """§3.3: the label comes from the ``Holdout`` passed in (default: ``pair_holdout`` of the
    matches); a dev/held pair that does not follow it raises: swapped sides, one held row in dev,
    one dev row in held."""
    from football_shot_models import cards

    xg = xg_data
    ho = data.pair_holdout(xg.matches)
    dev, held = data.split_holdout(xg.shots, ho)
    assert cards.card_context(xg, dev, held, ho).holdout_label == PL_LABEL
    assert cards.card_context(xg, dev, held).holdout_label == PL_LABEL
    custom = data.Holdout(ho.expr, "Custom hold-out 1999")
    context = cards.card_context(xg, dev, held, custom)
    assert context.holdout_label == "Custom hold-out 1999"
    assert (context.dev_shots, context.holdout_shots) == (dev.height, held.height)
    bad = {
        "swapped": (held, dev),
        "held row in dev": (pl.concat([dev, held.head(1)]), held.slice(1)),
        "dev row in held": (dev.slice(1), pl.concat([held, dev.head(1)])),
    }
    for label, (d, h) in bad.items():
        with pytest.raises(ValueError):
            cards.card_context(xg, d, h, ho)
        assert d.height and h.height, label
