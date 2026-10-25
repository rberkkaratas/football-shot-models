"""``data.season_holdout`` (spec MS4.v1 §2.1): the WhoScored split by season name, with the date
order asserted.

The matches frame lists the test matches first and the train matches after them, with the
latest train match before the earliest one, so a date check on the first or last row (instead
of the min / max) is wrong. The last train match is one day before the first test match: the
nearest valid date order.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from football_shot_models import data

TRAIN = ("2025/2026", "2026")
TEST = ("2026/2027",)
FIRST_TEST = dt.date(2026, 8, 8)


def _matches(
    ids: Sequence[str], seasons: Sequence[str | None], dates: Sequence[dt.date | None]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "match_id": pl.Series(list(ids), dtype=pl.String()),
            "season_name": pl.Series(list(seasons), dtype=pl.String()),
            "match_date": pl.Series(list(dates), dtype=pl.Date()),
        }
    )


# test matches first, then train: the latest train match (ws:m5, 2026-08-07) is listed before
# the earliest one (ws:m3)
IDS = ("ws:m1", "ws:m2", "ws:m5", "ws:m4", "ws:m3")
SEASONS = ("2026/2027", "2026/2027", "2026", "2025/2026", "2025/2026")
DATES = (
    dt.date(2026, 9, 1), FIRST_TEST, dt.date(2026, 8, 7), dt.date(2026, 1, 3),
    dt.date(2025, 8, 9),
)  # fmt: skip
MATCHES = _matches(IDS, SEASONS, DATES)


def _with(column: str, row: int, value: object) -> pl.DataFrame:
    values = MATCHES[column].to_list()
    values[row] = value
    return MATCHES.with_columns(pl.Series(column, values, dtype=MATCHES.schema[column]))


def test_season_holdout_split() -> None:
    """§2.1: the mask holds exactly the test matches (on any frame with ``match_id``), dev is
    exactly the train seasons, and the label is the sorted test season names present, joined by
    ", "; the last train date one day before the first test date passes."""
    ho = data.season_holdout(MATCHES, TRAIN, TEST)
    assert isinstance(ho, data.Holdout)
    assert ho.label == "2026/2027"
    assert MATCHES.select(ho.expr).to_series().to_list() == [True, True, False, False, False]
    dev, held = data.split_holdout(MATCHES, ho)
    assert set(dev["season_name"]) == set(TRAIN) and set(held["season_name"]) == set(TEST)
    shots = pl.DataFrame({"match_id": ["ws:m3", "ws:m1", "ws:m5", "ws:m2", "ws:m1"]})
    assert shots.select(ho.expr).to_series().to_list() == [False, True, False, True, True]
    # two test seasons: both present, sorted ("2026/2027" < "2027"), whatever the given order
    later = MATCHES.vstack(_matches(["ws:m6"], ["2027"], [dt.date(2027, 3, 1)]))
    two = data.season_holdout(later, list(TRAIN), ["2027", "2026/2027"])
    assert two.label == "2026/2027, 2027"
    assert later.select(two.expr).to_series().to_list() == [True, True, False, False, False, True]


BAD_CASES: dict[str, tuple[pl.DataFrame, object, object, str]] = {
    "extra season": (
        MATCHES.vstack(_matches(["ws:m9"], ["2027/2028"], [dt.date(2027, 8, 1)])),
        TRAIN, TEST, r"'2027/2028'.*outside",
    ),
    "season in both": (MATCHES, TRAIN, ("2026", "2026/2027"), r"\['2026'\] are in both"),
    "bare str train": (MATCHES, "2025/2026", TEST, "non-empty sequence"),
    "bare str test": (MATCHES, TRAIN, "2026/2027", "non-empty sequence"),
    "empty train": (MATCHES, (), TEST, "non-empty sequence"),
    "empty test": (MATCHES, TRAIN, [], "non-empty sequence"),
    "null season": (_with("season_name", 3, None), TRAIN, TEST, "season_name has 1 null"),
    "null train date": (_with("match_date", 4, None), TRAIN, TEST, "match_date has 1 null"),
    "null test date": (_with("match_date", 0, None), TRAIN, TEST, "match_date has 1 null"),
    "equal dates": (
        _with("match_date", 2, FIRST_TEST), TRAIN, TEST,
        r"last train match \(2026-08-08\) is not before the first test match \(2026-08-08\)",
    ),
    "train after test": (
        _with("match_date", 3, dt.date(2026, 9, 2)), TRAIN, TEST,
        r"last train match \(2026-09-02\) is not before the first test match \(2026-08-08\)",
    ),
    "listed train season with no match": (
        MATCHES, (*TRAIN, "2024/2025"), TEST, r"train season\(s\) \['2024/2025'\] have no match",
    ),
    "listed test season with no match": (
        MATCHES, TRAIN, (*TEST, "2027"), r"test season\(s\) \['2027'\] have no match",
    ),
    "no match at all": (MATCHES.head(0), TRAIN, TEST, "have no match"),
}  # fmt: skip


@pytest.mark.parametrize("case", list(BAD_CASES))
def test_season_holdout_refuses(case: str) -> None:
    """§2.1, §8: each refusal is a ``ValueError`` with its message; the null date is caught
    before the date arithmetic."""
    matches, train, test, message = BAD_CASES[case]
    with pytest.raises(ValueError, match=message):
        data.season_holdout(matches, train, test)  # type: ignore[arg-type]


@settings(deadline=None, max_examples=40)
@given(
    st.lists(st.sampled_from(TRAIN), min_size=2, max_size=8),
    st.lists(st.integers(1, 200), min_size=2, max_size=8),
    st.integers(0, 100),
)
def test_season_holdout_partition(train: list[str], test_days: list[int], gap: int) -> None:
    """Property: with every train date before every test date, each match is in exactly one
    side, by its season name, in any row order."""
    train = [*train, *TRAIN]  # both train seasons present
    n_train, n_test = len(train), len(test_days)
    start = FIRST_TEST + dt.timedelta(days=gap)
    dates = [FIRST_TEST - dt.timedelta(days=1 + 3 * i) for i in range(n_train)]
    dates += [start + dt.timedelta(days=d) for d in test_days]
    seasons = [*train, *(["2026/2027"] * n_test)]
    ids = [f"ws:{i}" for i in range(n_train + n_test)]
    order = sorted(range(len(ids)), key=lambda i: (i * 7919) % len(ids))  # a fixed shuffle
    matches = _matches([ids[i] for i in order], [seasons[i] for i in order],
                       [dates[i] for i in order])  # fmt: skip
    mask = matches.select(data.season_holdout(matches, TRAIN, TEST).expr).to_series()
    assert mask.to_list() == [s == "2026/2027" for s in matches["season_name"]]
