"""Hand-built synthetic frames for the MS6 gsax tests (spec ``runs/MS6/gsax`` §4, §8).

Every id, name and number is invented. Frames are built column by column with the dtypes of
``xgot.predictions`` plus the four columns ``xgot.season_out`` adds (``competition_name``,
``season_name``, ``family``, ``source``). Nothing here imports ``football_shot_models.gsax`` at
module level, so a missing module fails the tests that need it, not the collection.

``SEASONS`` holds near misses on the composite key: ``A18`` / ``A19`` / ``A20`` share their
``competition_id``; ``B18`` and ``C18`` share their ``season_id`` (``sb:30``) in two
competitions. Season ids do not sort in season order (``sb:108`` < ``sb:42`` < ``sb:90``), so a
rule that orders seasons by id instead of ``season_start`` is caught.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from football_core import schema

from football_shot_models import data

SEASONS: dict[str, tuple[str, str, str, str]] = {
    # key: (competition_id, season_id, competition_name, season_name)
    "A18": ("sb:9", "sb:42", "Alpha League", "2018/2019"),
    "A19": ("sb:9", "sb:90", "Alpha League", "2019/2020"),
    "A20": ("sb:9", "sb:108", "Alpha League", "2020/2021"),
    "B18": ("sb:72", "sb:30", "Bravo Cup", "2018"),
    "B19": ("sb:72", "sb:107", "Bravo Cup", "2019"),
    "C18": ("sb:16", "sb:30", "Charlie Cup", "2018"),
    "F1": ("sb:55", "sb:43", "Final Cup", "2020"),
}
LEAGUE = "league or tournament"


def _start(season: str) -> int:
    return int(SEASONS[season][3][:4])


def frame(
    *,
    gk: Sequence[str],
    season: Sequence[str],
    match: Sequence[str],
    goal: Sequence[bool],
    xgot: Sequence[float],
    kind: Sequence[str] | None = None,
    open_goal: Sequence[bool | None] | None = None,
    shot_id: Sequence[str] | None = None,
    date: Sequence[dt.date] | None = None,
    period: Sequence[int] | None = None,
    seconds: Sequence[float] | None = None,
    family: Sequence[str] | None = None,
) -> pl.DataFrame:
    """A season-out predictions frame (``gsax.rate``'s input), one row per list position.

    Defaults: open play; ``open_goal`` False; ``shot_id`` ``sb:s0000``...; a match's date is 1
    September of its season's start year plus a week per match in order of first appearance;
    ``period`` 1, ``seconds_in_period`` 30 s per row; family "league or tournament".
    """
    from football_shot_models import xgot as xg

    n = len(gk)
    for name, col in (("season", season), ("match", match), ("goal", goal), ("xgot", xgot)):
        assert len(col) == n, name
    kinds = list(kind) if kind is not None else ["open_play"] * n
    if date is None:
        rank: dict[str, int] = {}
        for m in match:
            rank.setdefault(m, len(rank))
        date = [
            dt.date(_start(s), 9, 1) + dt.timedelta(days=7 * rank[m])
            for s, m in zip(season, match, strict=True)
        ]
    goals = [bool(g) for g in goal]
    return pl.DataFrame(
        {
            "shot_id": list(shot_id) if shot_id is not None else [f"sb:s{i:04d}" for i in range(n)],
            "match_id": list(match),
            "player_id": ["sb:p01"] * n,
            "competition_id": [SEASONS[s][0] for s in season],
            "season_id": [SEASONS[s][1] for s in season],
            "match_date": pl.Series(list(date), dtype=pl.Date()),
            "period": pl.Series(list(period) if period is not None else [1] * n, dtype=pl.Int8()),
            "seconds_in_period": pl.Series(
                list(seconds) if seconds is not None else [30.0 * i for i in range(n)],
                dtype=pl.Float64(),
            ),
            "gk_id": pl.Series(list(gk), dtype=pl.String()),
            "kind": pl.Series(kinds, dtype=xg.KIND),
            "split": pl.Series(["dev"] * n, dtype=xg.SPLIT),
            "is_goal": pl.Series(goals, dtype=pl.Boolean()),
            "on_target": pl.Series([True] * n, dtype=pl.Boolean()),
            "outcome": pl.Series(["goal" if g else "saved" for g in goals], dtype=schema.OUTCOME),
            "open_goal": pl.Series(
                list(open_goal) if open_goal is not None else [False] * n, dtype=pl.Boolean()
            ),
            "xg": pl.Series([0.1 if k == "open_play" else None for k in kinds], dtype=pl.Float64()),
            "xgot": pl.Series(list(xgot), dtype=pl.Float64()),
            "competition_name": [SEASONS[s][2] for s in season],
            "season_name": [SEASONS[s][3] for s in season],
            "family": pl.Series(list(family) if family else [LEAGUE] * n, dtype=data.FAMILY),
            "source": pl.Series(["season_folds"] * n, dtype=xg.SOURCE),
        }
    )


def block(
    gk: str,
    season: str,
    *,
    goals: Sequence[bool],
    xgot: Sequence[float] | float,
    matches: Sequence[str],
    kind: str = "open_play",
) -> dict[str, list[Any]]:
    """Columns for one keeper's rows in one season (``frame`` keywords); ``matches`` cycles."""
    n = len(goals)
    p = [float(xgot)] * n if isinstance(xgot, float | int) else [float(v) for v in xgot]
    return {
        "gk": [gk] * n,
        "season": [season] * n,
        "match": [matches[i % len(matches)] for i in range(n)],
        "goal": list(goals),
        "xgot": p,
        "kind": [kind] * n,
    }


def blocks(*parts: dict[str, list[Any]], **extra: Any) -> pl.DataFrame:
    """``frame`` over the concatenated ``block`` columns; shot ids are unique and run backwards
    (so ``shot_id`` order is not row order)."""
    cols: dict[str, list[Any]] = {k: [] for k in ("gk", "season", "match", "goal", "xgot", "kind")}
    for part in parts:
        for k in cols:
            cols[k].extend(part[k])
    n = len(cols["gk"])
    ids = [f"sb:s{n - i:05d}" for i in range(n)]
    return frame(**cols, shot_id=ids, **extra)


def rated(f: pl.DataFrame) -> pl.DataFrame:
    """``f`` as rated at δ = 0 (``q`` = ``xgot``): the model, reliability and next-season inputs
    when the test controls ``q`` directly."""
    return f.with_columns(
        delta=pl.lit(0.0, dtype=pl.Float64()),
        q=pl.col("xgot"),
        small=pl.lit(False, dtype=pl.Boolean()),
    )


def pattern(n: int, every: int, offset: int = 0) -> list[bool]:
    """``n`` outcomes, a goal on every ``every``-th row starting at ``offset``."""
    return [(i + offset) % every == 0 for i in range(n)]


def spread(n: int, lo: float, hi: float, step: int = 7) -> list[float]:
    """``n`` xGOT values in [lo, hi], visited in a scrambled but fixed order."""
    m = max(n, 2)
    return [lo + (hi - lo) * ((i * step) % m) / (m - 1) for i in range(n)]


def logit(p: np.ndarray) -> np.ndarray:
    from football_shot_models import metrics

    out: np.ndarray = metrics.logit(np.asarray(p, dtype=np.float64))
    return out


# ---- the hand-worked table (test_gsax_table, test_gsax_rating) -------------------------------

# A18 (xGOT 0.3, 12 open-play shots, 5 goals: q = 5/12) and B18 (xGOT 0.25, 8 shots, 3 goals:
# q = 3/8, open_goal not recorded). Penalties: A18 all 3 scored (delta +inf, q = goal); B18 2 of
# 3 at xGOT 0.75 (q = 2/3). sb:g07 faces no penalty; sb:g12 keeps goal in both seasons.
TABLE_ROWS: tuple[tuple[str, str, str, bool, float, str, bool | None], ...] = (
    # gk, season, match, goal, xgot, kind, open_goal
    ("sb:g30", "A18", "sb:m01", True, 0.3, "open_play", True),
    ("sb:g30", "A18", "sb:m01", False, 0.3, "open_play", False),
    ("sb:g07", "A18", "sb:m03", False, 0.3, "open_play", False),
    ("sb:g30", "A18", "sb:m02", True, 0.3, "open_play", False),
    ("sb:g12", "A18", "sb:m04", True, 0.3, "open_play", False),
    ("sb:g30", "A18", "sb:m02", False, 0.3, "open_play", False),
    ("sb:g12", "B18", "sb:m06", True, 0.25, "open_play", None),
    ("sb:g07", "A18", "sb:m03", False, 0.3, "open_play", False),
    ("sb:g30", "A18", "sb:m02", False, 0.3, "open_play", False),
    ("sb:g12", "A18", "sb:m05", True, 0.3, "open_play", True),
    ("sb:g44", "B18", "sb:m08", True, 0.75, "penalty", None),
    ("sb:g12", "A18", "sb:m04", True, 0.3, "open_play", False),
    ("sb:g07", "A18", "sb:m03", False, 0.3, "open_play", False),
    ("sb:g12", "A18", "sb:m05", False, 0.3, "open_play", False),
    ("sb:g30", "A18", "sb:m02", True, 0.8, "penalty", False),
    ("sb:g12", "B18", "sb:m07", False, 0.25, "open_play", None),
    ("sb:g44", "B18", "sb:m08", True, 0.25, "open_play", None),
    ("sb:g12", "B18", "sb:m06", False, 0.25, "open_play", None),
    ("sb:g30", "A18", "sb:m01", True, 0.8, "penalty", False),
    ("sb:g12", "B18", "sb:m07", False, 0.25, "open_play", None),
    ("sb:g44", "B18", "sb:m08", True, 0.25, "open_play", None),
    ("sb:g12", "B18", "sb:m06", False, 0.25, "open_play", None),
    ("sb:g44", "B18", "sb:m08", False, 0.75, "penalty", None),
    ("sb:g12", "A18", "sb:m04", True, 0.8, "penalty", False),
    ("sb:g12", "B18", "sb:m07", False, 0.25, "open_play", None),
    ("sb:g12", "B18", "sb:m06", True, 0.75, "penalty", None),
)


def table_frame() -> pl.DataFrame:
    """``TABLE_ROWS`` as a predictions frame (rows out of key order, keepers unsorted)."""
    cols = list(zip(*TABLE_ROWS, strict=True))
    return frame(
        gk=cols[0], season=cols[1], match=cols[2], goal=cols[3], xgot=cols[4], kind=cols[5],
        open_goal=cols[6],
    )  # fmt: skip
