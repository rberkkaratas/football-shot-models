"""Hand-built synthetic frames for the MS6 gsax tests (spec ``runs/MS6/gsax`` §4, §8).

Every id, name and number is invented. Frames are built column by column with the dtypes of
``xgot.predictions`` plus the four columns ``season_out.season_out`` adds (``competition_name``,
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
    # PR1 keeper-fixes census (``census``): C19 follows C18; T18 / T19 are team-selected and share
    # their competition, with season ids "sb:3" / "sb:4" (near misses of B18's "sb:30"); H15 is a
    # hold-out season (source ``holdout_fit``).
    "C19": ("sb:16", "sb:31", "Charlie Cup", "2019"),
    "T18": ("sb:43", "sb:3", "Tango Cup", "2018"),
    "T19": ("sb:43", "sb:4", "Tango Cup", "2019"),
    "H15": ("sb:2", "sb:27", "Hotel League", "2015/2016"),
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
    source: Sequence[str] | None = None,
) -> pl.DataFrame:
    """A season-out predictions frame (``gsax.rate``'s input), one row per list position.

    Defaults: open play; ``open_goal`` False; ``shot_id`` ``sb:s0000``...; a match's date is 1
    September of its season's start year plus a week per match in order of first appearance;
    ``period`` 1, ``seconds_in_period`` 30 s per row; family "league or tournament"; source
    ``season_folds`` (``season_out.SOURCE``, PR1 keeper-fixes §2).
    """
    from football_shot_models import season_out as so
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
            "source": pl.Series(list(source) if source else ["season_folds"] * n, dtype=so.SOURCE),
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
        zero_family=pl.lit(False, dtype=pl.Boolean()),
        pool_small=pl.lit(False, dtype=pl.Boolean()),
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
    """``TABLE_ROWS`` as a predictions frame (rows out of key order, keepers unsorted). B18 is a
    ``holdout_fit`` season and A18 a ``season_folds`` one, so each penalty pool (one δ per
    source, PR1 keeper-fixes §3.1) holds one season's kicks and the hand-worked values hold."""
    cols = list(zip(*TABLE_ROWS, strict=True))
    return frame(
        gk=cols[0], season=cols[1], match=cols[2], goal=cols[3], xgot=cols[4], kind=cols[5],
        open_goal=cols[6],
        source=["holdout_fit" if s == "B18" else "season_folds" for s in cols[1]],
    )  # fmt: skip


# ---- PR1 keeper-fixes census (spec ``runs/PR1/keeper-fixes`` §9, "Integration fixtures") -------

TEAM = "team-selected"
# Every branch the GSAx wiring feeds, each at least once (``test_gsax_fit::test_census_reached``):
#   A18 -> A19 -> A20 (sb:9): sb:g01 paired twice (kept pairs; 120 shots in A18, the one
#     keeper-season >= TABLE_MIN_SHOTS); sb:g02 concedes 0 in A18 (a left-out pair into A19).
#   A19: test season with a fitted leave-out δ (remainder sb:g02, g05-g07: 4 keepers, 110 rows).
#   A20: sb:g01 alone (E2-small, 1 keeper); as a test season its remainder is empty (δ_loo NaN).
#   C18 -> C19 (sb:16): sb:g21 and sb:g22 paired; C19 has 4 keepers and 60 rows (not small) but
#     its remainder sb:g24, sb:g25 is 2 keepers (``loo_small``).
#   T18 -> T19 (sb:43, team-selected): T18 sb:g31 + sb:g32 (team-selected and E2-small); T19
#     4 keepers, 100 rows (team-selected, not small; sb:g31 the selected side, 40 shots): as a
#     test season it is AT_ZERO by its family alone, its remainder (3 keepers, 60 rows) would be
#     fitted.
#   F1: 2 keepers (E2-small, not team-selected). H15: source ``holdout_fit``, 4 keepers.
#   Kicks: A18 + A19 6 keepers, 60 kicks (the fitted ``season_folds`` pool); H15 2 keepers
#     (an E2-small ``holdout_fit`` pool); T18, A20 and F1 kicks sit in zero seasons (δ = 0).
CENSUS_OPEN: tuple[tuple[str, str, int, int, tuple[float, float], int], ...] = (
    # gk, season, shots, a goal every k-th shot (0: none), xGOT range, matches
    ("sb:g01", "A18", 120, 4, (0.08, 0.62), 6),
    ("sb:g02", "A18", 30, 0, (0.10, 0.50), 2),
    ("sb:g03", "A18", 30, 3, (0.10, 0.60), 2),
    ("sb:g04", "A18", 30, 5, (0.05, 0.45), 2),
    ("sb:g01", "A19", 40, 3, (0.10, 0.60), 4),
    ("sb:g02", "A19", 20, 4, (0.10, 0.50), 2),
    ("sb:g05", "A19", 30, 3, (0.15, 0.55), 2),
    ("sb:g06", "A19", 30, 4, (0.10, 0.50), 2),
    ("sb:g07", "A19", 30, 6, (0.05, 0.40), 2),
    ("sb:g01", "A20", 30, 3, (0.10, 0.60), 2),
    ("sb:g21", "C18", 20, 3, (0.10, 0.60), 2),
    ("sb:g22", "C18", 20, 4, (0.10, 0.50), 2),
    ("sb:g23", "C18", 20, 5, (0.20, 0.60), 2),
    ("sb:g21", "C19", 15, 3, (0.10, 0.60), 2),
    ("sb:g22", "C19", 15, 4, (0.10, 0.50), 2),
    ("sb:g24", "C19", 15, 3, (0.15, 0.65), 2),
    ("sb:g25", "C19", 15, 2, (0.10, 0.40), 2),
    ("sb:g31", "T18", 40, 5, (0.10, 0.60), 4),
    ("sb:g32", "T18", 15, 2, (0.20, 0.70), 2),
    ("sb:g31", "T19", 40, 4, (0.10, 0.60), 4),
    ("sb:g35", "T19", 20, 2, (0.10, 0.40), 2),
    ("sb:g36", "T19", 20, 2, (0.10, 0.40), 2),
    ("sb:g37", "T19", 20, 3, (0.10, 0.40), 2),
    ("sb:gf1", "F1", 10, 3, (0.20, 0.60), 1),
    ("sb:gf2", "F1", 10, 4, (0.20, 0.60), 1),
    ("sb:g41", "H15", 15, 3, (0.10, 0.60), 2),
    ("sb:g42", "H15", 15, 4, (0.10, 0.60), 2),
    ("sb:g43", "H15", 15, 3, (0.10, 0.50), 2),
    ("sb:g44", "H15", 15, 5, (0.10, 0.50), 2),
)
CENSUS_KICKS: tuple[tuple[str, str, int, int], ...] = (
    # gk, season, kicks, a miss every k-th kick
    ("sb:g01", "A18", 10, 4), ("sb:g03", "A18", 10, 3), ("sb:g04", "A18", 10, 5),
    ("sb:g05", "A19", 10, 4), ("sb:g06", "A19", 10, 3), ("sb:g07", "A19", 10, 6),
    ("sb:g32", "T18", 3, 3), ("sb:g01", "A20", 2, 2), ("sb:gf1", "F1", 2, 2),
    ("sb:g41", "H15", 4, 2), ("sb:g42", "H15", 4, 3),
)  # fmt: skip
CENSUS_FAMILY: dict[str, str] = {"T18": TEAM, "T19": TEAM}
CENSUS_SOURCE: dict[str, str] = {"H15": "holdout_fit"}
# The ``trained`` frame (``season_out.fold_seasons`` shape): rated season -> its training
# seasons. Kept pairs: g01 A18->A19 (A19 trained: seen), g01 A19->A20 (unseen: A19's list holds
# (sb:16, sb:108), A20's season id in another competition), g21 / g22 C18->C19 and g31
# T18->T19 (unseen). The left-out pair g02 A18->A19 would count as seen: counting left-out
# pairs gives 2.
CENSUS_TRAINED: dict[str, tuple[tuple[str, str], ...]] = {
    "A18": (SEASONS["A19"][:2], SEASONS["C18"][:2]),
    "A19": (SEASONS["A18"][:2], ("sb:16", "sb:108")),
    "A20": (SEASONS["A18"][:2],),
    "C18": (SEASONS["A18"][:2],),
    "C19": (SEASONS["A18"][:2],),
    "T18": (SEASONS["A18"][:2],),
    "T19": (SEASONS["A18"][:2],),
    "F1": (SEASONS["A18"][:2],),
    "H15": (SEASONS["A18"][:2], SEASONS["A19"][:2]),
}
CENSUS_SEEN, CENSUS_KEPT, CENSUS_LEFT_OUT = 1, 5, 1


def census_frame() -> pl.DataFrame:
    """``CENSUS_OPEN`` and ``CENSUS_KICKS`` as a season-out predictions frame (rows out of key
    order: shot ids run backwards), with ``CENSUS_FAMILY`` and ``CENSUS_SOURCE`` per season."""
    parts = []
    for i, (gk, season, n, every, (lo, hi), m) in enumerate(CENSUS_OPEN):
        goals = [False] * n if every == 0 else pattern(n, every, i % every)
        tag = season.lower()
        parts.append(block(gk, season, goals=goals, xgot=spread(n, lo, hi),
                           matches=[f"sb:m{tag}{gk[-2:]}{j}" for j in range(m)]))  # fmt: skip
    for gk, season, n, every in CENSUS_KICKS:
        goals = [not v for v in pattern(n, every, 1)]
        match = f"sb:m{season.lower()}{gk[-2:]}0"
        parts.append(block(gk, season, goals=goals, xgot=spread(n, 0.70, 0.82), matches=[match],
                           kind="penalty"))  # fmt: skip
    seasons = [s for part in parts for s in part["season"]]
    return blocks(
        *parts,
        family=[CENSUS_FAMILY.get(s, LEAGUE) for s in seasons],
        source=[CENSUS_SOURCE.get(s, "season_folds") for s in seasons],
    )


def census_stops(pred: pl.DataFrame, line_x: float) -> pl.DataFrame:
    """``shot_id``, ``end_x`` of every open-play row: goals at 105.0; per keeper-season the
    saves alternate between 104.5 (near) and 100.0 (far) with a far share that differs by
    keeper; the first save of sb:g01 A18 sits exactly on ``line_x`` and the first save of
    sb:g03 A18 one float below it (``np.nextafter``)."""
    open_ = pred.filter(pl.col("kind").cast(pl.String) == "open_play").sort("shot_id")
    below = float(np.nextafter(line_x, 0.0))
    out, seen = [], {}
    for sid, gk, season, goal in open_.select("shot_id", "gk_id", "season_id", "is_goal").rows():
        if goal:
            out.append((sid, 105.0))
            continue
        k = seen.setdefault((gk, season), 0)
        seen[(gk, season)] = k + 1
        if k == 0 and (gk, season) == ("sb:g01", "sb:42"):
            out.append((sid, line_x))
        elif k == 0 and (gk, season) == ("sb:g03", "sb:42"):
            out.append((sid, below))
        else:
            every = 2 + int(gk[-1]) % 4  # 1 in 2 .. 1 in 5 saves stopped far from the line
            out.append((sid, 100.0 if k % every == 0 else 104.5))
    return pl.DataFrame(out, schema={"shot_id": pl.String, "end_x": pl.Float64}, orient="row")


def census_trained() -> pl.DataFrame:
    """``CENSUS_TRAINED`` as a ``season_out.fold_seasons`` frame (sorted, String columns)."""
    rows = [(*SEASONS[s][:2], *t) for s, trained in CENSUS_TRAINED.items() for t in trained]
    cols = ("competition_id", "season_id", "trained_competition_id", "trained_season_id")
    return pl.DataFrame(rows, schema=dict.fromkeys(cols, pl.String), orient="row").sort(cols)


def census_geometry(pred: pl.DataFrame) -> pl.DataFrame:
    """Placement inputs for every row (``gsax.placement_bands`` reads them from
    ``f.inp.on_dev``): inside the frame, varied by row."""
    ids = pred["shot_id"].sort().to_list()
    n = len(ids)
    return pl.DataFrame({
        "shot_id": ids,
        "goal_y": [31.0 + (i % 7) for i in range(n)],
        "goal_z": [0.2 + 0.4 * (i % 5) for i in range(n)],
        "on_target": [True] * n,
        "angle_rad": [0.3 + 0.15 * (i % 6) for i in range(n)],
    })  # fmt: skip
