"""Hand-written synthetic fixtures for the M1.0 / M1.1 tests (spec ``runs/MS2/m1`` §8).

Every id, name and number is invented (``sb:m01``, teams ``sb:t01``...); nothing is copied from
an open-data file. Frames are built column by column with the C3 dtypes, never from row dicts.

The fixture snapshot has 14 matches in four competition-seasons: the hold-out ``sb:2`` /
``sb:27`` (m01-m02, 10 shots and 4 goals each; two teams per match, so each plays exactly
50 % and the family rule calls it team-selected), a league (m03-m06), a team-selected season
with ``sb:t13`` in every match (m07-m10), both 12 shots and 4 goals per match, and a women's
league (m11-m14, 6 shots and no goal each: a one-class segment). Plus four excluded shots, one
per reason: an own goal (m03), an in-game penalty (m04), a shot without a freeze frame (m05)
and a shoot-out kick (m07). No shot uses ``body_part=other``, so that segment is empty. The
women's matches are the smallest groups, so ``GroupKFold(5)`` puts at least one men's match
(with goals) in every test fold.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
from football_core import pitch, schema
from football_core.io import snapshot

SNAPSHOT = "statsbomb-open-v1"
HOLDOUT = ("sb:2", "sb:27")
SOURCE = snapshot.source_from_git("https://example.invalid/open-data", "0" * 40)
PERIOD_START = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}


@dataclass(frozen=True)
class Shot:
    """One synthetic shot; ``shots_frame`` turns a list of them into C3 columns."""

    shot_id: str
    match_id: str = "sb:m00"
    competition_id: str = "sb:11"
    season_id: str = "sb:4"
    team_id: str = "sb:t01"
    opponent_team_id: str = "sb:t02"
    is_home: bool = True
    x: float = 95.0
    y: float = 34.0
    body_part: str | None = "right_foot"
    goal: bool = False
    period: int = 1
    minute: int = 10
    penalty: bool = False
    own_goal: bool = False
    provider_xg: float | None = 0.1
    match_date: dt.date = dt.date(2019, 1, 1)
    situation: str | None = None  # None: "penalty" if penalty else "open_play"


@dataclass(frozen=True)
class Match:
    match_id: str
    competition_id: str
    season_id: str
    home: str
    away: str
    gender: str = "male"
    competition_name: str = "Fixture League"
    season_name: str = "2018/2019"
    match_date: dt.date = dt.date(2019, 1, 1)


def shots_frame(shots: Sequence[Shot]) -> pl.DataFrame:
    """A C3-valid shots frame: distance and angle recomputed by ``football_core.pitch``."""
    n = len(shots)
    xs = [s.x for s in shots]
    ys = [s.y for s in shots]
    distance = [float(v) for v in pitch.shot_distance(xs, ys)] if n else []
    angle = [float(v) for v in pitch.opening_angle(xs, ys)] if n else []
    second = [0 if s.period == 5 else 30 for s in shots]
    outcome = ["goal" if s.goal else "saved" for s in shots]
    cols: dict[str, list[object]] = {
        "shot_id": [s.shot_id for s in shots],
        "provider": ["statsbomb"] * n,
        "competition_id": [s.competition_id for s in shots],
        "season_id": [s.season_id for s in shots],
        "match_id": [s.match_id for s in shots],
        "match_date": [s.match_date for s in shots],
        "team_id": [s.team_id for s in shots],
        "opponent_team_id": [s.opponent_team_id for s in shots],
        "player_id": [f"sb:p-{s.team_id[3:]}" for s in shots],
        "is_home": [s.is_home for s in shots],
        "period": [s.period for s in shots],
        "minute": [s.minute for s in shots],
        "second": second,
        "seconds_in_period": [
            float((s.minute - PERIOD_START[s.period]) * 60 + sec)
            for s, sec in zip(shots, second, strict=True)
        ],
        "x": xs,
        "y": ys,
        **{c: [None] * n for c in ("end_x", "end_y", "end_z", "goal_y", "goal_z", "gk_x", "gk_y")},
        "distance_m": distance,
        "angle_rad": angle,
        "visible_area": [None] * n,
        "body_part": [None if s.own_goal else s.body_part for s in shots],
        "situation": [s.situation or ("penalty" if s.penalty else "open_play") for s in shots],
        "assist_type": [None] * n,
        "technique": [None] * n,
        "outcome": outcome,
        "is_goal": [s.goal for s in shots],
        "is_own_goal": [s.own_goal for s in shots],
        "is_penalty": [s.penalty for s in shots],
        "on_target": [
            o in ("goal", "saved") and not s.own_goal for s, o in zip(shots, outcome, strict=True)
        ],
        **{c: [None] * n for c in ("first_time", "follows_dribble", "open_goal", "one_on_one")},
        "big_chance": [None] * n,
        "deflected": [None] * n,
        "provider_xg": [None if s.own_goal else s.provider_xg for s in shots],
        "provider_psxg": [None] * n,
        "assist_event_id": [None] * n,
        "raw_type": ["Own Goal Against" if s.own_goal else "Shot" for s in shots],
        "raw_qualifiers": ["{}"] * n,
        "schema_version": [schema.SCHEMA_VERSION] * n,
    }
    return pl.DataFrame(cols, schema=schema.SHOT_SCHEMA, strict=True)


def frames_for(shots: pl.DataFrame) -> pl.DataFrame:
    """One freeze-frame row (the opposing keeper) per shot in ``shots``."""
    ids = shots["shot_id"].to_list()
    keepers = [f"sb:gk-{t[3:]}" for t in shots["opponent_team_id"].to_list()]
    return pl.DataFrame(
        {
            "shot_id": ids,
            "player_id": keepers,
            "is_teammate": [False] * len(ids),
            "is_keeper": [True] * len(ids),
            "x": [104.0] * len(ids),
            "y": [34.0] * len(ids),
        },
        schema=schema.FREEZE_FRAME_SCHEMA,
        strict=True,
    )


def matches_frame(matches: Sequence[Match]) -> pl.DataFrame:
    """A canonical ``matches`` table (``schema.MATCH_SCHEMA``; scores are 0, no 360 files)."""
    n = len(matches)
    cols: dict[str, list[object]] = {
        "match_id": [m.match_id for m in matches],
        "provider": ["statsbomb"] * n,
        "competition_id": [m.competition_id for m in matches],
        "season_id": [m.season_id for m in matches],
        "match_date": [m.match_date for m in matches],
        "home_team_id": [m.home for m in matches],
        "away_team_id": [m.away for m in matches],
        "competition_name": [m.competition_name for m in matches],
        "season_name": [m.season_name for m in matches],
        "gender": [m.gender for m in matches],
        "home_score": [0] * n,
        "away_score": [0] * n,
        "has_360": [False] * n,
        "schema_version": [schema.SCHEMA_VERSION] * n,
    }
    return pl.DataFrame(cols, schema=schema.MATCH_SCHEMA, strict=True)


# --- The fixture snapshot ---------------------------------------------------------------------

# (x, y, body part, goal): goals mostly close, misses overlapping them (no separation).
_PATTERN: tuple[tuple[float, float, str, bool], ...] = (
    (99.0, 34.0, "right_foot", True),
    (97.0, 30.0, "right_foot", False),
    (94.0, 38.0, "left_foot", False),
    (88.0, 34.0, "right_foot", False),
    (100.0, 36.0, "left_foot", True),
    (90.0, 28.0, "left_foot", False),
    (101.0, 33.0, "head", True),
    (98.0, 35.0, "head", False),
    (85.0, 40.0, "right_foot", False),
    (96.0, 33.0, "right_foot", True),
    (92.0, 24.0, "head", False),
    (80.0, 30.0, "left_foot", False),
)


_PL, _LG, _TS, _WM = (
    ("sb:2", "sb:27", "Premier League", "2015/2016"),
    ("sb:11", "sb:4", "Fixture League", "2018/2019"),
    ("sb:11", "sb:1", "Fixture League", "2010/2011"),
    ("sb:37", "sb:90", "Fixture Women's League", "2020/2021"),
)
# match id, competition-season, home, away; the team-selected season has sb:t13 in every match.
_MATCHES = (
    ("sb:m01", _PL, "sb:t01", "sb:t02"),
    ("sb:m02", _PL, "sb:t03", "sb:t04"),
    ("sb:m03", _LG, "sb:t05", "sb:t06"),
    ("sb:m04", _LG, "sb:t07", "sb:t08"),
    ("sb:m05", _LG, "sb:t09", "sb:t10"),
    ("sb:m06", _LG, "sb:t11", "sb:t12"),
    ("sb:m07", _TS, "sb:t13", "sb:t14"),
    ("sb:m08", _TS, "sb:t15", "sb:t13"),
    ("sb:m09", _TS, "sb:t13", "sb:t16"),
    ("sb:m10", _TS, "sb:t17", "sb:t13"),
    ("sb:m11", _WM, "sb:t18", "sb:t19"),
    ("sb:m12", _WM, "sb:t20", "sb:t21"),
    ("sb:m13", _WM, "sb:t22", "sb:t23"),
    ("sb:m14", _WM, "sb:t24", "sb:t25"),
)


def _fixture_matches() -> list[Match]:
    return [
        Match(mid, comp, season, home, away, "female" if comp == "sb:37" else "male", cname,
              sname, dt.date(2019, 1, 1) + dt.timedelta(days=7 * i))
        for i, (mid, (comp, season, cname, sname), home, away) in enumerate(_MATCHES)
    ]  # fmt: skip


def _match_shots(m: Match, k: int, n: int, *, goals: bool) -> list[Shot]:
    shots = []
    for i, (x, y, body, goal) in enumerate(_PATTERN[:n]):
        home = i % 2 == 0
        px, py = x - 0.2 * (k % 4), y + 0.3 * (k % 5)
        d = float(pitch.shot_distance([px], [py])[0])
        shots.append(Shot(
            f"sb:s-{m.match_id[3:]}-{i:02d}", m.match_id, m.competition_id, m.season_id,
            m.home if home else m.away, m.away if home else m.home, home, px, py, body,
            goal and goals, minute=2 + 3 * i, provider_xg=round(min(0.9, 1.5 / d), 4),
            match_date=m.match_date,
        ))  # fmt: skip
    return shots


def fixture_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(shots, freeze_frames, matches)`` of the fixture snapshot, shots sorted by ``shot_id``."""
    matches = _fixture_matches()
    shots: list[Shot] = []
    for k, m in enumerate(matches):
        n = 10 if m.competition_id == "sb:2" else 6 if m.gender == "female" else 12
        shots += _match_shots(m, k, n, goals=m.gender == "male")
    by_id = {m.match_id: m for m in matches}

    def extra(match_id: str, suffix: str, **kw: object) -> Shot:
        m = by_id[match_id]
        base = Shot(f"sb:s-{match_id[3:]}-{suffix}", m.match_id, m.competition_id,
                    m.season_id, m.home, m.away, True, match_date=m.match_date)  # fmt: skip
        return replace(base, **kw)

    excluded = [
        extra("sb:m03", "og", x=101.0, y=36.0, goal=True, own_goal=True, minute=40),
        extra("sb:m04", "pen", x=94.0, y=34.0, goal=True, penalty=True, period=2, minute=60),
        extra("sb:m05", "nff", x=90.0, y=34.0, minute=41),
        extra("sb:m07", "so", x=94.0, y=34.0, penalty=True, period=5, minute=120),
    ]
    frame = shots_frame(shots + excluded).sort("shot_id")
    with_frames = frame.filter(~pl.col("shot_id").is_in(EXCLUDED_IDS))
    return frame, frames_for(with_frames), matches_frame(matches)


EXCLUDED_IDS = ("sb:s-m03-og", "sb:s-m04-pen", "sb:s-m05-nff", "sb:s-m07-so")
EXCLUDED_REASONS = {
    "sb:s-m03-og": "own_goal",
    "sb:s-m04-pen": "penalty",
    "sb:s-m05-nff": "no_freeze_frame",
    "sb:s-m07-so": "shootout",
}
KEPT_SHOTS = 140  # 8 x 12 + 4 x 6 + 2 x 10
FAMILY_OF_COMPETITION = {
    ("sb:2", "sb:27"): "team-selected",  # 2 matches: each team plays exactly 50 %
    ("sb:11", "sb:4"): "league or tournament",
    ("sb:11", "sb:1"): "team-selected",
    ("sb:37", "sb:90"): "women",
}


def write_fixture_snapshot(
    root: Path,
    *,
    shots: pl.DataFrame | None = None,
    freeze_frames: pl.DataFrame | None = None,
    matches: pl.DataFrame | None = None,
    builder: str = "statsbomb",
) -> Path:
    """Write the fixture (or a variant) as snapshot ``statsbomb-open-v1``; the manifest path."""
    s, f, m = fixture_tables()
    result = snapshot.write_snapshot(
        SNAPSHOT,
        {
            "shots": s if shots is None else shots,
            "freeze_frames": f if freeze_frames is None else freeze_frames,
            "matches": m if matches is None else matches,
        },
        SOURCE,
        root=root,
        sort_by={"shots": ("shot_id",), "freeze_frames": ("shot_id",), "matches": ("match_id",)},
        builder=builder,
    )
    return result.manifest_path
