"""A hand-built synthetic WhoScored snapshot and v0 file for the MS4.v1 tests (spec
``runs/MS4/v1`` §9).

Every id, name and number is invented (``ws:m01``, teams ``ws:t01``...); nothing is copied from
the WhoScored archive or the legacy v0 file. Frames are built column by column with the C3
dtypes.

The snapshot (builder ``whoscored``, empty ``freeze_frames`` table) has 27 matches in three
competitions of unequal size: 15 train matches (Fixture League A and B in ``2025/2026``, Fixture
Cup in the calendar season ``2026``) and 12 test matches in ``2026/2027`` (5 / 4 / 3). Every
team plays one match, so no competition-season is team-selected. The last train match is
2026-07-04, the first test match 2026-08-08. The train matches are listed *after* the test
matches in the frame, so a date check on the first or last row is wrong.

Each match has 30 open-play or set-piece shots drawn from a seeded generator so that every
``body_part``, ``phase`` and ``assist`` level has goals and misses in every training fold (the
unpenalised newton-cholesky LR has a finite optimum). Goals depend on distance, big chance (a
strong effect, so a run that let the labelled ablation compete would make it headline) and one
on one. ``assist_type`` is never ``cutback``. Two kept shots have a null ``assist_type`` (one in
train, one in test); the test one also has a null ``big_chance``, so v0 scores one test shot
fewer than are kept (the two coverage denominators differ). Excluded shots: in-game penalties
(5 train kicks, 4 goals; 3 test kicks, 1 goal: the rates differ, so pooling them is visible),
two shoot-out kicks (train), an own goal (test) and a direct corner (train).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xg_cases as xc
from football_core import schema
from football_core.io import snapshot

SNAPSHOT = "whoscored-v1"
BUILDER = "whoscored"
SOURCE = snapshot.Source("files", "whoscored-csv-archive", "0" * 64)  # an invented digest
SEED = 20261025
SHOTS_PER_MATCH = 30
TRAIN_SEASONS = ("2025/2026", "2026")
TEST_SEASONS = ("2026/2027",)
LAST_TRAIN = dt.date(2026, 7, 4)
FIRST_TEST = dt.date(2026, 8, 8)
V0_DATE = "2026-07-06"  # after the last train match, before the first test match

# (competition id, name), season name, season id, matches
_A, _B, _C = (("ws:c10", "Fixture League A"), ("ws:c20", "Fixture League B"),
              ("ws:c30", "Fixture Cup"))  # fmt: skip
_BLOCKS: tuple[tuple[tuple[str, str], str, str, int], ...] = (
    (_A, "2026/2027", "ws:s2627", 5),
    (_B, "2026/2027", "ws:s2627", 4),
    (_C, "2026/2027", "ws:s2627", 3),
    (_A, "2025/2026", "ws:s2526", 6),
    (_B, "2025/2026", "ws:s2526", 5),
    (_C, "2026", "ws:s2026", 4),
)
TEST_COMPETITIONS = {"Fixture League A": 5, "Fixture League B": 4, "Fixture Cup": 3}
TRAIN_MATCHES, TEST_MATCHES = 15, 12

_SITUATIONS = (
    "open_play", "fast_break", "from_corner", "from_free_kick", "from_throw_in",
    "direct_free_kick", "kick_off",
)  # fmt: skip
_SITUATION_P = (0.30, 0.12, 0.13, 0.12, 0.12, 0.18, 0.03)
_BODY_P = (0.4, 0.25, 0.22, 0.13)
_ASSISTS = ("none", "pass", "cross", "through_ball")  # never cutback (C5 census)
_ASSIST_P = (0.3, 0.3, 0.22, 0.18)
# log-odds effects of the context levels (the steps have signal over distance + angle)
EFFECTS = {"head": -0.8, "fast_break": 0.8, "direct_free_kick": 0.6, "from_corner": -0.4,
           "through_ball": 1.0, "cross": -0.4}  # fmt: skip

NULL_ASSIST_TRAIN = "ws:s-m13-00"  # a train match (see _matches order)
NULL_ASSIST_TEST = "ws:s-m01-00"  # a test match; big_chance null too
OWN_GOAL_ID = "ws:s-m02-og"
DIRECT_CORNER_ID = "ws:s-m14-dc"
SHOOTOUT_IDS = ("ws:s-m27-so1", "ws:s-m27-so2")
# shot id suffix, match, goal: in-game penalties (train m13-m27, test m01-m12)
_KICKS: tuple[tuple[str, str, bool], ...] = (
    ("pen", "ws:m13", True),
    ("pen", "ws:m15", True),
    ("pen", "ws:m18", True),
    ("pen", "ws:m21", False),
    ("pen", "ws:m25", True),
    ("pen", "ws:m03", True),
    ("pen", "ws:m07", False),
    ("pen", "ws:m10", False),
)
TRAIN_KICKS, TRAIN_KICK_GOALS = 5, 4
TEST_KICKS, TEST_KICK_GOALS = 3, 1
DROPPED_ASSIST = 2


def _matches() -> list[xc.Match]:
    """Test matches first (m01-m12), then train (m13-m27); train dates end on LAST_TRAIN."""
    out: list[xc.Match] = []
    k = 0
    for (comp, cname), sname, season, n in _BLOCKS:
        for j in range(n):
            k += 1
            if sname == "2026/2027":
                day = FIRST_TEST + dt.timedelta(days=7 * j + (k % 3))
            elif sname == "2026":
                day = LAST_TRAIN - dt.timedelta(days=21 * (n - 1 - j))
            else:
                day = dt.date(2025, 8, 9) + dt.timedelta(days=14 * j + (k % 4))
            out.append(
                xc.Match(f"ws:m{k:02d}", comp, season, f"ws:t{2 * k - 1:02d}", f"ws:t{2 * k:02d}",
                         "male", cname, sname, day)
            )  # fmt: skip
    return out


def _location(rng: np.random.Generator, situation: str) -> tuple[float, float]:
    if situation == "direct_free_kick":
        if rng.uniform() < 0.3:  # wide of the box: phase free_kick
            y = rng.uniform(3.0, 12.5) if rng.uniform() < 0.5 else rng.uniform(55.5, 65.0)
            return float(rng.uniform(84.0, 95.0)), float(y)
        return float(rng.uniform(80.0, 88.0)), float(rng.uniform(15.0, 53.0))
    return float(rng.uniform(84.0, 103.5)), float(rng.uniform(18.0, 50.0))


def _to_whoscored(shots: pl.DataFrame) -> pl.DataFrame:
    """xg_cases' StatsBomb-shaped frame turned into WhoScored rows: provider, ws: player ids,
    and the columns the provider does not record set to null."""
    return shots.with_columns(
        pl.lit("whoscored").cast(schema.PROVIDER).alias("provider"),
        (pl.lit("ws:p-") + pl.col("team_id").str.slice(3)).alias("player_id"),
        pl.when(pl.col("gk_id").is_not_null())
        .then(pl.lit("ws:gk-") + pl.col("opponent_team_id").str.slice(3))
        .alias("gk_id"),  # MS5.goalmouth: the keeper id carries the provider prefix too
        *(pl.lit(None).cast(shots.schema[c]).alias(c)
          for c in ("technique", "open_goal", "follows_dribble", "provider_xg")),
        pl.lit("[]").alias("raw_qualifiers"),
    )  # fmt: skip


def v1_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(shots, freeze_frames, matches)`` of the snapshot; shots sorted by ``shot_id``, an empty
    freeze-frame table."""
    rng = np.random.default_rng(SEED)
    matches = _matches()
    base: list[xc.Shot] = []
    ctx: dict[str, list[object]] = {
        k: [] for k in ("situation", "assist_type", "first_time", "one_on_one", "big_chance")
    }
    for m in matches:
        for i in range(SHOTS_PER_MATCH):
            situation = str(rng.choice(_SITUATIONS, p=_SITUATION_P))
            x, y = _location(rng, situation)
            body = str(rng.choice(schema.BODY_PARTS, p=_BODY_P))
            big = bool(rng.uniform() < 0.2)
            one = bool(rng.uniform() < 0.12)
            distance = float(np.hypot(105.0 - x, 34.0 - y))
            assist = str(rng.choice(_ASSISTS, p=_ASSIST_P))
            logit = -1.3 - 0.11 * (distance - 12.0) + 2.0 * big + 1.4 * one + EFFECTS.get(body, 0.0)
            logit += EFFECTS.get(situation, 0.0) + EFFECTS.get(assist, 0.0)
            goal = bool(rng.uniform() < 1.0 / (1.0 + np.exp(-logit)))
            home = i % 2 == 0
            shot_id = f"ws:s-{m.match_id[3:]}-{i:02d}"
            base.append(
                xc.Shot(
                    shot_id, m.match_id, m.competition_id, m.season_id,
                    m.home if home else m.away, m.away if home else m.home, home,
                    round(x, 3), round(y, 3), body, goal, period=1 if i < 15 else 2,
                    minute=2 + 3 * i, provider_xg=None, match_date=m.match_date,
                    situation=situation,
                )
            )  # fmt: skip
            ctx["situation"].append(situation)
            ctx["assist_type"].append(None if shot_id in (NULL_ASSIST_TRAIN, NULL_ASSIST_TEST)
                                      else assist)  # fmt: skip
            ctx["first_time"].append(bool(rng.uniform() < 0.4))
            ctx["one_on_one"].append(one)
            ctx["big_chance"].append(None if shot_id == NULL_ASSIST_TEST else big)
    by_id = {m.match_id: m for m in matches}
    extra: list[xc.Shot] = []

    def add(shot_id: str, match_id: str, **kw: Any) -> None:
        m = by_id[match_id]
        shot = xc.Shot(shot_id, m.match_id, m.competition_id, m.season_id, m.home, m.away, True,
                       94.0, 34.0, match_date=m.match_date, provider_xg=None)  # fmt: skip
        extra.append(dataclasses.replace(shot, **kw))
        ctx["situation"].append(extra[-1].situation or ("penalty" if extra[-1].penalty
                                                        else "open_play"))  # fmt: skip
        ctx["assist_type"].append("none")
        for k in ("first_time", "one_on_one", "big_chance"):
            ctx[k].append(False)

    for suffix, match_id, goal in _KICKS:
        add(f"ws:s-{match_id[3:]}-{suffix}", match_id, goal=goal, penalty=True, period=2,
            minute=60)  # fmt: skip
    add(SHOOTOUT_IDS[0], "ws:m27", goal=True, penalty=True, period=5, minute=120)
    add(SHOOTOUT_IDS[1], "ws:m27", goal=False, penalty=True, period=5, minute=120)
    add(OWN_GOAL_ID, "ws:m02", x=101.0, y=36.0, goal=True, own_goal=True, minute=40)
    add(DIRECT_CORNER_ID, "ws:m14", x=104.9, y=0.4, goal=True, minute=70,
        situation="direct_corner")  # fmt: skip
    dtypes = {"situation": schema.SITUATION, "assist_type": schema.ASSIST_TYPE}
    shots = xc.shots_frame(base + extra).with_columns(
        pl.Series(k, v, dtype=dtypes.get(k, pl.Boolean())) for k, v in ctx.items()
    )
    shots = _to_whoscored(shots).sort("shot_id")
    frames = pl.DataFrame(schema=schema.FREEZE_FRAME_SCHEMA)
    matches_df = xc.matches_frame(matches).with_columns(
        pl.lit("whoscored").cast(schema.PROVIDER).alias("provider"),
        pl.lit(None, dtype=pl.Int16()).alias("home_score"),
        pl.lit(None, dtype=pl.Int16()).alias("away_score"),
        pl.lit(None, dtype=pl.Boolean()).alias("has_360"),
    )
    return shots, frames, matches_df


def write_v1_snapshot(root: Path, *, matches: pl.DataFrame | None = None) -> Path:
    """Write the fixture as snapshot ``whoscored-v1`` (builder ``whoscored``); the manifest."""
    shots, frames, m = v1_tables()
    result = snapshot.write_snapshot(
        SNAPSHOT,
        {"shots": shots, "freeze_frames": frames, "matches": m if matches is None else matches},
        SOURCE,
        root=root,
        sort_by={"shots": ("shot_id",), "freeze_frames": ("shot_id",), "matches": ("match_id",)},
        builder=BUILDER,
    )
    return result.manifest_path


# ---- v0 -----------------------------------------------------------------------------------

# Invented coefficients in v0's order (spec §3 V0_FEATURES); not the legacy file's values.
V0_COEF: Mapping[str, float] = {
    "log_dist": 0.4,
    "dist_m": -0.14,
    "angle_rad": 1.1,
    "header": -0.6,
    "big_chance": 1.8,
    "one_on_one": 0.5,
    "fast_break": 0.2,
    "from_corner": -0.1,
    "set_piece": -0.3,
    "direct_freekick": 0.35,
}
V0_INTERCEPT = -1.7
V0_PENALTY = 0.78
V0_SHOTS = 400


def v0_doc(
    *, when: object = V0_DATE, features: Sequence[str] | None = None,
    coefficients: Mapping[str, float] | None = None,
) -> dict[str, Any]:  # fmt: skip
    return {
        "features": list(V0_COEF if features is None else features),
        "coefficients": dict(V0_COEF if coefficients is None else coefficients),
        "intercept": V0_INTERCEPT,
        "penalty_xg": V0_PENALTY,
        "trained_on": {"shots": V0_SHOTS, "date": when},
    }


def write_v0(path: Path, doc: Mapping[str, Any] | None = None) -> str:
    """Write a fixture v0 JSON to ``path``; its SHA-256 (to patch into ``legacy.V0_SHA256``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(v0_doc() if doc is None else doc), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_model(coefficients: Mapping[str, float], intercept: float = 0.0) -> Any:
    """A ``legacy.LegacyModel`` with the given coefficients (missing ones 0)."""
    from football_shot_models import legacy

    coef = {f: float(coefficients.get(f, 0.0)) for f in legacy.V0_FEATURES}
    return legacy.LegacyModel(coef, intercept, V0_PENALTY, dt.date(2026, 7, 6), V0_SHOTS, "0" * 64)


def kept_open_play(shots: pl.DataFrame) -> pl.DataFrame:
    """The fixture shots a model sees: no penalty, own goal, shoot-out or direct corner, and a
    non-null assist type (the v1 population)."""
    return shots.filter(
        ~pl.col("is_penalty")
        & ~pl.col("is_own_goal")
        & (pl.col("situation") != "direct_corner")
        & pl.col("assist_type").is_not_null()
    )
