"""A hand-built synthetic snapshot for the M1.2 / M1.3 ablation tests (spec ``runs/MS3/features``).

Every id and number is invented: 23 matches (3 hold-out ``sb:2`` / ``sb:27``, 15 in a men's
league, 5 in a women's league), 36 shots each, drawn from a seeded generator so that every
level of ``body_part``, ``phase``, ``assist_type`` and ``contact`` has goals and misses in every
training fold (an unpenalised newton-cholesky fit has a finite optimum on every step and
variant). Goals depend on distance, open goal and one on one, so the steps have signal. Every
kept shot has a freeze frame (the opponent keeper is absent in about 10 %); about a quarter of
the direct free kicks are wide of the box (y < 13.84 or y > 54.16), and some central ones lie
on its edge. Two excluded shots carry frame rows: a direct corner (a goal) and an in-game
penalty. Nothing is copied from an open-data file.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import xg_cases as xc
from football_core import schema
from football_core.io import snapshot

SEED = 20261027
SHOTS_PER_MATCH = 36
_PL, _LG, _WM = (
    ("sb:2", "sb:27", "Premier League", "2015/2016"),
    ("sb:11", "sb:4", "Fixture League", "2018/2019"),
    ("sb:37", "sb:90", "Fixture Women's League", "2020/2021"),
)
_SITUATIONS = (
    "open_play", "fast_break", "from_corner", "from_free_kick", "from_throw_in",
    "direct_free_kick", "kick_off",
)  # fmt: skip
_SITUATION_P = (0.32, 0.11, 0.12, 0.12, 0.11, 0.19, 0.03)
_BODY_P = (0.4, 0.25, 0.23, 0.12)
_ASSIST_P = (0.3, 0.25, 0.2, 0.13, 0.12)
_TECHNIQUE_P = (0.34, 0.11, 0.11, 0.09, 0.11, 0.12, 0.12)
DIRECT_CORNER_ID = "sb:s-m05-dc"
PENALTY_ID = "sb:s-m06-pen"


def _matches() -> list[xc.Match]:
    out = []
    for i in range(23):
        comp, season, cname, sname = _PL if i < 3 else _WM if i >= 18 else _LG
        out.append(
            xc.Match(
                f"sb:m{i + 1:02d}", comp, season, f"sb:t{2 * i + 1:02d}", f"sb:t{2 * i + 2:02d}",
                "female" if comp == "sb:37" else "male", cname, sname,
                dt.date(2019, 1, 1) + dt.timedelta(days=7 * i),
            )
        )  # fmt: skip
    return out


def _location(rng: np.random.Generator, situation: str) -> tuple[float, float, bool]:
    """(x, y, wide): direct free kicks from distance, a quarter of them wide of the box and
    some central ones exactly on its edge (y = 13.84 or 54.16)."""
    if situation == "direct_free_kick":
        if rng.uniform() < 0.25:
            y = rng.uniform(3.0, 12.5) if rng.uniform() < 0.5 else rng.uniform(55.5, 65.0)
            return float(rng.uniform(82.0, 95.0)), float(y), True
        if rng.uniform() < 0.15:  # on the box edge: central (near miss for the wide rule)
            return float(rng.uniform(77.0, 88.0)), float(rng.choice([13.84, 54.16])), False
        return float(rng.uniform(77.0, 88.0)), float(rng.uniform(14.5, 53.5)), False
    return float(rng.uniform(84.0, 103.5)), float(rng.uniform(18.0, 50.0)), False


def _frame(rng: np.random.Generator, shot_id: str, sx: float, sy: float) -> dict[str, list[object]]:
    """2-6 frame players: maybe the opponent keeper, opponents and teammates near the goal."""
    rows: dict[str, list[object]] = {k: [] for k in ("player_id", "is_teammate", "is_keeper")}
    rows |= {"x": [], "y": []}
    players: list[tuple[bool, bool, float, float]] = []
    if rng.uniform() > 0.1:
        players.append((False, True, rng.uniform(99.0, 104.8), rng.uniform(31.0, 37.0)))
    for _ in range(int(rng.integers(1, 4))):
        players.append(
            (
                False,
                False,
                rng.uniform(min(sx, 104.0), 104.5),
                float(np.clip(sy + rng.normal(0, 5), 1, 67)),
            )
        )
    for _ in range(int(rng.integers(0, 3))):
        players.append(
            (
                True,
                False,
                rng.uniform(min(sx, 104.0), 104.5),
                float(np.clip(sy + rng.normal(0, 7), 1, 67)),
            )
        )
    for j, (mate, keeper, x, y) in enumerate(players):
        rows["player_id"].append(f"sb:pf-{shot_id[5:]}-{j}")
        rows["is_teammate"].append(mate)
        rows["is_keeper"].append(keeper)
        rows["x"].append(float(round(x, 3)))
        rows["y"].append(float(round(y, 3)))
    return {"shot_id": [shot_id] * len(players), **rows}


def ablation_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(shots, freeze_frames, matches)`` of the ablation snapshot, sorted by ``shot_id``."""
    rng = np.random.default_rng(SEED)
    matches = _matches()
    base: list[xc.Shot] = []
    ctx: dict[str, list[object]] = {
        k: []
        for k in (
            "situation", "assist_type", "technique", "first_time", "follows_dribble",
            "open_goal", "one_on_one",
        )
    }  # fmt: skip
    frames: dict[str, list[object]] = {
        k: [] for k in ("shot_id", "player_id", "is_teammate", "is_keeper", "x", "y")
    }
    for m in matches:
        for i in range(SHOTS_PER_MATCH):
            situation = str(rng.choice(_SITUATIONS, p=_SITUATION_P))
            x, y, _ = _location(rng, situation)
            body = str(rng.choice(schema.BODY_PARTS, p=_BODY_P))
            open_goal = bool(rng.uniform() < 0.1)
            one_on_one = bool(rng.uniform() < 0.14)
            distance = float(np.hypot(105.0 - x, 34.0 - y))
            logit = -0.9 - 0.13 * (distance - 12.0) + 1.3 * open_goal + 0.9 * one_on_one
            goal = bool(rng.uniform() < 1.0 / (1.0 + np.exp(-logit)))
            xg = 1.0 / (1.0 + np.exp(-(logit + rng.normal(0.0, 0.3))))
            home = i % 2 == 0
            shot_id = f"sb:s-{m.match_id[3:]}-{i:02d}"
            base.append(
                xc.Shot(
                    shot_id, m.match_id, m.competition_id, m.season_id,
                    m.home if home else m.away, m.away if home else m.home, home,
                    round(x, 3), round(y, 3), body, goal, minute=2 + 2 * i,
                    provider_xg=round(float(np.clip(xg, 0.005, 0.95)), 4),
                    match_date=m.match_date,
                )
            )  # fmt: skip
            ctx["situation"].append(situation)
            ctx["assist_type"].append(str(rng.choice(schema.ASSIST_TYPES, p=_ASSIST_P)))
            ctx["technique"].append(str(rng.choice(schema.TECHNIQUES, p=_TECHNIQUE_P)))
            ctx["first_time"].append(bool(rng.uniform() < 0.4))
            ctx["follows_dribble"].append(bool(rng.uniform() < 0.05))
            ctx["open_goal"].append(open_goal)
            ctx["one_on_one"].append(one_on_one)
            for k, v in _frame(rng, shot_id, base[-1].x, base[-1].y).items():
                frames[k] += v
    by_id = {m.match_id: m for m in matches}
    extra = []
    for shot_id, situation, x, y, goal in (
        (DIRECT_CORNER_ID, "direct_corner", 104.9, 0.4, True),
        (PENALTY_ID, "penalty", 94.0, 34.0, True),
    ):
        m = by_id[f"sb:{shot_id[5:8]}"]
        extra.append(
            xc.Shot(
                shot_id, m.match_id, m.competition_id, m.season_id, m.home, m.away, True, x, y,
                goal=goal, period=2, minute=89, penalty=situation == "penalty",
                match_date=m.match_date, situation=situation,
            )
        )  # fmt: skip
        for k, v in (("situation", situation), ("assist_type", "none"), ("technique", "normal")):
            ctx[k].append(v)
        for k in ("first_time", "follows_dribble", "open_goal", "one_on_one"):
            ctx[k].append(False)
        for k, v in _frame(rng, shot_id, x, y).items():
            frames[k] += v
    dtypes = {
        "situation": schema.SITUATION,
        "assist_type": schema.ASSIST_TYPE,
        "technique": schema.TECHNIQUE,
    }
    shots = xc.shots_frame(base + extra).with_columns(
        pl.Series(k, v, dtype=dtypes.get(k, pl.Boolean())) for k, v in ctx.items()
    )
    ff = pl.DataFrame(frames, schema=schema.FREEZE_FRAME_SCHEMA, strict=True)
    return shots.sort("shot_id"), ff.sort("shot_id"), xc.matches_frame(matches)


def write_ablation_snapshot(root: Path) -> Path:
    """Write the ablation fixture as snapshot ``statsbomb-open-v1`` under ``root``; the manifest."""
    shots, frames, matches = ablation_tables()
    result = snapshot.write_snapshot(
        xc.SNAPSHOT,
        {"shots": shots, "freeze_frames": frames, "matches": matches},
        xc.SOURCE,
        root=root,
        sort_by={"shots": ("shot_id",), "freeze_frames": ("shot_id",), "matches": ("match_id",)},
        builder="statsbomb",
    )
    return result.manifest_path
