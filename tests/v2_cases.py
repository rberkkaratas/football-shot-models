"""A hand-built synthetic WhoScored snapshot for the PR1.xg-context v2 tests (spec
``runs/PR1/xg-context`` §3.4, §6 integration fixture census).

It is ``v1_cases``' snapshot (27 matches, 15 train / 12 test, three test competitions of 5 / 4 /
3 matches, all under ``models.MIN_SEGMENT_SHOTS``) with the schema 1.3 columns filled and Opta
qualifier ids written into ``raw_qualifiers``. Every id, value and count is invented.

- ``prev_action``: drawn per shot from a seeded generator, with goals leaning to the opponent's
  keeper / block and the team's own shot (so the context carries signal); every one of the nine
  levels occurs in train and in test (``census_reached`` asserts it). Own goal: null.
- ``rebound``: True for about a quarter of goals and a twentieth of misses, in train and test.
  Own goal: null.
- ``raw_qualifiers``: the body part's id (q20 / q72 / q15 / q21), a pattern id (q22 / q23 / q24 /
  q25 / q26 / q160 / q9, q263 for the direct corner), q214 when ``big_chance``, q89 when
  ``one_on_one``, q328 when ``first_time``, q55 on assisted shots, q17 (a zone) on every third
  shot, q108 (volley) on every seventh, q82 (blocked, post-shot) on every fifth, q11113 (key
  pass, outcome-derived) on one train shot, and the unknown id ``UNKNOWN_ID`` on one train shot
  (so census check row 6 is missed on this fixture).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import v1_cases as vc
from football_core import schema
from football_core.io import snapshot

SEED = 20261029
UNKNOWN_ID = 99_999
OUTCOME_DERIVED_ID = 11_113
PREV_LEVELS = (
    "pass",
    "take_on",
    "regain",
    "shot",
    "other",
    "opp_clearance",
    "opp_block",
    "opp_keeper",
    "opp_other",
)  # fmt: skip  (spec §2.1, typed by hand)
_GOAL_P = (0.10, 0.10, 0.08, 0.14, 0.06, 0.06, 0.18, 0.20, 0.08)
_MISS_P = (0.40, 0.08, 0.07, 0.03, 0.08, 0.14, 0.06, 0.04, 0.10)
_BODY_ID = {"right_foot": 20, "left_foot": 72, "head": 15, "other": 21}
_PATTERN_ID = {"open_play": 22, "fast_break": 23, "from_free_kick": 24, "from_corner": 25,
               "direct_free_kick": 26, "from_throw_in": 160, "penalty": 9, "kick_off": 22,
               "direct_corner": 25}  # fmt: skip
NAMES = {2: "Cross", 4: "Throughball", 9: "Penalty", 15: "Head", 17: "BoxCentre", 20: "RightFoot",
         21: "OtherBodyPart", 22: "RegularPlay", 23: "FastBreak", 24: "SetPiece",
         25: "FromCorner", 26: "DirectFreekick", 55: "RelatedEventId", 72: "LeftFoot",
         82: "Blocked", 89: "OneOnOne", 108: "Volley", 160: "ThrowinSetPiece", 214: "BigChance",
         263: "DirectCorner", 328: "FirstTouch", OUTCOME_DERIVED_ID: "KeyPass",
         UNKNOWN_ID: "FixtureUnknown"}  # fmt: skip


def _ids(r: dict[str, Any], i: int) -> list[int]:
    out = [_PATTERN_ID[r["situation"]]]
    if r["situation"] == "direct_corner":
        out.append(263)
    if r["body_part"] is not None:
        out.append(_BODY_ID[r["body_part"]])
    for column, qid in (("big_chance", 214), ("one_on_one", 89), ("first_time", 328)):
        if r[column]:
            out.append(qid)
    if r["assist_type"] not in (None, "none"):
        out.append(55)
    out += [qid for every, qid in ((3, 17), (7, 108), (5, 82)) if i % every == 0]
    return out


def v2_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(shots, freeze_frames, matches)``: ``v1_cases.v1_tables`` with the context columns."""
    shots, frames, matches = vc.v1_tables()
    rng = np.random.default_rng(SEED)
    prev: list[str | None] = []
    rebound: list[bool | None] = []
    raw: list[str] = []
    train_ids = set(
        shots.join(matches.filter(pl.col("season_name").is_in(vc.TRAIN_SEASONS)), on="match_id",
                   how="semi")["shot_id"]
    )  # fmt: skip
    marked = {"unknown": False, "derived": False}
    for i, r in enumerate(shots.iter_rows(named=True)):
        ids = _ids(r, i)
        if r["is_own_goal"]:
            prev.append(None)
            rebound.append(None)
        else:
            weights = _GOAL_P if r["is_goal"] else _MISS_P
            prev.append(str(rng.choice(PREV_LEVELS, p=weights)))
            rebound.append(bool(rng.uniform() < (0.25 if r["is_goal"] else 0.05)))
            kept = r["assist_type"] is not None and r["situation"] != "direct_corner"
            if r["shot_id"] in train_ids and kept and not r["is_penalty"]:
                for key, qid in (("unknown", UNKNOWN_ID), ("derived", OUTCOME_DERIVED_ID)):
                    if not marked[key] and i % 11 == (1 if key == "unknown" else 2):
                        ids.append(qid)
                        marked[key] = True
        raw.append(json.dumps([{"type": {"value": q, "displayName": NAMES[q]}} for q in ids],
                              separators=(",", ":")))  # fmt: skip
    assert all(marked.values())
    shots = shots.with_columns(
        pl.Series("prev_action", prev, dtype=schema.PREV_ACTION),
        pl.Series("rebound", rebound, dtype=pl.Boolean()),
        pl.Series("raw_qualifiers", raw, dtype=pl.String()),
    )
    return shots, frames, matches


def write_v2_snapshot(root: Path) -> Path:
    """The fixture as snapshot ``whoscored-v1`` (builder ``whoscored``); the manifest path."""
    shots, frames, matches = v2_tables()
    result = snapshot.write_snapshot(
        vc.SNAPSHOT,
        {"shots": shots, "freeze_frames": frames, "matches": matches},
        vc.SOURCE,
        root=root,
        sort_by={"shots": ("shot_id",), "freeze_frames": ("shot_id",), "matches": ("match_id",)},
        builder=vc.BUILDER,
    )
    return result.manifest_path


def qualifier_ids(raw: str) -> set[int]:
    """The ids of one ``raw_qualifiers`` JSON text (read here, independently of ``v2``)."""
    return {int(item["type"]["value"]) for item in json.loads(raw)}


def census_reached(dev: pl.DataFrame, held: pl.DataFrame) -> None:
    """§6 integration fixture census: every ``prev_action`` level in train (both opp_block and
    opp_keeper), rebound True and False in train and test, ≥ 2 test competitions, and one id of
    every class the spec names plus the unknown one in train."""
    assert set(dev["prev_action"].cast(pl.String)) == set(PREV_LEVELS)
    for frame in (dev, held):
        assert set(frame["rebound"]) == {True, False}
    assert held["competition_name"].n_unique() >= 2
    ids = set().union(*(qualifier_ids(t) for t in dev["raw_qualifiers"]))
    for qid in (20, 22, 17, 55, 108, 214, 82, OUTCOME_DERIVED_ID, UNKNOWN_ID):
        assert qid in ids, qid
