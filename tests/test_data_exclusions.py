"""Exclusions and the exclusion log (spec §2): own goals, shoot-outs, penalties, no freeze frame."""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest
import xg_cases as xc
from football_core.io import snapshot

from football_shot_models import data

OPEN_PLAY = xc.Shot("sb:s1")
CASES = {
    "own_goal": xc.Shot("sb:s1", goal=True, own_goal=True),
    "shootout": xc.Shot("sb:s1", penalty=True, period=5, minute=120),
    "penalty": xc.Shot("sb:s1", penalty=True, goal=True),
    "no_freeze_frame": OPEN_PLAY,
}


def _reason(shot: xc.Shot, frame_ids: list[str]) -> str | None:
    shots = xc.shots_frame([shot])
    out = shots.select(data.exclusion_reason(pl.Series("shot_id", frame_ids, dtype=pl.String)))
    assert out.to_series().dtype == data.EXCLUSION_REASON
    value: str | None = out.item()
    return value


@pytest.mark.parametrize("reason", [*CASES, "kept"])
def test_exclusion_reason(reason: str) -> None:
    assert data.EXCLUSION_REASONS == ("own_goal", "shootout", "penalty", "no_freeze_frame")
    if reason == "kept":
        assert _reason(OPEN_PLAY, ["sb:s1", "sb:other"]) is None
    elif reason == "no_freeze_frame":
        assert _reason(CASES[reason], ["sb:other"]) == reason
    else:  # with a frame row, so only the flag can exclude it
        assert _reason(CASES[reason], ["sb:s1"]) == reason


def test_exclusion_precedence() -> None:
    """The first matching reason wins, so the reasons partition the excluded shots."""
    assert _reason(CASES["shootout"], []) == "shootout"  # a shoot-out kick is not "penalty"
    assert _reason(CASES["own_goal"], []) == "own_goal"  # never "no_freeze_frame"
    assert _reason(CASES["penalty"], []) == "penalty"
    shots = xc.shots_frame([xc.Shot("sb:a"), xc.Shot("sb:b")])
    reasons = shots.select(data.exclusion_reason(pl.Series("shot_id", [], dtype=pl.String)))
    assert reasons.to_series().to_list() == ["no_freeze_frame", "no_freeze_frame"]


def test_exclude_partitions() -> None:
    shots, frames, _ = xc.fixture_tables()
    # input order must not matter (not .reverse(): in polars 2.0.0 it turns an all-null
    # List(Array) column into List(Null))
    kept, log = data.exclude(shots.sort("shot_id", descending=True), frames)
    assert (
        pl.Schema(
            {
                "shot_id": pl.String(),
                "match_id": pl.String(),
                "competition_id": pl.String(),
                "season_id": pl.String(),
                "reason": data.EXCLUSION_REASON,
                "is_goal": pl.Boolean(),
            }
        )
        == data.EXCLUSION_SCHEMA
    )
    assert kept.height == xc.KEPT_SHOTS
    assert kept.height + log.height == shots.height
    assert set(kept["shot_id"]).isdisjoint(log["shot_id"])
    assert kept.schema == shots.schema  # the C3 columns, unchanged
    assert kept.equals(shots.filter(~pl.col("shot_id").is_in(xc.EXCLUDED_IDS)).sort("shot_id"))
    assert log.schema == data.EXCLUSION_SCHEMA
    assert log["shot_id"].to_list() == sorted(xc.EXCLUDED_IDS)
    assert dict(zip(log["shot_id"], log["reason"], strict=True)) == xc.EXCLUDED_REASONS
    goals = dict(zip(shots["shot_id"], shots["is_goal"], strict=True))
    assert log["is_goal"].to_list() == [goals[s] for s in log["shot_id"]]
    assert log.null_count().sum_horizontal().item() == 0


def test_exclusion_log_snapshot(manifest: Path, tmp_path: Path) -> None:
    _, log = data.exclude(*xc.fixture_tables()[:2])
    first = data.write_exclusion_log(log, manifest, root=tmp_path)
    assert first.status == "created"
    assert first.manifest_path == tmp_path / "manifests" / f"{data.EXCLUSIONS_SNAPSHOT}.json"
    assert data.EXCLUSIONS_SNAPSHOT == "xg-exclusions-v1"
    assert first.data_dir == tmp_path / "data" / "snapshots" / "xg-exclusions-v1"
    loaded = snapshot.load_snapshot(first.manifest_path)["exclusions"]
    assert loaded.schema == data.EXCLUSION_SCHEMA
    assert loaded.equals(log)
    (table,) = first.manifest.tables
    groups = {(g.competition_id, g.season_id): g.rows for g in table.groups or ()}
    assert groups == {("sb:11", "sb:4"): 3, ("sb:11", "sb:1"): 1}
    again = data.write_exclusion_log(log, manifest, root=tmp_path)
    assert again.status == "unchanged"


def test_exclusion_log_refuses_unignored_data(manifest: Path, tmp_path: Path) -> None:
    """The per-shot log must never land where git would track it (hard rule 1)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _, log = data.exclude(*xc.fixture_tables()[:2])
    with pytest.raises(ValueError, match="ignore"):
        data.write_exclusion_log(log, manifest, root=repo)
    assert not (repo / "data" / "snapshots" / "xg-exclusions-v1").exists()
