"""Exclusions and the exclusion log (spec §2; MS3.features §5): own goals, shoot-outs, penalties,
direct corners, no freeze frame."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
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
    "direct_corner": xc.Shot("sb:s1", x=104.9, y=0.4, goal=True, situation="direct_corner"),
    "no_freeze_frame": OPEN_PLAY,
}
REASONS = ("own_goal", "shootout", "penalty", "direct_corner", "no_freeze_frame")


def _reason(shot: xc.Shot, frame_ids: list[str]) -> str | None:
    shots = xc.shots_frame([shot])
    out = shots.select(data.exclusion_reason(pl.Series("shot_id", frame_ids, dtype=pl.String)))
    assert out.to_series().dtype == data.EXCLUSION_REASON
    value: str | None = out.item()
    return value


@pytest.mark.parametrize("reason", [*CASES, "kept"])
def test_exclusion_reason(reason: str) -> None:
    assert data.EXCLUSION_REASONS == REASONS  # MS3.features: direct_corner after penalty
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


def test_exclusion_direct_corner() -> None:
    """StatsBomb logs a direct corner as a shot on its outcome, so it leaves the population
    with its own reason, before ``no_freeze_frame``; the rule is on ``situation``, not on the
    spot (a corner-phase shot and a direct free kick from the same spot are kept)."""
    assert (
        data.EXCLUSION_REASONS.index("direct_corner") == data.EXCLUSION_REASONS.index("penalty") + 1
    )
    corner = CASES["direct_corner"]
    assert _reason(corner, ["sb:s1"]) == "direct_corner"
    assert _reason(corner, []) == "direct_corner"  # not "no_freeze_frame"
    for situation in ("from_corner", "direct_free_kick", "open_play"):
        assert _reason(replace(corner, situation=situation), ["sb:s1"]) is None
    shots = xc.shots_frame([corner, replace(corner, shot_id="sb:s2", situation="from_corner")])
    kept, log = data.exclude(shots, pl.DataFrame({"shot_id": ["sb:s1", "sb:s2"]}))
    assert kept["shot_id"].to_list() == ["sb:s2"]
    assert log.select("shot_id", "reason", "is_goal").rows() == [("sb:s1", "direct_corner", True)]


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
    # pinned to the source snapshot's name and data digest, not its manifest bytes
    meta = snapshot.read_manifest(manifest)
    assert first.manifest.source == snapshot.Source("files", meta.name, meta.data_digest)


def test_exclusion_log_ignores_source_environment(manifest: Path, tmp_path: Path) -> None:
    """A core commit changes only the StatsBomb manifest's ``environment``; the log's manifest
    must not change with it. A source snapshot of another name is named in the log."""
    _, log = data.exclude(*xc.fixture_tables()[:2])
    first = data.write_exclusion_log(log, manifest, root=tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["environment"] = {**raw["environment"], "football_core_git": "f" * 40}
    moved = tmp_path / "elsewhere" / manifest.name
    moved.parent.mkdir()
    moved.write_text(json.dumps(raw), encoding="utf-8")
    again = data.write_exclusion_log(log, moved, root=tmp_path)
    assert (again.status, again.manifest_written) == ("unchanged", False)
    assert again.manifest == first.manifest
    raw["name"], raw["data_dir"] = "statsbomb-open-v9", "../data/snapshots/statsbomb-open-v9"
    renamed = tmp_path / "elsewhere" / "statsbomb-open-v9.json"
    renamed.write_text(json.dumps(raw), encoding="utf-8")
    other = data.write_exclusion_log(log, renamed, root=tmp_path)
    assert other.manifest.source.uri == "statsbomb-open-v9"


def test_exclusion_log_refuses_unignored_data(manifest: Path, tmp_path: Path) -> None:
    """The per-shot log must never land where git would track it (hard rule 1)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _, log = data.exclude(*xc.fixture_tables()[:2])
    with pytest.raises(ValueError, match="ignore"):
        data.write_exclusion_log(log, manifest, root=repo)
    assert not (repo / "data" / "snapshots" / "xg-exclusions-v1").exists()


# ---- MS4.prep §4: provider seams (frames by builder) -------------------------------------------


def test_exclude_without_frames() -> None:
    """MS4.prep §4: ``exclude(shots, None)`` (a provider without frames) keeps the frameless
    open-play shots that ``exclude(shots, empty_frames)`` tags ``no_freeze_frame``; every other
    reason is the same row for row; ``freeze_frames`` has no default (a caller states "no
    frames", it never gets it by omission)."""
    import inspect

    shots = xc.shots_frame(
        [
            replace(OPEN_PLAY, shot_id="sb:s0"),
            *(replace(shot, shot_id=f"sb:s{i}") for i, shot in enumerate(CASES.values(), 1)),
        ]
    )
    empty = xc.frames_for(shots.head(0))
    assert empty.height == 0
    kept_none, log_none = data.exclude(shots, None)
    kept_empty, log_empty = data.exclude(shots, empty)
    assert kept_none["shot_id"].to_list() == ["sb:s0", "sb:s5"]  # both frameless open-play shots
    assert kept_none.schema == shots.schema and kept_empty.height == 0
    assert log_none.schema == data.EXCLUSION_SCHEMA
    reasons = dict(zip(log_empty["shot_id"], log_empty["reason"], strict=True))
    assert reasons == {
        "sb:s0": "no_freeze_frame", "sb:s1": "own_goal", "sb:s2": "shootout",
        "sb:s3": "penalty", "sb:s4": "direct_corner", "sb:s5": "no_freeze_frame",
    }  # fmt: skip
    assert log_none.equals(log_empty.filter(pl.col("reason") != "no_freeze_frame"))
    tags = shots.select(data.exclusion_reason(None)).to_series()
    assert "no_freeze_frame" not in tags.to_list() and tags.dtype == data.EXCLUSION_REASON
    default = inspect.signature(data.exclude).parameters["freeze_frames"].default
    assert default is inspect.Parameter.empty


def test_load_xg_data_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MS4.prep §4: the builder is checked against ``FRAMES_BY_BUILDER`` before the manifest is
    read; a frameless builder (patched in, read at call time) loads with no ``no_freeze_frame``
    row and keeps the frameless shot, while a builder mapped to True still excludes it; a
    manifest of another builder names the expected one."""
    from types import MappingProxyType

    assert data.FRAMES_BY_BUILDER == {"statsbomb": True, "whoscored": False}  # MS4.v1 §2.1
    assert isinstance(data.FRAMES_BY_BUILDER, MappingProxyType)
    missing = tmp_path / "manifests" / "nope.json"
    with pytest.raises(ValueError, match="(?=.*'wyscout')(?=.*statsbomb)"):
        data.load_xg_data(missing, builder="wyscout")
    manifest = xc.write_fixture_snapshot(tmp_path / "parquet", builder="parquet")
    expected = "was written by builder 'parquet'; expected 'statsbomb'"
    with pytest.raises(ValueError, match=expected):
        data.load_xg_data(manifest)
    monkeypatch.setattr(data, "FRAMES_BY_BUILDER", {"statsbomb": True, "parquet": False})
    with pytest.raises(ValueError, match=expected):
        data.load_xg_data(manifest, builder="statsbomb")
    xg = data.load_xg_data(manifest, builder="parquet")
    nff = "sb:s-m05-nff"
    assert "no_freeze_frame" not in xg.exclusions["reason"].to_list()
    assert set(xg.exclusions["shot_id"]) == set(xc.EXCLUDED_IDS) - {nff}
    assert xg.shots.height == xc.KEPT_SHOTS + 1 and nff in xg.shots["shot_id"].to_list()
    assert xg.freeze_frames.height == xc.KEPT_SHOTS  # still the kept shots' frame rows
    assert set(xg.freeze_frames["shot_id"]) == set(xg.shots["shot_id"]) - {nff}
    monkeypatch.setattr(data, "FRAMES_BY_BUILDER", {"statsbomb": True, "parquet": True})
    framed = data.load_xg_data(manifest, builder="parquet")
    assert framed.shots.height == xc.KEPT_SHOTS and nff not in framed.shots["shot_id"].to_list()
    statsbomb = xc.write_fixture_snapshot(tmp_path / "statsbomb")
    with pytest.raises(ValueError, match="was written by builder 'statsbomb'; expected 'parquet'"):
        data.load_xg_data(statsbomb, builder="parquet")
