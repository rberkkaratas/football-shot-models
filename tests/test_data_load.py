"""Loading the verified snapshot, match families and the hold-out split (spec §2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
import xg_cases as xc
from football_core import schema
from football_core.io import snapshot

from football_shot_models import data, features


def test_load_verifies_snapshot(tmp_path: Path) -> None:
    manifest = xc.write_fixture_snapshot(tmp_path)
    shots_file = tmp_path / "data" / "snapshots" / xc.SNAPSHOT / "shots.parquet"
    raw = bytearray(shots_file.read_bytes())
    raw[len(raw) // 2] ^= 0x01  # one changed byte, same size
    shots_file.write_bytes(bytes(raw))
    with pytest.raises(snapshot.SnapshotVerificationError, match="shots.parquet"):
        data.load_xg_data(manifest)


def _drop_match(matches: pl.DataFrame) -> pl.DataFrame:
    return matches.filter(pl.col("match_id") != "sb:m05")


def _duplicate_match(matches: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([matches, matches.filter(pl.col("match_id") == "sb:m06")])


def test_load_rejects_other_builder(tmp_path: Path) -> None:
    manifest = xc.write_fixture_snapshot(tmp_path, builder="parquet")
    with pytest.raises(ValueError, match="parquet"):
        data.load_xg_data(manifest)


@pytest.mark.parametrize(
    ("transform", "table", "check", "count"),
    [
        # all 13 shots of sb:m05, the excluded no-frame shot included: core checks every shot
        (_drop_match, "shots", "match_exists", 13),
        (_duplicate_match, "matches", "unique", 2),
    ],
    ids=["shot without a matches row", "duplicate match_id"],
)
def test_snapshot_refuses_bad_matches(
    transform: Any, table: str, check: str, count: int, tmp_path: Path
) -> None:
    """MS3.matches §1.5: shot/match consistency is core's, checked when the snapshot is
    written, so load_xg_data no longer needs its own orphan and duplicate checks."""
    _, _, matches = xc.fixture_tables()
    with pytest.raises(schema.SchemaValidationError) as info:
        xc.write_fixture_snapshot(tmp_path, matches=transform(matches))
    assert info.value.table == table
    (failure,) = [f for f in info.value.failures if f.check == check]
    assert failure.count == count
    assert not (tmp_path / "manifests").exists()


def test_load_joins_match_features(xg_data: Any) -> None:
    """Every matches pre_shot column (today gender) is joined onto the shots, typed as core
    types it, between the C3 columns and family; gender is then usable as a feature."""
    assert schema.column_names("matches", roles="pre_shot") == ("gender",)
    assert data.MATCH_FEATURES == ("gender",)
    shots = xg_data.shots
    assert shots.columns == [*schema.column_names("shots"), *data.MATCH_FEATURES, "family"]
    assert shots.schema["gender"] == schema.GENDER
    assert xg_data.matches.schema == schema.MATCH_SCHEMA
    matrix, names = features.design_matrix(shots, ["gender"])
    assert names == ("gender=female",)
    assert matrix[:, 0].sum() == 24  # 4 women's matches x 6 kept shots


def test_load_adds_segments_in_order(xg_data: Any, manifest: Path) -> None:
    shots, frames, matches = xc.fixture_tables()
    kept, log = data.exclude(shots, frames)
    loaded = xg_data.shots
    assert loaded.columns == [*schema.column_names("shots"), "gender", "family"]
    assert loaded.height == kept.height == xc.KEPT_SHOTS
    assert loaded["shot_id"].to_list() == sorted(loaded["shot_id"].to_list())
    assert loaded.drop("gender", "family").equals(kept)
    assert loaded.schema["family"] == data.FAMILY
    gender = dict(zip(matches["match_id"], matches["gender"], strict=True))
    assert loaded["gender"].to_list() == [gender[m] for m in loaded["match_id"]]
    family = [
        xc.FAMILY_OF_COMPETITION[(c, s)]
        for c, s in zip(loaded["competition_id"], loaded["season_id"], strict=True)
    ]
    assert loaded["family"].to_list() == family
    assert xg_data.exclusions.equals(log)
    assert xg_data.matches.sort("match_id").equals(matches.sort("match_id"))
    assert xg_data.manifest == snapshot.read_manifest(manifest)
    assert xg_data.manifest.builder == "statsbomb"


def _family(matches: list[xc.Match]) -> list[str]:
    out = data.match_family(xc.matches_frame(matches))
    assert out.columns == ["match_id", "family"]
    assert out.schema["family"] == data.FAMILY
    assert out["match_id"].to_list() == sorted(m.match_id for m in matches)
    return out.sort("match_id")["family"].to_list()


def _season(season: str, pairs: list[tuple[str, str]], gender: str = "male") -> list[xc.Match]:
    return [
        xc.Match(f"sb:{season}-{i}", "sb:1", f"sb:{season}", home, away, gender)
        for i, (home, away) in enumerate(pairs)
    ]


@pytest.mark.parametrize(
    ("matches", "expected"),
    [
        # gender wins over a single team
        (_season("w", [("sb:a", "sb:b"), ("sb:c", "sb:a")], "female"), ["women", "women"]),
        # sb:a (home or away) in every match of season s; season t is a league. Grouping by
        # competition only would give sb:a 2 / 6 and call season s a league.
        (
            _season("s", [("sb:a", "sb:b"), ("sb:c", "sb:a")])
            + _season(
                "t", [("sb:d", "sb:e"), ("sb:f", "sb:g"), ("sb:h", "sb:i"), ("sb:j", "sb:k")]
            ),
            ["team-selected"] * 2 + ["league or tournament"] * 4,
        ),
        (_season("f", [("sb:a", "sb:b")]), ["team-selected"]),
        (
            _season("h", [("sb:a", "sb:b"), ("sb:c", "sb:a"), ("sb:d", "sb:e"), ("sb:f", "sb:g")]),
            ["team-selected"] * 4,
        ),
        (
            _season(
                "l",
                [
                    ("sb:a", "sb:b"),
                    ("sb:c", "sb:a"),
                    ("sb:d", "sb:e"),
                    ("sb:f", "sb:g"),
                    ("sb:h", "sb:i"),
                ],
            ),  # fmt: skip
            ["league or tournament"] * 5,
        ),
    ],
    ids=["women", "single-team season", "one-match final", "exactly 50 %", "league"],
)
def test_match_family(matches: list[xc.Match], expected: list[str]) -> None:
    assert data.FAMILIES == ("league or tournament", "team-selected", "women")
    assert data.TEAM_SELECTED_SHARE == 0.5
    assert _family(matches) == expected


@pytest.mark.parametrize("case", ["ok", "empty holdout", "empty dev"])
def test_split_holdout(case: str) -> None:
    shots, _, _ = xc.fixture_tables()
    # another Premier League season, and another competition in season sb:27, must stay in
    # dev: the split is on the pair, not on either id alone
    other = pl.col("shot_id").is_in(["sb:s-m01-00", "sb:s-m01-01"])
    sibling = pl.col("shot_id").is_in(["sb:s-m03-00", "sb:s-m03-01"])
    shots = shots.with_columns(
        pl.when(other).then(pl.lit("sb:28")).otherwise("season_id").alias("season_id")
    ).with_columns(
        pl.when(sibling).then(pl.lit("sb:9")).otherwise("competition_id").alias("competition_id"),
        pl.when(sibling).then(pl.lit("sb:27")).otherwise("season_id").alias("season_id"),
    )
    shots = shots.sort("shot_id", descending=True)  # input order is kept, whatever it is
    in_pair = (pl.col("competition_id") == "sb:2") & (pl.col("season_id") == "sb:27")
    if case == "ok":
        assert data.HOLDOUT == ("sb:2", "sb:27")
        dev, held = data.split_holdout(shots)
        assert held.equals(shots.filter(in_pair))
        assert dev.equals(shots.filter(~in_pair))
        assert held.height == 18
        assert set(dev["shot_id"]) >= {"sb:s-m01-00", "sb:s-m01-01", "sb:s-m03-00", "sb:s-m03-01"}
    elif case == "empty holdout":
        with pytest.raises(ValueError, match="(?=.*sb:99)(?=.*sb:5)"):
            data.split_holdout(shots, ("sb:99", "sb:5"))
    else:
        with pytest.raises(ValueError, match="(?=.*sb:2)(?=.*sb:27)"):
            data.split_holdout(shots.filter(in_pair))


def test_load_xg_data_freeze_frames_kept_only(tmp_path: Path, xg_data: Any) -> None:
    """``XgData.freeze_frames`` is always a C3 freeze-frame DataFrame: the snapshot's frame rows
    of the kept shots only, in snapshot order. Excluded shots that have frame rows (a direct
    corner, an in-game penalty) lose them."""
    import ablation_cases as ac

    manifest = ac.write_ablation_snapshot(tmp_path)
    xg = data.load_xg_data(manifest)
    ff = xg.freeze_frames
    assert isinstance(ff, pl.DataFrame)
    assert ff.schema == schema.FREEZE_FRAME_SCHEMA
    raw = snapshot.load_snapshot(manifest, tables=("freeze_frames",))["freeze_frames"]
    assert raw.filter(pl.col("shot_id").is_in([ac.DIRECT_CORNER_ID, ac.PENALTY_ID])).height > 0
    reasons = dict(zip(xg.exclusions["shot_id"], xg.exclusions["reason"], strict=True))
    assert reasons == {ac.DIRECT_CORNER_ID: "direct_corner", ac.PENALTY_ID: "penalty"}
    kept = xg.shots["shot_id"]
    assert ff.equals(raw.filter(pl.col("shot_id").is_in(kept.implode())))
    assert set(ff["shot_id"]) == set(kept)
    # the shared fixture: one keeper row per kept shot, none for the four excluded shots
    assert xg_data.freeze_frames.height == xc.KEPT_SHOTS
    assert set(xg_data.freeze_frames["shot_id"]).isdisjoint(xc.EXCLUDED_IDS)
