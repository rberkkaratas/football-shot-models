"""The xG modelling population: the StatsBomb snapshot, minus the shots xG must not see.

xG answers "how likely is this shot to go in, from where and how it was struck". Some shots
do not belong in that question and are removed with a logged reason, never silently:

- **own goals** are 100 % goals by definition (StatsBomb logs them as Own Goal Against
  events, not shots), so keeping them would leak the label;
- **shoot-out kicks** (period 5) are not match play;
- **in-game penalties** get their own model later (M1.5);
- **shots without a freeze frame**, so that every M1.x model sees the same shots.

The reasons are checked in that order and the first match wins, so they partition the
excluded shots. The per-shot log is stored as the git-ignored snapshot ``xg-exclusions-v1``;
only its manifest (counts per competition-season) is committed.

Every match also gets a ``family`` used to segment the model cards: ``women``,
``team-selected`` (one team plays in at least half the matches of the competition-season) or
``league or tournament``. It is derived from the matches table alone, with no hand-made list
of competitions. Team-selected covers the Barcelona-only La Liga seasons and other
one-club collections, and also, trivially, every one- or two-match competition-season: both
teams then play in all of its matches. At ``statsbomb-open-v1`` that is 30 of the 56
team-selected competition-seasons, mostly single finals (18 Champions League finals, three
World Cup and three Copa del Rey matches), so the family means "curated matches", not
"one club's season".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import polars as pl
from football_core.io import snapshot

StrPath = str | os.PathLike[str]

DEFAULT_MANIFEST: Final[str] = "manifests/statsbomb-open-v1.json"
HOLDOUT: Final[tuple[str, str]] = ("sb:2", "sb:27")  # Premier League 2015/2016
EXCLUSIONS_SNAPSHOT: Final[str] = "xg-exclusions-v1"
TEAM_SELECTED_SHARE: Final[float] = 0.5
FAMILIES: Final = ("league or tournament", "team-selected", "women")
FAMILY: Final = pl.Enum(FAMILIES)
EXCLUSION_REASONS: Final = ("own_goal", "shootout", "penalty", "no_freeze_frame")
EXCLUSION_REASON: Final = pl.Enum(EXCLUSION_REASONS)
EXCLUSION_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "shot_id": pl.String(),
        "match_id": pl.String(),
        "competition_id": pl.String(),
        "season_id": pl.String(),
        "reason": EXCLUSION_REASON,
        "is_goal": pl.Boolean(),
    }
)
_PAIR: Final = ("competition_id", "season_id")


@dataclass(frozen=True)
class XgData:
    """The loaded xG population.

    Attributes
    ----------
    shots : polars.DataFrame
        Kept shots: the C3 columns plus ``gender`` and ``family`` (segments only, never
        features), sorted by ``shot_id``.
    exclusions : polars.DataFrame
        One ``EXCLUSION_SCHEMA`` row per excluded shot, sorted by ``shot_id``.
    matches : polars.DataFrame
        The snapshot's matches table, unchanged.
    manifest : football_core.io.snapshot.Manifest
        The manifest the data was verified against.
    """

    shots: pl.DataFrame
    exclusions: pl.DataFrame
    matches: pl.DataFrame
    manifest: snapshot.Manifest


def exclusion_reason(frame_shot_ids: pl.Series) -> pl.Expr:
    """First matching reason, in EXCLUSION_REASONS order; null for a kept shot.

    Parameters
    ----------
    frame_shot_ids : polars.Series
        The ``shot_id`` values that have freeze-frame rows.

    Returns
    -------
    polars.Expr
        An ``EXCLUSION_REASON`` expression over ``is_own_goal``, ``period``, ``is_penalty``
        and ``shot_id``.

    Examples
    --------
    >>> shots = pl.DataFrame({"shot_id": ["sb:a", "sb:b", "sb:c", "sb:d", "sb:e"],
    ...                       "is_own_goal": [True, False, False, False, False],
    ...                       "period": pl.Series([1, 5, 2, 1, 1], dtype=pl.Int8),
    ...                       "is_penalty": [False, True, True, False, False]})
    >>> shots.select(exclusion_reason(pl.Series(["sb:e"]))).to_series().to_list()
    ['own_goal', 'shootout', 'penalty', 'no_freeze_frame', None]
    """
    return (
        pl.when(pl.col("is_own_goal"))
        .then(pl.lit("own_goal"))
        .when(pl.col("period") == 5)
        .then(pl.lit("shootout"))
        .when(pl.col("is_penalty"))
        .then(pl.lit("penalty"))
        .when(~pl.col("shot_id").is_in(frame_shot_ids.implode()))
        .then(pl.lit("no_freeze_frame"))
        .otherwise(None)
        .cast(EXCLUSION_REASON)
    )


def exclude(shots: pl.DataFrame, freeze_frames: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split C3 shots into the xG population and an exclusion log.

    Parameters
    ----------
    shots : polars.DataFrame
        C3 shots.
    freeze_frames : polars.DataFrame
        C3 freeze frames; only their ``shot_id`` is read.

    Returns
    -------
    kept, log : polars.DataFrame
        ``kept`` has the input columns and the shots with no reason; ``log`` has one
        ``EXCLUSION_SCHEMA`` row per excluded shot. Both are sorted by ``shot_id``.

    Examples
    --------
    >>> shots = pl.DataFrame({"shot_id": ["sb:b", "sb:a"], "match_id": ["sb:1", "sb:1"],
    ...                       "competition_id": ["sb:2", "sb:2"], "season_id": ["sb:27"] * 2,
    ...                       "period": pl.Series([1, 1], dtype=pl.Int8),
    ...                       "is_own_goal": [False, True], "is_penalty": [False, False],
    ...                       "is_goal": [False, True]})
    >>> kept, log = exclude(shots, pl.DataFrame({"shot_id": ["sb:b"]}))
    >>> kept["shot_id"].to_list(), log.select("shot_id", "reason").rows()
    (['sb:b'], [('sb:a', 'own_goal')])
    """
    reason = exclusion_reason(freeze_frames["shot_id"].unique())
    tagged = shots.with_columns(_reason=reason).sort("shot_id")
    kept = tagged.filter(pl.col("_reason").is_null()).drop("_reason")
    log = tagged.filter(pl.col("_reason").is_not_null()).select(
        *(pl.col(c) for c in EXCLUSION_SCHEMA if c != "reason"), reason=pl.col("_reason")
    )
    return kept, log.select(EXCLUSION_SCHEMA.names())


def match_family(matches: pl.DataFrame) -> pl.DataFrame:
    """The segment family of each match: women, team-selected, or league or tournament.

    Parameters
    ----------
    matches : polars.DataFrame
        The snapshot's matches table (``match_id``, ``competition_id``, ``season_id``,
        ``home_team_id``, ``away_team_id``, ``gender``).

    Returns
    -------
    polars.DataFrame
        ``match_id`` (String) and ``family`` (``FAMILY``), one row per match, sorted by
        ``match_id``.

    Examples
    --------
    >>> matches = pl.DataFrame({"match_id": ["sb:1", "sb:2", "sb:3", "sb:4"],
    ...                         "competition_id": ["sb:9", "sb:9", "sb:8", "sb:7"],
    ...                         "season_id": ["sb:1"] * 4,
    ...                         "home_team_id": ["sb:t1", "sb:t3", "sb:t1", "sb:t1"],
    ...                         "away_team_id": ["sb:t2", "sb:t4", "sb:t2", "sb:t2"],
    ...                         "gender": ["male", "male", "male", "female"]})
    >>> match_family(matches)["family"].to_list()
    ['team-selected', 'team-selected', 'team-selected', 'women']
    """
    teams = pl.concat(
        [
            matches.select(*_PAIR, "match_id", team=pl.col("home_team_id")),
            matches.select(*_PAIR, "match_id", team=pl.col("away_team_id")),
        ]
    )
    top_share = (
        teams.group_by(*_PAIR, "team")
        .agg(n=pl.col("match_id").n_unique())
        .join(matches.group_by(_PAIR).agg(total=pl.len()), on=list(_PAIR))
        .group_by(_PAIR)
        .agg(top_share=(pl.col("n") / pl.col("total")).max())
    )
    family = (
        pl.when(pl.col("gender") == "female")
        .then(pl.lit("women"))
        .when(pl.col("top_share") >= TEAM_SELECTED_SHARE)
        .then(pl.lit("team-selected"))
        .otherwise(pl.lit("league or tournament"))
        .cast(FAMILY)
    )
    return (
        matches.join(top_share, on=list(_PAIR), how="left")
        .select("match_id", family=family)
        .sort("match_id")
    )


def split_holdout(
    shots: pl.DataFrame, holdout: tuple[str, str] = HOLDOUT
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split shots into development data and one held-out competition-season.

    Parameters
    ----------
    shots : polars.DataFrame
        Shots with ``competition_id`` and ``season_id``.
    holdout : tuple of str, default HOLDOUT
        The ``(competition_id, season_id)`` to hold out.

    Returns
    -------
    dev, held : polars.DataFrame
        The other shots and the held-out ones, each in input order.

    Raises
    ------
    ValueError
        If either side is empty; the message names the pair.

    Examples
    --------
    >>> shots = pl.DataFrame({"competition_id": ["sb:2", "sb:9"], "season_id": ["sb:27"] * 2})
    >>> [f.height for f in split_holdout(shots)]
    [1, 1]
    """
    is_held = (pl.col("competition_id") == holdout[0]) & (pl.col("season_id") == holdout[1])
    dev, held = shots.filter(~is_held), shots.filter(is_held)
    for side, frame in (("hold-out", held), ("development", dev)):
        if frame.height == 0:
            raise ValueError(f"hold-out {holdout}: the {side} set is empty")
    return dev, held


def load_xg_data(manifest: StrPath = DEFAULT_MANIFEST) -> XgData:
    """Verify and load the StatsBomb snapshot, apply the exclusions and add the segments.

    Parameters
    ----------
    manifest : str or path, default DEFAULT_MANIFEST
        Manifest of a snapshot written by the ``statsbomb`` builder.

    Returns
    -------
    XgData
        The kept shots (with ``gender`` and ``family``), the exclusion log, the matches and
        the manifest.

    Raises
    ------
    ValueError
        Another builder; kept shots without a matches row; duplicate ``match_id`` in matches.
    football_core.io.snapshot.ManifestError, football_core.io.snapshot.SnapshotVerificationError
        From core, unchanged: a bad manifest, or a data file that does not hash as recorded.

    Examples
    --------
    >>> data = load_xg_data()  # doctest: +SKIP
    >>> data.shots.height  # doctest: +SKIP
    99746
    """
    meta = snapshot.read_manifest(manifest)
    if meta.builder != "statsbomb":
        raise ValueError(
            f"snapshot {meta.name!r} was written by builder {meta.builder!r}; expected 'statsbomb'"
        )
    tables = snapshot.load_snapshot(manifest, tables=("shots", "freeze_frames", "matches"))
    matches = tables["matches"]
    kept, log = exclude(tables["shots"], tables["freeze_frames"])
    orphans = kept.filter(~pl.col("match_id").is_in(matches["match_id"].implode())).height
    if orphans:
        raise ValueError(f"{orphans} kept shot(s) without a matches row")
    ids = matches["match_id"]
    duplicates = ids.filter(ids.is_duplicated()).n_unique()
    if duplicates:
        raise ValueError(f"{duplicates} duplicate match_id(s) in matches")
    segments = matches.select("match_id", "gender").join(match_family(matches), on="match_id")
    shots = kept.join(segments, on="match_id", how="inner", maintain_order="left")
    return XgData(shots=shots, exclusions=log, matches=matches, manifest=meta)


def write_exclusion_log(
    log: pl.DataFrame, manifest: StrPath, *, root: StrPath = "."
) -> snapshot.SnapshotResult:
    """Store the exclusion log as the git-ignored snapshot ``xg-exclusions-v1``.

    The source is the StatsBomb snapshot's name and data digest, so the log's manifest pins
    which data it was derived from, and a rerun after a core commit (which changes only the
    StatsBomb manifest's ``environment``) leaves it unchanged. ``write_snapshot`` refuses to
    write data that git would track.

    Parameters
    ----------
    log : polars.DataFrame
        ``EXCLUSION_SCHEMA`` rows (from :func:`exclude`).
    manifest : str or path
        The StatsBomb manifest the log was derived from (read for its name and digest).
    root : str or path, default "."
        The repo root; data goes to ``data/snapshots/``, the manifest to ``manifests/``.

    Returns
    -------
    football_core.io.snapshot.SnapshotResult
        ``unchanged`` on a rerun over the same data.

    Raises
    ------
    ValueError
        From core, when ``data/`` is not git-ignored.
    OSError
        On a filesystem error.

    Examples
    --------
    >>> write_exclusion_log(load_xg_data().exclusions, DEFAULT_MANIFEST).status  # doctest: +SKIP
    'unchanged'
    """
    meta = snapshot.read_manifest(manifest)
    source = snapshot.Source("files", uri=meta.name, revision=meta.data_digest)
    return snapshot.write_snapshot(
        EXCLUSIONS_SNAPSHOT,
        {"exclusions": log},
        source,
        root=root,
        sort_by={"exclusions": ("shot_id",)},
        overwrite=True,
    )
