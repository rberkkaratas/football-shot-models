"""The xG modelling population: a provider snapshot (StatsBomb or WhoScored), minus the shots
xG must not see.

xG answers "how likely is this shot to go in, from where and how it was struck". Some shots
do not belong in that question and are removed with a logged reason, never silently:

- **own goals** are 100 % goals by definition, so keeping them would leak the label
  (StatsBomb logs them as Own Goal Against events; Opta marks the goal with q28);
- **shoot-out kicks** (period 5) are not match play;
- **in-game penalties** get their own model (M1.5, ``XgData.penalties``);
- **direct corners** are selected on outcome: StatsBomb logs a corner as a shot mostly when it
  threatens goal (all 33 sit on the goal line's corner arc and 18 are goals), and Opta tags
  them with q263;
- **shots without a freeze frame**, so that every M1.x model sees the same shots. Whether a
  provider records frames is a constant per snapshot builder (:data:`FRAMES_BY_BUILDER`):
  StatsBomb does, WhoScored does not, so for WhoScored this reason is switched off, never
  inferred from an empty table.

The reasons are checked in that order and the first match wins, so they partition the
excluded shots. The per-shot log is stored as the git-ignored snapshot ``xg-exclusions-v1``;
only its manifest (counts per competition-season) is committed.

The hold-out is one value, a :class:`Holdout` (a row expression and the cards' label), built
once per run and handed to every reader that splits a frame: :func:`split_holdout`, the penalty
rows and the card context. StatsBomb holds out one competition-season (:func:`pair_holdout`);
a provider with seasons in time order holds out whole seasons (:func:`season_holdout`), with
every train match before every test match.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import polars as pl
from football_core import schema
from football_core.io import snapshot

StrPath = str | os.PathLike[str]

DEFAULT_MANIFEST: Final[str] = "manifests/statsbomb-open-v1.json"
HOLDOUT: Final[tuple[str, str]] = ("sb:2", "sb:27")  # Premier League 2015/2016
EXCLUSIONS_SNAPSHOT: Final[str] = "xg-exclusions-v1"
TEAM_SELECTED_SHARE: Final[float] = 0.5
FAMILIES: Final = ("league or tournament", "team-selected", "women")
FAMILY: Final = pl.Enum(FAMILIES)
EXCLUSION_REASONS: Final = ("own_goal", "shootout", "penalty", "direct_corner", "no_freeze_frame")
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
MATCH_FEATURES: Final = schema.column_names("matches", roles="pre_shot")  # joined onto shots
# Does the builder's provider record freeze frames? Without them no_freeze_frame is not a reason.
FRAMES_BY_BUILDER: Final[Mapping[str, bool]] = MappingProxyType(
    {"statsbomb": True, "whoscored": False}
)
_PAIR: Final = ("competition_id", "season_id")


@dataclass(frozen=True)
class XgData:
    """The loaded xG population.

    Attributes
    ----------
    shots : polars.DataFrame
        Kept shots: the C3 columns, then the matches ``pre_shot`` columns
        (``MATCH_FEATURES``, today ``gender``), then ``family`` (a segment, never a feature),
        sorted by ``shot_id``.
    freeze_frames : polars.DataFrame
        The snapshot's freeze-frame rows of the kept shots only (C3 freeze-frame columns), in
        snapshot order; always a DataFrame, empty for a provider without frames.
    exclusions : polars.DataFrame
        One ``EXCLUSION_SCHEMA`` row per excluded shot, sorted by ``shot_id``.
    penalties : polars.DataFrame
        The excluded shots whose reason is ``penalty`` (in-game) or ``shootout``, for the
        penalty model: the C3 columns (``player_id`` included), then ``reason``,
        ``MATCH_FEATURES`` and ``family``, sorted by ``shot_id``; empty, with the same
        columns, when there are none.
    matches : polars.DataFrame
        The snapshot's matches table, unchanged.
    manifest : football_core.io.snapshot.Manifest
        The manifest the data was verified against.
    """

    shots: pl.DataFrame
    freeze_frames: pl.DataFrame
    exclusions: pl.DataFrame
    penalties: pl.DataFrame
    matches: pl.DataFrame
    manifest: snapshot.Manifest


def exclusion_reason(frame_shot_ids: pl.Series | None) -> pl.Expr:
    """First matching reason, in EXCLUSION_REASONS order; null for a kept shot.

    Parameters
    ----------
    frame_shot_ids : polars.Series or None
        The ``shot_id`` values that have freeze-frame rows; ``None`` for a provider without
        freeze frames, which drops the ``no_freeze_frame`` reason (an empty Series keeps it and
        tags every shot that reaches it).

    Returns
    -------
    polars.Expr
        An ``EXCLUSION_REASON`` expression over ``is_own_goal``, ``period``, ``is_penalty``,
        ``situation`` and ``shot_id``.

    Examples
    --------
    >>> shots = pl.DataFrame({"shot_id": ["sb:a", "sb:b", "sb:c", "sb:d", "sb:e", "sb:f"],
    ...                       "is_own_goal": [True, False, False, False, False, False],
    ...                       "period": pl.Series([1, 5, 2, 1, 1, 1], dtype=pl.Int8),
    ...                       "is_penalty": [False, True, True, False, False, False],
    ...                       "situation": ["open_play", "penalty", "penalty", "direct_corner",
    ...                                     "open_play", "open_play"]})
    >>> shots.select(exclusion_reason(pl.Series(["sb:d", "sb:f"]))).to_series().to_list()
    ['own_goal', 'shootout', 'penalty', 'direct_corner', 'no_freeze_frame', None]
    >>> shots.select(exclusion_reason(None)).to_series().to_list()
    ['own_goal', 'shootout', 'penalty', 'direct_corner', None, None]
    """
    chain = (
        pl.when(pl.col("is_own_goal"))
        .then(pl.lit("own_goal"))
        .when(pl.col("period") == 5)
        .then(pl.lit("shootout"))
        .when(pl.col("is_penalty"))
        .then(pl.lit("penalty"))
        .when(pl.col("situation") == "direct_corner")
        .then(pl.lit("direct_corner"))
    )
    if frame_shot_ids is not None:
        chain = chain.when(~pl.col("shot_id").is_in(frame_shot_ids.implode())).then(
            pl.lit("no_freeze_frame")
        )
    return chain.otherwise(None).cast(EXCLUSION_REASON)


def exclude(
    shots: pl.DataFrame, freeze_frames: pl.DataFrame | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split C3 shots into the xG population and an exclusion log.

    Parameters
    ----------
    shots : polars.DataFrame
        C3 shots.
    freeze_frames : polars.DataFrame or None
        C3 freeze frames; only their ``shot_id`` is read. ``None`` states that the provider
        records no frames, so ``no_freeze_frame`` is not a reason (no default: a caller says
        so, it never gets it by omission).

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
    ...                       "situation": ["open_play", "open_play"],
    ...                       "is_goal": [False, True]})
    >>> kept, log = exclude(shots, pl.DataFrame({"shot_id": ["sb:b"]}))
    >>> kept["shot_id"].to_list(), log.select("shot_id", "reason").rows()
    (['sb:b'], [('sb:a', 'own_goal')])
    """
    ids = None if freeze_frames is None else freeze_frames["shot_id"].unique()
    reason = exclusion_reason(ids)
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


def is_holdout(holdout: tuple[str, str] = HOLDOUT) -> pl.Expr:
    """True for rows of the held-out ``(competition_id, season_id)``: both must match.

    Examples
    --------
    >>> pairs = pl.DataFrame({"competition_id": ["sb:2", "sb:2", "sb:11"],
    ...                       "season_id": ["sb:27", "sb:44", "sb:27"]})
    >>> pairs.select(is_holdout()).to_series().to_list()
    [True, False, False]
    """
    return (pl.col("competition_id") == holdout[0]) & (pl.col("season_id") == holdout[1])


@dataclass(frozen=True, eq=False)  # dataclass eq on a pl.Expr raises; compare with `is`
class Holdout:
    """Which rows are held out, and how the cards name them.

    Attributes
    ----------
    expr : polars.Expr
        True for held-out rows; evaluated on every frame that is split (shots, penalties).
    label : str
        The cards' hold-out label, computed from the data, never typed.

    Examples
    --------
    >>> ho = Holdout(pl.col("match_id").is_in(pl.Series(["ws:9"]).implode()), "2026/2027")
    >>> holdout_mask(pl.DataFrame({"match_id": ["ws:1", "ws:9"]}), ho, "shots").to_list()
    [False, True]
    """

    expr: pl.Expr
    label: str


def pair_holdout(matches: pl.DataFrame, pair: tuple[str, str] = HOLDOUT) -> Holdout:
    """The hold-out of one ``(competition_id, season_id)`` pair, labelled from ``matches``.

    The label is ``competition_name + " " + season_name`` of the first matching match, or
    ``"n/a"`` when no match has the pair.

    Examples
    --------
    >>> matches = pl.DataFrame({"competition_id": ["sb:2"], "season_id": ["sb:27"],
    ...                         "competition_name": ["Premier League"],
    ...                         "season_name": ["2015/2016"]})
    >>> pair_holdout(matches).label, pair_holdout(matches, ("sb:9", "sb:1")).label
    ('Premier League 2015/2016', 'n/a')
    """
    named = matches.filter(is_holdout(pair)).select(
        pl.concat_str("competition_name", "season_name", separator=" ")
    )
    return Holdout(is_holdout(pair), str(named.item(0, 0)) if named.height else "n/a")


def holdout_mask(frame: pl.DataFrame, holdout: Holdout | pl.Expr, what: str) -> pl.Series:
    """The hold-out's Boolean mask on ``frame``, one value per row, never null.

    Parameters
    ----------
    frame : polars.DataFrame
        The frame to split.
    holdout : Holdout or polars.Expr
        The hold-out, or its row expression.
    what : str
        The frame's name, which starts every error message.

    Returns
    -------
    polars.Series
        Boolean, ``frame.height`` values (a literal broadcasts).

    Raises
    ------
    ValueError
        The expression reads a column the frame lacks or fails in polars (a length-changing
        expression, a dtype mismatch); the mask is not Boolean or holds a null.

    Examples
    --------
    >>> shots = pl.DataFrame({"competition_id": ["sb:2", None], "season_id": ["sb:27"] * 2})
    >>> holdout_mask(shots.head(1), is_holdout(), "shots").to_list()
    [True]
    >>> holdout_mask(shots, is_holdout(), "shots")
    Traceback (most recent call last):
    ...
    ValueError: shots: the hold-out mask has 1 null value(s)
    """
    expr = holdout.expr if isinstance(holdout, Holdout) else holdout
    try:
        mask = frame.with_columns(expr.alias("_held"))["_held"]
    except pl.exceptions.ColumnNotFoundError as e:
        raise ValueError(f"{what}: the hold-out reads a column the frame lacks ({e})") from None
    except pl.exceptions.PolarsError as e:  # ShapeError, ComputeError, InvalidOperationError
        kind = type(e).__name__
        raise ValueError(f"{what}: the hold-out expression fails ({kind}: {e})") from None
    if mask.dtype != pl.Boolean():
        raise ValueError(f"{what}: the hold-out mask is {mask.dtype}, not Boolean")
    if nulls := mask.null_count():
        raise ValueError(f"{what}: the hold-out mask has {nulls} null value(s)")
    return mask


def season_holdout(matches: pl.DataFrame, train: Sequence[str], test: Sequence[str]) -> Holdout:
    """Hold out the matches of the ``test`` seasons, after checking the time order.

    Every match must belong to a listed season, so the development data is exactly the
    ``train`` seasons: a season added to the archive later is an error, never silent training
    data. The last train match must be strictly before the first test match.

    Parameters
    ----------
    matches : polars.DataFrame
        The matches table (``match_id``, ``season_name``, ``match_date``).
    train, test : sequence of str
        ``season_name`` values, non-empty and disjoint.

    Returns
    -------
    Holdout
        ``match_id`` in the test matches; labelled by the test season names present, sorted
        and joined by ``", "``.

    Raises
    ------
    ValueError
        A bare ``str`` or an empty sequence; a season in both; a ``season_name`` outside
        ``train`` and ``test``, or null; a null ``match_date``; a listed season with no match;
        a last train date not before the first test date (both dates in the message).

    Examples
    --------
    >>> from datetime import date
    >>> m = pl.DataFrame({"match_id": ["ws:1", "ws:2", "ws:3"],
    ...                   "season_name": ["2025/2026", "2026", "2026/2027"],
    ...                   "match_date": [date(2025, 8, 2), date(2026, 7, 4), date(2026, 8, 7)]})
    >>> ho = season_holdout(m, ("2025/2026", "2026"), ("2026/2027",))
    >>> ho.label, m.select(ho.expr).to_series().to_list()
    ('2026/2027', [False, False, True])
    >>> season_holdout(m, ("2026",), ("2025/2026", "2026/2027"))
    Traceback (most recent call last):
    ...
    ValueError: the last train match (2026-07-04) is not before the first test match (2025-08-02)
    """
    for name, seasons in (("train", train), ("test", test)):
        if isinstance(seasons, str) or not seasons:
            raise ValueError(
                f"{name} must be a non-empty sequence of season names, got {seasons!r}"
            )
    if both := sorted(set(train) & set(test)):
        raise ValueError(f"season(s) {both} are in both train and test")
    names = matches["season_name"]
    if other := sorted(set(names.drop_nulls()) - set(train) - set(test)):
        raise ValueError(
            f"matches hold season(s) {other} outside train {list(train)} and test {list(test)}"
        )
    if nulls := names.null_count():
        raise ValueError(f"season_name has {nulls} null value(s)")
    if nulls := matches["match_date"].null_count():
        raise ValueError(f"match_date has {nulls} null value(s); the date order needs every date")
    held = names.is_in(list(test))
    for side, seasons, rows in (
        ("train", train, matches.filter(~held)),
        ("test", test, matches.filter(held)),
    ):
        if missing := sorted(set(seasons) - set(rows["season_name"])):
            raise ValueError(f"{side} season(s) {missing} have no match")
    days = matches["match_date"].cast(pl.Int32).to_numpy()  # physical days: total on any Date
    mask = held.to_numpy()
    last, first = days[~mask].max(), days[mask].min()
    if not last < first:
        last_d, first_d = (str(np.datetime64(int(d), "D")) for d in (last, first))
        raise ValueError(
            f"the last train match ({last_d}) is not before the first test match ({first_d})"
        )
    test_ids = matches.filter(held)["match_id"]
    label = ", ".join(sorted(set(matches.filter(held)["season_name"])))
    return Holdout(pl.col("match_id").is_in(test_ids.implode()), label)


def split_holdout(
    shots: pl.DataFrame, holdout: tuple[str, str] | Holdout = HOLDOUT
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split shots into development data and the held-out rows.

    Parameters
    ----------
    shots : polars.DataFrame
        Shots with the columns the hold-out reads.
    holdout : tuple of str or Holdout, default HOLDOUT
        A ``(competition_id, season_id)`` pair to hold out, or a :class:`Holdout`.

    Returns
    -------
    dev, held : polars.DataFrame
        The other shots and the held-out ones, each in input order.

    Raises
    ------
    ValueError
        Either side is empty (the message names the pair or the label); a null hold-out key
        or another :func:`holdout_mask` error (a row is never dropped from both sides).

    Examples
    --------
    >>> shots = pl.DataFrame({"competition_id": ["sb:2", "sb:9"], "season_id": ["sb:27"] * 2})
    >>> [f.height for f in split_holdout(shots)]
    [1, 1]
    """
    ho = holdout if isinstance(holdout, Holdout) else Holdout(is_holdout(holdout), repr(holdout))
    mask = holdout_mask(shots, ho, "shots")
    dev, held = shots.filter(~mask), shots.filter(mask)
    for side, frame in (("hold-out", held), ("development", dev)):
        if frame.height == 0:
            raise ValueError(f"hold-out {ho.label}: the {side} set is empty")
    return dev, held


def load_xg_data(manifest: StrPath = DEFAULT_MANIFEST, *, builder: str = "statsbomb") -> XgData:
    """Verify and load a snapshot, apply the exclusions and add match columns.

    Shot/match consistency is not rechecked here: ``write_snapshot`` validated the shots
    against the matches table, and the manifest pins those files.

    Parameters
    ----------
    manifest : str or path, default DEFAULT_MANIFEST
        Manifest of a snapshot written by ``builder``.
    builder : str, default "statsbomb"
        The snapshot builder, a key of :data:`FRAMES_BY_BUILDER`, which says whether
        ``no_freeze_frame`` is an exclusion reason. The snapshot holds a ``freeze_frames``
        table either way (empty for a provider without frames).

    Returns
    -------
    XgData
        The kept shots (with ``MATCH_FEATURES`` and ``family``), their freeze-frame rows, the
        exclusion log, the in-game and shoot-out penalties (the same match columns, plus
        ``reason``), the matches and the manifest.

    Raises
    ------
    ValueError
        A ``builder`` not in :data:`FRAMES_BY_BUILDER` (checked before the manifest is read);
        a snapshot written by another builder.
    football_core.io.snapshot.ManifestError, football_core.io.snapshot.SnapshotVerificationError
        From core, unchanged: a bad manifest, or a data file that does not hash as recorded.

    Examples
    --------
    >>> data = load_xg_data()  # doctest: +SKIP
    >>> data.shots.height, data.freeze_frames.height  # doctest: +SKIP
    (99713, 1304989)
    >>> data.penalties.group_by("reason").len().sort("reason").rows()  # doctest: +SKIP
    [('shootout', 363), ('penalty', 1118)]
    """
    if builder not in FRAMES_BY_BUILDER:
        raise ValueError(
            f"unknown builder {builder!r}; expected one of {sorted(FRAMES_BY_BUILDER)}"
        )
    meta = snapshot.read_manifest(manifest)
    if meta.builder != builder:
        raise ValueError(
            f"snapshot {meta.name!r} was written by builder {meta.builder!r}; expected {builder!r}"
        )
    tables = snapshot.load_snapshot(manifest, tables=("shots", "freeze_frames", "matches"))
    matches = tables["matches"]
    frames_or_none = tables["freeze_frames"] if FRAMES_BY_BUILDER[builder] else None
    kept, log = exclude(tables["shots"], frames_or_none)
    # m:1 keeps a hand-built snapshot with a duplicated match_id from multiplying shots.
    segments = matches.select("match_id", *MATCH_FEATURES).join(
        match_family(matches), on="match_id"
    )
    shots = kept.join(segments, on="match_id", how="inner", validate="m:1", maintain_order="left")
    frames = tables["freeze_frames"].join(
        shots.select("shot_id"), on="shot_id", how="semi", maintain_order="left"
    )
    kicks = log.filter(pl.col("reason").is_in(["penalty", "shootout"])).select("shot_id", "reason")
    penalties = (
        tables["shots"]
        .join(kicks, on="shot_id", how="inner", validate="1:1")
        .join(segments, on="match_id", how="inner", validate="m:1")
        .sort("shot_id")
    )
    return XgData(
        shots=shots,
        freeze_frames=frames,
        exclusions=log,
        penalties=penalties,
        matches=matches,
        manifest=meta,
    )


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
