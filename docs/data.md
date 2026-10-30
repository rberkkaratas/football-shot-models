# Data

## Sources
| Source | Why | Notes |
|---|---|---|
| StatsBomb open data | Public, reproducible; ships StatsBomb's own xG for benchmarking and shot freeze frames (player positions at the moment of the shot) | Limited set of competitions |
| WhoScored / Opta CSV archive | 4,025 event files, 11 competitions, 2025-26 and 2026-27; the `whoscored-v1` snapshot keeps 3,802 matches and 98,261 shots | Private; no provider xG; no player positions; one duplicated match and varying columns handled by `football_core.io.whoscored` |

## Shot table
The models read the canonical shot and freeze-frame tables of football-core: one row per
shot in the shared 105 x 68 m frame, every column with a dtype, nullability, allowed values and
a role (`pre_shot`, `outcome`, `post_shot`, …). The column list lives in one place,
`football_core.schema` (see football-core's schema docs and
`../football-core/docs/conventions.md`), and the leakage guard reads the roles from it. Each
shot is joined to its row of the canonical matches table (schema 1.1), whose only `pre_shot`
column, `gender`, the guard allows (`data.MATCH_FEATURES`).

## Filters (xG population)
`football_shot_models.data` removes shots from the xG population with a logged reason, never
silently. The reasons are checked in this order and the first match wins:

| Reason | Why | StatsBomb open data | WhoScored (`whoscored-v1`) |
|---|---|---|---|
| `own_goal` | Own goals are not the shooter's attempts and are 100 % goals | 387 | 330 |
| `shootout` | Period 5 is not match play | 363 | 20 |
| `penalty` | In-game penalties get their own model | 1,118 | 1,063 |
| `direct_corner` | A direct corner is logged as a shot | 33 | 13 |
| `no_freeze_frame` | Every StatsBomb xG model must see the same shots | 0 | n/a (no frames) |

That leaves 99,713 non-penalty shots in periods 1–4 (extra time, direct free kicks and corners
included). The per-shot log is the git-ignored snapshot `xg-exclusions-v1`; only its manifest
(counts per competition-season) is committed.

**Hold-out:** Premier League 2015/16 (380 matches, 9,816 shots), one of the four full 2015/16
men's league seasons (Ligue 1: 377 of 380 matches in the snapshot); the other three stay in dev
(3,581 matches, 89,897 shots). The penalty model uses the excluded `penalty` shots (1,118 kicks)
and reports the `shootout` kicks (363) apart.

**Match families** (card segments only, never features), derived from the matches table:
`women`; `team-selected`, where one team plays at least half the matches of a
competition-season; otherwise `league or tournament`. Team-selected holds the Barcelona-only La
Liga seasons and other one-club collections, and also every one- or two-match
competition-season, where both teams qualify trivially: 30 of its 56 competition-seasons, mostly
single finals (18 Champions League finals, three World Cup and three Copa del Rey matches).

## WhoScored xG v1
The private archive becomes the `whoscored-v1` snapshot through football-core's
`io.whoscored` builder: `make snapshot-whoscored WHOSCORED_ARCHIVE=<archive>/data`. The parquet
files go to `data/snapshots/whoscored-v1/` and the manifest to `manifests/whoscored-v1.json`;
both are git-ignored (the manifest holds per competition-season counts and dates), and the
target refuses to run unless git ignores the manifest. The legacy v0 xG, the benchmark,
is copied by `make legacy-v0 LEGACY_V0=<path>/xg_model.json` to `data/legacy/xg_model.json`
(git-ignored) and read only behind its SHA-256 pin (`legacy.V0_SHA256`).

The v1 population is the non-penalty shots kept by the filters above (96,835), less 7 shots
whose assisting event (Opta q55) cannot be resolved: 96,828. The split is by season
(`data.season_holdout`): train is every match of 2025/2026 and of the calendar-year World Cup
2026 (3,215 matches, 80,664 shots), test every match of 2026/2027 (587 matches, 16,164 shots);
the last train match is checked to precede the first test match, and v0's own training date
to precede it too. The [xG v1 card](models/xg-v1.md) carries counts and metrics only: no
match, team, player, date or file name.

The prediction cache, `data/cache/<data digest>-<code hash>/`, is git-ignored as well: it
holds per-shot predictions, so it is data too.

Provider fields that leak the outcome (post-shot columns, the subjective `big_chance` flag)
are never xG features; the leakage guard is described in [Methodology](methodology.md).
