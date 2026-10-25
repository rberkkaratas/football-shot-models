# Data

## Sources
| Source | Why | Notes |
|---|---|---|
| StatsBomb open data | Public, reproducible; ships StatsBomb's own xG for benchmarking and shot freeze frames (player positions at the moment of the shot) | Limited set of competitions |
| WhoScored / Opta CSV archive | 4,025 matches, 11 competitions, 2025-26 and 2026-27 (~104k shots); a season holdout is possible | Private; no provider xG; no player positions; one duplicated match and varying columns handled by `football_core.io.whoscored` |

## Shot table (planned)
One row per shot, in the shared 105 x 68 m frame (see the football-core conventions,
`../football-core/docs/conventions.md`).

| Column | Description |
|---|---|
| `shot_id`, `match_id`, `team_id`, `player_id` | Provider-prefixed identifiers |
| `x`, `y` | Shot location |
| `body_part` | foot (left/right), head, other |
| `situation` | open play, counter, corner, free kick (direct / indirect), penalty, throw-in |
| `assist_type` | cross, through ball, cut-back, none… (where available) |
| `goal_y`, `goal_z` | Goal-mouth end location (phase 2) |
| `outcome` | goal, saved, off target, blocked, post |
| `is_goal` | Target for xG |

## Filters (xG population, M1.1)
`football_shot_models.data` removes shots from the xG population with a logged reason, never
silently. The reasons are checked in this order and the first match wins:

| Reason | Why | StatsBomb open data |
|---|---|---|
| `own_goal` | Own Goal Against events are not shots and are 100 % goals | 387 |
| `shootout` | Period 5 is not match play | 363 |
| `penalty` | In-game penalties get their own model (M1.5) | 1,118 |
| `no_freeze_frame` | Every M1.x model must see the same shots | 0 |

That leaves 99,746 non-penalty shots in periods 1–4 (extra time, direct free kicks and corners
included). The per-shot log is the git-ignored snapshot `xg-exclusions-v1`; only its manifest
(counts per competition-season) is committed.

**Hold-out:** Premier League 2015/16 (380 matches, 9,817 shots), one of the four complete
2015/16 men's league seasons; its three sibling leagues stay in dev (3,581 matches, 89,929
shots).

**Match families** (card segments only, never features), derived from the matches table:
`women`; `team-selected`, where one team plays at least half the matches of a
competition-season (the Barcelona-only La Liga seasons); otherwise `league or tournament`.

Provider fields that leak the outcome (post-shot columns, the subjective `big_chance` flag)
are never xG features; the leakage guard is described in [Methodology](methodology.md).
