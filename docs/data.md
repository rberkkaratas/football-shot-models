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

## Filters (initial)
- Own goals are removed (not shots).
- Penalties are kept but modelled separately.
- Provider-specific fields that leak the outcome (e.g. subjective "big chance" flags) are
  evaluated explicitly before being used as features; see [Methodology](methodology.md).
