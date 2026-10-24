# Data

## Sources
| Source | Why | Notes |
|---|---|---|
| StatsBomb open data | Public, reproducible; ships StatsBomb's own xG for benchmarking and shot freeze frames (player positions at the moment of the shot) | Limited set of competitions |
| WhoScored / Opta snapshot | Large volume (~4k matches in the local engine), current leagues | Private; no provider xG; no player positions |

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
