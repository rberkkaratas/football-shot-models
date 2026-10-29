# Model card: Goals saved above expected (GSAx)

## Overview
- **Name / version:** `ws-gsax` v2
- **Task:** a keeper's record of on-target shots faced: Σ q − goals, q the xGOT rated against the league sum (P3.1)
- **Status:** experimental
- **Source(s):** private WhoScored / Opta event archive; snapshot `whoscored-v1`, data digest `c394a94944cdc12ee4d6bdd77594c9156a51aa3731955905b46ec80f648bb63e`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
A record of shots faced, rated against the league, not a skill rating: GSAx says how many fewer goals a keeper conceded than a league-average keeper facing the same on-target shots, given where they crossed the line (positive is better). It credits the keeper and the defence in front of him; the stability section says what it allows beyond that. For research; not for other providers' data without re-fitting.

## Data
- **Rated rows:** 31,471 on-target open-play shots (9,537 goals) and 993 in-game penalties (834 goals), faced by 473 keepers in 674 keeper-seasons and 20 competition-seasons.
- **Not rated:** 16 shoot-out kicks (reported apart by the penalty xGOT); 9 on-target direct corners; own goals are never on target, so they are in no keeper's record. Goals in the tables are goals from rated shots only, not goals conceded.
- **Open goals** (`open_goal`, kept in the population, PM X1): not recorded.
- **Small competition-seasons** (at most 2 keepers or under 50 shots): 0 rated at δ = 0 and flagged (PM E2 (b)), with 0 keeper-seasons; 0 competition-seasons have at most 2 keepers.

## Method

Every row's xGOT comes from fits (xG input and xGOT) that never saw its season (`season_out.season_out`, source `season_folds`: folds grouped by competition-season, the test season from the refit's dev fit, `holdout_fit`); the match out-of-fold xGOT of the xGOT card is beside it on the same rows.

| Source | Kind | Shots | Goals | Σ xGOT | Log loss | AUC | Match-OOF log loss | Match-OOF AUC |
|---|---|---|---|---|---|---|---|---|
| holdout_fit | open_play | 5,154 | 1,614 | 1,537.7 | 0.4385 | 0.8484 | 0.4394 | 0.8475 |
| holdout_fit | penalty | 137 | 120 | 114.8 | 0.3847 | 0.5279 | 0.3847 | 0.5279 |
| season_folds | open_play | 26,317 | 7,923 | 7,934.1 | 0.4384 | 0.8429 | 0.4383 | 0.8433 |
| season_folds | penalty | 856 | 714 | 712.2 | 0.4470 | 0.5689 | 0.4444 | 0.5922 |

Rating against the league sum: one logit shift δ per rating group makes Σ q = goals, q = expit(logit xGOT + δ), so the xGOT level gap is not charged to keepers (PM after MS5). Open play has one δ per competition-season, penalties one δ per season-out source (PM 2026-10-28 (a)); a group with at most 2 keepers or under 50 shots is at δ = 0 (E2's minimum, every grouping), and so are team-selected competition-seasons. Each row is rated once, so GSAx with penalties minus without is exactly the keeper's penalty GSAx.

- open play: δ from -0.17 to +0.31 (median +0.01) over 20 competition-seasons; 0 with every shot scored or none (0 shots), where every keeper gets 0 and δ is infinite; shots per δ: min 374, median 733.5; 0 competition-seasons at δ = 0
- penalty, δ per source: holdout_fit +0.31 on 137 kicks (102 keepers; league or tournament 137)
- penalty, δ per source: season_folds +0.02 on 856 kicks (257 keepers; league or tournament 856)
- penalty at δ = 0: 0 kicks of competition-seasons at δ = 0; 0 source pool(s) failing E2's minimum (0 kicks)

### Calibration of the rated q by placement band

Open play; `mouth_y_abs` and `mouth_z` (m) as on the xGOT card, `angle_rad` the opening angle; source `all` pools the rows. A band is judged with at least 100 shots, within 2 SD (√Σ q(1 − q)). The last column rates the same rows from the xGOT card's match out-of-fold xGOT (diagnostic: it saw the rated season).

| Feature | Band | Source | Shots | Goals | Σ q − goals | SD | Within 2 SD | Match out of fold: Σ q − goals (SD) |
|---|---|---|---|---|---|---|---|---|
| mouth_y_abs | [0, 1) | all | 8,981 | 1,500 | +47.3 | 30.7 | yes | +41.4 (30.8) |
| mouth_y_abs | [1, 2) | all | 9,363 | 2,346 | +1.8 | 35.4 | yes | +4.1 (35.3) |
| mouth_y_abs | [2, 3) | all | 9,091 | 3,665 | −15.8 | 39.4 | yes | −11.1 (39.4) |
| mouth_y_abs | [3, 3.3) | all | 3,061 | 1,617 | −30.7 | 23.4 | yes | −33.2 (23.4) |
| mouth_y_abs | [3.3, 3.55) | all | 943 | 404 | −6.8 | 12.8 | yes | −5.4 (12.7) |
| mouth_y_abs | >= 3.55 | all | 32 | 5 | +4.1 | 2.2 | not judged (n < 100) | +4.3 (2.2) |
| mouth_y_abs | outside the frame + 0.15 m | all | 0 | 0 | n/a | 0.0 | not judged (n < 100) | n/a |
| mouth_y_abs | [0, 1) | holdout_fit | 1,429 | 223 | +25.2 | 12.2 | no | +25.3 (12.3) |
| mouth_y_abs | [1, 2) | holdout_fit | 1,549 | 410 | −5.6 | 14.6 | yes | −4.7 (14.6) |
| mouth_y_abs | [2, 3) | holdout_fit | 1,500 | 621 | +3.2 | 16.1 | yes | +2.3 (16.1) |
| mouth_y_abs | [3, 3.3) | holdout_fit | 517 | 285 | −12.2 | 9.5 | yes | −12.0 (9.6) |
| mouth_y_abs | [3.3, 3.55) | holdout_fit | 156 | 75 | −11.6 | 5.1 | no | −11.9 (5.1) |
| mouth_y_abs | >= 3.55 | holdout_fit | 3 | 0 | +0.9 | 0.7 | not judged (n < 100) | +0.9 (0.7) |
| mouth_y_abs | outside the frame + 0.15 m | holdout_fit | 0 | 0 | n/a | 0.0 | not judged (n < 100) | n/a |
| mouth_y_abs | [0, 1) | season_folds | 7,552 | 1,277 | +22.1 | 28.2 | yes | +16.1 (28.2) |
| mouth_y_abs | [1, 2) | season_folds | 7,814 | 1,936 | +7.4 | 32.2 | yes | +8.7 (32.2) |
| mouth_y_abs | [2, 3) | season_folds | 7,591 | 3,044 | −19.0 | 36.0 | yes | −13.4 (35.9) |
| mouth_y_abs | [3, 3.3) | season_folds | 2,544 | 1,332 | −18.5 | 21.4 | yes | −21.2 (21.4) |
| mouth_y_abs | [3.3, 3.55) | season_folds | 787 | 329 | +4.8 | 11.7 | yes | +6.5 (11.7) |
| mouth_y_abs | >= 3.55 | season_folds | 29 | 5 | +3.2 | 2.0 | not judged (n < 100) | +3.3 (2.1) |
| mouth_y_abs | outside the frame + 0.15 m | season_folds | 0 | 0 | n/a | 0.0 | not judged (n < 100) | n/a |
| mouth_z | [0, 0.3) | all | 7,796 | 2,532 | −22.6 | 34.3 | yes | −23.7 (34.3) |
| mouth_z | [0.3, 0.6) | all | 5,823 | 1,581 | +24.2 | 28.0 | yes | +24.3 (28.0) |
| mouth_z | [0.6, 1) | all | 5,638 | 1,721 | −35.1 | 27.7 | yes | −34.4 (27.7) |
| mouth_z | [1, 1.5) | all | 6,574 | 1,676 | +35.5 | 29.1 | yes | +39.2 (29.0) |
| mouth_z | [1.5, 2) | all | 4,191 | 1,533 | −1.6 | 25.9 | yes | −6.2 (25.9) |
| mouth_z | [2, 2.3) | all | 1,379 | 486 | −4.9 | 14.5 | yes | −4.0 (14.5) |
| mouth_z | >= 2.3 | all | 70 | 8 | +4.4 | 2.8 | not judged (n < 100) | +4.7 (2.8) |
| mouth_z | [0, 0.3) | holdout_fit | 1,347 | 459 | −8.9 | 14.5 | yes | −8.0 (14.5) |
| mouth_z | [0.3, 0.6) | holdout_fit | 930 | 269 | +1.7 | 11.1 | yes | +0.8 (11.1) |
| mouth_z | [0.6, 1) | holdout_fit | 909 | 298 | −20.0 | 11.2 | yes | −20.3 (11.3) |
| mouth_z | [1, 1.5) | holdout_fit | 1,067 | 265 | +10.6 | 11.7 | yes | +9.7 (11.7) |
| mouth_z | [1.5, 2) | holdout_fit | 679 | 246 | +12.2 | 10.4 | yes | +13.2 (10.5) |
| mouth_z | [2, 2.3) | holdout_fit | 210 | 76 | +2.8 | 5.7 | yes | +3.3 (5.7) |
| mouth_z | >= 2.3 | holdout_fit | 12 | 1 | +1.4 | 1.2 | not judged (n < 100) | +1.4 (1.2) |
| mouth_z | [0, 0.3) | season_folds | 6,449 | 2,073 | −13.7 | 31.1 | yes | −15.7 (31.1) |
| mouth_z | [0.3, 0.6) | season_folds | 4,893 | 1,312 | +22.5 | 25.6 | yes | +23.6 (25.7) |
| mouth_z | [0.6, 1) | season_folds | 4,729 | 1,423 | −15.1 | 25.3 | yes | −14.1 (25.3) |
| mouth_z | [1, 1.5) | season_folds | 5,507 | 1,411 | +24.9 | 26.7 | yes | +29.5 (26.6) |
| mouth_z | [1.5, 2) | season_folds | 3,512 | 1,287 | −13.8 | 23.7 | yes | −19.4 (23.7) |
| mouth_z | [2, 2.3) | season_folds | 1,169 | 410 | −7.7 | 13.3 | yes | −7.3 (13.3) |
| mouth_z | >= 2.3 | season_folds | 58 | 7 | +3.0 | 2.5 | not judged (n < 100) | +3.4 (2.5) |
| angle_rad | [0, 0.3) | all | 9,935 | 1,222 | +27.1 | 30.4 | yes | +25.9 (30.4) |
| angle_rad | [0.3, 0.4) | all | 5,798 | 1,383 | −9.3 | 28.8 | yes | −16.5 (28.7) |
| angle_rad | [0.4, 0.5) | all | 3,831 | 1,238 | +17.3 | 25.3 | yes | +18.9 (25.3) |
| angle_rad | [0.5, 0.7) | all | 5,066 | 1,901 | −18.7 | 29.2 | yes | −16.1 (29.2) |
| angle_rad | [0.7, 1) | all | 4,139 | 1,899 | −13.8 | 27.4 | yes | −12.2 (27.3) |
| angle_rad | >= 1 | all | 2,702 | 1,894 | −2.6 | 21.8 | yes | −0.1 (21.8) |
| angle_rad | [0, 0.3) | holdout_fit | 1,673 | 208 | +14.2 | 12.8 | yes | +17.2 (12.9) |
| angle_rad | [0.3, 0.4) | holdout_fit | 977 | 256 | −14.6 | 11.9 | yes | −15.3 (11.9) |
| angle_rad | [0.4, 0.5) | holdout_fit | 653 | 227 | +2.1 | 10.6 | yes | +1.4 (10.6) |
| angle_rad | [0.5, 0.7) | holdout_fit | 782 | 314 | −2.6 | 11.5 | yes | −1.9 (11.5) |
| angle_rad | [0.7, 1) | holdout_fit | 641 | 314 | −14.0 | 10.8 | yes | −14.7 (10.8) |
| angle_rad | >= 1 | holdout_fit | 428 | 295 | +14.9 | 8.4 | yes | +13.2 (8.5) |
| angle_rad | [0, 0.3) | season_folds | 8,262 | 1,014 | +12.9 | 27.6 | yes | +8.7 (27.5) |
| angle_rad | [0.3, 0.4) | season_folds | 4,821 | 1,127 | +5.3 | 26.2 | yes | −1.2 (26.1) |
| angle_rad | [0.4, 0.5) | season_folds | 3,178 | 1,011 | +15.2 | 23.0 | yes | +17.4 (22.9) |
| angle_rad | [0.5, 0.7) | season_folds | 4,284 | 1,587 | −16.1 | 26.8 | yes | −14.2 (26.8) |
| angle_rad | [0.7, 1) | season_folds | 3,498 | 1,585 | +0.3 | 25.1 | yes | +2.4 (25.1) |
| angle_rad | >= 1 | season_folds | 2,274 | 1,599 | −17.5 | 20.1 | yes | −13.2 (20.1) |

Limitation: 2 of 51 judged bands are outside 2 SD (mouth_y_abs [0, 1) (holdout_fit): 1,429 shots, 223 goals, Σ q − goals +25.2 (SD 12.2, z +2.06; match out of fold +25.3), escalated to the PM; mouth_y_abs [3.3, 3.55) (holdout_fit): 156 shots, 75 goals, Σ q − goals −11.6 (SD 5.1, z -2.25; match out of fold −11.9), escalated to the PM); the league shift removes a season's level gap, not a misprice of shape, so a keeper whose shots lean to these bands is charged or credited for it.

Keeper-level effect: rated from the match out-of-fold xGOT instead, the 126 keeper-seasons with >= 100 shots move by a median of 0.30 goals (max 1.02; their mean SD 4.36).

## Checks

| Target | Measured | Result |
|---|---|---|
| Every rated row's xGOT from fits that never saw its season (season_out raises otherwise); the rating reads those rows, not the match out-of-fold xGOT | season_folds 27,173, holdout_fit 5,291; 32,327 of 32,464 differ from the match out-of-fold xGOT | met |
| Keeper totals sum to the league sum of xGOT − goals (raw and rated), both variants, within 1e-09; table shots = rated rows | max abs. difference < 1e-12 | met |
| Rated against the league sum: abs(Σ (q − goal)) <= 1e-09 per rating group (open play: one δ per competition-season; penalties: one δ per season-out source, PM 2026-10-28 (a)), groups at δ = 0 (E2-small, team-selected or an E2-small pool) excepted | max < 1e-12; 20 open-play and 2 penalty groups rated; 0 team-selected competition-seasons (0 also small), 0 other small, 0 penalty pool(s) at δ = 0 by E2; Σ (q − goal) at δ = 0: team-selected +0.0, other small +0.0 | met |
| Analytic SD and 90% match-bootstrap CI per keeper-season | SD finite; CI null on the 82 one-match keeper-seasons | met |
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0; judged on the keeper-season grouping, the career grouping (1 | keeper) beside it) | keeper-season +0.01034 [+0.00502, +0.01619] on 3,234 shots, 148 pairs; career +0.01037 [+0.00504, +0.01624] | met |
| Split-half reliability at 100 / 250 / 500 shots faced, empty tiers reported (a reporting check: met when every tier is reported) | 100: 146 keepers; 250: 0 keepers (empty); 500: 0 keepers (empty) | met |

## Uncertainty

`sd` is √Σ q(1 − q): the spread of GSAx for a league-average keeper facing these shots, not the uncertainty of this keeper's estimate. The 90% interval resamples the keeper-season's matches (2,000 draws, seed 0) with the rating held fixed; with few matches it undercovers. Against a null with goals drawn once from q (no keeper differences): 13 of 126 keeper-seasons with >= 100 shots have a 90% CI excluding 0; 13 would with no keeper differences. Over the 592 keeper-seasons with >= 2 matches: 161 observed, 135 under the null.

## Stability

Split-half reliability of q − goal per shot (without penalties), shots alternated within a keeper by match date, period, time and shot id (PM X5), over the keepers whose total shots faced (all seasons) reach the tier: first on each keeper's first T shots, then on all their shots; Spearman-Brown steps r up to the whole. 95% intervals by Fisher z. A tier is judged only with at least 30 keepers.

| Shots faced (tier) | Keepers | First T shots: r [95% CI] | Spearman-Brown | All their shots: r [95% CI] | Spearman-Brown | Reading |
|---|---|---|---|---|---|---|
| 100 | 146 | -0.08 [-0.24, +0.09] | -0.17 [-0.62, +0.16] | -0.05 [-0.21, +0.11] | -0.11 [-0.54, +0.20] | not distinguishable from noise at 100 shots |
| 250 | 0 | n/a | n/a | n/a | n/a | no keeper reaches 250 shots (empty tier) |
| 500 | 0 | n/a | n/a | n/a | n/a | no keeper reaches 500 shots (empty tier) |

## Distribution over keeper-seasons with >= 100 shots (without penalties)

126 keeper-seasons; no row per keeper (counts and metrics only). The second row rates the same keeper-seasons with their in-game penalties.

| GSAx | Min | 10% | 25% | Median | 75% | 90% | Max |
|---|---|---|---|---|---|---|---|
| without penalties | -9.8 | -3.5 | -1.6 | +0.4 | +3.4 | +6.4 | +12.9 |
| with penalties | -10.9 | -4.1 | -2.0 | +0.5 | +4.0 | +6.0 | +12.5 |

## Columns

- `goals`: goals from rated shots only (own goals, direct corners and shoot-outs are not in it), so not goals conceded.
- `GSAx raw`: Σ xGOT − goals, before the league rating; `GSAx`: Σ q − goals.
- `SD`: the null SD √Σ q(1 − q); `90% CI`: the match bootstrap above.
- `family`: women; team-selected (one team in at least half of a competition-season's matches; rated at δ = 0); league or tournament.

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
- v2 (PR1): WhoScored season-out from competition-season folds; one penalty δ per season-out source; team-selected competition-seasons at δ = 0 (flagged); goal_z sensitivity (StatsBomb).
