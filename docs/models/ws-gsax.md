# Model card: Goals saved above expected (GSAx)

## Overview
- **Name / version:** `ws-gsax` v1
- **Task:** a keeper's record of on-target shots faced: Σ q − goals, q the xGOT rated against its competition-season (P3.1)
- **Status:** experimental
- **Source(s):** private WhoScored / Opta event archive; snapshot `whoscored-v1`, data digest `8033836b08b03fcbaffe2a19cc3a9549f2adb2eebf10f2871fff57f267b79ebb`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
A record of shots faced, rated against the league, not a skill rating: GSAx says how many fewer goals a keeper conceded than a league-average keeper facing the same on-target shots, given where they crossed the line (positive is better). It credits the keeper and the defence in front of him; the stability section says what it allows beyond that. For research; not for other providers' data without re-fitting.

## Data
- **Rated rows:** 31,471 on-target open-play shots (9,537 goals) and 993 in-game penalties (834 goals), faced by 473 keepers in 674 keeper-seasons and 20 competition-seasons.
- **Not rated:** 16 shoot-out kicks (reported apart by the penalty xGOT); 9 on-target direct corners; own goals are never on target, so they are in no keeper's record. Goals in the tables are goals from rated shots only, not goals conceded.
- **Open goals** (`open_goal`, kept in the population, PM X1): not recorded.
- **Small competition-seasons** (at most 2 keepers or under 50 shots): 0 rated at δ = 0 and flagged (PM E2 (b)), with 0 keeper-seasons; 0 competition-seasons have at most 2 keepers.

## Method

Every row's xGOT comes from fits (xG input and xGOT) that never saw its season (`xgot.season_out`, source `swapped_fit`); the match out-of-fold xGOT of the xGOT card is beside it on the same rows.

| Source | Kind | Shots | Goals | Σ xGOT | Log loss | AUC | Match-OOF log loss | Match-OOF AUC |
|---|---|---|---|---|---|---|---|---|
| holdout_fit | open_play | 5,154 | 1,614 | 1,536.3 | 0.4394 | 0.8475 | 0.4394 | 0.8475 |
| holdout_fit | penalty | 137 | 120 | 114.8 | 0.3847 | 0.5279 | 0.3847 | 0.5279 |
| swapped_fit | open_play | 26,317 | 7,923 | 8,212.0 | 0.4452 | 0.8381 | 0.4383 | 0.8433 |
| swapped_fit | penalty | 856 | 714 | 750.5 | 0.4694 | 0.5001 | 0.4444 | 0.5922 |

Rating against the season's league sum: one logit shift δ per competition-season and kind makes Σ q = goals, q = expit(logit xGOT + δ), so the xGOT level gap of a season is not charged to its keepers (PM after MS5). Penalties have their own δ, so GSAx with penalties minus without is exactly the keeper's penalty GSAx.

- open play: δ from -0.17 to +0.31 (median -0.03) over 20 competition-seasons; 0 with every shot scored or none (0 shots), where every keeper gets 0 and δ is infinite; shots per δ: min 374, median 733.5
- penalty: δ from -1.96 to +1.31 (median -0.29) over 17 competition-seasons; 3 with every shot scored or none (46 shots), where every keeper gets 0 and δ is infinite; shots per δ: min 4, median 22.5

### Calibration of the rated q by placement band

Open play; `mouth_y_abs` and `mouth_z` (m) as on the xGOT card, `angle_rad` the opening angle; source `all` pools the rows. A band is judged with at least 100 shots, within 2 SD (√Σ q(1 − q)). The last column rates the same rows from the xGOT card's match out-of-fold xGOT (diagnostic: it saw the rated season).

| Feature | Band | Source | Shots | Goals | Σ q − goals | SD | Within 2 SD | Match out of fold: Σ q − goals (SD) |
|---|---|---|---|---|---|---|---|---|
| mouth_y_abs | [0, 1) | all | 8,981 | 1,500 | −39.2 | 30.3 | yes | +41.4 (30.8) |
| mouth_y_abs | [1, 2) | all | 9,363 | 2,346 | +18.9 | 36.4 | yes | +4.1 (35.3) |
| mouth_y_abs | [2, 3) | all | 9,091 | 3,665 | −34.1 | 39.5 | yes | −11.1 (39.4) |
| mouth_y_abs | [3, 3.3) | all | 3,061 | 1,617 | +4.2 | 23.5 | yes | −33.2 (23.4) |
| mouth_y_abs | [3.3, 3.55) | all | 943 | 404 | +44.2 | 13.1 | no | −5.4 (12.7) |
| mouth_y_abs | >= 3.55 | all | 32 | 5 | +6.0 | 2.4 | not judged (n < 100) | +4.3 (2.2) |
| mouth_y_abs | outside the frame + 0.15 m | all | 0 | 0 | n/a | 0.0 | not judged (n < 100) | n/a |
| mouth_y_abs | [0, 1) | holdout_fit | 1,429 | 223 | +25.3 | 12.3 | no | +25.3 (12.3) |
| mouth_y_abs | [1, 2) | holdout_fit | 1,549 | 410 | −4.7 | 14.6 | yes | −4.7 (14.6) |
| mouth_y_abs | [2, 3) | holdout_fit | 1,500 | 621 | +2.3 | 16.1 | yes | +2.3 (16.1) |
| mouth_y_abs | [3, 3.3) | holdout_fit | 517 | 285 | −12.0 | 9.6 | yes | −12.0 (9.6) |
| mouth_y_abs | [3.3, 3.55) | holdout_fit | 156 | 75 | −11.9 | 5.1 | no | −11.9 (5.1) |
| mouth_y_abs | >= 3.55 | holdout_fit | 3 | 0 | +0.9 | 0.7 | not judged (n < 100) | +0.9 (0.7) |
| mouth_y_abs | outside the frame + 0.15 m | holdout_fit | 0 | 0 | n/a | 0.0 | not judged (n < 100) | n/a |
| mouth_y_abs | [0, 1) | swapped_fit | 7,552 | 1,277 | −64.5 | 27.7 | no | +16.1 (28.2) |
| mouth_y_abs | [1, 2) | swapped_fit | 7,814 | 1,936 | +23.6 | 33.4 | yes | +8.7 (32.2) |
| mouth_y_abs | [2, 3) | swapped_fit | 7,591 | 3,044 | −36.4 | 36.1 | yes | −13.4 (35.9) |
| mouth_y_abs | [3, 3.3) | swapped_fit | 2,544 | 1,332 | +16.2 | 21.5 | yes | −21.2 (21.4) |
| mouth_y_abs | [3.3, 3.55) | swapped_fit | 787 | 329 | +56.1 | 12.1 | no | +6.5 (11.7) |
| mouth_y_abs | >= 3.55 | swapped_fit | 29 | 5 | +5.0 | 2.3 | not judged (n < 100) | +3.3 (2.1) |
| mouth_y_abs | outside the frame + 0.15 m | swapped_fit | 0 | 0 | n/a | 0.0 | not judged (n < 100) | n/a |
| mouth_z | [0, 0.3) | all | 7,796 | 2,532 | −29.7 | 34.1 | yes | −23.7 (34.3) |
| mouth_z | [0.3, 0.6) | all | 5,823 | 1,581 | +94.8 | 28.4 | no | +24.3 (28.0) |
| mouth_z | [0.6, 1) | all | 5,638 | 1,721 | +24.7 | 28.3 | yes | −34.4 (27.7) |
| mouth_z | [1, 1.5) | all | 6,574 | 1,676 | +42.8 | 29.6 | yes | +39.2 (29.0) |
| mouth_z | [1.5, 2) | all | 4,191 | 1,533 | −102.0 | 25.8 | no | −6.2 (25.9) |
| mouth_z | [2, 2.3) | all | 1,379 | 486 | −36.5 | 14.6 | no | −4.0 (14.5) |
| mouth_z | >= 2.3 | all | 70 | 8 | +6.0 | 2.9 | not judged (n < 100) | +4.7 (2.8) |
| mouth_z | [0, 0.3) | holdout_fit | 1,347 | 459 | −8.0 | 14.5 | yes | −8.0 (14.5) |
| mouth_z | [0.3, 0.6) | holdout_fit | 930 | 269 | +0.8 | 11.1 | yes | +0.8 (11.1) |
| mouth_z | [0.6, 1) | holdout_fit | 909 | 298 | −20.3 | 11.3 | yes | −20.3 (11.3) |
| mouth_z | [1, 1.5) | holdout_fit | 1,067 | 265 | +9.7 | 11.7 | yes | +9.7 (11.7) |
| mouth_z | [1.5, 2) | holdout_fit | 679 | 246 | +13.2 | 10.5 | yes | +13.2 (10.5) |
| mouth_z | [2, 2.3) | holdout_fit | 210 | 76 | +3.3 | 5.7 | yes | +3.3 (5.7) |
| mouth_z | >= 2.3 | holdout_fit | 12 | 1 | +1.4 | 1.2 | not judged (n < 100) | +1.4 (1.2) |
| mouth_z | [0, 0.3) | swapped_fit | 6,449 | 2,073 | −21.7 | 30.9 | yes | −15.7 (31.1) |
| mouth_z | [0.3, 0.6) | swapped_fit | 4,893 | 1,312 | +94.0 | 26.1 | no | +23.6 (25.7) |
| mouth_z | [0.6, 1) | swapped_fit | 4,729 | 1,423 | +45.0 | 26.0 | yes | −14.1 (25.3) |
| mouth_z | [1, 1.5) | swapped_fit | 5,507 | 1,411 | +33.1 | 27.2 | yes | +29.5 (26.6) |
| mouth_z | [1.5, 2) | swapped_fit | 3,512 | 1,287 | −115.2 | 23.6 | no | −19.4 (23.7) |
| mouth_z | [2, 2.3) | swapped_fit | 1,169 | 410 | −39.8 | 13.4 | no | −7.3 (13.3) |
| mouth_z | >= 2.3 | swapped_fit | 58 | 7 | +4.6 | 2.6 | not judged (n < 100) | +3.4 (2.5) |
| angle_rad | [0, 0.3) | all | 9,935 | 1,222 | −25.5 | 30.1 | yes | +25.9 (30.4) |
| angle_rad | [0.3, 0.4) | all | 5,798 | 1,383 | +64.5 | 29.3 | no | −16.5 (28.7) |
| angle_rad | [0.4, 0.5) | all | 3,831 | 1,238 | +41.9 | 25.4 | yes | +18.9 (25.3) |
| angle_rad | [0.5, 0.7) | all | 5,066 | 1,901 | +7.9 | 29.3 | yes | −16.1 (29.2) |
| angle_rad | [0.7, 1) | all | 4,139 | 1,899 | −4.0 | 27.3 | yes | −12.2 (27.3) |
| angle_rad | >= 1 | all | 2,702 | 1,894 | −84.8 | 22.7 | no | −0.1 (21.8) |
| angle_rad | [0, 0.3) | holdout_fit | 1,673 | 208 | +17.2 | 12.9 | yes | +17.2 (12.9) |
| angle_rad | [0.3, 0.4) | holdout_fit | 977 | 256 | −15.3 | 11.9 | yes | −15.3 (11.9) |
| angle_rad | [0.4, 0.5) | holdout_fit | 653 | 227 | +1.4 | 10.6 | yes | +1.4 (10.6) |
| angle_rad | [0.5, 0.7) | holdout_fit | 782 | 314 | −1.9 | 11.5 | yes | −1.9 (11.5) |
| angle_rad | [0.7, 1) | holdout_fit | 641 | 314 | −14.7 | 10.8 | yes | −14.7 (10.8) |
| angle_rad | >= 1 | holdout_fit | 428 | 295 | +13.2 | 8.5 | yes | +13.2 (8.5) |
| angle_rad | [0, 0.3) | swapped_fit | 8,262 | 1,014 | −42.7 | 27.3 | yes | +8.7 (27.5) |
| angle_rad | [0.3, 0.4) | swapped_fit | 4,821 | 1,127 | +79.8 | 26.8 | no | −1.2 (26.1) |
| angle_rad | [0.4, 0.5) | swapped_fit | 3,178 | 1,011 | +40.4 | 23.1 | yes | +17.4 (22.9) |
| angle_rad | [0.5, 0.7) | swapped_fit | 4,284 | 1,587 | +9.8 | 26.9 | yes | −14.2 (26.8) |
| angle_rad | [0.7, 1) | swapped_fit | 3,498 | 1,585 | +10.6 | 25.1 | yes | +2.4 (25.1) |
| angle_rad | >= 1 | swapped_fit | 2,274 | 1,599 | −97.9 | 21.0 | no | −13.2 (20.1) |

Limitation: 15 of 51 judged bands are outside 2 SD (mouth_y_abs [3.3, 3.55) (all): 943 shots, Σ q − goals +44.2 (SD 13.1; match out of fold −5.4); mouth_y_abs [0, 1) (holdout_fit): 1,429 shots, Σ q − goals +25.3 (SD 12.3; match out of fold +25.3); mouth_y_abs [3.3, 3.55) (holdout_fit): 156 shots, Σ q − goals −11.9 (SD 5.1; match out of fold −11.9); mouth_y_abs [0, 1) (swapped_fit): 7,552 shots, Σ q − goals −64.5 (SD 27.7; match out of fold +16.1); mouth_y_abs [3.3, 3.55) (swapped_fit): 787 shots, Σ q − goals +56.1 (SD 12.1; match out of fold +6.5); mouth_z [0.3, 0.6) (all): 5,823 shots, Σ q − goals +94.8 (SD 28.4; match out of fold +24.3); mouth_z [1.5, 2) (all): 4,191 shots, Σ q − goals −102.0 (SD 25.8; match out of fold −6.2); mouth_z [2, 2.3) (all): 1,379 shots, Σ q − goals −36.5 (SD 14.6; match out of fold −4.0); mouth_z [0.3, 0.6) (swapped_fit): 4,893 shots, Σ q − goals +94.0 (SD 26.1; match out of fold +23.6); mouth_z [1.5, 2) (swapped_fit): 3,512 shots, Σ q − goals −115.2 (SD 23.6; match out of fold −19.4); mouth_z [2, 2.3) (swapped_fit): 1,169 shots, Σ q − goals −39.8 (SD 13.4; match out of fold −7.3); angle_rad [0.3, 0.4) (all): 5,798 shots, Σ q − goals +64.5 (SD 29.3; match out of fold −16.5); angle_rad >= 1 (all): 2,702 shots, Σ q − goals −84.8 (SD 22.7; match out of fold −0.1); angle_rad [0.3, 0.4) (swapped_fit): 4,821 shots, Σ q − goals +79.8 (SD 26.8; match out of fold −1.2); angle_rad >= 1 (swapped_fit): 2,274 shots, Σ q − goals −97.9 (SD 21.0; match out of fold −13.2)); the league shift removes a season's level gap, not a misprice of shape, so a keeper whose shots lean to these bands is charged or credited for it.

Swapped fit: the `swapped_fit` rows (27,173, 84% of the rated rows) are rated from an xGOT fitted on the test season's 5,291 rated rows (`holdout_fit`) alone; a larger training set waits for the full test season or the polish round's rolling fit (PM E4).

Keeper-level effect: rated from the match out-of-fold xGOT instead, the 126 keeper-seasons with >= 100 shots move by a median of 0.54 goals (max 2.03; their mean SD 4.40).

## Checks

| Target | Measured | Result |
|---|---|---|
| Every rated row's xGOT from fits that never saw its season (season_out raises otherwise) | holdout_fit 5,291, swapped_fit 27,173 | met |
| Keeper totals sum to the league sum of xGOT − goals (raw and rated), both variants, within 1e-09; table shots = rated rows | max abs. difference < 1e-12 | met |
| Rated against the season's league sum: abs(Σ (q − goal)) <= 1e-09 per competition-season and kind | max < 1e-12; 0 small competition-season(s) at δ = 0 | met |
| Analytic SD and 90% match-bootstrap CI per keeper-season | SD finite; CI null on the 82 one-match keeper-seasons | met |
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0) | +0.01047 [+0.00513, +0.01630] on 3,234 shots, 148 pairs | met |
| Split-half reliability at 100 / 250 / 500 shots faced, empty tiers reported | 100: 146 keepers; 250: 0 keepers (empty); 500: 0 keepers (empty) | met |

## Uncertainty

`sd` is √Σ q(1 − q): the spread of GSAx for a league-average keeper facing these shots, not the uncertainty of this keeper's estimate. The 90% interval resamples the keeper-season's matches (2,000 draws, seed 0) with the rating held fixed; with few matches it undercovers. Against a null with goals drawn once from q (no keeper differences): 15 of 126 keeper-seasons with >= 100 shots have a 90% CI excluding 0; 13 would with no keeper differences. Over the 592 keeper-seasons with >= 2 matches: 152 observed, 123 under the null.

## Stability

Split-half reliability of q − goal per shot (without penalties), shots alternated within a keeper by match date, period, time and shot id (PM X5), over the keepers whose total shots faced (all seasons) reach the tier: first on each keeper's first T shots, then on all their shots; Spearman-Brown steps r up to the whole. 95% intervals by Fisher z. A tier is judged only with at least 30 keepers.

| Shots faced (tier) | Keepers | First T shots: r [95% CI] | Spearman-Brown | All their shots: r [95% CI] | Spearman-Brown | Reading |
|---|---|---|---|---|---|---|
| 100 | 146 | -0.09 [-0.25, +0.07] | -0.20 [-0.66, +0.14] | -0.05 [-0.21, +0.11] | -0.11 [-0.54, +0.20] | not distinguishable from noise at 100 shots |
| 250 | 0 | n/a | n/a | n/a | n/a | no keeper reaches 250 shots (empty tier) |
| 500 | 0 | n/a | n/a | n/a | n/a | no keeper reaches 500 shots (empty tier) |

## Distribution over keeper-seasons with >= 100 shots (without penalties)

126 keeper-seasons; no row per keeper (counts and metrics only). The second row rates the same keeper-seasons with their in-game penalties.

| GSAx | Min | 10% | 25% | Median | 75% | 90% | Max |
|---|---|---|---|---|---|---|---|
| without penalties | -10.2 | -4.2 | -1.6 | +0.7 | +3.6 | +6.7 | +12.9 |
| with penalties | -11.0 | -4.5 | -2.4 | +0.8 | +3.7 | +6.5 | +12.6 |

## Columns

- `goals`: goals from rated shots only (own goals, direct corners and shoot-outs are not in it), so not goals conceded.
- `GSAx raw`: Σ xGOT − goals, before the league rating; `GSAx`: Σ q − goals.
- `SD`: the null SD √Σ q(1 − q); `90% CI`: the match bootstrap above.
- `family`: women; team-selected (one team in at least half of a competition-season's matches); league or tournament.

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
