# Model card: Goals saved above expected (GSAx)

## Overview
- **Name / version:** `gsax` v1
- **Task:** a keeper's record of on-target shots faced: Σ q − goals, q the xGOT rated against its competition-season (P3.1)
- **Status:** experimental
- **Source(s):** StatsBomb open data; snapshot `statsbomb-open-v1`, data digest `067ddf5342cfd608045a15189a0b43008664d23f4c5ac4fab5ac403468e57ee5`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
A record of shots faced, rated against the league, not a skill rating: GSAx says how many fewer goals a keeper conceded than a league-average keeper facing the same on-target shots, given where they crossed the line (positive is better). It credits the keeper and the defence in front of him; the stability section says what it allows beyond that. For research; not for other providers' data without re-fitting.

## Data
- **Rated rows:** 32,483 on-target open-play shots (9,721 goals) and 1,000 in-game penalties (808 goals), faced by 729 keepers in 1,205 keeper-seasons and 70 competition-seasons.
- **Not rated:** 266 shoot-out kicks (reported apart by the penalty xGOT); 28 on-target direct corners; own goals are never on target, so they are in no keeper's record. Goals in the tables are goals from rated shots only, not goals conceded.
- **Open goals** (`open_goal`, kept in the population, PM X1): 755 shots, Σ GSAx -153.3.
- **Small competition-seasons** (at most 2 keepers or under 50 shots): 26 rated at δ = 0 and flagged (PM E2 (b)), with 66 keeper-seasons; 21 competition-seasons have at most 2 keepers.
- **Coarse-grid matches:** 1,855 on-target shots of matches recorded on the 1 yd grid are left out (PM E1 (a)); by competition-season below.

### Coarse-grid matches left out

Matches whose shot locations are all whole yards were recorded on the coarse (1 yd) grid (football-core's per-match rule), so their end locations are whole yards too and placement on them is a recording artefact. Their on-target shots are left out of the xGOT fit and of the goalkeeper models (PM E1 (a), MS6), reason *coarse recording grid* in the Filters table:

| Competition | Season | Matches | On-target shots | Goals |
|---|---|---|---|---|
| FA Women's Super League | 2018/2019 | 87 | 863 | 253 |
| FIFA World Cup | 2018 | 64 | 524 | 183 |
| NWSL | 2018 | 36 | 355 | 102 |
| Champions League | 2011/2012 | 1 | 20 | 9 |
| Champions League | 2015/2016 | 1 | 19 | 10 |
| Champions League | 2012/2013 | 1 | 16 | 3 |
| Champions League | 2014/2015 | 1 | 14 | 4 |
| Champions League | 2013/2014 | 1 | 13 | 5 |
| Champions League | 2010/2011 | 1 | 12 | 4 |
| Champions League | 2009/2010 | 1 | 10 | 2 |
| Champions League | 2016/2017 | 1 | 9 | 5 |
| **Total** | | 195 | 1,855 | 580 |

## Method

Every row's xGOT comes from fits (xG input and xGOT) that never saw its season (`xgot.season_out`, source `season_folds`); the match out-of-fold xGOT of the xGOT card is beside it on the same rows.

| Source | Kind | Shots | Goals | Σ xGOT | Log loss | AUC | Match-OOF log loss | Match-OOF AUC |
|---|---|---|---|---|---|---|---|---|
| holdout_fit | open_play | 3,134 | 913 | 897.8 | 0.3999 | 0.8705 | 0.3997 | 0.8708 |
| holdout_fit | penalty | 85 | 74 | 68.3 | 0.3898 | 0.5663 | 0.3898 | 0.5663 |
| season_folds | open_play | 29,349 | 8,808 | 8,814.1 | 0.4084 | 0.8671 | 0.4085 | 0.8671 |
| season_folds | penalty | 915 | 734 | 733.6 | 0.4877 | 0.6233 | 0.4856 | 0.6282 |

Rating against the season's league sum: one logit shift δ per competition-season and kind makes Σ q = goals, q = expit(logit xGOT + δ), so the xGOT level gap of a season is not charged to its keepers (PM after MS5). Penalties have their own δ, so GSAx with penalties minus without is exactly the keeper's penalty GSAx.

- open play: δ from -0.29 to +0.52 (median -0.05) over 44 competition-seasons; 0 with every shot scored or none (0 shots), where every keeper gets 0 and δ is infinite; shots per δ: min 54, median 340.5
- penalty: δ from -1.49 to +1.53 (median -0.02) over 38 competition-seasons; 5 with every shot scored or none (36 shots), where every keeper gets 0 and δ is infinite; shots per δ: min 1, median 12.0

### Calibration of the rated q by placement band

Open play; `mouth_y_abs` and `mouth_z` (m) as on the xGOT card, `angle_rad` the opening angle; source `all` pools the rows. A band is judged with at least 100 shots, within 2 SD (√Σ q(1 − q)). The last column rates the same rows from the xGOT card's match out-of-fold xGOT (diagnostic: it saw the rated season).

| Feature | Band | Source | Shots | Goals | Σ q − goals | SD | Within 2 SD | Match out of fold: Σ q − goals (SD) |
|---|---|---|---|---|---|---|---|---|
| mouth_y_abs | [0, 1) | all | 8,995 | 1,331 | +32.8 | 28.5 | yes | +31.7 (28.5) |
| mouth_y_abs | [1, 2) | all | 9,053 | 1,850 | +24.8 | 31.8 | yes | +24.3 (31.9) |
| mouth_y_abs | [2, 3) | all | 9,890 | 3,805 | −19.9 | 39.7 | yes | −15.4 (39.7) |
| mouth_y_abs | [3, 3.3) | all | 3,382 | 2,105 | −40.0 | 24.8 | yes | −42.2 (24.8) |
| mouth_y_abs | [3.3, 3.55) | all | 970 | 572 | −1.0 | 13.3 | yes | −1.8 (13.3) |
| mouth_y_abs | >= 3.55 | all | 159 | 55 | −0.9 | 5.4 | yes | −0.2 (5.4) |
| mouth_y_abs | outside the frame + 0.15 m | all | 34 | 3 | +8.8 | 2.5 | not judged (n < 100) | +8.3 (2.5) |
| mouth_y_abs | [0, 1) | holdout_fit | 916 | 119 | +12.6 | 9.0 | yes | +12.3 (9.0) |
| mouth_y_abs | [1, 2) | holdout_fit | 867 | 158 | +7.6 | 9.5 | yes | +7.4 (9.6) |
| mouth_y_abs | [2, 3) | holdout_fit | 921 | 376 | −15.9 | 12.2 | yes | −14.8 (12.2) |
| mouth_y_abs | [3, 3.3) | holdout_fit | 310 | 190 | −5.1 | 7.5 | yes | −5.5 (7.5) |
| mouth_y_abs | [3.3, 3.55) | holdout_fit | 104 | 67 | −2.1 | 4.4 | yes | −2.3 (4.4) |
| mouth_y_abs | >= 3.55 | holdout_fit | 12 | 3 | +1.9 | 1.6 | not judged (n < 100) | +1.9 (1.6) |
| mouth_y_abs | outside the frame + 0.15 m | holdout_fit | 4 | 0 | +1.1 | 0.8 | not judged (n < 100) | +1.1 (0.8) |
| mouth_y_abs | [0, 1) | season_folds | 8,079 | 1,212 | +20.3 | 27.1 | yes | +19.4 (27.1) |
| mouth_y_abs | [1, 2) | season_folds | 8,186 | 1,692 | +17.2 | 30.4 | yes | +16.9 (30.5) |
| mouth_y_abs | [2, 3) | season_folds | 8,969 | 3,429 | −4.0 | 37.8 | yes | −0.5 (37.8) |
| mouth_y_abs | [3, 3.3) | season_folds | 3,072 | 1,915 | −35.0 | 23.6 | yes | −36.7 (23.6) |
| mouth_y_abs | [3.3, 3.55) | season_folds | 866 | 505 | +1.1 | 12.6 | yes | +0.6 (12.6) |
| mouth_y_abs | >= 3.55 | season_folds | 147 | 52 | −2.8 | 5.1 | yes | −2.1 (5.1) |
| mouth_y_abs | outside the frame + 0.15 m | season_folds | 30 | 3 | +7.7 | 2.4 | not judged (n < 100) | +7.2 (2.4) |
| mouth_z | [0, 0.3) | all | 7,476 | 2,458 | −28.0 | 31.6 | yes | −24.6 (31.7) |
| mouth_z | [0.3, 0.6) | all | 7,003 | 2,160 | −8.7 | 30.3 | yes | −12.6 (30.3) |
| mouth_z | [0.6, 1) | all | 4,778 | 1,307 | +27.3 | 24.7 | yes | +31.0 (24.7) |
| mouth_z | [1, 1.5) | all | 5,185 | 1,313 | +32.8 | 25.5 | yes | +32.3 (25.5) |
| mouth_z | [1.5, 2) | all | 5,954 | 1,702 | −12.8 | 27.6 | yes | −14.5 (27.6) |
| mouth_z | [2, 2.3) | all | 2,043 | 760 | −0.3 | 17.0 | yes | −0.8 (17.0) |
| mouth_z | >= 2.3 | all | 44 | 21 | −5.8 | 2.5 | not judged (n < 100) | −5.9 (2.6) |
| mouth_z | [0, 0.3) | holdout_fit | 823 | 257 | −10.9 | 10.2 | yes | −11.4 (10.2) |
| mouth_z | [0.3, 0.6) | holdout_fit | 646 | 188 | −2.5 | 9.1 | yes | −1.8 (9.1) |
| mouth_z | [0.6, 1) | holdout_fit | 417 | 123 | −2.4 | 7.2 | yes | −2.5 (7.2) |
| mouth_z | [1, 1.5) | holdout_fit | 461 | 110 | +12.5 | 7.7 | yes | +11.9 (7.7) |
| mouth_z | [1.5, 2) | holdout_fit | 583 | 166 | −0.5 | 8.5 | yes | −0.4 (8.5) |
| mouth_z | [2, 2.3) | holdout_fit | 199 | 67 | +4.4 | 5.3 | yes | +4.7 (5.3) |
| mouth_z | >= 2.3 | holdout_fit | 5 | 2 | −0.5 | 0.9 | not judged (n < 100) | −0.6 (0.9) |
| mouth_z | [0, 0.3) | season_folds | 6,653 | 2,201 | −17.1 | 29.9 | yes | −13.3 (30.0) |
| mouth_z | [0.3, 0.6) | season_folds | 6,357 | 1,972 | −6.2 | 28.9 | yes | −10.8 (28.9) |
| mouth_z | [0.6, 1) | season_folds | 4,361 | 1,184 | +29.7 | 23.6 | yes | +33.4 (23.6) |
| mouth_z | [1, 1.5) | season_folds | 4,724 | 1,203 | +20.3 | 24.3 | yes | +20.3 (24.4) |
| mouth_z | [1.5, 2) | season_folds | 5,371 | 1,536 | −12.3 | 26.2 | yes | −14.1 (26.3) |
| mouth_z | [2, 2.3) | season_folds | 1,844 | 693 | −4.7 | 16.1 | yes | −5.6 (16.2) |
| mouth_z | >= 2.3 | season_folds | 39 | 19 | −5.2 | 2.4 | not judged (n < 100) | −5.3 (2.4) |
| angle_rad | [0, 0.3) | all | 10,196 | 1,089 | +14.2 | 28.1 | yes | +13.8 (28.1) |
| angle_rad | [0.3, 0.4) | all | 7,166 | 1,686 | +8.1 | 30.0 | yes | +7.4 (30.0) |
| angle_rad | [0.4, 0.5) | all | 4,230 | 1,445 | +0.8 | 25.8 | yes | +1.4 (25.8) |
| angle_rad | [0.5, 0.7) | all | 4,792 | 1,845 | −7.3 | 28.4 | yes | −7.6 (28.4) |
| angle_rad | [0.7, 1) | all | 3,616 | 1,832 | −16.0 | 25.9 | yes | −15.8 (26.0) |
| angle_rad | >= 1 | all | 2,483 | 1,824 | +4.7 | 19.9 | yes | +5.6 (19.9) |
| angle_rad | [0, 0.3) | holdout_fit | 1,024 | 110 | +2.9 | 9.0 | yes | +2.4 (9.0) |
| angle_rad | [0.3, 0.4) | holdout_fit | 710 | 163 | −2.3 | 9.3 | yes | −2.5 (9.3) |
| angle_rad | [0.4, 0.5) | holdout_fit | 372 | 135 | −1.3 | 7.8 | yes | −0.9 (7.7) |
| angle_rad | [0.5, 0.7) | holdout_fit | 440 | 182 | −16.1 | 8.5 | yes | −15.8 (8.5) |
| angle_rad | [0.7, 1) | holdout_fit | 367 | 164 | +9.4 | 8.2 | yes | +9.2 (8.2) |
| angle_rad | >= 1 | holdout_fit | 221 | 159 | +7.4 | 5.7 | yes | +7.7 (5.7) |
| angle_rad | [0, 0.3) | season_folds | 9,172 | 979 | +11.3 | 26.6 | yes | +11.4 (26.7) |
| angle_rad | [0.3, 0.4) | season_folds | 6,456 | 1,523 | +10.4 | 28.5 | yes | +9.9 (28.6) |
| angle_rad | [0.4, 0.5) | season_folds | 3,858 | 1,310 | +2.1 | 24.6 | yes | +2.3 (24.6) |
| angle_rad | [0.5, 0.7) | season_folds | 4,352 | 1,663 | +8.8 | 27.1 | yes | +8.2 (27.1) |
| angle_rad | [0.7, 1) | season_folds | 3,249 | 1,668 | −25.4 | 24.6 | yes | −25.0 (24.6) |
| angle_rad | >= 1 | season_folds | 2,262 | 1,665 | −2.7 | 19.1 | yes | −2.1 (19.1) |

Limitation: 0 of 53 judged bands are outside 2 SD.

Keeper-level effect: rated from the match out-of-fold xGOT instead, the 97 keeper-seasons with >= 100 shots move by a median of 0.24 goals (max 1.03; their mean SD 4.08).

## Checks

| Target | Measured | Result |
|---|---|---|
| Every rated row's xGOT from fits that never saw its season (season_out raises otherwise); the rating reads those rows, not the match out-of-fold xGOT | season_folds 30,264, holdout_fit 3,219; 33,398 of 33,483 differ from the match out-of-fold xGOT | met |
| Keeper totals sum to the league sum of xGOT − goals (raw and rated), both variants, within 1e-09; table shots = rated rows | max abs. difference < 1e-12 | met |
| Rated against the season's league sum: abs(Σ (q − goal)) <= 1e-09 per competition-season and kind | max < 1e-12; 26 small competition-season(s) at δ = 0 | met |
| Analytic SD and 90% match-bootstrap CI per keeper-season | SD finite; CI null on the 401 one-match keeper-seasons | met |
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0; judged on the keeper-season grouping, the career grouping (1 | keeper) beside it) | keeper-season +0.03630 [+0.02779, +0.04520] on 5,866 shots, 227 pairs; career +0.02252 [+0.01581, +0.03011] | met |
| Split-half reliability at 100 / 250 / 500 shots faced, empty tiers reported (a reporting check: met when every tier is reported) | 100: 111 keepers; 250: 6 keepers; 500: 2 keepers | met |

## Uncertainty

`sd` is √Σ q(1 − q): the spread of GSAx for a league-average keeper facing these shots, not the uncertainty of this keeper's estimate. The 90% interval resamples the keeper-season's matches (2,000 draws, seed 0) with the rating held fixed; with few matches it undercovers. Against a null with goals drawn once from q (no keeper differences): 18 of 97 keeper-seasons with >= 100 shots have a 90% CI excluding 0; 10 would with no keeper differences. Over the 804 keeper-seasons with >= 2 matches: 268 observed, 242 under the null.

## Stability

Split-half reliability of q − goal per shot (without penalties), shots alternated within a keeper by match date, period, time and shot id (PM X5), over the keepers whose total shots faced (all seasons) reach the tier: first on each keeper's first T shots, then on all their shots; Spearman-Brown steps r up to the whole. 95% intervals by Fisher z. A tier is judged only with at least 30 keepers.

| Shots faced (tier) | Keepers | First T shots: r [95% CI] | Spearman-Brown | All their shots: r [95% CI] | Spearman-Brown | Reading |
|---|---|---|---|---|---|---|
| 100 | 111 | +0.07 [-0.11, +0.26] | +0.14 [-0.26, +0.41] | +0.19 [+0.01, +0.36] | +0.32 [+0.01, +0.53] | not distinguishable from noise at 100 shots |
| 250 | 6 | +0.87 [+0.18, +0.99] | +0.93 [+0.30, +0.99] | +0.71 [-0.25, +0.96] | +0.83 [-0.66, +0.98] | 6 keepers reach 250 shots: too few to judge (r +0.87 [+0.18, +0.99]) |
| 500 | 2 | n/a | n/a | n/a | n/a | 2 keepers reach 500 shots: too few to estimate |

Limitation († team-selected): 8 of the 97 keeper-seasons with >= 100 shots are in team-selected competition-seasons, marked † where named. One δ per competition-season rates the selected side's keeper and its opponents' keepers against one league sum, and xGOT has no finisher term: the opponents' keepers are charged or credited for the selected side's finishing, and the selected side's keepers face 29% of the team-selected rated shots, so δ absorbs part of their own record (the reason for PM E2 (b)). Team-selected keeper-seasons by side (the selected side's keeper faces at least 30 shots), raw (before δ) and rated: selected side: 22 keeper-seasons, 1,939 shots, GSAx raw +79.3, rated +74.0 (SD 15.4); opponents: 536 keeper-seasons, 4,728 shots, GSAx raw −55.2, rated −69.5 (SD 25.8). A team- or shooter-strength term is a polish-round item.

## Keeper-seasons with >= 100 shots (without penalties, sorted by GSAx)

97 keeper-seasons. Names are the lineup display names (the nickname when StatsBomb gives one); 1 table keeper(s) appear under two spellings, resolved to the latest match. The chart is written by `make gsax` to `data/charts/gsax-keepers.svg` (git-ignored: it embeds the StatsBomb logo, PM E3); its SHA-256 is `c8ef5af94585fc943c2ccd61462958981b0a658e28c760c933ae903a0dfca994`, so `make reproduce` checks the chart through this card.

| # | Keeper | Family | Competition season | Shots | Goals | Open goals | Open-goal GSAx | GSAx raw | GSAx | SD | 90% CI | Shrunken GSAx | GSAx with penalties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Samir Handanovič | league or tournament | Serie A 2015/2016 | 140 | 28 | 3 | +0.7 | +11.2 | +11.3 | 4.0 | [+3.8, +18.7] | +2.3 | +11.2 |
| 2 | Keylor Navas | league or tournament | La Liga 2015/2016 | 118 | 27 | 1 | -0.1 | +9.5 | +9.9 | 4.0 | [+4.6, +15.1] | +2.0 | +11.9 |
| 3 | Claudio Bravo | league or tournament | La Liga 2015/2016 | 103 | 19 | 2 | -0.3 | +8.7 | +9.0 | 3.6 | [+4.4, +14.1] | +1.5 | +8.8 |
| 4 | Víctor Valdés | team-selected † | La Liga 2009/2010 | 104 | 20 | 0 | +0.0 | +9.4 | +8.6 | 3.5 | [+2.6, +14.6] | +1.4 | +8.6 |
| 5 | Vincent Enyeama | league or tournament | Ligue 1 2015/2016 | 116 | 22 | 2 | -0.5 | +6.6 | +8.5 | 3.7 | [+2.3, +15.1] | +1.5 | +8.1 |
| 6 | Łukasz Fabiański | league or tournament | Premier League 2015/2016 | 165 | 44 | 4 | -0.4 | +7.6 | +8.4 | 4.5 | [+0.8, +16.8] | +2.0 | +8.3 |
| 7 | Antonio Mirante | league or tournament | Serie A 2015/2016 | 132 | 34 | 4 | -1.6 | +7.2 | +7.3 | 4.4 | [+0.9, +14.5] | +1.7 | +6.6 |
| 8 | Marc-André ter Stegen | team-selected † | La Liga 2017/2018 | 116 | 21 | 3 | -0.6 | +10.1 | +7.0 | 3.7 | [+1.5, +12.4] | +1.3 | +6.3 |
| 9 | Marc-André ter Stegen | team-selected † | La Liga 2016/2017 | 131 | 27 | 3 | -1.2 | +6.7 | +6.8 | 4.0 | [+2.3, +11.7] | +1.4 | +6.8 |
| 10 | Doris Bačić | women | Serie A Women 2023/2024 | 124 | 28 | 1 | -0.1 | +9.0 | +6.5 | 3.9 | [+1.1, +11.7] | +1.3 | +7.7 |
| 11 | Wojciech Szczęsny | league or tournament | Serie A 2015/2016 | 129 | 27 | 4 | -0.6 | +6.3 | +6.4 | 3.9 | [+1.0, +12.3] | +1.2 | +5.6 |
| 12 | Steve Mandanda | league or tournament | Ligue 1 2015/2016 | 126 | 31 | 1 | -0.3 | +4.2 | +6.3 | 4.0 | [-1.5, +13.7] | +1.3 | +5.4 |
| 13 | Lukáš Hrádecký | team-selected † | 1. Bundesliga 2023/2024 | 104 | 24 | 2 | -0.4 | +4.9 | +6.2 | 3.7 | [+0.7, +12.0] | +1.1 | +6.2 |
| 14 | Emiliano Viviano | league or tournament | Serie A 2015/2016 | 185 | 49 | 6 | +0.1 | +5.9 | +6.0 | 4.9 | [-2.3, +14.6] | +1.7 | +8.0 |
| 15 | Javi Varas | league or tournament | La Liga 2015/2016 | 137 | 38 | 5 | -0.9 | +5.5 | +6.0 | 4.3 | [-0.7, +12.7] | +1.4 | +5.8 |
| 16 | Courtney Brosnan | women | FA Women's Super League 2023/2024 | 123 | 32 | 5 | -1.0 | +7.4 | +5.9 | 4.1 | [-2.6, +14.6] | +1.3 | +6.3 |
| 17 | David de Gea | league or tournament | Premier League 2015/2016 | 111 | 28 | 1 | -0.7 | +4.9 | +5.5 | 3.9 | [+0.5, +10.8] | +1.1 | +5.2 |
| 18 | Katie Lund | women | NWSL 2023 | 111 | 21 | 0 | +0.0 | +9.4 | +5.4 | 3.6 | [+1.0, +9.7] | +0.9 | +5.2 |
| 19 | Vicente Guaita | league or tournament | La Liga 2015/2016 | 182 | 58 | 4 | -0.6 | +4.6 | +5.2 | 4.9 | [-2.5, +13.4] | +1.4 | +4.8 |
| 20 | Hugo Lloris | league or tournament | Premier League 2015/2016 | 119 | 30 | 2 | -0.1 | +4.4 | +5.0 | 3.9 | [-1.1, +11.0] | +1.0 | +4.7 |
| 21 | Jan Oblak | league or tournament | La Liga 2015/2016 | 102 | 17 | 2 | -0.3 | +4.8 | +5.0 | 3.1 | [+0.2, +9.4] | +0.7 | +4.8 |
| 22 | Elvira Herzog | women | Frauen Bundesliga 2023/2024 | 105 | 38 | 4 | -1.3 | +4.0 | +4.9 | 3.9 | [-2.0, +11.7] | +1.0 | +4.4 |
| 23 | Zala Meršnik | women | Liga F 2023/2024 | 139 | 41 | 0 | +0.0 | +2.3 | +4.8 | 4.2 | [-0.9, +10.5] | +1.0 | +6.0 |
| 24 | Nayluisa Jhaylenny Cáceres Acevedo | women | Liga F 2023/2024 | 140 | 49 | 2 | -1.1 | +2.0 | +4.7 | 4.4 | [-2.5, +12.3] | +1.1 | +2.8 |
| 25 | Jens Lehmann | team-selected † | Premier League 2003/2004 | 110 | 22 | 1 | -0.2 | +4.2 | +4.4 | 3.6 | [-0.9, +9.8] | +0.7 | +4.4 |
| 26 | Heurelho Gomes | league or tournament | Premier League 2015/2016 | 166 | 43 | 2 | -1.0 | +3.6 | +4.4 | 4.6 | [-2.4, +11.3] | +1.1 | +6.5 |
| 27 | Jean-Louis Leca | league or tournament | Ligue 1 2015/2016 | 115 | 30 | 3 | -0.5 | +2.0 | +3.7 | 3.6 | [-3.4, +10.7] | +0.7 | +3.9 |
| 28 | Adrián | league or tournament | Premier League 2015/2016 | 130 | 32 | 3 | -0.9 | +3.0 | +3.7 | 4.0 | [-3.2, +10.0] | +0.8 | +4.8 |
| 29 | Kasper Schmeichel | league or tournament | Premier League 2015/2016 | 134 | 32 | 2 | -0.3 | +3.0 | +3.6 | 4.1 | [-1.3, +8.5] | +0.8 | +3.1 |
| 30 | Stefano Sorrentino | league or tournament | Serie A 2015/2016 | 174 | 49 | 4 | -0.7 | +3.4 | +3.6 | 4.7 | [-4.9, +11.9] | +0.9 | +2.4 |
| 31 | Petr Čech | league or tournament | Premier League 2015/2016 | 127 | 27 | 0 | +0.0 | +2.8 | +3.4 | 3.9 | [-4.8, +11.4] | +0.7 | +3.3 |
| 32 | Jane Campbell | women | NWSL 2023 | 109 | 15 | 1 | -0.3 | +6.8 | +3.4 | 3.3 | [-2.4, +8.9] | +0.5 | +3.2 |
| 33 | Marc-André ter Stegen | team-selected † | La Liga 2018/2019 | 108 | 27 | 2 | -0.2 | +5.3 | +3.3 | 3.8 | [-2.1, +9.4] | +0.6 | +3.2 |
| 34 | Gianluigi Donnarumma | team-selected † | Ligue 1 2022/2023 | 129 | 29 | 2 | -0.3 | +6.0 | +3.1 | 4.0 | [-3.4, +10.2] | +0.6 | +3.1 |
| 35 | Stéphane Ruffier | league or tournament | Ligue 1 2015/2016 | 140 | 33 | 2 | -0.7 | +0.6 | +2.9 | 4.2 | [-5.3, +10.6] | +0.7 | +3.3 |
| 36 | Albano Bizzarri | league or tournament | Serie A 2015/2016 | 160 | 39 | 4 | -1.0 | +2.4 | +2.5 | 4.2 | [-5.3, +10.2] | +0.6 | +3.5 |
| 37 | Anthony Lopes | league or tournament | Ligue 1 2015/2016 | 141 | 39 | 2 | -0.7 | -0.1 | +2.5 | 4.4 | [-6.4, +10.8] | +0.6 | +2.1 |
| 38 | Danijel Subašić | league or tournament | Ligue 1 2015/2016 | 165 | 48 | 2 | -0.2 | -0.3 | +2.3 | 4.4 | [-4.0, +8.5] | +0.5 | +3.6 |
| 39 | Jack Butland | league or tournament | Premier League 2015/2016 | 135 | 34 | 3 | -0.5 | +1.5 | +2.2 | 4.2 | [-5.1, +9.6] | +0.5 | +1.4 |
| 40 | Mária Korenčiová | women | Serie A Women 2023/2024 | 113 | 32 | 4 | -0.5 | +4.3 | +1.9 | 3.9 | [-2.8, +6.4] | +0.4 | +3.1 |
| 41 | Andrea Consigli | league or tournament | Serie A 2015/2016 | 137 | 35 | 0 | +0.0 | +1.7 | +1.8 | 4.1 | [-4.9, +8.2] | +0.4 | +2.7 |
| 42 | Anna Rosa James-Buhigas | women | Serie A Women 2023/2024 | 106 | 34 | 3 | -0.3 | +4.2 | +1.7 | 3.9 | [-4.9, +7.9] | +0.3 | +2.1 |
| 43 | Gorka Iraizoz | league or tournament | La Liga 2015/2016 | 122 | 31 | 1 | -0.4 | +1.2 | +1.6 | 3.9 | [-6.2, +9.2] | +0.3 | +1.5 |
| 44 | Amanda Tampieri | women | Serie A Women 2023/2024 | 124 | 30 | 4 | -0.6 | +4.1 | +1.6 | 3.9 | [-3.9, +7.2] | +0.3 | +2.3 |
| 45 | Tim Howard | league or tournament | Premier League 2015/2016 | 118 | 31 | 1 | -0.1 | +0.5 | +1.1 | 3.7 | [-4.6, +6.2] | +0.2 | +0.6 |
| 46 | Rémy Vercoutre | league or tournament | Ligue 1 2015/2016 | 169 | 48 | 0 | +0.0 | -2.2 | +0.8 | 4.8 | [-8.0, +10.0] | +0.2 | +0.3 |
| 47 | Alphonse Areola | league or tournament | La Liga 2015/2016 | 106 | 24 | 2 | -0.6 | +0.5 | +0.8 | 3.5 | [-4.3, +6.0] | +0.1 | +1.3 |
| 48 | Sergio Rico | league or tournament | La Liga 2015/2016 | 149 | 39 | 1 | -0.2 | +0.1 | +0.6 | 4.3 | [-4.7, +6.0] | +0.1 | +0.3 |
| 49 | Paula Vizoso | women | Liga F 2023/2024 | 154 | 51 | 5 | -1.7 | -2.3 | +0.6 | 4.5 | [-6.0, +6.8] | +0.1 | +0.2 |
| 50 | Solène Durand | women | Serie A Women 2023/2024 | 127 | 33 | 3 | +0.3 | +3.2 | +0.5 | 4.1 | [-7.6, +7.3] | +0.1 | +0.0 |
| 51 | Mattia Perin | league or tournament | Serie A 2015/2016 | 112 | 28 | 2 | -0.5 | -0.0 | +0.1 | 3.7 | [-6.4, +5.9] | +0.0 | +0.9 |
| 52 | Gianluigi Donnarumma | league or tournament | Serie A 2015/2016 | 109 | 26 | 0 | +0.0 | -0.1 | -0.0 | 3.6 | [-5.8, +6.2] | -0.0 | -0.5 |
| 53 | Germán Lux | league or tournament | La Liga 2015/2016 | 109 | 34 | 2 | -0.6 | -0.5 | -0.1 | 3.8 | [-7.5, +7.5] | -0.0 | +0.0 |
| 54 | Łukasz Skorupski | league or tournament | Serie A 2015/2016 | 127 | 41 | 1 | -0.2 | -0.3 | -0.2 | 4.1 | [-8.6, +8.2] | -0.0 | -0.5 |
| 55 | Bernd Leno | team-selected † | 1. Bundesliga 2015/2016 | 128 | 32 | 2 | -0.8 | -1.6 | -0.4 | 3.9 | [-7.7, +7.0] | -0.1 | -0.0 |
| 56 | Gero Rulli | league or tournament | La Liga 2015/2016 | 144 | 39 | 6 | -1.5 | -0.8 | -0.4 | 4.1 | [-7.2, +6.3] | -0.1 | -0.2 |
| 57 | Lisa Weiß | women | FA Women's Super League 2020/2021 | 115 | 39 | 2 | -0.3 | +0.5 | -0.4 | 4.0 | [-6.7, +6.0] | -0.1 | -0.8 |
| 58 | Wayne Hennessey | league or tournament | Premier League 2015/2016 | 114 | 33 | 3 | -0.9 | -1.2 | -0.6 | 3.7 | [-6.8, +5.1] | -0.1 | -0.9 |
| 59 | Diego Mariño | league or tournament | La Liga 2015/2016 | 116 | 37 | 4 | -0.0 | -1.1 | -0.7 | 3.9 | [-8.1, +6.4] | -0.1 | -1.4 |
| 60 | Sophie Baggaley | women | FA Women's Super League 2023/2024 | 120 | 40 | 3 | -0.9 | +0.5 | -0.9 | 4.1 | [-8.2, +6.7] | -0.2 | -0.9 |
| 61 | Orestis Karnezis | league or tournament | Serie A 2015/2016 | 162 | 52 | 5 | -0.9 | -1.4 | -1.3 | 4.7 | [-8.3, +5.8] | -0.3 | -2.0 |
| 62 | Antonio Adán | league or tournament | La Liga 2015/2016 | 176 | 49 | 4 | -0.8 | -2.0 | -1.5 | 4.7 | [-10.7, +7.9] | -0.4 | -1.5 |
| 63 | Benoît Guy Robert Costil | league or tournament | Ligue 1 2015/2016 | 133 | 42 | 1 | -0.2 | -4.0 | -1.6 | 4.3 | [-8.3, +5.5] | -0.3 | -1.3 |
| 64 | Pichu | league or tournament | La Liga 2015/2016 | 158 | 43 | 6 | -1.7 | -2.1 | -1.6 | 4.3 | [-8.1, +5.8] | -0.4 | -1.8 |
| 65 | Vid Belec | league or tournament | Serie A 2015/2016 | 126 | 37 | 4 | -1.2 | -1.8 | -1.7 | 4.0 | [-7.3, +3.9] | -0.4 | -1.5 |
| 66 | Elene Lete | women | Liga F 2023/2024 | 151 | 49 | 3 | -1.0 | -4.6 | -1.8 | 4.5 | [-9.2, +5.0] | -0.5 | -2.0 |
| 67 | Hannah Hampton | women | FA Women's Super League 2020/2021 | 126 | 42 | 4 | -0.4 | -1.0 | -1.9 | 4.0 | [-8.7, +4.7] | -0.4 | -0.7 |
| 68 | Antonia Ignacia Canales Pacheco | women | Liga F 2023/2024 | 126 | 38 | 3 | -0.7 | -4.1 | -1.9 | 4.0 | [-9.2, +5.2] | -0.4 | -1.6 |
| 69 | Nicola Leali | league or tournament | Serie A 2015/2016 | 176 | 54 | 4 | -0.4 | -2.3 | -2.2 | 4.6 | [-9.8, +5.9] | -0.6 | -2.2 |
| 70 | Mackenzie Arnold | women | FA Women's Super League 2023/2024 | 113 | 40 | 0 | +0.0 | -0.8 | -2.2 | 3.9 | [-6.5, +2.0] | -0.4 | -1.3 |
| 71 | Clément Maury | league or tournament | Ligue 1 2015/2016 | 139 | 46 | 1 | -0.3 | -4.9 | -2.3 | 4.4 | [-9.1, +4.7] | -0.5 | -3.2 |
| 72 | Ena Mahmutovic | women | Frauen Bundesliga 2023/2024 | 145 | 54 | 4 | -0.8 | -3.6 | -2.3 | 4.6 | [-11.2, +6.5] | -0.6 | -2.6 |
| 73 | Joe Hart | league or tournament | Premier League 2015/2016 | 107 | 33 | 2 | -0.3 | -3.0 | -2.5 | 3.8 | [-7.9, +3.2] | -0.5 | -2.6 |
| 74 | Esther Sullastres | women | Liga F 2023/2024 | 128 | 37 | 3 | -0.9 | -4.8 | -2.7 | 4.0 | [-11.2, +4.8] | -0.5 | -2.1 |
| 75 | Marco Sportiello | league or tournament | Serie A 2015/2016 | 144 | 38 | 2 | -0.8 | -3.1 | -3.0 | 4.2 | [-9.3, +2.7] | -0.7 | -1.3 |
| 76 | Benjamin Lecomte | league or tournament | Ligue 1 2015/2016 | 170 | 50 | 4 | -0.6 | -5.9 | -3.1 | 4.6 | [-11.5, +5.8] | -0.8 | -3.7 |
| 77 | Sandra Estévez | women | Liga F 2023/2024 | 118 | 45 | 10 | -1.2 | -5.4 | -3.2 | 4.0 | [-9.6, +3.1] | -0.7 | -3.5 |
| 78 | Sophie Baggaley | women | FA Women's Super League 2020/2021 | 159 | 61 | 4 | -0.6 | -1.9 | -3.2 | 4.9 | [-12.9, +6.5] | -0.9 | -3.7 |
| 79 | Andrés Fernández | league or tournament | La Liga 2015/2016 | 163 | 60 | 9 | -2.0 | -3.8 | -3.3 | 4.7 | [-10.4, +3.5] | -0.9 | -4.4 |
| 80 | Simon Mignolet | league or tournament | Premier League 2015/2016 | 109 | 39 | 3 | -0.4 | -4.0 | -3.4 | 3.9 | [-11.6, +4.9] | -0.7 | -2.7 |
| 81 | Pau López | league or tournament | La Liga 2015/2016 | 158 | 56 | 8 | -1.6 | -4.1 | -3.6 | 4.5 | [-10.9, +4.5] | -0.9 | -4.7 |
| 82 | Grace Moloney | women | FA Women's Super League 2020/2021 | 103 | 39 | 5 | -1.3 | -3.1 | -3.9 | 3.8 | [-12.8, +5.4] | -0.7 | -3.7 |
| 83 | Pierluigi Gollini | league or tournament | Serie A 2015/2016 | 109 | 38 | 2 | -0.3 | -4.1 | -4.0 | 3.7 | [-9.9, +1.8] | -0.7 | -4.9 |
| 84 | Federico Marchetti | league or tournament | Serie A 2015/2016 | 110 | 34 | 2 | -0.8 | -4.3 | -4.2 | 3.7 | [-11.8, +3.0] | -0.8 | -3.3 |
| 85 | Sergio Álvarez | league or tournament | La Liga 2015/2016 | 130 | 42 | 3 | -0.5 | -4.8 | -4.4 | 4.1 | [-10.4, +1.5] | -0.9 | -4.2 |
| 86 | Sophie Baggaley | women | FA Women's Super League 2019/2020 | 108 | 37 | 2 | -0.2 | -3.4 | -4.7 | 3.8 | [-14.0, +3.5] | -0.9 | -3.6 |
| 87 | John Ruddy | league or tournament | Premier League 2015/2016 | 108 | 39 | 3 | +0.6 | -5.3 | -4.8 | 3.9 | [-11.8, +1.1] | -0.9 | -5.4 |
| 88 | Carmen Carbonel | women | Liga F 2023/2024 | 118 | 41 | 3 | -1.1 | -6.9 | -4.9 | 3.9 | [-9.5, -0.6] | -0.9 | -5.6 |
| 89 | Daniele Padelli | league or tournament | Serie A 2015/2016 | 130 | 41 | 2 | -0.2 | -5.1 | -5.0 | 4.1 | [-11.0, +0.9] | -1.1 | -6.2 |
| 90 | Ciprian Tătăruşanu | league or tournament | Serie A 2015/2016 | 107 | 39 | 2 | -0.4 | -5.1 | -5.0 | 3.8 | [-11.4, +1.2] | -0.9 | -5.0 |
| 91 | Jonas Lössl | league or tournament | Ligue 1 2015/2016 | 158 | 46 | 7 | -1.6 | -7.7 | -5.3 | 4.3 | [-12.5, +1.5] | -1.2 | -5.1 |
| 92 | Asier Riesgo | league or tournament | La Liga 2015/2016 | 129 | 50 | 7 | -1.1 | -6.1 | -5.7 | 4.3 | [-13.0, +1.0] | -1.3 | -5.8 |
| 93 | Paul Bernardoni | league or tournament | Ligue 1 2015/2016 | 113 | 36 | 4 | -0.7 | -8.1 | -6.3 | 3.7 | [-12.5, -0.3] | -1.2 | -6.6 |
| 94 | Artur Boruc | league or tournament | Premier League 2015/2016 | 120 | 48 | 2 | -0.4 | -7.2 | -6.6 | 4.1 | [-13.8, +0.4] | -1.4 | -5.1 |
| 95 | Alyssa Naeher | women | NWSL 2023 | 125 | 45 | 3 | -0.3 | -2.1 | -7.0 | 4.1 | [-16.6, +2.0] | -1.5 | -7.2 |
| 96 | Paola Ulloa | women | Liga F 2023/2024 | 128 | 51 | 1 | -0.3 | -9.7 | -7.2 | 4.2 | [-14.7, +0.3] | -1.6 | -7.0 |
| 97 | Brad Guzan | league or tournament | Premier League 2015/2016 | 142 | 53 | 4 | -0.9 | -8.2 | -7.4 | 4.4 | [-13.5, -1.8] | -1.8 | -7.7 |

## Columns

- `goals`: goals from rated shots only (own goals, direct corners and shoot-outs are not in it), so not goals conceded.
- `GSAx raw`: Σ xGOT − goals, before the league rating; `GSAx`: Σ q − goals.
- `SD`: the null SD √Σ q(1 − q); `90% CI`: the match bootstrap above.
- `family`: women; team-selected (one team in at least half of a competition-season's matches); league or tournament.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
