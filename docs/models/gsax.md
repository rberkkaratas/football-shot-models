# Model card: Goals saved above expected (GSAx)

## Overview
- **Name / version:** `gsax` v2
- **Task:** a keeper's record of on-target shots faced: Σ q − goals, q the xGOT rated against the league sum (P3.1)
- **Status:** experimental
- **Source(s):** StatsBomb open data; snapshot `statsbomb-open-v1`, data digest `a10b7542f2dc13c3b1f46191c8af3e52b9b6f9281543cc7a436f9797b881209e`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
A record of shots faced, rated against the league, not a skill rating: GSAx says how many fewer goals a keeper conceded than a league-average keeper facing the same on-target shots, given where they crossed the line (positive is better). It credits the keeper and the defence in front of him; the stability section says what it allows beyond that. For research; not for other providers' data without re-fitting.

## Data
- **Rated rows:** 32,483 on-target open-play shots (9,721 goals) and 1,000 in-game penalties (808 goals), faced by 729 keepers in 1,205 keeper-seasons and 70 competition-seasons.
- **Not rated:** 266 shoot-out kicks (reported apart by the penalty xGOT); 28 on-target direct corners; own goals are never on target, so they are in no keeper's record. Goals in the tables are goals from rated shots only, not goals conceded.
- **Open goals** (`open_goal`, kept in the population, PM X1): 755 shots, Σ GSAx -151.4.
- **Small competition-seasons** (at most 2 keepers or under 50 shots): 26 rated at δ = 0 and flagged (PM E2 (b)), with 66 keeper-seasons; 21 competition-seasons have at most 2 keepers.
- **Team-selected competition-seasons:** 48 rated at δ = 0 and flagged (PM 2026-10-28), 26 also E2-small, with 558 keeper-seasons.
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

Every row's xGOT comes from fits (xG input and xGOT) that never saw its season (`season_out.season_out`, source `season_folds`: folds grouped by competition-season, the test season from the refit's dev fit, `holdout_fit`); the match out-of-fold xGOT of the xGOT card is beside it on the same rows.

| Source | Kind | Shots | Goals | Σ xGOT | Log loss | AUC | Match-OOF log loss | Match-OOF AUC |
|---|---|---|---|---|---|---|---|---|
| holdout_fit | open_play | 3,134 | 913 | 896.4 | 0.4033 | 0.8682 | 0.4030 | 0.8685 |
| holdout_fit | penalty | 85 | 74 | 68.3 | 0.3898 | 0.5663 | 0.3898 | 0.5663 |
| season_folds | open_play | 29,349 | 8,808 | 8,812.4 | 0.4091 | 0.8665 | 0.4087 | 0.8669 |
| season_folds | penalty | 915 | 734 | 733.6 | 0.4877 | 0.6233 | 0.4856 | 0.6282 |

Rating against the league sum: one logit shift δ per rating group makes Σ q = goals, q = expit(logit xGOT + δ), so the xGOT level gap is not charged to keepers (PM after MS5). Open play has one δ per competition-season, penalties one δ per season-out source (PM 2026-10-28 (a)); a group with at most 2 keepers or under 50 shots is at δ = 0 (E2's minimum, every grouping), and so are team-selected competition-seasons. Each row is rated once, so GSAx with penalties minus without is exactly the keeper's penalty GSAx.

- open play: δ from -0.27 to +0.33 (median -0.04) over 22 competition-seasons; 0 with every shot scored or none (0 shots), where every keeper gets 0 and δ is infinite; shots per δ: min 191, median 846.5; 48 competition-seasons at δ = 0
- penalty, δ per source: holdout_fit +0.51 on 85 kicks (34 keepers; league or tournament 85)
- penalty, δ per source: season_folds -0.09 on 717 kicks (280 keepers; league or tournament 398, women 319)
- penalty at δ = 0: 198 kicks of competition-seasons at δ = 0; 0 source pool(s) failing E2's minimum (0 kicks)

### Calibration of the rated q by placement band

Open play; `mouth_y_abs` and `mouth_z` (m) as on the xGOT card, `angle_rad` the opening angle; source `all` pools the rows. A band is judged with at least 100 shots, within 2 SD (√Σ q(1 − q)). The last column rates the same rows from the xGOT card's match out-of-fold xGOT (diagnostic: it saw the rated season).

| Feature | Band | Source | Shots | Goals | Σ q − goals | SD | Within 2 SD | Match out of fold: Σ q − goals (SD) |
|---|---|---|---|---|---|---|---|---|
| mouth_y_abs | [0, 1) | all | 8,995 | 1,331 | +34.3 | 28.5 | yes | +32.6 (28.6) |
| mouth_y_abs | [1, 2) | all | 9,053 | 1,850 | +31.0 | 31.9 | yes | +30.3 (32.0) |
| mouth_y_abs | [2, 3) | all | 9,890 | 3,805 | −8.0 | 39.7 | yes | −10.6 (39.8) |
| mouth_y_abs | [3, 3.3) | all | 3,382 | 2,105 | −42.6 | 24.8 | yes | −40.1 (24.8) |
| mouth_y_abs | [3.3, 3.55) | all | 970 | 572 | +1.3 | 13.3 | yes | +0.6 (13.3) |
| mouth_y_abs | >= 3.55 | all | 159 | 55 | +0.1 | 5.4 | yes | +0.2 (5.4) |
| mouth_y_abs | outside the frame + 0.15 m | all | 34 | 3 | +8.8 | 2.5 | not judged (n < 100) | +8.4 (2.5) |
| mouth_y_abs | [0, 1) | holdout_fit | 916 | 119 | +11.1 | 8.9 | yes | +13.0 (9.0) |
| mouth_y_abs | [1, 2) | holdout_fit | 867 | 158 | +8.6 | 9.6 | yes | +8.3 (9.6) |
| mouth_y_abs | [2, 3) | holdout_fit | 921 | 376 | −15.6 | 12.1 | yes | −17.3 (12.2) |
| mouth_y_abs | [3, 3.3) | holdout_fit | 310 | 190 | −5.0 | 7.5 | yes | −4.7 (7.5) |
| mouth_y_abs | [3.3, 3.55) | holdout_fit | 104 | 67 | −2.0 | 4.4 | yes | −2.4 (4.4) |
| mouth_y_abs | >= 3.55 | holdout_fit | 12 | 3 | +2.0 | 1.6 | not judged (n < 100) | +2.0 (1.6) |
| mouth_y_abs | outside the frame + 0.15 m | holdout_fit | 4 | 0 | +1.1 | 0.8 | not judged (n < 100) | +1.1 (0.8) |
| mouth_y_abs | [0, 1) | season_folds | 8,079 | 1,212 | +23.3 | 27.1 | yes | +19.6 (27.1) |
| mouth_y_abs | [1, 2) | season_folds | 8,186 | 1,692 | +22.4 | 30.4 | yes | +22.0 (30.5) |
| mouth_y_abs | [2, 3) | season_folds | 8,969 | 3,429 | +7.6 | 37.8 | yes | +6.6 (37.9) |
| mouth_y_abs | [3, 3.3) | season_folds | 3,072 | 1,915 | −37.6 | 23.6 | yes | −35.4 (23.6) |
| mouth_y_abs | [3.3, 3.55) | season_folds | 866 | 505 | +3.3 | 12.5 | yes | +3.0 (12.6) |
| mouth_y_abs | >= 3.55 | season_folds | 147 | 52 | −1.9 | 5.1 | yes | −1.8 (5.1) |
| mouth_y_abs | outside the frame + 0.15 m | season_folds | 30 | 3 | +7.7 | 2.4 | not judged (n < 100) | +7.3 (2.3) |
| mouth_z | [0, 0.3) | all | 7,476 | 2,458 | −22.4 | 31.6 | yes | −20.5 (31.7) |
| mouth_z | [0.3, 0.6) | all | 7,003 | 2,160 | −0.4 | 30.3 | yes | −8.9 (30.3) |
| mouth_z | [0.6, 1) | all | 4,778 | 1,307 | +31.6 | 24.7 | yes | +33.3 (24.7) |
| mouth_z | [1, 1.5) | all | 5,185 | 1,313 | +31.1 | 25.5 | yes | +34.4 (25.6) |
| mouth_z | [1.5, 2) | all | 5,954 | 1,702 | −8.9 | 27.6 | yes | −9.1 (27.7) |
| mouth_z | [2, 2.3) | all | 2,043 | 760 | −0.1 | 17.0 | yes | −1.3 (17.0) |
| mouth_z | >= 2.3 | all | 44 | 21 | −6.1 | 2.5 | not judged (n < 100) | −6.6 (2.6) |
| mouth_z | [0, 0.3) | holdout_fit | 823 | 257 | −9.9 | 10.2 | yes | −9.7 (10.2) |
| mouth_z | [0.3, 0.6) | holdout_fit | 646 | 188 | −1.4 | 9.1 | yes | −2.2 (9.1) |
| mouth_z | [0.6, 1) | holdout_fit | 417 | 123 | −3.6 | 7.2 | yes | −2.7 (7.2) |
| mouth_z | [1, 1.5) | holdout_fit | 461 | 110 | +10.8 | 7.7 | yes | +11.1 (7.7) |
| mouth_z | [1.5, 2) | holdout_fit | 583 | 166 | +0.1 | 8.5 | yes | −1.1 (8.5) |
| mouth_z | [2, 2.3) | holdout_fit | 199 | 67 | +4.7 | 5.3 | yes | +5.1 (5.3) |
| mouth_z | >= 2.3 | holdout_fit | 5 | 2 | −0.6 | 0.9 | not judged (n < 100) | −0.5 (0.9) |
| mouth_z | [0, 0.3) | season_folds | 6,653 | 2,201 | −12.5 | 29.9 | yes | −10.8 (30.0) |
| mouth_z | [0.3, 0.6) | season_folds | 6,357 | 1,972 | +1.0 | 28.9 | yes | −6.7 (28.9) |
| mouth_z | [0.6, 1) | season_folds | 4,361 | 1,184 | +35.1 | 23.7 | yes | +36.0 (23.6) |
| mouth_z | [1, 1.5) | season_folds | 4,724 | 1,203 | +20.3 | 24.3 | yes | +23.3 (24.4) |
| mouth_z | [1.5, 2) | season_folds | 5,371 | 1,536 | −8.9 | 26.3 | yes | −7.9 (26.3) |
| mouth_z | [2, 2.3) | season_folds | 1,844 | 693 | −4.8 | 16.2 | yes | −6.4 (16.1) |
| mouth_z | >= 2.3 | season_folds | 39 | 19 | −5.4 | 2.4 | not judged (n < 100) | −6.1 (2.4) |
| angle_rad | [0, 0.3) | all | 10,196 | 1,089 | +15.4 | 28.1 | yes | +12.9 (28.1) |
| angle_rad | [0.3, 0.4) | all | 7,166 | 1,686 | +6.9 | 30.0 | yes | +10.7 (30.1) |
| angle_rad | [0.4, 0.5) | all | 4,230 | 1,445 | +4.6 | 25.8 | yes | +6.7 (25.8) |
| angle_rad | [0.5, 0.7) | all | 4,792 | 1,845 | +1.3 | 28.5 | yes | −4.4 (28.5) |
| angle_rad | [0.7, 1) | all | 3,616 | 1,832 | −13.4 | 25.9 | yes | −13.2 (26.0) |
| angle_rad | >= 1 | all | 2,483 | 1,824 | +10.1 | 19.9 | yes | +8.8 (19.9) |
| angle_rad | [0, 0.3) | holdout_fit | 1,024 | 110 | +3.2 | 9.0 | yes | +3.0 (9.0) |
| angle_rad | [0.3, 0.4) | holdout_fit | 710 | 163 | −2.8 | 9.2 | yes | −2.6 (9.3) |
| angle_rad | [0.4, 0.5) | holdout_fit | 372 | 135 | −0.8 | 7.7 | yes | −0.1 (7.8) |
| angle_rad | [0.5, 0.7) | holdout_fit | 440 | 182 | −16.6 | 8.5 | yes | −16.8 (8.6) |
| angle_rad | [0.7, 1) | holdout_fit | 367 | 164 | +8.9 | 8.2 | yes | +8.5 (8.2) |
| angle_rad | >= 1 | holdout_fit | 221 | 159 | +8.1 | 5.7 | yes | +8.0 (5.7) |
| angle_rad | [0, 0.3) | season_folds | 9,172 | 979 | +12.3 | 26.7 | yes | +9.8 (26.7) |
| angle_rad | [0.3, 0.4) | season_folds | 6,456 | 1,523 | +9.7 | 28.6 | yes | +13.3 (28.6) |
| angle_rad | [0.4, 0.5) | season_folds | 3,858 | 1,310 | +5.3 | 24.7 | yes | +6.8 (24.6) |
| angle_rad | [0.5, 0.7) | season_folds | 4,352 | 1,663 | +17.9 | 27.2 | yes | +12.4 (27.2) |
| angle_rad | [0.7, 1) | season_folds | 3,249 | 1,668 | −22.3 | 24.6 | yes | −21.8 (24.7) |
| angle_rad | >= 1 | season_folds | 2,262 | 1,665 | +2.0 | 19.0 | yes | +0.8 (19.1) |

Limitation: 0 of 53 judged bands are outside 2 SD.

Keeper-level effect: rated from the match out-of-fold xGOT instead, the 97 keeper-seasons with >= 100 shots move by a median of 0.30 goals (max 1.12; their mean SD 4.08).

### goal_z sensitivity

A projected save's `goal_z` is where the keeper met the ball, not its height at the line. Re-rated on the goals and the saves stopped within 1 yd of the line (`end_x` >= 104.0375 m; δ and E2's minimum re-fitted on that subset): 3,176 of 22,762 open-play saves are kept and 19,586 (86.0%) left out; only saves are left out, every goal stays. Over the 89 keeper-seasons with >= 100 shots in competition-seasons with a δ fitted on that subset, GSAx moves by a median of 2.14 goals (90th percentile 5.13, max 10.98; their mean SD 4.10); Spearman with the headline +0.66. The shift tracks the share of saves stopped off the line (Spearman -0.71 with each keeper-season's far saves per shot). Open play only; a sensitivity beside the headline, not a rating.

Left out of both summaries: 8 keeper-seasons with >= 100 shots in competition-seasons at δ = 0 on the subset (team-selected, or E2-small there; median |shift| 13.99). A δ = 0 season cannot re-level a subset selected on the outcome, so their shift is that selection, not `goal_z`.

Without `mouth_z`: the headline xGOT refitted season-out without it (no selection on the outcome; open play only, the penalty xGOT reads it too) moves the same 89 keeper-seasons by a median of 0.39 goals (90th percentile 1.01, max 1.70; their mean SD 4.10); Spearman with the headline +0.99, and its shift with the far-save share +0.17. A sensitivity, not the headline.

## Checks

| Target | Measured | Result |
|---|---|---|
| Every rated row's xGOT from fits that never saw its season (season_out raises otherwise); the rating reads those rows, not the match out-of-fold xGOT | season_folds 30,264, holdout_fit 3,219; 33,398 of 33,483 differ from the match out-of-fold xGOT | met |
| Keeper totals sum to the league sum of xGOT − goals (raw and rated), both variants, within 1e-09; table shots = rated rows | max abs. difference < 1e-12 | met |
| Rated against the league sum: abs(Σ (q − goal)) <= 1e-09 per rating group (open play: one δ per competition-season; penalties: one δ per season-out source, PM 2026-10-28 (a)), groups at δ = 0 (E2-small, team-selected or an E2-small pool) excepted | max < 1e-12; 22 open-play and 2 penalty groups rated; 48 team-selected competition-seasons (26 also small), 0 other small, 0 penalty pool(s) at δ = 0 by E2; Σ (q − goal) at δ = 0: team-selected +14.4, other small +0.0 | met |
| Analytic SD and 90% match-bootstrap CI per keeper-season | SD finite; CI null on the 401 one-match keeper-seasons | met |
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0; judged on the keeper-season grouping, the career grouping (1 | keeper) beside it) | keeper-season +0.03759 [+0.02901, +0.04714] on 5,866 shots, 227 pairs; career +0.02228 [+0.01525, +0.03001] | met |
| Split-half reliability at 100 / 250 / 500 shots faced, empty tiers reported (a reporting check: met when every tier is reported) | 100: 111 keepers; 250: 6 keepers; 500: 2 keepers | met |

## Uncertainty

`sd` is √Σ q(1 − q): the spread of GSAx for a league-average keeper facing these shots, not the uncertainty of this keeper's estimate. The 90% interval resamples the keeper-season's matches (2,000 draws, seed 0) with the rating held fixed; with few matches it undercovers. Against a null with goals drawn once from q (no keeper differences): 18 of 97 keeper-seasons with >= 100 shots have a 90% CI excluding 0; 11 would with no keeper differences. Over the 804 keeper-seasons with >= 2 matches: 270 observed, 249 under the null.

## Stability

Split-half reliability of q − goal per shot (without penalties), shots alternated within a keeper by match date, period, time and shot id (PM X5), over the keepers whose total shots faced (all seasons) reach the tier: first on each keeper's first T shots, then on all their shots; Spearman-Brown steps r up to the whole. 95% intervals by Fisher z. A tier is judged only with at least 30 keepers.

| Shots faced (tier) | Keepers | First T shots: r [95% CI] | Spearman-Brown | All their shots: r [95% CI] | Spearman-Brown | Reading |
|---|---|---|---|---|---|---|
| 100 | 111 | +0.06 [-0.12, +0.25] | +0.12 [-0.28, +0.40] | +0.19 [+0.01, +0.37] | +0.32 [+0.01, +0.54] | not distinguishable from noise at 100 shots |
| 250 | 6 | +0.83 [+0.05, +0.98] | +0.91 [+0.09, +0.99] | +0.73 [-0.21, +0.97] | +0.84 [-0.53, +0.98] | 6 keepers reach 250 shots: too few to judge (r +0.83 [+0.05, +0.98]) |
| 500 | 2 | n/a | n/a | n/a | n/a | 2 keepers reach 500 shots: too few to estimate |

Team-selected competition-seasons (†) not E2-small: 22, at δ = 0 (PM 2026-10-28). By side (the selected side's keeper faces at least 30 shots that season): selected side: 22 keeper-seasons, 1,939 shots, GSAx +78.9 (SD 15.5); opponents: 470 keeper-seasons, 4,329 shots, GSAx −59.4 (SD 24.9); the seasons' Σ (q − goal) +19.5. δ = 0 removes the league-sum coupling, not the gap between the sides.

## Keeper-seasons with >= 100 shots (without penalties, sorted by GSAx)

97 keeper-seasons. Names are the lineup display names (the nickname when StatsBomb gives one); 1 table keeper(s) appear under two spellings, resolved to the latest match. The chart is written by `make gsax` to `data/charts/gsax-keepers.svg` (git-ignored: it embeds the StatsBomb logo, PM E3); its SHA-256 is `9dd6c0513418edaaf004048fa8250dbb057956f57a4c2aa1bb3ddcf5ebdbba3c`, so `make reproduce` checks the chart through this card.

| # | Keeper | Family | Competition season | Shots | Goals | Open goals | Open-goal GSAx | GSAx raw | GSAx | SD | 90% CI | Shrunken GSAx | GSAx with penalties |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Samir Handanovič | league or tournament | Serie A 2015/2016 | 140 | 28 | 3 | +0.7 | +10.7 | +10.9 | 4.0 | [+3.5, +18.3] | +2.7 | +10.9 |
| 2 | Keylor Navas | league or tournament | La Liga 2015/2016 | 118 | 27 | 1 | -0.1 | +9.6 | +9.9 | 4.0 | [+4.4, +15.3] | +2.4 | +12.0 |
| 3 | Marc-André ter Stegen | team-selected † | La Liga 2017/2018 | 116 | 21 | 3 | -0.6 | +9.8 | +9.8 | 3.8 | [+4.4, +15.2] | +2.2 | +9.5 |
| 4 | Víctor Valdés | team-selected † | La Liga 2009/2010 | 104 | 20 | 0 | +0.0 | +9.3 | +9.3 | 3.5 | [+3.4, +15.3] | +1.9 | +8.9 |
| 5 | Claudio Bravo | league or tournament | La Liga 2015/2016 | 103 | 19 | 2 | -0.3 | +9.0 | +9.2 | 3.6 | [+4.6, +14.5] | +1.9 | +9.0 |
| 6 | Vincent Enyeama | league or tournament | Ligue 1 2015/2016 | 116 | 22 | 2 | -0.6 | +6.8 | +8.5 | 3.7 | [+2.3, +15.1] | +1.9 | +8.2 |
| 7 | Łukasz Fabiański | league or tournament | Premier League 2015/2016 | 165 | 44 | 4 | -0.4 | +7.6 | +8.4 | 4.5 | [+0.6, +17.1] | +2.4 | +8.3 |
| 8 | Antonio Mirante | league or tournament | Serie A 2015/2016 | 132 | 34 | 4 | -1.6 | +6.6 | +6.9 | 4.4 | [+0.5, +13.9] | +1.9 | +6.2 |
| 9 | Wojciech Szczęsny | league or tournament | Serie A 2015/2016 | 129 | 27 | 4 | -0.6 | +6.4 | +6.6 | 3.9 | [+1.1, +12.6] | +1.5 | +5.8 |
| 10 | Emiliano Viviano | league or tournament | Serie A 2015/2016 | 185 | 49 | 6 | +0.1 | +6.1 | +6.5 | 4.9 | [-1.7, +15.2] | +2.1 | +8.5 |
| 11 | Steve Mandanda | league or tournament | Ligue 1 2015/2016 | 126 | 31 | 1 | -0.3 | +4.5 | +6.5 | 4.0 | [-1.2, +13.8] | +1.6 | +5.9 |
| 12 | Gianluigi Donnarumma | team-selected † | Ligue 1 2022/2023 | 129 | 29 | 2 | -0.2 | +6.4 | +6.4 | 4.1 | [-0.3, +13.6] | +1.7 | +6.9 |
| 13 | Doris Bačić | women | Serie A Women 2023/2024 | 124 | 28 | 1 | -0.1 | +8.7 | +6.3 | 3.9 | [+0.7, +11.5] | +1.5 | +7.1 |
| 14 | Marc-André ter Stegen | team-selected † | La Liga 2016/2017 | 131 | 27 | 3 | -1.2 | +6.3 | +6.3 | 4.0 | [+1.9, +11.0] | +1.5 | +6.3 |
| 15 | Courtney Brosnan | women | FA Women's Super League 2023/2024 | 123 | 32 | 5 | -1.0 | +7.5 | +6.2 | 4.2 | [-2.6, +15.3] | +1.6 | +6.5 |
| 16 | Javi Varas | league or tournament | La Liga 2015/2016 | 137 | 38 | 5 | -1.0 | +5.6 | +5.9 | 4.3 | [-0.7, +12.5] | +1.6 | +5.7 |
| 17 | Marc-André ter Stegen | team-selected † | La Liga 2018/2019 | 108 | 27 | 2 | -0.2 | +5.4 | +5.4 | 3.8 | [+0.0, +11.5] | +1.2 | +5.2 |
| 18 | Katie Lund | women | NWSL 2023 | 111 | 21 | 0 | +0.0 | +9.0 | +5.3 | 3.6 | [+0.9, +9.5] | +1.1 | +4.4 |
| 19 | David de Gea | league or tournament | Premier League 2015/2016 | 111 | 28 | 1 | -0.7 | +4.7 | +5.3 | 3.8 | [+0.1, +10.7] | +1.2 | +5.0 |
| 20 | Jan Oblak | league or tournament | La Liga 2015/2016 | 102 | 17 | 2 | -0.3 | +4.9 | +5.1 | 3.2 | [+0.4, +9.5] | +0.9 | +4.9 |
| 21 | Lukáš Hrádecký | team-selected † | 1. Bundesliga 2023/2024 | 104 | 24 | 2 | -0.4 | +5.0 | +5.0 | 3.7 | [-0.5, +10.8] | +1.1 | +5.0 |
| 22 | Hugo Lloris | league or tournament | Premier League 2015/2016 | 119 | 30 | 2 | -0.1 | +4.3 | +4.9 | 3.9 | [-0.8, +10.7] | +1.2 | +4.6 |
| 23 | Nayluisa Jhaylenny Cáceres Acevedo | women | Liga F 2023/2024 | 140 | 49 | 2 | -1.0 | +2.1 | +4.9 | 4.4 | [-2.3, +12.6] | +1.4 | +3.2 |
| 24 | Vicente Guaita | league or tournament | La Liga 2015/2016 | 182 | 58 | 4 | -0.6 | +4.5 | +4.9 | 4.9 | [-2.8, +13.0] | +1.6 | +4.7 |
| 25 | Heurelho Gomes | league or tournament | Premier League 2015/2016 | 166 | 43 | 2 | -0.9 | +3.9 | +4.8 | 4.6 | [-2.1, +11.9] | +1.5 | +6.9 |
| 26 | Elvira Herzog | women | Frauen Bundesliga 2023/2024 | 105 | 38 | 4 | -1.3 | +3.8 | +4.7 | 3.9 | [-2.1, +11.5] | +1.1 | +4.1 |
| 27 | Zala Meršnik | women | Liga F 2023/2024 | 139 | 41 | 0 | +0.0 | +2.0 | +4.6 | 4.2 | [-1.1, +10.5] | +1.2 | +6.0 |
| 28 | Jens Lehmann | team-selected † | Premier League 2003/2004 | 110 | 22 | 1 | -0.2 | +4.3 | +4.3 | 3.6 | [-1.1, +9.7] | +0.9 | +3.9 |
| 29 | Petr Čech | league or tournament | Premier League 2015/2016 | 127 | 27 | 0 | +0.0 | +3.3 | +4.0 | 3.9 | [-4.1, +12.0] | +1.0 | +3.9 |
| 30 | Adrián | league or tournament | Premier League 2015/2016 | 130 | 32 | 3 | -0.9 | +3.2 | +3.9 | 4.0 | [-2.9, +10.2] | +1.0 | +5.0 |
| 31 | Stefano Sorrentino | league or tournament | Serie A 2015/2016 | 174 | 49 | 4 | -0.7 | +3.4 | +3.8 | 4.7 | [-4.7, +12.1] | +1.2 | +2.6 |
| 32 | Jean-Louis Leca | league or tournament | Ligue 1 2015/2016 | 115 | 30 | 3 | -0.5 | +1.9 | +3.5 | 3.7 | [-3.7, +10.5] | +0.8 | +3.9 |
| 33 | Kasper Schmeichel | league or tournament | Premier League 2015/2016 | 134 | 32 | 2 | -0.4 | +2.8 | +3.5 | 4.1 | [-1.5, +8.3] | +0.9 | +2.9 |
| 34 | Stéphane Ruffier | league or tournament | Ligue 1 2015/2016 | 140 | 33 | 2 | -0.8 | +1.1 | +3.3 | 4.3 | [-4.9, +10.9] | +0.9 | +3.8 |
| 35 | Jane Campbell | women | NWSL 2023 | 109 | 15 | 1 | -0.2 | +6.5 | +3.3 | 3.3 | [-2.5, +8.7] | +0.6 | +2.7 |
| 36 | Albano Bizzarri | league or tournament | Serie A 2015/2016 | 160 | 39 | 4 | -0.9 | +2.4 | +2.7 | 4.2 | [-5.2, +10.3] | +0.7 | +3.7 |
| 37 | Danijel Subašić | league or tournament | Ligue 1 2015/2016 | 165 | 48 | 2 | -0.2 | -0.0 | +2.3 | 4.4 | [-3.9, +8.6] | +0.7 | +3.8 |
| 38 | Mária Korenčiová | women | Serie A Women 2023/2024 | 113 | 32 | 4 | -0.5 | +4.6 | +2.2 | 3.9 | [-2.5, +6.8] | +0.5 | +2.9 |
| 39 | Anthony Lopes | league or tournament | Ligue 1 2015/2016 | 141 | 39 | 2 | -0.7 | -0.2 | +2.1 | 4.4 | [-6.7, +10.5] | +0.6 | +1.8 |
| 40 | Amanda Tampieri | women | Serie A Women 2023/2024 | 124 | 30 | 4 | -0.5 | +4.6 | +2.1 | 4.0 | [-3.3, +7.7] | +0.5 | +2.7 |
| 41 | Jack Butland | league or tournament | Premier League 2015/2016 | 135 | 34 | 3 | -0.5 | +1.1 | +1.8 | 4.1 | [-5.2, +8.9] | +0.5 | +1.1 |
| 42 | Gorka Iraizoz | league or tournament | La Liga 2015/2016 | 122 | 31 | 1 | -0.4 | +1.4 | +1.7 | 4.0 | [-6.0, +9.3] | +0.4 | +1.7 |
| 43 | Anna Rosa James-Buhigas | women | Serie A Women 2023/2024 | 106 | 34 | 3 | -0.4 | +3.9 | +1.5 | 3.9 | [-5.1, +7.6] | +0.4 | +1.6 |
| 44 | Andrea Consigli | league or tournament | Serie A 2015/2016 | 137 | 35 | 0 | +0.0 | +1.2 | +1.5 | 4.1 | [-5.3, +8.1] | +0.4 | +2.4 |
| 45 | Tim Howard | league or tournament | Premier League 2015/2016 | 118 | 31 | 1 | -0.1 | +0.3 | +0.9 | 3.7 | [-4.7, +6.0] | +0.2 | +0.5 |
| 46 | Mattia Perin | league or tournament | Serie A 2015/2016 | 112 | 28 | 2 | -0.5 | +0.3 | +0.5 | 3.7 | [-6.0, +6.3] | +0.1 | +1.3 |
| 47 | Sergio Rico | league or tournament | La Liga 2015/2016 | 149 | 39 | 1 | -0.2 | +0.2 | +0.5 | 4.3 | [-4.8, +5.9] | +0.1 | +0.3 |
| 48 | Solène Durand | women | Serie A Women 2023/2024 | 127 | 33 | 3 | +0.3 | +3.1 | +0.5 | 4.1 | [-7.5, +7.4] | +0.1 | -0.2 |
| 49 | Alphonse Areola | league or tournament | La Liga 2015/2016 | 106 | 24 | 2 | -0.7 | +0.3 | +0.5 | 3.5 | [-4.9, +5.8] | +0.1 | +1.0 |
| 50 | Rémy Vercoutre | league or tournament | Ligue 1 2015/2016 | 169 | 48 | 0 | +0.0 | -2.3 | +0.4 | 4.8 | [-8.3, +9.4] | +0.1 | -0.1 |
| 51 | Paula Vizoso | women | Liga F 2023/2024 | 154 | 51 | 5 | -1.6 | -2.6 | +0.4 | 4.5 | [-6.4, +6.8] | +0.1 | +0.1 |
| 52 | Germán Lux | league or tournament | La Liga 2015/2016 | 109 | 34 | 2 | -0.5 | -0.0 | +0.2 | 3.8 | [-7.2, +7.9] | +0.0 | +0.4 |
| 53 | Gianluigi Donnarumma | league or tournament | Serie A 2015/2016 | 109 | 26 | 0 | +0.0 | -0.1 | +0.1 | 3.6 | [-5.7, +6.3] | +0.0 | -0.4 |
| 54 | Gero Rulli | league or tournament | La Liga 2015/2016 | 144 | 39 | 6 | -1.5 | -0.2 | +0.1 | 4.2 | [-6.7, +6.6] | +0.0 | +0.3 |
| 55 | Lisa Weiß | women | FA Women's Super League 2020/2021 | 115 | 39 | 2 | -0.3 | +1.0 | -0.0 | 4.0 | [-6.3, +6.3] | -0.0 | -0.2 |
| 56 | Łukasz Skorupski | league or tournament | Serie A 2015/2016 | 127 | 41 | 1 | -0.3 | -0.4 | -0.1 | 4.1 | [-8.5, +8.2] | -0.0 | -0.5 |
| 57 | Diego Mariño | league or tournament | La Liga 2015/2016 | 116 | 37 | 4 | -0.0 | -1.0 | -0.7 | 3.9 | [-8.2, +6.4] | -0.2 | -1.3 |
| 58 | Sophie Baggaley | women | FA Women's Super League 2023/2024 | 120 | 40 | 3 | -0.9 | +0.5 | -0.8 | 4.1 | [-8.0, +6.8] | -0.2 | -0.8 |
| 59 | Wayne Hennessey | league or tournament | Premier League 2015/2016 | 114 | 33 | 3 | -0.9 | -1.4 | -0.8 | 3.7 | [-7.1, +5.0] | -0.2 | -1.1 |
| 60 | Orestis Karnezis | league or tournament | Serie A 2015/2016 | 162 | 52 | 5 | -1.0 | -1.6 | -1.3 | 4.7 | [-8.1, +5.7] | -0.4 | -2.0 |
| 61 | Clément Maury | league or tournament | Ligue 1 2015/2016 | 139 | 46 | 1 | -0.4 | -3.7 | -1.3 | 4.5 | [-8.3, +5.6] | -0.4 | -2.1 |
| 62 | Pichu | league or tournament | La Liga 2015/2016 | 158 | 43 | 6 | -1.5 | -1.7 | -1.4 | 4.3 | [-8.0, +6.1] | -0.4 | -1.5 |
| 63 | Vid Belec | league or tournament | Serie A 2015/2016 | 126 | 37 | 4 | -1.1 | -1.7 | -1.4 | 4.0 | [-6.6, +3.9] | -0.3 | -1.1 |
| 64 | Antonio Adán | league or tournament | La Liga 2015/2016 | 176 | 49 | 4 | -0.8 | -2.0 | -1.6 | 4.7 | [-10.6, +7.6] | -0.5 | -1.6 |
| 65 | Antonia Ignacia Canales Pacheco | women | Liga F 2023/2024 | 126 | 38 | 3 | -0.7 | -4.0 | -1.7 | 4.0 | [-9.1, +5.6] | -0.4 | -1.2 |
| 66 | Bernd Leno | team-selected † | 1. Bundesliga 2015/2016 | 128 | 32 | 2 | -0.9 | -1.8 | -1.8 | 3.9 | [-9.0, +5.5] | -0.4 | -1.9 |
| 67 | Benoît Guy Robert Costil | league or tournament | Ligue 1 2015/2016 | 133 | 42 | 1 | -0.2 | -4.0 | -1.8 | 4.3 | [-8.5, +5.2] | -0.5 | -1.4 |
| 68 | Elene Lete | women | Liga F 2023/2024 | 151 | 49 | 3 | -1.0 | -4.8 | -1.8 | 4.5 | [-9.3, +5.0] | -0.5 | -2.0 |
| 69 | Esther Sullastres | women | Liga F 2023/2024 | 128 | 37 | 3 | -0.9 | -4.6 | -2.3 | 4.0 | [-10.9, +5.2] | -0.6 | -1.7 |
| 70 | Joe Hart | league or tournament | Premier League 2015/2016 | 107 | 33 | 2 | -0.3 | -3.0 | -2.4 | 3.8 | [-7.7, +3.3] | -0.5 | -2.5 |
| 71 | Nicola Leali | league or tournament | Serie A 2015/2016 | 176 | 54 | 4 | -0.4 | -2.8 | -2.5 | 4.6 | [-10.1, +5.6] | -0.8 | -2.5 |
| 72 | Ena Mahmutovic | women | Frauen Bundesliga 2023/2024 | 145 | 54 | 4 | -0.8 | -3.9 | -2.6 | 4.6 | [-11.5, +6.2] | -0.8 | -3.1 |
| 73 | Hannah Hampton | women | FA Women's Super League 2020/2021 | 126 | 42 | 4 | -0.4 | -1.7 | -2.7 | 4.0 | [-9.5, +3.9] | -0.7 | -1.4 |
| 74 | Mackenzie Arnold | women | FA Women's Super League 2023/2024 | 113 | 40 | 0 | +0.0 | -1.6 | -2.7 | 3.9 | [-7.1, +1.5] | -0.7 | -1.9 |
| 75 | Sophie Baggaley | women | FA Women's Super League 2020/2021 | 159 | 61 | 4 | -0.5 | -1.5 | -3.1 | 4.9 | [-12.7, +6.6] | -1.0 | -3.4 |
| 76 | Andrés Fernández | league or tournament | La Liga 2015/2016 | 163 | 60 | 9 | -2.0 | -3.5 | -3.1 | 4.7 | [-10.4, +3.9] | -1.0 | -4.1 |
| 77 | Benjamin Lecomte | league or tournament | Ligue 1 2015/2016 | 170 | 50 | 4 | -0.6 | -5.8 | -3.2 | 4.6 | [-11.8, +5.7] | -1.0 | -3.8 |
| 78 | Pau López | league or tournament | La Liga 2015/2016 | 158 | 56 | 8 | -1.7 | -3.6 | -3.3 | 4.5 | [-10.7, +5.1] | -1.0 | -4.2 |
| 79 | Sandra Estévez | women | Liga F 2023/2024 | 118 | 45 | 10 | -1.2 | -5.7 | -3.3 | 4.1 | [-9.6, +2.8] | -0.8 | -3.6 |
| 80 | Marco Sportiello | league or tournament | Serie A 2015/2016 | 144 | 38 | 2 | -0.9 | -3.7 | -3.4 | 4.2 | [-9.8, +2.4] | -0.9 | -1.8 |
| 81 | Grace Moloney | women | FA Women's Super League 2020/2021 | 103 | 39 | 5 | -1.3 | -2.6 | -3.5 | 3.8 | [-12.6, +5.7] | -0.8 | -3.2 |
| 82 | Simon Mignolet | league or tournament | Premier League 2015/2016 | 109 | 39 | 3 | -0.4 | -4.3 | -3.7 | 3.9 | [-11.9, +4.6] | -0.9 | -3.0 |
| 83 | Federico Marchetti | league or tournament | Serie A 2015/2016 | 110 | 34 | 2 | -0.9 | -4.3 | -4.1 | 3.7 | [-11.8, +3.1] | -0.9 | -3.1 |
| 84 | Pierluigi Gollini | league or tournament | Serie A 2015/2016 | 109 | 38 | 2 | -0.3 | -4.4 | -4.1 | 3.7 | [-9.9, +1.5] | -0.9 | -5.1 |
| 85 | Sergio Álvarez | league or tournament | La Liga 2015/2016 | 130 | 42 | 3 | -0.5 | -4.4 | -4.2 | 4.1 | [-10.1, +1.6] | -1.1 | -4.0 |
| 86 | Sophie Baggaley | women | FA Women's Super League 2019/2020 | 108 | 37 | 2 | -0.2 | -3.1 | -4.3 | 3.8 | [-13.4, +3.9] | -1.0 | -3.0 |
| 87 | John Ruddy | league or tournament | Premier League 2015/2016 | 108 | 39 | 3 | +0.5 | -5.2 | -4.6 | 3.9 | [-11.3, +1.1] | -1.1 | -5.2 |
| 88 | Carmen Carbonel | women | Liga F 2023/2024 | 118 | 41 | 3 | -1.1 | -7.0 | -4.9 | 3.9 | [-9.4, -0.5] | -1.2 | -5.5 |
| 89 | Ciprian Tătăruşanu | league or tournament | Serie A 2015/2016 | 107 | 39 | 2 | -0.4 | -5.5 | -5.3 | 3.8 | [-11.6, +0.7] | -1.2 | -5.3 |
| 90 | Jonas Lössl | league or tournament | Ligue 1 2015/2016 | 158 | 46 | 7 | -1.6 | -7.5 | -5.3 | 4.3 | [-12.5, +1.4] | -1.5 | -4.8 |
| 91 | Daniele Padelli | league or tournament | Serie A 2015/2016 | 130 | 41 | 2 | -0.3 | -5.9 | -5.6 | 4.1 | [-11.7, +0.4] | -1.5 | -6.9 |
| 92 | Asier Riesgo | league or tournament | La Liga 2015/2016 | 129 | 50 | 7 | -1.1 | -6.1 | -5.8 | 4.3 | [-13.2, +1.1] | -1.6 | -5.9 |
| 93 | Paul Bernardoni | league or tournament | Ligue 1 2015/2016 | 113 | 36 | 4 | -0.6 | -8.0 | -6.4 | 3.7 | [-12.7, -0.4] | -1.4 | -6.3 |
| 94 | Artur Boruc | league or tournament | Premier League 2015/2016 | 120 | 48 | 2 | -0.4 | -7.4 | -6.7 | 4.1 | [-13.8, +0.2] | -1.7 | -5.3 |
| 95 | Alyssa Naeher | women | NWSL 2023 | 125 | 45 | 3 | -0.3 | -2.5 | -7.1 | 4.0 | [-16.7, +2.0] | -1.8 | -7.9 |
| 96 | Paola Ulloa | women | Liga F 2023/2024 | 128 | 51 | 1 | -0.3 | -9.8 | -7.2 | 4.2 | [-14.7, +0.4] | -1.9 | -6.9 |
| 97 | Brad Guzan | league or tournament | Premier League 2015/2016 | 142 | 53 | 4 | -0.9 | -8.6 | -7.8 | 4.4 | [-13.8, -2.1] | -2.2 | -8.1 |

## Columns

- `goals`: goals from rated shots only (own goals, direct corners and shoot-outs are not in it), so not goals conceded.
- `GSAx raw`: Σ xGOT − goals, before the league rating; `GSAx`: Σ q − goals.
- `SD`: the null SD √Σ q(1 − q); `90% CI`: the match bootstrap above.
- `family`: women; team-selected (one team in at least half of a competition-season's matches; rated at δ = 0); league or tournament.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
- v2 (PR1): WhoScored season-out from competition-season folds; one penalty δ per season-out source; team-selected competition-seasons at δ = 0 (flagged); goal_z sensitivity (StatsBomb).
