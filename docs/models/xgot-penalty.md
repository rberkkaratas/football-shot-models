# Model card: Penalty xGOT (placement only)

## Overview
- **Name / version:** `xgot-penalty` v4
- **Task:** P(goal | on-target penalty) from placement only; fitted on in-game kicks, shoot-out kicks scored apart
- **Status:** experimental
- P2.4: penalty xGOT, a logistic regression on placement only; it reads no xG, since every in-game penalty has the same pre-shot xG. Placement-only, reported apart from the open-play xGOT.

## Intended use
Penalty xGOT: P(goal) of an on-target penalty from where it crossed the line, kept apart from the open-play xGOT. For research and as the penalty input of the MS6 goalkeeper models; not a rating of takers or keepers on its own.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `067ddf5342cfd608045a15189a0b43008664d23f4c5ac4fab5ac403468e57ee5`
- **Competitions & seasons:** 49 competition-seasons; hold-out Premier League 2015/2016
- **Rows:** train and validation (dev) 915 shots in 797 matches / test (hold-out) 85 shots in 75 matches; men 681 shots, women 319
- **Families (segments):** women; team-selected, where one team plays in at least half of a competition-season's matches (56 competition-seasons, 30 of them with one or two matches, mostly single finals, where both teams qualify trivially); league or tournament, the rest
- **Filters:** excluded shots, by first matching reason. C3 on-target shots by first matching reason; penalty and shoot-out rows are scored by the penalty xGOT, modelled rows by the LR and GBM.

| Filter | Shots | Goals |
|---|---|---|
| coarse recording grid (whole-yard locations; the match is left out of the xGOT fit, PM MS6 E1) | 1,855 | 580 |
| own goal (Own Goal Against events; not shots) | 0 | 0 |
| shoot-out kick (period 5; penalty xGOT, reported apart) | 266 | 201 |
| penalty (in-game; penalty xGOT) | 1,000 | 808 |
| direct corner (logged as a shot on outcome) | 28 | 18 |
| no freeze frame | 0 | 0 |
| unresolved assisting event (left out of the xG population) | 0 | 0 |
| modelled (open-play xGOT) | 32,483 | 9,721 |

## Target
`is_goal` of an on-target shot: the ball went in rather than being saved. Blocked shots (clearances off the line included), woodwork and misses are not on target (C3 `on_target`), so they are not in the population.

## Features
| Feature | Definition | Source field(s) |
|---|---|---|
| `mouth_y_abs` | Distance from the goal centre across the goal, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_z` | Height above the ground, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_yz` | `mouth_y_abs` × `mouth_z` | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |

## Model
`make_pipeline(StandardScaler(), LogisticRegression(C=inf, solver='newton-cholesky', tol=1e-10, max_iter=1000))`: unpenalised logistic regression, no class weights, no calibration step; the scaler is fitted on the training rows of each fit. scikit-learn 1.9.1.

Coefficients of the fit on all of dev (natural units: log-odds per unit of the feature).

| Term | Standardised | Natural units |
|---|---|---|
| intercept | 1.4542 | -0.4519 |
| mouth_y_abs | 0.4491 | 0.5742 |
| mouth_z | 0.9416 | 1.5655 |
| mouth_yz | -0.6303 | -0.3826 |

## Validation design
Grouped 5-fold cross-validation by `match_id` on the in-game dev kicks (their own folds); the hold-out kicks are scored once by the fit on all of them, and the shoot-out kicks of every season by that fit too. The penalty xGOT reads no xG input.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4826 | 0.4856 ± 0.0400 | 0.3898 |
| Brier score | 0.1555 | 0.1564 ± 0.0165 | 0.1149 |
| ROC AUC | 0.6419 | 0.6367 ± 0.0577 | 0.5663 |
| Log-loss skill | 0.0296 | 0.0254 | 0.0293 |
| Brier skill | 0.0202 | 0.0161 | 0.0210 |
| Calibration intercept | 0.0000 | 0.0086 | 0.5145 |
| Calibration slope | 1.0000 | 0.9448 | 0.8948 |
| Σ xGOT | 734.0000 | 733.5803 | 68.3270 |
| Goals | 734 | 734 | 74 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.06% | 7.67% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.87% (max 4.96%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 92 | 65.11% | 76.09% |
| 1 | 91 | 73.70% | 65.93% |
| 2 | 92 | 77.11% | 70.65% |
| 3 | 91 | 79.03% | 79.12% |
| 4 | 92 | 80.51% | 80.43% |
| 5 | 91 | 81.71% | 82.42% |
| 6 | 92 | 83.03% | 82.61% |
| 7 | 91 | 84.69% | 76.92% |
| 8 | 92 | 86.90% | 91.30% |
| 9 | 91 | 90.03% | 96.70% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 398 | 310 | 77.89% | 318.5891 | 1.0277 | 0.5025 | 0.1645 | 0.0489 | 1.4821 |
| family = team-selected | 198 | 169 | 85.35% | 158.5514 | 0.9382 | 0.4305 | 0.1322 | -0.0335 | 0.3701 |
| gender = female | 319 | 255 | 79.94% | 256.4398 | 1.0056 | 0.4989 | 0.1613 | 0.0048 | 0.5821 |
| gender = male | 596 | 479 | 80.37% | 477.1405 | 0.9961 | 0.4786 | 0.1538 | 0.0337 | 1.0437 |
| body_part = right_foot | 670 | 537 | 80.15% | 537.4298 | 1.0008 | 0.4866 | 0.1566 | 0.0236 | 0.9043 |
| body_part = left_foot | 245 | 197 | 80.41% | 196.1506 | 0.9957 | 0.4831 | 0.1559 | 0.0234 | 0.8330 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0842 | 0.0966 | 0.0453 | 0.1278 | 0.1064 |

### Shoot-out kicks (reported apart)

266 on-target shoot-out kicks (201 goals), scored by the fit on all in-game dev kicks and never pooled with them: Σ xGOT 214.9, log loss 0.5609.

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

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Penalty base rate (on target) | 0.4983 | 0.1590 | 0.5000 | n/a | 0.4016 | 0.1173 | 0.5000 | n/a |
| Penalty xGOT (placement only) | 0.4856 | 0.1564 | 0.6367 | 0.9448 | 0.3898 | 0.1149 | 0.5663 | 0.8948 |

## Limitations & known biases
- Against the open-play bounds, each on its own: CV calibration slope 0.9448 (inside [0.9, 1.1]), hold-out slope 0.8948 (outside), CV ECE 0.0842 (fold mean; not below 0.02). Its probabilities are not calibrated to the open-play standard; as a ranking of placements its AUC is 0.6367 on CV and 0.5663 on the hold-out (0.5 is no ranking).
- 0 of 1,000 on-target shots (hold-out included) are recorded beyond the frame + 0.15 m and are clipped to its edge; 997 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 1,000 of 1,000 on-target shots, so there is no provider xGOT benchmark.
- StatsBomb records a save where the keeper stopped the ball: its `goal_z` is that height, not the height at the goal line, and a save stopped short of the line has its `goal_y` projected along the line from the shot location and snapped to the 0.1 yd goal-mouth lattice of recorded points (0 of the on-target saves were deflected, so their projection starts at the shot, not at the deflection).

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
- v3 (MS6): saves recorded where the keeper stopped the ball and projected to the line are snapped to the 0.1 yd goal-mouth lattice of recorded points (football-core); the GBM's placement features carry no monotone constraint (chosen by the band rule); the band table prints the SD; the band rule is a check.
- v4 (MS6, GSAx): matches recorded on the coarse (1 yd) grid leave the xGOT fit (PM E1); a band is judged by the band rule only with at least 100 shots.
