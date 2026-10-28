# Model card: LR xG + placement

## Overview
- **Name / version:** `xgot-lr` v4
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.3: logistic regression of the goal on the logit of the cross-fitted xG, placement splines, the width past the post's inner edge, angle × placement and the near / far post. Not the headline model; that is Monotone GBM + Platt.

## Intended use
xGOT (post-shot xG) for on-target shots: P(goal) of a shot that reached the goal frame, from where it crossed the line and its pre-shot xG. Keeper-neutral (no keeper position), so goals saved against it credit the keeper's positioning too. For research and as the input of the MS6 goalkeeper models; not a rating of finishers, and not for other providers' data without re-fitting.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `067ddf5342cfd608045a15189a0b43008664d23f4c5ac4fab5ac403468e57ee5`
- **Competitions & seasons:** 70 competition-seasons; hold-out Premier League 2015/2016
- **Rows:** train and validation (dev) 29,349 shots in 3,386 matches / test (hold-out) 3,134 shots in 380 matches; men 21,832 shots, women 10,651
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
xG input: Monotone GBM + Platt without keeper position, cross-fitted; provider headline xG: Monotone GBM + Platt.

| Feature | Definition | Source field(s) |
|---|---|---|
| `xg_logit` | Logit of the shot's cross-fitted xG (the xG input, out of sample for the row) | xG model |
| `mouth_y_abs` | Distance from the goal centre across the goal, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_y_abs_rcs1` | Restricted cubic spline term 1 of `mouth_y_abs`, knots 0.15, 0.9, 1.7, 2.5, 3.25 m | `mouth_y_abs` |
| `mouth_y_abs_rcs2` | Restricted cubic spline term 2 of `mouth_y_abs`, knots 0.15, 0.9, 1.7, 2.5, 3.25 m | `mouth_y_abs` |
| `mouth_y_abs_rcs3` | Restricted cubic spline term 3 of `mouth_y_abs`, knots 0.15, 0.9, 1.7, 2.5, 3.25 m | `mouth_y_abs` |
| `mouth_y_past` | `mouth_y_abs − mouth_y_in`: how far past that point the ball crossed, m | `mouth_y_abs` |
| `mouth_z` | Height above the ground, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_z_rcs1` | Restricted cubic spline term 1 of `mouth_z`, knots 0.1, 0.3, 0.8, 1.25, 2 m | `mouth_z` |
| `mouth_z_rcs2` | Restricted cubic spline term 2 of `mouth_z`, knots 0.1, 0.3, 0.8, 1.25, 2 m | `mouth_z` |
| `mouth_z_rcs3` | Restricted cubic spline term 3 of `mouth_z`, knots 0.1, 0.3, 0.8, 1.25, 2 m | `mouth_z` |
| `angle_x_mouth_y` | `angle_rad` × `mouth_y_abs` | `angle_rad`, `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `angle_x_mouth_z` | `angle_rad` × `mouth_z` | `angle_rad`, `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_far` | `far_side` × `mouth_y_abs` | `y`, `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |

## Model
`make_pipeline(StandardScaler(), LogisticRegression(C=inf, solver='newton-cholesky', tol=1e-10, max_iter=1000))`: unpenalised logistic regression, no class weights, no calibration step; the scaler is fitted on the training rows of each fit. scikit-learn 1.9.1.

Coefficients of the fit on all of dev (natural units: log-odds per unit of the feature).

| Term | Standardised | Natural units |
|---|---|---|
| intercept | -1.2217 | -0.3489 |
| xg_logit | 1.1416 | 0.8944 |
| mouth_y_abs | 0.0381 | 0.0377 |
| mouth_y_abs_rcs1 | 0.0960 | 0.0969 |
| mouth_y_abs_rcs2 | 1.5634 | 3.7935 |
| mouth_y_abs_rcs3 | -0.9084 | -9.1246 |
| mouth_y_past | -0.2102 | -17.3602 |
| mouth_z | -0.1209 | -0.1909 |
| mouth_z_rcs1 | -1.9520 | -3.5790 |
| mouth_z_rcs2 | 1.9998 | 5.3665 |
| mouth_z_rcs3 | 0.2473 | 2.5136 |
| angle_x_mouth_y | 0.3701 | 0.4665 |
| angle_x_mouth_z | 0.1407 | 0.2970 |
| mouth_far | 0.3733 | 0.1933 |

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. **Test** is the hold-out competition-season named under Data, scored once by that fit. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4173 | 0.4181 ± 0.0088 | 0.4092 |
| Brier score | 0.1336 | 0.1339 ± 0.0035 | 0.1312 |
| ROC AUC | 0.8622 | 0.8620 ± 0.0054 | 0.8652 |
| Log-loss skill | 0.3169 | 0.3157 | 0.3220 |
| Brier skill | 0.3638 | 0.3626 | 0.3646 |
| Calibration intercept | -0.0000 | 0.0001 | 0.0202 |
| Calibration slope | 1.0000 | 1.0000 | 0.9984 |
| Σ xGOT | 8808.0000 | 8808.9303 | 904.7917 |
| Goals | 8,808 | 8,808 | 913 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.01% | 0.90% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.48% (max 4.99%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| `mouth_far` (near / far post): dropped | +0.0106 [+0.0089, +0.0123] | +0.0075 [+0.0023, +0.0126] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 2,935 | 2.36% | 1.98% |
| 1 | 2,935 | 4.45% | 3.54% |
| 2 | 2,935 | 7.11% | 5.49% |
| 3 | 2,935 | 10.88% | 9.34% |
| 4 | 2,935 | 16.02% | 16.29% |
| 5 | 2,935 | 23.47% | 25.04% |
| 6 | 2,935 | 33.79% | 37.44% |
| 7 | 2,935 | 47.88% | 48.45% |
| 8 | 2,935 | 66.13% | 67.97% |
| 9 | 2,934 | 88.09% | 84.59% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 12,031 | 3,499 | 29.08% | 3418.6036 | 0.9770 | 0.4065 | 0.1299 | 0.3257 | 1.0198 |
| family = team-selected | 6,667 | 2,090 | 31.35% | 2139.6674 | 1.0238 | 0.4330 | 0.1397 | 0.3037 | 0.9789 |
| gender = female | 10,651 | 3,219 | 30.22% | 3250.6594 | 1.0098 | 0.4216 | 0.1348 | 0.3119 | 0.9859 |
| gender = male | 18,698 | 5,589 | 29.89% | 5558.2709 | 0.9945 | 0.4160 | 0.1334 | 0.3180 | 1.0023 |
| body_part = right_foot | 15,585 | 4,511 | 28.94% | 4492.5022 | 0.9959 | 0.4077 | 0.1303 | 0.3223 | 1.0009 |
| body_part = left_foot | 9,314 | 2,707 | 29.06% | 2678.1526 | 0.9893 | 0.4100 | 0.1310 | 0.3197 | 1.0303 |
| body_part = head | 4,321 | 1,512 | 34.99% | 1556.6293 | 1.0295 | 0.4646 | 0.1502 | 0.2823 | 0.9509 |
| body_part = other | 129 | 78 | 60.47% | 81.6462 | 1.0467 | 0.6804 | 0.2253 | -0.0140 | 0.4366 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0206 | 0.0261 | 0.0158 | 0.0164 | 0.0154 |

### Calibration by placement band

Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, after rounding); a ball beyond 3.55 m touches the post. The reliability table bins by prediction, so it cannot show a placement the model misprices. SD is √Σ p(1 − p), the spread of the goal count if xGOT is calibrated; *Within 2 SD* is the PM's screening rule (MS6), not a proof of calibration: a calibrated model leaves a band outside it by chance in about 4.6% of cases. The rule is a normal approximation, so a band with fewer than 100 shots is printed but not judged (PM, MS6).

| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals | SD | Within 2 SD |
|---|---|---|---|---|---|---|---|
| [0, 1) | 8,079 | 1,212 | 0.1500 | 0.1521 | +16.8 | 28.4 | yes |
| [1, 2) | 8,186 | 1,692 | 0.2067 | 0.2029 | -31.2 | 31.0 | yes |
| [2, 3) | 8,969 | 3,429 | 0.3823 | 0.3865 | +37.1 | 37.9 | yes |
| [3, 3.3) | 3,072 | 1,915 | 0.6234 | 0.5945 | -88.6 | 23.0 | no |
| [3.3, 3.55) | 866 | 505 | 0.5831 | 0.6545 | +61.8 | 12.1 | no |
| >= 3.55 | 147 | 52 | 0.3537 | 0.3860 | +4.7 | 4.9 | yes |
| outside the frame + 0.15 m | 30 | 3 | 0.1000 | 0.1098 | +0.3 | 1.5 | not judged (n < 100) |

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
| Base rate (on target) | 0.6110 | 0.2101 | 0.5000 | n/a | 0.6035 | 0.2065 | 0.5000 | n/a |
| LR xG only | 0.5164 | 0.1712 | 0.7618 | 1.0014 | 0.5185 | 0.1720 | 0.7512 | 0.9630 |
| LR xG + placement, no near / far | 0.4286 | 0.1378 | 0.8547 | 1.0003 | 0.4167 | 0.1341 | 0.8604 | 1.0237 |
| LR xG + placement | 0.4181 | 0.1339 | 0.8620 | 1.0000 | 0.4092 | 0.1312 | 0.8652 | 0.9984 |
| Monotone GBM + Platt | 0.4085 | 0.1307 | 0.8674 | 1.0098 | 0.3997 | 0.1284 | 0.8708 | 1.0064 |

## Limitations & known biases
- 34 of 32,483 on-target shots (hold-out included) are recorded beyond the frame + 0.15 m and are clipped to its edge and flagged by `mouth_out`, since where they would have crossed the line is not known; 1,381 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 32,483 of 32,483 on-target shots, so there is no provider xGOT benchmark.
- StatsBomb records a save where the keeper stopped the ball: its `goal_z` is that height, not the height at the goal line, and a save stopped short of the line has its `goal_y` projected along the line from the shot location and snapped to the 0.1 yd goal-mouth lattice of recorded points (797 of the on-target saves were deflected, so their projection starts at the shot, not at the deflection).
- On the hold-out, Σ xGOT is 904.8 against 913 goals (-0.90%; calibration intercept 0.0202). A level gap moves every keeper's goals saved against this xGOT the same way, so a season's keepers read better or worse as a group.
- Calibration by placement band: 2 of 7 bands are outside 2 SD on dev out of fold ([3, 3.3): 3,072 shots, 1,915 goals, Σ xGOT − goals −88.6 (SD 23.0); [3.3, 3.55): 866 shots, 505 goals, Σ xGOT − goals +61.8 (SD 12.1)). This model's placement terms misprice them; the headline is Monotone GBM + Platt, which the band rule judges (PM E1).

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
- v3 (MS6): saves recorded where the keeper stopped the ball and projected to the line are snapped to the 0.1 yd goal-mouth lattice of recorded points (football-core); the GBM's placement features carry no monotone constraint (chosen by the band rule); the band table prints the SD; the band rule is a check.
- v4 (MS6, GSAx): matches recorded on the coarse (1 yd) grid leave the xGOT fit (PM E1); a band is judged by the band rule only with at least 100 shots.
