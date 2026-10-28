# Model card: Monotone GBM + Platt

## Overview
- **Name / version:** `xgot-gbm` v4
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.4: gradient boosting on the xG input (monotone), placement, side and angle, tuned and Platt-calibrated inside each fit by grouped CV. This is the headline model: its CV log-loss gain over LR xG + placement is +0.0096 [+0.0077, +0.0115], a 95% interval that lies above 0; on the hold-out it is +0.0095 [+0.0036, +0.0156], an interval that lies above 0.

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
| `mouth_out` | 1 when the point was recorded beyond the frame + 0.15 m (before clipping), else 0 | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_z` | Height above the ground, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `far_side` | +1 far post, −1 near post (the ball's side against the shot's), 0 for a shot within 0.5 m of the centre line or a ball at the centre | `y`, `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `angle_rad` | Opening angle between the posts seen from the shot location, radians | `angle_rad` (`pitch.opening_angle`) |

Monotone constraints of the fitted model (every other column is free):

- xg_logit: non-decreasing — a higher xG at the same placement never lowers xGOT

The constraint holds along the xG input with every other column held fixed. Placement carries no monotone constraint, so the trees shape the goal rate across the goal mouth and towards the posts from the data; the calibration-by-placement-band table below gives the goal rate per band.

## Model
`HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1, min_samples_leaf=100, early_stopping=False, random_state=0)`, `monotonic_cst` placed by column name, no class weights. scikit-learn 1.9.1.

Tuning inside each fit: 4-fold `GroupKFold` by match on its training rows, `max_leaf_nodes` in (8, 16, 32) and 50, 100, ..., 600 iterations (staged scores of one fit per leaf count); the lowest inner out-of-fold log loss wins, ties to fewer leaves, then fewer iterations. Platt scaling: an unpenalised logistic regression of the goal on the chosen configuration's inner out-of-fold log-odds, applied to the refit on all training rows; a slope <= 0 is refused.

Tuning and calibration see only the rows of the fit they belong to; each outer test fold is scored by a model that never saw it.

Settings of the fit on all of dev:

| Setting | Value |
|---|---|
| max_leaf_nodes | 16 |
| iterations | 150 |
| Platt intercept | -0.0038 |
| Platt slope | 0.9927 |

Calibration is close to identity (Platt slope 0.993, intercept -0.004): the boosted model is already calibrated and Platt is a safeguard.

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. **Test** is the hold-out competition-season named under Data, scored once by that fit. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.3941 | 0.4085 ± 0.0076 | 0.3997 |
| Brier score | 0.1257 | 0.1307 ± 0.0035 | 0.1284 |
| ROC AUC | 0.8773 | 0.8674 ± 0.0049 | 0.8708 |
| Log-loss skill | 0.3550 | 0.3314 | 0.3378 |
| Brier skill | 0.4017 | 0.3778 | 0.3781 |
| Calibration intercept | -0.0001 | -0.0006 | 0.0375 |
| Calibration slope | 1.0676 | 1.0098 | 1.0064 |
| Σ xGOT | 8808.2182 | 8811.2301 | 898.0649 |
| Goals | 8,808 | 8,808 | 913 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.04% | 1.64% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.42% (max 5.13%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR xG + placement | +0.0096 [+0.0077, +0.0115] | +0.0095 [+0.0036, +0.0156] |

### Milestone checks (MS5, MS6)

Bounds from the PM brief (MS5) and the band rule from the PM brief (MS6); AUC >= 0.80 reads "about 0.80 or better" strictly. ECE is judged on the CV fold mean and the point value of the scored-once split (PM X3).

| Target | Measured | Result |
|---|---|---|
| ROC AUC >= 0.80, CV fold mean | 0.8674 | met |
| ROC AUC >= 0.80, hold-out | 0.8708 | met |
| ECE < 0.02, CV fold mean | 0.0166 (max 0.0229, pooled 0.0057) | met |
| ECE < 0.02, hold-out | 0.0148 (noise floor 0.0152) | met |
| Calibration slope in [0.9, 1.1], CV fold mean | 1.0098 | met |
| Calibration slope in [0.9, 1.1], hold-out | 1.0064 | met |
| GBM headline iff its CV gain over LR xG + placement has a 95% interval above 0 | CV +0.0096 [+0.0077, +0.0115]; headline Monotone GBM + Platt | met |
| On-target shots scored reconcile with C3 (less PM-ruled exclusions) within 1% (two-sided) | 33,749 of 35,632 less 1,855 coarse-grid (PM E1) (-0.083%); unscored 0 | met |
| On-target goals scored reconcile with C3 (less PM-ruled exclusions) within 1% (two-sided) | 10,730 of 11,328 less 580 coarse-grid (-0.167%) | met |
| xG input Monotone GBM + Platt without keeper position: cross-fitted on the same folds, keeper-neutral, not an ablation (the run raises otherwise) | guards ran | met |
| No keeper-position feature in the headline xGOT (the xGOT guard raises otherwise) | xg_logit, mouth_y_abs, mouth_out, mouth_z, far_side, angle_rad | met |
| Σ xGOT − goals within 2 SD (√Σp(1−p)) in every placement band of at least 100 shots, dev out of fold | [3, 3.3): −34.5 (SD 23.6); 6 of 7 bands judged | met |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 2,935 | 1.85% | 1.57% |
| 1 | 2,935 | 3.41% | 3.41% |
| 2 | 2,935 | 5.80% | 5.11% |
| 3 | 2,935 | 9.65% | 10.77% |
| 4 | 2,935 | 15.44% | 14.34% |
| 5 | 2,935 | 23.80% | 23.75% |
| 6 | 2,935 | 35.13% | 36.56% |
| 7 | 2,935 | 50.73% | 50.49% |
| 8 | 2,935 | 68.77% | 68.25% |
| 9 | 2,934 | 85.66% | 85.89% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 12,031 | 3,499 | 29.08% | 3408.8586 | 0.9742 | 0.3947 | 0.1259 | 0.3454 | 1.0300 |
| family = team-selected | 6,667 | 2,090 | 31.35% | 2114.0052 | 1.0115 | 0.4265 | 0.1372 | 0.3142 | 0.9828 |
| gender = female | 10,651 | 3,219 | 30.22% | 3288.3664 | 1.0215 | 0.4128 | 0.1320 | 0.3263 | 1.0003 |
| gender = male | 18,698 | 5,589 | 29.89% | 5522.8637 | 0.9882 | 0.4060 | 0.1299 | 0.3344 | 1.0113 |
| body_part = right_foot | 15,585 | 4,511 | 28.94% | 4463.4704 | 0.9895 | 0.4006 | 0.1277 | 0.3342 | 0.9987 |
| body_part = left_foot | 9,314 | 2,707 | 29.06% | 2663.7336 | 0.9840 | 0.4001 | 0.1278 | 0.3362 | 1.0232 |
| body_part = head | 4,321 | 1,512 | 34.99% | 1600.0512 | 1.0582 | 0.4492 | 0.1454 | 0.3062 | 1.0449 |
| body_part = other | 129 | 78 | 60.47% | 83.9749 | 1.0766 | 0.6080 | 0.2086 | 0.0940 | 0.6970 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0166 | 0.0229 | 0.0057 | 0.0148 | 0.0152 |

### Calibration by placement band

Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, after rounding); a ball beyond 3.55 m touches the post. The reliability table bins by prediction, so it cannot show a placement the model misprices. SD is √Σ p(1 − p), the spread of the goal count if xGOT is calibrated; *Within 2 SD* is the PM's screening rule (MS6), not a proof of calibration: a calibrated model leaves a band outside it by chance in about 4.6% of cases. The rule is a normal approximation, so a band with fewer than 100 shots is printed but not judged (PM, MS6).

| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals | SD | Within 2 SD |
|---|---|---|---|---|---|---|---|
| [0, 1) | 8,079 | 1,212 | 0.1500 | 0.1519 | +15.2 | 27.1 | yes |
| [1, 2) | 8,186 | 1,692 | 0.2067 | 0.2086 | +15.2 | 30.5 | yes |
| [2, 3) | 8,969 | 3,429 | 0.3823 | 0.3823 | -0.5 | 37.9 | yes |
| [3, 3.3) | 3,072 | 1,915 | 0.6234 | 0.6121 | -34.5 | 23.6 | yes |
| [3.3, 3.55) | 866 | 505 | 0.5831 | 0.5855 | +2.0 | 12.6 | yes |
| >= 3.55 | 147 | 52 | 0.3537 | 0.3435 | -1.5 | 5.2 | yes |
| outside the frame + 0.15 m | 30 | 3 | 0.1000 | 0.3449 | +7.3 | 2.4 | not judged (n < 100) |

### Reconciliation with C3

Every C3 on-target shot is in one row of the Filters table; the scored rows are the open-play and penalty xGOT predictions, compared with C3 less the rows a PM ruling leaves out.

| | Shots | Goals |
|---|---|---|
| C3 on target | 35,632 | 11,328 |
| Left out by a PM ruling (coarse-grid matches) | 1,855 | 580 |
| Scored by xGOT | 33,749 | 10,730 |
| Difference | -0.083% | -0.167% |

Population rows without a prediction: 0.

### Shooting goals added (descriptive)

Shooting goals added (SGA) is xGOT − xG per shot, next to goals − xG; both subtract the same xG, the xG input above. Blocked and off-target shots count as 0 xGOT. Each player's matches are ordered by date (then match id) and alternated into two halves, so a match never spans halves; r is the Pearson correlation across players of the per-shot value in the two halves, with a 95% interval by Fisher z (n/a under 4 players). A descriptive measure of shots beating their xG on target, not a finishing-skill claim.

| Min shots | Blocked shots | Players | SGA per shot: r [95% CI] | Goals − xG per shot: r [95% CI] |
|---|---|---|---|---|
| 20 | in | 1,368 | +0.034 [-0.019, +0.087] | -0.024 [-0.077, +0.029] |
| 20 | left out | 1,001 | +0.041 [-0.021, +0.103] | -0.029 [-0.091, +0.033] |
| 40 | in | 597 | +0.096 [+0.016, +0.175] | +0.053 [-0.027, +0.133] |
| 40 | left out | 362 | +0.070 [-0.034, +0.171] | +0.008 [-0.096, +0.111] |
| 80 | in | 153 | -0.001 [-0.159, +0.158] | +0.043 [-0.116, +0.201] |
| 80 | left out | 71 | +0.016 [-0.218, +0.248] | -0.026 [-0.258, +0.208] |

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
- On the hold-out, Σ xGOT is 898.1 against 913 goals (-1.64%; calibration intercept 0.0375). A level gap moves every keeper's goals saved against this xGOT the same way, so a season's keepers read better or worse as a group.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
- v3 (MS6): saves recorded where the keeper stopped the ball and projected to the line are snapped to the 0.1 yd goal-mouth lattice of recorded points (football-core); the GBM's placement features carry no monotone constraint (chosen by the band rule); the band table prints the SD; the band rule is a check.
- v4 (MS6, GSAx): matches recorded on the coarse (1 yd) grid leave the xGOT fit (PM E1); a band is judged by the band rule only with at least 100 shots.
