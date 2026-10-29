# Model card: Monotone GBM + Platt

## Overview
- **Name / version:** `ws-xgot-gbm` v4
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.4: gradient boosting on the xG input (monotone), placement, side and angle, tuned and Platt-calibrated inside each fit by grouped CV. This is the headline model: its CV log-loss gain over LR xG + placement is +0.0086 [+0.0067, +0.0105], a 95% interval that lies above 0; on the test it is +0.0104 [+0.0064, +0.0143], an interval that lies above 0.

## Intended use
xGOT (post-shot xG) for on-target shots: P(goal) of a shot that reached the goal frame, from where it crossed the line and its pre-shot xG. Keeper-neutral (no keeper position), so goals saved against it credit the keeper's positioning too. For research and as the input of the MS6 goalkeeper models; not a rating of finishers, and not for other providers' data without re-fitting.

## Data
- **Source(s):** private WhoScored / Opta event archive, revision `84571265636c3c30fcfcf9b9b0b3bd1381f7488fc3d3870893d2539f8d7c3b23`; snapshot `whoscored-v1`, data digest `c394a94944cdc12ee4d6bdd77594c9156a51aa3731955905b46ec80f648bb63e`
- **Competitions & seasons:** 20 competition-seasons; hold-out 2026/2027
- **Rows:** train and validation (dev) 26,317 shots in 3,214 matches / test (hold-out) 5,154 shots in 587 matches
- **Filters:** excluded shots, by first matching reason. C3 on-target shots by first matching reason; penalty and shoot-out rows are scored by the penalty xGOT, modelled rows by the LR and GBM.

| Filter | Shots | Goals |
|---|---|---|
| coarse recording grid (whole-yard locations; the match is left out of the xGOT fit, PM MS6 E1) | 0 | 0 |
| own goal (Opta q28) | 0 | 0 |
| shoot-out kick (period 5; penalty xGOT, reported apart) | 16 | 13 |
| penalty (in-game; penalty xGOT) | 993 | 834 |
| direct corner (Opta q263) | 9 | 7 |
| no freeze frame | 0 | 0 |
| unresolved assisting event (left out of the xG population) | 1 | 0 |
| modelled (open-play xGOT) | 31,471 | 9,537 |

## Target
`is_goal` of an on-target shot: the ball went in rather than being saved. Blocked shots (clearances off the line included), woodwork and misses are not on target (C3 `on_target`), so they are not in the population.

## Features
xG input: Monotone GBM + Platt, cross-fitted; provider headline xG: Monotone GBM + Platt.

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
| max_leaf_nodes | 8 |
| iterations | 300 |
| Platt intercept | -0.0078 |
| Platt slope | 0.9878 |

Calibration is close to identity (Platt slope 0.988, intercept -0.008): the boosted model is already calibrated and Platt is a safeguard.

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. Train is the 2025/2026, 2026 seasons; **Test** is the 2026/2027 season, scored once by that fit and never used to choose. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4231 | 0.4383 ± 0.0090 | 0.4394 |
| Brier score | 0.1361 | 0.1415 ± 0.0032 | 0.1416 |
| ROC AUC | 0.8553 | 0.8435 ± 0.0057 | 0.8475 |
| Log-loss skill | 0.3084 | 0.2835 | 0.2935 |
| Brier skill | 0.3532 | 0.3277 | 0.3419 |
| Calibration intercept | 0.0006 | -0.0012 | 0.1061 |
| Calibration slope | 1.0739 | 1.0020 | 1.0314 |
| Σ xGOT | 7920.9194 | 7926.9584 | 1536.2917 |
| Goals | 7,923 | 7,923 | 1,614 |
| abs(Σ xGOT − goals) / goals | 0.03% | 0.05% | 4.81% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 1.04% (max 2.79%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR xG + placement | +0.0086 [+0.0067, +0.0105] | +0.0104 [+0.0064, +0.0143] |

### Milestone checks (MS5, MS6)

Bounds from the PM brief (MS5) and the band rule from the PM brief (MS6); AUC >= 0.80 reads "about 0.80 or better" strictly. ECE is judged on the CV fold mean and the point value of the scored-once split (PM X3).

| Target | Measured | Result |
|---|---|---|
| ROC AUC >= 0.80, CV fold mean | 0.8435 | met |
| ROC AUC >= 0.80, test | 0.8475 | met |
| ECE < 0.02, CV fold mean | 0.0121 (max 0.0170, pooled 0.0055) | met |
| ECE < 0.02, test | 0.0194 (noise floor 0.0129) | met |
| Calibration slope in [0.9, 1.1], CV fold mean | 1.0020 | met |
| Calibration slope in [0.9, 1.1], test | 1.0314 | met |
| GBM headline iff its CV gain over LR xG + placement has a 95% interval above 0 | CV +0.0086 [+0.0067, +0.0105]; headline Monotone GBM + Platt | met |
| On-target shots scored reconcile with C3 (less PM-ruled exclusions) within 1% (two-sided) | 32,480 of 32,490 (-0.031%); unscored 0 | met |
| On-target goals scored reconcile with C3 (less PM-ruled exclusions) within 1% (two-sided) | 10,384 of 10,391 (-0.067%) | met |
| xG input Monotone GBM + Platt: cross-fitted on the same folds, keeper-neutral, not an ablation (the run raises otherwise) | guards ran | met |
| No keeper-position feature in the headline xGOT (the xGOT guard raises otherwise) | xg_logit, mouth_y_abs, mouth_out, mouth_z, far_side, angle_rad | met |
| Σ xGOT − goals within 2 SD (√Σp(1−p)) in every placement band of at least 100 shots, dev out of fold | [3, 3.3): −20.6 (SD 21.4); 5 of 7 bands judged | met |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 2,632 | 2.66% | 2.36% |
| 1 | 2,632 | 4.80% | 4.75% |
| 2 | 2,632 | 7.55% | 8.02% |
| 3 | 2,631 | 11.73% | 10.83% |
| 4 | 2,632 | 17.62% | 16.87% |
| 5 | 2,632 | 25.37% | 26.56% |
| 6 | 2,631 | 35.36% | 36.37% |
| 7 | 2,632 | 48.21% | 47.91% |
| 8 | 2,632 | 64.37% | 64.02% |
| 9 | 2,631 | 83.55% | 83.39% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| gender = male | 26,317 | 7,923 | 30.11% | 7926.9584 | 1.0005 | 0.4383 | 0.1415 | 0.2835 | 1.0006 |
| body_part = right_foot | 13,163 | 4,000 | 30.39% | 3930.3290 | 0.9826 | 0.4421 | 0.1426 | 0.2800 | 1.0054 |
| body_part = left_foot | 8,647 | 2,430 | 28.10% | 2457.6354 | 1.0114 | 0.4392 | 0.1417 | 0.2605 | 0.9610 |
| body_part = head | 4,416 | 1,446 | 32.74% | 1487.4985 | 1.0287 | 0.4233 | 0.1367 | 0.3306 | 1.0793 |
| body_part = other | 91 | 47 | 51.65% | 51.4954 | 1.0956 | 0.5318 | 0.1765 | 0.2322 | 0.9575 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the test value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Test | Test floor |
|---|---|---|---|---|
| 0.0121 | 0.0170 | 0.0055 | 0.0194 | 0.0129 |

### Calibration by placement band

Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, after rounding); a ball beyond 3.55 m touches the post. The reliability table bins by prediction, so it cannot show a placement the model misprices. SD is √Σ p(1 − p), the spread of the goal count if xGOT is calibrated; *Within 2 SD* is the PM's screening rule (MS6), not a proof of calibration: a calibrated model leaves a band outside it by chance in about 4.6% of cases. The rule is a normal approximation, so a band with fewer than 100 shots is printed but not judged (PM, MS6).

| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals | SD | Within 2 SD |
|---|---|---|---|---|---|---|---|
| [0, 1) | 7,552 | 1,277 | 0.1691 | 0.1713 | +16.5 | 28.2 | yes |
| [1, 2) | 7,814 | 1,936 | 0.2478 | 0.2490 | +9.9 | 32.2 | yes |
| [2, 3) | 7,591 | 3,044 | 0.4010 | 0.3994 | -12.1 | 35.9 | yes |
| [3, 3.3) | 2,544 | 1,332 | 0.5236 | 0.5155 | -20.6 | 21.4 | yes |
| [3.3, 3.55) | 787 | 329 | 0.4180 | 0.4268 | +6.9 | 11.7 | yes |
| >= 3.55 | 29 | 5 | 0.1724 | 0.2889 | +3.4 | 2.1 | not judged (n < 100) |
| outside the frame + 0.15 m | 0 | 0 | n/a | n/a | n/a | 0.0 | not judged (n < 100) |

### Calibration by placement band, test

The same table on the 5,154 on-target shots of the test (the fit on all of dev), with the same rule and the same 100-shot minimum; printed beside check row 12, which judges dev out of fold.

| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals | SD | Within 2 SD |
|---|---|---|---|---|---|---|---|
| [0, 1) | 1,429 | 223 | 0.1561 | 0.1625 | +9.2 | 12.0 | yes |
| [1, 2) | 1,549 | 410 | 0.2647 | 0.2474 | -26.9 | 14.3 | yes |
| [2, 3) | 1,500 | 621 | 0.4140 | 0.3972 | -25.2 | 16.0 | yes |
| [3, 3.3) | 517 | 285 | 0.5513 | 0.5105 | -21.1 | 9.6 | no |
| [3.3, 3.55) | 156 | 75 | 0.4808 | 0.3866 | -14.7 | 5.1 | no |
| >= 3.55 | 3 | 0 | 0.0000 | 0.2920 | +0.9 | 0.7 | not judged (n < 100) |
| outside the frame + 0.15 m | 0 | 0 | n/a | n/a | n/a | 0.0 | not judged (n < 100) |

### Reconciliation with C3

Every C3 on-target shot is in one row of the Filters table; the scored rows are the open-play and penalty xGOT predictions, compared with C3 less the rows a PM ruling leaves out.

| | Shots | Goals |
|---|---|---|
| C3 on target | 32,490 | 10,391 |
| Scored by xGOT | 32,480 | 10,384 |
| Difference | -0.031% | -0.067% |

Population rows without a prediction: 0.

### Shooting goals added (descriptive)

Shooting goals added (SGA) is xGOT − xG per shot, next to goals − xG; both subtract the same xG, the xG input above. Blocked and off-target shots count as 0 xGOT. Each player's matches are ordered by date (then match id) and alternated into two halves, so a match never spans halves; r is the Pearson correlation across players of the per-shot value in the two halves, with a 95% interval by Fisher z (n/a under 4 players). A descriptive measure of shots beating their xG on target, not a finishing-skill claim.

| Min shots | Blocked shots | Players | SGA per shot: r [95% CI] | Goals − xG per shot: r [95% CI] |
|---|---|---|---|---|
| 20 | in | 1,731 | +0.014 [-0.033, +0.062] | +0.026 [-0.021, +0.073] |
| 20 | left out | 1,186 | +0.075 [+0.019, +0.132] | +0.015 [-0.042, +0.072] |
| 40 | in | 711 | -0.038 [-0.111, +0.036] | +0.040 [-0.034, +0.113] |
| 40 | left out | 380 | +0.001 [-0.099, +0.102] | +0.028 [-0.073, +0.128] |
| 80 | in | 111 | -0.124 [-0.303, +0.064] | +0.101 [-0.087, +0.282] |
| 80 | left out | 25 | +0.147 [-0.264, +0.512] | +0.338 [-0.066, +0.647] |

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Base rate (on target) | 0.6118 | 0.2104 | 0.5000 | n/a | 0.6219 | 0.2152 | 0.5000 | n/a |
| LR xG only | 0.5293 | 0.1762 | 0.7464 | 1.0004 | 0.5369 | 0.1794 | 0.7484 | 1.0076 |
| LR xG + placement, no near / far | 0.4592 | 0.1493 | 0.8270 | 0.9988 | 0.4627 | 0.1509 | 0.8296 | 1.0161 |
| LR xG + placement | 0.4469 | 0.1442 | 0.8368 | 0.9987 | 0.4498 | 0.1454 | 0.8388 | 1.0164 |
| Monotone GBM + Platt | 0.4383 | 0.1415 | 0.8435 | 1.0020 | 0.4394 | 0.1416 | 0.8475 | 1.0314 |

## Limitations & known biases
- 0 of 31,471 on-target shots (test included) are recorded beyond the frame + 0.15 m and are clipped to its edge and flagged by `mouth_out`, since where they would have crossed the line is not known; 1,471 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 31,471 of 31,471 on-target shots, so there is no provider xGOT benchmark.
- On the test, Σ xGOT is 1,536.3 against 1,614 goals (-4.81%; calibration intercept 0.1061). A level gap moves every keeper's goals saved against this xGOT the same way, so a season's keepers read better or worse as a group.

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
- v3 (MS6): saves recorded where the keeper stopped the ball and projected to the line are snapped to the 0.1 yd goal-mouth lattice of recorded points (football-core); the GBM's placement features carry no monotone constraint (chosen by the band rule); the band table prints the SD; the band rule is a check.
- v4 (MS6, GSAx): matches recorded on the coarse (1 yd) grid leave the xGOT fit (PM E1); a band is judged by the band rule only with at least 100 shots.
