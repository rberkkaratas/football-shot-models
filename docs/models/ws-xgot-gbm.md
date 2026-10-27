# Model card: Monotone GBM + Platt

## Overview
- **Name / version:** `ws-xgot-gbm` v1
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.4: monotone gradient boosting on the xG input, placement, side and angle, tuned and Platt-calibrated inside each fit by grouped CV. This is the headline model: its CV log-loss gain over LR xG + placement is +0.0075 [+0.0057, +0.0094], a 95% interval that lies above 0; on the test it is +0.0102 [+0.0064, +0.0140], an interval that lies above 0.

## Intended use
xGOT (post-shot xG) for on-target shots: P(goal) of a shot that reached the goal frame, from where it crossed the line and its pre-shot xG. Keeper-neutral (no keeper position), so goals saved against it credit the keeper's positioning too. For research and as the input of the MS6 goalkeeper models; not a rating of finishers, and not for other providers' data without re-fitting.

## Data
- **Source(s):** private WhoScored / Opta event archive, revision `84571265636c3c30fcfcf9b9b0b3bd1381f7488fc3d3870893d2539f8d7c3b23`; snapshot `whoscored-v1`, data digest `8033836b08b03fcbaffe2a19cc3a9549f2adb2eebf10f2871fff57f267b79ebb`
- **Competitions & seasons:** 20 competition-seasons; hold-out 2026/2027
- **Rows:** train and validation (dev) 26,317 shots in 3,214 matches / test (hold-out) 5,154 shots in 587 matches
- **Filters:** excluded shots, by first matching reason. C3 on-target shots by first matching reason; penalty and shoot-out rows are scored by the penalty xGOT, modelled rows by the LR and GBM.

| Filter | Shots | Goals |
|---|---|---|
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
| `mouth_z` | Height above the ground, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `far_side` | +1 far post, −1 near post (the ball's side against the shot's), 0 for a shot within 0.5 m of the centre line or a ball at the centre | `y`, `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `angle_rad` | Opening angle between the posts seen from the shot location, radians | `angle_rad` (`pitch.opening_angle`) |

Monotone constraints of the fitted model (every other column is free):

- xg_logit: non-decreasing — a higher xG at the same placement never lowers xGOT
- mouth_y_abs: non-decreasing — further from the centre at the same height and on the same side (`far_side` fixed) never helps the keeper

Each constraint holds along one column with every other column held fixed: `mouth_y_abs` moves the ball towards a post at the same height, on the same side (near or far).

## Model
`HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1, min_samples_leaf=100, early_stopping=False, random_state=0)`, `monotonic_cst` placed by column name, no class weights. scikit-learn 1.9.1.

Tuning inside each fit: 4-fold `GroupKFold` by match on its training rows, `max_leaf_nodes` in (8, 16, 32) and 50, 100, ..., 600 iterations (staged scores of one fit per leaf count); the lowest inner out-of-fold log loss wins, ties to fewer leaves, then fewer iterations. Platt scaling: an unpenalised logistic regression of the goal on the chosen configuration's inner out-of-fold log-odds, applied to the refit on all training rows; a slope <= 0 is refused.

Tuning and calibration see only the rows of the fit they belong to; each outer test fold is scored by a model that never saw it.

Settings of the fit on all of dev:

| Setting | Value |
|---|---|
| max_leaf_nodes | 32 |
| iterations | 100 |
| Platt intercept | +0.0080 |
| Platt slope | 1.0118 |

Calibration is close to identity (Platt slope 1.012, intercept +0.008): the boosted model is already calibrated and Platt is a safeguard.

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. Train is the 2025/2026, 2026 seasons; **Test** is the 2026/2027 season, scored once by that fit and never used to choose. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4222 | 0.4394 ± 0.0097 | 0.4398 |
| Brier score | 0.1356 | 0.1418 ± 0.0035 | 0.1419 |
| ROC AUC | 0.8561 | 0.8425 ± 0.0060 | 0.8472 |
| Log-loss skill | 0.3098 | 0.2818 | 0.2928 |
| Brier skill | 0.3555 | 0.3261 | 0.3405 |
| Calibration intercept | 0.0017 | -0.0003 | 0.1081 |
| Calibration slope | 1.0853 | 1.0013 | 1.0333 |
| Σ xGOT | 7916.5145 | 7923.7372 | 1534.6385 |
| Goals | 7,923 | 7,923 | 1,614 |
| abs(Σ xGOT − goals) / goals | 0.08% | 0.01% | 4.92% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 1.00% (max 2.57%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR xG + placement | +0.0075 [+0.0057, +0.0094] | +0.0102 [+0.0064, +0.0140] |

### Milestone checks (MS5)

Bounds from the PM brief (MS5); AUC >= 0.80 reads "about 0.80 or better" strictly. ECE is judged on the CV fold mean and the point value of the scored-once split (PM X3).

| Target | Measured | Result |
|---|---|---|
| ROC AUC >= 0.80, CV fold mean | 0.8425 | met |
| ROC AUC >= 0.80, test | 0.8472 | met |
| ECE < 0.02, CV fold mean | 0.0121 (max 0.0176, pooled 0.0047) | met |
| ECE < 0.02, test | 0.0178 (noise floor 0.0127) | met |
| Calibration slope in [0.9, 1.1], CV fold mean | 1.0013 | met |
| Calibration slope in [0.9, 1.1], test | 1.0333 | met |
| GBM headline iff its CV gain over LR xG + placement has a 95% interval above 0 | CV +0.0075 [+0.0057, +0.0094]; headline Monotone GBM + Platt | met |
| On-target shots scored reconcile with C3 within 1% (two-sided) | 32,480 of 32,490 (-0.031%); unscored 0 | met |
| On-target goals scored reconcile with C3 within 1% (two-sided) | 10,384 of 10,391 (-0.067%) | met |
| xG input Monotone GBM + Platt: cross-fitted on the same folds, keeper-neutral, not an ablation (the run raises otherwise) | guards ran | met |
| No keeper-position feature in the headline xGOT (the xGOT guard raises otherwise) | xg_logit, mouth_y_abs, mouth_z, far_side, angle_rad | met |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 2,632 | 2.50% | 2.47% |
| 1 | 2,632 | 4.89% | 4.56% |
| 2 | 2,632 | 7.98% | 7.52% |
| 3 | 2,631 | 12.03% | 12.01% |
| 4 | 2,632 | 17.73% | 17.10% |
| 5 | 2,632 | 25.44% | 26.44% |
| 6 | 2,631 | 35.13% | 35.23% |
| 7 | 2,632 | 47.60% | 48.59% |
| 8 | 2,632 | 63.78% | 64.02% |
| 9 | 2,631 | 84.02% | 83.12% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| gender = male | 26,317 | 7,923 | 30.11% | 7923.7372 | 1.0001 | 0.4394 | 0.1418 | 0.2818 | 0.9999 |
| body_part = right_foot | 13,163 | 4,000 | 30.39% | 3926.1325 | 0.9815 | 0.4432 | 0.1429 | 0.2783 | 1.0021 |
| body_part = left_foot | 8,647 | 2,430 | 28.10% | 2455.3129 | 1.0104 | 0.4403 | 0.1422 | 0.2587 | 0.9585 |
| body_part = head | 4,416 | 1,446 | 32.74% | 1490.9252 | 1.0311 | 0.4242 | 0.1371 | 0.3292 | 1.0919 |
| body_part = other | 91 | 47 | 51.65% | 51.3666 | 1.0929 | 0.5366 | 0.1796 | 0.2252 | 0.9320 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the test value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Test | Test floor |
|---|---|---|---|---|
| 0.0121 | 0.0176 | 0.0047 | 0.0178 | 0.0127 |

### Reconciliation with C3

Every C3 on-target shot is in one row of the Filters table; the scored rows are the open-play and penalty xGOT predictions.

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
| 20 | in | 1,731 | +0.008 [-0.039, +0.055] | +0.026 [-0.021, +0.073] |
| 20 | left out | 1,186 | +0.077 [+0.020, +0.133] | +0.015 [-0.042, +0.072] |
| 40 | in | 711 | -0.041 [-0.114, +0.033] | +0.040 [-0.034, +0.113] |
| 40 | left out | 380 | +0.011 [-0.089, +0.112] | +0.028 [-0.073, +0.128] |
| 80 | in | 111 | -0.130 [-0.309, +0.058] | +0.101 [-0.087, +0.282] |
| 80 | left out | 25 | +0.202 [-0.210, +0.553] | +0.338 [-0.066, +0.647] |

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Base rate (on target) | 0.6118 | 0.2104 | 0.5000 | n/a | 0.6219 | 0.2152 | 0.5000 | n/a |
| LR xG only | 0.5293 | 0.1762 | 0.7464 | 1.0004 | 0.5369 | 0.1794 | 0.7484 | 1.0076 |
| LR xG + placement, no near / far | 0.4593 | 0.1494 | 0.8268 | 0.9995 | 0.4629 | 0.1510 | 0.8293 | 1.0156 |
| LR xG + placement | 0.4469 | 0.1443 | 0.8367 | 0.9992 | 0.4500 | 0.1455 | 0.8386 | 1.0155 |
| Monotone GBM + Platt | 0.4394 | 0.1418 | 0.8425 | 1.0013 | 0.4398 | 0.1419 | 0.8472 | 1.0333 |

## Limitations & known biases
- 0 of 31,471 on-target shots (test included) are recorded beyond the frame + 0.15 m and are clipped to its edge; 1,471 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 31,471 of 31,471 on-target shots, so there is no provider xGOT benchmark.

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
