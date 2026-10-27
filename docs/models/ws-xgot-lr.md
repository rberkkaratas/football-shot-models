# Model card: LR xG + placement

## Overview
- **Name / version:** `ws-xgot-lr` v1
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.3: logistic regression of the goal on the logit of the cross-fitted xG, placement splines, angle × placement and the near / far post. Not the headline model; that is Monotone GBM + Platt.

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
| `mouth_y_abs_rcs1` | Restricted cubic spline term 1 of `mouth_y_abs`, knots 0.15, 0.9, 1.7, 2.5, 3.25 m | `mouth_y_abs` |
| `mouth_y_abs_rcs2` | Restricted cubic spline term 2 of `mouth_y_abs`, knots 0.15, 0.9, 1.7, 2.5, 3.25 m | `mouth_y_abs` |
| `mouth_y_abs_rcs3` | Restricted cubic spline term 3 of `mouth_y_abs`, knots 0.15, 0.9, 1.7, 2.5, 3.25 m | `mouth_y_abs` |
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
| intercept | -1.1486 | -0.1525 |
| xg_logit | 1.0848 | 0.9650 |
| mouth_y_abs | 0.4011 | 0.4115 |
| mouth_y_abs_rcs1 | -1.0674 | -1.1420 |
| mouth_y_abs_rcs2 | 2.6173 | 6.8101 |
| mouth_y_abs_rcs3 | -1.3201 | -14.5010 |
| mouth_z | -0.4408 | -0.7396 |
| mouth_z_rcs1 | 2.2011 | 4.6318 |
| mouth_z_rcs2 | -1.4494 | -4.4911 |
| mouth_z_rcs3 | -0.0379 | -0.4512 |
| angle_x_mouth_y | 0.4504 | 0.5878 |
| angle_x_mouth_z | -0.0350 | -0.0728 |
| mouth_far | 0.3849 | 0.2063 |

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. Train is the 2025/2026, 2026 seasons; **Test** is the 2026/2027 season, scored once by that fit and never used to choose. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4465 | 0.4469 ± 0.0083 | 0.4500 |
| Brier score | 0.1441 | 0.1443 ± 0.0027 | 0.1455 |
| ROC AUC | 0.8370 | 0.8367 ± 0.0056 | 0.8386 |
| Log-loss skill | 0.2702 | 0.2695 | 0.2764 |
| Brier skill | 0.3152 | 0.3145 | 0.3242 |
| Calibration intercept | 0.0000 | -0.0003 | 0.0885 |
| Calibration slope | 1.0000 | 0.9992 | 1.0155 |
| Σ xGOT | 7923.0000 | 7923.6506 | 1547.9349 |
| Goals | 7,923 | 7,923 | 1,614 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.01% | 4.09% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 1.05% (max 2.57%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| `mouth_far` (near / far post): dropped | +0.0123 [+0.0106, +0.0144] | +0.0129 [+0.0083, +0.0175] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 2,632 | 3.05% | 2.89% |
| 1 | 2,632 | 5.84% | 4.79% |
| 2 | 2,632 | 9.02% | 8.51% |
| 3 | 2,631 | 12.94% | 11.74% |
| 4 | 2,632 | 18.18% | 18.43% |
| 5 | 2,632 | 25.23% | 25.76% |
| 6 | 2,631 | 34.21% | 35.27% |
| 7 | 2,632 | 45.99% | 47.68% |
| 8 | 2,632 | 61.78% | 63.72% |
| 9 | 2,631 | 84.85% | 82.29% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| gender = male | 26,317 | 7,923 | 30.11% | 7923.6506 | 1.0001 | 0.4469 | 0.1443 | 0.2694 | 0.9976 |
| body_part = right_foot | 13,163 | 4,000 | 30.39% | 3942.9114 | 0.9857 | 0.4510 | 0.1455 | 0.2657 | 0.9946 |
| body_part = left_foot | 8,647 | 2,430 | 28.10% | 2444.5147 | 1.0060 | 0.4469 | 0.1438 | 0.2476 | 0.9570 |
| body_part = head | 4,416 | 1,446 | 32.74% | 1486.3611 | 1.0279 | 0.4339 | 0.1408 | 0.3138 | 1.0965 |
| body_part = other | 91 | 47 | 51.65% | 49.8634 | 1.0609 | 0.4989 | 0.1661 | 0.2797 | 0.9858 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the test value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Test | Test floor |
|---|---|---|---|---|
| 0.0144 | 0.0186 | 0.0110 | 0.0149 | 0.0129 |

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
