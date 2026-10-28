# Model card: LR xG + placement

## Overview
- **Name / version:** `ws-xgot-lr` v4
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.3: logistic regression of the goal on the logit of the cross-fitted xG, placement splines, the width past the post's inner edge, angle × placement and the near / far post. Not the headline model; that is Monotone GBM + Platt.

## Intended use
xGOT (post-shot xG) for on-target shots: P(goal) of a shot that reached the goal frame, from where it crossed the line and its pre-shot xG. Keeper-neutral (no keeper position), so goals saved against it credit the keeper's positioning too. For research and as the input of the MS6 goalkeeper models; not a rating of finishers, and not for other providers' data without re-fitting.

## Data
- **Source(s):** private WhoScored / Opta event archive, revision `84571265636c3c30fcfcf9b9b0b3bd1381f7488fc3d3870893d2539f8d7c3b23`; snapshot `whoscored-v1`, data digest `8033836b08b03fcbaffe2a19cc3a9549f2adb2eebf10f2871fff57f267b79ebb`
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
| intercept | -1.1487 | -0.1503 |
| xg_logit | 1.0849 | 0.9651 |
| mouth_y_abs | 0.3930 | 0.4032 |
| mouth_y_abs_rcs1 | -0.9901 | -1.0593 |
| mouth_y_abs_rcs2 | 2.4997 | 6.5041 |
| mouth_y_abs_rcs3 | -1.2671 | -13.9189 |
| mouth_y_past | -0.0454 | -15.2837 |
| mouth_z | -0.4411 | -0.7400 |
| mouth_z_rcs1 | 2.2326 | 4.6981 |
| mouth_z_rcs2 | -1.4910 | -4.6200 |
| mouth_z_rcs3 | -0.0273 | -0.3249 |
| angle_x_mouth_y | 0.4501 | 0.5874 |
| angle_x_mouth_z | -0.0350 | -0.0728 |
| mouth_far | 0.3840 | 0.2058 |

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. Train is the 2025/2026, 2026 seasons; **Test** is the 2026/2027 season, scored once by that fit and never used to choose. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4464 | 0.4469 ± 0.0081 | 0.4498 |
| Brier score | 0.1441 | 0.1442 ± 0.0027 | 0.1454 |
| ROC AUC | 0.8372 | 0.8368 ± 0.0055 | 0.8388 |
| Log-loss skill | 0.2704 | 0.2695 | 0.2768 |
| Brier skill | 0.3154 | 0.3147 | 0.3246 |
| Calibration intercept | 0.0000 | -0.0002 | 0.0876 |
| Calibration slope | 1.0000 | 0.9987 | 1.0164 |
| Σ xGOT | 7923.0000 | 7923.5008 | 1548.5496 |
| Goals | 7,923 | 7,923 | 1,614 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.01% | 4.06% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 1.05% (max 2.57%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| `mouth_far` (near / far post): dropped | +0.0123 [+0.0105, +0.0143] | +0.0129 [+0.0083, +0.0174] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 2,632 | 3.05% | 2.93% |
| 1 | 2,632 | 5.84% | 4.83% |
| 2 | 2,632 | 9.02% | 8.47% |
| 3 | 2,631 | 12.93% | 11.59% |
| 4 | 2,632 | 18.17% | 18.54% |
| 5 | 2,632 | 25.22% | 25.61% |
| 6 | 2,631 | 34.21% | 35.42% |
| 7 | 2,632 | 45.99% | 47.72% |
| 8 | 2,632 | 61.80% | 63.75% |
| 9 | 2,631 | 84.87% | 82.21% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| gender = male | 26,317 | 7,923 | 30.11% | 7923.5008 | 1.0001 | 0.4469 | 0.1442 | 0.2695 | 0.9972 |
| body_part = right_foot | 13,163 | 4,000 | 30.39% | 3943.5140 | 0.9859 | 0.4508 | 0.1455 | 0.2659 | 0.9951 |
| body_part = left_foot | 8,647 | 2,430 | 28.10% | 2443.5736 | 1.0056 | 0.4470 | 0.1438 | 0.2474 | 0.9554 |
| body_part = head | 4,416 | 1,446 | 32.74% | 1486.5416 | 1.0280 | 0.4340 | 0.1408 | 0.3137 | 1.0958 |
| body_part = other | 91 | 47 | 51.65% | 49.8715 | 1.0611 | 0.4990 | 0.1661 | 0.2796 | 0.9857 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the test value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Test | Test floor |
|---|---|---|---|---|
| 0.0143 | 0.0178 | 0.0113 | 0.0146 | 0.0130 |

### Calibration by placement band

Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, after rounding); a ball beyond 3.55 m touches the post. The reliability table bins by prediction, so it cannot show a placement the model misprices. SD is √Σ p(1 − p), the spread of the goal count if xGOT is calibrated; *Within 2 SD* is the PM's screening rule (MS6), not a proof of calibration: a calibrated model leaves a band outside it by chance in about 4.6% of cases. The rule is a normal approximation, so a band with fewer than 100 shots is printed but not judged (PM, MS6).

| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals | SD | Within 2 SD |
|---|---|---|---|---|---|---|---|
| [0, 1) | 7,552 | 1,277 | 0.1691 | 0.1700 | +6.5 | 29.1 | yes |
| [1, 2) | 7,814 | 1,936 | 0.2478 | 0.2457 | -16.5 | 32.5 | yes |
| [2, 3) | 7,591 | 3,044 | 0.4010 | 0.4044 | +25.7 | 36.1 | yes |
| [3, 3.3) | 2,544 | 1,332 | 0.5236 | 0.4976 | -66.1 | 21.1 | no |
| [3.3, 3.55) | 787 | 329 | 0.4180 | 0.4818 | +50.2 | 11.8 | no |
| >= 3.55 | 29 | 5 | 0.1724 | 0.1969 | +0.7 | 1.8 | not judged (n < 100) |
| outside the frame + 0.15 m | 0 | 0 | n/a | n/a | n/a | 0.0 | not judged (n < 100) |

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
- On the test, Σ xGOT is 1,548.5 against 1,614 goals (-4.06%; calibration intercept 0.0876). A level gap moves every keeper's goals saved against this xGOT the same way, so a season's keepers read better or worse as a group.
- Calibration by placement band: 2 of 7 bands are outside 2 SD on dev out of fold ([3, 3.3): 2,544 shots, 1,332 goals, Σ xGOT − goals −66.1 (SD 21.1); [3.3, 3.55): 787 shots, 329 goals, Σ xGOT − goals +50.2 (SD 11.8)). This model's placement terms misprice them; the headline is Monotone GBM + Platt, which the band rule judges (PM E1).

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
- v3 (MS6): saves recorded where the keeper stopped the ball and projected to the line are snapped to the 0.1 yd goal-mouth lattice of recorded points (football-core); the GBM's placement features carry no monotone constraint (chosen by the band rule); the band table prints the SD; the band rule is a check.
- v4 (MS6, GSAx): matches recorded on the coarse (1 yd) grid leave the xGOT fit (PM E1); a band is judged by the band rule only with at least 100 shots.
