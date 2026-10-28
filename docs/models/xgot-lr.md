# Model card: LR xG + placement

## Overview
- **Name / version:** `xgot-lr` v2
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.3: logistic regression of the goal on the logit of the cross-fitted xG, placement splines, the width past the post's inner edge, angle × placement and the near / far post. Not the headline model; that is Monotone GBM + Platt.

## Intended use
xGOT (post-shot xG) for on-target shots: P(goal) of a shot that reached the goal frame, from where it crossed the line and its pre-shot xG. Keeper-neutral (no keeper position), so goals saved against it credit the keeper's positioning too. For research and as the input of the MS6 goalkeeper models; not a rating of finishers, and not for other providers' data without re-fitting.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `63002fdd75a8889807a717c20035b3c42132534408074f2ecdd40118b35d62be`
- **Competitions & seasons:** 80 competition-seasons; hold-out Premier League 2015/2016
- **Rows:** train and validation (dev) 31,093 shots in 3,581 matches / test (hold-out) 3,134 shots in 380 matches; men 22,385 shots, women 11,842
- **Families (segments):** women; team-selected, where one team plays in at least half of a competition-season's matches (56 competition-seasons, 30 of them with one or two matches, mostly single finals, where both teams qualify trivially); league or tournament, the rest
- **Filters:** excluded shots, by first matching reason. C3 on-target shots by first matching reason; penalty and shoot-out rows are scored by the penalty xGOT, modelled rows by the LR and GBM.

| Filter | Shots | Goals |
|---|---|---|
| own goal (Own Goal Against events; not shots) | 0 | 0 |
| shoot-out kick (period 5; penalty xGOT, reported apart) | 321 | 242 |
| penalty (in-game; penalty xGOT) | 1,056 | 853 |
| direct corner (logged as a shot on outcome) | 28 | 18 |
| no freeze frame | 0 | 0 |
| unresolved assisting event (left out of the xG population) | 0 | 0 |
| modelled (open-play xGOT) | 34,227 | 10,215 |

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
| intercept | -1.2250 | -0.2632 |
| xg_logit | 1.1220 | 0.8763 |
| mouth_y_abs | -0.1428 | -0.1393 |
| mouth_y_abs_rcs1 | 1.2066 | 1.1523 |
| mouth_y_abs_rcs2 | 0.3485 | 0.7884 |
| mouth_y_abs_rcs3 | -0.6283 | -5.7308 |
| mouth_y_past | -0.4965 | -20.5777 |
| mouth_z | -0.1972 | -0.3116 |
| mouth_z_rcs1 | -1.1964 | -2.1831 |
| mouth_z_rcs2 | 1.2240 | 3.2673 |
| mouth_z_rcs3 | 0.3269 | 3.2998 |
| angle_x_mouth_y | 0.3693 | 0.4594 |
| angle_x_mouth_z | 0.1462 | 0.3068 |
| mouth_far | 0.3842 | 0.1951 |

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. **Test** is the hold-out competition-season named under Data, scored once by that fit. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4185 | 0.4191 ± 0.0097 | 0.4087 |
| Brier score | 0.1340 | 0.1343 ± 0.0037 | 0.1310 |
| ROC AUC | 0.8609 | 0.8608 ± 0.0058 | 0.8656 |
| Log-loss skill | 0.3142 | 0.3131 | 0.3228 |
| Brier skill | 0.3607 | 0.3597 | 0.3658 |
| Calibration intercept | 0.0000 | 0.0002 | 0.0385 |
| Calibration slope | 1.0000 | 1.0004 | 1.0214 |
| Σ xGOT | 9302.0000 | 9302.5145 | 897.2159 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.01% | 1.73% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.37% (max 4.76%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| `mouth_far` (near / far post): dropped | +0.0110 [+0.0092, +0.0127] | +0.0078 [+0.0025, +0.0130] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 3,110 | 2.38% | 2.12% |
| 1 | 3,109 | 4.53% | 3.41% |
| 2 | 3,109 | 7.18% | 5.56% |
| 3 | 3,110 | 10.92% | 9.49% |
| 4 | 3,109 | 16.02% | 16.47% |
| 5 | 3,109 | 23.38% | 24.45% |
| 6 | 3,110 | 33.62% | 37.68% |
| 7 | 3,109 | 47.54% | 47.86% |
| 8 | 3,109 | 65.81% | 68.00% |
| 9 | 3,109 | 87.81% | 84.14% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 12,492 | 3,634 | 29.09% | 3535.8889 | 0.9730 | 0.4073 | 0.1303 | 0.3245 | 1.0293 |
| family = team-selected | 6,759 | 2,115 | 31.29% | 2142.7700 | 1.0131 | 0.4313 | 0.1391 | 0.3060 | 1.0014 |
| gender = female | 11,842 | 3,553 | 30.00% | 3623.8556 | 1.0199 | 0.4246 | 0.1356 | 0.3049 | 0.9654 |
| gender = male | 19,251 | 5,749 | 29.86% | 5678.6589 | 0.9878 | 0.4157 | 0.1334 | 0.3182 | 1.0173 |
| body_part = right_foot | 16,587 | 4,784 | 28.84% | 4767.6956 | 0.9966 | 0.4084 | 0.1304 | 0.3201 | 1.0025 |
| body_part = left_foot | 9,803 | 2,829 | 28.86% | 2802.7240 | 0.9907 | 0.4132 | 0.1321 | 0.3124 | 1.0185 |
| body_part = head | 4,570 | 1,609 | 35.21% | 1649.4375 | 1.0251 | 0.4639 | 0.1503 | 0.2849 | 0.9642 |
| body_part = other | 133 | 80 | 60.15% | 82.6575 | 1.0332 | 0.6506 | 0.2154 | 0.0325 | 0.4986 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0197 | 0.0245 | 0.0162 | 0.0174 | 0.0157 |

### Calibration by placement band

Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, after rounding); a ball beyond 3.55 m touches the post. The reliability table bins by prediction, so it cannot show a placement the model misprices.

| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals |
|---|---|---|---|---|---|
| [0, 1) | 8,625 | 1,261 | 0.1462 | 0.1524 | +53.8 |
| [1, 2) | 8,412 | 1,757 | 0.2089 | 0.2041 | -40.0 |
| [2, 3) | 9,325 | 3,559 | 0.3817 | 0.3864 | +44.4 |
| [3, 3.3) | 3,113 | 2,012 | 0.6463 | 0.5906 | -173.4 |
| [3.3, 3.55) | 1,147 | 609 | 0.5310 | 0.6306 | +114.3 |
| >= 3.55 | 223 | 87 | 0.3901 | 0.3776 | -2.8 |
| outside the frame + 0.15 m | 248 | 17 | 0.0685 | 0.0857 | +4.3 |

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Base rate (on target) | 0.6102 | 0.2097 | 0.5000 | n/a | 0.6035 | 0.2065 | 0.5000 | n/a |
| LR xG only | 0.5165 | 0.1712 | 0.7612 | 1.0012 | 0.5184 | 0.1720 | 0.7512 | 0.9712 |
| LR xG + placement, no near / far | 0.4301 | 0.1384 | 0.8530 | 1.0005 | 0.4165 | 0.1340 | 0.8609 | 1.0500 |
| LR xG + placement | 0.4191 | 0.1343 | 0.8608 | 1.0004 | 0.4087 | 0.1310 | 0.8656 | 1.0214 |
| Monotone GBM + Platt | 0.4085 | 0.1307 | 0.8671 | 1.0115 | 0.3996 | 0.1284 | 0.8710 | 1.0083 |

## Limitations & known biases
- 252 of 34,227 on-target shots (hold-out included) are recorded beyond the frame + 0.15 m and are clipped to its edge and flagged by `mouth_out`, since where they would have crossed the line is not known; 1,534 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 34,227 of 34,227 on-target shots, so there is no provider xGOT benchmark.
- StatsBomb records a save where the keeper stopped the ball: its `goal_z` is that height, not the height at the goal line, and a save stopped short of the line has its `goal_y` projected along the line from the shot location (818 of the on-target saves were deflected, so their projection starts at the shot, not at the deflection).
- On the hold-out, Σ xGOT is 897.2 against 913 goals (-1.73%; calibration intercept 0.0385). A level gap moves every keeper's goals saved against this xGOT the same way, so a season's keepers read better or worse as a group.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
