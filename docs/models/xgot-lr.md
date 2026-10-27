# Model card: LR xG + placement

## Overview
- **Name / version:** `xgot-lr` v1
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.3: logistic regression of the goal on the logit of the cross-fitted xG, placement splines, angle × placement and the near / far post. Not the headline model; that is Monotone GBM + Platt.

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
| intercept | -1.2055 | -0.3466 |
| xg_logit | 1.0866 | 0.8486 |
| mouth_y_abs | 0.1579 | 0.1540 |
| mouth_y_abs_rcs1 | -1.7236 | -1.6460 |
| mouth_y_abs_rcs2 | 4.8495 | 10.9702 |
| mouth_y_abs_rcs3 | -2.6730 | -24.3826 |
| mouth_z | -0.2921 | -0.4615 |
| mouth_z_rcs1 | -0.3854 | -0.7033 |
| mouth_z_rcs2 | 0.4279 | 1.1422 |
| mouth_z_rcs3 | 0.3889 | 3.9258 |
| angle_x_mouth_y | 0.3552 | 0.4418 |
| angle_x_mouth_z | 0.1594 | 0.3346 |
| mouth_far | 0.3430 | 0.1742 |

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. **Test** is the hold-out competition-season named under Data, scored once by that fit. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4293 | 0.4299 ± 0.0080 | 0.4146 |
| Brier score | 0.1381 | 0.1383 ± 0.0030 | 0.1334 |
| ROC AUC | 0.8536 | 0.8535 ± 0.0048 | 0.8614 |
| Log-loss skill | 0.2965 | 0.2955 | 0.3130 |
| Brier skill | 0.3414 | 0.3405 | 0.3540 |
| Calibration intercept | -0.0000 | 0.0001 | 0.0563 |
| Calibration slope | 1.0000 | 1.0000 | 1.0500 |
| Σ xGOT | 9302.0000 | 9302.8646 | 889.2104 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.01% | 2.61% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.42% (max 4.90%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| `mouth_far` (near / far post): dropped | +0.0091 [+0.0075, +0.0107] | +0.0075 [+0.0028, +0.0122] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 3,110 | 2.60% | 2.22% |
| 1 | 3,109 | 4.85% | 3.38% |
| 2 | 3,109 | 7.71% | 5.95% |
| 3 | 3,110 | 11.67% | 9.74% |
| 4 | 3,109 | 16.90% | 17.05% |
| 5 | 3,109 | 24.12% | 27.02% |
| 6 | 3,110 | 33.86% | 36.56% |
| 7 | 3,109 | 46.88% | 48.21% |
| 8 | 3,109 | 64.29% | 66.90% |
| 9 | 3,109 | 86.34% | 82.15% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 12,492 | 3,634 | 29.09% | 3531.6861 | 0.9718 | 0.4160 | 0.1336 | 0.3100 | 1.0378 |
| family = team-selected | 6,759 | 2,115 | 31.29% | 2121.3892 | 1.0030 | 0.4386 | 0.1419 | 0.2942 | 1.0245 |
| gender = female | 11,842 | 3,553 | 30.00% | 3649.7893 | 1.0272 | 0.4393 | 0.1412 | 0.2809 | 0.9466 |
| gender = male | 19,251 | 5,749 | 29.86% | 5653.0753 | 0.9833 | 0.4240 | 0.1365 | 0.3046 | 1.0316 |
| body_part = right_foot | 16,587 | 4,784 | 28.84% | 4776.2298 | 0.9984 | 0.4205 | 0.1348 | 0.3000 | 0.9948 |
| body_part = left_foot | 9,803 | 2,829 | 28.86% | 2791.2764 | 0.9867 | 0.4227 | 0.1359 | 0.2965 | 1.0318 |
| body_part = head | 4,570 | 1,609 | 35.21% | 1652.1233 | 1.0268 | 0.4714 | 0.1537 | 0.2733 | 0.9743 |
| body_part = other | 133 | 80 | 60.15% | 83.2351 | 1.0404 | 0.6834 | 0.2233 | -0.0164 | 0.4308 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0232 | 0.0265 | 0.0194 | 0.0187 | 0.0160 |

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Base rate (on target) | 0.6102 | 0.2097 | 0.5000 | n/a | 0.6035 | 0.2065 | 0.5000 | n/a |
| LR xG only | 0.5165 | 0.1712 | 0.7612 | 1.0012 | 0.5184 | 0.1720 | 0.7512 | 0.9712 |
| LR xG + placement, no near / far | 0.4389 | 0.1419 | 0.8467 | 1.0002 | 0.4221 | 0.1365 | 0.8568 | 1.0678 |
| LR xG + placement | 0.4299 | 0.1383 | 0.8535 | 1.0000 | 0.4146 | 0.1334 | 0.8614 | 1.0500 |
| Monotone GBM + Platt | 0.4161 | 0.1337 | 0.8619 | 1.0109 | 0.4030 | 0.1298 | 0.8691 | 1.0254 |

## Limitations & known biases
- 252 of 34,227 on-target shots (hold-out included) are recorded beyond the frame + 0.15 m and are clipped to its edge; 1,534 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 34,227 of 34,227 on-target shots, so there is no provider xGOT benchmark.
- StatsBomb records a save where the keeper stopped the ball: its `goal_z` is that height, not the height at the goal line, and a save stopped short of the line has its `goal_y` projected along the line from the shot location (818 of the on-target saves were deflected, so their projection starts at the shot, not at the deflection).

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
