# Model card: Monotone GBM + Platt

## Overview
- **Name / version:** `xgot-gbm` v1
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.4: monotone gradient boosting on the xG input, placement, side and angle, tuned and Platt-calibrated inside each fit by grouped CV. This is the headline model: its CV log-loss gain over LR xG + placement is +0.0137 [+0.0118, +0.0157], a 95% interval that lies above 0; on the hold-out it is +0.0116 [+0.0059, +0.0170], an interval that lies above 0.

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
| max_leaf_nodes | 8 |
| iterations | 200 |
| Platt intercept | +0.0016 |
| Platt slope | 1.0019 |

Calibration is close to identity (Platt slope 1.002, intercept +0.002): the boosted model is already calibrated and Platt is a safeguard.

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. **Test** is the hold-out competition-season named under Data, scored once by that fit. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4073 | 0.4161 ± 0.0078 | 0.4030 |
| Brier score | 0.1307 | 0.1337 ± 0.0033 | 0.1298 |
| ROC AUC | 0.8680 | 0.8619 ± 0.0053 | 0.8691 |
| Log-loss skill | 0.3325 | 0.3180 | 0.3322 |
| Brier skill | 0.3768 | 0.3621 | 0.3713 |
| Calibration intercept | 0.0003 | 0.0010 | 0.0821 |
| Calibration slope | 1.0393 | 1.0109 | 1.0254 |
| Σ xGOT | 9300.6051 | 9298.9641 | 879.7047 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xGOT − goals) / goals | 0.01% | 0.03% | 3.65% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.24% (max 4.80%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR xG + placement | +0.0137 [+0.0118, +0.0157] | +0.0116 [+0.0059, +0.0170] |

### Milestone checks (MS5)

Bounds from the PM brief (MS5); AUC >= 0.80 reads "about 0.80 or better" strictly. ECE is judged on the CV fold mean and the point value of the scored-once split (PM X3).

| Target | Measured | Result |
|---|---|---|
| ROC AUC >= 0.80, CV fold mean | 0.8619 | met |
| ROC AUC >= 0.80, hold-out | 0.8691 | met |
| ECE < 0.02, CV fold mean | 0.0151 (max 0.0167, pooled 0.0062) | met |
| ECE < 0.02, hold-out | 0.0142 (noise floor 0.0151) | met |
| Calibration slope in [0.9, 1.1], CV fold mean | 1.0109 | met |
| Calibration slope in [0.9, 1.1], hold-out | 1.0254 | met |
| GBM headline iff its CV gain over LR xG + placement has a 95% interval above 0 | CV +0.0137 [+0.0118, +0.0157]; headline Monotone GBM + Platt | met |
| On-target shots scored reconcile with C3 within 1% (two-sided) | 35,604 of 35,632 (-0.079%); unscored 0 | met |
| On-target goals scored reconcile with C3 within 1% (two-sided) | 11,310 of 11,328 (-0.159%) | met |
| xG input Monotone GBM + Platt without keeper position: cross-fitted on the same folds, keeper-neutral, not an ablation (the run raises otherwise) | guards ran | met |
| No keeper-position feature in the headline xGOT (the xGOT guard raises otherwise) | xg_logit, mouth_y_abs, mouth_z, far_side, angle_rad | met |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 3,110 | 1.82% | 1.70% |
| 1 | 3,109 | 3.51% | 3.25% |
| 2 | 3,109 | 6.00% | 5.21% |
| 3 | 3,110 | 10.09% | 10.71% |
| 4 | 3,109 | 16.07% | 14.92% |
| 5 | 3,109 | 24.52% | 24.90% |
| 6 | 3,110 | 35.71% | 37.07% |
| 7 | 3,109 | 50.13% | 50.37% |
| 8 | 3,109 | 67.12% | 67.67% |
| 9 | 3,109 | 84.11% | 83.37% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 12,492 | 3,634 | 29.09% | 3514.1023 | 0.9670 | 0.4012 | 0.1286 | 0.3347 | 1.0402 |
| family = team-selected | 6,759 | 2,115 | 31.29% | 2094.1852 | 0.9902 | 0.4279 | 0.1376 | 0.3114 | 1.0181 |
| gender = female | 11,842 | 3,553 | 30.00% | 3690.6765 | 1.0387 | 0.4251 | 0.1369 | 0.3041 | 0.9777 |
| gender = male | 19,251 | 5,749 | 29.86% | 5608.2876 | 0.9755 | 0.4105 | 0.1318 | 0.3266 | 1.0315 |
| body_part = right_foot | 16,587 | 4,784 | 28.84% | 4731.7581 | 0.9891 | 0.4091 | 0.1311 | 0.3190 | 0.9958 |
| body_part = left_foot | 9,803 | 2,829 | 28.86% | 2780.5246 | 0.9829 | 0.4079 | 0.1308 | 0.3212 | 1.0267 |
| body_part = head | 4,570 | 1,609 | 35.21% | 1700.3715 | 1.0568 | 0.4535 | 0.1474 | 0.3010 | 1.0649 |
| body_part = other | 133 | 80 | 60.15% | 86.3099 | 1.0789 | 0.6109 | 0.2092 | 0.0915 | 0.6872 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0151 | 0.0167 | 0.0062 | 0.0142 | 0.0151 |

### Reconciliation with C3

Every C3 on-target shot is in one row of the Filters table; the scored rows are the open-play and penalty xGOT predictions.

| | Shots | Goals |
|---|---|---|
| C3 on target | 35,632 | 11,328 |
| Scored by xGOT | 35,604 | 11,310 |
| Difference | -0.079% | -0.159% |

Population rows without a prediction: 0.

### Shooting goals added (descriptive)

Shooting goals added (SGA) is xGOT − xG per shot, next to goals − xG; both subtract the same xG, the xG input above. Blocked and off-target shots count as 0 xGOT. Each player's matches are ordered by date (then match id) and alternated into two halves, so a match never spans halves; r is the Pearson correlation across players of the per-shot value in the two halves, with a 95% interval by Fisher z (n/a under 4 players). A descriptive measure of shots beating their xG on target, not a finishing-skill claim.

| Min shots | Blocked shots | Players | SGA per shot: r [95% CI] | Goals − xG per shot: r [95% CI] |
|---|---|---|---|---|
| 20 | in | 1,418 | +0.027 [-0.025, +0.079] | -0.010 [-0.062, +0.042] |
| 20 | left out | 1,058 | +0.031 [-0.029, +0.091] | -0.019 [-0.079, +0.041] |
| 40 | in | 633 | +0.083 [+0.005, +0.160] | +0.076 [-0.002, +0.153] |
| 40 | left out | 397 | +0.113 [+0.015, +0.209] | +0.021 [-0.078, +0.119] |
| 80 | in | 175 | +0.056 [-0.093, +0.202] | +0.029 [-0.119, +0.177] |
| 80 | left out | 85 | +0.159 [-0.056, +0.360] | -0.017 [-0.230, +0.196] |

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
