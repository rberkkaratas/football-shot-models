# Model card: Monotone GBM + Platt

## Overview
- **Name / version:** `xgot-gbm` v2
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.4: monotone gradient boosting on the xG input, placement, side and angle, tuned and Platt-calibrated inside each fit by grouped CV. This is the headline model: its CV log-loss gain over LR xG + placement is +0.0106 [+0.0089, +0.0124], a 95% interval that lies above 0; on the hold-out it is +0.0091 [+0.0039, +0.0143], an interval that lies above 0.

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
| `mouth_y_in` | `min(mouth_y_abs, 3.55)`: the distance from the goal centre up to where a ball starts to touch the post, m | `mouth_y_abs` |
| `mouth_y_past` | `mouth_y_abs − mouth_y_in`: how far past that point the ball crossed, m | `mouth_y_abs` |
| `mouth_out` | 1 when the point was recorded beyond the frame + 0.15 m (before clipping), else 0 | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_z` | Height above the ground, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `far_side` | +1 far post, −1 near post (the ball's side against the shot's), 0 for a shot within 0.5 m of the centre line or a ball at the centre | `y`, `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `angle_rad` | Opening angle between the posts seen from the shot location, radians | `angle_rad` (`pitch.opening_angle`) |

Monotone constraints of the fitted model (every other column is free):

- xg_logit: non-decreasing — a higher xG at the same placement never lowers xGOT
- mouth_y_in: non-decreasing — further from the centre, up to 3.55 m, at the same height and on the same side (`far_side` fixed) never helps the keeper

Each constraint holds along one column with every other column held fixed: `mouth_y_in` moves the ball towards a post at the same height, on the same side (near or far). Beyond 3.55 m from the centre a ball's centre is within its radius of the post, so a shot there hits the post or is saved: `mouth_y_past` and `mouth_out` are free.

## Model
`HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1, min_samples_leaf=100, early_stopping=False, random_state=0)`, `monotonic_cst` placed by column name, no class weights. scikit-learn 1.9.1.

Tuning inside each fit: 4-fold `GroupKFold` by match on its training rows, `max_leaf_nodes` in (8, 16, 32) and 50, 100, ..., 600 iterations (staged scores of one fit per leaf count); the lowest inner out-of-fold log loss wins, ties to fewer leaves, then fewer iterations. Platt scaling: an unpenalised logistic regression of the goal on the chosen configuration's inner out-of-fold log-odds, applied to the refit on all training rows; a slope <= 0 is refused.

Tuning and calibration see only the rows of the fit they belong to; each outer test fold is scored by a model that never saw it.

Settings of the fit on all of dev:

| Setting | Value |
|---|---|
| max_leaf_nodes | 8 |
| iterations | 250 |
| Platt intercept | -0.0028 |
| Platt slope | 0.9934 |

Calibration is close to identity (Platt slope 0.993, intercept -0.003): the boosted model is already calibrated and Platt is a safeguard.

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. **Test** is the hold-out competition-season named under Data, scored once by that fit. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.3987 | 0.4085 ± 0.0081 | 0.3996 |
| Brier score | 0.1273 | 0.1307 ± 0.0034 | 0.1284 |
| ROC AUC | 0.8737 | 0.8671 ± 0.0053 | 0.8710 |
| Log-loss skill | 0.3466 | 0.3305 | 0.3379 |
| Brier skill | 0.3926 | 0.3766 | 0.3784 |
| Calibration intercept | -0.0006 | 0.0001 | 0.0554 |
| Calibration slope | 1.0459 | 1.0115 | 1.0083 |
| Σ xGOT | 9304.3776 | 9302.8724 | 890.9170 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xGOT − goals) / goals | 0.03% | 0.01% | 2.42% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.32% (max 4.77%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR xG + placement | +0.0106 [+0.0089, +0.0124] | +0.0091 [+0.0039, +0.0143] |

### Milestone checks (MS5)

Bounds from the PM brief (MS5); AUC >= 0.80 reads "about 0.80 or better" strictly. ECE is judged on the CV fold mean and the point value of the scored-once split (PM X3).

| Target | Measured | Result |
|---|---|---|
| ROC AUC >= 0.80, CV fold mean | 0.8671 | met |
| ROC AUC >= 0.80, hold-out | 0.8710 | met |
| ECE < 0.02, CV fold mean | 0.0152 (max 0.0178, pooled 0.0050) | met |
| ECE < 0.02, hold-out | 0.0107 (noise floor 0.0151) | met |
| Calibration slope in [0.9, 1.1], CV fold mean | 1.0115 | met |
| Calibration slope in [0.9, 1.1], hold-out | 1.0083 | met |
| GBM headline iff its CV gain over LR xG + placement has a 95% interval above 0 | CV +0.0106 [+0.0089, +0.0124]; headline Monotone GBM + Platt | met |
| On-target shots scored reconcile with C3 within 1% (two-sided) | 35,604 of 35,632 (-0.079%); unscored 0 | met |
| On-target goals scored reconcile with C3 within 1% (two-sided) | 11,310 of 11,328 (-0.159%) | met |
| xG input Monotone GBM + Platt without keeper position: cross-fitted on the same folds, keeper-neutral, not an ablation (the run raises otherwise) | guards ran | met |
| No keeper-position feature in the headline xGOT (the xGOT guard raises otherwise) | xg_logit, mouth_y_in, mouth_y_past, mouth_out, mouth_z, far_side, angle_rad | met |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 3,110 | 1.74% | 1.74% |
| 1 | 3,109 | 3.40% | 3.12% |
| 2 | 3,109 | 5.78% | 5.28% |
| 3 | 3,110 | 9.62% | 10.29% |
| 4 | 3,109 | 15.35% | 14.09% |
| 5 | 3,109 | 23.87% | 23.99% |
| 6 | 3,110 | 35.33% | 36.75% |
| 7 | 3,109 | 50.68% | 50.31% |
| 8 | 3,109 | 68.06% | 68.35% |
| 9 | 3,109 | 85.38% | 85.27% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 12,492 | 3,634 | 29.09% | 3522.1192 | 0.9692 | 0.3953 | 0.1263 | 0.3444 | 1.0340 |
| family = team-selected | 6,759 | 2,115 | 31.29% | 2116.5384 | 1.0007 | 0.4224 | 0.1355 | 0.3203 | 1.0031 |
| gender = female | 11,842 | 3,553 | 30.00% | 3664.2148 | 1.0313 | 0.4145 | 0.1326 | 0.3215 | 0.9904 |
| gender = male | 19,251 | 5,749 | 29.86% | 5638.6576 | 0.9808 | 0.4048 | 0.1295 | 0.3360 | 1.0218 |
| body_part = right_foot | 16,587 | 4,784 | 28.84% | 4729.5363 | 0.9886 | 0.3998 | 0.1274 | 0.3345 | 1.0038 |
| body_part = left_foot | 9,803 | 2,829 | 28.86% | 2788.8985 | 0.9858 | 0.4014 | 0.1282 | 0.3320 | 1.0163 |
| body_part = head | 4,570 | 1,609 | 35.21% | 1699.7503 | 1.0564 | 0.4500 | 0.1458 | 0.3063 | 1.0499 |
| body_part = other | 133 | 80 | 60.15% | 84.6873 | 1.0586 | 0.5938 | 0.2029 | 0.1169 | 0.7312 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0152 | 0.0178 | 0.0050 | 0.0107 | 0.0151 |

### Calibration by placement band

Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, after rounding); a ball beyond 3.55 m touches the post. The reliability table bins by prediction, so it cannot show a placement the model misprices.

| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals |
|---|---|---|---|---|---|
| [0, 1) | 8,625 | 1,261 | 0.1462 | 0.1480 | +15.7 |
| [1, 2) | 8,412 | 1,757 | 0.2089 | 0.2097 | +7.4 |
| [2, 3) | 9,325 | 3,559 | 0.3817 | 0.3816 | -0.5 |
| [3, 3.3) | 3,113 | 2,012 | 0.6463 | 0.6069 | -122.8 |
| [3.3, 3.55) | 1,147 | 609 | 0.5310 | 0.6134 | +94.6 |
| >= 3.55 | 223 | 87 | 0.3901 | 0.4037 | +3.0 |
| outside the frame + 0.15 m | 248 | 17 | 0.0685 | 0.0823 | +3.4 |

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
| 20 | left out | 1,058 | +0.036 [-0.024, +0.096] | -0.019 [-0.079, +0.041] |
| 40 | in | 633 | +0.084 [+0.006, +0.161] | +0.076 [-0.002, +0.153] |
| 40 | left out | 397 | +0.110 [+0.011, +0.206] | +0.021 [-0.078, +0.119] |
| 80 | in | 175 | +0.031 [-0.118, +0.179] | +0.029 [-0.119, +0.177] |
| 80 | left out | 85 | +0.075 [-0.141, +0.283] | -0.017 [-0.230, +0.196] |

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
- On the hold-out, Σ xGOT is 890.9 against 913 goals (-2.42%; calibration intercept 0.0554). A level gap moves every keeper's goals saved against this xGOT the same way, so a season's keepers read better or worse as a group.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
