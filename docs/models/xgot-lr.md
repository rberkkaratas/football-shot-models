# Model card: LR xG + placement

## Overview
- **Name / version:** `xgot-lr` v4
- **Task:** binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty xG population (a goal or a keeper save; own goals are never on target)
- **Status:** experimental
- P2.3: logistic regression of the goal on the logit of the cross-fitted xG, placement splines, the width past the post's inner edge, angle × placement and the near / far post. Not the headline model; that is Monotone GBM + Platt.

## Intended use
xGOT (post-shot xG) for on-target shots: P(goal) of a shot that reached the goal frame, from where it crossed the line and its pre-shot xG. Keeper-neutral (no keeper position), so goals saved against it credit the keeper's positioning too. For research and as the input of the MS6 goalkeeper models; not a rating of finishers, and not for other providers' data without re-fitting.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `a10b7542f2dc13c3b1f46191c8af3e52b9b6f9281543cc7a436f9797b881209e`
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
xG input: Monotone GBM + Platt, trained without team-selected seasons without keeper position, cross-fitted; provider headline xG: Monotone GBM + Platt, trained without team-selected seasons.

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
| intercept | -1.2199 | -0.3134 |
| xg_logit | 1.1377 | 0.8978 |
| mouth_y_abs | 0.0369 | 0.0366 |
| mouth_y_abs_rcs1 | 0.0832 | 0.0840 |
| mouth_y_abs_rcs2 | 1.5667 | 3.8016 |
| mouth_y_abs_rcs3 | -0.9007 | -9.0469 |
| mouth_y_past | -0.2113 | -17.4475 |
| mouth_z | -0.1314 | -0.2074 |
| mouth_z_rcs1 | -1.8791 | -3.4454 |
| mouth_z_rcs2 | 1.9336 | 5.1890 |
| mouth_z_rcs3 | 0.2532 | 2.5732 |
| angle_x_mouth_y | 0.3770 | 0.4752 |
| angle_x_mouth_z | 0.1315 | 0.2776 |
| mouth_far | 0.3726 | 0.1930 |

## Validation design
Grouped 5-fold cross-validation by `match_id` (`GroupKFold`, no shuffle), the folds of the xG run restricted to on-target shots: every shot keeps its fold, so no match has shots in two folds and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets. **Train** is the fit on all on-target dev shots, scored in-sample. **Test** is the hold-out competition-season named under Data, scored once by that fit. The open-play xGOT reads the xG input cross-fitted on these folds: an on-target shot in test fold k reads the xG of the fit that never saw fold k, and a hold-out (test) shot the xG of the fit on all of dev (train), so no xGOT row sees an xG fitted on itself. The xGOT training rows of fold k read xG from fits that did see fold k: the standard cross-fitting dependence, accepted.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4177 | 0.4185 ± 0.0083 | 0.4113 |
| Brier score | 0.1338 | 0.1341 ± 0.0034 | 0.1318 |
| ROC AUC | 0.8619 | 0.8616 ± 0.0048 | 0.8637 |
| Log-loss skill | 0.3163 | 0.3151 | 0.3186 |
| Brier skill | 0.3630 | 0.3618 | 0.3616 |
| Calibration intercept | 0.0000 | 0.0001 | 0.0231 |
| Calibration slope | 1.0000 | 0.9999 | 0.9883 |
| Σ xGOT | 8808.0000 | 8809.0180 | 903.6475 |
| Goals | 8,808 | 8,808 | 913 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.01% | 1.02% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.44% (max 4.79%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| `mouth_far` (near / far post): dropped | +0.0105 [+0.0089, +0.0122] | +0.0074 [+0.0023, +0.0125] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 2,935 | 2.40% | 1.94% |
| 1 | 2,935 | 4.49% | 3.41% |
| 2 | 2,935 | 7.12% | 5.72% |
| 3 | 2,935 | 10.85% | 9.13% |
| 4 | 2,935 | 16.00% | 16.01% |
| 5 | 2,935 | 23.44% | 25.45% |
| 6 | 2,935 | 33.86% | 37.38% |
| 7 | 2,935 | 47.90% | 48.62% |
| 8 | 2,935 | 66.06% | 67.84% |
| 9 | 2,934 | 88.04% | 84.63% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 12,031 | 3,499 | 29.08% | 3420.0995 | 0.9775 | 0.4066 | 0.1299 | 0.3256 | 1.0210 |
| family = team-selected | 6,667 | 2,090 | 31.35% | 2137.6257 | 1.0228 | 0.4335 | 0.1399 | 0.3030 | 0.9795 |
| gender = female | 10,651 | 3,219 | 30.22% | 3251.2928 | 1.0100 | 0.4223 | 0.1350 | 0.3108 | 0.9844 |
| gender = male | 18,698 | 5,589 | 29.89% | 5557.7252 | 0.9944 | 0.4162 | 0.1335 | 0.3177 | 1.0034 |
| body_part = right_foot | 15,585 | 4,511 | 28.94% | 4490.6322 | 0.9955 | 0.4078 | 0.1304 | 0.3222 | 1.0033 |
| body_part = left_foot | 9,314 | 2,707 | 29.06% | 2663.7906 | 0.9840 | 0.4108 | 0.1313 | 0.3184 | 1.0331 |
| body_part = head | 4,321 | 1,512 | 34.99% | 1574.3793 | 1.0413 | 0.4656 | 0.1505 | 0.2809 | 0.9418 |
| body_part = other | 129 | 78 | 60.47% | 80.2158 | 1.0284 | 0.6706 | 0.2236 | 0.0007 | 0.4526 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0199 | 0.0259 | 0.0161 | 0.0164 | 0.0156 |

### Calibration by placement band

Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, after rounding); a ball beyond 3.55 m touches the post. The reliability table bins by prediction, so it cannot show a placement the model misprices. SD is √Σ p(1 − p), the spread of the goal count if xGOT is calibrated; *Within 2 SD* is the PM's screening rule (MS6), not a proof of calibration: a calibrated model leaves a band outside it by chance in about 4.6% of cases. The rule is a normal approximation, so a band with fewer than 100 shots is printed but not judged (PM, MS6).

| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals | SD | Within 2 SD |
|---|---|---|---|---|---|---|---|
| [0, 1) | 8,079 | 1,212 | 0.1500 | 0.1520 | +16.4 | 28.5 | yes |
| [1, 2) | 8,186 | 1,692 | 0.2067 | 0.2030 | -30.2 | 31.0 | yes |
| [2, 3) | 8,969 | 3,429 | 0.3823 | 0.3865 | +37.9 | 37.9 | yes |
| [3, 3.3) | 3,072 | 1,915 | 0.6234 | 0.5941 | -89.9 | 23.0 | no |
| [3.3, 3.55) | 866 | 505 | 0.5831 | 0.6544 | +61.8 | 12.1 | no |
| >= 3.55 | 147 | 52 | 0.3537 | 0.3862 | +4.8 | 4.9 | yes |
| outside the frame + 0.15 m | 30 | 3 | 0.1000 | 0.1099 | +0.3 | 1.4 | not judged (n < 100) |

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
| LR xG only | 0.5168 | 0.1713 | 0.7611 | 1.0010 | 0.5201 | 0.1726 | 0.7493 | 0.9490 |
| LR xG + placement, no near / far | 0.4290 | 0.1380 | 0.8544 | 1.0002 | 0.4187 | 0.1347 | 0.8588 | 1.0128 |
| LR xG + placement | 0.4185 | 0.1341 | 0.8616 | 0.9999 | 0.4113 | 0.1318 | 0.8637 | 0.9883 |
| Monotone GBM + Platt | 0.4087 | 0.1308 | 0.8672 | 1.0112 | 0.4030 | 0.1295 | 0.8685 | 0.9945 |

## Limitations & known biases
- 34 of 32,483 on-target shots (hold-out included) are recorded beyond the frame + 0.15 m and are clipped to its edge and flagged by `mouth_out`, since where they would have crossed the line is not known; 1,381 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 32,483 of 32,483 on-target shots, so there is no provider xGOT benchmark.
- StatsBomb records a save where the keeper stopped the ball: its `goal_z` is that height, not the height at the goal line, and a save stopped short of the line has its `goal_y` projected along the line from the shot location and snapped to the 0.1 yd goal-mouth lattice of recorded points (797 of the on-target saves were deflected, so their projection starts at the shot, not at the deflection).
- On the hold-out, Σ xGOT is 903.6 against 913 goals (-1.02%; calibration intercept 0.0231). A level gap moves every keeper's goals saved against this xGOT the same way, so a season's keepers read better or worse as a group.
- Calibration by placement band: 2 of 7 bands are outside 2 SD on dev out of fold ([3, 3.3): 3,072 shots, 1,915 goals, Σ xGOT − goals −89.9 (SD 23.0); [3.3, 3.55): 866 shots, 505 goals, Σ xGOT − goals +61.8 (SD 12.1)). This model's placement terms misprice them; the headline is Monotone GBM + Platt, which the band rule judges (PM E1).

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
- v3 (MS6): saves recorded where the keeper stopped the ball and projected to the line are snapped to the 0.1 yd goal-mouth lattice of recorded points (football-core); the GBM's placement features carry no monotone constraint (chosen by the band rule); the band table prints the SD; the band rule is a check.
- v4 (MS6, GSAx): matches recorded on the coarse (1 yd) grid leave the xGOT fit (PM E1); a band is judged by the band rule only with at least 100 shots.
