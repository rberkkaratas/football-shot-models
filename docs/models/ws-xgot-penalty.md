# Model card: Penalty xGOT (placement only)

## Overview
- **Name / version:** `ws-xgot-penalty` v2
- **Task:** P(goal | on-target penalty) from placement only; fitted on in-game kicks, shoot-out kicks scored apart
- **Status:** experimental
- P2.4: penalty xGOT, a logistic regression on placement only; it reads no xG, since every in-game penalty has the same pre-shot xG. Placement-only, reported apart from the open-play xGOT.

## Intended use
Penalty xGOT: P(goal) of an on-target penalty from where it crossed the line, kept apart from the open-play xGOT. For research and as the penalty input of the MS6 goalkeeper models; not a rating of takers or keepers on its own.

## Data
- **Source(s):** private WhoScored / Opta event archive, revision `84571265636c3c30fcfcf9b9b0b3bd1381f7488fc3d3870893d2539f8d7c3b23`; snapshot `whoscored-v1`, data digest `8033836b08b03fcbaffe2a19cc3a9549f2adb2eebf10f2871fff57f267b79ebb`
- **Competitions & seasons:** 20 competition-seasons; hold-out 2026/2027
- **Rows:** train and validation (dev) 856 shots in 739 matches / test (hold-out) 137 shots in 119 matches
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
| Feature | Definition | Source field(s) |
|---|---|---|
| `mouth_y_abs` | Distance from the goal centre across the goal, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_z` | Height above the ground, m, clipped to the frame + 0.15 m and rounded to 0.1 m | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |
| `mouth_yz` | `mouth_y_abs` × `mouth_z` | `goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only |

## Model
`make_pipeline(StandardScaler(), LogisticRegression(C=inf, solver='newton-cholesky', tol=1e-10, max_iter=1000))`: unpenalised logistic regression, no class weights, no calibration step; the scaler is fitted on the training rows of each fit. scikit-learn 1.9.1.

Coefficients of the fit on all of dev (natural units: log-odds per unit of the feature).

| Term | Standardised | Natural units |
|---|---|---|
| intercept | 1.6470 | 0.4772 |
| mouth_y_abs | 0.3428 | 0.4014 |
| mouth_z | 0.5494 | 1.0043 |
| mouth_yz | -0.3314 | -0.2584 |

## Validation design
Grouped 5-fold cross-validation by `match_id` on the in-game dev kicks (their own folds); the test kicks are scored once by the fit on all of them, and the shoot-out kicks of every season by that fit too. The penalty xGOT reads no xG input.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4424 | 0.4444 ± 0.0365 | 0.3847 |
| Brier score | 0.1370 | 0.1377 ± 0.0153 | 0.1121 |
| ROC AUC | 0.6015 | 0.5999 ± 0.0292 | 0.5279 |
| Log-loss skill | 0.0154 | 0.0129 | -0.0076 |
| Brier skill | 0.0098 | 0.0066 | -0.0153 |
| Calibration intercept | 0.0000 | 0.0082 | 0.3148 |
| Calibration slope | 1.0000 | 0.9784 | 0.0399 |
| Σ xGOT | 714.0000 | 713.9480 | 114.7961 |
| Goals | 714 | 714 | 120 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.01% | 4.34% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.83% (max 4.19%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 86 | 73.48% | 83.72% |
| 1 | 86 | 79.27% | 77.91% |
| 2 | 85 | 81.20% | 87.06% |
| 3 | 86 | 82.69% | 69.77% |
| 4 | 85 | 83.73% | 75.29% |
| 5 | 86 | 84.64% | 86.05% |
| 6 | 86 | 85.53% | 80.23% |
| 7 | 85 | 86.48% | 83.53% |
| 8 | 86 | 87.44% | 93.02% |
| 9 | 85 | 89.68% | 97.65% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| gender = male | 856 | 714 | 83.41% | 713.9480 | 0.9999 | 0.4444 | 0.1377 | 0.0109 | 0.8388 |
| body_part = right_foot | 635 | 523 | 82.36% | 529.9139 | 1.0132 | 0.4644 | 0.1456 | 0.0031 | 0.6276 |
| body_part = left_foot | 221 | 191 | 86.43% | 184.0341 | 0.9635 | 0.3870 | 0.1150 | 0.0256 | 1.6153 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the test value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Test | Test floor |
|---|---|---|---|---|
| 0.0975 | 0.1312 | 0.0620 | 0.0962 | 0.0809 |

### Shoot-out kicks (reported apart)

16 on-target shoot-out kicks (13 goals), scored by the fit on all in-game dev kicks and never pooled with them: Σ xGOT 13.3, log loss 0.5031.

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Penalty base rate (on target) | 0.4503 | 0.1386 | 0.5000 | n/a | 0.3818 | 0.1104 | 0.5000 | n/a |
| Penalty xGOT (placement only) | 0.4444 | 0.1377 | 0.5999 | 0.9784 | 0.3847 | 0.1121 | 0.5279 | 0.0399 |

## Limitations & known biases
- Against the open-play bounds, each on its own: CV calibration slope 0.9784 (inside [0.9, 1.1]), test slope 0.0399 (outside), CV ECE 0.0975 (fold mean; not below 0.02). Its probabilities are not calibrated to the open-play standard; as a ranking of placements its AUC is 0.5999 on CV and 0.5279 on the test (0.5 is no ranking).
- 0 of 993 on-target shots (test included) are recorded beyond the frame + 0.15 m and are clipped to its edge; 993 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 993 of 993 on-target shots, so there is no provider xGOT benchmark.

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
