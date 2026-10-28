# Model card: Penalty xGOT (placement only)

## Overview
- **Name / version:** `xgot-penalty` v2
- **Task:** P(goal | on-target penalty) from placement only; fitted on in-game kicks, shoot-out kicks scored apart
- **Status:** experimental
- P2.4: penalty xGOT, a logistic regression on placement only; it reads no xG, since every in-game penalty has the same pre-shot xG. Placement-only, reported apart from the open-play xGOT.

## Intended use
Penalty xGOT: P(goal) of an on-target penalty from where it crossed the line, kept apart from the open-play xGOT. For research and as the penalty input of the MS6 goalkeeper models; not a rating of takers or keepers on its own.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `63002fdd75a8889807a717c20035b3c42132534408074f2ecdd40118b35d62be`
- **Competitions & seasons:** 54 competition-seasons; hold-out Premier League 2015/2016
- **Rows:** train and validation (dev) 971 shots in 849 matches / test (hold-out) 85 shots in 75 matches; men 710 shots, women 346
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
| intercept | 1.4534 | -0.4156 |
| mouth_y_abs | 0.4389 | 0.5593 |
| mouth_z | 0.9157 | 1.5068 |
| mouth_yz | -0.6110 | -0.3628 |

## Validation design
Grouped 5-fold cross-validation by `match_id` on the in-game dev kicks (their own folds); the hold-out kicks are scored once by the fit on all of them, and the shoot-out kicks of every season by that fit too. The penalty xGOT reads no xG input.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.4829 | 0.4860 ± 0.0446 | 0.3904 |
| Brier score | 0.1555 | 0.1565 ± 0.0184 | 0.1150 |
| ROC AUC | 0.6400 | 0.6384 ± 0.0576 | 0.5663 |
| Log-loss skill | 0.0289 | 0.0247 | 0.0277 |
| Brier skill | 0.0197 | 0.0156 | 0.0201 |
| Calibration intercept | 0.0000 | 0.0130 | 0.5239 |
| Calibration slope | 1.0000 | 0.9860 | 0.9168 |
| Σ xGOT | 779.0000 | 778.1167 | 68.1950 |
| Goals | 779 | 779 | 74 |
| abs(Σ xGOT − goals) / goals | 0.00% | 0.11% | 7.84% |
| abs(Σ xGOT − goals) / goals, fold mean (max) | n/a | 2.81% (max 4.24%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xGOT | Goal rate |
|---|---|---|---|
| 0 | 98 | 65.38% | 76.53% |
| 1 | 97 | 73.84% | 63.92% |
| 2 | 97 | 77.12% | 73.20% |
| 3 | 97 | 78.97% | 78.35% |
| 4 | 97 | 80.41% | 78.35% |
| 5 | 97 | 81.68% | 81.44% |
| 6 | 97 | 83.01% | 84.54% |
| 7 | 97 | 84.75% | 80.41% |
| 8 | 97 | 86.67% | 87.63% |
| 9 | 97 | 89.67% | 97.94% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xGOT | Σ xGOT / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 424 | 332 | 78.30% | 339.1385 | 1.0215 | 0.4967 | 0.1622 | 0.0504 | 1.5119 |
| family = team-selected | 201 | 171 | 85.07% | 160.5628 | 0.9390 | 0.4354 | 0.1341 | -0.0331 | 0.3496 |
| gender = female | 346 | 276 | 79.77% | 278.4155 | 1.0088 | 0.5023 | 0.1625 | 0.0025 | 0.5515 |
| gender = male | 625 | 503 | 80.48% | 499.7012 | 0.9934 | 0.4770 | 0.1532 | 0.0338 | 1.0731 |
| body_part = right_foot | 715 | 575 | 80.42% | 573.1673 | 0.9968 | 0.4825 | 0.1549 | 0.0244 | 0.9479 |
| body_part = left_foot | 256 | 204 | 79.69% | 204.9494 | 1.0047 | 0.4960 | 0.1611 | 0.0173 | 0.7448 |

### Expected calibration error

ECE (10 equal-count bins): the shot-weighted mean of |mean prediction − goal rate| over the bins of the reliability table. CV is judged on the fold mean, with the worst fold and the pooled out-of-fold value beside it; the hold-out value has beside it its noise floor, the mean ECE of outcomes drawn from its own predictions (200 draws, seed 0), which a perfectly calibrated model reaches by chance.

| CV fold mean | CV max | CV pooled | Hold-out | Hold-out floor |
|---|---|---|---|---|
| 0.0892 | 0.1077 | 0.0431 | 0.1279 | 0.1073 |

### Shoot-out kicks (reported apart)

321 on-target shoot-out kicks (242 goals), scored by the fit on all in-game dev kicks and never pooled with them: Σ xGOT 259.0, log loss 0.5591.

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Penalty base rate (on target) | 0.4983 | 0.1590 | 0.5000 | n/a | 0.4016 | 0.1173 | 0.5000 | n/a |
| Penalty xGOT (placement only) | 0.4860 | 0.1565 | 0.6384 | 0.9860 | 0.3904 | 0.1150 | 0.5663 | 0.9168 |

## Limitations & known biases
- Against the open-play bounds, each on its own: CV calibration slope 0.9860 (inside [0.9, 1.1]), hold-out slope 0.9168 (inside), CV ECE 0.0892 (fold mean; not below 0.02). Its probabilities are not calibrated to the open-play standard; as a ranking of placements its AUC is 0.6384 on CV and 0.5663 on the hold-out (0.5 is no ranking).
- 3 of 1,056 on-target shots (hold-out included) are recorded beyond the frame + 0.15 m and are clipped to its edge; 1,053 are struck within 0.5 m of the centre line, so they have no near or far post.
- `provider_psxg` is null on 1,056 of 1,056 on-target shots, so there is no provider xGOT benchmark.
- StatsBomb records a save where the keeper stopped the ball: its `goal_z` is that height, not the height at the goal line, and a save stopped short of the line has its `goal_y` projected along the line from the shot location (0 of the on-target saves were deflected, so their projection starts at the shot, not at the deflection).

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS5, P2.2–P2.5): first version.
- v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and `mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the scored-once split's level gap; penalty calibration judged bound by bound.
