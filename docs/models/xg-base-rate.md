# Model card: Base rate

## Overview
- **Name / version:** `xg-base-rate` v1
- **Task:** binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time included; direct free kicks and corners included; own goals and shoot-outs excluded)
- **Status:** experimental
- Baseline for M1.1. Not the best baseline; that is LR distance + angle + body part.

## Intended use
A transparent floor for the xG models of this repo: how much of P(goal) shot geometry (and body part) alone explains on StatsBomb open data. Later models must beat it on the same folds. It is not for rating players or teams, not for penalties, and not for other providers' data without re-validation.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `9b34e93fa3c8300f9d49d107a4fa92dfd4eed55d2cf72d8ba878afd37cc7d0ad`
- **Competitions & seasons:** 80 competition-seasons; hold-out Premier League 2015/2016
- **Rows:** train and validation (dev) 89,929 shots in 3,581 matches / test (hold-out) 9,817 shots in 380 matches; men 65,822 shots, women 33,924
- **Families (segments):** women; team-selected, where one team plays in at least half of a competition-season's matches (56 competition-seasons, 30 of them with one or two matches, mostly single finals, where both teams qualify trivially); league or tournament, the rest
- **Filters:** excluded shots, by first matching reason. The per-shot log is snapshot `xg-exclusions-v1` (git-ignored; its committed manifest holds counts only).

| Filter | Shots | Goals |
|---|---|---|
| own goal (Own Goal Against events; not shots) | 387 | 387 |
| shoot-out (period 5; all are penalties) | 363 | 242 |
| penalty (in-game) | 1,118 | 853 |
| no freeze frame | 0 | 0 |

## Target
`is_goal`: the ball went in. Own goals are excluded, so every goal is the shooter's.

## Features
None: every shot gets the goal rate of the training shots.

## Model
Constant prediction: the goal rate of the training shots (10.36% on all of dev). Nothing else is fitted. scikit-learn 1.9.1 (metrics and folds).

## Validation design
Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match has shots in two folds, and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one of the four 2015/16 men's league seasons in the data, with the other three in dev. It tests transfer to an unseen league in the same season and data spec, not across time or gender.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.3330 | 0.3330 ± 0.0058 | 0.3103 |
| Brier score | 0.0929 | 0.0929 ± 0.0021 | 0.0845 |
| ROC AUC | 0.5000 | 0.5000 ± 0.0000 | 0.5000 |
| Log-loss skill | 0.0000 | 0.0000 | 0.0000 |
| Brier skill | 0.0000 | 0.0000 | 0.0000 |
| Calibration intercept | -0.0000 | -0.0002 | -0.1187 |
| Calibration slope | n/a | n/a | n/a |
| Σ xG | 9319.0000 | 9319.0092 | 1017.2983 |
| Goals | 9,319 | 9,319 | 914 |
| abs(Σ xG − goals) / goals | 0.00% | 0.00% | 11.30% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 2.56% (max 4.30%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Milestone thresholds

| Threshold | CV | Hold-out | CV result |
|---|---|---|---|
| log loss <= 0.275 | 0.3330 | 0.3103 | missed |
| Brier <= 0.077 | 0.0929 | 0.0845 | missed |
| ROC AUC >= 0.75 | 0.5000 | 0.5000 | missed |
| abs(Σ xG − goals) / goals <= 3.00% | 0.00% pooled; fold mean 2.56% (max 4.30%) | 11.30% | met |

Σ xG and calibration on grouped CV are close to their targets by construction for the base rate and for an unpenalised logistic regression with an intercept; see the hold-out column and the reliability table.

**Why.** Absolute log loss and Brier depend on the goal rate and on which matches are in the sample. The base rate alone scores 0.3330 / 0.0929 on the CV shots (goal rate 10.36%) and 0.3103 / 0.0845 on the hold-out (9.31%). By segment, out of fold, this model scores 0.3147 / 0.0862 (log-loss skill -0.0013) on league or tournament shots (Σ xG / goals 1.0890), 0.3662 / 0.1051 (log-loss skill -0.0034) on team-selected seasons (19.76% of dev shots, goal rate 11.90%, Σ xG / goals 0.8709) and 0.3362 / 0.0941 (log-loss skill -0.0001) on women's shots (Σ xG / goals 0.9858); skill is against each segment's own goal rate. Its log-loss skill against the base rate is 0.0000 on CV and 0.0000 on the hold-out. StatsBomb xG, with freeze-frame inputs, scores 0.2674 / 0.0758 on the CV shots. Thresholds come from a single-season pilot (roadmap); they are reported, not tuned to.

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 8,993 | 10.27% | 10.62% |
| 1 | 8,993 | 10.27% | 10.84% |
| 2 | 8,993 | 10.32% | 10.96% |
| 3 | 8,993 | 10.32% | 10.09% |
| 4 | 8,993 | 10.38% | 10.45% |
| 5 | 8,993 | 10.38% | 10.11% |
| 6 | 8,993 | 10.39% | 9.99% |
| 7 | 8,993 | 10.39% | 10.52% |
| 8 | 8,993 | 10.44% | 10.22% |
| 9 | 8,992 | 10.44% | 9.83% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier | Log-loss skill |
|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 38,233 | 3,638 | 9.52% | 3961.6520 | 1.0890 | 0.3147 | 0.0862 | -0.0013 |
| family = team-selected | 17,772 | 2,115 | 11.90% | 1841.8963 | 0.8709 | 0.3662 | 0.1051 | -0.0034 |
| family = women | 33,924 | 3,566 | 10.51% | 3515.4608 | 0.9858 | 0.3362 | 0.0941 | -0.0001 |
| gender = female | 33,924 | 3,566 | 10.51% | 3515.4608 | 0.9858 | 0.3362 | 0.0941 | -0.0001 |
| gender = male | 56,005 | 5,753 | 10.27% | 5803.5483 | 1.0088 | 0.3310 | 0.0922 | -0.0001 |
| body_part = right_foot | 47,477 | 4,794 | 10.10% | 4919.7296 | 1.0262 | 0.3273 | 0.0908 | -0.0002 |
| body_part = left_foot | 27,741 | 2,836 | 10.22% | 2874.7148 | 1.0137 | 0.3300 | 0.0918 | -0.0001 |
| body_part = head | 14,428 | 1,609 | 11.15% | 1495.2299 | 0.9293 | 0.3500 | 0.0991 | -0.0010 |
| body_part = other | 283 | 80 | 28.27% | 29.3349 | 0.3667 | 0.7190 | 0.2348 | -0.2074 |

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | Hold-out log loss | Hold-out Brier | Hold-out AUC |
|---|---|---|---|---|---|---|
| Base rate | 0.3330 | 0.0929 | 0.5000 | 0.3103 | 0.0845 | 0.5000 |
| LR distance + angle | 0.2968 | 0.0846 | 0.7376 | 0.2792 | 0.0784 | 0.7344 |
| LR distance + angle + body part | 0.2880 | 0.0821 | 0.7628 | 0.2726 | 0.0767 | 0.7548 |
| StatsBomb xG | 0.2674 | 0.0758 | 0.8057 | 0.2529 | 0.0705 | 0.7954 |

StatsBomb xG is a reference ceiling, not a like-for-like out-of-sample competitor: StatsBomb fits it on its own data, which likely includes these matches, and it uses freeze-frame and other inputs.

## Limitations & known biases
- Distance, angle and body part only: no defenders, keeper position or pressure.
- Men and women are pooled; see the gender and family segments.
- Team-selected seasons (one team in ≥ 50 % of a competition-season's matches) over-represent a few clubs. The rule also takes every one- or two-match competition-season (Champions League, World Cup and Copa del Rey finals, among others), since both teams then play in all of its matches; see Data.
- Direct corners and wide direct free kicks are logged as Shot events mostly when they threaten goal, so they are outcome-selected. They stay in the population, scored by geometry only, and `situation` must not become a feature before C3/C4 restrict those values (F5).
- `situation` is unused (`play_pattern` spans the possession, C4 F3).
- Penalties are a separate model (M1.5).

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS2): first baseline.
