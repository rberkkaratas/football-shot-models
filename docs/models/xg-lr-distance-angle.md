# Model card: LR distance + angle

## Overview
- **Name / version:** `xg-lr-distance-angle` v1
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
| Feature | Definition | Source field(s) |
|---|---|---|
| `distance_m` | Metres from the shot location to the goal centre | `distance_m` (`pitch.shot_distance`) |
| `angle_rad` | Opening angle between the posts seen from the shot location, radians | `angle_rad` (`pitch.opening_angle`) |

## Model
`make_pipeline(StandardScaler(), LogisticRegression(C=inf, tol=1e-10, max_iter=1000))`: unpenalised logistic regression (lbfgs), no class weights, no calibration step. The scaler is fitted on the training rows of each fit. scikit-learn 1.9.1.

Coefficients of the fit on all of dev. Natural units are log-odds per metre, per radian, or vs right foot at the same spot.

| Term | Standardised | Natural units | Odds ratio |
|---|---|---|---|
| intercept | -2.4525 | -1.7309 | n/a |
| distance_m | -0.5808 | -0.0741 | 0.93 |
| angle_rad | 0.3523 | 1.2743 | 3.58 |

## Validation design
Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match has shots in two folds, and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one of the four 2015/16 men's league seasons in the data, with the other three in dev. It tests transfer to an unseen league in the same season and data spec, not across time or gender.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.2968 | 0.2968 ± 0.0059 | 0.2792 |
| Brier score | 0.0846 | 0.0846 ± 0.0021 | 0.0784 |
| ROC AUC | 0.7375 | 0.7376 ± 0.0049 | 0.7344 |
| Log-loss skill | 0.1087 | 0.1086 | 0.1001 |
| Brier skill | 0.0895 | 0.0894 | 0.0729 |
| Calibration intercept | -0.0000 | -0.0002 | -0.1060 |
| Calibration slope | 1.0000 | 1.0005 | 0.9670 |
| Σ xG | 9319.0000 | 9319.0109 | 997.4701 |
| Goals | 9,319 | 9,319 | 914 |
| abs(Σ xG − goals) / goals | 0.00% | 0.00% | 9.13% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 2.96% (max 4.23%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Milestone thresholds

| Threshold | CV | Hold-out | CV result |
|---|---|---|---|
| log loss <= 0.275 | 0.2968 | 0.2792 | missed |
| Brier <= 0.077 | 0.0846 | 0.0784 | missed |
| ROC AUC >= 0.75 | 0.7376 | 0.7344 | missed |
| abs(Σ xG − goals) / goals <= 3.00% | 0.00% pooled; fold mean 2.96% (max 4.23%) | 9.13% | met |

Σ xG and calibration on grouped CV are close to their targets by construction for the base rate and for an unpenalised logistic regression with an intercept; see the hold-out column and the reliability table.

**Why.** Absolute log loss and Brier depend on the goal rate and on which matches are in the sample. The base rate alone scores 0.3330 / 0.0929 on the CV shots (goal rate 10.36%) and 0.3103 / 0.0845 on the hold-out (9.31%). By segment, out of fold, this model scores 0.2781 / 0.0782 (log-loss skill 0.1152) on league or tournament shots (Σ xG / goals 1.0370), 0.3252 / 0.0952 (log-loss skill 0.1088) on team-selected seasons (19.76% of dev shots, goal rate 11.90%, Σ xG / goals 0.8252) and 0.3030 / 0.0862 (log-loss skill 0.0986) on women's shots (Σ xG / goals 1.0659); skill is against each segment's own goal rate. Its log-loss skill against the base rate is 0.1086 on CV and 0.1001 on the hold-out. StatsBomb xG, with freeze-frame inputs, scores 0.2674 / 0.0758 on the CV shots. Thresholds come from a single-season pilot (roadmap); they are reported, not tuned to.

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 8,993 | 2.30% | 2.38% |
| 1 | 8,993 | 3.48% | 2.58% |
| 2 | 8,993 | 4.38% | 3.85% |
| 3 | 8,993 | 5.37% | 4.39% |
| 4 | 8,993 | 6.61% | 6.12% |
| 5 | 8,993 | 8.16% | 9.83% |
| 6 | 8,993 | 10.20% | 13.37% |
| 7 | 8,993 | 13.26% | 14.48% |
| 8 | 8,993 | 18.20% | 16.00% |
| 9 | 8,992 | 31.67% | 30.64% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier | Log-loss skill |
|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 38,233 | 3,638 | 9.52% | 3772.7494 | 1.0370 | 0.2781 | 0.0782 | 0.1152 |
| family = team-selected | 17,772 | 2,115 | 11.90% | 1745.4025 | 0.8252 | 0.3252 | 0.0952 | 0.1088 |
| family = women | 33,924 | 3,566 | 10.51% | 3800.8591 | 1.0659 | 0.3030 | 0.0862 | 0.0986 |
| gender = female | 33,924 | 3,566 | 10.51% | 3800.8591 | 1.0659 | 0.3030 | 0.0862 | 0.0986 |
| gender = male | 56,005 | 5,753 | 10.27% | 5518.1519 | 0.9592 | 0.2931 | 0.0836 | 0.1147 |
| body_part = right_foot | 47,477 | 4,794 | 10.10% | 3934.3693 | 0.8207 | 0.2830 | 0.0807 | 0.1350 |
| body_part = left_foot | 27,741 | 2,836 | 10.22% | 2348.7939 | 0.8282 | 0.2897 | 0.0824 | 0.1220 |
| body_part = head | 14,428 | 1,609 | 11.15% | 2956.9869 | 1.8378 | 0.3518 | 0.1000 | -0.0059 |
| body_part = other | 283 | 80 | 28.27% | 78.8608 | 0.9858 | 0.5038 | 0.1639 | 0.1539 |

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
