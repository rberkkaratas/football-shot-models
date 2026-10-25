# Model card: LR distance + angle

## Overview
- **Name / version:** `xg-lr-distance-angle` v4
- **Task:** binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time included; direct free kicks and corners from open play or set pieces included; direct corners, own goals and shoot-outs excluded)
- **Status:** experimental
- Baseline for M1.1. Not the best baseline; that is LR distance + angle + body part.

## Intended use
A transparent floor for the xG models of this repo: how much of P(goal) shot geometry (and body part) alone explains on StatsBomb open data. Later models must beat it on the same folds. It is not for rating players or teams, not for penalties, and not for other providers' data without re-validation.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `31146e70cefb5fe39056b8ef24216724a5d5162601decb2e214c0608b2eb3e78`
- **Competitions & seasons:** 80 competition-seasons; hold-out Premier League 2015/2016
- **Rows:** train and validation (dev) 89,897 shots in 3,581 matches / test (hold-out) 9,816 shots in 380 matches; men 65,803 shots, women 33,910
- **Families (segments):** women; team-selected, where one team plays in at least half of a competition-season's matches (56 competition-seasons, 30 of them with one or two matches, mostly single finals, where both teams qualify trivially); league or tournament, the rest
- **Filters:** excluded shots, by first matching reason. The per-shot log is snapshot `xg-exclusions-v1` (git-ignored; its committed manifest holds counts only).

| Filter | Shots | Goals |
|---|---|---|
| own goal (Own Goal Against events; not shots) | 387 | 387 |
| shoot-out (period 5; all are penalties) | 363 | 242 |
| penalty (in-game) | 1,118 | 853 |
| direct corner (logged as a shot on outcome) | 33 | 18 |
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
| intercept | -2.4593 | -1.7045 | n/a |
| distance_m | -0.5931 | -0.0758 | 0.93 |
| angle_rad | 0.3486 | 1.2615 | 3.53 |

## Validation design
Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match has shots in two folds, and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one of the four 2015/16 men's league seasons in the data, with the other three in dev. It tests transfer to an unseen league in the same season and data spec, not across time or gender.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.2961 | 0.2961 ± 0.0021 | 0.2788 |
| Brier score | 0.0844 | 0.0844 ± 0.0007 | 0.0783 |
| ROC AUC | 0.7388 | 0.7388 ± 0.0041 | 0.7351 |
| Log-loss skill | 0.1099 | 0.1098 | 0.1009 |
| Brier skill | 0.0899 | 0.0898 | 0.0731 |
| Calibration intercept | -0.0000 | -0.0002 | -0.1049 |
| Calibration slope | 1.0000 | 1.0001 | 0.9648 |
| Σ xG | 9302.0000 | 9302.4652 | 995.4495 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xG − goals) / goals | 0.00% | 0.01% | 9.03% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 2.16% (max 3.11%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### MS2 thresholds (retired; see the headline model's MS3 checks)

| Threshold | CV | Hold-out | CV result |
|---|---|---|---|
| log loss <= 0.275 | 0.2961 | 0.2788 | missed |
| Brier <= 0.077 | 0.0844 | 0.0783 | missed |
| ROC AUC >= 0.75 | 0.7388 | 0.7351 | missed |
| abs(Σ xG − goals) / goals <= 3.00% | 0.01% pooled; fold mean 2.16% (max 3.11%) | 9.03% | met |

Σ xG and calibration on grouped CV are close to their targets by construction for the base rate and for an unpenalised logistic regression with an intercept; see the hold-out column and the reliability table.

**Why.** Absolute log loss and Brier depend on the goal rate and on which matches are in the sample. The base rate alone scores 0.3327 / 0.0928 on the CV shots (goal rate 10.35%) and 0.3101 / 0.0845 on the hold-out (9.30%). By segment, out of fold, this model scores 0.2777 / 0.0782 (log-loss skill 0.1159) on league or tournament shots (Σ xG / goals 1.0362), 0.3254 / 0.0952 (log-loss skill 0.1086) on team-selected seasons (19.77% of dev shots, goal rate 11.90%, Σ xG / goals 0.8235) and 0.3015 / 0.0859 (log-loss skill 0.1012) on women's shots (Σ xG / goals 1.0682); skill is against each segment's own goal rate. Its log-loss skill against the base rate is 0.1098 on CV and 0.1009 on the hold-out. StatsBomb xG, with freeze-frame inputs, scores 0.2659 / 0.0757 on the CV shots. Thresholds come from a single-season pilot (roadmap); they are reported, not tuned to.

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 8,990 | 2.25% | 2.17% |
| 1 | 8,990 | 3.42% | 2.63% |
| 2 | 8,990 | 4.32% | 3.82% |
| 3 | 8,989 | 5.32% | 4.41% |
| 4 | 8,990 | 6.57% | 6.16% |
| 5 | 8,990 | 8.14% | 9.84% |
| 6 | 8,989 | 10.20% | 13.41% |
| 7 | 8,990 | 13.28% | 14.45% |
| 8 | 8,990 | 18.25% | 15.88% |
| 9 | 8,989 | 31.73% | 30.72% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 38,218 | 3,634 | 9.51% | 3765.4735 | 1.0362 | 0.2777 | 0.0782 | 0.1159 | 1.0419 |
| family = team-selected | 17,769 | 2,115 | 11.90% | 1741.6622 | 0.8235 | 0.3254 | 0.0952 | 0.1086 | 1.0657 |
| gender = female | 33,910 | 3,553 | 10.48% | 3795.3295 | 1.0682 | 0.3015 | 0.0859 | 0.1012 | 0.9457 |
| gender = male | 55,987 | 5,749 | 10.27% | 5507.1357 | 0.9579 | 0.2928 | 0.0836 | 0.1151 | 1.0454 |
| body_part = right_foot | 47,459 | 4,784 | 10.08% | 3920.6338 | 0.8195 | 0.2822 | 0.0805 | 0.1366 | 1.2561 |
| body_part = left_foot | 27,727 | 2,829 | 10.20% | 2340.6361 | 0.8274 | 0.2887 | 0.0822 | 0.1237 | 1.2576 |
| body_part = head | 14,428 | 1,609 | 11.15% | 2962.2121 | 1.8410 | 0.3520 | 0.1001 | -0.0067 | 1.4070 |
| body_part = other | 283 | 80 | 28.27% | 78.9832 | 0.9873 | 0.5039 | 0.1640 | 0.1537 | 1.4379 |

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Base rate | 0.3327 | 0.0928 | 0.5000 | n/a | 0.3101 | 0.0845 | 0.5000 | n/a |
| LR distance + angle | 0.2961 | 0.0844 | 0.7388 | 1.0001 | 0.2788 | 0.0783 | 0.7351 | 0.9648 |
| LR distance + angle + body part | 0.2872 | 0.0820 | 0.7640 | 0.9996 | 0.2721 | 0.0766 | 0.7556 | 0.9650 |
| StatsBomb xG | 0.2659 | 0.0757 | 0.8072 | 1.0166 | 0.2520 | 0.0704 | 0.7963 | 0.9916 |

StatsBomb xG is a reference ceiling, not a like-for-like out-of-sample competitor: StatsBomb fits it on its own data, which likely includes these matches, and it uses freeze-frame and other inputs.

## Limitations & known biases
- Distance, angle and body part only: no defenders, keeper position or pressure.
- Men and women are pooled; see the gender and family segments.
- Team-selected seasons (one team in ≥ 50 % of a competition-season's matches) over-represent a few clubs. The rule also takes every one- or two-match competition-season (Champions League, World Cup and Copa del Rey finals, among others), since both teams then play in all of its matches; see Data.
- Direct corners are logged as Shot events mostly when they threaten goal, so they are outcome-selected and excluded (see Filters). Wide direct free kicks are selected the same way; they stay in the population, scored by geometry only (F5).
- `situation` is unused (`play_pattern` spans the possession, C4 F3).
- Penalties are a separate model (M1.5).

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS2): first baseline.
- v2 (MS3): direct corners excluded (logged on outcome), all numbers re-measured; Segments show the pooled-OOF calibration slope.
- v3 (MS3, M1.4): the benchmark table shows calibration slopes.
- v4 (MS3 assessment): the MS2 thresholds are marked retired; the family = women segment row (the gender = female row) is printed once.
