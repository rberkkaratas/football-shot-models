# Model card: LR splines + body part

## Overview
- **Name / version:** `xg-lr-splines` v3
- **Task:** binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time included; direct free kicks and corners from open play or set pieces included; direct corners, own goals and shoot-outs excluded)
- **Status:** experimental
- M1.2 step S1: the MS2 best baseline plus spline terms of distance and angle. Not the headline model; that is Monotone GBM + Platt, trained without team-selected seasons.

## Intended use
A transparent, reproducible xG model on StatsBomb open data, built one feature group at a time so each group's gain is measured on the same folds. For research and for comparing with StatsBomb's own xG; not for rating players or teams without that comparison, not for penalties, and not for other providers' data (it needs StatsBomb freeze frames and annotations) without re-fitting.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `a10b7542f2dc13c3b1f46191c8af3e52b9b6f9281543cc7a436f9797b881209e`
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
| `distance_m_rcs1` | Restricted cubic spline term 1 of `distance_m`, knots 6, 11.5, 17, 22.5, 30 m (`features.rcs_term`) | `distance_m` |
| `distance_m_rcs2` | Restricted cubic spline term 2 of `distance_m`, knots 6, 11.5, 17, 22.5, 30 m (`features.rcs_term`) | `distance_m` |
| `distance_m_rcs3` | Restricted cubic spline term 3 of `distance_m`, knots 6, 11.5, 17, 22.5, 30 m (`features.rcs_term`) | `distance_m` |
| `angle_rad` | Opening angle between the posts seen from the shot location, radians | `angle_rad` (`pitch.opening_angle`) |
| `angle_rad_rcs1` | Restricted cubic spline term 1 of `angle_rad`, knots 0.2, 0.27, 0.35, 0.5, 1 rad (`features.rcs_term`) | `angle_rad` |
| `angle_rad_rcs2` | Restricted cubic spline term 2 of `angle_rad`, knots 0.2, 0.27, 0.35, 0.5, 1 rad (`features.rcs_term`) | `angle_rad` |
| `angle_rad_rcs3` | Restricted cubic spline term 3 of `angle_rad`, knots 0.2, 0.27, 0.35, 0.5, 1 rad (`features.rcs_term`) | `angle_rad` |
| `body_part` | Indicators for left foot, head and other; reference right foot | `body_part` |

## Model
`make_pipeline(StandardScaler(), LogisticRegression(C=inf, solver='newton-cholesky', tol=1e-10, max_iter=1000))`: unpenalised logistic regression, no class weights, no calibration step. The scaler is fitted on the training rows of each fit; a singular Hessian is an error, never a silent switch to another solver. scikit-learn 1.9.1.

Splines are restricted cubic splines with fixed knots (rounded quantiles of the shots, no labels): distance 6, 11.5, 17, 22.5, 30 m, angle 0.2, 0.27, 0.35, 0.5, 1 rad.

S0 (the MS2 baseline it is compared with) is fitted with lbfgs; on S0's features the two solvers agree to 1.0e-7 in every out-of-fold probability.

Coefficients of the fit on all of dev. Natural units are log-odds per unit of the feature (metre, radian, spline term) or vs the reference level at the same spot.

| Term | Standardised | Natural units |
|---|---|---|
| intercept | -2.5278 | -1.3592 |
| distance_m | -0.6449 | -0.0824 |
| distance_m_rcs1 | -0.4144 | -0.0493 |
| distance_m_rcs2 | 0.1966 | 0.0519 |
| distance_m_rcs3 | 0.3913 | 0.3426 |
| angle_rad | -0.2169 | -0.7848 |
| angle_rad_rcs1 | 45.8419 | 225.2386 |
| angle_rad_rcs2 | -74.6633 | -541.7246 |
| angle_rad_rcs3 | 29.5334 | 383.8582 |
| body_part=left_foot | -0.0047 | -0.0103 |
| body_part=head | -0.4566 | -1.2439 |
| body_part=other | -0.0300 | -0.5348 |

## Validation design
Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match has shots in two folds, and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one of the four 2015/16 men's league seasons in the data, with the other three in dev. It tests transfer to an unseen league in the same season and data spec, not across time or gender.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.2858 | 0.2860 ± 0.0031 | 0.2709 |
| Brier score | 0.0818 | 0.0818 ± 0.0008 | 0.0764 |
| ROC AUC | 0.7657 | 0.7650 ± 0.0053 | 0.7570 |
| Log-loss skill | 0.1409 | 0.1402 | 0.1263 |
| Brier skill | 0.1185 | 0.1181 | 0.0959 |
| Calibration intercept | -0.0000 | -0.0004 | -0.0726 |
| Calibration slope | 1.0000 | 0.9973 | 0.9522 |
| Σ xG | 9302.0000 | 9303.9093 | 967.5904 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xG − goals) / goals | 0.00% | 0.02% | 5.98% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 2.12% (max 2.89%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR distance + angle + body part | +0.0012 [+0.0008, +0.0016] | +0.0012 [+0.0003, +0.0021] |
| over StatsBomb xG | -0.0201 [-0.0215, -0.0188] | -0.0189 [-0.0231, -0.0148] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 8,990 | 2.35% | 2.38% |
| 1 | 8,990 | 2.58% | 2.48% |
| 2 | 8,990 | 3.16% | 3.40% |
| 3 | 8,989 | 4.31% | 4.23% |
| 4 | 8,990 | 5.91% | 5.06% |
| 5 | 8,990 | 7.84% | 7.12% |
| 6 | 8,989 | 10.34% | 10.99% |
| 7 | 8,990 | 13.72% | 14.12% |
| 8 | 8,990 | 18.37% | 19.35% |
| 9 | 8,989 | 34.93% | 34.34% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 38,218 | 3,634 | 9.51% | 3653.4752 | 1.0054 | 0.2673 | 0.0754 | 0.1491 | 1.0523 |
| family = team-selected | 17,769 | 2,115 | 11.90% | 1795.7269 | 0.8490 | 0.3105 | 0.0909 | 0.1493 | 1.0783 |
| gender = female | 33,910 | 3,553 | 10.48% | 3854.7072 | 1.0849 | 0.2943 | 0.0843 | 0.1228 | 0.9183 |
| gender = male | 55,987 | 5,749 | 10.27% | 5449.2021 | 0.9479 | 0.2810 | 0.0803 | 0.1509 | 1.0623 |
| body_part = right_foot | 47,459 | 4,784 | 10.08% | 4785.0489 | 1.0002 | 0.2767 | 0.0791 | 0.1534 | 0.9805 |
| body_part = left_foot | 27,727 | 2,829 | 10.20% | 2829.2912 | 1.0001 | 0.2830 | 0.0807 | 0.1410 | 0.9776 |
| body_part = head | 14,428 | 1,609 | 11.15% | 1609.3674 | 1.0002 | 0.3181 | 0.0915 | 0.0903 | 1.1593 |
| body_part = other | 283 | 80 | 28.27% | 80.2019 | 1.0025 | 0.5016 | 0.1621 | 0.1576 | 1.1363 |

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Base rate | 0.3327 | 0.0928 | 0.5000 | n/a | 0.3101 | 0.0845 | 0.5000 | n/a |
| LR distance + angle | 0.2961 | 0.0844 | 0.7388 | 1.0001 | 0.2788 | 0.0783 | 0.7351 | 0.9648 |
| LR distance + angle + body part | 0.2872 | 0.0820 | 0.7640 | 0.9996 | 0.2721 | 0.0766 | 0.7556 | 0.9650 |
| LR splines + body part | 0.2860 | 0.0818 | 0.7650 | 0.9973 | 0.2709 | 0.0764 | 0.7570 | 0.9522 |
| LR splines + context | 0.2740 | 0.0783 | 0.7901 | 0.9957 | 0.2595 | 0.0728 | 0.7798 | 0.9711 |
| LR splines + context + freeze frames | 0.2673 | 0.0765 | 0.8046 | 0.9954 | 0.2547 | 0.0716 | 0.7915 | 0.9664 |
| Monotone GBM + Platt, trained without team-selected seasons | 0.2641 | 0.0755 | 0.8099 | 1.0086 | 0.2533 | 0.0711 | 0.7943 | 0.9483 |
| StatsBomb xG | 0.2659 | 0.0757 | 0.8072 | 1.0166 | 0.2520 | 0.0704 | 0.7963 | 0.9916 |

StatsBomb xG is a reference ceiling, not a like-for-like out-of-sample competitor: StatsBomb fits it on its own data, which likely includes these matches, and it uses freeze-frame and other inputs.

## Limitations & known biases
- Geometry and body part only: no context, defenders or keeper.
- Men and women are pooled: adding `gender` gives no gain with a CV interval above 0 (see the gain table of `xg-lr-context`); the Segments table gives each one's calibration slope.
- Direct corners are excluded (logged on outcome); penalties are a separate model (M1.5).
- The hold-out (Premier League 2015/2016) has no women's shots, so women are judged on the pooled out-of-fold predictions only.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS3): first version.
- v2 (MS3, M1.4): the benchmark table lists the GBM and calibration slopes; the overview names the headline model (v1's 'best model' sentence held until the GBM was added).
- v3 (MS3 assessment): the hold-out drop-one and women's lines are computed; the family = women segment row (the gender = female row) is printed once.
