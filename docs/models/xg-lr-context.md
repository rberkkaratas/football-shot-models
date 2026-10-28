# Model card: LR splines + context

## Overview
- **Name / version:** `xg-lr-context` v3
- **Task:** binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time included; direct free kicks and corners from open play or set pieces included; direct corners, own goals and shoot-outs excluded)
- **Status:** experimental
- M1.2 step S2: S1 plus the context columns. Not the headline model; that is Monotone GBM + Platt.

## Intended use
A transparent, reproducible xG model on StatsBomb open data, built one feature group at a time so each group's gain is measured on the same folds. For research and for comparing with StatsBomb's own xG; not for rating players or teams without that comparison, not for penalties, and not for other providers' data (it needs StatsBomb freeze frames and annotations) without re-fitting.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `067ddf5342cfd608045a15189a0b43008664d23f4c5ac4fab5ac403468e57ee5`
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
| `phase` | Indicators for fast break, corner, free kick, throw in, direct free kick; reference open play. Kick-offs join open play; direct free kicks wide of the box (y < 13.84 or > 54.16) join free kick (they are logged on outcome) | `situation`, `y` |
| `assist_type` | Indicators for the assisting pass: pass, cross, through ball, cutback; reference none | `assist_type` |
| `contact` | Indicators for volley, half volley, backheel, diving header, overhead kick; reference normal. Lob is merged into normal (it is read from the ball's flight) | `technique` |
| `open_goal` | 1 when StatsBomb marks the goal as open (an annotator judgement) | `open_goal` |
| `one_on_one` | 1 when StatsBomb marks a one-on-one with the keeper (an annotator judgement) | `one_on_one` |

## Model
`make_pipeline(StandardScaler(), LogisticRegression(C=inf, solver='newton-cholesky', tol=1e-10, max_iter=1000))`: unpenalised logistic regression, no class weights, no calibration step. The scaler is fitted on the training rows of each fit; a singular Hessian is an error, never a silent switch to another solver. scikit-learn 1.9.1.

Splines are restricted cubic splines with fixed knots (rounded quantiles of the shots, no labels): distance 6, 11.5, 17, 22.5, 30 m, angle 0.2, 0.27, 0.35, 0.5, 1 rad.

Coefficients of the fit on all of dev. Natural units are log-odds per unit of the feature (metre, radian, spline term) or vs the reference level at the same spot.

| Term | Standardised | Natural units |
|---|---|---|
| intercept | -2.6077 | -1.0319 |
| distance_m | -1.0051 | -0.1284 |
| distance_m_rcs1 | 2.1017 | 0.2499 |
| distance_m_rcs2 | -2.5986 | -0.6862 |
| distance_m_rcs3 | 1.0173 | 0.8905 |
| angle_rad | -0.2059 | -0.7451 |
| angle_rad_rcs1 | 47.8135 | 234.9259 |
| angle_rad_rcs2 | -74.5728 | -541.0682 |
| angle_rad_rcs3 | 27.5058 | 357.5046 |
| body_part=left_foot | -0.0147 | -0.0319 |
| body_part=head | -0.4805 | -1.3090 |
| body_part=other | -0.0471 | -0.8403 |
| phase=fast_break | 0.0215 | 0.1014 |
| phase=corner | -0.1960 | -0.5297 |
| phase=free_kick | -0.0341 | -0.0936 |
| phase=throw_in | -0.0476 | -0.1231 |
| phase=direct_free_kick | 0.1354 | 0.6841 |
| assist_type=pass | -0.0458 | -0.0916 |
| assist_type=cross | -0.0670 | -0.1987 |
| assist_type=through_ball | 0.1695 | 0.8817 |
| assist_type=cutback | -0.0392 | -0.2836 |
| contact=volley | -0.1440 | -0.5935 |
| contact=half_volley | -0.1904 | -0.5550 |
| contact=backheel | -0.1053 | -1.6440 |
| contact=diving_header | 0.0313 | 0.5500 |
| contact=overhead_kick | -0.0757 | -1.1042 |
| open_goal | 0.1338 | 1.3116 |
| one_on_one | 0.1125 | 0.5184 |

## Validation design
Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match has shots in two folds, and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one of the four 2015/16 men's league seasons in the data, with the other three in dev. It tests transfer to an unseen league in the same season and data spec, not across time or gender.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.2736 | 0.2740 ± 0.0031 | 0.2595 |
| Brier score | 0.0782 | 0.0783 ± 0.0009 | 0.0728 |
| ROC AUC | 0.7909 | 0.7901 ± 0.0054 | 0.7798 |
| Log-loss skill | 0.1775 | 0.1764 | 0.1629 |
| Brier skill | 0.1572 | 0.1563 | 0.1378 |
| Calibration intercept | -0.0000 | -0.0002 | -0.0485 |
| Calibration slope | 1.0000 | 0.9957 | 0.9711 |
| Σ xG | 9302.0000 | 9302.5368 | 947.7290 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xG − goals) / goals | 0.00% | 0.01% | 3.80% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 1.67% (max 2.49%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR splines + body part | +0.0120 [+0.0109, +0.0131] | +0.0113 [+0.0081, +0.0146] |
| over LR distance + angle + body part | +0.0132 [+0.0121, +0.0143] | +0.0126 [+0.0092, +0.0159] |
| over StatsBomb xG | -0.0081 [-0.0092, -0.0069] | -0.0075 [-0.0108, -0.0043] |
| `phase`: dropped | +0.0014 [+0.0011, +0.0018] | +0.0011 [+0.0000, +0.0023] |
| `assist_type`: dropped | +0.0025 [+0.0020, +0.0031] | +0.0022 [+0.0007, +0.0036] |
| `contact`: dropped | +0.0021 [+0.0017, +0.0026] | +0.0022 [+0.0008, +0.0036] |
| `open_goal`: dropped | +0.0014 [+0.0010, +0.0017] | +0.0011 [+0.0001, +0.0021] |
| `one_on_one`: dropped | +0.0007 [+0.0005, +0.0010] | +0.0010 [+0.0002, +0.0018] |
| `first_time`: added (not in the model) | +0.0000 [-0.0001, +0.0001] | +0.0001 [-0.0001, +0.0003] |
| `gender`: added (not in the model) | +0.0001 [-0.0000, +0.0002] | -0.0003 [-0.0006, +0.0000] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 8,990 | 1.69% | 1.90% |
| 1 | 8,990 | 2.36% | 2.35% |
| 2 | 8,990 | 2.97% | 2.79% |
| 3 | 8,989 | 4.05% | 3.99% |
| 4 | 8,990 | 5.28% | 4.92% |
| 5 | 8,990 | 6.97% | 6.63% |
| 6 | 8,989 | 9.34% | 9.01% |
| 7 | 8,990 | 12.65% | 12.55% |
| 8 | 8,990 | 18.37% | 19.92% |
| 9 | 8,989 | 39.79% | 39.41% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 38,218 | 3,634 | 9.51% | 3591.6813 | 0.9884 | 0.2569 | 0.0724 | 0.1823 | 1.0297 |
| family = team-selected | 17,769 | 2,115 | 11.90% | 1956.8070 | 0.9252 | 0.2961 | 0.0865 | 0.1887 | 1.0108 |
| gender = female | 33,910 | 3,553 | 10.48% | 3754.0484 | 1.0566 | 0.2817 | 0.0805 | 0.1604 | 0.9541 |
| gender = male | 55,987 | 5,749 | 10.27% | 5548.4883 | 0.9651 | 0.2693 | 0.0769 | 0.1862 | 1.0255 |
| body_part = right_foot | 47,459 | 4,784 | 10.08% | 4784.8379 | 1.0002 | 0.2630 | 0.0749 | 0.1954 | 0.9987 |
| body_part = left_foot | 27,727 | 2,829 | 10.20% | 2828.2942 | 0.9998 | 0.2714 | 0.0772 | 0.1764 | 0.9752 |
| body_part = head | 14,428 | 1,609 | 11.15% | 1609.3439 | 1.0002 | 0.3111 | 0.0897 | 0.1103 | 1.0386 |
| body_part = other | 283 | 80 | 28.27% | 80.0608 | 1.0008 | 0.4836 | 0.1560 | 0.1880 | 1.0016 |

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Base rate | 0.3327 | 0.0928 | 0.5000 | n/a | 0.3101 | 0.0845 | 0.5000 | n/a |
| LR distance + angle | 0.2961 | 0.0844 | 0.7388 | 1.0001 | 0.2788 | 0.0783 | 0.7351 | 0.9648 |
| LR distance + angle + body part | 0.2872 | 0.0820 | 0.7640 | 0.9996 | 0.2721 | 0.0766 | 0.7556 | 0.9650 |
| LR splines + body part | 0.2860 | 0.0818 | 0.7650 | 0.9973 | 0.2709 | 0.0764 | 0.7570 | 0.9522 |
| LR splines + context | 0.2740 | 0.0783 | 0.7901 | 0.9957 | 0.2595 | 0.0728 | 0.7798 | 0.9711 |
| LR splines + context + freeze frames | 0.2673 | 0.0765 | 0.8046 | 0.9954 | 0.2547 | 0.0716 | 0.7915 | 0.9664 |
| Monotone GBM + Platt | 0.2638 | 0.0754 | 0.8107 | 1.0097 | 0.2533 | 0.0711 | 0.7949 | 0.9526 |
| StatsBomb xG | 0.2659 | 0.0757 | 0.8072 | 1.0166 | 0.2520 | 0.0704 | 0.7963 | 0.9916 |

StatsBomb xG is a reference ceiling, not a like-for-like out-of-sample competitor: StatsBomb fits it on its own data, which likely includes these matches, and it uses freeze-frame and other inputs.

## Limitations & known biases
- Wide direct free kicks (y < 13.84 or > 54.16, n 378 in dev): goal rate 0.074, this model's mean xG 0.025, StatsBomb 0.008. They are logged as shots mostly when they threaten goal and join `phase = free_kick`, so the model rates them low.
- `open_goal` and `one_on_one` are StatsBomb annotator judgements; other providers record them differently or not at all.
- The context columns were selected on CV among seven candidates, a small optimism; their drop-one rows are in the gain table of `xg-lr-context`.
- Lob is merged into normal contact: it is labelled from the ball's flight, after the strike.
- No score state: the game situation enters only through `phase` and the freeze frame.
- Men and women are pooled: adding `gender` gives no gain with a CV interval above 0 (see the gain table of `xg-lr-context`); the Segments table gives each one's calibration slope.
- Direct corners are excluded (logged on outcome); penalties are a separate model (M1.5).
- On the hold-out every drop-one interval in this card's gain table lies above 0.
- The hold-out (Premier League 2015/2016) has no women's shots, so women are judged on the pooled out-of-fold predictions only.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS3): first version.
- v2 (MS3, M1.4): the benchmark table lists the GBM and calibration slopes; the overview names the headline model (v1's 'best model' sentence held until the GBM was added).
- v3 (MS3 assessment): the hold-out drop-one and women's lines are computed; the family = women segment row (the gender = female row) is printed once.
