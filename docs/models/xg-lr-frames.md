# Model card: LR splines + context + freeze frames

## Overview
- **Name / version:** `xg-lr-frames` v3
- **Task:** binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time included; direct free kicks and corners from open play or set pieces included; direct corners, own goals and shoot-outs excluded)
- **Status:** experimental
- M1.3 step S3: S2 plus the freeze-frame features. Not the headline model; that is Monotone GBM + Platt, trained without team-selected seasons.

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
| `phase` | Indicators for fast break, corner, free kick, throw in, direct free kick; reference open play. Kick-offs join open play; direct free kicks wide of the box (y < 13.84 or > 54.16) join free kick (they are logged on outcome) | `situation`, `y` |
| `assist_type` | Indicators for the assisting pass: pass, cross, through ball, cutback; reference none | `assist_type` |
| `contact` | Indicators for volley, half volley, backheel, diving header, overhead kick; reference normal. Lob is merged into normal (it is read from the ball's flight) | `technique` |
| `open_goal` | 1 when StatsBomb marks the goal as open (an annotator judgement) | `open_goal` |
| `one_on_one` | 1 when StatsBomb marks a one-on-one with the keeper (an annotator judgement) | `one_on_one` |
| `opponents_in_cone` | Outfield opponents in the triangle shot, right post, left post (edges included) | freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y` |
| `teammates_in_cone` | Teammates, own keeper included, in that triangle | freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y` |
| `gk_in_cone` | 1 when the opponent keeper is in that triangle; 0 when absent | freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y` |
| `nearest_opponent_m` | Metres from the shot to the nearest outfield opponent, capped at 10 (also when none is in the frame) | freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y` |
| `gk_absent` | 1 when the frame has no opponent keeper (out of the recorded area or far from his goal at the strike) | freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y` |
| `gk_to_goal_m` | Opponent keeper to the goal centre, metres, capped at 20; 0 with `gk_absent` | freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y` |
| `gk_to_shot_m` | Opponent keeper to the shot location, metres, capped at 20; 0 with `gk_absent` | freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y` |
| `gk_off_line_m` | Opponent keeper's distance from the bisector of the shot's angle (the line that splits the goal mouth in the ratio of the shot's distances to the posts), metres, capped at 20; 0 with `gk_absent` | freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y` |

## Model
`make_pipeline(StandardScaler(), LogisticRegression(C=inf, solver='newton-cholesky', tol=1e-10, max_iter=1000))`: unpenalised logistic regression, no class weights, no calibration step. The scaler is fitted on the training rows of each fit; a singular Hessian is an error, never a silent switch to another solver. scikit-learn 1.9.1.

Splines are restricted cubic splines with fixed knots (rounded quantiles of the shots, no labels): distance 6, 11.5, 17, 22.5, 30 m, angle 0.2, 0.27, 0.35, 0.5, 1 rad.

Freeze-frame distances: nearest opponent capped at 10 m, keeper distances at 20 m.

Coefficients of the fit on all of dev. Natural units are log-odds per unit of the feature (metre, radian, spline term) or vs the reference level at the same spot.

| Term | Standardised | Natural units |
|---|---|---|
| intercept | -2.6715 | -2.5405 |
| distance_m | -0.8883 | -0.1135 |
| distance_m_rcs1 | 2.0186 | 0.2400 |
| distance_m_rcs2 | -2.6280 | -0.6940 |
| distance_m_rcs3 | 0.9737 | 0.8524 |
| angle_rad | 0.5089 | 1.8412 |
| angle_rad_rcs1 | 41.1616 | 202.2426 |
| angle_rad_rcs2 | -65.9490 | -478.4972 |
| angle_rad_rcs3 | 25.0133 | 325.1089 |
| body_part=left_foot | -0.0093 | -0.0202 |
| body_part=head | -0.4070 | -1.1088 |
| body_part=other | -0.0444 | -0.7932 |
| phase=fast_break | -0.0032 | -0.0150 |
| phase=corner | -0.0701 | -0.1895 |
| phase=free_kick | -0.0055 | -0.0151 |
| phase=throw_in | -0.0246 | -0.0637 |
| phase=direct_free_kick | 0.1673 | 0.8454 |
| assist_type=pass | -0.0441 | -0.0882 |
| assist_type=cross | -0.0603 | -0.1787 |
| assist_type=through_ball | 0.1016 | 0.5288 |
| assist_type=cutback | -0.0314 | -0.2270 |
| contact=volley | -0.1238 | -0.5102 |
| contact=half_volley | -0.1669 | -0.4865 |
| contact=backheel | -0.0989 | -1.5448 |
| contact=diving_header | 0.0242 | 0.4251 |
| contact=overhead_kick | -0.0637 | -0.9300 |
| open_goal | 0.0665 | 0.6519 |
| one_on_one | 0.0693 | 0.3192 |
| opponents_in_cone | -0.3347 | -0.3305 |
| teammates_in_cone | -0.0058 | -0.0124 |
| gk_in_cone | -0.0877 | -0.4394 |
| nearest_opponent_m | 0.2831 | 0.1421 |
| gk_absent | 0.0423 | 1.3607 |
| gk_to_goal_m | 0.2278 | 0.1038 |
| gk_to_shot_m | 0.1600 | 0.0269 |
| gk_off_line_m | -0.0107 | -0.0132 |

## Validation design
Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match has shots in two folds, and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one of the four 2015/16 men's league seasons in the data, with the other three in dev. It tests transfer to an unseen league in the same season and data spec, not across time or gender.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.2669 | 0.2673 ± 0.0035 | 0.2547 |
| Brier score | 0.0764 | 0.0765 ± 0.0010 | 0.0716 |
| ROC AUC | 0.8056 | 0.8046 ± 0.0064 | 0.7915 |
| Log-loss skill | 0.1978 | 0.1963 | 0.1784 |
| Brier skill | 0.1761 | 0.1749 | 0.1522 |
| Calibration intercept | -0.0000 | -0.0001 | -0.0670 |
| Calibration slope | 1.0000 | 0.9954 | 0.9664 |
| Σ xG | 9302.0000 | 9301.9964 | 960.2174 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xG − goals) / goals | 0.00% | 0.00% | 5.17% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 1.65% (max 3.17%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR splines + context | +0.0066 [+0.0058, +0.0074] | +0.0048 [+0.0029, +0.0067] |
| over LR distance + angle + body part | +0.0199 [+0.0185, +0.0212] | +0.0174 [+0.0135, +0.0212] |
| over StatsBomb xG | -0.0014 [-0.0024, -0.0005] | -0.0027 [-0.0053, -0.0002] |
| cone group (`opponents_in_cone`, `teammates_in_cone`, `gk_in_cone`): dropped | +0.0023 [+0.0018, +0.0027] | +0.0014 [+0.0001, +0.0026] |
| nearest group (`nearest_opponent_m`): dropped | +0.0016 [+0.0012, +0.0019] | +0.0020 [+0.0009, +0.0031] |
| keeper group (`gk_absent`, `gk_to_goal_m`, `gk_to_shot_m`, `gk_off_line_m`): dropped | +0.0010 [+0.0007, +0.0013] | +0.0005 [-0.0003, +0.0014] |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 8,990 | 1.27% | 1.48% |
| 1 | 8,990 | 2.07% | 1.95% |
| 2 | 8,990 | 2.84% | 3.01% |
| 3 | 8,989 | 3.79% | 3.40% |
| 4 | 8,990 | 5.02% | 4.86% |
| 5 | 8,990 | 6.64% | 5.83% |
| 6 | 8,989 | 8.86% | 8.86% |
| 7 | 8,990 | 12.24% | 12.31% |
| 8 | 8,990 | 18.66% | 19.94% |
| 9 | 8,989 | 42.10% | 41.83% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 38,218 | 3,634 | 9.51% | 3704.8380 | 1.0195 | 0.2502 | 0.0709 | 0.2037 | 1.0322 |
| family = team-selected | 17,769 | 2,115 | 11.90% | 2007.5293 | 0.9492 | 0.2897 | 0.0845 | 0.2064 | 0.9929 |
| gender = female | 33,910 | 3,553 | 10.48% | 3589.6291 | 1.0103 | 0.2750 | 0.0788 | 0.1802 | 0.9541 |
| gender = male | 55,987 | 5,749 | 10.27% | 5712.3673 | 0.9936 | 0.2627 | 0.0752 | 0.2062 | 1.0204 |
| body_part = right_foot | 47,459 | 4,784 | 10.08% | 4785.1125 | 1.0002 | 0.2561 | 0.0731 | 0.2165 | 0.9920 |
| body_part = left_foot | 27,727 | 2,829 | 10.20% | 2828.2353 | 0.9997 | 0.2643 | 0.0755 | 0.1980 | 0.9782 |
| body_part = head | 14,428 | 1,609 | 11.15% | 1608.8247 | 0.9999 | 0.3066 | 0.0885 | 0.1232 | 1.0527 |
| body_part = other | 283 | 80 | 28.27% | 79.8238 | 0.9978 | 0.4522 | 0.1458 | 0.2407 | 1.1934 |

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
- Wide direct free kicks (y < 13.84 or > 54.16, n 378 in dev): goal rate 0.074, this model's mean xG 0.032, StatsBomb 0.008. They are logged as shots mostly when they threaten goal and join `phase = free_kick`, so the model rates them low.
- Freeze frames hold the players the broadcast camera shows, so the counts depend on the camera; the features are local to the shot (cone, nearest opponent, keeper) rather than whole-frame counts.
- Keeper distances are capped at 20 m, where few shots are (a keeper far out of goal); `gk_absent` marks a frame without the opponent keeper.
- `open_goal` and `one_on_one` are StatsBomb annotator judgements; other providers record them differently or not at all.
- The context columns were selected on CV among seven candidates, a small optimism; their drop-one rows are in the gain table of `xg-lr-context`.
- Lob is merged into normal contact: it is labelled from the ball's flight, after the strike.
- No score state: the game situation enters only through `phase` and the freeze frame.
- Men and women are pooled: adding `gender` gives no gain with a CV interval above 0 (see the gain table of `xg-lr-context`); the Segments table gives each one's calibration slope.
- Direct corners are excluded (logged on outcome); penalties are a separate model (M1.5).
- On the hold-out the drop-one interval of keeper group (+0.0005 [-0.0003, +0.0014]) includes 0 (this card's gain table); the inclusion rule is decided on CV.
- The hold-out (Premier League 2015/2016) has no women's shots, so women are judged on the pooled out-of-fold predictions only.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS3): first version.
- v2 (MS3, M1.4): the benchmark table lists the GBM and calibration slopes; the overview names the headline model (v1's 'best model' sentence held until the GBM was added).
- v3 (MS3 assessment): the hold-out drop-one and women's lines are computed; the family = women segment row (the gender = female row) is printed once.
