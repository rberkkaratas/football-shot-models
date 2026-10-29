# Model card: Monotone GBM + Platt, trained without team-selected seasons

## Overview
- **Name / version:** `xg-gbm` v2
- **Task:** binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time included; direct free kicks and corners from open play or set pieces included; direct corners, own goals and shoot-outs excluded)
- **Status:** experimental
- M1.4: monotone gradient boosting on S3's inputs without spline terms, tuned and Platt-calibrated inside each fit by grouped CV. This is the headline model: its CV log-loss gain over LR splines + context + freeze frames is +0.0032 [+0.0025, +0.0040], a 95% interval that lies above 0; on the hold-out it is +0.0015 [-0.0007, +0.0035], an interval that includes 0. Against StatsBomb xG: CV gain +0.0018 [+0.0008, +0.0029], an interval that lies above 0; hold-out gain -0.0013 [-0.0041, +0.0017], an interval that includes 0. On the hold-out it predicts 963 goals for 913 scored (+5.45%); StatsBomb xG predicts 900 (-1.41%).

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
| `angle_rad` | Opening angle between the posts seen from the shot location, radians | `angle_rad` (`pitch.opening_angle`) |
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

Monotone constraints of the fitted model (every other column is free):

- distance_m: non-increasing — further from goal at the same opening angle never helps
- angle_rad: non-decreasing — a wider opening between the posts never hurts
- opponents_in_cone: non-increasing — another outfield body between ball and goal never helps
- nearest_opponent_m: non-decreasing — a closer outfield marker never helps (outfield opponents only, capped at 10 m, also the value with no marker in the frame)

Each constraint holds along one column with every other column held fixed: at a fixed opening angle, more distance is a move along an arc through both posts.

## Model
`HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1, min_samples_leaf=100, early_stopping=False, random_state=0)`, `monotonic_cst` placed by column name, no class weights. scikit-learn 1.9.1.

Tuning inside each fit: 4-fold `GroupKFold` by match on its training rows, `max_leaf_nodes` in (8, 16, 32) and 50, 100, ..., 600 iterations (staged scores of one fit per leaf count); the lowest inner out-of-fold log loss wins, ties to fewer leaves, then fewer iterations. Platt scaling: an unpenalised logistic regression of the goal on the chosen configuration's inner out-of-fold log-odds, applied to the refit on all training rows; a slope <= 0 is refused.

Tuning and calibration see only the rows of the fit they belong to; each outer test fold is scored by a model that never saw it.

Settings of the fit on all of dev:

| Setting | Value |
|---|---|
| max_leaf_nodes | 16 |
| iterations | 200 |
| Platt intercept | +0.0021 |
| Platt slope | 0.9995 |

Calibration is close to identity (Platt slope 0.999, intercept +0.002): the boosted model is already calibrated and Platt is a safeguard.

## Validation design
Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match has shots in two folds, and every model uses the same folds. **Validation** is the mean over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one of the four 2015/16 men's league seasons in the data, with the other three in dev. It tests transfer to an unseen league in the same season and data spec, not across time or gender.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.2572 | 0.2641 ± 0.0035 | 0.2533 |
| Brier score | 0.0736 | 0.0755 ± 0.0010 | 0.0711 |
| ROC AUC | 0.8235 | 0.8099 ± 0.0058 | 0.7943 |
| Log-loss skill | 0.2269 | 0.2061 | 0.1831 |
| Brier skill | 0.2062 | 0.1862 | 0.1587 |
| Calibration intercept | 0.0159 | 0.0179 | -0.0714 |
| Calibration slope | 1.0654 | 1.0086 | 0.9483 |
| Σ xG | 9194.5581 | 9180.8317 | 962.7394 |
| Goals | 9,302 | 9,302 | 913 |
| abs(Σ xG − goals) / goals | 1.16% | 1.30% | 5.45% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 1.92% (max 3.84%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over LR splines + context + freeze frames | +0.0032 [+0.0025, +0.0040] | +0.0015 [-0.0007, +0.0035] |
| over LR distance + angle + body part | +0.0231 [+0.0216, +0.0246] | +0.0188 [+0.0144, +0.0235] |
| over StatsBomb xG | +0.0018 [+0.0008, +0.0029] | -0.0013 [-0.0041, +0.0017] |

### Milestone checks (MS3)

Thresholds re-baselined by the PM: targets are relative to the same shots.

| Target | Measured | Result |
|---|---|---|
| CV log-loss gain over S0 with a 95% interval above 0 (LR distance + angle + body part 0.2872 on these folds; the MS2 best baseline 0.2880 is the MS2 population) | Monotone GBM + Platt, trained without team-selected seasons: +0.0231 [+0.0216, +0.0246] | met |
| CV log loss not more than 0.006 worse than StatsBomb xG (StatsBomb CV 0.2659 on these shots; the PM brief's 0.2674 is the MS2 population, with direct corners) | Monotone GBM + Platt, trained without team-selected seasons 0.2641, StatsBomb 0.2659: gap -0.0018 | met |
| calibration slope in [0.9, 1.1], CV fold mean | 1.0086 | met |
| calibration slope in [0.9, 1.1], hold-out | 0.9483 | met |
| calibration slope in [0.9, 1.1], men, pooled out of fold | 1.0235 | met |
| calibration slope in [0.9, 1.1], women, pooled out of fold | 0.9828 | met |
| calibration slope in [0.9, 1.1], men, hold-out | 0.9483 | met |
| calibration slope in [0.9, 1.1], women, hold-out | n/a (0 women's shots in the hold-out) | n/a |
| Σ xG error, fold mean <= 3% | 1.92% (pooled 1.30%) | met |
| Σ xG error, fold max <= 5% | 3.84% (pooled 1.30%) | met |
| GBM CV gain over LR splines + context + freeze frames, 95% interval above 0 (the GBM is headline iff met) | CV +0.0032 [+0.0025, +0.0040]; hold-out +0.0015 [-0.0007, +0.0035], an interval that includes 0 | met |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 8,990 | 1.31% | 1.22% |
| 1 | 8,990 | 2.10% | 1.95% |
| 2 | 8,990 | 2.84% | 2.80% |
| 3 | 8,989 | 3.63% | 3.59% |
| 4 | 8,990 | 4.59% | 4.48% |
| 5 | 8,990 | 6.00% | 6.26% |
| 6 | 8,989 | 8.09% | 8.53% |
| 7 | 8,990 | 11.66% | 12.09% |
| 8 | 8,990 | 19.22% | 19.35% |
| 9 | 8,989 | 42.70% | 43.19% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| family = league or tournament | 38,218 | 3,634 | 9.51% | 3677.8482 | 1.0121 | 0.2471 | 0.0699 | 0.2135 | 1.0329 |
| family = team-selected | 17,769 | 2,115 | 11.90% | 1990.8889 | 0.9413 | 0.2869 | 0.0837 | 0.2139 | 1.0005 |
| gender = female | 33,910 | 3,553 | 10.48% | 3512.0946 | 0.9885 | 0.2714 | 0.0776 | 0.1911 | 0.9828 |
| gender = male | 55,987 | 5,749 | 10.27% | 5668.7371 | 0.9860 | 0.2597 | 0.0743 | 0.2152 | 1.0235 |
| body_part = right_foot | 47,459 | 4,784 | 10.08% | 4651.2110 | 0.9722 | 0.2527 | 0.0719 | 0.2267 | 1.0202 |
| body_part = left_foot | 27,727 | 2,829 | 10.20% | 2781.2801 | 0.9831 | 0.2609 | 0.0744 | 0.2081 | 1.0056 |
| body_part = head | 14,428 | 1,609 | 11.15% | 1666.0224 | 1.0354 | 0.3035 | 0.0879 | 0.1320 | 0.9647 |
| body_part = other | 283 | 80 | 28.27% | 82.3182 | 1.0290 | 0.4679 | 0.1523 | 0.2142 | 1.0677 |

### Hold-out level (PR1)

The pooled GBM's hold-out Σ xG is 974.8 for 913 goals (+61.8, Poisson-binomial SD 26.9, z +2.30). Decomposition by segment (Σ xG − goals, z = gap / √Σ p(1−p)); the dev rows are out of fold:

| Split | Segment | Shots | Goals | Pooled Σ xG − goals (z) | Headline Σ xG − goals (z) |
|---|---|---|---|---|---|
| hold-out | phase = corner | 1,668 | 144 | +14.4 (+1.34) | +14.1 (+1.31) |
| hold-out | phase = direct_free_kick | 366 | 19 | +4.0 (+0.88) | +2.0 (+0.45) |
| hold-out | phase = fast_break | 464 | 68 | +1.2 (+0.17) | +0.6 (+0.08) |
| hold-out | phase = free_kick | 1,464 | 136 | +7.6 (+0.74) | +4.8 (+0.47) |
| hold-out | phase = open_play | 4,001 | 373 | +34.7 (+2.01) | +28.8 (+1.67) |
| hold-out | phase = throw_in | 1,853 | 173 | −0.2 (−0.02) | −0.6 (−0.05) |
| hold-out | body part = head | 1,608 | 180 | +11.4 (+0.95) | +13.4 (+1.10) |
| hold-out | body part = left_foot | 2,897 | 249 | +26.2 (+1.83) | +20.5 (+1.45) |
| hold-out | body part = other | 17 | 9 | −2.4 (−1.58) | −2.7 (−1.79) |
| hold-out | body part = right_foot | 5,294 | 475 | +26.6 (+1.38) | +18.6 (+0.98) |
| hold-out | distance = <= 6 m | 411 | 134 | +35.4 (+3.98) | +34.7 (+3.90) |
| hold-out | distance = (6, 11] m | 1,919 | 314 | −2.5 (−0.17) | −1.7 (−0.11) |
| hold-out | distance = (11, 16] m | 2,017 | 247 | +1.5 (+0.11) | −3.1 (−0.22) |
| hold-out | distance = (16, 20] m | 1,685 | 110 | +10.0 (+0.98) | +4.6 (+0.45) |
| hold-out | distance = (20, 25] m | 2,156 | 72 | +16.3 (+1.79) | +14.4 (+1.59) |
| hold-out | distance = > 25 m | 1,628 | 36 | +1.1 (+0.18) | +0.9 (+0.14) |
| hold-out | assist type = cross | 1,260 | 172 | +19.1 (+1.64) | +18.1 (+1.55) |
| hold-out | assist type = cutback | 181 | 23 | +5.9 (+1.28) | +5.6 (+1.22) |
| hold-out | assist type = none | 2,641 | 244 | +10.5 (+0.78) | +4.2 (+0.32) |
| hold-out | assist type = pass | 5,401 | 376 | +22.1 (+1.23) | +18.1 (+1.01) |
| hold-out | assist type = through_ball | 333 | 98 | +4.1 (+0.52) | +3.8 (+0.48) |
| hold-out | shot freeze frame: one level only (n = 9,816) | | | | |
| hold-out | 360 frame: one level only (n = 9,816) | | | | |
| dev OOF | family = league or tournament | 38,218 | 3,634 | +91.2 (+1.74) | +43.8 (+0.84) |
| dev OOF | family = team-selected | 17,769 | 2,115 | −100.2 (−2.65) | −124.1 (−3.30) |
| dev OOF | family = women | 33,910 | 3,553 | +12.5 (+0.24) | −40.9 (−0.80) |
| dev OOF | shot freeze frame: one level only (n = 89,897) | | | | |
| dev OOF | 360 frame = with | 10,228 | 1,038 | +40.8 (+1.45) | +26.9 (+0.96) |
| dev OOF | 360 frame = without | 79,669 | 8,264 | −37.4 (−0.48) | −148.0 (−1.92) |

At most two candidate fixes, one refit each. A fix is adopted iff its hold-out Σ xG is within 2 SD of goals and every MS3 check the pooled GBM meets stays met; the first such fix in this order is adopted.

| Model | Hold-out Σ xG | Goals | Gap | SD | z | Within | MS3 checks met | CV log loss | Hold-out log loss | Fold Σ xG error mean | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Monotone GBM + Platt, pooled training | 974.8 | 913 | +61.8 | 26.9 | +2.30 | no | 10 of 10 | 0.2638 | 0.2533 | 1.71% | 2.92% |
| Monotone GBM + Platt, team-selected indicator (0 at prediction) | 968.7 | 913 | +55.7 | 26.8 | +2.08 | no | 10 of 10 | 0.2638 | 0.2528 | 1.79% | 3.44% |
| Monotone GBM + Platt, trained without team-selected seasons (adopted) | 962.7 | 913 | +49.7 | 26.7 | +1.86 | yes | 10 of 10 | 0.2641 | 0.2533 | 1.92% | 3.84% |

- Adopted: Monotone GBM + Platt, trained without team-selected seasons. The hold-out chose it, so its hold-out level is no longer an independent check. It removes 19% of the pooled hold-out excess (+61.8 → +49.7).
- Largest headline hold-out segment gap (segments of 100+ shots): distance = <= 6 m, +34.7 on 411 shots (z +3.90).
- Dev out of fold, league or tournament: pooled +91.2 (+2.5% of 3,634 goals; Σ xG 3725.2); headline +43.8 (+1.2% of 3,634 goals; Σ xG 3677.8); this is support the hold-out did not choose.
- Dev out of fold, team-selected, under pooled training: −100.2 (−4.7% of 2,115 goals; Σ xG 2014.8).
- The LR steps are trained on all of dev (pooled) and the headline on dev without team-selected seasons, so the GBM-over-LR gain mixes training population with model class.

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
- The hold-out gain over LR splines + context + freeze frames is +0.0015 [-0.0007, +0.0035], an interval that includes 0; the headline rule is decided on CV, and the hold-out (Premier League 2015/2016, 380 matches) is a smaller test.
- The CV population mixes women's and team-selected seasons that convert more than a men's league (dev goal rate 9.51% league, 11.90% team-selected, 10.48% women; hold-out 9.30%). The hold-out over-prediction (+5.45%) and the CV gain over StatsBomb not carrying over are reported, not explained: this card does not test whether the mix causes them.
- Wide direct free kicks (y < 13.84 or > 54.16, n 378 in dev): goal rate 0.074, this model's mean xG 0.053, StatsBomb 0.008. They are logged as shots mostly when they threaten goal and join `phase = free_kick`, so the model rates them low.
- Freeze frames hold the players the broadcast camera shows, so the counts depend on the camera; the features are local to the shot (cone, nearest opponent, keeper) rather than whole-frame counts.
- Keeper distances are capped at 20 m, where few shots are (a keeper far out of goal); `gk_absent` marks a frame without the opponent keeper.
- `open_goal` and `one_on_one` are StatsBomb annotator judgements; other providers record them differently or not at all.
- The context columns were selected on CV among seven candidates, a small optimism; their drop-one rows are in the gain table of `xg-lr-context`.
- Lob is merged into normal contact: it is labelled from the ball's flight, after the strike.
- No score state: the game situation enters only through `phase` and the freeze frame.
- Men and women are pooled: adding `gender` gives no gain with a CV interval above 0 (see the gain table of `xg-lr-context`); the Segments table gives each one's calibration slope.
- Direct corners are excluded (logged on outcome); penalties are a separate model (M1.5).
- The hold-out (Premier League 2015/2016) has no women's shots, so women are judged on the pooled out-of-fold predictions only.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS3, M1.4): first version.
- v2 (MS3 assessment): the benchmark sentence names the benchmark; the training-mix line makes no causal claim; the hold-out women's line is computed.
