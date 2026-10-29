# Model card: xG v1 (WhoScored / Opta events)

## Overview
- **Name / version:** `xg-v1` v1
- **Task:** binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time included; direct free kicks and corners from open play or set pieces included; direct corners, own goals and shoot-outs excluded)
- **Status:** experimental
- M1.6 / M1.7: the WhoScored xG model of record. Headline model: Monotone GBM + Platt. This is the headline model: its CV log-loss gain over LR splines + context is +0.0009 [+0.0005, +0.0013], a 95% interval that lies above 0; on the test it is +0.0013 [+0.0005, +0.0022], an interval that lies above 0.

## Intended use
xG for WhoScored / Opta shot events: P(goal) of a non-penalty shot from its location, body part, phase of play, assisting event and Opta's 1-on-1 tag. For research and opposition analysis on this archive, and for comparing with the legacy v0 on the same test shots; not for rating players or teams without that comparison, not for penalties (a separate probability, below), and not for other providers' data without re-fitting.

## What it is not
- Not a StatsBomb model: it is fitted and tested on WhoScored / Opta events only, and its numbers are not comparable with the StatsBomb cards.
- No freeze frames: it sees no keeper or defender positions at the strike.
- It does not read Opta big chance (q214); Legacy v0 (uses Opta big chance q214) and the labelled ablation do.
- Men's competitions only (96,828 shots, 0 women's): 10 in train, 10 in test; FIFA World Cup only in train; Belgian Pro League only in test.
- One train season (2025/2026, 2026: 3,215 matches) and the first 587 test matches (2026/2027); not a multi-season model.
- The v0 comparison is test-only: v0 was fitted on 48,882 shots before the first test match (checked), so it has no CV row.
- No provider xG benchmark: v0 is the benchmark.

## Data
- **Source(s):** private WhoScored / Opta event archive, source revision `84571265636c3c30fcfcf9b9b0b3bd1381f7488fc3d3870893d2539f8d7c3b23`; snapshot `whoscored-v1`, data digest `c394a94944cdc12ee4d6bdd77594c9156a51aa3731955905b46ec80f648bb63e`; v0 file SHA-256 `24c8f23d059b5dca…`
- **Competitions & seasons:** 20 competition-seasons; train 2025/2026, 2026, test 2026/2027
- **Counts:** train 80,664 shots (7,923 goals) in 3,215 matches / test 16,164 shots (1,614 goals) in 587 matches
- **Filters:** excluded shots, by first matching reason.

| Filter | Shots | Goals |
|---|---|---|
| own goal (Opta q28) | 330 | 330 |
| shoot-out (period 5) | 20 | 13 |
| penalty (in-game; the penalty probability below) | 1,063 | 834 |
| direct corner (Opta q263) | 13 | 7 |
| no freeze frame | n/a (no frames) | n/a |

7 kept shots whose assisting event (Opta q55) is unresolved (a null `assist_type`) are dropped for every model, before the split.

v0 covers 16,164 of 16,164 test shots in the v1 population (2 dropped for unresolved assist, q55; v0 scores 16,166 of 16,166 kept test shots).

## Target
`is_goal`: the ball went in. Own goals (Opta q28) are excluded, so every goal is the shooter's.

## Split
Train: season(s) 2025/2026, 2026; test: 2026/2027. Every train match precedes every test match (checked). Grouped 5-fold CV by `match_id` on train only (**Validation**); **Train** is the fit on all of train, scored in-sample; **Test** is the test season, scored once by that fit and never used to choose. Any tuning (the GBM's grid, Platt) runs inside each fit on its training rows.

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
| `body_part` | Indicators for left foot, head, other; reference right foot | Opta q20 right foot, q72 left foot, q15 head, q21 other |
| `phase` | Indicators for fast break, corner, free kick, throw in, direct free kick; reference open play. Direct free kicks wide of the box (y < 13.84 or > 54.16) join free kick | Opta pattern q22 open play, q23 fast break, q24 from free kick, q25 from corner, q26 direct free kick, q160 from throw in (`situation`), `y` |
| `assist` | Indicators for the assisting event: pass, cross, through ball; reference none. No cutback level: the archive has no pull-back qualifier | Opta q55 related event, its q2 cross and q4 through ball (`assist_type`) |
| `one_on_one` | Opta 1-on-1 tag (annotator) | Opta q89 |
| `big_chance` | Opta big chance, an analyst's label of the chance: the labelled ablation only | Opta q214 |

`gender` is left out: 0 of 96,828 shots are women's, so it is one value.

## Model
Monotone gradient boosting (`gbm.MonotoneGBM`, scikit-learn 1.9.1), tuned inside each fit by 4-fold `GroupKFold` by match on its training rows (`max_leaf_nodes` in (8, 16, 32); 50, 100, ..., 600 iterations) and Platt-scaled on that inner out-of-fold score; the test season never reaches the tuning.

Monotone constraints (every other column is free): `distance_m` non-increasing (further from goal at the same opening angle never helps); `angle_rad` non-decreasing (a wider opening between the posts never hurts).

| Setting of the fit on all of train | Value |
|---|---|
| max_leaf_nodes | 8 |
| iterations | 200 |
| Platt intercept | -0.0269 |
| Platt slope | 0.9857 |

The chosen leaf count (8) is the grid's smallest (8, 16, 32).

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.2701 | 0.2724 ± 0.0047 | 0.2771 |
| Brier score | 0.0766 | 0.0772 ± 0.0014 | 0.0788 |
| ROC AUC | 0.7793 | 0.7737 ± 0.0068 | 0.7702 |
| Log-loss skill | 0.1589 | 0.1519 | 0.1468 |
| Brier skill | 0.1348 | 0.1286 | 0.1236 |
| Calibration intercept | -0.0000 | -0.0006 | 0.0129 |
| Calibration slope | 1.0289 | 1.0054 | 0.9786 |
| Σ xG | 7923.0320 | 7925.4714 | 1597.7586 |
| Goals | 7,923 | 7,923 | 1,614 |
| abs(Σ xG − goals) / goals | 0.00% | 0.03% | 1.01% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 1.22% (max 2.43%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

## Models
Every candidate on the same folds and test shots; the headline in bold. v0 is scored on the 16,164 test shots it covers.

| Model | CV log loss | Test log loss | Test Brier | Test AUC | Test cal. slope | Test Σ xG error (signed) |
|---|---|---|---|---|---|---|
| Base rate | 0.3212 | 0.3248 | 0.0899 | 0.5000 | n/a | -1.63% |
| LR distance + angle | 0.2892 | 0.2971 | 0.0837 | 0.7100 | 0.9258 | -1.91% |
| LR distance + angle + body part | 0.2810 | 0.2870 | 0.0814 | 0.7470 | 0.9635 | -1.44% |
| LR splines + body part | 0.2803 | 0.2863 | 0.0812 | 0.7466 | 0.9629 | -1.44% |
| LR splines + context | 0.2733 | 0.2784 | 0.0791 | 0.7669 | 0.9697 | -0.91% |
| **Monotone GBM + Platt** | 0.2724 | 0.2771 | 0.0788 | 0.7702 | 0.9786 | -1.01% |
| LR splines + context + big chance (labelled ablation) | 0.2559 | 0.2611 | 0.0742 | 0.7974 | 0.9728 | +0.72% |
| Legacy v0 (uses Opta big chance q214) | n/a (test-only benchmark) | 0.2615 | 0.0743 | 0.7978 | 0.9794 | +0.90% |

## Gains
Gain = log loss of the reference minus the model's; positive means the model is better. 95% percentile intervals of 2,000 match resamples (seed 0).

| Comparison | CV gain [95% CI] | Test gain [95% CI] |
|---|---|---|
| Monotone GBM + Platt over LR distance + angle | +0.0168 [+0.0155, +0.0181] | +0.0200 [+0.0171, +0.0228] |
| LR splines + context over LR distance + angle | +0.0159 [+0.0147, +0.0172] | +0.0187 [+0.0157, +0.0215] |
| Monotone GBM + Platt over LR splines + context | +0.0009 [+0.0005, +0.0013] | +0.0013 [+0.0005, +0.0022] |
| LR splines + context + big chance (labelled ablation) over LR splines + context | +0.0174 [+0.0161, +0.0187] | +0.0173 [+0.0144, +0.0201] |

Test gain over Legacy v0 (uses Opta big chance q214) on the 16,164 covered test shots:

| Model | Test gain over v0 [95% CI] | Interval |
|---|---|---|
| Monotone GBM + Platt | -0.0156 [-0.0183, -0.0127] | lies below 0 |
| LR distance + angle | -0.0356 [-0.0394, -0.0314] | lies below 0 |
| LR splines + context | -0.0169 [-0.0197, -0.0139] | lies below 0 |
| LR splines + context + big chance (labelled ablation) | +0.0004 [-0.0002, +0.0010] | includes 0 |
| v0's inputs refitted on train without q214 (test only, counterfactual) | -0.0182 [-0.0211, -0.0154] | lies below 0 |
| Monotone GBM + Platt over v0's inputs refitted on train without q214 (test only, counterfactual reference in place of v0) | +0.0027 [+0.0013, +0.0039] | lies above 0 |

## Checks
### Milestone checks (MS4)

Targets from the MS4 brief, judged on the test season (2026/2027). The last two rows restate rules the run applies.

| Target | Measured | Result |
|---|---|---|
| Test log loss <= 0.275 (PM window about 0.26-0.27; LR distance + angle start line: test 0.2971, PM about 0.297) | Monotone GBM + Platt: 0.2771 | missed |
| At least as good as Legacy v0 (uses Opta big chance q214) on the same test shots (16,164 of 16,164 covered, 100.00%) | test gain over v0 -0.0156 [-0.0183, -0.0127], an interval that lies below 0 | missed |
| Calibration slope in [0.9, 1.1], test season | 0.9786 | met |
| GBM headline iff its CV gain over LR splines + context has a 95% interval above 0 | CV +0.0009 [+0.0005, +0.0013]; headline Monotone GBM + Platt | met |
| Big chance only in the labelled ablation, never in the headline | headline features: distance_m, angle_rad, body_part, phase, assist, one_on_one | met |

- the best LR (LR splines + context) plus q214 alone: test log loss 0.2611 (meets it); v0's inputs refitted on train without q214: test log loss 0.2797 (misses it). Both labelled counterfactual refits agree that q214 accounts for the gap: adding it alone meets the bound, and removing it from v0's inputs misses it.
- the best LR (LR splines + context) plus q214 alone: gain over v0 +0.0004 [-0.0002, +0.0010] (meets it); v0's inputs refitted on train without q214: gain over v0 -0.0182 [-0.0211, -0.0154] (misses it). Both labelled counterfactual refits agree that q214 accounts for the gap: adding it alone meets the bound, and removing it from v0's inputs misses it.

## Reliability
Ten equal-count bins of the headline model's test predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 1,617 | 1.98% | 1.98% |
| 1 | 1,616 | 2.68% | 3.09% |
| 2 | 1,617 | 3.34% | 2.72% |
| 3 | 1,616 | 4.19% | 4.08% |
| 4 | 1,616 | 5.23% | 5.63% |
| 5 | 1,617 | 6.65% | 6.74% |
| 6 | 1,616 | 8.63% | 8.60% |
| 7 | 1,617 | 11.92% | 12.37% |
| 8 | 1,616 | 17.63% | 19.00% |
| 9 | 1,616 | 36.61% | 35.64% |

## Test by competition
The headline model on each competition's test shots; intervals from 2,000 resamples of that competition's matches. A slope needs 1,000 shots.

| Competition | Shots | Goals | Σ xG | Σ xG error, signed [95% CI] | Cal. slope [95% CI] | v0 Σ xG error, signed [95% CI] |
|---|---|---|---|---|---|---|
| 1. Bundesliga | 1,110 | 131 | 119.4 | -8.84% [-20.42%, +7.01%] | 1.0258 [0.825, 1.241] | -9.79% [-20.34%, +4.28%] |
| Belgian Pro League | 1,771 | 163 | 177.1 | +8.62% [-5.40%, +25.88%] | 1.0336 [0.887, 1.194] | +7.12% [-6.10%, +22.51%] |
| Championship | 2,543 | 251 | 252.5 | +0.60% [-8.82%, +11.38%] | 0.9505 [0.814, 1.087] | +0.02% [-8.78%, +9.91%] |
| Eredivisie | 1,875 | 224 | 196.1 | -12.46% [-21.38%, -1.83%] | 1.0165 [0.871, 1.181] | -11.40% [-20.09%, -1.58%] |
| La Liga | 1,798 | 180 | 173.9 | -3.37% [-13.49%, +9.57%] | 1.0707 [0.924, 1.248] | -0.54% [-10.82%, +12.22%] |
| Ligue 1 | 1,238 | 109 | 127.8 | +17.26% [+1.56%, +38.33%] | 0.9759 [0.791, 1.175] | +22.63% [+6.64%, +43.57%] |
| Premier League | 1,382 | 128 | 136.0 | +6.28% [-7.97%, +24.62%] | 0.8260 [0.674, 1.005] | +8.59% [-5.80%, +27.27%] |
| Primeira Liga | 1,496 | 146 | 144.5 | -1.04% [-13.56%, +14.50%] | 0.9789 [0.841, 1.136] | +3.63% [-9.87%, +20.31%] |
| Serie A | 1,452 | 141 | 141.2 | +0.13% [-13.82%, +15.82%] | 0.8254 [0.661, 1.005] | +3.08% [-10.34%, +19.14%] |
| Super Lig | 1,499 | 141 | 129.2 | -8.34% [-19.82%, +6.50%] | 1.0981 [0.924, 1.279] | -2.19% [-13.84%, +12.27%] |

2 of 10 signed-error intervals exclude 0; 0 of 10 slope intervals exclude 1.

## Big-chance ablation
LR splines + context + big chance (labelled ablation): CV log loss 0.2559, test 0.2611; gain over LR splines + context CV +0.0174 [+0.0161, +0.0187], test +0.0173 [+0.0144, +0.0201]. It is never a candidate: its guard is `features.check_ablation_features`, and `evaluation.best` skips any spec with an ablation-only feature.

| Counterfactual (labelled, never a candidate) | Test log loss | Test gain over v0 [95% CI] |
|---|---|---|
| the best LR (LR splines + context) plus q214 alone | 0.2611 | +0.0004 [-0.0002, +0.0010] |
| v0's inputs refitted on train without q214 | 0.2797 | -0.0182 [-0.0211, -0.0154] |

## Penalties
One probability for every in-game penalty: the train-season Beta-binomial posterior (Jeffreys prior), 0.7775 [95% interval 0.7500, 0.8038] from 918 kicks; v0's constant `penalty_xg` is 0.7795. The test season has 145 in-game kicks and 120 goals; constant log loss on the test kicks: dev posterior 0.4674, v0 `penalty_xg` 0.4668.

Match xG = Σ non-penalty xG (headline model) + P(goal | in-game penalty) × in-game penalties; shoot-out kicks are not part of match xG, and own goals and direct corners carry no xG.

20 shoot-out kicks are reported apart, never pooled.

## Limitations & known biases
- One train season and 587 test matches in 10 competitions: the per-competition intervals above are wide.
- `assist` and `one_on_one` are Opta annotations, read as recorded.
- No score state and no freeze frame: the game situation enters only through `phase`.
- 7 shots with an unresolved assisting event are left out of every model, train and test.
- Penalties, own goals, shoot-outs and direct corners are outside the xG population (see Data).

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS4, M1.6 and M1.7): first version.
