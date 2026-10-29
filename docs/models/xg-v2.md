# Model card: Monotone GBM + Platt, context v2

## Overview
- **Name / version:** `ws-gbm-v2` v1
- **Task:** binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time included; direct free kicks and corners from open play or set pieces included; direct corners, own goals and shoot-outs excluded)
- **Status:** experimental
- PR1: WhoScored xG with the pre-shot context of the previous play event (v1's inputs plus `prev_action` and `rebound`), beside xG v1. This is the headline model: its CV log-loss gain over LR splines + context v2 is +0.0008 [+0.0003, +0.0012], a 95% interval that lies above 0; on the test it is +0.0014 [+0.0005, +0.0024], an interval that lies above 0.

## Intended use
The context-model result beside xG v1 for WhoScored / Opta shot events. xG v1's card stays the record, and v1's keeper-neutral headline stays the WhoScored xGOT input (PM PR1 E4 (a); `prev_action` and `rebound` read the keeper's previous event, so they are refused in the xGOT input chain, PM MS6 E5). Not for penalties, not for other providers' data without re-fitting.

## Data
- **Source(s):** private WhoScored / Opta event archive, revision `84571265636c3c30fcfcf9b9b0b3bd1381f7488fc3d3870893d2539f8d7c3b23`; snapshot `whoscored-v1`, data digest `c394a94944cdc12ee4d6bdd77594c9156a51aa3731955905b46ec80f648bb63e`
- **Competitions & seasons:** 20 competition-seasons; hold-out 2026/2027
- **Rows:** train and validation (dev) 80,664 shots in 3,215 matches / test (hold-out) 16,164 shots in 587 matches
- **Filters:** excluded shots, by first matching reason.

| Filter | Shots | Goals |
|---|---|---|
| own goal (Opta q28) | 330 | 330 |
| shoot-out (period 5) | 20 | 13 |
| penalty (in-game; the penalty probability below) | 1,063 | 834 |
| direct corner (Opta q263) | 13 | 7 |
| no freeze frame | 0 | 0 |

## Target
`is_goal`: the ball went in. Own goals are excluded, so every goal is the shooter's.

## Features
| Feature | Definition | Source field(s) |
|---|---|---|
| `distance_m` | Metres from the shot location to the goal centre | `distance_m` (`pitch.shot_distance`) |
| `angle_rad` | Opening angle between the posts seen from the shot location, radians | `angle_rad` (`pitch.opening_angle`) |
| `body_part` | Indicators for left foot, head, other; reference right foot | Opta q20 right foot, q72 left foot, q15 head, q21 other |
| `phase` | Indicators for fast break, corner, free kick, throw in, direct free kick; reference open play. Direct free kicks wide of the box (y < 13.84 or > 54.16) join free kick | Opta pattern q22 open play, q23 fast break, q24 from free kick, q25 from corner, q26 direct free kick, q160 from throw in (`situation`), `y` |
| `assist` | Indicators for the assisting event: pass, cross, through ball; reference none. No cutback level: the archive has no pull-back qualifier | Opta q55 related event, its q2 cross and q4 through ball (`assist_type`) |
| `one_on_one` | Opta 1-on-1 tag (annotator) | Opta q89 |
| `prev_action` | Indicators for the last play event before the shot (stoppages skipped): the shooting team's take-on, regain, earlier shot or other action, the opponent's clearance, outfield block, keeper action or other; reference pass | Event types before the shot in file order; Opta q94 on a previous Save |
| `rebound` | The team's previous shot of the period is at most 3 s earlier | Shot event clocks |

## Model
`HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1, min_samples_leaf=100, early_stopping=False, random_state=0)`, `monotonic_cst` placed by column name, no class weights. scikit-learn 1.9.1.

Tuning inside each fit: 4-fold `GroupKFold` by match on its training rows, `max_leaf_nodes` in (8, 16, 32) and 50, 100, ..., 600 iterations (staged scores of one fit per leaf count); the lowest inner out-of-fold log loss wins, ties to fewer leaves, then fewer iterations. Platt scaling: an unpenalised logistic regression of the goal on the chosen configuration's inner out-of-fold log-odds, applied to the refit on all training rows; a slope <= 0 is refused.

Tuning and calibration see only the rows of the fit they belong to; each outer test fold is scored by a model that never saw it.

Settings of the fit on all of dev:

| Setting | Value |
|---|---|
| max_leaf_nodes | 16 |
| iterations | 150 |
| Platt intercept | -0.0482 |
| Platt slope | 0.9743 |

Calibration is close to identity (Platt slope 0.974, intercept -0.048): the boosted model is already calibrated and Platt is a safeguard.

## Validation design
Train: season(s) 2025/2026, 2026; test: 2026/2027, the population and split of xG v1. Grouped 5-fold CV by `match_id` on train only (**Validation**); **Train** is the fit on all of train, scored in-sample; **Hold-out** below is the test season, scored once by that fit and never used to choose.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | 0.2685 | 0.2720 ± 0.0047 | 0.2760 |
| Brier score | 0.0762 | 0.0770 ± 0.0014 | 0.0785 |
| ROC AUC | 0.7831 | 0.7743 ± 0.0070 | 0.7729 |
| Log-loss skill | 0.1639 | 0.1531 | 0.1500 |
| Brier skill | 0.1397 | 0.1302 | 0.1269 |
| Calibration intercept | -0.0005 | -0.0007 | 0.0114 |
| Calibration slope | 1.0451 | 1.0046 | 0.9867 |
| Σ xG | 7925.8789 | 7926.1359 | 1599.7649 |
| Goals | 7,923 | 7,923 | 1,614 |
| abs(Σ xG − goals) / goals | 0.04% | 0.04% | 0.88% |
| abs(Σ xG − goals) / goals, fold mean (max) | n/a | 1.22% (max 2.38%) | n/a |

Skill is `1 − metric / base-rate metric` on the same shots and column.

### Log-loss gain (95% match-bootstrap CI)

Gain = log loss of the reference minus this model, pooled out of fold over all dev shots (CV) and over the hold-out; positive means this model is better. The 95% interval is the percentile interval of 2,000 resamples of matches (seed 0) for the fixed fold models: it covers which matches were sampled, not refitting.

| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |
|---|---|---|
| over xG v1 | +0.0004 [+0.0001, +0.0006] | +0.0010 [+0.0004, +0.0017] |
| over LR splines + context v2 | +0.0008 [+0.0003, +0.0012] | +0.0014 [+0.0005, +0.0024] |

### Milestone checks (PR1)

Judged on the test season; the last two rows report, they do not judge.

| Target | Measured | Result |
|---|---|---|
| Test log-loss gain over xG v1, 95% match-bootstrap interval above 0 | Monotone GBM + Platt, context v2: +0.0010 [+0.0004, +0.0017], an interval that lies above 0 | met |
| Calibration slope in [0.9, 1.1], test season | 0.9867 | met |
| GBM headline iff its CV gain over LR splines + context v2 has a 95% interval above 0 | CV +0.0008 [+0.0003, +0.0012]; headline Monotone GBM + Platt, context v2 | met |
| No headline feature grades the chance or comes after the shot (big chance only in the v1 ablation; every feature's Opta qualifiers classed as allowed) | headline features: distance_m, angle_rad, body_part, phase, assist, one_on_one, prev_action, rebound | met |
| No keeper-derived context in the WhoScored xGOT input chain (PM MS6 E5) | keeper-derived features there: none | met |
| Every Opta qualifier on a population shot has a class | 56 qualifier(s), unclassified: none | met |
| Reporting row (not a check): share of the q214 ablation gap closed, test / CV | 6.5% / 2.3% | n/a |
| Reporting row (not a check): test Σ xG error, signed | -0.88% | n/a |

### Reliability

Ten equal-count bins of the out-of-fold predictions.

| Bin | Shots | Mean xG | Goal rate |
|---|---|---|---|
| 0 | 8,067 | 1.87% | 2.14% |
| 1 | 8,066 | 2.68% | 2.48% |
| 2 | 8,067 | 3.37% | 3.20% |
| 3 | 8,066 | 4.24% | 4.10% |
| 4 | 8,066 | 5.28% | 5.03% |
| 5 | 8,067 | 6.68% | 6.50% |
| 6 | 8,066 | 8.58% | 8.38% |
| 7 | 8,067 | 11.80% | 11.71% |
| 8 | 8,066 | 17.40% | 18.35% |
| 9 | 8,066 | 36.35% | 36.33% |

This table stands in for a reliability plot until a reliability chart is added to `viz` (F1).

### Segments

Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the segment's own goal rate`, so segments with different goal rates can be compared; `n/a` for a segment with one class.

| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier | Log-loss skill | Cal. slope (pooled OOF) |
|---|---|---|---|---|---|---|---|---|---|
| gender = male | 80,664 | 7,923 | 9.82% | 7926.1359 | 1.0004 | 0.2720 | 0.0770 | 0.1531 | 1.0047 |
| body_part = right_foot | 39,699 | 4,000 | 10.08% | 3941.9908 | 0.9855 | 0.2741 | 0.0779 | 0.1612 | 0.9938 |
| body_part = left_foot | 26,204 | 2,430 | 9.27% | 2459.6458 | 1.0122 | 0.2626 | 0.0734 | 0.1498 | 0.9952 |
| body_part = head | 14,536 | 1,446 | 9.95% | 1473.1084 | 1.0187 | 0.2813 | 0.0807 | 0.1316 | 1.0652 |
| body_part = other | 225 | 47 | 20.89% | 51.3910 | 1.0934 | 0.4076 | 0.1251 | 0.2046 | 1.2011 |

### Test by competition

The headline on each competition's test shots; intervals from 2,000 resamples of that competition's matches.

| Competition | Shots | Goals | Σ xG | Σ xG error, signed [95% CI] |
|---|---|---|---|---|
| 1. Bundesliga | 1,110 | 131 | 119.2 | -8.98% [-20.52%, +7.00%] |
| Belgian Pro League | 1,771 | 163 | 177.8 | +9.06% [-5.02%, +26.59%] |
| Championship | 2,543 | 251 | 252.8 | +0.73% [-8.71%, +11.69%] |
| Eredivisie | 1,875 | 224 | 195.0 | -12.96% [-21.74%, -2.41%] |
| La Liga | 1,798 | 180 | 175.0 | -2.78% [-13.04%, +10.30%] |
| Ligue 1 | 1,238 | 109 | 128.6 | +18.01% [+1.83%, +39.31%] |
| Premier League | 1,382 | 128 | 135.9 | +6.21% [-7.87%, +24.35%] |
| Primeira Liga | 1,496 | 146 | 144.4 | -1.10% [-13.56%, +13.98%] |
| Serie A | 1,452 | 141 | 140.9 | -0.08% [-13.89%, +15.45%] |
| Super Lig | 1,499 | 141 | 130.1 | -7.70% [-19.21%, +7.16%] |

### Share of the big-chance gap

- test: 6.5% = (v1 0.2771 − v2 0.2760) / (v1's headline (Monotone GBM + Platt, 0.2771) minus v1's labelled ablation (LR splines + context + big chance (labelled ablation), 0.2611)).
- CV: 2.3% = (v1 0.2724 − v2 0.2720) / (v1's headline (Monotone GBM + Platt, 0.2724) minus v1's labelled ablation (LR splines + context + big chance (labelled ablation), 0.2559)).

### Census of pre-shot information (counts only)

Every Opta qualifier on the population's shots, by class; situation qualifiers may enter xG, a grading, post-shot or outcome-derived one never does.

| Qualifier | Class | Train shots | Train goals | Test shots |
|---|---|---|---|---|
| q15 Head | body part | 14,536 | 1,446 | 2,845 |
| q16 SmallBoxCentre | location zone | 3,510 | 1,383 | 695 |
| q17 BoxCentre | location zone | 29,323 | 3,960 | 5,919 |
| q18 OutOfBoxCentre | location zone | 27,765 | 1,116 | 5,462 |
| q19 ThirtyFivePlusCentre | location zone | 417 | 15 | 90 |
| q20 RightFoot | body part | 39,699 | 4,000 | 8,038 |
| q21 OtherBodyPart | body part | 225 | 47 | 44 |
| q22 RegularPlay | pattern | 50,624 | 5,100 | 10,181 |
| q23 FastBreak | pattern | 6,159 | 869 | 1,337 |
| q24 SetPiece | pattern | 5,095 | 409 | 925 |
| q25 FromCorner | pattern | 13,261 | 1,175 | 2,682 |
| q26 DirectFreekick | pattern | 2,316 | 118 | 447 |
| q29 Assisted | assist link | 60,107 | 6,082 | 12,194 |
| q55 RelatedEventId | assist link | 60,107 | 6,082 | 12,194 |
| q56 Zone | location zone | 80,664 | 7,923 | 16,164 |
| q60 SmallBoxRight | location zone | 1,657 | 247 | 323 |
| q61 SmallBoxLeft | location zone | 1,780 | 227 | 335 |
| q62 DeepBoxRight | location zone | 896 | 73 | 196 |
| q63 BoxRight | location zone | 6,715 | 411 | 1,350 |
| q64 BoxLeft | location zone | 7,156 | 388 | 1,481 |
| q65 DeepBoxLeft | location zone | 888 | 67 | 203 |
| q66 OutOfBoxDeepRight | location zone | 105 | 9 | 24 |
| q67 OutOfBoxRight | location zone | 96 | 6 | 17 |
| q68 OutOfBoxLeft | location zone | 127 | 11 | 22 |
| q69 OutOfBoxDeepLeft | location zone | 170 | 9 | 38 |
| q70 ThirtyFivePlusRight | location zone | 40 | 1 | 7 |
| q71 ThirtyFivePlusLeft | location zone | 19 | 0 | 2 |
| q72 LeftFoot | body part | 26,204 | 2,430 | 5,237 |
| q73 MissLeft | post-shot | 6,330 | 0 | 1,458 |
| q74 MissHigh | post-shot | 6,159 | 0 | 1,201 |
| q75 MissRight | post-shot | 5,557 | 0 | 1,264 |
| q76 LowLeft | post-shot | 10,933 | 2,240 | 2,151 |
| q77 HighLeft | post-shot | 4,430 | 770 | 845 |
| q78 LowCentre | post-shot | 22,851 | 1,511 | 4,412 |
| q79 HighCentre | post-shot | 2,877 | 610 | 555 |
| q80 LowRight | post-shot | 8,887 | 2,058 | 1,803 |
| q81 HighRight | post-shot | 4,693 | 734 | 912 |
| q82 Blocked | post-shot | 22,925 | 0 | 4,474 |
| q89 OneOnOne | situation | 1,521 | 553 | 319 |
| q100 SixYardBlock | post-shot | 2,851 | 0 | 523 |
| q101 SavedOffline | post-shot | 689 | 0 | 143 |
| q102 GoalMouthY | post-shot | 80,664 | 7,923 | 16,164 |
| q103 GoalMouthZ | post-shot | 80,664 | 7,923 | 16,164 |
| q108 Volley | situation | 9,262 | 872 | 1,821 |
| q146 BlockedX | post-shot | 46,369 | 0 | 9,382 |
| q147 BlockedY | post-shot | 46,369 | 0 | 9,382 |
| q154 IntentionalAssist | situation | 37,804 | 4,766 | 7,820 |
| q160 ThrowinSetPiece | pattern | 3,209 | 252 | 592 |
| q178 StandingSave | post-shot | 211 | 19 | 0 |
| q214 BigChance | grading | 13,543 | 4,835 | 2,821 |
| q215 IndividualPlay | situation | 30,233 | 1,709 | 5,596 |
| q233 OppositeRelatedEvent | post-shot | 40,340 | 0 | 7,857 |
| q328 FirstTouch | situation | 40,749 | 4,946 | 8,091 |
| q11111 IntentionalGoalAssist | outcome-derived | 4 | 0 | 4 |
| q11112 BigChanceCreated | outcome-derived | 389 | 0 | 79 |
| q11113 KeyPass | outcome-derived | 722 | 0 | 166 |

| Context column = value | Class | Keeper-derived (xG only) | Train shots | Train goals | Test shots |
|---|---|---|---|---|---|
| prev_action = pass | situation | yes | 54,074 | 5,225 | 10,991 |
| prev_action = take_on | situation | yes | 4,133 | 451 | 894 |
| prev_action = regain | situation | yes | 3,083 | 247 | 586 |
| prev_action = shot | situation | yes | 793 | 187 | 207 |
| prev_action = other | situation | yes | 2,468 | 252 | 422 |
| prev_action = opp_clearance | situation | yes | 7,788 | 422 | 1,498 |
| prev_action = opp_block | situation | yes | 1,623 | 186 | 311 |
| prev_action = opp_keeper | situation | yes | 1,741 | 412 | 344 |
| prev_action = opp_other | situation | yes | 4,961 | 541 | 911 |
| rebound = true | situation | yes | 4,398 | 901 | 920 |
| rebound = false | situation | yes | 76,266 | 7,022 | 15,244 |

- carry: not recorded by Opta; dribble = TakeOn.

## Benchmarks
| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss | Hold-out Brier | Hold-out AUC | Hold-out cal. slope |
|---|---|---|---|---|---|---|---|---|
| Base rate | 0.3212 | 0.0886 | 0.5000 | n/a | 0.3248 | 0.0899 | 0.5000 | n/a |
| LR distance + angle | 0.2892 | 0.0812 | 0.7234 | 0.9993 | 0.2971 | 0.0837 | 0.7100 | 0.9258 |
| LR distance + angle + body part | 0.2810 | 0.0793 | 0.7545 | 0.9989 | 0.2870 | 0.0814 | 0.7470 | 0.9635 |
| LR splines + body part | 0.2803 | 0.0792 | 0.7531 | 0.9977 | 0.2863 | 0.0812 | 0.7466 | 0.9629 |
| LR splines + context | 0.2733 | 0.0773 | 0.7709 | 0.9960 | 0.2784 | 0.0791 | 0.7669 | 0.9697 |
| LR splines + context v2 | 0.2728 | 0.0771 | 0.7716 | 0.9943 | 0.2775 | 0.0788 | 0.7683 | 0.9736 |
| Monotone GBM + Platt, context v2 | 0.2720 | 0.0770 | 0.7743 | 1.0046 | 0.2760 | 0.0785 | 0.7729 | 0.9867 |

## Limitations & known biases
- `prev_action` and `rebound` are derived from the event order of the archive (file order with a fail-closed clock check); Opta records no carry.
- One train season; the test season is partial.
- Big chance (q214) stays out: it grades the chance; the gap share below says how much of its ablation gain the situation columns recover.

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (PR1, xg-context): first version.
