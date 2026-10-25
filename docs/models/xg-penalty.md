# Model card: Penalty conversion (Beta-binomial)

## Overview
- **Name / version:** `xg-penalty` v2
- **Task:** P(goal | in-game penalty), periods 1–4; shoot-out kicks reported apart
- **Status:** experimental
- M1.5: pooled conversion of in-game penalties. P(goal) = 0.7627 [0.7374, 0.7872] (95% interval) from 1,118 kicks in 967 matches.

## Intended use
The penalty term of match xG. Match xG = Σ non-penalty xG (headline model) + P(goal | in-game penalty) × in-game penalties; shoot-out kicks are not part of match xG, and own goals and direct corners carry no xG. Not for rating takers or keepers: every in-game penalty gets the same probability.

## Data
- **Source(s):** StatsBomb open data, revision `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`; snapshot `statsbomb-open-v1`, data digest `31146e70cefb5fe39056b8ef24216724a5d5162601decb2e214c0608b2eb3e78`
- **Rows:** the shots the xG population excludes as in-game penalties, dev and hold-out (Premier League 2015/2016) together, from 55 competition-seasons.

## Target
`is_goal`: the penalty kick went in.

## Features
None: one probability for every in-game penalty.

## Model
Beta-binomial with the Jeffreys prior Beta(0.5, 0.5): the posterior is Beta(0.5 + goals, 0.5 + misses); P(goal) is its mean and the interval its 95% equal-tailed interval. Nothing is fitted per taker, keeper or competition.

## Validation design
Nothing is tuned; the model pools all kicks.
The hold-out (Premier League 2015/2016) has 91 in-game kicks, so its 95% interval [0.7240, 0.8828] is 0.159 wide and this is a weak test: the dev estimate 0.7583 lies inside it.

## Metrics
| Group | Kicks | Matches | Goals | Conversion | P(goal) [95% interval] | StatsBomb mean xG |
|---|---|---|---|---|---|---|
| In-game: all | 1,118 | 967 | 853 | 0.7630 | 0.7627 [0.7374, 0.7872] | 0.7835 |
| In-game: dev | 1,027 | 889 | 779 | 0.7585 | 0.7583 [0.7316, 0.7839] | 0.7835 |
| In-game: hold-out | 91 | 78 | 74 | 0.8132 | 0.8098 [0.7240, 0.8828] | 0.7835 |
| In-game: men | 756 | 645 | 577 | 0.7632 | 0.7629 [0.7320, 0.7925] | 0.7835 |
| In-game: women | 362 | 322 | 276 | 0.7624 | 0.7617 [0.7166, 0.8041] | 0.7835 |
| In-game: league or tournament | 547 | 467 | 406 | 0.7422 | 0.7418 [0.7044, 0.7775] | 0.7835 |
| In-game: team-selected | 209 | 178 | 171 | 0.8182 | 0.8167 [0.7617, 0.8659] | 0.7835 |
| Shoot-out: all | 363 | 36 | 242 | 0.6667 | 0.6662 [0.6170, 0.7137] | 0.7835 |
| Shoot-out: men | 276 | 29 | 189 | 0.6848 | 0.6841 [0.6283, 0.7374] | 0.7835 |
| Shoot-out: women | 87 | 7 | 53 | 0.6092 | 0.6080 [0.5045, 0.7068] | 0.7835 |

### Milestone check

| Target | Measured | Result |
|---|---|---|
| posterior mean in [0.76, 0.80] (milestone wording) | 0.7627 [0.7374, 0.7872]; 0.0027 above the lower bound | met |

The 95% interval is 0.050 wide, wider than the 0.04 window, and extends below it: the check is on the point estimate.

Without the team-selected seasons (209 kicks, conversion 0.818) the pooled estimate is 0.7500 [0.7214, 0.7776] from 909 kicks, outside the window.

## Benchmarks
Each model as a constant on the 91 hold-out kicks:

| Model | Kicks | Log loss |
|---|---|---|
| Dev posterior mean (0.7583) | 91 | 0.4903 |
| StatsBomb mean xG (0.7835) | 91 | 0.4843 |

## Limitations & known biases
- No taker, keeper or foot: the model cannot tell a specialist from anyone else.
- Kicks are pooled as exchangeable: across 55 competition-seasons the Pearson dispersion is 0.94 on 54 df (p 0.59), and the most frequent taker has 80 of 1,118 kicks.
- In-game conversion by family: league or tournament 0.742 (547 kicks); team-selected 0.818 (209 kicks); women 0.762 (362 kicks).
- 363 shoot-out kicks (period 5) cluster in 36 matches; they are reported apart, never pooled, and their interval treats kicks of one shoot-out as independent.

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS3, M1.5): first version.
- v2 (MS3 assessment): the milestone check adds the pooled estimate without the team-selected seasons.
