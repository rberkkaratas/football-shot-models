# Model card: Hierarchical keeper model

## Overview
- **Name / version:** `gk-model` v1
- **Task:** goal ~ offset(logit q) + (1 | keeper-season), no fixed intercept: a shrunken GSAx per keeper-season (P3.2)
- **Status:** experimental
- **Source(s):** StatsBomb open data; snapshot `statsbomb-open-v1`, data digest `067ddf5342cfd608045a15189a0b43008664d23f4c5ac4fab5ac403468e57ee5`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
Shrinks each keeper-season's GSAx towards the league according to how many shots it rests on, and asks whether that predicts the next season better. For research; the stability section of the GSAx card decides what may be said about keeper skill.

## Model

Each on-target open-play shot's goal is Bernoulli with logit = logit q + u, q the rated xGOT (so the league level is fixed per season and there is no intercept), u ~ N(0, τ²) per group. τ̂ maximises the Laplace approximation of the marginal likelihood (bounded search on log τ in [0.001, 3]); u is the posterior mode at τ̂, by Newton steps. The Laplace τ̂ is biased low when the true τ is small, so the shrinkage is on the strong side. No PyMC or Stan: one random effect needs only the closed-form Laplace fit.

| Grouping | τ̂ (logit) | At a bound | One SD in goals per 100 shots |
|---|---|---|---|
| keeper-season | 0.1267 | no | 2.66 |
| keeper career | 0.1567 | no | 3.29 |

## Next-season test

Pairs: the same keeper in the same competition in consecutive seasons (by start year); a pair whose rated-season estimate is infinite (0 or every shot a goal) is left out. For each test year the model is fitted on every rated row of earlier seasons; raw is the keeper's own unpooled shift, none is xGOT alone (the league average). Log loss on the next season's on-target shots; gains with 95% match-bootstrap intervals; the leave-out columns rate the test season without the paired keepers' own shots.

| Model | Pairs | Left out | Test shots / matches | Log loss none | Log loss raw | Log loss shrunk | Gain over raw | Gain over none | Gain over raw (leave-out δ) | Gain over none (leave-out δ) |
|---|---|---|---|---|---|---|---|---|---|---|
| Keeper-season effect | 227 | 14 | 5,866 / 938 | 0.4123 | 0.4476 | 0.4113 | +0.03630 [+0.02779, +0.04520] | +0.00105 [+0.00014, +0.00193] | +0.03712 [+0.02843, +0.04619] | +0.00119 [+0.00025, +0.00209] |
| … without team-selected competition-seasons | 22 | 2 | 1,302 / 188 | 0.4347 | 0.4458 | 0.4341 | +0.01177 [-0.00097, +0.02546] | +0.00060 [-0.00041, +0.00164] | +0.01217 [-0.00070, +0.02596] | +0.00055 [-0.00045, +0.00161] |
| Career effect (the literal formula) | 238 | 3 | 6,049 / 953 | 0.4136 | 0.4348 | 0.4123 | +0.02252 [+0.01581, +0.03011] | +0.00129 [+0.00031, +0.00224] | +0.02359 [+0.01671, +0.03135] | +0.00127 [+0.00025, +0.00226] |

Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and improve log loss by 0.00105 over xGOT alone. Without team-selected competition-seasons (22 pairs) the gain over raw is +0.01177 [-0.00097, +0.02546], its interval not above 0, and over xGOT alone +0.00060 [-0.00041, +0.00164], its interval not above 0.

Raw GSAx is the unpooled estimate on small samples, so beating it says raw overfits; keeper differences are read from the gain over none.

Disclosures: (1) the test offsets use the test season's δ, which the paired keepers' own goals feed; the leave-out columns show the result without them. (2) The rated season's xGOT comes from fits that saw the test season (season-out fits train on later seasons too), the same for all three predictors; a rolling xGOT is a later item (PM E4).

## Checks

| Target | Measured | Result |
|---|---|---|
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0; judged on the keeper-season grouping, the career grouping (1 | keeper) beside it) | keeper-season +0.03630 [+0.02779, +0.04520] on 5,866 shots, 227 pairs; career +0.02252 [+0.01581, +0.03011] | met |

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
