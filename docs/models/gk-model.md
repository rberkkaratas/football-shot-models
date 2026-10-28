# Model card: Hierarchical keeper model

## Overview
- **Name / version:** `gk-model` v1
- **Task:** goal ~ offset(logit q) + (1 | keeper-season), no fixed intercept: a shrunken GSAx per keeper-season (P3.2)
- **Status:** experimental
- **Source(s):** StatsBomb open data; snapshot `statsbomb-open-v1`, data digest `067ddf5342cfd608045a15189a0b43008664d23f4c5ac4fab5ac403468e57ee5`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
Shrinks each keeper-season's GSAx towards the league according to how many shots it rests on, and asks whether that predicts the next season better. For research; the stability section of the GSAx card decides what may be said about keeper skill.

## Model

Each on-target open-play shot's goal is Bernoulli with logit = logit q + u, q the rated xGOT (so the league level is fixed per season and there is no intercept), u ~ N(0, τ²) per group. τ̂ maximises the Laplace approximation of the marginal likelihood (bounded search on log τ in [0.001, 3]); u is the posterior mode at τ̂, by Newton steps. The Laplace τ̂ is biased low when the true τ is small (a simulation probe in the run folder), so the shrinkage is on the strong side. No PyMC or Stan: the closed-form Laplace fit is exact enough for one random effect and takes well under a second.

| Grouping | τ̂ (logit) | At a bound | One SD in goals per 100 shots |
|---|---|---|---|
| keeper-season | 0.1225 | no | 2.57 |
| keeper career | 0.1536 | no | 3.22 |

## Next-season test

Pairs: the same keeper in the same competition in consecutive seasons (by start year); a pair whose rated-season estimate is infinite (0 or every shot a goal) is left out. For each test year the model is fitted on every rated row of earlier seasons; raw is the keeper's own unpooled shift, none is xGOT alone (the league average). Log loss on the next season's on-target shots; gains with 95% match-bootstrap intervals; the leave-out columns rate the test season without the paired keepers' own shots.

| Model | Pairs | Left out | Test shots / matches | Log loss none | Log loss raw | Log loss shrunk | Gain over raw | Gain over none | Gain over raw (leave-out δ) | Gain over none (leave-out δ) |
|---|---|---|---|---|---|---|---|---|---|---|
| Keeper-season effect | 227 | 14 | 5,866 / 938 | 0.4134 | 0.4474 | 0.4124 | +0.03501 [+0.02677, +0.04361] | +0.00099 [+0.00015, +0.00183] | n/a | n/a |
| … without team-selected competition-seasons | 22 | 2 | 1,302 / 188 | 0.4350 | 0.4463 | 0.4344 | +0.01188 [-0.00089, +0.02526] | +0.00059 [-0.00049, +0.00171] | +0.01227 [-0.00064, +0.02587] | +0.00054 [-0.00054, +0.00167] |
| Career effect (the literal formula) | 238 | 3 | 6,049 / 953 | 0.4146 | 0.4345 | 0.4133 | +0.02123 [+0.01471, +0.02861] | +0.00129 [+0.00031, +0.00225] | n/a | n/a |

Leave-out columns of *Keeper-season effect*: n/a, because 1 test season(s) have no shot faced by an unpaired keeper, or no finite δ without the paired ones.
Leave-out columns of *Career effect (the literal formula)*: n/a, because 1 test season(s) have no shot faced by an unpaired keeper, or no finite δ without the paired ones.

Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and add +0.00099 log loss over xGOT alone.

Raw GSAx is the unpooled estimate on small samples, so beating it says raw overfits; keeper differences are read from the gain over none.

Disclosures: (1) the test offsets use the test season's δ, which the paired keepers' own goals feed; the leave-out columns show the result without them. (2) The rated season's xGOT comes from fits that saw the test season (season-out fits train on later seasons too), the same for all three predictors; a rolling xGOT is a later item (PM E4).

## Checks

| Target | Measured | Result |
|---|---|---|
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0) | +0.03501 [+0.02677, +0.04361] on 5,866 shots, 227 pairs | met |

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
