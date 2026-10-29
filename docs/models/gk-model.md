# Model card: Hierarchical keeper model

## Overview
- **Name / version:** `gk-model` v2
- **Task:** goal ~ offset(logit q) + (1 | keeper-season), no fixed intercept: a shrunken GSAx per keeper-season (P3.2)
- **Status:** experimental
- **Source(s):** StatsBomb open data; snapshot `statsbomb-open-v1`, data digest `a10b7542f2dc13c3b1f46191c8af3e52b9b6f9281543cc7a436f9797b881209e`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
Shrinks each keeper-season's GSAx towards the league according to how many shots it rests on, and asks whether that predicts the next season better. For research; the stability section of the GSAx card decides what may be said about keeper skill.

## Model

Each on-target open-play shot's goal is Bernoulli with logit = logit q + u, q the rated xGOT (so the league level is fixed per season and there is no intercept), u ~ N(0, τ²) per group. τ̂ maximises the Laplace approximation of the marginal likelihood (bounded search on log τ in [0.001, 3]); u is the posterior mode at τ̂, by Newton steps. The Laplace τ̂ is biased low when the true τ is small, so the shrinkage is on the strong side. No PyMC or Stan: one random effect needs only the closed-form Laplace fit.

| Grouping | τ̂ (logit) | At a bound | One SD in goals per 100 shots |
|---|---|---|---|
| keeper-season | 0.1435 | no | 3.01 |
| keeper-season without team-selected competition-seasons | 0.1045 | no | 2.18 |
| keeper career | 0.1574 | no | 3.30 |

## Next-season test

Pairs: the same keeper in the same competition in consecutive seasons (by start year); a pair whose rated-season estimate is infinite (0 or every shot a goal) is left out. For each test year the model is fitted on every rated row of earlier seasons; raw is the keeper's own unpooled shift, none is xGOT alone (the league average). Log loss on the next season's on-target shots; gains with 95% match-bootstrap intervals; the leave-out columns rate the test season without the paired keepers' own shots.

| Model | Pairs | Left out | Test shots / matches | Log loss none | Log loss raw | Log loss shrunk | Gain over raw | Gain over none | Gain over raw (leave-out δ) | Gain over none (leave-out δ) |
|---|---|---|---|---|---|---|---|---|---|---|
| Keeper-season effect | 227 | 14 | 5,866 / 938 | 0.4144 | 0.4511 | 0.4135 | +0.03759 [+0.02901, +0.04714] | +0.00097 [-0.00018, +0.00214] | +0.03745 [+0.02888, +0.04696] | +0.00102 [-0.00016, +0.00219] |
| … without team-selected competition-seasons | 22 | 2 | 1,302 / 188 | 0.4345 | 0.4460 | 0.4340 | +0.01194 [-0.00019, +0.02517] | +0.00050 [-0.00047, +0.00148] | +0.01225 [-0.00005, +0.02549] | +0.00047 [-0.00051, +0.00146] |
| Career effect (the literal formula) | 238 | 3 | 6,049 / 953 | 0.4156 | 0.4367 | 0.4144 | +0.02228 [+0.01525, +0.03001] | +0.00118 [+0.00016, +0.00219] | +0.02198 [+0.01500, +0.02969] | +0.00121 [+0.00018, +0.00223] |

Leave-out δ per test season, E2's minimum (at most 2 keepers or under 50 shots: δ = 0) applied to the unpaired keepers: *Keeper-season effect* 0 left out, 0 at δ = 0; *… without team-selected competition-seasons* 0 left out, 0 at δ = 0; *Career effect (the literal formula)* 0 left out, 0 at δ = 0.

Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and are not distinguishable from xGOT alone (the league average). Without team-selected competition-seasons (22 pairs) the gain over raw is +0.01194 [-0.00019, +0.02517], its interval not above 0, and over xGOT alone +0.00050 [-0.00047, +0.00148], its interval not above 0.

Raw GSAx is the unpooled estimate on small samples, so beating it says raw overfits; keeper differences are read from the gain over none.

Disclosures: (1) the test offsets use the test season's δ, which the paired keepers' own goals feed; the leave-out columns show the result without them. (2) For 177 of the 227 pairs the test scores, the rated season's xGOT comes from fits that trained on the test season (season-out folds train on later seasons too), the same for all three predictors; a rolling xGOT is a later item (PM E4); 14 more pairs are left out by the test (infinite raw estimate).

## Checks

| Target | Measured | Result |
|---|---|---|
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0; judged on the keeper-season grouping, the career grouping (1 | keeper) beside it) | keeper-season +0.03759 [+0.02901, +0.04714] on 5,866 shots, 227 pairs; career +0.02228 [+0.01525, +0.03001] | met |

## Data licensing
Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never committed or redistributed; this card holds aggregates only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
- v2 (PR1): WhoScored season-out from competition-season folds; one penalty δ per season-out source; team-selected competition-seasons at δ = 0 (flagged); goal_z sensitivity (StatsBomb).
