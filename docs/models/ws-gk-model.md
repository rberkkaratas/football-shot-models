# Model card: Hierarchical keeper model

## Overview
- **Name / version:** `ws-gk-model` v2
- **Task:** goal ~ offset(logit q) + (1 | keeper-season), no fixed intercept: a shrunken GSAx per keeper-season (P3.2)
- **Status:** experimental
- **Source(s):** private WhoScored / Opta event archive; snapshot `whoscored-v1`, data digest `c394a94944cdc12ee4d6bdd77594c9156a51aa3731955905b46ec80f648bb63e`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
Shrinks each keeper-season's GSAx towards the league according to how many shots it rests on, and asks whether that predicts the next season better. For research; the stability section of the GSAx card decides what may be said about keeper skill.

## Model

Each on-target open-play shot's goal is Bernoulli with logit = logit q + u, q the rated xGOT (so the league level is fixed per season and there is no intercept), u ~ N(0, τ²) per group. τ̂ maximises the Laplace approximation of the marginal likelihood (bounded search on log τ in [0.001, 3]); u is the posterior mode at τ̂, by Newton steps. The Laplace τ̂ is biased low when the true τ is small, so the shrinkage is on the strong side. No PyMC or Stan: one random effect needs only the closed-form Laplace fit.

| Grouping | τ̂ (logit) | At a bound | One SD in goals per 100 shots |
|---|---|---|---|
| keeper-season | 0.0861 | no | 1.82 |
| keeper career | 0.0737 | no | 1.56 |

## Next-season test

Pairs: the same keeper in the same competition in consecutive seasons (by start year); a pair whose rated-season estimate is infinite (0 or every shot a goal) is left out. For each test year the model is fitted on every rated row of earlier seasons; raw is the keeper's own unpooled shift, none is xGOT alone (the league average). Log loss on the next season's on-target shots; gains with 95% match-bootstrap intervals; the leave-out columns rate the test season without the paired keepers' own shots.

| Model | Pairs | Left out | Test shots / matches | Log loss none | Log loss raw | Log loss shrunk | Gain over raw | Gain over none | Gain over raw (leave-out δ) | Gain over none (leave-out δ) |
|---|---|---|---|---|---|---|---|---|---|---|
| Keeper-season effect | 148 | 3 | 3,234 / 475 | 0.4329 | 0.4432 | 0.4329 | +0.01034 [+0.00502, +0.01619] | +0.00003 [-0.00010, +0.00017] | +0.01155 [+0.00503, +0.01858] | +0.00005 [-0.00010, +0.00020] |
| Career effect (the literal formula) | 148 | 3 | 3,234 / 475 | 0.4329 | 0.4433 | 0.4329 | +0.01037 [+0.00504, +0.01624] | +0.00003 [-0.00010, +0.00017] | +0.01160 [+0.00511, +0.01859] | +0.00005 [-0.00010, +0.00020] |

Leave-out δ per test season, E2's minimum (at most 2 keepers or under 50 shots: δ = 0) applied to the unpaired keepers: *Keeper-season effect* 0 left out, 0 at δ = 0; *Career effect (the literal formula)* 0 left out, 0 at δ = 0.

Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and are not distinguishable from xGOT alone (the league average).

Raw GSAx is the unpooled estimate on small samples, so beating it says raw overfits; keeper differences are read from the gain over none.

Disclosures: (1) the test offsets use the test season's δ, which the paired keepers' own goals feed; the leave-out columns show the result without them. (2) No pair's rated-season xGOT was fitted on its test season (148 pairs scored, 3 left out).

## Checks

| Target | Measured | Result |
|---|---|---|
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0; judged on the keeper-season grouping, the career grouping (1 | keeper) beside it) | keeper-season +0.01034 [+0.00502, +0.01619] on 3,234 shots, 148 pairs; career +0.01037 [+0.00504, +0.01624] | met |

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
- v2 (PR1): WhoScored season-out from competition-season folds; one penalty δ per season-out source; team-selected competition-seasons at δ = 0 (flagged); goal_z sensitivity (StatsBomb).
