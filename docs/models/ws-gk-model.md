# Model card: Hierarchical keeper model

## Overview
- **Name / version:** `ws-gk-model` v1
- **Task:** goal ~ offset(logit q) + (1 | keeper-season), no fixed intercept: a shrunken GSAx per keeper-season (P3.2)
- **Status:** experimental
- **Source(s):** private WhoScored / Opta event archive; snapshot `whoscored-v1`, data digest `8033836b08b03fcbaffe2a19cc3a9549f2adb2eebf10f2871fff57f267b79ebb`; xGOT: the headline Monotone GBM + Platt, refitted season-out

## Intended use
Shrinks each keeper-season's GSAx towards the league according to how many shots it rests on, and asks whether that predicts the next season better. For research; the stability section of the GSAx card decides what may be said about keeper skill.

## Model

Each on-target open-play shot's goal is Bernoulli with logit = logit q + u, q the rated xGOT (so the league level is fixed per season and there is no intercept), u ~ N(0, τ²) per group. τ̂ maximises the Laplace approximation of the marginal likelihood (bounded search on log τ in [0.001, 3]); u is the posterior mode at τ̂, by Newton steps. The Laplace τ̂ is biased low when the true τ is small, so the shrinkage is on the strong side. No PyMC or Stan: one random effect needs only the closed-form Laplace fit.

| Grouping | τ̂ (logit) | At a bound | One SD in goals per 100 shots |
|---|---|---|---|
| keeper-season | 0.0770 | no | 1.63 |
| keeper career | 0.0628 | no | 1.33 |

## Next-season test

Pairs: the same keeper in the same competition in consecutive seasons (by start year); a pair whose rated-season estimate is infinite (0 or every shot a goal) is left out. For each test year the model is fitted on every rated row of earlier seasons; raw is the keeper's own unpooled shift, none is xGOT alone (the league average). Log loss on the next season's on-target shots; gains with 95% match-bootstrap intervals; the leave-out columns rate the test season without the paired keepers' own shots.

| Model | Pairs | Left out | Test shots / matches | Log loss none | Log loss raw | Log loss shrunk | Gain over raw | Gain over none | Gain over raw (leave-out δ) | Gain over none (leave-out δ) |
|---|---|---|---|---|---|---|---|---|---|---|
| Keeper-season effect | 148 | 3 | 3,234 / 475 | 0.4332 | 0.4436 | 0.4332 | +0.01047 [+0.00513, +0.01630] | +0.00001 [-0.00007, +0.00010] | +0.01192 [+0.00529, +0.01915] | +0.00002 [-0.00007, +0.00011] |
| Career effect (the literal formula) | 148 | 3 | 3,234 / 475 | 0.4332 | 0.4437 | 0.4332 | +0.01050 [+0.00514, +0.01633] | +0.00001 [-0.00007, +0.00010] | +0.01197 [+0.00540, +0.01926] | +0.00002 [-0.00007, +0.00011] |

Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and predict next season no better than xGOT alone (the league average).

Raw GSAx is the unpooled estimate on small samples, so beating it says raw overfits; keeper differences are read from the gain over none.

Disclosures: (1) the test offsets use the test season's δ, which the paired keepers' own goals feed; the leave-out columns show the result without them. (2) The rated season's xGOT comes from fits that saw the test season (season-out fits train on later seasons too), the same for all three predictors; a rolling xGOT is a later item (PM E4).

## Checks

| Target | Measured | Result |
|---|---|---|
| Shrunken estimates predict the next season better than raw GSAx (log-loss gain, 95% match-bootstrap interval above 0; judged on the keeper-season grouping, the career grouping (1 | keeper) beside it) | keeper-season +0.01047 [+0.00513, +0.01630] on 3,234 shots, 148 pairs; career +0.01050 [+0.00514, +0.01633] | met |

## Data licensing
Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds counts and metrics only.

## Changelog
- v1 (MS6, P3.1-P3.3): first version.
