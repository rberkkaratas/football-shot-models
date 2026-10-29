# football-shot-models

Building, evaluating and documenting **shot models** for football from first principles:

| Phase | Model | Question it answers |
|---|---|---|
| 1 | **xG**: expected goals | How likely was this shot to be scored *before* it was struck? |
| 2 | **xGOT / PSxG**: post-shot xG | Given where the shot ended up in the goal mouth, how likely was it to go in? |
| 3 | **Goals prevented / GSAx** | How many goals did a goalkeeper save compared with an average keeper facing the same shots? |

> **Status:** milestone 6 built, the last of the shot-model roadmap: goalkeepers for StatsBomb
> and the private WhoScored / Opta archive. GSAx is rated from season-out xGOT against each
> competition-season's league sum, with and without penalties, an analytic SD and a 90 %
> match-bootstrap interval per keeper-season; a hierarchical keeper model is tested on the next
> season; split-half reliability is reported at 100 / 250 / 500 shots faced. Package 1 made xGOT
> keeper-ready: StatsBomb saves snapped to the 0.1 yd goal-mouth lattice, a placement-free GBM,
> coarse-grid matches left out (PM ruling E1 (a)) and the band rule judged on bands of at least
> 100 shots. Results are the generated MS5 / MS6 blocks below and the
> [GSAx](docs/models/gsax.md) and [xGOT](docs/models/xgot-gbm.md) cards; none is typed here.
> Milestone 4 (WhoScored xG v1, [card](docs/models/xg-v1.md)) has two check rows missed with
> computed reasons: the test log-loss bound and the gain over the legacy v0, which reads Opta's
> big-chance tag (q214, a column v1's leakage rule bars). PM ruling (2026-10-25): both misses
> stand as recorded misses, and the v0 comparison is re-baselined to v0's inputs without q214,
> which the headline beats with an interval above 0. See [`docs/roadmap.md`](docs/roadmap.md).
>
> **Polish round 1 (PR1)** after the six milestones: WhoScored season-out xGOT from
> competition-season folds, one penalty δ per season-out source, team-selected StatsBomb seasons
> rated at δ = 0 and a goal_z sensitivity on the GSAx cards; WhoScored xG v2 with pre-shot
> context ([card](docs/models/xg-v2.md), v1 stays the record and the xGOT input); the StatsBomb
> headline GBM is trained without team-selected seasons after a hold-out level decomposition
> ([card](docs/models/xg-gbm.md), section "Hold-out level"). Results are the generated blocks
> below; none is typed here.

### Results (MS3, every model and StatsBomb xG on the same shots)

<!-- models:summary:start -->
| Model | CV log loss | CV AUC | CV cal. slope | Hold-out log loss | Hold-out cal. slope | Hold-out Σ xG error (signed) |
|---|---|---|---|---|---|---|
| Base rate | 0.3327 | 0.5000 | n/a | 0.3101 | n/a | +11.25% |
| LR distance + angle | 0.2961 | 0.7388 | 1.0001 | 0.2788 | 0.9648 | +9.03% |
| LR distance + angle + body part | 0.2872 | 0.7640 | 0.9996 | 0.2721 | 0.9650 | +6.86% |
| LR splines + body part | 0.2860 | 0.7650 | 0.9973 | 0.2709 | 0.9522 | +5.98% |
| LR splines + context | 0.2740 | 0.7901 | 0.9957 | 0.2595 | 0.9711 | +3.80% |
| LR splines + context + freeze frames | 0.2673 | 0.8046 | 0.9954 | 0.2547 | 0.9664 | +5.17% |
| **Monotone GBM + Platt, trained without team-selected seasons** | 0.2641 | 0.8099 | 1.0086 | 0.2533 | 0.9483 | +5.45% |
| StatsBomb xG | 0.2659 | 0.8072 | 1.0166 | 0.2520 | 0.9916 | -1.41% |

Headline model: Monotone GBM + Platt, trained without team-selected seasons. Against StatsBomb xG: CV gain +0.0018 [+0.0008, +0.0029], an interval that lies above 0; hold-out gain -0.0013 [-0.0041, +0.0017], an interval that includes 0. On the hold-out it predicts 963 goals for 913 scored (+5.45%); StatsBomb xG predicts 900 (-1.41%).

Penalties (in-game): P(goal) point estimate 0.7627 [95% interval 0.7374, 0.7872], 1,118 kicks; shoot-outs reported apart.

Match xG = Σ non-penalty xG (headline model) + P(goal | in-game penalty) × in-game penalties; shoot-out kicks are not part of match xG, and own goals and direct corners carry no xG.
<!-- models:summary:end -->

### Results (MS4, WhoScored xG v1 and the legacy v0 on the 2026/27 test season)

<!-- v1:summary:start -->
| Model | CV log loss | Test log loss | Test AUC | Test cal. slope | Test Σ xG error (signed) | Test gain over v0 [95% CI] |
|---|---|---|---|---|---|---|
| Base rate | 0.3212 | 0.3248 | 0.5000 | n/a | -1.63% | n/a |
| LR distance + angle | 0.2892 | 0.2971 | 0.7100 | 0.9258 | -1.91% | -0.0356 [-0.0394, -0.0314] |
| LR distance + angle + body part | 0.2810 | 0.2870 | 0.7470 | 0.9635 | -1.44% | n/a |
| LR splines + body part | 0.2803 | 0.2863 | 0.7466 | 0.9629 | -1.44% | n/a |
| LR splines + context | 0.2733 | 0.2784 | 0.7669 | 0.9697 | -0.91% | -0.0169 [-0.0197, -0.0139] |
| **Monotone GBM + Platt** | 0.2724 | 0.2771 | 0.7702 | 0.9786 | -1.01% | -0.0156 [-0.0183, -0.0127] |
| LR splines + context + big chance (labelled ablation) | 0.2559 | 0.2611 | 0.7974 | 0.9728 | +0.72% | +0.0004 [-0.0002, +0.0010] |
| Legacy v0 (uses Opta big chance q214) | n/a (test-only) | 0.2615 | 0.7978 | 0.9794 | +0.90% | n/a (reference) |

Headline model: Monotone GBM + Platt. On the 16,164 test shots v0 covers, its gain over v0 is -0.0156 [-0.0183, -0.0127], an interval that lies below 0. Over v0's inputs refitted on train without q214 (a labelled counterfactual: v0's information without big chance) its gain is +0.0027 [+0.0013, +0.0039], an interval that lies above 0.

Penalties (in-game): P(goal) 0.7775 [95% interval 0.7500, 0.8038] from 918 train kicks; constant log loss on the test kicks 0.4674 (v0 0.4668).

WhoScored test season, 16,164 shots, goal rate 0.0999; not comparable with the StatsBomb block.
<!-- v1:summary:end -->

WhoScored xG v2 (PR1: v1 plus pre-shot context, [card](docs/models/xg-v2.md)):

<!-- v2:summary:start -->
| Model | CV log loss | Test log loss | Test AUC | Test cal. slope | Test Σ xG error (signed) |
|---|---|---|---|---|---|
| xG v1 (Monotone GBM + Platt) | 0.2724 | 0.2771 | 0.7702 | 0.9786 | -1.01% |
| **xG v2** (Monotone GBM + Platt, context v2) | 0.2720 | 0.2760 | 0.7729 | 0.9867 | -0.88% |

Test gain of v2 over v1: +0.0010 [+0.0004, +0.0017], an interval that lies above 0. Share of the big-chance ablation gap closed on test: 6.5%. PR1 checks met: 6 of 6. v1 stays the WhoScored xG of record and the xGOT input; test goal rate 9.99%.
<!-- v2:summary:end -->

### Results (MS5, MS6: xGOT of on-target shots, both providers)

Printed by `make xgot` (paste it between the markers) and checked byte for byte by `make reproduce`. Headline model per provider in bold; CV is grouped by match on on-target shots, the hold-out is Premier League 2015/16 (StatsBomb) and the test season 2026-27 (WhoScored). ECE: CV fold mean, hold-out / test point value (its noise floor is on the cards).

<!-- xgot:summary:start -->
| Provider | Model | CV AUC | CV ECE | CV cal. slope | Hold-out / test AUC | Hold-out / test ECE | Hold-out / test cal. slope |
|---|---|---|---|---|---|---|---|
| StatsBomb open data | Base rate (on target) | 0.5000 | 0.0155 | n/a | 0.5000 | 0.0233 | n/a |
| StatsBomb open data | LR xG only | 0.7611 | 0.0205 | 1.0010 | 0.7493 | 0.0288 | 0.9490 |
| StatsBomb open data | LR xG + placement, no near / far | 0.8544 | 0.0213 | 1.0002 | 0.8588 | 0.0293 | 1.0128 |
| StatsBomb open data | LR xG + placement | 0.8616 | 0.0199 | 0.9999 | 0.8637 | 0.0164 | 0.9883 |
| StatsBomb open data | **Monotone GBM + Platt** | 0.8672 | 0.0155 | 1.0112 | 0.8685 | 0.0160 | 0.9945 |
| StatsBomb open data | Penalty xGOT (placement only) | 0.6367 | 0.0842 | 0.9448 | 0.5663 | 0.1278 | 0.8948 |
| private WhoScored / Opta event archive | Base rate (on target) | 0.5000 | 0.0146 | n/a | 0.5000 | 0.0165 | n/a |
| private WhoScored / Opta event archive | LR xG only | 0.7464 | 0.0217 | 1.0004 | 0.7484 | 0.0234 | 1.0076 |
| private WhoScored / Opta event archive | LR xG + placement, no near / far | 0.8270 | 0.0195 | 0.9988 | 0.8296 | 0.0234 | 1.0161 |
| private WhoScored / Opta event archive | LR xG + placement | 0.8368 | 0.0143 | 0.9987 | 0.8388 | 0.0146 | 1.0164 |
| private WhoScored / Opta event archive | **Monotone GBM + Platt** | 0.8435 | 0.0121 | 1.0020 | 0.8475 | 0.0194 | 1.0314 |
| private WhoScored / Opta event archive | Penalty xGOT (placement only) | 0.5999 | 0.0975 | 0.9784 | 0.5279 | 0.0962 | 0.0399 |

StatsBomb open data: headline Monotone GBM + Platt; xG input Monotone GBM + Platt, trained without team-selected seasons without keeper position (provider headline xG Monotone GBM + Platt, trained without team-selected seasons); 12 of 12 checks met (MS5, MS6); 33,749 of 35,632 C3 on-target shots scored, 1,855 left out by a PM ruling (-0.083%), goals -0.167%.

StatsBomb open data, penalty xGOT (placement only, in-game kicks): CV AUC 0.6367, CV cal. slope 0.9448; shoot-out kicks scored apart: 266 kicks, 201 goals, Σ xGOT 214.9, log loss 0.5609.

StatsBomb open data, shooting goals added (descriptive), split-half r at >= 40 shots, 597 players: SGA per shot +0.103 [+0.023, +0.181], goals − xG per shot +0.056 [-0.024, +0.136].

private WhoScored / Opta event archive: headline Monotone GBM + Platt; xG input Monotone GBM + Platt (provider headline xG Monotone GBM + Platt); 12 of 12 checks met (MS5, MS6); 32,480 of 32,490 C3 on-target shots scored, 0 left out by a PM ruling (-0.031%), goals -0.067%.

private WhoScored / Opta event archive, penalty xGOT (placement only, in-game kicks): CV AUC 0.5999, CV cal. slope 0.9784; shoot-out kicks scored apart: 16 kicks, 13 goals, Σ xGOT 13.3, log loss 0.5031.

private WhoScored / Opta event archive, shooting goals added (descriptive), split-half r at >= 40 shots, 711 players: SGA per shot -0.038 [-0.111, +0.036], goals − xG per shot +0.040 [-0.034, +0.113].
<!-- xgot:summary:end -->

### Results (MS6, goalkeepers, both providers)

Printed by `make gsax` (paste it between the markers) and checked byte for byte by `make reproduce`. GSAx is rated against each competition-season's league sum from season-out xGOT; the next-season columns are log-loss gains of the shrunken keeper model with 95% match-bootstrap intervals. Counts and metrics only; the StatsBomb keeper table is on the [GSAx card](docs/models/gsax.md).

<!-- gsax:summary:start -->
| Provider | Rated shots | Keeper-seasons | Keeper-seasons >= 100 shots | Checks met | τ̂ keeper-season | Next season: gain over raw | Gain over xGOT alone |
|---|---|---|---|---|---|---|---|
| StatsBomb open data | 33,483 | 1,205 | 97 | 6 of 6 | 0.1435 | +0.03759 [+0.02901, +0.04714] | +0.00097 [-0.00018, +0.00214] |
| private WhoScored / Opta event archive | 32,464 | 674 | 126 | 6 of 6 | 0.0861 | +0.01034 [+0.00502, +0.01619] | +0.00003 [-0.00010, +0.00017] |

StatsBomb open data: Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and are not distinguishable from xGOT alone (the league average). Without team-selected competition-seasons (22 pairs) the gain over raw is +0.01194 [-0.00019, +0.02517], its interval not above 0, and over xGOT alone +0.00050 [-0.00047, +0.00148], its interval not above 0. Stability: not distinguishable from noise at 100 shots; 6 keepers reach 250 shots: too few to judge (r +0.83 [+0.05, +0.98]); 2 keepers reach 500 shots: too few to estimate.

private WhoScored / Opta event archive: Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and are not distinguishable from xGOT alone (the league average). Stability: not distinguishable from noise at 100 shots; no keeper reaches 250 shots (empty tier); no keeper reaches 500 shots (empty tier).
<!-- gsax:summary:end -->

Part of the `football-*-models` family; shared loaders and pitch conventions come from
[football-core](../football-core).

## Data
- **StatsBomb open data**: public training data, plus a benchmark against StatsBomb's own xG.
- **WhoScored / Opta events**: a private CSV archive (2025-26 and 2026-27), read by
  football-core's `io.whoscored` (C5, milestone 4) into a git-ignored snapshot and never
  published; cards and this README carry counts and metrics only. See
  [`docs/data.md`](docs/data.md).

## Modules
| Module | What | Status |
|---|---|---|
| `data` | xG population: verified snapshot, exclusion log, families, hold-out split | ✅ M1.0 / M1.1 |
| `features` | Leakage guard (pre-shot allow-list, derived features) and design matrix | ✅ M1.0 |
| `metrics` | Log loss, Brier, AUC, calibration, reliability, thresholds | ✅ M1.1 |
| `evaluation` | Model specs, grouped CV, hold-out, benchmark: one harness for every model | ✅ M1.1 |
| `cards` | Model-card rendering from an evaluation plus per-model text | ✅ M1.1 |
| `baselines` | The three M1.1 baselines, their card text and the CLI | ✅ M1.1 |
| `ablation` | M1.2 / M1.3 LR steps, drop-one / add-one variants and step cards | ✅ M1.2 / M1.3 |
| `models` | M1.4 GBM spec and card, headline rule, MS3 checks, the StatsBomb CLI; PR1 hold-out level fixes | ✅ M1.4 / M1.5, PR1 |
| `level` | PR1 hold-out level probe, adoption rule and segment decomposition (fits nothing) | ✅ PR1 |
| `gbm` | Monotone histogram GBM with inner tuning and cross-fitted Platt | ✅ M1.4 |
| `penalties` | Beta-binomial penalty model, shoot-outs apart | ✅ M1.5 |
| `cache` | Prediction cache keyed by data digest and code hash (git-ignored) | ✅ MS4 |
| `legacy` | Legacy v0 xG behind a SHA-256 pin, scored on its own inputs | ✅ M0 |
| `v1` | WhoScored xG run: season split, candidates, v0 comparison, CLI | ✅ M1.6 |
| `release` | The `xg-v1` card and README block | ✅ M1.7 |
| `v2` | WhoScored xG v2: v1 plus pre-shot context (`prev_action`, `rebound`), qualifier census, checks | ✅ PR1 |
| `v2_cards` | The `xg-v2` card and README block (render-only) | ✅ PR1 |
| `xgot` | xGOT of on-target shots, both providers: population, cross-fitted xG input, LR / GBM / penalty xGOT, shooting goals added; pure `fit`, coarse-grid filter | ✅ P2.2–P2.5, MS6 |
| `xgot_cards` | The xGOT cards and README block from an `XgotFit` (render-only) | ✅ MS6 |
| `season_out` | Season-out xGOT refits on competition-season folds (both providers), the z-free refit | ✅ MS6, PR1 |
| `gsax` | GSAx rated against the league sum, keeper tables with SD and CI, hierarchical keeper model, next-season test, stability; goal_z sensitivity | ✅ P3.1–P3.3, PR1 |
| `gsax_cards` | The goalkeeper cards, README block and StatsBomb keeper chart (render-only) | ✅ P3.1–P3.3 |

```bash
make snapshot && make xg-baselines   # freeze StatsBomb open data, write the M1.1 cards
make xg-models                       # every MS3 model and its card (about 10 min cold,
                                     #   about 20 s from the prediction cache)
make snapshot-whoscored WHOSCORED_ARCHIVE=<archive>/data && make legacy-v0 LEGACY_V0=<v0 json>
make xg-v1                           # the WhoScored v1 card (about 8-10 min cold)
make xg-v2                           # the WhoScored v2 card (PR1; after xg-v1)
make xgot                            # the six xGOT cards (after xg-models and xg-v1)
make statsbomb-logo                  # once (network): the logo every StatsBomb chart carries
make gsax                            # the four goalkeeper cards and the StatsBomb keeper chart
make reproduce                       # every card and README block, byte for byte
```

## Repository layout
```
src/football_shot_models/   # data, features, metrics, evaluation, cards, baselines, ablation,
                            #   models, level, gbm, penalties, cache, legacy, v1, release,
                            #   v2, v2_cards, xgot, xgot_cards, season_out, gsax, gsax_cards
notebooks/                  # exploration & narrative
docs/                       # methodology, model cards, roadmap (Zensical site)
data/                       # git-ignored
tests/
```

## Development
```bash
uv sync --all-groups        # expects ../football-core next to this repo
make check                  # ruff, mypy, pytest, strict docs build
make docs-serve             # documentation site at http://127.0.0.1:8000
```

## License
MIT. See [LICENSE](LICENSE). Data is **not** covered by this licence.
