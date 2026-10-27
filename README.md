# football-shot-models

Building, evaluating and documenting **shot models** for football from first principles:

| Phase | Model | Question it answers |
|---|---|---|
| 1 | **xG**: expected goals | How likely was this shot to be scored *before* it was struck? |
| 2 | **xGOT / PSxG**: post-shot xG | Given where the shot ended up in the goal mouth, how likely was it to go in? |
| 3 | **Goals prevented / GSAx** | How many goals did a goalkeeper save compared with an average keeper facing the same shots? |

> **Status:** milestone 5 built: xGOT (post-shot xG) of on-target shots for StatsBomb and the
> private WhoScored / Opta archive, keeper-neutral, with the cross-fitted xG as an input; a
> placement-only penalty xGOT reported apart; and shooting goals added as split-half
> correlations next to goals − xG, a descriptive measure. Its results are the generated MS5
> block below and the [xGOT cards](docs/models/xgot-gbm.md); none is typed here. Milestone 4
> (WhoScored xG v1, [card](docs/models/xg-v1.md)) has two check rows missed with computed
> reasons: the test log-loss bound and the gain over the legacy v0, which reads Opta's
> big-chance tag (q214, a column v1's leakage rule bars). PM ruling (2026-10-25): both misses
> stand as recorded misses, and the v0 comparison is re-baselined to v0's inputs without q214,
> which the headline beats with an interval above 0. Milestone 3 (the StatsBomb xG model) is
> below. See [`docs/roadmap.md`](docs/roadmap.md).

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
| **Monotone GBM + Platt** | 0.2638 | 0.8107 | 1.0097 | 0.2533 | 0.9526 | +6.77% |
| StatsBomb xG | 0.2659 | 0.8072 | 1.0166 | 0.2520 | 0.9916 | -1.41% |

Headline model: Monotone GBM + Platt. Against StatsBomb xG: CV gain +0.0021 [+0.0012, +0.0031], an interval that lies above 0; hold-out gain -0.0013 [-0.0040, +0.0016], an interval that includes 0. On the hold-out it predicts 975 goals for 913 scored (+6.77%); StatsBomb xG predicts 900 (-1.41%).

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

### Results (MS5, xGOT of on-target shots, both providers)

Written by `make xgot` and checked byte for byte by `make reproduce`. Headline model per provider in bold; CV is grouped by match on on-target shots, the hold-out is Premier League 2015/16 (StatsBomb) and the test season 2026-27 (WhoScored). ECE: CV fold mean, hold-out / test point value (its noise floor is on the cards).

<!-- xgot:summary:start -->
| Provider | Model | CV AUC | CV ECE | CV cal. slope | Hold-out / test AUC | Hold-out / test ECE | Hold-out / test cal. slope |
|---|---|---|---|---|---|---|---|
| StatsBomb open data | Base rate (on target) | 0.5000 | 0.0154 | n/a | 0.5000 | 0.0231 | n/a |
| StatsBomb open data | LR xG only | 0.7612 | 0.0198 | 1.0012 | 0.7512 | 0.0300 | 0.9712 |
| StatsBomb open data | LR xG + placement, no near / far | 0.8467 | 0.0243 | 1.0002 | 0.8568 | 0.0351 | 1.0678 |
| StatsBomb open data | LR xG + placement | 0.8535 | 0.0232 | 1.0000 | 0.8614 | 0.0187 | 1.0500 |
| StatsBomb open data | **Monotone GBM + Platt** | 0.8619 | 0.0151 | 1.0109 | 0.8691 | 0.0142 | 1.0254 |
| StatsBomb open data | Penalty xGOT (placement only) | 0.6384 | 0.0892 | 0.9860 | 0.5663 | 0.1279 | 0.9168 |
| private WhoScored / Opta event archive | Base rate (on target) | 0.5000 | 0.0146 | n/a | 0.5000 | 0.0165 | n/a |
| private WhoScored / Opta event archive | LR xG only | 0.7464 | 0.0217 | 1.0004 | 0.7484 | 0.0234 | 1.0076 |
| private WhoScored / Opta event archive | LR xG + placement, no near / far | 0.8268 | 0.0192 | 0.9995 | 0.8293 | 0.0235 | 1.0156 |
| private WhoScored / Opta event archive | LR xG + placement | 0.8367 | 0.0144 | 0.9992 | 0.8386 | 0.0149 | 1.0155 |
| private WhoScored / Opta event archive | **Monotone GBM + Platt** | 0.8425 | 0.0121 | 1.0013 | 0.8472 | 0.0178 | 1.0333 |
| private WhoScored / Opta event archive | Penalty xGOT (placement only) | 0.5999 | 0.0975 | 0.9784 | 0.5279 | 0.0962 | 0.0399 |

StatsBomb open data: headline Monotone GBM + Platt; xG input Monotone GBM + Platt without keeper position (provider headline xG Monotone GBM + Platt); 11 of 11 MS5 checks met; 35,604 of 35,632 C3 on-target shots scored (-0.079%), goals -0.159%.

StatsBomb open data, penalty xGOT (placement only, in-game kicks): CV AUC 0.6384, CV cal. slope 0.9860; shoot-out kicks scored apart: 321 kicks, 242 goals, Σ xGOT 259.0, log loss 0.5591.

StatsBomb open data, shooting goals added (descriptive), split-half r at >= 40 shots, 633 players: SGA per shot +0.083 [+0.005, +0.160], goals − xG per shot +0.076 [-0.002, +0.153].

private WhoScored / Opta event archive: headline Monotone GBM + Platt; xG input Monotone GBM + Platt (provider headline xG Monotone GBM + Platt); 11 of 11 MS5 checks met; 32,480 of 32,490 C3 on-target shots scored (-0.031%), goals -0.067%.

private WhoScored / Opta event archive, penalty xGOT (placement only, in-game kicks): CV AUC 0.5999, CV cal. slope 0.9784; shoot-out kicks scored apart: 16 kicks, 13 goals, Σ xGOT 13.3, log loss 0.5031.

private WhoScored / Opta event archive, shooting goals added (descriptive), split-half r at >= 40 shots, 711 players: SGA per shot -0.041 [-0.114, +0.033], goals − xG per shot +0.040 [-0.034, +0.113].
<!-- xgot:summary:end -->

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
| `models` | M1.4 GBM spec and card, headline rule, MS3 checks, the StatsBomb CLI | ✅ M1.4 / M1.5 |
| `gbm` | Monotone histogram GBM with inner tuning and cross-fitted Platt | ✅ M1.4 |
| `penalties` | Beta-binomial penalty model, shoot-outs apart | ✅ M1.5 |
| `cache` | Prediction cache keyed by data digest and code hash (git-ignored) | ✅ MS4 |
| `legacy` | Legacy v0 xG behind a SHA-256 pin, scored on its own inputs | ✅ M0 |
| `v1` | WhoScored xG run: season split, candidates, v0 comparison, CLI | ✅ M1.6 |
| `release` | The `xg-v1` card and README block | ✅ M1.7 |
| `xgot` | xGOT of on-target shots, both providers: population, cross-fitted xG input, LR / GBM / penalty xGOT, shooting goals added, cards | ✅ P2.2–P2.5 |

```bash
make snapshot && make xg-baselines   # freeze StatsBomb open data, write the M1.1 cards
make xg-models                       # every MS3 model and its card (about 10 min cold,
                                     #   about 20 s from the prediction cache)
make snapshot-whoscored WHOSCORED_ARCHIVE=<archive>/data && make legacy-v0 LEGACY_V0=<v0 json>
make xg-v1                           # the WhoScored v1 card (about 8-10 min cold)
make xgot                            # the six xGOT cards (after xg-models and xg-v1)
make reproduce                       # every card and README block, byte for byte
```

## Repository layout
```
src/football_shot_models/   # data, features, metrics, evaluation, cards, baselines, ablation,
                            #   models, gbm, penalties, cache, legacy, v1, release, xgot
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
