# football-shot-models

Building, evaluating and documenting **shot models** for football from first principles:

| Phase | Model | Question it answers |
|---|---|---|
| 1 | **xG**: expected goals | How likely was this shot to be scored *before* it was struck? |
| 2 | **xGOT / PSxG**: post-shot xG | Given where the shot ended up in the goal mouth, how likely was it to go in? |
| 3 | **Goals prevented / GSAx** | How many goals did a goalkeeper save compared with an average keeper facing the same shots? |

> **Status:** milestone 4 built: xG v1 on a private WhoScored / Opta archive, trained on 2025-26
> and tested on 2026-27 ([card](docs/models/xg-v1.md); its results are the generated v1 block
> below, none typed here). Two of its check rows are missed, each with a computed reason on the
> card: the test log-loss bound, and the gain over the legacy v0, which reads Opta's big-chance
> tag (q214, a column v1's leakage rule bars). The block also gives the headline's gain over
> v0's own inputs refitted without q214, a labelled counterfactual. PM ruling (2026-10-25):
> both misses stand as recorded misses, and the v0 comparison is re-baselined to v0's inputs
> without q214, which the headline beats with an interval above 0; milestone 4 is done on that
> basis. Milestone 3 (the StatsBomb xG model) is below. See [`docs/roadmap.md`](docs/roadmap.md).

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

```bash
make snapshot && make xg-baselines   # freeze StatsBomb open data, write the M1.1 cards
make xg-models                       # every MS3 model and its card (about 10 min cold,
                                     #   about 20 s from the prediction cache)
make snapshot-whoscored WHOSCORED_ARCHIVE=<archive>/data && make legacy-v0 LEGACY_V0=<v0 json>
make xg-v1                           # the WhoScored v1 card (about 8-10 min cold)
make reproduce                       # every card and README block, byte for byte
```

## Repository layout
```
src/football_shot_models/   # data, features, metrics, evaluation, cards, baselines, ablation,
                            #   models, gbm, penalties, cache, legacy, v1, release
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
