# football-shot-models

Building, evaluating and documenting **shot models** for football from first principles:

| Phase | Model | Question it answers |
|---|---|---|
| 1 | **xG**: expected goals | How likely was this shot to be scored *before* it was struck? |
| 2 | **xGOT / PSxG**: post-shot xG | Given where the shot ended up in the goal mouth, how likely was it to go in? |
| 3 | **Goals prevented / GSAx** | How many goals did a goalkeeper save compared with an average keeper facing the same shots? |

> **Status:** milestone 3 done: the xG model on StatsBomb open data. Logistic-regression steps
> (splines, context, freeze frames), a monotone gradient-boosted model with cross-fitted Platt
> calibration and a penalty model, each with a model card in [`docs/models/`](docs/models/). The
> headline model is the monotone GBM (grouped-CV log loss 0.2638, every MS3 check met; the
> women's hold-out slope is n/a because the hold-out has no women's shots). On the Premier
> League 2015/16 hold-out its gains over the best LR and over StatsBomb's own xG include 0, and
> it predicts 6.77 % more goals than were scored. See [`docs/roadmap.md`](docs/roadmap.md).

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

Headline model: Monotone GBM + Platt. Against StatsBomb's own xG: CV gain +0.0021 [+0.0012, +0.0031], an interval that lies above 0; hold-out gain -0.0013 [-0.0040, +0.0016], an interval that includes 0. On the hold-out it predicts 975 goals for 913 scored (+6.77%); StatsBomb predicts 900 (-1.41%).

Penalties (in-game): P(goal) point estimate 0.7627 [95% interval 0.7374, 0.7872], 1,118 kicks; shoot-outs reported apart.

Match xG = Σ non-penalty xG (headline model) + P(goal | in-game penalty) × in-game penalties; shoot-out kicks are not part of match xG, and own goals and direct corners carry no xG.
<!-- models:summary:end -->

Part of the `football-*-models` family; shared loaders and pitch conventions come from
[football-core](../football-core).

## Data
- **StatsBomb open data**: public training data, plus a benchmark against StatsBomb's own xG.
- **WhoScored / Opta events**: a private CSV archive (2025-26 and 2026-27), read by
  football-core's `io.whoscored` (C5, milestone 4) and never published. See
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
| `models` | M1.4 monotone GBM + Platt, headline rule, MS3 checks, M1.5 penalties, CLI | ✅ M1.4 / M1.5 |

```bash
make snapshot && make xg-baselines   # freeze StatsBomb open data, write the M1.1 cards
make xg-models                       # every MS3 model and its card (about 7 min)
```

## Repository layout
```
src/football_shot_models/   # data, features, metrics, evaluation, cards, baselines, ablation, models
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
