# football-shot-models

Building, evaluating and documenting **shot models** for football from first principles:

| Phase | Model | Question it answers |
|---|---|---|
| 1 | **xG**: expected goals | How likely was this shot to be scored *before* it was struck? |
| 2 | **xGOT / PSxG**: post-shot xG | Given where the shot ended up in the goal mouth, how likely was it to go in? |
| 3 | **Goals prevented / GSAx** | How many goals did a goalkeeper save compared with an average keeper facing the same shots? |

> **Status:** milestone 2 done: the M1.0 leakage guard and the M1.1 xG baselines on StatsBomb
> open data, each with a model card in [`docs/models/`](docs/models/). The best baseline (LR on
> distance, angle and body part) reaches grouped-CV log loss 0.2880, Brier 0.0821 and AUC 0.7628.
> Its Σ xG error is 0.02 % pooled over the CV folds (near 0 by construction; 2.66 % fold mean)
> but 6.98 % on the hold-out (Premier League 2015/16). The log-loss and Brier targets
> (0.275 / 0.077) are missed, and the card explains why. See [`docs/roadmap.md`](docs/roadmap.md).

Part of the `football-*-models` family; shared loaders and pitch conventions come from
[football-core](../football-core).

## Data
- **StatsBomb open data**: public training data, plus a benchmark against StatsBomb's own xG.
- **WhoScored / Opta events**: a private CSV archive (2025-26 and 2026-27), read by
  football-core's planned `io.whoscored` (C5) and never published. See
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

```bash
make snapshot && make xg-baselines   # freeze StatsBomb open data, then write docs/models/xg-*.md
```

## Repository layout
```
src/football_shot_models/   # package: data, features, metrics, evaluation, cards, baselines
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
