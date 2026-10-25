# Modules

```
src/football_shot_models/
├── data.py        # the xG population: verified snapshot, exclusion rules and log, families, hold-out
├── features.py    # M1.0 leakage guard: pre_shot allow-list from the C3 roles, design matrix
├── metrics.py     # log loss, Brier, AUC, calibration, reliability bins, milestone thresholds
├── evaluation.py  # ModelSpec / Benchmark, grouped CV, hold-out: one harness for every model
├── cards.py       # model-card rendering: shared sections + per-model CardText
└── baselines.py   # M1.1 baselines (specs and card text), CLI that writes docs/models
```

| Module | Status | Milestone |
|---|---|---|
| `data` | ✅ Done | M1.0 / M1.1 |
| `features` | ✅ Done | M1.0 |
| `metrics` | ✅ Done | M1.1 |
| `evaluation` | ✅ Done | M1.1 |
| `cards` | ✅ Done | M1.1 |
| `baselines` | ✅ Done | M1.1 |

Shared loading, the canonical shot table, snapshots and charts come from `football-core`
(`io.statsbomb`, `schema`, `io.snapshot`, `viz`).

Reproduce the M1.1 results from the pinned StatsBomb open data:

```bash
make snapshot        # uv run fc snapshot statsbomb-open-v1 --builder statsbomb ...
make xg-baselines    # uv run python -m football_shot_models.baselines -> docs/models/xg-*.md
make reproduce       # regenerate the cards from the snapshot and compare them byte for byte
                     # (fails, rather than skips, when the snapshot is missing)
```
