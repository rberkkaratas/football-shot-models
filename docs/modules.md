# Modules

```
src/football_shot_models/
├── data.py        # the xG population: verified snapshot, exclusion rules and log, families,
│                  #   hold-out, freeze frames and the penalty kicks of the kept matches
├── features.py    # M1.0 leakage guard (shots + matches pre_shot roles, restricted readers),
│                  #   derived features (splines, phase, contact, freeze frames), design matrix
├── metrics.py     # log loss, Brier, AUC, calibration, thresholds, match bootstrap, Beta interval
├── evaluation.py  # ModelSpec / Benchmark, grouped CV, hold-out, compare: one harness for all
├── cards.py       # model-card rendering: shared sections, gain table, checks + per-model text
├── baselines.py   # M1.1 baselines (specs and card text), CLI that writes their cards
├── ablation.py    # M1.2 / M1.3 LR steps, drop-one / add-one variants, gain rows, step cards
└── models.py      # M1.4 monotone GBM + Platt, headline rule, MS3 checks, M1.5 penalty
                   #   model, summary table, CLI that evaluates every model and writes the cards
```

| Module | Status | Milestone |
|---|---|---|
| `data` | ✅ Done | M1.0 / M1.1, MS3 (matches join, direct corners, freeze frames, penalties) |
| `features` | ✅ Done | M1.0, M1.2 / M1.3 (derived features) |
| `metrics` | ✅ Done | M1.1, MS3 (bootstrap gain, Beta interval, checks) |
| `evaluation` | ✅ Done | M1.1, MS3 (`compare`, `DesignFit`) |
| `cards` | ✅ Done | M1.1, MS3 (gain table, checks, slope columns) |
| `baselines` | ✅ Done | M1.1 |
| `ablation` | ✅ Done | M1.2 / M1.3 |
| `models` | ✅ Done | M1.4 / M1.5 |

Shared loading, the canonical shot table, snapshots and charts come from `football-core`
(`io.statsbomb`, `schema`, `io.snapshot`, `viz`).

Reproduce the M1.1 and MS3 results from the pinned StatsBomb open data:

```bash
make snapshot        # uv run fc snapshot statsbomb-open-v1 --builder statsbomb ...
make xg-baselines    # uv run python -m football_shot_models.baselines -> the 3 M1.1 cards
make xg-models       # uv run python -m football_shot_models.models -> step, GBM and penalty
                     # cards (about 7 min)
make reproduce       # regenerate every card and the README summary from the snapshot and
                     # compare them byte for byte (fails, rather than skips, without the snapshot)
```
