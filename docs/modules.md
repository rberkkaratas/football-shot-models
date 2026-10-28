# Modules

```
src/football_shot_models/
├── data.py        # the xG population: verified snapshot (StatsBomb or WhoScored builder),
│                  #   exclusion rules and log, families, hold-out (pair or season split),
│                  #   freeze frames and the penalty kicks of the kept matches
├── features.py    # M1.0 leakage guard (shots + matches pre_shot roles, restricted readers),
│                  #   derived features (splines, phase, contact, assist, freeze frames), the
│                  #   ablation guard for big chance, design matrix; MS5: the xGOT guard
│                  #   (declared post-shot reads, keeper-position refusal) and placement features
├── metrics.py     # log loss, Brier, AUC, calibration, thresholds, match bootstrap, Beta
│                  #   interval, per-group error and slope with intervals, ECE and its noise floor,
│                  #   split-half and Spearman-Brown
├── evaluation.py  # ModelSpec (guard, derived registry, xG input) / Benchmark, grouped CV,
│                  #   hold-out, compare, fold restriction, season-grouped folds: one harness for
│                  #   all, reading and writing the prediction cache
├── cards.py       # model-card rendering: shared sections, gain table, checks + per-model text;
│                  #   provider wording (ProviderText) and the quantity (xG / xGOT) are parameters
├── baselines.py   # M1.1 baselines (specs and card text), CLI that writes their cards
├── ablation.py    # M1.2 / M1.3 LR steps, drop-one / add-one variants, gain rows, step cards
├── models.py      # M1.4 GBM spec and card, headline rule, MS3 checks, summary table, CLI that
│                  #   evaluates every StatsBomb model and writes the cards
├── gbm.py         # M1.4 monotone histogram GBM: inner grouped-CV tuning, cross-fitted Platt
├── penalties.py   # M1.5 Beta-binomial penalty model, shoot-outs apart, penalty card
├── cache.py       # prediction cache under git-ignored data/cache, keyed by data digest + code
│                  #   (render-only and orchestration-only modules are left out of the code hash)
├── legacy.py      # M0 legacy v0 xG: SHA-256 pin, its own inputs, coverage, refit without a flag
├── v1.py          # M1.6 WhoScored run: season split, candidates, v0 comparison, checks, CLI
├── release.py     # M1.7 xg-v1 card and README block, miss reasons from two counterfactuals
├── xgot.py        # MS5 xGOT of on-target shots, both providers: population and reconciliation,
│                  #   keeper-neutral cross-fitted xG input, LR / GBM / penalty xGOT, checks,
│                  #   band table, shooting goals added; pure fit -> XgotFit, coarse-grid
│                  #   population filter, season-out refits (MS6)
├── xgot_cards.py  # MS6 render-only: the xGOT cards from an XgotFit, README block, CLI
├── gsax.py        # MS6 P3.1-P3.3: season-out xGOT, league-sum rating, keeper tables with SD and
│                  #   match-bootstrap CI, hierarchical keeper model, next-season test, stability;
│                  #   pure gsax.fit -> GsaxFit
└── gsax_cards.py  # MS6 render-only: GSAx and keeper-model cards, StatsBomb keeper chart (git-
                   #   ignored, with the logo), README block, CLI
```

| Module | Status | Milestone |
|---|---|---|
| `data` | ✅ Done | M1.0 / M1.1, MS3 (matches join, direct corners, freeze frames, penalties) |
| `features` | ✅ Done | M1.0, M1.2 / M1.3 (derived features), MS5 (xGOT guard, placement) |
| `metrics` | ✅ Done | M1.1, MS3 (bootstrap gain, Beta interval, checks), MS5 (ECE, public `logit`), MS6 (`spearman_brown`, empty `split_half`) |
| `evaluation` | ✅ Done | M1.1, MS3 (`compare`, `DesignFit`), MS5 (`ModelSpec.derived` / `xg_input`, `restrict_folds`), MS6 (`season_folds`) |
| `cards` | ✅ Done | M1.1, MS3 (gain table, checks, slope columns), MS5 (`ProviderText`, quantity, optional benchmark) |
| `baselines` | ✅ Done | M1.1 |
| `ablation` | ✅ Done | M1.2 / M1.3 |
| `models` | ✅ Done | M1.4 / M1.5 (GBM and penalties split out in MS4), MS5 (`select_headline`), MS6 (`evaluate_candidates`, one candidate set) |
| `gbm` | ✅ Done (moved from `models`, no behaviour change) | M1.4, MS4 |
| `penalties` | ✅ Done (moved from `models`, no behaviour change) | M1.5, MS4 |
| `cache` | ✅ Done | MS4, MS5 (input digest in the identity), MS6 (`RENDER_ONLY`, `ORCHESTRATION_ONLY`) |
| `legacy` | ✅ Done | M0 (MS4) |
| `v1` | ✅ Done | M1.6 (MS4) |
| `release` | ✅ Done | M1.7 (MS4), MS5 (`WHOSCORED_TEXT`) |
| `xgot` | ✅ Done | P2.2–P2.5 (MS5); pure `fit`, band rule, coarse-grid filter, season-out refits (MS6) |
| `xgot_cards` | ✅ Done (render-only, split from `xgot`) | MS6 |
| `gsax` | ✅ Done | P3.1–P3.3 (MS6) |
| `gsax_cards` | ✅ Done (render-only) | P3.1–P3.3 (MS6) |

Shared loading, the canonical shot table, snapshots and charts come from `football-core`
(`io.statsbomb`, `schema`, `io.snapshot`, `viz`).

Reproduce the M1.1, MS3, MS4, MS5 and MS6 results:

```bash
make snapshot        # uv run fc snapshot statsbomb-open-v1 --builder statsbomb ...
make xg-baselines    # uv run python -m football_shot_models.baselines -> the 3 M1.1 cards
make xg-models       # uv run python -m football_shot_models.models -> step, GBM and penalty
                     # cards (about 10 min cold, about 20 s from the prediction cache)
make snapshot-whoscored WHOSCORED_ARCHIVE=<archive>/data   # private; data and manifest ignored
make legacy-v0 LEGACY_V0=<path>/xg_model.json              # private; copied to data/legacy/
make xg-v1           # uv run python -m football_shot_models.v1 -> docs/models/xg-v1.md and the
                     # README block (about 8-10 min cold)
make xgot            # uv run python -m football_shot_models.xgot_cards --provider both -> six
                     # xGOT cards and the README block; run after xg-models and xg-v1, which
                     # fill the cache with the xG inputs (about 30 min cold)
make statsbomb-logo  # once (needs the network): the logo every StatsBomb chart carries,
                     # into the git-ignored data/raw cache
make gsax            # uv run python -m football_shot_models.gsax_cards --provider both -> four
                     # goalkeeper cards, the README block and the git-ignored StatsBomb keeper
                     # chart data/charts/gsax-keepers.svg; run after xgot
make reproduce       # regenerate every card and the README blocks and compare them byte for
                     # byte (fails, rather than skips, without either snapshot or v0); the xGOT
                     # tests run last and reuse the xG entries the earlier files cached
make reproduce-public  # the StatsBomb cards only, without the private data
make reproduce-cold  # the reproduction of record: FSM_CACHE=off, every fit runs (needs the
                     # private data, as reproduce does)
make reproduce-public-cold  # the cold public reproduction: FSM_CACHE=off reproduce-public
```

The prediction cache (`data/cache`, git-ignored) is keyed by the snapshot's data digest and a
hash of the code of both packages, so any `.py` change in either one makes the next run cold.
