# Modules

```
src/football_shot_models/
├── data.py        # the xG population: verified snapshot (StatsBomb or WhoScored builder),
│                  #   exclusion rules and log, families, hold-out (pair or season split),
│                  #   freeze frames and the penalty kicks of the kept matches
├── features.py    # leakage guard (shots + matches pre_shot roles, restricted readers),
│                  #   derived features (splines, phase, contact, assist, freeze frames), the
│                  #   ablation guard for big chance, design matrix; the xGOT guard
│                  #   (declared post-shot reads, keeper-position refusal) and placement features
├── metrics.py     # log loss, Brier, AUC, calibration, thresholds, match bootstrap, Beta
│                  #   interval, per-group error and slope with intervals, ECE and its noise floor,
│                  #   split-half and Spearman-Brown
├── evaluation.py  # ModelSpec (guard, derived registry, xG input) / Benchmark, grouped CV,
│                  #   hold-out, compare, fold restriction, season-grouped folds: one harness for
│                  #   all, reading and writing the prediction cache
├── cards.py       # model-card rendering: shared sections, gain table, checks + per-model text;
│                  #   provider wording (ProviderText) and the quantity (xG / xGOT) are parameters
├── baselines.py   # the three baselines (specs and card text), CLI that writes their cards
├── ablation.py    # LR steps, drop-one / add-one variants, gain rows, step cards
├── models.py      # GBM spec and card, headline rule, xG checks, summary table, CLI that
│                  #   evaluates every StatsBomb model and writes the cards; the two
│                  #   hold-out level fixes and the adopted league-trained GBM
├── level.py       # orchestration-only: hold-out level probe, adoption rule, segment
│                  #   decomposition and its card lines (fits nothing)
├── gbm.py         # monotone histogram GBM: inner grouped-CV tuning, cross-fitted Platt
├── penalties.py   # Beta-binomial penalty model, shoot-outs apart, penalty card
├── cache.py       # prediction cache under git-ignored data/cache, keyed by data digest + code
│                  #   (render-only and orchestration-only modules are left out of the code hash)
├── legacy.py      # legacy v0 xG: SHA-256 pin, its own inputs, coverage, refit without a flag
├── v1.py          # WhoScored run: season split, candidates, v0 comparison, checks, CLI
├── release.py     # xg-v1 card and README block, miss reasons from two counterfactuals
├── v2.py          # WhoScored xG v2: v1 plus pre-shot context, qualifier census, checks;
│                  #   pure fit -> V2Fit
├── v2_cards.py    # render-only: the xg-v2 card and README block, CLI
├── xgot.py        # xGOT of on-target shots, both providers: population and reconciliation,
│                  #   keeper-neutral cross-fitted xG input, LR / GBM / penalty xGOT, checks,
│                  #   band table, shooting goals added; pure fit -> XgotFit, coarse-grid
│                  #   population filter
├── season_out.py  # (moved from xgot) season-out refits on competition-season folds for
│                  #   both providers, trained-season pairs, the z-free refit
├── xgot_cards.py  # render-only: the xGOT cards from an XgotFit, README block, CLI
├── gsax.py        # season-out xGOT, league-sum rating, keeper tables with SD and
│                  #   match-bootstrap CI, hierarchical keeper model, next-season test, stability;
│                  #   pure gsax.fit -> GsaxFit; penalty δ per source, team-selected
│                  #   seasons at δ = 0, goal_z sensitivity (1 yd line subset, z-free refit)
└── gsax_cards.py  # render-only: GSAx and keeper-model cards, README block, CLI; the keeper
                   #   charts (light, dark; git-ignored, with the logo) and their digests

src/football_shot_charts/   # render-only charts, a sibling package outside the cache hash
├── rows.py        # the row panel all charts share (dot, tick, band with caps), scale and tick
│                  #   helpers, write_all: the one writer (Inter, git-ignored path, WhoScored
│                  #   private, viz.check)
├── shots.py       # team shot chart and player shot profile: zones, goals against xG with the
│                  #   league-rate band, computed statement titles, stat strip, model level
├── keepers.py     # keeper (GSAx) chart: top and bottom keeper-seasons with intervals
├── diagnostics.py # model-card visuals: calibration diagram, log-loss gain comparison
└── render.py      # warm-cache reads (no fit), views, file names, `make charts` CLI,
                   #   charts.json titles, `make docs-charts` publishing (logo checked)
```

| Module | Status | Milestones |
|---|---|---|
| `data` | ✅ Done | StatsBomb xG baseline; xG model (matches join, direct corners, freeze frames, penalties) |
| `features` | ✅ Done | StatsBomb xG baseline; xG model (derived features); xGOT (xGOT guard, placement); polish round 1 (`KEEPER_DERIVED` refused by the xGOT guards) |
| `metrics` | ✅ Done | StatsBomb xG baseline; xG model (bootstrap gain, Beta interval, checks); xGOT (ECE, public `logit`); goalkeepers (`spearman_brown`, empty `split_half`); polish round 1 (`MIN_JUDGED_SHOTS`; `BAND_SDS`, `poisson_binomial`, `within`: the one band rule) |
| `evaluation` | ✅ Done | StatsBomb xG baseline; xG model (`compare`, `DesignFit`); xGOT (`ModelSpec.derived` / `xg_input`, `restrict_folds`); goalkeepers (`season_folds`); polish round 1 (`ModelSpec.train_exclude` / `train_indicator`) |
| `cards` | ✅ Done | StatsBomb xG baseline; xG model (gain table, checks, slope columns); xGOT (`ProviderText`, quantity, optional benchmark) |
| `baselines` | ✅ Done | StatsBomb xG baseline |
| `ablation` | ✅ Done | xG model |
| `models` | ✅ Done | xG model (GBM and penalties split out later); xGOT (`select_headline`); goalkeepers (`evaluate_candidates`, one candidate set); polish round 1 (`LEVEL_FIXES`, `ADOPTED`, league-trained headline GBM) |
| `level` | ✅ Done (orchestration-only) | polish round 1 |
| `gbm` | ✅ Done (moved from `models`, no behaviour change) | xG model; WhoScored xG v1 |
| `penalties` | ✅ Done (moved from `models`, no behaviour change) | xG model; WhoScored xG v1 |
| `cache` | ✅ Done | WhoScored xG v1; xGOT (input digest in the identity); goalkeepers (`RENDER_ONLY`, `ORCHESTRATION_ONLY`); polish round 1 (`v2_cards` render-only, `level` orchestration-only) |
| `legacy` | ✅ Done | WhoScored xG v1 |
| `v1` | ✅ Done | WhoScored xG v1 |
| `release` | ✅ Done | WhoScored xG v1; xGOT (`WHOSCORED_TEXT`) |
| `v2` | ✅ Done | polish round 1 |
| `v2_cards` | ✅ Done (render-only) | polish round 1 |
| `xgot` | ✅ Done | xGOT; goalkeepers (pure `fit`, band rule, coarse-grid filter); polish round 1 (season-out block moved to `season_out`) |
| `season_out` | ✅ Done (moved from `xgot`; competition-season folds for both providers, z-free refit) | goalkeepers; polish round 1 |
| `xgot_cards` | ✅ Done (render-only, split from `xgot`) | goalkeepers |
| `gsax` | ✅ Done | goalkeepers; polish round 1 (penalty δ per source, team-selected seasons at δ = 0, goal_z sensitivity, `assemble`, `season_flags`) |
| `gsax_cards` | ✅ Done (render-only) | goalkeepers; polish round 1 (card v2); chart round 1 (keeper charts in both themes, PNG digests on the card) |
| `football_shot_charts.rows` | ✅ Done (outside the hash) | chart round 1; docs round 1 (`logo_pixels`) |
| `football_shot_charts.shots` | ✅ Done (outside the hash) | chart round 1 |
| `football_shot_charts.keepers` | ✅ Done (outside the hash) | chart round 1 |
| `football_shot_charts.diagnostics` | ✅ Done (outside the hash) | chart round 1 |
| `football_shot_charts.render` | ✅ Done (outside the hash; `make charts`, `make charts-private`, `make docs-charts`) | chart round 1; docs round 1 (`charts.json`, `publish`) |

Shared loading, the canonical shot table, snapshots and charts come from `football-core`
(`io.statsbomb`, `schema`, `io.snapshot`, `viz`).

Reproduce every result:

```bash
make snapshot        # uv run fc snapshot statsbomb-open-v1 --builder statsbomb ...
make xg-baselines    # uv run python -m football_shot_models.baselines -> the 3 baseline cards
make xg-models       # uv run python -m football_shot_models.models -> step, GBM and penalty
                     # cards (about 10 min cold, about 20 s from the prediction cache)
make snapshot-whoscored WHOSCORED_ARCHIVE=<archive>/data   # private; data and manifest ignored
make legacy-v0 LEGACY_V0=<path>/xg_model.json              # private; copied to data/legacy/
make xg-v1           # uv run python -m football_shot_models.v1 -> docs/models/xg-v1.md and the
                     # README block (about 8-10 min cold)
make xg-v2           # uv run python -m football_shot_models.v2_cards -> docs/models/xg-v2.md and
                     # the README v2 block; after xg-v1; exits 1 on a missed v2 check
make xgot            # uv run python -m football_shot_models.xgot_cards --provider both -> six
                     # xGOT cards and the README block; run after xg-models and xg-v1, which
                     # fill the cache with the xG inputs (about 30 min cold)
make statsbomb-logo  # once (needs the network): the logo every StatsBomb chart carries,
                     # into the git-ignored data/raw cache
make gsax            # uv run python -m football_shot_models.gsax_cards --provider both -> four
                     # goalkeeper cards, the README block and the git-ignored StatsBomb keeper
                     # charts data/charts/gsax-keepers-{light,dark}.png; run after xgot
make charts          # 18 StatsBomb charts (team shot chart and player profile of
                     # CHART_TEAM / --team-name, keeper chart, xG and xGOT calibration and
                     # comparison; light and dark) into the git-ignored outputs/charts, from the
                     # warm cache (no model of record refitted); run after xg-models, xgot
                     # and gsax
make docs-charts     # copy the 14 published StatsBomb charts and charts.json into
                     # docs/assets/charts, byte for byte, logo checked; after charts
make charts-private  # private: the 10 WhoScored charts (unnamed, no logo) into the
                     # git-ignored data/charts/private; never committed or published
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
