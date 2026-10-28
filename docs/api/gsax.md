# gsax

MS6 P3.1-P3.3: goals saved above expected, the hierarchical keeper model and stability, as pure
functions of an `xgot.XgotFit`.

- **Season-out xGOT.** `fit(f)` calls `xgot.season_out(f)` for every fit: each rated row's xGOT
  comes from fits that never saw its season (StatsBomb: folds grouped by competition-season;
  WhoScored: the train-only fit for the test season and a swapped fit for the train seasons).
- **Rating.** `rate(pred, small=...)` shifts the logit of xGOT by one `league_shift` per
  competition-season and kind, so each season's keepers sum to its league (the xGOT level gap
  is not charged to keepers); `SMALL_LEAGUE` (PM E2 (b)) rates tiny competition-seasons at 0.
- **Tables.** `keeper_table` (GSAx raw and rated, the null SD and a 90% match-bootstrap CI per
  keeper-season), `league_table`, `ci_counts` (CI against a no-difference null) and
  `keeper_rows(g)`, the card and chart rows.
- **Keeper model.** `fit_keeper_model(rated, by=KEEPER_SEASON | CAREER)`: `goal ~ offset(logit
  q) + (1 | group)` by a Laplace approximation (`keeper_modes`); `next_season` scores it
  against raw GSAx and xGOT alone on the next season.
- **Stability.** `reliability(rated)`: split-half by alternating shots within a keeper (PM X5)
  at `TIERS` shots faced, empty tiers kept.
- **Cache.** Hash-exempt (`cache.ORCHESTRATION_ONLY`): it fits only through `xgot.season_out`.

::: football_shot_models.gsax
