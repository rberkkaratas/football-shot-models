# gsax

Goals saved above expected, the hierarchical keeper model and stability, as pure
functions of an `xgot.XgotFit`.

- **Season-out xGOT.** `fit(f)` calls `season_out.season_out(f)` for every fit: each rated
  row's xGOT comes from fits that never saw its season (folds grouped by competition-season,
  both providers); `assemble(f, pred, *, stops, trained, noz)` is the arithmetic on those rows,
  so a `GsaxFit` can be built from frames without a refit.
- **Rating.** `rate(pred, small=..., zero_families=...)` shifts the logit of xGOT by one
  `league_shift` per rating group (`RATING_GROUPS`: a competition-season for open play, the
  season-out source for penalties: one penalty δ per source), so each group sums to its league (the
  xGOT level gap is not charged to keepers). `too_small` is the minimum group size on every
  grouping (`SMALL_LEAGUE`: a league offset from at most two keepers or fewer than 50 open-play
  shots would absorb the keepers' own performance, so such seasons are rated at δ = 0 and flagged); team-selected competition-seasons (`ZERO_FAMILIES`) are at δ = 0
  and flagged; `rating_groups` is the δ table.
- **Sensitivities (StatsBomb).** `line_rated` re-rates GSAx on goals and the saves stopped
  within `LINE_YD` of the line; `noz_rated` rates the z-free refit; `line_summary` and
  `shift_summary` compare each with the headline over the keeper-seasons whose
  competition-season has a δ fitted on the line subset (`season_flags`, `GsaxFit.line_seasons`);
  those at δ = 0 there cannot re-level that outcome-selected subset and are only counted.
- **Tables.** `keeper_table` (GSAx raw and rated, the null SD and a 90% match-bootstrap CI per
  keeper-season), `league_table`, `ci_counts` (CI against a no-difference null) and
  `keeper_rows(g)`, the card and chart rows.
- **Keeper model.** `fit_keeper_model(rated, by=KEEPER_SEASON | CAREER)`: `goal ~ offset(logit
  q) + (1 | group)` by a Laplace approximation (`keeper_modes`); `next_season` scores it
  against raw GSAx and xGOT alone on the next season.
- **Stability.** `reliability(rated)`: split-half by alternating shots within a keeper at
  `TIERS` total shots faced, empty tiers reported, never dropped.
- **Cache.** Hash-exempt (`cache.ORCHESTRATION_ONLY`): it fits only through `season_out.season_out`.

::: football_shot_models.gsax
