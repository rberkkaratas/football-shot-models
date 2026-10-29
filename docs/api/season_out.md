# season_out

MS6, moved here in PR1: season-out xGOT for the goalkeeper models. Each rated row's xGOT comes
from fits (xG input and xGOT both) that never saw its season.

- **Season-out refit.** `season_out(f)` refits the `xgot.fit` chain of one provider (`refit`:
  xG input, headline xGOT, penalty xGOT) on folds grouped by competition-season; dev rows read
  the fold that left their season out (`season_folds`), test-season rows the refit's dev fit
  (`holdout_fit`). `SEASON_OUT` is `season_folds` for both providers (PM 2026-10-28).
- **Z-free refit.** `season_out(f, without=("mouth_z",))` refits the open-play headline without
  the named features, the xG input reused through the cache (PM E2 (c), PR1): the goal_z
  sensitivity on the StatsBomb GSAx card.
- **Training seasons.** `fold_seasons(dev, held, folds)` lists, per rated season, the seasons
  that trained the fit that rated it (frame-level, any frames with `SEASON_KEY`);
  `trained_seasons(f)` is that list for the chain `season_out` refits. `gsax.saw_test` counts
  from it the next-season pairs whose rated season's xGOT saw the test season (PM E4).
- **Cache.** Hashed (it fits through `evaluate`); `gsax` calls it for every fit.

::: football_shot_models.season_out
