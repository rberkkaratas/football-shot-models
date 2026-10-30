# charts: render

CH1: reads fit outputs from the warm prediction cache (never fitting) and writes the
git-ignored charts. `cached_evaluation` / `holdout_xg` read the `xg-gbm` hold-out predictions
for the shot charts; `fits` and `candidates` run the existing `xgot.fit`, `gsax.fit` and
`models.evaluate_candidates` inside `read_only`, so a cold entry or a cache write is an error
(the one uncached fit is `xgot.fit`'s shoot-out penalty LR, PM E1 (a)). `calibration_bins`,
`xg_view` and `xgot_view` turn evaluations into the plain views the model-card charts draw.
`make charts` writes the 18 StatsBomb charts; `make charts-private` the 10 WhoScored ones
into `data/charts/private`.

::: football_shot_charts.render
