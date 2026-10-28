# xgot

MS5 / MS6: xGOT (post-shot xG) of on-target shots for both providers (P2.2–P2.5), fitted by a
pure `fit` (MS6); the cards are rendered by [`xgot_cards`](xgot_cards.md).

- **Population and reconciliation (P2.2).** `reconcile` puts every C3 on-target shot (a goal or
  a keeper save) in one row: an exclusion-log reason, an unresolved assist, or modelled. Shots
  and goals are counted from the rows actually scored and must agree with C3 within
  `RECONCILE_TOL`. Shoot-out kicks are a separately labelled sub-population, scored by the
  penalty xGOT.
- **xG input.** `xg_input_eval` evaluates the provider's xG candidates with
  `models.evaluate_candidates` (the set the xG run evaluates), picks the headline through
  `models.select_headline` and, for StatsBomb, refits it without the keeper-position features (`keeper_neutral`).
  `xg_input` adds its out-of-fold predictions as `xg_logit` and restricts the folds to the
  on-target rows, so an xGOT test row never reads an xG fitted on itself.
- **Models (P2.3, P2.4).** `xgot_specs` builds the base rate, the xG-only LR, the LR on xG +
  placement (with and without near / far post), the monotone GBM and the placement-only
  penalty xGOT. Every spec uses the xGOT guard, `features.check_xgot_features`, which reads the
  goal mouth only through the placement features and refuses keeper position.
- **Checks.** `checks` returns the eleven MS5 rows (AUC, ECE, calibration slope, the GBM
  headline rule, reconciliation, and the two guard-backed rows) and the MS6 band rule (row 12):
  every band of `band_table` of the headline's out-of-fold xGOT with at least `MIN_BAND_SHOTS`
  shots (`judged`) is within `BAND_SDS` Poisson-binomial SDs (`within`, the one rule the GSAx
  band table reuses). `ece_stats` is the one ECE rule behind checks, cards and README.
- **Shooting goals added (P2.5).** `predictions` holds one row per shot of the population with
  the xG input and the out-of-sample xGOT (built in memory, never written). `sga` gives
  xGOT − xG and goals − xG per shot; `metrics.split_half(by=, value=, unit="match")` gives the
  split-half correlation by alternating matches, a descriptive measure, not a skill claim.
  `predictions` carries `period` and `seconds_in_period`, so MS6 P3.3 calls
  `metrics.split_half(by="gk_id", value=..., unit="row")`, which alternates shots within a keeper
  in time order (PM X5).
- **Placement (MS6).** The GBM is monotone in the xG input only; its placement features are
  free, chosen by the band rule over a monotone width cut. `band_table` is the calibration by
  |y| band (Σ xGOT − goals, its SD and whether it is within the rule), pure, for cards and
  charts.
- **Fit (MS6).** `fit(provider, *, cache)` returns an `XgotFit` with every evaluation, the
  checks and `predictions`; it writes and prints nothing (a cache root is the only place it
  writes), and it refuses a StatsBomb snapshot whose `goal_y` is off the 0.1 yd goal-mouth
  lattice (built before the snap).
- **CLI.** `python -m football_shot_models.xgot_cards --provider statsbomb|whoscored|both`
  (`make xgot`) writes the six cards and prints the README block, which is pasted between the
  README markers by hand. `python -m football_shot_models.xgot` exits 2 and points there.

::: football_shot_models.xgot
