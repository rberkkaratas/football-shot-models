# xgot

MS5: xGOT (post-shot xG) of on-target shots for both providers (P2.2–P2.5).

- **Population and reconciliation (P2.2).** `reconcile` puts every C3 on-target shot (a goal or
  a keeper save) in one row: an exclusion-log reason, an unresolved assist, or modelled. Shots
  and goals are counted from the rows actually scored and must agree with C3 within
  `RECONCILE_TOL`. Shoot-out kicks are a separately labelled sub-population, scored by the
  penalty xGOT.
- **xG input.** `xg_input_eval` picks the provider's headline xG through `models.select_headline`
  and, for StatsBomb, refits it without the keeper-position features (`keeper_neutral`).
  `xg_input` adds its out-of-fold predictions as `xg_logit` and restricts the folds to the
  on-target rows, so an xGOT test row never reads an xG fitted on itself.
- **Models (P2.3, P2.4).** `xgot_specs` builds the base rate, the xG-only LR, the LR on xG +
  placement (with and without near / far post), the monotone GBM and the placement-only
  penalty xGOT. Every spec uses the xGOT guard, `features.check_xgot_features`, which reads the
  goal mouth only through the placement features and refuses keeper position.
- **Checks.** `checks` returns the eleven MS5 rows (AUC, ECE, calibration slope, the GBM
  headline rule, reconciliation, and the two guard-backed rows); `ece_stats` / `ece_lines` are
  the one ECE rule behind checks, cards and README.
- **Shooting goals added (P2.5).** `predictions` holds one row per shot of the population with
  the xG input and the out-of-sample xGOT (built in memory, never written). `sga` gives
  xGOT − xG and goals − xG per shot; `split_half(by=, value=)` gives the split-half correlation
  by alternating matches, a descriptive measure, not a skill claim. MS6 reuses both with
  `by="gk_id"`.
- **CLI.** `python -m football_shot_models.xgot --provider statsbomb|whoscored|both` (`make
  xgot`) writes the six cards and prints the README block.

::: football_shot_models.xgot
