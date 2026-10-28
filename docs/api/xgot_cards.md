# xgot_cards

MS6: the xGOT cards, rendered from an `xgot.XgotFit`.

- **Render.** `render(fit)` returns the three cards of one provider (LR, GBM, penalty xGOT) as
  Markdown: no fit, no write, no print. `band_lines` renders `xgot.band_table` with its SD and
  the *Within 2 SD* column; `band_limit` adds the limitation line of a non-headline card whose
  bands fail the rule (PM E1: the rule judges the headline only).
- **Run.** `run(provider, *, docs_dir, cache)` is `xgot.fit`, `render`, `write_cards` and the
  printed summary; `summary_block` gives the README block.
- **Cache.** The module is render-only (`cache.RENDER_ONLY`): no fit-side module imports it, so
  editing it keeps every cached prediction warm.
- **CLI.** `python -m football_shot_models.xgot_cards --provider statsbomb|whoscored|both`
  (`make xgot`).

::: football_shot_models.xgot_cards
