# gsax_cards

MS6: the goalkeeper cards and the StatsBomb keeper chart, rendered from a `gsax.GsaxFit`.

- **Render.** `render(g, *, names)` returns the `gsax` and `gk-model` cards (`ws-` prefixed for
  WhoScored) as Markdown: no fit, no write, no print. StatsBomb cards need the lineup names
  (`football_core.io.statsbomb.player_names`); WhoScored cards refuse them (Rule 1: counts and
  metrics only). `stability_sentence` and `next_season_sentence` are the only places the cards
  word what the numbers allow.
- **Chart.** `chart(table, *, names, logo)` draws the keeper forest plot with the StatsBomb logo
  as deterministic SVG; it is written to the git-ignored `CHART_PATH` (PM E3 (b)).
- **Run.** `run_from(f, *, docs_dir, raw)` is `gsax.fit`, names, render, write and the printed
  checks; `summary_block` gives the README block.
- **CLI.** `python -m football_shot_models.gsax_cards --provider statsbomb|whoscored|both
  --statsbomb-raw <open-data cache>` (`make gsax`, after `make xgot`). StatsBomb needs
  `--statsbomb-raw` pointing at a synced open-data cache (`<commit>/tree.json` and the
  lineups; the Makefile passes football-core's); without it the run stops before any fit.
  The StatsBomb GSAx card prints the chart's SHA-256, so `make reproduce` checks the
  git-ignored chart through the committed card; `make gsax GSAX_DOCS=<dir>` writes the chart
  without touching the committed cards.

::: football_shot_models.gsax_cards
