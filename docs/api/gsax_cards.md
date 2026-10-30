# gsax_cards

The goalkeeper cards and the keeper charts, rendered from a `gsax.GsaxFit`.

- **Render.** `render(g, *, names)` returns the `gsax` and `gk-model` cards (`ws-` prefixed for
  WhoScored) as Markdown: no fit, no write, no print. StatsBomb cards need the lineup names
  (`football_core.io.statsbomb.player_names`); WhoScored cards refuse them (the private archive
  publishes counts and metrics only). `stability_sentence` and `next_season_sentence` are the only places the cards
  word what the numbers allow.
- **Charts.** `keeper_charts(g, *, names, logo, team_selected)` draws the keeper chart
  (`football_shot_charts.keepers`) in both themes; `caveat(tier)` is its subtitle, the same
  verdict as `stability_sentence` (`stability_branch`). `write_charts` writes them to the
  git-ignored `CHART_DIR` (`data/charts/gsax-keepers-{light,dark}.png`) only when every one
  passes the write guard; the docs site publishes a byte-identical copy (`make docs-charts`),
  checked against the card's digests.
- **Run.** `run_from(f, *, docs_dir, raw)` is `gsax.fit`, names, render, write and the printed
  checks; `summary_block` gives the README block.
- **CLI.** `python -m football_shot_models.gsax_cards --provider statsbomb|whoscored|both
  --statsbomb-raw <open-data cache>` (`make gsax`, after `make xgot`). StatsBomb needs
  `--statsbomb-raw` pointing at a synced open-data cache (`<commit>/tree.json` and the
  lineups; the Makefile passes football-core's); without it the run stops before any fit.
  The StatsBomb GSAx card prints each chart's SHA-256, so `make reproduce` checks the
  git-ignored charts through the committed card; `make gsax GSAX_DOCS=<dir>` writes the charts
  without touching the committed cards.

::: football_shot_models.gsax_cards
