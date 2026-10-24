# football-shot-models

Building, evaluating and documenting **shot models** for football from first principles:

| Phase | Model | Question it answers |
|---|---|---|
| 1 | **xG**: expected goals | How likely was this shot to be scored *before* it was struck? |
| 2 | **xGOT / PSxG**: post-shot xG | Given where the shot ended up in the goal mouth, how likely was it to go in? |
| 3 | **Goals prevented / GSAx** | How many goals did a goalkeeper save compared with an average keeper facing the same shots? |

> **Status:** skeleton. No models are trained yet; see [`docs/roadmap.md`](docs/roadmap.md).

Part of the `football-*-models` family; shared loaders and pitch conventions come from
[football-core](../football-core).

## Data
- **StatsBomb open data**: public training data, plus a benchmark against StatsBomb's own xG.
- **WhoScored / Opta events**: private parquet snapshots from a local database, never
  published. See [`docs/data.md`](docs/data.md).

## Repository layout
```
src/football_shot_models/   # package: features, models, evaluation
notebooks/                  # exploration & narrative
docs/                       # methodology, model cards, roadmap (MkDocs site)
data/                       # git-ignored
tests/
```

## Development
```bash
uv sync --all-groups        # expects ../football-core next to this repo
uv run ruff check .
uv run pytest
uv run mkdocs serve
```

## License
MIT. See [LICENSE](LICENSE). Data is **not** covered by this licence.
