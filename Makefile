.PHONY: check lint format typecheck test docs docs-notebooks docs-serve snapshot xg-baselines xg-models reproduce

## Run every quality gate (CI runs the same target)
check: lint typecheck test docs

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src

test:
	uv run pytest

## Render executed notebooks to Markdown so Zensical can include them
docs-notebooks:
	@if ls notebooks/*.ipynb >/dev/null 2>&1; then \
		uv run jupyter nbconvert --to markdown --output-dir docs/notebooks notebooks/*.ipynb; \
	fi

docs: docs-notebooks
	uv run zensical build --strict

docs-serve: docs-notebooks
	uv run zensical serve

## Freeze StatsBomb open data (football-core's raw checkout) as the 'statsbomb-open-v1' snapshot
snapshot:
	uv run fc snapshot statsbomb-open-v1 --builder statsbomb --input ../football-core/data/raw/statsbomb/open-data

## Evaluate the M1.1 xG baselines on the snapshot and write docs/models/xg-*.md
xg-baselines:
	uv run python -m football_shot_models.baselines

## Evaluate every xG model (baselines, M1.2/M1.3 steps, M1.4 GBM) and the M1.5 penalty model;
## write the step, GBM and penalty cards (about 8 min)
xg-models:
	uv run python -m football_shot_models.models

## Regenerate the model cards from the real snapshot and compare them with the committed ones.
## Without the snapshot this fails: a skipped comparison must not look like a reproduction.
reproduce:
	@test -d data/snapshots/statsbomb-open-v1 || { \
		echo "reproduce: no snapshot at data/snapshots/statsbomb-open-v1; run make snapshot first" >&2; \
		exit 1; }
	uv run pytest -m slow
