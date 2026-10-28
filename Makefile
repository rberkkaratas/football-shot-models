.PHONY: check lint format typecheck test docs docs-notebooks docs-serve snapshot xg-baselines xg-models reproduce \
	reproduce-cold reproduce-public reproduce-public-cold snapshot-whoscored legacy-v0 xg-v1 xgot

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

## Rebuild a snapshot after a schema bump with OVERWRITE=--overwrite (default: refuse to replace)
OVERWRITE ?=

## Freeze StatsBomb open data (football-core's raw checkout) as the 'statsbomb-open-v1' snapshot
snapshot:
	uv run fc snapshot statsbomb-open-v1 --builder statsbomb --input ../football-core/data/raw/statsbomb/open-data $(OVERWRITE)

## Freeze the private WhoScored CSV archive as 'whoscored-v1' (WHOSCORED_ARCHIVE=<archive>/data).
## The data and the manifest are git-ignored; the manifest check fails closed (exit code only).
snapshot-whoscored:
	@test -n "$(WHOSCORED_ARCHIVE)" || { echo "snapshot-whoscored: WHOSCORED_ARCHIVE must point at \
	the archive's data/ folder (events/, match_ids/)" >&2; exit 1; }
	@git check-ignore -q manifests/whoscored-v1.json; rc=$$?; test $$rc -eq 0 || { echo \
	"snapshot-whoscored: manifests/whoscored-v1.json is not git-ignored (git check-ignore exit \
	$$rc); refusing to write it" >&2; exit 1; }
	uv run fc snapshot whoscored-v1 --builder whoscored --input "$(WHOSCORED_ARCHIVE)" $(OVERWRITE)

## Copy the legacy v0 xG (LEGACY_V0=<path>/xg_model.json) to the git-ignored data/legacy/
legacy-v0:
	@test -n "$(LEGACY_V0)" || { echo "legacy-v0: LEGACY_V0 must point at the legacy \
	xg_model.json" >&2; exit 1; }
	mkdir -p data/legacy
	cp "$(LEGACY_V0)" data/legacy/xg_model.json

## Evaluate the WhoScored xG v1 models against v0; write docs/models/xg-v1.md (about 8-10 min cold)
xg-v1:
	uv run python -m football_shot_models.v1

## Evaluate the M1.1 xG baselines on the snapshot and write docs/models/xg-*.md
xg-baselines:
	uv run python -m football_shot_models.baselines

## Evaluate every xG model (baselines, M1.2/M1.3 steps, M1.4 GBM) and the M1.5 penalty model;
## write the step, GBM and penalty cards (about 10 min). Fills the prediction cache data/cache.
xg-models:
	uv run python -m football_shot_models.models

## Evaluate the MS5 xGOT models of both providers and write docs/models/xgot-*.md and ws-xgot-*.md
## (about 30 min cold). No Make prerequisites (xg-models / xg-v1 are phony and would re-run):
## run `make xg-models xg-v1 xgot` so the xG candidates come from the prediction cache.
xgot:
	uv run python -m football_shot_models.xgot --provider both

## Regenerate the model cards from the real snapshot and compare them with the committed ones.
## First `fc goalmouth --check`: >= 99 % of each provider's goals in the frame, Opta direction confirmed.
## pytest -m slow runs test_models_run and test_v1_run before test_xgot_run, which warm the cache.
## Without the snapshot this fails: a skipped comparison must not look like a reproduction.
## Reads the prediction cache (data/cache) when data and code match; reproduce-cold does not.
reproduce:
	@test -d data/snapshots/statsbomb-open-v1 || { \
		echo "reproduce: no snapshot at data/snapshots/statsbomb-open-v1; run make snapshot first" >&2; \
		exit 1; }
	@test -d data/snapshots/whoscored-v1 && test -f manifests/whoscored-v1.json || { \
		echo "reproduce: no WhoScored snapshot (data/snapshots/whoscored-v1 and its manifest); run \
		make snapshot-whoscored first" >&2; exit 1; }
	@test -f data/legacy/xg_model.json || { \
		echo "reproduce: no legacy v0 at data/legacy/xg_model.json; run make legacy-v0 first" >&2; \
		exit 1; }
	uv run fc goalmouth --check manifests/statsbomb-open-v1.json manifests/whoscored-v1.json >/dev/null
	uv run pytest -m slow

## The public part only: the StatsBomb cards, without the private WhoScored snapshot and v0
reproduce-public:
	@test -d data/snapshots/statsbomb-open-v1 || { \
		echo "reproduce-public: no snapshot at data/snapshots/statsbomb-open-v1; run make snapshot \
		first" >&2; exit 1; }
	uv run fc goalmouth --check manifests/statsbomb-open-v1.json >/dev/null
	uv run pytest -m "slow and not whoscored"

## The reproduction of record: every fit runs, the prediction cache is neither read nor written
reproduce-cold:
	FSM_CACHE=off $(MAKE) reproduce

## The public reproduction of record (no private data): reproduce-public with every fit run
reproduce-public-cold:
	FSM_CACHE=off $(MAKE) reproduce-public
