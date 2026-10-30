.PHONY: check lint format typecheck test docs docs-notebooks docs-serve snapshot xg-baselines xg-models reproduce \
	reproduce-cold reproduce-public reproduce-public-cold snapshot-whoscored legacy-v0 xg-v1 xg-v2 xgot \
	statsbomb-logo gsax charts charts-private docs-charts docs-theme docs-results

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

## The strict site build, then fail on any Sphinx role left raw in the built pages, a stale home
## results block or a stale generated theme (the Pages workflow deploys what this target checks)
docs: docs-notebooks
	uv run zensical build --strict
	uv run python -m football_docs.pages site site
	uv run python -m football_docs.pages results docs/index.md
	uv run python -m football_docs.theme docs --check

## Regenerate the committed site theme (docs/stylesheets/theme.css, docs/assets/logo.svg) from
## football-core's viz themes
docs-theme:
	uv run python -m football_docs.theme docs

## Rewrite the home page's results block from the README's card blocks ([[tool.football_docs.results]])
docs-results:
	uv run python -m football_docs.pages results docs/index.md --write

docs-serve: docs-notebooks
	uv run zensical serve

## Rebuild a snapshot after a schema bump with OVERWRITE=--overwrite (default: refuse to replace)
OVERWRITE ?=

## football-core's StatsBomb open-data cache: the snapshot input and the lineups (keeper names);
## the one place this repo names the sibling checkout
STATSBOMB_RAW ?= ../football-core/data/raw/statsbomb/open-data
export STATSBOMB_RAW  # the slow tests read it (gsax_cards defaults to this repo's own cache)

## Freeze StatsBomb open data (football-core's raw checkout) as the 'statsbomb-open-v1' snapshot
snapshot:
	uv run fc snapshot statsbomb-open-v1 --builder statsbomb --input "$(STATSBOMB_RAW)" $(OVERWRITE)

## Fetch the StatsBomb logo once (needs the network) into the git-ignored data/raw/statsbomb cache;
## every StatsBomb chart carries it and it is never committed
statsbomb-logo:
	uv run python -c "from football_core.io import statsbomb; print(statsbomb.fetch_logo())"

## The logo and the StatsBomb checkout must be there before any StatsBomb chart or keeper name
define NEED_STATSBOMB_EXTRAS
	@uv run python -c "import sys; from football_core.io import statsbomb; \
	sys.exit(0 if statsbomb.logo_file().is_file() else 1)" || { echo "$@: no StatsBomb logo; \
	run make statsbomb-logo first; it needs the network once" >&2; exit 1; }
	@test -d "$(STATSBOMB_RAW)" || { echo "$@: no StatsBomb checkout at $(STATSBOMB_RAW) (the \
	lineups name the keepers); set STATSBOMB_RAW" >&2; exit 1; }
endef

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

## PR1: WhoScored xG v2 (v1 plus pre-shot context); write docs/models/xg-v2.md. Run after
## `make xg-v1` (v1's candidates come from the cache); exits 1 when a PR1 check is missed
xg-v2:
	uv run python -m football_shot_models.v2_cards

## Evaluate the M1.1 xG baselines on the snapshot and write docs/models/xg-*.md
xg-baselines:
	uv run python -m football_shot_models.baselines

## Evaluate every xG model (baselines, M1.2/M1.3 steps, M1.4 GBM) and the M1.5 penalty model;
## write the step, GBM and penalty cards (about 10 min). Fills the prediction cache data/cache.
xg-models:
	uv run python -m football_shot_models.models

## Evaluate the MS5 / MS6 xGOT models of both providers and write docs/models/xgot-*.md and
## ws-xgot-*.md (about 30 min cold). No Make prerequisites (xg-models / xg-v1 are phony and would re-run):
## run `make xg-models xg-v1 xgot` so the xG candidates come from the prediction cache.
xgot:
	uv run python -m football_shot_models.xgot_cards --provider both

## Rate the goalkeepers (MS6 P3.1-P3.3) of both providers from season-out xGOT refits; write
## docs/models/gsax.md, gk-model.md, ws-gsax.md, ws-gk-model.md and the git-ignored StatsBomb
## keeper charts data/charts/gsax-keepers-light.png and -dark.png (CH1). Run after `make xgot`
## (the xGOT fits come from the cache).
## GSAX_DOCS=<dir> writes the cards elsewhere, so the chart can be made (e.g. on a fresh clone)
## without touching the committed cards; the StatsBomb card carries the chart's SHA-256.
GSAX_DOCS ?= docs/models
gsax:
	$(NEED_STATSBOMB_EXTRAS)
	uv run python -m football_shot_models.gsax_cards --provider both --statsbomb-raw "$(STATSBOMB_RAW)" \
		--docs-dir "$(GSAX_DOCS)"

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
	$(NEED_STATSBOMB_EXTRAS)
	uv run fc goalmouth --check manifests/statsbomb-open-v1.json manifests/whoscored-v1.json >/dev/null
	uv run pytest -m slow

## The public part only: the StatsBomb cards, without the private WhoScored snapshot and v0
reproduce-public:
	@test -d data/snapshots/statsbomb-open-v1 || { \
		echo "reproduce-public: no snapshot at data/snapshots/statsbomb-open-v1; run make snapshot \
		first" >&2; exit 1; }
	$(NEED_STATSBOMB_EXTRAS)
	uv run fc goalmouth --check manifests/statsbomb-open-v1.json >/dev/null
	uv run pytest -m "slow and not whoscored"

## The reproduction of record: every fit runs, the prediction cache is neither read nor written
reproduce-cold:
	FSM_CACHE=off $(MAKE) reproduce

## The public reproduction of record (no private data): reproduce-public with every fit run
reproduce-public-cold:
	FSM_CACHE=off $(MAKE) reproduce-public

## CH1: the themed team shot chart and player shot profile of CHART_TEAM (PL 2015/16 hold-out,
## both themes and sizes; the player is the team's most-shooting one, PM E2), the keeper (GSAx)
## chart and the xG / xGOT calibration and model-comparison charts, both themes: 18 StatsBomb
## PNGs into the git-ignored outputs/charts. Everything is read from the warm cache, never
## fitted: run make xg-models xgot gsax first.
CHART_TEAM ?= sb:22
CHART_TEAM_NAME ?= Leicester City
CHARTS := uv run python -m football_shot_charts.render
charts:
	$(NEED_STATSBOMB_EXTRAS)
	$(CHARTS) --team "$(CHART_TEAM)" --team-name "$(CHART_TEAM_NAME)" \
		--statsbomb-raw "$(STATSBOMB_RAW)" --out outputs/charts

## CH1, private: the WhoScored keeper, calibration and comparison charts (10 PNGs, unnamed) into
## the git-ignored data/charts/private; never committed or published. Run after make xg-v1 xgot
## gsax (warm cache).
charts-private:
	@test -d data/snapshots/whoscored-v1 && test -f manifests/whoscored-v1.json || { \
		echo "charts-private: no WhoScored snapshot (data/snapshots/whoscored-v1 and its \
		manifest); run make snapshot-whoscored first" >&2; exit 1; }
	@git check-ignore -q data/charts/private; rc=$$?; test $$rc -eq 0 || { echo \
	"charts-private: data/charts/private is not git-ignored (git check-ignore exit $$rc); \
	refusing to write there" >&2; exit 1; }
	$(CHARTS) --private

## Copy the published StatsBomb charts (team and player squares, keeper, calibration and
## comparison, both themes: 14 PNGs) and their charts.json from outputs/charts into the docs
## site, byte for byte; refuses a chart without the StatsBomb logo. Run after make charts.
docs-charts:
	$(CHARTS) --publish docs/assets/charts
