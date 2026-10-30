---
hide:
  - navigation
  - toc
---

# football-shot-models

Three shot models for football, built step by step and documented with model cards: expected
goals (xG) before the shot, post-shot expected goals (xGOT / PSxG) of shots on target, and
goalkeeper shot-stopping rated against them. They train on StatsBomb open data, which is public,
and on a private WhoScored / Opta event archive, of which only counts and metrics are published.

<div class="fd-home-grid" markdown>

[![Team shot map: 54 of 57 non-penalty goals came from inside the box, the league's highest share (rest of league: 84%) (Leicester City · Premier League 2015/2016)](assets/charts/team-leicester-city-light-square.png#only-light){ loading=lazy }](charts.md#team-shot-map)
[![Team shot map: 54 of 57 non-penalty goals came from inside the box, the league's highest share (rest of league: 84%) (Leicester City · Premier League 2015/2016)](assets/charts/team-leicester-city-dark-square.png#only-dark){ loading=lazy }](charts.md#team-shot-map)

[![Keeper GSAx interval chart: None of 97 keeper-seasons stands out from average once luck is accounted for (At 100 shots keepers can't be told apart from noise: reliability r +0.12 \[−0.28, +0.40\] on the first 100 shots of 111 keepers)](assets/charts/gsax-keepers-light.png#only-light){ loading=lazy }](charts.md#keeper-gsax-chart)
[![Keeper GSAx interval chart: None of 97 keeper-seasons stands out from average once luck is accounted for (At 100 shots keepers can't be told apart from noise: reliability r +0.12 \[−0.28, +0.40\] on the first 100 shots of 111 keepers)](assets/charts/gsax-keepers-dark.png#only-dark){ loading=lazy }](charts.md#keeper-gsax-chart)

</div>

Data: StatsBomb open data; charts drawn from cached predictions ([more charts](charts.md)).

## Headline results

<!-- results:start -->
| Result | Measured |
|---|---|
| xG, monotone GBM + Platt trained without team-selected seasons, StatsBomb open data: log loss | CV log loss: 0.2641<br>Hold-out log loss: 0.2533 |
| StatsBomb's own xG on the same shots: log loss | CV log loss: 0.2659<br>Hold-out log loss: 0.2520 |
| xG v1, monotone GBM + Platt, private WhoScored / Opta archive: log loss | CV log loss: 0.2724<br>Test log loss: 0.2771 |
| xGOT, monotone GBM + Platt, StatsBomb open data: AUC | CV AUC: 0.8672<br>Hold-out / test AUC: 0.8685 |
| xGOT, monotone GBM + Platt, private WhoScored / Opta archive: AUC | CV AUC: 0.8435<br>Hold-out / test AUC: 0.8475 |
| Keeper model, StatsBomb open data: next-season log-loss gain of the shrunken keeper estimates | Next season: gain over raw: +0.03759 [+0.02901, +0.04714]<br>Gain over xGOT alone: +0.00097 [-0.00018, +0.00214] |
| Keeper model, private WhoScored / Opta archive: next-season log-loss gain of the shrunken keeper estimates | Next season: gain over raw: +0.01034 [+0.00502, +0.01619]<br>Gain over xGOT alone: +0.00003 [-0.00010, +0.00017] |

StatsBomb open data: Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and are not distinguishable from xGOT alone (the league average).

Without team-selected competition-seasons (22 pairs) the gain over raw is +0.01194 [-0.00019, +0.02517], its interval not above 0, and over xGOT alone +0.00050 [-0.00047, +0.00148], its interval not above 0.

private WhoScored / Opta event archive: Shrunken estimates predict next season better than raw GSAx, because raw GSAx overfits small samples, and are not distinguishable from xGOT alone (the league average).

<!-- results:end -->

Log loss: lower is better; AUC and a log-loss gain: higher is better; brackets hold the cards'
intervals. Copied by code from the README's card blocks, checked against the [model cards](models/xg-gbm.md).

## Install and reproduce

`uv sync` installs [football-core](https://github.com/rberkkaratas/football-core) (shared loaders, schema, pitch
frame and chart theme) from GitHub at a pinned tag; then:

```bash
uv sync --all-groups      # runtime, dev and docs dependencies, football-core included
make check                # ruff, mypy, pytest with doctests, strict docs build
make snapshot             # public chain: freeze StatsBomb open data as a verified snapshot
make statsbomb-logo       # once (network): the logo every StatsBomb chart carries
make reproduce-public     # refit and regenerate the StatsBomb cards, compared byte for byte
make charts               # the StatsBomb example charts, from the warm prediction cache
make docs-charts          # copy the published charts into this site, logo checked
make reproduce            # every card: the WhoScored ones need the private archive
```

## Read next

<div class="grid cards" markdown>

- [**Charts**](charts.md): the StatsBomb example charts and how to read them.
- [**Data**](data.md): sources, the shot table, filters and the hold-out.
- [**Methodology**](methodology.md): how each model is defined and evaluated.
- [**Model cards**](models/xg-gbm.md): one card per model, with checks and limitations.
- [**Modules**](modules.md): the package layout and what each module does.
- [**Roadmap**](roadmap.md): the milestones, in order, with their results.
- [**References**](references.md): the sources behind the models.
- [**API reference**](api/data.md): every public function, with examples.

</div>
