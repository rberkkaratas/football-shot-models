# Charts

Example charts drawn from StatsBomb open data: the shot charts show the Premier League 2015/2016
hold-out season, which no xG model of record was trained on. Every chart is drawn from the
models' cached predictions, and no model of record is refitted to draw it (rendering re-runs
only two small existing estimations, the shoot-out penalty LR inside the xGOT fit and the keeper
model's estimation, both deterministic and disclosed in the [roadmap](roadmap.md) and the
[render module's reference](api/charts_render.md)). Every chart carries the StatsBomb logo and
attribution, as the open-data licence asks. Titles are computed from the data, never typed. Each
image opens the full-size PNG; the light or dark version follows the site's colour scheme.

The 14 published PNGs come from `make charts` and are copied byte for byte by `make
docs-charts`, which checks that each one carries the logo; together they are 2.92 MB, so
re-publishing them is a deliberate step. WhoScored charts are never published.

## Team shot map

Every non-penalty shot of one team on a pitch cropped to where it shot from: marker area is
proportional to the shot's xG, goals are filled in red and drawn above the hollow misses. The
strip gives the season totals. The panel compares goals (red dot) with xG (vertical tick) per zone,
with the band of what the rest of the league scores from the same xG in that zone; a zone with
under 100 shots is printed but marked "too few to judge". Goals − xG is printed with its normal
range (± 2 SD of the model's own noise). Numbers come from the [xG GBM card](models/xg-gbm.md)'s
model and the hold-out predictions.

[![Team shot map: 54 of 57 non-penalty goals came from inside the box, the league's highest share (rest of league: 84%) (Leicester City · Premier League 2015/2016)](assets/charts/team-leicester-city-light-square.png#only-light){ loading=lazy }](assets/charts/team-leicester-city-light-square.png)
[![Team shot map: 54 of 57 non-penalty goals came from inside the box, the league's highest share (rest of league: 84%) (Leicester City · Premier League 2015/2016)](assets/charts/team-leicester-city-dark-square.png#only-dark){ loading=lazy }](assets/charts/team-leicester-city-dark-square.png)

## Player shot map

The same marks for the team's player with the most kept shots (chosen without reading
outcomes), with the panel split by body part. The title says whether goals differ from xG by more
than the model's own noise band, and never why: the model's known misprice by zone can explain a
gap as well as finishing can. See the [xG GBM card](models/xg-gbm.md).

[![Player shot map: 19 non-penalty goals from 19.2 xG: in line with the model (Jamie Vardy · Leicester City · Premier League 2015/2016)](assets/charts/player-jamie-vardy-light-square.png#only-light){ loading=lazy }](assets/charts/player-jamie-vardy-light-square.png)
[![Player shot map: 19 non-penalty goals from 19.2 xG: in line with the model (Jamie Vardy · Leicester City · Premier League 2015/2016)](assets/charts/player-jamie-vardy-dark-square.png#only-dark){ loading=lazy }](assets/charts/player-jamie-vardy-dark-square.png)

## Keeper (GSAx) chart

Only the highest and lowest keeper-seasons are drawn, because the reliability study says a
ranked list of every season would be read as skill it does not show. On each row the red dot is
the keeper model's shrunken ("luck-adjusted") estimate of goals saved above expected per 100
on-target shots, the band its posterior interval and the tick the unshrunk ("raw") value; right of 0 means fewer goals conceded
than the model expects of an average keeper facing the same shots. A starred row is a
team-selected season, rated against the model rather than its league, with the gap disclosed in
the footer. Numbers come from the [GSAx card](models/gsax.md), which also carries these two
PNGs' digests, and the [keeper-model card](models/gk-model.md).

[![Keeper GSAx interval chart: None of 97 keeper-seasons stands out from average once luck is accounted for (At 100 shots keepers can't be told apart from noise: reliability r +0.12 \[−0.28, +0.40\] on the first 100 shots of 111 keepers)](assets/charts/gsax-keepers-light.png#only-light){ loading=lazy }](assets/charts/gsax-keepers-light.png)
[![Keeper GSAx interval chart: None of 97 keeper-seasons stands out from average once luck is accounted for (At 100 shots keepers can't be told apart from noise: reliability r +0.12 \[−0.28, +0.40\] on the first 100 shots of 111 keepers)](assets/charts/gsax-keepers-dark.png#only-dark){ loading=lazy }](assets/charts/gsax-keepers-dark.png)

## Calibration

A reliability diagram on logit axes: each bin of predictions is a marker at its mean prediction
and observed goal rate, cross-validation as hollow rings and the hold-out as red dots, each with
the match-bootstrap band of the observed rate. A calibrated model keeps the dots on the dashed
diagonal; the title counts the bins whose band holds the prediction ("agree") and calls the
model well calibrated only when each split misses at most one bin and the hold-out total of
goals against xG stays inside its band. Numbers come from the
[xG GBM card](models/xg-gbm.md) and the [xGOT GBM card](models/xgot-gbm.md).

The hold-out chose the adopted xG fix (training without team-selected seasons), so the xG
chart's hold-out total of goals against xG is not an independent check of the model's level;
see the [xG GBM card](models/xg-gbm.md). Hold-out log loss is unaffected: both candidates score
0.2533.

[![xG calibration chart: xG is well calibrated: 10 of 10 bins agree in cross-validation, 9 of 10 on the hold-out season (Shots in 10 equal-size bins · a bin agrees when the 95% match-bootstrap band of its goal rate holds the prediction)](assets/charts/calibration-xg-light.png#only-light){ loading=lazy }](assets/charts/calibration-xg-light.png)
[![xG calibration chart: xG is well calibrated: 10 of 10 bins agree in cross-validation, 9 of 10 on the hold-out season (Shots in 10 equal-size bins · a bin agrees when the 95% match-bootstrap band of its goal rate holds the prediction)](assets/charts/calibration-xg-dark.png#only-dark){ loading=lazy }](assets/charts/calibration-xg-dark.png)

[![xGOT calibration chart: xGOT is well calibrated: 10 of 10 bins agree in cross-validation, 10 of 10 on the hold-out season (On-target shots in 10 equal-size bins · a bin agrees when the 95% match-bootstrap band of its goal rate holds the prediction)](assets/charts/calibration-xgot-light.png#only-light){ loading=lazy }](assets/charts/calibration-xgot-light.png)
[![xGOT calibration chart: xGOT is well calibrated: 10 of 10 bins agree in cross-validation, 10 of 10 on the hold-out season (On-target shots in 10 equal-size bins · a bin agrees when the 95% match-bootstrap band of its goal rate holds the prediction)](assets/charts/calibration-xgot-dark.png#only-dark){ loading=lazy }](assets/charts/calibration-xgot-dark.png)

## Model comparison

The log-loss gain of the headline model over each reference on the same shots, as a percent of
the reference's log loss, with its match-bootstrap interval, in cross-validation and on the
hold-out; right of 0 means lower log loss than the reference. The title gives the hold-out
verdict per reference: "clearly better than" when the interval lies above 0, "level with" when
it crosses 0. For xG the references are the best logistic regression, the standard
baseline (logistic regression on distance, angle and body part) and StatsBomb's own xG, which
was likely trained on these matches and is a reference rather than an out-of-sample rival; for
xGOT the reference is the logistic regression. Numbers come from the [xG GBM
card](models/xg-gbm.md) and the [xGOT GBM card](models/xgot-gbm.md).

[![xG model comparison chart: Hold-out: clearly better than the standard baseline, level with the best regression and StatsBomb's xG (The xG model against each reference, in % of the reference's log loss · 95% intervals from 2,000 match resamples)](assets/charts/comparison-xg-light.png#only-light){ loading=lazy }](assets/charts/comparison-xg-light.png)
[![xG model comparison chart: Hold-out: clearly better than the standard baseline, level with the best regression and StatsBomb's xG (The xG model against each reference, in % of the reference's log loss · 95% intervals from 2,000 match resamples)](assets/charts/comparison-xg-dark.png#only-dark){ loading=lazy }](assets/charts/comparison-xg-dark.png)

[![xGOT model comparison chart: Hold-out: clearly better than the logistic regression (The xGOT model against each reference, in % of the reference's log loss · 95% intervals from 2,000 match resamples)](assets/charts/comparison-xgot-light.png#only-light){ loading=lazy }](assets/charts/comparison-xgot-light.png)
[![xGOT model comparison chart: Hold-out: clearly better than the logistic regression (The xGOT model against each reference, in % of the reference's log loss · 95% intervals from 2,000 match resamples)](assets/charts/comparison-xgot-dark.png#only-dark){ loading=lazy }](assets/charts/comparison-xgot-dark.png)
