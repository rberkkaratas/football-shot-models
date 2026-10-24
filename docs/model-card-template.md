# Model card template

Copy this file to `docs/models/<model-name>.md` for every trained model version.

## Overview
- **Name / version:**
- **Task:** (e.g. binary classification: P(goal | shot))
- **Status:** experimental / candidate / released

## Intended use
What questions the model answers, and what it must not be used for.

## Data
- **Source(s):** StatsBomb open data / WhoScored snapshot (private)
- **Competitions & seasons:**
- **Rows:** train / validation / test
- **Filters:** (e.g. own goals removed, penalties modelled separately)

## Target
Exact definition of the label.

## Features
| Feature | Definition | Source field(s) |
|---|---|---|

## Model
Algorithm, hyper-parameters, calibration step.

## Validation design
Split strategy (by match / by season, never by row) and why.

## Metrics
| Metric | Train | Validation | Test |
|---|---|---|---|
| Log loss | | | |
| Brier score | | | |
| ROC AUC | | | |

Include a calibration (reliability) plot and per-segment metrics.

## Benchmarks
Comparison against reference models (e.g. provider xG on the same events).

## Limitations & known biases

## Data licensing
Which data may be shown publicly, which must stay private.

## Changelog
