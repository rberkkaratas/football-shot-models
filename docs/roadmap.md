# Roadmap

Milestone IDs match the plan in `football-core/docs/roadmap.md`. Every milestone has a
measurable acceptance criterion; thresholds come from a single-season StatsBomb pilot and are
re-baselined once on the final split, then frozen.

## Stage 2: Public data path and first result
| ID | Milestone | Done when |
|---|---|---|
| M1.0 | Data contract and leakage guard | An allow-list test fails if any post-shot field enters the xG features |
| M1.1 | Baselines: base rate → LR distance + angle → + body part | Grouped-CV log loss ≤ 0.275, Brier ≤ 0.077, AUC ≥ 0.75, \|Σ xG − goals\| / goals ≤ 3% |

## Stage 3: Private data path
| ID | Milestone | Done when |
|---|---|---|
| M0 | Score the legacy v0 xG (seria-chance-creators) on the new split | v0 metrics sit in the benchmark table on the same held-out shots |

## Stage 4: xG modelling
| ID | Milestone | Done when |
|---|---|---|
| M1.2 | Splines and context features | Log loss ≤ 0.265; match-bootstrap 95% CI for the gain over M1.1 excludes 0 |
| M1.3 | Freeze-frame features (StatsBomb) | Log loss ≤ 0.258 and within 0.006 of StatsBomb xG on the same shots |
| M1.4 | Monotone GBM + calibration | Ships only if it beats the best LR with CI excluding 0; calibration slope in [0.9, 1.1] |
| M1.5 | Penalty model | Pooled conversion with Beta CI, inside 0.76–0.80 |
| M1.6 | WhoScored xG (train 2025-26, test 2026-27) | Non-penalty log loss ≈ 0.26–0.27; matches or beats v0. Starting line: 0.297 (distance + angle only) |
| M1.7 | xG v1 release | Every number in the model card reproduces with `uv run` from a pinned manifest |

## Stage 5: xGOT / PSxG
| ID | Milestone | Done when |
|---|---|---|
| P2.1 | Goal-mouth harmonisation | ≥ 99% of goals inside the frame ± 0.15 m; Opta left/right confirmed against zone qualifiers |
| P2.2 | Population spec (on target, faced by the keeper) | Shots on target and goals reconcile within ± 1% |
| P2.3 | xGOT logistic regression | AUC well above 0.70; ECE < 0.02; total-probability check per xG decile within ± 10% |
| P2.4 | Monotone GBM xGOT + penalty xGOT | No worse than P2.3 after recalibration |
| P2.5 | Shooting goals added (xGOT − xG) | Split-half correlations reported next to goals − xG |

## Stage 6: Goalkeepers
| ID | Milestone | Done when |
|---|---|---|
| P3.1 | GSAx table | Keeper totals sum to league Σ PSxG − GA; every value has an SD and a 90% bootstrap CI |
| P3.2 | Hierarchical keeper model | Shrunken estimates predict the next season better than raw GSAx |
| P3.3 | Stability study | Reliability at 100 / 250 / 500 shots faced, with a Spearman–Brown curve |
| P3.4 | Positioning variant (optional, StatsBomb only) | Documented and labelled as unavailable for WhoScored |
