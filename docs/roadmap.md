# Roadmap

Six milestones, each delivered end to end in one run. Component IDs (C* in `football-core`,
M* / P* here) stay stable so specs, commits and docs can refer to them. Thresholds come from a
single-season StatsBomb pilot; they are re-baselined once on the final split, then frozen.

| # | Milestone | Components | Done when |
|---|---|---|---|
| 1 | **Data foundation** ✅ | C3 schema, C6 snapshots + manifest CLI (core) | ✅ Shot and freeze-frame tables validate with clear errors; snapshots are byte-identical on rerun and `fc verify` catches a one-byte change |
| 2 | **StatsBomb xG baseline** ✅ | C4 `io.statsbomb`, C7 `viz` (core); M1.0 leakage guard, M1.1 baselines | ✅ 101,614 shots of 3,961 matches load with zero schema violations and goals reconcile with the score in every match (shoot-outs excluded). The allow-list test rejects every C3 column that is not `pre_shot`. Base rate → LR distance + angle → + body part each have a [model card](models/xg-lr-distance-angle-body-part.md). Best (LR + body part), grouped 5-fold CV on 89,929 shots: log loss **0.2880** (target ≤ 0.275, *missed*), Brier **0.0821** (≤ 0.077, *missed*), AUC **0.7628** (≥ 0.75, met), \|Σ xG − goals\| / goals **0.02 %** (≤ 3 %, met); hold-out PL 2015/16 0.2726 / 0.0767 / 0.7548 / 6.98 %. The miss comes from the sample mix: team-selected seasons (one-club collections and single finals) are fitted as well as league or tournament shots (log-loss skill against their own goal rate 0.146 vs 0.145) but convert more (11.90 % goals; 0.3117 log loss), and the pooled geometry model is less skilful on women's shots (skill 0.116; 0.2974). Without the team-selected seasons dev still scores 0.2821 out of fold, above the target: only the men's league or tournament shots meet it (0.2686 / 0.0756). StatsBomb's own xG gets 0.2674 / 0.0758. The pooled CV Σ xG error is near 0 by construction; per fold it is 2.66 % on average (max 3.99 %), and 6.98 % on the hold-out. Reported, not tuned. Shot map and goal-mouth plot pass image tests |
| 3 | **xG model** | M1.2 splines + context, M1.3 freeze-frame features, M1.4 monotone GBM + calibration, M1.5 penalty model | Log loss ≤ 0.258 and within 0.006 of StatsBomb xG on the same shots; GBM ships only if its gain over the best LR has a CI excluding 0; calibration slope in [0.9, 1.1]; penalty conversion inside 0.76–0.80 |
| 4 | **WhoScored xG v1** | C5 `io.whoscored` (core); M0 legacy v0 score, M1.6 WhoScored model, M1.7 release | Duplicate match dropped and goals reconcile with scores. Trained on 2025-26, tested on 2026-27: non-penalty log loss ≈ 0.26–0.27 and at least as good as v0 (start line 0.297 with distance + angle). Every number in the model card reproduces with `uv run` from a pinned manifest |
| 5 | **xGOT / PSxG** | P2.1 goal-mouth harmonisation, P2.2 population, P2.3 LR, P2.4 GBM + penalty xGOT, P2.5 shooting goals added | ≥ 99% of goals inside the frame ± 0.15 m; on-target shots and goals reconcile within ± 1%; AUC ≈ 0.80 and ECE < 0.02; split-half correlations reported next to goals − xG |
| 6 | **Goalkeepers** | P3.1 GSAx, P3.2 hierarchical keeper model, P3.3 stability study | Keeper totals sum to league Σ PSxG − GA with SDs and 90% CIs; shrunken estimates predict the next season better than raw GSAx; reliability at 100 / 250 / 500 shots faced |

Dropped: P3.4 (keeper positioning variant). It was optional and StatsBomb-only.

## Publishing
Charts or numbers go to LinkedIn only after Berk approves them. Milestone 2 gives the first
publishable chart; milestone 4 is the xG v1 release.
