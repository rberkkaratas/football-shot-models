# Methodology

## Phase 1: xG (pre-shot)
**Target:** `is_goal ∈ {0, 1}` for non-penalty shots.

**Features, added in layers so each step's gain is measurable** (`ablation.STEPS`, M1.2 /
M1.3). A step stays only if its CV log-loss gain over the previous step has a 95%
match-bootstrap interval above 0; drop-one and add-one variants show each column's share.

1. Geometry: distance to the goal centre and the opening angle between the posts, each as a
   restricted cubic spline (Harrell, five fixed knots: `features.DISTANCE_KNOTS`,
   `ANGLE_KNOTS`), plus body part.
2. Context: `phase` (StatsBomb `situation` with kick-offs joined to open play and direct free
   kicks wide of the box joined to free kick), `assist_type`, `contact` (`technique` with lob,
   which is read from the ball's flight, merged into normal), `open_goal`, `one_on_one`.
   `situation` and `technique` are refused as raw features (`features.RESTRICTED_UNTIL`); only
   their registered readers `phase` and `contact` may read them (`RESTRICTED_READERS`). Direct
   corners are excluded from the population: StatsBomb logs them as shots on outcome.
   `first_time` and `gender` were tried and rejected (no CV gain with an interval above 0).
3. Freeze frames (StatsBomb only; recorded at the strike, so pre-shot): outfield opponents,
   teammates and the keeper in the cone from the ball to both posts, the nearest outfield
   opponent (capped at 10 m), and the opponent keeper's distances to goal, to the shot and off
   the angle bisector (capped at 20 m), with `gk_absent` when the frame has no keeper.

**Models, in order of complexity:**

1. Logistic regression on geometry (the classic baseline, M1.1).
2. Logistic regression on splines, context and freeze frames (M1.2 / M1.3; unpenalised,
   newton-cholesky, scaler fitted on training rows only).
3. A monotone gradient-boosted model (M1.4): scikit-learn's `HistGradientBoostingClassifier`
   on S3's inputs without the spline terms, non-increasing in distance and in opponents in the
   cone, non-decreasing in opening angle and in the nearest opponent's distance, each with
   every other column held fixed. Inside every fit, on that fit's training rows only, a 4-fold
   `GroupKFold` by match picks the leaf count and number of iterations, and Platt scaling is
   fitted on the chosen configuration's inner out-of-fold scores (a slope <= 0 is refused, so
   calibration cannot reverse the monotone direction). No class weights.

**Headline rule.** The GBM is the headline model only if its CV log-loss gain over the best LR
has a 95% match-bootstrap interval above 0; otherwise the LR stays headline and the cards say
so. The hold-out gain is printed next to it and does not decide.

**Penalties (M1.5):** in-game penalties (periods 1-4) are one pooled Beta-binomial with the
Jeffreys prior: P(goal) is the posterior mean, with its 95% equal-tailed interval. Shoot-out
kicks are reported in their own rows, never pooled. **Match xG** = Σ non-penalty xG (headline
model) + P(goal | in-game penalty) × in-game penalties; shoot-out kicks are not part of match
xG, and own goals and direct corners carry no xG.

## Phase 2: xGOT / PSxG (post-shot)
xGOT is P(goal) of a shot that reached the goal frame, given where it crossed the line and its
pre-shot xG (`football_shot_models.xgot`, MS5). Results are in the [xGOT cards](models/xgot-gbm.md)
and the README block printed by `make xgot` (pasted between the README markers; `make reproduce`
checks it byte for byte); none is typed here.

**Population (P2.2).** On target means a goal or a keeper save (C3 `on_target`; own goals are
never on target). Two populations, never mixed:

| Population | Rows | Model |
|---|---|---|
| open play | the provider's non-penalty xG population ∩ on target (deflected shots stay in) | LR and monotone GBM |
| in-game penalties | penalty kicks of periods 1–4 on target | penalty xGOT, fitted on dev kicks |
| shoot-out kicks | period 5 on target, a separately labelled sub-population | scored by the penalty xGOT dev fit, reported apart |
| direct corners | on target but outside the xG population | not scored (the reconciliation gap) |

Every C3 on-target shot falls in one row of the card's Filters table, and the shots and goals
actually scored must reconcile with C3 within 1 % per provider. Opta's published shots on target
also count last-line blocks and clearances off the line (Stats Perform, Opta event
definitions); here those are C3 `blocked` (WhoScored q101 SavedOffline), so our on-target counts
are lower by them. That is intended: the keeper did not stop those shots, so a keeper-neutral
xGOT and the goalkeeper models must not credit them.

**Placement.** The goal mouth is read only through `pitch.to_goal_frame` (metres from the goal
centre, y positive = the shooter's right) and only by the placement features of the xGOT guard
(`features.check_xgot_features`). Each point is clipped to the frame + 0.15 m, then rounded to
0.1 m (half to even) for both providers, so a goal and a save in one 0.1 m cell get the same
values. Rounding does not remove StatsBomb's recording grid, though: its goals sit on the 0.1 yd
(0.0914 m) lattice while its projected saves are continuous, so a 0.1 m cell that holds two
lattice values gets about twice the goal mass of its neighbours (a beat of about 1.07 m). The
run5-ms5 census of the 1.2 snapshot (on-target, non-penalty |y| cells; counts only) shows it:

| Cell (m) | StatsBomb goal rate | Neighbours |
|---|---|---|
| 0.0 | 0.228 | 0.150 |
| 1.1 | 0.288 | 0.141 / 0.199 |
| 2.2 | 0.445 | 0.286 / 0.303 |
| 3.2 | 0.754 | 0.608 / 0.588 |

The correlation of lattice values per cell with the cell's goal rate is 0.457 on StatsBomb and
0.046 on WhoScored, where Opta records goals and saves on one grid. The smooth LR cannot follow
these spikes; the GBM partly can, so part of the StatsBomb GBM's gain over the LR may be the
grid. Snapping the projected points to the match's recording grid in the loader is the open fix
(it needs a snapshot rebuild). A ball whose centre is more than 3.55 m (`features.INNER_Y`) from
the goal centre touches the post, so on-target shots out there are mostly saves: the GBM is
monotone in |y| only up to that point (`mouth_y_in`), and the width past it (`mouth_y_past`)
and the flag for a point recorded outside the frame + 0.15 m (`mouth_out`) are free. A
StatsBomb save stopped just in front of the shot projects far outside the frame; it is clipped
and flagged, never read as a ball in the corner. The cards print calibration by |y| band, which
the reliability table (binned by prediction) cannot show. Near / far post is the ball's side against the shot's side; a shot within 0.5 m of the
centre line has no near or far post (`far_side = 0`). StatsBomb saves stopped short of the line
are projected to the goal line from the shot location (see football-core's conventions); their
height is the stop height, and a deflected save is projected from the shot, not the deflection.
The cards count both cases.

**Keeper-neutral (PM E3, X1).** xGOT reads placement, shot context and xG only, never keeper
position (`gk_x`, `gk_y`, the freeze-frame keeper features or `open_goal`), so goals saved
against it credit the keeper's positioning too, and both providers' xGOT leave the same thing to
the keeper. They do not read quite the same placement: a StatsBomb save's height is where the
keeper stopped the ball, not where it would have crossed the line (next paragraph).
The StatsBomb xG input is therefore the headline xG refitted without the keeper-position
features; WhoScored's headline reads none and is used as it is.

**Cross-fitting.** The xG input is cross-fitted on the same match folds: an xGOT test row in
fold k reads the xG of the fit that never saw fold k, and a hold-out row the xG of the fit on
all of dev. The folds are the xG folds restricted to on-target rows, so every shot keeps its
fold.

**Models (P2.3, P2.4).** LR on logit xG, placement splines, the width past the post's inner
edge, angle × placement and the near / far term; a monotone GBM on xG, |y| (up to and past the
inner edge), the outside-the-frame flag, height, side and angle (non-decreasing in xG and in |y|
up to 3.55 m at a fixed side), headline only when its CV gain over the LR has a match-bootstrap interval
above 0; a placement-only penalty xGOT, reported apart with no target.

**Checks.** AUC ≥ 0.80, ECE < 0.02 and calibration slope in [0.9, 1.1], on CV and on the hold-out
(StatsBomb PL 2015/16) or test season (WhoScored 2026-27). ECE uses ten equal-count bins; on CV
it is judged on the fold mean with the worst fold and the pooled value beside it, and the
hold-out / test value has beside it its noise floor, the mean ECE of outcomes drawn from its own
predictions (PM X3).

**Shooting goals added (P2.5).** SGA = xGOT − xG per shot, next to goals − xG; both subtract the
same xG (the xGOT input). Blocked and off-target shots count as 0 xGOT. Reported only as
split-half correlations (each player's matches alternated by date, `metrics.split_half(unit=
"match")`) at 20 / 40 / 80 shots: a descriptive measure, not a finishing-skill claim. The MS6
keeper split-half alternates shots within a keeper in time order (`unit="row"`, PM X5).

## Phase 3: Goalkeeper shot-stopping
**Goals prevented = Σ PSxG faced − goals conceded** (penalties reported separately).
Rate versions per shot and per 90, uncertainty intervals and season-to-season stability
checks, because small samples dominate goalkeeper metrics.

**Leakage guard (M1.0).** xG features come only from the C3 columns with role `pre_shot`
(`football_shot_models.features.XG_ALLOWED`, derived from the schema, never typed by hand),
and the `pre_shot` columns of the C3.1 matches table (`gender`), minus the columns in
`RESTRICTED_UNTIL` (`situation`, `technique`), and `design_matrix` is
the only way to turn a frame into a feature matrix. A post-shot, outcome, restricted or lineage
column, or any name outside the C3 table, raises before a model sees it; the allow-list test
covers every non-`pre_shot` C3 column automatically. A derived feature (a freeze-frame count, a
spline input) is a `DerivedFeature` that names its C3 source columns: each source passes the
same check when the feature is made, and its function is handed only those columns. A
restricted column may be the source only of its registered reader, and a feature that `needs`
another (a keeper distance needs `gk_absent`) is refused without it.

**Baselines (M1.1).** Base rate → logistic regression on distance + angle → + body part
(indicators, reference right foot). The LR is unpenalised, with the scaler fitted on training
rows only and no class weights; see the [model cards](models/xg-lr-distance-angle-body-part.md).
Each model is a `ModelSpec` (features plus a `make()` that builds a fresh estimator), and every
model goes through the same `evaluation.evaluate` on the same `cv_folds`; the StatsBomb
benchmark is a separate `Benchmark` type that is scored, never fitted.

## Evaluation (all phases)
- Split by **match / season**, never by row. M1.1 onwards: grouped 5-fold CV by `match_id` on dev
  (the same folds for every model), plus one held-out competition-season (Premier League
  2015/16) scored by the fit on all of dev.
- Metrics: log loss (primary), Brier score, ROC AUC; CV values are fold means with the fold
  standard deviation. **Skill** is `1 − metric / base-rate metric` on the same shots.
- Σ xG error: `abs(Σ xG − goals) / goals`, pooled over the folds. For an unpenalised LR with an
  intercept it is close to 0 on CV by construction (fold errors of opposite sign cancel), so
  the cards also print the fold mean and maximum, and the hold-out value matters more.
- Calibration: intercept (calibration-in-the-large, `logit(p)` as offset) and slope of
  `y ~ a + b·logit(p)`, plus a reliability table of ten equal-count bins of the out-of-fold
  predictions. Segments: family, gender and body part, each with its log-loss skill against its
  own goal rate, which separates a harder sample from a weaker model.
- Comparisons (MS3): the log-loss gain of one model over another on the same out-of-fold and
  hold-out predictions, with a 95% percentile interval from 2,000 resamples of matches
  (`metrics.bootstrap_gain`, `evaluation.compare`).
- Benchmark: StatsBomb provider xG on the same open-data shots (a reference ceiling: it uses
  freeze-frame inputs and was likely fitted on these matches).
- Every trained model gets a [model card](model-card-template.md).
