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
**Population:** on-target shots only. Off-target and blocked shots have post-shot value 0.
**Features:** pre-shot xG + goal-mouth end location (`goal_y`, `goal_z`). Corner placement
should score higher than shots struck at the middle of the goal.

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
