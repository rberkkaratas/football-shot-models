# Methodology

## Phase 1: xG (pre-shot)
**Target:** `is_goal ∈ {0, 1}` for non-penalty shots.

**Features, added in layers so each step's gain is measurable:**

1. Geometry: distance to goal centre, angle subtended by the goal posts.
2. Shot context: body part, situation (open play / set piece / counter).
3. Build-up: assist type, previous action, first-time shot.
4. Freeze frames (StatsBomb only): goalkeeper position, defenders in the shot cone.

**Models, in order of complexity:**

1. Logistic regression on geometry (the classic baseline).
2. Logistic regression with context features.
3. Gradient-boosted trees, followed by probability calibration.

**Penalties:** a separate, near-constant model estimated from data.

## Phase 2: xGOT / PSxG (post-shot)
**Population:** on-target shots only. Off-target and blocked shots have post-shot value 0.
**Features:** pre-shot xG + goal-mouth end location (`goal_y`, `goal_z`). Corner placement
should score higher than shots struck at the middle of the goal.

## Phase 3: Goalkeeper shot-stopping
**Goals prevented = Σ PSxG faced − goals conceded** (penalties reported separately).
Rate versions per shot and per 90, uncertainty intervals and season-to-season stability
checks, because small samples dominate goalkeeper metrics.

**Leakage guard (M1.0).** xG features come only from the C3 columns with role `pre_shot`
(`football_shot_models.features.XG_ALLOWED`, derived from the schema, never typed by hand), and
`design_matrix` is the only way to turn a frame into a feature matrix. A post-shot, outcome,
restricted or lineage column, or any name outside the C3 table, raises before a model sees
it; the allow-list test covers every non-`pre_shot` C3 column automatically.

**Baselines (M1.1).** Base rate → logistic regression on distance + angle → + body part
(indicators, reference right foot). The LR is unpenalised, with the scaler fitted on training
rows only and no class weights; see the [model cards](models/xg-lr-distance-angle-body-part.md).

## Evaluation (all phases)
- Split by **match / season**, never by row. M1.1: grouped 5-fold CV by `match_id` on dev
  (the same folds for every model), plus one held-out competition-season (Premier League
  2015/16) scored by the fit on all of dev.
- Metrics: log loss (primary), Brier score, ROC AUC; CV values are fold means with the fold
  standard deviation. **Skill** is `1 − metric / base-rate metric` on the same shots.
- Σ xG error: `abs(Σ xG − goals) / goals`, pooled over the folds. For an unpenalised LR with an
  intercept it is close to 0 on CV by construction, so the hold-out value matters more.
- Calibration: intercept (calibration-in-the-large, `logit(p)` as offset) and slope of
  `y ~ a + b·logit(p)`, plus a reliability table of ten equal-count bins of the out-of-fold
  predictions. Segments: family, gender and body part.
- Benchmark: StatsBomb provider xG on the same open-data shots (a reference ceiling: it uses
  freeze-frame inputs and was likely fitted on these matches).
- Every trained model gets a [model card](model-card-template.md).
