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

## Evaluation (all phases)
- Split by **match / season**, never by row.
- Metrics: log loss (primary), Brier score, ROC AUC.
- Calibration plots overall and per segment (headers, set pieces, competitions).
- Benchmark: StatsBomb provider xG on the same open-data shots.
- Every trained model gets a [model card](model-card-template.md).
