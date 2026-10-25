# football-shot-models

Expected goals (xG), post-shot expected goals (xGOT / PSxG) and goalkeeper
shot-stopping models, built step by step and documented with model cards.

- [Data](data.md): sources, the shot table, filters.
- [Methodology](methodology.md): how each model is defined and evaluated.
- [Roadmap](roadmap.md): phases and milestones.
- [Modules](modules.md): package layout and status.
- Model cards: [base rate](models/xg-base-rate.md),
  [LR distance + angle](models/xg-lr-distance-angle.md),
  [LR distance + angle + body part](models/xg-lr-distance-angle-body-part.md) (the M1.1 baselines);
  [LR splines + body part](models/xg-lr-splines.md), [+ context](models/xg-lr-context.md),
  [+ freeze frames](models/xg-lr-frames.md) (M1.2 / M1.3);
  [monotone GBM + Platt](models/xg-gbm.md), the headline xG model (M1.4);
  [penalty conversion](models/xg-penalty.md) (M1.5).
- [Model card template](model-card-template.md): filled in for every trained model.
- [References](references.md)
