# Roadmap

## Phase 0: Foundations
- [ ] football-core: pitch conversions, shot schema, StatsBomb loader, WhoScored snapshot export
- [ ] Shot table built for both sources; EDA notebook (`01-shot-eda`)

## Phase 1: xG
- [ ] Baseline: logistic regression (distance + angle)
- [ ] Context features (body part, situation, assist type)
- [ ] Gradient-boosted model + calibration
- [ ] Penalty model
- [ ] Benchmark vs StatsBomb xG; model card `xg-v1`
- [ ] Freeze-frame features (StatsBomb 360 / shot freeze frames)

## Phase 2: xGOT / PSxG
- [ ] Goal-mouth coordinates verified for both providers
- [ ] Post-shot model on on-target shots; model card `xgot-v1`
- [ ] Finishing analysis: xGOT − xG per player

## Phase 3: Goalkeepers
- [ ] Goals prevented / GSAx, per shot and per 90
- [ ] Uncertainty and stability analysis
- [ ] Goalkeeper profiles (shot-stopping by shot type)
