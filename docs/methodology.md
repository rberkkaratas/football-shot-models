# Methodology

## Phase 1: xG (pre-shot)
**Target:** `is_goal ∈ {0, 1}` for non-penalty shots.

**Features, added in layers so each step's gain is measurable** (`ablation.STEPS`). A step
stays only if its CV log-loss gain over the previous step has a 95%
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

1. Logistic regression on geometry (the classic baseline).
2. Logistic regression on splines, context and freeze frames (unpenalised,
   newton-cholesky, scaler fitted on training rows only).
3. A monotone gradient-boosted model: scikit-learn's `HistGradientBoostingClassifier`
   on the third LR step's inputs without the spline terms, non-increasing in distance and in opponents in the
   cone, non-decreasing in opening angle and in the nearest opponent's distance, each with
   every other column held fixed. Inside every fit, on that fit's training rows only, a 4-fold
   `GroupKFold` by match picks the leaf count and number of iterations, and Platt scaling is
   fitted on the chosen configuration's inner out-of-fold scores (a slope <= 0 is refused, so
   calibration cannot reverse the monotone direction). No class weights.

**Headline rule.** The GBM is the headline model only if its CV log-loss gain over the best LR
has a 95% match-bootstrap interval above 0; otherwise the LR stays headline and the cards say
so. The hold-out gain is printed next to it and does not decide.

**Hold-out level (`level`).** The pooled GBM over-predicted the StatsBomb hold-out. The
xG GBM card decomposes the gap by segment (Σ xG − goals with z = gap / √Σ p(1 − p)) and tests
at most two candidate fixes, chosen after a hold-out probe, one refit each
(`models.LEVEL_FIXES`): a team-selected
indicator set to 0 at prediction, and training without team-selected seasons
(`ModelSpec.train_exclude`; held predictions never read the family). A fix is adopted iff its
hold-out Σ xG is within 2 SD of goals and every xG model check the pooled GBM meets stays met; the
adopted fix is the headline GBM (`models.ADOPTED`). The hold-out chose it, so the card says its
hold-out level is no longer an independent check. The 2-SD band is `metrics.BAND_SDS`, the one
rule of the xGOT and GSAx band tables too. Beside the largest remaining segment the card prints
StatsBomb's own xG on the same shots, a computed counterfactual, not a causal claim.

**WhoScored xG v2 (`v2`).** v1's inputs plus the pre-shot context of the previous play
event, written by football-core's WhoScored loader (schema 1.3): `prev_action` (the shooting
team's take-on, regain, earlier shot or other action, or the opponent's clearance, outfield
block, keeper action or other, reference pass; stoppages skipped) and `rebound` (the team's
previous shot of the period at most 3 s earlier). Opta big chance (q214) and every qualifier
that grades the chance or is known only after the shot stay out (an ablation only); the card's
census classes every Opta qualifier on a population shot. v1 stays the WhoScored xG of record
and the xGOT input, because `prev_action` and `rebound` read the keeper's previous event
(`features.KEEPER_DERIVED`, refused by the xGOT guards, so that GSAx keeps the keeper's own
credit). The design
probe chose the two columns on the test season, so the test gain carries selection optimism; the
card names the CV gain over v1 (training season only) as the evidence the test did not choose.

**Penalties:** in-game penalties (periods 1-4) are one pooled Beta-binomial with the
Jeffreys prior: P(goal) is the posterior mean, with its 95% equal-tailed interval. Shoot-out
kicks are reported in their own rows, never pooled. **Match xG** = Σ non-penalty xG (headline
model) + P(goal | in-game penalty) × in-game penalties; shoot-out kicks are not part of match
xG, and own goals and direct corners carry no xG.

## Phase 2: xGOT / PSxG (post-shot)
xGOT is P(goal) of a shot that reached the goal frame, given where it crossed the line and its
pre-shot xG (`football_shot_models.xgot`). Results are in the [xGOT cards](models/xgot-gbm.md)
and the README block printed by `make xgot` (pasted between the README markers; `make reproduce`
checks it byte for byte); none is typed here.

**Population.** On target means a goal or a keeper save (the shot table's `on_target`; own goals are
never on target). Two populations, never mixed:

| Population | Rows | Model |
|---|---|---|
| open play | the provider's non-penalty xG population ∩ on target (deflected shots stay in) | LR and monotone GBM |
| in-game penalties | penalty kicks of periods 1–4 on target | penalty xGOT, fitted on dev kicks |
| shoot-out kicks | period 5 on target, a separately labelled sub-population | scored by the penalty xGOT dev fit, reported apart |
| direct corners | on target but outside the xG population | not scored (the reconciliation gap) |

Every on-target shot of the shot table falls in one row of the card's Filters table, and the
shots and goals actually scored must reconcile with the shot table within 1 % per provider. Opta's published shots on target
also count last-line blocks and clearances off the line (Stats Perform, Opta event
definitions); here those are `blocked` shots (WhoScored q101 SavedOffline), so our on-target counts
are lower by them. That is intended: the keeper did not stop those shots, so a keeper-neutral
xGOT and the goalkeeper models must not credit them.

**Placement.** The goal mouth is read only through `pitch.to_goal_frame` (metres from the goal
centre, y positive = the shooter's right) and only by the placement features of the xGOT guard
(`features.check_xgot_features`). Each point is clipped to the frame + 0.15 m, then rounded to
0.1 m (half to even) for both providers, so a goal and a save in one 0.1 m cell get the same
values. Rounding does not remove StatsBomb's recording grid, though: its goals sit on the 0.1 yd
(0.0914 m) lattice while its projected saves are continuous, so a 0.1 m cell that holds two
lattice values gets about twice the goal mass of its neighbours (a beat of about 1.07 m). A
census of the schema 1.2 snapshot (on-target, non-penalty |y| cells; counts only) shows it:

| Cell (m) | StatsBomb goal rate | Neighbours |
|---|---|---|
| 0.0 | 0.228 | 0.150 |
| 1.1 | 0.288 | 0.141 / 0.199 |
| 2.2 | 0.445 | 0.286 / 0.303 |
| 3.2 | 0.754 | 0.608 / 0.588 |

The correlation of lattice values per cell with the cell's goal rate is 0.457 on StatsBomb and
0.046 on WhoScored, where Opta records goals and saves on one grid. The smooth LR cannot follow
these spikes; the GBM partly can, so part of the StatsBomb GBM's gain over the LR may be the
grid. The goalkeeper milestone fixed this in two steps. (1) football-core snaps every projected save to the centre of
its 0.1 yd goal-mouth cell (`io.statsbomb.snap_goal_y`, outcome-blind), so every StatsBomb
`goal_y` sits on the lattice and `fc goalmouth --check` fails otherwise. (2) Matches recorded on
the coarse 1 yd grid (football-core's per-match rule, `goalmouth.match_grids`) leave the
StatsBomb xGOT fit and the goalkeeper models: a pooled fit nets
two recording grids, which a placement model cannot tell from placement. They keep feeding the
xG input fit, in `xgot.fit` and in the goalkeepers' season-out refits (`season_out`) alike (they leave after
the xG input is cross-fitted); the reconciliation counts them under the reason *coarse recording grid* and
measures the gap against the shot table less these left-out rows, and the cards list them by
competition-season. A ball whose centre is more than 3.55 m (`features.INNER_Y`) from the goal
centre touches the post, so on-target shots out there are mostly saves; the flag for a point
recorded outside the frame + 0.15 m (`mouth_out`) marks them. A StatsBomb save stopped just in
front of the shot projects far outside the frame; it is clipped and flagged, never read as a
ball in the corner. The cards print calibration by |y| band, which the reliability table (binned
by prediction) cannot show: a band is judged by the 2-SD rule (|Σ xGOT − goals| ≤ 2 √Σ p(1 − p))
only with at least 100 shots (`metrics.MIN_JUDGED_SHOTS`); smaller bands are printed with their counts
as "not judged (n < 100)". The rule judges the headline xGOT (the GBM), which
GSAx reads; the LR cards keep the table and name their failing bands as a limitation. Near / far post is the ball's side against the shot's side; a shot within 0.5 m of the
centre line has no near or far post (`far_side = 0`). StatsBomb saves stopped short of the line
are projected to the goal line from the shot location (see football-core's conventions); their
height is the stop height, and a deflected save is projected from the shot, not the deflection.
The cards count both cases.

**Keeper-neutral.** xGOT reads placement, shot context and xG only, never keeper
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

**Models.** LR on logit xG, placement splines, the width past the post's inner
edge, angle × placement and the near / far term; a monotone GBM on xG, |y| (up to and past the
inner edge), the outside-the-frame flag, height, side and angle; since the goalkeeper milestone it is monotone
in xG only and free in placement (`xgot.GBM_SIGNS`), which removed the near-post misprice that
the earlier |y| constraint caused. It is the headline only when its CV gain over the LR has a match-bootstrap
interval above 0; a placement-only penalty xGOT, reported apart with no target.

**Checks.** AUC ≥ 0.80, ECE < 0.02 and calibration slope in [0.9, 1.1], on CV and on the hold-out
(StatsBomb PL 2015/16) or test season (WhoScored 2026-27). ECE uses ten equal-count bins; on CV
it is judged on the fold mean with the worst fold and the pooled value beside it, and the
hold-out / test value has beside it its noise floor, the mean ECE of outcomes drawn from its own
predictions.

**Shooting goals added.** SGA = xGOT − xG per shot, next to goals − xG; both subtract the
same xG (the xGOT input). Blocked and off-target shots count as 0 xGOT. Reported only as
split-half correlations (each player's matches alternated by date, `metrics.split_half(unit=
"match")`) at 20 / 40 / 80 shots: a descriptive measure, not a finishing-skill claim. The
keeper split-half alternates shots within a keeper in time order (`unit="row"`).

## Phase 3: Goalkeeper shot-stopping
`football_shot_models.gsax`, pure functions of an `xgot.XgotFit`; the cards are
written by `gsax_cards` (`make gsax`). Results are in the [GSAx card](models/gsax.md), the
[keeper-model card](models/gk-model.md) and the README block printed by `make gsax`; none is
typed here.

**Season-out xGOT.** Each rated shot's xGOT comes from fits (xG input and xGOT) that never saw
its season (`season_out.season_out`): on both providers the dev rows read the fold that left
their competition-season out, and the hold-out (StatsBomb) or test-season (WhoScored) rows the
refit's dev fit (the first polish round replaced WhoScored's year swap, whose fit trained on
the partial test season mispriced placement). The rated season's xGOT can still come from fits that
saw later seasons (disclosed with the count of next-season pairs it touches).

**GSAx.** GSAx = Σ q − goals over a keeper's on-target shots, with q = expit(logit xGOT +
δ) and one δ per competition-season and kind chosen so that Σ q = goals: each season's keepers
are rated against that season's league sum, so the xGOT level gap is not charged to them. Competition-seasons with at most 2 keepers or under 50 open-play shots are rated at
δ = 0 and flagged (`gsax.SMALL_LEAGUE`), because a league offset from two keepers
absorbs their own performance; team-selected competition-seasons are rated at δ = 0 too,
because their league sum is one team's keeper against its opponents. Penalties have
one δ per season-out source, since per-season groups hold a handful of kicks;
The minimum group size applies to every league-level offset, not to a keeper's own shift. GSAx
with penalties minus GSAx without is the keeper's penalty GSAx; shoot-outs are not rated. Keeper totals sum to the
league sum exactly; each keeper-season has the analytic SD √Σ q(1 − q) and a 90% match-bootstrap
interval. Team-selected seasons (one club's matches) rate the selected keeper against the
opponents' finishing, so the StatsBomb table flags them (†) and the chart (*).

**goal_z sensitivity (StatsBomb).** A projected save's `goal_z` is where the keeper met the
ball, not its height at the line. The GSAx card prints two sensitivities beside the headline,
never instead of it: GSAx re-rated on the goals and the saves stopped within 1 yd of
the line (δ re-fitted on that subset, judged only where the subset's season has a fitted δ), and
GSAx from a season-out refit of the open-play headline xGOT without `mouth_z`.

**Hierarchical keeper model.** goal ~ offset(logit q) + (1 | keeper-season), and the
literal per-keeper (career) grouping beside it, fitted by a Laplace approximation of the marginal
likelihood (`gsax.fit_keeper_model`). The shrunken estimates are tested on the next season's
on-target shots (same keeper, same competition, next start year; fitted on earlier seasons only)
by log loss against raw GSAx and against xGOT alone, with match-bootstrap intervals.

**Stability.** Split-half reliability of q − goal per shot by alternating shots within a
keeper in time order, stepped up with Spearman-Brown, at 100 / 250 / 500 total shots
faced; a tier with no keeper is reported as empty, and a tier is judged only with at least 30
keepers. The card sentences that say what GSAx allows about keeper skill are computed from these
intervals.

**Leakage guard.** xG features come only from the shot-table columns with role `pre_shot`
(`football_shot_models.features.XG_ALLOWED`, derived from the schema, never typed by hand),
and the `pre_shot` columns of the matches table (`gender`), minus the columns in
`RESTRICTED_UNTIL` (`situation`, `technique`), and `design_matrix` is
the only way to turn a frame into a feature matrix. A post-shot, outcome, restricted or lineage
column, or any name outside the shot table, raises before a model sees it; the allow-list test
covers every non-`pre_shot` column automatically. A derived feature (a freeze-frame count, a
spline input) is a `DerivedFeature` that names its source columns: each source passes the
same check when the feature is made, and its function is handed only those columns. A
restricted column may be the source only of its registered reader, and a feature that `needs`
another (a keeper distance needs `gk_absent`) is refused without it.

**Baselines.** Base rate → logistic regression on distance + angle → + body part
(indicators, reference right foot). The LR is unpenalised, with the scaler fitted on training
rows only and no class weights; see the [model cards](models/xg-lr-distance-angle-body-part.md).
Each model is a `ModelSpec` (features plus a `make()` that builds a fresh estimator), and every
model goes through the same `evaluation.evaluate` on the same `cv_folds`; the StatsBomb
benchmark is a separate `Benchmark` type that is scored, never fitted.

## Evaluation (all phases)
- Split by **match / season**, never by row. From the baselines on: grouped 5-fold CV by `match_id` on dev
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
- Comparisons: the log-loss gain of one model over another on the same out-of-fold and
  hold-out predictions, with a 95% percentile interval from 2,000 resamples of matches
  (`metrics.bootstrap_gain`, `evaluation.compare`).
- Benchmark: StatsBomb provider xG on the same open-data shots (a reference ceiling: it uses
  freeze-frame inputs and was likely fitted on these matches).
- Every trained model gets a [model card](model-card-template.md).
