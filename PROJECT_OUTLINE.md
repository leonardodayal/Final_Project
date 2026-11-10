# Fantasy Football Trade Calculator Project Outline

## Project Objective

Build a machine-learning trade calculator for full-PPR fantasy football. The
calculator will predict each player's future weekly fantasy scores, combine
those predictions into a rest-of-season value, and determine whether a proposed
trade improves both teams.

## Scope and Current Assumptions

- The scoring format is full point per reception (full PPR).
- The model predicts future weekly fantasy scores.
- A player's rest-of-season value is derived from their predicted scores over
  the remaining weeks of the season.
- Trade fairness is based on whether both teams improve after the trade, rather
  than whether both sides exchange identical player value.
- Starting players receive their full projected value.
- Bench players receive two-thirds of their projected value. This weight is a
  provisional assumption and should be tested during model evaluation.
- Team evaluation must account for roster construction, not only the sum of all
  player projections.

## End-to-End Workflow

1. Collect fantasy rankings, historical statistics, and weekly full-PPR scores.
2. Clean the data and create a consistent player-week dataset.
3. Engineer features that are available before each predicted week.
4. Train regression models to predict future weekly fantasy scores.
5. Generate predictions for every remaining week of the season.
6. Aggregate weekly predictions into rest-of-season player values.
7. Assign players to starting and bench roster slots.
8. Calculate team strength using full weight for starters and two-thirds weight
   for bench players.
9. Simulate a proposed trade and recalculate both team-strength ratings.
10. Report each team's change in value and whether both teams improve.

## Data Outline

### Candidate Inputs

- Historical weekly fantasy points
- Receptions, targets, carries, and touches
- Passing, rushing, and receiving production
- Snap counts and participation rates, when available
- Games played and recent usage trends
- Player position, age, and experience
- Team offensive context
- Opponent and matchup information
- Injury status and bye weeks, if reliable historical data is available
- Current fantasy rankings or expert projections for comparison

### Target Variable

The primary target is a player's full-PPR fantasy score for a future week. Each
training row must use only information that would have been known before that
week to prevent data leakage.

### Data Validation

- Verify player identities across data sources.
- Check that weekly scoring matches the selected full-PPR rules.
- Identify missing or duplicated player-week records.
- Distinguish a true zero-point performance from a missed game.
- Ensure features do not contain information from the week being predicted or
  from later weeks.

## Player Prediction Outline

### Baselines

- Previous-week fantasy score
- Rolling average of recent fantasy scores
- Position-level average
- Rest-of-season value based on a simple per-game average

### Candidate Regression Models

- Linear or regularized linear regression
- Random forest regression
- Gradient-boosted tree regression

Begin with interpretable baselines and add model complexity only when validation
results justify it.

### Evaluation

- Use chronological train, validation, and test splits.
- Compare models using mean absolute error and root mean squared error.
- Report results overall and separately by position.
- Compare the learned models against the simple baselines.
- Evaluate both weekly prediction accuracy and aggregated rest-of-season
  accuracy.

## Rest-of-Season Player Value

For player \(p\), define rest-of-season value as:

```text
ROS_value(p) = sum(predicted_weekly_score(p, week))
```

The sum covers only the remaining fantasy-relevant weeks. A later iteration may
adjust this value for uncertainty, injury risk, or playoff-week importance, but
those adjustments are outside the initial scope.

## Team-Strength Calculation

1. Place eligible players into the lineup that maximizes projected starter
   value while respecting roster-position rules.
2. Give each starter 100% of their rest-of-season value.
3. Give each bench player 66.7% of their rest-of-season value.
4. Sum the weighted player values to calculate team strength.

```text
team_strength = sum(starter_values) + (2 / 3) * sum(bench_values)
```

This design captures both immediate starting-lineup strength and the value of
depth. The bench coefficient should remain configurable for experiments, even
though its initial value is two-thirds.

## Trade Evaluation

For each team, calculate:

```text
team_delta = team_strength_after_trade - team_strength_before_trade
```

Classify a trade as mutually beneficial when both team deltas are greater than
zero. The output should include:

- Each team's strength before the trade
- Each team's strength after the trade
- Each team's absolute and percentage change
- The positions or roster slots responsible for the change
- Whether both teams improve
- A warning when one or both changes are too small relative to model uncertainty
  to support a confident conclusion

## Testing Outline

- Test full-PPR scoring calculations.
- Test feature calculations without future-data leakage.
- Test predicted weekly-score aggregation.
- Test starter and bench assignment under roster constraints.
- Test the two-thirds bench weighting.
- Test team strength before and after a trade.
- Test mutually beneficial, one-sided, and neutral trade classifications.
- Test reproducible model training with a fixed random seed.

## Suggested Repository Structure

```text
data/              Small, shareable input samples
docs/              Report assets and supporting documentation
models/            Version-appropriate trained artifacts
notebooks/         Exploratory analysis
src/
  data/            Collection, cleaning, and validation
  features/        Feature engineering
  models/          Training and weekly prediction
  evaluation/      Metrics and model comparison
  trades/          Lineup optimization and trade evaluation
tests/             Automated tests mirroring src/
```

## Report Outline

1. Introduction and motivation
2. Problem statement and definition of a mutually beneficial trade
3. Data sources and full-PPR scoring rules
4. Data cleaning and exploratory analysis
5. Feature engineering
6. Weekly player-score prediction methodology
7. Rest-of-season value calculation
8. Team-strength and roster-weighting methodology
9. Trade simulation methodology
10. Experimental results and baseline comparisons
11. Limitations, uncertainty, and potential bias
12. Future improvements
13. Conclusion

## Initial Implementation Milestones

1. Define league roster slots and exact full-PPR scoring rules.
2. Select and document data sources and licensing constraints.
3. Build and validate a small player-week dataset.
4. Implement a simple rolling-average baseline.
5. Train and compare the first regression models.
6. Implement rest-of-season aggregation and lineup assignment.
7. Implement weighted team strength and trade deltas.
8. Produce an end-to-end example using two sample rosters.
9. Add automated tests and document reproducible commands.

## Open Decisions

- Which fantasy platform or exact scoring rules will define passing, rushing,
  receiving, kicking, and defensive points?
- Which roster slots and lineup limits will the calculator support?
- Does the rest-of-season horizon include fantasy playoffs, and should playoff
  weeks receive different weights?
- How should injury designations, expected missed games, and bye weeks affect
  weekly predictions?
- Should a zero delta count as improvement, neutrality, or a failed fairness
  condition?
- How large must both improvements be before the trade is called meaningfully
  beneficial rather than effectively even?
- Should the two-thirds bench weight be fixed, tuned using historical outcomes,
  or compared against several candidate weights?
