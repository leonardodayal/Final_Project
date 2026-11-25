# Fantasy Football Trade Calculator Project Outline

## Project Objective

Build a machine-learning trade calculator for full-PPR fantasy football. The
calculator will predict each player's future weekly fantasy scores, combine
those predictions into a rest-of-season value, and determine whether a proposed
trade improves both teams.

## Scope and Current Assumptions

- The scoring format is full point per reception (full PPR).
- Rosters are entered manually; the project does not require an ESPN league ID
  or private league credentials.
- Data collection will use the Python `nflreadpy` package to access public
  nflverse player statistics and historical FantasyPros rankings.
- Direct synchronization with a fantasy league is outside the project scope.
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

### Primary Collection Interface

Use the Python `nflreadpy` package to retrieve public nflverse weekly player
statistics, FantasyPros historical rankings, and player-ID crosswalks. The
collector saves the downloaded data as local Parquet files and does not require
an ESPN league ID. The package is currently marked experimental, so record the
package version and data provenance when reporting results.

The weekly player-stat dataset should include a player identifier, season,
week, and full-PPR score. Verify these fields and the score calculation before
using the data for modeling. Historical rankings are supplemental; if the
source does not return them, document the gap rather than silently substituting
another ranking source. Do not commit generated datasets to the repository.

### Step 2: Cleaned Player-Week Dataset

Run the cleaning pipeline after collecting the raw player stats:

```bash
python -m src.data.clean --seasons 2021 2022 2023 2024 2025
```

The pipeline keeps regular-season weeks, calculates ESPN standard full-PPR
scores from stat components, and writes
`data/processed/player_weekly_<seasons>.parquet`. It includes QB, RB, WR, TE,
K, and D/ST rows; fullbacks are normalized to RB. Player rows retain their raw
stat components and include `full_ppr_points`, a score-source label, a flag for
observed stats, and a bye-week flag. The source PPR score is retained separately
for auditing. ESPN-standard kicker scores are calculated from field-goal
distance buckets, missed field goals, and made PATs. D/ST scores use sacks,
turnovers, touchdowns, blocks, safeties, and the ESPN points-allowed tiers; an
opponent defensive touchdown is excluded from points allowed, while special
teams points remain included. Canceled games are marked separately from bye
weeks; the canceled 2022 Bills-Bengals Week 17 matchup is scored as zero.

Missing player rows are filled with zero only when a weekly roster confirms the
player was on a regular-season roster with ACT, INA, or RES status. Development
and cut players are excluded from this zero-fill universe. D/ST bye weeks are
added as explicit zero-score rows. Historical rankings remain separate in the
Step 2 output and are joined as-of the prediction cutoff in Step 3. Generated
raw and processed datasets are not committed.

### Step 3: Leakage-Safe Features

Build historical features from the cleaned player-week dataset:

```bash
python -m src.data.collect --seasons 2020
python -m src.data.clean --seasons 2020
python -m src.features.build --seasons 2021 2022 2023 2024 2025 --history-seasons 2020
```

The output is `data/processed/player_week_features_<seasons>.parquet`.
`target_full_ppr_points` is separate from the allowlisted `FEATURE_COLUMNS`;
current-week stat components are not included as predictors. Player features
include prior-week scores and usage, rolling 3- and 5-player-week averages, and
current-season-to-date averages. Prior-week and rolling player histories carry
across seasons; the season-to-date average resets at each season boundary. The
season immediately before the target range is loaded as history-only input and
is not included in the output. Players without earlier rows have null
historical averages and a zero history count. Explicit zero-score player weeks
from observed games are included. Synthetic roster-zero and bye rows remain
zero-point targets but are excluded from player-performance lag and rolling
features; this prevents non-participation from being treated as an observed
performance.

Matchup features are joined from the NFL schedule. They summarize the
opponent's prior 3 and 5 games: passing and rushing yards allowed and points
allowed by its defense. Opponent offensive passing/rushing yards and points
scored are also summarized for D/ST forecasts. All matchup history is shifted
before the prediction week; Week 1 matchup history is null rather than using
prior-season games.

The feature builder adds the latest same-season FantasyPros positional PPR rank
available before the week's first kickoff. It uses `wp` weekly pages and `rp`
redraft PPR pages for QB, RB, WR, TE, K, and D/ST, matches player IDs through
the crosswalk (team IDs for D/ST), and records whether each rank came from the
weekly or redraft series. If both series have a snapshot on the same date, the
weekly rank is preferred. Ranking snapshots are date-only, so snapshots from
the kickoff date are excluded to avoid using a ranking that may have been
published after kickoff. Rows without an eligible, mapped rank remain missing.

Injury report and practice statuses use the latest non-null values timestamped
strictly before the first scheduled regular-season kickoff of the week. The NFL
schedule's kickoff time is interpreted as Eastern Time and injury timestamps as
UTC. Missing reports and statuses remain unknown rather than being labeled
healthy. Injury data is cached locally under `data/raw/`.

### Candidate Inputs

- Historical weekly fantasy points
- Receptions, targets, carries, and touches
- Passing, rushing, and receiving production
- Snap counts and participation rates, when available
- Games played and recent usage trends
- Player position, age, and experience
- Team offensive context
- Opponent and matchup information
- Pre-kickoff injury and practice statuses, with missing reports kept unknown
- Same-season, pre-kickoff FantasyPros weekly PPR positional rankings, with
  redraft PPR rankings as a fallback

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

### Step 4: Weekly-Score Regression

The initial supervised-learning target is a player's full-PPR score in a
specific week. This is a regression task, not reinforcement learning: there is
no sequential agent action or reward-driven environment interaction. This
distinction follows Sutton and Barto, Chapter 1. The model methods connect to
Bishop and Bishop's course material on linear regression and bias-variance
(Chapter 4), neural networks and error functions (Chapter 6), gradient descent
(Chapter 7), backpropagation (Chapter 8), and regularization (Chapter 9).

Compare these baselines: position mean, previous-week score, and rolling
3-/5-week score averages. Train Ridge regression over alpha values 0.1, 1, and
10 and a small one-hidden-layer (32-unit, ReLU) multilayer perceptron. Compare
each learned model with and without injury features as an ablation. Numeric
inputs use training-only median imputation and standard scaling; categorical
inputs use training-only missing-value imputation and one-hot encoding.

Use the 2021–2023 seasons for training, 2024 for validation and model selection,
then refit the selected model on 2021–2024 and evaluate once on the held-out
2025 season. Report MAE and RMSE overall and by position; compare the held-out
results with the simple baselines. Do not select or tune a model based on 2025
test results. The score regressor is trained on observed-stat weeks only, and
its position-level calibration is learned from 2024. A player's if-active score
is used at full value when they rank within the configured starter slots for
their team and position; other rostered players receive half. Starter slots are
1 QB, 3 RB, 4 WR, 2 TE, 1 K, and 1 D/ST per team. FantasyPros positional rank
determines the slots; unranked players fill vacancies by conditional score.
This is a ranking-based approximation, not an official depth chart.

Run the pipeline with
`python -m src.models.train --feature-path data\processed\player_week_features_2021-2022-2023-2024-2025.parquet --output-dir models`.
It writes a serialized model, a JSON metrics report, and validation/test
predictions under `models/`. The reference run's three small artifacts are
checked in; future generated files remain ignored by default.

### Step 5: Live Weekly Projections

Refresh current-season player statistics and ranking snapshots, then generate
forecasts for each fully unplayed regular-season week:

```bash
python -m src.data.collect --seasons 2026
python -m src.models.predict --season 2026
```

The prediction command refreshes schedules, weekly rosters, team statistics,
and injury reports from `nflreadpy`. It uses the latest eligible roster
snapshot, the Step 4 model artifact, and rankings no later than its UTC
forecast cutoff. It includes rostered fantasy positions and all D/ST teams,
forces bye-week scores to zero, and excludes incomplete or already-started
weeks.

For every future week, player and matchup history is based only on games
completed by the forecast cutoff; projected future scores are not fed back as
if they were observed. The latest known history is held fixed across the
forecast horizon, and forecasts should be refreshed after new games. Injury
features remain unknown when the feed lacks verifiable timestamps. The model
artifact is fitted on 2021–2024. After correcting synthetic zero-fill history,
the latest model selected Ridge regression with injury features. Its 2025
starter-weighted MAE/RMSE are 3.644/5.289 versus 4.259/5.938 for the
rolling-5-week baseline, evaluated using the same full-or-half starter rule.
The MLP candidate still reaches its iteration limit; it was not selected.
Treat forecasts as an initial benchmark, not a guarantee of predictive
improvement. Negative conditional QB scores are floored at zero; non-QB
conditional scores are unchanged.

The command writes `models/weekly_predictions_<season>.parquet` with one row
per player/week and a JSON metadata file recording the as-of time, completed
week, roster snapshot, model, and data limitations. The output includes
`predicted_full_ppr_points` (the starter-weighted score),
`predicted_full_ppr_points_if_active`, `predicted_starter`, and
`starter_score_multiplier`. A predicted starter receives a multiplier of 1.0;
other players receive 0.5. These generated forecasts are ignored by Git by
default.

### Strict Preseason Forecast

For a full-season forecast that ignores all 2026 results, run:

```bash
python -m src.models.predict --season 2026 --strict-preseason
```

This mode uses the Week 1 roster snapshot for the full player pool, all 18
weeks of schedule matchups, player history through 2025, and FantasyPros
snapshots strictly before the first 2026 regular-season date. It does not load
2026 player stats, team stats, or injury reports; schedule scores are masked
before feature generation. Bye-week predictions are zero. Since no 2026 results
are used, in-season matchup history and injury features are unknown, and later
roster transactions are not reflected. The same ranking-based starter rule is
applied; negative conditional QB scores are floored at zero.

The output is separate from the live forecast:
`models/weekly_predictions_2026_preseason.parquet` and its JSON metadata file.
Both are ignored by Git by default.

## Rest-of-Season Player Value

For player \(p\), define rest-of-season value as:

```text
ROS_value(p) = sum(predicted_weekly_score(p, week))
```

The default sum uses the starter-weighted `predicted_full_ppr_points`, not the
if-active-only column. It covers only the remaining fantasy-relevant weeks. A
later iteration may adjust this value for uncertainty, injury risk, or
playoff-week importance, but those adjustments are outside the initial scope.

## Team-Strength Calculation

1. Accept two manually entered rosters and identify each player's eligible
   positions.
2. Place eligible players into the lineup that maximizes projected starter
   value while respecting roster-position rules.
3. Give each starter 100% of their rest-of-season value.
4. Give each bench player 66.7% of their rest-of-season value.
5. Sum the weighted player values to calculate team strength.

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
- Test manual roster parsing and player matching.
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
2. Add `nflreadpy` to the dependency manifest and collect weekly player stats,
   historical rankings, and player-ID mappings for selected seasons.
3. Audit the fields, season coverage, full-PPR scoring, and data provenance.
4. Document any limitations in historical ranking coverage.
5. Build and validate a small player-week dataset.
6. Implement a simple rolling-average baseline.
7. Train and compare the first regression models.
8. Implement rest-of-season aggregation and lineup assignment.
9. Implement weighted team strength and trade deltas.
10. Produce an end-to-end example using two sample rosters.
11. Add automated tests and document reproducible commands.

## Open Decisions

- Does the historical FantasyPros ranking data cover the same seasons and
  players as the nflverse weekly player statistics?
- What manual roster input format will be used: a Python structure, CSV file, or
  simple command-line prompts?
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
- What year does the course's October 9 abstract deadline refer to, and is that
  deadline still applicable? Confirm the date in the current course schedule.
- Which machine-learning topics were covered in lectures, and what specific
  project method or question should be framed as extending them?
- Are there report formatting, citation-style, or page-count constraints beyond
  the assignment's guidance to write an approximately 15-page research-style
  report?
- Does the instructor provide a detailed grading rubric or weights for report
  sections and project components?
- Is any disclosure of AI-assisted tools required beyond the assignment's
  statement that such tools are permitted?

## Recommended Defaults

These are project recommendations, not confirmed course requirements. Keep
course-specific items open until they are checked against the current course
materials.

- **Data source:** Use `nflreadpy` for public nflverse player-week stats,
  FantasyPros historical rankings, player-ID mappings, and weekly injury
  reports. No ESPN league ID or private credentials are needed. Preserve
  missing rankings and reports when the historical sources have gaps.
- **Roster input:** Use a small CSV with one row per player and fields for team,
  player name, and position. Resolve players against a consistent player
  dataset and report ambiguous or unmatched names rather than silently
  guessing.
- **Scoring:** Fix the initial project to full PPR and document the scoring
  assumptions explicitly. Do not imply support for league-specific scoring
  variations unless they are implemented and tested.
- **Roster slots:** Start with one documented lineup configuration:
  1 QB, 2 RB, 2 WR, 1 TE, 1 RB/WR/TE FLEX, 1 D/ST, and 1 K. Treat these as
  project assumptions, not universal ESPN defaults.
- **Prediction horizon:** Sum predictions for the remaining regular-season
  weeks in the selected dataset. Exclude custom fantasy-playoff schedules from
  the first version and state that limitation.
- **Injuries and bye weeks:** Score a bye week as zero. Use injury information
  only if a reliable status is available before the predicted week; otherwise,
  do not invent a missed-game probability and describe injury risk as a
  limitation.
- **Trade classification:** Treat a zero team delta as neutral. Call a trade
  mutually beneficial only when both team deltas are strictly positive. Show
  both raw deltas and avoid a separate "meaningfully beneficial" threshold
  until uncertainty is estimated from validation results.
- **Bench weight:** Keep two-thirds as the initial comparison value, but test
  it against a few alternatives (for example 0, 0.5, and 1.0) and report
  whether the trade conclusion changes.
- **Abstract deadline:** Check the current course schedule or ask the
  instructor to confirm the year and whether October 9 is still applicable;
  do not infer a deadline year from the screenshot.
- **Lecture alignment:** Check the syllabus and frame the contribution around
  an appropriate comparison, such as a more advanced regression model against
  a simple baseline. Do not claim that a method extends lecture material until
  the covered topics are confirmed.
- **Report format:** Target approximately 15 pages and include the listed
  research-article sections. Follow any official course template if one exists;
  confirm citation style and whether references count toward the page target.
- **Grading rubric:** Look for a rubric in the course site and ask the
  instructor if none is available. Until then, organize the report around the
  stated deliverables and do not invent grading weights.
- **AI disclosure:** Although the screenshot permits AI-assisted tools,
  confirm whether the course requires disclosure. In the absence of a
  conflicting policy, include a brief factual note describing any AI
  assistance used.
