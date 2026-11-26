# Project Handoff

## Current Status

- Added `AGENTS.md` with repository-specific contributor guidance.
- Added Karpathy behavioral guidelines for cautious, focused implementation.
- Added a requirement to update this handoff regularly and immediately before each Git commit.
- Added `PROJECT_OUTLINE.md` with project, data, modeling, team-strength,
  trade-evaluation, testing, report, and implementation outlines.
- Added `DESIGN_CRITERIA.md` translating the course assignment screenshot into
  explicit course requirements and project-specific acceptance criteria.
- Added `data/roster_template.csv` as an empty roster-entry template with team,
  player name, and position columns.
- Recorded that the screenshot omits the year for the October 9 abstract
  deadline and does not provide a detailed grading rubric or AI disclosure
  requirement.
- Added unresolved course requirements to `PROJECT_OUTLINE.md`, including the
  deadline year, lecture-topic alignment, report formatting/page constraints,
  grading rubric, and AI-tool disclosure policy.
- Added recommended defaults for the project's open decisions while retaining
  course-policy questions as items to confirm with official course materials.
- Defined the initial scope as full-PPR scoring with weekly predictions aggregated
  into rest-of-season player value.
- Defined trade fairness as both teams improving after the simulated trade.
- Set the provisional bench contribution to two-thirds of projected player value.
- Selected `nflreadpy` for public nflverse weekly player statistics,
  FantasyPros historical rankings, and player-ID mappings; no ESPN league ID
  or private credentials are required.
- Added the Step 1 collector in `src/data/collect.py`; it downloads selected
  seasons, saves Parquet files under `data/raw/`, and records collection
  metadata. Generated raw files are ignored by Git.
- Added `requirements.txt`, collector tests, and setup/collection commands to
  `AGENTS.md`.
- The Step 1 commits are now included on the local `trenner` branch; the
  `docs/project-design-criteria` branch was left intact.
- Installed Python 3.13 and project dependencies in the ignored `.venv/`.
- Collected the five completed seasons 2021–2025 locally: 94,848 weekly
  player-stat rows, 12,542 player-ID mappings, and 1,890,527 historical
  ranking rows.
- Step 1 raw collection is complete for the selected seasons; cleaning,
  joining rankings, validating scoring rules, and classifying missing weeks
  were initially outstanding for Step 2.
- Step 1 files changed: `.gitignore`, `AGENTS.md`, `HANDOFF.md`,
  `PROJECT_OUTLINE.md`, `requirements.txt`, `src/__init__.py`,
  `src/data/__init__.py`, `src/data/collect.py`, and `tests/test_collect.py`.
- Scoped roster entry to manual input; private ESPN league synchronization and
  user credentials are not required for the school project.
- Implemented Step 2 in `src/data/clean.py`. It filters to regular-season weeks,
  builds player and D/ST rows, independently calculates ESPN-standard full-PPR
  scores, adds roster-confirmed player zero rows and explicit D/ST bye zeros,
  and saves a cleaning metadata file.
- For player zero-fill, included weekly roster statuses ACT, INA, and RES;
  excluded developmental and cut players. Fullbacks are normalized to RB.
- Preserved source `fantasy_points_ppr` and added `full_ppr_points`,
  `source_ppr_difference`, `score_source`, `stats_observed`, `is_bye_week`, and
  `is_cancelled_game_week`.
- Collected local raw weekly rosters, team stats, and schedules as cleaning
  inputs. Generated `data/processed/` output is ignored by Git.
- Built the cleaned 2021–2025 regular-season dataset: 56,952 rows, including
  32,561 player-stat rows, 21,511 roster-confirmed player zero rows, 2,718
  D/ST game rows, and 160 D/ST bye rows plus two canceled-game zero rows.
- Added a separate canceled-game flag for the 2022 Bills-Bengals Week 17
  cancellation; those teams' zero scores are not mislabeled as bye weeks.
- Implemented Step 3 in `src/features/build.py`. It creates one feature row per
  player-week, keeps the full-PPR target separate from `FEATURE_COLUMNS`, and
  excludes same-week raw stats from predictors.
- Player features include previous-week score and usage, rolling 3/5 prior
  player-week averages, current-season-to-date averages, and prior-history
  counts. Previous-week and rolling histories carry across season boundaries;
  season-to-date averages reset at each season.
- Player-performance features use observed stat rows only. Synthetic
  roster-zero and bye rows remain zero-point targets but are excluded from
  lagged and rolling performance metrics; observed zero-point performances
  are retained.
- Matchup features use only the opponent's prior games: passing/rushing yards
  and points allowed by its defense, plus opponent offensive production for
  D/ST forecasts. Player weeks and team-game history are shifted before the
  target week.
- Added same-season FantasyPros positional PPR ranks using weekly ranks and
  redraft PPR fallbacks, plus report/practice status features strictly before
  each week's first kickoff. Weekly ranks win same-date ties; date-only rank
  snapshots on the kickoff date are excluded. A source feature identifies
  weekly versus redraft ranks; missing rankings and reports remain unknown.
- Cleaned 2020 as a history-only warm-up season, allowing Week 1 2021 player
  features to use 2020 results without adding 2020 target rows.
- Built 56,952 feature rows locally with 57 allowlisted feature columns.
  Feature datasets, injury input, and metadata are ignored under `data/`.
- Step 3 files changed: `AGENTS.md`, `HANDOFF.md`, `PROJECT_OUTLINE.md`,
  `src/features/__init__.py`, `src/features/build.py`, and
  `tests/test_build_features.py`.
- Step 2 files changed: `.gitignore`, `AGENTS.md`, `HANDOFF.md`,
  `PROJECT_OUTLINE.md`, `src/data/clean.py`, and `tests/test_clean.py`.
- Implemented the initial Step 4 supervised regression pipeline in
  `src/models/train.py`. It compares position-mean, previous-week, and rolling
  score baselines against Ridge regression and a 32-unit MLP, with/without
  injury features, using 2021–2023 training, 2024 validation, and a final
  2025 test. The winning validation candidate is refit on 2021–2024.
- After excluding synthetic zero-fill rows from player-performance history,
  the full data run selected the 32-unit MLP with injury features. Validation
  MAE/RMSE were 3.443/5.302; 2025 test MAE/RMSE were 3.361/5.243. Test MAE
  beat the rolling 3-/5-week baselines (4.585/4.498). The MLP still reached
  its 100-iteration limit without convergence; do not retune against 2025.
- Step 4 uses a training-fitted preprocessing pipeline: numeric median
  imputation and standardization, categorical imputation and one-hot encoding.
  The reference model, metrics, and prediction artifacts are included under
  `models/`; future generated files there remain ignored by default.
- Added `pandas` and `scikit-learn` dependencies and corrected `.gitignore` to
  ignore only root-level generated `models/`, so the Python source package
  `src/models/` remains trackable.
- Added an end-to-end Step 4 regression test for chronological splits and
  artifact generation. It shortens MLP iterations for test speed.
- Implemented live Step 5 forecasting in `src/models/predict.py`. It refreshes
  2026 weekly rosters, team stats, schedules, and injuries; requires collected
  player stats and FantasyPros rankings; filters to fully completed weeks;
  and forecasts every fully unplayed week after the UTC as-of time.
- At the 2026-11-25 07:22 UTC run, the current data had completed Week 11,
  Week 12 rosters, and future schedule Weeks 12–18. The forecast output has
  4,634 rows (662 per week), with 125 bye rows forced to exactly zero and no
  duplicate player-week keys.
- Step 5 generates each week's player and matchup features from observed
  player/team history only through Week 11. It does not recursively treat
  predicted scores as observed results; that recent-history snapshot is held
  fixed across the remaining weeks. It uses the Step 4 model artifact trained
  on 2021–2024, and ranking snapshots are capped at the forecast as-of date.
- The 2026 injury feed has no `date_modified` timestamp, so 2026 injury
  features remain unknown. Forecast Parquet and JSON outputs are local under
  the ignored `models/` directory.
- Added Step 5 tests for completed-week detection, future roster/bye/D/ST row
  construction, and matching predictions back to player keys.
- Fixed player-performance history so rows with `stats_observed=False`
  contribute neither zero scores nor zero usage to lag, rolling, or
  season-to-date performance features. Their zero target remains intact, and
  actual observed zero-point games still count in history.
- Josh Allen's 2025 Week 18 `roster_zero` row was the trigger: its zero had
  become the Week 1 2026 previous-score feature. After rebuilding features and
  retraining, his strict-preseason Week 1 forecast increased from 4.5 to 15.9
  points and Week 4 from 4.1 to 12.8.
- Rebuilt the checked-in Step 4 model, metrics, and prediction artifacts, and
  regenerated both live and strict-preseason 2026 forecasts with the updated
  model. The strict-preseason output still has 12,042 rows (669 per week),
  unique player-week keys, and exact zero bye predictions.
- Added `--strict-preseason` for a separate 2026 Weeks 1–18 forecast. It uses
  the Week 1 roster snapshot, FantasyPros snapshots strictly before the
  September 9 opening date, and player history through 2025. It does not load
  2026 player stats, team stats, or injuries; all 2026 schedule scores are
  masked before feature generation.
- The strict-preseason output is `models/weekly_predictions_2026_preseason.parquet`
  with a matching JSON metadata file. It has 12,042 rows (669 per week), 18
  forecast weeks, no duplicate player-week keys, and all 669 bye rows have
  zero predictions. Negative weekly QB predictions are floored at zero; other
  positions are unchanged. The existing live output file remains separate.
- Preseason matchup-history and injury features are unknown, and the fixed
  Week 1 roster does not reflect later transactions.
- The score model trains only on `target_stats_observed=True`; position-score
  calibration uses 2024, and the held-out 2025 season remains untouched by
  model selection.
- Replaced the participation-probability multiplier with the requested
  ranking-based role rule: per team, top 1 QB, 3 RB, 4 WR, 2 TE, 1 K, and
  1 D/ST receive full if-active scores; others receive half. Unranked players
  fill open slots by descending conditional score. This approximates a depth
  chart using FantasyPros position ranks; it is not an official starter feed.
- Forecast outputs now expose `predicted_full_ppr_points`,
  `predicted_full_ppr_points_if_active`, `predicted_starter`, and
  `starter_score_multiplier`. The first is the requested full-or-half score.
- The latest selected candidate is `ridge_alpha_10_with_injury`. Validation
  MAE/RMSE are 3.861/5.446; held-out 2025 MAE/RMSE are 3.644/5.289 versus
  4.259/5.938 for the rolling-5-week baseline. MLP candidates still emit
  max-iteration warnings but were not selected.
- Rebuilt the model artifacts and regenerated live 2026 Weeks 12–18 (4,634
  rows) and strict-preseason Weeks 1–18 (12,042 rows). Both have unique
  player-week keys, zero bye scores, and starter counts consistent with the
  configured team-position slots. Strict-preseason Week 1 gives Josh Allen
  14.84 points and Joe Burrow 17.96 points; both are ranked starters.
- Full test suite passes (22 tests); `git diff --check` passes.
- Files changed in this milestone: `AGENTS.md`, `HANDOFF.md`,
  `PROJECT_OUTLINE.md`, `models/step4_metrics.json`,
  `models/step4_predictions.parquet`, `models/weekly_score_model.joblib`,
  `src/features/build.py`, `src/models/train.py`, `src/models/predict.py`,
  `src/models/roles.py`, `tests/test_build_features.py`,
  `tests/test_train_model.py`, and `tests/test_predict.py`. Generated
  2026 forecast Parquet/JSON files remain Git-ignored.
- Implemented rest-of-season aggregation in `src/trades/values.py`, using the
  unweighted `predicted_full_ppr_points_if_active` column so NFL-rank discounts
  are not compounded with fantasy lineup weights.
- Added CSV roster matching, the documented 1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX /
  1 D/ST / 1 K lineup, configurable bench weighting, and two-team trade
  simulation in `src/trades/roster.py` and `src/trades/evaluate.py`.
- Added `data/trade_template.csv`, a JSON-reporting CLI, input validation, and
  synthetic end-to-end tests for lineup fit and trade outcomes.
- The test suite now passes (43 tests), `pip check` reports no broken
  requirements, and `git diff --check` passes.
- Real-data trade evaluation has not been run: no generated weekly forecast,
  completed roster CSV, or proposed trade CSV is available in this worktree.
- Started `feat/trade-calculator-app` from the trade-evaluation branch.
- Expanded the Streamlit app to generate 2026 weekly projections using the
  existing model and public data pipeline, or load a CSV/Parquet forecast.
- Added manual player selection for both rosters and trade sides, while keeping
  CSV roster/trade uploads as an alternative. The app displays weekly scores,
  before/after lineup impact, and a downloadable JSON report.
- Uploaded data remains in memory. Forecast generation requires the trained
  model artifact; missing public inputs are downloaded and cached locally.
- Added Streamlit to `requirements.txt` and a dark theme configuration in
  `.streamlit/config.toml`.
- Added tests for forecast input preparation/caching and manual player
  validation/serialization. The full suite passes (43 tests), `pip check` is
  clean, and a local Streamlit smoke test returned HTTP 200.
- The app workflow is committed locally on `feat/trade-calculator-app` as
  `3404af1` (`Build weekly projection trade calculator app`).
- Code review found no blocking issues. The lineup detail now includes bench
  players as well as starters so the displayed roster matches the strength
  calculation.
- Final validation: `.venv/bin/python -m pytest -q` (43 passed),
  `.venv/bin/python -m pip check` (clean), Python compilation,
  `git diff --check`, and a local app request (HTTP 200).
- No live forecast was generated during this review; first-time generation may
  download public data and requires the local trained model artifact.
- `origin/main` is eight commits behind this branch: seven existing project
  commits plus the app implementation. Pushing this branch to `main` will
  include the existing data, feature, model, and trade-evaluation commits as
  well as the app changes. The untracked `app 2.py` and `.DS_Store` are not
  part of the planned commits.
- Next step: push the reviewed branch to `origin/main`.

## Verification

- Reviewed the repository layout and recent Git history.
- Confirmed Markdown has no whitespace errors with `git diff --check`.
- Reviewed the project outline against the decisions supplied by the project owner.
- Reviewed the design criteria against the assignment screenshot and project outline.
- Updated the outline's open decisions to include unanswered details from the
  assignment criteria.
- Added proposed project defaults for data, roster input, scoring, prediction
  horizon, trade interpretation, bench weighting, and course deliverables.
- Added the initial CSV roster template and implemented strict CSV parsing with
  explicit errors for unmatched and ambiguous player names.
- Implemented ROS aggregation, fantasy lineup optimization, bench-weighted team
  strength, trade simulation, and JSON reporting in `src/trades/`.
- Added `data/trade_template.csv` and documented the valuation/evaluation CLI
  commands in `PROJECT_OUTLINE.md`.
- The current full suite passes (32 tests); `pip check` and `git diff --check`
  pass. Tests include synthetic end-to-end valuation and trade evaluation.
- Reviewed `nflreadpy` loader documentation and confirmed it provides weekly
  player stats, historical FantasyPros rankings, and player ID mappings.
- Confirmed the 2021–2025 stats include `season`, `week`, `player_id`, and
  `fantasy_points_ppr`, with no missing PPR values in any season.
- Season row counts: 2021 (18,969), 2022 (18,831), 2023 (18,643), 2024
  (18,983), and 2025 (19,422).
- Confirmed ranking data includes weekly ranking categories and scrape dates
  from 2019-12-27 through 2026-11-20.
- `.\.venv\Scripts\python.exe -m src.data.collect --seasons 2021 2022 2023
  2024 2025` completed successfully.
- `.\.venv\Scripts\python.exe -m pytest -q` passes (3 tests);
  `.\.venv\Scripts\python.exe -m pip check` reports no broken requirements;
  `git diff --check` passes.
- `.\.venv\Scripts\python.exe -m src.data.clean --seasons 2021 2022 2023
  2024 2025` completed successfully and wrote the processed dataset and local
  copies of roster, team-stat, and schedule inputs.
- The cleaned dataset has no duplicate `(season, week, player_id)` keys and
  includes regular-season rows only. Its player-score audit found 29,687
  skill-position rows matching the source PPR score exactly; the remaining
  differences are explained by ESPN fumble-loss scoring, fumble-recovery TDs,
  and one RB who kicked a field goal.
- `.\.venv\Scripts\python.exe -m pytest -q` passes (8 tests), covering PPR and
  kicker calculations, zero-fill provenance, D/ST bye rows, D/ST points
  allowed treatment, and canceled-game labeling.
- `.\.venv\Scripts\python.exe -m src.features.build --seasons 2021 2022 2023
  2024 2025 --history-seasons 2020` completed successfully and wrote the
  56,952-row feature dataset.
- Collected 17,602 weekly player-stat rows and cleaned 11,015 player-week rows
  for 2020 history-only use.
- Feature output has 56,952 unique player-week keys and a 57-column
  `FEATURE_COLUMNS` allowlist that excludes the target and current-week raw
  stats. Josh Allen's 2021 Week 1 previous-week score is 19.26, matching his
  2020 Week 17 score; current-season-to-date averages correctly reset to null.
- Rank coverage now includes redraft PPR fallbacks for early weeks; 55,029 rows
  have an eligible rank, with the source recorded as weekly or redraft. The
  remaining 1,923 rows have no eligible mapped rank.
- Injury statuses are available for 2021–2024. The 2025 injury records have no
  `date_modified` timestamps, so they cannot be safely used as-of a kickoff and
  remain unknown.
- Josh Allen's 2024 Week 2 record has target 9.76, previous-week score 31.18,
  and Miami matchup context from its prior game: 162 passing yards, 128 rushing
  yards, and 17 points allowed.
- `.\.venv\Scripts\python.exe -m pytest -q` passes (14 tests), including
  cross-season history, rank fallback and cutoff behavior, injury cutoff,
  unmatched-player flags, and D/ST rank mapping.
- `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` completed;
  `.\.venv\Scripts\python.exe -m pip check` reports no broken requirements.
- `.\.venv\Scripts\python.exe -m pytest -q tests/test_train_model.py` passes
  (1 test); `.\.venv\Scripts\python.exe -m pytest -q` passes (15 tests).
- `.\.venv\Scripts\python.exe -m src.models.train --feature-path
  data\processed\player_week_features_2021-2022-2023-2024-2025.parquet
  --output-dir models` completed after the history fix. It selected
  `mlp_32_alpha_0.001_with_injury`, with validation MAE/RMSE 3.443/5.302 and
  2025 test MAE/RMSE 3.361/5.243. Rolling 3-/5-week baseline test MAEs were
  4.585/4.498. The MLP reached its 100-iteration limit without convergence.
- `.\.venv\Scripts\python.exe -m src.models.predict --season 2026` completed
  at 2026-11-25 07:22 UTC. It generated forecasts for Weeks 12–18, 4,634
  rows total. Audit confirmed 662 rows per week, 125 bye-week rows with zero
  predictions, no non-bye rows missing opponents, and no duplicate player-week
  keys.
- The prediction artifact records the missing 2026 injury timestamp and the
  frozen-through-Week-11 history policy in
  `models/weekly_predictions_2026.json`.
- `.\.venv\Scripts\python.exe -m src.models.predict --season 2026
  --strict-preseason` completed. It generated 12,042 predictions for Weeks
  1–18 using only Week 1 rosters, pre-opening rankings, and player history
  through 2025. Metadata records an exclusive ranking cutoff before
  2026-09-09 and confirms that injury and 2026 matchup-history features are
  excluded/unknown.
- The strict-preseason audit found 669 rows per week, 669 bye rows with exact
  zero predictions, no non-bye candidates missing opponents, and no duplicate
  player-week keys. A regression test verifies schedule scores are masked and
  same-opening-day rank snapshots are excluded.
- Rebuilt the 56,952-row feature set with
  `.\.venv\Scripts\python.exe -m src.features.build --seasons 2021 2022
  2023 2024 2025 --history-seasons 2020`.
- Regenerated the live 2026 forecast (Weeks 12–18, 4,634 rows) and strict
  preseason forecast (Weeks 1–18, 12,042 rows) using the retrained model.
- `.\.venv\Scripts\python.exe -m pytest -q` passes (20 tests), including a
  regression test that preserves synthetic zero targets but excludes them
  from player-performance history;
  `.\.venv\Scripts\python.exe -m pip check` reports no broken requirements.
- `git diff --check` passes.

## Known Issues

- FantasyPros rankings are position-specific, not a single overall ranking.
  Redraft PPR ranks fill many early-week gaps; some players still lack an
  eligible mapped rank. Date-only snapshots on the first-kickoff date remain
  excluded.
- The 2025 injury feed rows have null `date_modified` values, so their statuses
  cannot be verified against the weekly kickoff cutoff and are left unknown.
- The nflverse source PPR score does not apply the ESPN -2 fumble-lost penalty
  or add fumble-recovery TD points, and kicker source scores are not populated;
  the processed `full_ppr_points` column applies those project rules.
- Roster-confirmed zero rows depend on the weekly roster source and the selected
  ACT/INA/RES statuses; check this universe if future source versions change.
- Previous player-week and rolling history cross season boundaries when the
  history season is supplied. Season-to-date means and team matchup history
  reset each season. The default CLI requires the season immediately preceding
  the target range as a history-only cleaned input.
- Player rolling windows use observed stat weeks only; target zero-fill rows
  are retained for actual fantasy outcomes. Matchup windows use prior team
  games.
- The roster CSV supports one position per player and a single fixed lineup
  configuration; multi-position eligibility and custom league slots are not
  implemented.
- Trade reports do not estimate prediction uncertainty, so small positive
  deltas may not be meaningful.
- MLP candidates still reach the configured 100-iteration limit without
  convergence; the current selected model is Ridge regression. Treat the
  resulting predictions as an initial benchmark, not a guarantee of predictive
  improvement.
- The `src.data.clean --seasons 2026` command expects completed schedules and
  fails when the source includes future games without team stats. Step 5
  therefore cleans only fully completed weeks before building forecast rows.
- Future roster membership uses the latest available weekly roster snapshot;
  later transactions are unknown until the forecast is refreshed. Player/team
  history is held at the latest completed week across each projected week.
- No verifiable timestamps exist in the 2026 injury feed, so injury features
  are unknown rather than interpreted as healthy.
- Strict-preseason forecasts intentionally omit 2026 matchup history and
  injury information and retain the Week 1 roster for every week; they are
  season-start projections, not live rest-of-season projections.
- Synthetic zero-fill rows do not inform player performance-history features,
  but remain zero-valued training outcomes to represent missed fantasy weeks.
- Starter status is approximated by FantasyPros position ranks within each
  team's player pool, not an official depth chart. Unranked players fill
  remaining slots by conditional score.
- The two-thirds bench weight is arbitrary and has not yet been validated.
- The abstract deadline year and any additional formatting or disclosure rules
  need confirmation from the course schedule or instructor.

## Next Steps

- Resolve the open decisions listed in `PROJECT_OUTLINE.md`.
- Confirm course-specific deadline and submission details omitted from the screenshot.
- Run the new valuation and trade-evaluation commands with generated weekly
  forecasts and the project's real roster/trade CSVs.
- Compare trade results across bench weights and decide whether to add a
  validation-based uncertainty estimate.
- Keep this file synchronized with future milestones, validation results, and remaining work.
