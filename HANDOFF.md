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
- The full data run selected Ridge (`alpha=10`, injury features included).
  Validation MAE/RMSE were 3.429/5.222; 2025 test MAE/RMSE were 4.060/5.872.
  The selected model beat the best validation baseline by MAE, but its 2025
  test MAE was worse than the rolling 3-/5-week baselines (3.407/3.407).
  Preserve this holdout result; do not retune against 2025.
- Step 4 uses a training-fitted preprocessing pipeline: numeric median
  imputation and standardization, categorical imputation and one-hot encoding.
  The reference model, metrics, and prediction artifacts are included under
  `models/`; future generated files there remain ignored by default.
- Added `pandas` and `scikit-learn` dependencies and corrected `.gitignore` to
  ignore only root-level generated `models/`, so the Python source package
  `src/models/` remains trackable.
- Added an end-to-end Step 4 regression test for chronological splits and
  artifact generation. It shortens MLP iterations for test speed.

## Verification

- Reviewed the repository layout and recent Git history.
- Confirmed Markdown has no whitespace errors with `git diff --check`.
- Reviewed the project outline against the decisions supplied by the project owner.
- Reviewed the design criteria against the assignment screenshot and project outline.
- Updated the outline's open decisions to include unanswered details from the
  assignment criteria.
- Added proposed project defaults for data, roster input, scoring, prediction
  horizon, trade interpretation, bench weighting, and course deliverables.
- Added the initial CSV roster template; no parser or player-matching logic has
  been implemented yet.
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
  --output-dir models` completed. It selected `ridge_alpha_10_with_injury`,
  with validation MAE 3.429 and 2025 test MAE 4.060. The rolling 3-/5-week
  baselines scored test MAE 3.407/3.407, so the selected model did not beat
  those baselines on the held-out season. Both MLP candidates reached the
  100-iteration limit without convergence.
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
- Player rolling windows use prior player-week rows (including explicit zero
  weeks); matchup windows use prior team games.
- The manual roster input format has not been selected.
- Roster slots and prediction horizon have not been finalized.
- The Step 4 Ridge model did not outperform the rolling-score baselines on the
  held-out 2025 season; treat it as an initial benchmark, not evidence of
  predictive improvement. The MLP candidates did not converge within the
  configured 100 iterations.
- The two-thirds bench weight is arbitrary and has not yet been validated.
- The abstract deadline year and any additional formatting or disclosure rules
  need confirmation from the course schedule or instructor.

## Next Steps

- Resolve the open decisions listed in `PROJECT_OUTLINE.md`.
- Confirm course-specific deadline and submission details omitted from the screenshot.
- Proceed to Step 5: use the trained model to generate weekly predictions and
  aggregate them into rest-of-season player values. Retain the Step 4 test
  comparison and investigate model limitations without tuning on 2025.
- Keep this file synchronized with future milestones, validation results, and remaining work.
