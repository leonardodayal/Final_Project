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
- Historical rankings remain separate until a leakage-safe week alignment is
  defined.
- Step 2 files changed: `.gitignore`, `AGENTS.md`, `HANDOFF.md`,
  `PROJECT_OUTLINE.md`, `src/data/clean.py`, and `tests/test_clean.py`.

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

## Known Issues

- Historical ranking rows have `scrape_date`, `page_type`, and `ecr_type`
  fields rather than explicit season/week fields; their alignment to player-week
  stats still needs to be defined.
- The nflverse source PPR score does not apply the ESPN -2 fumble-lost penalty
  or add fumble-recovery TD points, and kicker source scores are not populated;
  the processed `full_ppr_points` column applies those project rules.
- Roster-confirmed zero rows depend on the weekly roster source and the selected
  ACT/INA/RES statuses; check this universe if future source versions change.
- The manual roster input format has not been selected.
- Roster slots, prediction horizon, and the treatment of injury risk in
  predictions have not been finalized.
- The two-thirds bench weight is arbitrary and has not yet been validated.
- The abstract deadline year and any additional formatting or disclosure rules
  need confirmation from the course schedule or instructor.

## Next Steps

- Resolve the open decisions listed in `PROJECT_OUTLINE.md`.
- Confirm course-specific deadline and submission details omitted from the screenshot.
- Align historical rankings to the player-week data and document any gaps.
- Review the cleaned dataset and begin Step 3 feature engineering with
  strictly pre-week information.
- Implement a weekly-score baseline before introducing more complex models.
- Keep this file synchronized with future milestones, validation results, and remaining work.
