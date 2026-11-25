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
- Installed Python 3.13 and project dependencies in the ignored `.venv/`.
- Collected the five completed seasons 2021–2025 locally: 94,848 weekly
  player-stat rows, 12,542 player-ID mappings, and 1,890,527 historical
  ranking rows.
- Step 1 raw collection is complete for the selected seasons; cleaning,
  joining rankings, validating scoring rules, and classifying missing weeks
  remain for Step 2.
- Step 1 files changed: `.gitignore`, `AGENTS.md`, `HANDOFF.md`,
  `PROJECT_OUTLINE.md`, `requirements.txt`, `src/__init__.py`,
  `src/data/__init__.py`, `src/data/collect.py`, and `tests/test_collect.py`.
- Scoped roster entry to manual input; private ESPN league synchronization and
  user credentials are not required for the school project.

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

## Known Issues

- Historical ranking rows have `scrape_date`, `page_type`, and `ecr_type`
  fields rather than explicit season/week fields; their alignment to player-week
  stats still needs to be defined.
- Verify that the source's `fantasy_points_ppr` calculation matches the
  project's final full-PPR scoring rules.
- The manual roster input format has not been selected.
- Detailed scoring rules, roster slots, prediction horizon, and the treatment of
  injuries and bye weeks have not been finalized.
- The two-thirds bench weight is arbitrary and has not yet been validated.
- The abstract deadline year and any additional formatting or disclosure rules
  need confirmation from the course schedule or instructor.

## Next Steps

- Resolve the open decisions listed in `PROJECT_OUTLINE.md`.
- Confirm course-specific deadline and submission details omitted from the screenshot.
- Align historical rankings to the player-week data and document any gaps.
- Build a small, leakage-safe player-week dataset.
- Implement a weekly-score baseline before introducing more complex models.
- Keep this file synchronized with future milestones, validation results, and remaining work.
