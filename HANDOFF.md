# Project Handoff

## Current Status

- Added `AGENTS.md` with repository-specific contributor guidance.
- Added Karpathy behavioral guidelines for cautious, focused implementation.
- Added a requirement to update this handoff regularly and immediately before each Git commit.
- Added `PROJECT_OUTLINE.md` with project, data, modeling, team-strength,
  trade-evaluation, testing, report, and implementation outlines.
- Defined the initial scope as full-PPR scoring with weekly predictions aggregated
  into rest-of-season player value.
- Defined trade fairness as both teams improving after the simulated trade.
- Set the provisional bench contribution to two-thirds of projected player value.
- Selected ESPN as the supported fantasy platform and the Python `espn-api`
  package as the primary data-collection interface.
- Scoped roster entry to manual input; private ESPN league synchronization and
  user credentials are not required for the school project.

## Verification

- Reviewed the repository layout and recent Git history.
- Confirmed Markdown has no whitespace errors with `git diff --check`.
- Reviewed the project outline against the decisions supplied by the project owner.

## Known Issues

- The repository remains a minimal scaffold with no source code, dependency manifest, or test suite.
- The historical fields available through `espn-api` have not been verified;
  a public reference league or another source may be needed for model-training
  features.
- The manual roster input format has not been selected.
- Detailed scoring rules, roster slots, prediction horizon, and the treatment of
  injuries and bye weeks have not been finalized.
- The two-thirds bench weight is arbitrary and has not yet been validated.

## Next Steps

- Resolve the open decisions listed in `PROJECT_OUTLINE.md`.
- Establish the project structure and dependency tooling when implementation begins.
- Add `espn-api` to the dependency manifest when implementation begins, then
  audit its available league and historical player-week data.
- Build a small, leakage-safe player-week dataset.
- Implement a weekly-score baseline before introducing more complex models.
- Keep this file synchronized with future milestones, validation results, and remaining work.
