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

## Verification

- Reviewed the repository layout and recent Git history.
- Confirmed Markdown has no whitespace errors with `git diff --check`.
- Reviewed the project outline against the decisions supplied by the project owner.

## Known Issues

- The repository remains a minimal scaffold with no source code, dependency manifest, or test suite.
- Data sources, detailed scoring rules, roster slots, prediction horizon, and the
  treatment of injuries and bye weeks have not been finalized.
- The two-thirds bench weight is arbitrary and has not yet been validated.

## Next Steps

- Resolve the open decisions listed in `PROJECT_OUTLINE.md`.
- Establish the project structure and dependency tooling when implementation begins.
- Select data sources and build a small, leakage-safe player-week dataset.
- Implement a weekly-score baseline before introducing more complex models.
- Keep this file synchronized with future milestones, validation results, and remaining work.
