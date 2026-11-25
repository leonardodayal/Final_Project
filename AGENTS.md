# Repository Guidelines

## Project Structure & Module Organization

This repository is currently a minimal project scaffold; no source, test, or data directories have been committed yet. As the machine-learning project grows, keep reusable Python code in `src/`, automated tests in `tests/`, exploratory notebooks in `notebooks/`, and documentation or report assets in `docs/`. Put small, shareable input samples in `data/`; do not commit large, generated, sensitive, or licensed datasets. Store trained artifacts in `models/` only when their size and provenance make version control appropriate.

## Build, Test, and Development Commands

The project requires Python 3.10 or later. Dependencies are listed in
`requirements.txt`. Set up and test the project with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Run modules from the repository root, for example
`python -m src.data.collect --seasons 2021 2022`, so imports behave
consistently. After collecting weekly stats, build the cleaned player-week
dataset with `python -m src.data.clean --seasons 2021 2022`.
Collect and clean the history-only season first with
`python -m src.data.collect --seasons 2020` and
`python -m src.data.clean --seasons 2020`.
Build leakage-safe prediction features with
`python -m src.features.build --seasons 2021 2022 --history-seasons 2020`.
Train and evaluate weekly-score regression models with
`python -m src.models.train --feature-path data\processed\player_week_features_2021-2022-2023-2024-2025.parquet --output-dir models`.
Refresh current-season player stats and rankings with
`python -m src.data.collect --seasons 2026`, then generate live weekly
projections with `python -m src.models.predict --season 2026`. Generate the
separate strict-preseason Weeks 1–18 forecast with
`python -m src.models.predict --season 2026 --strict-preseason`.

## Coding Style & Naming Conventions

Use four spaces for Python indentation and follow PEP 8. Name modules, functions, and variables with `snake_case`; classes with `PascalCase`; constants with `UPPER_SNAKE_CASE`. Keep notebooks focused on exploration and move repeatable preprocessing, training, and evaluation logic into `src/`. Prefer deterministic experiments: expose random seeds and keep configuration separate from implementation. Preserve LF line endings as configured by `.gitattributes`.

## Testing Guidelines

Use `pytest` when tests are added. Mirror source paths under `tests/` and name files `test_<module>.py` and test functions `test_<behavior>()`. Cover data validation, feature transformations, model input/output shapes, and metric calculations. Use small fixtures rather than production datasets. Every bug fix should include a regression test.

## Commit & Pull Request Guidelines

The history currently contains only `Initial commit`, so no detailed convention is established. Use short, imperative subjects such as `Add feature preprocessing pipeline`; keep each commit focused. Pull requests should explain the objective, implementation, validation commands and results, and any dataset or model provenance. Link relevant issues and include plots or metric tables when model behavior changes. Never commit credentials, API keys, local virtual environments, or large generated artifacts.

## Handoff Documentation

Keep `HANDOFF.md` current throughout the work. Update it frequently after meaningful milestones, decisions, discoveries, verification runs, or changes to the remaining plan so another contributor can resume without reconstructing context. Immediately before every Git commit, review and update `HANDOFF.md` with the latest status, files changed, commands and results, known issues, and next steps. Do not commit with stale handoff information.

# Karpathy Behavioral Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them—don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it—don't delete it.

When your changes create orphans:

- Remove imports, variables, or functions that your changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after."

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
