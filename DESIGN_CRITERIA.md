# Final Project Design Criteria

## Purpose and Source

This document translates the course final-project instructions provided in the
assignment screenshot into requirements for the fantasy football trade
calculator. Course requirements are separated from project-specific criteria
derived to make those requirements actionable. This is not a grading rubric;
the screenshot does not specify detailed scoring weights or formatting rules.

## Course Requirements

| Area | Requirement | Acceptance criterion |
| --- | --- | --- |
| Team | The project team consists of 2–3 students. | The abstract and final report identify all team members. |
| Abstract | Submit a one-paragraph abstract to Gradescope by October 9. It must include team member names and a brief project description. | A concise, single-paragraph abstract with names and project summary is prepared and submitted. The screenshot does not specify the year for the deadline; confirm it against the course schedule. |
| Topic | The topic is flexible and should investigate a machine-learning aspect extending topics covered in lectures. | The report explains the machine-learning question and how the work extends or applies course material. |
| AI tools | AI-assisted tools, including Codex and Claude Code, are permitted. | Any AI-assisted project work is reviewed and incorporated by the team; the screenshot states permission but does not prescribe a disclosure format. |
| Final report | Submit a report of approximately 15 pages to Gradescope. | The final report is research-article style and includes, at minimum, introduction, methodology, results, discussion, and conclusions. |
| Code and repository | Upload project code developed or generated for the project to GitHub and include the repository link in the final report. | The report contains a working GitHub repository link, and the repository contains the project code. |

## Project-Specific Design Criteria

These criteria apply the course requirements to the current project scope in
`PROJECT_OUTLINE.md`; they are not additional instructions quoted from the
course assignment.

### Research and Method

- State the problem as predicting future full-PPR player scores and evaluating
  the effect of a proposed trade on both teams.
- Describe the data source, collection limits, scoring rules, preprocessing,
  feature construction, and prediction horizon.
- Ensure that each training example uses only information available before its
  target week.
- Compare machine-learning results with simple baselines and use chronological
  validation to avoid training on future observations.
- Report appropriate prediction metrics, including MAE and RMSE, overall and
  by position where the available data supports it.

### Results and Interpretation

- Show prediction results and explain how weekly predictions become
  rest-of-season player values.
- Explain how lineup construction, bench weighting, and trade deltas affect
  the team-strength comparison.
- Discuss limitations, uncertainty, data availability, and cases where the
  calculator's recommendation should not be treated as conclusive.
- Make claims in the report traceable to reported experiments or cited sources.

### Implementation and Reproducibility

- Keep reusable implementation code in the GitHub repository and provide
  instructions sufficient to reproduce the reported experiment.
- Record data-source provenance and avoid committing credentials, private league
  data, or datasets whose licenses do not permit redistribution.
- Include a small, runnable example of player predictions and before/after
  trade evaluation if project scope and data allow.

## Submission Checklist

- [ ] Confirm the team has 2–3 members and names are consistent across
  submissions.
- [ ] Confirm the abstract deadline and year with the course schedule.
- [ ] Submit the one-paragraph abstract to Gradescope.
- [ ] Complete an approximately 15-page research-style report with the required
  sections.
- [ ] Include the GitHub repository link in the report.
- [ ] Verify the repository contains the code developed or generated for the
  project and that reproduction instructions match the final implementation.
- [ ] Submit the final report to Gradescope.

## Source Limitation

The provided screenshot does not specify the project deadline year, report
formatting or citation style, a strict page-count range, grading weights, or an
AI-tool disclosure policy. Confirm any such requirements with the course
materials or instructor rather than inferring them from this document.
