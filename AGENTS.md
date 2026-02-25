# AGENTS.md

Repository-wide rules for all contributors (Codex, Claude, and humans).

## Required Behavior
- Create a run log for every substantive change.
- Use append-only logs under `logs/runs/`.
- Never edit or delete a prior run log file.
- If a prior log is wrong, create a new log with `correction_of` pointing to the original `run_id`.
- Never fabricate citations or source URLs.
- For `analysis/results/summary_*.json`, include `verdict` where applicable.

## Logging Rules
- Log files are immutable records.
- Corrections are new records, not rewrites.
- Run logs must validate against `docs/schemas/run_log_schema.json`.
- Use UTC timestamps in ISO-8601 form (for example: `2026-02-25T21:10:00Z`).

## Summary JSON Rules
- Minimum required keys: `test_name`, `description`.
- Strong recommendation: include `verdict` to avoid ambiguous downstream reporting.
- Optional but useful keys: `data_sources`, `artifacts`, `metrics`.

## Correct Examples
- New feature implementation:
  - Create code/docs changes.
  - Create log via `tools/osf_packaging/log_run.py`.
  - Commit code and log.
- Correcting an old log:
  - Do not edit old `logs/runs/<old>.json`.
  - Create new `logs/runs/<new>.json` with `"correction_of": "<old_run_id>"`.
  - Commit only the new correction log (or with related corrective code).
- Citation uncertainty:
  - If citation metadata is unknown, write a TODO note in docs.
  - Do not invent DOI/arXiv/URL values.
