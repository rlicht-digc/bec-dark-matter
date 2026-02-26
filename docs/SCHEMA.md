# Project Schema Contract

This document defines stable locations and data contracts for generated artifacts and append-only run logging.

## Output Locations
- Test summaries: `analysis/results/summary_*.json`
- OSF HTML artifacts: `analysis/results/tests_results_osf.html`, `analysis/results/references_osf.html`
- Run logs: `logs/runs/*.json`
- Schemas:
  - `docs/schemas/run_log_schema.json`
  - `docs/schemas/summary_schema.json`

## Summary JSON Contract
- File pattern: `analysis/results/summary_*.json`
- Required keys:
  - `test_name` (string)
  - `description` (string)
- Strongly recommended:
  - `verdict` (string)
- Optional:
  - `data_sources` (array of strings)
  - `artifacts` (object, for example figures/tables paths)
  - `metrics` (object with scalar or nested metrics)

Notes:
- `verdict` is strongly recommended because report generators use it as the primary finding when present.
- Missing `verdict` should be treated as a quality warning, not a schema hard-failure.

## Run Logging Contract
- Every substantive change must create exactly one new run log (or more, if split into phases).
- Logs are append-only:
  - Never modify old run logs.
  - Never delete old run logs.
  - Corrections are new logs with `correction_of`.
- Run logs must validate against `docs/schemas/run_log_schema.json`.

## Run Log Naming Convention
- Recommended file name:
  - `logs/runs/<YYYYMMDDTHHMMSSZ>_<tool>_<purpose-slug>_<suffix>.json`
- `run_id` should match file stem where practical.
- `tool` must be one of: `codex`, `claude`, `human`.

## Minimal Logging Workflow
1. Commit code or docs changes first (so HEAD points to the change).
2. Create the run log:
   - `python3 tools/osf_packaging/log_run.py --tool codex --purpose "<purpose>" --inputs "<paths>" --outputs "<paths>"`
   - The tool captures `git HEAD` automatically. To record a specific commit (e.g. retroactive logging), pass `--git-commit <sha>`.
3. Commit the log file (or use `tools/osf_packaging/commit_log.sh`).

## Optional Local Hook Setup
Hooks are optional and local only. To enable project hooks locally:
- `git config core.hooksPath .githooks`

Do not rely on hooks for policy enforcement. CI or manual review remains authoritative.

## Mailbox Contract

Bidirectional agent communication uses structured messages in the `mailbox/` directory.

- **Message queue**: `mailbox/queue/*.md`
- **Archive**: `mailbox/archive/*.md`
- **Challenges**: `mailbox/challenges/*.md`
- **Active task**: `mailbox/active_task.md`
- **Sequence counter**: `mailbox/.seq`

### Naming Conventions
- Messages: `M{NNNN}_{from}_{slug}.md` (e.g., `M0001_claude_step3-move-plan-osf-regen.md`)
- Challenges: `C{NNN}_{from}_{slug}.md` (e.g., `C001_codex_schema-validation-gap.md`)

### Run Log Linkage
Run logs may include an optional `task_id` field (pattern: `^T[0-9]{3,}$`) linking to the active mailbox task. This field is defined in `docs/schemas/run_log_schema.json`.

See `mailbox/PROTOCOL.md` for the full communication protocol specification.
