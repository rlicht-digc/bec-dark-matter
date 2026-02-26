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

## Agent Roles

| Agent | Primary Role | Strengths |
|-------|-------------|-----------|
| **Claude** | Architecture, theory, review, management | Schema design, methodology review, theoretical interpretation, nuanced judgment, long-context reasoning |
| **Codex** | Implementation, computation, logic, validation | Code execution, file manipulation, data integrity, deterministic operations, performance optimization |
| **Human** | Relay, authority, tiebreaker | Final approval, conflict resolution, external actions (deploy, upload, publish) |

## Communication Protocol

Agents communicate via structured messages in the `mailbox/` directory.

- **Message queue**: `mailbox/queue/*.md` — active messages awaiting action.
- **Archive**: `mailbox/archive/*.md` — completed messages, moved after task closes.
- **Challenges**: `mailbox/challenges/*.md` — red team challenges between agents.
- **Templates**: `mailbox/templates/` — message format specs per recipient.
- **Active task**: `mailbox/active_task.md` — current task definition (one at a time).
- **Sequence counter**: `mailbox/.seq` — message and challenge numbering.

### Message Format
- YAML frontmatter with `message_id`, `task_id`, `from`, `to`, `type`, `timestamp_utc`, `repo_state`.
- Body formatted for the **recipient's** optimal prompting style.
- File naming: `M{NNNN}_{from}_{slug}.md` for messages, `C{NNN}_{from}_{slug}.md` for challenges.

### Verification
Before starting work from a mailbox message, the receiving agent must verify `repo_state.git_sha` matches `git rev-parse HEAD`. See `mailbox/PROTOCOL.md` for full verification rules.

### Run Log Linkage
Run logs may include an optional `task_id` field linking to the active mailbox task. Use `--task-id T{NNN}` with `tools/osf_packaging/log_run.py`.

## Red Team Protocol

Either agent can file a challenge in `mailbox/challenges/`.

- **Severity levels**: `critical` (blocks work), `high`, `medium`, `low`.
- **Categories**: `correctness`, `methodology`, `completeness`, `efficiency`.
- Challenged agent must respond in their next message (accept + fix, or reject + explain).
- Human has veto power over any resolution.
- Accepted challenges producing code changes create a run log with `correction_of`.

**Role-based challenge expectations:**
- Claude challenges: methodology, schema design, test coverage, theoretical interpretation.
- Codex challenges: computational correctness, implementation bugs, performance, data integrity.

## Prompting Guidelines

When writing messages to another agent, format for the **recipient's** architecture:

- **Writing to Codex**: Imperative, action-biased, no preambles. Numbered deliverable lists (4-6 items). File paths in inline code. Fenced code blocks. "Deliver working code, not plans."
- **Writing to Claude**: XML tags (`<context>`, `<documents>`, `<instructions>`, `<verification>`, `<examples>`). Context-first, long documents at top, queries at bottom. Role framing. Include motivation behind instructions.
