# Mailbox Communication Protocol

Bidirectional agent communication system for the bec-dark-matter repository.

## Agent Roles

| Agent | Role | Strengths |
|-------|------|-----------|
| **Claude** | Architect, reviewer, manager | Theory, schema design, methodology review, nuanced judgment, long-context reasoning |
| **Codex** | Implementer, validator | Code execution, computation, logic, file manipulation, data integrity |
| **Human** | Relay, authority, tiebreaker | Final approval, conflict resolution, external actions (deploy, upload) |

The human relays messages between agents by copy-pasting from `mailbox/queue/` into each agent's prompt. Agents write messages formatted for the *recipient's* optimal prompting architecture.

## Message Format

### Frontmatter (YAML, required for all messages)

```yaml
---
message_id: M{NNNN}          # Sequential, zero-padded to 4 digits
task_id: T{NNN}              # Links to active_task.md
from: {claude|codex}
to: {claude|codex}
type: {handoff|review_request|review_response|challenge_response}
timestamp_utc: "YYYY-MM-DDTHH:MM:SSZ"
repo_state:
  git_sha: "{40-char hex}"
  dirty: {true|false}
---
```

### Body: Two Templates

Messages use the template optimized for the **recipient**:

- **To Codex** (`templates/claude_to_codex.md`): Imperative, action-biased, numbered deliverables, inline code paths, fenced code blocks, no preambles.
- **To Claude** (`templates/codex_to_claude.md`): XML tags (`<context>`, `<documents>`, `<instructions>`, `<verification>`, `<examples>`), context-first, long documents at top, queries at bottom.

See `mailbox/templates/` for full template specs.

## File Locations

| Purpose | Path | Naming Convention |
|---------|------|-------------------|
| Active messages | `mailbox/queue/*.md` | `M{NNNN}_{from}_{slug}.md` |
| Archived messages | `mailbox/archive/*.md` | Same as queue |
| Challenges | `mailbox/challenges/*.md` | `C{NNN}_{from}_{slug}.md` |
| Current task | `mailbox/active_task.md` | Single file, overwritten |
| Sequence counter | `mailbox/.seq` | Single file |
| Templates | `mailbox/templates/*.md` | Fixed names |

## Message Numbering

Counters are stored in `mailbox/.seq`:

```
message_seq: 1
challenge_seq: 0
```

- **Messages**: `M0001`, `M0002`, ... (zero-padded to 4 digits)
- **Challenges**: `C001`, `C002`, ... (zero-padded to 3 digits)
- **Tasks**: `T001`, `T002`, ... (sequential, set in `active_task.md`)

After writing a message or challenge, increment the relevant counter in `.seq`.

## Task Lifecycle

Only one active task at a time, defined in `mailbox/active_task.md`.

```
Human creates active_task.md
  status: PENDING
    |
    v
Agent picks up task
  status: IN_PROGRESS (owner: {agent})
    |
    +---> Work done, handoff written to queue/
    |     status: IN_PROGRESS (owner changes)
    |
    +---> Agent blocked
    |     status: BLOCKED (blocker described)
    |
    v
All steps complete
  status: COMPLETED (human closes)
  Messages moved: queue/ -> archive/
```

### active_task.md Format

```yaml
---
task_id: T{NNN}
title: "Short task title"
status: {PENDING|IN_PROGRESS|BLOCKED|COMPLETED}
owner: {claude|codex|human}
created_utc: "YYYY-MM-DDTHH:MM:SSZ"
updated_utc: "YYYY-MM-DDTHH:MM:SSZ"
---

## Objective
What needs to be accomplished.

## Steps
1. [ ] Step description (owner: agent)
2. [ ] Step description (owner: agent)

## Blocker (if BLOCKED)
Description of what is blocking progress.
```

## Verification Protocol

Before starting any work from a mailbox message, the receiving agent **MUST**:

1. Read `repo_state.git_sha` from the message frontmatter.
2. Run `git rev-parse HEAD` (or equivalent).
3. **If MATCH** -> Proceed normally.
4. **If HEAD is AHEAD** -> Check whether the message's referenced files were modified in intervening commits.
   - If referenced files are unmodified -> safe to proceed.
   - If referenced files were modified -> warn human before proceeding.
5. **If HEAD is BEHIND** -> Refuse to proceed. Ask human to update the repo.
6. **If dirty working tree** -> Warn human. Do not proceed until tree is clean or human approves.

## Red Team / Challenge Protocol

Either agent can challenge the other's work. Challenges are filed in `mailbox/challenges/`.

### Severity Levels

| Level | Meaning | Effect |
|-------|---------|--------|
| `critical` | Blocks all work | Must resolve before any further progress |
| `high` | Significant concern | Should resolve before next handoff |
| `medium` | Notable issue | Address in current task cycle |
| `low` | Minor suggestion | Address when convenient |

### Categories

- `correctness` — factual or computational errors
- `methodology` — flawed approach or reasoning
- `completeness` — missing cases, untested paths
- `efficiency` — unnecessary complexity or waste

### Challenge Format

```yaml
---
challenge_id: C{NNN}
task_id: T{NNN}
from: {claude|codex}
to: {claude|codex}
severity: {critical|high|medium|low}
category: {correctness|methodology|completeness|efficiency}
timestamp_utc: "YYYY-MM-DDTHH:MM:SSZ"
---
```

### Rules

1. Challenged agent must respond in their next message (type: `challenge_response`).
2. Response must be one of:
   - **Accept + fix**: Acknowledge the issue, describe the fix, implement it.
   - **Reject + explain**: Explain why the challenge is invalid or does not apply.
3. Human has veto power over any challenge resolution.
4. Accepted challenges that result in code changes produce a run log with `correction_of` linking to the original run.

### Role-Based Challenge Expectations

- **Claude challenges**: methodology, schema design, test coverage, theoretical interpretation.
- **Codex challenges**: computational correctness, implementation bugs, performance, data integrity.

## Run Log Linkage

Every handoff that includes code changes should reference the associated run log:
- The run log's `task_id` field (optional) links to the mailbox task.
- Run logs are created via `tools/osf_packaging/log_run.py --task-id T{NNN}`.

## Graceful Degradation

If the mailbox system is unavailable or impractical:

1. Agents may communicate via inline prompts (the pre-mailbox method).
2. Any work done outside the mailbox should be retroactively logged:
   - Create a message in `mailbox/queue/` summarizing what was done.
   - Set `type: handoff` and note "retroactive" in the body.
3. The sequence counter in `.seq` must still be incremented.
4. Run logs are always required regardless of whether the mailbox is used.
