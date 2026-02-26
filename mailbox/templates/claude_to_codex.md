# Template: Claude -> Codex

Use this template when Claude writes a message for Codex to consume.
Codex reads AGENTS.md before every task. Prefers short, imperative, action-biased instructions.

---

## Template

```markdown
---
message_id: M{NNNN}
task_id: T{NNN}
from: claude
to: codex
type: {handoff|review_request|review_response|challenge_response}
timestamp_utc: "{YYYY-MM-DDTHH:MM:SSZ}"
repo_state:
  git_sha: "{40-char SHA}"
  dirty: false
---

{IMPERATIVE TITLE — no preamble, no greeting}

CONTEXT (2-3 sentences max)
State what was done, what changed, and why this handoff exists.

DELIVERABLES
1. In `path/to/file` — what to produce or change
2. In `path/to/other/file` — what to produce or change
3. Run `command` — expected outcome
4. ...

FILES CHANGED SINCE LAST HANDOFF
- `path/to/file` — what changed and why

ACCEPTANCE CRITERIA
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

VERIFICATION
Run: `git rev-parse HEAD` — must match {git_sha}

DO NOTs
- Do not modify X
- Do not change Y without Z
```

---

## Style Rules for Writing to Codex

1. **No preambles.** Start with the imperative title immediately after frontmatter.
2. **Action-biased.** Every sentence should drive toward a deliverable.
3. **Numbered deliverable lists.** 4-6 items per list. Each item = one concrete action.
4. **File paths in inline code.** Always use backtick-wrapped paths: `path/to/file`.
5. **Fenced code blocks** for any commands, config snippets, or expected output.
6. **"Deliver working code, not plans."** Do not ask Codex to "consider" or "think about" — ask it to produce.
7. **Acceptance criteria as checkboxes.** Codex can check them off as it works.
8. **DO NOTs section.** Explicit guardrails prevent scope creep.
