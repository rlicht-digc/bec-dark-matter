# Template: Codex -> Claude

Use this template when Codex writes a message for Claude to consume.
Claude responds best to XML tags, context-first structure, and long documents at top with queries at bottom.

---

## Template

```markdown
---
message_id: M{NNNN}
task_id: T{NNN}
from: codex
to: claude
type: {handoff|review_request|review_response|challenge_response}
timestamp_utc: "{YYYY-MM-DDTHH:MM:SSZ}"
repo_state:
  git_sha: "{40-char SHA}"
  dirty: false
---

<context>
What was done and why. Background motivation. What prompted this handoff.
Include enough detail for Claude to understand the full picture without
reading every changed file.
</context>

<documents>
<document index="1">
<source>path/to/changed/file</source>
<document_content>
Key excerpt, diff summary, or full content if short.
For large files, include only the relevant sections.
</document_content>
</document>
<document index="2">
<source>path/to/another/file</source>
<document_content>
...
</document_content>
</document>
</documents>

<instructions>
What Claude should do with this. Be specific:
- Review criteria and what to look for.
- Questions that need answers.
- Judgment calls to make.
- Architectural decisions needed.
- Whether to approve, revise, or redirect.
</instructions>

<verification>
git_sha: {40-char SHA}
Expected state: description of what the repo should look like.
Files that should exist, tests that should pass, etc.
</verification>

<examples>
(If applicable — show what a good review or response looks like.
Include 1-2 concrete examples of the expected output format.)
</examples>
```

---

## Style Rules for Writing to Claude

1. **XML tags structure everything.** Use `<context>`, `<documents>`, `<instructions>`, `<verification>`, `<examples>`.
2. **Context first.** Put background and motivation at the top so Claude understands "why" before "what."
3. **Long documents at top, queries at bottom.** Place `<documents>` before `<instructions>`.
4. **Role framing is helpful.** E.g., "You are reviewing this as the project architect."
5. **Include motivation behind instructions.** Don't just say "check X" — say "check X because Y could cause Z."
6. **Few-shot examples in `<examples>` tags.** Show the desired output format when possible.
7. **Nuance and judgment calls are welcome.** Claude handles ambiguity well — frame open questions explicitly.
8. **Provide enough context to reason independently.** Claude should not need to re-read the entire repo.
