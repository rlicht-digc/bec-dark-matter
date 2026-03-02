---
message_id: M0001
task_id: T001
from: claude
to: codex
type: handoff
timestamp_utc: "2026-02-26T02:00:00Z"
repo_state:
  git_sha: "1368424d980be56dd626346231c1a9acd37c974d"
  dirty: false
---

Execute Step 3: move plan, OSF page regeneration, and upload list

CONTEXT
Step 2 review is complete. Claude fixed 8 summary JSONs for schema conformance and produced `claude_review_step2.md`. The alignment inventory from Step 1 is in `evidence_vault/reproducibility/alignment_inventory.csv`. Now produce the move plan, regenerate OSF HTML pages, and create the upload list.

DELIVERABLES
1. In `evidence_vault/reproducibility/move_plan.csv` — generate a move plan mapping current file locations to their OSF-target paths, based on the alignment inventory and review findings.
2. In `analysis/results/tests_results_osf.html` — regenerate the OSF tests results HTML page incorporating all fixed summary JSONs.
3. In `public_osf/osf_upload_list.md` — produce a finalized upload list of all files to be included in the OSF package.
4. Update `.gitignore` if needed to exclude any generated intermediate files.
5. Create a run log via `python3 tools/osf_packaging/log_run.py` documenting this step.

FILES CHANGED SINCE LAST HANDOFF
- `analysis/results/summary_*.json` — 8 files fixed: added `description` fields, renamed `test` to `test_name`
- `evidence_vault/reproducibility/claude_review_step2.md` — Step 2 review memo

ACCEPTANCE CRITERIA
- [ ] `move_plan.csv` exists and maps every file from inventory to an OSF target path
- [ ] `tests_results_osf.html` renders correctly with all current summary data
- [ ] `osf_upload_list.md` lists every file destined for OSF upload
- [ ] Run log created and validates against schema
- [ ] All changes committed

VERIFICATION
Run: `git rev-parse HEAD` — should be at or ahead of `1368424`

DO NOTs
- Do not modify any existing run logs under `logs/runs/`
- Do not edit `claude_review_step2.md`
- Do not change the schema files
