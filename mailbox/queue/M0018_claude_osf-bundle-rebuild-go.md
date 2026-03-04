---
message_id: M0018
task_id: null
from: claude
to: codex
type: handoff
timestamp_utc: "2026-03-04T17:00:00Z"
repo_state:
  git_sha: "bc7fb1635c8560acb1eb1aed60c4451ff9096cbd"
  dirty: true
references:
  - M0017
  - M0016
  - M0015
---

OSF BUNDLE REBUILD — APPROVED BY RUSSELL

Russell has reviewed and approved the merged manuscript (`paper/main_merged.tex`,
17 pages, clean pdflatex build). Green light to rebuild the full OSF bundle.

---

## Scope

Rebuild the OSF bundle from `paper/main_merged.tex` to reflect the full
35-analysis merged paper (23 primary + 12 supplementary diagnostics).

## Checklist

1. **Compile final PDF** from `paper/main_merged.tex` (3-pass pdflatex + bibtex).
   Place as the canonical paper PDF in the bundle.

2. **Stage 87 files** using the updated `tools/osf_packaging/stage_for_osf.sh`.
   Verify PASS with breakdown: core 16, supporting 7, supplementary 12,
   metadata 2, datasets 48, wiki 2.

3. **Regenerate MANIFEST.csv** — must list all 87 staged files with paths and
   descriptions.

4. **Regenerate CHECKSUMS.csv** — SHA256 for every file in the bundle.

5. **Update reproducibility.md** — reflect 35-analysis scope (23 primary + 12
   supplementary), merged paper title, updated file counts.

6. **Regenerate `tests_results_osf.html`** from updated `build_osf_html.py` —
   confirm Supplementary Diagnostics (12) section present, zero TODO findings.

7. **Copy source LaTeX** into bundle `source/` directory — `main_merged.tex`,
   `references.bib`, `tables/*.tex`, all figure files.

8. **Verify the supplementary/ staging directory** contains exactly 12 JSON files
   matching the list in M0016.

## Design Decisions (confirmed in M0017)

- `supplementary/` stays as a distinct OSF namespace (not folded into `supporting/`)
- Verdict strings accepted as-is (free-text diagnostic summaries)
- `build_osf_html.py` precedence fix (override > record_summary) is approved

## Acceptance Criteria

- [ ] `stage_for_osf.sh` returns PASS, total 87 files
- [ ] MANIFEST.csv has 87 entries
- [ ] CHECKSUMS.csv has matching SHA256 for every file
- [ ] `tests_results_osf.html` has no TODO findings
- [ ] PDF compiles cleanly (no errors, no unresolved refs)
- [ ] reproducibility.md references 35 analyses
- [ ] `source/` directory contains `main_merged.tex` (not stale `main.tex`)

## DO NOTs

- Do not modify any test scripts
- Do not modify `main_merged.tex` (Russell has approved the current version)
- Do not delete `gdagger_hunt_paper.tex` or `main.tex` (keep as references)
- Do not upload to OSF yet — stage and verify only, Russell will trigger upload

VERIFICATION
git_sha: bc7fb1635c8560acb1eb1aed60c4451ff9096cbd
