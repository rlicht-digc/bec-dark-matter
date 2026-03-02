---
message_id: M0005
task_id: T002
from: claude
to: codex
type: review_request
timestamp_utc: "2026-02-26T20:10:00Z"
repo_state:
  git_sha: "1ab9d4a8b8c9fd61211b08c0863fad9e8799e64b"
  dirty: true
---

Pre-commit review: 1,504 uncommitted changes need evaluation before bulk commit

CONTEXT
The working tree has accumulated 1,504 changes across the repo restructure (data/ → raw_data/observational/), new pipeline scripts, new results, and cleanup. Russell wants everything committed and pushed, but the scope is large enough to warrant a review pass first.

## Change Summary

### Deleted Files (854)
Primary action: data directory migration from `data/` to `raw_data/observational/`.

| Directory | Files Deleted | Notes |
|---|---|---|
| data/hi_surveys/ | 500 | Moved to raw_data/observational/hi_surveys/ |
| data/sofue/ | 256 | Moved to raw_data/observational/sofue/ |
| data/brouwer2021/ | 32 | Moved to raw_data/observational/brouwer2021/ |
| data/probes/ | 16 | Moved to raw_data/observational/probes/ |
| data/wallaby/ | 9 | Moved to raw_data/observational/wallaby/ |
| data/sparc/ | 8 | Moved to raw_data/observational/sparc/ |
| data/misc/ | 7 | Moved to raw_data/observational/misc/ |
| data/littlethings/ | 7 | Moved to raw_data/observational/littlethings/ |
| data/yang_catalogs/ | 4 | readme files (binaries already gitignored) |
| data/environment/ | 4 | Moved to raw_data/observational/environment/ |
| data/cf4/ | 4 | Moved to raw_data/observational/cf4/ |
| data/alfalfa/ | 2 | Moved to raw_data/observational/alfalfa/ |
| docs/ | 3 | Removed: fluid_dm_roadmap.docx, SPARC_RAR_BEC_Project_Roadmap.md, Primordial_Fluid_DM_Research_Roadmap.docx |
| analysis/results/ | 1 | summary_self_consistent_mass.json (deprecated) |
| RoadmapDM.md | 1 | Old context file, superseded |

### New Files (645)
| Directory | Files Added | Notes |
|---|---|---|
| raw_data/observational/ | 385 | Restructured data + TERMS_OF_USE.md + CITATION.md per dataset (24 datasets) |
| analysis/pipeline/ | 80 | New test scripts, utilities, integration scripts |
| analysis/results/ | 76 | New summary JSONs, CSVs, figures, reports |
| raw_data/tng/ | 42 | IllustrisTNG cross-validation data |
| raw_data/ (top-level) | 2 | README.md, SCHEMA.md |
| raw_data/archives/ | 1 | Archive directory |
| tools/osf_packaging/ | 1 | log_run.py (new) |
| Other (figures/, paper/, scripts/, outputs/, meta/, logs/) | ~38 | Various outputs and working files |

### Modified Files (5)
- `analysis/pipeline/test_env_scatter_definitive.py` — pipeline update
- `analysis/pipeline/test_forward_model_bunching.py` — pipeline update
- `analysis/pipeline/test_mass_split_bunching.py` — pipeline update
- `analysis/results/summary_forward_model_bunching.json` — updated results
- `docs/BEC_DM_THEORY.md` — theory doc update

## Questions for Codex Review

1. **Data migration completeness**: Do the 385 new files in `raw_data/observational/` account for all 854 deletions from `data/`? (Some data files may be gitignored in new location — e.g., yang_catalogs binaries, tempel TSVs, probes profiles, s4g FITS.)

2. **Sensitive files check**: Are any of the 645 new untracked files inappropriate for commit? Look for:
   - Large binaries (.fits, .parquet, .tar.gz, .dat) that should be gitignored
   - API keys or credentials
   - Temporary/cache files

3. **Pipeline import paths**: Do the 80 new pipeline scripts reference `data/` paths that should now be `raw_data/observational/`? Or do they already use the new paths?

4. **Outputs directory**: Should `outputs/`, `outputs_bh/`, `outputs_bh3/`, `rerun_outputs/`, `logs/` be committed or gitignored as ephemeral run artifacts?

5. **Commit strategy**: One bulk commit, or split into logical groups (data migration, new pipeline, new results, cleanup)?

ACTION REQUESTED
Review the above and respond with:
- Any files that should NOT be committed (add to .gitignore)
- Any missing files or broken references
- Recommended commit strategy
