# Phase 4 Unified Model Adjudication

Generated (UTC): 2026-02-22T04:10:09.639199+00:00

## Claim Decisions

| Claim | Status | Key metrics |
|---|---|---|
| Mass-dependent bunching should strengthen toward high mass / low X | **rejected** | rho_logMs_vs_daic=-0.5; rho_X_vs_daic=0.5 |
| Forward-model preference should become more BEC-like at higher stellar mass | **rejected** | rho_logMs_vs_daic=-0.0518 |
| Lc–xi scaling should remain significant after size controls | **weak** | raw_p=1.3349800418972305e-06; controlled_log_xi_p=0.0812 |
| Phase-diagram peak near g_dagger should discriminate from null/LCDM and hold up in validation | **weak** | perm_p=0.0 |
| Galaxy-to-cluster scaling should be internally consistent with xi~sqrt(M) and g_dagger,eff mass trend | **supported** | alpha_p=2.2238486158373725e-23 |

## Summary

- Supported: 1
- Weak: 2
- Rejected: 2
- Missing: 0
- Overall: **NOT_SUPPORTED_BY_COMBINED_TESTS**

## Notes

- Phase-diagram summary is read from analysis/pipeline/results; if a new long-running phase-diagram job is active, rerun this script after it completes.

## Source Files

- `mass_split_bunching`: `/Users/russelllicht/bec-dark-matter/analysis/results/summary_mass_split_bunching.json` (exists=True, mtime_utc=2026-02-22T04:08:26.436357+00:00)
- `forward_model_bunching`: `/Users/russelllicht/bec-dark-matter/analysis/results/summary_forward_model_bunching.json` (exists=True, mtime_utc=2026-02-22T03:45:06.455184+00:00)
- `healing_length_scaling`: `/Users/russelllicht/bec-dark-matter/analysis/results/summary_healing_length_scaling.json` (exists=True, mtime_utc=2026-02-22T03:45:11.343327+00:00)
- `hierarchical_healing_length`: `/Users/russelllicht/bec-dark-matter/analysis/results/summary_hierarchical_healing_length.json` (exists=True, mtime_utc=2026-02-22T03:45:06.107762+00:00)
- `phase_diagram_model`: `/Users/russelllicht/bec-dark-matter/analysis/pipeline/results/summary_phase_diagram_model.json` (exists=True, mtime_utc=2026-02-20T00:23:04.158412+00:00)
