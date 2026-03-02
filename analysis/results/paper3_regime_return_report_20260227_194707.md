# Paper3 Stability & Regime Return Report (20260227_194707)

## Stamp
- timestamp_utc: 2026-02-27T19:47:07.532620+00:00
- git_head: f8f914933f3b0e3d9b2d1935c6377cd1f5db9a73
- current_rar_points_csv: /Users/russelllicht/bec-dark-matter/analysis/results/rar_points_unified.csv
- current_rar_points_csv_sha256: 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c

## Parsed Run Comparison
| label | run | actual_csv_sha | input_rows | after_nonfinite_drop | after_physical_cuts | N_gal | slope | bootstrap_ci_lo | bootstrap_ci_hi | bootstrap_p | pooled_shift | pooled_perm_p | pergal_shift | pergal_perm_p | sign_consistent | perm_p_lt_005_all_windows | include_ss20 | ss20_excluded_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stable_early_162156 | 20260227_162156 | None | 15745 | 11369 | 11369 | 113 | 0.043813 | 0.000764 | 0.135792 | 0.044 | 0.184776 | 9.999e-05 | 0.020178 | 0.129887 | True | True | None | None |
| broken_drift_175543 | 20260227_175543 | None | 9974 | 8679 | 8679 | 61 | -0.020133 | -0.160255 | 0.122605 | 0.8144 | 0.156497 | 9.999e-05 | 0.103712 | 0.428557 | False | False | None | None |
| restored_default_193632 | 20260227_193632 | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | 12621 | 11326 | 11326 | 112 | 0.006696 | -0.027697 | 0.046488 | 0.6832 | 0.027282 | 9.999e-05 | -0.005223 | 0.693731 | True | False | False | 40 |
| restored_include_ss20_193650 | 20260227_193650 | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | 12621 | 11326 | 11326 | 112 | 0.006696 | -0.027697 | 0.046488 | 0.6832 | 0.027282 | 9.999e-05 | -0.005223 | 0.693731 | True | False | True | 0 |
| restored_enforced_194611 | 20260227_194611 | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | 12621 | 11326 | 11326 | 112 | 0.006696 | -0.027697 | 0.046488 | 0.6832 | 0.027282 | 9.999e-05 | -0.005223 | 0.693731 | True | False | False | 40 |


## Interpretation
- Broken drift run (`20260227_175543`) shows dataset contraction to N_gal=61 with input_rows=9974; restored runs return to N_gal=112 with input_rows=12621 once SPARC per-radius points are present again.
- Stable early run (`20260227_162156`) had N_gal=113; restored default run has N_gal=112 (delta=1).
- Remaining 113→112 delta is explained by galaxy `193-001` having only 19 points in the restored unified CSV (min_points=20), source breakdown={'Catinella2005': 19}.
- `--include_ss20` does not change eligibility here (112 vs 112) because SS20_* entries are single-point stubs and remain below the per-galaxy minimum-point floor.
- Enforced run (`20260227_194611`) confirms hash-locked execution (`require_csv_sha=11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c` == `actual_csv_sha=11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c`).

## Red-Team Hardening Impact
- Regression tests now fail if SPARC drops below expected loader/CSV floors.
- Paper3 can now refuse execution on hash mismatch via `--require_csv_sha` and on low SPARC coverage via `--require_min_sparc_points`.
- Report headers now surface dataset hash + source histograms + SS20 inclusion mode, making drift immediately visible.

## Inputs Parsed
- stable_early_162156: /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_162156/paper3_density_bridge_report.txt
- broken_drift_175543: /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_175543/paper3_density_bridge_report.txt
- restored_default_193632: /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_193632/paper3_density_bridge_report.txt
- restored_include_ss20_193650: /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_193650/paper3_density_bridge_report.txt
- restored_enforced_194611: /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/paper3_density_bridge_report.txt
