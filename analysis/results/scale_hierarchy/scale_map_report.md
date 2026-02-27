# Scale Map Report

## Data Integration
- Canonical galaxies: **23791** (22 sources)
- Structured SPARC sample: **67** galaxies
- Output table: `/Users/russelllicht/bec-dark-matter/analysis/results/scale_hierarchy/galaxy_scale_table.parquet`

## Within-Galaxy Test (Coherence vs R/xi)
- Galaxies used: **66**
- x_peak concentration: observed std=2.508, null mean std=3.949
- Xi-permutation p (smaller std than null): **0.0090**

## Between-Galaxy Matched Test (High-xi vs Low-xi)
- Unmatched Cliff's delta: **-0.469** [-0.690, -0.219], perm p=0.0080
- Primary matched (caliper=1.50, pairs=22): Cliff's delta=-0.620 [-0.835, -0.355], perm p=0.0030

## Dimensionless Scales
- `Lc_over_xi`: median=0.220, p16-p84=[0.060, 0.367], n=67
- `lambda_over_xi`: median=1.395, p16-p84=[0.520, 2.868], n=67
- `xi_over_Rext`: median=0.189, p16-p84=[0.125, 0.333], n=67

## Interpretation
- If x_peak concentration remains significant under xi-permutation null, xi is acting as an organizing radial scale.
- If matched high-vs-low xi effect survives geometry/sampling controls, xi carries independent information beyond size/sampling artifacts.
- Compare with universal-scale diagnostics (g† phase-peak tests) to maintain a two-scale interpretation: global acceleration scale + local coherence scale.

## Output Files
- `galaxy_table`: `/Users/russelllicht/bec-dark-matter/analysis/results/scale_hierarchy/galaxy_scale_table.parquet`
- `point_table`: `/Users/russelllicht/bec-dark-matter/analysis/results/scale_hierarchy/point_scale_table.parquet`
- `results_table`: `/Users/russelllicht/bec-dark-matter/analysis/results/scale_hierarchy/results_scale_hierarchy.parquet`
- `figure`: `/Users/russelllicht/bec-dark-matter/analysis/results/scale_hierarchy/fig_scale_map.png`
- `report`: `/Users/russelllicht/bec-dark-matter/analysis/results/scale_hierarchy/scale_map_report.md`
- `run_log`: `/Users/russelllicht/bec-dark-matter/analysis/results/scale_hierarchy/run_log.json`
