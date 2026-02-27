# cluster_rar

## What this dataset is
- Cluster-scale RAR reference data and archived catalogs.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `HIFLUGCS_Clusters.zip`
- `eROClusters.zip`
- `tian2020_fig2.dat`
- `tian2020_table1.dat`

## Preprocessing scripts involved
- `analysis/gdagger_hunt.py`
- `analysis/pipeline/02_cf4_rar_pipeline.py`
- `analysis/pipeline/03_fetch_cf4_distances.py`
- `analysis/pipeline/04_yang_crossmatch.py`
- `analysis/pipeline/05_wallaby_rar_pipeline.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
