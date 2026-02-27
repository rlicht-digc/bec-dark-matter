# Raw Data Hub

This folder contains source and raw datasets used across the project.

## Top-level layout
- `tng/`: TNG-derived raw and packaged datasets (verified, big-base, quarantine, archives, ingestion staging).
- `observational/`: non-TNG raw sources (SPARC, CF4, ALFALFA, WALLABY, clusters, void catalogs, literature tables, etc).

## Important notes
- Legacy paths are preserved for compatibility via symlinks:
  - `/Users/russelllicht/bec-dark-matter/data` -> `raw_data/observational`
  - `/Users/russelllicht/bec-dark-matter/datasets` -> `raw_data/tng`
  - `/Users/russelllicht/bec-dark-matter/Remaining_TNG` -> `raw_data/tng/ingestion_staging`
- Duplicate big-base TNG payloads were deduplicated; staging now links to the canonical big-base dataset.

## Quick navigation
- SPARC raw data: `observational/sparc/`
- Cluster-related data: `observational/cluster_rar/`
- TNG 3000x50 verified: `tng/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/`
- TNG 48133x50 big-base: `tng/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/`
