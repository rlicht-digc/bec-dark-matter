# Raw Data Schema

## Required directories
- `tng/`
  - `TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/`
  - `TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/`
  - `quarantine_contaminated/`
  - `raw_archives/`
  - `ingestion_staging/`
- `observational/`
  - `sparc/`
  - `cluster_rar/`
  - `cf4/`
  - `voids/`
  - `yang_catalogs/`
  - `wallaby/`
  - `alfalfa/`
  - additional survey/catalog directories as needed.

## Rules
- Do not store generated analysis outputs here.
- Quarantine corrupted/contaminated source payloads under `tng/quarantine_contaminated/` or an observational quarantine folder.
- Keep naming close to source provenance to retain traceability.
