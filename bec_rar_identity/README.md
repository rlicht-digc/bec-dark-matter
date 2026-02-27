# BEC = RAR Identity Hub

This directory is the canonical, BH/singularity-excluded organization for BEC↔RAR identity work.

## Schema
- `datasets/`: canonical dataset links (SPARC unified, TNG 3000x50 verified, TNG 48133x50 big-base)
- `tests/core/scripts/`: core referee battery orchestration scripts
- `tests/extended/scripts/`: all discovered `test_*.py` BEC↔RAR analysis scripts
- `runs/`: run-level output roots (referee battery, universality audit, legacy outputs)
- `artifacts/summaries/`: linked `summary_*.json` files (mirrored by original path)
- `artifacts/figures/`, `artifacts/tables/`, `artifacts/markdown/`: linked plots/tables/docs relevant to tests
- `manifests/`: inventories, viability checks, relocation and cleanup logs
- `archive/root_orphans/`: previously ambiguous root-level artifacts moved out of project root
- `archive/nonviable/`: reserved for invalid/unusable assets if encountered

## Current inventory snapshot
- test scripts organized: `82` (core `7`, extended `75`)
- summary JSON artifacts indexed: `124`
- figure artifacts indexed: `82`
- table artifacts indexed: `106`
- markdown artifacts indexed: `24`
- datasets linked: `7` / `7`

## 48k dataset coverage
- `datasets/tng_48133x50_big_base/rar_points.parquet` points to the 48,133×50 big-base per-point dataset.

## Manifests to use
- `manifests/test_inventory.csv`
- `manifests/artifact_inventory.csv`
- `manifests/dataset_inventory.csv`
- `manifests/viability_report.json`
- `manifests/relocation_map.json`
- `manifests/cleanup_actions.json`
- `manifests/root_orphans_inventory.json`
- `manifests/nonviable_inventory.json`
