# Download Instructions

This folder is treated as non-redistributable raw third-party data by default.

## Steps
1. Download files from the official source URLs below.
2. Place the files into `raw_data/observational/cluster_rar` preserving filenames.
3. Run the manifest/checksum tool and compare output hashes for integrity.

## Official source URLs
- https://doi.org/10.3847/1538-4357/ab8e3d
- http://cdsarc.cds.unistra.fr

## Accessed
- Accessed: 2026-02-25

## Integrity verification
Run:
```bash
python3 tools/osf_packaging/build_dataset_manifests.py
```
