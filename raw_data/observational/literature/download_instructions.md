# Download Instructions

This folder is treated as non-redistributable raw third-party data by default.

## Steps
1. Download files from the official source URLs below.
2. Place the files into `raw_data/observational/literature` preserving filenames.
3. Run the manifest/checksum tool and compare output hashes for integrity.

## Official source URLs
- Multiple-source bundle (no single canonical URL): see per-file provenance in `manifest.csv` and `TERMS_OF_USE.md`.

## Integrity verification
Run:
```bash
python3 tools/osf_packaging/build_dataset_manifests.py
```
