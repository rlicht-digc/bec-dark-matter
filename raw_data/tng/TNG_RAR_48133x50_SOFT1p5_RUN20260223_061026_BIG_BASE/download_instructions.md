# Download Instructions

This folder is treated as non-redistributable raw third-party data by default.

## Steps
1. Download files from the official source URLs below.
2. Place the files into `raw_data/tng/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE` preserving filenames.
3. Run the manifest/checksum tool and compare output hashes for integrity.

## Official source URLs
- https://www.tng-project.org/data/

## Accessed
- Accessed: 2026-02-25

## Integrity verification
Run:
```bash
python3 tools/osf_packaging/build_dataset_manifests.py
```
