# Download Instructions

This folder is treated as non-redistributable raw third-party data by default.

## Steps
1. Download files from the official source URLs below.
2. Place the files into `raw_data/observational/by_system` preserving filenames.
3. Run the manifest/checksum tool and compare output hashes for integrity.

## Official source URLs
- http://www.astro.yale.edu/viva/protected/dataindex.html
- https://www.ioa.s.u-tokyo.ac.jp/~sofue/RC99/rc99.htm

## Accessed
- Accessed: 2026-02-25

## Integrity verification
Run:
```bash
python3 tools/osf_packaging/build_dataset_manifests.py
```
