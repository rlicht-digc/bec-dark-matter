# TNG↔SPARC Strict Feature Reproduction Report

- Timestamp (UTC): 2026-03-01T21:55:52.805484+00:00
- Mode: `smoke`
- g†: log10 = -9.9208

## Matched Sample
- Matched pairs (raw): 24
- Matched pairs after K-sampling: 24
- SPARC matched: 403 points across 24 galaxies
- TNG matched: 403 points across 24 galaxies

## Feature Results
- SPARC inversion: -9.996152908662275
- TNG inversion: -9.704595653697158
- SPARC kurtosis peak: -10.415014504001775
- TNG kurtosis peak: -10.105877643694626

## Pass/Fail Rule
- Reproduces only if TNG inversion and kurtosis-peak locations are both inside SPARC bootstrap CI and both within ±0.10 dex of log g†.
- Verdict: **DOES NOT REPRODUCE**

## Secondary Diagnostics
- Split-half replication: included for SPARC and TNG.
- Shuffle null proximity to g†: included (galaxy-block residual shuffle).

## Note
- This run uses identical residual definition (non-parametric spline mean relation), identical binning parameters, and identical inversion/kurtosis codepaths for both datasets.
