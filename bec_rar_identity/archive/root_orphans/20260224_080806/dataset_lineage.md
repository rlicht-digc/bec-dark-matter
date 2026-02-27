# Dataset Lineage Audit

## Source Table (Point-level)

| source | n_galaxies | n_points | ppg_min | ppg_med | ppg_max | log_gbar_min | log_gbar_max | n_field | n_dense | n_missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Catinella2005 | 322 | 2236 | 5 | 5.0 | 20 | -12.786 | -9.336 | 2216 | 20 | 0 |
| GHASP | 69 | 3875 | 3 | 37.0 | 274 | -11.835 | -8.984 | 3799 | 76 | 0 |
| LITTLETHINGS | 11 | 11 | 1 | 1.0 | 1 | -12.354 | -10.191 | 11 | 0 | 0 |
| LVHIS | 28 | 35 | 1 | 1.0 | 2 | -12.632 | -11.608 | 28 | 7 | 0 |
| MaNGA | 1424 | 3082 | 1 | 2.0 | 3 | -13.126 | -8.472 | 2156 | 926 | 0 |
| Noordermeer2005 | 47 | 47 | 1 | 1.0 | 1 | -18.250 | -9.389 | 42 | 5 | 0 |
| PHANGS | 41 | 1212 | 1 | 27.0 | 77 | -10.603 | -7.991 | 540 | 672 | 0 |
| SPARC | 131 | 2784 | 5 | 15.0 | 115 | -13.654 | -7.803 | 2266 | 518 | 0 |
| SS20_A | 2 | 2 | 1 | 1.0 | 1 | -11.248 | -10.930 | 2 | 0 | 0 |
| SS20_LT | 11 | 11 | 1 | 1.0 | 1 | -11.970 | -10.768 | 11 | 0 | 0 |
| SS20_R | 3 | 3 | 1 | 1.0 | 1 | -10.950 | -10.227 | 2 | 1 | 0 |
| SS20_S | 22 | 22 | 1 | 1.0 | 1 | -11.351 | -8.784 | 18 | 4 | 0 |
| SS20_TH | 2 | 2 | 1 | 1.0 | 1 | -10.717 | -10.150 | 2 | 0 | 0 |
| Swaters2025 | 85 | 85 | 1 | 1.0 | 1 | -11.341 | -8.231 | 82 | 3 | 0 |
| Verheijen2001 | 12 | 82 | 3 | 6.5 | 11 | -12.589 | -9.483 | 0 | 82 | 0 |
| VirgoRC | 1 | 18 | 18 | 18.0 | 18 | -10.518 | -10.069 | 0 | 18 | 0 |
| Vogt2004 | 327 | 327 | 1 | 1.0 | 1 | -11.885 | -9.687 | 31 | 296 | 0 |
| WALLABY | 165 | 1220 | 4 | 6.0 | 46 | -12.496 | -11.184 | 1016 | 204 | 0 |
| WALLABY_DR2 | 39 | 194 | 1 | 4.0 | 25 | -12.240 | -10.047 | 0 | 194 | 0 |
| Yu2020 | 132 | 132 | 1 | 1.0 | 1 | -13.352 | -10.939 | 93 | 39 | 0 |
| deBlok2002 | 18 | 365 | 4 | 15.0 | 59 | -14.393 | -10.674 | 365 | 0 | 0 |

## Source Table (Galaxy-level)

| source | n_galaxies | sigma_res_median | sigma_res_IQR | logMh_min | logMh_max |
|---|---:|---:|---:|---:|---:|
| Catinella2005 | 322 | 0.0987 | 0.0581 | 11.000 | 13.500 |
| GHASP | 69 | 0.1283 | 0.0739 | 11.000 | 14.900 |
| LITTLETHINGS | 11 | 0.0000 | 0.0000 | 11.000 | 12.447 |
| LVHIS | 28 | 0.0000 | 0.0010 | 11.000 | 13.246 |
| MaNGA | 1424 | 0.0346 | 0.0497 | 4.895 | 15.001 |
| Noordermeer2005 | 47 | 0.0000 | 0.0000 | 11.000 | 13.294 |
| PHANGS | 41 | 0.1508 | 0.1364 | 11.000 | 14.867 |
| SPARC | 131 | 0.0724 | 0.0841 | 0.000 | 13.200 |
| SS20_A | 2 | 0.0000 | 0.0000 | 11.000 | 11.000 |
| SS20_LT | 11 | 0.0000 | 0.0000 | 11.000 | 12.348 |
| SS20_R | 3 | 0.0000 | 0.0000 | 11.000 | 14.867 |
| SS20_S | 22 | 0.0000 | 0.0000 | 11.000 | 12.893 |
| SS20_TH | 2 | 0.0000 | 0.0000 | 11.000 | 12.340 |
| Swaters2025 | 85 | 0.0000 | 0.0000 | 11.000 | 14.900 |
| Verheijen2001 | 12 | 0.0609 | 0.1661 | 12.800 | 12.800 |
| VirgoRC | 1 | 0.1068 | 0.0000 | 14.200 | 14.200 |
| Vogt2004 | 327 | 0.0000 | 0.0000 | 11.000 | 15.000 |
| WALLABY | 165 | 0.1401 | 0.1148 | 11.000 | 15.000 |
| WALLABY_DR2 | 39 | 0.0481 | 0.0496 | 12.500 | 15.000 |
| Yu2020 | 132 | 0.0000 | 0.0000 | 11.000 | 14.900 |
| deBlok2002 | 18 | 0.1736 | 0.1276 | 11.000 | 11.000 |

## TNG Dataset Lineage

| tag | path | n_galaxies | n_points | ppg_median | status |
|---|---|---:|---:|---:|---|
| verified_clean_3000x50 | /Users/russelllicht/bec-dark-matter/datasets/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/rar_points_CLEAN.parquet | 3000 | 150000 | 50.0 | present |
| verified_clean_48133x50 | /Users/russelllicht/bec-dark-matter/datasets/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/rar_points.parquet | 48060 | 2402467 | 50.0 | present |
| contaminated_20899x15 | /Users/russelllicht/bec-dark-matter/datasets/quarantine_contaminated/rar_points_20899x15_CONTAMINATED.parquet | 20899 | 313485 | 15.0 | present |

## Contamination Note

CONTAMINATION NOTE:
- A mixed TNG dataset was initially used for SPARC-vs-TNG scatter comparisons.
- That mixed set reported TNG scatter substantially higher than SPARC (historical ratio near 4.13 in archived fairness sweeps).
- After verification/cleanup, clean TNG datasets reverse the direction (mass-matched median sigma ratio is near 0.5 in current clean-base checks).
- All current analyses use verified clean datasets only (3000x50 and/or 48133x50).
- Affected earlier outputs include TNG-vs-SPARC composition/fairness sweeps based on 20899x15 mixed extraction outputs.
- Historical contaminated scatter-ratio proxy: 4.128
- Current clean mass-matched scatter-ratio proxy: 0.517
- Contaminated file: /Users/russelllicht/bec-dark-matter/datasets/quarantine_contaminated/rar_points_20899x15_CONTAMINATED.parquet
- Diagnosed issue: mixed extraction passes produced 20899 galaxies x 15 points (non-uniform provenance); quarantined in datasets/quarantine_contaminated
