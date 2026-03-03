# MBH↔xi bridge — censored Tobit addendum (referee-safe)
Date: 2026-03-02

## Inputs / outputs (frozen)
Input:
- outputs/mbh_xi_bridge_direct/20260302_134505/  (direct-only MBH set)

Outputs:
- outputs/mbh_xi_bridge_censored/20260302_164633/            (MBH vs xi)
- outputs/mbh_xi_bridge_censored/20260302_164643_mdyn/       (MBH vs Mdyn)
Each contains:
- summary_mbh_censored_fit.json
- fig_mbh_censored_fit.png
- report_mbh_censored_fit.md

Counts:
- n_total=10, n_detections=6, n_upper_limits=4

## Slopes
MBH vs xi:
- OLS (detections):  3.6340903339647044
- Tobit (censored):  3.389713879838549

MBH vs Mdyn:
- OLS (detections):  1.8170451669823515
- Tobit (censored):  1.69485074844081

## Sanity check (must be stated)
Because log10(xi) = 0.5*log10(Mdyn) + const, the slope relation is constrained:
slope(MBH–xi) ≈ 2 × slope(MBH–Mdyn)
Observed:
- OLS: 3.6340903339 = 2×1.8170451670
- Tobit: 3.3897138798 ≈ 2×1.6948507484

Therefore xi is not an independent predictor here; it is a deterministic transform of Mdyn in this pipeline.

## What we can claim
- Adding upper limits via correct censored likelihood slightly reduces the slope (expected direction).
- Result remains exploratory due to small-N; report as sensitivity/supporting analysis only.

## What we cannot claim
- No “confirmed organism bridge” (sample too small; selection + systematics remain).
- Greene+2016 late-type −0.6 dex offset is NOT applicable to this direct-only (non–M–σ) set; treat only as a separate sensitivity on M–σ-heavy catalogs.
