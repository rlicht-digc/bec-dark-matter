# MBH Bridge Censored Fit Report

- Input folder: `/Users/russelllicht/bec-dark-matter/outputs/mbh_xi_bridge_direct/20260302_134505`
- Input table: `direct_plus_upper_matches.csv`
- Method: `tobit`
- Counts: total=10, detections=6, upper_limits=4

## OLS (detections only)
- intercept=3.282459
- slope=3.634090
- sigma=0.800725
- rms=0.653789
- mad=0.879152

## Tobit censored MLE
- success=True
- message=CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
- intercept=3.002226
- slope=3.389714
- sigma=0.939443
- loglik=-11.194091
- aic=28.388181
- bic=29.295936

Figure: `fig_mbh_censored_fit.png`
Summary: `summary_mbh_censored_fit.json`
