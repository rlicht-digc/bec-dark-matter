# Test-7 Intrinsic Scatter Calibration (Paper-Ready Methods Box)

We calibrate an intrinsic scatter term in the normalized residual space used by Test-7 (z), inflating σ_tot² = σ_meas² + σ_int² until χ²_red ≈ 1.
This calibration is applied to the Test-7 model-comparison residuals only; model formulas are unchanged.
The reported σ_int values below are in z-space (dimensionless), not dex or km/s.
For Test-7, the effective measurement term is defined in the normalized fit space (`sigma_meas = 1/weights` at the bin level, with equivalent expanded-point representation for calibration).

| Model | chi2_red_uncal | sigma_int_z | chi2_red_cal |
|---|---:|---:|---:|
| BEC | 15.0011924252 | 3.7418166170 | 1.0000000553 |
| Linear | 13.8081360692 | 3.5788458258 | 0.9999999004 |
| Constant | 12.9423411868 | 3.4557692615 | 0.9999999999 |

Notes:
- Calibration solver: `analysis/utils/chi2_calibration.py` (`solve_sigma_int_for_chi2_1`).
- Values sourced from `analysis/results/summary_unified.json` under `refined_bec_tests.bec_transition_function`.
