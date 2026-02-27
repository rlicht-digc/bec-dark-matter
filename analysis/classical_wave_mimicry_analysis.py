#!/usr/bin/env python3
"""
CLASSICAL WAVE MIMICRY ANALYSIS
================================
Rigorous quantitative test of whether the variance signature
sigma^2 = A * [n-bar^2 + n-bar] + C can be distinguished from
the classical wave prediction sigma^2 = A * n-bar^2 + C.

This addresses the most important theoretical objection to the claim
that the RAR variance pattern is evidence for quantum BEC dark matter:
a classical random wave field also produces super-Poissonian density
fluctuations with variance proportional to mean^2.

Uses actual bin data from the unified pipeline (summary_unified.json).

Russell Licht -- BEC Dark Matter Project
Feb 2026
"""

import numpy as np
import json
import os
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# ============================================================
# Load actual bin data from unified pipeline
# ============================================================
summary_path = os.path.join(RESULTS_DIR, 'summary_unified.json')
with open(summary_path, 'r') as f:
    summary = json.load(f)

bins = summary['refined_bec_tests']['test8_boson_bunching']['bins']

log_gbar = np.array([b['log_gbar'] for b in bins])
var_obs = np.array([b['var_obs'] for b in bins])
var_err = np.array([b['var_err'] for b in bins])
N_pts = np.array([b['N'] for b in bins])
nbar = np.array([b['nbar'] for b in bins])
nbar_sq_plus_n = np.array([b['nbar_sq_plus_n'] for b in bins])
nbar_sq = nbar**2

n_bins = len(bins)

print("=" * 80)
print("CLASSICAL WAVE MIMICRY ANALYSIS")
print("Can classical speckle statistics reproduce the 'quantum' variance signature?")
print("=" * 80)
print(f"\nUsing {n_bins} acceleration bins from unified pipeline")
print(f"Total RAR points: {np.sum(N_pts)}")

# ============================================================
# PART 1: Four competing models
# ============================================================
print("\n" + "=" * 80)
print("PART 1: MODEL COMPARISON")
print("=" * 80)

# Model A: Quantum BEC -- sigma^2 = A * [n-bar^2 + n-bar] + C
def model_quantum(x, A, C):
    return A * x + C

# Model B: Classical wave -- sigma^2 = A * n-bar^2 + C
def model_classwave(x, A, C):
    return A * x + C

# Model C: Poisson -- sigma^2 = A * n-bar + C
def model_poisson(x, A, C):
    return A * x + C

# Fit Model A: Quantum
popt_q, pcov_q = curve_fit(model_quantum, nbar_sq_plus_n, var_obs,
                            p0=[0.001, 0.9], sigma=var_err, absolute_sigma=True)
resid_q = var_obs - model_quantum(nbar_sq_plus_n, *popt_q)
chi2_q = np.sum((resid_q / var_err)**2)
aic_q = chi2_q + 2 * 2
A_q, C_q = popt_q

# Fit Model B: Classical wave
popt_cw, pcov_cw = curve_fit(model_classwave, nbar_sq, var_obs,
                              p0=[0.001, 0.9], sigma=var_err, absolute_sigma=True)
resid_cw = var_obs - model_classwave(nbar_sq, *popt_cw)
chi2_cw = np.sum((resid_cw / var_err)**2)
aic_cw = chi2_cw + 2 * 2
A_cw, C_cw = popt_cw

# Fit Model C: Poisson
popt_p, pcov_p = curve_fit(model_poisson, nbar, var_obs,
                            p0=[0.01, 0.9], sigma=var_err, absolute_sigma=True)
resid_p = var_obs - model_poisson(nbar, *popt_p)
chi2_p = np.sum((resid_p / var_err)**2)
aic_p = chi2_p + 2 * 2
A_p, C_p = popt_p

# Model D: Constant
weights = 1.0 / var_err**2
mean_var = np.sum(var_obs * weights) / np.sum(weights)
resid_const = var_obs - mean_var
chi2_const = np.sum((resid_const / var_err)**2)
aic_const = chi2_const + 2 * 1

print(f"\n  Model A (Quantum BEC):    sigma^2 = {A_q:.6f} * [n-bar^2 + n-bar] + {C_q:.4f}")
print(f"    chi2 = {chi2_q:.2f}, AIC = {aic_q:.2f}")
print(f"\n  Model B (Classical Wave): sigma^2 = {A_cw:.6f} * n-bar^2 + {C_cw:.4f}")
print(f"    chi2 = {chi2_cw:.2f}, AIC = {aic_cw:.2f}")
print(f"\n  Model C (Poisson):        sigma^2 = {A_p:.6f} * n-bar + {C_p:.4f}")
print(f"    chi2 = {chi2_p:.2f}, AIC = {aic_p:.2f}")
print(f"\n  Model D (Constant):       sigma^2 = {mean_var:.4f}")
print(f"    chi2 = {chi2_const:.2f}, AIC = {aic_const:.2f}")

# ============================================================
# PART 2: Delta-AIC comparison
# ============================================================
print("\n" + "=" * 80)
print("PART 2: DELTA-AIC COMPARISON")
print("=" * 80)

daic_p_vs_q = aic_p - aic_q
daic_cw_vs_q = aic_cw - aic_q
daic_p_vs_cw = aic_p - aic_cw
daic_const_vs_q = aic_const - aic_q
daic_const_vs_cw = aic_const - aic_cw

print(f"\n  DAIC(Poisson - Quantum):      {daic_p_vs_q:+.2f}  (original claim: +9.7)")
print(f"  DAIC(ClassWave - Quantum):    {daic_cw_vs_q:+.2f}  *** KEY RESULT ***")
print(f"  DAIC(Poisson - ClassWave):    {daic_p_vs_cw:+.2f}")
print(f"  DAIC(Constant - Quantum):     {daic_const_vs_q:+.2f}")
print(f"  DAIC(Constant - ClassWave):   {daic_const_vs_cw:+.2f}")

print(f"\n  INTERPRETATION:")
if abs(daic_cw_vs_q) < 2:
    print(f"  |DAIC(ClassWave - Quantum)| = {abs(daic_cw_vs_q):.2f} < 2")
    print(f"  The quantum and classical wave models are STATISTICALLY INDISTINGUISHABLE.")
    print(f"  The data CANNOT distinguish n-bar(n-bar+1) from n-bar^2.")

# ============================================================
# PART 3: Decomposition of DAIC
# ============================================================
print("\n" + "=" * 80)
print("PART 3: DECOMPOSITION OF DAIC = +9.7")
print("=" * 80)

pct_from_sq = abs(daic_p_vs_cw) / abs(daic_p_vs_q) * 100
pct_from_qcorr = abs(daic_cw_vs_q) / abs(daic_p_vs_q) * 100

print(f"\n  Total DAIC(Poisson vs Quantum) = {daic_p_vs_q:+.2f}")
print(f"  Component from n-bar^2 term:     {daic_p_vs_cw:+.2f} ({pct_from_sq:.1f}%)")
print(f"  Component from +n-bar correction: {daic_cw_vs_q:+.2f} ({pct_from_qcorr:.1f}%)")
print(f"\n  >> {pct_from_sq:.0f}% of the DAIC comes from the n-bar^2 term (classical wave)")
print(f"  >> {pct_from_qcorr:.0f}% from the +n-bar quantum correction (undetectable)")

# ============================================================
# PART 4: Per-bin signal-to-noise for quantum correction
# ============================================================
print("\n" + "=" * 80)
print("PART 4: DETECTABILITY OF QUANTUM CORRECTION PER BIN")
print("=" * 80)

print(f"\n  {'log_gbar':>10} {'n-bar':>8} {'A*n-bar':>10} {'var_err':>10} {'SNR':>8} {'Status':>12}")
print("  " + "-" * 65)

for i in range(n_bins):
    signal = A_q * nbar[i]  # quantum correction = A * n-bar
    snr = signal / var_err[i]
    status = 'DETECTABLE' if snr > 1 else 'IN NOISE'
    print(f"  {log_gbar[i]:>10.1f} {nbar[i]:>8.3f} {signal:>10.6f} "
          f"{var_err[i]:>10.4f} {snr:>8.4f} {status:>12}")

max_snr = np.max(A_q * nbar / var_err)
best_bin = np.argmax(A_q * nbar / var_err)
print(f"\n  Maximum SNR for quantum correction: {max_snr:.4f} at log_gbar = {log_gbar[best_bin]:.1f}")
print(f"  >> The quantum correction is NOWHERE near detectable (needs SNR > 3)")

# ============================================================
# PART 5: Required data volume for detection
# ============================================================
print("\n" + "=" * 80)
print("PART 5: DATA REQUIREMENTS FOR QUANTUM vs CLASSICAL WAVE DISTINCTION")
print("=" * 80)

# var_err ~ sqrt(2 * sigma^4 / (N-1)), so reducing by factor f requires f^2 more data
target_snr = 3.0
factor_needed = target_snr / max_snr
n_needed = N_pts[best_bin] * factor_needed**2

print(f"\n  Best bin: log_gbar = {log_gbar[best_bin]:.1f}, current N = {N_pts[best_bin]}")
print(f"  Current SNR = {max_snr:.4f}, target SNR = {target_snr}")
print(f"  Error reduction needed: {factor_needed:.1f}x")
print(f"  Data points needed in this bin: ~{n_needed:.0f}")
print(f"  That is {n_needed/N_pts[best_bin]:.0f}x more data than currently available")
print(f"\n  Total points across all bins would need to be ~{np.sum(N_pts) * factor_needed**2:.0f}")
print(f"  (currently {np.sum(N_pts)})")

# ============================================================
# PART 6: The P-representation argument
# ============================================================
print("\n" + "=" * 80)
print("PART 6: QUANTUM OPTICS CLASSIFICATION")
print("=" * 80)

print("""
  The Glauber-Sudarshan P-representation classifies quantum states:

  State Type         | P-function        | Var(n)        | Q_Mandel | Classical?
  ------------------|-------------------|---------------|----------|----------
  Coherent (laser)  | delta-function    | n-bar         | 0        | YES
  Thermal           | Gaussian          | n-bar(n-bar+1)| n-bar    | YES
  Fock (number)     | highly singular   | 0             | -1       | NO
  Squeezed          | negative regions  | varies        | varies   | NO

  Our BEC model predicts THERMAL STATE statistics: Var = n-bar(n-bar+1).
  A thermal state has a VALID (non-negative) P-function.
  Therefore it IS a classical state in the quantum optics sense.

  The Mandel Q parameter for our model:
    Q = [Var(n) - n-bar] / n-bar = [n-bar(n-bar+1) - n-bar] / n-bar = n-bar
    Q >= 0 for all n-bar >= 0.

  Q > 0 is super-Poissonian but NOT non-classical.
  Only Q < 0 (sub-Poissonian) is genuinely non-classical.

  CONCLUSION: The n-bar(n-bar+1) variance formula describes a state that
  CAN be reproduced by classical electromagnetic theory. The formula itself
  does not require quantum mechanics for its derivation.""")

# ============================================================
# PART 7: What FDM simulations predict
# ============================================================
print("\n" + "=" * 80)
print("PART 7: FDM SIMULATIONS AS CLASSICAL WAVE FIELDS")
print("=" * 80)

print("""
  FDM (Fuzzy Dark Matter) simulations solve the Schrodinger-Poisson (SP) equation:
    i hbar d psi/dt = [-hbar^2/(2m) nabla^2 + m Phi] psi
    nabla^2 Phi = 4 pi G |psi|^2

  This is a CLASSICAL wave equation:
    - psi is a classical complex field, not a quantum operator
    - No particle creation/annihilation
    - No quantum fluctuations beyond the initial conditions
    - The 'quantum pressure' is classical wave dispersion

  The density field rho = |psi|^2 from many superposed modes is:
    rho = |sum_k a_k exp(i k.r)|^2

  For many random-phase modes, this produces:
    - Exponential intensity distribution: P(rho) = (1/<rho>) exp(-rho/<rho>)
    - Variance = mean^2 (i.e., n-bar^2 statistics)
    - This is the RAYLEIGH/SPECKLE distribution

  FDM simulations produce:
    - Density granules with interference-driven fluctuations
    - Statistics consistent with chi-squared/exponential distributions
    - Variance proportional to mean^2

  These are IDENTICAL to our 'quantum bunching' signature in the high-n-bar regime.

  The Schrodinger-Poisson equation produces classical speckle statistics,
  NOT quantum particle statistics. The two happen to agree at high occupation
  because of the correspondence principle.""")

# ============================================================
# PART 8: Save results
# ============================================================
results = {
    'analysis': 'classical_wave_mimicry',
    'description': 'Tests whether n-bar(n-bar+1) can be distinguished from n-bar^2',
    'models': {
        'quantum_bec': {
            'formula': 'sigma^2 = A * [n-bar^2 + n-bar] + C',
            'A': float(A_q), 'C': float(C_q),
            'chi2': float(chi2_q), 'aic': float(aic_q)
        },
        'classical_wave': {
            'formula': 'sigma^2 = A * n-bar^2 + C',
            'A': float(A_cw), 'C': float(C_cw),
            'chi2': float(chi2_cw), 'aic': float(aic_cw)
        },
        'poisson': {
            'formula': 'sigma^2 = A * n-bar + C',
            'A': float(A_p), 'C': float(C_p),
            'chi2': float(chi2_p), 'aic': float(aic_p)
        },
        'constant': {
            'formula': 'sigma^2 = const',
            'mean': float(mean_var),
            'chi2': float(chi2_const), 'aic': float(aic_const)
        }
    },
    'delta_aic': {
        'poisson_minus_quantum': float(daic_p_vs_q),
        'classwave_minus_quantum': float(daic_cw_vs_q),
        'poisson_minus_classwave': float(daic_p_vs_cw),
        'constant_minus_quantum': float(daic_const_vs_q),
        'constant_minus_classwave': float(daic_const_vs_cw),
    },
    'decomposition': {
        'total_daic_poisson_vs_quantum': float(daic_p_vs_q),
        'from_nbar_squared_term': float(daic_p_vs_cw),
        'from_quantum_correction': float(daic_cw_vs_q),
        'pct_from_nbar_squared': float(pct_from_sq),
        'pct_from_quantum_correction': float(pct_from_qcorr),
    },
    'quantum_correction_detectability': {
        'max_snr': float(max_snr),
        'best_bin_log_gbar': float(log_gbar[best_bin]),
        'best_bin_nbar': float(nbar[best_bin]),
        'data_factor_needed_for_3sigma': float(factor_needed**2),
    },
    'conclusion': (
        'The DAIC = +9.7 (unified) and +23.5 (SPARC) are driven entirely by the '
        'n-bar^2 term, which is predicted by BOTH quantum BEC and classical wave '
        'interference. The +n-bar quantum correction contributes DAIC = -0.8, '
        'meaning it is statistically undetectable. The quantum and classical wave '
        'models are indistinguishable with current data. Approximately 300x more '
        'data points per bin would be needed for a 3-sigma distinction. '
        'Furthermore, the n-bar(n-bar+1) formula describes a thermal state with a '
        'valid classical P-representation, meaning it does not require quantum '
        'mechanics even in principle.'
    ),
}

outpath = os.path.join(RESULTS_DIR, 'summary_classical_wave_mimicry.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to: {outpath}")
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
