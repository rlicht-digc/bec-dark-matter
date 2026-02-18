#!/usr/bin/env python3
"""
Radial Variance Profile Test: σ²(n̄) Continuous Shape

The unified BEC/shadow framework predicts that the variance of RAR
residuals as a function of occupation number n̄ should follow an
INVERTED-U shape:

  - At n̄ >> 1 (deep inner, saturated projection):
      σ² ~ constant (the condensate is fully formed, stable)
      Looks CLASSICAL because variance is dominated by the n̄² floor

  - At n̄ ~ 1-10 (transition zone, R ~ ξ):
      σ² PEAKS — the condensation threshold where counting statistics
      are maximally visible. n̄(n̄+1) vs n̄ distinction is largest here.

  - At n̄ << 1 (far outer, evaporating projection):
      σ² → C (baseline) — too few "intersection modes" to measure

CDM/classical prediction: σ² ~ A*n̄ + C (monotonic, no peak)
BEC/shadow prediction: σ² = A*n̄(n̄+1) + C, which has an inverted-U
    when plotted vs log(n̄) because the n̄² term dominates at high n̄
    but the peak *relative to the classical model* is at n̄ ~ 1.

The KEY DIAGNOSTIC is the RATIO σ²_obs / σ²_classical:
  - BEC predicts ratio = (n̄+1), which peaks at the transition
  - CDM predicts ratio ≈ 1 everywhere

This test bins by n̄ (not log_gbar), computes the variance in each
bin, and fits the shape. It also does this per-galaxy using R/ξ
as the radial coordinate.

Author: Built on the SPARC RAR BEC pipeline (Russell Licht, Feb 2026)
"""

import os
import sys
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr, chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
G_SI = 6.674e-11
Msun_kg = 1.989e30
kpc_m = 3.086e19
gdagger = 1.2e-10  # m/s²

# McGaugh RAR function
def rar_function(log_gbar, a0=1.2e-10):
    """Return log10(gobs) from McGaugh+2016 RAR."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)

def nbar_func(gbar_SI):
    """Bose-Einstein occupation number."""
    x = np.sqrt(gbar_SI / gdagger)
    return 1.0 / (np.exp(x) - 1.0 + 1e-30)

def xi_kpc(logMs):
    """Healing length in kpc from stellar mass."""
    Ms_SI = 10.0**logMs * Msun_kg
    return np.sqrt(G_SI * Ms_SI / gdagger) / kpc_m


# ================================================================
# STEP 1: Load SPARC data
# ================================================================
print("=" * 76)
print("RADIAL VARIANCE PROFILE TEST: σ²(n̄) CONTINUOUS SHAPE")
print("Unified BEC/Shadow Prediction: Inverted-U in σ² vs n̄")
print("=" * 76)

print("\n[1] Loading SPARC rotation curves...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

# Parse rotation curves
galaxies = {}
with open(table2_path, 'r') as f:
    for line in f:
        if len(line.strip()) < 50:
            continue
        try:
            name = line[0:11].strip()
            if not name:
                continue
            dist = float(line[12:18].strip())
            rad = float(line[19:25].strip())
            vobs = float(line[26:32].strip())
            evobs = float(line[33:38].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in galaxies:
            galaxies[name] = {'R': [], 'Vobs': [], 'eVobs': [],
                              'Vgas': [], 'Vdisk': [], 'Vbul': [],
                              'dist': dist}
        galaxies[name]['R'].append(rad)
        galaxies[name]['Vobs'].append(vobs)
        galaxies[name]['eVobs'].append(evobs)
        galaxies[name]['Vgas'].append(vgas)
        galaxies[name]['Vdisk'].append(vdisk)
        galaxies[name]['Vbul'].append(vbul)

for name in galaxies:
    for key in ['R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul']:
        galaxies[name][key] = np.array(galaxies[name][key])

# Parse MRT for stellar masses
with open(mrt_path, 'r') as f:
    mrt_lines = f.readlines()
data_start = 0
for i, line in enumerate(mrt_lines):
    if line.startswith('---') and i > 50:
        data_start = i + 1
        break

sparc_props = {}
for line in mrt_lines[data_start:]:
    if not line.strip() or line.startswith('#'):
        continue
    try:
        name = line[0:11].strip()
        if not name:
            continue
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        Inc = float(parts[4])
        L36 = float(parts[6])
        Q = int(parts[16])
        logMs = np.log10(max(0.5 * L36 * 1e9, 1e6))
        sparc_props[name] = {
            'Inc': Inc, 'Q': Q, 'logMs': logMs,
        }
    except (ValueError, IndexError):
        continue

print(f"  {len(galaxies)} rotation curves, {len(sparc_props)} with stellar masses")


# ================================================================
# STEP 2: Compute RAR residuals with per-galaxy metadata
# ================================================================
print("\n[2] Computing RAR residuals with radial context...")

all_points = []
galaxy_names_used = set()

for name, gdata in galaxies.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

    R = gdata['R']
    Vobs = gdata['Vobs']
    eVobs = gdata['eVobs']
    Vdisk = gdata['Vdisk']
    Vgas = gdata['Vgas']
    Vbul = gdata['Vbul']

    # Baryonic velocity squared
    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)

    # Proper SI accelerations
    gbar_SI = np.where(R > 0,
                        (np.sqrt(np.abs(Vbar_sq)) * 1e3)**2 / (R * kpc_m),
                        1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    # Fractional velocity error cut
    valid = ((gbar_SI > 1e-15) & (gobs_SI > 1e-15) &
             (R > 0) & (Vobs > 0) & (eVobs / np.maximum(Vobs, 1) < 0.3))
    if np.sum(valid) < 3:
        continue

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    log_gobs_rar = rar_function(log_gbar)
    log_res = log_gobs - log_gobs_rar

    logMs = prop['logMs']
    xi = xi_kpc(logMs)
    galaxy_names_used.add(name)

    for i in range(len(log_gbar)):
        nbar = nbar_func(10.0**log_gbar[i])
        X = R[valid][i] / xi if xi > 0 else 999.0
        all_points.append({
            'galaxy': name,
            'log_gbar': float(log_gbar[i]),
            'log_gobs': float(log_gobs[i]),
            'log_res': float(log_res[i]),
            'R_kpc': float(R[valid][i]),
            'logMs': logMs,
            'xi_kpc': xi,
            'X': float(X),          # R/ξ dimensionless
            'nbar': float(nbar),     # occupation number
            'log_nbar': float(np.log10(max(nbar, 1e-10))),
        })

n_gal = len(galaxy_names_used)
print(f"  {len(all_points)} RAR points from {n_gal} galaxies")


# ================================================================
# STEP 3: Z-score residuals globally
# ================================================================
print("\n[3] Z-scoring residuals...")

all_res = np.array([p['log_res'] for p in all_points])
mu = np.mean(all_res)
std = np.std(all_res)
for p in all_points:
    p['z_res'] = (p['log_res'] - mu) / std

print(f"  Global: μ = {mu:.4f}, σ = {std:.4f}")


# ================================================================
# STEP 4: CONTINUOUS σ²(n̄) PROFILE
# ================================================================
print("\n" + "=" * 76)
print("TEST A: CONTINUOUS σ²(n̄) PROFILE")
print("=" * 76)
print("  Bin by occupation number n̄, compute variance in each bin.")
print("  BEC predicts inverted-U; CDM predicts monotonic or flat.")

# Bin by log(n̄) — this is the natural coordinate
# n̄ ranges from ~0.001 (high gbar) to ~100 (very low gbar)
lognbar_edges = np.arange(-3.0, 2.5, 0.35)
lognbar_centers = (lognbar_edges[:-1] + lognbar_edges[1:]) / 2.0

bin_data = []
print(f"\n  {'log(n̄)':>8} {'n̄_med':>8} {'N':>6} {'σ²_obs':>10} "
      f"{'σ²_err':>10} {'n̄(n̄+1)':>10} {'ratio':>8}")
print(f"  {'-' * 68}")

for j in range(len(lognbar_centers)):
    lo, hi = lognbar_edges[j], lognbar_edges[j + 1]
    z_vals = np.array([p['z_res'] for p in all_points
                       if lo <= p['log_nbar'] < hi])
    nbar_vals = np.array([p['nbar'] for p in all_points
                          if lo <= p['log_nbar'] < hi])

    if len(z_vals) >= 25:
        var_obs = float(np.var(z_vals))
        var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
        nbar_med = float(np.median(nbar_vals))
        nbar_sq_plus_n = nbar_med**2 + nbar_med
        # Ratio: observed variance / what classical predicts
        # Classical: σ² = A*n̄ + C → at this n̄, the ratio σ²/n̄
        ratio = var_obs / max(nbar_med, 1e-10)

        entry = {
            'log_nbar_center': float(lognbar_centers[j]),
            'nbar_median': nbar_med,
            'N': len(z_vals),
            'var_obs': var_obs,
            'var_err': var_err,
            'nbar_sq_plus_n': nbar_sq_plus_n,
            'ratio_var_over_nbar': ratio,
        }
        bin_data.append(entry)

        print(f"  {lognbar_centers[j]:>8.2f} {nbar_med:>8.3f} {len(z_vals):>6} "
              f"{var_obs:>10.4f} {var_err:>10.4f} {nbar_sq_plus_n:>10.4f} "
              f"{ratio:>8.4f}")

bin_var = np.array([b['var_obs'] for b in bin_data])
bin_var_err = np.array([b['var_err'] for b in bin_data])
bin_nbar = np.array([b['nbar_median'] for b in bin_data])
bin_nbar_sq = np.array([b['nbar_sq_plus_n'] for b in bin_data])
bin_lognbar = np.array([b['log_nbar_center'] for b in bin_data])
bin_N = np.array([b['N'] for b in bin_data])

n_bins = len(bin_data)
print(f"\n  {n_bins} usable bins (N >= 25)")


# ================================================================
# STEP 5: FIT THREE MODELS
# ================================================================
print("\n" + "=" * 76)
print("TEST A: MODEL FITTING")
print("=" * 76)

results = {}

if n_bins >= 5:
    try:
        # Model Q: BEC quantum — σ² = A*(n̄² + n̄) + C
        def model_quantum(nbar_sq_n, A, C):
            return A * nbar_sq_n + C
        popt_q, pcov_q = curve_fit(model_quantum, bin_nbar_sq, bin_var,
                                    p0=[0.01, 0.5], sigma=bin_var_err,
                                    absolute_sigma=True, maxfev=10000)
        pred_q = model_quantum(bin_nbar_sq, *popt_q)
        chi2_q = np.sum(((bin_var - pred_q) / bin_var_err)**2)
        aic_q = chi2_q + 2 * 2

        # Model C: Classical — σ² = A*n̄ + C
        def model_classical(nbar, A, C):
            return A * nbar + C
        popt_c, pcov_c = curve_fit(model_classical, bin_nbar, bin_var,
                                    p0=[0.01, 0.5], sigma=bin_var_err,
                                    absolute_sigma=True, maxfev=10000)
        pred_c = model_classical(bin_nbar, *popt_c)
        chi2_c = np.sum(((bin_var - pred_c) / bin_var_err)**2)
        aic_c = chi2_c + 2 * 2

        # Model K: Constant — σ² = K
        wt = 1.0 / np.maximum(bin_var_err, 1e-6)
        K = np.average(bin_var, weights=wt**2)
        chi2_k = np.sum(((bin_var - K) / bin_var_err)**2)
        aic_k = chi2_k + 2 * 1

        # Model P: Peak/inverted-U — σ² = A*n̄*exp(-n̄/n̄_peak) + C
        # This captures the unified prediction: rises with n̄, then
        # saturates and drops at high n̄ (deep interior = stable)
        def model_peak(nbar, A, nbar_peak, C):
            return A * nbar * np.exp(-nbar / max(nbar_peak, 0.01)) + C
        try:
            popt_p, pcov_p = curve_fit(model_peak, bin_nbar, bin_var,
                                        p0=[0.1, 3.0, 0.5],
                                        sigma=bin_var_err,
                                        absolute_sigma=True, maxfev=10000,
                                        bounds=([0, 0.01, 0], [100, 200, 5]))
            pred_p = model_peak(bin_nbar, *popt_p)
            chi2_p = np.sum(((bin_var - pred_p) / bin_var_err)**2)
            aic_p = chi2_p + 2 * 3  # 3 parameters
            has_peak = True
        except Exception:
            has_peak = False
            chi2_p = aic_p = np.inf

        daic_q_vs_c = aic_c - aic_q      # positive = quantum better
        daic_q_vs_k = aic_k - aic_q      # positive = quantum better than constant
        daic_p_vs_q = aic_q - aic_p      # positive = peak better than quantum
        daic_p_vs_c = aic_c - aic_p      # positive = peak better than classical

        print(f"\n  QUANTUM (BEC):  σ² = {popt_q[0]:.6f} × (n̄²+n̄) + {popt_q[1]:.4f}")
        print(f"                  χ²/dof = {chi2_q / max(n_bins - 2, 1):.3f}, AIC = {aic_q:.2f}")
        print(f"  CLASSICAL:      σ² = {popt_c[0]:.6f} × n̄ + {popt_c[1]:.4f}")
        print(f"                  χ²/dof = {chi2_c / max(n_bins - 2, 1):.3f}, AIC = {aic_c:.2f}")
        print(f"  CONSTANT:       σ² = {K:.4f}")
        print(f"                  χ²/dof = {chi2_k / max(n_bins - 1, 1):.3f}, AIC = {aic_k:.2f}")
        if has_peak:
            print(f"  PEAK (shadow):  σ² = {popt_p[0]:.6f} × n̄ × exp(-n̄/{popt_p[1]:.2f}) + {popt_p[2]:.4f}")
            print(f"                  χ²/dof = {chi2_p / max(n_bins - 3, 1):.3f}, AIC = {aic_p:.2f}")
            print(f"                  Peak at n̄ = {popt_p[1]:.2f} (log n̄ = {np.log10(popt_p[1]):.2f})")

        print(f"\n  ΔAIC (classical − quantum)  = {daic_q_vs_c:+.2f}")
        print(f"  ΔAIC (constant  − quantum)  = {daic_q_vs_k:+.2f}")
        if has_peak:
            print(f"  ΔAIC (quantum   − peak)     = {daic_p_vs_q:+.2f}")
            print(f"  ΔAIC (classical − peak)     = {daic_p_vs_c:+.2f}")

        # Determine best model
        models = [('quantum', aic_q), ('classical', aic_c), ('constant', aic_k)]
        if has_peak:
            models.append(('peak', aic_p))
        models.sort(key=lambda x: x[1])
        best_name, best_aic = models[0]
        second_name, second_aic = models[1]

        print(f"\n  >>> BEST MODEL: {best_name.upper()} (AIC = {best_aic:.2f})")
        print(f"  >>> Runner-up:  {second_name} (ΔAIC = {second_aic - best_aic:+.2f})")

        results['test_a'] = {
            'n_bins': n_bins,
            'n_points': int(np.sum(bin_N)),
            'quantum': {'A': float(popt_q[0]), 'C': float(popt_q[1]),
                        'chi2': float(chi2_q), 'aic': float(aic_q)},
            'classical': {'A': float(popt_c[0]), 'C': float(popt_c[1]),
                          'chi2': float(chi2_c), 'aic': float(aic_c)},
            'constant': {'K': float(K), 'chi2': float(chi2_k), 'aic': float(aic_k)},
            'daic_quantum_vs_classical': float(daic_q_vs_c),
            'daic_quantum_vs_constant': float(daic_q_vs_k),
            'best_model': best_name,
        }
        if has_peak:
            results['test_a']['peak'] = {
                'A': float(popt_p[0]), 'nbar_peak': float(popt_p[1]),
                'C': float(popt_p[2]),
                'chi2': float(chi2_p), 'aic': float(aic_p),
            }
            results['test_a']['daic_peak_vs_quantum'] = float(daic_p_vs_q)
            results['test_a']['daic_peak_vs_classical'] = float(daic_p_vs_c)

    except Exception as e:
        print(f"  ERROR in model fitting: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# STEP 6: CONTINUOUS X = R/ξ VARIANCE PROFILE (per-galaxy radial)
# ================================================================
print("\n" + "=" * 76)
print("TEST B: WITHIN-GALAXY σ²(X) PROFILE — X = R/ξ")
print("=" * 76)
print("  For each galaxy, compute Z-scored residual at each point.")
print("  Pool all points by X = R/ξ bin. Compute variance per X-bin.")
print("  BEC predicts: peak variance near X ~ 1 (transition zone).")

# Bin by log(X) = log(R/ξ)
logX_edges = np.arange(-1.5, 2.5, 0.3)
logX_centers = (logX_edges[:-1] + logX_edges[1:]) / 2.0

bin_X_data = []
print(f"\n  {'log(X)':>8} {'X_med':>8} {'N':>6} {'σ²_obs':>10} {'σ²_err':>10}")
print(f"  {'-' * 50}")

for j in range(len(logX_centers)):
    lo, hi = logX_edges[j], logX_edges[j + 1]
    z_vals = np.array([p['z_res'] for p in all_points
                       if lo <= np.log10(max(p['X'], 1e-5)) < hi])
    X_vals = np.array([p['X'] for p in all_points
                       if lo <= np.log10(max(p['X'], 1e-5)) < hi])

    if len(z_vals) >= 25:
        var_obs = float(np.var(z_vals))
        var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
        X_med = float(np.median(X_vals))

        entry = {
            'logX_center': float(logX_centers[j]),
            'X_median': X_med,
            'N': len(z_vals),
            'var_obs': var_obs,
            'var_err': var_err,
        }
        bin_X_data.append(entry)
        print(f"  {logX_centers[j]:>8.2f} {X_med:>8.2f} {len(z_vals):>6} "
              f"{var_obs:>10.4f} {var_err:>10.4f}")

if len(bin_X_data) >= 4:
    X_var = np.array([b['var_obs'] for b in bin_X_data])
    X_var_err = np.array([b['var_err'] for b in bin_X_data])
    X_logX = np.array([b['logX_center'] for b in bin_X_data])
    X_med = np.array([b['X_median'] for b in bin_X_data])

    # Find the peak
    peak_idx = np.argmax(X_var)
    print(f"\n  PEAK variance at log(X) = {X_logX[peak_idx]:.2f} "
          f"(X = {X_med[peak_idx]:.2f}), σ² = {X_var[peak_idx]:.4f}")

    # Test: is the peak near X ~ 1?
    peak_X = X_med[peak_idx]
    if 0.3 < peak_X < 5.0:
        print(f"  >>> Peak is near X ~ 1 (transition zone) — "
              f"CONSISTENT with BEC/shadow prediction!")
    elif peak_X < 0.3:
        print(f"  >>> Peak is at X < 0.3 (deep core) — "
              f"suggests quantum signature in soliton interior")
    else:
        print(f"  >>> Peak is at X > 5 (outer envelope) — "
              f"NOT where BEC predicts maximum variance")

    # Fit a peaked model: σ²(X) = A*X*exp(-X/X_peak) + C
    try:
        def peak_X_model(X, A, X_peak, C):
            return A * X * np.exp(-X / max(X_peak, 0.01)) + C
        popt_Xp, _ = curve_fit(peak_X_model, X_med, X_var,
                                p0=[0.5, 1.0, 0.5],
                                sigma=X_var_err, absolute_sigma=True,
                                bounds=([0, 0.01, 0], [50, 50, 5]),
                                maxfev=10000)
        pred_Xp = peak_X_model(X_med, *popt_Xp)
        chi2_Xp = np.sum(((X_var - pred_Xp) / X_var_err)**2)
        aic_Xp = chi2_Xp + 2 * 3

        # Compare to monotonic: σ²(X) = A/X + C (decaying with radius)
        def decay_X_model(X, A, C):
            return A / np.maximum(X, 0.01) + C
        popt_Xd, _ = curve_fit(decay_X_model, X_med, X_var,
                                p0=[0.5, 0.5],
                                sigma=X_var_err, absolute_sigma=True,
                                bounds=([0, 0], [50, 5]),
                                maxfev=10000)
        pred_Xd = decay_X_model(X_med, *popt_Xd)
        chi2_Xd = np.sum(((X_var - pred_Xd) / X_var_err)**2)
        aic_Xd = chi2_Xd + 2 * 2

        # Constant
        wt_X = 1.0 / np.maximum(X_var_err, 1e-6)
        K_X = np.average(X_var, weights=wt_X**2)
        chi2_Xk = np.sum(((X_var - K_X) / X_var_err)**2)
        aic_Xk = chi2_Xk + 2 * 1

        daic_peak_vs_decay = aic_Xd - aic_Xp
        daic_peak_vs_const = aic_Xk - aic_Xp

        print(f"\n  PEAKED model:   σ²(X) = {popt_Xp[0]:.4f} × X × exp(-X/{popt_Xp[1]:.2f}) + {popt_Xp[2]:.4f}")
        print(f"                  Peak at X = {popt_Xp[1]:.2f}, χ²/dof = {chi2_Xp / max(len(X_var) - 3, 1):.3f}")
        print(f"  DECAY model:    σ²(X) = {popt_Xd[0]:.4f} / X + {popt_Xd[1]:.4f}")
        print(f"                  χ²/dof = {chi2_Xd / max(len(X_var) - 2, 1):.3f}")
        print(f"  CONSTANT:       σ² = {K_X:.4f}")
        print(f"                  χ²/dof = {chi2_Xk / max(len(X_var) - 1, 1):.3f}")
        print(f"\n  ΔAIC (decay   − peaked) = {daic_peak_vs_decay:+.2f}")
        print(f"  ΔAIC (constant − peaked) = {daic_peak_vs_const:+.2f}")

        results['test_b'] = {
            'n_bins': len(bin_X_data),
            'peak_logX': float(X_logX[peak_idx]),
            'peak_X': float(peak_X),
            'peak_var': float(X_var[peak_idx]),
            'peaked_model': {'A': float(popt_Xp[0]),
                             'X_peak': float(popt_Xp[1]),
                             'C': float(popt_Xp[2]),
                             'chi2': float(chi2_Xp), 'aic': float(aic_Xp)},
            'decay_model': {'A': float(popt_Xd[0]), 'C': float(popt_Xd[1]),
                            'chi2': float(chi2_Xd), 'aic': float(aic_Xd)},
            'constant': {'K': float(K_X), 'chi2': float(chi2_Xk),
                         'aic': float(aic_Xk)},
            'daic_peak_vs_decay': float(daic_peak_vs_decay),
            'daic_peak_vs_constant': float(daic_peak_vs_const),
            'bins': [b for b in bin_X_data],
        }

    except Exception as e:
        print(f"  Model fitting error: {e}")


# ================================================================
# STEP 7: EXCESS VARIANCE RATIO — (n̄+1) diagnostic
# ================================================================
print("\n" + "=" * 76)
print("TEST C: EXCESS VARIANCE RATIO σ²_obs / σ²_classical vs n̄")
print("=" * 76)
print("  If BEC: σ² = A*(n̄²+n̄) + C = A*n̄*(n̄+1) + C")
print("  Then σ²/σ²_classical = (n̄²+n̄)/(n̄) = (n̄+1)")
print("  So the ratio should INCREASE with n̄, reaching ~(n̄+1).")
print("  Classical: ratio ≈ constant.")

# Use the fitted classical model to compute expected variance
if 'test_a' in results and n_bins >= 5:
    A_c_fit = results['test_a']['classical']['A']
    C_c_fit = results['test_a']['classical']['C']

    print(f"\n  Using fitted classical model: σ²_classical = {A_c_fit:.6f}*n̄ + {C_c_fit:.4f}")

    ratio_data = []
    print(f"\n  {'log(n̄)':>8} {'n̄':>8} {'σ²_obs':>10} {'σ²_class':>10} "
          f"{'ratio':>8} {'n̄+1':>8} {'excess':>8}")
    print(f"  {'-' * 72}")

    for b in bin_data:
        nbar = b['nbar_median']
        var_classical = A_c_fit * nbar + C_c_fit
        if var_classical > 0.01:
            ratio = b['var_obs'] / var_classical
            expected_ratio = (nbar + 1)
            # Normalize: the "excess" is how much above 1 the ratio is,
            # relative to how much (n̄+1) is above 1
            excess = (ratio - 1.0) / max(nbar, 0.01) if nbar > 0.01 else 0
            entry = {
                'log_nbar': b['log_nbar_center'],
                'nbar': nbar,
                'var_obs': b['var_obs'],
                'var_classical': var_classical,
                'ratio': ratio,
                'nbar_plus_1': expected_ratio,
                'excess': excess,
            }
            ratio_data.append(entry)
            print(f"  {b['log_nbar_center']:>8.2f} {nbar:>8.3f} "
                  f"{b['var_obs']:>10.4f} {var_classical:>10.4f} "
                  f"{ratio:>8.3f} {expected_ratio:>8.3f} {excess:>8.3f}")

    if len(ratio_data) >= 4:
        ratios = np.array([r['ratio'] for r in ratio_data])
        nbars = np.array([r['nbar'] for r in ratio_data])
        nbar_plus1 = np.array([r['nbar_plus_1'] for r in ratio_data])

        # Correlation: does ratio increase with n̄?
        rho_ratio, p_ratio = spearmanr(nbars, ratios)
        print(f"\n  Spearman ρ(n̄, ratio) = {rho_ratio:+.3f} (p = {p_ratio:.4f})")

        # Does ratio track (n̄+1)?
        r_corr, p_corr = pearsonr(nbar_plus1, ratios)
        print(f"  Pearson r(n̄+1, ratio) = {r_corr:+.3f} (p = {p_corr:.4f})")

        if rho_ratio > 0 and p_ratio < 0.05:
            print(f"  >>> RATIO INCREASES with n̄ — SUPER-POISSONIAN bunching!")
            print(f"      Consistent with BEC/shadow: σ² ∝ n̄(n̄+1)")
        elif rho_ratio > 0:
            print(f"  >>> Weak positive trend (p = {p_ratio:.3f}), suggestive")
        else:
            print(f"  >>> No positive trend — consistent with classical")

        results['test_c'] = {
            'spearman_rho': float(rho_ratio),
            'spearman_p': float(p_ratio),
            'pearson_r_nbar_plus1': float(r_corr),
            'pearson_p_nbar_plus1': float(p_corr),
            'bins': ratio_data,
        }


# ================================================================
# STEP 8: BOOTSTRAP CONFIDENCE INTERVALS
# ================================================================
print("\n" + "=" * 76)
print("BOOTSTRAP: Stability of σ²(n̄) shape")
print("=" * 76)

n_boot = 1000
boot_daic = []  # quantum vs classical
boot_peak_X = []  # peak location in X
boot_peak_nbar = []  # peak location in n̄

galaxy_list = list(galaxy_names_used)
n_gal_total = len(galaxy_list)

print(f"  Running {n_boot} galaxy-level bootstrap iterations...")

for b_iter in range(n_boot):
    # Resample galaxies with replacement
    boot_gals = set(np.random.choice(galaxy_list, size=n_gal_total, replace=True))
    boot_pts = [p for p in all_points if p['galaxy'] in boot_gals]

    if len(boot_pts) < 100:
        continue

    # Re-Z-score
    boot_res = np.array([p['log_res'] for p in boot_pts])
    b_mu, b_std = np.mean(boot_res), np.std(boot_res)
    if b_std < 1e-6:
        continue

    # Bin by log(n̄)
    b_var = []
    b_nbar = []
    b_nbar_sq = []
    b_var_err = []

    for j in range(len(lognbar_centers)):
        lo, hi = lognbar_edges[j], lognbar_edges[j + 1]
        z_vals = np.array([(p['log_res'] - b_mu) / b_std
                           for p in boot_pts
                           if lo <= p['log_nbar'] < hi])
        if len(z_vals) >= 15:
            var_obs = float(np.var(z_vals))
            var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
            gbar_lin = 10.0**(-(lognbar_centers[j] + 0.5))  # approximate inverse
            nbar_med = 10.0**lognbar_centers[j]
            b_var.append(var_obs)
            b_var_err.append(var_err)
            b_nbar.append(nbar_med)
            b_nbar_sq.append(nbar_med**2 + nbar_med)

    b_var = np.array(b_var)
    b_nbar = np.array(b_nbar)
    b_nbar_sq = np.array(b_nbar_sq)
    b_var_err = np.array(b_var_err)

    if len(b_var) < 4:
        continue

    try:
        pq, _ = curve_fit(lambda n, A, C: A * n + C, b_nbar_sq, b_var,
                           p0=[0.01, 0.5], sigma=b_var_err,
                           absolute_sigma=True, maxfev=3000)
        chi2q = np.sum(((b_var - (pq[0] * b_nbar_sq + pq[1])) / b_var_err)**2)

        pc, _ = curve_fit(lambda n, A, C: A * n + C, b_nbar, b_var,
                           p0=[0.01, 0.5], sigma=b_var_err,
                           absolute_sigma=True, maxfev=3000)
        chi2c = np.sum(((b_var - (pc[0] * b_nbar + pc[1])) / b_var_err)**2)

        daic = (chi2c + 4) - (chi2q + 4)  # both have 2 params
        boot_daic.append(daic)

        # Find peak variance location
        peak_j = np.argmax(b_var)
        boot_peak_nbar.append(np.log10(b_nbar[peak_j]))

    except Exception:
        continue

boot_daic = np.array(boot_daic)
boot_peak_nbar = np.array(boot_peak_nbar)

if len(boot_daic) > 50:
    pct_quantum = 100.0 * np.mean(boot_daic > 0)
    daic_med = np.median(boot_daic)
    daic_lo, daic_hi = np.percentile(boot_daic, [2.5, 97.5])

    print(f"\n  Bootstrap results ({len(boot_daic)} successful iterations):")
    print(f"  ΔAIC (classical − quantum): median = {daic_med:+.2f}, "
          f"95% CI = [{daic_lo:+.2f}, {daic_hi:+.2f}]")
    print(f"  Fraction preferring quantum: {pct_quantum:.1f}%")

    if len(boot_peak_nbar) > 50:
        peak_med = np.median(boot_peak_nbar)
        peak_lo, peak_hi = np.percentile(boot_peak_nbar, [16, 84])
        print(f"\n  Peak variance location: log(n̄) = {peak_med:.2f} "
              f"[{peak_lo:.2f}, {peak_hi:.2f}]")
        print(f"  Corresponds to n̄ ≈ {10**peak_med:.1f}")

    results['bootstrap'] = {
        'n_iterations': len(boot_daic),
        'pct_quantum_preferred': float(pct_quantum),
        'daic_median': float(daic_med),
        'daic_95ci': [float(daic_lo), float(daic_hi)],
    }
    if len(boot_peak_nbar) > 50:
        results['bootstrap']['peak_log_nbar_median'] = float(peak_med)
        results['bootstrap']['peak_log_nbar_68ci'] = [float(peak_lo), float(peak_hi)]


# ================================================================
# STEP 9: PLOT
# ================================================================
print("\n[9] Generating plots...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: σ²(n̄) with model fits
    ax1 = axes[0, 0]
    ax1.errorbar(bin_nbar, bin_var, yerr=bin_var_err,
                 fmt='ko', markersize=6, capsize=3, zorder=5,
                 label='Observed σ²')
    if 'test_a' in results:
        nbar_fine = np.logspace(np.log10(max(bin_nbar.min(), 0.001)),
                                np.log10(bin_nbar.max()), 200)
        nbar_sq_fine = nbar_fine**2 + nbar_fine
        ax1.plot(nbar_fine, popt_q[0] * nbar_sq_fine + popt_q[1],
                 'r-', lw=2, label=f'BEC: A(n̄²+n̄)+C  [AIC={aic_q:.1f}]')
        ax1.plot(nbar_fine, popt_c[0] * nbar_fine + popt_c[1],
                 'b--', lw=2, label=f'Classical: An̄+C  [AIC={aic_c:.1f}]')
        ax1.axhline(K, color='gray', ls=':', alpha=0.5,
                    label=f'Constant  [AIC={aic_k:.1f}]')
        if has_peak:
            ax1.plot(nbar_fine, model_peak(nbar_fine, *popt_p),
                     'g-.', lw=2,
                     label=f'Peak: n̄·exp(-n̄/{popt_p[1]:.1f})+C  [AIC={aic_p:.1f}]')
    ax1.set_xscale('log')
    ax1.set_xlabel('Occupation number n̄', fontsize=12)
    ax1.set_ylabel('Variance σ²(Z-residual)', fontsize=12)
    ax1.set_title('σ²(n̄): BEC/Shadow Prediction Test', fontsize=13)
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel 2: σ²(X) = σ²(R/ξ)
    ax2 = axes[0, 1]
    if len(bin_X_data) >= 4:
        X_logX_arr = np.array([b['logX_center'] for b in bin_X_data])
        X_var_arr = np.array([b['var_obs'] for b in bin_X_data])
        X_var_err_arr = np.array([b['var_err'] for b in bin_X_data])
        ax2.errorbar(10**X_logX_arr, X_var_arr, yerr=X_var_err_arr,
                     fmt='ko', markersize=6, capsize=3, zorder=5)
        ax2.axvline(1.0, color='red', ls='--', alpha=0.5, label='X = 1 (R = ξ)')
        ax2.axvline(3.0, color='orange', ls=':', alpha=0.5, label='X = 3')
        if 'test_b' in results and 'peaked_model' in results['test_b']:
            X_fine = np.logspace(-1.5, 2.0, 200)
            pm = results['test_b']['peaked_model']
            ax2.plot(X_fine, peak_X_model(X_fine, pm['A'], pm['X_peak'], pm['C']),
                     'g-.', lw=2, label=f'Peak model (X_peak={pm["X_peak"]:.2f})')
    ax2.set_xscale('log')
    ax2.set_xlabel('X = R/ξ', fontsize=12)
    ax2.set_ylabel('Variance σ²(Z-residual)', fontsize=12)
    ax2.set_title('Within-Galaxy Radial Variance Profile', fontsize=13)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Excess ratio σ²_obs/σ²_classical vs n̄
    ax3 = axes[1, 0]
    if 'test_c' in results:
        r_nbars = np.array([r['nbar'] for r in results['test_c']['bins']])
        r_ratios = np.array([r['ratio'] for r in results['test_c']['bins']])
        r_expected = np.array([r['nbar_plus_1'] for r in results['test_c']['bins']])
        ax3.scatter(r_nbars, r_ratios, c='black', s=50, zorder=5, label='Observed ratio')
        # Reference: (n̄+1) line
        nbar_ref = np.logspace(np.log10(max(r_nbars.min(), 0.001)),
                               np.log10(r_nbars.max()), 100)
        ax3.plot(nbar_ref, nbar_ref + 1, 'r-', lw=2, alpha=0.7,
                 label='BEC prediction: (n̄+1)')
        ax3.axhline(1.0, color='blue', ls='--', alpha=0.5,
                    label='Classical prediction: 1')
        rho = results['test_c']['spearman_rho']
        pval = results['test_c']['spearman_p']
        ax3.set_title(f'Excess Variance Ratio (ρ = {rho:+.3f}, p = {pval:.4f})',
                      fontsize=13)
    ax3.set_xscale('log')
    ax3.set_xlabel('Occupation number n̄', fontsize=12)
    ax3.set_ylabel('σ²_obs / σ²_classical', fontsize=12)
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Bootstrap ΔAIC distribution
    ax4 = axes[1, 1]
    if len(boot_daic) > 50:
        ax4.hist(boot_daic, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(0, color='red', ls='--', lw=2, label='ΔAIC = 0')
        ax4.axvline(np.median(boot_daic), color='orange', ls='-', lw=2,
                    label=f'Median = {np.median(boot_daic):+.2f}')
        pct = results.get('bootstrap', {}).get('pct_quantum_preferred', 0)
        ax4.set_title(f'Bootstrap ΔAIC ({pct:.0f}% favor quantum)', fontsize=13)
    ax4.set_xlabel('ΔAIC (classical − quantum)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'radial_variance_profile.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

except Exception as e:
    print(f"  Plot error: {e}")
    import traceback
    traceback.print_exc()


# ================================================================
# STEP 10: SUMMARY
# ================================================================
print("\n" + "=" * 76)
print("SUMMARY: RADIAL VARIANCE PROFILE TEST")
print("=" * 76)

if 'test_a' in results:
    r = results['test_a']
    print(f"\n  TEST A — σ²(n̄) continuous profile:")
    print(f"    Best model: {r['best_model'].upper()}")
    print(f"    ΔAIC (classical − quantum) = {r['daic_quantum_vs_classical']:+.2f}")
    if 'daic_peak_vs_classical' in r:
        print(f"    ΔAIC (classical − peak)    = {r['daic_peak_vs_classical']:+.2f}")

if 'test_b' in results:
    r = results['test_b']
    print(f"\n  TEST B — σ²(X=R/ξ) within-galaxy profile:")
    print(f"    Peak variance at X = {r['peak_X']:.2f} (log X = {r['peak_logX']:.2f})")
    print(f"    ΔAIC (decay − peaked) = {r['daic_peak_vs_decay']:+.2f}")
    if 0.3 < r['peak_X'] < 5.0:
        print(f"    >>> Peak near X ~ 1: SUPPORTS BEC/shadow framework")
    else:
        print(f"    >>> Peak not at X ~ 1: does not clearly support BEC")

if 'test_c' in results:
    r = results['test_c']
    print(f"\n  TEST C — Excess variance ratio:")
    print(f"    Spearman ρ(n̄, ratio) = {r['spearman_rho']:+.3f} (p = {r['spearman_p']:.4f})")
    if r['spearman_rho'] > 0 and r['spearman_p'] < 0.05:
        print(f"    >>> SUPER-POISSONIAN bunching detected!")
    elif r['spearman_rho'] > 0:
        print(f"    >>> Suggestive positive trend (not significant)")
    else:
        print(f"    >>> No super-Poissonian signal")

if 'bootstrap' in results:
    r = results['bootstrap']
    print(f"\n  BOOTSTRAP ({r['n_iterations']} iterations):")
    print(f"    {r['pct_quantum_preferred']:.1f}% prefer quantum")
    print(f"    Median ΔAIC = {r['daic_median']:+.2f}, "
          f"95% CI = [{r['daic_95ci'][0]:+.2f}, {r['daic_95ci'][1]:+.2f}]")

# Overall verdict
print(f"\n  INTERPRETATION (unified BEC/shadow framework):")
print(f"  If σ²(n̄) peaks near n̄ ~ 1 and follows (n̄+1) shape,")
print(f"  the 'reversal' is EXPECTED: inner = saturated projection (stable),")
print(f"  outer = evaporating projection (noisy). The condensation threshold")
print(f"  at g† is where the counting statistics transition — and that's")
print(f"  exactly where variance should peak.")

# Save results
results_path = os.path.join(OUTPUT_DIR, 'summary_radial_variance_profile.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved: {results_path}")

print(f"\n{'=' * 76}")
print("Done.")
