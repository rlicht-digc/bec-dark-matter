#!/usr/bin/env python3
"""
Mass-Split Bunching Test: X = R/ξ Prediction

The BEC framework predicts that bunching (σ² ∝ n̄(n̄+1)) should be
STRONGER for massive galaxies (large ξ → small X = R/ξ → inside
coherence length) and WEAKER for dwarfs (small ξ → large X → outside
coherence length).

This test splits the SPARC+multi-survey data by stellar mass proxy
(from gbar: logMs ∝ logMh from the pipeline) and runs the bunching
test in each mass bin. If ΔAIC increases with mass, it confirms
the quantum→classical transition at the healing length scale.

Prediction:
  Dwarfs  (M* ~ 10⁸):  ξ ~ 0.3 kpc → X ~ 3-100 → WEAK bunching
  Spirals (M* ~ 10¹⁰): ξ ~ 3.4 kpc → X ~ 1-10  → MODERATE bunching
  Massive (M* ~ 10¹¹): ξ ~ 10.8 kpc → X ~ 0.5-3 → STRONG bunching
"""
import os
import sys
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physical constants
G_conv = 4.302e-3  # pc (km/s)^2 / Msun
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

def xi_kpc(logMs):
    """Healing length in kpc."""
    Ms_SI = 10.0**logMs * Msun_kg
    return np.sqrt(G_SI * Ms_SI / gdagger) / kpc_m


print("=" * 72)
print("MASS-SPLIT BUNCHING TEST: X = R/ξ PREDICTION")
print("=" * 72)

# ================================================================
# STEP 1: Load SPARC data (the largest, cleanest dataset)
# ================================================================
print("\n[1] Loading SPARC rotation curves...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

if not os.path.exists(table2_path) or not os.path.exists(mrt_path):
    # Fallbacks for the standard repository layout (data/sparc/*).
    alt_table2 = [
        os.path.join(DATA_DIR, 'sparc', 'SPARC_table2_rotmods.dat'),
        os.path.join(PROJECT_ROOT, 'data', 'sparc', 'SPARC_table2_rotmods.dat'),
    ]
    alt_mrt = [
        os.path.join(DATA_DIR, 'sparc', 'SPARC_Lelli2016c.mrt'),
        os.path.join(PROJECT_ROOT, 'data', 'sparc', 'SPARC_Lelli2016c.mrt'),
    ]
    for p in alt_table2:
        if os.path.exists(p):
            table2_path = p
            break
    for p in alt_mrt:
        if os.path.exists(p):
            mrt_path = p
            break

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
# STEP 2: Compute RAR residuals per galaxy
# ================================================================
print("\n[2] Computing RAR residuals...")

all_points = []
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

    # Compute gbar from baryonic components (disk Υ* = 0.5, bulge Υ* = 0.7)
    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    # V in km/s, R in kpc: gbar_SI = (V*1000)² / (R * 3.086e19)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * 1e6 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 0)
    if np.sum(valid) < 3:
        continue

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    log_gobs_rar = rar_function(log_gbar)
    log_res = log_gobs - log_gobs_rar  # RAR residual

    logMs = prop['logMs']

    for i in range(len(log_gbar)):
        all_points.append({
            'galaxy': name,
            'log_gbar': float(log_gbar[i]),
            'log_gobs': float(log_gobs[i]),
            'log_res': float(log_res[i]),
            'R_kpc': float(R[valid][i]),
            'logMs': logMs,
        })

print(f"  {len(all_points)} total RAR points from {len(set(p['galaxy'] for p in all_points))} galaxies")

# ================================================================
# STEP 3: Z-score residuals (remove systematic offsets)
# ================================================================
print("\n[3] Z-scoring residuals...")

# Z-score globally (single dataset)
all_res = np.array([p['log_res'] for p in all_points])
mu = np.mean(all_res)
std = np.std(all_res)
for p in all_points:
    p['z_res'] = (p['log_res'] - mu) / std

print(f"  Global: μ = {mu:.4f}, σ = {std:.4f}")

# ================================================================
# STEP 4: Split by stellar mass and run bunching test
# ================================================================
print("\n[4] Running mass-split bunching test...")

# Mass bins: logMs = [8, 9, 9.5, 10, 10.5, 11]
mass_bins = [(8.0, 9.5, "Dwarf (logM*<9.5)"),
             (9.5, 10.3, "Low-mass spiral (9.5-10.3)"),
             (10.3, 11.0, "Massive spiral (logM*>10.3)")]

# Bunching test function
def run_bunching_test(points, label):
    """Run boson bunching test on a set of Z-scored RAR points."""
    bunching_edges = np.arange(-13.0, -8.0, 0.4)
    bunching_centers = (bunching_edges[:-1] + bunching_edges[1:]) / 2.0

    bin_var = []
    bin_var_err = []
    bin_nbar = []
    bin_nbar_sq = []
    bin_gbar_c = []
    bin_N = []

    for j in range(len(bunching_centers)):
        lo, hi = bunching_edges[j], bunching_edges[j + 1]
        z_vals = np.array([p['z_res'] for p in points
                           if lo <= p['log_gbar'] < hi])
        if len(z_vals) >= 20:  # slightly relaxed threshold for subsamples
            var_obs = float(np.var(z_vals))
            var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
            gbar_lin = 10.0 ** bunching_centers[j]
            x = np.sqrt(gbar_lin / gdagger)
            nbar = 1.0 / (np.exp(x) - 1.0 + 1e-30)
            bin_var.append(var_obs)
            bin_var_err.append(var_err)
            bin_nbar.append(nbar)
            bin_nbar_sq.append(nbar**2 + nbar)
            bin_gbar_c.append(bunching_centers[j])
            bin_N.append(len(z_vals))

    bin_var = np.array(bin_var)
    bin_var_err = np.array(bin_var_err)
    bin_nbar = np.array(bin_nbar)
    bin_nbar_sq = np.array(bin_nbar_sq)
    bin_N = np.array(bin_N)

    if len(bin_var) < 4:
        return None

    try:
        # Quantum: σ² = A*(n̄²+n̄) + C
        def quantum_model(n, A, C):
            return A * n + C
        popt_q, _ = curve_fit(quantum_model, bin_nbar_sq, bin_var,
                               p0=[0.1, 0.5], sigma=bin_var_err,
                               absolute_sigma=True, maxfev=5000)
        resid_q = bin_var - quantum_model(bin_nbar_sq, *popt_q)
        chi2_q = np.sum((resid_q / bin_var_err)**2)

        # Classical: σ² = A*n̄ + C
        def classical_model(n, A, C):
            return A * n + C
        popt_c, _ = curve_fit(classical_model, bin_nbar, bin_var,
                               p0=[0.1, 0.5], sigma=bin_var_err,
                               absolute_sigma=True, maxfev=5000)
        resid_c = bin_var - classical_model(bin_nbar, *popt_c)
        chi2_c = np.sum((resid_c / bin_var_err)**2)

        # Constant
        wt = 1.0 / np.maximum(bin_var_err, 1e-6)
        mean_var = np.average(bin_var, weights=wt**2)
        chi2_const = np.sum(((bin_var - mean_var) / bin_var_err)**2)

        aic_q = chi2_q + 2 * 2
        aic_c = chi2_c + 2 * 2
        aic_const = chi2_const + 2 * 1

        delta_aic = aic_c - aic_q  # positive = quantum preferred

        return {
            'n_bins': len(bin_var),
            'n_points': int(np.sum(bin_N)),
            'chi2_quantum': chi2_q,
            'chi2_classical': chi2_c,
            'chi2_constant': chi2_const,
            'aic_quantum': aic_q,
            'aic_classical': aic_c,
            'delta_aic': delta_aic,
            'A_quantum': float(popt_q[0]),
            'C_quantum': float(popt_q[1]),
            'A_classical': float(popt_c[0]),
            'C_classical': float(popt_c[1]),
        }
    except Exception as e:
        return None


# Run bunching test for ALL data first
print("\n  --- ALL SPARC GALAXIES ---")
result_all = run_bunching_test(all_points, "All")
if result_all:
    print(f"  {result_all['n_points']} points in {result_all['n_bins']} gbar bins")
    print(f"  ΔAIC (classical - quantum) = {result_all['delta_aic']:+.2f}")
    print(f"  {'→ Quantum preferred' if result_all['delta_aic'] > 0 else '→ Classical preferred'}")

# Run per mass bin
results_by_mass = []
print("\n  --- MASS-SPLIT RESULTS ---")
for logMs_lo, logMs_hi, label in mass_bins:
    pts = [p for p in all_points if logMs_lo <= p['logMs'] < logMs_hi]
    n_gal = len(set(p['galaxy'] for p in pts))
    logMs_med = np.median([p['logMs'] for p in pts]) if pts else 0
    xi = xi_kpc(logMs_med) if logMs_med > 0 else 0
    R_med = np.median([p['R_kpc'] for p in pts]) if pts else 0
    X_med = R_med / xi if xi > 0 else 999

    print(f"\n  {label} ({n_gal} galaxies, {len(pts)} points)")
    print(f"    Median logM* = {logMs_med:.2f}, ξ = {xi:.2f} kpc, "
          f"median R = {R_med:.1f} kpc → X = {X_med:.1f}")

    result = run_bunching_test(pts, label)
    if result:
        print(f"    {result['n_bins']} gbar bins, "
              f"ΔAIC = {result['delta_aic']:+.2f} "
              f"{'→ Quantum' if result['delta_aic'] > 0 else '→ Classical'}")
        results_by_mass.append({
            'label': label,
            'logMs_lo': logMs_lo,
            'logMs_hi': logMs_hi,
            'logMs_med': logMs_med,
            'xi_kpc': xi,
            'R_median': R_med,
            'X_median': X_med,
            'n_galaxies': n_gal,
            'n_points': len(pts),
            'delta_aic': result['delta_aic'],
            'result': result,
        })
    else:
        print(f"    Insufficient data for bunching test")


# ================================================================
# STEP 5: Correlation of ΔAIC with X
# ================================================================
rho_X = None
p_X = None
rho_M = None
p_M = None
mass_trend_verdict = "INSUFFICIENT_BINS"
if len(results_by_mass) >= 2:
    print("\n" + "=" * 72)
    print("MASS-DEPENDENT BUNCHING RESULTS")
    print("=" * 72)

    print(f"\n  {'Mass bin':30s} {'logM*':>6s} {'ξ(kpc)':>7s} "
          f"{'R_med':>6s} {'X_med':>6s} {'ΔAIC':>8s} {'Result':>10s}")
    print(f"  {'-'*80}")
    for r in results_by_mass:
        verdict = "QUANTUM" if r['delta_aic'] > 2 else "INDIST." if r['delta_aic'] > -2 else "CLASSICAL"
        print(f"  {r['label']:30s} {r['logMs_med']:>6.2f} {r['xi_kpc']:>7.2f} "
              f"{r['R_median']:>6.1f} {r['X_median']:>6.1f} "
              f"{r['delta_aic']:>+8.2f} {verdict:>10s}")

    X_vals = np.array([r['X_median'] for r in results_by_mass])
    daic_vals = np.array([r['delta_aic'] for r in results_by_mass])
    logMs_vals = np.array([r['logMs_med'] for r in results_by_mass])

    # BEC prediction: ΔAIC should INCREASE with M* (decrease with X)
    # because larger M* → larger ξ → smaller X → more coherent → stronger bunching
    print(f"\n  BEC PREDICTION TEST:")
    print(f"    Does ΔAIC increase with stellar mass (decrease with X)?")

    if len(results_by_mass) >= 3:
        rho_X, p_X = spearmanr(X_vals, daic_vals)
        rho_M, p_M = spearmanr(logMs_vals, daic_vals)
        print(f"    Spearman ρ(X, ΔAIC) = {rho_X:+.3f} (p = {p_X:.3f})")
        print(f"    Spearman ρ(logM*, ΔAIC) = {rho_M:+.3f} (p = {p_M:.3f})")
        if rho_X < 0 and rho_M > 0:
            mass_trend_verdict = "CONSISTENT_WITH_BEC_TRANSITION"
            print(f"    ✓ ΔAIC increases with M* and decreases with X!")
            print(f"      Consistent with quantum→classical transition at ξ")
        elif rho_M > 0:
            mass_trend_verdict = "PARTIAL_MASS_TREND"
            print(f"    ~ ΔAIC increases with M* but X correlation unclear")
        else:
            mass_trend_verdict = "NO_EXPECTED_MASS_TREND"
            print(f"    ✗ No trend: ΔAIC does not increase with M*")

    # More detailed: show the X trend
    print(f"\n  INTERPRETATION:")
    for r in results_by_mass:
        xi = r['xi_kpc']
        X = r['X_median']
        daic = r['delta_aic']
        if X < 3:
            regime = "CORE (X<3): coherent → expect STRONG bunching"
        elif X < 10:
            regime = "TRANSITION (3≤X≤10): partially coherent → MODERATE"
        else:
            regime = "ENVELOPE (X>10): classical → expect WEAK bunching"
        match = "✓" if (X < 5 and daic > 0) or (X > 10 and daic < 0) or (5 <= X <= 10) else "~"
        print(f"    {match} {r['label']:30s}: X={X:.1f} → {regime}")
        print(f"      Observed ΔAIC = {daic:+.2f}")


# ================================================================
# STEP 6: Within-galaxy X test (strongest version)
# ================================================================
print("\n" + "=" * 72)
print("WITHIN-GALAXY RADIAL TEST: inner vs outer points")
print("=" * 72)
print("  For each galaxy, split rotation curve at R = ξ")
print("  Inner (R < ξ): should show bunching")
print("  Outer (R >> ξ): should approach Poisson")

inner_points = []
outer_points = []
for name, gdata in galaxies.items():
    if name not in sparc_props:
        continue
    logMs = sparc_props[name]['logMs']
    xi = xi_kpc(logMs)
    pts = [p for p in all_points if p['galaxy'] == name]
    for p in pts:
        if p['R_kpc'] < xi:
            inner_points.append(p)
        elif p['R_kpc'] > 3.0 * xi:
            outer_points.append(p)

print(f"\n  Inner (R < ξ): {len(inner_points)} points")
print(f"  Outer (R > 3ξ): {len(outer_points)} points")

result_inner = run_bunching_test(inner_points, "Inner (R<ξ)")
result_outer = run_bunching_test(outer_points, "Outer (R>3ξ)")
inner_minus_outer_daic = None
radial_verdict = "INSUFFICIENT_DATA"

if result_inner:
    print(f"\n  Inner bunching: {result_inner['n_bins']} bins, "
          f"ΔAIC = {result_inner['delta_aic']:+.2f} "
          f"{'→ Quantum' if result_inner['delta_aic'] > 0 else '→ Classical'}")
else:
    print(f"\n  Inner: insufficient data (only {len(inner_points)} points)")

if result_outer:
    print(f"  Outer bunching: {result_outer['n_bins']} bins, "
          f"ΔAIC = {result_outer['delta_aic']:+.2f} "
          f"{'→ Quantum' if result_outer['delta_aic'] > 0 else '→ Classical'}")
else:
    print(f"  Outer: insufficient data")

if result_inner and result_outer:
    inner_minus_outer_daic = result_inner['delta_aic'] - result_outer['delta_aic']
    print(f"\n  Inner ΔAIC − Outer ΔAIC = {inner_minus_outer_daic:+.2f}")
    if result_inner['delta_aic'] > result_outer['delta_aic']:
        radial_verdict = "INNER_STRONGER_THAN_OUTER"
        print(f"  ✓ INNER shows STRONGER bunching than OUTER!")
        print(f"    Consistent with quantum coherence at R < ξ")
    else:
        radial_verdict = "NO_EXPECTED_RADIAL_TREND"
        print(f"  ✗ No radial trend in bunching strength")


# ================================================================
# STEP 7: Save summary JSON
# ================================================================
overall_verdict = "INCONCLUSIVE"
if mass_trend_verdict == "CONSISTENT_WITH_BEC_TRANSITION" and radial_verdict == "INNER_STRONGER_THAN_OUTER":
    overall_verdict = "SUPPORTS_BEC_TRANSITION"
elif mass_trend_verdict in {"NO_EXPECTED_MASS_TREND"} or radial_verdict == "NO_EXPECTED_RADIAL_TREND":
    overall_verdict = "CONTRADICTS_BEC_TRANSITION"

summary = {
    'test_name': 'mass_split_bunching',
    'description': ('Mass-binned bunching test using X=R/xi to assess whether quantum '
                    'bunching strengthens with stellar mass and at radii inside the healing length.'),
    'n_total_points': int(len(all_points)),
    'n_total_galaxies': int(len(set(p['galaxy'] for p in all_points))),
    'global_zscore': {
        'mu': float(mu),
        'sigma': float(std),
    },
    'all_sparc': result_all,
    'mass_bin_results': [
        {
            'label': r['label'],
            'logMs_range': [r['logMs_lo'], r['logMs_hi']],
            'logMs_median': float(r['logMs_med']),
            'xi_kpc': float(r['xi_kpc']),
            'R_median_kpc': float(r['R_median']),
            'X_median': float(r['X_median']),
            'n_galaxies': int(r['n_galaxies']),
            'n_points': int(r['n_points']),
            'delta_aic_classical_minus_quantum': float(r['delta_aic']),
            'fit': r['result'],
        }
        for r in results_by_mass
    ],
    'mass_trend': {
        'spearman_rho_X_vs_daic': float(rho_X) if rho_X is not None else None,
        'spearman_p_X_vs_daic': float(p_X) if p_X is not None else None,
        'spearman_rho_logMs_vs_daic': float(rho_M) if rho_M is not None else None,
        'spearman_p_logMs_vs_daic': float(p_M) if p_M is not None else None,
        'verdict': mass_trend_verdict,
    },
    'radial_inner_outer_test': {
        'n_inner_points': int(len(inner_points)),
        'n_outer_points': int(len(outer_points)),
        'inner_fit': result_inner,
        'outer_fit': result_outer,
        'inner_minus_outer_delta_aic': float(inner_minus_outer_daic) if inner_minus_outer_daic is not None else None,
        'verdict': radial_verdict,
    },
    'overall_verdict': overall_verdict,
}

outpath = os.path.join(RESULTS_DIR, 'summary_mass_split_bunching.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {outpath}")

print(f"\n{'=' * 72}")
print("Done.")
