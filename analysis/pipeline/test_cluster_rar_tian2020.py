#!/usr/bin/env python3
"""
Cluster-Scale RAR from Tian et al. 2020 (CLASH)
=================================================

Uses pre-computed g_bar(r) and g_obs(r) for 20 CLASH clusters from
Tian et al. 2020 (ApJ 896, 70), downloaded from VizieR.

Tests:
  1. What is the cluster-scale acceleration scale a₀?
  2. Is it related to the galaxy-scale g† = 1.2×10⁻¹⁰?
  3. What is the scatter, and does it depend on radius?
  4. Does the RAR functional form g_obs = g_bar/(1-exp(-√(g_bar/a₀))) fit?

If BEC condensation operates at cluster scale with a mass-dependent
healing length, we expect a₀(cluster) > g†(galaxy) by a factor
related to the cluster-to-galaxy mass ratio.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'cluster_rar')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

g_dagger = 1.20e-10  # galaxy-scale
LOG_G_DAGGER = np.log10(g_dagger)

print("=" * 72)
print("CLUSTER-SCALE RAR FROM TIAN ET AL. 2020 (CLASH)")
print("=" * 72)


# ================================================================
# PARSE TIAN+2020 DATA
# ================================================================
print("\n[1] Parsing Tian+2020 VizieR data...")

fig2_path = os.path.join(DATA_DIR, 'tian2020_fig2.dat')
table1_path = os.path.join(DATA_DIR, 'tian2020_table1.dat')

# Parse fig2: radial RAR data (pipe-separated VizieR format)
cluster_data = {}
with open(fig2_path, 'r') as f:
    for line in f:
        line_s = line.strip()
        if not line_s or line_s.startswith('#') or line_s.startswith('-'):
            continue
        # Header lines contain letters in the numeric columns
        parts = line_s.split('|')
        if len(parts) < 6:
            continue
        name = parts[0].strip()
        try:
            rad = float(parts[1].strip())
            log_gbar = float(parts[2].strip())
            log_gtot = float(parts[3].strip())
            e_log_gbar = float(parts[4].strip())
            e_log_gtot = float(parts[5].strip())
        except ValueError:
            continue

        if name not in cluster_data:
            cluster_data[name] = {'rad': [], 'log_gbar': [], 'log_gtot': [],
                                   'e_log_gbar': [], 'e_log_gtot': []}
        cluster_data[name]['rad'].append(rad)
        cluster_data[name]['log_gbar'].append(log_gbar)
        cluster_data[name]['log_gtot'].append(log_gtot)
        cluster_data[name]['e_log_gbar'].append(e_log_gbar)
        cluster_data[name]['e_log_gtot'].append(e_log_gtot)

for name in cluster_data:
    for key in cluster_data[name]:
        cluster_data[name][key] = np.array(cluster_data[name][key])

n_clusters = len(cluster_data)
n_points = sum(len(v['rad']) for v in cluster_data.values())
print(f"  {n_clusters} clusters, {n_points} radial data points")
for name in sorted(cluster_data.keys()):
    print(f"    {name}: {len(cluster_data[name]['rad'])} radii "
          f"({cluster_data[name]['rad'].min():.0f}–{cluster_data[name]['rad'].max():.0f} kpc)")

# Collect all points
all_log_gbar = np.concatenate([v['log_gbar'] for v in cluster_data.values()])
all_log_gtot = np.concatenate([v['log_gtot'] for v in cluster_data.values()])
all_e_gbar = np.concatenate([v['e_log_gbar'] for v in cluster_data.values()])
all_e_gtot = np.concatenate([v['e_log_gtot'] for v in cluster_data.values()])

print(f"\n  g_bar range: [{all_log_gbar.min():.3f}, {all_log_gbar.max():.3f}]")
print(f"  g_tot range: [{all_log_gtot.min():.3f}, {all_log_gtot.max():.3f}]")


# ================================================================
# TEST 1: FIT THE RAR FUNCTION WITH FREE a₀
# ================================================================
print("\n" + "=" * 72)
print("TEST 1: FIT RAR g_obs = g_bar/(1 - exp(-√(g_bar/a₀)))")
print("=" * 72)

def rar_pred(log_gbar, log_a0):
    """Predict log g_obs given log g_bar and log a₀."""
    gbar = 10**log_gbar
    a0 = 10**log_a0
    ratio = np.sqrt(gbar / a0)
    # Avoid overflow in exp
    ratio = np.clip(ratio, 0, 500)
    gobs = gbar / (1 - np.exp(-ratio))
    return np.log10(gobs)

def chi2(log_a0):
    """Chi-squared for the RAR fit."""
    pred = rar_pred(all_log_gbar, log_a0)
    resid = all_log_gtot - pred
    sigma2 = all_e_gtot**2 + all_e_gbar**2  # add errors in quadrature
    return np.sum(resid**2 / sigma2)

# Scan a₀ from 10⁻¹¹ to 10⁻⁸
log_a0_grid = np.linspace(-11, -8, 300)
chi2_grid = [chi2(la) for la in log_a0_grid]
best_idx = np.argmin(chi2_grid)
best_log_a0 = log_a0_grid[best_idx]

# Refine with optimization
result = minimize_scalar(chi2, bounds=(best_log_a0 - 0.5, best_log_a0 + 0.5), method='bounded')
best_log_a0 = result.x
best_a0 = 10**best_log_a0
best_chi2 = result.fun
dof = n_points - 1
chi2_red = best_chi2 / dof

# 1σ interval: Δχ² = 1
chi2_min = best_chi2
log_a0_lo = best_log_a0
log_a0_hi = best_log_a0
for la in np.linspace(best_log_a0, best_log_a0 - 1, 200):
    if chi2(la) > chi2_min + 1:
        log_a0_lo = la
        break
for la in np.linspace(best_log_a0, best_log_a0 + 1, 200):
    if chi2(la) > chi2_min + 1:
        log_a0_hi = la
        break

ratio_to_gdagger = best_a0 / g_dagger

print(f"  Best-fit log a₀ = {best_log_a0:.4f} ({log_a0_lo:.4f}, {log_a0_hi:.4f})")
print(f"  a₀ = {best_a0:.3e} m/s²")
print(f"  Ratio a₀/g† = {ratio_to_gdagger:.1f}×")
print(f"  χ²/dof = {best_chi2:.1f}/{dof} = {chi2_red:.2f}")

# Residuals
pred_best = rar_pred(all_log_gbar, best_log_a0)
resid_best = all_log_gtot - pred_best
rms_scatter = np.std(resid_best)
print(f"  RMS scatter = {rms_scatter:.4f} dex")


# ================================================================
# TEST 2: COMPARE WITH GALAXY-SCALE g†
# ================================================================
print("\n" + "=" * 72)
print("TEST 2: COMPARE WITH GALAXY-SCALE g†")
print("=" * 72)

chi2_gdagger = chi2(LOG_G_DAGGER)
pred_gdagger = rar_pred(all_log_gbar, LOG_G_DAGGER)
resid_gdagger = all_log_gtot - pred_gdagger
rms_gdagger = np.std(resid_gdagger)

print(f"  Using galaxy g† = {g_dagger:.2e}:")
print(f"    χ² = {chi2_gdagger:.1f} (vs best-fit {best_chi2:.1f})")
print(f"    Δχ² = {chi2_gdagger - best_chi2:.1f}")
print(f"    RMS scatter = {rms_gdagger:.4f} dex (vs best-fit {rms_scatter:.4f})")
print(f"    Mean offset = {np.mean(resid_gdagger):+.4f} dex")

# Galaxy g† underpredicts g_obs at cluster scale?
if np.mean(resid_gdagger) > 0:
    print(f"    → Galaxy g† UNDERPREDICTS cluster g_obs (clusters are more DM-dominated)")
else:
    print(f"    → Galaxy g† OVERPREDICTS cluster g_obs")


# ================================================================
# TEST 3: SCATTER vs RADIUS
# ================================================================
print("\n" + "=" * 72)
print("TEST 3: SCATTER vs RADIUS (with best-fit a₀)")
print("=" * 72)

all_rad = np.concatenate([v['rad'] for v in cluster_data.values()])
unique_rad = sorted(set(all_rad))

print(f"  {'Radius (kpc)':<15} {'N_points':>10} {'σ (dex)':>10} {'Mean resid':>12}")
print(f"  " + "-" * 50)

scatter_by_radius = []
for r in unique_rad:
    mask = all_rad == r
    if np.sum(mask) < 3:
        continue
    r_resid = resid_best[mask]
    sigma = np.std(r_resid)
    mean_r = np.mean(r_resid)
    n_r = np.sum(mask)
    print(f"  {r:<15.0f} {n_r:>10} {sigma:>10.4f} {mean_r:>+12.4f}")
    scatter_by_radius.append({
        'radius_kpc': float(r), 'n_points': int(n_r),
        'sigma': float(sigma), 'mean_resid': float(mean_r),
    })


# ================================================================
# TEST 4: PER-CLUSTER FIT — Does each cluster have the same a₀?
# ================================================================
print("\n" + "=" * 72)
print("TEST 4: PER-CLUSTER a₀ (is it universal?)")
print("=" * 72)

print(f"  {'Cluster':<12} {'N_pts':>6} {'log a₀':>8} {'a₀/g†':>8} {'σ':>8}")
print(f"  " + "-" * 45)

per_cluster_a0 = []
for name in sorted(cluster_data.keys()):
    cd = cluster_data[name]
    if len(cd['rad']) < 3:
        continue

    def chi2_cl(la):
        pred = rar_pred(cd['log_gbar'], la)
        res = cd['log_gtot'] - pred
        return np.sum(res**2)

    res_cl = minimize_scalar(chi2_cl, bounds=(-12, -8), method='bounded')
    la_cl = res_cl.x
    a0_cl = 10**la_cl
    pred_cl = rar_pred(cd['log_gbar'], la_cl)
    sig_cl = np.std(cd['log_gtot'] - pred_cl)
    ratio_cl = a0_cl / g_dagger

    print(f"  {name:<12} {len(cd['rad']):>6} {la_cl:>8.3f} {ratio_cl:>8.1f} {sig_cl:>8.4f}")
    per_cluster_a0.append({
        'name': name, 'n_pts': len(cd['rad']),
        'log_a0': float(la_cl), 'a0_over_gdagger': float(ratio_cl),
        'scatter': float(sig_cl),
    })

all_cluster_log_a0 = [p['log_a0'] for p in per_cluster_a0]
print(f"\n  Mean log a₀ across clusters: {np.mean(all_cluster_log_a0):.3f} ± {np.std(all_cluster_log_a0):.3f}")
print(f"  Range: [{min(all_cluster_log_a0):.3f}, {max(all_cluster_log_a0):.3f}]")

# Is the scatter in log a₀ small (universal) or large (mass-dependent)?
if np.std(all_cluster_log_a0) < 0.3:
    universality = "CONSISTENT_WITH_UNIVERSAL"
    print(f"  → σ(log a₀) = {np.std(all_cluster_log_a0):.3f} < 0.3: CONSISTENT with universal a₀")
else:
    universality = "MASS_DEPENDENT"
    print(f"  → σ(log a₀) = {np.std(all_cluster_log_a0):.3f} ≥ 0.3: suggests MASS-DEPENDENT a₀")


# ================================================================
# TEST 5: MASS-DEPENDENT a₀ — Does a₀ correlate with cluster mass?
# ================================================================
print("\n" + "=" * 72)
print("TEST 5: a₀ vs CLUSTER MASS")
print("=" * 72)

# Parse table1 for cluster masses
cluster_masses = {}
with open(table1_path, 'r') as f:
    for line in f:
        if line.startswith('#') or line.startswith('-') or not line.strip():
            continue
        # The last field is AName
        parts = line.split('|')
        if len(parts) < 3:
            continue
        try:
            aname = parts[-1].strip()
            # Mtot is in units of 10^12 Msun
            # Field positions vary; look for numeric values
            # Format: Name|z|coords|Band|n|Re|eRe|Rad|M*|Mgas|eMgas|Mtot|eMtot|AName
            fields = line.strip().split('|')
            if len(fields) >= 14:
                mtot_str = fields[-3].strip()
                mstar_str = fields[-6].strip()
                if mtot_str and mstar_str:
                    mtot = float(mtot_str)  # 10^12 Msun
                    mstar = float(mstar_str)  # 10^12 Msun
                    cluster_masses[aname] = {
                        'Mtot_1e12': mtot,
                        'Mstar_1e12': mstar,
                    }
        except (ValueError, IndexError):
            continue

if cluster_masses:
    print(f"  Parsed masses for {len(cluster_masses)} clusters")

    # Match with per-cluster a₀
    matched = []
    for p in per_cluster_a0:
        if p['name'] in cluster_masses:
            cm = cluster_masses[p['name']]
            matched.append({
                'name': p['name'],
                'log_a0': p['log_a0'],
                'log_Mtot': np.log10(cm['Mtot_1e12'] * 1e12),  # in Msun
            })

    if len(matched) >= 5:
        log_a0_arr = np.array([m['log_a0'] for m in matched])
        log_M_arr = np.array([m['log_Mtot'] for m in matched])

        # Pearson correlation
        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(log_M_arr, log_a0_arr)
        r_s, p_s = spearmanr(log_M_arr, log_a0_arr)

        print(f"\n  log a₀ vs log M_tot:")
        print(f"    Pearson r = {r_p:.3f}, p = {p_p:.4f}")
        print(f"    Spearman ρ = {r_s:.3f}, p = {p_s:.4f}")

        if p_p < 0.05:
            print(f"    → SIGNIFICANT correlation: a₀ depends on cluster mass")
        else:
            print(f"    → No significant correlation (p > 0.05)")

        # BEC prediction: if ξ ∝ M^α, then a₀ ∝ M^(-2α)
        # Fit: log a₀ = slope * log M + intercept
        slope = np.polyfit(log_M_arr, log_a0_arr, 1)[0]
        print(f"    Best-fit slope d(log a₀)/d(log M) = {slope:.3f}")
        if abs(slope) > 0.01:
            alpha = -slope / 2
            print(f"    → Implies healing length ξ ∝ M^{alpha:.2f}")
else:
    print("  (Could not parse cluster masses from table1)")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

print(f"\n  Cluster RAR:")
print(f"    {n_clusters} CLASH clusters, {n_points} data points")
print(f"    Best-fit a₀ = {best_a0:.3e} m/s² (log = {best_log_a0:.3f})")
print(f"    a₀/g† = {ratio_to_gdagger:.1f}× (galaxy g† = 1.2×10⁻¹⁰)")
print(f"    RMS scatter = {rms_scatter:.4f} dex")
print(f"    Per-cluster σ(log a₀) = {np.std(all_cluster_log_a0):.3f}")

if ratio_to_gdagger > 5 and ratio_to_gdagger < 50:
    verdict = f"DISTINCT_SCALE_{ratio_to_gdagger:.0f}x"
    print(f"\n  >>> Cluster a₀ is {ratio_to_gdagger:.0f}× larger than galaxy g†")
    print(f"      This is consistent with Tian+2020 finding (~17×)")
    print(f"      Interpretation: different condensation scale at cluster mass")
else:
    verdict = "UNEXPECTED"

# ================================================================
# SAVE
# ================================================================
results = {
    'test_name': 'cluster_rar_tian2020',
    'n_clusters': n_clusters,
    'n_points': n_points,
    'best_fit': {
        'log_a0': float(best_log_a0),
        'log_a0_1sigma': [float(log_a0_lo), float(log_a0_hi)],
        'a0': float(best_a0),
        'a0_over_gdagger': float(ratio_to_gdagger),
        'chi2': float(best_chi2),
        'dof': int(dof),
        'chi2_red': float(chi2_red),
        'rms_scatter': float(rms_scatter),
    },
    'galaxy_gdagger_comparison': {
        'chi2_gdagger': float(chi2_gdagger),
        'delta_chi2': float(chi2_gdagger - best_chi2),
        'rms_scatter_gdagger': float(rms_gdagger),
        'mean_offset': float(np.mean(resid_gdagger)),
    },
    'scatter_by_radius': scatter_by_radius,
    'per_cluster_a0': per_cluster_a0,
    'per_cluster_stats': {
        'mean_log_a0': float(np.mean(all_cluster_log_a0)),
        'std_log_a0': float(np.std(all_cluster_log_a0)),
        'universality': universality,
    },
    'overall_verdict': verdict,
}

out_path = os.path.join(RESULTS_DIR, 'summary_cluster_rar_tian2020.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_path}")
print("=" * 72)
print("Done.")
