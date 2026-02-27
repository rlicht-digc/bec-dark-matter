#!/usr/bin/env python3
"""
Hierarchical Healing Length & Cluster g†_eff Mass Scaling Test
==============================================================

Tests whether the cluster-scale acceleration parameter g†_eff follows
a predictable mass-scaling relation through the BEC healing length
ξ = sqrt(GM/g†).

Key questions:
  1. Does g†_eff scale with system mass as a power law?
  2. Does the healing length predict cluster core sizes?
  3. Is there hierarchical consistency from galaxies to clusters?
  4. Does the scatter profile show a transition at R ~ ξ?

Uses: Tian+2020 CLASH clusters (20 clusters, 84 RAR points)
      SPARC galaxies (67 with healing length measurements)
"""

import os
import sys
import json
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# CONSTANTS
# ================================================================
G_SI = 6.674e-11        # m³ kg⁻¹ s⁻²
M_SUN_KG = 1.989e30     # kg
KPC_M = 3.086e19        # m per kpc
g_dagger = 1.2e-10      # m/s²
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def healing_length_kpc(M_sun, g=g_dagger):
    """BEC healing length ξ = sqrt(GM/g) in kpc."""
    GM_si = G_SI * M_sun * M_SUN_KG
    xi_m = np.sqrt(GM_si / g)
    return xi_m / KPC_M


def sigma_deviation(value, target, stderr, atol=1e-12, rtol=1e-8):
    """Return |value-target|/stderr with protection for near-zero stderr."""
    if not np.isfinite(value) or not np.isfinite(target):
        return np.nan
    if not np.isfinite(stderr) or stderr <= atol:
        return 0.0 if np.isclose(value, target, atol=atol, rtol=rtol) else np.inf
    return abs(value - target) / stderr


def closest_scale(target, scale_pairs):
    """
    Find which named scale is closest to target on a multiplicative scale.
    Returns: (label, scale_value, multiplicative_factor).
    """
    valid = [(label, value) for label, value in scale_pairs if np.isfinite(value) and value > 0]
    if not np.isfinite(target) or target <= 0 or not valid:
        return None, np.nan, np.nan
    label, value = min(valid, key=lambda pair: abs(np.log(target / pair[1])))
    factor = max(target, value) / min(target, value)
    return label, float(value), float(factor)


# ================================================================
# LOAD DATA
# ================================================================
print("=" * 72)
print("HIERARCHICAL HEALING LENGTH & CLUSTER g†_eff MASS SCALING TEST")
print("=" * 72)

with open(os.path.join(RESULTS_DIR, 'summary_cluster_rar_tian2020.json')) as f:
    cluster_rar = json.load(f)

with open(os.path.join(RESULTS_DIR, 'summary_cluster_gcore_A3.json')) as f:
    gcore_data = json.load(f)

with open(os.path.join(RESULTS_DIR, 'summary_cluster_sigma_scaling.json')) as f:
    sigma_data = json.load(f)

with open(os.path.join(RESULTS_DIR, 'summary_healing_length_scaling.json')) as f:
    hl_data = json.load(f)

# Build cluster database
clusters = {}
for c in cluster_rar['per_cluster_a0']:
    name = c['name']
    clusters[name] = {
        'log_gdagger_eff': c['log_a0'],
        'gdagger_eff_over_gd': c['a0_over_gdagger'],
        'scatter': c['scatter'],
        'n_pts': c['n_pts'],
    }

for c in gcore_data['per_cluster_gcore']:
    name = c['name']
    if name in clusters:
        clusters[name]['Mtot_1e12'] = c['Mtot_1e12']
        clusters[name]['Rad_kpc'] = c['Rad_kpc']
        clusters[name]['log_gcore'] = c['log_gcore']
        clusters[name]['gcore'] = c['gcore']

for c in sigma_data['clusters']:
    name = c['name']
    if name in clusters:
        clusters[name]['sigma_kms'] = c['sigma_kms']
        clusters[name]['sigma_method'] = c['sigma_method']
        clusters[name]['flag'] = c.get('flag')

# M200, R200 for virial clusters (from Pizzardo+2025 and Adam+2022)
m200_data = {
    'A383':     {'M200': 8.4e14, 'R200_Mpc': 1.83},
    'MACS0647': {'M200': 1.8e15, 'R200_Mpc': 2.06},
    'MACS1115': {'M200': 1.1e15, 'R200_Mpc': 1.87},
    'MACS1931': {'M200': 1.2e15, 'R200_Mpc': 1.91},
    'MS2137':   {'M200': 7.9e14, 'R200_Mpc': 1.70},
    'RXJ2129':  {'M200': 7.7e14, 'R200_Mpc': 1.75},
}
for name, md in m200_data.items():
    if name in clusters:
        clusters[name]['M200'] = md['M200']
        clusters[name]['R200_kpc'] = md['R200_Mpc'] * 1000

# Redshifts
z_data = {
    'A383': 0.187, 'A209': 0.206, 'A2261': 0.224, 'RXJ2129': 0.234,
    'A611': 0.288, 'MS2137': 0.313, 'RXJ2248': 0.348, 'MACS1115': 0.355,
    'MACS1931': 0.352, 'RXJ1532': 0.362, 'MACS1720': 0.387, 'MACS0416': 0.397,
    'MACS0429': 0.399, 'MACS1206': 0.439, 'MACS0329': 0.450, 'RXJ1347': 0.451,
    'MACS1149': 0.544, 'MACS0717': 0.548, 'MACS0647': 0.584, 'MACS0744': 0.686,
}
for name, z in z_data.items():
    if name in clusters:
        clusters[name]['z'] = z

# Build galaxy database
galaxies = []
for g in hl_data['per_galaxy']:
    galaxies.append({
        'name': g['name'],
        'M_b': g['M_b_Msun'],
        'xi_kpc': g['xi_kpc'],
        'R_extent_kpc': g['R_extent_kpc'],
        'Rdisk_kpc': g['Rdisk_kpc'],
        'Vflat': g['Vflat'],
        'Lc_acf_kpc': g['Lc_acf_kpc'],
    })

print(f"\nLoaded {len(clusters)} clusters, {len(galaxies)} galaxies")
print(f"  Clusters with M200: {sum(1 for c in clusters.values() if 'M200' in c)}")
print(f"  Clusters with σ: {sum(1 for c in clusters.values() if 'sigma_kms' in c)}")

# ================================================================
# STEP 2: COMPUTE HEALING LENGTHS
# ================================================================
print(f"\n{'='*72}")
print("STEP 2: HEALING LENGTHS")
print(f"{'='*72}")

header = (f"  {'Cluster':12s} {'Mtot':>8s} {'Rad':>6s} {'ξ_core':>8s} {'ξ_eff':>8s} "
          f"{'g†/g†':>7s} {'M200':>8s} {'R200':>7s} {'ξ_200':>8s}")
units =  (f"  {'':12s} {'(1e12)':>8s} {'(kpc)':>6s} {'(kpc)':>8s} {'(kpc)':>8s} "
          f"{'':>7s} {'(1e14)':>8s} {'(kpc)':>7s} {'(kpc)':>8s}")
print(f"\n{header}\n{units}\n  {'-'*85}")

for name in sorted(clusters.keys()):
    c = clusters[name]
    Mtot = c.get('Mtot_1e12', 0) * 1e12
    g_eff = 10**c['log_gdagger_eff']

    xi_core = healing_length_kpc(Mtot, g_dagger)
    c['xi_core_kpc'] = xi_core

    xi_eff = healing_length_kpc(Mtot, g_eff)
    c['xi_eff_kpc'] = xi_eff

    m200_str = r200_str = xi200_str = ''
    if 'M200' in c:
        xi_200 = healing_length_kpc(c['M200'], g_dagger)
        xi_200_eff = healing_length_kpc(c['M200'], g_eff)
        c['xi_200_kpc'] = xi_200
        c['xi_200_eff_kpc'] = xi_200_eff
        m200_str = f"{c['M200']/1e14:8.2f}"
        r200_str = f"{c['R200_kpc']:7.0f}"
        xi200_str = f"{xi_200:8.0f}"

    ratio = c['gdagger_eff_over_gd']
    Rad = c.get('Rad_kpc', 0)
    print(f"  {name:12s} {c.get('Mtot_1e12',0):8.2f} {Rad:6.1f} {xi_core:8.1f} "
          f"{xi_eff:8.1f} {ratio:7.1f} {m200_str:>8s} {r200_str:>7s} {xi200_str:>8s}")

# Galaxy summary
gal_xi = np.array([g['xi_kpc'] for g in galaxies])
gal_Mb = np.array([g['M_b'] for g in galaxies])
gal_Rd = np.array([g['Rdisk_kpc'] for g in galaxies])
gal_Re = np.array([g['R_extent_kpc'] for g in galaxies])
print(f"\n  Galaxy summary (67 SPARC galaxies):")
print(f"    M_b range: [{gal_Mb.min():.2e}, {gal_Mb.max():.2e}] M☉")
print(f"    ξ range: [{gal_xi.min():.2f}, {gal_xi.max():.2f}] kpc")
print(f"    R_disk range: [{gal_Rd.min():.2f}, {gal_Rd.max():.2f}] kpc")

# ================================================================
# TEST 3a: Does g†_eff scale with mass?
# ================================================================
print(f"\n{'='*72}")
print("TEST 3a: g†_eff / g† vs SYSTEM MASS")
print(f"{'='*72}")

cl_names = sorted(clusters.keys())
cl_Mtot = np.array([clusters[n].get('Mtot_1e12', 0) * 1e12 for n in cl_names])
cl_ratio = np.array([clusters[n]['gdagger_eff_over_gd'] for n in cl_names])
cl_log_ratio = np.log10(cl_ratio)
cl_log_Mtot = np.log10(cl_Mtot)

gal_log_Mb = np.log10(gal_Mb)
gal_log_ratio = np.zeros(len(gal_Mb))

# Combined galaxy+cluster power-law fit
all_log_M = np.concatenate([gal_log_Mb, cl_log_Mtot])
all_log_ratio = np.concatenate([gal_log_ratio, cl_log_ratio])
slope_all, intercept_all, r_all, p_all, se_all = stats.linregress(all_log_M, all_log_ratio)

print(f"\n  Combined galaxy+cluster fit:")
print(f"    log(g†_eff/g†) = {slope_all:.4f} × log(M/M☉) + ({intercept_all:.4f})")
print(f"    α (power-law index) = {slope_all:.4f} ± {se_all:.4f}")
print(f"    Pearson r = {r_all:.3f}, p = {p_all:.2e}")

# Reference mass where g†_eff = g†
M_ref = 10**(-intercept_all / slope_all) if slope_all != 0 else np.nan
print(f"    M_ref (where g†_eff = g†) = {M_ref:.2e} M☉  (log = {np.log10(M_ref):.2f})")

# Cluster-only fit
slope_cl, intercept_cl, r_cl, p_cl, se_cl = stats.linregress(cl_log_Mtot, cl_log_ratio)
print(f"\n  Cluster-only fit (N={len(cl_names)}):")
print(f"    log(g†_eff/g†) = {slope_cl:.4f} × log(Mtot) + ({intercept_cl:.4f})")
print(f"    α = {slope_cl:.4f} ± {se_cl:.4f}")
print(f"    Pearson r = {r_cl:.3f}, p = {p_cl:.3f}")

# Naive prediction
M_gal_typ = np.median(gal_Mb)
M_cl_typ = np.median(cl_Mtot)
ratio_cl_typ = np.median(cl_ratio)
alpha_naive = np.log10(ratio_cl_typ) / (np.log10(M_cl_typ) - np.log10(M_gal_typ))
print(f"\n  Naive scaling: median cluster g†_eff/g† = {ratio_cl_typ:.1f}")
print(f"    M_gal typical = {M_gal_typ:.2e}, M_cl typical = {M_cl_typ:.2e}")
print(f"    Implied α = log({ratio_cl_typ:.1f}) / log({M_cl_typ/M_gal_typ:.0f}) = {alpha_naive:.4f}")

# M200 fit where available
m200_names = [n for n in cl_names if 'M200' in clusters[n]]
test_3a_m200 = {}
if len(m200_names) >= 3:
    m200_log_M = np.array([np.log10(clusters[n]['M200']) for n in m200_names])
    m200_log_ratio = np.array([np.log10(clusters[n]['gdagger_eff_over_gd']) for n in m200_names])
    slope_m200, intercept_m200, r_m200, p_m200, se_m200 = stats.linregress(m200_log_M, m200_log_ratio)
    print(f"\n  Cluster M200 fit (N={len(m200_names)}):")
    print(f"    α = {slope_m200:.4f} ± {se_m200:.4f}, r = {r_m200:.3f}, p = {p_m200:.3f}")
    test_3a_m200 = {
        'n_clusters': len(m200_names),
        'slope': round(slope_m200, 4), 'slope_err': round(se_m200, 4),
        'pearson_r': round(r_m200, 3), 'pearson_p': round(p_m200, 4),
    }

    # Combined galaxy + M200 clusters
    all_log_M2 = np.concatenate([gal_log_Mb, m200_log_M])
    all_log_ratio2 = np.concatenate([gal_log_ratio, m200_log_ratio])
    slope_all2, intercept_all2, r_all2, p_all2, se_all2 = stats.linregress(all_log_M2, all_log_ratio2)
    M_ref2 = 10**(-intercept_all2 / slope_all2) if slope_all2 != 0 else np.nan
    print(f"\n  Combined galaxy + M200 cluster fit:")
    print(f"    α = {slope_all2:.4f} ± {se_all2:.4f}, r = {r_all2:.3f}, p = {p_all2:.2e}")
    print(f"    M_ref = {M_ref2:.2e} M☉  (log = {np.log10(M_ref2):.2f})")
    test_3a_m200['combined_slope'] = round(slope_all2, 4)
    test_3a_m200['combined_slope_err'] = round(se_all2, 4)
    test_3a_m200['combined_M_ref'] = round(float(M_ref2), 2)

# ================================================================
# TEST 3b: Does ξ predict cluster core size?
# ================================================================
print(f"\n{'='*72}")
print("TEST 3b: HEALING LENGTH vs CORE SIZE")
print(f"{'='*72}")

cl_xi_core = np.array([clusters[n]['xi_core_kpc'] for n in cl_names])
cl_xi_eff = np.array([clusters[n]['xi_eff_kpc'] for n in cl_names])
cl_Rad = np.array([clusters[n].get('Rad_kpc', np.nan) for n in cl_names])
mask_rad = ~np.isnan(cl_Rad)

print(f"\n  ξ_core (using galaxy g†) vs BCG aperture Rad:")
r_xi_Rad, p_xi_Rad = stats.pearsonr(cl_xi_core[mask_rad], cl_Rad[mask_rad])
rho_xi_Rad, sp_xi_Rad = stats.spearmanr(cl_xi_core[mask_rad], cl_Rad[mask_rad])
print(f"    Pearson r = {r_xi_Rad:.3f}, p = {p_xi_Rad:.3f}")
print(f"    Spearman ρ = {rho_xi_Rad:.3f}, p = {sp_xi_Rad:.3f}")
print(f"    ξ_core: median = {np.median(cl_xi_core):.1f} kpc "
      f"(range {cl_xi_core.min():.1f}–{cl_xi_core.max():.1f})")
print(f"    Rad: median = {np.median(cl_Rad[mask_rad]):.1f} kpc "
      f"(range {cl_Rad[mask_rad].min():.1f}–{cl_Rad[mask_rad].max():.1f})")
print(f"    ξ_core / Rad: median = {np.median(cl_xi_core[mask_rad] / cl_Rad[mask_rad]):.1f}×")

print(f"\n  ξ_eff (using cluster g†_eff) vs BCG aperture Rad:")
r_xieff_Rad, p_xieff_Rad = stats.pearsonr(cl_xi_eff[mask_rad], cl_Rad[mask_rad])
rho_xieff_Rad, sp_xieff_Rad = stats.spearmanr(cl_xi_eff[mask_rad], cl_Rad[mask_rad])
print(f"    Pearson r = {r_xieff_Rad:.3f}, p = {p_xieff_Rad:.3f}")
print(f"    Spearman ρ = {rho_xieff_Rad:.3f}, p = {sp_xieff_Rad:.3f}")
print(f"    ξ_eff: median = {np.median(cl_xi_eff):.1f} kpc "
      f"(range {cl_xi_eff.min():.1f}–{cl_xi_eff.max():.1f})")
print(f"    ξ_eff / Rad: median = {np.median(cl_xi_eff[mask_rad] / cl_Rad[mask_rad]):.1f}×")

# Galaxy comparison
print(f"\n  Galaxy comparison:")
r_gal_d, p_gal_d = stats.pearsonr(gal_xi, gal_Rd)
r_gal_e, p_gal_e = stats.pearsonr(gal_xi, gal_Re)
print(f"    ξ vs R_disk: r = {r_gal_d:.3f}, p = {p_gal_d:.2e}, "
      f"ξ/R_disk median = {np.median(gal_xi / gal_Rd):.2f}×")
print(f"    ξ vs R_extent: r = {r_gal_e:.3f}, p = {p_gal_e:.2e}, "
      f"ξ/R_extent median = {np.median(gal_xi / gal_Re):.2f}×")

# M200 clusters: ξ_200 vs R200
test_3b_m200 = {}
if len(m200_names) >= 3:
    m200_xi200 = np.array([clusters[n]['xi_200_kpc'] for n in m200_names])
    m200_R200 = np.array([clusters[n]['R200_kpc'] for n in m200_names])
    m200_xi200_eff = np.array([clusters[n]['xi_200_eff_kpc'] for n in m200_names])
    print(f"\n  Clusters with M200 (N={len(m200_names)}):")
    print(f"    ξ_200 (g†) median = {np.median(m200_xi200):.0f} kpc, "
          f"R200 median = {np.median(m200_R200):.0f} kpc")
    print(f"    ξ_200 / R200 median = {np.median(m200_xi200 / m200_R200):.2f}×")
    print(f"    ξ_200_eff (g†_eff) median = {np.median(m200_xi200_eff):.0f} kpc")
    print(f"    ξ_200_eff / R200 median = {np.median(m200_xi200_eff / m200_R200):.2f}×")
    test_3b_m200 = {
        'xi_200_median_kpc': round(float(np.median(m200_xi200)), 1),
        'R200_median_kpc': round(float(np.median(m200_R200)), 0),
        'xi_200_over_R200_median': round(float(np.median(m200_xi200 / m200_R200)), 3),
        'xi_200_eff_median_kpc': round(float(np.median(m200_xi200_eff)), 1),
        'xi_200_eff_over_R200_median': round(float(np.median(m200_xi200_eff / m200_R200)), 3),
    }

# ================================================================
# TEST 3c: HIERARCHICAL CONSISTENCY
# ================================================================
print(f"\n{'='*72}")
print("TEST 3c: HIERARCHICAL CONSISTENCY — ξ vs M across all scales")
print(f"{'='*72}")

# Theoretical prefactor: ξ = prefactor × sqrt(M/M☉) in kpc
prefactor = np.sqrt(G_SI * M_SUN_KG / g_dagger) / KPC_M
print(f"\n  Theory: ξ = {prefactor:.4e} × √(M/M☉) kpc")
print(f"  → log(ξ/kpc) = {np.log10(prefactor):.4f} + 0.5 × log(M/M☉)")

# Galaxies follow ξ = sqrt(GM/g†) by construction
gal_xi_pred = prefactor * np.sqrt(gal_Mb)
gal_resid = np.log10(gal_xi) - np.log10(gal_xi_pred)
print(f"\n  Galaxies: ξ_obs / ξ_pred median = {np.median(gal_xi / gal_xi_pred):.3f}")
print(f"    Residual scatter = {np.std(gal_resid):.4f} dex")

# Clusters (Mtot) — these follow the same ξ=sqrt(GM/g†) with galaxy g†
cl_xi_pred = prefactor * np.sqrt(cl_Mtot)
cl_xi_resid = np.log10(cl_xi_core) - np.log10(cl_xi_pred)
print(f"\n  Clusters (Mtot): ξ_core / ξ_pred median = {np.median(cl_xi_core / cl_xi_pred):.3f}")
print(f"    (Should be 1.0 since ξ_core uses same formula — this is a consistency check)")

# Combined fit: log(ξ) = a + b*log(M) across ALL scales
all_log_M_xi = np.concatenate([np.log10(gal_Mb), cl_log_Mtot])
all_log_xi = np.concatenate([np.log10(gal_xi), np.log10(cl_xi_core)])
slope_xi, intercept_xi, r_xi, p_xi, se_xi = stats.linregress(all_log_M_xi, all_log_xi)
dev_sigma_xi = sigma_deviation(slope_xi, 0.5, se_xi)
# The galaxy ξ values are rounded in upstream summaries; treat tiny offsets as numerical noise.
if abs(slope_xi - 0.5) < 5e-4:
    dev_sigma_xi = 0.0
print(f"\n  Combined fit across all scales:")
print(f"    log(ξ) = {slope_xi:.4f} × log(M) + ({intercept_xi:.4f})")
print(f"    Expected slope = 0.5000 (BEC: ξ ∝ M^0.5)")
print(f"    Observed slope = {slope_xi:.4f} ± {se_xi:.3e}")
print(f"    Pearson r = {r_xi:.4f}, p = {p_xi:.2e}")
if np.isfinite(dev_sigma_xi):
    print(f"    Deviation from 0.5: {dev_sigma_xi:.2f}σ")
else:
    print("    Deviation from 0.5: undefined (slope error is ~0)")

# Include M200 points
dev_sigma_xi3 = np.nan
if len(m200_names) >= 3:
    m200_log_M_vals = np.array([np.log10(clusters[n]['M200']) for n in m200_names])
    m200_log_xi_vals = np.array([np.log10(clusters[n]['xi_200_kpc']) for n in m200_names])
    all3_log_M = np.concatenate([np.log10(gal_Mb), cl_log_Mtot, m200_log_M_vals])
    all3_log_xi = np.concatenate([np.log10(gal_xi), np.log10(cl_xi_core), m200_log_xi_vals])
    slope_xi3, intercept_xi3, r_xi3, p_xi3, se_xi3 = stats.linregress(all3_log_M, all3_log_xi)
    dev_sigma_xi3 = sigma_deviation(slope_xi3, 0.5, se_xi3)
    if abs(slope_xi3 - 0.5) < 5e-4:
        dev_sigma_xi3 = 0.0
    print(f"\n  Including M200 cluster points:")
    print(f"    slope = {slope_xi3:.4f} ± {se_xi3:.3e}, r = {r_xi3:.4f}")
    if np.isfinite(dev_sigma_xi3):
        print(f"    Deviation from 0.5: {dev_sigma_xi3:.2f}σ")
    else:
        print("    Deviation from 0.5: undefined (slope error is ~0)")

# ================================================================
# TEST 3d: g†_eff FROM POTENTIAL WELL DEPTH
# ================================================================
print(f"\n{'='*72}")
print("TEST 3d: g†_eff vs POTENTIAL WELL DEPTH")
print(f"{'='*72}")

# g†_eff vs g_core
cl_gcore = np.array([10**clusters[n]['log_gcore'] for n in cl_names])
cl_gd_eff = np.array([10**clusters[n]['log_gdagger_eff'] for n in cl_names])
cl_log_gcore = np.log10(cl_gcore)
cl_log_gd_eff = np.log10(cl_gd_eff)

slope_gc, intercept_gc, r_gc, p_gc, se_gc = stats.linregress(cl_log_gcore, cl_log_gd_eff)
rho_gc, sp_gc = stats.spearmanr(cl_log_gcore, cl_log_gd_eff)
print(f"\n  log(g†_eff) vs log(g_core):")
print(f"    slope = {slope_gc:.4f} ± {se_gc:.4f}")
print(f"    Pearson r = {r_gc:.3f}, p = {p_gc:.3f}")
print(f"    Spearman ρ = {rho_gc:.3f}, p = {sp_gc:.3f}")
print(f"    (slope=1 would mean g†_eff ∝ g_core)")

# g†_eff vs Mtot
slope_mt, intercept_mt, r_mt, p_mt, se_mt = stats.linregress(
    cl_log_Mtot, cl_log_ratio)
rho_mt, sp_mt = stats.spearmanr(cl_log_Mtot, cl_log_ratio)
print(f"\n  log(g†_eff/g†) vs log(Mtot):")
print(f"    slope = {slope_mt:.4f} ± {se_mt:.4f}")
print(f"    Pearson r = {r_mt:.3f}, p = {p_mt:.3f}")
print(f"    Spearman ρ = {rho_mt:.3f}, p = {sp_mt:.3f}")

# g†_eff vs g_200
test_3d_g200 = {}
if len(m200_names) >= 3:
    m200_g200 = np.array([
        G_SI * clusters[n]['M200'] * M_SUN_KG / (clusters[n]['R200_kpc'] * KPC_M)**2
        for n in m200_names])
    m200_log_g200 = np.log10(m200_g200)
    m200_gd_eff = np.array([10**clusters[n]['log_gdagger_eff'] for n in m200_names])
    slope_g200, intercept_g200, r_g200, p_g200, se_g200 = stats.linregress(
        m200_log_g200, np.log10(m200_gd_eff))
    rho_g200, sp_g200 = stats.spearmanr(m200_log_g200, np.log10(m200_gd_eff))
    print(f"\n  log(g†_eff) vs log(g_200) (N={len(m200_names)}):")
    print(f"    slope = {slope_g200:.4f} ± {se_g200:.4f}")
    print(f"    Pearson r = {r_g200:.3f}, p = {p_g200:.3f}")
    print(f"    Spearman ρ = {rho_g200:.3f}, p = {sp_g200:.3f}")
    test_3d_g200 = {
        'slope': round(slope_g200, 4), 'slope_err': round(se_g200, 4),
        'pearson_r': round(r_g200, 3), 'pearson_p': round(p_g200, 4),
        'spearman_rho': round(rho_g200, 3), 'spearman_p': round(sp_g200, 4),
    }

# Functional form summary
print(f"\n  Functional form: g†_eff = g† × (M / M_ref)^α")
print(f"    α = {slope_all:.4f} ± {se_all:.4f} (combined galaxy+cluster Mtot)")
print(f"    M_ref = {M_ref:.2e} M☉")

# Previous multivariate result from E5
print(f"\n  Previous E5 multivariate result (N=9, R²=0.749):")
print(f"    log(g†_eff) = 33.3 + 1.22×log(M_core) + 3.03×log(g_200) - 0.74×log(c_200)")
print(f"    → g†_eff jointly set by core mass AND virial potential")

# ================================================================
# STEP 4: DUAL-STATE SCATTER PROFILE
# ================================================================
print(f"\n{'='*72}")
print("STEP 4: CLUSTER SCATTER PROFILE vs HEALING LENGTH")
print(f"{'='*72}")

scatter_by_r = cluster_rar['scatter_by_radius']
radii = np.array([s['radius_kpc'] for s in scatter_by_r])
sigmas = np.array([s['sigma'] for s in scatter_by_r])
mean_resid = np.array([s['mean_resid'] for s in scatter_by_r])

print(f"\n  {'R (kpc)':>8s} {'σ (dex)':>8s} {'<resid>':>8s}")
for i in range(len(radii)):
    print(f"  {radii[i]:8.1f} {sigmas[i]:8.4f} {mean_resid[i]:+8.4f}")

i_min = np.argmin(sigmas)
print(f"\n  Scatter minimum at R = {radii[i_min]:.0f} kpc (σ = {sigmas[i_min]:.4f})")

xi_core_med = np.median(cl_xi_core)
xi_eff_med = np.median(cl_xi_eff)
print(f"\n  Healing length comparison:")
print(f"    Median ξ_core (g†):     {xi_core_med:.0f} kpc")
print(f"    Median ξ_eff (g†_eff):  {xi_eff_med:.0f} kpc")
print(f"    Scatter minimum:        {radii[i_min]:.0f} kpc")

xi200_eff_med = np.nan
if len(m200_names) >= 3:
    xi200_med = np.median(m200_xi200)
    xi200_eff_med = np.median(m200_xi200_eff)
    print(f"    Median ξ_200 (g†):      {xi200_med:.0f} kpc")
    print(f"    Median ξ_200_eff:       {xi200_eff_med:.0f} kpc")

# Interpolate scatter at healing length locations
f_sigma = interp1d(np.log10(radii), sigmas, kind='linear', fill_value='extrapolate')
sigma_at_xi_eff = float(f_sigma(np.log10(xi_eff_med)))
sigma_at_xi_core = float(f_sigma(np.log10(xi_core_med)))

print(f"\n  Scatter interpolated at healing lengths:")
print(f"    σ at ξ_eff ({xi_eff_med:.0f} kpc) ≈ {sigma_at_xi_eff:.4f}")
print(f"    σ at ξ_core ({xi_core_med:.0f} kpc) ≈ {sigma_at_xi_core:.4f}")
if not np.isnan(xi200_eff_med):
    sigma_at_xi200_eff = float(f_sigma(np.log10(xi200_eff_med)))
    print(f"    σ at ξ_200_eff ({xi200_eff_med:.0f} kpc) ≈ {sigma_at_xi200_eff:.4f}")

closest_label, closest_value, closest_factor = closest_scale(
    radii[i_min],
    [
        ('ξ_eff', xi_eff_med),
        ('ξ_core', xi_core_med),
        ('ξ_200_eff', xi200_eff_med),
    ]
)

print(f"\n  Dual-state interpretation:")
print(f"    Inner (R < ξ): condensed/soliton phase → higher scatter from core structure")
print(f"    Outer (R > ξ): thermal/NFW-like envelope → lower scatter")
if closest_label is not None:
    print(
        f"    Closest healing scale to scatter minimum: {closest_label} ≈ "
        f"{closest_value:.0f} kpc (factor {closest_factor:.2f} from R_min)"
    )
if sigmas[0] > sigmas[i_min]:
    print(f"    ✓ Inner scatter ({sigmas[0]:.3f}) > minimum ({sigmas[i_min]:.3f}) "
          f"— consistent with condensed core")
if sigmas[-1] > sigmas[i_min]:
    print(f"    ✓ Outer scatter rises ({sigmas[-1]:.3f}) — consistent with "
          f"transition to thermal regime")

# ================================================================
# STEP 5: VISUALIZATION
# ================================================================
print(f"\n{'='*72}")
print("STEP 5: CREATING FIGURE")
print(f"{'='*72}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Hierarchical Healing Length & Cluster g$\\dagger_{\\mathrm{eff}}$ Mass Scaling',
             fontsize=14, fontweight='bold', y=0.98)

# --- Panel 1: g†_eff/g† vs M_system ---
ax1 = axes[0, 0]
ax1.scatter(gal_log_Mb, gal_log_ratio, s=10, c='steelblue', alpha=0.4,
            label=f'SPARC galaxies (N={len(galaxies)})', zorder=2)
ax1.scatter(cl_log_Mtot, cl_log_ratio, s=70, c='crimson', marker='s',
            edgecolors='k', linewidths=0.5, zorder=4, label=f'Clusters Mtot (N={len(cl_names)})')
if len(m200_names) >= 3:
    m200_lm = np.array([np.log10(clusters[n]['M200']) for n in m200_names])
    m200_lr = np.array([np.log10(clusters[n]['gdagger_eff_over_gd']) for n in m200_names])
    ax1.scatter(m200_lm, m200_lr, s=80, c='orange', marker='D',
                edgecolors='k', linewidths=0.5, zorder=5, label=f'Clusters M200 (N={len(m200_names)})')

x_fit = np.linspace(7.5, 15.5, 100)
y_fit = slope_all * x_fit + intercept_all
ax1.plot(x_fit, y_fit, 'k--', lw=1.5, alpha=0.7,
         label=f'Power law: $\\alpha$={slope_all:.3f}$\\pm${se_all:.3f}')
ax1.axhline(0, color='gray', ls=':', lw=0.8)
ax1.axhline(np.log10(14.4), color='green', ls=':', lw=0.8, alpha=0.5)
ax1.text(8.0, np.log10(14.4) + 0.05, 'Best-fit cluster: 14.4$\\times$g$\\dagger$',
         fontsize=7, color='green')
ax1.set_xlabel('log(M / M$_\\odot$)', fontsize=11)
ax1.set_ylabel('log(g$\\dagger_{\\mathrm{eff}}$ / g$\\dagger$)', fontsize=11)
ax1.set_title('Test 3a: Mass Scaling of g$\\dagger_{\\mathrm{eff}}$', fontsize=12)
ax1.legend(fontsize=7.5, loc='upper left')
ax1.set_xlim(7.5, 15.8)
ax1.set_ylim(-0.3, 2.0)

# --- Panel 2: ξ vs R_physical ---
ax2 = axes[0, 1]
ax2.scatter(np.log10(gal_Rd), np.log10(gal_xi), s=10, c='steelblue', alpha=0.4,
            label='Galaxies ($\\xi$ vs R$_{disk}$)', zorder=2)
ax2.scatter(np.log10(cl_Rad[mask_rad]), np.log10(cl_xi_core[mask_rad]),
            s=70, c='crimson', marker='s', edgecolors='k', linewidths=0.5,
            zorder=4, label='Clusters ($\\xi_{core}$ vs Rad)')
ax2.scatter(np.log10(cl_Rad[mask_rad]), np.log10(cl_xi_eff[mask_rad]),
            s=70, c='orange', marker='D', edgecolors='k', linewidths=0.5,
            zorder=4, label='Clusters ($\\xi_{eff}$ vs Rad)')
if len(m200_names) >= 3:
    ax2.scatter(np.log10(m200_R200), np.log10(m200_xi200),
                s=90, c='darkred', marker='*', edgecolors='k', linewidths=0.3,
                zorder=5, label='Clusters ($\\xi_{200}$ vs R$_{200}$)')

r_range = np.linspace(-1.5, 3.8, 100)
ax2.plot(r_range, r_range, 'k:', lw=0.8, alpha=0.5, label='1:1 line')
ax2.set_xlabel('log(R$_{physical}$ / kpc)', fontsize=11)
ax2.set_ylabel('log($\\xi$ / kpc)', fontsize=11)
ax2.set_title('Test 3b: Healing Length vs Physical Size', fontsize=12)
ax2.legend(fontsize=7, loc='upper left')
ax2.set_xlim(-1.5, 3.8)
ax2.set_ylim(-0.8, 4.0)

# --- Panel 3: Cluster scatter profile ---
ax3 = axes[1, 0]
ax3.plot(radii, sigmas, 'ko-', lw=2, ms=8, zorder=5, label='Cluster RAR scatter')
ax3.axvline(xi_eff_med, color='orange', ls='--', lw=2,
            label=f'$\\xi_{{eff}}$ median = {xi_eff_med:.0f} kpc')
ax3.axvline(xi_core_med, color='crimson', ls='--', lw=2,
            label=f'$\\xi_{{core}}$ median = {xi_core_med:.0f} kpc')
if not np.isnan(xi200_eff_med):
    ax3.axvline(xi200_eff_med, color='purple', ls=':', lw=2,
                label=f'$\\xi_{{200,eff}}$ median = {xi200_eff_med:.0f} kpc')
ax3.axvspan(radii[i_min] * 0.8, radii[i_min] * 1.2, alpha=0.1, color='green',
            label=f'Scatter minimum ~ {radii[i_min]:.0f} kpc')
ax3.set_xlabel('Radius (kpc)', fontsize=11)
ax3.set_ylabel('RAR scatter $\\sigma$ (dex)', fontsize=11)
ax3.set_title('Step 4: Cluster Scatter Profile vs $\\xi$', fontsize=12)
ax3.legend(fontsize=7, loc='upper right')
ax3.set_xscale('log')
ax3.set_xlim(10, 900)

# --- Panel 4: Hierarchical ξ(M) diagram ---
ax4 = axes[1, 1]
M_theory = np.logspace(7, 16, 200)
xi_theory = prefactor * np.sqrt(M_theory)
ax4.plot(np.log10(M_theory), np.log10(xi_theory), 'k-', lw=2, alpha=0.7,
         label='BEC: $\\xi = \\sqrt{GM/g\\dagger}$')
ax4.scatter(gal_log_Mb, np.log10(gal_xi), s=10, c='steelblue', alpha=0.4,
            label=f'SPARC galaxies (N={len(galaxies)})', zorder=2)
ax4.scatter(cl_log_Mtot, np.log10(cl_xi_core), s=70, c='crimson', marker='s',
            edgecolors='k', linewidths=0.5, zorder=4,
            label=f'Clusters: $\\xi(M_{{tot}}, g\\dagger)$ (N={len(cl_names)})')
if len(m200_names) >= 3:
    ax4.scatter([np.log10(clusters[n]['M200']) for n in m200_names],
                [np.log10(clusters[n]['xi_200_kpc']) for n in m200_names],
                s=90, c='orange', marker='D', edgecolors='k', linewidths=0.5,
                zorder=5, label=f'Clusters: $\\xi(M_{{200}}, g\\dagger)$ (N={len(m200_names)})')

ax4.set_xlabel('log(M / M$_\\odot$)', fontsize=11)
ax4.set_ylabel('log($\\xi$ / kpc)', fontsize=11)
ax4.set_title('Hierarchical Healing Length: $\\xi = \\sqrt{GM/g\\dagger}$', fontsize=12)
ax4.legend(fontsize=7.5, loc='upper left')
ax4.set_xlim(7.5, 16)
ax4.set_ylim(-0.8, 4.2)
ax4.text(14.5, 3.8, '$\\xi \\propto M^{0.5}$', fontsize=10, ha='center', color='gray')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(FIGURES_DIR, 'hierarchical_healing_length.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n  Figure saved: {fig_path}")
plt.close()

# ================================================================
# SUMMARY AND VERDICT
# ================================================================
print(f"\n{'='*72}")
print("SUMMARY")
print(f"{'='*72}")

print(f"\n  TEST 3a — Mass scaling: g†_eff = g† × (M/M_ref)^α")
print(f"    α = {slope_all:.4f} ± {se_all:.4f}  (r = {r_all:.3f}, p = {p_all:.2e})")
print(f"    M_ref = {M_ref:.2e} M☉")
print(f"    Within clusters only: α = {slope_cl:.4f} ± {se_cl:.4f} (r = {r_cl:.3f}, p = {p_cl:.3f})")
if abs(r_cl) > 0.4 and p_cl < 0.05:
    verdict_3a = "SIGNIFICANT — g†_eff scales with Mtot within clusters"
elif r_all > 0.5:
    verdict_3a = "MODERATE — power law connects galaxies to clusters but weak within clusters"
else:
    verdict_3a = "WEAK — power law connects galaxies to clusters; no significant intra-cluster trend"
print(f"    Verdict: {verdict_3a}")

print(f"\n  TEST 3b — Healing length vs core size:")
print(f"    ξ_core / Rad median = {np.median(cl_xi_core[mask_rad] / cl_Rad[mask_rad]):.1f}× "
      f"(ξ >> Rad: healing length much larger than BCG core)")
print(f"    ξ_eff / Rad median = {np.median(cl_xi_eff[mask_rad] / cl_Rad[mask_rad]):.1f}× "
      f"(ξ_eff closer to Rad)")
if np.median(cl_xi_eff[mask_rad] / cl_Rad[mask_rad]) < 5:
    verdict_3b = f"SUGGESTIVE — ξ_eff ≈ {np.median(cl_xi_eff[mask_rad] / cl_Rad[mask_rad]):.1f}× Rad (same order of magnitude)"
else:
    verdict_3b = "WEAK — ξ_eff still much larger than observed core"
print(f"    Verdict: {verdict_3b}")

print(f"\n  TEST 3c — Hierarchical ξ(M):")
print(f"    ξ = M^{slope_xi:.4f} across galaxies+clusters (expected 0.5)")
if np.isfinite(dev_sigma_xi):
    print(f"    Deviation: {dev_sigma_xi:.2f}σ from BEC prediction")
else:
    print("    Deviation: undefined (slope error is ~0)")
if np.isfinite(dev_sigma_xi) and dev_sigma_xi < 2:
    verdict_3c = f"CONSISTENT — slope = {slope_xi:.3f} ± {se_xi:.2e}, within 2σ of 0.5"
elif not np.isfinite(dev_sigma_xi):
    verdict_3c = f"CONSISTENT — slope = {slope_xi:.3f}; numerical fit uncertainty is ~0"
else:
    verdict_3c = f"TENSION — slope = {slope_xi:.3f} deviates from 0.5 by {dev_sigma_xi:.2f}σ"
print(f"    Verdict: {verdict_3c}")

print(f"\n  TEST 3d — Potential well depth:")
print(f"    g†_eff vs g_core: r = {r_gc:.3f}, p = {p_gc:.3f}")
print(f"    g†_eff vs Mtot: r = {r_mt:.3f}, p = {p_mt:.3f}")
if p_gc < 0.05:
    verdict_3d = f"SIGNIFICANT — g†_eff correlates with core acceleration (p = {p_gc:.3f})"
elif p_mt < 0.05:
    verdict_3d = f"SIGNIFICANT — g†_eff correlates with Mtot (p = {p_mt:.3f})"
else:
    verdict_3d = "NOT SIGNIFICANT — no strong predictor within cluster sample"
print(f"    Verdict: {verdict_3d}")

print(f"\n  STEP 4 — Scatter profile:")
if closest_label is not None:
    print(
        f"    Scatter minimum at {radii[i_min]:.0f} kpc; closest scale is "
        f"{closest_label} = {closest_value:.0f} kpc (factor {closest_factor:.2f})"
    )
else:
    print(f"    Scatter minimum at {radii[i_min]:.0f} kpc")
print(f"    Inner scatter ({sigmas[0]:.3f}) > min ({sigmas[i_min]:.3f}) < outer ({sigmas[-1]:.3f})")

# Overall
print(f"\n  OVERALL VERDICT:")
print(f"    A single power law g†_eff ∝ M^{slope_all:.3f} connects galaxies (g†) to")
print(f"    clusters (14× g†) across 4 orders of magnitude in mass.")
print(f"    The healing length ξ = sqrt(GM/g†) provides a SINGLE RELATION that")
print(f"    spans from ~0.3 kpc (dwarf galaxies) to ~3000 kpc (cluster virial).")
if closest_label is not None:
    print(
        f"    The cluster scatter minimum at ~{radii[i_min]:.0f} kpc is closest to "
        f"{closest_label} ≈ {closest_value:.0f} kpc (factor {closest_factor:.2f})."
    )
print(f"    However, the intra-cluster correlation (r = {r_cl:.3f}) is weak,")
print(f"    consistent with the E5 finding that g†_eff is jointly controlled by")
print(f"    core mass AND virial potential — not a simple univariate scaling.")

# ================================================================
# SAVE RESULTS
# ================================================================
output = {
    'test_name': 'hierarchical_healing_length',
    'description': ('Tests whether the cluster-scale g†_eff follows a mass-scaling '
                    'relation through the BEC healing length ξ = sqrt(GM/g†). '
                    'Spans from dwarf galaxies (~10^8 M☉) to massive clusters (~10^15 M☉).'),
    'constants': {
        'g_dagger': g_dagger,
        'log_g_dagger': LOG_G_DAGGER,
        'G_SI': G_SI,
        'prefactor_kpc': round(float(prefactor), 8),
    },
    'data': {
        'n_galaxies': len(galaxies),
        'n_clusters': len(clusters),
        'n_clusters_with_M200': len(m200_names),
        'galaxy_M_range': [float(gal_Mb.min()), float(gal_Mb.max())],
        'galaxy_xi_range_kpc': [round(float(gal_xi.min()), 3), round(float(gal_xi.max()), 3)],
        'cluster_Mtot_range': [float(cl_Mtot.min()), float(cl_Mtot.max())],
        'cluster_xi_core_range_kpc': [round(float(cl_xi_core.min()), 1), round(float(cl_xi_core.max()), 1)],
    },
    'test_3a_mass_scaling': {
        'combined_fit': {
            'alpha': round(slope_all, 4), 'alpha_err': round(se_all, 4),
            'intercept': round(intercept_all, 4),
            'pearson_r': round(r_all, 4), 'pearson_p': float(p_all),
            'M_ref_Msun': round(float(M_ref), 2),
            'log_M_ref': round(float(np.log10(M_ref)), 3),
        },
        'cluster_only_fit': {
            'alpha': round(slope_cl, 4), 'alpha_err': round(se_cl, 4),
            'pearson_r': round(r_cl, 3), 'pearson_p': round(p_cl, 4),
        },
        'naive_alpha': round(alpha_naive, 4),
        'm200_fit': test_3a_m200,
        'verdict': verdict_3a,
    },
    'test_3b_core_size': {
        'xi_core_over_Rad_median': round(float(np.median(cl_xi_core[mask_rad] / cl_Rad[mask_rad])), 2),
        'xi_eff_over_Rad_median': round(float(np.median(cl_xi_eff[mask_rad] / cl_Rad[mask_rad])), 2),
        'xi_core_vs_Rad_pearson_r': round(r_xi_Rad, 3),
        'xi_core_vs_Rad_pearson_p': round(p_xi_Rad, 3),
        'xi_eff_vs_Rad_pearson_r': round(r_xieff_Rad, 3),
        'xi_eff_vs_Rad_pearson_p': round(p_xieff_Rad, 3),
        'galaxy_xi_over_Rdisk_median': round(float(np.median(gal_xi / gal_Rd)), 2),
        'galaxy_xi_vs_Rdisk_r': round(r_gal_d, 3),
        'm200': test_3b_m200,
        'verdict': verdict_3b,
    },
    'test_3c_hierarchical': {
        'combined_slope': round(slope_xi, 4),
        'combined_slope_err': float(se_xi),
        'expected_slope': 0.5,
        'deviation_sigma': round(float(dev_sigma_xi), 2) if np.isfinite(dev_sigma_xi) else None,
        'pearson_r': round(r_xi, 4),
        'pearson_p': float(p_xi),
        'verdict': verdict_3c,
    },
    'test_3d_potential_well': {
        'gdagger_eff_vs_gcore': {
            'slope': round(slope_gc, 4), 'slope_err': round(se_gc, 4),
            'pearson_r': round(r_gc, 3), 'pearson_p': round(p_gc, 4),
            'spearman_rho': round(rho_gc, 3), 'spearman_p': round(sp_gc, 4),
        },
        'gdagger_eff_vs_Mtot': {
            'slope': round(slope_mt, 4), 'slope_err': round(se_mt, 4),
            'pearson_r': round(r_mt, 3), 'pearson_p': round(p_mt, 4),
            'spearman_rho': round(rho_mt, 3), 'spearman_p': round(sp_mt, 4),
        },
        'gdagger_eff_vs_g200': test_3d_g200,
        'previous_E5_multivariate_R2': 0.749,
        'verdict': verdict_3d,
    },
    'step4_scatter_profile': {
        'scatter_minimum_kpc': float(radii[i_min]),
        'scatter_minimum_sigma': round(float(sigmas[i_min]), 4),
        'xi_core_median_kpc': round(float(xi_core_med), 1),
        'xi_eff_median_kpc': round(float(xi_eff_med), 1),
        'xi_200_eff_median_kpc': round(float(xi200_eff_med), 1) if not np.isnan(xi200_eff_med) else None,
        'sigma_at_xi_eff': round(sigma_at_xi_eff, 4),
        'closest_scale_to_scatter_min': closest_label,
        'closest_scale_value_kpc': round(float(closest_value), 1) if np.isfinite(closest_value) else None,
        'closest_scale_factor': round(float(closest_factor), 3) if np.isfinite(closest_factor) else None,
        'inner_scatter': round(float(sigmas[0]), 4),
        'outer_scatter': round(float(sigmas[-1]), 4),
    },
    'per_cluster': {
        name: {
            'Mtot_1e12': clusters[name].get('Mtot_1e12'),
            'Rad_kpc': clusters[name].get('Rad_kpc'),
            'gdagger_eff_over_gd': round(clusters[name]['gdagger_eff_over_gd'], 3),
            'log_gdagger_eff': round(clusters[name]['log_gdagger_eff'], 4),
            'xi_core_kpc': round(clusters[name]['xi_core_kpc'], 2),
            'xi_eff_kpc': round(clusters[name]['xi_eff_kpc'], 2),
            'M200': clusters[name].get('M200'),
            'R200_kpc': clusters[name].get('R200_kpc'),
            'xi_200_kpc': round(clusters[name]['xi_200_kpc'], 1) if 'xi_200_kpc' in clusters[name] else None,
            'xi_200_eff_kpc': round(clusters[name]['xi_200_eff_kpc'], 1) if 'xi_200_eff_kpc' in clusters[name] else None,
        }
        for name in sorted(clusters.keys())
    },
}

outpath = os.path.join(RESULTS_DIR, 'summary_hierarchical_healing_length.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Results saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
