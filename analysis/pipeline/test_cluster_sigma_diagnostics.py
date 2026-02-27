#!/usr/bin/env python3
"""
Test A Diagnostics + Test A2: Alternative Cluster-Scale Predictors
===================================================================

Check 1: Dynamic range in σ and heterogeneous systematics
Check 2: Is per-cluster g‡ actually varying? (intrinsic vs measurement scatter)
Test A2: Alternative predictors — g_200, c_200, BCG M*, baryon fraction

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'cluster_rar')

g_dagger = 1.20e-10
G_pc = 4.302e-3              # G in pc (km/s)² / M_sun
G_SI = 6.674e-11             # G in m³ / (kg s²)
M_sun = 1.989e30             # kg
Mpc_m = 3.086e22             # metres per Mpc
kpc_m = 3.086e19             # metres per kpc

print("=" * 72)
print("TEST A DIAGNOSTICS + TEST A2: ALTERNATIVE PREDICTORS")
print("=" * 72)

# ================================================================
# LOAD DATA
# ================================================================

# Per-cluster g‡ from E2
e2_path = os.path.join(RESULTS_DIR, 'summary_cluster_rar_tian2020.json')
with open(e2_path) as f:
    e2 = json.load(f)

cluster_gdagger = {}
for entry in e2['per_cluster_a0']:
    cluster_gdagger[entry['name']] = {
        'log_gdagger': entry['log_a0'],
        'gdagger': 10**entry['log_a0'],
        'scatter': entry['scatter'],
        'n_pts': entry['n_pts'],
    }

# Test A results
a_path = os.path.join(RESULTS_DIR, 'summary_cluster_sigma_scaling.json')
with open(a_path) as f:
    test_a = json.load(f)

# Parse Tian+2020 Table 1 for BCG properties
print("\n[0] Parsing Tian+2020 Table 1 for BCG properties...")
table1_path = os.path.join(DATA_DIR, 'tian2020_table1.dat')
bcg_props = {}
with open(table1_path, 'r') as f:
    for line in f:
        line_s = line.strip()
        if not line_s or line_s.startswith('#') or line_s.startswith('-') or 'Name' in line_s:
            continue
        parts = line_s.split('|')
        if len(parts) < 14:
            continue
        try:
            aname = parts[-1].strip()
            if not aname or not aname[0].isalpha():
                continue
            z_str = parts[1].strip()
            z = float(z_str)
            # M* is field index -6, Mgas is -5, e_Mgas is -4, Mtot is -3, e_Mtot is -2
            re_str = parts[5].strip() if len(parts) > 5 else ''  # Re (effective radius)
            rad_str = parts[7].strip() if len(parts) > 7 else ''  # Rad (aperture radius)
            mstar_str = parts[-6].strip()  # M* in 10^12 Msun
            mgas_str = parts[-5].strip()
            mtot_str = parts[-3].strip()   # Mtot in 10^12 Msun

            mstar = float(mstar_str) if mstar_str else None
            mgas = float(mgas_str) if mgas_str else None
            mtot = float(mtot_str) if mtot_str else None
            rad = float(rad_str) if rad_str else None  # kpc — aperture where Mtot measured
            re = float(re_str) if re_str else None     # kpc — BCG effective radius

            bcg_props[aname] = {
                'z': z, 'Mstar_1e12': mstar, 'Mgas_1e12': mgas,
                'Mtot_1e12': mtot, 'Rad_kpc': rad, 'Re_kpc': re,
            }
        except (ValueError, IndexError):
            continue

print(f"  Parsed BCG properties for {len(bcg_props)} clusters")

# M200/r200 data from Pizzardo+2025 + Adam+2022
m200_data = {
    'A209':    {'m200': 17.3e14, 'r200': 2.31, 'c200': 3.4, 'e_c200': 0.7, 'src': 'Pizzardo+2025'},
    'A383':    {'m200': 8.4e14,  'r200': 1.83, 'c200': 2.5, 'e_c200': 1.6, 'src': 'Pizzardo+2025'},
    'MACS0329':{'m200': 11.5e14, 'r200': 1.84, 'c200': 5.4, 'e_c200': 1.3, 'src': 'Pizzardo+2025'},
    'MACS1115':{'m200': 10.7e14, 'r200': 1.87, 'c200': 2.5, 'e_c200': 0.7, 'src': 'Pizzardo+2025'},
    'MACS1206':{'m200': 15.9e14, 'r200': 2.06, 'c200': 5.8, 'e_c200': 1.7, 'src': 'Pizzardo+2025'},
    'MACS1931':{'m200': 11.5e14, 'r200': 1.91, 'c200': 7.8, 'e_c200': 1.7, 'src': 'Pizzardo+2025'},
    'MS2137':  {'m200': 7.9e14,  'r200': 1.70, 'c200': 2.4, 'e_c200': 1.0, 'src': 'Pizzardo+2025'},
    'RXJ2129': {'m200': 7.7e14,  'r200': 1.75, 'c200': 2.9, 'e_c200': 1.2, 'src': 'Pizzardo+2025'},
    'RXJ2248': {'m200': 22.7e14, 'r200': 2.40, 'c200': 1.6, 'e_c200': 0.7, 'src': 'Pizzardo+2025'},
    'MACS0647':{'m200': 18.1e14, 'r200': 2.06, 'c200': None, 'e_c200': None, 'src': 'Adam+2022'},
}

# σ data from Test A
sigma_data = {}
for cl in test_a['clusters']:
    sigma_data[cl['name']] = {
        'sigma': cl['sigma_kms'],
        'e_sigma': cl['e_sigma'],
        'method': cl['sigma_method'],
        'flag': cl['flag'],
    }

# ================================================================
# CHECK 1: Dynamic range in σ and systematics
# ================================================================
print("\n" + "=" * 72)
print("CHECK 1: Dynamic range in σ and heterogeneous systematics")
print("=" * 72)

all_sigma = np.array([sigma_data[n]['sigma'] for n in sigma_data])
all_e_sigma = np.array([sigma_data[n]['e_sigma'] for n in sigma_data])
names_all = list(sigma_data.keys())

print(f"\n  N = {len(all_sigma)} clusters")
print(f"  σ range: {all_sigma.min():.0f} – {all_sigma.max():.0f} km/s")
print(f"  log(σ²) range: {np.log10(all_sigma.min()**2):.3f} – {np.log10(all_sigma.max()**2):.3f}")
print(f"  Dynamic range in log(σ²): {np.log10(all_sigma.max()**2) - np.log10(all_sigma.min()**2):.3f} dex")
print(f"  Typical σ uncertainty: {np.median(all_e_sigma):.0f} km/s ({np.median(all_e_sigma/all_sigma)*100:.1f}%)")

# Spectroscopic-only subset
spec_names = [n for n in sigma_data if sigma_data[n]['method'] == 'spectroscopic']
spec_sigma = np.array([sigma_data[n]['sigma'] for n in spec_names])
spec_log_gd = np.array([cluster_gdagger[n]['log_gdagger'] for n in spec_names])
spec_log_s2 = np.log10(spec_sigma**2)
spec_flags = [sigma_data[n]['flag'] for n in spec_names]

# Clean spectroscopic (no flags)
clean_spec = [(n, sigma_data[n]['sigma'], cluster_gdagger[n]['log_gdagger'])
              for n in spec_names if sigma_data[n]['flag'] is None]

print(f"\n  --- Spectroscopic-only refit (N={len(spec_names)}) ---")
sl, ic, rv, pv, se = linregress(spec_log_s2, spec_log_gd)
rp, pp = pearsonr(spec_log_s2, spec_log_gd)
print(f"  B = {sl:.4f} ± {se:.4f}")
print(f"  Pearson r = {rp:+.4f}, p = {pp:.4e}")
print(f"  σ range (spec only): {spec_sigma.min():.0f} – {spec_sigma.max():.0f} km/s")

if len(clean_spec) >= 4:
    cs_sigma = np.array([x[1] for x in clean_spec])
    cs_gd = np.array([x[2] for x in clean_spec])
    cs_ls2 = np.log10(cs_sigma**2)
    sl2, ic2, rv2, pv2, se2 = linregress(cs_ls2, cs_gd)
    rp2, pp2 = pearsonr(cs_ls2, cs_gd)
    print(f"\n  --- Clean spectroscopic only (N={len(clean_spec)}, no flags) ---")
    print(f"  B = {sl2:.4f} ± {se2:.4f}")
    print(f"  Pearson r = {rp2:+.4f}, p = {pp2:.4e}")
    print(f"  σ range: {cs_sigma.min():.0f} – {cs_sigma.max():.0f} km/s")

# ================================================================
# CHECK 2: Is per-cluster g‡ actually varying?
# ================================================================
print("\n" + "=" * 72)
print("CHECK 2: Is per-cluster g‡ actually varying?")
print("=" * 72)

# All 20 clusters have g‡
all_log_gd = np.array([cluster_gdagger[n]['log_gdagger'] for n in sorted(cluster_gdagger)])
all_scatter = np.array([cluster_gdagger[n]['scatter'] for n in sorted(cluster_gdagger)])
all_npts = np.array([cluster_gdagger[n]['n_pts'] for n in sorted(cluster_gdagger)])

print(f"\n  All 20 clusters:")
print(f"    Mean log g‡ = {np.mean(all_log_gd):.4f}")
print(f"    Std log g‡  = {np.std(all_log_gd):.4f} dex (observed spread)")
print(f"    Range: [{np.min(all_log_gd):.3f}, {np.max(all_log_gd):.3f}] = {np.max(all_log_gd)-np.min(all_log_gd):.3f} dex")
print(f"    Median per-cluster RAR scatter = {np.median(all_scatter):.4f} dex")
print(f"    Median N_pts per cluster = {np.median(all_npts):.0f}")

# Estimate measurement uncertainty on log g‡ from RAR scatter and N_pts
# σ(log g‡) ~ RAR_scatter / sqrt(N_pts)
est_meas_err = all_scatter / np.sqrt(all_npts)
print(f"    Estimated σ(log g‡) per cluster ≈ RAR_scatter/√N")
print(f"      Range: {est_meas_err.min():.4f} – {est_meas_err.max():.4f} dex")
print(f"      Median: {np.median(est_meas_err):.4f} dex")

# Hierarchical variance decomposition
# observed_var = intrinsic_var + mean(measurement_var)
obs_var = np.var(all_log_gd)
meas_var = np.mean(est_meas_err**2)
intrinsic_var = max(0, obs_var - meas_var)
print(f"\n  Variance decomposition:")
print(f"    Observed variance:  {obs_var:.6f} (σ_obs = {np.sqrt(obs_var):.4f} dex)")
print(f"    Mean meas variance: {meas_var:.6f} (σ_meas = {np.sqrt(meas_var):.4f} dex)")
print(f"    Intrinsic variance: {intrinsic_var:.6f} (σ_int = {np.sqrt(intrinsic_var):.4f} dex)")
intrinsic_frac = intrinsic_var / obs_var if obs_var > 0 else 0
print(f"    Intrinsic fraction: {intrinsic_frac:.1%}")

if intrinsic_var > meas_var:
    gd_verdict = "REAL VARIATION dominates — g‡ genuinely varies across clusters"
else:
    gd_verdict = "MEASUREMENT NOISE dominates — g‡ is consistent with universal"
print(f"    → {gd_verdict}")

# ================================================================
# TEST A2: ALTERNATIVE PREDICTORS
# ================================================================
print("\n" + "=" * 72)
print("TEST A2: ALTERNATIVE CLUSTER-SCALE PREDICTORS")
print("=" * 72)

def fit_and_report(x, y, names, x_label, n_label):
    """Fit log(g‡) = A + B × x, report stats."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, names = x[mask], y[mask], [names[i] for i in range(len(names)) if mask[i]]
    if len(x) < 4:
        print(f"    Only {len(x)} points — insufficient for fit")
        return None
    sl, ic, rv, pv, se = linregress(x, y)
    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)
    print(f"    N = {len(x)}")
    print(f"    B (slope)  = {sl:.4f} ± {se:.4f}")
    print(f"    Pearson  r = {rp:+.4f}, p = {pp:.4e}")
    print(f"    Spearman ρ = {rs:+.4f}, p = {ps:.4e}")
    if pp < 0.01:
        sig = "*** HIGHLY SIGNIFICANT ***"
    elif pp < 0.05:
        sig = "** SIGNIFICANT **"
    elif pp < 0.10:
        sig = "* marginally significant *"
    else:
        sig = "(not significant)"
    print(f"    {sig}")
    return {
        'predictor': n_label,
        'n': len(x),
        'slope': float(sl),
        'slope_err': float(se),
        'intercept': float(ic),
        'pearson_r': float(rp),
        'pearson_p': float(pp),
        'spearman_rho': float(rs),
        'spearman_p': float(ps),
    }

results_predictors = []

# --- Predictor 1: g_200 = G M200 / R200² ---
print(f"\n  --- Predictor 1: g_200 = G M200 / R200² (characteristic acceleration at R200) ---")

g200_names = []
g200_vals = []
g200_gd = []
for name in sorted(m200_data.keys()):
    if name in cluster_gdagger:
        d = m200_data[name]
        r200_m = d['r200'] * Mpc_m
        m200_kg = d['m200'] * M_sun
        g200 = G_SI * m200_kg / r200_m**2
        g200_names.append(name)
        g200_vals.append(g200)
        g200_gd.append(cluster_gdagger[name]['log_gdagger'])

g200_arr = np.array(g200_vals)
log_g200 = np.log10(g200_arr)
gd_arr = np.array(g200_gd)

print(f"    g_200 range: {g200_arr.min():.3e} – {g200_arr.max():.3e} m/s²")
print(f"    log g_200 range: {log_g200.min():.3f} – {log_g200.max():.3f} ({log_g200.max()-log_g200.min():.3f} dex)")

res = fit_and_report(log_g200, gd_arr, g200_names, 'log g_200', 'g_200 = GM200/R200²')
if res:
    results_predictors.append(res)

# --- Predictor 2: V200²/R200 (same as g_200, verify) ---
# V200 = sqrt(GM200/R200), so V200²/R200 = GM200/R200² = g200. Confirmed identical.
print(f"\n  (Predictor 2: V200²/R200 is algebraically identical to g_200 — skipped)")

# --- Predictor 3: c_200 (concentration) ---
print(f"\n  --- Predictor 3: c_200 (NFW concentration) ---")

c200_names = []
c200_vals = []
c200_gd = []
for name in sorted(m200_data.keys()):
    if name in cluster_gdagger and m200_data[name]['c200'] is not None:
        c200_names.append(name)
        c200_vals.append(m200_data[name]['c200'])
        c200_gd.append(cluster_gdagger[name]['log_gdagger'])

c200_arr = np.array(c200_vals)
c200_gd_arr = np.array(c200_gd)

print(f"    c_200 range: {c200_arr.min():.1f} – {c200_arr.max():.1f}")

res = fit_and_report(np.log10(c200_arr), c200_gd_arr, c200_names,
                     'log c_200', 'c_200 (NFW concentration)')
if res:
    results_predictors.append(res)

# --- Predictor 4: BCG stellar mass M* ---
print(f"\n  --- Predictor 4: BCG stellar mass M* (from Tian+2020 Table 1) ---")

mstar_names = []
mstar_vals = []
mstar_gd = []
for name in sorted(bcg_props.keys()):
    if name in cluster_gdagger and bcg_props[name]['Mstar_1e12'] is not None:
        mstar_names.append(name)
        mstar_vals.append(bcg_props[name]['Mstar_1e12'] * 1e12)  # -> Msun
        mstar_gd.append(cluster_gdagger[name]['log_gdagger'])

mstar_arr = np.array(mstar_vals)
mstar_gd_arr = np.array(mstar_gd)

print(f"    M* range: {mstar_arr.min():.2e} – {mstar_arr.max():.2e} M_sun")
print(f"    log M* range: {np.log10(mstar_arr.min()):.2f} – {np.log10(mstar_arr.max()):.2f}")

res = fit_and_report(np.log10(mstar_arr), mstar_gd_arr, mstar_names,
                     'log M*', 'BCG stellar mass M*')
if res:
    results_predictors.append(res)

# --- Predictor 5: BCG baryon fraction (M* + Mgas) / Mtot ---
print(f"\n  --- Predictor 5: BCG baryon fraction f_bar = (M* + Mgas) / Mtot ---")

fbar_names = []
fbar_vals = []
fbar_gd = []
for name in sorted(bcg_props.keys()):
    p = bcg_props[name]
    if (name in cluster_gdagger and p['Mstar_1e12'] is not None
            and p['Mgas_1e12'] is not None and p['Mtot_1e12'] is not None
            and p['Mtot_1e12'] > 0):
        fbar = (p['Mstar_1e12'] + p['Mgas_1e12']) / p['Mtot_1e12']
        fbar_names.append(name)
        fbar_vals.append(fbar)
        fbar_gd.append(cluster_gdagger[name]['log_gdagger'])

fbar_arr = np.array(fbar_vals)
fbar_gd_arr = np.array(fbar_gd)

print(f"    f_bar range: {fbar_arr.min():.3f} – {fbar_arr.max():.3f}")

res = fit_and_report(np.log10(fbar_arr), fbar_gd_arr, fbar_names,
                     'log f_bar', 'BCG baryon fraction')
if res:
    results_predictors.append(res)

# --- Predictor 6: BCG total mass Mtot (aperture) ---
print(f"\n  --- Predictor 6: BCG total (aperture) mass Mtot ---")

mtot_names = []
mtot_vals = []
mtot_gd = []
for name in sorted(bcg_props.keys()):
    if name in cluster_gdagger and bcg_props[name]['Mtot_1e12'] is not None:
        mtot_names.append(name)
        mtot_vals.append(bcg_props[name]['Mtot_1e12'] * 1e12)
        mtot_gd.append(cluster_gdagger[name]['log_gdagger'])

mtot_arr = np.array(mtot_vals)
mtot_gd_arr = np.array(mtot_gd)

print(f"    Mtot range: {mtot_arr.min():.2e} – {mtot_arr.max():.2e} M_sun")

res = fit_and_report(np.log10(mtot_arr), mtot_gd_arr, mtot_names,
                     'log Mtot', 'BCG aperture total mass')
if res:
    results_predictors.append(res)

# --- Predictor 7: M200 (cluster virial mass) ---
print(f"\n  --- Predictor 7: M200 (cluster virial mass) ---")

m200_names = []
m200_vals_arr = []
m200_gd_list = []
for name in sorted(m200_data.keys()):
    if name in cluster_gdagger:
        m200_names.append(name)
        m200_vals_arr.append(m200_data[name]['m200'])
        m200_gd_list.append(cluster_gdagger[name]['log_gdagger'])

m200_np = np.array(m200_vals_arr)
m200_gd_np = np.array(m200_gd_list)

print(f"    M200 range: {m200_np.min():.2e} – {m200_np.max():.2e} M_sun")

res = fit_and_report(np.log10(m200_np), m200_gd_np, m200_names,
                     'log M200', 'Cluster virial mass M200')
if res:
    results_predictors.append(res)

# --- Predictor 8: Redshift z ---
print(f"\n  --- Predictor 8: Redshift z (selection effect check) ---")

z_names = []
z_vals = []
z_gd = []
for name in sorted(bcg_props.keys()):
    if name in cluster_gdagger:
        z_names.append(name)
        z_vals.append(bcg_props[name]['z'])
        z_gd.append(cluster_gdagger[name]['log_gdagger'])

z_arr = np.array(z_vals)
z_gd_arr = np.array(z_gd)

res = fit_and_report(z_arr, z_gd_arr, z_names, 'z', 'Redshift z')
if res:
    results_predictors.append(res)

# --- Predictor 9: σ (from Test A, for comparison) ---
print(f"\n  --- Predictor 9: σ² (velocity dispersion, from Test A) ---")

sig_names = list(sigma_data.keys())
sig_sigma = np.array([sigma_data[n]['sigma'] for n in sig_names])
sig_gd = np.array([cluster_gdagger[n]['log_gdagger'] for n in sig_names])
sig_log_s2 = np.log10(sig_sigma**2)

res = fit_and_report(sig_log_s2, sig_gd, sig_names, 'log σ²', 'Velocity dispersion σ²')
if res:
    results_predictors.append(res)

# ================================================================
# PREDICTOR COMPARISON TABLE
# ================================================================
print("\n" + "=" * 72)
print("PREDICTOR COMPARISON (sorted by |Pearson r|)")
print("=" * 72)

results_predictors.sort(key=lambda x: abs(x['pearson_r']), reverse=True)

print(f"\n  {'Predictor':<30} {'N':>3} {'r':>7} {'p':>10} {'ρ':>7} {'B±σ_B':>14}")
print(f"  " + "-" * 75)
for r in results_predictors:
    sig = '***' if r['pearson_p'] < 0.01 else '**' if r['pearson_p'] < 0.05 else '*' if r['pearson_p'] < 0.10 else ''
    print(f"  {r['predictor']:<30} {r['n']:>3} {r['pearson_r']:>+7.3f} {r['pearson_p']:>10.4e} "
          f"{r['spearman_rho']:>+7.3f} {r['slope']:>+7.3f}±{r['slope_err']:.3f} {sig}")

# Best predictor
best = results_predictors[0]
print(f"\n  Best predictor: {best['predictor']} (r = {best['pearson_r']:+.3f}, p = {best['pearson_p']:.3e})")

# ================================================================
# PLOT
# ================================================================
print("\n[PLOT] Generating diagnostic and predictor plots...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 14))

    # --- Panel 1: Check 1 — σ distribution with method coloring ---
    ax1 = fig.add_subplot(3, 3, 1)
    spec_s = [(sigma_data[n]['sigma'], sigma_data[n]['e_sigma'])
              for n in sigma_data if sigma_data[n]['method'] == 'spectroscopic']
    vir_s = [(sigma_data[n]['sigma'], sigma_data[n]['e_sigma'])
             for n in sigma_data if sigma_data[n]['method'] == 'virial']
    ax1.errorbar([x[0] for x in spec_s], range(len(spec_s)),
                 xerr=[x[1] for x in spec_s], fmt='o', color='royalblue',
                 label=f'Spectroscopic (N={len(spec_s)})', capsize=3)
    ax1.errorbar([x[0] for x in vir_s], range(len(spec_s), len(spec_s)+len(vir_s)),
                 xerr=[x[1] for x in vir_s], fmt='s', color='orange',
                 label=f'Virial (N={len(vir_s)})', capsize=3)
    ax1.set_xlabel('σ (km/s)')
    ax1.set_title('Check 1: σ distribution by method')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Check 1 — spectroscopic-only fit ---
    ax2 = fig.add_subplot(3, 3, 2)
    for i, n in enumerate(spec_names):
        c = 'orange' if spec_flags[i] else 'royalblue'
        ax2.scatter(spec_log_s2[i], spec_log_gd[i], c=c, s=60, zorder=5,
                    edgecolors='k', linewidth=0.5)
        ax2.annotate(n, (spec_log_s2[i], spec_log_gd[i]), fontsize=6,
                     xytext=(3, 3), textcoords='offset points')
    xf = np.linspace(spec_log_s2.min()-0.1, spec_log_s2.max()+0.1, 50)
    ax2.plot(xf, sl * xf + ic, 'r-', lw=2, label=f'B={sl:.3f}±{se:.3f}')
    ax2.set_xlabel('log(σ²) [spectroscopic only]')
    ax2.set_ylabel('log(g‡)')
    ax2.set_title(f'Check 1: Spec-only (N={len(spec_names)}), r={rp:.3f}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Check 2 — g‡ ordered by σ with error bars ---
    ax3 = fig.add_subplot(3, 3, 3)
    # Sort matched clusters by σ
    matched_sorted = sorted(test_a['clusters'], key=lambda x: x['sigma_kms'])
    ms_names = [m['name'] for m in matched_sorted]
    ms_gd = [m['log_gdagger'] for m in matched_sorted]
    ms_sigma = [m['sigma_kms'] for m in matched_sorted]
    # Estimate g‡ error bar from RAR scatter / sqrt(N)
    ms_err = [cluster_gdagger[m['name']]['scatter'] / np.sqrt(cluster_gdagger[m['name']]['n_pts'])
              for m in matched_sorted]

    y_pos = range(len(ms_names))
    ax3.errorbar(ms_gd, y_pos, xerr=ms_err, fmt='o', color='royalblue', capsize=3, markersize=5)
    ax3.set_yticks(list(y_pos))
    ax3.set_yticklabels([f"{n} (σ={s:.0f})" for n, s in zip(ms_names, ms_sigma)], fontsize=7)
    ax3.axvline(np.mean(all_log_gd), color='red', ls='--', alpha=0.5,
                label=f'Mean = {np.mean(all_log_gd):.3f}')
    ax3.set_xlabel('log(g‡)')
    ax3.set_title('Check 2: g‡ ordered by σ')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='x')

    # --- Panels 4-9: Predictor plots ---
    predictor_plots = [
        ('g_200', log_g200, gd_arr, g200_names, r'$\log(g_{200})$'),
        ('c_200', np.log10(c200_arr), c200_gd_arr, c200_names, r'$\log(c_{200})$'),
        ('M*_BCG', np.log10(mstar_arr), mstar_gd_arr, mstar_names, r'$\log(M_*/M_\odot)$'),
        ('f_bar', np.log10(fbar_arr), fbar_gd_arr, fbar_names, r'$\log(f_{\rm bar})$'),
        ('Mtot_BCG', np.log10(mtot_arr), mtot_gd_arr, mtot_names, r'$\log(M_{\rm tot}/M_\odot)$'),
        ('z', z_arr, z_gd_arr, z_names, 'Redshift z'),
    ]

    for idx, (label, x_data, y_data, nm, xlabel) in enumerate(predictor_plots):
        ax = fig.add_subplot(3, 3, idx + 4)
        ax.scatter(x_data, y_data, c='royalblue', s=40, zorder=5,
                   edgecolors='k', linewidth=0.5)
        for i, n in enumerate(nm):
            ax.annotate(n, (x_data[i], y_data[i]), fontsize=5,
                        xytext=(2, 2), textcoords='offset points')

        # Fit line
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        if np.sum(mask) >= 4:
            s_, i_, _, _, se_ = linregress(x_data[mask], y_data[mask])
            rp_, pp_ = pearsonr(x_data[mask], y_data[mask])
            xf = np.linspace(x_data[mask].min(), x_data[mask].max(), 50)
            ax.plot(xf, s_ * xf + i_, 'r-', lw=1.5)
            sig_str = '***' if pp_ < 0.01 else '**' if pp_ < 0.05 else '*' if pp_ < 0.10 else ''
            ax.set_title(f'{label}: r={rp_:.3f} (p={pp_:.2e}) {sig_str}', fontsize=9)
        else:
            ax.set_title(f'{label}: N<4, no fit', fontsize=9)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(r'$\log(g^\ddagger)$', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Test A Diagnostics + Alternative Predictors for Cluster g‡', fontsize=13, y=1.01)
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, 'cluster_sigma_diagnostics.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

except ImportError:
    print("  matplotlib not available")
    fig_path = None

# ================================================================
# SAVE RESULTS
# ================================================================
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

summary = {
    'test_name': 'cluster_sigma_diagnostics_and_A2',
    'check1_dynamic_range': {
        'sigma_min': float(all_sigma.min()),
        'sigma_max': float(all_sigma.max()),
        'log_sigma2_range_dex': float(np.log10(all_sigma.max()**2) - np.log10(all_sigma.min()**2)),
        'median_sigma_uncertainty_pct': float(np.median(all_e_sigma / all_sigma) * 100),
        'spectroscopic_only': {
            'n': len(spec_names),
            'B_slope': float(sl),
            'B_stderr': float(se),
            'pearson_r': float(rp),
            'pearson_p': float(pp),
        },
    },
    'check2_gdagger_variation': {
        'n_clusters': 20,
        'mean_log_gdagger': float(np.mean(all_log_gd)),
        'std_log_gdagger': float(np.std(all_log_gd)),
        'range_dex': float(np.max(all_log_gd) - np.min(all_log_gd)),
        'obs_variance': float(obs_var),
        'meas_variance': float(meas_var),
        'intrinsic_variance': float(intrinsic_var),
        'intrinsic_sigma': float(np.sqrt(intrinsic_var)),
        'intrinsic_fraction': float(intrinsic_frac),
        'verdict': gd_verdict,
    },
    'test_A2_predictors': results_predictors,
    'best_predictor': best['predictor'] if results_predictors else None,
}

out_path = os.path.join(RESULTS_DIR, 'summary_cluster_diagnostics_A2.json')
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {out_path}")
print(f"Figure saved to: {fig_path}")
print("=" * 72)
