#!/usr/bin/env python3
"""
Test A3: Core-Acceleration Proxy & Multivariate Regression
============================================================

1. g_core = G M_tot(Rad) / Rad² — direct core acceleration vs g‡
2. Multivariate: log g‡ = a + b log M_core + c log c_200 + d log g_200
3. Deeper probe of the σ anti-correlation in clean spectroscopic subset

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'cluster_rar')

g_dagger_gal = 1.20e-10
G_SI = 6.674e-11             # m³ / (kg s²)
M_sun = 1.989e30             # kg
kpc_m = 3.086e19             # metres per kpc
Mpc_m = 3.086e22

print("=" * 72)
print("TEST A3: CORE-ACCELERATION PROXY & MULTIVARIATE REGRESSION")
print("=" * 72)

# ================================================================
# LOAD ALL DATA
# ================================================================
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

# Parse Tian+2020 Table 1
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
            z = float(parts[1].strip())
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

# M200/r200/c200 from Pizzardo+2025 + Adam+2022
m200_data = {
    'A209':    {'m200': 17.3e14, 'r200': 2.31, 'c200': 3.4},
    'A383':    {'m200': 8.4e14,  'r200': 1.83, 'c200': 2.5},
    'MACS0329':{'m200': 11.5e14, 'r200': 1.84, 'c200': 5.4},
    'MACS1115':{'m200': 10.7e14, 'r200': 1.87, 'c200': 2.5},
    'MACS1206':{'m200': 15.9e14, 'r200': 2.06, 'c200': 5.8},
    'MACS1931':{'m200': 11.5e14, 'r200': 1.91, 'c200': 7.8},
    'MS2137':  {'m200': 7.9e14,  'r200': 1.70, 'c200': 2.4},
    'RXJ2129': {'m200': 7.7e14,  'r200': 1.75, 'c200': 2.9},
    'RXJ2248': {'m200': 22.7e14, 'r200': 2.40, 'c200': 1.6},
    'MACS0647':{'m200': 18.1e14, 'r200': 2.06, 'c200': None},
}

# σ data
a_path = os.path.join(RESULTS_DIR, 'summary_cluster_sigma_scaling.json')
with open(a_path) as f:
    test_a = json.load(f)
sigma_data = {}
for cl in test_a['clusters']:
    sigma_data[cl['name']] = {
        'sigma': cl['sigma_kms'], 'e_sigma': cl['e_sigma'],
        'method': cl['sigma_method'], 'flag': cl['flag'],
    }

# ================================================================
# 1. g_core = G M_tot(Rad) / Rad²
# ================================================================
print("\n" + "=" * 72)
print("1. CORE ACCELERATION: g_core = G M_tot / Rad²")
print("=" * 72)

print(f"\n  {'Cluster':<12} {'Rad':>6} {'Mtot':>8} {'g_core':>12} {'log g_core':>10} {'log g‡':>8}")
print(f"  " + "-" * 62)

gcore_names = []
gcore_vals = []
gcore_gd = []
gcore_rad = []
gcore_mtot = []

for name in sorted(bcg_props.keys()):
    p = bcg_props[name]
    if name not in cluster_gdagger or p['Mtot_1e12'] is None or p['Rad_kpc'] is None:
        continue
    mtot_kg = p['Mtot_1e12'] * 1e12 * M_sun
    rad_m = p['Rad_kpc'] * kpc_m
    gcore = G_SI * mtot_kg / rad_m**2

    gcore_names.append(name)
    gcore_vals.append(gcore)
    gcore_gd.append(cluster_gdagger[name]['log_gdagger'])
    gcore_rad.append(p['Rad_kpc'])
    gcore_mtot.append(p['Mtot_1e12'] * 1e12)

    print(f"  {name:<12} {p['Rad_kpc']:>5.1f}k {p['Mtot_1e12']:>7.2f}T "
          f"{gcore:>12.3e} {np.log10(gcore):>10.4f} {cluster_gdagger[name]['log_gdagger']:>8.3f}")

gcore_arr = np.array(gcore_vals)
log_gcore = np.log10(gcore_arr)
gd_arr = np.array(gcore_gd)

print(f"\n  g_core range: {gcore_arr.min():.3e} – {gcore_arr.max():.3e} m/s²")
print(f"  log g_core range: {log_gcore.min():.3f} – {log_gcore.max():.3f} ({log_gcore.max()-log_gcore.min():.3f} dex)")
print(f"  For reference: g† (galaxy) = 1.2×10⁻¹⁰ = {np.log10(g_dagger_gal):.3f}")

# Fit log g‡ = A + B × log g_core
sl, ic, rv, pv, se = linregress(log_gcore, gd_arr)
rp, pp = pearsonr(log_gcore, gd_arr)
rs, ps = spearmanr(log_gcore, gd_arr)

print(f"\n  FIT: log g‡ = A + B × log g_core")
print(f"    B = {sl:.4f} ± {se:.4f}")
print(f"    A = {ic:.4f}")
print(f"    Pearson  r = {rp:+.4f}, p = {pp:.4e}")
print(f"    Spearman ρ = {rs:+.4f}, p = {ps:.4e}")

# Bootstrap confidence on B
np.random.seed(42)
n_boot = 10000
B_boot = np.zeros(n_boot)
for i in range(n_boot):
    idx = np.random.choice(len(gcore_arr), len(gcore_arr), replace=True)
    b_, _, _, _, _ = linregress(log_gcore[idx], gd_arr[idx])
    B_boot[i] = b_
boot_ci = np.percentile(B_boot, [2.5, 97.5])
print(f"    Bootstrap 95% CI for B: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
print(f"    B=1 in CI? {'YES' if boot_ci[0] <= 1.0 <= boot_ci[1] else 'NO'}")
print(f"    B=0 in CI? {'YES' if boot_ci[0] <= 0.0 <= boot_ci[1] else 'NO'}")

if pp < 0.01:
    gcore_verdict = f"HIGHLY SIGNIFICANT (p={pp:.3e}): g‡ tracks g_core with B={sl:.3f}"
elif pp < 0.05:
    gcore_verdict = f"SIGNIFICANT (p={pp:.3e}): g‡ tracks g_core with B={sl:.3f}"
else:
    gcore_verdict = f"Not significant (p={pp:.3e})"
print(f"    → {gcore_verdict}")

# Check if B ≈ 1 (linear scaling)
if abs(sl - 1.0) < 2 * se:
    linear_verdict = "CONSISTENT with linear scaling g‡ ∝ g_core"
elif sl > 0 and pp < 0.05:
    linear_verdict = f"Sub-linear: g‡ ∝ g_core^{sl:.2f} (B=1 {'in' if boot_ci[0]<=1<=boot_ci[1] else 'outside'} 95% CI)"
else:
    linear_verdict = "No clear scaling"
print(f"    → {linear_verdict}")

# Ratio g‡ / g_core
ratio = np.array([10**gd_arr[i] for i in range(len(gd_arr))]) / gcore_arr
print(f"\n  g‡ / g_core ratio:")
print(f"    Mean:   {np.mean(ratio):.4f}")
print(f"    Median: {np.median(ratio):.4f}")
print(f"    Std:    {np.std(ratio):.4f}")
print(f"    → g‡ is {np.mean(ratio):.2f}× g_core on average")

# ================================================================
# 1b. Also try g_core with FIXED 14 kpc for all clusters
# ================================================================
print(f"\n  --- Variant: g_core with fixed R = 14 kpc for all clusters ---")

gcore14_names = []
gcore14_vals = []
gcore14_gd = []
rad_fixed = 14.0  # kpc

for name in sorted(bcg_props.keys()):
    p = bcg_props[name]
    if name not in cluster_gdagger or p['Mtot_1e12'] is None:
        continue
    mtot_kg = p['Mtot_1e12'] * 1e12 * M_sun
    rad_m = rad_fixed * kpc_m
    gcore14 = G_SI * mtot_kg / rad_m**2

    gcore14_names.append(name)
    gcore14_vals.append(gcore14)
    gcore14_gd.append(cluster_gdagger[name]['log_gdagger'])

gcore14_arr = np.array(gcore14_vals)
log_gcore14 = np.log10(gcore14_arr)
gd14_arr = np.array(gcore14_gd)

sl14, ic14, _, _, se14 = linregress(log_gcore14, gd14_arr)
rp14, pp14 = pearsonr(log_gcore14, gd14_arr)
print(f"    B = {sl14:.4f} ± {se14:.4f}, r = {rp14:+.4f}, p = {pp14:.4e}")
print(f"    (Since Mtot is measured at Rad, not 14 kpc, this variant is less clean)")

# ================================================================
# 2. MULTIVARIATE REGRESSION
# ================================================================
print("\n" + "=" * 72)
print("2. MULTIVARIATE REGRESSION")
print("=" * 72)

# Find clusters with ALL predictors: g_core, c200, g200
multi_names = []
X_core = []
X_c200 = []
X_g200 = []
Y_gd = []

for name in sorted(bcg_props.keys()):
    p = bcg_props[name]
    if (name in cluster_gdagger and name in m200_data
            and p['Mtot_1e12'] is not None and p['Rad_kpc'] is not None
            and m200_data[name]['c200'] is not None):
        # g_core
        mtot_kg = p['Mtot_1e12'] * 1e12 * M_sun
        rad_m = p['Rad_kpc'] * kpc_m
        gc = G_SI * mtot_kg / rad_m**2

        # g_200
        d = m200_data[name]
        r200_m = d['r200'] * Mpc_m
        m200_kg = d['m200'] * M_sun
        g200 = G_SI * m200_kg / r200_m**2

        multi_names.append(name)
        X_core.append(np.log10(gc))
        X_c200.append(np.log10(d['c200']))
        X_g200.append(np.log10(g200))
        Y_gd.append(cluster_gdagger[name]['log_gdagger'])

X_core = np.array(X_core)
X_c200 = np.array(X_c200)
X_g200 = np.array(X_g200)
Y_gd_multi = np.array(Y_gd)

print(f"\n  Clusters with ALL predictors: {len(multi_names)}")
print(f"  Names: {', '.join(multi_names)}")

if len(multi_names) >= 6:
    # Build design matrix
    X = np.column_stack([np.ones(len(multi_names)), X_core, X_c200, X_g200])
    labels = ['intercept', 'log M_core (via g_core)', 'log c_200', 'log g_200']

    # OLS via pseudo-inverse
    beta = np.linalg.lstsq(X, Y_gd_multi, rcond=None)[0]
    y_pred = X @ beta
    resid = Y_gd_multi - y_pred
    n, p_dim = X.shape
    dof = n - p_dim
    mse = np.sum(resid**2) / dof if dof > 0 else np.inf

    # Standard errors
    try:
        cov = mse * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se_beta = np.full(p_dim, np.nan)

    # t-statistics and p-values
    from scipy.stats import t as t_dist
    t_vals = beta / se_beta
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_vals), dof)) if dof > 0 else np.full(p_dim, 1.0)

    print(f"\n  Full model: log g‡ = a + b×log(g_core) + c×log(c200) + d×log(g200)")
    print(f"  N = {n}, dof = {dof}, RMS resid = {np.std(resid):.4f} dex")
    print(f"\n  {'Predictor':<25} {'Coeff':>8} {'SE':>8} {'t':>8} {'p':>10} {'Sig':>5}")
    print(f"  " + "-" * 67)
    for i, lab in enumerate(labels):
        sig = '***' if p_vals[i] < 0.01 else '**' if p_vals[i] < 0.05 else '*' if p_vals[i] < 0.10 else ''
        print(f"  {lab:<25} {beta[i]:>+8.4f} {se_beta[i]:>8.4f} {t_vals[i]:>+8.3f} {p_vals[i]:>10.4e} {sig:>5}")

    # R² and adjusted R²
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((Y_gd_multi - np.mean(Y_gd_multi))**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else 0
    print(f"\n  R² = {r2:.4f}, Adjusted R² = {r2_adj:.4f}")

    # Pairwise correlations among predictors (check for collinearity)
    print(f"\n  Predictor correlations (collinearity check):")
    pred_pairs = [('g_core', 'c_200', X_core, X_c200),
                  ('g_core', 'g_200', X_core, X_g200),
                  ('c_200', 'g_200', X_c200, X_g200)]
    for n1, n2, v1, v2 in pred_pairs:
        rc, pc = pearsonr(v1, v2)
        print(f"    {n1:>8} vs {n2:<8}: r = {rc:+.3f} (p = {pc:.3e})")

    # Stepwise: try bivariate models to see which pair is best
    print(f"\n  --- Bivariate models (leave-one-out) ---")
    biv_combos = [
        ('g_core + c200', np.column_stack([np.ones(n), X_core, X_c200])),
        ('g_core + g200', np.column_stack([np.ones(n), X_core, X_g200])),
        ('c200 + g200', np.column_stack([np.ones(n), X_c200, X_g200])),
        ('g_core only', np.column_stack([np.ones(n), X_core])),
        ('c200 only', np.column_stack([np.ones(n), X_c200])),
        ('g200 only', np.column_stack([np.ones(n), X_g200])),
    ]
    for label, X_biv in biv_combos:
        b_biv = np.linalg.lstsq(X_biv, Y_gd_multi, rcond=None)[0]
        pred_biv = X_biv @ b_biv
        res_biv = Y_gd_multi - pred_biv
        ss_res_biv = np.sum(res_biv**2)
        r2_biv = 1 - ss_res_biv / ss_tot
        dof_biv = n - X_biv.shape[1]
        r2_adj_biv = 1 - (1 - r2_biv) * (n - 1) / dof_biv if dof_biv > 0 else 0
        aic_biv = n * np.log(ss_res_biv / n) + 2 * X_biv.shape[1]
        print(f"    {label:<20} R²={r2_biv:.4f}  R²_adj={r2_adj_biv:.4f}  AIC={aic_biv:.2f}")

else:
    print(f"  Only {len(multi_names)} clusters with all predictors — insufficient")
    beta, se_beta, p_vals = None, None, None

# ================================================================
# 2b. Bivariate on FULL 20-cluster sample: g_core + M*
# ================================================================
print(f"\n  --- Extended bivariate: g_core + M* (N=20) ---")

ext_names = []
ext_gcore = []
ext_mstar = []
ext_gd = []
for name in sorted(bcg_props.keys()):
    p = bcg_props[name]
    if (name in cluster_gdagger and p['Mtot_1e12'] is not None
            and p['Rad_kpc'] is not None and p['Mstar_1e12'] is not None):
        mtot_kg = p['Mtot_1e12'] * 1e12 * M_sun
        rad_m = p['Rad_kpc'] * kpc_m
        gc = G_SI * mtot_kg / rad_m**2
        ext_names.append(name)
        ext_gcore.append(np.log10(gc))
        ext_mstar.append(np.log10(p['Mstar_1e12'] * 1e12))
        ext_gd.append(cluster_gdagger[name]['log_gdagger'])

ext_gcore = np.array(ext_gcore)
ext_mstar = np.array(ext_mstar)
ext_gd_arr = np.array(ext_gd)
n_ext = len(ext_names)

if n_ext >= 6:
    # g_core only
    X_1 = np.column_stack([np.ones(n_ext), ext_gcore])
    b1 = np.linalg.lstsq(X_1, ext_gd_arr, rcond=None)[0]
    r1 = ext_gd_arr - X_1 @ b1
    ss_tot_ext = np.sum((ext_gd_arr - np.mean(ext_gd_arr))**2)
    r2_1 = 1 - np.sum(r1**2) / ss_tot_ext

    # g_core + M*
    X_2 = np.column_stack([np.ones(n_ext), ext_gcore, ext_mstar])
    b2 = np.linalg.lstsq(X_2, ext_gd_arr, rcond=None)[0]
    r2_2 = ext_gd_arr - X_2 @ b2
    r2_gc_ms = 1 - np.sum(r2_2**2) / ss_tot_ext
    dof_2 = n_ext - 3

    mse2 = np.sum(r2_2**2) / dof_2
    try:
        cov2 = mse2 * np.linalg.inv(X_2.T @ X_2)
        se2 = np.sqrt(np.diag(cov2))
        t2 = b2 / se2
        p2 = 2 * (1 - t_dist.cdf(np.abs(t2), dof_2))
    except:
        se2 = np.full(3, np.nan)
        t2 = np.full(3, np.nan)
        p2 = np.full(3, 1.0)

    print(f"    N = {n_ext}")
    print(f"    g_core only:        R² = {r2_1:.4f}")
    print(f"    g_core + log M*:    R² = {r2_gc_ms:.4f}")
    print(f"    Coefficients for g_core + M*:")
    for lab, b, s, t, p in zip(['intercept', 'log g_core', 'log M*'],
                                b2, se2, t2, p2):
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        print(f"      {lab:<15} {b:>+8.4f} ± {s:.4f}  (t={t:>+6.2f}, p={p:.3e}) {sig}")

    # Check collinearity
    rc_gm, _ = pearsonr(ext_gcore, ext_mstar)
    print(f"    Collinearity: r(g_core, M*) = {rc_gm:+.3f}")

# ================================================================
# 3. DEEPER PROBE OF σ ANTI-CORRELATION
# ================================================================
print("\n" + "=" * 72)
print("3. σ ANTI-CORRELATION ANALYSIS")
print("=" * 72)

# Clean spectroscopic: no flags
clean_spec = [(n, sigma_data[n]['sigma'], cluster_gdagger[n]['log_gdagger'])
              for n in sigma_data
              if sigma_data[n]['method'] == 'spectroscopic' and sigma_data[n]['flag'] is None]

print(f"\n  Clean spectroscopic sample (N={len(clean_spec)}):")
for name, sig, lgd in clean_spec:
    has_core = name in bcg_props and bcg_props[name]['Mtot_1e12'] is not None
    mtot = bcg_props[name]['Mtot_1e12'] if has_core else None
    rad = bcg_props[name]['Rad_kpc'] if has_core else None
    print(f"    {name:<12} σ={sig:>6.0f}  log g‡={lgd:>7.3f}"
          f"  Mtot={mtot:.2f}T  Rad={rad:.1f}kpc" if has_core else
          f"    {name:<12} σ={sig:>6.0f}  log g‡={lgd:>7.3f}")

cs_names = [x[0] for x in clean_spec]
cs_sigma = np.array([x[1] for x in clean_spec])
cs_gd = np.array([x[2] for x in clean_spec])

sl_cs, ic_cs, _, _, se_cs = linregress(np.log10(cs_sigma**2), cs_gd)
rp_cs, pp_cs = pearsonr(np.log10(cs_sigma**2), cs_gd)
print(f"\n  log g‡ vs log σ²: B = {sl_cs:.4f} ± {se_cs:.4f}, r = {rp_cs:+.4f}, p = {pp_cs:.4e}")

# Investigate: do high-σ clusters have different core structure?
print(f"\n  Core structure vs σ:")
print(f"  {'Cluster':<12} {'σ':>6} {'log g‡':>8} {'Mtot':>8} {'Rad':>6} {'g_core':>12} {'g‡/g_core':>10}")
print(f"  " + "-" * 72)

for name, sig, lgd in sorted(clean_spec, key=lambda x: x[1]):
    p = bcg_props.get(name, {})
    if p.get('Mtot_1e12') is not None and p.get('Rad_kpc') is not None:
        mtot_kg = p['Mtot_1e12'] * 1e12 * M_sun
        rad_m = p['Rad_kpc'] * kpc_m
        gc = G_SI * mtot_kg / rad_m**2
        rat_cs = 10**lgd / gc
        print(f"  {name:<12} {sig:>6.0f} {lgd:>8.3f} {p['Mtot_1e12']:>7.2f}T {p['Rad_kpc']:>5.1f}k "
              f"{gc:>12.3e} {rat_cs:>10.4f}")

# Partial correlation: g‡ vs σ, controlling for g_core
print(f"\n  Partial correlation: log g‡ vs log σ², controlling for log g_core")
# For the N=5 clean spec clusters
if len(clean_spec) >= 5:
    cs_gcore = []
    for name in cs_names:
        p = bcg_props[name]
        mtot_kg = p['Mtot_1e12'] * 1e12 * M_sun
        rad_m = p['Rad_kpc'] * kpc_m
        cs_gcore.append(np.log10(G_SI * mtot_kg / rad_m**2))
    cs_gcore = np.array(cs_gcore)
    cs_log_s2 = np.log10(cs_sigma**2)

    # Partial correlation of g‡ vs σ² | g_core
    # Regress both on g_core, correlate residuals
    _, _, _, _, _ = linregress(cs_gcore, cs_gd)
    res_gd = cs_gd - (linregress(cs_gcore, cs_gd)[0] * cs_gcore + linregress(cs_gcore, cs_gd)[1])
    res_s2 = cs_log_s2 - (linregress(cs_gcore, cs_log_s2)[0] * cs_gcore + linregress(cs_gcore, cs_log_s2)[1])
    rp_partial, pp_partial = pearsonr(res_s2, res_gd)
    print(f"    Partial r(g‡, σ² | g_core) = {rp_partial:+.4f}, p = {pp_partial:.4e}")

    # Also: σ vs g_core correlation
    rp_sg, pp_sg = pearsonr(cs_log_s2, cs_gcore)
    print(f"    r(σ², g_core) = {rp_sg:+.4f}, p = {pp_sg:.4e}")
    if rp_sg < -0.3:
        print(f"    → High-σ clusters tend to have LOWER g_core — suggests core disruption")
    elif rp_sg > 0.3:
        print(f"    → High-σ clusters have HIGHER g_core")
    else:
        print(f"    → No σ–g_core correlation")

# Extend to ALL clusters with both σ and g_core (N=14)
print(f"\n  Extended: ALL σ clusters with core data (N=?)")
all_with_both = []
for name in sigma_data:
    if name in bcg_props and bcg_props[name]['Mtot_1e12'] is not None:
        p = bcg_props[name]
        mtot_kg = p['Mtot_1e12'] * 1e12 * M_sun
        rad_m = p['Rad_kpc'] * kpc_m
        gc = G_SI * mtot_kg / rad_m**2
        all_with_both.append({
            'name': name,
            'sigma': sigma_data[name]['sigma'],
            'log_gcore': np.log10(gc),
            'log_gd': cluster_gdagger[name]['log_gdagger'],
            'method': sigma_data[name]['method'],
            'flag': sigma_data[name]['flag'],
        })

n_both = len(all_with_both)
print(f"    N = {n_both}")

if n_both >= 6:
    ab_sigma = np.array([x['sigma'] for x in all_with_both])
    ab_gcore = np.array([x['log_gcore'] for x in all_with_both])
    ab_gd = np.array([x['log_gd'] for x in all_with_both])
    ab_ls2 = np.log10(ab_sigma**2)

    # σ vs g_core
    rsg, psg = pearsonr(ab_ls2, ab_gcore)
    print(f"    r(σ², g_core) = {rsg:+.4f}, p = {psg:.4e}")

    # Partial: g‡ vs σ² | g_core
    res_gd_all = ab_gd - (linregress(ab_gcore, ab_gd)[0] * ab_gcore + linregress(ab_gcore, ab_gd)[1])
    res_s2_all = ab_ls2 - (linregress(ab_gcore, ab_ls2)[0] * ab_gcore + linregress(ab_gcore, ab_ls2)[1])
    rp_part_all, pp_part_all = pearsonr(res_s2_all, res_gd_all)
    print(f"    Partial r(g‡, σ² | g_core) = {rp_part_all:+.4f}, p = {pp_part_all:.4e}")

    # Residual from g_core-only model vs σ
    sl_gc, ic_gc, _, _, _ = linregress(ab_gcore, ab_gd)
    resid_gcore = ab_gd - (sl_gc * ab_gcore + ic_gc)
    r_resid_sig, p_resid_sig = pearsonr(ab_ls2, resid_gcore)
    print(f"    r(g_core_residual, σ²) = {r_resid_sig:+.4f}, p = {p_resid_sig:.4e}")

# ================================================================
# PLOT
# ================================================================
print("\n[PLOT] Generating figures...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 12))

    # --- Panel 1: g‡ vs g_core (the money plot) ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(log_gcore, gd_arr, c='royalblue', s=60, zorder=5,
                edgecolors='k', linewidth=0.5)
    for i, n in enumerate(gcore_names):
        ax1.annotate(n, (log_gcore[i], gd_arr[i]), fontsize=6,
                     xytext=(3, 3), textcoords='offset points')

    xf = np.linspace(log_gcore.min() - 0.05, log_gcore.max() + 0.05, 50)
    ax1.plot(xf, sl * xf + ic, 'r-', lw=2,
             label=f'B={sl:.3f}±{se:.3f}\nr={rp:.3f}, p={pp:.2e}')
    # B=1 reference (shifted to pass through median)
    mx, my = np.median(log_gcore), np.median(gd_arr)
    ax1.plot(xf, 1.0 * (xf - mx) + my, 'g--', lw=1.5, alpha=0.6, label='B=1 (linear)')
    ax1.set_xlabel(r'$\log(g_{\rm core})$ [m/s²]', fontsize=11)
    ax1.set_ylabel(r'$\log(g^\ddagger)$ [m/s²]', fontsize=11)
    ax1.set_title(r'$g^\ddagger$ vs $g_{\rm core} = GM_{\rm tot}/R_{\rm ap}^2$', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: g‡ / g_core ratio vs cluster mass ---
    ax2 = fig.add_subplot(2, 3, 2)
    for i, n in enumerate(gcore_names):
        ax2.scatter(np.log10(gcore_mtot[i]), ratio[i], c='royalblue', s=60,
                    zorder=5, edgecolors='k', linewidth=0.5)
        ax2.annotate(n, (np.log10(gcore_mtot[i]), ratio[i]), fontsize=6,
                     xytext=(3, 3), textcoords='offset points')
    ax2.axhline(np.median(ratio), color='red', ls='--', alpha=0.5,
                label=f'Median = {np.median(ratio):.3f}')
    ax2.set_xlabel(r'$\log(M_{\rm tot}/M_\odot)$', fontsize=11)
    ax2.set_ylabel(r'$g^\ddagger / g_{\rm core}$', fontsize=11)
    ax2.set_title(r'$g^\ddagger / g_{\rm core}$ ratio', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Multivariate — actual vs predicted ---
    if beta is not None:
        ax3 = fig.add_subplot(2, 3, 3)
        y_pred_mv = X @ beta
        ax3.scatter(y_pred_mv, Y_gd_multi, c='royalblue', s=60, zorder=5,
                    edgecolors='k', linewidth=0.5)
        for i, n in enumerate(multi_names):
            ax3.annotate(n, (y_pred_mv[i], Y_gd_multi[i]), fontsize=6,
                         xytext=(3, 3), textcoords='offset points')
        lims = [min(y_pred_mv.min(), Y_gd_multi.min()) - 0.05,
                max(y_pred_mv.max(), Y_gd_multi.max()) + 0.05]
        ax3.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        ax3.set_xlabel('Predicted log g‡', fontsize=11)
        ax3.set_ylabel('Observed log g‡', fontsize=11)
        ax3.set_title(f'Multivariate (R²={r2:.3f}, N={len(multi_names)})', fontsize=11)
        ax3.grid(True, alpha=0.3)

    # --- Panel 4: σ anti-correlation deep dive ---
    ax4 = fig.add_subplot(2, 3, 4)
    # Plot all σ clusters color-coded by method
    for x in all_with_both:
        c = 'orange' if x['flag'] else ('royalblue' if x['method'] == 'spectroscopic' else 'green')
        m = 'o' if x['method'] == 'spectroscopic' else 's'
        ax4.scatter(np.log10(x['sigma']**2), x['log_gd'], c=c, marker=m,
                    s=60, zorder=5, edgecolors='k', linewidth=0.5)
        ax4.annotate(x['name'], (np.log10(x['sigma']**2), x['log_gd']), fontsize=5,
                     xytext=(2, 2), textcoords='offset points')
    # Clean spec fit line
    xcs = np.linspace(5.6, 6.6, 50)
    ax4.plot(xcs, sl_cs * xcs + ic_cs, 'r-', lw=2,
             label=f'Clean spec: B={sl_cs:.2f}±{se_cs:.2f}')
    ax4.set_xlabel(r'$\log(\sigma^2)$', fontsize=11)
    ax4.set_ylabel(r'$\log(g^\ddagger)$', fontsize=11)
    ax4.set_title(f'σ anti-correlation (spec r={rp_cs:.3f}, p={pp_cs:.3e})', fontsize=11)
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
               markersize=8, label='Spec (clean)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
               markersize=8, label='Virial'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=8, label='Flagged'),
    ]
    ax4.legend(handles=legend_els, fontsize=7, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: σ vs g_core (testing if high-σ → low g_core) ---
    ax5 = fig.add_subplot(2, 3, 5)
    for x in all_with_both:
        c = 'orange' if x['flag'] else ('royalblue' if x['method'] == 'spectroscopic' else 'green')
        m = 'o' if x['method'] == 'spectroscopic' else 's'
        ax5.scatter(np.log10(x['sigma']**2), x['log_gcore'], c=c, marker=m,
                    s=60, zorder=5, edgecolors='k', linewidth=0.5)
        ax5.annotate(x['name'], (np.log10(x['sigma']**2), x['log_gcore']), fontsize=5,
                     xytext=(2, 2), textcoords='offset points')
    ax5.set_xlabel(r'$\log(\sigma^2)$', fontsize=11)
    ax5.set_ylabel(r'$\log(g_{\rm core})$', fontsize=11)
    ax5.set_title(f'σ vs g_core: r={rsg:+.3f} (p={psg:.2e})', fontsize=11)
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Residual from g_core model vs σ ---
    ax6 = fig.add_subplot(2, 3, 6)
    for i, x in enumerate(all_with_both):
        c = 'orange' if x['flag'] else ('royalblue' if x['method'] == 'spectroscopic' else 'green')
        m = 'o' if x['method'] == 'spectroscopic' else 's'
        ax6.scatter(np.log10(x['sigma']**2), resid_gcore[i], c=c, marker=m,
                    s=60, zorder=5, edgecolors='k', linewidth=0.5)
        ax6.annotate(x['name'], (np.log10(x['sigma']**2), resid_gcore[i]),
                     fontsize=5, xytext=(2, 2), textcoords='offset points')
    ax6.axhline(0, color='k', ls='--', alpha=0.3)
    ax6.set_xlabel(r'$\log(\sigma^2)$', fontsize=11)
    ax6.set_ylabel(r'$\Delta\log(g^\ddagger)$ after g_core', fontsize=11)
    ax6.set_title(f'g_core residual vs σ: r={r_resid_sig:+.3f} (p={p_resid_sig:.2e})', fontsize=11)
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Test A3: Core Acceleration, Multivariate, and σ Anti-Trend', fontsize=13, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'cluster_gcore_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

except ImportError:
    fig_path = None
    print("  matplotlib not available")

# ================================================================
# SUMMARY & SAVE
# ================================================================
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

print(f"\n  1. g_core = GM_tot/Rad²:")
print(f"     B = {sl:.4f} ± {se:.4f}, r = {rp:+.4f}, p = {pp:.4e}")
print(f"     g‡ / g_core = {np.mean(ratio):.3f} ± {np.std(ratio):.3f}")
print(f"     → {gcore_verdict}")

if beta is not None:
    print(f"\n  2. Multivariate (N={len(multi_names)}):")
    best_pred_idx = np.argmin(p_vals[1:]) + 1  # skip intercept
    print(f"     Strongest predictor: {labels[best_pred_idx]} (p={p_vals[best_pred_idx]:.3e})")
    print(f"     R² = {r2:.4f}")

print(f"\n  3. σ anti-correlation:")
print(f"     Clean spec (N={len(clean_spec)}): B = {sl_cs:.3f}, r = {rp_cs:+.3f}, p = {pp_cs:.3e}")
if n_both >= 6:
    print(f"     Partial r(g‡, σ² | g_core) = {rp_part_all:+.4f}, p = {pp_part_all:.4e}")
    print(f"     σ vs g_core: r = {rsg:+.4f} — {'HIGH σ → LOW g_core' if rsg < -0.3 else 'no clear pattern'}")

summary = {
    'test_name': 'cluster_gcore_scaling_A3',
    'gcore_analysis': {
        'description': 'g_core = G M_tot(Rad) / Rad², where Rad is the BCG aperture radius from Tian+2020',
        'n_clusters': len(gcore_names),
        'gcore_range': [float(gcore_arr.min()), float(gcore_arr.max())],
        'log_gcore_range_dex': float(log_gcore.max() - log_gcore.min()),
        'fit_B': float(sl),
        'fit_B_err': float(se),
        'fit_A': float(ic),
        'pearson_r': float(rp),
        'pearson_p': float(pp),
        'spearman_rho': float(rs),
        'spearman_p': float(ps),
        'bootstrap_B_95ci': [float(boot_ci[0]), float(boot_ci[1])],
        'B1_in_ci': bool(boot_ci[0] <= 1.0 <= boot_ci[1]),
        'gdagger_over_gcore_mean': float(np.mean(ratio)),
        'gdagger_over_gcore_median': float(np.median(ratio)),
        'gdagger_over_gcore_std': float(np.std(ratio)),
        'verdict': gcore_verdict,
        'linear_verdict': linear_verdict,
    },
    'multivariate': {
        'n_clusters': len(multi_names),
        'cluster_names': multi_names,
        'coefficients': {
            labels[i]: {
                'value': float(beta[i]),
                'stderr': float(se_beta[i]),
                'p_value': float(p_vals[i]),
            } for i in range(len(labels))
        } if beta is not None else None,
        'R2': float(r2) if beta is not None else None,
        'R2_adj': float(r2_adj) if beta is not None else None,
    },
    'sigma_anti_correlation': {
        'clean_spectroscopic_N': len(clean_spec),
        'B_slope': float(sl_cs),
        'B_stderr': float(se_cs),
        'pearson_r': float(rp_cs),
        'pearson_p': float(pp_cs),
        'partial_r_gd_sigma_given_gcore': float(rp_part_all) if n_both >= 6 else None,
        'partial_p': float(pp_part_all) if n_both >= 6 else None,
        'sigma_vs_gcore_r': float(rsg) if n_both >= 6 else None,
    },
    'per_cluster_gcore': [
        {
            'name': gcore_names[i],
            'log_gcore': float(log_gcore[i]),
            'gcore': float(gcore_arr[i]),
            'log_gdagger': float(gd_arr[i]),
            'gdagger_over_gcore': float(ratio[i]),
            'Mtot_1e12': bcg_props[gcore_names[i]]['Mtot_1e12'],
            'Rad_kpc': bcg_props[gcore_names[i]]['Rad_kpc'],
        }
        for i in range(len(gcore_names))
    ],
}

out_path = os.path.join(RESULTS_DIR, 'summary_cluster_gcore_A3.json')
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {out_path}")
print("=" * 72)
