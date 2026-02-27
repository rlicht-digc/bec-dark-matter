#!/usr/bin/env python3
"""
TESTS B.1–B.3: Periodic Galaxy Properties, λ-ξ Scaling, and Bump Sharpening
=============================================================================

Three discriminating tests following TEST B (residual power spectrum):

PART 1: Periodic vs Non-Periodic Property Comparison (with controls)
  - Raw MW/KS tests for ~12 properties, BH-FDR corrected
  - Control A: matched subsample on data quality (N_pts, R_extent, errV, Inc)
  - Control B: k-space coverage as covariate

PART 2: λ_peak vs ξ Scaling (the correlation done properly)
  - log-log regression: log λ = a + b log ξ
  - Spearman ρ (robust headline)
  - λ_peak/ξ distribution

PART 3: Dimensionless Bump Sharpening (the killer check)
  - Stack periodic PSDs in physical k and k×ξ
  - Quantify bump width (FWHM + 2nd moment in log-k)
  - Control: non-periodic PSDs in k×ξ

Russell Licht — BEC Dark Matter Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Constants
g_dagger = 1.20e-10
G_SI = 6.674e-11
Msun_kg = 1.989e30
Lsun_W = 3.828e26
kpc_m = 3.086e19
ML_36 = 0.5            # M/L at 3.6 micron
HE_CORR = 1.33         # Helium correction for HI mass
MIN_POINTS = 15
N_SURR = 200
PERM_ALPHA = 0.05

np.random.seed(42)

print("=" * 76)
print("TESTS B.1-B.3: PERIODIC PROPERTIES, LAMBDA-XI SCALING, BUMP SHARPENING")
print("=" * 76)


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def power_law(log_k, log_A, beta):
    return log_A - beta * log_k


def benjamini_hochberg(pvals, alpha=0.05):
    """Return BH-FDR adjusted p-values and significance flags."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[sorted_idx]

    # Adjusted p-values
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            sorted_p[i] * n / (i + 1),
            adjusted[sorted_idx[i + 1]]
        )
    adjusted = np.minimum(adjusted, 1.0)

    significant = adjusted < alpha
    return adjusted, significant


# ================================================================
# 1. LOAD DATA (extended properties)
# ================================================================
print("\n[1] Loading SPARC data with full property set...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

# --- Table 2: rotation curve data + velocity errors ---
rc_data = {}
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
            errv = float(line[33:38].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in rc_data:
            rc_data[name] = {
                'R': [], 'Vobs': [], 'errV': [], 'Vgas': [],
                'Vdisk': [], 'Vbul': [], 'dist': dist
            }
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['errV'].append(errv)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in ['R', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul']:
        rc_data[name][key] = np.array(rc_data[name][key])

# --- MRT: galaxy properties ---
sparc_props = {}
with open(mrt_path, 'r') as f:
    mrt_lines = f.readlines()
data_start = 0
for i, line in enumerate(mrt_lines):
    if line.startswith('---') and i > 50:
        data_start = i + 1
        break
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
        T_val = int(parts[0]) if parts[0].lstrip('-').isdigit() else None
        L36_val = float(parts[6])    # 10^9 L_sun
        MHI_val = float(parts[12])   # 10^9 M_sun
        sparc_props[name] = {
            'T': T_val,
            'D': float(parts[1]),
            'Inc': float(parts[4]),
            'L36': L36_val,
            'MHI': MHI_val,
            'Reff': float(parts[8]),
            'SBeff': float(parts[9]),
            'Rdisk': float(parts[10]),
            'SBdisk': float(parts[11]),
            'Vflat': float(parts[14]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue


# --- Build galaxy data with full properties ---
galaxy_data = []
perm_rng = np.random.default_rng(789)

for name, gdata in rc_data.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

    R = gdata['R']
    Vobs = gdata['Vobs']
    errV = gdata['errV']
    Vgas = gdata['Vgas']
    Vdisk = gdata['Vdisk']
    Vbul = gdata['Vbul']

    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < MIN_POINTS:
        continue

    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]
    n = len(R_sorted)

    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]
    residuals = log_gobs - rar_function(log_gbar)

    # Spline detrending
    var_eps = np.var(residuals)
    s_param = n * var_eps * 0.5
    try:
        spline = UnivariateSpline(R_sorted, residuals, k=min(3, n - 1), s=s_param)
        eps_det = residuals - spline(R_sorted)
    except Exception:
        eps_det = residuals - np.mean(residuals)

    # Dynamical mass + healing length
    Vobs_sorted = Vobs[valid][sort_idx]
    errV_sorted = errV[valid][sort_idx]
    R_max_kpc = R_sorted[-1]
    V_max_kms = Vobs_sorted[-1]
    M_dyn = (V_max_kms * 1e3)**2 * (R_max_kpc * kpc_m) / G_SI
    xi_kpc = np.sqrt(G_SI * M_dyn / g_dagger) / kpc_m

    # Derived properties
    M_star = ML_36 * prop['L36'] * 1e9 * Msun_kg       # kg
    M_gas = HE_CORR * prop['MHI'] * 1e9 * Msun_kg      # kg
    M_bar = M_star + M_gas
    f_gas = M_gas / M_bar if M_bar > 0 else 0.0
    M_bar_solar = M_bar / Msun_kg

    # k-space coverage
    R_extent = R_sorted[-1] - R_sorted[0]
    f_min = 1.0 / R_extent if R_extent > 0 else np.inf  # min detectable freq
    f_max = (n / 2.0) / R_extent if R_extent > 0 else 0  # Nyquist-like

    # Lomb-Scargle periodogram
    std_eps = np.std(eps_det)
    if std_eps < 1e-30:
        continue
    y = (eps_det - np.mean(eps_det)) / std_eps

    if R_extent <= 0:
        continue

    n_freq_gal = min(500, 10 * n)
    freq_gal = np.linspace(f_min, f_max, n_freq_gal)

    ls = LombScargle(R_sorted, y, fit_mean=False, center_data=True)
    power_gal = ls.power(freq_gal)

    idx_peak = np.argmax(power_gal)
    f_peak = float(freq_gal[idx_peak])
    power_peak = float(power_gal[idx_peak])
    wl_peak = 1.0 / f_peak

    # Permutation test
    null_peaks = np.zeros(N_SURR)
    for s in range(N_SURR):
        y_shuf = perm_rng.permutation(y)
        ls_null = LombScargle(R_sorted, y_shuf, fit_mean=False, center_data=True)
        null_peaks[s] = np.max(ls_null.power(freq_gal))
    p_val = float(np.mean(null_peaks >= power_peak))

    # PSD on common grid for stacking
    N_FREQ = 200
    k_common = np.logspace(np.log10(0.02), np.log10(3.0), N_FREQ)
    valid_k = (k_common >= f_min) & (k_common <= f_max)
    psd_common = np.full(N_FREQ, np.nan)
    if np.sum(valid_k) > 3:
        psd_common[valid_k] = ls.power(k_common[valid_k])

    galaxy_data.append({
        'name': name,
        'R': R_sorted,
        'eps_det': eps_det,
        'n_pts': n,
        'R_extent': R_extent,
        'R_max': R_max_kpc,
        'xi_kpc': xi_kpc,
        'Vflat': prop['Vflat'],
        'M_dyn': M_dyn,
        'M_star_solar': M_star / Msun_kg,
        'M_bar_solar': M_bar_solar,
        'f_gas': f_gas,
        'T': prop['T'],
        'D': prop['D'],
        'Inc': prop['Inc'],
        'L36': prop['L36'],
        'SBeff': prop['SBeff'],
        'Reff': prop['Reff'],
        'median_errV': float(np.median(errV_sorted)),
        'mean_errV': float(np.mean(errV_sorted)),
        'med_log_gbar': float(np.median(log_gbar)),
        'f_min': f_min,
        'f_max': f_max,
        'f_peak': f_peak,
        'wl_peak': wl_peak,
        'power_peak': power_peak,
        'perm_p': p_val,
        'is_periodic': p_val < PERM_ALPHA,
        'psd_common': psd_common,
        'freq_gal': freq_gal,
        'power_gal': power_gal,
    })

n_total = len(galaxy_data)
periodic = [g for g in galaxy_data if g['is_periodic']]
nonperiodic = [g for g in galaxy_data if not g['is_periodic']]
n_per = len(periodic)
n_nper = len(nonperiodic)

print(f"  Total: {n_total}, Periodic: {n_per}, Non-periodic: {n_nper}")


# ================================================================
# PART 1: PROPERTY COMPARISON WITH CONTROLS
# ================================================================
print("\n" + "=" * 76)
print("PART 1: PERIODIC vs NON-PERIODIC PROPERTY COMPARISON")
print("=" * 76)

# Define properties to test
property_defs = [
    # (key, label, is_log, category)
    ('Vflat',         'V_flat [km/s]',        False, 'physical'),
    ('M_bar_solar',   'M_bar [M_sun]',        True,  'physical'),
    ('M_star_solar',  'M_star [M_sun]',       True,  'physical'),
    ('f_gas',         'Gas fraction',         False, 'physical'),
    ('T',             'Hubble type T',        False, 'physical'),
    ('SBeff',         'SB_eff [L/pc^2]',      True,  'physical'),
    ('Reff',          'R_eff [kpc]',          False, 'physical'),
    ('xi_kpc',        'Healing length [kpc]', False, 'physical'),
    ('med_log_gbar',  'Median log g_bar',     False, 'physical'),
    ('n_pts',         'N points',             False, 'quality'),
    ('R_extent',      'R extent [kpc]',       False, 'quality'),
    ('median_errV',   'Median errV [km/s]',   False, 'quality'),
    ('Inc',           'Inclination [deg]',    False, 'quality'),
    ('D',             'Distance [Mpc]',       False, 'quality'),
    ('f_min',         'f_min [1/kpc]',        True,  'quality'),
    ('f_max',         'f_max [1/kpc]',        False, 'quality'),
]


def extract_property(galaxies, key, is_log):
    """Extract property array, handling missing T values and log transform."""
    vals = []
    for g in galaxies:
        v = g[key]
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        if is_log and v > 0:
            vals.append(np.log10(v))
        elif not is_log:
            vals.append(v)
    return np.array(vals, dtype=float)


print("\n[1a] Raw property comparisons (Mann-Whitney U + KS)...")

raw_results = []
raw_pvals = []

for key, label, is_log, category in property_defs:
    per_vals = extract_property(periodic, key, is_log)
    nper_vals = extract_property(nonperiodic, key, is_log)

    if len(per_vals) < 5 or len(nper_vals) < 5:
        raw_results.append({
            'property': label, 'key': key, 'category': category,
            'n_per': len(per_vals), 'n_nper': len(nper_vals),
            'note': 'insufficient data'
        })
        raw_pvals.append(1.0)
        continue

    # Mann-Whitney U
    mw_stat, mw_p = stats.mannwhitneyu(per_vals, nper_vals, alternative='two-sided')
    # KS test
    ks_stat, ks_p = stats.ks_2samp(per_vals, nper_vals)
    # Effect size: rank-biserial r = 1 - 2U/(n1*n2)
    n1, n2 = len(per_vals), len(nper_vals)
    r_biserial = 1 - 2 * mw_stat / (n1 * n2)

    raw_results.append({
        'property': label, 'key': key, 'category': category,
        'n_per': n1, 'n_nper': n2,
        'per_median': round(float(np.median(per_vals)), 4),
        'nper_median': round(float(np.median(nper_vals)), 4),
        'MW_p': float(mw_p),
        'KS_p': float(ks_p),
        'r_biserial': round(float(r_biserial), 3),
    })
    raw_pvals.append(float(mw_p))

# BH-FDR correction
print("\n[1b] Benjamini-Hochberg FDR correction...")

mw_pvals = np.array(raw_pvals)
adjusted_p, sig_flags = benjamini_hochberg(mw_pvals)

for i, r in enumerate(raw_results):
    r['MW_p_adjusted'] = round(float(adjusted_p[i]), 4)
    r['significant_FDR'] = bool(sig_flags[i])

print(f"\n  {'Property':<25} {'Med(P)':<10} {'Med(NP)':<10} {'MW p':>10} {'BH-adj':>10} {'r_bis':>8} {'Sig':>5}")
print("  " + "-" * 78)
for r in raw_results:
    if 'note' in r:
        print(f"  {r['property']:<25} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>8} {'—':>5}")
        continue
    sig_str = "***" if r['significant_FDR'] else ""
    print(f"  {r['property']:<25} {r['per_median']:<10.4g} {r['nper_median']:<10.4g} "
          f"{r['MW_p']:>10.4f} {r['MW_p_adjusted']:>10.4f} {r['r_biserial']:>8.3f} {sig_str:>5}")

n_sig_raw = sum(1 for r in raw_results if r.get('significant_FDR', False))
n_sig_physical = sum(1 for r in raw_results
                     if r.get('significant_FDR', False) and r['category'] == 'physical')
n_sig_quality = sum(1 for r in raw_results
                    if r.get('significant_FDR', False) and r['category'] == 'quality')
print(f"\n  Significant after FDR: {n_sig_raw}/{len(raw_results)}")
print(f"    Physical: {n_sig_physical}, Data quality: {n_sig_quality}")


# ----------------------------------------------------------------
# Control A: Matched subsample on data quality
# ----------------------------------------------------------------
print("\n[1c] Control A: Nearest-neighbor matching on data quality...")

# Matching features: n_pts, log R_extent, median_errV, Inc
def get_match_features(galaxies):
    features = []
    for g in galaxies:
        features.append([
            g['n_pts'],
            np.log10(g['R_extent']) if g['R_extent'] > 0 else 0,
            g['median_errV'],
            g['Inc'],
        ])
    return np.array(features)

feat_per = get_match_features(periodic)
feat_nper = get_match_features(nonperiodic)

# Standardize using pooled statistics
feat_all = np.vstack([feat_per, feat_nper])
feat_mean = feat_all.mean(axis=0)
feat_std = feat_all.std(axis=0)
feat_std[feat_std == 0] = 1.0

feat_per_std = (feat_per - feat_mean) / feat_std
feat_nper_std = (feat_nper - feat_mean) / feat_std

# Nearest-neighbor matching (without replacement)
dists = cdist(feat_per_std, feat_nper_std, metric='euclidean')
matched_nper_idx = []
available = set(range(len(nonperiodic)))

for i in range(len(periodic)):
    candidates = sorted(available, key=lambda j: dists[i, j])
    best = candidates[0]
    matched_nper_idx.append(best)
    available.remove(best)

matched_nonperiodic = [nonperiodic[j] for j in matched_nper_idx]
n_matched = len(matched_nonperiodic)

# Report matching quality
print(f"  Matched {n_per} pairs")
for fi, fname in enumerate(['N_pts', 'log R_ext', 'median errV', 'Inc']):
    med_per = np.median(feat_per[:, fi])
    med_matched = np.median(feat_nper[matched_nper_idx, fi])
    med_unmatched = np.median(feat_nper[:, fi])
    print(f"    {fname:<15}: Periodic={med_per:.2f}, Matched NP={med_matched:.2f}, "
          f"All NP={med_unmatched:.2f}")

# Rerun property comparisons on matched sample
print(f"\n  Matched-sample property comparisons:")

matched_results = []
matched_pvals = []

for key, label, is_log, category in property_defs:
    if category == 'quality':
        continue  # Skip quality vars (we matched on those)

    per_vals = extract_property(periodic, key, is_log)
    m_nper_vals = extract_property(matched_nonperiodic, key, is_log)

    if len(per_vals) < 5 or len(m_nper_vals) < 5:
        matched_results.append({
            'property': label, 'key': key,
            'note': 'insufficient data'
        })
        matched_pvals.append(1.0)
        continue

    mw_stat, mw_p = stats.mannwhitneyu(per_vals, m_nper_vals, alternative='two-sided')
    n1, n2 = len(per_vals), len(m_nper_vals)
    r_biserial = 1 - 2 * mw_stat / (n1 * n2)

    matched_results.append({
        'property': label, 'key': key,
        'per_median': round(float(np.median(per_vals)), 4),
        'matched_nper_median': round(float(np.median(m_nper_vals)), 4),
        'MW_p': float(mw_p),
        'r_biserial': round(float(r_biserial), 3),
    })
    matched_pvals.append(float(mw_p))

# BH-FDR on matched sample
matched_adj_p, matched_sig = benjamini_hochberg(np.array(matched_pvals))
for i, r in enumerate(matched_results):
    r['MW_p_adjusted'] = round(float(matched_adj_p[i]), 4)
    r['significant_FDR'] = bool(matched_sig[i])

print(f"\n  {'Property':<25} {'Med(P)':<10} {'Med(mNP)':<10} {'MW p':>10} {'BH-adj':>10} {'r_bis':>8} {'Sig':>5}")
print("  " + "-" * 78)
for r in matched_results:
    if 'note' in r:
        continue
    sig_str = "***" if r['significant_FDR'] else ""
    print(f"  {r['property']:<25} {r['per_median']:<10.4g} {r['matched_nper_median']:<10.4g} "
          f"{r['MW_p']:>10.4f} {r['MW_p_adjusted']:>10.4f} {r['r_biserial']:>8.3f} {sig_str:>5}")

n_matched_sig = sum(1 for r in matched_results if r.get('significant_FDR', False))
print(f"\n  Significant after matching + FDR: {n_matched_sig}/{len(matched_results)}")


# ----------------------------------------------------------------
# Control B: k-space coverage
# ----------------------------------------------------------------
print("\n[1d] Control B: k-space coverage analysis...")

# Logistic regression: can k-space coverage alone predict periodicity?
from scipy.optimize import minimize

# Features: log(f_min), f_max, n_pts — does adding physical properties improve?
X_quality = np.array([
    [np.log10(g['f_min']) if g['f_min'] > 0 and np.isfinite(g['f_min']) else 0,
     g['f_max'],
     g['n_pts'],
     g['R_extent']]
    for g in galaxy_data
])
y_periodic = np.array([1 if g['is_periodic'] else 0 for g in galaxy_data])

# Standardize
X_mean = X_quality.mean(axis=0)
X_std = X_quality.std(axis=0)
X_std[X_std == 0] = 1
X_norm = (X_quality - X_mean) / X_std

# Simple logistic regression (no sklearn dependency)
def logistic_nll(beta, X, y):
    z = X @ beta[1:] + beta[0]
    z = np.clip(z, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

# Fit quality-only model
res_q = minimize(logistic_nll, np.zeros(5), args=(X_norm, y_periodic), method='BFGS')
nll_quality = res_q.fun

# Null model (intercept only)
def nll_null(beta0, y):
    z = beta0[0]
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

res_null = minimize(nll_null, [0.0], args=(y_periodic,), method='BFGS')
nll_null_val = res_null.fun

# McFadden R²
R2_quality = 1 - nll_quality / nll_null_val
print(f"  Quality-only logistic model: McFadden R² = {R2_quality:.3f}")

# Add physical properties
X_phys = np.array([
    [np.log10(g['Vflat']) if g['Vflat'] > 0 else 0,
     np.log10(g['M_bar_solar']) if g['M_bar_solar'] > 0 else 0,
     g['f_gas']]
    for g in galaxy_data
])
X_full = np.hstack([X_norm, (X_phys - X_phys.mean(axis=0)) / np.maximum(X_phys.std(axis=0), 1e-10)])

res_full = minimize(logistic_nll, np.zeros(X_full.shape[1] + 1),
                    args=(X_full, y_periodic), method='BFGS')
nll_full = res_full.fun
R2_full = 1 - nll_full / nll_null_val

# LR test: full vs quality-only
lr_stat = 2 * (nll_quality - nll_full)
lr_df = X_full.shape[1] - X_norm.shape[1]
lr_p = float(stats.chi2.sf(lr_stat, lr_df))

print(f"  Quality+physics logistic model: McFadden R² = {R2_full:.3f}")
print(f"  LR test (physics improves over quality): chi2 = {lr_stat:.2f}, "
      f"df = {lr_df}, p = {lr_p:.4f}")
if lr_p < 0.05:
    print(f"  => Physical properties SIGNIFICANTLY improve prediction beyond data quality")
else:
    print(f"  => Physical properties do NOT significantly improve over quality alone")

# Also: partial correlation of Vflat with periodicity, controlling for N_pts + R_extent
# Use point-biserial correlation, then partial out quality
from scipy.stats import spearmanr

# Rank-based partial correlation
def partial_spearman(x, y, z_arr):
    """Partial Spearman: ρ(x,y | z) via residual ranks."""
    # Regress x on z, y on z (rank-based)
    ranks_x = stats.rankdata(x)
    ranks_y = stats.rankdata(y)
    resid_x = ranks_x.copy()
    resid_y = ranks_y.copy()
    for z in z_arr.T:
        ranks_z = stats.rankdata(z)
        # Linear regression of rank(x) on rank(z)
        slope_x = np.cov(resid_x, ranks_z)[0, 1] / np.var(ranks_z)
        resid_x = resid_x - slope_x * ranks_z
        slope_y = np.cov(resid_y, ranks_z)[0, 1] / np.var(ranks_z)
        resid_y = resid_y - slope_y * ranks_z
    rho, p = spearmanr(resid_x, resid_y)
    return rho, p

vflat_arr = np.array([np.log10(g['Vflat']) if g['Vflat'] > 0 else 0 for g in galaxy_data])
quality_arr = np.column_stack([
    np.array([g['n_pts'] for g in galaxy_data]),
    np.array([g['R_extent'] for g in galaxy_data]),
    np.array([g['median_errV'] for g in galaxy_data]),
])

rho_vflat_raw, p_vflat_raw = spearmanr(vflat_arr, y_periodic)
rho_vflat_partial, p_vflat_partial = partial_spearman(vflat_arr, y_periodic, quality_arr)

print(f"\n  Spearman ρ(log Vflat, periodic):")
print(f"    Raw:     ρ = {rho_vflat_raw:.3f}, p = {p_vflat_raw:.4f}")
print(f"    Partial (controlling quality): ρ = {rho_vflat_partial:.3f}, p = {p_vflat_partial:.4f}")


# ================================================================
# PART 2: λ_peak vs ξ SCALING
# ================================================================
print("\n" + "=" * 76)
print("PART 2: λ_peak vs ξ SCALING")
print("=" * 76)

# Use only periodic galaxies
wl_per = np.array([g['wl_peak'] for g in periodic])
xi_per = np.array([g['xi_kpc'] for g in periodic])

log_wl = np.log10(wl_per)
log_xi = np.log10(xi_per)

# Spearman correlation
rho_sp, p_sp = spearmanr(log_xi, log_wl)
print(f"\n  Spearman ρ(log ξ, log λ_peak) = {rho_sp:.3f}, p = {p_sp:.4f}")

# Linear fit: log λ = a + b * log ξ
slope, intercept, r_val, p_ols, se_slope = stats.linregress(log_xi, log_wl)
print(f"  OLS fit: log λ = {intercept:.3f} + {slope:.3f} * log ξ")
print(f"    slope b = {slope:.3f} ± {se_slope:.3f}")
print(f"    r² = {r_val**2:.3f}, p = {p_ols:.4f}")

# Bootstrap slope uncertainty
n_boot = 5000
boot_slopes = np.zeros(n_boot)
boot_rng = np.random.default_rng(456)
for b in range(n_boot):
    idx = boot_rng.choice(n_per, size=n_per, replace=True)
    s, _, _, _, _ = stats.linregress(log_xi[idx], log_wl[idx])
    boot_slopes[b] = s

slope_boot_lo, slope_boot_hi = np.percentile(boot_slopes, [2.5, 97.5])
print(f"    Bootstrap 95% CI for slope: [{slope_boot_lo:.3f}, {slope_boot_hi:.3f}]")

# Is slope consistent with 1?
z_b1 = abs(slope - 1.0) / se_slope if se_slope > 0 else np.inf
print(f"    |z| vs b=1: {z_b1:.2f} ({'consistent' if z_b1 < 2 else 'rejected'} at 2σ)")

# λ/ξ ratio distribution
ratio_lam_xi = wl_per / xi_per
log_ratio = np.log10(ratio_lam_xi)
print(f"\n  λ_peak / ξ distribution:")
print(f"    Median: {np.median(ratio_lam_xi):.2f}")
print(f"    Mean:   {np.mean(ratio_lam_xi):.2f}")
print(f"    Scatter (std of log ratio): {np.std(log_ratio):.3f} dex")
print(f"    IQR: [{np.percentile(ratio_lam_xi, 25):.2f}, {np.percentile(ratio_lam_xi, 75):.2f}]")

# Control: same for non-periodic galaxies
wl_nper = np.array([g['wl_peak'] for g in nonperiodic])
xi_nper = np.array([g['xi_kpc'] for g in nonperiodic])
rho_nper, p_nper = spearmanr(np.log10(xi_nper), np.log10(wl_nper))
print(f"\n  Control — non-periodic galaxies:")
print(f"    Spearman ρ(log ξ, log λ_peak) = {rho_nper:.3f}, p = {p_nper:.4f}")

# Confound check: is λ vs ξ driven by R_extent?
R_ext_per = np.array([g['R_extent'] for g in periodic])

rho_wl_R, p_wl_R = spearmanr(log_wl, np.log10(R_ext_per))
rho_xi_R, p_xi_R = spearmanr(log_xi, np.log10(R_ext_per))

print(f"\n  Confound check:")
print(f"    ρ(log λ, log R_ext) = {rho_wl_R:.3f} (p={p_wl_R:.4f})")
print(f"    ρ(log ξ, log R_ext) = {rho_xi_R:.3f} (p={p_xi_R:.4f})")

# Partial correlation: λ vs ξ controlling for R_extent
rho_partial_R, p_partial_R = partial_spearman(
    log_wl, log_xi,
    np.column_stack([np.log10(R_ext_per), np.array([g['n_pts'] for g in periodic])])
)
print(f"    Partial ρ(log λ, log ξ | log R_ext, N_pts) = {rho_partial_R:.3f} (p={p_partial_R:.4f})")


# ================================================================
# PART 3: DIMENSIONLESS BUMP SHARPENING
# ================================================================
print("\n" + "=" * 76)
print("PART 3: DIMENSIONLESS BUMP SHARPENING")
print("=" * 76)

N_FREQ = 200
k_common = np.logspace(np.log10(0.02), np.log10(3.0), N_FREQ)
N_DIM = 200
k_dim_common = np.logspace(np.log10(0.1), np.log10(30.0), N_DIM)


def stack_psds(galaxies, k_phys, k_dim, label=""):
    """Stack PSDs in both physical and dimensionless coordinates."""
    n_gal = len(galaxies)
    mat_phys = np.array([g['psd_common'] for g in galaxies])

    # Dimensionless mapping
    mat_dim = np.full((n_gal, len(k_dim)), np.nan)
    for gi, g in enumerate(galaxies):
        xi = g['xi_kpc']
        psd = g['psd_common']
        k_dim_this = k_phys * xi
        valid = ~np.isnan(psd) & (psd > 0)
        if np.sum(valid) < 5:
            continue
        log_kd = np.log10(k_dim_this[valid])
        log_psd = np.log10(psd[valid])
        in_range = (np.log10(k_dim) >= log_kd.min()) & (np.log10(k_dim) <= log_kd.max())
        if np.sum(in_range) < 3:
            continue
        mat_dim[gi, in_range] = 10**np.interp(np.log10(k_dim[in_range]), log_kd, log_psd)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_phys = np.nanmedian(mat_phys, axis=0)
        p16_phys = np.nanpercentile(mat_phys, 16, axis=0)
        p84_phys = np.nanpercentile(mat_phys, 84, axis=0)
        n_phys = np.sum(~np.isnan(mat_phys), axis=0)

        med_dim = np.nanmedian(mat_dim, axis=0)
        p16_dim = np.nanpercentile(mat_dim, 16, axis=0)
        p84_dim = np.nanpercentile(mat_dim, 84, axis=0)
        n_dim = np.sum(~np.isnan(mat_dim), axis=0)

    return {
        'phys': {'median': med_phys, 'p16': p16_phys, 'p84': p84_phys, 'n': n_phys},
        'dim': {'median': med_dim, 'p16': p16_dim, 'p84': p84_dim, 'n': n_dim},
    }


# Stack periodic and non-periodic separately
print("\n[3a] Stacking periodic and non-periodic PSDs...")
stk_per = stack_psds(periodic, k_common, k_dim_common, "Periodic")
stk_nper = stack_psds(nonperiodic, k_common, k_dim_common, "Non-periodic")


# Fit + subtract power law from periodic stacked PSD, then measure bump
def measure_bump(k_arr, psd_median, n_arr, min_n=5, label=""):
    """Fit power law and measure bump properties (excess over fit)."""
    good = (n_arr >= min_n) & (psd_median > 0) & np.isfinite(psd_median)
    if np.sum(good) < 8:
        print(f"  [{label}] Too few valid points ({np.sum(good)})")
        return None

    log_k = np.log10(k_arr[good])
    log_psd = np.log10(psd_median[good])

    # Fit power law
    try:
        popt, pcov = curve_fit(power_law, log_k, log_psd, p0=[0.0, 0.5])
    except Exception:
        print(f"  [{label}] Power-law fit failed")
        return None

    beta = popt[1]
    predicted = power_law(log_k, *popt)
    excess = log_psd - predicted  # positive = bump

    # Bump detection: is there a coherent positive region?
    if np.max(excess) < 0.01:
        print(f"  [{label}] No bump (max excess {np.max(excess):.4f} dex)")
        return {
            'beta': float(beta),
            'bump_detected': False,
            'max_excess': float(np.max(excess)),
        }

    # Find peak
    idx_peak = np.argmax(excess)
    k_peak = 10**log_k[idx_peak]
    peak_excess = excess[idx_peak]

    # FWHM in log-k space
    half_max = peak_excess / 2.0
    above_half = excess >= half_max
    if np.sum(above_half) >= 2:
        above_indices = np.where(above_half)[0]
        fwhm_log_k = log_k[above_indices[-1]] - log_k[above_indices[0]]
    else:
        fwhm_log_k = np.nan

    # Second moment around peak in log-k
    # Weight by excess (only positive parts)
    pos = excess > 0
    if np.sum(pos) >= 3:
        weights = excess[pos]
        weighted_mean = np.average(log_k[pos], weights=weights)
        second_moment = np.sqrt(np.average((log_k[pos] - weighted_mean)**2, weights=weights))
    else:
        second_moment = np.nan

    print(f"  [{label}] beta={beta:.3f}, peak at k={k_peak:.3f}, "
          f"excess={peak_excess:.3f} dex, FWHM={fwhm_log_k:.3f} dex, "
          f"σ_bump={second_moment:.3f} dex")

    return {
        'beta': float(beta),
        'bump_detected': True,
        'k_peak': float(k_peak),
        'wl_peak': float(1.0 / k_peak) if k_peak > 0 else None,
        'peak_excess_dex': float(peak_excess),
        'fwhm_log_k': float(fwhm_log_k) if not np.isnan(fwhm_log_k) else None,
        'second_moment_log_k': float(second_moment) if not np.isnan(second_moment) else None,
        'excess_curve_log_k': log_k.tolist(),
        'excess_curve_vals': excess.tolist(),
    }


print("\n[3b] Measuring bump in periodic PSDs...")

print("  Physical coordinates:")
bump_per_phys = measure_bump(k_common, stk_per['phys']['median'],
                              stk_per['phys']['n'], label="Periodic-Physical")

print("  Dimensionless coordinates:")
bump_per_dim = measure_bump(k_dim_common, stk_per['dim']['median'],
                             stk_per['dim']['n'], label="Periodic-Dimensionless")

# Compare FWHM
if (bump_per_phys and bump_per_dim and
        bump_per_phys.get('fwhm_log_k') and bump_per_dim.get('fwhm_log_k')):
    fwhm_phys = bump_per_phys['fwhm_log_k']
    fwhm_dim = bump_per_dim['fwhm_log_k']
    print(f"\n  FWHM comparison:")
    print(f"    Physical:      {fwhm_phys:.3f} dex")
    print(f"    Dimensionless: {fwhm_dim:.3f} dex")
    ratio_fwhm = fwhm_dim / fwhm_phys if fwhm_phys > 0 else np.inf
    print(f"    Ratio (dim/phys): {ratio_fwhm:.3f}")
    if ratio_fwhm < 1:
        print(f"    => Bump SHARPENS in k×ξ — tied to healing length")
    else:
        print(f"    => Bump does NOT sharpen — may be a fixed physical scale")

if (bump_per_phys and bump_per_dim and
        bump_per_phys.get('second_moment_log_k') and bump_per_dim.get('second_moment_log_k')):
    sm_phys = bump_per_phys['second_moment_log_k']
    sm_dim = bump_per_dim['second_moment_log_k']
    print(f"  Second moment comparison:")
    print(f"    Physical:      {sm_phys:.3f} dex")
    print(f"    Dimensionless: {sm_dim:.3f} dex")
    ratio_sm = sm_dim / sm_phys if sm_phys > 0 else np.inf
    print(f"    Ratio (dim/phys): {ratio_sm:.3f}")


# Control #4: non-periodic in k×ξ
print("\n[3c] Control: non-periodic PSDs in k×ξ (does a bump emerge when aligned?)...")

print("  Non-periodic, physical:")
bump_nper_phys = measure_bump(k_common, stk_nper['phys']['median'],
                               stk_nper['phys']['n'], label="NonPer-Physical")

print("  Non-periodic, dimensionless:")
bump_nper_dim = measure_bump(k_dim_common, stk_nper['dim']['median'],
                              stk_nper['dim']['n'], label="NonPer-Dimensionless")


# ================================================================
# PUBLICATION FIGURES
# ================================================================
print("\n[4] Generating publication-quality figures...")

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})


# --- Figure 1: Property comparison forest plot ---
fig1, ax1 = plt.subplots(figsize=(8, 7))

phys_results = [r for r in raw_results if r['category'] == 'physical' and 'note' not in r]
phys_results_sorted = sorted(phys_results, key=lambda x: x['MW_p'])

y_pos = np.arange(len(phys_results_sorted))
colors = ['firebrick' if r['significant_FDR'] else 'steelblue' for r in phys_results_sorted]
r_vals = [r['r_biserial'] for r in phys_results_sorted]
labels = [r['property'] for r in phys_results_sorted]

ax1.barh(y_pos, r_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=10)
ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_xlabel('Rank-biserial correlation r', fontsize=12)
ax1.set_title('Periodic vs Non-Periodic: Physical Properties\n(red = significant after BH-FDR)', fontsize=13)

# Add p-values as text
for i, r in enumerate(phys_results_sorted):
    p_text = f"p={r['MW_p_adjusted']:.3f}"
    x_pos = r['r_biserial']
    ha = 'left' if x_pos >= 0 else 'right'
    offset = 0.01 if x_pos >= 0 else -0.01
    ax1.text(x_pos + offset, i, p_text, va='center', ha=ha, fontsize=8,
             fontweight='bold' if r['significant_FDR'] else 'normal')

fig1.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'periodic_property_comparison.png')
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig1_path}")
plt.close(fig1)


# --- Figure 2: λ vs ξ scaling ---
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Panel a: log-log scatter
ax = axes2[0]
ax.scatter(xi_per, wl_per, c='crimson', s=50, alpha=0.7, edgecolors='black',
           linewidth=0.5, zorder=5, label=f'Periodic (N={n_per})')
ax.scatter(xi_nper, wl_nper, c='royalblue', s=30, alpha=0.4, edgecolors='gray',
           linewidth=0.3, zorder=3, label=f'Non-periodic (N={n_nper})')

# Fit line (periodic only)
xi_line = np.logspace(np.log10(xi_per.min()) - 0.2, np.log10(xi_per.max()) + 0.2, 50)
wl_fit = 10**(intercept + slope * np.log10(xi_line))
ax.plot(xi_line, wl_fit, '--', color='crimson', linewidth=1.5,
        label=rf'$b = {slope:.2f} \pm {se_slope:.2f}$')

# 1:1 line
ax.plot([0.3, 100], [0.3, 100], ':', color='gray', linewidth=0.8, label=r'$\lambda = \xi$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Healing length $\xi = \sqrt{GM/g^\dagger}$ [kpc]', fontsize=11)
ax.set_ylabel(r'Peak wavelength $\lambda_{\rm peak}$ [kpc]', fontsize=11)
ax.set_title(rf'$\rho_s = {rho_sp:.2f}$, $p = {p_sp:.3f}$', fontsize=12)
ax.legend(fontsize=8, loc='upper left')

# Panel b: λ/ξ histogram
ax = axes2[1]
ax.hist(ratio_lam_xi, bins=12, color='crimson', alpha=0.6, edgecolor='black', linewidth=0.5)
ax.axvline(np.median(ratio_lam_xi), color='firebrick', linestyle='--', linewidth=1.5,
           label=f'Median = {np.median(ratio_lam_xi):.2f}')
ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.0, label=r'$\lambda/\xi = 1$')
ax.set_xlabel(r'$\lambda_{\rm peak} / \xi$', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(r'$\lambda/\xi$ Distribution (Periodic Galaxies)', fontsize=12)
ax.legend(fontsize=9)

# Panel c: λ/ξ vs Vflat (looking for secondary trends)
ax = axes2[2]
vflat_per = np.array([g['Vflat'] for g in periodic])
valid_vf = vflat_per > 0
ax.scatter(vflat_per[valid_vf], ratio_lam_xi[valid_vf], c='crimson', s=50, alpha=0.7,
           edgecolors='black', linewidth=0.5)
rho_ratio_vf, p_ratio_vf = spearmanr(vflat_per[valid_vf], ratio_lam_xi[valid_vf])
ax.set_xlabel(r'$V_{\rm flat}$ [km/s]', fontsize=11)
ax.set_ylabel(r'$\lambda_{\rm peak} / \xi$', fontsize=11)
ax.set_title(rf'$\rho_s = {rho_ratio_vf:.2f}$, $p = {p_ratio_vf:.3f}$', fontsize=12)
ax.axhline(np.median(ratio_lam_xi), color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

fig2.suptitle(r'$\lambda_{\rm peak}$ vs Healing Length $\xi$', fontsize=14, y=1.02)
fig2.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'lambda_xi_scaling.png')
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig2_path}")
plt.close(fig2)


# --- Figure 3: Bump sharpening ---
fig3, axes3 = plt.subplots(2, 2, figsize=(13, 10))

# Top-left: Periodic PSD in physical k
ax = axes3[0, 0]
gp = (stk_per['phys']['n'] >= 5) & (stk_per['phys']['median'] > 0) & np.isfinite(stk_per['phys']['median'])
if np.sum(gp) > 3:
    ax.fill_between(k_common[gp], stk_per['phys']['p16'][gp], stk_per['phys']['p84'][gp],
                    alpha=0.2, color='crimson')
    ax.plot(k_common[gp], stk_per['phys']['median'][gp], 'o-', color='crimson',
            markersize=2, linewidth=1.2, label=f'Periodic (N={n_per})')
    if bump_per_phys and bump_per_phys.get('bump_detected'):
        log_k_fit = np.log10(k_common[gp])
        psd_pl = 10**(power_law(log_k_fit, bump_per_phys['beta'] * 0 +
                                np.log10(stk_per['phys']['median'][gp]).mean() +
                                bump_per_phys['beta'] * np.log10(k_common[gp]).mean(),
                                bump_per_phys['beta']))
        # Just show the fit line properly
        popt_temp, _ = curve_fit(power_law, np.log10(k_common[gp]),
                                  np.log10(stk_per['phys']['median'][gp]), p0=[0, 0.5])
        psd_pl = 10**(power_law(np.log10(k_common[gp]), *popt_temp))
        ax.plot(k_common[gp], psd_pl, '--', color='gray', linewidth=1.0,
                label=rf'Power law ($\beta={popt_temp[1]:.2f}$)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k$ [kpc$^{-1}$]', fontsize=11)
ax.set_ylabel(r'$P(k)$', fontsize=11)
ax.set_title('Periodic — Physical k', fontsize=12)
ax.legend(fontsize=8)

# Top-right: Periodic PSD in k×ξ
ax = axes3[0, 1]
gd = (stk_per['dim']['n'] >= 5) & (stk_per['dim']['median'] > 0) & np.isfinite(stk_per['dim']['median'])
if np.sum(gd) > 3:
    ax.fill_between(k_dim_common[gd], stk_per['dim']['p16'][gd], stk_per['dim']['p84'][gd],
                    alpha=0.2, color='darkorange')
    ax.plot(k_dim_common[gd], stk_per['dim']['median'][gd], 'o-', color='darkorange',
            markersize=2, linewidth=1.2, label=f'Periodic (N={n_per})')
    if bump_per_dim and bump_per_dim.get('bump_detected'):
        popt_temp2, _ = curve_fit(power_law, np.log10(k_dim_common[gd]),
                                   np.log10(stk_per['dim']['median'][gd]), p0=[0, 0.5])
        psd_pl2 = 10**(power_law(np.log10(k_dim_common[gd]), *popt_temp2))
        ax.plot(k_dim_common[gd], psd_pl2, '--', color='gray', linewidth=1.0,
                label=rf'Power law ($\beta={popt_temp2[1]:.2f}$)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k \cdot \xi$', fontsize=11)
ax.set_ylabel(r'$P(k\xi)$', fontsize=11)
ax.set_title(r'Periodic — Dimensionless $k\xi$', fontsize=12)
ax.legend(fontsize=8)

# FWHM annotations
if (bump_per_phys and bump_per_dim and
        bump_per_phys.get('fwhm_log_k') and bump_per_dim.get('fwhm_log_k')):
    axes3[0, 0].text(0.03, 0.05, f"FWHM = {bump_per_phys['fwhm_log_k']:.3f} dex",
                     transform=axes3[0, 0].transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    axes3[0, 1].text(0.03, 0.05, f"FWHM = {bump_per_dim['fwhm_log_k']:.3f} dex",
                     transform=axes3[0, 1].transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Bottom-left: Excess curves (periodic, both coords)
ax = axes3[1, 0]
if bump_per_phys and bump_per_phys.get('excess_curve_log_k'):
    lk = np.array(bump_per_phys['excess_curve_log_k'])
    exc = np.array(bump_per_phys['excess_curve_vals'])
    ax.plot(10**lk, exc, 'o-', color='crimson', markersize=3, linewidth=1.2, label='Physical k')
if bump_per_dim and bump_per_dim.get('excess_curve_log_k'):
    lk = np.array(bump_per_dim['excess_curve_log_k'])
    exc = np.array(bump_per_dim['excess_curve_vals'])
    ax.plot(10**lk, exc, 's-', color='darkorange', markersize=3, linewidth=1.2, label=r'Dimensionless $k\xi$')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xscale('log')
ax.set_xlabel(r'Frequency (physical or dimensionless)', fontsize=11)
ax.set_ylabel('Excess over power law [dex]', fontsize=11)
ax.set_title('Bump Excess — Periodic Galaxies', fontsize=12)
ax.legend(fontsize=9)

# Bottom-right: Non-periodic control in k×ξ
ax = axes3[1, 1]
gd_np = (stk_nper['dim']['n'] >= 5) & (stk_nper['dim']['median'] > 0) & np.isfinite(stk_nper['dim']['median'])
if np.sum(gd_np) > 3:
    ax.fill_between(k_dim_common[gd_np], stk_nper['dim']['p16'][gd_np], stk_nper['dim']['p84'][gd_np],
                    alpha=0.2, color='royalblue')
    ax.plot(k_dim_common[gd_np], stk_nper['dim']['median'][gd_np], 'o-', color='royalblue',
            markersize=2, linewidth=1.2, label=f'Non-periodic (N={n_nper})')
    if bump_nper_dim and bump_nper_dim.get('bump_detected'):
        ax.text(0.03, 0.05, f"Bump: {bump_nper_dim['peak_excess_dex']:.3f} dex",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax.text(0.03, 0.05, "No bump detected",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k \cdot \xi$', fontsize=11)
ax.set_ylabel(r'$P(k\xi)$', fontsize=11)
ax.set_title(r'Control: Non-Periodic in $k\xi$', fontsize=12)
ax.legend(fontsize=8)

fig3.suptitle('Dimensionless Bump Sharpening Test', fontsize=14)
fig3.tight_layout(rect=[0, 0, 1, 0.96])
fig3_path = os.path.join(FIGURES_DIR, 'bump_sharpening.png')
fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig3_path}")
plt.close(fig3)


# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 76)
print("FINAL SUMMARY — TESTS B.1-B.3")
print("=" * 76)

print(f"\n  PART 1 — Property Comparison:")
print(f"    Raw: {n_sig_raw} significant (FDR) out of {len(raw_results)} properties")
print(f"      Physical: {n_sig_physical}, Quality: {n_sig_quality}")
print(f"    After matching on quality: {n_matched_sig} physical properties remain significant")
print(f"    Logistic regression: quality R²={R2_quality:.3f}, full R²={R2_full:.3f}, "
      f"LR p={lr_p:.4f}")
print(f"    Partial ρ(Vflat, periodic | quality) = {rho_vflat_partial:.3f}, p={p_vflat_partial:.4f}")

print(f"\n  PART 2 — λ_peak vs ξ Scaling:")
print(f"    Spearman ρ = {rho_sp:.3f} (p = {p_sp:.4f})")
print(f"    Slope b = {slope:.3f} ± {se_slope:.3f} (95% CI: [{slope_boot_lo:.3f}, {slope_boot_hi:.3f}])")
print(f"    b vs 1.0: |z| = {z_b1:.2f}")
print(f"    λ/ξ median = {np.median(ratio_lam_xi):.2f}, scatter = {np.std(log_ratio):.3f} dex")
print(f"    Partial ρ(λ, ξ | R_ext, N_pts) = {rho_partial_R:.3f} (p = {p_partial_R:.4f})")

print(f"\n  PART 3 — Bump Sharpening:")
if (bump_per_phys and bump_per_dim):
    fwhm_p = bump_per_phys.get('fwhm_log_k', None)
    fwhm_d = bump_per_dim.get('fwhm_log_k', None)
    sm_p = bump_per_phys.get('second_moment_log_k', None)
    sm_d = bump_per_dim.get('second_moment_log_k', None)
    if fwhm_p and fwhm_d:
        print(f"    FWHM: physical={fwhm_p:.3f}, dimensionless={fwhm_d:.3f}, "
              f"ratio={fwhm_d/fwhm_p:.3f}")
    if sm_p and sm_d:
        print(f"    σ_bump: physical={sm_p:.3f}, dimensionless={sm_d:.3f}, "
              f"ratio={sm_d/sm_p:.3f}")
    if bump_nper_dim:
        print(f"    Non-periodic control: bump detected = {bump_nper_dim.get('bump_detected', False)}")


# ================================================================
# SAVE RESULTS
# ================================================================

results = {
    'test': 'periodic_properties_and_scaling',
    'description': 'Three discriminating tests: property comparison, lambda-xi scaling, bump sharpening.',
    'sample': {
        'n_total': n_total,
        'n_periodic': n_per,
        'n_nonperiodic': n_nper,
    },
    'part1_property_comparison': {
        'raw_results': [{k: v for k, v in r.items()} for r in raw_results],
        'matched_results': [{k: v for k, v in r.items()} for r in matched_results],
        'n_significant_raw_FDR': n_sig_raw,
        'n_significant_physical': n_sig_physical,
        'n_significant_quality': n_sig_quality,
        'n_significant_matched': n_matched_sig,
        'logistic_quality_R2': round(R2_quality, 4),
        'logistic_full_R2': round(R2_full, 4),
        'logistic_LR_p': round(lr_p, 4),
        'partial_rho_Vflat': round(rho_vflat_partial, 4),
        'partial_p_Vflat': round(p_vflat_partial, 4),
    },
    'part2_lambda_xi_scaling': {
        'spearman_rho': round(float(rho_sp), 4),
        'spearman_p': round(float(p_sp), 4),
        'ols_slope': round(float(slope), 4),
        'ols_slope_err': round(float(se_slope), 4),
        'ols_intercept': round(float(intercept), 4),
        'ols_r2': round(float(r_val**2), 4),
        'bootstrap_slope_95CI': [round(float(slope_boot_lo), 4), round(float(slope_boot_hi), 4)],
        'z_vs_slope1': round(float(z_b1), 2),
        'lambda_over_xi_median': round(float(np.median(ratio_lam_xi)), 3),
        'lambda_over_xi_scatter_dex': round(float(np.std(log_ratio)), 3),
        'partial_rho_controlling_R_Npts': round(float(rho_partial_R), 4),
        'partial_p_controlling_R_Npts': round(float(p_partial_R), 4),
        'control_nonperiodic_rho': round(float(rho_nper), 4),
        'control_nonperiodic_p': round(float(p_nper), 4),
        'per_galaxy': [
            {
                'name': g['name'],
                'lambda_peak_kpc': round(g['wl_peak'], 2),
                'xi_kpc': round(g['xi_kpc'], 2),
                'lambda_over_xi': round(g['wl_peak'] / g['xi_kpc'], 3),
                'Vflat': g['Vflat'],
                'M_bar_solar': round(g['M_bar_solar'], 0),
            }
            for g in periodic
        ],
    },
    'part3_bump_sharpening': {
        'periodic_physical': {k: v for k, v in bump_per_phys.items()
                              if k not in ('excess_curve_log_k', 'excess_curve_vals')}
            if bump_per_phys else None,
        'periodic_dimensionless': {k: v for k, v in bump_per_dim.items()
                                    if k not in ('excess_curve_log_k', 'excess_curve_vals')}
            if bump_per_dim else None,
        'nonperiodic_physical_control': {k: v for k, v in bump_nper_phys.items()
                                          if k not in ('excess_curve_log_k', 'excess_curve_vals')}
            if bump_nper_phys else None,
        'nonperiodic_dimensionless_control': {k: v for k, v in bump_nper_dim.items()
                                               if k not in ('excess_curve_log_k', 'excess_curve_vals')}
            if bump_nper_dim else None,
    },
    'figures': [
        'periodic_property_comparison.png',
        'lambda_xi_scaling.png',
        'bump_sharpening.png',
    ],
}

outpath = os.path.join(RESULTS_DIR, 'summary_periodic_properties_and_scaling.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {outpath}")

print("\n" + "=" * 76)
print("ALL THREE TESTS COMPLETE")
print("=" * 76)
