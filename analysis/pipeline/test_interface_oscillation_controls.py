#!/usr/bin/env python3
"""
Interface Oscillation Controls (Step 26) — Detrending + Permutation Nulls
=========================================================================

Three controls to determine whether the positive autocorrelation in RAR
residuals reflects oscillatory radial structure vs smooth drift:

A) First-difference control
   d = np.diff(eps)  — removes any smooth monotonic trend
   Compute lag-1 and lag-2 on d.

B) Trend-removal control (within galaxy)
   Fit eps ~ UnivariateSpline(R, s=...) with moderate smoothing
   Compute autocorr on residuals from that smooth trend.

C) Radius-order permutation null (within galaxy)
   For each galaxy, shuffle residual order 5000 times.
   Per-galaxy p-value, Fisher combined p, and aggregate z-score.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics constants
g_dagger = 1.20e-10   # m/s^2
kpc_m = 3.086e19       # m per kpc
LOG_G_DAGGER = np.log10(g_dagger)

MIN_POINTS = 15
N_BOOT = 10000
N_PERM = 5000

np.random.seed(42)

print("=" * 72)
print("INTERFACE OSCILLATION CONTROLS (Step 26)")
print("  Detrending + Permutation Nulls")
print("=" * 72)
print(f"  Minimum points per galaxy: {MIN_POINTS}")
print(f"  Bootstrap resamples: {N_BOOT}")
print(f"  Permutations per galaxy: {N_PERM}")


# ================================================================
# ENVIRONMENT CLASSIFICATION
# ================================================================
UMA_GALAXIES = {
    'NGC3726', 'NGC3769', 'NGC3877', 'NGC3893', 'NGC3917',
    'NGC3949', 'NGC3953', 'NGC3972', 'NGC3992', 'NGC4010',
    'NGC4013', 'NGC4051', 'NGC4085', 'NGC4088', 'NGC4100',
    'NGC4138', 'NGC4157', 'NGC4183', 'NGC4217',
    'UGC06399', 'UGC06446', 'UGC06667', 'UGC06786', 'UGC06787',
    'UGC06818', 'UGC06917', 'UGC06923', 'UGC06930', 'UGC06973',
    'UGC06983', 'UGC07089',
}

GROUP_MEMBERS = {
    'NGC2403': 'M81', 'NGC2976': 'M81', 'IC2574': 'M81',
    'DDO154': 'M81', 'DDO168': 'M81', 'UGC04483': 'M81',
    'NGC0300': 'Sculptor', 'NGC0055': 'Sculptor',
    'NGC0247': 'Sculptor', 'NGC7793': 'Sculptor',
    'NGC2915': 'CenA', 'UGCA442': 'CenA', 'ESO444-G084': 'CenA',
    'UGC07577': 'CVnI', 'UGC07232': 'CVnI', 'NGC3741': 'CVnI',
    'NGC4068': 'CVnI', 'UGC07866': 'CVnI', 'UGC07524': 'CVnI',
    'UGC08490': 'CVnI', 'UGC07559': 'CVnI',
    'NGC3109': 'Antlia', 'NGC5055': 'M101',
}


def classify_env(name):
    if name in UMA_GALAXIES or name in GROUP_MEMBERS:
        return 'dense'
    return 'field'


# ================================================================
# RAR FUNCTIONS
# ================================================================
def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def lag_autocorrelation(x, lag=1):
    n = len(x)
    if n <= lag + 1:
        return np.nan
    xbar = np.mean(x)
    var = np.var(x, ddof=0)
    if var < 1e-30:
        return np.nan
    cov = np.mean((x[:n - lag] - xbar) * (x[lag:] - xbar))
    return cov / var


# ================================================================
# STATS HELPERS
# ================================================================
def compute_autocorr_stats(values, label, rng_seed=42):
    n = len(values)
    if n < 3:
        return None
    mean_val = float(np.mean(values))
    se_val = float(np.std(values, ddof=1) / np.sqrt(n))
    t_stat, p_two = stats.ttest_1samp(values, 0.0)
    p_one = float(p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0)
    boot_rng = np.random.default_rng(rng_seed)
    boot_means = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = boot_rng.integers(0, n, size=n)
        boot_means[b] = np.mean(values[idx])
    ci = np.percentile(boot_means, [2.5, 97.5])
    n_pos = int(np.sum(values > 0))
    frac_pos = float(n_pos / n)
    binom_p = float(stats.binomtest(n_pos, n, 0.5, alternative='greater').pvalue)
    return {
        'mean': round(mean_val, 4),
        'se': round(se_val, 4),
        'median': round(float(np.median(values)), 4),
        'std': round(float(np.std(values, ddof=1)), 4),
        'ci_95_lower': round(float(ci[0]), 4),
        'ci_95_upper': round(float(ci[1]), 4),
        'ci_excludes_zero': bool(ci[0] > 0),
        'ttest_t': round(float(t_stat), 3),
        'ttest_p_one_sided': p_one,
        'frac_positive': round(frac_pos, 3),
        'n_positive': n_pos,
        'n_total': n,
        'binom_p': binom_p,
    }


def compute_split_stats(arr_a, arr_b, label_a, label_b):
    if len(arr_a) < 5 or len(arr_b) < 5:
        return None
    t_val, p_welch = stats.ttest_ind(arr_a, arr_b, equal_var=False)
    _, p_mwu = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
    return {
        f'{label_a}_mean': round(float(np.mean(arr_a)), 4),
        f'{label_b}_mean': round(float(np.mean(arr_b)), 4),
        f'{label_a}_se': round(float(np.std(arr_a, ddof=1) / np.sqrt(len(arr_a))), 4),
        f'{label_b}_se': round(float(np.std(arr_b, ddof=1) / np.sqrt(len(arr_b))), 4),
        'welch_t': round(float(t_val), 3),
        'welch_p': round(float(p_welch), 4),
        'mwu_p': round(float(p_mwu), 4),
        f'n_{label_a}': len(arr_a),
        f'n_{label_b}': len(arr_b),
    }


# ================================================================
# 1. LOAD SPARC DATA
# ================================================================
print("\n[1] Loading SPARC data...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

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
            evobs = float(line[33:38].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in rc_data:
            rc_data[name] = {'R': [], 'Vobs': [], 'eVobs': [],
                             'Vgas': [], 'Vdisk': [], 'Vbul': [],
                             'dist': dist}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['eVobs'].append(evobs)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in ['R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul']:
        rc_data[name][key] = np.array(rc_data[name][key])

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
        sparc_props[name] = {
            'T': int(parts[0]),
            'D': float(parts[1]),
            'eD': float(parts[2]),
            'fD': int(parts[3]),
            'Inc': float(parts[4]),
            'eInc': float(parts[5]),
            'L36': float(parts[6]),
            'Vflat': float(parts[14]),
            'eVflat': float(parts[15]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

print(f"  Loaded {len(rc_data)} galaxies with rotation curves")
print(f"  Loaded {len(sparc_props)} galaxies with properties")


# ================================================================
# 2. COMPUTE PER-GALAXY RAR RESIDUALS
# ================================================================
print("\n[2] Computing per-galaxy RAR residuals (radius-ordered)...")

galaxy_data = []
n_rejected = 0

for name, gdata in rc_data.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

    R = gdata['R']
    Vobs = gdata['Vobs']
    Vgas = gdata['Vgas']
    Vdisk = gdata['Vdisk']
    Vbul = gdata['Vbul']

    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < MIN_POINTS:
        n_rejected += 1
        continue

    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]

    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]

    log_gobs_pred = rar_function(log_gbar)
    residuals = log_gobs - log_gobs_pred

    env = classify_env(name)
    galaxy_data.append({
        'name': name,
        'R': R_sorted,
        'residuals': residuals,
        'env': env,
        'Vflat': prop['Vflat'],
        'n_pts': len(residuals),
    })

n_galaxies = len(galaxy_data)
print(f"  Galaxies with N >= {MIN_POINTS}: {n_galaxies}")
print(f"  Rejected: {n_rejected}")


# ================================================================
# 3. COMPUTE ALL AUTOCORRELATION VARIANTS PER GALAXY
# ================================================================
print("\n[3] Computing raw, first-difference, and detrended autocorrelation...")

per_galaxy_results = []

for g in galaxy_data:
    eps = g['residuals']
    R = g['R']
    n = len(eps)

    # --- RAW (demeaned already by lag_autocorrelation internally) ---
    r1_raw = lag_autocorrelation(eps, lag=1)
    r2_raw = lag_autocorrelation(eps, lag=2)

    # --- A) FIRST DIFFERENCE ---
    d = np.diff(eps)
    r1_diff = lag_autocorrelation(d, lag=1) if len(d) >= 3 else np.nan
    r2_diff = lag_autocorrelation(d, lag=2) if len(d) >= 4 else np.nan

    # --- B) TREND-REMOVAL (UnivariateSpline) ---
    # Use smoothing factor s = n * var(eps) * 0.5 for moderate smoothing
    # This captures broad radial trends but not point-to-point wiggles
    var_eps = np.var(eps)
    s_param = n * var_eps * 0.5
    try:
        # k=3 cubic spline, moderate smoothing
        spline = UnivariateSpline(R, eps, k=min(3, n - 1), s=s_param)
        eps_trend = spline(R)
        eps_detrended = eps - eps_trend
        r1_detrend = lag_autocorrelation(eps_detrended, lag=1)
        r2_detrend = lag_autocorrelation(eps_detrended, lag=2)
    except Exception:
        eps_detrended = eps - np.mean(eps)  # fallback: just demean
        r1_detrend = lag_autocorrelation(eps_detrended, lag=1)
        r2_detrend = lag_autocorrelation(eps_detrended, lag=2)

    per_galaxy_results.append({
        'name': g['name'],
        'n_pts': n,
        'env': g['env'],
        'Vflat': g['Vflat'],
        'r1_raw': r1_raw,
        'r2_raw': r2_raw,
        'r1_diff': r1_diff,
        'r2_diff': r2_diff,
        'r1_detrend': r1_detrend,
        'r2_detrend': r2_detrend,
    })

# Extract arrays
def get_valid(key):
    return np.array([g[key] for g in per_galaxy_results if not np.isnan(g[key])])

r1_raw_vals = get_valid('r1_raw')
r2_raw_vals = get_valid('r2_raw')
r1_diff_vals = get_valid('r1_diff')
r2_diff_vals = get_valid('r2_diff')
r1_det_vals = get_valid('r1_detrend')
r2_det_vals = get_valid('r2_detrend')

print(f"  Raw:          lag-1 N={len(r1_raw_vals)}, lag-2 N={len(r2_raw_vals)}")
print(f"  Differenced:  lag-1 N={len(r1_diff_vals)}, lag-2 N={len(r2_diff_vals)}")
print(f"  Detrended:    lag-1 N={len(r1_det_vals)}, lag-2 N={len(r2_det_vals)}")


# ================================================================
# 4. AGGREGATE STATS FOR ALL THREE METHODS
# ================================================================
print("\n[4] Aggregate statistics...")

stats_raw_r1 = compute_autocorr_stats(r1_raw_vals, 'raw_lag1', rng_seed=42)
stats_raw_r2 = compute_autocorr_stats(r2_raw_vals, 'raw_lag2', rng_seed=43)
stats_diff_r1 = compute_autocorr_stats(r1_diff_vals, 'diff_lag1', rng_seed=44)
stats_diff_r2 = compute_autocorr_stats(r2_diff_vals, 'diff_lag2', rng_seed=45)
stats_det_r1 = compute_autocorr_stats(r1_det_vals, 'det_lag1', rng_seed=46)
stats_det_r2 = compute_autocorr_stats(r2_det_vals, 'det_lag2', rng_seed=47)


# ================================================================
# 5. ENVIRONMENT + MASS SPLITS FOR DIFF AND DETRENDED
# ================================================================
print("\n[5] Environment and mass splits for all methods...")

def get_split_arrays(key, split_key, split_val_a, split_val_b=None):
    """Get arrays split by a binary key."""
    a = np.array([g[key] for g in per_galaxy_results
                  if not np.isnan(g[key]) and g[split_key] == split_val_a])
    if split_val_b is not None:
        b = np.array([g[key] for g in per_galaxy_results
                      if not np.isnan(g[key]) and g[split_key] == split_val_b])
    else:
        b = np.array([g[key] for g in per_galaxy_results
                      if not np.isnan(g[key]) and g[split_key] != split_val_a])
    return a, b

# Median Vflat for mass split
vflat_arr = np.array([g['Vflat'] for g in per_galaxy_results])
med_vflat = np.median(vflat_arr)

# Add mass_bin to per_galaxy_results
for g in per_galaxy_results:
    g['mass_bin'] = 'low' if g['Vflat'] < med_vflat else 'high'

env_splits = {}
mass_splits = {}

for method_label, r1_key, r2_key in [
    ('raw', 'r1_raw', 'r2_raw'),
    ('diff', 'r1_diff', 'r2_diff'),
    ('detrended', 'r1_detrend', 'r2_detrend'),
]:
    # Env split
    f_r1, d_r1 = get_split_arrays(r1_key, 'env', 'field', 'dense')
    f_r2, d_r2 = get_split_arrays(r2_key, 'env', 'field', 'dense')
    env_splits[method_label] = {
        'lag1': compute_split_stats(f_r1, d_r1, 'field', 'dense'),
        'lag2': compute_split_stats(f_r2, d_r2, 'field', 'dense'),
    }
    # Mass split
    lo_r1, hi_r1 = get_split_arrays(r1_key, 'mass_bin', 'low', 'high')
    lo_r2, hi_r2 = get_split_arrays(r2_key, 'mass_bin', 'low', 'high')
    mass_splits[method_label] = {
        'lag1': compute_split_stats(lo_r1, hi_r1, 'low', 'high'),
        'lag2': compute_split_stats(lo_r2, hi_r2, 'low', 'high'),
    }


# ================================================================
# 6. PERMUTATION NULL (C)
# ================================================================
print("\n[6] Radius-order permutation null (within-galaxy)...")
print(f"    N_perm = {N_PERM} per galaxy, this may take a minute...")

perm_rng = np.random.default_rng(123)

perm_results = []

for g in galaxy_data:
    eps = g['residuals']
    n = len(eps)

    # Observed autocorrelations
    obs_r1 = lag_autocorrelation(eps, lag=1)
    obs_r2 = lag_autocorrelation(eps, lag=2)

    if np.isnan(obs_r1):
        continue

    # Permutation distribution
    perm_r1 = np.zeros(N_PERM)
    perm_r2 = np.zeros(N_PERM)

    for p in range(N_PERM):
        eps_shuf = perm_rng.permutation(eps)
        perm_r1[p] = lag_autocorrelation(eps_shuf, lag=1)
        perm_r2[p] = lag_autocorrelation(eps_shuf, lag=2)

    # One-sided p-value: fraction of perms >= observed
    p_r1 = float(np.mean(perm_r1 >= obs_r1))
    p_r2 = float(np.mean(perm_r2 >= obs_r2))

    # Avoid p=0 for Fisher (use 1/(N_PERM+1) floor)
    p_r1_adj = max(p_r1, 1.0 / (N_PERM + 1))
    p_r2_adj = max(p_r2, 1.0 / (N_PERM + 1))

    perm_results.append({
        'name': g['name'],
        'n_pts': n,
        'obs_r1': float(obs_r1),
        'obs_r2': float(obs_r2),
        'perm_mean_r1': float(np.mean(perm_r1)),
        'perm_std_r1': float(np.std(perm_r1)),
        'perm_p_r1': p_r1,
        'perm_p_r1_adj': p_r1_adj,
        'perm_p_r2': p_r2,
        'perm_p_r2_adj': p_r2_adj,
    })

n_perm_galaxies = len(perm_results)
print(f"  Permutation done for {n_perm_galaxies} galaxies")

# Fisher combined p-value for lag-1
log_p_sum_r1 = -2.0 * np.sum([np.log(g['perm_p_r1_adj']) for g in perm_results])
fisher_df_r1 = 2 * n_perm_galaxies
fisher_p_r1 = float(stats.chi2.sf(log_p_sum_r1, fisher_df_r1))

log_p_sum_r2 = -2.0 * np.sum([np.log(g['perm_p_r2_adj']) for g in perm_results])
fisher_df_r2 = 2 * n_perm_galaxies
fisher_p_r2 = float(stats.chi2.sf(log_p_sum_r2, fisher_df_r2))

print(f"  Fisher combined p (lag-1): {fisher_p_r1:.4e}")
print(f"  Fisher combined p (lag-2): {fisher_p_r2:.4e}")

# Fraction of galaxies with perm p < 0.05
n_sig_r1 = sum(1 for g in perm_results if g['perm_p_r1'] < 0.05)
n_sig_r2 = sum(1 for g in perm_results if g['perm_p_r2'] < 0.05)
frac_sig_r1 = n_sig_r1 / n_perm_galaxies
frac_sig_r2 = n_sig_r2 / n_perm_galaxies

print(f"  Galaxies with perm p < 0.05 (lag-1): {n_sig_r1}/{n_perm_galaxies} ({frac_sig_r1:.1%})")
print(f"  Galaxies with perm p < 0.05 (lag-2): {n_sig_r2}/{n_perm_galaxies} ({frac_sig_r2:.1%})")

# Aggregate: compare observed mean r1 to permutation-null distribution of mean r1
obs_mean_r1 = float(np.mean([g['obs_r1'] for g in perm_results]))
# Under the null, each galaxy's expected r1 ≈ -1/(n-1), aggregate these
null_means_r1 = []
# Use 10000 resamples of the aggregate mean under the null
agg_rng = np.random.default_rng(456)
for _ in range(10000):
    # For each galaxy, draw one r1 from its permutation distribution
    agg_r1 = 0.0
    for g_idx, g in enumerate(perm_results):
        # We don't have the full perm distribution stored, so estimate from mean/std
        agg_r1 += agg_rng.normal(g['perm_mean_r1'], g['perm_std_r1'])
    null_means_r1.append(agg_r1 / n_perm_galaxies)

null_means_r1 = np.array(null_means_r1)
null_mean_mean = float(np.mean(null_means_r1))
null_mean_std = float(np.std(null_means_r1))
perm_z_score = float((obs_mean_r1 - null_mean_mean) / null_mean_std) if null_mean_std > 0 else np.nan
perm_z_p = float(stats.norm.sf(perm_z_score))  # one-sided

print(f"\n  Aggregate observed mean r1: {obs_mean_r1:.4f}")
print(f"  Null mean (from perms): {null_mean_mean:.4f} ± {null_mean_std:.4f}")
print(f"  Z-score: {perm_z_score:.2f}")
print(f"  Z-based p (one-sided): {perm_z_p:.4e}")

perm_summary = {
    'n_galaxies': n_perm_galaxies,
    'n_perm_per_galaxy': N_PERM,
    'fisher_combined_p_lag1': fisher_p_r1,
    'fisher_combined_p_lag2': fisher_p_r2,
    'n_sig_005_lag1': n_sig_r1,
    'n_sig_005_lag2': n_sig_r2,
    'frac_sig_005_lag1': round(frac_sig_r1, 3),
    'frac_sig_005_lag2': round(frac_sig_r2, 3),
    'observed_mean_r1': round(obs_mean_r1, 4),
    'null_mean_r1': round(null_mean_mean, 4),
    'null_std_r1': round(null_mean_std, 4),
    'z_score': round(perm_z_score, 2),
    'z_p_one_sided': perm_z_p,
}


# ================================================================
# 7. CONSOLE SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("COMPARISON: RAW vs FIRST-DIFF vs DETRENDED")
print("=" * 72)

def fmt_pm(mean, se):
    return f"{mean:+.4f}±{se:.4f}"

def fmt_p(p):
    if p < 1e-10:
        return f"{p:.2e}"
    elif p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"

print(f"\n  {'Metric':<28} {'RAW':>16} {'FIRST-DIFF':>16} {'DETRENDED':>16}")
print(f"  {'-' * 76}")

# Lag-1
print(f"\n  --- Lag-1 ---")
print(f"  {'Mean ± SE':<28} "
      f"{fmt_pm(stats_raw_r1['mean'], stats_raw_r1['se']):>16} "
      f"{fmt_pm(stats_diff_r1['mean'], stats_diff_r1['se']):>16} "
      f"{fmt_pm(stats_det_r1['mean'], stats_det_r1['se']):>16}")
print(f"  {'95% CI lower':<28} "
      f"{stats_raw_r1['ci_95_lower']:>16.4f} "
      f"{stats_diff_r1['ci_95_lower']:>16.4f} "
      f"{stats_det_r1['ci_95_lower']:>16.4f}")
print(f"  {'95% CI upper':<28} "
      f"{stats_raw_r1['ci_95_upper']:>16.4f} "
      f"{stats_diff_r1['ci_95_upper']:>16.4f} "
      f"{stats_det_r1['ci_95_upper']:>16.4f}")
print(f"  {'CI excludes zero':<28} "
      f"{str(stats_raw_r1['ci_excludes_zero']):>16} "
      f"{str(stats_diff_r1['ci_excludes_zero']):>16} "
      f"{str(stats_det_r1['ci_excludes_zero']):>16}")
print(f"  {'t-test p (>0)':<28} "
      f"{fmt_p(stats_raw_r1['ttest_p_one_sided']):>16} "
      f"{fmt_p(stats_diff_r1['ttest_p_one_sided']):>16} "
      f"{fmt_p(stats_det_r1['ttest_p_one_sided']):>16}")
print(f"  {'Frac positive':<28} "
      f"{stats_raw_r1['frac_positive']:>16.3f} "
      f"{stats_diff_r1['frac_positive']:>16.3f} "
      f"{stats_det_r1['frac_positive']:>16.3f}")
print(f"  {'Binomial p':<28} "
      f"{fmt_p(stats_raw_r1['binom_p']):>16} "
      f"{fmt_p(stats_diff_r1['binom_p']):>16} "
      f"{fmt_p(stats_det_r1['binom_p']):>16}")

# Lag-2
print(f"\n  --- Lag-2 ---")
print(f"  {'Mean ± SE':<28} "
      f"{fmt_pm(stats_raw_r2['mean'], stats_raw_r2['se']):>16} "
      f"{fmt_pm(stats_diff_r2['mean'], stats_diff_r2['se']):>16} "
      f"{fmt_pm(stats_det_r2['mean'], stats_det_r2['se']):>16}")
print(f"  {'95% CI lower':<28} "
      f"{stats_raw_r2['ci_95_lower']:>16.4f} "
      f"{stats_diff_r2['ci_95_lower']:>16.4f} "
      f"{stats_det_r2['ci_95_lower']:>16.4f}")
print(f"  {'95% CI upper':<28} "
      f"{stats_raw_r2['ci_95_upper']:>16.4f} "
      f"{stats_diff_r2['ci_95_upper']:>16.4f} "
      f"{stats_det_r2['ci_95_upper']:>16.4f}")
print(f"  {'t-test p (>0)':<28} "
      f"{fmt_p(stats_raw_r2['ttest_p_one_sided']):>16} "
      f"{fmt_p(stats_diff_r2['ttest_p_one_sided']):>16} "
      f"{fmt_p(stats_det_r2['ttest_p_one_sided']):>16}")
print(f"  {'Frac positive':<28} "
      f"{stats_raw_r2['frac_positive']:>16.3f} "
      f"{stats_diff_r2['frac_positive']:>16.3f} "
      f"{stats_det_r2['frac_positive']:>16.3f}")

# Env split lag-1
print(f"\n  --- Environment Split (Welch p, lag-1) ---")
for method in ['raw', 'diff', 'detrended']:
    es = env_splits[method]['lag1']
    if es:
        print(f"  {method:<12}: field {es['field_mean']:+.4f}  dense {es['dense_mean']:+.4f}  "
              f"Welch p={es['welch_p']:.4f}")

# Mass split lag-1
print(f"\n  --- Mass Split (Welch p, lag-1) ---")
for method in ['raw', 'diff', 'detrended']:
    ms = mass_splits[method]['lag1']
    if ms:
        print(f"  {method:<12}: low {ms['low_mean']:+.4f}  high {ms['high_mean']:+.4f}  "
              f"Welch p={ms['welch_p']:.4f}")

# Permutation null
print(f"\n  --- Permutation Null ---")
print(f"  Observed aggregate <r1>:  {perm_summary['observed_mean_r1']:+.4f}")
print(f"  Null (shuffled) <r1>:     {perm_summary['null_mean_r1']:+.4f} ± {perm_summary['null_std_r1']:.4f}")
print(f"  Z-score:                  {perm_summary['z_score']:.2f}")
print(f"  Z-based p (one-sided):    {fmt_p(perm_summary['z_p_one_sided'])}")
print(f"  Fisher combined p (r1):   {fmt_p(perm_summary['fisher_combined_p_lag1'])}")
print(f"  Fisher combined p (r2):   {fmt_p(perm_summary['fisher_combined_p_lag2'])}")
print(f"  Frac galaxies p<0.05 r1:  {perm_summary['frac_sig_005_lag1']:.3f} "
      f"({perm_summary['n_sig_005_lag1']}/{perm_summary['n_galaxies']})")


# ================================================================
# VERDICT
# ================================================================
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

# Check if differencing kills the signal
diff_killed = stats_diff_r1['ttest_p_one_sided'] > 0.05
detrend_killed = stats_det_r1['ttest_p_one_sided'] > 0.05

if diff_killed and detrend_killed:
    structure_verdict = "SMOOTH_DRIFT"
    print("\n  Both differencing and detrending KILL the autocorrelation.")
    print("  >>> The signal is SMOOTH RADIAL DRIFT, not oscillatory structure.")
elif not diff_killed and not detrend_killed:
    structure_verdict = "OSCILLATORY_STRUCTURE"
    print("\n  Autocorrelation SURVIVES both differencing and detrending.")
    print("  >>> Genuine OSCILLATORY STRUCTURE in RAR residuals.")
elif diff_killed and not detrend_killed:
    structure_verdict = "MIXED_MOSTLY_DRIFT"
    print("\n  Differencing kills it but detrending doesn't fully remove it.")
    print("  >>> Mostly smooth drift with some residual structure.")
else:
    structure_verdict = "MIXED_SOME_OSCILLATION"
    print("\n  Differencing preserves signal but detrending removes it.")
    print("  >>> Short-range correlations (nearest-neighbor), not broad oscillation.")

perm_significant = perm_summary['z_p_one_sided'] < 0.001
print(f"\n  Permutation null: {'REJECTED' if perm_significant else 'NOT rejected'} "
      f"(z = {perm_summary['z_score']:.2f}, p = {fmt_p(perm_summary['z_p_one_sided'])})")

print(f"\n  Overall: {structure_verdict}")


# ================================================================
# SAVE JSON
# ================================================================
results = {
    'test': 'interface_oscillation_controls',
    'description': ('Detrending controls (first-difference, spline trend-removal) '
                     'and within-galaxy permutation null for RAR residual autocorrelation.'),
    'parameters': {
        'min_points_per_galaxy': MIN_POINTS,
        'n_bootstrap': N_BOOT,
        'n_perm': N_PERM,
        'spline_smoothing': 'n * var(eps) * 0.5',
    },
    'sample': {
        'n_galaxies': n_galaxies,
        'n_rejected': n_rejected,
        'median_Vflat': round(float(med_vflat), 1),
    },
    'aggregate_raw': {
        'lag1': stats_raw_r1,
        'lag2': stats_raw_r2,
    },
    'aggregate_diff': {
        'lag1': stats_diff_r1,
        'lag2': stats_diff_r2,
    },
    'aggregate_detrended': {
        'lag1': stats_det_r1,
        'lag2': stats_det_r2,
    },
    'environment_split': env_splits,
    'mass_split': mass_splits,
    'permutation_null_summary': perm_summary,
    'structure_verdict': structure_verdict,
    'perm_null_rejected': perm_significant,
}

outpath = os.path.join(RESULTS_DIR, 'summary_interface_oscillation_controls.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("=" * 72)
print("Done.")
