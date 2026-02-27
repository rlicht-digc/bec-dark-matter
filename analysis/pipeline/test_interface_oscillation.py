#!/usr/bin/env python3
"""
Interface Oscillation Test (Step 25) — with Demeaning Control
==============================================================

For each galaxy with N >= 15 rotation curve points, compute RAR residuals
r(R) ordered by galactocentric radius, then measure lag-1 and lag-2
autocorrelation within each galaxy.

CRITICAL CONTROL: also compute autocorrelation on demeaned residuals
  eps0 = eps - mean(eps)
This removes per-galaxy mean offsets (distance errors, M/L systematics)
and isolates genuine radial structure from global shifts.

If residuals are independent (pure noise around the RAR), autocorrelation = 0.
Positive autocorrelation means residuals are correlated in radius — i.e.,
coherent oscillatory structure in the rotation curve relative to the RAR.

Tests:
  1. Per-galaxy lag-1 and lag-2 autocorrelation (raw and demeaned)
  2. Aggregate: mean and SE across galaxies
  3. One-sample t-test and bootstrap CI for mean autocorrelation > 0
  4. Split by environment (cluster/group vs field)
  5. Split by mass (low vs high Vflat)
  6. Side-by-side raw vs demeaned summary
  7. JSON output with parallel raw/demeaned blocks

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy import stats
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
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921

MIN_POINTS = 15  # Minimum RC points per galaxy for autocorrelation
N_BOOT = 10000

np.random.seed(42)

print("=" * 72)
print("INTERFACE OSCILLATION TEST (Step 25) — with Demeaning Control")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")
print(f"  Minimum points per galaxy: {MIN_POINTS}")
print(f"  Bootstrap resamples: {N_BOOT}")


# ================================================================
# ENVIRONMENT CLASSIFICATION (same as other pipeline scripts)
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
# 1. LOAD SPARC DATA
# ================================================================
print("\n[1] Loading SPARC data...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

# Load rotation curves
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

# Load galaxy properties from MRT
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
# 2. COMPUTE PER-GALAXY RAR RESIDUALS ORDERED BY RADIUS
# ================================================================
print("\n[2] Computing per-galaxy RAR residuals (radius-ordered)...")
print("    Computing BOTH raw and demeaned autocorrelation...")


def rar_function(log_gbar, a0=1.2e-10):
    """Standard RAR prediction: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def lag_autocorrelation(x, lag=1, center=True):
    """Compute lag-k autocorrelation of a 1D array.

    Args:
        center: If True (default), subtract mean before computing (standard ACF).
                If False, compute uncentered ACF: sum(x_i * x_{i+k}) / sum(x_i^2).
                The uncentered version tests whether the raw signal has serial
                correlation WITHOUT removing the mean offset first.
    """
    n = len(x)
    if n <= lag + 1:
        return np.nan
    if center:
        xbar = np.mean(x)
        x_use = x - xbar
    else:
        x_use = x
    var = np.mean(x_use**2)
    if var < 1e-30:
        return np.nan
    cov = np.mean(x_use[:n - lag] * x_use[lag:])
    return cov / var


galaxy_autocorr = []
n_rejected = 0

for name, gdata in rc_data.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]

    # Standard quality cuts
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

    R = gdata['R']
    Vobs = gdata['Vobs']
    Vgas = gdata['Vgas']
    Vdisk = gdata['Vdisk']
    Vbul = gdata['Vbul']

    # Compute accelerations
    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < MIN_POINTS:
        n_rejected += 1
        continue

    # Extract valid data, ordered by radius (already should be, but ensure)
    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]

    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]

    # RAR residuals ordered by radius
    log_gobs_pred = rar_function(log_gbar)
    residuals = log_gobs - log_gobs_pred

    # Demeaned residuals: remove per-galaxy mean offset
    residuals_dm = residuals - np.mean(residuals)

    # Compute lag-1 and lag-2 autocorrelation — RAW (uncentered: includes mean offset)
    r1_raw = lag_autocorrelation(residuals, lag=1, center=False)
    r2_raw = lag_autocorrelation(residuals, lag=2, center=False)

    # Compute lag-1 and lag-2 autocorrelation — DEMEANED (standard centered ACF)
    r1_dm = lag_autocorrelation(residuals_dm, lag=1, center=True)
    r2_dm = lag_autocorrelation(residuals_dm, lag=2, center=True)

    # Median log(gbar) — to classify low/high acceleration regime
    med_log_gbar = np.median(log_gbar)

    env = classify_env(name)

    galaxy_autocorr.append({
        'name': name,
        'n_pts': int(np.sum(valid)),
        'r1_raw': r1_raw,
        'r2_raw': r2_raw,
        'r1_dm': r1_dm,
        'r2_dm': r2_dm,
        'env': env,
        'Vflat': prop['Vflat'],
        'med_log_gbar': float(med_log_gbar),
        'std_resid': float(np.std(residuals)),
        'mean_resid': float(np.mean(residuals)),
    })

n_galaxies = len(galaxy_autocorr)
print(f"  Galaxies with N >= {MIN_POINTS}: {n_galaxies}")
print(f"  Rejected (N < {MIN_POINTS}): {n_rejected}")


# ================================================================
# HELPER: compute full statistics for an autocorrelation array
# ================================================================
def compute_autocorr_stats(values, label, rng_seed=42):
    """Compute mean, SE, bootstrap CI, t-test, binomial for an array."""
    n = len(values)
    if n < 3:
        return None

    mean_val = float(np.mean(values))
    se_val = float(np.std(values, ddof=1) / np.sqrt(n))
    median_val = float(np.median(values))
    std_val = float(np.std(values, ddof=1))

    # One-sided t-test: mean > 0
    t_stat, p_two = stats.ttest_1samp(values, 0.0)
    p_one = float(p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0)

    # Bootstrap CI
    boot_rng = np.random.default_rng(rng_seed)
    boot_means = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = boot_rng.integers(0, n, size=n)
        boot_means[b] = np.mean(values[idx])
    ci = np.percentile(boot_means, [2.5, 97.5])

    # Fraction positive + binomial test
    n_pos = int(np.sum(values > 0))
    frac_pos = float(n_pos / n)
    binom_p = float(stats.binomtest(n_pos, n, 0.5, alternative='greater').pvalue)

    return {
        'mean': round(mean_val, 4),
        'se': round(se_val, 4),
        'median': round(median_val, 4),
        'std': round(std_val, 4),
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


# ================================================================
# HELPER: compute env/mass split statistics
# ================================================================
def compute_split_stats(arr_a, arr_b, label_a, label_b):
    """Welch t-test and Mann-Whitney for two groups."""
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
# 3. EXTRACT ARRAYS FOR RAW AND DEMEANED
# ================================================================
print("\n[3] Extracting raw and demeaned autocorrelation arrays...")

r1_raw_vals = np.array([g['r1_raw'] for g in galaxy_autocorr
                         if not np.isnan(g['r1_raw'])])
r2_raw_vals = np.array([g['r2_raw'] for g in galaxy_autocorr
                         if not np.isnan(g['r2_raw'])])
r1_dm_vals = np.array([g['r1_dm'] for g in galaxy_autocorr
                        if not np.isnan(g['r1_dm'])])
r2_dm_vals = np.array([g['r2_dm'] for g in galaxy_autocorr
                        if not np.isnan(g['r2_dm'])])

print(f"  Raw:      lag-1 N={len(r1_raw_vals)}, lag-2 N={len(r2_raw_vals)}")
print(f"  Demeaned: lag-1 N={len(r1_dm_vals)}, lag-2 N={len(r2_dm_vals)}")


# ================================================================
# 4. COMPUTE AGGREGATE STATS — ALL FOUR COMBINATIONS
# ================================================================
print("\n[4] Computing aggregate statistics (raw & demeaned)...")

stats_lag1_raw = compute_autocorr_stats(r1_raw_vals, 'lag1_raw', rng_seed=42)
stats_lag2_raw = compute_autocorr_stats(r2_raw_vals, 'lag2_raw', rng_seed=43)
stats_lag1_dm = compute_autocorr_stats(r1_dm_vals, 'lag1_dm', rng_seed=44)
stats_lag2_dm = compute_autocorr_stats(r2_dm_vals, 'lag2_dm', rng_seed=45)

for tag, s in [('Lag-1 RAW', stats_lag1_raw), ('Lag-2 RAW', stats_lag2_raw),
               ('Lag-1 DEMEANED', stats_lag1_dm), ('Lag-2 DEMEANED', stats_lag2_dm)]:
    print(f"\n  {tag}:")
    print(f"    Mean: {s['mean']:.4f} ± {s['se']:.4f}")
    print(f"    95% CI: [{s['ci_95_lower']:.4f}, {s['ci_95_upper']:.4f}]  "
          f"(excludes 0: {s['ci_excludes_zero']})")
    print(f"    t = {s['ttest_t']:.3f}, p(>0) = {s['ttest_p_one_sided']:.4e}")
    print(f"    Frac positive: {s['frac_positive']:.3f} "
          f"({s['n_positive']}/{s['n_total']}), binom p = {s['binom_p']:.4e}")


# ================================================================
# 5. ENVIRONMENT SPLIT — raw and demeaned
# ================================================================
print("\n[5] Environment split (dense vs field) — raw & demeaned...")

field_gals = [g for g in galaxy_autocorr if g['env'] == 'field']
dense_gals = [g for g in galaxy_autocorr if g['env'] == 'dense']

# Extract arrays per env × metric
def extract_env_arrays(key):
    f_arr = np.array([g[key] for g in field_gals if not np.isnan(g[key])])
    d_arr = np.array([g[key] for g in dense_gals if not np.isnan(g[key])])
    return f_arr, d_arr

f_r1_raw, d_r1_raw = extract_env_arrays('r1_raw')
f_r2_raw, d_r2_raw = extract_env_arrays('r2_raw')
f_r1_dm, d_r1_dm = extract_env_arrays('r1_dm')
f_r2_dm, d_r2_dm = extract_env_arrays('r2_dm')

print(f"\n  {'Subsample':<12} {'N':>5}  {'<r1 raw>':>9} {'<r1 dm>':>9}  "
      f"{'<r2 raw>':>9} {'<r2 dm>':>9}")
print(f"  {'-' * 62}")
for label, fr1, dr1_d, fr2, dr2_d in [
    ('Field', f_r1_raw, f_r1_dm, f_r2_raw, f_r2_dm),
    ('Dense', d_r1_raw, d_r1_dm, d_r2_raw, d_r2_dm),
    ('All', r1_raw_vals, r1_dm_vals, r2_raw_vals, r2_dm_vals),
]:
    n = len(fr1)
    if n >= 3:
        print(f"  {label:<12} {n:5d}  {np.mean(fr1):9.4f} {np.mean(dr1_d):9.4f}  "
              f"{np.mean(fr2):9.4f} {np.mean(dr2_d):9.4f}")

# Compute split stats
env_results_raw = {}
env_results_dm = {}

for lag_key, f_raw, d_raw, f_dm, d_dm in [
    ('lag1', f_r1_raw, d_r1_raw, f_r1_dm, d_r1_dm),
    ('lag2', f_r2_raw, d_r2_raw, f_r2_dm, d_r2_dm),
]:
    s_raw = compute_split_stats(f_raw, d_raw, 'field', 'dense')
    s_dm = compute_split_stats(f_dm, d_dm, 'field', 'dense')
    if s_raw:
        env_results_raw[lag_key] = s_raw
    if s_dm:
        env_results_dm[lag_key] = s_dm

if 'lag1' in env_results_raw:
    print(f"\n  Field vs Dense — lag-1:")
    print(f"    RAW:      Welch p = {env_results_raw['lag1']['welch_p']:.4f}, "
          f"MWU p = {env_results_raw['lag1']['mwu_p']:.4f}")
    print(f"    DEMEANED: Welch p = {env_results_dm['lag1']['welch_p']:.4f}, "
          f"MWU p = {env_results_dm['lag1']['mwu_p']:.4f}")
if 'lag2' in env_results_raw:
    print(f"  Field vs Dense — lag-2:")
    print(f"    RAW:      Welch p = {env_results_raw['lag2']['welch_p']:.4f}, "
          f"MWU p = {env_results_raw['lag2']['mwu_p']:.4f}")
    print(f"    DEMEANED: Welch p = {env_results_dm['lag2']['welch_p']:.4f}, "
          f"MWU p = {env_results_dm['lag2']['mwu_p']:.4f}")


# ================================================================
# 6. MASS SPLIT — raw and demeaned
# ================================================================
print("\n[6] Mass split (median Vflat) — raw & demeaned...")

vflat_arr = np.array([g['Vflat'] for g in galaxy_autocorr])
med_vflat = np.median(vflat_arr)

lo_mass = [g for g in galaxy_autocorr if g['Vflat'] < med_vflat]
hi_mass = [g for g in galaxy_autocorr if g['Vflat'] >= med_vflat]

def extract_mass_arrays(key):
    lo = np.array([g[key] for g in lo_mass if not np.isnan(g[key])])
    hi = np.array([g[key] for g in hi_mass if not np.isnan(g[key])])
    return lo, hi

lo_r1_raw, hi_r1_raw = extract_mass_arrays('r1_raw')
lo_r2_raw, hi_r2_raw = extract_mass_arrays('r2_raw')
lo_r1_dm, hi_r1_dm = extract_mass_arrays('r1_dm')
lo_r2_dm, hi_r2_dm = extract_mass_arrays('r2_dm')

print(f"\n  Median Vflat = {med_vflat:.1f} km/s")
print(f"\n  {'Subsample':<16} {'N':>5}  {'<r1 raw>':>9} {'<r1 dm>':>9}  "
      f"{'<r2 raw>':>9} {'<r2 dm>':>9}")
print(f"  {'-' * 66}")
for label, r1r, r1d, r2r, r2d in [
    (f'Low (<{med_vflat:.0f})', lo_r1_raw, lo_r1_dm, lo_r2_raw, lo_r2_dm),
    (f'High (>={med_vflat:.0f})', hi_r1_raw, hi_r1_dm, hi_r2_raw, hi_r2_dm),
]:
    n = len(r1r)
    if n >= 3:
        print(f"  {label:<16} {n:5d}  {np.mean(r1r):9.4f} {np.mean(r1d):9.4f}  "
              f"{np.mean(r2r):9.4f} {np.mean(r2d):9.4f}")

mass_results_raw = {}
mass_results_dm = {}

for lag_key, lo_raw, hi_raw, lo_dm, hi_dm in [
    ('lag1', lo_r1_raw, hi_r1_raw, lo_r1_dm, hi_r1_dm),
    ('lag2', lo_r2_raw, hi_r2_raw, lo_r2_dm, hi_r2_dm),
]:
    s_raw = compute_split_stats(lo_raw, hi_raw, 'low', 'high')
    s_dm = compute_split_stats(lo_dm, hi_dm, 'low', 'high')
    if s_raw:
        mass_results_raw[lag_key] = s_raw
    if s_dm:
        mass_results_dm[lag_key] = s_dm

if 'lag1' in mass_results_raw:
    print(f"\n  Low vs High mass — lag-1:")
    print(f"    RAW:      Welch p = {mass_results_raw['lag1']['welch_p']:.4f}, "
          f"MWU p = {mass_results_raw['lag1']['mwu_p']:.4f}")
    print(f"    DEMEANED: Welch p = {mass_results_dm['lag1']['welch_p']:.4f}, "
          f"MWU p = {mass_results_dm['lag1']['mwu_p']:.4f}")
if 'lag2' in mass_results_raw:
    print(f"  Low vs High mass — lag-2:")
    print(f"    RAW:      Welch p = {mass_results_raw['lag2']['welch_p']:.4f}, "
          f"MWU p = {mass_results_raw['lag2']['mwu_p']:.4f}")
    print(f"    DEMEANED: Welch p = {mass_results_dm['lag2']['welch_p']:.4f}, "
          f"MWU p = {mass_results_dm['lag2']['mwu_p']:.4f}")


# ================================================================
# 7. TOP/BOTTOM GALAXIES (demeaned)
# ================================================================
print("\n[7] Galaxies with strongest demeaned lag-1 autocorrelation...")

sorted_by_r1_dm = sorted(galaxy_autocorr,
                          key=lambda g: g['r1_dm'] if not np.isnan(g['r1_dm']) else -999,
                          reverse=True)

print(f"\n  Top 10 (highest demeaned lag-1):")
print(f"  {'Name':<14} {'N':>5} {'r1_raw':>8} {'r1_dm':>8} {'r2_dm':>8} "
      f"{'Env':>6} {'Vflat':>7}")
print(f"  {'-' * 60}")
for g in sorted_by_r1_dm[:10]:
    print(f"  {g['name']:<14} {g['n_pts']:5d} {g['r1_raw']:8.4f} {g['r1_dm']:8.4f} "
          f"{g['r2_dm']:8.4f} {g['env']:>6} {g['Vflat']:7.1f}")

print(f"\n  Bottom 10 (most negative demeaned lag-1):")
print(f"  {'Name':<14} {'N':>5} {'r1_raw':>8} {'r1_dm':>8} {'r2_dm':>8} "
      f"{'Env':>6} {'Vflat':>7}")
print(f"  {'-' * 60}")
for g in sorted_by_r1_dm[-10:]:
    print(f"  {g['name']:<14} {g['n_pts']:5d} {g['r1_raw']:8.4f} {g['r1_dm']:8.4f} "
          f"{g['r2_dm']:8.4f} {g['env']:>6} {g['Vflat']:7.1f}")


# ================================================================
# 8. CORRELATION WITH GALAXY PROPERTIES (demeaned)
# ================================================================
print("\n[8] Correlation of demeaned lag-1 with galaxy properties...")

valid_gals = [g for g in galaxy_autocorr if not np.isnan(g['r1_dm'])]
r1_dm_arr = np.array([g['r1_dm'] for g in valid_gals])
npts_arr = np.array([g['n_pts'] for g in valid_gals])
vflat_arr_all = np.array([g['Vflat'] for g in valid_gals])
med_gbar_arr = np.array([g['med_log_gbar'] for g in valid_gals])

correlations = {}
for prop_name, prop_arr in [('N_pts', npts_arr), ('Vflat', vflat_arr_all),
                             ('med_log_gbar', med_gbar_arr)]:
    rho, p_rho = stats.spearmanr(prop_arr, r1_dm_arr)
    print(f"  r1_dm vs {prop_name:<14}: Spearman ρ = {rho:+.3f}, p = {p_rho:.4f}")
    correlations[f'r1_dm_vs_{prop_name}'] = {
        'spearman_rho': round(float(rho), 3),
        'p_value': round(float(p_rho), 4),
    }


# ================================================================
# 9. SIDE-BY-SIDE SUMMARY: RAW vs DEMEANED
# ================================================================
print("\n" + "=" * 72)
print("SIDE-BY-SIDE SUMMARY: RAW vs DEMEANED")
print("=" * 72)

header = f"  {'Metric':<28} {'RAW':>14} {'DEMEANED':>14}"
print(header)
print(f"  {'-' * 56}")

def fmt_pm(mean, se):
    return f"{mean:+.4f}±{se:.4f}"

def fmt_ci(lo, hi):
    return f"[{lo:.4f},{hi:.4f}]"

def fmt_p(p):
    if p < 1e-10:
        return f"{p:.2e}"
    elif p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"

# Lag-1
print(f"\n  --- Lag-1 ---")
print(f"  {'Mean ± SE':<28} {fmt_pm(stats_lag1_raw['mean'], stats_lag1_raw['se']):>14} "
      f"{fmt_pm(stats_lag1_dm['mean'], stats_lag1_dm['se']):>14}")
print(f"  {'95% CI':<28} {fmt_ci(stats_lag1_raw['ci_95_lower'], stats_lag1_raw['ci_95_upper']):>14} "
      f"{fmt_ci(stats_lag1_dm['ci_95_lower'], stats_lag1_dm['ci_95_upper']):>14}")
print(f"  {'t-test p (>0)':<28} {fmt_p(stats_lag1_raw['ttest_p_one_sided']):>14} "
      f"{fmt_p(stats_lag1_dm['ttest_p_one_sided']):>14}")
print(f"  {'Frac positive':<28} {stats_lag1_raw['frac_positive']:>14.3f} "
      f"{stats_lag1_dm['frac_positive']:>14.3f}")
print(f"  {'Binomial p':<28} {fmt_p(stats_lag1_raw['binom_p']):>14} "
      f"{fmt_p(stats_lag1_dm['binom_p']):>14}")

# Lag-2
print(f"\n  --- Lag-2 ---")
print(f"  {'Mean ± SE':<28} {fmt_pm(stats_lag2_raw['mean'], stats_lag2_raw['se']):>14} "
      f"{fmt_pm(stats_lag2_dm['mean'], stats_lag2_dm['se']):>14}")
print(f"  {'95% CI':<28} {fmt_ci(stats_lag2_raw['ci_95_lower'], stats_lag2_raw['ci_95_upper']):>14} "
      f"{fmt_ci(stats_lag2_dm['ci_95_lower'], stats_lag2_dm['ci_95_upper']):>14}")
print(f"  {'t-test p (>0)':<28} {fmt_p(stats_lag2_raw['ttest_p_one_sided']):>14} "
      f"{fmt_p(stats_lag2_dm['ttest_p_one_sided']):>14}")
print(f"  {'Frac positive':<28} {stats_lag2_raw['frac_positive']:>14.3f} "
      f"{stats_lag2_dm['frac_positive']:>14.3f}")
print(f"  {'Binomial p':<28} {fmt_p(stats_lag2_raw['binom_p']):>14} "
      f"{fmt_p(stats_lag2_dm['binom_p']):>14}")

# Env split summary
print(f"\n  --- Environment (Welch p, lag-1) ---")
if 'lag1' in env_results_raw and 'lag1' in env_results_dm:
    print(f"  {'Field vs Dense':<28} {env_results_raw['lag1']['welch_p']:>14.4f} "
          f"{env_results_dm['lag1']['welch_p']:>14.4f}")

# Mass split summary
print(f"\n  --- Mass (Welch p, lag-1) ---")
if 'lag1' in mass_results_raw and 'lag1' in mass_results_dm:
    print(f"  {'Low vs High':<28} {mass_results_raw['lag1']['welch_p']:>14.4f} "
          f"{mass_results_dm['lag1']['welch_p']:>14.4f}")


# ================================================================
# FINAL VERDICT
# ================================================================
print("\n" + "=" * 72)
print("FINAL VERDICT")
print("=" * 72)

# Verdict based on DEMEANED (the controlled metric)
p_dm_lag1 = stats_lag1_dm['ttest_p_one_sided']
ci_dm_excludes = stats_lag1_dm['ci_excludes_zero']

print(f"\n  DEMEANED lag-1 autocorrelation: {stats_lag1_dm['mean']:.4f} ± {stats_lag1_dm['se']:.4f}")
print(f"  DEMEANED lag-2 autocorrelation: {stats_lag2_dm['mean']:.4f} ± {stats_lag2_dm['se']:.4f}")
print(f"  T-test p (demeaned lag-1 > 0): {p_dm_lag1:.4e}")
print(f"  Bootstrap 95% CI (demeaned lag-1): "
      f"[{stats_lag1_dm['ci_95_lower']:.4f}, {stats_lag1_dm['ci_95_upper']:.4f}]")
print(f"  Frac r1_dm > 0: {stats_lag1_dm['frac_positive']:.3f}")

if p_dm_lag1 < 0.001 and ci_dm_excludes:
    verdict_dm = "STRONG_POSITIVE_AUTOCORRELATION"
    print(f"\n  >>> STRONG positive demeaned autocorrelation detected.")
    print(f"  >>> Radial coherence persists after removing per-galaxy mean offset.")
    print(f"  >>> This is genuine oscillatory structure, not distance/M-L systematics.")
elif p_dm_lag1 < 0.05:
    verdict_dm = "SIGNIFICANT_POSITIVE_AUTOCORRELATION"
    print(f"\n  >>> Significant positive demeaned autocorrelation (p < 0.05).")
    print(f"  >>> Radial coherence survives demeaning.")
elif p_dm_lag1 > 0.95:
    verdict_dm = "NEGATIVE_AUTOCORRELATION"
    print(f"\n  >>> Demeaned residuals show anti-correlation.")
else:
    verdict_dm = "NO_SIGNIFICANT_AUTOCORRELATION"
    print(f"\n  >>> No significant demeaned autocorrelation.")
    print(f"  >>> Raw autocorrelation was driven by per-galaxy mean offsets.")

# Raw verdict for comparison
p_raw_lag1 = stats_lag1_raw['ttest_p_one_sided']
if p_raw_lag1 < 0.001 and stats_lag1_raw['ci_excludes_zero']:
    verdict_raw = "STRONG_POSITIVE_AUTOCORRELATION"
elif p_raw_lag1 < 0.05:
    verdict_raw = "SIGNIFICANT_POSITIVE_AUTOCORRELATION"
elif p_raw_lag1 > 0.95:
    verdict_raw = "NEGATIVE_AUTOCORRELATION"
else:
    verdict_raw = "NO_SIGNIFICANT_AUTOCORRELATION"

# Environment verdict (demeaned)
if env_results_dm and 'lag1' in env_results_dm:
    ep = env_results_dm['lag1']['welch_p']
    if ep < 0.05:
        env_verdict_dm = "SIGNIFICANT_DIFFERENCE"
        print(f"\n  Environment (demeaned): autocorrelation DIFFERS (p = {ep:.4f})")
    else:
        env_verdict_dm = "NO_DIFFERENCE"
        print(f"\n  Environment (demeaned): autocorrelation similar (p = {ep:.4f})")
else:
    env_verdict_dm = "INSUFFICIENT_DATA"

# Mass verdict (demeaned)
if mass_results_dm and 'lag1' in mass_results_dm:
    mp = mass_results_dm['lag1']['welch_p']
    if mp < 0.05:
        mass_verdict_dm = "SIGNIFICANT_DIFFERENCE"
        print(f"  Mass split (demeaned): autocorrelation DIFFERS (p = {mp:.4f})")
    else:
        mass_verdict_dm = "NO_DIFFERENCE"
        print(f"  Mass split (demeaned): autocorrelation similar (p = {mp:.4f})")
else:
    mass_verdict_dm = "INSUFFICIENT_DATA"


# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    'test': 'interface_oscillation_with_demeaning',
    'description': ('Lag autocorrelation of RAR residuals ordered by galactocentric radius. '
                     'Both raw and per-galaxy demeaned (eps0 = eps - mean(eps)) are computed.'),
    'parameters': {
        'min_points_per_galaxy': MIN_POINTS,
        'n_bootstrap': N_BOOT,
    },
    'sample': {
        'n_galaxies': n_galaxies,
        'n_rejected': n_rejected,
        'n_valid_lag1': int(len(r1_raw_vals)),
        'n_valid_lag2': int(len(r2_raw_vals)),
    },
    # Raw aggregate
    'aggregate_lag1_raw': stats_lag1_raw,
    'aggregate_lag2_raw': stats_lag2_raw,
    # Demeaned aggregate
    'aggregate_lag1_demeaned': stats_lag1_dm,
    'aggregate_lag2_demeaned': stats_lag2_dm,
    # Environment split
    'environment_split_raw': env_results_raw,
    'environment_split_demeaned': env_results_dm,
    # Mass split
    'mass_split_raw': mass_results_raw,
    'mass_split_demeaned': mass_results_dm,
    # Correlations (demeaned)
    'correlations_demeaned': correlations,
    # Per-galaxy
    'per_galaxy': [
        {
            'name': g['name'],
            'n_pts': g['n_pts'],
            'r1_raw': round(float(g['r1_raw']), 4) if not np.isnan(g['r1_raw']) else None,
            'r2_raw': round(float(g['r2_raw']), 4) if not np.isnan(g['r2_raw']) else None,
            'r1_dm': round(float(g['r1_dm']), 4) if not np.isnan(g['r1_dm']) else None,
            'r2_dm': round(float(g['r2_dm']), 4) if not np.isnan(g['r2_dm']) else None,
            'env': g['env'],
            'Vflat': round(float(g['Vflat']), 1),
            'mean_resid': round(float(g['mean_resid']), 4),
        }
        for g in galaxy_autocorr
    ],
    # Verdicts
    'verdict_raw': verdict_raw,
    'verdict_demeaned': verdict_dm,
    'env_verdict_demeaned': env_verdict_dm,
    'mass_verdict_demeaned': mass_verdict_dm,
}

outpath = os.path.join(RESULTS_DIR, 'summary_interface_oscillation_demeaned.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("=" * 72)
print("Done.")
