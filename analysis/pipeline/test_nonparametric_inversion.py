#!/usr/bin/env python3
"""
Non-Parametric Mean Relation Inversion Robustness Test
========================================================

CRITICAL ROBUSTNESS CHECK: Is the scatter derivative inversion point at g†
an artifact of assuming the RAR interpolating function?

A skeptic could argue: "You computed residuals from g_obs = g_bar/(1-exp(-√(g_bar/g†))),
which has a characteristic scale at g†. Of course the scatter changes character there —
you built it in."

This test computes residuals from THREE non-parametric mean relations that know
nothing about g† or the RAR:
  1. LOESS (locally weighted scatterplot smoothing)
  2. Natural cubic spline
  3. Isotonic regression (monotone-increasing, which the RAR is)

If the inversion point persists at g† with all three, it is a property of the DATA,
not the assumed functional form.

SECOND TEST: Fit separate mean relations for field vs dense galaxies, then
check whether residual scatter trends survive. If the scatter pattern persists
when each environment has its OWN mean subtracted, it can't be driven by the
mean relation differing between environments.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy.stats import levene
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10
kpc_m = 3.086e19
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921

print("=" * 72)
print("NON-PARAMETRIC MEAN RELATION INVERSION ROBUSTNESS TEST")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")


# ================================================================
# ENVIRONMENT CLASSIFICATION (same as all other tests)
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
    if name in UMA_GALAXIES:
        return 'dense'
    if name in GROUP_MEMBERS:
        return 'dense'
    return 'field'


# ================================================================
# DATA LOADING
# ================================================================
print("\n[1] Loading SPARC data...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

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
            'T': int(parts[0]), 'D': float(parts[1]), 'eD': float(parts[2]),
            'fD': int(parts[3]), 'Inc': float(parts[4]), 'eInc': float(parts[5]),
            'L36': float(parts[6]), 'Vflat': float(parts[14]), 'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue


# ================================================================
# COMPUTE RAW g_bar, g_obs FOR ALL QUALITY-CUT GALAXIES
# ================================================================
print("\n[2] Computing accelerations for quality-cut galaxies...")

all_log_gbar = []
all_log_gobs = []
all_env = []
all_galname = []

gal_data = {}  # per-galaxy storage

for name, gdata in galaxies.items():
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
    if np.sum(valid) < 5:
        continue

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    env = classify_env(name)

    gal_data[name] = {
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'env': env,
        'n_pts': len(log_gbar),
    }

    all_log_gbar.extend(log_gbar.tolist())
    all_log_gobs.extend(log_gobs.tolist())
    all_env.extend([env] * len(log_gbar))
    all_galname.extend([name] * len(log_gbar))

all_log_gbar = np.array(all_log_gbar)
all_log_gobs = np.array(all_log_gobs)
all_env = np.array(all_env)

n_field = np.sum(all_env == 'field')
n_dense = np.sum(all_env == 'dense')
n_gals = len(gal_data)
n_field_gals = sum(1 for g in gal_data.values() if g['env'] == 'field')
n_dense_gals = sum(1 for g in gal_data.values() if g['env'] == 'dense')

print(f"  {n_gals} galaxies, {len(all_log_gbar)} points")
print(f"  Field: {n_field_gals} gals, {n_field} pts")
print(f"  Dense: {n_dense_gals} gals, {n_dense} pts")
print(f"  g_bar range: [{all_log_gbar.min():.2f}, {all_log_gbar.max():.2f}]")


# ================================================================
# NON-PARAMETRIC MEAN RELATIONS
# ================================================================

def loess_smooth(x, y, frac=0.15, n_grid=200):
    """LOESS (locally weighted regression) on a grid."""
    from numpy.polynomial.polynomial import polyfit, polyval
    x_grid = np.linspace(x.min(), x.max(), n_grid)
    y_grid = np.zeros(n_grid)
    h = frac * (x.max() - x.min())

    for i, xg in enumerate(x_grid):
        w = np.exp(-0.5 * ((x - xg) / h)**2)
        w /= w.sum()
        # Weighted local linear regression
        xc = x - xg
        W = np.diag(w)
        X = np.column_stack([np.ones(len(x)), xc])
        try:
            beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
            y_grid[i] = beta[0]
        except np.linalg.LinAlgError:
            y_grid[i] = np.average(y, weights=w)

    return x_grid, y_grid


def isotonic_regression(x, y):
    """Isotonic (monotone-increasing) regression via PAVA algorithm."""
    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Pool Adjacent Violators Algorithm
    n = len(y_sorted)
    result = y_sorted.copy()
    weight = np.ones(n)
    blocks = list(range(n))

    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Merge blocks
            total_w = weight[i] + weight[i + 1]
            result[i] = (weight[i] * result[i] + weight[i + 1] * result[i + 1]) / total_w
            result[i + 1] = result[i]
            weight[i] = total_w
            weight[i + 1] = total_w

            # Check backward
            j = i
            while j > 0 and result[j - 1] > result[j]:
                total_w = weight[j - 1] + weight[j]
                result[j - 1] = (weight[j - 1] * result[j - 1] +
                                  weight[j] * result[j]) / total_w
                result[j] = result[j - 1]
                weight[j - 1] = total_w
                weight[j] = total_w
                j -= 1
            i = j + 1
        else:
            i += 1

    # Bin average to get unique x values
    from collections import defaultdict
    x_bins = defaultdict(list)
    for xi, yi in zip(x_sorted, result):
        x_bins[round(xi, 6)].append(yi)

    return x_sorted, result


def spline_smooth(x, y, s_factor=None):
    """Smoothing cubic spline."""
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]

    # Bin the data to avoid duplicate x values
    n_bins = 200
    x_grid = np.linspace(x_s.min(), x_s.max(), n_bins)
    y_binned = np.zeros(n_bins)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (x_s >= x_grid[i]) & (x_s < x_grid[i + 1])
        else:
            mask = (x_s >= x_grid[i])
        if np.sum(mask) > 0:
            y_binned[i] = np.mean(y_s[mask])
        elif i > 0:
            y_binned[i] = y_binned[i - 1]

    if s_factor is None:
        s_factor = n_bins * 0.1

    spl = UnivariateSpline(x_grid, y_binned, s=s_factor)
    return x_grid, spl(x_grid), spl


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def compute_residuals_from_mean(log_gbar, log_gobs, x_mean, y_mean):
    """Compute residuals from a non-parametric mean relation."""
    # Interpolate the mean relation to each data point
    y_pred = np.interp(log_gbar, x_mean, y_mean)
    return log_gobs - y_pred


def compute_scatter_profile(gbar, res, bin_width=0.30, offset=0.0, min_count=20):
    """Compute scatter as a function of log g_bar."""
    lo = max(np.percentile(gbar, 2), -13.0)
    hi = min(np.percentile(gbar, 98), -8.0)
    edges = np.arange(lo + offset, hi, bin_width)

    centers = []
    sigmas = []
    counts = []
    for edge in edges:
        mask = (gbar >= edge) & (gbar < edge + bin_width)
        n = np.sum(mask)
        if n >= min_count:
            centers.append(edge + bin_width / 2)
            sigmas.append(np.std(res[mask]))
            counts.append(int(n))

    return np.array(centers), np.array(sigmas), np.array(counts)


def numerical_derivative(x, y):
    """Central difference derivative."""
    dy = np.zeros_like(y)
    if len(y) < 2:
        return dy
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(1, len(y) - 1):
        dy[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return dy


def find_zero_crossings(x, y):
    """Find x values where y crosses zero."""
    crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            x_cross = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            crossings.append(x_cross)
    return crossings


def rar_function(log_gbar, a0=1.2e-10):
    """Standard RAR interpolating function."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


# ================================================================
# STEP 3: FIT NON-PARAMETRIC MEAN RELATIONS
# ================================================================
print("\n[3] Fitting non-parametric mean relations...")

# 3a. Standard RAR (for comparison)
res_rar = all_log_gobs - rar_function(all_log_gbar)
print(f"  RAR (parametric):  RMS = {np.std(res_rar):.4f} dex")

# 3b. LOESS
x_loess, y_loess = loess_smooth(all_log_gbar, all_log_gobs, frac=0.15)
res_loess = compute_residuals_from_mean(all_log_gbar, all_log_gobs, x_loess, y_loess)
print(f"  LOESS (frac=0.15): RMS = {np.std(res_loess):.4f} dex")

# 3c. Cubic spline
x_spline, y_spline, spl_func = spline_smooth(all_log_gbar, all_log_gobs)
res_spline = compute_residuals_from_mean(all_log_gbar, all_log_gobs, x_spline, y_spline)
print(f"  Cubic spline:      RMS = {np.std(res_spline):.4f} dex")

# 3d. Isotonic regression
x_iso, y_iso = isotonic_regression(all_log_gbar, all_log_gobs)
res_isotonic = compute_residuals_from_mean(all_log_gbar, all_log_gobs, x_iso, y_iso)
print(f"  Isotonic:          RMS = {np.std(res_isotonic):.4f} dex")


# ================================================================
# STEP 4: SCATTER PROFILES AND INVERSIONS FOR EACH METHOD
# ================================================================
print("\n" + "=" * 72)
print("PART 1: INVERSION POINT BY MEAN RELATION METHOD")
print("=" * 72)

methods = {
    'RAR (parametric)': res_rar,
    'LOESS': res_loess,
    'Cubic spline': res_spline,
    'Isotonic': res_isotonic,
}

method_results = {}

for method_name, residuals in methods.items():
    print(f"\n--- {method_name} ---")
    centers, sigmas, counts = compute_scatter_profile(
        all_log_gbar, residuals, bin_width=0.30)

    if len(centers) < 3:
        print(f"  Too few bins ({len(centers)})")
        method_results[method_name] = {'crossing': None, 'dist_gdagger': None}
        continue

    dsigma = numerical_derivative(centers, sigmas)
    crossings = find_zero_crossings(centers, dsigma)

    print(f"  {'log_gbar':>10} {'σ':>8} {'dσ/dx':>10} {'N':>6}")
    print(f"  {'-' * 38}")
    for i in range(len(centers)):
        flag = " <-- g†" if abs(centers[i] - LOG_G_DAGGER) < 0.18 else ""
        print(f"  {centers[i]:+10.2f} {sigmas[i]:8.4f} {dsigma[i]:+10.5f} "
              f"{counts[i]:6d}{flag}")

    if crossings:
        best = min(crossings, key=lambda c: abs(c - LOG_G_DAGGER))
        dist_gd = best - LOG_G_DAGGER
        tag = "MATCH" if abs(dist_gd) < 0.20 else ("partial" if abs(dist_gd) < 0.50 else "miss")
        print(f"  Inversion at: {best:+.3f} (Δ from g† = {dist_gd:+.3f}) [{tag}]")
        method_results[method_name] = {
            'crossing': round(float(best), 4),
            'dist_gdagger': round(float(dist_gd), 4),
            'all_crossings': [round(float(c), 4) for c in crossings],
        }
    else:
        print(f"  No zero-crossing found")
        method_results[method_name] = {'crossing': None, 'dist_gdagger': None}

# Summary table
print("\n" + "=" * 72)
print("SUMMARY: INVERSION POINT BY METHOD")
print("=" * 72)
print(f"\n  {'Method':<20} {'Crossing':>10} {'Δ from g†':>10} {'Status':>10}")
print(f"  {'-' * 55}")
for method_name, mr in method_results.items():
    if mr['crossing'] is not None:
        status = "MATCH" if abs(mr['dist_gdagger']) < 0.20 else "partial"
        print(f"  {method_name:<20} {mr['crossing']:+10.3f} "
              f"{mr['dist_gdagger']:+10.3f} {status:>10}")
    else:
        print(f"  {method_name:<20} {'--':>10} {'--':>10} {'no cross':>10}")

# Check convergence
valid_crossings = [mr['crossing'] for mr in method_results.values()
                   if mr['crossing'] is not None]
if len(valid_crossings) >= 3:
    mean_crossing = np.mean(valid_crossings)
    std_crossing = np.std(valid_crossings)
    mean_dist = mean_crossing - LOG_G_DAGGER
    n_match = sum(1 for mr in method_results.values()
                  if mr['crossing'] is not None and abs(mr['dist_gdagger']) < 0.20)
    print(f"\n  Mean crossing: {mean_crossing:+.3f} ± {std_crossing:.3f}")
    print(f"  Mean distance from g†: {mean_dist:+.3f}")
    print(f"  Methods within 0.20 dex of g†: {n_match}/{len(valid_crossings)}")

    if n_match == len(valid_crossings):
        inversion_verdict = "ALL_MATCH"
        print(f"\n  >>> ALL METHODS AGREE: Inversion at g† is NOT an artifact")
        print(f"  >>> of the assumed functional form.")
    elif n_match >= 2:
        inversion_verdict = "MAJORITY_MATCH"
        print(f"\n  >>> MAJORITY of methods confirm inversion near g†.")
    else:
        inversion_verdict = "NO_CONSENSUS"
else:
    inversion_verdict = "INSUFFICIENT"


# ================================================================
# STEP 5: BINNING ROBUSTNESS ACROSS METHODS
# ================================================================
print("\n" + "=" * 72)
print("BINNING ROBUSTNESS (per method, 3 widths × 3 offsets)")
print("=" * 72)

bin_widths = [0.25, 0.30, 0.40]
offsets = [0.0, 0.08, 0.15]

robustness_by_method = {}

for method_name, residuals in methods.items():
    configs = []
    for bw in bin_widths:
        for off in offsets:
            c, s, n = compute_scatter_profile(all_log_gbar, residuals,
                                               bin_width=bw, offset=off)
            if len(c) >= 3:
                ds = numerical_derivative(c, s)
                zc = find_zero_crossings(c, ds)
                if zc:
                    best = min(zc, key=lambda x: abs(x - LOG_G_DAGGER))
                    configs.append({
                        'bw': bw, 'off': off,
                        'crossing': round(float(best), 4),
                        'dist': round(float(best - LOG_G_DAGGER), 4),
                    })

    n_within = sum(1 for c in configs if abs(c['dist']) < 0.20)
    robustness_by_method[method_name] = {
        'n_configs': len(configs),
        'n_within_020': n_within,
        'crossings': configs,
    }
    print(f"  {method_name}: {n_within}/{len(configs)} configs within 0.20 dex of g†")


# ================================================================
# PART 2: SEPARATE MEAN RELATIONS BY ENVIRONMENT
# ================================================================
print("\n" + "=" * 72)
print("PART 2: SEPARATE MEAN RELATIONS BY ENVIRONMENT")
print("=" * 72)

field_mask = all_env == 'field'
dense_mask = all_env == 'dense'

gbar_field = all_log_gbar[field_mask]
gobs_field = all_log_gobs[field_mask]
gbar_dense = all_log_gbar[dense_mask]
gobs_dense = all_log_gobs[dense_mask]

print(f"\n  Field: {len(gbar_field)} points")
print(f"  Dense: {len(gbar_dense)} points")

# Fit separate LOESS to each environment
print("\n  Fitting separate LOESS to field and dense...")
x_loess_f, y_loess_f = loess_smooth(gbar_field, gobs_field, frac=0.15)
x_loess_d, y_loess_d = loess_smooth(gbar_dense, gobs_dense, frac=0.15)

# Compute residuals from each group's OWN mean
res_field_own = compute_residuals_from_mean(gbar_field, gobs_field,
                                             x_loess_f, y_loess_f)
res_dense_own = compute_residuals_from_mean(gbar_dense, gobs_dense,
                                             x_loess_d, y_loess_d)

print(f"  Field residual RMS (own LOESS): {np.std(res_field_own):.4f} dex")
print(f"  Dense residual RMS (own LOESS): {np.std(res_dense_own):.4f} dex")

# Fit separate splines
print("\n  Fitting separate cubic splines to field and dense...")
x_spl_f, y_spl_f, _ = spline_smooth(gbar_field, gobs_field)
x_spl_d, y_spl_d, _ = spline_smooth(gbar_dense, gobs_dense)

res_field_own_spl = compute_residuals_from_mean(gbar_field, gobs_field,
                                                  x_spl_f, y_spl_f)
res_dense_own_spl = compute_residuals_from_mean(gbar_dense, gobs_dense,
                                                  x_spl_d, y_spl_d)

print(f"  Field residual RMS (own spline): {np.std(res_field_own_spl):.4f} dex")
print(f"  Dense residual RMS (own spline): {np.std(res_dense_own_spl):.4f} dex")


# ================================================================
# STEP 6: SCATTER TRENDS WITH ENVIRONMENT-SPECIFIC MEANS
# ================================================================
print("\n" + "=" * 72)
print("SCATTER TRENDS WITH ENVIRONMENT-SPECIFIC MEANS")
print("=" * 72)

env_specific_methods = {
    'LOESS (env-specific)': (gbar_field, res_field_own,
                              gbar_dense, res_dense_own),
    'Spline (env-specific)': (gbar_field, res_field_own_spl,
                               gbar_dense, res_dense_own_spl),
}

# Also use the shared LOESS/spline for comparison
env_shared_methods = {
    'LOESS (shared mean)': (
        gbar_field,
        compute_residuals_from_mean(gbar_field, gobs_field, x_loess, y_loess),
        gbar_dense,
        compute_residuals_from_mean(gbar_dense, gobs_dense, x_loess, y_loess),
    ),
    'Spline (shared mean)': (
        gbar_field,
        compute_residuals_from_mean(gbar_field, gobs_field, x_spline, y_spline),
        gbar_dense,
        compute_residuals_from_mean(gbar_dense, gobs_dense, x_spline, y_spline),
    ),
}

all_env_methods = {**env_shared_methods, **env_specific_methods}
env_scatter_results = {}

accel_bins = [
    ('very_low', -13.0, -11.0),
    ('low', -11.0, -10.3),
    ('transition', -10.3, -9.5),
    ('high', -9.5, -8.0),
]

for method_name, (gb_f, res_f, gb_d, res_d) in all_env_methods.items():
    print(f"\n--- {method_name} ---")
    print(f"  {'Regime':<12} {'σ_field':>8} {'σ_dense':>8} {'Δσ':>8} "
          f"{'Levene_p':>9} {'N_f':>5} {'N_d':>5}")
    print(f"  {'-' * 58}")

    bin_results = []
    for regime, lo, hi in accel_bins:
        f_mask = (gb_f >= lo) & (gb_f < hi)
        d_mask = (gb_d >= lo) & (gb_d < hi)
        nf = np.sum(f_mask)
        nd = np.sum(d_mask)

        if nf >= 10 and nd >= 5:
            sf = np.std(res_f[f_mask])
            sd = np.std(res_d[d_mask])
            delta = sd - sf
            try:
                _, lev_p = levene(res_f[f_mask], res_d[d_mask])
            except Exception:
                lev_p = np.nan

            bin_results.append({
                'regime': regime,
                'sigma_field': round(float(sf), 4),
                'sigma_dense': round(float(sd), 4),
                'delta_sigma': round(float(delta), 4),
                'levene_p': round(float(lev_p), 4),
                'n_field': int(nf), 'n_dense': int(nd),
            })
            tag = "*" if lev_p < 0.05 else ""
            print(f"  {regime:<12} {sf:8.4f} {sd:8.4f} {delta:+8.4f} "
                  f"{lev_p:9.4f}{tag} {nf:5d} {nd:5d}")
        else:
            bin_results.append({
                'regime': regime,
                'sigma_field': None, 'sigma_dense': None,
                'delta_sigma': None, 'levene_p': None,
                'n_field': int(nf), 'n_dense': int(nd),
            })
            print(f"  {regime:<12} {'--':>8} {'--':>8} {'--':>8} "
                  f"{'--':>9}  {nf:5d} {nd:5d}")

    # Assess pattern
    valid_bins = [b for b in bin_results if b['delta_sigma'] is not None]
    low_bins = [b for b in valid_bins if b['regime'] in ('very_low', 'low')]
    high_bins = [b for b in valid_bins if b['regime'] in ('transition', 'high')]

    if low_bins and high_bins:
        low_delta = np.mean([b['delta_sigma'] for b in low_bins])
        high_delta = np.mean([b['delta_sigma'] for b in high_bins])
        low_p_min = min(b['levene_p'] for b in low_bins)

        print(f"\n  Low-accel mean Δσ: {low_delta:+.4f}")
        print(f"  High-accel mean Δσ: {high_delta:+.4f}")

        if abs(low_delta) < 0.010 and abs(high_delta) < 0.010:
            verdict = "UNIFORM_BOTH"
            print(f"  -> Scatter uniform across both regimes")
        elif low_delta < -0.005 and high_delta > -0.005:
            verdict = "TWO_REGIME"
            print(f"  -> TWO-REGIME: dense < field at low, converges/flips at high")
        elif low_delta > 0.005:
            verdict = "DENSE_HIGHER_LOW"
            print(f"  -> Dense scatter > field at low accel — tidal disruption")
        else:
            verdict = "OTHER"
            print(f"  -> Other pattern")
    else:
        verdict = "INSUFFICIENT"

    env_scatter_results[method_name] = {
        'bins': bin_results,
        'verdict': verdict,
    }


# ================================================================
# STEP 7: OVERALL SCATTER PROFILE BY ENVIRONMENT (non-parametric)
# ================================================================
print("\n" + "=" * 72)
print("ENVIRONMENT SCATTER PROFILES (LOESS env-specific residuals)")
print("=" * 72)

bw = 0.30
print(f"\n  {'log_gbar':>10} {'σ_field':>8} {'σ_dense':>8} {'Δσ':>8}")
print(f"  {'-' * 40}")

c_f, s_f, n_f = compute_scatter_profile(gbar_field, res_field_own,
                                          bin_width=bw, min_count=15)
c_d, s_d, n_d = compute_scatter_profile(gbar_dense, res_dense_own,
                                          bin_width=bw, min_count=10)

env_profile_rows = []
for i in range(len(c_f)):
    # Find matching dense bin
    d_match = None
    for j in range(len(c_d)):
        if abs(c_f[i] - c_d[j]) < 0.05:
            d_match = j
            break
    if d_match is not None:
        delta = s_d[d_match] - s_f[i]
        flag = " <-- g†" if abs(c_f[i] - LOG_G_DAGGER) < 0.20 else ""
        print(f"  {c_f[i]:+10.2f} {s_f[i]:8.4f} {s_d[d_match]:8.4f} "
              f"{delta:+8.4f}{flag}")
        env_profile_rows.append({
            'log_gbar': round(float(c_f[i]), 3),
            'sigma_field': round(float(s_f[i]), 4),
            'sigma_dense': round(float(s_d[d_match]), 4),
            'delta': round(float(delta), 4),
        })
    else:
        print(f"  {c_f[i]:+10.2f} {s_f[i]:8.4f} {'--':>8} {'--':>8}")


# ================================================================
# FINAL VERDICT
# ================================================================
print("\n" + "=" * 72)
print("FINAL VERDICT")
print("=" * 72)

print(f"\n  PART 1 — Non-parametric inversion:")
n_methods_match = sum(1 for mr in method_results.values()
                      if mr['crossing'] is not None and abs(mr['dist_gdagger']) < 0.20)
n_methods_total = sum(1 for mr in method_results.values()
                      if mr['crossing'] is not None)
print(f"    {n_methods_match}/{n_methods_total} methods have inversion within "
      f"0.20 dex of g†")
print(f"    Verdict: {inversion_verdict}")

print(f"\n  PART 2 — Environment-specific means:")
for method_name, er in env_scatter_results.items():
    print(f"    {method_name}: {er['verdict']}")

# Overall
all_verdicts = [er['verdict'] for er in env_scatter_results.values()]
env_specific_verdicts = [env_scatter_results[k]['verdict']
                          for k in env_specific_methods.keys()
                          if k in env_scatter_results]

print(f"\n  Overall inversion: {'ROBUST' if inversion_verdict in ('ALL_MATCH', 'MAJORITY_MATCH') else 'NOT ROBUST'}")
print(f"  Overall env pattern: {', '.join(set(all_verdicts))}")


# ================================================================
# SAVE
# ================================================================
summary = {
    'test_name': 'nonparametric_inversion_robustness',
    'n_galaxies': n_gals,
    'n_points': len(all_log_gbar),
    'n_field_gals': n_field_gals,
    'n_dense_gals': n_dense_gals,
    'part1_inversion_by_method': method_results,
    'part1_verdict': inversion_verdict,
    'part1_binning_robustness': robustness_by_method,
    'part2_env_scatter': env_scatter_results,
    'part2_env_profile': env_profile_rows,
}

outpath = os.path.join(RESULTS_DIR, 'summary_nonparametric_inversion.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("=" * 72)
print("Done.")
