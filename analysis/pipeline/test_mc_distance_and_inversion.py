#!/usr/bin/env python3
"""
Monte Carlo Distance Error Injection & Empirical Inversion Point Analysis
==========================================================================

Two analyses critical for the Letter:

PART 1 — Monte Carlo Distance Error Injection:
  Inject known Gaussian distance errors (5%, 10%, 15%, 20%) into ALL galaxies,
  recompute RAR scatter in acceleration bins, repeat 1000 times.
  Question: does distance error inflate scatter at low accelerations?
  If not, the universal scatter at g < g† is a real physical result,
  not a distance artifact.

  Key insight from CF4 analysis: changing a galaxy's distance shifts its
  MEAN residual but barely touches its WITHIN-GALAXY scatter. So the
  question is really: does the mean-residual dispersion across galaxies
  change when we add distance noise? And does that change differ by
  environment?

PART 2 — Empirical Inversion Points:
  Compute scatter σ and skewness as functions of acceleration in fine bins.
  Take derivatives: dσ/d(log g_bar) and d(skew)/d(log g_bar).
  Zero-crossings of these derivatives are empirical phase boundaries —
  where the physics changes character.

  BEC prediction: inversion at g_bar ≈ g† = 1.2×10⁻¹⁰ m/s² (log = -9.92)
  where the condensate fraction transitions from dominant to negligible.
"""

import os
import json
import numpy as np
from scipy.stats import levene, skew, kurtosis
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


def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


# Environment classification
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
def load_sparc():
    """Load SPARC rotation curves and properties."""
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

    return galaxies, sparc_props


def compute_residuals_with_distance_perturbation(galaxies, sparc_props,
                                                   distance_error_frac=0.0,
                                                   rng=None):
    """
    Compute RAR residuals with optional distance perturbation.

    If distance_error_frac > 0, each galaxy's distance is perturbed by
    D_perturbed = D * (1 + N(0, distance_error_frac))
    before computing accelerations.
    """
    results = {}

    for name, gdata in galaxies.items():
        if name not in sparc_props:
            continue
        prop = sparc_props[name]
        if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
            continue

        D_true = prop['D']

        # Apply distance perturbation
        if distance_error_frac > 0 and rng is not None:
            D_use = D_true * (1 + rng.normal(0, distance_error_frac))
            D_use = max(D_use, 0.1)  # floor
        else:
            D_use = D_true

        D_ratio = D_use / gdata['dist']

        R = gdata['R'] * D_ratio
        Vobs = gdata['Vobs']
        sqrt_ratio = np.sqrt(D_ratio)
        Vgas = gdata['Vgas'] * sqrt_ratio
        Vdisk = gdata['Vdisk'] * sqrt_ratio
        Vbul = gdata['Vbul'] * sqrt_ratio

        Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
        gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
        gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

        valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
        if np.sum(valid) < 5:
            continue

        log_gbar = np.log10(gbar_SI[valid])
        log_gobs = np.log10(gobs_SI[valid])
        log_gobs_rar = rar_function(log_gbar)
        log_res = log_gobs - log_gobs_rar

        env = classify_env(name)

        results[name] = {
            'log_gbar': log_gbar,
            'log_res': log_res,
            'mean_res': float(np.mean(log_res)),
            'std_res': float(np.std(log_res)),
            'n_points': len(log_res),
            'env': env,
            'D_true': D_true,
            'D_use': D_use,
        }

    return results


def compute_binned_scatter(results, gbar_lo, gbar_hi, env_filter=None):
    """Compute scatter of RAR residuals in an acceleration bin."""
    pts = []
    for r in results.values():
        if env_filter and r['env'] != env_filter:
            continue
        mask = (r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)
        if np.any(mask):
            pts.extend(r['log_res'][mask])
    pts = np.array(pts)
    if len(pts) < 5:
        return np.nan, 0
    return float(np.std(pts)), len(pts)


# ================================================================
# PART 1: MONTE CARLO DISTANCE ERROR INJECTION
# ================================================================
print("=" * 72)
print("PART 1: MONTE CARLO DISTANCE ERROR INJECTION")
print("=" * 72)

galaxies, sparc_props = load_sparc()
print(f"  Loaded {len(galaxies)} galaxies, {len(sparc_props)} with properties")

# Baseline (no error injection)
results_baseline = compute_residuals_with_distance_perturbation(
    galaxies, sparc_props, distance_error_frac=0.0)
n_dense = sum(1 for r in results_baseline.values() if r['env'] == 'dense')
n_field = sum(1 for r in results_baseline.values() if r['env'] == 'field')
print(f"  After cuts: {len(results_baseline)} galaxies ({n_dense} dense, {n_field} field)")

# Acceleration bins
bins_2 = [
    (-13.0, -10.5, 'LOW (condensate)'),
    (-10.5, -8.0,  'HIGH (baryon)'),
]
bins_4 = [
    (-13.0, -11.5, 'Deep DM'),
    (-11.5, -10.5, 'Transition'),
    (-10.5, -9.5,  'Baryon-dom'),
    (-9.5,  -8.0,  'High gbar'),
]

# Baseline scatter per bin
print(f"\n  Baseline (zero injected error):")
print(f"  {'Regime':20s} {'σ_all':>8s} {'σ_dense':>8s} {'σ_field':>8s} {'Δσ(f-d)':>8s} {'N_all':>6s}")
print(f"  {'-'*56}")
for lo, hi, label in bins_2 + [(-13.0, -8.0, 'OVERALL')]:
    sa, na = compute_binned_scatter(results_baseline, lo, hi)
    sd, nd = compute_binned_scatter(results_baseline, lo, hi, 'dense')
    sf, nf = compute_binned_scatter(results_baseline, lo, hi, 'field')
    delta = sf - sd if not (np.isnan(sf) or np.isnan(sd)) else np.nan
    delta_str = f"{delta:+8.4f}" if not np.isnan(delta) else "     ---"
    print(f"  {label:20s} {sa:8.4f} {sd:8.4f} {sf:8.4f} {delta_str} {na:6d}")

# Monte Carlo injection
error_fracs = [0.05, 0.10, 0.15, 0.20, 0.30]
N_MC = 1000

print(f"\n  Running {N_MC} Monte Carlo iterations for each error level...")
print(f"  Injecting into ALL galaxies (both environments equally).")

mc_results = {}

for frac in error_fracs:
    rng = np.random.default_rng(42)

    # Storage for each MC iteration
    scatter_low_all = np.zeros(N_MC)
    scatter_low_dense = np.zeros(N_MC)
    scatter_low_field = np.zeros(N_MC)
    scatter_high_all = np.zeros(N_MC)
    scatter_high_dense = np.zeros(N_MC)
    scatter_high_field = np.zeros(N_MC)
    delta_low = np.zeros(N_MC)
    delta_high = np.zeros(N_MC)

    for i in range(N_MC):
        res = compute_residuals_with_distance_perturbation(
            galaxies, sparc_props, distance_error_frac=frac, rng=rng)

        scatter_low_all[i], _ = compute_binned_scatter(res, -13.0, -10.5)
        scatter_low_dense[i], _ = compute_binned_scatter(res, -13.0, -10.5, 'dense')
        scatter_low_field[i], _ = compute_binned_scatter(res, -13.0, -10.5, 'field')
        scatter_high_all[i], _ = compute_binned_scatter(res, -10.5, -8.0)
        scatter_high_dense[i], _ = compute_binned_scatter(res, -10.5, -8.0, 'dense')
        scatter_high_field[i], _ = compute_binned_scatter(res, -10.5, -8.0, 'field')

        delta_low[i] = scatter_low_field[i] - scatter_low_dense[i]
        delta_high[i] = scatter_high_field[i] - scatter_high_dense[i]

    # Compute statistics
    baseline_low_all, _ = compute_binned_scatter(results_baseline, -13.0, -10.5)
    baseline_low_dense, _ = compute_binned_scatter(results_baseline, -13.0, -10.5, 'dense')
    baseline_low_field, _ = compute_binned_scatter(results_baseline, -13.0, -10.5, 'field')
    baseline_high_all, _ = compute_binned_scatter(results_baseline, -10.5, -8.0)
    baseline_high_dense, _ = compute_binned_scatter(results_baseline, -10.5, -8.0, 'dense')
    baseline_high_field, _ = compute_binned_scatter(results_baseline, -10.5, -8.0, 'field')

    mc_results[f'{int(frac*100)}pct'] = {
        'error_frac': frac,
        'n_mc': N_MC,
        'low_accel': {
            'baseline_all': round(float(baseline_low_all), 5),
            'mc_mean_all': round(float(np.mean(scatter_low_all)), 5),
            'mc_std_all': round(float(np.std(scatter_low_all)), 5),
            'inflation_all': round(float(np.mean(scatter_low_all) - baseline_low_all), 5),
            'baseline_dense': round(float(baseline_low_dense), 5),
            'mc_mean_dense': round(float(np.mean(scatter_low_dense)), 5),
            'baseline_field': round(float(baseline_low_field), 5),
            'mc_mean_field': round(float(np.mean(scatter_low_field)), 5),
            'baseline_delta': round(float(baseline_low_field - baseline_low_dense), 5),
            'mc_mean_delta': round(float(np.mean(delta_low)), 5),
            'mc_std_delta': round(float(np.std(delta_low)), 5),
            'mc_ci95_delta': [round(float(np.percentile(delta_low, 2.5)), 5),
                              round(float(np.percentile(delta_low, 97.5)), 5)],
        },
        'high_accel': {
            'baseline_all': round(float(baseline_high_all), 5),
            'mc_mean_all': round(float(np.mean(scatter_high_all)), 5),
            'mc_std_all': round(float(np.std(scatter_high_all)), 5),
            'inflation_all': round(float(np.mean(scatter_high_all) - baseline_high_all), 5),
            'baseline_dense': round(float(baseline_high_dense), 5),
            'mc_mean_dense': round(float(np.mean(scatter_high_dense)), 5),
            'baseline_field': round(float(baseline_high_field), 5),
            'mc_mean_field': round(float(np.mean(scatter_high_field)), 5),
            'baseline_delta': round(float(baseline_high_field - baseline_high_dense), 5),
            'mc_mean_delta': round(float(np.mean(delta_high)), 5),
            'mc_std_delta': round(float(np.std(delta_high)), 5),
            'mc_ci95_delta': [round(float(np.percentile(delta_high, 2.5)), 5),
                              round(float(np.percentile(delta_high, 97.5)), 5)],
        },
    }

    print(f"\n  --- {int(frac*100)}% distance error injection ---")
    print(f"  LOW accel (condensate):  baseline σ={baseline_low_all:.4f}"
          f"  → MC mean σ={np.mean(scatter_low_all):.4f}"
          f"  (inflation: {np.mean(scatter_low_all)-baseline_low_all:+.4f} dex)")
    print(f"    Δσ(f-d): baseline={baseline_low_field-baseline_low_dense:+.4f}"
          f"  → MC mean={np.mean(delta_low):+.4f} ± {np.std(delta_low):.4f}"
          f"  95%CI=[{np.percentile(delta_low, 2.5):+.4f}, {np.percentile(delta_low, 97.5):+.4f}]")
    print(f"  HIGH accel (baryon):     baseline σ={baseline_high_all:.4f}"
          f"  → MC mean σ={np.mean(scatter_high_all):.4f}"
          f"  (inflation: {np.mean(scatter_high_all)-baseline_high_all:+.4f} dex)")
    print(f"    Δσ(f-d): baseline={baseline_high_field-baseline_high_dense:+.4f}"
          f"  → MC mean={np.mean(delta_high):+.4f} ± {np.std(delta_high):.4f}"
          f"  95%CI=[{np.percentile(delta_high, 2.5):+.4f}, {np.percentile(delta_high, 97.5):+.4f}]")

# Summary table
print(f"\n{'='*72}")
print("MC INJECTION SUMMARY")
print(f"{'='*72}")
print(f"\n  {'Error':>6s} | {'LOW ACCEL scatter':^30s} | {'HIGH ACCEL scatter':^30s}")
print(f"  {'%':>6s} | {'baseline':>8s} {'MC mean':>8s} {'inflate':>8s} | {'baseline':>8s} {'MC mean':>8s} {'inflate':>8s}")
print(f"  {'-'*72}")
print(f"  {'0':>6s} | {baseline_low_all:8.4f} {baseline_low_all:8.4f} {0:+8.4f} | {baseline_high_all:8.4f} {baseline_high_all:8.4f} {0:+8.4f}")
for frac in error_fracs:
    key = f'{int(frac*100)}pct'
    lo = mc_results[key]['low_accel']
    hi = mc_results[key]['high_accel']
    print(f"  {int(frac*100):>5d}% | {lo['baseline_all']:8.4f} {lo['mc_mean_all']:8.4f} {lo['inflation_all']:+8.4f}"
          f" | {hi['baseline_all']:8.4f} {hi['mc_mean_all']:8.4f} {hi['inflation_all']:+8.4f}")

print(f"\n  Δσ(field-dense) stability under injection:")
print(f"  {'Error':>6s} | {'LOW Δσ baseline':>14s} {'LOW Δσ MC':>10s} {'± MC':>8s} | {'HIGH Δσ baseline':>16s} {'HIGH Δσ MC':>11s} {'± MC':>8s}")
print(f"  {'-'*80}")
for frac in error_fracs:
    key = f'{int(frac*100)}pct'
    lo = mc_results[key]['low_accel']
    hi = mc_results[key]['high_accel']
    print(f"  {int(frac*100):>5d}% | {lo['baseline_delta']:+14.4f} {lo['mc_mean_delta']:+10.4f} {lo['mc_std_delta']:8.4f}"
          f" | {hi['baseline_delta']:+16.4f} {hi['mc_mean_delta']:+11.4f} {hi['mc_std_delta']:8.4f}")


# ================================================================
# PART 2: EMPIRICAL INVERSION POINTS
# ================================================================
print(f"\n\n{'='*72}")
print("PART 2: EMPIRICAL INVERSION POINTS")
print(f"{'='*72}")
print(f"  Finding where dσ/d(log g_bar) and d(skew)/d(log g_bar) change sign")
print(f"  BEC prediction: inversion at log g_bar ≈ {LOG_G_DAGGER:.2f} (g†)")

# Fine bins for derivative analysis
n_fine = 15
fine_edges = np.linspace(-12.5, -8.5, n_fine + 1)
fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2
bin_width = fine_edges[1] - fine_edges[0]

# Collect all residuals
all_pts = []
for r in results_baseline.values():
    for i in range(len(r['log_gbar'])):
        all_pts.append((r['log_gbar'][i], r['log_res'][i], r['env']))
all_pts_arr = np.array([(p[0], p[1]) for p in all_pts])
all_envs = [p[2] for p in all_pts]

# Per-bin statistics
print(f"\n  {'Bin center':>10s} {'N_all':>6s} {'σ_all':>8s} {'skew':>8s} {'kurt':>8s}"
      f" {'σ_dense':>8s} {'σ_field':>8s} {'Δσ(f-d)':>8s}")
print(f"  {'-'*72}")

bin_stats = []
for j in range(n_fine):
    lo, hi = fine_edges[j], fine_edges[j+1]
    center = fine_centers[j]

    # All points
    mask = (all_pts_arr[:, 0] >= lo) & (all_pts_arr[:, 0] < hi)
    res_all = all_pts_arr[mask, 1]

    # By environment
    d_mask = mask & np.array([e == 'dense' for e in all_envs])
    f_mask = mask & np.array([e == 'field' for e in all_envs])
    res_dense = all_pts_arr[d_mask, 1]
    res_field = all_pts_arr[f_mask, 1]

    if len(res_all) < 10:
        bin_stats.append({
            'center': center, 'n': len(res_all),
            'sigma': np.nan, 'skewness': np.nan, 'kurtosis_excess': np.nan,
            'sigma_dense': np.nan, 'sigma_field': np.nan, 'delta_sigma': np.nan,
        })
        continue

    s = float(np.std(res_all))
    sk = float(skew(res_all))
    ku = float(kurtosis(res_all, fisher=True))

    sd = float(np.std(res_dense)) if len(res_dense) >= 5 else np.nan
    sf = float(np.std(res_field)) if len(res_field) >= 5 else np.nan
    ds = sf - sd if not (np.isnan(sf) or np.isnan(sd)) else np.nan

    bin_stats.append({
        'center': center,
        'n': len(res_all),
        'n_dense': len(res_dense),
        'n_field': len(res_field),
        'sigma': s,
        'skewness': sk,
        'kurtosis_excess': ku,
        'sigma_dense': sd,
        'sigma_field': sf,
        'delta_sigma': ds,
    })

    ds_str = f"{ds:+8.4f}" if not np.isnan(ds) else "     ---"
    sd_str = f"{sd:8.4f}" if not np.isnan(sd) else "     ---"
    sf_str = f"{sf:8.4f}" if not np.isnan(sf) else "     ---"

    print(f"  {center:10.2f} {len(res_all):6d} {s:8.4f} {sk:+8.3f} {ku:+8.3f}"
          f" {sd_str} {sf_str} {ds_str}")

# Compute derivatives
valid_stats = [b for b in bin_stats if not np.isnan(b['sigma'])]
centers_v = np.array([b['center'] for b in valid_stats])
sigma_v = np.array([b['sigma'] for b in valid_stats])
skew_v = np.array([b['skewness'] for b in valid_stats])
kurt_v = np.array([b['kurtosis_excess'] for b in valid_stats])

# Environmental delta derivative
delta_stats = [b for b in bin_stats if not np.isnan(b.get('delta_sigma', np.nan))]
centers_d = np.array([b['center'] for b in delta_stats])
delta_v = np.array([b['delta_sigma'] for b in delta_stats])

# Numerical derivatives (central differences)
def numerical_derivative(x, y):
    """Central difference derivative."""
    dy = np.zeros_like(y)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(1, len(y) - 1):
        dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return dy


def find_zero_crossings(x, y):
    """Find x-values where y crosses zero (linear interpolation)."""
    crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i+1] < 0:  # sign change
            x_cross = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
            crossings.append(float(x_cross))
    return crossings

dsigma_dx = numerical_derivative(centers_v, sigma_v)
dskew_dx = numerical_derivative(centers_v, skew_v)
dkurt_dx = numerical_derivative(centers_v, kurt_v)
ddelta_dx = numerical_derivative(centers_d, delta_v)

# Find zero crossings
sigma_crossings = find_zero_crossings(centers_v, dsigma_dx)
skew_crossings = find_zero_crossings(centers_v, dskew_dx)
kurt_crossings = find_zero_crossings(centers_v, dkurt_dx)
delta_crossings = find_zero_crossings(centers_d, ddelta_dx)

# Also find zero crossings of the delta itself (where Δσ changes sign)
delta_sign_crossings = find_zero_crossings(centers_d, delta_v)

# Also find zero crossings of skewness itself
skew_sign_crossings = find_zero_crossings(centers_v, skew_v)

print(f"\n{'='*72}")
print("DERIVATIVE ANALYSIS — Zero Crossings (Inversion Points)")
print(f"{'='*72}")
print(f"  g† = 10^{LOG_G_DAGGER:.2f} m/s²")

print(f"\n  dσ/d(log g_bar) = 0 at log g_bar = {sigma_crossings}")
print(f"    → Scatter extrema: where scatter peaks or troughs")

print(f"\n  d(skewness)/d(log g_bar) = 0 at log g_bar = {skew_crossings}")
print(f"    → Skewness extrema: where asymmetry peaks")

print(f"\n  d(kurtosis)/d(log g_bar) = 0 at log g_bar = {kurt_crossings}")
print(f"    → Kurtosis extrema: where tail weight peaks")

print(f"\n  d(Δσ)/d(log g_bar) = 0 at log g_bar = {delta_crossings}")
print(f"    → Environmental scatter difference extrema")

print(f"\n  Δσ(field-dense) = 0 at log g_bar ≈ {delta_sign_crossings}")
print(f"    → Where environmental scatter advantage flips sign")

print(f"\n  skewness = 0 at log g_bar ≈ {skew_sign_crossings}")
print(f"    → Where residual distribution changes from left-skewed to right-skewed")

# Identify the most significant inversion point
all_inversions = []
for x in sigma_crossings:
    all_inversions.append(('dσ/dx=0', x, abs(x - LOG_G_DAGGER)))
for x in skew_crossings:
    all_inversions.append(('d(skew)/dx=0', x, abs(x - LOG_G_DAGGER)))
for x in delta_sign_crossings:
    all_inversions.append(('Δσ=0', x, abs(x - LOG_G_DAGGER)))
for x in skew_sign_crossings:
    all_inversions.append(('skew=0', x, abs(x - LOG_G_DAGGER)))

if all_inversions:
    all_inversions.sort(key=lambda t: t[2])  # sort by distance from g†
    print(f"\n  Inversion points ranked by proximity to g†:")
    for label, x, dist in all_inversions:
        marker = " ← CLOSEST TO g†" if dist == all_inversions[0][2] else ""
        print(f"    {label:20s} at log g_bar = {x:+.2f}  "
              f"(Δ from g† = {x - LOG_G_DAGGER:+.2f} dex){marker}")


# Print derivative table
print(f"\n  {'center':>8s} {'dσ/dx':>9s} {'d(skew)/dx':>11s} {'d(kurt)/dx':>11s}")
print(f"  {'-'*42}")
for i, c in enumerate(centers_v):
    print(f"  {c:8.2f} {dsigma_dx[i]:+9.4f} {dskew_dx[i]:+11.4f} {dkurt_dx[i]:+11.4f}")


# ================================================================
# PART 3: ENVIRONMENT-SPECIFIC INVERSION
# ================================================================
print(f"\n\n{'='*72}")
print("PART 3: ENVIRONMENT-SPECIFIC SCATTER PROFILES")
print(f"{'='*72}")
print(f"  BEC predicts: dense and field scatter profiles converge at low accel")
print(f"  and may diverge near the condensate boundary (g ~ g†)")

# Use coarser bins for per-environment analysis (need enough points)
n_env = 10
env_edges = np.linspace(-12.5, -8.5, n_env + 1)
env_centers = (env_edges[:-1] + env_edges[1:]) / 2

print(f"\n  {'center':>8s} {'σ_dense':>8s} {'σ_field':>8s} {'Δσ(f-d)':>8s} {'Lev_p':>9s}")
print(f"  {'-'*50}")

env_profile = []
for j in range(n_env):
    lo, hi = env_edges[j], env_edges[j+1]

    d_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= lo) & (r['log_gbar'] < hi)]
                            for r in results_baseline.values() if r['env'] == 'dense'])
    f_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= lo) & (r['log_gbar'] < hi)]
                            for r in results_baseline.values() if r['env'] == 'field'])

    if len(d_pts) < 5 or len(f_pts) < 5:
        print(f"  {env_centers[j]:8.2f}      ---      ---      ---       ---")
        continue

    sd, sf = np.std(d_pts), np.std(f_pts)
    delta = sf - sd
    stat_L, p_L = levene(d_pts, f_pts)

    sig = '***' if p_L < 0.001 else '**' if p_L < 0.01 else '*' if p_L < 0.05 else ''

    env_profile.append({
        'center': float(env_centers[j]),
        'sigma_dense': round(sd, 5),
        'sigma_field': round(sf, 5),
        'delta': round(delta, 5),
        'levene_p': round(float(p_L), 8),
        'n_dense': len(d_pts),
        'n_field': len(f_pts),
    })

    print(f"  {env_centers[j]:8.2f} {sd:8.4f} {sf:8.4f} {delta:+8.4f} {p_L:9.6f} {sig}")


# ================================================================
# SAVE ALL RESULTS
# ================================================================
print(f"\n{'='*72}")
print("SAVING RESULTS")
print(f"{'='*72}")

output = {
    'test_name': 'mc_distance_injection_and_inversion_points',
    'description': ('Monte Carlo distance error injection test + empirical '
                    'inversion point analysis via derivative zero-crossings'),
    'part1_mc_injection': {
        'n_galaxies': len(results_baseline),
        'n_dense': n_dense,
        'n_field': n_field,
        'n_mc_iterations': N_MC,
        'error_levels_tested': error_fracs,
        'results_by_error_level': mc_results,
    },
    'part2_inversion_points': {
        'bin_statistics': [b for b in bin_stats if not np.isnan(b.get('sigma', np.nan))],
        'derivatives': {
            'centers': [round(float(c), 3) for c in centers_v],
            'dsigma_dx': [round(float(d), 5) for d in dsigma_dx],
            'dskew_dx': [round(float(d), 5) for d in dskew_dx],
            'dkurt_dx': [round(float(d), 5) for d in dkurt_dx],
        },
        'zero_crossings': {
            'dsigma_dx': sigma_crossings,
            'dskew_dx': skew_crossings,
            'dkurt_dx': kurt_crossings,
            'delta_sigma': delta_sign_crossings,
            'skewness': skew_sign_crossings,
            'd_delta_dx': delta_crossings,
        },
        'g_dagger_log': LOG_G_DAGGER,
        'all_inversions_ranked': [
            {'type': t[0], 'log_gbar': round(t[1], 3),
             'distance_from_gdagger': round(t[2], 3)}
            for t in all_inversions
        ] if all_inversions else [],
    },
    'part3_env_profile': env_profile,
}

outpath = os.path.join(RESULTS_DIR, 'summary_mc_distance_and_inversion.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
