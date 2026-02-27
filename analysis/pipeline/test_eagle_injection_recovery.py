#!/usr/bin/env python3
"""
TEST C: SPARC → EAGLE Sampling Injection–Recovery
===================================================

Goal: Can EAGLE's 10 fixed radii detect a ~7 kpc periodic signal if present?

C1 — Core injection-recovery:
  1. Take 25 periodic + 42 non-periodic SPARC galaxies
  2. Resample each residual series onto EAGLE's fixed radii {1,3,5,10,20,30,40,50,70,100} kpc
  3. Run LS periodicity detection on resampled series (noise-free + noisy variants)
  4. Compute detection power curve, recovered wavelength distribution

C2 — Window matching:
  - Restrict SPARC to same radial window (1-100 kpc)
  - Also resample both in units of R_half (0.5-5 R_half, log spaced)

Deliverables:
  - Detection power curve: P(detect | λ_true ≈ 7 kpc)
  - Distribution of recovered λ_peak (bias + variance)
  - Effective false negative rate from EAGLE sampling

Russell Licht — BEC Dark Matter Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
kpc_m = 3.086e19
MIN_POINTS = 15
N_SURR = 200
PERM_ALPHA = 0.05

# EAGLE fixed aperture radii (kpc)
EAGLE_RADII = np.array([1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0])

N_NOISE_REAL = 500  # noise realizations per galaxy
MIN_EAGLE_PTS = 5   # minimum resampled points for LS to be meaningful

np.random.seed(42)

print("=" * 76)
print("TEST C: SPARC → EAGLE SAMPLING INJECTION-RECOVERY")
print("=" * 76)


def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


# ================================================================
# 1. LOAD SPARC DATA (same as previous tests)
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
            errv = float(line[33:38].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in rc_data:
            rc_data[name] = {'R': [], 'Vobs': [], 'errV': [], 'Vgas': [],
                             'Vdisk': [], 'Vbul': [], 'dist': dist}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['errV'].append(errv)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in ['R', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul']:
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
            'D': float(parts[1]),
            'Inc': float(parts[4]),
            'L36': float(parts[6]),
            'Reff': float(parts[8]),
            'Vflat': float(parts[14]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue


# Build per-galaxy data with RAR residuals AND original (non-detrended) residuals
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
    residuals = log_gobs - rar_function(log_gbar)  # raw residuals
    errV_sorted = errV[valid][sort_idx]
    Vobs_sorted = Vobs[valid][sort_idx]

    # Spline detrending
    var_eps = np.var(residuals)
    s_param = n * var_eps * 0.5
    try:
        spline = UnivariateSpline(R_sorted, residuals, k=min(3, n - 1), s=s_param)
        eps_det = residuals - spline(R_sorted)
    except Exception:
        eps_det = residuals - np.mean(residuals)

    R_extent = R_sorted[-1] - R_sorted[0]
    if R_extent <= 0:
        continue

    # LS periodicity test on original SPARC data
    std_eps = np.std(eps_det)
    if std_eps < 1e-30:
        continue
    y = (eps_det - np.mean(eps_det)) / std_eps

    f_min = 1.0 / R_extent
    f_max = (n / 2.0) / R_extent
    n_freq = min(500, 10 * n)
    freq_grid = np.linspace(f_min, f_max, n_freq)

    ls = LombScargle(R_sorted, y, fit_mean=False, center_data=True)
    power = ls.power(freq_grid)
    idx_peak = np.argmax(power)
    f_peak = float(freq_grid[idx_peak])
    power_peak = float(power[idx_peak])
    wl_peak = 1.0 / f_peak

    # Permutation test
    null_peaks = np.zeros(N_SURR)
    for s in range(N_SURR):
        y_shuf = perm_rng.permutation(y)
        ls_null = LombScargle(R_sorted, y_shuf, fit_mean=False, center_data=True)
        null_peaks[s] = np.max(ls_null.power(freq_grid))
    p_val = float(np.mean(null_peaks >= power_peak))

    # Residual noise level (for noise injection)
    # Convert velocity errors to log(g_obs) errors: d(log g)/dV ≈ 2/(V ln 10)
    log_g_err = 2.0 * errV_sorted / (Vobs_sorted * np.log(10))
    median_log_err = float(np.median(log_g_err))

    galaxy_data.append({
        'name': name,
        'R': R_sorted,
        'residuals_raw': residuals,
        'eps_det': eps_det,
        'log_g_err': log_g_err,
        'median_log_err': median_log_err,
        'n_pts': n,
        'R_extent': R_extent,
        'Reff': prop['Reff'],
        'Vflat': prop['Vflat'],
        'f_peak': f_peak,
        'wl_peak': wl_peak,
        'power_peak': power_peak,
        'perm_p': p_val,
        'is_periodic': p_val < PERM_ALPHA,
    })

n_total = len(galaxy_data)
periodic = [g for g in galaxy_data if g['is_periodic']]
nonperiodic = [g for g in galaxy_data if not g['is_periodic']]
n_per = len(periodic)
n_nper = len(nonperiodic)
print(f"  Total: {n_total}, Periodic: {n_per}, Non-periodic: {n_nper}")


# ================================================================
# 2. INJECTION-RECOVERY: RESAMPLE ONTO EAGLE RADII
# ================================================================
print(f"\n[2] Injection-recovery onto EAGLE fixed radii...")
print(f"    EAGLE radii: {EAGLE_RADII.tolist()} kpc")
print(f"    Noise realizations: {N_NOISE_REAL}")

noise_rng = np.random.default_rng(314)


def run_ls_detection(R_resamp, resid_resamp, n_surr=100):
    """Run LS + permutation test on resampled data. Returns (detected, wl_recovered, p_val)."""
    n = len(R_resamp)
    if n < MIN_EAGLE_PTS:
        return False, np.nan, 1.0

    std_r = np.std(resid_resamp)
    if std_r < 1e-30:
        return False, np.nan, 1.0

    y = (resid_resamp - np.mean(resid_resamp)) / std_r
    R_extent = R_resamp[-1] - R_resamp[0]
    if R_extent <= 0:
        return False, np.nan, 1.0

    f_min = 1.0 / R_extent
    f_max = (n / 2.0) / R_extent
    if f_max <= f_min:
        return False, np.nan, 1.0

    n_freq = min(300, max(50, 10 * n))
    freq_grid = np.linspace(f_min, f_max, n_freq)

    ls = LombScargle(R_resamp, y, fit_mean=False, center_data=True)
    power = ls.power(freq_grid)
    idx_peak = np.argmax(power)
    f_peak = freq_grid[idx_peak]
    power_peak = power[idx_peak]
    wl_recovered = 1.0 / f_peak if f_peak > 0 else np.nan

    # Quick permutation test
    rng_local = np.random.default_rng(hash(tuple(R_resamp)) % (2**31))
    null_peaks = np.zeros(n_surr)
    for s in range(n_surr):
        y_shuf = rng_local.permutation(y)
        ls_null = LombScargle(R_resamp, y_shuf, fit_mean=False, center_data=True)
        null_peaks[s] = np.max(ls_null.power(freq_grid))

    p_val = float(np.mean(null_peaks >= power_peak))
    detected = p_val < PERM_ALPHA
    return detected, float(wl_recovered), p_val


def resample_onto_radii(galaxy, target_radii, add_noise=False, noise_scale=1.0, rng=None):
    """Interpolate galaxy's detrended residuals onto target radii."""
    R = galaxy['R']
    eps = galaxy['eps_det']
    log_err = galaxy['log_g_err']

    # Only use target radii within galaxy's observed range
    r_min, r_max = R[0], R[-1]
    valid_mask = (target_radii >= r_min) & (target_radii <= r_max)
    R_target = target_radii[valid_mask]

    if len(R_target) < MIN_EAGLE_PTS:
        return None, None

    # Interpolate residuals (linear in physical space)
    interp_func = interp1d(R, eps, kind='linear', fill_value='extrapolate')
    resid_resampled = interp_func(R_target)

    if add_noise and rng is not None:
        # Interpolate error bars too
        err_func = interp1d(R, log_err, kind='linear', fill_value='extrapolate')
        err_resampled = err_func(R_target)
        noise = rng.normal(0, np.abs(err_resampled) * noise_scale)
        resid_resampled = resid_resampled + noise

    return R_target, resid_resampled


# Run injection-recovery for all galaxies
results_per_galaxy = []

for gi, g in enumerate(galaxy_data):
    name = g['name']
    is_per = g['is_periodic']
    wl_true = g['wl_peak']

    # --- Noise-free recovery ---
    R_eagle, resid_eagle = resample_onto_radii(g, EAGLE_RADII, add_noise=False)
    if R_eagle is None:
        results_per_galaxy.append({
            'name': name, 'is_periodic': is_per,
            'n_eagle_pts': 0, 'insufficient_coverage': True,
        })
        continue

    n_eagle = len(R_eagle)
    det_nf, wl_nf, p_nf = run_ls_detection(R_eagle, resid_eagle)

    # --- Noisy recovery (N_NOISE_REAL realizations) ---
    det_count = 0
    wl_recovered_list = []
    p_vals_list = []

    for nr in range(N_NOISE_REAL):
        R_noisy, resid_noisy = resample_onto_radii(
            g, EAGLE_RADII, add_noise=True, noise_scale=1.0, rng=noise_rng
        )
        if R_noisy is None:
            continue
        det, wl_rec, p_rec = run_ls_detection(R_noisy, resid_noisy, n_surr=50)
        if det:
            det_count += 1
        wl_recovered_list.append(wl_rec)
        p_vals_list.append(p_rec)

    n_trials = len(wl_recovered_list)
    det_rate = det_count / n_trials if n_trials > 0 else 0.0
    wl_arr = np.array(wl_recovered_list)
    valid_wl = wl_arr[np.isfinite(wl_arr)]

    results_per_galaxy.append({
        'name': name,
        'is_periodic': is_per,
        'wl_true': round(wl_true, 2),
        'n_sparc_pts': g['n_pts'],
        'R_extent': round(g['R_extent'], 2),
        'n_eagle_pts': n_eagle,
        'eagle_radii_used': R_eagle.tolist(),
        'insufficient_coverage': False,
        # Noise-free
        'noisefree_detected': det_nf,
        'noisefree_wl': round(wl_nf, 2) if np.isfinite(wl_nf) else None,
        'noisefree_p': round(p_nf, 4),
        # Noisy
        'detection_rate': round(det_rate, 4),
        'n_trials': n_trials,
        'wl_recovered_median': round(float(np.median(valid_wl)), 2) if len(valid_wl) > 0 else None,
        'wl_recovered_std': round(float(np.std(valid_wl)), 2) if len(valid_wl) > 5 else None,
        'wl_bias': round(float(np.median(valid_wl) - wl_true), 2) if len(valid_wl) > 0 else None,
    })

    if (gi + 1) % 20 == 0:
        print(f"    {gi+1}/{n_total} galaxies done...")

print(f"  Injection-recovery complete for {n_total} galaxies")


# ================================================================
# 3. AGGREGATE RESULTS
# ================================================================
print("\n[3] Aggregating results...")

# Separate by periodic / non-periodic
valid_results = [r for r in results_per_galaxy if not r.get('insufficient_coverage', False)]
per_results = [r for r in valid_results if r['is_periodic']]
nper_results = [r for r in valid_results if not r['is_periodic']]
insuf_results = [r for r in results_per_galaxy if r.get('insufficient_coverage', False)]

print(f"  Valid: {len(valid_results)}, Insufficient coverage: {len(insuf_results)}")

# Detection power (periodic galaxies)
det_rates_per = [r['detection_rate'] for r in per_results]
det_rates_nper = [r['detection_rate'] for r in nper_results]

mean_det_per = np.mean(det_rates_per) if det_rates_per else 0
mean_det_nper = np.mean(det_rates_nper) if det_rates_nper else 0

n_nf_det_per = sum(1 for r in per_results if r['noisefree_detected'])
n_nf_det_nper = sum(1 for r in nper_results if r['noisefree_detected'])

print(f"\n  PERIODIC GALAXIES (N={len(per_results)}):")
print(f"    Noise-free detection: {n_nf_det_per}/{len(per_results)} "
      f"({100*n_nf_det_per/len(per_results):.1f}%)")
print(f"    Mean noisy detection rate: {mean_det_per:.3f} ({mean_det_per*100:.1f}%)")
print(f"    Detection rate distribution: "
      f"median={np.median(det_rates_per):.3f}, "
      f"IQR=[{np.percentile(det_rates_per, 25):.3f}, {np.percentile(det_rates_per, 75):.3f}]")

print(f"\n  NON-PERIODIC GALAXIES (N={len(nper_results)}):")
print(f"    Noise-free detection: {n_nf_det_nper}/{len(nper_results)} "
      f"({100*n_nf_det_nper/len(nper_results):.1f}%)")
print(f"    Mean noisy detection rate: {mean_det_nper:.3f} ({mean_det_nper*100:.1f}%)")

# Wavelength recovery (periodic only)
wl_true_per = np.array([r['wl_true'] for r in per_results])
wl_rec_per = np.array([r['wl_recovered_median'] for r in per_results
                        if r['wl_recovered_median'] is not None])
wl_bias_per = np.array([r['wl_bias'] for r in per_results if r['wl_bias'] is not None])

if len(wl_rec_per) > 3:
    print(f"\n  WAVELENGTH RECOVERY (periodic, median across noise):")
    print(f"    True λ: median={np.median(wl_true_per):.1f} kpc")
    print(f"    Recovered λ: median={np.median(wl_rec_per):.1f} kpc")
    print(f"    Bias (recovered - true): median={np.median(wl_bias_per):.2f} kpc")

# False negative rate
fn_rate = 1.0 - mean_det_per
print(f"\n  EFFECTIVE FALSE NEGATIVE RATE: {fn_rate:.3f} ({fn_rate*100:.1f}%)")
print(f"  (This is the fraction of truly periodic SPARC galaxies that EAGLE sampling would miss)")


# ================================================================
# 4. DETECTION POWER CURVE: P(detect) vs λ_true
# ================================================================
print("\n[4] Computing detection power curve...")

# Bin periodic galaxies by their true wavelength
wl_true_all = np.array([r['wl_true'] for r in per_results])
det_rate_all = np.array([r['detection_rate'] for r in per_results])

# Create wavelength bins
wl_bins = [0, 3, 7, 15, 30, 100]
bin_labels = ['<3', '3-7', '7-15', '15-30', '>30']
power_curve = []

for i in range(len(wl_bins) - 1):
    lo, hi = wl_bins[i], wl_bins[i + 1]
    in_bin = (wl_true_all >= lo) & (wl_true_all < hi)
    if np.sum(in_bin) > 0:
        mean_rate = float(np.mean(det_rate_all[in_bin]))
        n_in_bin = int(np.sum(in_bin))
        power_curve.append({
            'wl_range': f'{lo}-{hi}',
            'label': bin_labels[i],
            'n_galaxies': n_in_bin,
            'mean_detection_rate': round(mean_rate, 4),
        })
        print(f"    λ ∈ [{lo}, {hi}) kpc: N={n_in_bin}, P(detect) = {mean_rate:.3f}")

# Finer: per-galaxy scatter
print(f"\n  Per-galaxy detection rate vs true wavelength:")
for r in sorted(per_results, key=lambda x: x['wl_true']):
    det_pct = r['detection_rate'] * 100
    marker = "***" if r['detection_rate'] > 0.3 else ""
    print(f"    {r['name']:<12} λ_true={r['wl_true']:6.1f} kpc, "
          f"N_eagle={r['n_eagle_pts']}, P(det)={det_pct:5.1f}% {marker}")


# ================================================================
# 5. C2: WINDOW MATCHING
# ================================================================
print("\n[5] C2: Window matching variants...")

# Variant A: Restrict SPARC to [1, 100] kpc window
print("\n  [5a] SPARC restricted to [1, 100] kpc (EAGLE window)...")

window_results = []
for g in galaxy_data:
    R = g['R']
    eps = g['eps_det']

    # Restrict to [1, 100] kpc
    mask = (R >= 1.0) & (R <= 100.0)
    R_win = R[mask]
    eps_win = eps[mask]

    if len(R_win) < MIN_EAGLE_PTS:
        continue

    det, wl_rec, p_rec = run_ls_detection(R_win, eps_win)
    window_results.append({
        'name': g['name'],
        'is_periodic_orig': g['is_periodic'],
        'n_pts_window': len(R_win),
        'detected_in_window': det,
        'wl_window': round(wl_rec, 2) if np.isfinite(wl_rec) else None,
        'p_window': round(p_rec, 4),
    })

n_win = len(window_results)
n_orig_per_in_win = sum(1 for r in window_results if r['is_periodic_orig'])
n_det_in_win = sum(1 for r in window_results if r['detected_in_window'])
n_per_det_in_win = sum(1 for r in window_results
                        if r['is_periodic_orig'] and r['detected_in_window'])

print(f"    Galaxies in [1,100] kpc window: {n_win}")
print(f"    Originally periodic: {n_orig_per_in_win}")
print(f"    Detected in window: {n_det_in_win} "
      f"({100*n_det_in_win/n_win:.1f}% of all)")
print(f"    Originally periodic + detected: {n_per_det_in_win} "
      f"({100*n_per_det_in_win/n_orig_per_in_win:.1f}% of periodic)")

# Variant B: Resample in units of R_eff (0.5-5 R_eff, log spaced)
print("\n  [5b] Resampling in R/R_eff coordinates (0.5-5 R_eff)...")

R_eff_factors = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])

rhalf_results = []
for g in galaxy_data:
    Reff = g['Reff']
    if Reff <= 0:
        continue

    target_R = R_eff_factors * Reff  # physical radii

    R_target, resid_target = resample_onto_radii(g, target_R, add_noise=False)
    if R_target is None:
        continue

    det, wl_rec, p_rec = run_ls_detection(R_target, resid_target)
    rhalf_results.append({
        'name': g['name'],
        'is_periodic_orig': g['is_periodic'],
        'Reff': round(Reff, 2),
        'n_pts_rhalf': len(R_target),
        'detected_rhalf': det,
        'wl_rhalf': round(wl_rec, 2) if np.isfinite(wl_rec) else None,
        'p_rhalf': round(p_rec, 4),
    })

n_rhalf = len(rhalf_results)
n_per_rhalf = sum(1 for r in rhalf_results if r['is_periodic_orig'])
n_det_rhalf = sum(1 for r in rhalf_results if r['detected_rhalf'])
n_per_det_rhalf = sum(1 for r in rhalf_results
                       if r['is_periodic_orig'] and r['detected_rhalf'])

print(f"    Galaxies with R_eff-based resampling: {n_rhalf}")
print(f"    Detected: {n_det_rhalf} ({100*n_det_rhalf/n_rhalf:.1f}%)")
if n_per_rhalf > 0:
    print(f"    Periodic + detected: {n_per_det_rhalf} "
          f"({100*n_per_det_rhalf/n_per_rhalf:.1f}% of periodic)")


# ================================================================
# 6. PUBLICATION FIGURES
# ================================================================
print("\n[6] Generating figures...")

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

# --- Figure 1: Detection power curve + per-galaxy scatter ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: Per-galaxy detection rate vs true wavelength
per_wl = np.array([r['wl_true'] for r in per_results])
per_det = np.array([r['detection_rate'] for r in per_results])
nper_wl = np.array([r['wl_true'] for r in nper_results])
nper_det = np.array([r['detection_rate'] for r in nper_results])

ax1.scatter(per_wl, per_det * 100, c='crimson', s=60, alpha=0.7,
            edgecolors='black', linewidth=0.5, zorder=5, label=f'Periodic (N={len(per_results)})')
ax1.scatter(nper_wl, nper_det * 100, c='royalblue', s=30, alpha=0.4,
            edgecolors='gray', linewidth=0.3, zorder=3, label=f'Non-periodic (N={len(nper_results)})')

ax1.axhline(5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, label='5% (expected false positive)')
ax1.axhline(30, color='firebrick', linestyle='--', linewidth=0.8, alpha=0.5, label='30% threshold')

ax1.set_xscale('log')
ax1.set_xlabel(r'True peak wavelength $\lambda_{\rm true}$ [kpc]', fontsize=12)
ax1.set_ylabel('Detection rate [%]', fontsize=12)
ax1.set_title('Detection Power: SPARC Signals on EAGLE Grid', fontsize=13)
ax1.legend(fontsize=8, loc='upper right')
ax1.set_ylim(-5, 105)

# Right: Recovered vs true wavelength
ax2.scatter(per_wl, [r['wl_recovered_median'] if r['wl_recovered_median'] else np.nan
                      for r in per_results],
            c='crimson', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

# 1:1 line
wl_range = np.array([0.3, 50])
ax2.plot(wl_range, wl_range, '--', color='gray', linewidth=1.0, label=r'$\lambda_{\rm rec} = \lambda_{\rm true}$')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'True $\lambda_{\rm peak}$ [kpc]', fontsize=12)
ax2.set_ylabel(r'Recovered $\lambda_{\rm peak}$ [kpc]', fontsize=12)
ax2.set_title('Wavelength Recovery (Periodic Galaxies)', fontsize=13)
ax2.legend(fontsize=9)

# Text box
txt = (f"Mean detection rate:\n"
       f"  Periodic: {mean_det_per*100:.1f}%\n"
       f"  Non-periodic: {mean_det_nper*100:.1f}%\n"
       f"False negative rate: {fn_rate*100:.1f}%")
ax1.text(0.03, 0.97, txt, transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig1.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'eagle_injection_recovery.png')
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig1_path}")
plt.close(fig1)

# --- Figure 2: Sampling illustration + window matching ---
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Example galaxy — SPARC data + EAGLE grid overlay
example = None
for g in periodic:
    if g['R_extent'] > 15 and g['n_pts'] > 25:
        example = g
        break
if example is None:
    example = periodic[0]

ax = axes[0]
ax.plot(example['R'], example['eps_det'], 'o-', color='steelblue', markersize=4,
        linewidth=1.0, label=f"SPARC ({example['n_pts']} pts)")

# EAGLE resampled
R_eagle, resid_eagle = resample_onto_radii(example, EAGLE_RADII, add_noise=False)
if R_eagle is not None:
    ax.plot(R_eagle, resid_eagle, 's', color='firebrick', markersize=8,
            markeredgecolor='black', linewidth=0.5, label=f'EAGLE grid ({len(R_eagle)} pts)', zorder=5)
    for r in EAGLE_RADII:
        if r >= example['R'][0] and r <= example['R'][-1]:
            ax.axvline(r, color='firebrick', alpha=0.1, linewidth=0.5)

ax.set_xlabel('Radius [kpc]', fontsize=11)
ax.set_ylabel('Detrended RAR residual', fontsize=11)
ax.set_title(f"Example: {example['name']} (λ={example['wl_peak']:.1f} kpc)", fontsize=12)
ax.legend(fontsize=8)

# Middle: Detection rate histogram
ax = axes[1]
ax.hist(np.array(det_rates_per) * 100, bins=12, color='crimson', alpha=0.6,
        edgecolor='black', linewidth=0.5, label='Periodic')
ax.hist(np.array(det_rates_nper) * 100, bins=12, color='royalblue', alpha=0.4,
        edgecolor='gray', linewidth=0.5, label='Non-periodic')
ax.axvline(5, color='gray', linestyle=':', linewidth=1.0)
ax.set_xlabel('Detection rate [%]', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Detection Rate Distribution', fontsize=12)
ax.legend(fontsize=9)

# Right: N_eagle_pts vs detection rate
ax = axes[2]
n_eagle_per = np.array([r['n_eagle_pts'] for r in per_results])
n_eagle_nper = np.array([r['n_eagle_pts'] for r in nper_results])
ax.scatter(n_eagle_per, np.array(det_rates_per) * 100, c='crimson', s=50, alpha=0.7,
           edgecolors='black', linewidth=0.5, label='Periodic')
ax.scatter(n_eagle_nper, np.array(det_rates_nper) * 100, c='royalblue', s=30, alpha=0.4,
           edgecolors='gray', linewidth=0.3, label='Non-periodic')
ax.set_xlabel('N EAGLE radii within galaxy extent', fontsize=11)
ax.set_ylabel('Detection rate [%]', fontsize=11)
ax.set_title('Detection vs Sampling Density', fontsize=12)
ax.legend(fontsize=9)

fig2.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'eagle_sampling_illustration.png')
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig2_path}")
plt.close(fig2)


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 76)
print("SUMMARY — TEST C: SPARC → EAGLE INJECTION-RECOVERY")
print("=" * 76)

print(f"\n  C1 — Core injection-recovery:")
print(f"    EAGLE sampling: 10 fixed radii, {MIN_EAGLE_PTS}+ needed for detection")
print(f"    Periodic galaxies: {len(per_results)} analyzed")
print(f"    Noise-free detection rate: {100*n_nf_det_per/len(per_results):.1f}%")
print(f"    Noisy detection rate (mean): {mean_det_per*100:.1f}%")
print(f"    False negative rate: {fn_rate*100:.1f}%")

if mean_det_per > 0.30:
    print(f"\n    => Detection power STAYS HIGH ({mean_det_per*100:.0f}%)")
    print(f"       If EAGLE shows low periodic fraction, the deficit is MEANINGFUL")
elif mean_det_per > 0.10:
    print(f"\n    => Detection power is MODERATE ({mean_det_per*100:.0f}%)")
    print(f"       EAGLE comparison weakened but not uninformative")
else:
    print(f"\n    => Detection power COLLAPSES to ~{mean_det_per*100:.0f}%")
    print(f"       EAGLE comparison is RESOLUTION-LIMITED and likely UNINFORMATIVE")
    print(f"       Conclusion: simulations need denser, galaxy-adapted radii")

print(f"\n  C2 — Window matching:")
print(f"    SPARC in [1,100] kpc: {n_per_det_in_win}/{n_orig_per_in_win} periodic detected "
      f"({100*n_per_det_in_win/n_orig_per_in_win:.1f}%)")
print(f"    R_eff-based resampling: {n_per_det_rhalf}/{n_per_rhalf} periodic detected "
      f"({100*n_per_det_rhalf/n_per_rhalf:.1f}% of periodic)")


# ================================================================
# SAVE
# ================================================================
results = {
    'test': 'eagle_injection_recovery',
    'description': 'SPARC->EAGLE sampling injection-recovery: can EAGLE 10 fixed radii detect periodic signals?',
    'parameters': {
        'eagle_radii_kpc': EAGLE_RADII.tolist(),
        'n_noise_realizations': N_NOISE_REAL,
        'min_eagle_pts': MIN_EAGLE_PTS,
        'n_surrogates_per_ls': 50,
        'perm_alpha': PERM_ALPHA,
    },
    'sample': {
        'n_total': n_total,
        'n_periodic': n_per,
        'n_nonperiodic': n_nper,
        'n_valid_periodic': len(per_results),
        'n_valid_nonperiodic': len(nper_results),
        'n_insufficient_coverage': len(insuf_results),
    },
    'c1_injection_recovery': {
        'periodic': {
            'noisefree_detection_rate': round(n_nf_det_per / len(per_results), 4),
            'mean_noisy_detection_rate': round(mean_det_per, 4),
            'median_noisy_detection_rate': round(float(np.median(det_rates_per)), 4),
            'detection_rate_iqr': [
                round(float(np.percentile(det_rates_per, 25)), 4),
                round(float(np.percentile(det_rates_per, 75)), 4),
            ],
            'false_negative_rate': round(fn_rate, 4),
        },
        'nonperiodic': {
            'noisefree_detection_rate': round(n_nf_det_nper / len(nper_results), 4) if nper_results else None,
            'mean_noisy_detection_rate': round(mean_det_nper, 4),
        },
        'wavelength_recovery': {
            'true_median_kpc': round(float(np.median(wl_true_per)), 2),
            'recovered_median_kpc': round(float(np.median(wl_rec_per)), 2) if len(wl_rec_per) > 0 else None,
            'bias_median_kpc': round(float(np.median(wl_bias_per)), 2) if len(wl_bias_per) > 0 else None,
        },
        'power_curve': power_curve,
    },
    'c2_window_matching': {
        'sparc_1_100kpc': {
            'n_galaxies': n_win,
            'n_periodic_orig': n_orig_per_in_win,
            'n_detected': n_det_in_win,
            'n_periodic_detected': n_per_det_in_win,
            'periodic_recovery_rate': round(n_per_det_in_win / n_orig_per_in_win, 4) if n_orig_per_in_win > 0 else None,
        },
        'reff_resampling': {
            'n_galaxies': n_rhalf,
            'reff_factors': R_eff_factors.tolist(),
            'n_detected': n_det_rhalf,
            'n_periodic_detected': n_per_det_rhalf,
            'periodic_recovery_rate': round(n_per_det_rhalf / n_per_rhalf, 4) if n_per_rhalf > 0 else None,
        },
    },
    'per_galaxy': [r for r in results_per_galaxy if not r.get('insufficient_coverage', False)],
    'figures': [
        'eagle_injection_recovery.png',
        'eagle_sampling_illustration.png',
    ],
}

outpath = os.path.join(RESULTS_DIR, 'summary_eagle_injection_recovery.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {outpath}")

print("\n" + "=" * 76)
print("TEST C COMPLETE")
print("=" * 76)
