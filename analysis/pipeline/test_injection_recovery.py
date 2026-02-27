#!/usr/bin/env python3
"""
Injection-Recovery Test (C1) — Can EAGLE's 10 Fixed Radii Detect SPARC Periodicity?
====================================================================================

Context: SPARC shows 37.3% periodicity at ~7 kpc vs EAGLE 2.8%. But EAGLE has only
10 fixed radii at {1, 3, 5, 10, 20, 30, 40, 50, 70, 100} kpc. This test asks:
if the SPARC signal were present in EAGLE galaxies, could EAGLE's sampling detect it?

Method:
  1. Take the 25 SPARC galaxies with significant LS periodicity (perm p < 0.05)
  2. For each, treat the full RAR residual profile as ground truth
  3. Interpolate onto EAGLE's 10 fixed radii (only within galaxy's actual R range)
  4. Noise-free: apply LS with identical significance threshold (200 surrogates)
  5. With noise: add Gaussian noise consistent with measurement errors, 100 MC trials
  6. Compute detection power and wavelength recovery bias

Key output:
  - Detection power: X/25 noise-free, mean fraction with noise
  - If < 10%: EAGLE comparison uninformative
  - If > 30%: EAGLE's 2.8% is meaningful (real absence of periodicity)
  - Recovered vs true wavelength: systematic bias?

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d, UnivariateSpline
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────
g_dagger = 1.20e-10   # m/s^2
kpc_m = 3.086e19       # m per kpc
EAGLE_RADII = np.array([1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0])
MIN_PTS_LS = 5         # Minimum resampled points for LS analysis
N_SURR = 200           # Surrogates per galaxy (same as SPARC test)
N_MC = 100             # Monte Carlo noise trials per galaxy
ALPHA = 0.05           # Significance threshold

np.random.seed(42)

print("=" * 76)
print("INJECTION-RECOVERY TEST (C1)")
print("  Can EAGLE's 10 fixed radii detect SPARC's ~7 kpc periodicity?")
print("=" * 76)
print(f"  EAGLE radii: {EAGLE_RADII} kpc")
print(f"  LS surrogates: {N_SURR}, MC noise trials: {N_MC}")
print(f"  Significance threshold: {ALPHA}")


# ══════════════════════════════════════════════════════════════════════
#  1. LOAD SPARC DATA & IDENTIFY THE 25 SIGNIFICANT GALAXIES
# ══════════════════════════════════════════════════════════════════════
print("\n[1] Loading SPARC data and spectral test results...")

# Load spectral test results to identify the 25 galaxies
spectral_path = os.path.join(RESULTS_DIR, 'summary_interface_spectral_test.json')
with open(spectral_path) as f:
    spectral_results = json.load(f)

sig_galaxy_info = {}
for g in spectral_results['per_galaxy']:
    if g['perm_p'] is not None and g['perm_p'] < ALPHA:
        sig_galaxy_info[g['name']] = {
            'true_wl_kpc': g['wl_kpc'],
            'true_power': g['power_peak'],
            'true_perm_p': g['perm_p'],
            'R_extent_kpc': g['R_extent_kpc'],
            'n_pts_orig': g['n_pts'],
            'Vflat': g['Vflat'],
        }

print(f"  Found {len(sig_galaxy_info)} galaxies with significant periodicity")


# Load SPARC rotation curve data
def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


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
                             'Vgas': [], 'Vdisk': [], 'Vbul': []}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['eVobs'].append(evobs)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in rc_data[name]:
        rc_data[name][key] = np.array(rc_data[name][key])

# Load galaxy properties
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
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        sparc_props[name] = {
            'Inc': float(parts[4]),
            'Vflat': float(parts[14]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue


# ══════════════════════════════════════════════════════════════════════
#  2. BUILD FULL-RESOLUTION RESIDUAL PROFILES FOR THE 25 GALAXIES
# ══════════════════════════════════════════════════════════════════════
print("\n[2] Computing full-resolution RAR residual profiles...")

galaxy_profiles = {}

for name in sig_galaxy_info:
    if name not in rc_data:
        print(f"  WARNING: {name} not found in rotation curve data")
        continue

    gdata = rc_data[name]
    R = gdata['R']
    Vobs = gdata['Vobs']
    eVobs = gdata['eVobs']
    Vgas = gdata['Vgas']
    Vdisk = gdata['Vdisk']
    Vbul = gdata['Vbul']

    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < 10:
        continue

    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]

    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]
    residuals = log_gobs - rar_function(log_gbar)

    # Measurement error in log space: δ(log g_obs) ≈ 2 * eVobs / (Vobs * ln(10))
    Vobs_valid = Vobs[valid][sort_idx]
    eVobs_valid = eVobs[valid][sort_idx]
    # Protect against zero velocity
    resid_err = np.where(Vobs_valid > 5,
                          2.0 * eVobs_valid / (Vobs_valid * np.log(10)),
                          0.05)  # default 0.05 dex

    galaxy_profiles[name] = {
        'R': R_sorted,
        'residuals': residuals,
        'resid_err': resid_err,
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'R_min': float(R_sorted[0]),
        'R_max': float(R_sorted[-1]),
    }

n_profiles = len(galaxy_profiles)
print(f"  Built residual profiles for {n_profiles} / {len(sig_galaxy_info)} galaxies")


# ══════════════════════════════════════════════════════════════════════
#  3. DEFINE LOMB-SCARGLE DETECTION FUNCTION (same as SPARC test)
# ══════════════════════════════════════════════════════════════════════

def run_ls_test(R, eps, n_surr=N_SURR, rng=None):
    """Run Lomb-Scargle with permutation null, return (detected, wl_kpc, power, perm_p)."""
    n = len(R)
    if n < MIN_PTS_LS:
        return False, np.nan, np.nan, 1.0

    # Spline detrending (identical to SPARC spectral test)
    var_eps = np.var(eps)
    s_param = n * var_eps * 0.5
    try:
        k = min(3, n - 1)
        if k >= 1 and n > k:
            spline = UnivariateSpline(R, eps, k=k, s=s_param)
            eps_det = eps - spline(R)
        else:
            eps_det = eps - np.mean(eps)
    except Exception:
        eps_det = eps - np.mean(eps)

    std_det = np.std(eps_det)
    if std_det < 1e-30:
        return False, np.nan, np.nan, 1.0

    y = (eps_det - np.mean(eps_det)) / std_det

    R_extent = R[-1] - R[0]
    if R_extent <= 0:
        return False, np.nan, np.nan, 1.0

    # Frequency grid (same as SPARC test)
    f_min = 1.0 / R_extent
    f_max = (n / 2.0) / R_extent
    n_freq = min(500, 10 * n)
    if f_max <= f_min:
        return False, np.nan, np.nan, 1.0

    freq_grid = np.linspace(f_min, f_max, n_freq)

    ls = LombScargle(R, y, fit_mean=False, center_data=True)
    power = ls.power(freq_grid)
    idx_peak = np.argmax(power)
    f_peak = freq_grid[idx_peak]
    power_peak = float(power[idx_peak])
    wl_kpc = 1.0 / f_peak

    # Permutation null
    if rng is None:
        rng = np.random.default_rng(789)
    null_peaks = np.zeros(n_surr)
    for s in range(n_surr):
        y_shuf = rng.permutation(y)
        ls_null = LombScargle(R, y_shuf, fit_mean=False, center_data=True)
        null_peaks[s] = np.max(ls_null.power(freq_grid))

    perm_p = float(np.mean(null_peaks >= power_peak))
    perm_p = max(perm_p, 1.0 / (n_surr + 1))

    detected = perm_p < ALPHA
    return detected, float(wl_kpc), power_peak, perm_p


# ══════════════════════════════════════════════════════════════════════
#  4. INJECTION-RECOVERY: NOISE-FREE
# ══════════════════════════════════════════════════════════════════════
print("\n[3] Noise-free injection-recovery...")

noisefree_results = []
rng_nf = np.random.default_rng(789)

for name in sorted(galaxy_profiles.keys()):
    prof = galaxy_profiles[name]
    info = sig_galaxy_info[name]

    R_full = prof['R']
    res_full = prof['residuals']
    R_min, R_max = prof['R_min'], prof['R_max']

    # Select EAGLE radii within galaxy's data range (no extrapolation)
    eagle_in_range = EAGLE_RADII[(EAGLE_RADII >= R_min) & (EAGLE_RADII <= R_max)]

    if len(eagle_in_range) < MIN_PTS_LS:
        noisefree_results.append({
            'name': name,
            'n_eagle_radii': len(eagle_in_range),
            'true_wl': info['true_wl_kpc'],
            'detected': False,
            'reason': f'too_few_radii ({len(eagle_in_range)} < {MIN_PTS_LS})',
            'recovered_wl': np.nan,
            'perm_p': np.nan,
        })
        continue

    # Interpolate residuals onto EAGLE radii
    interp_func = interp1d(R_full, res_full, kind='linear', fill_value='extrapolate')
    res_eagle = interp_func(eagle_in_range)

    # Run LS test
    detected, wl_rec, power, perm_p = run_ls_test(eagle_in_range, res_eagle, rng=rng_nf)

    noisefree_results.append({
        'name': name,
        'n_eagle_radii': len(eagle_in_range),
        'R_range': [float(eagle_in_range[0]), float(eagle_in_range[-1])],
        'true_wl': info['true_wl_kpc'],
        'detected': detected,
        'recovered_wl': wl_rec,
        'power_peak': power,
        'perm_p': perm_p,
        'R_min_gal': R_min,
        'R_max_gal': R_max,
        'n_pts_orig': info['n_pts_orig'],
    })

    status = "DETECTED" if detected else "missed"
    print(f"  {name:<14} N_eagle={len(eagle_in_range):2d}  "
          f"λ_true={info['true_wl_kpc']:6.2f}  λ_rec={wl_rec:6.2f}  "
          f"p={perm_p:.3f}  [{status}]")

n_detected_nf = sum(1 for r in noisefree_results if r['detected'])
n_testable = sum(1 for r in noisefree_results if r.get('reason') is None
                  or 'too_few' not in str(r.get('reason', '')))
n_too_few = sum(1 for r in noisefree_results if 'too_few' in str(r.get('reason', '')))

print(f"\n  Noise-free detection: {n_detected_nf} / {len(noisefree_results)} "
      f"({100*n_detected_nf/len(noisefree_results):.1f}%)")
print(f"  Testable (≥{MIN_PTS_LS} EAGLE radii in range): {n_testable}")
print(f"  Too few radii: {n_too_few}")
if n_testable > 0:
    n_det_testable = sum(1 for r in noisefree_results
                          if r['detected'] and ('too_few' not in str(r.get('reason', ''))))
    print(f"  Detection among testable: {n_det_testable}/{n_testable} "
          f"({100*n_det_testable/n_testable:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
#  5. INJECTION-RECOVERY: WITH NOISE (100 MC trials per galaxy)
# ══════════════════════════════════════════════════════════════════════
print(f"\n[4] Noisy injection-recovery ({N_MC} MC trials per galaxy)...")

noise_rng = np.random.default_rng(12345)
noisy_results = []

for gi, name in enumerate(sorted(galaxy_profiles.keys())):
    prof = galaxy_profiles[name]
    info = sig_galaxy_info[name]

    R_full = prof['R']
    res_full = prof['residuals']
    err_full = prof['resid_err']
    R_min, R_max = prof['R_min'], prof['R_max']

    eagle_in_range = EAGLE_RADII[(EAGLE_RADII >= R_min) & (EAGLE_RADII <= R_max)]
    if len(eagle_in_range) < MIN_PTS_LS:
        noisy_results.append({
            'name': name,
            'n_eagle_radii': len(eagle_in_range),
            'detection_rate': 0.0,
            'reason': 'too_few_radii',
        })
        continue

    # Interpolate both residuals and errors onto EAGLE radii
    interp_res = interp1d(R_full, res_full, kind='linear', fill_value='extrapolate')
    interp_err = interp1d(R_full, err_full, kind='linear', fill_value='extrapolate')

    res_eagle_true = interp_res(eagle_in_range)
    err_eagle = interp_err(eagle_in_range)
    # Floor on errors
    err_eagle = np.maximum(err_eagle, 0.01)

    n_detected = 0
    recovered_wls = []

    for trial in range(N_MC):
        # Add Gaussian noise
        noise = noise_rng.normal(0, err_eagle)
        res_noisy = res_eagle_true + noise

        rng_trial = np.random.default_rng(trial * 1000 + gi)
        detected, wl_rec, _, perm_p = run_ls_test(eagle_in_range, res_noisy,
                                                     rng=rng_trial)
        if detected:
            n_detected += 1
            recovered_wls.append(wl_rec)

    detection_rate = n_detected / N_MC

    noisy_results.append({
        'name': name,
        'n_eagle_radii': len(eagle_in_range),
        'true_wl': info['true_wl_kpc'],
        'detection_rate': detection_rate,
        'n_detected': n_detected,
        'n_trials': N_MC,
        'median_recovered_wl': float(np.median(recovered_wls)) if recovered_wls else np.nan,
    })

    print(f"  {name:<14} N_eagle={len(eagle_in_range):2d}  "
          f"detection={n_detected}/{N_MC} ({detection_rate:.0%})  "
          f"λ_true={info['true_wl_kpc']:.2f}")

# Aggregate noisy results
testable_noisy = [r for r in noisy_results if r.get('reason') != 'too_few_radii']
mean_detection = np.mean([r['detection_rate'] for r in testable_noisy]) if testable_noisy else 0
median_detection = np.median([r['detection_rate'] for r in testable_noisy]) if testable_noisy else 0

print(f"\n  Noisy detection (testable galaxies):")
print(f"    Mean detection rate: {mean_detection:.1%}")
print(f"    Median detection rate: {median_detection:.1%}")
print(f"    Galaxies with rate > 50%: "
      f"{sum(1 for r in testable_noisy if r['detection_rate'] > 0.5)}/{len(testable_noisy)}")
print(f"    Galaxies with rate > 10%: "
      f"{sum(1 for r in testable_noisy if r['detection_rate'] > 0.1)}/{len(testable_noisy)}")


# ══════════════════════════════════════════════════════════════════════
#  6. WAVELENGTH RECOVERY ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n[5] Wavelength recovery analysis...")

# Noise-free: true vs recovered wavelength
true_wls = []
rec_wls = []
for r in noisefree_results:
    if r['detected'] and not np.isnan(r['recovered_wl']):
        true_wls.append(r['true_wl'])
        rec_wls.append(r['recovered_wl'])

true_wls = np.array(true_wls)
rec_wls = np.array(rec_wls)

if len(true_wls) >= 3:
    log_ratio = np.log10(rec_wls / true_wls)
    mean_log_ratio = float(np.mean(log_ratio))
    std_log_ratio = float(np.std(log_ratio, ddof=1))
    rho, p_rho = stats.spearmanr(true_wls, rec_wls)
    print(f"  Noise-free wavelength recovery (N={len(true_wls)}):")
    print(f"    Mean log10(λ_rec/λ_true): {mean_log_ratio:+.3f} "
          f"(systematic bias: {'YES' if abs(mean_log_ratio) > 0.1 else 'NO'})")
    print(f"    Std:  {std_log_ratio:.3f}")
    print(f"    Spearman ρ(true, rec): {rho:.3f} (p={p_rho:.4f})")
else:
    mean_log_ratio = np.nan
    std_log_ratio = np.nan
    rho = np.nan
    p_rho = np.nan
    print(f"  Too few detections for wavelength recovery analysis")


# ══════════════════════════════════════════════════════════════════════
#  7. NYQUIST ANALYSIS — What wavelengths CAN EAGLE detect?
# ══════════════════════════════════════════════════════════════════════
print("\n[6] Nyquist / resolution analysis for EAGLE radii...")

eagle_spacings = np.diff(EAGLE_RADII)
print(f"  EAGLE radii spacings: {eagle_spacings} kpc")
print(f"  Minimum spacing: {eagle_spacings.min():.0f} kpc (between {EAGLE_RADII[0]}-{EAGLE_RADII[1]} kpc)")
print(f"  Nyquist wavelength (2 × min spacing): {2*eagle_spacings.min():.0f} kpc")
print(f"  Practical minimum λ (~3× median spacing): {3*np.median(eagle_spacings):.0f} kpc")

# How many of the 25 galaxies have true_wl BELOW the Nyquist limit?
nyquist_limit = 2 * eagle_spacings.min()  # 4 kpc
practical_limit = 3 * np.median(eagle_spacings)

n_below_nyquist = sum(1 for g in sig_galaxy_info.values()
                       if g['true_wl_kpc'] < nyquist_limit)
n_below_practical = sum(1 for g in sig_galaxy_info.values()
                          if g['true_wl_kpc'] < practical_limit)

print(f"\n  SPARC significant galaxies:")
print(f"    True λ < {nyquist_limit:.0f} kpc (Nyquist): {n_below_nyquist}/25 "
      f"({100*n_below_nyquist/25:.0f}%) — fundamentally undetectable")
print(f"    True λ < {practical_limit:.0f} kpc (practical): {n_below_practical}/25 "
      f"({100*n_below_practical/25:.0f}%) — very difficult to detect")

# Distribution of true wavelengths
true_wl_all = np.array([g['true_wl_kpc'] for g in sig_galaxy_info.values()])
print(f"    True λ distribution: median={np.median(true_wl_all):.2f}, "
      f"IQR={np.percentile(true_wl_all, 25):.2f}–{np.percentile(true_wl_all, 75):.2f} kpc")


# ══════════════════════════════════════════════════════════════════════
#  8. VERDICT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)

nf_power = n_detected_nf / len(noisefree_results)
noisy_power = mean_detection

print(f"\n  DETECTION POWER:")
print(f"    Noise-free: {n_detected_nf}/{len(noisefree_results)} ({nf_power:.1%})")
print(f"    With noise (mean):   {noisy_power:.1%}")
print(f"    With noise (median): {median_detection:.1%}")

if nf_power < 0.10:
    verdict = "UNINFORMATIVE"
    verdict_text = ("EAGLE's 10 fixed radii have <10% detection power. "
                    "The 2.8% periodicity rate in EAGLE is UNINFORMATIVE — "
                    "the sampling cannot detect ~7 kpc signals even when present.")
elif nf_power < 0.30:
    verdict = "LOW_POWER"
    verdict_text = ("EAGLE's detection power is 10-30%. The 2.8% EAGLE rate is "
                    "marginally informative but underpowered. Finer radial sampling needed.")
elif nf_power >= 0.30:
    verdict = "INFORMATIVE"
    verdict_text = ("EAGLE's detection power is ≥30%. The 2.8% periodicity rate "
                    "is meaningful — EAGLE CAN detect these signals but DOESN'T find them. "
                    "This supports SPARC periodicity as potentially physical.")
else:
    verdict = "ERROR"
    verdict_text = "Computation error."

print(f"\n  VERDICT: {verdict}")
print(f"  {verdict_text}")

# Breakdown by wavelength regime
if len(noisefree_results) > 0:
    short_wl = [r for r in noisefree_results if r['true_wl'] < 5]
    medium_wl = [r for r in noisefree_results if 5 <= r['true_wl'] < 15]
    long_wl = [r for r in noisefree_results if r['true_wl'] >= 15]

    print(f"\n  Detection by wavelength regime (noise-free):")
    for label, subset in [("λ < 5 kpc", short_wl),
                           ("5 ≤ λ < 15", medium_wl),
                           ("λ ≥ 15 kpc", long_wl)]:
        n_s = len(subset)
        n_d = sum(1 for r in subset if r['detected'])
        if n_s > 0:
            print(f"    {label}: {n_d}/{n_s} ({100*n_d/n_s:.0f}%)")


# ══════════════════════════════════════════════════════════════════════
#  9. PUBLICATION FIGURE
# ══════════════════════════════════════════════════════════════════════
print("\n[7] Generating publication figure...")

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# ── Panel A: Noise-free detection per galaxy ──────────────────────
ax = axes[0, 0]
names_sorted = sorted(galaxy_profiles.keys())
colors = []
true_wl_plot = []
detected_plot = []

for r in noisefree_results:
    true_wl_plot.append(r['true_wl'])
    if 'too_few' in str(r.get('reason', '')):
        colors.append('#999999')
        detected_plot.append(False)
    elif r['detected']:
        colors.append('#4CAF50')
        detected_plot.append(True)
    else:
        colors.append('#F44336')
        detected_plot.append(False)

y_pos = np.arange(len(noisefree_results))
ax.barh(y_pos, true_wl_plot, color=colors, alpha=0.7, height=0.8, edgecolor='white')
ax.axvline(nyquist_limit, color='red', ls='--', lw=1.2, label=f'Nyquist = {nyquist_limit:.0f} kpc')
ax.axvline(practical_limit, color='orange', ls=':', lw=1.2,
           label=f'Practical limit = {practical_limit:.0f} kpc')
ax.set_yticks(y_pos)
ax.set_yticklabels([r['name'] for r in noisefree_results], fontsize=7)
ax.set_xlabel('True peak wavelength (kpc)')
ax.set_title(f'(a) Noise-Free Detection ({n_detected_nf}/{len(noisefree_results)})')
ax.legend(fontsize=7, loc='lower right')

# Custom legend for detection colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', alpha=0.7, label='Detected'),
    Patch(facecolor='#F44336', alpha=0.7, label='Not detected'),
    Patch(facecolor='#999999', alpha=0.7, label='Too few radii'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

# ── Panel B: Noisy detection rate per galaxy ──────────────────────
ax = axes[0, 1]
testable_names = []
testable_rates = []
testable_wls = []
for r in noisy_results:
    if r.get('reason') != 'too_few_radii':
        testable_names.append(r['name'])
        testable_rates.append(r['detection_rate'])
        testable_wls.append(r['true_wl'])

if testable_names:
    # Sort by detection rate
    sort_idx = np.argsort(testable_rates)[::-1]
    sorted_names = [testable_names[i] for i in sort_idx]
    sorted_rates = [testable_rates[i] for i in sort_idx]
    sorted_wls = [testable_wls[i] for i in sort_idx]

    y_pos2 = np.arange(len(sorted_names))
    bar_colors = ['#4CAF50' if r > 0.5 else '#FF9800' if r > 0.1 else '#F44336'
                  for r in sorted_rates]
    ax.barh(y_pos2, [r * 100 for r in sorted_rates], color=bar_colors,
            alpha=0.7, height=0.8, edgecolor='white')
    ax.axvline(50, color='green', ls='--', lw=0.8, alpha=0.5)
    ax.axvline(10, color='orange', ls='--', lw=0.8, alpha=0.5)
    ax.set_yticks(y_pos2)
    ax.set_yticklabels([f"{n} (λ={w:.1f})" for n, w in zip(sorted_names, sorted_wls)],
                        fontsize=6.5)
    ax.set_xlabel('Detection rate with noise (%)')
    ax.set_title(f'(b) Noisy Detection Rate ({N_MC} MC trials)')
    ax.set_xlim(0, 105)

# ── Panel C: True vs recovered wavelength ─────────────────────────
ax = axes[1, 0]
if len(true_wls) >= 2:
    ax.scatter(true_wls, rec_wls, c='#2196F3', s=60, alpha=0.7, edgecolors='white',
               linewidth=0.5, zorder=3)
    wl_range = [min(true_wls.min(), rec_wls.min()) * 0.5,
                max(true_wls.max(), rec_wls.max()) * 2]
    ax.plot(wl_range, wl_range, 'k--', lw=1, alpha=0.5, label='1:1 line')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('True peak wavelength (kpc)')
    ax.set_ylabel('Recovered peak wavelength (kpc)')
    ax.set_title(f'(c) Wavelength Recovery (noise-free, N={len(true_wls)})')
    if not np.isnan(rho):
        ax.text(0.05, 0.92, f'ρ = {rho:.2f}\nbias = {mean_log_ratio:+.2f} dex',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, 'Too few detections\nfor recovery analysis',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_title('(c) Wavelength Recovery')

# ── Panel D: Summary statistics ──────────────────────────────────
ax = axes[1, 1]
ax.axis('off')

summary_text = (
    f"INJECTION-RECOVERY SUMMARY\n"
    f"{'─' * 40}\n\n"
    f"25 SPARC galaxies with significant\n"
    f"Lomb-Scargle periodicity (p < 0.05)\n\n"
    f"EAGLE sampling: 10 radii at\n"
    f"{{1,3,5,10,20,30,40,50,70,100}} kpc\n\n"
    f"{'─' * 40}\n"
    f"Noise-free detection:  {n_detected_nf}/{len(noisefree_results)} "
    f"({nf_power:.0%})\n"
    f"Noisy detection (mean): {noisy_power:.0%}\n"
    f"Noisy detection (med):  {median_detection:.0%}\n"
    f"{'─' * 40}\n\n"
    f"Nyquist limit: {nyquist_limit:.0f} kpc\n"
    f"Galaxies with λ < Nyquist: {n_below_nyquist}/25\n"
    f"SPARC median λ: {np.median(true_wl_all):.1f} kpc\n\n"
    f"{'─' * 40}\n"
    f"VERDICT: {verdict}\n"
)

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9.5,
        va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

fig.suptitle('Injection-Recovery Test: Can EAGLE Sampling Detect SPARC Periodicity?',
             fontsize=14, y=1.01)
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'injection_recovery_power.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fig_path}")


# ══════════════════════════════════════════════════════════════════════
#  10. SAVE JSON RESULTS
# ══════════════════════════════════════════════════════════════════════
results = {
    'test': 'injection_recovery_C1',
    'description': ('Injection-recovery test: can EAGLE 10 fixed radii detect the '
                    'Lomb-Scargle periodicity observed in SPARC? Tests 25 galaxies '
                    'with significant periodicity by resampling onto EAGLE radii.'),
    'parameters': {
        'eagle_radii_kpc': EAGLE_RADII.tolist(),
        'n_surrogates': N_SURR,
        'n_mc_noise_trials': N_MC,
        'significance_threshold': ALPHA,
        'min_pts_ls': MIN_PTS_LS,
    },
    'sample': {
        'n_significant_sparc': len(sig_galaxy_info),
        'n_with_profiles': n_profiles,
        'n_testable': n_testable,
        'n_too_few_radii': n_too_few,
    },
    'noise_free': {
        'n_detected': n_detected_nf,
        'n_total': len(noisefree_results),
        'detection_power': round(nf_power, 4),
        'per_galaxy': [{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in r.items()}
                        for r in noisefree_results],
    },
    'noisy': {
        'mean_detection_rate': round(mean_detection, 4),
        'median_detection_rate': round(median_detection, 4),
        'n_mc_trials': N_MC,
        'per_galaxy': [{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in r.items() if k != 'reason' or v is not None}
                        for r in noisy_results],
    },
    'wavelength_recovery': {
        'n_detected': len(true_wls) if len(true_wls) > 0 else 0,
        'mean_log_ratio': round(mean_log_ratio, 4) if not np.isnan(mean_log_ratio) else None,
        'std_log_ratio': round(std_log_ratio, 4) if not np.isnan(std_log_ratio) else None,
        'spearman_rho': round(rho, 4) if not np.isnan(rho) else None,
        'spearman_p': round(p_rho, 4) if not np.isnan(p_rho) else None,
    },
    'nyquist_analysis': {
        'nyquist_limit_kpc': nyquist_limit,
        'practical_limit_kpc': practical_limit,
        'n_below_nyquist': n_below_nyquist,
        'n_below_practical': n_below_practical,
        'sparc_wl_median': round(float(np.median(true_wl_all)), 2),
        'sparc_wl_q25': round(float(np.percentile(true_wl_all, 25)), 2),
        'sparc_wl_q75': round(float(np.percentile(true_wl_all, 75)), 2),
    },
    'verdict': verdict,
    'verdict_text': verdict_text,
    'figures': {
        'injection_recovery': fig_path,
    },
}

outpath = os.path.join(RESULTS_DIR, 'summary_injection_recovery.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {outpath}")
print("=" * 76)
print("Done.")
