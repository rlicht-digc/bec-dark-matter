#!/usr/bin/env python3
"""
Simulation Coherence Test — EAGLE vs SPARC Radial Autocorrelation & Periodicity
================================================================================

Critical discriminator: do ΛCDM simulations produce the same radial coherence
in RAR residuals as observed in SPARC?

SPARC reference (67 galaxies, ≥15 points each):
  - Lag-1 ACF (demeaned): 0.6997 ± 0.0226
  - Fraction with significant LS periodicity (p<0.05): 37.3%
  - Peak wavelength: median 6.23 kpc (IQR 2.61–10.86)

EAGLE data: 10 fixed aperture radii per galaxy (1, 3, 5, 10, 20, 30, 40, 50, 70, 100 kpc)
  - ~29,700 galaxies from RefL0100N1504, snapshot 28 (z≈0)
  - M_star, M_gas, M_dm at each aperture

TNG status: only 4 fixed aperture radii (5, 10, 30, 100 kpc) — insufficient for
  coherence analysis. Extraction requirements documented at end.

Method:
  1. Compute g_bar = G*(M_star+M_gas)/r², g_obs = G*(M_star+M_gas+M_dm)/r² at each aperture
  2. RAR residuals: eps = log10(g_obs) - log10(g_obs_predicted) using same g† = 1.2e-10
  3. Per-galaxy: lag-1 ACF (demeaned), Lomb-Scargle periodogram with permutation null
  4. Compare EAGLE distribution to SPARC

Quality cuts (EAGLE):
  - Total baryonic mass M_bar > 10^9 M_sun at 30 kpc (avoids dwarfs with resolution artifacts)
  - All 10 aperture radii must have M_bar > 0 and M_total > M_bar
  - Minimum 8 valid radii after filtering

SPARC matched-resolution control:
  - Also compute SPARC ACF subsampled to ~10 points per galaxy
  - This controls for the effect of radial resolution on ACF estimates

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import LombScargle

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SPARC_DIR = os.path.join(DATA_DIR, 'sparc')
EAGLE_CACHE = os.path.join(DATA_DIR, 'eagle_rar', 'eagle_aperture_masses.json')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────
G = 6.674e-11       # m^3 kg^-1 s^-2
M_sun = 1.989e30    # kg
kpc_m = 3.086e19    # m per kpc
g_dagger = 1.20e-10 # m/s^2
LOG_GD = np.log10(g_dagger)  # -9.921

MIN_PTS_EAGLE = 8   # Min valid radii per EAGLE galaxy (max possible = 10)
MIN_PTS_SPARC = 15  # Standard SPARC quality cut
MIN_MBAR_MSUN = 1e9 # Baryonic mass cut at 30 kpc
N_SURR = 200        # Surrogates per galaxy for Lomb-Scargle
N_BOOT = 10000      # Bootstrap resamples
N_PERM = 2000       # Permutations for within-galaxy null

np.random.seed(42)
t0 = time.time()

def elapsed():
    return f"[{time.time()-t0:.0f}s]"

print("=" * 76)
print("SIMULATION COHERENCE TEST — EAGLE vs SPARC")
print("  Radial Autocorrelation & Lomb-Scargle Periodicity")
print("=" * 76)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_GD:.3f}")
print(f"  EAGLE min points: {MIN_PTS_EAGLE}, SPARC min points: {MIN_PTS_SPARC}")
print(f"  Baryonic mass cut: M_bar > {MIN_MBAR_MSUN:.0e} M_sun at 30 kpc")
print(f"  LS surrogates/galaxy: {N_SURR}, Bootstrap: {N_BOOT}")


# ══════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS (identical to SPARC pipeline)
# ══════════════════════════════════════════════════════════════════════

def rar_function(log_gbar, a0=1.2e-10):
    """Standard RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def lag_autocorrelation(x, lag=1, center=True):
    """Lag-k autocorrelation. center=True gives standard (demeaned) ACF."""
    n = len(x)
    if n <= lag + 1:
        return np.nan
    if center:
        x_use = x - np.mean(x)
    else:
        x_use = x.copy()
    var = np.mean(x_use**2)
    if var < 1e-30:
        return np.nan
    cov = np.mean(x_use[:n - lag] * x_use[lag:])
    return cov / var


def compute_stats(values, label='', rng_seed=42):
    """Aggregate stats: mean, SE, bootstrap CI, t-test, binomial."""
    n = len(values)
    if n < 3:
        return None
    mean_val = float(np.mean(values))
    se_val = float(np.std(values, ddof=1) / np.sqrt(n))
    median_val = float(np.median(values))
    std_val = float(np.std(values, ddof=1))
    t_stat, p_two = stats.ttest_1samp(values, 0.0)
    p_one = float(p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0)
    boot_rng = np.random.default_rng(rng_seed)
    boot_means = np.array([np.mean(boot_rng.choice(values, size=n)) for _ in range(N_BOOT)])
    ci = np.percentile(boot_means, [2.5, 97.5])
    n_pos = int(np.sum(values > 0))
    frac_pos = float(n_pos / n)
    binom_p = float(stats.binomtest(n_pos, n, 0.5, alternative='greater').pvalue)
    return {
        'mean': round(mean_val, 4), 'se': round(se_val, 4),
        'median': round(median_val, 4), 'std': round(std_val, 4),
        'ci_95': [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        'ci_excludes_zero': bool(ci[0] > 0),
        't_stat': round(float(t_stat), 3),
        'p_one_sided': float(p_one),
        'frac_positive': round(frac_pos, 3),
        'n_positive': n_pos, 'n_total': n,
        'binom_p': float(binom_p),
    }


def two_sample_comparison(arr_a, arr_b, label_a='EAGLE', label_b='SPARC'):
    """Welch t-test + Mann-Whitney for two distributions."""
    if len(arr_a) < 5 or len(arr_b) < 5:
        return None
    t_val, p_welch = stats.ttest_ind(arr_a, arr_b, equal_var=False)
    _, p_mwu = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
    # Kolmogorov-Smirnov
    ks_stat, p_ks = stats.ks_2samp(arr_a, arr_b)
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(arr_a, ddof=1) + np.var(arr_b, ddof=1)) / 2)
    cohens_d = (np.mean(arr_a) - np.mean(arr_b)) / pooled_std if pooled_std > 0 else 0
    return {
        f'{label_a}_mean': round(float(np.mean(arr_a)), 4),
        f'{label_a}_se': round(float(np.std(arr_a, ddof=1) / np.sqrt(len(arr_a))), 4),
        f'{label_a}_median': round(float(np.median(arr_a)), 4),
        f'{label_a}_N': len(arr_a),
        f'{label_b}_mean': round(float(np.mean(arr_b)), 4),
        f'{label_b}_se': round(float(np.std(arr_b, ddof=1) / np.sqrt(len(arr_b))), 4),
        f'{label_b}_median': round(float(np.median(arr_b)), 4),
        f'{label_b}_N': len(arr_b),
        'welch_t': round(float(t_val), 3),
        'welch_p': float(p_welch),
        'mwu_p': float(p_mwu),
        'ks_stat': round(float(ks_stat), 4),
        'ks_p': float(p_ks),
        'cohens_d': round(float(cohens_d), 3),
    }


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1: LOAD EAGLE DATA & COMPUTE RAR RESIDUALS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 1] Loading EAGLE aperture masses...")

with open(EAGLE_CACHE) as f:
    cached = json.load(f)
ap_sizes = cached['aperture_sizes']   # [1, 3, 5, 10, 20, 30, 40, 50, 70, 100]
gal_data = cached['galaxy_data']
all_gids = sorted(gal_data.keys())
print(f"  {len(all_gids)} galaxies, apertures: {ap_sizes} kpc")

print(f"{elapsed()} Computing per-galaxy RAR residuals...")

eagle_galaxies = []
n_mass_cut = 0
n_npts_cut = 0
n_negative_resid = 0

for gid in all_gids:
    aps = gal_data[gid]['apertures']

    # Mass cut: M_bar at 30 kpc
    if '30.0' in aps:
        mbar_30 = aps['30.0']['m_star'] + aps['30.0']['m_gas']
    elif '30' in aps:
        mbar_30 = aps['30']['m_star'] + aps['30']['m_gas']
    else:
        n_mass_cut += 1
        continue
    if mbar_30 < MIN_MBAR_MSUN / M_sun * M_sun:  # compare in same units
        # aps stores masses in M_sun already
        if mbar_30 < MIN_MBAR_MSUN:
            n_mass_cut += 1
            continue

    radii, log_gb, log_go = [], [], []
    for ap_str in sorted(aps, key=lambda x: float(x)):
        r_kpc = float(ap_str)
        if r_kpc < 1.0:
            continue
        ms = aps[ap_str]['m_star']
        mg = aps[ap_str]['m_gas']
        md = aps[ap_str]['m_dm']
        mb = ms + mg
        mt = mb + md
        if mb <= 0 or mt <= 0 or mt <= mb:
            continue
        r_m = r_kpc * kpc_m
        gb = G * mb * M_sun / r_m**2
        go = G * mt * M_sun / r_m**2
        if gb <= 1e-15 or go <= 1e-15:
            continue
        radii.append(r_kpc)
        log_gb.append(np.log10(gb))
        log_go.append(np.log10(go))

    if len(radii) < MIN_PTS_EAGLE:
        n_npts_cut += 1
        continue

    radii = np.array(radii)
    log_gb = np.array(log_gb)
    log_go = np.array(log_go)

    # RAR residuals ordered by radius (already sorted by aperture size)
    log_go_pred = rar_function(log_gb)
    residuals = log_go - log_go_pred

    # Demeaned residuals
    residuals_dm = residuals - np.mean(residuals)

    # Lag-1 ACF (demeaned — the controlled metric)
    r1_dm = lag_autocorrelation(residuals_dm, lag=1, center=True)
    r1_raw = lag_autocorrelation(residuals, lag=1, center=False)
    r2_dm = lag_autocorrelation(residuals_dm, lag=2, center=True)

    if np.isnan(r1_dm):
        continue

    eagle_galaxies.append({
        'gid': gid,
        'n_pts': len(radii),
        'r1_dm': float(r1_dm),
        'r1_raw': float(r1_raw),
        'r2_dm': float(r2_dm),
        'radii': radii,
        'residuals': residuals,
        'residuals_dm': residuals_dm,
        'mean_resid': float(np.mean(residuals)),
        'std_resid': float(np.std(residuals)),
        'log_mbar_30': float(np.log10(mbar_30)) if mbar_30 > 0 else None,
    })

n_eagle = len(eagle_galaxies)
print(f"  Galaxies passing cuts: {n_eagle}")
print(f"  Rejected — mass cut: {n_mass_cut}")
print(f"  Rejected — too few points: {n_npts_cut}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2: EAGLE LOMB-SCARGLE PERIODOGRAMS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 2] Computing EAGLE Lomb-Scargle periodograms...")

perm_rng = np.random.default_rng(789)

for gi, g in enumerate(eagle_galaxies):
    R = g['radii']
    eps = g['residuals_dm']
    n = g['n_pts']

    std_eps = np.std(eps)
    if std_eps < 1e-30:
        g['ls_valid'] = False
        continue

    # Spline detrending (same as SPARC spectral test)
    var_eps = np.var(eps)
    s_param = n * var_eps * 0.5
    try:
        k_order = min(3, n - 1)
        if k_order >= 1:
            spline = UnivariateSpline(R, eps, k=k_order, s=s_param)
            eps_det = eps - spline(R)
        else:
            eps_det = eps - np.mean(eps)
    except Exception:
        eps_det = eps - np.mean(eps)

    # Standardize
    std_det = np.std(eps_det)
    if std_det < 1e-30:
        g['ls_valid'] = False
        continue
    y = (eps_det - np.mean(eps_det)) / std_det

    R_extent = R[-1] - R[0]
    if R_extent <= 0:
        g['ls_valid'] = False
        continue

    # Frequency grid
    f_min = 1.0 / R_extent
    f_max = (n / 2.0) / R_extent
    n_freq = min(500, 10 * n)
    freq_grid = np.linspace(f_min, f_max, n_freq)

    # Observed LS
    ls = LombScargle(R, y, fit_mean=False, center_data=True)
    power = ls.power(freq_grid)
    idx_peak = np.argmax(power)
    f_peak = float(freq_grid[idx_peak])
    power_peak = float(power[idx_peak])
    wl_kpc = 1.0 / f_peak

    # Permutation null
    null_peaks = np.zeros(N_SURR)
    for s in range(N_SURR):
        y_shuf = perm_rng.permutation(y)
        ls_null = LombScargle(R, y_shuf, fit_mean=False, center_data=True)
        null_peaks[s] = np.max(ls_null.power(freq_grid))

    p_val = float(np.mean(null_peaks >= power_peak))
    p_val = max(p_val, 1.0 / (N_SURR + 1))

    g['ls_valid'] = True
    g['f_peak'] = f_peak
    g['wl_kpc'] = wl_kpc
    g['power_peak'] = power_peak
    g['null_mean_peak'] = float(np.mean(null_peaks))
    g['perm_p'] = p_val
    g['power_spectrum'] = power.tolist()
    g['freq_grid'] = freq_grid.tolist()
    g['R_extent'] = float(R_extent)

    if (gi + 1) % 500 == 0:
        print(f"    {gi+1}/{n_eagle} done... {elapsed()}")

print(f"  {elapsed()} Completed all {n_eagle} galaxies")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3: EAGLE PERMUTATION NULL FOR ACF (within-galaxy)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 3] EAGLE within-galaxy permutation null for ACF...")

perm_rng2 = np.random.default_rng(456)
eagle_perm_results = []

for gi, g in enumerate(eagle_galaxies):
    eps = g['residuals_dm']
    obs_acf = g['r1_dm']
    null_acfs = np.zeros(N_PERM)
    for p in range(N_PERM):
        null_acfs[p] = lag_autocorrelation(perm_rng2.permutation(eps), lag=1, center=True)
    valid_null = null_acfs[~np.isnan(null_acfs)]
    if len(valid_null) > 10:
        perm_p = float(np.mean(valid_null >= obs_acf))
        perm_p = max(perm_p, 1.0 / (len(valid_null) + 1))
    else:
        perm_p = np.nan
    g['acf_perm_p'] = perm_p
    eagle_perm_results.append(perm_p)

    if (gi + 1) % 1000 == 0:
        print(f"    {gi+1}/{n_eagle} done... {elapsed()}")

eagle_perm_arr = np.array([p for p in eagle_perm_results if not np.isnan(p)])
n_acf_sig = int(np.sum(eagle_perm_arr < 0.05))
print(f"  Galaxies with ACF significantly > 0 (perm p<0.05): {n_acf_sig}/{len(eagle_perm_arr)}")

# Fisher combined p-value
fisher_pvals = eagle_perm_arr[eagle_perm_arr > 0]
if len(fisher_pvals) > 10:
    log_pvals = np.log(np.clip(fisher_pvals, 1e-300, 1.0))
    fisher_stat = -2.0 * np.sum(log_pvals)
    fisher_df = 2 * len(log_pvals)
    fisher_p = float(stats.chi2.sf(fisher_stat, fisher_df))
    print(f"  Fisher combined p-value: {fisher_p:.4e} (df={fisher_df})")
else:
    fisher_p = np.nan


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4: LOAD SPARC REFERENCE (recompute for consistency)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 4] Loading SPARC data for reference comparison...")

table2_path = os.path.join(SPARC_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(SPARC_DIR, 'SPARC_Lelli2016c.mrt')

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
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in rc_data:
            rc_data[name] = {'R': [], 'Vobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in rc_data[name]:
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
            'Inc': float(parts[4]),
            'Vflat': float(parts[14]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

# Compute SPARC per-galaxy ACF and LS (full resolution)
sparc_galaxies = []

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
    if np.sum(valid) < MIN_PTS_SPARC:
        continue

    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]
    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]
    residuals = log_gobs - rar_function(log_gbar)
    residuals_dm = residuals - np.mean(residuals)

    r1_dm = lag_autocorrelation(residuals_dm, lag=1, center=True)
    r1_raw = lag_autocorrelation(residuals, lag=1, center=False)
    r2_dm = lag_autocorrelation(residuals_dm, lag=2, center=True)

    if np.isnan(r1_dm):
        continue

    # Lomb-Scargle (detrended)
    n = len(R_sorted)
    var_eps = np.var(residuals_dm)
    s_param = n * var_eps * 0.5
    try:
        spline = UnivariateSpline(R_sorted, residuals_dm, k=min(3, n-1), s=s_param)
        eps_det = residuals_dm - spline(R_sorted)
    except Exception:
        eps_det = residuals_dm - np.mean(residuals_dm)

    std_det = np.std(eps_det)
    ls_valid = False
    wl_kpc = np.nan
    perm_p_ls = np.nan
    power_peak = np.nan

    if std_det > 1e-30:
        y = (eps_det - np.mean(eps_det)) / std_det
        R_extent = R_sorted[-1] - R_sorted[0]
        if R_extent > 0:
            f_min = 1.0 / R_extent
            f_max = (n / 2.0) / R_extent
            n_freq = min(500, 10 * n)
            freq_grid = np.linspace(f_min, f_max, n_freq)
            ls_obj = LombScargle(R_sorted, y, fit_mean=False, center_data=True)
            power = ls_obj.power(freq_grid)
            idx_peak = np.argmax(power)
            power_peak = float(power[idx_peak])
            wl_kpc = 1.0 / freq_grid[idx_peak]

            null_peaks = np.zeros(N_SURR)
            for s in range(N_SURR):
                y_shuf = perm_rng.permutation(y)
                null_peaks[s] = np.max(LombScargle(R_sorted, y_shuf, fit_mean=False,
                                                     center_data=True).power(freq_grid))
            perm_p_ls = float(np.mean(null_peaks >= power_peak))
            perm_p_ls = max(perm_p_ls, 1.0 / (N_SURR + 1))
            ls_valid = True

    sparc_galaxies.append({
        'name': name,
        'n_pts': n,
        'r1_dm': float(r1_dm),
        'r1_raw': float(r1_raw),
        'r2_dm': float(r2_dm),
        'ls_valid': ls_valid,
        'wl_kpc': float(wl_kpc) if not np.isnan(wl_kpc) else None,
        'perm_p': float(perm_p_ls) if not np.isnan(perm_p_ls) else None,
        'power_peak': float(power_peak) if not np.isnan(power_peak) else None,
        'Vflat': prop['Vflat'],
    })

n_sparc = len(sparc_galaxies)
print(f"  SPARC galaxies (≥{MIN_PTS_SPARC} points): {n_sparc}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 5: SPARC SUBSAMPLED TO ~10 POINTS (matched-resolution control)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 5] SPARC subsampled to ~10 radial points (matched resolution)...")

sparc_subsampled = []
sub_rng = np.random.default_rng(999)

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
    if np.sum(valid) < MIN_PTS_SPARC:
        continue

    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]
    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]

    n_full = len(R_sorted)
    n_sub = min(10, n_full)

    # Select ~10 approximately evenly-spaced indices
    sub_indices = np.round(np.linspace(0, n_full - 1, n_sub)).astype(int)
    R_sub = R_sorted[sub_indices]
    log_gbar_sub = log_gbar[sub_indices]
    log_gobs_sub = log_gobs[sub_indices]

    residuals_sub = log_gobs_sub - rar_function(log_gbar_sub)
    residuals_dm_sub = residuals_sub - np.mean(residuals_sub)

    r1_dm_sub = lag_autocorrelation(residuals_dm_sub, lag=1, center=True)
    if np.isnan(r1_dm_sub):
        continue

    sparc_subsampled.append({
        'name': name,
        'n_pts': n_sub,
        'r1_dm': float(r1_dm_sub),
    })

n_sparc_sub = len(sparc_subsampled)
print(f"  SPARC subsampled galaxies: {n_sparc_sub}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 6: AGGREGATE STATISTICS & COMPARISONS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 6] Computing aggregate statistics...")

# --- ACF distributions ---
eagle_r1 = np.array([g['r1_dm'] for g in eagle_galaxies])
sparc_r1 = np.array([g['r1_dm'] for g in sparc_galaxies])
sparc_sub_r1 = np.array([g['r1_dm'] for g in sparc_subsampled])

eagle_stats = compute_stats(eagle_r1, 'eagle', rng_seed=100)
sparc_stats = compute_stats(sparc_r1, 'sparc', rng_seed=101)
sparc_sub_stats = compute_stats(sparc_sub_r1, 'sparc_sub', rng_seed=102)

print(f"\n  Lag-1 ACF (demeaned):")
print(f"    EAGLE:         {eagle_stats['mean']:.4f} ± {eagle_stats['se']:.4f}  "
      f"(median {eagle_stats['median']:.4f}, N={eagle_stats['n_total']})")
print(f"    SPARC (full):  {sparc_stats['mean']:.4f} ± {sparc_stats['se']:.4f}  "
      f"(median {sparc_stats['median']:.4f}, N={sparc_stats['n_total']})")
print(f"    SPARC (10pt):  {sparc_sub_stats['mean']:.4f} ± {sparc_sub_stats['se']:.4f}  "
      f"(median {sparc_sub_stats['median']:.4f}, N={sparc_sub_stats['n_total']})")
print(f"    EAGLE frac>0:  {eagle_stats['frac_positive']:.3f}")
print(f"    SPARC frac>0:  {sparc_stats['frac_positive']:.3f}")

# Two-sample comparisons
comp_eagle_sparc = two_sample_comparison(eagle_r1, sparc_r1, 'EAGLE', 'SPARC_full')
comp_eagle_sub = two_sample_comparison(eagle_r1, sparc_sub_r1, 'EAGLE', 'SPARC_10pt')

print(f"\n  EAGLE vs SPARC (full resolution):")
print(f"    Welch p = {comp_eagle_sparc['welch_p']:.4e}, "
      f"KS p = {comp_eagle_sparc['ks_p']:.4e}, "
      f"Cohen's d = {comp_eagle_sparc['cohens_d']}")
print(f"  EAGLE vs SPARC (matched 10-point resolution):")
print(f"    Welch p = {comp_eagle_sub['welch_p']:.4e}, "
      f"KS p = {comp_eagle_sub['ks_p']:.4e}, "
      f"Cohen's d = {comp_eagle_sub['cohens_d']}")

# --- Lomb-Scargle periodicity fractions ---
eagle_ls = [g for g in eagle_galaxies if g.get('ls_valid', False)]
sparc_ls = [g for g in sparc_galaxies if g.get('ls_valid', False)]

eagle_n_sig = sum(1 for g in eagle_ls if g['perm_p'] < 0.05)
sparc_n_sig = sum(1 for g in sparc_ls if g['perm_p'] < 0.05)

eagle_frac_sig = eagle_n_sig / len(eagle_ls) if eagle_ls else 0
sparc_frac_sig = sparc_n_sig / len(sparc_ls) if sparc_ls else 0

# Fisher exact or chi-squared test for difference in proportions
from scipy.stats import fisher_exact
contingency = np.array([
    [eagle_n_sig, len(eagle_ls) - eagle_n_sig],
    [sparc_n_sig, len(sparc_ls) - sparc_n_sig],
])
_, fisher_p_periodicity = fisher_exact(contingency, alternative='two-sided')

print(f"\n  Lomb-Scargle periodicity (p<0.05):")
print(f"    EAGLE: {eagle_n_sig}/{len(eagle_ls)} ({eagle_frac_sig:.1%})")
print(f"    SPARC: {sparc_n_sig}/{len(sparc_ls)} ({sparc_frac_sig:.1%})")
print(f"    Fisher exact p: {fisher_p_periodicity:.4e}")

# Wavelength distributions (significant galaxies only)
eagle_wl_sig = np.array([g['wl_kpc'] for g in eagle_ls if g['perm_p'] < 0.05])
sparc_wl_sig = np.array([g['wl_kpc'] for g in sparc_ls
                           if g['perm_p'] is not None and g['perm_p'] < 0.05
                           and g['wl_kpc'] is not None])

if len(eagle_wl_sig) >= 5:
    print(f"\n  Peak wavelength (significant galaxies only):")
    eq = np.percentile(eagle_wl_sig, [25, 50, 75])
    print(f"    EAGLE: median {eq[1]:.2f} kpc (IQR {eq[0]:.2f}–{eq[2]:.2f}), N={len(eagle_wl_sig)}")
if len(sparc_wl_sig) >= 5:
    sq = np.percentile(sparc_wl_sig, [25, 50, 75])
    print(f"    SPARC: median {sq[1]:.2f} kpc (IQR {sq[0]:.2f}–{sq[2]:.2f}), N={len(sparc_wl_sig)}")

if len(eagle_wl_sig) >= 5 and len(sparc_wl_sig) >= 5:
    wl_comp = two_sample_comparison(eagle_wl_sig, sparc_wl_sig, 'EAGLE', 'SPARC')
    print(f"    Wavelength KS p: {wl_comp['ks_p']:.4e}")

# --- Stacked PSD comparison ---
# Compute power spectral density slope β (PSD ∝ f^{-β})
eagle_psd_slopes = []
sparc_psd_slopes = []

# For EAGLE galaxies with valid LS
for g in eagle_ls:
    if g.get('freq_grid') and g.get('power_spectrum'):
        freq = np.array(g['freq_grid'])
        psd = np.array(g['power_spectrum'])
        mask = (freq > 0) & (psd > 0)
        if np.sum(mask) >= 3:
            log_f = np.log10(freq[mask])
            log_p = np.log10(psd[mask])
            slope, intercept, r, p, se = stats.linregress(log_f, log_p)
            eagle_psd_slopes.append(-slope)  # β = -slope (PSD ∝ f^{-β})

eagle_psd_slopes = np.array(eagle_psd_slopes)
if len(eagle_psd_slopes) > 5:
    print(f"\n  PSD slope β (PSD ∝ f^-β):")
    print(f"    EAGLE: mean {np.mean(eagle_psd_slopes):.3f} ± "
          f"{np.std(eagle_psd_slopes, ddof=1)/np.sqrt(len(eagle_psd_slopes)):.3f}, "
          f"median {np.median(eagle_psd_slopes):.3f}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 7: PUBLICATION FIGURES
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 7] Generating publication figures...")

# ── Figure 1: ACF Distribution Comparison ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: Histograms overlaid
ax = axes[0]
bins = np.linspace(-0.6, 1.0, 35)
ax.hist(eagle_r1, bins=bins, density=True, alpha=0.6, color='#2196F3',
        label=f'EAGLE (N={len(eagle_r1)})', edgecolor='white', linewidth=0.5)
ax.hist(sparc_r1, bins=bins, density=True, alpha=0.6, color='#E91E63',
        label=f'SPARC (N={len(sparc_r1)})', edgecolor='white', linewidth=0.5)
ax.axvline(np.mean(eagle_r1), color='#1565C0', ls='--', lw=1.5,
           label=f'EAGLE mean: {np.mean(eagle_r1):.3f}')
ax.axvline(np.mean(sparc_r1), color='#C2185B', ls='--', lw=1.5,
           label=f'SPARC mean: {np.mean(sparc_r1):.3f}')
ax.axvline(0, color='grey', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Lag-1 ACF (demeaned)')
ax.set_ylabel('Density')
ax.set_title('(a) ACF Distributions')
ax.legend(fontsize=7.5, loc='upper left')

# Panel B: EAGLE vs SPARC (10-pt subsampled, fair comparison)
ax = axes[1]
ax.hist(eagle_r1, bins=bins, density=True, alpha=0.6, color='#2196F3',
        label=f'EAGLE 10 radii', edgecolor='white', linewidth=0.5)
ax.hist(sparc_sub_r1, bins=bins, density=True, alpha=0.6, color='#FF9800',
        label=f'SPARC 10-pt sub', edgecolor='white', linewidth=0.5)
ax.axvline(np.mean(eagle_r1), color='#1565C0', ls='--', lw=1.5,
           label=f'EAGLE: {np.mean(eagle_r1):.3f}')
ax.axvline(np.mean(sparc_sub_r1), color='#E65100', ls='--', lw=1.5,
           label=f'SPARC sub: {np.mean(sparc_sub_r1):.3f}')
ax.axvline(0, color='grey', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Lag-1 ACF (demeaned)')
ax.set_title('(b) Matched Resolution')
ax.legend(fontsize=7.5, loc='upper left')

# Panel C: CDF comparison
ax = axes[2]
for arr, label, color in [(eagle_r1, f'EAGLE (N={len(eagle_r1)})', '#2196F3'),
                            (sparc_r1, f'SPARC full (N={len(sparc_r1)})', '#E91E63'),
                            (sparc_sub_r1, f'SPARC 10pt (N={len(sparc_sub_r1)})', '#FF9800')]:
    sorted_vals = np.sort(arr)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, label=label, lw=1.8)
ax.axvline(0, color='grey', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Lag-1 ACF (demeaned)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('(c) CDF Comparison')
ax.legend(fontsize=7.5, loc='lower right')

fig.suptitle('Radial Coherence: EAGLE ΛCDM vs SPARC Observations', fontsize=14, y=1.02)
plt.tight_layout()
fig1_path = os.path.join(RESULTS_DIR, 'simulation_coherence_acf.png')
fig.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fig1_path}")


# ── Figure 2: Periodicity Comparison ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: Periodicity fraction bar chart
ax = axes[0]
categories = ['EAGLE', 'SPARC']
fracs = [eagle_frac_sig * 100, sparc_frac_sig * 100]
colors = ['#2196F3', '#E91E63']
bars = ax.bar(categories, fracs, color=colors, alpha=0.8, edgecolor='white', width=0.5)
ax.axhline(5.0, color='grey', ls='--', lw=1, label='5% null expectation')
for bar, frac, n_s, n_t in zip(bars, fracs, [eagle_n_sig, sparc_n_sig],
                                  [len(eagle_ls), len(sparc_ls)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{frac:.1f}%\n({n_s}/{n_t})', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Fraction with significant periodicity (%)')
ax.set_title('(a) Periodicity Fraction')
ax.legend(fontsize=8)
ax.set_ylim(0, max(fracs) * 1.3 + 5)

# Panel B: Wavelength distributions (significant galaxies)
ax = axes[1]
if len(eagle_wl_sig) >= 3:
    wl_bins = np.logspace(np.log10(0.5), np.log10(200), 25)
    ax.hist(eagle_wl_sig, bins=wl_bins, density=True, alpha=0.6, color='#2196F3',
            label=f'EAGLE (N={len(eagle_wl_sig)})', edgecolor='white', linewidth=0.5)
if len(sparc_wl_sig) >= 3:
    ax.hist(sparc_wl_sig, bins=wl_bins, density=True, alpha=0.6, color='#E91E63',
            label=f'SPARC (N={len(sparc_wl_sig)})', edgecolor='white', linewidth=0.5)
ax.set_xscale('log')
ax.set_xlabel('Peak wavelength (kpc)')
ax.set_ylabel('Density')
ax.set_title('(b) Peak Wavelength (sig. only)')
ax.legend(fontsize=8)

# Panel C: Peak power distributions
ax = axes[2]
eagle_pp = np.array([g['power_peak'] for g in eagle_ls])
sparc_pp = np.array([g['power_peak'] for g in sparc_ls
                       if g['power_peak'] is not None])
pp_bins = np.linspace(0, 1, 30)
if len(eagle_pp) > 0:
    ax.hist(eagle_pp, bins=pp_bins, density=True, alpha=0.6, color='#2196F3',
            label=f'EAGLE', edgecolor='white', linewidth=0.5)
if len(sparc_pp) > 0:
    ax.hist(sparc_pp, bins=pp_bins, density=True, alpha=0.6, color='#E91E63',
            label=f'SPARC', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Peak Lomb-Scargle power')
ax.set_ylabel('Density')
ax.set_title('(c) Peak Power Distribution')
ax.legend(fontsize=8)

fig.suptitle('Spectral Analysis: EAGLE ΛCDM vs SPARC', fontsize=14, y=1.02)
plt.tight_layout()
fig2_path = os.path.join(RESULTS_DIR, 'simulation_coherence_periodicity.png')
fig.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fig2_path}")


# ── Figure 3: Summary Panel ────────────────────────────────────────
fig = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# Panel A: ACF comparison scatter / violin-like
ax = fig.add_subplot(gs[0, 0])
positions = [1, 2, 3]
data_plot = [eagle_r1, sparc_r1, sparc_sub_r1]
labels_plot = ['EAGLE\n(10 radii)', 'SPARC\n(full)', 'SPARC\n(10pt sub)']
colors_plot = ['#2196F3', '#E91E63', '#FF9800']
bp = ax.boxplot(data_plot, positions=positions, widths=0.5, patch_artist=True,
                showfliers=False, medianprops=dict(color='black', lw=2))
for patch, color in zip(bp['boxes'], colors_plot):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_xticks(positions)
ax.set_xticklabels(labels_plot, fontsize=9)
ax.set_ylabel('Lag-1 ACF (demeaned)')
ax.axhline(0, color='grey', ls=':', lw=0.8)
ax.set_title('(a) ACF Summary')

# Panel B: ACF perm-p distribution (EAGLE)
ax = fig.add_subplot(gs[0, 1])
valid_pp = eagle_perm_arr[~np.isnan(eagle_perm_arr)]
ax.hist(valid_pp, bins=np.linspace(0, 1, 30), density=True, alpha=0.7,
        color='#2196F3', edgecolor='white', linewidth=0.5)
ax.axhline(1.0, color='grey', ls='--', lw=1, label='Uniform null')
ax.axvline(0.05, color='red', ls='--', lw=1, label='p=0.05')
ax.set_xlabel('Per-galaxy permutation p-value')
ax.set_ylabel('Density')
ax.set_title(f'(b) EAGLE ACF Perm. Null\n'
             f'({n_acf_sig}/{len(valid_pp)} sig. at 0.05)')
ax.legend(fontsize=8)

# Panel C: Key comparison metrics
ax = fig.add_subplot(gs[1, :])
ax.axis('off')

table_data = [
    ['Metric', 'EAGLE', 'SPARC (full)', 'SPARC (10pt)', 'p-value'],
    ['N galaxies', f'{n_eagle}', f'{n_sparc}', f'{n_sparc_sub}', '—'],
    ['Lag-1 ACF mean', f'{eagle_stats["mean"]:.4f}',
     f'{sparc_stats["mean"]:.4f}', f'{sparc_sub_stats["mean"]:.4f}',
     f'{comp_eagle_sparc["welch_p"]:.2e}'],
    ['Lag-1 ACF median', f'{eagle_stats["median"]:.4f}',
     f'{sparc_stats["median"]:.4f}', f'{sparc_sub_stats["median"]:.4f}',
     f'{comp_eagle_sub["welch_p"]:.2e}'],
    ['Fraction ACF > 0', f'{eagle_stats["frac_positive"]:.3f}',
     f'{sparc_stats["frac_positive"]:.3f}', f'{sparc_sub_stats["frac_positive"]:.3f}', '—'],
    ['KS statistic', f'{comp_eagle_sparc["ks_stat"]:.4f}', '—',
     f'{comp_eagle_sub["ks_stat"]:.4f}',
     f'{comp_eagle_sparc["ks_p"]:.2e}'],
    ['LS periodicity %', f'{eagle_frac_sig*100:.1f}%',
     f'{sparc_frac_sig*100:.1f}%', '—', f'{fisher_p_periodicity:.2e}'],
    ["Cohen's d (vs SPARC full)", f'{comp_eagle_sparc["cohens_d"]:.3f}',
     '—', f'{comp_eagle_sub["cohens_d"]:.3f}', '—'],
]

# Draw table
cell_colors = []
for i, row in enumerate(table_data):
    if i == 0:
        cell_colors.append(['#E0E0E0'] * 5)
    else:
        cell_colors.append(['#F5F5F5' if i % 2 == 1 else 'white'] * 5)

table = ax.table(cellText=table_data, cellColours=cell_colors,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.5)
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_text_props(weight='bold')
ax.set_title('(c) Quantitative Comparison', fontsize=12, pad=10)

fig.suptitle('Simulation Coherence Test: EAGLE ΛCDM vs SPARC\n'
             'Do simulations reproduce observed radial coherence in RAR residuals?',
             fontsize=13, y=1.04)
fig3_path = os.path.join(RESULTS_DIR, 'simulation_coherence_summary.png')
fig.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fig3_path}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 8: VERDICT & SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [PHASE 8] Final verdict...")

# Determine discriminating power
acf_diff = abs(eagle_stats['mean'] - sparc_stats['mean'])
acf_diff_sub = abs(eagle_stats['mean'] - sparc_sub_stats['mean'])

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)

print(f"\n  ACF COMPARISON:")
print(f"    EAGLE mean ACF:     {eagle_stats['mean']:.4f} ± {eagle_stats['se']:.4f}")
print(f"    SPARC mean ACF:     {sparc_stats['mean']:.4f} ± {sparc_stats['se']:.4f}")
print(f"    SPARC 10pt ACF:     {sparc_sub_stats['mean']:.4f} ± {sparc_sub_stats['se']:.4f}")
print(f"    Difference (full):  {acf_diff:.4f}")
print(f"    Difference (10pt):  {acf_diff_sub:.4f}")
print(f"    Welch p (full):     {comp_eagle_sparc['welch_p']:.4e}")
print(f"    Welch p (10pt):     {comp_eagle_sub['welch_p']:.4e}")

print(f"\n  PERIODICITY COMPARISON:")
print(f"    EAGLE fraction:     {eagle_frac_sig:.1%}")
print(f"    SPARC fraction:     {sparc_frac_sig:.1%}")
print(f"    Fisher p:           {fisher_p_periodicity:.4e}")

print(f"\n  PERMUTATION NULL (EAGLE):")
print(f"    Galaxies with ACF sig. > 0: {n_acf_sig}/{len(eagle_perm_arr)} "
      f"({100*n_acf_sig/len(eagle_perm_arr):.1f}%)")
print(f"    Fisher combined p:  {fisher_p:.4e}")

# Classification
if comp_eagle_sparc['welch_p'] < 0.001:
    if eagle_stats['mean'] > sparc_stats['mean']:
        acf_verdict = "EAGLE_HIGHER_ACF"
        acf_text = "EAGLE shows HIGHER ACF than SPARC — simulations are MORE coherent"
    else:
        acf_verdict = "EAGLE_LOWER_ACF"
        acf_text = "EAGLE shows LOWER ACF than SPARC — simulations are LESS coherent"
elif comp_eagle_sparc['welch_p'] < 0.05:
    acf_verdict = "MARGINALLY_DIFFERENT"
    acf_text = "ACF distributions marginally different (p < 0.05)"
else:
    acf_verdict = "CONSISTENT"
    acf_text = "ACF distributions CONSISTENT — ΛCDM reproduces observed coherence"

if fisher_p_periodicity < 0.001:
    if eagle_frac_sig > sparc_frac_sig:
        ls_verdict = "EAGLE_MORE_PERIODIC"
    else:
        ls_verdict = "SPARC_MORE_PERIODIC"
elif fisher_p_periodicity < 0.05:
    ls_verdict = "MARGINALLY_DIFFERENT"
else:
    ls_verdict = "CONSISTENT"

print(f"\n  ACF VERDICT: {acf_verdict}")
print(f"    {acf_text}")
print(f"  LS VERDICT:  {ls_verdict}")

# Overall
if acf_verdict == "CONSISTENT" and ls_verdict == "CONSISTENT":
    overall = ("NOT_DISCRIMINATING — ΛCDM simulations reproduce the same radial "
               "coherence as observations. Coherence is NOT unique to BEC.")
elif acf_verdict in ["EAGLE_LOWER_ACF"] or ls_verdict == "SPARC_MORE_PERIODIC":
    overall = ("POTENTIALLY_DISCRIMINATING — SPARC shows MORE coherence/periodicity "
               "than ΛCDM. This could support a quantum coherence interpretation, "
               "but resolution effects need investigation.")
else:
    overall = ("INCONCLUSIVE — Mixed results; further analysis needed.")

print(f"\n  OVERALL: {overall}")


# ── TNG Requirements Documentation ───────────────────────────────
tng_doc = {
    'status': 'INSUFFICIENT_DATA',
    'available_radii': 4,
    'radii_kpc': [5, 10, 30, 100],
    'minimum_needed': 8,
    'extraction_requirements': {
        'method': 'TNG API particle data download',
        'description': ('Download particle data for each subhalo and compute '
                        'enclosed mass profiles at ~15 logarithmically spaced '
                        'radii from 0.5×R_half to 5×R_half'),
        'api_endpoint': 'https://www.tng-project.org/api/TNG100-1/snapshots/99/subhalos/<id>/',
        'data_needed': ['PartType0 (gas) coordinates + masses',
                        'PartType1 (DM) coordinates + masses',
                        'PartType4 (stars) coordinates + masses'],
        'n_galaxies': '~20,000 (after mass cut)',
        'estimated_download': '~50-100 GB of particle data',
        'estimated_compute': '~2-4 hours on cloud with 8+ cores',
        'alternative': ('Use TNG JupyterLab (https://www.tng-project.org/data/lab/) '
                        'to run batch extraction script server-side, avoiding download'),
    },
}


# ── Save JSON results ─────────────────────────────────────────────
results = {
    'test': 'simulation_coherence',
    'description': ('Comparison of radial coherence (ACF, Lomb-Scargle periodicity) '
                    'in RAR residuals between EAGLE ΛCDM simulation and SPARC observations.'),
    'parameters': {
        'g_dagger': g_dagger,
        'min_pts_eagle': MIN_PTS_EAGLE,
        'min_pts_sparc': MIN_PTS_SPARC,
        'min_mbar': MIN_MBAR_MSUN,
        'n_surrogates': N_SURR,
        'n_bootstrap': N_BOOT,
        'n_permutations': N_PERM,
        'eagle_aperture_radii_kpc': ap_sizes,
    },
    'sample': {
        'eagle_n_galaxies': n_eagle,
        'sparc_n_galaxies': n_sparc,
        'sparc_subsampled_n': n_sparc_sub,
    },
    'eagle_acf': eagle_stats,
    'sparc_acf': sparc_stats,
    'sparc_subsampled_acf': sparc_sub_stats,
    'comparison_eagle_vs_sparc_full': comp_eagle_sparc,
    'comparison_eagle_vs_sparc_10pt': comp_eagle_sub,
    'eagle_permutation_null': {
        'n_sig_005': n_acf_sig,
        'n_total': len(eagle_perm_arr),
        'frac_sig': round(n_acf_sig / len(eagle_perm_arr), 4) if len(eagle_perm_arr) > 0 else None,
        'fisher_combined_p': fisher_p,
    },
    'periodicity': {
        'eagle_n_sig': eagle_n_sig,
        'eagle_n_total': len(eagle_ls),
        'eagle_frac': round(eagle_frac_sig, 4),
        'sparc_n_sig': sparc_n_sig,
        'sparc_n_total': len(sparc_ls),
        'sparc_frac': round(sparc_frac_sig, 4),
        'fisher_p': fisher_p_periodicity,
    },
    'wavelength_eagle_sig': {
        'n': len(eagle_wl_sig),
        'median_kpc': round(float(np.median(eagle_wl_sig)), 2) if len(eagle_wl_sig) > 0 else None,
        'q25_kpc': round(float(np.percentile(eagle_wl_sig, 25)), 2) if len(eagle_wl_sig) >= 4 else None,
        'q75_kpc': round(float(np.percentile(eagle_wl_sig, 75)), 2) if len(eagle_wl_sig) >= 4 else None,
    },
    'wavelength_sparc_sig': {
        'n': len(sparc_wl_sig),
        'median_kpc': round(float(np.median(sparc_wl_sig)), 2) if len(sparc_wl_sig) > 0 else None,
        'q25_kpc': round(float(np.percentile(sparc_wl_sig, 25)), 2) if len(sparc_wl_sig) >= 4 else None,
        'q75_kpc': round(float(np.percentile(sparc_wl_sig, 75)), 2) if len(sparc_wl_sig) >= 4 else None,
    },
    'psd_slopes_eagle': {
        'n': len(eagle_psd_slopes),
        'mean': round(float(np.mean(eagle_psd_slopes)), 3) if len(eagle_psd_slopes) > 0 else None,
        'se': round(float(np.std(eagle_psd_slopes, ddof=1)/np.sqrt(len(eagle_psd_slopes))), 3)
              if len(eagle_psd_slopes) > 1 else None,
        'median': round(float(np.median(eagle_psd_slopes)), 3) if len(eagle_psd_slopes) > 0 else None,
    },
    'verdicts': {
        'acf': acf_verdict,
        'periodicity': ls_verdict,
        'overall': overall,
    },
    'tng_requirements': tng_doc,
    'figures': {
        'acf_comparison': fig1_path,
        'periodicity_comparison': fig2_path,
        'summary': fig3_path,
    },
}

outpath = os.path.join(RESULTS_DIR, 'summary_simulation_coherence.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n{elapsed()} Results saved to: {outpath}")
print("=" * 76)
print("Done.")
