#!/usr/bin/env python3
"""
TEST B: Power Spectrum Shape of RAR Residuals — Turbulence vs Oscillation
=========================================================================

Extends D3 (test_interface_spectral.py) by analyzing the FULL spectral shape
of RAR residuals, not just peak significance.

Diagnostic:
  - Power-law decay P(k) ~ k^(-beta) => turbulence
    (Kolmogorov beta ~ 5/3, Burgers beta ~ 2, self-gravitating ~ 1.5-2.5)
  - Spectral peak at characteristic frequency => coherent oscillation
  - Flat/white noise => no spatial structure

Steps:
  1. Load SPARC RAR residuals (67 galaxies, >=15 pts, same as D3)
  2. Compute full Lomb-Scargle PSD per galaxy
  3. Stack PSDs in physical k [1/kpc] and dimensionless k*xi coordinates
  4. Fit power-law P(k) = A * k^(-beta); compare to Kolmogorov/Burgers
  5. Test whether PSD collapses better in k*xi (ξ as natural coherence scale)
  6. Compare "periodic" (25 sig) vs "non-periodic" (42) stacked PSDs

Russell Licht -- BEC Dark Matter Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
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
g_dagger = 1.20e-10       # m/s^2
G_SI = 6.674e-11           # m^3 kg^-1 s^-2
Msun_kg = 1.989e30          # kg
kpc_m = 3.086e19            # m
MIN_POINTS = 15
N_SURR = 200                # surrogates for significance (same as D3)
PERM_ALPHA = 0.05           # significance threshold for "periodic" classification

np.random.seed(42)

print("=" * 72)
print("TEST B: POWER SPECTRUM SHAPE OF RAR RESIDUALS")
print("  Turbulence vs Coherent Oscillation")
print("=" * 72)


def rar_function(log_gbar, a0=1.2e-10):
    """Standard RAR prediction: log(g_obs) from log(g_bar)."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def power_law(log_k, log_A, beta):
    """Power law in log-log: log P = log A - beta * log k."""
    return log_A - beta * log_k


# ================================================================
# 1. LOAD SPARC DATA + COMPUTE DETRENDED RESIDUALS (identical to D3)
# ================================================================
print("\n[1] Loading SPARC data and computing detrended residuals...")

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
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in rc_data:
            rc_data[name] = {'R': [], 'Vobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': [], 'dist': dist}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in ['R', 'Vobs', 'Vgas', 'Vdisk', 'Vbul']:
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
            'Vflat': float(parts[14]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

# Build per-galaxy data
galaxy_data = []

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
        continue

    R_valid = R[valid]
    sort_idx = np.argsort(R_valid)
    R_sorted = R_valid[sort_idx]

    log_gbar = np.log10(gbar_SI[valid])[sort_idx]
    log_gobs = np.log10(gobs_SI[valid])[sort_idx]
    residuals = log_gobs - rar_function(log_gbar)

    # Spline detrending (same as D3)
    n = len(residuals)
    var_eps = np.var(residuals)
    s_param = n * var_eps * 0.5
    try:
        spline = UnivariateSpline(R_sorted, residuals, k=min(3, n - 1), s=s_param)
        eps_det = residuals - spline(R_sorted)
    except Exception:
        eps_det = residuals - np.mean(residuals)

    # Dynamical mass at outermost point for healing length
    Vobs_sorted = gdata['Vobs'][valid][sort_idx]
    R_max_kpc = R_sorted[-1]
    V_max_kms = Vobs_sorted[-1]
    M_dyn = (V_max_kms * 1e3)**2 * (R_max_kpc * kpc_m) / G_SI  # kg
    xi_kpc = np.sqrt(G_SI * M_dyn / g_dagger) / kpc_m  # healing length in kpc

    galaxy_data.append({
        'name': name,
        'R': R_sorted,
        'eps_det': eps_det,
        'n_pts': n,
        'Vflat': prop['Vflat'],
        'M_dyn': M_dyn,
        'xi_kpc': xi_kpc,
        'R_extent': float(R_sorted[-1] - R_sorted[0]),
    })

n_galaxies = len(galaxy_data)
print(f"  {n_galaxies} galaxies with N >= {MIN_POINTS}")
xi_arr = np.array([g['xi_kpc'] for g in galaxy_data])
print(f"  Healing lengths: median={np.median(xi_arr):.1f} kpc, "
      f"range=[{np.min(xi_arr):.1f}, {np.max(xi_arr):.1f}] kpc")


# ================================================================
# 2. COMPUTE FULL LOMB-SCARGLE PSD PER GALAXY
# ================================================================
print("\n[2] Computing full Lomb-Scargle PSD per galaxy...")

# We also run the permutation test to classify periodic vs non-periodic
perm_rng = np.random.default_rng(789)

# Common frequency grid for stacking: sample in log-space
# Use physical k in [0.01, 5] 1/kpc (wavelengths from 200 kpc down to 0.2 kpc)
N_FREQ_COMMON = 200
k_common = np.logspace(np.log10(0.02), np.log10(3.0), N_FREQ_COMMON)  # 1/kpc

psd_per_galaxy = []

for gi, g in enumerate(galaxy_data):
    R = g['R']
    eps = g['eps_det']
    n = g['n_pts']

    # Standardize residuals
    std_eps = np.std(eps)
    if std_eps < 1e-30:
        continue
    y = (eps - np.mean(eps)) / std_eps

    R_extent = R[-1] - R[0]
    if R_extent <= 0:
        continue

    # Galaxy-specific frequency grid (dense, for accurate PSD)
    f_min = 1.0 / R_extent
    f_max = (n / 2.0) / R_extent
    n_freq_gal = min(500, 10 * n)
    freq_gal = np.linspace(f_min, f_max, n_freq_gal)

    # Compute Lomb-Scargle PSD on galaxy-specific grid
    ls = LombScargle(R, y, fit_mean=False, center_data=True)
    power_gal = ls.power(freq_gal)

    # Find peak + permutation p-value (same as D3)
    idx_peak = np.argmax(power_gal)
    f_peak = float(freq_gal[idx_peak])
    power_peak = float(power_gal[idx_peak])

    null_peaks = np.zeros(N_SURR)
    for s in range(N_SURR):
        y_shuf = perm_rng.permutation(y)
        ls_null = LombScargle(R, y_shuf, fit_mean=False, center_data=True)
        p_null = ls_null.power(freq_gal)
        null_peaks[s] = np.max(p_null)
    p_val = float(np.mean(null_peaks >= power_peak))

    # Also compute PSD on the common k grid (for stacking)
    # Only evaluate at frequencies within the galaxy's valid range
    valid_k = (k_common >= f_min) & (k_common <= f_max)
    psd_common = np.full(N_FREQ_COMMON, np.nan)
    if np.sum(valid_k) > 3:
        psd_common[valid_k] = ls.power(k_common[valid_k])

    # Dimensionless coordinates: k_dim = k * xi
    xi = g['xi_kpc']

    psd_per_galaxy.append({
        'name': g['name'],
        'n_pts': n,
        'R_extent': g['R_extent'],
        'xi_kpc': xi,
        'Vflat': g['Vflat'],
        'f_peak': f_peak,
        'wl_kpc': 1.0 / f_peak,
        'power_peak': power_peak,
        'perm_p': p_val,
        'is_periodic': p_val < PERM_ALPHA,
        'psd_common': psd_common,  # PSD on common k grid
        'freq_gal': freq_gal,      # galaxy-specific grid
        'power_gal': power_gal,    # galaxy-specific PSD
    })

    if (gi + 1) % 20 == 0:
        print(f"    {gi+1}/{n_galaxies} done...")

n_total = len(psd_per_galaxy)
n_periodic = sum(1 for g in psd_per_galaxy if g['is_periodic'])
n_nonperiodic = n_total - n_periodic
print(f"  Total galaxies: {n_total}")
print(f"  Periodic (p<0.05): {n_periodic}, Non-periodic: {n_nonperiodic}")


# ================================================================
# 3. STACK PSDs IN PHYSICAL AND DIMENSIONLESS COORDINATES
# ================================================================
print("\n[3] Stacking PSDs...")

# 3a: Physical coordinates k [1/kpc]
psd_matrix_phys = np.array([g['psd_common'] for g in psd_per_galaxy])  # (N_gal, N_FREQ_COMMON)

# Compute median and percentile envelopes, ignoring NaN
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    psd_median_phys = np.nanmedian(psd_matrix_phys, axis=0)
    psd_p16_phys = np.nanpercentile(psd_matrix_phys, 16, axis=0)
    psd_p84_phys = np.nanpercentile(psd_matrix_phys, 84, axis=0)
    n_contributing = np.sum(~np.isnan(psd_matrix_phys), axis=0)

# Mask where too few galaxies contribute
min_contrib = 10
good_phys = n_contributing >= min_contrib

print(f"  Physical stacking: {np.sum(good_phys)}/{N_FREQ_COMMON} frequency bins with >= {min_contrib} galaxies")

# 3b: Dimensionless coordinates k_dim = k * xi
# Each galaxy has a different xi, so we need to remap
N_DIM_COMMON = 200
k_dim_common = np.logspace(np.log10(0.1), np.log10(30.0), N_DIM_COMMON)

psd_matrix_dim = np.full((n_total, N_DIM_COMMON), np.nan)

for gi, g in enumerate(psd_per_galaxy):
    xi = g['xi_kpc']
    # Map this galaxy's PSD from k_common to k_dim = k * xi
    k_phys_this = k_common  # 1/kpc
    k_dim_this = k_phys_this * xi  # dimensionless
    psd_this = g['psd_common']

    # Interpolate onto common dimensionless grid
    valid = ~np.isnan(psd_this) & (psd_this > 0)
    if np.sum(valid) < 5:
        continue

    # Use log-log interpolation for smooth PSD mapping
    log_k_dim_valid = np.log10(k_dim_this[valid])
    log_psd_valid = np.log10(psd_this[valid])

    # Only interpolate within the range of valid data
    dim_in_range = (np.log10(k_dim_common) >= log_k_dim_valid.min()) & \
                   (np.log10(k_dim_common) <= log_k_dim_valid.max())
    if np.sum(dim_in_range) < 3:
        continue

    psd_matrix_dim[gi, dim_in_range] = 10**np.interp(
        np.log10(k_dim_common[dim_in_range]),
        log_k_dim_valid,
        log_psd_valid
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    psd_median_dim = np.nanmedian(psd_matrix_dim, axis=0)
    psd_p16_dim = np.nanpercentile(psd_matrix_dim, 16, axis=0)
    psd_p84_dim = np.nanpercentile(psd_matrix_dim, 84, axis=0)
    n_contrib_dim = np.sum(~np.isnan(psd_matrix_dim), axis=0)

good_dim = n_contrib_dim >= min_contrib
print(f"  Dimensionless stacking: {np.sum(good_dim)}/{N_DIM_COMMON} bins with >= {min_contrib} galaxies")


# ================================================================
# 4. FIT POWER LAW TO STACKED PSD
# ================================================================
print("\n[4] Fitting power-law P(k) = A * k^(-beta) to stacked PSD...")

def fit_power_law(k_arr, psd_arr, good_mask, label=""):
    """Fit power law in log-log space where PSD is well-defined."""
    use = good_mask & (psd_arr > 0) & np.isfinite(psd_arr) & (k_arr > 0)
    if np.sum(use) < 5:
        print(f"  [{label}] Too few valid points for fit ({np.sum(use)})")
        return None, None, None, None

    log_k = np.log10(k_arr[use])
    log_psd = np.log10(psd_arr[use])

    # Identify the approximately linear region: use the full range first
    try:
        popt, pcov = curve_fit(power_law, log_k, log_psd, p0=[0.0, 1.5])
        perr = np.sqrt(np.diag(pcov))

        beta = popt[1]
        beta_err = perr[1]
        log_A = popt[0]

        # Residuals from power-law fit
        resid = log_psd - power_law(log_k, *popt)
        chi2_red = np.sum(resid**2) / (len(resid) - 2)

        print(f"  [{label}] beta = {beta:.3f} +/- {beta_err:.3f}, "
              f"chi2_red = {chi2_red:.3f}")

        return beta, beta_err, log_A, chi2_red
    except Exception as e:
        print(f"  [{label}] Fit failed: {e}")
        return None, None, None, None


# Fit physical-coordinate stacked PSD
beta_phys, beta_phys_err, logA_phys, chi2_phys = fit_power_law(
    k_common, psd_median_phys, good_phys, "Physical k"
)

# Fit dimensionless-coordinate stacked PSD
beta_dim, beta_dim_err, logA_dim, chi2_dim = fit_power_law(
    k_dim_common, psd_median_dim, good_dim, "Dimensionless k*xi"
)

# Compare to theoretical predictions
print("\n  Theoretical comparisons:")
print(f"    Kolmogorov (incompressible):      beta = 5/3 = 1.667")
print(f"    Burgers (shock-dominated):        beta = 2.0")
print(f"    Self-gravitating turbulence:      beta ~ 1.5-2.5")
print(f"    White noise:                      beta = 0")

if beta_phys is not None:
    for model, beta_pred in [("Kolmogorov", 5/3), ("Burgers", 2.0)]:
        z_score = abs(beta_phys - beta_pred) / beta_phys_err if beta_phys_err > 0 else np.inf
        print(f"    Physical beta vs {model}: |z| = {z_score:.2f} "
              f"({'consistent' if z_score < 2 else 'rejected'} at 2sigma)")


# ================================================================
# 5. COLLAPSE QUALITY: PHYSICAL vs DIMENSIONLESS
# ================================================================
print("\n[5] Comparing PSD collapse quality (physical vs dimensionless)...")

def envelope_width(psd_p16, psd_p84, good_mask):
    """Mean log-space envelope width where data is valid."""
    use = good_mask & (psd_p16 > 0) & (psd_p84 > 0) & np.isfinite(psd_p16) & np.isfinite(psd_p84)
    if np.sum(use) < 5:
        return np.nan
    widths = np.log10(psd_p84[use]) - np.log10(psd_p16[use])
    return float(np.mean(widths))

width_phys = envelope_width(psd_p16_phys, psd_p84_phys, good_phys)
width_dim = envelope_width(psd_p16_dim, psd_p84_dim, good_dim)

print(f"  Physical k: mean envelope width = {width_phys:.4f} dex")
print(f"  Dimensionless k*xi: mean envelope width = {width_dim:.4f} dex")
if not np.isnan(width_phys) and not np.isnan(width_dim):
    ratio = width_dim / width_phys
    better = "dimensionless" if ratio < 1.0 else "physical"
    print(f"  Ratio (dim/phys) = {ratio:.3f} => {better} coordinates give tighter collapse")

    # Bootstrap test for significance
    n_boot = 1000
    boot_rng = np.random.default_rng(123)
    width_phys_boots = []
    width_dim_boots = []

    for _ in range(n_boot):
        idx = boot_rng.choice(n_total, size=n_total, replace=True)

        # Physical
        boot_phys = psd_matrix_phys[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bp16 = np.nanpercentile(boot_phys, 16, axis=0)
            bp84 = np.nanpercentile(boot_phys, 84, axis=0)
        w = envelope_width(bp16, bp84, good_phys)
        width_phys_boots.append(w)

        # Dimensionless
        boot_dim = psd_matrix_dim[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dp16 = np.nanpercentile(boot_dim, 16, axis=0)
            dp84 = np.nanpercentile(boot_dim, 84, axis=0)
        w = envelope_width(dp16, dp84, good_dim)
        width_dim_boots.append(w)

    width_phys_boots = np.array(width_phys_boots)
    width_dim_boots = np.array(width_dim_boots)
    diff_boots = width_dim_boots - width_phys_boots
    valid_boots = np.isfinite(diff_boots)
    if np.sum(valid_boots) > 50:
        p_tighter_dim = float(np.mean(diff_boots[valid_boots] < 0))
        print(f"  Bootstrap P(dimensionless tighter) = {p_tighter_dim:.3f} ({n_boot} iterations)")
    else:
        p_tighter_dim = None
else:
    ratio = None
    p_tighter_dim = None


# ================================================================
# 6. PERIODIC vs NON-PERIODIC COMPARISON
# ================================================================
print("\n[6] Comparing periodic vs non-periodic galaxy PSDs...")

periodic_idx = [i for i, g in enumerate(psd_per_galaxy) if g['is_periodic']]
nonperiodic_idx = [i for i, g in enumerate(psd_per_galaxy) if not g['is_periodic']]

psd_matrix_periodic = psd_matrix_phys[periodic_idx]
psd_matrix_nonperiodic = psd_matrix_phys[nonperiodic_idx]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    psd_med_per = np.nanmedian(psd_matrix_periodic, axis=0)
    psd_p16_per = np.nanpercentile(psd_matrix_periodic, 16, axis=0)
    psd_p84_per = np.nanpercentile(psd_matrix_periodic, 84, axis=0)
    n_per = np.sum(~np.isnan(psd_matrix_periodic), axis=0)

    psd_med_nonper = np.nanmedian(psd_matrix_nonperiodic, axis=0)
    psd_p16_nonper = np.nanpercentile(psd_matrix_nonperiodic, 16, axis=0)
    psd_p84_nonper = np.nanpercentile(psd_matrix_nonperiodic, 84, axis=0)
    n_nonper = np.sum(~np.isnan(psd_matrix_nonperiodic), axis=0)

good_per = n_per >= 5
good_nonper = n_nonper >= 5

# Fit power laws to each subset
print("  Periodic galaxies:")
beta_per, beta_per_err, logA_per, chi2_per = fit_power_law(
    k_common, psd_med_per, good_per, "Periodic"
)

print("  Non-periodic galaxies:")
beta_nonper, beta_nonper_err, logA_nonper, chi2_nonper = fit_power_law(
    k_common, psd_med_nonper, good_nonper, "Non-periodic"
)

# Compare power levels and slopes
if beta_per is not None and beta_nonper is not None:
    delta_beta = beta_per - beta_nonper
    delta_beta_err = np.sqrt(beta_per_err**2 + beta_nonper_err**2)
    print(f"\n  Delta(beta) = {delta_beta:.3f} +/- {delta_beta_err:.3f}")
    print(f"  Periodic {'steeper' if delta_beta > 0 else 'flatter'} than non-periodic")

# Check for spectral bump in periodic galaxies: excess over power law
# Compute ratio of periodic PSD to its own power-law fit
use_per = good_per & (psd_med_per > 0) & np.isfinite(psd_med_per)
use_nonper = good_nonper & (psd_med_nonper > 0) & np.isfinite(psd_med_nonper)

if beta_per is not None and np.sum(use_per) > 5:
    log_k_per = np.log10(k_common[use_per])
    log_psd_per = np.log10(psd_med_per[use_per])
    predicted_per = power_law(log_k_per, logA_per, beta_per)
    excess_per = log_psd_per - predicted_per  # positive = bump above power law

    # Find peak excess
    idx_max_excess = np.argmax(excess_per)
    k_bump = 10**log_k_per[idx_max_excess]
    wl_bump = 1.0 / k_bump
    max_excess = excess_per[idx_max_excess]
    print(f"\n  Periodic galaxies — max excess above power law:")
    print(f"    At k = {k_bump:.3f} 1/kpc (lambda = {wl_bump:.1f} kpc)")
    print(f"    Excess = {max_excess:.3f} dex")
    print(f"    {'SPECTRAL BUMP DETECTED' if max_excess > 0.05 else 'No significant bump'}")
else:
    k_bump = None
    wl_bump = None
    max_excess = None


# ================================================================
# 7. PUBLICATION-QUALITY FIGURES
# ================================================================
print("\n[7] Generating publication-quality figures...")

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# --- Figure 1: Stacked PSD in physical coordinates ---
fig1, ax1 = plt.subplots(figsize=(7, 5))

use = good_phys & (psd_median_phys > 0) & np.isfinite(psd_median_phys)
ax1.fill_between(k_common[use], psd_p16_phys[use], psd_p84_phys[use],
                 alpha=0.25, color='steelblue', label='16th-84th percentile')
ax1.plot(k_common[use], psd_median_phys[use], 'o-', color='steelblue',
         markersize=2, linewidth=1.2, label='Median PSD')

# Overlay power-law fit
if beta_phys is not None:
    k_fit = k_common[use]
    psd_fit = 10**(power_law(np.log10(k_fit), logA_phys, beta_phys))
    ax1.plot(k_fit, psd_fit, '--', color='firebrick', linewidth=1.5,
             label=rf'$P(k) \propto k^{{-{beta_phys:.2f} \pm {beta_phys_err:.2f}}}$')

# Reference slopes
k_ref = np.array([k_common[use].min(), k_common[use].max()])
for beta_ref, name_ref, color_ref, ls_ref in [
    (5/3, 'Kolmogorov (5/3)', 'gray', ':'),
    (2.0, 'Burgers (2)', 'gray', '-.'),
]:
    norm = psd_median_phys[use][len(psd_median_phys[use])//4]  # normalize at 1/4 point
    k_norm = k_common[use][len(k_common[use])//4]
    psd_ref = norm * (k_ref / k_norm)**(-beta_ref)
    ax1.plot(k_ref, psd_ref, ls_ref, color=color_ref, linewidth=1.0,
             alpha=0.6, label=rf'$k^{{-{beta_ref:.1f}}}$ ({name_ref})')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Spatial frequency $k$ [kpc$^{-1}$]', fontsize=12)
ax1.set_ylabel(r'Lomb-Scargle Power $P(k)$', fontsize=12)
ax1.set_title('Stacked PSD of RAR Residuals — Physical Coordinates', fontsize=13)
ax1.legend(fontsize=9, loc='upper right')

# Add text box with galaxy count
n_range = f"{int(n_contributing[use].min())}-{int(n_contributing[use].max())}"
ax1.text(0.03, 0.05, f'{n_total} galaxies\n{n_range} per bin',
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

fig1.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'psd_stacked_physical.png')
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig1_path}")
plt.close(fig1)


# --- Figure 2: Stacked PSD in dimensionless coordinates ---
fig2, ax2 = plt.subplots(figsize=(7, 5))

use_d = good_dim & (psd_median_dim > 0) & np.isfinite(psd_median_dim)
ax2.fill_between(k_dim_common[use_d], psd_p16_dim[use_d], psd_p84_dim[use_d],
                 alpha=0.25, color='darkorange', label='16th-84th percentile')
ax2.plot(k_dim_common[use_d], psd_median_dim[use_d], 'o-', color='darkorange',
         markersize=2, linewidth=1.2, label='Median PSD')

# Power-law fit
if beta_dim is not None:
    k_fit_d = k_dim_common[use_d]
    psd_fit_d = 10**(power_law(np.log10(k_fit_d), logA_dim, beta_dim))
    ax2.plot(k_fit_d, psd_fit_d, '--', color='firebrick', linewidth=1.5,
             label=rf'$P(k\xi) \propto (k\xi)^{{-{beta_dim:.2f} \pm {beta_dim_err:.2f}}}$')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Dimensionless frequency $k \cdot \xi$', fontsize=12)
ax2.set_ylabel(r'Lomb-Scargle Power $P(k\xi)$', fontsize=12)
ax2.set_title(r'Stacked PSD — Dimensionless Coordinates ($\xi = \sqrt{GM/g^\dagger}$)', fontsize=13)
ax2.legend(fontsize=9, loc='upper right')

n_range_d = f"{int(n_contrib_dim[use_d].min())}-{int(n_contrib_dim[use_d].max())}"
text_dim = f'{n_total} galaxies\n{n_range_d} per bin'
if not np.isnan(width_dim) and not np.isnan(width_phys):
    text_dim += f'\nEnvelope: {width_dim:.3f} dex'
    text_dim += f'\n(Physical: {width_phys:.3f} dex)'
ax2.text(0.03, 0.05, text_dim,
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

fig2.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'psd_stacked_dimensionless.png')
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig2_path}")
plt.close(fig2)


# --- Figure 3: Periodic vs Non-Periodic comparison ---
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: stacked PSDs
use_p = good_per & (psd_med_per > 0) & np.isfinite(psd_med_per)
use_np = good_nonper & (psd_med_nonper > 0) & np.isfinite(psd_med_nonper)

if np.sum(use_p) > 3:
    ax3a.fill_between(k_common[use_p], psd_p16_per[use_p], psd_p84_per[use_p],
                      alpha=0.2, color='crimson')
    ax3a.plot(k_common[use_p], psd_med_per[use_p], 'o-', color='crimson',
              markersize=2.5, linewidth=1.2, label=f'Periodic (N={n_periodic})')
    if beta_per is not None:
        k_fit_p = k_common[use_p]
        psd_fit_p = 10**(power_law(np.log10(k_fit_p), logA_per, beta_per))
        ax3a.plot(k_fit_p, psd_fit_p, '--', color='crimson', linewidth=1.0, alpha=0.7,
                  label=rf'$\beta = {beta_per:.2f} \pm {beta_per_err:.2f}$')

if np.sum(use_np) > 3:
    ax3a.fill_between(k_common[use_np], psd_p16_nonper[use_np], psd_p84_nonper[use_np],
                      alpha=0.2, color='royalblue')
    ax3a.plot(k_common[use_np], psd_med_nonper[use_np], 'o-', color='royalblue',
              markersize=2.5, linewidth=1.2, label=f'Non-periodic (N={n_nonperiodic})')
    if beta_nonper is not None:
        k_fit_np = k_common[use_np]
        psd_fit_np = 10**(power_law(np.log10(k_fit_np), logA_nonper, beta_nonper))
        ax3a.plot(k_fit_np, psd_fit_np, '--', color='royalblue', linewidth=1.0, alpha=0.7,
                  label=rf'$\beta = {beta_nonper:.2f} \pm {beta_nonper_err:.2f}$')

ax3a.set_xscale('log')
ax3a.set_yscale('log')
ax3a.set_xlabel(r'Spatial frequency $k$ [kpc$^{-1}$]', fontsize=12)
ax3a.set_ylabel(r'Lomb-Scargle Power $P(k)$', fontsize=12)
ax3a.set_title('Stacked PSD: Periodic vs Non-Periodic', fontsize=13)
ax3a.legend(fontsize=9, loc='upper right')

# Right panel: ratio of periodic to non-periodic (looking for bump)
both_valid = use_p & use_np & (psd_med_nonper > 0)
if np.sum(both_valid) > 5:
    ratio_psd = psd_med_per[both_valid] / psd_med_nonper[both_valid]
    ax3b.plot(k_common[both_valid], ratio_psd, 'o-', color='purple',
              markersize=3, linewidth=1.2)
    ax3b.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Highlight bump region if detected
    if k_bump is not None and max_excess is not None and max_excess > 0.02:
        ax3b.axvline(k_bump, color='firebrick', linestyle=':', linewidth=1.0, alpha=0.7,
                     label=rf'Bump at $\lambda = {wl_bump:.1f}$ kpc')
        ax3b.legend(fontsize=9)

    ax3b.set_xscale('log')
    ax3b.set_xlabel(r'Spatial frequency $k$ [kpc$^{-1}$]', fontsize=12)
    ax3b.set_ylabel(r'$P_{\rm periodic}(k) / P_{\rm non-periodic}(k)$', fontsize=12)
    ax3b.set_title('PSD Ratio (Periodic / Non-Periodic)', fontsize=13)
else:
    ax3b.text(0.5, 0.5, 'Insufficient overlap\nfor ratio',
              transform=ax3b.transAxes, ha='center', va='center', fontsize=14)

fig3.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, 'psd_periodic_vs_nonperiodic.png')
fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig3_path}")
plt.close(fig3)


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY — TEST B: POWER SPECTRUM SHAPE")
print("=" * 72)

print(f"\n  Sample: {n_total} SPARC galaxies (>= {MIN_POINTS} points)")
print(f"  Periodic (p<0.05): {n_periodic}, Non-periodic: {n_nonperiodic}")

print(f"\n  STACKED PSD — PHYSICAL COORDINATES:")
if beta_phys is not None:
    print(f"    Power-law index: beta = {beta_phys:.3f} +/- {beta_phys_err:.3f}")
    print(f"    Reduced chi2: {chi2_phys:.3f}")
    for model, beta_pred in [("Kolmogorov (5/3)", 5/3), ("Burgers (2)", 2.0)]:
        z = abs(beta_phys - beta_pred) / beta_phys_err if beta_phys_err > 0 else np.inf
        print(f"    vs {model}: |z| = {z:.1f} ({'consistent' if z < 2 else 'rejected'} at 2σ)")

print(f"\n  STACKED PSD — DIMENSIONLESS COORDINATES (k*ξ):")
if beta_dim is not None:
    print(f"    Power-law index: beta = {beta_dim:.3f} +/- {beta_dim_err:.3f}")
    print(f"    Reduced chi2: {chi2_dim:.3f}")

print(f"\n  COLLAPSE QUALITY:")
print(f"    Physical envelope width: {width_phys:.4f} dex")
print(f"    Dimensionless envelope width: {width_dim:.4f} dex")
if ratio is not None:
    print(f"    Ratio (dim/phys): {ratio:.3f}")
    if p_tighter_dim is not None:
        print(f"    Bootstrap P(dim tighter): {p_tighter_dim:.3f}")

print(f"\n  PERIODIC vs NON-PERIODIC:")
if beta_per is not None and beta_nonper is not None:
    print(f"    Periodic beta: {beta_per:.3f} +/- {beta_per_err:.3f}")
    print(f"    Non-periodic beta: {beta_nonper:.3f} +/- {beta_nonper_err:.3f}")
    print(f"    Delta(beta): {beta_per - beta_nonper:.3f}")
if max_excess is not None:
    print(f"    Periodic spectral bump: {max_excess:.3f} dex excess at lambda={wl_bump:.1f} kpc")
    if max_excess > 0.05:
        print(f"    => BUMP DETECTED above power-law baseline")
    else:
        print(f"    => No significant bump (< 0.05 dex)")

# Interpretation
print(f"\n  INTERPRETATION:")
if beta_phys is not None:
    if 1.0 < beta_phys < 3.0:
        print(f"    Power-law PSD with beta ~ {beta_phys:.1f} => TURBULENT-like spatial structure")
        if abs(beta_phys - 5/3) < 2 * beta_phys_err:
            print(f"    Consistent with Kolmogorov turbulence (incompressible)")
        elif abs(beta_phys - 2.0) < 2 * beta_phys_err:
            print(f"    Consistent with Burgers/compressible turbulence")
        else:
            print(f"    Distinct from standard turbulence models")
    elif beta_phys < 0.5:
        print(f"    Approximately white noise — no significant spatial structure")
    else:
        print(f"    Steep spectrum — possible coherent structure")


# ================================================================
# SAVE RESULTS
# ================================================================
print("\n[8] Saving results...")

results = {
    'test': 'residual_power_spectrum',
    'description': 'Full PSD shape analysis of RAR residuals: turbulence vs oscillation diagnostic.',
    'parameters': {
        'min_points': MIN_POINTS,
        'n_surrogates': N_SURR,
        'detrending': 'UnivariateSpline, s=n*var*0.5',
        'n_freq_common': N_FREQ_COMMON,
        'k_range_physical': [float(k_common[0]), float(k_common[-1])],
        'k_range_dimensionless': [float(k_dim_common[0]), float(k_dim_common[-1])],
    },
    'sample': {
        'n_galaxies': n_total,
        'n_periodic': n_periodic,
        'n_nonperiodic': n_nonperiodic,
        'healing_length_median_kpc': round(float(np.median(xi_arr)), 2),
        'healing_length_range_kpc': [round(float(np.min(xi_arr)), 2), round(float(np.max(xi_arr)), 2)],
    },
    'stacked_psd_physical': {
        'beta': round(float(beta_phys), 4) if beta_phys is not None else None,
        'beta_err': round(float(beta_phys_err), 4) if beta_phys_err is not None else None,
        'log_A': round(float(logA_phys), 4) if logA_phys is not None else None,
        'chi2_reduced': round(float(chi2_phys), 4) if chi2_phys is not None else None,
        'envelope_width_dex': round(float(width_phys), 4) if not np.isnan(width_phys) else None,
    },
    'stacked_psd_dimensionless': {
        'beta': round(float(beta_dim), 4) if beta_dim is not None else None,
        'beta_err': round(float(beta_dim_err), 4) if beta_dim_err is not None else None,
        'log_A': round(float(logA_dim), 4) if logA_dim is not None else None,
        'chi2_reduced': round(float(chi2_dim), 4) if chi2_dim is not None else None,
        'envelope_width_dex': round(float(width_dim), 4) if not np.isnan(width_dim) else None,
    },
    'collapse_comparison': {
        'envelope_ratio_dim_over_phys': round(float(ratio), 4) if ratio is not None else None,
        'better_collapse': 'dimensionless' if (ratio is not None and ratio < 1) else 'physical' if ratio is not None else None,
        'bootstrap_p_dim_tighter': round(float(p_tighter_dim), 4) if p_tighter_dim is not None else None,
    },
    'periodic_vs_nonperiodic': {
        'beta_periodic': round(float(beta_per), 4) if beta_per is not None else None,
        'beta_periodic_err': round(float(beta_per_err), 4) if beta_per_err is not None else None,
        'beta_nonperiodic': round(float(beta_nonper), 4) if beta_nonper is not None else None,
        'beta_nonperiodic_err': round(float(beta_nonper_err), 4) if beta_nonper_err is not None else None,
        'delta_beta': round(float(beta_per - beta_nonper), 4) if (beta_per is not None and beta_nonper is not None) else None,
        'spectral_bump_excess_dex': round(float(max_excess), 4) if max_excess is not None else None,
        'spectral_bump_wavelength_kpc': round(float(wl_bump), 2) if wl_bump is not None else None,
        'bump_detected': bool(max_excess is not None and max_excess > 0.05),
    },
    'theoretical_comparison': {
        'kolmogorov_beta': round(5/3, 4),
        'burgers_beta': 2.0,
        'self_gravitating_range': [1.5, 2.5],
        'measured_beta': round(float(beta_phys), 4) if beta_phys is not None else None,
        'consistent_with_kolmogorov': bool(beta_phys is not None and abs(beta_phys - 5/3) < 2 * beta_phys_err),
        'consistent_with_burgers': bool(beta_phys is not None and abs(beta_phys - 2.0) < 2 * beta_phys_err),
    },
    'figures': [
        'psd_stacked_physical.png',
        'psd_stacked_dimensionless.png',
        'psd_periodic_vs_nonperiodic.png',
    ],
    'per_galaxy': [
        {
            'name': g['name'],
            'n_pts': g['n_pts'],
            'R_extent_kpc': round(g['R_extent'], 2),
            'xi_kpc': round(g['xi_kpc'], 2),
            'Vflat': g['Vflat'],
            'f_peak': round(g['f_peak'], 4),
            'wl_kpc': round(g['wl_kpc'], 2),
            'power_peak': round(g['power_peak'], 4),
            'perm_p': round(g['perm_p'], 4),
            'is_periodic': g['is_periodic'],
        }
        for g in psd_per_galaxy
    ],
}

outpath = os.path.join(RESULTS_DIR, 'summary_residual_power_spectrum.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {outpath}")

print("\n" + "=" * 72)
print("TEST B COMPLETE")
print("=" * 72)
