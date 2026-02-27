#!/usr/bin/env python3
"""
Interface Spectral Test (Step 27) — Lomb-Scargle on Detrended RAR Residuals
============================================================================

A) Per-galaxy Lomb-Scargle periodogram on spline-detrended residuals
   - frequency grid: 1 to N/2 cycles across radial extent
   - peak power + FAP via radius-order permutation surrogates (200/gal for speed)

B) Cross-galaxy shared wavelength test
   - convert f_peak to physical wavelength (kpc)
   - compare clustering to null from surrogates

C) Scaling relations: wavelength vs Vflat, med_log_gbar

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

g_dagger = 1.20e-10
kpc_m = 3.086e19
MIN_POINTS = 15
N_SURR = 200  # surrogates per galaxy (fast pass)

np.random.seed(42)

print("=" * 72)
print("INTERFACE SPECTRAL TEST (Step 27)")
print("  Lomb-Scargle on Detrended RAR Residuals")
print("=" * 72)
print(f"  Min points: {MIN_POINTS}, Surrogates/galaxy: {N_SURR}")


# ================================================================
# ENVIRONMENT
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
    return 'dense' if name in UMA_GALAXIES or name in GROUP_MEMBERS else 'field'


def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


# ================================================================
# 1. LOAD + COMPUTE DETRENDED RESIDUALS
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

# Build per-galaxy data with detrended residuals
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

    # Spline detrending (same as Step 26)
    n = len(residuals)
    var_eps = np.var(residuals)
    s_param = n * var_eps * 0.5
    try:
        spline = UnivariateSpline(R_sorted, residuals, k=min(3, n - 1), s=s_param)
        eps_det = residuals - spline(R_sorted)
    except Exception:
        eps_det = residuals - np.mean(residuals)

    med_log_gbar = float(np.median(log_gbar))

    galaxy_data.append({
        'name': name,
        'R': R_sorted,            # kpc
        'eps_det': eps_det,
        'n_pts': n,
        'env': classify_env(name),
        'Vflat': prop['Vflat'],
        'med_log_gbar': med_log_gbar,
    })

n_galaxies = len(galaxy_data)
print(f"  {n_galaxies} galaxies with N >= {MIN_POINTS}")


# ================================================================
# 2. LOMB-SCARGLE PER GALAXY + PERMUTATION NULL
# ================================================================
print(f"\n[2] Lomb-Scargle periodogram + {N_SURR} surrogates per galaxy...")

perm_rng = np.random.default_rng(789)

spectral_results = []

for gi, g in enumerate(galaxy_data):
    R = g['R']
    eps = g['eps_det']
    n = g['n_pts']

    # Standardize
    std_eps = np.std(eps)
    if std_eps < 1e-30:
        continue
    y = (eps - np.mean(eps)) / std_eps

    # Radial extent
    R_extent = R[-1] - R[0]
    if R_extent <= 0:
        continue

    # Frequency grid: 1 cycle to N/2 cycles across extent
    # angular frequency = 2*pi*f, but LS uses f directly
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

    # Wavelength in kpc
    wl_kpc = 1.0 / f_peak

    # Null: shuffle residuals, compute peak power
    null_peaks = np.zeros(N_SURR)
    for s in range(N_SURR):
        y_shuf = perm_rng.permutation(y)
        ls_null = LombScargle(R, y_shuf, fit_mean=False, center_data=True)
        p_null = ls_null.power(freq_grid)
        null_peaks[s] = np.max(p_null)

    p_val = float(np.mean(null_peaks >= power_peak))
    p_val_adj = max(p_val, 1.0 / (N_SURR + 1))

    spectral_results.append({
        'name': g['name'],
        'n_pts': n,
        'R_extent_kpc': round(float(R_extent), 2),
        'f_peak': round(f_peak, 4),
        'wl_kpc': round(wl_kpc, 2),
        'power_peak': round(power_peak, 4),
        'null_mean_peak': round(float(np.mean(null_peaks)), 4),
        'null_std_peak': round(float(np.std(null_peaks)), 4),
        'perm_p': round(p_val, 4),
        'env': g['env'],
        'Vflat': g['Vflat'],
        'med_log_gbar': g['med_log_gbar'],
    })

    if (gi + 1) % 20 == 0:
        print(f"    {gi+1}/{n_galaxies} done...")

n_spectral = len(spectral_results)
print(f"  Completed {n_spectral} galaxies")


# ================================================================
# 3. AGGREGATE SPECTRAL RESULTS
# ================================================================
print("\n[3] Aggregate spectral results...")

n_sig = sum(1 for g in spectral_results if g['perm_p'] < 0.05)
frac_sig = n_sig / n_spectral
# Expected under null: 5%
expected_sig = 0.05 * n_spectral
binom_p_excess = float(stats.binomtest(n_sig, n_spectral, 0.05, alternative='greater').pvalue)

print(f"  Galaxies with significant peak (p<0.05): {n_sig}/{n_spectral} ({frac_sig:.1%})")
print(f"  Expected under null: {expected_sig:.1f}")
print(f"  Binomial p for excess: {binom_p_excess:.4e}")

# Peak power distribution
power_peaks = np.array([g['power_peak'] for g in spectral_results])
null_means = np.array([g['null_mean_peak'] for g in spectral_results])
print(f"  Observed peak power: mean={np.mean(power_peaks):.3f}, median={np.median(power_peaks):.3f}")
print(f"  Null peak power:     mean={np.mean(null_means):.3f}")


# ================================================================
# 4. CROSS-GALAXY WAVELENGTH CLUSTERING (B)
# ================================================================
print("\n[4] Cross-galaxy wavelength clustering...")

# Only use galaxies with significant peaks
sig_gals = [g for g in spectral_results if g['perm_p'] < 0.05]
all_wl = np.array([g['wl_kpc'] for g in spectral_results])
sig_wl = np.array([g['wl_kpc'] for g in sig_gals])

if len(sig_wl) >= 5:
    # Use log-wavelength for clustering analysis
    log_wl_sig = np.log10(sig_wl)
    log_wl_all = np.log10(all_wl)

    # IQR of significant peaks
    wl_q25, wl_median, wl_q75 = np.percentile(sig_wl, [25, 50, 75])
    log_wl_q25, log_wl_median, log_wl_q75 = np.percentile(log_wl_sig, [25, 50, 75])
    wl_iqr = wl_q75 - wl_q25
    log_wl_iqr = log_wl_q75 - log_wl_q25

    print(f"  Significant peaks (N={len(sig_wl)}):")
    print(f"    Wavelength median: {wl_median:.2f} kpc (IQR: {wl_q25:.2f}–{wl_q75:.2f})")
    print(f"    log(λ) median: {log_wl_median:.2f} (IQR: {log_wl_q25:.2f}–{log_wl_q75:.2f})")

    # All galaxies
    wl_all_q25, wl_all_median, wl_all_q75 = np.percentile(all_wl, [25, 50, 75])
    print(f"  All galaxies (N={len(all_wl)}):")
    print(f"    Wavelength median: {wl_all_median:.2f} kpc (IQR: {wl_all_q25:.2f}–{wl_all_q75:.2f})")

    # Clustering test: is the IQR of significant peaks narrower than expected from null?
    # Null: for each galaxy, draw peak freq uniformly from freq_grid
    clust_rng = np.random.default_rng(321)
    null_iqrs = []
    for _ in range(2000):
        # Sample N_sig wavelengths from the full distribution
        idx = clust_rng.choice(len(all_wl), size=len(sig_wl), replace=True)
        null_iqrs.append(np.percentile(np.log10(all_wl[idx]), 75) - np.percentile(np.log10(all_wl[idx]), 25))
    null_iqrs = np.array(null_iqrs)
    cluster_p = float(np.mean(null_iqrs <= log_wl_iqr))  # is observed IQR unusually narrow?

    print(f"  Clustering test (log-wavelength IQR):")
    print(f"    Observed log-IQR: {log_wl_iqr:.3f}")
    print(f"    Null median log-IQR: {np.median(null_iqrs):.3f}")
    print(f"    p(narrower): {cluster_p:.4f}")
else:
    log_wl_median = None
    wl_median = None
    wl_q25 = wl_q75 = None
    log_wl_iqr = None
    cluster_p = None
    wl_all_median = float(np.median(all_wl))
    wl_all_q25, wl_all_q75 = np.percentile(all_wl, [25, 75])
    print(f"  Too few significant peaks ({len(sig_wl)}) for clustering test")


# ================================================================
# 5. SCALING RELATIONS (C)
# ================================================================
print("\n[5] Scaling relations...")

wl_arr = np.array([g['wl_kpc'] for g in spectral_results])
vflat_arr = np.array([g['Vflat'] for g in spectral_results])
med_gbar_arr = np.array([g['med_log_gbar'] for g in spectral_results])
r_extent_arr = np.array([g['R_extent_kpc'] for g in spectral_results])

# Also test normalized wavelength: wl / R_extent
wl_norm = wl_arr / r_extent_arr

correlations = {}
for prop_name, prop_arr in [('Vflat', vflat_arr), ('med_log_gbar', med_gbar_arr),
                              ('R_extent', r_extent_arr)]:
    # wavelength vs property
    rho, p = stats.spearmanr(wl_arr, prop_arr)
    rho_norm, p_norm = stats.spearmanr(wl_norm, prop_arr)
    print(f"  λ_peak vs {prop_name:<14}: ρ = {rho:+.3f} (p={p:.4f})")
    print(f"  λ/R_ext vs {prop_name:<13}: ρ = {rho_norm:+.3f} (p={p_norm:.4f})")
    correlations[f'wl_vs_{prop_name}'] = {'rho': round(float(rho), 3), 'p': round(float(p), 4)}
    correlations[f'wl_norm_vs_{prop_name}'] = {'rho': round(float(rho_norm), 3), 'p': round(float(p_norm), 4)}

# Wavelength vs Vflat for sig galaxies only
if len(sig_wl) >= 10:
    sig_vflat = np.array([g['Vflat'] for g in sig_gals])
    rho_s, p_s = stats.spearmanr(sig_wl, sig_vflat)
    print(f"  λ_peak vs Vflat (sig only): ρ = {rho_s:+.3f} (p={p_s:.4f})")
    correlations['wl_vs_Vflat_sig'] = {'rho': round(float(rho_s), 3), 'p': round(float(p_s), 4)}


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"\n  Galaxies analyzed: {n_spectral}")
print(f"  Significant spectral peaks (p<0.05): {n_sig}/{n_spectral} ({frac_sig:.1%})")
print(f"  Expected under null: {expected_sig:.1f} ({0.05:.0%})")
print(f"  Binomial excess p: {binom_p_excess:.4e}")

print(f"\n  All galaxies — peak wavelength:")
print(f"    Median: {wl_all_median:.2f} kpc  (IQR: {wl_all_q25:.2f}–{wl_all_q75:.2f})")

if len(sig_wl) >= 5:
    print(f"  Significant-only — peak wavelength:")
    print(f"    Median: {wl_median:.2f} kpc  (IQR: {wl_q25:.2f}–{wl_q75:.2f})")
    if cluster_p is not None:
        clustered = cluster_p < 0.05
        print(f"  Wavelength clustering: {'YES' if clustered else 'NO'} (p={cluster_p:.4f})")

print(f"\n  Strongest scaling correlations:")
best_corr = max(correlations.items(), key=lambda x: abs(x[1]['rho']))
print(f"    {best_corr[0]}: ρ = {best_corr[1]['rho']:+.3f}, p = {best_corr[1]['p']:.4f}")

# Caution notes
print(f"\n  Notes:")
print(f"    - Using actual galactocentric radii (kpc), not index proxy")
print(f"    - Detrending: UnivariateSpline with s = n*var*0.5")
print(f"    - {N_SURR} surrogates/galaxy (fast pass; increase to 1000+ for paper)")
print(f"    - Freq grid: 1 to N/2 cycles across radial extent")


# ================================================================
# SAVE
# ================================================================
results = {
    'test': 'interface_spectral_test',
    'description': 'Lomb-Scargle periodogram on spline-detrended RAR residuals with permutation null.',
    'parameters': {
        'min_points': MIN_POINTS,
        'n_surrogates': N_SURR,
        'detrending': 'UnivariateSpline, s=n*var*0.5',
        'radii': 'galactocentric_kpc',
    },
    'sample': {
        'n_galaxies': n_spectral,
    },
    'significance': {
        'n_significant_005': n_sig,
        'frac_significant': round(frac_sig, 3),
        'expected_null': round(expected_sig, 1),
        'binom_p_excess': binom_p_excess,
    },
    'peak_power': {
        'mean_observed': round(float(np.mean(power_peaks)), 4),
        'median_observed': round(float(np.median(power_peaks)), 4),
        'mean_null': round(float(np.mean(null_means)), 4),
    },
    'wavelength_all': {
        'median_kpc': round(float(wl_all_median), 2),
        'q25_kpc': round(float(wl_all_q25), 2),
        'q75_kpc': round(float(wl_all_q75), 2),
    },
    'wavelength_significant': {
        'n': len(sig_wl),
        'median_kpc': round(float(wl_median), 2) if wl_median else None,
        'q25_kpc': round(float(wl_q25), 2) if wl_q25 else None,
        'q75_kpc': round(float(wl_q75), 2) if wl_q75 else None,
        'log_iqr': round(float(log_wl_iqr), 3) if log_wl_iqr else None,
        'clustering_p': round(float(cluster_p), 4) if cluster_p is not None else None,
    },
    'correlations': correlations,
    'per_galaxy': [
        {k: v for k, v in g.items()}
        for g in spectral_results
    ],
}

outpath = os.path.join(RESULTS_DIR, 'summary_interface_spectral_test.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {outpath}")
print("=" * 72)
