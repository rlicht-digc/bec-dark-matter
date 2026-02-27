#!/usr/bin/env python3
"""
test_korsaga_ml_sensitivity.py — Korsaga M/L sensitivity test for kurtosis spike
==================================================================================

The kurtosis spike at g† (κ₄ ≈ 24) is internal to Korsaga (§8.27b).
But Korsaga uses B-band M/L from BFM fitting, which is noisier than SPARC's
3.6μm (σ ≈ 0.33 vs 0.17 dex). Of 100 galaxies, 34 hit the M/L boundary
at 0.10 Msun/Lsun — ill-conditioned fits.

Key question: does the kurtosis spike track M/L quality, or persist under
perturbation? If it persists → physical. If it collapses → systematic.

Tests:
  1. Baseline: Korsaga with BFM M/L (original)
  2. Color-based M/L: use fML (from B-V color) instead of BFM
  3. M/L perturbation: inject ±0.1, ±0.2, ±0.3 dex noise into M/L
  4. Boundary galaxies excluded: remove the 34 galaxies at M/L = 0.10
  5. Jackknife: leave-one-galaxy-out kurtosis stability
  6. M/L scaling: systematically shift M/L by ×0.5, ×0.8, ×1.2, ×2.0
  7. Summary: tabulate spike amplitude vs M/L treatment

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
VIZIER_DIR = os.path.join(PROJECT_ROOT, 'data', 'vizier_catalogs')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from load_extended_rar import load_all, build_rar, compute_rar_point
from match_korsaga_massmodels import (
    load_korsaga_tablea1, load_korsaga_tablea2,
    load_ghasp_rotation_curves, freeman_vdisk, sersic_vbulge,
    _normalize_name, _normalize_ugc,
)
from analysis_tools import (
    g_dagger, LOG_G_DAGGER,
    rar_function, rar_residuals,
    binned_stats, get_at_gdagger,
    numerical_derivative, find_zero_crossings,
)

np.random.seed(42)
N_FINE = 15


# ================================================================
# HELPER: Build Korsaga RAR with modified M/L
# ================================================================

def build_korsaga_rar_modified(ml_mode='bfm', ml_noise=0.0, ml_scale=1.0,
                                exclude_boundary=False, rng=None):
    """Reconstruct Korsaga RAR with modified mass-to-light ratios.

    Args:
        ml_mode: 'bfm' (default BFM fits), 'fml' (color-based M/L)
        ml_noise: log-normal noise to add to M/L (dex)
        ml_scale: multiplicative scale factor for M/L
        exclude_boundary: if True, skip galaxies where BFM M/L = 0.10
        rng: numpy RandomState for reproducibility

    Returns:
        (log_gbar, log_gobs, names) arrays for RAR
    """
    tablea1 = load_korsaga_tablea1()
    tablea2 = load_korsaga_tablea2()
    ghasp_rcs = load_ghasp_rotation_curves()

    matched = set(tablea1.keys()) & set(tablea2.keys()) & set(ghasp_rcs.keys())

    all_gbar, all_gobs, all_names = [], [], []

    for ugc in sorted(matched):
        a1 = tablea1[ugc]
        a2 = tablea2[ugc]
        rc = ghasp_rcs[ugc]

        # Quality cut: skip f_ID=3 and non-param galaxies
        if a1['f_ID'] >= 3 or a1['param'] != '*':
            continue

        # Get M/L
        if ml_mode == 'fml':
            ML_disk = a2.get('ML_fML')
            ML_bulge = a2.get('ML_fML')
        else:
            ML_disk = a2.get('ML_disk')
            ML_bulge = a2.get('ML_bulge')

        if ML_disk is None or ML_disk <= 0:
            continue

        # Exclude boundary galaxies (BFM M/L at lower limit)
        if exclude_boundary and a2.get('ML_disk') is not None:
            if abs(a2['ML_disk'] - 0.10) < 0.005:
                continue

        # Apply M/L modifications
        ML_disk_mod = ML_disk * ml_scale
        ML_bulge_mod = ML_bulge * ml_scale if ML_bulge else None

        if ml_noise > 0 and rng is not None:
            noise_factor = 10**(rng.normal(0, ml_noise))
            ML_disk_mod *= noise_factor
            if ML_bulge_mod:
                noise_factor_b = 10**(rng.normal(0, ml_noise))
                ML_bulge_mod *= noise_factor_b

        # Compute disk mass and Vdisk
        h_kpc = a1['h_kpc']
        LD = a1['LD_1e8Lsun']
        Mdisk = ML_disk_mod * LD * 1e8
        R_kpc = rc['R_kpc']
        Vdisk = freeman_vdisk(R_kpc, h_kpc, Mdisk)

        # Bulge
        re_kpc = a1['re_kpc']
        n_sersic = a1['n_sersic']
        LB = a1['LB_1e8Lsun']

        if (LB > 0 and re_kpc > 0 and n_sersic > 0
                and ML_bulge_mod is not None and ML_bulge_mod > 0):
            Mbulge = ML_bulge_mod * LB * 1e8
            Vbul = sersic_vbulge(R_kpc, re_kpc, n_sersic, Mbulge)
        else:
            Vbul = np.zeros_like(R_kpc)

        Vgas = np.zeros_like(R_kpc)
        mask = R_kpc > 0
        gobs, gbar = compute_rar_point(
            rc['Vobs'][mask], Vgas[mask], Vdisk[mask], Vbul[mask], R_kpc[mask])

        valid = (gobs > 0) & (gbar > 0)
        if np.sum(valid) >= 5:
            all_gbar.extend(np.log10(gbar[valid]))
            all_gobs.extend(np.log10(gobs[valid]))
            all_names.extend([ugc] * np.sum(valid))

    return np.array(all_gbar), np.array(all_gobs), np.array(all_names)


def kurtosis_at_gdagger(log_gbar, log_gobs, n_bins=N_FINE):
    """Get kurtosis in the g†-containing bin."""
    stats = binned_stats(log_gbar, log_gobs, n_bins=n_bins)
    b = get_at_gdagger(stats, key='kurtosis', tol=0.15)
    if b is not None:
        return b['kurtosis'], b['n'], b['sigma']
    return np.nan, 0, np.nan


def kurtosis_derivative_peak(log_gbar, log_gobs, n_bins=N_FINE):
    """Find kurtosis derivative peak nearest to g†."""
    stats = binned_stats(log_gbar, log_gobs, n_bins=n_bins)
    valid = [(s['center'], s['kurtosis']) for s in stats
             if not np.isnan(s['kurtosis'])]
    if len(valid) < 4:
        return np.nan
    centers = np.array([v[0] for v in valid])
    kurts = np.array([v[1] for v in valid])
    dk = numerical_derivative(centers, kurts)
    peaks = find_zero_crossings(centers, dk, direction='pos_to_neg')
    if not peaks:
        return np.nan
    return min(peaks, key=lambda x: abs(x - LOG_G_DAGGER))


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("KORSAGA M/L SENSITIVITY TEST FOR KURTOSIS SPIKE")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")

results = {
    'test_name': 'korsaga_ml_sensitivity',
    'log_g_dagger': LOG_G_DAGGER,
    'treatments': {},
}


# ================================================================
# 1. BASELINE: BFM M/L (original)
# ================================================================
print("\n" + "=" * 72)
print("[1] Baseline: BFM M/L (original Korsaga)")
print("=" * 72)

gbar_bfm, gobs_bfm, names_bfm = build_korsaga_rar_modified(ml_mode='bfm')
k_bfm, n_bfm, s_bfm = kurtosis_at_gdagger(gbar_bfm, gobs_bfm)
peak_bfm = kurtosis_derivative_peak(gbar_bfm, gobs_bfm)
n_gal_bfm = len(set(names_bfm))

print(f"  Galaxies: {n_gal_bfm}, RAR points: {len(gbar_bfm)}")
print(f"  κ₄ at g†: {k_bfm:+.2f} (N={n_bfm}, σ={s_bfm:.4f})")
print(f"  κ₄ derivative peak: {peak_bfm:.3f} (Δ = {peak_bfm - LOG_G_DAGGER:+.3f} dex)")

results['treatments']['BFM_baseline'] = {
    'kurtosis': round(float(k_bfm), 3),
    'n_points': n_bfm,
    'sigma': round(float(s_bfm), 5),
    'derivative_peak': round(float(peak_bfm), 3) if not np.isnan(peak_bfm) else None,
    'n_galaxies': n_gal_bfm,
    'n_rar_points': len(gbar_bfm),
}


# ================================================================
# 2. COLOR-BASED M/L (fML)
# ================================================================
print("\n" + "=" * 72)
print("[2] Color-based M/L (fML from B-V)")
print("=" * 72)

gbar_fml, gobs_fml, names_fml = build_korsaga_rar_modified(ml_mode='fml')
k_fml, n_fml, s_fml = kurtosis_at_gdagger(gbar_fml, gobs_fml)
peak_fml = kurtosis_derivative_peak(gbar_fml, gobs_fml)
n_gal_fml = len(set(names_fml))

print(f"  Galaxies: {n_gal_fml}, RAR points: {len(gbar_fml)}")
print(f"  κ₄ at g†: {k_fml:+.2f} (N={n_fml}, σ={s_fml:.4f})")
print(f"  κ₄ derivative peak: {peak_fml:.3f} (Δ = {peak_fml - LOG_G_DAGGER:+.3f} dex)")
print(f"  Δκ₄ from BFM: {k_fml - k_bfm:+.2f}")

results['treatments']['fML_color'] = {
    'kurtosis': round(float(k_fml), 3),
    'n_points': n_fml,
    'sigma': round(float(s_fml), 5),
    'derivative_peak': round(float(peak_fml), 3) if not np.isnan(peak_fml) else None,
    'n_galaxies': n_gal_fml,
    'n_rar_points': len(gbar_fml),
    'delta_kurtosis_from_bfm': round(float(k_fml - k_bfm), 3),
}


# ================================================================
# 3. M/L PERTURBATION TESTS
# ================================================================
print("\n" + "=" * 72)
print("[3] M/L perturbation tests (inject log-normal noise)")
print("=" * 72)

noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
n_mc = 50  # Monte Carlo iterations per noise level

print(f"\n  {'Noise (dex)':>12s} {'κ₄ mean':>9s} {'κ₄ std':>8s} {'κ₄ range':>18s} "
      f"{'peak mean':>10s} {'peak std':>9s}")
print(f"  {'-'*70}")

perturbation_results = {}
for noise in noise_levels:
    kurtoses = []
    peaks = []
    for mc in range(n_mc):
        rng_mc = np.random.RandomState(42 + mc)
        gb, go, nm = build_korsaga_rar_modified(
            ml_mode='bfm', ml_noise=noise, rng=rng_mc)
        k, n, s = kurtosis_at_gdagger(gb, go)
        p = kurtosis_derivative_peak(gb, go)
        if not np.isnan(k):
            kurtoses.append(k)
        if not np.isnan(p):
            peaks.append(p)

    k_arr = np.array(kurtoses)
    p_arr = np.array(peaks)

    k_mean = np.mean(k_arr) if len(k_arr) > 0 else np.nan
    k_std = np.std(k_arr) if len(k_arr) > 0 else np.nan
    k_min = np.min(k_arr) if len(k_arr) > 0 else np.nan
    k_max = np.max(k_arr) if len(k_arr) > 0 else np.nan
    p_mean = np.mean(p_arr) if len(p_arr) > 0 else np.nan
    p_std = np.std(p_arr) if len(p_arr) > 0 else np.nan

    range_str = f"[{k_min:+.1f}, {k_max:+.1f}]"
    print(f"  {noise:12.2f} {k_mean:+9.2f} {k_std:8.2f} {range_str:>18s} "
          f"{p_mean:10.3f} {p_std:9.3f}")

    perturbation_results[f'noise_{noise:.2f}'] = {
        'noise_dex': noise,
        'n_mc': n_mc,
        'kurtosis_mean': round(float(k_mean), 3),
        'kurtosis_std': round(float(k_std), 3),
        'kurtosis_min': round(float(k_min), 3),
        'kurtosis_max': round(float(k_max), 3),
        'peak_mean': round(float(p_mean), 3) if not np.isnan(p_mean) else None,
        'peak_std': round(float(p_std), 3) if not np.isnan(p_std) else None,
        'frac_kurtosis_positive': round(float(np.mean(k_arr > 0)), 3) if len(k_arr) else 0,
        'frac_kurtosis_gt5': round(float(np.mean(k_arr > 5)), 3) if len(k_arr) else 0,
    }

results['treatments']['perturbation'] = perturbation_results


# ================================================================
# 4. EXCLUDE BOUNDARY GALAXIES (M/L = 0.10)
# ================================================================
print("\n" + "=" * 72)
print("[4] Exclude boundary galaxies (BFM M/L = 0.10)")
print("=" * 72)

gbar_excl, gobs_excl, names_excl = build_korsaga_rar_modified(
    ml_mode='bfm', exclude_boundary=True)
k_excl, n_excl, s_excl = kurtosis_at_gdagger(gbar_excl, gobs_excl)
peak_excl = kurtosis_derivative_peak(gbar_excl, gobs_excl)
n_gal_excl = len(set(names_excl))

# Count how many were excluded
tablea2 = load_korsaga_tablea2()
n_boundary = sum(1 for v in tablea2.values()
                 if v.get('ML_disk') is not None and abs(v['ML_disk'] - 0.10) < 0.005)

print(f"  Boundary galaxies excluded: {n_gal_bfm - n_gal_excl}")
print(f"  Remaining: {n_gal_excl} galaxies, {len(gbar_excl)} points")
print(f"  κ₄ at g†: {k_excl:+.2f} (N={n_excl}, σ={s_excl:.4f})")
print(f"  κ₄ derivative peak: {peak_excl:.3f} (Δ = {peak_excl - LOG_G_DAGGER:+.3f} dex)")
print(f"  Δκ₄ from BFM baseline: {k_excl - k_bfm:+.2f}")

results['treatments']['exclude_boundary'] = {
    'kurtosis': round(float(k_excl), 3),
    'n_points': n_excl,
    'sigma': round(float(s_excl), 5),
    'derivative_peak': round(float(peak_excl), 3) if not np.isnan(peak_excl) else None,
    'n_galaxies': n_gal_excl,
    'n_rar_points': len(gbar_excl),
    'n_excluded': n_gal_bfm - n_gal_excl,
    'delta_kurtosis_from_bfm': round(float(k_excl - k_bfm), 3),
}


# ================================================================
# 5. JACKKNIFE: leave-one-galaxy-out kurtosis stability
# ================================================================
print("\n" + "=" * 72)
print("[5] Jackknife: leave-one-galaxy-out kurtosis stability")
print("=" * 72)

unique_galaxies = sorted(set(names_bfm))
n_jk = len(unique_galaxies)
jk_kurtoses = []
jk_peaks = []
jk_names = []

for gal in unique_galaxies:
    mask = names_bfm != gal
    gb_jk = gbar_bfm[mask]
    go_jk = gobs_bfm[mask]
    k_jk, n_jk_pts, s_jk = kurtosis_at_gdagger(gb_jk, go_jk)
    p_jk = kurtosis_derivative_peak(gb_jk, go_jk)
    jk_kurtoses.append(k_jk)
    jk_peaks.append(p_jk)
    jk_names.append(gal)

jk_k = np.array(jk_kurtoses)
jk_p = np.array(jk_peaks)
jk_valid = ~np.isnan(jk_k)
jk_p_valid = ~np.isnan(jk_p)

print(f"  Galaxies tested: {len(unique_galaxies)}")
print(f"  Baseline κ₄: {k_bfm:+.2f}")
print(f"  JK mean κ₄:  {np.mean(jk_k[jk_valid]):+.2f} ± {np.std(jk_k[jk_valid]):.2f}")
print(f"  JK range:     [{np.min(jk_k[jk_valid]):+.1f}, {np.max(jk_k[jk_valid]):+.1f}]")
print(f"  Max shift:    {np.max(np.abs(jk_k[jk_valid] - k_bfm)):.2f}")

# Identify most influential galaxies
influences = np.abs(jk_k[jk_valid] - k_bfm)
top_idx = np.argsort(influences)[-5:][::-1]
valid_names = np.array(jk_names)[jk_valid]
print(f"\n  Most influential galaxies (largest |Δκ₄| when removed):")
for idx in top_idx:
    print(f"    {valid_names[idx]:12s}: κ₄ = {jk_k[jk_valid][idx]:+.2f} "
          f"(Δ = {jk_k[jk_valid][idx] - k_bfm:+.2f})")

if np.sum(jk_p_valid) > 0:
    print(f"\n  JK derivative peak: {np.mean(jk_p[jk_p_valid]):.3f} "
          f"± {np.std(jk_p[jk_p_valid]):.3f}")

results['treatments']['jackknife'] = {
    'n_galaxies': len(unique_galaxies),
    'baseline_kurtosis': round(float(k_bfm), 3),
    'jk_mean': round(float(np.mean(jk_k[jk_valid])), 3),
    'jk_std': round(float(np.std(jk_k[jk_valid])), 3),
    'jk_min': round(float(np.min(jk_k[jk_valid])), 3),
    'jk_max': round(float(np.max(jk_k[jk_valid])), 3),
    'max_shift': round(float(np.max(np.abs(jk_k[jk_valid] - k_bfm))), 3),
    'frac_positive': round(float(np.mean(jk_k[jk_valid] > 0)), 3),
    'most_influential': [
        {'name': valid_names[idx], 'kurtosis_when_removed': round(float(jk_k[jk_valid][idx]), 3)}
        for idx in top_idx
    ],
}


# ================================================================
# 6. SYSTEMATIC M/L SCALING
# ================================================================
print("\n" + "=" * 72)
print("[6] Systematic M/L scaling")
print("=" * 72)

scale_factors = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
print(f"\n  {'Scale':>8s} {'κ₄':>8s} {'N':>6s} {'σ':>8s} {'peak':>8s}")
print(f"  {'-'*44}")

scaling_results = {}
for sf in scale_factors:
    gb_sf, go_sf, nm_sf = build_korsaga_rar_modified(ml_mode='bfm', ml_scale=sf)
    k_sf, n_sf, s_sf = kurtosis_at_gdagger(gb_sf, go_sf)
    p_sf = kurtosis_derivative_peak(gb_sf, go_sf)

    p_str = f"{p_sf:.3f}" if not np.isnan(p_sf) else "  ---"
    marker = " ← baseline" if sf == 1.0 else ""
    print(f"  {sf:8.1f} {k_sf:+8.2f} {n_sf:6d} {s_sf:8.4f} {p_str:>8s}{marker}")

    scaling_results[f'scale_{sf:.1f}'] = {
        'scale_factor': sf,
        'kurtosis': round(float(k_sf), 3),
        'n_points': n_sf,
        'sigma': round(float(s_sf), 5),
        'derivative_peak': round(float(p_sf), 3) if not np.isnan(p_sf) else None,
    }

results['treatments']['scaling'] = scaling_results


# ================================================================
# 7. FULL KURTOSIS PROFILES (BFM vs fML vs excluded)
# ================================================================
print("\n" + "=" * 72)
print("[7] Full kurtosis profiles comparison")
print("=" * 72)

configs = [
    ('BFM baseline', gbar_bfm, gobs_bfm),
    ('fML (color)', gbar_fml, gobs_fml),
    ('Excl boundary', gbar_excl, gobs_excl),
]

print(f"\n  {'center':>8s}", end='')
for label, _, _ in configs:
    print(f"  {label:>14s}", end='')
print()
print(f"  {'-'*54}")

profile_data = {}
for label, gb, go in configs:
    profile_data[label] = binned_stats(gb, go)

for i in range(N_FINE):
    center = profile_data['BFM baseline'][i]['center']
    line = f"  {center:8.2f}"
    for label, _, _ in configs:
        k = profile_data[label][i]['kurtosis']
        if np.isnan(k):
            line += f"  {'---':>14s}"
        else:
            line += f"  {k:+14.2f}"
    marker = " ← g†" if abs(center - LOG_G_DAGGER) < 0.15 else ""
    print(line + marker)


# ================================================================
# 8. SUMMARY AND VERDICT
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"\n  Treatment                  κ₄ at g†   Δκ₄ from BFM   Peak location")
print(f"  {'-'*72}")

treatments_summary = [
    ('BFM baseline', k_bfm, 0, peak_bfm),
    ('fML (color)', k_fml, k_fml - k_bfm, peak_fml),
    ('Excl boundary', k_excl, k_excl - k_bfm, peak_excl),
]
for label, k, dk, p in treatments_summary:
    p_str = f"{p:.3f}" if not np.isnan(p) else "  ---"
    print(f"  {label:28s} {k:+8.2f}   {dk:+12.2f}   {p_str}")

# Perturbation summary
print(f"\n  Perturbation (50 MC each):")
for noise in noise_levels:
    key = f'noise_{noise:.2f}'
    pr = perturbation_results[key]
    print(f"    ±{noise:.2f} dex: κ₄ = {pr['kurtosis_mean']:+.1f} ± {pr['kurtosis_std']:.1f}, "
          f"P(κ₄ > 5) = {pr['frac_kurtosis_gt5']:.0%}")

# Verdict
spike_persists = (k_fml > 5 and k_excl > 5 and
                  perturbation_results['noise_0.20']['frac_kurtosis_gt5'] > 0.5)

if spike_persists:
    verdict = ("ROBUST: Kurtosis spike persists under all M/L treatments. "
               "Survives fML color M/L, boundary exclusion, and 0.20 dex perturbation. "
               "Consistent with physical phase transition origin.")
else:
    # Check if spike collapses
    if k_fml < 3 or k_excl < 3:
        verdict = ("FRAGILE: Kurtosis spike collapses under alternative M/L treatment. "
                   "Likely reflects B-band mass model systematics, not physical phase transition.")
    else:
        verdict = ("MIXED: Kurtosis spike is sensitive to M/L treatment but does not fully "
                   "collapse. Physical origin possible but cannot exclude systematic contribution.")

print(f"\n  VERDICT: {verdict}")
results['verdict'] = verdict

# Save
outpath = os.path.join(RESULTS_DIR, 'summary_korsaga_ml_sensitivity.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved: {outpath}")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
