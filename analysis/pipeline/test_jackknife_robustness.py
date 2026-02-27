#!/usr/bin/env python3
"""
Jackknife Robustness Test for Environmental Signal
====================================================

Two questions:

1. Does any single galaxy drive the low-acceleration environmental result
   (uniform scatter, Δσ ≈ 0 at g < 10^-10.5)?
   If removing one galaxy flips the Levene p from >0.05 to <0.01, that galaxy
   dominates the result.

2. Does any single galaxy drive the dσ/dx=0 inversion point at -9.86?
   If removing one galaxy moves the zero-crossing by >0.3 dex, it's fragile.

Method: leave-one-out jackknife over all 131 quality-cut galaxies.
For each removed galaxy, recompute:
  (a) Low-accel (g < 10^-10.5) scatter by environment, Levene p, Δσ
  (b) Fine-binned scatter derivative, closest zero-crossing to g†
  (c) Full-sample scatter in low-accel bin

Additionally: N-galaxy jackknife (remove 2, 3, 5 at random, 1000× each) to
test fragility under random galaxy subsampling.
"""

import os
import json
import numpy as np
from scipy.stats import levene
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

# Acceleration cut for "condensate regime"
LOW_ACCEL_CUT = -10.5


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


def compute_galaxy_residuals(galaxies, sparc_props):
    """Compute per-galaxy RAR residuals. Returns dict[name] -> {log_gbar, log_res, env}."""
    results = {}
    for name, gdata in galaxies.items():
        if name not in sparc_props:
            continue
        prop = sparc_props[name]
        if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
            continue

        D_ratio = prop['D'] / gdata['dist']
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
            'env': env,
            'n_points': len(log_gbar),
        }
    return results


def compute_low_accel_stats(gal_results, exclude_names=None):
    """Compute low-acceleration environmental scatter stats, optionally excluding galaxies."""
    d_pts = []
    f_pts = []

    for name, r in gal_results.items():
        if exclude_names and name in exclude_names:
            continue
        mask = r['log_gbar'] < LOW_ACCEL_CUT
        if not np.any(mask):
            continue
        pts = r['log_res'][mask]
        if r['env'] == 'dense':
            d_pts.extend(pts)
        else:
            f_pts.extend(pts)

    d_pts = np.array(d_pts)
    f_pts = np.array(f_pts)

    if len(d_pts) < 5 or len(f_pts) < 5:
        return {
            'sigma_dense': np.nan, 'sigma_field': np.nan,
            'delta_sigma': np.nan, 'levene_p': np.nan,
            'n_dense': len(d_pts), 'n_field': len(f_pts),
        }

    sd = float(np.std(d_pts))
    sf = float(np.std(f_pts))
    stat_L, p_L = levene(d_pts, f_pts)

    return {
        'sigma_dense': sd, 'sigma_field': sf,
        'delta_sigma': sf - sd, 'levene_p': float(p_L),
        'n_dense': len(d_pts), 'n_field': len(f_pts),
    }


def compute_inversion_point(gal_results, exclude_names=None):
    """Compute dσ/dx zero-crossing closest to g† from fine-binned derivative."""
    # Collect all points
    all_gbar = []
    all_res = []
    for name, r in gal_results.items():
        if exclude_names and name in exclude_names:
            continue
        all_gbar.extend(r['log_gbar'])
        all_res.extend(r['log_res'])

    all_gbar = np.array(all_gbar)
    all_res = np.array(all_res)

    # Fine bins (same as original analysis)
    n_fine = 15
    edges = np.linspace(-12.5, -8.5, n_fine + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    sigmas = []
    valid_centers = []
    for j in range(n_fine):
        mask = (all_gbar >= edges[j]) & (all_gbar < edges[j+1])
        res_bin = all_res[mask]
        if len(res_bin) < 10:
            continue
        sigmas.append(float(np.std(res_bin)))
        valid_centers.append(centers[j])

    if len(valid_centers) < 4:
        return None

    vc = np.array(valid_centers)
    vs = np.array(sigmas)

    # Derivative
    dy = np.zeros_like(vs)
    dy[0] = (vs[1] - vs[0]) / (vc[1] - vc[0])
    dy[-1] = (vs[-1] - vs[-2]) / (vc[-1] - vc[-2])
    for i in range(1, len(vs) - 1):
        dy[i] = (vs[i+1] - vs[i-1]) / (vc[i+1] - vc[i-1])

    # Zero crossings
    crossings = []
    for i in range(len(dy) - 1):
        if dy[i] * dy[i+1] < 0:
            x_cross = vc[i] - dy[i] * (vc[i+1] - vc[i]) / (dy[i+1] - dy[i])
            crossings.append(float(x_cross))

    # Closest to g†
    closest = None
    closest_dist = np.inf
    for x in crossings:
        d = abs(x - LOG_G_DAGGER)
        if d < closest_dist:
            closest_dist = d
            closest = x

    return closest


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("JACKKNIFE ROBUSTNESS TEST")
print("=" * 72)

galaxies, sparc_props = load_sparc()
gal_results = compute_galaxy_residuals(galaxies, sparc_props)
galaxy_names = sorted(gal_results.keys())
n_gal = len(galaxy_names)

n_dense = sum(1 for r in gal_results.values() if r['env'] == 'dense')
n_field = sum(1 for r in gal_results.values() if r['env'] == 'field')
print(f"  {n_gal} galaxies ({n_dense} dense, {n_field} field)")

# Baseline
baseline_stats = compute_low_accel_stats(gal_results)
baseline_inversion = compute_inversion_point(gal_results)

print(f"\n  BASELINE (full sample):")
print(f"    Low-accel σ_dense = {baseline_stats['sigma_dense']:.4f}  (N={baseline_stats['n_dense']})")
print(f"    Low-accel σ_field = {baseline_stats['sigma_field']:.4f}  (N={baseline_stats['n_field']})")
print(f"    Δσ(field-dense) = {baseline_stats['delta_sigma']:+.4f}")
print(f"    Levene p = {baseline_stats['levene_p']:.6f}")
print(f"    dσ/dx=0 inversion = {baseline_inversion:.3f}  (Δ from g† = {abs(baseline_inversion - LOG_G_DAGGER):.3f})")

# ================================================================
# LEAVE-ONE-OUT JACKKNIFE
# ================================================================
print(f"\n{'='*72}")
print("LEAVE-ONE-OUT JACKKNIFE ({} iterations)".format(n_gal))
print(f"{'='*72}")

jk_results = []

for gname in galaxy_names:
    stats = compute_low_accel_stats(gal_results, exclude_names={gname})
    inv = compute_inversion_point(gal_results, exclude_names={gname})

    env = gal_results[gname]['env']
    n_pts = gal_results[gname]['n_points']
    n_low = int(np.sum(gal_results[gname]['log_gbar'] < LOW_ACCEL_CUT))

    jk_results.append({
        'removed': gname,
        'env': env,
        'n_points': n_pts,
        'n_low_accel_points': n_low,
        'sigma_dense': round(stats['sigma_dense'], 5),
        'sigma_field': round(stats['sigma_field'], 5),
        'delta_sigma': round(stats['delta_sigma'], 5),
        'levene_p': round(stats['levene_p'], 8),
        'inversion_point': round(inv, 4) if inv is not None else None,
        'inv_delta_from_gdagger': round(abs(inv - LOG_G_DAGGER), 4) if inv is not None else None,
    })

# Sort by impact on delta_sigma
jk_by_delta = sorted(jk_results, key=lambda r: abs(r['delta_sigma'] - baseline_stats['delta_sigma']), reverse=True)

print(f"\n  Top 10 most influential galaxies on Δσ(field-dense):")
print(f"  {'Removed':15s} {'env':>5s} {'N_low':>5s} {'Δσ':>8s} {'Shift':>8s} {'Lev_p':>10s} {'Inv pt':>8s} {'Inv Δg†':>8s}")
print(f"  {'-'*75}")

for r in jk_by_delta[:10]:
    shift = r['delta_sigma'] - baseline_stats['delta_sigma']
    inv_str = f"{r['inversion_point']:.3f}" if r['inversion_point'] is not None else "  ---"
    inv_d_str = f"{r['inv_delta_from_gdagger']:.3f}" if r['inv_delta_from_gdagger'] is not None else "  ---"
    print(f"  {r['removed']:15s} {r['env']:>5s} {r['n_low_accel_points']:5d} "
          f"{r['delta_sigma']:+8.4f} {shift:+8.4f} {r['levene_p']:10.6f} {inv_str:>8s} {inv_d_str:>8s}")

# Sort by impact on inversion point
jk_with_inv = [r for r in jk_results if r['inversion_point'] is not None and baseline_inversion is not None]
jk_by_inv = sorted(jk_with_inv, key=lambda r: abs(r['inversion_point'] - baseline_inversion), reverse=True)

print(f"\n  Top 10 most influential galaxies on inversion point:")
print(f"  {'Removed':15s} {'env':>5s} {'N_low':>5s} {'Inv pt':>8s} {'Shift':>8s} {'Δ from g†':>10s}")
print(f"  {'-'*60}")

for r in jk_by_inv[:10]:
    shift = r['inversion_point'] - baseline_inversion
    print(f"  {r['removed']:15s} {r['env']:>5s} {r['n_low_accel_points']:5d} "
          f"{r['inversion_point']:8.3f} {shift:+8.3f} {r['inv_delta_from_gdagger']:10.3f}")

# ================================================================
# JACKKNIFE SUMMARY STATISTICS
# ================================================================
print(f"\n{'='*72}")
print("JACKKNIFE SUMMARY")
print(f"{'='*72}")

delta_arr = np.array([r['delta_sigma'] for r in jk_results if not np.isnan(r['delta_sigma'])])
levene_arr = np.array([r['levene_p'] for r in jk_results if not np.isnan(r['levene_p'])])
inv_arr = np.array([r['inversion_point'] for r in jk_results if r['inversion_point'] is not None])

print(f"\n  Δσ(field-dense) across jackknife:")
print(f"    Baseline: {baseline_stats['delta_sigma']:+.4f}")
print(f"    JK mean:  {np.mean(delta_arr):+.4f}")
print(f"    JK std:   {np.std(delta_arr):.4f}")
print(f"    JK range: [{np.min(delta_arr):+.4f}, {np.max(delta_arr):+.4f}]")

n_sign_flip = np.sum(delta_arr * baseline_stats['delta_sigma'] < 0)
print(f"    Sign flips: {n_sign_flip}/{len(delta_arr)} "
      f"({'NONE' if n_sign_flip == 0 else f'{n_sign_flip} galaxies flip the sign'})")

# Does any removal make Levene p < 0.01?
n_significant = np.sum(levene_arr < 0.01)
print(f"\n  Levene p-values across jackknife:")
print(f"    Baseline: {baseline_stats['levene_p']:.6f}")
print(f"    JK range: [{np.min(levene_arr):.6f}, {np.max(levene_arr):.6f}]")
print(f"    JK with p < 0.01: {n_significant}/{len(levene_arr)}")
print(f"    JK with p < 0.05: {np.sum(levene_arr < 0.05)}/{len(levene_arr)}")

if n_significant > 0:
    sig_gals = [r['removed'] for r in jk_results if r['levene_p'] < 0.01]
    print(f"    Galaxies whose removal makes p < 0.01: {sig_gals}")

print(f"\n  Inversion point across jackknife:")
if baseline_inversion is not None and len(inv_arr) > 0:
    print(f"    Baseline: {baseline_inversion:.3f}")
    print(f"    JK mean:  {np.mean(inv_arr):.3f}")
    print(f"    JK std:   {np.std(inv_arr):.3f}")
    print(f"    JK range: [{np.min(inv_arr):.3f}, {np.max(inv_arr):.3f}]")
    print(f"    Max shift from baseline: {np.max(np.abs(inv_arr - baseline_inversion)):.3f} dex")

    # Distance from g† statistics
    dist_from_gd = np.abs(inv_arr - LOG_G_DAGGER)
    n_within_02 = np.sum(dist_from_gd < 0.20)
    n_within_03 = np.sum(dist_from_gd < 0.30)
    print(f"    Within 0.2 dex of g†: {n_within_02}/{len(inv_arr)} ({100*n_within_02/len(inv_arr):.0f}%)")
    print(f"    Within 0.3 dex of g†: {n_within_03}/{len(inv_arr)} ({100*n_within_03/len(inv_arr):.0f}%)")

# ================================================================
# N-GALAXY RANDOM REMOVAL (bootstrap jackknife)
# ================================================================
print(f"\n{'='*72}")
print("N-GALAXY RANDOM REMOVAL (bootstrap jackknife)")
print(f"{'='*72}")

rng = np.random.default_rng(42)
N_BOOT = 1000

for n_remove in [2, 3, 5, 10]:
    if n_remove >= n_gal:
        continue

    delta_boot = np.zeros(N_BOOT)
    inv_boot = np.zeros(N_BOOT)
    inv_valid = 0

    for i in range(N_BOOT):
        remove_idx = rng.choice(n_gal, size=n_remove, replace=False)
        remove_set = {galaxy_names[j] for j in remove_idx}

        stats = compute_low_accel_stats(gal_results, exclude_names=remove_set)
        inv = compute_inversion_point(gal_results, exclude_names=remove_set)

        delta_boot[i] = stats['delta_sigma'] if not np.isnan(stats['delta_sigma']) else 0.0
        if inv is not None:
            inv_boot[inv_valid] = inv
            inv_valid += 1

    inv_boot = inv_boot[:inv_valid]

    n_sign_flips = np.sum(delta_boot * baseline_stats['delta_sigma'] < 0)

    print(f"\n  Remove {n_remove} galaxies ({N_BOOT} iterations):")
    print(f"    Δσ: mean={np.mean(delta_boot):+.4f} ± {np.std(delta_boot):.4f}"
          f"  range=[{np.min(delta_boot):+.4f}, {np.max(delta_boot):+.4f}]"
          f"  sign flips: {n_sign_flips}/{N_BOOT}")
    if inv_valid > 0:
        print(f"    Inv: mean={np.mean(inv_boot):.3f} ± {np.std(inv_boot):.3f}"
              f"  range=[{np.min(inv_boot):.3f}, {np.max(inv_boot):.3f}]"
              f"  mean |Δ from g†|={np.mean(np.abs(inv_boot - LOG_G_DAGGER)):.3f}")


# ================================================================
# VERDICT
# ================================================================
print(f"\n{'='*72}")
print("OVERALL VERDICT")
print(f"{'='*72}")

# Environmental scatter
max_delta_shift = np.max(np.abs(delta_arr - baseline_stats['delta_sigma']))
if n_sign_flip == 0 and max_delta_shift < 0.02:
    env_verdict = "ROBUST — no single galaxy drives the environmental scatter result"
elif n_sign_flip == 0:
    env_verdict = "MOSTLY ROBUST — no sign flips, but some galaxies shift Δσ substantially"
else:
    flip_gals = [r['removed'] for r in jk_results
                 if r['delta_sigma'] * baseline_stats['delta_sigma'] < 0]
    env_verdict = f"FRAGILE — removing {flip_gals} flips the Δσ sign"

# Inversion point
if len(inv_arr) > 0:
    max_inv_shift = np.max(np.abs(inv_arr - baseline_inversion))
    if max_inv_shift < 0.15:
        inv_verdict = "ROBUST — inversion point stable to <0.15 dex under any single removal"
    elif max_inv_shift < 0.30:
        inv_verdict = "MOSTLY ROBUST — max shift <0.3 dex"
    else:
        worst = jk_by_inv[0]
        inv_verdict = f"FRAGILE — removing {worst['removed']} shifts inversion by {max_inv_shift:.2f} dex"
else:
    inv_verdict = "UNKNOWN — no inversion points found"

print(f"\n  Environmental scatter: {env_verdict}")
print(f"  Inversion point:      {inv_verdict}")

# ================================================================
# SAVE
# ================================================================
output = {
    'test_name': 'jackknife_robustness',
    'description': ('Leave-one-out jackknife over 131 galaxies testing stability '
                    'of (1) low-accel environmental scatter uniformity and '
                    '(2) dσ/dx=0 inversion point near g†'),
    'n_galaxies': n_gal,
    'n_dense': n_dense,
    'n_field': n_field,
    'baseline': {
        'sigma_dense': round(baseline_stats['sigma_dense'], 5),
        'sigma_field': round(baseline_stats['sigma_field'], 5),
        'delta_sigma': round(baseline_stats['delta_sigma'], 5),
        'levene_p': round(baseline_stats['levene_p'], 8),
        'inversion_point': round(baseline_inversion, 4) if baseline_inversion is not None else None,
        'inv_distance_from_gdagger': round(abs(baseline_inversion - LOG_G_DAGGER), 4) if baseline_inversion is not None else None,
    },
    'jackknife_summary': {
        'delta_sigma': {
            'mean': round(float(np.mean(delta_arr)), 5),
            'std': round(float(np.std(delta_arr)), 5),
            'min': round(float(np.min(delta_arr)), 5),
            'max': round(float(np.max(delta_arr)), 5),
            'n_sign_flips': int(n_sign_flip),
            'max_shift_from_baseline': round(float(max_delta_shift), 5),
        },
        'levene_p': {
            'min': round(float(np.min(levene_arr)), 8),
            'max': round(float(np.max(levene_arr)), 8),
            'n_below_001': int(n_significant),
            'n_below_005': int(np.sum(levene_arr < 0.05)),
        },
        'inversion_point': {
            'mean': round(float(np.mean(inv_arr)), 4) if len(inv_arr) > 0 else None,
            'std': round(float(np.std(inv_arr)), 4) if len(inv_arr) > 0 else None,
            'min': round(float(np.min(inv_arr)), 4) if len(inv_arr) > 0 else None,
            'max': round(float(np.max(inv_arr)), 4) if len(inv_arr) > 0 else None,
            'max_shift_from_baseline': round(float(np.max(np.abs(inv_arr - baseline_inversion))), 4) if len(inv_arr) > 0 else None,
            'n_within_02_of_gdagger': int(np.sum(np.abs(inv_arr - LOG_G_DAGGER) < 0.20)) if len(inv_arr) > 0 else 0,
            'n_within_03_of_gdagger': int(np.sum(np.abs(inv_arr - LOG_G_DAGGER) < 0.30)) if len(inv_arr) > 0 else 0,
        },
    },
    'verdicts': {
        'environmental_scatter': env_verdict,
        'inversion_point': inv_verdict,
    },
    'top_10_influential_delta': [
        {k: v for k, v in r.items()}
        for r in jk_by_delta[:10]
    ],
    'top_10_influential_inversion': [
        {k: v for k, v in r.items()}
        for r in jk_by_inv[:10]
    ],
    'full_jackknife': jk_results,
}

outpath = os.path.join(RESULTS_DIR, 'summary_jackknife_robustness.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
