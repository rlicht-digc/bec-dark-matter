#!/usr/bin/env python3
"""
Binning Robustness Test for Inversion Point
=============================================

The derivative analysis (test_mc_distance_and_inversion.py) found dσ/d(log g_bar) = 0
at log g_bar = -9.86, just 0.065 dex from g† = -9.92.

A reviewer could ask: is that a binning artifact? If you shift your bins by half a
width, does the zero-crossing jump by 0.5 dex?

This test sweeps:
  - 5 bin widths:  0.20, 0.27, 0.35, 0.50, 0.70 dex
  - 5 bin offsets per width (evenly spaced within one bin width)
  = 25 total binning configurations

For each, we compute the full derivative profile and find the dσ/dx zero-crossing
closest to g†. If the inversion point stays within ±0.1 dex of g† across all 25
configurations, it's robust. If it wanders by >0.3 dex, it's a binning artifact.

Additionally, we test the environmental Δσ sign-flip (where field scatter exceeds
dense scatter) under all binning configurations.
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


def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


# Environment classification (same as mc_distance_and_inversion.py)
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


def compute_all_residuals(galaxies, sparc_props):
    """Compute RAR residuals for all quality-cut galaxies. Returns list of (log_gbar, log_res, env, name)."""
    all_pts = []

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

        for i in range(len(log_gbar)):
            all_pts.append((log_gbar[i], log_res[i], env, name))

    return all_pts


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
        if y[i] * y[i+1] < 0:
            x_cross = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
            crossings.append(float(x_cross))
    return crossings


def analyze_binning(all_pts, bin_width, offset, gbar_range=(-12.5, -8.5)):
    """
    Compute scatter profile and derivatives for a given binning.

    Returns dict with bin stats, derivatives, and zero-crossings, or None if too few bins.
    """
    lo_edge = gbar_range[0] + offset
    hi_edge = gbar_range[1]

    # Build bin edges
    edges = []
    e = lo_edge
    while e < hi_edge + bin_width:
        edges.append(e)
        e += bin_width
    edges = np.array(edges)

    # Filter to edges that have data
    gbar_arr = np.array([p[0] for p in all_pts])
    res_arr = np.array([p[1] for p in all_pts])
    env_arr = np.array([p[2] for p in all_pts])

    centers = []
    sigmas = []
    sigmas_dense = []
    sigmas_field = []
    deltas = []
    n_all_list = []

    for j in range(len(edges) - 1):
        lo, hi = edges[j], edges[j+1]
        center = (lo + hi) / 2.0

        mask = (gbar_arr >= lo) & (gbar_arr < hi)
        res_bin = res_arr[mask]

        if len(res_bin) < 10:
            continue

        d_mask = mask & (env_arr == 'dense')
        f_mask = mask & (env_arr == 'field')
        res_d = res_arr[d_mask]
        res_f = res_arr[f_mask]

        s_all = float(np.std(res_bin))
        s_d = float(np.std(res_d)) if len(res_d) >= 5 else np.nan
        s_f = float(np.std(res_f)) if len(res_f) >= 5 else np.nan
        delta = s_f - s_d if not (np.isnan(s_f) or np.isnan(s_d)) else np.nan

        centers.append(center)
        sigmas.append(s_all)
        sigmas_dense.append(s_d)
        sigmas_field.append(s_f)
        deltas.append(delta)
        n_all_list.append(len(res_bin))

    if len(centers) < 4:
        return None

    centers = np.array(centers)
    sigmas = np.array(sigmas)

    # Derivative of sigma
    dsigma_dx = numerical_derivative(centers, sigmas)

    # Find zero crossings of dsigma/dx
    sigma_crossings = find_zero_crossings(centers, dsigma_dx)

    # Find the crossing closest to g†
    closest = None
    closest_dist = np.inf
    for x in sigma_crossings:
        d = abs(x - LOG_G_DAGGER)
        if d < closest_dist:
            closest_dist = d
            closest = x

    # Environmental delta zero crossings
    valid_delta = [(c, d) for c, d in zip(centers, deltas) if not np.isnan(d)]
    delta_crossings = []
    if len(valid_delta) >= 3:
        dc = np.array([v[0] for v in valid_delta])
        dd = np.array([v[1] for v in valid_delta])
        delta_crossings = find_zero_crossings(dc, dd)

    return {
        'bin_width': round(float(bin_width), 3),
        'offset': round(float(offset), 4),
        'n_bins_used': len(centers),
        'centers': [round(float(c), 3) for c in centers],
        'sigmas': [round(float(s), 5) for s in sigmas],
        'n_per_bin': n_all_list,
        'dsigma_dx_crossings': [round(x, 4) for x in sigma_crossings],
        'closest_to_gdagger': round(closest, 4) if closest is not None else None,
        'distance_from_gdagger': round(closest_dist, 4) if closest is not None else None,
        'delta_sigma_crossings': [round(x, 4) for x in delta_crossings],
    }


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("BINNING ROBUSTNESS TEST FOR INVERSION POINT")
print("=" * 72)

galaxies, sparc_props = load_sparc()
all_pts = compute_all_residuals(galaxies, sparc_props)
n_total = len(all_pts)
n_gal = len(set(p[3] for p in all_pts))
print(f"  Loaded {n_gal} galaxies, {n_total} data points")

# Bin widths to sweep
bin_widths = [0.20, 0.27, 0.35, 0.50, 0.70]
n_offsets = 5

print(f"\n  Testing {len(bin_widths)} bin widths × {n_offsets} offsets = {len(bin_widths)*n_offsets} configurations")
print(f"  g† = 10^{LOG_G_DAGGER:.3f}")

all_configs = []
all_closest = []

print(f"\n  {'Width':>6s} {'Offset':>7s} {'N_bins':>6s} {'Closest dσ/dx=0':>16s} {'Δ from g†':>10s} {'status':>10s}")
print(f"  {'-'*60}")

for bw in bin_widths:
    offsets = np.linspace(0, bw * 0.8, n_offsets)  # 0 to 80% of bin width

    for off in offsets:
        result = analyze_binning(all_pts, bw, off)

        if result is None:
            print(f"  {bw:6.2f} {off:7.3f}   (too few bins)")
            continue

        closest = result['closest_to_gdagger']
        dist = result['distance_from_gdagger']

        if closest is not None:
            status = "ROBUST" if dist < 0.15 else ("MARGINAL" if dist < 0.30 else "DRIFTED")
            print(f"  {bw:6.2f} {off:7.3f} {result['n_bins_used']:6d} {closest:16.3f} {dist:10.3f} {status:>10s}")
            all_closest.append(closest)
        else:
            print(f"  {bw:6.2f} {off:7.3f} {result['n_bins_used']:6d}     (no crossing)       ---        ---")

        all_configs.append(result)

# ================================================================
# SUMMARY STATISTICS
# ================================================================
print(f"\n{'='*72}")
print("SUMMARY")
print(f"{'='*72}")

if all_closest:
    arr = np.array(all_closest)
    print(f"\n  Total binning configurations tested: {len(all_configs)}")
    print(f"  Configurations with dσ/dx=0 crossings: {len(all_closest)}")
    print(f"\n  Closest-to-g† inversion point across all configs:")
    print(f"    Mean:    {np.mean(arr):.3f}")
    print(f"    Median:  {np.median(arr):.3f}")
    print(f"    Std:     {np.std(arr):.3f}")
    print(f"    Min:     {np.min(arr):.3f}")
    print(f"    Max:     {np.max(arr):.3f}")
    print(f"    Range:   {np.max(arr) - np.min(arr):.3f} dex")
    print(f"\n  g† = {LOG_G_DAGGER:.3f}")
    print(f"    Mean distance from g†: {np.mean(np.abs(arr - LOG_G_DAGGER)):.3f} dex")
    print(f"    Max distance from g†:  {np.max(np.abs(arr - LOG_G_DAGGER)):.3f} dex")

    n_within_01 = np.sum(np.abs(arr - LOG_G_DAGGER) < 0.10)
    n_within_015 = np.sum(np.abs(arr - LOG_G_DAGGER) < 0.15)
    n_within_02 = np.sum(np.abs(arr - LOG_G_DAGGER) < 0.20)
    n_within_03 = np.sum(np.abs(arr - LOG_G_DAGGER) < 0.30)

    print(f"\n  Robustness (fraction within threshold of g†):")
    print(f"    Within 0.10 dex: {n_within_01}/{len(arr)} ({100*n_within_01/len(arr):.0f}%)")
    print(f"    Within 0.15 dex: {n_within_015}/{len(arr)} ({100*n_within_015/len(arr):.0f}%)")
    print(f"    Within 0.20 dex: {n_within_02}/{len(arr)} ({100*n_within_02/len(arr):.0f}%)")
    print(f"    Within 0.30 dex: {n_within_03}/{len(arr)} ({100*n_within_03/len(arr):.0f}%)")

    if np.max(np.abs(arr - LOG_G_DAGGER)) < 0.15:
        verdict = "ROBUST — inversion stays within 0.15 dex of g† across ALL binnings"
    elif np.mean(np.abs(arr - LOG_G_DAGGER)) < 0.15:
        verdict = "MOSTLY ROBUST — mean distance < 0.15 dex but some configurations drift"
    elif np.mean(np.abs(arr - LOG_G_DAGGER)) < 0.30:
        verdict = "MARGINAL — inversion wanders but stays in the neighborhood of g†"
    else:
        verdict = "NOT ROBUST — inversion point is sensitive to binning"

    print(f"\n  VERDICT: {verdict}")

    # Per-width summary
    print(f"\n  Per-width breakdown:")
    print(f"    {'Width':>6s} {'N_configs':>9s} {'Mean':>8s} {'Std':>8s} {'Mean |Δ|':>9s}")
    print(f"    {'-'*44}")
    for bw in bin_widths:
        bw_vals = [c['closest_to_gdagger'] for c in all_configs
                   if c['closest_to_gdagger'] is not None
                   and abs(c['bin_width'] - bw) < 0.01]
        if bw_vals:
            bv = np.array(bw_vals)
            print(f"    {bw:6.2f} {len(bv):9d} {np.mean(bv):8.3f} {np.std(bv):8.3f}"
                  f" {np.mean(np.abs(bv - LOG_G_DAGGER)):9.3f}")

else:
    print("  No zero-crossings found in any configuration!")
    verdict = "FAILED — no inversions detected"

# ================================================================
# SAVE
# ================================================================
output = {
    'test_name': 'binning_robustness_inversion_point',
    'description': ('Sweep bin width (0.20-0.70 dex) and offset (5 per width) to test '
                    'whether the dσ/d(log g_bar) = 0 inversion point is stable across '
                    'binning configurations.'),
    'g_dagger_log': LOG_G_DAGGER,
    'bin_widths_tested': bin_widths,
    'n_offsets_per_width': n_offsets,
    'n_configurations': len(all_configs),
    'n_with_crossings': len(all_closest),
    'closest_values': [round(float(x), 4) for x in all_closest] if all_closest else [],
    'summary': {
        'mean': round(float(np.mean(all_closest)), 4) if all_closest else None,
        'median': round(float(np.median(all_closest)), 4) if all_closest else None,
        'std': round(float(np.std(all_closest)), 4) if all_closest else None,
        'min': round(float(np.min(all_closest)), 4) if all_closest else None,
        'max': round(float(np.max(all_closest)), 4) if all_closest else None,
        'range': round(float(np.max(all_closest) - np.min(all_closest)), 4) if all_closest else None,
        'mean_distance_from_gdagger': round(float(np.mean(np.abs(np.array(all_closest) - LOG_G_DAGGER))), 4) if all_closest else None,
        'max_distance_from_gdagger': round(float(np.max(np.abs(np.array(all_closest) - LOG_G_DAGGER))), 4) if all_closest else None,
    },
    'verdict': verdict,
    'all_configurations': all_configs,
}

outpath = os.path.join(RESULTS_DIR, 'summary_binning_robustness.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
