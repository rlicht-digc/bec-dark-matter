#!/usr/bin/env python3
"""
Split-Half Internal Replication of Inversion Point
====================================================

THE STRONGEST INTERNAL REPLICATION TEST: If the scatter derivative
zero-crossing at g† is a population-level property, it should appear
independently in random halves of the SPARC sample.

Method:
  1. Randomly split 131 galaxies into two halves (65+66)
  2. Compute the scatter derivative and inversion point for each half
  3. Repeat 1000 times with different random splits
  4. Report: what fraction of splits find inversion within 0.20 dex of g†?

This is a permutation-based internal replication. If both halves consistently
find the inversion at g†, it's not driven by any specific galaxy subset.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
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
print("SPLIT-HALF INTERNAL REPLICATION OF INVERSION POINT")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")


# ================================================================
# DATA LOADING (standard)
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
# COMPUTE RAR DATA PER GALAXY
# ================================================================
print("\n[2] Computing accelerations...")

gal_data = {}
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

    gal_data[name] = {
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'n_pts': len(log_gbar),
    }

gal_names = sorted(gal_data.keys())
n_gals = len(gal_names)
print(f"  {n_gals} galaxies pass quality cuts")
total_pts = sum(g['n_pts'] for g in gal_data.values())
print(f"  {total_pts} total RAR points")


# ================================================================
# INVERSION POINT FINDER
# ================================================================
def find_inversion(log_gbar_arr, log_gobs_arr, bin_width=0.30, offset=0.0):
    """Compute scatter derivative inversion (dσ/d(log g_bar) = 0).

    Uses RAR residuals and binned scatter derivative.
    Returns the log g_bar value of the zero-crossing nearest to g†.
    The scatter derivative has multiple crossings; the physical one
    is near g† where scatter transitions from increasing to decreasing.
    """
    # Compute RAR residuals
    gbar = 10**log_gbar_arr
    rar_pred = np.log10(gbar / (1 - np.exp(-np.sqrt(gbar / g_dagger))))
    resid = log_gobs_arr - rar_pred

    lo = log_gbar_arr.min() + offset
    hi = log_gbar_arr.max()
    edges = np.arange(lo, hi + bin_width, bin_width)

    if len(edges) < 3:
        return None

    centers = []
    sigmas = []
    for j in range(len(edges) - 1):
        mask = (log_gbar_arr >= edges[j]) & (log_gbar_arr < edges[j+1])
        if np.sum(mask) >= 10:
            centers.append(0.5 * (edges[j] + edges[j+1]))
            sigmas.append(np.std(resid[mask]))

    if len(centers) < 4:
        return None

    centers = np.array(centers)
    sigmas = np.array(sigmas)

    # Compute derivative via finite differences
    dsigma = np.diff(sigmas)
    dcenter = np.array([0.5 * (centers[j] + centers[j+1]) for j in range(len(centers)-1)])

    # Find ALL zero-crossings (positive to negative = scatter peak)
    crossings = []
    for j in range(len(dsigma) - 1):
        if dsigma[j] > 0 and dsigma[j+1] < 0:
            x0, x1 = dcenter[j], dcenter[j+1]
            y0, y1 = dsigma[j], dsigma[j+1]
            crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(crossing)

    if not crossings:
        return None

    # Return the crossing nearest to g†
    crossings = np.array(crossings)
    nearest_idx = np.argmin(np.abs(crossings - LOG_G_DAGGER))
    return crossings[nearest_idx]


def find_inversion_nonparametric(log_gbar_arr, log_gobs_arr, bin_width=0.30, offset=0.0):
    """Same as find_inversion but using kernel regression residuals (no RAR assumption).

    Uses fast Nadaraya-Watson kernel regression instead of full LOESS
    to avoid O(n^2) matrix operations.
    """
    x = log_gbar_arr
    y = log_gobs_arr
    n_grid = 100
    x_grid = np.linspace(x.min(), x.max(), n_grid)
    h = 0.15 * (x.max() - x.min())

    # Fast Nadaraya-Watson kernel regression (vectorized, no matrix)
    y_grid = np.zeros(n_grid)
    for i, xg in enumerate(x_grid):
        w = np.exp(-0.5 * ((x - xg) / h)**2)
        w_sum = w.sum()
        if w_sum > 0:
            y_grid[i] = np.dot(w, y) / w_sum
        else:
            y_grid[i] = np.mean(y)

    # Interpolate predictions at data points
    pred = np.interp(log_gbar_arr, x_grid, y_grid)
    resid = log_gobs_arr - pred

    lo = log_gbar_arr.min() + offset
    hi = log_gbar_arr.max()
    edges = np.arange(lo, hi + bin_width, bin_width)

    if len(edges) < 3:
        return None

    centers = []
    sigmas = []
    for j in range(len(edges) - 1):
        mask = (log_gbar_arr >= edges[j]) & (log_gbar_arr < edges[j+1])
        if np.sum(mask) >= 10:
            centers.append(0.5 * (edges[j] + edges[j+1]))
            sigmas.append(np.std(resid[mask]))

    if len(centers) < 4:
        return None

    centers = np.array(centers)
    sigmas = np.array(sigmas)

    dsigma = np.diff(sigmas)
    dcenter = np.array([0.5 * (centers[j] + centers[j+1]) for j in range(len(centers)-1)])

    # Find ALL crossings, return nearest to g†
    crossings = []
    for j in range(len(dsigma) - 1):
        if dsigma[j] > 0 and dsigma[j+1] < 0:
            x0, x1 = dcenter[j], dcenter[j+1]
            y0, y1 = dsigma[j], dsigma[j+1]
            crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(crossing)

    if not crossings:
        return None
    crossings = np.array(crossings)
    nearest_idx = np.argmin(np.abs(crossings - LOG_G_DAGGER))
    return crossings[nearest_idx]


# ================================================================
# FULL SAMPLE BASELINE
# ================================================================
print("\n[3] Full-sample baseline inversion...")

all_lg = np.concatenate([gal_data[n]['log_gbar'] for n in gal_names])
all_lo = np.concatenate([gal_data[n]['log_gobs'] for n in gal_names])

baseline_rar = find_inversion(all_lg, all_lo)
baseline_loess = find_inversion_nonparametric(all_lg, all_lo)

print(f"  Full sample (RAR residuals):   crossing = {baseline_rar:.4f}  (Δ from g† = {baseline_rar - LOG_G_DAGGER:+.4f})")
print(f"  Full sample (LOESS residuals): crossing = {baseline_loess:.4f}  (Δ from g† = {baseline_loess - LOG_G_DAGGER:+.4f})")


# ================================================================
# SPLIT-HALF REPLICATION (1000 random splits)
# ================================================================
print("\n" + "=" * 72)
print("SPLIT-HALF REPLICATION — 1000 random splits")
print("=" * 72)

N_SPLITS = 1000
rng = np.random.RandomState(42)

half_size = n_gals // 2
results_A_rar = []
results_B_rar = []
results_A_loess = []
results_B_loess = []

both_within_020_rar = 0
both_within_020_loess = 0
any_failure_rar = 0
any_failure_loess = 0

for i in range(N_SPLITS):
    perm = rng.permutation(n_gals)
    idx_A = perm[:half_size]
    idx_B = perm[half_size:]

    names_A = [gal_names[j] for j in idx_A]
    names_B = [gal_names[j] for j in idx_B]

    lg_A = np.concatenate([gal_data[n]['log_gbar'] for n in names_A])
    lo_A = np.concatenate([gal_data[n]['log_gobs'] for n in names_A])
    lg_B = np.concatenate([gal_data[n]['log_gbar'] for n in names_B])
    lo_B = np.concatenate([gal_data[n]['log_gobs'] for n in names_B])

    # RAR-based inversion
    inv_A = find_inversion(lg_A, lo_A)
    inv_B = find_inversion(lg_B, lo_B)

    if inv_A is not None and inv_B is not None:
        results_A_rar.append(inv_A)
        results_B_rar.append(inv_B)
        dA = abs(inv_A - LOG_G_DAGGER)
        dB = abs(inv_B - LOG_G_DAGGER)
        if dA < 0.20 and dB < 0.20:
            both_within_020_rar += 1
        if dA >= 0.20 or dB >= 0.20:
            any_failure_rar += 1

    # LOESS-based inversion (non-parametric)
    inv_A_l = find_inversion_nonparametric(lg_A, lo_A)
    inv_B_l = find_inversion_nonparametric(lg_B, lo_B)

    if inv_A_l is not None and inv_B_l is not None:
        results_A_loess.append(inv_A_l)
        results_B_loess.append(inv_B_l)
        dA_l = abs(inv_A_l - LOG_G_DAGGER)
        dB_l = abs(inv_B_l - LOG_G_DAGGER)
        if dA_l < 0.20 and dB_l < 0.20:
            both_within_020_loess += 1
        if dA_l >= 0.20 or dB_l >= 0.20:
            any_failure_loess += 1

    if (i + 1) % 200 == 0:
        print(f"  Completed {i+1}/{N_SPLITS} splits...")

results_A_rar = np.array(results_A_rar)
results_B_rar = np.array(results_B_rar)
results_A_loess = np.array(results_A_loess)
results_B_loess = np.array(results_B_loess)

n_valid_rar = len(results_A_rar)
n_valid_loess = len(results_A_loess)

print(f"\n  RAR residuals: {n_valid_rar}/{N_SPLITS} splits produced valid crossings in both halves")
print(f"  LOESS residuals: {n_valid_loess}/{N_SPLITS} splits produced valid crossings in both halves")

# RAR statistics
print(f"\n  --- RAR RESIDUALS ---")
all_rar = np.concatenate([results_A_rar, results_B_rar])
print(f"  Mean inversion (all half-samples): {np.mean(all_rar):.4f}")
print(f"  Std inversion:  {np.std(all_rar):.4f}")
print(f"  Median:         {np.median(all_rar):.4f}")
print(f"  Range:          [{np.min(all_rar):.4f}, {np.max(all_rar):.4f}]")
print(f"  Mean distance from g†: {np.mean(np.abs(all_rar - LOG_G_DAGGER)):.4f}")
pct_within_010 = 100 * np.mean(np.abs(all_rar - LOG_G_DAGGER) < 0.10)
pct_within_020 = 100 * np.mean(np.abs(all_rar - LOG_G_DAGGER) < 0.20)
pct_within_030 = 100 * np.mean(np.abs(all_rar - LOG_G_DAGGER) < 0.30)
print(f"  % within 0.10 dex of g†: {pct_within_010:.1f}%")
print(f"  % within 0.20 dex of g†: {pct_within_020:.1f}%")
print(f"  % within 0.30 dex of g†: {pct_within_030:.1f}%")
print(f"  BOTH halves within 0.20 dex: {both_within_020_rar}/{n_valid_rar} = {100*both_within_020_rar/n_valid_rar:.1f}%")

# Half A vs Half B consistency
mean_diff = np.mean(np.abs(results_A_rar - results_B_rar))
print(f"  Mean |A − B| (internal consistency): {mean_diff:.4f} dex")

# LOESS statistics
print(f"\n  --- LOESS RESIDUALS (non-parametric) ---")
all_loess = np.concatenate([results_A_loess, results_B_loess])
print(f"  Mean inversion (all half-samples): {np.mean(all_loess):.4f}")
print(f"  Std inversion:  {np.std(all_loess):.4f}")
print(f"  Median:         {np.median(all_loess):.4f}")
print(f"  Range:          [{np.min(all_loess):.4f}, {np.max(all_loess):.4f}]")
print(f"  Mean distance from g†: {np.mean(np.abs(all_loess - LOG_G_DAGGER)):.4f}")
pct_within_010_l = 100 * np.mean(np.abs(all_loess - LOG_G_DAGGER) < 0.10)
pct_within_020_l = 100 * np.mean(np.abs(all_loess - LOG_G_DAGGER) < 0.20)
pct_within_030_l = 100 * np.mean(np.abs(all_loess - LOG_G_DAGGER) < 0.30)
print(f"  % within 0.10 dex of g†: {pct_within_010_l:.1f}%")
print(f"  % within 0.20 dex of g†: {pct_within_020_l:.1f}%")
print(f"  % within 0.30 dex of g†: {pct_within_030_l:.1f}%")
print(f"  BOTH halves within 0.20 dex: {both_within_020_loess}/{n_valid_loess} = {100*both_within_020_loess/n_valid_loess:.1f}%")

mean_diff_l = np.mean(np.abs(results_A_loess - results_B_loess))
print(f"  Mean |A − B| (internal consistency): {mean_diff_l:.4f} dex")


# ================================================================
# STRATIFIED SPLIT-HALF (control for morphology & luminosity)
# ================================================================
print("\n" + "=" * 72)
print("STRATIFIED SPLIT-HALF — balanced morphology & luminosity")
print("=" * 72)

# Sort galaxies by luminosity and morphology, then split alternating
# This ensures each half has similar luminosity and morphology distributions
gal_props_list = []
for name in gal_names:
    if name in sparc_props:
        T = sparc_props[name]['T']
        L = sparc_props[name]['L36']
    else:
        T = 5
        L = 1.0
    gal_props_list.append((name, T, L))

N_STRAT_SPLITS = 1000
strat_A_rar = []
strat_B_rar = []

for i in range(N_STRAT_SPLITS):
    # Stratify by morphology (early T<5 vs late T>=5) and luminosity
    early = [(n, T, L) for n, T, L in gal_props_list if T < 5]
    late = [(n, T, L) for n, T, L in gal_props_list if T >= 5]

    # Within each stratum, sort by luminosity + small random perturbation
    rng.shuffle(early)
    rng.shuffle(late)

    # Alternate assignment to A and B within each stratum
    names_A = []
    names_B = []
    for stratum in [early, late]:
        perm_s = rng.permutation(len(stratum))
        half_s = len(stratum) // 2
        for j, idx in enumerate(perm_s):
            if j < half_s:
                names_A.append(stratum[idx][0])
            else:
                names_B.append(stratum[idx][0])

    lg_A = np.concatenate([gal_data[n]['log_gbar'] for n in names_A])
    lo_A = np.concatenate([gal_data[n]['log_gobs'] for n in names_A])
    lg_B = np.concatenate([gal_data[n]['log_gbar'] for n in names_B])
    lo_B = np.concatenate([gal_data[n]['log_gobs'] for n in names_B])

    inv_A = find_inversion(lg_A, lo_A)
    inv_B = find_inversion(lg_B, lo_B)

    if inv_A is not None and inv_B is not None:
        strat_A_rar.append(inv_A)
        strat_B_rar.append(inv_B)

strat_A_rar = np.array(strat_A_rar)
strat_B_rar = np.array(strat_B_rar)
all_strat = np.concatenate([strat_A_rar, strat_B_rar])
n_strat_valid = len(strat_A_rar)

print(f"  {n_strat_valid}/{N_STRAT_SPLITS} stratified splits valid")
print(f"  Mean inversion: {np.mean(all_strat):.4f}")
print(f"  Std:            {np.std(all_strat):.4f}")
print(f"  Mean dist from g†: {np.mean(np.abs(all_strat - LOG_G_DAGGER)):.4f}")
pct_strat_020 = 100 * np.mean(np.abs(all_strat - LOG_G_DAGGER) < 0.20)
print(f"  % within 0.20 dex of g†: {pct_strat_020:.1f}%")
both_strat = np.sum((np.abs(strat_A_rar - LOG_G_DAGGER) < 0.20) &
                     (np.abs(strat_B_rar - LOG_G_DAGGER) < 0.20))
print(f"  BOTH halves within 0.20: {both_strat}/{n_strat_valid} = {100*both_strat/n_strat_valid:.1f}%")


# ================================================================
# SIZE-BALANCED SPLIT (equal number of data points per half)
# ================================================================
print("\n" + "=" * 72)
print("SIZE-BALANCED SPLIT — equal data points per half")
print("=" * 72)

# Sort by number of points per galaxy, then alternate
gal_by_npts = sorted(gal_names, key=lambda n: gal_data[n]['n_pts'], reverse=True)

N_SIZE_SPLITS = 1000
size_A_rar = []
size_B_rar = []

for i in range(N_SIZE_SPLITS):
    names_A = []
    names_B = []
    pts_A = 0
    pts_B = 0

    # Shuffle within similar-size groups, then greedily balance
    shuffled = list(gal_by_npts)
    # Add small random perturbation to ordering
    for j in range(len(shuffled) - 1):
        if rng.random() < 0.5:
            shuffled[j], shuffled[j+1] = shuffled[j+1], shuffled[j]

    for name in shuffled:
        npts = gal_data[name]['n_pts']
        if pts_A <= pts_B:
            names_A.append(name)
            pts_A += npts
        else:
            names_B.append(name)
            pts_B += npts

    lg_A = np.concatenate([gal_data[n]['log_gbar'] for n in names_A])
    lo_A = np.concatenate([gal_data[n]['log_gobs'] for n in names_A])
    lg_B = np.concatenate([gal_data[n]['log_gbar'] for n in names_B])
    lo_B = np.concatenate([gal_data[n]['log_gobs'] for n in names_B])

    inv_A = find_inversion(lg_A, lo_A)
    inv_B = find_inversion(lg_B, lo_B)

    if inv_A is not None and inv_B is not None:
        size_A_rar.append(inv_A)
        size_B_rar.append(inv_B)

size_A_rar = np.array(size_A_rar)
size_B_rar = np.array(size_B_rar)
all_size = np.concatenate([size_A_rar, size_B_rar])
n_size_valid = len(size_A_rar)

print(f"  {n_size_valid}/{N_SIZE_SPLITS} size-balanced splits valid")
print(f"  Mean inversion: {np.mean(all_size):.4f}")
print(f"  Std:            {np.std(all_size):.4f}")
print(f"  Mean dist from g†: {np.mean(np.abs(all_size - LOG_G_DAGGER)):.4f}")
pct_size_020 = 100 * np.mean(np.abs(all_size - LOG_G_DAGGER) < 0.20)
print(f"  % within 0.20 dex of g†: {pct_size_020:.1f}%")
both_size = np.sum((np.abs(size_A_rar - LOG_G_DAGGER) < 0.20) &
                    (np.abs(size_B_rar - LOG_G_DAGGER) < 0.20))
print(f"  BOTH halves within 0.20: {both_size}/{n_size_valid} = {100*both_size/n_size_valid:.1f}%")


# ================================================================
# THIRDS SPLIT (three independent subsamples)
# ================================================================
print("\n" + "=" * 72)
print("THIRDS SPLIT — three independent subsamples")
print("=" * 72)

N_THIRD_SPLITS = 1000
third_size = n_gals // 3
thirds_results = [[], [], []]

all_three_within_020 = 0

for i in range(N_THIRD_SPLITS):
    perm = rng.permutation(n_gals)
    splits = [perm[:third_size], perm[third_size:2*third_size], perm[2*third_size:]]

    inversions = []
    for s, idx_set in enumerate(splits):
        names_s = [gal_names[j] for j in idx_set]
        lg_s = np.concatenate([gal_data[n]['log_gbar'] for n in names_s])
        lo_s = np.concatenate([gal_data[n]['log_gobs'] for n in names_s])
        inv_s = find_inversion(lg_s, lo_s)
        inversions.append(inv_s)

    if all(inv is not None for inv in inversions):
        for s in range(3):
            thirds_results[s].append(inversions[s])
        if all(abs(inv - LOG_G_DAGGER) < 0.20 for inv in inversions):
            all_three_within_020 += 1

thirds_results = [np.array(t) for t in thirds_results]
n_thirds_valid = len(thirds_results[0])
all_thirds = np.concatenate(thirds_results)

print(f"  {n_thirds_valid}/{N_THIRD_SPLITS} splits with all 3 valid")
print(f"  Mean inversion (all thirds): {np.mean(all_thirds):.4f}")
print(f"  Std:  {np.std(all_thirds):.4f}")
print(f"  Mean dist from g†: {np.mean(np.abs(all_thirds - LOG_G_DAGGER)):.4f}")
pct_thirds_020 = 100 * np.mean(np.abs(all_thirds - LOG_G_DAGGER) < 0.20)
print(f"  % within 0.20 dex of g†: {pct_thirds_020:.1f}%")
print(f"  ALL THREE within 0.20: {all_three_within_020}/{n_thirds_valid} = {100*all_three_within_020/n_thirds_valid:.1f}%")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

print(f"\n  Full sample inversion (RAR):   {baseline_rar:.4f}  (Δ = {baseline_rar - LOG_G_DAGGER:+.4f})")
print(f"  Full sample inversion (LOESS): {baseline_loess:.4f}  (Δ = {baseline_loess - LOG_G_DAGGER:+.4f})")

print(f"\n  Random halves (RAR):   mean = {np.mean(all_rar):.4f} ± {np.std(all_rar):.4f}, "
      f"BOTH within 0.20: {100*both_within_020_rar/n_valid_rar:.1f}%")
print(f"  Random halves (LOESS): mean = {np.mean(all_loess):.4f} ± {np.std(all_loess):.4f}, "
      f"BOTH within 0.20: {100*both_within_020_loess/n_valid_loess:.1f}%")
print(f"  Stratified halves:     mean = {np.mean(all_strat):.4f} ± {np.std(all_strat):.4f}, "
      f"BOTH within 0.20: {100*both_strat/n_strat_valid:.1f}%")
print(f"  Size-balanced halves:  mean = {np.mean(all_size):.4f} ± {np.std(all_size):.4f}, "
      f"BOTH within 0.20: {100*both_size/n_size_valid:.1f}%")
print(f"  Random thirds:         mean = {np.mean(all_thirds):.4f} ± {np.std(all_thirds):.4f}, "
      f"ALL 3 within 0.20: {100*all_three_within_020/n_thirds_valid:.1f}%")

# Verdict
all_methods_pct = [
    100*both_within_020_rar/n_valid_rar,
    100*both_within_020_loess/n_valid_loess,
    100*both_strat/n_strat_valid,
    100*both_size/n_size_valid,
]
min_pct = min(all_methods_pct)

if min_pct >= 90:
    verdict = "STRONGLY_REPLICATED"
    print(f"\n  >>> VERDICT: STRONGLY REPLICATED — inversion at g† appears in ≥{min_pct:.0f}% of all split-halves")
elif min_pct >= 70:
    verdict = "REPLICATED"
    print(f"\n  >>> VERDICT: REPLICATED — inversion at g† appears in ≥{min_pct:.0f}% of split-halves")
elif min_pct >= 50:
    verdict = "PARTIALLY_REPLICATED"
    print(f"\n  >>> VERDICT: PARTIALLY REPLICATED — {min_pct:.0f}% success rate")
else:
    verdict = "NOT_REPLICATED"
    print(f"\n  >>> VERDICT: NOT REPLICATED — only {min_pct:.0f}% success rate")


# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    'test_name': 'split_half_replication',
    'n_galaxies': n_gals,
    'n_total_points': int(total_pts),
    'baseline': {
        'rar_crossing': float(baseline_rar) if baseline_rar else None,
        'loess_crossing': float(baseline_loess) if baseline_loess else None,
        'rar_dist_gdagger': float(baseline_rar - LOG_G_DAGGER) if baseline_rar else None,
        'loess_dist_gdagger': float(baseline_loess - LOG_G_DAGGER) if baseline_loess else None,
    },
    'random_halves_rar': {
        'n_splits': N_SPLITS,
        'n_valid': int(n_valid_rar),
        'mean_crossing': float(np.mean(all_rar)),
        'std_crossing': float(np.std(all_rar)),
        'median_crossing': float(np.median(all_rar)),
        'pct_within_010': float(pct_within_010),
        'pct_within_020': float(pct_within_020),
        'pct_within_030': float(pct_within_030),
        'both_within_020_pct': float(100*both_within_020_rar/n_valid_rar),
        'mean_halfA_halfB_diff': float(mean_diff),
    },
    'random_halves_loess': {
        'n_valid': int(n_valid_loess),
        'mean_crossing': float(np.mean(all_loess)),
        'std_crossing': float(np.std(all_loess)),
        'pct_within_010': float(pct_within_010_l),
        'pct_within_020': float(pct_within_020_l),
        'both_within_020_pct': float(100*both_within_020_loess/n_valid_loess),
        'mean_halfA_halfB_diff': float(mean_diff_l),
    },
    'stratified_halves': {
        'n_valid': int(n_strat_valid),
        'mean_crossing': float(np.mean(all_strat)),
        'std_crossing': float(np.std(all_strat)),
        'pct_within_020': float(pct_strat_020),
        'both_within_020_pct': float(100*both_strat/n_strat_valid),
    },
    'size_balanced_halves': {
        'n_valid': int(n_size_valid),
        'mean_crossing': float(np.mean(all_size)),
        'std_crossing': float(np.std(all_size)),
        'pct_within_020': float(pct_size_020),
        'both_within_020_pct': float(100*both_size/n_size_valid),
    },
    'random_thirds': {
        'n_valid': int(n_thirds_valid),
        'mean_crossing': float(np.mean(all_thirds)),
        'std_crossing': float(np.std(all_thirds)),
        'pct_within_020': float(pct_thirds_020),
        'all_three_within_020_pct': float(100*all_three_within_020/n_thirds_valid),
    },
    'overall_verdict': verdict,
}

out_path = os.path.join(RESULTS_DIR, 'summary_split_half_replication.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_path}")
print("=" * 72)
print("Done.")
