#!/usr/bin/env python3
"""
INVERSION POINT DISTANCE SENSITIVITY — Phase 5
=================================================

Tests whether the RAR scatter inversion survives SYSTEMATIC distance shifts,
filling the gap left by test_mc_distance_and_inversion.py which only tested
random Gaussian perturbations.

12 scenarios:
   1. Baseline (SPARC distances)
   2. All +10%          7. fD=1 -20%
   3. All +20%          8. CF4 for fD=1 only
   4. All -10%          9. Full CF4 (all galaxies)
   5. All -20%         10. Hybrid catalog
   6. fD=1 +20%        11. UMa → 18.6 Mpc (Verheijen)
                       12. UMa → CF4 (pathological test)

Key physics: Under uniform scaling, gbar is distance-invariant
(V_bar² and R both scale as D_ratio, canceling in V²/R). Only gobs
shifts (Vobs fixed, R scales). Non-uniform shifts differentially
affect galaxies and can alter the scatter profile.

Inputs:
  - data/sparc/SPARC_table2_rotmods.dat
  - data/sparc/SPARC_Lelli2016c.mrt
  - data/cf4/cf4_distance_cache.json
  - data/distance_catalog_sparc.json (from Phase 2)
  - data/distance_catalog_hybrid.json
  - data/distance_catalog_cf4.json

Output:
  - analysis/results/summary_inversion_distance_sensitivity.json
"""

import os
import json
import numpy as np
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10
kpc_m = 3.086e19
LOG_G_DAGGER = np.log10(g_dagger)
UMA_DISTANCE = 18.6  # Verheijen+2001


def rar_function(log_gbar, a0=g_dagger):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def numerical_derivative(x, y):
    dy = np.zeros_like(y)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(1, len(y) - 1):
        dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return dy


def find_zero_crossings(x, y, direction=None):
    crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i+1] < 0:
            if direction == 'pos_to_neg' and y[i] < 0:
                continue
            if direction == 'neg_to_pos' and y[i] > 0:
                continue
            x_cross = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
            crossings.append(float(x_cross))
    return crossings


# UMa membership
UMA_GALAXIES = {
    'NGC3726', 'NGC3769', 'NGC3877', 'NGC3893', 'NGC3917',
    'NGC3949', 'NGC3953', 'NGC3972', 'NGC3992', 'NGC4010',
    'NGC4013', 'NGC4051', 'NGC4085', 'NGC4088', 'NGC4100',
    'NGC4138', 'NGC4157', 'NGC4183', 'NGC4217',
    'UGC06399', 'UGC06446', 'UGC06667', 'UGC06786', 'UGC06787',
    'UGC06818', 'UGC06917', 'UGC06923', 'UGC06930', 'UGC06973',
    'UGC06983', 'UGC07089',
}


print("=" * 72)
print("PHASE 5: INVERSION POINT DISTANCE SENSITIVITY")
print("=" * 72)

# ================================================================
# STEP 1: Load data
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

print(f"  {len(galaxies)} RCs, {len(sparc_props)} with properties")

# Load CF4 cache
cf4_path = os.path.join(PROJECT_ROOT, 'data', 'cf4', 'cf4_distance_cache.json')
with open(cf4_path, 'r') as f:
    cf4_cache = json.load(f)

# Load catalogs
catalogs = {}
for cat_name in ['sparc', 'hybrid', 'cf4']:
    cat_path = os.path.join(PROJECT_ROOT, 'data', f'distance_catalog_{cat_name}.json')
    with open(cat_path, 'r') as f:
        catalogs[cat_name] = json.load(f)

print(f"  CF4 cache: {len(cf4_cache)} entries")
print(f"  Catalogs loaded: {list(catalogs.keys())}")

# ================================================================
# STEP 2: Define scenarios
# ================================================================

def make_distance_map(galaxies, sparc_props, scenario_name):
    """Return {galaxy_name: D_use} for a given scenario."""
    dist_map = {}

    for name in galaxies:
        if name not in sparc_props:
            continue
        prop = sparc_props[name]
        D_sparc = prop['D']
        fD = prop['fD']

        if scenario_name == 'baseline':
            dist_map[name] = D_sparc

        elif scenario_name == 'all_plus_10':
            dist_map[name] = D_sparc * 1.10

        elif scenario_name == 'all_plus_20':
            dist_map[name] = D_sparc * 1.20

        elif scenario_name == 'all_minus_10':
            dist_map[name] = D_sparc * 0.90

        elif scenario_name == 'all_minus_20':
            dist_map[name] = D_sparc * 0.80

        elif scenario_name == 'fd1_plus_20':
            dist_map[name] = D_sparc * 1.20 if fD == 1 else D_sparc

        elif scenario_name == 'fd1_minus_20':
            dist_map[name] = D_sparc * 0.80 if fD == 1 else D_sparc

        elif scenario_name == 'cf4_for_fd1':
            if fD == 1 and name in cf4_cache and cf4_cache[name].get('status') == 'success':
                dist_map[name] = cf4_cache[name]['D_cf4']
            else:
                dist_map[name] = D_sparc

        elif scenario_name == 'full_cf4':
            if name in catalogs['cf4']:
                dist_map[name] = catalogs['cf4'][name]['D_Mpc']
            else:
                dist_map[name] = D_sparc

        elif scenario_name == 'hybrid':
            if name in catalogs['hybrid']:
                dist_map[name] = catalogs['hybrid'][name]['D_Mpc']
            else:
                dist_map[name] = D_sparc

        elif scenario_name == 'uma_18.6':
            dist_map[name] = UMA_DISTANCE if name in UMA_GALAXIES else D_sparc

        elif scenario_name == 'uma_cf4':
            if name in UMA_GALAXIES and name in cf4_cache and cf4_cache[name].get('status') == 'success':
                dist_map[name] = cf4_cache[name]['D_cf4']
            else:
                dist_map[name] = D_sparc

    return dist_map


SCENARIOS = [
    ('baseline',      'Baseline (SPARC distances)'),
    ('all_plus_10',   'All +10%'),
    ('all_plus_20',   'All +20%'),
    ('all_minus_10',  'All -10%'),
    ('all_minus_20',  'All -20%'),
    ('fd1_plus_20',   'fD=1 only +20%'),
    ('fd1_minus_20',  'fD=1 only -20%'),
    ('cf4_for_fd1',   'CF4 for fD=1 only'),
    ('full_cf4',      'Full CF4 (all galaxies)'),
    ('hybrid',        'Hybrid catalog'),
    ('uma_18.6',      'UMa → 18.6 Mpc (Verheijen)'),
    ('uma_cf4',       'UMa → CF4 (pathological)'),
]


def compute_inversion_analysis(galaxies, sparc_props, dist_map):
    """Compute RAR residuals and find inversion points for a distance map."""
    # Compute residuals
    all_log_gbar = []
    all_log_res = []

    for name, gdata in galaxies.items():
        if name not in sparc_props or name not in dist_map:
            continue
        prop = sparc_props[name]
        if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
            continue

        D_use = dist_map[name]
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
        log_res = log_gobs - rar_function(log_gbar)

        all_log_gbar.extend(log_gbar)
        all_log_res.extend(log_res)

    all_log_gbar = np.array(all_log_gbar)
    all_log_res = np.array(all_log_res)

    # Fine binning for derivative analysis
    n_fine = 15
    fine_edges = np.linspace(-12.5, -8.5, n_fine + 1)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2

    bin_sigmas = []
    bin_skews = []
    bin_kurts = []
    bin_ns = []
    valid_centers = []

    for j in range(n_fine):
        mask = (all_log_gbar >= fine_edges[j]) & (all_log_gbar < fine_edges[j+1])
        res = all_log_res[mask]
        if len(res) >= 10:
            bin_sigmas.append(float(np.std(res)))
            bin_skews.append(float(skew(res)))
            bin_kurts.append(float(kurtosis(res, fisher=True)))
            bin_ns.append(len(res))
            valid_centers.append(fine_centers[j])

    if len(valid_centers) < 4:
        return {
            'n_points': len(all_log_gbar),
            'n_bins_valid': len(valid_centers),
            'inversion_point': None,
            'all_crossings': [],
            'overall_sigma': float(np.std(all_log_res)),
        }

    centers = np.array(valid_centers)
    sigmas = np.array(bin_sigmas)
    skews = np.array(bin_skews)
    kurts = np.array(bin_kurts)

    # Derivatives
    dsigma = numerical_derivative(centers, sigmas)
    dskew = numerical_derivative(centers, skews)

    # Zero crossings
    sigma_crossings = find_zero_crossings(centers, dsigma, direction='pos_to_neg')
    skew_sign_crossings = find_zero_crossings(centers, skews)

    # Inversion: dσ/dx = 0 with pos_to_neg (scatter peak)
    all_inversions = []
    for x in sigma_crossings:
        all_inversions.append(('dsigma=0', x, abs(x - LOG_G_DAGGER)))
    for x in skew_sign_crossings:
        all_inversions.append(('skew=0', x, abs(x - LOG_G_DAGGER)))

    nearest = None
    if all_inversions:
        all_inversions.sort(key=lambda t: t[2])
        nearest = all_inversions[0][1]

    return {
        'n_points': len(all_log_gbar),
        'n_bins_valid': len(valid_centers),
        'overall_sigma': round(float(np.std(all_log_res)), 5),
        'overall_mean': round(float(np.mean(all_log_res)), 5),
        'inversion_point': round(nearest, 3) if nearest else None,
        'inversion_offset_from_gdagger': round(nearest - LOG_G_DAGGER, 3) if nearest else None,
        'all_crossings': [
            {'type': t[0], 'log_gbar': round(t[1], 3), 'dist_from_gdagger': round(t[2], 3)}
            for t in all_inversions
        ],
        'dsigma_crossings': [round(x, 3) for x in sigma_crossings],
        'skew_sign_crossings': [round(x, 3) for x in skew_sign_crossings],
        'bin_profile': [
            {
                'center': round(float(centers[i]), 3),
                'sigma': round(bin_sigmas[i], 5),
                'skewness': round(bin_skews[i], 4),
                'kurtosis': round(bin_kurts[i], 4),
                'n': bin_ns[i],
            }
            for i in range(len(valid_centers))
        ],
    }


# ================================================================
# STEP 3: Run all scenarios
# ================================================================
print(f"\n{'='*72}")
print("RUNNING 12 SCENARIOS")
print(f"{'='*72}")

scenario_results = {}

print(f"\n  {'#':>3s} {'Scenario':35s} {'N_pts':>7s} {'σ_all':>8s} {'Inversion':>10s} {'Offset':>8s}")
print(f"  {'-'*76}")

for idx, (sc_name, sc_desc) in enumerate(SCENARIOS, 1):
    dist_map = make_distance_map(galaxies, sparc_props, sc_name)
    result = compute_inversion_analysis(galaxies, sparc_props, dist_map)
    scenario_results[sc_name] = {
        'description': sc_desc,
        'scenario_number': idx,
        **result,
    }

    inv_str = f"{result['inversion_point']:+.3f}" if result['inversion_point'] else "    N/A"
    off_str = f"{result['inversion_offset_from_gdagger']:+.3f}" if result['inversion_offset_from_gdagger'] else "    N/A"
    print(f"  {idx:3d} {sc_desc:35s} {result['n_points']:7d} {result['overall_sigma']:8.5f} "
          f"{inv_str:>10s} {off_str:>8s}")

# ================================================================
# STEP 4: Analysis of results
# ================================================================
print(f"\n{'='*72}")
print("ANALYSIS")
print(f"{'='*72}")

baseline = scenario_results['baseline']
baseline_inv = baseline['inversion_point']

print(f"\n  Baseline inversion: log g_bar = {baseline_inv}")
print(f"  g† (BEC prediction): log g_bar = {LOG_G_DAGGER:.3f}")
if baseline_inv:
    print(f"  Offset: {baseline_inv - LOG_G_DAGGER:+.3f} dex")

# Uniform scaling analysis
print(f"\n  Uniform scaling (physics expectation: inversion should be stable):")
uniform_scenarios = ['all_plus_10', 'all_plus_20', 'all_minus_10', 'all_minus_20']
for sc in uniform_scenarios:
    r = scenario_results[sc]
    inv = r['inversion_point']
    if inv and baseline_inv:
        shift = inv - baseline_inv
        print(f"    {r['description']:25s}: inversion={inv:+.3f}, shift from baseline={shift:+.3f} dex")
    else:
        print(f"    {r['description']:25s}: inversion={'N/A' if not inv else f'{inv:+.3f}'}")

# Non-uniform shifts
print(f"\n  Non-uniform shifts (expected to alter scatter profile):")
nonuniform_scenarios = ['fd1_plus_20', 'fd1_minus_20', 'cf4_for_fd1', 'full_cf4', 'hybrid', 'uma_18.6', 'uma_cf4']
for sc in nonuniform_scenarios:
    r = scenario_results[sc]
    inv = r['inversion_point']
    if inv and baseline_inv:
        shift = inv - baseline_inv
        stable = abs(shift) < 0.3
        print(f"    {r['description']:35s}: inversion={inv:+.3f}, shift={shift:+.3f} dex "
              f"{'(STABLE)' if stable else '(SHIFTED)'}")
    else:
        print(f"    {r['description']:35s}: inversion={'N/A' if not inv else f'{inv:+.3f}'}")

# UMa pathological
print(f"\n  UMa pathological test:")
uma_sparc = scenario_results.get('uma_18.6', {})
uma_cf4 = scenario_results.get('uma_cf4', {})
if uma_sparc.get('inversion_point') and uma_cf4.get('inversion_point'):
    shift = uma_cf4['inversion_point'] - uma_sparc['inversion_point']
    print(f"    UMa → 18.6 Mpc: inversion = {uma_sparc['inversion_point']:+.3f}")
    print(f"    UMa → CF4:      inversion = {uma_cf4['inversion_point']:+.3f}")
    print(f"    Shift: {shift:+.3f} dex")
    if abs(shift) > 0.3:
        print(f"    -> CF4 UMa distances DESTABILIZE the inversion point")
    else:
        print(f"    -> Inversion is robust to UMa distance choice")

# Scatter profile comparison (sigma at key acceleration bins)
print(f"\n  Scatter at key accelerations:")
print(f"  {'Scenario':35s}", end='')
# Show a few key bin centers from baseline profile
key_bins = []
for b in baseline.get('bin_profile', []):
    if abs(b['center'] - LOG_G_DAGGER) < 0.3 or b['center'] < -11.5 or b['center'] > -9.0:
        key_bins.append(b['center'])
for c in key_bins[:4]:
    print(f" {c:>8.2f}", end='')
print()
print(f"  {'-'*72}")

for sc_name, _ in SCENARIOS:
    r = scenario_results[sc_name]
    profile = {b['center']: b['sigma'] for b in r.get('bin_profile', [])}
    desc = r['description'][:35]
    print(f"  {desc:35s}", end='')
    for c in key_bins[:4]:
        # Find closest bin
        closest = min(profile.keys(), key=lambda x: abs(x - c)) if profile else None
        if closest and abs(closest - c) < 0.2:
            print(f" {profile[closest]:8.5f}", end='')
        else:
            print(f" {'---':>8s}", end='')
    print()

# ================================================================
# STEP 5: Save
# ================================================================
summary = {
    'test_name': 'inversion_distance_sensitivity',
    'description': '12-scenario systematic distance shift test for RAR scatter inversion stability',
    'g_dagger_log': LOG_G_DAGGER,
    'n_scenarios': len(SCENARIOS),
    'scenarios': scenario_results,
    'key_findings': {
        'baseline_inversion': baseline_inv,
        'uniform_shifts_stable': all(
            abs((scenario_results[sc].get('inversion_point') or 0) - (baseline_inv or 0)) < 0.3
            for sc in uniform_scenarios
            if scenario_results[sc].get('inversion_point') is not None
        ) if baseline_inv else None,
        'cf4_full_inversion': scenario_results['full_cf4'].get('inversion_point'),
        'hybrid_inversion': scenario_results['hybrid'].get('inversion_point'),
        'uma_cf4_shift': (
            (scenario_results['uma_cf4'].get('inversion_point') or 0) -
            (scenario_results['uma_18.6'].get('inversion_point') or 0)
        ) if (scenario_results['uma_cf4'].get('inversion_point') and
              scenario_results['uma_18.6'].get('inversion_point')) else None,
    },
}

outpath = os.path.join(RESULTS_DIR, 'summary_inversion_distance_sensitivity.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("Done.")
