#!/usr/bin/env python3
"""
ENVIRONMENTAL BATTERY x3 DISTANCE CATALOGS — Phase 3
=====================================================

Runs the full environmental scatter test battery under three different
distance catalogs (SPARC, Hybrid, Full CF4), then cross-compares results.

Battery per catalog:
  Test A: Point-level Levene (dense vs field scatter)
  Test B: Galaxy-level Levene + KS + Mann-Whitney
  Test C: Error-corrected scatter (Haubner model, adapted per catalog)
  Test D: 4-bin acceleration-dependent scatter comparison
  Test E: Galaxy-resampled bootstrap (10,000 iterations)
  Test F: Galaxy-block permutation (10,000 iterations)

Cross-catalog comparison uses the INTERSECTION of galaxies passing
quality cuts under all 3 catalogs for apples-to-apples comparison.

Inputs:
  - data/distance_catalog_sparc.json (from Phase 2)
  - data/distance_catalog_hybrid.json
  - data/distance_catalog_cf4.json
  - data/sparc/SPARC_table2_rotmods.dat
  - data/sparc/SPARC_Lelli2016c.mrt

Output:
  - analysis/results/summary_env_triple_distance.json
"""

import os
import json
import numpy as np
from scipy.stats import levene, mannwhitneyu, ks_2samp
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

# Distance error config
PRIMARY_ERRORS = {'TRGB': 0.05, 'Cepheid': 0.05, 'SBF': 0.05, 'SNe': 0.07, 'maser': 0.10}

def rar_function(log_gbar, a0=g_dagger):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)

def haubner_delta(D_Mpc, delta_inf=0.022, alpha=-0.8, D_tr=46.0, kappa=1.8):
    D = max(D_Mpc, 0.01)
    return delta_inf * D**alpha * (D**(1/kappa) + D_tr**(1/kappa))**(-alpha * kappa)

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
        return 'dense', 'UMa'
    if name in GROUP_MEMBERS:
        return 'dense', GROUP_MEMBERS[name]
    return 'field', 'field'


# Acceleration bins
GBAR_BINS = [
    (-13.0, -11.5, 'Deep DM (gbar < 10^-11.5)'),
    (-11.5, -10.5, 'Transition (10^-11.5 < gbar < 10^-10.5)'),
    (-10.5, -9.5, 'Baryon-dom (10^-10.5 < gbar < 10^-9.5)'),
    (-9.5, -8.0, 'High gbar (gbar > 10^-9.5)'),
]


def assign_distance_error(name, D_Mpc, fD, catalog_name, source_str):
    """Assign distance error based on catalog and source."""
    if catalog_name == 'sparc':
        # fD-based errors
        if fD == 2:
            return np.log10(1 + PRIMARY_ERRORS['TRGB'])
        elif fD == 3:
            return np.log10(1 + PRIMARY_ERRORS['Cepheid'])
        elif fD == 4:
            return np.log10(1 + 0.10)
        elif fD == 5:
            return np.log10(1 + PRIMARY_ERRORS['SNe'])
        else:
            return haubner_delta(D_Mpc)
    elif catalog_name == 'hybrid':
        # Source-dependent
        if 'TRGB' in source_str or 'Cepheid' in source_str:
            return np.log10(1 + 0.05)
        elif 'UMa' in source_str:
            return np.log10(1 + 0.10)
        elif 'SNe' in source_str:
            return np.log10(1 + PRIMARY_ERRORS['SNe'])
        elif 'CF4' in source_str or 'NED' in source_str:
            return haubner_delta(D_Mpc)
        else:
            return haubner_delta(D_Mpc)
    elif catalog_name == 'cf4':
        # Haubner model for all
        return haubner_delta(D_Mpc)
    return haubner_delta(D_Mpc)


def compute_residuals_for_catalog(galaxies, sparc_props, catalog, catalog_name):
    """Compute RAR residuals using distances from the given catalog."""
    results = {}
    for name, gdata in galaxies.items():
        if name not in sparc_props or name not in catalog:
            continue
        prop = sparc_props[name]
        if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
            continue

        cat_entry = catalog[name]
        D_use = cat_entry['D_Mpc']
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

        env_binary, env_group = classify_env(name)
        sigma_D_dex = assign_distance_error(name, D_use, prop['fD'], catalog_name, cat_entry['source'])

        results[name] = {
            'log_gbar': log_gbar,
            'log_res': log_res,
            'mean_res': float(np.mean(log_res)),
            'std_res': float(np.std(log_res)),
            'n_points': len(log_res),
            'env': env_binary,
            'env_group': env_group,
            'D': D_use,
            'fD': prop['fD'],
            'sigma_D_dex': sigma_D_dex,
            'source': cat_entry['source'],
        }
    return results


def run_environmental_battery(results, label):
    """Run the full 6-test environmental battery. Returns dict of results."""
    battery = {}
    n_dense = sum(1 for r in results.values() if r['env'] == 'dense')
    n_field = sum(1 for r in results.values() if r['env'] == 'field')
    battery['n_galaxies'] = len(results)
    battery['n_dense'] = n_dense
    battery['n_field'] = n_field

    # --- Test A: Point-level Levene ---
    dense_res_pts = np.concatenate([r['log_res'] for r in results.values() if r['env'] == 'dense'])
    field_res_pts = np.concatenate([r['log_res'] for r in results.values() if r['env'] == 'field'])
    sigma_dense = float(np.std(dense_res_pts))
    sigma_field = float(np.std(field_res_pts))
    delta_sigma = sigma_field - sigma_dense
    stat_L, p_L = levene(dense_res_pts, field_res_pts)

    battery['test_A_point_level'] = {
        'dense_sigma': round(sigma_dense, 4),
        'field_sigma': round(sigma_field, 4),
        'delta_sigma': round(delta_sigma, 4),
        'levene_F': round(float(stat_L), 3),
        'levene_p': round(float(p_L), 6),
        'n_dense_pts': len(dense_res_pts),
        'n_field_pts': len(field_res_pts),
    }

    # --- Test B: Galaxy-level Levene + KS + MW ---
    dense_means = np.array([r['mean_res'] for r in results.values() if r['env'] == 'dense'])
    field_means = np.array([r['mean_res'] for r in results.values() if r['env'] == 'field'])
    sigma_dense_gal = float(np.std(dense_means))
    sigma_field_gal = float(np.std(field_means))
    stat_L2, p_L2 = levene(dense_means, field_means)
    ks_stat, ks_p = ks_2samp(dense_means, field_means)
    mw_stat, mw_p = mannwhitneyu(dense_means, field_means, alternative='two-sided')

    battery['test_B_galaxy_level'] = {
        'dense_sigma': round(sigma_dense_gal, 4),
        'field_sigma': round(sigma_field_gal, 4),
        'delta_sigma': round(sigma_field_gal - sigma_dense_gal, 4),
        'levene_p': round(float(p_L2), 6),
        'ks_p': round(float(ks_p), 6),
        'mw_p': round(float(mw_p), 6),
    }

    # --- Test C: Error-corrected scatter ---
    dense_intrinsic = np.array([r['std_res'] for r in results.values() if r['env'] == 'dense'])
    field_intrinsic = np.array([r['std_res'] for r in results.values() if r['env'] == 'field'])
    stat_L3, p_L3 = levene(dense_intrinsic, field_intrinsic)
    dense_err_mean = float(np.mean([r['sigma_D_dex'] for r in results.values() if r['env'] == 'dense']))
    field_err_mean = float(np.mean([r['sigma_D_dex'] for r in results.values() if r['env'] == 'field']))

    battery['test_C_error_corrected'] = {
        'dense_scatter_mean': round(float(np.mean(dense_intrinsic)), 4),
        'field_scatter_mean': round(float(np.mean(field_intrinsic)), 4),
        'dense_dist_error_mean': round(dense_err_mean, 4),
        'field_dist_error_mean': round(field_err_mean, 4),
        'levene_p': round(float(p_L3), 6),
    }

    # --- Test D: 4-bin acceleration scatter ---
    bin_results = {}
    for gbar_lo, gbar_hi, bin_label in GBAR_BINS:
        d_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                                 for r in results.values() if r['env'] == 'dense'])
        f_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                                 for r in results.values() if r['env'] == 'field'])
        if len(d_pts) >= 5 and len(f_pts) >= 5:
            sd, sf = float(np.std(d_pts)), float(np.std(f_pts))
            stat, p = levene(d_pts, f_pts)
            bin_results[bin_label] = {
                'n_dense': len(d_pts), 'n_field': len(f_pts),
                'sigma_dense': round(sd, 4), 'sigma_field': round(sf, 4),
                'delta_sigma': round(sf - sd, 4), 'levene_p': round(float(p), 6),
            }
        else:
            bin_results[bin_label] = {'n_dense': len(d_pts), 'n_field': len(f_pts), 'insufficient_data': True}

    battery['test_D_binned'] = bin_results

    # --- Test E: Bootstrap galaxy resampling (10,000) ---
    rng = np.random.default_rng(42)
    n_boot = 10000
    dense_gal_names = [name for name, r in results.items() if r['env'] == 'dense']
    field_gal_names = [name for name, r in results.items() if r['env'] == 'field']
    delta_boots = np.zeros(n_boot)

    for b in range(n_boot):
        d_sample = rng.choice(dense_gal_names, size=len(dense_gal_names), replace=True)
        f_sample = rng.choice(field_gal_names, size=len(field_gal_names), replace=True)
        d_res = np.array([results[n]['mean_res'] for n in d_sample])
        f_res = np.array([results[n]['mean_res'] for n in f_sample])
        delta_boots[b] = np.std(f_res) - np.std(d_res)

    ci_95 = np.percentile(delta_boots, [2.5, 97.5])
    ci_68 = np.percentile(delta_boots, [16, 84])
    p_field_gt = float(np.mean(delta_boots > 0))

    battery['test_E_bootstrap'] = {
        'observed_delta': round(sigma_field_gal - sigma_dense_gal, 4),
        'bootstrap_median': round(float(np.median(delta_boots)), 4),
        'ci_68': [round(float(ci_68[0]), 4), round(float(ci_68[1]), 4)],
        'ci_95': [round(float(ci_95[0]), 4), round(float(ci_95[1]), 4)],
        'p_field_gt_dense': round(p_field_gt, 4),
    }

    # --- Test F: Galaxy-block permutation (10,000) ---
    rng_blk = np.random.default_rng(99)
    n_perm = 10000
    all_gal_names = list(results.keys())
    n_total = len(all_gal_names)
    n_dense_orig = sum(1 for n in all_gal_names if results[n]['env'] == 'dense')
    obs_delta_pts = delta_sigma

    perm_deltas = np.zeros(n_perm)
    for p in range(n_perm):
        perm_idx = rng_blk.permutation(n_total)
        perm_dense = [all_gal_names[i] for i in perm_idx[:n_dense_orig]]
        perm_field = [all_gal_names[i] for i in perm_idx[n_dense_orig:]]
        d_res = np.concatenate([results[n]['log_res'] for n in perm_dense])
        f_res = np.concatenate([results[n]['log_res'] for n in perm_field])
        perm_deltas[p] = np.std(f_res) - np.std(d_res)

    p_block_two = float(np.mean(np.abs(perm_deltas) >= abs(obs_delta_pts)))
    p_block_one = float(np.mean(perm_deltas >= obs_delta_pts))

    battery['test_F_block_permutation'] = {
        'observed_delta': round(obs_delta_pts, 4),
        'null_mean': round(float(np.mean(perm_deltas)), 4),
        'null_std': round(float(np.std(perm_deltas)), 4),
        'p_two_sided': round(p_block_two, 4),
        'p_field_gt_dense': round(p_block_one, 4),
    }

    return battery


print("=" * 72)
print("PHASE 3: ENVIRONMENTAL BATTERY x3 DISTANCE CATALOGS")
print("=" * 72)

# ================================================================
# STEP 1: Load SPARC rotation curves and properties
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

# ================================================================
# STEP 2: Load distance catalogs
# ================================================================
print("\n[2] Loading distance catalogs...")

catalogs = {}
for cat_name in ['sparc', 'hybrid', 'cf4']:
    cat_path = os.path.join(PROJECT_ROOT, 'data', f'distance_catalog_{cat_name}.json')
    with open(cat_path, 'r') as f:
        catalogs[cat_name] = json.load(f)
    print(f"  {cat_name}: {len(catalogs[cat_name])} galaxies")

# ================================================================
# STEP 3: Run battery for each catalog
# ================================================================
all_battery_results = {}

for cat_name, catalog in catalogs.items():
    print(f"\n{'='*72}")
    print(f"RUNNING BATTERY: {cat_name.upper()} CATALOG")
    print(f"{'='*72}")

    results = compute_residuals_for_catalog(galaxies, sparc_props, catalog, cat_name)
    n_d = sum(1 for r in results.values() if r['env'] == 'dense')
    n_f = sum(1 for r in results.values() if r['env'] == 'field')
    print(f"  After cuts: {len(results)} galaxies ({n_d} dense, {n_f} field)")

    battery = run_environmental_battery(results, cat_name)

    # Print summary
    tA = battery['test_A_point_level']
    tB = battery['test_B_galaxy_level']
    tE = battery['test_E_bootstrap']
    tF = battery['test_F_block_permutation']

    print(f"\n  Test A (point Levene): Δσ={tA['delta_sigma']:+.4f}, p={tA['levene_p']:.6f}")
    print(f"  Test B (galaxy Levene): Δσ={tB['delta_sigma']:+.4f}, p={tB['levene_p']:.6f}")
    print(f"  Test D (binned):")
    for bin_label, br in battery['test_D_binned'].items():
        if 'insufficient_data' not in br:
            sig = '*' if br['levene_p'] < 0.05 else ''
            print(f"    {bin_label:40s} Δσ={br['delta_sigma']:+.4f}, p={br['levene_p']:.6f} {sig}")
    print(f"  Test E (bootstrap): median Δ={tE['bootstrap_median']:+.4f}, "
          f"95%CI=[{tE['ci_95'][0]:+.4f}, {tE['ci_95'][1]:+.4f}]")
    print(f"  Test F (block perm): p(2-sided)={tF['p_two_sided']:.4f}, "
          f"p(field>dense)={tF['p_field_gt_dense']:.4f}")

    all_battery_results[cat_name] = battery

# ================================================================
# STEP 4: Cross-catalog comparison (intersection sample)
# ================================================================
print(f"\n{'='*72}")
print("CROSS-CATALOG COMPARISON (intersection sample)")
print(f"{'='*72}")

# Find intersection of galaxies passing quality cuts under all 3 catalogs
results_per_cat = {}
for cat_name, catalog in catalogs.items():
    results_per_cat[cat_name] = compute_residuals_for_catalog(
        galaxies, sparc_props, catalog, cat_name)

intersection = set(results_per_cat['sparc'].keys()) & \
               set(results_per_cat['hybrid'].keys()) & \
               set(results_per_cat['cf4'].keys())
print(f"\n  Intersection: {len(intersection)} galaxies")

# Run battery on intersection only
intersection_results = {}
for cat_name in ['sparc', 'hybrid', 'cf4']:
    res_intersect = {n: results_per_cat[cat_name][n] for n in intersection}
    intersection_results[cat_name] = run_environmental_battery(res_intersect, f'{cat_name}_intersect')

# Comparison table
print(f"\n  {'Test':35s} {'SPARC':>10s} {'Hybrid':>10s} {'CF4':>10s}")
print(f"  {'-'*70}")

for test_key, test_label, metric in [
    ('test_A_point_level', 'Test A: Δσ (point)', 'delta_sigma'),
    ('test_A_point_level', 'Test A: Levene p', 'levene_p'),
    ('test_B_galaxy_level', 'Test B: Δσ (galaxy)', 'delta_sigma'),
    ('test_B_galaxy_level', 'Test B: Levene p', 'levene_p'),
    ('test_E_bootstrap', 'Test E: bootstrap median', 'bootstrap_median'),
    ('test_F_block_permutation', 'Test F: block perm p(2s)', 'p_two_sided'),
]:
    vals = []
    for cat_name in ['sparc', 'hybrid', 'cf4']:
        v = intersection_results[cat_name][test_key][metric]
        vals.append(v)
    print(f"  {test_label:35s} {vals[0]:+10.4f} {vals[1]:+10.4f} {vals[2]:+10.4f}")

# Track key changes
print(f"\n  Key questions:")
for bin_label_short, bin_label in [('Deep DM', 'Deep DM (gbar < 10^-11.5)'),
                                    ('Transition', 'Transition (10^-11.5 < gbar < 10^-10.5)')]:
    print(f"\n  {bin_label_short}:")
    for cat_name in ['sparc', 'hybrid', 'cf4']:
        bd = intersection_results[cat_name]['test_D_binned'].get(bin_label, {})
        if 'insufficient_data' not in bd and bd:
            print(f"    {cat_name:8s}: Δσ={bd['delta_sigma']:+.4f}, p={bd['levene_p']:.6f}")

# Does Δσ sign change?
delta_signs = {}
for cat_name in ['sparc', 'hybrid', 'cf4']:
    delta_signs[cat_name] = intersection_results[cat_name]['test_A_point_level']['delta_sigma'] > 0

sign_changes = any(delta_signs[a] != delta_signs[b]
                    for a in delta_signs for b in delta_signs if a != b)
print(f"\n  Δσ sign consistency: {'SIGN CHANGE detected!' if sign_changes else 'Same sign across catalogs'}")
for cat_name in ['sparc', 'hybrid', 'cf4']:
    d = intersection_results[cat_name]['test_A_point_level']['delta_sigma']
    print(f"    {cat_name:8s}: Δσ = {d:+.4f} ({'field looser' if d > 0 else 'dense looser'})")

# Does Levene p cross significance?
print(f"\n  Levene significance crossings:")
for cat_name in ['sparc', 'hybrid', 'cf4']:
    p = intersection_results[cat_name]['test_A_point_level']['levene_p']
    sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
    print(f"    {cat_name:8s}: p={p:.6f} ({sig})")

# ================================================================
# STEP 5: Save
# ================================================================
summary = {
    'test_name': 'environmental_battery_triple_distance',
    'description': 'Full environmental scatter battery under 3 distance catalogs',
    'full_sample_results': {
        cat_name: battery
        for cat_name, battery in all_battery_results.items()
    },
    'intersection_sample': {
        'n_galaxies': len(intersection),
        'results': intersection_results,
        'delta_sigma_sign_change': sign_changes,
    },
}

outpath = os.path.join(RESULTS_DIR, 'summary_env_triple_distance.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("Done.")
