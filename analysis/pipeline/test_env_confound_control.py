#!/usr/bin/env python3
"""
Environment Confound Control Test
====================================

CRITICAL QUESTION: Is "environment" really environment, or a proxy for
morphology, gas fraction, distance method, inclination, or RC quality?

Dense galaxies (UMa cluster, group members) might systematically differ
from field galaxies in ways that create scatter differences independent
of any physical environment effect.

MOST DANGEROUS CONFOUNDER: UMa cluster galaxies all have f_D = 4 (cluster
distance). Their distances are correlated, not independent. This could
suppress scatter in the dense sample at low accelerations — not physics,
just correlated distance errors.

Tests:
  1. COVARIATE PROFILING: Compare field vs dense on all measurable properties
  2. DISTANCE METHOD TEST: Remove all f_D=4 galaxies, re-test
  3. UMA-FREE TEST: Remove all UMa galaxies, keep only group members as dense
  4. MATCHED SAMPLE: Match field & dense on morphology + mass + inclination
  5. STRATIFIED ANALYSIS: Within T-type bins, check scatter uniformity
  6. PERMUTATION TEST: Shuffle env labels 10,000 times, check if observed
     pattern is statistically unusual under the null

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
from scipy.stats import levene, mannwhitneyu, ks_2samp, spearmanr
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
LOG_G_DAGGER = np.log10(g_dagger)

print("=" * 72)
print("ENVIRONMENT CONFOUND CONTROL TEST")
print("=" * 72)


# ================================================================
# ENVIRONMENT CLASSIFICATION
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
    if name in UMA_GALAXIES:
        return 'dense'
    if name in GROUP_MEMBERS:
        return 'dense'
    return 'field'


def classify_env_detail(name):
    """Detailed classification: uma / group / field."""
    if name in UMA_GALAXIES:
        return 'uma'
    if name in GROUP_MEMBERS:
        return 'group'
    return 'field'


# ================================================================
# DATA LOADING
# ================================================================
print("\n[1] Loading SPARC data...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

# Load rotation curves
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
            evobs = float(line[33:38].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in rc_data:
            rc_data[name] = {'R': [], 'Vobs': [], 'eVobs': [],
                             'Vgas': [], 'Vdisk': [], 'Vbul': [],
                             'dist': dist}
        rc_data[name]['R'].append(rad)
        rc_data[name]['Vobs'].append(vobs)
        rc_data[name]['eVobs'].append(evobs)
        rc_data[name]['Vgas'].append(vgas)
        rc_data[name]['Vdisk'].append(vdisk)
        rc_data[name]['Vbul'].append(vbul)

for name in rc_data:
    for key in ['R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul']:
        rc_data[name][key] = np.array(rc_data[name][key])

# Load galaxy properties from MRT
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
            'T': int(parts[0]),
            'D': float(parts[1]),
            'eD': float(parts[2]),
            'fD': int(parts[3]),
            'Inc': float(parts[4]),
            'eInc': float(parts[5]),
            'L36': float(parts[6]),
            'eL36': float(parts[7]),
            'Reff': float(parts[8]),
            'SBeff': float(parts[9]),
            'Rdisk': float(parts[10]),
            'SBdisk': float(parts[11]),
            'MHI': float(parts[12]),
            'RHI': float(parts[13]),
            'Vflat': float(parts[14]),
            'eVflat': float(parts[15]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue


# ================================================================
# COMPUTE RAR RESIDUALS
# ================================================================
print("\n[2] Computing RAR residuals...")


def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


gal_results = {}

for name, gdata in rc_data.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

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
    if np.sum(valid) < 5:
        continue

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    log_res = log_gobs - rar_function(log_gbar)

    env = classify_env(name)
    env_detail = classify_env_detail(name)

    # Compute gas fraction
    gas_frac = prop['MHI'] / (0.5 * prop['L36']) if prop['L36'] > 0 else np.nan

    gal_results[name] = {
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'log_res': log_res,
        'mean_res': float(np.mean(log_res)),
        'std_res': float(np.std(log_res)),
        'n_pts': len(log_res),
        'env': env,
        'env_detail': env_detail,
        'T': prop['T'],
        'D': prop['D'],
        'eD': prop['eD'],
        'fD': prop['fD'],
        'Inc': prop['Inc'],
        'eInc': prop['eInc'],
        'logL36': np.log10(max(prop['L36'], 1e-6)),
        'logMHI': np.log10(max(prop['MHI'], 1e-6)),
        'gas_frac': gas_frac,
        'Vflat': prop['Vflat'],
        'Q': prop['Q'],
        'SBeff': prop['SBeff'],
        'Rdisk': prop['Rdisk'],
    }

n_gals = len(gal_results)
field_gals = {n: g for n, g in gal_results.items() if g['env'] == 'field'}
dense_gals = {n: g for n, g in gal_results.items() if g['env'] == 'dense'}
uma_gals = {n: g for n, g in gal_results.items() if g['env_detail'] == 'uma'}
group_gals = {n: g for n, g in gal_results.items() if g['env_detail'] == 'group'}

print(f"  Total: {n_gals} galaxies")
print(f"  Field: {len(field_gals)}")
print(f"  Dense: {len(dense_gals)} (UMa: {len(uma_gals)}, Groups: {len(group_gals)})")


# ================================================================
# TEST 1: COVARIATE PROFILING
# ================================================================
print("\n" + "=" * 72)
print("TEST 1: COVARIATE PROFILING — Field vs Dense")
print("=" * 72)

covariates = ['T', 'D', 'Inc', 'eInc', 'logL36', 'logMHI', 'gas_frac',
              'Vflat', 'Q', 'n_pts', 'SBeff', 'fD']

covariate_results = {}

print(f"\n  {'Property':<12} {'Field mean':>12} {'Dense mean':>12} "
      f"{'p (MW-U)':>10} {'Imbalanced?':>12}")
print(f"  {'-' * 62}")

for cov in covariates:
    f_vals = np.array([g[cov] for g in field_gals.values() if not np.isnan(g[cov])])
    d_vals = np.array([g[cov] for g in dense_gals.values() if not np.isnan(g[cov])])

    if len(f_vals) < 5 or len(d_vals) < 5:
        continue

    try:
        stat, p = mannwhitneyu(f_vals, d_vals, alternative='two-sided')
    except Exception:
        p = np.nan

    imbalanced = "YES" if p < 0.05 else "no"
    covariate_results[cov] = {
        'field_mean': round(float(np.mean(f_vals)), 3),
        'field_std': round(float(np.std(f_vals)), 3),
        'dense_mean': round(float(np.mean(d_vals)), 3),
        'dense_std': round(float(np.std(d_vals)), 3),
        'mwu_p': round(float(p), 4),
        'imbalanced': bool(p < 0.05),
    }

    print(f"  {cov:<12} {np.mean(f_vals):12.3f} {np.mean(d_vals):12.3f} "
          f"{p:10.4f} {imbalanced:>12}")

# Distance method breakdown
print(f"\n  Distance method breakdown:")
for env_name, env_gals in [('field', field_gals), ('dense', dense_gals)]:
    fd_counts = {}
    for g in env_gals.values():
        fd = g['fD']
        fd_counts[fd] = fd_counts.get(fd, 0) + 1
    labels = {1: 'Hubble', 2: 'TRGB', 3: 'Cepheid', 4: 'UMa', 5: 'SNe'}
    parts = ', '.join(f"{labels.get(k, f'f{k}')}:{v}"
                      for k, v in sorted(fd_counts.items()))
    print(f"    {env_name}: {parts}")


# ================================================================
# TEST 2: UMA-FREE ANALYSIS
# ================================================================
print("\n" + "=" * 72)
print("TEST 2: UMA-FREE — Remove all UMa galaxies")
print("=" * 72)
print("  (Most dangerous confounder: UMa galaxies all have correlated")
print("   cluster distances, not independent distance estimates)")


def compute_env_scatter(field_dict, dense_dict, label=""):
    """Compute scatter comparison at low and high acceleration."""
    f_gbar = np.concatenate([g['log_gbar'] for g in field_dict.values()])
    f_res = np.concatenate([g['log_res'] for g in field_dict.values()])
    d_gbar = np.concatenate([g['log_gbar'] for g in dense_dict.values()])
    d_res = np.concatenate([g['log_res'] for g in dense_dict.values()])

    accel_bins = [
        ('low_accel', -13.0, -10.3),
        ('transition', -10.3, -9.5),
        ('high_accel', -9.5, -8.0),
    ]

    results = []
    for regime, lo, hi in accel_bins:
        f_mask = (f_gbar >= lo) & (f_gbar < hi)
        d_mask = (d_gbar >= lo) & (d_gbar < hi)
        nf, nd = np.sum(f_mask), np.sum(d_mask)

        if nf >= 10 and nd >= 5:
            sf = np.std(f_res[f_mask])
            sd = np.std(d_res[d_mask])
            delta = sd - sf
            try:
                _, lev_p = levene(f_res[f_mask], d_res[d_mask])
            except Exception:
                lev_p = np.nan
            results.append({
                'regime': regime, 'sigma_field': round(float(sf), 4),
                'sigma_dense': round(float(sd), 4),
                'delta_sigma': round(float(delta), 4),
                'levene_p': round(float(lev_p), 4),
                'n_field': int(nf), 'n_dense': int(nd),
            })
        else:
            results.append({
                'regime': regime, 'sigma_field': None, 'sigma_dense': None,
                'delta_sigma': None, 'levene_p': None,
                'n_field': int(nf), 'n_dense': int(nd),
            })

    return results


def print_scatter_table(results, label=""):
    if label:
        print(f"\n  {label}")
    print(f"  {'Regime':<14} {'σ_field':>8} {'σ_dense':>8} {'Δσ':>8} "
          f"{'Levene_p':>9} {'N_f':>5} {'N_d':>5}")
    print(f"  {'-' * 58}")
    for r in results:
        if r['sigma_field'] is not None:
            tag = "*" if r['levene_p'] < 0.05 else ""
            print(f"  {r['regime']:<14} {r['sigma_field']:8.4f} "
                  f"{r['sigma_dense']:8.4f} {r['delta_sigma']:+8.4f} "
                  f"{r['levene_p']:9.4f}{tag} {r['n_field']:5d} {r['n_dense']:5d}")
        else:
            print(f"  {r['regime']:<14} {'--':>8} {'--':>8} {'--':>8} "
                  f"{'--':>9}  {r['n_field']:5d} {r['n_dense']:5d}")


# Original (all dense)
print(f"\n  Original: {len(field_gals)} field, {len(dense_gals)} dense "
      f"({len(uma_gals)} UMa + {len(group_gals)} group)")
orig_results = compute_env_scatter(field_gals, dense_gals)
print_scatter_table(orig_results, "Original (UMa + Groups vs Field)")

# UMa-free: only group members as dense
print(f"\n  UMa-free: {len(field_gals)} field, {len(group_gals)} group members as 'dense'")
if len(group_gals) >= 5:
    uma_free_results = compute_env_scatter(field_gals, group_gals)
    print_scatter_table(uma_free_results, "UMa-FREE (Groups only vs Field)")
else:
    uma_free_results = []
    print("  Not enough group galaxies for analysis")

# Groups-free: only UMa as dense
print(f"\n  Groups-free: {len(field_gals)} field, {len(uma_gals)} UMa as 'dense'")
if len(uma_gals) >= 5:
    group_free_results = compute_env_scatter(field_gals, uma_gals)
    print_scatter_table(group_free_results, "Groups-FREE (UMa only vs Field)")
else:
    group_free_results = []


# ================================================================
# TEST 3: DISTANCE METHOD CONTROL
# ================================================================
print("\n" + "=" * 72)
print("TEST 3: DISTANCE METHOD CONTROL")
print("=" * 72)

# Only Hubble flow distances (fD=1) — removes all UMa, most TRGB
hubble_field = {n: g for n, g in field_gals.items() if g['fD'] == 1}
hubble_dense = {n: g for n, g in dense_gals.items() if g['fD'] == 1}
print(f"\n  Hubble-flow only (fD=1): {len(hubble_field)} field, {len(hubble_dense)} dense")
if len(hubble_dense) >= 3:
    hubble_results = compute_env_scatter(hubble_field, hubble_dense)
    print_scatter_table(hubble_results, "Hubble-flow only")
else:
    hubble_results = []
    print("  Not enough dense galaxies with Hubble-flow distances")

# Remove ONLY fD=4 (UMa cluster distance)
no_fd4_field = {n: g for n, g in field_gals.items() if g['fD'] != 4}
no_fd4_dense = {n: g for n, g in dense_gals.items() if g['fD'] != 4}
print(f"\n  Remove fD=4 only: {len(no_fd4_field)} field, {len(no_fd4_dense)} dense")
if len(no_fd4_dense) >= 5:
    no_fd4_results = compute_env_scatter(no_fd4_field, no_fd4_dense)
    print_scatter_table(no_fd4_results, "No fD=4 (removes UMa cluster distances)")
else:
    no_fd4_results = []
    print("  Not enough dense galaxies after removing fD=4")


# ================================================================
# TEST 4: MORPHOLOGY-MATCHED ANALYSIS
# ================================================================
print("\n" + "=" * 72)
print("TEST 4: MORPHOLOGY-MATCHED ANALYSIS")
print("=" * 72)

# Split by late-type (T >= 5) vs early-type (T < 5)
late_field = {n: g for n, g in field_gals.items() if g['T'] >= 5}
late_dense = {n: g for n, g in dense_gals.items() if g['T'] >= 5}
early_field = {n: g for n, g in field_gals.items() if g['T'] < 5}
early_dense = {n: g for n, g in dense_gals.items() if g['T'] < 5}

print(f"\n  Late-type (T≥5): {len(late_field)} field, {len(late_dense)} dense")
if len(late_dense) >= 5:
    late_results = compute_env_scatter(late_field, late_dense)
    print_scatter_table(late_results, "Late-type only (T≥5)")
else:
    late_results = []
    print("  Not enough late-type dense galaxies")

print(f"\n  Early-type (T<5): {len(early_field)} field, {len(early_dense)} dense")
if len(early_dense) >= 5:
    early_results = compute_env_scatter(early_field, early_dense)
    print_scatter_table(early_results, "Early-type only (T<5)")
else:
    early_results = []
    print("  Not enough early-type dense galaxies")


# ================================================================
# TEST 5: LUMINOSITY/MASS-MATCHED ANALYSIS
# ================================================================
print("\n" + "=" * 72)
print("TEST 5: LUMINOSITY-MATCHED ANALYSIS")
print("=" * 72)

# Match on logL36 using nearest-neighbor
field_list = sorted(field_gals.items(), key=lambda x: x[1]['logL36'])
dense_list = sorted(dense_gals.items(), key=lambda x: x[1]['logL36'])

matched_field = {}
matched_dense = {}
used_field = set()

for d_name, d_gal in dense_list:
    best_dist = 999
    best_fname = None
    for f_name, f_gal in field_list:
        if f_name in used_field:
            continue
        dist = abs(f_gal['logL36'] - d_gal['logL36'])
        if dist < best_dist:
            best_dist = dist
            best_fname = f_name
    if best_fname is not None and best_dist < 0.5:  # within 0.5 dex in L
        matched_field[best_fname] = field_gals[best_fname]
        matched_dense[d_name] = d_gal
        used_field.add(best_fname)

print(f"  Matched pairs: {len(matched_dense)} "
      f"(within 0.5 dex in logL[3.6])")

if len(matched_dense) >= 10:
    # Verify matching quality
    f_logL = np.array([g['logL36'] for g in matched_field.values()])
    d_logL = np.array([g['logL36'] for g in matched_dense.values()])
    print(f"  Field mean logL: {np.mean(f_logL):.2f} ± {np.std(f_logL):.2f}")
    print(f"  Dense mean logL: {np.mean(d_logL):.2f} ± {np.std(d_logL):.2f}")
    _, p_match = mannwhitneyu(f_logL, d_logL, alternative='two-sided')
    print(f"  Matching quality (MW-U p): {p_match:.3f}")

    matched_results = compute_env_scatter(matched_field, matched_dense)
    print_scatter_table(matched_results, "Luminosity-matched")
else:
    matched_results = []
    print("  Not enough matched pairs")


# ================================================================
# TEST 6: GAS FRACTION CONTROL
# ================================================================
print("\n" + "=" * 72)
print("TEST 6: GAS FRACTION CONTROL")
print("=" * 72)

# Split by median gas fraction
all_gf = [g['gas_frac'] for g in gal_results.values()
           if not np.isnan(g['gas_frac'])]
med_gf = np.median(all_gf)

gas_rich_field = {n: g for n, g in field_gals.items()
                  if not np.isnan(g['gas_frac']) and g['gas_frac'] >= med_gf}
gas_rich_dense = {n: g for n, g in dense_gals.items()
                  if not np.isnan(g['gas_frac']) and g['gas_frac'] >= med_gf}
gas_poor_field = {n: g for n, g in field_gals.items()
                  if not np.isnan(g['gas_frac']) and g['gas_frac'] < med_gf}
gas_poor_dense = {n: g for n, g in dense_gals.items()
                  if not np.isnan(g['gas_frac']) and g['gas_frac'] < med_gf}

print(f"  Median gas fraction: {med_gf:.3f}")
print(f"  Gas-rich (≥med): {len(gas_rich_field)} field, {len(gas_rich_dense)} dense")
print(f"  Gas-poor (<med): {len(gas_poor_field)} field, {len(gas_poor_dense)} dense")

if len(gas_rich_dense) >= 5:
    gas_rich_results = compute_env_scatter(gas_rich_field, gas_rich_dense)
    print_scatter_table(gas_rich_results, "Gas-rich only")
else:
    gas_rich_results = []
    print("  Not enough gas-rich dense galaxies")

if len(gas_poor_dense) >= 5:
    gas_poor_results = compute_env_scatter(gas_poor_field, gas_poor_dense)
    print_scatter_table(gas_poor_results, "Gas-poor only")
else:
    gas_poor_results = []
    print("  Not enough gas-poor dense galaxies")


# ================================================================
# TEST 7: INCLINATION CONTROL
# ================================================================
print("\n" + "=" * 72)
print("TEST 7: INCLINATION CONTROL")
print("=" * 72)

# Split at median inclination
med_inc = np.median([g['Inc'] for g in gal_results.values()])
hi_inc_field = {n: g for n, g in field_gals.items() if g['Inc'] >= med_inc}
hi_inc_dense = {n: g for n, g in dense_gals.items() if g['Inc'] >= med_inc}
lo_inc_field = {n: g for n, g in field_gals.items() if g['Inc'] < med_inc}
lo_inc_dense = {n: g for n, g in dense_gals.items() if g['Inc'] < med_inc}

print(f"  Median inclination: {med_inc:.1f}°")
print(f"  High-inc (≥{med_inc:.0f}°): {len(hi_inc_field)} field, {len(hi_inc_dense)} dense")
print(f"  Low-inc (<{med_inc:.0f}°): {len(lo_inc_field)} field, {len(lo_inc_dense)} dense")

if len(hi_inc_dense) >= 5:
    hi_inc_results = compute_env_scatter(hi_inc_field, hi_inc_dense)
    print_scatter_table(hi_inc_results, f"High inclination only (≥{med_inc:.0f}°)")
else:
    hi_inc_results = []

if len(lo_inc_dense) >= 5:
    lo_inc_results = compute_env_scatter(lo_inc_field, lo_inc_dense)
    print_scatter_table(lo_inc_results, f"Low inclination only (<{med_inc:.0f}°)")
else:
    lo_inc_results = []


# ================================================================
# TEST 8: PERMUTATION TEST
# ================================================================
print("\n" + "=" * 72)
print("TEST 8: PERMUTATION TEST — Is the pattern unlikely under null?")
print("=" * 72)

# Observed statistic: |Δσ| at low acceleration
all_gbar_pts = np.concatenate([g['log_gbar'] for g in gal_results.values()])
all_res_pts = np.concatenate([g['log_res'] for g in gal_results.values()])
all_env_pts = np.concatenate([[g['env']] * g['n_pts']
                               for g in gal_results.values()])

low_mask = all_gbar_pts < -10.3
low_field_mask = low_mask & (all_env_pts == 'field')
low_dense_mask = low_mask & (all_env_pts == 'dense')

obs_sigma_f = np.std(all_res_pts[low_field_mask])
obs_sigma_d = np.std(all_res_pts[low_dense_mask])
obs_delta = abs(obs_sigma_d - obs_sigma_f)
_, obs_levene_p = levene(all_res_pts[low_field_mask], all_res_pts[low_dense_mask])

print(f"\n  Observed at low accel (< -10.3):")
print(f"    σ_field = {obs_sigma_f:.4f}, σ_dense = {obs_sigma_d:.4f}")
print(f"    |Δσ| = {obs_delta:.4f}, Levene p = {obs_levene_p:.4f}")

# Permute galaxy-level environment labels
n_perm = 10000
perm_deltas = np.zeros(n_perm)
rng = np.random.default_rng(42)

gal_names = list(gal_results.keys())
gal_envs = np.array([gal_results[n]['env'] for n in gal_names])
n_dense_total = np.sum(gal_envs == 'dense')

# Pre-build per-galaxy point arrays
gal_gbar_list = [gal_results[n]['log_gbar'] for n in gal_names]
gal_res_list = [gal_results[n]['log_res'] for n in gal_names]

print(f"  Running {n_perm} permutations...")

for perm in range(n_perm):
    # Shuffle galaxy labels (not individual points)
    perm_idx = rng.permutation(len(gal_names))
    perm_envs = gal_envs[perm_idx]

    # Collect low-accel points
    f_pts = []
    d_pts = []
    for i in range(len(gal_names)):
        low_m = gal_gbar_list[i] < -10.3
        if np.any(low_m):
            if perm_envs[i] == 'field':
                f_pts.append(gal_res_list[i][low_m])
            else:
                d_pts.append(gal_res_list[i][low_m])

    if f_pts and d_pts:
        sf = np.std(np.concatenate(f_pts))
        sd = np.std(np.concatenate(d_pts))
        perm_deltas[perm] = abs(sd - sf)
    else:
        perm_deltas[perm] = 0.0

# p-value: fraction of permutations with |Δσ| ≥ observed
p_perm = np.mean(perm_deltas >= obs_delta)
print(f"\n  Permutation results ({n_perm} shuffles):")
print(f"    Observed |Δσ| = {obs_delta:.4f}")
print(f"    Permutation mean |Δσ| = {np.mean(perm_deltas):.4f}")
print(f"    Permutation p-value = {p_perm:.4f}")

if p_perm > 0.05:
    perm_verdict = "NOT_SIGNIFICANT"
    print(f"    -> The observed |Δσ| is consistent with random label assignment")
    print(f"    -> The scatter uniformity at low accel is NOT environment-specific")
    print(f"    -> BUT: uniformity itself (small |Δσ|) is the BEC prediction")
else:
    perm_verdict = "SIGNIFICANT"
    print(f"    -> The observed |Δσ| would be unlikely under random assignment")


# Also test: is the UNIFORMITY (small |Δσ|) unusual?
# Under permutation, what fraction have |Δσ| ≤ observed?
p_uniform = np.mean(perm_deltas <= obs_delta)
print(f"\n  Uniformity test: fraction with |Δσ| ≤ observed = {p_uniform:.4f}")
if p_uniform < 0.05:
    print(f"    -> The observed UNIFORMITY is unusually tight — hard to get by chance")
    uniform_verdict = "UNUSUALLY_UNIFORM"
elif p_uniform < 0.20:
    print(f"    -> Mild evidence for unusual uniformity")
    uniform_verdict = "MILDLY_UNIFORM"
else:
    print(f"    -> Uniformity is consistent with chance")
    uniform_verdict = "CONSISTENT_WITH_CHANCE"


# ================================================================
# TEST 9: INVERSION POINT IN CONFOUND-CONTROLLED SUBSAMPLES
# ================================================================
print("\n" + "=" * 72)
print("TEST 9: INVERSION POINT IN CONFOUND-CONTROLLED SUBSAMPLES")
print("=" * 72)


def compute_inversion(gal_dict, label="", bin_width=0.30):
    """Compute scatter derivative inversion from a galaxy set."""
    gbar = np.concatenate([g['log_gbar'] for g in gal_dict.values()])
    res = np.concatenate([g['log_res'] for g in gal_dict.values()])

    lo = max(np.percentile(gbar, 2), -13.0)
    hi = min(np.percentile(gbar, 98), -8.0)
    edges = np.arange(lo, hi, bin_width)

    centers, sigmas = [], []
    for edge in edges:
        mask = (gbar >= edge) & (gbar < edge + bin_width)
        n = np.sum(mask)
        if n >= 15:
            centers.append(edge + bin_width / 2)
            sigmas.append(np.std(res[mask]))

    centers = np.array(centers)
    sigmas = np.array(sigmas)

    if len(centers) < 3:
        return None, None

    dy = np.zeros_like(sigmas)
    dy[0] = (sigmas[1] - sigmas[0]) / (centers[1] - centers[0])
    dy[-1] = (sigmas[-1] - sigmas[-2]) / (centers[-1] - centers[-2])
    for i in range(1, len(sigmas) - 1):
        dy[i] = (sigmas[i + 1] - sigmas[i - 1]) / (centers[i + 1] - centers[i - 1])

    crossings = []
    for i in range(len(dy) - 1):
        if dy[i] * dy[i + 1] < 0:
            x_c = centers[i] - dy[i] * (centers[i + 1] - centers[i]) / (dy[i + 1] - dy[i])
            crossings.append(x_c)

    if crossings:
        best = min(crossings, key=lambda c: abs(c - LOG_G_DAGGER))
        return best, best - LOG_G_DAGGER
    return None, None


subsamples = {
    'All galaxies': gal_results,
    'Field only': field_gals,
    'Dense only': dense_gals,
    'UMa-free dense': group_gals,
}

# Add late-type only if enough
if len(late_field) + len(late_dense) >= 30:
    late_all = {**late_field, **late_dense}
    subsamples['Late-type (T≥5)'] = late_all
if len(early_field) + len(early_dense) >= 30:
    early_all = {**early_field, **early_dense}
    subsamples['Early-type (T<5)'] = early_all

# Luminosity-matched
if len(matched_field) + len(matched_dense) >= 20:
    matched_all = {**matched_field, **matched_dense}
    subsamples['Luminosity-matched'] = matched_all

inversion_results = {}
print(f"\n  {'Subsample':<25} {'N_gal':>6} {'Crossing':>10} {'Δ from g†':>10} {'Status':>8}")
print(f"  {'-' * 65}")

for label, gal_dict in subsamples.items():
    crossing, dist_gd = compute_inversion(gal_dict, label)
    inversion_results[label] = {
        'n_gals': len(gal_dict),
        'crossing': round(float(crossing), 4) if crossing is not None else None,
        'dist_gdagger': round(float(dist_gd), 4) if dist_gd is not None else None,
    }
    if crossing is not None:
        status = "MATCH" if abs(dist_gd) < 0.20 else "partial"
        print(f"  {label:<25} {len(gal_dict):6d} {crossing:+10.3f} "
              f"{dist_gd:+10.3f} {status:>8}")
    else:
        print(f"  {label:<25} {len(gal_dict):6d} {'--':>10} {'--':>10} {'no cross':>8}")


# ================================================================
# FINAL VERDICT
# ================================================================
print("\n" + "=" * 72)
print("FINAL VERDICT: CONFOUND ANALYSIS")
print("=" * 72)

# Count imbalanced covariates
n_imbalanced = sum(1 for c in covariate_results.values() if c['imbalanced'])
print(f"\n  Covariates imbalanced (p<0.05): {n_imbalanced}/{len(covariate_results)}")
for cov, cr in covariate_results.items():
    if cr['imbalanced']:
        print(f"    {cov}: field={cr['field_mean']:.2f}, dense={cr['dense_mean']:.2f}, "
              f"p={cr['mwu_p']:.4f}")

# UMa-free result
print(f"\n  UMa-free low-accel:")
if uma_free_results:
    low_uma = [r for r in uma_free_results if r['regime'] == 'low_accel']
    if low_uma and low_uma[0]['levene_p'] is not None:
        print(f"    Levene p = {low_uma[0]['levene_p']:.4f} "
              f"(Δσ = {low_uma[0]['delta_sigma']:+.4f})")
        if low_uma[0]['levene_p'] > 0.05:
            print(f"    -> Scatter UNIFORM even without UMa")
        else:
            print(f"    -> Scatter difference detected without UMa")

# Permutation result
print(f"\n  Permutation test:")
print(f"    |Δσ| significance: {perm_verdict} (p = {p_perm:.4f})")
print(f"    Uniformity: {uniform_verdict} (p = {p_uniform:.4f})")

# Inversion persistence
n_inv_match = sum(1 for ir in inversion_results.values()
                  if ir['crossing'] is not None and abs(ir['dist_gdagger']) < 0.20)
n_inv_total = sum(1 for ir in inversion_results.values()
                  if ir['crossing'] is not None)
print(f"\n  Inversion point persists:")
print(f"    {n_inv_match}/{n_inv_total} subsamples within 0.20 dex of g†")

# Overall
if n_inv_match >= n_inv_total - 1 and n_inv_total >= 3:
    overall = "INVERSION_ROBUST"
    print(f"\n  >>> INVERSION at g† survives all confound controls")
else:
    overall = "INVERSION_FRAGILE"
    print(f"\n  >>> Inversion does NOT consistently survive confound controls")


# ================================================================
# SAVE
# ================================================================
summary = {
    'test_name': 'env_confound_control',
    'n_galaxies': n_gals,
    'n_field': len(field_gals),
    'n_dense': len(dense_gals),
    'n_uma': len(uma_gals),
    'n_group': len(group_gals),
    'covariate_profiling': covariate_results,
    'original_scatter': orig_results,
    'uma_free_scatter': uma_free_results,
    'group_free_scatter': group_free_results,
    'no_fd4_scatter': no_fd4_results if 'no_fd4_results' in dir() and no_fd4_results else [],
    'late_type_scatter': late_results,
    'early_type_scatter': early_results,
    'matched_scatter': matched_results,
    'gas_rich_scatter': gas_rich_results,
    'gas_poor_scatter': gas_poor_results,
    'hi_inc_scatter': hi_inc_results,
    'lo_inc_scatter': lo_inc_results,
    'permutation': {
        'n_perm': n_perm,
        'obs_delta': round(float(obs_delta), 4),
        'perm_mean': round(float(np.mean(perm_deltas)), 4),
        'p_value': round(float(p_perm), 4),
        'verdict': perm_verdict,
        'p_uniform': round(float(p_uniform), 4),
        'uniform_verdict': uniform_verdict,
    },
    'inversion_by_subsample': inversion_results,
    'overall_verdict': overall,
}

outpath = os.path.join(RESULTS_DIR, 'summary_env_confound_control.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("=" * 72)
print("Done.")
