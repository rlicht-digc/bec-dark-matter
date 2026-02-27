#!/usr/bin/env python3
"""
Propensity-Score Matched Environment Test
==========================================

The UMa cancellation problem: low-accel scatter uniformity between field
and dense is partly a cancellation (UMa suppresses scatter, groups enhance).
This test addresses this by propensity-score matching.

Method:
  1. Compute propensity scores: P(dense | covariates) using logistic regression
     on distance, inclination error, quality, distance flag, morphology, luminosity
  2. Match each dense galaxy to a field galaxy with similar propensity score
  3. Test scatter uniformity in the matched sample
  4. Also test: remove UMa entirely and match remaining dense (groups) to field
  5. Test inversion point in propensity-matched subsamples

If scatter uniformity persists after propensity matching (removing all
covariate imbalances), then the uniformity is genuine, not an artifact
of confounders.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
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

g_dagger = 1.20e-10
kpc_m = 3.086e19
LOG_G_DAGGER = np.log10(g_dagger)

print("=" * 72)
print("PROPENSITY-SCORE MATCHED ENVIRONMENT TEST")
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

def env_detail(name):
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
        # Add HI mass if available
        try:
            sparc_props[name]['MHI'] = float(parts[8])
        except (ValueError, IndexError):
            sparc_props[name]['MHI'] = None
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

    gbar = 10**log_gbar
    rar_pred = np.log10(gbar / (1 - np.exp(-np.sqrt(gbar / g_dagger))))
    resid = log_gobs - rar_pred

    gal_data[name] = {
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'resid': resid,
        'env': classify_env(name),
        'env_detail': env_detail(name),
        'n_pts': len(log_gbar),
    }

gal_names = sorted(gal_data.keys())
n_gals = len(gal_names)
field_names = [n for n in gal_names if gal_data[n]['env'] == 'field']
dense_names = [n for n in gal_names if gal_data[n]['env'] == 'dense']
uma_names = [n for n in gal_names if gal_data[n]['env_detail'] == 'uma']
group_names = [n for n in gal_names if gal_data[n]['env_detail'] == 'group']

print(f"  {n_gals} galaxies: {len(field_names)} field, {len(dense_names)} dense")
print(f"    Dense breakdown: {len(uma_names)} UMa, {len(group_names)} group")


# ================================================================
# LOGISTIC REGRESSION PROPENSITY SCORES (manual implementation)
# ================================================================
print("\n" + "=" * 72)
print("PROPENSITY SCORE ESTIMATION")
print("=" * 72)

# Covariates: D, eInc, Q, fD, T, logL, Inc
def get_covariates(name):
    """Return covariate vector for propensity model."""
    p = sparc_props[name]
    return np.array([
        np.log10(p['D']),     # log distance (most imbalanced)
        p['eInc'],            # inclination error
        p['Q'],               # quality flag
        float(p['fD'] == 4),  # UMa cluster distance indicator
        p['T'],               # morphology
        np.log10(max(p['L36'], 0.01)),  # log luminosity
        p['Inc'],             # inclination
    ])

# Build feature matrix
all_names_for_ps = field_names + dense_names
X = np.array([get_covariates(n) for n in all_names_for_ps])
y = np.array([0]*len(field_names) + [1]*len(dense_names))

# Standardize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1
X_norm = (X - X_mean) / X_std

# Simple logistic regression via iteratively reweighted least squares
# P(y=1|X) = sigmoid(X @ beta)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

# Add intercept
X_aug = np.column_stack([np.ones(len(X_norm)), X_norm])
beta = np.zeros(X_aug.shape[1])

for iteration in range(100):
    p_hat = sigmoid(X_aug @ beta)
    p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)
    W_diag = p_hat * (1 - p_hat)
    W = np.diag(W_diag)
    z = X_aug @ beta + (y - p_hat) / W_diag
    try:
        # Ridge regularization to prevent singularity
        XtWX = X_aug.T @ W @ X_aug + 0.01 * np.eye(X_aug.shape[1])
        beta = np.linalg.solve(XtWX, X_aug.T @ W @ z)
    except np.linalg.LinAlgError:
        break

propensity = sigmoid(X_aug @ beta)

ps_dict = {}
for i, name in enumerate(all_names_for_ps):
    ps_dict[name] = propensity[i]

field_ps = np.array([ps_dict[n] for n in field_names])
dense_ps = np.array([ps_dict[n] for n in dense_names])

print(f"  Propensity scores:")
print(f"    Field: mean = {field_ps.mean():.3f}, range [{field_ps.min():.3f}, {field_ps.max():.3f}]")
print(f"    Dense: mean = {dense_ps.mean():.3f}, range [{dense_ps.min():.3f}, {dense_ps.max():.3f}]")
print(f"    Overlap region: [{max(field_ps.min(), dense_ps.min()):.3f}, "
      f"{min(field_ps.max(), dense_ps.max()):.3f}]")


# ================================================================
# NEAREST-NEIGHBOR MATCHING (without replacement)
# ================================================================
print("\n" + "=" * 72)
print("TEST 1: PROPENSITY-MATCHED (ALL DENSE vs FIELD)")
print("=" * 72)

def match_nearest(dense_list, field_list, ps, caliper=0.25):
    """Match each dense galaxy to nearest field galaxy by propensity score."""
    matched_pairs = []
    used_field = set()

    # Sort dense by propensity (match hardest-to-match first)
    dense_sorted = sorted(dense_list, key=lambda n: abs(ps[n] - 0.5), reverse=True)

    for d_name in dense_sorted:
        best_dist = float('inf')
        best_match = None
        for f_name in field_list:
            if f_name in used_field:
                continue
            dist = abs(ps[d_name] - ps[f_name])
            if dist < best_dist and dist < caliper:
                best_dist = dist
                best_match = f_name
        if best_match is not None:
            matched_pairs.append((d_name, best_match))
            used_field.add(best_match)

    return matched_pairs

pairs_all = match_nearest(dense_names, field_names, ps_dict, caliper=0.50)
print(f"  Matched pairs: {len(pairs_all)} / {len(dense_names)} dense galaxies")

if len(pairs_all) > 0:
    # Check covariate balance after matching
    matched_dense = [p[0] for p in pairs_all]
    matched_field = [p[1] for p in pairs_all]

    props_check = ['D', 'eInc', 'Q', 'T', 'Inc']
    print(f"\n  Covariate balance after matching:")
    print(f"  {'Property':<12} {'Field':>10} {'Dense':>10} {'Δ':>10}")
    for prop_name in props_check:
        f_vals = [sparc_props[n][prop_name] for n in matched_field]
        d_vals = [sparc_props[n][prop_name] for n in matched_dense]
        print(f"  {prop_name:<12} {np.mean(f_vals):>10.2f} {np.mean(d_vals):>10.2f} "
              f"{np.mean(d_vals) - np.mean(f_vals):>+10.2f}")

    # Compute scatter in acceleration bins
    accel_bins = [
        ('low_accel', -13.0, -10.3),
        ('transition', -10.3, -9.5),
        ('high_accel', -9.5, -8.0),
    ]

    print(f"\n  Scatter comparison (propensity-matched):")
    print(f"  {'Regime':<14} {'σ_field':>8} {'σ_dense':>8} {'Δσ':>8} {'Levene p':>10} {'N_f':>6} {'N_d':>6}")
    print(f"  " + "-" * 60)

    matched_scatter = []
    for regime, lo, hi in accel_bins:
        f_resid = []
        d_resid = []
        for d_name, f_name in pairs_all:
            gd = gal_data[d_name]
            gf = gal_data[f_name]
            mask_d = (gd['log_gbar'] >= lo) & (gd['log_gbar'] < hi)
            mask_f = (gf['log_gbar'] >= lo) & (gf['log_gbar'] < hi)
            f_resid.extend(gf['resid'][mask_f].tolist())
            d_resid.extend(gd['resid'][mask_d].tolist())

        f_resid = np.array(f_resid)
        d_resid = np.array(d_resid)

        if len(f_resid) >= 10 and len(d_resid) >= 10:
            sf = np.std(f_resid)
            sd = np.std(d_resid)
            stat, pval = levene(f_resid, d_resid)
            ds = sd - sf
            flag = '*' if pval < 0.05 else ''
            print(f"  {regime:<14} {sf:>8.4f} {sd:>8.4f} {ds:>+8.4f} {pval:>9.4f}{flag} {len(f_resid):>6} {len(d_resid):>6}")
            matched_scatter.append({
                'regime': regime, 'sigma_field': float(sf), 'sigma_dense': float(sd),
                'delta_sigma': float(ds), 'levene_p': float(pval),
                'n_field': len(f_resid), 'n_dense': len(d_resid),
            })
        else:
            print(f"  {regime:<14} {'--':>8} {'--':>8} {'--':>8} {'--':>10} {len(f_resid):>6} {len(d_resid):>6}")
            matched_scatter.append({
                'regime': regime, 'sigma_field': None, 'sigma_dense': None,
                'delta_sigma': None, 'levene_p': None,
                'n_field': len(f_resid), 'n_dense': len(d_resid),
            })


# ================================================================
# TEST 2: UMa-FREE PROPENSITY MATCHING (groups only as dense)
# ================================================================
print("\n" + "=" * 72)
print("TEST 2: UMA-FREE PROPENSITY MATCHING (groups only)")
print("=" * 72)

pairs_no_uma = match_nearest(group_names, field_names, ps_dict, caliper=0.50)
print(f"  Matched pairs: {len(pairs_no_uma)} / {len(group_names)} group galaxies")

uma_free_scatter = []
if len(pairs_no_uma) > 0:
    matched_groups = [p[0] for p in pairs_no_uma]
    matched_field2 = [p[1] for p in pairs_no_uma]

    print(f"\n  Covariate balance (UMa-free matching):")
    print(f"  {'Property':<12} {'Field':>10} {'Group':>10} {'Δ':>10}")
    for prop_name in props_check:
        f_vals = [sparc_props[n][prop_name] for n in matched_field2]
        d_vals = [sparc_props[n][prop_name] for n in matched_groups]
        print(f"  {prop_name:<12} {np.mean(f_vals):>10.2f} {np.mean(d_vals):>10.2f} "
              f"{np.mean(d_vals) - np.mean(f_vals):>+10.2f}")

    print(f"\n  Scatter comparison (UMa-free propensity-matched):")
    print(f"  {'Regime':<14} {'σ_field':>8} {'σ_group':>8} {'Δσ':>8} {'Levene p':>10} {'N_f':>6} {'N_g':>6}")
    print(f"  " + "-" * 60)

    for regime, lo, hi in accel_bins:
        f_resid = []
        d_resid = []
        for d_name, f_name in pairs_no_uma:
            gd = gal_data[d_name]
            gf = gal_data[f_name]
            mask_d = (gd['log_gbar'] >= lo) & (gd['log_gbar'] < hi)
            mask_f = (gf['log_gbar'] >= lo) & (gf['log_gbar'] < hi)
            f_resid.extend(gf['resid'][mask_f].tolist())
            d_resid.extend(gd['resid'][mask_d].tolist())

        f_resid = np.array(f_resid)
        d_resid = np.array(d_resid)

        if len(f_resid) >= 10 and len(d_resid) >= 10:
            sf = np.std(f_resid)
            sd = np.std(d_resid)
            stat, pval = levene(f_resid, d_resid)
            ds = sd - sf
            flag = '*' if pval < 0.05 else ''
            print(f"  {regime:<14} {sf:>8.4f} {sd:>8.4f} {ds:>+8.4f} {pval:>9.4f}{flag} {len(f_resid):>6} {len(d_resid):>6}")
            uma_free_scatter.append({
                'regime': regime, 'sigma_field': float(sf), 'sigma_dense': float(sd),
                'delta_sigma': float(ds), 'levene_p': float(pval),
                'n_field': len(f_resid), 'n_dense': len(d_resid),
            })
        else:
            print(f"  {regime:<14} {'--':>8} {'--':>8} {'--':>8} {'--':>10} {len(f_resid):>6} {len(d_resid):>6}")
            uma_free_scatter.append({
                'regime': regime, 'sigma_field': None, 'sigma_dense': None,
                'delta_sigma': None, 'levene_p': None,
                'n_field': len(f_resid), 'n_dense': len(d_resid),
            })


# ================================================================
# TEST 3: MAHALANOBIS DISTANCE MATCHING (alternative to propensity)
# ================================================================
print("\n" + "=" * 72)
print("TEST 3: MAHALANOBIS DISTANCE MATCHING")
print("=" * 72)
print("  (Direct covariate matching without propensity model)")

def mahalanobis_match(dense_list, field_list, caliper_std=2.0):
    """Match using Mahalanobis distance on covariates directly."""
    cov_names = ['D', 'eInc', 'Q', 'T', 'Inc']

    dense_covs = np.array([[sparc_props[n][c] for c in cov_names] for n in dense_list])
    field_covs = np.array([[sparc_props[n][c] for c in cov_names] for n in field_list])

    # Compute covariance matrix from combined sample
    all_covs = np.vstack([dense_covs, field_covs])
    cov_matrix = np.cov(all_covs.T)
    try:
        cov_inv = np.linalg.inv(cov_matrix + 0.01 * np.eye(len(cov_names)))
    except np.linalg.LinAlgError:
        cov_inv = np.eye(len(cov_names))

    matched_pairs = []
    used_field = set()

    for i, d_name in enumerate(dense_list):
        best_dist = float('inf')
        best_j = None
        for j, f_name in enumerate(field_list):
            if f_name in used_field:
                continue
            diff = dense_covs[i] - field_covs[j]
            m_dist = np.sqrt(diff @ cov_inv @ diff)
            if m_dist < best_dist and m_dist < caliper_std:
                best_dist = m_dist
                best_j = j
                best_match = f_name
        if best_j is not None:
            matched_pairs.append((d_name, best_match))
            used_field.add(best_match)

    return matched_pairs

pairs_maha = mahalanobis_match(dense_names, field_names, caliper_std=3.0)
print(f"  Matched pairs: {len(pairs_maha)} / {len(dense_names)} dense galaxies")

maha_scatter = []
if len(pairs_maha) > 0:
    matched_dense_m = [p[0] for p in pairs_maha]
    matched_field_m = [p[1] for p in pairs_maha]

    print(f"\n  Covariate balance (Mahalanobis-matched):")
    print(f"  {'Property':<12} {'Field':>10} {'Dense':>10} {'Δ':>10}")
    for prop_name in props_check:
        f_vals = [sparc_props[n][prop_name] for n in matched_field_m]
        d_vals = [sparc_props[n][prop_name] for n in matched_dense_m]
        print(f"  {prop_name:<12} {np.mean(f_vals):>10.2f} {np.mean(d_vals):>10.2f} "
              f"{np.mean(d_vals) - np.mean(f_vals):>+10.2f}")

    print(f"\n  Scatter comparison (Mahalanobis-matched):")
    print(f"  {'Regime':<14} {'σ_field':>8} {'σ_dense':>8} {'Δσ':>8} {'Levene p':>10} {'N_f':>6} {'N_d':>6}")
    print(f"  " + "-" * 60)

    for regime, lo, hi in accel_bins:
        f_resid = []
        d_resid = []
        for d_name, f_name in pairs_maha:
            gd = gal_data[d_name]
            gf = gal_data[f_name]
            mask_d = (gd['log_gbar'] >= lo) & (gd['log_gbar'] < hi)
            mask_f = (gf['log_gbar'] >= lo) & (gf['log_gbar'] < hi)
            f_resid.extend(gf['resid'][mask_f].tolist())
            d_resid.extend(gd['resid'][mask_d].tolist())

        f_resid = np.array(f_resid)
        d_resid = np.array(d_resid)

        if len(f_resid) >= 10 and len(d_resid) >= 10:
            sf = np.std(f_resid)
            sd = np.std(d_resid)
            stat, pval = levene(f_resid, d_resid)
            ds = sd - sf
            flag = '*' if pval < 0.05 else ''
            print(f"  {regime:<14} {sf:>8.4f} {sd:>8.4f} {ds:>+8.4f} {pval:>9.4f}{flag} {len(f_resid):>6} {len(d_resid):>6}")
            maha_scatter.append({
                'regime': regime, 'sigma_field': float(sf), 'sigma_dense': float(sd),
                'delta_sigma': float(ds), 'levene_p': float(pval),
                'n_field': len(f_resid), 'n_dense': len(d_resid),
            })
        else:
            print(f"  {regime:<14} {'--':>8} {'--':>8} {'--':>8} {'--':>10} {len(f_resid):>6} {len(d_resid):>6}")
            maha_scatter.append({
                'regime': regime, 'sigma_field': None, 'sigma_dense': None,
                'delta_sigma': None, 'levene_p': None,
                'n_field': len(f_resid), 'n_dense': len(d_resid),
            })


# ================================================================
# TEST 4: INVERSION POINT IN PROPENSITY-MATCHED SUBSAMPLES
# ================================================================
print("\n" + "=" * 72)
print("TEST 4: INVERSION POINT IN MATCHED SUBSAMPLES")
print("=" * 72)

def find_inversion(log_gbar_arr, log_gobs_arr, bin_width=0.30, offset=0.0):
    gbar = 10**log_gbar_arr
    rar_pred = np.log10(gbar / (1 - np.exp(-np.sqrt(gbar / g_dagger))))
    resid = log_gobs_arr - rar_pred

    lo = log_gbar_arr.min() + offset
    hi = log_gbar_arr.max()
    edges = np.arange(lo, hi + bin_width, bin_width)
    if len(edges) < 3:
        return None

    centers, sigmas = [], []
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

    for j in range(len(dsigma) - 1):
        if dsigma[j] > 0 and dsigma[j+1] < 0:
            x0, x1 = dcenter[j], dcenter[j+1]
            y0, y1 = dsigma[j], dsigma[j+1]
            return x0 - y0 * (x1 - x0) / (y1 - y0)
    return None

inversion_results = {}

# Full propensity-matched
if len(pairs_all) > 0:
    matched_names = [p[0] for p in pairs_all] + [p[1] for p in pairs_all]
    lg_m = np.concatenate([gal_data[n]['log_gbar'] for n in matched_names])
    lo_m = np.concatenate([gal_data[n]['log_gobs'] for n in matched_names])
    inv_m = find_inversion(lg_m, lo_m)
    if inv_m is not None:
        print(f"  Propensity-matched (all):  crossing = {inv_m:.4f}  (Δ = {inv_m - LOG_G_DAGGER:+.4f})")
        inversion_results['propensity_all'] = {
            'n_gals': len(matched_names), 'crossing': float(inv_m),
            'dist_gdagger': float(inv_m - LOG_G_DAGGER),
        }

# UMa-free propensity-matched
if len(pairs_no_uma) > 0:
    matched_names2 = [p[0] for p in pairs_no_uma] + [p[1] for p in pairs_no_uma]
    lg_m2 = np.concatenate([gal_data[n]['log_gbar'] for n in matched_names2])
    lo_m2 = np.concatenate([gal_data[n]['log_gobs'] for n in matched_names2])
    inv_m2 = find_inversion(lg_m2, lo_m2)
    if inv_m2 is not None:
        print(f"  UMa-free matched:         crossing = {inv_m2:.4f}  (Δ = {inv_m2 - LOG_G_DAGGER:+.4f})")
        inversion_results['uma_free_matched'] = {
            'n_gals': len(matched_names2), 'crossing': float(inv_m2),
            'dist_gdagger': float(inv_m2 - LOG_G_DAGGER),
        }

# Mahalanobis-matched
if len(pairs_maha) > 0:
    matched_names3 = [p[0] for p in pairs_maha] + [p[1] for p in pairs_maha]
    lg_m3 = np.concatenate([gal_data[n]['log_gbar'] for n in matched_names3])
    lo_m3 = np.concatenate([gal_data[n]['log_gobs'] for n in matched_names3])
    inv_m3 = find_inversion(lg_m3, lo_m3)
    if inv_m3 is not None:
        print(f"  Mahalanobis-matched:      crossing = {inv_m3:.4f}  (Δ = {inv_m3 - LOG_G_DAGGER:+.4f})")
        inversion_results['mahalanobis'] = {
            'n_gals': len(matched_names3), 'crossing': float(inv_m3),
            'dist_gdagger': float(inv_m3 - LOG_G_DAGGER),
        }

# Matched field-only (the field galaxies selected by matching)
if len(pairs_all) > 0:
    matched_field_only = [p[1] for p in pairs_all]
    lg_fo = np.concatenate([gal_data[n]['log_gbar'] for n in matched_field_only])
    lo_fo = np.concatenate([gal_data[n]['log_gobs'] for n in matched_field_only])
    inv_fo = find_inversion(lg_fo, lo_fo)
    if inv_fo is not None:
        print(f"  Matched field only:       crossing = {inv_fo:.4f}  (Δ = {inv_fo - LOG_G_DAGGER:+.4f})")
        inversion_results['matched_field'] = {
            'n_gals': len(matched_field_only), 'crossing': float(inv_fo),
            'dist_gdagger': float(inv_fo - LOG_G_DAGGER),
        }

# Matched dense-only
if len(pairs_all) > 0:
    matched_dense_only = [p[0] for p in pairs_all]
    lg_do = np.concatenate([gal_data[n]['log_gbar'] for n in matched_dense_only])
    lo_do = np.concatenate([gal_data[n]['log_gobs'] for n in matched_dense_only])
    inv_do = find_inversion(lg_do, lo_do)
    if inv_do is not None:
        print(f"  Matched dense only:       crossing = {inv_do:.4f}  (Δ = {inv_do - LOG_G_DAGGER:+.4f})")
        inversion_results['matched_dense'] = {
            'n_gals': len(matched_dense_only), 'crossing': float(inv_do),
            'dist_gdagger': float(inv_do - LOG_G_DAGGER),
        }


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

# Determine verdicts
low_accel_uniform = False
if matched_scatter:
    low_entry = next((s for s in matched_scatter if s['regime'] == 'low_accel'), None)
    if low_entry and low_entry['levene_p'] is not None:
        low_accel_uniform = low_entry['levene_p'] > 0.05

uma_free_uniform = False
if uma_free_scatter:
    low_entry_uf = next((s for s in uma_free_scatter if s['regime'] == 'low_accel'), None)
    if low_entry_uf and low_entry_uf['levene_p'] is not None:
        uma_free_uniform = low_entry_uf['levene_p'] > 0.05

n_inv_match = sum(1 for v in inversion_results.values() if abs(v['dist_gdagger']) < 0.20)
n_inv_total = len(inversion_results)

print(f"\n  Propensity-matched low-accel uniformity: {'YES' if low_accel_uniform else 'NO'}")
print(f"  UMa-free matched low-accel uniformity:   {'YES' if uma_free_uniform else 'NO'}")
print(f"  Inversion within 0.20 dex: {n_inv_match}/{n_inv_total}")

if low_accel_uniform and n_inv_match >= 3:
    verdict = "CONFOUND_CONTROLLED_CONFIRMED"
    print(f"\n  >>> VERDICT: Scatter uniformity SURVIVES propensity matching")
    print(f"      Inversion at g† confirmed in matched samples")
elif n_inv_match >= 3:
    verdict = "INVERSION_CONFIRMED_UNIFORMITY_UNCERTAIN"
    print(f"\n  >>> VERDICT: Inversion robust, uniformity uncertain after matching")
else:
    verdict = "INCONCLUSIVE"
    print(f"\n  >>> VERDICT: Inconclusive — too few matched pairs or inversions")


# ================================================================
# SAVE
# ================================================================
results = {
    'test_name': 'propensity_matched_env',
    'n_galaxies': n_gals,
    'n_field': len(field_names),
    'n_dense': len(dense_names),
    'n_uma': len(uma_names),
    'n_group': len(group_names),
    'propensity_matched': {
        'n_pairs': len(pairs_all),
        'scatter': matched_scatter,
    },
    'uma_free_matched': {
        'n_pairs': len(pairs_no_uma),
        'scatter': uma_free_scatter,
    },
    'mahalanobis_matched': {
        'n_pairs': len(pairs_maha),
        'scatter': maha_scatter,
    },
    'inversion_results': inversion_results,
    'overall_verdict': verdict,
}

out_path = os.path.join(RESULTS_DIR, 'summary_propensity_matched_env.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_path}")
print("=" * 72)
print("Done.")
