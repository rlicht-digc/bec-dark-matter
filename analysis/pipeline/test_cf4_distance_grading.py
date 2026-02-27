#!/usr/bin/env python3
"""
CF4 DISTANCE GRADING — Phase 1
================================

Grade Cosmicflows-4 flow-model distances against the 40-galaxy TRGB/Cepheid
subsample with independently measured distances.

For each matched galaxy:
  delta_D_frac = (D_cf4 - D_trgb) / D_trgb

Statistics grouped by distance method (TRGB, Cepheid, NED-upgraded Hubble flow).
Flags outliers with |delta_D_frac| > 0.2 and multi-solution CF4 entries.
One-sample t-test for systematic bias.
Grade: RMS < 0.1 → "reliable", < 0.2 → "marginal", else "unreliable".

Inputs:
  - analysis/results/trgb_cepheid_subsample.csv
  - data/cf4/cf4_distance_cache.json

Output:
  - analysis/results/summary_cf4_distance_grading.json
"""

import os
import csv
import json
import numpy as np
from scipy.stats import ttest_1samp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 72)
print("PHASE 1: CF4 DISTANCE GRADING")
print("=" * 72)

# ================================================================
# STEP 1: Load TRGB/Cepheid subsample
# ================================================================
print("\n[1] Loading TRGB/Cepheid subsample...")

subsample_path = os.path.join(RESULTS_DIR, 'trgb_cepheid_subsample.csv')
subsample = []
with open(subsample_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        subsample.append(row)

print(f"  Loaded {len(subsample)} galaxies from TRGB/Cepheid subsample")

# ================================================================
# STEP 2: Load CF4 distance cache
# ================================================================
print("\n[2] Loading CF4 distance cache...")

cf4_path = os.path.join(PROJECT_ROOT, 'data', 'cf4', 'cf4_distance_cache.json')
with open(cf4_path, 'r') as f:
    cf4_cache = json.load(f)

print(f"  Loaded {len(cf4_cache)} galaxies from CF4 cache")

# ================================================================
# STEP 3: Cross-match and compute fractional distance differences
# ================================================================
print("\n[3] Cross-matching and computing distance differences...")

matches = []
unmatched = []

for gal in subsample:
    name = gal['galaxy']
    sparc_D = float(gal['sparc_D_Mpc'])
    ned_D = float(gal['ned_D_Mpc'])
    method = gal['dist_method']
    sparc_fD = int(gal['sparc_fD'])

    # Use NED distance as the "truth" for gold-standard galaxies
    D_truth = ned_D

    if name in cf4_cache and cf4_cache[name].get('status') == 'success':
        cf4 = cf4_cache[name]
        D_cf4 = cf4['D_cf4']
        delta_D_frac = (D_cf4 - D_truth) / D_truth
        n_solutions = cf4.get('n_solutions', 1)
        D_cf4_all = cf4.get('D_cf4_all', [D_cf4])

        # Spread for multi-solution entries
        if n_solutions > 1 and len(D_cf4_all) > 1:
            spread = max(D_cf4_all) - min(D_cf4_all)
            spread_frac = spread / D_truth
        else:
            spread = 0.0
            spread_frac = 0.0

        matches.append({
            'galaxy': name,
            'D_truth': D_truth,
            'D_sparc': sparc_D,
            'D_cf4': D_cf4,
            'delta_D_frac': delta_D_frac,
            'method': method,
            'sparc_fD': sparc_fD,
            'n_solutions': n_solutions,
            'D_cf4_all': D_cf4_all,
            'spread_Mpc': spread,
            'spread_frac': spread_frac,
            'is_outlier': bool(abs(delta_D_frac) > 0.2),
        })
    else:
        unmatched.append(name)

print(f"  Matched: {len(matches)}, Unmatched: {len(unmatched)}")
if unmatched:
    print(f"  Unmatched galaxies: {', '.join(unmatched)}")

# ================================================================
# STEP 4: Group by distance method
# ================================================================
print("\n[4] Grouping by distance method...")

methods = {}
for m in matches:
    method = m['method']
    if method not in methods:
        methods[method] = []
    methods[method].append(m)

# Also group by original SPARC fD code
fd_groups = {}
for m in matches:
    fd = m['sparc_fD']
    label = {1: 'Hubble_flow', 2: 'TRGB', 3: 'Cepheid', 4: 'UMa_cluster', 5: 'SNe'}.get(fd, f'fD={fd}')
    if label not in fd_groups:
        fd_groups[label] = []
    fd_groups[label].append(m)

# ================================================================
# STEP 5: Compute per-group statistics
# ================================================================
print("\n[5] Computing per-group statistics...")

def group_stats(group_matches, label):
    """Compute statistics for a group of matches."""
    deltas = np.array([m['delta_D_frac'] for m in group_matches])
    n = len(deltas)
    if n == 0:
        return None

    stats = {
        'label': label,
        'N': n,
        'mean': float(np.mean(deltas)),
        'median': float(np.median(deltas)),
        'rms': float(np.sqrt(np.mean(deltas**2))),
        'std': float(np.std(deltas, ddof=1)) if n > 1 else 0.0,
        'min': float(np.min(deltas)),
        'max': float(np.max(deltas)),
        'n_outliers': int(np.sum(np.abs(deltas) > 0.2)),
    }

    # Grade
    rms = stats['rms']
    if rms < 0.1:
        stats['grade'] = 'reliable'
    elif rms < 0.2:
        stats['grade'] = 'marginal'
    else:
        stats['grade'] = 'unreliable'

    # One-sample t-test for systematic bias
    if n >= 3:
        t_stat, p_value = ttest_1samp(deltas, 0.0)
        stats['ttest_t'] = float(t_stat)
        stats['ttest_p'] = float(p_value)
        stats['systematic_bias'] = bool(p_value < 0.05)
    else:
        stats['ttest_t'] = None
        stats['ttest_p'] = None
        stats['systematic_bias'] = None

    return stats


print(f"\n  {'Method':20s} {'N':>4s} {'Mean':>8s} {'Median':>8s} {'RMS':>8s} {'Std':>8s} {'Outliers':>8s} {'Grade':>12s}")
print(f"  {'-'*76}")

method_stats = {}
for method, group in sorted(methods.items()):
    s = group_stats(group, method)
    if s:
        method_stats[method] = s
        sig = '*' if s.get('systematic_bias') else ''
        print(f"  {method:20s} {s['N']:4d} {s['mean']:+8.3f} {s['median']:+8.3f} "
              f"{s['rms']:8.3f} {s['std']:8.3f} {s['n_outliers']:8d} {s['grade']:>12s} {sig}")

print(f"\n  By SPARC fD code:")
print(f"  {'fD Group':20s} {'N':>4s} {'Mean':>8s} {'Median':>8s} {'RMS':>8s} {'Std':>8s} {'Outliers':>8s} {'Grade':>12s}")
print(f"  {'-'*76}")

fd_stats = {}
for label, group in sorted(fd_groups.items()):
    s = group_stats(group, label)
    if s:
        fd_stats[label] = s
        sig = '*' if s.get('systematic_bias') else ''
        print(f"  {label:20s} {s['N']:4d} {s['mean']:+8.3f} {s['median']:+8.3f} "
              f"{s['rms']:8.3f} {s['std']:8.3f} {s['n_outliers']:8d} {s['grade']:>12s} {sig}")

# Overall
all_stats = group_stats(matches, 'ALL')
print(f"\n  {'ALL':20s} {all_stats['N']:4d} {all_stats['mean']:+8.3f} {all_stats['median']:+8.3f} "
      f"{all_stats['rms']:8.3f} {all_stats['std']:8.3f} {all_stats['n_outliers']:8d} {all_stats['grade']:>12s}")

# ================================================================
# STEP 6: Flag outliers and multi-solution entries
# ================================================================
print(f"\n{'='*72}")
print("OUTLIERS (|delta_D_frac| > 0.2)")
print(f"{'='*72}")

outliers = [m for m in matches if m['is_outlier']]
outliers.sort(key=lambda m: -abs(m['delta_D_frac']))

print(f"\n  {'Galaxy':15s} {'D_truth':>8s} {'D_CF4':>8s} {'delta':>8s} {'Method':>10s} {'fD':>4s}")
print(f"  {'-'*60}")
for m in outliers:
    print(f"  {m['galaxy']:15s} {m['D_truth']:8.2f} {m['D_cf4']:8.2f} "
          f"{m['delta_D_frac']:+8.3f} {m['method']:>10s} {m['sparc_fD']:4d}")

# Special attention: D564-8
d564 = [m for m in matches if m['galaxy'] == 'D564-8']
if d564:
    m = d564[0]
    print(f"\n  SPECIAL CASE — D564-8:")
    print(f"    SPARC distance:  {m['D_sparc']:.2f} Mpc (fD=2, TRGB)")
    print(f"    NED distance:    {m['D_truth']:.2f} Mpc")
    print(f"    CF4 distance:    {m['D_cf4']:.2f} Mpc")
    print(f"    SPARC-NED gap:   {(m['D_sparc']-m['D_truth'])/m['D_truth']:+.1%}")
    print(f"    CF4-NED gap:     {m['delta_D_frac']:+.1%}")

# Multi-solution entries
print(f"\n{'='*72}")
print("MULTI-SOLUTION CF4 ENTRIES")
print(f"{'='*72}")

multi = [m for m in matches if m['n_solutions'] > 1]
if multi:
    print(f"\n  {'Galaxy':15s} {'D_truth':>8s} {'D_CF4':>8s} {'N_sol':>6s} {'All solutions':>30s} {'Spread':>8s}")
    print(f"  {'-'*80}")
    for m in multi:
        sols_str = ', '.join(f'{d:.2f}' for d in m['D_cf4_all'])
        print(f"  {m['galaxy']:15s} {m['D_truth']:8.2f} {m['D_cf4']:8.2f} "
              f"{m['n_solutions']:6d} {sols_str:>30s} {m['spread_Mpc']:8.2f}")
else:
    print("\n  No multi-solution entries found in the subsample.")

# ================================================================
# STEP 7: One-sample t-test on full sample
# ================================================================
print(f"\n{'='*72}")
print("SYSTEMATIC BIAS TEST (one-sample t-test, H0: mean delta = 0)")
print(f"{'='*72}")

all_deltas = np.array([m['delta_D_frac'] for m in matches])
t_stat, p_val = ttest_1samp(all_deltas, 0.0)
print(f"\n  N = {len(all_deltas)}")
print(f"  Mean delta_D_frac: {np.mean(all_deltas):+.4f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_val:.6f}")
if p_val < 0.05:
    direction = "overestimates" if np.mean(all_deltas) > 0 else "underestimates"
    print(f"  -> SIGNIFICANT: CF4 systematically {direction} distances (p={p_val:.4f})")
else:
    print(f"  -> No significant systematic bias detected (p={p_val:.4f})")

# Without D564-8 (known huge outlier)
deltas_no_d564 = np.array([m['delta_D_frac'] for m in matches if m['galaxy'] != 'D564-8'])
if len(deltas_no_d564) >= 3:
    t2, p2 = ttest_1samp(deltas_no_d564, 0.0)
    print(f"\n  Excluding D564-8 (known outlier):")
    print(f"  N = {len(deltas_no_d564)}, Mean = {np.mean(deltas_no_d564):+.4f}")
    print(f"  t = {t2:.3f}, p = {p2:.6f}")

# ================================================================
# STEP 8: Summary and save
# ================================================================
print(f"\n{'='*72}")
print("SUMMARY")
print(f"{'='*72}")

print(f"\n  Overall grade: {all_stats['grade'].upper()}")
print(f"  RMS fractional error: {all_stats['rms']:.3f}")
print(f"  Outliers (>20%): {all_stats['n_outliers']} of {all_stats['N']}")
print(f"  Systematic bias: {'YES' if all_stats.get('systematic_bias') else 'NO'}")

summary = {
    'test_name': 'cf4_distance_grading',
    'description': 'Grade CF4 flow-model distances against TRGB/Cepheid gold-standard subsample',
    'n_subsample': len(subsample),
    'n_cf4_matched': len(matches),
    'n_unmatched': len(unmatched),
    'unmatched_galaxies': unmatched,
    'overall': {
        'N': all_stats['N'],
        'mean_delta': round(all_stats['mean'], 4),
        'median_delta': round(all_stats['median'], 4),
        'rms': round(all_stats['rms'], 4),
        'std': round(all_stats['std'], 4),
        'grade': all_stats['grade'],
        'n_outliers': all_stats['n_outliers'],
        'ttest_t': round(t_stat, 4),
        'ttest_p': round(p_val, 6),
        'systematic_bias': bool(p_val < 0.05),
    },
    'by_distance_method': {
        k: {key: (round(v, 4) if isinstance(v, float) else v)
            for key, v in s.items()}
        for k, s in method_stats.items()
    },
    'by_sparc_fD': {
        k: {key: (round(v, 4) if isinstance(v, float) else v)
            for key, v in s.items()}
        for k, s in fd_stats.items()
    },
    'outliers': [
        {
            'galaxy': m['galaxy'],
            'D_truth_Mpc': round(m['D_truth'], 2),
            'D_cf4_Mpc': round(m['D_cf4'], 2),
            'delta_D_frac': round(m['delta_D_frac'], 4),
            'method': m['method'],
            'sparc_fD': m['sparc_fD'],
        }
        for m in outliers
    ],
    'multi_solution': [
        {
            'galaxy': m['galaxy'],
            'D_truth_Mpc': round(m['D_truth'], 2),
            'D_cf4_Mpc': round(m['D_cf4'], 2),
            'n_solutions': m['n_solutions'],
            'D_cf4_all': [round(d, 2) for d in m['D_cf4_all']],
            'spread_Mpc': round(m['spread_Mpc'], 2),
            'spread_frac': round(m['spread_frac'], 4),
        }
        for m in multi
    ],
    'per_galaxy': [
        {
            'galaxy': m['galaxy'],
            'D_truth_Mpc': round(m['D_truth'], 2),
            'D_sparc_Mpc': round(m['D_sparc'], 2),
            'D_cf4_Mpc': round(m['D_cf4'], 2),
            'delta_D_frac': round(m['delta_D_frac'], 4),
            'method': m['method'],
            'sparc_fD': m['sparc_fD'],
            'n_solutions': m['n_solutions'],
            'is_outlier': m['is_outlier'],
        }
        for m in sorted(matches, key=lambda m: m['delta_D_frac'])
    ],
}

outpath = os.path.join(RESULTS_DIR, 'summary_cf4_distance_grading.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("Done.")
