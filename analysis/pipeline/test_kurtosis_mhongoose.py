#!/usr/bin/env python3
"""
MHONGOOSE Kurtosis Augmentation Test
=====================================

Adds 82 MHONGOOSE RAR points (4 galaxies from Nkomo+2025 & Sorgho+2019)
to the existing Tier K extended RAR dataset, recomputes the kurtosis
profile κ₄(g_bar), and checks whether:

  1. The kurtosis spike at g† persists or strengthens
  2. The derivative peak location shifts
  3. The MHONGOOSE points contribute to the spike or dilute it

Baseline result: κ₄ = 20.71 at the g†-containing bin,
                 derivative peak at log g = -9.939.

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import numpy as np
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from load_extended_rar import load_all, build_rar
from match_bouquin_photometry import compute_baryonic_bouquin
from match_korsaga_massmodels import add_korsaga_to_galaxies
from analysis_tools import (
    g_dagger, LOG_G_DAGGER,
    rar_function, rar_residuals,
    binned_stats, bootstrap_kurtosis as _bootstrap_kurtosis,
)

np.random.seed(42)

N_BOOTSTRAP = 10000
N_FINE = 15
GBAR_RANGE = (-12.5, -8.5)
MHONGOOSE_TSV = os.path.join(PROJECT_ROOT, 'data', 'mhongoose', 'mhongoose_rar_all.tsv')

# Baseline values from test_kurtosis_phase_transition.py
BASELINE_KURTOSIS_AT_GDAGGER = 20.71
BASELINE_DERIV_PEAK = -9.939


def load_mhongoose(filepath):
    """Load MHONGOOSE RAR data from TSV file."""
    names, log_gbar, log_gobs, sources = [], [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            try:
                float(parts[1])
            except ValueError:
                continue  # skip header
            names.append(parts[0].strip())
            log_gbar.append(float(parts[1]))
            log_gobs.append(float(parts[2]))
            sources.append(parts[3].strip())
    return (np.array(names), np.array(log_gbar),
            np.array(log_gobs), np.array(sources))


def compute_kurtosis_profile(log_gbar, log_gobs, n_bins=N_FINE,
                              gbar_range=GBAR_RANGE):
    """Compute binned kurtosis profile and derivative."""
    stats = binned_stats(log_gbar, log_gobs, n_bins=n_bins, gbar_range=gbar_range)
    edges = np.linspace(gbar_range[0], gbar_range[1], n_bins + 1)

    # Extract valid bins for derivative computation
    valid = [(b['center'], b['kurtosis']) for b in stats
             if not np.isnan(b['kurtosis'])]

    deriv_peaks = []
    centers_arr = kurt_arr = dkurt = None
    if len(valid) >= 4:
        centers_arr = np.array([v[0] for v in valid])
        kurt_arr = np.array([v[1] for v in valid])

        # Central differences
        dkurt = np.zeros_like(kurt_arr)
        dkurt[0] = (kurt_arr[1] - kurt_arr[0]) / (centers_arr[1] - centers_arr[0])
        dkurt[-1] = (kurt_arr[-1] - kurt_arr[-2]) / (centers_arr[-1] - centers_arr[-2])
        for i in range(1, len(kurt_arr) - 1):
            dkurt[i] = ((kurt_arr[i+1] - kurt_arr[i-1]) /
                         (centers_arr[i+1] - centers_arr[i-1]))

        # Zero crossings (positive → negative = peak)
        for i in range(len(dkurt) - 1):
            if dkurt[i] > 0 and dkurt[i+1] < 0:
                x_cross = (centers_arr[i] - dkurt[i] *
                           (centers_arr[i+1] - centers_arr[i]) /
                           (dkurt[i+1] - dkurt[i]))
                deriv_peaks.append(float(x_cross))

    return stats, edges, centers_arr, kurt_arr, dkurt, deriv_peaks


def find_at_gdagger(bins_list):
    """Find the bin containing g†."""
    for b in bins_list:
        if abs(b['center'] - LOG_G_DAGGER) < 0.15 and not np.isnan(b['kurtosis']):
            return b
    return None


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("MHONGOOSE KURTOSIS AUGMENTATION TEST")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")
print(f"  Baseline: κ₄ = {BASELINE_KURTOSIS_AT_GDAGGER} at g†, "
      f"deriv peak at log g = {BASELINE_DERIV_PEAK}")

# ================================================================
# 1. LOAD TIER K DATA (baseline)
# ================================================================
print("\n" + "=" * 72)
print("[1] Loading Tier K extended RAR dataset (baseline)...")
print("=" * 72)

galaxies = load_all(include_vizier=True, match_photometry=True)
compute_baryonic_bouquin(galaxies)
add_korsaga_to_galaxies(galaxies, use_fixed_ml=True, verbose=False)

log_gobs_all, log_gbar_all, names_all, sources_all = build_rar(galaxies)

# Build Tier K mask
sparc_mask = sources_all == 'SPARC'
things_mask = sources_all == 'THINGS'
deblok_mask = sources_all == 'deBlok2002'

good_bouquin = set()
for name, g in galaxies.items():
    mm = g.get('mass_model_source', '')
    if 'Bouquin' in mm and g['source'] in ('Verheijen2001', 'PHANGS_Lang2020'):
        good_bouquin.add(name)

bouquin_mask = np.array([n in good_bouquin for n in names_all])
korsaga_mask = sources_all == 'Korsaga2019'

tier_k_mask = sparc_mask | things_mask | deblok_mask | bouquin_mask | korsaga_mask

log_gobs_k = log_gobs_all[tier_k_mask]
log_gbar_k = log_gbar_all[tier_k_mask]
names_k = names_all[tier_k_mask]
sources_k = sources_all[tier_k_mask]

n_gal_k = len(set(names_k))
print(f"  Tier K: {n_gal_k} galaxies, {len(log_gobs_k)} RAR points")


# ================================================================
# 2. LOAD MHONGOOSE DATA
# ================================================================
print("\n" + "=" * 72)
print("[2] Loading MHONGOOSE RAR data...")
print("=" * 72)

mh_names, mh_gbar, mh_gobs, mh_sources = load_mhongoose(MHONGOOSE_TSV)
mh_galaxies = sorted(set(mh_names))
print(f"  MHONGOOSE: {len(mh_galaxies)} galaxies, {len(mh_names)} RAR points")
print(f"  Galaxies: {', '.join(mh_galaxies)}")
print(f"  Sources: {', '.join(sorted(set(mh_sources)))}")
print(f"  log g_bar range: [{mh_gbar.min():.2f}, {mh_gbar.max():.2f}]")

# Check for overlap with existing Tier K galaxies
existing_names = set(names_k)
overlap = set(mh_galaxies) & existing_names
print(f"\n  Overlap with Tier K: {overlap if overlap else 'none'}")

# Count existing points for overlapping galaxies
if overlap:
    for gal in overlap:
        n_existing = np.sum(names_k == gal)
        n_mh = np.sum(mh_names == gal)
        print(f"    {gal}: {n_existing} existing + {n_mh} MHONGOOSE points")

# Where do MHONGOOSE points fall in the binning?
edges_check = np.linspace(GBAR_RANGE[0], GBAR_RANGE[1], N_FINE + 1)
print(f"\n  MHONGOOSE points per bin:")
for j in range(N_FINE):
    lo, hi = edges_check[j], edges_check[j+1]
    center = (lo + hi) / 2
    n_in_bin = np.sum((mh_gbar >= lo) & (mh_gbar < hi))
    if n_in_bin > 0:
        marker = " ← g†" if abs(center - LOG_G_DAGGER) < 0.15 else ""
        print(f"    bin {center:+.2f}: {n_in_bin} points{marker}")

# Count how many fall outside binning range
n_below = np.sum(mh_gbar < GBAR_RANGE[0])
n_above = np.sum(mh_gbar > GBAR_RANGE[1])
n_in_range = len(mh_gbar) - n_below - n_above
print(f"\n  In range [{GBAR_RANGE[0]}, {GBAR_RANGE[1]}]: {n_in_range}")
print(f"  Below range: {n_below}")
print(f"  Above range: {n_above}")


# ================================================================
# 3. BASELINE KURTOSIS PROFILE (Tier K only)
# ================================================================
print("\n" + "=" * 72)
print("[3] Baseline kurtosis profile (Tier K only)")
print("=" * 72)

(bins_base, edges_base, centers_base, kurt_base,
 dkurt_base, peaks_base) = compute_kurtosis_profile(log_gbar_k, log_gobs_k)

base_at_gd = find_at_gdagger(bins_base)
print(f"\n  {'center':>8s} {'N':>6s} {'κ₄':>9s} {'dκ₄/dx':>10s}")
print(f"  {'-'*36}")
if centers_base is not None:
    for i, b in enumerate(bins_base):
        if np.isnan(b['kurtosis']):
            continue
        # Find matching derivative
        idx = np.argmin(np.abs(centers_base - b['center']))
        dk = dkurt_base[idx] if dkurt_base is not None else np.nan
        marker = " ← g†" if abs(b['center'] - LOG_G_DAGGER) < 0.15 else ""
        print(f"  {b['center']:8.2f} {b['n']:6d} {b['kurtosis']:+9.2f} "
              f"{dk:+10.2f}{marker}")

if base_at_gd:
    print(f"\n  κ₄ at g†: {base_at_gd['kurtosis']:.2f} "
          f"(N = {base_at_gd['n']})")
if peaks_base:
    nearest_base = min(peaks_base, key=lambda x: abs(x - LOG_G_DAGGER))
    print(f"  Derivative peak nearest g†: log g = {nearest_base:.3f} "
          f"(Δ = {nearest_base - LOG_G_DAGGER:+.3f} dex)")


# ================================================================
# 4. COMBINED KURTOSIS PROFILE (Tier K + MHONGOOSE)
# ================================================================
print("\n" + "=" * 72)
print("[4] Combined kurtosis profile (Tier K + MHONGOOSE)")
print("=" * 72)

# Merge datasets — include all MHONGOOSE points (even for overlapping galaxies,
# MHONGOOSE provides independent HI rotation curve data at different radii)
log_gbar_combined = np.concatenate([log_gbar_k, mh_gbar])
log_gobs_combined = np.concatenate([log_gobs_k, mh_gobs])
names_combined = np.concatenate([names_k, mh_names])
sources_combined = np.concatenate([sources_k,
                                   np.array(['MHONGOOSE'] * len(mh_names))])

n_gal_combined = len(set(names_combined))
print(f"  Combined: {n_gal_combined} galaxies, {len(log_gbar_combined)} RAR points")

(bins_comb, edges_comb, centers_comb, kurt_comb,
 dkurt_comb, peaks_comb) = compute_kurtosis_profile(log_gbar_combined,
                                                      log_gobs_combined)

comb_at_gd = find_at_gdagger(bins_comb)
print(f"\n  {'center':>8s} {'N':>6s} {'N_mh':>6s} {'κ₄':>9s} {'dκ₄/dx':>10s} {'Δκ₄':>8s}")
print(f"  {'-'*54}")
if centers_comb is not None:
    for b in bins_comb:
        if np.isnan(b['kurtosis']):
            continue
        # Find matching baseline bin
        base_match = next((bb for bb in bins_base
                           if abs(bb['center'] - b['center']) < 0.01
                           and not np.isnan(bb['kurtosis'])), None)
        delta_k = (b['kurtosis'] - base_match['kurtosis']
                   if base_match else np.nan)

        # Count MHONGOOSE points in this bin
        lo = b['center'] - (GBAR_RANGE[1] - GBAR_RANGE[0]) / (2 * N_FINE)
        hi = b['center'] + (GBAR_RANGE[1] - GBAR_RANGE[0]) / (2 * N_FINE)
        n_mh_bin = int(np.sum((mh_gbar >= lo) & (mh_gbar < hi)))

        idx = np.argmin(np.abs(centers_comb - b['center']))
        dk = dkurt_comb[idx] if dkurt_comb is not None else np.nan
        marker = " ← g†" if abs(b['center'] - LOG_G_DAGGER) < 0.15 else ""
        delta_str = f"{delta_k:+8.2f}" if not np.isnan(delta_k) else "     N/A"
        print(f"  {b['center']:8.2f} {b['n']:6d} {n_mh_bin:6d} "
              f"{b['kurtosis']:+9.2f} {dk:+10.2f} {delta_str}{marker}")

if comb_at_gd:
    print(f"\n  κ₄ at g†: {comb_at_gd['kurtosis']:.2f} "
          f"(N = {comb_at_gd['n']})")
if peaks_comb:
    nearest_comb = min(peaks_comb, key=lambda x: abs(x - LOG_G_DAGGER))
    print(f"  Derivative peak nearest g†: log g = {nearest_comb:.3f} "
          f"(Δ = {nearest_comb - LOG_G_DAGGER:+.3f} dex)")


# ================================================================
# 5. BOOTSTRAP COMPARISON AT g†
# ================================================================
print("\n" + "=" * 72)
print("[5] Bootstrap comparison at g†")
print("=" * 72)

print("  Bootstrapping baseline (Tier K)...")
boot_base = _bootstrap_kurtosis(log_gbar_k, log_gobs_k, n_bins=N_FINE,
                                 gbar_range=GBAR_RANGE, n_boot=N_BOOTSTRAP)
print("  Bootstrapping combined (Tier K + MHONGOOSE)...")
boot_comb = _bootstrap_kurtosis(log_gbar_combined, log_gobs_combined,
                                 n_bins=N_FINE, gbar_range=GBAR_RANGE,
                                 n_boot=N_BOOTSTRAP)

boot_base_gd = next((b for b in boot_base
                      if abs(b['center'] - LOG_G_DAGGER) < 0.15
                      and not np.isnan(b.get('kurtosis_mean', np.nan))), None)
boot_comb_gd = next((b for b in boot_comb
                      if abs(b['center'] - LOG_G_DAGGER) < 0.15
                      and not np.isnan(b.get('kurtosis_mean', np.nan))), None)

if boot_base_gd and boot_comb_gd:
    print(f"\n  At g† (log g_bar ≈ {LOG_G_DAGGER:.2f}):")
    print(f"    Baseline:  κ₄ = {boot_base_gd['kurtosis_mean']:.2f} "
          f"(95% CI: [{boot_base_gd['kurtosis_ci_lo']:.1f}, "
          f"{boot_base_gd['kurtosis_ci_hi']:.1f}], σ = {boot_base_gd['kurtosis_std']:.2f})")
    print(f"    Combined:  κ₄ = {boot_comb_gd['kurtosis_mean']:.2f} "
          f"(95% CI: [{boot_comb_gd['kurtosis_ci_lo']:.1f}, "
          f"{boot_comb_gd['kurtosis_ci_hi']:.1f}], σ = {boot_comb_gd['kurtosis_std']:.2f})")

    delta = boot_comb_gd['kurtosis_mean'] - boot_base_gd['kurtosis_mean']
    pct = 100 * delta / boot_base_gd['kurtosis_mean'] if boot_base_gd['kurtosis_mean'] != 0 else 0
    print(f"    Δκ₄ = {delta:+.2f} ({pct:+.1f}%)")


# ================================================================
# 6. MHONGOOSE-ONLY KURTOSIS PROFILE
# ================================================================
print("\n" + "=" * 72)
print("[6] MHONGOOSE-only residual statistics")
print("=" * 72)

# Compute RAR residuals for MHONGOOSE points
mh_residuals = rar_residuals(mh_gbar, mh_gobs)
print(f"  MHONGOOSE residuals: mean = {np.mean(mh_residuals):.4f}, "
      f"std = {np.std(mh_residuals):.4f}")
print(f"  Overall kurtosis: {kurtosis(mh_residuals, fisher=True):.2f}")
print(f"  Overall skewness: {float(np.mean(((mh_residuals - np.mean(mh_residuals)) / np.std(mh_residuals))**3)):.2f}")

# Per-galaxy residual stats
print(f"\n  Per-galaxy MHONGOOSE residual statistics:")
print(f"  {'Galaxy':>15s} {'N':>4s} {'mean':>8s} {'std':>8s} {'κ₄':>8s}")
print(f"  {'-'*48}")
for gal in sorted(set(mh_names)):
    mask_g = mh_names == gal
    res_g = mh_residuals[mask_g]
    k4 = kurtosis(res_g, fisher=True) if len(res_g) >= 10 else np.nan
    print(f"  {gal:>15s} {len(res_g):4d} {np.mean(res_g):+8.4f} "
          f"{np.std(res_g):8.4f} {k4:+8.2f}" if not np.isnan(k4)
          else f"  {gal:>15s} {len(res_g):4d} {np.mean(res_g):+8.4f} "
               f"{np.std(res_g):8.4f}      N/A")

# Which bins do MHONGOOSE residuals affect most?
print(f"\n  MHONGOOSE impact by bin (combined vs baseline residual kurtosis):")
print(f"  {'center':>8s} {'N_base':>7s} {'N_comb':>7s} {'N_mh':>5s} "
      f"{'κ₄_base':>9s} {'κ₄_comb':>9s} {'Δκ₄':>8s}")
print(f"  {'-'*58}")
for bb, bc in zip(bins_base, bins_comb):
    if np.isnan(bb['kurtosis']) and np.isnan(bc['kurtosis']):
        continue
    lo = bb['center'] - (GBAR_RANGE[1] - GBAR_RANGE[0]) / (2 * N_FINE)
    hi = bb['center'] + (GBAR_RANGE[1] - GBAR_RANGE[0]) / (2 * N_FINE)
    n_mh_bin = int(np.sum((mh_gbar >= lo) & (mh_gbar < hi)))
    kb = bb['kurtosis'] if not np.isnan(bb['kurtosis']) else 0
    kc = bc['kurtosis'] if not np.isnan(bc['kurtosis']) else 0
    delta = kc - kb
    marker = " ← g†" if abs(bb['center'] - LOG_G_DAGGER) < 0.15 else ""
    kb_str = f"{kb:+9.2f}" if not np.isnan(bb['kurtosis']) else "      N/A"
    kc_str = f"{kc:+9.2f}" if not np.isnan(bc['kurtosis']) else "      N/A"
    d_str = f"{delta:+8.2f}" if not (np.isnan(bb['kurtosis']) or np.isnan(bc['kurtosis'])) else "     N/A"
    print(f"  {bb['center']:8.2f} {bb['n']:7d} {bc['n']:7d} {n_mh_bin:5d} "
          f"{kb_str} {kc_str} {d_str}{marker}")


# ================================================================
# 7. DERIVATIVE PEAK COMPARISON
# ================================================================
print("\n" + "=" * 72)
print("[7] Derivative peak comparison")
print("=" * 72)

if peaks_base and peaks_comb:
    nearest_base_peak = min(peaks_base, key=lambda x: abs(x - LOG_G_DAGGER))
    nearest_comb_peak = min(peaks_comb, key=lambda x: abs(x - LOG_G_DAGGER))
    shift = nearest_comb_peak - nearest_base_peak

    print(f"  Baseline derivative peak: log g = {nearest_base_peak:.3f} "
          f"(Δ from g† = {nearest_base_peak - LOG_G_DAGGER:+.3f} dex)")
    print(f"  Combined derivative peak: log g = {nearest_comb_peak:.3f} "
          f"(Δ from g† = {nearest_comb_peak - LOG_G_DAGGER:+.3f} dex)")
    print(f"  Shift: {shift:+.4f} dex")
    print(f"  Shift significant (> 0.05 dex): {'YES' if abs(shift) > 0.05 else 'NO'}")
elif peaks_base:
    print(f"  Baseline derivative peak: found")
    print(f"  Combined: no derivative peak detected — profile may have changed shape")
else:
    print(f"  No derivative peaks found in either profile")


# ================================================================
# 8. SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

base_kurt_gd = base_at_gd['kurtosis'] if base_at_gd else np.nan
comb_kurt_gd = comb_at_gd['kurtosis'] if comb_at_gd else np.nan

print(f"\n  Dataset comparison:")
print(f"    Tier K baseline:      {n_gal_k} galaxies, {len(log_gbar_k)} points")
print(f"    MHONGOOSE addition:   {len(mh_galaxies)} galaxies, {len(mh_names)} points")
print(f"    Combined:             {n_gal_combined} galaxies, {len(log_gbar_combined)} points")

print(f"\n  (1) Kurtosis spike at g† (log g_bar ≈ {LOG_G_DAGGER:.2f}):")
print(f"    Baseline κ₄:  {base_kurt_gd:.2f}")
print(f"    Combined κ₄:  {comb_kurt_gd:.2f}")
delta_gd = comb_kurt_gd - base_kurt_gd
pct_gd = 100 * delta_gd / base_kurt_gd if base_kurt_gd != 0 else 0
print(f"    Change:        {delta_gd:+.2f} ({pct_gd:+.1f}%)")

if abs(delta_gd) < 0.5:
    spike_verdict = "UNCHANGED — spike persists at same strength"
elif delta_gd > 0:
    spike_verdict = "STRENGTHENED — MHONGOOSE data reinforces the spike"
else:
    spike_verdict = "WEAKENED — MHONGOOSE data dilutes the spike"

if not np.isnan(comb_kurt_gd) and comb_kurt_gd > 5.0:
    spike_verdict += f" (κ₄ = {comb_kurt_gd:.1f} still strongly leptokurtic)"

print(f"    Verdict:       {spike_verdict}")

print(f"\n  (2) Derivative peak location:")
if peaks_base and peaks_comb:
    print(f"    Baseline: log g = {nearest_base_peak:.3f}")
    print(f"    Combined: log g = {nearest_comb_peak:.3f}")
    print(f"    Shift:    {shift:+.4f} dex "
          f"({'negligible' if abs(shift) < 0.05 else 'significant'})")
else:
    print(f"    Could not compare (missing peaks)")

print(f"\n  (3) MHONGOOSE contribution:")
# Count how many MHONGOOSE points fall in the g† bin
edges_final = np.linspace(GBAR_RANGE[0], GBAR_RANGE[1], N_FINE + 1)
gd_bin_idx = None
for j in range(N_FINE):
    center = (edges_final[j] + edges_final[j+1]) / 2
    if abs(center - LOG_G_DAGGER) < 0.15:
        gd_bin_idx = j
        break

if gd_bin_idx is not None:
    lo_gd = edges_final[gd_bin_idx]
    hi_gd = edges_final[gd_bin_idx + 1]
    n_mh_at_gd = int(np.sum((mh_gbar >= lo_gd) & (mh_gbar < hi_gd)))
    print(f"    MHONGOOSE points in g† bin: {n_mh_at_gd}")
    if n_mh_at_gd == 0:
        print(f"    MHONGOOSE data is entirely in the low-acceleration regime")
        print(f"    (log g_bar < {lo_gd:.1f}), below the g† bin")
        print(f"    → MHONGOOSE does NOT directly affect the spike at g†")
        print(f"    → The spike is robust to addition of independent low-g data")
    else:
        print(f"    MHONGOOSE directly contributes {n_mh_at_gd} points to the g† bin")

# Which bins were most affected?
max_delta = -np.inf
max_delta_center = np.nan
for bb, bc in zip(bins_base, bins_comb):
    if np.isnan(bb['kurtosis']) or np.isnan(bc['kurtosis']):
        continue
    d = abs(bc['kurtosis'] - bb['kurtosis'])
    if d > max_delta:
        max_delta = d
        max_delta_center = bb['center']

if not np.isnan(max_delta_center):
    print(f"\n    Largest kurtosis change: |Δκ₄| = {max_delta:.2f} "
          f"at log g = {max_delta_center:.2f}")


# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    'test_name': 'kurtosis_mhongoose_augmentation',
    'description': (
        'Tests whether adding 82 MHONGOOSE RAR points to Tier K data '
        'affects the kurtosis spike at g†. MHONGOOSE provides deep HI '
        'rotation curves from Nkomo+2025 and Sorgho+2019.'
    ),
    'g_dagger_log': LOG_G_DAGGER,
    'baseline': {
        'n_galaxies': n_gal_k,
        'n_points': int(len(log_gbar_k)),
        'kurtosis_at_gdagger': round(float(base_kurt_gd), 3) if not np.isnan(base_kurt_gd) else None,
        'derivative_peak': round(float(nearest_base_peak), 3) if peaks_base else None,
    },
    'mhongoose': {
        'n_galaxies': len(mh_galaxies),
        'n_points': len(mh_names),
        'galaxies': mh_galaxies,
        'log_gbar_range': [round(float(mh_gbar.min()), 3),
                           round(float(mh_gbar.max()), 3)],
        'overlap_with_tier_k': list(overlap),
    },
    'combined': {
        'n_galaxies': n_gal_combined,
        'n_points': int(len(log_gbar_combined)),
        'kurtosis_at_gdagger': round(float(comb_kurt_gd), 3) if not np.isnan(comb_kurt_gd) else None,
        'derivative_peak': round(float(nearest_comb_peak), 3) if peaks_comb else None,
    },
    'comparison': {
        'delta_kurtosis_at_gdagger': round(float(delta_gd), 3) if not np.isnan(delta_gd) else None,
        'pct_change': round(float(pct_gd), 2) if not np.isnan(pct_gd) else None,
        'derivative_peak_shift': round(float(shift), 4) if (peaks_base and peaks_comb) else None,
        'spike_verdict': spike_verdict,
    },
}

if boot_base_gd and boot_comb_gd:
    results['bootstrap'] = {
        'n_iterations': N_BOOTSTRAP,
        'baseline_ci': [round(float(boot_base_gd['kurtosis_ci_lo']), 2),
                        round(float(boot_base_gd['kurtosis_ci_hi']), 2)],
        'combined_ci': [round(float(boot_comb_gd['kurtosis_ci_lo']), 2),
                        round(float(boot_comb_gd['kurtosis_ci_hi']), 2)],
    }

outpath = os.path.join(RESULTS_DIR, 'summary_kurtosis_mhongoose.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved: {outpath}")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
