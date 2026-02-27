#!/usr/bin/env python3
"""
Kurtosis Phase Transition Test (§8.27)
=======================================

Tests whether the excess kurtosis of RAR residuals peaks at g† — the
predicted signature of a continuous phase transition in the BEC framework.

At a phase boundary, intermittent switching between condensed and thermal
states produces heavy-tailed (leptokurtic) residual distributions. Away
from the boundary, residuals are approximately Gaussian.

Three analyses:
  1. Bootstrap significance of the kurtosis spike in Tier K data (250 gal, 9,406 pts)
  2. ΛCDM semi-analytic comparison (500 mock galaxies, same binning)
  3. SPARC-only comparison (baseline)

BEC prediction: κ₄ peaks in the bin containing g† (log g_bar ≈ -9.92)
ΛCDM prediction: κ₄ varies smoothly, no localized spike at g†

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import numpy as np
from scipy.stats import kurtosis, skew
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
    generate_lcdm_mock,
    numerical_derivative, find_zero_crossings, get_at_gdagger,
    classify_env_extended,
    UMA_GALAXIES, GROUP_MEMBERS,
)

np.random.seed(42)

N_BOOTSTRAP = 10000
N_FINE = 15  # Number of bins (matches extended RAR script)


def compute_binned_kurtosis(log_gbar, log_gobs, n_bins=N_FINE,
                            gbar_range=(-12.5, -8.5)):
    """Backward-compatible wrapper. Returns (stats, residuals, edges)."""
    stats = binned_stats(log_gbar, log_gobs, n_bins=n_bins, gbar_range=gbar_range)
    residuals = rar_residuals(log_gbar, log_gobs)
    edges = np.linspace(gbar_range[0], gbar_range[1], n_bins + 1)
    return stats, residuals, edges


def bootstrap_kurtosis_local(log_gbar, log_gobs, n_bins=N_FINE,
                             gbar_range=(-12.5, -8.5), n_boot=N_BOOTSTRAP):
    """Backward-compatible wrapper around analysis_tools.bootstrap_kurtosis."""
    return _bootstrap_kurtosis(log_gbar, log_gobs, n_bins=n_bins,
                               gbar_range=gbar_range, n_boot=n_boot)


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("KURTOSIS PHASE TRANSITION TEST (§8.27)")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")
print(f"  Bootstrap iterations: {N_BOOTSTRAP}")

# ================================================================
# 1. LOAD TIER K DATA
# ================================================================
print("\n" + "=" * 72)
print("[1] Loading Tier K extended RAR dataset...")
print("=" * 72)

galaxies = load_all(include_vizier=True, match_photometry=True)
print("  Applying Bouquin+2018 profile matching...")
compute_baryonic_bouquin(galaxies)
print("  Applying Korsaga+2019 mass model reconstruction...")
add_korsaga_to_galaxies(galaxies, use_fixed_ml=True, verbose=False)

log_gobs_all, log_gbar_all, names_all, sources_all = build_rar(galaxies)

# Build Tier K mask (SPARC + THINGS + deBlok2002 + quality Bouquin + Korsaga)
sparc_mask = sources_all == 'SPARC'
things_mask = sources_all == 'THINGS'
deblok_mask = sources_all == 'deBlok2002'

# Quality Bouquin galaxies (Verheijen, PHANGS — not GomezLopez)
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

n_gal_k = len(set(names_k))
print(f"\n  Tier K: {n_gal_k} galaxies, {len(log_gobs_k)} RAR points")

# Also extract SPARC-only
log_gobs_s = log_gobs_all[sparc_mask]
log_gbar_s = log_gbar_all[sparc_mask]
n_gal_s = len(set(names_all[sparc_mask]))
print(f"  SPARC:  {n_gal_s} galaxies, {len(log_gobs_s)} RAR points")


# ================================================================
# 2. KURTOSIS PROFILE — TIER K
# ================================================================
print("\n" + "=" * 72)
print("[2] Kurtosis profile — Tier K (observed data)")
print("=" * 72)

bins_k, _, _ = compute_binned_kurtosis(log_gbar_k, log_gobs_k)
boot_k = bootstrap_kurtosis_local(log_gbar_k, log_gobs_k)

print(f"\n  {'center':>8s} {'N':>6s} {'σ':>8s} {'κ₄':>8s} "
      f"{'95% CI':>18s} {'boot σ':>8s}")
print(f"  {'-'*56}")

peak_kurt = -np.inf
peak_center = np.nan
for b, bk in zip(bins_k, boot_k):
    if np.isnan(b['kurtosis']):
        continue
    ci_str = f"[{bk['kurtosis_ci_lo']:+.1f}, {bk['kurtosis_ci_hi']:+.1f}]"
    marker = ""
    if not np.isnan(b['kurtosis']) and b['kurtosis'] > peak_kurt:
        peak_kurt = b['kurtosis']
        peak_center = b['center']
    if abs(b['center'] - LOG_G_DAGGER) < 0.15:
        marker = " ← g†"
    print(f"  {b['center']:8.2f} {b['n']:6d} {b['sigma']:8.4f} "
          f"{b['kurtosis']:+8.2f} {ci_str:>18s} {bk['kurtosis_std']:8.2f}{marker}")

print(f"\n  Peak kurtosis: κ₄ = {peak_kurt:.2f} at log g_bar = {peak_center:.2f}")
print(f"  Distance from g†: {peak_center - LOG_G_DAGGER:+.3f} dex")

# Is the peak in the g†-containing bin?
gdagger_bin_kurt = np.nan
for b in bins_k:
    if abs(b['center'] - LOG_G_DAGGER) < 0.15:
        gdagger_bin_kurt = b['kurtosis']
        break


# ================================================================
# 3. KURTOSIS PROFILE — SPARC ONLY
# ================================================================
print("\n" + "=" * 72)
print("[3] Kurtosis profile — SPARC only (baseline)")
print("=" * 72)

bins_s, _, _ = compute_binned_kurtosis(log_gbar_s, log_gobs_s)
boot_s = bootstrap_kurtosis_local(log_gbar_s, log_gobs_s)

print(f"\n  {'center':>8s} {'N':>6s} {'σ':>8s} {'κ₄':>8s} "
      f"{'95% CI':>18s}")
print(f"  {'-'*48}")
for b, bk in zip(bins_s, boot_s):
    if np.isnan(b['kurtosis']):
        continue
    ci_str = f"[{bk['kurtosis_ci_lo']:+.1f}, {bk['kurtosis_ci_hi']:+.1f}]"
    marker = " ← g†" if abs(b['center'] - LOG_G_DAGGER) < 0.15 else ""
    print(f"  {b['center']:8.2f} {b['n']:6d} {b['sigma']:8.4f} "
          f"{b['kurtosis']:+8.2f} {ci_str:>18s}{marker}")


# ================================================================
# 4. ΛCDM MOCK — KURTOSIS PROFILE
# ================================================================
print("\n" + "=" * 72)
print("[4] Kurtosis profile — ΛCDM semi-analytic mock (500 galaxies)")
print("=" * 72)

# Generate 5 independent realizations to assess variance
N_LCDM_REALIZATIONS = 5
lcdm_kurtosis_profiles = []

for r in range(N_LCDM_REALIZATIONS):
    lcdm_gbar, lcdm_gobs = generate_lcdm_mock(n_gal=500, seed=42 + r)
    bins_l, _, _ = compute_binned_kurtosis(lcdm_gbar, lcdm_gobs)
    lcdm_kurtosis_profiles.append(bins_l)

    if r == 0:
        print(f"\n  ΛCDM realization 0: {len(lcdm_gbar)} RAR points")
        print(f"\n  {'center':>8s} {'N':>6s} {'σ':>8s} {'κ₄':>8s}")
        print(f"  {'-'*36}")
        for b in bins_l:
            if np.isnan(b['kurtosis']):
                continue
            marker = " ← g†" if abs(b['center'] - LOG_G_DAGGER) < 0.15 else ""
            print(f"  {b['center']:8.2f} {b['n']:6d} {b['sigma']:8.4f} "
                  f"{b['kurtosis']:+8.2f}{marker}")

# Compute mean and std of ΛCDM kurtosis across realizations
print(f"\n  ΛCDM kurtosis across {N_LCDM_REALIZATIONS} realizations:")
print(f"  {'center':>8s} {'mean κ₄':>9s} {'std κ₄':>9s} {'max κ₄':>9s}")
print(f"  {'-'*40}")

lcdm_peak_kurt = -np.inf
lcdm_peak_center = np.nan

for j in range(N_FINE):
    vals = []
    for prof in lcdm_kurtosis_profiles:
        if j < len(prof) and not np.isnan(prof[j]['kurtosis']):
            vals.append(prof[j]['kurtosis'])
    if vals:
        center = lcdm_kurtosis_profiles[0][j]['center']
        mean_k = np.mean(vals)
        std_k = np.std(vals)
        max_k = np.max(vals)
        if mean_k > lcdm_peak_kurt:
            lcdm_peak_kurt = mean_k
            lcdm_peak_center = center
        marker = " ← g†" if abs(center - LOG_G_DAGGER) < 0.15 else ""
        print(f"  {center:8.2f} {mean_k:+9.2f} {std_k:9.2f} {max_k:+9.2f}{marker}")


# ================================================================
# 5. SPIKE SIGNIFICANCE TEST
# ================================================================
print("\n" + "=" * 72)
print("[5] Kurtosis spike significance")
print("=" * 72)

# Find the g†-containing bin in Tier K bootstrap results
gdagger_boot = None
adjacent_boots = []
for bk in boot_k:
    if np.isnan(bk.get('kurtosis_mean', np.nan)):
        continue
    if abs(bk['center'] - LOG_G_DAGGER) < 0.15:
        gdagger_boot = bk
    elif abs(bk['center'] - LOG_G_DAGGER) < 0.50:
        adjacent_boots.append(bk)

if gdagger_boot is not None:
    print(f"\n  g†-containing bin (center = {gdagger_boot['center']:.2f}):")
    print(f"    κ₄ = {gdagger_boot['kurtosis_mean']:.2f} "
          f"(95% CI: [{gdagger_boot['kurtosis_ci_lo']:.1f}, "
          f"{gdagger_boot['kurtosis_ci_hi']:.1f}])")

    # Is the peak significantly above adjacent bins?
    if adjacent_boots:
        adj_kurts = [b['kurtosis_mean'] for b in adjacent_boots
                     if not np.isnan(b['kurtosis_mean'])]
        max_adjacent = max(adj_kurts) if adj_kurts else 0
        print(f"    Maximum adjacent bin κ₄: {max_adjacent:.2f}")
        print(f"    Ratio (peak / max adjacent): "
              f"{gdagger_boot['kurtosis_mean'] / max_adjacent:.1f}×"
              if max_adjacent > 0 else "")

        # Is the g† bin's lower CI above the adjacent upper CI?
        adj_upper_cis = [b['kurtosis_ci_hi'] for b in adjacent_boots
                         if not np.isnan(b['kurtosis_ci_hi'])]
        max_adj_upper = max(adj_upper_cis) if adj_upper_cis else 0
        separated = gdagger_boot['kurtosis_ci_lo'] > max_adj_upper
        print(f"    g† lower 95% CI ({gdagger_boot['kurtosis_ci_lo']:.1f}) > "
              f"max adjacent upper 95% CI ({max_adj_upper:.1f}): {separated}")

    # Compare to ΛCDM at same bin
    lcdm_at_gdagger = []
    for prof in lcdm_kurtosis_profiles:
        for b in prof:
            if abs(b['center'] - LOG_G_DAGGER) < 0.15 and not np.isnan(b['kurtosis']):
                lcdm_at_gdagger.append(b['kurtosis'])
    if lcdm_at_gdagger:
        lcdm_mean = np.mean(lcdm_at_gdagger)
        lcdm_max = np.max(lcdm_at_gdagger)
        print(f"\n  ΛCDM at g†: mean κ₄ = {lcdm_mean:.2f}, max = {lcdm_max:.2f}")
        print(f"  Observed/ΛCDM ratio: {gdagger_boot['kurtosis_mean'] / lcdm_mean:.1f}×"
              if lcdm_mean > 0 else "")
        exceeds = gdagger_boot['kurtosis_ci_lo'] > lcdm_max
        print(f"  Observed lower 95% CI ({gdagger_boot['kurtosis_ci_lo']:.1f}) > "
              f"ΛCDM max ({lcdm_max:.1f}): {exceeds}")


# ================================================================
# 6. KURTOSIS DERIVATIVE — WHERE DOES d(κ₄)/d(log g) PEAK?
# ================================================================
print("\n" + "=" * 72)
print("[6] Kurtosis derivative analysis")
print("=" * 72)

valid_k = [(b['center'], b['kurtosis']) for b in bins_k
           if not np.isnan(b['kurtosis'])]
if len(valid_k) >= 4:
    centers_arr = np.array([v[0] for v in valid_k])
    kurt_arr = np.array([v[1] for v in valid_k])

    # Central differences
    dkurt = np.zeros_like(kurt_arr)
    dkurt[0] = (kurt_arr[1] - kurt_arr[0]) / (centers_arr[1] - centers_arr[0])
    dkurt[-1] = (kurt_arr[-1] - kurt_arr[-2]) / (centers_arr[-1] - centers_arr[-2])
    for i in range(1, len(kurt_arr) - 1):
        dkurt[i] = (kurt_arr[i+1] - kurt_arr[i-1]) / (centers_arr[i+1] - centers_arr[i-1])

    # Zero crossings of d(κ₄)/d(log g) — positive-to-negative = peak location
    kurt_peaks = []
    for i in range(len(dkurt) - 1):
        if dkurt[i] > 0 and dkurt[i+1] < 0:
            # Linear interpolation
            x_cross = (centers_arr[i] - dkurt[i] * (centers_arr[i+1] - centers_arr[i])
                       / (dkurt[i+1] - dkurt[i]))
            kurt_peaks.append(float(x_cross))

    print(f"\n  κ₄ peaks (d(κ₄)/d(log g) = 0, pos→neg) at: {kurt_peaks}")
    if kurt_peaks:
        nearest = min(kurt_peaks, key=lambda x: abs(x - LOG_G_DAGGER))
        print(f"  Nearest to g†: log g = {nearest:.3f} "
              f"(Δ = {nearest - LOG_G_DAGGER:+.3f} dex)")

    print(f"\n  {'center':>8s} {'κ₄':>9s} {'dκ₄/dx':>10s}")
    print(f"  {'-'*30}")
    for i in range(len(centers_arr)):
        marker = " ← g†" if abs(centers_arr[i] - LOG_G_DAGGER) < 0.15 else ""
        print(f"  {centers_arr[i]:8.2f} {kurt_arr[i]:+9.2f} {dkurt[i]:+10.2f}{marker}")


# ================================================================
# 7. ENVIRONMENT-SPLIT KURTOSIS (DENSE vs FIELD)
# ================================================================
print("\n" + "=" * 72)
print("[7] Environment-split kurtosis analysis")
print("=" * 72)

# Environment classification — replicate from test_env_scatter_definitive.py
# SPARC galaxies: hardcoded UMa + group memberships
# Verheijen2001: all UMa cluster by definition
# Others: unclassified → exclude from env split

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

# Normalize name helper (strip quotes, normalize spacing)
import re
def _norm(n):
    n = n.strip().strip('"')
    n = re.sub(r'^(NGC|UGC|IC|DDO|ESO|PGC|CGCG|MCG|LSBC?|UGCA)\s+',
               r'\1', n, flags=re.IGNORECASE)
    return n.upper()

# Build Verheijen UMa set from loaded galaxies
verheijen_names = set()
for name, g in galaxies.items():
    if g.get('source') == 'Verheijen2001':
        verheijen_names.add(name)

# Also Korsaga galaxies that overlap with UMa names
korsaga_env = {}
for name, g in galaxies.items():
    if g.get('source') == 'Korsaga2019' or 'Korsaga' in g.get('mass_model_source', ''):
        norm = _norm(name)
        if norm in {_norm(u) for u in UMA_GALAXIES}:
            korsaga_env[name] = 'dense'
        elif norm in {_norm(g) for g in GROUP_MEMBERS}:
            korsaga_env[name] = 'dense'

def classify_env_extended(name, source):
    """Classify galaxy environment for extended Tier K dataset."""
    norm = _norm(name)
    # SPARC environment lists
    if norm in {_norm(u) for u in UMA_GALAXIES}:
        return 'dense'
    if norm in {_norm(g) for g in GROUP_MEMBERS}:
        return 'dense'
    # All Verheijen2001 galaxies are UMa cluster
    if source == 'Verheijen2001':
        return 'dense'
    if name in verheijen_names:
        return 'dense'
    # Korsaga matched to known groups
    if name in korsaga_env:
        return korsaga_env[name]
    # deBlok2002 LSBs: all isolated field galaxies (by selection)
    if source == 'deBlok2002':
        return 'field'
    # SPARC default: field
    if source in ('SPARC', 'THINGS'):
        return 'field'
    # Unknown: mark as unclassified
    return 'unclassified'

# Classify all Tier K points
sources_k = sources_all[tier_k_mask]
env_k = np.array([classify_env_extended(n, s) for n, s in zip(names_k, sources_k)])

dense_mask = env_k == 'dense'
field_mask = env_k == 'field'
unclass_mask = env_k == 'unclassified'

n_dense_gal = len(set(names_k[dense_mask]))
n_field_gal = len(set(names_k[field_mask]))
n_unclass_gal = len(set(names_k[unclass_mask]))

print(f"\n  Environment classification:")
print(f"    Dense:  {n_dense_gal} galaxies, {dense_mask.sum()} points")
print(f"    Field:  {n_field_gal} galaxies, {field_mask.sum()} points")
print(f"    Unclassified: {n_unclass_gal} galaxies, {unclass_mask.sum()} points")

# Source breakdown by env
for env_label, env_mask in [('Dense', dense_mask), ('Field', field_mask)]:
    src_k = sources_k[env_mask]
    print(f"\n    {env_label} sources:")
    for src in sorted(set(src_k)):
        n = np.sum(src_k == src)
        n_g = len(set(names_k[env_mask & (sources_k == src)]))
        print(f"      {src}: {n_g} galaxies, {n} points")

# Kurtosis profiles by environment
print(f"\n  --- Dense environment kurtosis ---")
bins_dense, _, _ = compute_binned_kurtosis(
    log_gbar_k[dense_mask], log_gobs_k[dense_mask])

print(f"\n  {'center':>8s} {'N':>6s} {'σ':>8s} {'κ₄':>8s}")
print(f"  {'-'*36}")
for b in bins_dense:
    if np.isnan(b['kurtosis']):
        continue
    marker = " ← g†" if abs(b['center'] - LOG_G_DAGGER) < 0.15 else ""
    print(f"  {b['center']:8.2f} {b['n']:6d} {b['sigma']:8.4f} "
          f"{b['kurtosis']:+8.2f}{marker}")

print(f"\n  --- Field environment kurtosis ---")
bins_field, _, _ = compute_binned_kurtosis(
    log_gbar_k[field_mask], log_gobs_k[field_mask])

print(f"\n  {'center':>8s} {'N':>6s} {'σ':>8s} {'κ₄':>8s}")
print(f"  {'-'*36}")
for b in bins_field:
    if np.isnan(b['kurtosis']):
        continue
    marker = " ← g†" if abs(b['center'] - LOG_G_DAGGER) < 0.15 else ""
    print(f"  {b['center']:8.2f} {b['n']:6d} {b['sigma']:8.4f} "
          f"{b['kurtosis']:+8.2f}{marker}")


# ================================================================
# 8. ENVIRONMENT COMPARISON AT g†
# ================================================================
print("\n" + "=" * 72)
print("[8] Environment comparison at g†")
print("=" * 72)

dense_at_gdagger = next((b for b in bins_dense
                         if abs(b['center'] - LOG_G_DAGGER) < 0.15
                         and not np.isnan(b['kurtosis'])), None)
field_at_gdagger = next((b for b in bins_field
                         if abs(b['center'] - LOG_G_DAGGER) < 0.15
                         and not np.isnan(b['kurtosis'])), None)

if dense_at_gdagger and field_at_gdagger:
    print(f"\n  At g† (log g_bar ≈ {LOG_G_DAGGER:.2f}):")
    print(f"    Dense: κ₄ = {dense_at_gdagger['kurtosis']:+.2f} "
          f"(N = {dense_at_gdagger['n']}, σ = {dense_at_gdagger['sigma']:.4f})")
    print(f"    Field: κ₄ = {field_at_gdagger['kurtosis']:+.2f} "
          f"(N = {field_at_gdagger['n']}, σ = {field_at_gdagger['sigma']:.4f})")

    # Bootstrap the difference
    print(f"\n  Bootstrap comparison (10,000 iterations)...")
    log_gobs_pred_k = rar_function(log_gbar_k)
    resid_k = log_gobs_k - log_gobs_pred_k

    edges = np.linspace(-12.5, -8.5, N_FINE + 1)
    # Find the g†-containing bin edges
    gdagger_bin_idx = None
    for j in range(N_FINE):
        center = (edges[j] + edges[j+1]) / 2
        if abs(center - LOG_G_DAGGER) < 0.15:
            gdagger_bin_idx = j
            break

    if gdagger_bin_idx is not None:
        lo_e, hi_e = edges[gdagger_bin_idx], edges[gdagger_bin_idx + 1]
        bin_mask = (log_gbar_k >= lo_e) & (log_gbar_k < hi_e)

        res_dense = resid_k[bin_mask & dense_mask]
        res_field = resid_k[bin_mask & field_mask]

        n_d, n_f = len(res_dense), len(res_field)
        print(f"    Dense points in bin: {n_d}")
        print(f"    Field points in bin: {n_f}")

        if n_d >= 20 and n_f >= 20:
            # Bootstrap kurtosis difference
            n_boot_env = 10000
            kurt_diff = np.zeros(n_boot_env)
            for b in range(n_boot_env):
                idx_d = np.random.randint(0, n_d, n_d)
                idx_f = np.random.randint(0, n_f, n_f)
                kurt_diff[b] = (kurtosis(res_dense[idx_d], fisher=True) -
                                kurtosis(res_field[idx_f], fisher=True))

            mean_diff = np.mean(kurt_diff)
            ci_lo = np.percentile(kurt_diff, 2.5)
            ci_hi = np.percentile(kurt_diff, 97.5)
            frac_positive = np.mean(kurt_diff > 0)

            print(f"\n    Kurtosis difference (dense − field):")
            print(f"      Mean: {mean_diff:+.2f}")
            print(f"      95% CI: [{ci_lo:+.1f}, {ci_hi:+.1f}]")
            print(f"      P(dense > field): {frac_positive:.3f}")
            print(f"      Significant (CI excludes 0): {ci_lo > 0 or ci_hi < 0}")

            # Find kurtosis derivative peaks by environment
            for env_label, env_bins in [('Dense', bins_dense), ('Field', bins_field)]:
                valid_env = [(b['center'], b['kurtosis']) for b in env_bins
                             if not np.isnan(b['kurtosis'])]
                if len(valid_env) >= 4:
                    c_env = np.array([v[0] for v in valid_env])
                    k_env = np.array([v[1] for v in valid_env])
                    dk_env = np.zeros_like(k_env)
                    dk_env[0] = (k_env[1] - k_env[0]) / (c_env[1] - c_env[0])
                    dk_env[-1] = (k_env[-1] - k_env[-2]) / (c_env[-1] - c_env[-2])
                    for i in range(1, len(k_env) - 1):
                        dk_env[i] = (k_env[i+1] - k_env[i-1]) / (c_env[i+1] - c_env[i-1])

                    env_peaks = []
                    for i in range(len(dk_env) - 1):
                        if dk_env[i] > 0 and dk_env[i+1] < 0:
                            x_cross = (c_env[i] - dk_env[i] * (c_env[i+1] - c_env[i])
                                       / (dk_env[i+1] - dk_env[i]))
                            env_peaks.append(float(x_cross))

                    if env_peaks:
                        near_env = min(env_peaks, key=lambda x: abs(x - LOG_G_DAGGER))
                        print(f"\n    {env_label} κ₄ derivative peak nearest g†: "
                              f"log g = {near_env:.3f} (Δ = {near_env - LOG_G_DAGGER:+.3f} dex)")
                    else:
                        print(f"\n    {env_label}: no κ₄ derivative peak found")
        else:
            print(f"    Insufficient points for bootstrap ({n_d} dense, {n_f} field)")
            print(f"    Need ≥20 per environment in g† bin")

env_results = {
    'n_dense_galaxies': n_dense_gal,
    'n_field_galaxies': n_field_gal,
    'n_unclassified_galaxies': n_unclass_gal,
    'dense_points': int(dense_mask.sum()),
    'field_points': int(field_mask.sum()),
}
if dense_at_gdagger:
    env_results['dense_at_gdagger'] = {
        'kurtosis': round(float(dense_at_gdagger['kurtosis']), 3),
        'n': dense_at_gdagger['n'],
        'sigma': round(float(dense_at_gdagger['sigma']), 5),
    }
if field_at_gdagger:
    env_results['field_at_gdagger'] = {
        'kurtosis': round(float(field_at_gdagger['kurtosis']), 3),
        'n': field_at_gdagger['n'],
        'sigma': round(float(field_at_gdagger['sigma']), 5),
    }


# ================================================================
# 9. SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

# Collect all kurtosis values for comparison
tier_k_at_gdagger = next((b for b in bins_k
                          if abs(b['center'] - LOG_G_DAGGER) < 0.15
                          and not np.isnan(b['kurtosis'])), None)
sparc_at_gdagger = next((b for b in bins_s
                         if abs(b['center'] - LOG_G_DAGGER) < 0.15
                         and not np.isnan(b['kurtosis'])), None)
lcdm_at_gdagger_mean = np.mean(lcdm_at_gdagger) if lcdm_at_gdagger else np.nan

print(f"\n  Kurtosis at g† (log g_bar ≈ {LOG_G_DAGGER:.2f}):")
if tier_k_at_gdagger:
    print(f"    Tier K (250 gal): κ₄ = {tier_k_at_gdagger['kurtosis']:.2f}")
if sparc_at_gdagger:
    print(f"    SPARC (126 gal):  κ₄ = {sparc_at_gdagger['kurtosis']:.2f}")
print(f"    ΛCDM mock:        κ₄ = {lcdm_at_gdagger_mean:.2f}")

# Verdict
if tier_k_at_gdagger and not np.isnan(lcdm_at_gdagger_mean):
    observed = tier_k_at_gdagger['kurtosis']
    lcdm_val = lcdm_at_gdagger_mean

    if observed > 3 * lcdm_val and observed > 5.0:
        verdict = "DISCRIMINATING"
        detail = (f"κ₄ = {observed:.1f} at g† is {observed/lcdm_val:.0f}× "
                  f"the ΛCDM value ({lcdm_val:.1f})")
    elif observed > 2 * lcdm_val:
        verdict = "SUGGESTIVE"
        detail = f"κ₄ elevated but not decisively ({observed:.1f} vs {lcdm_val:.1f})"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"κ₄ not significantly elevated ({observed:.1f} vs {lcdm_val:.1f})"

    print(f"\n  Verdict: {verdict}")
    print(f"    {detail}")
    print(f"    BEC prediction (kurtosis spike at phase boundary): "
          f"{'CONFIRMED' if verdict == 'DISCRIMINATING' else 'PARTIALLY CONFIRMED' if verdict == 'SUGGESTIVE' else 'NOT CONFIRMED'}")


# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    'test_name': 'kurtosis_phase_transition',
    'description': (
        'Tests whether excess kurtosis of RAR residuals peaks at g†, '
        'the predicted signature of a continuous phase transition in the '
        'BEC framework. Compares Tier K observed data to ΛCDM semi-analytic mock.'
    ),
    'g_dagger_log': LOG_G_DAGGER,
    'n_bootstrap': N_BOOTSTRAP,
    'tier_k': {
        'n_galaxies': n_gal_k,
        'n_points': int(len(log_gobs_k)),
        'kurtosis_profile': [{k: round(v, 5) if isinstance(v, float) else v
                              for k, v in b.items()} for b in bins_k
                             if not np.isnan(b.get('kurtosis', np.nan))],
        'bootstrap_ci': [{k: round(v, 5) if isinstance(v, float) else v
                          for k, v in b.items()} for b in boot_k
                         if not np.isnan(b.get('kurtosis_mean', np.nan))],
    },
    'sparc_only': {
        'n_galaxies': n_gal_s,
        'n_points': int(len(log_gobs_s)),
        'kurtosis_profile': [{k: round(v, 5) if isinstance(v, float) else v
                              for k, v in b.items()} for b in bins_s
                             if not np.isnan(b.get('kurtosis', np.nan))],
        'bootstrap_ci': [{k: round(v, 5) if isinstance(v, float) else v
                          for k, v in b.items()} for b in boot_s
                         if not np.isnan(b.get('kurtosis_mean', np.nan))],
    },
    'lcdm_mock': {
        'n_realizations': N_LCDM_REALIZATIONS,
        'n_galaxies_per': 500,
        'kurtosis_profiles': [
            [{k: round(v, 5) if isinstance(v, float) else v
              for k, v in b.items()} for b in prof
             if not np.isnan(b.get('kurtosis', np.nan))]
            for prof in lcdm_kurtosis_profiles
        ],
    },
}

# Add comparison summary
if tier_k_at_gdagger and not np.isnan(lcdm_at_gdagger_mean):
    results['comparison'] = {
        'tier_k_kurtosis_at_gdagger': round(float(tier_k_at_gdagger['kurtosis']), 3),
        'sparc_kurtosis_at_gdagger': round(float(sparc_at_gdagger['kurtosis']), 3) if sparc_at_gdagger else None,
        'lcdm_mean_kurtosis_at_gdagger': round(float(lcdm_at_gdagger_mean), 3),
        'ratio_observed_over_lcdm': round(float(tier_k_at_gdagger['kurtosis'] / lcdm_at_gdagger_mean), 2) if lcdm_at_gdagger_mean > 0 else None,
        'verdict': verdict,
    }
    if gdagger_boot:
        results['comparison']['bootstrap_ci_at_gdagger'] = {
            'lo': round(float(gdagger_boot['kurtosis_ci_lo']), 3),
            'hi': round(float(gdagger_boot['kurtosis_ci_hi']), 3),
        }

if kurt_peaks:
    results['kurtosis_derivative_peaks'] = {
        'locations': [round(x, 3) for x in kurt_peaks],
        'nearest_to_gdagger': round(float(nearest), 3),
        'distance_from_gdagger': round(float(nearest - LOG_G_DAGGER), 3),
    }

results['environment_split'] = env_results
results['environment_split']['dense_kurtosis_profile'] = [
    {k: round(v, 5) if isinstance(v, float) else v for k, v in b.items()}
    for b in bins_dense if not np.isnan(b.get('kurtosis', np.nan))
]
results['environment_split']['field_kurtosis_profile'] = [
    {k: round(v, 5) if isinstance(v, float) else v for k, v in b.items()}
    for b in bins_field if not np.isnan(b.get('kurtosis', np.nan))
]

outpath = os.path.join(RESULTS_DIR, 'summary_kurtosis_phase_transition.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved: {outpath}")
print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
