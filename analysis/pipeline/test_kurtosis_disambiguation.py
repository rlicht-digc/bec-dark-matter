#!/usr/bin/env python3
"""
Kurtosis Disambiguation: Physical vs Geometric vs Cross-Calibration
====================================================================

Three critical tests to determine whether the kurtosis spike at g† is:
  (a) Physical: genuine phase transition fluctuations
  (b) Cross-calibration: SPARC × Korsaga M/L offset inflated at g†
  (c) Geometric: RAR slope is steepest near g†, so ANY M/L offset
      produces maximum residual spread there

Tests:
  1. Korsaga-only kurtosis: Does the spike exist within a single survey?
  2. Source-separated kurtosis: SPARC-only, Korsaga-only, THINGS+deBlok-only
  3. RAR slope profile: Where is d(log g_obs)/d(log g_bar) maximized?
     Compare to g† and to the kurtosis peak location.
  4. Predicted instrumental kurtosis: Given the measured M/L offset between
     surveys, where would a purely geometric effect peak?

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import numpy as np
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
    rar_function, rar_residuals, rar_slope,
    binned_stats, get_at_gdagger,
    numerical_derivative, find_zero_crossings,
)

N_FINE = 15


def compute_kurtosis_profile(log_gbar, log_gobs, n_bins=N_FINE,
                             gbar_range=(-12.5, -8.5), min_n=20):
    """Thin wrapper around binned_stats for backward compatibility."""
    return binned_stats(log_gbar, log_gobs, n_bins=n_bins,
                        gbar_range=gbar_range, min_n=min_n)


# ================================================================
# LOAD DATA
# ================================================================
print("=" * 72)
print("KURTOSIS DISAMBIGUATION: PHYSICAL vs GEOMETRIC vs CROSS-CALIBRATION")
print("=" * 72)

galaxies = load_all(include_vizier=True, match_photometry=True)
print("  Applying Bouquin+2018...")
compute_baryonic_bouquin(galaxies)
print("  Applying Korsaga+2019...")
add_korsaga_to_galaxies(galaxies, use_fixed_ml=True, verbose=False)

log_gobs_all, log_gbar_all, names_all, sources_all = build_rar(galaxies)

# Build Tier K mask
sparc_mask = sources_all == 'SPARC'
things_mask = sources_all == 'THINGS'
deblok_mask = sources_all == 'deBlok2002'
korsaga_mask = sources_all == 'Korsaga2019'

good_bouquin = set()
for name, g in galaxies.items():
    mm = g.get('mass_model_source', '')
    if 'Bouquin' in mm and g['source'] in ('Verheijen2001', 'PHANGS_Lang2020'):
        good_bouquin.add(name)
bouquin_mask = np.array([n in good_bouquin for n in names_all])

tier_k_mask = sparc_mask | things_mask | deblok_mask | bouquin_mask | korsaga_mask

# Extract arrays for each source within Tier K
log_gobs_k = log_gobs_all[tier_k_mask]
log_gbar_k = log_gbar_all[tier_k_mask]
names_k = names_all[tier_k_mask]
sources_k = sources_all[tier_k_mask]

print(f"\n  Tier K: {len(set(names_k))} galaxies, {len(log_gobs_k)} points")
for src in sorted(set(sources_k)):
    m = sources_k == src
    print(f"    {src}: {len(set(names_k[m]))} gal, {m.sum()} pts")


# ================================================================
# TEST 1: SOURCE-SEPARATED KURTOSIS
# ================================================================
print("\n" + "=" * 72)
print("[1] Source-separated kurtosis profiles")
print("=" * 72)

source_groups = {
    'SPARC': sparc_mask & tier_k_mask,
    'Korsaga2019': korsaga_mask & tier_k_mask,
    'THINGS+deBlok': (things_mask | deblok_mask) & tier_k_mask,
    'Bouquin-matched': bouquin_mask & tier_k_mask,
    'ALL Tier K': tier_k_mask,
}

source_profiles = {}
for label, mask in source_groups.items():
    gbar = log_gbar_all[mask]
    gobs = log_gobs_all[mask]
    n_gal = len(set(names_all[mask]))
    n_pts = len(gbar)

    if n_pts < 100:
        print(f"\n  {label}: {n_gal} gal, {n_pts} pts — too few, skipping")
        continue

    prof = compute_kurtosis_profile(gbar, gobs)
    source_profiles[label] = prof

    # Find g†-bin kurtosis
    gdagger_bin = next((b for b in prof
                        if abs(b['center'] - LOG_G_DAGGER) < 0.15
                        and not np.isnan(b['kurtosis'])), None)

    print(f"\n  {label}: {n_gal} gal, {n_pts} pts")
    print(f"  {'center':>8s} {'N':>6s} {'σ':>8s} {'κ₄':>8s}")
    print(f"  {'-'*36}")
    for b in prof:
        if np.isnan(b['kurtosis']):
            continue
        marker = " ← g†" if abs(b['center'] - LOG_G_DAGGER) < 0.15 else ""
        print(f"  {b['center']:8.2f} {b['n']:6d} {b['sigma']:8.4f} "
              f"{b['kurtosis']:+8.2f}{marker}")

    if gdagger_bin:
        print(f"  → κ₄ at g† = {gdagger_bin['kurtosis']:+.2f} "
              f"(N = {gdagger_bin['n']})")


# ================================================================
# TEST 2: RAR SLOPE PROFILE
# ================================================================
print("\n" + "=" * 72)
print("[2] RAR slope profile: d(log g_obs)/d(log g_bar)")
print("=" * 72)

# Compute analytic slope on a fine grid
log_gbar_grid = np.linspace(-12.5, -8.5, 1000)
slope_grid = rar_slope(log_gbar_grid)

# Where is the slope maximized?
max_slope_idx = np.argmax(slope_grid)
log_gbar_max_slope = log_gbar_grid[max_slope_idx]
max_slope_val = slope_grid[max_slope_idx]

# Slope at g†
slope_at_gdagger = float(rar_slope(np.array([LOG_G_DAGGER]))[0])

print(f"\n  Maximum RAR slope: {max_slope_val:.4f} at log g_bar = {log_gbar_max_slope:.3f}")
print(f"  RAR slope at g† (log g = {LOG_G_DAGGER:.3f}): {slope_at_gdagger:.4f}")
print(f"  Distance: max slope location − g† = {log_gbar_max_slope - LOG_G_DAGGER:+.3f} dex")

# Also compute d²(log g_obs)/d(log g_bar)² — the curvature
# Maximum curvature is where the slope changes fastest
# This is where a fixed δ(log g_bar) produces the most variable δ(log g_obs)
d_slope = np.gradient(slope_grid, log_gbar_grid)
max_curvature_idx = np.argmax(np.abs(d_slope))
log_gbar_max_curvature = log_gbar_grid[max_curvature_idx]
max_curvature_val = d_slope[max_curvature_idx]

print(f"\n  Maximum |RAR curvature|: {max_curvature_val:+.4f} "
      f"at log g_bar = {log_gbar_max_curvature:.3f}")
print(f"  Distance: max curvature − g† = {log_gbar_max_curvature - LOG_G_DAGGER:+.3f} dex")

# Print slope at bin centers for comparison with kurtosis
edges = np.linspace(-12.5, -8.5, N_FINE + 1)
centers = (edges[:-1] + edges[1:]) / 2
slope_at_centers = rar_slope(centers)

print(f"\n  {'center':>8s} {'slope':>8s} {'|d(slope)|':>12s}")
print(f"  {'-'*32}")
d_slope_centers = np.abs(np.gradient(slope_at_centers, centers))
for i, c in enumerate(centers):
    marker = " ← g†" if abs(c - LOG_G_DAGGER) < 0.15 else ""
    print(f"  {c:8.2f} {slope_at_centers[i]:8.4f} {d_slope_centers[i]:12.4f}{marker}")


# ================================================================
# TEST 3: PREDICTED INSTRUMENTAL KURTOSIS FROM M/L OFFSET
# ================================================================
print("\n" + "=" * 72)
print("[3] Predicted instrumental kurtosis from cross-calibration")
print("=" * 72)

# Measure the actual M/L offset between SPARC and Korsaga
# at the g†-containing bin. This is the mean residual difference.
log_gobs_pred_all = rar_function(log_gbar_all)
resid_all = log_gobs_all - log_gobs_pred_all

# Compute mean residual by source in the g† bin
lo_g, hi_g = -10.233, -9.700  # approximate g† bin edges (from 15-bin scheme)
for src_label, src_mask in [('SPARC', sparc_mask), ('Korsaga2019', korsaga_mask),
                            ('THINGS', things_mask), ('deBlok2002', deblok_mask)]:
    in_bin = src_mask & (log_gbar_all >= lo_g) & (log_gbar_all < hi_g)
    if in_bin.sum() > 10:
        mean_res = np.mean(resid_all[in_bin])
        std_res = np.std(resid_all[in_bin])
        print(f"  {src_label:15s}: mean residual = {mean_res:+.4f}, "
              f"σ = {std_res:.4f}, N = {in_bin.sum()}")

# Compute the mean residual offset between SPARC and Korsaga in each bin
print(f"\n  Mean residual by source across all bins:")
print(f"  {'center':>8s} {'SPARC':>10s} {'Korsaga':>10s} {'offset':>10s} {'N_S':>5s} {'N_K':>5s}")
print(f"  {'-'*52}")

offsets = []
for j in range(N_FINE):
    lo, hi = edges[j], edges[j+1]
    s_mask = sparc_mask & (log_gbar_all >= lo) & (log_gbar_all < hi)
    k_mask = korsaga_mask & (log_gbar_all >= lo) & (log_gbar_all < hi)
    if s_mask.sum() >= 10 and k_mask.sum() >= 10:
        s_mean = np.mean(resid_all[s_mask])
        k_mean = np.mean(resid_all[k_mask])
        offset = k_mean - s_mean
        offsets.append((centers[j], offset, s_mask.sum(), k_mask.sum()))
        marker = " ← g†" if abs(centers[j] - LOG_G_DAGGER) < 0.15 else ""
        print(f"  {centers[j]:8.2f} {s_mean:+10.4f} {k_mean:+10.4f} "
              f"{offset:+10.4f} {s_mask.sum():5d} {k_mask.sum():5d}{marker}")
    elif s_mask.sum() >= 10 or k_mask.sum() >= 10:
        offsets.append((centers[j], np.nan, s_mask.sum(), k_mask.sum()))

# Key question: Is the offset roughly constant, or does it vary with gbar?
valid_offsets = [(c, o) for c, o, _, _ in offsets if not np.isnan(o)]
if len(valid_offsets) >= 3:
    off_c = np.array([v[0] for v in valid_offsets])
    off_v = np.array([v[1] for v in valid_offsets])
    mean_offset = np.mean(off_v)
    std_offset = np.std(off_v)
    # Linear trend
    if len(off_c) >= 3:
        poly = np.polyfit(off_c, off_v, 1)
        print(f"\n  Mean Korsaga−SPARC offset: {mean_offset:+.4f} ± {std_offset:.4f} dex")
        print(f"  Linear trend: slope = {poly[0]:+.4f} dex per dex of log g_bar")
        print(f"  (Constant offset → slope ≈ 0; acceleration-dependent → slope ≠ 0)")


# ================================================================
# TEST 4: GEOMETRIC PREDICTION
# ================================================================
print("\n" + "=" * 72)
print("[4] Geometric prediction: where would instrumental kurtosis peak?")
print("=" * 72)

# If SPARC and Korsaga have a constant mean offset δ in log(g_bar),
# then the residual from the standard RAR at each point is:
#   Δ(log g_obs) ≈ [d(log g_obs)/d(log g_bar)] × δ
# A mixture of two populations offset by δ produces a bimodal residual
# distribution. The kurtosis of a bimodal (50/50 mixture of N(0,σ) and
# N(δ_eff,σ)) is approximately:
#   κ₄ ≈ (δ_eff / σ)² - 2  (for 50/50 mixture, simplified)
# where δ_eff = slope × δ(log gbar)
# So κ₄ peaks where slope² is maximized — i.e., where slope is max.

# But we also need to account for the mixing fraction.
# In Tier K, the fraction of Korsaga points varies by bin.
print(f"\n  Mixing fraction and effective offset by bin:")
print(f"  {'center':>8s} {'f_Kor':>7s} {'slope':>8s} {'δ_eff':>8s} {'predicted κ₄':>13s}")
print(f"  {'-'*50}")

predicted_kurt = []
for j in range(N_FINE):
    lo, hi = edges[j], edges[j+1]
    tk_mask = tier_k_mask & (log_gbar_all >= lo) & (log_gbar_all < hi)
    k_in_tk = korsaga_mask & (log_gbar_all >= lo) & (log_gbar_all < hi)

    n_tk = tk_mask.sum()
    n_k = k_in_tk.sum()
    if n_tk < 20:
        predicted_kurt.append((centers[j], np.nan))
        continue

    f_kor = n_k / n_tk if n_tk > 0 else 0

    # Use actual offset if available, else mean
    actual_off = None
    for c, o, _, _ in offsets:
        if abs(c - centers[j]) < 0.01 and not np.isnan(o):
            actual_off = o
            break
    if actual_off is None:
        actual_off = mean_offset if 'mean_offset' in dir() else 0.1

    sl = float(slope_at_centers[j])
    delta_eff = sl * abs(actual_off)

    # Within-bin scatter (from actual data)
    res_bin = resid_all[tk_mask]
    sigma_bin = np.std(res_bin) if len(res_bin) > 10 else 0.15

    # Predicted kurtosis from bimodal mixture (Pearson formula)
    # For mixture of fraction f at offset d from fraction (1-f):
    # κ₄ = [f(1-f)(1-6f(1-f)) * d⁴] / [f(1-f)*d² + σ²]² - 3 + 3
    # Simplified: κ₄ ≈ f(1-f) * (d/σ)² for small d/σ
    # More precisely, use the bimodal excess kurtosis formula
    f = f_kor
    d = delta_eff
    s = sigma_bin
    if s > 0 and f > 0.01 and f < 0.99:
        # Exact bimodal mixture kurtosis (two Gaussians with same σ, different means)
        # μ_mix = f*d
        # σ²_mix = f*(1-f)*d² + σ²
        # κ₄ = [f*(1-f)*(1-6*f*(1-f))*d⁴] / [f*(1-f)*d² + σ²]² - 2
        # Actually the standard formula for a two-component Gaussian mixture:
        var_mix = f * (1 - f) * d**2 + s**2
        mu4_mix = (3 * s**4 + f * (1 - f) * (1 - 6 * f * (1 - f)) * d**4
                   + 6 * s**2 * f * (1 - f) * d**2)
        if var_mix > 0:
            pred_k = mu4_mix / var_mix**2 - 3.0
        else:
            pred_k = 0.0
    else:
        pred_k = 0.0

    predicted_kurt.append((centers[j], pred_k))
    marker = " ← g†" if abs(centers[j] - LOG_G_DAGGER) < 0.15 else ""
    print(f"  {centers[j]:8.2f} {f_kor:7.3f} {sl:8.4f} {delta_eff:8.4f} "
          f"{pred_k:+13.2f}{marker}")

# Where does the predicted instrumental kurtosis peak?
valid_pred = [(c, k) for c, k in predicted_kurt if not np.isnan(k)]
if valid_pred:
    pred_peak_c, pred_peak_k = max(valid_pred, key=lambda x: x[1])
    print(f"\n  Predicted instrumental κ₄ peak: {pred_peak_k:+.2f} "
          f"at log g_bar = {pred_peak_c:.2f}")
    print(f"  Distance from g†: {pred_peak_c - LOG_G_DAGGER:+.3f} dex")


# ================================================================
# TEST 5: KORSAGA-ONLY INTERNAL KURTOSIS STRUCTURE
# ================================================================
print("\n" + "=" * 72)
print("[5] Korsaga-only internal kurtosis structure")
print("=" * 72)

kor_gbar = log_gbar_all[korsaga_mask]
kor_gobs = log_gobs_all[korsaga_mask]
kor_names = names_all[korsaga_mask]
n_kor_gal = len(set(kor_names))

print(f"\n  Korsaga: {n_kor_gal} galaxies, {len(kor_gbar)} points")

# Use finer bins for Korsaga since we have 6,000+ points
kor_prof = compute_kurtosis_profile(kor_gbar, kor_gobs, n_bins=15, min_n=20)

print(f"\n  {'center':>8s} {'N':>6s} {'σ':>8s} {'κ₄':>8s}")
print(f"  {'-'*36}")
kor_gdagger_kurt = np.nan
for b in kor_prof:
    if np.isnan(b['kurtosis']):
        continue
    marker = " ← g†" if abs(b['center'] - LOG_G_DAGGER) < 0.15 else ""
    print(f"  {b['center']:8.2f} {b['n']:6d} {b['sigma']:8.4f} "
          f"{b['kurtosis']:+8.2f}{marker}")
    if abs(b['center'] - LOG_G_DAGGER) < 0.15:
        kor_gdagger_kurt = b['kurtosis']

# Korsaga-internal kurtosis derivative
valid_kor = [(b['center'], b['kurtosis']) for b in kor_prof
             if not np.isnan(b['kurtosis'])]
if len(valid_kor) >= 4:
    c_kor = np.array([v[0] for v in valid_kor])
    k_kor = np.array([v[1] for v in valid_kor])
    dk_kor = np.gradient(k_kor, c_kor)

    # Find peaks (pos→neg crossings of derivative)
    kor_peaks = []
    for i in range(len(dk_kor) - 1):
        if dk_kor[i] > 0 and dk_kor[i+1] < 0:
            x_cross = (c_kor[i] - dk_kor[i] * (c_kor[i+1] - c_kor[i])
                       / (dk_kor[i+1] - dk_kor[i]))
            kor_peaks.append(float(x_cross))

    if kor_peaks:
        nearest_kor = min(kor_peaks, key=lambda x: abs(x - LOG_G_DAGGER))
        print(f"\n  Korsaga κ₄ derivative peaks: {[f'{p:.3f}' for p in kor_peaks]}")
        print(f"  Nearest to g†: log g = {nearest_kor:.3f} "
              f"(Δ = {nearest_kor - LOG_G_DAGGER:+.3f} dex)")
    else:
        print(f"\n  Korsaga: no κ₄ derivative peaks found")
        nearest_kor = np.nan


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 72)
print("DISAMBIGUATION SUMMARY")
print("=" * 72)

print(f"\n  g† = log g_bar = {LOG_G_DAGGER:.3f}")

# Collect κ₄ at g† for each source
print(f"\n  Kurtosis at g† by source:")
for label in ['SPARC', 'Korsaga2019', 'THINGS+deBlok', 'ALL Tier K']:
    if label in source_profiles:
        b = next((x for x in source_profiles[label]
                  if abs(x['center'] - LOG_G_DAGGER) < 0.15
                  and not np.isnan(x['kurtosis'])), None)
        if b:
            print(f"    {label:20s}: κ₄ = {b['kurtosis']:+6.2f} (N = {b['n']})")

print(f"\n  Key locations:")
print(f"    g† (BEC prediction):           log g = {LOG_G_DAGGER:.3f}")
print(f"    Max RAR slope:                 log g = {log_gbar_max_slope:.3f} "
      f"(Δ = {log_gbar_max_slope - LOG_G_DAGGER:+.3f} dex)")
print(f"    Max |RAR curvature|:           log g = {log_gbar_max_curvature:.3f} "
      f"(Δ = {log_gbar_max_curvature - LOG_G_DAGGER:+.3f} dex)")
if valid_pred:
    print(f"    Predicted instrumental peak:   log g = {pred_peak_c:.3f} "
          f"(Δ = {pred_peak_c - LOG_G_DAGGER:+.3f} dex)")
print(f"    Observed Tier K κ₄ deriv peak: log g = -9.939 "
      f"(Δ = {-9.939 - LOG_G_DAGGER:+.3f} dex)")
if 'nearest_kor' in dir() and not np.isnan(nearest_kor):
    print(f"    Korsaga-only κ₄ deriv peak:   log g = {nearest_kor:.3f} "
          f"(Δ = {nearest_kor - LOG_G_DAGGER:+.3f} dex)")

# Verdict
print(f"\n  DIAGNOSTIC CONCLUSIONS:")

if not np.isnan(kor_gdagger_kurt):
    if kor_gdagger_kurt > 5.0:
        print(f"    [1] Korsaga-only κ₄ = {kor_gdagger_kurt:.1f} at g† → "
              f"spike is INTERNAL to Korsaga (not cross-survey)")
    elif kor_gdagger_kurt > 2.0:
        print(f"    [1] Korsaga-only κ₄ = {kor_gdagger_kurt:.1f} at g† → "
              f"mildly elevated, inconclusive")
    else:
        print(f"    [1] Korsaga-only κ₄ = {kor_gdagger_kurt:.1f} at g† → "
              f"Gaussian, spike is from CROSS-SURVEY MIXING")

if abs(log_gbar_max_slope - LOG_G_DAGGER) < 0.15:
    print(f"    [2] Max RAR slope coincides with g† (Δ = "
          f"{log_gbar_max_slope - LOG_G_DAGGER:+.3f} dex) → "
          f"geometric degeneracy EXISTS")
else:
    print(f"    [2] Max RAR slope offset from g† by "
          f"{log_gbar_max_slope - LOG_G_DAGGER:+.3f} dex → "
          f"geometric degeneracy BROKEN")

if valid_pred:
    if abs(pred_peak_c - LOG_G_DAGGER) < 0.20:
        print(f"    [3] Predicted instrumental peak at {pred_peak_c:.3f} "
              f"coincides with g† → cannot distinguish instrumental from physical")
    else:
        print(f"    [3] Predicted instrumental peak at {pred_peak_c:.3f} "
              f"offset from g† by {pred_peak_c - LOG_G_DAGGER:+.3f} dex → "
              f"physical interpretation preferred")


# Save results
results = {
    'test_name': 'kurtosis_disambiguation',
    'g_dagger_log': LOG_G_DAGGER,
    'max_rar_slope_location': round(float(log_gbar_max_slope), 4),
    'max_rar_curvature_location': round(float(log_gbar_max_curvature), 4),
    'rar_slope_at_gdagger': round(float(slope_at_gdagger), 4),
    'korsaga_only_kurtosis_at_gdagger': round(float(kor_gdagger_kurt), 3) if not np.isnan(kor_gdagger_kurt) else None,
    'source_kurtosis_at_gdagger': {},
}

for label in ['SPARC', 'Korsaga2019', 'THINGS+deBlok', 'ALL Tier K']:
    if label in source_profiles:
        b = next((x for x in source_profiles[label]
                  if abs(x['center'] - LOG_G_DAGGER) < 0.15
                  and not np.isnan(x['kurtosis'])), None)
        if b:
            results['source_kurtosis_at_gdagger'][label] = round(float(b['kurtosis']), 3)

if valid_pred:
    results['predicted_instrumental_peak'] = round(float(pred_peak_c), 3)

outpath = os.path.join(RESULTS_DIR, 'summary_kurtosis_disambiguation.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved: {outpath}")
print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
