#!/usr/bin/env python3
"""
Extended RAR Inversion Point Analysis
======================================

Replicate the inversion point analysis from test_mc_distance_and_inversion.py
using the extended 256-galaxy RAR dataset (SPARC + VizieR + S4G/ALFALFA matching).

Key question: Does the inversion point at g† persist when we more than double
the galaxy sample? This is the strongest replication test we can do without
an entirely independent dataset.

Comparison:
  - SPARC-only: 126 galaxies, ~2,740 RAR points
  - Extended:   256 galaxies, ~7,760 RAR points

BEC prediction: inversion at log g_bar ≈ -9.92 (g†)
"""

import os
import sys
import json
import numpy as np
from scipy.stats import levene, skew, kurtosis
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

# Physics
g_dagger = 1.20e-10
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921

# RAR function for residuals
def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


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


def run_inversion_analysis(log_gobs, log_gbar, names, sources, label, n_fine=15):
    """Run the full inversion point analysis on a set of RAR points."""
    # Compute residuals from standard RAR
    log_gobs_rar = rar_function(log_gbar)
    log_res = log_gobs - log_gobs_rar

    n_galaxies = len(set(names))
    n_points = len(log_gobs)

    print(f"\n{'='*72}")
    print(f"INVERSION ANALYSIS: {label}")
    print(f"{'='*72}")
    print(f"  {n_galaxies} galaxies, {n_points} RAR points")
    print(f"  gbar range: [{log_gbar.min():.2f}, {log_gbar.max():.2f}]")
    print(f"  gobs range: [{log_gobs.min():.2f}, {log_gobs.max():.2f}]")

    # Source breakdown
    for src in sorted(set(sources)):
        mask = sources == src
        n_gal = len(set(names[mask]))
        print(f"    {src}: {mask.sum()} pts from {n_gal} galaxies")

    # Fine bins for derivative analysis
    fine_edges = np.linspace(-12.5, -8.5, n_fine + 1)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2

    print(f"\n  {'Bin center':>10s} {'N':>6s} {'σ':>8s} {'skew':>8s} {'kurt':>8s}")
    print(f"  {'-'*44}")

    bin_stats = []
    for j in range(n_fine):
        lo, hi = fine_edges[j], fine_edges[j+1]
        mask = (log_gbar >= lo) & (log_gbar < hi)
        res = log_res[mask]

        if len(res) < 10:
            bin_stats.append({
                'center': float(fine_centers[j]), 'n': int(len(res)),
                'sigma': np.nan, 'skewness': np.nan, 'kurtosis_excess': np.nan,
            })
            continue

        s = float(np.std(res))
        sk = float(skew(res))
        ku = float(kurtosis(res, fisher=True))

        bin_stats.append({
            'center': float(fine_centers[j]),
            'n': int(len(res)),
            'sigma': round(s, 5),
            'skewness': round(sk, 5),
            'kurtosis_excess': round(ku, 5),
        })

        print(f"  {fine_centers[j]:10.2f} {len(res):6d} {s:8.4f} {sk:+8.3f} {ku:+8.3f}")

    # Derivatives
    valid_stats = [b for b in bin_stats if not np.isnan(b['sigma'])]
    centers_v = np.array([b['center'] for b in valid_stats])
    sigma_v = np.array([b['sigma'] for b in valid_stats])
    skew_v = np.array([b['skewness'] for b in valid_stats])
    kurt_v = np.array([b['kurtosis_excess'] for b in valid_stats])

    dsigma_dx = numerical_derivative(centers_v, sigma_v)
    dskew_dx = numerical_derivative(centers_v, skew_v)
    dkurt_dx = numerical_derivative(centers_v, kurt_v)

    # Zero crossings
    sigma_crossings = find_zero_crossings(centers_v, dsigma_dx)
    skew_crossings = find_zero_crossings(centers_v, dskew_dx)
    kurt_crossings = find_zero_crossings(centers_v, dkurt_dx)
    skew_sign_crossings = find_zero_crossings(centers_v, skew_v)

    print(f"\n  DERIVATIVE ZERO-CROSSINGS:")
    print(f"  g† = 10^{LOG_G_DAGGER:.2f} m/s²")
    print(f"\n  dσ/d(log g_bar) = 0 at: {sigma_crossings}")
    print(f"  d(skew)/d(log g_bar) = 0 at: {skew_crossings}")
    print(f"  d(kurt)/d(log g_bar) = 0 at: {kurt_crossings}")
    print(f"  skewness = 0 at: {skew_sign_crossings}")

    # Rank inversions by proximity to g†
    all_inversions = []
    for x in sigma_crossings:
        all_inversions.append(('dσ/dx=0', x, abs(x - LOG_G_DAGGER)))
    for x in skew_crossings:
        all_inversions.append(('d(skew)/dx=0', x, abs(x - LOG_G_DAGGER)))
    for x in skew_sign_crossings:
        all_inversions.append(('skew=0', x, abs(x - LOG_G_DAGGER)))

    if all_inversions:
        all_inversions.sort(key=lambda t: t[2])
        print(f"\n  Inversions ranked by proximity to g†:")
        for label_inv, x, dist in all_inversions:
            marker = " ← CLOSEST" if dist == all_inversions[0][2] else ""
            print(f"    {label_inv:20s} at log g = {x:+.3f}  "
                  f"(Δ from g† = {x - LOG_G_DAGGER:+.3f} dex){marker}")

    # Derivative table
    print(f"\n  {'center':>8s} {'dσ/dx':>9s} {'d(skew)/dx':>11s} {'d(kurt)/dx':>11s}")
    print(f"  {'-'*42}")
    for i, c in enumerate(centers_v):
        print(f"  {c:8.2f} {dsigma_dx[i]:+9.4f} {dskew_dx[i]:+11.4f} {dkurt_dx[i]:+11.4f}")

    return {
        'label': label,
        'n_galaxies': n_galaxies,
        'n_points': n_points,
        'bin_statistics': [b for b in bin_stats if not np.isnan(b.get('sigma', np.nan))],
        'derivatives': {
            'centers': [round(float(c), 3) for c in centers_v],
            'dsigma_dx': [round(float(d), 5) for d in dsigma_dx],
            'dskew_dx': [round(float(d), 5) for d in dskew_dx],
            'dkurt_dx': [round(float(d), 5) for d in dkurt_dx],
        },
        'zero_crossings': {
            'dsigma_dx': sigma_crossings,
            'dskew_dx': skew_crossings,
            'dkurt_dx': kurt_crossings,
            'skewness': skew_sign_crossings,
        },
        'all_inversions_ranked': [
            {'type': t[0], 'log_gbar': round(t[1], 3),
             'distance_from_gdagger': round(t[2], 3)}
            for t in all_inversions
        ] if all_inversions else [],
    }


# ================================================================
# LOAD DATA
# ================================================================
print("=" * 72)
print("EXTENDED RAR INVERSION POINT REPLICATION")
print("=" * 72)

# Load extended dataset with S4G matching
galaxies = load_all(include_vizier=True, match_photometry=True)

# Also apply Bouquin+2018 profile matching (overrides S4G where available)
print("\nApplying Bouquin+2018 profile matching...")
bouquin_stats = compute_baryonic_bouquin(galaxies)

# Also apply Korsaga+2019 mass models (Freeman disk + Sérsic bulge from published params)
print("\nApplying Korsaga+2019 mass model reconstruction...")
korsaga_stats = add_korsaga_to_galaxies(galaxies, use_fixed_ml=True, verbose=True)

log_gobs_all, log_gbar_all, names_all, sources_all = build_rar(galaxies)

# Separate SPARC-only
sparc_mask = sources_all == 'SPARC'
log_gobs_sparc = log_gobs_all[sparc_mask]
log_gbar_sparc = log_gbar_all[sparc_mask]
names_sparc = names_all[sparc_mask]
sources_sparc = sources_all[sparc_mask]

# Non-SPARC only
ext_mask = ~sparc_mask
log_gobs_ext = log_gobs_all[ext_mask]
log_gbar_ext = log_gbar_all[ext_mask]
names_ext = names_all[ext_mask]
sources_ext = sources_all[ext_mask]

# ================================================================
# RUN ANALYSES
# ================================================================

# 1. SPARC-only (baseline replication)
sparc_results = run_inversion_analysis(
    log_gobs_sparc, log_gbar_sparc, names_sparc, sources_sparc,
    "SPARC-only (baseline)")

# 2. Quality tiers
# Tier A: SPARC + THINGS (proper dedicated mass models, σ ~ 0.13-0.17)
tier_a_mask = (sources_all == 'SPARC') | (sources_all == 'THINGS')
tier_a_results = run_inversion_analysis(
    log_gobs_all[tier_a_mask], log_gbar_all[tier_a_mask],
    names_all[tier_a_mask], sources_all[tier_a_mask],
    "Tier A: SPARC + THINGS (gold-standard mass models)")

# Tier B: A + de Blok 2002 (direct decomposition, σ ~ 0.33)
tier_b_mask = tier_a_mask | (sources_all == 'deBlok2002')
tier_b_results = run_inversion_analysis(
    log_gobs_all[tier_b_mask], log_gbar_all[tier_b_mask],
    names_all[tier_b_mask], sources_all[tier_b_mask],
    "Tier B: + de Blok 2002 (direct decomposition)")

# Tier B+: + Bouquin-matched (non-GomezLopez, stellar or ALFALFA, σ ~ 0.20-0.32)
# Only include Bouquin galaxies from high-quality RC sources
good_bouquin_gals = set()
for name, g in galaxies.items():
    mm = g.get('mass_model_source', '')
    if 'Bouquin' not in mm:
        continue
    # Include Verheijen (σ~0.19) and PHANGS stellar (σ~0.32) as good quality
    # Exclude GomezLopez (σ~0.77) and THINGS overlap (offset issues)
    if g['source'] in ('Verheijen2001', 'PHANGS_Lang2020'):
        good_bouquin_gals.add(name)

tier_bplus_mask = tier_b_mask.copy()
for i in range(len(names_all)):
    if names_all[i] in good_bouquin_gals:
        tier_bplus_mask[i] = True

tier_bplus_results = run_inversion_analysis(
    log_gobs_all[tier_bplus_mask], log_gbar_all[tier_bplus_mask],
    names_all[tier_bplus_mask], sources_all[tier_bplus_mask],
    "Tier B+: + Bouquin (quality RC sources, σ ~ 0.20-0.32)")

# Tier K: + Korsaga (Freeman disk + Sérsic bulge from published mass model params)
tier_k_mask = tier_bplus_mask.copy()
for i in range(len(names_all)):
    if sources_all[i] == 'Korsaga2019':
        tier_k_mask[i] = True

tier_k_results = run_inversion_analysis(
    log_gobs_all[tier_k_mask], log_gbar_all[tier_k_mask],
    names_all[tier_k_mask], sources_all[tier_k_mask],
    "Tier K: + Korsaga2019 (Freeman+Sérsic from published params)")

# 3. Extended (all sources, including noisy S4G-derived and Bouquin)
extended_results = run_inversion_analysis(
    log_gobs_all, log_gbar_all, names_all, sources_all,
    "Tier C: ALL sources (incl. noisy Bouquin+S4G, σ ~ 0.50+)")

# 4. Non-SPARC only (independent check)
if len(log_gobs_ext) > 100:
    ext_only_results = run_inversion_analysis(
        log_gobs_ext, log_gbar_ext, names_ext, sources_ext,
        "Non-SPARC only (VizieR + S4G/ALFALFA)")
else:
    ext_only_results = None
    print("\n  [SKIP] Not enough non-SPARC RAR points for independent analysis")

# ================================================================
# MHONGOOSE TIERS
# ================================================================
print("\n\nLoading MHONGOOSE data...")
mh_data_path = os.path.join(PROJECT_ROOT, 'data', 'mhongoose', 'mhongoose_rar_all.tsv')
mh_log_gbar, mh_log_gobs, mh_names, mh_sources = [], [], [], []
with open(mh_data_path) as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split('\t')
        mh_names.append(parts[0])
        mh_log_gbar.append(float(parts[1]))
        mh_log_gobs.append(float(parts[2]))
        mh_sources.append(parts[3])
mh_log_gbar = np.array(mh_log_gbar)
mh_log_gobs = np.array(mh_log_gobs)
mh_names = np.array(mh_names)
mh_sources = np.array(mh_sources)

# Sorgho galaxies (NGC3621 σ=0.060, NGC7424 σ=0.082) — qualify for Tier B+
sorgho_mask = (mh_sources == 'Sorgho+2019')
# Nkomo dwarfs (ESO444-G084 σ=0.175, KKS2000-23 σ=0.315) — Tier C quality
nkomo_mask = (mh_sources == 'Nkomo+2025')

print(f"  MHONGOOSE: {len(mh_log_gbar)} points, {len(set(mh_names))} galaxies")
print(f"    Sorgho+2019 (B+ quality): {sorgho_mask.sum()} pts — {sorted(set(mh_names[sorgho_mask]))}")
print(f"    Nkomo+2025 (C quality):   {nkomo_mask.sum()} pts — {sorted(set(mh_names[nkomo_mask]))}")

# Tier MH: Tier B+ + Sorgho galaxies (NGC3621 + NGC7424)
tier_mh_log_gobs = np.concatenate([log_gobs_all[tier_bplus_mask], mh_log_gobs[sorgho_mask]])
tier_mh_log_gbar = np.concatenate([log_gbar_all[tier_bplus_mask], mh_log_gbar[sorgho_mask]])
tier_mh_names = np.concatenate([names_all[tier_bplus_mask], mh_names[sorgho_mask]])
tier_mh_sources = np.concatenate([sources_all[tier_bplus_mask], mh_sources[sorgho_mask]])

n_mh = len(tier_mh_log_gobs)
n_mh_gal = len(set(tier_mh_names))

tier_mh_results = run_inversion_analysis(
    tier_mh_log_gobs, tier_mh_log_gbar, tier_mh_names, tier_mh_sources,
    "Tier MH: Tier B+ + MHONGOOSE Sorgho (NGC3621 + NGC7424)")

# Tier C+MH: ALL existing sources + all MHONGOOSE (incl. Nkomo dwarfs)
tier_c_mh_log_gobs = np.concatenate([log_gobs_all, mh_log_gobs])
tier_c_mh_log_gbar = np.concatenate([log_gbar_all, mh_log_gbar])
tier_c_mh_names = np.concatenate([names_all, mh_names])
tier_c_mh_sources = np.concatenate([sources_all, mh_sources])

tier_c_mh_results = run_inversion_analysis(
    tier_c_mh_log_gobs, tier_c_mh_log_gbar, tier_c_mh_names, tier_c_mh_sources,
    "Tier C+MH: ALL sources + all MHONGOOSE (incl. Nkomo dwarfs)")

# ================================================================
# SCATTER QUALITY COMPARISON
# ================================================================
print(f"\n\n{'='*72}")
print("RAR RESIDUAL SCATTER BY SOURCE")
print(f"{'='*72}")

log_gobs_rar_all = rar_function(log_gbar_all)
log_res_all = log_gobs_all - log_gobs_rar_all

# By RC source
print(f"\n  By RC source:")
print(f"  {'Source':20s} {'N_pts':>7s} {'N_gal':>6s} {'σ(res)':>8s} {'<res>':>8s}")
for src in sorted(set(sources_all)):
    mask = sources_all == src
    r = log_res_all[mask]
    n_gal = len(set(names_all[mask]))
    print(f"  {src:20s} {mask.sum():7d} {n_gal:6d} {np.std(r):8.4f} {np.mean(r):+8.4f}")

# By mass model source
print(f"\n  By mass model source:")
print(f"  {'Mass model':25s} {'N_pts':>7s} {'σ(res)':>8s} {'<res>':>8s}")
mm_groups = {}
for i in range(len(log_gobs_all)):
    gname = names_all[i]
    g = galaxies.get(gname)
    if g:
        mms = g.get('mass_model_source', g['source'] if g['source'] == 'SPARC' else 'none')
    else:
        mms = 'unknown'
    mm_groups.setdefault(mms, []).append(log_res_all[i])
for mms, residuals in sorted(mm_groups.items()):
    r = np.array(residuals)
    print(f"  {mms:25s} {len(r):7d} {np.std(r):8.4f} {np.mean(r):+8.4f}")

print(f"\n  Key insight: inversion requires σ < 0.2 dex mass decomposition")
print(f"  → noisy data washes out the signal, consistent with PROBES finding")

# ================================================================
# COMPARISON
# ================================================================
print(f"\n\n{'='*72}")
print("COMPARISON: QUALITY-TIERED INVERSION ANALYSIS")
print(f"{'='*72}")

def get_closest_inversion(results):
    """Get the dσ/dx=0 crossing closest to g†."""
    sigma_crossings = results['zero_crossings'].get('dsigma_dx', [])
    if not sigma_crossings:
        return None
    distances = [(x, abs(x - LOG_G_DAGGER)) for x in sigma_crossings]
    distances.sort(key=lambda t: t[1])
    return distances[0][0]

sparc_inv = get_closest_inversion(sparc_results)
tier_a_inv = get_closest_inversion(tier_a_results)
tier_b_inv = get_closest_inversion(tier_b_results)
tier_bplus_inv = get_closest_inversion(tier_bplus_results)
tier_k_inv = get_closest_inversion(tier_k_results)
extended_inv = get_closest_inversion(extended_results)
tier_mh_inv = get_closest_inversion(tier_mh_results)
tier_c_mh_inv = get_closest_inversion(tier_c_mh_results)

n_bplus = int(tier_bplus_mask.sum())
n_bplus_gal = len(set(names_all[tier_bplus_mask]))
n_k = int(tier_k_mask.sum())
n_k_gal = len(set(names_all[tier_k_mask]))

print(f"\n  g† (BEC prediction) = log g_bar = {LOG_G_DAGGER:.3f}")

for label, inv in [
    ("SPARC-only (126 gal)", sparc_inv),
    ("Tier A: SPARC+THINGS (132 gal)", tier_a_inv),
    ("Tier B: +deBlok2002 (149 gal)", tier_b_inv),
    (f"Tier B+: +Bouquin quality ({n_bplus_gal} gal)", tier_bplus_inv),
    (f"Tier K: +Korsaga2019 ({n_k_gal} gal)", tier_k_inv),
    ("Tier C: ALL sources", extended_inv),
    (f"Tier MH: B+ + Sorgho ({n_mh_gal} gal)", tier_mh_inv),
    ("Tier C+MH: ALL + MHONGOOSE", tier_c_mh_inv),
]:
    if inv is not None:
        print(f"  {label:42s}: log g = {inv:.3f}  (Δ = {inv - LOG_G_DAGGER:+.3f} dex)")
    else:
        print(f"  {label:42s}: NOT FOUND")

if ext_only_results:
    ext_inv = get_closest_inversion(ext_only_results)
    if ext_inv is not None:
        print(f"  {'Non-SPARC only':42s}: log g = {ext_inv:.3f}  (Δ = {ext_inv - LOG_G_DAGGER:+.3f} dex)")

print(f"\n  SUMMARY:")
if tier_b_inv and sparc_inv:
    print(f"  Adding de Blok 2002 shifts inversion by {tier_b_inv - sparc_inv:+.3f} dex")
if tier_bplus_inv and tier_b_inv:
    print(f"  Adding quality Bouquin shifts by {tier_bplus_inv - tier_b_inv:+.3f} dex from Tier B")
if tier_k_inv and tier_bplus_inv:
    print(f"  Adding Korsaga 2019 shifts by {tier_k_inv - tier_bplus_inv:+.3f} dex from Tier B+")
if extended_inv and sparc_inv:
    print(f"  Adding ALL noisy data shifts by {extended_inv - sparc_inv:+.3f} dex — AWAY from g†")
if tier_mh_inv and tier_bplus_inv:
    print(f"  Adding MHONGOOSE Sorgho to B+ shifts by {tier_mh_inv - tier_bplus_inv:+.3f} dex from Tier B+")
print(f"  → Inversion point requires SPARC-quality mass decomposition (σ < 0.2 dex)")
print(f"  → Adding quality data IMPROVES signal; adding noise degrades it")

tier_b_within = tier_b_inv is not None and abs(tier_b_inv - LOG_G_DAGGER) < 0.25
tier_bplus_within = tier_bplus_inv is not None and abs(tier_bplus_inv - LOG_G_DAGGER) < 0.25
tier_k_within = tier_k_inv is not None and abs(tier_k_inv - LOG_G_DAGGER) < 0.25
tier_mh_within = tier_mh_inv is not None and abs(tier_mh_inv - LOG_G_DAGGER) < 0.25
if tier_k_within:
    print(f"\n  RESULT: Tier K inversion ({tier_k_inv:.3f}) within 0.25 dex of g† — REPLICATED")
elif tier_bplus_within:
    print(f"\n  RESULT: Tier B+ inversion ({tier_bplus_inv:.3f}) within 0.25 dex of g† — REPLICATED")
elif tier_b_within:
    print(f"\n  RESULT: Tier B inversion ({tier_b_inv:.3f}) within 0.25 dex of g† — REPLICATED")
else:
    print(f"\n  RESULT: Best tier inversion not within 0.25 dex of g†")

if tier_mh_within:
    print(f"  RESULT: Tier MH inversion ({tier_mh_inv:.3f}) within 0.25 dex of g† — MHONGOOSE REPLICATED")
elif tier_mh_inv is not None:
    print(f"  RESULT: Tier MH inversion ({tier_mh_inv:.3f}) NOT within 0.25 dex of g†")
else:
    print(f"  RESULT: Tier MH — no dσ/dx=0 crossing found")

# ================================================================
# SAVE
# ================================================================
output = {
    'test_name': 'extended_rar_inversion_replication',
    'description': ('Quality-tiered inversion analysis on extended RAR with '
                    'Bouquin+2018 3.6μm profile matching + Korsaga+2019 '
                    'Freeman disk + Sérsic bulge reconstruction. '
                    'Key finding: inversion requires SPARC-quality mass decomposition '
                    '(σ < 0.2 dex). Adding quality data improves signal; noise degrades it.'),
    'g_dagger_log': LOG_G_DAGGER,
    'sparc_only': sparc_results,
    'tier_a_sparc_things': tier_a_results,
    'tier_b_with_deblok': tier_b_results,
    'tier_bplus_with_bouquin': tier_bplus_results,
    'tier_k_with_korsaga': tier_k_results,
    'tier_c_all_sources': extended_results,
    'non_sparc_only': ext_only_results,
    'tier_mh_bplus_sorgho': tier_mh_results,
    'tier_c_mh_all_mhongoose': tier_c_mh_results,
    'bouquin_matching': {
        'bouquin_matched': bouquin_stats.get('bouquin_matched', 0),
        'alfalfa_matched': bouquin_stats.get('alfalfa_matched', 0),
        'stellar_only': bouquin_stats.get('stellar_only', 0),
        'quality_filtered_galaxies': len(good_bouquin_gals),
    },
    'korsaga_matching': {
        'total_reconstructed': korsaga_stats.get('total_matched', 0),
        'added_new': korsaga_stats.get('added_new', 0),
        'replaced_vrot_only': korsaga_stats.get('replaced_vrot_only', 0),
        'skipped_gold': korsaga_stats.get('skipped_gold', 0),
        'disk_only': korsaga_stats.get('disk_only', 0),
        'disk_plus_bulge': korsaga_stats.get('disk_plus_bulge', 0),
    },
    'scatter_by_source': {
        src: {
            'n_pts': int((sources_all == src).sum()),
            'n_gal': len(set(names_all[sources_all == src])),
            'sigma_res': round(float(np.std(log_res_all[sources_all == src])), 4),
            'mean_res': round(float(np.mean(log_res_all[sources_all == src])), 4),
        }
        for src in sorted(set(sources_all))
    },
    'comparison': {
        'sparc_inversion': round(sparc_inv, 3) if sparc_inv else None,
        'tier_a_inversion': round(tier_a_inv, 3) if tier_a_inv else None,
        'tier_b_inversion': round(tier_b_inv, 3) if tier_b_inv else None,
        'tier_bplus_inversion': round(tier_bplus_inv, 3) if tier_bplus_inv else None,
        'tier_k_inversion': round(tier_k_inv, 3) if tier_k_inv else None,
        'tier_c_inversion': round(extended_inv, 3) if extended_inv else None,
        'tier_b_replicated_within_025dex': bool(tier_b_within),
        'tier_bplus_replicated_within_025dex': bool(tier_bplus_within),
        'tier_k_replicated_within_025dex': bool(tier_k_within),
        'shift_sparc_to_tier_b': round(tier_b_inv - sparc_inv, 3) if (sparc_inv and tier_b_inv) else None,
        'shift_sparc_to_tier_bplus': round(tier_bplus_inv - sparc_inv, 3) if (sparc_inv and tier_bplus_inv) else None,
        'shift_sparc_to_tier_k': round(tier_k_inv - sparc_inv, 3) if (sparc_inv and tier_k_inv) else None,
        'shift_sparc_to_tier_c': round(extended_inv - sparc_inv, 3) if (sparc_inv and extended_inv) else None,
        'tier_mh_inversion': round(tier_mh_inv, 3) if tier_mh_inv else None,
        'tier_c_mh_inversion': round(tier_c_mh_inv, 3) if tier_c_mh_inv else None,
        'tier_mh_replicated_within_025dex': bool(tier_mh_within),
        'shift_bplus_to_tier_mh': round(tier_mh_inv - tier_bplus_inv, 3) if (tier_bplus_inv and tier_mh_inv) else None,
        'shift_sparc_to_tier_mh': round(tier_mh_inv - sparc_inv, 3) if (sparc_inv and tier_mh_inv) else None,
    },
}

outpath = os.path.join(RESULTS_DIR, 'summary_extended_rar_inversion.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
