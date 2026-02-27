#!/usr/bin/env python3
"""
EAGLE Simulation RAR & Scatter Derivative Inversion Test — v2
================================================================

Corrected version with:
  A) Restriction to SPARC acceleration range only
  B) Galaxy-level scatter derivative with bootstrap CIs (no pseudoreplication)
  C) Mock-observed rotation curves bridge test (apples-to-apples with SPARC)

Uses cached aperture masses from EAGLE Ref-L100N1504 (z=0),
29,716 star-forming centrals at 10 aperture radii.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import hashlib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'eagle_rar')
SPARC_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics constants
G = 6.674e-11        # m^3 kg^-1 s^-2
M_sun = 1.989e30     # kg
kpc_m = 3.086e19     # m
g_dagger = 1.20e-10  # m/s^2
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921

np.random.seed(42)


def stable_seed(value):
    """Deterministic seed independent of Python hash randomization."""
    hx = hashlib.sha256(str(value).encode('utf-8')).hexdigest()
    return int(hx[:8], 16)

print("=" * 72)
print("EAGLE SIMULATION: RAR SCATTER DERIVATIVE INVERSION — v2")
print("  A) SPARC-range restriction  B) Galaxy-level stats  C) Mock RC bridge")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")


# ================================================================
# 0. LOAD CACHED EAGLE DATA
# ================================================================
print("\n[0] Loading cached EAGLE aperture masses...")

cache_path = os.path.join(DATA_DIR, 'eagle_aperture_masses.json')
if not os.path.exists(cache_path):
    print("  ERROR: No cached data. Run test_eagle_rar_inversion.py first.")
    sys.exit(1)

with open(cache_path, 'r') as f:
    cached = json.load(f)

aperture_sizes = cached['aperture_sizes']  # [1, 3, 5, 10, 20, 30, 40, 50, 70, 100] kpc
galaxy_data = cached['galaxy_data']
print(f"  {len(galaxy_data)} galaxies, apertures: {aperture_sizes} kpc")

# UNIT NOTE: eagleSqlTools returns EAGLE masses in M_sun already.
# The v1 script incorrectly multiplied by 1e10, shifting accelerations ~10 dex too high.
# Verification: median M_star(30kpc) ~ 10^8.2 M_sun, max ~ 10^11.7 — physically correct.
MASS_UNIT = 1.0  # masses are already in M_sun


# ================================================================
# 1. LOAD SPARC DATA (REFERENCE)
# ================================================================
print("\n[1] Loading SPARC observational data...")

table2_path = os.path.join(SPARC_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(SPARC_DIR, 'SPARC_Lelli2016c.mrt')

sparc_galaxies = {}
with open(table2_path, 'r') as f:
    for line in f:
        if len(line.strip()) < 50:
            continue
        try:
            name = line[0:11].strip()
            if not name:
                continue
            rad = float(line[19:25].strip())
            vobs = float(line[26:32].strip())
            vgas = float(line[39:45].strip())
            vdisk = float(line[46:52].strip())
            vbul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue
        if name not in sparc_galaxies:
            sparc_galaxies[name] = {'R': [], 'Vobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
        sparc_galaxies[name]['R'].append(rad)
        sparc_galaxies[name]['Vobs'].append(vobs)
        sparc_galaxies[name]['Vgas'].append(vgas)
        sparc_galaxies[name]['Vdisk'].append(vdisk)
        sparc_galaxies[name]['Vbul'].append(vbul)

for name in sparc_galaxies:
    for key in sparc_galaxies[name]:
        sparc_galaxies[name][key] = np.array(sparc_galaxies[name][key])

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
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        sparc_props[name] = {'Inc': float(parts[4]), 'Q': int(parts[16])}
    except (ValueError, IndexError):
        continue

# Build SPARC RAR (per-galaxy)
sparc_per_galaxy = {}
sparc_log_gbar_all = []
sparc_log_gobs_all = []

for name, gdata in sparc_galaxies.items():
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

    lg = np.log10(gbar_SI[valid])
    lo = np.log10(gobs_SI[valid])
    sparc_per_galaxy[name] = {'log_gbar': lg, 'log_gobs': lo}
    sparc_log_gbar_all.extend(lg)
    sparc_log_gobs_all.extend(lo)

sparc_log_gbar_all = np.array(sparc_log_gbar_all)
sparc_log_gobs_all = np.array(sparc_log_gobs_all)

SPARC_GBAR_MIN = np.percentile(sparc_log_gbar_all, 1)   # ~-12.5
SPARC_GBAR_MAX = np.percentile(sparc_log_gbar_all, 99)   # ~-8.5

print(f"  SPARC galaxies (quality): {len(sparc_per_galaxy)}")
print(f"  SPARC RAR points: {len(sparc_log_gbar_all)}")
print(f"  SPARC log g_bar range (1-99%): [{SPARC_GBAR_MIN:.2f}, {SPARC_GBAR_MAX:.2f}]")
print(f"  SPARC log g_obs range: [{sparc_log_gobs_all.min():.2f}, {sparc_log_gobs_all.max():.2f}]")


# ================================================================
# 2. COMPUTE EAGLE RAR — CORRECTED UNITS + SPARC RANGE
# ================================================================
print("\n[2] Computing EAGLE RAR with corrected units...")

# Per-galaxy EAGLE data
eagle_per_galaxy = {}
eagle_log_gbar_all = []
eagle_log_gobs_all = []
eagle_log_gbar_full = []  # Before SPARC range cut
eagle_log_gobs_full = []

for gid, gdata in galaxy_data.items():
    aps = gdata['apertures']
    gal_lgbar = []
    gal_lgobs = []

    for ap_str, masses in sorted(aps.items(), key=lambda x: float(x[0])):
        r_kpc = float(ap_str)
        if r_kpc < 1.0:
            continue

        m_star = masses['m_star'] * MASS_UNIT  # M_sun
        m_gas = masses['m_gas'] * MASS_UNIT
        m_dm = masses['m_dm'] * MASS_UNIT

        m_bar = m_star + m_gas
        m_total = m_bar + m_dm

        if m_bar <= 0 or m_total <= 0:
            continue

        r_m = r_kpc * kpc_m
        g_bar = G * m_bar * M_sun / r_m**2
        g_obs = G * m_total * M_sun / r_m**2

        if g_bar > 1e-16 and g_obs > 1e-16:
            lg = np.log10(g_bar)
            lo = np.log10(g_obs)
            eagle_log_gbar_full.append(lg)
            eagle_log_gobs_full.append(lo)
            gal_lgbar.append(lg)
            gal_lgobs.append(lo)

    if len(gal_lgbar) >= 3:
        gal_lgbar = np.array(gal_lgbar)
        gal_lgobs = np.array(gal_lgobs)

        # Apply SPARC range filter
        in_range = (gal_lgbar >= SPARC_GBAR_MIN) & (gal_lgbar <= SPARC_GBAR_MAX)
        if np.sum(in_range) >= 3:
            eagle_per_galaxy[gid] = {
                'log_gbar': gal_lgbar[in_range],
                'log_gobs': gal_lgobs[in_range],
            }
            eagle_log_gbar_all.extend(gal_lgbar[in_range])
            eagle_log_gobs_all.extend(gal_lgobs[in_range])

eagle_log_gbar_all = np.array(eagle_log_gbar_all)
eagle_log_gobs_all = np.array(eagle_log_gobs_all)
eagle_log_gbar_full = np.array(eagle_log_gbar_full)
eagle_log_gobs_full = np.array(eagle_log_gobs_full)

print(f"  EAGLE full range: {len(eagle_log_gbar_full)} points")
print(f"    log g_bar: [{eagle_log_gbar_full.min():.2f}, {eagle_log_gbar_full.max():.2f}]")
print(f"  EAGLE in SPARC range ({SPARC_GBAR_MIN:.1f} to {SPARC_GBAR_MAX:.1f}):")
print(f"    {len(eagle_log_gbar_all)} points from {len(eagle_per_galaxy)} galaxies")
print(f"    log g_bar: [{eagle_log_gbar_all.min():.2f}, {eagle_log_gbar_all.max():.2f}]")


# ================================================================
# HELPER: Scatter derivative analysis
# ================================================================
def rar_residual(log_gbar, log_gobs):
    """Compute residual from McGaugh+2016 RAR fit."""
    gbar = 10**log_gbar
    with np.errstate(over='ignore', invalid='ignore'):
        rar_pred = np.log10(gbar / (1.0 - np.exp(-np.sqrt(gbar / g_dagger))))
    resid = log_gobs - rar_pred
    valid = np.isfinite(resid)
    return resid[valid], log_gbar[valid]


def scatter_profile(log_gbar, log_gobs, bin_width=0.30, offset=0.0, min_pts=10):
    """Compute binned scatter profile."""
    resid, lg = rar_residual(log_gbar, log_gobs)
    if len(resid) < 20:
        return None, None, None

    lo = lg.min() + offset
    hi = lg.max()
    edges = np.arange(lo, hi + bin_width, bin_width)

    centers = []
    sigmas = []
    counts = []
    for j in range(len(edges) - 1):
        mask = (lg >= edges[j]) & (lg < edges[j+1])
        n = np.sum(mask)
        if n >= min_pts:
            centers.append(0.5 * (edges[j] + edges[j+1]))
            sigmas.append(np.std(resid[mask]))
            counts.append(n)

    if len(centers) < 4:
        return None, None, None

    return np.array(centers), np.array(sigmas), np.array(counts)


def find_inversion(centers, sigmas):
    """Find scatter derivative zero-crossing nearest to g†."""
    if centers is None or len(centers) < 4:
        return None

    dsigma = np.diff(sigmas)
    dcenter = 0.5 * (centers[:-1] + centers[1:])

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
    return float(crossings[nearest_idx])


# ================================================================
# PART A: POOLED SCATTER DERIVATIVE — SPARC RANGE ONLY
# ================================================================
print("\n" + "=" * 72)
print("PART A: POOLED SCATTER DERIVATIVE (SPARC acceleration range only)")
print("=" * 72)

# SPARC
c_sparc, s_sparc, n_sparc = scatter_profile(sparc_log_gbar_all, sparc_log_gobs_all)
inv_sparc = find_inversion(c_sparc, s_sparc)

print(f"\n  SPARC scatter profile:")
if c_sparc is not None:
    for j in range(len(c_sparc)):
        marker = " <-- g†" if abs(c_sparc[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {c_sparc[j]:+.2f}: σ = {s_sparc[j]:.4f} dex (N={n_sparc[j]}){marker}")
    if inv_sparc is not None:
        print(f"  => Inversion at log g = {inv_sparc:.3f}, Δ from g† = {abs(inv_sparc - LOG_G_DAGGER):.3f} dex")
    else:
        print(f"  => No inversion found")

# EAGLE (SPARC range)
c_eagle, s_eagle, n_eagle = scatter_profile(eagle_log_gbar_all, eagle_log_gobs_all)
inv_eagle = find_inversion(c_eagle, s_eagle)

print(f"\n  EAGLE scatter profile (SPARC range only):")
if c_eagle is not None:
    for j in range(len(c_eagle)):
        marker = " <-- g†" if abs(c_eagle[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {c_eagle[j]:+.2f}: σ = {s_eagle[j]:.4f} dex (N={n_eagle[j]}){marker}")
    if inv_eagle is not None:
        print(f"  => Inversion at log g = {inv_eagle:.3f}, Δ from g† = {abs(inv_eagle - LOG_G_DAGGER):.3f} dex")
    else:
        print(f"  => No inversion found")
else:
    print(f"  => Insufficient data for scatter profile")

# Robustness: multiple bin offsets
print(f"\n  Robustness (10 bin offsets):")
inv_offsets_sparc = []
inv_offsets_eagle = []
for off in np.linspace(0, 0.25, 10):
    cs, ss, _ = scatter_profile(sparc_log_gbar_all, sparc_log_gobs_all, offset=off)
    inv_s = find_inversion(cs, ss)
    if inv_s is not None:
        inv_offsets_sparc.append(inv_s)

    ce, se, _ = scatter_profile(eagle_log_gbar_all, eagle_log_gobs_all, offset=off)
    inv_e = find_inversion(ce, se)
    if inv_e is not None:
        inv_offsets_eagle.append(inv_e)

if inv_offsets_sparc:
    arr = np.array(inv_offsets_sparc)
    print(f"    SPARC: {len(arr)}/10 offsets, mean = {arr.mean():.3f} ± {arr.std():.3f}")
if inv_offsets_eagle:
    arr = np.array(inv_offsets_eagle)
    near = np.sum(np.abs(arr - LOG_G_DAGGER) < 0.30)
    print(f"    EAGLE: {len(arr)}/10 offsets, mean = {arr.mean():.3f} ± {arr.std():.3f}, near g†: {near}/10")
else:
    print(f"    EAGLE: 0/10 offsets found inversions")

# Multiple bin widths
print(f"\n  Robustness (bin widths 0.15-0.50):")
for bw in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    ce, se, ne = scatter_profile(eagle_log_gbar_all, eagle_log_gobs_all, bin_width=bw)
    inv_e = find_inversion(ce, se)
    cs, ss, ns = scatter_profile(sparc_log_gbar_all, sparc_log_gobs_all, bin_width=bw)
    inv_s = find_inversion(cs, ss)
    s_str = f"log g = {inv_s:.3f}" if inv_s is not None else "none"
    e_str = f"log g = {inv_e:.3f}" if inv_e is not None else "none"
    print(f"    bw={bw:.2f}: SPARC={s_str}, EAGLE={e_str}")


# ================================================================
# PART B: GALAXY-LEVEL SCATTER DERIVATIVE + BOOTSTRAP
# ================================================================
print("\n" + "=" * 72)
print("PART B: GALAXY-LEVEL SCATTER DERIVATIVE (no pseudoreplication)")
print("=" * 72)

# Use common bin edges for all galaxies
BIN_WIDTH = 0.30
common_lo = max(SPARC_GBAR_MIN, -12.5)
common_hi = min(SPARC_GBAR_MAX, -8.5)
common_edges = np.arange(common_lo, common_hi + BIN_WIDTH, BIN_WIDTH)
common_centers = 0.5 * (common_edges[:-1] + common_edges[1:])
n_bins = len(common_centers)
print(f"  Common bins: {n_bins} bins from {common_lo:.1f} to {common_hi:.1f}, width={BIN_WIDTH}")


def galaxy_scatter_in_bins(per_galaxy_dict, common_edges, min_pts_per_bin=2):
    """Compute per-galaxy scatter in common bins, return matrix [n_galaxies x n_bins]."""
    n_bins = len(common_edges) - 1
    all_scatter = []
    gal_ids = []

    for gid, gdata in per_galaxy_dict.items():
        log_gbar = gdata['log_gbar']
        log_gobs = gdata['log_gobs']

        gbar = 10**log_gbar
        with np.errstate(over='ignore', invalid='ignore'):
            rar_pred = np.log10(gbar / (1.0 - np.exp(-np.sqrt(gbar / g_dagger))))
        resid = log_gobs - rar_pred
        valid = np.isfinite(resid)
        log_gbar = log_gbar[valid]
        resid = resid[valid]

        row = np.full(n_bins, np.nan)
        for j in range(n_bins):
            mask = (log_gbar >= common_edges[j]) & (log_gbar < common_edges[j+1])
            if np.sum(mask) >= min_pts_per_bin:
                row[j] = np.std(resid[mask]) if np.sum(mask) > 1 else np.abs(resid[mask][0])

        if np.sum(np.isfinite(row)) >= 3:
            all_scatter.append(row)
            gal_ids.append(gid)

    return np.array(all_scatter), gal_ids


# SPARC: galaxy-level scatter
sparc_scatter, sparc_gids = galaxy_scatter_in_bins(sparc_per_galaxy, common_edges, min_pts_per_bin=2)
print(f"\n  SPARC: {len(sparc_gids)} galaxies with ≥3 populated bins")

# EAGLE: galaxy-level scatter
eagle_scatter, eagle_gids = galaxy_scatter_in_bins(eagle_per_galaxy, common_edges, min_pts_per_bin=1)
print(f"  EAGLE: {len(eagle_gids)} galaxies with ≥3 populated bins")


def bootstrap_median_ci(data_matrix, n_boot=2000, ci=95):
    """Bootstrap median and CI across galaxies (axis=0)."""
    n_gals, n_bins = data_matrix.shape
    medians = np.nanmedian(data_matrix, axis=0)

    boot_medians = np.zeros((n_boot, n_bins))
    for b in range(n_boot):
        idx = np.random.choice(n_gals, size=n_gals, replace=True)
        boot_medians[b] = np.nanmedian(data_matrix[idx], axis=0)

    lo_pct = (100 - ci) / 2
    hi_pct = 100 - lo_pct
    ci_lo = np.nanpercentile(boot_medians, lo_pct, axis=0)
    ci_hi = np.nanpercentile(boot_medians, hi_pct, axis=0)

    return medians, ci_lo, ci_hi


# Bootstrap for both
if len(sparc_scatter) >= 10:
    sparc_med, sparc_ci_lo, sparc_ci_hi = bootstrap_median_ci(sparc_scatter)
    print(f"\n  SPARC galaxy-level median scatter (95% bootstrap CI):")
    for j in range(n_bins):
        n_gals = np.sum(np.isfinite(sparc_scatter[:, j]))
        marker = " <-- g†" if abs(common_centers[j] - LOG_G_DAGGER) < 0.20 else ""
        if np.isfinite(sparc_med[j]):
            print(f"    log g = {common_centers[j]:+.2f}: σ = {sparc_med[j]:.4f} [{sparc_ci_lo[j]:.4f}, {sparc_ci_hi[j]:.4f}] (N_gal={n_gals}){marker}")
        else:
            print(f"    log g = {common_centers[j]:+.2f}: insufficient data{marker}")

if len(eagle_scatter) >= 10:
    eagle_med, eagle_ci_lo, eagle_ci_hi = bootstrap_median_ci(eagle_scatter)
    print(f"\n  EAGLE galaxy-level median scatter (95% bootstrap CI):")
    for j in range(n_bins):
        n_gals = np.sum(np.isfinite(eagle_scatter[:, j]))
        marker = " <-- g†" if abs(common_centers[j] - LOG_G_DAGGER) < 0.20 else ""
        if np.isfinite(eagle_med[j]):
            print(f"    log g = {common_centers[j]:+.2f}: σ = {eagle_med[j]:.4f} [{eagle_ci_lo[j]:.4f}, {eagle_ci_hi[j]:.4f}] (N_gal={n_gals}){marker}")
        else:
            print(f"    log g = {common_centers[j]:+.2f}: insufficient data{marker}")

    # Scatter derivative from galaxy-level medians
    print(f"\n  Galaxy-level scatter derivatives:")
    if len(sparc_scatter) >= 10:
        sparc_dsig = np.diff(sparc_med)
        sparc_dcenter = 0.5 * (common_centers[:-1] + common_centers[1:])
        inv_sparc_gal = find_inversion(common_centers, sparc_med)
        print(f"\n    SPARC:")
        for j in range(len(sparc_dsig)):
            if np.isfinite(sparc_dsig[j]):
                marker = " <-- g†" if abs(sparc_dcenter[j] - LOG_G_DAGGER) < 0.20 else ""
                print(f"      log g = {sparc_dcenter[j]:+.2f}: Δσ = {sparc_dsig[j]:+.4f}{marker}")
        if inv_sparc_gal is not None:
            print(f"    => Galaxy-level inversion at log g = {inv_sparc_gal:.3f}, Δ from g† = {abs(inv_sparc_gal - LOG_G_DAGGER):.3f}")
        else:
            print(f"    => No galaxy-level inversion")

    eagle_dsig = np.diff(eagle_med)
    eagle_dcenter = 0.5 * (common_centers[:-1] + common_centers[1:])
    inv_eagle_gal = find_inversion(common_centers, eagle_med)
    print(f"\n    EAGLE:")
    for j in range(len(eagle_dsig)):
        if np.isfinite(eagle_dsig[j]):
            marker = " <-- g†" if abs(eagle_dcenter[j] - LOG_G_DAGGER) < 0.20 else ""
            print(f"      log g = {eagle_dcenter[j]:+.2f}: Δσ = {eagle_dsig[j]:+.4f}{marker}")
    if inv_eagle_gal is not None:
        print(f"    => Galaxy-level inversion at log g = {inv_eagle_gal:.3f}, Δ from g† = {abs(inv_eagle_gal - LOG_G_DAGGER):.3f}")
    else:
        print(f"    => No galaxy-level inversion")

    # Bootstrap the inversions themselves
    print(f"\n  Bootstrap inversion location (2000 resamples):")
    n_boot = 2000
    sparc_boot_inv = []
    eagle_boot_inv = []
    for b in range(n_boot):
        if len(sparc_scatter) >= 10:
            idx_s = np.random.choice(len(sparc_scatter), size=len(sparc_scatter), replace=True)
            med_s = np.nanmedian(sparc_scatter[idx_s], axis=0)
            inv_s = find_inversion(common_centers, med_s)
            if inv_s is not None:
                sparc_boot_inv.append(inv_s)

        idx_e = np.random.choice(len(eagle_scatter), size=len(eagle_scatter), replace=True)
        med_e = np.nanmedian(eagle_scatter[idx_e], axis=0)
        inv_e = find_inversion(common_centers, med_e)
        if inv_e is not None:
            eagle_boot_inv.append(inv_e)

    if sparc_boot_inv:
        arr = np.array(sparc_boot_inv)
        print(f"    SPARC: {len(arr)}/{n_boot} boots found inversion")
        print(f"      Mean: {arr.mean():.3f} ± {arr.std():.3f}")
        print(f"      95% CI: [{np.percentile(arr,2.5):.3f}, {np.percentile(arr,97.5):.3f}]")
        near = np.sum(np.abs(arr - LOG_G_DAGGER) < 0.30)
        print(f"      Near g† (|Δ|<0.30): {near}/{len(arr)} ({100*near/len(arr):.1f}%)")

    if eagle_boot_inv:
        arr = np.array(eagle_boot_inv)
        print(f"    EAGLE: {len(arr)}/{n_boot} boots found inversion")
        print(f"      Mean: {arr.mean():.3f} ± {arr.std():.3f}")
        print(f"      95% CI: [{np.percentile(arr,2.5):.3f}, {np.percentile(arr,97.5):.3f}]")
        near = np.sum(np.abs(arr - LOG_G_DAGGER) < 0.30)
        print(f"      Near g† (|Δ|<0.30): {near}/{len(arr)} ({100*near/len(arr):.1f}%)")
    else:
        print(f"    EAGLE: 0/{n_boot} boots found inversion")

else:
    print(f"  Insufficient EAGLE galaxies for bootstrap analysis")
    inv_eagle_gal = None
    inv_sparc_gal = None if len(sparc_scatter) < 10 else None


# ================================================================
# PART C: MOCK-OBSERVED ROTATION CURVES (BRIDGE TEST)
# ================================================================
print("\n" + "=" * 72)
print("PART C: MOCK-OBSERVED ROTATION CURVES (apples-to-apples bridge)")
print("=" * 72)

# Convert aperture masses to circular velocity curves:
# V_circ(r) = sqrt(G * M_enc(r) / r)
# Decompose: V_bar(r) from M_star + M_gas, V_obs(r) from M_total

print("\n  Building mock rotation curves from aperture masses...")

mock_per_galaxy = {}
mock_log_gbar_all = []
mock_log_gobs_all = []
n_mock_good = 0

# Realistic observational scatter for mock curves
# SPARC typical uncertainties: ~5-10% in V_obs, ~20-30% in mass-to-light
V_OBS_ERR_FRAC = 0.07   # 7% velocity error
ML_ERR_FRAC = 0.15      # 15% mass-to-light uncertainty
DIST_ERR_FRAC = 0.10    # 10% distance uncertainty

for gid, gdata in galaxy_data.items():
    aps = gdata['apertures']

    radii = []
    v_bar = []
    v_obs = []
    v_dm = []

    for ap_str, masses in sorted(aps.items(), key=lambda x: float(x[0])):
        r_kpc = float(ap_str)
        if r_kpc < 1.0:
            continue

        m_star = masses['m_star'] * MASS_UNIT
        m_gas = masses['m_gas'] * MASS_UNIT
        m_dm = masses['m_dm'] * MASS_UNIT

        m_bar = m_star + m_gas
        m_total = m_bar + m_dm

        if m_bar <= 0 or m_total <= 0 or r_kpc <= 0:
            continue

        r_m = r_kpc * kpc_m

        # V = sqrt(G * M * M_sun / r)
        v_bar_ms = np.sqrt(G * m_bar * M_sun / r_m)   # m/s
        v_obs_ms = np.sqrt(G * m_total * M_sun / r_m)  # m/s
        v_dm_ms = np.sqrt(max(G * m_dm * M_sun / r_m, 0))

        radii.append(r_kpc)
        v_bar.append(v_bar_ms / 1e3)   # km/s
        v_obs.append(v_obs_ms / 1e3)   # km/s
        v_dm.append(v_dm_ms / 1e3)     # km/s

    if len(radii) < 4:
        continue

    radii = np.array(radii)
    v_bar = np.array(v_bar)
    v_obs = np.array(v_obs)
    v_dm = np.array(v_dm)

    # Add realistic observational noise (different seed per galaxy)
    rng = np.random.default_rng(stable_seed(gid))

    # Distance uncertainty shifts all velocities coherently
    dist_factor = 1.0 + rng.normal(0, DIST_ERR_FRAC)

    # Velocity errors
    v_obs_noisy = v_obs * dist_factor * (1.0 + rng.normal(0, V_OBS_ERR_FRAC, len(v_obs)))

    # Mass-to-light ratio uncertainty on baryonic component
    ml_factor = 1.0 + rng.normal(0, ML_ERR_FRAC)
    v_bar_noisy = v_bar * np.sqrt(max(ml_factor, 0.3)) * dist_factor

    # Now compute g_bar and g_obs in SPARC-like fashion:
    # g_bar = V_bar^2 / R,  g_obs = V_obs^2 / R
    gbar = np.where(radii > 0, (v_bar_noisy * 1e3)**2 / (radii * kpc_m), 1e-15)
    gobs = np.where(radii > 0, (v_obs_noisy * 1e3)**2 / (radii * kpc_m), 1e-15)

    valid = (gbar > 1e-16) & (gobs > 1e-16) & (v_obs_noisy > 5)
    if np.sum(valid) < 3:
        continue

    lg = np.log10(gbar[valid])
    lo = np.log10(gobs[valid])

    # Apply SPARC range
    in_range = (lg >= SPARC_GBAR_MIN) & (lg <= SPARC_GBAR_MAX)
    if np.sum(in_range) >= 3:
        mock_per_galaxy[gid] = {'log_gbar': lg[in_range], 'log_gobs': lo[in_range]}
        mock_log_gbar_all.extend(lg[in_range])
        mock_log_gobs_all.extend(lo[in_range])
        n_mock_good += 1

mock_log_gbar_all = np.array(mock_log_gbar_all)
mock_log_gobs_all = np.array(mock_log_gobs_all)

print(f"  Mock RC galaxies: {n_mock_good}")
print(f"  Mock RAR points (SPARC range): {len(mock_log_gbar_all)}")
if len(mock_log_gbar_all) > 0:
    print(f"  log g_bar range: [{mock_log_gbar_all.min():.2f}, {mock_log_gbar_all.max():.2f}]")

# C.1: Pooled scatter derivative on mock RCs
print(f"\n  [C.1] Pooled scatter derivative (mock rotation curves):")
c_mock, s_mock, n_mock = scatter_profile(mock_log_gbar_all, mock_log_gobs_all)
inv_mock = find_inversion(c_mock, s_mock)

if c_mock is not None:
    for j in range(len(c_mock)):
        marker = " <-- g†" if abs(c_mock[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {c_mock[j]:+.2f}: σ = {s_mock[j]:.4f} dex (N={n_mock[j]}){marker}")
    if inv_mock is not None:
        print(f"  => Mock RC inversion at log g = {inv_mock:.3f}, Δ from g† = {abs(inv_mock - LOG_G_DAGGER):.3f} dex")
    else:
        print(f"  => No inversion in mock RCs")
else:
    print(f"  => Insufficient mock RC data for scatter profile")

# C.2: Galaxy-level on mock RCs
print(f"\n  [C.2] Galaxy-level scatter derivative (mock RCs):")
mock_scatter, mock_gids = galaxy_scatter_in_bins(mock_per_galaxy, common_edges, min_pts_per_bin=1)
print(f"    {len(mock_gids)} galaxies with ≥3 populated bins")

if len(mock_scatter) >= 10:
    mock_med, mock_ci_lo, mock_ci_hi = bootstrap_median_ci(mock_scatter)
    for j in range(n_bins):
        n_gals = np.sum(np.isfinite(mock_scatter[:, j]))
        marker = " <-- g†" if abs(common_centers[j] - LOG_G_DAGGER) < 0.20 else ""
        if np.isfinite(mock_med[j]):
            print(f"    log g = {common_centers[j]:+.2f}: σ = {mock_med[j]:.4f} [{mock_ci_lo[j]:.4f}, {mock_ci_hi[j]:.4f}] (N_gal={n_gals}){marker}")

    inv_mock_gal = find_inversion(common_centers, mock_med)
    if inv_mock_gal is not None:
        print(f"  => Galaxy-level mock RC inversion at log g = {inv_mock_gal:.3f}, Δ = {abs(inv_mock_gal - LOG_G_DAGGER):.3f}")
    else:
        print(f"  => No galaxy-level inversion in mock RCs")

    # Bootstrap inversion
    mock_boot_inv = []
    for b in range(2000):
        idx = np.random.choice(len(mock_scatter), size=len(mock_scatter), replace=True)
        med = np.nanmedian(mock_scatter[idx], axis=0)
        inv = find_inversion(common_centers, med)
        if inv is not None:
            mock_boot_inv.append(inv)

    if mock_boot_inv:
        arr = np.array(mock_boot_inv)
        print(f"\n    Bootstrap (mock RCs): {len(arr)}/2000 found inversion")
        print(f"      Mean: {arr.mean():.3f} ± {arr.std():.3f}")
        print(f"      95% CI: [{np.percentile(arr,2.5):.3f}, {np.percentile(arr,97.5):.3f}]")
        near = np.sum(np.abs(arr - LOG_G_DAGGER) < 0.30)
        print(f"      Near g† (|Δ|<0.30): {near}/{len(arr)} ({100*near/len(arr):.1f}%)")
    else:
        print(f"\n    Bootstrap (mock RCs): 0/2000 found inversion")
else:
    inv_mock_gal = None
    print(f"  => Insufficient galaxies for galaxy-level mock RC analysis")

# C.3: No-noise control (intrinsic scatter only)
print(f"\n  [C.3] No-noise control (intrinsic scatter only):")
mock_clean_gbar = []
mock_clean_gobs = []

for gid, gdata in galaxy_data.items():
    aps = gdata['apertures']
    for ap_str, masses in sorted(aps.items(), key=lambda x: float(x[0])):
        r_kpc = float(ap_str)
        if r_kpc < 1.0:
            continue
        m_bar = (masses['m_star'] + masses['m_gas']) * MASS_UNIT
        m_total = m_bar + masses['m_dm'] * MASS_UNIT
        if m_bar <= 0 or m_total <= 0:
            continue
        r_m = r_kpc * kpc_m
        g_bar = G * m_bar * M_sun / r_m**2
        g_obs = G * m_total * M_sun / r_m**2
        if g_bar > 1e-16 and g_obs > 1e-16:
            lg = np.log10(g_bar)
            lo = np.log10(g_obs)
            if SPARC_GBAR_MIN <= lg <= SPARC_GBAR_MAX:
                mock_clean_gbar.append(lg)
                mock_clean_gobs.append(lo)

mock_clean_gbar = np.array(mock_clean_gbar)
mock_clean_gobs = np.array(mock_clean_gobs)

c_clean, s_clean, n_clean = scatter_profile(mock_clean_gbar, mock_clean_gobs)
inv_clean = find_inversion(c_clean, s_clean)

if c_clean is not None:
    print(f"    {len(mock_clean_gbar)} clean RAR points")
    for j in range(len(c_clean)):
        marker = " <-- g†" if abs(c_clean[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {c_clean[j]:+.2f}: σ = {s_clean[j]:.4f} dex (N={n_clean[j]}){marker}")
    if inv_clean is not None:
        print(f"  => Clean (no noise) inversion at log g = {inv_clean:.3f}")
    else:
        print(f"  => No inversion in clean data (intrinsic scatter only)")


# ================================================================
# FINAL VERDICT
# ================================================================
print("\n" + "=" * 72)
print("FINAL VERDICT: EAGLE ΛCDM HYDRO vs OBSERVED RAR")
print("=" * 72)

# Determine if EAGLE shows inversion near g†
eagle_near = False
eagle_inv_summary = {}

for label, inv_val in [
    ("Pooled (SPARC range)", inv_eagle),
    ("Galaxy-level median", inv_eagle_gal if 'inv_eagle_gal' in dir() else None),
    ("Mock rotation curves", inv_mock if 'inv_mock' in dir() else None),
    ("Mock RC galaxy-level", inv_mock_gal if 'inv_mock_gal' in dir() else None),
    ("Clean (no noise)", inv_clean if 'inv_clean' in dir() else None),
]:
    if inv_val is not None:
        delta = abs(inv_val - LOG_G_DAGGER)
        eagle_inv_summary[label] = {'log_g': inv_val, 'delta': delta}
        print(f"  EAGLE {label}: inversion at log g = {inv_val:.3f} (Δ = {delta:.3f} dex)")
        if delta < 0.20:
            eagle_near = True
    else:
        eagle_inv_summary[label] = None
        print(f"  EAGLE {label}: NO inversion")

sparc_inv_val = inv_sparc
if sparc_inv_val is not None:
    print(f"\n  SPARC (observed): inversion at log g = {sparc_inv_val:.3f} (Δ = {abs(sparc_inv_val - LOG_G_DAGGER):.3f} dex)")
sparc_gal_val = inv_sparc_gal if 'inv_sparc_gal' in dir() else None
if sparc_gal_val is not None:
    print(f"  SPARC galaxy-level: inversion at log g = {sparc_gal_val:.3f} (Δ = {abs(sparc_gal_val - LOG_G_DAGGER):.3f} dex)")

if not eagle_near:
    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  DISCRIMINATING: EAGLE ΛCDM hydro does NOT reproduce the       ║")
    print(f"  ║  scatter derivative inversion at g† across ALL methods:         ║")
    print(f"  ║    - Pooled (SPARC range)                                      ║")
    print(f"  ║    - Galaxy-level (no pseudoreplication)                        ║")
    print(f"  ║    - Mock rotation curves (apples-to-apples)                   ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")
else:
    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  CAUTION: EAGLE produces inversion near g† in some methods.    ║")
    print(f"  ║  Carefully check if this is a binning artifact.                ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")


# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    'test': 'eagle_rar_inversion_v2',
    'description': 'EAGLE aperture masses with corrected units, SPARC-range restriction, galaxy-level stats, mock RCs',
    'unit_fix': 'v1 incorrectly multiplied masses by 1e10; cached masses are already in M_sun',
    'n_eagle_galaxies_total': len(galaxy_data),
    'n_eagle_in_sparc_range': len(eagle_per_galaxy),
    'n_eagle_rar_points_sparc_range': len(eagle_log_gbar_all),
    'n_sparc_galaxies': len(sparc_per_galaxy),
    'n_sparc_rar_points': len(sparc_log_gbar_all),
    'sparc_gbar_range': [float(SPARC_GBAR_MIN), float(SPARC_GBAR_MAX)],

    'part_a_pooled': {
        'sparc_inversion': float(inv_sparc) if inv_sparc is not None else None,
        'eagle_inversion': float(inv_eagle) if inv_eagle is not None else None,
        'eagle_robustness_offsets': {
            'n_found': len(inv_offsets_eagle),
            'values': [float(x) for x in inv_offsets_eagle] if inv_offsets_eagle else [],
        },
        'sparc_scatter': {
            'centers': [float(x) for x in c_sparc] if c_sparc is not None else [],
            'sigmas': [float(x) for x in s_sparc] if s_sparc is not None else [],
        },
        'eagle_scatter': {
            'centers': [float(x) for x in c_eagle] if c_eagle is not None else [],
            'sigmas': [float(x) for x in s_eagle] if s_eagle is not None else [],
        },
    },

    'part_b_galaxy_level': {
        'n_sparc_galaxies': len(sparc_gids) if 'sparc_gids' in dir() else 0,
        'n_eagle_galaxies': len(eagle_gids) if 'eagle_gids' in dir() else 0,
        'sparc_inversion': float(inv_sparc_gal) if (inv_sparc_gal is not None and 'inv_sparc_gal' in dir()) else None,
        'eagle_inversion': float(inv_eagle_gal) if (inv_eagle_gal is not None and 'inv_eagle_gal' in dir()) else None,
        'sparc_bootstrap': {
            'n_found': len(sparc_boot_inv) if 'sparc_boot_inv' in dir() else 0,
            'mean': float(np.mean(sparc_boot_inv)) if ('sparc_boot_inv' in dir() and sparc_boot_inv) else None,
            'std': float(np.std(sparc_boot_inv)) if ('sparc_boot_inv' in dir() and sparc_boot_inv) else None,
        },
        'eagle_bootstrap': {
            'n_found': len(eagle_boot_inv) if 'eagle_boot_inv' in dir() else 0,
            'mean': float(np.mean(eagle_boot_inv)) if ('eagle_boot_inv' in dir() and eagle_boot_inv) else None,
            'std': float(np.std(eagle_boot_inv)) if ('eagle_boot_inv' in dir() and eagle_boot_inv) else None,
        },
        'common_bins': [float(x) for x in common_centers],
        'sparc_median_scatter': [float(x) if np.isfinite(x) else None for x in sparc_med] if 'sparc_med' in dir() else [],
        'eagle_median_scatter': [float(x) if np.isfinite(x) else None for x in eagle_med] if 'eagle_med' in dir() else [],
    },

    'part_c_mock_rc': {
        'n_galaxies': n_mock_good,
        'n_rar_points': len(mock_log_gbar_all),
        'pooled_inversion': float(inv_mock) if (inv_mock is not None and 'inv_mock' in dir()) else None,
        'galaxy_level_inversion': float(inv_mock_gal) if ('inv_mock_gal' in dir() and inv_mock_gal is not None) else None,
        'no_noise_inversion': float(inv_clean) if ('inv_clean' in dir() and inv_clean is not None) else None,
        'mock_bootstrap': {
            'n_found': len(mock_boot_inv) if 'mock_boot_inv' in dir() else 0,
            'mean': float(np.mean(mock_boot_inv)) if ('mock_boot_inv' in dir() and mock_boot_inv) else None,
            'std': float(np.std(mock_boot_inv)) if ('mock_boot_inv' in dir() and mock_boot_inv) else None,
        },
        'mock_scatter': {
            'centers': [float(x) for x in c_mock] if c_mock is not None else [],
            'sigmas': [float(x) for x in s_mock] if s_mock is not None else [],
        },
    },

    'verdict': 'DISCRIMINATING' if not eagle_near else 'NOT_DISCRIMINATING',
    'eagle_inversions_summary': {
        k: v for k, v in eagle_inv_summary.items()
    },
}

outpath = os.path.join(RESULTS_DIR, 'summary_eagle_rar_inversion_v2.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Results saved to {outpath}")
print("=" * 72)
