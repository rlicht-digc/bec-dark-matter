#!/usr/bin/env python3
"""
SPARC-Matched Analog Analysis in EAGLE & TNG
==============================================

For each of 131 quality SPARC galaxies, find 5 closest analogs in EAGLE
and TNG simulations (matched on stellar mass and size), then run identical
RAR analyses on all three samples and compare feature-by-feature.

This controls for sample selection effects that could bias full-catalog
comparisons (EAGLE=29,716; TNG=203,524 vs SPARC=175).

Tests performed:
  1. Scatter inversion (minimum scatter → g†?)
  2. Inversion proximity to g†
  3. Radial coherence (ACF) — EAGLE & SPARC only (TNG lacks radial resolution)
  4. Scatter profile shape comparison
  5. Per-galaxy scatter distribution
  6. Environmental scatter (flagged as infeasible — no env data in sim catalogs)

Known limitations:
  - TNG HDF5 has only 4 apertures (5,10,30,100 kpc) → no ACF/periodicity
  - TNG g_bar is stellar-only (no gas mass in HDF5)
  - EAGLE R_half derived from 10-aperture interpolation
  - R_disk↔R_half conversion assumes pure exponential disks (R_half ≈ 1.678 × R_disk)

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SPARC_DIR = os.path.join(DATA_DIR, 'sparc')
EAGLE_CACHE = os.path.join(DATA_DIR, 'eagle_rar', 'eagle_aperture_masses.json')
TNG_HDF5 = os.path.expanduser('~/Desktop/tng_cross_validation/aperture_masses.hdf5')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────
G = 6.674e-11        # m^3 kg^-1 s^-2
M_sun = 1.989e30     # kg
kpc_m = 3.086e19     # m per kpc
g_dagger = 1.20e-10  # m/s^2
LOG_GD = np.log10(g_dagger)  # -9.921

Y_DISK = 0.5         # stellar mass-to-light ratio [3.6μm]
R_HALF_FACTOR = 1.678  # R_half / R_disk for exponential disk

N_ANALOGS = 5        # analogs per SPARC galaxy
N_BOOT = 1000        # bootstrap resamples
BIN_WIDTH = 0.30     # dex bin width for scatter profiles
MIN_BIN_PTS = 10     # minimum points per acceleration bin
MAX_MATCH_DIST = 5.0 # maximum normalized distance for a valid match

np.random.seed(42)
t0 = time.time()

def elapsed():
    return f"[{time.time()-t0:.0f}s]"

print("=" * 76)
print("SPARC-MATCHED ANALOG ANALYSIS — EAGLE & TNG")
print("  Controlling for sample selection in RAR feature comparison")
print("=" * 76)


# ══════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def rar_function(log_gbar, a0=g_dagger):
    """Standard RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))."""
    gbar = 10.0**np.asarray(log_gbar, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def rar_residual(log_gbar, log_gobs):
    """Residual from RAR fit."""
    pred = rar_function(log_gbar)
    resid = np.asarray(log_gobs) - pred
    valid = np.isfinite(resid)
    return resid[valid], np.asarray(log_gbar)[valid]


def scatter_profile(log_gbar, log_gobs, bin_width=BIN_WIDTH, offset=0.0,
                    min_pts=MIN_BIN_PTS):
    """Binned scatter profile of RAR residuals."""
    resid, lg = rar_residual(log_gbar, log_gobs)
    if len(resid) < 20:
        return None, None, None
    lo = lg.min() + offset
    hi = lg.max()
    edges = np.arange(lo, hi + bin_width, bin_width)
    centers, sigmas, counts = [], [], []
    for j in range(len(edges) - 1):
        mask = (lg >= edges[j]) & (lg < edges[j + 1])
        n = np.sum(mask)
        if n >= min_pts:
            centers.append(0.5 * (edges[j] + edges[j + 1]))
            sigmas.append(np.std(resid[mask]))
            counts.append(n)
    if len(centers) < 4:
        return None, None, None
    return np.array(centers), np.array(sigmas), np.array(counts)


def find_inversion(centers, sigmas):
    """Find scatter peak (sign change in derivative) nearest to g†."""
    if centers is None or len(centers) < 4:
        return None
    dsigma = np.diff(sigmas)
    dcenter = 0.5 * (centers[:-1] + centers[1:])
    crossings = []
    for j in range(len(dsigma) - 1):
        if dsigma[j] > 0 and dsigma[j + 1] < 0:
            x0, x1 = dcenter[j], dcenter[j + 1]
            y0, y1 = dsigma[j], dsigma[j + 1]
            crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(crossing)
    if not crossings:
        return None
    crossings = np.array(crossings)
    nearest_idx = np.argmin(np.abs(crossings - LOG_GD))
    return float(crossings[nearest_idx])


def lag_autocorrelation(x, lag=1, center=True):
    """Lag-k autocorrelation (demeaned by default)."""
    n = len(x)
    if n <= lag + 1:
        return np.nan
    x_use = x - np.mean(x) if center else x.copy()
    var = np.mean(x_use**2)
    if var < 1e-30:
        return np.nan
    return np.mean(x_use[:n - lag] * x_use[lag:]) / var


def derive_rhalf(radii_kpc, m_star_cumulative):
    """Derive stellar half-mass radius from cumulative mass profile.

    Uses log-linear interpolation between the two apertures bracketing
    50% of the total stellar mass.
    """
    if len(radii_kpc) < 2:
        return np.nan
    m_total = m_star_cumulative[-1]
    if m_total <= 0:
        return np.nan
    m_half = 0.5 * m_total
    # Already above half at innermost aperture
    if m_star_cumulative[0] >= m_half:
        return radii_kpc[0] * 0.5  # assign half of innermost
    for i in range(len(radii_kpc) - 1):
        if m_star_cumulative[i] < m_half <= m_star_cumulative[i + 1]:
            # Log-linear interpolation
            r0, r1 = np.log10(max(radii_kpc[i], 0.01)), np.log10(max(radii_kpc[i + 1], 0.01))
            m0, m1 = m_star_cumulative[i], m_star_cumulative[i + 1]
            if m1 - m0 < 1e-10:
                return 10**((r0 + r1) / 2)
            frac = (m_half - m0) / (m1 - m0)
            return 10**(r0 + frac * (r1 - r0))
    return np.nan


def bootstrap_scatter_profile(log_gbar, log_gobs, galaxy_ids, n_boot=N_BOOT):
    """Bootstrap scatter profile by resampling galaxies (not individual points)."""
    unique_ids = np.unique(galaxy_ids)
    n_gal = len(unique_ids)
    boot_profiles = []
    rng = np.random.default_rng(42)
    # Reference profile for bin edges
    ref_c, ref_s, ref_n = scatter_profile(log_gbar, log_gobs)
    if ref_c is None:
        return None, None, None, None
    for _ in range(n_boot):
        idx = rng.choice(unique_ids, size=n_gal, replace=True)
        mask = np.isin(galaxy_ids, idx)
        c, s, n = scatter_profile(log_gbar[mask], log_gobs[mask])
        if c is not None and len(c) == len(ref_c):
            boot_profiles.append(s)
    if len(boot_profiles) < 50:
        return ref_c, ref_s, None, ref_n
    boot_arr = np.array(boot_profiles)
    ci_lo = np.percentile(boot_arr, 2.5, axis=0)
    ci_hi = np.percentile(boot_arr, 97.5, axis=0)
    return ref_c, ref_s, (ci_lo, ci_hi), ref_n


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1: LOAD SPARC PROPERTIES
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [1] Loading SPARC properties...")

mrt_path = os.path.join(SPARC_DIR, 'SPARC_Lelli2016c.mrt')
table2_path = os.path.join(SPARC_DIR, 'SPARC_table2_rotmods.dat')

# --- Parse rotation curves from table2 ---
sparc_rc = {}
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
        if name not in sparc_rc:
            sparc_rc[name] = {'R': [], 'Vobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
        sparc_rc[name]['R'].append(rad)
        sparc_rc[name]['Vobs'].append(vobs)
        sparc_rc[name]['Vgas'].append(vgas)
        sparc_rc[name]['Vdisk'].append(vdisk)
        sparc_rc[name]['Vbul'].append(vbul)

for name in sparc_rc:
    for key in sparc_rc[name]:
        sparc_rc[name][key] = np.array(sparc_rc[name][key])

# Count points per galaxy
sparc_npts = {name: len(sparc_rc[name]['R']) for name in sparc_rc}

# --- Parse galaxy properties from MRT ---
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
        T = int(parts[0])
        D = float(parts[1])         # Mpc
        Inc = float(parts[4])       # deg
        L36 = float(parts[6])       # 10^9 L_sun
        Reff = float(parts[8])      # kpc
        Rdisk = float(parts[10])    # kpc
        MHI = float(parts[12])      # 10^9 M_sun
        Vflat = float(parts[14])    # km/s
        Q = int(parts[16])

        M_star = Y_DISK * L36 * 1e9  # M_sun
        M_gas = MHI * 1e9            # M_sun
        R_half_equiv = R_HALF_FACTOR * Rdisk if Rdisk > 0 else np.nan

        npts = sparc_npts.get(name, 0)

        sparc_props[name] = {
            'T': T, 'D': D, 'Inc': Inc, 'Q': Q,
            'L36': L36, 'M_star': M_star, 'M_gas': M_gas,
            'Rdisk': Rdisk, 'R_half_equiv': R_half_equiv,
            'Reff': Reff, 'Vflat': Vflat, 'N_points': npts,
        }
    except (ValueError, IndexError):
        continue

# Apply quality cuts
sparc_quality = {}
for name, p in sparc_props.items():
    if p['Q'] <= 2 and 30 <= p['Inc'] <= 85 and p['N_points'] >= 10:
        if p['M_star'] > 0 and p['R_half_equiv'] > 0:
            sparc_quality[name] = p

print(f"  Total SPARC galaxies: {len(sparc_props)}")
print(f"  After quality cuts (Q<=2, 30<=Inc<=85, N>=10): {len(sparc_quality)}")

# Compute SPARC RAR
sparc_rar = {}
for name in sparc_quality:
    if name not in sparc_rc:
        continue
    rc = sparc_rc[name]
    R = rc['R']
    Vobs, Vgas, Vdisk, Vbul = rc['Vobs'], rc['Vgas'], rc['Vdisk'], rc['Vbul']

    Vbar_sq = Y_DISK * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < 5:
        continue

    sparc_rar[name] = {
        'log_gbar': np.log10(gbar_SI[valid]),
        'log_gobs': np.log10(gobs_SI[valid]),
        'R_kpc': R[valid],
    }

print(f"  SPARC galaxies with valid RAR: {len(sparc_rar)}")

# Collect SPARC arrays for matching
sparc_names = sorted(sparc_quality.keys())
sparc_log_Mstar = np.array([np.log10(sparc_quality[n]['M_star']) for n in sparc_names])
sparc_log_Rhalf = np.array([np.log10(sparc_quality[n]['R_half_equiv']) for n in sparc_names])
sparc_Vflat = np.array([sparc_quality[n]['Vflat'] for n in sparc_names])

sigma_Mstar = np.std(sparc_log_Mstar)
sigma_R = np.std(sparc_log_Rhalf[np.isfinite(sparc_log_Rhalf)])
print(f"  Matching normalization: σ(log M*) = {sigma_Mstar:.3f}, σ(log R) = {sigma_R:.3f}")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2: LOAD EAGLE DATA & DERIVE PROPERTIES
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [2] Loading EAGLE data & deriving properties...")

with open(EAGLE_CACHE, 'r') as f:
    eagle_raw = json.load(f)

aperture_sizes = eagle_raw['aperture_sizes']  # [1,3,5,10,20,30,40,50,70,100] kpc
galaxy_data = eagle_raw['galaxy_data']
print(f"  Raw EAGLE galaxies: {len(galaxy_data)}, apertures: {aperture_sizes}")

eagle_props = {}
eagle_rar = {}

for gid, gdata in galaxy_data.items():
    aps = gdata['apertures']

    # Build cumulative mass profiles at each aperture
    radii = []
    m_star_cum = []
    m_gas_cum = []
    m_dm_cum = []
    m_total_cum = []

    for ap_str in sorted(aps.keys(), key=float):
        r_kpc = float(ap_str)
        masses = aps[ap_str]
        m_star = masses['m_star']
        m_gas = masses['m_gas']
        m_dm = masses['m_dm']

        if m_star <= 0 and m_gas <= 0:
            continue

        radii.append(r_kpc)
        m_star_cum.append(m_star)
        m_gas_cum.append(m_gas)
        m_dm_cum.append(m_dm)
        m_total_cum.append(m_star + m_gas + m_dm)

    if len(radii) < 3:
        continue

    radii = np.array(radii)
    m_star_cum = np.array(m_star_cum)
    m_gas_cum = np.array(m_gas_cum)
    m_total_cum = np.array(m_total_cum)

    # Total stellar mass (at 100 kpc)
    M_star_total = m_star_cum[-1]
    if M_star_total < 1e6:  # skip very low mass
        continue

    # Derive R_half
    R_half = derive_rhalf(radii, m_star_cum)
    if np.isnan(R_half) or R_half <= 0:
        continue

    # Compute V_circ profile and Vflat
    m_bar = m_star_cum + m_gas_cum
    v_circ = np.sqrt(G * m_total_cum * M_sun / (radii * kpc_m))  # m/s
    v_circ_kms = v_circ / 1e3
    Vflat_eagle = float(np.max(v_circ_kms))

    # Compute RAR at each aperture
    g_bar = G * m_bar * M_sun / (radii * kpc_m)**2
    g_obs = G * m_total_cum * M_sun / (radii * kpc_m)**2

    valid = (g_bar > 1e-16) & (g_obs > 1e-16)
    if np.sum(valid) < 3:
        continue

    log_Mstar = np.log10(M_star_total)
    log_Rhalf = np.log10(R_half)

    eagle_props[gid] = {
        'log_Mstar': log_Mstar,
        'log_Rhalf': log_Rhalf,
        'Vflat': Vflat_eagle,
        'M_star': M_star_total,
        'R_half': R_half,
    }
    eagle_rar[gid] = {
        'log_gbar': np.log10(g_bar[valid]),
        'log_gobs': np.log10(g_obs[valid]),
        'R_kpc': radii[valid],
    }

print(f"  EAGLE galaxies with valid properties: {len(eagle_props)}")
eagle_log_Mstar_all = np.array([eagle_props[g]['log_Mstar'] for g in eagle_props])
eagle_log_Rhalf_all = np.array([eagle_props[g]['log_Rhalf'] for g in eagle_props])
print(f"  EAGLE log M* range: [{eagle_log_Mstar_all.min():.1f}, {eagle_log_Mstar_all.max():.1f}]")
print(f"  EAGLE log R_half range: [{eagle_log_Rhalf_all.min():.2f}, {eagle_log_Rhalf_all.max():.2f}]")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3: LOAD TNG DATA & DERIVE PROPERTIES
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [3] Loading TNG data...")

tng_props = {}
tng_rar = {}
TNG_AVAILABLE = False

if os.path.exists(TNG_HDF5):
    try:
        import h5py
        with h5py.File(TNG_HDF5, 'r') as f:
            snap = f['Snapshot_99']
            subfind_ids = snap['SubfindID'][:]
            m_star_5 = snap['SubhaloStellarMass_in_r5pkpc'][:]
            m_star_10 = snap['SubhaloStellarMass_in_r10pkpc'][:]
            m_star_30 = snap['SubhaloStellarMass_in_r30pkpc'][:]
            m_star_100 = snap['SubhaloStellarMass_in_r100pkpc'][:]
            m_total_5 = snap['SubhaloTotalMass_in_r5pkpc'][:]
            m_total_10 = snap['SubhaloTotalMass_in_r10pkpc'][:]
            m_total_30 = snap['SubhaloTotalMass_in_r30pkpc'][:]
            m_total_100 = snap['SubhaloTotalMass_in_r100pkpc'][:]

        print(f"  TNG Snapshot_99: {len(subfind_ids)} subhalos loaded")

        tng_apertures = np.array([5.0, 10.0, 30.0, 100.0])  # pkpc

        n_valid = 0
        for i in range(len(subfind_ids)):
            M_star_total = m_star_100[i]
            if M_star_total < 1e8:  # minimum stellar mass
                continue

            ms = np.array([m_star_5[i], m_star_10[i], m_star_30[i], m_star_100[i]])
            mt = np.array([m_total_5[i], m_total_10[i], m_total_30[i], m_total_100[i]])

            # Skip if non-monotonic or invalid
            if np.any(ms <= 0) or np.any(mt <= ms):
                continue

            # Derive R_half from 4-point profile
            R_half = derive_rhalf(tng_apertures, ms)
            if np.isnan(R_half) or R_half <= 0.1:
                continue

            # Compute RAR (stellar-only g_bar — no gas mass available)
            g_bar = G * ms * M_sun / (tng_apertures * kpc_m)**2
            g_obs = G * mt * M_sun / (tng_apertures * kpc_m)**2

            valid = (g_bar > 1e-16) & (g_obs > 1e-16)
            if np.sum(valid) < 2:
                continue

            gid = str(subfind_ids[i])
            tng_props[gid] = {
                'log_Mstar': np.log10(M_star_total),
                'log_Rhalf': np.log10(R_half),
                'M_star': M_star_total,
                'R_half': R_half,
            }
            tng_rar[gid] = {
                'log_gbar': np.log10(g_bar[valid]),
                'log_gobs': np.log10(g_obs[valid]),
                'R_kpc': tng_apertures[valid],
            }
            n_valid += 1

        TNG_AVAILABLE = len(tng_props) > 100
        print(f"  TNG galaxies with valid properties: {len(tng_props)}")
        if TNG_AVAILABLE:
            tng_log_Mstar_all = np.array([tng_props[g]['log_Mstar'] for g in tng_props])
            tng_log_Rhalf_all = np.array([tng_props[g]['log_Rhalf'] for g in tng_props])
            print(f"  TNG log M* range: [{tng_log_Mstar_all.min():.1f}, {tng_log_Mstar_all.max():.1f}]")
            print(f"  TNG log R_half range: [{tng_log_Rhalf_all.min():.2f}, {tng_log_Rhalf_all.max():.2f}]")
    except Exception as e:
        print(f"  WARNING: Failed to load TNG HDF5: {e}")
        TNG_AVAILABLE = False
else:
    print(f"  TNG HDF5 not found at {TNG_HDF5}")

if not TNG_AVAILABLE:
    print(f"  TNG analysis will be skipped where data is unavailable.")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4: MATCH ANALOGS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [4] Matching analogs...")

def find_analogs(sparc_names, sparc_log_M, sparc_log_R, sim_props, sim_ids,
                 sim_log_M, sim_log_R, sigma_M, sigma_R, n_analogs=N_ANALOGS,
                 max_dist=MAX_MATCH_DIST):
    """For each SPARC galaxy, find n_analogs closest simulation galaxies."""
    matches = {}
    unmatched = []
    all_distances = []

    for i, name in enumerate(sparc_names):
        lm = sparc_log_M[i]
        lr = sparc_log_R[i]
        if not np.isfinite(lm) or not np.isfinite(lr):
            unmatched.append(name)
            continue

        # Normalized Euclidean distance
        d_m = (sim_log_M - lm) / sigma_M
        d_r = (sim_log_R - lr) / sigma_R
        distances = np.sqrt(d_m**2 + d_r**2)

        # Sort and pick closest
        order = np.argsort(distances)
        selected = []
        for idx in order[:n_analogs * 2]:  # check extras in case of invalids
            if distances[idx] <= max_dist:
                selected.append((sim_ids[idx], float(distances[idx])))
            if len(selected) >= n_analogs:
                break

        if len(selected) == 0:
            unmatched.append(name)
            continue

        matches[name] = selected
        all_distances.extend([d for _, d in selected])

    return matches, unmatched, np.array(all_distances)


# EAGLE matching
eagle_ids = list(eagle_props.keys())
eagle_lm = np.array([eagle_props[g]['log_Mstar'] for g in eagle_ids])
eagle_lr = np.array([eagle_props[g]['log_Rhalf'] for g in eagle_ids])

eagle_matches, eagle_unmatched, eagle_dists = find_analogs(
    sparc_names, sparc_log_Mstar, sparc_log_Rhalf,
    eagle_props, eagle_ids, eagle_lm, eagle_lr,
    sigma_Mstar, sigma_R
)

n_eagle_unique = len(set(gid for matches in eagle_matches.values() for gid, _ in matches))
print(f"  EAGLE: {len(eagle_matches)} SPARC galaxies matched, "
      f"{len(eagle_unmatched)} unmatched")
print(f"    Total analog assignments: {sum(len(v) for v in eagle_matches.values())}")
print(f"    Unique EAGLE galaxies used: {n_eagle_unique}")
print(f"    Match distance: median={np.median(eagle_dists):.3f}, "
      f"max={np.max(eagle_dists):.3f}")

# TNG matching
tng_matches = {}
tng_unmatched = sparc_names.copy()
tng_dists = np.array([])

if TNG_AVAILABLE:
    tng_ids = list(tng_props.keys())
    tng_lm = np.array([tng_props[g]['log_Mstar'] for g in tng_ids])
    tng_lr = np.array([tng_props[g]['log_Rhalf'] for g in tng_ids])

    tng_matches, tng_unmatched, tng_dists = find_analogs(
        sparc_names, sparc_log_Mstar, sparc_log_Rhalf,
        tng_props, tng_ids, tng_lm, tng_lr,
        sigma_Mstar, sigma_R
    )
    n_tng_unique = len(set(gid for matches in tng_matches.values() for gid, _ in matches))
    print(f"  TNG: {len(tng_matches)} SPARC galaxies matched, "
          f"{len(tng_unmatched)} unmatched")
    print(f"    Total analog assignments: {sum(len(v) for v in tng_matches.values())}")
    print(f"    Unique TNG galaxies used: {n_tng_unique}")
    print(f"    Match distance: median={np.median(tng_dists):.3f}, "
          f"max={np.max(tng_dists):.3f}")

# Alternative EAGLE matching on (log M_star, Vflat)
eagle_Vflat_arr = np.array([eagle_props[g]['Vflat'] for g in eagle_ids])
sparc_Vflat_valid = sparc_Vflat.copy()
sparc_Vflat_valid[sparc_Vflat_valid <= 0] = np.nan
sigma_Vflat = np.nanstd(sparc_Vflat_valid)

# Only for galaxies with Vflat > 0
eagle_matches_vflat = {}
for i, name in enumerate(sparc_names):
    vf = sparc_Vflat[i]
    lm = sparc_log_Mstar[i]
    if vf <= 0 or not np.isfinite(lm):
        continue
    d_m = (eagle_lm - lm) / sigma_Mstar
    d_v = (eagle_Vflat_arr - vf) / max(sigma_Vflat, 1.0)
    distances = np.sqrt(d_m**2 + d_v**2)
    order = np.argsort(distances)
    selected = [(eagle_ids[idx], float(distances[idx]))
                for idx in order[:N_ANALOGS]
                if distances[idx] <= MAX_MATCH_DIST]
    if selected:
        eagle_matches_vflat[name] = selected

print(f"  EAGLE (Vflat matching): {len(eagle_matches_vflat)} SPARC galaxies matched")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5: VERIFY MATCH QUALITY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [5] Verifying match quality...")

def get_matched_properties(matches, sim_props, prop_key):
    """Collect property values for all matched analogs."""
    vals = []
    for name, analogs in matches.items():
        for gid, _ in analogs:
            if gid in sim_props and prop_key in sim_props[gid]:
                vals.append(sim_props[gid][prop_key])
    return np.array(vals)

# EAGLE matched distributions
eagle_matched_lm = get_matched_properties(eagle_matches, eagle_props, 'log_Mstar')
eagle_matched_lr = get_matched_properties(eagle_matches, eagle_props, 'log_Rhalf')
eagle_matched_vf = get_matched_properties(eagle_matches, eagle_props, 'Vflat')

print(f"\n  Match quality — KS tests (p > 0.05 = good match):")
print(f"  {'Property':<20} {'SPARC med':>10} {'EAGLE med':>10} {'KS stat':>8} {'p-value':>10}")
print(f"  {'-'*60}")

# log M_star
ks_m, p_m = stats.ks_2samp(sparc_log_Mstar, eagle_matched_lm)
print(f"  {'log M_star':<20} {np.median(sparc_log_Mstar):>10.2f} {np.median(eagle_matched_lm):>10.2f} {ks_m:>8.3f} {p_m:>10.4f}")

# log R_half
ks_r, p_r = stats.ks_2samp(sparc_log_Rhalf, eagle_matched_lr)
print(f"  {'log R_half':<20} {np.median(sparc_log_Rhalf):>10.2f} {np.median(eagle_matched_lr):>10.2f} {ks_r:>8.3f} {p_r:>10.4f}")

# Vflat (SPARC galaxies with Vflat > 0)
sparc_vf_valid = sparc_Vflat[sparc_Vflat > 0]
ks_v, p_v = stats.ks_2samp(sparc_vf_valid, eagle_matched_vf)
print(f"  {'Vflat (km/s)':<20} {np.median(sparc_vf_valid):>10.1f} {np.median(eagle_matched_vf):>10.1f} {ks_v:>8.3f} {p_v:>10.4f}")

match_quality = {
    'eagle': {
        'log_Mstar_KS': {'stat': float(ks_m), 'p': float(p_m)},
        'log_Rhalf_KS': {'stat': float(ks_r), 'p': float(p_r)},
        'Vflat_KS': {'stat': float(ks_v), 'p': float(p_v)},
        'n_matched': len(eagle_matches),
        'n_unmatched': len(eagle_unmatched),
        'median_distance': float(np.median(eagle_dists)),
    },
}

if TNG_AVAILABLE:
    tng_matched_lm = get_matched_properties(tng_matches, tng_props, 'log_Mstar')
    tng_matched_lr = get_matched_properties(tng_matches, tng_props, 'log_Rhalf')

    ks_tm, p_tm = stats.ks_2samp(sparc_log_Mstar, tng_matched_lm)
    ks_tr, p_tr = stats.ks_2samp(sparc_log_Rhalf, tng_matched_lr)

    print(f"\n  {'Property':<20} {'SPARC med':>10} {'TNG med':>10} {'KS stat':>8} {'p-value':>10}")
    print(f"  {'-'*60}")
    print(f"  {'log M_star':<20} {np.median(sparc_log_Mstar):>10.2f} {np.median(tng_matched_lm):>10.2f} {ks_tm:>8.3f} {p_tm:>10.4f}")
    print(f"  {'log R_half':<20} {np.median(sparc_log_Rhalf):>10.2f} {np.median(tng_matched_lr):>10.2f} {ks_tr:>8.3f} {p_tr:>10.4f}")

    match_quality['tng'] = {
        'log_Mstar_KS': {'stat': float(ks_tm), 'p': float(p_tm)},
        'log_Rhalf_KS': {'stat': float(ks_tr), 'p': float(p_tr)},
        'n_matched': len(tng_matches),
        'n_unmatched': len(tng_unmatched),
        'median_distance': float(np.median(tng_dists)),
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 6: COLLECT RAR DATA FOR MATCHED SAMPLES
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [6] Collecting RAR data for matched samples...")

def collect_matched_rar(matches, sim_rar):
    """Collect pooled RAR data for matched analogs + per-galaxy info."""
    all_lgbar, all_lgobs, all_gids = [], [], []
    per_galaxy = {}
    for name, analogs in matches.items():
        for gid, _ in analogs:
            if gid in sim_rar:
                d = sim_rar[gid]
                all_lgbar.extend(d['log_gbar'])
                all_lgobs.extend(d['log_gobs'])
                all_gids.extend([gid] * len(d['log_gbar']))
                if gid not in per_galaxy:
                    per_galaxy[gid] = d
    return (np.array(all_lgbar), np.array(all_lgobs),
            np.array(all_gids), per_galaxy)

# SPARC pooled RAR
sparc_lgbar_all = np.concatenate([sparc_rar[n]['log_gbar'] for n in sparc_rar])
sparc_lgobs_all = np.concatenate([sparc_rar[n]['log_gobs'] for n in sparc_rar])
sparc_gids_all = np.concatenate([[n] * len(sparc_rar[n]['log_gbar']) for n in sparc_rar])
print(f"  SPARC RAR: {len(sparc_lgbar_all)} points from {len(sparc_rar)} galaxies")

# EAGLE matched RAR
eagle_m_lgbar, eagle_m_lgobs, eagle_m_gids, eagle_m_pergal = \
    collect_matched_rar(eagle_matches, eagle_rar)
print(f"  EAGLE matched RAR: {len(eagle_m_lgbar)} points from "
      f"{len(eagle_m_pergal)} unique galaxies")

# TNG matched RAR
if TNG_AVAILABLE and tng_matches:
    tng_m_lgbar, tng_m_lgobs, tng_m_gids, tng_m_pergal = \
        collect_matched_rar(tng_matches, tng_rar)
    print(f"  TNG matched RAR: {len(tng_m_lgbar)} points from "
          f"{len(tng_m_pergal)} unique galaxies")
else:
    tng_m_lgbar = np.array([])
    tng_m_lgobs = np.array([])
    tng_m_gids = np.array([])
    tng_m_pergal = {}


# ══════════════════════════════════════════════════════════════════════
#  SECTION 7: TEST 1 — SCATTER INVERSION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [7] Test 1: Scatter Inversion")
print("=" * 72)

# SPARC
c_sparc, s_sparc, n_sparc = scatter_profile(sparc_lgbar_all, sparc_lgobs_all)
inv_sparc = find_inversion(c_sparc, s_sparc)

print(f"\n  SPARC scatter profile:")
if c_sparc is not None:
    for j in range(len(c_sparc)):
        marker = " <-- g†" if abs(c_sparc[j] - LOG_GD) < 0.20 else ""
        print(f"    log g = {c_sparc[j]:+.2f}: σ = {s_sparc[j]:.4f} dex (N={n_sparc[j]}){marker}")
    if inv_sparc is not None:
        print(f"  => Inversion at log g = {inv_sparc:.3f}, "
              f"Δ from g† = {abs(inv_sparc - LOG_GD):.3f} dex")

# EAGLE matched
c_eagle_m, s_eagle_m, n_eagle_m = scatter_profile(eagle_m_lgbar, eagle_m_lgobs)
inv_eagle_m = find_inversion(c_eagle_m, s_eagle_m)

print(f"\n  EAGLE matched scatter profile:")
if c_eagle_m is not None:
    for j in range(len(c_eagle_m)):
        marker = " <-- g†" if abs(c_eagle_m[j] - LOG_GD) < 0.20 else ""
        print(f"    log g = {c_eagle_m[j]:+.2f}: σ = {s_eagle_m[j]:.4f} dex "
              f"(N={n_eagle_m[j]}){marker}")
    if inv_eagle_m is not None:
        print(f"  => Inversion at log g = {inv_eagle_m:.3f}, "
              f"Δ from g† = {abs(inv_eagle_m - LOG_GD):.3f} dex")
    else:
        print(f"  => No inversion found")

# TNG matched
inv_tng_m = None
c_tng_m, s_tng_m, n_tng_m = None, None, None
if TNG_AVAILABLE and len(tng_m_lgbar) > 50:
    c_tng_m, s_tng_m, n_tng_m = scatter_profile(tng_m_lgbar, tng_m_lgobs)
    inv_tng_m = find_inversion(c_tng_m, s_tng_m)

    print(f"\n  TNG matched scatter profile:")
    if c_tng_m is not None:
        for j in range(len(c_tng_m)):
            marker = " <-- g†" if abs(c_tng_m[j] - LOG_GD) < 0.20 else ""
            print(f"    log g = {c_tng_m[j]:+.2f}: σ = {s_tng_m[j]:.4f} dex "
                  f"(N={n_tng_m[j]}){marker}")
        if inv_tng_m is not None:
            print(f"  => Inversion at log g = {inv_tng_m:.3f}, "
                  f"Δ from g† = {abs(inv_tng_m - LOG_GD):.3f} dex")
        else:
            print(f"  => No inversion found")

# Bootstrap CIs
print(f"\n  Bootstrapping scatter profiles (N={N_BOOT})...")
bc_sparc, bs_sparc, bci_sparc, bn_sparc = bootstrap_scatter_profile(
    sparc_lgbar_all, sparc_lgobs_all, sparc_gids_all)
bc_eagle, bs_eagle, bci_eagle, bn_eagle = bootstrap_scatter_profile(
    eagle_m_lgbar, eagle_m_lgobs, eagle_m_gids)

bci_tng = None
if TNG_AVAILABLE and len(tng_m_lgbar) > 50:
    bc_tng, bs_tng, bci_tng, bn_tng = bootstrap_scatter_profile(
        tng_m_lgbar, tng_m_lgobs, tng_m_gids)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 8: TEST 2 — INVERSION PROXIMITY TO g†
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [8] Test 2: Inversion Proximity to g†")
print("=" * 72)

test2_results = {}
if inv_sparc is not None:
    delta_sparc = abs(inv_sparc - LOG_GD)
    test2_results['SPARC'] = {'inversion': inv_sparc, 'delta_gdagger': delta_sparc}
    print(f"  SPARC: inversion at {inv_sparc:.3f}, Δ = {delta_sparc:.3f} dex from g†")

if inv_eagle_m is not None:
    delta_eagle = abs(inv_eagle_m - LOG_GD)
    test2_results['EAGLE_matched'] = {'inversion': inv_eagle_m, 'delta_gdagger': delta_eagle}
    print(f"  EAGLE matched: inversion at {inv_eagle_m:.3f}, Δ = {delta_eagle:.3f} dex from g†")
else:
    test2_results['EAGLE_matched'] = {'inversion': None, 'delta_gdagger': None}
    print(f"  EAGLE matched: no inversion found")

if inv_tng_m is not None:
    delta_tng = abs(inv_tng_m - LOG_GD)
    test2_results['TNG_matched'] = {'inversion': inv_tng_m, 'delta_gdagger': delta_tng}
    print(f"  TNG matched: inversion at {inv_tng_m:.3f}, Δ = {delta_tng:.3f} dex from g†")
elif TNG_AVAILABLE:
    test2_results['TNG_matched'] = {'inversion': None, 'delta_gdagger': None}
    print(f"  TNG matched: no inversion found")

# Robustness across bin offsets
print(f"\n  Robustness (10 bin offsets):")
inv_offsets = {'SPARC': [], 'EAGLE': [], 'TNG': []}
for off in np.linspace(0, 0.25, 10):
    cs, ss, _ = scatter_profile(sparc_lgbar_all, sparc_lgobs_all, offset=off)
    inv_s = find_inversion(cs, ss)
    if inv_s is not None:
        inv_offsets['SPARC'].append(inv_s)

    ce, se, _ = scatter_profile(eagle_m_lgbar, eagle_m_lgobs, offset=off)
    inv_e = find_inversion(ce, se)
    if inv_e is not None:
        inv_offsets['EAGLE'].append(inv_e)

    if TNG_AVAILABLE and len(tng_m_lgbar) > 50:
        ct, st, _ = scatter_profile(tng_m_lgbar, tng_m_lgobs, offset=off)
        inv_t = find_inversion(ct, st)
        if inv_t is not None:
            inv_offsets['TNG'].append(inv_t)

for label, vals in inv_offsets.items():
    if vals:
        arr = np.array(vals)
        print(f"    {label}: {len(arr)}/10 offsets, "
              f"mean = {arr.mean():.3f} ± {arr.std():.3f}")
    else:
        print(f"    {label}: 0/10 offsets found inversions")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 9: TEST 3 — RADIAL COHERENCE (ACF)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [9] Test 3: Radial Coherence (ACF)")
print("=" * 72)

def compute_acf_distribution(per_galaxy_rar, min_pts=5):
    """Compute lag-1 ACF for each galaxy's RAR residuals."""
    acf_values = []
    for gid, d in per_galaxy_rar.items():
        lg, lo = d['log_gbar'], d['log_gobs']
        if len(lg) < min_pts:
            continue
        # Sort by radius (or by log_gbar as proxy)
        order = np.argsort(lg)
        lg_s, lo_s = lg[order], lo[order]
        resid = lo_s - rar_function(lg_s)
        valid = np.isfinite(resid)
        if np.sum(valid) < min_pts:
            continue
        acf = lag_autocorrelation(resid[valid])
        if np.isfinite(acf):
            acf_values.append(acf)
    return np.array(acf_values)


# SPARC ACF (full resolution)
sparc_acf = compute_acf_distribution(sparc_rar, min_pts=10)
print(f"\n  SPARC ACF (full res): N={len(sparc_acf)}, "
      f"mean={np.mean(sparc_acf):.3f} ± {np.std(sparc_acf)/np.sqrt(len(sparc_acf)):.3f}, "
      f"median={np.median(sparc_acf):.3f}")

# EAGLE matched ACF
eagle_acf = compute_acf_distribution(eagle_m_pergal, min_pts=5)
print(f"  EAGLE matched ACF: N={len(eagle_acf)}, "
      f"mean={np.mean(eagle_acf):.3f} ± {np.std(eagle_acf)/np.sqrt(len(eagle_acf)):.3f}, "
      f"median={np.median(eagle_acf):.3f}")

# KS test
ks_acf, p_acf = stats.ks_2samp(sparc_acf, eagle_acf)
print(f"\n  KS test (SPARC full vs EAGLE matched): stat={ks_acf:.3f}, p={p_acf:.4f}")

# SPARC subsampled to 10 points (matching EAGLE resolution)
sparc_acf_10pt = []
for name, d in sparc_rar.items():
    lg, lo, rr = d['log_gbar'], d['log_gobs'], d['R_kpc']
    if len(lg) < 10:
        continue
    # Select 10 equally-spaced indices
    indices = np.round(np.linspace(0, len(lg) - 1, 10)).astype(int)
    lg_sub = lg[indices]
    lo_sub = lo[indices]
    resid = lo_sub - rar_function(lg_sub)
    valid = np.isfinite(resid)
    if np.sum(valid) >= 5:
        acf = lag_autocorrelation(resid[valid])
        if np.isfinite(acf):
            sparc_acf_10pt.append(acf)
sparc_acf_10pt = np.array(sparc_acf_10pt)
print(f"  SPARC ACF (10pt subsample): N={len(sparc_acf_10pt)}, "
      f"mean={np.mean(sparc_acf_10pt):.3f}, median={np.median(sparc_acf_10pt):.3f}")

ks_acf_10, p_acf_10 = stats.ks_2samp(sparc_acf_10pt, eagle_acf)
print(f"  KS test (SPARC 10pt vs EAGLE matched): stat={ks_acf_10:.3f}, p={p_acf_10:.4f}")

# TNG ACF — not feasible (only 2-4 points per galaxy)
tng_acf = np.array([])
if TNG_AVAILABLE and tng_m_pergal:
    tng_acf = compute_acf_distribution(tng_m_pergal, min_pts=3)
    if len(tng_acf) > 0:
        print(f"  TNG matched ACF (cautionary — only 2-4 pts/galaxy): N={len(tng_acf)}, "
              f"mean={np.mean(tng_acf):.3f}")
    else:
        print(f"  TNG ACF: insufficient radial points (2-4 per galaxy)")

acf_results = {
    'SPARC_full': {
        'N': len(sparc_acf), 'mean': float(np.mean(sparc_acf)),
        'median': float(np.median(sparc_acf)),
        'std': float(np.std(sparc_acf)),
        'frac_positive': float(np.mean(sparc_acf > 0)),
    },
    'SPARC_10pt': {
        'N': len(sparc_acf_10pt), 'mean': float(np.mean(sparc_acf_10pt)),
        'median': float(np.median(sparc_acf_10pt)),
    },
    'EAGLE_matched': {
        'N': len(eagle_acf), 'mean': float(np.mean(eagle_acf)),
        'median': float(np.median(eagle_acf)),
        'std': float(np.std(eagle_acf)),
        'frac_positive': float(np.mean(eagle_acf > 0)),
    },
    'KS_SPARC_full_vs_EAGLE': {'stat': float(ks_acf), 'p': float(p_acf)},
    'KS_SPARC_10pt_vs_EAGLE': {'stat': float(ks_acf_10), 'p': float(p_acf_10)},
}


# ══════════════════════════════════════════════════════════════════════
#  SECTION 10: TEST 4 — SCATTER PROFILE SHAPE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [10] Test 4: Scatter Profile Shape Comparison")
print("=" * 72)

# Use common bin edges across all samples
gbar_range = [max(sparc_lgbar_all.min(), -12.5), min(sparc_lgbar_all.max(), -8.5)]
common_edges = np.arange(gbar_range[0], gbar_range[1] + BIN_WIDTH, BIN_WIDTH)
common_centers = 0.5 * (common_edges[:-1] + common_edges[1:])

def scatter_on_common_bins(log_gbar, log_gobs, edges):
    """Compute scatter on pre-defined bins."""
    resid, lg = rar_residual(log_gbar, log_gobs)
    centers, sigmas, counts = [], [], []
    for j in range(len(edges) - 1):
        mask = (lg >= edges[j]) & (lg < edges[j + 1])
        n = np.sum(mask)
        if n >= MIN_BIN_PTS:
            centers.append(0.5 * (edges[j] + edges[j + 1]))
            sigmas.append(np.std(resid[mask]))
            counts.append(n)
    return np.array(centers), np.array(sigmas), np.array(counts)


sc_sparc = scatter_on_common_bins(sparc_lgbar_all, sparc_lgobs_all, common_edges)
sc_eagle = scatter_on_common_bins(eagle_m_lgbar, eagle_m_lgobs, common_edges)

if TNG_AVAILABLE and len(tng_m_lgbar) > 50:
    sc_tng = scatter_on_common_bins(tng_m_lgbar, tng_m_lgobs, common_edges)
else:
    sc_tng = (np.array([]), np.array([]), np.array([]))

# Shape correlation (on overlapping bins)
if len(sc_sparc[0]) > 3 and len(sc_eagle[0]) > 3:
    # Find common bin centers
    common_set = set(np.round(sc_sparc[0], 2)) & set(np.round(sc_eagle[0], 2))
    if len(common_set) >= 4:
        sparc_vals = [sc_sparc[1][i] for i, c in enumerate(np.round(sc_sparc[0], 2))
                      if c in common_set]
        eagle_vals = [sc_eagle[1][i] for i, c in enumerate(np.round(sc_eagle[0], 2))
                      if c in common_set]
        if len(sparc_vals) == len(eagle_vals):
            corr_se, p_corr_se = stats.pearsonr(sparc_vals, eagle_vals)
            print(f"  Shape correlation (SPARC vs EAGLE matched): r={corr_se:.3f}, p={p_corr_se:.4f}")
        else:
            corr_se, p_corr_se = np.nan, np.nan
    else:
        corr_se, p_corr_se = np.nan, np.nan
else:
    corr_se, p_corr_se = np.nan, np.nan

# Overall scatter comparison
sparc_overall_sigma = float(np.std(sparc_lgobs_all - rar_function(sparc_lgbar_all)))
eagle_overall_sigma = float(np.std(eagle_m_lgobs - rar_function(eagle_m_lgbar)))
print(f"  Overall scatter: SPARC={sparc_overall_sigma:.4f} dex, "
      f"EAGLE matched={eagle_overall_sigma:.4f} dex")

if TNG_AVAILABLE and len(tng_m_lgbar) > 50:
    tng_overall_sigma = float(np.std(tng_m_lgobs - rar_function(tng_m_lgbar)))
    print(f"  TNG matched overall scatter: {tng_overall_sigma:.4f} dex")
else:
    tng_overall_sigma = None


# ══════════════════════════════════════════════════════════════════════
#  SECTION 11: TEST 5 — PER-GALAXY SCATTER DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [11] Test 5: Per-Galaxy Scatter Distribution")
print("=" * 72)

def per_galaxy_scatter(per_galaxy_rar, min_pts=5):
    """Compute per-galaxy RAR scatter."""
    scatters = {}
    for gid, d in per_galaxy_rar.items():
        lg, lo = d['log_gbar'], d['log_gobs']
        if len(lg) < min_pts:
            continue
        resid = lo - rar_function(lg)
        valid = np.isfinite(resid)
        if np.sum(valid) >= min_pts:
            scatters[gid] = float(np.std(resid[valid]))
    return scatters


sparc_pgscatter = per_galaxy_scatter(sparc_rar, min_pts=10)
eagle_pgscatter = per_galaxy_scatter(eagle_m_pergal, min_pts=5)

# Paired comparison: for each SPARC galaxy, compare its scatter to mean of analog scatter
paired_sparc_sigma = []
paired_eagle_sigma = []
for name in sparc_quality:
    if name not in sparc_pgscatter or name not in eagle_matches:
        continue
    sparc_sig = sparc_pgscatter[name]
    analog_sigs = [eagle_pgscatter.get(gid, np.nan)
                   for gid, _ in eagle_matches[name]]
    analog_sigs = [s for s in analog_sigs if np.isfinite(s)]
    if analog_sigs:
        paired_sparc_sigma.append(sparc_sig)
        paired_eagle_sigma.append(np.mean(analog_sigs))

paired_sparc_sigma = np.array(paired_sparc_sigma)
paired_eagle_sigma = np.array(paired_eagle_sigma)

print(f"  Paired galaxies: {len(paired_sparc_sigma)}")
print(f"  SPARC per-galaxy σ: mean={np.mean(paired_sparc_sigma):.4f}, "
      f"median={np.median(paired_sparc_sigma):.4f}")
print(f"  EAGLE analog σ: mean={np.mean(paired_eagle_sigma):.4f}, "
      f"median={np.median(paired_eagle_sigma):.4f}")

if len(paired_sparc_sigma) >= 10:
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(paired_sparc_sigma, paired_eagle_sigma)
    print(f"  Wilcoxon signed-rank: stat={wilcoxon_stat:.1f}, p={wilcoxon_p:.4f}")
else:
    wilcoxon_stat, wilcoxon_p = np.nan, np.nan


# ══════════════════════════════════════════════════════════════════════
#  SECTION 12: TEST 6 — ENVIRONMENTAL SCATTER
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [12] Test 6: Environmental Scatter")
print("=" * 72)
print("  SKIPPED — No halo mass / group membership in EAGLE JSON or TNG HDF5.")
print("  Environmental analysis done separately in test_env_scatter_definitive.py")
print("  (Result: field σ > cluster σ at 99.8% confidence in SPARC)")

test6_results = {
    'status': 'NOT_FEASIBLE',
    'reason': 'No environment metadata in available simulation catalogs',
    'reference': 'test_env_scatter_definitive.py (SPARC: 99.8% confidence)',
}


# ══════════════════════════════════════════════════════════════════════
#  SECTION 13: RESOLUTION-MATCHED COMPARISON
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [13] Resolution-Matched Comparison")
print("=" * 72)

# Subsample SPARC to 10 points per galaxy, matching EAGLE resolution
sparc_10pt_rar = {}
for name, d in sparc_rar.items():
    lg, lo, rr = d['log_gbar'], d['log_gobs'], d['R_kpc']
    if len(lg) < 10:
        continue
    indices = np.round(np.linspace(0, len(lg) - 1, 10)).astype(int)
    sparc_10pt_rar[name] = {
        'log_gbar': lg[indices],
        'log_gobs': lo[indices],
        'R_kpc': rr[indices],
    }

sparc_10pt_lgbar = np.concatenate([sparc_10pt_rar[n]['log_gbar'] for n in sparc_10pt_rar])
sparc_10pt_lgobs = np.concatenate([sparc_10pt_rar[n]['log_gobs'] for n in sparc_10pt_rar])
sparc_10pt_gids = np.concatenate([[n] * len(sparc_10pt_rar[n]['log_gbar']) for n in sparc_10pt_rar])

print(f"  SPARC 10pt subsample: {len(sparc_10pt_rar)} galaxies, "
      f"{len(sparc_10pt_lgbar)} total points")

# Scatter inversion on 10pt SPARC
c_sparc10, s_sparc10, n_sparc10 = scatter_profile(sparc_10pt_lgbar, sparc_10pt_lgobs)
inv_sparc10 = find_inversion(c_sparc10, s_sparc10)
print(f"  SPARC 10pt inversion: {inv_sparc10:.3f}" if inv_sparc10 else
      "  SPARC 10pt inversion: not found")

# Overall scatter
sparc_10pt_sigma = float(np.std(sparc_10pt_lgobs - rar_function(sparc_10pt_lgbar)))
print(f"  SPARC 10pt overall σ: {sparc_10pt_sigma:.4f} dex")

# Scatter profile comparison
sc_sparc10 = scatter_on_common_bins(sparc_10pt_lgbar, sparc_10pt_lgobs, common_edges)

resolution_results = {
    'sparc_full_inversion': inv_sparc,
    'sparc_10pt_inversion': inv_sparc10,
    'eagle_matched_inversion': inv_eagle_m,
    'sparc_full_sigma': sparc_overall_sigma,
    'sparc_10pt_sigma': sparc_10pt_sigma,
    'eagle_matched_sigma': eagle_overall_sigma,
    'sparc_full_acf_mean': float(np.mean(sparc_acf)),
    'sparc_10pt_acf_mean': float(np.mean(sparc_acf_10pt)),
    'eagle_matched_acf_mean': float(np.mean(eagle_acf)),
}

print(f"\n  Resolution comparison:")
print(f"  {'Metric':<25} {'SPARC full':>12} {'SPARC 10pt':>12} {'EAGLE match':>12}")
print(f"  {'-'*65}")
print(f"  {'Inversion log g':<25} {inv_sparc if inv_sparc else 'N/A':>12} "
      f"{inv_sparc10 if inv_sparc10 else 'N/A':>12} "
      f"{inv_eagle_m if inv_eagle_m else 'N/A':>12}")
print(f"  {'Overall σ (dex)':<25} {sparc_overall_sigma:>12.4f} "
      f"{sparc_10pt_sigma:>12.4f} {eagle_overall_sigma:>12.4f}")
print(f"  {'ACF mean':<25} {np.mean(sparc_acf):>12.3f} "
      f"{np.mean(sparc_acf_10pt):>12.3f} {np.mean(eagle_acf):>12.3f}")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 14: FEATURE SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [14] Feature Summary Table")
print("=" * 72)

def same_check(val_sparc, val_sim, threshold):
    """Check if values are within threshold."""
    if val_sparc is None or val_sim is None:
        return "N/A"
    if abs(val_sparc - val_sim) <= threshold:
        return "YES"
    return "NO"

features = []

# Feature 1: Inversion exists?
features.append({
    'feature': 'Inversion exists',
    'SPARC': 'Yes' if inv_sparc else 'No',
    'EAGLE_matched': 'Yes' if inv_eagle_m else 'No',
    'TNG_matched': ('Yes' if inv_tng_m else 'No') if TNG_AVAILABLE else 'N/A',
    'same': 'YES' if (inv_sparc is not None) == (inv_eagle_m is not None) else 'NO',
})

# Feature 2: Inversion location
features.append({
    'feature': 'Inversion location',
    'SPARC': f"{inv_sparc:.3f}" if inv_sparc else "N/A",
    'EAGLE_matched': f"{inv_eagle_m:.3f}" if inv_eagle_m else "N/A",
    'TNG_matched': (f"{inv_tng_m:.3f}" if inv_tng_m else "N/A") if TNG_AVAILABLE else "N/A",
    'same': same_check(inv_sparc, inv_eagle_m, 0.3),
})

# Feature 3: Overall scatter
features.append({
    'feature': 'Overall σ (dex)',
    'SPARC': f"{sparc_overall_sigma:.4f}",
    'EAGLE_matched': f"{eagle_overall_sigma:.4f}",
    'TNG_matched': f"{tng_overall_sigma:.4f}" if tng_overall_sigma else "N/A",
    'same': same_check(sparc_overall_sigma, eagle_overall_sigma, 0.03),
})

# Feature 4: ACF mean
features.append({
    'feature': 'ACF mean',
    'SPARC': f"{np.mean(sparc_acf):.3f}",
    'EAGLE_matched': f"{np.mean(eagle_acf):.3f}",
    'TNG_matched': f"{np.mean(tng_acf):.3f}" if len(tng_acf) > 0 else "N/A (2-4 pts)",
    'same': same_check(np.mean(sparc_acf), np.mean(eagle_acf), 0.10),
})

# Feature 5: ACF KS p-value
features.append({
    'feature': 'ACF KS p-value',
    'SPARC': f"ref",
    'EAGLE_matched': f"{p_acf:.4f}",
    'TNG_matched': "N/A",
    'same': "YES" if p_acf > 0.05 else "NO",
})

# Feature 6: Per-galaxy scatter
features.append({
    'feature': 'Per-galaxy σ (Wilcoxon p)',
    'SPARC': f"{np.mean(paired_sparc_sigma):.4f}",
    'EAGLE_matched': f"{np.mean(paired_eagle_sigma):.4f}",
    'TNG_matched': "N/A",
    'same': "YES" if wilcoxon_p > 0.05 else "NO",
})

# Feature 7: Shape correlation
features.append({
    'feature': 'Scatter shape r',
    'SPARC': "ref",
    'EAGLE_matched': f"{corr_se:.3f}" if np.isfinite(corr_se) else "N/A",
    'TNG_matched': "N/A",
    'same': "YES" if (np.isfinite(corr_se) and corr_se > 0.7) else "NO",
})

print(f"\n  {'Feature':<28} {'SPARC':>12} {'EAGLE-match':>12} {'TNG-match':>12} {'Same?':>6}")
print(f"  {'-'*76}")
for feat in features:
    print(f"  {feat['feature']:<28} {feat['SPARC']:>12} {feat['EAGLE_matched']:>12} "
          f"{feat['TNG_matched']:>12} {feat['same']:>6}")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 15: FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [15] Generating figures...")

# --- Figure 1: Match Quality (4 panels) ---
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))

# Panel a: log(M_star) distributions
ax = axes1[0, 0]
bins_m = np.linspace(min(sparc_log_Mstar.min(), eagle_matched_lm.min()) - 0.3,
                      max(sparc_log_Mstar.max(), eagle_matched_lm.max()) + 0.3, 25)
ax.hist(sparc_log_Mstar, bins=bins_m, alpha=0.6, color='#2196F3', label='SPARC', density=True)
ax.hist(eagle_matched_lm, bins=bins_m, alpha=0.5, color='#F44336', label='EAGLE matched', density=True)
if TNG_AVAILABLE and len(tng_matched_lm) > 0:
    ax.hist(tng_matched_lm, bins=bins_m, alpha=0.4, color='#FF9800', label='TNG matched', density=True)
ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
ax.set_ylabel('Density')
ax.set_title(f'(a) Stellar Mass (KS p={p_m:.3f})')
ax.legend(fontsize=8)

# Panel b: log(R_half) distributions
ax = axes1[0, 1]
valid_sparc_lr = sparc_log_Rhalf[np.isfinite(sparc_log_Rhalf)]
bins_r = np.linspace(min(valid_sparc_lr.min(), eagle_matched_lr.min()) - 0.3,
                      max(valid_sparc_lr.max(), eagle_matched_lr.max()) + 0.3, 25)
ax.hist(valid_sparc_lr, bins=bins_r, alpha=0.6, color='#2196F3', label='SPARC (1.678×Rdisk)', density=True)
ax.hist(eagle_matched_lr, bins=bins_r, alpha=0.5, color='#F44336', label='EAGLE R_half', density=True)
if TNG_AVAILABLE and len(tng_matched_lr) > 0:
    ax.hist(tng_matched_lr, bins=bins_r, alpha=0.4, color='#FF9800', label='TNG R_half', density=True)
ax.set_xlabel(r'$\log_{10}(R_{\rm half} / {\rm kpc})$')
ax.set_ylabel('Density')
ax.set_title(f'(b) Size (KS p={p_r:.3f})')
ax.legend(fontsize=8)

# Panel c: Vflat distributions
ax = axes1[1, 0]
bins_v = np.linspace(0, max(sparc_vf_valid.max(), eagle_matched_vf.max()) + 20, 25)
ax.hist(sparc_vf_valid, bins=bins_v, alpha=0.6, color='#2196F3', label='SPARC', density=True)
ax.hist(eagle_matched_vf, bins=bins_v, alpha=0.5, color='#F44336', label='EAGLE matched', density=True)
ax.set_xlabel(r'$V_{\rm flat}$ (km/s)')
ax.set_ylabel('Density')
ax.set_title(f'(c) Flat Velocity (KS p={p_v:.3f})')
ax.legend(fontsize=8)

# Panel d: Mass-size relation
ax = axes1[1, 1]
ax.scatter(sparc_log_Mstar, sparc_log_Rhalf, s=20, alpha=0.7,
           color='#2196F3', label='SPARC', zorder=3)
ax.scatter(eagle_matched_lm, eagle_matched_lr, s=5, alpha=0.15,
           color='#F44336', label='EAGLE matched', zorder=2)
if TNG_AVAILABLE and len(tng_matched_lm) > 0:
    ax.scatter(tng_matched_lm, tng_matched_lr, s=5, alpha=0.1,
               color='#FF9800', label='TNG matched', zorder=1)
ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
ax.set_ylabel(r'$\log_{10}(R_{\rm half} / {\rm kpc})$')
ax.set_title('(d) Mass-Size Relation')
ax.legend(fontsize=8, loc='upper left')

fig1.suptitle('Match Quality: SPARC vs Simulation Analogs', fontsize=14, y=1.02)
fig1.tight_layout()
fig1.savefig(os.path.join(FIGURES_DIR, 'matched_analog_match_quality.png'))
plt.close(fig1)
print(f"  Figure 1 saved: matched_analog_match_quality.png")


# --- Figure 2: RAR Comparison (3 panels) ---
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

rar_line_x = np.linspace(-13, -8, 200)
rar_line_y = rar_function(rar_line_x)

for ax, lgbar, lgobs, title in [
    (axes2[0], sparc_lgbar_all, sparc_lgobs_all, f'SPARC (N={len(sparc_rar)})'),
    (axes2[1], eagle_m_lgbar, eagle_m_lgobs, f'EAGLE matched (N={len(eagle_m_pergal)})'),
    (axes2[2], tng_m_lgbar, tng_m_lgobs,
     f'TNG matched (N={len(tng_m_pergal)})' if TNG_AVAILABLE else 'TNG (no data)'),
]:
    if len(lgbar) > 0:
        ax.hexbin(lgbar, lgobs, gridsize=40, cmap='viridis', mincnt=1)
        ax.plot(rar_line_x, rar_line_y, 'r-', lw=1.5, alpha=0.8, label='RAR fit')
        ax.plot(rar_line_x, rar_line_x, 'k--', lw=0.8, alpha=0.5, label='1:1')
        ax.axvline(LOG_GD, color='orange', ls=':', lw=1, alpha=0.7, label=r'$g^\dagger$')
    ax.set_xlim(-13, -8.5)
    ax.set_ylim(-13, -8.5)
    ax.set_xlabel(r'$\log_{10}(g_{\rm bar})$ [m/s²]')
    ax.set_ylabel(r'$\log_{10}(g_{\rm obs})$ [m/s²]')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='upper left')

fig2.suptitle('Radial Acceleration Relation — Matched Samples', fontsize=14, y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES_DIR, 'matched_analog_rar_comparison.png'))
plt.close(fig2)
print(f"  Figure 2 saved: matched_analog_rar_comparison.png")


# --- Figure 3: Scatter Profile Comparison (key figure) ---
fig3, ax3 = plt.subplots(figsize=(10, 7))

# SPARC with bootstrap CI
if bc_sparc is not None:
    ax3.plot(bc_sparc, bs_sparc, 'o-', color='#2196F3', lw=2.5,
             label=f'SPARC (N={len(sparc_rar)})', zorder=5)
    if bci_sparc is not None:
        ax3.fill_between(bc_sparc, bci_sparc[0], bci_sparc[1],
                         alpha=0.15, color='#2196F3')

# EAGLE matched with bootstrap CI
if bc_eagle is not None:
    ax3.plot(bc_eagle, bs_eagle, 's-', color='#F44336', lw=2,
             label=f'EAGLE matched (N={len(eagle_m_pergal)})', zorder=4)
    if bci_eagle is not None:
        ax3.fill_between(bc_eagle, bci_eagle[0], bci_eagle[1],
                         alpha=0.12, color='#F44336')

# TNG matched
if TNG_AVAILABLE and c_tng_m is not None:
    ax3.plot(c_tng_m, s_tng_m, '^-', color='#FF9800', lw=2,
             label=f'TNG matched (N={len(tng_m_pergal)})', zorder=3)
    if bci_tng is not None:
        ax3.fill_between(bc_tng, bci_tng[0], bci_tng[1],
                         alpha=0.12, color='#FF9800')

# SPARC 10pt subsample
if c_sparc10 is not None and len(c_sparc10) > 0:
    ax3.plot(sc_sparc10[0], sc_sparc10[1], 'o--', color='#64B5F6', lw=1.5,
             label=f'SPARC 10pt subsample', zorder=2, alpha=0.7)

# g† line
ax3.axvline(LOG_GD, color='green', ls=':', lw=2, alpha=0.7, label=r'$g^\dagger$')

# Mark inversions
if inv_sparc is not None:
    ax3.axvline(inv_sparc, color='#2196F3', ls='--', lw=1, alpha=0.5)
    ax3.annotate(f'SPARC inv\n{inv_sparc:.2f}', xy=(inv_sparc, ax3.get_ylim()[1]),
                 fontsize=8, color='#2196F3', ha='center', va='top')
if inv_eagle_m is not None:
    ax3.axvline(inv_eagle_m, color='#F44336', ls='--', lw=1, alpha=0.5)

ax3.set_xlabel(r'$\log_{10}(g_{\rm bar})$ [m/s²]', fontsize=14)
ax3.set_ylabel(r'$\sigma(\log_{10}\,g_{\rm bar})$ [dex]', fontsize=14)
ax3.set_title('Scatter Profile: SPARC vs Matched Simulation Analogs', fontsize=14)
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(os.path.join(FIGURES_DIR, 'matched_analog_scatter_profiles.png'))
plt.close(fig3)
print(f"  Figure 3 saved: matched_analog_scatter_profiles.png")


# --- Figure 4: ACF Comparison ---
fig4, ax4 = plt.subplots(figsize=(8, 6))

acf_data = [sparc_acf, sparc_acf_10pt, eagle_acf]
acf_labels = ['SPARC\n(full res)', 'SPARC\n(10pt)', 'EAGLE\nmatched']
acf_colors = ['#2196F3', '#64B5F6', '#F44336']

if len(tng_acf) > 5:
    acf_data.append(tng_acf)
    acf_labels.append('TNG matched\n(2-4 pts)')
    acf_colors.append('#FF9800')

bp = ax4.boxplot(acf_data, labels=acf_labels, patch_artist=True,
                  widths=0.5, showmeans=True,
                  meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markersize': 6})
for patch, color in zip(bp['boxes'], acf_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

ax4.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
ax4.set_ylabel('Lag-1 ACF (demeaned)', fontsize=13)
ax4.set_title('Radial Coherence: ACF Distribution Comparison', fontsize=14)

# Annotate KS p-values
ax4.annotate(f'KS(SPARC full vs EAGLE): p={p_acf:.4f}',
             xy=(0.98, 0.95), xycoords='axes fraction',
             fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
ax4.annotate(f'KS(SPARC 10pt vs EAGLE): p={p_acf_10:.4f}',
             xy=(0.98, 0.88), xycoords='axes fraction',
             fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax4.grid(True, alpha=0.3, axis='y')
fig4.tight_layout()
fig4.savefig(os.path.join(FIGURES_DIR, 'matched_analog_acf_comparison.png'))
plt.close(fig4)
print(f"  Figure 4 saved: matched_analog_acf_comparison.png")


# --- Figure 5: Feature Summary ---
fig5, ax5 = plt.subplots(figsize=(12, 5))

n_feat = len(features)
y_pos = np.arange(n_feat)[::-1]
colors_map = {'YES': '#4CAF50', 'NO': '#F44336', 'N/A': '#9E9E9E'}

for i, feat in enumerate(features):
    c = colors_map.get(feat['same'], '#9E9E9E')
    ax5.barh(y_pos[i], 1, color=c, alpha=0.7, height=0.6)
    ax5.text(-0.02, y_pos[i], feat['feature'], ha='right', va='center', fontsize=10)
    ax5.text(0.15, y_pos[i] + 0.12, f"SPARC: {feat['SPARC']}", fontsize=8, va='bottom')
    ax5.text(0.15, y_pos[i] - 0.12, f"EAGLE: {feat['EAGLE_matched']}", fontsize=8, va='top',
             color='#D32F2F')
    if feat['TNG_matched'] != 'N/A':
        ax5.text(0.65, y_pos[i], f"TNG: {feat['TNG_matched']}", fontsize=8, va='center',
                 color='#E65100')
    ax5.text(1.05, y_pos[i], feat['same'], ha='left', va='center', fontsize=10,
             fontweight='bold', color=c)

ax5.set_xlim(-0.5, 1.3)
ax5.set_ylim(-0.5, n_feat - 0.5)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_title('Feature Comparison Scorecard: SPARC vs Matched Analogs', fontsize=14)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', alpha=0.7, label='Match'),
                   Patch(facecolor='#F44336', alpha=0.7, label='Mismatch'),
                   Patch(facecolor='#9E9E9E', alpha=0.7, label='Not testable')]
ax5.legend(handles=legend_elements, loc='lower right', fontsize=9)

fig5.tight_layout()
fig5.savefig(os.path.join(FIGURES_DIR, 'matched_analog_feature_summary.png'))
plt.close(fig5)
print(f"  Figure 5 saved: matched_analog_feature_summary.png")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 16: SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} [16] Saving results...")

# Summary JSON
summary = {
    'description': 'SPARC-matched analog analysis in EAGLE & TNG',
    'date': '2026-02-21',
    'parameters': {
        'n_analogs': N_ANALOGS,
        'matching_metric': 'Euclidean in (log M_star / σ_M, log R_half / σ_R)',
        'sigma_Mstar': float(sigma_Mstar),
        'sigma_R': float(sigma_R),
        'max_match_distance': MAX_MATCH_DIST,
        'bin_width_dex': BIN_WIDTH,
        'n_bootstrap': N_BOOT,
        'rdisk_to_rhalf_factor': R_HALF_FACTOR,
        'mass_to_light_disk': Y_DISK,
    },
    'sample_sizes': {
        'SPARC_quality': len(sparc_quality),
        'SPARC_with_RAR': len(sparc_rar),
        'EAGLE_total': len(eagle_props),
        'EAGLE_matched_galaxies': len(eagle_matches),
        'EAGLE_matched_unique_analogs': n_eagle_unique,
        'EAGLE_total_RAR_points': int(len(eagle_m_lgbar)),
        'TNG_available': TNG_AVAILABLE,
        'TNG_total': len(tng_props) if TNG_AVAILABLE else 0,
        'TNG_matched_galaxies': len(tng_matches),
        'TNG_total_RAR_points': int(len(tng_m_lgbar)),
    },
    'match_quality': match_quality,
    'test1_scatter_inversion': {
        'SPARC_inversion': inv_sparc,
        'EAGLE_matched_inversion': inv_eagle_m,
        'TNG_matched_inversion': inv_tng_m,
        'SPARC_overall_sigma': sparc_overall_sigma,
        'EAGLE_matched_overall_sigma': eagle_overall_sigma,
        'TNG_matched_overall_sigma': tng_overall_sigma,
        'robustness_offsets': {k: [float(v) for v in vals] for k, vals in inv_offsets.items()},
    },
    'test2_inversion_proximity': test2_results,
    'test3_acf': acf_results,
    'test4_scatter_shape': {
        'shape_correlation_SPARC_vs_EAGLE': float(corr_se) if np.isfinite(corr_se) else None,
        'shape_correlation_p': float(p_corr_se) if np.isfinite(p_corr_se) else None,
    },
    'test5_per_galaxy_scatter': {
        'n_paired': len(paired_sparc_sigma),
        'SPARC_mean_sigma': float(np.mean(paired_sparc_sigma)),
        'EAGLE_mean_sigma': float(np.mean(paired_eagle_sigma)),
        'wilcoxon_stat': float(wilcoxon_stat) if np.isfinite(wilcoxon_stat) else None,
        'wilcoxon_p': float(wilcoxon_p) if np.isfinite(wilcoxon_p) else None,
    },
    'test6_environmental': test6_results,
    'resolution_matched': resolution_results,
    'feature_summary': features,
    'limitations': [
        'TNG only 4 apertures: no ACF/periodicity tests possible',
        'TNG g_bar stellar-only (no gas mass in HDF5)',
        'EAGLE R_half derived from 10-aperture interpolation',
        'R_disk to R_half conversion assumes pure exponential disk',
        'No environment metadata in simulation catalogs',
    ],
}

json_path = os.path.join(RESULTS_DIR, 'summary_matched_analog_comparison.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  Saved: {json_path}")

# EAGLE analog CSV
csv_lines = ['sparc_galaxy,analog_rank,eagle_galaxy_id,match_distance,eagle_log_Mstar,eagle_log_Rhalf,eagle_Vflat']
for name in sorted(eagle_matches.keys()):
    for rank, (gid, dist) in enumerate(eagle_matches[name]):
        ep = eagle_props[gid]
        csv_lines.append(f"{name},{rank+1},{gid},{dist:.4f},"
                         f"{ep['log_Mstar']:.3f},{ep['log_Rhalf']:.3f},{ep['Vflat']:.1f}")

eagle_csv_path = os.path.join(RESULTS_DIR, 'eagle_matched_analogs.csv')
with open(eagle_csv_path, 'w') as f:
    f.write('\n'.join(csv_lines))
print(f"  Saved: {eagle_csv_path}")

# TNG analog CSV
if TNG_AVAILABLE and tng_matches:
    csv_lines_tng = ['sparc_galaxy,analog_rank,tng_subfind_id,match_distance,tng_log_Mstar,tng_log_Rhalf']
    for name in sorted(tng_matches.keys()):
        for rank, (gid, dist) in enumerate(tng_matches[name]):
            tp = tng_props[gid]
            csv_lines_tng.append(f"{name},{rank+1},{gid},{dist:.4f},"
                                 f"{tp['log_Mstar']:.3f},{tp['log_Rhalf']:.3f}")

    tng_csv_path = os.path.join(RESULTS_DIR, 'tng_matched_analogs.csv')
    with open(tng_csv_path, 'w') as f:
        f.write('\n'.join(csv_lines_tng))
    print(f"  Saved: {tng_csv_path}")


# ══════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{elapsed()} " + "=" * 72)
print("ANALYSIS COMPLETE")
print("=" * 72)

n_match = sum(1 for f in features if f['same'] == 'YES')
n_mismatch = sum(1 for f in features if f['same'] == 'NO')
n_na = sum(1 for f in features if f['same'] == 'N/A')

print(f"\n  Feature scorecard: {n_match} match, {n_mismatch} mismatch, {n_na} N/A")
print(f"\n  Key findings:")
print(f"    SPARC inversion: {inv_sparc}")
print(f"    EAGLE matched inversion: {inv_eagle_m}")
if TNG_AVAILABLE:
    print(f"    TNG matched inversion: {inv_tng_m}")
print(f"    SPARC overall σ: {sparc_overall_sigma:.4f} dex")
print(f"    EAGLE matched σ: {eagle_overall_sigma:.4f} dex")
print(f"    SPARC ACF mean: {np.mean(sparc_acf):.3f}")
print(f"    EAGLE matched ACF mean: {np.mean(eagle_acf):.3f}")
print(f"    ACF KS p-value (SPARC full vs EAGLE): {p_acf:.4f}")
print(f"    ACF KS p-value (SPARC 10pt vs EAGLE): {p_acf_10:.4f}")

print(f"\n  Interpretation:")
if n_mismatch == 0:
    print("    All testable features match → selection effects explain differences")
    print("    Supports: BEC statistics may emerge from classical gravitational physics")
elif n_mismatch >= 3:
    print("    Multiple features differ → SPARC has properties ΛCDM doesn't reproduce")
    print("    Supports: real quantum physics signature in observations")
else:
    print("    Mixed results — some features match, some differ")
    print("    Further investigation needed with higher-resolution sim data")

print(f"\n  Output files:")
print(f"    {json_path}")
print(f"    {eagle_csv_path}")
if TNG_AVAILABLE and tng_matches:
    print(f"    {tng_csv_path}")
print(f"    figures/matched_analog_match_quality.png")
print(f"    figures/matched_analog_rar_comparison.png")
print(f"    figures/matched_analog_scatter_profiles.png")
print(f"    figures/matched_analog_acf_comparison.png")
print(f"    figures/matched_analog_feature_summary.png")

print(f"\n{elapsed()} Done.")
