#!/usr/bin/env python3
"""
Void-Field-Dense RAR Scatter Gradient Test
===========================================

Three-tier environmental test: does RAR scatter differ across
void → field → dense environments?

BEC prediction:
  - Void galaxies: maximal isolation, pristine condensate → LOWEST scatter
  - Field galaxies: moderate environment → intermediate scatter
  - Dense galaxies: cluster/group tidal disruption → HIGHEST scatter

Or alternatively (if BEC coupling is truly universal):
  - All three environments show IDENTICAL scatter

This test extends test_env_scatter_definitive.py by adding cosmic void
classification using the VAST VoidFinder catalog (Douglass et al. 2023).

Method:
  1. Load SPARC galaxy coordinates (RA, Dec) and distances (Mpc)
  2. Convert to comoving Cartesian coordinates (Mpc/h, Planck 2018)
  3. Cross-match against VoidFinder hole spheres for void membership
  4. Combine with existing Kourkchi/NED group classification for dense
  5. Run 7-test statistical battery on three-tier split

VoidFinder catalog: Douglass et al. 2023 (VAST, Zenodo 11043278)
  - SDSS DR7 volume-limited sample, z ≤ 0.114
  - Coordinates: comoving Mpc/h (Planck 2018: h=0.6736, Ωm=0.3153)
  - 1163 unique voids, 39735 hole spheres
  - Survey footprint: ~8200 deg², Dec -10° to +70°, mostly RA 100-260°
"""

import os
import sys
import json
import numpy as np
from scipy.stats import levene, ks_2samp, mannwhitneyu
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
VOID_DIR = os.path.join(PROJECT_ROOT, 'data', 'voids')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Planck 2018 cosmology (matching VoidFinder catalog)
H0 = 67.36       # km/s/Mpc
h = H0 / 100.0   # 0.6736
OMEGA_M = 0.3153
OMEGA_L = 1 - OMEGA_M
c_km = 299792.458  # km/s

# Physics
g_dagger = 1.20e-10
kpc_m = 3.086e19


def rar_function(log_gbar, a0=1.2e-10):
    """Standard RAR/BEC prediction."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def haubner_delta(D_Mpc, delta_inf=0.022, alpha=-0.8, D_tr=46.0, kappa=1.8):
    """Haubner+2025 Eq 7: distance uncertainty in dex."""
    D = max(D_Mpc, 0.01)
    return delta_inf * D**alpha * (D**(1/kappa) + D_tr**(1/kappa))**(-alpha * kappa)


def distance_to_comoving(D_Mpc, h=0.6736):
    """
    Convert luminosity/physical distance (Mpc) to comoving distance (Mpc/h).

    For z << 1 (all SPARC galaxies), comoving distance ≈ physical distance.
    The h factor converts Mpc to Mpc/h.

    More precisely: d_comoving = d_physical * (1+z), but for z < 0.03
    this correction is < 3%, well within SPARC distance uncertainties.
    """
    return D_Mpc * h


def radec_dist_to_xyz(ra_deg, dec_deg, d_comoving_mpc_h):
    """
    Convert (RA, Dec, comoving distance) to Cartesian (x, y, z) in Mpc/h.

    Uses the standard SDSS convention:
      x = -d * cos(dec) * sin(ra)      [toward RA=270°]
      y =  d * cos(dec) * cos(ra)      [toward RA=0°]
      z =  d * sin(dec)                [toward Dec=90°]

    Note: VoidFinder uses the SDSS survey coordinate convention.
    We need to match this exactly for proper containment checks.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    d = d_comoving_mpc_h

    # Standard astronomical convention
    x = d * np.cos(dec) * np.cos(ra)
    y = d * np.cos(dec) * np.sin(ra)
    z = d * np.sin(dec)

    return x, y, z


def check_sdss_footprint(ra_deg, dec_deg):
    """
    Rough check if a position falls within the SDSS DR7 spectroscopic footprint.

    The SDSS DR7 main contiguous area covers approximately:
      - Dec: -10° to +70° (main stripe)
      - RA: roughly 100° to 260° (with gaps and some southern stripes)

    This is a conservative approximation. We flag galaxies outside this
    region as 'outside_survey' rather than classifying them.
    """
    # Basic declination cut
    if dec_deg < -10 or dec_deg > 70:
        return False

    # Main contiguous region: RA ~100-260° (Northern Galactic Cap)
    # Plus three southern stripes around RA ~310-60° at specific Dec ranges
    # We use a generous cut and flag marginal cases

    # Northern Galactic Cap (main area)
    if 100 < ra_deg < 260 and -5 < dec_deg < 70:
        return True

    # Some coverage at high Dec, wide RA range
    if dec_deg > 55:
        return True

    # Southern Galactic stripes (limited coverage)
    if -10 < dec_deg < 5:
        # Stripe 82 region and nearby
        if 310 < ra_deg or ra_deg < 60:
            return True

    return False


# ============================================================
# Environment classification (from test_env_scatter_definitive.py)
# ============================================================
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

PRIMARY_ERRORS = {'TRGB': 0.05, 'Cepheid': 0.05, 'SBF': 0.05,
                  'SNe': 0.07, 'maser': 0.10}


def classify_dense(name):
    """Classify as dense (group/cluster member) based on known associations."""
    if name in UMA_GALAXIES:
        return True, 'UMa'
    if name in GROUP_MEMBERS:
        return True, GROUP_MEMBERS[name]
    return False, None


# ================================================================
# MAIN PIPELINE
# ================================================================
print("=" * 72)
print("VOID-FIELD-DENSE RAR SCATTER GRADIENT TEST")
print("=" * 72)

# ================================================================
# STEP 1: Load VoidFinder hole spheres
# ================================================================
print("\n[1] Loading VoidFinder catalog...")

holes_path = os.path.join(VOID_DIR, 'VoidFinder_Planck2018_holes.txt')
if not os.path.exists(holes_path):
    print(f"  ERROR: VoidFinder catalog not found at {holes_path}")
    print(f"  Download from: https://zenodo.org/records/11043278")
    sys.exit(1)

holes = np.loadtxt(holes_path, skiprows=1)
hole_xyz = holes[:, :3]   # x, y, z in comoving Mpc/h
hole_r = holes[:, 3]      # radius in comoving Mpc/h
hole_vid = holes[:, 4].astype(int)  # void ID

n_holes = len(holes)
n_voids = len(np.unique(hole_vid))
print(f"  Loaded {n_holes} hole spheres from {n_voids} unique voids")
print(f"  Radius range: {hole_r.min():.1f} - {hole_r.max():.1f} Mpc/h")
print(f"  Radius mean: {hole_r.mean():.1f} Mpc/h")

# Build KD-tree for efficient nearest-sphere lookup
# We use center positions for tree, then check radius containment
hole_tree = cKDTree(hole_xyz)

# Pre-compute maximum hole radius for search cutoff
max_hole_radius = hole_r.max()


def is_in_void(x, y, z, margin=0.0):
    """
    Check if position (x,y,z) in Mpc/h falls inside any VoidFinder hole sphere.

    Uses KD-tree for efficient lookup: query all holes within max_radius,
    then check exact containment for each candidate.

    Args:
        x, y, z: Cartesian comoving coordinates in Mpc/h
        margin: shrink the void radius by this factor (0 = full sphere,
                0.1 = require being inside 90% of radius)

    Returns:
        (in_void: bool, void_id: int or None, n_containing: int,
         min_distance_to_center: float)
    """
    point = np.array([x, y, z])

    # Query all hole centers within max_hole_radius
    candidates = hole_tree.query_ball_point(point, max_hole_radius)

    if not candidates:
        return False, None, 0, np.inf

    # Check exact containment for each candidate
    containing_voids = []
    min_dist = np.inf

    for idx in candidates:
        dist = np.sqrt(np.sum((point - hole_xyz[idx])**2))
        effective_r = hole_r[idx] * (1 - margin)

        if dist < effective_r:
            containing_voids.append((hole_vid[idx], dist, hole_r[idx]))

        if dist < min_dist:
            min_dist = dist

    if containing_voids:
        # Return the void with the largest sphere containing this point
        best = max(containing_voids, key=lambda v: v[2])
        return True, best[0], len(containing_voids), min_dist

    return False, None, 0, min_dist


# ================================================================
# STEP 2: Load SPARC data (same as test_env_scatter_definitive.py)
# ================================================================
print("\n[2] Loading SPARC data...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')
coords_path = os.path.join(DATA_DIR, 'sparc_coordinates.json')

# Load coordinates
with open(coords_path, 'r') as f:
    sparc_coords = json.load(f)

# Load rotation curves
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

# Parse MRT for properties
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
            'L36': float(parts[6]), 'Reff': float(parts[8]),
            'SBeff': float(parts[9]), 'MHI': float(parts[12]),
            'Vflat': float(parts[14]), 'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

print(f"  {len(galaxies)} rotation curves, {len(sparc_props)} with properties")
print(f"  {len(sparc_coords)} with RA/Dec coordinates")

# ================================================================
# STEP 3: Cross-match SPARC with void catalog
# ================================================================
print("\n[3] Cross-matching SPARC galaxies with VoidFinder catalog...")

void_classification = {}
n_in_void = 0
n_in_field = 0
n_outside_survey = 0
n_too_close = 0
n_no_coords = 0

# VoidFinder survey volume limits
# The catalog is volume-limited to z=0.114 → d_comoving ≈ 332 Mpc/h
# Minimum distance is set by the closest void spheres (~24 Mpc/h ≈ 36 Mpc)
D_MIN_MPH = 20.0    # Mpc/h — below this, no void classification possible
D_MAX_MPH = 335.0   # Mpc/h — above this, beyond survey volume

print(f"  Survey volume: {D_MIN_MPH:.0f} - {D_MAX_MPH:.0f} Mpc/h")
print(f"                 ({D_MIN_MPH/h:.0f} - {D_MAX_MPH/h:.0f} Mpc)")

for name in galaxies:
    if name not in sparc_coords:
        void_classification[name] = {
            'env_void': 'no_coords',
            'reason': 'no RA/Dec available'
        }
        n_no_coords += 1
        continue

    if name not in sparc_props:
        continue

    ra = sparc_coords[name]['ra']
    dec = sparc_coords[name]['dec']
    D_Mpc = sparc_props[name]['D']

    # Convert to comoving Mpc/h
    d_com = distance_to_comoving(D_Mpc, h=h)

    # Check if within survey volume
    if d_com < D_MIN_MPH:
        void_classification[name] = {
            'env_void': 'too_close',
            'ra': ra, 'dec': dec, 'D_Mpc': D_Mpc, 'd_com_mph': d_com,
            'reason': f'd_com={d_com:.1f} Mpc/h < {D_MIN_MPH} Mpc/h minimum'
        }
        n_too_close += 1
        continue

    if d_com > D_MAX_MPH:
        void_classification[name] = {
            'env_void': 'beyond_survey',
            'ra': ra, 'dec': dec, 'D_Mpc': D_Mpc, 'd_com_mph': d_com,
            'reason': f'd_com={d_com:.1f} Mpc/h > {D_MAX_MPH} Mpc/h maximum'
        }
        n_outside_survey += 1
        continue

    # Check SDSS footprint
    if not check_sdss_footprint(ra, dec):
        void_classification[name] = {
            'env_void': 'outside_footprint',
            'ra': ra, 'dec': dec, 'D_Mpc': D_Mpc, 'd_com_mph': d_com,
            'reason': f'RA={ra:.1f}, Dec={dec:.1f} outside SDSS DR7 footprint'
        }
        n_outside_survey += 1
        continue

    # Convert to Cartesian
    x, y, z = radec_dist_to_xyz(ra, dec, d_com)

    # Check void containment
    in_void, void_id, n_containing, min_dist = is_in_void(x, y, z)

    if in_void:
        void_classification[name] = {
            'env_void': 'void',
            'void_id': int(void_id),
            'n_containing_holes': n_containing,
            'min_dist_to_center': float(min_dist),
            'ra': ra, 'dec': dec, 'D_Mpc': D_Mpc, 'd_com_mph': d_com,
            'x': float(x), 'y': float(y), 'z': float(z),
        }
        n_in_void += 1
    else:
        void_classification[name] = {
            'env_void': 'wall',
            'min_dist_to_void_center': float(min_dist),
            'ra': ra, 'dec': dec, 'D_Mpc': D_Mpc, 'd_com_mph': d_com,
            'x': float(x), 'y': float(y), 'z': float(z),
        }
        n_in_field += 1

print(f"\n  Classification results:")
print(f"    In void:           {n_in_void}")
print(f"    In wall/filament:  {n_in_field}")
print(f"    Too close (<{D_MIN_MPH:.0f} Mpc/h): {n_too_close}")
print(f"    Outside footprint: {n_outside_survey}")
print(f"    No coordinates:    {n_no_coords}")

# ================================================================
# STEP 4: Merge void + dense classifications into three-tier
# ================================================================
print("\n[4] Building three-tier environment classification...")
print("    void → field (wall) → dense (group/cluster)")

# For galaxies TOO CLOSE for void survey:
# We can still use Kourkchi/NED group membership to classify as dense vs field.
# But we CANNOT classify them as void (we just don't know).
# Strategy: mark as 'field_unresolved' — they go into field bin but flagged.

results = {}
env_counts = {'void': 0, 'field': 0, 'dense': 0, 'unclassifiable': 0}

for name, gdata in galaxies.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]

    # Quality cuts (same as definitive test)
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

    R = gdata['R']
    Vobs = gdata['Vobs']
    Vdisk = gdata['Vdisk']
    Vgas = gdata['Vgas']
    Vbul = gdata['Vbul']

    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < 5:
        continue

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    log_gobs_rar = rar_function(log_gbar)
    log_res = log_gobs - log_gobs_rar

    # Dense classification (Kourkchi/NED — takes priority)
    is_dense, group_name = classify_dense(name)

    # Void classification
    vc = void_classification.get(name, {})
    env_void = vc.get('env_void', 'unknown')

    # Three-tier logic:
    # 1. Dense (group/cluster) — always overrides void status
    #    (a galaxy can be in a group that happens to be near a void)
    # 2. Void — from VoidFinder containment
    # 3. Field — everything else (wall/filament from VoidFinder,
    #    or unresolved galaxies too close for void survey)

    if is_dense:
        env3 = 'dense'
        env_detail = f'group:{group_name}'
    elif env_void == 'void':
        env3 = 'void'
        env_detail = f'void:{vc.get("void_id", "?")}'
    elif env_void in ('wall', 'outside_footprint', 'too_close',
                       'beyond_survey', 'no_coords', 'unknown'):
        env3 = 'field'
        env_detail = f'wall/field ({env_void})'
    else:
        env3 = 'field'
        env_detail = 'default'

    # Distance uncertainty (Haubner scheme)
    fD = prop['fD']
    D = prop['D']
    if fD == 2:
        sigma_D_dex = np.log10(1 + PRIMARY_ERRORS['TRGB'])
    elif fD == 3:
        sigma_D_dex = np.log10(1 + PRIMARY_ERRORS['Cepheid'])
    elif fD == 4:
        sigma_D_dex = np.log10(1 + 0.10)
    elif fD == 5:
        sigma_D_dex = np.log10(1 + PRIMARY_ERRORS['SNe'])
    else:
        sigma_D_dex = haubner_delta(D)

    env_counts[env3] += 1

    results[name] = {
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'log_res': log_res,
        'mean_res': float(np.mean(log_res)),
        'std_res': float(np.std(log_res)),
        'n_points': len(log_res),
        'env3': env3,
        'env_detail': env_detail,
        'env_void_raw': env_void,
        'D': D,
        'fD': fD,
        'sigma_D_dex': sigma_D_dex,
        'logMs': np.log10(max(0.5 * prop['L36'] * 1e9, 1e6)),
        'Vflat': prop['Vflat'],
        'T': prop['T'],
    }

print(f"\n  After quality cuts: {len(results)} galaxies")
print(f"    Void:  {env_counts['void']}")
print(f"    Field: {env_counts['field']}")
print(f"    Dense: {env_counts['dense']}")

# List void galaxies
void_gals = sorted([(name, r) for name, r in results.items() if r['env3'] == 'void'],
                    key=lambda x: x[1]['D'])
if void_gals:
    print(f"\n  Void galaxies ({len(void_gals)}):")
    for name, r in void_gals:
        vc = void_classification.get(name, {})
        print(f"    {name:15s}  D={r['D']:6.1f} Mpc  N_pts={r['n_points']:3d}  "
              f"void_id={vc.get('void_id', '?')}  n_holes={vc.get('n_containing_holes', '?')}")

# ================================================================
# STEP 5: Statistical tests — three-tier gradient
# ================================================================
print("\n" + "=" * 72)
print("STATISTICAL TESTS: Three-Tier Environmental Gradient")
print("=" * 72)

# Collect data by environment
void_res_pts = np.concatenate([r['log_res'] for r in results.values() if r['env3'] == 'void']) if env_counts['void'] > 0 else np.array([])
field_res_pts = np.concatenate([r['log_res'] for r in results.values() if r['env3'] == 'field'])
dense_res_pts = np.concatenate([r['log_res'] for r in results.values() if r['env3'] == 'dense'])

void_means = np.array([r['mean_res'] for r in results.values() if r['env3'] == 'void'])
field_means = np.array([r['mean_res'] for r in results.values() if r['env3'] == 'field'])
dense_means = np.array([r['mean_res'] for r in results.values() if r['env3'] == 'dense'])

void_stds = np.array([r['std_res'] for r in results.values() if r['env3'] == 'void'])
field_stds = np.array([r['std_res'] for r in results.values() if r['env3'] == 'field'])
dense_stds = np.array([r['std_res'] for r in results.values() if r['env3'] == 'dense'])


# --- TEST A: Point-level scatter ---
print("\n" + "-" * 72)
print("TEST A: Point-level RAR scatter by environment")
print("-" * 72)

sigma_void = np.std(void_res_pts) if len(void_res_pts) > 0 else np.nan
sigma_field = np.std(field_res_pts)
sigma_dense = np.std(dense_res_pts)

print(f"\n  {'Env':8s} {'N_gal':>6s} {'N_pts':>6s} {'sigma(dex)':>12s}")
print(f"  {'-'*36}")
if env_counts['void'] > 0:
    print(f"  {'Void':8s} {env_counts['void']:6d} {len(void_res_pts):6d} {sigma_void:12.4f}")
print(f"  {'Field':8s} {env_counts['field']:6d} {len(field_res_pts):6d} {sigma_field:12.4f}")
print(f"  {'Dense':8s} {env_counts['dense']:6d} {len(dense_res_pts):6d} {sigma_dense:12.4f}")

if len(void_res_pts) >= 10 and len(dense_res_pts) >= 10:
    stat_vd, p_vd = levene(void_res_pts, dense_res_pts)
    stat_vf, p_vf = levene(void_res_pts, field_res_pts)
    stat_fd, p_fd = levene(field_res_pts, dense_res_pts)
    print(f"\n  Levene pairwise:")
    print(f"    Void vs Dense:  F={stat_vd:.3f}, p={p_vd:.6f}")
    print(f"    Void vs Field:  F={stat_vf:.3f}, p={p_vf:.6f}")
    print(f"    Field vs Dense: F={stat_fd:.3f}, p={p_fd:.6f}")
elif len(void_res_pts) >= 10:
    stat_vf, p_vf = levene(void_res_pts, field_res_pts)
    stat_fd, p_fd = levene(field_res_pts, dense_res_pts)
    print(f"\n  Levene pairwise:")
    print(f"    Void vs Field:  F={stat_vf:.3f}, p={p_vf:.6f}")
    print(f"    Field vs Dense: F={stat_fd:.3f}, p={p_fd:.6f}")
else:
    stat_fd, p_fd = levene(field_res_pts, dense_res_pts)
    print(f"\n  Insufficient void galaxies for Levene test")
    print(f"    Field vs Dense: F={stat_fd:.3f}, p={p_fd:.6f}")


# --- TEST B: Galaxy-level scatter ---
print("\n" + "-" * 72)
print("TEST B: Galaxy-level mean residual scatter")
print("-" * 72)

sigma_void_gal = np.std(void_means) if len(void_means) > 0 else np.nan
sigma_field_gal = np.std(field_means)
sigma_dense_gal = np.std(dense_means)

print(f"\n  {'Env':8s} {'N_gal':>6s} {'sigma(mean_res)':>16s} {'mean(mean_res)':>16s}")
print(f"  {'-'*50}")
if len(void_means) > 0:
    print(f"  {'Void':8s} {len(void_means):6d} {sigma_void_gal:16.4f} {np.mean(void_means):+16.4f}")
print(f"  {'Field':8s} {len(field_means):6d} {sigma_field_gal:16.4f} {np.mean(field_means):+16.4f}")
print(f"  {'Dense':8s} {len(dense_means):6d} {sigma_dense_gal:16.4f} {np.mean(dense_means):+16.4f}")

if len(void_means) >= 5:
    print(f"\n  Pairwise tests:")
    for label, arr1, arr2 in [('Void vs Dense', void_means, dense_means),
                               ('Void vs Field', void_means, field_means),
                               ('Field vs Dense', field_means, dense_means)]:
        if len(arr1) >= 5 and len(arr2) >= 5:
            stat_L, p_L = levene(arr1, arr2)
            ks_stat, ks_p = ks_2samp(arr1, arr2)
            print(f"    {label:15s}  Levene p={p_L:.6f}  KS p={ks_p:.6f}")


# --- TEST C: Error-corrected scatter ---
print("\n" + "-" * 72)
print("TEST C: Distance-error-corrected intrinsic scatter")
print("-" * 72)

def get_intrinsic_scatter(env_label):
    gals = [(r['std_res'], r['sigma_D_dex']) for r in results.values() if r['env3'] == env_label]
    intrinsic = [np.sqrt(max(obs**2 - derr**2, 0)) for obs, derr in gals]
    return np.array(intrinsic), gals

void_intr, void_raw = get_intrinsic_scatter('void')
field_intr, field_raw = get_intrinsic_scatter('field')
dense_intr, dense_raw = get_intrinsic_scatter('dense')

for label, intr, raw in [('Void', void_intr, void_raw),
                          ('Field', field_intr, field_raw),
                          ('Dense', dense_intr, dense_raw)]:
    if len(intr) > 0:
        print(f"\n  {label} (N={len(intr)}):")
        print(f"    Observed scatter:  {np.mean([s for s, _ in raw]):.4f} dex")
        print(f"    Distance error:    {np.mean([d for _, d in raw]):.4f} dex")
        print(f"    Intrinsic scatter: {np.mean(intr):.4f} dex")

if len(void_intr) >= 5 and len(dense_intr) >= 5:
    stat_L, p_L = levene(void_intr, dense_intr)
    print(f"\n  Levene (void intrinsic vs dense intrinsic): F={stat_L:.3f}, p={p_L:.6f}")
if len(void_intr) >= 5 and len(field_intr) >= 5:
    stat_L, p_L = levene(void_intr, field_intr)
    print(f"  Levene (void intrinsic vs field intrinsic): F={stat_L:.3f}, p={p_L:.6f}")
if len(field_intr) >= 5:
    stat_L, p_L = levene(field_intr, dense_intr)
    print(f"  Levene (field intrinsic vs dense intrinsic): F={stat_L:.3f}, p={p_L:.6f}")


# --- TEST D: Acceleration-binned scatter ---
print("\n" + "-" * 72)
print("TEST D: Scatter by acceleration regime and environment")
print("-" * 72)

gbar_bins = [(-13.0, -11.5, 'Deep DM'),
             (-11.5, -10.5, 'Transition'),
             (-10.5, -9.5, 'Baryon-dom'),
             (-9.5, -8.0, 'High gbar')]

print(f"\n  {'Regime':12s} {'N_void':>7s} {'sig_v':>7s} {'N_field':>8s} {'sig_f':>7s} {'N_dense':>8s} {'sig_d':>7s} {'Gradient':>10s}")
print(f"  {'-'*75}")

for gbar_lo, gbar_hi, label in gbar_bins:
    v_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                            for r in results.values() if r['env3'] == 'void']) if env_counts['void'] > 0 else np.array([])
    f_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                            for r in results.values() if r['env3'] == 'field'])
    d_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                            for r in results.values() if r['env3'] == 'dense'])

    sv = np.std(v_pts) if len(v_pts) >= 5 else np.nan
    sf = np.std(f_pts) if len(f_pts) >= 5 else np.nan
    sd = np.std(d_pts) if len(d_pts) >= 5 else np.nan

    # Gradient: void→dense scatter trend
    if not np.isnan(sv) and not np.isnan(sd):
        grad = f"{sd - sv:+.4f}"
    elif not np.isnan(sf) and not np.isnan(sd):
        grad = f"{sd - sf:+.4f} (f-d)"
    else:
        grad = "---"

    sv_str = f"{sv:.4f}" if not np.isnan(sv) else "---"
    sf_str = f"{sf:.4f}" if not np.isnan(sf) else "---"
    sd_str = f"{sd:.4f}" if not np.isnan(sd) else "---"

    print(f"  {label:12s} {len(v_pts):7d} {sv_str:>7s} {len(f_pts):8d} {sf_str:>7s} {len(d_pts):8d} {sd_str:>7s} {grad:>10s}")


# --- TEST E: Bootstrap ---
print("\n" + "-" * 72)
print("TEST E: Bootstrap (10,000 galaxy-resampled iterations)")
print("-" * 72)

rng = np.random.default_rng(42)
n_boot = 10000

void_names = [name for name, r in results.items() if r['env3'] == 'void']
field_names = [name for name, r in results.items() if r['env3'] == 'field']
dense_names = [name for name, r in results.items() if r['env3'] == 'dense']

# Bootstrap: field vs dense (always available)
delta_fd_boots = np.zeros(n_boot)
for b in range(n_boot):
    f_sample = rng.choice(field_names, size=len(field_names), replace=True)
    d_sample = rng.choice(dense_names, size=len(dense_names), replace=True)
    f_res = np.array([results[n]['mean_res'] for n in f_sample])
    d_res = np.array([results[n]['mean_res'] for n in d_sample])
    delta_fd_boots[b] = np.std(f_res) - np.std(d_res)

ci_fd_95 = np.percentile(delta_fd_boots, [2.5, 97.5])
print(f"\n  Field - Dense scatter difference:")
print(f"    Observed: {sigma_field_gal - sigma_dense_gal:+.4f} dex")
print(f"    Bootstrap median: {np.median(delta_fd_boots):+.4f} dex")
print(f"    95% CI: [{ci_fd_95[0]:+.4f}, {ci_fd_95[1]:+.4f}]")
print(f"    P(field > dense): {np.mean(delta_fd_boots > 0):.4f}")

# Bootstrap: void vs dense (if void sample large enough)
if len(void_names) >= 3:
    delta_vd_boots = np.zeros(n_boot)
    delta_vf_boots = np.zeros(n_boot)
    for b in range(n_boot):
        v_sample = rng.choice(void_names, size=len(void_names), replace=True)
        f_sample = rng.choice(field_names, size=len(field_names), replace=True)
        d_sample = rng.choice(dense_names, size=len(dense_names), replace=True)
        v_res = np.array([results[n]['mean_res'] for n in v_sample])
        f_res = np.array([results[n]['mean_res'] for n in f_sample])
        d_res = np.array([results[n]['mean_res'] for n in d_sample])
        delta_vd_boots[b] = np.std(v_res) - np.std(d_res)
        delta_vf_boots[b] = np.std(v_res) - np.std(f_res)

    ci_vd_95 = np.percentile(delta_vd_boots, [2.5, 97.5])
    ci_vf_95 = np.percentile(delta_vf_boots, [2.5, 97.5])

    print(f"\n  Void - Dense scatter difference:")
    print(f"    Observed: {sigma_void_gal - sigma_dense_gal:+.4f} dex")
    print(f"    Bootstrap median: {np.median(delta_vd_boots):+.4f} dex")
    print(f"    95% CI: [{ci_vd_95[0]:+.4f}, {ci_vd_95[1]:+.4f}]")

    print(f"\n  Void - Field scatter difference:")
    print(f"    Observed: {sigma_void_gal - sigma_field_gal:+.4f} dex")
    print(f"    Bootstrap median: {np.median(delta_vf_boots):+.4f} dex")
    print(f"    95% CI: [{ci_vf_95[0]:+.4f}, {ci_vf_95[1]:+.4f}]")
else:
    print(f"\n  Void sample too small ({len(void_names)}) for bootstrap")


# --- TEST F: Monotonicity check ---
print("\n" + "-" * 72)
print("TEST F: Scatter gradient monotonicity")
print("-" * 72)
print("  BEC predicts: sigma_void ≤ sigma_field ≤ sigma_dense")
print("  (or all equal if coupling is truly universal)")

gradient_data = []
for label, sigma_pt, sigma_gal, n_gal in [
    ('Void', sigma_void, sigma_void_gal, env_counts['void']),
    ('Field', sigma_field, sigma_field_gal, env_counts['field']),
    ('Dense', sigma_dense, sigma_dense_gal, env_counts['dense']),
]:
    if n_gal > 0:
        gradient_data.append((label, sigma_pt, sigma_gal, n_gal))
        print(f"  {label:8s}: point_sigma={sigma_pt:.4f}  gal_sigma={sigma_gal:.4f}  N={n_gal}")

if len(gradient_data) == 3:
    v_pt, f_pt, d_pt = [g[1] for g in gradient_data]
    v_gl, f_gl, d_gl = [g[2] for g in gradient_data]

    mono_pt = (v_pt <= f_pt <= d_pt) or (v_pt >= f_pt >= d_pt)
    mono_gl = (v_gl <= f_gl <= d_gl) or (v_gl >= f_gl >= d_gl)

    # BEC direction: void ≤ field ≤ dense (or all equal)
    bec_direction_pt = v_pt <= f_pt <= d_pt
    bec_direction_gl = v_gl <= f_gl <= d_gl

    print(f"\n  Point-level monotonic: {'YES' if mono_pt else 'NO'}")
    print(f"  Galaxy-level monotonic: {'YES' if mono_gl else 'NO'}")
    print(f"  BEC direction (void ≤ field ≤ dense):")
    print(f"    Point-level: {'YES' if bec_direction_pt else 'NO'}")
    print(f"    Galaxy-level: {'YES' if bec_direction_gl else 'NO'}")


# --- TEST G: Void-specific deep look ---
print("\n" + "-" * 72)
print("TEST G: Void galaxy detailed analysis")
print("-" * 72)

if len(void_names) > 0:
    print(f"\n  {'Galaxy':15s} {'D(Mpc)':>8s} {'N_pts':>6s} {'mean_res':>10s} {'std_res':>10s} {'void_id':>8s} {'T':>4s}")
    print(f"  {'-'*65}")
    for name in sorted(void_names):
        r = results[name]
        vc = void_classification.get(name, {})
        print(f"  {name:15s} {r['D']:8.1f} {r['n_points']:6d} {r['mean_res']:+10.4f} {r['std_res']:10.4f} "
              f"{vc.get('void_id', '?'):>8} {r['T']:4d}")

    # Compare void galaxy properties to field
    if len(void_names) >= 3:
        void_D = np.array([results[n]['D'] for n in void_names])
        field_D = np.array([results[n]['D'] for n in field_names])
        void_logMs = np.array([results[n]['logMs'] for n in void_names])
        field_logMs = np.array([results[n]['logMs'] for n in field_names])

        print(f"\n  Property comparison:")
        print(f"    Distance: Void mean={np.mean(void_D):.1f} Mpc, Field mean={np.mean(field_D):.1f} Mpc")
        print(f"    logMs:    Void mean={np.mean(void_logMs):.2f}, Field mean={np.mean(field_logMs):.2f}")
else:
    print("\n  No void galaxies found in cross-match.")
    print("  This is expected: most SPARC galaxies are at D < 30 Mpc,")
    print("  while the VoidFinder volume-limited catalog starts at ~30 Mpc.")
    print("  See discussion in STEP 6 for alternative approaches.")


# ================================================================
# STEP 6: Discussion and alternative approaches
# ================================================================
print("\n" + "=" * 72)
print("DISCUSSION: Coverage and Alternative Void Proxies")
print("=" * 72)

# Analyze distance distribution overlap
sparc_dists = np.array([r['D'] for r in results.values()])
void_min_d = D_MIN_MPH / h  # ~30 Mpc
void_max_d = D_MAX_MPH / h  # ~497 Mpc

n_in_range = np.sum((sparc_dists >= void_min_d) & (sparc_dists <= void_max_d))
n_total = len(sparc_dists)
pct_in_range = 100 * n_in_range / n_total

print(f"\n  SPARC distance range: {sparc_dists.min():.1f} - {sparc_dists.max():.1f} Mpc")
print(f"  VoidFinder coverage: {void_min_d:.0f} - {void_max_d:.0f} Mpc")
print(f"  SPARC galaxies in VoidFinder volume: {n_in_range}/{n_total} ({pct_in_range:.0f}%)")

# Alternative: use local density from Kourkchi catalog as void proxy
print(f"""
  Alternative void proxies for nearby SPARC galaxies:

  1. Kourkchi+2017 group membership:
     - Galaxies NOT in any group → candidate void/field galaxies
     - Group mass → density proxy
     - Already available in project data

  2. Local galaxy density:
     - N-th nearest neighbor density from PGC/2MASS catalogs
     - Low density (<0.1 Mpc⁻³) → void-like environment
     - Requires additional catalog data

  3. Tully+2023 Cosmic Flows:
     - Peculiar velocity field → density field (linear theory)
     - Underdense regions (δ < -0.5) → void candidates
     - Would give continuous density measure for ALL SPARC galaxies
""")


# ================================================================
# SUMMARY & SAVE RESULTS
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY: Void-Field-Dense Gradient Test")
print("=" * 72)

summary = {
    "test_name": "void_field_dense_gradient",
    "description": "Three-tier environmental RAR scatter gradient: void → field → dense",
    "void_catalog": "Douglass+2023 VAST VoidFinder, SDSS DR7, Planck2018 cosmology",
    "void_catalog_url": "https://zenodo.org/records/11043278",
    "dense_catalog": "Verheijen+2001 (UMa), Tully+2013 (NED groups)",
    "n_galaxies_total": len(results),
    "n_void": env_counts['void'],
    "n_field": env_counts['field'],
    "n_dense": env_counts['dense'],
    "void_galaxies": {name: {
        'D_Mpc': results[name]['D'],
        'n_points': results[name]['n_points'],
        'mean_res': results[name]['mean_res'],
        'std_res': results[name]['std_res'],
        'void_id': void_classification.get(name, {}).get('void_id'),
    } for name in void_names} if len(void_names) > 0 else {},
    "point_level_scatter": {
        "void_sigma": round(float(sigma_void), 4) if not np.isnan(sigma_void) else None,
        "field_sigma": round(float(sigma_field), 4),
        "dense_sigma": round(float(sigma_dense), 4),
    },
    "galaxy_level_scatter": {
        "void_sigma": round(float(sigma_void_gal), 4) if len(void_means) > 0 else None,
        "field_sigma": round(float(sigma_field_gal), 4),
        "dense_sigma": round(float(sigma_dense_gal), 4),
    },
    "bootstrap_field_vs_dense": {
        "observed_delta": round(float(sigma_field_gal - sigma_dense_gal), 4),
        "ci_95": [round(float(ci_fd_95[0]), 4), round(float(ci_fd_95[1]), 4)],
    },
    "sparc_void_overlap": {
        "sparc_distance_range_Mpc": [round(float(sparc_dists.min()), 1),
                                      round(float(sparc_dists.max()), 1)],
        "voidfinder_coverage_Mpc": [round(void_min_d, 0), round(void_max_d, 0)],
        "n_sparc_in_voidfinder_volume": int(n_in_range),
        "pct_in_range": round(pct_in_range, 1),
    },
    "quality_cuts": "Q <= 2, 30° < Inc < 85°, N_points ≥ 5",
    "error_model": "Haubner+2025",
    "distances_used": "SPARC",
}

# Add bootstrap void results if available
if len(void_names) >= 3:
    summary["bootstrap_void_vs_dense"] = {
        "observed_delta": round(float(sigma_void_gal - sigma_dense_gal), 4),
        "ci_95": [round(float(ci_vd_95[0]), 4), round(float(ci_vd_95[1]), 4)],
    }
    summary["bootstrap_void_vs_field"] = {
        "observed_delta": round(float(sigma_void_gal - sigma_field_gal), 4),
        "ci_95": [round(float(ci_vf_95[0]), 4), round(float(ci_vf_95[1]), 4)],
    }

# Determine overall result
if env_counts['void'] >= 3:
    if not np.isnan(sigma_void):
        if sigma_void <= sigma_field <= sigma_dense:
            gradient_result = "MONOTONIC_BEC_DIRECTION"
            interpretation = ("Scatter increases from void → field → dense, "
                             "consistent with BEC coherence disruption by environment")
        elif abs(sigma_void - sigma_field) < 0.01 and abs(sigma_field - sigma_dense) < 0.01:
            gradient_result = "UNIFORM"
            interpretation = ("All three environments show indistinguishable scatter, "
                             "consistent with universal BEC coupling")
        else:
            gradient_result = "NON_MONOTONIC"
            interpretation = ("Scatter does not follow void → field → dense gradient, "
                             "environment may not correlate simply with BEC coherence")
    else:
        gradient_result = "INSUFFICIENT_VOID_DATA"
        interpretation = "Too few void galaxies for reliable statistics"
else:
    gradient_result = "INSUFFICIENT_VOID_DATA"
    interpretation = ("Only {0} void galaxies found. Most SPARC galaxies are at D < 30 Mpc, "
                     "below the VoidFinder volume-limited threshold. Alternative approaches "
                     "needed: (1) Kourkchi group non-membership as void proxy, "
                     "(2) Local density from 2MASS/PGC, (3) CAVITY/MaNGA void galaxy surveys.").format(env_counts['void'])

summary["gradient_result"] = gradient_result
summary["interpretation"] = interpretation

print(f"\n  Overall result: {gradient_result}")
print(f"  {interpretation}")

# Save
os.makedirs(RESULTS_DIR, exist_ok=True)
outpath = os.path.join(RESULTS_DIR, 'summary_void_gradient.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n  Results saved to: {outpath}")

# ================================================================
# STEP 7: Kourkchi-based void proxy (alternative for nearby galaxies)
# ================================================================
print("\n" + "=" * 72)
print("ALTERNATIVE: Kourkchi Group Non-Membership as Void/Isolation Proxy")
print("=" * 72)

kourkchi_path = os.path.join(PROJECT_ROOT, 'data', 'misc', 'kourkchi2017_galaxies.tsv')
pgc_path = os.path.join(DATA_DIR, 'sparc_pgc_crossmatch.csv')

if os.path.exists(kourkchi_path) and os.path.exists(pgc_path):
    # Load PGC crossmatch
    sparc_pgc = {}
    with open(pgc_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                sparc_pgc[parts[0]] = int(parts[1])

    # Load Kourkchi group assignments
    # Format (TSV): PGC  Name  RAJ2000  DEJ2000  HRV  PGC1
    # PGC1 = group principal galaxy PGC number
    # If PGC == PGC1, the galaxy is the group's principal member (singleton)
    # If PGC != PGC1, it's a secondary member of that group
    kourkchi_groups = {}
    with open(kourkchi_path, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 6:
                try:
                    pgc = int(parts[0])
                    pgc1 = int(parts[5])
                    kourkchi_groups[pgc] = pgc1
                except ValueError:
                    continue

    print(f"  Loaded {len(sparc_pgc)} SPARC-PGC crossmatches")
    print(f"  Loaded {len(kourkchi_groups)} Kourkchi galaxy group assignments")

    # Load group properties for mass estimation
    group_props_path = os.path.join(PROJECT_ROOT, 'data', 'misc', 'kourkchi2017_massive_groups.tsv')
    group_masses = {}
    if os.path.exists(group_props_path):
        with open(group_props_path, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 22:
                    try:
                        pgc1 = int(parts[1])
                        nm = int(parts[3])
                        logMd = float(parts[20]) if parts[20].strip() else 0.0
                        group_masses[pgc1] = {'Nm': nm, 'logMd': logMd}
                    except (ValueError, IndexError):
                        continue

    # Count group membership from the galaxies catalog itself
    # PGC1 is the principal galaxy; count how many galaxies share each PGC1
    from collections import Counter
    group_member_counts = Counter(kourkchi_groups.values())

    print(f"  Loaded {len(group_masses)} massive group properties")
    print(f"  Group size distribution from galaxy catalog:")
    size_bins = [0, 1, 2, 5, 10, 50, 1000]
    for i in range(len(size_bins) - 1):
        n = sum(1 for c in group_member_counts.values() if size_bins[i] < c <= size_bins[i+1])
        print(f"    {size_bins[i]+1:4d}-{size_bins[i+1]:4d} members: {n:5d} groups")

    # Classify SPARC galaxies by Kourkchi isolation
    isolation_class = {}
    for name, r in results.items():
        pgc = sparc_pgc.get(name, -1)
        if pgc == -1:
            isolation_class[name] = 'no_pgc'
            continue

        if pgc in kourkchi_groups:
            pgc1 = kourkchi_groups[pgc]
            nm_catalog = group_member_counts.get(pgc1, 1)

            # Also check massive_groups table for properties
            gprops = group_masses.get(pgc1, {})
            nm_massive = gprops.get('Nm', 0)
            logMd = gprops.get('logMd', 0)

            # Use the larger of the two membership counts
            nm = max(nm_catalog, nm_massive)

            if pgc == pgc1 and nm <= 1:
                # Singleton — galaxy IS the group (isolated)
                isolation_class[name] = 'isolated'
            elif nm >= 5:
                isolation_class[name] = 'rich_group'
            elif nm >= 2:
                isolation_class[name] = 'poor_group'
            else:
                isolation_class[name] = 'isolated'
        else:
            # Not in Kourkchi catalog at all — likely very isolated
            isolation_class[name] = 'isolated'

    # Count and report
    iso_counts = {}
    for v in isolation_class.values():
        iso_counts[v] = iso_counts.get(v, 0) + 1

    print(f"\n  Kourkchi isolation classification:")
    for k in ['isolated', 'poor_group', 'rich_group', 'no_pgc']:
        print(f"    {k:15s}: {iso_counts.get(k, 0)}")

    # Run scatter comparison: isolated vs group
    isolated_names = [n for n, c in isolation_class.items() if c == 'isolated' and n in results]
    group_names = [n for n, c in isolation_class.items() if c in ('poor_group', 'rich_group') and n in results]

    if len(isolated_names) >= 5 and len(group_names) >= 5:
        iso_means = np.array([results[n]['mean_res'] for n in isolated_names])
        grp_means = np.array([results[n]['mean_res'] for n in group_names])

        iso_pts = np.concatenate([results[n]['log_res'] for n in isolated_names])
        grp_pts = np.concatenate([results[n]['log_res'] for n in group_names])

        sigma_iso_pt = np.std(iso_pts)
        sigma_grp_pt = np.std(grp_pts)
        sigma_iso_gal = np.std(iso_means)
        sigma_grp_gal = np.std(grp_means)

        stat_L_pt, p_L_pt = levene(iso_pts, grp_pts)
        stat_L_gl, p_L_gl = levene(iso_means, grp_means)

        print(f"\n  Isolated vs Group scatter comparison:")
        print(f"    Point-level:  Isolated sigma={sigma_iso_pt:.4f}, Group sigma={sigma_grp_pt:.4f}")
        print(f"                  Levene p={p_L_pt:.6f}")
        print(f"    Galaxy-level: Isolated sigma={sigma_iso_gal:.4f}, Group sigma={sigma_grp_gal:.4f}")
        print(f"                  Levene p={p_L_gl:.6f}")

        # Add to summary
        summary["kourkchi_isolation_test"] = {
            "n_isolated": len(isolated_names),
            "n_group": len(group_names),
            "point_level": {
                "isolated_sigma": round(float(sigma_iso_pt), 4),
                "group_sigma": round(float(sigma_grp_pt), 4),
                "levene_p": round(float(p_L_pt), 6),
            },
            "galaxy_level": {
                "isolated_sigma": round(float(sigma_iso_gal), 4),
                "group_sigma": round(float(sigma_grp_gal), 4),
                "levene_p": round(float(p_L_gl), 6),
            },
        }

        # Re-save with updated results
        with open(outpath, 'w') as f:
            json.dump(summary, f, indent=2)
else:
    print("  Kourkchi catalog or PGC crossmatch not found, skipping alternative test")


print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
print(f"\nFull results: {outpath}")
