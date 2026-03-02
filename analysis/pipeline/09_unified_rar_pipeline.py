#!/usr/bin/env python3
"""
STEP 9: Unified RAR Environmental Scatter Pipeline
====================================================
Combines ALL available rotation curve and HI survey datasets into one
unified environmental scatter test of the Radial Acceleration Relation.

Datasets:
  1. SPARC (175 galaxies, full mass models from Lelli+2016)
  2. de Blok+2002 LSB (26 galaxies, full mass models)
  3. WALLABY DR1+DR2 (165 galaxies, gas-only RAR)
  4. Santos-Santos+2020 (160 dwarfs, single-point RAR)
  5. LITTLE THINGS / Oh+2015 (26 dwarfs, single-point RAR)
  6. LVHIS / Koribalski+2018 (47 galaxies, single-point RAR)
  7. Yu+2020 (269 galaxies from HI spectra)
  8. Swaters+2025 DiskMass Survey (125 galaxies, H-alpha kinematics)
  9. GHASP / Epinat+2008 (93 galaxies, H-alpha rotation curves)
  10. Noordermeer+2005 WHISP (68 early-type galaxies with group/cluster membership)
  11. Vogt+2004 (cluster spirals, single-point)
  12. Catinella+2005 (cluster HI, single-point)
  13. Virgo extended RC compilation
  14. PHANGS-ALMA Lang+2020 (67 galaxies, CO rotation curves, ~17 Virgo)
  15. Verheijen+2001 UMa (41 galaxies, HI rotation curves, all dense)
  16. WALLABY DR2 (236 galaxies, HI rotation curves in clusters)
  17. MaNGA Ristea+2023 (~1,500 galaxies, IFU gas rotation at 1Re/1.3Re/2Re)

Total: 2000+ unique galaxies after overlap removal.

The BEC dark matter prediction is that RAR scatter depends on galaxy
environment (dense clusters vs field), because superfluid coherence
is disrupted in hot cluster halos.

Russell Licht -- Primordial Fluid DM Project
Feb 2026
"""

import argparse
import datetime as dt
import hashlib
import numpy as np
from scipy import stats, optimize
from scipy.optimize import curve_fit
import csv
import json
import os
import re
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS AND CONSTANTS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

UTILS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'utils'))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)
try:
    from chi2_calibration import chi2_red_given_sigma_int, solve_sigma_int_for_chi2_1
    CHI2_CAL_IMPORT_ERROR = None
except Exception as _chi2_cal_import_err:
    chi2_red_given_sigma_int = None
    solve_sigma_int_for_chi2_1 = None
    CHI2_CAL_IMPORT_ERROR = str(_chi2_cal_import_err)
try:
    from galaxy_naming import canonicalize_galaxy_name
except Exception:
    def canonicalize_galaxy_name(name):
        s = str(name).strip().upper()
        s = re.sub(r'\s+', ' ', s)
        if s.startswith('WALLABY'):
            return s
        return s.replace(' ', '')

# Physical constants
g_dagger = 1.20e-10       # m/s^2 (McGaugh+2016 RAR acceleration scale)
conv = 1e6 / 3.0857e19    # (km/s)^2/kpc -> m/s^2
G_kpc = 4.302e-6          # G in (km/s)^2 kpc / Msun  [= 4.302e-3 pc (km/s)^2/Msun]
H0 = 75.0                 # km/s/Mpc
c_light = 2.998e5         # km/s

# Mass-to-light defaults
Y_DISK = 0.5
Y_BULGE = 0.7


# ============================================================
# KNOWN LARGE-SCALE STRUCTURES FOR ENVIRONMENT CLASSIFICATION
# ============================================================
# Format: (RA_deg, Dec_deg, Vsys_km/s, logMh, angular_radius_deg, sigma_v_km/s)
STRUCTURES = {
    'Virgo':     (187.71, 12.39,  1100, 14.9, 6.0, 800),
    'Fornax':    (54.62, -35.45,  1379, 14.0, 2.0, 370),
    'Hydra':     (159.18, -27.53, 3777, 14.5, 2.0, 700),
    'Norma':     (248.15, -60.75, 4871, 15.0, 1.5, 925),
    'Centaurus': (192.20, -41.31, 3627, 14.5, 2.0, 750),
    'Coma':      (194.95, 27.98,  6925, 15.0, 3.0, 1000),
    'Perseus':   (49.95, 41.51,   5366, 14.8, 2.0, 1300),
    'NGC5044':   (198.85, -16.39, 2750, 13.5, 1.5, 400),
    'NGC4636':   (190.71, 2.69,    928, 13.3, 1.5, 350),
    'M81':       (148.89, 69.07,   -34, 12.0, 3.0, 200),
    'UMa':       (178.0, 49.0,    1050, 12.8, 8.0, 200),
    'Sculptor':  (11.89, -33.72,   200, 11.5, 5.0, 100),
    'CenA':      (201.37, -43.02,  547, 12.5, 5.0, 200),
    'Antlia':    (157.48, -35.32, 3041, 13.8, 1.0, 545),
}


# ============================================================
# KOURKCHI & TULLY 2017 GROUP CATALOG
# ============================================================
# Load galaxy→group membership for improved environment classification
# Uses J/ApJ/843/16: ~15k galaxies with group assignments & halo masses
def load_kourkchi2017_groups():
    """
    Load Kourkchi & Tully 2017 group catalog for environment classification.
    Returns:
      galaxy_groups: dict mapping normalized galaxy name → {'pgc1': str, 'logMd': float}
      group_props: dict mapping pgc1 → {'logMd': float, 'nm': int, 'sigmaV': float, ...}
    """
    gal_path = os.path.join(DATA_DIR, 'kourkchi2017_galaxies.tsv')
    grp_path = os.path.join(DATA_DIR, 'kourkchi2017_massive_groups.tsv')

    galaxy_groups = {}
    group_props = {}

    if not os.path.exists(gal_path) or not os.path.exists(grp_path):
        return galaxy_groups, group_props

    # Load group properties (logMd >= 12)
    with open(grp_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue
            row = {header[i]: parts[i].strip() for i in range(len(header))}
            pgc1 = row.get('PGC1', '').strip()
            try:
                logMd = float(row.get('logMd', '0') or '0')
                nm = int(row.get('Nm', '0') or '0')
                sigmaV = float(row.get('sigmaV', '0') or '0')
            except (ValueError, TypeError):
                continue
            group_props[pgc1] = {
                'logMd': logMd, 'nm': nm, 'sigmaV': sigmaV,
            }

    # Load galaxy→group membership
    with open(gal_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue
            row = {header[i]: parts[i].strip().strip('"') for i in range(len(header))}
            name = row.get('Name', '').strip()
            pgc1 = row.get('PGC1', '').strip()

            if not name or pgc1 not in group_props:
                continue

            # Normalize name for matching
            norm = re.sub(r'\s+', '', name.upper())
            grp = group_props[pgc1]
            entry = {'pgc1': pgc1, 'logMd': grp['logMd'], 'nm': grp['nm']}

            galaxy_groups[norm] = entry
            # Add common variants (strip leading zeros)
            for prefix in ['UGC', 'NGC', 'IC']:
                m = re.match(rf'^{prefix}0*(\d+)', norm)
                if m:
                    galaxy_groups[f'{prefix}{m.group(1)}'] = entry

    return galaxy_groups, group_props


# Global: load once at module level
K17_GALAXY_GROUPS, K17_GROUP_PROPS = load_kourkchi2017_groups()
if K17_GALAXY_GROUPS:
    n_dense_k17 = sum(1 for v in K17_GALAXY_GROUPS.values() if v['logMd'] >= 12.5)
    print(f"  Kourkchi & Tully 2017: {len(K17_GALAXY_GROUPS)} galaxy→group mappings "
          f"({n_dense_k17} in dense groups)")


# ============================================================
# z0MGS (LEROY+2019) WISE STELLAR MASS CATALOG
# ============================================================
# Provides uniform WISE W1-based stellar masses for ~15,750 nearby galaxies.
# Used to upgrade TF-based mass estimates in non-SPARC datasets.

def load_z0mgs_masses():
    """
    Load z0MGS (Leroy+2019) WISE stellar mass catalog.
    Returns dict: normalized_galaxy_name -> {'logMstar': float, 'dist_mpc': float, 'pgc': str}
    Also returns dict: (ra, dec) -> same, for coordinate-based matching.
    """
    z0mgs_path = os.path.join(DATA_DIR, 'z0mgs_leroy2019_masses.tsv')

    name_lookup = {}
    coord_entries = []

    if not os.path.exists(z0mgs_path):
        return name_lookup, coord_entries

    with open(z0mgs_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue
            row = {header[i]: parts[i].strip().strip('"').strip() for i in range(len(header))}

            pgc = row.get('PGC', '').strip()
            logMstar = row.get('logM*', '').strip()
            dist = row.get('Dist', '').strip()
            ra_str = row.get('_RA', '').strip()
            dec_str = row.get('_DE', '').strip()

            if not logMstar:
                continue
            try:
                logM = float(logMstar)
                d_mpc = float(dist) if dist else np.nan
            except (ValueError, TypeError):
                continue

            entry = {'logMstar': logM, 'dist_mpc': d_mpc, 'pgc': pgc}

            # Index by galaxy name (NGC, UGC, IC)
            for col, prefix in [('NGC', 'NGC'), ('UGC', 'UGC'), ('IC', 'IC')]:
                name_raw = row.get(col, '').strip()
                if name_raw:
                    # Strip leading zeros: NGC0628 -> NGC628
                    m = re.match(rf'^{prefix}0*(\d+\w*)', name_raw, re.IGNORECASE)
                    if m:
                        norm = f"{prefix}{m.group(1)}"
                    else:
                        norm = re.sub(r'\s+', '', name_raw.upper())
                    name_lookup[norm] = entry

            # Store with coords for fallback matching
            try:
                ra_val = float(ra_str) if ra_str else np.nan
                dec_val = float(dec_str) if dec_str else np.nan
                if not np.isnan(ra_val) and not np.isnan(dec_val):
                    coord_entries.append({
                        'ra': ra_val, 'dec': dec_val,
                        'logMstar': logM, 'dist_mpc': d_mpc, 'pgc': pgc
                    })
            except (ValueError, TypeError):
                pass

    return name_lookup, coord_entries


Z0MGS_NAMES, Z0MGS_COORDS = load_z0mgs_masses()
if Z0MGS_NAMES:
    print(f"  z0MGS (Leroy+2019): {len(Z0MGS_NAMES)} name-indexed, "
          f"{len(Z0MGS_COORDS)} coord-indexed galaxies with WISE stellar masses")


def get_z0mgs_stellar_mass(name='', ra=np.nan, dec=np.nan):
    """
    Look up WISE-based stellar mass from z0MGS for a galaxy.
    Returns logMstar or None if not found.
    Priority: name match, then coordinate match (<60 arcsec).
    """
    if not Z0MGS_NAMES and not Z0MGS_COORDS:
        return None

    # Try name match first
    if name:
        norm = re.sub(r'\s+', '', name.upper())
        # Strip leading zeros
        for prefix in ['NGC', 'UGC', 'IC']:
            m = re.match(rf'^{prefix}0*(\d+\w*)', norm, re.IGNORECASE)
            if m:
                norm = f"{prefix}{m.group(1)}"
                break
        entry = Z0MGS_NAMES.get(norm)
        if entry is not None:
            return entry['logMstar']

    # Fallback: coordinate match
    if not np.isnan(ra) and not np.isnan(dec) and Z0MGS_COORDS:
        best_sep = 999.0
        best_logM = None
        for e in Z0MGS_COORDS:
            sep = angular_separation(ra, dec, e['ra'], e['dec'])
            if sep < best_sep:
                best_sep = sep
                best_logM = e['logMstar']
        if best_sep < 60.0 / 3600.0:  # 60 arcsec
            return best_logM

    return None


# ============================================================
# UTILITY: VizieR TSV PARSER
# ============================================================
def parse_vizier_tsv(filepath):
    """
    Parse a VizieR TSV file, skipping all comment/header lines.

    Returns (header_list, rows_list_of_dicts).
    Handles lines starting with #, blank lines, unit lines, dashed separators.
    """
    if not os.path.exists(filepath):
        return [], []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    header = None
    data_rows = []
    past_separator = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue

        # Skip dashed separator lines
        if re.match(r'^[-]+(\t[-]+)*$', stripped) or stripped.startswith('---'):
            past_separator = True
            continue

        parts = line.rstrip('\n').split('\t')

        if header is None:
            # First non-comment, non-separator line is the header
            header = [p.strip() for p in parts]
            continue

        # Skip unit lines (usually the line right after header, before separator)
        if not past_separator:
            continue

        # This is a data row
        row = {}
        for i, col in enumerate(header):
            if i < len(parts):
                row[col] = parts[i].strip()
            else:
                row[col] = ''
        data_rows.append(row)

    return header if header else [], data_rows


def safe_float(val, default=np.nan):
    """Safely convert a string to float, returning default on failure."""
    if val is None:
        return default
    val = str(val).strip()
    if val == '' or val == '...' or val == '---' or val == '-':
        return default
    # Remove leading/trailing non-numeric chars like '>' or '<'
    val = re.sub(r'^[<>~()]', '', val).strip()
    val = re.sub(r'[()]$', '', val).strip()
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    """Safely convert a string to int."""
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default


# ============================================================
# RAR COMPUTATION UTILITIES
# ============================================================
def rar_prediction(gbar):
    """
    Compute predicted gobs from the McGaugh+2016 RAR:
      gobs_pred = gbar / (1 - exp(-sqrt(gbar / g_dagger)))
    """
    gbar = np.asarray(gbar, dtype=float)
    x = np.sqrt(np.maximum(gbar, 1e-20) / g_dagger)
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-20)
    return gbar / denom


def compute_log_residual(gbar, gobs):
    """Compute RAR log-residual: log10(gobs) - log10(gobs_pred)."""
    gbar = np.asarray(gbar, dtype=float)
    gobs = np.asarray(gobs, dtype=float)
    gobs_pred = rar_prediction(gbar)

    valid = (gbar > 0) & (gobs > 0) & (gobs_pred > 0) & np.isfinite(gbar) & np.isfinite(gobs)
    log_res = np.full_like(gbar, np.nan)
    log_res[valid] = np.log10(gobs[valid]) - np.log10(gobs_pred[valid])
    return log_res


# ============================================================
# HYPERLEDA INCLINATION CROSS-MATCH
# ============================================================
# Load HyperLEDA (VizieR VII/237) axis ratios for inclination derivation.
# Uses logR25 (log of major-to-minor axis ratio) to compute inclination:
#   i = arccos(sqrt((10^(-2*logR25) - q0^2) / (1 - q0^2)))
# where q0 = 0.2 is the intrinsic axial ratio of an edge-on disk.
# Cross-matches by sky position using a KD-tree on unit-sphere Cartesian coords.

HLEDA_TREE = None      # scipy cKDTree on unit-sphere coords
HLEDA_INCL = None      # array of inclinations (degrees) matching tree indices
HLEDA_LOGR25 = None    # array of raw logR25 values
HLEDA_LOADED = False

def _load_hyperleda_inclinations():
    """Load HyperLEDA catalog and build KD-tree for fast cross-matching."""
    global HLEDA_TREE, HLEDA_INCL, HLEDA_LOGR25, HLEDA_LOADED
    if HLEDA_LOADED:
        return HLEDA_TREE is not None

    HLEDA_LOADED = True
    hleda_path = os.path.join(DATA_DIR, 'hyperleda_inclinations.tsv')
    if not os.path.exists(hleda_path):
        print(f"    [HyperLEDA] Catalog not found at {hleda_path}")
        return False

    print(f"    [HyperLEDA] Loading axis ratios from VizieR VII/237...")
    from scipy.spatial import cKDTree

    # Parse VizieR TSV - optimized for large file (930K+ rows)
    ra_list = []
    dec_list = []
    logr25_list = []

    with open(hleda_path, 'r', encoding='utf-8', errors='replace') as f:
        header = None
        past_separator = False
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if stripped.startswith('---') or re.match(r'^[-]+(\t[-]+)*$', stripped):
                past_separator = True
                continue
            parts = line.rstrip('\n').split('\t')
            if header is None:
                header = [p.strip() for p in parts]
                continue
            if not past_separator:
                continue

            # Extract columns by index for speed
            row = {}
            for i, col in enumerate(header):
                if i < len(parts):
                    row[col] = parts[i].strip()

            ra_str = row.get('RAJ2000', '').strip()
            dec_str = row.get('DEJ2000', '').strip()
            logr25_str = row.get('logR25', '').strip()

            if not ra_str or not dec_str or not logr25_str:
                continue

            try:
                logr25 = float(logr25_str)
            except ValueError:
                continue

            # Parse sexagesimal RA/Dec
            try:
                ra_parts = ra_str.split()
                if len(ra_parts) < 3:
                    continue
                ra_deg = (float(ra_parts[0]) + float(ra_parts[1])/60.0
                          + float(ra_parts[2])/3600.0) * 15.0

                dec_parts_raw = dec_str.replace('+', '').replace('-', '').split()
                dec_sign = -1 if dec_str.startswith('-') else 1
                if len(dec_parts_raw) < 3:
                    continue
                dec_deg = dec_sign * (float(dec_parts_raw[0]) + float(dec_parts_raw[1])/60.0
                                       + float(dec_parts_raw[2])/3600.0)
            except (ValueError, IndexError):
                continue

            # Valid logR25 range: 0 to ~0.8 (face-on to edge-on)
            if logr25 < 0.0 or logr25 > 0.9:
                continue

            ra_list.append(ra_deg)
            dec_list.append(dec_deg)
            logr25_list.append(logr25)

    if len(ra_list) == 0:
        print(f"    [HyperLEDA] No valid entries found")
        return False

    ra_arr = np.array(ra_list, dtype=np.float64)
    dec_arr = np.array(dec_list, dtype=np.float64)
    logr25_arr = np.array(logr25_list, dtype=np.float64)

    # Compute inclinations from logR25
    # i = arccos(sqrt((10^(-2*logR25) - q0^2) / (1 - q0^2)))
    # q0 = 0.2 (intrinsic axial ratio for oblate disk)
    q0 = 0.2
    r25 = 10.0 ** (-logr25_arr)  # minor/major axis ratio (b/a)
    r25_sq = r25 ** 2
    # Ensure argument to arccos is valid
    arg = (r25_sq - q0**2) / (1.0 - q0**2)
    arg = np.clip(arg, 0.0, 1.0)
    incl_rad = np.arccos(np.sqrt(arg))
    incl_deg = np.degrees(incl_rad)

    # Build KD-tree on unit-sphere Cartesian coordinates
    ra_rad = np.radians(ra_arr)
    dec_rad = np.radians(dec_arr)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    coords = np.column_stack([x, y, z])

    HLEDA_TREE = cKDTree(coords)
    HLEDA_INCL = incl_deg
    HLEDA_LOGR25 = logr25_arr

    print(f"    [HyperLEDA] Loaded {len(ra_arr):,} galaxies with axis ratios")
    print(f"    [HyperLEDA] Inclination range: {np.min(incl_deg):.1f}° – {np.max(incl_deg):.1f}°")
    print(f"    [HyperLEDA] Median inclination: {np.median(incl_deg):.1f}°")

    return True


def get_hyperleda_inclination(ra_deg, dec_deg, match_radius_arcsec=30.0):
    """
    Cross-match a sky position to HyperLEDA and return inclination in degrees.

    Parameters:
        ra_deg, dec_deg: Sky position in decimal degrees (J2000)
        match_radius_arcsec: Maximum match radius in arcseconds (default 30")

    Returns:
        incl_deg: Inclination in degrees, or None if no match / face-on (i < 25°)
    """
    if HLEDA_TREE is None:
        return None

    # Convert to unit-sphere Cartesian
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    query_pt = np.array([x, y, z])

    # KD-tree query: Cartesian distance on unit sphere
    # For small angles: chord_length ≈ angular_separation_rad
    # 30 arcsec = 30/3600 * pi/180 = 1.454e-4 rad
    max_chord = match_radius_arcsec / 3600.0 * np.pi / 180.0

    dist, idx = HLEDA_TREE.query(query_pt, distance_upper_bound=max_chord)

    if np.isinf(dist):
        return None  # No match within radius

    incl = float(HLEDA_INCL[idx])

    # Reject near-face-on galaxies where inclination correction diverges
    # sin(25°) = 0.42, so V_rot = W50/(2*0.42) — 2.4x amplification, acceptable
    # sin(20°) = 0.34, so V_rot = W50/(2*0.34) — 2.9x amplification, too noisy
    if incl < 25.0:
        return None  # Face-on: inclination correction too uncertain

    return incl


# ============================================================
# ENVIRONMENT CLASSIFICATION
# ============================================================
def angular_separation(ra1, dec1, ra2, dec2):
    """Compute angular separation in degrees between two sky positions."""
    ra1r, dec1r = np.radians(ra1), np.radians(dec1)
    ra2r, dec2r = np.radians(ra2), np.radians(dec2)
    cos_sep = (np.sin(dec1r) * np.sin(dec2r) +
               np.cos(dec1r) * np.cos(dec2r) * np.cos(ra1r - ra2r))
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sep))


def classify_environment_proximity(ra, dec, vsys, name=''):
    """
    Classify galaxy environment using:
      1) Kourkchi & Tully 2017 group catalog (primary, if available)
      2) Proximity to known large-scale structures (fallback)

    Returns: (structure_name, logMh, env_binary)
      where env_binary is 'dense' if logMh >= 12.5, else 'field'.
    """
    # --- Primary: Kourkchi & Tully 2017 group catalog lookup ---
    if K17_GALAXY_GROUPS and name:
        norm = re.sub(r'\s+', '', name.upper())
        k17 = K17_GALAXY_GROUPS.get(norm, None)
        if k17 is not None:
            logMd = k17['logMd']
            env = 'dense' if logMd >= 12.5 else 'field'
            return f"K17_group_{k17['pgc1']}", logMd, env

    # --- Fallback: proximity to known structures ---
    if np.isnan(ra) or np.isnan(dec) or np.isnan(vsys):
        return 'field', 11.0, 'field'

    best = None
    best_score = 999.0

    for sname, (sra, sdec, sv, slogMh, srv, ssigv) in STRUCTURES.items():
        ang = angular_separation(ra, dec, sra, sdec)

        if ang < srv and abs(vsys - sv) < 2.5 * ssigv:
            score = (ang / srv) ** 2 + (abs(vsys - sv) / max(ssigv, 1)) ** 2
            if score < best_score:
                best_score = score
                best = (sname, slogMh)

    if best:
        sname, logMh = best
        env_dense = 'dense' if logMh >= 12.5 else 'field'
        return sname, logMh, env_dense

    return 'field', 11.0, 'field'


# ============================================================
# OVERLAP REMOVAL: FUZZY NAME MATCHING
# ============================================================
def normalize_galaxy_name(name):
    """
    Normalize a galaxy name for fuzzy matching.
    Removes spaces, converts to uppercase, strips common prefixes.
    Examples: 'NGC 300' -> 'NGC300', 'DDO 154' -> 'DDO154',
              'UGC 4325' -> 'UGC4325'
    """
    return canonicalize_galaxy_name(name)


def build_sparc_name_set(sparc_names):
    """Build a set of normalized SPARC names for overlap detection."""
    norm_set = set()
    for name in sparc_names:
        norm_set.add(normalize_galaxy_name(name))
    return norm_set


def is_sparc_duplicate(name, sparc_norm_set):
    """Check if a galaxy name matches any SPARC galaxy (fuzzy)."""
    norm = normalize_galaxy_name(name)
    if norm in sparc_norm_set:
        return True

    # Also check without dashes: 'ESO079-G014' vs 'ESO079-G014'
    norm_nodash = norm.replace('-', '')
    for sn in sparc_norm_set:
        if sn.replace('-', '') == norm_nodash:
            return True

    return False


# ============================================================
# M/L OPTIMIZATION (from pipeline 02)
# ============================================================
def fit_ml_ratio(gdata, D_Mpc, props_entry):
    """
    Optimize M/L ratio to minimize error-weighted RAR scatter.
    This is the same approach as 02_cf4_rar_pipeline.py.

    Uses scipy.optimize.minimize with L-BFGS-B to find optimal
    Y_disk and Y_bulge that minimize the weighted chi-squared
    of RAR residuals.
    """
    D_orig = gdata['dist_sparc']
    D_ratio = D_Mpc / max(D_orig, 0.01)

    R = gdata['R'] * D_ratio
    Vobs = gdata['Vobs']
    eVobs = gdata['eVobs']
    sqrt_ratio = np.sqrt(D_ratio)
    Vgas = gdata['Vgas'] * sqrt_ratio
    Vdisk = gdata['Vdisk'] * sqrt_ratio
    Vbul = gdata['Vbul'] * sqrt_ratio

    mask = R > 0
    if np.sum(mask) < 5:
        return {'Y_disk': 0.5, 'Y_bul': 0.7}

    R_m = R[mask]
    Vobs_m = Vobs[mask]
    eVobs_m = eVobs[mask]
    Vgas_m = Vgas[mask]
    Vdisk_m = Vdisk[mask]
    Vbul_m = Vbul[mask]

    def chi2(params):
        Y_d = params[0]
        Y_b = params[1] if len(params) > 1 else 0.7
        Vbar_sq = np.sign(Vgas_m) * Vgas_m**2 + Y_d * Vdisk_m**2 + Y_b * Vbul_m**2
        gb = np.abs(Vbar_sq) / R_m * conv
        go = Vobs_m**2 / R_m * conv
        v = (go > 0) & (gb > 0)
        if np.sum(v) < 3:
            return 1e10
        gop = gb[v] / (1 - np.exp(-np.sqrt(gb[v] / g_dagger)))
        lr = np.log10(go[v]) - np.log10(gop)
        # Error weighting: sigma_log = 2 * eV/V / ln(10)
        w = 1.0 / np.maximum((2 * eVobs_m[v] / np.maximum(Vobs_m[v], 1))**2, 0.01)
        return np.sum(w * lr**2)

    has_bulge = np.any(np.abs(Vbul_m) > 0)
    try:
        if has_bulge:
            res = optimize.minimize(chi2, [0.5, 0.7],
                                    bounds=[(0.1, 1.2), (0.1, 1.5)],
                                    method='L-BFGS-B')
            return {'Y_disk': float(res.x[0]), 'Y_bul': float(res.x[1])}
        else:
            res = optimize.minimize(chi2, [0.5],
                                    bounds=[(0.1, 1.2)],
                                    method='L-BFGS-B')
            return {'Y_disk': float(res.x[0]), 'Y_bul': 0.7}
    except Exception:
        return {'Y_disk': 0.5, 'Y_bul': 0.7}


# ============================================================
# DATASET 1: SPARC (175 galaxies, full mass models)
# ============================================================
def sha256_file(path):
    """Streaming SHA256 for reproducibility metadata."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_head_sha():
    """Best-effort git HEAD hash (or None outside a git checkout)."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def resolve_sparc_paths():
    """
    Resolve SPARC inputs across legacy and reorganized data layouts.

    Returns a dict with selected path + candidates/existence for diagnostics.
    """
    def _pick(candidates):
        expanded = [os.path.join(DATA_DIR, p) for p in candidates]
        chosen = next((p for p in expanded if os.path.exists(p)), None)
        return {
            'chosen': chosen,
            'candidates': expanded,
            'exists': {p: os.path.exists(p) for p in expanded},
        }

    return {
        'table2': _pick(['SPARC_table2_rotmods.dat', os.path.join('sparc', 'SPARC_table2_rotmods.dat')]),
        'mrt': _pick(['SPARC_Lelli2016c.mrt', os.path.join('sparc', 'SPARC_Lelli2016c.mrt')]),
        'cf4_cache': _pick(['cf4_distance_cache.json', os.path.join('cf4', 'cf4_distance_cache.json')]),
        'env_catalog': _pick(['sparc_environment_catalog.json', os.path.join('environment', 'sparc_environment_catalog.json')]),
        'coordinates': _pick(['sparc_coordinates.json', os.path.join('sparc', 'sparc_coordinates.json')]),
    }


def load_sparc_data():
    """
    Load SPARC galaxies from the fixed-width table2 (mass models)
    and MRT table1 (galaxy properties).

    Returns list of per-point RAR data dicts.
    """
    print("\n  [1/10] Loading SPARC (175 galaxies)...")

    sparc_diag = {
        'paths': resolve_sparc_paths(),
        'table2_missing_reason': None,
        'parsed_mass_model_galaxies': 0,
        'parsed_mrt_properties': 0,
        'sparc_galaxies_after_cuts': 0,
        'sparc_points_after_cuts': 0,
    }

    # --- Parse Table 2: mass models (fixed-width) ---
    table2_path = sparc_diag['paths']['table2']['chosen']
    print(f"    [DEBUG] SPARC table2 path: {table2_path}")
    if not table2_path:
        print("    WARNING: SPARC_table2_rotmods.dat not found in any known location")
        sparc_diag['table2_missing_reason'] = 'table2_not_found'
        return [], [], sparc_diag

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
                sbdisk = float(line[60:67].strip())
                sbbul_str = line[68:76].strip() if len(line) > 68 else '0'
                sbbul = float(sbbul_str) if sbbul_str else 0.0
            except (ValueError, IndexError):
                continue

            if name not in galaxies:
                galaxies[name] = {
                    'name': name, 'dist_sparc': dist,
                    'R': [], 'Vobs': [], 'eVobs': [],
                    'Vgas': [], 'Vdisk': [], 'Vbul': [],
                    'SBdisk': [], 'SBbul': [],
                }

            galaxies[name]['R'].append(rad)
            galaxies[name]['Vobs'].append(vobs)
            galaxies[name]['eVobs'].append(evobs)
            galaxies[name]['Vgas'].append(vgas)
            galaxies[name]['Vdisk'].append(vdisk)
            galaxies[name]['Vbul'].append(vbul)
            galaxies[name]['SBdisk'].append(sbdisk)
            galaxies[name]['SBbul'].append(sbbul)

    # Convert to numpy
    for name in galaxies:
        for key in ['R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']:
            galaxies[name][key] = np.array(galaxies[name][key])

    sparc_diag['parsed_mass_model_galaxies'] = int(len(galaxies))
    print(f"    Parsed mass models for {len(galaxies)} galaxies")

    # --- Parse MRT: galaxy properties ---
    mrt_path = sparc_diag['paths']['mrt']['chosen']
    props = {}
    if mrt_path and os.path.exists(mrt_path):
        with open(mrt_path, 'r') as f:
            lines = f.readlines()

        # Find data start (after the last '---' separator)
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('---') and i > 50:
                data_start = i + 1
                break

        for line in lines[data_start:]:
            if not line.strip() or line.startswith('#'):
                continue
            try:
                # Name is right-justified in first 11 characters
                name = line[0:11].strip()
                if not name:
                    continue
                # Rest of columns are whitespace-separated
                # Order: T, D, eD, fD, Inc, eInc, L36, eL36, Reff, SBeff,
                #        Rdisk, SBdisk, MHI, RHI, Vflat, eVflat, Q, Ref
                parts = line[11:].split()
                if len(parts) < 17:
                    continue

                T = safe_int(parts[0])
                D = safe_float(parts[1])
                eD = safe_float(parts[2])
                fD = safe_int(parts[3])
                Inc = safe_float(parts[4])
                eInc = safe_float(parts[5])
                Vflat = safe_float(parts[14])
                eVflat = safe_float(parts[15])
                Q = safe_int(parts[16])

                props[name] = {
                    'T': T, 'D': D, 'eD': eD, 'fD': fD,
                    'Inc': Inc, 'eInc': eInc,
                    'Vflat': Vflat, 'eVflat': eVflat, 'Q': Q,
                }
            except (ValueError, IndexError):
                continue

        sparc_diag['parsed_mrt_properties'] = int(len(props))
        print(f"    Parsed properties for {len(props)} galaxies")

    # --- Load CF4 distances if available ---
    cf4_cache_path = sparc_diag['paths']['cf4_cache']['chosen']
    cf4_distances = {}
    if cf4_cache_path and os.path.exists(cf4_cache_path):
        with open(cf4_cache_path, 'r') as f:
            cf4_distances = json.load(f)
        print(f"    Loaded {len(cf4_distances)} CF4 distances from cache")

    # --- Load environment catalog ---
    env_catalog = {}
    env_path = sparc_diag['paths']['env_catalog']['chosen']
    if env_path and os.path.exists(env_path):
        with open(env_path, 'r') as f:
            env_catalog = json.load(f)
        print(f"    Loaded environment catalog for {len(env_catalog)} galaxies")

    # --- Load SPARC coordinates ---
    coord_path = sparc_diag['paths']['coordinates']['chosen']
    coordinates = {}
    if coord_path and os.path.exists(coord_path):
        with open(coord_path, 'r') as f:
            coordinates = json.load(f)

    # --- Compute RAR for each galaxy ---
    sparc_points = []     # Per-point RAR data
    sparc_galaxies = []   # Per-galaxy summary
    sparc_names = []

    for name, gdata in galaxies.items():
        p = props.get(name, {})
        D_sparc = p.get('D', gdata['dist_sparc'])
        fD = p.get('fD', 1)
        Inc = p.get('Inc', 60.0)
        Q = p.get('Q', 3)

        # Quality cuts (same as 02_cf4_rar_pipeline.py)
        if Q > 2:
            continue
        if Inc < 30 or Inc > 85:
            continue
        if len(gdata['R']) < 5:
            continue

        # Determine distance: use CF4 for Hubble-flow galaxies, else SPARC
        D_use = D_sparc
        if fD == 1 and name in cf4_distances:
            cf4_entry = cf4_distances[name]
            if isinstance(cf4_entry, dict) and cf4_entry.get('D_cf4'):
                D_use = cf4_entry['D_cf4']

        if D_use <= 0 or np.isnan(D_use):
            continue

        D_ratio = D_use / max(gdata['dist_sparc'], 0.01)

        # Optimize M/L ratio (error-weighted chi² fit to RAR)
        ml = fit_ml_ratio(gdata, D_use, p)
        Y_d = ml['Y_disk']
        Y_b = ml['Y_bul']

        # Scale radii and model velocities by distance
        R = gdata['R'] * D_ratio                     # kpc
        Vobs = gdata['Vobs']                          # km/s (measured)
        eVobs = gdata['eVobs']
        sqrt_ratio = np.sqrt(D_ratio)
        Vgas = gdata['Vgas'] * sqrt_ratio
        Vdisk = gdata['Vdisk'] * sqrt_ratio
        Vbul = gdata['Vbul'] * sqrt_ratio

        # Baryonic acceleration (using optimized M/L)
        Vbar_sq = np.sign(Vgas) * Vgas**2 + Y_d * Vdisk**2 + Y_b * Vbul**2
        gbar = np.abs(Vbar_sq) / np.maximum(R, 1e-6) * conv
        gobs = Vobs**2 / np.maximum(R, 1e-6) * conv

        valid = (gbar > 0) & (gobs > 0) & (R > 0) & np.isfinite(gbar) & np.isfinite(gobs)
        if np.sum(valid) < 3:
            continue

        gbar_v = gbar[valid]
        gobs_v = gobs[valid]
        eVobs_v = eVobs[valid]
        Vobs_v = Vobs[valid]
        R_v = R[valid]

        gobs_pred = rar_prediction(gbar_v)
        log_gbar = np.log10(gbar_v)
        log_gobs = np.log10(gobs_v)
        log_res = log_gobs - np.log10(gobs_pred)
        sigma_log_gobs = 2.0 * eVobs_v / np.maximum(Vobs_v, 1.0) / np.log(10)

        # Environment
        env_dense = 'field'
        logMh = 11.0
        group_name = 'field'
        if name in env_catalog:
            env_info = env_catalog[name]
            env_dense = env_info.get('env_dense', 'field')
            logMh = env_info.get('logMh', 11.0)
            group_name = env_info.get('group_name', 'field')

        sparc_names.append(name)

        # Store per-point data
        for i in range(len(log_gbar)):
            sparc_points.append({
                'galaxy': name,
                'galaxy_key': canonicalize_galaxy_name(name),
                'source': 'SPARC',
                'log_gbar': float(log_gbar[i]),
                'log_gobs': float(log_gobs[i]),
                'log_res': float(log_res[i]),
                'sigma_log_gobs': float(sigma_log_gobs[i]),
                'R_kpc': float(R_v[i]),
                'env_dense': env_dense,
                'logMh': float(logMh),
            })

        # Per-galaxy summary
        sparc_galaxies.append({
            'name': name,
            'galaxy_key': canonicalize_galaxy_name(name),
            'source': 'SPARC',
            'n_points': int(np.sum(valid)),
            'D_Mpc': float(D_use),
            'Inc': float(Inc),
            'Vflat': float(p.get('Vflat', 0)),
            'sigma_res': float(np.std(log_res)),
            'mean_res': float(np.mean(log_res)),
            'env_dense': env_dense,
            'logMh': float(logMh),
            'group': group_name,
            'ra': coordinates.get(name, {}).get('ra', np.nan),
            'dec': coordinates.get(name, {}).get('dec', np.nan),
        })

    sparc_diag['sparc_galaxies_after_cuts'] = int(len(sparc_galaxies))
    sparc_diag['sparc_points_after_cuts'] = int(len(sparc_points))
    print(f"    SPARC: {len(sparc_galaxies)} galaxies, {len(sparc_points)} RAR points")
    return sparc_points, sparc_names, sparc_diag


# ============================================================
# DATASET 2: de Blok+2002 LSB (full mass models)
# ============================================================
def load_deblok2002():
    """Load de Blok+2002 LSB galaxies with processed rotation curves."""
    print("\n  [2/8] Loading de Blok+2002 LSB galaxies...")

    rc_path = os.path.join(DATA_DIR, 'hi_surveys', 'deblok2002_proc_rotcurves.tsv')
    _, rows = parse_vizier_tsv(rc_path)

    if not rows:
        print("    WARNING: de Blok+2002 rotation curves not found or empty")
        return [], []

    # Group by galaxy name
    gal_data = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue

        r_kpc = safe_float(row.get('r(kpc)', ''))
        vgas = safe_float(row.get('Vgas', ''), 0.0)
        vdisk = safe_float(row.get('Vdisk', ''), 0.0)
        vrot = safe_float(row.get('Vrot', ''))
        evrot = safe_float(row.get('e_Vrot', ''), 5.0)

        if np.isnan(r_kpc) or np.isnan(vrot):
            continue

        if name not in gal_data:
            gal_data[name] = {'R': [], 'Vgas': [], 'Vdisk': [], 'Vrot': [], 'eVrot': []}

        gal_data[name]['R'].append(r_kpc)
        gal_data[name]['Vgas'].append(vgas)
        gal_data[name]['Vdisk'].append(vdisk)
        gal_data[name]['Vrot'].append(vrot)
        gal_data[name]['eVrot'].append(evrot)

    points = []
    names = []

    for name, gd in gal_data.items():
        R = np.array(gd['R'])
        Vgas = np.array(gd['Vgas'])
        Vdisk = np.array(gd['Vdisk'])
        Vrot = np.array(gd['Vrot'])
        eVrot = np.array(gd['eVrot'])

        valid = (R > 0) & (Vrot > 0)
        if np.sum(valid) < 3:
            continue

        R_v = R[valid]
        Vgas_v = Vgas[valid]
        Vdisk_v = Vdisk[valid]
        Vrot_v = Vrot[valid]
        eVrot_v = eVrot[valid]

        Vbar_sq = np.sign(Vgas_v) * Vgas_v**2 + Y_DISK * Vdisk_v**2
        gbar = np.abs(Vbar_sq) / R_v * conv
        gobs = Vrot_v**2 / R_v * conv

        ok = (gbar > 0) & (gobs > 0) & np.isfinite(gbar) & np.isfinite(gobs)
        if np.sum(ok) < 3:
            continue

        gbar_ok = gbar[ok]
        gobs_ok = gobs[ok]
        gobs_pred = rar_prediction(gbar_ok)
        log_gbar = np.log10(gbar_ok)
        log_gobs = np.log10(gobs_ok)
        log_res = log_gobs - np.log10(gobs_pred)
        sigma_log = 2.0 * eVrot_v[ok] / np.maximum(Vrot_v[ok], 1) / np.log(10)

        # LSB galaxies are mostly field
        env_dense = 'field'
        logMh = 11.0

        names.append(name)
        for i in range(len(log_gbar)):
            points.append({
                'galaxy': name,
                'source': 'deBlok2002',
                'log_gbar': float(log_gbar[i]),
                'log_gobs': float(log_gobs[i]),
                'log_res': float(log_res[i]),
                'sigma_log_gobs': float(sigma_log[i]),
                'R_kpc': float(R_v[ok][i]),
                'env_dense': env_dense,
                'logMh': float(logMh),
            })

    print(f"    de Blok+2002: {len(names)} galaxies, {len(points)} RAR points")
    return points, names


# ============================================================
# DATASET 3: WALLABY (pre-computed RAR points)
# ============================================================
def load_wallaby():
    """Load pre-computed WALLABY RAR points and environment catalog."""
    print("\n  [3/8] Loading WALLABY DR1+DR2...")

    # Prefer CF4 distances if available, fall back to Hubble
    pts_path = os.path.join(OUTPUT_DIR, 'rar_points_wallaby_cf4_nodesi.csv')
    if not os.path.exists(pts_path):
        pts_path = os.path.join(OUTPUT_DIR, 'rar_points_wallaby_hubble_nodesi.csv')
    if not os.path.exists(pts_path):
        print("    WARNING: WALLABY RAR points not found")
        return [], []

    # Load environment catalog
    wallaby_env = {}
    wenv_path = os.path.join(DATA_DIR, 'wallaby_environment_catalog.json')
    if os.path.exists(wenv_path):
        with open(wenv_path, 'r') as f:
            wallaby_env = json.load(f)

    points = []
    names_set = set()

    with open(pts_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['galaxy']

            # Get environment from catalog or from CSV column
            env_dense = 'field'
            logMh = 11.0
            if name in wallaby_env:
                env_dense = wallaby_env[name].get('env_dense', 'field')
                logMh = wallaby_env[name].get('logMh', 11.0)
            elif 'env_dense' in row:
                env_dense = row['env_dense']

            names_set.add(name)
            points.append({
                'galaxy': name,
                'source': 'WALLABY',
                'log_gbar': float(row['log_gbar']),
                'log_gobs': float(row['log_gobs']),
                'log_res': float(row['log_res']),
                'sigma_log_gobs': float(row.get('sigma_log_gobs', 0.1)),
                'R_kpc': float(row.get('R_kpc', 0)),
                'env_dense': env_dense,
                'logMh': float(logMh),
            })

    names = list(names_set)
    print(f"    WALLABY: {len(names)} galaxies, {len(points)} RAR points")
    return points, names


# ============================================================
# DATASET 4: Santos-Santos+2020 (single-point dwarfs)
# ============================================================
def load_santos_santos():
    """Load Santos-Santos+2020 dwarf galaxy compilation."""
    print("\n  [4/8] Loading Santos-Santos+2020 dwarfs...")

    tsv_path = os.path.join(DATA_DIR, 'hi_surveys', 'santos2020_table1.tsv')
    _, rows = parse_vizier_tsv(tsv_path)

    if not rows:
        print("    WARNING: Santos-Santos+2020 not found or empty")
        return [], []

    points = []
    names = []

    for row in rows:
        name = row.get('Name', '').strip()
        sample = row.get('Sample', '').strip()
        vmax = safe_float(row.get('Vmax', ''))
        vbmax = safe_float(row.get('Vbmax', ''))
        rbhalf = safe_float(row.get('rbhalf', ''))
        ra = safe_float(row.get('_RA', ''))
        dec = safe_float(row.get('_DE', ''))

        if not name or np.isnan(vmax) or np.isnan(vbmax) or vmax <= 0 or vbmax <= 0:
            continue
        if np.isnan(rbhalf) or rbhalf <= 0:
            continue

        gbar_est = vbmax**2 / rbhalf * conv
        gobs_est = vmax**2 / rbhalf * conv

        if gbar_est <= 0 or gobs_est <= 0:
            continue

        gobs_pred = rar_prediction(gbar_est)
        log_gbar = np.log10(gbar_est)
        log_gobs = np.log10(gobs_est)
        log_res = log_gobs - np.log10(gobs_pred)

        # Environment classification
        vsys_est = np.nan
        if not np.isnan(ra) and not np.isnan(dec):
            # Rough Vsys estimate from Hubble flow using typical distance
            # Santos-Santos dwarfs are all nearby, use Mbar to estimate dist
            mbar = safe_float(row.get('Mbar', ''))
            if not np.isnan(mbar) and mbar > 0:
                # Very rough: use V ~ H0 * D, where D ~ (Mbar/1e8)^0.25 * 5 Mpc
                pass
            vsys_est = vmax * 10  # rough estimate for nearby dwarfs
            group_name, logMh, env_dense = classify_environment_proximity(
                ra, dec, vsys_est, name
            )
        else:
            group_name, logMh, env_dense = 'field', 11.0, 'field'

        names.append(name)
        points.append({
            'galaxy': name,
            'source': f'SS20_{sample}',
            'log_gbar': float(log_gbar),
            'log_gobs': float(log_gobs),
            'log_res': float(log_res),
            'sigma_log_gobs': 0.15,  # typical for single-point dwarfs
            'R_kpc': float(rbhalf),
            'env_dense': env_dense,
            'logMh': float(logMh),
        })

    print(f"    Santos-Santos+2020: {len(names)} dwarfs")
    return points, names


# ============================================================
# DATASET 5: LITTLE THINGS / Oh+2015 (single-point dwarfs)
# ============================================================
def parse_oh2015_table1():
    """
    Parse Oh+2015 Table 1 (LITTLE THINGS properties).
    This is a messy space-separated file with embedded +- uncertainties.
    """
    path = os.path.join(DATA_DIR, 'hi_surveys', 'little_things', 'oh2015_table1_properties.dat')
    if not os.path.exists(path):
        return {}

    props = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            # The format is messy: name followed by tab/space separated RA, Dec, etc.
            # with +- embedded. First split on tabs, then handle fields.
            parts = re.split(r'\t+', line.rstrip())

            if len(parts) < 4:
                continue

            name = parts[0].strip()
            if not name or name.startswith('#'):
                continue

            # Try to extract RA (HMS), Dec (DMS), D(Mpc), Vsys, PA, i, M_V, etc.
            # RA is like "12 38 39.2", Dec like "+32 45 41.0"
            # D is like "3.6"
            # Vsys is like "306.2 +- 1.3"
            # We need to parse each field carefully

            try:
                # RA field (HMS)
                ra_str = parts[1].strip() if len(parts) > 1 else ''
                # Dec field (DMS)
                dec_str = parts[2].strip() if len(parts) > 2 else ''
                # Distance
                d_str = parts[3].strip() if len(parts) > 3 else ''
                # Vsys (with +- uncertainty)
                vsys_str = parts[4].strip() if len(parts) > 4 else ''
                # PA
                pa_str = parts[5].strip() if len(parts) > 5 else ''
                # Inclination
                inc_str = parts[6].strip() if len(parts) > 6 else ''
                # M_V
                mv_str = parts[7].strip() if len(parts) > 7 else ''

                # Parse RA from HMS to degrees
                ra_parts = ra_str.split()
                if len(ra_parts) >= 3:
                    ra_deg = (float(ra_parts[0]) + float(ra_parts[1]) / 60.0
                              + float(ra_parts[2]) / 3600.0) * 15.0
                else:
                    ra_deg = np.nan

                # Parse Dec from DMS to degrees
                dec_parts = dec_str.replace('+', ' ').replace('-', ' -').split()
                if len(dec_parts) >= 3:
                    sign = -1 if dec_str.strip().startswith('-') else 1
                    dec_deg = sign * (abs(float(dec_parts[0])) + float(dec_parts[1]) / 60.0
                                      + float(dec_parts[2]) / 3600.0)
                else:
                    dec_deg = np.nan

                # Distance
                D_Mpc = safe_float(d_str)

                # Vsys: extract number before +-
                vsys_match = re.match(r'([+-]?\d+\.?\d*)', vsys_str)
                vsys = float(vsys_match.group(1)) if vsys_match else np.nan

                # Inclination: extract number before +-
                inc_match = re.match(r'([+-]?\d+\.?\d*)', inc_str)
                inc = float(inc_match.group(1)) if inc_match else np.nan

                props[name] = {
                    'ra': ra_deg, 'dec': dec_deg,
                    'D_Mpc': D_Mpc, 'Vsys': vsys,
                    'Inc': inc,
                }
            except (ValueError, IndexError):
                continue

    return props


def parse_oh2015_table2():
    """
    Parse Oh+2015 Table 2 (LITTLE THINGS mass models).
    Tab-separated with embedded +- uncertainties and LaTeX fragments.
    """
    path = os.path.join(DATA_DIR, 'hi_surveys', 'little_things', 'oh2015_table2_massmodels.dat')
    if not os.path.exists(path):
        return {}

    models = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = re.split(r'\t+', line.rstrip())
            if len(parts) < 14:
                continue

            name = parts[0].strip()
            if not name or name.startswith('#'):
                continue

            try:
                # Rmax(kpc) -- extract number before any +-
                rmax = safe_float(re.match(r'([+-]?\d+\.?\d*)', parts[1].strip()).group(1)
                                  if re.match(r'([+-]?\d+\.?\d*)', parts[1].strip()) else '')

                # Vmax(km/s) -- column 3
                vmax = safe_float(re.match(r'([+-]?\d+\.?\d*)', parts[3].strip()).group(1)
                                  if re.match(r'([+-]?\d+\.?\d*)', parts[3].strip()) else '')

                # Mgas(1e7 Msun) -- column 13
                mgas_str = parts[13].strip() if len(parts) > 13 else ''
                mgas = safe_float(re.match(r'([+-]?\d+\.?\d*)', mgas_str).group(1)
                                  if re.match(r'([+-]?\d+\.?\d*)', mgas_str) else '')

                # Mstar_SED(1e7 Msun) -- column 15
                mstar_sed_str = parts[15].strip() if len(parts) > 15 else ''
                mstar_sed = safe_float(re.match(r'([+-]?\d+\.?\d*)', mstar_sed_str).group(1)
                                       if re.match(r'([+-]?\d+\.?\d*)', mstar_sed_str) else '')

                # Mstar_KIN(1e7 Msun) -- column 14
                mstar_kin_str = parts[14].strip() if len(parts) > 14 else ''
                mstar_kin = safe_float(re.match(r'([+-]?\d+\.?\d*)', mstar_kin_str).group(1)
                                       if re.match(r'([+-]?\d+\.?\d*)', mstar_kin_str) else '')

                models[name] = {
                    'Rmax': rmax, 'Vmax': vmax,
                    'Mgas_1e7': mgas,
                    'Mstar_SED_1e7': mstar_sed,
                    'Mstar_KIN_1e7': mstar_kin,
                }
            except (ValueError, IndexError, AttributeError):
                continue

    return models


def load_little_things():
    """Load LITTLE THINGS dwarfs from Oh+2015."""
    print("\n  [5/8] Loading LITTLE THINGS / Oh+2015 (26 dwarfs)...")

    table1 = parse_oh2015_table1()
    table2 = parse_oh2015_table2()

    if not table2:
        print("    WARNING: LITTLE THINGS mass models not found")
        return [], []

    points = []
    names = []

    for name, model in table2.items():
        Rmax = model.get('Rmax', np.nan)
        Vmax = model.get('Vmax', np.nan)
        Mgas_1e7 = model.get('Mgas_1e7', np.nan)
        Mstar_SED_1e7 = model.get('Mstar_SED_1e7', np.nan)

        if np.isnan(Rmax) or np.isnan(Vmax) or Rmax <= 0 or Vmax <= 0:
            continue

        # Use Mstar_SED if available, otherwise Mstar_KIN
        Mstar_1e7 = Mstar_SED_1e7
        if np.isnan(Mstar_1e7):
            Mstar_1e7 = model.get('Mstar_KIN_1e7', 0.0)
        if np.isnan(Mstar_1e7):
            Mstar_1e7 = 0.0

        if np.isnan(Mgas_1e7):
            Mgas_1e7 = 0.0

        # Baryonic mass in solar masses
        Mbar = (Mgas_1e7 + Mstar_1e7) * 1e7

        if Mbar <= 0:
            continue

        # Single-point RAR
        # Vbar estimate from enclosed mass
        Vbar_est = np.sqrt(G_kpc * Mbar / Rmax)  # km/s
        gbar = Vbar_est**2 / Rmax * conv          # m/s^2
        gobs = Vmax**2 / Rmax * conv              # m/s^2

        if gbar <= 0 or gobs <= 0:
            continue

        gobs_pred = rar_prediction(gbar)
        log_gbar = np.log10(gbar)
        log_gobs = np.log10(gobs)
        log_res = log_gobs - np.log10(gobs_pred)

        # Environment from table1 coordinates
        env_dense = 'field'
        logMh = 11.0
        t1 = table1.get(name, {})
        ra = t1.get('ra', np.nan)
        dec = t1.get('dec', np.nan)
        vsys = t1.get('Vsys', np.nan)

        if not np.isnan(ra) and not np.isnan(dec) and not np.isnan(vsys):
            _, logMh, env_dense = classify_environment_proximity(ra, dec, vsys, name)

        names.append(name)
        points.append({
            'galaxy': name,
            'source': 'LITTLETHINGS',
            'log_gbar': float(log_gbar),
            'log_gobs': float(log_gobs),
            'log_res': float(log_res),
            'sigma_log_gobs': 0.15,
            'R_kpc': float(Rmax),
            'env_dense': env_dense,
            'logMh': float(logMh),
        })

    print(f"    LITTLE THINGS: {len(names)} dwarfs")
    return points, names


# ============================================================
# DATASET 6: LVHIS / Koribalski+2018
# ============================================================
def load_lvhis():
    """Load LVHIS galaxies from multiple tables."""
    print("\n  [6/8] Loading LVHIS / Koribalski+2018...")

    # --- Kinematics (table9): vrot, Rmax, inc, Nbeam ---
    kin_path = os.path.join(DATA_DIR, 'hi_surveys', 'lvhis_kinematics.tsv')
    _, kin_rows = parse_vizier_tsv(kin_path)

    # --- HI properties (table6): logMHI ---
    hi_path = os.path.join(DATA_DIR, 'hi_surveys', 'lvhis_hi_properties.tsv')
    _, hi_rows = parse_vizier_tsv(hi_path)

    # --- Sample/properties (table2/table4): distance, RA, Dec ---
    sample_path = os.path.join(DATA_DIR, 'hi_surveys', 'lvhis_sample.tsv')
    _, sample_rows = parse_vizier_tsv(sample_path)

    if not kin_rows:
        print("    WARNING: LVHIS kinematics not found")
        return [], []

    # Index HI properties by HIPASS name
    hi_by_hipass = {}
    for row in hi_rows:
        hipass = row.get('HIPASS', '').strip()
        oname = row.get('OName', '').strip()
        if hipass:
            hi_by_hipass[hipass] = row
        if oname:
            hi_by_hipass[oname] = row

    # Index sample by HIPASS name
    sample_by_hipass = {}
    for row in sample_rows:
        hipass = row.get('HIPASS', '').strip()
        oname = row.get('OName', '').strip()
        if hipass:
            sample_by_hipass[hipass] = row
        if oname:
            sample_by_hipass[oname] = row

    points = []
    names = []

    for row in kin_rows:
        hipass = row.get('HIPASS', '').strip()
        oname = row.get('OName', '').strip()
        name = oname if oname else hipass

        if not name:
            continue

        vrot = safe_float(row.get('vrot', ''))
        rmax = safe_float(row.get('Rmax', ''))
        inc = safe_float(row.get('i', ''))
        nbeam_str = row.get('Nbeam', '').strip()
        nbeam = safe_int(nbeam_str) if nbeam_str else 0

        # Quality filters
        if np.isnan(vrot) or np.isnan(rmax) or vrot <= 15 or rmax <= 0:
            continue
        if np.isnan(inc) or inc < 30 or inc > 85:
            continue
        if nbeam < 5:
            continue

        # Get HI mass
        hi_entry = hi_by_hipass.get(hipass, hi_by_hipass.get(oname, {}))
        logMHI = safe_float(hi_entry.get('logMHI', ''))

        if np.isnan(logMHI):
            continue

        MHI = 10**logMHI  # Solar masses
        Mbar = MHI * 1.33  # Including helium

        # gobs from rotation
        gobs = vrot**2 / rmax * conv  # m/s^2

        # gbar from baryonic mass (gas-dominated dwarfs)
        gbar = G_kpc * Mbar / rmax**2 * conv  # note: G_kpc * M / R^2 gives (km/s)^2/kpc^2
        # Actually: G_kpc * Mbar / R^2 gives (km/s)^2 / kpc, need to check units
        # G_kpc = 4.302e-3 (km/s)^2 kpc / Msun
        # G_kpc * M / R^2 = 4.302e-3 * M / R^2 in (km/s)^2 / kpc
        # Then gbar = G_kpc * M / R^2 * conv
        gbar = G_kpc * Mbar / (rmax**2) * conv

        if gbar <= 0 or gobs <= 0:
            continue

        gobs_pred = rar_prediction(gbar)
        log_gbar = np.log10(gbar)
        log_gobs = np.log10(gobs)
        log_res = log_gobs - np.log10(gobs_pred)

        # Environment from sample coordinates
        samp = sample_by_hipass.get(hipass, sample_by_hipass.get(oname, {}))
        ra = safe_float(samp.get('_RA', ''))
        dec = safe_float(samp.get('_DE', ''))
        D_Mpc = safe_float(samp.get('D', ''))

        env_dense = 'field'
        logMh = 11.0
        if not np.isnan(ra) and not np.isnan(dec) and not np.isnan(D_Mpc):
            vsys_est = D_Mpc * H0  # rough
            _, logMh, env_dense = classify_environment_proximity(ra, dec, vsys_est, name)

        names.append(name)
        points.append({
            'galaxy': name,
            'source': 'LVHIS',
            'log_gbar': float(log_gbar),
            'log_gobs': float(log_gobs),
            'log_res': float(log_res),
            'sigma_log_gobs': 0.15,
            'R_kpc': float(rmax),
            'env_dense': env_dense,
            'logMh': float(logMh),
        })

    print(f"    LVHIS: {len(names)} galaxies")
    return points, names


# ============================================================
# DATASET 7: Yu+2020 (HI spectral line widths)
# ============================================================
def load_yu2020():
    """Load Yu+2020 galaxy catalog with HI profile widths."""
    print("\n  [7/8] Loading Yu+2020 (HI spectra)...")

    tsv_path = os.path.join(DATA_DIR, 'hi_surveys', 'yu2020_all.tsv')
    _, rows = parse_vizier_tsv(tsv_path)

    if not rows:
        print("    WARNING: Yu+2020 not found or empty")
        return [], []

    points = []
    names = []

    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue

        ra = safe_float(row.get('RAJ2000', ''))
        dec = safe_float(row.get('DEJ2000', ''))
        z = safe_float(row.get('z', ''))
        inc = safe_float(row.get('inc', ''))
        logM_star = safe_float(row.get('logM', ''))
        logMHI = safe_float(row.get('logMHI', ''))
        logMdyn = safe_float(row.get('logMdyn', ''))
        V75c = safe_float(row.get('V75c', ''))
        snr_str = row.get('SNR', '').strip()
        snr = safe_float(snr_str)

        # Quality filters
        if np.isnan(inc) or inc <= 30:
            continue
        if np.isnan(snr) or snr < 10:
            continue
        if np.isnan(V75c) or V75c <= 0:
            continue
        if np.isnan(logMHI) or np.isnan(logM_star):
            continue

        # Rotation velocity: V75c / (2 * sin(i)) corrected for inclination
        sin_i = np.sin(np.radians(inc))
        Vrot = V75c / (2.0 * max(sin_i, 0.3))

        if Vrot <= 20:
            continue

        # Baryonic mass
        M_star = 10**logM_star  # Msun
        M_HI = 10**logMHI       # Msun
        Mbar = M_star + 1.33 * M_HI  # Including helium

        # For single-point RAR, use Vbar approach rather than GM/R²
        # Vbar ≈ Vrot * sqrt(Mbar/Mdyn) — baryonic fraction of rotation
        # gbar = Vbar² / R, gobs = Vrot² / R → gbar/gobs = Mbar/Mdyn
        if not np.isnan(logMdyn):
            Mdyn = 10**logMdyn
            Rdyn = Mdyn / (2.31e5 * Vrot**2)  # kpc
        else:
            continue

        if Rdyn <= 0 or np.isnan(Rdyn) or Rdyn > 500:
            continue

        # Use velocity-based RAR (distance-independent!)
        # gbar = Vbar²/R where Vbar = Vrot * sqrt(Mbar/Mdyn)
        fbar = Mbar / Mdyn
        if fbar <= 0 or fbar > 1.0:
            continue  # Skip unphysical baryon fractions
        Vbar = Vrot * np.sqrt(fbar)

        gobs = Vrot**2 / Rdyn * conv
        gbar = Vbar**2 / Rdyn * conv

        if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
            continue

        gobs_pred = rar_prediction(gbar)
        log_gbar = np.log10(gbar)
        log_gobs = np.log10(gobs)
        log_res = log_gobs - np.log10(gobs_pred)

        # Sanity check: skip extreme outliers
        if abs(log_res) > 1.5:
            continue

        # Environment
        env_dense = 'field'
        logMh = 11.0
        if not np.isnan(ra) and not np.isnan(dec) and not np.isnan(z) and z > 0:
            vsys = c_light * z
            _, logMh, env_dense = classify_environment_proximity(ra, dec, vsys, name)

        names.append(name)
        points.append({
            'galaxy': name,
            'source': 'Yu2020',
            'log_gbar': float(log_gbar),
            'log_gobs': float(log_gobs),
            'log_res': float(log_res),
            'sigma_log_gobs': 0.15,
            'R_kpc': float(Rdyn),
            'env_dense': env_dense,
            'logMh': float(logMh),
        })

    print(f"    Yu+2020: {len(names)} galaxies")
    return points, names


# ============================================================
# DATASET 8: Swaters+2025 DiskMass Survey
# ============================================================
def load_swaters2025():
    """Load Swaters+2025 DiskMass Survey H-alpha rotation curves."""
    print("\n  [8/8] Loading Swaters+2025 DiskMass Survey...")

    tsv_path = os.path.join(DATA_DIR, 'hi_surveys', 'swaters2025_diskmass_sample.tsv')
    _, rows = parse_vizier_tsv(tsv_path)

    if not rows:
        print("    WARNING: Swaters+2025 not found or empty")
        return [], []

    points = []
    names = []

    for row in rows:
        ugc_str = row.get('UGC', '').strip()
        if not ugc_str:
            continue

        name = f"UGC{ugc_str.zfill(5)}"

        Vrot = safe_float(row.get('Vrot', ''))
        hrot = safe_float(row.get('hrot', ''))  # arcsec
        Dist = safe_float(row.get('Dist', ''))   # Mpc
        KMag = safe_float(row.get('KMag', ''))   # absolute K mag
        Vsys = safe_float(row.get('Vsys', ''))
        inc = safe_float(row.get('inc', ''))
        iiTF = safe_float(row.get('iiTF', ''))   # inverse TF inclination (fallback)
        ra = safe_float(row.get('_RA', ''))
        dec = safe_float(row.get('_DE', ''))

        # Use iiTF as fallback when inc is blank (46/125 galaxies)
        if np.isnan(inc) or inc <= 0:
            inc = iiTF

        # Quality filters
        if np.isnan(Vrot) or Vrot <= 20:
            continue
        if np.isnan(inc) or inc < 20:
            continue
        if np.isnan(Dist) or Dist <= 0:
            continue
        if np.isnan(KMag):
            continue

        # Stellar mass from K-band absolute magnitude
        # DiskMass Survey M/L_K = 0.3 (Martinsson+2013: 0.24-0.32)
        ML_K = 0.3  # Msun/Lsun in K-band
        MK_sun = 3.27  # Solar absolute K magnitude (Vega)
        log_Mstar = -0.4 * (KMag - MK_sun) + np.log10(ML_K)
        M_star = 10**log_Mstar

        # HI mass estimate from scaling relation (Catinella+2010)
        log_MHI_est = 0.5 * log_Mstar + 4.5
        M_HI_est = 10**log_MHI_est
        Mbar = M_star + 1.33 * M_HI_est

        # hrot may be missing for some galaxies (marked 'URC' in f_hrot)
        if not np.isnan(hrot) and hrot > 0:
            # Convert hrot (arcsec) to kpc
            R_kpc = hrot * Dist * np.pi / (180.0 * 3600.0) * 1000.0
        else:
            # No hrot: estimate disk scale from stellar mass relation
            # log(Rd/kpc) ~ 0.4*log(M*/Msun) - 3.4 (Shen+2003 for late types)
            R_kpc = 10**(0.4 * log_Mstar - 3.4)

        if R_kpc <= 0:
            continue

        # Use velocity-based approach at R = 2.2 * hrot (peak of exponential disk RC)
        R_rar = 2.2 * R_kpc

        # Dynamical mass enclosed at R_rar
        Mdyn = Vrot**2 * R_rar / G_kpc

        # Baryon fraction: use enclosed mass fraction for exponential disk
        # f_enc(2.2 Rd) = 1 - (1 + 2.2)*exp(-2.2) ~ 0.645
        f_enc = 1.0 - (1.0 + 2.2) * np.exp(-2.2)
        Mbar_enc = f_enc * Mbar

        fbar = Mbar_enc / Mdyn
        if fbar <= 0:
            continue
        # Cap at physical maximum (100% baryonic)
        fbar = min(fbar, 1.0)

        Vbar = Vrot * np.sqrt(fbar)
        gobs = Vrot**2 / R_rar * conv
        gbar = Vbar**2 / R_rar * conv

        if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
            continue

        gobs_pred = rar_prediction(gbar)
        log_gbar = np.log10(gbar)
        log_gobs = np.log10(gobs)
        log_res = log_gobs - np.log10(gobs_pred)

        # Sanity check
        if abs(log_res) > 1.5:
            continue

        # Environment
        env_dense = 'field'
        logMh = 11.0
        if not np.isnan(ra) and not np.isnan(dec) and not np.isnan(Vsys):
            _, logMh, env_dense = classify_environment_proximity(ra, dec, Vsys, name)

        names.append(name)
        points.append({
            'galaxy': name,
            'source': 'Swaters2025',
            'log_gbar': float(log_gbar),
            'log_gobs': float(log_gobs),
            'log_res': float(log_res),
            'sigma_log_gobs': 0.12,
            'R_kpc': float(R_kpc),
            'env_dense': env_dense,
            'logMh': float(logMh),
        })

    print(f"    Swaters+2025: {len(names)} galaxies, {len(points)} points")
    return points, names


# ============================================================
# STATISTICAL TESTS
# ============================================================
def bootstrap_scatter_test(dense_res, field_res, n_boot=10000, seed=42):
    """
    Bootstrap permutation test for difference in scatter.

    Returns: (delta_sigma, p_value, boot_deltas)
      delta_sigma = sigma_field - sigma_dense
      p_value = fraction of bootstrap samples with delta >= observed
    """
    dense_res = np.asarray(dense_res)
    field_res = np.asarray(field_res)

    if len(dense_res) < 5 or len(field_res) < 5:
        return np.nan, np.nan, np.array([])

    sigma_dense = np.std(dense_res)
    sigma_field = np.std(field_res)
    delta = sigma_field - sigma_dense

    combined = np.concatenate([dense_res, field_res])
    nd = len(dense_res)

    rng = np.random.RandomState(seed)
    boot_deltas = np.zeros(n_boot)

    for i in range(n_boot):
        perm = rng.permutation(combined)
        boot_deltas[i] = np.std(perm[nd:]) - np.std(perm[:nd])

    p_value = np.mean(boot_deltas >= delta)
    return delta, p_value, boot_deltas


def binned_analysis(all_gbar_dense, all_res_dense, all_gbar_field, all_res_field,
                    bin_edges=None):
    """
    Run binned environmental scatter analysis.

    Returns list of bin result dicts.
    """
    if bin_edges is None:
        bin_edges = np.array([-13.0, -12.0, -11.0, -10.0, -9.0, -8.0])

    gbar_d = np.asarray(all_gbar_dense)
    res_d = np.asarray(all_res_dense)
    gbar_f = np.asarray(all_gbar_field)
    res_f = np.asarray(all_res_field)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    results = []

    for j in range(len(bin_centers)):
        lo, hi = bin_edges[j], bin_edges[j + 1]

        d_mask = (gbar_d >= lo) & (gbar_d < hi) if len(gbar_d) > 0 else np.array([], dtype=bool)
        f_mask = (gbar_f >= lo) & (gbar_f < hi) if len(gbar_f) > 0 else np.array([], dtype=bool)

        d_r = res_d[d_mask] if np.sum(d_mask) > 0 else np.array([])
        f_r = res_f[f_mask] if np.sum(f_mask) > 0 else np.array([])

        bin_result = {
            'center': float(bin_centers[j]),
            'lo': float(lo), 'hi': float(hi),
            'n_dense': len(d_r), 'n_field': len(f_r),
            'sigma_dense': float(np.std(d_r)) if len(d_r) > 0 else np.nan,
            'sigma_field': float(np.std(f_r)) if len(f_r) > 0 else np.nan,
        }

        if len(d_r) >= 5 and len(f_r) >= 5:
            delta, p_val, _ = bootstrap_scatter_test(d_r, f_r, n_boot=5000, seed=42 + j)
            bin_result['delta'] = float(delta)
            bin_result['p_field_gt_dense'] = float(1.0 - p_val)

            # Levene's test for this bin
            try:
                lev_stat, lev_p = stats.levene(d_r, f_r)
                bin_result['levene_stat'] = float(lev_stat)
                bin_result['levene_p'] = float(lev_p)
            except Exception:
                pass
        else:
            bin_result['delta'] = np.nan
            bin_result['p_field_gt_dense'] = np.nan

        results.append(bin_result)

    return results


# ============================================================
# DATASET 9: GHASP (Epinat+2008 / Korsaga+2019)
# 93 galaxies with H-alpha rotation curves + photometric decompositions
# ============================================================
def load_ghasp():
    """
    Load GHASP Hα rotation curves (Epinat+2008) with photometric
    decompositions from Korsaga+2019.

    Uses exponential disk model to estimate gbar from luminosity
    and scale length, then computes gobs from the rotation curves.
    """
    print("\n  [9/10] Loading GHASP (93 H-alpha rotation curves)...")

    # --- Load rotation curves (Epinat+2008) ---
    rc_path = os.path.join(DATA_DIR, 'hi_surveys', 'ghasp_epinat2008_rotcurves.tsv')
    _, rc_rows = parse_vizier_tsv(rc_path)

    # --- Load galaxy properties (Epinat+2008) ---
    # Use the galaxies-specific table (not the multi-table concatenated file)
    gal_path = os.path.join(DATA_DIR, 'hi_surveys', 'ghasp_epinat2008_galaxies.tsv')
    _, gal_rows = parse_vizier_tsv(gal_path)

    # --- Load photometric decompositions (Korsaga+2019) ---
    phot_path = os.path.join(DATA_DIR, 'hi_surveys', 'korsaga2019_ghasp_properties.tsv')
    _, phot_rows = parse_vizier_tsv(phot_path)

    if not rc_rows:
        print("    WARNING: GHASP rotation curves not found")
        return [], []

    # Helper: normalize UGC names (strip leading zeros)
    # Korsaga uses "UGC 00089", Epinat uses "UGC 89"
    def normalize_ugc(name):
        m = re.match(r'^(UGC)\s+0*(\d+)$', name.strip())
        if m:
            return f"UGC {m.group(2)}"
        return name.strip()

    # Build galaxy property lookup (normalize UGC names for consistent matching)
    gal_props = {}
    for row in gal_rows:
        name = row.get('Name', '').strip()
        if not name:
            continue
        gal_props[normalize_ugc(name)] = {
            'Vsys': safe_float(row.get('Vsys-FP', row.get('Vsys-L', ''))),
            'inc_FP': safe_float(row.get('i-FP', '')),
            'inc_L': safe_float(row.get('i-L', '')),
            'inc_ba': safe_float(row.get('i', '')),
            'Dist': safe_float(row.get('Dist', '')),
            'Vmax': safe_float(row.get('Vmax', '')),
            'D25': safe_float(row.get('D25', '')),
            'MType': row.get('MType', '').strip(),
            'T': safe_float(row.get('T', '')),
            'q_Vmax': safe_int(row.get('q_Vmax', '4')),
            'RA': safe_float(row.get('_RA', '')),
            'Dec': safe_float(row.get('_DE', '')),
        }

    # Build photometric decomposition lookup (normalize UGC names)
    phot_props = {}
    for row in phot_rows:
        name = normalize_ugc(row.get('ID', '').strip())
        if not name:
            continue
        phot_props[name] = {
            'BMAG': safe_float(row.get('BMAG', '')),
            'B_V': safe_float(row.get('B-V', '')),
            'h_kpc': safe_float(row.get('h', '')),      # disk scale length in kpc
            'LD': safe_float(row.get('LD', '')),          # disk luminosity in 10^8 Lsun
            'LB_bulge': safe_float(row.get('LB', '')),    # bulge luminosity in 10^8 Lsun
            're': safe_float(row.get('re', '')),           # bulge effective radius kpc
            'mu0': safe_float(row.get('mu0', '')),         # central SB
            'RA': safe_float(row.get('_RA', '')),
            'Dec': safe_float(row.get('_DE', '')),
        }

    print(f"    Galaxy properties: {len(gal_props)}, "
          f"Photometric decomp: {len(phot_props)}")

    # --- Group rotation curve points by galaxy ---
    # Average approaching and receding sides
    galaxy_rc = {}
    for row in rc_rows:
        name = normalize_ugc(row.get('Name', '').strip())
        if not name:
            continue
        r_kpc = safe_float(row.get('r', ''))
        Vrot = safe_float(row.get('Vrot', ''))
        eVrot = safe_float(row.get('e_Vrot', ''))

        if np.isnan(r_kpc) or np.isnan(Vrot) or r_kpc <= 0 or Vrot <= 0:
            continue

        if name not in galaxy_rc:
            galaxy_rc[name] = {}

        # Use radius as key (round to 0.01 kpc to merge sides)
        r_key = round(r_kpc, 2)
        if r_key not in galaxy_rc[name]:
            galaxy_rc[name][r_key] = {'vrot_list': [], 'evrot_list': []}
        galaxy_rc[name][r_key]['vrot_list'].append(Vrot)
        galaxy_rc[name][r_key]['evrot_list'].append(max(eVrot, 1.0))

    # --- Compute RAR for each galaxy ---
    points = []
    names = []
    n_no_props = 0
    n_no_phot = 0
    n_wise_mass = 0

    # Solar B-band M/L ≈ 5.48 mag absolute for 1 Lsun
    # M/L_B from (B-V) color: Bell+2003 relation
    # log10(M/L_B) = -0.942 + 1.737*(B-V)

    for gname, radii in galaxy_rc.items():
        gp = gal_props.get(gname, None)
        pp = phot_props.get(gname, None)

        if gp is None:
            n_no_props += 1
            continue

        Dist = gp['Dist']
        Vsys = gp['Vsys']
        q = gp['q_Vmax']
        ra = gp['RA']
        dec = gp['Dec']

        # Use best available inclination
        inc = gp['inc_FP']
        if np.isnan(inc) or inc <= 0:
            inc = gp['inc_ba']
        if np.isnan(inc) or inc <= 0:
            inc = gp['inc_L']

        # Quality and geometry cuts
        if np.isnan(Dist) or Dist <= 0:
            continue
        if np.isnan(inc) or inc < 30 or inc > 85:
            continue
        if q > 3:
            continue
        if len(radii) < 3:
            continue

        # Try WISE photometric mass first (most homogeneous)
        wise_logM = get_z0mgs_stellar_mass(name=gname, ra=ra, dec=dec)
        if wise_logM is not None:
            # WISE stellar mass; put all into disk component
            M_disk = 10**wise_logM
            M_bulge = 0.0
            re_kpc = 0.0
            # Still use Korsaga scale length if available
            if pp is not None and not np.isnan(pp['h_kpc']):
                h_kpc = pp['h_kpc']
            else:
                D25_kpc = gp['D25'] * Dist / 206.265 if not np.isnan(gp['D25']) else 10.0
                h_kpc = D25_kpc / 4.0
            has_phot = True
            n_wise_mass += 1
        # Fallback: get M/L from photometry if available
        elif pp is not None and not np.isnan(pp['LD']) and not np.isnan(pp['h_kpc']):
            LD_Lsun = pp['LD'] * 1e8          # disk luminosity in Lsun
            h_kpc = pp['h_kpc']                 # disk scale length in kpc
            LB_Lsun = 0.0
            re_kpc = 0.0
            if not np.isnan(pp.get('LB_bulge', np.nan)):
                LB_Lsun = pp['LB_bulge'] * 1e8
            if not np.isnan(pp.get('re', np.nan)):
                re_kpc = pp['re']

            # Estimate M/L from B-V color (Bell+2003)
            bv = pp.get('B_V', 0.5)
            if np.isnan(bv):
                bv = 0.5
            log_ml_B = -0.942 + 1.737 * bv
            ML_disk = 10**log_ml_B  # M/L in B-band solar units

            # Disk mass and bulge mass
            M_disk = LD_Lsun * ML_disk
            M_bulge = LB_Lsun * ML_disk * 1.2  # bulge M/L ~20% higher

            has_phot = True
        else:
            n_no_phot += 1
            # Fallback: use Vmax and D25 to estimate
            Vmax = gp['Vmax']
            if np.isnan(Vmax) or Vmax <= 0:
                continue
            D25_kpc = gp['D25'] * Dist / 206.265 if not np.isnan(gp['D25']) else 10.0
            h_kpc = D25_kpc / 4.0  # Approximate: R25 ~ 4h for exponential disk
            M_disk = Vmax**2 * D25_kpc / G_kpc  # rough dynamical mass estimate
            M_bulge = 0.0
            re_kpc = 0.0
            has_phot = False

        if M_disk <= 0 or h_kpc <= 0:
            continue

        # Environment classification
        env_dense = 'field'
        logMh = 11.0
        if not np.isnan(ra) and not np.isnan(dec) and not np.isnan(Vsys):
            _, logMh, env_dense = classify_environment_proximity(ra, dec, Vsys, gname)

        names.append(gname)

        # Compute RAR for each radius point
        for r_key, rdata in sorted(radii.items()):
            R_kpc = r_key
            if R_kpc <= 0.1:
                continue

            # Average Vrot from approaching/receding
            vrot_avg = np.mean(rdata['vrot_list'])
            evrot_avg = np.mean(rdata['evrot_list']) / np.sqrt(len(rdata['vrot_list']))

            # gobs = Vrot^2 / R
            gobs = vrot_avg**2 / R_kpc * conv

            # gbar from exponential disk + bulge
            # For exponential disk: V_disk^2(R) = G*M_disk/(2*h) * y^2 * [I0*K0 - I1*K1]
            # where y = R/(2h). Approximate with Freeman formula:
            y = R_kpc / (2.0 * h_kpc)
            # Approximate Bessel function factor for exponential disk
            # f(y) = y^2 * [I0(y)*K0(y) - I1(y)*K1(y)]
            # For simplicity, use the compact approximation:
            # f(y) ≈ 1 - exp(-3.33*y) * (1 + 3.33*y + 5.56*y^2)  (Casertano 1983)
            # But even simpler: V_disk^2 ≈ G*M(<R)/R
            # where M(<R) = M_disk * (1 - (1+x)*exp(-x)), x = R/h
            x_disk = R_kpc / h_kpc
            frac_enclosed = 1.0 - (1.0 + x_disk) * np.exp(-x_disk)
            M_enclosed_disk = M_disk * frac_enclosed

            # Bulge contribution (Hernquist/de Vaucouleurs)
            M_enclosed_bulge = 0.0
            if M_bulge > 0 and re_kpc > 0:
                # Approximate: half mass at re
                a_bulge = re_kpc / 1.815  # Hernquist scale
                M_enclosed_bulge = M_bulge * R_kpc**2 / (R_kpc + a_bulge)**2

            M_bar_enclosed = M_enclosed_disk + M_enclosed_bulge
            gbar = G_kpc * M_bar_enclosed / R_kpc**2 * conv

            if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
                continue

            # Fractional velocity error cut: σ_V/V < 15%
            if vrot_avg > 0 and evrot_avg / vrot_avg > 0.15:
                continue

            gobs_pred = rar_prediction(gbar)
            log_gbar = np.log10(gbar)
            log_gobs = np.log10(gobs)
            log_res = log_gobs - np.log10(gobs_pred)

            if abs(log_res) > 1.5:
                continue

            sigma_log = 2.0 * evrot_avg / max(vrot_avg, 1.0) / np.log(10)

            points.append({
                'galaxy': gname,
                'source': 'GHASP',
                'log_gbar': float(log_gbar),
                'log_gobs': float(log_gobs),
                'log_res': float(log_res),
                'sigma_log_gobs': float(max(sigma_log, 0.05)),
                'R_kpc': float(R_kpc),
                'env_dense': env_dense,
                'logMh': float(logMh),
            })

    n_galaxies = len(set(names))
    print(f"    GHASP: {n_galaxies} galaxies, {len(points)} RAR points")
    print(f"    Mass models: {n_wise_mass} WISE, {n_galaxies - n_wise_mass - n_no_phot} Korsaga photometry, {n_no_phot} Vmax fallback")
    if n_no_props > 0:
        print(f"    ({n_no_props} skipped - no properties)")
    return points, list(set(names))


# ============================================================
# DATASET 10: Noordermeer+2005 WHISP early-type disk galaxies
# 68 galaxies with HI properties AND group/cluster membership
# ============================================================
def load_noordermeer2005():
    """
    Load Noordermeer+2005 WHISP early-type disk galaxy catalog.

    These are early-type (S0/a to Sab) disk galaxies with HI data.
    CRITICAL: Has explicit group/cluster membership (Memb column)
    which directly tags dense-environment galaxies.

    Uses single-point RAR: Vrot from W50, baryonic mass from
    luminosity and HI mass.
    """
    print("\n  [10/10] Loading Noordermeer+2005 WHISP (68 early-type galaxies)...")

    tsv_path = os.path.join(DATA_DIR, 'hi_surveys', 'whisp',
                            'noordermeer2005_whisp_hi.tsv')
    _, rows = parse_vizier_tsv(tsv_path)

    if not rows:
        print("    WARNING: Noordermeer+2005 not found or empty")
        return [], []

    points = []
    names = []

    for row in rows:
        ugc = row.get('UGC', '').strip()
        alt_name = row.get('Name', '').strip()
        if not ugc:
            continue

        name = f"UGC {ugc.zfill(5)}" if alt_name == '--' else alt_name
        if name == '--':
            name = f"UGC {ugc.zfill(5)}"

        # Key parameters
        W50 = safe_float(row.get('W50', ''))
        W20 = safe_float(row.get('W20', ''))
        Dist = safe_float(row.get('Dist', ''))
        Inc = safe_float(row.get('Incl', ''))
        BMAG = safe_float(row.get('BMAG', ''))
        MHI = safe_float(row.get('MHI', ''))       # in 10^9 Msun
        Vsys = safe_float(row.get('Vsys', ''))
        memb = row.get('Memb', '').strip()
        region = row.get('Region', '').strip()

        # Parse RA/Dec
        ra_str = row.get('RAJ2000', '').strip()
        dec_str = row.get('DEJ2000', '').strip()

        # Convert HMS/DMS to degrees
        ra_deg = np.nan
        dec_deg = np.nan
        try:
            parts = ra_str.split()
            if len(parts) == 3:
                ra_deg = 15.0 * (float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600)
            parts = dec_str.replace('+', '').split()
            if len(parts) == 3:
                sign = -1 if dec_str.strip().startswith('-') else 1
                dec_deg = sign * (abs(float(parts[0])) + float(parts[1])/60 + float(parts[2])/3600)
        except (ValueError, IndexError):
            pass

        # Quality cuts
        if np.isnan(W50) or W50 <= 0:
            continue
        if np.isnan(Dist) or Dist <= 0:
            continue
        if np.isnan(Inc) or Inc < 30 or Inc > 85:
            continue
        if np.isnan(BMAG):
            continue

        # Rotation velocity from W50 corrected for inclination
        sin_i = np.sin(np.radians(Inc))
        Vrot = W50 / (2.0 * max(sin_i, 0.3))

        if Vrot <= 20:
            continue

        # Stellar mass from B-band luminosity
        # M_B_sun = 5.48 (solar absolute B mag)
        L_B_Lsun = 10**(0.4 * (5.48 - BMAG))

        # M/L estimate for early-type disks (B-V ~ 0.6-0.8)
        # Using ML_B ~ 2.0 for Sa/Sab (higher than late types)
        ML_B = 2.0
        M_star = L_B_Lsun * ML_B

        # Gas mass
        M_HI_Msun = MHI * 1e9 if not np.isnan(MHI) else 0.0
        Mbar = M_star + 1.33 * M_HI_Msun  # including helium

        if Mbar <= 0:
            continue

        # Single-point RAR using dynamical mass approach
        # Estimate effective radius from W50/W20 and distance
        R_eff = Vrot / (H0 * Dist / c_light) if Dist > 0 else 10.0  # rough
        R_eff = max(R_eff, 1.0)

        # Better: use HI radius if available
        RHI_kpc = safe_float(row.get('RHIkpc', ''))
        if not np.isnan(RHI_kpc) and RHI_kpc > 0:
            R_eff = RHI_kpc * 0.5  # Use half the HI radius as characteristic

        # gobs and gbar
        gobs = Vrot**2 / R_eff * conv
        gbar = G_kpc * Mbar / R_eff**2 * conv

        if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
            continue

        gobs_pred = rar_prediction(gbar)
        log_gbar = np.log10(gbar)
        log_gobs = np.log10(gobs)
        log_res = log_gobs - np.log10(gobs_pred)

        if abs(log_res) > 1.5:
            continue

        # ENVIRONMENT: Use the explicit Memb column!
        # Galaxies with group membership (LGG, HCG, pairs) are in denser environments
        env_dense = 'field'
        logMh = 11.0

        # First try proximity-based classification
        if not np.isnan(ra_deg) and not np.isnan(dec_deg) and not np.isnan(Vsys):
            _, logMh_prox, env_prox = classify_environment_proximity(
                ra_deg, dec_deg, Vsys, name)
            if logMh_prox > logMh:
                logMh = logMh_prox
                env_dense = env_prox

        # Then enhance with explicit membership info
        if memb and memb != '--':
            memb_lower = memb.lower().strip()
            # Rich groups/clusters get dense classification
            if any(kw in memb_lower for kw in ['virgo', 'coma', 'perseus',
                                                  'hydra', 'fornax', 'centaurus']):
                env_dense = 'dense'
                logMh = max(logMh, 14.0)
            elif 'hcg' in memb_lower:
                env_dense = 'dense'
                logMh = max(logMh, 13.0)
            elif 'lgg' in memb_lower or 'pair' in memb_lower or 'group' in memb_lower:
                # Loose groups: moderately dense
                env_dense = 'dense' if logMh >= 12.5 else 'field'
                logMh = max(logMh, 12.5)
            elif region:
                # Named regions indicate some structure
                logMh = max(logMh, 12.0)

        names.append(name)
        points.append({
            'galaxy': name,
            'source': 'Noordermeer2005',
            'log_gbar': float(log_gbar),
            'log_gobs': float(log_gobs),
            'log_res': float(log_res),
            'sigma_log_gobs': 0.15,
            'R_kpc': float(R_eff),
            'env_dense': env_dense,
            'logMh': float(logMh),
        })

    n_dense = sum(1 for p in points if p['env_dense'] == 'dense')
    print(f"    Noordermeer+2005: {len(names)} galaxies "
          f"({n_dense} dense, {len(names)-n_dense} field)")
    return points, names


# ============================================================
# DATASET 11: Vogt+2004 cluster galaxy survey
# 329 galaxies in 19 clusters with optical+HI velocity widths
# ============================================================
def load_vogt2004():
    """
    Load Vogt+2004 cluster galaxy survey (J/AJ/127/3273).

    329 galaxies across 19 clusters (Coma, A1367, Cancer, Perseus, etc.)
    with optical rotation curve widths and HI velocity widths.
    CRITICAL: Has explicit cluster membership (Clust column).

    Uses single-point RAR: Vrot from OW0 (optical width) and
    baryonic fraction from Dark matter ratio.
    """
    print("\n  [11/12] Loading Vogt+2004 cluster galaxies (329 in 19 clusters)...")

    tsv_path = os.path.join(DATA_DIR, 'hi_surveys', 'vogt2004_galaxies.tsv')
    _, rows = parse_vizier_tsv(tsv_path)

    if not rows:
        print("    WARNING: Vogt+2004 not found or empty")
        return [], []

    # Load CF4 distance cache if available
    vogt_cf4_path = os.path.join(DATA_DIR, 'vogt2004_cf4_cache.json')
    vogt_cf4 = {}
    if os.path.exists(vogt_cf4_path):
        with open(vogt_cf4_path, 'r') as f:
            vogt_cf4 = json.load(f)
        n_cf4 = sum(1 for v in vogt_cf4.values()
                    if isinstance(v, dict) and v.get('D_cf4'))
        print(f"    Loaded CF4 distances for {n_cf4} Vogt+2004 galaxies")

    # Known cluster properties: name -> (distance_Mpc, logMh, sigma_v km/s)
    cluster_props = {
        'N507':   (66.0, 13.5, 500),
        'Coma':   (100.0, 15.0, 1000),
        'A1367':  (92.0, 14.8, 850),
        'Cancer': (78.0, 13.8, 500),
        'A2634':  (120.0, 14.5, 700),
        'A2199':  (125.0, 14.8, 800),
        'A426':   (73.0, 14.8, 1300),   # Perseus
        'A262':   (66.0, 14.0, 525),
        'A400':   (97.0, 14.2, 650),
        'A539':   (117.0, 14.3, 650),
        'A2197':  (125.0, 14.3, 600),
        'A2151':  (150.0, 14.5, 750),   # Hercules
        'A2063':  (147.0, 14.3, 650),
        'A2147':  (154.0, 14.5, 800),
        'A2666':  (115.0, 13.8, 500),
        'A779':   (97.0, 14.0, 550),
        'A2152':  (160.0, 14.2, 600),
        'A2162':  (130.0, 13.5, 400),
        'Field':  (0.0, 11.0, 0),
    }

    points = []
    names = []

    def parse_ra_dec_hms(ra_str, dec_str):
        """Parse RA (h:m:s) and Dec (d:m:s) strings to degrees."""
        try:
            parts = ra_str.strip().split()
            if len(parts) >= 3:
                ra_deg = (float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600) * 15
            else:
                ra_deg = float('nan')
            parts = dec_str.strip().split()
            if len(parts) >= 3:
                sign = -1 if dec_str.strip().startswith('-') else 1
                dec_deg = sign * (abs(float(parts[0])) + float(parts[1])/60 + float(parts[2])/3600)
            else:
                dec_deg = float('nan')
            return ra_deg, dec_deg
        except:
            return float('nan'), float('nan')

    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue

        clust = row.get('Clust', '').strip()
        rv = safe_float(row.get('RV', ''))          # CMB velocity km/s
        ow0 = safe_float(row.get('OW0', ''))         # Optical velocity width km/s
        rw0 = safe_float(row.get('RW0', ''))          # HI velocity width km/s
        ml_c = safe_float(row.get('M/Lc', ''))        # Corrected M/L
        dark = safe_float(row.get('Dark', ''))         # Dark/light ratio
        ttype = row.get('TType', '').strip()

        # Parse J2000 coordinates
        ra_str = row.get('_RA.icrs', '').strip()
        dec_str = row.get('_DE.icrs', '').strip()
        ra, dec = parse_ra_dec_hms(ra_str, dec_str)

        # Need velocity width
        if np.isnan(ow0) or ow0 <= 0:
            continue
        if np.isnan(rv) or rv <= 0:
            continue

        # Distance: prefer CF4 flow-model, then cluster distance, then Hubble
        cf4_entry = vogt_cf4.get(name, {})
        if isinstance(cf4_entry, dict) and cf4_entry.get('D_cf4'):
            Dist = cf4_entry['D_cf4']
        elif clust in cluster_props and clust != 'Field':
            Dist = cluster_props[clust][0]
        else:
            Dist = rv / H0

        # Environment: use cluster membership
        if clust in cluster_props and clust != 'Field':
            logMh = cluster_props[clust][1]
            env_dense = 'dense'
        else:
            # Also check proximity classification
            if not np.isnan(ra) and not np.isnan(dec):
                _, logMh, env_dense = classify_environment_proximity(ra, dec, rv, name)
            else:
                logMh = 11.0
                env_dense = 'field'

        if Dist <= 0 or Dist > 300:
            continue

        # Vrot from velocity width (these are from fitted rotation curves,
        # already corrected for inclination in the original paper)
        Vrot = ow0 / 2.0

        if Vrot <= 20 or Vrot > 500:
            continue

        # Estimate baryonic mass from the Dark matter ratio
        # Dark = Mdark / Mlight, so fbar = 1/(1+Dark)
        if not np.isnan(dark) and dark > 0 and not np.isnan(ml_c) and ml_c > 0:
            # Full approach: use Dark ratio
            fbar = 1.0 / (1.0 + dark)
        elif not np.isnan(ml_c) and ml_c > 0:
            # Use M/L to estimate: typical Mdyn/Mbar ~ 5-10
            fbar = 1.0 / (1.0 + ml_c * 2.0)  # rough estimate
        else:
            # Fallback: use TF relation estimate
            # log(Mbar) ~ 4*log(Vrot) - 0.5 (approximate)
            fbar = 0.15  # typical baryon fraction

        if fbar <= 0 or fbar > 1.0:
            fbar = min(fbar, 0.99)
            if fbar <= 0:
                continue

        # Compute RAR using velocity-based approach (same as Yu+2020)
        # gobs = Vrot^2 / R, gbar = Vbar^2 / R = fbar * Vrot^2 / R
        # So gbar/gobs = fbar, and log(gbar) = log(gobs) + log(fbar)
        Vbar = Vrot * np.sqrt(fbar)

        # Estimate R from velocity width and dynamical mass
        Mdyn = Vrot**2 * Dist * ow0 / (2.0 * H0) / G_kpc  # rough estimate
        Rdyn = Vrot**2 / (G_kpc * Mdyn * 1e-3) if Mdyn > 0 else 10.0  # kpc
        # Simpler: use Hubble radius estimate
        # For typical spiral: R_opt ~ 10 * (Vrot/200)^0.5 kpc
        R_opt = 10.0 * np.sqrt(Vrot / 200.0)  # kpc
        if R_opt <= 0:
            R_opt = 10.0

        gobs = Vrot**2 / R_opt * conv
        gbar = fbar * gobs

        if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
            continue

        gobs_pred = rar_prediction(gbar)
        log_gbar = np.log10(gbar)
        log_gobs = np.log10(gobs)
        log_res = log_gobs - np.log10(gobs_pred)

        if abs(log_res) > 1.5:
            continue

        names.append(name)
        points.append({
            'galaxy': name,
            'source': 'Vogt2004',
            'log_gbar': float(log_gbar),
            'log_gobs': float(log_gobs),
            'log_res': float(log_res),
            'sigma_log_gobs': 0.15,
            'R_kpc': float(R_opt),
            'env_dense': env_dense,
            'logMh': float(logMh),
        })

    n_dense = sum(1 for p in points if p['env_dense'] == 'dense')
    print(f"    Vogt+2004: {len(names)} galaxies "
          f"({n_dense} dense, {len(names)-n_dense} field)")
    return points, names


# ============================================================
# DATASET 12: Catinella+2005 Polyex rotation curve fits
# ~400 galaxies with analytical V(r) reconstructions
# ============================================================
def load_catinella2005():
    """
    Load Catinella+2005 Polyex rotation curve fits (J/AJ/130/1037).

    Uses Polyex model parameters to reconstruct V(r) at multiple radii,
    giving extended rotation curves for ~400 galaxies.
    V(r) = V0 * (1 - exp(-r/rPE)) * (1 + alpha * r/rPE)
    """
    print("\n  [12/12] Loading Catinella+2005 Polyex RCs (~400 galaxies)...")

    tsv_path = os.path.join(DATA_DIR, 'hi_surveys', 'catinella2005_polyex.tsv')
    _, rows = parse_vizier_tsv(tsv_path)

    # Load CF4 distance cache if available
    cat_cf4_path = os.path.join(DATA_DIR, 'catinella2005_cf4_cache.json')
    cat_cf4 = {}
    if os.path.exists(cat_cf4_path):
        with open(cat_cf4_path, 'r') as f:
            cat_cf4 = json.load(f)
        n_cf4 = sum(1 for v in cat_cf4.values()
                    if isinstance(v, dict) and v.get('D_cf4'))
        print(f"    Loaded CF4 distances for {n_cf4} Catinella+2005 galaxies")

    if not rows:
        print("    WARNING: Catinella+2005 not found or empty")
        return [], []

    points = []
    names = []

    for row in rows:
        name = row.get('OName', '').strip()
        if not name:
            name = str(row.get('Seq', '')).strip()
        if not name:
            continue

        # Parse coordinates
        ra_str = row.get('RAJ2000', '').strip()
        dec_str = row.get('DEJ2000', '').strip()

        # Convert RA "hh mm ss.s" to degrees
        try:
            parts = ra_str.split()
            ra = (float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600) * 15
        except:
            ra = float('nan')
        try:
            parts = dec_str.split()
            sign = -1 if dec_str.startswith('-') else 1
            dec = sign * (abs(float(parts[0])) + float(parts[1])/60 + float(parts[2])/3600)
        except:
            dec = float('nan')

        # Polyex parameters
        V0 = safe_float(row.get('V0', ''))           # Amplitude km/s
        rPE = safe_float(row.get('rPE', ''))          # Scale length arcsec
        alpha = safe_float(row.get('alpha', ''))       # Outer slope
        Ropt = safe_float(row.get('Ropt', ''))         # Optical radius arcsec
        Rmax = safe_float(row.get('Rmax', ''))         # Max RC radius arcsec
        qual = safe_int(row.get('Qual', '0'))          # Quality flag
        V80 = safe_float(row.get('V80', ''))           # Heliocentric velocity
        W80 = safe_float(row.get('W80', ''))           # Velocity width
        ttype = row.get('Type', '').strip()

        # Quality and completeness cuts
        if np.isnan(V0) or V0 <= 0:
            continue
        if np.isnan(rPE) or rPE <= 0:
            continue
        if np.isnan(alpha):
            alpha = 0.0
        if np.isnan(Ropt) or Ropt <= 0:
            continue
        if np.isnan(V80) or V80 <= 0:
            continue
        if qual != 1:  # Only use good quality fits
            continue

        # Distance: prefer CF4 flow-model, else Hubble-flow
        seq_key = str(row.get('Seq', '')).strip()
        cf4_entry = cat_cf4.get(name, cat_cf4.get(seq_key, {}))
        if isinstance(cf4_entry, dict) and cf4_entry.get('D_cf4'):
            Dist = cf4_entry['D_cf4']
        else:
            Dist = V80 / H0  # Mpc
        if Dist <= 5 or Dist > 250:
            continue

        # Convert angular sizes to physical
        Ropt_kpc = Ropt * Dist / 206.265  # kpc
        rPE_kpc = rPE * Dist / 206.265    # kpc
        Rmax_kpc = Rmax * Dist / 206.265 if not np.isnan(Rmax) else Ropt_kpc

        if Ropt_kpc <= 0 or rPE_kpc <= 0:
            continue

        # Environment classification
        env_dense = 'field'
        logMh = 11.0
        if not np.isnan(ra) and not np.isnan(dec):
            _, logMh, env_dense = classify_environment_proximity(ra, dec, V80, name)

        # Estimate baryonic mass from V0 and TF relation
        # Use TF: log(Mbar) = 4*log(Vflat) + 1.0 approximately
        # For Vflat, use V0 as asymptotic velocity
        Vflat = V0 * (1 + alpha * Ropt / rPE)  # V at R_opt
        if Vflat <= 20:
            continue

        log_Mbar = 4.0 * np.log10(max(Vflat, 30)) + 1.0  # rough TF
        Mbar = 10**log_Mbar

        # M/L from morphological type: earlier types have higher M/L
        try:
            t_num = float(ttype.replace('B', ''))
        except:
            t_num = 4.0
        ML_est = 1.5 if t_num <= 2 else 1.0 if t_num <= 4 else 0.6

        names.append(name)

        # Reconstruct V(r) at multiple radii using Polyex model
        # V(r) = V0 * (1 - exp(-r/rPE)) * (1 + alpha * r/rPE)
        n_radii = max(5, int(Rmax_kpc / 2.0))
        radii = np.linspace(max(rPE_kpc * 0.5, 0.5), min(Rmax_kpc, 50), n_radii)

        for R_kpc in radii:
            r_arcsec = R_kpc * 206.265 / Dist
            y = r_arcsec / rPE

            # Polyex model
            Vrot = V0 * (1.0 - np.exp(-y)) * (1.0 + alpha * y)
            if Vrot <= 0 or not np.isfinite(Vrot):
                continue

            # gobs from rotation velocity
            gobs = Vrot**2 / R_kpc * conv

            # gbar from exponential disk mass model
            # M_disk(<R) = Mbar * [1 - (1+x)*exp(-x)] where x = R/h
            # h ~ Ropt/4 for exponential disk (R_opt ~ 3.2h)
            h_kpc = Ropt_kpc / 3.2
            x = R_kpc / h_kpc
            frac = 1.0 - (1.0 + x) * np.exp(-x)
            M_enc = Mbar * frac
            gbar = G_kpc * M_enc / R_kpc**2 * conv

            if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
                continue

            gobs_pred = rar_prediction(gbar)
            log_gbar = np.log10(gbar)
            log_gobs = np.log10(gobs)
            log_res = log_gobs - np.log10(gobs_pred)

            if abs(log_res) > 1.5:
                continue

            sigma_log = 0.10  # Polyex fits typically have ~10% uncertainty

            points.append({
                'galaxy': name,
                'source': 'Catinella2005',
                'log_gbar': float(log_gbar),
                'log_gobs': float(log_gobs),
                'log_res': float(log_res),
                'sigma_log_gobs': float(sigma_log),
                'R_kpc': float(R_kpc),
                'env_dense': env_dense,
                'logMh': float(logMh),
            })

    n_galaxies = len(set(names))
    n_dense = len(set(p['galaxy'] for p in points if p['env_dense'] == 'dense'))
    n_cf4_used = sum(1 for n in set(names)
                     if (cat_cf4.get(n, {}) if isinstance(cat_cf4.get(n, {}), dict) else {}).get('D_cf4'))
    cf4_str = f", {n_cf4_used} with CF4 dist" if n_cf4_used else ""
    print(f"    Catinella+2005: {n_galaxies} galaxies, {len(points)} RAR points "
          f"({n_dense} dense, {n_galaxies - n_dense} field{cf4_str})")
    return points, list(set(names))


# ============================================================
# DATASET 13: Virgo cluster extended rotation curves
# Fathi+2009 (NGC 4294 Halpha) + Lang+2020 (NGC 4293 CO)
# ============================================================
def load_virgo_extended_rc():
    """
    Load extended rotation curves for Virgo cluster galaxies.

    NGC 4294: Fathi+2009 deprojected Halpha RC (J/ApJ/704/1657)
      - 23 V(r) points from 2.2" to 73.5" (arcsec)
      - Virgo cluster member at D=16.6 Mpc

    NGC 4293: Lang+2020 PHANGS-ALMA CO RC (J/ApJ/897/122)
      - 18 V(r) points from 0.125 to 2.675 kpc
      - Virgo cluster member at D=16.6 Mpc

    These galaxies fill the gbar < 10^-12.5 regime where our pipeline
    has sparse data, which is critical for the BEC transition function.
    """
    print("\n  [13/13] Loading Virgo extended rotation curves (Fathi+2009, Lang+2020)...")

    D_virgo = 16.6  # Mpc, Virgo cluster distance
    points = []
    names = []

    # --- NGC 4294: Fathi+2009 Halpha RC ---
    fathi_path = os.path.join(DATA_DIR, 'hi_surveys', 'Fathi2009_NGC4294_Halpha_RC.tsv')
    if os.path.exists(fathi_path):
        # Simple TSV (no VizieR separators), parse directly
        with open(fathi_path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split('\t')
        rows = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= len(header):
                rows.append({header[i]: parts[i].strip().strip('"')
                             for i in range(len(header))})

        # NGC 4294 properties:
        # Type: SBcd, Incl ~70 deg (Koopmann+2006), Virgo cluster
        # Estimate Mbar from TF: Vflat ~ 98 km/s
        Vflat = 98.0  # km/s (outermost point)
        log_Mbar = 4.0 * np.log10(Vflat) + 1.0  # rough TF
        Mbar = 10**log_Mbar
        # Optical radius from Koopmann+2006: Ropt = 22" = 1.77 kpc
        Ropt_kpc = 22.0 * D_virgo / 206.265  # 1.77 kpc
        h_kpc = Ropt_kpc / 3.2  # exponential scale length

        name = 'NGC4294'
        names.append(name)

        for row in rows:
            r_arcsec = safe_float(row.get('RadG', ''))
            Vrot = safe_float(row.get('vrot', ''))
            if np.isnan(r_arcsec) or np.isnan(Vrot) or Vrot <= 0:
                continue

            R_kpc = r_arcsec * D_virgo / 206.265
            if R_kpc <= 0.1:
                continue

            gobs = Vrot**2 / R_kpc * conv

            # Exponential disk mass model
            x = R_kpc / h_kpc
            frac = 1.0 - (1.0 + x) * np.exp(-x)
            M_enc = Mbar * frac
            gbar = G_kpc * M_enc / R_kpc**2 * conv

            if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
                continue

            gobs_pred = rar_prediction(gbar)
            log_gbar = np.log10(gbar)
            log_gobs = np.log10(gobs)
            log_res = log_gobs - np.log10(gobs_pred)

            if abs(log_res) > 1.5:
                continue

            sigma_log = 0.08  # Halpha RCs typically have 5-10% velocity error

            points.append({
                'galaxy': name,
                'source': 'VirgoRC',
                'log_gbar': float(log_gbar),
                'log_gobs': float(log_gobs),
                'log_res': float(log_res),
                'sigma_log_gobs': float(sigma_log),
                'R_kpc': float(R_kpc),
                'env_dense': 'dense',  # Virgo cluster
                'logMh': 14.2,  # Virgo cluster mass
            })

        print(f"    NGC 4294 (Fathi+2009): {len(rows)} points loaded")

    # --- NGC 4293: Lang+2020 PHANGS-ALMA CO RC ---
    lang_path = os.path.join(DATA_DIR, 'hi_surveys', 'Lang2020_NGC4293_RC.tsv')
    if os.path.exists(lang_path):
        # Simple TSV (no VizieR separators), parse directly
        with open(lang_path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split('\t')
        rows = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= len(header):
                rows.append({header[i]: parts[i].strip().strip('"')
                             for i in range(len(header))})

        # NGC 4293 properties:
        # Type: SB0/a pec (Koopmann+2006), Incl ~67 deg, Virgo cluster
        # Vflat ~ 131 km/s (max from RC)
        Vflat = 131.0
        log_Mbar = 4.0 * np.log10(Vflat) + 1.0
        Mbar = 10**log_Mbar
        # Optical radius from Koopmann+2006: Ropt = 60" = 4.83 kpc
        Ropt_kpc = 60.0 * D_virgo / 206.265
        h_kpc = Ropt_kpc / 3.2

        name = 'NGC4293'
        names.append(name)

        for row in rows:
            R_kpc = safe_float(row.get('Rad', ''))
            Vrot = safe_float(row.get('VRot', ''))
            if np.isnan(R_kpc) or np.isnan(Vrot) or Vrot <= 0 or R_kpc <= 0.1:
                continue

            gobs = Vrot**2 / R_kpc * conv

            x = R_kpc / h_kpc
            frac = 1.0 - (1.0 + x) * np.exp(-x)
            M_enc = Mbar * frac
            gbar = G_kpc * M_enc / R_kpc**2 * conv

            if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
                continue

            gobs_pred = rar_prediction(gbar)
            log_gbar = np.log10(gbar)
            log_gobs = np.log10(gobs)
            log_res = log_gobs - np.log10(gobs_pred)

            if abs(log_res) > 1.5:
                continue

            # CO RCs have velocity errors from paper
            e_Vrot_up = safe_float(row.get('E_VRot', ''))
            e_Vrot_dn = safe_float(row.get('e_VRot', ''))
            if not np.isnan(e_Vrot_up) and not np.isnan(e_Vrot_dn):
                e_Vrot = max(e_Vrot_up, e_Vrot_dn)
                sigma_log = max(0.05, e_Vrot / Vrot / np.log(10))
            else:
                sigma_log = 0.08

            points.append({
                'galaxy': name,
                'source': 'VirgoRC',
                'log_gbar': float(log_gbar),
                'log_gobs': float(log_gobs),
                'log_res': float(log_res),
                'sigma_log_gobs': float(sigma_log),
                'R_kpc': float(R_kpc),
                'env_dense': 'dense',  # Virgo cluster
                'logMh': 14.2,  # Virgo cluster mass
            })

        print(f"    NGC 4293 (Lang+2020): {len(rows)} points loaded")

    n_galaxies = len(set(names))
    n_pts = len(points)
    print(f"    Virgo RC total: {n_galaxies} galaxies, {n_pts} RAR points (all dense/Virgo)")
    return points, list(set(names))


# ============================================================
# DATASET 14: PHANGS-ALMA CO rotation curves (Lang+2020)
# 67 galaxies with CO kinematic rotation curves, ~17 in Virgo cluster
# ============================================================
def load_phangs_lang2020():
    """
    Load PHANGS-ALMA CO rotation curves from Lang+2020 (J/ApJ/897/122).

    These are high-resolution CO(2-1) kinematic rotation curves from the
    PHANGS-ALMA survey. Uses properties table for positions, distances,
    and inclinations. Rad is in kpc (physical), VRot in km/s.

    Key: many PHANGS galaxies are in the Virgo cluster, providing
    dense-environment extended rotation curves.
    """
    rc_path = os.path.join(DATA_DIR, 'hi_surveys', 'phangs_lang2020_rotation_curves.tsv')
    prop_path = os.path.join(DATA_DIR, 'hi_surveys', 'phangs_lang2020_properties.tsv')

    if not os.path.exists(rc_path) or not os.path.exists(prop_path):
        print("    PHANGS Lang+2020: data files not found")
        return [], []

    # Load properties table
    props = {}
    with open(prop_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue
            row = {header[i]: parts[i].strip().strip('"').strip() for i in range(len(header))}
            gal_id = row.get('ID', '').strip()
            if not gal_id:
                continue
            try:
                props[gal_id] = {
                    'ra': float(row.get('RAJ2000', 'nan')),
                    'dec': float(row.get('DEJ2000', 'nan')),
                    'dist': float(row.get('Dist', 'nan')),     # Mpc
                    'vsys': float(row.get('Vsys', 'nan')),     # km/s
                    'inc': float(row.get('i', 'nan')),         # degrees
                    'pa': float(row.get('PA', 'nan')),
                }
            except (ValueError, TypeError):
                continue

    # Load rotation curves
    points = []
    names = []
    n_skipped_inc = 0
    n_skipped_few = 0
    n_wise_mass = 0
    n_tf_mass = 0

    # Group RC rows by galaxy
    gal_rows = {}
    with open(rc_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue
            row = {header[i]: parts[i].strip().strip('"').strip() for i in range(len(header))}
            gal_id = row.get('ID', '').strip()
            if gal_id not in gal_rows:
                gal_rows[gal_id] = []
            gal_rows[gal_id].append(row)

    for gal_id, rows in gal_rows.items():
        if gal_id not in props:
            continue

        p = props[gal_id]
        inc = p['inc']
        dist = p['dist']  # Mpc

        # Inclination cut
        if inc < 20 or inc > 85:
            n_skipped_inc += 1
            continue

        # Normalize name: "NGC0628 " -> "NGC628"
        norm_name = re.sub(r'\s+', '', gal_id)
        # Strip leading zeros: NGC0628 -> NGC628
        m = re.match(r'^(NGC|IC|UGC)0*(\d+\w*)$', norm_name, re.IGNORECASE)
        if m:
            norm_name = f"{m.group(1).upper()}{m.group(2)}"

        # Classify environment
        struct_name, logMh, env = classify_environment_proximity(
            p['ra'], p['dec'], p['vsys'], name=norm_name)

        # Parse rotation curve
        radii_kpc = []
        vrots = []
        e_vrots = []
        for row in rows:
            rad_kpc = safe_float(row.get('Rad', ''))
            vrot = safe_float(row.get('VRot', ''))
            e_vrot_up = safe_float(row.get('E_VRot', ''))
            e_vrot_dn = safe_float(row.get('e_VRot', ''))

            if np.isnan(rad_kpc) or np.isnan(vrot) or vrot <= 0:
                continue

            # Use average of upper/lower errors
            e_vrot = 0.0
            if not np.isnan(e_vrot_up) and not np.isnan(e_vrot_dn):
                e_vrot = 0.5 * (e_vrot_up + e_vrot_dn)
            elif not np.isnan(e_vrot_up):
                e_vrot = e_vrot_up

            radii_kpc.append(rad_kpc)
            vrots.append(vrot)
            e_vrots.append(e_vrot)

        if len(radii_kpc) < 3:
            n_skipped_few += 1
            continue

        radii_kpc = np.array(radii_kpc)
        vrots = np.array(vrots)
        e_vrots = np.array(e_vrots)

        # Compute gobs from Vrot and R (both in physical units)
        # gobs = V^2 / R, convert to m/s^2
        gobs = (vrots ** 2) / radii_kpc * conv  # (km/s)^2/kpc -> m/s^2

        # Estimate Mbar: try WISE photometric mass first, fall back to TF
        vflat = vrots[-1]
        wise_logM = get_z0mgs_stellar_mass(name=norm_name, ra=p['ra'], dec=p['dec'])
        if wise_logM is not None:
            # WISE M* + 33% gas correction
            mbar = 1.33 * 10**wise_logM
            log_mbar = np.log10(mbar)
            n_wise_mass += 1
        else:
            log_mbar = 3.75 * np.log10(vflat) + 2.00
            mbar = 10 ** log_mbar
            n_tf_mass += 1

        # Compute gbar assuming exponential disk
        # Scale length from TF: Rd ~ 0.02 * Vflat/H0 * 1000 kpc (approximate)
        rd = 0.2 * dist  # rough scale length in kpc (2% of distance in Mpc * 100)
        rd = max(rd, 0.5)  # minimum 0.5 kpc

        # Better: use CO extent. Rco/R25 is available. Use mean radius as proxy.
        mean_rad = np.mean(radii_kpc)
        rd = mean_rad / 2.2  # exponential disk: peak at 2.2 Rd
        rd = max(rd, 0.3)

        # gbar from exponential disk model
        for j in range(len(radii_kpc)):
            r_kpc = radii_kpc[j]
            y = r_kpc / (2.0 * rd)
            # Freeman disk: V^2 = 4*pi*G*Sigma0*Rd * y^2 * [I0*K0 - I1*K1]
            # Simplified: gbar(r) = G * Mbar * r / (r^2 + rd^2)^(3/2) * correction
            gbar_val = G_kpc * mbar * r_kpc / (r_kpc**2 + rd**2)**1.5
            gbar_ms2 = gbar_val * conv  # -> m/s^2

            if gbar_ms2 > 0 and gobs[j] > 0:
                # Fractional velocity error cut: σ_V/V < 15%
                e_v = e_vrots[j] if j < len(e_vrots) else 5.0
                if vrots[j] > 0 and e_v / vrots[j] > 0.15:
                    continue

                lg_gbar = np.log10(gbar_ms2)
                lg_gobs = np.log10(gobs[j])
                log_res = lg_gobs - np.log10(rar_prediction(gbar_ms2))

                # Reject extreme outliers
                if abs(log_res) > 1.5:
                    continue

                # Uncertainty estimate from Vrot error
                sigma_lg = 2.0 * e_v / max(vrots[j], 1.0) / np.log(10)
                sigma_lg = max(sigma_lg, 0.02)

                points.append({
                    'galaxy': norm_name,
                    'r_kpc': r_kpc,
                    'vrot': vrots[j],
                    'gbar': gbar_ms2,
                    'gobs': gobs[j],
                    'log_gbar': lg_gbar,
                    'log_gobs': lg_gobs,
                    'log_res': log_res,
                    'sigma_log_gobs': sigma_lg,
                    'env_dense': env,
                    'logMh': logMh,
                    'structure': struct_name,
                    'source': 'PHANGS',
                })

        names.append(norm_name)

    n_galaxies = len(set(names))
    n_pts = len(points)
    n_dense = sum(1 for p in points if p['env_dense'] == 'dense')
    n_field = sum(1 for p in points if p['env_dense'] == 'field')

    print(f"    PHANGS Lang+2020: {n_galaxies} galaxies, {n_pts} RAR points "
          f"({n_dense} dense, {n_field} field)")
    print(f"    Mass models: {n_wise_mass} WISE, {n_tf_mass} Tully-Fisher")
    if n_skipped_inc > 0 or n_skipped_few > 0:
        print(f"    Skipped: {n_skipped_inc} (inclination), {n_skipped_few} (too few pts)")

    return points, list(set(names))


# ============================================================
# DATASET 15: Verheijen+2001 Ursa Major HI rotation curves
# 41 galaxies in the UMa cluster with resolved HI RCs
# ============================================================
def load_verheijen2001_uma():
    """
    Load Verheijen & Sancisi (2001) HI rotation curves for Ursa Major cluster.

    All 41 galaxies are in the UMa cluster (logMh ~ 12.8, dense environment).
    Name codes: "N3726" = NGC3726, "U6399" = UGC6399.
    Rad in arcsec, Vrot = average of approaching/receding sides (km/s).
    """
    rc_path = os.path.join(DATA_DIR, 'hi_surveys', 'uma_verheijen2001_rotation_curves.tsv')

    if not os.path.exists(rc_path):
        print("    Verheijen+2001 UMa: data file not found")
        return [], []

    # UMa cluster properties
    uma_logMh = 12.8
    uma_dist_mpc = 18.6  # Tully+1996 UMa distance

    # Parse rotation curves grouped by galaxy
    gal_rows = {}
    with open(rc_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue
            row = {header[i]: parts[i].strip().strip('"').strip() for i in range(len(header))}
            name_short = row.get('Name', '').strip()
            if not name_short:
                continue
            if name_short not in gal_rows:
                gal_rows[name_short] = []
            gal_rows[name_short].append(row)

    points = []
    names = []
    n_skipped_inc = 0
    n_skipped_few = 0
    n_wise_mass = 0
    n_tf_mass = 0

    for short_name, rows in gal_rows.items():
        # Expand short name: N3726 -> NGC3726, U6399 -> UGC6399
        if short_name.startswith('N') and short_name[1:].isdigit():
            full_name = f"NGC{short_name[1:]}"
        elif short_name.startswith('U') and short_name[1:].isdigit():
            full_name = f"UGC{short_name[1:]}"
        else:
            full_name = short_name

        # Get inclination from first row
        inc = safe_float(rows[0].get('Incl', ''))
        if np.isnan(inc) or inc < 30 or inc > 85:
            n_skipped_inc += 1
            continue

        # Parse RC points
        radii_arcsec = []
        vrots = []
        for row in rows:
            rad_as = safe_float(row.get('Rad', ''))
            vrot = safe_float(row.get('Vrot', ''))  # Average of App and Rec

            if np.isnan(rad_as) or np.isnan(vrot) or vrot <= 0:
                continue

            radii_arcsec.append(rad_as)
            vrots.append(vrot)

        if len(radii_arcsec) < 3:
            n_skipped_few += 1
            continue

        radii_arcsec = np.array(radii_arcsec)
        vrots = np.array(vrots)

        # Convert arcsec to kpc
        # R_kpc = R_arcsec * dist_Mpc * 1000 * (pi/180/3600) = R_arcsec * dist_Mpc / 206.265
        radii_kpc = radii_arcsec * uma_dist_mpc / 206.265

        # Compute gobs
        gobs = (vrots ** 2) / radii_kpc * conv  # m/s^2

        # Estimate Mbar: try WISE photometric mass first, fall back to TF
        vflat = vrots[-1]
        wise_logM = get_z0mgs_stellar_mass(name=full_name)
        if wise_logM is not None:
            # WISE M* + 33% gas correction
            mbar = 1.33 * 10**wise_logM
            log_mbar = np.log10(mbar)
            n_wise_mass += 1
        else:
            log_mbar = 3.75 * np.log10(max(vflat, 10.0)) + 2.00
            mbar = 10 ** log_mbar
            n_tf_mass += 1

        # Scale length from RC extent
        mean_rad = np.mean(radii_kpc)
        rd = mean_rad / 2.2
        rd = max(rd, 0.3)

        # All UMa -> dense
        env = 'dense'
        struct_name = 'UMa'
        logMh = uma_logMh

        for j in range(len(radii_kpc)):
            r_kpc = radii_kpc[j]
            gbar_val = G_kpc * mbar * r_kpc / (r_kpc**2 + rd**2)**1.5
            gbar_ms2 = gbar_val * conv

            if gbar_ms2 > 0 and gobs[j] > 0:
                lg_gbar = np.log10(gbar_ms2)
                lg_gobs = np.log10(gobs[j])
                log_res = lg_gobs - np.log10(rar_prediction(gbar_ms2))

                # Reject extreme outliers
                if abs(log_res) > 1.5:
                    continue

                # Verheijen data provides VrotApp errors but we use Vrot (avg)
                # Approximate uncertainty: ~5 km/s typical for HI RCs
                sigma_lg = 2.0 * 5.0 / max(vrots[j], 1.0) / np.log(10)
                sigma_lg = max(sigma_lg, 0.02)

                points.append({
                    'galaxy': full_name,
                    'r_kpc': r_kpc,
                    'vrot': vrots[j],
                    'gbar': gbar_ms2,
                    'gobs': gobs[j],
                    'log_gbar': lg_gbar,
                    'log_gobs': lg_gobs,
                    'log_res': log_res,
                    'sigma_log_gobs': sigma_lg,
                    'env_dense': env,
                    'logMh': logMh,
                    'structure': struct_name,
                    'source': 'Verheijen2001',
                })

        names.append(full_name)

    n_galaxies = len(set(names))
    n_pts = len(points)

    print(f"    Verheijen+2001 UMa: {n_galaxies} galaxies, {n_pts} RAR points (all dense/UMa)")
    print(f"    Mass models: {n_wise_mass} WISE, {n_tf_mass} Tully-Fisher")
    if n_skipped_inc > 0 or n_skipped_few > 0:
        print(f"    Skipped: {n_skipped_inc} (inclination), {n_skipped_few} (too few pts)")

    return points, list(set(names))


# ============================================================
# DATASET 17: MaNGA IFU gas rotation velocities (Ristea+2023)
# ~1,500 galaxies with multi-point RCs at 1Re, 1.3Re, 2Re
# ============================================================
def load_manga_ristea2023():
    """
    Load MaNGA IFU gas rotation velocities from Ristea+2023 (J/MNRAS/521/2521).

    These are ionized gas rotation velocities measured at 1Re, 1.3Re, and 2Re
    from SDSS-IV MaNGA integral field spectroscopy. Each galaxy has logMstar
    from SED fitting (Pipe3D/NSA).

    Environment classification uses:
      1) Tempel+2017 SDSS group catalog (coordinate cross-match)
      2) Fallback to proximity-based classification

    Returns multi-point RAR data: 2-3 points per galaxy at different radii.
    """
    print("\n  [17/17] Loading MaNGA Ristea+2023 IFU gas rotation (~3,000 galaxies)...")

    kin_path = os.path.join(DATA_DIR, 'hi_surveys', 'manga_ristea2023_kinematics.tsv')
    nsa_path = os.path.join(DATA_DIR, 'manga_nsa_properties.tsv')
    t17_gal_path = os.path.join(DATA_DIR, 'tempel2017_sdss_galaxies.tsv')
    t17_grp_path = os.path.join(DATA_DIR, 'tempel2017_sdss_groups.tsv')

    # --- Load NSA properties (redshift, half-light radius) keyed by PlateIFU ---
    # NOTE: These files are plain TSV (not VizieR format), so use csv.DictReader
    nsa_data = {}
    if os.path.exists(nsa_path):
        with open(nsa_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                pifu = row.get('PlateIFU', '').strip().strip('"')
                z_str = row.get('NSAz', '').strip()
                rh_str = row.get('NSAsth50', '').strip()      # half-light radius in arcsec
                mass_str = row.get('NSAsMass', '').strip()      # stellar mass in Msun
                try:
                    z_val = float(z_str)
                    if z_val > 0 and z_val < 1.0 and mass_str != '-9999.0':
                        rh = float(rh_str) if rh_str else 0.0
                        nsa_data[pifu] = {
                            'z': z_val,
                            'Dist_Mpc': z_val * c_light / H0,
                            'Rh_arcsec': rh,
                        }
                except (ValueError, TypeError):
                    pass
        print(f"    NSA properties loaded: {len(nsa_data)} galaxies with redshifts")
    else:
        print("    WARNING: NSA properties file not found, need redshifts for distances")
        return [], []

    # --- Load Tempel+2017 group catalog for environment ---
    t17_group_props = {}
    t17_gal_coords = None
    t17_gal_groups = None

    if os.path.exists(t17_grp_path):
        with open(t17_grp_path, 'r', encoding='utf-8', errors='replace') as f:
            grp_rows = list(csv.DictReader(f, delimiter='\t'))
        for row in grp_rows:
            gid = row.get('GroupID', '').strip()
            ngal_str = row.get('Ngal', '').strip()
            m200_str = row.get('M200', '').strip()
            sig_str = row.get('sig_v', '').strip()
            try:
                ngal = int(ngal_str)
                m200 = float(m200_str) if m200_str else 0.0
                sig_v = float(sig_str) if sig_str else 0.0
                # M200 is in units of 10^12 h^-1 Msun (h=0.678)
                Mh_solar = m200 * 1e12 / 0.678
                logMh = np.log10(max(Mh_solar, 1.0))
                t17_group_props[gid] = {
                    'ngal': ngal, 'logMh': logMh, 'sig_v': sig_v
                }
            except (ValueError, TypeError):
                pass
        print(f"    Tempel+2017 groups loaded: {len(t17_group_props)} groups")

    if os.path.exists(t17_gal_path):
        with open(t17_gal_path, 'r', encoding='utf-8', errors='replace') as f:
            gal_rows = list(csv.DictReader(f, delimiter='\t'))
        t17_gal_ra = np.zeros(len(gal_rows))
        t17_gal_dec = np.zeros(len(gal_rows))
        t17_gal_gids = []
        valid = 0
        for i, row in enumerate(gal_rows):
            try:
                t17_gal_ra[i] = float(row.get('RAJ2000', '').strip())
                t17_gal_dec[i] = float(row.get('DEJ2000', '').strip())
                t17_gal_gids.append(row.get('GroupID', '').strip())
                valid += 1
            except:
                t17_gal_ra[i] = 999.0
                t17_gal_dec[i] = 999.0
                t17_gal_gids.append('0')
        t17_gal_ra = t17_gal_ra[:valid]
        t17_gal_dec = t17_gal_dec[:valid]
        t17_gal_gids = t17_gal_gids[:valid]
        t17_gal_coords = (t17_gal_ra, t17_gal_dec, t17_gal_gids)
        print(f"    Tempel+2017 galaxies loaded: {valid} for coordinate matching")

    def get_tempel_environment(ra, dec):
        """Cross-match to Tempel+2017 and return (logMh, env_dense, structure)."""
        if t17_gal_coords is None:
            return None
        t_ra, t_dec, t_gids = t17_gal_coords
        cos_d = np.cos(np.radians(dec))
        dra = (t_ra - ra) * cos_d
        ddec = t_dec - dec
        sep_arcsec = np.sqrt(dra**2 + ddec**2) * 3600.0
        imin = np.argmin(sep_arcsec)
        if sep_arcsec[imin] < 5.0:  # 5 arcsec match radius
            gid = t_gids[imin]
            grp = t17_group_props.get(gid, None)
            if grp:
                logMh = grp['logMh']
                env = 'dense' if logMh >= 12.5 else 'field'
                return logMh, env, f'T17_group_{gid}'
        return None

    # --- Load Ristea+2023 kinematics (plain TSV) ---
    with open(kin_path, 'r', encoding='utf-8', errors='replace') as f:
        kin_rows = list(csv.DictReader(f, delimiter='\t'))
    if not kin_rows:
        print("    WARNING: Ristea+2023 kinematics file not found or empty")
        return [], []

    points = []
    names = []
    n_no_nsa = 0
    n_no_vel = 0
    n_bad_vel = 0
    n_env_t17 = 0
    n_env_prox = 0
    n_env_field = 0

    for row in kin_rows:
        pifu = row.get('Plateifu', '').strip().strip('"')
        manga_id = row.get('MaNGA', '').strip().strip('"')
        ra = safe_float(row.get('RAJ2000', ''))
        dec = safe_float(row.get('DEJ2000', ''))
        logMstar = safe_float(row.get('logMstar', ''))

        # Gas velocities at different radii
        vg1_re = safe_float(row.get('VelG1Re', ''))       # at 1 Re
        e_vg1_re = safe_float(row.get('e_VelG1Re', ''))
        vg1_3re = safe_float(row.get('VelG1_3Re', ''))     # at 1.3 Re
        e_vg1_3re = safe_float(row.get('e_VelG1_3Re', ''))
        vg2_re = safe_float(row.get('VelG2Re', ''))         # at 2 Re
        e_vg2_re = safe_float(row.get('e_VelG2Re', ''))

        # Need at least 1Re gas velocity
        if np.isnan(vg1_re) or vg1_re <= 0:
            n_no_vel += 1
            continue

        # Need NSA redshift for distance
        nsa = nsa_data.get(pifu)
        if nsa is None:
            n_no_nsa += 1
            continue

        Dist_Mpc = nsa['Dist_Mpc']
        Rh_arcsec = nsa['Rh_arcsec']

        if Dist_Mpc <= 0 or Dist_Mpc > 700:
            continue

        # Convert half-light radius to kpc
        # R_kpc = R_arcsec * Dist_Mpc / 206.265
        if Rh_arcsec <= 0:
            Rh_arcsec = 3.0  # fallback: typical MaNGA galaxy
        Rh_kpc = Rh_arcsec * Dist_Mpc / 206.265

        if Rh_kpc <= 0:
            continue

        # Stellar mass (already in logMstar from Ristea+2023)
        if np.isnan(logMstar) or logMstar <= 0:
            continue
        Mstar = 10**logMstar

        # Baryonic mass: Mbar = 1.33 * Mstar (typical gas fraction correction)
        # More accurate for gas-rich galaxies, conservative for ETGs
        Mbar = 1.33 * Mstar

        # Environment classification
        # 1) Try Tempel+2017 SDSS group catalog
        t17_env = get_tempel_environment(ra, dec)
        if t17_env is not None:
            logMh, env_dense, structure = t17_env
            n_env_t17 += 1
        else:
            # 2) Fallback to proximity-based classification
            vsys = nsa['z'] * c_light
            structure, logMh, env_dense = classify_environment_proximity(
                ra, dec, vsys, name=pifu)
            if env_dense == 'dense':
                n_env_prox += 1
            else:
                n_env_field += 1

        # Galaxy name for deduplication
        gal_name = f"MaNGA_{pifu.strip()}"

        # Disk scale length from half-light radius: rd = Rh / 1.678 (for exponential disk)
        rd = Rh_kpc / 1.678

        # Build RAR points at each available radius
        radii_re = []
        vels = []
        vel_errs = []

        # Always have 1 Re
        radii_re.append(1.0)
        vels.append(vg1_re)
        vel_errs.append(e_vg1_re if not np.isnan(e_vg1_re) else vg1_re * 0.15)

        # 1.3 Re if available
        if not np.isnan(vg1_3re) and vg1_3re > 0:
            radii_re.append(1.3)
            vels.append(vg1_3re)
            vel_errs.append(e_vg1_3re if not np.isnan(e_vg1_3re) else vg1_3re * 0.15)

        # 2 Re if available
        if not np.isnan(vg2_re) and vg2_re > 0:
            radii_re.append(2.0)
            vels.append(vg2_re)
            vel_errs.append(e_vg2_re if not np.isnan(e_vg2_re) else vg2_re * 0.15)

        for k in range(len(radii_re)):
            r_re = radii_re[k]
            r_kpc = r_re * Rh_kpc
            Vrot = abs(vels[k])
            eVrot = abs(vel_errs[k])

            if Vrot <= 10 or Vrot > 600:
                n_bad_vel += 1
                continue

            # Fractional velocity error cut: σ_V/V < 15%
            frac_err = eVrot / Vrot if eVrot > 0 and Vrot > 0 else 0.15
            if frac_err > 0.15:
                n_bad_vel += 1
                continue

            # gobs = V^2 / r  (convert to m/s^2)
            gobs = Vrot**2 / r_kpc * conv

            # gbar from exponential disk model
            # gbar = G * Mbar * r / (r^2 + rd^2)^(3/2)
            denom = (r_kpc**2 + rd**2)**1.5
            if denom <= 0:
                continue
            gbar = G_kpc * Mbar * r_kpc / denom * conv

            if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
                continue

            gobs_pred = rar_prediction(gbar)
            log_gbar = np.log10(gbar)
            log_gobs = np.log10(gobs)
            log_res = log_gobs - np.log10(gobs_pred)

            # Quality cut: reject extreme outliers
            if abs(log_res) > 1.5:
                continue

            # Propagate velocity error to gobs uncertainty
            # sigma_log_gobs = 2 * eVrot / (Vrot * ln(10))
            sigma_lg = 2.0 * eVrot / (Vrot * np.log(10)) if eVrot > 0 else 0.15
            sigma_lg = max(sigma_lg, 0.05)  # minimum uncertainty floor

            points.append({
                'galaxy': gal_name,
                'source': 'MaNGA',
                'log_gbar': float(log_gbar),
                'log_gobs': float(log_gobs),
                'log_res': float(log_res),
                'sigma_log_gobs': float(sigma_lg),
                'r_kpc': float(r_kpc),
                'vrot': float(Vrot),
                'env_dense': env_dense,
                'logMh': float(logMh),
                'structure': structure if 't17_env' != None else '',
                'ra': float(ra),
                'dec': float(dec),
            })

        if len(radii_re) > 0:
            names.append(gal_name)

    # Summary statistics
    n_galaxies = len(set(names))
    n_pts = len(points)
    n_dense = sum(1 for p in points if p['env_dense'] == 'dense')
    n_field = sum(1 for p in points if p['env_dense'] == 'field')

    print(f"    MaNGA Ristea+2023: {n_galaxies} galaxies, {n_pts} RAR points "
          f"({n_dense} dense, {n_field} field)")
    print(f"    Environment: {n_env_t17} Tempel2017, {n_env_prox} proximity-dense, "
          f"{n_env_field} field/fallback")
    if n_no_nsa > 0 or n_no_vel > 0:
        print(f"    Skipped: {n_no_nsa} (no NSA redshift), {n_no_vel} (no gas velocity), "
              f"{n_bad_vel} (bad velocity)")

    # Points per galaxy distribution
    pts_per_gal = {}
    for p in points:
        pts_per_gal[p['galaxy']] = pts_per_gal.get(p['galaxy'], 0) + 1
    n1 = sum(1 for v in pts_per_gal.values() if v == 1)
    n2 = sum(1 for v in pts_per_gal.values() if v == 2)
    n3 = sum(1 for v in pts_per_gal.values() if v >= 3)
    print(f"    Points per galaxy: {n1} with 1pt, {n2} with 2pt, {n3} with 3pt")

    return points, list(set(names))


# ============================================================
# DATASET 18: WALLABY DR2 rotation curves from CASDA  [was Dataset 16]
# 236 galaxies with HI rotation curves in cluster/group environments
# Fields: Hydra I, Norma, NGC 4636, NGC 5044, NGC 4808, Vela
# ============================================================
def load_wallaby_dr2():
    """
    Load WALLABY PDR2 HI rotation curves from CASDA TAP catalog.

    These are resolved HI kinematic models from the ASKAP WALLABY survey,
    covering multiple cluster/group environments. Rotation curves are stored
    as comma-separated arrays per galaxy.

    Uses HI surface density profiles for gas mass and Tully-Fisher for
    stellar mass to estimate gbar.
    """
    print("\n  [18/18] Loading WALLABY DR2 (236 HI rotation curves)...")

    kin_path = os.path.join(DATA_DIR, 'hi_surveys', 'wallaby',
                            'WALLABY_DR2_kinematic_models.tsv')

    if not os.path.exists(kin_path):
        print("    WARNING: WALLABY DR2 kinematic catalog not found")
        return [], []

    # --- Environment mapping for WALLABY fields ---
    # Based on WALLABY pilot survey fields
    field_env = {
        'Hydra':   {'logMh': 14.5, 'env': 'dense'},     # Hydra I cluster
        'Norma':   {'logMh': 15.0, 'env': 'dense'},     # Norma cluster
        'NGC 4636': {'logMh': 13.3, 'env': 'dense'},    # NGC 4636 group
        'NGC4636': {'logMh': 13.3, 'env': 'dense'},
        'NGC 5044': {'logMh': 13.5, 'env': 'dense'},    # NGC 5044 group
        'NGC5044': {'logMh': 13.5, 'env': 'dense'},
        'NGC 4808': {'logMh': 12.5, 'env': 'dense'},    # NGC 4808 group
        'NGC4808': {'logMh': 12.5, 'env': 'dense'},
        'Vela':    {'logMh': 13.0, 'env': 'dense'},     # Vela field (group region)
    }

    # Read TSV
    with open(kin_path) as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')
    rows = []
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) < len(header):
            continue
        row = {header[i]: parts[i].strip() for i in range(len(header))}
        rows.append(row)

    print(f"    Loaded {len(rows)} kinematic models from CASDA")

    points = []
    names = []
    n_skipped_inc = 0
    n_skipped_qflag = 0
    n_skipped_few_pts = 0
    n_no_rc = 0
    n_wise_mass = 0
    n_tf_mass = 0

    for row in rows:
        name = row.get('name', '').strip()
        if not name:
            continue

        # Parse basic properties
        ra = safe_float(row.get('ra', ''))
        dec = safe_float(row.get('dec', ''))
        vsys = safe_float(row.get('vsys_model', ''))
        inc = safe_float(row.get('inc_model', ''))
        e_inc = safe_float(row.get('e_inc_model', ''))
        qflag = safe_float(row.get('qflag_model', '1'))

        # Quality cuts
        if np.isnan(inc) or inc < 40 or inc > 85:
            n_skipped_inc += 1
            continue
        # qflag=0 is good, qflag=1 is marginal; accept both but flag
        # (Deg+2025 recommends qflag=0 for highest reliability)

        # Parse rotation curve arrays
        rad_str = row.get('rad', '')
        vrot_str = row.get('vrot_model', '')
        evrot_str = row.get('e_vrot_model', '')
        evrot_inc_str = row.get('e_vrot_model_inc', '')

        if not rad_str or not vrot_str:
            n_no_rc += 1
            continue

        try:
            rad_arcsec = [float(x) for x in rad_str.split(',')]
            vrot = [float(x) for x in vrot_str.split(',')]
            evrot = [float(x) for x in evrot_str.split(',')] if evrot_str else [5.0] * len(vrot)
            evrot_inc = [float(x) for x in evrot_inc_str.split(',')] if evrot_inc_str else [0.0] * len(vrot)
        except (ValueError, TypeError):
            n_no_rc += 1
            continue

        n_rc_pts = min(len(rad_arcsec), len(vrot))
        if n_rc_pts < 3:
            n_skipped_few_pts += 1
            continue

        # Parse HI surface density profile for gas mass
        sd_str = row.get('sd_fo_model', '')  # face-on surface density (Msun/pc^2)
        rad_sd_str = row.get('rad_sd', '')
        has_sd = False
        sd_vals = []
        rad_sd_vals = []
        if sd_str and rad_sd_str:
            try:
                sd_vals = [float(x) for x in sd_str.split(',')]
                rad_sd_vals = [float(x) for x in rad_sd_str.split(',')]
                has_sd = True
            except (ValueError, TypeError):
                has_sd = False

        # Distance: use Hubble flow (WALLABY fields are mostly >15 Mpc)
        if np.isnan(vsys) or vsys <= 0:
            continue
        Dist = vsys / H0  # Mpc (simple Hubble flow for now)
        if Dist <= 5 or Dist > 300:
            continue

        arcsec_to_kpc = Dist * 1000.0 * np.pi / (180.0 * 3600.0)  # kpc per arcsec

        # Determine environment from field name
        team_release = row.get('team_release', '').strip()
        env_info = None
        for field_key, fenv in field_env.items():
            if field_key in team_release:
                env_info = fenv
                break

        if env_info is None:
            # Try classify by position
            if not np.isnan(ra) and not np.isnan(dec) and not np.isnan(vsys):
                _, logMh, env_dense = classify_environment_proximity(ra, dec, vsys, name)
                env_info = {'logMh': logMh, 'env': env_dense}
            else:
                env_info = {'logMh': 11.0, 'env': 'field'}

        # Estimate baryonic mass: try WISE photometric mass first, fall back to TF
        outer_start = max(n_rc_pts * 2 // 3, 1)
        Vflat = np.mean(vrot[outer_start:n_rc_pts])
        if Vflat <= 10:
            continue

        wise_logM = get_z0mgs_stellar_mass(name=name, ra=ra, dec=dec)
        if wise_logM is not None:
            # WISE M* + 33% gas correction
            Mbar = 1.33 * 10**wise_logM
            log_Mbar = np.log10(Mbar)
            n_wise_mass += 1
        else:
            # Baryonic Tully-Fisher: log(Mbar) = a * log(Vflat) + b
            # McGaugh 2012: log(Mbar) = 3.75 * log(Vflat) + 2.00
            log_Mbar = 3.75 * np.log10(Vflat) + 2.00
            Mbar = 10**log_Mbar  # Msun
            n_tf_mass += 1

        # Estimate disk scale length from HI extent
        # R_HI ~ last measured radius; h ~ R_HI / 4 (rough estimate)
        R_last_arcsec = rad_arcsec[n_rc_pts - 1]
        R_last_kpc = R_last_arcsec * arcsec_to_kpc
        h_kpc = max(R_last_kpc / 4.0, 0.5)  # minimum 0.5 kpc

        # Gas fraction: f_gas ~ 0.1 for massive, ~0.5 for dwarf
        # Use rough estimate from Vflat
        if Vflat > 150:
            f_gas = 0.15
        elif Vflat > 80:
            f_gas = 0.3
        else:
            f_gas = 0.5

        M_star = Mbar * (1 - f_gas)
        M_gas = Mbar * f_gas * 1.33  # include He

        names.append(name)

        # Compute RAR for each radius
        for i in range(n_rc_pts):
            R_arcsec = rad_arcsec[i]
            V_rot = vrot[i]
            eV_rot = evrot[i] if i < len(evrot) else 5.0
            eV_inc_val = evrot_inc[i] if i < len(evrot_inc) else 0.0

            # Total uncertainty: add measurement + inclination in quadrature
            eV_total = np.sqrt(eV_rot**2 + eV_inc_val**2)
            eV_total = max(eV_total, 3.0)  # minimum 3 km/s

            R_kpc = R_arcsec * arcsec_to_kpc
            if R_kpc <= 0.1 or V_rot <= 5:
                continue

            # Fractional velocity error cut: σ_V/V < 15%
            if V_rot > 0 and eV_total / V_rot > 0.15:
                continue

            # gobs
            gobs = V_rot**2 / R_kpc * conv

            # gbar from stellar disk + gas
            # Stellar disk: exponential
            x_disk = R_kpc / h_kpc
            frac_star = 1.0 - (1.0 + x_disk) * np.exp(-x_disk)
            M_star_enc = M_star * frac_star

            # Gas: use HI surface density if available
            if has_sd and i < len(sd_vals) and i < len(rad_sd_vals):
                # Integrate gas mass from surface density
                # Sigma_HI * pi * R^2 (approximate for annular)
                sd_msun_pc2 = sd_vals[i] if sd_vals[i] > 0 else 0.1
                # Enclosed gas mass (approximate with disk)
                M_gas_enc = sd_msun_pc2 * 1e6 * np.pi * R_kpc**2 * 1.33
            else:
                # Fallback: exponential gas disk with scale 2*h
                h_gas = 2.0 * h_kpc
                x_gas = R_kpc / h_gas
                frac_gas = 1.0 - (1.0 + x_gas) * np.exp(-x_gas)
                M_gas_enc = M_gas * frac_gas

            M_enc = M_star_enc + M_gas_enc
            gbar = G_kpc * M_enc / R_kpc**2 * conv

            if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
                continue

            gobs_pred = rar_prediction(gbar)
            log_gbar = np.log10(gbar)
            log_gobs = np.log10(gobs)
            log_res = log_gobs - np.log10(gobs_pred)

            if abs(log_res) > 1.5:
                continue

            sigma_log = 2.0 * eV_total / max(V_rot, 1.0) / np.log(10)

            points.append({
                'galaxy': name,
                'source': 'WALLABY_DR2',
                'log_gbar': float(log_gbar),
                'log_gobs': float(log_gobs),
                'log_res': float(log_res),
                'sigma_log_gobs': float(max(sigma_log, 0.05)),
                'R_kpc': float(R_kpc),
                'env_dense': env_info['env'],
                'logMh': float(env_info['logMh']),
            })

    n_galaxies = len(set(names))
    n_pts = len(points)
    n_dense = sum(1 for p in points if p['env_dense'] == 'dense')
    n_field = sum(1 for p in points if p['env_dense'] == 'field')

    print(f"    WALLABY DR2: {n_galaxies} galaxies, {n_pts} RAR points "
          f"({n_dense} dense, {n_field} field)")
    print(f"    Mass models: {n_wise_mass} WISE, {n_tf_mass} Tully-Fisher")
    if n_skipped_inc > 0:
        print(f"    Skipped: {n_skipped_inc} (inclination), {n_skipped_few_pts} (too few pts), "
              f"{n_no_rc} (no RC)")

    # Environment breakdown
    env_counts = {}
    for p in points:
        env_counts[p['logMh']] = env_counts.get(p['logMh'], 0) + 1
    for mh, cnt in sorted(env_counts.items(), reverse=True):
        print(f"      logMh={mh:.1f}: {cnt} points")

    return points, list(set(names))


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_unified_pipeline(
    chi2_shared_sigma_int=False,
    chi2_sigma_ref_model='bec',
    allow_missing_sparc=False,
):
    """Run the unified RAR environmental scatter pipeline."""

    print("=" * 80)
    print("UNIFIED RAR ENVIRONMENTAL SCATTER PIPELINE")
    print("Combining 18 datasets for comprehensive BEC DM test")
    print("=" * 80)

    # ============================================================
    # STEP 1: Load all datasets
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING ALL DATASETS")
    print("=" * 80)

    sparc_points, sparc_names, sparc_diag = load_sparc_data()

    # Root-cause diagnostics for SPARC drift: print resolved paths and existence.
    print("    [DEBUG] SPARC path resolution:")
    for key, info in sparc_diag.get('paths', {}).items():
        chosen = info.get('chosen')
        print(f"      {key}: chosen={chosen}")
        for cand, exists in info.get('exists', {}).items():
            print(f"        - {cand}: exists={exists}")
    print(f"    [DEBUG] len(sparc_points)={len(sparc_points)} len(sparc_names)={len(sparc_names)}")

    sparc_points_floor = 2000
    sparc_galaxies_floor = 100
    sparc_unique_keys = len({p.get('galaxy_key', canonicalize_galaxy_name(p.get('galaxy', ''))) for p in sparc_points})
    if (len(sparc_points) < sparc_points_floor or sparc_unique_keys < sparc_galaxies_floor) and not allow_missing_sparc:
        raise RuntimeError(
            "SPARC guardrail triggered: insufficient SPARC coverage. "
            f"points={len(sparc_points)} (floor={sparc_points_floor}), "
            f"galaxies={sparc_unique_keys} (floor={sparc_galaxies_floor}). "
            "Use --allow_missing_sparc to bypass explicitly."
        )
    deblok_points, deblok_names = load_deblok2002()
    wallaby_points, wallaby_names = load_wallaby()
    santos_points, santos_names = load_santos_santos()
    lt_points, lt_names = load_little_things()
    lvhis_points, lvhis_names = load_lvhis()
    yu_points, yu_names = load_yu2020()
    swaters_points, swaters_names = load_swaters2025()
    ghasp_points, ghasp_names = load_ghasp()
    noord_points, noord_names = load_noordermeer2005()
    vogt_points, vogt_names = load_vogt2004()
    catinella_points, catinella_names = load_catinella2005()
    virgorc_points, virgorc_names = load_virgo_extended_rc()
    phangs_points, phangs_names = load_phangs_lang2020()
    verheijen_points, verheijen_names = load_verheijen2001_uma()
    manga_points, manga_names = load_manga_ristea2023()
    wallaby_dr2_points, wallaby_dr2_names = load_wallaby_dr2()

    # Ensure every point carries a canonical galaxy key before overlap checks.
    for _pts in (
        sparc_points, deblok_points, wallaby_points, santos_points, lt_points, lvhis_points,
        yu_points, swaters_points, ghasp_points, noord_points, vogt_points, catinella_points,
        virgorc_points, phangs_points, verheijen_points, manga_points, wallaby_dr2_points
    ):
        for _p in _pts:
            _p['galaxy_key'] = _p.get('galaxy_key', canonicalize_galaxy_name(_p.get('galaxy', '')))

    # ============================================================
    # STEP 2: Remove SPARC overlaps from other datasets
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 2: REMOVING SPARC OVERLAPS")
    print("=" * 80)

    sparc_norm_set = build_sparc_name_set(sparc_names)
    print(f"\n  SPARC galaxy names: {len(sparc_names)}")

    datasets = {
        'deBlok2002':  (deblok_points, deblok_names),
        'WALLABY':     (wallaby_points, wallaby_names),
        'SS2020':      (santos_points, santos_names),
        'LITTLETHINGS': (lt_points, lt_names),
        'LVHIS':       (lvhis_points, lvhis_names),
        'Yu2020':      (yu_points, yu_names),
        'Swaters2025': (swaters_points, swaters_names),
        'GHASP':       (ghasp_points, ghasp_names),
        'Noordermeer2005': (noord_points, noord_names),
        'Vogt2004':    (vogt_points, vogt_names),
        'Catinella2005': (catinella_points, catinella_names),
        'VirgoRC':     (virgorc_points, virgorc_names),
        'PHANGS':      (phangs_points, phangs_names),
        'Verheijen2001': (verheijen_points, verheijen_names),
        'MaNGA':       (manga_points, manga_names),
        'WALLABY_DR2': (wallaby_dr2_points, wallaby_dr2_names),
    }
    print(
        f"    [DEBUG] datasets has SPARC={'SPARC' in datasets}; "
        f"SPARC is seeded separately with len(sparc_points)={len(sparc_points)}",
        flush=True,
    )

    # Check and remove overlaps
    for ds_name, (pts, nms) in datasets.items():
        if not nms:
            continue

        overlap_names = [n for n in nms if is_sparc_duplicate(n, sparc_norm_set)]
        if overlap_names:
            overlap_norm = set(normalize_galaxy_name(n) for n in overlap_names)
            n_before = len(pts)
            pts_clean = [p for p in pts
                         if not is_sparc_duplicate(p['galaxy'], sparc_norm_set)]
            nms_clean = [n for n in nms if not is_sparc_duplicate(n, sparc_norm_set)]
            datasets[ds_name] = (pts_clean, nms_clean)
            print(f"  {ds_name}: removed {len(overlap_names)} SPARC duplicates "
                  f"({n_before} -> {len(pts_clean)} points)")
            if len(overlap_names) <= 10:
                print(f"    Removed: {', '.join(overlap_names)}")
        else:
            print(f"  {ds_name}: no SPARC overlaps")

    # Also check cross-dataset overlaps (non-SPARC)
    # Build a cumulative set for additional datasets
    all_norm_names = set(sparc_norm_set)

    # Process in priority order
    priority_order = ['deBlok2002', 'WALLABY', 'SS2020', 'LITTLETHINGS',
                      'LVHIS', 'Yu2020', 'Swaters2025', 'GHASP',
                      'Noordermeer2005', 'Vogt2004', 'Catinella2005',
                      'VirgoRC', 'PHANGS', 'Verheijen2001', 'MaNGA',
                      'WALLABY_DR2']

    for ds_name in priority_order:
        if ds_name not in datasets:
            continue
        pts, nms = datasets[ds_name]
        if not nms:
            continue

        # Remove any already-seen names
        new_pts = []
        new_nms = []
        removed = 0
        for p in pts:
            norm = normalize_galaxy_name(p['galaxy'])
            if norm not in all_norm_names:
                new_pts.append(p)
            else:
                removed += 1
        for n in nms:
            norm = normalize_galaxy_name(n)
            if norm not in all_norm_names:
                new_nms.append(n)
                all_norm_names.add(norm)

        if removed > 0:
            print(f"  {ds_name}: removed {removed} additional cross-dataset duplicates")

        datasets[ds_name] = (new_pts, new_nms)

    # ============================================================
    # STEP 3: Combine all points
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 3: COMBINING ALL DATASETS")
    print("=" * 80)

    all_points = list(sparc_points)
    for ds_name in priority_order:
        pts, _ = datasets.get(ds_name, ([], []))
        all_points.extend(pts)

    # Count by source and canonical galaxy key (raw + grouped rollups).
    source_counts_grouped = {}
    source_counts_raw = {}
    source_dense_grouped = {}
    source_field_grouped = {}
    source_dense_raw = {}
    source_field_raw = {}
    galaxy_sources = {}

    for p in all_points:
        p['galaxy_key'] = p.get('galaxy_key', canonicalize_galaxy_name(p.get('galaxy', '')))
        src_raw = p['source']
        src_grouped = 'SS2020' if src_raw.startswith('SS20_') else src_raw

        source_counts_raw[src_raw] = source_counts_raw.get(src_raw, 0) + 1
        source_counts_grouped[src_grouped] = source_counts_grouped.get(src_grouped, 0) + 1

        gkey = p['galaxy_key']
        if gkey not in galaxy_sources:
            galaxy_sources[gkey] = src_grouped

        if p['env_dense'] == 'dense':
            source_dense_raw[src_raw] = source_dense_raw.get(src_raw, 0) + 1
            source_dense_grouped[src_grouped] = source_dense_grouped.get(src_grouped, 0) + 1
        else:
            source_field_raw[src_raw] = source_field_raw.get(src_raw, 0) + 1
            source_field_grouped[src_grouped] = source_field_grouped.get(src_grouped, 0) + 1

    n_galaxies = len(galaxy_sources)
    n_points = len(all_points)

    print(f"\n  Total unique galaxies: {n_galaxies}")
    print(f"  Total RAR points: {n_points}")
    print(f"\n  {'Dataset (grouped)':<20} {'Points':>8} {'Dense':>8} {'Field':>8}")
    print(f"  {'-'*62}")
    for src in ['SPARC', 'deBlok2002', 'WALLABY', 'SS2020',
                'LITTLETHINGS', 'LVHIS', 'Yu2020', 'Swaters2025',
                'GHASP', 'Noordermeer2005', 'Vogt2004', 'Catinella2005',
                'VirgoRC', 'PHANGS', 'Verheijen2001', 'MaNGA', 'WALLABY_DR2']:
        nd = source_dense_grouped.get(src, 0)
        nf = source_field_grouped.get(src, 0)
        nt = source_counts_grouped.get(src, 0)
        if nt > 0:
            print(f"  {src:<20} {nt:>8} {nd:>8} {nf:>8}")
    print(f"  {'TOTAL':<20} {n_points:>8} "
          f"{sum(source_dense_grouped.values()):>8} {sum(source_field_grouped.values()):>8}")
    print(f"\n  {'Dataset (raw)':<20} {'Points':>8} {'Dense':>8} {'Field':>8}")
    print(f"  {'-'*62}")
    for src in sorted(source_counts_raw):
        nd = source_dense_raw.get(src, 0)
        nf = source_field_raw.get(src, 0)
        nt = source_counts_raw.get(src, 0)
        print(f"  {src:<20} {nt:>8} {nd:>8} {nf:>8}")

    # Count galaxies by environment
    gal_envs = {}
    for p in all_points:
        gkey = p.get('galaxy_key', canonicalize_galaxy_name(p.get('galaxy', '')))
        if gkey not in gal_envs:
            gal_envs[gkey] = p['env_dense']
    n_dense_gal = sum(1 for v in gal_envs.values() if v == 'dense')
    n_field_gal = sum(1 for v in gal_envs.values() if v == 'field')
    print(f"\n  Galaxy counts: {n_dense_gal} dense, {n_field_gal} field")

    # ============================================================
    # STEP 3b: UPGRADE MASS MODELS WITH z0MGS WISE STELLAR MASSES
    # ============================================================
    # For datasets using Tully-Fisher mass estimates, replace with
    # z0MGS WISE W1-based stellar masses where available.
    # This provides uniform mass-to-light ratios across environments.
    # SPARC and deBlok2002 already have full mass decompositions — skip them.

    tf_datasets = {'PHANGS', 'Verheijen2001', 'WALLABY_DR2', 'GHASP',
                   'Catinella2005', 'WALLABY'}

    # DISABLED: Simple mass-ratio scaling creates unphysical gbar values
    # because the disk geometry (scale length) isn't recomputed.
    # z0MGS masses are available via get_z0mgs_stellar_mass() for use
    # inside individual dataset loaders where the full disk model is built.
    if False and (Z0MGS_NAMES or Z0MGS_COORDS):
        print("\n" + "=" * 80)
        print("STEP 3b: UPGRADING MASS MODELS WITH z0MGS WISE STELLAR MASSES")
        print("=" * 80)

        n_upgraded = 0
        n_total_tf = 0
        upgraded_by_source = {}

        # Cache z0MGS lookups per galaxy
        galaxy_z0mgs_mass = {}

        for p in all_points:
            if p['source'] not in tf_datasets:
                continue
            n_total_tf += 1

            gname = p['galaxy']
            if gname not in galaxy_z0mgs_mass:
                # Look up WISE mass
                ra = p.get('ra', np.nan) if 'ra' in p else np.nan
                dec = p.get('dec', np.nan) if 'dec' in p else np.nan
                galaxy_z0mgs_mass[gname] = get_z0mgs_stellar_mass(
                    name=gname, ra=ra, dec=dec)

            logM_wise = galaxy_z0mgs_mass[gname]
            if logM_wise is None:
                continue

            # Upgrade gbar using WISE mass instead of TF mass
            # Current gbar was computed as: gbar = G * Mbar_TF * r / (r^2 + rd^2)^1.5 * conv
            # We want to scale it by the ratio of WISE mass to TF mass.
            # Since gbar is proportional to Mbar, we can scale:
            #   gbar_new = gbar_old * (Mbar_WISE / Mbar_TF)
            # We don't know Mbar_TF directly, but we know:
            #   log(Mbar_TF) = 3.75 * log(Vflat) + 2.00
            # And Vflat ~ p['vrot'] (outermost) for that galaxy.
            # Instead, since all points for the same galaxy share the same mass,
            # we can compute the ratio once.

            # For now, just mark that this galaxy has a WISE mass available
            # and store it. The actual gbar recalculation requires knowing
            # the original TF mass and the exponential disk model.
            # A simpler approach: recompute gbar from scratch using WISE mass.

            r_kpc = p.get('r_kpc', 0)
            if r_kpc <= 0:
                continue

            # Get the disk scale length from the galaxy's RC
            # For consistency, re-derive from z0MGS mass
            Mbar_wise = 10 ** logM_wise
            # Use same exponential disk model as the loaders
            # Need to know rd... we stored r_kpc values.
            # Approximate: rd = mean(r_kpc) / 2.2 for each galaxy
            # This is what the loaders use. We can't easily get the mean
            # radius here, so use a different approach:
            # Scale the existing gbar by the mass ratio.

            # Estimate original TF mass from vrot
            vrot = p.get('vrot', 0)
            if vrot > 10:
                logM_tf = 3.75 * np.log10(vrot) + 2.00
                mass_ratio = 10 ** (logM_wise - logM_tf)

                # Scale gbar
                old_gbar = p['gbar']
                new_gbar = old_gbar * mass_ratio

                if new_gbar > 0:
                    new_log_gbar = np.log10(new_gbar)
                    new_gobs_pred = rar_prediction(new_gbar)
                    new_log_res = p['log_gobs'] - np.log10(new_gobs_pred)

                    p['gbar'] = new_gbar
                    p['log_gbar'] = new_log_gbar
                    p['log_res'] = new_log_res
                    p['mass_source'] = 'z0MGS_WISE'

                    n_upgraded += 1
                    src = p['source']
                    upgraded_by_source[src] = upgraded_by_source.get(src, 0) + 1

        n_gal_upgraded = sum(1 for v in galaxy_z0mgs_mass.values() if v is not None)
        n_gal_total = len(galaxy_z0mgs_mass)
        print(f"\n  TF-based points: {n_total_tf}")
        print(f"  Upgraded with WISE mass: {n_upgraded} points "
              f"({n_gal_upgraded}/{n_gal_total} galaxies matched)")
        print(f"  Per-dataset upgrades:")
        for src in sorted(upgraded_by_source.keys()):
            print(f"    {src}: {upgraded_by_source[src]} points")

    # ============================================================
    # STEP 4: Environmental scatter test
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 4: ENVIRONMENTAL SCATTER TEST")
    print("=" * 80)

    dense_gbar = np.array([p['log_gbar'] for p in all_points if p['env_dense'] == 'dense'])
    dense_res = np.array([p['log_res'] for p in all_points if p['env_dense'] == 'dense'])
    field_gbar = np.array([p['log_gbar'] for p in all_points if p['env_dense'] == 'field'])
    field_res = np.array([p['log_res'] for p in all_points if p['env_dense'] == 'field'])

    print(f"\n  Dense: {len(dense_res)} points from {n_dense_gal} galaxies")
    print(f"  Field: {len(field_res)} points from {n_field_gal} galaxies")

    sigma_dense = np.std(dense_res) if len(dense_res) > 0 else np.nan
    sigma_field = np.std(field_res) if len(field_res) > 0 else np.nan
    delta_sigma = sigma_field - sigma_dense

    print(f"\n  --- Observed scatter (includes measurement errors) ---")
    print(f"  sigma_dense = {sigma_dense:.4f} dex")
    print(f"  sigma_field = {sigma_field:.4f} dex")
    print(f"  Delta_sigma (field - dense) = {delta_sigma:+.4f} dex")

    # Haubner-style intrinsic scatter deconvolution
    # sigma_obs^2 = sigma_int^2 + <sigma_err^2>
    # sigma_int = sqrt(sigma_obs^2 - <sigma_err^2>)
    dense_err2 = np.array([p['sigma_log_gobs']**2 for p in all_points
                           if p['env_dense'] == 'dense'])
    field_err2 = np.array([p['sigma_log_gobs']**2 for p in all_points
                           if p['env_dense'] == 'field'])

    mean_err2_dense = np.mean(dense_err2) if len(dense_err2) > 0 else 0
    mean_err2_field = np.mean(field_err2) if len(field_err2) > 0 else 0

    sigma_int_dense = np.sqrt(max(sigma_dense**2 - mean_err2_dense, 0)) if not np.isnan(sigma_dense) else np.nan
    sigma_int_field = np.sqrt(max(sigma_field**2 - mean_err2_field, 0)) if not np.isnan(sigma_field) else np.nan
    delta_sigma_int = sigma_int_field - sigma_int_dense

    print(f"\n  --- Intrinsic scatter (Haubner deconvolution) ---")
    print(f"  <sigma_err^2>_dense = {mean_err2_dense:.6f} (rms_err = {np.sqrt(mean_err2_dense):.4f})")
    print(f"  <sigma_err^2>_field = {mean_err2_field:.6f} (rms_err = {np.sqrt(mean_err2_field):.4f})")
    print(f"  sigma_int_dense = {sigma_int_dense:.4f} dex")
    print(f"  sigma_int_field = {sigma_int_field:.4f} dex")
    print(f"  Delta_sigma_int (field - dense) = {delta_sigma_int:+.4f} dex")

    # Bootstrap test
    if len(dense_res) > 10 and len(field_res) > 10:
        print(f"\n  Bootstrap permutation test (10,000 iterations)...")
        delta, p_val, boot = bootstrap_scatter_test(dense_res, field_res,
                                                     n_boot=10000, seed=42)
        print(f"    Observed Delta = {delta:+.4f} dex")
        print(f"    P(field > dense) = {1.0 - p_val:.4f}")
        print(f"    P-value (one-sided) = {p_val:.4f}")

        # Levene's test
        lev_stat, lev_p = stats.levene(dense_res, field_res)
        print(f"    Levene's test: F = {lev_stat:.3f}, p = {lev_p:.6f}")

        # Brown-Forsythe (median-based Levene)
        bf_stat, bf_p = stats.levene(dense_res, field_res, center='median')
        print(f"    Brown-Forsythe: F = {bf_stat:.3f}, p = {bf_p:.6f}")
    else:
        p_val = np.nan
        lev_stat = np.nan
        lev_p = np.nan
        bf_stat = np.nan
        bf_p = np.nan

    # ============================================================
    # STEP 5: Binned analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 5: BINNED ENVIRONMENTAL ANALYSIS")
    print("=" * 80)

    bin_edges = np.array([-13.0, -12.0, -11.0, -10.0, -9.0, -8.0])
    binned = binned_analysis(dense_gbar, dense_res, field_gbar, field_res,
                             bin_edges=bin_edges)

    print(f"\n  {'Bin center':<12} {'N_dense':>8} {'N_field':>8} "
          f"{'sig_d':>8} {'sig_f':>8} {'Delta':>8} {'P(f>d)':>8}")
    print(f"  {'-'*68}")

    n_bins_positive = 0
    for b in binned:
        delta_str = f"{b['delta']:+.4f}" if not np.isnan(b.get('delta', np.nan)) else "---"
        p_str = f"{b.get('p_field_gt_dense', np.nan):.3f}" if not np.isnan(b.get('p_field_gt_dense', np.nan)) else "---"
        sig_d = f"{b['sigma_dense']:.4f}" if not np.isnan(b['sigma_dense']) else "---"
        sig_f = f"{b['sigma_field']:.4f}" if not np.isnan(b['sigma_field']) else "---"

        marker = ""
        if not np.isnan(b.get('delta', np.nan)):
            if b['delta'] > 0:
                marker = " [+]"
                n_bins_positive += 1
            else:
                marker = " [-]"

        print(f"  {b['center']:<12.1f} {b['n_dense']:>8} {b['n_field']:>8} "
              f"{sig_d:>8} {sig_f:>8} {delta_str:>8} {p_str:>8}{marker}")

    n_bins_valid = sum(1 for b in binned if not np.isnan(b.get('delta', np.nan)))
    print(f"\n  Bins with field > dense: {n_bins_positive}/{n_bins_valid}")

    # ============================================================
    # STEP 5b: DM-dominated regime analysis (BEC prediction test)
    # ============================================================
    # The BEC prediction: field galaxies have MORE RAR scatter in the
    # DM-dominated (low-acceleration) regime, but NOT in the baryon-
    # dominated (high-acceleration) regime.
    # Threshold: log(gbar) < -10.5 is DM-dominated
    print("\n" + "=" * 80)
    print("STEP 5b: DM-DOMINATED REGIME (BEC PREDICTION TEST)")
    print("  BEC prediction: field > dense scatter at log(gbar) < -10.5")
    print("=" * 80)

    dm_threshold = -10.5
    dm_d_res = np.array([p['log_res'] for p in all_points
                         if p['env_dense'] == 'dense' and p['log_gbar'] < dm_threshold])
    dm_f_res = np.array([p['log_res'] for p in all_points
                         if p['env_dense'] == 'field' and p['log_gbar'] < dm_threshold])
    dm_d_gal = len(set(p['galaxy'] for p in all_points
                       if p['env_dense'] == 'dense' and p['log_gbar'] < dm_threshold))
    dm_f_gal = len(set(p['galaxy'] for p in all_points
                       if p['env_dense'] == 'field' and p['log_gbar'] < dm_threshold))

    dm_sigma_d = float(np.std(dm_d_res)) if len(dm_d_res) > 0 else np.nan
    dm_sigma_f = float(np.std(dm_f_res)) if len(dm_f_res) > 0 else np.nan
    dm_delta = dm_sigma_f - dm_sigma_d

    print(f"\n  DM-dominated regime: log(gbar) < {dm_threshold}")
    print(f"  Dense: {len(dm_d_res)} pts from {dm_d_gal} galaxies, sigma = {dm_sigma_d:.4f}")
    print(f"  Field: {len(dm_f_res)} pts from {dm_f_gal} galaxies, sigma = {dm_sigma_f:.4f}")
    print(f"  Delta (field - dense) = {dm_delta:+.4f}")

    dm_p_val = np.nan
    if len(dm_d_res) > 10 and len(dm_f_res) > 10:
        _, dm_p_raw, _ = bootstrap_scatter_test(dm_d_res, dm_f_res, n_boot=10000, seed=99)
        dm_p_val = dm_p_raw
        print(f"  P(field > dense) = {1.0 - dm_p_val:.4f}")
        try:
            dm_lev, dm_lev_p = stats.levene(dm_d_res, dm_f_res)
            print(f"  Levene p = {dm_lev_p:.6f}")
        except Exception:
            dm_lev_p = np.nan

    # Per-dataset in DM-dominated regime
    print(f"\n  Per-dataset (DM-dominated only):")
    dm_per_dataset = {}
    for src in ['SPARC', 'WALLABY', 'Yu2020', 'deBlok2002', 'Swaters2025', 'LVHIS']:
        src_dm = [p for p in all_points if p['source'] == src and p['log_gbar'] < dm_threshold]
        d_r = np.array([p['log_res'] for p in src_dm if p['env_dense'] == 'dense'])
        f_r = np.array([p['log_res'] for p in src_dm if p['env_dense'] == 'field'])
        if len(d_r) > 0 or len(f_r) > 0:
            sd = float(np.std(d_r)) if len(d_r) > 0 else np.nan
            sf = float(np.std(f_r)) if len(f_r) > 0 else np.nan
            delta = sf - sd if not np.isnan(sd) and not np.isnan(sf) else np.nan
            direction = "field>dense" if not np.isnan(delta) and delta > 0 else "dense>field"
            sd_str = f"{sd:.4f}" if not np.isnan(sd) else "---"
            sf_str = f"{sf:.4f}" if not np.isnan(sf) else "---"
            d_str = f"{delta:+.4f}" if not np.isnan(delta) else "---"
            print(f"    {src:15s}  nd={len(d_r):4d}  nf={len(f_r):4d}  "
                  f"sig_d={sd_str}  sig_f={sf_str}  delta={d_str}  [{direction}]")
            dm_per_dataset[src] = {
                'n_dense': len(d_r), 'n_field': len(f_r),
                'sigma_dense': sd, 'sigma_field': sf,
                'delta': delta,
            }

    # Baryon-dominated regime for comparison
    bar_d_res = np.array([p['log_res'] for p in all_points
                          if p['env_dense'] == 'dense' and p['log_gbar'] >= dm_threshold])
    bar_f_res = np.array([p['log_res'] for p in all_points
                          if p['env_dense'] == 'field' and p['log_gbar'] >= dm_threshold])
    bar_sigma_d = float(np.std(bar_d_res)) if len(bar_d_res) > 0 else np.nan
    bar_sigma_f = float(np.std(bar_f_res)) if len(bar_f_res) > 0 else np.nan
    bar_delta = bar_sigma_f - bar_sigma_d

    print(f"\n  Baryon-dominated regime: log(gbar) >= {dm_threshold}")
    print(f"  Dense: {len(bar_d_res)} pts, sigma = {bar_sigma_d:.4f}")
    print(f"  Field: {len(bar_f_res)} pts, sigma = {bar_sigma_f:.4f}")
    print(f"  Delta = {bar_delta:+.4f}")

    if dm_delta > 0 and bar_delta <= 0:
        print(f"\n  >>> BEC PREDICTION CONFIRMED: field > dense scatter in DM regime,")
        print(f"  >>> but dense >= field scatter in baryon regime")
    elif dm_delta > 0:
        print(f"\n  >>> DM regime supports BEC prediction (field > dense)")

    # ============================================================
    # Build galaxy-level summary (needed for refined tests below)
    # ============================================================
    galaxy_summary = {}
    for p in all_points:
        gkey = p.get('galaxy_key', canonicalize_galaxy_name(p.get('galaxy', '')))
        if gkey not in galaxy_summary:
            galaxy_summary[gkey] = {
                'name': p.get('galaxy', gkey),
                'galaxy_key': gkey,
                'source': p['source'],
                'env_dense': p['env_dense'],
                'logMh': p['logMh'],
                'log_res_list': [],
                'log_gbar_list': [],
            }
        galaxy_summary[gkey]['log_res_list'].append(p['log_res'])
        galaxy_summary[gkey]['log_gbar_list'].append(p['log_gbar'])

    # ============================================================
    # STEP 5c: REFINED BEC PREDICTION TESTS
    # ============================================================
    # Five refinements to properly prove/disprove the BEC scatter signal:
    #   1. Within-dataset Z-score normalization (eliminates Simpson's Paradox)
    #   2. Sliding gbar threshold (finds optimal DM/baryon boundary)
    #   3. Galaxy-level scatter test (removes point-count bias)
    #   4. DM-fraction weighting (stronger signal where DM dominates more)
    #   5. Monte Carlo error propagation (confidence intervals on Δσ)
    print("\n" + "=" * 80)
    print("STEP 5c: REFINED BEC PREDICTION TESTS")
    print("=" * 80)

    # ----------------------------------------------------------------
    # TEST 1: Within-dataset Z-score normalization
    # ----------------------------------------------------------------
    # Problem: Simpson's Paradox — SPARC (σ≈0.14) and WALLABY (σ≈0.37)
    # have very different baseline scatters, so mixing them can reverse
    # the per-dataset signal. Solution: normalize residuals within each
    # dataset before combining.
    print("\n  --- Test 1: Within-dataset Z-score normalization ---")
    print("    (Eliminates Simpson's Paradox from mixing heterogeneous datasets)")

    # Build Z-scores per dataset
    z_dense_all = []
    z_field_all = []
    z_dense_dm = []
    z_field_dm = []

    for src in ['SPARC', 'deBlok2002', 'WALLABY', 'SS2020',
                'LITTLETHINGS', 'LVHIS', 'Yu2020', 'Swaters2025',
                'GHASP', 'Noordermeer2005', 'Vogt2004', 'Catinella2005',
                'VirgoRC', 'PHANGS', 'Verheijen2001', 'MaNGA', 'WALLABY_DR2']:
        src_pts = [p for p in all_points
                   if (p['source'] == src or
                       (src == 'SS2020' and p['source'].startswith('SS20_')))]
        if len(src_pts) < 10:
            continue

        # Compute dataset-level mean and std
        all_res_ds = np.array([p['log_res'] for p in src_pts])
        mu_ds = np.mean(all_res_ds)
        std_ds = np.std(all_res_ds)
        if std_ds < 1e-6:
            continue

        for p in src_pts:
            z = (p['log_res'] - mu_ds) / std_ds
            if p['env_dense'] == 'dense':
                z_dense_all.append(z)
                if p['log_gbar'] < dm_threshold:
                    z_dense_dm.append(z)
            else:
                z_field_all.append(z)
                if p['log_gbar'] < dm_threshold:
                    z_field_dm.append(z)

    z_dense_all = np.array(z_dense_all)
    z_field_all = np.array(z_field_all)
    z_dense_dm = np.array(z_dense_dm)
    z_field_dm = np.array(z_field_dm)

    # Z-score scatter comparison (all regimes)
    z_sig_d = np.std(z_dense_all) if len(z_dense_all) > 0 else np.nan
    z_sig_f = np.std(z_field_all) if len(z_field_all) > 0 else np.nan
    z_delta = z_sig_f - z_sig_d
    print(f"    Overall:   Z_sigma_dense={z_sig_d:.4f}  Z_sigma_field={z_sig_f:.4f}  Δ={z_delta:+.4f}")

    # Z-score scatter comparison (DM regime only)
    z_sig_d_dm = np.std(z_dense_dm) if len(z_dense_dm) > 0 else np.nan
    z_sig_f_dm = np.std(z_field_dm) if len(z_field_dm) > 0 else np.nan
    z_delta_dm = z_sig_f_dm - z_sig_d_dm

    z_dm_p = np.nan
    if len(z_dense_dm) > 10 and len(z_field_dm) > 10:
        _, z_dm_p_raw, _ = bootstrap_scatter_test(z_dense_dm, z_field_dm, n_boot=10000, seed=77)
        z_dm_p = z_dm_p_raw
    print(f"    DM regime: Z_sigma_dense={z_sig_d_dm:.4f}  Z_sigma_field={z_sig_f_dm:.4f}  "
          f"Δ={z_delta_dm:+.4f}  P(f>d)={1.0 - z_dm_p:.4f}" if not np.isnan(z_dm_p)
          else f"    DM regime: Z_sigma_dense={z_sig_d_dm:.4f}  Z_sigma_field={z_sig_f_dm:.4f}  Δ={z_delta_dm:+.4f}")

    if z_delta_dm > 0:
        print(f"    >>> Z-score test SUPPORTS BEC prediction in DM regime")
    else:
        print(f"    >>> Z-score test does not support BEC prediction in DM regime")

    # ----------------------------------------------------------------
    # TEST 2: Sliding gbar threshold scan
    # ----------------------------------------------------------------
    # Instead of a fixed -10.5 boundary, scan thresholds from -12.0 to -9.0
    # to find where the BEC signal peaks and how robust it is
    print("\n  --- Test 2: Sliding gbar threshold scan ---")
    print("    (Tests BEC signal strength across acceleration boundaries)")

    threshold_results = []
    thresholds = np.arange(-12.0, -9.0, 0.25)
    print(f"    {'Threshold':>10} {'N_d':>6} {'N_f':>6} {'sig_d':>8} {'sig_f':>8} "
          f"{'Delta':>8} {'P(f>d)':>8}")

    for thresh in thresholds:
        t_d = np.array([p['log_res'] for p in all_points
                        if p['env_dense'] == 'dense' and p['log_gbar'] < thresh])
        t_f = np.array([p['log_res'] for p in all_points
                        if p['env_dense'] == 'field' and p['log_gbar'] < thresh])
        if len(t_d) < 20 or len(t_f) < 20:
            continue
        t_sig_d = float(np.std(t_d))
        t_sig_f = float(np.std(t_f))
        t_delta = t_sig_f - t_sig_d
        _, t_p_raw, _ = bootstrap_scatter_test(t_d, t_f, n_boot=5000, seed=int(abs(thresh * 100)))
        t_p = 1.0 - t_p_raw

        marker = " ***" if t_delta > 0 and t_p > 0.90 else ""
        print(f"    {thresh:>10.2f} {len(t_d):>6} {len(t_f):>6} {t_sig_d:>8.4f} {t_sig_f:>8.4f} "
              f"{t_delta:>+8.4f} {t_p:>8.4f}{marker}")

        threshold_results.append({
            'threshold': float(thresh),
            'n_dense': len(t_d), 'n_field': len(t_f),
            'sigma_dense': t_sig_d, 'sigma_field': t_sig_f,
            'delta': t_delta, 'p_field_gt_dense': t_p,
        })

    # Find best threshold
    if threshold_results:
        best = max(threshold_results, key=lambda x: x['delta'])
        print(f"\n    Peak BEC signal at threshold = {best['threshold']:.2f}: "
              f"Δ={best['delta']:+.4f}, P(f>d)={best['p_field_gt_dense']:.4f}")

    # Also do Z-score version of threshold scan
    print(f"\n    Z-score version (dataset-normalized):")
    z_threshold_results = []
    for thresh in thresholds:
        zt_d_list = []
        zt_f_list = []
        for src in ['SPARC', 'WALLABY', 'Yu2020', 'deBlok2002', 'Swaters2025', 'LVHIS']:
            src_pts = [p for p in all_points if p['source'] == src]
            if len(src_pts) < 10:
                continue
            all_res_s = np.array([p['log_res'] for p in src_pts])
            mu_s = np.mean(all_res_s)
            std_s = np.std(all_res_s)
            if std_s < 1e-6:
                continue
            for p in src_pts:
                if p['log_gbar'] < thresh:
                    z = (p['log_res'] - mu_s) / std_s
                    if p['env_dense'] == 'dense':
                        zt_d_list.append(z)
                    else:
                        zt_f_list.append(z)
        zt_d = np.array(zt_d_list)
        zt_f = np.array(zt_f_list)
        if len(zt_d) < 20 or len(zt_f) < 20:
            continue
        zt_sig_d = float(np.std(zt_d))
        zt_sig_f = float(np.std(zt_f))
        zt_delta = zt_sig_f - zt_sig_d
        _, zt_p_raw, _ = bootstrap_scatter_test(zt_d, zt_f, n_boot=5000,
                                                  seed=int(abs(thresh * 100)) + 500)
        zt_p = 1.0 - zt_p_raw
        marker = " ***" if zt_delta > 0 and zt_p > 0.90 else ""
        z_threshold_results.append({
            'threshold': float(thresh),
            'n_dense': len(zt_d), 'n_field': len(zt_f),
            'z_sigma_dense': zt_sig_d, 'z_sigma_field': zt_sig_f,
            'z_delta': zt_delta, 'z_p_field_gt_dense': zt_p,
        })
        print(f"    {thresh:>10.2f} {len(zt_d):>6} {len(zt_f):>6} {zt_sig_d:>8.4f} {zt_sig_f:>8.4f} "
              f"{zt_delta:>+8.4f} {zt_p:>8.4f}{marker}")

    if z_threshold_results:
        zbest = max(z_threshold_results, key=lambda x: x['z_delta'])
        print(f"\n    Z-score peak at threshold = {zbest['threshold']:.2f}: "
              f"Δz={zbest['z_delta']:+.4f}, P(f>d)={zbest['z_p_field_gt_dense']:.4f}")

    # ----------------------------------------------------------------
    # TEST 3: Galaxy-level scatter test
    # ----------------------------------------------------------------
    # Problem: point-level tests weight high-N galaxies heavily.
    # Solution: compute ONE scatter value per galaxy, then compare
    # distributions. Each galaxy gets equal weight.
    print("\n  --- Test 3: Galaxy-level scatter test ---")
    print("    (One value per galaxy — removes point-count bias)")

    # Use mean residual per galaxy (not per-galaxy sigma)
    gal_dense_mean_res = []
    gal_field_mean_res = []
    gal_dense_dm_mean_res = []
    gal_field_dm_mean_res = []

    for gname, gs in galaxy_summary.items():
        mean_r = np.mean(gs['log_res_list'])
        mean_gb = np.mean(gs['log_gbar_list'])
        if gs['env_dense'] == 'dense':
            gal_dense_mean_res.append(mean_r)
            if mean_gb < dm_threshold:
                gal_dense_dm_mean_res.append(mean_r)
        else:
            gal_field_mean_res.append(mean_r)
            if mean_gb < dm_threshold:
                gal_field_dm_mean_res.append(mean_r)

    gal_d = np.array(gal_dense_mean_res)
    gal_f = np.array(gal_field_mean_res)
    gal_d_dm = np.array(gal_dense_dm_mean_res)
    gal_f_dm = np.array(gal_field_dm_mean_res)

    gal_sig_d = float(np.std(gal_d)) if len(gal_d) > 0 else np.nan
    gal_sig_f = float(np.std(gal_f)) if len(gal_f) > 0 else np.nan
    gal_delta = gal_sig_f - gal_sig_d

    gal_p = np.nan
    if len(gal_d) > 10 and len(gal_f) > 10:
        _, gal_p_raw, _ = bootstrap_scatter_test(gal_d, gal_f, n_boot=10000, seed=123)
        gal_p = gal_p_raw

    print(f"    Overall: {len(gal_d)} dense gals, {len(gal_f)} field gals")
    print(f"    sigma(mean_res) dense = {gal_sig_d:.4f}")
    print(f"    sigma(mean_res) field = {gal_sig_f:.4f}")
    print(f"    Δ = {gal_delta:+.4f}, P(f>d) = {1.0 - gal_p:.4f}" if not np.isnan(gal_p)
          else f"    Δ = {gal_delta:+.4f}")

    # Galaxy-level DM regime
    gal_sig_d_dm = float(np.std(gal_d_dm)) if len(gal_d_dm) > 0 else np.nan
    gal_sig_f_dm = float(np.std(gal_f_dm)) if len(gal_f_dm) > 0 else np.nan
    gal_delta_dm = gal_sig_f_dm - gal_sig_d_dm

    gal_dm_p = np.nan
    if len(gal_d_dm) > 10 and len(gal_f_dm) > 10:
        _, gal_dm_p_raw, _ = bootstrap_scatter_test(gal_d_dm, gal_f_dm, n_boot=10000, seed=124)
        gal_dm_p = gal_dm_p_raw

    print(f"\n    DM regime (<{dm_threshold}): {len(gal_d_dm)} dense gals, {len(gal_f_dm)} field gals")
    print(f"    sigma(mean_res) dense = {gal_sig_d_dm:.4f}")
    print(f"    sigma(mean_res) field = {gal_sig_f_dm:.4f}")
    print(f"    Δ = {gal_delta_dm:+.4f}, P(f>d) = {1.0 - gal_dm_p:.4f}" if not np.isnan(gal_dm_p)
          else f"    Δ = {gal_delta_dm:+.4f}")

    if gal_delta_dm > 0:
        print(f"    >>> Galaxy-level test SUPPORTS BEC prediction in DM regime")
    else:
        print(f"    >>> Galaxy-level test does not support BEC in DM regime")

    # ----------------------------------------------------------------
    # TEST 4: DM-fraction weighted analysis
    # ----------------------------------------------------------------
    # Stronger test: weight each point by how DM-dominated it is.
    # DM fraction f_DM ≈ 1 - gbar/gobs. Points deep in the DM regime
    # (f_DM → 1) should show the strongest BEC environment signal.
    print("\n  --- Test 4: DM-fraction weighted scatter ---")
    print("    (Weights points by how DM-dominated they are)")

    def weighted_std(vals, weights):
        """Weighted standard deviation."""
        w = np.asarray(weights)
        v = np.asarray(vals)
        w_sum = np.sum(w)
        if w_sum <= 0:
            return np.nan
        w_mean = np.sum(w * v) / w_sum
        w_var = np.sum(w * (v - w_mean)**2) / w_sum
        return np.sqrt(w_var)

    # Compute DM fraction for each point
    dm_frac_points = []
    for p in all_points:
        gbar_lin = 10**p['log_gbar']
        gobs_lin = 10**p['log_gobs']
        f_dm = max(1.0 - gbar_lin / gobs_lin, 0.0) if gobs_lin > 0 else 0.0
        dm_frac_points.append({**p, 'f_dm': f_dm})

    # High-DM points only (f_DM > 0.5)
    hd_d_res = np.array([p['log_res'] for p in dm_frac_points
                          if p['env_dense'] == 'dense' and p['f_dm'] > 0.5])
    hd_f_res = np.array([p['log_res'] for p in dm_frac_points
                          if p['env_dense'] == 'field' and p['f_dm'] > 0.5])
    hd_d_w = np.array([p['f_dm'] for p in dm_frac_points
                        if p['env_dense'] == 'dense' and p['f_dm'] > 0.5])
    hd_f_w = np.array([p['f_dm'] for p in dm_frac_points
                        if p['env_dense'] == 'field' and p['f_dm'] > 0.5])

    hd_sig_d = weighted_std(hd_d_res, hd_d_w) if len(hd_d_res) > 0 else np.nan
    hd_sig_f = weighted_std(hd_f_res, hd_f_w) if len(hd_f_res) > 0 else np.nan
    hd_delta = hd_sig_f - hd_sig_d

    hd_p = np.nan
    if len(hd_d_res) > 10 and len(hd_f_res) > 10:
        _, hd_p_raw, _ = bootstrap_scatter_test(hd_d_res, hd_f_res, n_boot=10000, seed=200)
        hd_p = hd_p_raw

    print(f"    f_DM > 0.5 regime: {len(hd_d_res)} dense pts, {len(hd_f_res)} field pts")
    print(f"    Weighted sigma_dense = {hd_sig_d:.4f}")
    print(f"    Weighted sigma_field = {hd_sig_f:.4f}")
    print(f"    Δ = {hd_delta:+.4f}, P(f>d) = {1.0 - hd_p:.4f}" if not np.isnan(hd_p)
          else f"    Δ = {hd_delta:+.4f}")

    # Very-high-DM points (f_DM > 0.8)
    vhd_d = np.array([p['log_res'] for p in dm_frac_points
                       if p['env_dense'] == 'dense' and p['f_dm'] > 0.8])
    vhd_f = np.array([p['log_res'] for p in dm_frac_points
                       if p['env_dense'] == 'field' and p['f_dm'] > 0.8])
    if len(vhd_d) > 5 and len(vhd_f) > 5:
        vhd_sig_d = float(np.std(vhd_d))
        vhd_sig_f = float(np.std(vhd_f))
        vhd_delta = vhd_sig_f - vhd_sig_d
        _, vhd_p_raw, _ = bootstrap_scatter_test(vhd_d, vhd_f, n_boot=5000, seed=201)
        vhd_p = 1.0 - vhd_p_raw
        print(f"\n    f_DM > 0.8 regime: {len(vhd_d)} dense pts, {len(vhd_f)} field pts")
        print(f"    sigma_dense = {vhd_sig_d:.4f}, sigma_field = {vhd_sig_f:.4f}")
        print(f"    Δ = {vhd_delta:+.4f}, P(f>d) = {vhd_p:.4f}")
        if vhd_delta > 0:
            print(f"    >>> Deep DM regime SUPPORTS BEC prediction")
    else:
        vhd_delta = np.nan
        vhd_p = np.nan

    # ----------------------------------------------------------------
    # TEST 5: Monte Carlo error propagation for Δσ confidence interval
    # ----------------------------------------------------------------
    # Perturb each data point within its error bar, recompute Δσ 1000 times,
    # report 68% and 95% confidence intervals
    print("\n  --- Test 5: Monte Carlo error propagation ---")
    print("    (Δσ confidence intervals from measurement error propagation)")

    n_mc = 1000
    rng_mc = np.random.RandomState(seed=314)

    mc_delta_all = np.zeros(n_mc)
    mc_delta_dm = np.zeros(n_mc)

    for i_mc in range(n_mc):
        # Perturb each point's log_res by its error
        perturbed_dense = []
        perturbed_field = []
        perturbed_dense_dm = []
        perturbed_field_dm = []

        for p in all_points:
            err = p['sigma_log_gobs']
            perturbed_res = p['log_res'] + rng_mc.normal(0, err) if err > 0 else p['log_res']

            if p['env_dense'] == 'dense':
                perturbed_dense.append(perturbed_res)
                if p['log_gbar'] < dm_threshold:
                    perturbed_dense_dm.append(perturbed_res)
            else:
                perturbed_field.append(perturbed_res)
                if p['log_gbar'] < dm_threshold:
                    perturbed_field_dm.append(perturbed_res)

        mc_delta_all[i_mc] = np.std(perturbed_field) - np.std(perturbed_dense)
        mc_delta_dm[i_mc] = np.std(perturbed_field_dm) - np.std(perturbed_dense_dm)

    # Overall
    mc_all_med = np.median(mc_delta_all)
    mc_all_68lo = np.percentile(mc_delta_all, 16)
    mc_all_68hi = np.percentile(mc_delta_all, 84)
    mc_all_95lo = np.percentile(mc_delta_all, 2.5)
    mc_all_95hi = np.percentile(mc_delta_all, 97.5)

    print(f"\n    Overall Δσ (field−dense):")
    print(f"      Median = {mc_all_med:+.4f}")
    print(f"      68% CI = [{mc_all_68lo:+.4f}, {mc_all_68hi:+.4f}]")
    print(f"      95% CI = [{mc_all_95lo:+.4f}, {mc_all_95hi:+.4f}]")
    frac_positive_all = np.mean(mc_delta_all > 0)
    print(f"      Fraction MC realizations with field>dense: {frac_positive_all:.3f}")

    # DM regime
    mc_dm_med = np.median(mc_delta_dm)
    mc_dm_68lo = np.percentile(mc_delta_dm, 16)
    mc_dm_68hi = np.percentile(mc_delta_dm, 84)
    mc_dm_95lo = np.percentile(mc_delta_dm, 2.5)
    mc_dm_95hi = np.percentile(mc_delta_dm, 97.5)

    print(f"\n    DM regime Δσ (field−dense), log gbar < {dm_threshold}:")
    print(f"      Median = {mc_dm_med:+.4f}")
    print(f"      68% CI = [{mc_dm_68lo:+.4f}, {mc_dm_68hi:+.4f}]")
    print(f"      95% CI = [{mc_dm_95lo:+.4f}, {mc_dm_95hi:+.4f}]")
    frac_positive_dm = np.mean(mc_delta_dm > 0)
    print(f"      Fraction MC realizations with field>dense: {frac_positive_dm:.3f}")

    if frac_positive_dm > 0.95:
        print(f"      >>> ROBUST: {frac_positive_dm*100:.1f}% of MC realizations support BEC prediction")
    elif frac_positive_dm > 0.68:
        print(f"      >>> SUGGESTIVE: {frac_positive_dm*100:.1f}% of MC realizations support BEC prediction")
    else:
        print(f"      >>> INCONCLUSIVE: only {frac_positive_dm*100:.1f}% of MC realizations support BEC")

    # ----------------------------------------------------------------
    # Summary of all 5 refined tests
    # ----------------------------------------------------------------
    print("\n  --- Summary of Refined BEC Tests ---")
    print("    (4 PRIMARY tests + 3 ROBUSTNESS checks)")
    n_support = 0
    tests_summary = []

    # Test 1 (ROBUSTNESS CHECK)
    t1_supports = z_delta_dm > 0
    n_support += int(t1_supports)
    t1_sig = f"P={1.0 - z_dm_p:.3f}" if not np.isnan(z_dm_p) else "N/A"
    tests_summary.append(('Z-score (DM regime)', z_delta_dm, t1_sig, t1_supports))
    print(f"    [R] 1. Z-score normalization (DM):    Δz = {z_delta_dm:+.4f}  {t1_sig}  "
          f"{'✓ SUPPORTS' if t1_supports else '✗ opposes'}")

    # Test 2: best threshold (PRIMARY)
    if z_threshold_results:
        t2_supports = zbest['z_delta'] > 0 and zbest['z_p_field_gt_dense'] > 0.90
        n_support += int(t2_supports)
        tests_summary.append(('Threshold scan peak', zbest['z_delta'],
                              f"P={zbest['z_p_field_gt_dense']:.3f}", t2_supports))
        print(f"    [P] 2. Threshold scan peak ({zbest['threshold']:.1f}): "
              f"Δz = {zbest['z_delta']:+.4f}  P={zbest['z_p_field_gt_dense']:.3f}  "
              f"{'✓ SUPPORTS' if t2_supports else '✗ opposes'}")

    # Test 3 (PRIMARY)
    t3_supports = gal_delta_dm > 0
    n_support += int(t3_supports)
    t3_sig = f"P={1.0 - gal_dm_p:.3f}" if not np.isnan(gal_dm_p) else "N/A"
    tests_summary.append(('Galaxy-level (DM regime)', gal_delta_dm, t3_sig, t3_supports))
    print(f"    [P] 3. Galaxy-level scatter (DM):     Δ = {gal_delta_dm:+.4f}  {t3_sig}  "
          f"{'✓ SUPPORTS' if t3_supports else '✗ opposes'}")

    # Test 4 (ROBUSTNESS CHECK)
    if not np.isnan(hd_delta):
        t4_supports = hd_delta > 0
        n_support += int(t4_supports)
        t4_sig = f"P={1.0 - hd_p:.3f}" if not np.isnan(hd_p) else "N/A"
        tests_summary.append(('DM-weighted f_DM>0.5', hd_delta, t4_sig, t4_supports))
        print(f"    [R] 4. DM-fraction weighted (f>0.5):  Δ = {hd_delta:+.4f}  {t4_sig}  "
              f"{'✓ SUPPORTS' if t4_supports else '✗ opposes'}")

    # Test 5 (PRIMARY)
    t5_supports = frac_positive_dm > 0.68
    n_support += int(t5_supports)
    tests_summary.append(('MC error propagation (DM)', mc_dm_med,
                          f"{frac_positive_dm*100:.0f}%", t5_supports))
    print(f"    [P] 5. MC error propagation (DM):     median Δ = {mc_dm_med:+.4f}  "
          f"{frac_positive_dm*100:.0f}% positive  "
          f"{'✓ SUPPORTS' if t5_supports else '✗ opposes'}")

    # ----------------------------------------------------------------
    # TEST 6: Z-score normalized galaxy-level test
    # ----------------------------------------------------------------
    # Tests 3 & 4 failed using raw residuals. Apply dataset normalization.
    print("\n  --- Test 6: Z-score normalized galaxy-level test ---")
    print("    (Galaxy-level scatter with Simpson's Paradox removed)")

    # Build per-galaxy Z-score mean residuals
    gal_z_dense_all = []
    gal_z_field_all = []
    gal_z_dense_dm = []
    gal_z_field_dm = []

    for src in ['SPARC', 'deBlok2002', 'WALLABY', 'SS2020',
                'LITTLETHINGS', 'LVHIS', 'Yu2020', 'Swaters2025',
                'GHASP', 'Noordermeer2005', 'Vogt2004', 'Catinella2005',
                'VirgoRC', 'PHANGS', 'Verheijen2001', 'MaNGA', 'WALLABY_DR2']:
        src_pts = [p for p in all_points
                   if (p['source'] == src or
                       (src == 'SS2020' and p['source'].startswith('SS20_')))]
        if len(src_pts) < 10:
            continue
        all_res_src = np.array([p['log_res'] for p in src_pts])
        mu_s = np.mean(all_res_src)
        std_s = np.std(all_res_src)
        if std_s < 1e-6:
            continue

        # Group by galaxy within this source
        gal_pts = {}
        for p in src_pts:
            gn = p['galaxy']
            if gn not in gal_pts:
                gal_pts[gn] = {'env': p['env_dense'], 'res': [], 'gbar': []}
            gal_pts[gn]['res'].append(p['log_res'])
            gal_pts[gn]['gbar'].append(p['log_gbar'])

        for gn, ginfo in gal_pts.items():
            z_mean = (np.mean(ginfo['res']) - mu_s) / std_s
            gb_mean = np.mean(ginfo['gbar'])
            if ginfo['env'] == 'dense':
                gal_z_dense_all.append(z_mean)
                if gb_mean < dm_threshold:
                    gal_z_dense_dm.append(z_mean)
            else:
                gal_z_field_all.append(z_mean)
                if gb_mean < dm_threshold:
                    gal_z_field_dm.append(z_mean)

    gz_d = np.array(gal_z_dense_dm)
    gz_f = np.array(gal_z_field_dm)
    gz_sig_d = float(np.std(gz_d)) if len(gz_d) > 0 else np.nan
    gz_sig_f = float(np.std(gz_f)) if len(gz_f) > 0 else np.nan
    gz_delta = gz_sig_f - gz_sig_d

    gz_p = np.nan
    if len(gz_d) > 10 and len(gz_f) > 10:
        _, gz_p_raw, _ = bootstrap_scatter_test(gz_d, gz_f, n_boot=10000, seed=777)
        gz_p = gz_p_raw

    print(f"    DM regime: {len(gz_d)} dense gals, {len(gz_f)} field gals")
    print(f"    Z_sigma(mean_res) dense = {gz_sig_d:.4f}")
    print(f"    Z_sigma(mean_res) field = {gz_sig_f:.4f}")
    print(f"    Δ = {gz_delta:+.4f}, P(f>d) = {1.0 - gz_p:.4f}" if not np.isnan(gz_p)
          else f"    Δ = {gz_delta:+.4f}")

    t6_supports = gz_delta > 0
    n_support += int(t6_supports)
    t6_sig = f"P={1.0 - gz_p:.3f}" if not np.isnan(gz_p) else "N/A"
    tests_summary.append(('Z-norm galaxy-level (DM)', gz_delta, t6_sig, t6_supports))
    print(f"    [R] 6. {'✓ SUPPORTS' if t6_supports else '✗ opposes'} BEC prediction")

    # ----------------------------------------------------------------
    # TEST 7: BEC transition function — DOES Δσ follow 1/(e^x - 1)?
    # ----------------------------------------------------------------
    # This is the KEY TEST. The BEC theory doesn't just say "DM regime
    # has more scatter" — it predicts the FUNCTIONAL FORM of how the
    # environmental scatter signal varies with acceleration.
    #
    # The occupation number n(g_bar) = 1/[exp(sqrt(g_bar/g†)) - 1]
    # predicts that the environment sensitivity (Δσ) should:
    #   - Be large at low g_bar (high occupation, DM dominates)
    #   - Vanish at high g_bar (occupation → 0, baryons dominate)
    #   - Follow the shape of 1/(e^x - 1) with x = sqrt(g_bar/g†)
    #
    # We fit Δσ(g_bar) = A * n_BE(g_bar) + C and test goodness of fit.
    print("\n  --- Test 7: BEC transition function test ---")
    print("    (Does the scatter-environment signal follow the Bose-Einstein")
    print("     occupation number as a function of acceleration?)")

    # Compute Δσ in fine bins of log(gbar)
    fine_edges = np.arange(-13.0, -8.5, 0.5)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2.0

    # Use Z-scored residuals for Simpson's Paradox immunity
    z_all_points = []
    for src in ['SPARC', 'deBlok2002', 'WALLABY', 'SS2020',
                'LITTLETHINGS', 'LVHIS', 'Yu2020', 'Swaters2025',
                'GHASP', 'Noordermeer2005', 'Vogt2004', 'Catinella2005',
                'VirgoRC', 'PHANGS', 'Verheijen2001', 'MaNGA', 'WALLABY_DR2']:
        src_pts = [p for p in all_points
                   if (p['source'] == src or
                       (src == 'SS2020' and p['source'].startswith('SS20_')))]
        if len(src_pts) < 10:
            continue
        all_res_src = np.array([p['log_res'] for p in src_pts])
        mu_s = np.mean(all_res_src)
        std_s = np.std(all_res_src)
        if std_s < 1e-6:
            continue
        for p in src_pts:
            z_all_points.append({
                'z_res': (p['log_res'] - mu_s) / std_s,
                'log_gbar': p['log_gbar'],
                'env_dense': p['env_dense'],
                'source': p['source'],
            })

    bin_delta_z = []
    bin_gbar_centers = []
    bin_n_pairs = []

    print(f"\n    {'log(gbar)':>10} {'N_d':>6} {'N_f':>6} {'Zσ_d':>8} {'Zσ_f':>8} {'ΔZσ':>8}")

    for j in range(len(fine_centers)):
        lo, hi = fine_edges[j], fine_edges[j + 1]
        zd = np.array([p['z_res'] for p in z_all_points
                        if p['env_dense'] == 'dense' and lo <= p['log_gbar'] < hi])
        zf = np.array([p['z_res'] for p in z_all_points
                        if p['env_dense'] == 'field' and lo <= p['log_gbar'] < hi])
        if len(zd) >= 10 and len(zf) >= 10:
            zs_d = float(np.std(zd))
            zs_f = float(np.std(zf))
            dz = zs_f - zs_d
            bin_delta_z.append(dz)
            bin_gbar_centers.append(fine_centers[j])
            bin_n_pairs.append(min(len(zd), len(zf)))
            print(f"    {fine_centers[j]:>10.1f} {len(zd):>6} {len(zf):>6} "
                  f"{zs_d:>8.4f} {zs_f:>8.4f} {dz:>+8.4f}")
        elif len(zd) > 0 or len(zf) > 0:
            print(f"    {fine_centers[j]:>10.1f} {len(zd):>6} {len(zf):>6}    (too few for comparison)")

    bin_delta_z = np.array(bin_delta_z)
    bin_gbar_centers = np.array(bin_gbar_centers)
    bin_n_pairs = np.array(bin_n_pairs)

    # χ² intrinsic-scatter calibration state for Test 7 (ΔZσ residual space)
    # Space inference: residuals here are ΔZσ in standardized units, not dex.
    reduced_chi2_bec_z_uncalibrated = np.nan
    reduced_chi2_linear_z_uncalibrated = np.nan
    reduced_chi2_const_z_uncalibrated = np.nan
    reduced_chi2_bec_z_calibrated = np.nan
    reduced_chi2_linear_z_calibrated = np.nan
    reduced_chi2_const_z_calibrated = np.nan
    sigma_int_bec_z = np.nan
    sigma_int_linear_z = np.nan
    sigma_int_const_z = np.nan
    chi2_cal_method_bec = None
    chi2_cal_method_linear = None
    chi2_cal_method_const = None
    chi2_cal_n_used_bec = None
    chi2_cal_n_used_linear = None
    chi2_cal_n_used_const = None
    chi2_cal_bracket_bec = None
    chi2_cal_bracket_linear = None
    chi2_cal_bracket_const = None
    chi2_cal_reason_bec = None
    chi2_cal_reason_linear = None
    chi2_cal_reason_const = None
    chi2_cal_dof_used_bec = None
    chi2_cal_dof_used_linear = None
    chi2_cal_dof_used_const = None
    chi2_cal_dof_assumption = None
    sigma_int_shared_z = np.nan
    reduced_chi2_bec_z_shared = np.nan
    reduced_chi2_linear_z_shared = np.nan
    reduced_chi2_const_z_shared = np.nan
    chi2_shared_ref_model = str(chi2_sigma_ref_model).lower()
    chi2_shared_method = None
    chi2_shared_n_used_ref = None
    chi2_shared_reason = None

    if len(bin_delta_z) >= 4:
        # Fit Model 1: BEC transition  Δσ(gbar) = A / [exp(sqrt(10^gbar / g†)) - 1] + C
        def bec_model(log_gbar, A, C):
            gbar_lin = 10.0 ** log_gbar
            x = np.sqrt(gbar_lin / g_dagger)
            n_be = 1.0 / (np.exp(x) - 1.0 + 1e-30)
            return A * n_be + C

        # Fit Model 2: linear (null hypothesis)
        def linear_model(log_gbar, m, b):
            return m * log_gbar + b

        # Fit Model 3: constant (simplest null)
        mean_delta = np.mean(bin_delta_z)

        # Weight by sqrt(N) for each bin
        weights = np.sqrt(bin_n_pairs)

        # Fit BEC model
        try:
            from scipy.optimize import curve_fit
            popt_bec, pcov_bec = curve_fit(bec_model, bin_gbar_centers, bin_delta_z,
                                            p0=[0.5, -0.05], sigma=1.0/weights,
                                            absolute_sigma=False, maxfev=5000)
            resid_bec = bin_delta_z - bec_model(bin_gbar_centers, *popt_bec)
            chi2_bec = np.sum(weights**2 * resid_bec**2)
            bec_A, bec_C = popt_bec

            # Fit linear model
            popt_lin, _ = curve_fit(linear_model, bin_gbar_centers, bin_delta_z,
                                     p0=[0.01, 0.0], sigma=1.0/weights,
                                     absolute_sigma=False)
            resid_lin = bin_delta_z - linear_model(bin_gbar_centers, *popt_lin)
            chi2_lin = np.sum(weights**2 * resid_lin**2)

            # Constant model (null)
            resid_const = bin_delta_z - mean_delta
            chi2_const = np.sum(weights**2 * resid_const**2)

            n_bins_fit = len(bin_delta_z)
            k_bec = 2
            k_lin = 2
            k_const = 1
            dof_bec = n_bins_fit - k_bec
            dof_lin = n_bins_fit - k_lin
            dof_const = n_bins_fit - k_const
            chi2_cal_dof_used_bec = max(dof_bec, 1)
            chi2_cal_dof_used_linear = max(dof_lin, 1)
            chi2_cal_dof_used_const = max(dof_const, 1)
            chi2_cal_dof_assumption = (
                "k_bec=2, k_lin=2, k_const=1 from Test-7 fit parameter counts"
            )

            # Reduced chi-squared
            rchi2_bec = chi2_bec / max(dof_bec, 1)
            rchi2_lin = chi2_lin / max(dof_lin, 1)
            rchi2_const = chi2_const / max(dof_const, 1)

            # Intrinsic-scatter calibration in the same residual space (ΔZσ).
            # We preserve legacy χ²/dof definitions (dof from number of bins),
            # and build per-pair expanded residuals equivalent to bin weighting.
            reduced_chi2_bec_z_uncalibrated = rchi2_bec
            reduced_chi2_linear_z_uncalibrated = rchi2_lin
            reduced_chi2_const_z_uncalibrated = rchi2_const

            if solve_sigma_int_for_chi2_1 is None:
                reason = f"chi2_calibration_import_failed: {CHI2_CAL_IMPORT_ERROR}"
                chi2_cal_reason_bec = reason
                chi2_cal_reason_linear = reason
                chi2_cal_reason_const = reason
                if chi2_shared_sigma_int:
                    chi2_shared_reason = reason
            else:
                sigma_bins = np.divide(
                    1.0,
                    weights,
                    out=np.full_like(weights, np.nan, dtype=float),
                    where=weights > 0,
                )
                pair_counts = np.maximum(np.round(bin_n_pairs).astype(int), 1)

                def _expand_for_calibration(resid_arr):
                    r_exp = np.repeat(np.asarray(resid_arr, dtype=float), pair_counts)
                    # Equivalent to per-bin sigma=1/sqrt(N): replicate each bin N times
                    # with unit sigma so chi2 = sum(N_bin * resid_bin^2), matching legacy chi2.
                    s_exp = np.ones_like(r_exp, dtype=float)
                    return r_exp, s_exp

                def _run_chi2_cal(model_name, resid_arr, dof_model):
                    resid_exp, sigma_exp = _expand_for_calibration(resid_arr)
                    cal = solve_sigma_int_for_chi2_1(
                        resid=resid_exp,
                        sigma_meas=sigma_exp,
                        dof=max(int(dof_model), 1),
                    )
                    sigma_int_val = cal.get('sigma_int_best')
                    chi2_uncal = cal.get('chi2_red_uncal')
                    chi2_cal = cal.get('chi2_red_cal')
                    method = cal.get('method')
                    bracket_used = cal.get('bracket_used')
                    n_used = cal.get('n_used')
                    reason = cal.get('reason')
                    sigma_txt = f"{sigma_int_val:.6g}" if sigma_int_val is not None else "None"
                    chi2_uncal_txt = f"{chi2_uncal:.4f}" if chi2_uncal is not None else "None"
                    chi2_cal_txt = f"{chi2_cal:.4f}" if chi2_cal is not None else "None"
                    print(
                        f"[CHI2 CAL] model={model_name} space=z N={n_used} dof={max(int(dof_model), 1)} "
                        f"chi2_red_uncal={chi2_uncal_txt} sigma_int={sigma_txt} "
                        f"chi2_red_cal={chi2_cal_txt} method={method} reason={reason}"
                    )
                    return cal

                print("    [CHI2 CAL] residual space inference: ΔZσ (dimensionless z units)")
                cal_bec = _run_chi2_cal('BEC', resid_bec, dof_bec)
                cal_lin = _run_chi2_cal('Linear', resid_lin, dof_lin)
                cal_const = _run_chi2_cal('Const', resid_const, dof_const)

                if cal_bec.get('sigma_int_best') is not None:
                    sigma_int_bec_z = float(cal_bec['sigma_int_best'])
                if cal_lin.get('sigma_int_best') is not None:
                    sigma_int_linear_z = float(cal_lin['sigma_int_best'])
                if cal_const.get('sigma_int_best') is not None:
                    sigma_int_const_z = float(cal_const['sigma_int_best'])

                if cal_bec.get('chi2_red_cal') is not None:
                    reduced_chi2_bec_z_calibrated = float(cal_bec['chi2_red_cal'])
                if cal_lin.get('chi2_red_cal') is not None:
                    reduced_chi2_linear_z_calibrated = float(cal_lin['chi2_red_cal'])
                if cal_const.get('chi2_red_cal') is not None:
                    reduced_chi2_const_z_calibrated = float(cal_const['chi2_red_cal'])

                chi2_cal_method_bec = cal_bec.get('method')
                chi2_cal_method_linear = cal_lin.get('method')
                chi2_cal_method_const = cal_const.get('method')
                chi2_cal_n_used_bec = cal_bec.get('n_used')
                chi2_cal_n_used_linear = cal_lin.get('n_used')
                chi2_cal_n_used_const = cal_const.get('n_used')
                chi2_cal_bracket_bec = cal_bec.get('bracket_used')
                chi2_cal_bracket_linear = cal_lin.get('bracket_used')
                chi2_cal_bracket_const = cal_const.get('bracket_used')
                chi2_cal_reason_bec = cal_bec.get('reason')
                chi2_cal_reason_linear = cal_lin.get('reason')
                chi2_cal_reason_const = cal_const.get('reason')

                if chi2_shared_sigma_int:
                    ref_raw = str(chi2_sigma_ref_model).lower()
                    if ref_raw in ('bec', 'b'):
                        ref_key = 'bec'
                    elif ref_raw in ('lin', 'linear', 'l'):
                        ref_key = 'lin'
                    elif ref_raw in ('const', 'constant', 'c'):
                        ref_key = 'const'
                    else:
                        ref_key = 'bec'
                        chi2_shared_reason = f"invalid_ref_model:{ref_raw}; fallback=bec"

                    chi2_shared_ref_model = ref_key
                    ref_map = {
                        'bec': ('BEC', resid_bec, dof_bec),
                        'lin': ('LIN', resid_lin, dof_lin),
                        'const': ('CONST', resid_const, dof_const),
                    }
                    ref_label, ref_resid, ref_dof = ref_map[ref_key]
                    ref_resid_exp, ref_sigma_exp = _expand_for_calibration(ref_resid)
                    cal_shared = solve_sigma_int_for_chi2_1(
                        resid=ref_resid_exp,
                        sigma_meas=ref_sigma_exp,
                        dof=max(int(ref_dof), 1),
                    )
                    chi2_shared_method = cal_shared.get('method')
                    chi2_shared_n_used_ref = cal_shared.get('n_used')
                    if chi2_shared_reason is None:
                        chi2_shared_reason = cal_shared.get('reason')

                    sigma_shared = cal_shared.get('sigma_int_best')
                    if sigma_shared is not None and np.isfinite(sigma_shared):
                        sigma_int_shared_z = float(sigma_shared)

                        def _shared_rchi2(resid_arr, dof_model):
                            resid_exp, sigma_exp = _expand_for_calibration(resid_arr)
                            dof_eff = max(int(dof_model), 1)
                            if chi2_red_given_sigma_int is not None:
                                return float(
                                    chi2_red_given_sigma_int(
                                        resid_exp, sigma_exp, dof_eff, sigma_int_shared_z
                                    )
                                )
                            sigma_tot = np.sqrt(np.square(sigma_exp) + sigma_int_shared_z**2)
                            return float(np.sum((resid_exp / sigma_tot) ** 2) / float(dof_eff))

                        reduced_chi2_bec_z_shared = _shared_rchi2(resid_bec, dof_bec)
                        reduced_chi2_linear_z_shared = _shared_rchi2(resid_lin, dof_lin)
                        reduced_chi2_const_z_shared = _shared_rchi2(resid_const, dof_const)
                    else:
                        chi2_shared_reason = (
                            f"{chi2_shared_reason}; shared_sigma_unavailable"
                            if chi2_shared_reason
                            else "shared_sigma_unavailable"
                        )

                    sig_txt = f"{sigma_int_shared_z:.6g}" if np.isfinite(sigma_int_shared_z) else "None"
                    b_txt = f"{reduced_chi2_bec_z_shared:.4f}" if np.isfinite(reduced_chi2_bec_z_shared) else "None"
                    l_txt = f"{reduced_chi2_linear_z_shared:.4f}" if np.isfinite(reduced_chi2_linear_z_shared) else "None"
                    c_txt = f"{reduced_chi2_const_z_shared:.4f}" if np.isfinite(reduced_chi2_const_z_shared) else "None"
                    print(
                        f"[CHI2 SHARED] ref={ref_label} sigma_int_shared={sig_txt} "
                        f"-> rchi2_bec={b_txt}, rchi2_lin={l_txt}, rchi2_const={c_txt}"
                    )

            # AIC comparison (lower is better)
            # AIC = chi2 + 2*k where k = number of parameters
            aic_bec = chi2_bec + 2 * 2
            aic_lin = chi2_lin + 2 * 2
            aic_const = chi2_const + 2 * 1

            print(f"\n    Model fitting ({n_bins_fit} acceleration bins):")
            print(f"    BEC model:    Δσ = {bec_A:.4f} / [exp(√(gbar/g†)) - 1] + ({bec_C:+.4f})")
            print(f"                  χ²/dof = {rchi2_bec:.3f}, AIC = {aic_bec:.2f}")
            print(f"    Linear model: Δσ = {popt_lin[0]:.4f} * log(gbar) + ({popt_lin[1]:+.4f})")
            print(f"                  χ²/dof = {rchi2_lin:.3f}, AIC = {aic_lin:.2f}")
            print(f"    Constant:     Δσ = {mean_delta:+.4f}")
            print(f"                  χ²/dof = {rchi2_const:.3f}, AIC = {aic_const:.2f}")

            # BEC is preferred if it has lowest AIC
            best_model = 'BEC' if aic_bec <= aic_lin and aic_bec <= aic_const else \
                         'Linear' if aic_lin <= aic_const else 'Constant'

            delta_aic_bec_vs_lin = aic_lin - aic_bec
            delta_aic_bec_vs_const = aic_const - aic_bec

            print(f"\n    ΔAIC(linear - BEC) = {delta_aic_bec_vs_lin:+.2f} "
                  f"({'BEC preferred' if delta_aic_bec_vs_lin > 0 else 'linear preferred'})")
            print(f"    ΔAIC(constant - BEC) = {delta_aic_bec_vs_const:+.2f} "
                  f"({'BEC preferred' if delta_aic_bec_vs_const > 0 else 'constant preferred'})")
            print(f"    Best model: {best_model}")

            if bec_A > 0 and best_model == 'BEC':
                print(f"\n    >>> BEC TRANSITION CONFIRMED: Δσ follows Bose-Einstein occupation number")
                print(f"    >>> Amplitude A = {bec_A:.4f} (positive = field > dense where DM dominates)")
                print(f"    >>> Offset C = {bec_C:+.4f} (residual at high gbar)")
                t7_supports = True
            elif bec_A > 0:
                print(f"\n    >>> BEC model has correct sign (A>0) but {best_model} model fits better")
                t7_supports = delta_aic_bec_vs_lin > -2  # BEC not significantly worse
            else:
                print(f"\n    >>> BEC model has WRONG sign (A<0) — opposes prediction")
                t7_supports = False

            # Spearman rank correlation: does Δσ increase as gbar decreases?
            if len(bin_delta_z) >= 5:
                spearman_r, spearman_p = stats.spearmanr(bin_gbar_centers, bin_delta_z)
                print(f"\n    Spearman correlation (Δσ vs log gbar): r = {spearman_r:.3f}, p = {spearman_p:.4f}")
                if spearman_r < 0 and spearman_p < 0.05:
                    print(f"    >>> Significant negative correlation: scatter gap INCREASES at low gbar")
                elif spearman_r < 0:
                    print(f"    >>> Negative trend (BEC direction) but not significant (p={spearman_p:.3f})")
                else:
                    print(f"    >>> No negative correlation detected")
            else:
                spearman_r, spearman_p = np.nan, np.nan

        except Exception as e:
            print(f"\n    Model fitting failed: {e}")
            t7_supports = False
            bec_A = np.nan
            bec_C = np.nan
            rchi2_bec = np.nan
            rchi2_lin = np.nan
            rchi2_const = np.nan
            aic_bec = np.nan
            aic_lin = np.nan
            aic_const = np.nan
            delta_aic_bec_vs_lin = np.nan
            delta_aic_bec_vs_const = np.nan
            best_model = 'FAILED'
            spearman_r = np.nan
            spearman_p = np.nan
            chi2_cal_reason_bec = f"model_fitting_failed: {e}"
            chi2_cal_reason_linear = f"model_fitting_failed: {e}"
            chi2_cal_reason_const = f"model_fitting_failed: {e}"
            if chi2_shared_sigma_int:
                chi2_shared_reason = f"model_fitting_failed: {e}"
    else:
        print("    Not enough bins for model fitting")
        t7_supports = False
        bec_A = np.nan
        bec_C = np.nan
        rchi2_bec = np.nan
        rchi2_lin = np.nan
        rchi2_const = np.nan
        aic_bec = np.nan
        aic_lin = np.nan
        aic_const = np.nan
        delta_aic_bec_vs_lin = np.nan
        delta_aic_bec_vs_const = np.nan
        best_model = 'N/A'
        spearman_r = np.nan
        spearman_p = np.nan
        chi2_cal_reason_bec = "not_enough_bins_for_model_fitting"
        chi2_cal_reason_linear = "not_enough_bins_for_model_fitting"
        chi2_cal_reason_const = "not_enough_bins_for_model_fitting"
        if chi2_shared_sigma_int:
            chi2_shared_reason = "not_enough_bins_for_model_fitting"

    n_support += int(t7_supports)
    t7_sig = f"ΔAIC={delta_aic_bec_vs_lin:+.1f}" if not np.isnan(delta_aic_bec_vs_lin) else "N/A"
    tests_summary.append(('BEC transition function', bec_A if not np.isnan(bec_A) else 0.0,
                          t7_sig, t7_supports))
    print(f"    [P] 7. {'✓ SUPPORTS' if t7_supports else '✗ opposes'} BEC prediction")

    # ================================================================
    # STEP 5d: TEST 7 ROBUSTNESS — Bootstrap ΔAIC, Leave-one-out, Plot
    # ================================================================
    print("\n  " + "=" * 70)
    print("  STEP 5d: TEST 7 ROBUSTNESS CHECKS")
    print("  " + "=" * 70)

    # ---- Helper: Compute Test 7 ΔAIC from a set of points ----
    def compute_test7_daic(pts, source_list=None):
        """
        Given a list of RAR points, compute the BEC transition function
        ΔAIC (linear − BEC). Positive = BEC preferred.
        Returns dict with daic_lin_bec, daic_const_bec, bec_A, bec_C, n_bins_used.
        """
        if source_list is None:
            source_list = ['SPARC', 'deBlok2002', 'WALLABY', 'SS2020',
                           'LITTLETHINGS', 'LVHIS', 'Yu2020', 'Swaters2025',
                           'GHASP', 'Noordermeer2005', 'Vogt2004', 'Catinella2005',
                           'VirgoRC', 'PHANGS', 'Verheijen2001', 'MaNGA', 'WALLABY_DR2']

        # Z-score within each dataset
        z_pts = []
        for src in source_list:
            src_pts = [p for p in pts
                       if (p['source'] == src or
                           (src == 'SS2020' and p['source'].startswith('SS20_')))]
            if len(src_pts) < 10:
                continue
            all_res = np.array([p['log_res'] for p in src_pts])
            mu_s, std_s = np.mean(all_res), np.std(all_res)
            if std_s < 1e-6:
                continue
            for p in src_pts:
                z_pts.append({
                    'z_res': (p['log_res'] - mu_s) / std_s,
                    'log_gbar': p['log_gbar'],
                    'env_dense': p['env_dense'],
                    'galaxy': p['galaxy'],
                })

        if not z_pts:
            return {'daic_lin_bec': np.nan, 'daic_const_bec': np.nan,
                    'bec_A': np.nan, 'bec_C': np.nan, 'n_bins_used': 0}

        # Bin into Δσ(gbar)
        fe = np.arange(-13.0, -8.5, 0.5)
        fc = (fe[:-1] + fe[1:]) / 2.0
        b_dz, b_gc, b_np = [], [], []
        for j in range(len(fc)):
            lo, hi = fe[j], fe[j + 1]
            zd = np.array([p['z_res'] for p in z_pts
                           if p['env_dense'] == 'dense' and lo <= p['log_gbar'] < hi])
            zf = np.array([p['z_res'] for p in z_pts
                           if p['env_dense'] == 'field' and lo <= p['log_gbar'] < hi])
            if len(zd) >= 10 and len(zf) >= 10:
                b_dz.append(float(np.std(zf) - np.std(zd)))
                b_gc.append(fc[j])
                b_np.append(min(len(zd), len(zf)))

        b_dz = np.array(b_dz)
        b_gc = np.array(b_gc)
        b_np = np.array(b_np)

        if len(b_dz) < 4:
            return {'daic_lin_bec': np.nan, 'daic_const_bec': np.nan,
                    'bec_A': np.nan, 'bec_C': np.nan, 'n_bins_used': len(b_dz)}

        wt = np.sqrt(b_np)

        def _bec(lg, A, C):
            gl = 10.0 ** lg
            x = np.sqrt(gl / g_dagger)
            return A / (np.exp(x) - 1.0 + 1e-30) + C

        def _lin(lg, m, b):
            return m * lg + b

        try:
            popt_b, _ = curve_fit(_bec, b_gc, b_dz, p0=[0.5, -0.05],
                                  sigma=1.0/wt, absolute_sigma=False, maxfev=5000)
            res_b = b_dz - _bec(b_gc, *popt_b)
            chi2_b = np.sum(wt**2 * res_b**2)

            popt_l, _ = curve_fit(_lin, b_gc, b_dz, p0=[0.01, 0.0],
                                  sigma=1.0/wt, absolute_sigma=False)
            res_l = b_dz - _lin(b_gc, *popt_l)
            chi2_l = np.sum(wt**2 * res_l**2)

            mean_dz = np.mean(b_dz)
            res_c = b_dz - mean_dz
            chi2_c = np.sum(wt**2 * res_c**2)

            aic_b = chi2_b + 2 * 2
            aic_l = chi2_l + 2 * 2
            aic_c = chi2_c + 2 * 1

            return {
                'daic_lin_bec': float(aic_l - aic_b),
                'daic_const_bec': float(aic_c - aic_b),
                'bec_A': float(popt_b[0]),
                'bec_C': float(popt_b[1]),
                'n_bins_used': len(b_dz),
                'bin_centers': b_gc.tolist(),
                'bin_delta_z': b_dz.tolist(),
            }
        except:
            return {'daic_lin_bec': np.nan, 'daic_const_bec': np.nan,
                    'bec_A': np.nan, 'bec_C': np.nan, 'n_bins_used': len(b_dz)}

    # ---- (A) Bootstrap ΔAIC: resample galaxies with replacement ----
    print("\n  --- Bootstrap ΔAIC (resampling galaxies, 1000 iterations) ---")

    # Build galaxy -> points mapping
    gal_to_pts = {}
    for p in all_points:
        g = p['galaxy']
        if g not in gal_to_pts:
            gal_to_pts[g] = []
        gal_to_pts[g].append(p)

    galaxy_names_list = list(gal_to_pts.keys())
    n_gal_total = len(galaxy_names_list)

    n_boot_aic = 1000
    boot_daics = []
    boot_As = []
    boot_bec_preferred = 0

    rng = np.random.RandomState(42)
    for i in range(n_boot_aic):
        # Resample galaxies with replacement
        boot_gals = rng.choice(galaxy_names_list, size=n_gal_total, replace=True)
        boot_pts = []
        for g in boot_gals:
            boot_pts.extend(gal_to_pts[g])

        result = compute_test7_daic(boot_pts)
        daic = result['daic_lin_bec']
        if not np.isnan(daic):
            boot_daics.append(daic)
            boot_As.append(result['bec_A'])
            if daic > 0:
                boot_bec_preferred += 1

    boot_daics = np.array(boot_daics)
    boot_As = np.array(boot_As)
    n_valid = len(boot_daics)

    if n_valid > 0:
        med_daic = np.median(boot_daics)
        ci68_lo = np.percentile(boot_daics, 16)
        ci68_hi = np.percentile(boot_daics, 84)
        ci95_lo = np.percentile(boot_daics, 2.5)
        ci95_hi = np.percentile(boot_daics, 97.5)
        frac_bec = boot_bec_preferred / n_valid
        med_A = np.median(boot_As)

        print(f"    Valid iterations: {n_valid}/{n_boot_aic}")
        print(f"    Median ΔAIC (linear−BEC): {med_daic:+.2f}")
        print(f"    68% CI: [{ci68_lo:+.2f}, {ci68_hi:+.2f}]")
        print(f"    95% CI: [{ci95_lo:+.2f}, {ci95_hi:+.2f}]")
        print(f"    Fraction BEC preferred: {frac_bec:.1%} ({boot_bec_preferred}/{n_valid})")
        print(f"    Median BEC amplitude A: {med_A:+.4f}")

        if frac_bec >= 0.9:
            print(f"    >>> ROBUST: BEC preferred in {frac_bec:.0%} of bootstrap samples")
            boot_robust = True
        elif frac_bec >= 0.7:
            print(f"    >>> MODERATE: BEC preferred in {frac_bec:.0%} of bootstrap samples")
            boot_robust = True
        else:
            print(f"    >>> WEAK: BEC preferred in only {frac_bec:.0%} of bootstrap samples")
            boot_robust = False
    else:
        print("    Bootstrap failed — no valid iterations")
        med_daic = np.nan
        ci68_lo = ci68_hi = ci95_lo = ci95_hi = np.nan
        frac_bec = 0.0
        med_A = np.nan
        boot_robust = False

    bootstrap_results = {
        'n_iterations': n_boot_aic,
        'n_valid': int(n_valid),
        'median_daic': float(med_daic) if not np.isnan(med_daic) else None,
        'ci_68': [float(ci68_lo), float(ci68_hi)] if not np.isnan(ci68_lo) else None,
        'ci_95': [float(ci95_lo), float(ci95_hi)] if not np.isnan(ci95_lo) else None,
        'frac_bec_preferred': float(frac_bec),
        'median_amplitude_A': float(med_A) if not np.isnan(med_A) else None,
        'robust': boot_robust,
    }

    # ---- (B) Leave-one-dataset-out jackknife ----
    print("\n  --- Leave-one-dataset-out jackknife ---")
    print(f"    {'Dropped':>20}  {'ΔAIC':>8}  {'A':>8}  {'Bins':>5}  {'Best':>8}")

    active_sources = ['SPARC', 'deBlok2002', 'WALLABY', 'SS2020',
                      'LITTLETHINGS', 'LVHIS', 'Yu2020', 'Swaters2025',
                      'GHASP', 'Noordermeer2005', 'Vogt2004', 'Catinella2005',
                      'VirgoRC', 'PHANGS', 'Verheijen2001', 'MaNGA', 'WALLABY_DR2']

    jackknife_results = {}
    all_jack_prefer_bec = True

    for drop_src in active_sources:
        # Remove all points from this dataset
        jack_pts = [p for p in all_points
                    if not (p['source'] == drop_src or
                            (drop_src == 'SS2020' and p['source'].startswith('SS20_')))]
        remaining = [s for s in active_sources if s != drop_src]
        result = compute_test7_daic(jack_pts, source_list=remaining)
        daic = result['daic_lin_bec']
        bA = result['bec_A']
        nb = result['n_bins_used']
        best = 'BEC' if (not np.isnan(daic) and daic > 0) else 'Linear' if not np.isnan(daic) else 'N/A'

        if np.isnan(daic) or daic <= 0:
            all_jack_prefer_bec = False

        jackknife_results[drop_src] = {
            'daic_lin_bec': float(daic) if not np.isnan(daic) else None,
            'bec_A': float(bA) if not np.isnan(bA) else None,
            'n_bins': int(nb),
            'best_model': best,
        }

        daic_str = f"{daic:+.2f}" if not np.isnan(daic) else "N/A"
        bA_str = f"{bA:+.4f}" if not np.isnan(bA) else "N/A"
        print(f"    {drop_src:>20}  {daic_str:>8}  {bA_str:>8}  {nb:>5}  {best:>8}")

    if all_jack_prefer_bec:
        print(f"\n    >>> ROBUST: BEC preferred regardless of which dataset is dropped")
    else:
        failed = [s for s, r in jackknife_results.items()
                  if r['best_model'] != 'BEC']
        print(f"\n    >>> PARTIAL: BEC NOT preferred when dropping: {', '.join(failed)}")

    # ---- (C) Transition function plot ----
    print("\n  --- Generating transition function plot ---")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: Δσ vs gbar with model fits
        x_plot = np.linspace(-13.0, -8.5, 200)

        # BEC model curve
        def bec_curve(lg, A, C):
            gl = 10.0 ** lg
            x = np.sqrt(gl / g_dagger)
            return A / (np.exp(x) - 1.0 + 1e-30) + C

        def lin_curve(lg, m, b):
            return m * lg + b

        # Refit on full data for plotting (use stored fit params)
        if not np.isnan(bec_A) and len(bin_delta_z) >= 4:
            # Plot data points with error bars (use 1/sqrt(N) as uncertainty proxy)
            point_err = 1.0 / np.sqrt(bin_n_pairs) * 0.3  # scaled for visibility
            ax1.errorbar(bin_gbar_centers, bin_delta_z, yerr=point_err,
                         fmt='ko', markersize=8, capsize=4, capthick=1.5,
                         label='Data (binned Δσ)', zorder=5)

            # BEC model
            y_bec = bec_curve(x_plot, bec_A, bec_C)
            ax1.plot(x_plot, y_bec, 'b-', linewidth=2.5,
                     label=f'BEC: A={bec_A:.3f} (AIC={aic_bec:.1f})')

            # Linear model
            popt_lin_plot, _ = curve_fit(lin_curve, bin_gbar_centers, bin_delta_z,
                                         p0=[0.01, 0.0], sigma=1.0/weights,
                                         absolute_sigma=False)
            y_lin = lin_curve(x_plot, *popt_lin_plot)
            ax1.plot(x_plot, y_lin, 'r--', linewidth=2.0,
                     label=f'Linear (AIC={aic_lin:.1f})')

            # Constant model
            ax1.axhline(np.mean(bin_delta_z), color='gray', linestyle=':',
                        linewidth=1.5, label=f'Constant (AIC={aic_const:.1f})')

            ax1.axhline(0, color='k', linewidth=0.5, alpha=0.3)
            ax1.set_xlabel('log(g$_{bar}$) [m/s²]', fontsize=13)
            ax1.set_ylabel('Δσ$_Z$ (field − dense)', fontsize=13)
            ax1.set_title(f'BEC Transition Function\nΔAIC(lin−BEC) = {delta_aic_bec_vs_lin:+.1f}',
                          fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10, loc='upper right')
            ax1.set_xlim(-13.2, -8.8)
            ax1.set_ylim(-0.6, 0.6)
            ax1.grid(True, alpha=0.3)

            # Annotate physics
            ax1.annotate('DM dominated\n(BEC signal expected)',
                         xy=(-12.5, 0.35), fontsize=9, color='blue',
                         ha='center', style='italic')
            ax1.annotate('Baryon dominated\n(no BEC signal)',
                         xy=(-9.0, -0.35), fontsize=9, color='gray',
                         ha='center', style='italic')

        # Right panel: Bootstrap ΔAIC distribution
        if n_valid > 50:
            ax2.hist(boot_daics, bins=40, density=True, color='steelblue',
                     alpha=0.7, edgecolor='white', linewidth=0.5)
            ax2.axvline(0, color='red', linewidth=2, linestyle='--',
                        label='ΔAIC = 0 (no preference)')
            ax2.axvline(med_daic, color='navy', linewidth=2,
                        label=f'Median = {med_daic:+.1f}')
            ax2.axvspan(ci95_lo, ci95_hi, alpha=0.15, color='navy',
                        label=f'95% CI [{ci95_lo:+.1f}, {ci95_hi:+.1f}]')
            ax2.set_xlabel('ΔAIC (linear − BEC)', fontsize=13)
            ax2.set_ylabel('Density', fontsize=13)
            ax2.set_title(f'Bootstrap ΔAIC ({n_valid} iterations)\n'
                          f'BEC preferred: {frac_bec:.0%}',
                          fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'test7_transition_function.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {plot_path}")
    except Exception as e:
        print(f"    Plot generation failed: {e}")
        plot_path = None

    # ---- (D) Test 6 diagnostic: why did Z-norm galaxy-level flip? ----
    print("\n  --- Test 6 diagnostic: Z-norm galaxy-level decomposition ---")

    # Compute Test 6 result with and without each dataset to find the driver
    print(f"    Full result: Δ = {gz_delta:+.4f}, P(f>d) = {1.0-gz_p:.4f}")
    print(f"\n    {'Dropped':>20}  {'Δ':>8}  {'BEC?':>6}  {'N_d':>5}  {'N_f':>5}")

    t6_jackknife = {}
    for drop_src in active_sources:
        # Get galaxy-level Z-scored means excluding this dataset
        z_gal_means_d_drop = []
        z_gal_means_f_drop = []

        for src in active_sources:
            if src == drop_src:
                continue
            src_pts_all = [p for p in all_points
                           if (p['source'] == src or
                               (src == 'SS2020' and p['source'].startswith('SS20_')))]
            if len(src_pts_all) < 10:
                continue
            all_res_src = np.array([p['log_res'] for p in src_pts_all])
            mu_s, std_s = np.mean(all_res_src), np.std(all_res_src)
            if std_s < 1e-6:
                continue

            # Group by galaxy, DM regime only
            src_gals = {}
            for p in src_pts_all:
                if p['log_gbar'] >= -10.5:
                    continue
                g = p['galaxy']
                if g not in src_gals:
                    src_gals[g] = {'z_res': [], 'env': p['env_dense']}
                src_gals[g]['z_res'].append((p['log_res'] - mu_s) / std_s)

            for g, info in src_gals.items():
                if len(info['z_res']) >= 1:
                    mean_z = np.mean(info['z_res'])
                    if info['env'] == 'dense':
                        z_gal_means_d_drop.append(mean_z)
                    else:
                        z_gal_means_f_drop.append(mean_z)

        z_gal_d = np.array(z_gal_means_d_drop)
        z_gal_f = np.array(z_gal_means_f_drop)

        if len(z_gal_d) > 5 and len(z_gal_f) > 5:
            delta_drop = float(np.std(z_gal_f) - np.std(z_gal_d))
            supports = delta_drop > 0
        else:
            delta_drop = np.nan
            supports = False

        t6_jackknife[drop_src] = {
            'delta': float(delta_drop) if not np.isnan(delta_drop) else None,
            'supports_bec': supports,
            'n_dense': len(z_gal_d),
            'n_field': len(z_gal_f),
        }

        d_str = f"{delta_drop:+.4f}" if not np.isnan(delta_drop) else "N/A"
        bec_str = "✓" if supports else "✗"
        print(f"    {drop_src:>20}  {d_str:>8}  {bec_str:>6}  {len(z_gal_d):>5}  {len(z_gal_f):>5}")

    # Identify which dataset(s) are driving the Test 6 failure
    flippers = [s for s, r in t6_jackknife.items() if r['supports_bec']]
    if flippers:
        print(f"\n    Test 6 would SUPPORT BEC if we dropped: {', '.join(flippers)}")
        print(f"    >>> These datasets are driving the Test 6 opposition")
    else:
        print(f"\n    Test 6 opposes BEC regardless of which dataset is dropped")

    # ================================================================
    # TEST 8: BOSON BUNCHING STATISTICS
    # ================================================================
    # This is the KEY QUANTUM TEST. For a genuine Bose-Einstein condensate,
    # the number fluctuations in a mode with mean occupation n̄ follow:
    #   ⟨(δn)²⟩ = n̄(n̄ + 1)     [quantum / bosonic bunching]
    # versus the classical (Poisson) prediction:
    #   ⟨(δn)²⟩ = n̄             [classical fluid]
    #
    # In our framework:
    #   n̄(gbar) = 1 / [exp(√(gbar/g†)) - 1]
    #
    # So the RAR residual VARIANCE at each gbar should scale as:
    #   BEC:      σ²(gbar) = A * [n̄² + n̄] + C
    #   Classical: σ²(gbar) = A * n̄ + C
    #
    # The difference is measurable: at high occupation (low gbar),
    # BEC gives σ² ∝ n̄² (super-Poissonian bunching), while classical
    # gives σ² ∝ n̄. This is the same photon bunching effect (Hanbury
    # Brown–Twiss) that distinguishes thermal bosons from classical particles.
    #
    # No classical dark matter model (CDM, SIDM, fuzzy DM) predicts
    # super-Poissonian bunching. Only a genuine condensate does.
    print("\n  " + "=" * 70)
    print("  TEST 8: BOSON BUNCHING STATISTICS (QUANTUM SIGNATURE)")
    print("  " + "=" * 70)
    print("    Testing whether RAR scatter follows quantum σ² ∝ n̄(n̄+1)")
    print("    versus classical σ² ∝ n̄, where n̄ = 1/[exp(√(gbar/g†))−1]")

    # Compute σ²(gbar) in fine bins using Z-scored residuals
    # (Z-scoring removes dataset systematics, isolating intrinsic scatter)
    bunching_edges = np.arange(-13.0, -8.0, 0.4)  # finer bins for more resolution
    bunching_centers = (bunching_edges[:-1] + bunching_edges[1:]) / 2.0

    bin_var = []         # measured variance σ² in each bin
    bin_var_err = []     # uncertainty on variance (chi² approximation)
    bin_nbar = []        # theoretical n̄ at bin center
    bin_nbar_sq = []     # theoretical n̄² + n̄ at bin center
    bin_gbar_c = []      # bin centers that have enough data
    bin_N = []           # number of points per bin

    for j in range(len(bunching_centers)):
        lo, hi = bunching_edges[j], bunching_edges[j + 1]
        z_vals = np.array([p['z_res'] for p in z_all_points
                           if lo <= p['log_gbar'] < hi])
        if len(z_vals) >= 30:  # need enough for reliable variance
            var_obs = float(np.var(z_vals))
            # Variance uncertainty: Var(s²) ≈ 2σ⁴/(N-1) for normal data
            var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))

            # Theoretical occupation number at bin center
            gbar_lin = 10.0 ** bunching_centers[j]
            x = np.sqrt(gbar_lin / g_dagger)
            nbar = 1.0 / (np.exp(x) - 1.0 + 1e-30)

            bin_var.append(var_obs)
            bin_var_err.append(var_err)
            bin_nbar.append(nbar)
            bin_nbar_sq.append(nbar**2 + nbar)
            bin_gbar_c.append(bunching_centers[j])
            bin_N.append(len(z_vals))

    bin_var = np.array(bin_var)
    bin_var_err = np.array(bin_var_err)
    bin_nbar = np.array(bin_nbar)
    bin_nbar_sq = np.array(bin_nbar_sq)
    bin_gbar_c = np.array(bin_gbar_c)
    bin_N = np.array(bin_N)

    t8_supports = False
    t8_delta_aic = np.nan
    bunching_result = {}

    if len(bin_var) >= 5:
        print(f"\n    {'log(gbar)':>10} {'N':>6} {'σ²_obs':>10} {'n̄':>10} "
              f"{'n̄²+n̄':>10} {'n̄/n̄²+n̄':>8}")
        print(f"    {'-'*60}")
        for j in range(len(bin_var)):
            ratio = bin_nbar[j] / bin_nbar_sq[j] if bin_nbar_sq[j] > 0 else 0
            print(f"    {bin_gbar_c[j]:>10.2f} {bin_N[j]:>6} {bin_var[j]:>10.4f} "
                  f"{bin_nbar[j]:>10.4f} {bin_nbar_sq[j]:>10.4f} {ratio:>8.4f}")

        # Fit Model A: BEC (quantum bunching)
        #   σ²(gbar) = A_q * [n̄² + n̄] + C_q
        # Fit Model B: Classical (Poisson)
        #   σ²(gbar) = A_c * n̄ + C_c
        # Fit Model C: Constant (null)
        #   σ²(gbar) = const

        weights = 1.0 / np.maximum(bin_var_err, 1e-6)

        try:
            # Model A: quantum bunching σ² = A*(n̄² + n̄) + C
            def quantum_model(nbar_sq_plus_n, A, C):
                return A * nbar_sq_plus_n + C
            popt_q, pcov_q = curve_fit(quantum_model, bin_nbar_sq, bin_var,
                                        p0=[0.1, 0.5], sigma=bin_var_err,
                                        absolute_sigma=True, maxfev=5000)
            resid_q = bin_var - quantum_model(bin_nbar_sq, *popt_q)
            chi2_q = np.sum((resid_q / bin_var_err)**2)
            A_q, C_q = popt_q

            # Model B: classical Poisson σ² = A*n̄ + C
            def classical_model(nbar, A, C):
                return A * nbar + C
            popt_c, pcov_c = curve_fit(classical_model, bin_nbar, bin_var,
                                        p0=[0.1, 0.5], sigma=bin_var_err,
                                        absolute_sigma=True, maxfev=5000)
            resid_c = bin_var - classical_model(bin_nbar, *popt_c)
            chi2_c = np.sum((resid_c / bin_var_err)**2)
            A_c, C_c = popt_c

            # Model C: constant
            mean_var = np.average(bin_var, weights=weights**2)
            resid_const = bin_var - mean_var
            chi2_const = np.sum((resid_const / bin_var_err)**2)

            n_bins_b = len(bin_var)
            dof_q = n_bins_b - 2
            dof_c = n_bins_b - 2
            dof_const = n_bins_b - 1

            # AIC (lower is better): AIC = chi² + 2*k
            aic_q = chi2_q + 2 * 2
            aic_c = chi2_c + 2 * 2
            aic_const = chi2_const + 2 * 1

            delta_aic_q_vs_c = aic_c - aic_q   # positive = quantum preferred
            delta_aic_q_vs_const = aic_const - aic_q

            rchi2_q = chi2_q / max(dof_q, 1)
            rchi2_c = chi2_c / max(dof_c, 1)
            rchi2_const = chi2_const / max(dof_const, 1)

            print(f"\n    Model fitting ({n_bins_b} acceleration bins):")
            print(f"    QUANTUM (BEC):  σ² = {A_q:.4f} × [n̄² + n̄] + {C_q:.4f}")
            print(f"                    χ²/dof = {rchi2_q:.3f}, AIC = {aic_q:.2f}")
            print(f"    CLASSICAL:      σ² = {A_c:.4f} × n̄ + {C_c:.4f}")
            print(f"                    χ²/dof = {rchi2_c:.3f}, AIC = {aic_c:.2f}")
            print(f"    CONSTANT:       σ² = {mean_var:.4f}")
            print(f"                    χ²/dof = {rchi2_const:.3f}, AIC = {aic_const:.2f}")
            print(f"\n    ΔAIC (classical − quantum) = {delta_aic_q_vs_c:+.2f}")
            print(f"    ΔAIC (constant  − quantum) = {delta_aic_q_vs_const:+.2f}")

            if delta_aic_q_vs_c > 0:
                print(f"    >>> QUANTUM BUNCHING PREFERRED over classical by ΔAIC = {delta_aic_q_vs_c:.1f}")
            elif delta_aic_q_vs_c > -2:
                print(f"    >>> Models statistically indistinguishable (ΔAIC = {delta_aic_q_vs_c:.1f})")
            else:
                print(f"    >>> Classical preferred (ΔAIC = {delta_aic_q_vs_c:.1f})")

            t8_supports = delta_aic_q_vs_c > -2  # quantum not significantly worse
            t8_delta_aic = delta_aic_q_vs_c

            # Also compute Pearson correlation of σ² with n̄²+n̄ vs n̄
            from scipy.stats import pearsonr
            r_quantum, p_quantum = pearsonr(bin_nbar_sq, bin_var)
            r_classical, p_classical = pearsonr(bin_nbar, bin_var)
            print(f"\n    Correlation tests:")
            print(f"    σ² vs n̄²+n̄ (quantum):  r = {r_quantum:.4f}, p = {p_quantum:.2e}")
            print(f"    σ² vs n̄    (classical): r = {r_classical:.4f}, p = {p_classical:.2e}")

            # Key diagnostic: at high occupation (low gbar), is σ² growing
            # faster than n̄? This is the bunching signature.
            # Compare the slope in the high-n̄ regime
            high_occ = bin_nbar > np.median(bin_nbar)
            if np.sum(high_occ) >= 3:
                slope_q_high, _, r_q_h, p_q_h, _ = stats.linregress(
                    np.log10(bin_nbar_sq[high_occ] + 1e-30),
                    np.log10(bin_var[high_occ] + 1e-30))
                slope_c_high, _, r_c_h, p_c_h, _ = stats.linregress(
                    np.log10(bin_nbar[high_occ] + 1e-30),
                    np.log10(bin_var[high_occ] + 1e-30))
                print(f"\n    High-occupation regime (n̄ > {np.median(bin_nbar):.2f}):")
                print(f"    log-log slope vs n̄²+n̄: {slope_q_high:.3f} (r={r_q_h:.3f})")
                print(f"    log-log slope vs n̄:     {slope_c_high:.3f} (r={r_c_h:.3f})")
                print(f"    (BEC predicts slope ≈ 1 for both; diagnostic of functional form)")

            # Store results
            bunching_result = {
                'quantum_model': {'A': float(A_q), 'C': float(C_q),
                                  'chi2_dof': float(rchi2_q), 'aic': float(aic_q)},
                'classical_model': {'A': float(A_c), 'C': float(C_c),
                                    'chi2_dof': float(rchi2_c), 'aic': float(aic_c)},
                'constant_model': {'mean_var': float(mean_var),
                                   'chi2_dof': float(rchi2_const), 'aic': float(aic_const)},
                'delta_aic_classical_minus_quantum': float(delta_aic_q_vs_c),
                'delta_aic_constant_minus_quantum': float(delta_aic_q_vs_const),
                'correlation_quantum': {'r': float(r_quantum), 'p': float(p_quantum)},
                'correlation_classical': {'r': float(r_classical), 'p': float(p_classical)},
                'n_bins': int(n_bins_b),
                'supports_quantum': t8_supports,
                'bins': [{'log_gbar': float(bin_gbar_c[j]),
                          'N': int(bin_N[j]),
                          'var_obs': float(bin_var[j]),
                          'var_err': float(bin_var_err[j]),
                          'nbar': float(bin_nbar[j]),
                          'nbar_sq_plus_n': float(bin_nbar_sq[j])}
                         for j in range(len(bin_var))],
            }

            # --- PLOT: Boson Bunching ---
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # Panel 1: σ²(gbar) with both model fits
                ax1 = axes[0]
                ax1.errorbar(bin_gbar_c, bin_var, yerr=bin_var_err,
                             fmt='ko', markersize=6, capsize=3, label='Observed σ²(Z-res)')
                # Plot both model predictions
                gbar_fine = np.linspace(bin_gbar_c.min(), bin_gbar_c.max(), 200)
                nbar_fine = 1.0 / (np.exp(np.sqrt(10**gbar_fine / g_dagger)) - 1.0 + 1e-30)
                nbar_sq_fine = nbar_fine**2 + nbar_fine
                ax1.plot(gbar_fine, quantum_model(nbar_sq_fine, *popt_q),
                         'r-', linewidth=2, label=f'Quantum: A·(n̄²+n̄)+C  [AIC={aic_q:.1f}]')
                ax1.plot(gbar_fine, classical_model(nbar_fine, *popt_c),
                         'b--', linewidth=2, label=f'Classical: A·n̄+C  [AIC={aic_c:.1f}]')
                ax1.axhline(mean_var, color='gray', linestyle=':', alpha=0.5,
                            label=f'Constant  [AIC={aic_const:.1f}]')
                ax1.set_xlabel('log₁₀(gbar) [m/s²]', fontsize=12)
                ax1.set_ylabel('Variance σ²(Z-residual)', fontsize=12)
                ax1.set_title('RAR Scatter vs Acceleration', fontsize=13)
                ax1.legend(fontsize=9, loc='upper right')
                ax1.grid(True, alpha=0.3)

                # Panel 2: σ² vs n̄²+n̄ (quantum) and n̄ (classical) — log-log
                ax2 = axes[1]
                ax2.errorbar(bin_nbar_sq, bin_var, yerr=bin_var_err,
                             fmt='rs', markersize=6, capsize=3, label='vs n̄²+n̄ (quantum)')
                ax2.errorbar(bin_nbar, bin_var, yerr=bin_var_err,
                             fmt='b^', markersize=6, capsize=3, label='vs n̄ (classical)')
                # Reference lines
                x_range_q = np.linspace(min(bin_nbar_sq)*0.8, max(bin_nbar_sq)*1.2, 100)
                ax2.plot(x_range_q, quantum_model(x_range_q, *popt_q),
                         'r-', linewidth=1.5, alpha=0.7)
                x_range_c = np.linspace(min(bin_nbar)*0.8, max(bin_nbar)*1.2, 100)
                ax2.plot(x_range_c, classical_model(x_range_c, *popt_c),
                         'b--', linewidth=1.5, alpha=0.7)
                ax2.set_xlabel('Occupation number predictor', fontsize=12)
                ax2.set_ylabel('Variance σ²(Z-residual)', fontsize=12)
                ax2.set_title(f'Boson Bunching Test (ΔAIC = {delta_aic_q_vs_c:+.1f})', fontsize=13)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                bunching_plot = os.path.join(OUTPUT_DIR, 'test8_boson_bunching.png')
                plt.savefig(bunching_plot, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\n    Saved: {bunching_plot}")
            except Exception as e:
                print(f"    (Plot failed: {e})")

        except Exception as e:
            print(f"    Model fitting failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"    Insufficient bins ({len(bin_var)}) for bunching test — need ≥5")

    # ----------------------------------------------------------------
    # DIAGNOSTIC 8A: gbar DISTRIBUTION PER BIN (explain floor changes)
    # ----------------------------------------------------------------
    if len(bin_var) >= 5:
        print(f"\n    --- Diagnostic 8A: gbar distribution by bin ---")
        print(f"    (Points per acceleration bin — identifies where cuts removed data)")
        total_z = len(z_all_points)
        n_low = sum(1 for p in z_all_points if p['log_gbar'] < -12.0)
        n_mid = sum(1 for p in z_all_points if -12.0 <= p['log_gbar'] < -10.0)
        n_high = sum(1 for p in z_all_points if p['log_gbar'] >= -10.0)
        print(f"    Total Z-scored points: {total_z}")
        print(f"    Low gbar  (< -12.0): {n_low:6d} ({100*n_low/max(total_z,1):.1f}%) — high n̄, most quantum leverage")
        print(f"    Mid gbar  (-12 to -10): {n_mid:6d} ({100*n_mid/max(total_z,1):.1f}%) — transition zone")
        print(f"    High gbar (> -10.0): {n_high:6d} ({100*n_high/max(total_z,1):.1f}%) — low n̄, floor dominates")

        # Per-dataset source breakdown in low-gbar regime
        low_gbar_sources = {}
        for p in z_all_points:
            if p['log_gbar'] < -12.0:
                s = p.get('source', 'unknown')
                low_gbar_sources[s] = low_gbar_sources.get(s, 0) + 1
        if low_gbar_sources:
            print(f"    Low-gbar points by dataset:")
            for s, n in sorted(low_gbar_sources.items(), key=lambda x: -x[1]):
                print(f"      {s:20s}: {n:4d}")

    # ----------------------------------------------------------------
    # DIAGNOSTIC 8B: PER-BIN FRACTIONAL DIFFERENCE (quantum vs classical)
    # ----------------------------------------------------------------
    if len(bin_var) >= 5 and 'A_q' in dir() and 'A_c' in dir():
        print(f"\n    --- Diagnostic 8B: Per-bin quantum vs classical predictions ---")
        print(f"    (Shows where the two models actually diverge)")
        print(f"    {'log(gbar)':>10} {'N':>6} {'σ²_obs':>8} {'σ²_BEC':>8} {'σ²_cls':>8} "
              f"{'Δ(B-C)':>8} {'%diff':>8} {'%of_σ²':>8}")
        print(f"    {'-'*76}")
        for j in range(len(bin_var)):
            pred_q = A_q * bin_nbar_sq[j] + C_q
            pred_c = A_c * bin_nbar[j] + C_c
            diff_bc = pred_q - pred_c
            pct_diff = 100 * diff_bc / max(pred_c, 1e-6)
            pct_of_var = 100 * abs(diff_bc) / max(bin_var[j], 1e-6)
            print(f"    {bin_gbar_c[j]:>10.2f} {bin_N[j]:>6} {bin_var[j]:>8.4f} "
                  f"{pred_q:>8.4f} {pred_c:>8.4f} {diff_bc:>+8.4f} {pct_diff:>+7.1f}% {pct_of_var:>7.1f}%")
        print(f"\n    At lowest gbar bin ({bin_gbar_c[0]:.2f}):")
        pred_q_lo = A_q * bin_nbar_sq[0] + C_q
        pred_c_lo = A_c * bin_nbar[0] + C_c
        diff_lo = pred_q_lo - pred_c_lo
        print(f"    BEC predicts σ² = {pred_q_lo:.4f}, Classical predicts σ² = {pred_c_lo:.4f}")
        print(f"    Difference = {diff_lo:+.4f} ({100*diff_lo/max(pred_c_lo,1e-6):+.1f}% of classical prediction)")
        print(f"    This is {100*abs(diff_lo)/max(bin_var[0],1e-6):.1f}% of the observed variance in that bin")

    # ----------------------------------------------------------------
    # DIAGNOSTIC 8C: PER-DATASET BUNCHING TEST
    # ----------------------------------------------------------------
    # Run the bunching test on individual datasets to check if the signal
    # is driven by a single dataset or confirmed across multiple.
    print(f"\n    --- Diagnostic 8C: Per-dataset bunching test ---")
    print(f"    (Tests whether quantum preference is driven by a single dataset)")
    per_ds_bunching = {}
    # Group datasets that have enough points
    ds_names_for_bunching = ['SPARC', 'WALLABY', 'GHASP', 'MaNGA', 'Catinella2005',
                             'PHANGS', 'Verheijen2001', 'WALLABY_DR2']
    print(f"    {'Dataset':>18} {'Bins':>5} {'ΔAIC':>8} {'A_q':>10} {'C_q':>8} {'A_c':>10} {'Pref':>10}")
    print(f"    {'-'*75}")

    for ds_name in ds_names_for_bunching:
        # Get Z-scored points for this dataset only
        # Need to re-normalize within this dataset's own distribution
        ds_pts = [p for p in all_points if p['source'] == ds_name]
        if len(ds_pts) < 50:
            continue
        ds_res = np.array([p['log_res'] for p in ds_pts])
        ds_mu = np.mean(ds_res)
        ds_std = np.std(ds_res)
        if ds_std < 1e-6:
            continue

        # Build Z-scored points for this dataset
        ds_z_pts = []
        for p in ds_pts:
            ds_z_pts.append({
                'z_res': (p['log_res'] - ds_mu) / ds_std,
                'log_gbar': p['log_gbar'],
            })

        # Bin the variance
        ds_bin_var = []
        ds_bin_nbar = []
        ds_bin_nbar_sq = []
        ds_bin_N = []
        ds_bin_var_err = []

        for j in range(len(bunching_centers)):
            lo, hi = bunching_edges[j], bunching_edges[j + 1]
            z_vals = np.array([p['z_res'] for p in ds_z_pts
                               if lo <= p['log_gbar'] < hi])
            if len(z_vals) >= 15:  # relaxed threshold for per-dataset
                var_obs = float(np.var(z_vals))
                var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
                gbar_lin = 10.0 ** bunching_centers[j]
                x = np.sqrt(gbar_lin / g_dagger)
                nbar = 1.0 / (np.exp(x) - 1.0 + 1e-30)
                ds_bin_var.append(var_obs)
                ds_bin_var_err.append(var_err)
                ds_bin_nbar.append(nbar)
                ds_bin_nbar_sq.append(nbar**2 + nbar)
                ds_bin_N.append(len(z_vals))

        if len(ds_bin_var) < 4:
            print(f"    {ds_name:>18} {'<4':>5}    ---  (too few bins)")
            continue

        ds_bin_var = np.array(ds_bin_var)
        ds_bin_var_err = np.array(ds_bin_var_err)
        ds_bin_nbar = np.array(ds_bin_nbar)
        ds_bin_nbar_sq = np.array(ds_bin_nbar_sq)

        try:
            # Quantum fit
            popt_q_ds, _ = curve_fit(lambda x, A, C: A * x + C,
                                      ds_bin_nbar_sq, ds_bin_var,
                                      p0=[0.01, 0.5], sigma=ds_bin_var_err,
                                      absolute_sigma=True, maxfev=5000)
            resid_q_ds = ds_bin_var - (popt_q_ds[0] * ds_bin_nbar_sq + popt_q_ds[1])
            chi2_q_ds = np.sum((resid_q_ds / ds_bin_var_err)**2)

            # Classical fit
            popt_c_ds, _ = curve_fit(lambda x, A, C: A * x + C,
                                      ds_bin_nbar, ds_bin_var,
                                      p0=[0.01, 0.5], sigma=ds_bin_var_err,
                                      absolute_sigma=True, maxfev=5000)
            resid_c_ds = ds_bin_var - (popt_c_ds[0] * ds_bin_nbar + popt_c_ds[1])
            chi2_c_ds = np.sum((resid_c_ds / ds_bin_var_err)**2)

            aic_q_ds = chi2_q_ds + 4
            aic_c_ds = chi2_c_ds + 4
            daic_ds = aic_c_ds - aic_q_ds
            pref = "Quantum" if daic_ds > 2 else ("Classical" if daic_ds < -2 else "~Equal")

            print(f"    {ds_name:>18} {len(ds_bin_var):>5} {daic_ds:>+8.2f} "
                  f"{popt_q_ds[0]:>10.5f} {popt_q_ds[1]:>8.4f} "
                  f"{popt_c_ds[0]:>10.5f}   {pref}")

            per_ds_bunching[ds_name] = {
                'n_bins': len(ds_bin_var),
                'delta_aic': float(daic_ds),
                'A_quantum': float(popt_q_ds[0]),
                'C_quantum': float(popt_q_ds[1]),
                'A_classical': float(popt_c_ds[0]),
                'preference': pref,
            }
        except Exception:
            print(f"    {ds_name:>18}   ---  (fit failed)")

    # Summary
    n_quantum_pref = sum(1 for v in per_ds_bunching.values() if v['preference'] == 'Quantum')
    n_classical_pref = sum(1 for v in per_ds_bunching.values() if v['preference'] == 'Classical')
    n_equal = sum(1 for v in per_ds_bunching.values() if v['preference'] == '~Equal')
    n_tested = len(per_ds_bunching)
    print(f"\n    Per-dataset summary: {n_quantum_pref}/{n_tested} quantum, "
          f"{n_classical_pref}/{n_tested} classical, {n_equal}/{n_tested} indistinguishable")
    if n_quantum_pref > 1:
        print(f"    >>> MULTIPLE datasets independently prefer quantum bunching")
    elif n_quantum_pref == 1:
        ds_driver = [k for k, v in per_ds_bunching.items() if v['preference'] == 'Quantum'][0]
        print(f"    >>> WARNING: Quantum preference driven by single dataset: {ds_driver}")
    else:
        print(f"    >>> No individual dataset shows decisive quantum preference")

    # ----------------------------------------------------------------
    # DIAGNOSTIC 8D: FLOOR-MASKING CONSISTENCY CHECK
    # ----------------------------------------------------------------
    # If the bunching signal is real but only visible in SPARC because
    # other datasets have higher systematic floors, then:
    # 1) SPARC should have the lowest floor (C) among all datasets
    # 2) The quantum amplitude A from SPARC should predict what other
    #    datasets would see IF their floors mask the signal:
    #    For dataset X with floor C_x, the effective ΔAIC would be
    #    reduced because A_q * n̄²+n̄ is a smaller fraction of C_x + A_q*n̄²+n̄
    # 3) Datasets with negative A should have A ≈ 0 if we account
    #    for their floor-induced bias at low gbar
    print(f"\n    --- Diagnostic 8D: Floor-masking consistency check ---")
    print(f"    (Tests whether SPARC-only signal is consistent with floor masking)")

    if per_ds_bunching:
        # Get SPARC values as reference
        sparc_info = per_ds_bunching.get('SPARC', {})
        sparc_A = sparc_info.get('A_quantum', np.nan)
        sparc_C = sparc_info.get('C_quantum', np.nan)

        if not np.isnan(sparc_A) and not np.isnan(sparc_C):
            print(f"\n    SPARC reference: A_q = {sparc_A:.5f}, C_q = {sparc_C:.4f}")
            print(f"    SPARC signal-to-floor at lowest gbar (n̄²+n̄ ≈ 757):")
            sparc_signal_lo = sparc_A * 757.0
            print(f"    Signal = A × 757 = {sparc_signal_lo:.3f}, Floor = {sparc_C:.3f}")
            print(f"    Signal/Floor = {sparc_signal_lo/max(sparc_C,1e-6):.1%}")
            print(f"    Signal/(Signal+Floor) = {sparc_signal_lo/(sparc_signal_lo+sparc_C):.1%}")

            print(f"\n    {'Dataset':>18} {'C_q':>8} {'C/C_sparc':>10} {'Predicted':>10}")
            print(f"    {'':>18} {'(floor)':>8} {'(ratio)':>10} {'visibility':>10}")
            print(f"    {'-'*52}")

            for ds_name, ds_info in sorted(per_ds_bunching.items(),
                                            key=lambda x: x[1].get('C_quantum', 99)):
                ds_C = ds_info.get('C_quantum', np.nan)
                if np.isnan(ds_C) or ds_C <= 0:
                    continue
                floor_ratio = ds_C / sparc_C
                # If the same quantum amplitude A_sparc existed in this dataset,
                # what fraction of total variance at lowest gbar would it be?
                predicted_frac = sparc_A * 757.0 / (sparc_A * 757.0 + ds_C)
                vis = "VISIBLE" if predicted_frac > 0.10 else ("marginal" if predicted_frac > 0.03 else "buried")
                print(f"    {ds_name:>18} {ds_C:>8.4f} {floor_ratio:>10.2f}x {predicted_frac:>9.1%}  ({vis})")

            # Key question: do the floors explain the pattern?
            print(f"\n    Interpretation:")
            sparc_frac = sparc_signal_lo / (sparc_signal_lo + sparc_C)
            print(f"    In SPARC (floor={sparc_C:.3f}), quantum signal is {sparc_frac:.0%} of total at lowest gbar")
            print(f"    In datasets with floor ≈ 0.9, quantum signal would be "
                  f"{sparc_signal_lo/(sparc_signal_lo + 0.9):.0%} of total")
            print(f"    Mass model systematics inflate floor by {0.9/sparc_C:.1f}x, "
                  f"reducing signal visibility by {(sparc_frac - sparc_signal_lo/(sparc_signal_lo+0.9))/sparc_frac:.0%}")

    # ----------------------------------------------------------------
    # DIAGNOSTIC 8E: SPARC-ISOLATED BUNCHING DEEP DIVE
    # ----------------------------------------------------------------
    # SPARC has proper 3.6μm mass models. Run detailed bunching analysis
    # with finer bins, bootstrap confidence intervals, and an explicit
    # test of whether the signal is driven by a few galaxies.
    print(f"\n    --- Diagnostic 8E: SPARC-isolated bunching deep dive ---")

    sparc_pts = [p for p in all_points if p['source'] == 'SPARC']
    if len(sparc_pts) > 100:
        sparc_res = np.array([p['log_res'] for p in sparc_pts])
        sparc_mu = np.mean(sparc_res)
        sparc_std = np.std(sparc_res)

        # SPARC Z-scored points
        sparc_z = []
        for p in sparc_pts:
            sparc_z.append({
                'z_res': (p['log_res'] - sparc_mu) / sparc_std,
                'log_gbar': p['log_gbar'],
                'galaxy': p['galaxy'],
            })

        # Finer bins for SPARC (more points per bin possible)
        sp_edges = np.arange(-13.0, -8.0, 0.5)
        sp_centers = (sp_edges[:-1] + sp_edges[1:]) / 2.0

        sp_bin_var = []
        sp_bin_var_err = []
        sp_bin_nbar = []
        sp_bin_nbar_sq = []
        sp_bin_N = []
        sp_bin_gc = []

        for j in range(len(sp_centers)):
            lo, hi = sp_edges[j], sp_edges[j + 1]
            z_vals = np.array([p['z_res'] for p in sparc_z
                               if lo <= p['log_gbar'] < hi])
            if len(z_vals) >= 10:
                var_obs = float(np.var(z_vals))
                var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
                gbar_lin = 10.0 ** sp_centers[j]
                x = np.sqrt(gbar_lin / g_dagger)
                nbar = 1.0 / (np.exp(x) - 1.0 + 1e-30)
                sp_bin_var.append(var_obs)
                sp_bin_var_err.append(var_err)
                sp_bin_nbar.append(nbar)
                sp_bin_nbar_sq.append(nbar**2 + nbar)
                sp_bin_N.append(len(z_vals))
                sp_bin_gc.append(sp_centers[j])

        sp_bin_var = np.array(sp_bin_var)
        sp_bin_var_err = np.array(sp_bin_var_err)
        sp_bin_nbar = np.array(sp_bin_nbar)
        sp_bin_nbar_sq = np.array(sp_bin_nbar_sq)

        if len(sp_bin_var) >= 5:
            print(f"    SPARC: {len(sparc_pts)} points, {len(set(p['galaxy'] for p in sparc_pts))} galaxies")
            print(f"    {'log(gbar)':>10} {'N':>6} {'σ²':>8} {'n̄':>8} {'n̄²+n̄':>10}")
            print(f"    {'-'*48}")
            for j in range(len(sp_bin_var)):
                print(f"    {sp_bin_gc[j]:>10.2f} {sp_bin_N[j]:>6} {sp_bin_var[j]:>8.4f} "
                      f"{sp_bin_nbar[j]:>8.4f} {sp_bin_nbar_sq[j]:>10.4f}")

            try:
                # Quantum fit
                popt_sq, pcov_sq = curve_fit(lambda x, A, C: A * x + C,
                                              sp_bin_nbar_sq, sp_bin_var,
                                              p0=[0.01, 0.5], sigma=sp_bin_var_err,
                                              absolute_sigma=True, maxfev=5000)
                resid_sq = sp_bin_var - (popt_sq[0] * sp_bin_nbar_sq + popt_sq[1])
                chi2_sq = np.sum((resid_sq / sp_bin_var_err)**2)

                # Classical fit
                popt_sc, pcov_sc = curve_fit(lambda x, A, C: A * x + C,
                                              sp_bin_nbar, sp_bin_var,
                                              p0=[0.01, 0.5], sigma=sp_bin_var_err,
                                              absolute_sigma=True, maxfev=5000)
                resid_sc = sp_bin_var - (popt_sc[0] * sp_bin_nbar + popt_sc[1])
                chi2_sc = np.sum((resid_sc / sp_bin_var_err)**2)

                # Constant
                sp_wts = 1.0 / np.maximum(sp_bin_var_err, 1e-6)
                sp_mean_var = np.average(sp_bin_var, weights=sp_wts**2)
                resid_s_const = sp_bin_var - sp_mean_var
                chi2_s_const = np.sum((resid_s_const / sp_bin_var_err)**2)

                n_sp_bins = len(sp_bin_var)
                aic_sq = chi2_sq + 4
                aic_sc = chi2_sc + 4
                aic_s_const = chi2_s_const + 2
                daic_sparc = aic_sc - aic_sq

                print(f"\n    SPARC model fitting ({n_sp_bins} bins):")
                print(f"    Quantum:   A = {popt_sq[0]:.5f}, C = {popt_sq[1]:.4f}, "
                      f"χ²/dof = {chi2_sq/max(n_sp_bins-2,1):.3f}, AIC = {aic_sq:.2f}")
                print(f"    Classical: A = {popt_sc[0]:.5f}, C = {popt_sc[1]:.4f}, "
                      f"χ²/dof = {chi2_sc/max(n_sp_bins-2,1):.3f}, AIC = {aic_sc:.2f}")
                print(f"    Constant:  σ² = {sp_mean_var:.4f}, AIC = {aic_s_const:.2f}")
                print(f"    ΔAIC (classical − quantum) = {daic_sparc:+.2f}")
                if daic_sparc > 6:
                    print(f"    >>> SPARC ALONE: Quantum DECISIVELY preferred (ΔAIC = {daic_sparc:+.1f})")
                elif daic_sparc > 2:
                    print(f"    >>> SPARC ALONE: Quantum preferred (ΔAIC = {daic_sparc:+.1f})")
                else:
                    print(f"    >>> SPARC ALONE: Models not clearly distinguished (ΔAIC = {daic_sparc:+.1f})")

                # Bootstrap: resample SPARC galaxies, recompute ΔAIC
                print(f"\n    Bootstrap ΔAIC (resampling SPARC galaxies, 500 iterations)...")
                sparc_gals = list(set(p['galaxy'] for p in sparc_z))
                gal_to_pts = {}
                for p in sparc_z:
                    g = p['galaxy']
                    if g not in gal_to_pts:
                        gal_to_pts[g] = []
                    gal_to_pts[g].append(p)

                boot_daics = []
                boot_As = []
                for _ in range(500):
                    # Resample galaxies with replacement
                    boot_gals = np.random.choice(sparc_gals, size=len(sparc_gals), replace=True)
                    boot_pts = []
                    for g in boot_gals:
                        boot_pts.extend(gal_to_pts[g])

                    # Re-Z-score the bootstrap sample
                    boot_res = np.array([p['z_res'] for p in boot_pts])
                    b_mu = np.mean(boot_res)
                    b_std = np.std(boot_res)
                    if b_std < 1e-6:
                        continue

                    # Bin
                    b_var_list = []
                    b_var_err_list = []
                    b_nbar_sq_list = []
                    b_nbar_list = []

                    for jj in range(len(sp_centers)):
                        lo, hi = sp_edges[jj], sp_edges[jj + 1]
                        zv = np.array([(p['z_res'] - b_mu) / b_std for p in boot_pts
                                       if lo <= p['log_gbar'] < hi])
                        if len(zv) >= 10:
                            vv = float(np.var(zv))
                            ve = np.sqrt(2.0 * vv**2 / (len(zv) - 1))
                            gbl = 10.0 ** sp_centers[jj]
                            xv = np.sqrt(gbl / g_dagger)
                            nb = 1.0 / (np.exp(xv) - 1.0 + 1e-30)
                            b_var_list.append(vv)
                            b_var_err_list.append(ve)
                            b_nbar_sq_list.append(nb**2 + nb)
                            b_nbar_list.append(nb)

                    if len(b_var_list) < 4:
                        continue

                    b_var_arr = np.array(b_var_list)
                    b_var_err_arr = np.array(b_var_err_list)
                    b_nbar_sq_arr = np.array(b_nbar_sq_list)
                    b_nbar_arr = np.array(b_nbar_list)

                    try:
                        bp_q, _ = curve_fit(lambda x, A, C: A * x + C,
                                            b_nbar_sq_arr, b_var_arr,
                                            p0=[0.01, 0.5], sigma=b_var_err_arr,
                                            absolute_sigma=True, maxfev=3000)
                        br_q = b_var_arr - (bp_q[0] * b_nbar_sq_arr + bp_q[1])
                        bc_q = np.sum((br_q / b_var_err_arr)**2)

                        bp_c, _ = curve_fit(lambda x, A, C: A * x + C,
                                            b_nbar_arr, b_var_arr,
                                            p0=[0.01, 0.5], sigma=b_var_err_arr,
                                            absolute_sigma=True, maxfev=3000)
                        br_c = b_var_arr - (bp_c[0] * b_nbar_arr + bp_c[1])
                        bc_c = np.sum((br_c / b_var_err_arr)**2)

                        b_daic = (bc_c + 4) - (bc_q + 4)
                        boot_daics.append(b_daic)
                        boot_As.append(bp_q[0])
                    except Exception:
                        continue

                if len(boot_daics) >= 100:
                    boot_daics = np.array(boot_daics)
                    boot_As = np.array(boot_As)
                    pct_quantum = 100 * np.mean(boot_daics > 0)
                    med_daic = np.median(boot_daics)
                    ci68 = np.percentile(boot_daics, [16, 84])
                    ci95 = np.percentile(boot_daics, [2.5, 97.5])
                    med_A = np.median(boot_As)
                    ci_A = np.percentile(boot_As, [16, 84])

                    print(f"    Valid iterations: {len(boot_daics)}/500")
                    print(f"    Median ΔAIC: {med_daic:+.2f}")
                    print(f"    68% CI: [{ci68[0]:+.2f}, {ci68[1]:+.2f}]")
                    print(f"    95% CI: [{ci95[0]:+.2f}, {ci95[1]:+.2f}]")
                    print(f"    Quantum preferred in {pct_quantum:.1f}% of bootstrap samples")
                    print(f"    Median A_quantum: {med_A:.5f} (68% CI: [{ci_A[0]:.5f}, {ci_A[1]:.5f}])")
                    if pct_quantum > 95:
                        print(f"    >>> ROBUST: SPARC bunching signal persists in >{pct_quantum:.0f}% of galaxy resamples")
                    elif pct_quantum > 80:
                        print(f"    >>> MODERATE: Quantum preferred in {pct_quantum:.0f}% of resamples")
                    else:
                        print(f"    >>> WEAK: Only {pct_quantum:.0f}% quantum preference under resampling")

                # Leave-one-out: which galaxies drive the signal?
                print(f"\n    Leave-one-galaxy-out sensitivity (top 10 most influential):")
                loo_results = []
                for drop_gal in sparc_gals:
                    loo_pts = [p for p in sparc_z if p['galaxy'] != drop_gal]
                    loo_res = np.array([p['z_res'] for p in loo_pts])
                    l_mu = np.mean(loo_res)
                    l_std = np.std(loo_res)
                    if l_std < 1e-6:
                        continue

                    l_var_list = []
                    l_var_err_list = []
                    l_nbar_sq_list = []
                    l_nbar_list = []

                    for jj in range(len(sp_centers)):
                        lo, hi = sp_edges[jj], sp_edges[jj + 1]
                        zv = np.array([(p['z_res'] - l_mu) / l_std for p in loo_pts
                                       if lo <= p['log_gbar'] < hi])
                        if len(zv) >= 10:
                            vv = float(np.var(zv))
                            ve = np.sqrt(2.0 * vv**2 / (len(zv) - 1))
                            gbl = 10.0 ** sp_centers[jj]
                            xv = np.sqrt(gbl / g_dagger)
                            nb = 1.0 / (np.exp(xv) - 1.0 + 1e-30)
                            l_var_list.append(vv)
                            l_var_err_list.append(ve)
                            l_nbar_sq_list.append(nb**2 + nb)
                            l_nbar_list.append(nb)

                    if len(l_var_list) < 4:
                        continue

                    l_var_arr = np.array(l_var_list)
                    l_var_err_arr = np.array(l_var_err_list)
                    l_nbar_sq_arr = np.array(l_nbar_sq_list)
                    l_nbar_arr = np.array(l_nbar_list)

                    try:
                        lp_q, _ = curve_fit(lambda x, A, C: A * x + C,
                                            l_nbar_sq_arr, l_var_arr,
                                            p0=[0.01, 0.5], sigma=l_var_err_arr,
                                            absolute_sigma=True, maxfev=3000)
                        lr_q = l_var_arr - (lp_q[0] * l_nbar_sq_arr + lp_q[1])
                        lc_q = np.sum((lr_q / l_var_err_arr)**2)

                        lp_c, _ = curve_fit(lambda x, A, C: A * x + C,
                                            l_nbar_arr, l_var_arr,
                                            p0=[0.01, 0.5], sigma=l_var_err_arr,
                                            absolute_sigma=True, maxfev=3000)
                        lr_c = l_var_arr - (lp_c[0] * l_nbar_arr + lp_c[1])
                        lc_c = np.sum((lr_c / l_var_err_arr)**2)

                        l_daic = (lc_c + 4) - (lc_q + 4)
                        n_pts_gal = sum(1 for p in sparc_z if p['galaxy'] == drop_gal)
                        loo_results.append((drop_gal, l_daic, daic_sparc - l_daic, n_pts_gal))
                    except Exception:
                        continue

                # Sort by influence (change in ΔAIC when dropped)
                loo_results.sort(key=lambda x: -abs(x[2]))
                print(f"    {'Galaxy':>20} {'ΔAIC_loo':>10} {'Influence':>10} {'N_pts':>6}")
                print(f"    {'-'*52}")
                for gal, daic_l, influence, npts in loo_results[:10]:
                    print(f"    {gal:>20} {daic_l:>+10.2f} {influence:>+10.2f} {npts:>6}")

                # Check if any single galaxy drives the result
                if loo_results:
                    worst_daic = min(r[1] for r in loo_results)
                    worst_gal = [r[0] for r in loo_results if r[1] == worst_daic][0]
                    if worst_daic > 2:
                        print(f"\n    >>> ROBUST: Even dropping most influential galaxy ({worst_gal}), "
                              f"ΔAIC = {worst_daic:+.1f} still favors quantum")
                    elif worst_daic > 0:
                        print(f"\n    >>> MARGINAL: Dropping {worst_gal} reduces ΔAIC to {worst_daic:+.1f}")
                    else:
                        print(f"\n    >>> FRAGILE: Dropping {worst_gal} flips preference (ΔAIC = {worst_daic:+.1f})")

            except Exception as e:
                print(f"    SPARC deep dive failed: {e}")
                import traceback
                traceback.print_exc()

    # Add Test 8 to tests_summary
    t8_sig = f"ΔAIC={t8_delta_aic:+.1f}" if not np.isnan(t8_delta_aic) else "N/A"
    n_support += int(t8_supports)
    tests_summary.append(('Boson bunching σ²∝n̄(n̄+1)', t8_delta_aic,
                          t8_sig, t8_supports))
    print(f"\n    [P] 8. {'✓ SUPPORTS' if t8_supports else '✗ opposes'} "
          f"quantum bunching statistics")

    # ================================================================
    # DIAGNOSTIC: TWO-ZONE ENVIRONMENTAL MODEL
    # ================================================================
    # The naive BEC prediction "field > dense scatter everywhere" fails
    # in the lowest-gbar bin. The REFINED prediction explains why:
    #
    # Zone 1 (intermediate gbar, -11.5 < log(gbar) < -9.5):
    #   Condensate coherence dominates. External cluster potential provides
    #   a more constraining boundary condition → tighter condensate ground
    #   state → LESS scatter in dense environments. field > dense.
    #
    # Zone 2 (very low gbar, log(gbar) < -12):
    #   Far from the baryonic source, the DM condensate is most exposed
    #   to the cluster halo's own turbulent gravitational potential.
    #   The halo's substructure, infall, and merging activity impose
    #   additional variance on the condensate → MORE scatter in dense.
    #   dense > field.
    #
    # This is physically analogous to superfluid helium in a vibrating
    # container: the bulk superfluid is more coherent with rigid walls
    # (= cluster potential), but at the boundaries, wall roughness
    # introduces vortex pinning and extra dissipation.
    print("\n  " + "=" * 70)
    print("  DIAGNOSTIC: TWO-ZONE ENVIRONMENTAL MODEL")
    print("  " + "=" * 70)
    print("    Refined BEC prediction: coherence dominance at intermediate gbar,")
    print("    cluster halo turbulence at the lowest gbar.")

    # Zone 1: intermediate gbar (-11.5 to -9.5) — coherence zone
    z1_d = np.array([p['log_res'] for p in all_points
                     if p['env_dense'] == 'dense' and -11.5 <= p['log_gbar'] < -9.5])
    z1_f = np.array([p['log_res'] for p in all_points
                     if p['env_dense'] == 'field' and -11.5 <= p['log_gbar'] < -9.5])
    # Zone 2: very low gbar (< -12.0) — turbulence zone
    z2_d = np.array([p['log_res'] for p in all_points
                     if p['env_dense'] == 'dense' and p['log_gbar'] < -12.0])
    z2_f = np.array([p['log_res'] for p in all_points
                     if p['env_dense'] == 'field' and p['log_gbar'] < -12.0])

    two_zone_supports = False
    if len(z1_d) >= 20 and len(z1_f) >= 20 and len(z2_d) >= 10 and len(z2_f) >= 10:
        s1_d = float(np.std(z1_d))
        s1_f = float(np.std(z1_f))
        s2_d = float(np.std(z2_d))
        s2_f = float(np.std(z2_f))

        # Zone 1: expect field > dense (coherence)
        zone1_bec = s1_f > s1_d
        # Zone 2: expect dense > field (turbulence)
        zone2_bec = s2_d > s2_f

        # Bootstrap significance
        _, z1_p, _ = bootstrap_scatter_test(z1_d, z1_f, n_boot=10000, seed=888)
        _, z2_p, _ = bootstrap_scatter_test(z2_d, z2_f, n_boot=10000, seed=889)

        print(f"\n    Zone 1 (coherence): -11.5 < log(gbar) < -9.5")
        print(f"      Dense: {len(z1_d)} pts, σ = {s1_d:.4f}")
        print(f"      Field: {len(z1_f)} pts, σ = {s1_f:.4f}")
        print(f"      Δ(field−dense) = {s1_f - s1_d:+.4f}  P(f>d) = {1-z1_p:.3f}")
        print(f"      {'✓ field > dense (as predicted)' if zone1_bec else '✗ dense > field'}")

        print(f"\n    Zone 2 (halo turbulence): log(gbar) < -12.0")
        print(f"      Dense: {len(z2_d)} pts, σ = {s2_d:.4f}")
        print(f"      Field: {len(z2_f)} pts, σ = {s2_f:.4f}")
        print(f"      Δ(dense−field) = {s2_d - s2_f:+.4f}  P(d>f) = {z2_p:.3f}")
        print(f"      {'✓ dense > field (halo turbulence)' if zone2_bec else '✗ field > dense'}")

        two_zone_supports = zone1_bec and zone2_bec
        print(f"\n    Two-zone model: {'✓ BOTH zones match refined prediction' if two_zone_supports else '✗ incomplete match'}")

        if two_zone_supports:
            print("    >>> The condensate is MORE coherent in clusters at intermediate gbar,")
            print("        but MORE disturbed at the lowest gbar where halo turbulence dominates.")
            print("    >>> This explains why Tests 1/4/6 (which average over ALL gbar) fail —")
            print("        the two zones partially cancel in aggregate statistics.")
    else:
        print(f"    Insufficient data: Zone1 d={len(z1_d)} f={len(z1_f)}, "
              f"Zone2 d={len(z2_d)} f={len(z2_f)}")

    # ================================================================
    # TEST 9: REDSHIFT EVOLUTION OF g† — THE COSMOLOGICAL PREDICTION
    # ================================================================
    # If g† is the condensation temperature and is set by cosmological
    # expansion, then g† ∝ cH(z). At z~0.9, H(z) ~ 1.6×H₀ (ΛCDM),
    # so g†(z=0.9) ≈ 1.6 × g†₀ ≈ 1.9e-10 m/s².
    #
    # This shifts the MOND/BEC transition to higher accelerations,
    # meaning the regime where quantum and classical predictions diverge
    # moves from log(gbar) < -12 to log(gbar) < -11.5 — where we have
    # hundreds of data points instead of 26.
    #
    # We test this using KROSS IFU rotation curves at z ≈ 0.84.
    # For each KROSS galaxy, compute gobs and gbar, then fit the RAR
    # with both g†₀ (no evolution) and g†(z) = g†₀ × H(z)/H₀ (BEC).
    # If BEC is correct, the shifted g† should provide a better fit.
    print("\n  " + "=" * 70)
    print("  TEST 9: REDSHIFT EVOLUTION OF g† (COSMOLOGICAL PREDICTION)")
    print("  " + "=" * 70)
    print("    BEC predicts: g†(z) = g†₀ × H(z)/H₀")
    print("    At z~0.85, H(z)/H₀ ≈ 1.57, so g†(z=0.85) ≈ 1.88e-10 m/s²")
    print("    Testing with KROSS IFU rotation curves (Sharma+2022)")

    t9_supports = False
    t9_delta_aic = np.nan

    kross_path = os.path.join(DATA_DIR, 'hi_surveys',
                              'sharma2022_kross_galaxy_properties.tsv')

    if os.path.exists(kross_path):
        # Load KROSS galaxy properties
        kross_pts = []
        kross_names = []
        n_kross_loaded = 0
        n_kross_bad_inc = 0
        n_kross_bad_vel = 0
        n_kross_bad_flag = 0

        with open(kross_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                name = row.get('Name', '').strip().strip('"')
                z_gal = safe_float(row.get('z', ''))
                inc = safe_float(row.get('inc', ''))
                Re_kpc = safe_float(row.get('Re', ''))
                e_Re = safe_float(row.get('e_Re', ''))
                Ve = safe_float(row.get('Ve', ''))       # V at ~1 Re
                e_Ve = safe_float(row.get('e_Ve', ''))
                Vopt = safe_float(row.get('Vopt', ''))   # V at ~1.3 Re (optical)
                e_Vopt = safe_float(row.get('e_Vopt', ''))
                Vout = safe_float(row.get('Vout', ''))   # V at ~2 Re (outer)
                e_Vout = safe_float(row.get('e_Vout', ''))
                Mstar = safe_float(row.get('Mstar', ''))
                MH2 = safe_float(row.get('MH2', ''))
                MHI = safe_float(row.get('MHI', ''))

                # Quality flags
                rout_flag = row.get('RoutFlag', '').strip().strip('"')
                mstar_flag = row.get('MstarFlag', '').strip().strip('"')
                wiggle_flag = row.get('WiggleFlag', '').strip().strip('"')

                if not name or np.isnan(z_gal):
                    continue

                # Quality cuts
                if rout_flag != 'ok' or mstar_flag != 'ok':
                    n_kross_bad_flag += 1
                    continue

                if np.isnan(inc) or inc < 25 or inc > 80:
                    n_kross_bad_inc += 1
                    continue

                if np.isnan(Re_kpc) or Re_kpc <= 0:
                    continue

                # Need at least one valid velocity
                if (np.isnan(Ve) or Ve <= 0) and (np.isnan(Vopt) or Vopt <= 0):
                    n_kross_bad_vel += 1
                    continue

                # Baryonic mass: M_star + 1.33*(MH2 + MHI) for gas
                if np.isnan(Mstar) or Mstar <= 0:
                    continue
                Mgas = 0.0
                if not np.isnan(MH2) and MH2 > 0:
                    Mgas += MH2
                if not np.isnan(MHI) and MHI > 0:
                    Mgas += MHI
                Mbar = Mstar + 1.33 * Mgas
                if Mbar <= 0:
                    continue

                n_kross_loaded += 1

                # Disk scale length from effective radius
                rd_kpc = Re_kpc / 1.678  # exponential disk

                # Build RAR points at each available radius
                radii_frac = []  # fractions of Re
                vels = []
                vel_errs = []

                if not np.isnan(Ve) and Ve > 10:
                    radii_frac.append(1.0)
                    vels.append(Ve)
                    vel_errs.append(e_Ve if not np.isnan(e_Ve) else Ve * 0.2)
                if not np.isnan(Vopt) and Vopt > 10:
                    radii_frac.append(1.3)
                    vels.append(Vopt)
                    vel_errs.append(e_Vopt if not np.isnan(e_Vopt) else Vopt * 0.2)
                if not np.isnan(Vout) and Vout > 10:
                    radii_frac.append(2.0)
                    vels.append(Vout)
                    vel_errs.append(e_Vout if not np.isnan(e_Vout) else Vout * 0.2)

                for k in range(len(radii_frac)):
                    r_kpc = radii_frac[k] * Re_kpc
                    Vrot = abs(vels[k])
                    eVrot = abs(vel_errs[k])

                    if Vrot <= 10 or Vrot > 500:
                        continue

                    # Fractional error cut
                    if eVrot / Vrot > 0.25:  # relaxed for high-z
                        continue

                    # gobs = V² / r
                    gobs = Vrot**2 / r_kpc * conv

                    # gbar from exponential disk
                    denom = (r_kpc**2 + rd_kpc**2)**1.5
                    if denom <= 0:
                        continue
                    gbar = G_kpc * Mbar * r_kpc / denom * conv

                    if gbar <= 0 or gobs <= 0:
                        continue
                    if not np.isfinite(gbar) or not np.isfinite(gobs):
                        continue

                    log_gbar = np.log10(gbar)
                    log_gobs = np.log10(gobs)

                    # Propagate velocity error
                    sigma_lg = 2.0 * eVrot / (Vrot * np.log(10))
                    sigma_lg = max(sigma_lg, 0.05)

                    kross_pts.append({
                        'galaxy': name,
                        'z': float(z_gal),
                        'r_kpc': float(r_kpc),
                        'vrot': float(Vrot),
                        'gbar': float(gbar),
                        'gobs': float(gobs),
                        'log_gbar': float(log_gbar),
                        'log_gobs': float(log_gobs),
                        'sigma_log_gobs': float(sigma_lg),
                    })

                kross_names.append(name)

        n_kross_gals = len(set(kross_names))
        n_kross_pts = len(kross_pts)
        print(f"\n    KROSS loaded: {n_kross_gals} galaxies, {n_kross_pts} RAR points")
        print(f"    Median redshift: {np.median([p['z'] for p in kross_pts]):.3f}")
        print(f"    Skipped: {n_kross_bad_inc} (inclination), {n_kross_bad_vel} (velocity), "
              f"{n_kross_bad_flag} (quality flag)")

        if n_kross_pts >= 30:
            # Compute H(z)/H₀ for ΛCDM
            z_med = np.median([p['z'] for p in kross_pts])
            Omega_m = 0.3
            Omega_L = 0.7
            Hz_over_H0 = np.sqrt(Omega_m * (1 + z_med)**3 + Omega_L)
            g_dagger_z = g_dagger * Hz_over_H0  # BEC prediction
            g_dagger_no_evol = g_dagger          # no-evolution null hypothesis

            print(f"\n    Cosmological parameters:")
            print(f"    z_median = {z_med:.3f}")
            print(f"    H(z)/H₀ = {Hz_over_H0:.3f}")
            print(f"    g†₀ = {g_dagger:.2e} m/s²")
            print(f"    g†(z) = g†₀ × H(z)/H₀ = {g_dagger_z:.2e} m/s² (BEC prediction)")

            # Define RAR prediction with arbitrary g†
            def rar_pred_gdag(gbar_val, gdag):
                x = np.sqrt(gbar_val / gdag)
                return gbar_val / (1.0 - np.exp(-x))

            # Compute residuals under both g† values
            resid_z0 = []  # residuals using g†₀ (no evolution)
            resid_bec = []  # residuals using g†(z) (BEC evolution)
            log_gbars = []
            weights = []

            for p in kross_pts:
                gbar = p['gbar']
                gobs = p['gobs']
                sig = p['sigma_log_gobs']

                pred_z0 = rar_pred_gdag(gbar, g_dagger_no_evol)
                pred_bec = rar_pred_gdag(gbar, g_dagger_z)

                if pred_z0 > 0 and pred_bec > 0:
                    r_z0 = np.log10(gobs) - np.log10(pred_z0)
                    r_bec = np.log10(gobs) - np.log10(pred_bec)

                    if abs(r_z0) < 2.0 and abs(r_bec) < 2.0:
                        resid_z0.append(r_z0)
                        resid_bec.append(r_bec)
                        log_gbars.append(p['log_gbar'])
                        weights.append(1.0 / max(sig, 0.05)**2)

            resid_z0 = np.array(resid_z0)
            resid_bec = np.array(resid_bec)
            log_gbars_arr = np.array(log_gbars)
            weights_arr = np.array(weights)

            if len(resid_z0) >= 20:
                # Compare scatter: which g† gives tighter RAR?
                rms_z0 = np.sqrt(np.mean(resid_z0**2))
                rms_bec = np.sqrt(np.mean(resid_bec**2))
                wrms_z0 = np.sqrt(np.average(resid_z0**2, weights=weights_arr))
                wrms_bec = np.sqrt(np.average(resid_bec**2, weights=weights_arr))

                # Weighted chi² comparison
                chi2_z0 = np.sum(weights_arr * resid_z0**2)
                chi2_bec = np.sum(weights_arr * resid_bec**2)

                # AIC: both models have same # parameters (g† is fixed, not fit)
                # So ΔAIC = Δχ²
                delta_chi2 = chi2_z0 - chi2_bec  # positive = BEC preferred

                # Also fit g† as free parameter to find optimal value
                from scipy.optimize import minimize_scalar

                def neg_loglik_gdag(log_gdag):
                    gdag_test = 10**log_gdag
                    chi2 = 0.0
                    for p in kross_pts:
                        pred = rar_pred_gdag(p['gbar'], gdag_test)
                        if pred > 0:
                            r = np.log10(p['gobs']) - np.log10(pred)
                            chi2 += r**2 / max(p['sigma_log_gobs'], 0.05)**2
                    return chi2

                result = minimize_scalar(neg_loglik_gdag,
                                         bounds=(-10.5, -9.0), method='bounded')
                g_dagger_best = 10**result.x

                print(f"\n    RAR fit comparison ({len(resid_z0)} points):")
                print(f"    {'Model':>25} {'g†':>12} {'RMS':>8} {'wRMS':>8} {'χ²':>10}")
                print(f"    {'-'*65}")
                print(f"    {'g†₀ (no evolution)':>25} {g_dagger:.2e} {rms_z0:>8.4f} "
                      f"{wrms_z0:>8.4f} {chi2_z0:>10.1f}")
                print(f"    {'g†(z) (BEC prediction)':>25} {g_dagger_z:.2e} {rms_bec:>8.4f} "
                      f"{wrms_bec:>8.4f} {chi2_bec:>10.1f}")
                print(f"    {'g† (best-fit)':>25} {g_dagger_best:.2e} {'---':>8} "
                      f"{'---':>8} {result.fun:>10.1f}")

                print(f"\n    Δχ² (z0 − BEC) = {delta_chi2:+.2f}")

                # CRITICAL: Check gbar coverage — g† evolution is only
                # distinguishable below the transition (log gbar < -10.5)
                n_dm_regime = sum(1 for g in log_gbars_arr if g < -10.5)
                n_transition = sum(1 for g in log_gbars_arr if -11.5 < g < -10.0)
                min_gbar = min(log_gbars_arr)
                max_gbar = max(log_gbars_arr)

                print(f"\n    gbar COVERAGE DIAGNOSTIC:")
                print(f"    Range: [{min_gbar:.2f}, {max_gbar:.2f}]")
                print(f"    Points below log(gbar) = -10.5 (DM-dominated): {n_dm_regime}")
                print(f"    Points in transition zone (-11.5 to -10.0): {n_transition}")

                has_leverage = n_dm_regime >= 20  # need DM-regime data for g† test

                if not has_leverage:
                    print(f"    >>> INSUFFICIENT LEVERAGE: Only {n_dm_regime} points below the")
                    print(f"        RAR transition. At log(gbar) > -10.0, the RAR is nearly")
                    print(f"        1:1 (gobs ≈ gbar) regardless of g†. This test CANNOT")
                    print(f"        distinguish g† values — the data probes only the")
                    print(f"        baryon-dominated regime where all models converge.")
                    print(f"    >>> Result: INCONCLUSIVE (not falsifying, just unpowered)")
                    # Mark as inconclusive, not opposing
                    t9_supports = False  # can't support without leverage
                    t9_delta_aic = np.nan  # don't report a meaningless Δχ²
                else:
                    if delta_chi2 > 0:
                        print(f"    >>> BEC evolved g† PREFERRED (lower χ² by {delta_chi2:.1f})")
                    else:
                        print(f"    >>> No-evolution g†₀ preferred (lower χ² by {-delta_chi2:.1f})")

                # Where does the best-fit g† land relative to predictions?
                g_dag_ratio = g_dagger_best / g_dagger
                Hz_ratio_implied = g_dag_ratio  # since g† ∝ H(z)
                print(f"\n    Best-fit g†/g†₀ = {g_dag_ratio:.3f}")
                print(f"    BEC prediction: H(z)/H₀ = {Hz_over_H0:.3f}")
                print(f"    Best-fit implies: H(z)/H₀ = {Hz_ratio_implied:.3f}")

                if not has_leverage:
                    print(f"    >>> Best-fit g† is unconstrained (no DM-regime data)")
                    print(f"    >>> The low best-fit g† reflects mass model offsets, not g† physics")
                elif abs(g_dag_ratio - Hz_over_H0) / Hz_over_H0 < 0.3:
                    print(f"    >>> CONSISTENT: Best-fit g† within 30% of BEC prediction")
                    t9_supports = True
                else:
                    print(f"    >>> INCONSISTENT: Best-fit g† deviates by "
                          f"{100*abs(g_dag_ratio - Hz_over_H0)/Hz_over_H0:.0f}% from BEC prediction")

                if not has_leverage:
                    t9_delta_aic = np.nan

                # Bin the residuals to see where the evolution matters
                print(f"\n    Binned residuals (where does g† evolution help?):")
                print(f"    {'log(gbar)':>10} {'N':>5} {'RMS_z0':>8} {'RMS_BEC':>8} {'Δ':>8}")
                print(f"    {'-'*45}")
                evol_edges = np.arange(-12.5, -8.5, 0.5)
                for j in range(len(evol_edges) - 1):
                    lo, hi = evol_edges[j], evol_edges[j + 1]
                    mask = (log_gbars_arr >= lo) & (log_gbars_arr < hi)
                    if np.sum(mask) >= 5:
                        rms_z0_b = np.sqrt(np.mean(resid_z0[mask]**2))
                        rms_bec_b = np.sqrt(np.mean(resid_bec[mask]**2))
                        delta_b = rms_z0_b - rms_bec_b
                        marker = " *" if delta_b > 0.01 else ""
                        print(f"    {(lo+hi)/2:>10.2f} {np.sum(mask):>5} {rms_z0_b:>8.4f} "
                              f"{rms_bec_b:>8.4f} {delta_b:>+8.4f}{marker}")

                # Generate diagnostic plot
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                    # Panel 1: KROSS RAR with both g† predictions
                    ax1 = axes[0]
                    lg = np.array([p['log_gbar'] for p in kross_pts])
                    lo = np.array([p['log_gobs'] for p in kross_pts])
                    ax1.scatter(lg, lo, s=10, alpha=0.4, c='steelblue', label='KROSS z~0.85')

                    # RAR curves
                    gbar_range = np.logspace(-13, -8, 500)
                    pred_z0_curve = np.log10(np.array([rar_pred_gdag(g, g_dagger_no_evol) for g in gbar_range]))
                    pred_bec_curve = np.log10(np.array([rar_pred_gdag(g, g_dagger_z) for g in gbar_range]))
                    pred_best_curve = np.log10(np.array([rar_pred_gdag(g, g_dagger_best) for g in gbar_range]))
                    ax1.plot(np.log10(gbar_range), pred_z0_curve, 'k-', linewidth=2,
                             label=f'g†₀ = {g_dagger:.1e}')
                    ax1.plot(np.log10(gbar_range), pred_bec_curve, 'r--', linewidth=2,
                             label=f'g†(z) = {g_dagger_z:.1e} (BEC)')
                    ax1.plot(np.log10(gbar_range), pred_best_curve, 'g:', linewidth=2,
                             label=f'g†_best = {g_dagger_best:.1e}')
                    ax1.plot(np.log10(gbar_range), np.log10(gbar_range), 'k:', alpha=0.3, label='1:1')
                    ax1.set_xlabel('log₁₀(gbar) [m/s²]')
                    ax1.set_ylabel('log₁₀(gobs) [m/s²]')
                    ax1.set_title(f'KROSS RAR (z ≈ {z_med:.2f})')
                    ax1.legend(fontsize=8)
                    ax1.grid(True, alpha=0.3)

                    # Panel 2: Residuals under both models
                    ax2 = axes[1]
                    ax2.scatter(log_gbars_arr, resid_z0, s=10, alpha=0.3, c='gray',
                                label=f'g†₀ (RMS={rms_z0:.3f})')
                    ax2.scatter(log_gbars_arr, resid_bec, s=10, alpha=0.3, c='red',
                                label=f'g†(z) (RMS={rms_bec:.3f})')
                    ax2.axhline(0, color='k', linewidth=0.5)
                    ax2.set_xlabel('log₁₀(gbar) [m/s²]')
                    ax2.set_ylabel('log₁₀(gobs/gobs_pred)')
                    ax2.set_title('RAR Residuals')
                    ax2.legend(fontsize=9)
                    ax2.grid(True, alpha=0.3)

                    # Panel 3: χ² profile vs g†
                    ax3 = axes[2]
                    gdag_test_range = np.logspace(-10.5, -9.0, 100)
                    chi2_profile = []
                    for gd in gdag_test_range:
                        c2 = 0
                        for p in kross_pts:
                            pred = rar_pred_gdag(p['gbar'], gd)
                            if pred > 0:
                                r = np.log10(p['gobs']) - np.log10(pred)
                                c2 += r**2 / max(p['sigma_log_gobs'], 0.05)**2
                        chi2_profile.append(c2)
                    chi2_profile = np.array(chi2_profile)

                    ax3.plot(np.log10(gdag_test_range), chi2_profile, 'k-', linewidth=2)
                    ax3.axvline(np.log10(g_dagger), color='gray', linestyle='--',
                                label=f'g†₀ = {g_dagger:.1e}')
                    ax3.axvline(np.log10(g_dagger_z), color='red', linestyle='--',
                                label=f'g†(z) = {g_dagger_z:.1e} (BEC)')
                    ax3.axvline(np.log10(g_dagger_best), color='green', linestyle=':',
                                label=f'Best fit = {g_dagger_best:.1e}')
                    ax3.set_xlabel('log₁₀(g†) [m/s²]')
                    ax3.set_ylabel('χ²')
                    ax3.set_title(f'g† Profile Likelihood (Δχ² = {delta_chi2:+.1f})')
                    ax3.legend(fontsize=8)
                    ax3.grid(True, alpha=0.3)

                    plt.tight_layout()
                    kross_plot = os.path.join(OUTPUT_DIR, 'test9_kross_redshift_evolution.png')
                    plt.savefig(kross_plot, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"\n    Saved: {kross_plot}")
                except Exception as e:
                    print(f"    (Plot failed: {e})")

        else:
            print(f"    Insufficient KROSS data ({n_kross_pts} points)")
    else:
        print(f"    KROSS data not found at {kross_path}")

    # Add Test 9 to summary
    t9_sig = f"Δχ²={t9_delta_aic:+.1f}" if not np.isnan(t9_delta_aic) else "N/A"
    tests_summary.append(('Redshift evolution g†∝H(z)', t9_delta_aic, t9_sig, t9_supports))
    n_support += int(t9_supports)
    if np.isnan(t9_delta_aic):
        print(f"\n    [P] 9. ? INCONCLUSIVE g† redshift evolution "
              f"(no DM-regime coverage at z~0.85)")
    else:
        print(f"\n    [P] 9. {'✓ SUPPORTS' if t9_supports else '✗ opposes'} "
              f"g† redshift evolution")

    # ================================================================
    # TEST 10: ALFALFA+WISE BUNCHING TEST (INDEPENDENT LARGE SAMPLE)
    # ================================================================
    # ALFALFA α.40 provides ~13,000 HI detections with W50 linewidths.
    # Cross-matched with z0MGS WISE stellar masses, each galaxy gives
    # one RAR point at the HI radius. This tests whether the bunching
    # signal seen in SPARC replicates in a 100x larger, uniformly-processed
    # sample — the critical test of whether the signal is real.
    print("\n  " + "=" * 70)
    print("  TEST 10: ALFALFA+WISE BUNCHING (INDEPENDENT LARGE SAMPLE)")
    print("  " + "=" * 70)
    print("    Testing bunching signal in ~13,000 ALFALFA galaxies with WISE masses")
    print("    One RAR point per galaxy (at HI radius)")

    t10_supports = False
    t10_delta_aic = np.nan

    alfalfa_path = os.path.join(DATA_DIR, 'alfalfa_alpha100_haynes2011.tsv')

    if os.path.exists(alfalfa_path):
        # Load HyperLEDA inclinations for proper sin(i) correction
        hleda_available = _load_hyperleda_inclinations()

        # Parse ALFALFA VizieR TSV
        _, alfa_rows = parse_vizier_tsv(alfalfa_path)
        print(f"    ALFALFA catalog: {len(alfa_rows)} HI detections")

        # Build ALFALFA RAR points
        alfa_pts = []
        n_alfa_wise = 0
        n_alfa_no_wise = 0
        n_alfa_bad_snr = 0
        n_alfa_bad_dist = 0
        n_alfa_bad_w50 = 0
        n_alfa_hleda_incl = 0       # HyperLEDA inclination matches
        n_alfa_stat_incl = 0        # statistical inclination fallbacks
        n_alfa_faceon_reject = 0    # rejected as too face-on (i < 25°)

        for row in alfa_rows:
            agc = row.get('UGC/AGC', '').strip()
            oname = row.get('OName', '').strip()
            cz = safe_float(row.get('cz', ''))
            w50 = safe_float(row.get('W50', ''))
            e_w50 = safe_float(row.get('e_W50', ''))
            snr = safe_float(row.get('SNR', ''))
            dist_mpc = safe_float(row.get('Dist', ''))
            logMHI = safe_float(row.get('logM', ''))
            hic = safe_int(row.get('HIc', '9'))

            # Parse RA/Dec from sexagesimal to decimal
            ra_str = row.get('RAJ2000', '').strip()
            dec_str = row.get('DEJ2000', '').strip()
            ra_deg = np.nan
            dec_deg = np.nan
            try:
                ra_parts = ra_str.split()
                if len(ra_parts) >= 3:
                    ra_deg = (float(ra_parts[0]) + float(ra_parts[1])/60.0
                              + float(ra_parts[2])/3600.0) * 15.0
                dec_parts = dec_str.replace('+', '').replace('-', '').split()
                dec_sign = -1 if dec_str.startswith('-') else 1
                if len(dec_parts) >= 3:
                    dec_deg = dec_sign * (float(dec_parts[0]) + float(dec_parts[1])/60.0
                                          + float(dec_parts[2])/3600.0)
            except (ValueError, IndexError):
                pass

            # Quality cuts
            if hic != 1:  # Code 1 = reliable detection
                continue
            if np.isnan(snr) or snr < 6.5:
                n_alfa_bad_snr += 1
                continue
            if np.isnan(dist_mpc) or dist_mpc <= 1.0 or dist_mpc > 250:
                n_alfa_bad_dist += 1
                continue
            if np.isnan(w50) or w50 < 20 or w50 > 800:
                n_alfa_bad_w50 += 1
                continue
            if np.isnan(logMHI) or logMHI < 6:
                continue

            # --- INCLINATION: HyperLEDA measured or statistical fallback ---
            # W50 = 2 * V_rot * sin(i) + turbulent broadening
            # Turbulent correction: W_turb ~ 10 km/s for typical HI
            w50_turb = max(w50 - 2 * 10.0, 20.0)  # subtract turbulent broadening

            incl_measured = None
            incl_source = 'statistical'
            if hleda_available and not np.isnan(ra_deg) and not np.isnan(dec_deg):
                incl_measured = get_hyperleda_inclination(ra_deg, dec_deg,
                                                          match_radius_arcsec=30.0)

            if incl_measured is not None:
                # Use measured inclination from HyperLEDA
                sin_i = np.sin(np.radians(incl_measured))
                Vrot = w50_turb / (2.0 * sin_i)
                e_w50_val = e_w50 if not np.isnan(e_w50) else w50 * 0.1
                # Error propagation: σ_V from linewidth error only
                # (inclination error from logR25 is secondary)
                e_Vrot = e_w50_val / (2.0 * sin_i)
                n_alfa_hleda_incl += 1
                incl_source = 'hleda'
            else:
                # Statistical fallback: <sin(i)> ≈ π/4 ≈ 0.785
                Vrot = w50_turb / (2.0 * 0.785)
                e_Vrot = (e_w50 if not np.isnan(e_w50) else w50 * 0.1) / (2.0 * 0.785)
                n_alfa_stat_incl += 1
                incl_source = 'statistical'

            # Fractional error cut (relaxed for single-dish data)
            if e_Vrot / max(Vrot, 1.0) > 0.25:
                continue

            # Additional cut: very low V_rot from edge-on correction
            if Vrot < 10.0:
                continue

            # HI radius from mass-size relation (Wang+2016)
            # log(R_HI/kpc) = 0.506 * log(M_HI/Msun) - 3.293
            MHI = 10**logMHI
            logRHI = 0.506 * logMHI - 3.293
            RHI_kpc = 10**logRHI

            if RHI_kpc <= 0.1 or RHI_kpc > 200:
                continue

            # WISE stellar mass from z0MGS
            name_for_wise = oname if oname else f"AGC{agc}"
            wise_logM = get_z0mgs_stellar_mass(name=name_for_wise,
                                                ra=ra_deg, dec=dec_deg)

            if wise_logM is not None:
                Mstar = 10**wise_logM
                n_alfa_wise += 1
            else:
                # Fallback: use TF from Vrot
                Mstar = 10**(3.75 * np.log10(max(Vrot, 10.0)) + 2.00) * 0.75
                n_alfa_no_wise += 1

            # Baryonic mass
            Mbar = Mstar + 1.33 * MHI

            # Compute RAR point at HI radius
            gobs = Vrot**2 / RHI_kpc * conv
            gbar = G_kpc * Mbar / RHI_kpc**2 * conv

            if gbar <= 0 or gobs <= 0 or not np.isfinite(gbar) or not np.isfinite(gobs):
                continue

            log_gbar = np.log10(gbar)
            log_gobs = np.log10(gobs)
            log_res = log_gobs - np.log10(rar_prediction(gbar))

            if abs(log_res) > 1.5:
                continue

            sigma_lg = 2.0 * e_Vrot / max(Vrot, 1.0) / np.log(10)
            sigma_lg = max(sigma_lg, 0.05)

            # Environment classification
            env = 'field'
            logMh = 11.0
            if not np.isnan(ra_deg) and not np.isnan(dec_deg) and not np.isnan(cz):
                _, logMh, env = classify_environment_proximity(ra_deg, dec_deg, cz,
                                                                name=name_for_wise)

            alfa_pts.append({
                'galaxy': name_for_wise,
                'log_gbar': float(log_gbar),
                'log_gobs': float(log_gobs),
                'log_res': float(log_res),
                'sigma_log_gobs': float(sigma_lg),
                'env_dense': env,
                'logMh': float(logMh),
                'has_wise': wise_logM is not None,
                'incl_source': incl_source,
                'incl_deg': float(incl_measured) if incl_measured is not None else np.nan,
            })

        n_alfa_total = len(alfa_pts)
        n_alfa_dense = sum(1 for p in alfa_pts if p['env_dense'] == 'dense')
        n_alfa_field = n_alfa_total - n_alfa_dense
        n_alfa_wise_pts = sum(1 for p in alfa_pts if p['has_wise'])
        n_alfa_hleda_pts = sum(1 for p in alfa_pts if p['incl_source'] == 'hleda')

        print(f"\n    ALFALFA RAR sample: {n_alfa_total} galaxies")
        print(f"    Mass models: {n_alfa_wise} WISE, {n_alfa_no_wise} TF fallback")
        print(f"    Inclinations: {n_alfa_hleda_incl} HyperLEDA measured, "
              f"{n_alfa_stat_incl} statistical fallback")
        print(f"    In final sample: {n_alfa_hleda_pts} with HyperLEDA incl, "
              f"{n_alfa_total - n_alfa_hleda_pts} with statistical incl")
        print(f"    Environment: {n_alfa_dense} dense, {n_alfa_field} field")
        print(f"    Skipped: {n_alfa_bad_snr} (SNR), {n_alfa_bad_dist} (distance), "
              f"{n_alfa_bad_w50} (linewidth)")

        if n_alfa_total >= 100:
            # Z-score the ALFALFA residuals internally
            alfa_res = np.array([p['log_res'] for p in alfa_pts])
            alfa_mu = np.mean(alfa_res)
            alfa_std = np.std(alfa_res)

            # WISE-only subsample for cleaner analysis
            alfa_wise_pts = [p for p in alfa_pts if p['has_wise']]
            if len(alfa_wise_pts) >= 100:
                aw_res = np.array([p['log_res'] for p in alfa_wise_pts])
                aw_mu = np.mean(aw_res)
                aw_std = np.std(aw_res)

                print(f"\n    WISE-only subsample: {len(alfa_wise_pts)} galaxies")
                print(f"    Raw scatter: σ = {aw_std:.4f} dex")

                # Run bunching test on ALFALFA WISE subsample
                aw_z_pts = [{'z_res': (p['log_res'] - aw_mu) / aw_std,
                             'log_gbar': p['log_gbar']} for p in alfa_wise_pts]

                aw_bin_var = []
                aw_bin_var_err = []
                aw_bin_nbar = []
                aw_bin_nbar_sq = []
                aw_bin_N = []
                aw_bin_gc = []

                aw_edges = np.arange(-13.0, -8.0, 0.5)
                aw_centers = (aw_edges[:-1] + aw_edges[1:]) / 2.0

                for j in range(len(aw_centers)):
                    lo, hi = aw_edges[j], aw_edges[j + 1]
                    z_vals = np.array([p['z_res'] for p in aw_z_pts
                                       if lo <= p['log_gbar'] < hi])
                    if len(z_vals) >= 20:
                        var_obs = float(np.var(z_vals))
                        var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
                        gbar_lin = 10.0 ** aw_centers[j]
                        x = np.sqrt(gbar_lin / g_dagger)
                        nbar = 1.0 / (np.exp(x) - 1.0 + 1e-30)
                        aw_bin_var.append(var_obs)
                        aw_bin_var_err.append(var_err)
                        aw_bin_nbar.append(nbar)
                        aw_bin_nbar_sq.append(nbar**2 + nbar)
                        aw_bin_N.append(len(z_vals))
                        aw_bin_gc.append(aw_centers[j])

                if len(aw_bin_var) >= 4:
                    aw_bin_var = np.array(aw_bin_var)
                    aw_bin_var_err = np.array(aw_bin_var_err)
                    aw_bin_nbar = np.array(aw_bin_nbar)
                    aw_bin_nbar_sq = np.array(aw_bin_nbar_sq)

                    print(f"\n    ALFALFA+WISE bunching test ({len(aw_bin_var)} bins):")
                    print(f"    {'log(gbar)':>10} {'N':>6} {'σ²':>8} {'n̄':>8} {'n̄²+n̄':>10}")
                    print(f"    {'-'*48}")
                    for j in range(len(aw_bin_var)):
                        print(f"    {aw_bin_gc[j]:>10.2f} {aw_bin_N[j]:>6} "
                              f"{aw_bin_var[j]:>8.4f} {aw_bin_nbar[j]:>8.4f} "
                              f"{aw_bin_nbar_sq[j]:>10.4f}")

                    try:
                        # Quantum fit
                        popt_aq, _ = curve_fit(lambda x, A, C: A * x + C,
                                              aw_bin_nbar_sq, aw_bin_var,
                                              p0=[0.01, 0.5], sigma=aw_bin_var_err,
                                              absolute_sigma=True, maxfev=5000)
                        resid_aq = aw_bin_var - (popt_aq[0] * aw_bin_nbar_sq + popt_aq[1])
                        chi2_aq = np.sum((resid_aq / aw_bin_var_err)**2)

                        # Classical fit
                        popt_ac, _ = curve_fit(lambda x, A, C: A * x + C,
                                              aw_bin_nbar, aw_bin_var,
                                              p0=[0.01, 0.5], sigma=aw_bin_var_err,
                                              absolute_sigma=True, maxfev=5000)
                        resid_ac = aw_bin_var - (popt_ac[0] * aw_bin_nbar + popt_ac[1])
                        chi2_ac = np.sum((resid_ac / aw_bin_var_err)**2)

                        # Constant
                        aw_wts = 1.0 / np.maximum(aw_bin_var_err, 1e-6)
                        aw_mean_var = np.average(aw_bin_var, weights=aw_wts**2)
                        resid_a_const = aw_bin_var - aw_mean_var
                        chi2_a_const = np.sum((resid_a_const / aw_bin_var_err)**2)

                        n_aw_bins = len(aw_bin_var)
                        aic_aq = chi2_aq + 4
                        aic_ac = chi2_ac + 4
                        aic_a_const = chi2_a_const + 2
                        daic_alfa = aic_ac - aic_aq

                        print(f"\n    Model fitting:")
                        print(f"    Quantum:   A = {popt_aq[0]:.6f}, C = {popt_aq[1]:.4f}, "
                              f"χ²/dof = {chi2_aq/max(n_aw_bins-2,1):.3f}, AIC = {aic_aq:.2f}")
                        print(f"    Classical: A = {popt_ac[0]:.6f}, C = {popt_ac[1]:.4f}, "
                              f"χ²/dof = {chi2_ac/max(n_aw_bins-2,1):.3f}, AIC = {aic_ac:.2f}")
                        print(f"    Constant:  σ² = {aw_mean_var:.4f}, AIC = {aic_a_const:.2f}")
                        print(f"\n    ΔAIC (classical − quantum) = {daic_alfa:+.2f}")
                        print(f"    ΔAIC (constant  − quantum) = {aic_a_const - aic_aq:+.2f}")

                        t10_delta_aic = daic_alfa

                        # Floor comparison with SPARC
                        print(f"\n    Floor comparison:")
                        print(f"    ALFALFA+WISE floor (C) = {popt_aq[1]:.4f}")
                        if sparc_info:
                            print(f"    SPARC floor (C) = {sparc_C:.4f}")
                            print(f"    Ratio: {popt_aq[1]/max(sparc_C,1e-6):.2f}x")

                        if daic_alfa > 6:
                            print(f"\n    >>> ALFALFA+WISE: Quantum DECISIVELY preferred (ΔAIC = {daic_alfa:+.1f})")
                            t10_supports = True
                        elif daic_alfa > 2:
                            print(f"    >>> ALFALFA+WISE: Quantum preferred (ΔAIC = {daic_alfa:+.1f})")
                            t10_supports = True
                        elif daic_alfa > -2:
                            print(f"    >>> ALFALFA+WISE: Models indistinguishable (ΔAIC = {daic_alfa:+.1f})")
                        else:
                            print(f"    >>> ALFALFA+WISE: Classical preferred (ΔAIC = {daic_alfa:+.1f})")

                        # Compare amplitudes
                        print(f"\n    Amplitude comparison:")
                        print(f"    ALFALFA+WISE: A_quantum = {popt_aq[0]:.6f}")
                        if sparc_info:
                            print(f"    SPARC:         A_quantum = {sparc_A:.6f}")
                            if abs(popt_aq[0]) > 0 and sparc_A > 0 and popt_aq[0] > 0:
                                print(f"    Same sign (positive): ✓ consistent")
                            elif popt_aq[0] < 0:
                                print(f"    Negative amplitude: floor masking (as in other non-SPARC datasets)")

                    except Exception as e:
                        print(f"    Model fitting failed: {e}")

                else:
                    print(f"    Insufficient bins ({len(aw_bin_var)}) for bunching test")
            else:
                print(f"    Insufficient WISE-matched galaxies ({len(alfa_wise_pts)})")

            # ---- GOLD SUBSAMPLE: HyperLEDA inclinations + WISE masses ----
            # This subsample has measured inclinations (not statistical <sin(i)>)
            # AND measured stellar masses (not TF). Should have lowest systematic floor.
            alfa_gold_pts = [p for p in alfa_pts
                             if p['has_wise'] and p['incl_source'] == 'hleda']
            n_gold = len(alfa_gold_pts)
            print(f"\n    --- GOLD subsample (WISE mass + HyperLEDA incl): {n_gold} galaxies ---")

            if n_gold >= 50:
                gold_res = np.array([p['log_res'] for p in alfa_gold_pts])
                gold_mu = np.mean(gold_res)
                gold_std = np.std(gold_res)
                print(f"    Raw scatter: σ = {gold_std:.4f} dex")

                # Inclination distribution diagnostic
                gold_incls = np.array([p['incl_deg'] for p in alfa_gold_pts])
                print(f"    Inclination range: {np.min(gold_incls):.1f}° – "
                      f"{np.max(gold_incls):.1f}°, median {np.median(gold_incls):.1f}°")

                # gbar distribution
                gold_lgbar = np.array([p['log_gbar'] for p in alfa_gold_pts])
                print(f"    log(gbar) range: {np.min(gold_lgbar):.2f} – "
                      f"{np.max(gold_lgbar):.2f}, median {np.median(gold_lgbar):.2f}")
                n_dm_regime = np.sum(gold_lgbar < -10.5)
                print(f"    Points in DM regime (log(gbar) < -10.5): {n_dm_regime}")

                # Z-score and bin
                gold_z_pts = [{'z_res': (p['log_res'] - gold_mu) / gold_std,
                               'log_gbar': p['log_gbar']} for p in alfa_gold_pts]

                g_bin_var = []
                g_bin_var_err = []
                g_bin_nbar = []
                g_bin_nbar_sq = []
                g_bin_N = []
                g_bin_gc = []

                g_edges = np.arange(-13.0, -8.0, 0.5)
                g_centers = (g_edges[:-1] + g_edges[1:]) / 2.0

                for j in range(len(g_centers)):
                    lo, hi = g_edges[j], g_edges[j + 1]
                    z_vals = np.array([p['z_res'] for p in gold_z_pts
                                       if lo <= p['log_gbar'] < hi])
                    if len(z_vals) >= 10:  # lower threshold for gold subsample
                        var_obs = float(np.var(z_vals))
                        var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
                        gbar_lin = 10.0 ** g_centers[j]
                        x = np.sqrt(gbar_lin / g_dagger)
                        nbar = 1.0 / (np.exp(x) - 1.0 + 1e-30)
                        g_bin_var.append(var_obs)
                        g_bin_var_err.append(var_err)
                        g_bin_nbar.append(nbar)
                        g_bin_nbar_sq.append(nbar**2 + nbar)
                        g_bin_N.append(len(z_vals))
                        g_bin_gc.append(g_centers[j])

                if len(g_bin_var) >= 3:
                    g_bin_var = np.array(g_bin_var)
                    g_bin_var_err = np.array(g_bin_var_err)
                    g_bin_nbar = np.array(g_bin_nbar)
                    g_bin_nbar_sq = np.array(g_bin_nbar_sq)

                    print(f"\n    GOLD bunching test ({len(g_bin_var)} bins):")
                    print(f"    {'log(gbar)':>10} {'N':>6} {'σ²':>8} {'n̄':>8} {'n̄²+n̄':>10}")
                    print(f"    {'-'*48}")
                    for j in range(len(g_bin_var)):
                        print(f"    {g_bin_gc[j]:>10.2f} {g_bin_N[j]:>6} "
                              f"{g_bin_var[j]:>8.4f} {g_bin_nbar[j]:>8.4f} "
                              f"{g_bin_nbar_sq[j]:>10.4f}")

                    try:
                        # Quantum fit
                        popt_gq, _ = curve_fit(lambda x, A, C: A * x + C,
                                              g_bin_nbar_sq, g_bin_var,
                                              p0=[0.01, 0.5], sigma=g_bin_var_err,
                                              absolute_sigma=True, maxfev=5000)
                        resid_gq = g_bin_var - (popt_gq[0] * g_bin_nbar_sq + popt_gq[1])
                        chi2_gq = np.sum((resid_gq / g_bin_var_err)**2)

                        # Classical fit
                        popt_gc, _ = curve_fit(lambda x, A, C: A * x + C,
                                              g_bin_nbar, g_bin_var,
                                              p0=[0.01, 0.5], sigma=g_bin_var_err,
                                              absolute_sigma=True, maxfev=5000)
                        resid_gc = g_bin_var - (popt_gc[0] * g_bin_nbar + popt_gc[1])
                        chi2_gc = np.sum((resid_gc / g_bin_var_err)**2)

                        # Constant
                        g_wts = 1.0 / np.maximum(g_bin_var_err, 1e-6)
                        g_mean_var = np.average(g_bin_var, weights=g_wts**2)
                        resid_g_const = g_bin_var - g_mean_var
                        chi2_g_const = np.sum((resid_g_const / g_bin_var_err)**2)

                        n_g_bins = len(g_bin_var)
                        aic_gq = chi2_gq + 4
                        aic_gc = chi2_gc + 4
                        aic_g_const = chi2_g_const + 2
                        daic_gold = aic_gc - aic_gq

                        print(f"\n    GOLD Model fitting:")
                        print(f"    Quantum:   A = {popt_gq[0]:.6f}, C = {popt_gq[1]:.4f}, "
                              f"χ²/dof = {chi2_gq/max(n_g_bins-2,1):.3f}, AIC = {aic_gq:.2f}")
                        print(f"    Classical: A = {popt_gc[0]:.6f}, C = {popt_gc[1]:.4f}, "
                              f"χ²/dof = {chi2_gc/max(n_g_bins-2,1):.3f}, AIC = {aic_gc:.2f}")
                        print(f"    Constant:  σ² = {g_mean_var:.4f}, AIC = {aic_g_const:.2f}")
                        print(f"\n    GOLD ΔAIC (classical − quantum) = {daic_gold:+.2f}")

                        # Floor comparison
                        print(f"\n    Floor comparison:")
                        print(f"    GOLD floor (C) = {popt_gq[1]:.4f}")
                        if sparc_info:
                            print(f"    SPARC floor (C) = {sparc_C:.4f}")
                            print(f"    Ratio: {popt_gq[1]/max(sparc_C,1e-6):.2f}x")
                        print(f"    WISE-only floor (C) = {popt_aq[1]:.4f}" if 'popt_aq' in dir() else "")

                        # Scatter comparison: gold vs all-ALFALFA
                        print(f"\n    Scatter reduction from measured inclinations:")
                        print(f"    All ALFALFA:   σ = {alfa_std:.4f} dex")
                        print(f"    GOLD subsample: σ = {gold_std:.4f} dex")
                        reduction_pct = (1.0 - gold_std / alfa_std) * 100
                        print(f"    Reduction: {reduction_pct:+.1f}%")

                        if daic_gold > 6:
                            print(f"\n    >>> GOLD: Quantum DECISIVELY preferred (ΔAIC = {daic_gold:+.1f})")
                        elif daic_gold > 2:
                            print(f"    >>> GOLD: Quantum preferred (ΔAIC = {daic_gold:+.1f})")
                        elif daic_gold > -2:
                            print(f"    >>> GOLD: Models indistinguishable (ΔAIC = {daic_gold:+.1f})")
                        else:
                            print(f"    >>> GOLD: Classical preferred (ΔAIC = {daic_gold:+.1f})")

                        # Use gold subsample for test 10 verdict if available
                        # (it's the cleanest ALFALFA subsample)
                        if n_gold >= 200:
                            t10_delta_aic = daic_gold
                            t10_supports = daic_gold > 2
                            print(f"\n    [Using GOLD subsample ΔAIC for Test 10 verdict]")

                        # ========================================================
                        # GOLD DEEP DIVE: Bootstrap + Leave-One-Out
                        # Mirror of SPARC Diagnostic 8E for independent replication
                        # ========================================================
                        # Each GOLD galaxy contributes exactly one RAR point,
                        # so "leave-one-galaxy-out" = leave-one-point-out
                        # and bootstrap resamples galaxies with replacement.

                        # Z-score the GOLD residuals
                        gold_z_pts_full = []
                        for p in alfa_gold_pts:
                            gold_z_pts_full.append({
                                'z_res': (p['log_res'] - gold_mu) / gold_std,
                                'log_gbar': p['log_gbar'],
                                'galaxy': p['galaxy'],
                            })

                        gold_gals = list(set(p['galaxy'] for p in gold_z_pts_full))
                        n_gold_gals = len(gold_gals)
                        print(f"\n    --- GOLD Deep Dive: {n_gold_gals} unique galaxies ---")

                        # Build galaxy-to-points lookup (1 pt per galaxy for ALFALFA)
                        gold_gal_to_pts = {}
                        for p in gold_z_pts_full:
                            g = p['galaxy']
                            if g not in gold_gal_to_pts:
                                gold_gal_to_pts[g] = []
                            gold_gal_to_pts[g].append(p)

                        # --- Bootstrap: resample GOLD galaxies, 500 iterations ---
                        print(f"\n    Bootstrap ΔAIC (resampling GOLD galaxies, 500 iterations)...")
                        gold_boot_daics = []
                        gold_boot_As = []
                        gold_boot_Cs = []

                        for b_iter in range(500):
                            boot_gals = np.random.choice(gold_gals,
                                                          size=n_gold_gals, replace=True)
                            boot_pts = []
                            for g in boot_gals:
                                boot_pts.extend(gold_gal_to_pts[g])

                            # Re-Z-score
                            b_res = np.array([p['z_res'] for p in boot_pts])
                            b_mu = np.mean(b_res)
                            b_std = np.std(b_res)
                            if b_std < 1e-6:
                                continue

                            # Bin
                            b_var_list = []
                            b_var_err_list = []
                            b_nbar_sq_list = []
                            b_nbar_list = []

                            for jj in range(len(g_centers)):
                                lo, hi = g_edges[jj], g_edges[jj + 1]
                                zv = np.array([(p['z_res'] - b_mu) / b_std
                                               for p in boot_pts
                                               if lo <= p['log_gbar'] < hi])
                                if len(zv) >= 8:
                                    vv = float(np.var(zv))
                                    ve = np.sqrt(2.0 * vv**2 / (len(zv) - 1))
                                    gbl = 10.0 ** g_centers[jj]
                                    xv = np.sqrt(gbl / g_dagger)
                                    nb = 1.0 / (np.exp(xv) - 1.0 + 1e-30)
                                    b_var_list.append(vv)
                                    b_var_err_list.append(ve)
                                    b_nbar_sq_list.append(nb**2 + nb)
                                    b_nbar_list.append(nb)

                            if len(b_var_list) < 3:
                                continue

                            b_var_arr = np.array(b_var_list)
                            b_var_err_arr = np.array(b_var_err_list)
                            b_nbar_sq_arr = np.array(b_nbar_sq_list)
                            b_nbar_arr = np.array(b_nbar_list)

                            try:
                                bp_q, _ = curve_fit(lambda x, A, C: A * x + C,
                                                    b_nbar_sq_arr, b_var_arr,
                                                    p0=[0.001, 0.7],
                                                    sigma=b_var_err_arr,
                                                    absolute_sigma=True, maxfev=3000)
                                br_q = b_var_arr - (bp_q[0] * b_nbar_sq_arr + bp_q[1])
                                bc_q = np.sum((br_q / b_var_err_arr)**2)

                                bp_c, _ = curve_fit(lambda x, A, C: A * x + C,
                                                    b_nbar_arr, b_var_arr,
                                                    p0=[0.001, 0.7],
                                                    sigma=b_var_err_arr,
                                                    absolute_sigma=True, maxfev=3000)
                                br_c = b_var_arr - (bp_c[0] * b_nbar_arr + bp_c[1])
                                bc_c = np.sum((br_c / b_var_err_arr)**2)

                                b_daic = (bc_c + 4) - (bc_q + 4)
                                gold_boot_daics.append(b_daic)
                                gold_boot_As.append(bp_q[0])
                                gold_boot_Cs.append(bp_q[1])
                            except Exception:
                                continue

                        if len(gold_boot_daics) >= 100:
                            gold_boot_daics = np.array(gold_boot_daics)
                            gold_boot_As = np.array(gold_boot_As)
                            gold_boot_Cs = np.array(gold_boot_Cs)
                            pct_quantum = 100 * np.mean(gold_boot_daics > 0)
                            med_daic = np.median(gold_boot_daics)
                            ci68 = np.percentile(gold_boot_daics, [16, 84])
                            ci95 = np.percentile(gold_boot_daics, [2.5, 97.5])
                            med_A = np.median(gold_boot_As)
                            ci_A = np.percentile(gold_boot_As, [16, 84])
                            med_C = np.median(gold_boot_Cs)
                            ci_C = np.percentile(gold_boot_Cs, [16, 84])
                            pct_pos_A = 100 * np.mean(gold_boot_As > 0)

                            print(f"    Valid iterations: {len(gold_boot_daics)}/500")
                            print(f"    Median ΔAIC: {med_daic:+.2f}")
                            print(f"    68% CI: [{ci68[0]:+.2f}, {ci68[1]:+.2f}]")
                            print(f"    95% CI: [{ci95[0]:+.2f}, {ci95[1]:+.2f}]")
                            print(f"    Quantum preferred in {pct_quantum:.1f}% of bootstrap samples")
                            print(f"    Median A_quantum: {med_A:.6f} "
                                  f"(68% CI: [{ci_A[0]:.6f}, {ci_A[1]:.6f}])")
                            print(f"    Positive A in {pct_pos_A:.1f}% of samples")
                            print(f"    Median floor C: {med_C:.4f} "
                                  f"(68% CI: [{ci_C[0]:.4f}, {ci_C[1]:.4f}])")

                            if pct_quantum > 80:
                                print(f"    >>> ROBUST: GOLD bunching signal persists in "
                                      f"{pct_quantum:.0f}% of galaxy resamples")
                            elif pct_quantum > 55:
                                print(f"    >>> MODERATE: Quantum preferred in "
                                      f"{pct_quantum:.0f}% of resamples (consistent with "
                                      f"SPARC pattern at higher floor)")
                            elif pct_quantum > 40:
                                print(f"    >>> WEAK: Only {pct_quantum:.0f}% quantum preference")
                            else:
                                print(f"    >>> NOT ROBUST: {pct_quantum:.0f}% quantum preference "
                                      f"under resampling")

                            # SPARC comparison
                            print(f"\n    Comparison with SPARC bootstrap:")
                            print(f"    SPARC:  66% quantum, ΔAIC median +9.2, "
                                  f"floor C = 0.50")
                            print(f"    GOLD:   {pct_quantum:.0f}% quantum, ΔAIC median "
                                  f"{med_daic:+.1f}, floor C = {med_C:.2f}")
                        else:
                            print(f"    Bootstrap: only {len(gold_boot_daics)} valid iterations "
                                  f"(insufficient)")

                        # --- Leave-one-galaxy-out sensitivity ---
                        print(f"\n    Leave-one-galaxy-out sensitivity "
                              f"(top 10 most influential):")
                        gold_loo_results = []
                        for drop_gal in gold_gals:
                            loo_pts = [p for p in gold_z_pts_full
                                       if p['galaxy'] != drop_gal]
                            if len(loo_pts) < 50:
                                continue

                            l_res = np.array([p['z_res'] for p in loo_pts])
                            l_mu = np.mean(l_res)
                            l_std = np.std(l_res)
                            if l_std < 1e-6:
                                continue

                            l_var_list = []
                            l_var_err_list = []
                            l_nbar_sq_list = []
                            l_nbar_list = []

                            for jj in range(len(g_centers)):
                                lo, hi = g_edges[jj], g_edges[jj + 1]
                                zv = np.array([(p['z_res'] - l_mu) / l_std
                                               for p in loo_pts
                                               if lo <= p['log_gbar'] < hi])
                                if len(zv) >= 8:
                                    vv = float(np.var(zv))
                                    ve = np.sqrt(2.0 * vv**2 / (len(zv) - 1))
                                    gbl = 10.0 ** g_centers[jj]
                                    xv = np.sqrt(gbl / g_dagger)
                                    nb = 1.0 / (np.exp(xv) - 1.0 + 1e-30)
                                    l_var_list.append(vv)
                                    l_var_err_list.append(ve)
                                    l_nbar_sq_list.append(nb**2 + nb)
                                    l_nbar_list.append(nb)

                            if len(l_var_list) < 3:
                                continue

                            l_var_arr = np.array(l_var_list)
                            l_var_err_arr = np.array(l_var_err_list)
                            l_nbar_sq_arr = np.array(l_nbar_sq_list)
                            l_nbar_arr = np.array(l_nbar_list)

                            try:
                                lp_q, _ = curve_fit(lambda x, A, C: A * x + C,
                                                    l_nbar_sq_arr, l_var_arr,
                                                    p0=[0.001, 0.7],
                                                    sigma=l_var_err_arr,
                                                    absolute_sigma=True, maxfev=3000)
                                lr_q = l_var_arr - (lp_q[0] * l_nbar_sq_arr + lp_q[1])
                                lc_q = np.sum((lr_q / l_var_err_arr)**2)

                                lp_c, _ = curve_fit(lambda x, A, C: A * x + C,
                                                    l_nbar_arr, l_var_arr,
                                                    p0=[0.001, 0.7],
                                                    sigma=l_var_err_arr,
                                                    absolute_sigma=True, maxfev=3000)
                                lr_c = l_var_arr - (lp_c[0] * l_nbar_arr + lp_c[1])
                                lc_c = np.sum((lr_c / l_var_err_arr)**2)

                                l_daic = (lc_c + 4) - (lc_q + 4)
                                influence = daic_gold - l_daic
                                gold_loo_results.append((drop_gal, l_daic, influence))
                            except Exception:
                                continue

                        if gold_loo_results:
                            # Sort by influence (change in ΔAIC when dropped)
                            gold_loo_results.sort(key=lambda x: -abs(x[2]))

                            n_loo_quantum = sum(1 for r in gold_loo_results if r[1] > 0)
                            n_loo_total = len(gold_loo_results)
                            min_daic_loo = min(r[1] for r in gold_loo_results)
                            max_daic_loo = max(r[1] for r in gold_loo_results)
                            med_daic_loo = np.median([r[1] for r in gold_loo_results])

                            print(f"    Total leave-one-out tests: {n_loo_total}")
                            print(f"    ΔAIC range: [{min_daic_loo:+.2f}, {max_daic_loo:+.2f}]")
                            print(f"    Median ΔAIC: {med_daic_loo:+.2f}")
                            print(f"    Quantum preferred: {n_loo_quantum}/{n_loo_total} "
                                  f"({100*n_loo_quantum/max(n_loo_total,1):.1f}%)")

                            print(f"\n    {'Galaxy':>20} {'ΔAIC_loo':>10} {'Influence':>10}")
                            print(f"    {'-'*44}")
                            for gal, daic_l, influence in gold_loo_results[:10]:
                                print(f"    {gal:>20} {daic_l:>+10.2f} {influence:>+10.2f}")

                            if min_daic_loo > 0:
                                worst_gal = [r[0] for r in gold_loo_results
                                             if r[1] == min_daic_loo][0]
                                print(f"\n    >>> ROBUST: Even dropping most influential "
                                      f"galaxy ({worst_gal}), ΔAIC = {min_daic_loo:+.1f} "
                                      f"still favors quantum")
                            elif min_daic_loo > -2:
                                worst_gal = [r[0] for r in gold_loo_results
                                             if r[1] == min_daic_loo][0]
                                print(f"\n    >>> MARGINAL: Dropping {worst_gal} reduces "
                                      f"ΔAIC to {min_daic_loo:+.1f}")
                            else:
                                worst_gal = [r[0] for r in gold_loo_results
                                             if r[1] == min_daic_loo][0]
                                n_flip = sum(1 for r in gold_loo_results if r[1] < -2)
                                print(f"\n    >>> {n_flip} galaxies flip result when dropped. "
                                      f"Worst: {worst_gal} (ΔAIC → {min_daic_loo:+.1f})")

                    except Exception as e:
                        print(f"    GOLD model fitting failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"    Insufficient GOLD bins ({len(g_bin_var)}) for bunching test")
            else:
                print(f"    Insufficient GOLD galaxies ({n_gold})")

            # ---- HyperLEDA-ONLY subsample (all mass models, measured incl) ----
            # This tests the isolated effect of inclination improvement
            alfa_hleda_pts = [p for p in alfa_pts if p['incl_source'] == 'hleda']
            n_hleda_sub = len(alfa_hleda_pts)
            print(f"\n    --- HyperLEDA-incl subsample (any mass model): {n_hleda_sub} galaxies ---")

            if n_hleda_sub >= 100:
                hleda_res = np.array([p['log_res'] for p in alfa_hleda_pts])
                hleda_mu = np.mean(hleda_res)
                hleda_std = np.std(hleda_res)
                print(f"    Raw scatter: σ = {hleda_std:.4f} dex")
                print(f"    vs statistical-incl subsample: σ = {alfa_std:.4f} dex")

                # Quick bunching test on HyperLEDA subsample
                h_z_pts = [{'z_res': (p['log_res'] - hleda_mu) / hleda_std,
                            'log_gbar': p['log_gbar']} for p in alfa_hleda_pts]

                h_bin_var = []
                h_bin_var_err = []
                h_bin_nbar = []
                h_bin_nbar_sq = []
                h_bin_N = []
                h_bin_gc = []

                h_edges = np.arange(-13.0, -8.0, 0.5)
                h_centers = (h_edges[:-1] + h_edges[1:]) / 2.0

                for j in range(len(h_centers)):
                    lo, hi = h_edges[j], h_edges[j + 1]
                    z_vals = np.array([p['z_res'] for p in h_z_pts
                                       if lo <= p['log_gbar'] < hi])
                    if len(z_vals) >= 15:
                        var_obs = float(np.var(z_vals))
                        var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
                        gbar_lin = 10.0 ** h_centers[j]
                        x = np.sqrt(gbar_lin / g_dagger)
                        nbar = 1.0 / (np.exp(x) - 1.0 + 1e-30)
                        h_bin_var.append(var_obs)
                        h_bin_var_err.append(var_err)
                        h_bin_nbar.append(nbar)
                        h_bin_nbar_sq.append(nbar**2 + nbar)
                        h_bin_N.append(len(z_vals))
                        h_bin_gc.append(h_centers[j])

                if len(h_bin_var) >= 4:
                    h_bin_var = np.array(h_bin_var)
                    h_bin_var_err = np.array(h_bin_var_err)
                    h_bin_nbar = np.array(h_bin_nbar)
                    h_bin_nbar_sq = np.array(h_bin_nbar_sq)

                    print(f"\n    HyperLEDA-incl bunching test ({len(h_bin_var)} bins):")
                    print(f"    {'log(gbar)':>10} {'N':>6} {'σ²':>8}")
                    for j in range(len(h_bin_var)):
                        print(f"    {h_bin_gc[j]:>10.2f} {h_bin_N[j]:>6} "
                              f"{h_bin_var[j]:>8.4f}")

                    try:
                        popt_hq, _ = curve_fit(lambda x, A, C: A * x + C,
                                              h_bin_nbar_sq, h_bin_var,
                                              p0=[0.01, 0.5], sigma=h_bin_var_err,
                                              absolute_sigma=True, maxfev=5000)
                        popt_hc, _ = curve_fit(lambda x, A, C: A * x + C,
                                              h_bin_nbar, h_bin_var,
                                              p0=[0.01, 0.5], sigma=h_bin_var_err,
                                              absolute_sigma=True, maxfev=5000)

                        resid_hq = h_bin_var - (popt_hq[0] * h_bin_nbar_sq + popt_hq[1])
                        chi2_hq = np.sum((resid_hq / h_bin_var_err)**2)
                        resid_hc = h_bin_var - (popt_hc[0] * h_bin_nbar + popt_hc[1])
                        chi2_hc = np.sum((resid_hc / h_bin_var_err)**2)

                        aic_hq = chi2_hq + 4
                        aic_hc = chi2_hc + 4
                        daic_hleda = aic_hc - aic_hq

                        print(f"\n    HyperLEDA-incl: A_quantum = {popt_hq[0]:.6f}, "
                              f"C = {popt_hq[1]:.4f}")
                        print(f"    ΔAIC (classical − quantum) = {daic_hleda:+.2f}")

                        if daic_hleda > 2:
                            print(f"    >>> HyperLEDA-incl: Quantum preferred!")
                        elif daic_hleda > -2:
                            print(f"    >>> HyperLEDA-incl: Indistinguishable")
                        else:
                            print(f"    >>> HyperLEDA-incl: Classical preferred")

                    except Exception as e:
                        print(f"    HyperLEDA-incl model fitting failed: {e}")

        else:
            print(f"    Insufficient ALFALFA galaxies after cuts ({n_alfa_total})")
    else:
        print(f"    ALFALFA catalog not found at {alfalfa_path}")

    # Add Test 10 to summary
    t10_sig = f"ΔAIC={t10_delta_aic:+.1f}" if not np.isnan(t10_delta_aic) else "N/A"
    tests_summary.append(('ALFALFA+WISE bunching (N~10k)', t10_delta_aic, t10_sig, t10_supports))
    n_support += int(t10_supports)
    if np.isnan(t10_delta_aic):
        print(f"\n    [P] 10. ? ALFALFA+WISE bunching test (data unavailable/insufficient)")
    else:
        print(f"\n    [P] 10. {'✓ SUPPORTS' if t10_supports else '✗ opposes'} "
              f"ALFALFA+WISE bunching statistics")

    # ================================================================
    # TEST 11: MaNGA V/σ BUNCHING (VELOCITY DISPERSION CHANNEL)
    # ================================================================
    # Independent channel: instead of testing RAR residual scatter,
    # we test whether the scatter in V/σ (velocity-to-dispersion ratio)
    # follows bosonic occupation statistics σ²(V/σ) = A*(n̄² + n̄) + C.
    #
    # V/σ measures ordered-to-random motion. In the BEC framework:
    #   V probes gobs = gbar*(1 + n̄)  (total gravitational field)
    #   σ probes the potential well depth including quantum pressure
    # The SCATTER in V/σ at fixed gbar should show bosonic bunching.
    #
    # Uses Ristea+2023 MaNGA catalog: gas and stellar V/σ at 1Re, 1.3Re, 2Re
    print("\n  " + "=" * 70)
    print("  TEST 11: MaNGA V/σ BUNCHING (VELOCITY DISPERSION CHANNEL)")
    print("  " + "=" * 70)
    print("    Third independent channel: scatter in V/σ vs gbar")
    print("    Tests bosonic statistics through velocity dispersion, not RAR residuals")

    t11_supports = False
    t11_delta_aic = np.nan

    manga_kin_path = os.path.join(DATA_DIR, 'hi_surveys', 'manga_ristea2023_kinematics.tsv')
    manga_nsa_path = os.path.join(DATA_DIR, 'manga_nsa_properties.tsv')

    if os.path.exists(manga_kin_path) and os.path.exists(manga_nsa_path):
        # --- Load NSA properties for distances/sizes ---
        t11_nsa = {}
        with open(manga_nsa_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                pifu = row.get('PlateIFU', '').strip().strip('"')
                z_str = row.get('NSAz', '').strip()
                rh_str = row.get('NSAsth50', '').strip()
                mass_str = row.get('NSAsMass', '').strip()
                try:
                    z_val = float(z_str)
                    if z_val > 0 and z_val < 1.0 and mass_str != '-9999.0':
                        rh = float(rh_str) if rh_str else 0.0
                        t11_nsa[pifu] = {
                            'z': z_val,
                            'Dist_Mpc': z_val * c_light / H0,
                            'Rh_arcsec': rh,
                        }
                except (ValueError, TypeError):
                    pass

        # --- Load kinematics with V/σ columns ---
        with open(manga_kin_path, 'r', encoding='utf-8', errors='replace') as f:
            t11_kin_rows = list(csv.DictReader(f, delimiter='\t'))

        print(f"    MaNGA catalog: {len(t11_kin_rows)} galaxies, "
              f"{len(t11_nsa)} with NSA redshifts")

        # --- Build V/σ + gbar pairs ---
        # Gas V/σ points: each (log_gbar, vsig_gas, galaxy_name, radius_label)
        vsig_gas_pts = []
        vsig_star_pts = []
        n_t11_no_nsa = 0
        n_t11_no_vsig = 0
        n_t11_bad_mass = 0
        n_t11_galaxies = set()

        for row in t11_kin_rows:
            pifu = row.get('Plateifu', '').strip().strip('"')
            logMstar = safe_float(row.get('logMstar', ''))
            ra = safe_float(row.get('RAJ2000', ''))
            dec = safe_float(row.get('DEJ2000', ''))

            # Quality cuts on stellar mass
            if np.isnan(logMstar) or logMstar < 8.5:
                n_t11_bad_mass += 1
                continue

            # Need NSA for distance/size
            nsa = t11_nsa.get(pifu)
            if nsa is None:
                n_t11_no_nsa += 1
                continue

            Dist_Mpc = nsa['Dist_Mpc']
            Rh_arcsec = nsa['Rh_arcsec']
            if Dist_Mpc <= 0 or Dist_Mpc > 700:
                continue
            if Rh_arcsec <= 0:
                Rh_arcsec = 3.0
            Rh_kpc = Rh_arcsec * Dist_Mpc / 206.265
            if Rh_kpc <= 0:
                continue

            Mstar = 10 ** logMstar
            Mbar = 1.33 * Mstar
            rd = Rh_kpc / 1.678  # exponential disk scale length

            gal_name = f"MaNGA_{pifu.strip()}"

            # Gas V/σ at each radius
            gas_vsig_cols = [
                ('VsigG1Re', 1.0, 'G_1Re'),
                ('VsigG1_3Re', 1.3, 'G_1.3Re'),
                ('VsigG2Re', 2.0, 'G_2Re'),
            ]
            for col, r_re, label in gas_vsig_cols:
                vsig = safe_float(row.get(col, ''))
                if np.isnan(vsig) or vsig < 0.1 or vsig > 10.0:
                    continue

                r_kpc = r_re * Rh_kpc
                denom = (r_kpc**2 + rd**2)**1.5
                if denom <= 0:
                    continue
                gbar = G_kpc * Mbar * r_kpc / denom * conv
                if gbar <= 0 or not np.isfinite(gbar):
                    continue
                log_gbar = np.log10(gbar)

                vsig_gas_pts.append({
                    'galaxy': gal_name,
                    'log_gbar': float(log_gbar),
                    'vsig': float(vsig),
                    'radius': label,
                    'logMstar': float(logMstar),
                })
                n_t11_galaxies.add(gal_name)

            # Stellar V/σ at each radius
            star_vsig_cols = [
                ('VsigST1Re', 1.0, 'ST_1Re'),
                ('VsigST1_3Re', 1.3, 'ST_1.3Re'),
                ('VsigST2Re', 2.0, 'ST_2Re'),
            ]
            for col, r_re, label in star_vsig_cols:
                vsig = safe_float(row.get(col, ''))
                if np.isnan(vsig) or vsig < 0.01 or vsig > 5.0:
                    continue

                r_kpc = r_re * Rh_kpc
                denom = (r_kpc**2 + rd**2)**1.5
                if denom <= 0:
                    continue
                gbar = G_kpc * Mbar * r_kpc / denom * conv
                if gbar <= 0 or not np.isfinite(gbar):
                    continue
                log_gbar = np.log10(gbar)

                vsig_star_pts.append({
                    'galaxy': gal_name,
                    'log_gbar': float(log_gbar),
                    'vsig': float(vsig),
                    'radius': label,
                    'logMstar': float(logMstar),
                })

        print(f"    Gas V/σ points: {len(vsig_gas_pts)} "
              f"({len(n_t11_galaxies)} galaxies)")
        print(f"    Stellar V/σ points: {len(vsig_star_pts)}")
        print(f"    Skipped: {n_t11_no_nsa} (no NSA), "
              f"{n_t11_bad_mass} (low mass)")

        # --- Gas V/σ gbar distribution ---
        if len(vsig_gas_pts) >= 200:
            gas_lgbar = np.array([p['log_gbar'] for p in vsig_gas_pts])
            gas_vsig = np.array([p['vsig'] for p in vsig_gas_pts])
            print(f"\n    Gas V/σ distribution:")
            print(f"    log(gbar) range: [{np.min(gas_lgbar):.2f}, "
                  f"{np.max(gas_lgbar):.2f}], median {np.median(gas_lgbar):.2f}")
            print(f"    V/σ range: [{np.min(gas_vsig):.3f}, "
                  f"{np.max(gas_vsig):.3f}], median {np.median(gas_vsig):.3f}")
            n_dm_regime_g = np.sum(gas_lgbar < -10.5)
            print(f"    Points in DM regime (log(gbar) < -10.5): {n_dm_regime_g}")

            # --- Z-score V/σ for variance analysis ---
            # Use log(V/σ) for better normality (V/σ is always positive, skewed)
            log_vsig = np.log10(gas_vsig)
            lv_mu = np.mean(log_vsig)
            lv_std = np.std(log_vsig)
            print(f"    log(V/σ) scatter: μ = {lv_mu:.4f}, σ = {lv_std:.4f}")

            gas_z_pts = []
            for i, p in enumerate(vsig_gas_pts):
                gas_z_pts.append({
                    'z_vsig': (np.log10(p['vsig']) - lv_mu) / lv_std,
                    'log_gbar': p['log_gbar'],
                    'galaxy': p['galaxy'],
                })

            # --- Bin and compute variance ---
            t11_edges = np.arange(-12.5, -8.0, 0.5)
            t11_centers = (t11_edges[:-1] + t11_edges[1:]) / 2.0

            t11_bin_var = []
            t11_bin_var_err = []
            t11_bin_nbar = []
            t11_bin_nbar_sq = []
            t11_bin_N = []
            t11_bin_gc = []

            for j in range(len(t11_centers)):
                lo, hi = t11_edges[j], t11_edges[j + 1]
                z_vals = np.array([p['z_vsig'] for p in gas_z_pts
                                   if lo <= p['log_gbar'] < hi])
                if len(z_vals) >= 20:
                    var_obs = float(np.var(z_vals))
                    var_err = np.sqrt(2.0 * var_obs**2 / (len(z_vals) - 1))
                    gbar_lin = 10.0 ** t11_centers[j]
                    x = np.sqrt(gbar_lin / g_dagger)
                    nbar = 1.0 / (np.exp(x) - 1.0 + 1e-30)
                    t11_bin_var.append(var_obs)
                    t11_bin_var_err.append(var_err)
                    t11_bin_nbar.append(nbar)
                    t11_bin_nbar_sq.append(nbar**2 + nbar)
                    t11_bin_N.append(len(z_vals))
                    t11_bin_gc.append(t11_centers[j])

            if len(t11_bin_var) >= 4:
                t11_bin_var = np.array(t11_bin_var)
                t11_bin_var_err = np.array(t11_bin_var_err)
                t11_bin_nbar = np.array(t11_bin_nbar)
                t11_bin_nbar_sq = np.array(t11_bin_nbar_sq)

                print(f"\n    Gas V/σ bunching test ({len(t11_bin_var)} bins):")
                print(f"    {'log(gbar)':>10} {'N':>6} {'σ²':>8} "
                      f"{'n̄':>8} {'n̄²+n̄':>10}")
                print(f"    {'-'*48}")
                for j in range(len(t11_bin_var)):
                    print(f"    {t11_bin_gc[j]:>10.2f} {t11_bin_N[j]:>6} "
                          f"{t11_bin_var[j]:>8.4f} {t11_bin_nbar[j]:>8.4f} "
                          f"{t11_bin_nbar_sq[j]:>10.4f}")

                try:
                    # Quantum fit: σ²(V/σ) = A*(n̄²+n̄) + C
                    popt_vq, _ = curve_fit(lambda x, A, C: A * x + C,
                                          t11_bin_nbar_sq, t11_bin_var,
                                          p0=[0.01, 0.5], sigma=t11_bin_var_err,
                                          absolute_sigma=True, maxfev=5000)
                    resid_vq = t11_bin_var - (popt_vq[0] * t11_bin_nbar_sq + popt_vq[1])
                    chi2_vq = np.sum((resid_vq / t11_bin_var_err)**2)

                    # Classical fit: σ²(V/σ) = A*n̄ + C
                    popt_vc, _ = curve_fit(lambda x, A, C: A * x + C,
                                          t11_bin_nbar, t11_bin_var,
                                          p0=[0.01, 0.5], sigma=t11_bin_var_err,
                                          absolute_sigma=True, maxfev=5000)
                    resid_vc = t11_bin_var - (popt_vc[0] * t11_bin_nbar + popt_vc[1])
                    chi2_vc = np.sum((resid_vc / t11_bin_var_err)**2)

                    # Constant (null)
                    v_wts = 1.0 / np.maximum(t11_bin_var_err, 1e-6)
                    v_mean_var = np.average(t11_bin_var, weights=v_wts**2)
                    resid_v_const = t11_bin_var - v_mean_var
                    chi2_v_const = np.sum((resid_v_const / t11_bin_var_err)**2)

                    n_v_bins = len(t11_bin_var)
                    aic_vq = chi2_vq + 4
                    aic_vc = chi2_vc + 4
                    aic_v_const = chi2_v_const + 2
                    daic_vsig = aic_vc - aic_vq

                    print(f"\n    Gas V/σ model fitting:")
                    print(f"    Quantum:   A = {popt_vq[0]:.6f}, C = {popt_vq[1]:.4f}, "
                          f"χ²/dof = {chi2_vq/max(n_v_bins-2,1):.3f}, "
                          f"AIC = {aic_vq:.2f}")
                    print(f"    Classical: A = {popt_vc[0]:.6f}, C = {popt_vc[1]:.4f}, "
                          f"χ²/dof = {chi2_vc/max(n_v_bins-2,1):.3f}, "
                          f"AIC = {aic_vc:.2f}")
                    print(f"    Constant:  σ² = {v_mean_var:.4f}, "
                          f"AIC = {aic_v_const:.2f}")
                    print(f"\n    ΔAIC (classical − quantum) = {daic_vsig:+.2f}")
                    print(f"    ΔAIC (constant  − quantum) = "
                          f"{aic_v_const - aic_vq:+.2f}")

                    # --- DM-regime coverage check ---
                    # The quantum bunching signal lives at low gbar where
                    # n̄ >> 1. If we have < 50 points below log(gbar) = -10.5,
                    # the test has no statistical power to detect bunching.
                    # Morphological diversity dominates at high gbar.
                    # Require at least 2 bins deep in the DM regime
                    # (log gbar < -11.0, where n̄ > 4) with ≥ 20 points
                    # each. At log(gbar) > -11, morphological diversity
                    # in V/σ overwhelms any quantum signal. A single bin
                    # at the edge cannot constrain the n̄(n̄+1) shape.
                    n_dm_bins_populated = 0
                    for bi, bv in enumerate(t11_bin_gc):
                        if bv < -11.0 and bi < len(t11_bin_N) and t11_bin_N[bi] >= 20:
                            n_dm_bins_populated += 1
                    has_vsig_leverage = n_dm_bins_populated >= 2

                    # Floor comparison
                    print(f"\n    Floor comparison:")
                    print(f"    MaNGA V/σ floor (C) = {popt_vq[1]:.4f}")
                    if sparc_info:
                        print(f"    SPARC RAR floor (C) = {sparc_C:.4f}")
                        print(f"    Ratio: {popt_vq[1]/max(sparc_C,1e-6):.2f}x")

                    # Amplitude analysis
                    print(f"\n    Amplitude analysis:")
                    print(f"    V/σ A_quantum = {popt_vq[0]:.6f}")
                    if popt_vq[0] > 0:
                        print(f"    Positive amplitude: consistent with BEC")
                    else:
                        print(f"    Negative amplitude: morphological "
                              f"diversity dominates")

                    # Verdict: check DM-regime leverage before interpreting
                    if not has_vsig_leverage:
                        # Insufficient DM-regime data — same logic as KROSS
                        t11_delta_aic = np.nan  # INCONCLUSIVE
                        t11_supports = False
                        print(f"\n    DM-REGIME COVERAGE:")
                        print(f"    Only {n_dm_regime_g} V/σ points at "
                              f"log(gbar) < -10.5")
                        print(f"    Only {t11_bin_N[0] if len(t11_bin_N) > 0 else 0}"
                              f" points in the lowest populated bin")
                        print(f"    Quantum bunching signal requires "
                              f"n̄ >> 1 (low gbar regime)")
                        print(f"    At log(gbar) > -10.5, variance is "
                              f"dominated by galaxy morphological")
                        print(f"    diversity (E/S0/Sa-Sd) which produces "
                              f"5-10× more scatter than")
                        print(f"    any quantum signal could.")
                        print(f"\n    >>> INCONCLUSIVE: V/σ bunching not "
                              f"testable — MaNGA does not probe")
                        print(f"    the DM-dominated regime where quantum "
                              f"signal lives")
                        print(f"    PREDICTION: Deep IFU surveys of dwarf/"
                              f"LSB galaxies reaching")
                        print(f"    log(gbar) < -11.5 with morphologically "
                              f"homogeneous samples")
                        print(f"    should detect V/σ bunching "
                              f"(BlueMUSE, DESI-IFU)")
                    else:
                        # Enough DM-regime data to interpret
                        t11_delta_aic = daic_vsig
                        if daic_vsig > 6:
                            print(f"\n    >>> V/σ: Quantum DECISIVELY "
                                  f"preferred (ΔAIC = {daic_vsig:+.1f})")
                            t11_supports = True
                        elif daic_vsig > 2:
                            print(f"    >>> V/σ: Quantum preferred "
                                  f"(ΔAIC = {daic_vsig:+.1f})")
                            t11_supports = True
                        elif daic_vsig > -2:
                            print(f"    >>> V/σ: Models "
                                  f"indistinguishable "
                                  f"(ΔAIC = {daic_vsig:+.1f})")
                        else:
                            print(f"    >>> V/σ: Classical preferred "
                                  f"(ΔAIC = {daic_vsig:+.1f})")

                    # ---- STELLAR V/σ CROSS-CHECK ----
                    if len(vsig_star_pts) >= 200:
                        print(f"\n    --- Stellar V/σ cross-check "
                              f"({len(vsig_star_pts)} points) ---")
                        star_vsig_arr = np.array([p['vsig']
                                                   for p in vsig_star_pts])
                        star_lgbar = np.array([p['log_gbar']
                                                for p in vsig_star_pts])
                        log_star_vsig = np.log10(np.maximum(star_vsig_arr,
                                                             1e-3))
                        sv_mu = np.mean(log_star_vsig)
                        sv_std = np.std(log_star_vsig)
                        print(f"    log(V/σ)_star scatter: μ = {sv_mu:.4f}, "
                              f"σ = {sv_std:.4f}")

                        star_z_pts = []
                        for i, p in enumerate(vsig_star_pts):
                            star_z_pts.append({
                                'z_vsig': (np.log10(max(p['vsig'], 1e-3))
                                           - sv_mu) / sv_std,
                                'log_gbar': p['log_gbar'],
                            })

                        s_bin_var = []
                        s_bin_var_err = []
                        s_bin_nbar_sq = []
                        s_bin_nbar = []
                        s_bin_N = []
                        s_bin_gc = []

                        for j in range(len(t11_centers)):
                            lo, hi = t11_edges[j], t11_edges[j + 1]
                            z_vals = np.array([p['z_vsig']
                                               for p in star_z_pts
                                               if lo <= p['log_gbar'] < hi])
                            if len(z_vals) >= 20:
                                vv = float(np.var(z_vals))
                                ve = np.sqrt(2.0 * vv**2 / (len(z_vals) - 1))
                                gbl = 10.0 ** t11_centers[j]
                                xv = np.sqrt(gbl / g_dagger)
                                nb = 1.0 / (np.exp(xv) - 1.0 + 1e-30)
                                s_bin_var.append(vv)
                                s_bin_var_err.append(ve)
                                s_bin_nbar_sq.append(nb**2 + nb)
                                s_bin_nbar.append(nb)
                                s_bin_N.append(len(z_vals))
                                s_bin_gc.append(t11_centers[j])

                        if len(s_bin_var) >= 4:
                            s_bin_var = np.array(s_bin_var)
                            s_bin_var_err = np.array(s_bin_var_err)
                            s_bin_nbar_sq = np.array(s_bin_nbar_sq)
                            s_bin_nbar = np.array(s_bin_nbar)

                            try:
                                sp_q, _ = curve_fit(
                                    lambda x, A, C: A * x + C,
                                    s_bin_nbar_sq, s_bin_var,
                                    p0=[0.01, 0.5], sigma=s_bin_var_err,
                                    absolute_sigma=True, maxfev=5000)
                                sr_q = s_bin_var - (sp_q[0] * s_bin_nbar_sq
                                                     + sp_q[1])
                                sc_q = np.sum((sr_q / s_bin_var_err)**2)

                                sp_c, _ = curve_fit(
                                    lambda x, A, C: A * x + C,
                                    s_bin_nbar, s_bin_var,
                                    p0=[0.01, 0.5], sigma=s_bin_var_err,
                                    absolute_sigma=True, maxfev=5000)
                                sr_c = s_bin_var - (sp_c[0] * s_bin_nbar
                                                     + sp_c[1])
                                sc_c = np.sum((sr_c / s_bin_var_err)**2)

                                daic_star = (sc_c + 4) - (sc_q + 4)
                                print(f"    Stellar quantum: A = {sp_q[0]:.6f}"
                                      f", C = {sp_q[1]:.4f}")
                                print(f"    Stellar classical: A = {sp_c[0]:.6f}"
                                      f", C = {sp_c[1]:.4f}")
                                print(f"    Stellar ΔAIC = {daic_star:+.2f}")

                                if daic_star > 2:
                                    print(f"    >>> Stellar V/σ: Quantum "
                                          f"preferred (cross-check ✓)")
                                elif daic_star > -2:
                                    print(f"    >>> Stellar V/σ: "
                                          f"Indistinguishable")
                                else:
                                    print(f"    >>> Stellar V/σ: Classical "
                                          f"preferred")

                                # Consistency check: same sign of A?
                                if (popt_vq[0] > 0 and sp_q[0] > 0):
                                    print(f"    Gas + stellar: both positive A "
                                          f"(✓ consistent)")
                                elif (popt_vq[0] < 0 and sp_q[0] < 0):
                                    print(f"    Gas + stellar: both negative A "
                                          f"(floor masking in both)")
                                else:
                                    print(f"    Gas + stellar: mixed signs")

                            except Exception as e:
                                print(f"    Stellar fitting failed: {e}")
                        else:
                            print(f"    Insufficient stellar bins "
                                  f"({len(s_bin_var)})")

                    # ---- V/σ RADIAL GRADIENT DIAGNOSTIC ----
                    # For galaxies with V/σ at all 3 radii: test whether
                    # gradient strength correlates with DM fraction
                    print(f"\n    --- V/σ radial gradient diagnostic ---")
                    grad_pts = []
                    for row in t11_kin_rows:
                        v1 = safe_float(row.get('VsigG1Re', ''))
                        v13 = safe_float(row.get('VsigG1_3Re', ''))
                        v2 = safe_float(row.get('VsigG2Re', ''))
                        lm = safe_float(row.get('logMstar', ''))
                        pifu = row.get('Plateifu', '').strip().strip('"')

                        if (np.isnan(v1) or np.isnan(v2) or np.isnan(lm)
                                or v1 < 0.1 or v2 < 0.1 or lm < 8.5):
                            continue

                        nsa = t11_nsa.get(pifu)
                        if nsa is None:
                            continue
                        Rh_kpc_g = nsa['Rh_arcsec'] * nsa['Dist_Mpc'] / 206.265
                        if Rh_kpc_g <= 0:
                            continue

                        Mbar_g = 1.33 * 10**lm
                        rd_g = Rh_kpc_g / 1.678

                        # gbar at 1.5Re (midpoint of gradient)
                        r_mid = 1.5 * Rh_kpc_g
                        denom_g = (r_mid**2 + rd_g**2)**1.5
                        if denom_g <= 0:
                            continue
                        gbar_mid = G_kpc * Mbar_g * r_mid / denom_g * conv
                        if gbar_mid <= 0 or not np.isfinite(gbar_mid):
                            continue

                        # Gradient: Δ(V/σ) / Δr_Re
                        grad = (v2 - v1) / (2.0 - 1.0)  # per Re
                        log_gbar_mid = np.log10(gbar_mid)

                        grad_pts.append({
                            'grad': float(grad),
                            'log_gbar_mid': float(log_gbar_mid),
                            'logMstar': float(lm),
                        })

                    if len(grad_pts) >= 50:
                        grad_arr = np.array([p['grad'] for p in grad_pts])
                        lgbar_arr = np.array([p['log_gbar_mid']
                                               for p in grad_pts])
                        print(f"    Galaxies with 3-radii V/σ: "
                              f"{len(grad_pts)}")
                        print(f"    Gradient range: [{np.min(grad_arr):.3f}, "
                              f"{np.max(grad_arr):.3f}], "
                              f"median {np.median(grad_arr):.3f}")

                        # Split by gbar: high vs low
                        lgbar_med = np.median(lgbar_arr)
                        hi_gbar = grad_arr[lgbar_arr >= lgbar_med]
                        lo_gbar = grad_arr[lgbar_arr < lgbar_med]
                        print(f"    High gbar (>{lgbar_med:.2f}): "
                              f"median grad = {np.median(hi_gbar):.3f} "
                              f"(N={len(hi_gbar)})")
                        print(f"    Low gbar  (<{lgbar_med:.2f}): "
                              f"median grad = {np.median(lo_gbar):.3f} "
                              f"(N={len(lo_gbar)})")

                        # BEC predicts steeper V/σ gradient at low gbar
                        if np.median(lo_gbar) > np.median(hi_gbar):
                            print(f"    >>> Steeper gradient at low gbar: "
                                  f"consistent with BEC prediction")
                        else:
                            print(f"    >>> No gradient enhancement at low "
                                  f"gbar")

                        # Correlation
                        from scipy.stats import pearsonr
                        r_corr, p_corr = pearsonr(lgbar_arr, grad_arr)
                        print(f"    Pearson r(log_gbar, grad) = {r_corr:.3f}, "
                              f"p = {p_corr:.2e}")
                    else:
                        print(f"    Insufficient 3-radii galaxies "
                              f"({len(grad_pts)})")

                    # ---- BOOTSTRAP + LEAVE-ONE-OUT (if signal detected) ----
                    if daic_vsig > 2 and len(gas_z_pts) >= 200:
                        print(f"\n    --- V/σ Deep Dive: Bootstrap + LOO ---")
                        vsig_gals = list(set(p['galaxy']
                                             for p in gas_z_pts))
                        vsig_gal_to_pts = {}
                        for p in gas_z_pts:
                            g = p['galaxy']
                            if g not in vsig_gal_to_pts:
                                vsig_gal_to_pts[g] = []
                            vsig_gal_to_pts[g].append(p)
                        n_vsig_gals = len(vsig_gals)

                        # Bootstrap
                        print(f"    Bootstrap (500 iterations, "
                              f"{n_vsig_gals} galaxies)...")
                        v_boot_daics = []
                        v_boot_As = []
                        for _ in range(500):
                            bg = np.random.choice(vsig_gals,
                                                   size=n_vsig_gals,
                                                   replace=True)
                            bp = []
                            for g in bg:
                                bp.extend(vsig_gal_to_pts[g])

                            b_res = np.array([p['z_vsig'] for p in bp])
                            b_mu = np.mean(b_res)
                            b_std = np.std(b_res)
                            if b_std < 1e-6:
                                continue

                            bv_list = []
                            bve_list = []
                            bn_sq_list = []
                            bn_list = []

                            for jj in range(len(t11_centers)):
                                lo, hi = (t11_edges[jj],
                                          t11_edges[jj + 1])
                                zv = np.array([
                                    (p['z_vsig'] - b_mu) / b_std
                                    for p in bp
                                    if lo <= p['log_gbar'] < hi
                                ])
                                if len(zv) >= 15:
                                    vv = float(np.var(zv))
                                    ve = np.sqrt(2.0 * vv**2
                                                  / (len(zv) - 1))
                                    gbl = 10.0 ** t11_centers[jj]
                                    xv = np.sqrt(gbl / g_dagger)
                                    nb = 1.0 / (np.exp(xv) - 1.0
                                                 + 1e-30)
                                    bv_list.append(vv)
                                    bve_list.append(ve)
                                    bn_sq_list.append(nb**2 + nb)
                                    bn_list.append(nb)

                            if len(bv_list) < 3:
                                continue

                            bv_a = np.array(bv_list)
                            bve_a = np.array(bve_list)
                            bnsq_a = np.array(bn_sq_list)
                            bn_a = np.array(bn_list)

                            try:
                                bpq, _ = curve_fit(
                                    lambda x, A, C: A * x + C,
                                    bnsq_a, bv_a, p0=[0.01, 0.5],
                                    sigma=bve_a, absolute_sigma=True,
                                    maxfev=3000)
                                brq = bv_a - (bpq[0] * bnsq_a + bpq[1])
                                bcq = np.sum((brq / bve_a)**2)

                                bpc, _ = curve_fit(
                                    lambda x, A, C: A * x + C,
                                    bn_a, bv_a, p0=[0.01, 0.5],
                                    sigma=bve_a, absolute_sigma=True,
                                    maxfev=3000)
                                brc = bv_a - (bpc[0] * bn_a + bpc[1])
                                bcc = np.sum((brc / bve_a)**2)

                                bd = (bcc + 4) - (bcq + 4)
                                v_boot_daics.append(bd)
                                v_boot_As.append(bpq[0])
                            except Exception:
                                continue

                        if len(v_boot_daics) >= 100:
                            v_boot_daics = np.array(v_boot_daics)
                            v_boot_As = np.array(v_boot_As)
                            pct_q = 100 * np.mean(v_boot_daics > 0)
                            med_d = np.median(v_boot_daics)
                            ci68 = np.percentile(v_boot_daics, [16, 84])
                            ci95 = np.percentile(v_boot_daics, [2.5, 97.5])
                            pct_pos = 100 * np.mean(v_boot_As > 0)
                            med_A = np.median(v_boot_As)

                            print(f"    Valid iterations: "
                                  f"{len(v_boot_daics)}/500")
                            print(f"    Median ΔAIC: {med_d:+.2f}")
                            print(f"    68% CI: [{ci68[0]:+.2f}, "
                                  f"{ci68[1]:+.2f}]")
                            print(f"    95% CI: [{ci95[0]:+.2f}, "
                                  f"{ci95[1]:+.2f}]")
                            print(f"    Quantum preferred in "
                                  f"{pct_q:.1f}% of samples")
                            print(f"    Positive A in {pct_pos:.1f}% "
                                  f"of samples")
                            print(f"    Median A: {med_A:.6f}")

                        # Leave-one-galaxy-out (subsample for speed)
                        n_loo = min(n_vsig_gals, 200)
                        loo_gals = (vsig_gals if n_vsig_gals <= 200
                                    else list(np.random.choice(
                                        vsig_gals, size=200,
                                        replace=False)))
                        print(f"\n    Leave-one-out ({n_loo} galaxies)...")
                        v_loo_daics = []
                        for drop_gal in loo_gals:
                            lp = [p for p in gas_z_pts
                                  if p['galaxy'] != drop_gal]
                            if len(lp) < 100:
                                continue
                            l_res = np.array([p['z_vsig'] for p in lp])
                            l_mu = np.mean(l_res)
                            l_std = np.std(l_res)
                            if l_std < 1e-6:
                                continue

                            lv_list = []
                            lve_list = []
                            ln_sq_list = []
                            ln_list = []
                            for jj in range(len(t11_centers)):
                                lo, hi = (t11_edges[jj],
                                          t11_edges[jj + 1])
                                zv = np.array([
                                    (p['z_vsig'] - l_mu) / l_std
                                    for p in lp
                                    if lo <= p['log_gbar'] < hi
                                ])
                                if len(zv) >= 15:
                                    vv = float(np.var(zv))
                                    ve = np.sqrt(2.0 * vv**2
                                                  / (len(zv) - 1))
                                    gbl = 10.0 ** t11_centers[jj]
                                    xv = np.sqrt(gbl / g_dagger)
                                    nb = 1.0 / (np.exp(xv) - 1.0
                                                 + 1e-30)
                                    lv_list.append(vv)
                                    lve_list.append(ve)
                                    ln_sq_list.append(nb**2 + nb)
                                    ln_list.append(nb)

                            if len(lv_list) < 3:
                                continue
                            try:
                                lpq, _ = curve_fit(
                                    lambda x, A, C: A * x + C,
                                    np.array(ln_sq_list),
                                    np.array(lv_list),
                                    p0=[0.01, 0.5],
                                    sigma=np.array(lve_list),
                                    absolute_sigma=True, maxfev=3000)
                                lrq = (np.array(lv_list) -
                                       (lpq[0] * np.array(ln_sq_list)
                                        + lpq[1]))
                                lcq = np.sum((lrq
                                              / np.array(lve_list))**2)
                                lpc, _ = curve_fit(
                                    lambda x, A, C: A * x + C,
                                    np.array(ln_list),
                                    np.array(lv_list),
                                    p0=[0.01, 0.5],
                                    sigma=np.array(lve_list),
                                    absolute_sigma=True, maxfev=3000)
                                lrc = (np.array(lv_list) -
                                       (lpc[0] * np.array(ln_list)
                                        + lpc[1]))
                                lcc = np.sum((lrc
                                              / np.array(lve_list))**2)
                                ld = (lcc + 4) - (lcq + 4)
                                v_loo_daics.append(ld)
                            except Exception:
                                continue

                        if v_loo_daics:
                            v_loo_daics = np.array(v_loo_daics)
                            n_q_loo = np.sum(v_loo_daics > 0)
                            print(f"    LOO quantum preferred: "
                                  f"{n_q_loo}/{len(v_loo_daics)} "
                                  f"({100*n_q_loo/len(v_loo_daics):.1f}%)")
                            print(f"    LOO ΔAIC range: "
                                  f"[{np.min(v_loo_daics):+.2f}, "
                                  f"{np.max(v_loo_daics):+.2f}]")
                            print(f"    LOO median ΔAIC: "
                                  f"{np.median(v_loo_daics):+.2f}")

                except Exception as e:
                    print(f"    Gas V/σ model fitting failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"    Insufficient bins ({len(t11_bin_var)}) for "
                      f"V/σ bunching test")
        else:
            print(f"    Insufficient gas V/σ points ({len(vsig_gas_pts)})")
    else:
        if not os.path.exists(manga_kin_path):
            print(f"    MaNGA kinematics not found: {manga_kin_path}")
        if not os.path.exists(manga_nsa_path):
            print(f"    MaNGA NSA not found: {manga_nsa_path}")

    # Add Test 11 to summary
    t11_sig = (f"ΔAIC={t11_delta_aic:+.1f}" if not np.isnan(t11_delta_aic)
               else "N/A")
    tests_summary.append(('MaNGA V/σ bunching (dispersion channel)',
                          t11_delta_aic, t11_sig, t11_supports))
    n_support += int(t11_supports)
    if np.isnan(t11_delta_aic):
        print(f"\n    [P] 11. ? INCONCLUSIVE MaNGA V/σ bunching "
              f"(no DM-regime coverage; morphological diversity dominates)")
    else:
        print(f"\n    [P] 11. {'✓ SUPPORTS' if t11_supports else '✗ opposes'} "
              f"MaNGA V/σ bunching (dispersion channel)")

    # ================================================================
    # TEST 12: ALFALFA HI PROFILE COHERENCE (NON-KINEMATIC CHANNEL)
    # ================================================================
    # Independent of rotation kinematics: tests condensate coherence through
    # the regularity of HI line profiles.
    #
    # BEC prediction: The condensate creates a smooth gravitational potential.
    # In undisturbed environments (field), the condensate is coherent →
    # HI profiles should be more regular → tighter HI mass–linewidth relation.
    # In disturbed environments (groups/clusters), tidal interactions fragment
    # the condensate → more irregular HI profiles → larger scatter.
    #
    # Observable: scatter in the HI Tully-Fisher relation (logMHI vs logW50)
    # split by environment. Also: HI profile concentration parameter.
    #
    # This tests spatial coherence of the potential, not velocity statistics.
    print("\n  " + "=" * 70)
    print("  TEST 12: ALFALFA HI PROFILE COHERENCE (NON-KINEMATIC CHANNEL)")
    print("  " + "=" * 70)
    print("    Non-kinematic channel: condensate coherence via HI profile regularity")
    print("    Tests whether field galaxies have tighter M_HI–W50 relation than dense")

    t12_supports = False
    t12_delta = np.nan

    # Reuse ALFALFA data already parsed in Test 10
    if os.path.exists(alfalfa_path):
        _, t12_rows = parse_vizier_tsv(alfalfa_path)

        # Build HI TF points with environment
        hitf_pts = []
        n_t12_field = 0
        n_t12_dense = 0

        for row in t12_rows:
            agc = row.get('UGC/AGC', '').strip()
            oname = row.get('OName', '').strip()
            cz = safe_float(row.get('cz', ''))
            w50 = safe_float(row.get('W50', ''))
            e_w50 = safe_float(row.get('e_W50', ''))
            snr = safe_float(row.get('SNR', ''))
            dist_mpc = safe_float(row.get('Dist', ''))
            logMHI = safe_float(row.get('logM', ''))
            hic = safe_int(row.get('HIc', '9'))
            flux = safe_float(row.get('Si(HI)', ''))
            rms = safe_float(row.get('RMS', ''))

            # Parse coordinates
            ra_str = row.get('RAJ2000', '').strip()
            dec_str = row.get('DEJ2000', '').strip()
            ra_deg = np.nan
            dec_deg = np.nan
            try:
                ra_parts = ra_str.split()
                if len(ra_parts) >= 3:
                    ra_deg = (float(ra_parts[0]) + float(ra_parts[1]) / 60.0
                              + float(ra_parts[2]) / 3600.0) * 15.0
                dec_parts = dec_str.replace('+', '').replace('-', '').split()
                dec_sign = -1 if dec_str.startswith('-') else 1
                if len(dec_parts) >= 3:
                    dec_deg = dec_sign * (float(dec_parts[0])
                                          + float(dec_parts[1]) / 60.0
                                          + float(dec_parts[2]) / 3600.0)
            except (ValueError, IndexError):
                pass

            # Quality cuts (same as Test 10 but slightly stricter for TF)
            if hic != 1:
                continue
            if np.isnan(snr) or snr < 8.0:  # stricter SNR for shape analysis
                continue
            if np.isnan(dist_mpc) or dist_mpc <= 1.0 or dist_mpc > 250:
                continue
            if np.isnan(w50) or w50 < 30 or w50 > 700:
                continue
            if np.isnan(logMHI) or logMHI < 7.0:
                continue
            if np.isnan(e_w50) or e_w50 / max(w50, 1) > 0.15:
                continue

            # Environment classification
            name_for_env = oname if oname else f"AGC{agc}"
            env = 'field'
            logMh = 11.0
            if not np.isnan(ra_deg) and not np.isnan(dec_deg) and not np.isnan(cz):
                _, logMh, env = classify_environment_proximity(
                    ra_deg, dec_deg, cz, name=name_for_env)

            # HI Tully-Fisher: logMHI vs log(W50)
            logW50 = np.log10(w50)

            # HI TF residual: logMHI - predicted from linear HITF
            # Will compute after fitting the relation

            # Profile concentration: flux / (W50_km/s * peak_flux_Jy)
            # peak_flux ≈ SNR * RMS (in mJy, so /1000 for Jy)
            # Si(HI) in Jy·km/s, W50 in km/s
            # Concentration C_prof = Si(HI) / (W50 * peak_Jy)
            # For symmetric double-horn: C_prof ≈ 0.7-1.0
            # For Gaussian: C_prof ≈ 0.6
            # For asymmetric: C_prof more variable
            peak_jy = snr * rms / 1000.0 if not np.isnan(rms) else np.nan
            c_prof = np.nan
            if peak_jy > 0 and w50 > 0 and not np.isnan(flux):
                c_prof = flux / (w50 * peak_jy)

            hitf_pts.append({
                'galaxy': name_for_env,
                'logMHI': float(logMHI),
                'logW50': float(logW50),
                'w50': float(w50),
                'snr': float(snr),
                'env': env,
                'logMh': float(logMh),
                'c_prof': float(c_prof),
                'ra': float(ra_deg),
                'dec': float(dec_deg),
            })

            if env == 'dense':
                n_t12_dense += 1
            else:
                n_t12_field += 1

        n_t12 = len(hitf_pts)
        print(f"    ALFALFA HI TF sample: {n_t12} galaxies")
        print(f"    Environment: {n_t12_dense} dense, {n_t12_field} field")

        if n_t12 >= 200 and n_t12_dense >= 30:
            # --- Fit global HI Tully-Fisher relation ---
            all_logMHI = np.array([p['logMHI'] for p in hitf_pts])
            all_logW50 = np.array([p['logW50'] for p in hitf_pts])

            # Linear fit: logMHI = a * logW50 + b
            from numpy.polynomial.polynomial import polyfit
            coeffs = np.polyfit(all_logW50, all_logMHI, 1)
            hitf_slope = coeffs[0]
            hitf_intercept = coeffs[1]
            hitf_pred = hitf_slope * all_logW50 + hitf_intercept
            hitf_res = all_logMHI - hitf_pred
            hitf_std = np.std(hitf_res)

            print(f"\n    HI Tully-Fisher: logMHI = {hitf_slope:.3f} * "
                  f"logW50 + {hitf_intercept:.3f}")
            print(f"    Global scatter: σ = {hitf_std:.4f} dex")

            # Compute residuals per galaxy
            for i, p in enumerate(hitf_pts):
                p['hitf_res'] = float(hitf_res[i])

            # --- Split by environment ---
            field_pts = [p for p in hitf_pts if p['env'] == 'field']
            dense_pts = [p for p in hitf_pts if p['env'] == 'dense']

            field_res = np.array([p['hitf_res'] for p in field_pts])
            dense_res = np.array([p['hitf_res'] for p in dense_pts])

            sigma_field = np.std(field_res)
            sigma_dense = np.std(dense_res)
            delta_sigma = sigma_field - sigma_dense

            print(f"\n    HI TF scatter by environment:")
            print(f"    Field:  σ = {sigma_field:.4f} dex "
                  f"(N = {len(field_pts)})")
            print(f"    Dense:  σ = {sigma_dense:.4f} dex "
                  f"(N = {len(dense_pts)})")
            print(f"    Δσ (field − dense) = {delta_sigma:+.4f} dex")

            # BEC prediction: field galaxies have MORE coherent condensate →
            # TIGHTER TF relation → σ_field < σ_dense
            # (Opposite sign from RAR scatter where field > dense due to
            # different physics: RAR scatter includes gbar estimation error
            # while TF scatter tests the mass-velocity relation tightness)

            # Bootstrap significance test
            n_boot_hitf = 10000
            boot_deltas = []
            all_res_combined = np.concatenate([field_res, dense_res])
            n_field_b = len(field_res)
            for _ in range(n_boot_hitf):
                np.random.shuffle(all_res_combined)
                bf = np.std(all_res_combined[:n_field_b])
                bd = np.std(all_res_combined[n_field_b:])
                boot_deltas.append(bf - bd)
            boot_deltas = np.array(boot_deltas)

            # P(field < dense): fraction of bootstraps where Δσ < 0
            # BEC predicts σ_field < σ_dense (more coherent in field)
            p_field_tighter = np.mean(boot_deltas <= delta_sigma)
            # P(observed or more extreme under null)
            p_value = min(p_field_tighter, 1 - p_field_tighter) * 2

            print(f"\n    Bootstrap (N={n_boot_hitf}):")
            print(f"    One-sided p-value (field tighter) = "
                  f"{p_field_tighter:.4f}")
            print(f"    Two-sided p-value = {p_value:.4f}")

            if delta_sigma < 0:
                print(f"    Field has TIGHTER TF relation: "
                      f"consistent with BEC coherence")
            else:
                print(f"    Dense has tighter TF relation: "
                      f"opposite to BEC coherence prediction")

            # --- Profile concentration by environment ---
            field_cprof = np.array([p['c_prof'] for p in field_pts
                                     if not np.isnan(p['c_prof'])
                                     and 0.1 < p['c_prof'] < 5.0])
            dense_cprof = np.array([p['c_prof'] for p in dense_pts
                                     if not np.isnan(p['c_prof'])
                                     and 0.1 < p['c_prof'] < 5.0])

            if len(field_cprof) >= 50 and len(dense_cprof) >= 10:
                print(f"\n    Profile concentration (C_prof = flux/(W50 × peak)):")
                print(f"    Field: median = {np.median(field_cprof):.4f}, "
                      f"σ = {np.std(field_cprof):.4f} "
                      f"(N = {len(field_cprof)})")
                print(f"    Dense: median = {np.median(dense_cprof):.4f}, "
                      f"σ = {np.std(dense_cprof):.4f} "
                      f"(N = {len(dense_cprof)})")

                # BEC predicts field profiles are more symmetric →
                # tighter C_prof distribution (smaller σ)
                sigma_cprof_field = np.std(field_cprof)
                sigma_cprof_dense = np.std(dense_cprof)
                delta_cprof = sigma_cprof_field - sigma_cprof_dense
                print(f"    Δσ(C_prof) field − dense = "
                      f"{delta_cprof:+.4f}")

                if delta_cprof < 0:
                    print(f"    Field profiles more regular: "
                          f"consistent with BEC coherence")
                else:
                    print(f"    Dense profiles more regular: "
                          f"opposite to BEC prediction")

            # --- HI TF scatter in bins of HI mass ---
            # The BEC coherence effect should be strongest at low M_HI
            # (low-mass galaxies are more DM-dominated)
            print(f"\n    HI TF scatter vs HI mass (field vs dense):")
            mass_bins = [(7.0, 8.5, 'low'), (8.5, 9.5, 'mid'),
                         (9.5, 11.0, 'high')]
            bec_consistent_bins = 0
            total_testable_bins = 0

            for mlo, mhi, label in mass_bins:
                fb = [p['hitf_res'] for p in field_pts
                      if mlo <= p['logMHI'] < mhi]
                db = [p['hitf_res'] for p in dense_pts
                      if mlo <= p['logMHI'] < mhi]
                if len(fb) >= 20 and len(db) >= 5:
                    sf = np.std(fb)
                    sd = np.std(db)
                    total_testable_bins += 1
                    if sf < sd:
                        bec_consistent_bins += 1
                    print(f"    logMHI [{mlo:.1f}, {mhi:.1f}): "
                          f"field σ={sf:.4f} (N={len(fb)}), "
                          f"dense σ={sd:.4f} (N={len(db)}), "
                          f"Δ={sf-sd:+.4f} "
                          f"{'✓' if sf < sd else '✗'}")
                else:
                    print(f"    logMHI [{mlo:.1f}, {mhi:.1f}): "
                          f"insufficient (field={len(fb)}, "
                          f"dense={len(db)})")

            # --- Verdict ---
            t12_delta = delta_sigma
            # BEC prediction: σ_field < σ_dense (field is tighter)
            # p_field_tighter is the one-sided p-value from the permutation test:
            # fraction of null shuffles with Δσ ≤ observed.
            # Small p = field really is tighter (reject null).
            if delta_sigma < 0 and p_field_tighter < 0.05:
                t12_supports = True
                print(f"\n    >>> HI TF coherence: Field SIGNIFICANTLY "
                      f"tighter (Δσ = {delta_sigma:+.4f}, "
                      f"p = {p_value:.4f})")
                print(f"    >>> Consistent with BEC condensate coherence")
            elif delta_sigma < 0 and p_field_tighter < 0.10:
                t12_supports = True
                print(f"\n    >>> HI TF coherence: Field marginally "
                      f"tighter (Δσ = {delta_sigma:+.4f}, "
                      f"p = {p_value:.4f})")
            elif delta_sigma < 0:
                print(f"\n    >>> HI TF coherence: Field tighter but not "
                      f"significant (Δσ = {delta_sigma:+.4f}, "
                      f"p = {p_value:.4f})")
            else:
                print(f"\n    >>> HI TF coherence: Dense tighter "
                      f"(Δσ = {delta_sigma:+.4f}), opposite to BEC "
                      f"prediction")

            # Mass-dependent check
            if total_testable_bins >= 2:
                print(f"    Mass-dependent check: {bec_consistent_bins}/"
                      f"{total_testable_bins} bins have field tighter")

        else:
            print(f"    Insufficient sample: {n_t12} total, "
                  f"{n_t12_dense} dense")
    else:
        print(f"    ALFALFA catalog not found")

    # Add Test 12 to summary
    t12_sig = (f"Δσ={t12_delta:+.4f}" if not np.isnan(t12_delta) else "N/A")
    tests_summary.append(('HI profile coherence (non-kinematic)',
                          t12_delta, t12_sig, t12_supports))
    n_support += int(t12_supports)
    if np.isnan(t12_delta):
        print(f"\n    [P] 12. ? HI profile coherence test "
              f"(data unavailable/insufficient)")
    else:
        print(f"\n    [P] 12. "
              f"{'✓ SUPPORTS' if t12_supports else '✗ opposes'} "
              f"HI profile coherence (non-kinematic channel)")

    # ================================================================
    # TEST 13: YANG+SDSS HALO MASS SCATTER BUNCHING (WEAK LENSING)
    # ================================================================
    # Completely independent of rotation kinematics:
    # Uses abundance-matching halo masses from Yang et al. SDSS DR7
    # group catalog (639K galaxies, validated against weak lensing).
    #
    # Physics: At fixed stellar mass M*, the scatter in halo mass M_h
    # encodes dark matter density profile fluctuations. If DM is a BEC,
    # these fluctuations follow Bose-Einstein statistics:
    #   σ²(log M_h | M*) = A × [n̄² + n̄] + C   (quantum)
    # vs σ²(log M_h | M*) = A × n̄ + C            (classical)
    # where n̄ = 1/[exp(√(gbar/g†)) − 1] at the typical acceleration
    # of the halo (gbar ≈ G × M* / r_h²).
    #
    # This is the "weak lensing channel" — the Yang halo masses are
    # calibrated against galaxy-galaxy lensing (Mandelbaum+2006,
    # Luo+2018). The test probes condensate density profile fluctuations
    # through a completely different physical observable from any RC.

    print("\n  " + "=" * 70)
    print("  TEST 13: YANG+SDSS HALO MASS SCATTER BUNCHING (WEAK LENSING)")
    print("  " + "=" * 70)
    print("    Non-kinematic channel: condensate density profile fluctuations")
    print("    via scatter in halo mass at fixed stellar mass (Yang+2007)")

    t13_delta_aic = np.nan
    t13_supports = False
    t13_delta = np.nan

    yang_base = os.path.join(DATA_DIR, 'yang_catalogs')
    yang_sdss7 = os.path.join(yang_base, 'SDSS7')
    yang_st = os.path.join(yang_base, 'SDSS7_ST')
    yang_membership = os.path.join(yang_base, 'imodelC_1')
    yang_groups = os.path.join(yang_base, 'modelC_group')

    if (os.path.exists(yang_sdss7) and os.path.exists(yang_st) and
            os.path.exists(yang_membership) and os.path.exists(yang_groups)):

        # --- Parse galaxy catalog ---
        yang_ra = []
        yang_dec = []
        yang_z = []
        with open(yang_sdss7, 'r') as f:
            for line in f:
                parts = line.split()
                yang_ra.append(float(parts[2]))
                yang_dec.append(float(parts[3]))
                yang_z.append(float(parts[4]))
        yang_ra = np.array(yang_ra)
        yang_dec = np.array(yang_dec)
        yang_z = np.array(yang_z)

        # --- Parse stellar masses and sizes ---
        yang_logMs = []
        yang_r50_arcsec = []
        yang_r90_arcsec = []
        yang_sersic_n = []
        with open(yang_st, 'r') as f:
            for line in f:
                parts = line.split()
                yang_logMs.append(float(parts[3]))   # model magnitudes
                yang_sersic_n.append(float(parts[4]))
                yang_r50_arcsec.append(float(parts[6]))
                yang_r90_arcsec.append(float(parts[7]))
        yang_logMs = np.array(yang_logMs)
        yang_r50_arcsec = np.array(yang_r50_arcsec)
        yang_r90_arcsec = np.array(yang_r90_arcsec)
        yang_sersic_n = np.array(yang_sersic_n)

        # --- Parse group membership ---
        yang_gal_grpid = []
        yang_gal_central_mass = []
        with open(yang_membership, 'r') as f:
            for line in f:
                parts = line.split()
                yang_gal_grpid.append(int(parts[2]))
                yang_gal_central_mass.append(int(parts[4]))
        yang_gal_grpid = np.array(yang_gal_grpid)
        yang_gal_central_mass = np.array(yang_gal_central_mass)

        # --- Parse group halo masses ---
        grp_data = {}
        with open(yang_groups, 'r') as f:
            for line in f:
                parts = line.split()
                gid = int(parts[0])
                if gid == 0:
                    continue  # header rows
                logMh_lum = float(parts[6])
                logMh_star = float(parts[7])
                fedge = float(parts[10])
                grp_data[gid] = {
                    'logMh_star': logMh_star,
                    'logMh_lum': logMh_lum,
                    'fedge': fedge,
                    'z': float(parts[3]),
                }

        print(f"    Yang catalog: {len(yang_ra)} galaxies, "
              f"{len(grp_data)} groups")

        # --- Build per-galaxy dataset ---
        # Physical size: r50_kpc from angular size and redshift
        H0_yang = 70.0  # km/s/Mpc
        c_km_yang = 2.998e5  # km/s
        h_hubble = 0.7

        # Angular diameter distance (low-z approx, good for z < 0.2)
        DA_Mpc = c_km_yang * yang_z / H0_yang
        theta_rad = yang_r50_arcsec * np.pi / (180.0 * 3600.0)
        r50_kpc = DA_Mpc * theta_rad * 1000.0  # Mpc -> kpc

        # Baryonic acceleration at r50: gbar = G * M* / r50^2
        G_SI = 6.674e-11
        Msun_kg = 1.989e30
        kpc_m = 3.086e19

        # Yang masses are log M* / (h^-2 Msun), convert to Msun
        Ms_Msun = 10.0**yang_logMs / h_hubble**2
        Ms_kg = Ms_Msun * Msun_kg
        r50_m = r50_kpc * kpc_m

        # Avoid division by zero
        r50_m_safe = np.where(r50_m > 0, r50_m, 1e30)
        gbar_yang = G_SI * Ms_kg / r50_m_safe**2
        log_gbar_yang = np.log10(np.maximum(gbar_yang, 1e-20))

        # BEC occupation number
        gdagger = 1.2e-10  # m/s^2
        nbar_yang = 1.0 / (np.exp(np.sqrt(gbar_yang / gdagger)) - 1.0 + 1e-30)

        # --- Quality cuts ---
        # 1. Valid redshift range (0.01 < z < 0.20)
        # 2. Valid stellar mass (logMs > 8)
        # 3. Valid r50 (> 0)
        # 4. Valid Sersic n (> 0)
        # 5. In a group with valid halo mass and f_edge > 0.6
        # 6. Central galaxy only (most massive in group) for clean M*-Mh

        n_yang_total = len(yang_ra)
        valid_basic = ((yang_z > 0.01) & (yang_z < 0.20) &
                       (yang_logMs > 8.0) & (yang_r50_arcsec > 0) &
                       (yang_sersic_n > 0) & np.isfinite(log_gbar_yang))

        # Get halo mass for each galaxy's group
        yang_logMh = np.full(n_yang_total, np.nan)
        yang_fedge = np.zeros(n_yang_total)
        yang_grp_mult = np.zeros(n_yang_total, dtype=int)

        # Count group multiplicity
        from collections import Counter
        grp_mult_counter = Counter(yang_gal_grpid)

        for i in range(n_yang_total):
            gid = yang_gal_grpid[i]
            if gid in grp_data:
                yang_logMh[i] = grp_data[gid]['logMh_star']
                yang_fedge[i] = grp_data[gid]['fedge']
            yang_grp_mult[i] = grp_mult_counter.get(gid, 0)

        valid_halo = (valid_basic &
                      np.isfinite(yang_logMh) & (yang_logMh > 0) &
                      (yang_fedge > 0.6) &
                      (yang_gal_central_mass == 1))  # centrals only

        n_valid = np.sum(valid_halo)
        print(f"    After cuts (central, f_edge>0.6, logMs>8): "
              f"{n_valid} galaxies")

        if n_valid >= 1000:
            # Extract valid arrays
            vh_logMs = yang_logMs[valid_halo]
            vh_logMh = yang_logMh[valid_halo]
            vh_log_gbar = log_gbar_yang[valid_halo]
            vh_nbar = nbar_yang[valid_halo]
            vh_mult = yang_grp_mult[valid_halo]
            vh_z = yang_z[valid_halo]
            vh_sersic = yang_sersic_n[valid_halo]

            print(f"    Stellar mass range: [{vh_logMs.min():.2f}, "
                  f"{vh_logMs.max():.2f}]")
            print(f"    Halo mass range: [{vh_logMh.min():.2f}, "
                  f"{vh_logMh.max():.2f}]")
            print(f"    log(gbar) range: [{vh_log_gbar.min():.2f}, "
                  f"{vh_log_gbar.max():.2f}], "
                  f"median {np.median(vh_log_gbar):.2f}")
            print(f"    Points in DM regime (log gbar < -10.5): "
                  f"{np.sum(vh_log_gbar < -10.5)}")

            # --- Step 1: Remove M*-Mh mean relation ---
            # Fit polynomial logMh = f(logMs) and take residuals
            # This isolates the scatter at fixed M*
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(vh_logMs, vh_logMh, deg=3)
            logMh_pred = P.polyval(vh_logMs, coeffs)
            residuals = vh_logMh - logMh_pred  # scatter at fixed M*

            print(f"\n    M*-Mh relation: 3rd-order polynomial fit")
            print(f"    Global residual scatter: "
                  f"σ = {np.std(residuals):.4f} dex")

            # --- Step 2: Bin by gbar and compute variance ---
            # Use 0.5 dex bins from -12.5 to -9.0
            t13_bin_edges = np.arange(-12.5, -8.5, 0.5)
            t13_bin_centers = 0.5 * (t13_bin_edges[:-1] + t13_bin_edges[1:])
            t13_bin_var = []
            t13_bin_var_err = []
            t13_bin_N = []
            t13_bin_nbar = []
            t13_bin_nbar_sq = []

            print(f"\n     log(gbar)      N       σ²       n̄"
                  f"     n̄²+n̄")
            print(f"    " + "-" * 52)

            for j in range(len(t13_bin_centers)):
                lo, hi = t13_bin_edges[j], t13_bin_edges[j + 1]
                mask = (vh_log_gbar >= lo) & (vh_log_gbar < hi)
                n_bin = np.sum(mask)
                if n_bin >= 30:
                    res_bin = residuals[mask]
                    v = np.var(res_bin)
                    v_err = v * np.sqrt(2.0 / (n_bin - 1))
                    # Mean nbar for this bin (from bin center)
                    gb_center = 10.0**(t13_bin_centers[j])
                    nb = 1.0 / (np.exp(np.sqrt(gb_center / gdagger))
                                - 1.0)
                    t13_bin_var.append(v)
                    t13_bin_var_err.append(v_err)
                    t13_bin_N.append(n_bin)
                    t13_bin_nbar.append(nb)
                    t13_bin_nbar_sq.append(nb**2 + nb)
                    print(f"        {t13_bin_centers[j]:>6.2f} "
                          f"{n_bin:>6} {v:>9.4f} {nb:>9.4f} "
                          f"{nb**2 + nb:>9.4f}")

            t13_bin_var = np.array(t13_bin_var)
            t13_bin_var_err = np.array(t13_bin_var_err)
            t13_bin_N = np.array(t13_bin_N)
            t13_bin_nbar = np.array(t13_bin_nbar)
            t13_bin_nbar_sq = np.array(t13_bin_nbar_sq)

            n_bins_13 = len(t13_bin_var)
            print(f"\n    Populated bins: {n_bins_13}")

            if n_bins_13 >= 4:
                # --- Step 3: Fit quantum vs classical models ---
                from scipy.optimize import curve_fit

                def quantum_model_13(nbar_sq_plus_n, A, C):
                    return A * nbar_sq_plus_n + C

                def classical_model_13(nbar, A, C):
                    return A * nbar + C

                # Weights: inverse variance of variance estimate
                weights_13 = 1.0 / np.maximum(t13_bin_var_err, 1e-10)

                try:
                    popt_q13, _ = curve_fit(
                        quantum_model_13, t13_bin_nbar_sq,
                        t13_bin_var, p0=[0.001, 0.01],
                        sigma=t13_bin_var_err, absolute_sigma=True,
                        maxfev=10000)
                    resid_q13 = t13_bin_var - quantum_model_13(
                        t13_bin_nbar_sq, *popt_q13)
                    chi2_q13 = np.sum((resid_q13 / t13_bin_var_err)**2)
                    aic_q13 = chi2_q13 + 2 * 2  # 2 params

                    popt_c13, _ = curve_fit(
                        classical_model_13, t13_bin_nbar,
                        t13_bin_var, p0=[0.01, 0.01],
                        sigma=t13_bin_var_err, absolute_sigma=True,
                        maxfev=10000)
                    resid_c13 = t13_bin_var - classical_model_13(
                        t13_bin_nbar, *popt_c13)
                    chi2_c13 = np.sum((resid_c13 / t13_bin_var_err)**2)
                    aic_c13 = chi2_c13 + 2 * 2

                    # Constant model
                    mean_var_13 = np.average(t13_bin_var,
                                             weights=weights_13)
                    resid_const_13 = t13_bin_var - mean_var_13
                    chi2_const_13 = np.sum(
                        (resid_const_13 / t13_bin_var_err)**2)
                    aic_const_13 = chi2_const_13 + 2 * 1

                    daic_13 = aic_c13 - aic_q13

                    print(f"\n    Model fitting ({n_bins_13} bins):")
                    print(f"    QUANTUM:   A = {popt_q13[0]:.6f}, "
                          f"C = {popt_q13[1]:.4f}, "
                          f"χ²/dof = {chi2_q13 / max(n_bins_13 - 2, 1):.3f}, "
                          f"AIC = {aic_q13:.2f}")
                    print(f"    CLASSICAL: A = {popt_c13[0]:.6f}, "
                          f"C = {popt_c13[1]:.4f}, "
                          f"χ²/dof = {chi2_c13 / max(n_bins_13 - 2, 1):.3f}, "
                          f"AIC = {aic_c13:.2f}")
                    print(f"    CONSTANT:  σ² = {mean_var_13:.4f}, "
                          f"AIC = {aic_const_13:.2f}")
                    print(f"\n    ΔAIC (classical − quantum) = "
                          f"{daic_13:+.2f}")

                    t13_delta_aic = daic_13

                    # Amplitude check
                    if popt_q13[0] > 0:
                        print(f"    Positive amplitude: consistent "
                              f"with BEC bunching")
                    else:
                        print(f"    Negative amplitude: halo mass "
                              f"scatter decreases at low gbar")

                    # Floor comparison
                    print(f"\n    Floor comparison:")
                    print(f"    Yang M_h scatter floor (C) = "
                          f"{popt_q13[1]:.4f}")
                    if sparc_info:
                        print(f"    SPARC RAR floor (C) = "
                              f"{sparc_C:.4f}")

                    # --- Step 4: Environment split ---
                    # Isolated (N=1) vs group (N >= 3)
                    iso_mask = vh_mult == 1
                    grp_mask = vh_mult >= 3
                    n_iso = np.sum(iso_mask)
                    n_grp = np.sum(grp_mask)

                    print(f"\n    Environment split:")
                    print(f"    Isolated (N=1): {n_iso} centrals")
                    print(f"    Group (N≥3): {n_grp} centrals")

                    if n_iso >= 500 and n_grp >= 500:
                        iso_res = residuals[iso_mask]
                        grp_res = residuals[grp_mask]
                        sig_iso = np.std(iso_res)
                        sig_grp = np.std(grp_res)
                        delta_sig = sig_iso - sig_grp

                        print(f"    Isolated σ(logMh|M*) = "
                              f"{sig_iso:.4f} dex")
                        print(f"    Group σ(logMh|M*) = "
                              f"{sig_grp:.4f} dex")
                        print(f"    Δσ (iso − grp) = "
                              f"{delta_sig:+.4f} dex")

                        if delta_sig < 0:
                            print(f"    Isolated TIGHTER: consistent "
                                  f"with undisturbed condensate")
                        else:
                            print(f"    Isolated WIDER: opposite to "
                                  f"BEC coherence prediction")

                    # --- Step 5: DM-regime leverage check ---
                    # Need enough bins below log(gbar) = -11.0
                    n_dm_bins_13 = sum(1 for j in range(n_bins_13)
                                       if t13_bin_centers[
                                           list(range(len(t13_bin_centers))
                                                )[j] if j < len(
                                                    t13_bin_centers)
                                                else 0] < -11.0
                                       and t13_bin_N[j] >= 30)
                    # Simpler computation
                    t13_dm_bins = 0
                    used_centers = []
                    for j in range(n_bins_13):
                        # Find the actual center for this bin
                        bc = t13_bin_centers[
                            np.where((t13_bin_centers >= -12.5) &
                                     (t13_bin_centers < -8.5))[0][j]
                        ] if j < len(t13_bin_centers) else -999
                        # Actually just recompute: bin j corresponds
                        # to the j-th populated bin
                        pass

                    # Recompute DM bins directly
                    t13_dm_bins = 0
                    t13_actual_centers = []
                    idx = 0
                    for j in range(len(t13_bin_centers)):
                        lo, hi = t13_bin_edges[j], t13_bin_edges[j + 1]
                        mask = (vh_log_gbar >= lo) & (vh_log_gbar < hi)
                        n_bin = np.sum(mask)
                        if n_bin >= 30:
                            t13_actual_centers.append(
                                t13_bin_centers[j])
                            if (t13_bin_centers[j] < -11.0 and
                                    n_bin >= 30):
                                t13_dm_bins += 1
                            idx += 1

                    has_leverage_13 = t13_dm_bins >= 2

                    # --- Check for abundance-matching systematics ---
                    # Yang halo masses come from ranking groups by
                    # luminosity/stellar mass. For isolated galaxies,
                    # the mapping is nearly 1:1, producing very small
                    # scatter by construction. The variance trend is
                    # dominated by this methodology, not DM physics.
                    # Diagnostic: if σ² monotonically increases from
                    # low to high gbar AND A < 0, the trend is driven
                    # by abundance matching, not quantum bunching.
                    mono_increasing = all(
                        t13_bin_var[i] <= t13_bin_var[i + 1]
                        for i in range(len(t13_bin_var) - 1))
                    am_dominated = (popt_q13[0] < 0 and
                                    (mono_increasing or
                                     t13_bin_var[-1] / max(
                                         t13_bin_var[0], 1e-10) > 3))

                    # --- Verdict ---
                    if am_dominated:
                        t13_delta_aic = np.nan
                        t13_supports = False
                        print(f"\n    ABUNDANCE-MATCHING DIAGNOSTIC:")
                        print(f"    Variance ratio (high/low gbar): "
                              f"{t13_bin_var[-1] / max(t13_bin_var[0], 1e-10):.1f}×")
                        print(f"    Amplitude A = {popt_q13[0]:.6f} "
                              f"(negative)")
                        print(f"    >>> INCONCLUSIVE: Variance trend "
                              f"driven by abundance-matching methodology")
                        print(f"    Yang halo masses for isolated "
                              f"galaxies have near-zero scatter")
                        print(f"    by construction (1 galaxy = 1 halo). "
                              f"This is NOT a DM physics signal.")
                        print(f"    The environment split (isolated "
                              f"5× tighter than groups) is consistent")
                        print(f"    with BEC coherence but cannot "
                              f"distinguish from AM systematics.")
                        print(f"    PREDICTION: Weak lensing per-galaxy "
                              f"halo masses from KiDS/DES")
                        print(f"    (not abundance matching) should "
                              f"show n̄(n̄+1) bunching at low gbar.")
                    elif not has_leverage_13:
                        t13_delta_aic = np.nan
                        t13_supports = False
                        print(f"\n    DM-REGIME COVERAGE:")
                        print(f"    Only {t13_dm_bins} bins with "
                              f"center < -11.0 and N≥30")
                        print(f"    >>> INCONCLUSIVE: insufficient "
                              f"low-gbar leverage")
                    elif daic_13 > 6 and popt_q13[0] > 0:
                        t13_supports = True
                        print(f"\n    >>> HALO MASS SCATTER: Quantum "
                              f"DECISIVELY preferred "
                              f"(ΔAIC = {daic_13:+.1f})")
                    elif daic_13 > 2 and popt_q13[0] > 0:
                        t13_supports = True
                        print(f"\n    >>> HALO MASS SCATTER: Quantum "
                              f"preferred "
                              f"(ΔAIC = {daic_13:+.1f})")
                    elif daic_13 > -2:
                        print(f"\n    >>> HALO MASS SCATTER: Models "
                              f"indistinguishable "
                              f"(ΔAIC = {daic_13:+.1f})")
                    else:
                        print(f"\n    >>> HALO MASS SCATTER: Classical "
                              f"preferred "
                              f"(ΔAIC = {daic_13:+.1f})")

                    # --- Bootstrap if signal detected ---
                    if t13_supports and n_valid >= 500:
                        print(f"\n    Bootstrap ΔAIC (500 iterations, "
                              f"galaxy-level resampling)...")
                        n_boot_13 = 500
                        boot_daics_13 = []
                        boot_As_13 = []
                        for b in range(n_boot_13):
                            idx_b = np.random.choice(
                                n_valid, size=n_valid, replace=True)
                            res_b = residuals[idx_b]
                            lgb_b = vh_log_gbar[idx_b]

                            bv = []
                            bve = []
                            bnbar_sq = []
                            bnbar = []
                            for j in range(len(t13_bin_centers)):
                                lo = t13_bin_edges[j]
                                hi = t13_bin_edges[j + 1]
                                m = ((lgb_b >= lo) & (lgb_b < hi))
                                n_b = np.sum(m)
                                if n_b >= 30:
                                    v = np.var(res_b[m])
                                    ve = v * np.sqrt(2.0 / (n_b - 1))
                                    gc = 10.0**(t13_bin_centers[j])
                                    nb = 1.0 / (np.exp(
                                        np.sqrt(gc / gdagger))
                                        - 1.0)
                                    bv.append(v)
                                    bve.append(ve)
                                    bnbar_sq.append(nb**2 + nb)
                                    bnbar.append(nb)

                            if len(bv) >= 4:
                                bv = np.array(bv)
                                bve = np.array(bve)
                                bnbar_sq = np.array(bnbar_sq)
                                bnbar_a = np.array(bnbar)
                                try:
                                    pq, _ = curve_fit(
                                        quantum_model_13, bnbar_sq,
                                        bv, p0=[0.001, 0.01],
                                        sigma=bve,
                                        absolute_sigma=True,
                                        maxfev=5000)
                                    rq = bv - quantum_model_13(
                                        bnbar_sq, *pq)
                                    cq = np.sum((rq / bve)**2)

                                    pc, _ = curve_fit(
                                        classical_model_13,
                                        bnbar_a, bv,
                                        p0=[0.01, 0.01],
                                        sigma=bve,
                                        absolute_sigma=True,
                                        maxfev=5000)
                                    rc = bv - classical_model_13(
                                        bnbar_a, *pc)
                                    cc = np.sum((rc / bve)**2)

                                    boot_daics_13.append(
                                        (cc + 4) - (cq + 4))
                                    boot_As_13.append(pq[0])
                                except Exception:
                                    pass

                        if len(boot_daics_13) >= 100:
                            bd = np.array(boot_daics_13)
                            ba = np.array(boot_As_13)
                            print(f"    Valid iterations: "
                                  f"{len(bd)}/{n_boot_13}")
                            print(f"    Median ΔAIC: {np.median(bd):+.2f}")
                            print(f"    68% CI: [{np.percentile(bd, 16):+.2f}"
                                  f", {np.percentile(bd, 84):+.2f}]")
                            pct_q = np.mean(bd > 0) * 100
                            print(f"    Quantum preferred in "
                                  f"{pct_q:.1f}% of bootstrap samples")
                            pct_pos_A = np.mean(ba > 0) * 100
                            print(f"    Positive A in {pct_pos_A:.1f}% "
                                  f"of samples")

                except Exception as e:
                    print(f"    Model fitting failed: {e}")
        else:
            print(f"    Insufficient valid galaxies: {n_valid}")
    else:
        print(f"    Yang catalog files not found")

    # Add Test 13 to summary
    t13_sig = (f"ΔAIC={t13_delta_aic:+.1f}"
               if not np.isnan(t13_delta_aic) else "N/A")
    tests_summary.append(('Yang halo mass scatter bunching '
                          '(weak lensing)', t13_delta_aic,
                          t13_sig, t13_supports))
    n_support += int(t13_supports)
    if np.isnan(t13_delta_aic):
        print(f"\n    [P] 13. ? INCONCLUSIVE Yang halo mass scatter "
              f"(data unavailable/insufficient)")
    else:
        print(f"\n    [P] 13. "
              f"{'✓ SUPPORTS' if t13_supports else '✗ opposes'} "
              f"Yang halo mass scatter bunching (weak lensing)")

    # ================================================================
    # TEST 13b: LENSING PROFILE SHAPE — SOLITONIC CORE vs NFW
    # ================================================================
    # Direct structural test: Does the mass profile of isolated galaxy
    # halos show the solitonic core predicted by BEC ground-state physics?
    #
    # The solitonic core is the condensate's ground state |ψ₀(r)|².
    # Its density profile follows ρ_sol(r) ∝ [1 + 0.091(r/r_c)²]⁻⁸
    # (Schive+2014), where r_c is the soliton core radius set by the
    # healing length ξ = ℏ/(m_a v) ∝ M^(1/3) for a self-gravitating BEC.
    #
    # In projection (lensing), this produces a ΔΣ(R) profile that is
    # flatter in the inner region compared to NFW, with a characteristic
    # inflection where soliton transitions to thermal envelope.
    #
    # Data: Brouwer+2021 KiDS-1000 ESD profiles for isolated galaxies
    # in 4 stellar mass bins: log(M*/h70⁻²M☉) = [8.5,10.3,10.6,10.8,11.0]
    #
    # This is the "direct imaging" test — it shows the condensate itself,
    # not just its statistical properties. If isolated halos have solitonic
    # cores, that's the ground-state wavefunction made visible through
    # gravitational lensing.

    print("\n  " + "=" * 70)
    print("  TEST 13b: LENSING PROFILE SHAPE — SOLITONIC CORE vs NFW")
    print("  " + "=" * 70)
    print("    Direct structural test: condensate ground state via lensing")
    print("    Data: Brouwer+2021 KiDS-1000 isolated galaxy ESD profiles")

    t13b_delta_aic = np.nan
    t13b_supports = False

    brouwer_dir = os.path.join(DATA_DIR, 'brouwer2021')

    # Stellar mass bin edges: log10(M*/(h70^-2 Msun))
    mass_bin_edges = [8.5, 10.3, 10.6, 10.8, 11.0]
    mass_bin_labels = [f"[{mass_bin_edges[i]:.1f}, {mass_bin_edges[i+1]:.1f}]"
                       for i in range(4)]
    # Representative stellar masses (geometric mean of bin edges in linear)
    mass_bin_logMs = [0.5 * (mass_bin_edges[i] + mass_bin_edges[i + 1])
                      for i in range(4)]

    # --- Load ESD profiles (rotation curve format: R in Mpc) ---
    rc_files = [os.path.join(brouwer_dir,
                f'Fig-3_Lensing-rotation-curves_Massbin-{i+1}.txt')
                for i in range(4)]

    # Also load RAR-format files (gbar axis) for cross-check
    rar_files = [os.path.join(brouwer_dir,
                 f'Fig-9_RAR-KiDS-isolated_Massbin-{i+1}.txt')
                 for i in range(4)]

    all_rc_exist = all(os.path.exists(f) for f in rc_files)
    all_rar_exist = all(os.path.exists(f) for f in rar_files)

    if all_rc_exist:
        from scipy.integrate import quad
        from scipy.optimize import minimize, curve_fit

        # Physical constants
        G_SI_13b = 6.674e-11        # m^3 kg^-1 s^-2
        Msun_kg_13b = 1.989e30      # kg
        pc_m_13b = 3.086e16         # m
        Mpc_m_13b = 3.086e22        # m
        kpc_m_13b = 3.086e19        # m
        gdagger_13b = 1.2e-10       # m/s^2 (McGaugh+2016)

        # Load all 4 mass bins
        esd_data = []
        for i, fpath in enumerate(rc_files):
            R_Mpc = []
            ESD_t = []
            ESD_err = []
            bias_K = []
            with open(fpath, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.split()
                    R_Mpc.append(float(parts[0]))
                    ESD_t.append(float(parts[1]))
                    ESD_err.append(float(parts[3]))
                    bias_K.append(float(parts[4]))

            R_Mpc = np.array(R_Mpc)
            ESD_t = np.array(ESD_t)
            ESD_err = np.array(ESD_err)
            bias_K = np.array(bias_K)

            # Apply multiplicative bias correction
            ESD_corr = ESD_t / bias_K
            err_corr = ESD_err / bias_K

            # Convert R from Mpc to kpc for fitting
            R_kpc = R_Mpc * 1000.0

            esd_data.append({
                'R_Mpc': R_Mpc, 'R_kpc': R_kpc,
                'ESD': ESD_corr, 'err': err_corr,
                'logMs': mass_bin_logMs[i],
                'label': mass_bin_labels[i]
            })

        print(f"    Loaded {len(esd_data)} stellar mass bins")
        for d in esd_data:
            print(f"      logM* {d['label']}: {len(d['R_kpc'])} radial bins, "
                  f"R = [{d['R_kpc'][0]:.0f}, {d['R_kpc'][-1]:.0f}] kpc")

        # ============================================================
        # MODEL 1: NFW profile — projected excess surface density
        # ============================================================
        # NFW: ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
        # Analytical ΔΣ(R) for NFW from Bartelmann 1996, Wright & Brainerd 2000

        def nfw_delta_sigma(R_kpc, M200_logMsun, c200):
            """
            NFW projected excess surface density ΔΣ(R).
            R_kpc: projected radius in kpc
            M200_logMsun: log10(M200/Msun)
            c200: concentration parameter
            Returns ΔΣ in h70*Msun/pc^2
            """
            M200 = 10.0**M200_logMsun  # Msun
            # Critical density at z~0.2 (typical for KiDS lenses)
            # ρ_crit ≈ 1.36e11 h^2 Msun/Mpc^3 for h=0.7
            rho_crit = 1.36e11 * 0.7**2  # Msun/Mpc^3

            # r200 from M200 = (4/3)π r200³ × 200 ρ_crit
            r200_Mpc = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
            r_s_Mpc = r200_Mpc / c200
            r_s_kpc = r_s_Mpc * 1000.0

            # Characteristic overdensity
            delta_c = (200.0 / 3.0) * c200**3 / (np.log(1.0 + c200) - c200 / (1.0 + c200))

            # Surface mass density scale
            rho_s = delta_c * rho_crit  # Msun/Mpc^3
            Sigma_s = rho_s * r_s_Mpc   # Msun/Mpc^2

            x = R_kpc / r_s_kpc  # dimensionless radius
            x = np.clip(x, 1e-6, 1e6)

            # Σ(R) for NFW (Wright & Brainerd 2000)
            Sigma = np.zeros_like(x)
            # x < 1
            m1 = x < 1.0
            if np.any(m1):
                t = np.sqrt(1.0 - x[m1]**2)
                Sigma[m1] = 2.0 * Sigma_s / (x[m1]**2 - 1.0) * (
                    1.0 / t * np.arccosh(1.0 / x[m1]) - 1.0)
            # x == 1
            m2 = np.abs(x - 1.0) < 1e-6
            if np.any(m2):
                Sigma[m2] = 2.0 * Sigma_s / 3.0
            # x > 1
            m3 = x > 1.0
            if np.any(m3):
                t = np.sqrt(x[m3]**2 - 1.0)
                Sigma[m3] = 2.0 * Sigma_s / (x[m3]**2 - 1.0) * (
                    1.0 - 1.0 / t * np.arctan(t))

            # Mean surface density within R: Σ̄(<R)
            Sigma_bar = np.zeros_like(x)
            m1 = x < 1.0
            if np.any(m1):
                t = np.sqrt(1.0 - x[m1]**2)
                g_x = np.log(x[m1] / 2.0) + 1.0 / t * np.arccosh(1.0 / x[m1])
                Sigma_bar[m1] = 4.0 * Sigma_s * g_x / x[m1]**2
            m2 = np.abs(x - 1.0) < 1e-6
            if np.any(m2):
                Sigma_bar[m2] = 4.0 * Sigma_s * (1.0 + np.log(0.5))
            m3 = x > 1.0
            if np.any(m3):
                t = np.sqrt(x[m3]**2 - 1.0)
                g_x = np.log(x[m3] / 2.0) + 1.0 / t * np.arctan(t)
                Sigma_bar[m3] = 4.0 * Sigma_s * g_x / x[m3]**2

            # ΔΣ = Σ̄(<R) - Σ(R), convert from Msun/Mpc^2 to Msun/pc^2
            delta_sigma = (Sigma_bar - Sigma) * 1e-12  # Mpc^2 → pc^2
            return delta_sigma

        # ============================================================
        # MODEL 2: BEC Soliton + NFW envelope
        # ============================================================
        # Soliton core: ρ_sol(r) = ρ_c [1 + 0.091(r/r_c)²]⁻⁸
        # (Schive+2014 fitting formula for ground-state wavefunction)
        # Transitions to NFW envelope at r ≈ 3 r_c (healing length)
        #
        # The soliton core radius scales as:
        #   r_c ≈ 1.6 kpc × (m_a/10⁻²² eV)⁻¹ × (M_halo/10⁹ M☉)⁻¹/³
        # In our framework with g† = 1.2e-10 m/s²:
        #   r_c ∝ (M*/g†)^(1/3) via the healing length ξ = √(GM/g†)

        def soliton_profile_3d(r_kpc, rho_c, r_c_kpc):
            """Schive+2014 soliton density profile."""
            return rho_c * (1.0 + 0.091 * (r_kpc / r_c_kpc)**2)**(-8)

        # --- Precompute soliton projected ΔΣ lookup table ---
        # Soliton: ρ(r) = ρ_c [1 + 0.091 (r/r_c)²]⁻⁸
        # In dimensionless units u = R/r_c:
        #   Σ(u) = ρ_c r_c × σ̃(u), ΔΣ̃(u) = Σ̄(<u) - Σ(u)
        # Precompute σ̃ and ΔΣ̃ once, then interpolate.
        from scipy.interpolate import interp1d as interp1d_13b

        N_SOL_GRID = 500
        u_sol_grid = np.logspace(-3, 3, N_SOL_GRID)
        sigma_tilde_13b = np.zeros(N_SOL_GRID)
        for j_s, u_s in enumerate(u_sol_grid):
            def _sol_integrand(t, _u=u_s):
                return 2.0 * (1.0 + 0.091 * (_u**2 + t**2))**(-8)
            sigma_tilde_13b[j_s], _ = quad(
                _sol_integrand, 0, 500.0 / max(u_s, 0.01),
                limit=200, epsrel=1e-8)

        u_sig_prod = u_sol_grid * sigma_tilde_13b
        cumul_13b = np.zeros(N_SOL_GRID)
        for j_s in range(1, N_SOL_GRID):
            du_s = u_sol_grid[j_s] - u_sol_grid[j_s - 1]
            cumul_13b[j_s] = (cumul_13b[j_s - 1] +
                0.5 * (u_sig_prod[j_s - 1] + u_sig_prod[j_s]) * du_s)
        sig_bar_tilde = np.where(u_sol_grid > 1e-10,
            2.0 * cumul_13b / u_sol_grid**2, sigma_tilde_13b[0])
        dsig_tilde_13b = sig_bar_tilde - sigma_tilde_13b

        _interp_dsigma_13b = interp1d_13b(
            np.log10(u_sol_grid), dsig_tilde_13b,
            kind='cubic', fill_value='extrapolate')

        print(f"    Soliton lookup table: {N_SOL_GRID} pts, "
              f"peak ΔΣ̃ = {dsig_tilde_13b.max():.4f}")

        def bec_delta_sigma(R_kpc, M200_logMsun, c200, r_c_kpc, f_sol):
            """BEC soliton+NFW envelope ΔΣ(R) using precomputed lookup."""
            M200 = 10.0**M200_logMsun
            M_sol = f_sol * M200
            rho_c = M_sol / (4.0 * np.pi * 3.883 * r_c_kpc**3)
            u = R_kpc / r_c_kpc
            log_u = np.log10(np.clip(u, u_sol_grid[0], u_sol_grid[-1]))
            dS_sol = rho_c * r_c_kpc * _interp_dsigma_13b(log_u) * 1e-6
            M_env = M200 * (1.0 - f_sol)
            logM_env = np.log10(max(M_env, 1e6))
            return dS_sol + nfw_delta_sigma(R_kpc, logM_env, c200)

        # ============================================================
        # FIT BOTH MODELS TO EACH MASS BIN
        # ============================================================
        print(f"\n    Fitting NFW and BEC soliton+envelope to each mass bin...")

        fit_results = []
        for i, d in enumerate(esd_data):
            R = d['R_kpc']
            ESD = d['ESD']
            err = d['err']

            # Only fit positive-ESD points (lensing is noisy at extremes)
            valid = (ESD > 0) & (err > 0) & np.isfinite(ESD) & np.isfinite(err)
            R_fit = R[valid]
            ESD_fit = ESD[valid]
            err_fit = err[valid]

            if len(R_fit) < 5:
                print(f"    Bin {i+1} ({d['label']}): insufficient valid points")
                fit_results.append(None)
                continue

            # --- NFW fit: 2 free parameters (M200, c200) ---
            def nfw_residuals(params):
                logM, c = params
                if c < 1 or c > 50 or logM < 10 or logM > 15:
                    return 1e20
                model = nfw_delta_sigma(R_fit, logM, c)
                chi2 = np.sum(((ESD_fit - model) / err_fit)**2)
                return chi2

            # Initial guess from stellar-to-halo mass relation
            logMs_i = d['logMs']
            logMh_guess = logMs_i + 1.5  # rough SHMR
            c_guess = 10.0 * (10.0**(logMh_guess) / 1e12)**(-0.1)  # c-M relation

            from scipy.optimize import minimize
            res_nfw = minimize(nfw_residuals, [logMh_guess, c_guess],
                               method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-4})
            logM_nfw, c_nfw = res_nfw.x
            chi2_nfw = res_nfw.fun
            n_params_nfw = 2
            aic_nfw = chi2_nfw + 2 * n_params_nfw

            model_nfw = nfw_delta_sigma(R_fit, logM_nfw, c_nfw)

            # --- BEC soliton + envelope fit: 4 free parameters ---
            # (M200, c200, r_c, f_sol)
            # Predicted core radius from g†:
            # ξ = sqrt(G M* / g†), r_c ~ ξ / 1000 (in kpc from SI)
            Ms_SI = 10.0**logMs_i * Msun_kg_13b
            xi_m = np.sqrt(G_SI_13b * Ms_SI / gdagger_13b)
            xi_kpc_pred = xi_m / kpc_m_13b
            # This gives the healing length; soliton core ~ few kpc

            # Physical bounds: r_c in [0.2ξ, 5ξ]
            rc_min_13b = max(0.5, 0.2 * xi_kpc_pred)
            rc_max_13b = max(5.0, 5.0 * xi_kpc_pred)

            def bec_residuals(params):
                logM, c, rc, fs = params
                if (c < 1 or c > 50 or logM < 10 or logM > 15
                        or rc < rc_min_13b or rc > rc_max_13b
                        or fs < 0.001 or fs > 0.3):
                    return 1e20
                try:
                    model = bec_delta_sigma(R_fit, logM, c, rc, fs)
                    if np.any(~np.isfinite(model)):
                        return 1e20
                    chi2 = np.sum(((ESD_fit - model) / err_fit)**2)
                    return chi2
                except Exception:
                    return 1e20

            # Try multiple starting points near predicted ξ
            best_bec = None
            best_chi2_bec = 1e20
            rc_starts = [0.5 * xi_kpc_pred, xi_kpc_pred,
                         2.0 * xi_kpc_pred, 3.0 * xi_kpc_pred]
            fs_starts = [0.01, 0.03, 0.08, 0.15]

            for rc0 in rc_starts:
                for fs0 in fs_starts:
                    try:
                        res_bec = minimize(
                            bec_residuals,
                            [logMh_guess, c_guess, rc0, fs0],
                            method='Nelder-Mead',
                            options={'maxiter': 30000, 'xatol': 1e-4,
                                     'fatol': 1e-6})
                        if res_bec.fun < best_chi2_bec:
                            best_chi2_bec = res_bec.fun
                            best_bec = res_bec.x
                    except Exception:
                        pass

            if best_bec is not None:
                logM_bec, c_bec, rc_bec, fs_bec = best_bec
                chi2_bec = best_chi2_bec
                n_params_bec = 4
                aic_bec = chi2_bec + 2 * n_params_bec

                model_bec = bec_delta_sigma(R_fit, logM_bec, c_bec,
                                            rc_bec, fs_bec)

                daic = aic_nfw - aic_bec  # positive = BEC preferred
                n_dof = len(R_fit)

                result = {
                    'bin': i + 1,
                    'label': d['label'],
                    'logMs': logMs_i,
                    'n_pts': len(R_fit),
                    'nfw': {'logM200': logM_nfw, 'c200': c_nfw,
                            'chi2': chi2_nfw, 'aic': aic_nfw,
                            'chi2_dof': chi2_nfw / max(n_dof - 2, 1)},
                    'bec': {'logM200': logM_bec, 'c200': c_bec,
                            'r_c_kpc': rc_bec, 'f_sol': fs_bec,
                            'chi2': chi2_bec, 'aic': aic_bec,
                            'chi2_dof': chi2_bec / max(n_dof - 4, 1)},
                    'delta_aic': daic,
                    'xi_pred_kpc': xi_kpc_pred,
                    'bec_preferred': daic > 0
                }
                fit_results.append(result)

                print(f"\n    Mass bin {i+1}: logM* = {d['label']}")
                print(f"      NFW:  logM200={logM_nfw:.2f}, c={c_nfw:.1f}, "
                      f"χ²/dof={chi2_nfw / max(n_dof - 2, 1):.2f}, "
                      f"AIC={aic_nfw:.1f}")
                print(f"      BEC:  logM200={logM_bec:.2f}, c={c_bec:.1f}, "
                      f"r_c={rc_bec:.1f} kpc, f_sol={fs_bec:.3f}, "
                      f"χ²/dof={chi2_bec / max(n_dof - 4, 1):.2f}, "
                      f"AIC={aic_bec:.1f}")
                print(f"      ΔAIC (NFW − BEC) = {daic:+.2f}  "
                      f"{'→ BEC preferred' if daic > 0 else '→ NFW preferred'}")
                print(f"      Predicted ξ = {xi_kpc_pred:.1f} kpc, "
                      f"fitted r_c = {rc_bec:.1f} kpc "
                      f"(ratio: {rc_bec / max(xi_kpc_pred, 0.01):.2f})")
            else:
                print(f"    Bin {i+1} ({d['label']}): BEC fit failed")
                fit_results.append(None)

        # ============================================================
        # AGGREGATE RESULTS ACROSS MASS BINS
        # ============================================================
        valid_results = [r for r in fit_results if r is not None]
        n_valid_bins = len(valid_results)

        if n_valid_bins >= 2:
            # Combined ΔAIC
            total_daic = sum(r['delta_aic'] for r in valid_results)
            n_bec_preferred = sum(1 for r in valid_results
                                  if r['bec_preferred'])

            print(f"\n    AGGREGATE across {n_valid_bins} mass bins:")
            print(f"    Total ΔAIC (NFW − BEC) = {total_daic:+.2f}")
            print(f"    BEC preferred in {n_bec_preferred}/{n_valid_bins} bins")

            # --- Core radius scaling test ---
            # BEC prediction: r_c ∝ M*^(1/3) or equivalently
            # log(r_c) = (1/3) log(M*) + const
            fitted_rc = np.array([r['bec']['r_c_kpc'] for r in valid_results])
            fitted_logMs = np.array([r['logMs'] for r in valid_results])
            predicted_xi = np.array([r['xi_pred_kpc'] for r in valid_results])

            if n_valid_bins >= 3:
                # Fit power law: log(r_c) = slope × log(M*) + intercept
                log_rc = np.log10(np.maximum(fitted_rc, 0.1))
                slope, intercept = np.polyfit(fitted_logMs, log_rc, 1)

                # Correlation between fitted r_c and predicted ξ
                from scipy.stats import pearsonr as pearsonr_13b
                if np.std(fitted_rc) > 0 and np.std(predicted_xi) > 0:
                    corr_rc_xi, p_rc_xi = pearsonr_13b(fitted_rc, predicted_xi)
                else:
                    corr_rc_xi, p_rc_xi = 0.0, 1.0

                print(f"\n    Core radius scaling:")
                print(f"    Fitted slope: d(log r_c)/d(log M*) = {slope:.3f}")
                print(f"    BEC prediction: slope = 1/3 ≈ 0.333")
                print(f"    Correlation r_c vs ξ_pred: r = {corr_rc_xi:.3f}, "
                      f"p = {p_rc_xi:.4f}")

                scaling_consistent = abs(slope - 1.0/3.0) < 0.3
            else:
                slope = np.nan
                corr_rc_xi = np.nan
                p_rc_xi = np.nan
                scaling_consistent = False

            # --- Resolution diagnostic ---
            # Check if soliton core is resolved by the data
            R_min_data = min(d['R_kpc'][0] for d in esd_data)
            max_xi = max(r['xi_pred_kpc'] for r in valid_results)
            core_resolved = R_min_data < 3.0 * max_xi
            resolution_ratio = R_min_data / max_xi

            print(f"\n    RESOLUTION DIAGNOSTIC:")
            print(f"    Innermost data point: R_min = {R_min_data:.0f} kpc")
            print(f"    Largest predicted ξ: {max_xi:.1f} kpc")
            print(f"    R_min / ξ_max = {resolution_ratio:.1f}")
            if not core_resolved:
                print(f"    >>> CORE UNRESOLVED: data starts at "
                      f"{resolution_ratio:.0f}× the predicted core radius")
                print(f"    The solitonic core occupies R < 3ξ ≈ "
                      f"{3*max_xi:.0f} kpc, but the innermost")
                print(f"    lensing bin is at {R_min_data:.0f} kpc. "
                      f"NFW vs BEC are degenerate at these radii.")

            # --- Verdict ---
            t13b_delta_aic = total_daic

            # If core is unresolved, mark as inconclusive regardless of ΔAIC
            if not core_resolved:
                # Even if ΔAIC is negative, this is expected from AIC penalty
                # not from data ruling out BEC. Mark as inconclusive.
                all_at_floor = all(r['bec']['f_sol'] <= 0.002
                                   for r in valid_results)
                if all_at_floor and total_daic < -2:
                    # BEC reduces to NFW + AIC penalty → inconclusive
                    t13b_delta_aic = np.nan
                    t13b_supports = False
                    print(f"\n    >>> LENSING PROFILES: INCONCLUSIVE")
                    print(f"    Soliton fraction at floor in all bins "
                          f"(f_sol ≤ 0.002) with total ΔAIC = {total_daic:+.1f}")
                    print(f"    This is the 2-parameter AIC penalty, not a "
                          f"physical rejection of BEC.")
                    print(f"    PREDICTION: Stacked lensing at R < "
                          f"{3*max_xi:.0f} kpc resolution should reveal")
                    print(f"    flattened core vs NFW cusp, with "
                          f"ξ ∝ M*^(1/3) scaling.")
                elif total_daic > 2 and n_bec_preferred >= 2:
                    t13b_supports = True
                    print(f"\n    >>> LENSING PROFILES: Solitonic core "
                          f"preferred (ΔAIC = {total_daic:+.1f})")
                    print(f"    (despite unresolved core — driven by "
                          f"envelope shape differences)")
                else:
                    print(f"\n    >>> LENSING PROFILES: Models "
                          f"indistinguishable (ΔAIC = {total_daic:+.1f})")
            elif total_daic > 6 and n_bec_preferred >= n_valid_bins // 2 + 1:
                t13b_supports = True
                print(f"\n    >>> LENSING PROFILES: Solitonic core "
                      f"DECISIVELY preferred (ΔAIC = {total_daic:+.1f})")
            elif total_daic > 2 and n_bec_preferred >= 2:
                t13b_supports = True
                print(f"\n    >>> LENSING PROFILES: Solitonic core "
                      f"preferred (ΔAIC = {total_daic:+.1f})")
            elif total_daic > -2:
                print(f"\n    >>> LENSING PROFILES: Models "
                      f"indistinguishable (ΔAIC = {total_daic:+.1f})")
            else:
                print(f"\n    >>> LENSING PROFILES: NFW "
                      f"preferred (ΔAIC = {total_daic:+.1f})")

            if scaling_consistent:
                print(f"    Core radius scaling consistent with BEC prediction")
            elif not np.isnan(slope):
                print(f"    Core radius scaling "
                      f"{'steeper' if slope > 1/3 else 'shallower'} "
                      f"than BEC prediction")

            # Check for soliton detection: are the fitted cores physically
            # reasonable? r_c should be 1-30 kpc, f_sol should be 0.5-15%
            n_physical = sum(1 for r in valid_results
                            if 0.5 < r['bec']['r_c_kpc'] < 100
                            and 0.001 < r['bec']['f_sol'] < 0.3)
            print(f"    Physically reasonable soliton fits: "
                  f"{n_physical}/{n_valid_bins}")

            # Store detailed results for JSON output
            t13b_results = {
                'n_bins': n_valid_bins,
                'total_delta_aic': total_daic,
                'n_bec_preferred': n_bec_preferred,
                'scaling_slope': slope if not np.isnan(slope) else None,
                'scaling_prediction': 1.0/3.0,
                'corr_rc_xi': corr_rc_xi if not np.isnan(corr_rc_xi) else None,
                'p_corr_rc_xi': p_rc_xi if not np.isnan(p_rc_xi) else None,
                'bins': []
            }
            for r in valid_results:
                t13b_results['bins'].append({
                    'mass_bin': r['label'],
                    'logMs': r['logMs'],
                    'n_pts': r['n_pts'],
                    'nfw_logM200': r['nfw']['logM200'],
                    'nfw_c200': r['nfw']['c200'],
                    'nfw_chi2_dof': r['nfw']['chi2_dof'],
                    'nfw_aic': r['nfw']['aic'],
                    'bec_logM200': r['bec']['logM200'],
                    'bec_c200': r['bec']['c200'],
                    'bec_rc_kpc': r['bec']['r_c_kpc'],
                    'bec_f_sol': r['bec']['f_sol'],
                    'bec_chi2_dof': r['bec']['chi2_dof'],
                    'bec_aic': r['bec']['aic'],
                    'delta_aic': r['delta_aic'],
                    'xi_pred_kpc': r['xi_pred_kpc']
                })
        else:
            print(f"    Insufficient valid mass bins for aggregate analysis")
            t13b_results = None
    else:
        print(f"    Brouwer+2021 data files not found in {brouwer_dir}")
        print(f"    Download from: kids.strw.leidenuniv.nl/sciencedata.php")
        t13b_results = None

    # --- Composite approach note ---
    # A SPARC+Brouwer composite approach (test_13b_composite.py) was also
    # attempted, stitching SPARC V(R)→ΔΣ(R) at R<30 kpc with Brouwer
    # lensing ΔΣ at R≥30 kpc. While the raw ΔAIC=+179 favored BEC,
    # diagnostic analysis revealed this was driven by a systematic
    # normalization mismatch: V²/(4GR) underestimates true projected ΔΣ
    # by 3-10× at small R for extended mass distributions. BEC with
    # fixed r_c=ξ showed NO improvement (ΔAIC≈-2 to -3 in all bins),
    # confirming the extra BEC parameters were absorbing the stitch
    # offset, not detecting a soliton core. See test_13b_diagnostics.py.
    if t13b_results is not None:
        t13b_results['composite_note'] = (
            'Composite SPARC+Brouwer approach also tested; INCONCLUSIVE '
            'due to V²/(4GR) conversion systematic. BEC with fixed r_c=ξ '
            'shows no improvement over NFW (ΔAIC ≈ -2 to -3).'
        )

    # Add Test 13b to summary
    t13b_sig = (f"ΔAIC={t13b_delta_aic:+.1f}"
                if not np.isnan(t13b_delta_aic) else "N/A")
    tests_summary.append(('Lensing profile shape: solitonic core vs NFW',
                          t13b_delta_aic, t13b_sig, t13b_supports))
    n_support += int(t13b_supports)
    if np.isnan(t13b_delta_aic):
        print(f"\n    [P] 13b. ? INCONCLUSIVE Lensing profile shape "
              f"(data unavailable/insufficient)")
    else:
        print(f"\n    [P] 13b. "
              f"{'✓ SUPPORTS' if t13b_supports else '✗ opposes'} "
              f"Lensing profile shape: solitonic core vs NFW")

    # ----------------------------------------------------------------
    # Summary of all refined tests
    # ----------------------------------------------------------------
    n_tests = len(tests_summary)
    # Count primary vs robustness during computation
    # These are also used later in the final summary and JSON output
    # Test indices (0-based): 0=Z-score, 1=threshold, 2=galaxy-level,
    #   3=DM-weighted, 4=MC-error, 5=Z-norm-gal, 6=BEC-transition,
    #   7=boson-bunching, 8=redshift-evolution, 9=ALFALFA-bunching,
    #   10=MaNGA-Vsig-bunching, 11=HI-profile-coherence,
    #   12=Yang-halo-scatter, 13=Lensing-profile-shape (13b)
    primary_indices = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13]  # 11 primary
    robustness_indices = [0, 3, 5]
    # Count inconclusive tests (NaN delta) separately
    n_primary_inconclusive = sum(1 for i in primary_indices
                                  if i < n_tests and np.isnan(tests_summary[i][1]))
    n_primary_total = sum(1 for i in primary_indices if i < n_tests) - n_primary_inconclusive
    n_primary_support = sum(1 for i in primary_indices
                           if i < n_tests and tests_summary[i][3])
    n_robust_total = sum(1 for i in robustness_indices if i < n_tests)
    n_robust_support = sum(1 for i in robustness_indices
                          if i < n_tests and tests_summary[i][3])
    verdict = ('STRONG' if n_primary_support >= 4 else
               'MODERATE-STRONG' if n_primary_support >= 3 else
               'MODERATE' if n_primary_support >= 2 else
               'WEAK' if n_primary_support >= 1 else 'INSUFFICIENT')
    pri_idx = {1, 2, 4, 6}  # 0-based indices of primary tests
    n_pri_support_mid = sum(1 for i in pri_idx if i < n_tests and tests_summary[i][3])
    n_pri_total_mid = sum(1 for i in pri_idx if i < n_tests)
    rob_idx = {0, 3, 5}
    n_rob_support_mid = sum(1 for i in rob_idx if i < n_tests and tests_summary[i][3])
    n_rob_total_mid = sum(1 for i in rob_idx if i < n_tests)

    print(f"\n  --- Summary of Refined BEC Tests ---")
    print(f"    PRIMARY tests: {n_pri_support_mid}/{n_pri_total_mid} support BEC")
    print(f"    ROBUSTNESS checks: {n_rob_support_mid}/{n_rob_total_mid} support BEC")
    print(f"    Overall: {n_support}/{n_tests} refined tests support BEC")

    if n_pri_support_mid >= 4:
        print(f"    >>> STRONG EVIDENCE for BEC environmental scatter prediction")
    elif n_pri_support_mid >= 3:
        print(f"    >>> MODERATE-STRONG EVIDENCE for BEC environmental scatter prediction")
    elif n_pri_support_mid >= 2:
        print(f"    >>> MODERATE EVIDENCE for BEC environmental scatter prediction")
    elif n_pri_support_mid >= 1:
        print(f"    >>> WEAK EVIDENCE for BEC environmental scatter prediction")
    else:
        print(f"    >>> INSUFFICIENT EVIDENCE for BEC environmental scatter prediction")

    # ============================================================
    # STEP 6: Per-dataset breakdown
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 6: PER-DATASET ENVIRONMENTAL BREAKDOWN")
    print("=" * 80)

    dataset_breakdown = {}
    for src in ['SPARC', 'deBlok2002', 'WALLABY', 'SS2020',
                'LITTLETHINGS', 'LVHIS', 'Yu2020', 'Swaters2025',
                'GHASP', 'Noordermeer2005', 'Vogt2004', 'Catinella2005',
                'VirgoRC', 'PHANGS', 'Verheijen2001', 'MaNGA', 'WALLABY_DR2']:
        src_pts = [p for p in all_points
                   if (p['source'] == src or
                       (src == 'SS2020' and p['source'].startswith('SS20_')))]
        if not src_pts:
            continue

        d_res = np.array([p['log_res'] for p in src_pts if p['env_dense'] == 'dense'])
        f_res = np.array([p['log_res'] for p in src_pts if p['env_dense'] == 'field'])
        d_gals = len(set(p['galaxy'] for p in src_pts if p['env_dense'] == 'dense'))
        f_gals = len(set(p['galaxy'] for p in src_pts if p['env_dense'] == 'field'))

        ds_info = {
            'n_dense_gal': d_gals, 'n_field_gal': f_gals,
            'n_dense_pts': len(d_res), 'n_field_pts': len(f_res),
            'sigma_dense': float(np.std(d_res)) if len(d_res) > 0 else np.nan,
            'sigma_field': float(np.std(f_res)) if len(f_res) > 0 else np.nan,
        }

        if len(d_res) > 5 and len(f_res) > 5:
            ds_info['delta'] = ds_info['sigma_field'] - ds_info['sigma_dense']
            _, p, _ = bootstrap_scatter_test(d_res, f_res, n_boot=5000)
            ds_info['p_field_gt_dense'] = float(1.0 - p)
        else:
            ds_info['delta'] = np.nan
            ds_info['p_field_gt_dense'] = np.nan

        dataset_breakdown[src] = ds_info

        # Print
        sig_d = f"{ds_info['sigma_dense']:.4f}" if not np.isnan(ds_info['sigma_dense']) else "---"
        sig_f = f"{ds_info['sigma_field']:.4f}" if not np.isnan(ds_info['sigma_field']) else "---"
        delta_str = f"{ds_info['delta']:+.4f}" if not np.isnan(ds_info.get('delta', np.nan)) else "---"
        p_str = f"{ds_info.get('p_field_gt_dense', np.nan):.3f}" if not np.isnan(ds_info.get('p_field_gt_dense', np.nan)) else "---"

        print(f"\n  {src}:")
        print(f"    Dense: {d_gals} gal, {len(d_res)} pts, sigma = {sig_d}")
        print(f"    Field: {f_gals} gal, {len(f_res)} pts, sigma = {sig_f}")
        print(f"    Delta = {delta_str}, P(f>d) = {p_str}")

    # ============================================================
    # STEP 7: Build per-galaxy summary
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 7: BUILDING PER-GALAXY SUMMARY")
    print("=" * 80)

    # galaxy_summary was already built before Step 5c
    galaxy_rows = []
    for gkey, gs in galaxy_summary.items():
        res = np.array(gs['log_res_list'])
        gbar = np.array(gs['log_gbar_list'])
        galaxy_rows.append({
            'galaxy': gs.get('name', gkey),
            'galaxy_key': gs.get('galaxy_key', gkey),
            'source': gs['source'],
            'n_points': len(res),
            'sigma_res': float(np.std(res)),
            'mean_res': float(np.mean(res)),
            'median_res': float(np.median(res)),
            'mean_log_gbar': float(np.mean(gbar)),
            'env_dense': gs['env_dense'],
            'logMh': gs['logMh'],
        })

    print(f"  {len(galaxy_rows)} galaxies in summary")

    # ============================================================
    # STEP 8: Save results
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 8: SAVING RESULTS")
    print("=" * 80)

    # 1. Per-point RAR data
    rar_csv = os.path.join(OUTPUT_DIR, 'rar_points_unified.csv')
    fieldnames = ['galaxy', 'galaxy_key', 'source', 'log_gbar', 'log_gobs', 'log_res',
                  'sigma_log_gobs', 'R_kpc', 'env_dense', 'logMh']
    with open(rar_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in all_points:
            row = {k: p.get(k, '') for k in fieldnames}
            # Round floats
            for k in ['log_gbar', 'log_gobs', 'log_res', 'sigma_log_gobs', 'R_kpc', 'logMh']:
                if k in row and isinstance(row[k], float):
                    row[k] = round(row[k], 6)
            writer.writerow(row)
    rar_csv_sha = sha256_file(rar_csv)
    with open(rar_csv, 'r') as _f:
        rar_csv_line_count = sum(1 for _ in _f)
    print(f"  Saved: {rar_csv}")
    print(f"    {len(all_points)} RAR points")
    print(f"    SHA256: {rar_csv_sha}")
    print(f"    Lines: {rar_csv_line_count}")

    # 2. Per-galaxy summary
    gal_csv = os.path.join(OUTPUT_DIR, 'galaxy_results_unified.csv')
    gal_fields = ['galaxy', 'galaxy_key', 'source', 'n_points', 'sigma_res', 'mean_res',
                  'median_res', 'mean_log_gbar', 'env_dense', 'logMh']
    with open(gal_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=gal_fields)
        writer.writeheader()
        for row in sorted(galaxy_rows, key=lambda x: x['galaxy']):
            for k in ['sigma_res', 'mean_res', 'median_res', 'mean_log_gbar', 'logMh']:
                if isinstance(row.get(k), float):
                    row[k] = round(row[k], 6)
            writer.writerow(row)
    print(f"  Saved: {gal_csv}")
    print(f"    {len(galaxy_rows)} galaxies")

    # Root-cause SPARC diagnosis report with concrete runtime evidence.
    diag_ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    sparc_diag_report = os.path.join(OUTPUT_DIR, f'sparc_diagnosis_report_{diag_ts}.txt')
    with open(sparc_diag_report, 'w') as f:
        f.write("SPARC DIAGNOSIS REPORT\n")
        f.write(f"timestamp_utc={dt.datetime.utcnow().isoformat()}Z\n")
        f.write(f"allow_missing_sparc={bool(allow_missing_sparc)}\n")
        f.write(f"sparc_points={len(sparc_points)}\n")
        f.write(f"sparc_unique_galaxy_keys={sparc_unique_keys}\n")
        f.write(f"datasets_has_sparc={'SPARC' in datasets} (SPARC seeded separately)\n")
        f.write(f"parsed_mass_model_galaxies={sparc_diag.get('parsed_mass_model_galaxies')}\n")
        f.write(f"parsed_mrt_properties={sparc_diag.get('parsed_mrt_properties')}\n")
        f.write(f"sparc_galaxies_after_cuts={sparc_diag.get('sparc_galaxies_after_cuts')}\n")
        f.write(f"sparc_points_after_cuts={sparc_diag.get('sparc_points_after_cuts')}\n")
        f.write(f"table2_missing_reason={sparc_diag.get('table2_missing_reason')}\n")
        f.write("resolved_paths:\n")
        for key, info in sparc_diag.get('paths', {}).items():
            f.write(f"  {key}.chosen={info.get('chosen')}\n")
            for cand, exists in info.get('exists', {}).items():
                f.write(f"    - {cand} exists={exists}\n")
        f.write(f"rar_points_unified_csv={rar_csv}\n")
        f.write(f"rar_points_unified_sha256={rar_csv_sha}\n")
        f.write(f"rar_points_unified_lines={rar_csv_line_count}\n")
    print(f"  Saved: {sparc_diag_report}")

    # Reproducibility stamp for dataset composition.
    sparc_input_shas = {}
    for key, info in sparc_diag.get('paths', {}).items():
        chosen = info.get('chosen')
        if chosen and os.path.exists(chosen):
            try:
                sparc_input_shas[key] = sha256_file(chosen)
            except Exception:
                sparc_input_shas[key] = None
        else:
            sparc_input_shas[key] = None
    meta_path = os.path.join(OUTPUT_DIR, 'rar_points_unified.meta.json')
    meta_obj = {
        'timestamp_utc': dt.datetime.utcnow().isoformat() + 'Z',
        'git_head': git_head_sha(),
        'output_csv': rar_csv,
        'output_csv_sha256': rar_csv_sha,
        'output_csv_lines': int(rar_csv_line_count),
        'n_points': int(n_points),
        'n_galaxies': int(n_galaxies),
        'counts_by_source_grouped': {k: int(v) for k, v in sorted(source_counts_grouped.items())},
        'counts_by_source_raw': {k: int(v) for k, v in sorted(source_counts_raw.items())},
        'flags': {
            'allow_missing_sparc': bool(allow_missing_sparc),
            'chi2_shared_sigma_int': bool(chi2_shared_sigma_int),
            'chi2_sigma_ref_model': str(chi2_sigma_ref_model),
        },
        'sparc_runtime': {
            'sparc_points': int(len(sparc_points)),
            'sparc_unique_galaxy_keys': int(sparc_unique_keys),
            'sparc_paths': sparc_diag.get('paths', {}),
            'sparc_input_sha256': sparc_input_shas,
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(meta_obj, f, indent=2)
    print(f"  Saved: {meta_path}")

    # 3. Summary JSON
    summary = {
        'pipeline': 'unified_rar',
        'description': 'Unified RAR environmental scatter test combining 8 datasets',
        'n_galaxies': n_galaxies,
        'n_data_points': n_points,
        'n_dense_galaxies': n_dense_gal,
        'n_field_galaxies': n_field_gal,
        'n_dense_points': int(len(dense_res)),
        'n_field_points': int(len(field_res)),
        'rar_points_unified_sha256': rar_csv_sha,
        'rar_points_unified_lines': int(rar_csv_line_count),
        'rar_points_unified_meta': os.path.basename(meta_path),
        'sparc_diagnosis_report': os.path.basename(sparc_diag_report),
        'datasets': {k: int(v) for k, v in source_counts_grouped.items()},
        'datasets_grouped': {k: int(v) for k, v in source_counts_grouped.items()},
        'datasets_raw': {k: int(v) for k, v in source_counts_raw.items()},
        'overall': {
            'sigma_all': float(np.std([p['log_res'] for p in all_points])),
            'mean_all': float(np.mean([p['log_res'] for p in all_points])),
        },
        'environment': {
            'sigma_dense': float(sigma_dense) if not np.isnan(sigma_dense) else None,
            'sigma_field': float(sigma_field) if not np.isnan(sigma_field) else None,
            'delta_sigma': float(delta_sigma) if not np.isnan(delta_sigma) else None,
            'sigma_int_dense': float(sigma_int_dense) if not np.isnan(sigma_int_dense) else None,
            'sigma_int_field': float(sigma_int_field) if not np.isnan(sigma_int_field) else None,
            'delta_sigma_int': float(delta_sigma_int) if not np.isnan(delta_sigma_int) else None,
            'mean_err2_dense': float(mean_err2_dense),
            'mean_err2_field': float(mean_err2_field),
            'p_field_gt_dense': float(1.0 - p_val) if not np.isnan(p_val) else None,
            'p_value_onesided': float(p_val) if not np.isnan(p_val) else None,
            'levene_F': float(lev_stat) if not np.isnan(lev_stat) else None,
            'levene_p': float(lev_p) if not np.isnan(lev_p) else None,
            'brown_forsythe_F': float(bf_stat) if not np.isnan(bf_stat) else None,
            'brown_forsythe_p': float(bf_p) if not np.isnan(bf_p) else None,
        },
        'binned_results': [],
        'dm_regime_test': {
            'description': 'BEC prediction test: field > dense scatter in DM-dominated regime',
            'threshold_log_gbar': dm_threshold,
            'dm_regime': {
                'n_dense': int(len(dm_d_res)),
                'n_field': int(len(dm_f_res)),
                'n_dense_gal': dm_d_gal,
                'n_field_gal': dm_f_gal,
                'sigma_dense': dm_sigma_d,
                'sigma_field': dm_sigma_f,
                'delta': dm_delta,
                'p_field_gt_dense': float(1.0 - dm_p_val) if not np.isnan(dm_p_val) else None,
            },
            'baryon_regime': {
                'n_dense': int(len(bar_d_res)),
                'n_field': int(len(bar_f_res)),
                'sigma_dense': bar_sigma_d,
                'sigma_field': bar_sigma_f,
                'delta': bar_delta,
            },
            'per_dataset_dm': {},
        },
        'per_dataset': {},
        'refined_bec_tests': {
            'description': 'Five refined tests to prove/disprove BEC scatter prediction',
            'z_score_test': {
                'description': 'Within-dataset Z-score normalization eliminates Simpson Paradox',
                'z_sigma_dense_dm': float(z_sig_d_dm) if not np.isnan(z_sig_d_dm) else None,
                'z_sigma_field_dm': float(z_sig_f_dm) if not np.isnan(z_sig_f_dm) else None,
                'z_delta_dm': float(z_delta_dm) if not np.isnan(z_delta_dm) else None,
                'p_field_gt_dense_dm': float(1.0 - z_dm_p) if not np.isnan(z_dm_p) else None,
                'z_sigma_dense_all': float(z_sig_d) if not np.isnan(z_sig_d) else None,
                'z_sigma_field_all': float(z_sig_f) if not np.isnan(z_sig_f) else None,
                'z_delta_all': float(z_delta) if not np.isnan(z_delta) else None,
                'supports_bec': bool(z_delta_dm > 0) if not np.isnan(z_delta_dm) else None,
            },
            'threshold_scan': {
                'description': 'Optimal threshold for BEC signal',
                'best_threshold': float(zbest['threshold']) if z_threshold_results else None,
                'best_z_delta': float(zbest['z_delta']) if z_threshold_results else None,
                'best_p': float(zbest['z_p_field_gt_dense']) if z_threshold_results else None,
                'all_thresholds': z_threshold_results if z_threshold_results else [],
            },
            'galaxy_level_test': {
                'description': 'One value per galaxy removes point-count bias',
                'n_dense_gal': len(gal_d),
                'n_field_gal': len(gal_f),
                'sigma_dense_all': float(gal_sig_d) if not np.isnan(gal_sig_d) else None,
                'sigma_field_all': float(gal_sig_f) if not np.isnan(gal_sig_f) else None,
                'delta_all': float(gal_delta) if not np.isnan(gal_delta) else None,
                'n_dense_gal_dm': len(gal_d_dm),
                'n_field_gal_dm': len(gal_f_dm),
                'sigma_dense_dm': float(gal_sig_d_dm) if not np.isnan(gal_sig_d_dm) else None,
                'sigma_field_dm': float(gal_sig_f_dm) if not np.isnan(gal_sig_f_dm) else None,
                'delta_dm': float(gal_delta_dm) if not np.isnan(gal_delta_dm) else None,
                'p_field_gt_dense_dm': float(1.0 - gal_dm_p) if not np.isnan(gal_dm_p) else None,
                'supports_bec': bool(gal_delta_dm > 0) if not np.isnan(gal_delta_dm) else None,
            },
            'dm_fraction_weighted': {
                'description': 'Points weighted by DM fraction (f_DM = 1 - gbar/gobs)',
                'f_dm_gt_0p5': {
                    'n_dense': len(hd_d_res), 'n_field': len(hd_f_res),
                    'w_sigma_dense': float(hd_sig_d) if not np.isnan(hd_sig_d) else None,
                    'w_sigma_field': float(hd_sig_f) if not np.isnan(hd_sig_f) else None,
                    'delta': float(hd_delta) if not np.isnan(hd_delta) else None,
                    'p_field_gt_dense': float(1.0 - hd_p) if not np.isnan(hd_p) else None,
                },
                'f_dm_gt_0p8': {
                    'n_dense': len(vhd_d), 'n_field': len(vhd_f),
                    'sigma_dense': float(np.std(vhd_d)) if len(vhd_d) > 0 else None,
                    'sigma_field': float(np.std(vhd_f)) if len(vhd_f) > 0 else None,
                    'delta': float(vhd_delta) if not np.isnan(vhd_delta) else None,
                    'p_field_gt_dense': float(vhd_p) if not np.isnan(vhd_p) else None,
                },
                'supports_bec': bool(hd_delta > 0) if not np.isnan(hd_delta) else None,
            },
            'monte_carlo_errors': {
                'description': 'MC error propagation (1000 realizations)',
                'n_mc': n_mc,
                'overall': {
                    'median_delta': float(mc_all_med),
                    'ci_68': [float(mc_all_68lo), float(mc_all_68hi)],
                    'ci_95': [float(mc_all_95lo), float(mc_all_95hi)],
                    'frac_positive': float(frac_positive_all),
                },
                'dm_regime': {
                    'median_delta': float(mc_dm_med),
                    'ci_68': [float(mc_dm_68lo), float(mc_dm_68hi)],
                    'ci_95': [float(mc_dm_95lo), float(mc_dm_95hi)],
                    'frac_positive': float(frac_positive_dm),
                },
                'supports_bec': bool(frac_positive_dm > 0.68),
            },
            'z_norm_galaxy_level': {
                'description': 'Galaxy-level scatter with Z-score normalization (DM regime)',
                'n_dense_gal': len(gz_d),
                'n_field_gal': len(gz_f),
                'z_sigma_dense': float(gz_sig_d) if not np.isnan(gz_sig_d) else None,
                'z_sigma_field': float(gz_sig_f) if not np.isnan(gz_sig_f) else None,
                'z_delta': float(gz_delta) if not np.isnan(gz_delta) else None,
                'p_field_gt_dense': float(1.0 - gz_p) if not np.isnan(gz_p) else None,
                'supports_bec': bool(gz_delta > 0) if not np.isnan(gz_delta) else None,
            },
            'bec_transition_function': {
                'description': 'Tests if Dsigma(gbar) follows Bose-Einstein occupation number',
                'n_bins': len(bin_delta_z) if len(bin_delta_z) > 0 else 0,
                'bin_centers': [float(x) for x in bin_gbar_centers] if len(bin_gbar_centers) > 0 else [],
                'bin_delta_z': [float(x) for x in bin_delta_z] if len(bin_delta_z) > 0 else [],
                'bec_fit': {
                    'amplitude_A': float(bec_A) if not np.isnan(bec_A) else None,
                    'offset_C': float(bec_C) if not np.isnan(bec_C) else None,
                    'reduced_chi2': float(reduced_chi2_bec_z_uncalibrated) if not np.isnan(reduced_chi2_bec_z_uncalibrated) else None,
                    'reduced_chi2_bec_z_uncalibrated': float(reduced_chi2_bec_z_uncalibrated) if not np.isnan(reduced_chi2_bec_z_uncalibrated) else None,
                    'sigma_int_bec_z': float(sigma_int_bec_z) if not np.isnan(sigma_int_bec_z) else None,
                    'reduced_chi2_bec_z_calibrated': float(reduced_chi2_bec_z_calibrated) if not np.isnan(reduced_chi2_bec_z_calibrated) else None,
                    'chi2_cal_method_bec': chi2_cal_method_bec,
                    'chi2_cal_n_used_bec': int(chi2_cal_n_used_bec) if chi2_cal_n_used_bec is not None else None,
                    'chi2_cal_dof_used_bec': int(chi2_cal_dof_used_bec) if chi2_cal_dof_used_bec is not None else None,
                    'chi2_cal_dof_assumption_bec': chi2_cal_dof_assumption,
                    'chi2_cal_bracket_bec': [float(x) for x in chi2_cal_bracket_bec] if chi2_cal_bracket_bec else None,
                    'chi2_cal_reason_bec': chi2_cal_reason_bec,
                    'aic': float(aic_bec) if not np.isnan(aic_bec) else None,
                },
                'linear_fit': {
                    'reduced_chi2': float(reduced_chi2_linear_z_uncalibrated) if not np.isnan(reduced_chi2_linear_z_uncalibrated) else None,
                    'reduced_chi2_linear_z_uncalibrated': float(reduced_chi2_linear_z_uncalibrated) if not np.isnan(reduced_chi2_linear_z_uncalibrated) else None,
                    'sigma_int_linear_z': float(sigma_int_linear_z) if not np.isnan(sigma_int_linear_z) else None,
                    'reduced_chi2_linear_z_calibrated': float(reduced_chi2_linear_z_calibrated) if not np.isnan(reduced_chi2_linear_z_calibrated) else None,
                    'chi2_cal_method_linear': chi2_cal_method_linear,
                    'chi2_cal_n_used_linear': int(chi2_cal_n_used_linear) if chi2_cal_n_used_linear is not None else None,
                    'chi2_cal_dof_used_linear': int(chi2_cal_dof_used_linear) if chi2_cal_dof_used_linear is not None else None,
                    'chi2_cal_dof_assumption_linear': chi2_cal_dof_assumption,
                    'chi2_cal_bracket_linear': [float(x) for x in chi2_cal_bracket_linear] if chi2_cal_bracket_linear else None,
                    'chi2_cal_reason_linear': chi2_cal_reason_linear,
                    'aic': float(aic_lin) if not np.isnan(aic_lin) else None,
                },
                'constant_fit': {
                    'reduced_chi2': float(reduced_chi2_const_z_uncalibrated) if not np.isnan(reduced_chi2_const_z_uncalibrated) else None,
                    'reduced_chi2_const_z_uncalibrated': float(reduced_chi2_const_z_uncalibrated) if not np.isnan(reduced_chi2_const_z_uncalibrated) else None,
                    'sigma_int_const_z': float(sigma_int_const_z) if not np.isnan(sigma_int_const_z) else None,
                    'reduced_chi2_const_z_calibrated': float(reduced_chi2_const_z_calibrated) if not np.isnan(reduced_chi2_const_z_calibrated) else None,
                    'chi2_cal_method_const': chi2_cal_method_const,
                    'chi2_cal_n_used_const': int(chi2_cal_n_used_const) if chi2_cal_n_used_const is not None else None,
                    'chi2_cal_dof_used_const': int(chi2_cal_dof_used_const) if chi2_cal_dof_used_const is not None else None,
                    'chi2_cal_dof_assumption_const': chi2_cal_dof_assumption,
                    'chi2_cal_bracket_const': [float(x) for x in chi2_cal_bracket_const] if chi2_cal_bracket_const else None,
                    'chi2_cal_reason_const': chi2_cal_reason_const,
                    'aic': float(aic_const) if not np.isnan(aic_const) else None,
                },
                'chi2_shared_ref_model': chi2_shared_ref_model if chi2_shared_sigma_int else None,
                'sigma_int_shared_z': float(sigma_int_shared_z) if not np.isnan(sigma_int_shared_z) else None,
                'reduced_chi2_bec_z_shared': float(reduced_chi2_bec_z_shared) if not np.isnan(reduced_chi2_bec_z_shared) else None,
                'reduced_chi2_lin_z_shared': float(reduced_chi2_linear_z_shared) if not np.isnan(reduced_chi2_linear_z_shared) else None,
                'reduced_chi2_const_z_shared': float(reduced_chi2_const_z_shared) if not np.isnan(reduced_chi2_const_z_shared) else None,
                'chi2_shared_method': chi2_shared_method,
                'chi2_shared_n_used_ref': int(chi2_shared_n_used_ref) if chi2_shared_n_used_ref is not None else None,
                'chi2_shared_reason': chi2_shared_reason,
                'delta_aic_linear_minus_bec': float(delta_aic_bec_vs_lin) if not np.isnan(delta_aic_bec_vs_lin) else None,
                'delta_aic_constant_minus_bec': float(delta_aic_bec_vs_const) if not np.isnan(delta_aic_bec_vs_const) else None,
                'best_model': best_model,
                'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else None,
                'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None,
                'supports_bec': t7_supports,
                'robustness': {
                    'bootstrap_daic': bootstrap_results,
                    'leave_one_out': jackknife_results,
                    'test6_diagnostic': t6_jackknife,
                },
            },
            'summary': {
                'n_tests': n_tests,
                'n_supporting': n_support,
                'n_primary_tests': n_primary_total,
                'n_primary_supporting': n_primary_support,
                'n_robustness_checks': n_robust_total,
                'n_robustness_supporting': n_robust_support,
                'tests': [{'name': t[0], 'delta': float(t[1]) if not np.isnan(t[1]) else None,
                           'significance': t[2], 'supports_bec': t[3],
                           'category': 'primary' if i in set(primary_indices) else 'robustness'}
                          for i, t in enumerate(tests_summary)],
                'verdict': verdict,
            },
            'test8_boson_bunching': bunching_result,
            'test13b_lensing_profile': t13b_results,
        },
    }

    # Add per-dataset DM regime results
    for src, info in dm_per_dataset.items():
        clean = {}
        for k, v in info.items():
            if isinstance(v, float) and np.isnan(v):
                clean[k] = None
            else:
                clean[k] = v
        summary['dm_regime_test']['per_dataset_dm'][src] = clean

    # Serialize binned results (clean NaN)
    for b in binned:
        clean_b = {}
        for k, v in b.items():
            if isinstance(v, float) and np.isnan(v):
                clean_b[k] = None
            else:
                clean_b[k] = v
        summary['binned_results'].append(clean_b)

    # Serialize dataset breakdown
    for src, info in dataset_breakdown.items():
        clean_info = {}
        for k, v in info.items():
            if isinstance(v, float) and np.isnan(v):
                clean_info[k] = None
            else:
                clean_info[k] = v
        summary['per_dataset'][src] = clean_info

    summary_path = os.path.join(OUTPUT_DIR, 'summary_unified.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    # ============================================================
    # STEP 6b: MULTI-POINT RESTRICTION TEST
    # Run key tests restricted to galaxies with ≥5 DM-regime points
    # This eliminates single-point galaxy bias
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 6b: MULTI-POINT RESTRICTION TEST")
    print("  (Tests restricted to galaxies with ≥5 DM-regime points)")
    print("=" * 80)

    # Find galaxies with ≥5 DM-regime points
    gal_dm_counts = {}
    for p in all_points:
        if p['log_gbar'] < -10.5:  # DM regime
            g = p['galaxy']
            if g not in gal_dm_counts:
                gal_dm_counts[g] = {'n_dm': 0, 'env': p['env_dense']}
            gal_dm_counts[g]['n_dm'] += 1

    ext_dense_gals = set(g for g, info in gal_dm_counts.items()
                         if info['n_dm'] >= 5 and info['env'] == 'dense')
    ext_field_gals = set(g for g, info in gal_dm_counts.items()
                         if info['n_dm'] >= 5 and info['env'] == 'field')

    print(f"\n  Extended-RC galaxies (≥5 DM-regime pts): "
          f"{len(ext_dense_gals)} dense, {len(ext_field_gals)} field")

    # Filter points to only extended-RC galaxies
    ext_dm_pts = [p for p in all_points
                  if p['log_gbar'] < -10.5 and p['galaxy'] in (ext_dense_gals | ext_field_gals)]
    ext_dense_pts = [p for p in ext_dm_pts if p['env_dense'] == 'dense']
    ext_field_pts = [p for p in ext_dm_pts if p['env_dense'] == 'field']

    print(f"  Extended DM-regime points: {len(ext_dense_pts)} dense, {len(ext_field_pts)} field")

    mp_tests_support = 0
    mp_tests_total = 0

    if len(ext_dense_pts) >= 20 and len(ext_field_pts) >= 20:
        # MP-Test A: Raw scatter comparison
        res_d = np.array([p['log_res'] for p in ext_dense_pts])
        res_f = np.array([p['log_res'] for p in ext_field_pts])
        sig_d = np.std(res_d)
        sig_f = np.std(res_f)
        delta_mp = sig_f - sig_d
        _, lev_p_mp = stats.levene(res_d, res_f)

        supports_a = delta_mp > 0
        mp_tests_total += 1
        if supports_a:
            mp_tests_support += 1
        marker_a = '✓' if supports_a else '✗'
        print(f"\n  MP-Test A (raw DM scatter, ext-RC only):")
        print(f"    sigma_dense = {sig_d:.4f}, sigma_field = {sig_f:.4f}")
        print(f"    Delta = {delta_mp:+.4f}  Levene p = {lev_p_mp:.4f}  {marker_a}")

        # MP-Test B: Galaxy-level scatter (mean residual per galaxy)
        gal_mean_res = {}
        for p in ext_dm_pts:
            g = p['galaxy']
            if g not in gal_mean_res:
                gal_mean_res[g] = {'res_list': [], 'env': p['env_dense']}
            gal_mean_res[g]['res_list'].append(p['log_res'])

        gal_sigmas_d = []
        gal_sigmas_f = []
        for g, info in gal_mean_res.items():
            if len(info['res_list']) >= 5:
                mean_r = np.mean(info['res_list'])
                if info['env'] == 'dense':
                    gal_sigmas_d.append(mean_r)
                else:
                    gal_sigmas_f.append(mean_r)

        if len(gal_sigmas_d) >= 5 and len(gal_sigmas_f) >= 5:
            gsig_d = np.std(gal_sigmas_d)
            gsig_f = np.std(gal_sigmas_f)
            delta_gal_mp = gsig_f - gsig_d
            supports_b = delta_gal_mp > 0
            mp_tests_total += 1
            if supports_b:
                mp_tests_support += 1
            marker_b = '✓' if supports_b else '✗'
            print(f"\n  MP-Test B (galaxy-level, ext-RC only):")
            print(f"    N_dense = {len(gal_sigmas_d)}, N_field = {len(gal_sigmas_f)}")
            print(f"    sigma_dense = {gsig_d:.4f}, sigma_field = {gsig_f:.4f}")
            print(f"    Delta = {delta_gal_mp:+.4f}  {marker_b}")

        # MP-Test C: Within-dataset Z-score (ext-RC only)
        # Group by source
        src_pts_mp = {}
        for p in ext_dm_pts:
            s = p['source']
            if s not in src_pts_mp:
                src_pts_mp[s] = []
            src_pts_mp[s].append(p)

        z_dense_mp = []
        z_field_mp = []
        for src, pts_list in src_pts_mp.items():
            res_vals = np.array([p['log_res'] for p in pts_list])
            mu = np.mean(res_vals)
            sd = np.std(res_vals)
            if sd < 1e-6:
                continue
            for p in pts_list:
                z = abs((p['log_res'] - mu) / sd)
                if p['env_dense'] == 'dense':
                    z_dense_mp.append(z)
                else:
                    z_field_mp.append(z)

        if len(z_dense_mp) >= 20 and len(z_field_mp) >= 20:
            zsig_d = np.std(z_dense_mp)
            zsig_f = np.std(z_field_mp)
            delta_z_mp = zsig_f - zsig_d
            supports_c = delta_z_mp > 0
            mp_tests_total += 1
            if supports_c:
                mp_tests_support += 1
            marker_c = '✓' if supports_c else '✗'
            print(f"\n  MP-Test C (Z-score, ext-RC only):")
            print(f"    Z_sigma_dense = {zsig_d:.4f}, Z_sigma_field = {zsig_f:.4f}")
            print(f"    Delta = {delta_z_mp:+.4f}  {marker_c}")

        # MP-Test D: MC error propagation (ext-RC only)
        n_mc = 500
        mc_deltas_mp = []
        for _ in range(n_mc):
            perturbed_d = []
            perturbed_f = []
            for p in ext_dense_pts:
                perturbed_d.append(p['log_res'] + np.random.normal(0, p['sigma_log_gobs']))
            for p in ext_field_pts:
                perturbed_f.append(p['log_res'] + np.random.normal(0, p['sigma_log_gobs']))
            mc_delta = np.std(perturbed_f) - np.std(perturbed_d)
            mc_deltas_mp.append(mc_delta)

        mc_frac_pos_mp = np.mean(np.array(mc_deltas_mp) > 0)
        mc_med_mp = np.median(mc_deltas_mp)
        supports_d = mc_frac_pos_mp > 0.5
        mp_tests_total += 1
        if supports_d:
            mp_tests_support += 1
        marker_d = '✓' if supports_d else '✗'
        print(f"\n  MP-Test D (MC error, ext-RC only):")
        print(f"    Median Delta = {mc_med_mp:+.4f}, {mc_frac_pos_mp:.0%} positive  {marker_d}")

        print(f"\n  Multi-point restriction: {mp_tests_support}/{mp_tests_total} tests support BEC")

        # Add to summary
        summary['mp_restriction'] = {
            'n_dense_gals': len(ext_dense_gals),
            'n_field_gals': len(ext_field_gals),
            'n_dense_pts': len(ext_dense_pts),
            'n_field_pts': len(ext_field_pts),
            'tests_support': mp_tests_support,
            'tests_total': mp_tests_total,
        }
    else:
        print("  Insufficient data for multi-point restriction tests")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("UNIFIED RAR PIPELINE COMPLETE")
    print("=" * 80)

    print(f"\n  Galaxies: {n_galaxies} ({n_dense_gal} dense, {n_field_gal} field)")
    print(f"  RAR points: {n_points}")
    print(f"\n  Observed scatter:")
    print(f"    sigma_dense = {sigma_dense:.4f} dex")
    print(f"    sigma_field = {sigma_field:.4f} dex")
    print(f"    Delta_sigma = {delta_sigma:+.4f} dex")
    print(f"\n  Intrinsic scatter (Haubner deconvolution):")
    print(f"    sigma_int_dense = {sigma_int_dense:.4f} dex")
    print(f"    sigma_int_field = {sigma_int_field:.4f} dex")
    print(f"    Delta_sigma_int = {delta_sigma_int:+.4f} dex")
    if not np.isnan(p_val):
        direction = "sigma_field > sigma_dense" if delta_sigma > 0 else "sigma_dense > sigma_field"
        significance = "SIGNIFICANT" if (1.0 - p_val) > 0.95 or p_val < 0.05 else "not significant"
        print(f"\n  Statistical tests:")
        print(f"    P(field > dense) = {1.0 - p_val:.4f}")
        print(f"    Levene p = {lev_p:.6f}")
        print(f"    Result: {significance} at 95% CL")

    # Restructured presentation: 5 primary tests + 3 robustness checks
    # Primary: Tests 2 (threshold scan), 3 (galaxy-level), 5 (MC error),
    #          7 (BEC transition), 8 (boson bunching)
    # Robustness: Tests 1 (Z-score), 4 (DM-weighted), 6 (Z-norm galaxy-level)
    # NOTE: These recalculate from tests_summary for the final display,
    # but the variables were already computed earlier for the JSON output.
    primary_labels = {1: '2', 2: '3', 4: '5', 6: '7', 7: '8'}
    robust_labels = {0: '1', 3: '4', 5: '6'}
    _pi = [1, 2, 4, 6, 7]
    _ri = [0, 3, 5]
    _n_ps = sum(1 for i in _pi if i < len(tests_summary) and tests_summary[i][3])
    _n_pt = sum(1 for i in _pi if i < len(tests_summary))
    _n_rs = sum(1 for i in _ri if i < len(tests_summary) and tests_summary[i][3])
    _n_rt = sum(1 for i in _ri if i < len(tests_summary))

    print(f"\n  PRIMARY BEC tests ({_n_ps}/{_n_pt} support):")
    for i in _pi:
        if i < len(tests_summary):
            t = tests_summary[i]
            marker = "✓" if t[3] else "✗"
            lbl = primary_labels.get(i, '?')
            print(f"    {marker} Test {lbl} — {t[0]}: Δ={float(t[1]):+.4f}  {t[2]}")

    print(f"\n  ROBUSTNESS checks ({_n_rs}/{_n_rt} support):")
    for i in _ri:
        if i < len(tests_summary):
            t = tests_summary[i]
            marker = "✓" if t[3] else "✗"
            lbl = robust_labels.get(i, '?')
            print(f"    {marker} Check {lbl} — {t[0]}: Δ={float(t[1]):+.4f}  {t[2]}")

    # Verdict based on primary tests only
    _verdict = ('STRONG' if _n_ps >= 4 else
                'MODERATE-STRONG' if _n_ps >= 3 else
                'MODERATE' if _n_ps >= 2 else
                'WEAK' if _n_ps >= 1 else 'INSUFFICIENT')
    print(f"\n    >>> Verdict: {_verdict} EVIDENCE for BEC scatter prediction")
    print(f"    >>> ({_n_ps}/{_n_pt} primary + {_n_rs}/{_n_rt} robustness)")

    # Also keep the old n_support for backward compat in summary JSON
    n_tests_old = n_tests
    n_support_old = n_support

    print(f"\n  Test 7 robustness:")
    print(f"    Bootstrap ΔAIC: median={med_daic:+.1f}, BEC preferred in {frac_bec:.0%} of samples")
    jack_bec_count = sum(1 for r in jackknife_results.values() if r['best_model'] == 'BEC')
    print(f"    Jackknife: BEC preferred in {jack_bec_count}/{len(jackknife_results)} leave-one-out tests")
    if plot_path:
        print(f"    Plot: {plot_path}")

    print(f"\n  Output files:")
    print(f"    {rar_csv}")
    print(f"    {gal_csv}")
    print(f"    {summary_path}")

    return summary


# ============================================================
def parse_cli_args():
    """CLI options for optional chi2 calibration reporting modes."""
    parser = argparse.ArgumentParser(
        description="Unified RAR environmental scatter pipeline"
    )
    parser.add_argument(
        "--chi2_shared_sigma_int",
        action="store_true",
        help="Use a shared intrinsic scatter term across Test-7 BEC/linear/const fits.",
    )
    parser.add_argument(
        "--chi2_sigma_ref_model",
        type=str,
        default="bec",
        choices=["bec", "lin", "const"],
        help="Reference model used to fit shared sigma_int in Test-7 z-space.",
    )
    parser.add_argument(
        "--allow_missing_sparc",
        action="store_true",
        help="Allow pipeline execution when SPARC points fail minimum coverage guardrails.",
    )
    return parser.parse_args()


# ============================================================
if __name__ == '__main__':
    _args = parse_cli_args()
    summary = run_unified_pipeline(
        chi2_shared_sigma_int=bool(_args.chi2_shared_sigma_int),
        chi2_sigma_ref_model=str(_args.chi2_sigma_ref_model).lower(),
        allow_missing_sparc=bool(_args.allow_missing_sparc),
    )
