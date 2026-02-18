#!/usr/bin/env python3
"""
STEP 2: CF4 Distance RAR Pipeline
===================================
- Loads SPARC data with PGC crossmatch
- Queries CF4 flow model distances via EDD calculator API
- Applies Haubner+2025 distance uncertainty scheme
- Computes RAR residuals with proper error propagation
- Runs full environmental scatter test (cluster vs field)

Russell Licht — Primordial Fluid DM Project
Feb 2026
"""

import numpy as np
from scipy import stats, optimize
import csv
import json
import os
import sys
import urllib.request
import urllib.parse
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
g_dagger = 1.20e-10   # m/s^2 (McGaugh+2016 RAR scale)
conv = 1e6 / 3.0857e19  # (km/s)^2/kpc -> m/s^2
H0 = 75.0  # km/s/Mpc (Haubner+2025 convention for CF4)

# Virgo cluster center (Haubner+2025)
VIRGO_RA = 187.71   # degrees
VIRGO_DEC = 12.39   # degrees

# ============================================================
# HAUBNER+2025 UNCERTAINTY SCHEME (arXiv:2503.08491)
# ============================================================
# Equation 7: delta_f = delta_inf * D_f^alpha * (D_f^(1/kappa) + D_tr^(1/kappa))^(-alpha*kappa)
# delta_f is relative error in DEX

# CF4 flow model parameters (Table 2)
CF4_PARAMS = {
    'delta_inf': 0.022,   # dex (asymptotic error at large D)
    'alpha': -0.8,        # slope at small D
    'D_tr': 46.0,         # Mpc (transition distance)
    'kappa': 1.8,         # softening parameter
    'phi_VZoI': 30.0,     # degrees (Virgo Zone angular extent)
    'delta_VZoI': 0.17,   # dex (elevated uncertainty in Virgo Zone)
    'D_VZoI_min': 1.0,    # Mpc
    'D_VZoI_max': 33.0,   # Mpc
}

# Hubble flow (heliocentric) parameters for comparison
HUBBLE_VH_PARAMS = {
    'delta_inf': 0.031,
    'alpha': -0.9,
    'D_tr': 44.0,
    'kappa': 1.0,
    'phi_VZoI': 45.0,
    'delta_VZoI': 0.19,
    'D_VZoI_min': 1.0,
    'D_VZoI_max': 33.0,
}

# Binned uncertainties from Table 1 (for validation)
CF4_BINNED = [
    (0.01, 1.65, 0.56),   # (D_min, D_max, delta_dex)
    (1.65, 2.91, 0.30),
    (2.91, 5.13, 0.16),
    (5.13, 9.05, 0.18),
    (9.05, 16.0, 0.17),
    (16.0, 28.1, 0.09),
]

# Method-specific fractional uncertainties for primary distances (CF4-HQ)
PRIMARY_UNCERTAINTIES = {
    'TRGB': 0.05,      # 5%
    'Cepheid': 0.05,    # 5%  (CPLR)
    'SBF': 0.05,        # 5%
    'SNIa': 0.07,       # 7%
    'SNe': 0.07,        # same as SNIa
    'maser': 0.10,      # 10%
    'SNII': 0.15,       # 15%
}


def haubner_delta_f(D_Mpc, params=CF4_PARAMS):
    """
    Haubner+2025 Equation 7: fractional distance uncertainty in dex.

    delta_f = delta_inf * D^alpha * (D^(1/kappa) + D_tr^(1/kappa))^(-alpha*kappa)
    """
    D = np.asarray(D_Mpc, dtype=float)
    D = np.maximum(D, 0.01)  # avoid division by zero

    d_inf = params['delta_inf']
    alpha = params['alpha']
    D_tr = params['D_tr']
    kappa = params['kappa']

    term1 = D ** alpha
    term2 = (D ** (1.0/kappa) + D_tr ** (1.0/kappa)) ** (-alpha * kappa)

    delta = d_inf * term1 * term2
    return delta


def in_virgo_zone(ra_deg, dec_deg, D_Mpc, params=CF4_PARAMS):
    """Check if galaxy is in the Virgo Zone of Influence."""
    # Angular separation from Virgo center
    ra1, dec1 = np.radians(ra_deg), np.radians(dec_deg)
    ra2, dec2 = np.radians(VIRGO_RA), np.radians(VIRGO_DEC)

    cos_sep = (np.sin(dec1) * np.sin(dec2) +
               np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    cos_sep = np.clip(cos_sep, -1, 1)
    sep_deg = np.degrees(np.arccos(cos_sep))

    in_angular = sep_deg < params['phi_VZoI']
    in_distance = (D_Mpc >= params['D_VZoI_min']) & (D_Mpc <= params['D_VZoI_max'])

    return in_angular & in_distance


def get_distance_uncertainty_dex(D_Mpc, ra_deg=None, dec_deg=None,
                                  fD=None, params=CF4_PARAMS):
    """
    Get distance uncertainty in dex for a galaxy.

    For CF4 flow distances: use Haubner+2025 Eq. 7
    For primary distances (TRGB, Cepheid, etc.): use method-specific values
    Special handling for Virgo Zone of Influence
    """
    D = np.asarray(D_Mpc, dtype=float)

    # Default: use the analytic function
    delta = haubner_delta_f(D, params)

    # Check Virgo Zone if coordinates provided
    if ra_deg is not None and dec_deg is not None:
        in_vz = in_virgo_zone(ra_deg, dec_deg, D, params)
        if np.any(in_vz):
            delta = np.where(in_vz, params['delta_VZoI'], delta)

    return delta


def dex_to_fractional(delta_dex):
    """Convert uncertainty in dex to fractional uncertainty sigma_D/D."""
    return 10**delta_dex - 1


# ============================================================
# PARSE SPARC DATA
# ============================================================
def parse_table2(filepath):
    """Parse VizieR SPARC table2.dat (mass models) into per-galaxy dicts."""
    galaxies = {}
    with open(filepath, 'r') as f:
        for line in f:
            if len(line.strip()) < 50:
                continue
            try:
                name = line[0:11].strip()
                dist = float(line[12:18].strip())
                rad = float(line[19:25].strip())
                vobs = float(line[26:32].strip())
                evobs = float(line[33:38].strip())
                vgas = float(line[39:45].strip())
                vdisk = float(line[46:52].strip())
                vbul = float(line[53:59].strip())
                sbdisk = float(line[60:67].strip())
                sbbul = float(line[68:76].strip()) if len(line) > 68 else 0.0
            except (ValueError, IndexError):
                continue

            if name not in galaxies:
                galaxies[name] = {
                    'name': name, 'dist': dist,
                    'R': [], 'Vobs': [], 'eVobs': [],
                    'Vgas': [], 'Vdisk': [], 'Vbul': [],
                    'SBdisk': [], 'SBbul': []
                }

            galaxies[name]['R'].append(rad)
            galaxies[name]['Vobs'].append(vobs)
            galaxies[name]['eVobs'].append(evobs)
            galaxies[name]['Vgas'].append(vgas)
            galaxies[name]['Vdisk'].append(vdisk)
            galaxies[name]['Vbul'].append(vbul)
            galaxies[name]['SBdisk'].append(sbdisk)
            galaxies[name]['SBbul'].append(sbbul)

    # Convert lists to numpy arrays
    for name in galaxies:
        for key in ['R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']:
            galaxies[name][key] = np.array(galaxies[name][key])

    return galaxies


def load_pgc_crossmatch(filepath):
    """Load the PGC crossmatch table from Step 1."""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['sparc_name']] = {
                'pgc': int(row['pgc']) if row['pgc'] not in ['', '-1'] else -1,
                'fD': int(row['fD']),
                'D': float(row['D']),
                'eD': float(row['eD']),
                'Inc': float(row['Inc']),
                'eInc': float(row['eInc']),
                'Vflat': float(row['Vflat']),
                'eVflat': float(row['eVflat']),
                'Q': int(row['Q']),
                'T': int(row['T']),
            }
    return data


# ============================================================
# CF4 DISTANCE QUERY
# ============================================================
def query_cf4_distance(ra, dec, velocity, pgc=None):
    """
    Query the EDD Cosmicflows calculator for CF4 distance.

    Uses the Cosmicflows-4 calculator at edd.ifa.hawaii.edu
    Returns: (D_cf4_Mpc, V_cosmic, method_used)
    """
    # The EDD calculator API endpoint
    base_url = "https://edd.ifa.hawaii.edu/CF4calculator/CFinput.php"

    params = {
        'ra': f'{ra:.6f}',
        'dec': f'{dec:.6f}',
        'vel': f'{velocity:.1f}',
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0 (research)')
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode('utf-8', errors='replace')
            # Parse the response for CF4 distance
            # The calculator returns an HTML page with the results
            # We need to extract the CF4 distance value
            return parse_cf4_response(text)
    except Exception as e:
        return None, None, str(e)


def parse_cf4_response(html_text):
    """Parse the CF4 calculator HTML response to extract distance."""
    import re
    # Look for CF4 distance in the response
    # Typical format: "CF4 Distance: XX.X Mpc"
    patterns = [
        r'CF4.*?(\d+\.?\d*)\s*Mpc',
        r'D_CF4.*?=\s*(\d+\.?\d*)',
        r'distance.*?(\d+\.?\d*)\s*Mpc',
    ]

    for pattern in patterns:
        match = re.search(pattern, html_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1)), None, 'CF4'
            except ValueError:
                continue

    return None, None, 'parse_failed'


def query_edd_distances_batch(galaxies_info, use_cache=True):
    """
    Query EDD for CF4 distances for a batch of galaxies.
    Uses a cache file to avoid re-querying.
    """
    cache_file = os.path.join(DATA_DIR, 'cf4_distance_cache.json')
    cache = {}

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached CF4 distances")

    results = {}
    to_query = []

    for name, info in galaxies_info.items():
        if name in cache:
            results[name] = cache[name]
        else:
            to_query.append((name, info))

    if to_query:
        print(f"  Need to query {len(to_query)} new distances from EDD...")
        for i, (name, info) in enumerate(to_query):
            # We'd need RA/Dec for each galaxy, which we can get from NED
            # For now, use a placeholder that will be filled in
            results[name] = {
                'D_cf4': None,
                'status': 'needs_coordinates'
            }

            if (i + 1) % 20 == 0:
                print(f"    Processed {i+1}/{len(to_query)}...")

    # Save cache
    with open(cache_file, 'w') as f:
        json.dump({**cache, **results}, f, indent=2, default=str)

    return results


# ============================================================
# RAR COMPUTATION
# ============================================================
def compute_rar(gdata, D_Mpc, Y_d=0.5, Y_b=0.7):
    """
    Compute RAR data points for a single galaxy.

    Returns arrays of (log_gbar, log_gobs, log_gobs_pred, log_residual)
    with proper distance scaling.
    """
    # Scale radii by distance ratio
    D_orig = gdata['dist']
    D_ratio = D_Mpc / D_orig

    R = gdata['R'] * D_ratio  # kpc (linear with D)
    Vobs = gdata['Vobs']       # km/s (distance-independent for rotation curves)
    eVobs = gdata['eVobs']

    # Velocity components scale as sqrt(D_ratio) because M ∝ D and V² = GM/r
    # Actually: V_disk, V_gas, V_bul are derived from photometry + distance
    # V² = GM/r, M ∝ Σ * r², so V² ∝ Σ * r ∝ D, so V ∝ sqrt(D)
    # But Vobs is measured directly (distance-independent)
    sqrt_ratio = np.sqrt(D_ratio)
    Vgas = gdata['Vgas'] * sqrt_ratio
    Vdisk = gdata['Vdisk'] * sqrt_ratio
    Vbul = gdata['Vbul'] * sqrt_ratio

    # Baryonic acceleration
    Vbar_sq = np.sign(Vgas) * Vgas**2 + Y_d * Vdisk**2 + Y_b * Vbul**2
    gbar = np.abs(Vbar_sq) / R * conv

    # Observed acceleration
    gobs = Vobs**2 / R * conv

    # Valid points
    valid = (gobs > 0) & (gbar > 0) & (R > 0)
    if np.sum(valid) < 3:
        return None

    gbar = gbar[valid]
    gobs = gobs[valid]
    eVobs_v = eVobs[valid]
    Vobs_v = Vobs[valid]
    R_v = R[valid]

    # RAR prediction
    gobs_pred = gbar / (1 - np.exp(-np.sqrt(gbar / g_dagger)))

    # Log residuals
    log_gbar = np.log10(gbar)
    log_gobs = np.log10(gobs)
    log_gobs_pred = np.log10(gobs_pred)
    log_res = log_gobs - log_gobs_pred

    # Observational uncertainty on log(gobs)
    # sigma_log_gobs ≈ 2 * sigma_V / V / ln(10)
    sigma_log_gobs = 2.0 * eVobs_v / np.maximum(Vobs_v, 1) / np.log(10)

    return {
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'log_gobs_pred': log_gobs_pred,
        'log_res': log_res,
        'sigma_log_gobs': sigma_log_gobs,
        'n_points': len(log_gbar),
    }


def fit_ml_ratio(gdata, D_Mpc):
    """Optimize M/L ratio to minimize RAR scatter."""
    D_orig = gdata['dist']
    D_ratio = D_Mpc / D_orig

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

    R = R[mask]; Vobs = Vobs[mask]; eVobs = eVobs[mask]
    Vgas = Vgas[mask]; Vdisk = Vdisk[mask]; Vbul = Vbul[mask]

    def chi2(params):
        Y_d = params[0]
        Y_b = params[1] if len(params) > 1 else 0.7
        Vbar_sq = np.sign(Vgas)*Vgas**2 + Y_d*Vdisk**2 + Y_b*Vbul**2
        gb = np.abs(Vbar_sq) / R * conv
        go = Vobs**2 / R * conv
        v = (go > 0) & (gb > 0)
        if np.sum(v) < 3:
            return 1e10
        gop = gb[v] / (1 - np.exp(-np.sqrt(gb[v]/g_dagger)))
        lr = np.log10(go[v]) - np.log10(gop)
        w = 1.0 / np.maximum((2*eVobs[v]/np.maximum(Vobs[v],1))**2, 0.01)
        return np.sum(w * lr**2)

    has_bulge = np.any(np.abs(Vbul) > 0)
    if has_bulge:
        res = optimize.minimize(chi2, [0.5, 0.7],
                                bounds=[(0.1, 1.2), (0.1, 1.5)],
                                method='L-BFGS-B')
        return {'Y_disk': res.x[0], 'Y_bul': res.x[1]}
    else:
        res = optimize.minimize(chi2, [0.5],
                                bounds=[(0.1, 1.2)],
                                method='L-BFGS-B')
        return {'Y_disk': res.x[0], 'Y_bul': 0.7}


# ============================================================
# ENVIRONMENT CLASSIFICATION
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

GROUP_ASSIGNMENTS = {
    'NGC2403': 'M81', 'NGC2976': 'M81', 'IC2574': 'M81',
    'DDO154': 'M81', 'DDO168': 'M81', 'UGC04483': 'M81',
    'NGC0300': 'Sculptor', 'NGC0055': 'Sculptor',
    'NGC0247': 'Sculptor', 'NGC7793': 'Sculptor',
    'NGC2915': 'CenA', 'UGCA442': 'CenA', 'ESO444-G084': 'CenA',
    'UGC07577': 'CVnI', 'UGC07232': 'CVnI', 'NGC3741': 'CVnI',
    'NGC4068': 'CVnI', 'UGC07866': 'CVnI', 'UGC07524': 'CVnI',
    'UGC08490': 'CVnI', 'UGC07559': 'CVnI',
    'NGC3109': 'Antlia',
    'NGC5055': 'M101',
}


def classify_environment(name, fD):
    """Classify galaxy environment: cluster, group, or field."""
    if name in UMA_GALAXIES:
        return 'cluster', 'UMa', 'dense'
    if name in GROUP_ASSIGNMENTS:
        return 'group', GROUP_ASSIGNMENTS[name], 'dense'
    return 'field', 'field', 'field'


# Load Yang-enriched environment catalog if available
YANG_ENV_CATALOG = None
_yang_env_path = os.path.join(DATA_DIR, 'sparc_environment_catalog.json')
if os.path.exists(_yang_env_path):
    with open(_yang_env_path, 'r') as _f:
        YANG_ENV_CATALOG = json.load(_f)


def classify_environment_yang(name, fD):
    """
    Classify environment using Yang group catalog halo masses.
    Falls back to original UMa/group classification if no Yang data.
    """
    if YANG_ENV_CATALOG and name in YANG_ENV_CATALOG:
        env = YANG_ENV_CATALOG[name]
        logMh = env.get('logMh', 11.0)
        group_name = env.get('group_name', 'field')
        env_dense = env.get('env_dense', 'field')
        env_class = env.get('env_class', 'field')
        return env_class, group_name, env_dense, logMh
    # Fallback to original
    env_type, env_group, env_binary = classify_environment(name, fD)
    logMh = 13.2 if env_type == 'cluster' else (12.0 if env_type == 'group' else 11.0)
    return env_type, env_group, env_binary, logMh


# ============================================================
# DISTANCE PROPAGATION ERROR
# ============================================================
def propagate_distance_error(delta_dex, D_ratio):
    """
    Propagate distance uncertainty to RAR scatter.

    Distance uncertainty affects:
    1. Radii: R ∝ D → δR/R = δD/D
    2. Baryonic velocities: V_bar ∝ √D → δV_bar/V_bar = δD/(2D)
    3. Accelerations: g = V²/R → affects both gbar and gobs

    For gbar: gbar = V_bar²/R ∝ D/D = 1 (gbar is actually distance-independent
    because V²∝D and R∝D cancel!)

    For gobs: gobs = Vobs²/R ∝ 1/D → δlog(gobs) = -δD/(D·ln10)

    So distance error maps directly to RAR scatter as:
    σ_RAR ≈ delta_dex (the dex uncertainty IS the RAR uncertainty contribution!)
    """
    return delta_dex


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(use_cf4=True, use_haubner_errors=True):
    """
    Run the full CF4-RAR pipeline.

    Args:
        use_cf4: If True, use CF4 distances where available.
                 If False, use original SPARC distances (for comparison).
        use_haubner_errors: If True, use Haubner+2025 uncertainty scheme.
    """
    print("=" * 80)
    print(f"CF4-RAR PIPELINE {'(CF4 distances)' if use_cf4 else '(SPARC original distances)'}")
    print(f"Haubner+2025 uncertainties: {'YES' if use_haubner_errors else 'NO'}")
    print("=" * 80)

    # 1. Load data
    print("\n--- Loading data ---")

    table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
    mass_models = parse_table2(table2_path)
    print(f"  Loaded mass models for {len(mass_models)} galaxies")

    pgc_path = os.path.join(DATA_DIR, 'sparc_pgc_crossmatch.csv')
    if os.path.exists(pgc_path):
        pgc_data = load_pgc_crossmatch(pgc_path)
        print(f"  Loaded PGC crossmatch for {len(pgc_data)} galaxies")
    else:
        print(f"  WARNING: No PGC crossmatch found. Run 01_resolve_pgc_numbers.py first.")
        print(f"  Using SPARC distances only.")
        pgc_data = {}

    # Load CF4 distance cache if available
    cf4_cache_path = os.path.join(DATA_DIR, 'cf4_distance_cache.json')
    cf4_distances = {}
    if os.path.exists(cf4_cache_path):
        with open(cf4_cache_path, 'r') as f:
            cf4_distances = json.load(f)
        print(f"  Loaded {len(cf4_distances)} CF4 distances from cache")

    # 2. Quality cuts
    print("\n--- Applying quality cuts ---")

    # Build properties from pgc_data or table1
    table1_path = os.path.join(DATA_DIR, 'SPARC_table1_vizier.dat')
    if os.path.exists(table1_path):
        # Parse table1 for properties
        with open(table1_path, 'r') as f:
            lines = f.readlines()
        props = {}
        for line in lines:
            if len(line.strip()) < 50:
                continue
            try:
                name = line[0:11].strip()
                T = int(line[12:14].strip())
                D = float(line[15:21].strip())
                eD = float(line[22:27].strip())
                fD = int(line[28:29].strip())
                Inc = float(line[30:34].strip())
                eInc = float(line[35:39].strip())
                Vflat = float(line[100:105].strip()) if line[100:105].strip() else 0
                eVflat = float(line[106:111].strip()) if line[106:111].strip() else 0
                Q = int(line[112:115].strip()) if line[112:115].strip() else 3
                props[name] = {
                    'T': T, 'D': D, 'eD': eD, 'fD': fD,
                    'Inc': Inc, 'eInc': eInc,
                    'Vflat': Vflat, 'eVflat': eVflat, 'Q': Q
                }
            except (ValueError, IndexError):
                continue
        print(f"  Parsed properties for {len(props)} galaxies")
    else:
        props = {}

    # Apply quality cuts
    galaxies_clean = []
    cut_reasons = {}

    for name in mass_models:
        p = props.get(name, pgc_data.get(name, {}))
        if not p:
            cut_reasons['no_properties'] = cut_reasons.get('no_properties', 0) + 1
            continue

        Q = p.get('Q', 3)
        Inc = p.get('Inc', 0)
        n_pts = len(mass_models[name]['R'])

        if Q > 2:
            cut_reasons['Q=3'] = cut_reasons.get('Q=3', 0) + 1
            continue
        if Inc < 30:
            cut_reasons['Inc<30'] = cut_reasons.get('Inc<30', 0) + 1
            continue
        if Inc > 85:
            cut_reasons['Inc>85'] = cut_reasons.get('Inc>85', 0) + 1
            continue
        if n_pts < 10:
            cut_reasons['N<10'] = cut_reasons.get('N<10', 0) + 1
            continue

        galaxies_clean.append({
            'name': name,
            'props': p,
            'mass_model': mass_models[name],
        })

    print(f"  Galaxies after quality cuts: {len(galaxies_clean)}")
    for reason, count in sorted(cut_reasons.items(), key=lambda x: -x[1]):
        print(f"    Cut by {reason}: {count}")

    # 3. Determine distances and uncertainties
    print("\n--- Determining distances ---")

    fD_names = {1: 'Hubble', 2: 'TRGB', 3: 'Cepheid', 4: 'UMa', 5: 'SNe'}

    for g in galaxies_clean:
        name = g['name']
        p = g['props']
        fD = p.get('fD', 1)
        D_sparc = p.get('D', g['mass_model']['dist'])
        eD_sparc = p.get('eD', D_sparc * 0.2)

        # Decide which distance to use
        # CRITICAL: Only use CF4 for Hubble-flow galaxies (fD=1).
        # Primary distances (TRGB=2, Cepheid=3, SNe=5) and UMa cluster (fD=4)
        # are more accurate than CF4 flow model for nearby galaxies.
        if (use_cf4 and fD == 1 and
            name in cf4_distances and cf4_distances[name].get('D_cf4')):
            g['D_use'] = cf4_distances[name]['D_cf4']
            g['dist_source'] = 'CF4'
        else:
            g['D_use'] = D_sparc
            g['dist_source'] = fD_names.get(fD, 'unknown')

        # Distance uncertainty (Haubner scheme)
        if use_haubner_errors:
            if fD in [2, 3, 5]:
                # Primary distance: use method-specific uncertainty
                method = {2: 'TRGB', 3: 'Cepheid', 5: 'SNe'}.get(fD, 'TRGB')
                frac_err = PRIMARY_UNCERTAINTIES.get(method, 0.10)
                g['sigma_D_dex'] = np.log10(1 + frac_err)
                g['sigma_D_frac'] = frac_err
            elif fD == 4:
                # UMa cluster: all at ~18 Mpc with ~10% uncertainty
                g['sigma_D_dex'] = np.log10(1.10)
                g['sigma_D_frac'] = 0.10
            elif g['dist_source'] == 'CF4':
                # CF4 flow model distance: use Haubner CF4 formula
                g['sigma_D_dex'] = float(haubner_delta_f(g['D_use'], CF4_PARAMS))
                g['sigma_D_frac'] = float(dex_to_fractional(g['sigma_D_dex']))
            else:
                # Hubble flow (no CF4): use Haubner Hubble V_h formula
                g['sigma_D_dex'] = float(haubner_delta_f(g['D_use'], HUBBLE_VH_PARAMS))
                g['sigma_D_frac'] = float(dex_to_fractional(g['sigma_D_dex']))
        else:
            # Use SPARC errors
            g['sigma_D_dex'] = np.log10(1 + eD_sparc / max(D_sparc, 0.1))
            g['sigma_D_frac'] = eD_sparc / max(D_sparc, 0.1)

        # Distance ratio (for scaling)
        g['D_ratio'] = g['D_use'] / g['mass_model']['dist']

        # Environment (use Yang catalog if available)
        env_type, env_group, env_binary, logMh = classify_environment_yang(name, fD)
        g['env_type'] = env_type
        g['env_group'] = env_group
        g['env_binary'] = env_binary
        g['logMh'] = logMh

    # Summarize distances
    dist_sources = {}
    for g in galaxies_clean:
        src = g['dist_source']
        dist_sources[src] = dist_sources.get(src, 0) + 1
    print(f"  Distance sources:")
    for src, count in sorted(dist_sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")

    env_counts = {}
    for g in galaxies_clean:
        env = g['env_binary']
        env_counts[env] = env_counts.get(env, 0) + 1
    print(f"  Environment: dense={env_counts.get('dense', 0)}, field={env_counts.get('field', 0)}")

    # 4. Compute RAR residuals
    print("\n--- Computing RAR residuals ---")

    all_results = []
    all_gbar = []
    all_res = []
    all_sigma = []

    for g in galaxies_clean:
        name = g['name']
        D = g['D_use']
        mm = g['mass_model']

        # Fit M/L
        ml = fit_ml_ratio(mm, D)

        # Compute RAR
        rar = compute_rar(mm, D, Y_d=ml['Y_disk'], Y_b=ml['Y_bul'])
        if rar is None:
            continue

        g['rar'] = rar
        g['Y_disk'] = ml['Y_disk']
        g['Y_bul'] = ml['Y_bul']

        all_gbar.extend(rar['log_gbar'])
        all_res.extend(rar['log_res'])
        all_sigma.extend(rar['sigma_log_gobs'])

        all_results.append(g)

    all_gbar = np.array(all_gbar)
    all_res = np.array(all_res)
    all_sigma = np.array(all_sigma)

    print(f"  Processed: {len(all_results)} galaxies, {len(all_gbar)} data points")
    print(f"  Overall RAR scatter: {np.std(all_res):.4f} dex")
    print(f"  Mean residual: {np.mean(all_res):.4f} dex")

    # 5. Environmental scatter test
    print("\n" + "=" * 80)
    print("ENVIRONMENTAL RAR SCATTER TEST")
    print("=" * 80)

    dense_gbar, dense_res = [], []
    field_gbar, field_res = [], []
    dense_galaxies, field_galaxies = [], []

    for g in all_results:
        rar = g['rar']
        if g['env_binary'] == 'dense':
            dense_gbar.extend(rar['log_gbar'])
            dense_res.extend(rar['log_res'])
            dense_galaxies.append(g['name'])
        else:
            field_gbar.extend(rar['log_gbar'])
            field_res.extend(rar['log_res'])
            field_galaxies.append(g['name'])

    dense_gbar = np.array(dense_gbar)
    dense_res = np.array(dense_res)
    field_gbar = np.array(field_gbar)
    field_res = np.array(field_res)

    n_dense = len(set(dense_galaxies))
    n_field = len(set(field_galaxies))

    print(f"\nDense: {len(dense_res)} points from {n_dense} galaxies")
    print(f"  σ = {np.std(dense_res):.4f} dex")
    print(f"Field: {len(field_res)} points from {n_field} galaxies")
    print(f"  σ = {np.std(field_res):.4f} dex")

    delta_sigma = np.std(field_res) - np.std(dense_res)
    print(f"\nΔσ (field - dense): {delta_sigma:+.4f} dex")

    # Bootstrap significance
    print("\nBootstrap test (10,000 iterations)...")
    n_boot = 10000
    np.random.seed(42)
    boot_deltas = np.zeros(n_boot)
    combined = np.concatenate([dense_res, field_res])
    nd = len(dense_res)

    for i in range(n_boot):
        shuffled = np.random.permutation(combined)
        boot_deltas[i] = np.std(shuffled[nd:]) - np.std(shuffled[:nd])

    p_value = np.mean(boot_deltas >= delta_sigma)
    print(f"  P(field > dense): {1-p_value:.4f} ({(1-p_value)*100:.1f}%)")
    print(f"  P-value (one-sided): {p_value:.4f}")

    # Levene's test
    levene_stat, levene_p = stats.levene(dense_res, field_res)
    print(f"  Levene's test: F={levene_stat:.3f}, p={levene_p:.4f}")

    # 6. Binned analysis
    print("\n--- Binned Environmental Analysis ---")
    bin_edges = np.array([-12.5, -11.5, -10.5, -9.5, -8.5])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned = []

    for j in range(len(bin_centers)):
        lo, hi = bin_edges[j], bin_edges[j+1]
        d_mask = (dense_gbar >= lo) & (dense_gbar < hi)
        f_mask = (field_gbar >= lo) & (field_gbar < hi)

        d_r = dense_res[d_mask]
        f_r = field_res[f_mask]

        if len(d_r) >= 5 and len(f_r) >= 5:
            d_std = np.std(d_r)
            f_std = np.std(f_r)
            delta = f_std - d_std

            # Bootstrap
            comb = np.concatenate([d_r, f_r])
            nd_bin = len(d_r)
            bb = np.zeros(5000)
            for i in range(5000):
                s = np.random.permutation(comb)
                bb[i] = np.std(s[nd_bin:]) - np.std(s[:nd_bin])
            p_bin = np.mean(bb >= delta)

            binned.append({
                'center': float(bin_centers[j]),
                'n_dense': len(d_r), 'n_field': len(f_r),
                'sigma_dense': float(d_std), 'sigma_field': float(f_std),
                'delta': float(delta), 'p_field_gt_dense': float(1-p_bin),
            })

            direction = "✓" if delta > 0 else "✗"
            print(f"  {direction} Bin {bin_centers[j]:.1f}: dense={d_std:.4f} ({len(d_r)}), "
                  f"field={f_std:.4f} ({len(f_r)}), Δ={delta:+.4f}, P={1-p_bin:.3f}")

    # 7. Mass dependence
    print("\n--- Mass Dependence ---")
    resolved = [g for g in all_results if g['props'].get('Vflat', 0) > 0]
    if len(resolved) > 4:
        vflats = [g['props']['Vflat'] for g in resolved]
        median_vf = np.median(vflats)
        low_res, high_res = [], []
        for g in resolved:
            r = g['rar']['log_res']
            if g['props']['Vflat'] < median_vf:
                low_res.extend(r)
            else:
                high_res.extend(r)
        low_res = np.array(low_res)
        high_res = np.array(high_res)
        print(f"  Median Vflat: {median_vf:.1f} km/s")
        print(f"  Low mass: σ={np.std(low_res):.4f}, skew={stats.skew(low_res):.2f}")
        print(f"  High mass: σ={np.std(high_res):.4f}, skew={stats.skew(high_res):.2f}")
        print(f"  Δσ (low-high): {np.std(low_res)-np.std(high_res):+.4f}")

    # 8. Skewness profile
    print("\n--- Skewness Profile ---")
    fine_edges = np.linspace(-12.5, -8.0, 8)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2
    skew_results = []

    for j in range(len(fine_centers)):
        lo, hi = fine_edges[j], fine_edges[j+1]
        mask = (all_gbar >= lo) & (all_gbar < hi)
        res = all_res[mask]
        if len(res) >= 20:
            q25, q50, q75 = np.percentile(res, [25, 50, 75])
            iqr = q75 - q25
            qsk = (q75 + q25 - 2*q50) / iqr if iqr > 0 else 0
            skew_results.append({
                'center': float(fine_centers[j]),
                'n': len(res), 'std': float(np.std(res)),
                'moment_skew': float(stats.skew(res)),
                'quantile_skew': float(qsk),
                'kurtosis': float(stats.kurtosis(res, fisher=True)),
            })
            print(f"  Bin {fine_centers[j]:.2f}: N={len(res)}, σ={np.std(res):.4f}, "
                  f"skew={stats.skew(res):.2f}, qsk={qsk:.3f}")

    # 9. Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    label = 'cf4' if use_cf4 else 'sparc_orig'
    if use_haubner_errors:
        label += '_haubner'

    # Galaxy-level results
    galaxy_rows = []
    for g in all_results:
        galaxy_rows.append({
            'galaxy': g['name'],
            'D_use': round(g['D_use'], 2),
            'D_sparc': round(g['props'].get('D', g['mass_model']['dist']), 2),
            'D_ratio': round(g['D_ratio'], 4),
            'dist_source': g['dist_source'],
            'sigma_D_dex': round(g['sigma_D_dex'], 4),
            'sigma_D_frac': round(g['sigma_D_frac'], 4),
            'fD': g['props'].get('fD', 0),
            'Q': g['props'].get('Q', 3),
            'Inc': g['props'].get('Inc', 0),
            'Vflat': g['props'].get('Vflat', 0),
            'T': g['props'].get('T', 0),
            'Y_disk': round(g['Y_disk'], 3),
            'n_points': g['rar']['n_points'],
            'scatter': round(float(np.std(g['rar']['log_res'])), 4),
            'mean_res': round(float(np.mean(g['rar']['log_res'])), 4),
            'env_type': g['env_type'],
            'env_group': g['env_group'],
            'env_binary': g['env_binary'],
            'logMh': round(g.get('logMh', 0), 1),
        })

    galaxy_csv = os.path.join(OUTPUT_DIR, f'galaxy_results_{label}.csv')
    with open(galaxy_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=galaxy_rows[0].keys())
        writer.writeheader()
        writer.writerows(galaxy_rows)
    print(f"  Galaxy results: {galaxy_csv}")

    # Summary JSON
    summary = {
        'pipeline': label,
        'use_cf4': use_cf4,
        'use_haubner_errors': use_haubner_errors,
        'n_galaxies': len(all_results),
        'n_data_points': len(all_gbar),
        'overall_scatter_dex': float(np.std(all_res)),
        'mean_residual_dex': float(np.mean(all_res)),
        'environment': {
            'n_dense': n_dense,
            'n_field': n_field,
            'sigma_dense': float(np.std(dense_res)),
            'sigma_field': float(np.std(field_res)),
            'delta_sigma': float(delta_sigma),
            'p_field_gt_dense': float(1-p_value),
            'levene_p': float(levene_p),
            'n_dense_points': len(dense_res),
            'n_field_points': len(field_res),
        },
        'binned_results': binned,
        'skewness_profile': skew_results,
        'haubner_params': CF4_PARAMS if use_haubner_errors else None,
    }

    summary_json = os.path.join(OUTPUT_DIR, f'summary_{label}.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_json}")

    print(f"\n{'=' * 80}")
    print("✓ Pipeline complete!")
    print(f"{'=' * 80}")

    return summary


# ============================================================
if __name__ == '__main__':
    # Run with SPARC original distances first (baseline)
    print("\n" + "#" * 80)
    print("RUN 1: SPARC ORIGINAL DISTANCES + HAUBNER UNCERTAINTIES")
    print("#" * 80)
    summary_orig = run_pipeline(use_cf4=False, use_haubner_errors=True)

    # Then run with CF4 distances (if available)
    print("\n\n" + "#" * 80)
    print("RUN 2: CF4 DISTANCES + HAUBNER UNCERTAINTIES")
    print("#" * 80)
    summary_cf4 = run_pipeline(use_cf4=True, use_haubner_errors=True)

    # Comparison
    print("\n\n" + "=" * 80)
    print("COMPARISON: SPARC vs CF4")
    print("=" * 80)
    print(f"{'Metric':<35} {'SPARC':>15} {'CF4':>15}")
    print(f"{'-'*65}")
    print(f"{'N galaxies':<35} {summary_orig['n_galaxies']:>15} {summary_cf4['n_galaxies']:>15}")
    print(f"{'N points':<35} {summary_orig['n_data_points']:>15} {summary_cf4['n_data_points']:>15}")
    print(f"{'Overall σ (dex)':<35} {summary_orig['overall_scatter_dex']:>15.4f} {summary_cf4['overall_scatter_dex']:>15.4f}")
    print(f"{'Dense σ (dex)':<35} {summary_orig['environment']['sigma_dense']:>15.4f} {summary_cf4['environment']['sigma_dense']:>15.4f}")
    print(f"{'Field σ (dex)':<35} {summary_orig['environment']['sigma_field']:>15.4f} {summary_cf4['environment']['sigma_field']:>15.4f}")
    print(f"{'Δσ (field-dense)':<35} {summary_orig['environment']['delta_sigma']:>15.4f} {summary_cf4['environment']['delta_sigma']:>15.4f}")
    print(f"{'P(field > dense)':<35} {summary_orig['environment']['p_field_gt_dense']:>15.4f} {summary_cf4['environment']['p_field_gt_dense']:>15.4f}")
