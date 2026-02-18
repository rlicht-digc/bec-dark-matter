#!/usr/bin/env python3
"""
PROBES × ALFALFA Gas-Corrected RAR Tightness Test
===================================================
Cross-matches PROBES rotation curves with ALFALFA alpha.100 (Haynes+2018)
HI masses to correct for missing gas in the RAR, then re-tests property
dependence and scatter uniformity.

The PROBES-only test (test_rar_tightness_probes.py) found:
  - PROBES scatter: 0.325 dex (vs SPARC 0.159 dex)
  - Mean offset: +0.25 dex from missing gas
  - R² = 2.3% (property-independent)

With gas correction from ALFALFA, we expect:
  - Scatter reduction (gas adds low-gbar signal)
  - Offset reduction (approaching SPARC baseline)
  - R² either stays low (BEC: universal) or increases (systematics unmasked)

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import numpy as np
import os
import re
import csv
import json
from scipy import stats
from scipy.stats import levene, ks_2samp, mannwhitneyu, spearmanr

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PROBES_DIR = os.path.join(DATA_DIR, 'probes')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10  # m/s^2
G_kpc = 4.302e-6     # (km/s)^2 kpc / Msun
conv = 1e6 / 3.0857e19  # (km/s)^2/kpc -> m/s^2

print("=" * 72)
print("PROBES × ALFALFA GAS-CORRECTED RAR TIGHTNESS TEST")
print("=" * 72)

# ============================================================
# STEP 1: Load ALFALFA alpha.100 (Haynes+2018)
# ============================================================
print("\n[1] Loading ALFALFA alpha.100...")

alfalfa_path = os.path.join(DATA_DIR, 'alfalfa_alpha100_haynes2018.tsv')

alfalfa = []  # list of dicts with ra_deg, dec_deg, logMHI, Vhel, Name
with open(alfalfa_path) as f:
    header = None
    past_sep = False
    for line in f:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('---') or re.match(r'^[-]+(\t[-]+)*$', s):
            past_sep = True
            continue
        parts = line.rstrip('\n').split('\t')
        if header is None:
            header = [p.strip() for p in parts]
            continue
        if not past_sep:
            continue
        row = {header[i]: parts[i].strip() if i < len(parts) else ''
               for i in range(len(header))}

        # Parse RA/Dec from HMS/DMS to degrees
        ra_str = row.get('RAJ2000', '').strip()
        dec_str = row.get('DEJ2000', '').strip()
        if not ra_str or not dec_str:
            continue

        try:
            ra_parts = ra_str.split()
            ra_deg = 15.0 * (float(ra_parts[0]) + float(ra_parts[1]) / 60.0
                             + float(ra_parts[2]) / 3600.0)
            dec_parts = dec_str.split()
            dec_sign = -1.0 if dec_str.strip().startswith('-') else 1.0
            dec_deg = dec_sign * (abs(float(dec_parts[0])) + float(dec_parts[1]) / 60.0
                                  + float(dec_parts[2]) / 3600.0)
        except (ValueError, IndexError):
            continue

        try:
            logMHI = float(row.get('logMHI', ''))
        except (ValueError, TypeError):
            continue

        try:
            vhel = float(row.get('Vhel', ''))
        except (ValueError, TypeError):
            vhel = np.nan

        try:
            dist = float(row.get('Dist', ''))
        except (ValueError, TypeError):
            dist = np.nan

        name = row.get('Name', '').strip()

        alfalfa.append({
            'ra': ra_deg,
            'dec': dec_deg,
            'logMHI': logMHI,
            'vhel': vhel,
            'dist': dist,
            'name': name,
        })

print(f"  Loaded {len(alfalfa)} ALFALFA alpha.100 sources")

# Build KD-tree for fast coordinate matching
from scipy.spatial import cKDTree

def radec_to_cart(ra, dec):
    """Convert RA/Dec (degrees) to unit-sphere Cartesian."""
    ra_r = np.radians(ra)
    dec_r = np.radians(dec)
    return np.array([np.cos(dec_r) * np.cos(ra_r),
                     np.cos(dec_r) * np.sin(ra_r),
                     np.sin(dec_r)])

alf_coords = np.array([radec_to_cart(a['ra'], a['dec']) for a in alfalfa])  # (N, 3)
alf_tree = cKDTree(alf_coords)
alf_logMHI = np.array([a['logMHI'] for a in alfalfa])
alf_vhel = np.array([a['vhel'] for a in alfalfa])

# ============================================================
# STEP 2: Load PROBES galaxies (non-SPARC only)
# ============================================================
print("\n[2] Loading PROBES galaxies...")

# Load SPARC galaxy list to exclude
sparc_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')
sparc_names = set()
if os.path.exists(sparc_path):
    with open(sparc_path) as f:
        in_data = False
        for line in f:
            s = line.strip()
            if s.startswith('===='):
                in_data = True
                continue
            if not in_data or not s:
                continue
            name = line[:12].strip().upper().replace(' ', '')
            if name and name[0].isalpha():
                sparc_names.add(name)
print(f"  {len(sparc_names)} SPARC galaxy names loaded for exclusion")

# Load PROBES main table
main_path = os.path.join(PROBES_DIR, 'main_table.csv')
with open(main_path) as f:
    lines = f.readlines()
header = lines[1].strip().split(',')
probes_galaxies = []
for line in lines[2:]:
    parts = line.strip().split(',')
    if len(parts) < len(header):
        continue
    row = {header[i]: parts[i].strip() for i in range(len(header))}

    name = row.get('name', '').strip()
    norm = name.upper().replace(' ', '')

    # Skip SPARC galaxies
    if norm in sparc_names:
        continue

    try:
        ra = float(row.get('RA', ''))
        dec = float(row.get('DEC', ''))
        dist = float(row.get('distance', ''))
    except (ValueError, TypeError):
        continue

    try:
        vhel = float(row.get('redshift_helio', ''))
    except (ValueError, TypeError):
        vhel = np.nan

    probes_galaxies.append({
        'name': name,
        'norm': norm,
        'ra': ra,
        'dec': dec,
        'dist': dist,
        'vhel': vhel,
    })

print(f"  {len(probes_galaxies)} non-SPARC PROBES galaxies")

# ============================================================
# STEP 3: Load structural parameters (Mstar, inclination)
# ============================================================
print("\n[3] Loading structural parameters...")

struct_path = os.path.join(PROBES_DIR, 'structural_parameters.csv')
struct_data = {}
with open(struct_path) as f:
    # Skip units comment line (starts with #)
    line1 = f.readline()
    # Read actual header
    header_line = f.readline()
    header_sp = header_line.strip().split(',')

    # Find column indices for Mstar and inclination
    # Try multiple aperture suffixes in priority order
    mstar_col = None
    inc_col = None
    for suffix in ['|Rlast:rc', '|Ri22:r', '|Ri22.5:r', '|Ri23:r']:
        if mstar_col is None:
            cname = f'Mstar{suffix}'
            if cname in header_sp:
                mstar_col = header_sp.index(cname)
        if inc_col is None:
            cname = f'inclination{suffix}'
            if cname in header_sp:
                inc_col = header_sp.index(cname)

    name_col = header_sp.index('name')
    print(f"  Using Mstar col: {header_sp[mstar_col] if mstar_col else 'NONE'}")
    print(f"  Using inclination col: {header_sp[inc_col] if inc_col else 'NONE'}")

    for line in f:
        parts = line.strip().split(',')
        if len(parts) <= max(name_col, mstar_col or 0, inc_col or 0):
            continue
        name = parts[name_col].strip()
        if not name:
            continue
        try:
            mstar = float(parts[mstar_col]) if mstar_col else np.nan
            inc_ba = float(parts[inc_col]) if inc_col else np.nan
        except (ValueError, TypeError):
            continue
        if mstar > 0 and not np.isnan(inc_ba):
            struct_data[name] = {
                'mstar': mstar,
                'inc_ba': inc_ba,
            }

print(f"  {len(struct_data)} galaxies with structural parameters")

# ============================================================
# STEP 4: Cross-match PROBES × ALFALFA
# ============================================================
print("\n[4] Cross-matching PROBES × ALFALFA...")

match_radius_deg = 2.0 / 60.0  # 2 arcmin
match_radius_cart = 2 * np.sin(np.radians(match_radius_deg / 2))
vel_tolerance = 200  # km/s

matched = 0
unmatched = 0
too_south = 0

for gal in probes_galaxies:
    cart = radec_to_cart(gal['ra'], gal['dec'])

    # Check if in ALFALFA footprint (rough: 0 < Dec < 36)
    if gal['dec'] < -2 or gal['dec'] > 38:
        gal['logMHI'] = np.nan
        gal['gas_source'] = 'out_of_footprint'
        too_south += 1
        continue

    # Query KD-tree
    dists_cart, idxs = alf_tree.query(cart, k=5, distance_upper_bound=match_radius_cart)

    best_idx = None
    best_sep = 999
    for d, i in zip(dists_cart, idxs):
        if i >= len(alfalfa):
            continue
        # Velocity check
        if not np.isnan(gal['vhel']) and not np.isnan(alf_vhel[i]):
            if abs(gal['vhel'] - alf_vhel[i]) > vel_tolerance:
                continue
        sep_deg = 2 * np.degrees(np.arcsin(d / 2))
        if sep_deg < best_sep:
            best_sep = sep_deg
            best_idx = i

    if best_idx is not None:
        gal['logMHI'] = alf_logMHI[best_idx]
        gal['gas_source'] = 'ALFALFA'
        gal['match_sep_arcsec'] = best_sep * 3600
        matched += 1
    else:
        gal['logMHI'] = np.nan
        gal['gas_source'] = 'no_match'
        unmatched += 1

print(f"  Matched: {matched}")
print(f"  No match (in footprint): {unmatched}")
print(f"  Out of ALFALFA footprint: {too_south}")

# ============================================================
# STEP 5: Load rotation curves and compute gas-corrected RAR
# ============================================================
print("\n[5] Computing gas-corrected RAR residuals...")

profiles_dir = os.path.join(PROBES_DIR, 'profiles', 'profiles')

results = {}
n_skipped_inc = 0
n_skipped_struct = 0
n_skipped_rc = 0
n_gas_corrected = 0
n_stars_only = 0


def rar_pred(gbar):
    """McGaugh+2016 RAR prediction."""
    gbar = np.asarray(gbar, dtype=float)
    x = np.sqrt(np.maximum(gbar, 1e-20) / g_dagger)
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-20)
    return gbar / denom


for gal in probes_galaxies:
    name = gal['name']

    # Get structural parameters
    sp = struct_data.get(name)
    if sp is None:
        n_skipped_struct += 1
        continue

    mstar = sp['mstar']
    inc_ba = sp['inc_ba']

    if np.isnan(mstar) or mstar <= 0 or np.isnan(inc_ba):
        n_skipped_struct += 1
        continue

    # Convert b/a to inclination in degrees
    ba = min(max(inc_ba, 0.01), 0.99)
    inc_deg = np.degrees(np.arccos(ba))

    # Quality cuts
    if inc_deg < 30 or inc_deg > 85:
        n_skipped_inc += 1
        continue

    # Load rotation curve
    rc_file = os.path.join(profiles_dir, f"{name}_rc.prof")
    if not os.path.exists(rc_file):
        n_skipped_rc += 1
        continue

    try:
        with open(rc_file) as f:
            rc_lines = f.readlines()
        # Skip comment line, then CSV header
        data_start = 0
        for i, line in enumerate(rc_lines):
            if line.strip().startswith('#'):
                data_start = i + 1
                continue
            if 'R' in line and 'V' in line:
                data_start = i + 1
                break

        R_arcsec = []
        V_kms = []
        Ve_kms = []
        for line in rc_lines[data_start:]:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                r = float(parts[0])
                v = float(parts[1])
                ve = float(parts[2])
            except ValueError:
                continue
            if r > 0 and v > 0 and ve > 0:  # positive side only
                R_arcsec.append(r)
                V_kms.append(v)
                Ve_kms.append(ve)

        if len(R_arcsec) < 5:
            n_skipped_rc += 1
            continue

    except Exception:
        n_skipped_rc += 1
        continue

    R_arcsec = np.array(R_arcsec)
    V_kms = np.array(V_kms)
    Ve_kms = np.array(Ve_kms)

    # Convert to physical units
    dist_mpc = gal['dist']
    R_kpc = R_arcsec * dist_mpc * np.pi / (180 * 3600) * 1000  # arcsec -> kpc

    # Correct velocity for inclination
    sin_i = np.sin(np.radians(inc_deg))
    V_rot = V_kms / sin_i
    Ve_rot = Ve_kms / sin_i

    # gobs = V^2/R in m/s^2
    gobs = V_rot**2 / R_kpc * conv

    # Compute baryonic mass: stars + gas (if available)
    logMs = np.log10(mstar)

    if not np.isnan(gal.get('logMHI', np.nan)):
        M_HI = 10**gal['logMHI']
        M_gas = 1.33 * M_HI  # helium correction
        M_baryonic = mstar + M_gas
        n_gas_corrected += 1
        gas_flag = 'corrected'
    else:
        M_baryonic = mstar  # stars only
        n_stars_only += 1
        gas_flag = 'stars_only'

    logMb = np.log10(M_baryonic)

    # Compute gbar using exponential disk model
    # R_50 from structural parameters if available, else use median R
    R50_kpc = np.median(R_kpc)
    Rd = R50_kpc / 1.678  # exponential scale length
    Rd = max(Rd, 0.1)

    log_gbar = np.zeros(len(R_kpc))
    log_gobs = np.zeros(len(R_kpc))
    log_res = np.zeros(len(R_kpc))
    valid = np.ones(len(R_kpc), dtype=bool)

    for j in range(len(R_kpc)):
        r = R_kpc[j]
        # Enclosed stellar mass (exponential disk)
        x = r / Rd
        M_enc_star = mstar * (1 - (1 + x) * np.exp(-x))

        # Gas: assume flat HI distribution (constant surface density out to last point)
        if gas_flag == 'corrected':
            # Simple: distribute gas uniformly within R_max
            R_max = R_kpc[-1]
            M_enc_gas = M_gas * min(r / R_max, 1.0)**2  # uniform disk
        else:
            M_enc_gas = 0.0

        M_enc = M_enc_star + M_enc_gas

        gbar_val = G_kpc * M_enc / r**2 * conv  # -> m/s^2

        if gbar_val <= 0 or gobs[j] <= 0:
            valid[j] = False
            continue

        lg_gbar = np.log10(gbar_val)
        lg_gobs = np.log10(gobs[j])
        lg_res = lg_gobs - np.log10(rar_pred(gbar_val))

        if abs(lg_res) > 1.5:
            valid[j] = False
            continue

        log_gbar[j] = lg_gbar
        log_gobs[j] = lg_gobs
        log_res[j] = lg_res

    n_valid = np.sum(valid)
    if n_valid < 5:
        n_skipped_rc += 1
        continue

    results[name] = {
        'logMs': logMs,
        'logMb': logMb,
        'dist': dist_mpc,
        'inc': inc_deg,
        'n_pts': int(n_valid),
        'log_res': log_res[valid],
        'log_gbar': log_gbar[valid],
        'mean_res': float(np.mean(log_res[valid])),
        'std_res': float(np.std(log_res[valid])),
        'gas_flag': gas_flag,
        'gas_source': gal.get('gas_source', 'none'),
    }

print(f"\n  Galaxies processed: {len(results)}")
print(f"  Gas-corrected (ALFALFA match): {n_gas_corrected}")
print(f"  Stars-only: {n_stars_only}")
print(f"  Skipped (no struct): {n_skipped_struct}")
print(f"  Skipped (inclination): {n_skipped_inc}")
print(f"  Skipped (RC quality): {n_skipped_rc}")

# ============================================================
# STEP 6: Compare gas-corrected vs stars-only scatter
# ============================================================
print("\n" + "=" * 72)
print("RESULTS: Gas-Corrected vs Stars-Only RAR Scatter")
print("=" * 72)

gas_corr = {n: r for n, r in results.items() if r['gas_flag'] == 'corrected'}
stars_only = {n: r for n, r in results.items() if r['gas_flag'] == 'stars_only'}

if gas_corr:
    gc_res = np.concatenate([r['log_res'] for r in gas_corr.values()])
    gc_means = np.array([r['mean_res'] for r in gas_corr.values()])
    print(f"\n  Gas-corrected: {len(gas_corr)} galaxies, {len(gc_res)} points")
    print(f"    Point scatter: {np.std(gc_res):.4f} dex")
    print(f"    Mean offset:   {np.mean(gc_res):+.4f} dex")
    print(f"    Galaxy scatter: {np.std(gc_means):.4f} dex")

if stars_only:
    so_res = np.concatenate([r['log_res'] for r in stars_only.values()])
    so_means = np.array([r['mean_res'] for r in stars_only.values()])
    print(f"\n  Stars-only: {len(stars_only)} galaxies, {len(so_res)} points")
    print(f"    Point scatter: {np.std(so_res):.4f} dex")
    print(f"    Mean offset:   {np.mean(so_res):+.4f} dex")
    print(f"    Galaxy scatter: {np.std(so_means):.4f} dex")

# Combined
all_res = np.concatenate([r['log_res'] for r in results.values()])
all_means = np.array([r['mean_res'] for r in results.values()])
print(f"\n  Combined: {len(results)} galaxies, {len(all_res)} points")
print(f"    Point scatter: {np.std(all_res):.4f} dex")
print(f"    Mean offset:   {np.mean(all_res):+.4f} dex")

# ============================================================
# STEP 7: Property correlations (gas-corrected subset)
# ============================================================
print("\n" + "=" * 72)
print("PROPERTY CORRELATIONS (Galaxy-Level Mean Residuals)")
print("=" * 72)

logMs_arr = np.array([r['logMs'] for r in results.values()])
mean_res_arr = np.array([r['mean_res'] for r in results.values()])
std_res_arr = np.array([r['std_res'] for r in results.values()])
n_pts_arr = np.array([r['n_pts'] for r in results.values()])

# Weighted by sqrt(N)
weights = np.sqrt(n_pts_arr)

# Spearman correlations
rho_ms, p_ms = spearmanr(logMs_arr, mean_res_arr)
print(f"\n  logMs:  rho = {rho_ms:+.3f}, p = {p_ms:.4f}")

# Split gas-corrected subsample
if len(gas_corr) > 20:
    gc_logMs = np.array([r['logMs'] for r in gas_corr.values()])
    gc_means_arr = np.array([r['mean_res'] for r in gas_corr.values()])
    rho_gc, p_gc = spearmanr(gc_logMs, gc_means_arr)
    print(f"\n  Gas-corrected subsample (N={len(gas_corr)}):")
    print(f"    logMs:  rho = {rho_gc:+.3f}, p = {p_gc:.4f}")

# ============================================================
# STEP 8: Variance uniformity (Levene test by stellar mass)
# ============================================================
print("\n" + "=" * 72)
print("VARIANCE UNIFORMITY (Levene Test)")
print("=" * 72)

# Split by stellar mass
med_logMs = np.median(logMs_arr)
low_mass = mean_res_arr[logMs_arr < med_logMs]
high_mass = mean_res_arr[logMs_arr >= med_logMs]

stat_L, p_L = levene(low_mass, high_mass)
print(f"\n  Low-mass (<{med_logMs:.1f}): N={len(low_mass)}, sigma={np.std(low_mass):.4f}")
print(f"  High-mass (>={med_logMs:.1f}): N={len(high_mass)}, sigma={np.std(high_mass):.4f}")
print(f"  Levene: F={stat_L:.3f}, p={p_L:.4f}")

if p_L > 0.05:
    print("  -> UNIFORM variance (BEC-consistent)")
else:
    print("  -> NON-UNIFORM variance")

# Split gas-corrected vs stars-only
if gas_corr and stars_only and len(gc_means) >= 5 and len(so_means) >= 5:
    stat_L2, p_L2 = levene(gc_means, so_means)
    print(f"\n  Gas-corrected vs Stars-only:")
    print(f"    Gas-corrected: sigma={np.std(gc_means):.4f}")
    print(f"    Stars-only:    sigma={np.std(so_means):.4f}")
    print(f"    Levene: F={stat_L2:.3f}, p={p_L2:.4f}")

# ============================================================
# STEP 9: Multivariate R² (using available properties)
# ============================================================
print("\n" + "=" * 72)
print("MULTIVARIATE EXPLAINED VARIANCE")
print("=" * 72)

features = np.column_stack([logMs_arr])
feature_names = ['logMs']

# Add gas flag as binary
gas_binary = np.array([1.0 if r['gas_flag'] == 'corrected' else 0.0
                        for r in results.values()])
features = np.column_stack([features, gas_binary])
feature_names.append('has_gas')

valid_feat = np.all(np.isfinite(features), axis=1) & np.isfinite(mean_res_arr)
X = features[valid_feat]
y = mean_res_arr[valid_feat]

if len(X) > 10:
    # Manual OLS regression: y = X_aug @ beta
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_pred = X_aug @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"\n  Features: {feature_names}")
    print(f"  N = {len(X)}")
    print(f"  R² = {r2:.4f} ({100*r2:.1f}%)")
    print(f"  Coefficients: {dict(zip(feature_names, beta[1:].round(4)))}")
    print(f"  Intercept: {beta[0]:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

improvement = None
if gas_corr and stars_only:
    sigma_gc = np.std(gc_res)
    sigma_so = np.std(so_res)
    improvement = sigma_so - sigma_gc
    print(f"\n  Gas correction reduces scatter by {improvement:.4f} dex")
    print(f"    Stars-only scatter: {sigma_so:.4f} dex")
    print(f"    Gas-corrected scatter: {sigma_gc:.4f} dex")
    print(f"    Offset (stars-only): {np.mean(so_res):+.4f} dex")
    print(f"    Offset (gas-corrected): {np.mean(gc_res):+.4f} dex")

print(f"\n  Property R²: {r2:.4f}")
if r2 < 0.05:
    print("  -> Scatter is property-INDEPENDENT (BEC-consistent)")
elif r2 < 0.15:
    print("  -> Weak property dependence")
else:
    print("  -> Significant property dependence")

if p_L > 0.05:
    print(f"  Variance uniformity: UNIFORM (Levene p={p_L:.4f})")
else:
    print(f"  Variance uniformity: NON-UNIFORM (Levene p={p_L:.4f})")

# ============================================================
# SAVE RESULTS
# ============================================================
summary = {
    'test_name': 'probes_gas_corrected_rar_tightness',
    'n_total': len(results),
    'n_gas_corrected': len(gas_corr),
    'n_stars_only': len(stars_only),
    'alfalfa_matched': matched,
    'alfalfa_unmatched': unmatched,
    'alfalfa_out_of_footprint': too_south,
    'scatter': {
        'combined_point': round(float(np.std(all_res)), 4),
        'combined_galaxy': round(float(np.std(all_means)), 4),
        'gas_corrected_point': round(float(np.std(gc_res)), 4) if gas_corr else None,
        'gas_corrected_galaxy': round(float(np.std(gc_means)), 4) if gas_corr else None,
        'stars_only_point': round(float(np.std(so_res)), 4) if stars_only else None,
        'stars_only_galaxy': round(float(np.std(so_means)), 4) if stars_only else None,
    },
    'offsets': {
        'combined': round(float(np.mean(all_res)), 4),
        'gas_corrected': round(float(np.mean(gc_res)), 4) if gas_corr else None,
        'stars_only': round(float(np.mean(so_res)), 4) if stars_only else None,
    },
    'correlations': {
        'logMs_rho': round(float(rho_ms), 4),
        'logMs_p': round(float(p_ms), 4),
    },
    'levene_by_mass': {
        'F': round(float(stat_L), 3),
        'p': round(float(p_L), 4),
    },
    'r_squared': round(float(r2), 4),
    'improvement_dex': round(float(improvement), 4) if improvement is not None else None,
}

outpath = os.path.join(RESULTS_DIR, 'summary_probes_gas_corrected.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("=" * 72)
print("Done.")
