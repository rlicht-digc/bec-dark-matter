#!/usr/bin/env python3
"""
PROBES Inversion Point Replication Test
=========================================

THE DECISIVE TEST: Does the scatter derivative inversion point appear at g†
in an independent dataset of ~3,000 galaxies with resolved rotation curves?

SPARC result: dσ/d(log g_bar) = 0 at log g_bar = -9.86 (0.07 dex from g†)
SPARC: N = 131 quality-cut galaxies
PROBES: N ~ 3,163 galaxies with rotation curves, photometric mass models

This script:
  1. Loads PROBES rotation curves and structural parameters (Mstar, inclination)
  2. Cross-matches with ALFALFA α.100 for gas masses (HI)
  3. Cross-matches with Yang DR7 for environment (halo mass)
  4. Computes RAR: g_obs = V²/R, g_bar = G × M_enc(R) / R²
  5. Computes scatter profile σ(log g_bar) and derivative dσ/d(log g_bar)
  6. Finds zero-crossings — compares to g† = -9.92
  7. Tests two-regime environmental pattern:
     - Dense scatter > field at low acceleration (tidal disruption of condensate)
     - Scatter converges at high acceleration (baryon-dominated)
  8. Runs binning robustness (3 widths × 3 offsets = 9 configs)

If PROBES replicates the SPARC inversion at g†, the BEC DM framework has
independent confirmation in a dataset 18× larger, from different telescopes,
with independent photometry.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import csv
import re
import numpy as np
from scipy.stats import levene, spearmanr
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROBES_DIR = os.path.join(DATA_DIR, 'probes')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics constants
g_dagger = 1.20e-10   # m/s², condensation scale
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921
G_kpc = 4.302e-6      # (km/s)² kpc / Msun
conv = 1e6 / 3.0857e19  # (km/s)²/kpc -> m/s²

print("=" * 72)
print("PROBES INVERSION POINT REPLICATION TEST")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")


# ================================================================
# STEP 1: Load PROBES main table (galaxy metadata)
# ================================================================
print("\n[1] Loading PROBES main table...")

main_path = os.path.join(PROBES_DIR, 'main_table.csv')
probes_meta = {}
with open(main_path) as f:
    lines = f.readlines()
# Line 0 is units, line 1 is header
header = lines[1].strip().split(',')
for line in lines[2:]:
    parts = line.strip().split(',')
    if len(parts) < len(header):
        continue
    row = {header[i]: parts[i].strip() for i in range(len(header))}
    name = row.get('name', '').strip()
    if not name:
        continue
    try:
        probes_meta[name] = {
            'ra': float(row['RA']),
            'dec': float(row['DEC']),
            'dist': float(row['distance']),
            'vhel': float(row.get('redshift_helio', 'nan')),
        }
    except (ValueError, KeyError):
        continue

print(f"  {len(probes_meta)} galaxies loaded")


# ================================================================
# STEP 2: Load structural parameters (Mstar, inclination)
# ================================================================
print("\n[2] Loading structural parameters...")

struct_path = os.path.join(PROBES_DIR, 'structural_parameters.csv')
struct_data = {}

with open(struct_path) as f:
    # Line 0 is units comment
    f.readline()
    # Line 1 is header
    header_line = f.readline()
    header_sp = header_line.strip().split(',')

    # Find column indices for Mstar and inclination
    # Priority: Rlast:rc (at RC endpoint) > Ri23:r > Ri25:r
    mstar_col = None
    inc_col = None
    for suffix in ['|Rlast:rc', '|Ri23:r', '|Ri25:r', '|Ri22:r']:
        if mstar_col is None:
            cname = f'Mstar{suffix}'
            if cname in header_sp:
                mstar_col = header_sp.index(cname)
        if inc_col is None:
            cname = f'inclination{suffix}'
            if cname in header_sp:
                inc_col = header_sp.index(cname)

    # Also find Re (effective radius) for scale length estimate
    re_col = None
    for suffix in ['|Rlast:rc', '|Ri23:r', '|Ri25:r']:
        if re_col is None:
            cname = f'physR{suffix}'
            if cname in header_sp:
                re_col = header_sp.index(cname)

    name_col = header_sp.index('name')
    print(f"  Mstar column: {header_sp[mstar_col] if mstar_col else 'NONE'}")
    print(f"  Inclination column: {header_sp[inc_col] if inc_col else 'NONE'}")
    print(f"  physR column: {header_sp[re_col] if re_col else 'NONE'}")

    max_col = max(c for c in [name_col, mstar_col, inc_col, re_col] if c is not None)

    for line in f:
        parts = line.strip().split(',')
        if len(parts) <= max_col:
            continue
        name = parts[name_col].strip()
        if not name:
            continue
        try:
            mstar = float(parts[mstar_col]) if mstar_col else np.nan
            inc_ba = float(parts[inc_col]) if inc_col else np.nan
            phys_r = float(parts[re_col]) if re_col else np.nan
        except (ValueError, TypeError):
            continue
        # 99.999 = missing data sentinel in PROBES
        # Mstar is in solar masses (can be ~10^6 to 10^16), so check for exact sentinel
        if abs(mstar - 99.999) < 0.01 or mstar <= 0:
            mstar = np.nan
        if abs(inc_ba - 99.999) < 0.01 or inc_ba <= 0:
            inc_ba = np.nan
        if abs(phys_r - 99.999) < 0.01 or phys_r <= 0:
            phys_r = np.nan
        struct_data[name] = {
            'mstar': mstar,
            'inc_ba': inc_ba,
            'phys_r_kpc': phys_r,
        }

print(f"  {len(struct_data)} galaxies with structural parameters")
n_valid_mstar = sum(1 for v in struct_data.values()
                    if not np.isnan(v['mstar']))
n_valid_inc = sum(1 for v in struct_data.values()
                  if not np.isnan(v['inc_ba']))
print(f"  Valid Mstar: {n_valid_mstar}")
print(f"  Valid inclination: {n_valid_inc}")


# ================================================================
# STEP 3: Load SPARC names for exclusion (independence)
# ================================================================
print("\n[3] Loading SPARC names for exclusion...")

sparc_path = os.path.join(DATA_DIR, 'sparc', 'SPARC_Lelli2016c.mrt')
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
else:
    # Try alternative path
    alt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')
    if os.path.exists(alt_path):
        with open(alt_path) as f:
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

print(f"  {len(sparc_names)} SPARC names loaded for exclusion")


# ================================================================
# STEP 4: Load ALFALFA α.100 for gas masses
# ================================================================
print("\n[4] Loading ALFALFA α.100 for gas cross-match...")


def parse_sexagesimal_ra(s):
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    try:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return 15.0 * (h + m / 60.0 + sec / 3600.0)
    except ValueError:
        return None


def parse_sexagesimal_dec(s):
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    try:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        sign = -1 if d < 0 or s.strip().startswith('-') else 1
        return sign * (abs(d) + m / 60.0 + sec / 3600.0)
    except ValueError:
        return None


alfalfa = []
alf_path = os.path.join(DATA_DIR, 'alfalfa', 'alfalfa_alpha100_haynes2018.tsv')
if os.path.exists(alf_path):
    with open(alf_path) as f:
        header_found = False
        col_names = None
        for line in f:
            if line.startswith('#'):
                continue
            if line.strip().startswith('"') or line.strip().startswith('---'):
                continue
            parts = line.strip().split('\t')
            if not header_found:
                if parts[0].strip() == 'AGC':
                    col_names = [p.strip() for p in parts]
                    header_found = True
                    continue
                continue
            if len(parts) < 10 or parts[0].strip().startswith('-'):
                continue
            try:
                ra_str = parts[2].strip() if len(parts) > 2 else ''
                dec_str = parts[3].strip() if len(parts) > 3 else ''
                ra = parse_sexagesimal_ra(ra_str)
                dec = parse_sexagesimal_dec(dec_str)
                if ra is None or dec is None:
                    continue
                vhel = int(parts[6].strip()) if len(parts) > 6 else 0
                logmhi = float(parts[16].strip()) if len(parts) > 16 else np.nan
                hi_code = int(parts[18].strip()) if len(parts) > 18 else 0
                if hi_code != 1 or np.isnan(logmhi):
                    continue
                alfalfa.append({
                    'ra': ra, 'dec': dec, 'vhel': vhel, 'logMHI': logmhi
                })
            except (ValueError, IndexError):
                continue

print(f"  {len(alfalfa)} ALFALFA sources loaded")

# Build KD-tree for fast matching
if alfalfa:
    from scipy.spatial import cKDTree

    def radec_to_cart(ra, dec):
        ra_r, dec_r = np.radians(ra), np.radians(dec)
        return np.array([np.cos(dec_r) * np.cos(ra_r),
                         np.cos(dec_r) * np.sin(ra_r),
                         np.sin(dec_r)])

    alf_coords = np.array([radec_to_cart(a['ra'], a['dec']) for a in alfalfa])
    alf_tree = cKDTree(alf_coords)
    alf_logMHI = np.array([a['logMHI'] for a in alfalfa])
    alf_vhel = np.array([a['vhel'] for a in alfalfa])


# ================================================================
# STEP 5: Load Yang DR7 for environment classification
# ================================================================
print("\n[5] Loading Yang DR7 catalogs...")

yang_dir = os.path.join(DATA_DIR, 'yang_catalogs')
yang_loaded = False

if os.path.exists(os.path.join(yang_dir, 'SDSS7')):
    # Galaxy catalog
    print("  Loading SDSS7 galaxy catalog...")
    yang_ra = []
    yang_dec = []
    yang_z = []
    yang_galid = []
    with open(os.path.join(yang_dir, 'SDSS7')) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                yang_galid.append(int(parts[0]))
                yang_ra.append(float(parts[2]))
                yang_dec.append(float(parts[3]))
                yang_z.append(float(parts[4]))
            except (ValueError, IndexError):
                continue

    yang_ra = np.array(yang_ra)
    yang_dec = np.array(yang_dec)
    yang_z = np.array(yang_z)
    yang_galid = np.array(yang_galid)
    print(f"    {len(yang_galid)} galaxies")

    # Galaxy-to-group mapping
    print("  Loading galaxy-to-group mapping...")
    gal_to_grp = {}
    with open(os.path.join(yang_dir, 'imodelC_1')) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                gal_id = int(parts[0])
                grp_id = int(parts[2])
                gal_to_grp[gal_id] = grp_id
            except (ValueError, IndexError):
                continue
    print(f"    {len(gal_to_grp)} mappings")

    # Group catalog
    print("  Loading group catalog...")
    groups = {}
    with open(os.path.join(yang_dir, 'modelC_group')) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                grp_id = int(parts[0])
                if grp_id == 0:
                    continue
                logMh_L = float(parts[6])
                logMh_M = float(parts[7])
                logMh = logMh_L if logMh_L > 0 else logMh_M
                groups[grp_id] = logMh
            except (ValueError, IndexError):
                continue
    print(f"    {len(groups)} groups")

    # Build RA index for fast cross-match
    ra_step = 0.5
    yang_ra_bins = {}
    for i in range(len(yang_ra)):
        ra_bin = int(yang_ra[i] / ra_step)
        if ra_bin not in yang_ra_bins:
            yang_ra_bins[ra_bin] = []
        yang_ra_bins[ra_bin].append(i)

    yang_loaded = True
else:
    print("  Yang DR7 not found — skipping environment analysis")


# ================================================================
# STEP 6: Process rotation curves and compute RAR
# ================================================================
print("\n[6] Processing rotation curves...")

profiles_dir = os.path.join(PROBES_DIR, 'profiles', 'profiles')

# RAR prediction
def rar_pred(gbar):
    gbar = np.asarray(gbar, dtype=float)
    x = np.sqrt(np.maximum(gbar, 1e-20) / g_dagger)
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-20)
    return gbar / denom


# Statistics counters
n_processed = 0
n_sparc_excluded = 0
n_no_struct = 0
n_bad_inc = 0
n_no_rc = 0
n_bad_mstar = 0
n_too_few_pts = 0
n_gas_matched = 0
n_yang_matched = 0

# Results storage
gal_results = {}

all_names = sorted(probes_meta.keys())
for name in all_names:
    # Check SPARC exclusion
    norm = name.upper().replace(' ', '').replace('-', '')
    if norm in sparc_names:
        n_sparc_excluded += 1
        continue

    meta = probes_meta[name]
    sp = struct_data.get(name)
    if sp is None:
        n_no_struct += 1
        continue

    mstar = sp['mstar']
    inc_ba = sp['inc_ba']
    phys_r = sp['phys_r_kpc']

    if np.isnan(mstar) or mstar <= 0:
        n_bad_mstar += 1
        continue
    if np.isnan(inc_ba):
        n_bad_inc += 1
        continue

    # PROBES inclination is in RADIANS (0.31 to 1.57), convert to degrees
    inc_deg = np.degrees(inc_ba)

    # Quality cut on inclination
    if inc_deg < 30 or inc_deg > 85:
        n_bad_inc += 1
        continue

    # Load rotation curve
    rc_file = os.path.join(profiles_dir, f"{name}_rc.prof")
    if not os.path.exists(rc_file):
        n_no_rc += 1
        continue

    try:
        with open(rc_file) as f:
            rc_lines = f.readlines()

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
            n_too_few_pts += 1
            continue

    except Exception:
        n_no_rc += 1
        continue

    R_arcsec = np.array(R_arcsec)
    V_kms = np.array(V_kms)
    Ve_kms = np.array(Ve_kms)

    # Convert to physical units
    dist_mpc = meta['dist']
    R_kpc = R_arcsec * dist_mpc * np.pi / (180 * 3600) * 1000

    # Correct velocity for inclination
    sin_i = np.sin(np.radians(inc_deg))
    V_rot = V_kms / sin_i

    # g_obs = V²/R
    gobs = V_rot**2 / R_kpc * conv  # m/s²

    # Gas mass from ALFALFA
    logMHI = np.nan
    if alfalfa:
        cart = radec_to_cart(meta['ra'], meta['dec'])
        # Only search if in ALFALFA footprint
        if -2 < meta['dec'] < 38:
            match_radius_cart = 2 * np.sin(np.radians(2.0 / 60.0 / 2))
            dists_cart, idxs = alf_tree.query(cart, k=5,
                                               distance_upper_bound=match_radius_cart)
            for d, i in zip(dists_cart, idxs):
                if i >= len(alfalfa):
                    continue
                vhel_probes = meta['vhel']
                if not np.isnan(vhel_probes) and not np.isnan(alf_vhel[i]):
                    if abs(vhel_probes - alf_vhel[i]) > 200:
                        continue
                logMHI = alf_logMHI[i]
                n_gas_matched += 1
                break

    # Total baryonic mass
    if not np.isnan(logMHI):
        M_HI = 10**logMHI
        M_gas = 1.33 * M_HI  # helium correction
        M_baryonic = mstar + M_gas
        gas_flag = 'corrected'
    else:
        M_baryonic = mstar
        M_gas = 0.0
        gas_flag = 'stars_only'

    # Compute g_bar at each radius using exponential disk model
    # Use phys_r as scale estimate, else use median R
    if not np.isnan(phys_r) and phys_r > 0:
        Rd = phys_r / 1.678  # R50 -> exponential scale length
    else:
        Rd = np.median(R_kpc) / 1.678
    Rd = max(Rd, 0.1)

    log_gbar_pts = []
    log_gobs_pts = []
    log_res_pts = []

    for j in range(len(R_kpc)):
        r = R_kpc[j]
        x = r / Rd
        # Enclosed stellar mass (exponential disk)
        M_enc_star = mstar * (1 - (1 + x) * np.exp(-x))
        # Gas: uniform disk within R_max
        if M_gas > 0:
            R_max = R_kpc[-1]
            M_enc_gas = M_gas * min(r / R_max, 1.0)**2
        else:
            M_enc_gas = 0.0

        M_enc = M_enc_star + M_enc_gas
        gbar_val = G_kpc * M_enc / r**2 * conv  # m/s²

        if gbar_val <= 0 or gobs[j] <= 0:
            continue

        lg_gbar = np.log10(gbar_val)
        lg_gobs = np.log10(gobs[j])
        lg_rar = np.log10(rar_pred(gbar_val))
        lg_res = lg_gobs - lg_rar

        # Outlier cut
        if abs(lg_res) > 1.5:
            continue
        # Physical range cut
        if lg_gbar < -13 or lg_gbar > -8:
            continue

        log_gbar_pts.append(lg_gbar)
        log_gobs_pts.append(lg_gobs)
        log_res_pts.append(lg_res)

    if len(log_gbar_pts) < 5:
        n_too_few_pts += 1
        continue

    # Environment from Yang DR7
    logMh = np.nan
    if yang_loaded:
        c_speed = 299792.458
        z_probes = meta['vhel'] / c_speed if meta['vhel'] > 0 else np.nan
        if not np.isnan(z_probes):
            ra_bin = int(meta['ra'] / ra_step)
            candidates = []
            for b in [ra_bin - 1, ra_bin, ra_bin + 1]:
                if b in yang_ra_bins:
                    candidates.extend(yang_ra_bins[b])
            if candidates:
                cand = np.array(candidates)
                max_sep_deg = 10.0 / 3600.0
                dec_ok = np.abs(yang_dec[cand] - meta['dec']) < max_sep_deg * 2
                if np.any(dec_ok):
                    filtered = cand[dec_ok]
                    ra1, dec1 = np.radians(meta['ra']), np.radians(meta['dec'])
                    ra2 = np.radians(yang_ra[filtered])
                    dec2 = np.radians(yang_dec[filtered])
                    cos_sep = (np.sin(dec1) * np.sin(dec2) +
                               np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
                    cos_sep = np.clip(cos_sep, -1, 1)
                    sep_arcsec = np.degrees(np.arccos(cos_sep)) * 3600

                    z_ok = np.abs(yang_z[filtered] - z_probes) < 0.005
                    combined = (sep_arcsec < 10.0) & z_ok
                    if np.any(combined):
                        sub = np.where(combined)[0]
                        best = filtered[sub[np.argmin(sep_arcsec[sub])]]
                        gal_id = yang_galid[best]
                        if gal_id in gal_to_grp:
                            grp_id = gal_to_grp[gal_id]
                            if grp_id > 0 and grp_id in groups:
                                logMh = groups[grp_id]
                                n_yang_matched += 1

    gal_results[name] = {
        'log_gbar': np.array(log_gbar_pts),
        'log_gobs': np.array(log_gobs_pts),
        'log_res': np.array(log_res_pts),
        'n_pts': len(log_gbar_pts),
        'logMs': np.log10(mstar),
        'logMb': np.log10(M_baryonic),
        'dist': dist_mpc,
        'inc': inc_deg,
        'gas_flag': gas_flag,
        'logMh': logMh,
    }
    n_processed += 1

    if n_processed % 200 == 0:
        print(f"    ... processed {n_processed} galaxies")

print(f"\n  Total processed: {n_processed}")
print(f"  SPARC excluded: {n_sparc_excluded}")
print(f"  No structural data: {n_no_struct}")
print(f"  Bad Mstar: {n_bad_mstar}")
print(f"  Bad inclination: {n_bad_inc}")
print(f"  No/bad RC: {n_no_rc}")
print(f"  Too few valid points: {n_too_few_pts}")
print(f"  ALFALFA gas matched: {n_gas_matched}")
print(f"  Yang env matched: {n_yang_matched}")

if n_processed < 20:
    print("\n  ERROR: Too few galaxies processed. Check data paths.")
    sys.exit(1)


# ================================================================
# STEP 7: Aggregate all RAR points
# ================================================================
print("\n[7] Aggregating RAR data...")

all_gbar = np.concatenate([r['log_gbar'] for r in gal_results.values()])
all_gobs = np.concatenate([r['log_gobs'] for r in gal_results.values()])
all_res = np.concatenate([r['log_res'] for r in gal_results.values()])

print(f"  Total points: {len(all_gbar)}")
print(f"  g_bar range: [{all_gbar.min():.2f}, {all_gbar.max():.2f}]")
print(f"  Mean residual: {np.mean(all_res):+.4f} dex")
print(f"  RMS scatter: {np.std(all_res):.4f} dex")


# ================================================================
# STEP 7b: Compute CORRECTED residuals (per-galaxy offset removed)
# ================================================================
print("\n[7b] Computing corrected residuals (per-galaxy offset removed)...")

# For each galaxy, subtract its mean residual to isolate within-galaxy scatter
all_gbar_corr = []
all_res_corr = []
for name, r in gal_results.items():
    mean_res = np.mean(r['log_res'])
    corrected = r['log_res'] - mean_res
    r['log_res_corr'] = corrected
    r['mean_offset'] = mean_res
    all_gbar_corr.extend(r['log_gbar'].tolist())
    all_res_corr.extend(corrected.tolist())

all_gbar_corr = np.array(all_gbar_corr)
all_res_corr = np.array(all_res_corr)
print(f"  Corrected residual RMS: {np.std(all_res_corr):.4f} dex "
      f"(was {np.std(all_res):.4f})")
print(f"  Mean per-galaxy offset: {np.mean([r['mean_offset'] for r in gal_results.values()]):+.4f} dex")


# ================================================================
# STEP 8: Scatter profile and derivative (MAIN TEST)
# ================================================================
print("\n" + "=" * 72)
print("MAIN TEST: Scatter Profile and Inversion Point")
print("=" * 72)


def compute_scatter_profile(gbar, res, bin_width=0.35, offset=0.0):
    """Compute scatter as a function of log(g_bar) in sliding bins."""
    lo = max(np.percentile(gbar, 2), -13.0)
    hi = min(np.percentile(gbar, 98), -8.0)
    edges = np.arange(lo + offset, hi, bin_width)

    centers = []
    sigmas = []
    counts = []

    for edge in edges:
        mask = (gbar >= edge) & (gbar < edge + bin_width)
        n = np.sum(mask)
        if n >= 30:  # minimum for reliable scatter
            centers.append(edge + bin_width / 2)
            sigmas.append(np.std(res[mask]))
            counts.append(int(n))

    return np.array(centers), np.array(sigmas), np.array(counts)


def numerical_derivative(x, y):
    """Central difference derivative."""
    dy = np.zeros_like(y)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0]) if len(y) > 1 else 0
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]) if len(y) > 1 else 0
    for i in range(1, len(y) - 1):
        dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return dy


def find_zero_crossings(x, y):
    """Find x values where y crosses zero (linear interpolation)."""
    crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i+1] < 0:
            # Linear interpolation
            x_cross = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
            crossings.append(x_cross)
    return crossings


# Default analysis
centers, sigmas, counts = compute_scatter_profile(all_gbar, all_res,
                                                    bin_width=0.35, offset=0.0)

print(f"\n  Scatter profile ({len(centers)} bins, width=0.35 dex):")
for i in range(len(centers)):
    flag = " <-- g†" if abs(centers[i] - LOG_G_DAGGER) < 0.20 else ""
    print(f"    log g_bar = {centers[i]:+6.2f}: σ = {sigmas[i]:.4f} dex "
          f"(N={counts[i]:4d}){flag}")

# Derivative
if len(centers) >= 3:
    dsigma = numerical_derivative(centers, sigmas)
    crossings = find_zero_crossings(centers, dsigma)

    print(f"\n  Scatter derivative dσ/d(log g_bar):")
    for i in range(len(centers)):
        print(f"    log g_bar = {centers[i]:+6.2f}: dσ/dx = {dsigma[i]:+.5f}")

    print(f"\n  Zero-crossings: {len(crossings)}")
    if crossings:
        # Find crossing nearest to g†
        dists_from_gdagger = [abs(c - LOG_G_DAGGER) for c in crossings]
        best_idx = np.argmin(dists_from_gdagger)
        best_crossing = crossings[best_idx]
        dist_to_gdagger = best_crossing - LOG_G_DAGGER

        for c in crossings:
            flag = " *** NEAREST TO g†" if c == best_crossing else ""
            print(f"    log g_bar = {c:+.3f} "
                  f"(Δ from g† = {c - LOG_G_DAGGER:+.3f} dex){flag}")

        print(f"\n  *** INVERSION POINT: log g_bar = {best_crossing:+.3f}")
        print(f"  *** g† = {LOG_G_DAGGER:+.3f}")
        print(f"  *** DISTANCE FROM g†: {dist_to_gdagger:+.3f} dex")
        if abs(dist_to_gdagger) < 0.20:
            print(f"  *** WITHIN 0.20 dex -- REPLICATION CONFIRMED")
        elif abs(dist_to_gdagger) < 0.50:
            print(f"  *** WITHIN 0.50 dex -- PARTIAL REPLICATION")
        else:
            print(f"  *** > 0.50 dex -- NO REPLICATION")
    else:
        best_crossing = None
        dist_to_gdagger = None
        print("    No zero-crossings found in scatter derivative")
else:
    crossings = []
    best_crossing = None
    dist_to_gdagger = None
    print("  Too few bins for derivative analysis")


# ================================================================
# STEP 8b: CORRECTED scatter profile (per-galaxy offset removed)
# ================================================================
print("\n" + "=" * 72)
print("CORRECTED SCATTER PROFILE (per-galaxy M/L offset removed)")
print("=" * 72)

centers_c, sigmas_c, counts_c = compute_scatter_profile(
    all_gbar_corr, all_res_corr, bin_width=0.35, offset=0.0)

print(f"\n  Corrected scatter profile ({len(centers_c)} bins):")
for i in range(len(centers_c)):
    flag = " <-- g†" if abs(centers_c[i] - LOG_G_DAGGER) < 0.20 else ""
    print(f"    log g_bar = {centers_c[i]:+6.2f}: σ = {sigmas_c[i]:.4f} dex "
          f"(N={counts_c[i]:4d}){flag}")

if len(centers_c) >= 3:
    dsigma_c = numerical_derivative(centers_c, sigmas_c)
    crossings_c = find_zero_crossings(centers_c, dsigma_c)

    print(f"\n  Corrected derivative dσ/d(log g_bar):")
    for i in range(len(centers_c)):
        print(f"    log g_bar = {centers_c[i]:+6.2f}: dσ/dx = {dsigma_c[i]:+.5f}")

    print(f"\n  Corrected zero-crossings: {len(crossings_c)}")
    if crossings_c:
        for c in crossings_c:
            print(f"    log g_bar = {c:+.3f} (Δ from g† = {c - LOG_G_DAGGER:+.3f} dex)")
        best_crossing_c = min(crossings_c, key=lambda x: abs(x - LOG_G_DAGGER))
        dist_c = best_crossing_c - LOG_G_DAGGER
        print(f"\n  *** CORRECTED INVERSION: log g_bar = {best_crossing_c:+.3f}")
        print(f"  *** DISTANCE FROM g†: {dist_c:+.3f} dex")
        if abs(dist_c) < 0.20:
            print(f"  *** WITHIN 0.20 dex -- REPLICATION IN CORRECTED DATA")
        elif abs(dist_c) < 0.50:
            print(f"  *** WITHIN 0.50 dex -- PARTIAL REPLICATION IN CORRECTED DATA")
    else:
        best_crossing_c = None
        dist_c = None
        print("    No zero-crossings found")
else:
    best_crossing_c = None
    dist_c = None


# ================================================================
# STEP 9: Binning robustness (3 widths × 3 offsets)
# ================================================================
print("\n" + "=" * 72)
print("BINNING ROBUSTNESS")
print("=" * 72)

bin_widths = [0.25, 0.35, 0.50]
offsets = [0.0, 0.10, 0.20]
robustness_results = []

for bw in bin_widths:
    for off in offsets:
        c, s, n = compute_scatter_profile(all_gbar, all_res,
                                          bin_width=bw, offset=off)
        if len(c) >= 3:
            ds = numerical_derivative(c, s)
            zc = find_zero_crossings(c, ds)
            if zc:
                # Nearest to g†
                nearest = min(zc, key=lambda x: abs(x - LOG_G_DAGGER))
                robustness_results.append({
                    'bin_width': bw, 'offset': off,
                    'crossing': nearest,
                    'dist_gdagger': nearest - LOG_G_DAGGER,
                })
                tag = "MATCH" if abs(nearest - LOG_G_DAGGER) < 0.20 else "miss"
                print(f"  bw={bw:.2f}, off={off:.2f}: crossing at "
                      f"{nearest:+.3f} (Δ = {nearest - LOG_G_DAGGER:+.3f}) [{tag}]")
            else:
                print(f"  bw={bw:.2f}, off={off:.2f}: no zero-crossing")
        else:
            print(f"  bw={bw:.2f}, off={off:.2f}: too few bins")

if robustness_results:
    all_crossings = [r['crossing'] for r in robustness_results]
    all_dists = [abs(r['dist_gdagger']) for r in robustness_results]
    n_within_020 = sum(1 for d in all_dists if d < 0.20)
    print(f"\n  Configs with crossing: {len(robustness_results)}/{len(bin_widths)*len(offsets)}")
    print(f"  Mean crossing: {np.mean(all_crossings):+.3f}")
    print(f"  Std of crossings: {np.std(all_crossings):.3f}")
    print(f"  Within 0.20 dex of g†: {n_within_020}/{len(robustness_results)}")


# ================================================================
# STEP 10: Environmental analysis
# ================================================================
print("\n" + "=" * 72)
print("ENVIRONMENTAL ANALYSIS")
print("=" * 72)

# Split by halo mass
env_gals = {n: r for n, r in gal_results.items() if not np.isnan(r['logMh'])}
print(f"  Galaxies with environment: {len(env_gals)}")

if len(env_gals) >= 40:
    # Environment threshold: logMh = 12.5 (same as ALFALFA test)
    DENSE_THRESH = 12.5
    field_gals = {n: r for n, r in env_gals.items() if r['logMh'] < DENSE_THRESH}
    dense_gals = {n: r for n, r in env_gals.items() if r['logMh'] >= DENSE_THRESH}

    print(f"  Field (logMh < {DENSE_THRESH}): {len(field_gals)} galaxies")
    print(f"  Dense (logMh >= {DENSE_THRESH}): {len(dense_gals)} galaxies")

    if len(field_gals) >= 15 and len(dense_gals) >= 10:
        field_gbar = np.concatenate([r['log_gbar'] for r in field_gals.values()])
        field_res = np.concatenate([r['log_res'] for r in field_gals.values()])
        dense_gbar = np.concatenate([r['log_gbar'] for r in dense_gals.values()])
        dense_res = np.concatenate([r['log_res'] for r in dense_gals.values()])
        # Also corrected residuals
        field_res_c = np.concatenate([r['log_res_corr'] for r in field_gals.values()])
        dense_res_c = np.concatenate([r['log_res_corr'] for r in dense_gals.values()])

        print(f"\n  Field points: {len(field_gbar)}")
        print(f"  Dense points: {len(dense_gbar)}")

        # Acceleration-binned scatter comparison
        accel_bins = [
            ('very_low', -13.0, -11.0),
            ('low', -11.0, -10.3),
            ('transition', -10.3, -9.5),
            ('high', -9.5, -8.0),
        ]

        env_results = []
        print(f"\n  {'Regime':<12} {'σ_field':>8} {'σ_dense':>8} {'Δσ':>8} "
              f"{'Levene_p':>9} {'N_f':>5} {'N_d':>5}")
        print("  " + "-" * 60)

        for regime, lo, hi in accel_bins:
            f_mask = (field_gbar >= lo) & (field_gbar < hi)
            d_mask = (dense_gbar >= lo) & (dense_gbar < hi)
            nf = np.sum(f_mask)
            nd = np.sum(d_mask)

            if nf >= 10 and nd >= 5:
                sf = np.std(field_res[f_mask])
                sd = np.std(dense_res[d_mask])
                delta_sigma = sd - sf
                try:
                    lev_stat, lev_p = levene(field_res[f_mask], dense_res[d_mask])
                except Exception:
                    lev_p = np.nan

                env_results.append({
                    'regime': regime, 'lo': lo, 'hi': hi,
                    'sigma_field': round(float(sf), 4),
                    'sigma_dense': round(float(sd), 4),
                    'delta_sigma': round(float(delta_sigma), 4),
                    'levene_p': round(float(lev_p), 4),
                    'n_field': int(nf), 'n_dense': int(nd),
                })
                tag = "*" if lev_p < 0.05 else ""
                print(f"  {regime:<12} {sf:8.4f} {sd:8.4f} {delta_sigma:+8.4f} "
                      f"{lev_p:9.4f}{tag} {nf:5d} {nd:5d}")
            else:
                env_results.append({
                    'regime': regime, 'lo': lo, 'hi': hi,
                    'sigma_field': None, 'sigma_dense': None,
                    'delta_sigma': None, 'levene_p': None,
                    'n_field': int(nf), 'n_dense': int(nd),
                })
                print(f"  {regime:<12} {'--':>8} {'--':>8} {'--':>8} "
                      f"{'--':>9}  {nf:5d} {nd:5d}")

        # Check for two-regime pattern:
        # Dense > field at low accel, converging at high accel
        valid_env = [r for r in env_results
                     if r['delta_sigma'] is not None]
        if len(valid_env) >= 2:
            low_delta = [r['delta_sigma'] for r in valid_env
                         if r['lo'] < -10.3]
            high_delta = [r['delta_sigma'] for r in valid_env
                          if r['lo'] >= -10.3]

            if low_delta and high_delta:
                mean_low = np.mean(low_delta)
                mean_high = np.mean(high_delta)
                print(f"\n  Low-accel mean Δσ: {mean_low:+.4f}")
                print(f"  High-accel mean Δσ: {mean_high:+.4f}")

                if mean_low > 0 and abs(mean_high) < abs(mean_low):
                    print("  -> TWO-REGIME PATTERN: dense > field at low accel, "
                          "converging at high — BEC-CONSISTENT")
                    env_verdict = "BEC-CONSISTENT"
                elif mean_low > 0:
                    print("  -> Dense > field at low accel but also at high — "
                          "uniform excess, not BEC-specific")
                    env_verdict = "UNIFORM_EXCESS"
                else:
                    print("  -> No dense > field at low accel — "
                          "does not match SPARC pattern")
                    env_verdict = "NO_PATTERN"
            else:
                env_verdict = "INSUFFICIENT"
        else:
            env_verdict = "INSUFFICIENT"
    else:
        env_results = []
        env_verdict = "INSUFFICIENT"
        print("  Too few galaxies in one or both environment bins")
else:
    env_results = []
    env_verdict = "NO_ENV_DATA"
    print("  Too few galaxies with environment classification")


# ================================================================
# STEP 10b: CORRECTED environmental analysis
# ================================================================
if len(env_gals) >= 40 and len(field_gals) >= 15 and len(dense_gals) >= 10:
    print("\n" + "=" * 72)
    print("CORRECTED ENVIRONMENTAL ANALYSIS (per-galaxy offset removed)")
    print("=" * 72)

    env_results_corr = []
    print(f"\n  {'Regime':<12} {'σ_field':>8} {'σ_dense':>8} {'Δσ':>8} "
          f"{'Levene_p':>9} {'N_f':>5} {'N_d':>5}")
    print("  " + "-" * 60)

    for regime, lo, hi in accel_bins:
        f_mask = (field_gbar >= lo) & (field_gbar < hi)
        d_mask = (dense_gbar >= lo) & (dense_gbar < hi)
        nf = np.sum(f_mask)
        nd = np.sum(d_mask)

        if nf >= 10 and nd >= 5:
            sf = np.std(field_res_c[f_mask])
            sd = np.std(dense_res_c[d_mask])
            delta_sigma = sd - sf
            try:
                lev_stat, lev_p = levene(field_res_c[f_mask], dense_res_c[d_mask])
            except Exception:
                lev_p = np.nan

            env_results_corr.append({
                'regime': regime,
                'sigma_field': round(float(sf), 4),
                'sigma_dense': round(float(sd), 4),
                'delta_sigma': round(float(delta_sigma), 4),
                'levene_p': round(float(lev_p), 4),
            })
            tag = "*" if lev_p < 0.05 else ""
            print(f"  {regime:<12} {sf:8.4f} {sd:8.4f} {delta_sigma:+8.4f} "
                  f"{lev_p:9.4f}{tag} {nf:5d} {nd:5d}")
        else:
            env_results_corr.append({
                'regime': regime,
                'sigma_field': None, 'sigma_dense': None,
                'delta_sigma': None, 'levene_p': None,
            })
            print(f"  {regime:<12} {'--':>8} {'--':>8} {'--':>8} "
                  f"{'--':>9}  {nf:5d} {nd:5d}")

    # Check pattern in corrected data
    valid_corr = [r for r in env_results_corr if r['delta_sigma'] is not None]
    if len(valid_corr) >= 2:
        low_d = [r['delta_sigma'] for r in valid_corr
                 if r['regime'] in ('very_low', 'low')]
        high_d = [r['delta_sigma'] for r in valid_corr
                  if r['regime'] in ('transition', 'high')]
        if low_d and high_d:
            print(f"\n  Corrected low-accel mean Δσ: {np.mean(low_d):+.4f}")
            print(f"  Corrected high-accel mean Δσ: {np.mean(high_d):+.4f}")
            if np.mean(low_d) > 0 and abs(np.mean(high_d)) < abs(np.mean(low_d)):
                print("  -> CORRECTED: BEC-CONSISTENT pattern emerges")
                env_verdict_corr = "BEC-CONSISTENT"
            else:
                env_verdict_corr = "NO_PATTERN"
                print(f"  -> CORRECTED: No BEC pattern")
        else:
            env_verdict_corr = "INSUFFICIENT"
    else:
        env_verdict_corr = "INSUFFICIENT"
else:
    env_results_corr = []
    env_verdict_corr = env_verdict


# ================================================================
# STEP 11: Separate scatter profiles by environment
# ================================================================
if len(env_gals) >= 40 and len(field_gals) >= 15 and len(dense_gals) >= 10:
    print("\n" + "=" * 72)
    print("SCATTER PROFILES BY ENVIRONMENT")
    print("=" * 72)

    c_f, s_f, n_f = compute_scatter_profile(field_gbar, field_res,
                                              bin_width=0.50, offset=0.0)
    c_d, s_d, n_d = compute_scatter_profile(dense_gbar, dense_res,
                                              bin_width=0.50, offset=0.0)

    print(f"\n  {'log_gbar':>10} {'σ_field':>8} {'σ_dense':>8} {'Δσ':>8}")
    print("  " + "-" * 40)

    # Match bins
    for i in range(len(c_f)):
        # Find matching dense bin
        d_match = None
        for j in range(len(c_d)):
            if abs(c_f[i] - c_d[j]) < 0.10:
                d_match = j
                break
        if d_match is not None:
            delta = s_d[d_match] - s_f[i]
            flag = " <-- g†" if abs(c_f[i] - LOG_G_DAGGER) < 0.30 else ""
            print(f"  {c_f[i]:+10.2f} {s_f[i]:8.4f} {s_d[d_match]:8.4f} "
                  f"{delta:+8.4f}{flag}")
        else:
            print(f"  {c_f[i]:+10.2f} {s_f[i]:8.4f} {'--':>8} {'--':>8}")


# ================================================================
# STEP 12: Compare PROBES vs SPARC inversion points
# ================================================================
print("\n" + "=" * 72)
print("COMPARISON: PROBES vs SPARC INVERSION POINTS")
print("=" * 72)

sparc_inversion = -9.86  # from test_mc_distance_and_inversion.py

if best_crossing is not None:
    print(f"\n  SPARC inversion point:  log g_bar = {sparc_inversion:+.3f}")
    print(f"  PROBES inversion point: log g_bar = {best_crossing:+.3f}")
    print(f"  g† (theory):            log g_bar = {LOG_G_DAGGER:+.3f}")
    print(f"\n  SPARC distance from g†:  {sparc_inversion - LOG_G_DAGGER:+.3f} dex")
    print(f"  PROBES distance from g†: {best_crossing - LOG_G_DAGGER:+.3f} dex")
    print(f"  SPARC-PROBES agreement:  {abs(best_crossing - sparc_inversion):.3f} dex")

    if abs(best_crossing - LOG_G_DAGGER) < 0.20:
        replication = "CONFIRMED"
    elif abs(best_crossing - LOG_G_DAGGER) < 0.50:
        replication = "PARTIAL"
    else:
        replication = "NOT_CONFIRMED"
else:
    replication = "NO_CROSSING"
    print("  No inversion point found in PROBES data")


# ================================================================
# VERDICT
# ================================================================
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

print(f"\n  Galaxies analyzed: {n_processed} (SPARC excluded)")
print(f"  Total RAR points: {len(all_gbar)}")
print(f"  RMS scatter: {np.std(all_res):.4f} dex")

if best_crossing is not None:
    print(f"  Inversion point: log g_bar = {best_crossing:+.3f}")
    print(f"  Distance from g†: {dist_to_gdagger:+.3f} dex")
else:
    print(f"  Inversion point: NOT FOUND")

print(f"  Replication status: {replication}")
print(f"  Environment verdict: {env_verdict}")

if robustness_results:
    n_robust = sum(1 for r in robustness_results
                   if abs(r['dist_gdagger']) < 0.20)
    print(f"  Binning robustness: {n_robust}/{len(robustness_results)} "
          f"configs within 0.20 dex of g†")

if replication == "CONFIRMED" and env_verdict == "BEC-CONSISTENT":
    print("\n  >>> FULL REPLICATION: Both inversion point and environmental")
    print("  >>> pattern confirmed in PROBES (independent dataset).")
    overall = "FULL_REPLICATION"
elif replication == "CONFIRMED":
    print("\n  >>> INVERSION REPLICATED: Scatter derivative zero-crossing")
    print(f"  >>> at g† confirmed. Environment: {env_verdict}")
    overall = "INVERSION_REPLICATED"
elif replication == "PARTIAL":
    print("\n  >>> PARTIAL REPLICATION: Inversion point within 0.50 dex")
    overall = "PARTIAL_REPLICATION"
else:
    print("\n  >>> NOT REPLICATED in PROBES data")
    overall = "NOT_REPLICATED"


# ================================================================
# SAVE RESULTS
# ================================================================
summary = {
    'test_name': 'probes_inversion_replication',
    'n_galaxies': n_processed,
    'n_sparc_excluded': n_sparc_excluded,
    'n_total_points': len(all_gbar),
    'n_gas_matched': n_gas_matched,
    'n_yang_matched': n_yang_matched,
    'rms_scatter': round(float(np.std(all_res)), 4),
    'mean_offset': round(float(np.mean(all_res)), 4),
    'inversion_point': round(float(best_crossing), 4) if best_crossing is not None else None,
    'dist_from_gdagger': round(float(dist_to_gdagger), 4) if dist_to_gdagger is not None else None,
    'sparc_inversion': sparc_inversion,
    'replication_status': replication,
    'binning_robustness': {
        'n_configs': len(robustness_results),
        'n_within_020': sum(1 for r in robustness_results
                           if abs(r['dist_gdagger']) < 0.20),
        'mean_crossing': round(float(np.mean([r['crossing']
                               for r in robustness_results])), 4)
                         if robustness_results else None,
        'std_crossing': round(float(np.std([r['crossing']
                              for r in robustness_results])), 4)
                        if robustness_results else None,
        'configs': robustness_results,
    },
    'environment': {
        'n_env_galaxies': len(env_gals) if 'env_gals' in dir() else 0,
        'n_field': len(field_gals) if 'field_gals' in dir() else 0,
        'n_dense': len(dense_gals) if 'dense_gals' in dir() else 0,
        'verdict': env_verdict,
        'accel_bins': env_results,
    },
    'corrected_analysis': {
        'rms_scatter': round(float(np.std(all_res_corr)), 4),
        'inversion_point': round(float(best_crossing_c), 4) if best_crossing_c is not None else None,
        'dist_from_gdagger': round(float(dist_c), 4) if dist_c is not None else None,
        'env_verdict': env_verdict_corr if 'env_verdict_corr' in dir() else None,
        'env_bins': env_results_corr if 'env_results_corr' in dir() else [],
    },
    'overall_verdict': overall,
}

outpath = os.path.join(RESULTS_DIR, 'summary_probes_inversion_replication.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("=" * 72)
print("Done.")
