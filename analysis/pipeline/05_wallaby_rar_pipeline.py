#!/usr/bin/env python3
"""
STEP 5: WALLABY Expanded-Sample RAR Pipeline
=============================================
Ingests WALLABY DR1+DR2 kinematic catalogs to build an independent
RAR environmental scatter test.

WALLABY provides:
  - Rotation curves: V_rot(R) from tilted-ring fits (3D kinematic models)
  - HI surface density profiles: Σ_HI(R) and face-on corrected Σ_HI(R)
  - Source catalogs: distances (Hubble flow), HI masses, w20/w50

Key difference from SPARC:
  - WALLABY does NOT provide stellar mass decomposition (Vdisk, Vbul).
  - We compute gobs from Vrot and g_gas from HI surface densities.
  - At low accelerations (log gbar < -11), gas dominates → gbar ≈ g_gas.
  - This is the exact regime where the BEC environmental signal is strongest.
  - We also attempt WISE W1 cross-matching for stellar mass estimates.

Strategy:
  1. Gas-dominated RAR: Use only g_gas for gbar (conservative, valid at low-a)
  2. WISE-enhanced RAR: Cross-match with WISE for stellar mass estimates
  3. Environmental test on gas-dominated points only

Russell Licht — Primordial Fluid DM Project
Feb 2026
"""

import numpy as np
from scipy import stats
import csv
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
WALLABY_DIR = os.path.join(DATA_DIR, 'wallaby')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONSTANTS (same as main pipeline)
# ============================================================
g_dagger = 1.20e-10   # m/s^2 (RAR scale)
conv = 1e6 / 3.0857e19  # (km/s)^2/kpc -> m/s^2
H0 = 75.0  # km/s/Mpc
c_light = 299792.458  # km/s
HI_freq = 1.420405751e9  # Hz (21 cm line)

# Haubner+2025 CF4 parameters
CF4_PARAMS = {
    'delta_inf': 0.022, 'alpha': -0.8, 'D_tr': 46.0, 'kappa': 1.8,
    'phi_VZoI': 30.0, 'delta_VZoI': 0.17, 'D_VZoI_min': 1.0, 'D_VZoI_max': 33.0,
}

HUBBLE_VH_PARAMS = {
    'delta_inf': 0.031, 'alpha': -0.9, 'D_tr': 44.0, 'kappa': 1.0,
    'phi_VZoI': 45.0, 'delta_VZoI': 0.19, 'D_VZoI_min': 1.0, 'D_VZoI_max': 33.0,
}


def haubner_delta_f(D_Mpc, params=CF4_PARAMS):
    """Haubner+2025 Eq.7: distance uncertainty in dex."""
    D = np.maximum(np.asarray(D_Mpc, dtype=float), 0.01)
    d_inf = params['delta_inf']
    alpha = params['alpha']
    D_tr = params['D_tr']
    kappa = params['kappa']
    return d_inf * D**alpha * (D**(1.0/kappa) + D_tr**(1.0/kappa))**(-alpha*kappa)


# ============================================================
# PARSE WALLABY DATA
# ============================================================
def parse_array_field(field_str):
    """Parse a comma-separated array field from WALLABY CSV."""
    if not field_str or field_str.strip() == '':
        return np.array([])
    try:
        parts = field_str.strip().strip('"').split(',')
        return np.array([float(x) for x in parts])
    except (ValueError, TypeError):
        return np.array([])


def load_wallaby_source_catalog():
    """Load WALLABY source catalogs (DR1+DR2) for distances and HI masses."""
    sources = {}

    for fname in ['wallaby_dr2_source_catalogue.csv', 'wallaby_dr1_source_catalogue.csv']:
        fpath = os.path.join(WALLABY_DIR, fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name'].strip()
                if name in sources:
                    continue  # DR2 takes priority

                try:
                    src = {
                        'name': name,
                        'ra': float(row['ra']),
                        'dec': float(row['dec']),
                        'freq': float(row['freq']),
                        'w20': float(row.get('w20', 0)),
                        'w50': float(row.get('w50', 0)),
                        'qflag': float(row.get('qflag', -1)),
                        'kflag': float(row.get('kflag', -1)),
                    }

                    # Hubble distance (Mpc)
                    if 'dist_h' in row and row['dist_h'].strip():
                        src['dist_h'] = float(row['dist_h'])
                    else:
                        # Compute from frequency: v = c * (f_HI - f) / f
                        v_sys = c_light * (HI_freq - src['freq']) / src['freq']
                        src['dist_h'] = v_sys / H0

                    # HI mass
                    if 'log_m_hi' in row and row['log_m_hi'].strip():
                        src['log_mhi'] = float(row['log_m_hi'])

                    if 'log_m_hi_corr' in row and row['log_m_hi_corr'].strip():
                        src['log_mhi_corr'] = float(row['log_m_hi_corr'])

                    sources[name] = src

                except (ValueError, KeyError) as e:
                    continue

    return sources


def load_wallaby_kinematic_catalog():
    """
    Load WALLABY kinematic catalogs (DR1+DR2).

    Returns dict of galaxies with rotation curves and HI surface density profiles.
    """
    galaxies = {}

    for fname in ['wallaby_dr2_kinematic_catalogue.csv', 'wallaby_dr1_kinematic_catalogue.csv']:
        fpath = os.path.join(WALLABY_DIR, fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name'].strip()
                if name in galaxies:
                    continue  # DR2 takes priority

                try:
                    g = {
                        'name': name,
                        'ra': float(row['ra']),
                        'dec': float(row['dec']),
                        'freq': float(row['freq']),
                        'Vsys': float(row['Vsys_model']),
                        'Inc': float(row['Inc_model']),
                        'eInc': float(row['e_Inc_model']),
                        'PA': float(row['PA_model']),
                        'QFlag': float(row.get('QFlag_model', 0)),
                        'team_release': row.get('team_release', ''),
                        'team_release_kin': row.get('team_release_kin', ''),
                    }

                    # Parse rotation curve arrays
                    g['Rad'] = parse_array_field(row.get('Rad', ''))
                    g['Vrot'] = parse_array_field(row.get('Vrot_model', ''))
                    g['eVrot'] = parse_array_field(row.get('e_Vrot_model', ''))
                    g['eVrot_inc'] = parse_array_field(row.get('e_Vrot_model_inc', ''))

                    # Parse HI surface density arrays
                    g['Rad_SD'] = parse_array_field(row.get('Rad_SD', ''))
                    g['SD'] = parse_array_field(row.get('SD_model', ''))
                    g['eSD'] = parse_array_field(row.get('e_SD_model', ''))
                    g['SD_FO'] = parse_array_field(row.get('SD_FO_model', ''))
                    g['eSD_FO'] = parse_array_field(row.get('e_SD_FO_model_inc', ''))

                    # Quality checks
                    if len(g['Rad']) < 3 or len(g['Vrot']) < 3:
                        continue
                    if len(g['Rad']) != len(g['Vrot']):
                        min_len = min(len(g['Rad']), len(g['Vrot']))
                        g['Rad'] = g['Rad'][:min_len]
                        g['Vrot'] = g['Vrot'][:min_len]
                        g['eVrot'] = g['eVrot'][:min_len] if len(g['eVrot']) >= min_len else g['eVrot']

                    galaxies[name] = g

                except (ValueError, KeyError) as e:
                    continue

    return galaxies


# ============================================================
# GAS-ONLY RAR COMPUTATION
# ============================================================
def compute_vgas_from_sd(R_kpc, SD_Msun_pc2):
    """
    Compute gas rotation velocity from HI surface density profile.

    Uses the thin-disk approximation:
    V_gas²(R) = 4πG * Σ_gas * R * I(y)

    where I(y) is an integral over Bessel functions.

    For a simplified approach, we use the cumulative mass method:
    V²(R) = G * M(<R) / R

    With the 1.33 helium correction factor.
    """
    if len(R_kpc) < 2 or len(SD_Msun_pc2) < 2:
        return np.zeros_like(R_kpc)

    # Convert surface density from 10^20 cm^-2 to M_sun/pc^2
    # WALLABY SD_model is in units of 10^20 atoms/cm^2
    # Convert: N_HI [cm^-2] * m_H / M_sun * (pc/cm)^2
    # 1 M_sun/pc^2 = 1.249e20 atoms/cm^2
    # So SD [M_sun/pc^2] = SD_model / 1.249
    # With 1.33 He correction:
    SD_gas = SD_Msun_pc2 * 1.33  # Include helium

    G = 4.302e-3  # pc M_sun^-1 (km/s)^2

    # Cumulative mass in annular rings
    R_pc = R_kpc * 1000.0
    Vgas = np.zeros_like(R_kpc)

    for i in range(len(R_kpc)):
        if R_pc[i] <= 0:
            continue
        # Integrate surface density to get enclosed mass
        # M(<R) = sum of 2π * R' * Σ(R') * ΔR' for R' < R
        mask = np.arange(len(R_kpc)) <= i
        if np.sum(mask) < 1:
            continue

        # Trapezoidal integration
        R_ring = R_pc[mask]
        SD_ring = SD_gas[mask] if len(SD_gas) > i else SD_gas[:len(R_ring)]
        if len(SD_ring) < len(R_ring):
            SD_ring = np.interp(R_ring, R_pc[:len(SD_gas)], SD_gas)

        # Use scipy or manual trapezoid (np.trapz removed in numpy 2.0+)
        try:
            M_enc = np.trapezoid(2 * np.pi * R_ring * SD_ring, R_ring)
        except AttributeError:
            # Manual trapezoidal integration
            y = 2 * np.pi * R_ring * SD_ring
            if len(y) < 2:
                M_enc = 0.0
            else:
                M_enc = np.sum((y[:-1] + y[1:]) / 2 * np.diff(R_ring))

        if M_enc > 0:
            Vgas[i] = np.sqrt(G * M_enc / R_pc[i])

    return Vgas  # km/s


def compute_rar_gas_only(g, D_Mpc):
    """
    Compute RAR using gas-only baryonic acceleration.

    This is valid at low accelerations where gas dominates.
    At high accelerations (inner regions), this underestimates gbar.

    We flag points where gas fraction is likely high (outer disk).
    """
    # Convert radii from arcsec to kpc
    R_arcsec = g['Rad']
    R_kpc = R_arcsec * D_Mpc * 1e3 * np.pi / (180 * 3600)  # arcsec -> kpc
    # Simpler: R_kpc = R_arcsec * D_Mpc / 206.265
    R_kpc = R_arcsec * D_Mpc / 206.265

    Vrot = g['Vrot']
    eVrot = g['eVrot'] if len(g['eVrot']) == len(Vrot) else np.ones_like(Vrot) * 5.0

    # Compute gobs
    valid = (R_kpc > 0) & (Vrot > 0)
    if np.sum(valid) < 3:
        return None

    R_v = R_kpc[valid]
    Vrot_v = Vrot[valid]
    eVrot_v = eVrot[valid] if len(eVrot) == len(Vrot) else np.ones_like(Vrot_v) * 5.0

    gobs = Vrot_v**2 / R_v * conv  # m/s^2

    # Compute gas acceleration from HI surface density
    # Interpolate SD onto rotation curve radii
    if len(g['SD_FO']) > 1 and len(g['Rad_SD']) > 1:
        # Use face-on corrected surface density
        # SD_FO is in units of M_sun/pc^2 (face-on corrected)
        SD_interp = np.interp(R_arcsec[valid], g['Rad_SD'], g['SD_FO'],
                              left=g['SD_FO'][0], right=0.0)

        # Compute V_gas from cumulative mass
        Vgas = compute_vgas_from_sd(R_v, SD_interp)
        ggas = np.where(R_v > 0, Vgas**2 / R_v * conv, 0)
    elif len(g['SD']) > 1 and len(g['Rad_SD']) > 1:
        SD_interp = np.interp(R_arcsec[valid], g['Rad_SD'], g['SD'],
                              left=g['SD'][0], right=0.0)
        Vgas = compute_vgas_from_sd(R_v, SD_interp)
        ggas = np.where(R_v > 0, Vgas**2 / R_v * conv, 0)
    else:
        return None

    # Filter to valid points with non-zero gas acceleration
    ok = (gobs > 0) & (ggas > 0)
    if np.sum(ok) < 3:
        return None

    gobs_ok = gobs[ok]
    ggas_ok = ggas[ok]
    R_ok = R_v[ok]
    Vrot_ok = Vrot_v[ok]
    eVrot_ok = eVrot_v[ok]

    # Use gas acceleration as a LOWER BOUND on gbar
    # gbar_true = ggas + gstars >= ggas
    # At low accelerations (outer disk), ggas >> gstars, so gbar ≈ ggas
    gbar_gas = ggas_ok

    # RAR prediction
    gobs_pred = gbar_gas / (1 - np.exp(-np.sqrt(gbar_gas / g_dagger)))

    log_gbar = np.log10(gbar_gas)
    log_gobs = np.log10(gobs_ok)
    log_gobs_pred = np.log10(gobs_pred)
    log_res = log_gobs - log_gobs_pred

    # Observational uncertainty on log(gobs)
    sigma_log_gobs = 2.0 * eVrot_ok / np.maximum(Vrot_ok, 1) / np.log(10)

    # Flag "gas-dominated" points (outer disk, low acceleration)
    # These are where our gas-only gbar is most reliable
    gas_dominated = log_gbar < -10.5

    return {
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'log_gobs_pred': log_gobs_pred,
        'log_res': log_res,
        'sigma_log_gobs': sigma_log_gobs,
        'gas_dominated': gas_dominated,
        'n_points': len(log_gbar),
        'n_gas_dominated': int(np.sum(gas_dominated)),
        'R_kpc': R_ok,
    }


# ============================================================
# DESI DR9 YANG CATALOG CROSS-MATCH
# ============================================================
def crossmatch_wallaby_desi_efficient(wallaby_galaxies, match_radius_arcsec=30.0):
    """
    Cross-match WALLABY galaxies with DESI DR9 Yang catalog.
    Uses memory-efficient approach: loads DESI data in chunks and uses
    spatial pre-filtering to avoid loading the full 134M-galaxy catalog.

    DESI DR9 structure:
      - galaxy.fits: IGAL, RA, DEC, Z columns (134M rows)
      - group.fits: IGRP, RICH, GRP_RA, GRP_DEC, GRP_Z, GRP_LOGM, GRP_LOGL (99.6M rows)
      - index.fits: IGAL, IGRP, RANK (134M rows, maps galaxies to groups)

    Returns dict mapping WALLABY name -> environment info.
    """
    yang_dir = os.path.join(DATA_DIR, 'yang_catalogs')
    galaxy_fits = os.path.join(yang_dir, 'DESIDR9.y1.v1_galaxy.fits')
    group_fits = os.path.join(yang_dir, 'DESIDR9.y1.v1_group.fits')
    index_fits = os.path.join(yang_dir, 'iDESIDR9.y1.v1_1.fits')

    if not all(os.path.exists(f) for f in [galaxy_fits, group_fits, index_fits]):
        print("  DESI DR9 FITS files not found. Skipping.")
        return {}

    # Check for cached WALLABY-DESI crossmatch
    cache_file = os.path.join(DATA_DIR, 'wallaby_desi_crossmatch.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached)} cached WALLABY-DESI matches")
        return cached

    try:
        from astropy.io import fits
    except ImportError:
        print("  astropy not available. Install with: pip install astropy")
        return {}

    print("\n--- Cross-matching WALLABY with DESI DR9 (134M galaxies) ---")

    # Get WALLABY sky coverage for pre-filtering
    w_ras = np.array([g['ra'] for g in wallaby_galaxies.values()])
    w_decs = np.array([g['dec'] for g in wallaby_galaxies.values()])
    w_names = list(wallaby_galaxies.keys())
    w_vsys = np.array([g['Vsys'] for g in wallaby_galaxies.values()])

    ra_min, ra_max = w_ras.min() - 1.0, w_ras.max() + 1.0
    dec_min, dec_max = w_decs.min() - 1.0, w_decs.max() + 1.0
    z_max = np.max(w_vsys) / c_light + 0.01

    print(f"  WALLABY sky coverage: RA [{ra_min:.1f}, {ra_max:.1f}], "
          f"DEC [{dec_min:.1f}, {dec_max:.1f}], z < {z_max:.4f}")

    match_radius_deg = match_radius_arcsec / 3600.0

    # Strategy: read DESI in chunks, pre-filter by sky position
    # Since DESI is 134M rows, process in chunks of 5M
    chunk_size = 5_000_000
    results = {}

    print(f"  Loading DESI galaxy catalog in chunks...")
    with fits.open(galaxy_fits, memmap=True) as gal_hdu, \
         fits.open(index_fits, memmap=True) as idx_hdu, \
         fits.open(group_fits, memmap=True) as grp_hdu:

        gal_data = gal_hdu[1].data
        idx_data = idx_hdu[1].data
        grp_data = grp_hdu[1].data
        n_total = len(gal_data)

        # Pre-load group halo masses (smaller catalog, ~100M but mainly need logM)
        # Actually too big to load all at once. We'll look up individually.
        print(f"  Total DESI galaxies: {n_total:,}")

        for chunk_start in range(0, n_total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_total)

            # Read chunk coordinates
            chunk_ra = gal_data['RA'][chunk_start:chunk_end]
            chunk_dec = gal_data['DEC'][chunk_start:chunk_end]
            chunk_z = gal_data['Z'][chunk_start:chunk_end]

            # Pre-filter: only keep galaxies in WALLABY sky region
            # Handle RA wrapping
            if ra_min < 0:
                ra_mask = (chunk_ra > ra_min + 360) | (chunk_ra < ra_max)
            elif ra_max > 360:
                ra_mask = (chunk_ra > ra_min) | (chunk_ra < ra_max - 360)
            else:
                ra_mask = (chunk_ra >= ra_min) & (chunk_ra <= ra_max)

            sky_mask = ra_mask & (chunk_dec >= dec_min) & (chunk_dec <= dec_max) & (chunk_z <= z_max)

            n_in_sky = np.sum(sky_mask)
            if n_in_sky == 0:
                continue

            # Get indices of sky-filtered galaxies (relative to full catalog)
            sky_indices = np.where(sky_mask)[0] + chunk_start

            # Get their coordinates
            sky_ra = chunk_ra[sky_mask]
            sky_dec = chunk_dec[sky_mask]
            sky_z = chunk_z[sky_mask]

            # Cross-match each WALLABY galaxy
            for wi in range(len(w_names)):
                w_name = w_names[wi]
                if w_name in results:
                    continue

                wr = w_ras[wi]
                wd = w_decs[wi]
                wz = w_vsys[wi] / c_light

                cos_dec = np.cos(np.radians(wd))
                dra = (sky_ra - wr) * cos_dec
                ddec = sky_dec - wd
                sep = np.sqrt(dra**2 + ddec**2)

                # Redshift consistency
                dz = np.abs(sky_z - wz)
                good = (sep < match_radius_deg) & (dz < 0.01)

                if np.sum(good) > 0:
                    best_j = np.argmin(sep[good] + 100 * dz[good])
                    best_full_idx = sky_indices[np.where(good)[0][best_j]]

                    # Look up group membership
                    grp_id = int(idx_data['IGRP'][best_full_idx])

                    if grp_id > 0 and grp_id < len(grp_data):
                        logMh = float(grp_data['GRP_LOGM'][grp_id])
                        grp_z = float(grp_data['GRP_Z'][grp_id])
                        richness = int(grp_data['RICH'][grp_id])

                        if logMh >= 14.0:
                            env_class, env_dense = 'cluster', 'dense'
                        elif logMh >= 13.0:
                            env_class, env_dense = 'rich_group', 'dense'
                        elif logMh >= 12.5:
                            env_class, env_dense = 'group', 'dense'
                        elif logMh >= 11.0:
                            env_class, env_dense = 'poor_group', 'field'
                        else:
                            env_class, env_dense = 'field', 'field'

                        results[w_name] = {
                            'logMh': logMh,
                            'env_class': env_class,
                            'env_dense': env_dense,
                            'match_sep_arcsec': float(np.min(sep[good]) * 3600),
                            'desi_grp_id': grp_id,
                            'grp_z': grp_z,
                            'richness': richness,
                        }

            print(f"    Chunk {chunk_start//chunk_size + 1}/"
                  f"{(n_total + chunk_size - 1)//chunk_size}: "
                  f"{n_in_sky} in sky region, {len(results)} matches so far")

            # Early exit if all matched
            if len(results) >= len(w_names):
                break

    # Cache results
    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  DESI cross-match complete: {len(results)}/{len(w_names)} matched")

    # Summary
    env_summary = {}
    for v in results.values():
        ec = v['env_class']
        env_summary[ec] = env_summary.get(ec, 0) + 1
    for ec, n in sorted(env_summary.items(), key=lambda x: -x[1]):
        print(f"    {ec}: {n}")

    return results


# ============================================================
# CF4 DISTANCE INTEGRATION
# ============================================================
def fetch_cf4_distances_batch(galaxies, cache_file=None):
    """
    Get CF4 distances for WALLABY galaxies.
    Uses the EDD CF4 calculator batch API.
    """
    import urllib.request
    import ssl

    if cache_file is None:
        cache_file = os.path.join(DATA_DIR, 'wallaby_cf4_cache.json')

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached WALLABY CF4 distances")

    to_query = {}
    results = {}

    for name, g in galaxies.items():
        if name in cache and cache[name].get('D_cf4'):
            results[name] = cache[name]
        else:
            to_query[name] = g

    if not to_query:
        return results

    print(f"  Querying {len(to_query)} WALLABY galaxies from EDD CF4...")

    # Build batch query
    batch = []
    name_order = []
    for name, g in to_query.items():
        if g['Vsys'] > 0:
            batch.append({
                "coordinate": [g['ra'], g['dec']],
                "system": "equatorial",
                "parameter": "velocity",
                "value": g['Vsys']
            })
            name_order.append(name)

    if not batch:
        return results

    # Query in chunks of 50
    chunk_size = 50
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for chunk_start in range(0, len(batch), chunk_size):
        chunk = batch[chunk_start:chunk_start + chunk_size]
        names_chunk = name_order[chunk_start:chunk_start + chunk_size]

        payload = json.dumps({"galaxies": chunk}).encode('utf-8')

        try:
            url = "https://edd.ifa.hawaii.edu/CF4calculator/api.php"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'WALLABY-CF4-Pipeline/1.0 (research)'
                },
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                response_data = json.loads(resp.read().decode('utf-8'))

            # Parse response: API returns {"results": [...]} with nested
            # "observed": {"distance": [...]} structure
            entries = []
            if isinstance(response_data, dict) and 'results' in response_data:
                entries = response_data['results']
            elif isinstance(response_data, dict) and 'galaxies' in response_data:
                entries = response_data['galaxies']
            elif isinstance(response_data, list):
                entries = response_data

            for j, entry in enumerate(entries):
                if j >= len(names_chunk):
                    break
                gname = names_chunk[j]
                if not isinstance(entry, dict):
                    results[gname] = {'D_cf4': None, 'status': 'bad_entry'}
                    continue

                # Extract distance: check observed->distance, then distance
                d_vals = None
                obs = entry.get('observed', {})
                if isinstance(obs, dict) and 'distance' in obs:
                    d_vals = obs['distance']
                elif 'distance' in entry:
                    d_vals = entry['distance']

                if d_vals is not None:
                    if isinstance(d_vals, list) and len(d_vals) > 0:
                        results[gname] = {
                            'D_cf4': d_vals[0],
                            'D_cf4_all': d_vals,
                            'n_solutions': len(d_vals),
                            'V_input': to_query[gname]['Vsys'],
                            'status': 'ok'
                        }
                    elif isinstance(d_vals, (int, float)):
                        results[gname] = {
                            'D_cf4': float(d_vals),
                            'V_input': to_query[gname]['Vsys'],
                            'status': 'ok'
                        }
                    else:
                        results[gname] = {'D_cf4': None, 'status': 'no_distance'}
                else:
                    results[gname] = {
                        'D_cf4': None,
                        'status': entry.get('message', 'unknown_format'),
                        'response': str(entry)[:200]
                    }

            print(f"    Processed chunk {chunk_start//chunk_size + 1}/"
                  f"{(len(batch) + chunk_size - 1)//chunk_size}")

        except Exception as e:
            print(f"    CF4 query error (chunk {chunk_start}): {e}")
            # Try GET method as fallback
            for j, entry in enumerate(chunk):
                if j < len(names_chunk):
                    gname = names_chunk[j]
                    try:
                        params = (
                            f"ra={entry['coordinate'][0]:.6f}&"
                            f"dec={entry['coordinate'][1]:.6f}&"
                            f"velocity={entry['value']:.1f}"
                        )
                        get_url = f"{url}?{params}"
                        get_req = urllib.request.Request(get_url)
                        get_req.add_header('User-Agent', 'WALLABY-CF4/1.0')
                        with urllib.request.urlopen(get_req, timeout=30, context=ctx) as resp:
                            resp_data = json.loads(resp.read().decode('utf-8'))
                            if 'distance' in resp_data:
                                d = resp_data['distance']
                                results[gname] = {
                                    'D_cf4': d[0] if isinstance(d, list) else d,
                                    'status': 'ok_get'
                                }
                    except:
                        pass
            continue

    # Update cache
    all_cached = {**cache, **results}
    with open(cache_file, 'w') as f:
        json.dump(all_cached, f, indent=2, default=str)

    n_ok = sum(1 for v in results.values() if v.get('D_cf4'))
    print(f"  Got CF4 distances for {n_ok}/{len(to_query)} queried galaxies")

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_wallaby_pipeline(use_cf4=True, use_desi=True, gas_only_threshold=-10.5):
    """
    Run the WALLABY expanded-sample RAR environmental test.

    Args:
        use_cf4: Use CF4 flow distances instead of Hubble distances
        use_desi: Cross-match with DESI DR9 Yang catalog for environments
        gas_only_threshold: log(gbar) threshold below which gas dominates
    """
    print("=" * 80)
    print("WALLABY EXPANDED-SAMPLE RAR PIPELINE")
    print(f"CF4 distances: {'YES' if use_cf4 else 'NO (Hubble flow)'}")
    print(f"DESI DR9 environments: {'YES' if use_desi else 'NO'}")
    print(f"Gas-dominated threshold: log(gbar) < {gas_only_threshold}")
    print("=" * 80)

    # 1. Load WALLABY data
    print("\n--- Loading WALLABY data ---")
    sources = load_wallaby_source_catalog()
    print(f"  Source catalog: {len(sources)} galaxies")

    kin = load_wallaby_kinematic_catalog()
    print(f"  Kinematic catalog: {len(kin)} galaxies with rotation curves")

    # 2. Cross-reference: merge source + kinematic data
    print("\n--- Merging catalogs ---")
    merged = {}
    for name, g in kin.items():
        if name in sources:
            g['dist_h'] = sources[name].get('dist_h', None)
            g['log_mhi'] = sources[name].get('log_mhi', None)
            g['w50'] = sources[name].get('w50', 0)
            g['qflag_source'] = sources[name].get('qflag', -1)
        else:
            # Compute distance from Vsys
            if g['Vsys'] > 0:
                g['dist_h'] = g['Vsys'] / H0
            else:
                g['dist_h'] = None

        merged[name] = g

    print(f"  Merged: {len(merged)} galaxies")

    # 3. Quality cuts
    print("\n--- Quality cuts ---")
    galaxies_clean = {}
    cut_counts = {}

    for name, g in merged.items():
        # Distance cut
        if g.get('dist_h') is None or g['dist_h'] <= 0 or g['dist_h'] > 200:
            cut_counts['bad_dist'] = cut_counts.get('bad_dist', 0) + 1
            continue

        # Inclination cut (same as SPARC: 30 < Inc < 85)
        if g['Inc'] < 30 or g['Inc'] > 85:
            cut_counts['Inc'] = cut_counts.get('Inc', 0) + 1
            continue

        # Minimum points
        if len(g['Rad']) < 5:
            cut_counts['n_pts'] = cut_counts.get('n_pts', 0) + 1
            continue

        # Quality flag (keep 0 and 1; reject 2, 3)
        if g['QFlag'] > 1.5:
            cut_counts['QFlag'] = cut_counts.get('QFlag', 0) + 1
            continue

        # Velocity sanity (at least some points > 20 km/s)
        if np.max(g['Vrot']) < 20:
            cut_counts['low_V'] = cut_counts.get('low_V', 0) + 1
            continue

        galaxies_clean[name] = g

    print(f"  After cuts: {len(galaxies_clean)} galaxies")
    for reason, count in sorted(cut_counts.items(), key=lambda x: -x[1]):
        print(f"    Cut by {reason}: {count}")

    # 4. Get distances
    print("\n--- Determining distances ---")
    cf4_distances = {}
    if use_cf4:
        cf4_distances = fetch_cf4_distances_batch(galaxies_clean)

    for name, g in galaxies_clean.items():
        if use_cf4 and name in cf4_distances and cf4_distances[name].get('D_cf4'):
            g['D_use'] = cf4_distances[name]['D_cf4']
            g['dist_source'] = 'CF4'
        else:
            g['D_use'] = g['dist_h']
            g['dist_source'] = 'Hubble'

        # Haubner uncertainty
        if g['dist_source'] == 'CF4':
            g['sigma_D_dex'] = float(haubner_delta_f(g['D_use'], CF4_PARAMS))
        else:
            g['sigma_D_dex'] = float(haubner_delta_f(g['D_use'], HUBBLE_VH_PARAMS))

    dist_stats = {}
    for g in galaxies_clean.values():
        src = g['dist_source']
        dist_stats[src] = dist_stats.get(src, 0) + 1
    for src, n in dist_stats.items():
        print(f"  {src}: {n}")

    # 5. Environment classification
    print("\n--- Environment classification ---")
    env_catalog = {}

    # Load literature-based environment catalog
    wallaby_env_path = os.path.join(DATA_DIR, 'wallaby_environment_catalog.json')
    if os.path.exists(wallaby_env_path):
        with open(wallaby_env_path, 'r') as f:
            env_catalog = json.load(f)
        print(f"  Loaded WALLABY environment catalog ({len(env_catalog)} entries)")
    elif use_desi:
        env_catalog = crossmatch_wallaby_desi_efficient(galaxies_clean)
    else:
        env_catalog = {}

    # Assign environments
    for name, g in galaxies_clean.items():
        if name in env_catalog:
            g['logMh'] = float(env_catalog[name].get('logMh', 11.0))
            g['env_class'] = env_catalog[name].get('env_class', 'field')
            g['env_dense'] = env_catalog[name].get('env_dense', 'field')
        else:
            # Default to field for unmatched galaxies
            g['logMh'] = 11.0
            g['env_class'] = 'field'
            g['env_dense'] = 'field'

    env_counts = {'dense': 0, 'field': 0}
    for g in galaxies_clean.values():
        env_counts[g['env_dense']] = env_counts.get(g['env_dense'], 0) + 1
    print(f"  Dense: {env_counts.get('dense', 0)}, Field: {env_counts.get('field', 0)}")

    # 6. Compute RAR
    print("\n--- Computing gas-only RAR ---")
    all_results = []
    n_failed = 0

    for name, g in galaxies_clean.items():
        D = g['D_use']
        rar = compute_rar_gas_only(g, D)

        if rar is None:
            n_failed += 1
            continue

        g['rar'] = rar
        all_results.append(g)

    print(f"  Computed RAR: {len(all_results)} galaxies ({n_failed} failed)")

    total_pts = sum(g['rar']['n_points'] for g in all_results)
    total_gas = sum(g['rar']['n_gas_dominated'] for g in all_results)
    print(f"  Total data points: {total_pts}")
    print(f"  Gas-dominated points (log gbar < {gas_only_threshold}): {total_gas}")

    # 7. Overall scatter
    all_res = np.concatenate([g['rar']['log_res'] for g in all_results])
    all_gbar = np.concatenate([g['rar']['log_gbar'] for g in all_results])

    # Gas-dominated subset
    gas_mask = all_gbar < gas_only_threshold
    all_res_gas = all_res[gas_mask]
    all_gbar_gas = all_gbar[gas_mask]

    print(f"\n  Overall RAR scatter (all points): {np.std(all_res):.4f} dex")
    print(f"  Overall RAR scatter (gas-dom): {np.std(all_res_gas):.4f} dex")
    print(f"  Mean residual (all): {np.mean(all_res):.4f} dex")
    print(f"  Mean residual (gas-dom): {np.mean(all_res_gas):.4f} dex")

    # 8. Environmental test
    print("\n" + "=" * 80)
    print("ENVIRONMENTAL RAR SCATTER TEST (WALLABY)")
    print("=" * 80)

    # All-points test
    dense_res = np.concatenate([g['rar']['log_res'] for g in all_results
                                if g['env_dense'] == 'dense']) if any(
        g['env_dense'] == 'dense' for g in all_results) else np.array([])
    field_res = np.concatenate([g['rar']['log_res'] for g in all_results
                                if g['env_dense'] == 'field']) if any(
        g['env_dense'] == 'field' for g in all_results) else np.array([])

    dense_gbar = np.concatenate([g['rar']['log_gbar'] for g in all_results
                                  if g['env_dense'] == 'dense']) if any(
        g['env_dense'] == 'dense' for g in all_results) else np.array([])
    field_gbar = np.concatenate([g['rar']['log_gbar'] for g in all_results
                                  if g['env_dense'] == 'field']) if any(
        g['env_dense'] == 'field' for g in all_results) else np.array([])

    n_dense_gal = sum(1 for g in all_results if g['env_dense'] == 'dense')
    n_field_gal = sum(1 for g in all_results if g['env_dense'] == 'field')

    print(f"\n  All points:")
    if len(dense_res) > 0 and len(field_res) > 0:
        print(f"    Dense: {len(dense_res)} pts from {n_dense_gal} galaxies, "
              f"σ={np.std(dense_res):.4f}")
        print(f"    Field: {len(field_res)} pts from {n_field_gal} galaxies, "
              f"σ={np.std(field_res):.4f}")
        delta = np.std(field_res) - np.std(dense_res)
        print(f"    Δσ (field - dense): {delta:+.4f}")

        # Bootstrap
        n_boot = 10000
        np.random.seed(42)
        combined = np.concatenate([dense_res, field_res])
        nd = len(dense_res)
        boot_deltas = np.zeros(n_boot)
        for i in range(n_boot):
            s = np.random.permutation(combined)
            boot_deltas[i] = np.std(s[nd:]) - np.std(s[:nd])
        p_all = np.mean(boot_deltas >= delta)
        print(f"    P(field > dense): {1-p_all:.4f}")
    else:
        print("    Insufficient dense/field separation!")
        p_all = 0.5

    # Gas-dominated test
    print(f"\n  Gas-dominated points (log gbar < {gas_only_threshold}):")
    dense_mask_gas = dense_gbar < gas_only_threshold
    field_mask_gas = field_gbar < gas_only_threshold
    dense_res_gas = dense_res[dense_mask_gas] if len(dense_res) > 0 else np.array([])
    field_res_gas = field_res[field_mask_gas] if len(field_res) > 0 else np.array([])

    if len(dense_res_gas) > 5 and len(field_res_gas) > 5:
        print(f"    Dense: {len(dense_res_gas)} pts, σ={np.std(dense_res_gas):.4f}")
        print(f"    Field: {len(field_res_gas)} pts, σ={np.std(field_res_gas):.4f}")
        delta_gas = np.std(field_res_gas) - np.std(dense_res_gas)
        print(f"    Δσ (field - dense): {delta_gas:+.4f}")

        combined_gas = np.concatenate([dense_res_gas, field_res_gas])
        nd_gas = len(dense_res_gas)
        boot_gas = np.zeros(n_boot)
        for i in range(n_boot):
            s = np.random.permutation(combined_gas)
            boot_gas[i] = np.std(s[nd_gas:]) - np.std(s[:nd_gas])
        p_gas = np.mean(boot_gas >= delta_gas)
        print(f"    P(field > dense): {1-p_gas:.4f}")
    else:
        print(f"    Insufficient gas-dominated points for test")
        print(f"    Dense: {len(dense_res_gas)}, Field: {len(field_res_gas)}")
        delta_gas = 0
        p_gas = 0.5

    # 9. Binned analysis
    print("\n--- Binned Analysis (gas-dominated regime) ---")
    bin_edges = np.array([-13.0, -12.0, -11.0, -10.0, -9.0])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned = []

    for j in range(len(bin_centers)):
        lo, hi = bin_edges[j], bin_edges[j+1]
        d_mask = (dense_gbar >= lo) & (dense_gbar < hi) if len(dense_gbar) > 0 else np.array([], bool)
        f_mask = (field_gbar >= lo) & (field_gbar < hi) if len(field_gbar) > 0 else np.array([], bool)

        d_r = dense_res[d_mask] if np.sum(d_mask) > 0 else np.array([])
        f_r = field_res[f_mask] if np.sum(f_mask) > 0 else np.array([])

        if len(d_r) >= 5 and len(f_r) >= 5:
            d_std = np.std(d_r)
            f_std = np.std(f_r)
            delta = f_std - d_std

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
        else:
            print(f"  - Bin {bin_centers[j]:.1f}: dense={len(d_r)}, field={len(f_r)} (insufficient)")

    # 10. Save results
    print("\n--- Saving results ---")

    run_name = f"wallaby_{'cf4' if use_cf4 else 'hubble'}_{'desi' if use_desi else 'nodesi'}"

    # Summary JSON
    summary = {
        'pipeline': run_name,
        'n_galaxies': len(all_results),
        'n_data_points': total_pts,
        'n_gas_dominated': total_gas,
        'gas_dom_threshold': gas_only_threshold,
        'overall_scatter_dex': float(np.std(all_res)),
        'overall_scatter_gas_dex': float(np.std(all_res_gas)) if len(all_res_gas) > 0 else None,
        'mean_residual_dex': float(np.mean(all_res)),
        'environment': {
            'n_dense': n_dense_gal,
            'n_field': n_field_gal,
            'sigma_dense': float(np.std(dense_res)) if len(dense_res) > 0 else None,
            'sigma_field': float(np.std(field_res)) if len(field_res) > 0 else None,
            'delta_sigma': float(np.std(field_res) - np.std(dense_res)) if len(dense_res) > 0 and len(field_res) > 0 else None,
            'p_field_gt_dense': float(1-p_all) if len(dense_res) > 0 and len(field_res) > 0 else None,
        },
        'environment_gas_dom': {
            'sigma_dense': float(np.std(dense_res_gas)) if len(dense_res_gas) > 5 else None,
            'sigma_field': float(np.std(field_res_gas)) if len(field_res_gas) > 5 else None,
            'delta_sigma': float(delta_gas),
            'p_field_gt_dense': float(1-p_gas),
            'n_dense_pts': int(len(dense_res_gas)),
            'n_field_pts': int(len(field_res_gas)),
        },
        'binned_results': binned,
        'distance_sources': dist_stats,
    }

    summary_path = os.path.join(OUTPUT_DIR, f'summary_{run_name}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    # Per-galaxy CSV
    csv_path = os.path.join(OUTPUT_DIR, f'galaxy_results_{run_name}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'name', 'ra', 'dec', 'Vsys', 'D_use', 'dist_source',
            'sigma_D_dex', 'Inc', 'n_pts', 'n_gas_dom',
            'rms_all', 'rms_gas', 'env_class', 'env_dense', 'logMh',
            'team_release'
        ])
        for g in all_results:
            rar = g['rar']
            gas_mask = rar['gas_dominated']
            writer.writerow([
                g['name'], f"{g['ra']:.6f}", f"{g['dec']:.6f}",
                f"{g['Vsys']:.1f}", f"{g['D_use']:.2f}", g['dist_source'],
                f"{g['sigma_D_dex']:.4f}", f"{g['Inc']:.1f}",
                rar['n_points'], rar['n_gas_dominated'],
                f"{np.std(rar['log_res']):.4f}",
                f"{np.std(rar['log_res'][gas_mask]):.4f}" if np.sum(gas_mask) > 2 else '',
                g['env_class'], g['env_dense'], f"{g['logMh']:.1f}",
                g.get('team_release', ''),
            ])
    print(f"  Galaxy results: {csv_path}")

    # RAR data points CSV
    pts_path = os.path.join(OUTPUT_DIR, f'rar_points_{run_name}.csv')
    with open(pts_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'galaxy', 'log_gbar', 'log_gobs', 'log_res', 'sigma_log_gobs',
            'gas_dominated', 'R_kpc', 'env_dense'
        ])
        for g in all_results:
            rar = g['rar']
            for k in range(rar['n_points']):
                writer.writerow([
                    g['name'],
                    f"{rar['log_gbar'][k]:.6f}",
                    f"{rar['log_gobs'][k]:.6f}",
                    f"{rar['log_res'][k]:.6f}",
                    f"{rar['sigma_log_gobs'][k]:.6f}",
                    int(rar['gas_dominated'][k]),
                    f"{rar['R_kpc'][k]:.3f}",
                    g['env_dense'],
                ])
    print(f"  RAR points: {pts_path}")

    print("\n" + "=" * 80)
    print("WALLABY PIPELINE COMPLETE")
    print("=" * 80)

    return summary


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='WALLABY RAR Pipeline')
    parser.add_argument('--no-cf4', action='store_true',
                        help='Use Hubble distances instead of CF4')
    parser.add_argument('--no-desi', action='store_true',
                        help='Skip DESI DR9 cross-match')
    parser.add_argument('--gas-threshold', type=float, default=-10.5,
                        help='Gas-dominated threshold (default: -10.5)')

    args = parser.parse_args()

    # Run with Hubble distances first (no API dependency)
    summary_hubble = run_wallaby_pipeline(
        use_cf4=False,
        use_desi=not args.no_desi,
        gas_only_threshold=args.gas_threshold,
    )

    # Then with CF4 if requested
    if not args.no_cf4:
        print("\n\n")
        summary_cf4 = run_wallaby_pipeline(
            use_cf4=True,
            use_desi=not args.no_desi,
            gas_only_threshold=args.gas_threshold,
        )
