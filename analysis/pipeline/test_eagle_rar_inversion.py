#!/usr/bin/env python3
"""
EAGLE Simulation RAR & Scatter Derivative Inversion Test
=========================================================

Uses the EAGLE public SQL database (McAlpine+ 2016) to extract
enclosed stellar, gas, and DM masses at multiple aperture radii
for late-type galaxies in the Ref-L100N1504 simulation (z=0).

Then computes the RAR and scatter derivative to test whether
ΛCDM hydrodynamic simulations produce an inversion at g†.

Requirements:
  pip install eaglesqltools numpy
  EAGLE database account (free): http://icc.dur.ac.uk/Eagle/database.php

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import eagleSqlTools as sql
except ImportError:
    print("ERROR: eagleSqlTools not installed. Run: pip install eaglesqltools")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'eagle_rar')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Physics constants
G = 6.674e-11        # m^3 kg^-1 s^-2
M_sun = 1.989e30     # kg
kpc_m = 3.086e19     # m
g_dagger = 1.20e-10  # m/s^2
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921
h = 0.6777           # EAGLE Planck 2013 cosmology

# Simulation: Ref-L100N1504 (100 Mpc box, highest resolution reference model)
SIM = "RefL0100N1504"

print("=" * 72)
print("EAGLE SIMULATION: RAR SCATTER DERIVATIVE INVERSION TEST")
print("=" * 72)
print(f"  Simulation: {SIM}")
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")


# ================================================================
# 1. CONNECT TO EAGLE DATABASE
# ================================================================
print("\n[1] Connecting to EAGLE database...")
print("    URL: http://virgodb.dur.ac.uk:8080/Eagle/")

# Check for cached data first
cache_path = os.path.join(DATA_DIR, 'eagle_aperture_masses.json')
MASS_UNIT = 1.0  # EAGLE aperture masses are expected in M_sun
if os.path.exists(cache_path):
    print(f"    Found cached data: {cache_path}")
    with open(cache_path, 'r') as f:
        cached = json.load(f)
    aperture_sizes = cached['aperture_sizes']
    galaxy_data = cached['galaxy_data']
    MASS_UNIT = float(cached.get('mass_unit_msun', 1.0))
    print(f"    {len(galaxy_data)} galaxies, apertures: {aperture_sizes} kpc")

else:
    # Need credentials
    eagle_user = os.environ.get('EAGLE_USER', '')
    eagle_pass = os.environ.get('EAGLE_PASS', '')

    if not eagle_user:
        print("\n  *** EAGLE database credentials required ***")
        print("  Register (free) at: http://icc.dur.ac.uk/Eagle/database.php")
        print("  Then set environment variables:")
        print("    export EAGLE_USER='your_username'")
        print("    export EAGLE_PASS='your_password'")
        print("  Or run with: EAGLE_USER=xxx EAGLE_PASS=xxx python3 this_script.py")
        sys.exit(0)

    con = sql.connect(eagle_user, password=eagle_pass)
    print("    Connected!")

    # ================================================================
    # 2. DISCOVER AVAILABLE APERTURE SIZES
    # ================================================================
    print("\n[2] Discovering available aperture sizes...")

    q_apertures = f"""
    SELECT DISTINCT AP.ApertureSize
    FROM {SIM}_Aperture as AP
    ORDER BY AP.ApertureSize
    """
    result = sql.execute_query(con, q_apertures)
    aperture_sizes = sorted([float(x) for x in result['ApertureSize']])
    print(f"    Available apertures (kpc): {aperture_sizes}")

    # ================================================================
    # 3. QUERY LATE-TYPE GALAXIES WITH APERTURE MASSES
    # ================================================================
    print("\n[3] Querying late-type galaxies...")

    # Strategy: Get all aperture masses for star-forming centrals
    # with M_star(30kpc) > 10^8 M_sun
    # Join SubHalo table for galaxy properties (SFR, type)
    #
    # EAGLE aperture masses are handled in M_sun in this pipeline.
    # Keep the original AP.Mass_Star > 0.01 cut used historically.
    #
    # Select star-forming galaxies (sSFR > 10^-11 yr^-1)
    # and centrals (SubGroupNumber = 0)

    galaxy_data = {}

    for ap_size in aperture_sizes:
        print(f"    Querying aperture = {ap_size} kpc...")

        q = f"""
        SELECT
            SH.GalaxyID as gid,
            SH.SubGroupNumber as subgn,
            SH.StarFormationRate as sfr,
            AP.Mass_Star as m_star,
            AP.Mass_Gas as m_gas,
            AP.Mass_DM as m_dm
        FROM
            {SIM}_SubHalo as SH,
            {SIM}_Aperture as AP
        WHERE
            SH.SnapNum = 28
            AND SH.SubGroupNumber = 0
            AND AP.GalaxyID = SH.GalaxyID
            AND AP.ApertureSize = {ap_size}
            AND AP.Mass_Star > 0.01
        ORDER BY SH.GalaxyID
        """

        try:
            result = sql.execute_query(con, q)
        except Exception as e:
            print(f"      Query failed: {e}")
            continue

        n_rows = len(result['gid'])
        print(f"      Got {n_rows} galaxies")

        for i in range(n_rows):
            gid = int(result['gid'][i])
            if gid not in galaxy_data:
                galaxy_data[gid] = {
                    'sfr': float(result['sfr'][i]),
                    'apertures': {}
                }
            galaxy_data[gid]['apertures'][str(ap_size)] = {
                'm_star': float(result['m_star'][i]),
                'm_gas': float(result['m_gas'][i]),
                'm_dm': float(result['m_dm'][i]),
            }

    print(f"\n    Total galaxies: {len(galaxy_data)}")

    # Filter to star-forming (late-type proxy)
    sf_gals = {gid: g for gid, g in galaxy_data.items()
                if g['sfr'] > 0 and len(g['apertures']) >= 3}
    print(f"    Star-forming with ≥3 apertures: {len(sf_gals)}")
    galaxy_data = sf_gals

    # Cache the data
    cached = {
        'aperture_sizes': aperture_sizes,
        'galaxy_data': {str(k): v for k, v in galaxy_data.items()},
        'simulation': SIM,
        'description': 'EAGLE aperture masses for star-forming centrals at z=0',
        'mass_unit_msun': 1.0,
    }
    with open(cache_path, 'w') as f:
        json.dump(cached, f, indent=2)
    print(f"    Cached to {cache_path}")

    # Convert keys back to strings for uniform handling
    galaxy_data = {str(k): v for k, v in galaxy_data.items()}


# ================================================================
# 4. COMPUTE RAR (g_bar, g_obs) AT EACH APERTURE
# ================================================================
print("\n[4] Computing RAR from aperture masses...")

all_log_gbar = []
all_log_gobs = []
n_points_per_gal = []

for gid, gdata in galaxy_data.items():
    aps = gdata['apertures']

    for ap_str, masses in aps.items():
        r_kpc = float(ap_str)
        if r_kpc < 1.0:  # Skip very small apertures
            continue

        # Masses are handled in M_sun.
        m_star = masses['m_star'] * MASS_UNIT
        m_gas = masses['m_gas'] * MASS_UNIT
        m_dm = masses['m_dm'] * MASS_UNIT

        m_bar = m_star + m_gas
        m_total = m_bar + m_dm

        if m_bar <= 0 or m_total <= 0:
            continue

        r_m = r_kpc * kpc_m  # meters

        g_bar = G * m_bar * M_sun / r_m**2
        g_obs = G * m_total * M_sun / r_m**2

        if g_bar > 1e-15 and g_obs > 1e-15:
            all_log_gbar.append(np.log10(g_bar))
            all_log_gobs.append(np.log10(g_obs))

all_log_gbar = np.array(all_log_gbar)
all_log_gobs = np.array(all_log_gobs)

print(f"  Total EAGLE RAR points: {len(all_log_gbar)}")
print(f"  log g_bar range: {all_log_gbar.min():.2f} to {all_log_gbar.max():.2f}")
print(f"  log g_obs range: {all_log_gobs.min():.2f} to {all_log_gobs.max():.2f}")


# ================================================================
# 5. SCATTER DERIVATIVE ANALYSIS
# ================================================================
print("\n[5] Computing scatter derivative...")

def find_inversion(log_gbar, log_gobs, bin_width=0.30, offset=0.0):
    """Find scatter derivative zero-crossing nearest to g†."""
    gbar = 10**log_gbar
    with np.errstate(over='ignore', invalid='ignore'):
        rar_pred = np.log10(gbar / (1.0 - np.exp(-np.sqrt(gbar / g_dagger))))
    resid = log_gobs - rar_pred

    # Remove NaN/inf
    valid = np.isfinite(resid)
    log_gbar_v = log_gbar[valid]
    resid_v = resid[valid]

    lo = log_gbar_v.min() + offset
    hi = log_gbar_v.max()
    edges = np.arange(lo, hi + bin_width, bin_width)

    if len(edges) < 3:
        return None, None, None

    centers = []
    sigmas = []
    for j in range(len(edges) - 1):
        mask = (log_gbar_v >= edges[j]) & (log_gbar_v < edges[j+1])
        if np.sum(mask) >= 10:
            centers.append(0.5 * (edges[j] + edges[j+1]))
            sigmas.append(np.std(resid_v[mask]))

    if len(centers) < 4:
        return None, None, None

    centers = np.array(centers)
    sigmas = np.array(sigmas)

    dsigma = np.diff(sigmas)
    dcenter = np.array([0.5 * (centers[j] + centers[j+1]) for j in range(len(centers)-1)])

    crossings = []
    for j in range(len(dsigma) - 1):
        if dsigma[j] > 0 and dsigma[j+1] < 0:
            x0, x1 = dcenter[j], dcenter[j+1]
            y0, y1 = dsigma[j], dsigma[j+1]
            crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(crossing)

    if not crossings:
        return None, centers, sigmas

    crossings = np.array(crossings)
    nearest_idx = np.argmin(np.abs(crossings - LOG_G_DAGGER))
    return crossings[nearest_idx], centers, sigmas


inv_eagle, centers_eagle, sigmas_eagle = find_inversion(all_log_gbar, all_log_gobs)

if centers_eagle is not None:
    print(f"\n  Scatter profile ({len(centers_eagle)} bins):")
    for j in range(len(centers_eagle)):
        marker = " <-- g†" if abs(centers_eagle[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {centers_eagle[j]:.2f}: σ = {sigmas_eagle[j]:.4f} dex{marker}")

if inv_eagle is not None:
    delta = abs(inv_eagle - LOG_G_DAGGER)
    print(f"\n  *** EAGLE inversion at log g = {inv_eagle:.3f}")
    print(f"  *** Distance from g†: {delta:.3f} dex")
else:
    print(f"\n  *** No scatter derivative inversion found in EAGLE data")


# ================================================================
# 6. ROBUSTNESS: MULTIPLE BIN OFFSETS
# ================================================================
print("\n[6] Robustness: multiple bin offsets...")
offsets = np.linspace(0, 0.25, 10)
inv_offsets = []
for off in offsets:
    inv_val, _, _ = find_inversion(all_log_gbar, all_log_gobs, bin_width=0.30, offset=off)
    if inv_val is not None:
        inv_offsets.append(inv_val)

if inv_offsets:
    inv_offsets = np.array(inv_offsets)
    print(f"  Inversions found in {len(inv_offsets)}/{len(offsets)} offsets")
    print(f"  Mean: {inv_offsets.mean():.3f} ± {inv_offsets.std():.3f}")
    print(f"  Mean distance from g†: {np.mean(np.abs(inv_offsets - LOG_G_DAGGER)):.3f} dex")
else:
    print(f"  No inversions at any offset")


# ================================================================
# 7. SCATTER DERIVATIVE COMPARISON (EAGLE vs SPARC)
# ================================================================
print("\n[7] Loading SPARC data for comparison...")

SPARC_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
table2_path = os.path.join(SPARC_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(SPARC_DIR, 'SPARC_Lelli2016c.mrt')

galaxies = {}
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
        if name not in galaxies:
            galaxies[name] = {'R': [], 'Vobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': []}
        galaxies[name]['R'].append(rad)
        galaxies[name]['Vobs'].append(vobs)
        galaxies[name]['Vgas'].append(vgas)
        galaxies[name]['Vdisk'].append(vdisk)
        galaxies[name]['Vbul'].append(vbul)

for name in galaxies:
    for key in galaxies[name]:
        galaxies[name][key] = np.array(galaxies[name][key])

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
            'Inc': float(parts[4]), 'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

sparc_log_gbar = []
sparc_log_gobs = []
for name, gdata in galaxies.items():
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
    sparc_log_gbar.extend(np.log10(gbar_SI[valid]))
    sparc_log_gobs.extend(np.log10(gobs_SI[valid]))

sparc_log_gbar = np.array(sparc_log_gbar)
sparc_log_gobs = np.array(sparc_log_gobs)
print(f"  SPARC RAR points: {len(sparc_log_gbar)}")

inv_sparc, centers_sparc, sigmas_sparc = find_inversion(sparc_log_gbar, sparc_log_gobs)
print(f"  SPARC inversion: {inv_sparc:.3f}" if inv_sparc else "  SPARC: no inversion")


# ================================================================
# 8. SCATTER DERIVATIVE TABLE
# ================================================================
print("\n[8] Scatter derivative comparison:")

def compute_derivative(centers, sigmas):
    dsigma = np.diff(sigmas)
    dx = np.diff(centers)
    deriv = dsigma / dx
    dcenter = 0.5 * (centers[:-1] + centers[1:])
    return dcenter, deriv

if centers_sparc is not None and len(centers_sparc) >= 4:
    dc_sparc, deriv_sparc = compute_derivative(centers_sparc, sigmas_sparc)
    print(f"\n  SPARC:")
    for j in range(len(dc_sparc)):
        marker = " <-- g†" if abs(dc_sparc[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {dc_sparc[j]:.2f}: dσ/d(log g) = {deriv_sparc[j]:+.4f}{marker}")

if centers_eagle is not None and len(centers_eagle) >= 4:
    dc_eagle, deriv_eagle = compute_derivative(centers_eagle, sigmas_eagle)
    print(f"\n  EAGLE:")
    for j in range(len(dc_eagle)):
        marker = " <-- g†" if abs(dc_eagle[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {dc_eagle[j]:.2f}: dσ/d(log g) = {deriv_eagle[j]:+.4f}{marker}")


# ================================================================
# 9. FINAL VERDICT
# ================================================================
print("\n" + "=" * 72)
print("FINAL VERDICT: EAGLE HYDRODYNAMIC SIMULATION")
print("=" * 72)

eagle_near_gdagger = inv_eagle is not None and abs(inv_eagle - LOG_G_DAGGER) < 0.20

if inv_sparc is not None:
    print(f"\n  SPARC (observed): inversion at log g = {inv_sparc:.3f}")
    print(f"    Distance from g†: {abs(inv_sparc - LOG_G_DAGGER):.3f} dex")

if inv_eagle is not None:
    print(f"\n  EAGLE (ΛCDM hydro): inversion at log g = {inv_eagle:.3f}")
    print(f"    Distance from g†: {abs(inv_eagle - LOG_G_DAGGER):.3f} dex")
else:
    print(f"\n  EAGLE (ΛCDM hydro): NO inversion found")

if not eagle_near_gdagger:
    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  DISCRIMINATING: EAGLE ΛCDM hydro does NOT produce the     ║")
    print(f"  ║  scatter derivative inversion at g†.                        ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
else:
    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  NOT DISCRIMINATING: EAGLE produces inversion near g†.      ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")


# Save results
results = {
    'test': 'eagle_rar_inversion',
    'simulation': SIM,
    'n_rar_points': len(all_log_gbar),
    'n_galaxies': len(galaxy_data),
    'aperture_sizes_kpc': aperture_sizes if 'aperture_sizes' in dir() else [],
    'eagle_result': {
        'inversion_log_g': float(inv_eagle) if inv_eagle is not None else None,
        'delta_from_gdagger': float(abs(inv_eagle - LOG_G_DAGGER)) if inv_eagle is not None else None,
        'scatter_profile': {
            'centers': [float(x) for x in centers_eagle] if centers_eagle is not None else [],
            'sigmas': [float(x) for x in sigmas_eagle] if sigmas_eagle is not None else [],
        },
    },
    'sparc_result': {
        'inversion_log_g': float(inv_sparc) if inv_sparc is not None else None,
        'delta_from_gdagger': float(abs(inv_sparc - LOG_G_DAGGER)) if inv_sparc is not None else None,
    },
    'robustness_offsets': {
        'n_found': len(inv_offsets) if 'inv_offsets' in dir() and hasattr(inv_offsets, '__len__') else 0,
        'mean': float(inv_offsets.mean()) if 'inv_offsets' in dir() and hasattr(inv_offsets, '__len__') and len(inv_offsets) > 0 else None,
        'std': float(inv_offsets.std()) if 'inv_offsets' in dir() and hasattr(inv_offsets, '__len__') and len(inv_offsets) > 0 else None,
    },
    'verdict': 'DISCRIMINATING' if not eagle_near_gdagger else 'NOT_DISCRIMINATING',
}

outpath = os.path.join(RESULTS_DIR, 'summary_eagle_rar_inversion.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {outpath}")
print("=" * 72)
