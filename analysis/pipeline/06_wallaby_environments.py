#!/usr/bin/env python3
"""
STEP 6: WALLABY Environment Classification
============================================
Classifies WALLABY galaxies into dense/field environments using:
1. Known galaxy group/cluster memberships from literature
2. Proximity to known clusters (angular + velocity)
3. Local galaxy density estimation from WALLABY source catalog

WALLABY fields are centered on known structures:
  - NGC 5044 group (logMh ~ 13.5, Vsys ~ 2750 km/s)
  - Hydra cluster (Abell 1060, logMh ~ 14.5, Vsys ~ 3777 km/s)
  - Norma cluster (Abell 3627, logMh ~ 15.0, Vsys ~ 4871 km/s)
  - NGC 4636 group (logMh ~ 13.3, Vsys ~ 928 km/s)
  - NGC 4808 / Virgo outskirts
  - Vela overdensity

Russell Licht — Primordial Fluid DM Project
Feb 2026
"""

import numpy as np
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
c_light = 299792.458  # km/s
H0 = 75.0

# ============================================================
# KNOWN GALAXY CLUSTERS/GROUPS IN WALLABY FIELDS
# ============================================================
# Format: {name: (RA, DEC, Vsys_km/s, logMh, R_vir_deg, sigma_v_km/s)}
# R_vir_deg: approximate virial radius in degrees
# sigma_v: velocity dispersion for membership

KNOWN_STRUCTURES = {
    # Major clusters
    'Hydra': (159.18, -27.53, 3777, 14.5, 2.0, 700),
    'Norma': (248.15, -60.75, 4871, 15.0, 1.5, 925),
    'Antlia': (157.48, -35.32, 3041, 13.8, 1.0, 545),

    # Major galaxy groups in WALLABY fields
    'NGC5044': (198.85, -16.39, 2750, 13.5, 1.5, 400),
    'NGC4636': (190.71, 2.69, 928, 13.3, 1.5, 350),
    'NGC5084': (199.95, -21.83, 1730, 13.0, 1.0, 250),
    'NGC4808': (194.17, 4.30, 760, 12.7, 1.0, 200),

    # Smaller groups
    'NGC5054': (199.07, -16.63, 1760, 12.5, 0.8, 200),
    'NGC4697': (192.15, -5.80, 1241, 12.8, 0.8, 200),
    'NGC4038': (180.47, -18.87, 1642, 12.3, 0.5, 150),

    # Vela region structures
    'Vela_overdensity': (150.0, -46.0, 3000, 13.5, 3.0, 500),
}


def angular_separation(ra1, dec1, ra2, dec2):
    """Angular separation in degrees between two sky positions."""
    ra1r, dec1r = np.radians(ra1), np.radians(dec1)
    ra2r, dec2r = np.radians(ra2), np.radians(dec2)
    cos_sep = (np.sin(dec1r) * np.sin(dec2r) +
               np.cos(dec1r) * np.cos(dec2r) * np.cos(ra1r - ra2r))
    cos_sep = np.clip(cos_sep, -1, 1)
    return np.degrees(np.arccos(cos_sep))


def classify_wallaby_environment(ra, dec, vsys, name=''):
    """
    Classify a WALLABY galaxy's environment.

    Returns: (env_class, group_name, env_dense, logMh)
    """
    best_match = None
    best_score = 999

    for struct_name, (s_ra, s_dec, s_vsys, s_logMh, s_rvir, s_sigmav) in KNOWN_STRUCTURES.items():
        ang_sep = angular_separation(ra, dec, s_ra, s_dec)
        vel_sep = abs(vsys - s_vsys)

        # Membership criteria:
        # 1. Within virial radius (angular)
        # 2. Within 2.5 sigma_v velocity dispersion
        if ang_sep < s_rvir and vel_sep < 2.5 * s_sigmav:
            # Score: normalized angular + velocity distance
            score = (ang_sep / s_rvir)**2 + (vel_sep / s_sigmav)**2

            if score < best_score:
                best_score = score
                best_match = (struct_name, s_logMh, ang_sep, vel_sep)

    if best_match:
        struct_name, logMh, ang_sep, vel_sep = best_match

        if logMh >= 14.0:
            env_class = 'cluster'
        elif logMh >= 13.0:
            env_class = 'rich_group'
        elif logMh >= 12.5:
            env_class = 'group'
        else:
            env_class = 'poor_group'

        env_dense = 'dense' if logMh >= 12.5 else 'field'

        return env_class, struct_name, env_dense, logMh

    return 'field', 'field', 'field', 11.0


def compute_local_density(target_ra, target_dec, target_vsys,
                          all_ras, all_decs, all_vsys,
                          n_nearest=5, vel_window=500):
    """
    Compute local galaxy number density using Nth nearest neighbor.

    Args:
        target_ra, target_dec: target galaxy position
        target_vsys: target galaxy velocity
        all_*: arrays of all galaxies in source catalog
        n_nearest: Nth nearest neighbor for density estimate
        vel_window: velocity window for neighbors (km/s)

    Returns: density (galaxies / Mpc^3)
    """
    # Velocity filtering
    vel_mask = np.abs(all_vsys - target_vsys) < vel_window
    if np.sum(vel_mask) < n_nearest + 1:
        return 0.0

    # Angular separations
    cos_dec = np.cos(np.radians(target_dec))
    dra = (all_ras[vel_mask] - target_ra) * cos_dec
    ddec = all_decs[vel_mask] - target_dec
    ang_sep = np.sqrt(dra**2 + ddec**2)

    # Convert to physical separation (Mpc) at galaxy's distance
    D_Mpc = target_vsys / H0 if target_vsys > 0 else 10.0
    phys_sep = ang_sep * np.pi / 180 * D_Mpc  # Mpc

    # Sort and get Nth nearest
    sorted_sep = np.sort(phys_sep)
    if len(sorted_sep) > n_nearest:
        r_n = sorted_sep[n_nearest]  # skip self (index 0)
        if r_n > 0:
            volume = 4/3 * np.pi * r_n**3
            density = n_nearest / volume
            return density

    return 0.0


def build_wallaby_environment_catalog():
    """
    Build environment catalog for all WALLABY galaxies.

    Uses:
    1. Known cluster/group proximity
    2. Local density from WALLABY source catalog
    """
    print("=" * 70)
    print("BUILDING WALLABY ENVIRONMENT CATALOG")
    print("=" * 70)

    # Load all WALLABY source catalog galaxies for density estimation
    print("\n--- Loading WALLABY source catalogs ---")
    all_sources = []
    for fname in ['wallaby_dr2_source_catalogue.csv', 'wallaby_dr1_source_catalogue.csv']:
        fpath = os.path.join(DATA_DIR, 'wallaby', fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row['name'].strip()
                    ra = float(row['ra'])
                    dec = float(row['dec'])
                    freq = float(row['freq'])
                    vsys = 299792.458 * (1.420405751e9 - freq) / freq
                    all_sources.append({
                        'name': name, 'ra': ra, 'dec': dec, 'vsys': vsys
                    })
                except:
                    pass

    print(f"  Loaded {len(all_sources)} source catalog galaxies")

    all_ras = np.array([s['ra'] for s in all_sources])
    all_decs = np.array([s['dec'] for s in all_sources])
    all_vsys = np.array([s['vsys'] for s in all_sources])

    # Load WALLABY kinematic catalog galaxies
    print("\n--- Loading WALLABY kinematic catalog ---")
    kin_galaxies = {}
    seen = set()
    for fname in ['wallaby_dr2_kinematic_catalogue.csv', 'wallaby_dr1_kinematic_catalogue.csv']:
        fpath = os.path.join(DATA_DIR, 'wallaby', fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row['name'].strip()
                    if name in seen:
                        continue
                    seen.add(name)
                    kin_galaxies[name] = {
                        'ra': float(row['ra']),
                        'dec': float(row['dec']),
                        'Vsys': float(row['Vsys_model']),
                        'Inc': float(row['Inc_model']),
                        'team': row.get('team_release', ''),
                    }
                except:
                    pass

    print(f"  Loaded {len(kin_galaxies)} kinematic galaxies")

    # Classify each galaxy
    print("\n--- Classifying environments ---")
    env_catalog = {}

    for name, g in kin_galaxies.items():
        ra, dec, vsys = g['ra'], g['dec'], g['Vsys']

        # 1. Known structure membership
        env_class, group_name, env_dense, logMh = classify_wallaby_environment(
            ra, dec, vsys, name
        )

        # 2. Local density
        density = compute_local_density(
            ra, dec, vsys,
            all_ras, all_decs, all_vsys,
            n_nearest=5, vel_window=500
        )

        env_catalog[name] = {
            'ra': ra,
            'dec': dec,
            'Vsys': vsys,
            'env_class': env_class,
            'group_name': group_name,
            'env_dense': env_dense,
            'logMh': logMh,
            'local_density': density,
            'team_release': g['team'],
        }

    # Summary
    env_summary = {}
    group_summary = {}
    for v in env_catalog.values():
        ec = v['env_class']
        env_summary[ec] = env_summary.get(ec, 0) + 1
        gn = v['group_name']
        group_summary[gn] = group_summary.get(gn, 0) + 1

    print(f"\nEnvironment classification:")
    for ec, n in sorted(env_summary.items(), key=lambda x: -x[1]):
        print(f"  {ec}: {n}")

    print(f"\nGroup membership:")
    for gn, n in sorted(group_summary.items(), key=lambda x: -x[1]):
        if n > 1:
            print(f"  {gn}: {n}")

    n_dense = sum(1 for v in env_catalog.values() if v['env_dense'] == 'dense')
    n_field = sum(1 for v in env_catalog.values() if v['env_dense'] == 'field')
    print(f"\nBinary: dense={n_dense}, field={n_field}")

    # Local density statistics
    densities = [v['local_density'] for v in env_catalog.values() if v['local_density'] > 0]
    if densities:
        print(f"\nLocal density (N5 galaxies/Mpc³):")
        print(f"  Median: {np.median(densities):.3f}")
        print(f"  Mean: {np.mean(densities):.3f}")
        print(f"  Dense galaxies median: {np.median([v['local_density'] for v in env_catalog.values() if v['env_dense'] == 'dense' and v['local_density'] > 0]):.3f}")
        field_dens = [v['local_density'] for v in env_catalog.values() if v['env_dense'] == 'field' and v['local_density'] > 0]
        if field_dens:
            print(f"  Field galaxies median: {np.median(field_dens):.3f}")

    # Save
    output_path = os.path.join(DATA_DIR, 'wallaby_environment_catalog.json')
    with open(output_path, 'w') as f:
        json.dump(env_catalog, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    csv_path = os.path.join(DATA_DIR, 'wallaby_environment_catalog.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'ra', 'dec', 'Vsys', 'env_class', 'group_name',
                          'env_dense', 'logMh', 'local_density', 'team_release'])
        for name, v in sorted(env_catalog.items()):
            writer.writerow([
                name, f"{v['ra']:.6f}", f"{v['dec']:.6f}", f"{v['Vsys']:.1f}",
                v['env_class'], v['group_name'], v['env_dense'],
                f"{v['logMh']:.1f}", f"{v['local_density']:.4f}", v['team_release']
            ])
    print(f"Saved: {csv_path}")

    return env_catalog


if __name__ == '__main__':
    build_wallaby_environment_catalog()
