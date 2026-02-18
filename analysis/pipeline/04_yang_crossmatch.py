#!/usr/bin/env python3
"""
STEP 4: Cross-match SPARC galaxies with Yang+2007 DR7 Group Catalog
=====================================================================
- Loads Yang DR7 galaxy catalog (639,359 galaxies with RA/Dec/z)
- Loads Yang DR7 group catalog (472,419 groups with halo masses)
- Loads galaxy-to-group mapping
- Matches SPARC galaxies by RA/Dec
- Extracts: halo mass, group richness, central/satellite flag
- Falls back to known environment data for very nearby galaxies
- Outputs enriched environment catalog for pipeline re-run

Russell Licht — Primordial Fluid DM Project
Feb 2026
"""

import numpy as np
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
YANG_DIR = os.path.join(DATA_DIR, 'yang_catalogs')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')


# ============================================================
# LOAD YANG DR7 CATALOG FILES
# ============================================================
def load_yang_dr7_galaxy():
    """
    Load SDSS7 galaxy catalog.
    Format (whitespace-delimited):
      col 1: galaxy ID
      col 2: NYU-VAGC ID
      col 3: RA (degrees)
      col 4: Dec (degrees)
      col 5: z (redshift)
      col 6: r-band apparent magnitude
      col 7: magnitude limit
      col 8: completeness
      col 9-12: absolute magnitudes and colors
      col 13: redshift source type
    """
    filepath = os.path.join(YANG_DIR, 'SDSS7')
    if not os.path.exists(filepath):
        print(f"  ERROR: {filepath} not found")
        return None

    print(f"  Loading {filepath}...")

    # Use numpy for fast loading
    data = np.loadtxt(filepath, dtype={
        'names': ('gal_id', 'vagc_id', 'ra', 'dec', 'z',
                  'r_app', 'r_lim', 'completeness',
                  'Mr_petro', 'gr_petro', 'Mr_model', 'gr_model',
                  'z_type'),
        'formats': ('i4', 'i4', 'f8', 'f8', 'f8',
                    'f4', 'f4', 'f4',
                    'f4', 'f4', 'f4', 'f4',
                    'i2')
    })

    print(f"  Loaded {len(data)} galaxies")
    print(f"  RA range: {data['ra'].min():.2f} - {data['ra'].max():.2f}")
    print(f"  Dec range: {data['dec'].min():.2f} - {data['dec'].max():.2f}")
    print(f"  z range: {data['z'].min():.4f} - {data['z'].max():.4f}")

    return data


def load_yang_dr7_groups():
    """
    Load modelC_group: group properties.
    Format (whitespace-delimited):
      col 1: group ID
      col 2: RA (luminosity-weighted center)
      col 3: Dec
      col 4: z (luminosity-weighted)
      col 5: group L_{-19.5} (log L_solar/h^2)
      col 6: group M_stellar (log M_solar/h^2)
      col 7: halo mass1 (log M_halo/(M_solar/h)) from L ranking
      col 8: halo mass2 (log M_halo/(M_solar/h)) from M* ranking
      col 9-10: mean separation (Mpc/h)
      col 11: f_edge
      col 12-13: ID flags

    First 3 rows are completeness info (group_id = 0), skip them.
    """
    filepath = os.path.join(YANG_DIR, 'modelC_group')
    if not os.path.exists(filepath):
        print(f"  ERROR: {filepath} not found")
        return None

    print(f"  Loading {filepath}...")

    # Read all lines, skip first 3 (completeness info with group_id=0)
    groups = {}
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            parts = line.split()
            if len(parts) < 8:
                continue

            grp_id = int(parts[0])
            if grp_id == 0:
                continue  # Skip completeness header rows

            groups[grp_id] = {
                'grp_id': grp_id,
                'ra': float(parts[1]),
                'dec': float(parts[2]),
                'z': float(parts[3]),
                'logL': float(parts[4]),
                'logMstar': float(parts[5]),
                'logMh_L': float(parts[6]),    # halo mass from luminosity
                'logMh_Mstar': float(parts[7]), # halo mass from stellar mass
                'f_edge': float(parts[10]) if len(parts) > 10 else 0,
            }

    print(f"  Loaded {len(groups)} groups")

    # Count richness (will be filled from galaxy-group mapping)
    return groups


def load_yang_dr7_mapping():
    """
    Load imodelC_1: galaxy-to-group mapping.
    Format:
      col 1: galaxy ID (matches SDSS7 col 1)
      col 2: NYU-VAGC ID
      col 3: group ID (0 = not in a group)
      col 4: brightest galaxy flag (1=brightest, 2=other)
      col 5: most massive galaxy flag (1=most massive, 2=other)
    """
    filepath = os.path.join(YANG_DIR, 'imodelC_1')
    if not os.path.exists(filepath):
        print(f"  ERROR: {filepath} not found")
        return None

    print(f"  Loading {filepath}...")

    data = np.loadtxt(filepath, dtype={
        'names': ('gal_id', 'vagc_id', 'grp_id', 'brightest', 'most_massive'),
        'formats': ('i4', 'i4', 'i4', 'i2', 'i2')
    })

    print(f"  Loaded {len(data)} galaxy-group mappings")

    return data


# ============================================================
# COORDINATE MATCHING
# ============================================================
def build_spatial_index(ra, dec, n_bins=360):
    """Build a simple spatial index for fast RA/Dec matching."""
    # Bin by RA for quick filtering
    ra_bins = {}
    ra_step = 360.0 / n_bins

    for i in range(len(ra)):
        ra_bin = int(ra[i] / ra_step) % n_bins
        if ra_bin not in ra_bins:
            ra_bins[ra_bin] = []
        ra_bins[ra_bin].append(i)

    return ra_bins, ra_step


def find_closest_match(target_ra, target_dec, cat_ra, cat_dec,
                       spatial_index, ra_step, max_sep_arcsec=30.0):
    """Find the closest catalog match within max_sep_arcsec."""
    max_sep_deg = max_sep_arcsec / 3600.0

    # Determine which RA bins to search
    ra_bin = int(target_ra / ra_step) % int(360.0 / ra_step)
    n_bins = int(360.0 / ra_step)

    # Search neighboring bins
    search_bins = [(ra_bin - 1) % n_bins, ra_bin, (ra_bin + 1) % n_bins]
    candidates = []
    for b in search_bins:
        if b in spatial_index:
            candidates.extend(spatial_index[b])

    if not candidates:
        return None, None

    candidates = np.array(candidates)

    # Pre-filter by Dec
    dec_diff = np.abs(cat_dec[candidates] - target_dec)
    dec_mask = dec_diff < max_sep_deg * 1.5
    if not np.any(dec_mask):
        return None, None

    filtered = candidates[dec_mask]

    # Compute angular separations
    ra1 = np.radians(target_ra)
    dec1 = np.radians(target_dec)
    ra2 = np.radians(cat_ra[filtered])
    dec2 = np.radians(cat_dec[filtered])

    cos_sep = (np.sin(dec1) * np.sin(dec2) +
               np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    cos_sep = np.clip(cos_sep, -1, 1)
    sep_deg = np.degrees(np.arccos(cos_sep))
    sep_arcsec = sep_deg * 3600

    min_idx = np.argmin(sep_arcsec)
    if sep_arcsec[min_idx] <= max_sep_arcsec:
        return filtered[min_idx], sep_arcsec[min_idx]

    return None, None


# ============================================================
# KNOWN ENVIRONMENTS FOR VERY NEARBY GALAXIES
# ============================================================
KNOWN_ENVIRONMENTS = {
    # Ursa Major cluster — logMh ~ 13.2 (Tully+1996, Verheijen+2001)
    'NGC3726': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC3769': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC3877': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC3893': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC3917': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC3949': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC3953': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC3972': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC3992': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 1},
    'NGC4010': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4013': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4051': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4085': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4088': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4100': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4138': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4157': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4183': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC4217': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06399': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06446': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06667': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06786': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06787': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06818': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06917': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06923': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06930': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06973': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC06983': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'UGC07089': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},

    # M81 group — logMh ~ 12.0
    'NGC2403': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},
    'NGC2976': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},
    'IC2574': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},
    'DDO154': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},
    'DDO168': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},
    'UGC04483': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},

    # Sculptor group — logMh ~ 11.5
    'NGC0300': {'group': 'Sculptor', 'logMh': 11.5, 'richness': 15, 'central': 0},
    'NGC0055': {'group': 'Sculptor', 'logMh': 11.5, 'richness': 15, 'central': 0},
    'NGC0247': {'group': 'Sculptor', 'logMh': 11.5, 'richness': 15, 'central': 0},
    'NGC7793': {'group': 'Sculptor', 'logMh': 11.5, 'richness': 15, 'central': 0},

    # Centaurus A group — logMh ~ 12.5
    'NGC2915': {'group': 'CenA', 'logMh': 12.5, 'richness': 30, 'central': 0},
    'UGCA442': {'group': 'CenA', 'logMh': 12.5, 'richness': 30, 'central': 0},
    'ESO444-G084': {'group': 'CenA', 'logMh': 12.5, 'richness': 30, 'central': 0},

    # CVn I cloud — logMh ~ 11.0
    'UGC07577': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'UGC07232': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'NGC3741': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'NGC4068': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'UGC07866': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'UGC07524': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'UGC08490': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'UGC07559': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},

    # Other known associations
    'NGC3109': {'group': 'Antlia', 'logMh': 10.5, 'richness': 5, 'central': 1},
    'NGC5055': {'group': 'M101', 'logMh': 12.0, 'richness': 10, 'central': 0},
    'NGC6946': {'group': 'NGC6946', 'logMh': 11.5, 'richness': 5, 'central': 1},

    # Very nearby dwarfs / Local Volume
    'CamB': {'group': 'field', 'logMh': 9.5, 'richness': 1, 'central': 1},
    'NGC6789': {'group': 'field', 'logMh': 9.5, 'richness': 1, 'central': 1},
    'UGC04305': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},
    'DDO064': {'group': 'field', 'logMh': 10.0, 'richness': 1, 'central': 1},
    'DDO161': {'group': 'field', 'logMh': 10.0, 'richness': 1, 'central': 1},
    'DDO170': {'group': 'field', 'logMh': 10.5, 'richness': 1, 'central': 1},
    'NGC2366': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},
    'NGC1705': {'group': 'field', 'logMh': 10.0, 'richness': 1, 'central': 1},
    'UGC01281': {'group': 'field', 'logMh': 10.0, 'richness': 1, 'central': 1},
    'UGCA444': {'group': 'CenA', 'logMh': 12.5, 'richness': 30, 'central': 0},
    'UGCA281': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'NGC4214': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'NGC4389': {'group': 'UMa', 'logMh': 13.2, 'richness': 79, 'central': 0},
    'NGC5585': {'group': 'M101', 'logMh': 12.0, 'richness': 10, 'central': 0},
    'NGC6503': {'group': 'field', 'logMh': 11.0, 'richness': 1, 'central': 1},
    'UGC05764': {'group': 'M81', 'logMh': 12.0, 'richness': 30, 'central': 0},
    'UGC05918': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'UGC08837': {'group': 'field', 'logMh': 10.0, 'richness': 1, 'central': 1},
    'UGC12632': {'group': 'field', 'logMh': 10.0, 'richness': 1, 'central': 1},
    'KK98-251': {'group': 'field', 'logMh': 9.5, 'richness': 1, 'central': 1},
    'D512-2': {'group': 'CVnI', 'logMh': 11.0, 'richness': 12, 'central': 0},
    'D564-8': {'group': 'field', 'logMh': 10.0, 'richness': 1, 'central': 1},
    'D631-7': {'group': 'field', 'logMh': 10.0, 'richness': 1, 'central': 1},
}


def classify_by_halo_mass(log_Mh):
    """Classify environment by halo mass."""
    if log_Mh is None or np.isnan(log_Mh):
        return 'unknown'
    if log_Mh >= 14.0:
        return 'cluster'
    elif log_Mh >= 13.0:
        return 'rich_group'
    elif log_Mh >= 12.0:
        return 'group'
    elif log_Mh >= 11.0:
        return 'poor_group'
    else:
        return 'field'


def classify_dense_field(log_Mh, threshold=12.5):
    """Binary: dense (logMh >= threshold) vs field."""
    if log_Mh is None or np.isnan(log_Mh):
        return 'unknown'
    return 'dense' if log_Mh >= threshold else 'field'


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("STEP 4: Cross-match SPARC with Yang DR7 Group Catalog")
    print("=" * 80)

    # Load SPARC coordinates
    coord_file = os.path.join(DATA_DIR, 'sparc_coordinates.json')
    with open(coord_file, 'r') as f:
        sparc_coords = json.load(f)
    print(f"\nLoaded coordinates for {len(sparc_coords)} SPARC galaxies")

    # Load CF4 cache for velocities/redshifts
    cf4_file = os.path.join(DATA_DIR, 'cf4_distance_cache.json')
    with open(cf4_file, 'r') as f:
        cf4_data = json.load(f)

    # Add velocities to sparc_coords
    for name in sparc_coords:
        if name in cf4_data and 'V_helio' in cf4_data[name]:
            sparc_coords[name]['z'] = cf4_data[name]['V_helio'] / 299792.458

    # --------------------------------------------------------
    # Load Yang DR7 catalogs
    # --------------------------------------------------------
    print("\n--- Loading Yang DR7 Catalogs ---")
    yang_gal = load_yang_dr7_galaxy()
    yang_grp = load_yang_dr7_groups()
    yang_map = load_yang_dr7_mapping()

    environment_results = {}

    if yang_gal is not None and yang_grp is not None and yang_map is not None:
        # Build galaxy_id -> group mapping
        gal_to_grp = {}
        for i in range(len(yang_map)):
            gal_id = yang_map['gal_id'][i]
            grp_id = yang_map['grp_id'][i]
            brightest = yang_map['brightest'][i]
            most_massive = yang_map['most_massive'][i]
            gal_to_grp[gal_id] = {
                'grp_id': grp_id,
                'central': 1 if brightest == 1 else 0,
            }

        # Compute group richness
        grp_richness = {}
        for g in gal_to_grp.values():
            gid = g['grp_id']
            if gid > 0:
                grp_richness[gid] = grp_richness.get(gid, 0) + 1

        # Build spatial index for fast matching
        print("\n--- Building spatial index ---")
        yang_ra = yang_gal['ra']
        yang_dec = yang_gal['dec']
        spatial_idx, ra_step = build_spatial_index(yang_ra, yang_dec)
        print(f"  Index built with {len(spatial_idx)} RA bins")

        # Cross-match SPARC galaxies
        print("\n--- Cross-matching SPARC with Yang DR7 ---")

        matched = 0
        matched_with_group = 0
        yang_z_min = 0.01  # Yang DR7 lower redshift limit

        for name, coords in sparc_coords.items():
            ra = coords['ra']
            dec = coords['dec']
            z = coords.get('z', 0)

            # Skip very nearby galaxies (z < 0.01) — outside Yang coverage
            if z < yang_z_min:
                continue

            idx, sep = find_closest_match(ra, dec, yang_ra, yang_dec,
                                          spatial_idx, ra_step, max_sep_arcsec=30.0)

            if idx is not None:
                matched += 1
                gal_id = yang_gal['gal_id'][idx]
                yang_z = yang_gal['z'][idx]

                # Check redshift consistency (within 0.01)
                if abs(yang_z - z) > 0.01 and z > 0:
                    continue  # Wrong match — redshift mismatch

                # Get group info
                if gal_id in gal_to_grp:
                    grp_info = gal_to_grp[gal_id]
                    grp_id = grp_info['grp_id']

                    if grp_id > 0 and grp_id in yang_grp:
                        grp = yang_grp[grp_id]
                        logMh = grp['logMh_L']  # Use luminosity-based halo mass
                        if logMh == 0:
                            logMh = grp['logMh_Mstar']  # Fallback to stellar mass

                        richness = grp_richness.get(grp_id, 1)

                        environment_results[name] = {
                            'source': 'Yang_DR7',
                            'group_name': f'Yang_{grp_id}',
                            'logMh': logMh,
                            'richness': richness,
                            'central': grp_info['central'],
                            'yang_z': float(yang_z),
                            'separation_arcsec': float(sep),
                            'env_class': classify_by_halo_mass(logMh),
                            'env_dense': classify_dense_field(logMh),
                            'grp_ra': grp['ra'],
                            'grp_dec': grp['dec'],
                        }
                        matched_with_group += 1
                    else:
                        # Galaxy in catalog but not in a group (singleton)
                        environment_results[name] = {
                            'source': 'Yang_DR7_singleton',
                            'group_name': 'singleton',
                            'logMh': 11.0,  # typical isolated halo
                            'richness': 1,
                            'central': 1,
                            'yang_z': float(yang_z),
                            'separation_arcsec': float(sep),
                            'env_class': 'field',
                            'env_dense': 'field',
                        }

        print(f"  Yang DR7 matches: {matched} (with group: {matched_with_group})")

    # --------------------------------------------------------
    # Fill in known environments for nearby galaxies
    # --------------------------------------------------------
    print("\n--- Applying known environments for nearby galaxies ---")
    known_count = 0
    for name in sparc_coords:
        if name not in environment_results and name in KNOWN_ENVIRONMENTS:
            env = KNOWN_ENVIRONMENTS[name]
            environment_results[name] = {
                'source': 'literature',
                'group_name': env['group'],
                'logMh': env['logMh'],
                'richness': env['richness'],
                'central': env['central'],
                'env_class': classify_by_halo_mass(env['logMh']),
                'env_dense': classify_dense_field(env['logMh']),
            }
            known_count += 1
    print(f"  Added {known_count} from literature")

    # --------------------------------------------------------
    # Default: remaining galaxies as field
    # --------------------------------------------------------
    default_count = 0
    for name in sparc_coords:
        if name not in environment_results:
            # Estimate halo mass from velocity (Vflat) if available
            logMh_est = 11.0  # default
            environment_results[name] = {
                'source': 'default_field',
                'group_name': 'field',
                'logMh': logMh_est,
                'richness': 1,
                'central': 1,
                'env_class': 'field',
                'env_dense': 'field',
            }
            default_count += 1

    print(f"  Defaulted {default_count} as field")

    # --------------------------------------------------------
    # Summary and output
    # --------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("ENVIRONMENT CLASSIFICATION RESULTS")
    print(f"{'=' * 80}")

    # By source
    sources = {}
    for v in environment_results.values():
        s = v['source']
        sources[s] = sources.get(s, 0) + 1
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n} galaxies")

    # By class
    print()
    for ec in ['cluster', 'rich_group', 'group', 'poor_group', 'field', 'unknown']:
        count = sum(1 for v in environment_results.values() if v['env_class'] == ec)
        if count > 0:
            print(f"  {ec}: {count} galaxies")

    # Dense vs field
    dense = sum(1 for v in environment_results.values() if v['env_dense'] == 'dense')
    field = sum(1 for v in environment_results.values() if v['env_dense'] == 'field')
    print(f"\n  Dense (logMh >= 12.5): {dense}")
    print(f"  Field (logMh < 12.5):  {field}")

    # Halo mass distribution
    logMh_vals = [v['logMh'] for v in environment_results.values()
                  if v['logMh'] is not None and v['logMh'] > 0]
    if logMh_vals:
        arr = np.array(logMh_vals)
        print(f"\n  Halo mass distribution:")
        print(f"    Min:    {arr.min():.1f}")
        print(f"    Median: {np.median(arr):.1f}")
        print(f"    Max:    {arr.max():.1f}")

    # Show matched groups
    print(f"\n--- Galaxies in Yang DR7 groups ---")
    yang_matches = [(k, v) for k, v in environment_results.items()
                    if v['source'] == 'Yang_DR7']
    yang_matches.sort(key=lambda x: -x[1]['logMh'])
    for name, env in yang_matches[:20]:
        print(f"  {name:<15} logMh={env['logMh']:.1f}  N={env['richness']:>4}  "
              f"{'central' if env['central'] else 'satellite'}  "
              f"sep={env.get('separation_arcsec', 0):.1f}\"  "
              f"group={env['group_name']}")
    if len(yang_matches) > 20:
        print(f"  ... ({len(yang_matches) - 20} more)")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    # JSON
    output_json = os.path.join(DATA_DIR, 'sparc_environment_catalog.json')
    with open(output_json, 'w') as f:
        json.dump(environment_results, f, indent=2, default=str)
    print(f"\n  Saved JSON: {output_json}")

    # CSV
    output_csv = os.path.join(DATA_DIR, 'sparc_environment_catalog.csv')
    fieldnames = ['sparc_name', 'source', 'group_name', 'logMh', 'richness',
                  'central', 'env_class', 'env_dense', 'separation_arcsec']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in sorted(environment_results.keys()):
            row = {'sparc_name': name}
            for k in fieldnames[1:]:
                row[k] = environment_results[name].get(k, '')
            writer.writerow(row)
    print(f"  Saved CSV: {output_csv}")

    print(f"\n{'=' * 80}")
    print("Step 4 complete!")
    print(f"{'=' * 80}")

    return environment_results


if __name__ == '__main__':
    main()
