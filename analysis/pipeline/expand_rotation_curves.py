#!/usr/bin/env python3
"""
expand_rotation_curves.py — Data expansion for SPARC RAR BEC pipeline

Downloads and integrates additional rotation curve datasets beyond SPARC:
  1. Sofue RC99 (50 galaxies, R & V, no errors)
  2. Sofue 2016 RCAtlas (~192 galaxies, R & V, smoothed, no errors)
  3. Oh+2015 LITTLE THINGS (26 dwarf irregulars, scaled R & V with errors)

Cross-matches with SPARC to identify overlap and new unique galaxies.
For non-SPARC galaxies, provides rotation curves but flags lack of mass decomposition.

PROBES (Stone & Courteau 2022, 3163 galaxies):
  - Data is on IOPscience supplementary material but requires browser download
  - Format: Name, R(arcsec), V(km/s), V_e(km/s) + main_table.csv with distances
  - TODO: Download via browser and integrate

BIG-SPARC (Haubner+2024, ~3882 galaxies):
  - NOT YET RELEASED — conference proceedings only (arXiv:2411.13329)
  - Will contain homogeneous 3DBarolo RCs from 7914 HI cubes
  - Expected to include WISE W1 photometry for mass models

Author: Claude Code (BEC dark matter pipeline)
"""

import os
import sys
import json
import re
import numpy as np
from collections import defaultdict

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# ============================================================================
# NGC NAME MAPPING
# ============================================================================

# Map Sofue file numbers to standard galaxy names
# Sofue uses NGC numbers (e.g., 0224 = NGC 224 = M31)
# For UGC/IC galaxies, the first digit encodes the catalog:
#   1xxxxx = NGC, 2xxxxx = UGC, 3xxxxx = IC (in Sofue 2016)
#   0xxxx = NGC (in Sofue RC99)

def sofue_rc99_ngc_to_name(filename):
    """Convert Sofue RC99 filename (e.g., '0224.dat') to galaxy name."""
    base = os.path.splitext(os.path.basename(filename))[0]
    num = int(base)
    if num == 0:
        return 'MilkyWay'
    # Read the header for the actual name
    return None  # Will be read from file header


def sofue2016_to_name(filename):
    """Convert Sofue 2016 filename (e.g., '100224.dat') to galaxy name."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Remove any trailing letters (e.g., '612276c' -> '612276')
    base_clean = re.sub(r'[a-zA-Z]+$', '', base)
    try:
        num = int(base_clean)
    except ValueError:
        return base  # Return as-is if unparseable
    if num >= 200000:
        # UGC
        ugc_num = num - 200000
        return f'UGC{ugc_num:05d}'
    elif num >= 100000:
        # NGC
        ngc_num = num - 100000
        return f'NGC{ngc_num:04d}'
    else:
        return f'NGC{num:04d}'


# ============================================================================
# SPARC GALAXY NAME NORMALIZATION
# ============================================================================

def normalize_galaxy_name(name):
    """
    Normalize galaxy name for cross-matching.
    Handles: NGC0224 -> NGC224, UGC 02885 -> UGC02885, etc.
    """
    name = name.strip().upper()
    # Remove spaces
    name = name.replace(' ', '')
    # Normalize NGC/UGC/IC with leading zeros
    for prefix in ['NGC', 'UGC', 'IC', 'DDO', 'ESO', 'PGC', 'UGCA']:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            # Remove leading zeros but keep at least one digit
            rest = rest.lstrip('0') or '0'
            # Handle suffixes like NGC0224 -> NGC224
            # But also ESO079-G014 -> ESO79-G014
            if '-' in rest:
                parts = rest.split('-', 1)
                parts[0] = parts[0].lstrip('0') or '0'
                rest = '-'.join(parts)
            return prefix + rest
    return name


# ============================================================================
# LOAD SPARC GALAXY LIST
# ============================================================================

def load_sparc_galaxy_list():
    """Load list of SPARC galaxy names for cross-matching."""
    mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')
    sparc_names = set()
    sparc_props = {}

    if not os.path.exists(mrt_path):
        print("  WARNING: SPARC_Lelli2016c.mrt not found")
        return sparc_names, sparc_props

    with open(mrt_path, 'r') as f:
        lines = f.readlines()

    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---') and i > 50:
            data_start = i + 1
            break

    for line in lines[data_start:]:
        if not line.strip() or line.startswith('#'):
            continue
        name = line[0:11].strip()
        if not name:
            continue
        norm_name = normalize_galaxy_name(name)
        sparc_names.add(norm_name)

        try:
            parts = line[11:].split()
            if len(parts) >= 17:
                sparc_props[norm_name] = {
                    'original_name': name,
                    'T': int(parts[0]),
                    'D': float(parts[1]),
                    'eD': float(parts[2]),
                    'Inc': float(parts[4]),
                    'Vflat': float(parts[14]),
                    'eVflat': float(parts[15]),
                    'Q': int(parts[16]),
                }
        except (ValueError, IndexError):
            pass

    print(f"  Loaded {len(sparc_names)} SPARC galaxy names")
    return sparc_names, sparc_props


# ============================================================================
# LOAD SOFUE RC99
# ============================================================================

def load_sofue_rc99():
    """
    Load Sofue RC99 rotation curves (50 galaxies).
    Format: header with galaxy info, then R(kpc) V(km/s) columns.
    """
    rc99_dir = os.path.join(DATA_DIR, 'sofue_rc99')
    if not os.path.exists(rc99_dir):
        print("  Sofue RC99 data not found")
        return {}

    galaxies = {}
    dat_files = sorted([f for f in os.listdir(rc99_dir) if f.endswith('.dat')])

    for fname in dat_files:
        filepath = os.path.join(rc99_dir, fname)

        # Parse header and data
        name = None
        distance = None
        inclination = None
        gtype = None
        R_data = []
        V_data = []

        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#Name'):
                        name = line.split(':')[1].strip()
                    elif line.startswith('#Distance'):
                        try:
                            distance = float(line.split(':')[1].strip())
                        except ValueError:
                            pass
                    elif line.startswith('#Inclination'):
                        try:
                            inclination = float(line.split(':')[1].strip())
                        except ValueError:
                            pass
                    elif line.startswith('#Type'):
                        gtype = line.split(':')[1].strip()
                    elif line.startswith('#'):
                        continue
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                r = float(parts[0])
                                v = float(parts[1])
                                R_data.append(r)
                                V_data.append(v)
                            except ValueError:
                                continue
        except Exception as e:
            print(f"    Error reading {fname}: {e}")
            continue

        if name and len(R_data) > 3:
            norm_name = normalize_galaxy_name(name)
            galaxies[norm_name] = {
                'original_name': name,
                'source': 'Sofue_RC99',
                'distance_Mpc': distance,
                'inclination': inclination,
                'type': gtype,
                'R_kpc': np.array(R_data),
                'V_kms': np.array(V_data),
                'V_err': None,  # No errors available
                'has_mass_model': False,
                'n_points': len(R_data),
            }

    print(f"  Loaded {len(galaxies)} Sofue RC99 galaxies")
    return galaxies


# ============================================================================
# LOAD SOFUE 2016 RC ATLAS
# ============================================================================

def load_sofue2016():
    """
    Load Sofue 2016 RC Atlas (~192 galaxies).
    Format: NGC number on first line, then R(kpc) V(km/s) columns.
    These are Gaussian-smoothed rotation curves — no errors available.
    """
    s16_dir = os.path.join(DATA_DIR, 'sofue2016')
    if not os.path.exists(s16_dir):
        print("  Sofue 2016 data not found")
        return {}

    galaxies = {}
    dat_files = sorted([f for f in os.listdir(s16_dir) if f.endswith('.dat')])

    for fname in dat_files:
        filepath = os.path.join(s16_dir, fname)

        name_from_file = sofue2016_to_name(fname)
        R_data = []
        V_data = []

        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                first_line = True
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if first_line:
                        # First line is the NGC number
                        first_line = False
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            r = float(parts[0])
                            v = float(parts[1])
                            R_data.append(r)
                            V_data.append(v)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"    Error reading {fname}: {e}")
            continue

        if len(R_data) > 3:
            norm_name = normalize_galaxy_name(name_from_file)
            galaxies[norm_name] = {
                'original_name': name_from_file,
                'source': 'Sofue_2016',
                'distance_Mpc': None,  # Not in data files; from Sofue 2016 paper
                'inclination': None,
                'type': None,
                'R_kpc': np.array(R_data),
                'V_kms': np.array(V_data),
                'V_err': None,  # Smoothed curves, no errors
                'has_mass_model': False,
                'n_points': len(R_data),
                'note': 'Gaussian-smoothed rotation curve from literature compilation',
            }

    print(f"  Loaded {len(galaxies)} Sofue 2016 galaxies")
    return galaxies


# ============================================================================
# LOAD OH+2015 LITTLE THINGS
# ============================================================================

def load_oh2015():
    """
    Load Oh+2015 LITTLE THINGS rotation curves (26 dwarf irregulars).
    Data is in SCALED form: R/R0.3 and V/V0.3 with V0.3 and R0.3 given.
    We need to UN-SCALE to get physical units.

    Files: dbf2A.txt (first 9 galaxies), dbf2C.txt (remaining galaxies)
    Both contain 'Data' rows (observations) and 'Model' rows (fits).
    """
    oh_dir = os.path.join(DATA_DIR, 'oh2015_littlethings', 'aj513259f2')
    if not os.path.exists(oh_dir):
        print("  Oh+2015 LITTLE THINGS data not found")
        return {}

    galaxies = {}

    # Parse data files (dbf2A.txt has first 9, dbf2C.txt has rest)
    for datafile in ['dbf2A.txt', 'dbf2C.txt']:
        filepath = os.path.join(oh_dir, datafile)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip header (find data start — after '---' separator)
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('---'):
                data_start = i + 1

        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                gal_name = parts[0]
                data_type = parts[1]

                # Only use 'Data' rows, not 'Model' rows
                if data_type != 'Data':
                    continue

                R0_3 = float(parts[2])  # R_0.3 in kpc
                V0_3 = float(parts[3])  # V_0.3 in km/s
                R_scaled = float(parts[4])  # R/R_0.3
                V_scaled = float(parts[5])  # V/V_0.3
                V_err_scaled = float(parts[6])  # error in V/V_0.3

                # Convert to physical units
                R_kpc = R_scaled * R0_3
                V_kms = V_scaled * V0_3
                V_err = V_err_scaled * V0_3

                # Normalize name (e.g., CVnldwA -> CVnIdwA)
                norm_name = normalize_galaxy_name(gal_name)

                if norm_name not in galaxies:
                    galaxies[norm_name] = {
                        'original_name': gal_name,
                        'source': 'Oh2015_LITTLE_THINGS',
                        'R0_3_kpc': R0_3,
                        'V0_3_kms': V0_3,
                        'distance_Mpc': None,
                        'inclination': None,
                        'type': 'Im/BCD',  # All are dwarf irregulars
                        'R_kpc': [],
                        'V_kms': [],
                        'V_err': [],
                        'has_mass_model': False,
                    }

                galaxies[norm_name]['R_kpc'].append(R_kpc)
                galaxies[norm_name]['V_kms'].append(V_kms)
                galaxies[norm_name]['V_err'].append(V_err)

            except (ValueError, IndexError):
                continue

    # Convert to numpy and add point counts
    for name in galaxies:
        for key in ['R_kpc', 'V_kms', 'V_err']:
            galaxies[name][key] = np.array(galaxies[name][key])
        galaxies[name]['n_points'] = len(galaxies[name]['R_kpc'])

    # Filter out galaxies with too few points
    galaxies = {k: v for k, v in galaxies.items() if v['n_points'] > 3}

    print(f"  Loaded {len(galaxies)} Oh+2015 LITTLE THINGS galaxies")
    return galaxies


# ============================================================================
# LOAD PROBES (if available)
# ============================================================================

def load_probes():
    """
    Load PROBES rotation curves if downloaded.
    Expected format: CSV with columns Name, R(arcsec), V(km/s), V_e(km/s)
    Plus main_table.csv with distances, RA, Dec, etc.

    PROBES data must be downloaded from IOPscience supplementary material:
    https://iopscience.iop.org/article/10.3847/1538-4365/ac83ad

    Place files in: data/probes/
    """
    probes_dir = os.path.join(DATA_DIR, 'probes')

    # Check for expected PROBES files
    main_table = os.path.join(probes_dir, 'main_table.csv')
    rc_files = [f for f in os.listdir(probes_dir) if f.endswith('.csv') and 'rotation' in f.lower()] if os.path.exists(probes_dir) else []

    if not os.path.exists(main_table) and not rc_files:
        print("  PROBES data not found. To add PROBES:")
        print("    1. Visit https://iopscience.iop.org/article/10.3847/1538-4365/ac83ad")
        print("    2. Download supplementary data files")
        print("    3. Place main_table.csv, rotation curve CSV, etc. in data/probes/")
        return {}

    # TODO: implement PROBES loader when data is available
    print("  PROBES loader: placeholder — data format needs verification after download")
    return {}


# ============================================================================
# CROSS-MATCHING AND INTEGRATION
# ============================================================================

def cross_match_datasets(sparc_names, *datasets):
    """
    Cross-match additional datasets against SPARC.
    Returns summary of overlap and unique galaxies.
    """
    results = {
        'sparc_count': len(sparc_names),
        'datasets': {},
        'all_unique_names': set(sparc_names),
        'new_galaxies': {},
    }

    for dataset_name, galaxies in datasets:
        overlap = set()
        new = set()

        for norm_name in galaxies:
            if norm_name in sparc_names:
                overlap.add(norm_name)
            else:
                new.add(norm_name)
                results['new_galaxies'][norm_name] = galaxies[norm_name]

        results['datasets'][dataset_name] = {
            'total': len(galaxies),
            'overlap_with_sparc': len(overlap),
            'new_unique': len(new),
            'overlap_names': sorted(overlap),
            'new_names': sorted(new),
        }

        results['all_unique_names'].update(new)

    # Check inter-dataset overlap (between Sofue RC99 and Sofue 2016)
    all_new = set()
    for ds_info in results['datasets'].values():
        all_new.update(ds_info['new_names'])

    results['total_unique_galaxies'] = len(results['all_unique_names'])
    results['new_beyond_sparc'] = len(results['all_unique_names']) - len(sparc_names)

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ROTATION CURVE DATA EXPANSION")
    print("=" * 70)

    # 1. Load SPARC galaxy list
    print("\n[1] Loading SPARC reference galaxy list...")
    sparc_names, sparc_props = load_sparc_galaxy_list()

    # 2. Load additional datasets
    print("\n[2] Loading additional rotation curve datasets...")

    rc99 = load_sofue_rc99()
    s2016 = load_sofue2016()
    oh2015 = load_oh2015()
    probes = load_probes()

    # 3. Cross-match
    print("\n[3] Cross-matching datasets...")
    datasets_to_match = [
        ('Sofue_RC99', rc99),
        ('Sofue_2016', s2016),
        ('Oh2015_LITTLE_THINGS', oh2015),
    ]
    if probes:
        datasets_to_match.append(('PROBES', probes))

    xmatch = cross_match_datasets(sparc_names, *datasets_to_match)

    # 4. Report
    print("\n" + "=" * 70)
    print("CROSS-MATCH RESULTS")
    print("=" * 70)
    print(f"\n  SPARC (reference): {xmatch['sparc_count']} galaxies")

    for ds_name, ds_info in xmatch['datasets'].items():
        print(f"\n  {ds_name}:")
        print(f"    Total loaded: {ds_info['total']}")
        print(f"    Overlap with SPARC: {ds_info['overlap_with_sparc']}")
        print(f"    New unique: {ds_info['new_unique']}")
        if ds_info['overlap_with_sparc'] > 0:
            overlap_list = ', '.join(ds_info['overlap_names'][:10])
            if ds_info['overlap_with_sparc'] > 10:
                overlap_list += f' ... (+{ds_info["overlap_with_sparc"]-10} more)'
            print(f"    Overlapping: {overlap_list}")

    print(f"\n  TOTAL UNIQUE GALAXIES: {xmatch['total_unique_galaxies']}")
    print(f"  New beyond SPARC: {xmatch['new_beyond_sparc']}")

    # 5. Analyze what the new galaxies CAN be used for
    print("\n" + "=" * 70)
    print("DATA UTILITY ANALYSIS")
    print("=" * 70)

    n_with_errors = 0
    n_without_errors = 0
    n_with_distance = 0
    n_without_distance = 0

    for name, gal in xmatch['new_galaxies'].items():
        if gal.get('V_err') is not None and len(gal['V_err']) > 0:
            n_with_errors += 1
        else:
            n_without_errors += 1
        if gal.get('distance_Mpc') is not None:
            n_with_distance += 1
        else:
            n_without_distance += 1

    total_new = n_with_errors + n_without_errors
    print(f"\n  New galaxies: {total_new}")
    print(f"    With velocity errors: {n_with_errors}")
    print(f"    Without velocity errors: {n_without_errors}")
    print(f"    With distance: {n_with_distance}")
    print(f"    Without distance: {n_without_distance}")
    print(f"    With mass decomposition: 0 (none — only SPARC has Vgas/Vdisk/Vbul)")

    print(f"\n  Usable for RAR analysis: 0 new (requires mass decomposition)")
    print(f"  Usable for Tully-Fisher: {n_with_distance} (need distance + Vflat)")
    print(f"  Usable for RC shape analysis: {total_new} (R, V only)")

    # 6. Check overlap between Sofue datasets
    print("\n" + "=" * 70)
    print("INTER-DATASET OVERLAP")
    print("=" * 70)

    rc99_names = set(rc99.keys())
    s2016_names = set(s2016.keys())
    oh_names = set(oh2015.keys())

    rc99_s2016_overlap = rc99_names & s2016_names
    print(f"\n  Sofue RC99 ∩ Sofue 2016: {len(rc99_s2016_overlap)} galaxies")
    if rc99_s2016_overlap:
        print(f"    {', '.join(sorted(rc99_s2016_overlap)[:20])}")

    rc99_oh_overlap = rc99_names & oh_names
    s2016_oh_overlap = s2016_names & oh_names
    print(f"  Sofue RC99 ∩ Oh2015: {len(rc99_oh_overlap)} galaxies")
    print(f"  Sofue 2016 ∩ Oh2015: {len(s2016_oh_overlap)} galaxies")

    # After removing all overlaps, truly unique new galaxies
    all_external = rc99_names | s2016_names | oh_names
    unique_external = all_external - sparc_names
    print(f"\n  Total unique external (non-SPARC): {len(unique_external)}")

    # 7. Save results
    results_path = os.path.join(RESULTS_DIR, 'data_expansion_crossmatch.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    save_data = {
        'sparc_count': xmatch['sparc_count'],
        'total_unique': xmatch['total_unique_galaxies'],
        'new_beyond_sparc': xmatch['new_beyond_sparc'],
        'datasets': {},
    }
    for ds_name, ds_info in xmatch['datasets'].items():
        save_data['datasets'][ds_name] = {
            'total': ds_info['total'],
            'overlap_with_sparc': ds_info['overlap_with_sparc'],
            'new_unique': ds_info['new_unique'],
            'overlap_names': ds_info['overlap_names'],
            'new_names': ds_info['new_names'],
        }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # 8. Key limitation note
    print("\n" + "=" * 70)
    print("CRITICAL LIMITATION")
    print("=" * 70)
    print("""
  For the RAR (Radial Acceleration Relation) analysis, we need:
    - R (kpc), Vobs (km/s), eVobs (km/s)
    - Vgas, Vdisk, Vbul (baryonic mass model decomposition)
    - Distance, inclination, luminosity (galaxy properties)

  Only SPARC (175 galaxies) provides all of these.

  The additional datasets provide R and V only — no mass decomposition.
  To use them for RAR, we would need:
    1. WISE W1 photometry → stellar mass → Vdisk
    2. HI surveys → gas mass → Vgas
    3. CF4 distances → physical radii
    4. Inclination corrections

  This is essentially what BIG-SPARC is doing (~3882 galaxies).

  FOR THE RAR TIGHTNESS TEST: Continue using SPARC (175 galaxies).
  The test already shows R²=19.4% property dependence — more galaxies
  would tighten the statistics but wouldn't change the qualitative finding.

  FOR PROBES (3163 galaxies): Download from IOPscience supplementary:
    https://iopscience.iop.org/article/10.3847/1538-4365/ac83ad
  PROBES has distances and inclinations, but still no mass decomposition.
  However, PROBES + WISE W1 could provide approximate mass models.
""")

    return xmatch


if __name__ == '__main__':
    results = main()
