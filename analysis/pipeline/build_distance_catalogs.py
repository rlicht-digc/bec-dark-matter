#!/usr/bin/env python3
"""
BUILD DISTANCE CATALOGS — Phase 2
====================================

Creates three distance catalogs for 175 SPARC galaxies, enabling
apples-to-apples comparison of environmental scatter under different
distance assumptions.

Three catalogs:

  SPARC:    All distances from SPARC (original publication values)
  Hybrid:   SPARC for fD=2,3,4,5 (gold-standard & UMa cluster);
            NED TRGB if available for fD=1, else CF4 for fD=1
  Full CF4: All distances from CF4 flow model (including UMa cluster —
            pathological test, since CF4 flow model fails for cluster members)

Key design choice: Hybrid uses SPARC for UMa (fD=4) because CF4's flow
model systematically fails for Virgo Supercluster substructure members.

Inputs:
  - data/sparc/SPARC_Lelli2016c.mrt
  - data/cf4/cf4_distance_cache.json
  - analysis/results/trgb_cepheid_subsample.csv

Outputs:
  - data/distance_catalog_sparc.json
  - data/distance_catalog_hybrid.json
  - data/distance_catalog_cf4.json
  - analysis/results/summary_distance_catalogs.json
"""

import os
import csv
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
CF4_DIR = os.path.join(PROJECT_ROOT, 'data', 'cf4')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 72)
print("PHASE 2: BUILD DISTANCE CATALOGS")
print("=" * 72)

# ================================================================
# STEP 1: Parse SPARC MRT for all 175 galaxies
# ================================================================
print("\n[1] Parsing SPARC MRT...")

mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')
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
            'T': int(parts[0]),
            'D': float(parts[1]),
            'eD': float(parts[2]),
            'fD': int(parts[3]),
            'Inc': float(parts[4]),
            'eInc': float(parts[5]),
            'L36': float(parts[6]),
            'Vflat': float(parts[14]),
            'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

print(f"  Parsed {len(sparc_props)} galaxies from SPARC MRT")

# ================================================================
# STEP 2: Load CF4 distance cache
# ================================================================
print("\n[2] Loading CF4 distance cache...")

cf4_path = os.path.join(CF4_DIR, 'cf4_distance_cache.json')
with open(cf4_path, 'r') as f:
    cf4_cache = json.load(f)

cf4_available = {name for name, entry in cf4_cache.items()
                 if entry.get('status') == 'success' and entry.get('D_cf4') is not None}
print(f"  {len(cf4_cache)} total, {len(cf4_available)} with valid CF4 distances")

# ================================================================
# STEP 3: Load TRGB/Cepheid subsample for NED upgrades
# ================================================================
print("\n[3] Loading TRGB/Cepheid subsample for NED upgrades...")

subsample_path = os.path.join(RESULTS_DIR, 'trgb_cepheid_subsample.csv')
ned_upgrades = {}

with open(subsample_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['galaxy']
        sparc_fD = int(row['sparc_fD'])
        # NED upgrades are fD=1 galaxies that gained TRGB/Cepheid distance
        if sparc_fD == 1 and row['dist_method'] in ('TRGB', 'Cepheid'):
            ned_upgrades[name] = {
                'D_ned': float(row['ned_D_Mpc']),
                'eD_ned': float(row['ned_eD_Mpc']),
                'method': row['dist_method'],
            }

print(f"  Found {len(ned_upgrades)} NED-upgraded Hubble-flow galaxies: "
      f"{', '.join(sorted(ned_upgrades.keys()))}")

# ================================================================
# STEP 4: Build three catalogs
# ================================================================
print("\n[4] Building three distance catalogs...")

catalog_sparc = {}
catalog_hybrid = {}
catalog_cf4 = {}

fd_labels = {1: 'Hubble_flow', 2: 'TRGB', 3: 'Cepheid', 4: 'UMa_cluster', 5: 'SNe'}

for name, prop in sparc_props.items():
    D_sparc = prop['D']
    eD_sparc = prop['eD']
    fD = prop['fD']
    fD_label = fd_labels.get(fD, f'fD={fD}')

    # ---- SPARC catalog: always use SPARC distances ----
    catalog_sparc[name] = {
        'D_Mpc': D_sparc,
        'source': f'SPARC_{fD_label}',
        'fD_original': fD,
    }

    # ---- Hybrid catalog ----
    if fD in (2, 3, 5):
        # Gold-standard: use SPARC distance
        catalog_hybrid[name] = {
            'D_Mpc': D_sparc,
            'source': f'SPARC_{fD_label}',
            'fD_original': fD,
        }
    elif fD == 4:
        # UMa cluster: use SPARC distance (CF4 flow model fails for cluster members)
        catalog_hybrid[name] = {
            'D_Mpc': D_sparc,
            'source': 'SPARC_UMa_cluster',
            'fD_original': fD,
        }
    elif fD == 1:
        # Hubble flow: prefer NED TRGB upgrade, else CF4
        if name in ned_upgrades:
            catalog_hybrid[name] = {
                'D_Mpc': ned_upgrades[name]['D_ned'],
                'source': f'NED_{ned_upgrades[name]["method"]}',
                'fD_original': fD,
            }
        elif name in cf4_available:
            catalog_hybrid[name] = {
                'D_Mpc': cf4_cache[name]['D_cf4'],
                'source': 'CF4_flow',
                'fD_original': fD,
            }
        else:
            # Fallback to SPARC Hubble flow
            catalog_hybrid[name] = {
                'D_Mpc': D_sparc,
                'source': 'SPARC_Hubble_flow',
                'fD_original': fD,
            }
    else:
        catalog_hybrid[name] = {
            'D_Mpc': D_sparc,
            'source': f'SPARC_fD{fD}',
            'fD_original': fD,
        }

    # ---- Full CF4 catalog: use CF4 for everything ----
    if name in cf4_available:
        catalog_cf4[name] = {
            'D_Mpc': cf4_cache[name]['D_cf4'],
            'source': 'CF4_flow',
            'fD_original': fD,
        }
    else:
        # Fallback to SPARC if no CF4 available
        catalog_cf4[name] = {
            'D_Mpc': D_sparc,
            'source': f'SPARC_{fD_label}_fallback',
            'fD_original': fD,
        }

print(f"  SPARC catalog:  {len(catalog_sparc)} galaxies")
print(f"  Hybrid catalog: {len(catalog_hybrid)} galaxies")
print(f"  CF4 catalog:    {len(catalog_cf4)} galaxies")

# ================================================================
# STEP 5: Compute per-galaxy ratios between catalogs
# ================================================================
print("\n[5] Computing inter-catalog distance ratios...")

ratios_hybrid_sparc = []
ratios_cf4_sparc = []
ratios_cf4_hybrid = []

change_10_hs = 0
change_20_hs = 0
change_50_hs = 0
change_10_cs = 0
change_20_cs = 0
change_50_cs = 0

for name in catalog_sparc:
    D_s = catalog_sparc[name]['D_Mpc']
    D_h = catalog_hybrid[name]['D_Mpc']
    D_c = catalog_cf4[name]['D_Mpc']

    r_hs = D_h / D_s
    r_cs = D_c / D_s
    r_ch = D_c / D_h

    ratios_hybrid_sparc.append(r_hs)
    ratios_cf4_sparc.append(r_cs)
    ratios_cf4_hybrid.append(r_ch)

    # Count galaxies changing by >10%, >20%, >50%
    if abs(r_hs - 1.0) > 0.10:
        change_10_hs += 1
    if abs(r_hs - 1.0) > 0.20:
        change_20_hs += 1
    if abs(r_hs - 1.0) > 0.50:
        change_50_hs += 1
    if abs(r_cs - 1.0) > 0.10:
        change_10_cs += 1
    if abs(r_cs - 1.0) > 0.20:
        change_20_cs += 1
    if abs(r_cs - 1.0) > 0.50:
        change_50_cs += 1

ratios_hybrid_sparc = np.array(ratios_hybrid_sparc)
ratios_cf4_sparc = np.array(ratios_cf4_sparc)
ratios_cf4_hybrid = np.array(ratios_cf4_hybrid)

print(f"\n  Hybrid vs SPARC:")
print(f"    Mean ratio: {np.mean(ratios_hybrid_sparc):.3f}")
print(f"    Std ratio:  {np.std(ratios_hybrid_sparc):.3f}")
print(f"    Galaxies changing by >10%: {change_10_hs}")
print(f"    Galaxies changing by >20%: {change_20_hs}")
print(f"    Galaxies changing by >50%: {change_50_hs}")

print(f"\n  Full CF4 vs SPARC:")
print(f"    Mean ratio: {np.mean(ratios_cf4_sparc):.3f}")
print(f"    Std ratio:  {np.std(ratios_cf4_sparc):.3f}")
print(f"    Galaxies changing by >10%: {change_10_cs}")
print(f"    Galaxies changing by >20%: {change_20_cs}")
print(f"    Galaxies changing by >50%: {change_50_cs}")

# Show the most changed galaxies
print(f"\n  Top 10 most-changed galaxies (CF4 vs SPARC):")
print(f"  {'Galaxy':15s} {'D_SPARC':>8s} {'D_CF4':>8s} {'Ratio':>8s} {'fD':>4s}")
print(f"  {'-'*48}")
changes = [(name, catalog_sparc[name]['D_Mpc'], catalog_cf4[name]['D_Mpc'],
             catalog_cf4[name]['D_Mpc'] / catalog_sparc[name]['D_Mpc'],
             catalog_sparc[name]['fD_original'])
           for name in catalog_sparc]
changes.sort(key=lambda x: -abs(x[3] - 1.0))
for name, ds, dc, ratio, fd in changes[:10]:
    print(f"  {name:15s} {ds:8.2f} {dc:8.2f} {ratio:8.3f} {fd:4d}")

# ================================================================
# STEP 6: Catalog source breakdown
# ================================================================
print(f"\n{'='*72}")
print("CATALOG SOURCE BREAKDOWN")
print(f"{'='*72}")

for catalog_name, catalog in [('SPARC', catalog_sparc), ('Hybrid', catalog_hybrid), ('CF4', catalog_cf4)]:
    sources = {}
    for entry in catalog.values():
        src = entry['source']
        sources[src] = sources.get(src, 0) + 1
    print(f"\n  {catalog_name}:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src:30s} {count:4d}")

# ================================================================
# STEP 7: Save catalogs and summary
# ================================================================
print(f"\n{'='*72}")
print("SAVING CATALOGS")
print(f"{'='*72}")

# Save individual catalogs
for catalog_name, catalog in [('sparc', catalog_sparc), ('hybrid', catalog_hybrid), ('cf4', catalog_cf4)]:
    outpath = os.path.join(PROJECT_ROOT, 'data', f'distance_catalog_{catalog_name}.json')
    with open(outpath, 'w') as f:
        json.dump(catalog, f, indent=2)
    print(f"  Saved: {outpath}")

# Save summary
summary = {
    'test_name': 'distance_catalog_builder',
    'description': 'Three distance catalogs for 175 SPARC galaxies',
    'n_galaxies': len(catalog_sparc),
    'catalogs': {
        'sparc': {
            'description': 'All SPARC original distances',
            'n_galaxies': len(catalog_sparc),
        },
        'hybrid': {
            'description': 'SPARC for gold-standard & UMa; NED TRGB or CF4 for Hubble flow',
            'n_galaxies': len(catalog_hybrid),
            'n_ned_upgrades': len(ned_upgrades),
            'ned_upgraded_galaxies': sorted(ned_upgrades.keys()),
        },
        'cf4': {
            'description': 'CF4 flow model for all (pathological for cluster members)',
            'n_galaxies': len(catalog_cf4),
            'n_cf4_available': len(cf4_available),
            'n_sparc_fallback': sum(1 for e in catalog_cf4.values() if 'fallback' in e['source']),
        },
    },
    'inter_catalog_ratios': {
        'hybrid_vs_sparc': {
            'mean': round(float(np.mean(ratios_hybrid_sparc)), 4),
            'std': round(float(np.std(ratios_hybrid_sparc)), 4),
            'median': round(float(np.median(ratios_hybrid_sparc)), 4),
            'n_change_gt_10pct': change_10_hs,
            'n_change_gt_20pct': change_20_hs,
            'n_change_gt_50pct': change_50_hs,
        },
        'cf4_vs_sparc': {
            'mean': round(float(np.mean(ratios_cf4_sparc)), 4),
            'std': round(float(np.std(ratios_cf4_sparc)), 4),
            'median': round(float(np.median(ratios_cf4_sparc)), 4),
            'n_change_gt_10pct': change_10_cs,
            'n_change_gt_20pct': change_20_cs,
            'n_change_gt_50pct': change_50_cs,
        },
    },
    'top_10_most_changed_cf4_vs_sparc': [
        {
            'galaxy': name,
            'D_sparc': round(ds, 2),
            'D_cf4': round(dc, 2),
            'ratio': round(ratio, 3),
            'fD': fd,
        }
        for name, ds, dc, ratio, fd in changes[:10]
    ],
}

outpath = os.path.join(RESULTS_DIR, 'summary_distance_catalogs.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {outpath}")

print("\nDone.")
