#!/usr/bin/env python3
"""
assess_vizier_catalogs.py — Assess all downloaded VizieR rotation curve catalogs.

For each catalog:
  1. Count unique galaxies and total data points
  2. Check overlap with SPARC
  3. Assess RAR-readiness (has baryonic decomposition?)
  4. Print summary
"""

import os
import sys
import re
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIZIER_DIR = os.path.join(BASE, 'data', 'vizier_catalogs')
SPARC_DIR = os.path.join(BASE, 'data', 'sparc')


def load_sparc_names():
    """Get set of SPARC galaxy names (various forms for matching)."""
    names = set()
    table2 = os.path.join(SPARC_DIR, 'SPARC_table2_rotmods.dat')
    if os.path.exists(table2):
        with open(table2) as f:
            for line in f:
                if len(line) > 11 and not line.startswith('#'):
                    n = line[0:11].strip()
                    if n:
                        names.add(n)
    return names


def normalize_name(name):
    """Normalize galaxy names for cross-matching.

    Handles variations like:
      NGC 3031 -> NGC3031
      UGC 128 -> UGC128
      DDO 154 -> DDO154
      N3726 -> NGC3726
      IC 2574 -> IC2574
    """
    name = name.strip().strip('"').strip()
    # Remove spaces between prefix and number
    name = re.sub(r'^(NGC|UGC|IC|DDO|ESO|PGC|CGCG|MCG|LSBC?|UGCA)\s+', r'\1', name, flags=re.IGNORECASE)
    # Handle abbreviated NGC names like N3726
    name = re.sub(r'^N(\d)', r'NGC\1', name)
    # Uppercase
    name = name.upper()
    return name


def build_sparc_normalized():
    """Build normalized SPARC name set + mapping."""
    raw_names = load_sparc_names()
    normalized = {}
    for n in raw_names:
        nn = normalize_name(n)
        normalized[nn] = n
    return normalized


def parse_tsv(filepath):
    """Generic TSV parser — returns header + list of row dicts."""
    rows = []
    header = None
    with open(filepath) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if header is None:
                header = [h.strip() for h in parts]
                continue
            row = {}
            for i, h in enumerate(header):
                if i < len(parts):
                    row[h] = parts[i].strip().strip('"')
                else:
                    row[h] = ''
            rows.append(row)
    return header, rows


def assess_deblok2002_processed():
    """de Blok & Bosma 2002 — 24 LSB galaxies WITH Vgas, Vdisk decomposition."""
    fp = os.path.join(VIZIER_DIR, 'deblok2002_processed.tsv')
    if not os.path.exists(fp):
        return None
    header, rows = parse_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue
        if name not in galaxies:
            galaxies[name] = {'n_pts': 0, 'has_Vgas': False, 'has_Vdisk': False}
        galaxies[name]['n_pts'] += 1
        if row.get('Vgas', ''):
            galaxies[name]['has_Vgas'] = True
        if row.get('Vdisk', ''):
            galaxies[name]['has_Vdisk'] = True

    return {
        'catalog': 'de Blok & Bosma 2002 (J/A+A/385/816)',
        'description': 'LSB galaxies with HI rotation curves + mass models',
        'n_galaxies': len(galaxies),
        'n_points': sum(g['n_pts'] for g in galaxies.values()),
        'has_decomposition': True,
        'columns': header,
        'galaxies': galaxies,
        'names': list(galaxies.keys()),
    }


def assess_ghasp_epinat2008a():
    """GHASP Epinat+2008a — 93 galaxies, Hα Fabry-Perot RCs."""
    fp = os.path.join(VIZIER_DIR, 'ghasp_epinat2008a.tsv')
    if not os.path.exists(fp):
        return None
    header, rows = parse_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue
        if name not in galaxies:
            galaxies[name] = {'n_pts': 0}
        galaxies[name]['n_pts'] += 1

    return {
        'catalog': 'GHASP Epinat+2008a (J/MNRAS/388/500)',
        'description': 'Hα Fabry-Perot rotation curves (approaching/receding sides)',
        'n_galaxies': len(galaxies),
        'n_points': sum(g['n_pts'] for g in galaxies.values()),
        'has_decomposition': False,
        'columns': header,
        'galaxies': galaxies,
        'names': list(galaxies.keys()),
    }


def assess_ghasp_epinat2008b():
    """GHASP Epinat+2008b — 82 galaxies."""
    fp = os.path.join(VIZIER_DIR, 'ghasp_epinat2008b.tsv')
    if not os.path.exists(fp):
        return None
    header, rows = parse_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue
        if name not in galaxies:
            galaxies[name] = {'n_pts': 0}
        galaxies[name]['n_pts'] += 1

    return {
        'catalog': 'GHASP Epinat+2008b (J/MNRAS/390/466)',
        'description': 'GHASP kinematic parameters + rotation curves',
        'n_galaxies': len(galaxies),
        'n_points': sum(g['n_pts'] for g in galaxies.values()),
        'has_decomposition': False,
        'columns': header,
        'galaxies': galaxies,
        'names': list(galaxies.keys()),
    }


def assess_ghasp_gomezlopez2019():
    """GHASP Gomez-Lopez+2019 — ~60 galaxies."""
    fp = os.path.join(VIZIER_DIR, 'ghasp_gomezlopez2019.tsv')
    if not os.path.exists(fp):
        return None
    header, rows = parse_tsv(fp)

    # This catalog uses HRS number instead of name
    galaxies = {}
    for row in rows:
        name = row.get('HRS', '').strip()
        if not name:
            continue
        if name not in galaxies:
            galaxies[name] = {'n_pts': 0}
        galaxies[name]['n_pts'] += 1

    return {
        'catalog': 'Gomez-Lopez+2019 (J/A+A/631/A71)',
        'description': 'GHASP Hα Fabry-Perot rotation curves (HRS galaxies)',
        'n_galaxies': len(galaxies),
        'n_points': sum(g['n_pts'] for g in galaxies.values()),
        'has_decomposition': False,
        'columns': header,
        'galaxies': galaxies,
        'names': list(galaxies.keys()),
        'note': 'Uses HRS number IDs, NOT galaxy names — need cross-match table',
    }


def assess_verheijen2001():
    """Verheijen+2001 — 41 UMa cluster galaxies, HI RCs."""
    fp = os.path.join(VIZIER_DIR, 'verheijen2001_rc.tsv')
    if not os.path.exists(fp):
        return None
    header, rows = parse_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('Name', '').strip()
        if not name:
            continue
        if name not in galaxies:
            galaxies[name] = {'n_pts': 0}
        galaxies[name]['n_pts'] += 1

    return {
        'catalog': 'Verheijen+2001 (J/A+A/370/765)',
        'description': 'UMa cluster HI rotation curves',
        'n_galaxies': len(galaxies),
        'n_points': sum(g['n_pts'] for g in galaxies.values()),
        'has_decomposition': False,
        'columns': header,
        'galaxies': galaxies,
        'names': list(galaxies.keys()),
    }


def assess_phangs_lang2020():
    """PHANGS Lang+2020 — 70 galaxies, CO rotation curves."""
    fp = os.path.join(VIZIER_DIR, 'phangs_lang2020.tsv')
    if not os.path.exists(fp):
        return None
    header, rows = parse_tsv(fp)

    galaxies = {}
    for row in rows:
        name = row.get('ID', '').strip()
        if not name:
            continue
        if name not in galaxies:
            galaxies[name] = {'n_pts': 0}
        galaxies[name]['n_pts'] += 1

    return {
        'catalog': 'PHANGS Lang+2020 (J/ApJ/897/122)',
        'description': 'CO rotation curves from PHANGS-ALMA',
        'n_galaxies': len(galaxies),
        'n_points': sum(g['n_pts'] for g in galaxies.values()),
        'has_decomposition': False,
        'columns': header,
        'galaxies': galaxies,
        'names': list(galaxies.keys()),
    }


def main():
    sparc_normalized = build_sparc_normalized()
    sparc_raw = load_sparc_names()
    print(f"SPARC: {len(sparc_raw)} galaxies, {len(sparc_normalized)} normalized names")

    # Assess all catalogs
    assessors = [
        assess_deblok2002_processed,
        assess_ghasp_epinat2008a,
        assess_ghasp_epinat2008b,
        assess_ghasp_gomezlopez2019,
        assess_verheijen2001,
        assess_phangs_lang2020,
    ]

    all_new_galaxies = set()

    print("\n" + "=" * 80)
    for assess_fn in assessors:
        result = assess_fn()
        if result is None:
            print(f"\n[SKIP] {assess_fn.__name__}: file not found")
            continue

        print(f"\n{'─' * 80}")
        print(f"  Catalog: {result['catalog']}")
        print(f"  Description: {result['description']}")
        print(f"  Galaxies: {result['n_galaxies']}")
        print(f"  Data points: {result['n_points']}")
        print(f"  Has baryonic decomposition: {result['has_decomposition']}")
        print(f"  Columns: {result['columns']}")
        if 'note' in result:
            print(f"  NOTE: {result['note']}")

        # Check SPARC overlap
        in_sparc = []
        not_in_sparc = []
        for name in result['names']:
            nn = normalize_name(name)
            if nn in sparc_normalized:
                in_sparc.append((name, sparc_normalized[nn]))
            else:
                not_in_sparc.append(name)

        print(f"\n  SPARC overlap: {len(in_sparc)}/{result['n_galaxies']} in SPARC")
        if in_sparc:
            print(f"    Matched: {', '.join(f'{a}->{b}' for a, b in in_sparc[:20])}")
            if len(in_sparc) > 20:
                print(f"    ... and {len(in_sparc)-20} more")

        new_count = len(not_in_sparc)
        new_pts = sum(result['galaxies'][n]['n_pts'] for n in not_in_sparc)
        print(f"  NEW (not in SPARC): {new_count} galaxies, {new_pts} data points")
        if not_in_sparc and new_count <= 30:
            for n in sorted(not_in_sparc):
                pts = result['galaxies'][n]['n_pts']
                sufficient = "✓" if pts >= 5 else "✗ (<5 pts)"
                print(f"    {n}: {pts} pts {sufficient}")

        # Count those with ≥5 points
        new_sufficient = [n for n in not_in_sparc if result['galaxies'][n]['n_pts'] >= 5]
        print(f"  NEW with ≥5 pts: {len(new_sufficient)} galaxies")

        all_new_galaxies.update(normalize_name(n) for n in new_sufficient)

    # Grand summary
    print(f"\n{'=' * 80}")
    print(f"GRAND SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total unique NEW galaxies (not in SPARC, ≥5 pts): {len(all_new_galaxies)}")
    print(f"SPARC: {len(sparc_raw)} galaxies")
    print(f"Potential extended dataset: {len(sparc_raw) + len(all_new_galaxies)} galaxies")


if __name__ == '__main__':
    main()
