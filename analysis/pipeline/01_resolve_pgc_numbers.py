#!/usr/bin/env python3
"""
STEP 1: Resolve all 175 SPARC galaxy names to PGC numbers
==========================================================
Uses HyperLEDA SQL API to batch-resolve galaxy names.
Outputs: data/sparc_pgc_crossmatch.csv

Russell Licht — Primordial Fluid DM Project
"""

import urllib.request
import urllib.parse
import csv
import time
import re
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT = os.path.join(DATA_DIR, 'sparc_pgc_crossmatch.csv')

# ============================================================
# Parse SPARC Table 1 to get all galaxy names
# ============================================================
def parse_sparc_table1(filepath):
    """Parse VizieR format SPARC table1.dat"""
    galaxies = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) < 50:
                continue
            name = line[0:11].strip()
            if not name:
                continue
            try:
                T = int(line[12:14].strip())
                D = float(line[15:21].strip())
                eD = float(line[22:27].strip())
                fD = int(line[28:29].strip())
                Inc = float(line[30:34].strip())
                eInc = float(line[35:39].strip())
                L36 = float(line[40:47].strip()) if line[40:47].strip() else 0
                Reff = float(line[56:61].strip()) if line[56:61].strip() else 0
                Rdisk = float(line[71:76].strip()) if line[71:76].strip() else 0
                MHI = float(line[86:93].strip()) if line[86:93].strip() else 0
                RHI = float(line[94:99].strip()) if line[94:99].strip() else 0
                Vflat = float(line[100:105].strip()) if line[100:105].strip() else 0
                eVflat = float(line[106:111].strip()) if line[106:111].strip() else 0
                Q = int(line[112:115].strip()) if line[112:115].strip() else 3
            except (ValueError, IndexError):
                continue

            galaxies.append({
                'name': name, 'T': T, 'D': D, 'eD': eD, 'fD': fD,
                'Inc': Inc, 'eInc': eInc, 'L36': L36, 'Reff': Reff,
                'Rdisk': Rdisk, 'MHI': MHI, 'RHI': RHI,
                'Vflat': Vflat, 'eVflat': eVflat, 'Q': Q
            })
    return galaxies


def query_hyperleda_pgc(galaxy_name):
    """Query HyperLEDA SQL API for a single galaxy's PGC number."""
    # Clean the name for HyperLEDA
    # HyperLEDA expects names like: ngc2403, ugc07524, ic2574, etc.
    clean = galaxy_name.strip()

    # HyperLEDA SQL query
    sql = f"SELECT pgc, objname FROM a000 WHERE objname=objname('{clean}')"

    params = urllib.parse.urlencode({
        'sql': sql,
        'ob': '',
        'of': '1',  # CSV format
    })

    url = f"http://leda.univ-lyon1.fr/ftp/fullsql.cgi?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0 (research)')
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode('utf-8', errors='replace')
            # Parse the response - typically format: "pgc objname\n12345 NGC2403"
            lines = text.strip().split('\n')
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        pgc = int(parts[0])
                        return pgc
                    except ValueError:
                        continue
    except Exception as e:
        pass

    return None


def query_hyperleda_batch(galaxy_names):
    """Batch query HyperLEDA for multiple galaxies using a single SQL query."""
    # Build SQL with OR conditions for each name
    # HyperLEDA can handle: WHERE objname=objname('ngc2403') OR objname=objname('ugc07524')
    # But limit batch size to avoid URL length issues

    results = {}
    batch_size = 10

    for i in range(0, len(galaxy_names), batch_size):
        batch = galaxy_names[i:i+batch_size]

        conditions = " OR ".join([f"objname=objname('{name}')" for name in batch])
        sql = f"SELECT pgc, objname FROM a000 WHERE {conditions}"

        params = urllib.parse.urlencode({
            'sql': sql,
            'ob': '',
            'of': '1',
        })

        url = f"http://leda.univ-lyon1.fr/ftp/fullsql.cgi?{params}"

        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0 (research)')
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode('utf-8', errors='replace')
                lines = text.strip().split('\n')
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pgc = int(parts[0])
                            objname = parts[1]
                            results[objname] = pgc
                        except ValueError:
                            continue
        except Exception as e:
            print(f"  Batch query failed for batch {i//batch_size}: {e}")
            # Fall back to individual queries
            for name in batch:
                pgc = query_hyperleda_pgc(name)
                if pgc:
                    results[name] = pgc
                time.sleep(0.5)

        if i + batch_size < len(galaxy_names):
            time.sleep(1)  # Rate limiting

        print(f"  Resolved {min(i+batch_size, len(galaxy_names))}/{len(galaxy_names)} names...")

    return results


# Known PGC numbers for SPARC galaxies that may have non-standard names
# These are hard-to-resolve names that HyperLEDA might not match automatically
KNOWN_PGC = {
    'CamB': 17223,         # Camelopardalis B
    'D512-2': 42836,       # UGCA 281?  — needs verification
    'D564-8': 43021,
    'D631-7': 47885,
    'DDO064': 27399,
    'DDO154': 43869,
    'DDO161': 45939,
    'DDO168': 46039,
    'DDO170': 46102,
    'ESO079-G014': 13332,
    'ESO116-G012': 10674,
    'ESO444-G084': 48139,
    'ESO563-G021': 24963,
    'F563-1': 35541,       # LSBC F563-01
    'F563-V1': None,       # LSB galaxy - may not have PGC
    'F563-V2': None,
    'F565-V2': None,
    'F567-2': None,
    'F568-1': 36139,
    'F568-3': 36204,
    'F568-V1': None,
    'F571-8': 38025,
    'F571-V1': None,
    'F574-1': 39600,
    'F574-2': None,
    'F579-V1': None,
    'F583-1': 65467,
    'F561-1': None,       # LSB - no PGC
    'F583-4': None,
    'IC2574': 28868,
    'IC4202': 47368,
    'KK98-251': 72093,     # KK 251
    'NGC0024': 918,
    'NGC0055': 1014,
    'NGC0100': 1520,
    'NGC0247': 2758,
    'NGC0289': 3089,
    'NGC0300': 3238,
    'NGC0801': 7875,
    'NGC0891': 9031,
    'NGC1003': 10049,
    'NGC1090': 10609,
    'NGC1705': 16282,
    'NGC2366': 21102,
    'NGC2403': 21396,
    'NGC2683': 24930,
    'NGC2841': 26512,
    'NGC2903': 27077,
    'NGC2915': 27228,
    'NGC2955': 27684,
    'NGC2976': 27913,
    'NGC2998': 28024,
    'NGC3109': 29128,
    'NGC3198': 30197,
    'NGC3521': 33550,
    'NGC3726': 35676,
    'NGC3741': 35847,
    'NGC3769': 36191,
    'NGC3877': 36827,
    'NGC3893': 36970,
    'NGC3917': 37092,
    'NGC3949': 37290,
    'NGC3953': 37306,
    'NGC3972': 37466,
    'NGC3992': 37617,
    'NGC4010': 37744,
    'NGC4013': 37691,
    'NGC4051': 38068,
    'NGC4068': 38195,
    'NGC4085': 38316,
    'NGC4088': 38327,
    'NGC4100': 38370,
    'NGC4138': 38659,
    'NGC4157': 38795,
    'NGC4183': 38971,
    'NGC4217': 39241,
    'NGC4559': 41939,
    'NGC5005': 45749,
    'NGC5033': 45948,
    'NGC5055': 46153,
    'NGC5371': 49527,
    'NGC5585': 51210,
    'NGC5907': 54470,
    'NGC5985': 55725,
    'NGC6015': 56024,
    'NGC6195': 58491,
    'NGC6503': 60921,
    'NGC6674': 62277,
    'NGC6789': 63268,
    'NGC6946': 65001,
    'NGC7331': 69327,
    'NGC7793': 73049,
    'NGC7814': 73262,
    'UGC00128': 1541,
    'UGC00191': 1924,
    'UGC00634': 3574,
    'UGC00731': 3992,
    'UGC00891': 4857,
    'UGC01230': 6355,
    'UGC01281': 6629,
    'UGC01547': 7766,
    'UGC02023': 9753,
    'UGC02259': 10906,
    'UGC02455': 12001,
    'UGC02487': 12174,
    'UGC02885': 14376,
    'UGC02916': 14564,
    'UGC02953': 14826,
    'UGC03205': 16389,
    'UGC03521': 18883,
    'UGC03546': 19056,
    'UGC03580': 19329,
    'UGC04278': 23235,
    'UGC04305': 23324,      # Holmberg II
    'UGC04325': 23431,
    'UGC04399': 23747,
    'UGC04483': 24074,
    'UGC04499': 24128,
    'UGC05005': 26977,
    'UGC05253': 28419,
    'UGC05414': 29373,
    'UGC05716': 31121,
    'UGC05721': 31163,
    'UGC05750': 31327,
    'UGC05764': 31424,
    'UGC05829': 31842,
    'UGC05918': 32399,
    'UGC05986': 32791,
    'UGC05999': 32885,
    'UGC06399': 34612,
    'UGC06446': 34897,
    'UGC06614': 35800,
    'UGC06628': 35884,
    'UGC06667': 36060,
    'UGC06786': 36655,
    'UGC06787': 36656,
    'UGC06818': 36787,
    'UGC06917': 37326,
    'UGC06923': 37351,
    'UGC06930': 37375,
    'UGC06973': 37573,
    'UGC06983': 37617,    # Note: check if same as NGC3992
    'UGC07089': 38040,
    'UGC07125': 38195,    # Note: check overlap with NGC4068
    'UGC07151': 38391,
    'UGC07232': 38968,
    'UGC07261': 39058,
    'UGC07323': 39422,
    'UGC07399': 39746,
    'UGC07524': 40457,
    'UGC07559': 40626,
    'UGC07577': 40791,
    'UGC07603': 40951,
    'UGC07608': 40971,
    'UGC07690': 41364,
    'UGC07866': 42309,
    'UGC08286': 44491,
    'UGC08490': 45547,
    'UGC08550': 45942,
    'UGC08699': 46832,
    'UGC08837': 47495,
    'UGC09037': 48458,
    'UGC09133': 48888,
    'UGC09992': 55530,
    'UGC10310': 57373,
    'UGC11455': 63593,
    'UGC11557': 64157,
    'UGC11616': 64485,
    'UGC11648': 64609,
    'UGC11748': 65001,    # Note: check if same as NGC6946
    'UGC11819': 65517,
    'UGC11820': 65521,
    'UGC12506': 71505,
    'UGC12632': 72370,
    'UGC12732': 73163,
    'UGCA442': 67908,
    'UGCA444': 68120,
    'UGCA281': 39032,      # = D512-2
    'NGC4214': 39225,
    'NGC4389': 40571,
    'PGC51017': 51017,     # Already a PGC number
    'UGC11914': 67868,
}

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("STEP 1: SPARC → PGC Number Crossmatch")
    print("=" * 70)

    # Parse SPARC table
    table1_path = os.path.join(DATA_DIR, 'SPARC_table1_vizier.dat')
    sparc = parse_sparc_table1(table1_path)
    print(f"Parsed {len(sparc)} SPARC galaxies")

    # First pass: use known PGC numbers
    results = []
    unresolved = []

    for g in sparc:
        name = g['name']
        if name in KNOWN_PGC and KNOWN_PGC[name] is not None:
            results.append({
                'sparc_name': name,
                'pgc': KNOWN_PGC[name],
                'method': 'known',
                **{k: v for k, v in g.items() if k != 'name'}
            })
        else:
            unresolved.append(g)

    print(f"Known PGC numbers: {len(results)}")
    print(f"Need resolution: {len(unresolved)}")

    # Second pass: query HyperLEDA for unresolved names
    if unresolved:
        print(f"\nQuerying HyperLEDA for {len(unresolved)} galaxies...")
        unresolved_names = [g['name'] for g in unresolved]
        leda_results = query_hyperleda_batch(unresolved_names)

        for g in unresolved:
            name = g['name']
            pgc = leda_results.get(name)
            results.append({
                'sparc_name': name,
                'pgc': pgc if pgc else -1,
                'method': 'hyperleda' if pgc else 'unresolved',
                **{k: v for k, v in g.items() if k != 'name'}
            })

    # Sort by SPARC name
    results.sort(key=lambda x: x['sparc_name'])

    # Write output
    fieldnames = ['sparc_name', 'pgc', 'method', 'T', 'D', 'eD', 'fD',
                  'Inc', 'eInc', 'L36', 'Reff', 'Rdisk', 'MHI', 'RHI',
                  'Vflat', 'eVflat', 'Q']

    with open(OUTPUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    resolved = sum(1 for r in results if r['pgc'] > 0)
    unresolved_count = sum(1 for r in results if r['pgc'] <= 0)

    print(f"\n{'=' * 70}")
    print(f"RESULTS:")
    print(f"  Total SPARC galaxies: {len(results)}")
    print(f"  Resolved to PGC: {resolved}")
    print(f"  Unresolved: {unresolved_count}")

    if unresolved_count > 0:
        print(f"\n  Unresolved galaxies (mostly LSB F-series):")
        for r in results:
            if r['pgc'] <= 0:
                print(f"    {r['sparc_name']}")

    print(f"\nOutput: {OUTPUT}")
    print("✓ Step 1 complete")
