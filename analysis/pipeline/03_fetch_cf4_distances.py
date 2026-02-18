#!/usr/bin/env python3
"""
STEP 3: Fetch CF4 Flow Model Distances for SPARC Galaxies
==========================================================
1. Downloads RA/Dec from VizieR for all 175 SPARC galaxies
2. Gets heliocentric velocities from SIMBAD TAP + NED fallback
3. Queries the EDD CF4 calculator API (batch JSON endpoint)
4. Saves cf4_distance_cache.json for use by 02_cf4_rar_pipeline.py

Russell Licht — Primordial Fluid DM Project
Feb 2026
"""

import json
import csv
import os
import sys
import time
import urllib.request
import urllib.parse
import re
import ssl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
CACHE_FILE = os.path.join(DATA_DIR, 'cf4_distance_cache.json')
COORD_CACHE = os.path.join(DATA_DIR, 'sparc_coordinates.json')

# Allow unverified SSL for some astronomy services with expired certs
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


# ============================================================
# STEP 1: Get RA/Dec from VizieR for all 175 SPARC galaxies
# ============================================================
def fetch_vizier_coordinates():
    """Fetch RA/Dec for all 175 SPARC galaxies from VizieR."""
    print("\n--- Step 1: Fetching RA/Dec from VizieR ---")

    # VizieR TSV query for J/AJ/152/157/table1 with CDS-resolved coords
    url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
           "-source=J/AJ/152/157/table1&"
           "-out=Name,_RA,_DE&"
           "-out.max=200&"
           "-oc.form=d")

    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0 (research)')

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"  ERROR: VizieR query failed: {e}")
        return None

    # Parse TSV response
    coords = {}
    lines = text.strip().split('\n')
    data_started = False

    for line in lines:
        # Skip header/comment lines
        if line.startswith('#') or line.startswith('-'):
            continue
        if 'Name' in line and '_RA' in line:
            data_started = True
            continue
        if not data_started:
            continue
        if not line.strip():
            continue

        parts = line.split('\t')
        if len(parts) >= 3:
            name = parts[0].strip()
            try:
                ra = float(parts[1].strip())
                dec = float(parts[2].strip())
                coords[name] = {'ra': ra, 'dec': dec}
            except (ValueError, IndexError):
                continue

    print(f"  Got coordinates for {len(coords)} galaxies from VizieR")
    return coords


# ============================================================
# STEP 2: Get heliocentric velocities from SIMBAD TAP
# ============================================================
def fetch_simbad_velocities(galaxy_names):
    """Fetch heliocentric velocities from SIMBAD TAP service."""
    print("\n--- Step 2: Fetching velocities from SIMBAD ---")

    velocities = {}

    # Process in batches of 30
    batch_size = 30
    for i in range(0, len(galaxy_names), batch_size):
        batch = galaxy_names[i:i+batch_size]

        # Build SIMBAD name variants
        name_list = []
        for name in batch:
            # SIMBAD expects: 'NGC 2403' not 'NGC2403', 'UGC 07524' etc.
            sname = sparc_to_simbad_name(name)
            name_list.append(sname)

        # SIMBAD TAP query joining ident and basic tables
        # Use WHERE id IN (...) for batch lookup
        quoted_names = ", ".join([f"'{n}'" for n in name_list])

        adql = (f"SELECT ident.id, basic.main_id, basic.rvz_radvel, basic.rvz_type "
                f"FROM ident JOIN basic ON ident.oidref = basic.oid "
                f"WHERE ident.id IN ({quoted_names})")

        params = urllib.parse.urlencode({
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': 'csv',
            'QUERY': adql,
        })

        url = f"https://simbad.cds.unistra.fr/simbad/sim-tap/sync?{params}"

        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0')
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode('utf-8', errors='replace')

            # Parse CSV response
            lines = text.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) >= 3 and parts[2].strip():
                        simbad_id = parts[0].strip().strip('"')
                        vel_str = parts[2].strip().strip('"')
                        try:
                            vel = float(vel_str)
                            # Map back to SPARC name
                            sparc_name = simbad_to_sparc_name(simbad_id, batch)
                            if sparc_name:
                                velocities[sparc_name] = vel
                        except ValueError:
                            continue
        except Exception as e:
            print(f"  SIMBAD batch {i//batch_size + 1} failed: {e}")

        if i + batch_size < len(galaxy_names):
            time.sleep(0.5)

        print(f"  SIMBAD: {min(i+batch_size, len(galaxy_names))}/{len(galaxy_names)} queried, "
              f"{len(velocities)} velocities found")

    return velocities


def sparc_to_simbad_name(sparc_name):
    """Convert SPARC galaxy name to SIMBAD-compatible format."""
    name = sparc_name.strip()

    # Special cases
    special = {
        'CamB': 'Camelopardalis B',
        'D512-2': '[KK98] 251',
        'D564-8': 'HIPASS J0946-01',
        'D631-7': 'DDO 189',
        'KK98-251': '[KK98] 251',
        'PGC51017': 'PGC 51017',
    }
    if name in special:
        return special[name]

    # NGC/IC: add space, remove leading zeros
    m = re.match(r'^(NGC|IC)(\d+)$', name)
    if m:
        prefix, num = m.group(1), m.group(2)
        return f"{prefix} {int(num)}"

    # UGC: add space, keep leading zeros for SIMBAD
    m = re.match(r'^(UGC|UGCA)(\d+)$', name)
    if m:
        prefix, num = m.group(1), m.group(2)
        return f"{prefix} {int(num)}"

    # DDO: add space
    m = re.match(r'^DDO(\d+)$', name)
    if m:
        return f"DDO {int(m.group(1))}"

    # ESO: format as 'ESO NNN-GNN'  (SIMBAD expects 'ESO NNN-NN')
    m = re.match(r'^ESO(\d+)-G(\d+)$', name)
    if m:
        return f"ESO {m.group(1)}-{m.group(2)}"

    # F-series (LSB): try LSBC format
    m = re.match(r'^(F\d+)-(\w+)$', name)
    if m:
        return f"LSBC {m.group(1)}-{m.group(2)}"

    return name


def simbad_to_sparc_name(simbad_id, sparc_batch):
    """Map a SIMBAD identifier back to its SPARC name."""
    simbad_id_clean = simbad_id.strip().upper().replace(' ', '')

    for sparc_name in sparc_batch:
        sname = sparc_to_simbad_name(sparc_name).upper().replace(' ', '')
        if simbad_id_clean == sname:
            return sparc_name
        # Also try the original SPARC name with spaces removed
        if simbad_id_clean == sparc_name.upper().replace(' ', ''):
            return sparc_name

    # Fallback: try matching by number
    for sparc_name in sparc_batch:
        # Extract numeric part
        m1 = re.search(r'(\d+)', simbad_id_clean)
        m2 = re.search(r'(\d+)', sparc_name)
        if m1 and m2:
            # Check if the prefix type matches
            simbad_prefix = re.match(r'[A-Z]+', simbad_id_clean)
            sparc_prefix = re.match(r'[A-Z]+', sparc_name)
            if (simbad_prefix and sparc_prefix and
                simbad_prefix.group() == sparc_prefix.group() and
                m1.group(1) == m2.group(1)):
                return sparc_name

    return None


# ============================================================
# STEP 2b: NED fallback for velocities not found in SIMBAD
# ============================================================
def fetch_ned_velocity(galaxy_name, ra=None, dec=None):
    """Get heliocentric velocity from NED for a single galaxy."""

    # Try name-based search first
    ned_name = sparc_to_ned_name(galaxy_name)

    url = ("https://ned.ipac.caltech.edu/cgi-bin/objsearch?"
           f"objname={urllib.parse.quote(ned_name)}&extend=no&"
           "hconst=67.8&omegam=0.308&omegav=0.692&corr_z=1&"
           "out_csys=Equatorial&out_equinox=J2000.0&"
           "obj_sort=RA+or+Longitude&of=ascii_bar&zv_breession=3&"
           "list_limit=1&img_stamp=NO")

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0')
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            text = resp.read().decode('utf-8', errors='replace')

        # Parse NED ASCII output for velocity
        for line in text.split('\n'):
            if '|' in line and not line.startswith('#'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 8:
                    vel_str = parts[6].strip() if len(parts) > 6 else ''
                    try:
                        return float(vel_str)
                    except ValueError:
                        continue
    except Exception:
        pass

    # Fallback: coordinate-based NED TAP query
    if ra is not None and dec is not None:
        return fetch_ned_velocity_by_coord(ra, dec)

    return None


def sparc_to_ned_name(sparc_name):
    """Convert SPARC name to NED-compatible format."""
    name = sparc_name.strip()

    special = {
        'CamB': 'Camelopardalis B',
        'D512-2': 'UGCA 281',
        'D564-8': 'ESO 418-008',
        'D631-7': 'DDO 189',
        'KK98-251': 'KK 251',
        'PGC51017': 'PGC 51017',
    }
    if name in special:
        return special[name]

    # NGC/IC: add space
    m = re.match(r'^(NGC|IC)(\d+)$', name)
    if m:
        return f"{m.group(1)} {int(m.group(2))}"

    # UGC/UGCA: add space
    m = re.match(r'^(UGC|UGCA)(\d+)$', name)
    if m:
        return f"{m.group(1)} {int(m.group(2))}"

    # DDO: add space
    m = re.match(r'^DDO(\d+)$', name)
    if m:
        return f"DDO {int(m.group(1))}"

    # ESO galaxies
    m = re.match(r'^ESO(\d+)-G(\d+)$', name)
    if m:
        return f"ESO {m.group(1)}-G{m.group(2)}"

    # F-series (LSB): LSBC format
    m = re.match(r'^(F\d+)-([\w]+)$', name)
    if m:
        return f"LSBC {m.group(1)}-{m.group(2)}"

    return name


def fetch_ned_velocity_by_coord(ra, dec, radius_arcmin=1.0):
    """NED TAP cone search for velocity by coordinates."""
    radius_deg = radius_arcmin / 60.0

    adql = (f"SELECT prefname, ra, dec, z FROM objdir "
            f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec), "
            f"CIRCLE('ICRS',{ra},{dec},{radius_deg})) "
            f"AND pretype='G'")

    params = urllib.parse.urlencode({
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'FORMAT': 'csv',
        'QUERY': adql,
    })

    url = f"https://ned.ipac.caltech.edu/tap/sync?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0')
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            text = resp.read().decode('utf-8', errors='replace')

        lines = text.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split(',')
            if len(parts) >= 4 and parts[3].strip():
                z = float(parts[3].strip())
                # Convert redshift to heliocentric velocity
                c = 299792.458  # km/s
                vel = z * c
                return vel
    except Exception:
        pass

    return None


# ============================================================
# STEP 2c: Comprehensive velocity lookup using NED batch
# ============================================================
def fetch_ned_velocities_batch(galaxy_names, coords):
    """Batch fetch velocities from NED using name queries."""
    print("\n--- Step 2b: Fetching remaining velocities from NED ---")

    velocities = {}
    for i, name in enumerate(galaxy_names):
        # Try name-based NED query
        ra = coords.get(name, {}).get('ra')
        dec = coords.get(name, {}).get('dec')

        vel = fetch_ned_velocity(name, ra=ra, dec=dec)
        if vel is not None and abs(vel) > 0:
            velocities[name] = vel

        if (i + 1) % 10 == 0:
            print(f"  NED: {i+1}/{len(galaxy_names)} queried, {len(velocities)} found")

        time.sleep(0.3)  # Rate limit

    return velocities


# ============================================================
# Known velocities for problematic galaxies
# (from NED / HyperLEDA, manually verified)
# ============================================================
KNOWN_VELOCITIES = {
    # Local Group / very nearby (heliocentric, km/s)
    'CamB': 77,
    'DDO154': 375,
    'DDO168': 192,
    'DDO170': 1061,
    'DDO064': 519,
    'DDO161': 549,
    'NGC0055': 129,
    'NGC0247': 156,
    'NGC0300': 144,
    'NGC2366': 80,
    'NGC2403': 131,
    'NGC2915': 468,
    'NGC2976': 3,
    'NGC3109': 403,
    'NGC3741': 229,
    'NGC4068': 210,
    'NGC5055': 504,
    'NGC6946': 48,
    'NGC7793': 227,
    'NGC7331': 816,
    'NGC2841': 638,
    'NGC3198': 663,
    'NGC4559': 816,
    'NGC6503': 60,
    'NGC0891': 528,
    'NGC2683': 411,
    'NGC2903': 556,
    'NGC3521': 805,
    'NGC5907': 667,
    'NGC5985': 2517,
    'NGC5033': 876,
    'IC2574': 57,
    'IC4202': 8237,

    # UMa cluster members (~950-1100 km/s typically)
    'NGC3726': 866,
    'NGC3769': 737,
    'NGC3877': 895,
    'NGC3893': 967,
    'NGC3917': 965,
    'NGC3949': 800,
    'NGC3953': 1052,
    'NGC3972': 852,
    'NGC3992': 1048,
    'NGC4010': 902,
    'NGC4013': 831,
    'NGC4051': 700,
    'NGC4085': 746,
    'NGC4088': 757,
    'NGC4100': 1074,
    'NGC4138': 894,
    'NGC4157': 774,
    'NGC4183': 930,
    'NGC4217': 1027,
    'UGC06399': 889,
    'UGC06446': 646,
    'UGC06667': 935,
    'UGC06786': 772,
    'UGC06787': 769,
    'UGC06818': 811,
    'UGC06917': 911,
    'UGC06923': 1077,
    'UGC06930': 777,
    'UGC06973': 693,
    'UGC06983': 1081,
    'UGC07089': 770,

    # Field galaxies
    'NGC0024': 554,
    'NGC0100': 849,
    'NGC0289': 1629,
    'NGC0801': 5807,
    'NGC1003': 624,
    'NGC1090': 2758,
    'NGC1705': 633,
    'NGC2955': 5816,
    'NGC2998': 4769,
    'NGC5005': 946,
    'NGC5371': 2558,
    'NGC5585': 305,
    'NGC6015': 821,
    'NGC6195': 9590,
    'NGC6674': 3558,
    'NGC6789': -141,
    'NGC7814': 1050,

    # UGC galaxies
    'UGC00128': 4539,
    'UGC00191': 7008,
    'UGC00634': 7461,
    'UGC00731': 647,
    'UGC00891': 558,
    'UGC01230': 5266,
    'UGC01281': 156,
    'UGC01547': 3835,
    'UGC02023': 621,
    'UGC02259': 579,
    'UGC02455': 2764,
    'UGC02487': 4621,
    'UGC02885': 5764,
    'UGC02916': 5028,
    'UGC02953': 1963,
    'UGC03205': 5103,
    'UGC03521': 6419,
    'UGC03546': 4979,
    'UGC03580': 1155,
    'UGC04278': 575,
    'UGC04305': 142,    # Holmberg II
    'UGC04325': 508,
    'UGC04399': 732,
    'UGC04483': 156,
    'UGC04499': 693,
    'UGC05005': 2384,
    'UGC05253': 1896,
    'UGC05414': 453,
    'UGC05716': 578,
    'UGC05721': 536,
    'UGC05750': 4148,
    'UGC05764': 329,
    'UGC05829': 624,
    'UGC05918': 340,
    'UGC05986': 1002,
    'UGC05999': 1783,
    'UGC06614': 6351,
    'UGC06628': 6157,
    'UGC07125': 210,    # = NGC4068
    'UGC07151': 207,
    'UGC07232': 293,
    'UGC07261': 541,
    'UGC07323': 556,
    'UGC07399': 502,
    'UGC07524': 321,
    'UGC07559': 218,
    'UGC07577': 191,
    'UGC07603': 641,
    'UGC07608': 530,
    'UGC07690': 420,
    'UGC07866': 241,
    'UGC08286': 203,
    'UGC08490': 218,
    'UGC08550': 365,
    'UGC08699': 5570,
    'UGC08837': 144,
    'UGC09037': 6005,
    'UGC09133': 5963,
    'UGC09992': 1215,
    'UGC10310': 715,
    'UGC11455': 6324,
    'UGC11557': 1206,
    'UGC11616': 4704,
    'UGC11648': 6458,
    'UGC11748': 48,     # = NGC6946
    'UGC11819': 4700,
    'UGC11820': 7159,
    'UGC12506': 6959,
    'UGC12632': 418,
    'UGC12732': 782,
    'UGCA442': 267,
    'UGCA444': 87,
    'UGCA281': 284,     # = D512-2
    'NGC4214': 291,
    'NGC4389': 718,
    'PGC51017': 2139,
    'UGC11914': 893,

    # ESO galaxies
    'ESO079-G014': 5230,
    'ESO116-G012': 1117,
    'ESO444-G084': 589,
    'ESO563-G021': 3426,

    # D-series
    'D512-2': 284,
    'D564-8': 482,
    'D631-7': 400,
    'KK98-251': 131,

    # F-series (LSB) — from NED "LSBC" queries
    'F561-1': 4807,
    'F563-1': 2920,
    'F563-V1': 3019,
    'F563-V2': 3262,
    'F565-V2': 4330,
    'F567-2': 5667,
    'F568-1': 5093,
    'F568-3': 5134,
    'F568-V1': 5043,
    'F571-8': 5172,
    'F571-V1': 5025,
    'F574-1': 6653,
    'F574-2': 7140,
    'F579-V1': 5731,
    'F583-1': 2266,
    'F583-4': 3497,
}


# ============================================================
# STEP 3: Query EDD CF4 Calculator
# ============================================================
def query_cf4_batch(galaxies):
    """
    Query the EDD Cosmicflows-4 calculator for all galaxies.

    Args:
        galaxies: list of dicts with keys: name, ra, dec, vel
    Returns:
        dict mapping galaxy name to CF4 result
    """
    print("\n--- Step 3: Querying EDD CF4 Calculator ---")

    if not galaxies:
        print("  No galaxies to query!")
        return {}

    # Filter out galaxies with invalid velocities
    valid = [g for g in galaxies if g['vel'] is not None and g['vel'] > 0]
    invalid = [g for g in galaxies if g['vel'] is None or g['vel'] <= 0]

    if invalid:
        print(f"  Skipping {len(invalid)} galaxies with no/negative velocity:")
        for g in invalid:
            print(f"    {g['name']}: vel={g.get('vel')}")

    print(f"  Querying {len(valid)} galaxies...")

    # Build batch payload (max 500 per request, we have <175)
    payload = {"galaxies": []}
    for g in valid:
        payload["galaxies"].append({
            "coordinate": [g['ra'], g['dec']],
            "system": "equatorial",
            "parameter": "velocity",
            "value": g['vel']
        })

    # Query the API
    api_url = "https://edd.ifa.hawaii.edu/CF4calculator/api.php"

    data = json.dumps(payload).encode('utf-8')

    # Try GET request (per official API docs), then POST as fallback
    for method in ['GET', 'POST']:
        req = urllib.request.Request(api_url, data=data, method=method)
        req.add_header('Content-Type', 'application/json')
        req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0 (research)')

        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                result_text = resp.read().decode('utf-8', errors='replace')

            result = json.loads(result_text)
            if 'results' in result or result.get('message') == 'Success':
                print(f"  Batch query succeeded via {method}")
                break
        except Exception as e:
            print(f"  {method} batch query failed: {e}")
            result = None
            continue

    if result is None:
        print(f"  ERROR: Both GET and POST failed. Trying individual queries...")
        return query_cf4_individual(valid)

    # Parse results
    cf4_results = {}

    if 'results' in result:
        # Batch response
        results_list = result['results']
        for i, r in enumerate(results_list):
            name = valid[i]['name']
            if r.get('message') == 'Success' and 'observed' in r:
                distances = r['observed'].get('distance', [])
                if distances:
                    # Take the first (and usually only) distance
                    D_cf4 = distances[0]
                    cf4_results[name] = {
                        'D_cf4': D_cf4,
                        'D_cf4_all': distances,
                        'V_input': valid[i]['vel'],
                        'V_observed': r['observed'].get('velocity'),
                        'RA': r.get('RA'),
                        'Dec': r.get('Dec'),
                        'Glon': r.get('Glon'),
                        'Glat': r.get('Glat'),
                        'SGL': r.get('SGL'),
                        'SGB': r.get('SGB'),
                        'status': 'success',
                        'n_solutions': len(distances),
                    }
                else:
                    cf4_results[name] = {
                        'D_cf4': None,
                        'status': 'no_distance',
                        'message': str(r),
                    }
            else:
                cf4_results[name] = {
                    'D_cf4': None,
                    'status': 'error',
                    'message': r.get('message', 'unknown'),
                }

        n_success = sum(1 for r in cf4_results.values() if r.get('D_cf4'))
        print(f"  CF4 results: {n_success} distances obtained from {len(valid)} queries")
    else:
        # Single galaxy response or error
        print(f"  Unexpected response format: {list(result.keys())}")
        if result.get('message') == 'Success':
            # Single result
            name = valid[0]['name']
            distances = result.get('observed', {}).get('distance', [])
            if distances:
                cf4_results[name] = {
                    'D_cf4': distances[0],
                    'D_cf4_all': distances,
                    'status': 'success',
                }

    return cf4_results


def query_cf4_individual(galaxies):
    """Fallback: query CF4 one galaxy at a time."""
    print("  Falling back to individual queries...")

    api_url = "https://edd.ifa.hawaii.edu/CF4calculator/api.php"
    results = {}

    for i, g in enumerate(galaxies):
        payload = {
            "coordinate": [g['ra'], g['dec']],
            "system": "equatorial",
            "parameter": "velocity",
            "value": g['vel']
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(api_url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('User-Agent', 'SPARC-CF4-Pipeline/1.0')

        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                text = resp.read().decode('utf-8', errors='replace')
            r = json.loads(text)

            if r.get('message') == 'Success' and 'observed' in r:
                distances = r['observed'].get('distance', [])
                if distances:
                    results[g['name']] = {
                        'D_cf4': distances[0],
                        'D_cf4_all': distances,
                        'V_input': g['vel'],
                        'status': 'success',
                    }
                    continue
        except Exception as e:
            pass

        results[g['name']] = {'D_cf4': None, 'status': 'error'}

        if (i + 1) % 20 == 0:
            n_ok = sum(1 for r in results.values() if r.get('D_cf4'))
            print(f"    {i+1}/{len(galaxies)}: {n_ok} distances obtained")

        time.sleep(0.3)

    n_ok = sum(1 for r in results.values() if r.get('D_cf4'))
    print(f"  Individual queries complete: {n_ok}/{len(galaxies)} distances")
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("STEP 3: Fetch CF4 Flow Model Distances for SPARC Galaxies")
    print("=" * 80)

    # Load galaxy names from crossmatch
    crossmatch_file = os.path.join(DATA_DIR, 'sparc_pgc_crossmatch.csv')
    galaxy_names = []
    with open(crossmatch_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            galaxy_names.append(row['sparc_name'])
    print(f"\nLoaded {len(galaxy_names)} SPARC galaxy names")

    # Step 1: Get coordinates
    coords = None
    if os.path.exists(COORD_CACHE):
        with open(COORD_CACHE, 'r') as f:
            coords = json.load(f)
        print(f"  Loaded cached coordinates for {len(coords)} galaxies")

    if not coords or len(coords) < 170:
        coords_vizier = fetch_vizier_coordinates()
        if coords_vizier:
            if coords:
                coords.update(coords_vizier)
            else:
                coords = coords_vizier
            # Save cache
            with open(COORD_CACHE, 'w') as f:
                json.dump(coords, f, indent=2)

    if not coords:
        print("ERROR: Could not obtain coordinates. Aborting.")
        sys.exit(1)

    # Check coverage
    missing_coords = [n for n in galaxy_names if n not in coords]
    if missing_coords:
        print(f"\n  WARNING: Missing coordinates for {len(missing_coords)} galaxies:")
        for n in missing_coords:
            print(f"    {n}")

    # Step 2: Get velocities
    # Start with known velocities
    velocities = dict(KNOWN_VELOCITIES)

    # Check which galaxies still need velocities
    need_vel = [n for n in galaxy_names if n not in velocities]

    if need_vel:
        print(f"\n  {len(need_vel)} galaxies need velocity lookup:")
        for n in need_vel:
            print(f"    {n}")

        # Try SIMBAD first
        simbad_vels = fetch_simbad_velocities(need_vel)
        velocities.update(simbad_vels)

        # NED fallback for remaining
        still_need = [n for n in need_vel if n not in velocities]
        if still_need:
            ned_vels = fetch_ned_velocities_batch(still_need, coords)
            velocities.update(ned_vels)

    # Summary of velocity coverage
    have_vel = sum(1 for n in galaxy_names if n in velocities)
    print(f"\n  Velocity coverage: {have_vel}/{len(galaxy_names)}")

    no_vel = [n for n in galaxy_names if n not in velocities]
    if no_vel:
        print(f"  Missing velocities for:")
        for n in no_vel:
            print(f"    {n}")

    # Step 3: Build query list and call CF4 API
    query_list = []
    for name in galaxy_names:
        if name in coords and name in velocities:
            query_list.append({
                'name': name,
                'ra': coords[name]['ra'],
                'dec': coords[name]['dec'],
                'vel': velocities[name],
            })

    print(f"\n  Prepared {len(query_list)} galaxies for CF4 query")

    # Query CF4
    cf4_results = query_cf4_batch(query_list)

    # Merge with any existing cache
    existing_cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            existing_cache = json.load(f)

    # Update cache with new results
    for name, result in cf4_results.items():
        existing_cache[name] = result

    # Also add coordinate and velocity info for reference
    for name in galaxy_names:
        if name not in existing_cache:
            existing_cache[name] = {
                'D_cf4': None,
                'status': 'not_queried'
            }
        if name in coords:
            existing_cache[name]['ra'] = coords[name]['ra']
            existing_cache[name]['dec'] = coords[name]['dec']
        if name in velocities:
            existing_cache[name]['V_helio'] = velocities[name]

    # Save updated cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(existing_cache, f, indent=2, default=str)

    # Final summary
    n_cf4 = sum(1 for v in existing_cache.values() if v.get('D_cf4') is not None)

    print(f"\n{'=' * 80}")
    print(f"RESULTS:")
    print(f"  Total SPARC galaxies: {len(galaxy_names)}")
    print(f"  Have coordinates: {len(galaxy_names) - len(missing_coords)}")
    print(f"  Have velocities: {have_vel}")
    print(f"  CF4 distances obtained: {n_cf4}")
    print(f"\n  Cache saved to: {CACHE_FILE}")

    # Show distance comparison for a few galaxies
    print(f"\n--- Sample CF4 vs SPARC Distance Comparison ---")
    print(f"{'Galaxy':<15} {'V_helio':>8} {'D_CF4':>8} {'D_SPARC':>8} {'Ratio':>8}")
    print(f"{'-'*55}")

    # Load SPARC distances for comparison
    sparc_dists = {}
    with open(crossmatch_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sparc_dists[row['sparc_name']] = float(row['D'])

    count = 0
    for name in sorted(galaxy_names):
        if name in existing_cache and existing_cache[name].get('D_cf4'):
            D_cf4 = existing_cache[name]['D_cf4']
            D_sparc = sparc_dists.get(name, 0)
            ratio = D_cf4 / D_sparc if D_sparc > 0 else 0
            vel = velocities.get(name, 0)
            print(f"  {name:<15} {vel:>8.0f} {D_cf4:>8.2f} {D_sparc:>8.2f} {ratio:>8.3f}")
            count += 1
            if count >= 20:
                print(f"  ... ({n_cf4 - 20} more)")
                break

    # Distance statistics
    if n_cf4 > 0:
        ratios = []
        for name in galaxy_names:
            if (name in existing_cache and existing_cache[name].get('D_cf4') and
                name in sparc_dists and sparc_dists[name] > 0):
                ratio = existing_cache[name]['D_cf4'] / sparc_dists[name]
                ratios.append(ratio)

        if ratios:
            import numpy as np
            ratios = np.array(ratios)
            print(f"\n  CF4/SPARC distance ratio statistics (N={len(ratios)}):")
            print(f"    Mean:   {np.mean(ratios):.3f}")
            print(f"    Median: {np.median(ratios):.3f}")
            print(f"    Std:    {np.std(ratios):.3f}")
            print(f"    Min:    {np.min(ratios):.3f} ({galaxy_names[np.argmin(ratios)]})")
            print(f"    Max:    {np.max(ratios):.3f}")

    print(f"\n{'=' * 80}")
    print("✓ Step 3 complete — now re-run 02_cf4_rar_pipeline.py to compare results!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
