#!/usr/bin/env python3
"""
Fetch CF4 flow-model distances for Catinella+2005 galaxies.

Reads Catinella+2005 VizieR TSV (polyex rotation curve catalog),
parses coordinates and heliocentric velocities, queries the EDD CF4
calculator API, and caches results.

Russell Licht -- Primordial Fluid DM Project
Feb 2026
"""

import json
import math
import os
import ssl
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'hi_surveys', 'catinella2005_polyex.tsv')
CACHE_FILE = os.path.join(DATA_DIR, 'catinella2005_cf4_cache.json')

CF4_API_URL = "https://edd.ifa.hawaii.edu/CF4calculator/api.php"

# SSL context -- some astronomy services have expired certs
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


# ============================================================
# Coordinate parsing
# ============================================================
def parse_hms_to_deg(ra_str, dec_str):
    """
    Convert RA 'hh mm ss.s' and Dec '+dd mm ss.s' to decimal degrees.
    Returns (ra_deg, dec_deg) or (None, None) on failure.
    """
    try:
        ra_parts = ra_str.strip().split()
        if len(ra_parts) != 3:
            return None, None
        ra_h = float(ra_parts[0])
        ra_m = float(ra_parts[1])
        ra_s = float(ra_parts[2])
        ra_deg = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0

        dec_parts = dec_str.strip().split()
        if len(dec_parts) != 3:
            return None, None
        dec_sign = -1.0 if dec_parts[0].startswith('-') else 1.0
        dec_d = abs(float(dec_parts[0]))
        dec_m = float(dec_parts[1])
        dec_s = float(dec_parts[2])
        dec_deg = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)

        return ra_deg, dec_deg
    except (ValueError, IndexError):
        return None, None


# ============================================================
# Parse the VizieR TSV file
# ============================================================
def parse_catinella2005_tsv(filepath):
    """
    Parse the Catinella+2005 VizieR TSV file.

    Returns list of dicts with keys: name, seq, ra, dec, vel, qual
    Only keeps rows with Qual == 1 and valid V80 > 0.
    """
    print(f"Reading: {filepath}")

    galaxies = []
    header_cols = None
    data_started = False
    n_total_data = 0
    n_skipped_qual = 0
    n_skipped_vel = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            raw = line.rstrip('\n')

            # Skip comment lines
            if raw.startswith('#'):
                continue

            # Skip blank lines
            if not raw.strip():
                continue

            # Detect the header row (contains column names, tab-separated)
            if not data_started:
                if 'OName' in raw and 'RAJ2000' in raw and 'DEJ2000' in raw:
                    header_cols = [c.strip() for c in raw.split('\t')]
                    data_started = True
                    continue
                continue

            # Skip unit / separator lines that come right after the header
            if raw.strip().startswith('-'):
                continue
            if '"h:m:s"' in raw or '"d:m:s"' in raw or '"km/s"' in raw:
                continue

            # Data row
            parts = raw.split('\t')
            if header_cols is None or len(parts) < len(header_cols):
                continue

            row = {}
            for i, col in enumerate(header_cols):
                row[col] = parts[i].strip() if i < len(parts) else ''

            n_total_data += 1

            # Extract fields
            seq = row.get('Seq', '').strip()
            oname = row.get('OName', '').strip()
            ra_str = row.get('RAJ2000', '').strip()
            dec_str = row.get('DEJ2000', '').strip()
            qual_str = row.get('Qual', '').strip()
            v80_str = row.get('V80', '').strip()

            # Build a name: prefer OName, fall back to Seq
            name = oname if oname else f"Seq{seq}"

            # Quality filter: only Qual=1
            try:
                qual = int(qual_str)
            except (ValueError, TypeError):
                qual = -1
            if qual != 1:
                n_skipped_qual += 1
                continue

            # Parse heliocentric velocity V80
            vel = None
            if v80_str:
                try:
                    vel = float(v80_str)
                except ValueError:
                    vel = None
            if vel is None or vel <= 0:
                n_skipped_vel += 1
                continue

            # Parse coordinates
            ra_deg, dec_deg = parse_hms_to_deg(ra_str, dec_str)
            if ra_deg is None or dec_deg is None:
                continue

            galaxies.append({
                'name': name,
                'seq': seq,
                'ra': ra_deg,
                'dec': dec_deg,
                'vel': vel,
                'qual': qual,
                'ra_str': ra_str,
                'dec_str': dec_str,
            })

    print(f"  Total data rows: {n_total_data}")
    print(f"  Skipped (Qual != 1): {n_skipped_qual}")
    print(f"  Skipped (V80 missing/invalid): {n_skipped_vel}")
    print(f"  Valid Qual=1 galaxies with V80 > 0: {len(galaxies)}")
    return galaxies


# ============================================================
# Query CF4 API
# ============================================================
def query_cf4_batch(galaxies_batch, batch_num, n_batches):
    """
    Query the EDD CF4 calculator for a batch of galaxies.

    Returns dict mapping galaxy name to CF4 result dict.
    """
    payload = {"galaxies": []}
    for g in galaxies_batch:
        payload["galaxies"].append({
            "coordinate": [g['ra'], g['dec']],
            "system": "equatorial",
            "parameter": "velocity",
            "value": g['vel']
        })

    data = json.dumps(payload).encode('utf-8')

    print(f"  Batch {batch_num}/{n_batches}: querying {len(galaxies_batch)} galaxies...")

    result = None
    for method in ['GET', 'POST']:
        req = urllib.request.Request(CF4_API_URL, data=data, method=method)
        req.add_header('Content-Type', 'application/json')
        req.add_header('User-Agent', 'Catinella2005-CF4-Pipeline/1.0 (research)')

        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                result_text = resp.read().decode('utf-8', errors='replace')
            result = json.loads(result_text)
            if 'results' in result or result.get('message') == 'Success':
                print(f"    Succeeded via {method}")
                break
        except Exception as e:
            print(f"    {method} failed: {e}")
            result = None
            continue

    if result is None:
        print(f"    ERROR: Both GET and POST failed for batch {batch_num}")
        errors = {}
        for g in galaxies_batch:
            errors[g['name']] = {
                'D_cf4': None,
                'V_input': g['vel'],
                'ra': g['ra'],
                'dec': g['dec'],
                'seq': g.get('seq', ''),
                'status': 'error',
                'message': 'API request failed',
            }
        return errors

    # Parse results
    cf4_results = {}

    if 'results' in result:
        results_list = result['results']
        for i, r in enumerate(results_list):
            if i >= len(galaxies_batch):
                break
            g = galaxies_batch[i]
            name = g['name']

            if r.get('message') == 'Success' and 'observed' in r:
                distances = r['observed'].get('distance', [])
                if distances:
                    cf4_results[name] = {
                        'D_cf4': distances[0],
                        'D_cf4_all': distances,
                        'V_input': g['vel'],
                        'V_observed': r['observed'].get('velocity'),
                        'ra': g['ra'],
                        'dec': g['dec'],
                        'seq': g.get('seq', ''),
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
                        'V_input': g['vel'],
                        'ra': g['ra'],
                        'dec': g['dec'],
                        'seq': g.get('seq', ''),
                        'status': 'no_distance',
                        'message': str(r),
                    }
            else:
                cf4_results[name] = {
                    'D_cf4': None,
                    'V_input': g['vel'],
                    'ra': g['ra'],
                    'dec': g['dec'],
                    'seq': g.get('seq', ''),
                    'status': 'error',
                    'message': r.get('message', 'unknown'),
                }
    elif result.get('message') == 'Success' and len(galaxies_batch) == 1:
        # Single galaxy response
        g = galaxies_batch[0]
        distances = result.get('observed', {}).get('distance', [])
        if distances:
            cf4_results[g['name']] = {
                'D_cf4': distances[0],
                'D_cf4_all': distances,
                'V_input': g['vel'],
                'V_observed': result.get('observed', {}).get('velocity'),
                'ra': g['ra'],
                'dec': g['dec'],
                'seq': g.get('seq', ''),
                'status': 'success',
                'n_solutions': len(distances),
            }
        else:
            cf4_results[g['name']] = {
                'D_cf4': None,
                'V_input': g['vel'],
                'ra': g['ra'],
                'dec': g['dec'],
                'seq': g.get('seq', ''),
                'status': 'no_distance',
            }
    else:
        print(f"    Unexpected response format: {list(result.keys())}")
        for g in galaxies_batch:
            cf4_results[g['name']] = {
                'D_cf4': None,
                'V_input': g['vel'],
                'ra': g['ra'],
                'dec': g['dec'],
                'seq': g.get('seq', ''),
                'status': 'error',
                'message': f'Unexpected response: {str(result)[:200]}',
            }

    n_ok = sum(1 for r in cf4_results.values() if r.get('D_cf4') is not None)
    print(f"    Got {n_ok}/{len(galaxies_batch)} distances")

    return cf4_results


# ============================================================
# Save cache
# ============================================================
def save_cache(cache, filepath):
    """Save the cache to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("Fetch CF4 Flow-Model Distances for Catinella+2005 Galaxies")
    print("=" * 65)

    # Step 1: Read and parse the TSV
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    galaxies = parse_catinella2005_tsv(INPUT_FILE)
    if not galaxies:
        print("ERROR: No galaxies parsed!")
        sys.exit(1)

    print(f"\n  Valid galaxies for CF4 query: {len(galaxies)}")

    # Step 2: Load existing cache (to resume if interrupted)
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(f"\n  Loaded existing cache with {len(cache)} entries")

    # Filter out galaxies already in cache with success status
    to_query = [g for g in galaxies if g['name'] not in cache
                or cache[g['name']].get('status') != 'success']
    already_cached = len(galaxies) - len(to_query)
    if already_cached:
        print(f"  Already cached (success): {already_cached}")
    print(f"  Galaxies to query: {len(to_query)}")

    if not to_query:
        print("\n  All galaxies already cached. Nothing to do.")
    else:
        # Step 3: Query CF4 in batches of 100
        BATCH_SIZE = 100
        n_batches = math.ceil(len(to_query) / BATCH_SIZE)

        print(f"\n--- Querying CF4 API in {n_batches} batch(es) of up to {BATCH_SIZE} ---")

        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(to_query))
            batch = to_query[start:end]

            results = query_cf4_batch(batch, batch_idx + 1, n_batches)
            cache.update(results)

            # Save intermediate results after each batch
            save_cache(cache, CACHE_FILE)
            print(f"    Saved intermediate cache ({len(cache)} total entries)")

            # Rate limit between batches
            if batch_idx < n_batches - 1:
                time.sleep(1.0)

    # Step 4: Summary statistics
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    n_total = len(cache)
    n_success = sum(1 for r in cache.values() if r.get('status') == 'success')
    n_error = sum(1 for r in cache.values() if r.get('status') == 'error')
    n_nodist = sum(1 for r in cache.values() if r.get('status') == 'no_distance')

    print(f"  Total entries in cache: {n_total}")
    print(f"  Successful CF4 distances: {n_success}")
    print(f"  Errors: {n_error}")
    print(f"  No distance returned: {n_nodist}")

    # Distance statistics
    distances = [r['D_cf4'] for r in cache.values() if r.get('D_cf4') is not None]
    if distances:
        print(f"\n  Distance statistics (N={len(distances)}):")
        print(f"    Min:    {min(distances):.1f} Mpc")
        print(f"    Max:    {max(distances):.1f} Mpc")
        print(f"    Mean:   {sum(distances)/len(distances):.1f} Mpc")
        sorted_d = sorted(distances)
        median = sorted_d[len(sorted_d) // 2]
        print(f"    Median: {median:.1f} Mpc")

    # Show first few results as a sample
    print(f"\n  Sample results (first 10):")
    print(f"  {'Name':<20s} {'Seq':<8s} {'V80':>7s} {'D_cf4':>8s} {'Status':<12s}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*8} {'-'*12}")
    for i, (name, r) in enumerate(cache.items()):
        if i >= 10:
            break
        d_str = f"{r['D_cf4']:.1f}" if r.get('D_cf4') is not None else '---'
        v_str = f"{r.get('V_input', 0):.0f}" if r.get('V_input') else '---'
        print(f"  {name:<20s} {r.get('seq',''):<8s} {v_str:>7s} {d_str:>8s} {r.get('status',''):<12s}")

    # Multi-solution analysis
    multi = [(name, r) for name, r in cache.items() if r.get('n_solutions', 0) > 1]
    if multi:
        print(f"\n  Galaxies with multiple distance solutions: {len(multi)}")
        for name, r in multi[:5]:
            print(f"    {name}: {r.get('D_cf4_all', [])}")

    print(f"\n  Cache saved to: {CACHE_FILE}")
    print("  Done.")


if __name__ == '__main__':
    main()
