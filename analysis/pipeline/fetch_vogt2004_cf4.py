#!/usr/bin/env python3
"""
Fetch CF4 flow-model distances for Vogt+2004 cluster galaxies.

Reads Vogt+2004 VizieR TSV, parses coordinates and CMB velocities,
queries the EDD CF4 calculator API, and caches results.

Russell Licht -- Primordial Fluid DM Project
Feb 2026
"""

import json
import os
import sys
import time
import urllib.request
import ssl
import re
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'hi_surveys', 'vogt2004_galaxies.tsv')
CACHE_FILE = os.path.join(DATA_DIR, 'vogt2004_cf4_cache.json')

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
def parse_vogt2004_tsv(filepath):
    """
    Parse the Vogt+2004 VizieR TSV file.

    Returns list of dicts with keys: name, ra, dec, vel, cluster
    """
    print(f"Reading: {filepath}")

    galaxies = []
    header_cols = None
    data_started = False

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
                if 'Name' in raw and '_RA.icrs' in raw and '_DE.icrs' in raw:
                    header_cols = [c.strip() for c in raw.split('\t')]
                    data_started = True
                    continue
                # Skip unit line or separator line
                continue

            # Skip unit / separator lines that come right after the header
            if raw.strip().startswith('"') or raw.strip().startswith('-'):
                continue
            # The unit line has entries like "h:m:s" or "km/s"
            if '"h:m:s"' in raw or '"d:m:s"' in raw or '"km/s"' in raw:
                continue

            # Data row
            parts = raw.split('\t')
            if len(parts) < len(header_cols):
                continue

            row = {}
            for i, col in enumerate(header_cols):
                row[col] = parts[i].strip() if i < len(parts) else ''

            name = row.get('Name', '').strip()
            ra_str = row.get('_RA.icrs', '').strip()
            dec_str = row.get('_DE.icrs', '').strip()
            vel_str = row.get('RV', '').strip()
            cluster = row.get('Clust', '').strip()

            if not name:
                continue

            # Parse coordinates (ICRS J2000)
            ra_deg, dec_deg = parse_hms_to_deg(ra_str, dec_str)

            # Parse CMB velocity
            vel = None
            if vel_str:
                try:
                    vel = float(vel_str)
                except ValueError:
                    vel = None

            galaxies.append({
                'name': name,
                'ra': ra_deg,
                'dec': dec_deg,
                'vel': vel,
                'cluster': cluster,
                'ra_str': ra_str,
                'dec_str': dec_str,
            })

    print(f"  Parsed {len(galaxies)} galaxies from TSV")
    return galaxies


# ============================================================
# Query CF4 API
# ============================================================
def query_cf4_batch(galaxies_batch, batch_num, n_batches):
    """
    Query the EDD CF4 calculator for a batch of galaxies.

    Args:
        galaxies_batch: list of dicts with keys: name, ra, dec, vel
        batch_num: current batch number (for logging)
        n_batches: total number of batches

    Returns:
        dict mapping galaxy name to CF4 result dict
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
        req.add_header('User-Agent', 'Vogt2004-CF4-Pipeline/1.0 (research)')

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
        # Return error entries for all galaxies in this batch
        errors = {}
        for g in galaxies_batch:
            errors[g['name']] = {
                'D_cf4': None,
                'V_input': g['vel'],
                'ra': g['ra'],
                'dec': g['dec'],
                'cluster': g.get('cluster', ''),
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
                        'cluster': g.get('cluster', ''),
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
                        'cluster': g.get('cluster', ''),
                        'status': 'no_distance',
                        'message': str(r),
                    }
            else:
                cf4_results[name] = {
                    'D_cf4': None,
                    'V_input': g['vel'],
                    'ra': g['ra'],
                    'dec': g['dec'],
                    'cluster': g.get('cluster', ''),
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
                'cluster': g.get('cluster', ''),
                'status': 'success',
                'n_solutions': len(distances),
            }
        else:
            cf4_results[g['name']] = {
                'D_cf4': None,
                'V_input': g['vel'],
                'ra': g['ra'],
                'dec': g['dec'],
                'cluster': g.get('cluster', ''),
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
                'cluster': g.get('cluster', ''),
                'status': 'error',
                'message': f'Unexpected response: {str(result)[:200]}',
            }

    n_ok = sum(1 for r in cf4_results.values() if r.get('D_cf4') is not None)
    print(f"    Got {n_ok}/{len(galaxies_batch)} distances")

    return cf4_results


# ============================================================
# Save cache incrementally
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
    print("Fetch CF4 Flow-Model Distances for Vogt+2004 Cluster Galaxies")
    print("=" * 65)

    # Step 1: Read and parse the TSV
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    galaxies = parse_vogt2004_tsv(INPUT_FILE)
    if not galaxies:
        print("ERROR: No galaxies parsed!")
        sys.exit(1)

    # Step 2: Filter to galaxies with valid coordinates and velocities
    valid = []
    skipped_coords = 0
    skipped_vel = 0

    for g in galaxies:
        if g['ra'] is None or g['dec'] is None:
            skipped_coords += 1
            continue
        if g['vel'] is None or g['vel'] <= 0:
            skipped_vel += 1
            continue
        valid.append(g)

    print(f"\n  Valid galaxies for CF4 query: {len(valid)}")
    if skipped_coords:
        print(f"  Skipped (missing coordinates): {skipped_coords}")
    if skipped_vel:
        print(f"  Skipped (missing/invalid velocity): {skipped_vel}")

    # Show cluster distribution
    cluster_counts = {}
    for g in valid:
        c = g.get('cluster', 'unknown')
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    print(f"\n  Cluster distribution:")
    for c in sorted(cluster_counts.keys()):
        print(f"    {c:12s}: {cluster_counts[c]:3d} galaxies")

    # Step 3: Load existing cache (to resume if interrupted)
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(f"\n  Loaded existing cache with {len(cache)} entries")

    # Filter out galaxies already in cache with success status
    to_query = [g for g in valid if g['name'] not in cache or cache[g['name']].get('status') != 'success']
    already_cached = len(valid) - len(to_query)
    if already_cached:
        print(f"  Already cached (success): {already_cached}")
    print(f"  Galaxies to query: {len(to_query)}")

    if not to_query:
        print("\n  All galaxies already cached. Nothing to do.")
    else:
        # Step 4: Query CF4 in batches
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
                time.sleep(0.5)

    # Step 5: Summary statistics
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

    # Per-cluster summary
    print(f"\n  Per-cluster CF4 distance summary:")
    print(f"  {'Cluster':<12s} {'N_total':>8s} {'N_cf4':>6s} {'D_mean':>8s} {'D_median':>9s}")
    print(f"  {'-'*12} {'-'*8} {'-'*6} {'-'*8} {'-'*9}")

    cluster_dists = {}
    for name, r in cache.items():
        c = r.get('cluster', 'unknown')
        if c not in cluster_dists:
            cluster_dists[c] = {'total': 0, 'dists': []}
        cluster_dists[c]['total'] += 1
        if r.get('D_cf4') is not None:
            cluster_dists[c]['dists'].append(r['D_cf4'])

    for c in sorted(cluster_dists.keys()):
        info = cluster_dists[c]
        n = len(info['dists'])
        if n > 0:
            d_mean = sum(info['dists']) / n
            d_sorted = sorted(info['dists'])
            d_median = d_sorted[n // 2]
            print(f"  {c:<12s} {info['total']:>8d} {n:>6d} {d_mean:>8.1f} {d_median:>9.1f}")
        else:
            print(f"  {c:<12s} {info['total']:>8d} {n:>6d} {'---':>8s} {'---':>9s}")

    # Show first few results as a sample
    print(f"\n  Sample results (first 10):")
    print(f"  {'Name':<20s} {'Cluster':<8s} {'V_cmb':>7s} {'D_cf4':>8s} {'Status':<12s}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*8} {'-'*12}")
    for i, (name, r) in enumerate(cache.items()):
        if i >= 10:
            break
        d_str = f"{r['D_cf4']:.1f}" if r.get('D_cf4') is not None else '---'
        v_str = f"{r.get('V_input', 0):.0f}" if r.get('V_input') else '---'
        print(f"  {name:<20s} {r.get('cluster',''):<8s} {v_str:>7s} {d_str:>8s} {r.get('status',''):<12s}")

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
