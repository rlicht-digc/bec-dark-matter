#!/usr/bin/env python3
"""
ALFALFA × Yang+2007 BTFR Environmental Scatter Test
=====================================================

THE IRONCLAD REPLICATION TEST.

Cross-match ALFALFA α.100 (31,500 HI-selected galaxies) with Yang DR7 group
catalog (639,359 galaxies with halo masses). Compute BTFR scatter as a function
of velocity width (acceleration proxy) and host halo mass (environment).

BEC prediction: BTFR scatter should be uniform across environments at LOW
velocity widths (deep condensate, low acceleration) and potentially divergent
at HIGH velocity widths (baryon-dominated, high acceleration). The transition
should occur at a W50 corresponding to g† = 1.2×10⁻¹⁰ m/s².

This is different data (ALFALFA), different observable (BTFR not RAR),
different environment classification (Yang halo masses not binary), and
thousands of galaxies instead of 131.
"""

import os
import json
import numpy as np
from scipy.stats import levene
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921


# ================================================================
# PARSE ALFALFA α.100
# ================================================================
def parse_sexagesimal_ra(s):
    """Convert 'HH MM SS.S' to degrees."""
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    try:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return 15.0 * (h + m / 60.0 + sec / 3600.0)
    except ValueError:
        return None


def parse_sexagesimal_dec(s):
    """Convert '+DD MM SS' to degrees."""
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    try:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        sign = -1 if d < 0 or s.strip().startswith('-') else 1
        return sign * (abs(d) + m / 60.0 + sec / 3600.0)
    except ValueError:
        return None


def load_alfalfa():
    """Load ALFALFA α.100 catalog from VizieR TSV."""
    filepath = os.path.join(DATA_DIR, 'alfalfa', 'alfalfa_alpha100_haynes2018.tsv')

    galaxies = []
    header_found = False
    col_names = None

    with open(filepath, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                continue
            # Skip unit line and separator
            if line.strip().startswith('"') or line.strip().startswith('---'):
                continue

            parts = line.strip().split('\t')

            # Find the header line with column names
            if not header_found:
                if parts[0].strip() == 'AGC':
                    col_names = [p.strip() for p in parts]
                    header_found = True
                    continue
                continue

            # Skip unit and separator lines after header
            if len(parts) < 10 or parts[0].strip().startswith('"') or parts[0].strip().startswith('-'):
                continue

            try:
                agc = parts[0].strip()
                if not agc or agc.startswith('-'):
                    continue

                # RA/Dec (HI centroid)
                ra_str = parts[2].strip() if len(parts) > 2 else ''
                dec_str = parts[3].strip() if len(parts) > 3 else ''

                ra = parse_sexagesimal_ra(ra_str)
                dec = parse_sexagesimal_dec(dec_str)
                if ra is None or dec is None:
                    continue

                # Key quantities
                vhel_str = parts[6].strip() if len(parts) > 6 else ''
                w50_str = parts[7].strip() if len(parts) > 7 else ''
                ew50_str = parts[8].strip() if len(parts) > 8 else ''
                dist_str = parts[14].strip() if len(parts) > 14 else ''
                logmhi_str = parts[16].strip() if len(parts) > 16 else ''
                hi_code_str = parts[18].strip() if len(parts) > 18 else ''

                if not w50_str or not logmhi_str or not dist_str:
                    continue

                vhel = int(vhel_str)
                w50 = int(w50_str)
                ew50 = int(ew50_str) if ew50_str else 0
                dist = float(dist_str)
                logmhi = float(logmhi_str)
                hi_code = int(hi_code_str) if hi_code_str else 1

                # Quality cuts
                if hi_code != 1:  # Only code 1 (reliable detections)
                    continue
                if w50 < 20 or w50 > 600:  # Reasonable velocity widths
                    continue
                if dist < 1.0 or dist > 250:  # Reasonable distances
                    continue
                if logmhi < 6.0:  # Minimum HI mass
                    continue

                galaxies.append({
                    'agc': agc,
                    'ra': ra,
                    'dec': dec,
                    'vhel': vhel,
                    'w50': w50,
                    'ew50': ew50,
                    'dist': dist,
                    'logmhi': logmhi,
                    'logw50': np.log10(w50),
                })

            except (ValueError, IndexError):
                continue

    return galaxies


# ================================================================
# PARSE YANG DR7
# ================================================================
def load_yang_dr7():
    """Load Yang DR7 galaxy catalog, group catalog, and mapping."""
    yang_dir = os.path.join(DATA_DIR, 'yang_catalogs')

    # Galaxy catalog (SDSS7)
    print("  Loading Yang SDSS7 galaxy catalog...")
    gal_file = os.path.join(yang_dir, 'SDSS7')
    yang_ra = []
    yang_dec = []
    yang_z = []
    yang_galid = []

    with open(gal_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                yang_galid.append(int(parts[0]))
                yang_ra.append(float(parts[2]))
                yang_dec.append(float(parts[3]))
                yang_z.append(float(parts[4]))
            except (ValueError, IndexError):
                continue

    yang_ra = np.array(yang_ra)
    yang_dec = np.array(yang_dec)
    yang_z = np.array(yang_z)
    yang_galid = np.array(yang_galid)
    print(f"    {len(yang_galid)} galaxies")

    # Galaxy-to-group mapping (imodelC_1)
    print("  Loading galaxy-to-group mapping...")
    map_file = os.path.join(yang_dir, 'imodelC_1')
    gal_to_grp = {}
    with open(map_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                gal_id = int(parts[0])
                grp_id = int(parts[2])
                brightest = int(parts[3])
                gal_to_grp[gal_id] = {
                    'grp_id': grp_id,
                    'central': brightest == 1,
                }
            except (ValueError, IndexError):
                continue
    print(f"    {len(gal_to_grp)} mappings")

    # Group catalog (modelC_group)
    print("  Loading group catalog...")
    grp_file = os.path.join(yang_dir, 'modelC_group')
    groups = {}
    with open(grp_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                grp_id = int(parts[0])
                if grp_id == 0:
                    continue  # Skip header rows
                logMh_L = float(parts[6])
                logMh_M = float(parts[7])
                logMh = logMh_L if logMh_L > 0 else logMh_M
                groups[grp_id] = {
                    'logMh': logMh,
                    'z': float(parts[3]),
                }
            except (ValueError, IndexError):
                continue
    print(f"    {len(groups)} groups")

    # Compute group richness
    grp_richness = {}
    for info in gal_to_grp.values():
        gid = info['grp_id']
        if gid > 0:
            grp_richness[gid] = grp_richness.get(gid, 0) + 1

    return yang_ra, yang_dec, yang_z, yang_galid, gal_to_grp, groups, grp_richness


# ================================================================
# CROSS-MATCH
# ================================================================
def crossmatch_alfalfa_yang(alfalfa, yang_ra, yang_dec, yang_z, yang_galid,
                              gal_to_grp, groups, grp_richness,
                              max_sep_arcsec=10.0, max_dz=0.005):
    """Cross-match ALFALFA with Yang DR7 by position and redshift."""

    max_sep_deg = max_sep_arcsec / 3600.0
    c = 299792.458  # km/s

    # Build simple RA index for Yang
    print("  Building spatial index...")
    ra_step = 0.5  # degrees
    ra_bins = {}
    for i in range(len(yang_ra)):
        ra_bin = int(yang_ra[i] / ra_step)
        if ra_bin not in ra_bins:
            ra_bins[ra_bin] = []
        ra_bins[ra_bin].append(i)

    matched = []
    n_total = len(alfalfa)

    print(f"  Cross-matching {n_total} ALFALFA galaxies...")

    for gal in alfalfa:
        ra = gal['ra']
        dec = gal['dec']
        z_alf = gal['vhel'] / c

        # Search nearby RA bins
        ra_bin = int(ra / ra_step)
        candidates = []
        for b in [ra_bin - 1, ra_bin, ra_bin + 1]:
            if b in ra_bins:
                candidates.extend(ra_bins[b])

        if not candidates:
            continue

        cand = np.array(candidates)

        # Dec pre-filter
        dec_diff = np.abs(yang_dec[cand] - dec)
        dec_ok = dec_diff < max_sep_deg * 2
        if not np.any(dec_ok):
            continue

        filtered = cand[dec_ok]

        # Angular separation
        ra1, dec1 = np.radians(ra), np.radians(dec)
        ra2, dec2 = np.radians(yang_ra[filtered]), np.radians(yang_dec[filtered])
        cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
        cos_sep = np.clip(cos_sep, -1, 1)
        sep_arcsec = np.degrees(np.arccos(cos_sep)) * 3600

        # Also filter by redshift
        z_yang = yang_z[filtered]
        z_ok = np.abs(z_yang - z_alf) < max_dz

        combined = (sep_arcsec < max_sep_arcsec) & z_ok

        if not np.any(combined):
            # Try position-only match (for galaxies without good z)
            pos_ok = sep_arcsec < max_sep_arcsec
            if not np.any(pos_ok):
                continue
            best = filtered[np.argmin(sep_arcsec)]
        else:
            sub = np.where(combined)[0]
            best = filtered[sub[np.argmin(sep_arcsec[sub])]]

        gal_id = yang_galid[best]

        # Get group info
        if gal_id in gal_to_grp:
            grp_info = gal_to_grp[gal_id]
            grp_id = grp_info['grp_id']

            if grp_id > 0 and grp_id in groups:
                logMh = groups[grp_id]['logMh']
                richness = grp_richness.get(grp_id, 1)
                central = grp_info['central']
            else:
                logMh = 0.0  # singleton
                richness = 1
                central = True
        else:
            logMh = 0.0
            richness = 1
            central = True

        matched.append({
            **gal,
            'yang_galid': int(gal_id),
            'logMh': logMh,
            'richness': richness,
            'central': central,
            'sep_arcsec': float(sep_arcsec[np.argmin(sep_arcsec)]),
        })

    return matched


# ================================================================
# BTFR ANALYSIS
# ================================================================
def compute_btfr(galaxies):
    """Fit BTFR: logMHI = a × log(W50) + b. Return residuals."""
    logw = np.array([g['logw50'] for g in galaxies])
    logm = np.array([g['logmhi'] for g in galaxies])

    # Ordinary least squares fit
    A = np.vstack([logw, np.ones(len(logw))]).T
    result = np.linalg.lstsq(A, logm, rcond=None)
    slope, intercept = result[0]

    residuals = logm - (slope * logw + intercept)

    return slope, intercept, residuals


def numerical_derivative(x, y):
    """Central difference derivative."""
    dy = np.zeros_like(y)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(1, len(y) - 1):
        dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    return dy


def find_zero_crossings(x, y):
    """Find x-values where y crosses zero."""
    crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i+1] < 0:
            x_cross = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
            crossings.append(float(x_cross))
    return crossings


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("ALFALFA × YANG DR7 BTFR ENVIRONMENTAL SCATTER TEST")
print("=" * 72)

# Load ALFALFA
print("\n--- Loading ALFALFA α.100 ---")
alfalfa = load_alfalfa()
print(f"  {len(alfalfa)} galaxies after quality cuts")

# Load Yang
print("\n--- Loading Yang DR7 ---")
yang_ra, yang_dec, yang_z, yang_galid, gal_to_grp, groups, grp_richness = load_yang_dr7()

# Cross-match
print("\n--- Cross-matching ---")
matched = crossmatch_alfalfa_yang(alfalfa, yang_ra, yang_dec, yang_z, yang_galid,
                                    gal_to_grp, groups, grp_richness)
print(f"  {len(matched)} ALFALFA galaxies matched to Yang DR7")

# Classify environment by halo mass
# Use logMh thresholds: field < 12.0 < group < 13.0 < cluster
for g in matched:
    mh = g['logMh']
    if mh <= 0 or mh < 12.0:
        g['env'] = 'field'
    elif mh < 13.0:
        g['env'] = 'group'
    else:
        g['env'] = 'cluster'

    # Binary: isolated vs dense
    if mh <= 0 or mh < 12.5:
        g['env_binary'] = 'field'
    else:
        g['env_binary'] = 'dense'

n_field = sum(1 for g in matched if g['env_binary'] == 'field')
n_dense = sum(1 for g in matched if g['env_binary'] == 'dense')
n_cluster = sum(1 for g in matched if g['env'] == 'cluster')
n_group = sum(1 for g in matched if g['env'] == 'group')
print(f"\n  Environment breakdown:")
print(f"    Field (logMh < 12.5): {n_field}")
print(f"    Dense (logMh ≥ 12.5): {n_dense}")
print(f"      of which group (12.5-13): {n_group}")
print(f"      of which cluster (≥13):   {n_cluster}")

# ================================================================
# BTFR FIT
# ================================================================
print(f"\n{'='*72}")
print("BTFR FIT")
print(f"{'='*72}")

slope, intercept, residuals = compute_btfr(matched)
print(f"  log(M_HI) = {slope:.3f} × log(W50) + {intercept:.3f}")
print(f"  Overall scatter: {np.std(residuals):.4f} dex")
print(f"  N = {len(residuals)}")

# Add residuals to galaxies
for i, g in enumerate(matched):
    g['btfr_res'] = float(residuals[i])

# ================================================================
# W50-BINNED SCATTER BY ENVIRONMENT
# ================================================================
print(f"\n{'='*72}")
print("BTFR SCATTER BY W50 BIN AND ENVIRONMENT")
print(f"{'='*72}")

# W50 bins
w50_edges = [20, 50, 80, 120, 180, 260, 400, 600]
w50_labels = [f"{w50_edges[i]}-{w50_edges[i+1]}" for i in range(len(w50_edges)-1)]

print(f"\n  {'W50 range':>12s} {'N_field':>8s} {'N_dense':>8s} {'σ_field':>8s} {'σ_dense':>8s}"
      f" {'Δσ(f-d)':>8s} {'Lev_p':>10s}")
print(f"  {'-'*72}")

bin_results = []
for i in range(len(w50_edges) - 1):
    lo, hi = w50_edges[i], w50_edges[i + 1]

    field_res = np.array([g['btfr_res'] for g in matched
                           if g['env_binary'] == 'field' and lo <= g['w50'] < hi])
    dense_res = np.array([g['btfr_res'] for g in matched
                           if g['env_binary'] == 'dense' and lo <= g['w50'] < hi])

    if len(field_res) < 10 or len(dense_res) < 10:
        print(f"  {w50_labels[i]:>12s} {len(field_res):8d} {len(dense_res):8d}      ---      ---      ---        ---")
        continue

    sf = np.std(field_res)
    sd = np.std(dense_res)
    delta = sf - sd
    stat_L, p_L = levene(field_res, dense_res)

    sig = '***' if p_L < 0.001 else '**' if p_L < 0.01 else '*' if p_L < 0.05 else ''

    bin_results.append({
        'w50_lo': lo, 'w50_hi': hi,
        'label': w50_labels[i],
        'log_w50_center': np.log10(np.sqrt(lo * hi)),
        'n_field': len(field_res), 'n_dense': len(dense_res),
        'sigma_field': round(sf, 5), 'sigma_dense': round(sd, 5),
        'delta_sigma': round(delta, 5),
        'levene_stat': round(float(stat_L), 5),
        'levene_p': round(float(p_L), 8),
    })

    print(f"  {w50_labels[i]:>12s} {len(field_res):8d} {len(dense_res):8d}"
          f" {sf:8.4f} {sd:8.4f} {delta:+8.4f} {p_L:10.6f} {sig}")

# ================================================================
# FINE-BINNED SCATTER DERIVATIVE (look for inversion)
# ================================================================
print(f"\n{'='*72}")
print("FINE-BINNED SCATTER + DERIVATIVE ANALYSIS")
print(f"{'='*72}")

# Use log(W50) bins
n_fine = 12
logw_edges = np.linspace(1.3, 2.8, n_fine + 1)  # log10(20) ≈ 1.3, log10(600) ≈ 2.78
logw_centers = (logw_edges[:-1] + logw_edges[1:]) / 2

all_res = np.array([g['btfr_res'] for g in matched])
all_logw = np.array([g['logw50'] for g in matched])
all_env = np.array([g['env_binary'] for g in matched])

print(f"\n  {'log(W50)':>10s} {'N_all':>6s} {'σ_all':>8s} {'N_field':>8s} {'N_dense':>8s}"
      f" {'σ_field':>8s} {'σ_dense':>8s} {'Δσ(f-d)':>8s} {'Lev_p':>10s}")
print(f"  {'-'*80}")

fine_stats = []
for j in range(n_fine):
    lo, hi = logw_edges[j], logw_edges[j+1]
    center = logw_centers[j]

    mask = (all_logw >= lo) & (all_logw < hi)
    res_bin = all_res[mask]
    f_mask = mask & (all_env == 'field')
    d_mask = mask & (all_env == 'dense')
    res_f = all_res[f_mask]
    res_d = all_res[d_mask]

    if len(res_bin) < 20:
        continue

    s_all = float(np.std(res_bin))
    s_f = float(np.std(res_f)) if len(res_f) >= 10 else np.nan
    s_d = float(np.std(res_d)) if len(res_d) >= 10 else np.nan
    delta = s_f - s_d if not (np.isnan(s_f) or np.isnan(s_d)) else np.nan

    p_L = np.nan
    if len(res_f) >= 10 and len(res_d) >= 10:
        stat_L, p_L = levene(res_f, res_d)
        p_L = float(p_L)

    fine_stats.append({
        'center': float(center),
        'n_all': len(res_bin),
        'n_field': len(res_f),
        'n_dense': len(res_d),
        'sigma_all': round(s_all, 5),
        'sigma_field': round(s_f, 5) if not np.isnan(s_f) else None,
        'sigma_dense': round(s_d, 5) if not np.isnan(s_d) else None,
        'delta_sigma': round(delta, 5) if not np.isnan(delta) else None,
        'levene_p': round(p_L, 8) if not np.isnan(p_L) else None,
    })

    sf_str = f"{s_f:8.4f}" if not np.isnan(s_f) else "     ---"
    sd_str = f"{s_d:8.4f}" if not np.isnan(s_d) else "     ---"
    delta_str = f"{delta:+8.4f}" if not np.isnan(delta) else "     ---"
    p_str = f"{p_L:10.6f}" if not np.isnan(p_L) else "       ---"

    print(f"  {center:10.3f} {len(res_bin):6d} {s_all:8.4f} {len(res_f):8d} {len(res_d):8d}"
          f" {sf_str} {sd_str} {delta_str} {p_str}")

# Derivative of total scatter
valid_fine = [s for s in fine_stats if s['sigma_all'] is not None]
if len(valid_fine) >= 4:
    fc = np.array([s['center'] for s in valid_fine])
    fs = np.array([s['sigma_all'] for s in valid_fine])
    dsigma = numerical_derivative(fc, fs)
    sigma_crossings = find_zero_crossings(fc, dsigma)

    print(f"\n  dσ/d(logW50) zero crossings: {[f'{x:.3f}' for x in sigma_crossings]}")

    # Environmental delta derivative
    valid_delta = [s for s in fine_stats
                   if s['delta_sigma'] is not None]
    if len(valid_delta) >= 3:
        dc = np.array([s['center'] for s in valid_delta])
        dd = np.array([s['delta_sigma'] for s in valid_delta])
        delta_crossings = find_zero_crossings(dc, dd)
        print(f"  Δσ(field-dense) sign change at log(W50) = {[f'{x:.3f}' for x in delta_crossings]}")

# ================================================================
# HALO MASS GRADIENT (continuous environment)
# ================================================================
print(f"\n{'='*72}")
print("BTFR SCATTER VS HALO MASS (CONTINUOUS ENVIRONMENT)")
print(f"{'='*72}")

# Bin by halo mass
mh_edges = [0, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 15.0]
mh_labels = [f"{mh_edges[i]}-{mh_edges[i+1]}" for i in range(len(mh_edges)-1)]

all_logmh = np.array([g['logMh'] for g in matched])

print(f"\n  {'Halo mass':>12s} {'N':>6s} {'σ_BTFR':>8s} {'mean_res':>10s}")
print(f"  {'-'*40}")

for i in range(len(mh_edges) - 1):
    lo, hi = mh_edges[i], mh_edges[i+1]
    mask = (all_logmh >= lo) & (all_logmh < hi) if lo > 0 else (all_logmh <= 0) | ((all_logmh >= lo) & (all_logmh < hi))

    res_bin = all_res[mask]
    if len(res_bin) < 5:
        continue

    print(f"  {mh_labels[i]:>12s} {len(res_bin):6d} {np.std(res_bin):8.4f} {np.mean(res_bin):+10.4f}")

# ================================================================
# 2D GRID: W50 × Halo Mass
# ================================================================
print(f"\n{'='*72}")
print("2D SCATTER GRID: W50 × HALO MASS")
print(f"{'='*72}")

# Coarser bins for 2D grid
w50_grid = [20, 80, 150, 250, 600]
mh_grid = [0, 12.0, 12.5, 13.0, 15.0]

print(f"\n  BTFR scatter σ (dex) in each cell:")
header = f"  {'W50 \\ Mh':>12s}"
for k in range(len(mh_grid)-1):
    header += f" {mh_grid[k]:.0f}-{mh_grid[k+1]:.0f}".rjust(12)
print(header)
print(f"  {'-'*60}")

grid_data = []
for i in range(len(w50_grid)-1):
    row_str = f"  {w50_grid[i]:3d}-{w50_grid[i+1]:3d}"
    row_data = {'w50_range': f"{w50_grid[i]}-{w50_grid[i+1]}"}
    for k in range(len(mh_grid)-1):
        w_mask = np.array([(w50_grid[i] <= g['w50'] < w50_grid[i+1]) for g in matched])
        m_mask = np.array([(mh_grid[k] <= g['logMh'] < mh_grid[k+1]) if g['logMh'] > 0
                           else (mh_grid[k] == 0) for g in matched])
        mask = w_mask & m_mask
        res_cell = all_res[mask]

        if len(res_cell) >= 10:
            s = np.std(res_cell)
            row_str += f" {s:8.4f}({len(res_cell):4d})"
            row_data[f"mh_{mh_grid[k]:.0f}_{mh_grid[k+1]:.0f}"] = {
                'sigma': round(s, 5), 'n': len(res_cell)
            }
        else:
            row_str += "      ---     "
    print(row_str)
    grid_data.append(row_data)

# ================================================================
# VERDICT
# ================================================================
print(f"\n{'='*72}")
print("VERDICT")
print(f"{'='*72}")

# Check: at low W50, is scatter uniform across environments?
low_w50 = [b for b in bin_results if b['w50_hi'] <= 150]
high_w50 = [b for b in bin_results if b['w50_lo'] >= 150]

low_uniform = all(b['levene_p'] > 0.05 for b in low_w50) if low_w50 else False
high_different = any(b['levene_p'] < 0.05 for b in high_w50) if high_w50 else False

if low_uniform and high_different:
    verdict = ("BEC-CONSISTENT: BTFR scatter is uniform across environments at low W50 "
               "(condensate regime) and diverges at high W50 (baryon regime)")
elif low_uniform:
    verdict = ("PARTIALLY CONSISTENT: scatter uniform at low W50 but no clear divergence at high W50")
elif not low_w50:
    verdict = "INSUFFICIENT DATA at low W50 for environment comparison"
else:
    # Check if all bins are uniform
    all_uniform = all(b['levene_p'] > 0.05 for b in bin_results)
    if all_uniform:
        verdict = "UNIFORM EVERYWHERE: no environmental dependence detected at any W50"
    else:
        verdict = "COMPLEX: scatter pattern does not match simple BEC prediction"

print(f"\n  {verdict}")

# ================================================================
# SAVE
# ================================================================
output = {
    'test_name': 'alfalfa_yang_btfr_environmental_scatter',
    'description': ('ALFALFA α.100 × Yang DR7 cross-match: BTFR scatter as function '
                    'of velocity width (acceleration proxy) and halo mass (environment)'),
    'n_alfalfa': len(alfalfa),
    'n_matched': len(matched),
    'n_field': n_field,
    'n_dense': n_dense,
    'btfr_fit': {
        'slope': round(slope, 4),
        'intercept': round(intercept, 4),
        'overall_scatter': round(float(np.std(residuals)), 5),
    },
    'w50_binned_results': bin_results,
    'fine_binned_results': fine_stats,
    'grid_2d': grid_data,
    'verdict': verdict,
}

outpath = os.path.join(RESULTS_DIR, 'summary_alfalfa_yang_btfr.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
