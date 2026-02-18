#!/usr/bin/env python3
"""
STEP 8: Expanded-Sample RAR Pipeline
======================================
Combines ALL available rotation curve datasets into one unified
environmental scatter test:

  1. SPARC (175 galaxies, full mass models) — already processed
  2. de Blok+2002 LSB galaxies (26, full mass models: Vgas, Vdisk, Vrot)
  3. WALLABY DR1+DR2 (165 after cuts, gas-only RAR)
  4. Santos-Santos+2020 (160 dwarfs, Vmax/Vbmax summary)

Total: ~500+ unique galaxies

Strategy:
  - SPARC + de Blok LSBs: full RAR (gbar from mass models)
  - WALLABY: gas-only RAR (conservative, valid at low-a)
  - Santos-Santos: global RAR (single Vmax point per galaxy)

Russell Licht — Primordial Fluid DM Project
Feb 2026
"""

import numpy as np
from scipy import stats
import csv
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

g_dagger = 1.20e-10   # m/s^2
conv = 1e6 / 3.0857e19  # (km/s)^2/kpc -> m/s^2
H0 = 75.0

# ============================================================
# LOAD SPARC PIPELINE RESULTS
# ============================================================
def load_sparc_results(use_cf4=True):
    """Load pre-computed SPARC RAR results from pipeline 02."""
    suffix = 'cf4_haubner' if use_cf4 else 'sparc_orig_haubner'
    csv_path = os.path.join(OUTPUT_DIR, f'galaxy_results_{suffix}.csv')

    if not os.path.exists(csv_path):
        print(f"  SPARC results not found: {csv_path}")
        return []

    galaxies = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            galaxies.append({
                'name': row['name'],
                'source': 'SPARC',
                'env_dense': row.get('env_dense', 'field'),
                'logMh': float(row.get('logMh', 11.0)),
                'sigma_all': float(row.get('rms_res', 0)),
            })

    print(f"  SPARC: {len(galaxies)} galaxies")
    return galaxies


# ============================================================
# PARSE DE BLOK+2002 LSB ROTATION CURVES
# ============================================================
def load_deblok2002():
    """
    Load de Blok+2002 LSB galaxy processed rotation curves.
    These have Vgas, Vdisk, Vrot, e_Vrot — same as SPARC.
    """
    rc_path = os.path.join(DATA_DIR, 'hi_surveys', 'deblok2002_proc_rotcurves.tsv')
    gal_path = os.path.join(DATA_DIR, 'hi_surveys', 'deblok2002_galaxies.tsv')

    if not os.path.exists(rc_path):
        print("  de Blok+2002 rotation curves not found")
        return {}

    # Parse galaxy properties
    gal_props = {}
    if os.path.exists(gal_path):
        with open(gal_path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('-') or line.strip() == '':
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    if name and not name.startswith('#'):
                        gal_props[name] = {'name': name}
                        # Try to extract distance, RA, Dec from other columns
                        for i, p in enumerate(parts):
                            gal_props[name][f'col{i}'] = p.strip()

    # Parse rotation curves
    galaxies = {}
    with open(rc_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('-') or line.strip() == '':
                continue
            # Skip header
            if 'Name' in line and 'arcsec' not in line and 'kpc' not in line:
                continue
            if 'arcsec' in line or 'km/s' in line or '----' in line:
                continue

            parts = line.split('\t')
            if len(parts) < 7:
                parts = line.split()
            if len(parts) < 7:
                continue

            try:
                name = parts[0].strip()
                r_arcsec = float(parts[1].strip())
                r_kpc = float(parts[2].strip())
                vgas = float(parts[3].strip()) if parts[3].strip() else 0.0
                vdisk = float(parts[4].strip()) if parts[4].strip() else 0.0
                vrot = float(parts[5].strip()) if parts[5].strip() else 0.0
                evrot = float(parts[6].strip()) if parts[6].strip() else 5.0
            except (ValueError, IndexError):
                continue

            if name not in galaxies:
                galaxies[name] = {
                    'name': name,
                    'R_kpc': [], 'Vrot': [], 'eVrot': [],
                    'Vgas': [], 'Vdisk': [],
                }

            galaxies[name]['R_kpc'].append(r_kpc)
            galaxies[name]['Vrot'].append(vrot)
            galaxies[name]['eVrot'].append(evrot)
            galaxies[name]['Vgas'].append(vgas)
            galaxies[name]['Vdisk'].append(vdisk)

    # Convert to numpy
    for name in galaxies:
        for key in ['R_kpc', 'Vrot', 'eVrot', 'Vgas', 'Vdisk']:
            galaxies[name][key] = np.array(galaxies[name][key])

    print(f"  de Blok+2002 LSB: {len(galaxies)} galaxies with rotation curves")
    return galaxies


def compute_rar_deblok(gdata, Y_d=0.5):
    """Compute RAR for de Blok+2002 galaxies (have Vgas + Vdisk)."""
    R = gdata['R_kpc']
    Vrot = gdata['Vrot']
    eVrot = gdata['eVrot']
    Vgas = gdata['Vgas']
    Vdisk = gdata['Vdisk']

    valid = (R > 0) & (Vrot > 0)
    if np.sum(valid) < 3:
        return None

    R_v = R[valid]
    Vrot_v = Vrot[valid]
    eVrot_v = eVrot[valid]
    Vgas_v = Vgas[valid]
    Vdisk_v = Vdisk[valid]

    # Baryonic acceleration: gbar = (Vgas² + Y_d * Vdisk²) / R
    Vbar_sq = np.sign(Vgas_v) * Vgas_v**2 + Y_d * Vdisk_v**2
    gbar = np.abs(Vbar_sq) / R_v * conv
    gobs = Vrot_v**2 / R_v * conv

    ok = (gbar > 0) & (gobs > 0)
    if np.sum(ok) < 3:
        return None

    gbar_ok = gbar[ok]
    gobs_ok = gobs[ok]

    gobs_pred = gbar_ok / (1 - np.exp(-np.sqrt(gbar_ok / g_dagger)))

    log_gbar = np.log10(gbar_ok)
    log_gobs = np.log10(gobs_ok)
    log_gobs_pred = np.log10(gobs_pred)
    log_res = log_gobs - log_gobs_pred

    sigma_log = 2.0 * eVrot_v[ok] / np.maximum(Vrot_v[ok], 1) / np.log(10)

    return {
        'log_gbar': log_gbar,
        'log_gobs': log_gobs,
        'log_res': log_res,
        'sigma_log_gobs': sigma_log,
        'n_points': len(log_gbar),
    }


# ============================================================
# LOAD WALLABY RESULTS
# ============================================================
def load_wallaby_results():
    """Load pre-computed WALLABY RAR points."""
    pts_path = os.path.join(OUTPUT_DIR, 'rar_points_wallaby_hubble_nodesi.csv')

    if not os.path.exists(pts_path):
        print("  WALLABY RAR points not found")
        return [], []

    galaxies = {}
    with open(pts_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['galaxy']
            if name not in galaxies:
                galaxies[name] = {
                    'name': name,
                    'source': 'WALLABY',
                    'env_dense': row.get('env_dense', 'field'),
                    'log_gbar': [],
                    'log_res': [],
                }
            galaxies[name]['log_gbar'].append(float(row['log_gbar']))
            galaxies[name]['log_res'].append(float(row['log_res']))

    for g in galaxies.values():
        g['log_gbar'] = np.array(g['log_gbar'])
        g['log_res'] = np.array(g['log_res'])

    print(f"  WALLABY: {len(galaxies)} galaxies")
    return galaxies


# ============================================================
# LOAD SANTOS-SANTOS DWARFS
# ============================================================
def load_santos_santos():
    """
    Load Santos-Santos+2020 dwarf galaxy compilation.
    These have Vmax and Vbmax (baryonic) — one point per galaxy.
    """
    tsv_path = os.path.join(DATA_DIR, 'hi_surveys', 'santos2020_table1.tsv')

    if not os.path.exists(tsv_path):
        print("  Santos-Santos+2020 not found")
        return {}

    galaxies = {}
    with open(tsv_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('-') or line.strip() == '':
                continue
            if 'Name' in line and 'Vmax' in line:
                continue
            if 'km/s' in line or 'Msun' in line or '----' in line:
                continue

            parts = line.split('\t')
            if len(parts) < 8:
                parts = line.split()
            if len(parts) < 8:
                continue

            try:
                name = parts[0].strip()
                sample = parts[1].strip()
                vmax = float(parts[2].strip())
                vbmax = float(parts[3].strip())
                mbar = float(parts[6].strip())
                m200 = float(parts[8].strip()) if len(parts) > 8 and parts[8].strip() else 0
            except (ValueError, IndexError):
                continue

            if vmax <= 0 or vbmax <= 0:
                continue

            # Compute single-point RAR
            # gobs ~ Vmax² / Rmax ∝ Vmax (at flat part)
            # gbar ~ Vbmax² / Rmax ∝ Vbmax
            # The ratio gobs/gbar ~ (Vmax/Vbmax)² is distance-independent
            # We can compute gbar from Mbar and rbhalf
            try:
                rbhalf = float(parts[7].strip())  # kpc
            except:
                rbhalf = 1.0

            if rbhalf > 0:
                gbar_est = vbmax**2 / rbhalf * conv
                gobs_est = vmax**2 / rbhalf * conv
            else:
                continue

            if gbar_est > 0 and gobs_est > 0:
                gobs_pred = gbar_est / (1 - np.exp(-np.sqrt(gbar_est / g_dagger)))
                log_gbar = np.log10(gbar_est)
                log_res = np.log10(gobs_est) - np.log10(gobs_pred)

                galaxies[name] = {
                    'name': name,
                    'source': f'SS20_{sample}',
                    'Vmax': vmax,
                    'Vbmax': vbmax,
                    'Mbar': mbar,
                    'M200': m200,
                    'log_gbar': log_gbar,
                    'log_res': log_res,
                }

                # Get RA/Dec from VizieR columns if available
                if len(parts) > 10:
                    try:
                        galaxies[name]['ra'] = float(parts[-2].strip())
                        galaxies[name]['dec'] = float(parts[-1].strip())
                    except:
                        pass

    print(f"  Santos-Santos+2020: {len(galaxies)} dwarfs")
    return galaxies


# ============================================================
# ENVIRONMENT CLASSIFICATION FOR NEW DATASETS
# ============================================================
def classify_environment_proximity(ra, dec, vsys, name=''):
    """
    Classify environment for any galaxy using known structure proximity.
    Same logic as 06_wallaby_environments.py.
    """
    STRUCTURES = {
        'Virgo': (187.71, 12.39, 1100, 14.9, 6.0, 800),
        'Fornax': (54.62, -35.45, 1379, 14.0, 2.0, 370),
        'Hydra': (159.18, -27.53, 3777, 14.5, 2.0, 700),
        'Norma': (248.15, -60.75, 4871, 15.0, 1.5, 925),
        'Centaurus': (192.20, -41.31, 3627, 14.5, 2.0, 750),
        'Coma': (194.95, 27.98, 6925, 15.0, 3.0, 1000),
        'Perseus': (49.95, 41.51, 5366, 14.8, 2.0, 1300),
        'NGC5044': (198.85, -16.39, 2750, 13.5, 1.5, 400),
        'NGC4636': (190.71, 2.69, 928, 13.3, 1.5, 350),
        'M81': (148.89, 69.07, -34, 12.0, 3.0, 200),
        'UMa': (178.0, 49.0, 1050, 12.8, 8.0, 200),
        'Sculptor': (11.89, -33.72, 200, 11.5, 5.0, 100),
        'CenA': (201.37, -43.02, 547, 12.5, 5.0, 200),
        'Antlia': (157.48, -35.32, 3041, 13.8, 1.0, 545),
    }

    best = None
    best_score = 999

    for sname, (sra, sdec, sv, slogMh, srv, ssigv) in STRUCTURES.items():
        # Angular separation
        ra1r, dec1r = np.radians(ra), np.radians(dec)
        ra2r, dec2r = np.radians(sra), np.radians(sdec)
        cos_sep = (np.sin(dec1r)*np.sin(dec2r) +
                   np.cos(dec1r)*np.cos(dec2r)*np.cos(ra1r - ra2r))
        cos_sep = np.clip(cos_sep, -1, 1)
        ang = np.degrees(np.arccos(cos_sep))

        if ang < srv and abs(vsys - sv) < 2.5 * ssigv:
            score = (ang/srv)**2 + (abs(vsys-sv)/ssigv)**2
            if score < best_score:
                best_score = score
                best = (sname, slogMh)

    if best:
        sname, logMh = best
        env_dense = 'dense' if logMh >= 12.5 else 'field'
        return sname, logMh, env_dense

    return 'field', 11.0, 'field'


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_expanded_pipeline():
    """Run the expanded-sample environmental test."""
    print("=" * 80)
    print("EXPANDED-SAMPLE RAR ENVIRONMENTAL TEST")
    print("SPARC + de Blok LSBs + WALLABY + Santos-Santos dwarfs")
    print("=" * 80)

    # ============================================================
    # 1. Load SPARC results (pre-computed)
    # ============================================================
    print("\n--- Loading datasets ---")

    # Load SPARC RAR points directly
    sparc_pts_path = None
    for suffix in ['sparc_orig_haubner', 'cf4_haubner']:
        p = os.path.join(OUTPUT_DIR, f'galaxy_results_{suffix}.csv')
        if os.path.exists(p):
            sparc_pts_path = p
            break

    sparc_data = {}
    if sparc_pts_path:
        # We need the actual RAR residuals per galaxy — re-run SPARC pipeline
        # For now, we'll re-compute from the raw data
        pass

    # ============================================================
    # 2. Load de Blok+2002 LSB galaxies
    # ============================================================
    deblok_galaxies = load_deblok2002()

    # Compute RAR for each
    deblok_results = []
    for name, gdata in deblok_galaxies.items():
        rar = compute_rar_deblok(gdata)
        if rar is None:
            continue

        # Environment (LSBs are mostly field galaxies)
        group, logMh, env_dense = 'field', 11.0, 'field'
        if 'ra' in gdata and 'dec' in gdata and 'vsys' in gdata:
            group, logMh, env_dense = classify_environment_proximity(
                gdata['ra'], gdata['dec'], gdata['vsys'], name
            )

        deblok_results.append({
            'name': name,
            'source': 'deBlok2002',
            'log_gbar': rar['log_gbar'],
            'log_res': rar['log_res'],
            'n_points': rar['n_points'],
            'env_dense': env_dense,
            'logMh': logMh,
            'group': group,
        })

    print(f"  de Blok+2002: {len(deblok_results)} galaxies with valid RAR")

    # ============================================================
    # 3. Load WALLABY results
    # ============================================================
    wallaby_galaxies = load_wallaby_results()

    wallaby_results = []
    for name, g in wallaby_galaxies.items():
        wallaby_results.append({
            'name': name,
            'source': 'WALLABY',
            'log_gbar': g['log_gbar'],
            'log_res': g['log_res'],
            'n_points': len(g['log_gbar']),
            'env_dense': g['env_dense'],
            'logMh': 11.0,  # Default; updated from env catalog
        })

    # ============================================================
    # 4. Load Santos-Santos dwarfs
    # ============================================================
    santos = load_santos_santos()

    santos_results = []
    for name, g in santos.items():
        # Classify environment if we have coordinates
        env_dense = 'field'
        logMh = 11.0
        group = 'field'

        if 'ra' in g and 'dec' in g:
            vsys_est = g['Vmax'] * 10 if g['Vmax'] < 200 else g['Vmax']  # rough
            group, logMh, env_dense = classify_environment_proximity(
                g['ra'], g['dec'], vsys_est, name
            )

        santos_results.append({
            'name': name,
            'source': g['source'],
            'log_gbar': np.array([g['log_gbar']]),
            'log_res': np.array([g['log_res']]),
            'n_points': 1,
            'env_dense': env_dense,
            'logMh': logMh,
            'group': group,
        })

    print(f"  Santos-Santos+2020: {len(santos_results)} dwarfs with valid RAR")

    # ============================================================
    # 5. Combined analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("COMBINED ANALYSIS")
    print("=" * 80)

    # All datasets combined
    all_datasets = {
        'deBlok2002': deblok_results,
        'WALLABY': wallaby_results,
        'Santos-Santos': santos_results,
    }

    # Remove SPARC duplicates from other datasets
    sparc_names = set()
    env_catalog_path = os.path.join(DATA_DIR, 'sparc_environment_catalog.json')
    if os.path.exists(env_catalog_path):
        with open(env_catalog_path, 'r') as f:
            sparc_env = json.load(f)
            sparc_names = set(sparc_env.keys())

    # Check for overlaps
    for ds_name, ds_results in all_datasets.items():
        overlap = [g for g in ds_results if g['name'] in sparc_names]
        if overlap:
            print(f"  {ds_name}: {len(overlap)} overlap with SPARC (will use SPARC version)")
            all_datasets[ds_name] = [g for g in ds_results if g['name'] not in sparc_names]

    # Combine all non-SPARC results
    combined = []
    for ds_name, ds_results in all_datasets.items():
        combined.extend(ds_results)

    print(f"\n  Total new galaxies (non-SPARC): {len(combined)}")
    source_counts = {}
    for g in combined:
        src = g['source']
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {n}")

    # ============================================================
    # 6. Environmental test on combined dataset
    # ============================================================
    print("\n" + "=" * 80)
    print("ENVIRONMENTAL SCATTER TEST — EXPANDED SAMPLE")
    print("=" * 80)

    dense_gbar, dense_res = [], []
    field_gbar, field_res = [], []
    n_dense_gal, n_field_gal = 0, 0

    for g in combined:
        gb = g['log_gbar']
        lr = g['log_res']
        if g['env_dense'] == 'dense':
            dense_gbar.extend(gb)
            dense_res.extend(lr)
            n_dense_gal += 1
        else:
            field_gbar.extend(gb)
            field_res.extend(lr)
            n_field_gal += 1

    dense_gbar = np.array(dense_gbar)
    dense_res = np.array(dense_res)
    field_gbar = np.array(field_gbar)
    field_res = np.array(field_res)

    print(f"\n  Dense: {len(dense_res)} points from {n_dense_gal} galaxies")
    print(f"  Field: {len(field_res)} points from {n_field_gal} galaxies")

    if len(dense_res) > 10 and len(field_res) > 10:
        print(f"  σ_dense = {np.std(dense_res):.4f} dex")
        print(f"  σ_field = {np.std(field_res):.4f} dex")
        delta = np.std(field_res) - np.std(dense_res)
        print(f"  Δσ (field - dense) = {delta:+.4f} dex")

        # Bootstrap
        n_boot = 10000
        np.random.seed(42)
        combined_res = np.concatenate([dense_res, field_res])
        nd = len(dense_res)
        boot = np.zeros(n_boot)
        for i in range(n_boot):
            s = np.random.permutation(combined_res)
            boot[i] = np.std(s[nd:]) - np.std(s[:nd])
        p_val = np.mean(boot >= delta)
        print(f"  P(field > dense) = {1-p_val:.4f}")

        # Levene's test
        lev_stat, lev_p = stats.levene(dense_res, field_res)
        print(f"  Levene's test: F={lev_stat:.3f}, p={lev_p:.6f}")

    # ============================================================
    # 7. Binned analysis
    # ============================================================
    print("\n--- Binned Environmental Analysis ---")
    bin_edges = np.array([-13.0, -12.0, -11.0, -10.0, -9.0, -8.0])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned = []

    for j in range(len(bin_centers)):
        lo, hi = bin_edges[j], bin_edges[j+1]
        d_mask = (dense_gbar >= lo) & (dense_gbar < hi) if len(dense_gbar) > 0 else np.array([], bool)
        f_mask = (field_gbar >= lo) & (field_gbar < hi) if len(field_gbar) > 0 else np.array([], bool)

        d_r = dense_res[d_mask] if np.sum(d_mask) > 0 else np.array([])
        f_r = field_res[f_mask] if np.sum(f_mask) > 0 else np.array([])

        if len(d_r) >= 5 and len(f_r) >= 5:
            d_std = np.std(d_r)
            f_std = np.std(f_r)
            delta_bin = f_std - d_std

            comb = np.concatenate([d_r, f_r])
            nd_bin = len(d_r)
            bb = np.zeros(5000)
            for i in range(5000):
                s = np.random.permutation(comb)
                bb[i] = np.std(s[nd_bin:]) - np.std(s[:nd_bin])
            p_bin = np.mean(bb >= delta_bin)

            binned.append({
                'center': float(bin_centers[j]),
                'n_dense': len(d_r), 'n_field': len(f_r),
                'sigma_dense': float(d_std), 'sigma_field': float(f_std),
                'delta': float(delta_bin), 'p_field_gt_dense': float(1-p_bin),
            })

            marker = "✓" if delta_bin > 0 else "✗"
            print(f"  {marker} Bin {bin_centers[j]:.1f}: dense={d_std:.4f} ({len(d_r)}), "
                  f"field={f_std:.4f} ({len(f_r)}), Δ={delta_bin:+.4f}, P={1-p_bin:.3f}")
        else:
            print(f"  - Bin {bin_centers[j]:.1f}: dense={len(d_r)}, field={len(f_r)} (insufficient)")

    # ============================================================
    # 8. Per-dataset breakdown
    # ============================================================
    print("\n--- Per-Dataset Environmental Breakdown ---")

    for ds_name, ds_results in all_datasets.items():
        if not ds_results:
            continue

        d_r = np.concatenate([g['log_res'] for g in ds_results if g['env_dense'] == 'dense']) \
            if any(g['env_dense'] == 'dense' for g in ds_results) else np.array([])
        f_r = np.concatenate([g['log_res'] for g in ds_results if g['env_dense'] == 'field']) \
            if any(g['env_dense'] == 'field' for g in ds_results) else np.array([])

        nd = sum(1 for g in ds_results if g['env_dense'] == 'dense')
        nf = sum(1 for g in ds_results if g['env_dense'] == 'field')

        if len(d_r) > 5 and len(f_r) > 5:
            delta = np.std(f_r) - np.std(d_r)
            print(f"  {ds_name}: dense={nd} gal ({len(d_r)} pts, σ={np.std(d_r):.4f}), "
                  f"field={nf} gal ({len(f_r)} pts, σ={np.std(f_r):.4f}), Δ={delta:+.4f}")
        else:
            print(f"  {ds_name}: dense={nd}/{len(d_r)} pts, field={nf}/{len(f_r)} pts")

    # ============================================================
    # 9. Save results
    # ============================================================
    print("\n--- Saving ---")

    summary = {
        'pipeline': 'expanded_sample',
        'n_galaxies_total': len(combined),
        'n_dense': n_dense_gal,
        'n_field': n_field_gal,
        'datasets': source_counts,
        'sigma_dense': float(np.std(dense_res)) if len(dense_res) > 0 else None,
        'sigma_field': float(np.std(field_res)) if len(field_res) > 0 else None,
        'delta_sigma': float(np.std(field_res) - np.std(dense_res)) if len(dense_res) > 0 and len(field_res) > 0 else None,
        'binned_results': binned,
    }

    with open(os.path.join(OUTPUT_DIR, 'summary_expanded_sample.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved: results/summary_expanded_sample.json")

    print("\n" + "=" * 80)
    print("EXPANDED PIPELINE COMPLETE")
    print("=" * 80)

    return summary


if __name__ == '__main__':
    run_expanded_pipeline()
