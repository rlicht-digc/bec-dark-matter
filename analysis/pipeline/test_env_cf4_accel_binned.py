#!/usr/bin/env python3
"""
Acceleration-Binned Environmental RAR Scatter Test
====================================================

Letter-ready analysis: does the environmental scatter difference live where
the BEC condensate dominates?

Core argument:
  The RAR scatter difference between dense and field environments is
  concentrated at LOW accelerations (g_bar < 10^-10.5 m/s²), where the
  BEC condensate dominates and distance systematics have minimal impact.

  This acceleration dependence is a prediction of the fluid model:
  external tidal pressure confines the condensate boundary, and the
  boundary physics operates at the condensation threshold g†, not in the
  baryon-dominated interior.

  The signal is robust to both SPARC and CF4 distance sets in this regime.

Method:
  1. Run full 7-test battery on BOTH distance sets (SPARC, CF4)
  2. Bin by acceleration: show where the scatter difference lives
  3. Exclude 3 pathological UMa galaxies (UGC06446, UGC06786, UGC06787)
     and show the low-acceleration result is unchanged
  4. Error-corrected intrinsic scatter with Haubner+2025 scheme
  5. Galaxy-resampled bootstrap per acceleration bin

The 3 pathological galaxies:
  UGC06446: UMa member, fD=1 (Hubble flow), CF4 gives D=15.7 vs SPARC 28.7 Mpc
  UGC06786: UMa member, fD=1 (Hubble flow), CF4 gives D=9.88 vs SPARC 29.3 Mpc
  UGC06787: UMa member, fD=1 (Hubble flow), CF4 gives D=15.5 vs SPARC 29.3 Mpc
  These are UMa cluster members misclassified as fD=1 in SPARC, meaning CF4
  assigns a flow-model distance instead of the correct cluster distance.
"""

import os
import json
import numpy as np
from scipy.stats import levene, ks_2samp, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
CF4_DIR = os.path.join(PROJECT_ROOT, 'data', 'cf4')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10  # m/s²
kpc_m = 3.086e19      # meters per kpc

# 3 pathological UMa galaxies: cluster members with fD=1 in SPARC
PATHOLOGICAL = {'UGC06446', 'UGC06786', 'UGC06787'}


def rar_function(log_gbar, a0=1.2e-10):
    """RAR / BEC prediction."""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def haubner_delta(D_Mpc, delta_inf=0.022, alpha=-0.8, D_tr=46.0, kappa=1.8):
    """Haubner+2025 Eq 7: CF4 distance uncertainty in dex."""
    D = max(D_Mpc, 0.01)
    return delta_inf * D**alpha * (D**(1/kappa) + D_tr**(1/kappa))**(-alpha * kappa)


def haubner_delta_hubble(D_Mpc, delta_inf=0.031, alpha=-0.9, D_tr=44.0, kappa=1.0):
    """Haubner+2025: Hubble flow (V_h) distance uncertainty in dex."""
    D = max(D_Mpc, 0.01)
    return delta_inf * D**alpha * (D**(1/kappa) + D_tr**(1/kappa))**(-alpha * kappa)


# Environment classification (identical to test_env_scatter_definitive.py)
UMA_GALAXIES = {
    'NGC3726', 'NGC3769', 'NGC3877', 'NGC3893', 'NGC3917',
    'NGC3949', 'NGC3953', 'NGC3972', 'NGC3992', 'NGC4010',
    'NGC4013', 'NGC4051', 'NGC4085', 'NGC4088', 'NGC4100',
    'NGC4138', 'NGC4157', 'NGC4183', 'NGC4217',
    'UGC06399', 'UGC06446', 'UGC06667', 'UGC06786', 'UGC06787',
    'UGC06818', 'UGC06917', 'UGC06923', 'UGC06930', 'UGC06973',
    'UGC06983', 'UGC07089',
}

GROUP_MEMBERS = {
    'NGC2403': 'M81', 'NGC2976': 'M81', 'IC2574': 'M81',
    'DDO154': 'M81', 'DDO168': 'M81', 'UGC04483': 'M81',
    'NGC0300': 'Sculptor', 'NGC0055': 'Sculptor',
    'NGC0247': 'Sculptor', 'NGC7793': 'Sculptor',
    'NGC2915': 'CenA', 'UGCA442': 'CenA', 'ESO444-G084': 'CenA',
    'UGC07577': 'CVnI', 'UGC07232': 'CVnI', 'NGC3741': 'CVnI',
    'NGC4068': 'CVnI', 'UGC07866': 'CVnI', 'UGC07524': 'CVnI',
    'UGC08490': 'CVnI', 'UGC07559': 'CVnI',
    'NGC3109': 'Antlia', 'NGC5055': 'M101',
}

PRIMARY_ERRORS = {'TRGB': 0.05, 'Cepheid': 0.05, 'SBF': 0.05,
                  'SNe': 0.07, 'maser': 0.10}


def classify_env(name):
    if name in UMA_GALAXIES:
        return 'dense', 'UMa'
    if name in GROUP_MEMBERS:
        return 'dense', GROUP_MEMBERS[name]
    return 'field', 'field'


# ================================================================
# LOAD DATA
# ================================================================
def load_sparc_data():
    """Load SPARC rotation curves, properties, and CF4 distances."""
    print("[1] Loading SPARC data + CF4 distances...")

    # Rotation curves
    table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
    galaxies = {}
    with open(table2_path, 'r') as f:
        for line in f:
            if len(line.strip()) < 50:
                continue
            try:
                name = line[0:11].strip()
                if not name:
                    continue
                dist = float(line[12:18].strip())
                rad = float(line[19:25].strip())
                vobs = float(line[26:32].strip())
                evobs = float(line[33:38].strip())
                vgas = float(line[39:45].strip())
                vdisk = float(line[46:52].strip())
                vbul = float(line[53:59].strip())
            except (ValueError, IndexError):
                continue
            if name not in galaxies:
                galaxies[name] = {'R': [], 'Vobs': [], 'eVobs': [],
                                  'Vgas': [], 'Vdisk': [], 'Vbul': [],
                                  'dist': dist}
            galaxies[name]['R'].append(rad)
            galaxies[name]['Vobs'].append(vobs)
            galaxies[name]['eVobs'].append(evobs)
            galaxies[name]['Vgas'].append(vgas)
            galaxies[name]['Vdisk'].append(vdisk)
            galaxies[name]['Vbul'].append(vbul)

    for name in galaxies:
        for key in ['R', 'Vobs', 'eVobs', 'Vgas', 'Vdisk', 'Vbul']:
            galaxies[name][key] = np.array(galaxies[name][key])

    # Properties from MRT
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
                'T': int(parts[0]), 'D': float(parts[1]), 'eD': float(parts[2]),
                'fD': int(parts[3]), 'Inc': float(parts[4]), 'eInc': float(parts[5]),
                'L36': float(parts[6]), 'Reff': float(parts[8]),
                'SBeff': float(parts[9]), 'MHI': float(parts[12]),
                'Vflat': float(parts[14]), 'Q': int(parts[16]),
            }
        except (ValueError, IndexError):
            continue

    # CF4 distances
    cf4_path = os.path.join(CF4_DIR, 'cf4_distance_cache.json')
    cf4_distances = {}
    if os.path.exists(cf4_path):
        with open(cf4_path, 'r') as f:
            cf4_raw = json.load(f)
        for name, data in cf4_raw.items():
            if data.get('D_cf4') and data.get('status') == 'success':
                cf4_distances[name] = data['D_cf4']

    print(f"  {len(galaxies)} RCs, {len(sparc_props)} props, {len(cf4_distances)} CF4 distances")

    return galaxies, sparc_props, cf4_distances


# ================================================================
# COMPUTE RAR RESIDUALS
# ================================================================
def compute_residuals(galaxies, sparc_props, cf4_distances,
                      use_cf4=False, exclude_set=None):
    """
    Compute RAR residuals for all quality-cut galaxies.

    Args:
        use_cf4: If True, use CF4 distances for fD=1 galaxies
        exclude_set: Set of galaxy names to exclude

    Returns:
        dict of {name: result_dict}
    """
    if exclude_set is None:
        exclude_set = set()

    results = {}
    n_cf4_applied = 0

    for name, gdata in galaxies.items():
        if name in exclude_set:
            continue
        if name not in sparc_props:
            continue

        prop = sparc_props[name]
        if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
            continue

        fD = prop['fD']
        D_sparc = prop['D']

        # Distance selection
        if use_cf4 and fD == 1 and name in cf4_distances:
            D_use = cf4_distances[name]
            dist_source = 'CF4'
            n_cf4_applied += 1
        else:
            D_use = D_sparc
            dist_source = {1: 'Hubble', 2: 'TRGB', 3: 'Cepheid',
                           4: 'UMa', 5: 'SNe'}.get(fD, 'unknown')

        # Distance ratio for scaling
        D_ratio = D_use / gdata['dist']
        R = gdata['R'] * D_ratio  # kpc scales with D
        Vobs = gdata['Vobs']       # km/s is distance-independent
        sqrt_ratio = np.sqrt(D_ratio)
        Vgas = gdata['Vgas'] * sqrt_ratio
        Vdisk = gdata['Vdisk'] * sqrt_ratio
        Vbul = gdata['Vbul'] * sqrt_ratio

        # Baryonic and observed accelerations
        Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
        gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
        gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

        valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
        if np.sum(valid) < 5:
            continue

        log_gbar = np.log10(gbar_SI[valid])
        log_gobs = np.log10(gobs_SI[valid])
        log_gobs_rar = rar_function(log_gbar)
        log_res = log_gobs - log_gobs_rar

        env_binary, env_group = classify_env(name)

        # Distance uncertainty (Haubner+2025)
        if fD == 2:
            sigma_D_dex = np.log10(1 + PRIMARY_ERRORS['TRGB'])
        elif fD == 3:
            sigma_D_dex = np.log10(1 + PRIMARY_ERRORS['Cepheid'])
        elif fD == 4:
            sigma_D_dex = np.log10(1 + 0.10)
        elif fD == 5:
            sigma_D_dex = np.log10(1 + PRIMARY_ERRORS['SNe'])
        else:
            if dist_source == 'CF4':
                sigma_D_dex = haubner_delta(D_use)
            else:
                sigma_D_dex = haubner_delta_hubble(D_use)

        results[name] = {
            'log_gbar': log_gbar,
            'log_gobs': log_gobs,
            'log_res': log_res,
            'mean_res': float(np.mean(log_res)),
            'std_res': float(np.std(log_res)),
            'n_points': len(log_res),
            'env': env_binary,
            'env_group': env_group,
            'D_sparc': D_sparc,
            'D_use': D_use,
            'D_ratio': D_ratio,
            'dist_source': dist_source,
            'fD': fD,
            'sigma_D_dex': sigma_D_dex,
            'logMs': np.log10(max(0.5 * prop['L36'] * 1e9, 1e6)),
            'Vflat': prop['Vflat'],
            'T': prop['T'],
        }

    return results, n_cf4_applied


# ================================================================
# ACCELERATION-BINNED ENVIRONMENTAL TEST
# ================================================================
def run_accel_binned_test(results, label=""):
    """
    Run the acceleration-binned environmental scatter test.

    This is the core analysis for the Letter: show WHERE the scatter
    difference lives and quantify its robustness.
    """
    print(f"\n{'='*72}")
    print(f"ACCELERATION-BINNED ENVIRONMENTAL TEST: {label}")
    print(f"{'='*72}")

    n_dense = sum(1 for r in results.values() if r['env'] == 'dense')
    n_field = sum(1 for r in results.values() if r['env'] == 'field')
    print(f"  Galaxies: {len(results)} total ({n_dense} dense, {n_field} field)")

    # Acceleration bins — the key structure
    # Low: where BEC condensate dominates, distance errors small
    # High: baryon-dominated interior, distance errors larger
    bins = [
        (-13.0, -11.5, 'Deep DM (g < g†)',         'condensate'),
        (-11.5, -10.5, 'Transition (g ~ g†)',       'boundary'),
        (-10.5, -9.5,  'Baryon-dominated',          'interior'),
        (-9.5,  -8.0,  'High acceleration',         'newtonian'),
    ]

    # Also the critical 2-bin split for the Letter
    bins_letter = [
        (-13.0, -10.5, 'LOW: g < 10^{-10.5} (condensate regime)', 'low_accel'),
        (-10.5, -8.0,  'HIGH: g > 10^{-10.5} (baryon regime)',    'high_accel'),
    ]

    rng = np.random.default_rng(42)
    n_boot = 10000

    all_bin_results = []

    for bin_set_label, bin_set in [('4-bin', bins), ('2-bin (Letter)', bins_letter)]:
        print(f"\n  --- {bin_set_label} ---")
        print(f"  {'Regime':42s} {'N_d':>5s} {'N_f':>5s} {'σ_d':>7s} {'σ_f':>7s}"
              f" {'Δσ':>8s} {'Lev_p':>9s} {'Boot_p':>8s}")
        print(f"  {'-'*90}")

        for gbar_lo, gbar_hi, label_bin, regime in bin_set:
            d_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) &
                                                  (r['log_gbar'] < gbar_hi)]
                                    for r in results.values() if r['env'] == 'dense'])
            f_pts = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) &
                                                  (r['log_gbar'] < gbar_hi)]
                                    for r in results.values() if r['env'] == 'field'])

            if len(d_pts) < 5 or len(f_pts) < 5:
                print(f"  {label_bin:42s} {len(d_pts):5d} {len(f_pts):5d}     ---     ---      ---       ---      ---")
                continue

            sd, sf = np.std(d_pts), np.std(f_pts)
            delta = sf - sd

            # Levene
            stat_L, p_L = levene(d_pts, f_pts)

            # Bootstrap: galaxy-resampled within this bin
            dense_names = [n for n, r in results.items() if r['env'] == 'dense']
            field_names = [n for n, r in results.items() if r['env'] == 'field']

            boot_deltas = np.zeros(n_boot)
            for b in range(n_boot):
                d_boot_names = rng.choice(dense_names, size=len(dense_names), replace=True)
                f_boot_names = rng.choice(field_names, size=len(field_names), replace=True)

                d_boot_pts = []
                for name in d_boot_names:
                    r = results[name]
                    mask = (r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)
                    if np.any(mask):
                        d_boot_pts.extend(r['log_res'][mask])

                f_boot_pts = []
                for name in f_boot_names:
                    r = results[name]
                    mask = (r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)
                    if np.any(mask):
                        f_boot_pts.extend(r['log_res'][mask])

                if len(d_boot_pts) >= 5 and len(f_boot_pts) >= 5:
                    boot_deltas[b] = np.std(f_boot_pts) - np.std(d_boot_pts)
                else:
                    boot_deltas[b] = np.nan

            valid_boots = boot_deltas[~np.isnan(boot_deltas)]
            if len(valid_boots) > 100:
                p_boot = float(np.mean(valid_boots > 0))
                ci_95 = np.percentile(valid_boots, [2.5, 97.5])
            else:
                p_boot = np.nan
                ci_95 = [np.nan, np.nan]

            sig_marker = '***' if p_L < 0.001 else '**' if p_L < 0.01 else '*' if p_L < 0.05 else ''

            print(f"  {label_bin:42s} {len(d_pts):5d} {len(f_pts):5d} "
                  f"{sd:7.4f} {sf:7.4f} {delta:+8.4f} {p_L:9.6f} {p_boot:8.4f} {sig_marker}")

            result_entry = {
                'regime': regime,
                'label': label_bin,
                'gbar_range': [gbar_lo, gbar_hi],
                'n_dense_pts': len(d_pts),
                'n_field_pts': len(f_pts),
                'sigma_dense': round(float(sd), 5),
                'sigma_field': round(float(sf), 5),
                'delta_sigma': round(float(delta), 5),
                'levene_p': round(float(p_L), 8),
                'boot_p_field_gt_dense': round(float(p_boot), 4),
                'boot_ci_95': [round(float(ci_95[0]), 5), round(float(ci_95[1]), 5)],
                'bin_set': bin_set_label,
            }
            all_bin_results.append(result_entry)

    # Overall (unbinned) for reference
    all_dense = np.concatenate([r['log_res'] for r in results.values() if r['env'] == 'dense'])
    all_field = np.concatenate([r['log_res'] for r in results.values() if r['env'] == 'field'])
    sd_all, sf_all = np.std(all_dense), np.std(all_field)
    stat_L, p_L = levene(all_dense, all_field)

    print(f"\n  {'OVERALL (unbinned)':42s} {len(all_dense):5d} {len(all_field):5d} "
          f"{sd_all:7.4f} {sf_all:7.4f} {sf_all-sd_all:+8.4f} {p_L:9.6f}")

    # Galaxy-level scatter
    dense_means = np.array([r['mean_res'] for r in results.values() if r['env'] == 'dense'])
    field_means = np.array([r['mean_res'] for r in results.values() if r['env'] == 'field'])
    stat_L_gl, p_L_gl = levene(dense_means, field_means)
    print(f"\n  Galaxy-level: dense σ(mean_res) = {np.std(dense_means):.4f}, "
          f"field σ = {np.std(field_means):.4f}, Levene p = {p_L_gl:.6f}")

    # Error-corrected intrinsic scatter
    dense_gals = [(r['std_res'], r['sigma_D_dex']) for r in results.values() if r['env'] == 'dense']
    field_gals = [(r['std_res'], r['sigma_D_dex']) for r in results.values() if r['env'] == 'field']

    dense_intr = np.array([np.sqrt(max(o**2 - d**2, 0)) for o, d in dense_gals])
    field_intr = np.array([np.sqrt(max(o**2 - d**2, 0)) for o, d in field_gals])
    stat_L_intr, p_L_intr = levene(dense_intr, field_intr)

    print(f"  Intrinsic (error-corrected): dense = {np.mean(dense_intr):.4f}, "
          f"field = {np.mean(field_intr):.4f}, Levene p = {p_L_intr:.6f}")

    summary = {
        'label': label,
        'n_galaxies': len(results),
        'n_dense': n_dense,
        'n_field': n_field,
        'overall': {
            'sigma_dense': round(float(sd_all), 5),
            'sigma_field': round(float(sf_all), 5),
            'delta': round(float(sf_all - sd_all), 5),
            'levene_p': round(float(levene(all_dense, all_field)[1]), 8),
        },
        'galaxy_level': {
            'sigma_dense': round(float(np.std(dense_means)), 5),
            'sigma_field': round(float(np.std(field_means)), 5),
            'levene_p': round(float(p_L_gl), 8),
        },
        'intrinsic': {
            'mean_dense': round(float(np.mean(dense_intr)), 5),
            'mean_field': round(float(np.mean(field_intr)), 5),
            'levene_p': round(float(p_L_intr), 8),
        },
        'binned_results': all_bin_results,
    }

    return summary


# ================================================================
# PATHOLOGICAL GALAXY DIAGNOSTIC
# ================================================================
def pathological_diagnostic(results_full, results_cf4_full):
    """
    Show exactly what happens with the 3 pathological UMa galaxies
    under SPARC vs CF4 distances.
    """
    print(f"\n{'='*72}")
    print("PATHOLOGICAL GALAXY DIAGNOSTIC")
    print(f"{'='*72}")
    print(f"  These 3 UMa cluster members have fD=1 in SPARC,")
    print(f"  so CF4 replaces their cluster distance with a flow-model distance.")
    print(f"\n  {'Galaxy':12s} {'D_SPARC':>8s} {'D_CF4':>8s} {'Ratio':>7s} "
          f"{'σ_SPARC':>8s} {'σ_CF4':>8s} {'Δσ':>8s}")
    print(f"  {'-'*65}")

    for name in sorted(PATHOLOGICAL):
        r_sparc = results_full.get(name, {})
        r_cf4 = results_cf4_full.get(name, {})

        d_sparc = r_sparc.get('D_use', np.nan)
        d_cf4 = r_cf4.get('D_use', np.nan)
        s_sparc = r_sparc.get('std_res', np.nan)
        s_cf4 = r_cf4.get('std_res', np.nan)

        ratio = d_cf4 / d_sparc if d_sparc > 0 else np.nan
        delta = s_cf4 - s_sparc

        print(f"  {name:12s} {d_sparc:8.1f} {d_cf4:8.1f} {ratio:7.3f} "
              f"{s_sparc:8.4f} {s_cf4:8.4f} {delta:+8.4f}")

    # Show impact on overall
    print(f"\n  Impact on aggregate dense scatter:")
    all_dense_sparc = np.concatenate([r['log_res'] for r in results_full.values()
                                       if r['env'] == 'dense'])
    all_dense_cf4 = np.concatenate([r['log_res'] for r in results_cf4_full.values()
                                     if r['env'] == 'dense'])
    print(f"    SPARC:  σ_dense = {np.std(all_dense_sparc):.4f} dex ({len(all_dense_sparc)} pts)")
    print(f"    CF4:    σ_dense = {np.std(all_dense_cf4):.4f} dex ({len(all_dense_cf4)} pts)")

    # Without pathological
    clean_dense_sparc = np.concatenate([r['log_res'] for name, r in results_full.items()
                                         if r['env'] == 'dense' and name not in PATHOLOGICAL])
    clean_dense_cf4 = np.concatenate([r['log_res'] for name, r in results_cf4_full.items()
                                       if r['env'] == 'dense' and name not in PATHOLOGICAL])
    print(f"    SPARC (no path): σ_dense = {np.std(clean_dense_sparc):.4f} dex")
    print(f"    CF4 (no path):   σ_dense = {np.std(clean_dense_cf4):.4f} dex")


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("ACCELERATION-BINNED ENVIRONMENTAL RAR SCATTER TEST")
print("Clean CF4 Reanalysis for Letter")
print("=" * 72)

# Load data once
galaxies, sparc_props, cf4_distances = load_sparc_data()

# ----------------------------------------------------------------
# RUN 1: SPARC distances, full sample
# ----------------------------------------------------------------
print("\n" + "#" * 72)
print("RUN 1: SPARC DISTANCES — FULL SAMPLE")
print("#" * 72)
results_sparc_full, _ = compute_residuals(galaxies, sparc_props, cf4_distances,
                                           use_cf4=False)
summary_sparc_full = run_accel_binned_test(results_sparc_full,
                                            "SPARC distances, full sample")

# ----------------------------------------------------------------
# RUN 2: CF4 distances, full sample
# ----------------------------------------------------------------
print("\n" + "#" * 72)
print("RUN 2: CF4 DISTANCES — FULL SAMPLE")
print("#" * 72)
results_cf4_full, n_cf4 = compute_residuals(galaxies, sparc_props, cf4_distances,
                                              use_cf4=True)
print(f"  CF4 distances applied to {n_cf4} galaxies")
summary_cf4_full = run_accel_binned_test(results_cf4_full,
                                          "CF4 distances, full sample")

# ----------------------------------------------------------------
# DIAGNOSTIC: Pathological galaxies
# ----------------------------------------------------------------
pathological_diagnostic(results_sparc_full, results_cf4_full)

# ----------------------------------------------------------------
# RUN 3: SPARC distances, clean sample (no pathological)
# ----------------------------------------------------------------
print("\n" + "#" * 72)
print("RUN 3: SPARC DISTANCES — CLEAN SAMPLE (exclude 3 pathological UMa)")
print("#" * 72)
results_sparc_clean, _ = compute_residuals(galaxies, sparc_props, cf4_distances,
                                            use_cf4=False, exclude_set=PATHOLOGICAL)
summary_sparc_clean = run_accel_binned_test(results_sparc_clean,
                                             "SPARC distances, clean sample")

# ----------------------------------------------------------------
# RUN 4: CF4 distances, clean sample (no pathological)
# ----------------------------------------------------------------
print("\n" + "#" * 72)
print("RUN 4: CF4 DISTANCES — CLEAN SAMPLE (exclude 3 pathological UMa)")
print("#" * 72)
results_cf4_clean, n_cf4_clean = compute_residuals(galaxies, sparc_props, cf4_distances,
                                                     use_cf4=True, exclude_set=PATHOLOGICAL)
print(f"  CF4 distances applied to {n_cf4_clean} galaxies")
summary_cf4_clean = run_accel_binned_test(results_cf4_clean,
                                           "CF4 distances, clean sample")

# ================================================================
# LETTER-READY COMPARISON TABLE
# ================================================================
print("\n" + "=" * 72)
print("LETTER-READY COMPARISON: Low-Acceleration Regime")
print("=" * 72)
print(f"\n  The critical test: g_bar < 10^{{-10.5}} m/s² (condensate regime)")
print(f"  BEC prediction: field scatter > dense scatter (tidal disruption)")
print()

# Extract low-acceleration 2-bin results from each run
runs = [
    ('SPARC full',  summary_sparc_full),
    ('CF4 full',    summary_cf4_full),
    ('SPARC clean', summary_sparc_clean),
    ('CF4 clean',   summary_cf4_clean),
]

print(f"  {'Run':14s} {'σ_dense':>8s} {'σ_field':>8s} {'Δσ':>8s} {'Levene p':>10s} {'Boot p(f>d)':>12s}")
print(f"  {'-'*70}")

for run_label, summary in runs:
    low_bins = [b for b in summary['binned_results']
                if b['bin_set'] == '2-bin (Letter)' and b['regime'] == 'low_accel']
    if low_bins:
        b = low_bins[0]
        print(f"  {run_label:14s} {b['sigma_dense']:8.4f} {b['sigma_field']:8.4f} "
              f"{b['delta_sigma']:+8.4f} {b['levene_p']:10.6f} {b['boot_p_field_gt_dense']:12.4f}")

print(f"\n  High-acceleration comparison (g_bar > 10^{{-10.5}}):")
print(f"  {'Run':14s} {'σ_dense':>8s} {'σ_field':>8s} {'Δσ':>8s} {'Levene p':>10s} {'Boot p(f>d)':>12s}")
print(f"  {'-'*70}")

for run_label, summary in runs:
    high_bins = [b for b in summary['binned_results']
                 if b['bin_set'] == '2-bin (Letter)' and b['regime'] == 'high_accel']
    if high_bins:
        b = high_bins[0]
        print(f"  {run_label:14s} {b['sigma_dense']:8.4f} {b['sigma_field']:8.4f} "
              f"{b['delta_sigma']:+8.4f} {b['levene_p']:10.6f} {b['boot_p_field_gt_dense']:12.4f}")

# ================================================================
# SAVE RESULTS
# ================================================================
print(f"\n{'='*72}")
print("SAVING RESULTS")
print(f"{'='*72}")

output = {
    'test_name': 'acceleration_binned_environmental_scatter',
    'description': ('Letter-ready: acceleration-dependent environmental RAR scatter '
                    'with clean CF4 reanalysis'),
    'pathological_galaxies': list(PATHOLOGICAL),
    'pathological_reason': ('UMa cluster members with fD=1 in SPARC; CF4 replaces '
                           'cluster distance with flow-model distance'),
    'runs': {
        'sparc_full': summary_sparc_full,
        'cf4_full': summary_cf4_full,
        'sparc_clean': summary_sparc_clean,
        'cf4_clean': summary_cf4_clean,
    },
}

outpath = os.path.join(RESULTS_DIR, 'summary_env_cf4_accel_binned.json')
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
