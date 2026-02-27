#!/usr/bin/env python3
"""
PATHOLOGICAL GALAXY ANALYSIS — Deep Dive
==========================================

Deep dive into three galaxies that cause pathological behavior in the
environmental scatter analysis: UGC06786, UGC06446, UGC06787.

All three are SPARC UMa cluster members (fD=4, D~18 Mpc) but have CF4
flow-model distances of ~9-16 Mpc, creating massive scatter when CF4
distances are used.

Per galaxy:
  1. RAR under 3 distances (SPARC, CF4, UMa cluster mean 18.6 Mpc)
  2. Acceleration-bin contribution analysis
  3. Environmental influence (jackknife) — remove galaxy, recompute Levene
  4. UMa membership check (angular separation, velocity offset)
  5. Virgo infall diagnostic (pure Hubble vs CF4 vs SPARC)
  6. Root cause determination

Inputs:
  - data/sparc/SPARC_table2_rotmods.dat
  - data/sparc/SPARC_Lelli2016c.mrt
  - data/cf4/cf4_distance_cache.json

Output:
  - analysis/results/summary_pathological_galaxies.json
"""

import os
import json
import numpy as np
from scipy.stats import levene
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics
g_dagger = 1.20e-10
kpc_m = 3.086e19
LOG_G_DAGGER = np.log10(g_dagger)
H0 = 73.0  # km/s/Mpc (SH0ES-consistent for Hubble flow calc)

TARGET_GALAXIES = ['UGC06786', 'UGC06446', 'UGC06787']
UMA_DISTANCE = 18.6  # Verheijen+2001 UMa cluster mean distance (Mpc)
UMA_CENTER_RA = 180.0   # UMa cluster center RA (degrees, approximate)
UMA_CENTER_DEC = 49.0   # UMa cluster center Dec (degrees, approximate)
UMA_MEAN_V = 950.0      # UMa cluster mean heliocentric velocity (km/s)


def rar_function(log_gbar, a0=g_dagger):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


# UMa and group classification (same as definitive test)
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


def classify_env(name):
    if name in UMA_GALAXIES:
        return 'dense', 'UMa'
    if name in GROUP_MEMBERS:
        return 'dense', GROUP_MEMBERS[name]
    return 'field', 'field'


print("=" * 72)
print("PATHOLOGICAL GALAXY ANALYSIS")
print("=" * 72)

# ================================================================
# STEP 1: Load data
# ================================================================
print("\n[1] Loading SPARC data...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

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

# Parse MRT
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
            'L36': float(parts[6]), 'Vflat': float(parts[14]), 'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

# Load CF4
cf4_path = os.path.join(PROJECT_ROOT, 'data', 'cf4', 'cf4_distance_cache.json')
with open(cf4_path, 'r') as f:
    cf4_cache = json.load(f)

print(f"  {len(galaxies)} RCs, {len(sparc_props)} properties, {len(cf4_cache)} CF4 entries")

# ================================================================
# STEP 2: Compute full-sample RAR residuals (for jackknife baseline)
# ================================================================
print("\n[2] Computing full-sample RAR residuals (SPARC distances)...")


def compute_rar_at_distance(gdata, D_use):
    """Compute RAR residuals at a given distance."""
    D_ratio = D_use / gdata['dist']
    R = gdata['R'] * D_ratio
    Vobs = gdata['Vobs']
    sqrt_ratio = np.sqrt(D_ratio)
    Vgas = gdata['Vgas'] * sqrt_ratio
    Vdisk = gdata['Vdisk'] * sqrt_ratio
    Vbul = gdata['Vbul'] * sqrt_ratio

    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < 3:
        return None, None, None

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    log_res = log_gobs - rar_function(log_gbar)
    return log_gbar, log_gobs, log_res


# Full sample with standard quality cuts
results_full = {}
for name, gdata in galaxies.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

    log_gbar, log_gobs, log_res = compute_rar_at_distance(gdata, prop['D'])
    if log_gbar is None or len(log_gbar) < 5:
        continue

    env_binary, env_group = classify_env(name)
    results_full[name] = {
        'log_gbar': log_gbar,
        'log_res': log_res,
        'env': env_binary,
        'env_group': env_group,
    }

n_dense = sum(1 for r in results_full.values() if r['env'] == 'dense')
n_field = sum(1 for r in results_full.values() if r['env'] == 'field')
print(f"  After cuts: {len(results_full)} galaxies ({n_dense} dense, {n_field} field)")

# Baseline Levene p-value
dense_res_all = np.concatenate([r['log_res'] for r in results_full.values() if r['env'] == 'dense'])
field_res_all = np.concatenate([r['log_res'] for r in results_full.values() if r['env'] == 'field'])
_, p_baseline = levene(dense_res_all, field_res_all)
delta_sigma_baseline = float(np.std(field_res_all) - np.std(dense_res_all))

# Low-acceleration bin baseline
gbar_lo, gbar_hi = -13.0, -10.5
dense_low = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                             for r in results_full.values() if r['env'] == 'dense'])
field_low = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                             for r in results_full.values() if r['env'] == 'field'])
_, p_baseline_low = levene(dense_low, field_low) if len(dense_low) >= 5 and len(field_low) >= 5 else (0, 1.0)
delta_sigma_baseline_low = float(np.std(field_low) - np.std(dense_low))

print(f"  Baseline Levene: p={p_baseline:.6f}, Δσ={delta_sigma_baseline:+.4f}")
print(f"  Baseline low-accel: p={p_baseline_low:.6f}, Δσ={delta_sigma_baseline_low:+.4f}")

# ================================================================
# STEP 3: Analyze each pathological galaxy
# ================================================================
galaxy_results = {}

for target in TARGET_GALAXIES:
    print(f"\n{'='*72}")
    print(f"GALAXY: {target}")
    print(f"{'='*72}")

    if target not in galaxies or target not in sparc_props:
        print(f"  WARNING: {target} not found in data")
        continue

    gdata = galaxies[target]
    prop = sparc_props[target]
    cf4_entry = cf4_cache.get(target, {})

    D_sparc = prop['D']
    D_cf4 = cf4_entry.get('D_cf4', None)
    D_uma = UMA_DISTANCE
    V_helio = cf4_entry.get('V_helio', cf4_entry.get('V_input', None))

    print(f"\n  Properties:")
    print(f"    SPARC D = {D_sparc:.2f} Mpc (fD={prop['fD']})")
    print(f"    CF4 D   = {D_cf4:.2f} Mpc" if D_cf4 else "    CF4 D   = N/A")
    print(f"    UMa D   = {D_uma:.1f} Mpc")
    print(f"    V_helio = {V_helio} km/s" if V_helio else "    V_helio = N/A")
    print(f"    Inc = {prop['Inc']}°, Q = {prop['Q']}, T = {prop['T']}")

    # ---- Analysis 1: RAR under 3 distances ----
    print(f"\n  [A] RAR residuals under 3 distances:")
    distances = {'SPARC': D_sparc, 'UMa_18.6': D_uma}
    if D_cf4:
        distances['CF4'] = D_cf4

    rar_results = {}
    for dist_label, D_use in distances.items():
        log_gbar, log_gobs, log_res = compute_rar_at_distance(gdata, D_use)
        if log_gbar is not None:
            rar_results[dist_label] = {
                'D_Mpc': D_use,
                'n_points': len(log_res),
                'mean_res': float(np.mean(log_res)),
                'std_res': float(np.std(log_res)),
                'min_log_gbar': float(np.min(log_gbar)),
                'max_log_gbar': float(np.max(log_gbar)),
            }
            print(f"    {dist_label:10s}: D={D_use:6.2f} Mpc, N={len(log_res):3d}, "
                  f"mean_res={np.mean(log_res):+.4f}, σ={np.std(log_res):.4f}, "
                  f"gbar range=[{np.min(log_gbar):.2f}, {np.max(log_gbar):.2f}]")

    # ---- Analysis 2: Acceleration contribution ----
    print(f"\n  [B] Acceleration bin contribution (SPARC distance):")
    log_gbar_s, _, log_res_s = compute_rar_at_distance(gdata, D_sparc)
    if log_gbar_s is not None:
        bins = [(-13.0, -11.5, 'Deep DM'),
                (-11.5, -10.5, 'Transition'),
                (-10.5, -9.5, 'Baryon-dom'),
                (-9.5, -8.0, 'High gbar')]
        accel_contrib = {}
        for lo, hi, label in bins:
            mask = (log_gbar_s >= lo) & (log_gbar_s < hi)
            n_pts = int(np.sum(mask))
            if n_pts > 0:
                mean_r = float(np.mean(log_res_s[mask]))
                std_r = float(np.std(log_res_s[mask]))
                accel_contrib[label] = {'n_points': n_pts, 'mean_res': mean_r, 'std_res': std_r}
                print(f"    {label:15s}: N={n_pts:3d}, mean_res={mean_r:+.4f}, σ={std_r:.4f}")
            else:
                accel_contrib[label] = {'n_points': 0}
                print(f"    {label:15s}: N=  0")

    # ---- Analysis 3: Jackknife influence ----
    print(f"\n  [C] Jackknife: removing {target} from environmental analysis")
    if target in results_full:
        results_jack = {n: r for n, r in results_full.items() if n != target}
        d_jack = np.concatenate([r['log_res'] for r in results_jack.values() if r['env'] == 'dense'])
        f_jack = np.concatenate([r['log_res'] for r in results_jack.values() if r['env'] == 'field'])
        _, p_jack = levene(d_jack, f_jack)
        delta_sigma_jack = float(np.std(f_jack) - np.std(d_jack))

        # Low-accel bin
        d_jack_low = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                                      for r in results_jack.values() if r['env'] == 'dense'])
        f_jack_low = np.concatenate([r['log_res'][(r['log_gbar'] >= gbar_lo) & (r['log_gbar'] < gbar_hi)]
                                      for r in results_jack.values() if r['env'] == 'field'])
        _, p_jack_low = levene(d_jack_low, f_jack_low) if len(d_jack_low) >= 5 and len(f_jack_low) >= 5 else (0, 1.0)
        delta_sigma_jack_low = float(np.std(f_jack_low) - np.std(d_jack_low))

        jackknife = {
            'overall': {
                'p_with': round(float(p_baseline), 6),
                'p_without': round(float(p_jack), 6),
                'delta_p': round(float(p_jack - p_baseline), 6),
                'delta_sigma_with': round(delta_sigma_baseline, 4),
                'delta_sigma_without': round(delta_sigma_jack, 4),
            },
            'low_accel': {
                'p_with': round(float(p_baseline_low), 6),
                'p_without': round(float(p_jack_low), 6),
                'delta_p': round(float(p_jack_low - p_baseline_low), 6),
                'delta_sigma_with': round(delta_sigma_baseline_low, 4),
                'delta_sigma_without': round(delta_sigma_jack_low, 4),
            },
        }

        print(f"    Overall: Levene p {p_baseline:.6f} → {p_jack:.6f} "
              f"(Δp={p_jack-p_baseline:+.6f})")
        print(f"    Overall: Δσ {delta_sigma_baseline:+.4f} → {delta_sigma_jack:+.4f}")
        print(f"    Low-acc: Levene p {p_baseline_low:.6f} → {p_jack_low:.6f} "
              f"(Δp={p_jack_low-p_baseline_low:+.6f})")
        print(f"    Low-acc: Δσ {delta_sigma_baseline_low:+.4f} → {delta_sigma_jack_low:+.4f}")

        influence = abs(p_jack - p_baseline) / max(p_baseline, 1e-10)
        if influence > 0.5:
            print(f"    -> HIGH influence on overall Levene p ({influence:.1%} change)")
        else:
            print(f"    -> Moderate influence ({influence:.1%} change)")
    else:
        jackknife = {'note': f'{target} not in quality-cut sample'}
        print(f"    {target} excluded by quality cuts (Q={prop['Q']}, Inc={prop['Inc']}°)")

    # ---- Analysis 4: UMa membership check ----
    print(f"\n  [D] UMa membership check:")
    ra = cf4_entry.get('RA', cf4_entry.get('ra', None))
    dec = cf4_entry.get('Dec', cf4_entry.get('dec', None))

    membership = {}
    if ra is not None and dec is not None:
        # Angular separation from UMa center
        cos_dec = np.cos(np.radians(dec))
        delta_ra = (ra - UMA_CENTER_RA) * cos_dec
        delta_dec = dec - UMA_CENTER_DEC
        ang_sep = np.sqrt(delta_ra**2 + delta_dec**2)

        membership['RA'] = round(ra, 3)
        membership['Dec'] = round(dec, 3)
        membership['angular_sep_deg'] = round(float(ang_sep), 2)
        membership['dec_offset_deg'] = round(float(delta_dec), 2)

        print(f"    RA = {ra:.3f}°, Dec = {dec:.3f}°")
        print(f"    Angular separation from UMa center: {ang_sep:.1f}°")
        print(f"    Dec offset from UMa center: {delta_dec:+.1f}°")

        if abs(delta_dec) > 15:
            membership['flag'] = 'QUESTIONABLE — far from UMa declination'
            print(f"    *** FLAG: {abs(delta_dec):.0f}° south of UMa center — membership questionable")
        else:
            membership['flag'] = 'consistent'
            print(f"    Position consistent with UMa membership")

    if V_helio is not None:
        v_offset = V_helio - UMA_MEAN_V
        membership['V_helio'] = V_helio
        membership['V_offset_kms'] = round(float(v_offset), 1)
        print(f"    V_helio = {V_helio} km/s, offset from UMa mean: {v_offset:+.0f} km/s")
        if abs(v_offset) > 500:
            membership['v_flag'] = 'QUESTIONABLE — large velocity offset'
            print(f"    *** FLAG: large velocity offset from UMa mean")
        else:
            membership['v_flag'] = 'consistent'

    # ---- Analysis 5: Virgo infall diagnostic ----
    print(f"\n  [E] Virgo infall diagnostic:")
    virgo_diag = {}

    if V_helio is not None:
        D_hubble = V_helio / H0
        virgo_diag['D_hubble_pure'] = round(D_hubble, 2)
        virgo_diag['D_cf4'] = round(D_cf4, 2) if D_cf4 else None
        virgo_diag['D_sparc'] = round(D_sparc, 2)

        print(f"    Pure Hubble: D = V_helio/H0 = {V_helio}/{H0} = {D_hubble:.1f} Mpc")
        if D_cf4:
            print(f"    CF4 flow:    D = {D_cf4:.1f} Mpc")
        print(f"    SPARC:       D = {D_sparc:.1f} Mpc")

        # Is CF4 close to pure Hubble? If so, flow model isn't correcting
        if D_cf4:
            cf4_hubble_ratio = D_cf4 / D_hubble
            virgo_diag['cf4_vs_hubble_ratio'] = round(cf4_hubble_ratio, 3)
            print(f"    CF4/Hubble ratio: {cf4_hubble_ratio:.3f}")
            if abs(cf4_hubble_ratio - 1.0) < 0.15:
                print(f"    -> CF4 ≈ pure Hubble: flow model NOT correcting for peculiar velocity")
                virgo_diag['cf4_correcting'] = False
            else:
                print(f"    -> CF4 differs from pure Hubble: flow model IS applying correction")
                virgo_diag['cf4_correcting'] = True

        # Implied peculiar velocity if SPARC distance is correct
        V_expected = H0 * D_sparc
        V_peculiar = V_helio - V_expected
        virgo_diag['V_expected_at_sparc_D'] = round(V_expected, 0)
        virgo_diag['V_peculiar_implied'] = round(V_peculiar, 0)
        print(f"    If SPARC D correct: V_expected = {V_expected:.0f} km/s, "
              f"V_pec = {V_peculiar:+.0f} km/s")

        if V_peculiar < -500:
            print(f"    -> Large negative V_pec: consistent with far-side Virgo infall")
            virgo_diag['interpretation'] = 'far-side Virgo infall'
        elif V_peculiar > 500:
            print(f"    -> Large positive V_pec: receding from us (unusual)")
            virgo_diag['interpretation'] = 'large positive peculiar velocity'
        else:
            print(f"    -> Moderate V_pec: mild flow correction needed")
            virgo_diag['interpretation'] = 'moderate peculiar velocity'

    # ---- Analysis 6: Root cause determination ----
    print(f"\n  [F] Root cause assessment:")
    root_cause = {}

    if D_cf4 and V_helio:
        # Three hypotheses:
        # (a) CF4 flow model doesn't handle Virgo infall
        # (b) SPARC Hubble-flow assignment was wrong
        # (c) Galaxy isn't actually a UMa member

        print(f"    Hypothesis A: CF4 flow model fails for Virgo infall region")
        D_hubble = V_helio / H0
        if D_cf4 and abs(D_cf4 / D_hubble - 1.0) < 0.15:
            print(f"      SUPPORTED: CF4 ≈ pure Hubble ({D_cf4:.1f} vs {D_hubble:.1f} Mpc)")
            root_cause['hyp_a_cf4_fails'] = 'SUPPORTED'
        else:
            print(f"      MIXED: CF4 differs from pure Hubble")
            root_cause['hyp_a_cf4_fails'] = 'MIXED'

        print(f"    Hypothesis B: SPARC distance is wrong")
        if prop['fD'] == 4:
            print(f"      fD=4 (UMa cluster) — distance based on cluster membership, not measured")
            print(f"      If membership is correct, SPARC D is reliable")
            root_cause['hyp_b_sparc_wrong'] = 'UNLIKELY if UMa member'
        elif prop['fD'] == 1:
            print(f"      fD=1 (Hubble flow) — subject to peculiar velocity errors")
            root_cause['hyp_b_sparc_wrong'] = 'POSSIBLE'

        print(f"    Hypothesis C: Not actually a UMa member")
        if dec is not None and abs(dec - UMA_CENTER_DEC) > 15:
            print(f"      SUPPORTED: Dec={dec:.1f}°, {abs(dec-UMA_CENTER_DEC):.0f}° from UMa center")
            root_cause['hyp_c_not_uma'] = 'SUPPORTED'
        else:
            print(f"      UNLIKELY: position consistent with UMa")
            root_cause['hyp_c_not_uma'] = 'UNLIKELY'

        # Most likely root cause
        if root_cause.get('hyp_c_not_uma') == 'SUPPORTED':
            root_cause['most_likely'] = 'Galaxy may not be a true UMa member'
        elif root_cause.get('hyp_a_cf4_fails') == 'SUPPORTED':
            root_cause['most_likely'] = 'CF4 flow model fails in Virgo infall region'
        else:
            root_cause['most_likely'] = 'Ambiguous — multiple factors possible'
        print(f"    -> MOST LIKELY: {root_cause['most_likely']}")

    galaxy_results[target] = {
        'properties': {
            'D_sparc_Mpc': D_sparc,
            'D_cf4_Mpc': D_cf4,
            'D_uma_Mpc': D_uma,
            'V_helio': V_helio,
            'fD': prop['fD'],
            'Inc': prop['Inc'],
            'Q': prop['Q'],
            'T': prop['T'],
            'Vflat': prop['Vflat'],
        },
        'rar_under_3_distances': {
            k: {key: round(v, 4) if isinstance(v, float) else v for key, v in d.items()}
            for k, d in rar_results.items()
        },
        'acceleration_contribution': accel_contrib if log_gbar_s is not None else {},
        'jackknife': jackknife,
        'membership_check': membership,
        'virgo_infall': virgo_diag,
        'root_cause': root_cause,
    }

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*72}")
print("SUMMARY — ROOT CAUSES")
print(f"{'='*72}")

for target in TARGET_GALAXIES:
    if target in galaxy_results:
        gr = galaxy_results[target]
        D_s = gr['properties']['D_sparc_Mpc']
        D_c = gr['properties']['D_cf4_Mpc']
        rc = gr['root_cause'].get('most_likely', 'N/A')
        print(f"\n  {target}:")
        print(f"    SPARC D = {D_s:.1f} Mpc, CF4 D = {D_c:.1f} Mpc (ratio = {D_c/D_s:.2f})")
        print(f"    Root cause: {rc}")

# Save results
summary = {
    'test_name': 'pathological_galaxy_analysis',
    'description': 'Deep dive into UGC06786, UGC06446, UGC06787 — galaxies causing pathological '
                   'behavior in environmental scatter analysis due to SPARC vs CF4 distance discrepancies',
    'target_galaxies': TARGET_GALAXIES,
    'baseline': {
        'n_galaxies': len(results_full),
        'n_dense': n_dense,
        'n_field': n_field,
        'levene_p_overall': round(float(p_baseline), 6),
        'delta_sigma_overall': round(delta_sigma_baseline, 4),
        'levene_p_low_accel': round(float(p_baseline_low), 6),
        'delta_sigma_low_accel': round(delta_sigma_baseline_low, 4),
    },
    'per_galaxy': galaxy_results,
}

outpath = os.path.join(RESULTS_DIR, 'summary_pathological_galaxies.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nResults saved to: {outpath}")
print("Done.")
