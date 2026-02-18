"""
TRGB/CEPHEID QUALITY SUBSAMPLE BUILDER & RAR ANALYSIS
=====================================================
Builds a distance-equalized subsample from SPARC using only galaxies 
with TRGB, Cepheid, or SNe Ia distances (≲5-10% precision).
Then runs the full environmental RAR scatter test.
"""
import numpy as np
from scipy import stats, optimize
import json
import os
import csv
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONSTANTS
# ================================================================
g_dagger = 1.20e-10  # m/s^2 (McGaugh et al. 2016 RAR scale)
conv = 1e6 / 3.0857e19  # (km/s)^2/kpc -> m/s^2

# ================================================================
# 1. PARSE SPARC MASTER TABLE
# ================================================================
print("=" * 80)
print("STEP 1: PARSE SPARC DATABASE")
print("=" * 80)

def parse_sparc_table(filepath):
    galaxies = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        if len(line) < 50:
            continue
        parts = line.split()
        if len(parts) < 18:
            continue
        try:
            T = int(parts[1])
            D = float(parts[2])
            galaxies.append({
                'galaxy': parts[0], 'T': T, 'D': D, 'eD': float(parts[3]),
                'fD': int(parts[4]), 'Inc': float(parts[5]), 'eInc': float(parts[6]),
                'L36': float(parts[7]), 'Vflat': float(parts[15]),
                'eVflat': float(parts[16]), 'Q': int(parts[17]),
                'MHI': float(parts[13]), 'Rdisk': float(parts[11]),
                'Reff': float(parts[9]), 'RHI': float(parts[14])
            })
        except (ValueError, IndexError):
            pass
    return galaxies

sparc = parse_sparc_table('SPARC_Lelli2016c.mrt')
print(f"Parsed {len(sparc)} SPARC galaxies")

# fD codes: 1=Hubble flow, 2=TRGB, 3=Cepheid, 4=UMa cluster, 5=SNe
fD_names = {1: 'Hubble flow', 2: 'TRGB', 3: 'Cepheid', 4: 'UMa cluster', 5: 'SNe'}
from collections import Counter
for fd, count in sorted(Counter(g['fD'] for g in sparc).items()):
    print(f"  fD={fd} ({fD_names[fd]}): {count}")

# ================================================================
# 2. PARSE MASS MODELS
# ================================================================
print("\n" + "=" * 80)
print("STEP 2: PARSE MASS MODELS")
print("=" * 80)

def parse_mass_model(filepath):
    """Parse a single galaxy mass model file."""
    data = {'R': [], 'Vobs': [], 'eVobs': [], 'Vgas': [], 'Vdisk': [], 'Vbul': [], 'SBdisk': []}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                data['R'].append(float(parts[0]))
                data['Vobs'].append(float(parts[1]))
                data['eVobs'].append(float(parts[2]))
                data['Vgas'].append(float(parts[3]))
                data['Vdisk'].append(float(parts[4]))
                data['Vbul'].append(float(parts[5]))
                data['SBdisk'].append(float(parts[6]))
            except (ValueError, IndexError):
                continue
    for key in data:
        data[key] = np.array(data[key])
    return data

mass_models = {}
mass_dir = 'sparc_database/sparc_database'
if not os.path.exists(mass_dir):
    mass_dir = 'sparc_database'

for g in sparc:
    fname = f"{g['galaxy']}_rotmod.dat"
    fpath = os.path.join(mass_dir, fname)
    if os.path.exists(fpath):
        mass_models[g['galaxy']] = parse_mass_model(fpath)
    else:
        # Try alternate naming
        for trial_dir in ['sparc_database/sparc_database', 'sparc_database', '.']:
            fpath2 = os.path.join(trial_dir, fname)
            if os.path.exists(fpath2):
                mass_models[g['galaxy']] = parse_mass_model(fpath2)
                break

print(f"Loaded mass models for {len(mass_models)} galaxies")

# ================================================================
# 3. NED-D CROSS-MATCH RESULTS
# ================================================================
print("\n" + "=" * 80)
print("STEP 3: IDENTIFY QUALITY-DISTANCE SUBSAMPLE")
print("=" * 80)

# Load NED results if available
ned_results = {}
if os.path.exists('ned_distance_survey.json'):
    with open('ned_distance_survey.json') as f:
        ned_results = json.load(f)
    print(f"Loaded NED-D survey results for {len(ned_results)} galaxies")

# ================================================================
# 4. BUILD QUALITY SUBSAMPLE
# ================================================================
# Tier 1: TRGB (fD=2), Cepheid (fD=3), SNe (fD=5) — all ≲10% distance error
# Tier 2: UMa cluster (fD=4) — ~10% distance error, acceptable for secondary analysis
# Excluded: Hubble flow (fD=1) — 20-30% error, the systematic we're eliminating

# Also check NED for Hubble-flow galaxies that gained TRGB/Cepheid since SPARC publication

# Galaxies with quality distances
tier1_galaxies = []  # TRGB + Cepheid + SNe
tier2_galaxies = []  # UMa cluster

# NED upgrades: Hubble flow galaxies that now have TRGB/Cepheid in NED
ned_upgrades = []

for g in sparc:
    name = g['galaxy']
    ned = ned_results.get(name, {})
    
    if g['fD'] in [2, 3, 5]:
        # Already quality distance in SPARC
        method = {2: 'TRGB', 3: 'Cepheid', 5: 'SNe'}[g['fD']]
        tier1_galaxies.append({**g, 'dist_method': method, 'dist_tier': 1})
    elif g['fD'] == 4:
        # UMa cluster
        tier2_galaxies.append({**g, 'dist_method': 'UMa_cluster', 'dist_tier': 2})
    elif g['fD'] == 1:
        # Hubble flow — check NED for upgrade
        if ned.get('has_trgb') or ned.get('has_cepheid'):
            method = 'TRGB' if ned.get('has_trgb') else 'Cepheid'
            ned_upgrades.append({**g, 'dist_method': f'{method}(NED)', 'dist_tier': 1, 'ned_upgrade': True})

print(f"\n--- Distance Tier Classification ---")
print(f"Tier 1 (TRGB/Cepheid/SNe from SPARC): {len(tier1_galaxies)}")
print(f"  NED upgrades (Hubble flow → TRGB/Cepheid): {len(ned_upgrades)}")
print(f"Tier 2 (UMa cluster membership): {len(tier2_galaxies)}")
print(f"Excluded (Hubble flow, no upgrade): {175 - len(tier1_galaxies) - len(tier2_galaxies) - len(ned_upgrades)}")

# Combine Tier 1 + NED upgrades
all_quality = tier1_galaxies + ned_upgrades

print(f"\nNED upgrade galaxies:")
for g in ned_upgrades:
    print(f"  {g['galaxy']}: D={g['D']} Mpc, fD was {g['fD']}, now {g['dist_method']}")

# ================================================================
# 5. APPLY QUALITY CUTS
# ================================================================
print("\n" + "=" * 80)
print("STEP 4: APPLY QUALITY CUTS")
print("=" * 80)

# Same cuts as Stage 1 analysis:
# - Q ≤ 2 (medium or high quality)
# - 30° ≤ Inc ≤ 85° (avoid extreme inclinations)
# - ≥ 10 data points per galaxy
# - Galaxy has mass model available

def apply_quality_cuts(galaxy_list, label=""):
    passed = []
    reasons = Counter()
    for g in galaxy_list:
        name = g['galaxy']
        if g['Q'] > 2:
            reasons['Q=3'] += 1; continue
        if g['Inc'] < 30:
            reasons['Inc<30'] += 1; continue
        if g['Inc'] > 85:
            reasons['Inc>85'] += 1; continue
        if name not in mass_models:
            reasons['no_model'] += 1; continue
        mm = mass_models[name]
        if len(mm['R']) < 10:
            reasons['N<10'] += 1; continue
        passed.append(g)
    
    print(f"\n{label}:")
    print(f"  Input: {len(galaxy_list)}")
    print(f"  Passed: {len(passed)}")
    for reason, count in reasons.most_common():
        print(f"  Cut by {reason}: {count}")
    return passed

tier1_clean = apply_quality_cuts(all_quality, "Tier 1 (TRGB/Cepheid/SNe)")
tier2_clean = apply_quality_cuts(tier2_galaxies, "Tier 2 (UMa cluster)")

print(f"\n=== FINAL QUALITY SUBSAMPLE ===")
print(f"Tier 1: {len(tier1_clean)} galaxies (TRGB/Cepheid/SNe, ≲10% distance)")
print(f"Tier 2: {len(tier2_clean)} galaxies (UMa cluster, ~10% distance)")
print(f"Combined: {len(tier1_clean) + len(tier2_clean)} galaxies")

# ================================================================
# 6. COMPUTE RAR RESIDUALS WITH PER-GALAXY M/L
# ================================================================
print("\n" + "=" * 80)
print("STEP 5: COMPUTE RAR WITH PER-GALAXY M/L OPTIMIZATION")
print("=" * 80)

def compute_rar_residuals(gdata, Y_d=0.5, Y_b=0.7):
    """Compute RAR residuals for a galaxy given M/L ratios."""
    R = gdata['R'].copy()
    Vobs = gdata['Vobs'].copy()
    Vgas = gdata['Vgas'].copy()
    Vdisk = gdata['Vdisk'].copy()
    Vbul = gdata['Vbul'].copy()
    
    mask = R > 0
    R, Vobs = R[mask], Vobs[mask]
    Vgas, Vdisk, Vbul = Vgas[mask], Vdisk[mask], Vbul[mask]
    
    # Baryonic acceleration
    Vbar_sq = np.sign(Vgas)*Vgas**2 + Y_d * Vdisk**2 + Y_b * Vbul**2
    gbar = np.abs(Vbar_sq) / R * conv
    
    # Observed acceleration
    gobs = Vobs**2 / R * conv
    
    valid = (gobs > 0) & (gbar > 0)
    if np.sum(valid) < 3:
        return None, None, None
    
    gbar = gbar[valid]
    gobs = gobs[valid]
    
    # RAR prediction (McGaugh+2016)
    gobs_pred = gbar / (1 - np.exp(-np.sqrt(gbar / g_dagger)))
    
    # Log residuals
    log_res = np.log10(gobs) - np.log10(gobs_pred)
    log_gbar = np.log10(gbar)
    
    return log_gbar, log_res, gobs

def fit_ml_ratio(name):
    """Optimize M/L ratio to minimize RAR scatter for a galaxy."""
    gdata = mass_models[name]
    R = gdata['R'].copy()
    Vobs = gdata['Vobs'].copy()
    eVobs = gdata['eVobs'].copy()
    Vgas = gdata['Vgas'].copy()
    Vdisk = gdata['Vdisk'].copy()
    Vbul = gdata['Vbul'].copy()
    
    mask = R > 0
    R, Vobs, eVobs = R[mask], Vobs[mask], eVobs[mask]
    Vgas, Vdisk, Vbul = Vgas[mask], Vdisk[mask], Vbul[mask]
    
    if len(R) < 5:
        return None
    
    def chi2(params):
        Y_d = params[0]
        Y_b = params[1] if len(params) > 1 else 0.7
        Vbar_sq = np.sign(Vgas)*Vgas**2 + Y_d * Vdisk**2 + Y_b * Vbul**2
        gbar = np.abs(Vbar_sq) / R * conv
        gobs = Vobs**2 / R * conv
        valid = (gobs > 0) & (gbar > 0)
        if np.sum(valid) < 3: return 1e10
        gobs_pred = gbar[valid] / (1 - np.exp(-np.sqrt(gbar[valid] / g_dagger)))
        log_res = np.log10(gobs[valid]) - np.log10(gobs_pred)
        w = 1.0 / np.maximum((2 * eVobs[valid] / np.maximum(Vobs[valid], 1))**2, 0.01)
        return np.sum(w * log_res**2)
    
    has_bulge = np.any(Vbul > 0)
    if has_bulge:
        result = optimize.minimize(chi2, [0.5, 0.7], bounds=[(0.1, 1.2), (0.1, 1.5)], method='L-BFGS-B')
        return {'Y_disk': result.x[0], 'Y_bul': result.x[1]}
    else:
        result = optimize.minimize(chi2, [0.5], bounds=[(0.1, 1.2)], method='L-BFGS-B')
        return {'Y_disk': result.x[0], 'Y_bul': 0.7}

# Compute RAR for all quality subsample galaxies
def process_galaxies(galaxy_list, label=""):
    """Process a list of galaxies: fit M/L, compute residuals."""
    all_log_gbar = []
    all_log_res = []
    galaxy_results = []
    
    for g in galaxy_list:
        name = g['galaxy']
        if name not in mass_models:
            continue
        
        # Fit M/L
        ml = fit_ml_ratio(name)
        if ml is None:
            continue
        
        # Compute residuals
        log_gbar, log_res, gobs = compute_rar_residuals(
            mass_models[name], Y_d=ml['Y_disk'], Y_b=ml['Y_bul']
        )
        if log_gbar is None:
            continue
        
        all_log_gbar.extend(log_gbar)
        all_log_res.extend(log_res)
        
        galaxy_results.append({
            'galaxy': name,
            'dist_method': g['dist_method'],
            'dist_tier': g.get('dist_tier', 1),
            'D': g['D'],
            'Inc': g['Inc'],
            'Vflat': g['Vflat'],
            'Q': g['Q'],
            'T': g['T'],
            'Y_disk': ml['Y_disk'],
            'n_points': len(log_gbar),
            'mean_res': np.mean(log_res),
            'std_res': np.std(log_res),
            'log_gbar': log_gbar.tolist(),
            'log_res': log_res.tolist(),
        })
    
    log_gbar = np.array(all_log_gbar)
    log_res = np.array(all_log_res)
    
    print(f"\n{label}:")
    print(f"  Galaxies processed: {len(galaxy_results)}")
    print(f"  Total data points: {len(log_gbar)}")
    print(f"  Overall scatter: {np.std(log_res):.4f} dex")
    print(f"  Mean residual: {np.mean(log_res):.4f} dex")
    
    return galaxy_results, log_gbar, log_res

tier1_results, tier1_gbar, tier1_res = process_galaxies(tier1_clean, "Tier 1 RAR")
tier2_results, tier2_gbar, tier2_res = process_galaxies(tier2_clean, "Tier 2 RAR")

# Combined
all_results = tier1_results + tier2_results
all_gbar = np.concatenate([tier1_gbar, tier2_gbar])
all_res = np.concatenate([tier1_res, tier2_res])
print(f"\nCombined: {len(all_results)} galaxies, {len(all_gbar)} points, σ = {np.std(all_res):.4f} dex")

# ================================================================
# 7. ENVIRONMENT CLASSIFICATION
# ================================================================
print("\n" + "=" * 80)
print("STEP 6: ENVIRONMENT CLASSIFICATION")
print("=" * 80)

# Known group/cluster memberships from SPARC documentation and literature
# UMa cluster members (all fD=4 galaxies, plus some TRGB galaxies in groups)
uma_galaxies_set = set(g['galaxy'] for g in sparc if g['fD'] == 4)

# Group assignments based on SPARC documentation and NED
# Groups: M81, Sculptor, CenA, CVnI cloud, Antlia, M101
group_assignments = {
    # M81 group
    'NGC2403': 'M81_group', 'NGC2976': 'M81_group', 'IC2574': 'M81_group',
    'DDO154': 'M81_group', 'DDO168': 'M81_group', 'UGC04483': 'M81_group',
    # Sculptor group
    'NGC0300': 'Sculptor', 'NGC0055': 'Sculptor', 'NGC0247': 'Sculptor', 'NGC7793': 'Sculptor',
    # CenA group
    'NGC2915': 'CenA_group', 'UGCA442': 'CenA_group', 'ESO444-G084': 'CenA_group',
    # CVnI cloud
    'UGC07577': 'CVnI', 'UGC07232': 'CVnI', 'NGC3741': 'CVnI', 'NGC4068': 'CVnI',
    'UGC07866': 'CVnI', 'UGC07524': 'CVnI', 'UGC08490': 'CVnI', 'UGC07559': 'CVnI',
    # Antlia
    'NGC3109': 'Antlia',
    # M101 group
    'NGC5055': 'M101_group',
}

def classify_environment(g):
    """Classify galaxy as cluster/group or field."""
    name = g['galaxy']
    
    # UMa cluster members
    if name in uma_galaxies_set:
        return 'cluster', 'UMa_cluster'
    
    # Known group members
    if name in group_assignments:
        return 'group', group_assignments[name]
    
    # Distance-based heuristic: fD=1 (Hubble flow) = likely field
    if g.get('fD', g.get('sparc_fD', 1)) == 1:
        return 'field', 'field'
    
    # TRGB galaxies not in known groups — check by distance and isolation
    # D > 7 Mpc and not in a known group = likely field
    if g['D'] > 7:
        return 'field', 'field'
    
    return 'field', 'field'

# Classify all quality galaxies
for r in all_results:
    env, group_name = classify_environment(r)
    r['env_class'] = env  # 'cluster', 'group', or 'field'
    r['group_name'] = group_name

# For the environmental test, we use a binary classification:
# "dense environment" = cluster + group (embedded in DM-rich medium)
# "field" = isolated (thinner DM medium)
for r in all_results:
    r['env_binary'] = 'dense' if r['env_class'] in ['cluster', 'group'] else 'field'

env_counts = Counter(r['env_binary'] for r in all_results)
print(f"Environment classification:")
print(f"  Dense (cluster+group): {env_counts['dense']}")
print(f"  Field: {env_counts['field']}")

# Detailed breakdown
group_counts = Counter(r['group_name'] for r in all_results)
for grp, cnt in group_counts.most_common():
    print(f"    {grp}: {cnt}")

# ================================================================
# 8. ENVIRONMENTAL RAR SCATTER TEST
# ================================================================
print("\n" + "=" * 80)
print("STEP 7: ENVIRONMENTAL RAR SCATTER TEST (DISTANCE-EQUALIZED)")
print("=" * 80)

# Collect residuals by environment
dense_gbar, dense_res = [], []
field_gbar, field_res = [], []

for r in all_results:
    gbar = np.array(r['log_gbar'])
    res = np.array(r['log_res'])
    if r['env_binary'] == 'dense':
        dense_gbar.extend(gbar)
        dense_res.extend(res)
    else:
        field_gbar.extend(gbar)
        field_res.extend(res)

dense_gbar = np.array(dense_gbar)
dense_res = np.array(dense_res)
field_gbar = np.array(field_gbar)
field_res = np.array(field_res)

print(f"\nDense environment: {len(dense_res)} points from {env_counts['dense']} galaxies")
print(f"  Overall scatter: {np.std(dense_res):.4f} dex")
print(f"  Mean residual: {np.mean(dense_res):.4f} dex")

print(f"\nField environment: {len(field_res)} points from {env_counts['field']} galaxies")
print(f"  Overall scatter: {np.std(field_res):.4f} dex")
print(f"  Mean residual: {np.mean(field_res):.4f} dex")

delta_sigma = np.std(field_res) - np.std(dense_res)
print(f"\nΔσ (field - dense): {delta_sigma:.4f} dex")

# Bootstrap significance test
n_boot = 10000
np.random.seed(42)
boot_deltas = np.zeros(n_boot)
combined = np.concatenate([dense_res, field_res])
n_dense = len(dense_res)

for i in range(n_boot):
    shuffled = np.random.permutation(combined)
    boot_dense = shuffled[:n_dense]
    boot_field = shuffled[n_dense:]
    boot_deltas[i] = np.std(boot_field) - np.std(boot_dense)

p_value = np.mean(boot_deltas >= delta_sigma)
print(f"Bootstrap P(field > dense): {1-p_value:.4f} ({(1-p_value)*100:.1f}%)")
print(f"P-value (one-sided): {p_value:.4f}")

# Also test with Levene's test (more robust to non-normality)
levene_stat, levene_p = stats.levene(dense_res, field_res)
print(f"Levene's test: F={levene_stat:.3f}, p={levene_p:.4f}")

# ================================================================
# 9. BINNED ENVIRONMENTAL ANALYSIS
# ================================================================
print("\n" + "=" * 80)
print("STEP 8: BINNED ENVIRONMENTAL ANALYSIS")
print("=" * 80)

bin_edges = np.array([-12.5, -11.5, -10.5, -9.5, -8.5])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

binned_results = []
for j in range(len(bin_centers)):
    lo, hi = bin_edges[j], bin_edges[j+1]
    
    d_mask = (dense_gbar >= lo) & (dense_gbar < hi)
    f_mask = (field_gbar >= lo) & (field_gbar < hi)
    
    d_res = dense_res[d_mask]
    f_res = field_res[f_mask]
    
    if len(d_res) >= 5 and len(f_res) >= 5:
        d_std = np.std(d_res)
        f_std = np.std(f_res)
        delta = f_std - d_std
        
        # Bootstrap for this bin
        boot_bin = np.zeros(5000)
        combined_bin = np.concatenate([d_res, f_res])
        nd = len(d_res)
        for i in range(5000):
            s = np.random.permutation(combined_bin)
            boot_bin[i] = np.std(s[nd:]) - np.std(s[:nd])
        p_bin = np.mean(boot_bin >= delta)
        
        binned_results.append({
            'bin_center': bin_centers[j],
            'n_dense': len(d_res), 'n_field': len(f_res),
            'sigma_dense': d_std, 'sigma_field': f_std,
            'delta': delta, 'p_field_gt_dense': 1 - p_bin
        })
        
        print(f"  Bin {bin_centers[j]:.1f}: dense={d_std:.4f} ({len(d_res)}), "
              f"field={f_std:.4f} ({len(f_res)}), Δ={delta:+.4f}, "
              f"P(field>dense)={1-p_bin:.3f}")

# ================================================================
# 10. TIER 1 ONLY ANALYSIS (STRICTEST DISTANCE QUALITY)
# ================================================================
print("\n" + "=" * 80)
print("STEP 9: TIER 1 ONLY (TRGB/CEPHEID/SNe ONLY, NO UMa)")
print("=" * 80)

# Repeat environmental test using ONLY Tier 1 galaxies
tier1_only = [r for r in all_results if r['dist_tier'] == 1]

t1_dense_res = []
t1_field_res = []
for r in tier1_only:
    res = np.array(r['log_res'])
    if r['env_binary'] == 'dense':
        t1_dense_res.extend(res)
    else:
        t1_field_res.extend(res)

t1_dense_res = np.array(t1_dense_res)
t1_field_res = np.array(t1_field_res)

n_t1_dense = sum(1 for r in tier1_only if r['env_binary'] == 'dense')
n_t1_field = sum(1 for r in tier1_only if r['env_binary'] == 'field')

print(f"Tier 1 Dense: {len(t1_dense_res)} points from {n_t1_dense} galaxies, σ = {np.std(t1_dense_res):.4f}")
print(f"Tier 1 Field: {len(t1_field_res)} points from {n_t1_field} galaxies, σ = {np.std(t1_field_res):.4f}")

if len(t1_dense_res) > 5 and len(t1_field_res) > 5:
    t1_delta = np.std(t1_field_res) - np.std(t1_dense_res)
    print(f"Tier 1 Δσ: {t1_delta:.4f} dex")
    
    # Bootstrap
    t1_combined = np.concatenate([t1_dense_res, t1_field_res])
    t1_nd = len(t1_dense_res)
    t1_boot = np.zeros(10000)
    for i in range(10000):
        s = np.random.permutation(t1_combined)
        t1_boot[i] = np.std(s[t1_nd:]) - np.std(s[:t1_nd])
    t1_p = np.mean(t1_boot >= t1_delta)
    print(f"Tier 1 P(field > dense): {1-t1_p:.4f} ({(1-t1_p)*100:.1f}%)")

# ================================================================
# 11. MASS DEPENDENCE TEST
# ================================================================
print("\n" + "=" * 80)
print("STEP 10: MASS DEPENDENCE TEST")
print("=" * 80)

# Split by Vflat (proxy for mass)
resolved = [r for r in all_results if r['Vflat'] > 0]
if len(resolved) > 4:
    vflats = [r['Vflat'] for r in resolved]
    median_vf = np.median(vflats)
    
    low_mass_res = []
    high_mass_res = []
    for r in resolved:
        res = np.array(r['log_res'])
        if r['Vflat'] < median_vf:
            low_mass_res.extend(res)
        else:
            high_mass_res.extend(res)
    
    low_mass_res = np.array(low_mass_res)
    high_mass_res = np.array(high_mass_res)
    
    print(f"Median Vflat: {median_vf:.1f} km/s")
    print(f"Low mass: {len(low_mass_res)} points, σ = {np.std(low_mass_res):.4f}, skew = {stats.skew(low_mass_res):.2f}")
    print(f"High mass: {len(high_mass_res)} points, σ = {np.std(high_mass_res):.4f}, skew = {stats.skew(high_mass_res):.2f}")
    print(f"Δσ (low-high): {np.std(low_mass_res)-np.std(high_mass_res):.4f}")
    print(f"Δskew (high-low): {stats.skew(high_mass_res)-stats.skew(low_mass_res):+.2f}")

# ================================================================
# 12. SKEWNESS ANALYSIS
# ================================================================
print("\n" + "=" * 80)
print("STEP 11: SKEWNESS ANALYSIS BY ACCELERATION BIN")
print("=" * 80)

fine_edges = np.linspace(-12.5, -8.0, 8)
fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2

skew_results = []
for j in range(len(fine_centers)):
    lo, hi = fine_edges[j], fine_edges[j+1]
    mask = (all_gbar >= lo) & (all_gbar < hi)
    res = all_res[mask]
    
    if len(res) >= 20:
        # Quantile skewness (robust)
        q25, q50, q75 = np.percentile(res, [25, 50, 75])
        iqr = q75 - q25
        qsk = (q75 + q25 - 2*q50) / iqr if iqr > 0 else 0
        
        skew_results.append({
            'center': fine_centers[j],
            'n': len(res),
            'std': np.std(res),
            'moment_skew': stats.skew(res),
            'quantile_skew': qsk,
            'kurtosis': stats.kurtosis(res, fisher=True)
        })
        
        print(f"  Bin {fine_centers[j]:.2f}: N={len(res)}, σ={np.std(res):.4f}, "
              f"skew={stats.skew(res):.2f}, qsk={qsk:.3f}, kurt={stats.kurtosis(res, fisher=True):.1f}")

# ================================================================
# 13. SAVE COMPREHENSIVE OUTPUT
# ================================================================
print("\n" + "=" * 80)
print("STEP 12: SAVE RESULTS")
print("=" * 80)

# Save galaxy-level results
output_rows = []
for r in all_results:
    output_rows.append({
        'galaxy': r['galaxy'],
        'dist_method': r['dist_method'],
        'dist_tier': r['dist_tier'],
        'D_Mpc': r['D'],
        'Inc': r['Inc'],
        'Vflat': r['Vflat'],
        'Q': r['Q'],
        'T': r['T'],
        'Y_disk': round(r['Y_disk'], 3),
        'n_points': r['n_points'],
        'scatter': round(r['std_res'], 4),
        'mean_res': round(r['mean_res'], 4),
        'env_class': r['env_class'],
        'env_binary': r['env_binary'],
        'group_name': r['group_name']
    })

# Sort by environment, then galaxy name
output_rows.sort(key=lambda x: (x['env_binary'], x['galaxy']))

with open('quality_subsample_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
    writer.writeheader()
    writer.writerows(output_rows)
print(f"Saved {len(output_rows)} galaxy results to quality_subsample_results.csv")

# Save summary statistics
summary = {
    'n_galaxies_total': len(all_results),
    'n_tier1': len(tier1_results),
    'n_tier2': len(tier2_results),
    'n_ned_upgrades': len(ned_upgrades),
    'n_data_points': len(all_gbar),
    'overall_scatter': float(np.std(all_res)),
    'dense_scatter': float(np.std(dense_res)),
    'field_scatter': float(np.std(field_res)),
    'delta_sigma': float(delta_sigma),
    'p_field_gt_dense': float(1 - p_value),
    'levene_p': float(levene_p),
    'n_dense_galaxies': int(env_counts['dense']),
    'n_field_galaxies': int(env_counts['field']),
    'n_dense_points': len(dense_res),
    'n_field_points': len(field_res),
    'binned_results': binned_results,
    'skewness_profile': skew_results,
}
if len(t1_dense_res) > 5 and len(t1_field_res) > 5:
    summary['tier1_only'] = {
        'delta_sigma': float(t1_delta),
        'p_field_gt_dense': float(1 - t1_p),
        'dense_scatter': float(np.std(t1_dense_res)),
        'field_scatter': float(np.std(t1_field_res)),
        'n_dense': n_t1_dense,
        'n_field': n_t1_field
    }

with open('quality_subsample_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("Saved summary to quality_subsample_summary.json")

# ================================================================
# 14. COMPARISON WITH FULL SAMPLE
# ================================================================
print("\n" + "=" * 80)
print("STEP 13: COMPARISON — QUALITY SUBSAMPLE vs FULL SAMPLE")
print("=" * 80)

# Load or compute full-sample results for comparison
# Process ALL 175 galaxies (including Hubble flow) for comparison
print("\nProcessing FULL sample (all distance methods)...")
full_galaxies = [g for g in sparc 
                 if g['Q'] <= 2 
                 and 30 <= g['Inc'] <= 85 
                 and g['galaxy'] in mass_models
                 and len(mass_models[g['galaxy']]['R']) >= 10]

full_results = []
full_gbar_all = []
full_res_all = []

for g in full_galaxies:
    name = g['galaxy']
    ml = fit_ml_ratio(name)
    if ml is None: continue
    lgbar, lres, _ = compute_rar_residuals(mass_models[name], Y_d=ml['Y_disk'], Y_b=ml['Y_bul'])
    if lgbar is None: continue
    
    env, grp = classify_environment(g)
    full_results.append({
        'galaxy': name, 'fD': g['fD'], 'D': g['D'], 'Inc': g['Inc'],
        'Vflat': g['Vflat'], 'log_gbar': lgbar.tolist(), 'log_res': lres.tolist(),
        'env_binary': 'dense' if env in ['cluster', 'group'] else 'field'
    })
    full_gbar_all.extend(lgbar)
    full_res_all.extend(lres)

full_gbar_all = np.array(full_gbar_all)
full_res_all = np.array(full_res_all)

# Full sample environmental test
full_dense = np.concatenate([np.array(r['log_res']) for r in full_results if r['env_binary'] == 'dense'])
full_field = np.concatenate([np.array(r['log_res']) for r in full_results if r['env_binary'] == 'field'])

n_full_dense_gal = sum(1 for r in full_results if r['env_binary'] == 'dense')
n_full_field_gal = sum(1 for r in full_results if r['env_binary'] == 'field')

print(f"\nFull sample: {len(full_results)} galaxies, {len(full_gbar_all)} points")
print(f"  Overall scatter: {np.std(full_res_all):.4f} dex")
print(f"  Dense: {len(full_dense)} points ({n_full_dense_gal} gal), σ = {np.std(full_dense):.4f}")
print(f"  Field: {len(full_field)} points ({n_full_field_gal} gal), σ = {np.std(full_field):.4f}")
print(f"  Full Δσ: {np.std(full_field) - np.std(full_dense):.4f} dex")

print(f"\n{'='*60}")
print(f"COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Metric':<30} {'Full Sample':>15} {'Quality Sub':>15}")
print(f"{'-'*60}")
print(f"{'N galaxies':<30} {len(full_results):>15} {len(all_results):>15}")
print(f"{'N data points':<30} {len(full_gbar_all):>15} {len(all_gbar):>15}")
print(f"{'Overall σ (dex)':<30} {np.std(full_res_all):>15.4f} {np.std(all_res):>15.4f}")
print(f"{'Dense σ (dex)':<30} {np.std(full_dense):>15.4f} {np.std(dense_res):>15.4f}")
print(f"{'Field σ (dex)':<30} {np.std(full_field):>15.4f} {np.std(field_res):>15.4f}")
print(f"{'Δσ (field-dense)':<30} {np.std(full_field)-np.std(full_dense):>15.4f} {delta_sigma:>15.4f}")
print(f"{'Dense N_gal':<30} {n_full_dense_gal:>15} {env_counts['dense']:>15}")
print(f"{'Field N_gal':<30} {n_full_field_gal:>15} {env_counts['field']:>15}")

print(f"\n{'='*60}")
print(f"DISTANCE METHOD IMPACT")
print(f"{'='*60}")
# Compare scatter by distance method
for fd_val, fd_label in [(1, 'Hubble flow'), (2, 'TRGB'), (3, 'Cepheid'), (4, 'UMa'), (5, 'SNe')]:
    fd_res = []
    for r in full_results:
        if r['fD'] == fd_val:
            fd_res.extend(r['log_res'])
    if len(fd_res) > 10:
        print(f"  fD={fd_val} ({fd_label}): {len(fd_res)} points, σ = {np.std(fd_res):.4f} dex")

print("\n✓ Pipeline complete!")
