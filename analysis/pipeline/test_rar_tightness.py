#!/usr/bin/env python3
"""
RAR Tightness as a Discriminant: The Dog That Didn't Bark

The central argument:
  - CDM particles collapse into halos with DIVERSE concentration-mass
    relations, merger histories, spin parameters. Each halo is different.
    RAR residuals should correlate with galaxy properties that trace
    formation history: morphology, gas fraction, size, concentration.

  - A BEC condensate with universal coupling g† produces a TIGHT,
    UNIVERSAL RAR with scatter determined ONLY by measurement error
    and baryonic modeling uncertainty. Residuals should be pure noise
    with NO dependence on any galaxy property.

This test:
  1. Measures intrinsic RAR scatter in SPARC (cleanest single dataset)
  2. Tests whether residuals correlate with:
     a. Stellar mass (logM*)
     b. Hubble type (T)
     c. Gas fraction (MHI / M*)
     d. Effective radius (Reff)
     e. Surface brightness (SBeff)
     f. Distance (D)
  3. Quantifies how many bits of "formation history" are encoded
     in the RAR residuals
"""
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# Physical constants
kpc_m = 3.086e19
gdagger = 1.2e-10

def rar_function(log_gbar, a0=1.2e-10):
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)

print("=" * 72)
print("RAR TIGHTNESS TEST: Formation History in Residuals")
print("=" * 72)

# ================================================================
# STEP 1: Load SPARC with FULL galaxy properties
# ================================================================
print("\n[1] Loading SPARC with full galaxy properties...")

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

# Parse MRT for FULL galaxy properties
with open(mrt_path, 'r') as f:
    mrt_lines = f.readlines()
data_start = 0
for i, line in enumerate(mrt_lines):
    if line.startswith('---') and i > 50:
        data_start = i + 1
        break

sparc_props = {}
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
        T = int(parts[0])           # Hubble type
        D = float(parts[1])         # Distance (Mpc)
        eD = float(parts[2])        # Error on D
        f_D = int(parts[3])         # Distance method
        Inc = float(parts[4])       # Inclination
        eInc = float(parts[5])      # Error on inclination
        L36 = float(parts[6])       # L_[3.6] in 10^9 Lsun
        eL36 = float(parts[7])      # Error on L_[3.6]
        Reff = float(parts[8])      # Effective radius (kpc)
        SBeff = float(parts[9])     # Effective surface brightness
        Rdisk = float(parts[10])    # Disk scale length (kpc)
        SBdisk = float(parts[11])   # Disk central surface brightness
        MHI = float(parts[12])      # HI mass in 10^9 Msun
        RHI = float(parts[13])      # HI radius (kpc)
        Vflat = float(parts[14])    # Flat rotation velocity (km/s)
        eVflat = float(parts[15])   # Error on Vflat
        Q = int(parts[16])          # Quality flag

        logMs = np.log10(max(0.5 * L36 * 1e9, 1e6))
        logMHI = np.log10(max(MHI * 1e9, 1e4))
        gas_frac = MHI * 1e9 / max(0.5 * L36 * 1e9, 1e6)  # M_HI / M*

        sparc_props[name] = {
            'T': T, 'D': D, 'eD': eD, 'f_D': f_D,
            'Inc': Inc, 'eInc': eInc,
            'L36_1e9': L36, 'logMs': logMs,
            'Reff': Reff, 'SBeff': SBeff,
            'Rdisk': Rdisk, 'SBdisk': SBdisk,
            'MHI_1e9': MHI, 'logMHI': logMHI,
            'gas_frac': gas_frac,
            'RHI': RHI, 'Vflat': Vflat, 'eVflat': eVflat,
            'Q': Q,
        }
    except (ValueError, IndexError):
        continue

print(f"  {len(galaxies)} rotation curves, {len(sparc_props)} with full properties")

# ================================================================
# STEP 2: Compute per-galaxy mean RAR residual
# ================================================================
print("\n[2] Computing per-galaxy mean RAR residuals...")

galaxy_residuals = {}
for name, gdata in galaxies.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

    R = gdata['R']
    Vobs = gdata['Vobs']
    Vdisk = gdata['Vdisk']
    Vgas = gdata['Vgas']
    Vbul = gdata['Vbul']

    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0,
                        np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < 5:
        continue

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    log_gobs_rar = rar_function(log_gbar)
    log_res = log_gobs - log_gobs_rar

    # Focus on DM-dominated regime for cleaner signal
    dm_mask = log_gbar < -10.5
    if np.sum(dm_mask) >= 3:
        mean_res_dm = np.mean(log_res[dm_mask])
        std_res_dm = np.std(log_res[dm_mask])
        rms_res_dm = np.sqrt(np.mean(log_res[dm_mask]**2))
    else:
        mean_res_dm = np.nan
        std_res_dm = np.nan
        rms_res_dm = np.nan

    mean_res_all = np.mean(log_res)
    std_res_all = np.std(log_res)
    rms_res_all = np.sqrt(np.mean(log_res**2))

    galaxy_residuals[name] = {
        'mean_res_all': mean_res_all,
        'std_res_all': std_res_all,
        'rms_res_all': rms_res_all,
        'mean_res_dm': mean_res_dm,
        'std_res_dm': std_res_dm,
        'rms_res_dm': rms_res_dm,
        'n_points': np.sum(valid),
        'n_dm_points': np.sum(dm_mask),
        'logMs': prop['logMs'],
        'T': prop['T'],
        'gas_frac': prop['gas_frac'],
        'Reff': prop['Reff'],
        'SBeff': prop['SBeff'],
        'D': prop['D'],
        'Vflat': prop['Vflat'],
        'Rdisk': prop['Rdisk'],
        'logMHI': prop['logMHI'],
    }

n_gal = len(galaxy_residuals)
n_dm = sum(1 for g in galaxy_residuals.values() if not np.isnan(g['mean_res_dm']))
print(f"  {n_gal} galaxies with valid RAR residuals")
print(f"  {n_dm} galaxies with DM-regime points (log gbar < -10.5)")

# Overall scatter
all_mean_res = [g['mean_res_all'] for g in galaxy_residuals.values()]
print(f"\n  Galaxy-level RAR scatter (all gbar):")
print(f"    Mean of means: {np.mean(all_mean_res):.4f} dex")
print(f"    Std of means:  {np.std(all_mean_res):.4f} dex")
print(f"    RMS of means:  {np.sqrt(np.mean(np.array(all_mean_res)**2)):.4f} dex")

dm_mean_res = [g['mean_res_dm'] for g in galaxy_residuals.values()
               if not np.isnan(g['mean_res_dm'])]
if dm_mean_res:
    print(f"  Galaxy-level RAR scatter (DM-dominated):")
    print(f"    Mean of means: {np.mean(dm_mean_res):.4f} dex")
    print(f"    Std of means:  {np.std(dm_mean_res):.4f} dex")

# ================================================================
# STEP 3: Correlations with galaxy properties
# ================================================================
print("\n" + "=" * 72)
print("CORRELATION TESTS: Do residuals remember formation history?")
print("=" * 72)

# Properties to test
properties = [
    ('logMs', 'Stellar mass (logM*)', 'logMs'),
    ('T', 'Hubble type (T)', 'T'),
    ('gas_frac', 'Gas fraction (MHI/M*)', 'gas_frac'),
    ('Reff', 'Effective radius (kpc)', 'Reff'),
    ('SBeff', 'Surface brightness (L/pc²)', 'SBeff'),
    ('D', 'Distance (Mpc)', 'D'),
    ('Vflat', 'Flat velocity (km/s)', 'Vflat'),
    ('Rdisk', 'Disk scale length (kpc)', 'Rdisk'),
]

# Test correlations for both ALL and DM-only residuals
for res_key, res_label in [('mean_res_all', 'ALL gbar'),
                             ('mean_res_dm', 'DM-dominated')]:
    print(f"\n  --- Residuals in {res_label} regime ---")
    print(f"  {'Property':35s} {'N':>4s} {'Spearman ρ':>11s} {'p-value':>10s} "
          f"{'Pearson r':>10s} {'p-value':>10s} {'Signal':>8s}")
    print(f"  {'-'*85}")

    for prop_key, prop_label, dict_key in properties:
        x_vals = []
        y_vals = []
        for g in galaxy_residuals.values():
            res = g[res_key]
            prop_val = g[dict_key]
            if np.isfinite(res) and np.isfinite(prop_val) and prop_val > 0:
                x_vals.append(prop_val)
                y_vals.append(res)

        if len(x_vals) < 10:
            print(f"  {prop_label:35s} {len(x_vals):>4d}       ---           ---")
            continue

        x = np.array(x_vals)
        y = np.array(y_vals)

        rho, p_rho = spearmanr(x, y)
        r, p_r = pearsonr(x, y)

        # Significance flag
        if p_rho < 0.001:
            sig = "***"
        elif p_rho < 0.01:
            sig = "**"
        elif p_rho < 0.05:
            sig = "*"
        else:
            sig = ""

        print(f"  {prop_label:35s} {len(x_vals):>4d} "
              f"{rho:>+11.4f} {p_rho:>10.4f} "
              f"{r:>+10.4f} {p_r:>10.4f} {sig:>8s}")


# ================================================================
# STEP 4: Variance of residuals BY galaxy property bins
# ================================================================
print("\n" + "=" * 72)
print("SCATTER BY GALAXY PROPERTY (not just mean, but SPREAD)")
print("=" * 72)
print("  CDM predicts: scatter should vary with M*, T, etc.")
print("  BEC predicts: scatter should be UNIFORM (pure noise)")

# Stellar mass bins
logMs_edges = [8.0, 9.0, 9.5, 10.0, 10.5, 11.5]
print(f"\n  {'logM* bin':15s} {'N':>4s} {'σ(residual)':>12s} {'mean(res)':>10s}")
print(f"  {'-'*45}")
for i in range(len(logMs_edges) - 1):
    lo, hi = logMs_edges[i], logMs_edges[i+1]
    res_vals = [g['mean_res_all'] for g in galaxy_residuals.values()
                if lo <= g['logMs'] < hi and np.isfinite(g['mean_res_all'])]
    if len(res_vals) >= 3:
        print(f"  [{lo:.1f}, {hi:.1f})      {len(res_vals):>4d} {np.std(res_vals):>12.4f} "
              f"{np.mean(res_vals):>10.4f}")
    else:
        print(f"  [{lo:.1f}, {hi:.1f})      {len(res_vals):>4d}          ---")

# Hubble type bins
print(f"\n  {'Type':15s} {'N':>4s} {'σ(residual)':>12s} {'mean(res)':>10s}")
print(f"  {'-'*45}")
type_bins = [(0, 3, 'Early (S0-Sb)'), (3, 6, 'Mid (Sb-Sc)'),
             (6, 9, 'Late (Scd-Sm)'), (9, 12, 'Irr (Im-BCD)')]
for t_lo, t_hi, t_label in type_bins:
    res_vals = [g['mean_res_all'] for g in galaxy_residuals.values()
                if t_lo <= g['T'] < t_hi and np.isfinite(g['mean_res_all'])]
    if len(res_vals) >= 3:
        print(f"  {t_label:15s} {len(res_vals):>4d} {np.std(res_vals):>12.4f} "
              f"{np.mean(res_vals):>10.4f}")

# Gas fraction bins
print(f"\n  {'Gas fraction':15s} {'N':>4s} {'σ(residual)':>12s} {'mean(res)':>10s}")
print(f"  {'-'*45}")
gf_edges = [0.0, 0.3, 1.0, 3.0, 100.0]
gf_labels = ['Gas-poor (<0.3)', 'Moderate (0.3-1)', 'Gas-rich (1-3)', 'Gas-dominated (>3)']
for i in range(len(gf_edges) - 1):
    lo, hi = gf_edges[i], gf_edges[i+1]
    res_vals = [g['mean_res_all'] for g in galaxy_residuals.values()
                if lo <= g['gas_frac'] < hi and np.isfinite(g['mean_res_all'])]
    if len(res_vals) >= 3:
        print(f"  {gf_labels[i]:15s} {len(res_vals):>4d} {np.std(res_vals):>12.4f} "
              f"{np.mean(res_vals):>10.4f}")

# ================================================================
# STEP 5: The key test — is scatter UNIFORM across properties?
# ================================================================
print("\n" + "=" * 72)
print("UNIFORMITY TEST: Levene's test for equal variances")
print("=" * 72)
print("  H0: scatter is the same across bins (BEC prediction)")
print("  H1: scatter differs between bins (CDM prediction)")

from scipy.stats import levene, kruskal

# By stellar mass
mass_groups = []
mass_labels = []
for i in range(len(logMs_edges) - 1):
    lo, hi = logMs_edges[i], logMs_edges[i+1]
    vals = [g['mean_res_all'] for g in galaxy_residuals.values()
            if lo <= g['logMs'] < hi and np.isfinite(g['mean_res_all'])]
    if len(vals) >= 5:
        mass_groups.append(vals)
        mass_labels.append(f"[{lo:.1f},{hi:.1f})")

if len(mass_groups) >= 2:
    # Levene's test for equal variances
    stat_L, p_L = levene(*mass_groups)
    # Kruskal-Wallis for equal medians
    stat_K, p_K = kruskal(*mass_groups)
    print(f"\n  By stellar mass ({len(mass_groups)} groups):")
    print(f"    Levene's F = {stat_L:.3f}, p = {p_L:.4f}")
    if p_L < 0.05:
        print(f"    ⚠ Variances DIFFER across mass bins (CDM-like)")
    else:
        print(f"    ✓ Variances UNIFORM across mass bins (BEC-consistent)")
    print(f"    Kruskal-Wallis H = {stat_K:.3f}, p = {p_K:.4f}")
    if p_K < 0.05:
        print(f"    ⚠ Medians DIFFER across mass bins (residuals mass-dependent)")
    else:
        print(f"    ✓ Medians UNIFORM across mass bins (no mass dependence)")

# By Hubble type
type_groups = []
for t_lo, t_hi, t_label in type_bins:
    vals = [g['mean_res_all'] for g in galaxy_residuals.values()
            if t_lo <= g['T'] < t_hi and np.isfinite(g['mean_res_all'])]
    if len(vals) >= 5:
        type_groups.append(vals)

if len(type_groups) >= 2:
    stat_L, p_L = levene(*type_groups)
    stat_K, p_K = kruskal(*type_groups)
    print(f"\n  By Hubble type ({len(type_groups)} groups):")
    print(f"    Levene's F = {stat_L:.3f}, p = {p_L:.4f}")
    if p_L < 0.05:
        print(f"    ⚠ Variances DIFFER across morphologies (CDM-like)")
    else:
        print(f"    ✓ Variances UNIFORM across morphologies (BEC-consistent)")
    print(f"    Kruskal-Wallis H = {stat_K:.3f}, p = {p_K:.4f}")
    if p_K < 0.05:
        print(f"    ⚠ Medians DIFFER across morphologies")
    else:
        print(f"    ✓ Medians UNIFORM across morphologies")

# By gas fraction
gf_groups = []
for i in range(len(gf_edges) - 1):
    lo, hi = gf_edges[i], gf_edges[i+1]
    vals = [g['mean_res_all'] for g in galaxy_residuals.values()
            if lo <= g['gas_frac'] < hi and np.isfinite(g['mean_res_all'])]
    if len(vals) >= 5:
        gf_groups.append(vals)

if len(gf_groups) >= 2:
    stat_L, p_L = levene(*gf_groups)
    stat_K, p_K = kruskal(*gf_groups)
    print(f"\n  By gas fraction ({len(gf_groups)} groups):")
    print(f"    Levene's F = {stat_L:.3f}, p = {p_L:.4f}")
    if p_L < 0.05:
        print(f"    ⚠ Variances DIFFER across gas fractions (CDM-like)")
    else:
        print(f"    ✓ Variances UNIFORM across gas fractions (BEC-consistent)")
    print(f"    Kruskal-Wallis H = {stat_K:.3f}, p = {p_K:.4f}")


# ================================================================
# STEP 6: Multivariate — what fraction of residual variance is
#          explainable by galaxy properties?
# ================================================================
print("\n" + "=" * 72)
print("MULTIVARIATE: How much residual variance is explainable?")
print("=" * 72)
print("  If residuals are pure noise → R² ≈ 0")
print("  If residuals encode formation history → R² > 0")

# Build feature matrix
features = []
targets = []
feat_names = ['logMs', 'T', 'gas_frac', 'Reff', 'SBeff', 'Vflat']

for g in galaxy_residuals.values():
    if not np.isfinite(g['mean_res_all']):
        continue
    row = []
    skip = False
    for fn in feat_names:
        v = g[fn]
        if not np.isfinite(v) or v <= 0:
            skip = True
            break
        row.append(v)
    if skip:
        continue
    features.append(row)
    targets.append(g['mean_res_all'])

features = np.array(features)
targets = np.array(targets)
print(f"\n  {len(targets)} galaxies with complete property set")

if len(targets) >= 20:
    # Standardize
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    X = (features - feat_mean) / feat_std

    # Simple multivariate regression
    # R² = fraction of variance explained
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ targets
    y_pred = X @ beta
    SS_res = np.sum((targets - y_pred)**2)
    SS_tot = np.sum((targets - np.mean(targets))**2)
    R2 = 1.0 - SS_res / SS_tot

    # Adjusted R²
    n, p = len(targets), len(feat_names)
    R2_adj = 1.0 - (1.0 - R2) * (n - 1) / (n - p - 1)

    print(f"\n  Linear regression R² = {R2:.4f}")
    print(f"  Adjusted R² = {R2_adj:.4f}")
    print(f"  Fraction of residual variance explained: {R2*100:.1f}%")
    print(f"  Fraction UNEXPLAINED (measurement + intrinsic noise): {(1-R2)*100:.1f}%")

    print(f"\n  Feature importance (|β|):")
    for i, fn in enumerate(feat_names):
        print(f"    {fn:15s}: β = {beta[i]:+.4f}")

    if R2 < 0.05:
        print(f"\n  ✓ Residuals are essentially PURE NOISE (R² < 5%)")
        print(f"    No detectable formation-history information")
        print(f"    Consistent with universal coupling (BEC/MOND)")
    elif R2 < 0.15:
        print(f"\n  ~ Weak property dependence (R² = {R2:.1%})")
        print(f"    Some formation-history information detected")
    else:
        print(f"\n  ⚠ Significant property dependence (R² = {R2:.1%})")
        print(f"    Residuals encode formation history (CDM-like)")


# ================================================================
# STEP 7: The punchline
# ================================================================
print("\n" + "=" * 72)
print("THE ARGUMENT")
print("=" * 72)
print("""
If dark matter is particles in halos:
  - Each halo has different concentration, spin, merger history
  - The M*-M_halo relation has ~0.2 dex scatter (Behroozi+2013)
  - RAR residuals should correlate with galaxy properties
  - Especially: morphology (merger history), gas fraction (assembly state)

If dark matter is a condensate with universal g†:
  - The RAR is an identity: g_DM = g_bar / [exp(√(gbar/g†)) - 1]
  - g† is a fundamental constant, not a per-galaxy parameter
  - Residuals are pure measurement error
  - NO correlation with any galaxy property

The RAR's observed intrinsic scatter of ~0.06-0.10 dex is ALREADY
remarkably tight given measurement uncertainties. The question is
whether this residual scatter encodes any galaxy-property information
(CDM) or is featureless noise (BEC/universal coupling).
""")

print("=" * 72)
print("Done.")
