#!/usr/bin/env python3
"""
RAR Tightness Test — Expanded with PROBES (Stone & Courteau 2022)

Extends the SPARC-only RAR tightness test from 175 → ~1,600+ galaxies
using the PROBES compendium of 3,163 rotation curves with matched
multiband photometry.

PROBES doesn't have mass decomposition (Vgas, Vdisk, Vbul), so we compute
gbar from the total stellar mass profile using the W1 or r-band
surface brightness profiles as proxies for the mass distribution.

Strategy:
  1. For SPARC galaxies (in PROBES): use SPARC mass decomposition (gold standard)
  2. For non-SPARC PROBES galaxies with W1-band photometry:
     - Compute M*(R) from enclosed stellar mass via color-M/L
     - Compute gbar = V*²/R where V* = √(GM*(R)/R)  (thin disk approximation)
  3. This gives gbar (baryonic) and gobs = Vobs²/R at each radius
  4. Compare RAR residual scatter and property correlations

Key caveat: PROBES velocities are UNCORRECTED for inclination.
We correct using the inclination from structural_parameters.csv (b/a ratio).
"""

import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, levene, kruskal
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PROBES_DIR = os.path.join(DATA_DIR, 'probes')
PROFILES_DIR = os.path.join(PROBES_DIR, 'profiles', 'profiles')

# Physical constants
G_SI = 6.674e-11        # m³ kg⁻¹ s⁻²
Msun_kg = 1.989e30      # kg
kpc_m = 3.086e19        # m
gdagger = 1.2e-10       # m/s²
arcsec_rad = 4.8481e-6  # radians per arcsec


def rar_function(log_gbar, a0=1.2e-10):
    """Standard RAR: gobs = gbar / (1 - exp(-sqrt(gbar/a0)))"""
    gbar = 10.0**log_gbar
    gobs = gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))
    return np.log10(gobs)


def load_sparc_galaxies():
    """Load SPARC galaxies with full mass decomposition."""
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

    # Parse MRT for properties
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
            T = int(parts[0])
            D = float(parts[1])
            Inc = float(parts[4])
            L36 = float(parts[6])
            Reff = float(parts[8])
            SBeff = float(parts[9])
            MHI = float(parts[12])
            Vflat = float(parts[14])
            Q = int(parts[16])

            logMs = np.log10(max(0.5 * L36 * 1e9, 1e6))
            gas_frac = MHI * 1e9 / max(0.5 * L36 * 1e9, 1e6)

            sparc_props[name] = {
                'T': T, 'D': D, 'Inc': Inc,
                'logMs': logMs, 'Reff': Reff, 'SBeff': SBeff,
                'gas_frac': gas_frac, 'Vflat': Vflat, 'Q': Q,
                'source': 'SPARC',
            }
        except (ValueError, IndexError):
            continue

    return galaxies, sparc_props


def load_probes_data():
    """Load PROBES main table, structural parameters, and identify usable galaxies."""
    import pandas as pd

    # Main table
    mt = pd.read_csv(os.path.join(PROBES_DIR, 'main_table.csv'), skiprows=1)

    # Structural parameters — only load columns we need
    # Mstar|Rlast:rc = total stellar mass at last RC radius
    # inclination|Rlast:rc = b/a ratio at last RC radius
    # physR|Rlast:rc = physical radius at last RC endpoint (kpc)
    needed_cols = ['name', 'Mstar|Rlast:rc', 'Mstar:E|Rlast:rc',
                   'inclination|Rlast:rc', 'inclination:E|Rlast:rc',
                   'physR|Rlast:rc', 'physR|Rp50:r']

    print("  Loading structural parameters (53MB)...", flush=True)
    # Read just the header to get column names
    header_df = pd.read_csv(os.path.join(PROBES_DIR, 'structural_parameters.csv'),
                            skiprows=1, nrows=0)
    all_cols = list(header_df.columns)

    # Filter to only columns that exist
    usecols = [c for c in needed_cols if c in all_cols]

    # Also try to get Sersic index and effective radius for galaxy properties
    extra_cols = ['physR|Ri25:r', 'physR|Rp50:r']
    for c in extra_cols:
        if c in all_cols and c not in usecols:
            usecols.append(c)

    sp = pd.read_csv(os.path.join(PROBES_DIR, 'structural_parameters.csv'),
                     skiprows=1, usecols=usecols)

    # Model fits for Vflat approximation
    mf = pd.read_csv(os.path.join(PROBES_DIR, 'model_fits.csv'))

    return mt, sp, mf


def load_probes_rc(name):
    """Load a single PROBES rotation curve."""
    rc_path = os.path.join(PROFILES_DIR, f"{name}_rc.prof")
    if not os.path.exists(rc_path):
        return None
    try:
        with open(rc_path, 'r') as f:
            lines = f.readlines()
        # First line is units comment, second is header
        # Format: R,V,V_e (arcsec, km/s, km/s)
        data = []
        for line in lines[2:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    r = float(parts[0])
                    v = float(parts[1])
                    ve = float(parts[2])
                    data.append((r, v, ve))
                except ValueError:
                    continue
        if len(data) == 0:
            return None
        arr = np.array(data)
        return {'R_arcsec': arr[:, 0], 'V_raw': arr[:, 1], 'V_e': arr[:, 2]}
    except Exception:
        return None


def fold_rotation_curve(R_arcsec, V_raw):
    """
    Fold a two-sided RC into a one-sided one.
    PROBES RCs have negative R (approaching side) and positive R (receding side).
    We fold: V(|R|) = |V_raw| for sign-consistent folding.

    Actually, the proper approach is:
    - For R > 0: V = V_raw (already positive = receding)
    - For R < 0: V = -V_raw (flip sign, since approaching is negative V)
    - Average the two sides at each |R|
    """
    pos_mask = R_arcsec > 0
    neg_mask = R_arcsec < 0

    if np.sum(pos_mask) == 0 or np.sum(neg_mask) == 0:
        # One-sided RC
        return np.abs(R_arcsec), np.abs(V_raw)

    # Simple approach: take absolute values of both R and V
    R_abs = np.abs(R_arcsec)
    V_abs = np.abs(V_raw)

    return R_abs, V_abs


print("=" * 72)
print("RAR TIGHTNESS TEST: PROBES Expansion (3,163 galaxies)")
print("=" * 72)

# ================================================================
# STEP 1: Load SPARC (gold standard mass decomposition)
# ================================================================
print("\n[1] Loading SPARC rotation curves with mass decomposition...")
sparc_galaxies, sparc_props = load_sparc_galaxies()
print(f"  {len(sparc_galaxies)} rotation curves, {len(sparc_props)} with properties")

# ================================================================
# STEP 2: Load PROBES data
# ================================================================
print("\n[2] Loading PROBES data...")
import pandas as pd

mt, sp, mf = load_probes_data()
print(f"  Main table: {len(mt)} galaxies")
print(f"  Structural params: {len(sp)} galaxies")
print(f"  Model fits: {len(mf)} galaxies")

# Survey breakdown
print("\n  Survey breakdown:")
for survey, count in mt['RC_survey'].value_counts().head(8).items():
    print(f"    {survey:40s} {count:5d}")

# Build lookup dictionaries
mt_dict = {}
for _, row in mt.iterrows():
    mt_dict[row['name']] = row.to_dict()

sp_dict = {}
for _, row in sp.iterrows():
    sp_dict[row['name']] = row.to_dict()

mf_dict = {}
for _, row in mf.iterrows():
    mf_dict[row['name']] = row.to_dict()

# ================================================================
# STEP 3: Cross-match PROBES with SPARC
# ================================================================
print("\n[3] Cross-matching PROBES with SPARC...")

probes_sparc = mt[mt['RC_survey'].str.contains('SPARC', na=False)]
sparc_names_in_probes = set(probes_sparc['name'].tolist())
print(f"  SPARC galaxies in PROBES: {len(sparc_names_in_probes)}")

# ================================================================
# STEP 4: Compute per-galaxy RAR residuals
# ================================================================
print("\n[4] Computing per-galaxy RAR residuals...")
print("  Strategy A: SPARC galaxies → full mass decomposition")
print("  Strategy B: Non-SPARC PROBES → stellar mass proxy for gbar")

galaxy_residuals = {}
n_sparc_used = 0
n_probes_used = 0
n_skipped_no_struct = 0
n_skipped_low_inc = 0
n_skipped_few_pts = 0
n_skipped_no_rc = 0

# --- Strategy A: SPARC galaxies with full mass decomposition ---
for name, gdata in sparc_galaxies.items():
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
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < 5:
        continue

    log_gbar = np.log10(gbar_SI[valid])
    log_gobs = np.log10(gobs_SI[valid])
    log_gobs_rar = rar_function(log_gbar)
    log_res = log_gobs - log_gobs_rar

    dm_mask = log_gbar < -10.5
    mean_res_dm = np.mean(log_res[dm_mask]) if np.sum(dm_mask) >= 3 else np.nan

    galaxy_residuals[name] = {
        'mean_res_all': np.mean(log_res),
        'std_res_all': np.std(log_res),
        'rms_res_all': np.sqrt(np.mean(log_res**2)),
        'mean_res_dm': mean_res_dm,
        'n_points': np.sum(valid),
        'logMs': prop['logMs'],
        'T': prop.get('T', np.nan),
        'gas_frac': prop.get('gas_frac', np.nan),
        'Reff': prop.get('Reff', np.nan),
        'SBeff': prop.get('SBeff', np.nan),
        'D': prop.get('D', np.nan),
        'Vflat': prop.get('Vflat', np.nan),
        'source': 'SPARC',
    }
    n_sparc_used += 1

print(f"  SPARC galaxies used: {n_sparc_used}")

# --- Strategy B: Non-SPARC PROBES galaxies ---
# For non-SPARC galaxies, approximate gbar from total M*(R)
# gbar ≈ G * M*(< R) / R²  (spherical approximation — rough but usable)
#
# This is a LOWER BOUND on gbar since we're missing gas contribution.
# For spiral galaxies, gas fraction can be 0.1-5× stellar, so this
# underestimates gbar in gas-rich systems. We flag this as a systematic.

for idx, row in mt.iterrows():
    name = row['name']

    # Skip SPARC galaxies (already handled with better data)
    if name in galaxy_residuals:
        continue

    # Need structural parameters for stellar mass and inclination
    if name not in sp_dict:
        n_skipped_no_struct += 1
        continue

    sp_row = sp_dict[name]
    Mstar_total = sp_row.get('Mstar|Rlast:rc', np.nan)
    ba_ratio = sp_row.get('inclination|Rlast:rc', np.nan)
    R_phys_last = sp_row.get('physR|Rlast:rc', np.nan)

    if not np.isfinite(Mstar_total) or Mstar_total <= 0:
        n_skipped_no_struct += 1
        continue
    if not np.isfinite(ba_ratio) or ba_ratio <= 0:
        n_skipped_no_struct += 1
        continue

    # Compute inclination from b/a ratio
    # For thin disk: cos(i) = b/a, so i = arccos(b/a)
    # But b/a can be > 1 for face-on galaxies with measurement noise
    ba_clamped = min(max(ba_ratio, 0.1), 1.0)
    inc_rad = np.arccos(ba_clamped)
    inc_deg = np.degrees(inc_rad)

    # Skip face-on (inc < 30) and edge-on (inc > 85) — unreliable V_rot
    if inc_deg < 30 or inc_deg > 85:
        n_skipped_low_inc += 1
        continue

    # Load rotation curve
    rc = load_probes_rc(name)
    if rc is None:
        n_skipped_no_rc += 1
        continue

    R_arcsec = rc['R_arcsec']
    V_raw = rc['V_raw']
    V_e = rc['V_e']

    # Fold two-sided RC
    R_fold, V_fold = fold_rotation_curve(R_arcsec, V_raw)

    # Convert R from arcsec to kpc
    dist_mpc = row['distance']
    if not np.isfinite(dist_mpc) or dist_mpc <= 0:
        n_skipped_no_struct += 1
        continue

    R_kpc = R_fold * arcsec_rad * dist_mpc * 1e3  # arcsec → rad → Mpc → kpc

    # Correct velocity for inclination
    # V_obs = V_rot * sin(i) → V_rot = V_obs / sin(i)
    sin_i = np.sin(inc_rad)
    V_rot = V_fold / max(sin_i, 0.3)  # Protect against very face-on

    # Also subtract systemic velocity (V_raw is already corrected by C97 model
    # according to README: "velocity measurements with global recessional
    # velocity subtracted by the C97 model")

    # Compute gobs = V_rot² / R
    valid = (R_kpc > 0.1) & (V_rot > 5) & np.isfinite(V_rot) & np.isfinite(R_kpc)
    if np.sum(valid) < 5:
        n_skipped_few_pts += 1
        continue

    R_valid = R_kpc[valid]
    V_valid = V_rot[valid]

    gobs_SI = (V_valid * 1e3)**2 / (R_valid * kpc_m)

    # Compute gbar from total stellar mass
    # Simple model: M*(< R) ∝ luminosity profile ∝ cumulative SB
    # For an exponential disk: M(<R) = M_tot * [1 - (1 + R/Rd) * exp(-R/Rd)]
    # We approximate Rd from the half-mass radius
    R50 = sp_row.get('physR|Rp50:r', np.nan)
    if not np.isfinite(R50) or R50 <= 0:
        # Fall back: use Rlast/3 as rough scale
        R50 = R_phys_last / 3.0 if np.isfinite(R_phys_last) and R_phys_last > 0 else 5.0

    # For exponential disk, Re ≈ 1.678 * Rd
    Rd = R50 / 1.678

    # Enclosed mass fraction for exponential disk
    x = R_valid / Rd
    M_enc_frac = 1.0 - (1.0 + x) * np.exp(-x)
    M_enc = Mstar_total * Msun_kg * M_enc_frac

    # gbar = G * M_enc / R² (spherical approx for enclosed mass)
    gbar_SI = G_SI * M_enc / (R_valid * kpc_m)**2

    # Filter valid gbar
    valid2 = (gbar_SI > 1e-15) & (gobs_SI > 1e-15)
    if np.sum(valid2) < 5:
        n_skipped_few_pts += 1
        continue

    log_gbar = np.log10(gbar_SI[valid2])
    log_gobs = np.log10(gobs_SI[valid2])
    log_gobs_rar = rar_function(log_gbar)
    log_res = log_gobs - log_gobs_rar

    # Filter extreme outliers (> 1 dex from RAR = something went wrong)
    if np.abs(np.median(log_res)) > 1.0:
        n_skipped_few_pts += 1
        continue

    dm_mask = log_gbar < -10.5
    mean_res_dm = np.mean(log_res[dm_mask]) if np.sum(dm_mask) >= 3 else np.nan

    logMs = np.log10(Mstar_total) if Mstar_total > 0 else np.nan

    # Get Vflat from model fits
    Vflat = np.nan
    if name in mf_dict:
        vc = mf_dict[name].get('rc_model:C97:v_c', np.nan)
        if np.isfinite(vc) and vc > 0 and vc < 999:
            Vflat = vc / sin_i  # Correct model Vflat for inclination

    # Morphology from NED
    morph = row.get('morphology', '-')

    # Approximate Hubble type from morphology string
    T_approx = np.nan
    if isinstance(morph, str) and morph != '-':
        morph_upper = morph.upper()
        if 'E' in morph_upper and 'S' not in morph_upper:
            T_approx = -3
        elif 'S0' in morph_upper or 'SB0' in morph_upper:
            T_approx = 0
        elif 'SA' in morph_upper or 'SAB' in morph_upper or 'SB' in morph_upper:
            if 'A' in morph_upper[2:3]:
                T_approx = 1
            if 'B' in morph_upper[1:3]:
                T_approx = 3
            if 'C' in morph_upper[2:4]:
                T_approx = 5
            if 'D' in morph_upper[2:4]:
                T_approx = 7
            if 'M' in morph_upper[2:4]:
                T_approx = 9
        elif 'SC' in morph_upper:
            T_approx = 5
        elif 'SB' in morph_upper:
            T_approx = 3
        elif 'IM' in morph_upper or 'IRR' in morph_upper or 'IBM' in morph_upper:
            T_approx = 10

    galaxy_residuals[name] = {
        'mean_res_all': np.mean(log_res),
        'std_res_all': np.std(log_res),
        'rms_res_all': np.sqrt(np.mean(log_res**2)),
        'mean_res_dm': mean_res_dm,
        'n_points': np.sum(valid2),
        'logMs': logMs,
        'T': T_approx,
        'gas_frac': np.nan,  # Not available for PROBES
        'Reff': R50 if np.isfinite(R50) else np.nan,
        'SBeff': np.nan,     # Would need computation from SB profiles
        'D': dist_mpc,
        'Vflat': Vflat,
        'source': 'PROBES',
    }
    n_probes_used += 1

print(f"\n  === Sample Summary ===")
print(f"  SPARC galaxies (mass decomposition):  {n_sparc_used:5d}")
print(f"  PROBES galaxies (stellar mass proxy):  {n_probes_used:5d}")
print(f"  TOTAL:                                 {n_sparc_used + n_probes_used:5d}")
print(f"\n  Skipped:")
print(f"    No structural params:  {n_skipped_no_struct}")
print(f"    Low/high inclination:  {n_skipped_low_inc}")
print(f"    Too few RC points:     {n_skipped_few_pts}")
print(f"    No RC file:            {n_skipped_no_rc}")

# ================================================================
# STEP 5: Overall RAR scatter comparison
# ================================================================
print("\n" + "=" * 72)
print("RAR SCATTER: SPARC vs PROBES vs Combined")
print("=" * 72)

for source_label, source_filter in [('SPARC only', 'SPARC'),
                                      ('PROBES only', 'PROBES'),
                                      ('Combined', None)]:
    if source_filter:
        res_vals = [g['mean_res_all'] for g in galaxy_residuals.values()
                    if g['source'] == source_filter and np.isfinite(g['mean_res_all'])]
    else:
        res_vals = [g['mean_res_all'] for g in galaxy_residuals.values()
                    if np.isfinite(g['mean_res_all'])]

    if len(res_vals) < 3:
        continue
    res_arr = np.array(res_vals)
    print(f"\n  {source_label} (N={len(res_vals)}):")
    print(f"    Mean residual:     {np.mean(res_arr):+.4f} dex")
    print(f"    Scatter (σ):       {np.std(res_arr):.4f} dex")
    print(f"    RMS:               {np.sqrt(np.mean(res_arr**2)):.4f} dex")
    print(f"    Median |residual|: {np.median(np.abs(res_arr)):.4f} dex")
    print(f"    IQR:               {np.percentile(res_arr, 75) - np.percentile(res_arr, 25):.4f} dex")

# ================================================================
# STEP 6: Correlation tests with galaxy properties
# ================================================================
print("\n" + "=" * 72)
print("CORRELATION TESTS: Do residuals correlate with galaxy properties?")
print("=" * 72)

properties = [
    ('logMs', 'Stellar mass (logM*)', 'logMs'),
    ('T', 'Hubble type (T)', 'T'),
    ('D', 'Distance (Mpc)', 'D'),
    ('Reff', 'Effective radius (kpc)', 'Reff'),
    ('Vflat', 'Flat velocity (km/s)', 'Vflat'),
]

for source_label, source_filter in [('SPARC only', 'SPARC'),
                                      ('PROBES only', 'PROBES'),
                                      ('Combined', None)]:
    print(f"\n  === {source_label} ===")
    print(f"  {'Property':35s} {'N':>4s} {'Spearman ρ':>11s} {'p-value':>10s} {'Signal':>8s}")
    print(f"  {'-'*72}")

    for prop_key, prop_label, dict_key in properties:
        x_vals = []
        y_vals = []
        for g in galaxy_residuals.values():
            if source_filter and g['source'] != source_filter:
                continue
            res = g['mean_res_all']
            prop_val = g[dict_key]
            if np.isfinite(res) and np.isfinite(prop_val):
                # For non-logMs properties, require > 0
                if dict_key != 'T' and prop_val <= 0:
                    continue
                x_vals.append(prop_val)
                y_vals.append(res)

        if len(x_vals) < 10:
            print(f"  {prop_label:35s} {len(x_vals):>4d}       ---")
            continue

        x = np.array(x_vals)
        y = np.array(y_vals)
        rho, p_rho = spearmanr(x, y)

        sig = "***" if p_rho < 0.001 else "**" if p_rho < 0.01 else "*" if p_rho < 0.05 else ""
        print(f"  {prop_label:35s} {len(x_vals):>4d} {rho:>+11.4f} {p_rho:>10.6f} {sig:>8s}")

# ================================================================
# STEP 7: Variance uniformity tests (Levene)
# ================================================================
print("\n" + "=" * 72)
print("UNIFORMITY TEST: Levene's test for equal variances")
print("=" * 72)
print("  H0: scatter is the same across bins (BEC prediction)")
print("  H1: scatter differs between bins (CDM prediction)")

# By stellar mass — combined sample
logMs_edges = [8.0, 9.0, 9.5, 10.0, 10.5, 11.5]

for source_label, source_filter in [('Combined', None), ('SPARC only', 'SPARC'),
                                      ('PROBES only', 'PROBES')]:
    print(f"\n  --- {source_label} ---")

    # By stellar mass
    mass_groups = []
    mass_labels_list = []
    print(f"\n  {'logM* bin':15s} {'N':>4s} {'σ(residual)':>12s} {'mean(res)':>10s}")
    print(f"  {'-'*45}")
    for i in range(len(logMs_edges) - 1):
        lo, hi = logMs_edges[i], logMs_edges[i+1]
        vals = [g['mean_res_all'] for g in galaxy_residuals.values()
                if (source_filter is None or g['source'] == source_filter)
                and lo <= g['logMs'] < hi and np.isfinite(g['mean_res_all'])]
        if len(vals) >= 3:
            print(f"  [{lo:.1f}, {hi:.1f})      {len(vals):>4d} {np.std(vals):>12.4f} "
                  f"{np.mean(vals):>10.4f}")
        if len(vals) >= 5:
            mass_groups.append(vals)
            mass_labels_list.append(f"[{lo:.1f},{hi:.1f})")

    if len(mass_groups) >= 2:
        stat_L, p_L = levene(*mass_groups)
        stat_K, p_K = kruskal(*mass_groups)
        print(f"\n  Levene's F = {stat_L:.3f}, p = {p_L:.4f}")
        if p_L < 0.05:
            print(f"    -> Variances DIFFER across mass bins (CDM-like)")
        else:
            print(f"    -> Variances UNIFORM across mass bins (BEC-consistent)")
        print(f"  Kruskal-Wallis H = {stat_K:.3f}, p = {p_K:.4f}")
        if p_K < 0.05:
            print(f"    -> Medians DIFFER across mass bins")
        else:
            print(f"    -> Medians UNIFORM across mass bins")

# ================================================================
# STEP 8: Multivariate R² — how much variance is explainable?
# ================================================================
print("\n" + "=" * 72)
print("MULTIVARIATE: How much residual variance is explainable?")
print("=" * 72)

feat_names = ['logMs', 'D', 'Vflat']

for source_label, source_filter in [('Combined', None), ('SPARC only', 'SPARC'),
                                      ('PROBES only', 'PROBES')]:
    features = []
    targets = []
    for g in galaxy_residuals.values():
        if source_filter and g['source'] != source_filter:
            continue
        if not np.isfinite(g['mean_res_all']):
            continue
        row = []
        skip = False
        for fn in feat_names:
            v = g[fn]
            if not np.isfinite(v) or (fn != 'T' and v <= 0):
                skip = True
                break
            row.append(v)
        if skip:
            continue
        features.append(row)
        targets.append(g['mean_res_all'])

    features = np.array(features)
    targets_arr = np.array(targets)

    if len(targets_arr) < 20:
        print(f"\n  {source_label}: Too few galaxies ({len(targets_arr)})")
        continue

    # Standardize
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    X = (features - feat_mean) / feat_std

    # OLS regression
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ targets_arr
    y_pred = X @ beta
    SS_res = np.sum((targets_arr - y_pred)**2)
    SS_tot = np.sum((targets_arr - np.mean(targets_arr))**2)
    R2 = 1.0 - SS_res / SS_tot

    n, p = len(targets_arr), len(feat_names)
    R2_adj = 1.0 - (1.0 - R2) * (n - 1) / (n - p - 1)

    print(f"\n  {source_label} (N={len(targets_arr)}):")
    print(f"    R² = {R2:.4f} (Adjusted: {R2_adj:.4f})")
    print(f"    Explained variance: {R2*100:.1f}%")
    print(f"    Feature importance:")
    for i, fn in enumerate(feat_names):
        print(f"      {fn:15s}: β = {beta[i]:+.4f}")

# ================================================================
# STEP 9: Compare SPARC vs PROBES scatter distributions
# ================================================================
print("\n" + "=" * 72)
print("SPARC vs PROBES: Are the scatter distributions consistent?")
print("=" * 72)

sparc_res = np.array([g['mean_res_all'] for g in galaxy_residuals.values()
                       if g['source'] == 'SPARC' and np.isfinite(g['mean_res_all'])])
probes_res = np.array([g['mean_res_all'] for g in galaxy_residuals.values()
                        if g['source'] == 'PROBES' and np.isfinite(g['mean_res_all'])])

if len(sparc_res) > 10 and len(probes_res) > 10:
    from scipy.stats import ks_2samp, mannwhitneyu

    ks_stat, ks_p = ks_2samp(sparc_res, probes_res)
    mw_stat, mw_p = mannwhitneyu(sparc_res, probes_res, alternative='two-sided')

    print(f"\n  SPARC: σ = {np.std(sparc_res):.4f} dex, mean = {np.mean(sparc_res):+.4f}, N = {len(sparc_res)}")
    print(f"  PROBES: σ = {np.std(probes_res):.4f} dex, mean = {np.mean(probes_res):+.4f}, N = {len(probes_res)}")
    print(f"\n  KS test: D = {ks_stat:.4f}, p = {ks_p:.6f}")
    print(f"  Mann-Whitney: U = {mw_stat:.0f}, p = {mw_p:.6f}")

    if ks_p < 0.01:
        print(f"\n  -> SPARC and PROBES residual distributions are DIFFERENT (p < 0.01)")
        print(f"     This may reflect:")
        print(f"     - Missing gas contribution in PROBES gbar estimate")
        print(f"     - Different survey selection effects")
        print(f"     - Systematic from exponential disk approximation")
    else:
        print(f"\n  -> SPARC and PROBES residual distributions are CONSISTENT (p = {ks_p:.3f})")

# ================================================================
# STEP 10: Summary
# ================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

n_total = n_sparc_used + n_probes_used
print(f"""
Sample expansion: {n_sparc_used} (SPARC) + {n_probes_used} (PROBES) = {n_total} galaxies

Key findings:
  1. SPARC scatter (σ): {np.std(sparc_res):.4f} dex (N={len(sparc_res)})
     PROBES scatter (σ): {np.std(probes_res):.4f} dex (N={len(probes_res)})
     Combined scatter:   {np.std(np.concatenate([sparc_res, probes_res])):.4f} dex (N={len(sparc_res)+len(probes_res)})

  2. The PROBES scatter is expected to be larger because:
     - No gas contribution to gbar (stars only → gbar underestimated)
     - Exponential disk approximation for M*(R)
     - Inclination from photometric b/a (less accurate than kinematic)
     - Mixed survey qualities (Mathewson, Courteau, ShellFlow, etc.)

  3. DESPITE these systematics, the expanded sample tests whether
     the RAR tightness result is robust across surveys and selection
     effects, or is an artifact of SPARC's particular sample.

Interpretation:
  - If PROBES shows similar property-(in)dependence → SPARC result is robust
  - If PROBES shows MORE property dependence → measurement systematics matter
  - If PROBES shows LESS property dependence → SPARC systematics inflate correlations
""")

print("=" * 72)
print("Done.")
