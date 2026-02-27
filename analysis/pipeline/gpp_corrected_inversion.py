#!/usr/bin/env python3
"""
GPP CORRECTED INVERSION — Paper-Ready Microphysics
====================================================

Implements the correct Gross-Pitaevskii-Poisson (GPP) microphysical relations
for the BEC dark matter framework.

Starting from GPP in physical units:
    i*hbar * d(psi)/dt = [-(hbar²/2m)*∇² + m*Phi + g_int*|psi|²] * psi
    ∇²Phi = 4*pi*G*rho,  rho = m*|psi|²

Contact interaction:  g_int = 4*pi*hbar²*a_s / m
Thomas-Fermi limit:   P = (g_int / 2m²) rho²
Sound speed:          c_s² = (g_int/m²) rho = (g_int/m) n
Healing length:       xi = hbar / sqrt(2 m g_int n)
Acceleration scale:   g* = c_s² / xi

Universality condition: g* ≈ g† requires (a_s n)^(3/2) / m² ≈ const

Outputs:
  - gpp_corrected_summary.json
  - galaxy_gpp_results.csv
  - 4 publication figures (dark background)
"""

import os
import sys
import json
import csv
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, norm
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'gpp_corrected')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# CONSTANTS (SI)
# ============================================================
G_SI    = 6.674e-11      # m³ kg⁻¹ s⁻²
hbar    = 1.054572e-34   # J s
g_dag   = 1.2e-10        # m/s² (empirical BEC scale)
KPC_M   = 3.085678e19    # m per kpc
KG_EV   = 1.782662e-36   # kg per eV/c²
MSUN_KG = 1.989e30       # kg per solar mass

# Quality cuts
MIN_POINTS = 6
MIN_VALID_RHO = 3
SAVGOL_WINDOW = 5
SAVGOL_POLY = 2

print("=" * 72)
print("GPP CORRECTED INVERSION — Paper-Ready Microphysics")
print("=" * 72)

# ============================================================
# STEP 0: Load data
# ============================================================
print("\n  STEP 0: Loading unified RAR data...")

rar_path = os.path.join(RESULTS_DIR, 'rar_points_unified.csv')
gal_path = os.path.join(RESULTS_DIR, 'galaxy_results_unified.csv')

if not os.path.exists(rar_path):
    print(f"ERROR: {rar_path} not found!")
    sys.exit(1)

df_pts = pd.read_csv(rar_path)
df_gal = pd.read_csv(gal_path)

print(f"    Loaded {len(df_pts)} RAR points from {df_pts['galaxy'].nunique()} galaxies")
print(f"    Sources: {df_pts['source'].unique().tolist()}")

# ============================================================
# PART 1: CORRECTED DENSITY INVERSION
# ============================================================
print("\n" + "=" * 72)
print("  PART 1: Corrected Spherical Density Inversion")
print("  NOTE: Effective spherical density — disk galaxy approximation")
print("=" * 72)

galaxy_results = []
all_profiles = {}  # Store per-galaxy profiles for plotting

galaxies = df_pts.groupby('galaxy')

n_total = 0
n_valid = 0
n_too_few = 0
n_all_neg_gdm = 0
total_neg_gdm_frac = []

for gname, gdf in galaxies:
    n_total += 1
    gdf = gdf.sort_values('R_kpc').reset_index(drop=True)
    npts = len(gdf)

    if npts < MIN_POINTS:
        n_too_few += 1
        continue

    # Get metadata
    source = gdf['source'].iloc[0]
    env_dense = gdf['env_dense'].iloc[0]
    logMh = gdf['logMh'].iloc[0]

    # Step 1: Compute g_DM
    gbar = 10.0 ** gdf['log_gbar'].values   # m/s²
    gobs = 10.0 ** gdf['log_gobs'].values   # m/s²
    gDM = gobs - gbar                        # m/s²

    neg_mask = gDM <= 0
    neg_frac = neg_mask.sum() / len(gDM)
    total_neg_gdm_frac.append(neg_frac)

    # Work with positive gDM points only
    valid = ~neg_mask
    if valid.sum() < MIN_VALID_RHO + 2:
        n_all_neg_gdm += 1
        continue

    r_kpc = gdf['R_kpc'].values[valid]
    gDM_v = gDM[valid]
    gbar_v = gbar[valid]
    gobs_v = gobs[valid]

    if len(r_kpc) < MIN_POINTS:
        continue

    # Step 2: Spherical density inversion
    # rho_DM(r) = (1/4piG) * (1/r²) * d/dr(r² * gDM)
    r_m = r_kpc * KPC_M   # meters

    y = r_m**2 * gDM_v    # r² * g_DM

    # Smooth before differentiation — guard against NaN/Inf
    y_finite = np.isfinite(y)
    if y_finite.sum() < MIN_POINTS:
        continue
    if not y_finite.all():
        # Interpolate over bad values
        good_idx = np.where(y_finite)[0]
        bad_idx = np.where(~y_finite)[0]
        y[bad_idx] = np.interp(bad_idx, good_idx, y[good_idx])

    n_sg = len(y)
    ww = min(SAVGOL_WINDOW, n_sg if n_sg % 2 == 1 else n_sg - 1)
    if ww >= 3:
        pp = min(SAVGOL_POLY, ww - 1)
        try:
            y_smooth = savgol_filter(y, ww, pp)
        except (ValueError, np.linalg.LinAlgError):
            y_smooth = y.copy()
    else:
        y_smooth = y.copy()

    # Numerical derivative dy/dr
    dy_dr = np.gradient(y_smooth, r_m)

    # rho_DM = (1/4piG) * dy_dr / r²
    rho_DM = dy_dr / (4.0 * np.pi * G_SI * r_m**2)  # kg/m³

    # Step 3: Sound speed from hydrostatic TF closure
    # c_s²(r) = gDM(r) / |d(ln rho)/dr|
    # where d(ln rho)/dr = (1/rho) * d(rho)/dr

    # Smooth rho before taking log derivative
    rho_pos = rho_DM > 0
    if rho_pos.sum() < MIN_VALID_RHO:
        n_all_neg_gdm += 1
        continue

    ln_rho = np.full_like(rho_DM, np.nan)
    ln_rho[rho_pos] = np.log(rho_DM[rho_pos])

    # Interpolate over gaps for derivative
    valid_ln = np.isfinite(ln_rho)
    if valid_ln.sum() < MIN_VALID_RHO:
        continue

    # Use only contiguous valid points for derivative
    r_m_val = r_m[valid_ln]
    ln_rho_val = ln_rho[valid_ln]

    if len(r_m_val) >= 3:
        n_lr = len(ln_rho_val)
        ww2 = min(SAVGOL_WINDOW, n_lr if n_lr % 2 == 1 else n_lr - 1)
        if ww2 >= 3 and np.all(np.isfinite(ln_rho_val)):
            pp2 = min(SAVGOL_POLY, ww2 - 1)
            try:
                ln_rho_smooth = savgol_filter(ln_rho_val, ww2, pp2)
            except (ValueError, np.linalg.LinAlgError):
                ln_rho_smooth = ln_rho_val.copy()
        else:
            ln_rho_smooth = ln_rho_val.copy()
    else:
        ln_rho_smooth = ln_rho_val.copy()

    dln_rho_dr = np.gradient(ln_rho_smooth, r_m_val)

    # L_rho = |d(ln rho)/dr|^(-1)   [TF diagnostic scale]
    abs_dln = np.abs(dln_rho_dr)
    L_rho_m = np.where(abs_dln > 0, 1.0 / abs_dln, np.inf)
    L_rho_kpc = L_rho_m / KPC_M

    # c_s² = gDM / |d(ln rho)/dr|
    gDM_at_valid = gDM_v[valid_ln]
    c_s2 = np.where(abs_dln > 0, gDM_at_valid / abs_dln, np.nan)

    # Filter: keep only positive, finite c_s2
    cs2_good = np.isfinite(c_s2) & (c_s2 > 0)

    if cs2_good.sum() < MIN_VALID_RHO:
        continue

    c_s2_valid = c_s2[cs2_good]
    r_kpc_cs2 = r_m_val[cs2_good] / KPC_M

    # Step 4: Regime parameter X (corrected label)
    # X(r) = sqrt(g†/gDM) — empirical acceleration-regime selector
    # NOT a microphysical TF validity test
    gDM_cs2 = gDM_at_valid[cs2_good]
    X_regime = np.sqrt(g_dag / gDM_cs2)

    # xi_eff(r) = r * sqrt(gDM / g†)
    xi_eff_m = r_m_val[cs2_good] * np.sqrt(gDM_cs2 / g_dag)
    xi_eff_kpc = xi_eff_m / KPC_M

    # Step 5: TF validity diagnostic
    L_rho_cs2 = L_rho_kpc[cs2_good]

    # Step 6: Per-galaxy summary
    log_cs2 = np.log10(c_s2_valid)
    cs2_median = np.median(c_s2_valid)
    cs2_scatter_dex = 1.4826 * np.median(np.abs(log_cs2 - np.median(log_cs2)))

    xi_eff_med_kpc = np.median(xi_eff_kpc)
    L_rho_med_kpc = np.median(L_rho_cs2[np.isfinite(L_rho_cs2)])

    n_valid += 1
    galaxy_results.append({
        'galaxy': gname,
        'source': source,
        'logMh': float(logMh) if np.isfinite(logMh) else 0.0,
        'env_dense': env_dense,
        'c_s2_median': float(cs2_median),
        'c_s2_scatter_dex': float(cs2_scatter_dex),
        'xi_eff_median_kpc': float(xi_eff_med_kpc),
        'L_rho_median_kpc': float(L_rho_med_kpc) if np.isfinite(L_rho_med_kpc) else -1.0,
        'n_valid': int(cs2_good.sum()),
        'neg_gDM_frac': float(neg_frac),
        'n_total_pts': npts,
        'log_c_s2_median': float(np.log10(cs2_median)),
    })

    # Store profiles for plotting
    all_profiles[gname] = {
        'r_kpc': r_kpc_cs2.tolist(),
        'c_s2': c_s2_valid.tolist(),
        'xi_eff_kpc': xi_eff_kpc.tolist(),
        'L_rho_kpc': L_rho_cs2.tolist(),
        'X_regime': X_regime.tolist(),
        'logMh': float(logMh) if np.isfinite(logMh) else 0.0,
        'source': source,
    }

print(f"\n    Total galaxies: {n_total}")
print(f"    Too few points (<{MIN_POINTS}): {n_too_few}")
print(f"    All negative gDM / insufficient valid: {n_all_neg_gdm}")
print(f"    Valid for inversion: {n_valid}")
print(f"    Median neg_gDM fraction: {np.median(total_neg_gdm_frac):.3f}")

if n_valid < 20:
    print(f"\nERROR: Only {n_valid} galaxies passed quality cuts (need ≥20). Aborting.")
    sys.exit(1)

# Convert to DataFrame
df_results = pd.DataFrame(galaxy_results)

# ============================================================
# PART 2: UNIVERSALITY TESTS
# ============================================================
print("\n" + "=" * 72)
print("  PART 2: Universality Tests")
print("=" * 72)

# Filter for valid logMh
df_test = df_results[df_results['logMh'] > 0].copy()
log_cs2 = df_test['log_c_s2_median'].values
logMh = df_test['logMh'].values

# --- Test A: Mass trend ---
print("\n  Test A: c_s² vs logMh (mass trend)")
mask_A = np.isfinite(log_cs2) & np.isfinite(logMh) & (logMh > 0)
x_A = logMh[mask_A]
y_A = log_cs2[mask_A]

if len(x_A) > 10:
    rho_mass, p_mass = spearmanr(x_A, y_A)
    # Linear regression
    coeffs = np.polyfit(x_A, y_A, 1)
    slope_mass = coeffs[0]
    intercept_mass = coeffs[1]
    # Slope standard error
    y_pred = np.polyval(coeffs, x_A)
    residuals = y_A - y_pred
    se_slope = np.sqrt(np.sum(residuals**2) / (len(x_A) - 2) / np.sum((x_A - x_A.mean())**2))

    print(f"    N = {len(x_A)}")
    print(f"    Spearman ρ = {rho_mass:.4f}, p = {p_mass:.4e}")
    print(f"    Slope = {slope_mass:.4f} ± {se_slope:.4f} dex/dex")
    print(f"    Scatter (MAD) = {1.4826 * np.median(np.abs(y_A - np.median(y_A))):.3f} dex")
else:
    rho_mass, p_mass, slope_mass, se_slope = 0, 1, 0, 999

cs2_scatter_all = 1.4826 * np.median(np.abs(y_A - np.median(y_A)))
cs2_median_all = np.median(10.0**y_A)

# Universality verdict
if abs(slope_mass) < 0.05 and cs2_scatter_all < 0.25:
    univ_verdict = "CONFIRMED"
elif abs(slope_mass) < 0.10 and cs2_scatter_all < 0.40:
    univ_verdict = "MARGINAL"
else:
    univ_verdict = "DISFAVORED"

print(f"    Verdict: {univ_verdict}")

# --- Test B: Environment split ---
print("\n  Test B: Field vs Dense environment")
field_mask = df_test['env_dense'] == 'field'
dense_mask = df_test['env_dense'] == 'dense'

cs2_field = df_test.loc[field_mask, 'log_c_s2_median'].dropna().values
cs2_dense = df_test.loc[dense_mask, 'log_c_s2_median'].dropna().values

if len(cs2_field) > 5 and len(cs2_dense) > 5:
    mwu_stat, mwu_p = mannwhitneyu(cs2_field, cs2_dense, alternative='two-sided')
    med_field = np.median(cs2_field)
    med_dense = np.median(cs2_dense)
    delta_env = med_field - med_dense
    print(f"    N_field = {len(cs2_field)}, N_dense = {len(cs2_dense)}")
    print(f"    Median field = {med_field:.3f}, dense = {med_dense:.3f}")
    print(f"    Δ = {delta_env:.3f} dex")
    print(f"    Mann-Whitney p = {mwu_p:.4e}")
else:
    mwu_p = 1.0
    med_field = np.median(cs2_field) if len(cs2_field) > 0 else 0
    med_dense = np.median(cs2_dense) if len(cs2_dense) > 0 else 0
    delta_env = med_field - med_dense

# --- Test C: Survey systematics ---
print("\n  Test C: Survey systematics")
survey_medians = {}
for src in df_test['source'].unique():
    src_data = df_test.loc[df_test['source'] == src, 'log_c_s2_median'].dropna().values
    if len(src_data) >= 5:
        survey_medians[src] = {
            'n': len(src_data),
            'median': float(np.median(src_data)),
            'scatter': float(1.4826 * np.median(np.abs(src_data - np.median(src_data)))),
        }
        print(f"    {src}: N={len(src_data)}, median log(c_s²)={np.median(src_data):.3f} ± {survey_medians[src]['scatter']:.3f}")

# Survey offset
if 'SPARC' in survey_medians and len(survey_medians) > 1:
    sparc_med = survey_medians['SPARC']['median']
    for src, vals in survey_medians.items():
        if src != 'SPARC':
            offset = vals['median'] - sparc_med
            print(f"    Offset {src} - SPARC = {offset:.3f} dex")

# --- Test D: Confound check ---
print("\n  Test D: Partial correlation (c_s² vs logMh | R_ext)")
# Get R_ext per galaxy from max R_kpc in point data
rext_map = df_pts.groupby('galaxy')['R_kpc'].max().to_dict()
df_test['log_Rext'] = df_test['galaxy'].map(rext_map).apply(lambda x: np.log10(x) if x and x > 0 else np.nan)

mask_D = np.isfinite(df_test['log_c_s2_median']) & np.isfinite(df_test['logMh']) & np.isfinite(df_test['log_Rext']) & (df_test['logMh'] > 0)
if mask_D.sum() > 15:
    x1 = df_test.loc[mask_D, 'logMh'].values
    x2 = df_test.loc[mask_D, 'log_Rext'].values
    y_d = df_test.loc[mask_D, 'log_c_s2_median'].values

    # Partial correlation: regress y on x2, regress x1 on x2, correlate residuals
    c1 = np.polyfit(x2, y_d, 1)
    c2 = np.polyfit(x2, x1, 1)
    res_y = y_d - np.polyval(c1, x2)
    res_x = x1 - np.polyval(c2, x2)
    partial_rho, partial_p = spearmanr(res_x, res_y)
    print(f"    Partial Spearman ρ(c_s² | R_ext) = {partial_rho:.4f}, p = {partial_p:.4e}")
    print(f"    {'CONFOUNDED' if abs(partial_rho) < 0.05 else 'INDEPENDENT'}")
else:
    partial_rho, partial_p = 0, 1
    print("    Insufficient data for partial correlation")

# ============================================================
# PART 3: BOSON MASS ESTIMATE
# ============================================================
print("\n" + "=" * 72)
print("  PART 3: Boson Mass Estimate (Paper 2 setup)")
print("=" * 72)

# Collect all valid point-level c_s2 and xi_eff values
all_cs2_points = []
all_xi_eff_points = []

for gname, prof in all_profiles.items():
    all_cs2_points.extend(prof['c_s2'])
    all_xi_eff_points.extend([x * KPC_M for x in prof['xi_eff_kpc']])  # convert to meters

all_cs2_points = np.array(all_cs2_points)
all_xi_eff_points = np.array(all_xi_eff_points)

# Filter valid
valid_bm = (all_cs2_points > 0) & np.isfinite(all_cs2_points) & (all_xi_eff_points > 0) & np.isfinite(all_xi_eff_points)
cs2_pts = all_cs2_points[valid_bm]
xi_pts = all_xi_eff_points[valid_bm]

C_median = np.median(cs2_pts)
xi_median = np.median(xi_pts)

# m = hbar / (sqrt(2*C) * xi)
m_kg = hbar / (np.sqrt(2.0 * C_median) * xi_median)
m_eV = m_kg / KG_EV

print(f"    Median c_s² = {C_median:.4e} m²/s²")
print(f"    Median ξ_eff = {xi_median:.4e} m = {xi_median/KPC_M:.2f} kpc")
print(f"    Boson mass m = {m_kg:.4e} kg = {m_eV:.4e} eV/c²")

# Bootstrap CI
n_boot = 2000
boot_masses = np.zeros(n_boot)
rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(len(cs2_pts), size=len(cs2_pts), replace=True)
    C_b = np.median(cs2_pts[idx])
    xi_b = np.median(xi_pts[idx])
    if C_b > 0 and xi_b > 0:
        boot_masses[i] = hbar / (np.sqrt(2.0 * C_b) * xi_b) / KG_EV
    else:
        boot_masses[i] = np.nan

boot_valid = boot_masses[np.isfinite(boot_masses)]
ci_lo = np.percentile(boot_valid, 2.5)
ci_hi = np.percentile(boot_valid, 97.5)

print(f"    Bootstrap 95% CI: [{ci_lo:.4e}, {ci_hi:.4e}] eV/c²")

# Lyman-alpha check
lya_bound = 1e-21  # eV lower bound for fuzzy DM (no self-interaction)
lya_compatible = m_eV > lya_bound
print(f"    Lyman-α bound (1e-21 eV, no SI): {'compatible' if lya_compatible else 'BELOW BOUND'}")
print(f"    Note: GPP with g_int > 0 relaxes this constraint")

# Interaction parameter
g_int = C_median * m_kg**2  # J m³ (from c_s² = g_int*n/m, combined)
lambda_coupling = g_int * m_kg**2 / (4.0 * np.pi * hbar**4)

print(f"    g_int estimate = {g_int:.4e} J·m³")
print(f"    Dimensionless coupling λ = {lambda_coupling:.4e}")

# ============================================================
# PART 4: CAVEATS
# ============================================================
caveats = [
    ("X(r) is an acceleration-regime proxy, not a microphysical TF validity test. "
     "True TF validity requires L_rho >> xi_micro which cannot be verified "
     "without knowing boson mass m and scattering length a_s independently."),
    ("Density inversion assumes spherical symmetry. Disk galaxies violate this. "
     "All rho_DM values are effective spherical approximations."),
    ("The RAR-occupation number mapping g_DM/g_bar = 1/(exp(epsilon)-1) is a "
     "structural analogy. Deriving it from GPP fluid equilibrium is Paper 2."),
    ("Boson mass estimate assumes TF regime holds at median galaxy scales. "
     "If quantum pressure is significant, xi_emp overestimates true xi and "
     "m is underestimated."),
    ("c_s² universality requires (a_s * n)^(3/2) / m² ≈ constant. The data "
     "tests this indirectly. Direct confirmation requires independent "
     "measurement of at least one of {m, a_s, n}."),
]

# ============================================================
# PART 5: COSMOLOGY BRIDGE
# ============================================================
if univ_verdict in ("CONFIRMED", "MARGINAL"):
    kz_interp = (
        "A primordial freeze-out scale at BEC condensation epoch "
        "can seed a universal coherence scale. Our empirical inversion is "
        "consistent with a universal primordial condensate state. Whether "
        "Kibble-Zurek freeze-out specifically produces g† = 1.2e-10 m/s² "
        "depends on the condensation epoch redshift z_c and cooling rate, "
        "which are free parameters of the model."
    )
    lambda_conn = (
        "The empirical acceleration scale g† is numerically "
        "consistent with c*sqrt(Lambda/3) where Lambda is the cosmological "
        "constant. This motivates the hypothesis that g† was locked in at "
        "the de Sitter-to-FRW transition. This is a motivated hypothesis, "
        "not a derivation."
    )
else:
    kz_interp = "Universality not confirmed — KZ interpretation not supported by data."
    lambda_conn = "Universality not confirmed — Lambda connection not testable."

# ============================================================
# SAVE OUTPUTS
# ============================================================
print("\n" + "=" * 72)
print("  SAVING OUTPUTS")
print("=" * 72)

# 1. Summary JSON
summary = {
    "n_galaxies_total": n_total,
    "n_galaxies_valid": n_valid,
    "n_too_few_points": n_too_few,
    "n_neg_gDM_rejected": n_all_neg_gdm,
    "neg_gDM_fraction_median": float(np.median(total_neg_gdm_frac)),
    "c_s2_median_m2s2": float(C_median),
    "c_s2_median_log": float(np.log10(C_median)),
    "c_s2_scatter_dex": float(cs2_scatter_all),
    "test_A_mass_trend": {
        "spearman_rho": float(rho_mass),
        "spearman_p": float(p_mass),
        "slope": float(slope_mass),
        "slope_se": float(se_slope),
        "n_galaxies": int(len(x_A)),
    },
    "test_B_environment": {
        "n_field": int(len(cs2_field)),
        "n_dense": int(len(cs2_dense)),
        "median_field_log": float(med_field),
        "median_dense_log": float(med_dense),
        "delta_dex": float(delta_env),
        "mwu_p": float(mwu_p),
    },
    "test_C_survey": survey_medians,
    "test_D_confound": {
        "partial_spearman_rho": float(partial_rho),
        "partial_spearman_p": float(partial_p),
    },
    "boson_mass_eV": float(m_eV),
    "boson_mass_kg": float(m_kg),
    "boson_mass_CI95": [float(ci_lo), float(ci_hi)],
    "g_int_Jm3": float(g_int),
    "lambda_coupling": float(lambda_coupling),
    "lyman_alpha_compatible": bool(lya_compatible),
    "lyman_alpha_note": "bound relaxed by self-interaction (GPP with g_int > 0)",
    "universality_verdict": univ_verdict,
    "caveats": caveats,
    "cosmology_bridge": {
        "kz_interpretation": kz_interp,
        "lambda_connection": lambda_conn,
    }
}

json_path = os.path.join(OUTPUT_DIR, 'gpp_corrected_summary.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"    Saved: {json_path}")

# 2. Per-galaxy CSV
csv_path = os.path.join(OUTPUT_DIR, 'galaxy_gpp_results.csv')
df_results.to_csv(csv_path, index=False)
print(f"    Saved: {csv_path}")

# ============================================================
# PART 6: PUBLICATION FIGURES
# ============================================================
print("\n" + "=" * 72)
print("  GENERATING PUBLICATION FIGURES")
print("=" * 72)

BG_COLOR = '#0d1117'
GRID_ALPHA = 0.3
TEXT_COLOR = '#c9d1d9'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'axes.edgecolor': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.alpha': GRID_ALPHA,
    'grid.color': '#30363d',
    'font.size': 11,
})

# --- Plot A: c_s²(r) for 12 representative galaxies ---
print("    Plot A: Sound speed profiles...")

# Select 12 galaxies spanning mass range
gal_list = [(g['galaxy'], g['logMh'], g['n_valid']) for g in galaxy_results if g['galaxy'] in all_profiles]
gal_list.sort(key=lambda x: x[1])

# Pick 4 from each mass tercile
n_per = 4
tercile_size = len(gal_list) // 3
selected = []
for t in range(3):
    start = t * tercile_size
    end = (t + 1) * tercile_size if t < 2 else len(gal_list)
    pool = gal_list[start:end]
    # Prefer galaxies with more valid points
    pool.sort(key=lambda x: x[2], reverse=True)
    selected.extend(pool[:n_per])

fig, ax = plt.subplots(figsize=(10, 7))
cmap = cm.viridis
mass_vals = [s[1] for s in selected]
norm_m = Normalize(vmin=min(mass_vals) - 0.5, vmax=max(mass_vals) + 0.5)

for gname, logmh, _ in selected:
    prof = all_profiles[gname]
    color = cmap(norm_m(logmh))
    ax.plot(prof['r_kpc'], prof['c_s2'], 'o-', color=color, markersize=3,
            alpha=0.8, linewidth=1.2, label=f"{gname} ({logmh:.1f})")

ax.set_yscale('log')
ax.set_xlabel('R [kpc]')
ax.set_ylabel('c$_s^2$(r)  [m²/s²]')
ax.set_title('Effective Sound Speed Profile c$_s^2$(r)\nSpherical approximation — disk galaxies',
             fontsize=13)
ax.axhline(C_median, color='#ff6b6b', ls='--', alpha=0.7, label=f'Median c$_s^2$ = {C_median:.2e}')
ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.3,
          facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
ax.grid(True)
sm = cm.ScalarMappable(cmap=cmap, norm=norm_m)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('log M$_h$ [M$_\\odot$]', color=TEXT_COLOR)
cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)

plt.tight_layout()
path_A = os.path.join(OUTPUT_DIR, 'plot_A_cs2_profiles.png')
plt.savefig(path_A, dpi=200, facecolor=BG_COLOR)
plt.close()
print(f"      Saved: {path_A}")

# --- Plot B: c_s² vs logMh (universality) ---
print("    Plot B: Universality test...")

fig, ax = plt.subplots(figsize=(10, 7))

field_df = df_test[df_test['env_dense'] == 'field']
dense_df = df_test[df_test['env_dense'] == 'dense']

ax.scatter(field_df['logMh'], field_df['log_c_s2_median'], c='#2ea043', s=25,
           alpha=0.6, label=f'Field (N={len(field_df)})', zorder=3)
ax.scatter(dense_df['logMh'], dense_df['log_c_s2_median'], c='#d29922', s=25,
           alpha=0.6, label=f'Dense (N={len(dense_df)})', zorder=3)

# Fit line
xfit = np.linspace(df_test['logMh'].min(), df_test['logMh'].max(), 100)
yfit = np.polyval(coeffs, xfit)
ax.plot(xfit, yfit, '--', color='#ff6b6b', linewidth=1.5,
        label=f'Slope = {slope_mass:.3f} ± {se_slope:.3f}')

# Median horizontal
ax.axhline(np.median(y_A), color='#58a6ff', ls=':', alpha=0.8,
           label=f'Median log(c$_s^2$) = {np.median(y_A):.2f}')

ax.set_xlabel('log M$_h$ [M$_\\odot$]')
ax.set_ylabel('log c$_s^2$  [m²/s²]')
ax.set_title(f'Sound Speed Universality Test — Verdict: {univ_verdict}\n'
             f'Scatter = {cs2_scatter_all:.3f} dex, ρ = {rho_mass:.3f} (p = {p_mass:.2e})',
             fontsize=13)
ax.legend(loc='upper left', framealpha=0.3, facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
ax.grid(True)

plt.tight_layout()
path_B = os.path.join(OUTPUT_DIR, 'plot_B_universality.png')
plt.savefig(path_B, dpi=200, facecolor=BG_COLOR)
plt.close()
print(f"      Saved: {path_B}")

# --- Plot C: Histogram of log(c_s²) ---
print("    Plot C: Distribution histogram...")

fig, ax = plt.subplots(figsize=(10, 7))

all_log_cs2 = df_results['log_c_s2_median'].dropna().values
hist_n, hist_bins, hist_patches = ax.hist(all_log_cs2, bins=30, color='#58a6ff',
                                           alpha=0.7, edgecolor='#0d1117', linewidth=0.5)

# Gaussian fit
mu_fit = np.mean(all_log_cs2)
sigma_fit = np.std(all_log_cs2)
x_gauss = np.linspace(all_log_cs2.min() - 0.5, all_log_cs2.max() + 0.5, 200)
y_gauss = norm.pdf(x_gauss, mu_fit, sigma_fit) * len(all_log_cs2) * (hist_bins[1] - hist_bins[0])
ax.plot(x_gauss, y_gauss, '--', color='#ff6b6b', linewidth=2,
        label=f'Gaussian: μ={mu_fit:.2f}, σ={sigma_fit:.2f} dex')

ax.axvline(mu_fit, color='#ff6b6b', ls=':', alpha=0.8)
ax.set_xlabel('log c$_s^2$  [m²/s²]')
ax.set_ylabel('Count')
ax.set_title(f'Distribution of Galaxy-Median Sound Speed\n'
             f'N = {len(all_log_cs2)}, μ = {mu_fit:.2f}, σ = {sigma_fit:.2f} dex',
             fontsize=13)
ax.legend(framealpha=0.3, facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
ax.grid(True)

plt.tight_layout()
path_C = os.path.join(OUTPUT_DIR, 'plot_C_cs2_distribution.png')
plt.savefig(path_C, dpi=200, facecolor=BG_COLOR)
plt.close()
print(f"      Saved: {path_C}")

# --- Plot D: L_rho vs xi_eff (TF validity) ---
print("    Plot D: TF validity diagnostic...")

fig, ax = plt.subplots(figsize=(10, 7))

x_plot = df_results['xi_eff_median_kpc'].values
y_plot = df_results['L_rho_median_kpc'].values
valid_plot = (x_plot > 0) & (y_plot > 0) & np.isfinite(x_plot) & np.isfinite(y_plot)

colors_env = np.array(['#2ea043' if e == 'field' else '#d29922'
                        for e in df_results['env_dense'].values])

ax.scatter(x_plot[valid_plot], y_plot[valid_plot], c=colors_env[valid_plot],
           s=30, alpha=0.6, zorder=3)

# Diagonal: L_rho = xi_eff
diag_range = np.logspace(np.log10(max(0.01, np.nanmin(x_plot[valid_plot]))),
                          np.log10(np.nanmax(x_plot[valid_plot])), 100)
ax.plot(diag_range, diag_range, '--', color='#ff6b6b', linewidth=1.5,
        label='L$_\\rho$ = ξ$_{eff}$ (TF boundary)')
ax.fill_between(diag_range, 0, diag_range, color='#ff6b6b', alpha=0.1)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('ξ$_{eff}$ median [kpc]')
ax.set_ylabel('L$_\\rho$ median [kpc]')
ax.set_title('Thomas-Fermi Validity Diagnostic\n'
             'Points above line: TF likely valid | Below: TF may be violated',
             fontsize=13)

# Legend patches
from matplotlib.lines import Line2D
leg_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ea043', markersize=8, label='Field'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d29922', markersize=8, label='Dense'),
    Line2D([0], [0], color='#ff6b6b', ls='--', linewidth=1.5, label='L$_\\rho$ = ξ$_{eff}$'),
]
ax.legend(handles=leg_elements, loc='upper left', framealpha=0.3,
          facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
ax.grid(True)

# Count above/below
n_above = np.sum(y_plot[valid_plot] > x_plot[valid_plot])
n_below = np.sum(y_plot[valid_plot] <= x_plot[valid_plot])
ax.text(0.98, 0.05, f'Above (TF valid): {n_above}\nBelow (TF uncertain): {n_below}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor=BG_COLOR, edgecolor=TEXT_COLOR, alpha=0.8))

plt.tight_layout()
path_D = os.path.join(OUTPUT_DIR, 'plot_D_tf_validity.png')
plt.savefig(path_D, dpi=200, facecolor=BG_COLOR)
plt.close()
print(f"      Saved: {path_D}")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 72)
print("  FINAL REPORT")
print("=" * 72)
print(f"    Galaxies analyzed: {n_valid}")
print(f"    Universality verdict: {univ_verdict}")
print(f"    Median c_s² = {C_median:.4e} m²/s²  (log = {np.log10(C_median):.3f})")
print(f"    Scatter = {cs2_scatter_all:.3f} dex")
print(f"    Mass slope = {slope_mass:.4f} ± {se_slope:.4f}")
print(f"    Env split Δ = {delta_env:.3f} dex (p = {mwu_p:.4e})")
print(f"    Boson mass = {m_eV:.4e} eV/c²  [{ci_lo:.4e}, {ci_hi:.4e}]")
print(f"    Lyman-α: {'compatible' if lya_compatible else 'below bound'} (relaxed by SI)")
print(f"\n    All outputs in: {OUTPUT_DIR}")
print("=" * 72)

# ============================================================
# PART 7: QUALITY-CUT SUB-ANALYSES
# ============================================================
print("\n" + "=" * 72)
print("  PART 7: Quality-Cut Sub-Analyses")
print("=" * 72)

# Helper for sub-analysis
def run_subanalysis(label, df_sub, profiles_sub):
    """Run universality tests on a subset. Returns dict of results."""
    if len(df_sub) < 10:
        print(f"    {label}: only {len(df_sub)} galaxies — skipping")
        return None

    lcs2 = df_sub['log_c_s2_median'].dropna().values
    lmh = df_sub['logMh'].values
    mask = np.isfinite(lcs2) & np.isfinite(lmh) & (lmh > 0)
    lcs2_m = lcs2[mask]
    lmh_m = lmh[mask]

    if len(lcs2_m) < 10:
        print(f"    {label}: only {len(lcs2_m)} with valid logMh — skipping")
        return None

    scatter = 1.4826 * np.median(np.abs(lcs2_m - np.median(lcs2_m)))
    rho_s, p_s = spearmanr(lmh_m, lcs2_m)
    coeffs_s = np.polyfit(lmh_m, lcs2_m, 1)
    slope_s = coeffs_s[0]
    y_pred_s = np.polyval(coeffs_s, lmh_m)
    res_s = lcs2_m - y_pred_s
    se_s = np.sqrt(np.sum(res_s**2) / (len(lmh_m) - 2) / np.sum((lmh_m - lmh_m.mean())**2))

    # Env split
    f_mask = df_sub['env_dense'] == 'field'
    d_mask = df_sub['env_dense'] == 'dense'
    cs2_f = df_sub.loc[f_mask, 'log_c_s2_median'].dropna().values
    cs2_d = df_sub.loc[d_mask, 'log_c_s2_median'].dropna().values
    if len(cs2_f) >= 3 and len(cs2_d) >= 3:
        _, env_p = mannwhitneyu(cs2_f, cs2_d, alternative='two-sided')
        env_delta = np.median(cs2_f) - np.median(cs2_d)
    else:
        env_p = np.nan
        env_delta = np.nan

    # Verdict
    if abs(slope_s) < 0.05 and scatter < 0.25:
        v = "CONFIRMED"
    elif abs(slope_s) < 0.10 and scatter < 0.40:
        v = "MARGINAL"
    else:
        v = "DISFAVORED"

    # Boson mass from this subset's profiles
    sub_cs2_pts = []
    sub_xi_pts = []
    for gname in df_sub['galaxy'].values:
        if gname in profiles_sub:
            sub_cs2_pts.extend(profiles_sub[gname]['c_s2'])
            sub_xi_pts.extend([x * KPC_M for x in profiles_sub[gname]['xi_eff_kpc']])
    sub_cs2_pts = np.array(sub_cs2_pts)
    sub_xi_pts = np.array(sub_xi_pts)
    vm = (sub_cs2_pts > 0) & np.isfinite(sub_cs2_pts) & (sub_xi_pts > 0) & np.isfinite(sub_xi_pts)
    if vm.sum() > 10:
        C_sub = np.median(sub_cs2_pts[vm])
        xi_sub = np.median(sub_xi_pts[vm])
        m_sub = hbar / (np.sqrt(2.0 * C_sub) * xi_sub) / KG_EV
    else:
        C_sub, xi_sub, m_sub = np.nan, np.nan, np.nan

    print(f"    {label}: N={len(lcs2_m)}, scatter={scatter:.3f} dex, "
          f"slope={slope_s:.4f}±{se_s:.4f}, ρ={rho_s:.3f} (p={p_s:.3e}), "
          f"env Δ={env_delta:.3f} dex (p={env_p:.3e}), "
          f"m_boson={m_sub:.3e} eV, verdict={v}")

    return {
        'label': label,
        'n_galaxies': int(len(lcs2_m)),
        'scatter_dex': float(scatter),
        'median_log_cs2': float(np.median(lcs2_m)),
        'slope': float(slope_s),
        'slope_se': float(se_s),
        'spearman_rho': float(rho_s),
        'spearman_p': float(p_s),
        'env_delta_dex': float(env_delta) if np.isfinite(env_delta) else None,
        'env_mwu_p': float(env_p) if np.isfinite(env_p) else None,
        'boson_mass_eV': float(m_sub) if np.isfinite(m_sub) else None,
        'verdict': v,
    }

sub_results = {}

# --- 7a: SPARC only ---
df_sparc = df_results[df_results['source'] == 'SPARC'].copy()
r = run_subanalysis("SPARC-only", df_sparc, all_profiles)
if r: sub_results['sparc_only'] = r

# --- 7b: SPARC + deBlok (high-quality decomposition) ---
df_hiq = df_results[df_results['source'].isin(['SPARC', 'deBlok2002'])].copy()
r = run_subanalysis("SPARC+deBlok (hi-qual)", df_hiq, all_profiles)
if r: sub_results['sparc_deblok'] = r

# --- 7c: TF-valid only (L_rho > xi_eff) ---
df_tf = df_results[df_results['L_rho_median_kpc'] > df_results['xi_eff_median_kpc']].copy()
r = run_subanalysis("TF-valid only", df_tf, all_profiles)
if r: sub_results['tf_valid'] = r

# --- 7d: SPARC + TF-valid ---
df_sparc_tf = df_sparc[df_sparc['L_rho_median_kpc'] > df_sparc['xi_eff_median_kpc']].copy()
r = run_subanalysis("SPARC + TF-valid", df_sparc_tf, all_profiles)
if r: sub_results['sparc_tf_valid'] = r

# --- 7e: ≥10 valid points ---
df_10pt = df_results[df_results['n_valid'] >= 10].copy()
r = run_subanalysis("≥10 valid pts", df_10pt, all_profiles)
if r: sub_results['min10pts'] = r

# --- 7f: SPARC + ≥10 pts ---
df_sparc_10 = df_sparc[df_sparc['n_valid'] >= 10].copy()
r = run_subanalysis("SPARC + ≥10 pts", df_sparc_10, all_profiles)
if r: sub_results['sparc_min10pts'] = r

# --- 7g: Per-source breakdown ---
print("\n    Per-source breakdown:")
for src in df_results['source'].unique():
    df_src = df_results[df_results['source'] == src].copy()
    if len(df_src) >= 5:
        r = run_subanalysis(f"  Source: {src}", df_src, all_profiles)
        if r: sub_results[f'source_{src}'] = r

# Save sub-analysis results
sub_json_path = os.path.join(OUTPUT_DIR, 'gpp_subanalysis_results.json')
with open(sub_json_path, 'w') as f:
    json.dump(sub_results, f, indent=2)
print(f"\n    Saved: {sub_json_path}")

# --- Plot E: Comparison across subsamples ---
print("\n    Plot E: Subsample comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left panel: scatter by subsample
labels_plot = []
scatters_plot = []
slopes_plot = []
slopes_se_plot = []
colors_plot = []

key_order = ['sparc_only', 'sparc_deblok', 'tf_valid', 'sparc_tf_valid',
             'min10pts', 'sparc_min10pts']
nice_names = {
    'sparc_only': 'SPARC only',
    'sparc_deblok': 'SPARC+deBlok',
    'tf_valid': 'TF-valid all',
    'sparc_tf_valid': 'SPARC+TF-valid',
    'min10pts': '≥10 pts all',
    'sparc_min10pts': 'SPARC ≥10 pts',
}

for k in key_order:
    if k in sub_results:
        sr = sub_results[k]
        labels_plot.append(f"{nice_names[k]}\n(N={sr['n_galaxies']})")
        scatters_plot.append(sr['scatter_dex'])
        slopes_plot.append(sr['slope'])
        slopes_se_plot.append(sr['slope_se'])
        colors_plot.append('#58a6ff' if 'sparc' in k else '#2ea043')

# Add full sample
labels_plot.insert(0, f'All surveys\n(N={len(x_A)})')
scatters_plot.insert(0, cs2_scatter_all)
slopes_plot.insert(0, slope_mass)
slopes_se_plot.insert(0, se_slope)
colors_plot.insert(0, '#ff6b6b')

y_pos = np.arange(len(labels_plot))

ax1 = axes[0]
ax1.barh(y_pos, scatters_plot, color=colors_plot, alpha=0.7, edgecolor=BG_COLOR)
ax1.axvline(0.25, color='#ff6b6b', ls='--', alpha=0.8, label='Threshold (0.25 dex)')
ax1.axvline(0.40, color='#d29922', ls=':', alpha=0.6, label='Marginal (0.40 dex)')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels_plot, fontsize=9)
ax1.set_xlabel('Scatter [dex]')
ax1.set_title('c$_s^2$ Scatter by Subsample', fontsize=13)
ax1.legend(fontsize=8, loc='lower right', framealpha=0.3,
           facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
ax1.grid(True, axis='x')
ax1.invert_yaxis()

# Right panel: slopes
ax2 = axes[1]
ax2.errorbar(slopes_plot, y_pos, xerr=slopes_se_plot, fmt='o',
             color='#58a6ff', markersize=8, capsize=4, ecolor='#58a6ff',
             elinewidth=1.5, markeredgecolor=BG_COLOR)
ax2.axvline(0, color='#2ea043', ls='-', alpha=0.8, label='Zero slope (universal)')
ax2.axvspan(-0.05, 0.05, color='#2ea043', alpha=0.15, label='±0.05 threshold')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels_plot, fontsize=9)
ax2.set_xlabel('Slope (dlog c$_s^2$ / dlog M$_h$)')
ax2.set_title('Mass Dependence by Subsample', fontsize=13)
ax2.legend(fontsize=8, loc='lower right', framealpha=0.3,
           facecolor=BG_COLOR, edgecolor=TEXT_COLOR)
ax2.grid(True, axis='x')
ax2.invert_yaxis()

plt.tight_layout()
path_E = os.path.join(OUTPUT_DIR, 'plot_E_subsample_comparison.png')
plt.savefig(path_E, dpi=200, facecolor=BG_COLOR)
plt.close()
print(f"      Saved: {path_E}")

# Update summary JSON with sub-analysis
summary['subanalysis'] = sub_results
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"    Updated: {json_path}")

print("\n" + "=" * 72)
print("  DONE — All analyses complete")
print("=" * 72)
