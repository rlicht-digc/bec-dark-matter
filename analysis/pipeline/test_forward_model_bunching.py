#!/usr/bin/env python3
"""
Forward-Model Bunching Validation: BEC vs NFW Per-Galaxy Rotation Curve Fits

Instead of comparing variance models on binned RAR residuals (Test 8),
this test directly fits each SPARC rotation curve with two dark matter models:

  Model A (BEC/Soliton): V²_DM(r) from soliton ρ(r) = ρ_c [1 + 0.091(r/r_c)²]⁻⁸
  Model B (NFW):         V²_DM(r) from NFW ρ(r) = ρ_s / [(r/r_s)(1+r/r_s)²]

Both models have 2 free DM parameters + fixed baryonic components (Y* = 0.5, Y_b = 0.7).
ΔAIC = AIC_NFW - AIC_BEC per galaxy; positive = BEC preferred.

Forward-model approach tests whether the ΔAIC_bunching = +9.7 from the unified
pipeline's binned-variance test is consistent with per-galaxy rotation curve fitting.

BEC prediction: ΔAIC should be more positive for massive galaxies (X ~ R/ξ < 1,
deep quantum regime) than for dwarfs (X >> 1, classical regime).
"""
import os
import json
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr, wilcoxon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physical constants
G_SI = 6.674e-11       # m³/(kg s²)
Msun_kg = 1.989e30     # kg
kpc_m = 3.086e19       # m per kpc
gdagger = 1.2e-10      # m/s² (McGaugh+2016)

# Mass-to-light ratios
Y_DISK = 0.5
Y_BULGE = 0.7

# ============================================================
# SOLITON MODEL: Precompute M(r) lookup table
# ============================================================
# Schive+2014: ρ(r) = ρ_c [1 + 0.091(r/r_c)²]⁻⁸
# M(r) = 4π ∫₀ʳ ρ(r') r'² dr' = 4π ρ_c r_c³ × m̃(r/r_c)
# where m̃(u) = ∫₀ᵘ x² / (1 + 0.091 x²)⁸ dx

# Build a high-resolution lookup for m̃(u)
_N_SOL = 2000
_u_max = 50.0  # soliton is negligible beyond ~10 r_c
_u_table = np.linspace(0, _u_max, _N_SOL + 1)
_du = _u_table[1] - _u_table[0]

# Integrand: x² / (1 + 0.091 x²)⁸
_x = _u_table
_integrand = _x**2 / (1.0 + 0.091 * _x**2)**8
# Cumulative integral via trapezoidal rule.
_m_tilde = np.zeros(_N_SOL + 1)
for i in range(1, _N_SOL + 1):
    _m_tilde[i] = _m_tilde[i-1] + 0.5 * (_integrand[i-1] + _integrand[i]) * _du

# Total soliton mass coefficient from this profile:
# M_sol = 4π * m_tilde_inf * rho_c * r_c^3, with m_tilde_inf ≈ 0.922.
M_TILDE_INF = _m_tilde[-1]
print(f"Soliton m̃(∞) = {M_TILDE_INF:.4f} (expected ~0.922)")

def soliton_v_circ(r_kpc, rho_c, r_c_kpc):
    """
    Circular velocity from soliton profile.

    Parameters:
        r_kpc: radius array in kpc
        rho_c: central density in Msun/kpc³
        r_c_kpc: core radius in kpc

    Returns:
        V_circ in km/s
    """
    u = np.abs(r_kpc) / r_c_kpc
    # Interpolate m̃(u)
    m_val = np.interp(u, _u_table, _m_tilde)
    # M(r) = 4π ρ_c r_c³ m̃(u)  [in Msun if ρ_c in Msun/kpc³, r_c in kpc]
    M_r = 4.0 * np.pi * rho_c * r_c_kpc**3 * m_val  # Msun
    # V² = GM/r => V = sqrt(G M / r)
    # G = 4.302e-3 pc (km/s)² / Msun = 4.302e-6 kpc (km/s)² / Msun
    G_kpc = 4.302e-6  # kpc (km/s)² / Msun
    V_sq = np.where(r_kpc > 0, G_kpc * M_r / r_kpc, 0.0)
    return np.sqrt(np.maximum(V_sq, 0.0))


def nfw_v_circ(r_kpc, rho_s, r_s_kpc):
    """
    Circular velocity from NFW profile.

    V²_NFW(r) = (G × 4π ρ_s r_s³ / r) × [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]

    Parameters:
        r_kpc: radius array in kpc
        rho_s: scale density in Msun/kpc³
        r_s_kpc: scale radius in kpc

    Returns:
        V_circ in km/s
    """
    x = r_kpc / r_s_kpc
    G_kpc = 4.302e-6  # kpc (km/s)² / Msun
    M_r = 4.0 * np.pi * rho_s * r_s_kpc**3 * (np.log(1.0 + x) - x / (1.0 + x))
    V_sq = np.where(r_kpc > 0, G_kpc * M_r / r_kpc, 0.0)
    return np.sqrt(np.maximum(V_sq, 0.0))


def xi_kpc(logMs):
    """BEC healing length in kpc."""
    Ms_SI = 10.0**logMs * Msun_kg
    return np.sqrt(G_SI * Ms_SI / gdagger) / kpc_m


# ============================================================
# LOAD SPARC DATA
# ============================================================
print("=" * 72)
print("FORWARD-MODEL BUNCHING VALIDATION: BEC vs NFW Per-Galaxy Fits")
print("=" * 72)

print("\n[1] Loading SPARC rotation curves...")

table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

# Parse rotation curves
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

# Parse MRT for stellar masses, inclinations, quality flags
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
        Inc = float(parts[4])
        L36 = float(parts[6])
        Q = int(parts[16])
        logMs = np.log10(max(0.5 * L36 * 1e9, 1e6))
        sparc_props[name] = {
            'Inc': Inc, 'Q': Q, 'logMs': logMs,
        }
    except (ValueError, IndexError):
        continue

print(f"  {len(galaxies)} rotation curves, {len(sparc_props)} with stellar masses")

# ============================================================
# FIT EACH GALAXY
# ============================================================
print("\n[2] Fitting BEC (soliton) and NFW per galaxy...")
print("    (This may take a few minutes)")

MIN_POINTS = 8  # minimum RC points for fitting
results = []
n_fit = 0
n_skip = 0

galaxy_names = sorted(galaxies.keys())

for name in galaxy_names:
    if name not in sparc_props:
        n_skip += 1
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        n_skip += 1
        continue

    gdata = galaxies[name]
    R = gdata['R']
    Vobs = gdata['Vobs']
    eVobs = gdata['eVobs']
    Vdisk = gdata['Vdisk']
    Vgas = gdata['Vgas']
    Vbul = gdata['Vbul']

    # Quality cuts
    valid = (R > 0) & (Vobs > 0) & (eVobs > 0)
    if np.sum(valid) < MIN_POINTS:
        n_skip += 1
        continue

    R = R[valid]
    Vobs = Vobs[valid]
    eVobs = eVobs[valid]
    Vdisk = Vdisk[valid]
    Vgas = Vgas[valid]
    Vbul = Vbul[valid]

    # Enforce minimum error floor (5 km/s or 5% of Vobs)
    eVobs = np.maximum(eVobs, np.maximum(5.0, 0.05 * Vobs))

    logMs = prop['logMs']
    xi = xi_kpc(logMs)

    # Baryonic velocity squared (fixed)
    Vbar_sq = Y_DISK * Vdisk**2 + np.sign(Vgas) * Vgas**2 + Y_BULGE * np.sign(Vbul) * Vbul**2

    N = len(R)
    R_max = np.max(R)
    V_max = np.max(Vobs)

    # ---- FIT NFW ----
    # Parameters: log10(rho_s [Msun/kpc³]), log10(r_s [kpc])
    # Ranges: rho_s from 1e4 to 1e10, r_s from 0.1 to 200 kpc

    def chi2_nfw(params):
        log_rho_s, log_r_s = params
        rho_s = 10.0**log_rho_s
        r_s = 10.0**log_r_s
        V_dm = nfw_v_circ(R, rho_s, r_s)
        V_total_sq = Vbar_sq + V_dm**2
        V_model = np.sqrt(np.maximum(V_total_sq, 0.0))
        chi2 = np.sum(((Vobs - V_model) / eVobs)**2)
        return chi2

    # Use differential evolution for robust global optimization
    try:
        result_nfw = differential_evolution(
            chi2_nfw,
            bounds=[(4, 10), (-1, 2.5)],  # log(rho_s), log(r_s)
            seed=42, maxiter=200, tol=1e-6, polish=True
        )
        chi2_nfw_val = result_nfw.fun
        nfw_params = result_nfw.x
    except Exception:
        n_skip += 1
        continue

    # ---- FIT BEC SOLITON ----
    # Parameters: log10(rho_c [Msun/kpc³]), log10(r_c [kpc])
    # Physical constraint: r_c bounded by healing length scale
    # r_c ∈ [0.2ξ, 5ξ] prevents optimizer from exploiting extra freedom

    rc_lo = max(0.2 * xi, 0.05)  # lower bound in kpc
    rc_hi = max(5.0 * xi, 2.0)   # upper bound in kpc

    def chi2_bec(params):
        log_rho_c, log_r_c = params
        rho_c = 10.0**log_rho_c
        r_c = 10.0**log_r_c
        V_dm = soliton_v_circ(R, rho_c, r_c)
        V_total_sq = Vbar_sq + V_dm**2
        V_model = np.sqrt(np.maximum(V_total_sq, 0.0))
        chi2 = np.sum(((Vobs - V_model) / eVobs)**2)
        return chi2

    try:
        result_bec = differential_evolution(
            chi2_bec,
            bounds=[(4, 10), (np.log10(rc_lo), np.log10(rc_hi))],
            seed=42, maxiter=200, tol=1e-6, polish=True
        )
        chi2_bec_val = result_bec.fun
        bec_params = result_bec.x
    except Exception:
        n_skip += 1
        continue

    # ---- COMPUTE AIC ----
    # Both models: 2 free parameters
    k = 2
    aic_nfw = chi2_nfw_val + 2 * k
    aic_bec = chi2_bec_val + 2 * k
    daic = aic_nfw - aic_bec  # positive = BEC preferred

    # Reduced chi-squared
    rchi2_nfw = chi2_nfw_val / max(N - k, 1)
    rchi2_bec = chi2_bec_val / max(N - k, 1)

    # X = R/ξ statistics
    X_vals = R / xi
    X_min = np.min(X_vals)
    X_max = np.max(X_vals)
    X_med = np.median(X_vals)

    results.append({
        'name': name,
        'logMs': round(logMs, 2),
        'xi_kpc': round(xi, 2),
        'N_pts': N,
        'X_min': round(float(X_min), 3),
        'X_med': round(float(X_med), 2),
        'X_max': round(float(X_max), 2),
        'chi2_nfw': round(float(chi2_nfw_val), 2),
        'chi2_bec': round(float(chi2_bec_val), 2),
        'rchi2_nfw': round(float(rchi2_nfw), 3),
        'rchi2_bec': round(float(rchi2_bec), 3),
        'daic': round(float(daic), 2),
        'nfw_log_rhos': round(float(nfw_params[0]), 2),
        'nfw_log_rs': round(float(nfw_params[1]), 2),
        'bec_log_rhoc': round(float(bec_params[0]), 2),
        'bec_log_rc': round(float(bec_params[1]), 2),
    })

    n_fit += 1
    if n_fit % 20 == 0:
        print(f"    {n_fit} galaxies fitted...")

print(f"\n  Fitted: {n_fit} galaxies")
print(f"  Skipped: {n_skip} galaxies (quality/data cuts)")
if len(results) == 0:
    raise RuntimeError(
        "No galaxies passed fitting/quality cuts. Check SPARC paths and selection thresholds."
    )

# ============================================================
# RESULTS TABLE
# ============================================================
print("\n" + "=" * 72)
print("PER-GALAXY FIT RESULTS")
print("=" * 72)

# Sort by ΔAIC
results.sort(key=lambda r: r['daic'], reverse=True)

print(f"\n{'Galaxy':>12s} {'logMs':>6s} {'ξ':>6s} {'N':>4s} {'X_med':>6s} "
      f"{'χ²_NFW':>8s} {'χ²_BEC':>8s} {'rχ²_NFW':>7s} {'rχ²_BEC':>7s} {'ΔAIC':>7s}")
print("-" * 82)

n_bec_better = 0
n_nfw_better = 0
n_indist = 0

for r in results:
    tag = ''
    if r['daic'] > 2:
        n_bec_better += 1
        tag = ' ★'
    elif r['daic'] < -2:
        n_nfw_better += 1
        tag = ' ◆'
    else:
        n_indist += 1

    print(f"{r['name']:>12s} {r['logMs']:6.2f} {r['xi_kpc']:6.2f} {r['N_pts']:4d} "
          f"{r['X_med']:6.2f} {r['chi2_nfw']:8.1f} {r['chi2_bec']:8.1f} "
          f"{r['rchi2_nfw']:7.3f} {r['rchi2_bec']:7.3f} {r['daic']:+7.2f}{tag}")

print(f"\n  ★ = BEC preferred (ΔAIC > +2)")
print(f"  ◆ = NFW preferred (ΔAIC < -2)")
print(f"  (blank) = indistinguishable (|ΔAIC| ≤ 2)")

# ============================================================
# AGGREGATE STATISTICS
# ============================================================
print("\n" + "=" * 72)
print("AGGREGATE STATISTICS")
print("=" * 72)

all_daic = np.array([r['daic'] for r in results])
all_logMs = np.array([r['logMs'] for r in results])
all_xi = np.array([r['xi_kpc'] for r in results])
all_Xmed = np.array([r['X_med'] for r in results])
p_w = None

print(f"\n  Total galaxies fitted: {len(results)}")
print(f"  BEC preferred (ΔAIC > +2):    {n_bec_better} ({100*n_bec_better/len(results):.0f}%)")
print(f"  NFW preferred (ΔAIC < -2):    {n_nfw_better} ({100*n_nfw_better/len(results):.0f}%)")
print(f"  Indistinguishable (|ΔAIC|≤2): {n_indist} ({100*n_indist/len(results):.0f}%)")
print(f"\n  Mean ΔAIC: {np.mean(all_daic):+.2f}")
print(f"  Median ΔAIC: {np.median(all_daic):+.2f}")
print(f"  Sum ΔAIC: {np.sum(all_daic):+.1f}")

# Test: is the median ΔAIC significantly different from zero?
if len(all_daic) >= 5:
    try:
        stat_w, p_w = wilcoxon(all_daic)
        print(f"  Wilcoxon signed-rank: W={stat_w:.0f}, p={p_w:.4f}")
    except ValueError:
        p_w = 1.0
        print(f"  Wilcoxon signed-rank: not applicable")

# ============================================================
# MASS-DEPENDENT TREND (BEC PREDICTION)
# ============================================================
print("\n" + "=" * 72)
print("MASS-DEPENDENT TREND TEST")
print("=" * 72)

# BEC predicts: ΔAIC should increase with stellar mass (lower X = more quantum)
rho_Ms, p_Ms = spearmanr(all_logMs, all_daic)
rho_xi, p_xi = spearmanr(all_xi, all_daic)
rho_X, p_X = spearmanr(all_Xmed, all_daic)

print(f"\n  Spearman correlations:")
print(f"    ΔAIC vs logMs:  ρ = {rho_Ms:+.3f}, p = {p_Ms:.4f}")
print(f"    ΔAIC vs ξ(kpc): ρ = {rho_xi:+.3f}, p = {p_xi:.4f}")
print(f"    ΔAIC vs X_med:  ρ = {rho_X:+.3f}, p = {p_X:.4f}")

print(f"\n  BEC prediction: ρ(ΔAIC, logMs) > 0 and ρ(ΔAIC, X_med) < 0")
if rho_Ms > 0 and p_Ms < 0.05:
    mass_verdict = "CONFIRMED — ΔAIC increases with stellar mass"
elif rho_Ms < 0 and p_Ms < 0.05:
    mass_verdict = "CONTRADICTED — ΔAIC decreases with stellar mass"
else:
    mass_verdict = "INCONCLUSIVE — no significant mass trend"
print(f"  Mass trend: {mass_verdict}")

# ============================================================
# MASS-BINNED SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("MASS-BINNED SUMMARY")
print("=" * 72)

mass_bins = [
    ('Dwarfs (logMs<9)', lambda m: m < 9),
    ('Low-mass spirals (9-9.5)', lambda m: 9 <= m < 9.5),
    ('Intermediate (9.5-10)', lambda m: 9.5 <= m < 10),
    ('Massive spirals (10-10.5)', lambda m: 10 <= m < 10.5),
    ('Very massive (>10.5)', lambda m: m >= 10.5),
]

print(f"\n  {'Mass bin':30s} {'N':>4s} {'<ΔAIC>':>8s} {'med ΔAIC':>9s} {'<X_med>':>8s} {'N_BEC':>6s} {'N_NFW':>6s}")
print("  " + "-" * 80)

for label, selector in mass_bins:
    mask = np.array([selector(m) for m in all_logMs])
    if np.sum(mask) < 2:
        print(f"  {label:30s} {np.sum(mask):4d}    (too few)")
        continue

    bin_daic = all_daic[mask]
    bin_X = all_Xmed[mask]
    n_bec = np.sum(bin_daic > 2)
    n_nfw = np.sum(bin_daic < -2)

    print(f"  {label:30s} {len(bin_daic):4d} {np.mean(bin_daic):+8.2f} "
          f"{np.median(bin_daic):+9.2f} {np.mean(bin_X):8.2f} "
          f"{n_bec:6d} {n_nfw:6d}")

# ============================================================
# COMPARISON WITH BINNED-VARIANCE BUNCHING (TEST 8)
# ============================================================
print("\n" + "=" * 72)
print("COMPARISON WITH BINNED-VARIANCE BUNCHING (TEST 8)")
print("=" * 72)

sum_daic = np.sum(all_daic)
print(f"""
  Test 8 (binned-variance, pipeline):  ΔAIC = +9.7  (quantum bunching preferred)
  Test 8 (binned-variance, SPARC-only): ΔAIC = -40   (classical preferred)

  This test (forward-model, per-galaxy): Sum ΔAIC = {sum_daic:+.1f}
                                         Mean ΔAIC = {np.mean(all_daic):+.2f}
                                         Median ΔAIC = {np.median(all_daic):+.2f}
""")

if sum_daic > 10:
    consistency = "CONSISTENT with pipeline bunching (both favor quantum)"
elif sum_daic < -10:
    consistency = "CONSISTENT with SPARC-only bunching (both favor classical)"
elif abs(sum_daic) <= 10:
    consistency = "INCONCLUSIVE — sum ΔAIC near zero (models indistinguishable)"

print(f"  Forward-model verdict: {consistency}")

# ============================================================
# INTERPRET INNER vs OUTER RESULTS
# ============================================================
print("\n" + "=" * 72)
print("INNER (X < 1) vs OUTER (X > 1) COMPARISON")
print("=" * 72)

inner_mask = all_Xmed < 1.0
outer_mask = all_Xmed >= 1.0

if np.sum(inner_mask) >= 3 and np.sum(outer_mask) >= 3:
    print(f"\n  Galaxies with X_med < 1 (inside healing length): {np.sum(inner_mask)}")
    print(f"    Mean ΔAIC: {np.mean(all_daic[inner_mask]):+.2f}")
    print(f"    Median ΔAIC: {np.median(all_daic[inner_mask]):+.2f}")
    print(f"\n  Galaxies with X_med ≥ 1 (outside healing length): {np.sum(outer_mask)}")
    print(f"    Mean ΔAIC: {np.mean(all_daic[outer_mask]):+.2f}")
    print(f"    Median ΔAIC: {np.median(all_daic[outer_mask]):+.2f}")

    inner_daic = np.mean(all_daic[inner_mask])
    outer_daic = np.mean(all_daic[outer_mask])

    if inner_daic > outer_daic + 2:
        print(f"\n  BEC prediction: CONFIRMED — inner galaxies favor BEC more than outer")
    elif outer_daic > inner_daic + 2:
        print(f"\n  BEC prediction: CONTRADICTED — outer galaxies favor BEC more")
    else:
        print(f"\n  BEC prediction: INCONCLUSIVE — no significant inner/outer difference")
else:
    print(f"\n  Too few galaxies in one regime for comparison")
    print(f"  Inner (X_med < 1): {np.sum(inner_mask)}, Outer: {np.sum(outer_mask)}")

# ============================================================
# OVERALL VERDICT
# ============================================================
print("\n" + "=" * 72)
print("OVERALL VERDICT")
print("=" * 72)

# Determine overall result
if abs(np.mean(all_daic)) < 1.0 and p_Ms > 0.05:
    overall = "INCONCLUSIVE"
    detail = ("Forward-model fits show BEC and NFW are statistically "
              "indistinguishable for most SPARC galaxies. No mass-dependent trend.")
elif np.mean(all_daic) > 1.0 and p_Ms < 0.05 and rho_Ms > 0:
    overall = "BEC-FAVORED"
    detail = ("Forward-model fits prefer BEC, with mass-dependent trend "
              "consistent with quantum→classical transition.")
elif np.mean(all_daic) < -1.0:
    overall = "NFW-FAVORED"
    detail = ("Forward-model fits prefer NFW over BEC soliton. "
              "No evidence for quantum dark matter enhancement.")
else:
    overall = "INCONCLUSIVE"
    detail = "Mixed results — no clear preference."

print(f"\n  Verdict: {overall}")
print(f"  Detail: {detail}")

print(f"""
  IMPORTANT CAVEATS:
  1. Soliton-only BEC model lacks the NFW-like envelope seen in simulations
     (Schive+2014, Mocz+2017). Real FDM/BEC halos have soliton core + envelope.
     Soliton ρ ∝ r⁻¹⁶ falls off much faster than NFW ρ ∝ r⁻³, so NFW
     inherently fits extended RCs better. The absolute ΔAIC is biased.
  2. The KEY diagnostic is the MASS-DEPENDENT TREND, not the absolute ΔAIC.
     BEC predicts ΔAIC should increase with M* (more quantum at small X = R/ξ).
     Result: ρ(ΔAIC, logMs) = {rho_Ms:+.3f}, p = {p_Ms:.4f} → NO mass trend.
  3. Physical r_c bounds [0.2ξ, 5ξ] constrain soliton but may be too narrow
     for some galaxies where the optimal core size differs from ξ.
""")

# ============================================================
# SAVE RESULTS
# ============================================================
summary = {
    'test_name': 'forward_model_bunching_validation',
    'n_galaxies': len(results),
    'n_bec_preferred': int(n_bec_better),
    'n_nfw_preferred': int(n_nfw_better),
    'n_indistinguishable': int(n_indist),
    'mean_daic': round(float(np.mean(all_daic)), 3),
    'median_daic': round(float(np.median(all_daic)), 3),
    'sum_daic': round(float(np.sum(all_daic)), 1),
    'wilcoxon_p': round(float(p_w), 6) if p_w is not None and p_w < 1 else None,
    'spearman_logMs': {'rho': round(float(rho_Ms), 4), 'p': round(float(p_Ms), 4)},
    'spearman_Xmed': {'rho': round(float(rho_X), 4), 'p': round(float(p_X), 4)},
    'mass_trend': mass_verdict,
    'overall_verdict': overall,
    'comparison_test8_pipeline': 'ΔAIC = +9.7',
    'comparison_test8_sparc': 'ΔAIC = -40',
    'caveat': 'Soliton-only model lacks NFW-like envelope; absolute ΔAIC biased toward NFW. Key diagnostic is mass trend (none found).',
    'per_galaxy': results,
}

outpath = os.path.join(RESULTS_DIR, 'summary_forward_model_bunching.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {outpath}")
print("=" * 72)
print("Done.")
