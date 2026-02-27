#!/usr/bin/env python3
"""
SOLITON + NFW COMPOSITE DARK MATTER PROFILE TEST
==================================================

Extends test C1 (forward_model_bunching) by fitting the physically correct
BEC dark matter model: a soliton core + NFW-like envelope.

In lab BECs (Cornell & Wieman 1995), the condensate density peak (ground state)
is surrounded by a thermal cloud (excited quasi-particles). The galaxy-scale
analogue has a soliton core surrounded by an NFW envelope, as seen in
cosmological simulations (Schive+2014, Mocz+2017, May & Springel 2021).

Five models per galaxy:
  Model 1: Pure NFW (2 params)
  Model 2: Burkert (2 params)
  Model 3: Pure Soliton (2 params)
  Model 4: Composite — additive V²_sol + V²_NFW (4 params)
  Model 5: Composite — ξ-constrained r_c (3 params + 1 global)

Key outputs:
  - Model comparison (AIC/BIC) across all galaxies
  - Fitted r_c vs predicted ξ = sqrt(GM/g†) correlation
  - Transition radius r_t vs healing length ξ
  - C2/C3 rehabilitation analysis
  - 4 publication figures
"""

import os
import sys
import json
import csv
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.stats import spearmanr, pearsonr, wilcoxon
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Physical constants
G_SI = 6.674e-11       # m³/(kg s²)
Msun_kg = 1.989e30     # kg
kpc_m = 3.086e19       # m per kpc
gdagger = 1.2e-10      # m/s² (McGaugh+2016)
G_kpc = 4.302e-6       # kpc (km/s)² / Msun

# Mass-to-light ratios (same as C1)
Y_DISK = 0.5
Y_BULGE = 0.7

# ============================================================
# SOLITON MODEL: Precompute M(r) lookup table
# ============================================================
_N_SOL = 2000
_u_max = 50.0
_u_table = np.linspace(0, _u_max, _N_SOL + 1)
_du = _u_table[1] - _u_table[0]
_x = _u_table
_integrand = _x**2 / (1.0 + 0.091 * _x**2)**8
_m_tilde = np.zeros(_N_SOL + 1)
for i in range(1, _N_SOL + 1):
    _m_tilde[i] = _m_tilde[i-1] + 0.5 * (_integrand[i-1] + _integrand[i]) * _du
M_TILDE_INF = _m_tilde[-1]


def soliton_v_circ(r_kpc, rho_c, r_c_kpc):
    """Circular velocity from soliton profile (km/s)."""
    u = np.abs(r_kpc) / r_c_kpc
    m_val = np.interp(u, _u_table, _m_tilde)
    M_r = 4.0 * np.pi * rho_c * r_c_kpc**3 * m_val
    V_sq = np.where(r_kpc > 0, G_kpc * M_r / r_kpc, 0.0)
    return np.sqrt(np.maximum(V_sq, 0.0))


def soliton_enclosed_mass(r_kpc, rho_c, r_c_kpc):
    """Enclosed mass from soliton profile (Msun)."""
    u = np.abs(r_kpc) / r_c_kpc
    m_val = np.interp(u, _u_table, _m_tilde)
    return 4.0 * np.pi * rho_c * r_c_kpc**3 * m_val


def soliton_density(r_kpc, rho_c, r_c_kpc):
    """Soliton density at radius r (Msun/kpc³)."""
    return rho_c / (1.0 + 0.091 * (r_kpc / r_c_kpc)**2)**8


def nfw_v_circ(r_kpc, rho_s, r_s_kpc):
    """Circular velocity from NFW profile (km/s)."""
    x = r_kpc / r_s_kpc
    M_r = 4.0 * np.pi * rho_s * r_s_kpc**3 * (np.log(1.0 + x) - x / (1.0 + x))
    V_sq = np.where(r_kpc > 0, G_kpc * M_r / r_kpc, 0.0)
    return np.sqrt(np.maximum(V_sq, 0.0))


def nfw_enclosed_mass(r_kpc, rho_s, r_s_kpc):
    """Enclosed mass from NFW profile (Msun)."""
    x = r_kpc / r_s_kpc
    return 4.0 * np.pi * rho_s * r_s_kpc**3 * (np.log(1.0 + x) - x / (1.0 + x))


def nfw_density(r_kpc, rho_s, r_s_kpc):
    """NFW density at radius r (Msun/kpc³)."""
    x = r_kpc / r_s_kpc
    return rho_s / (x * (1.0 + x)**2)


# ============================================================
# BURKERT PROFILE
# ============================================================

def burkert_v_circ(r_kpc, rho_0, r_0_kpc):
    """Circular velocity from Burkert (1995) profile (km/s).
    ρ(r) = ρ_0 / [(1 + r/r_0)(1 + (r/r_0)²)]
    M(<r) = π ρ_0 r_0³ [ln(1+x) + ½ ln(1+x²) - arctan(x)]  where x = r/r_0
    """
    x = np.abs(r_kpc) / r_0_kpc
    M_r = np.pi * rho_0 * r_0_kpc**3 * (
        np.log(1.0 + x) + 0.5 * np.log(1.0 + x**2) - np.arctan(x)
    )
    V_sq = np.where(r_kpc > 0, G_kpc * M_r / r_kpc, 0.0)
    return np.sqrt(np.maximum(V_sq, 0.0))


def burkert_density(r_kpc, rho_0, r_0_kpc):
    """Burkert density at radius r (Msun/kpc³)."""
    x = r_kpc / r_0_kpc
    return rho_0 / ((1.0 + x) * (1.0 + x**2))


# ============================================================
# COMPOSITE MODEL: additive V²_soliton + V²_NFW
# ============================================================

def composite_v_circ(r_kpc, rho_c, r_c_kpc, rho_s, r_s_kpc):
    """
    Circular velocity from additive soliton+NFW composite.
    V²_DM = V²_soliton + V²_NFW (independent mass components).
    """
    V_sol = soliton_v_circ(r_kpc, rho_c, r_c_kpc)
    V_nfw = nfw_v_circ(r_kpc, rho_s, r_s_kpc)
    return np.sqrt(V_sol**2 + V_nfw**2)


def find_transition_radius(rho_c, r_c_kpc, rho_s, r_s_kpc):
    """Find radius where soliton density = NFW density (crossover)."""
    # Search from 0.01 to 100 kpc
    r_test = np.logspace(-2, 2, 1000)
    rho_sol = soliton_density(r_test, rho_c, r_c_kpc)
    rho_nfw_vals = nfw_density(r_test, rho_s, r_s_kpc)

    # Find where soliton drops below NFW
    diff = rho_sol - rho_nfw_vals
    crossings = []
    for i in range(len(diff) - 1):
        if diff[i] > 0 and diff[i+1] <= 0:
            # Linear interpolation
            r_cross = r_test[i] - diff[i] * (r_test[i+1] - r_test[i]) / (diff[i+1] - diff[i])
            crossings.append(r_cross)

    if crossings:
        return crossings[0]  # first crossing = transition
    return None


def xi_kpc(logMs):
    """BEC healing length in kpc."""
    Ms_SI = 10.0**logMs * Msun_kg
    return np.sqrt(G_SI * Ms_SI / gdagger) / kpc_m


# ============================================================
# LOAD SPARC DATA
# ============================================================
print("=" * 72)
print("SOLITON + NFW COMPOSITE DARK MATTER PROFILE TEST")
print("=" * 72)

print("\n[1] Loading SPARC rotation curves...")

# Data paths
table2_path = os.path.join(DATA_DIR, 'SPARC_table2_rotmods.dat')
mrt_path = os.path.join(DATA_DIR, 'SPARC_Lelli2016c.mrt')

if not os.path.exists(table2_path):
    # Try project-level data directory
    table2_path = os.path.join(PROJECT_ROOT, 'data', 'sparc', 'SPARC_table2_rotmods.dat')
    mrt_path = os.path.join(PROJECT_ROOT, 'data', 'sparc', 'SPARC_Lelli2016c.mrt')

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
        Vflat = float(parts[14])
        logMs = np.log10(max(0.5 * L36 * 1e9, 1e6))
        sparc_props[name] = {
            'Inc': Inc, 'Q': Q, 'logMs': logMs, 'Vflat': Vflat,
        }
    except (ValueError, IndexError):
        continue

print(f"  {len(galaxies)} rotation curves, {len(sparc_props)} with properties")

# ============================================================
# LOAD PREVIOUS C1 RESULTS (to match galaxy set)
# ============================================================
c1_path = os.path.join(RESULTS_DIR, 'summary_forward_model_bunching.json')
c1_galaxies = set()
if os.path.exists(c1_path):
    with open(c1_path, 'r') as f:
        c1_data = json.load(f)
    c1_galaxies = {g['name'] for g in c1_data.get('per_galaxy', [])}
    print(f"  C1 test had {len(c1_galaxies)} galaxies — matching this set")

# ============================================================
# FIT EACH GALAXY WITH 5 MODELS
# ============================================================
print("\n[2] Fitting 5 models per galaxy...")
print("    (Pure NFW, Burkert, Pure Soliton, Composite Free, Composite ξ-constrained)")
print("    This will take several minutes...")

MIN_POINTS = 8
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

    # If C1 galaxy list exists, only fit those galaxies
    if c1_galaxies and name not in c1_galaxies:
        n_skip += 1
        continue

    gdata = galaxies[name]
    R = gdata['R']
    Vobs = gdata['Vobs']
    eVobs = gdata['eVobs']
    Vdisk = gdata['Vdisk']
    Vgas = gdata['Vgas']
    Vbul = gdata['Vbul']

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

    eVobs = np.maximum(eVobs, np.maximum(5.0, 0.05 * Vobs))

    logMs = prop['logMs']
    xi = xi_kpc(logMs)
    N = len(R)

    # Baryonic velocity squared (fixed)
    Vbar_sq = Y_DISK * Vdisk**2 + np.sign(Vgas) * Vgas**2 + Y_BULGE * np.sign(Vbul) * Vbul**2

    # ---- Model 1: Pure NFW (2 params) ----
    def chi2_nfw(params):
        log_rho_s, log_r_s = params
        V_dm = nfw_v_circ(R, 10.0**log_rho_s, 10.0**log_r_s)
        V_total = np.sqrt(np.maximum(Vbar_sq + V_dm**2, 0.0))
        return np.sum(((Vobs - V_total) / eVobs)**2)

    try:
        res_nfw = differential_evolution(chi2_nfw, bounds=[(4, 10), (-1, 2.5)],
                                          seed=42, maxiter=200, tol=1e-6, polish=True)
        chi2_nfw_val = res_nfw.fun
        nfw_params = res_nfw.x
    except Exception:
        n_skip += 1
        continue

    # ---- Model 2: Burkert (2 params) ----
    def chi2_burk(params):
        log_rho_0, log_r_0 = params
        V_dm = burkert_v_circ(R, 10.0**log_rho_0, 10.0**log_r_0)
        V_total = np.sqrt(np.maximum(Vbar_sq + V_dm**2, 0.0))
        return np.sum(((Vobs - V_total) / eVobs)**2)

    try:
        res_burk = differential_evolution(chi2_burk, bounds=[(4, 10), (-1, 2.5)],
                                            seed=42, maxiter=200, tol=1e-6, polish=True)
        chi2_burk_val = res_burk.fun
        burk_params = res_burk.x
    except Exception:
        n_skip += 1
        continue

    # ---- Model 3: Pure Soliton (2 params) ----
    rc_lo = max(0.2 * xi, 0.05)
    rc_hi = max(5.0 * xi, 2.0)

    def chi2_sol(params):
        log_rho_c, log_r_c = params
        V_dm = soliton_v_circ(R, 10.0**log_rho_c, 10.0**log_r_c)
        V_total = np.sqrt(np.maximum(Vbar_sq + V_dm**2, 0.0))
        return np.sum(((Vobs - V_total) / eVobs)**2)

    try:
        res_sol = differential_evolution(chi2_sol,
                                          bounds=[(4, 10), (np.log10(rc_lo), np.log10(rc_hi))],
                                          seed=42, maxiter=200, tol=1e-6, polish=True)
        chi2_sol_val = res_sol.fun
        sol_params = res_sol.x
    except Exception:
        n_skip += 1
        continue

    # ---- Model 3: Composite Free (4 params) ----
    def chi2_comp(params):
        log_rho_c, log_r_c, log_rho_s, log_r_s = params
        V_dm = composite_v_circ(R, 10.0**log_rho_c, 10.0**log_r_c,
                                 10.0**log_rho_s, 10.0**log_r_s)
        V_total = np.sqrt(np.maximum(Vbar_sq + V_dm**2, 0.0))
        return np.sum(((Vobs - V_total) / eVobs)**2)

    try:
        res_comp = differential_evolution(chi2_comp,
                                           bounds=[(4, 10), (-1.5, 2.0),
                                                   (4, 10), (-1, 2.5)],
                                           seed=42, maxiter=300, tol=1e-6, polish=True)
        chi2_comp_val = res_comp.fun
        comp_params = res_comp.x
    except Exception:
        # Fall back: use NFW result as composite
        chi2_comp_val = chi2_nfw_val
        comp_params = np.array([sol_params[0], sol_params[1], nfw_params[0], nfw_params[1]])

    # ---- Model 4: Composite ξ-constrained (3 params, r_c fixed to f*ξ) ----
    # We'll fit with r_c = xi (f=1 initially), then refine f globally later
    log_rc_fixed = np.log10(max(xi, 0.01))

    def chi2_comp_constrained(params):
        log_rho_c, log_rho_s, log_r_s = params
        V_dm = composite_v_circ(R, 10.0**log_rho_c, 10.0**log_rc_fixed,
                                 10.0**log_rho_s, 10.0**log_r_s)
        V_total = np.sqrt(np.maximum(Vbar_sq + V_dm**2, 0.0))
        return np.sum(((Vobs - V_total) / eVobs)**2)

    try:
        res_constrained = differential_evolution(chi2_comp_constrained,
                                                  bounds=[(4, 10), (4, 10), (-1, 2.5)],
                                                  seed=42, maxiter=300, tol=1e-6, polish=True)
        chi2_constrained_val = res_constrained.fun
        constrained_params = res_constrained.x
    except Exception:
        chi2_constrained_val = chi2_nfw_val + 100  # penalty
        constrained_params = np.array([7.0, 7.0, 1.0])

    # ---- Compute AIC/BIC ----
    k_nfw, k_burk, k_sol, k_comp, k_con = 2, 2, 2, 4, 3
    aic_nfw = chi2_nfw_val + 2 * k_nfw
    aic_burk = chi2_burk_val + 2 * k_burk
    aic_sol = chi2_sol_val + 2 * k_sol
    aic_comp = chi2_comp_val + 2 * k_comp
    aic_con = chi2_constrained_val + 2 * k_con

    bic_nfw = chi2_nfw_val + k_nfw * np.log(N)
    bic_burk = chi2_burk_val + k_burk * np.log(N)
    bic_sol = chi2_sol_val + k_sol * np.log(N)
    bic_comp = chi2_comp_val + k_comp * np.log(N)
    bic_con = chi2_constrained_val + k_con * np.log(N)

    # Winner by AIC
    aic_vals = {'NFW': aic_nfw, 'Burkert': aic_burk, 'Soliton': aic_sol,
                'Composite': aic_comp, 'Constrained': aic_con}
    winner = min(aic_vals, key=aic_vals.get)

    # Transition radius for free composite
    r_t = find_transition_radius(10.0**comp_params[0], 10.0**comp_params[1],
                                  10.0**comp_params[2], 10.0**comp_params[3])

    # X = R/ξ statistics
    X_vals = R / xi
    X_min = float(np.min(X_vals))
    X_med = float(np.median(X_vals))
    X_max = float(np.max(X_vals))

    results.append({
        'name': name,
        'logMs': round(logMs, 2),
        'xi_kpc': round(xi, 3),
        'Vflat': prop['Vflat'],
        'N_pts': N,
        'X_min': round(X_min, 3),
        'X_med': round(X_med, 3),
        'X_max': round(X_max, 3),
        # Chi-squared
        'chi2_nfw': round(float(chi2_nfw_val), 2),
        'chi2_burk': round(float(chi2_burk_val), 2),
        'chi2_sol': round(float(chi2_sol_val), 2),
        'chi2_comp': round(float(chi2_comp_val), 2),
        'chi2_con': round(float(chi2_constrained_val), 2),
        # Reduced chi-squared
        'rchi2_nfw': round(float(chi2_nfw_val / max(N - k_nfw, 1)), 3),
        'rchi2_burk': round(float(chi2_burk_val / max(N - k_burk, 1)), 3),
        'rchi2_sol': round(float(chi2_sol_val / max(N - k_sol, 1)), 3),
        'rchi2_comp': round(float(chi2_comp_val / max(N - k_comp, 1)), 3),
        'rchi2_con': round(float(chi2_constrained_val / max(N - k_con, 1)), 3),
        # AIC
        'aic_nfw': round(float(aic_nfw), 2),
        'aic_burk': round(float(aic_burk), 2),
        'aic_sol': round(float(aic_sol), 2),
        'aic_comp': round(float(aic_comp), 2),
        'aic_con': round(float(aic_con), 2),
        # BIC
        'bic_nfw': round(float(bic_nfw), 2),
        'bic_burk': round(float(bic_burk), 2),
        'bic_sol': round(float(bic_sol), 2),
        'bic_comp': round(float(bic_comp), 2),
        'bic_con': round(float(bic_con), 2),
        # ΔAIC relative to NFW (positive = model better than NFW)
        'daic_burk_nfw': round(float(aic_nfw - aic_burk), 2),
        'daic_sol_nfw': round(float(aic_nfw - aic_sol), 2),
        'daic_comp_nfw': round(float(aic_nfw - aic_comp), 2),
        'daic_con_nfw': round(float(aic_nfw - aic_con), 2),
        'daic_comp_sol': round(float(aic_sol - aic_comp), 2),
        'daic_con_comp': round(float(aic_comp - aic_con), 2),
        'daic_comp_burk': round(float(aic_burk - aic_comp), 2),
        # Winner
        'winner': winner,
        # Fit parameters
        'nfw_log_rhos': round(float(nfw_params[0]), 3),
        'nfw_log_rs': round(float(nfw_params[1]), 3),
        'burk_log_rho0': round(float(burk_params[0]), 3),
        'burk_log_r0': round(float(burk_params[1]), 3),
        'sol_log_rhoc': round(float(sol_params[0]), 3),
        'sol_log_rc': round(float(sol_params[1]), 3),
        'comp_log_rhoc': round(float(comp_params[0]), 3),
        'comp_log_rc': round(float(comp_params[1]), 3),
        'comp_log_rhos': round(float(comp_params[2]), 3),
        'comp_log_rs': round(float(comp_params[3]), 3),
        'con_log_rhoc': round(float(constrained_params[0]), 3),
        'con_log_rhos': round(float(constrained_params[1]), 3),
        'con_log_rs': round(float(constrained_params[2]), 3),
        # Derived quantities
        'fitted_rc_kpc': round(10.0**float(comp_params[1]), 3),
        'r_transition_kpc': round(r_t, 3) if r_t else None,
        # Store R, Vobs, Vbar_sq for figures
        '_R': R.tolist(),
        '_Vobs': Vobs.tolist(),
        '_eVobs': eVobs.tolist(),
        '_Vbar_sq': Vbar_sq.tolist(),
    })

    n_fit += 1
    if n_fit % 10 == 0:
        print(f"    {n_fit} galaxies fitted...")

print(f"\n  Fitted: {n_fit} galaxies, Skipped: {n_skip}")

# ============================================================
# STEP 3: MODEL COMPARISON
# ============================================================
print("\n" + "=" * 72)
print("MODEL COMPARISON")
print("=" * 72)

n_nfw_wins = sum(1 for r in results if r['winner'] == 'NFW')
n_burk_wins = sum(1 for r in results if r['winner'] == 'Burkert')
n_sol_wins = sum(1 for r in results if r['winner'] == 'Soliton')
n_comp_wins = sum(1 for r in results if r['winner'] == 'Composite')
n_con_wins = sum(1 for r in results if r['winner'] == 'Constrained')

print(f"\n  Total: {len(results)} galaxies")
print(f"  NFW wins:         {n_nfw_wins} ({100*n_nfw_wins/len(results):.1f}%)")
print(f"  Burkert wins:     {n_burk_wins} ({100*n_burk_wins/len(results):.1f}%)")
print(f"  Soliton wins:     {n_sol_wins} ({100*n_sol_wins/len(results):.1f}%)")
print(f"  Composite wins:   {n_comp_wins} ({100*n_comp_wins/len(results):.1f}%)")
print(f"  Constrained wins: {n_con_wins} ({100*n_con_wins/len(results):.1f}%)")

# ΔAIC statistics
daic_burk_nfw = np.array([r['daic_burk_nfw'] for r in results])
daic_comp_nfw = np.array([r['daic_comp_nfw'] for r in results])
daic_sol_nfw = np.array([r['daic_sol_nfw'] for r in results])
daic_con_nfw = np.array([r['daic_con_nfw'] for r in results])
daic_comp_sol = np.array([r['daic_comp_sol'] for r in results])
daic_con_comp = np.array([r['daic_con_comp'] for r in results])
daic_comp_burk = np.array([r['daic_comp_burk'] for r in results])

print(f"\n  ΔAIC vs NFW (positive = model better):")
print(f"    Burkert:           mean={np.mean(daic_burk_nfw):+.2f}, "
      f"median={np.median(daic_burk_nfw):+.2f}, sum={np.sum(daic_burk_nfw):+.1f}")
print(f"    Composite Free:    mean={np.mean(daic_comp_nfw):+.2f}, "
      f"median={np.median(daic_comp_nfw):+.2f}, sum={np.sum(daic_comp_nfw):+.1f}")
print(f"    Pure Soliton:      mean={np.mean(daic_sol_nfw):+.2f}, "
      f"median={np.median(daic_sol_nfw):+.2f}, sum={np.sum(daic_sol_nfw):+.1f}")
print(f"    ξ-Constrained:     mean={np.mean(daic_con_nfw):+.2f}, "
      f"median={np.median(daic_con_nfw):+.2f}, sum={np.sum(daic_con_nfw):+.1f}")

print(f"\n  Composite vs Burkert (positive = composite better):")
print(f"    mean={np.mean(daic_comp_burk):+.2f}, median={np.median(daic_comp_burk):+.2f}")

print(f"\n  Composite vs Soliton (positive = composite better):")
print(f"    mean={np.mean(daic_comp_sol):+.2f}, median={np.median(daic_comp_sol):+.2f}")

print(f"\n  ξ-Constrained vs Composite (positive = constrained better, neg = constraint cost):")
print(f"    mean={np.mean(daic_con_comp):+.2f}, median={np.median(daic_con_comp):+.2f}")

# Composite significantly better than NFW
n_comp_sig_better = sum(1 for d in daic_comp_nfw if d > 2)
n_comp_sig_worse = sum(1 for d in daic_comp_nfw if d < -2)
print(f"\n  Composite vs NFW:")
print(f"    Composite significantly better (ΔAIC>2): {n_comp_sig_better}")
print(f"    NFW significantly better (ΔAIC<-2):      {n_comp_sig_worse}")
print(f"    Indistinguishable (|ΔAIC|≤2):            {len(results) - n_comp_sig_better - n_comp_sig_worse}")

# Burkert vs NFW
n_burk_sig_better = sum(1 for d in daic_burk_nfw if d > 2)
n_burk_sig_worse = sum(1 for d in daic_burk_nfw if d < -2)
print(f"\n  Burkert vs NFW:")
print(f"    Burkert significantly better (ΔAIC>2):   {n_burk_sig_better}")
print(f"    NFW significantly better (ΔAIC<-2):      {n_burk_sig_worse}")
print(f"    Indistinguishable (|ΔAIC|≤2):            {len(results) - n_burk_sig_better - n_burk_sig_worse}")

# Mean chi²/dof
print(f"\n  Mean reduced χ²:")
print(f"    NFW:         {np.mean([r['rchi2_nfw'] for r in results]):.3f}")
print(f"    Burkert:     {np.mean([r['rchi2_burk'] for r in results]):.3f}")
print(f"    Soliton:     {np.mean([r['rchi2_sol'] for r in results]):.3f}")
print(f"    Composite:   {np.mean([r['rchi2_comp'] for r in results]):.3f}")
print(f"    Constrained: {np.mean([r['rchi2_con'] for r in results]):.3f}")

# ============================================================
# STEP 4: CORE RADIUS vs HEALING LENGTH
# ============================================================
print("\n" + "=" * 72)
print("CORE RADIUS vs HEALING LENGTH VALIDATION")
print("=" * 72)

# Extract fitted r_c and ξ
fitted_rc = np.array([r['fitted_rc_kpc'] for r in results])
xi_vals = np.array([r['xi_kpc'] for r in results])
logMs_vals = np.array([r['logMs'] for r in results])

# r_c / ξ ratio
rc_over_xi = fitted_rc / xi_vals

# Correlation
rho_pearson, p_pearson = pearsonr(np.log10(xi_vals), np.log10(fitted_rc))
rho_spearman, p_spearman = spearmanr(xi_vals, fitted_rc)

print(f"\n  Pearson r (log-log):  r = {rho_pearson:.4f}, p = {p_pearson:.6f}")
print(f"  Spearman ρ:           ρ = {rho_spearman:.4f}, p = {p_spearman:.6f}")
print(f"\n  r_c / ξ ratio:")
print(f"    Mean:   {np.mean(rc_over_xi):.3f}")
print(f"    Median: {np.median(rc_over_xi):.3f}")
print(f"    Std:    {np.std(rc_over_xi):.3f}")
print(f"    [16, 84] percentile: [{np.percentile(rc_over_xi, 16):.3f}, "
      f"{np.percentile(rc_over_xi, 84):.3f}]")

# ============================================================
# STEP 5: TRANSITION RADIUS ANALYSIS
# ============================================================
print("\n" + "=" * 72)
print("TRANSITION RADIUS ANALYSIS")
print("=" * 72)

rt_vals = np.array([r['r_transition_kpc'] for r in results if r['r_transition_kpc'] is not None])
xi_for_rt = np.array([r['xi_kpc'] for r in results if r['r_transition_kpc'] is not None])

if len(rt_vals) >= 5:
    rt_over_xi = rt_vals / xi_for_rt
    rho_rt, p_rt = spearmanr(xi_for_rt, rt_vals)
    print(f"\n  {len(rt_vals)} galaxies with detected transition radius")
    print(f"  Spearman (r_t vs ξ): ρ = {rho_rt:.4f}, p = {p_rt:.6f}")
    print(f"  r_t / ξ: mean = {np.mean(rt_over_xi):.3f}, "
          f"median = {np.median(rt_over_xi):.3f}")
else:
    print(f"\n  Only {len(rt_vals)} galaxies with transition — insufficient for analysis")
    rt_over_xi = np.array([])

# ============================================================
# STEP 6: GLOBAL f PARAMETER (ξ-constrained refinement)
# ============================================================
print("\n" + "=" * 72)
print("GLOBAL f PARAMETER REFINEMENT")
print("=" * 72)

# Find optimal f such that r_c = f × ξ minimizes total chi² across all galaxies
# Try a grid of f values
f_grid = np.logspace(-1.0, 1.0, 21)  # 0.1 to 10
total_chi2_per_f = []

for f_val in f_grid:
    total_chi2 = 0.0
    for r in results:
        R_arr = np.array(r['_R'])
        Vobs_arr = np.array(r['_Vobs'])
        eVobs_arr = np.array(r['_eVobs'])
        Vbar_sq_arr = np.array(r['_Vbar_sq'])

        # Use constrained params but with r_c = f * xi
        rc_f = f_val * r['xi_kpc']
        rho_c = 10.0**r['con_log_rhoc']
        rho_s = 10.0**r['con_log_rhos']
        r_s = 10.0**r['con_log_rs']

        V_dm = composite_v_circ(R_arr, rho_c, max(rc_f, 0.01), rho_s, r_s)
        V_total = np.sqrt(np.maximum(Vbar_sq_arr + V_dm**2, 0.0))
        chi2 = np.sum(((Vobs_arr - V_total) / eVobs_arr)**2)
        total_chi2 += chi2
    total_chi2_per_f.append(total_chi2)

best_f_idx = np.argmin(total_chi2_per_f)
best_f = f_grid[best_f_idx]
print(f"\n  Best global f = {best_f:.3f} (r_c = {best_f:.3f} × ξ)")
print(f"  Total χ² at f=1: {total_chi2_per_f[10]:.1f}")
print(f"  Total χ² at f={best_f:.3f}: {total_chi2_per_f[best_f_idx]:.1f}")
print(f"  Improvement: {total_chi2_per_f[10] - total_chi2_per_f[best_f_idx]:.1f}")

# ============================================================
# STEP 7: MASS-DEPENDENT TREND
# ============================================================
print("\n" + "=" * 72)
print("MASS-DEPENDENT TREND")
print("=" * 72)

# Does composite advantage correlate with stellar mass or X?
all_logMs = np.array([r['logMs'] for r in results])
all_Xmed = np.array([r['X_med'] for r in results])

rho_comp_Ms, p_comp_Ms = spearmanr(all_logMs, daic_comp_nfw)
rho_comp_X, p_comp_X = spearmanr(all_Xmed, daic_comp_nfw)
rho_burk_Ms, p_burk_Ms = spearmanr(all_logMs, daic_burk_nfw)

print(f"\n  ΔAIC(Composite-NFW) vs logMs: ρ = {rho_comp_Ms:+.3f}, p = {p_comp_Ms:.4f}")
print(f"  ΔAIC(Composite-NFW) vs X_med: ρ = {rho_comp_X:+.3f}, p = {p_comp_X:.4f}")
print(f"  ΔAIC(Burkert-NFW)   vs logMs: ρ = {rho_burk_Ms:+.3f}, p = {p_burk_Ms:.4f}")

# Mass-binned
mass_bins = [
    ('Dwarfs (logMs<9)', lambda m: m < 9),
    ('Low-mass (9-9.5)', lambda m: 9 <= m < 9.5),
    ('Intermediate (9.5-10)', lambda m: 9.5 <= m < 10),
    ('Massive (10-10.5)', lambda m: 10 <= m < 10.5),
    ('Very massive (>10.5)', lambda m: m >= 10.5),
]

print(f"\n  {'Mass bin':25s} {'N':>4s} {'<ΔAIC_comp>':>12s} {'<ΔAIC_burk>':>12s} {'med_comp':>9s} "
      f"{'N_comp':>6s} {'N_burk':>6s} {'N_NFW':>6s}")
print(f"  {'-'*88}")

for label, selector in mass_bins:
    mask = np.array([selector(m) for m in all_logMs])
    if np.sum(mask) < 2:
        continue
    bin_daic = daic_comp_nfw[mask]
    bin_daic_burk = daic_burk_nfw[mask]
    n_comp = int(np.sum(bin_daic > 2))
    n_burk = int(np.sum(bin_daic_burk > 2))
    n_nfw = int(np.sum(bin_daic < -2))
    print(f"  {label:25s} {np.sum(mask):4d} {np.mean(bin_daic):+12.2f} "
          f"{np.mean(bin_daic_burk):+12.2f} {np.median(bin_daic):+9.2f} "
          f"{n_comp:6d} {n_burk:6d} {n_nfw:6d}")

# ============================================================
# STEP 8: C2/C3 REHABILITATION
# ============================================================
print("\n" + "=" * 72)
print("C2/C3 REHABILITATION")
print("=" * 72)

# C2: Inner region (R < ξ) RAR residual analysis
print("\n  C2: Inner region (R < ξ) residual analysis")
print("  Previous C1 finding: inner deficit of -0.181 dex")

inner_res_nfw = []
inner_res_burk = []
inner_res_comp = []
inner_res_sol = []
n_inner_gal = 0

for r in results:
    R_arr = np.array(r['_R'])
    Vobs_arr = np.array(r['_Vobs'])
    eVobs_arr = np.array(r['_eVobs'])
    Vbar_sq_arr = np.array(r['_Vbar_sq'])
    xi_val = r['xi_kpc']

    inner_mask = R_arr < xi_val
    if np.sum(inner_mask) < 3:
        continue
    n_inner_gal += 1

    R_inner = R_arr[inner_mask]
    Vobs_inner = Vobs_arr[inner_mask]
    Vbar_sq_inner = Vbar_sq_arr[inner_mask]

    # NFW prediction
    V_nfw = nfw_v_circ(R_inner, 10.0**r['nfw_log_rhos'], 10.0**r['nfw_log_rs'])
    V_nfw_total = np.sqrt(np.maximum(Vbar_sq_inner + V_nfw**2, 0.0))

    # Burkert prediction
    V_burk = burkert_v_circ(R_inner, 10.0**r['burk_log_rho0'], 10.0**r['burk_log_r0'])
    V_burk_total = np.sqrt(np.maximum(Vbar_sq_inner + V_burk**2, 0.0))

    # Composite prediction
    V_comp = composite_v_circ(R_inner, 10.0**r['comp_log_rhoc'], 10.0**r['comp_log_rc'],
                               10.0**r['comp_log_rhos'], 10.0**r['comp_log_rs'])
    V_comp_total = np.sqrt(np.maximum(Vbar_sq_inner + V_comp**2, 0.0))

    # Soliton prediction
    V_sol = soliton_v_circ(R_inner, 10.0**r['sol_log_rhoc'], 10.0**r['sol_log_rc'])
    V_sol_total = np.sqrt(np.maximum(Vbar_sq_inner + V_sol**2, 0.0))

    # Residuals: log(Vobs / Vmodel)
    for Vobs_i, V_model in zip(Vobs_inner, V_nfw_total):
        if Vobs_i > 0:
            inner_res_nfw.append(np.log10(Vobs_i / max(V_model, 1.0)))
    for Vobs_i, V_model in zip(Vobs_inner, V_burk_total):
        if Vobs_i > 0:
            inner_res_burk.append(np.log10(Vobs_i / max(V_model, 1.0)))
    for Vobs_i, V_model in zip(Vobs_inner, V_comp_total):
        if Vobs_i > 0:
            inner_res_comp.append(np.log10(Vobs_i / max(V_model, 1.0)))
    for Vobs_i, V_model in zip(Vobs_inner, V_sol_total):
        if Vobs_i > 0:
            inner_res_sol.append(np.log10(Vobs_i / max(V_model, 1.0)))

inner_res_nfw = np.array(inner_res_nfw)
inner_res_burk = np.array(inner_res_burk)
inner_res_comp = np.array(inner_res_comp)
inner_res_sol = np.array(inner_res_sol)

print(f"  {n_inner_gal} galaxies with ≥3 points inside R < ξ")
print(f"\n  Inner region mean residual (log Vobs - log Vmodel):")
print(f"    NFW:       {np.mean(inner_res_nfw):+.4f} dex ({len(inner_res_nfw)} pts)")
print(f"    Burkert:   {np.mean(inner_res_burk):+.4f} dex ({len(inner_res_burk)} pts)")
print(f"    Composite: {np.mean(inner_res_comp):+.4f} dex ({len(inner_res_comp)} pts)")
print(f"    Soliton:   {np.mean(inner_res_sol):+.4f} dex ({len(inner_res_sol)} pts)")

print(f"\n  Inner region RMS residual:")
print(f"    NFW:       {np.std(inner_res_nfw):.4f} dex")
print(f"    Burkert:   {np.std(inner_res_burk):.4f} dex")
print(f"    Composite: {np.std(inner_res_comp):.4f} dex")
print(f"    Soliton:   {np.std(inner_res_sol):.4f} dex")

if np.std(inner_res_comp) < np.std(inner_res_nfw):
    c2_rehab = "Composite reduces inner scatter (partial rehabilitation)"
else:
    c2_rehab = "Composite does not reduce inner scatter"
print(f"  C2 rehabilitation: {c2_rehab}")

# C3: Radial variance profile
print(f"\n  C3: Radial variance profile analysis")
X_bins = [(-1.5, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]
print(f"  {'logX bin':15s} {'σ_NFW':>8s} {'σ_Burk':>8s} {'σ_Comp':>8s} {'σ_Sol':>8s} {'N':>6s}")
print(f"  {'-'*60}")

c3_bins = []
for logX_lo, logX_hi in X_bins:
    nfw_var_pts = []
    burk_var_pts = []
    comp_var_pts = []
    sol_var_pts = []

    for r_data in results:
        R_arr = np.array(r_data['_R'])
        Vobs_arr = np.array(r_data['_Vobs'])
        Vbar_sq_arr = np.array(r_data['_Vbar_sq'])
        xi_val = r_data['xi_kpc']

        X_arr = R_arr / xi_val
        logX_arr = np.log10(np.maximum(X_arr, 1e-5))
        bin_mask = (logX_arr >= logX_lo) & (logX_arr < logX_hi)
        if np.sum(bin_mask) < 2:
            continue

        R_bin = R_arr[bin_mask]
        Vobs_bin = Vobs_arr[bin_mask]
        Vbar_sq_bin = Vbar_sq_arr[bin_mask]

        V_nfw_b = nfw_v_circ(R_bin, 10.0**r_data['nfw_log_rhos'], 10.0**r_data['nfw_log_rs'])
        V_burk_b = burkert_v_circ(R_bin, 10.0**r_data['burk_log_rho0'], 10.0**r_data['burk_log_r0'])
        V_comp_b = composite_v_circ(R_bin, 10.0**r_data['comp_log_rhoc'], 10.0**r_data['comp_log_rc'],
                                     10.0**r_data['comp_log_rhos'], 10.0**r_data['comp_log_rs'])
        V_sol_b = soliton_v_circ(R_bin, 10.0**r_data['sol_log_rhoc'], 10.0**r_data['sol_log_rc'])

        V_nfw_t = np.sqrt(np.maximum(Vbar_sq_bin + V_nfw_b**2, 0.0))
        V_burk_t = np.sqrt(np.maximum(Vbar_sq_bin + V_burk_b**2, 0.0))
        V_comp_t = np.sqrt(np.maximum(Vbar_sq_bin + V_comp_b**2, 0.0))
        V_sol_t = np.sqrt(np.maximum(Vbar_sq_bin + V_sol_b**2, 0.0))

        for i in range(len(Vobs_bin)):
            if Vobs_bin[i] > 0:
                nfw_var_pts.append((Vobs_bin[i] - V_nfw_t[i])**2)
                burk_var_pts.append((Vobs_bin[i] - V_burk_t[i])**2)
                comp_var_pts.append((Vobs_bin[i] - V_comp_t[i])**2)
                sol_var_pts.append((Vobs_bin[i] - V_sol_t[i])**2)

    if len(nfw_var_pts) >= 5:
        s_nfw = np.sqrt(np.mean(nfw_var_pts))
        s_burk = np.sqrt(np.mean(burk_var_pts))
        s_comp = np.sqrt(np.mean(comp_var_pts))
        s_sol = np.sqrt(np.mean(sol_var_pts))
        print(f"  [{logX_lo:+.1f},{logX_hi:+.1f}]     {s_nfw:8.2f} {s_burk:8.2f} {s_comp:8.2f} {s_sol:8.2f} {len(nfw_var_pts):6d}")
        c3_bins.append({
            'logX_lo': logX_lo, 'logX_hi': logX_hi,
            'sigma_nfw': round(s_nfw, 4), 'sigma_burk': round(s_burk, 4),
            'sigma_comp': round(s_comp, 4), 'sigma_sol': round(s_sol, 4),
            'N': len(nfw_var_pts)
        })

# ============================================================
# STEP 9: SUMMARY TABLE
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY TABLE")
print("=" * 72)

print(f"\n  {'Metric':35s} {'NFW':>10s} {'Burkert':>10s} {'Soliton':>10s} {'Comp.Free':>10s} {'Comp.ξ':>10s}")
print(f"  {'-'*90}")
print(f"  {'N parameters':35s} {'2':>10s} {'2':>10s} {'2':>10s} {'4':>10s} {'3+1':>10s}")
print(f"  {'Mean χ²/dof':35s} "
      f"{np.mean([r['rchi2_nfw'] for r in results]):10.3f} "
      f"{np.mean([r['rchi2_burk'] for r in results]):10.3f} "
      f"{np.mean([r['rchi2_sol'] for r in results]):10.3f} "
      f"{np.mean([r['rchi2_comp'] for r in results]):10.3f} "
      f"{np.mean([r['rchi2_con'] for r in results]):10.3f}")
print(f"  {'Median ΔAIC vs NFW':35s} {'—':>10s} "
      f"{np.median(daic_burk_nfw):+10.2f} "
      f"{np.median(daic_sol_nfw):+10.2f} "
      f"{np.median(daic_comp_nfw):+10.2f} "
      f"{np.median(daic_con_nfw):+10.2f}")
print(f"  {'% galaxies winning':35s} "
      f"{100*n_nfw_wins/len(results):9.1f}% "
      f"{100*n_burk_wins/len(results):9.1f}% "
      f"{100*n_sol_wins/len(results):9.1f}% "
      f"{100*n_comp_wins/len(results):9.1f}% "
      f"{100*n_con_wins/len(results):9.1f}%")
print(f"  {'Sum ΔAIC vs NFW':35s} {'—':>10s} "
      f"{np.sum(daic_burk_nfw):+10.1f} "
      f"{np.sum(daic_sol_nfw):+10.1f} "
      f"{np.sum(daic_comp_nfw):+10.1f} "
      f"{np.sum(daic_con_nfw):+10.1f}")

if rho_pearson is not None:
    print(f"  {'r_c vs ξ Pearson r (log-log)':35s} {'—':>10s} {'—':>10s} {'—':>10s} "
          f"{rho_pearson:+10.4f} {'(constraint)':>10s}")
if len(rt_vals) >= 5:
    rho_rt_p, _ = spearmanr(xi_for_rt, rt_vals)
    print(f"  {'r_t vs ξ Spearman ρ':35s} {'—':>10s} {'—':>10s} {'—':>10s} "
          f"{rho_rt_p:+10.4f} {'—':>10s}")

# ============================================================
# STEP 10: OVERALL VERDICT
# ============================================================
print("\n" + "=" * 72)
print("OVERALL VERDICT")
print("=" * 72)

comp_fraction = (n_comp_wins + n_con_wins) / len(results)
if comp_fraction > 0.6:
    verdict = "BEC-CONSISTENT"
    detail = (f"Composite model wins for {100*comp_fraction:.0f}% of galaxies. "
              f"Soliton core + NFW envelope outperforms both pure models.")
elif comp_fraction > 0.4:
    verdict = "MIXED — PARTIAL SUPPORT"
    detail = (f"Composite wins for {100*comp_fraction:.0f}% of galaxies. "
              f"Evidence for soliton cores but not dominant.")
elif np.median(daic_comp_nfw) > 0:
    verdict = "MARGINAL"
    detail = (f"Composite slightly preferred on average but doesn't dominate. "
              f"Median ΔAIC(comp-NFW) = {np.median(daic_comp_nfw):+.2f}")
else:
    verdict = "NFW PREFERRED"
    detail = (f"Pure NFW still competitive. Composite doesn't justify extra parameters. "
              f"Median ΔAIC(comp-NFW) = {np.median(daic_comp_nfw):+.2f}")

print(f"\n  Verdict: {verdict}")
print(f"  Detail: {detail}")

if rho_pearson > 0.7 and p_pearson < 0.01:
    print(f"  Healing length: VALIDATED — fitted r_c correlates with ξ (r={rho_pearson:.3f})")
elif rho_pearson > 0.4 and p_pearson < 0.05:
    print(f"  Healing length: PARTIAL — weak correlation (r={rho_pearson:.3f})")
else:
    print(f"  Healing length: NOT VALIDATED — r_c does not correlate with ξ (r={rho_pearson:.3f})")

print(f"  Best global f = {best_f:.3f} (r_c = f × ξ)")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 72)
print("SAVING RESULTS")
print("=" * 72)

# Strip internal arrays from per-galaxy results for JSON
results_clean = []
for r in results:
    r_clean = {k: v for k, v in r.items() if not k.startswith('_')}
    results_clean.append(r_clean)

summary = {
    'test_name': 'soliton_nfw_composite',
    'description': 'Composite soliton core + NFW envelope dark matter profile test',
    'n_galaxies': len(results),
    'model_comparison': {
        'n_nfw_wins': n_nfw_wins,
        'n_burkert_wins': n_burk_wins,
        'n_soliton_wins': n_sol_wins,
        'n_composite_wins': n_comp_wins,
        'n_constrained_wins': n_con_wins,
        'pct_nfw': round(100 * n_nfw_wins / len(results), 1),
        'pct_burkert': round(100 * n_burk_wins / len(results), 1),
        'pct_soliton': round(100 * n_sol_wins / len(results), 1),
        'pct_composite': round(100 * n_comp_wins / len(results), 1),
        'pct_constrained': round(100 * n_con_wins / len(results), 1),
    },
    'daic_vs_nfw': {
        'burkert': {
            'mean': round(float(np.mean(daic_burk_nfw)), 2),
            'median': round(float(np.median(daic_burk_nfw)), 2),
            'sum': round(float(np.sum(daic_burk_nfw)), 1),
        },
        'composite_free': {
            'mean': round(float(np.mean(daic_comp_nfw)), 2),
            'median': round(float(np.median(daic_comp_nfw)), 2),
            'sum': round(float(np.sum(daic_comp_nfw)), 1),
        },
        'soliton': {
            'mean': round(float(np.mean(daic_sol_nfw)), 2),
            'median': round(float(np.median(daic_sol_nfw)), 2),
            'sum': round(float(np.sum(daic_sol_nfw)), 1),
        },
        'constrained': {
            'mean': round(float(np.mean(daic_con_nfw)), 2),
            'median': round(float(np.median(daic_con_nfw)), 2),
            'sum': round(float(np.sum(daic_con_nfw)), 1),
        },
    },
    'mean_rchi2': {
        'nfw': round(float(np.mean([r['rchi2_nfw'] for r in results])), 3),
        'burkert': round(float(np.mean([r['rchi2_burk'] for r in results])), 3),
        'soliton': round(float(np.mean([r['rchi2_sol'] for r in results])), 3),
        'composite': round(float(np.mean([r['rchi2_comp'] for r in results])), 3),
        'constrained': round(float(np.mean([r['rchi2_con'] for r in results])), 3),
    },
    'healing_length_validation': {
        'pearson_r': round(float(rho_pearson), 4),
        'pearson_p': round(float(p_pearson), 6),
        'spearman_rho': round(float(rho_spearman), 4),
        'spearman_p': round(float(p_spearman), 6),
        'rc_over_xi_mean': round(float(np.mean(rc_over_xi)), 3),
        'rc_over_xi_median': round(float(np.median(rc_over_xi)), 3),
        'rc_over_xi_std': round(float(np.std(rc_over_xi)), 3),
    },
    'transition_radius': {
        'n_detected': len(rt_vals),
        'rt_over_xi_mean': round(float(np.mean(rt_over_xi)), 3) if len(rt_over_xi) > 0 else None,
        'rt_over_xi_median': round(float(np.median(rt_over_xi)), 3) if len(rt_over_xi) > 0 else None,
    },
    'global_f': round(float(best_f), 3),
    'mass_trend': {
        'daic_comp_vs_logMs': {'rho': round(float(rho_comp_Ms), 4), 'p': round(float(p_comp_Ms), 4)},
        'daic_comp_vs_Xmed': {'rho': round(float(rho_comp_X), 4), 'p': round(float(p_comp_X), 4)},
        'daic_burk_vs_logMs': {'rho': round(float(rho_burk_Ms), 4), 'p': round(float(p_burk_Ms), 4)},
    },
    'c2_rehabilitation': {
        'inner_mean_res_nfw': round(float(np.mean(inner_res_nfw)), 4),
        'inner_mean_res_burkert': round(float(np.mean(inner_res_burk)), 4),
        'inner_mean_res_composite': round(float(np.mean(inner_res_comp)), 4),
        'inner_mean_res_soliton': round(float(np.mean(inner_res_sol)), 4),
        'inner_rms_nfw': round(float(np.std(inner_res_nfw)), 4),
        'inner_rms_burkert': round(float(np.std(inner_res_burk)), 4),
        'inner_rms_composite': round(float(np.std(inner_res_comp)), 4),
        'inner_rms_soliton': round(float(np.std(inner_res_sol)), 4),
        'verdict': c2_rehab,
    },
    'c3_radial_variance': c3_bins,
    'verdict': verdict,
    'per_galaxy': results_clean,
}

outpath = os.path.join(RESULTS_DIR, 'summary_soliton_nfw_composite.json')
with open(outpath, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {outpath}")

# Save per-galaxy CSV
csv_path = os.path.join(RESULTS_DIR, 'composite_per_galaxy.csv')
fieldnames = ['name', 'logMs', 'xi_kpc', 'Vflat', 'N_pts', 'X_med',
              'chi2_nfw', 'chi2_burk', 'chi2_sol', 'chi2_comp', 'chi2_con',
              'aic_nfw', 'aic_burk', 'aic_sol', 'aic_comp', 'aic_con',
              'daic_burk_nfw', 'daic_comp_nfw', 'daic_sol_nfw', 'daic_con_nfw', 'winner',
              'fitted_rc_kpc', 'r_transition_kpc', 'comp_log_rc']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for r in results_clean:
        writer.writerow(r)
print(f"  Saved: {csv_path}")

# ============================================================
# FIGURES
# ============================================================
print("\n" + "=" * 72)
print("GENERATING FIGURES")
print("=" * 72)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- Figure 1: Model Comparison ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel a: ΔAIC histogram (composite - NFW)
    ax = axes[0, 0]
    ax.hist(daic_comp_nfw, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ΔAIC=0')
    ax.axvline(np.median(daic_comp_nfw), color='orange', linestyle='-', linewidth=2,
               label=f'Median={np.median(daic_comp_nfw):+.1f}')
    ax.set_xlabel('ΔAIC (NFW - Composite)', fontsize=12)
    ax.set_ylabel('N galaxies', fontsize=12)
    ax.set_title('(a) Composite Free vs NFW', fontsize=13)
    ax.legend(fontsize=10)

    # Panel b: ΔAIC histogram (Burkert - NFW)
    ax = axes[0, 1]
    ax.hist(daic_burk_nfw, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ΔAIC=0')
    ax.axvline(np.median(daic_burk_nfw), color='orange', linestyle='-', linewidth=2,
               label=f'Median={np.median(daic_burk_nfw):+.1f}')
    ax.set_xlabel('ΔAIC (NFW - Burkert)', fontsize=12)
    ax.set_ylabel('N galaxies', fontsize=12)
    ax.set_title('(b) Burkert vs NFW', fontsize=13)
    ax.legend(fontsize=10)

    # Panel c: Win fraction vs stellar mass
    ax = axes[1, 0]
    mass_bin_edges = [7.5, 8.5, 9.0, 9.5, 10.0, 10.5, 11.5]
    mass_centers = [(mass_bin_edges[i] + mass_bin_edges[i+1]) / 2 for i in range(len(mass_bin_edges)-1)]
    frac_nfw_bins = []
    frac_burk_bins = []
    frac_comp_bins = []
    frac_sol_bins = []
    for i in range(len(mass_bin_edges) - 1):
        mask = (all_logMs >= mass_bin_edges[i]) & (all_logMs < mass_bin_edges[i+1])
        n_in_bin = np.sum(mask)
        if n_in_bin < 2:
            frac_nfw_bins.append(np.nan)
            frac_burk_bins.append(np.nan)
            frac_comp_bins.append(np.nan)
            frac_sol_bins.append(np.nan)
            continue
        winners_in_bin = [results[j]['winner'] for j in range(len(results)) if mask[j]]
        frac_nfw_bins.append(winners_in_bin.count('NFW') / n_in_bin)
        frac_burk_bins.append(winners_in_bin.count('Burkert') / n_in_bin)
        frac_comp_bins.append((winners_in_bin.count('Composite') + winners_in_bin.count('Constrained')) / n_in_bin)
        frac_sol_bins.append(winners_in_bin.count('Soliton') / n_in_bin)

    ax.plot(mass_centers, frac_nfw_bins, 'o-', color='darkblue', label='NFW', linewidth=2, markersize=8)
    ax.plot(mass_centers, frac_burk_bins, 'D-', color='purple', label='Burkert', linewidth=2, markersize=7)
    ax.plot(mass_centers, frac_comp_bins, 's-', color='darkred', label='Composite', linewidth=2, markersize=8)
    ax.plot(mass_centers, frac_sol_bins, '^-', color='darkgreen', label='Soliton', linewidth=2, markersize=8)
    ax.set_xlabel('log(M*/M_sun)', fontsize=12)
    ax.set_ylabel('Fraction winning', fontsize=12)
    ax.set_title('(c) Win fraction vs stellar mass', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    # Panel d: Win fraction vs X_median
    ax = axes[1, 1]
    X_bin_edges = [0.01, 0.3, 1.0, 3.0, 10.0, 100.0]
    X_centers_plot = [np.sqrt(X_bin_edges[i] * X_bin_edges[i+1]) for i in range(len(X_bin_edges)-1)]
    frac_nfw_X = []
    frac_burk_X = []
    frac_comp_X = []
    for i in range(len(X_bin_edges) - 1):
        mask = (all_Xmed >= X_bin_edges[i]) & (all_Xmed < X_bin_edges[i+1])
        n_in_bin = np.sum(mask)
        if n_in_bin < 2:
            frac_nfw_X.append(np.nan)
            frac_burk_X.append(np.nan)
            frac_comp_X.append(np.nan)
            continue
        winners_in_bin = [results[j]['winner'] for j in range(len(results)) if mask[j]]
        frac_nfw_X.append(winners_in_bin.count('NFW') / n_in_bin)
        frac_burk_X.append(winners_in_bin.count('Burkert') / n_in_bin)
        frac_comp_X.append((winners_in_bin.count('Composite') + winners_in_bin.count('Constrained')) / n_in_bin)

    ax.semilogx(X_centers_plot, frac_nfw_X, 'o-', color='darkblue', label='NFW', linewidth=2, markersize=8)
    ax.semilogx(X_centers_plot, frac_burk_X, 'D-', color='purple', label='Burkert', linewidth=2, markersize=7)
    ax.semilogx(X_centers_plot, frac_comp_X, 's-', color='darkred', label='Composite', linewidth=2, markersize=8)
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, label='X=1 (R=ξ)')
    ax.set_xlabel('X_median = R_median / ξ', fontsize=12)
    ax.set_ylabel('Fraction winning', fontsize=12)
    ax.set_title('(d) Win fraction vs X_median', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, 'soliton_nfw_model_comparison.png')
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig1_path}")

    # ---- Figure 2: Healing Length Validation ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel a: r_c vs ξ
    ax = axes[0]
    sc = ax.scatter(xi_vals, fitted_rc, c=logMs_vals, cmap='viridis', s=40, alpha=0.7,
                     edgecolors='black', linewidth=0.5)
    lim_lo = min(xi_vals.min(), fitted_rc.min()) * 0.5
    lim_hi = max(xi_vals.max(), fitted_rc.max()) * 2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', linewidth=1.5, label='1:1')
    ax.plot([lim_lo, lim_hi], [best_f * lim_lo, best_f * lim_hi], 'r-', linewidth=1.5,
            label=f'r_c = {best_f:.2f}ξ')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ξ = √(GM*/g†) [kpc]', fontsize=12)
    ax.set_ylabel('Fitted r_c [kpc]', fontsize=12)
    ax.set_title(f'(a) Core radius vs healing length\nr={rho_pearson:.3f}, p={p_pearson:.1e}', fontsize=12)
    ax.legend(fontsize=10)
    plt.colorbar(sc, ax=ax, label='log(M*/M_sun)')

    # Panel b: r_transition vs ξ
    ax = axes[1]
    if len(rt_vals) >= 5:
        ax.scatter(xi_for_rt, rt_vals, c='steelblue', s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
        lim_lo2 = min(xi_for_rt.min(), rt_vals.min()) * 0.5
        lim_hi2 = max(xi_for_rt.max(), rt_vals.max()) * 2
        ax.plot([lim_lo2, lim_hi2], [lim_lo2, lim_hi2], 'k--', linewidth=1.5, label='1:1')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('ξ [kpc]', fontsize=12)
        ax.set_ylabel('r_transition [kpc]', fontsize=12)
        rho_rt_val, _ = spearmanr(xi_for_rt, rt_vals)
        ax.set_title(f'(b) Transition radius vs ξ\nρ={rho_rt_val:.3f}', fontsize=12)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, f'Only {len(rt_vals)} galaxies\nwith transition', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        ax.set_title('(b) Transition radius vs ξ', fontsize=12)

    # Panel c: r_c/ξ distribution
    ax = axes[2]
    ax.hist(rc_over_xi, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='r_c = ξ')
    ax.axvline(np.median(rc_over_xi), color='orange', linestyle='-', linewidth=2,
               label=f'Median = {np.median(rc_over_xi):.2f}')
    ax.set_xlabel('r_c / ξ', fontsize=12)
    ax.set_ylabel('N galaxies', fontsize=12)
    ax.set_title('(c) Core-to-healing ratio distribution', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, 'healing_length_validation.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig2_path}")

    # ---- Figure 3: Example Rotation Curves ----
    # Pick representative galaxies — one per winner type + extras
    comp_winners = [r for r in results if r['winner'] in ('Composite', 'Constrained')]
    nfw_winners = [r for r in results if r['winner'] == 'NFW']
    burk_winners = [r for r in results if r['winner'] == 'Burkert']
    sol_winners = [r for r in results if r['winner'] == 'Soliton']

    examples = []
    # Sort by mass for diversity
    comp_winners.sort(key=lambda r: r['logMs'])
    nfw_winners.sort(key=lambda r: r['logMs'])
    burk_winners.sort(key=lambda r: r['logMs'])
    sol_winners.sort(key=lambda r: r['logMs'])

    if len(comp_winners) >= 2:
        examples.append(comp_winners[0])
        examples.append(comp_winners[-1])
    elif len(comp_winners) > 0:
        examples.append(comp_winners[0])

    if len(nfw_winners) >= 1:
        examples.append(nfw_winners[len(nfw_winners)//2])

    if len(burk_winners) >= 1:
        examples.append(burk_winners[len(burk_winners)//2])

    if len(sol_winners) >= 1:
        examples.append(sol_winners[len(sol_winners)//2])

    # Pad to 6 if needed
    while len(examples) < 6 and len(results) > len(examples):
        for r in results:
            if r not in examples:
                examples.append(r)
                break

    n_panels = min(len(examples), 6)
    if n_panels >= 4:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
        if n_panels == 1:
            axes = [axes]

    for idx in range(n_panels):
        ax = axes[idx]
        r_data = examples[idx]
        R_arr = np.array(r_data['_R'])
        Vobs_arr = np.array(r_data['_Vobs'])
        eVobs_arr = np.array(r_data['_eVobs'])
        Vbar_sq_arr = np.array(r_data['_Vbar_sq'])

        # Plot data
        ax.errorbar(R_arr, Vobs_arr, yerr=eVobs_arr, fmt='ko', markersize=3,
                     capsize=2, label='Observed', zorder=5)

        # Baryonic
        Vbar = np.sqrt(np.maximum(Vbar_sq_arr, 0.0))
        ax.plot(R_arr, Vbar, '--', color='gray', linewidth=1, label='Baryons')

        # Models
        r_fine = np.linspace(R_arr.min(), R_arr.max(), 200)
        Vbar_sq_fine = np.interp(r_fine, R_arr, Vbar_sq_arr)

        V_nfw = nfw_v_circ(r_fine, 10.0**r_data['nfw_log_rhos'], 10.0**r_data['nfw_log_rs'])
        V_nfw_t = np.sqrt(np.maximum(Vbar_sq_fine + V_nfw**2, 0.0))
        ax.plot(r_fine, V_nfw_t, '-', color='blue', linewidth=1.5, label='NFW')

        V_burk = burkert_v_circ(r_fine, 10.0**r_data['burk_log_rho0'], 10.0**r_data['burk_log_r0'])
        V_burk_t = np.sqrt(np.maximum(Vbar_sq_fine + V_burk**2, 0.0))
        ax.plot(r_fine, V_burk_t, '-', color='purple', linewidth=1.5, label='Burkert')

        V_sol = soliton_v_circ(r_fine, 10.0**r_data['sol_log_rhoc'], 10.0**r_data['sol_log_rc'])
        V_sol_t = np.sqrt(np.maximum(Vbar_sq_fine + V_sol**2, 0.0))
        ax.plot(r_fine, V_sol_t, '-', color='green', linewidth=1.5, label='Soliton')

        V_comp = composite_v_circ(r_fine, 10.0**r_data['comp_log_rhoc'], 10.0**r_data['comp_log_rc'],
                                   10.0**r_data['comp_log_rhos'], 10.0**r_data['comp_log_rs'])
        V_comp_t = np.sqrt(np.maximum(Vbar_sq_fine + V_comp**2, 0.0))
        ax.plot(r_fine, V_comp_t, '-', color='red', linewidth=2, label='Composite')

        # Mark ξ
        xi_val = r_data['xi_kpc']
        if xi_val < R_arr.max():
            ax.axvline(xi_val, color='purple', linestyle=':', linewidth=1, alpha=0.6)
            ax.text(xi_val, ax.get_ylim()[1] * 0.95, 'ξ', color='purple', fontsize=9, ha='center')

        # Mark r_t
        r_t_val = r_data['r_transition_kpc']
        if r_t_val and r_t_val < R_arr.max():
            ax.axvline(r_t_val, color='orange', linestyle=':', linewidth=1, alpha=0.6)
            ax.text(r_t_val, ax.get_ylim()[1] * 0.88, 'r_t', color='orange', fontsize=9, ha='center')

        ax.set_xlabel('R [kpc]', fontsize=10)
        ax.set_ylabel('V [km/s]', fontsize=10)
        ax.set_title(f'{r_data["name"]} (logMs={r_data["logMs"]:.1f})\n'
                     f'Winner: {r_data["winner"]}, ΔAIC(comp)={r_data["daic_comp_nfw"]:+.1f}',
                     fontsize=10)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    # Hide unused panels
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, 'composite_rotation_curves.png')
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig3_path}")

    # ---- Figure 4: C2/C3 Rehabilitation ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel a: Inner region residual comparison
    ax = axes[0]
    labels = ['NFW', 'Burkert', 'Composite', 'Soliton']
    means = [np.mean(inner_res_nfw), np.mean(inner_res_burk), np.mean(inner_res_comp), np.mean(inner_res_sol)]
    stds = [np.std(inner_res_nfw), np.std(inner_res_burk), np.std(inner_res_comp), np.std(inner_res_sol)]
    colors = ['blue', 'purple', 'red', 'green']
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.6, edgecolor='black',
                   capsize=5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(-0.181, color='gray', linestyle='--', linewidth=1.5,
               label='C2 observed (-0.181 dex)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Mean inner residual (dex)', fontsize=12)
    ax.set_title('(a) Inner region (R < ξ) residuals', fontsize=13)
    ax.legend(fontsize=10)

    # Panel b: Radial variance profile
    ax = axes[1]
    if c3_bins:
        bin_centers = [(b['logX_lo'] + b['logX_hi']) / 2 for b in c3_bins]
        s_nfw_arr = [b['sigma_nfw'] for b in c3_bins]
        s_burk_arr = [b['sigma_burk'] for b in c3_bins]
        s_comp_arr = [b['sigma_comp'] for b in c3_bins]
        s_sol_arr = [b['sigma_sol'] for b in c3_bins]
        ax.plot(bin_centers, s_nfw_arr, 'o-', color='blue', label='NFW', linewidth=2, markersize=8)
        ax.plot(bin_centers, s_burk_arr, 'D-', color='purple', label='Burkert', linewidth=2, markersize=7)
        ax.plot(bin_centers, s_comp_arr, 's-', color='red', label='Composite', linewidth=2, markersize=8)
        ax.plot(bin_centers, s_sol_arr, '^-', color='green', label='Soliton', linewidth=2, markersize=8)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, label='X=1 (R=ξ)')
        ax.set_xlabel('log(X) = log(R/ξ)', fontsize=12)
        ax.set_ylabel('RMS residual (km/s)', fontsize=12)
        ax.set_title('(b) Radial variance by X = R/ξ', fontsize=13)
        ax.legend(fontsize=10)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, 'c2_c3_rehabilitation.png')
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig4_path}")

except ImportError:
    print("  matplotlib not available — skipping figure generation")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
