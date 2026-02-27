#!/usr/bin/env python3
"""
test_lensing_profile_shape.py — Weak lensing ESD profile shape test
=====================================================================

BEC predicts distinct density profiles depending on environment:
  - Isolated galaxies: solitonic core → ρ(r) ∝ [1 + (r/ξ)²]⁻⁸
    This produces an ESD profile with a characteristic flattening
    (or turnover) at r ≈ ξ (the healing length).
  - Cluster satellites: tidal disruption strips the outer condensate,
    producing cuspy/truncated profiles closer to NFW.

Brouwer+2021 provides:
  - Lensing rotation curves (Fig 3): ESD(R) in 4 stellar mass bins
  - Both isolated and all (clustered) galaxy samples
  - We can fit these ESD profiles to NFW, cored (soliton-like),
    and truncated models.

Tests:
  1. Fit each mass bin ESD to NFW and solitonic models — compare χ²
  2. Compare isolated vs all: do all-galaxy profiles look more cuspy?
  3. Radial residual structure: systematic deviations from NFW?
  4. Characteristic scale: does the flattening scale correlate with g†?

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'brouwer2021')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from analysis_tools import (
    g_dagger, LOG_G_DAGGER, G_SI, M_SUN, MPC_M,
)

# Physical constants for lensing
G_pc3 = 4.52e-30    # G in pc³/(Msun·s²) [Brouwer convention]
PC_PER_M = 1.0 / 3.086e16
MPC_PER_PC = 1e-6


# ================================================================
# DATA LOADERS
# ================================================================

def load_rotation_curve_esd(massbin):
    """Load Brouwer+2021 lensing rotation curve ESD profile.

    Returns dict with R_Mpc, ESD, ESD_err arrays.
    """
    fname = f'Fig-3_Lensing-rotation-curves_Massbin-{massbin}.txt'
    filepath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(filepath):
        print(f"  WARNING: {fname} not found")
        return None

    data = {'R_Mpc': [], 'ESD': [], 'ESD_err': [], 'bias': []}
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                R = float(parts[0])      # Mpc
                esd_t = float(parts[1])   # h70*Msun/pc²
                err = float(parts[3])     # h70*Msun/pc²
                bias = float(parts[4])    # multiplicative
                data['R_Mpc'].append(R)
                data['ESD'].append(esd_t / bias)
                data['ESD_err'].append(err / bias)
                data['bias'].append(bias)
            except (ValueError, IndexError):
                continue

    for key in data:
        data[key] = np.array(data[key])

    return data


def esd_to_vcirc(esd, R_Mpc):
    """Convert ESD (h70*Msun/pc²) at R (Mpc) to circular velocity (km/s).

    v_circ = sqrt(4 * G * ESD * R * Mpc/pc) * pc/km
    """
    Mpc_per_pc = 1e6
    pc_per_km = 3.086e13
    v2 = 4 * G_pc3 * esd * R_Mpc * Mpc_per_pc
    v2 = np.maximum(v2, 0)
    return np.sqrt(v2) * pc_per_km


def esd_to_sigma(esd):
    """Convert ESD (h70*Msun/pc²) to surface mass density (Msun/pc²).
    ESD_t IS the surface mass density difference.
    """
    return esd  # Already in Msun/pc²


# ================================================================
# MODEL ESD PROFILES
# ================================================================

def nfw_esd(R_Mpc, M200, c, z_l=0.2):
    """NFW ESD profile.

    The Excess Surface Density is:
    ESD(R) = Σ̄(<R) - Σ(R)
    where Σ is the projected surface density and Σ̄(<R) is the mean
    within R.

    For NFW: uses analytic expressions from Bartelmann (1996),
    Wright & Brainerd (2000).
    """
    # Critical density at z_l
    H_z = 67.74 * np.sqrt(0.3089 * (1 + z_l)**3 + 0.6911)  # km/s/Mpc
    H_z_si = H_z * 1e3 / (3.086e22)  # 1/s
    rho_crit = 3 * H_z_si**2 / (8 * np.pi * G_SI)  # kg/m³
    rho_crit_Msun_Mpc3 = rho_crit / M_SUN * MPC_M**3

    R200 = (3 * M200 / (4 * np.pi * 200 * rho_crit_Msun_Mpc3))**(1.0/3.0)  # Mpc
    rs = R200 / c  # Mpc

    delta_c = (200.0 / 3.0) * c**3 / (np.log(1 + c) - c / (1 + c))
    rho_s = delta_c * rho_crit_Msun_Mpc3  # Msun/Mpc³

    Sigma_s = rho_s * rs  # Msun/Mpc²

    x = np.asarray(R_Mpc) / rs
    x = np.clip(x, 0.01, 100)

    # Σ(x) for NFW (Wright & Brainerd 2000)
    Sigma = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if abs(xi - 1.0) < 1e-5:
            Sigma[i] = Sigma_s * (1.0/3.0)
        elif xi < 1.0:
            val = np.sqrt(1.0 - xi**2)
            Sigma[i] = Sigma_s * 2.0 / (xi**2 - 1) * (1.0 / val * np.arctanh(val) - 1.0)
        else:
            val = np.sqrt(xi**2 - 1.0)
            Sigma[i] = Sigma_s * 2.0 / (xi**2 - 1) * (1.0 / val * np.arctan(val) - 1.0)

    # Σ̄(<x) — mean projected density within x
    Sigma_bar = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if abs(xi - 1.0) < 1e-5:
            Sigma_bar[i] = Sigma_s * (1.0 + np.log(0.5))
        elif xi < 1.0:
            val = np.sqrt(1.0 - xi**2)
            g_x = np.log(xi/2) + 1.0/val * np.arctanh(val)
            Sigma_bar[i] = Sigma_s * 4.0 / xi**2 * g_x
        else:
            val = np.sqrt(xi**2 - 1.0)
            g_x = np.log(xi/2) + 1.0/val * np.arctan(val)
            Sigma_bar[i] = Sigma_s * 4.0 / xi**2 * g_x

    # ESD = Σ̄ - Σ  (in Msun/Mpc²)
    esd = Sigma_bar - Sigma

    # Convert to h70*Msun/pc²
    Mpc_per_pc = 1e6
    esd_pc2 = esd / Mpc_per_pc**2  # Msun/pc²

    return esd_pc2


def cored_esd(R_Mpc, M_core, r_core, rho_0, z_l=0.2):
    """Cored (soliton-like) ESD profile.

    ρ(r) = ρ₀ / [1 + (r/r_core)²]^n with n=8 (BEC soliton)
    Projected → Σ(R) → ESD(R) = Σ̄(<R) - Σ(R)

    For computational simplicity, we numerically integrate.
    """
    R = np.asarray(R_Mpc)
    n_r = 200
    r_max = 10.0 * np.max(R)

    # Numerical projection: Σ(R) = 2 ∫₀^∞ ρ(sqrt(R² + z²)) dz
    # For cored profile: ρ(r) = ρ₀ / (1 + (r/r_core)²)^8
    z_grid = np.logspace(-4, np.log10(r_max), n_r)

    Sigma = np.zeros_like(R)
    for i, Ri in enumerate(R):
        r3d = np.sqrt(Ri**2 + z_grid**2)
        rho = rho_0 / (1 + (r3d / r_core)**2)**8
        Sigma[i] = 2.0 * np.trapz(rho, z_grid)  # Msun/Mpc³ × Mpc = Msun/Mpc²

    # Mean projected density within R
    Sigma_bar = np.zeros_like(R)
    R_fine = np.logspace(np.log10(1e-4), np.log10(np.max(R)), 100)

    # Compute Σ at fine radii
    Sigma_fine = np.zeros(len(R_fine))
    for i, Ri in enumerate(R_fine):
        r3d = np.sqrt(Ri**2 + z_grid**2)
        rho = rho_0 / (1 + (r3d / r_core)**2)**8
        Sigma_fine[i] = 2.0 * np.trapz(rho, z_grid)

    for i, Ri in enumerate(R):
        mask = R_fine <= Ri
        if np.sum(mask) < 2:
            Sigma_bar[i] = Sigma[i]
            continue
        r_sub = R_fine[mask]
        s_sub = Sigma_fine[mask]
        # Σ̄(<R) = (2/R²) ∫₀^R Σ(R') R' dR'
        integrand = s_sub * r_sub
        Sigma_bar[i] = 2.0 / Ri**2 * np.trapz(integrand, r_sub)

    esd = Sigma_bar - Sigma

    # Convert Msun/Mpc² to Msun/pc²
    Mpc_per_pc = 1e6
    return esd / Mpc_per_pc**2


def pseudo_isothermal_esd(R_Mpc, rho_0, r_c, z_l=0.2):
    """Pseudo-isothermal sphere (ISO) ESD.

    ρ(r) = ρ₀ / (1 + (r/r_c)²)
    Σ(R) = π ρ₀ r_c / sqrt(1 + (R/r_c)²) × [1 - R/r_c × arctan(r_max/R) like expression]

    For ISO: Σ(R) = π ρ₀ r_c (1 / sqrt(1 + (R/r_c)²) - 1/sqrt(1 + (r_t/r_c)²))
    truncated at r_t >> R.

    Analytic: Σ(R) ≈ π ρ₀ r_c / sqrt(1 + (R/r_c)²) for r_t → ∞
    Σ̄(<R) = 2π ρ₀ r_c × [sqrt(1 + (R/r_c)²) - 1] / (R/r_c)²
    """
    R = np.asarray(R_Mpc)
    u = R / r_c

    Sigma = np.pi * rho_0 * r_c / np.sqrt(1 + u**2)
    # Σ̄(<R) for ISO: integral gives
    # Σ̄ = (2πρ₀r_c/u²) × [√(1+u²) - 1]
    Sigma_bar = np.where(u > 1e-6,
                         2 * np.pi * rho_0 * r_c * (np.sqrt(1 + u**2) - 1) / u**2,
                         np.pi * rho_0 * r_c)

    esd = Sigma_bar - Sigma
    Mpc_per_pc = 1e6
    return esd / Mpc_per_pc**2


# ================================================================
# FITTING FUNCTIONS
# ================================================================

def fit_nfw(R_Mpc, esd_obs, esd_err):
    """Fit NFW to ESD data. Returns best-fit params and chi2."""
    def chi2_nfw(params):
        log_M200, log_c = params
        M200 = 10**log_M200
        c = 10**log_c
        if c < 1 or c > 50 or M200 < 1e9 or M200 > 1e16:
            return 1e10
        model = nfw_esd(R_Mpc, M200, c)
        if np.any(np.isnan(model)):
            return 1e10
        return np.sum(((esd_obs - model) / esd_err)**2)

    best = None
    best_chi2 = 1e10
    for log_M in [11.0, 11.5, 12.0, 12.5, 13.0]:
        for log_c in [0.3, 0.6, 0.9, 1.2]:
            try:
                res = minimize(chi2_nfw, [log_M, log_c], method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-2})
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best = res.x
            except:
                pass

    if best is None:
        return {'M200': np.nan, 'c': np.nan, 'chi2': np.nan, 'chi2_dof': np.nan}

    M200 = 10**best[0]
    c = 10**best[1]
    dof = max(len(esd_obs) - 2, 1)
    return {
        'M200': float(M200),
        'log_M200': round(float(best[0]), 3),
        'c': round(float(c), 2),
        'chi2': round(float(best_chi2), 2),
        'chi2_dof': round(float(best_chi2 / dof), 3),
        'dof': dof,
        'model': nfw_esd(R_Mpc, M200, c),
    }


def fit_iso(R_Mpc, esd_obs, esd_err):
    """Fit pseudo-isothermal (cored) profile to ESD data."""
    def chi2_iso(params):
        log_rho0, log_rc = params
        rho_0 = 10**log_rho0  # Msun/Mpc³
        r_c = 10**log_rc       # Mpc
        if rho_0 < 1e2 or rho_0 > 1e12 or r_c < 1e-3 or r_c > 10:
            return 1e10
        model = pseudo_isothermal_esd(R_Mpc, rho_0, r_c)
        if np.any(np.isnan(model)):
            return 1e10
        return np.sum(((esd_obs - model) / esd_err)**2)

    best = None
    best_chi2 = 1e10
    for log_rho in [5, 6, 7, 8]:
        for log_rc in [-2, -1.5, -1, -0.5, 0]:
            try:
                res = minimize(chi2_iso, [log_rho, log_rc], method='Nelder-Mead',
                               options={'maxiter': 5000})
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best = res.x
            except:
                pass

    if best is None:
        return {'rho_0': np.nan, 'r_core': np.nan, 'chi2': np.nan, 'chi2_dof': np.nan}

    rho_0 = 10**best[0]
    r_c = 10**best[1]
    dof = max(len(esd_obs) - 2, 1)
    return {
        'rho_0': float(rho_0),
        'log_rho_0': round(float(best[0]), 3),
        'r_core_Mpc': round(float(r_c), 4),
        'r_core_kpc': round(float(r_c * 1000), 1),
        'chi2': round(float(best_chi2), 2),
        'chi2_dof': round(float(best_chi2 / dof), 3),
        'dof': dof,
        'model': pseudo_isothermal_esd(R_Mpc, rho_0, r_c),
    }


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("WEAK LENSING ESD PROFILE SHAPE TEST")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")
print(f"  Data: Brouwer+2021 KiDS-1000 lensing rotation curves")

mass_bin_labels = {
    1: 'Dwarf (8.5-10.3)',
    2: 'Intermediate-low (10.3-10.6)',
    3: 'Intermediate-high (10.6-10.8)',
    4: 'Massive (10.8-11.0)',
}

results = {
    'test_name': 'lensing_profile_shape',
    'description': ('Fit NFW and cored profiles to Brouwer+2021 lensing ESD data. '
                    'BEC predicts solitonic cores for isolated galaxies.'),
    'mass_bins': {},
}


# ================================================================
# 1. LOAD AND DISPLAY DATA
# ================================================================
print("\n" + "=" * 72)
print("[1] Loading Brouwer+2021 lensing rotation curve data")
print("=" * 72)

all_data = {}
for mb in range(1, 5):
    data = load_rotation_curve_esd(mb)
    if data is not None:
        all_data[mb] = data
        vcirc = esd_to_vcirc(data['ESD'], data['R_Mpc'])
        print(f"\n  Mass bin {mb} — {mass_bin_labels[mb]}:")
        print(f"    N points: {len(data['R_Mpc'])}")
        print(f"    R range: [{data['R_Mpc'].min():.3f}, {data['R_Mpc'].max():.3f}] Mpc")
        print(f"    ESD range: [{data['ESD'].min():.2f}, {data['ESD'].max():.2f}] h70·Msun/pc²")
        print(f"    v_circ range: [{vcirc.min():.0f}, {vcirc.max():.0f}] km/s")


# ================================================================
# 2. FIT NFW AND CORED MODELS
# ================================================================
print("\n" + "=" * 72)
print("[2] Model fitting: NFW vs ISO (cored)")
print("=" * 72)

print(f"\n  {'Mass bin':>30s} {'χ²/dof NFW':>12s} {'χ²/dof ISO':>12s} "
      f"{'ΔAIC':>8s} {'r_core':>10s} {'Preferred':>10s}")
print(f"  {'-'*86}")

for mb in sorted(all_data.keys()):
    data = all_data[mb]
    R = data['R_Mpc']
    esd = data['ESD']
    err = data['ESD_err']

    # Only use positive ESD points (SNR > 0)
    mask = esd > 0
    if np.sum(mask) < 5:
        print(f"  {mass_bin_labels[mb]:>30s}  insufficient positive ESD points")
        continue

    R_fit = R[mask]
    esd_fit = esd[mask]
    err_fit = err[mask]

    # Fit NFW
    nfw_result = fit_nfw(R_fit, esd_fit, err_fit)

    # Fit ISO (cored)
    iso_result = fit_iso(R_fit, esd_fit, err_fit)

    # AIC comparison (2 params each for simple comparison)
    n_data = len(esd_fit)
    k_nfw = 2
    k_iso = 2
    aic_nfw = nfw_result['chi2'] + 2 * k_nfw
    aic_iso = iso_result['chi2'] + 2 * k_iso
    delta_aic = aic_iso - aic_nfw  # negative means ISO preferred

    preferred = "ISO (cored)" if delta_aic < -2 else ("NFW" if delta_aic > 2 else "~equal")

    rc_str = (f"{iso_result['r_core_kpc']:.0f} kpc"
              if not np.isnan(iso_result.get('r_core_kpc', np.nan)) else "---")

    print(f"  {mass_bin_labels[mb]:>30s} {nfw_result['chi2_dof']:12.2f} "
          f"{iso_result['chi2_dof']:12.2f} {delta_aic:+8.1f} {rc_str:>10s} {preferred:>10s}")

    # Compute residuals from NFW
    if 'model' in nfw_result and not np.any(np.isnan(nfw_result['model'])):
        nfw_resid = esd_fit - nfw_result['model']
        nfw_frac_resid = nfw_resid / esd_fit

        # Check for systematic pattern: positive residuals at small R (core)
        # and negative at large R (missing outer mass)
        n_inner = max(len(R_fit) // 3, 1)
        inner_excess = np.mean(nfw_frac_resid[:n_inner])
        outer_excess = np.mean(nfw_frac_resid[-n_inner:])
    else:
        inner_excess = np.nan
        outer_excess = np.nan

    results['mass_bins'][mb] = {
        'label': mass_bin_labels[mb],
        'n_points': len(R_fit),
        'nfw': {
            'log_M200': nfw_result.get('log_M200'),
            'c': nfw_result.get('c'),
            'chi2_dof': nfw_result.get('chi2_dof'),
        },
        'iso': {
            'r_core_kpc': iso_result.get('r_core_kpc'),
            'chi2_dof': iso_result.get('chi2_dof'),
        },
        'delta_aic': round(float(delta_aic), 1),
        'preferred': preferred,
        'inner_excess_frac': round(float(inner_excess), 3) if not np.isnan(inner_excess) else None,
        'outer_deficit_frac': round(float(outer_excess), 3) if not np.isnan(outer_excess) else None,
    }


# ================================================================
# 3. RADIAL RESIDUAL STRUCTURE
# ================================================================
print("\n" + "=" * 72)
print("[3] NFW residual structure — looking for inner core signature")
print("=" * 72)

for mb in sorted(all_data.keys()):
    if mb not in results['mass_bins']:
        continue
    res = results['mass_bins'][mb]

    data = all_data[mb]
    R = data['R_Mpc']
    esd = data['ESD']
    err = data['ESD_err']
    mask = esd > 0

    if np.sum(mask) < 5:
        continue

    R_fit = R[mask]
    esd_fit = esd[mask]
    err_fit = err[mask]

    nfw_res = fit_nfw(R_fit, esd_fit, err_fit)
    if 'model' not in nfw_res:
        continue

    nfw_model = nfw_res['model']
    resid = esd_fit - nfw_model
    frac_resid = resid / esd_fit

    print(f"\n  {mass_bin_labels[mb]}:")
    print(f"    {'R (Mpc)':>10s} {'ESD_obs':>10s} {'ESD_NFW':>10s} {'Resid':>10s} {'Frac':>8s}")
    print(f"    {'-'*52}")
    for j in range(len(R_fit)):
        marker = ""
        if frac_resid[j] > 0.3:
            marker = " ← excess (core?)"
        elif frac_resid[j] < -0.3:
            marker = " ← deficit"
        print(f"    {R_fit[j]:10.4f} {esd_fit[j]:10.2f} {nfw_model[j]:10.2f} "
              f"{resid[j]:+10.2f} {frac_resid[j]:+8.2%}{marker}")


# ================================================================
# 4. CORE RADIUS vs g† SCALE
# ================================================================
print("\n" + "=" * 72)
print("[4] Core radius vs g† characteristic scale")
print("=" * 72)

# The BEC healing length ξ sets the core size:
# ξ = ℏ/(m_a × c_s) where c_s = sound speed in condensate
# For galaxy-scale: ξ ≈ 0.1-10 kpc (depends on m_a and virialization)
# The lensing probes 35-2600 kpc → we see the OUTER profile shape
#
# At g_obs = g†: R_g† ≈ sqrt(GM/g†)
# For M* ≈ 10^10.5: R_g† ≈ 50 kpc ≈ 0.05 Mpc
# For M* ≈ 10^11: R_g† ≈ 150 kpc ≈ 0.15 Mpc
# This should be within the Brouwer data range.

print(f"\n  Characteristic scale R where g_obs = g† for each mass bin:")
mass_bin_Mstar = {1: 10**9.4, 2: 10**10.45, 3: 10**10.7, 4: 10**10.9}

for mb in sorted(all_data.keys()):
    Mstar = mass_bin_Mstar[mb]
    # R_g† where g(R) ≈ g† → GM*/R² = g† → R = sqrt(GM*/g†)
    R_gdagger = np.sqrt(G_SI * Mstar * M_SUN / g_dagger) / MPC_M  # in Mpc
    R_kpc = R_gdagger * 1000

    data = all_data[mb]
    in_range = (data['R_Mpc'].min() < R_gdagger < data['R_Mpc'].max())

    print(f"  {mass_bin_labels[mb]:>30s}: R(g†) = {R_kpc:.0f} kpc = {R_gdagger:.3f} Mpc  "
          f"{'(in data range)' if in_range else '(outside range)'}")

    if mb in results['mass_bins']:
        results['mass_bins'][mb]['R_gdagger_Mpc'] = round(float(R_gdagger), 4)
        results['mass_bins'][mb]['R_gdagger_kpc'] = round(float(R_kpc), 0)
        results['mass_bins'][mb]['R_gdagger_in_data_range'] = in_range


# ================================================================
# 5. LENSING ROTATION CURVE ANALYSIS
# ================================================================
print("\n" + "=" * 72)
print("[5] Lensing rotation curves — flatness test")
print("=" * 72)

# BEC prediction: rotation curves should flatten (or show weak rise)
# at R >> ξ, similar to real rotation curves.
# NFW gives continuously rising v(R) at large R for massive halos.

print(f"\n  {'Mass bin':>30s} {'V(inner)':>10s} {'V(outer)':>10s} {'V_out/V_in':>12s} {'Flat?':>8s}")
print(f"  {'-'*72}")

for mb in sorted(all_data.keys()):
    data = all_data[mb]
    vcirc = esd_to_vcirc(data['ESD'], data['R_Mpc'])
    mask = data['ESD'] > 0
    if np.sum(mask) < 4:
        continue

    R_good = data['R_Mpc'][mask]
    V_good = vcirc[mask]

    n_third = max(len(V_good) // 3, 1)
    V_inner = np.median(V_good[:n_third])
    V_outer = np.median(V_good[-n_third:])
    ratio = V_outer / V_inner if V_inner > 0 else np.nan

    flat = "YES" if 0.7 < ratio < 1.3 else ("RISING" if ratio > 1.3 else "FALLING")

    print(f"  {mass_bin_labels[mb]:>30s} {V_inner:10.0f} {V_outer:10.0f} {ratio:12.2f} {flat:>8s}")

    if mb in results['mass_bins']:
        results['mass_bins'][mb]['V_inner'] = round(float(V_inner), 1)
        results['mass_bins'][mb]['V_outer'] = round(float(V_outer), 1)
        results['mass_bins'][mb]['V_ratio'] = round(float(ratio), 3)
        results['mass_bins'][mb]['flat'] = flat


# ================================================================
# 6. VERDICT
# ================================================================
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

# Count preferences
n_iso_preferred = sum(1 for mb in results['mass_bins'].values()
                      if mb.get('preferred') == 'ISO (cored)')
n_nfw_preferred = sum(1 for mb in results['mass_bins'].values()
                      if mb.get('preferred') == 'NFW')
n_equal = sum(1 for mb in results['mass_bins'].values()
              if mb.get('preferred') == '~equal')

print(f"\n  Model preference: ISO preferred in {n_iso_preferred}/{len(results['mass_bins'])} bins, "
      f"NFW in {n_nfw_preferred}, equal in {n_equal}")

# Key diagnostic: do low-mass (dwarf) profiles prefer cores more than massive?
if 1 in results['mass_bins'] and 4 in results['mass_bins']:
    daic_low = results['mass_bins'][1]['delta_aic']
    daic_high = results['mass_bins'][4]['delta_aic']
    print(f"  ΔAIC (dwarf): {daic_low:+.1f}, ΔAIC (massive): {daic_high:+.1f}")
    if daic_low < daic_high:
        print(f"  → Dwarfs prefer cores MORE than massive — BEC-consistent")
    else:
        print(f"  → No clear mass-dependent core preference")

caveats = [
    "Stacked lensing averages over many galaxies — core signatures may wash out",
    "Brouwer data has 15 radial bins per mass bin — limited radial resolution",
    "Inner radius (35 kpc) may not resolve the BEC soliton core (ξ ~ 1-10 kpc)",
    "Both NFW and ISO are parametric — actual BEC profile (soliton+condensate) is more complex",
    "The lensing signal probes the TOTAL mass, not just DM",
]

# Overall verdict
if n_iso_preferred > n_nfw_preferred:
    verdict = ("BEC-SUGGESTIVE: Cored profiles preferred over NFW in the majority of mass bins. "
               "However, limited radial resolution prevents strong conclusions. "
               "The inner measurement radius (35 kpc) is larger than typical BEC soliton cores.")
elif n_iso_preferred == 0:
    verdict = ("NFW ADEQUATE: NFW fits as well or better than cored profiles at all masses. "
               "No evidence for solitonic cores at the stacked lensing resolution. "
               "Does not rule out BEC — cores may be too small to resolve.")
else:
    verdict = ("INCONCLUSIVE: Mixed preference between NFW and cored profiles. "
               "Limited radial resolution and stacking effects prevent firm conclusions.")

print(f"\n  VERDICT: {verdict}")
print(f"\n  Caveats:")
for c in caveats:
    print(f"    - {c}")

results['verdict'] = verdict
results['caveats'] = caveats

# Save
outpath = os.path.join(RESULTS_DIR, 'summary_lensing_profile_shape.json')
with open(outpath, 'w') as f:
    def json_default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return round(float(x), 5)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        raise TypeError(f"Object of type {type(x)} is not JSON serializable")
    json.dump(results, f, indent=2, default=json_default)
print(f"\n  Results saved: {outpath}")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
