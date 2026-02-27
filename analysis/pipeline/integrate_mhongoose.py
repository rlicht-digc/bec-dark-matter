#!/usr/bin/env python3
"""
integrate_mhongoose.py — Integrate MHONGOOSE rotation curves into RAR pipeline
================================================================================

Loads published mass-decomposed rotation curves from MHONGOOSE papers and
computes RAR (g_bar, g_obs) points for integration into the extended dataset.

Currently includes:
  - ESO444-G084 and [KKS2000]23 from Nkomo+2025 (A&A 699, A372)
    WISE 3.4μm stellar photometry, MeerKAT HI, ISO halo mass models

Strategy for RAR extraction:
  These are gas-dominated dwarfs (M_HI >> M_star by 10-20×).
  The paper provides:
    - Vc(R): asymmetric-drift-corrected circular velocity (= sqrt(g_obs × R))
    - ISO halo fit: ρ₀, r_c → V_halo(R) = sqrt(4πGρ₀r_c²(1 - (r_c/R)arctan(R/r_c)))
    - M_star, M_HI, (M/L)_3.4μm

  We compute:
    g_obs = Vc² / R
    V_halo² = 4πGρ₀r_c²[1 - (r_c/R)arctan(R/r_c)]
    V_bar² = Vc² - V_halo²  (residual baryonic)
    g_bar = V_bar² / R

  Cross-check: V_bar should be consistent with M_star + M_gas at each radius.

Russell Licht — Primordial Fluid DM Project, Feb 2026
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'mhongoose')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from analysis_tools import (
    g_dagger, LOG_G_DAGGER, G_SI, M_SUN, KPC_M,
    rar_function, rar_residuals,
)


# ================================================================
# Galaxy properties from Nkomo+2025
# ================================================================

GALAXIES = {
    'ESO444-G084': {
        'file': 'nkomo2025_eso444g084.tsv',
        'distance_Mpc': 4.6,
        'inclination_deg': 49.0,
        'M_HI': 1.1e8,        # Msun
        'M_star': 4.9e6,      # Msun
        'ML_band': '3.4um',
        'ML_value': 0.20,
        # ISO halo parameters (M/L = 0.20)
        'rho_0': 16.05e-3,    # Msun/pc³
        'r_c_kpc': 3.48,      # kpc
        'chi2_red': 2.6,
        'source': 'Nkomo+2025',
        'survey': 'MHONGOOSE',
    },
    'KKS2000-23': {
        'file': 'nkomo2025_kks2000-23.tsv',
        'distance_Mpc': 13.9,
        'inclination_deg': 62.0,
        'M_HI': 6.1e8,        # Msun
        'M_star': 3.2e7,      # Msun
        'ML_band': '3.4um',
        'ML_value': 0.18,
        # ISO halo parameters (M/L = 0.18)
        'rho_0': 4.29e-3,     # Msun/pc³
        'r_c_kpc': 5.49,      # kpc
        'chi2_red': None,      # not reported separately
        'source': 'Nkomo+2025',
        'survey': 'MHONGOOSE',
    },
}

# G in units compatible with Msun, pc, km/s
G_PC = 4.302e-3  # pc (km/s)² / Msun


def iso_vhalo(R_kpc, rho_0_Msun_pc3, r_c_kpc):
    """Pseudo-isothermal halo circular velocity.

    V²_halo(R) = 4πGρ₀r_c² [1 - (r_c/R) arctan(R/r_c)]

    Args:
        R_kpc: radius array in kpc
        rho_0_Msun_pc3: central density in Msun/pc³
        r_c_kpc: core radius in kpc

    Returns:
        V_halo in km/s
    """
    R = np.asarray(R_kpc, dtype=float)
    r_c = r_c_kpc
    rho_0 = rho_0_Msun_pc3

    # Convert r_c to pc for consistent units
    r_c_pc = r_c * 1000  # pc
    R_pc = R * 1000       # pc

    # V² = 4πGρ₀r_c² [1 - (r_c/R)arctan(R/r_c)]
    ratio = R_pc / r_c_pc
    V2 = 4 * np.pi * G_PC * rho_0 * r_c_pc**2 * (1 - np.where(
        ratio > 1e-6,
        (1.0 / ratio) * np.arctan(ratio),
        1.0 - ratio**2 / 3  # Taylor expansion for small R
    ))

    return np.sqrt(np.maximum(V2, 0))


def load_rotation_curve(filepath):
    """Load a MHONGOOSE rotation curve TSV file.

    Returns dict with R_kpc, Vrot, Vc, eVc arrays.
    """
    R, Vrot, Vc, eVc = [], [], [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            try:
                r = float(parts[0])
                vr = float(parts[1])
                vc = float(parts[2])
                ev = float(parts[3])
                R.append(r)
                Vrot.append(vr)
                Vc.append(vc)
                eVc.append(ev)
            except ValueError:
                continue

    return {
        'R_kpc': np.array(R),
        'Vrot': np.array(Vrot),
        'Vc': np.array(Vc),
        'eVc': np.array(eVc),
    }


def exponential_disk_vcirc(R_kpc, M_total, R_d_kpc):
    """Circular velocity for an exponential disk (Freeman 1970).

    V²(R) = (G M / R_d) × y² × [I₀K₀ - I₁K₁]
    where y = R/(2R_d)

    For simplicity, use the cumulative mass approximation:
    M(<R) = M_total × [1 - (1 + R/R_d) exp(-R/R_d)]
    V² = G M(<R) / R
    """
    R = np.asarray(R_kpc, dtype=float)
    y = R / R_d_kpc
    M_enc = M_total * (1.0 - (1.0 + y) * np.exp(-y))

    # V² = G M_enc / R  in (km/s)² with R in pc, M in Msun
    R_pc = R * 1000
    V2 = G_PC * M_enc / np.maximum(R_pc, 1.0)
    return np.sqrt(np.maximum(V2, 0))


def compute_rar_direct(rc_data, galaxy_props):
    """Compute RAR points directly from baryonic mass model + observed Vc.

    g_obs = Vc²/R  (from observed rotation curve)
    g_bar = V_bar²/R  where V_bar² = V_star² + V_gas²

    For these gas-dominated dwarfs:
      V_gas dominates (M_HI/M_star ~ 20)
      Approximate both as exponential disks:
        R_d_star ≈ estimated from typical dwarf scaling
        R_d_gas ≈ 2 × R_d_star (standard approximation)

    Returns (log_gbar, log_gobs, V_bar) arrays
    """
    R_kpc = rc_data['R_kpc']
    Vc = rc_data['Vc']

    # Skip R=0
    mask = R_kpc > 0.01
    R = R_kpc[mask]
    V = Vc[mask]

    # g_obs = Vc² / R
    R_m = R * KPC_M
    V_ms = V * 1000
    g_obs = V_ms**2 / R_m

    # Estimate scale lengths from the RC extent
    # For dwarfs: R_d ~ R_max / 4 (approximate)
    # Better: use the halo r_c as guide — stellar disk is smaller
    R_max = R.max()
    R_d_star = max(R_max / 5, 0.3)   # kpc, conservative
    R_d_gas = 2.0 * R_d_star          # gas more extended

    # Baryonic velocities
    M_star = galaxy_props['M_star']
    M_gas = galaxy_props['M_HI'] * 1.33  # Helium correction

    V_star = exponential_disk_vcirc(R, M_star, R_d_star)
    V_gas = exponential_disk_vcirc(R, M_gas, R_d_gas)

    V_bar = np.sqrt(V_star**2 + V_gas**2)
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m

    valid = (g_obs > 0) & (g_bar > 0) & (R > 0.05)

    log_gbar = np.log10(g_bar[valid])
    log_gobs = np.log10(g_obs[valid])

    return log_gbar, log_gobs, V_bar[valid], V_star[valid], V_gas[valid]


def compute_rar_from_iso_decomposition(rc_data, galaxy_props):
    """Compute RAR using V_bar = sqrt(Vc² - V_halo²) as cross-check.

    This method is less reliable because it depends on the halo model.
    Used only for comparison.
    """
    R_kpc = rc_data['R_kpc']
    Vc = rc_data['Vc']

    mask = R_kpc > 0.01
    R = R_kpc[mask]
    V = Vc[mask]

    V_halo = iso_vhalo(R, galaxy_props['rho_0'], galaxy_props['r_c_kpc'])

    R_m = R * KPC_M
    V_ms = V * 1000
    g_obs = V_ms**2 / R_m

    V_bar2 = V**2 - V_halo**2
    V_bar2_clipped = np.maximum(V_bar2, 1.0)
    V_bar_ms = np.sqrt(V_bar2_clipped) * 1000
    g_bar = V_bar_ms**2 / R_m

    valid = (g_obs > 0) & (g_bar > 0) & (R > 0.05)

    log_gbar = np.log10(g_bar[valid])
    log_gobs = np.log10(g_obs[valid])

    return log_gbar, log_gobs, V_halo[valid], np.sqrt(V_bar2_clipped[valid])


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("MHONGOOSE INTEGRATION: Nkomo+2025 Rotation Curves")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")

results = {
    'test_name': 'mhongoose_integration',
    'source': 'Nkomo+2025 (A&A 699, A372)',
    'survey': 'MHONGOOSE (MeerKAT)',
    'galaxies': {},
}

all_log_gbar = []
all_log_gobs = []
all_names = []

for name, props in GALAXIES.items():
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")

    # Load rotation curve
    filepath = os.path.join(DATA_DIR, props['file'])
    if not os.path.exists(filepath):
        print(f"  ERROR: {filepath} not found")
        continue

    rc = load_rotation_curve(filepath)
    print(f"  Distance: {props['distance_Mpc']} Mpc")
    print(f"  M_star: {props['M_star']:.1e} Msun")
    print(f"  M_HI: {props['M_HI']:.1e} Msun")
    print(f"  M_HI/M_star: {props['M_HI']/props['M_star']:.1f}")
    print(f"  (M/L)_3.4μm: {props['ML_value']}")
    print(f"  ISO halo: ρ₀ = {props['rho_0']:.2e} Msun/pc³, r_c = {props['r_c_kpc']:.2f} kpc")
    print(f"  RC points: {len(rc['R_kpc'])} (R = 0 to {rc['R_kpc'].max():.2f} kpc)")

    # === Method A: Direct baryonic mass model ===
    log_gbar, log_gobs, V_bar, V_star, V_gas_comp = compute_rar_direct(rc, props)
    resid = rar_residuals(log_gbar, log_gobs)

    print(f"\n  [Method A: Direct baryonic mass model]")
    print(f"  RAR points: {len(log_gbar)}")
    print(f"  log g_bar range: [{log_gbar.min():.2f}, {log_gbar.max():.2f}]")
    print(f"  log g_obs range: [{log_gobs.min():.2f}, {log_gobs.max():.2f}]")
    print(f"  RAR scatter: σ = {np.std(resid):.4f} dex")
    print(f"  Mean residual: {np.mean(resid):+.4f} dex")

    # === Method B: ISO halo subtraction (cross-check) ===
    log_gbar_iso, log_gobs_iso, V_halo, V_bar_iso = compute_rar_from_iso_decomposition(rc, props)
    resid_iso = rar_residuals(log_gbar_iso, log_gobs_iso)

    print(f"\n  [Method B: ISO halo subtraction (cross-check)]")
    print(f"  RAR scatter: σ = {np.std(resid_iso):.4f} dex")
    print(f"  Mean residual: {np.mean(resid_iso):+.4f} dex")

    # Compare
    print(f"\n  Method A (direct) is preferred — independent of halo model.")
    print(f"  Method B depends on ISO fit quality (χ²_red = {props.get('chi2_red', '?')})")

    # Decomposition summary
    mask = rc['R_kpc'] > 0.01
    R_valid = rc['R_kpc'][mask]
    Vc_valid = rc['Vc'][mask]

    print(f"\n  {'R(kpc)':>8s} {'Vc':>8s} {'V_star':>8s} {'V_gas':>8s} {'V_bar':>8s} "
          f"{'log_gbar':>9s} {'log_gobs':>9s} {'resid':>8s}")
    print(f"  {'-'*72}")
    for j in range(min(len(log_gbar), len(R_valid) - 1)):
        idx = j + 1
        if idx >= len(R_valid):
            break
        print(f"  {R_valid[idx]:8.2f} {Vc_valid[idx]:8.1f} {V_star[j]:8.1f} "
              f"{V_gas_comp[j]:8.1f} {V_bar[j]:8.1f} "
              f"{log_gbar[j]:9.3f} {log_gobs[j]:9.3f} {resid[j]:+8.4f}")

    all_log_gbar.extend(log_gbar)
    all_log_gobs.extend(log_gobs)
    all_names.extend([name] * len(log_gbar))

    results['galaxies'][name] = {
        'distance_Mpc': props['distance_Mpc'],
        'M_star': props['M_star'],
        'M_HI': props['M_HI'],
        'ML_value': props['ML_value'],
        'ML_band': props['ML_band'],
        'n_rar_points': len(log_gbar),
        'log_gbar_range': [round(float(log_gbar.min()), 3), round(float(log_gbar.max()), 3)],
        'log_gobs_range': [round(float(log_gobs.min()), 3), round(float(log_gobs.max()), 3)],
        'rar_scatter': round(float(np.std(resid)), 4),
        'mean_residual': round(float(np.mean(resid)), 4),
    }


# ================================================================
# COMBINED RAR ANALYSIS
# ================================================================
print(f"\n{'='*72}")
print("COMBINED MHONGOOSE RAR")
print(f"{'='*72}")

all_log_gbar = np.array(all_log_gbar)
all_log_gobs = np.array(all_log_gobs)
all_resid = rar_residuals(all_log_gbar, all_log_gobs)

print(f"  Total RAR points: {len(all_log_gbar)} from {len(GALAXIES)} galaxies")
print(f"  log g_bar range: [{all_log_gbar.min():.2f}, {all_log_gbar.max():.2f}]")
print(f"  Combined scatter: σ = {np.std(all_resid):.4f} dex")
print(f"  Combined mean residual: {np.mean(all_resid):+.4f} dex")

# Compare with SPARC ESO444-G084 (7 points in SPARC)
print(f"\n  NOTE: ESO444-G084 is also in SPARC (7 points). MHONGOOSE provides")
print(f"  {sum(1 for n in all_names if n == 'ESO444-G084')} points with 3.4μm WISE M/L — "
      f"an independent mass decomposition.")

# Where do these points fall relative to g†?
below_gdagger = np.sum(all_log_gbar < LOG_G_DAGGER)
above_gdagger = np.sum(all_log_gbar >= LOG_G_DAGGER)
print(f"\n  Points below g†: {below_gdagger} ({below_gdagger/len(all_log_gbar)*100:.0f}%)")
print(f"  Points above g†: {above_gdagger} ({above_gdagger/len(all_log_gbar)*100:.0f}%)")
print(f"  → Both galaxies are gas-dominated dwarfs, entirely in the condensate regime")

results['combined'] = {
    'n_galaxies': len(GALAXIES),
    'n_rar_points': len(all_log_gbar),
    'scatter': round(float(np.std(all_resid)), 4),
    'mean_residual': round(float(np.mean(all_resid)), 4),
    'frac_below_gdagger': round(float(below_gdagger / len(all_log_gbar)), 3),
}

# ================================================================
# SAVE
# ================================================================
outpath = os.path.join(RESULTS_DIR, 'summary_mhongoose_integration.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: {outpath}")

# Also save RAR points for pipeline integration
rar_outpath = os.path.join(DATA_DIR, 'mhongoose_rar_points.tsv')
with open(rar_outpath, 'w') as f:
    f.write("# MHONGOOSE RAR points from Nkomo+2025\n")
    f.write("# name\tlog_gbar\tlog_gobs\n")
    for i in range(len(all_log_gbar)):
        f.write(f"{all_names[i]}\t{all_log_gbar[i]:.6f}\t{all_log_gobs[i]:.6f}\n")
print(f"  RAR points saved: {rar_outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
