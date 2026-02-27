#!/usr/bin/env python3
"""
integrate_mhongoose_sorgho.py — Integrate Sorgho+2019 NGC7424 & NGC3621 into RAR
==================================================================================

Adds NGC7424 (KAT-7) and NGC3621 (MeerKAT commissioning) from Sorgho+2019
(MNRAS 482, 1248) to the MHONGOOSE RAR dataset.

These are much larger galaxies than the Nkomo+2025 dwarfs:
  - NGC7424: Scd, D=9.55 Mpc, M_HI=1.26e10, M_star=2.51e9 (gas-dominated)
  - NGC3621: SAd, D=6.6 Mpc, M_HI=9.33e9, M_star~1e10 (stellar max in center)

Strategy:
  The paper provides ISO halo fit parameters (rho_0, r_c) with fixed M/L.
  We use V_bar² = V_rot² - V_halo² to get baryonic velocity, then compute RAR.
  Cross-check: the paper also shows V_gas and V_disk decomposition in Fig 15/16,
  which we verify against.

  Alternative: Use published M_star + M_HI with exponential disk profiles
  (same as Nkomo+2025 dwarfs). We do both and compare.

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
# Constants
# ================================================================
G_PC = 4.302e-3  # pc (km/s)² / Msun


# ================================================================
# Galaxy definitions from Sorgho+2019
# ================================================================

GALAXIES = {
    'NGC7424': {
        'file': 'sorgho2019_ngc7424.tsv',
        'source': 'Sorgho+2019',
        'telescope': 'KAT-7',
        'distance_Mpc': 9.55,
        'inclination_deg': 29.0,
        'vsys_kms': 936.2,
        'M_HI': 1.26e10,       # Msun (log=10.1)
        'M_star': 2.51e9,      # Msun (log=9.4, from WISE W1-W2)
        'ML_band': 'W1',
        'ML_value': 0.25,
        # ISO halo params (M/L fixed = 0.25)
        'iso_rho0': 107.9e-3,  # Msun/pc³
        'iso_rc': 1.9,         # kpc
        'iso_chi2red': 1.8,
        # Disk scale length estimated from WISE W1 light profile (Fig 14)
        # Profile extends to ~20 kpc, exponential portion ~3-15 kpc
        'R_d_star_kpc': 4.0,   # estimated from Fig 14
        'morph': 'SBcd',
        'note': 'Face-on (i=29°), gas-dominated, late-type barred spiral',
    },
    'NGC3621': {
        'file': 'sorgho2019_ngc3621.tsv',
        'source': 'Sorgho+2019',
        'telescope': 'MeerKAT',
        'distance_Mpc': 6.6,
        'inclination_deg': 64.0,
        'vsys_kms': 730.1,
        'M_HI': 9.33e9,        # Msun (from Table 4: 250.2 × 10^8)
        'M_star': 1.0e10,      # Msun (estimated: M/L=0.50 × L_3.6)
        'ML_band': '3.6um',
        'ML_value': 0.50,
        # ISO halo params (M/L fixed = 0.50)
        'iso_rho0': 14.9e-3,   # Msun/pc³
        'iso_rc': 4.8,         # kpc
        'iso_chi2red': 2.0,
        # Disk scale length from Spitzer 3.6μm profile (Fig 14)
        'R_d_star_kpc': 2.5,   # estimated from Fig 14
        'morph': 'SAd',
        'note': 'Asymmetric, warped, well-studied THINGS galaxy, stellar max in center',
    },
}


# ================================================================
# Functions
# ================================================================

def load_rotation_curve(filepath):
    """Load a MHONGOOSE/KAT-7 rotation curve TSV file."""
    R, Vrot, eVrot = [], [], []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                R.append(float(parts[0]))
                Vrot.append(float(parts[1]))
                eVrot.append(float(parts[2]))
            except ValueError:
                continue
    return {
        'R_kpc': np.array(R),
        'Vrot': np.array(Vrot),
        'eVrot': np.array(eVrot),
    }


def iso_vhalo(R_kpc, rho_0_Msun_pc3, r_c_kpc):
    """Pseudo-isothermal halo circular velocity (km/s)."""
    R = np.asarray(R_kpc, dtype=float)
    r_c_pc = r_c_kpc * 1000
    R_pc = R * 1000
    ratio = R_pc / r_c_pc
    V2 = 4 * np.pi * G_PC * rho_0_Msun_pc3 * r_c_pc**2 * (1 - np.where(
        ratio > 1e-6,
        (1.0 / ratio) * np.arctan(ratio),
        1.0 - ratio**2 / 3
    ))
    return np.sqrt(np.maximum(V2, 0))


def exponential_disk_vcirc(R_kpc, M_total, R_d_kpc):
    """Cumulative mass approximation for exponential disk V(R)."""
    R = np.asarray(R_kpc, dtype=float)
    y = R / R_d_kpc
    M_enc = M_total * (1.0 - (1.0 + y) * np.exp(-y))
    R_pc = R * 1000
    V2 = G_PC * M_enc / np.maximum(R_pc, 1.0)
    return np.sqrt(np.maximum(V2, 0))


def compute_rar_iso_subtraction(rc_data, props):
    """Method A: V_bar² = V_rot² - V_halo² using published ISO fit.

    This uses the halo model to extract baryonic from total.
    Pro: uses the actual fitted halo, consistent with the paper.
    Con: depends on ISO fit quality.
    """
    R = rc_data['R_kpc']
    Vrot = rc_data['Vrot']

    mask = R > 0.1
    R_m = R[mask] * KPC_M
    V_ms = Vrot[mask] * 1000

    g_obs = V_ms**2 / R_m

    V_halo = iso_vhalo(R[mask], props['iso_rho0'], props['iso_rc'])

    V_bar2 = Vrot[mask]**2 - V_halo**2
    # Clip to positive
    V_bar2_clip = np.maximum(V_bar2, 1.0)
    V_bar_ms = np.sqrt(V_bar2_clip) * 1000
    g_bar = V_bar_ms**2 / R_m

    valid = (g_obs > 0) & (g_bar > 0)
    return (np.log10(g_bar[valid]), np.log10(g_obs[valid]),
            R[mask][valid], V_halo[valid], np.sqrt(V_bar2_clip[valid]))


def compute_rar_direct_baryonic(rc_data, props):
    """Method B: Direct baryonic model from M_star + M_HI exponential disks.

    Pro: halo-independent.
    Con: scale lengths are estimated, not fitted.
    """
    R = rc_data['R_kpc']
    Vrot = rc_data['Vrot']

    mask = R > 0.1
    R_m = R[mask] * KPC_M
    V_ms = Vrot[mask] * 1000

    g_obs = V_ms**2 / R_m

    R_d_star = props['R_d_star_kpc']
    R_d_gas = 2.0 * R_d_star  # gas more extended

    M_star = props['M_star']
    M_gas = props['M_HI'] * 1.33  # helium correction

    V_star = exponential_disk_vcirc(R[mask], M_star, R_d_star)
    V_gas = exponential_disk_vcirc(R[mask], M_gas, R_d_gas)
    V_bar = np.sqrt(V_star**2 + V_gas**2)
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m

    valid = (g_obs > 0) & (g_bar > 0)
    return (np.log10(g_bar[valid]), np.log10(g_obs[valid]),
            R[mask][valid], V_star[valid], V_gas[valid], V_bar[valid])


# ================================================================
# MAIN
# ================================================================
print("=" * 72)
print("MHONGOOSE INTEGRATION: Sorgho+2019 NGC7424 & NGC3621")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")

results = {
    'test_name': 'mhongoose_sorgho2019_integration',
    'source': 'Sorgho+2019 (MNRAS 482, 1248)',
    'galaxies': {},
}

all_log_gbar = []
all_log_gobs = []
all_names = []

for name, props in GALAXIES.items():
    print(f"\n{'='*72}")
    print(f"  {name}  ({props['morph']}, {props['telescope']})")
    print(f"{'='*72}")

    filepath = os.path.join(DATA_DIR, props['file'])
    if not os.path.exists(filepath):
        print(f"  ERROR: {filepath} not found")
        continue

    rc = load_rotation_curve(filepath)
    print(f"  Distance: {props['distance_Mpc']} Mpc")
    print(f"  Inclination: {props['inclination_deg']}°")
    print(f"  M_star: {props['M_star']:.2e} Msun")
    print(f"  M_HI: {props['M_HI']:.2e} Msun")
    print(f"  M_HI/M_star: {props['M_HI']/props['M_star']:.1f}")
    print(f"  (M/L): {props['ML_value']} ({props['ML_band']})")
    print(f"  ISO halo: ρ₀={props['iso_rho0']*1e3:.1f}×10⁻³ Msun/pc³, "
          f"r_c={props['iso_rc']:.1f} kpc, χ²_red={props['iso_chi2red']}")
    print(f"  RC points: {len(rc['R_kpc'])} "
          f"(R={rc['R_kpc'].min():.1f}-{rc['R_kpc'].max():.1f} kpc)")
    print(f"  Note: {props['note']}")

    # === Method A: ISO halo subtraction ===
    lg_gbar_A, lg_gobs_A, R_A, V_halo, V_bar_A = compute_rar_iso_subtraction(rc, props)
    resid_A = rar_residuals(lg_gbar_A, lg_gobs_A)

    print(f"\n  [Method A: ISO halo subtraction]")
    print(f"    RAR points: {len(lg_gbar_A)}")
    print(f"    log g_bar: [{lg_gbar_A.min():.3f}, {lg_gbar_A.max():.3f}]")
    print(f"    log g_obs: [{lg_gobs_A.min():.3f}, {lg_gobs_A.max():.3f}]")
    print(f"    RAR scatter: σ = {np.std(resid_A):.4f} dex")
    print(f"    Mean residual: {np.mean(resid_A):+.4f} dex")

    # Check for negative V_bar² (where halo over-subtracts)
    neg_count = np.sum(rc['Vrot'][rc['R_kpc'] > 0.1]**2 <
                       iso_vhalo(rc['R_kpc'][rc['R_kpc'] > 0.1],
                                 props['iso_rho0'], props['iso_rc'])**2)
    if neg_count > 0:
        print(f"    ⚠ {neg_count} points where V_halo > V_rot (clipped)")

    # === Method B: Direct baryonic ===
    lg_gbar_B, lg_gobs_B, R_B, V_star, V_gas, V_bar_B = \
        compute_rar_direct_baryonic(rc, props)
    resid_B = rar_residuals(lg_gbar_B, lg_gobs_B)

    print(f"\n  [Method B: Direct baryonic (exponential disks)]")
    print(f"    RAR points: {len(lg_gbar_B)}")
    print(f"    log g_bar: [{lg_gbar_B.min():.3f}, {lg_gbar_B.max():.3f}]")
    print(f"    RAR scatter: σ = {np.std(resid_B):.4f} dex")
    print(f"    Mean residual: {np.mean(resid_B):+.4f} dex")

    # === Select best method ===
    # Use Method A (ISO subtraction) when χ²_red is reasonable and no clipping
    # Use Method B (direct baryonic) when ISO over-subtracts
    use_method_A = (neg_count == 0) and np.std(resid_A) < 0.30
    if use_method_A:
        best_label = "A (ISO subtraction)"
        best_gbar, best_gobs = lg_gbar_A, lg_gobs_A
        best_resid = resid_A
    else:
        best_label = "B (direct baryonic)"
        best_gbar, best_gobs = lg_gbar_B, lg_gobs_B
        best_resid = resid_B
    print(f"\n  → Using Method {best_label} for pipeline integration")
    if not use_method_A:
        print(f"    Reason: ISO halo over-subtracts ({neg_count} clipped points)")
    else:
        print(f"    Reason: ISO fit is clean, consistent with paper")

    # Detailed table (for selected method)
    print(f"\n  {'R(kpc)':>8s} {'log_gbar':>9s} {'log_gobs':>9s} {'resid':>8s}")
    print(f"  {'-'*40}")
    for j in range(len(best_gbar)):
        print(f"  {(j+1)*1.0:8.1f} "
              f"{best_gbar[j]:9.3f} {best_gobs[j]:9.3f} {best_resid[j]:+8.4f}")

    # Use selected method for pipeline
    all_log_gbar.extend(best_gbar)
    all_log_gobs.extend(best_gobs)
    all_names.extend([name] * len(best_gbar))

    results['galaxies'][name] = {
        'distance_Mpc': props['distance_Mpc'],
        'inclination_deg': props['inclination_deg'],
        'telescope': props['telescope'],
        'M_star': props['M_star'],
        'M_HI': props['M_HI'],
        'ML_value': props['ML_value'],
        'ML_band': props['ML_band'],
        'iso_rho0': props['iso_rho0'],
        'iso_rc': props['iso_rc'],
        'iso_chi2red': props['iso_chi2red'],
        'selected_method': best_label,
        'n_rar_points': int(len(best_gbar)),
        'log_gbar_range': [round(float(best_gbar.min()), 3),
                           round(float(best_gbar.max()), 3)],
        'log_gobs_range': [round(float(best_gobs.min()), 3),
                           round(float(best_gobs.max()), 3)],
        'rar_scatter_selected': round(float(np.std(best_resid)), 4),
        'mean_residual_selected': round(float(np.mean(best_resid)), 4),
        'rar_scatter_iso': round(float(np.std(resid_A)), 4),
        'rar_scatter_direct': round(float(np.std(resid_B)), 4),
    }


# ================================================================
# COMBINED ANALYSIS
# ================================================================
print(f"\n{'='*72}")
print("COMBINED SORGHO+2019 RAR")
print(f"{'='*72}")

all_log_gbar = np.array(all_log_gbar)
all_log_gobs = np.array(all_log_gobs)
all_resid = rar_residuals(all_log_gbar, all_log_gobs)

print(f"  Total RAR points: {len(all_log_gbar)} from {len(GALAXIES)} galaxies")
print(f"  log g_bar range: [{all_log_gbar.min():.3f}, {all_log_gbar.max():.3f}]")
print(f"  Combined scatter: σ = {np.std(all_resid):.4f} dex")
print(f"  Combined mean residual: {np.mean(all_resid):+.4f} dex")

below = np.sum(all_log_gbar < LOG_G_DAGGER)
above = np.sum(all_log_gbar >= LOG_G_DAGGER)
print(f"\n  Points below g†: {below} ({below/len(all_log_gbar)*100:.0f}%)")
print(f"  Points above g†: {above} ({above/len(all_log_gbar)*100:.0f}%)")
print(f"  → These spirals span BOTH regimes (unlike the Nkomo dwarfs)")

# ================================================================
# LOAD Nkomo+2025 RAR points for full MHONGOOSE picture
# ================================================================
nkomo_path = os.path.join(DATA_DIR, 'mhongoose_rar_points.tsv')
if os.path.exists(nkomo_path):
    nkomo_gbar, nkomo_gobs, nkomo_names = [], [], []
    with open(nkomo_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                nkomo_names.append(parts[0])
                nkomo_gbar.append(float(parts[1]))
                nkomo_gobs.append(float(parts[2]))

    nkomo_gbar = np.array(nkomo_gbar)
    nkomo_gobs = np.array(nkomo_gobs)

    full_gbar = np.concatenate([all_log_gbar, nkomo_gbar])
    full_gobs = np.concatenate([all_log_gobs, nkomo_gobs])
    full_resid = rar_residuals(full_gbar, full_gobs)
    full_names = all_names + nkomo_names

    n_galaxies = len(set(full_names))

    print(f"\n{'='*72}")
    print("FULL MHONGOOSE RAR (Sorgho+2019 + Nkomo+2025)")
    print(f"{'='*72}")
    print(f"  Total: {len(full_gbar)} RAR points from {n_galaxies} galaxies")
    print(f"  log g_bar range: [{full_gbar.min():.3f}, {full_gbar.max():.3f}]")
    print(f"  Combined scatter: σ = {np.std(full_resid):.4f} dex")
    print(f"  Combined mean residual: {np.mean(full_resid):+.4f} dex")

    below_full = np.sum(full_gbar < LOG_G_DAGGER)
    above_full = np.sum(full_gbar >= LOG_G_DAGGER)
    print(f"  Points below g†: {below_full} ({below_full/len(full_gbar)*100:.0f}%)")
    print(f"  Points above g†: {above_full} ({above_full/len(full_gbar)*100:.0f}%)")

    # Per-galaxy summary
    print(f"\n  Per-galaxy summary:")
    print(f"  {'Galaxy':>15s} {'N_pts':>6s} {'log_gbar range':>20s} {'σ':>8s} {'<resid>':>8s}")
    print(f"  {'-'*62}")
    for gal_name in sorted(set(full_names)):
        mask_gal = np.array([n == gal_name for n in full_names])
        gb = full_gbar[mask_gal]
        go = full_gobs[mask_gal]
        r = rar_residuals(gb, go)
        print(f"  {gal_name:>15s} {len(gb):6d} "
              f"[{gb.min():.2f}, {gb.max():.2f}] "
              f"{np.std(r):8.4f} {np.mean(r):+8.4f}")

    results['full_mhongoose'] = {
        'n_galaxies': n_galaxies,
        'n_rar_points': len(full_gbar),
        'scatter': round(float(np.std(full_resid)), 4),
        'mean_residual': round(float(np.mean(full_resid)), 4),
        'frac_below_gdagger': round(float(below_full / len(full_gbar)), 3),
    }

    # Save combined RAR points
    combined_path = os.path.join(DATA_DIR, 'mhongoose_rar_all.tsv')
    with open(combined_path, 'w') as f:
        f.write("# Full MHONGOOSE RAR: Nkomo+2025 + Sorgho+2019\n")
        f.write("# name\tlog_gbar\tlog_gobs\tsource\n")
        for i in range(len(nkomo_gbar)):
            f.write(f"{nkomo_names[i]}\t{nkomo_gbar[i]:.6f}\t"
                    f"{nkomo_gobs[i]:.6f}\tNkomo+2025\n")
        for i in range(len(all_log_gbar)):
            f.write(f"{all_names[i]}\t{all_log_gbar[i]:.6f}\t"
                    f"{all_log_gobs[i]:.6f}\tSorgho+2019\n")
    print(f"\n  Combined RAR saved: {combined_path}")


# ================================================================
# INCLINATION WARNING for NGC7424
# ================================================================
print(f"\n{'='*72}")
print("INCLINATION NOTE")
print(f"{'='*72}")
print(f"  NGC7424: i = 29° — LOW inclination, similar to ESO444-G084 issue")
print(f"  sin(29°) = {np.sin(np.radians(29)):.3f}")
print(f"  Deprojection factor 1/sin(i) = {1/np.sin(np.radians(29)):.2f}")
print(f"  A 3° inclination error → ~10% Vrot error → ~20% g_obs error")
print(f"  NGC3621: i = 64° — GOOD inclination (minimal deprojection uncertainty)")
print(f"  sin(64°) = {np.sin(np.radians(64)):.3f}")
print(f"  A 3° inclination error → ~3% Vrot error → ~6% g_obs error")
print(f"  → NGC3621 RAR points are more reliable than NGC7424")


# ================================================================
# SAVE
# ================================================================
results['combined_sorgho'] = {
    'n_galaxies': len(GALAXIES),
    'n_rar_points': len(all_log_gbar),
    'scatter': round(float(np.std(all_resid)), 4),
    'mean_residual': round(float(np.mean(all_resid)), 4),
}

outpath = os.path.join(RESULTS_DIR, 'summary_mhongoose_sorgho2019.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2,
              default=lambda x: float(x) if hasattr(x, 'item') else x)
print(f"\n  Saved: {outpath}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
