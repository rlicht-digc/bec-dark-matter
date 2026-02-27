#!/usr/bin/env python3
"""
ΛCDM Null Test: Does the Inversion Point Exist in Pure ΛCDM?
===============================================================

THE CRITICAL DISCRIMINATION TEST: We generate a synthetic ΛCDM galaxy
population using:
  - NFW dark matter halos with the c(M) relation from Dutton & Macciò 2014
  - Stellar-mass-halo-mass (SMHM) relation from Moster+ 2013
  - Exponential stellar disks with R_d scaling from Kravtsov 2013
  - Gas disks with M_gas from the gas fraction scaling
  - Realistic scatter in all relations

If ΛCDM naturally produces a scatter derivative inversion at g†,
then our observed signal is just a ΛCDM prediction.
If it does NOT, then the observed inversion discriminates between
BEC dark matter and ΛCDM.

Russell Licht -- Primordial Fluid DM Project, Feb 2026
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Physics constants
G = 6.674e-11        # m^3 kg^-1 s^-2
M_sun = 1.989e30     # kg
kpc_m = 3.086e19     # m
g_dagger = 1.20e-10  # m/s^2
LOG_G_DAGGER = np.log10(g_dagger)  # -9.921
H0 = 67.74           # km/s/Mpc
h = H0 / 100.0
rho_crit = 1.27e11 * h**2  # M_sun / Mpc^3
Omega_m = 0.3089

np.random.seed(42)

N_GALAXIES = 500   # Synthetic galaxy count
N_RADII = 20       # Radial points per galaxy

print("=" * 72)
print("ΛCDM NULL TEST: SCATTER DERIVATIVE INVERSION")
print("=" * 72)
print(f"  g† = {g_dagger:.2e} m/s², log g† = {LOG_G_DAGGER:.3f}")
print(f"  N_galaxies = {N_GALAXIES}")
print(f"  N_radii per galaxy = {N_RADII}")


# ================================================================
# 1. GENERATE NFW HALO POPULATION
# ================================================================
print("\n[1] Generating NFW halo population...")

# Log-uniform halo mass distribution matching SPARC range
# SPARC galaxies span M_halo ~ 10^10 to 10^13 M_sun
log_Mhalo = np.random.uniform(10.0, 13.0, N_GALAXIES)
M_halo = 10**log_Mhalo  # M_sun

# Concentration-mass relation: Dutton & Macciò (2014), z=0
# log10(c) = a + b * log10(M_halo / (10^12 h^-1 M_sun))
# a = 0.905, b = -0.101 (relaxed halos, Planck cosmology)
a_cm = 0.905
b_cm = -0.101
sigma_c = 0.11  # Log-normal scatter in c(M)

log_c200 = a_cm + b_cm * (log_Mhalo - 12.0 + np.log10(h))
log_c200 += np.random.normal(0, sigma_c, N_GALAXIES)
c200 = 10**log_c200

# Virial radius R200 (where mean density = 200 * rho_crit)
# M200 = (4/3) pi R200^3 * 200 * rho_crit
# R200 in kpc
rho_200 = 200.0 * rho_crit  # M_sun / Mpc^3
R200_Mpc = (3.0 * M_halo / (4.0 * np.pi * rho_200))**(1.0/3.0)  # Mpc
R200_kpc = R200_Mpc * 1000.0  # kpc

# NFW scale radius
r_s = R200_kpc / c200  # kpc

print(f"  M_halo range: {M_halo.min():.2e} - {M_halo.max():.2e} M_sun")
print(f"  c200 range: {c200.min():.1f} - {c200.max():.1f}")
print(f"  R200 range: {R200_kpc.min():.0f} - {R200_kpc.max():.0f} kpc")


# ================================================================
# 2. ASSIGN STELLAR MASSES (SMHM relation)
# ================================================================
print("\n[2] Assigning stellar masses (Moster+ 2013 SMHM)...")

# Moster+ 2013 SMHM relation at z=0
# M_star / M_halo = 2 * N * [ (M_halo/M1)^-beta + (M_halo/M1)^gamma ]^-1
M1 = 10**11.590  # Characteristic halo mass
N_moster = 0.0351
beta_moster = 1.376
gamma_moster = 0.608

ratio = M_halo / M1
f_star = 2.0 * N_moster / (ratio**(-beta_moster) + ratio**gamma_moster)
log_Mstar_mean = np.log10(f_star * M_halo)

# Add scatter: 0.15 dex (observational constraint)
sigma_smhm = 0.15
log_Mstar = log_Mstar_mean + np.random.normal(0, sigma_smhm, N_GALAXIES)
M_star = 10**log_Mstar

print(f"  M_star range: {M_star.min():.2e} - {M_star.max():.2e} M_sun")


# ================================================================
# 3. ASSIGN GAS MASSES
# ================================================================
print("\n[3] Assigning gas masses...")

# Gas fraction scaling: f_gas = M_gas / M_star decreases with M_star
# Approximate: log(f_gas) ~ -0.5 * (log M_star - 9.0) + 0.3
# i.e., dwarfs are gas-rich (~200%), massive spirals are gas-poor (~10%)
log_fgas = -0.5 * (log_Mstar - 9.0) + 0.3
log_fgas += np.random.normal(0, 0.3, N_GALAXIES)  # Large scatter
log_fgas = np.clip(log_fgas, -1.5, 1.5)  # 3% to 3000%
f_gas = 10**log_fgas
M_gas = f_gas * M_star

print(f"  M_gas range: {M_gas.min():.2e} - {M_gas.max():.2e} M_sun")
print(f"  f_gas range: {f_gas.min():.2f} - {f_gas.max():.2f}")


# ================================================================
# 4. ASSIGN DISK SCALE LENGTHS
# ================================================================
print("\n[4] Assigning disk scale lengths...")

# Kravtsov 2013: R_d ~ 0.015 * R200 (with scatter)
# R_d in kpc
sigma_Rd = 0.2  # dex scatter
log_Rd = np.log10(0.015 * R200_kpc) + np.random.normal(0, sigma_Rd, N_GALAXIES)
R_d = 10**log_Rd  # kpc
R_d = np.clip(R_d, 0.3, 30.0)  # Physical bounds

# Gas disk scale length: typically 2-3x stellar disk
R_gas = 2.0 * R_d

print(f"  R_d range: {R_d.min():.1f} - {R_d.max():.1f} kpc")


# ================================================================
# 5. COMPUTE g_bar AND g_obs AT MULTIPLE RADII
# ================================================================
print("\n[5] Computing mock RAR data...")

def nfw_enclosed_mass(r_kpc, M200, c, R200):
    """Enclosed NFW mass within radius r (kpc). All in solar mass & kpc units."""
    rs = R200 / c
    x = r_kpc / rs
    x200 = c  # = R200/rs

    # M_NFW(r) = M200 * [ln(1+x) - x/(1+x)] / [ln(1+c) - c/(1+c)]
    nfw_norm = np.log(1.0 + x200) - x200 / (1.0 + x200)
    m_enclosed = M200 * (np.log(1.0 + x) - x / (1.0 + x)) / nfw_norm
    return m_enclosed


def exponential_disk_enclosed_mass(r_kpc, M_total, Rd):
    """Enclosed mass of an exponential disk within radius r.

    M(<r) = M_total * [1 - (1 + r/Rd) * exp(-r/Rd)]
    """
    y = r_kpc / Rd
    return M_total * (1.0 - (1.0 + y) * np.exp(-y))


all_log_gbar = []
all_log_gobs = []

for i in range(N_GALAXIES):
    # Radial grid: 0.5 R_d to 10 R_d (typical SPARC coverage)
    r_min = max(0.5 * R_d[i], 0.3)
    r_max = min(10.0 * R_d[i], R200_kpc[i] * 0.15)
    if r_max <= r_min:
        r_max = r_min * 10.0

    radii = np.linspace(r_min, r_max, N_RADII)  # kpc

    # Baryonic mass enclosed
    M_star_enc = exponential_disk_enclosed_mass(radii, M_star[i], R_d[i])
    M_gas_enc = exponential_disk_enclosed_mass(radii, M_gas[i], R_gas[i])
    M_bar_enc = M_star_enc + M_gas_enc

    # Total mass enclosed (NFW halo includes DM only; add baryons)
    # In ΛCDM, total = DM + baryons, but NFW is fit to total M200
    # Standard approach: M_DM(r) = NFW(r) * (1 - f_b) where f_b = M_bar/M200
    f_b = (M_star[i] + M_gas[i]) / M_halo[i]
    f_b = min(f_b, 0.90)  # Physical cap

    M_DM_enc = nfw_enclosed_mass(radii, M_halo[i] * (1.0 - f_b), c200[i], R200_kpc[i])
    M_total_enc = M_DM_enc + M_bar_enc

    # Accelerations in m/s^2
    # g = G * M / r^2, with M in kg and r in m
    r_m = radii * kpc_m  # meters

    g_bar = G * M_bar_enc * M_sun / r_m**2
    g_obs = G * M_total_enc * M_sun / r_m**2

    # Add observational noise: 10% velocity errors → ~20% acceleration errors
    # g ∝ V^2/r, so σ_g/g ≈ 2 * σ_V/V
    vel_noise = 0.10  # 10% velocity uncertainty
    log_noise = 2 * vel_noise / np.log(10)  # ≈ 0.087 dex

    log_gbar_noisy = np.log10(np.maximum(g_bar, 1e-15)) + np.random.normal(0, log_noise * 0.5, N_RADII)
    log_gobs_noisy = np.log10(np.maximum(g_obs, 1e-15)) + np.random.normal(0, log_noise, N_RADII)

    # Quality cut: physical range
    valid = (log_gbar_noisy > -13.0) & (log_gbar_noisy < -8.0)
    valid &= (log_gobs_noisy > -13.0) & (log_gobs_noisy < -8.0)

    if np.sum(valid) >= 5:
        all_log_gbar.extend(log_gbar_noisy[valid])
        all_log_gobs.extend(log_gobs_noisy[valid])

all_log_gbar = np.array(all_log_gbar)
all_log_gobs = np.array(all_log_gobs)

print(f"  Total mock RAR points: {len(all_log_gbar)}")
print(f"  log g_bar range: {all_log_gbar.min():.2f} to {all_log_gbar.max():.2f}")
print(f"  log g_obs range: {all_log_gobs.min():.2f} to {all_log_gobs.max():.2f}")


# ================================================================
# 6. SCATTER DERIVATIVE ANALYSIS (IDENTICAL to SPARC analysis)
# ================================================================
print("\n[6] Computing scatter derivative for mock ΛCDM data...")

def compute_scatter_profile(log_gbar, log_gobs, bin_width=0.30, min_pts=20):
    """Compute binned scatter profile and its derivative."""
    # Fit RAR to get residuals
    gbar = 10**log_gbar
    rar_pred = np.log10(gbar / (1.0 - np.exp(-np.sqrt(gbar / g_dagger))))
    resid = log_gobs - rar_pred

    lo = log_gbar.min()
    hi = log_gbar.max()
    edges = np.arange(lo, hi + bin_width, bin_width)

    centers = []
    sigmas = []
    n_per_bin = []
    for j in range(len(edges) - 1):
        mask = (log_gbar >= edges[j]) & (log_gbar < edges[j+1])
        n = np.sum(mask)
        if n >= min_pts:
            centers.append(0.5 * (edges[j] + edges[j+1]))
            sigmas.append(np.std(resid[mask]))
            n_per_bin.append(n)

    return np.array(centers), np.array(sigmas), np.array(n_per_bin)


def find_inversion(log_gbar, log_gobs, bin_width=0.30, offset=0.0):
    """Find scatter derivative zero-crossing nearest to g†."""
    gbar = 10**log_gbar
    rar_pred = np.log10(gbar / (1.0 - np.exp(-np.sqrt(gbar / g_dagger))))
    resid = log_gobs - rar_pred

    lo = log_gbar.min() + offset
    hi = log_gbar.max()
    edges = np.arange(lo, hi + bin_width, bin_width)

    if len(edges) < 3:
        return None, None, None

    centers = []
    sigmas = []
    for j in range(len(edges) - 1):
        mask = (log_gbar >= edges[j]) & (log_gbar < edges[j+1])
        if np.sum(mask) >= 10:
            centers.append(0.5 * (edges[j] + edges[j+1]))
            sigmas.append(np.std(resid[mask]))

    if len(centers) < 4:
        return None, None, None

    centers = np.array(centers)
    sigmas = np.array(sigmas)

    dsigma = np.diff(sigmas)
    dcenter = np.array([0.5 * (centers[j] + centers[j+1]) for j in range(len(centers)-1)])

    # Find ALL positive-to-negative crossings
    crossings = []
    for j in range(len(dsigma) - 1):
        if dsigma[j] > 0 and dsigma[j+1] < 0:
            x0, x1 = dcenter[j], dcenter[j+1]
            y0, y1 = dsigma[j], dsigma[j+1]
            crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(crossing)

    if not crossings:
        return None, centers, sigmas

    crossings = np.array(crossings)
    nearest_idx = np.argmin(np.abs(crossings - LOG_G_DAGGER))
    return crossings[nearest_idx], centers, sigmas


# Run the analysis on mock ΛCDM data
inv_lcdm, centers_lcdm, sigmas_lcdm = find_inversion(all_log_gbar, all_log_gobs)

print(f"\n  Scatter profile bins: {len(centers_lcdm)}")
print(f"  Scatter range: {sigmas_lcdm.min():.4f} - {sigmas_lcdm.max():.4f} dex")

if inv_lcdm is not None:
    delta = abs(inv_lcdm - LOG_G_DAGGER)
    print(f"\n  *** ΛCDM inversion found at log g = {inv_lcdm:.3f}")
    print(f"  *** Distance from g†: {delta:.3f} dex")
    if delta < 0.20:
        print(f"  *** RESULT: ΛCDM DOES produce inversion near g† → NOT discriminating")
    else:
        print(f"  *** RESULT: ΛCDM inversion at {inv_lcdm:.3f}, NOT at g† ({LOG_G_DAGGER:.3f})")
else:
    print(f"\n  *** NO scatter derivative inversion found in ΛCDM mock")
    print(f"  *** RESULT: ΛCDM does NOT produce the inversion → DISCRIMINATING")


# ================================================================
# 7. ROBUSTNESS: MULTIPLE BIN OFFSETS
# ================================================================
print("\n[7] Robustness check: multiple bin offsets...")

offsets = np.linspace(0, 0.25, 10)
inv_offsets = []
for off in offsets:
    inv_val, _, _ = find_inversion(all_log_gbar, all_log_gobs, bin_width=0.30, offset=off)
    if inv_val is not None:
        inv_offsets.append(inv_val)

if inv_offsets:
    inv_offsets = np.array(inv_offsets)
    print(f"  Inversions found in {len(inv_offsets)}/{len(offsets)} offsets")
    print(f"  Mean inversion: {inv_offsets.mean():.3f} ± {inv_offsets.std():.3f}")
    mean_delta = abs(inv_offsets.mean() - LOG_G_DAGGER)
    print(f"  Mean distance from g†: {mean_delta:.3f} dex")
else:
    print(f"  No inversions found at any offset")


# ================================================================
# 8. MULTIPLE BIN WIDTHS
# ================================================================
print("\n[8] Robustness check: multiple bin widths...")

bin_widths = [0.20, 0.25, 0.30, 0.35, 0.40]
inv_widths = []
for bw in bin_widths:
    inv_val, _, _ = find_inversion(all_log_gbar, all_log_gobs, bin_width=bw)
    if inv_val is not None:
        inv_widths.append(inv_val)
        print(f"  bin_width={bw:.2f}: inversion at {inv_val:.3f}, delta = {abs(inv_val - LOG_G_DAGGER):.3f}")
    else:
        print(f"  bin_width={bw:.2f}: no inversion found")


# ================================================================
# 9. COMPARISON WITH OBSERVED SPARC DATA
# ================================================================
print("\n[9] Loading real SPARC data for comparison...")

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'sparc')
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
            'L36': float(parts[6]), 'Q': int(parts[16]),
        }
    except (ValueError, IndexError):
        continue

# Compute SPARC RAR
sparc_log_gbar = []
sparc_log_gobs = []
for name, gdata in galaxies.items():
    if name not in sparc_props:
        continue
    prop = sparc_props[name]
    if prop['Q'] > 2 or prop['Inc'] < 30 or prop['Inc'] > 85:
        continue

    R = gdata['R']
    Vobs = gdata['Vobs']
    Vgas = gdata['Vgas']
    Vdisk = gdata['Vdisk']
    Vbul = gdata['Vbul']

    Vbar_sq = 0.5 * Vdisk**2 + Vgas * np.abs(Vgas) + 0.7 * Vbul * np.abs(Vbul)
    gbar_SI = np.where(R > 0, np.abs(Vbar_sq) * (1e3)**2 / (R * kpc_m), 1e-15)
    gobs_SI = np.where(R > 0, (Vobs * 1e3)**2 / (R * kpc_m), 1e-15)

    valid = (gbar_SI > 1e-15) & (gobs_SI > 1e-15) & (R > 0) & (Vobs > 5)
    if np.sum(valid) < 5:
        continue

    sparc_log_gbar.extend(np.log10(gbar_SI[valid]))
    sparc_log_gobs.extend(np.log10(gobs_SI[valid]))

sparc_log_gbar = np.array(sparc_log_gbar)
sparc_log_gobs = np.array(sparc_log_gobs)
print(f"  SPARC RAR points: {len(sparc_log_gbar)}")

# SPARC inversion
inv_sparc, centers_sparc, sigmas_sparc = find_inversion(sparc_log_gbar, sparc_log_gobs)
print(f"  SPARC inversion: {inv_sparc:.3f}" if inv_sparc is not None else "  SPARC: no inversion")


# ================================================================
# 10. FULL SCATTER PROFILE COMPARISON
# ================================================================
print("\n[10] Full scatter profile comparison:")
print(f"\n  {'log g_bar':>10} | {'σ_SPARC':>8} | {'σ_ΛCDM':>8} | {'Δσ':>8}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

# Unified binning for comparison
bin_width = 0.30
lo = max(all_log_gbar.min(), sparc_log_gbar.min())
hi = min(all_log_gbar.max(), sparc_log_gbar.max())
edges = np.arange(lo, hi + bin_width, bin_width)

for j in range(len(edges) - 1):
    mask_lcdm = (all_log_gbar >= edges[j]) & (all_log_gbar < edges[j+1])
    mask_sparc = (sparc_log_gbar >= edges[j]) & (sparc_log_gbar < edges[j+1])

    center = 0.5 * (edges[j] + edges[j+1])

    n_l = np.sum(mask_lcdm)
    n_s = np.sum(mask_sparc)

    if n_l >= 10 and n_s >= 10:
        sig_l = np.std(all_log_gobs[mask_lcdm] - np.log10(10**all_log_gbar[mask_lcdm] / (1 - np.exp(-np.sqrt(10**all_log_gbar[mask_lcdm] / g_dagger)))))
        sig_s = np.std(sparc_log_gobs[mask_sparc] - np.log10(10**sparc_log_gbar[mask_sparc] / (1 - np.exp(-np.sqrt(10**sparc_log_gbar[mask_sparc] / g_dagger)))))
        delta_sig = sig_l - sig_s
        marker = " <-- g†" if abs(center - LOG_G_DAGGER) < 0.20 else ""
        print(f"  {center:10.2f} | {sig_s:8.4f} | {sig_l:8.4f} | {delta_sig:+8.4f}{marker}")


# ================================================================
# 11. SCATTER DERIVATIVE COMPARISON
# ================================================================
print("\n[11] Scatter derivative comparison:")

def compute_derivative(centers, sigmas):
    """Compute dσ/d(log g_bar) via finite differences."""
    dsigma = np.diff(sigmas)
    dx = np.diff(centers)
    deriv = dsigma / dx
    dcenter = 0.5 * (centers[:-1] + centers[1:])
    return dcenter, deriv

if centers_sparc is not None and len(centers_sparc) >= 4:
    dc_sparc, deriv_sparc = compute_derivative(centers_sparc, sigmas_sparc)
    print(f"\n  SPARC scatter derivative:")
    for j in range(len(dc_sparc)):
        marker = " <-- g†" if abs(dc_sparc[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {dc_sparc[j]:.2f}: dσ/d(log g) = {deriv_sparc[j]:+.4f}{marker}")

if centers_lcdm is not None and len(centers_lcdm) >= 4:
    dc_lcdm, deriv_lcdm = compute_derivative(centers_lcdm, sigmas_lcdm)
    print(f"\n  ΛCDM scatter derivative:")
    for j in range(len(dc_lcdm)):
        marker = " <-- g†" if abs(dc_lcdm[j] - LOG_G_DAGGER) < 0.20 else ""
        print(f"    log g = {dc_lcdm[j]:.2f}: dσ/d(log g) = {deriv_lcdm[j]:+.4f}{marker}")


# ================================================================
# 12. MONTE CARLO: STABILITY OF ΛCDM NULL RESULT
# ================================================================
print("\n[12] Monte Carlo: regenerate ΛCDM population 100 times...")

n_mc = 100
mc_inversions = []
mc_near_gdagger = 0

for mc in range(n_mc):
    np.random.seed(1000 + mc)

    # Quick regeneration
    log_Mh = np.random.uniform(10.0, 13.0, N_GALAXIES)
    Mh = 10**log_Mh
    lc = a_cm + b_cm * (log_Mh - 12.0 + np.log10(h)) + np.random.normal(0, sigma_c, N_GALAXIES)
    c_mc = 10**lc
    R2_Mpc = (3.0 * Mh / (4.0 * np.pi * rho_200))**(1.0/3.0)
    R2_kpc = R2_Mpc * 1000.0
    rs_mc = R2_kpc / c_mc

    ratio_mc = Mh / M1
    f_mc = 2.0 * N_moster / (ratio_mc**(-beta_moster) + ratio_mc**gamma_moster)
    lMs = np.log10(f_mc * Mh) + np.random.normal(0, sigma_smhm, N_GALAXIES)
    Ms = 10**lMs

    lfg = -0.5 * (lMs - 9.0) + 0.3 + np.random.normal(0, 0.3, N_GALAXIES)
    lfg = np.clip(lfg, -1.5, 1.5)
    Mg = 10**lfg * Ms

    lRd = np.log10(0.015 * R2_kpc) + np.random.normal(0, sigma_Rd, N_GALAXIES)
    Rd_mc = np.clip(10**lRd, 0.3, 30.0)
    Rg_mc = 2.0 * Rd_mc

    mc_gbar = []
    mc_gobs = []

    for ii in range(N_GALAXIES):
        r_min_mc = max(0.5 * Rd_mc[ii], 0.3)
        r_max_mc = min(10.0 * Rd_mc[ii], R2_kpc[ii] * 0.15)
        if r_max_mc <= r_min_mc:
            r_max_mc = r_min_mc * 10.0
        radii_mc = np.linspace(r_min_mc, r_max_mc, N_RADII)

        Mse = exponential_disk_enclosed_mass(radii_mc, Ms[ii], Rd_mc[ii])
        Mge = exponential_disk_enclosed_mass(radii_mc, Mg[ii], Rg_mc[ii])
        Mbe = Mse + Mge

        fb_mc = min((Ms[ii] + Mg[ii]) / Mh[ii], 0.90)
        Mde = nfw_enclosed_mass(radii_mc, Mh[ii] * (1.0 - fb_mc), c_mc[ii], R2_kpc[ii])
        Mte = Mde + Mbe

        r_m_mc = radii_mc * kpc_m
        gb = G * Mbe * M_sun / r_m_mc**2
        go = G * Mte * M_sun / r_m_mc**2

        lgb = np.log10(np.maximum(gb, 1e-15)) + np.random.normal(0, 0.04, N_RADII)
        lgo = np.log10(np.maximum(go, 1e-15)) + np.random.normal(0, 0.087, N_RADII)

        v = (lgb > -13) & (lgb < -8) & (lgo > -13) & (lgo < -8)
        if np.sum(v) >= 5:
            mc_gbar.extend(lgb[v])
            mc_gobs.extend(lgo[v])

    mc_gbar = np.array(mc_gbar)
    mc_gobs = np.array(mc_gobs)

    inv_mc, _, _ = find_inversion(mc_gbar, mc_gobs)
    if inv_mc is not None:
        mc_inversions.append(inv_mc)
        if abs(inv_mc - LOG_G_DAGGER) < 0.20:
            mc_near_gdagger += 1

print(f"  Inversions found in {len(mc_inversions)}/{n_mc} realizations")
if mc_inversions:
    mc_inversions = np.array(mc_inversions)
    print(f"  Mean inversion: {mc_inversions.mean():.3f} ± {mc_inversions.std():.3f}")
    print(f"  Near g† (|Δ| < 0.20 dex): {mc_near_gdagger}/{n_mc} = {mc_near_gdagger/n_mc*100:.0f}%")
    print(f"  Mean distance from g†: {np.mean(np.abs(mc_inversions - LOG_G_DAGGER)):.3f} dex")
else:
    print(f"  NO inversions found in any realization")


# ================================================================
# 13. FINAL VERDICT
# ================================================================
print("\n" + "=" * 72)
print("FINAL VERDICT")
print("=" * 72)

# Determine conclusion
lcdm_has_inversion = inv_lcdm is not None
lcdm_near_gdagger = lcdm_has_inversion and abs(inv_lcdm - LOG_G_DAGGER) < 0.20
sparc_has_inversion = inv_sparc is not None

if sparc_has_inversion:
    print(f"\n  SPARC (observed): inversion at log g = {inv_sparc:.3f}")
    print(f"    Distance from g†: {abs(inv_sparc - LOG_G_DAGGER):.3f} dex")

if lcdm_has_inversion:
    print(f"\n  ΛCDM (synthetic): inversion at log g = {inv_lcdm:.3f}")
    print(f"    Distance from g†: {abs(inv_lcdm - LOG_G_DAGGER):.3f} dex")
else:
    print(f"\n  ΛCDM (synthetic): NO inversion found")

mc_rate = mc_near_gdagger / n_mc if len(mc_inversions) > 0 else 0.0
print(f"\n  Monte Carlo: {mc_near_gdagger}/{n_mc} ΛCDM realizations find inversion near g†")

if not lcdm_near_gdagger and mc_rate < 0.10:
    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  DISCRIMINATING: The scatter derivative inversion at g†     ║")
    print(f"  ║  is NOT a generic ΛCDM prediction.                         ║")
    print(f"  ║                                                             ║")
    print(f"  ║  OBSERVED in SPARC: YES (at log g = {inv_sparc:.3f})             ║" if inv_sparc else "")
    print(f"  ║  PREDICTED by BEC DM: YES (at g† = {LOG_G_DAGGER:.3f})           ║")
    print(f"  ║  PRODUCED by ΛCDM: NO ({mc_near_gdagger}/{n_mc} realizations)              ║")
    print(f"  ║                                                             ║")
    print(f"  ║  This turns 'consistent with BEC' → 'predicted by BEC,     ║")
    print(f"  ║  absent in ΛCDM, observed.'                                 ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
elif lcdm_near_gdagger or mc_rate > 0.50:
    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  NOT DISCRIMINATING: ΛCDM also produces the inversion.     ║")
    print(f"  ║  The inversion is a generic feature of galaxy formation,   ║")
    print(f"  ║  not specific evidence for BEC dark matter.                 ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
else:
    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  PARTIALLY DISCRIMINATING: ΛCDM rarely produces the        ║")
    print(f"  ║  inversion at g†. MC rate: {mc_rate:.0%}                           ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")


# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    'test': 'lcdm_null_inversion',
    'description': 'Does the scatter derivative inversion at g† exist in pure LCDM?',
    'parameters': {
        'N_galaxies': N_GALAXIES,
        'N_radii_per_galaxy': N_RADII,
        'total_rar_points': len(all_log_gbar),
        'smhm_scatter_dex': sigma_smhm,
        'concentration_scatter_dex': sigma_c,
        'disk_size_scatter_dex': sigma_Rd,
    },
    'sparc_result': {
        'inversion_log_g': float(inv_sparc) if inv_sparc is not None else None,
        'delta_from_gdagger': float(abs(inv_sparc - LOG_G_DAGGER)) if inv_sparc is not None else None,
        'n_rar_points': len(sparc_log_gbar),
    },
    'lcdm_result': {
        'inversion_log_g': float(inv_lcdm) if inv_lcdm is not None else None,
        'delta_from_gdagger': float(abs(inv_lcdm - LOG_G_DAGGER)) if inv_lcdm is not None else None,
        'n_rar_points': len(all_log_gbar),
    },
    'monte_carlo': {
        'n_realizations': n_mc,
        'inversions_found': len(mc_inversions),
        'near_gdagger_count': mc_near_gdagger,
        'near_gdagger_rate': mc_rate,
        'mean_inversion': float(mc_inversions.mean()) if len(mc_inversions) > 0 else None,
        'std_inversion': float(mc_inversions.std()) if len(mc_inversions) > 0 else None,
    },
    'scatter_profile_lcdm': {
        'centers': [float(x) for x in centers_lcdm] if centers_lcdm is not None else [],
        'sigmas': [float(x) for x in sigmas_lcdm] if sigmas_lcdm is not None else [],
    },
    'verdict': 'DISCRIMINATING' if (not lcdm_near_gdagger and mc_rate < 0.10) else
               'NOT_DISCRIMINATING' if (lcdm_near_gdagger or mc_rate > 0.50) else
               'PARTIALLY_DISCRIMINATING',
}

outpath = os.path.join(RESULTS_DIR, 'summary_lcdm_null_inversion.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {outpath}")
print("=" * 72)
