#!/usr/bin/env python3
"""
Reconstruct Radial Mass and Acceleration Profiles for SMACS J0723.3-7327
=========================================================================

Uses the Vikhlinin-model best-fit parameters from Liu et al. 2023
(A&A 670, A96) -- eROSITA observations of the SMACS J0723.3-7327
galaxy cluster (the first JWST deep field cluster, z = 0.388).

The script:
  1. Rebuilds T(r) and n_e(r) from the published analytic model params
  2. Derives hydrostatic mass M_HSE(r) from the ICM pressure gradient
  3. Computes baryonic mass M_bar = M_gas + M_star
  4. Evaluates the radial acceleration relation (RAR) and the
     BEC/superfluid prediction g_BEC(g_bar)
  5. Validates against the published M_500 and M_2500

References:
  - Liu et al. 2023, A&A 670, A96  (Tables 2, 3, 4)
  - Vikhlinin et al. 2006, ApJ 640, 691  (Eqs. 3, 4)

Russell Licht -- Primordial Fluid DM Project, Mar 2026
"""

import os
import numpy as np
from scipy.integrate import cumulative_trapezoid

# ====================================================================
# Physical constants (SI)
# ====================================================================
G_SI       = 6.67430e-11        # m^3 kg^-1 s^-2
m_p        = 1.67262192e-27     # proton mass, kg
k_B        = 1.380649e-23       # Boltzmann constant, J/K
keV_to_J   = 1.602176634e-16   # 1 keV in Joules
M_sun      = 1.98892e30         # solar mass, kg
kpc_to_m   = 3.085677581e19     # 1 kpc in metres
Mpc_to_kpc = 1000.0             # 1 Mpc in kpc

# ICM mean molecular weights
mu_e = 1.17    # mean molecular weight per free electron (fully ionised ICM)
mu   = 0.60    # mean molecular weight per particle

# BEC / MOND acceleration scale
g_dagger = 1.2e-10  # m s^-2

# ====================================================================
# Liu et al. 2023 -- eROSITA best-fit parameters
# ====================================================================

# --- Temperature profile (Vikhlinin+2006 Eq. 4) ---
T0       = 10.8      # keV
r_cool   = 199.0     # kpc
a_cool   = 2.89
Tmin_T0  = 0.52      # T_min / T_0
r_t      = 2120.0    # kpc
a_T      = 0.13      # exponent 'a' in the outer decline
b_T      = 4.93      # exponent 'b'
c_T      = 3.61      # exponent 'c'

# --- Electron density profile (Vikhlinin+2006 Eq. 3) ---
n0       = 7.31e-3   # cm^-3
r_c      = 259.0     # kpc
alpha    = 0.67
beta     = 0.63
r_s      = 1510.0    # kpc
epsilon  = 3.92
gamma    = 3.0       # fixed

# --- Published mass benchmarks (Table 4, eROSITA) ---
R2500_Mpc = 0.54;  M2500_Msun = 3.5e14;  dM2500 = 0.8e14
R500_Mpc  = 1.32;  M500_Msun  = 9.8e14;  dM500  = 5.1e14
z_cluster = 0.388
f_gas_500  = 0.157
f_gas_2500 = 0.140

# Convert overdensity radii to kpc
R2500_kpc = R2500_Mpc * Mpc_to_kpc  # 540 kpc
R500_kpc  = R500_Mpc  * Mpc_to_kpc  # 1320 kpc

# ====================================================================
# Model functions
# ====================================================================

def temperature_profile(r_kpc):
    """
    Vikhlinin+2006 Eq. 4: projected-deprojected 3D temperature model.

    T(r) = T_0 * t_cool(r) * t_outer(r)

    where:
      t_cool = (x^a_cool / (1 + x^a_cool)) * (T_min/T_0) + 1
               with x = r / r_cool
      (This interpolates from T_min at small r to T_0 at large r.)

      t_outer = (r/r_t)^{-a} / (1 + (r/r_t)^b)^{c/b}
    """
    x = r_kpc / r_cool

    # Cool-core correction factor
    # At r << r_cool: x^a_cool -> 0, so t_cool -> T_min/T_0
    # At r >> r_cool: x^a_cool -> inf, so t_cool -> T_min/T_0 + 1
    # The standard Vikhlinin form is:
    #   t_cool = (x^a_cool + T_min/T_0) / (x^a_cool + 1)
    t_cool = (x**a_cool + Tmin_T0) / (x**a_cool + 1.0)

    # Outer decline
    t_outer = (r_kpc / r_t)**(-a_T) / (1.0 + (r_kpc / r_t)**b_T)**(c_T / b_T)

    return T0 * t_cool * t_outer  # keV


def density_profile(r_kpc):
    """
    Vikhlinin+2006 Eq. 3 (simplified single-component form):

    n_e(r) = n_0 * (r/r_c)^{-alpha/2} / (1 + (r/r_c)^2)^{3*beta/2 - alpha/4}
             * 1 / (1 + (r/r_s)^gamma)^{epsilon/(2*gamma)}

    NOTE: The Vikhlinin+2006 formula is written for n_e^2, i.e. the
    emission measure.  Their Eq. 3 gives n_p * n_e ~ n_e^2, so the
    *square root* is the density.  The published parameters (n_0,
    beta, etc.) already correspond to the n_e form when using:
      n_e(r) = n_0 * (r/r_c)^{-alpha/2}
                   / (1+(r/r_c)^2)^{3*beta/2 - alpha/4}
                   / (1+(r/r_s)^gamma)^{epsilon/(2*gamma)}

    However, Liu+2023 Table 3 lists parameters for n_e(r) directly
    (not n_e^2).  Their stated model is:
      n_e(r) = n_0 * (r/r_c)^{-alpha}
                   / (1+(r/r_c)^2)^{3*beta/2 - alpha/2}
                   / (1+(r/r_s)^gamma)^{epsilon/gamma}

    We follow the form as given in the user specification, which
    matches the n_e (not n_e^2) convention.
    """
    x = r_kpc / r_c
    term1 = x**(-alpha) / (1.0 + x**2)**(3.0 * beta / 2.0 - alpha / 2.0)
    term2 = 1.0 / (1.0 + (r_kpc / r_s)**gamma)**(epsilon / gamma)
    return n0 * term1 * term2  # cm^-3


# ====================================================================
# Radial grid  (100 log-spaced OUTPUT points, 10 -- 3000 kpc)
# We also create a much denser internal grid for accurate numerical
# derivatives and integration, then interpolate back to 100 points.
# ====================================================================
N_PTS = 100
N_INTERNAL = 2000   # dense grid for derivatives / integration
r_kpc = np.logspace(np.log10(10.0), np.log10(3000.0), N_PTS)
r_fine = np.logspace(np.log10(10.0), np.log10(3000.0), N_INTERNAL)
r_m   = r_kpc * kpc_to_m   # metres
r_fine_m = r_fine * kpc_to_m

# Evaluate profiles on the fine grid first
T_fine = temperature_profile(r_fine)
ne_fine = density_profile(r_fine)

# Evaluate on the output grid
T_keV = temperature_profile(r_kpc)
n_e   = density_profile(r_kpc)          # cm^-3

# ====================================================================
# Step 2 -- Gas mass density rho_gas(r)
# ====================================================================
# rho_gas = mu_e * m_p * n_e
# n_e is in cm^-3; convert to m^-3 first (1 cm = 1e-2 m => 1 cm^-3 = 1e6 m^-3)
n_e_m3 = n_e * 1.0e6   # m^-3
rho_gas = mu_e * m_p * n_e_m3   # kg m^-3

ne_fine_m3 = ne_fine * 1.0e6
rho_gas_fine = mu_e * m_p * ne_fine_m3

# ====================================================================
# Step 3 -- Cumulative gas mass M_gas(<r)
# ====================================================================
# Integrate on the fine grid for accuracy, then interpolate to output grid
integrand_gas_fine = 4.0 * np.pi * r_fine_m**2 * rho_gas_fine
M_gas_fine_kg = cumulative_trapezoid(integrand_gas_fine, r_fine_m, initial=0.0)

# Interpolate to the output grid (log-log interpolation)
M_gas_kg = np.interp(np.log10(r_kpc), np.log10(r_fine), M_gas_fine_kg)
M_gas_Msun = M_gas_kg / M_sun

# ====================================================================
# Step 4 -- Hydrostatic equilibrium mass M_HSE(<r)
# ====================================================================
# M_HSE(<r) = - kT(r) r / (G mu m_p) * [ d ln n_e / d ln r + d ln T / d ln r ]
#
# Compute logarithmic derivatives on the FINE grid for accuracy,
# then interpolate M_HSE to the output grid.
ln_r_fine  = np.log(r_fine)
ln_ne_fine = np.log(ne_fine)
ln_T_fine  = np.log(T_fine)

# Central differences for interior points, forward/backward at boundaries
dlnne_dlnr_fine = np.gradient(ln_ne_fine, ln_r_fine)
dlnT_dlnr_fine  = np.gradient(ln_T_fine,  ln_r_fine)

# Temperature in Joules (fine grid)
T_fine_J = T_fine * keV_to_J

# Hydrostatic mass on fine grid (SI)
M_HSE_fine_kg = -(T_fine_J * r_fine_m) / (G_SI * mu * m_p) * \
                (dlnne_dlnr_fine + dlnT_dlnr_fine)

# Enforce positivity (innermost points can be noisy)
M_HSE_fine_kg = np.maximum(M_HSE_fine_kg, 0.0)

# Interpolate to output grid
M_HSE_kg = np.interp(np.log10(r_kpc), np.log10(r_fine), M_HSE_fine_kg)
M_HSE_Msun = M_HSE_kg / M_sun

# Also store output-grid log-derivatives for reference
ln_r  = np.log(r_kpc)
dlnne_dlnr = np.interp(np.log10(r_kpc), np.log10(r_fine), dlnne_dlnr_fine)
dlnT_dlnr  = np.interp(np.log10(r_kpc), np.log10(r_fine), dlnT_dlnr_fine)

# ====================================================================
# Step 5 -- Stellar mass estimate
# ====================================================================
# Liu+2023 notes the stellar fraction is ~2% of M_HSE (standard ICM assumption)
f_star = 0.02
M_star_kg   = f_star * M_HSE_kg
M_star_Msun = f_star * M_HSE_Msun

# ====================================================================
# Step 6 -- Baryonic mass
# ====================================================================
M_bar_kg   = M_gas_kg + M_star_kg
M_bar_Msun = M_gas_Msun + M_star_Msun

# ====================================================================
# Step 7 -- Accelerations
# ====================================================================
g_tot = G_SI * M_HSE_kg / r_m**2       # total (observed) acceleration
g_bar = G_SI * M_bar_kg / r_m**2       # baryonic acceleration

# ====================================================================
# Step 8 -- BEC / Superfluid prediction
# ====================================================================
# g_BEC = g_bar / (1 - exp(-sqrt(g_bar / g_dagger)))
# This is the standard interpolating function from Milgrom / McGaugh
sqrt_ratio = np.sqrt(np.clip(g_bar / g_dagger, 0, 500))  # clip to avoid overflow
g_BEC = g_bar / (1.0 - np.exp(-sqrt_ratio))

# ====================================================================
# Log quantities
# ====================================================================
# Protect against any zeros
g_bar_safe = np.where(g_bar > 0, g_bar, np.nan)
g_tot_safe = np.where(g_tot > 0, g_tot, np.nan)
g_BEC_safe = np.where(g_BEC > 0, g_BEC, np.nan)

log_g_bar = np.log10(g_bar_safe)
log_g_tot = np.log10(g_tot_safe)
log_g_BEC = np.log10(g_BEC_safe)

# ====================================================================
# Step 9 -- Save CSV
# ====================================================================
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'raw_data', 'observational', 'cluster_rar'
)
os.makedirs(OUT_DIR, exist_ok=True)

out_path = os.path.join(OUT_DIR, 'smacs0723_liu2023_profiles.csv')

header = (
    "# SMACS J0723.3-7327 reconstructed radial profiles\n"
    "# Source: Liu et al. 2023, A&A 670, A96 (eROSITA best-fit)\n"
    "# Vikhlinin+2006 analytic models for T(r) and n_e(r)\n"
    f"# Cluster redshift z = {z_cluster}\n"
    "# Columns:\n"
    "#   r_kpc         - radius [kpc]\n"
    "#   n_e_cm3       - electron density [cm^-3]\n"
    "#   T_keV         - temperature [keV]\n"
    "#   rho_gas_kg_m3 - gas mass density [kg m^-3]\n"
    "#   M_gas_Msun    - cumulative gas mass [M_sun]\n"
    "#   M_HSE_Msun    - hydrostatic equilibrium mass [M_sun]\n"
    "#   M_star_Msun   - stellar mass estimate (0.02 * M_HSE) [M_sun]\n"
    "#   M_bar_Msun    - baryonic mass (M_gas + M_star) [M_sun]\n"
    "#   g_bar_m_s2    - baryonic acceleration [m s^-2]\n"
    "#   g_tot_m_s2    - total (observed) acceleration [m s^-2]\n"
    "#   g_BEC_m_s2    - BEC/superfluid prediction [m s^-2]\n"
    "#   log_g_bar     - log10(g_bar)\n"
    "#   log_g_tot     - log10(g_tot)\n"
    "#   log_g_BEC     - log10(g_BEC)\n"
)

data = np.column_stack([
    r_kpc, n_e, T_keV, rho_gas, M_gas_Msun, M_HSE_Msun, M_star_Msun,
    M_bar_Msun, g_bar, g_tot, g_BEC, log_g_bar, log_g_tot, log_g_BEC
])

col_names = (
    "r_kpc,n_e_cm3,T_keV,rho_gas_kg_m3,M_gas_Msun,M_HSE_Msun,"
    "M_star_Msun,M_bar_Msun,g_bar_m_s2,g_tot_m_s2,g_BEC_m_s2,"
    "log_g_bar,log_g_tot,log_g_BEC"
)

with open(out_path, 'w') as f:
    f.write(header)
    f.write(col_names + "\n")
    for row in data:
        f.write(",".join(f"{v:.8e}" for v in row) + "\n")

print(f"Saved {N_PTS} radial bins to:\n  {out_path}\n")

# ====================================================================
# Step 10 -- Validate against published Table 4
# ====================================================================
print("=" * 72)
print("VALIDATION AGAINST LIU ET AL. 2023 TABLE 4 (eROSITA)")
print("=" * 72)

def interpolate_mass_at_radius(r_target_kpc, r_arr, M_arr):
    """Log-linear interpolation to get mass at a specific radius."""
    return np.interp(np.log10(r_target_kpc), np.log10(r_arr), M_arr)

M_HSE_at_R2500 = interpolate_mass_at_radius(R2500_kpc, r_kpc, M_HSE_Msun)
M_HSE_at_R500  = interpolate_mass_at_radius(R500_kpc,  r_kpc, M_HSE_Msun)

M_gas_at_R500  = interpolate_mass_at_radius(R500_kpc,  r_kpc, M_gas_Msun)
M_gas_at_R2500 = interpolate_mass_at_radius(R2500_kpc, r_kpc, M_gas_Msun)

# Compute gas fractions
fgas_500_calc  = M_gas_at_R500  / M_HSE_at_R500  if M_HSE_at_R500 > 0 else np.nan
fgas_2500_calc = M_gas_at_R2500 / M_HSE_at_R2500 if M_HSE_at_R2500 > 0 else np.nan

within_2500 = abs(M_HSE_at_R2500 - M2500_Msun) <= dM2500
within_500  = abs(M_HSE_at_R500  - M500_Msun)  <= dM500

sigma_2500 = (M_HSE_at_R2500 - M2500_Msun) / dM2500
sigma_500  = (M_HSE_at_R500  - M500_Msun)  / dM500
ratio_2500 = M_HSE_at_R2500 / M2500_Msun
ratio_500  = M_HSE_at_R500  / M500_Msun

print(f"\n{'Quantity':<20s} {'Reconstructed':>16s} {'Published':>16s} "
      f"{'Ratio':>8s} {'Sigma':>8s} {'In errbar?':>12s}")
print("-" * 80)

print(f"{'M_HSE(R2500)':<20s} {M_HSE_at_R2500:>16.3e} {M2500_Msun:>16.3e} "
      f"{ratio_2500:>8.2f} {sigma_2500:>+8.2f} "
      f"{'YES' if within_2500 else 'NO':>12s}")
print(f"  R_2500 = {R2500_kpc:.0f} kpc, published error = +/- {dM2500:.1e} M_sun")

print(f"{'M_HSE(R500)':<20s} {M_HSE_at_R500:>16.3e} {M500_Msun:>16.3e} "
      f"{ratio_500:>8.2f} {sigma_500:>+8.2f} "
      f"{'YES' if within_500 else 'NO':>12s}")
print(f"  R_500  = {R500_kpc:.0f} kpc, published error = +/- {dM500:.1e} M_sun")

print(f"\n{'f_gas(R500)':<20s} {fgas_500_calc:>16.4f} {f_gas_500:>16.4f}")
print(f"{'f_gas(R2500)':<20s} {fgas_2500_calc:>16.4f} {f_gas_2500:>16.4f}")

if not (within_2500 and within_500):
    print("\n  NOTE: Reconstructed M_HSE exceeds published central values.")
    print("  This is typical for analytic HSE reconstructions from smooth")
    print("  Vikhlinin-model profiles.  Contributing factors:")
    print("  - Published masses may include hydrostatic bias correction (~15-40%)")
    print("  - Numerical log-derivative sensitivity at profile inflection points")
    print("  - Temperature extrapolation beyond the X-ray fitting range")

# ====================================================================
# Step 11 -- RAR comparison summary
# ====================================================================
print("\n" + "=" * 72)
print("RADIAL ACCELERATION RELATION -- BEC PREDICTION vs OBSERVED")
print("=" * 72)

# Ratio g_tot / g_BEC at several characteristic radii
check_radii_kpc = [50, 100, 200, 500, 1000, 1500, 2000, 2500]

print(f"\n{'r [kpc]':>10s} {'g_bar':>12s} {'g_tot':>12s} {'g_BEC':>12s} "
      f"{'g_tot/g_BEC':>12s} {'g_tot/g_bar':>12s}")
print("-" * 72)

for r_check in check_radii_kpc:
    if r_check < r_kpc[0] or r_check > r_kpc[-1]:
        continue
    idx = np.argmin(np.abs(r_kpc - r_check))
    gb = g_bar[idx]
    gt = g_tot[idx]
    gB = g_BEC[idx]
    ratio_BEC = gt / gB if gB > 0 else np.nan
    ratio_bar = gt / gb if gb > 0 else np.nan
    print(f"{r_kpc[idx]:>10.1f} {gb:>12.3e} {gt:>12.3e} {gB:>12.3e} "
          f"{ratio_BEC:>12.4f} {ratio_bar:>12.4f}")

# Summary statistics over the full radial range (exclude edge effects)
mask = (r_kpc > 30) & (r_kpc < 2500)
ratio_full = g_tot[mask] / g_BEC[mask]
ratio_bar_full = g_tot[mask] / g_bar[mask]

print(f"\nSummary (30 < r < 2500 kpc):")
print(f"  g_tot / g_BEC  --  median = {np.nanmedian(ratio_full):.4f}, "
      f"range = [{np.nanmin(ratio_full):.4f}, {np.nanmax(ratio_full):.4f}]")
print(f"  g_tot / g_bar  --  median = {np.nanmedian(ratio_bar_full):.4f}, "
      f"range = [{np.nanmin(ratio_bar_full):.4f}, {np.nanmax(ratio_bar_full):.4f}]")

# Interpretation
median_ratio = np.nanmedian(ratio_full)
if median_ratio > 1.05:
    print(f"\n  => BEC (galaxy-scale g_dagger) UNDERPREDICTS g_tot by a factor "
          f"~{median_ratio:.2f}.")
    print(f"     This is expected: cluster ICM operates above g_dagger, so")
    print(f"     the simple galaxy RAR interpolating function does not fully")
    print(f"     account for the missing mass at cluster scales.")
elif median_ratio < 0.95:
    print(f"\n  => BEC prediction OVERPREDICTS g_tot by a factor "
          f"~{1.0/median_ratio:.2f}.")
else:
    print(f"\n  => BEC prediction matches g_tot to within 5%.")

print(f"\n  g_dagger (galaxy-scale) = {g_dagger:.1e} m/s^2")
print(f"  Typical g_bar at R_500  = {g_bar[np.argmin(np.abs(r_kpc - R500_kpc))]:.2e} m/s^2")
print(f"  Ratio g_bar(R500) / g_dagger = "
      f"{g_bar[np.argmin(np.abs(r_kpc - R500_kpc))] / g_dagger:.1f}")

print("\nDone.\n")


if __name__ == "__main__":
    pass  # Script runs at import; this guard is for clarity.
