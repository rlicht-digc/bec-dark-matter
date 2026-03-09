#!/usr/bin/env python3
"""
Reconstruct radial lensing mass profiles for the 20 CLASH clusters
from NFW parameters published in Umetsu et al. 2016 (ApJ 821, 116).

Computes NFW enclosed mass, baryonic mass (gas + stars), gravitational
accelerations g_bar and g_tot, and the BEC/MOND interpolation g_BEC.
Cross-validates against Tian et al. 2020 (ApJ 896, 70) radial bins.

Usage:
    python reconstruct_clash_nfw.py
"""

import os
import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
Msun_kg = 1.98848e30        # kg
kpc_m = 3.08568e19          # metres per kpc
Mpc_m = 3.08568e22          # metres per Mpc
H0_SI = 70.0e3 / Mpc_m     # H0 = 70 km/s/Mpc in s^-1

# Cosmology
Omega_m = 0.3
Omega_L = 0.7
g_dagger = 1.2e-10          # m/s^2  (RAR scale)

# Baryon fractions
f_b_cosmic = 0.157          # Omega_b / Omega_m ~ 0.157
f_gas_500 = 0.13            # gas fraction at R500
f_star_500 = 0.02           # stellar fraction at R500

# ---------------------------------------------------------------------------
# Umetsu+2016 Table 2 — NFW parameters for 20 CLASH clusters
# M200c in units of 1e14 Msun, c200c dimensionless
# ---------------------------------------------------------------------------
CLASH_NFW = [
    ("A209",      0.206, 14.8, 4.3),
    ("A383",      0.187, 11.3, 6.4),
    ("A611",      0.288, 14.3, 3.0),
    ("A2261",     0.224, 21.1, 4.2),
    ("RXJ2129",   0.235,  7.5, 8.5),
    ("A963",      0.206,  9.5, 4.4),
    ("RXJ2248",   0.348, 24.5, 4.6),
    ("MACS1931",  0.352,  9.2, 4.2),
    ("MACS1115",  0.352, 15.2, 5.0),
    ("MACS1720",  0.391, 10.5, 5.7),
    ("MACS0416",  0.396, 12.4, 3.6),
    ("MACS1206",  0.440, 15.1, 5.3),
    ("MACS0329",  0.450, 12.0, 4.9),
    ("RXJ1347",   0.451, 33.0, 3.5),
    ("MACS1311",  0.494,  6.3, 6.7),
    ("MACS1149",  0.544, 21.5, 3.5),
    ("MACS0717",  0.548, 27.4, 3.3),
    ("MACS0647",  0.584, 14.5, 4.1),
    ("MACS0744",  0.686, 12.2, 5.0),
    ("CLJ1226",   0.890, 13.2, 4.0),
]

# ---------------------------------------------------------------------------
# Tian+2020 fig2.dat path (for cross-validation)
# ---------------------------------------------------------------------------
TIAN_FIG2_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "raw_data", "observational", "cluster_rar", "tian2020_fig2.dat"
)
TIAN_TABLE1_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "raw_data", "observational", "cluster_rar", "tian2020_table1.dat"
)

# Output paths
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "raw_data", "observational", "cluster_rar"
)
PROFILE_CSV = os.path.join(OUTPUT_DIR, "clash_nfw_profiles.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "clash_nfw_summary.csv")


# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------
def H_z(z):
    """Hubble parameter H(z) in s^-1."""
    return H0_SI * np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)


def rho_crit_z(z):
    """Critical density at redshift z in kg/m^3."""
    Hz = H_z(z)
    return 3.0 * Hz**2 / (8.0 * np.pi * G_SI)


# ---------------------------------------------------------------------------
# NFW profile functions
# ---------------------------------------------------------------------------
def nfw_r200(M200c, z):
    """R200c in kpc given M200c in Msun and redshift z."""
    M200_kg = M200c * Msun_kg
    rho_c = rho_crit_z(z)
    r200_m = (3.0 * M200_kg / (4.0 * np.pi * 200.0 * rho_c))**(1.0 / 3.0)
    return r200_m / kpc_m


def nfw_rho_s(M200c, c200c, r_s_kpc):
    """Characteristic NFW density rho_s in kg/m^3."""
    M200_kg = M200c * Msun_kg
    r_s_m = r_s_kpc * kpc_m
    g_c = np.log(1.0 + c200c) - c200c / (1.0 + c200c)
    rho_s = M200_kg / (4.0 * np.pi * r_s_m**3 * g_c)
    return rho_s


def nfw_enclosed_mass(r_kpc, rho_s, r_s_kpc):
    """NFW enclosed mass M(<r) in Msun.

    M(<r) = 4 pi rho_s r_s^3 [ln(1 + x) - x/(1+x)]  where x = r/r_s.
    """
    r_s_m = r_s_kpc * kpc_m
    x = r_kpc / r_s_kpc
    mass_kg = 4.0 * np.pi * rho_s * r_s_m**3 * (np.log(1.0 + x) - x / (1.0 + x))
    return mass_kg / Msun_kg


def nfw_density(r_kpc, rho_s, r_s_kpc):
    """NFW density at radius r in kg/m^3."""
    x = r_kpc / r_s_kpc
    return rho_s / (x * (1.0 + x)**2)


# ---------------------------------------------------------------------------
# R500 solver — find r where mean enclosed density = 500 * rho_crit(z)
# ---------------------------------------------------------------------------
def find_r500(rho_s, r_s_kpc, z):
    """Find R500 in kpc by bisection."""
    rho_c = rho_crit_z(z)
    target_density = 500.0 * rho_c  # kg/m^3

    def mean_density(r_kpc):
        """Mean enclosed density in kg/m^3."""
        M_kg = nfw_enclosed_mass(r_kpc, rho_s, r_s_kpc) * Msun_kg
        V_m3 = (4.0 / 3.0) * np.pi * (r_kpc * kpc_m)**3
        return M_kg / V_m3

    # Bisection between 10 kpc and 5000 kpc
    r_lo, r_hi = 10.0, 5000.0
    for _ in range(100):
        r_mid = 0.5 * (r_lo + r_hi)
        if mean_density(r_mid) > target_density:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return 0.5 * (r_lo + r_hi)


# ---------------------------------------------------------------------------
# Baryon fraction profiles
# ---------------------------------------------------------------------------
def f_gas_profile(r_kpc, r500_kpc):
    """Gas fraction as a function of radius, increasing outward."""
    f = f_gas_500 * (r_kpc / r500_kpc)**0.2
    return np.minimum(f, f_b_cosmic)


def f_star_profile(r_kpc, r500_kpc):
    """Stellar fraction, more concentrated toward centre."""
    f = f_star_500 * (r500_kpc / r_kpc)**0.5
    # Cap at a reasonable maximum
    return np.minimum(f, 0.10)


def baryonic_mass(r_kpc, M_nfw, r500_kpc):
    """Baryonic enclosed mass M_bar at each radius."""
    fg = f_gas_profile(r_kpc, r500_kpc)
    fs = f_star_profile(r_kpc, r500_kpc)
    return (fg + fs) * M_nfw


# ---------------------------------------------------------------------------
# Accelerations
# ---------------------------------------------------------------------------
def gravitational_acceleration(M_msun, r_kpc):
    """g = G M / r^2  in m/s^2."""
    M_kg = M_msun * Msun_kg
    r_m = r_kpc * kpc_m
    return G_SI * M_kg / r_m**2


def g_bec_interpolation(g_bar):
    """BEC/MOND interpolation function.

    g_BEC = g_bar / (1 - exp(-sqrt(g_bar / g_dagger)))
    """
    ratio = np.sqrt(np.abs(g_bar) / g_dagger)
    # Avoid division by zero for very large g_bar (exp term -> 0)
    denom = 1.0 - np.exp(-ratio)
    # For safety with very small g_bar
    denom = np.maximum(denom, 1e-30)
    return g_bar / denom


# ---------------------------------------------------------------------------
# Parse Tian+2020 fig2.dat
# ---------------------------------------------------------------------------
def parse_tian_fig2():
    """Parse Tian+2020 fig2.dat, return list of dicts.

    Each dict: {name, rad_arcsec, log_gbar, log_gtot, e_log_gbar, e_log_gtot}
    """
    path = os.path.abspath(TIAN_FIG2_PATH)
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Skip header lines
            if "AName" in line or "Rad" in line or "log" in line or "e_" in line:
                continue
            # Parse pipe-delimited: AName|Rad|log(gbar)|log(gtot)|e_log(gbar)|e_
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6:
                continue
            try:
                name = parts[0].strip()
                rad = float(parts[1])
                log_gbar = float(parts[2])
                log_gtot = float(parts[3])
                e_log_gbar = float(parts[4])
                e_log_gtot = float(parts[5])
                records.append({
                    "name": name,
                    "rad_arcsec": rad,
                    "log_gbar": log_gbar,
                    "log_gtot": log_gtot,
                    "e_log_gbar": e_log_gbar,
                    "e_log_gtot": e_log_gtot,
                })
            except ValueError:
                continue
    return records


# ---------------------------------------------------------------------------
# Parse Tian+2020 table1.dat for redshifts (to convert arcsec -> kpc)
# ---------------------------------------------------------------------------
def parse_tian_table1():
    """Parse Tian+2020 table1.dat, return dict of {AName: z}."""
    path = os.path.abspath(TIAN_TABLE1_PATH)
    cluster_z = {}
    with open(path, "r") as f:
        for line in f:
            line_s = line.strip()
            if not line_s or line_s.startswith("#") or line_s.startswith("-"):
                continue
            if "Name" in line_s and "z" in line_s:
                continue
            # Header continuation lines
            if line_s.startswith("|") or "Band" in line_s or "Re" in line_s:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            try:
                # Last column is AName (short name)
                aname = parts[-1].strip()
                z = float(parts[1])
                if aname and z > 0:
                    cluster_z[aname] = z
            except (ValueError, IndexError):
                continue
    return cluster_z


# ---------------------------------------------------------------------------
# Angular diameter distance (flat LCDM)
# ---------------------------------------------------------------------------
def angular_diameter_distance_kpc(z):
    """Angular diameter distance in kpc for flat LCDM."""
    from scipy.integrate import quad

    def integrand(zp):
        return 1.0 / np.sqrt(Omega_m * (1.0 + zp)**3 + Omega_L)

    result, _ = quad(integrand, 0, z)
    # Comoving distance in Mpc
    d_c_Mpc = (2.998e5 / 70.0) * result  # c/H0 * integral, c in km/s, H0 in km/s/Mpc
    d_a_Mpc = d_c_Mpc / (1.0 + z)
    return d_a_Mpc * 1000.0  # kpc


# ---------------------------------------------------------------------------
# Main reconstruction
# ---------------------------------------------------------------------------
def reconstruct_all():
    """Reconstruct NFW profiles for all 20 CLASH clusters."""

    # Radial grid: 100 log-spaced points from 10 to 3000 kpc
    r_kpc = np.logspace(np.log10(10.0), np.log10(3000.0), 100)

    # Storage for profiles and summary
    profile_rows = []
    summary_rows = []

    print("=" * 80)
    print("CLASH NFW Profile Reconstruction — Umetsu et al. 2016")
    print("=" * 80)
    print(f"{'Cluster':<12} {'z':>5} {'M200c':>8} {'c200c':>5} "
          f"{'r200c':>8} {'r500c':>8} {'M500c':>10} {'deficit':>8}")
    print(f"{'':12} {'':>5} {'[1e14]':>8} {'':>5} "
          f"{'[kpc]':>8} {'[kpc]':>8} {'[1e14]':>10} {'@R500':>8}")
    print("-" * 80)

    for name, z, M200c_14, c200c in CLASH_NFW:
        M200c = M200c_14 * 1e14  # Msun

        # NFW scale parameters
        r200_kpc = nfw_r200(M200c, z)
        r_s_kpc = r200_kpc / c200c
        rho_s = nfw_rho_s(M200c, c200c, r_s_kpc)

        # R500
        r500_kpc = find_r500(rho_s, r_s_kpc, z)
        M500 = nfw_enclosed_mass(r500_kpc, rho_s, r_s_kpc)

        # Profiles at each radius
        M_nfw = nfw_enclosed_mass(r_kpc, rho_s, r_s_kpc)
        M_bar = baryonic_mass(r_kpc, M_nfw, r500_kpc)

        g_tot = gravitational_acceleration(M_nfw, r_kpc)
        g_bar = gravitational_acceleration(M_bar, r_kpc)
        g_bec = g_bec_interpolation(g_bar)

        log_g_bar = np.log10(g_bar)
        log_g_tot = np.log10(g_tot)
        log_g_bec = np.log10(g_bec)

        r_over_r500 = r_kpc / r500_kpc
        r_over_r200 = r_kpc / r200_kpc

        # Deficit at R500: ratio of NFW total to BEC prediction
        g_tot_r500 = gravitational_acceleration(
            nfw_enclosed_mass(r500_kpc, rho_s, r_s_kpc), r500_kpc
        )
        M_bar_r500 = baryonic_mass(
            np.array([r500_kpc]),
            np.array([nfw_enclosed_mass(r500_kpc, rho_s, r_s_kpc)]),
            r500_kpc
        )[0]
        g_bar_r500 = gravitational_acceleration(M_bar_r500, r500_kpc)
        g_bec_r500 = g_bec_interpolation(np.array([g_bar_r500]))[0]
        deficit_r500 = g_tot_r500 / g_bec_r500

        # Store profiles
        for i in range(len(r_kpc)):
            profile_rows.append({
                "cluster": name,
                "z": z,
                "r_kpc": r_kpc[i],
                "M_NFW_Msun": M_nfw[i],
                "M_bar_Msun": M_bar[i],
                "g_bar_m_s2": g_bar[i],
                "g_tot_m_s2": g_tot[i],
                "g_BEC_m_s2": g_bec[i],
                "log_g_bar": log_g_bar[i],
                "log_g_tot": log_g_tot[i],
                "log_g_BEC": log_g_bec[i],
                "r_over_r500": r_over_r500[i],
                "r_over_r200": r_over_r200[i],
            })

        # Summary row
        summary_rows.append({
            "cluster": name,
            "z": z,
            "M_200c": M200c,
            "c_200c": c200c,
            "r_200c_kpc": r200_kpc,
            "r_500_kpc": r500_kpc,
            "M_500_Msun": M500,
            "deficit_at_r500": deficit_r500,
        })

        print(f"{name:<12} {z:5.3f} {M200c_14:8.1f} {c200c:5.1f} "
              f"{r200_kpc:8.1f} {r500_kpc:8.1f} {M500/1e14:10.2f} {deficit_r500:8.3f}")

    print("-" * 80)

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    deficits_r500 = [s["deficit_at_r500"] for s in summary_rows]
    print(f"\nMedian g_tot/g_BEC deficit at R500: {np.median(deficits_r500):.3f}")
    print(f"  Mean: {np.mean(deficits_r500):.3f},  "
          f"Min: {np.min(deficits_r500):.3f},  Max: {np.max(deficits_r500):.3f}")

    # Deficit at R200
    deficits_r200 = []
    for name, z, M200c_14, c200c in CLASH_NFW:
        M200c = M200c_14 * 1e14
        r200_kpc = nfw_r200(M200c, z)
        r_s_kpc = r200_kpc / c200c
        rho_s = nfw_rho_s(M200c, c200c, r_s_kpc)
        r500_kpc = find_r500(rho_s, r_s_kpc, z)

        g_tot_r200 = gravitational_acceleration(M200c, r200_kpc)
        M_bar_r200 = baryonic_mass(
            np.array([r200_kpc]),
            np.array([M200c]),
            r500_kpc
        )[0]
        g_bar_r200 = gravitational_acceleration(M_bar_r200, r200_kpc)
        g_bec_r200 = g_bec_interpolation(np.array([g_bar_r200]))[0]
        deficits_r200.append(g_tot_r200 / g_bec_r200)

    print(f"\nMedian g_tot/g_BEC deficit at R200: {np.median(deficits_r200):.3f}")
    print(f"  Mean: {np.mean(deficits_r200):.3f},  "
          f"Min: {np.min(deficits_r200):.3f},  Max: {np.max(deficits_r200):.3f}")

    # -----------------------------------------------------------------------
    # Cross-validation against Tian+2020
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Cross-validation against Tian et al. 2020 (ApJ 896, 70)")
    print("=" * 80)

    tian_records = parse_tian_fig2()
    tian_redshifts = parse_tian_table1()

    if not tian_records:
        print("WARNING: Could not parse Tian+2020 fig2.dat — skipping cross-validation.")
    else:
        # Build lookup: for each CLASH cluster, get its NFW profile interpolator
        nfw_interp = {}
        for name, z, M200c_14, c200c in CLASH_NFW:
            M200c = M200c_14 * 1e14
            r200_kpc = nfw_r200(M200c, z)
            r_s_kpc = r200_kpc / c200c
            rho_s = nfw_rho_s(M200c, c200c, r_s_kpc)
            r500_kpc = find_r500(rho_s, r_s_kpc, z)

            M_nfw_arr = nfw_enclosed_mass(r_kpc, rho_s, r_s_kpc)
            M_bar_arr = baryonic_mass(r_kpc, M_nfw_arr, r500_kpc)
            g_tot_arr = gravitational_acceleration(M_nfw_arr, r_kpc)
            g_bar_arr = gravitational_acceleration(M_bar_arr, r_kpc)

            nfw_interp[name] = {
                "log_gtot": interp1d(np.log10(r_kpc), np.log10(g_tot_arr),
                                     kind="cubic", fill_value="extrapolate"),
                "log_gbar": interp1d(np.log10(r_kpc), np.log10(g_bar_arr),
                                     kind="cubic", fill_value="extrapolate"),
                "z": z,
            }

        # Compare at each Tian radial bin
        delta_log_gtot = []
        delta_log_gbar = []
        n_matched = 0

        print(f"\n{'Cluster':<12} {'R_arcsec':>8} {'R_kpc':>8} "
              f"{'Tian_lg':>9} {'NFW_lg':>9} {'d_lg_gt':>8} "
              f"{'Tian_lgb':>9} {'NFW_lgb':>9} {'d_lg_gb':>8}")
        print("-" * 95)

        for rec in tian_records:
            tname = rec["name"]
            # Match Tian cluster name to CLASH NFW name
            if tname not in nfw_interp:
                continue

            z_cl = nfw_interp[tname]["z"]
            # Convert arcsec to kpc using angular diameter distance
            d_a_kpc = angular_diameter_distance_kpc(z_cl)
            r_phys_kpc = rec["rad_arcsec"] * (d_a_kpc / 206265.0)  # arcsec -> rad -> kpc

            if r_phys_kpc < 5.0 or r_phys_kpc > 5000.0:
                continue

            log_r = np.log10(r_phys_kpc)
            nfw_log_gtot = float(nfw_interp[tname]["log_gtot"](log_r))
            nfw_log_gbar = float(nfw_interp[tname]["log_gbar"](log_r))

            d_gt = rec["log_gtot"] - nfw_log_gtot
            d_gb = rec["log_gbar"] - nfw_log_gbar

            delta_log_gtot.append(d_gt)
            delta_log_gbar.append(d_gb)
            n_matched += 1

            print(f"{tname:<12} {rec['rad_arcsec']:8.1f} {r_phys_kpc:8.1f} "
                  f"{rec['log_gtot']:9.3f} {nfw_log_gtot:9.3f} {d_gt:8.3f} "
                  f"{rec['log_gbar']:9.3f} {nfw_log_gbar:9.3f} {d_gb:8.3f}")

        if n_matched > 0:
            print("-" * 95)
            print(f"\nMatched {n_matched} Tian radial bins across "
                  f"{len(set(r['name'] for r in tian_records if r['name'] in nfw_interp))} clusters")
            print(f"Median offset in log(g_tot): {np.median(delta_log_gtot):+.3f} dex")
            print(f"  (mean {np.mean(delta_log_gtot):+.3f}, "
                  f"std {np.std(delta_log_gtot):.3f})")
            print(f"Median offset in log(g_bar): {np.median(delta_log_gbar):+.3f} dex")
            print(f"  (mean {np.mean(delta_log_gbar):+.3f}, "
                  f"std {np.std(delta_log_gbar):.3f})")
        else:
            print("No matching clusters found between CLASH NFW and Tian+2020.")

    # -----------------------------------------------------------------------
    # Save profiles CSV
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(PROFILE_CSV)), exist_ok=True)

    with open(os.path.abspath(PROFILE_CSV), "w") as f:
        header = ("cluster,z,r_kpc,M_NFW_Msun,M_bar_Msun,"
                  "g_bar_m_s2,g_tot_m_s2,g_BEC_m_s2,"
                  "log_g_bar,log_g_tot,log_g_BEC,r_over_r500,r_over_r200")
        f.write(header + "\n")
        for row in profile_rows:
            f.write(f"{row['cluster']},{row['z']:.3f},{row['r_kpc']:.4f},"
                    f"{row['M_NFW_Msun']:.6e},{row['M_bar_Msun']:.6e},"
                    f"{row['g_bar_m_s2']:.6e},{row['g_tot_m_s2']:.6e},"
                    f"{row['g_BEC_m_s2']:.6e},"
                    f"{row['log_g_bar']:.4f},{row['log_g_tot']:.4f},"
                    f"{row['log_g_BEC']:.4f},"
                    f"{row['r_over_r500']:.4f},{row['r_over_r200']:.4f}\n")

    print(f"\nProfiles saved to: {os.path.abspath(PROFILE_CSV)}")
    print(f"  {len(profile_rows)} rows ({len(CLASH_NFW)} clusters x {len(r_kpc)} radii)")

    # -----------------------------------------------------------------------
    # Save summary CSV
    # -----------------------------------------------------------------------
    with open(os.path.abspath(SUMMARY_CSV), "w") as f:
        f.write("cluster,z,M_200c,c_200c,r_200c_kpc,r_500_kpc,"
                "M_500_Msun,deficit_at_r500\n")
        for row in summary_rows:
            f.write(f"{row['cluster']},{row['z']:.3f},"
                    f"{row['M_200c']:.6e},{row['c_200c']:.1f},"
                    f"{row['r_200c_kpc']:.2f},{row['r_500_kpc']:.2f},"
                    f"{row['M_500_Msun']:.6e},{row['deficit_at_r500']:.4f}\n")

    print(f"Summary saved to: {os.path.abspath(SUMMARY_CSV)}")

    return profile_rows, summary_rows


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    reconstruct_all()
